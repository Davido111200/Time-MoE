#!/usr/bin/env python
# -*- coding:utf-8 _*-
import time
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
import random
import math

from transformers import AutoModelForCausalLM

from data_provider.data_factory import data_provider
from time_moe.datasets.benchmark_dataset import (
    BenchmarkEvalDataset,
    BenchmarkEvalDatasetTrain,
    GeneralEvalDataset,
)
from util import (
    obtain_icv_interpolate,
    remove_icv_layers_timer,
    add_icv_layers_timer,
    retrieve_examples_new,
    add_icv_layers_interpolate
)

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

logging.basicConfig(level=logging.INFO)


def setup_nccl(rank, world_size, master_addr="127.0.0.1", master_port=9899):
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


def count_num_tensor_elements(tensor: torch.Tensor) -> int:
    return int(tensor.numel())


def _model_device_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype


def _time_dim_for_series(x: torch.Tensor, seq_len: int | None) -> int:
    """
    Heuristic for common time-series layouts:
      - [B, T] -> time dim=1
      - [B, T, C] with T==seq_len -> time dim=1
      - [B, C, T] -> time dim=2
    """
    if x.ndim == 2:
        return 1
    if x.ndim == 3:
        if seq_len is not None and x.shape[1] == seq_len:
            return 1
        return 2
    return max(0, x.ndim - 1)


def _align_preds_to_inputs_shape(inputs: torch.Tensor, preds: torch.Tensor, time_dim: int) -> torch.Tensor:
    """
    Make preds have same ndim as inputs, inserting singleton feature/channel dim if needed.
    """
    if inputs.ndim == preds.ndim:
        return preds

    if inputs.ndim == 3 and preds.ndim == 2:
        # inputs either [B, T, C] or [B, C, T]
        if time_dim == 1:
            return preds.unsqueeze(-1)  # [B, P, 1]
        else:
            return preds.unsqueeze(1)   # [B, 1, P]

    return preds


def _concat_time(inputs: torch.Tensor, preds: torch.Tensor, seq_len: int | None) -> torch.Tensor:
    """
    Concatenate along inferred time dimension.
    """
    time_dim = _time_dim_for_series(inputs, seq_len)
    preds = _align_preds_to_inputs_shape(inputs, preds, time_dim)
    return torch.cat([inputs, preds], dim=time_dim)



def predict(model, batch: dict, pred_len: int):
    """
    Returns:
      preds: only the forecast part (last pred_len steps, same behavior as your working file)
      labels: labels on the model device
      inputs_dev: inputs on the model device/dtype
    """
    device, dtype = _model_device_dtype(model)

    inputs_dev = batch["inputs"].to(device=device, dtype=dtype)
    outputs = model.generate(inputs=inputs_dev, max_new_tokens=pred_len)

    # Keep only the newly generated part.
    # Covers [B, T_total], [B, C, T_total], [B, T_total, C]
    if outputs.ndim == 2:
        preds = outputs[:, -pred_len:]
    elif outputs.ndim == 3:
        time_dim = _time_dim_for_series(inputs_dev, seq_len=None)
        sl = [slice(None)] * 3
        sl[time_dim] = slice(outputs.shape[time_dim] - pred_len, outputs.shape[time_dim])
        preds = outputs[tuple(sl)]
    else:
        preds = outputs[..., -pred_len:]


    labels = batch["labels"].to(device=device)
    if preds.ndim > labels.ndim:
        labels = labels[..., None]

    return preds, labels, inputs_dev



def get_rep(model, series_1d: torch.Tensor):
    """
    Representation for retrieval. Returns [B, H] always.
    """
    device, dtype = _model_device_dtype(model)
    x = series_1d.to(device=device, dtype=dtype)
    out = model(x, output_hidden_states=True, use_cache=False, return_dict=True)
    h = out.hidden_states[-1]          # [B, T, H]
    rep = h.mean(dim=1).contiguous()   # [B, H]
    return rep


def get_rep_with_hidden_states(args, model, series_1d_input_only, series_1d: torch.Tensor):
    """
    Returns:
        rep: [B, H]      mean over time of the final layer (computed on input-only forward)
        hs:  [B, L*H]    concat of last-token states from all layers (computed on full sequence)
    """
    device, dtype = _model_device_dtype(model)
    x = series_1d.to(device=device, dtype=dtype)
    x_input = series_1d_input_only.to(device=device, dtype=dtype)

    if args.pred_len in [24, 36, 48, 60]:
        # padding for 24,36,48,60 pred len to avoid length mismatch
        pad_len = 96 - args.pred_len
        pad_tensor = torch.zeros((x.shape[0], pad_len), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad_tensor], dim=1)

    out = model(
        x,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )
    out_inp_only = model(
        x_input,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )

    hidden_list = list(out.hidden_states)
    h_final_input_only = list(out_inp_only.hidden_states)[-1]

    last_tokens = [h[:, -1, :] for h in hidden_list]          # L x [B, H]

    hs_layers = torch.stack(last_tokens, dim=1).contiguous()  # [B, L, H]
    hs = hs_layers.view(hs_layers.size(0), -1).contiguous()   # [B, L*H]

    rep = h_final_input_only.mean(dim=1).contiguous()         # [B, H]
    return rep, hs


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        raise NotImplementedError


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


def evaluate(args):
    batch_size = args.batch_size
    seq_len = args.seq_len
    pred_len = args.pred_len

    SCRATCH_DIR = "/scratch/s223540177/Time-Moe/timerxl/cache_data_finetuned_fixed_timerxl"

    world_size = int(os.getenv("WORLD_SIZE") or 1)
    rank = int(os.getenv("RANK") or 0)
    local_rank = int(os.getenv("LOCAL_RANK") or 0)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    metric_list = [
        MSEMetric(name="mse"),
        MAEMetric(name="mae"),
    ]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()

    # -------- caches --------
    gt = {i: [] for i in range(args.dec_in)}
    gt_path = os.path.join(SCRATCH_DIR, f"{args.data}/timerxl/gt_cache.pt")

    pred = {i: [] for i in range(args.dec_in)}
    pred_path = os.path.join(SCRATCH_DIR, f"{args.data}/timerxl/pred_cache.pt")

    # -------- pred cache --------
    if os.path.exists(pred_path):
        print("Loaded pred cache:", pred_path)
        pred = torch.load(pred_path)
    else:
        if not args.data_path.endswith(".csv"):
            raise ValueError("Only csv data is supported for training gathering.")

        dataset_train = BenchmarkEvalDatasetTrain(
            args,
            args.data_path,
            seq_len=seq_len,
            pred_len=pred_len,
            max_train_samples=args.pool_number,
        )

        sampler = DistributedSampler(dataset=dataset_train, shuffle=False) if dist.is_initialized() else None

        train_dl = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            drop_last=False,
        )

        num_each = max(1, len(dataset_train) // args.dec_in)

        pbar = tqdm(total=len(train_dl), desc="Gathering pred", ncols=100)
        with torch.no_grad():
            for idx, batch in enumerate(train_dl):
                preds, _labels, inputs_dev = predict(model, batch, pred_len=args.pred_len)

                ip_and_preds = _concat_time(inputs_dev, preds, seq_len=seq_len)
                r, hs = get_rep_with_hidden_states(args, model, inputs_dev, ip_and_preds)

                r_cpu = r.detach().cpu().to(torch.float16).contiguous()
                hs_cpu = hs.detach().cpu().to(torch.float16).contiguous()
                inp_cpu = batch["inputs"].detach().cpu().to(torch.float16).contiguous().clone()

                B = r_cpu.shape[0]
                base_idx = idx * B

                for b in range(B):
                    global_idx = base_idx + b
                    channel_id = global_idx // num_each
                    channel_id = min(channel_id, args.dec_in - 1)
                    pred[channel_id].append((global_idx, r_cpu[b], hs_cpu[b], inp_cpu[b].clone()))

                pbar.update(1)

        pbar.close()
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        torch.save(pred, pred_path)
        print(f"Saved pred cache to {pred_path}")

    # -------- gt cache --------
    if os.path.exists(gt_path):
        print("Loaded gt cache:", gt_path)
        gt = torch.load(gt_path)
    else:
        if not args.data_path.endswith(".csv"):
            raise ValueError("Only csv data is supported for training gathering.")

        dataset_train = BenchmarkEvalDatasetTrain(
            args,
            args.data_path,
            seq_len=seq_len,
            pred_len=pred_len,
            max_train_samples=args.pool_number,
        )

        train_dl_gt = dataset_train.data_loader
        num_each = max(1, len(dataset_train) // args.dec_in)

        pbar = tqdm(total=len(train_dl_gt), desc="Gathering gt", ncols=100)
        with torch.no_grad():
            for idx, (x, y, x_mark, y_mark) in enumerate(train_dl_gt):
                # Keep batch dimension.
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)

                # If channel dim is singleton, squeeze it (keep batch).
                if x.ndim == 3 and x.shape[1] == 1:
                    x = x.squeeze(1)
                if y.ndim == 3 and y.shape[1] == 1:
                    y = y.squeeze(1)

                dev, dt = _model_device_dtype(model)
                x_dev = x.to(device=dev, dtype=dt)
                y_dev = y.to(device=dev, dtype=dt)

                ip_and_gt = _concat_time(x_dev, y_dev, seq_len=seq_len)
                r, hs = get_rep_with_hidden_states(args, model, x_dev, ip_and_gt)

                r_cpu = r.detach().cpu().to(torch.float16).contiguous()
                hs_cpu = hs.detach().cpu().to(torch.float16).contiguous()
                x_cpu = x.detach().cpu().to(torch.float16).contiguous().clone()

                B = r_cpu.shape[0]
                base_idx = idx * B

                for b in range(B):
                    global_idx = base_idx + b
                    channel_id = global_idx // num_each
                    channel_id = min(channel_id, args.dec_in - 1)
                    gt[channel_id].append((global_idx, r_cpu[b], hs_cpu[b], x_cpu[b].clone()))

                pbar.update(1)

        pbar.close()
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        torch.save(gt, gt_path)
        print(f"Saved gt cache to {gt_path}")

    print("Caches ready.")

    # Optional pool truncation
    for i in range(args.dec_in):
        if args.pool_number >= len(pred[i]):
            continue
        idxs = random.sample(range(len(pred[i])), min(args.pool_number, len(pred[i])))
        pred[i] = [pred[i][j] for j in idxs]
        gt[i] = [gt[i][j] for j in idxs]

    # -------- eval dataset --------
    if args.data_path.endswith(".csv"):
        dataset = BenchmarkEvalDataset(args.data_path, seq_len=seq_len, pred_len=pred_len)
    else:
        dataset = GeneralEvalDataset(args.data_path, seq_len=seq_len, pred_len=pred_len)

    sampler = DistributedSampler(dataset=dataset, shuffle=False) if dist.is_initialized() else None
    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
        persistent_workers=False,
        pin_memory=False,
    )
    acc_count = 0
    nan_counter = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            channel_id = int(batch["channel_id"].cpu().numpy()[0])
            
            gt_correspond = gt[channel_id]
            pred_correspond = pred[channel_id]
            

            rep = get_rep(model, batch["inputs"])  # [B, H]
            rep = rep[0]

            H = int(rep.shape[-1])

            dists, samples = retrieve_examples_new(
                args,
                rep,
                gt_correspond,
                pool_number=args.pool_number,
                topk=args.num_closest_samples,
                query_series=batch["inputs"].cpu().numpy(),
            )

            gt_list = [gt_correspond[k] for k in samples]
            pred_list = [pred_correspond[k] for k in samples]

            # ICV: enforce correct orientation for add_icv_layers_timer
            icv = obtain_icv_interpolate(
                args,
                gt_list,
                pred_list,
                hidden_size=H,                 # use actual H
                beta=args.collapse_weight,
                whiten=False,
            )

            icv = icv[1:]  # keep in sync with your pipeline

            # IMPORTANT FIX:
            # Ensure icv is [L, H], not [H, L]. This is what prevents the 1024 vs 5 mismatch.
            # icv = _ensure_icv_layer_major(icv, hidden_size=H)
            dev, dt = _model_device_dtype(model)
            icv = icv.to(device=dev, dtype=dt)

            if torch.isnan(icv).any().item():
                # print out the exact row containing nan
                nan_row = torch.isnan(icv).any(dim=1).nonzero(as_tuple=True)[0]
                for r in nan_row:
                    print(f"NaN in icv at row {r}: {icv[r]}")
                print(f"NaN in icv at step {idx}, skipping...")
                quit()

            # For add_icv_layers_timer, pass icv directly (no torch.stack([icv], dim=1)).
            add_icv_layers_timer(
                model,
                icv,                       # [L,H]
                lam=[args.lam],
                beta=args.collapse_weight,
                max_frac=1.0,             # start small, 1.0 is usually destructive
            )

            preds, labels, _ = predict(model, batch, pred_len=args.pred_len)

            print("preds", preds)
            # if any nan in preds, print out 
            if torch.isnan(preds).any().item():
                print(f"NaN in preds at step {idx}, skipping...")
                quit()

            remove_icv_layers_timer(model)

            mse_step = torch.mean((preds - labels) ** 2).item()

            if math.isnan(mse_step):
                nan_counter += 1
                print(f"NaN encountered at step {idx}, total NaNs: {nan_counter}")
            else:
                for metric in metric_list:
                    metric.push(preds, labels)

                acc_count += count_num_tensor_elements(preds)

    ret_metric = {m.name: (m.value / acc_count) for m in metric_list}
    print(acc_count)
    print(f"{rank} - {ret_metric}")

    out_path = os.path.join(
        "/home/s223540177/Time-MoE/metric_results_timer_xl",
        f"{args.seq_len}_{args.pred_len}/{args.data}/final_metrics/",
    )
    os.makedirs(out_path, exist_ok=True)

    with open(
        os.path.join(
            out_path,
            f"weight{args.collapse_weight}_lamb{args.lam}_neighbor{args.num_closest_samples}.txt",
        ),
        "w",
    ) as f:
        for metric in metric_list:
            f.write(f"{metric.name}: {ret_metric[metric.name]}\n")
            # save the nan count as well
            f.write(f"nan_count: {nan_counter}\n")

    print(f"Saved final metrics to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TimeMoE Evaluate")
    parser.add_argument("--model", "-m", type=str, help="Model path")
    parser.add_argument("--data_path", "-d", type=str, help="Benchmark data path")
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size of evaluation")
    parser.add_argument("--seq_len", "-c", type=int, help="Context length")
    parser.add_argument("--pred_len", "-p", type=int, default=96, help="Prediction length")
    parser.add_argument("--label_len", type=int, default=0, help="Label length")

    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--task_name", type=str, required=False, default="long_term_forecast")
    parser.add_argument("--root_path", type=str, required=False, default="./data/ETT/")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--dec_in", type=int, required=True)
    parser.add_argument("--c_out", type=int, required=True)
    parser.add_argument("--num_closest_samples", type=int, default=16)
    parser.add_argument("--lam", type=float, default=0.5)

    parser.add_argument("--pool_number", type=int, default=10000)
    parser.add_argument("--retrieval", type=str, default="euclidean")
    parser.add_argument("--tail_n", default=None)
    parser.add_argument("--collapse_weight", type=float, default=0.0)

    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_method", type=str, default="avg")
    parser.add_argument("--down_sampling_layers", type=int, default=3)
    parser.add_argument(
        "--channel_independence",
        type=int,
        default=1,
        help="0: channel dependence 1: channel independence for FreTS model",
    )
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--decomp_method",
        type=str,
        default="moving_avg",
        help="method of series decompsition, only support moving_avg or dft_decomp",
    )
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--use_norm", type=int, default=1, help="whether to use normalize; True 1 False 0")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder", default=False)
    parser.add_argument("--factor", type=int, default=5, help="attn factor")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")

    args = parser.parse_args()

    if args.seq_len is None:
        if args.pred_len in [96, 192, 336, 720]:
            args.seq_len = 512
        else:
            args.seq_len = 96

    evaluate(args)
