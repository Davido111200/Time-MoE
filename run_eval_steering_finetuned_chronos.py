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
from chronos.chronos2 import Chronos2Pipeline
from chronos.chronos2.model import Chronos2Model

from typing import Tuple


from data_provider.data_factory import data_provider
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, BenchmarkEvalDatasetTrain, GeneralEvalDataset
from utils.forward_tracer import ForwardTracer, ForwardTrace
from util import obtain_icv_interpolate, add_icv_layers_interpolate, remove_icv_layers_chronos, retrieve_examples_new, add_icv_layers_interpolate_chronos
# put this at the very top of your entry script (before creating any DataLoader)

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

logging.basicConfig(level=logging.INFO)


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n

def get_rep(pipeline, series_1d: torch.Tensor) -> torch.Tensor:
    """
    Representation used by retrieve_examples.

    Args:
        pipeline: Chronos2Pipeline (e.g., `pipe`)
        series_1d:
            - [T]        single series
            - [B, T]     batch of univariate series
            - [B, V, T]  batch of multivariate series

    Returns:
        h: [H] if batch=1, else [B, H]
           mean-pooled encoder embedding over variates and patches.
    """

    x = series_1d

    # Chronos2Pipeline.embed expects [B, V, T]
    if x.ndim == 1:
        # [T] -> [1, 1, T]
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        # [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)
    elif x.ndim == 3:
        # [B, V, T] already
        pass
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D tensor, got shape {tuple(x.shape)}")

    x = x.to(dtype=torch.float32)

    with torch.no_grad():
        # embeddings_list: list of length B, each [V, P, H]
        embeddings_list, _ = pipeline.embed(x)

    # Stack into [B, V, P, H]
    embeddings = torch.stack(embeddings_list, dim=0)

    # Mean over variates and patches -> [B, H]
    h = embeddings.mean(dim=(1, 2)).contiguous()

    # Match old behavior: return [H] if B == 1
    if h.size(0) == 1:
        h = h.squeeze(0)

    return h

HIDDEN_SIZE = 768
NUM_ENCODER_LAYERS = 12

def _ensure_bt(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure x is [B, T] for Chronos2Model.encode (univariate context).
    Accepts [T] or [B, T]. If you ever pass [B, V, T], you'll need to adapt this.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)          # [T] -> [1, T]
    elif x.ndim == 2:
        pass                        # [B, T]
    else:
        raise ValueError(f"Expected [T] or [B,T] for Chronos context, got {tuple(x.shape)}")
    return x


def _chronos_capture_encoder_layers(
    pipeline,
    context: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Chronos-2 encoder once and capture:

      - hs_layers: [B, L, H]   last-token hidden from each encoder block
      - last_full: [B, S, H]   full sequence from the *last* encoder block

    We hook `Chronos2EncoderBlock` outputs (Chronos2EncoderBlockOutput.hidden_states).
    """

    # Unwrap underlying Chronos2Model from the pipeline if needed
    if hasattr(pipeline, "model") and isinstance(pipeline.model, Chronos2Model):
        model: Chronos2Model = pipeline.model
    elif isinstance(pipeline, Chronos2Model):
        model = pipeline
    else:
        raise ValueError(f"Expected Chronos2Pipeline or Chronos2Model, got {type(pipeline)}")

    encoder = model.encoder
    blocks = encoder.block
    L = len(blocks)

    # Storage for per-layer outputs
    layer_hidden = [None] * L       # each will be [B, S, H]

    def make_hook(idx: int):
        def hook(module, inputs, outputs):
            # outputs is Chronos2EncoderBlockOutput
            h = outputs.hidden_states  # [B, S, H]
            if h is None:
                return
            layer_hidden[idx] = h.detach()
        return hook

    # Register hooks on each encoder block
    handles = []
    for i, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(i)))

    # Prepare context
    ctx_bt = _ensure_bt(context)
    params = next(model.parameters())
    device, dtype = params.device, params.dtype
    ctx_bt = ctx_bt.to(device=device, dtype=dtype)

    # Run encode (no future covariates / targets, 1 output patch is enough)
    with torch.no_grad():
        _ = model.encode(
            context=ctx_bt,
            context_mask=None,
            group_ids=None,
            future_covariates=None,
            future_covariates_mask=None,
            num_output_patches=1,
            future_target=None,
            future_target_mask=None,
            output_attentions=False,
        )

    # Remove hooks
    for h in handles:
        h.remove()

    # Check we captured all layers
    if any(h is None for h in layer_hidden):
        missing = [i for i, v in enumerate(layer_hidden) if v is None]
        raise RuntimeError(f"Did not capture encoder activations for layers: {missing}")

    # Build [B, L, H] from last token of each layer
    last_tokens = []
    for h in layer_hidden:
        # h: [B, S, H]
        last_tokens.append(h[:, -1, :])        # [B, H]
    hs_layers = torch.stack(last_tokens, dim=1).contiguous()  # [B, L, H]

    # Full sequence from last layer
    last_full = layer_hidden[-1].contiguous()  # [B, S, H]

    return hs_layers, last_full

def get_rep_with_hidden_states(
    pipeline,
    series_1d_input_only: torch.Tensor,
    series_1d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chronos-2 analogue of the original get_rep_with_hidden_states.

    Args:
        pipeline: Chronos2Pipeline (or Chronos2Model).
        series_1d_input_only: [B, T_in]   teacher-forcing context only.
        series_1d:            [B, T_fr]   ip+pred (or FR) sequence.

    Returns:
        rep: [B, H]
            mean over time of the final encoder layer for the input-only sequence.

        hs:  [B, L*H]
            concat over layers of the last-token hidden state from each encoder block,
            for the full (FR) sequence.
    """

    # 1) FR path: use series_1d to get per-layer last-token states
    hs_layers_full, _ = _chronos_capture_encoder_layers(pipeline, series_1d)  # [B, L, H]
    B, L, H = hs_layers_full.shape

    # Optional sanity check vs known config
    # assert H == HIDDEN_SIZE, f"Expected H={HIDDEN_SIZE}, got {H}"
    # assert L == NUM_ENCODER_LAYERS, f"Expected L={NUM_ENCODER_LAYERS}, got {L}"

    hs_flat = hs_layers_full.view(B, L * H).contiguous()  # [B, L*H]

    # 2) TF path: use input-only series to get full last-layer sequence and pool in time
    _, last_full_input = _chronos_capture_encoder_layers(pipeline, series_1d_input_only)  # [B, S_in, H]
    rep = last_full_input.mean(dim=1).contiguous()  # [B, H]

    return rep, hs_flat



# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


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

    # min_len = seq_len + pred_len
    # if args.pool_number < min_len:
    #     print(f'Adjusting pool_number from {args.pool_number} to {min_len} to fit seq_len + pred_len')
    #     max_train_samples = min_len
    # else:
    #     max_train_samples = args.pool_number

    max_train_samples = args.pool_number

    output_log = f"/home/s223540177/Time-MoE/results_steering_chronos/{args.data}.txt"

    # create output_log directory if not exist
    os.makedirs(os.path.dirname(output_log), exist_ok=True)

    SCRATCH_DIR = f"/scratch/s223540177/Time-Moe/chronos/cache_data_finetuned_fixed_4096"
    

    # print("DATA:", args.data)
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    device = f"cuda:{local_rank}"
    
    # evaluation
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map="cuda")
    model = pipe.model
    model.eval()

    ### Training gathering
    gt = {i: [] for i in range(args.dec_in)}
    if args.pred_len in [96, 192, 336, 720]:
        gt_path = os.path.join(SCRATCH_DIR, f'{args.data}/512_96/gt_cache.pt')
    else:
        gt_path = os.path.join(SCRATCH_DIR, f'{args.data}/96_24/gt_cache.pt')
    # gt_path = os.path.join(SCRATCH_DIR, f'{args.data}/gt_cache.pt')

    pred = {i: [] for i in range(args.dec_in)}
    if args.pred_len in [96, 192, 336, 720]:
        pred_path = os.path.join(SCRATCH_DIR, f'{args.data}/512_96/pred_cache.pt')
    else:
        pred_path = os.path.join(SCRATCH_DIR, f'{args.data}/96_24/pred_cache.pt')
    # pred_path = os.path.join(SCRATCH_DIR, f'{args.data}/{args.seq_len}_{args.pred_len}/pred_cache.pt')
    # pred_path = os.path.join(SCRATCH_DIR, f'{args.data}/pred_cache.pt')

    if os.path.exists(pred_path):
        print("LOADDEDEDED")
        pred = torch.load(pred_path)
    else:
        if args.data_path.endswith('.csv'):
            dataset_train = BenchmarkEvalDatasetTrain(
                args,
                args.data_path,
                seq_len=seq_len,
                pred_len=pred_len,
                max_train_samples=max_train_samples
            )
        else:
            raise ValueError("Only csv data is supported for training gathering.")

        if torch.cuda.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset=dataset_train, shuffle=False)
        else:
            sampler = None

        train_dl = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            drop_last=False,
        )

        train_dl_gt = dataset_train.data_loader

        num_each = len(dataset_train) // args.dec_in

        print("len dataset train:", len(dataset_train))
        print("dec_in:", args.dec_in)

        pbar = tqdm(total=len(train_dl), desc="Gathering pred", ncols=100)
        with torch.no_grad():
            for idx, (batch) in enumerate(tqdm(train_dl)):
                # assert batch['inputs] exactly matches x
                channel_id = idx // num_each

                # get the prediction first
                quantiles, mean = pipe.predict_quantiles(batch["inputs"].unsqueeze(0), prediction_length=args.pred_len)
                preds = mean[0]  # this is what predict_df calls "predictions"
                labels = batch['labels']
                ip_and_preds = torch.cat([batch['inputs'].to("cuda"), preds.to("cuda")], dim=1).squeeze(-1)  # [1, D, T1+T2]
                r, hs = get_rep_with_hidden_states(pipe, batch['inputs'], ip_and_preds)


                # pred[channel_id].append((idx, r, hs, batch['inputs']))
                r_cpu  = r.detach().cpu().to(torch.float16).contiguous()
                hs_cpu = hs.detach().cpu().to(torch.float16).contiguous()
                inp    = batch['inputs'].detach().cpu().to(torch.float16).contiguous().clone()

                B = r_cpu.shape[0]  # batch size
                base_idx = idx * B  # base index for this batch

                for b in range(B):
                    global_idx = base_idx + b
                    channel_id = global_idx // num_each
                    # Clamp in case the last batch is incomplete
                    channel_id = min(channel_id, args.dec_in - 1)
                    pred[channel_id].append(
                        (
                            global_idx,
                            r_cpu[b],      # per‑example representation
                            hs_cpu[b],     # per‑example hidden state
                            inp[b].clone(),  # per‑example input
                        )
                    )
                del r, hs, inp, batch

                pbar.update(1)
            pbar.close()

        # create directory if not exist
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        # save pred to pred_path
        torch.save(pred, pred_path)
        print(f'Saved Pred to {pred_path}')


    if os.path.exists(gt_path):
        print("LOADDEDEDED")
        gt = torch.load(gt_path)
    else:
        if args.data_path.endswith('.csv'):
            dataset_train = BenchmarkEvalDatasetTrain(
                args,
                args.data_path,
                seq_len=seq_len,
                pred_len=pred_len,
                max_train_samples=args.pool_number
            )
        else:
            raise ValueError("Only csv data is supported for training gathering.")

        train_dl_gt = dataset_train.data_loader
        num_each = len(dataset_train) // args.dec_in
        
        pbar = tqdm(total=len(train_dl_gt), desc="Gathering gt", ncols=100)
        with torch.no_grad():
            for idx, (x, y, x_mark, y_mark) in enumerate(tqdm(train_dl_gt)):
                # assert batch['inputs] exactly matches x
                channel_id = idx // num_each

                x = x.transpose(1, 2)  # [D, T]
                y = y.transpose(1, 2)  # [D, T]
                
                if x.dim() == 3:
                    x = x.squeeze(1)  # [D, T]
                if y.dim() == 3:
                    y = y.squeeze(1)  # [D, T]

                # concatenate x and y along time dimension
                ip_and_gt = torch.cat([x, y], dim=-1)  # [D, T1+T2]
                r, hs = get_rep_with_hidden_states(pipe, x, ip_and_gt)

                # gt[channel_id].append((idx, r, hs, x))
                r_cpu  = r.detach().cpu().to(torch.float16).contiguous()
                hs_cpu = hs.detach().cpu().to(torch.float16).contiguous()
                x_cpu = x.detach().cpu().to(torch.float16).contiguous().clone()


                B = r_cpu.shape[0]
                base_idx = idx * B

                for b in range(B):
                    global_idx = base_idx + b
                    channel_id = global_idx // num_each
                    channel_id = min(channel_id, args.dec_in - 1)
                    gt[channel_id].append(
                        (
                            global_idx,
                            r_cpu[b],
                            hs_cpu[b],
                            x_cpu[b].clone(),  # per‑example input sequence
                        )
                    )

                del r, hs, x

                pbar.update(1)
            pbar.close()

        # create directory if not exist
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        # save gt to gt_path
        torch.save(gt, gt_path)
        print(f'Saved gt to {gt_path}')
    print("SORTED DONE")




    # NOTE: Start inference with retrieval
    if args.data_path.endswith('.csv'):
        dataset = BenchmarkEvalDataset(
            args.data_path,
            seq_len=seq_len,
            pred_len=pred_len,
        )
    else:
        dataset = GeneralEvalDataset(
            args.data_path,
            seq_len=seq_len,
            pred_len=pred_len,
        )

    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
    else:
        sampler = None


    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
        persistent_workers=False, # be explicit
        pin_memory=False,         # no need here
    )


    acc_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            channel_id = batch['channel_id'].numpy()[0]
            gt_correspond = gt[channel_id]
            pred_correspond = pred[channel_id]

            rep = get_rep(pipe, batch['inputs'])

            dists, samples = retrieve_examples_new(
                args, rep, gt_correspond,
                pool_number=args.pool_number,
                topk=args.num_closest_samples,
                query_series=batch['inputs'].cpu().numpy()
            )

            selected_dists = dists[samples]            # fancy indexing -> NumPy array

            gt_list = [gt_correspond[k] for k in samples]
            pred_list = [pred_correspond[k] for k in samples]
                        
            icv = obtain_icv_interpolate(args, gt_list, pred_list, hidden_size=768,
                                        beta=args.collapse_weight, whiten=True)
            icv = icv.to(device=device, dtype=model.dtype)
            add_icv_layers_interpolate_chronos(
                model,
                torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1),
                # mlp_keywords=("mlp",),
                lam=[args.lam],
                # beta=args.collapse_weight,
                max_frac=1.0
            )
            quantiles, mean = pipe.predict_quantiles(batch["inputs"].unsqueeze(0), prediction_length=args.pred_len)
            preds = mean[0]  # this is what predict_df calls "predictions"
            labels = batch['labels']
            remove_icv_layers_chronos(model)
            mse_base_step = torch.mean((preds - labels) ** 2).item()

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            ret_metric = {}
            for metric in metric_list:
                ret_metric[metric.name] = metric.value / acc_count

            with open(output_log, 'a') as f:
                f.write(json.dumps({'step': idx, 'mse': mse_base_step, "dists": selected_dists.tolist()}) + '\n')

    print(acc_count)
    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count

    print(f'{rank} - {ret_metric}')

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]

    out_path = os.path.join("/home/s223540177/Time-MoE/chronos/metric_results", f'{args.seq_len}_{args.pred_len}/{args.data}/final_metrics/')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # save the results
    if rank == 0:
        with open(os.path.join(out_path, f'weight{args.collapse_weight}_lamb{args.lam}_neighbor{args.num_closest_samples}.txt'), 'w') as f:
            for metric in metric_list:
                f.write(f'{metric.name}: {ret_metric[metric.name]}\n')
        print(f'Saved final metrics to {out_path}')

    # if is_dist:
    #     stat_tensor = torch.tensor(metric_tensors).to(model.device)
    #     gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
    #     dist.all_gather(gathered_results, stat_tensor)
    #     all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    # else:
    #     all_stat = metric_tensors

    # if rank == 0:
    #     item = {
    #         'model': args.model,
    #         'data': args.data_path,
    #         'seq_len': args.seq_len,
    #         'pred_len': args.pred_len,
    #     }

    #     count = all_stat[-1]
    #     for i, metric in enumerate(metric_list):
    #         val = all_stat[i] / count
    #         item[metric.name] = float(val.cpu().numpy())
    #     logging.info(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Maple728/TimeMoE-50M',
        help='Model path'
    )
    parser.add_argument(
        '--data_path', '-d',
        type=str,
        help='Benchmark data path'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=1,
        help='Batch size of evaluation'
    )

    parser.add_argument(
        '--seq_len', '-c',
        type=int,
        help='Context length'
    )
    parser.add_argument(
        '--pred_len', '-p',
        type=int,
        default=96,
        help='Prediction length'
    )
    parser.add_argument(
        '--label_len',
        type=int,
        default=0,
        help='Label length'
    )

    # args from 
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast')
    parser.add_argument('--root_path', type=str, required=False, default='./data/ETT/')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--enc_in', type=int, required=True)
    parser.add_argument('--dec_in', type=int, required=True)
    parser.add_argument('--c_out', type=int, required=True)
    parser.add_argument('--num_closest_samples', type=int, default=4)
    parser.add_argument('--lam', type=float, default=0.5)

    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--pool_number', type=int, default=10000)
    parser.add_argument('--retrieval', type=str, default='cosine')
    parser.add_argument('--tail_n', default=None)
    parser.add_argument('--collapse_weight', type=float, default=0.0)

    args = parser.parse_args()
    # if args.seq_len is None:
    #     if args.pred_len == 96:
    #         args.seq_len = 512
    #     elif args.pred_len == 192:
    #         args.seq_len = 1024
    #     elif args.pred_len == 336:
    #         args.seq_len = 2048
    #     elif args.pred_len == 720:
    #         args.seq_len = 3072
    #     else:
    #         args.seq_len = args.pred_len * 4

    print("ARGS SEQ LEN:", args.seq_len)
    print("ARGS PRED LEN:", args.pred_len)
    if args.seq_len is None:
        if args.pred_len == 96:
            args.seq_len = 512
        elif args.pred_len == 192:
            args.seq_len = 512
        elif args.pred_len == 336:
            args.seq_len = 512
        elif args.pred_len == 720:
            args.seq_len = 512
        else:
            args.seq_len = 96
    print("ARGS SEQ LEN:", args.seq_len)
    print("ARGS PRED LEN:", args.pred_len)
    evaluate(args)
