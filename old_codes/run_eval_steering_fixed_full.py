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
import wandb

from transformers import AutoModelForCausalLM

from data_provider.data_factory import data_provider
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, BenchmarkEvalDatasetTrain, GeneralEvalDataset
from utils.forward_tracer import ForwardTracer, ForwardTrace
from util import obtain_icv_interpolate, obtain_icv_new, add_icv_layers, add_icv_layers_confidence, add_icv_layers_new, remove_icv_layers, retrieve_examples_new, icv_tightness
# put this at the very top of your entry script (before creating any DataLoader)

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

wandb.init(project='Time-MoE-Eval')
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

def get_rep(model, series_1d: torch.Tensor):
    # representation used by retrieve_examples (mean pooled last hidden state)
    dtype = next(model.parameters()).dtype
    device = model.device

    series_1d = series_1d.to(device=device, dtype=dtype)

    # ask model for hidden states; trust_remote_code should support this
    out = model(series_1d, output_hidden_states=True, use_cache=False, return_dict=True)
    # final hidden state: [B, T, H]
    h = out.hidden_states[-1]          # also [B, T, H]

    # mean pooling over second dimension, shape should be [B, H]
    h = h.mean(dim=1).squeeze(0).contiguous()  
    return h


def get_rep_with_hidden_states(model, series_1d_input_only, series_1d: torch.Tensor, pred_len: int):
    """
    Returns:
        rep      : [B, H]              mean over time of the final layer on input-only
        hs       : [B, L*H]            concat of last-token states from all layers (unchanged)
        future_h : [B, pred_len, H]    final-layer hidden states over prediction horizon
    """
    params = next(model.parameters())
    device = params.device
    dtype  = params.dtype

    x_full = series_1d.to(device=device, dtype=dtype)
    x_inp  = series_1d_input_only.to(device=device, dtype=dtype)

    # Full sequence: input + future (preds or targets)
    out_full = model(
        x_full,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )

    # Input-only (for retrieval rep)
    out_inp = model(
        x_inp,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )

    hidden_full = list(out_full.hidden_states)   # length L, each [B, T_full, H]
    hidden_inp  = list(out_inp.hidden_states)    # length L, each [B, T_inp, H]

    # 1) Retrieval representation: mean over time of final layer on input-only
    h_final_inp = hidden_inp[-1]                # [B, T_inp, H]
    rep = h_final_inp.mean(dim=1).contiguous()  # [B, H]

    # 2) hs as before: last-token from each layer, flattened
    last_tokens = [h[:, -1, :] for h in hidden_full]       # L x [B, H]
    hs_layers   = torch.stack(last_tokens, dim=1).contiguous()  # [B, L, H]
    hs = hs_layers.view(hs_layers.size(0), -1).contiguous()     # [B, L*H]

    # 3) New: final-layer hidden states over prediction horizon
    h_final_full = hidden_full[-1]              # [B, T_full, H]
    future_h = h_final_full[:, -pred_len:, :].contiguous()  # [B, pred_len, H]

    return rep, hs, future_h


# def get_rep_with_hidden_states(model, series_1d_input_only, series_1d: torch.Tensor):
#     """
#     Returns:
#         rep: [B, H]      mean over time of the final layer
#         hs:  [B, L*H]    concat of last-token states from all layers
#     """
#     params = next(model.parameters())
#     device = params.device
#     dtype  = params.dtype

#     x = series_1d.to(device=device, dtype=dtype)

#     x_input = series_1d_input_only.to(device=device, dtype=dtype)

#     # You don't need the tracer if hidden_states are returned.
#     out = model(
#         x,
#         output_hidden_states=True,
#         output_attentions=False,
#         use_cache=False,
#         return_dict=True,
#     )

#     out_inp_only = model(
#         x_input,
#         output_hidden_states=True,
#         output_attentions=False,
#         use_cache=False,
#         return_dict=True,
#     )

#     # Use the model's own hidden_states (list/tuple of length L), each [B, T, H]
#     hidden_list = list(out.hidden_states)
#     h_final_input_only = list(out_inp_only.hidden_states)[-1]

#     # Last token from each layer -> [B, H], then stack along a new L-dim -> [B, L, H]
#     last_tokens = [h[:, -1, :] for h in hidden_list]          # L × [B, H]
#     hs_layers   = torch.stack(last_tokens, dim=1).contiguous() # [B, L, H]

#     # Flatten layers into features -> [B, L*H]
#     hs = hs_layers.view(hs_layers.size(0), -1).contiguous()

#     # Final layer mean over time -> [B, H]
#     # h_final = hidden_list[-1]                                  # [B, T, H]
#     rep = h_final_input_only.mean(dim=1).contiguous()                     # [B, H]

#     return rep, hs


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


# class TimeMoE:
#     def __init__(self, model_path, device, seq_len, pred_len, **kwargs):
#         try:
#             from time_moe.models.modeling_time_moe import TimeMoeForPrediction
#             model = TimeMoeForPrediction.from_pretrained(
#                 model_path,
#                 device_map=device,
#                 # attn_implementation='flash_attention_2',
#                 torch_dtype='auto',
#             )
#         except:
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 device_map=device,
#                 # attn_implementation='flash_attention_2',
#                 torch_dtype='auto',
#                 trust_remote_code=True,
#             )

#         print(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

#         self.model = model
#         self.device = device
#         self.pred_len = pred_len
#         self.dtype = model.dtype
#         self.model.eval()

#     def predict(self, batch):
#         model = self.model
#         device = self.device
#         pred_len = self.pred_len


#         outputs = model.generate(
#             inputs=batch['inputs'].to(device).to(self.dtype),
#             max_new_tokens=pred_len,
#         )
#         preds = outputs[:, -pred_len:]
#         labels = batch['labels'].to(device)
#         if len(preds.shape) > len(labels.shape):
#             labels = labels[..., None]
#         return preds, labels

class TimeMoE:
    def __init__(self, model_path, device, seq_len, pred_len, **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype="auto",
                trust_remote_code=True,
            )

        print(f">>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}")
        self.model = model
        self.device = device
        self.pred_len = pred_len
        self.dtype = model.dtype
        self.model.eval()

    def predict(self, batch):
        # your old version (no confidence gating)
        model = self.model
        device = self.device
        pred_len = self.pred_len

        outputs = model.generate(
            inputs=batch['inputs'].to(device).to(self.dtype),
            max_new_tokens=pred_len,
        )
        preds = outputs[:, -pred_len:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels




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

    output_log = f"/home/s223540177/Time-MoE/results_steering/{args.data}.txt"

    # create output_log directory if not exist
    os.makedirs(os.path.dirname(output_log), exist_ok=True)

    # SCRATCH_DIR = f"/scratch/s223540177/Time-Moe/cache_data_finetuned_fixed"
    

    SCRATCH_DIR = f"/scratch/s223540177/Time-Moe/cache_data_fixed_full_{args.pool_number}"

    # print("DATA:", args.data)
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    # if torch.cuda.is_available():
    #     try:
    #         setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
    #         device = f"cuda:{local_rank}"
    #         is_dist = True
    #     except Exception as e:
    #         print('Error: ', f'Setup nccl fail, so set device to cpu: {e}')
    #         device = 'cpu'
    #         is_dist = False
    # else:
    #     device = 'cpu'
    #     is_dist = False
    device = f"cuda:{local_rank}"
    is_dist = True
    
    # evaluation
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    model = TimeMoE(
        args.model,
        device,
        seq_len=seq_len,
        pred_len=pred_len
    )

    ### Training gathering
    gt = {i: [] for i in range(args.dec_in)}
    gt_path = os.path.join(SCRATCH_DIR, f'{args.data}/gt_cache.pt')

    pred = {i: [] for i in range(args.dec_in)}
    pred_path = os.path.join(SCRATCH_DIR, f'{args.data}/pred_cache.pt')

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
                preds, labels = model.predict(batch)

                ip_and_preds = torch.cat(
                    [batch["inputs"].to("cuda"), preds],
                    dim=1
                ).squeeze(-1)  # [B, T_inp + pred_len]

                r, hs, future_h = get_rep_with_hidden_states(
                    model.model,
                    batch["inputs"],   # input-only
                    ip_and_preds,
                    pred_len=pred_len,
                )

                r_cpu      = r.detach().cpu().to(torch.float16).contiguous()
                hs_cpu     = hs.detach().cpu().to(torch.float16).contiguous()
                future_cpu = future_h.detach().cpu().to(torch.float16).contiguous()   # [B, pred_len, H]
                inp_cpu    = batch["inputs"].detach().cpu().to(torch.float16).contiguous().clone()

                # NEW layout: (idx, rep, hs, future_h, input_series)
                pred[channel_id].append((idx, r_cpu, hs_cpu, future_cpu, inp_cpu))

                del r, hs, batch

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

                x = x.transpose(1, 2).squeeze(0)  # [D, T_inp]
                y = y.transpose(1, 2).squeeze(0)  # [D, T_label+pred]

                ip_and_gt = torch.cat([x, y], dim=1)  # [D, T_inp + T_label + pred_len]

                r, hs, future_h = get_rep_with_hidden_states(
                    model.model,
                    x,            # input-only
                    ip_and_gt,    # input + full gt future
                    pred_len=pred_len,
                )

                r_cpu      = r.detach().cpu().to(torch.float16).contiguous()
                hs_cpu     = hs.detach().cpu().to(torch.float16).contiguous()
                future_cpu = future_h.detach().cpu().to(torch.float16).contiguous()   # [B, pred_len, H]
                x_cpu      = x.detach().cpu().to(torch.float16).contiguous().clone()

                # NEW layout: (idx, rep, hs, future_h, input_series)
                gt[channel_id].append((idx, r_cpu, hs_cpu, future_cpu, x_cpu))

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


            rep = get_rep(model.model, batch['inputs'])

            dists, samples = retrieve_examples_new(
                args, rep, gt_correspond,
                pool_number=args.pool_number,
                topk=args.num_closest_samples,
                query_series=batch['inputs'].cpu().numpy()
            )

            selected_dists = dists[samples]            # fancy indexing -> NumPy array

            gt_list   = [gt_correspond[k] for k in samples]
            pred_list = [pred_correspond[k] for k in samples]

            # Hidden size H from future_h: [B, pred_len, H]
            hidden_size = int(gt_list[0][3].shape[-1])

            # Get a single direction in H-dim hidden space
            direction_h = obtain_icv_new(
                gt_list,
                pred_list,
                hidden_shape=hidden_size,
                rank=1,
            )

            # We still need one vector per layer. Use hs (index 2) to infer number of layers.
            hs_sample = pred_list[0][2]         # [B, L*H]
            L = hs_sample.shape[-1] // hidden_size

            # Broadcast direction across layers: [L, H]
            icv_layerwise = direction_h.unsqueeze(0).expand(L, -1)  # [L, H]

            # (Optional) skip the bottom layer if you used icv[1:] before:
            icv_layerwise = icv_layerwise[1:]

            # Shape for add_icv_layers: [L, 1, H]
            icv_stack = icv_layerwise.unsqueeze(1)  # [L, 1, H]
            icv_stack = icv_stack.to(device=device, dtype=model.dtype)

            add_icv_layers(model.model, icv_stack, [args.lam])

            preds, labels = model.predict(batch)

            remove_icv_layers(model.model)
            mse_base_step = torch.mean((preds - labels) ** 2).item()

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            ret_metric = {}
            for metric in metric_list:
                ret_metric[metric.name] = metric.value / acc_count
            wandb.log({f'eval/{metric.name}': ret_metric[metric.name] for metric in metric_list}, step=idx)

            with open(output_log, 'a') as f:
                f.write(json.dumps({'step': idx, 'mse': mse_base_step, "dists": selected_dists.tolist()}) + '\n')

    print(acc_count)
    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count

    print(f'{rank} - {ret_metric}')

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    if rank == 0:
        item = {
            'model': args.model,
            'data': args.data_path,
            'seq_len': args.seq_len,
            'pred_len': args.pred_len,
        }

        count = all_stat[-1]
        for i, metric in enumerate(metric_list):
            val = all_stat[i] / count
            item[metric.name] = float(val.cpu().numpy())
        logging.info(item)


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
    if args.seq_len is None:
        if args.pred_len == 96:
            args.seq_len = 512
        elif args.pred_len == 192:
            args.seq_len = 1024
        elif args.pred_len == 336:
            args.seq_len = 2048
        elif args.pred_len == 720:
            args.seq_len = 3072
        else:
            args.seq_len = args.pred_len * 4
    evaluate(args)
