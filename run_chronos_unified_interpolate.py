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
from chronos import BaseChronosPipeline, Chronos2Pipeline


from data_provider.data_factory import data_provider
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, BenchmarkEvalDatasetTrain, GeneralEvalDataset
from utils.forward_tracer import ForwardTracer, ForwardTrace
from util import obtain_icv_interpolate, obtain_icv_dtf, add_icv_layers_interpolate_chronos, add_icv_layers_confidence, add_icv_layers_new, remove_icv_layers_chronos, retrieve_examples_new, icv_tightness

from model_classes.util import build_model


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

def get_rep(args, pipe, series_1d: torch.Tensor):
    # representation used by retrieve_examples (mean pooled last hidden state)
    quantiles, mean, hidden_list = pipe.predict_quantiles(series_1d, prediction_length=args.pred_len)
    preds = mean[0]  # this is what predict_df calls "predictions"

    h = hidden_list[0][-1]  # final hidden state: [B, T, H]

    # mean pooling over second dimension, shape should be [B, H]
    h = h.mean(dim=1).squeeze(0).contiguous()  
    return h



def get_rep_with_hidden_states(args, pipe, series_1d: torch.Tensor):
    """
    Returns:
        rep: [B, H]      mean over time of the final layer
        hs:  [B, L*H]    concat of last-token states from all layers
    """
    quantiles, mean, hidden_list = pipe.predict_quantiles(series_1d, prediction_length=args.pred_len)
    preds = mean[0]  # this is what predict_df calls "predictions"

    # Use the model's own hidden_states (list/tuple of length L), each [B, T, H]

    # print(len(hidden_states[0]))

    last_tokens = [h[:, -1, :] for h in hidden_list[0]]          # L × [B, H]
    hs_layers   = torch.stack(last_tokens, dim=1).contiguous() # [B, L, H]

    # Flatten layers into features -> [B, L*H]
    hs = hs_layers.view(hs_layers.size(0), -1).contiguous()

    # Final layer mean over time -> [B, H]
    h_final = hidden_list[0][-1]                                  # [B, T, H]
    rep = h_final.mean(dim=1).contiguous()                     # [B, H]
    return rep, hs



def pred_classic(args, model, test_loader):
    """
    Returns (preds, labels) like TimeMoE.predict:
      preds  : [N, pred_len] or [N, pred_len, C]
      labels : [N, pred_len] or [N, pred_len, C]
    """

    model.eval()

    # Prefer args.pred_len; fall back to self.pred_len if present.
    pred_len = getattr(args, 'pred_len', None)

    # Model dtype/device
    try:
        model_dtype = model.dtype
    except Exception:
        model_dtype = next(model.parameters()).dtype
    device = device

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Move to device / dtype
            batch_x = batch_x.to(device).to(model_dtype)
            batch_y = batch_y.to(device)

            if hasattr(model, "generate"):
                # LLM-style generation
                outputs = model.generate(inputs=batch_x, max_new_tokens=pred_len)
                preds = outputs[:, -pred_len:]
            else:
                # Fallback: assume forward returns a sequence; take last pred_len on time dim
                out = model(batch_x)
                if out.dim() == 3:       # [B, T, C]
                    preds = out[:, -pred_len:, :]
                elif out.dim() == 2:     # [B, T]
                    preds = out[:, -pred_len:]
                else:
                    raise RuntimeError(
                        f"Unsupported model output shape {tuple(out.shape)}; "
                        "expected 2D [B,T] or 3D [B,T,C]."
                    )

            # --- Labels to match shape/time window ---
            labels = batch_y
            # If labels are longer than pred_len, align to last pred_len steps
            if labels.size(1) != pred_len:
                labels = labels[:, -pred_len:, ...] if labels.dim() >= 2 else labels[:, -pred_len:]

            # If preds has an extra channel dim but labels doesn't, unsqueeze labels
            if preds.dim() == 3 and labels.dim() == 2:
                labels = labels.unsqueeze(-1)
            # Conversely, if preds is 2D and labels is 3D with singleton channel, squeeze
            if preds.dim() == 2 and labels.dim() == 3 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)

            preds_all.append(preds.detach().cpu())
            labels_all.append(labels.detach().cpu())

    preds = torch.cat(preds_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    return preds, labels



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


def evaluate(args):
    batch_size = args.batch_size
    seq_len = args.seq_len
    pred_len = args.pred_len

    SCRATCH_DIR = f"/scratch/s223540177/Time-Moe/chronos/cache_data_finetuned_fixed_chronos"


    print("DATA:", args.data)
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

    # model = TimeMoE(
    #     args.model,
    #     device,
    #     seq_len=seq_len,
    #     pred_len=pred_len
    # )

    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    model = pipe.model
    model.eval()

    ### Training gathering
    gt = {i: [] for i in range(args.dec_in)}
    gt_path = os.path.join(SCRATCH_DIR, f'{args.data}/chronos/gt_cache.pt')

    pred = {i: [] for i in range(args.dec_in)}
    pred_path = os.path.join(SCRATCH_DIR, f'{args.data}/chronos/pred_cache.pt')

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
                max_train_samples=args.pool_number
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
        pbar = tqdm(total=len(train_dl), desc="Gathering pred", ncols=100)
        with torch.no_grad():
            for idx, (batch) in enumerate(tqdm(train_dl)):
                # assert batch['inputs] exactly matches x
                channel_id = idx // num_each

                # get the prediction first
                quantiles, mean, hidden_states = pipe.predict_quantiles(batch["inputs"].unsqueeze(0), prediction_length=args.pred_len)
                preds = mean[0]  # this is what predict_df calls "predictions"

                ip_and_preds = torch.cat([batch['inputs'], preds], dim=1)  # [1, D, T1+T2]
                ip_and_preds = ip_and_preds.unsqueeze(0)  # [B=1, D, T1+T2]
                
                r, hs = get_rep_with_hidden_states(args, pipe, ip_and_preds)

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

                x = x.transpose(1, 2).squeeze(0)  # [D, T]
                y = y.transpose(1, 2).squeeze(0)  # [D, T]

                if x.dim() == 3:
                    x = x.squeeze(1)  # [D, T]
                if y.dim() == 3:
                    y = y.squeeze(1)  # [D, T]

                # concatenate x and y along time dimension
                ip_and_gt = torch.cat([x, y], dim=1)  # [D, T1+T2]
                ip_and_gt = ip_and_gt.unsqueeze(0)  # [B=1, D, T1+T2]

                r, hs = get_rep_with_hidden_states(args, pipe, ip_and_gt)

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


    # for each channel, we only want to take args.pool_number samples
    for i in range(args.dec_in):
        # randomly select args.pool_number samples
        if args.pool_number >= len(pred[i]):
            continue
        indices = random.sample(range(len(pred[i])), min(args.pool_number, len(pred[i])))
        pred[i] = [pred[i][j] for j in indices]
        gt[i] = [gt[i][j] for j in indices]

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


    # for each channel, we only want to take args.pool_number samples
    for i in range(args.dec_in):
        # randomly select args.pool_number samples
        # indices = random.sample(range(len(pred[i])), min(args.pool_number, len(pred[i])))
        # select the last args.pool_number samples
        if args.pool_number >= len(pred[i]):
            continue
        indices = [j for j in range(len(pred[i]) - args.pool_number, len(pred[i]))]
        pred[i] = [pred[i][j] for j in indices]
        gt[i] = [gt[i][j] for j in indices]


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

    print("Test dl:", test_dl)
    print("Len test dl:", len(test_dl))


    acc_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            channel_id = batch['channel_id'].numpy()[0]
            gt_correspond = gt[channel_id]
            pred_correspond = pred[channel_id]

            rep = get_rep(args, pipe, batch['inputs'].unsqueeze(0))  # [H]

            dists, samples = retrieve_examples_new(
                args, rep, gt_correspond,
                pool_number=args.pool_number,
                topk=args.num_closest_samples,
                query_series=batch['inputs'].cpu().numpy()
            )

            # print("retrieved samples:", samples)

            gt_list = [gt_correspond[k] for k in samples]
            pred_list = [pred_correspond[k] for k in samples]
                        
            # NOTE: hidden_size should be set according to model
            # icv = obtain_icv_unified(args, gt_list, pred_list, hidden_size=384, whiten=True, normalize_per_layer=True)
            icv = obtain_icv_interpolate(args, gt_list, pred_list, hidden_size=768, beta=args.collapse_weight , whiten=True)
            # icv = obtain_icv_interpolate(gt_list, pred_list, hidden_size=384)
            icv = icv[1:]  # keep in sync with your pipeline
            icv = icv.to(device=device, dtype=model.dtype)

            # inject ICV
            # add_icv_layers_unified(model.model, torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1), [args.lam], max_frac=1.0)
            # add_icv_layers_new(model.model, torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1), [args.lam])

            # sched = {'tau': 14, 'warmup': 0, 'cutoff': 10**9, 'scale': 1.0}
            # icv_list = [icv[i].contiguous() for i in range(icv.shape[0])]
            
            add_icv_layers_interpolate_chronos(pipe.model, torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1), lam=[args.lam], beta=args.collapse_weight, max_frac=1.0)

            quantiles, mean, hidden_states = pipe.predict_quantiles(batch["inputs"].unsqueeze(0), prediction_length=args.pred_len)
            preds = mean[0]  # this is what predict_df calls "predictions"

            labels = batch['labels']
            
            remove_icv_layers_chronos(pipe.model)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            ret_metric = {}
            for metric in metric_list:
                ret_metric[metric.name] = metric.value / acc_count
            wandb.log({f'eval/{metric.name}': ret_metric[metric.name] for metric in metric_list}, step=idx)


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
        # default='Maple728/TimeMoE-50M',
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
        default=24,
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
    parser.add_argument('--num_closest_samples', type=int, default=16)
    parser.add_argument('--lam', type=float, default=0.5)

    parser.add_argument('--pool_number', type=int, default=10000)
    parser.add_argument('--retrieval', type=str, default='euclidean')
    parser.add_argument('--tail_n', default=None)
    parser.add_argument('--collapse_weight', type=float, default=0.0)

    # classical configs
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--down_sampling_layers', type=int, default=3)
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=False)
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')


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
