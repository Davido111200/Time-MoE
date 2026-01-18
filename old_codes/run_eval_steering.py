#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from data_provider.data_factory import data_provider
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset
from utils.forward_tracer import ForwardTracer, ForwardTrace
from util import obtain_icv, add_icv_layers, remove_icv_layers, retrieve_examples, visualize_data

SCRATCH_DIR = "/scratch/s223540177/Time-Moe/cache_data"

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


def get_rep_with_hidden_states(model, series_1d: torch.Tensor):
    """
    Returns:
        rep: [B, H]      mean over time of the final layer
        hs:  [B, L*H]    concat of last-token states from all layers
    """

    params = next(model.parameters())
    device = params.device
    dtype  = params.dtype

    x = series_1d.to(device=device, dtype=dtype)

    forward_trace = ForwardTrace()
    contextmanager = ForwardTracer(model, forward_trace)
    with contextmanager:
        out = model(
            x,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )
        hi = forward_trace.residual_stream.hidden
    embedding_token = []
    for layer in range(len(hi)):
        embedding_token.append(hi[layer][:,-1])
    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
    # final layer: [B, T, H]
    h_final = out.hidden_states[-1]

    # mean-pool over time -> [B, H]  (no squeeze!)
    rep = h_final.mean(dim=1).contiguous()

    return rep, embedding_token


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


class TimeMoE:
    def __init__(self, model_path, device, seq_len, pred_len, **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        print(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.pred_len = pred_len
        self.dtype = model.dtype
        self.model.eval()

    def predict(self, batch):
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

    model = TimeMoE(
        args.model,
        device,
        seq_len=seq_len,
        pred_len=pred_len
    )

    ### Training gathering
    train_data, train_loader = get_data(args, flag='train')
    

    items = {i: [] for i in range(args.dec_in)}
    items_path = os.path.join(SCRATCH_DIR, f'{args.data}/cache.pt')

    # if items_path exists, load it
    if os.path.exists(items_path):
        items = torch.load(items_path)
    else:
        os.makedirs(os.path.dirname(items_path), exist_ok=True)
        print(f'Extracting representations and saving to {items_path}')
        pbar = tqdm(total=len(train_loader), desc="Training", ncols=100)
        with torch.no_grad():
            for i, (x, y, x_mark, y_mark) in enumerate(train_loader):
                x = x.transpose(1, 2).squeeze(0)  # [D, T]
                
                # loop through each channel
                for channel in range(x.shape[0]):
                    series = x[channel].unsqueeze(0)
                    r, hs = get_rep_with_hidden_states(model.model, series)
                    items[channel].append((i, r, hs, series))
                pbar.update(1)
            pbar.close()

        # save items to items_path
        torch.save(items, items_path)
        print(f'Saved items to {items_path}')


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
    )

    acc_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            channel_id = batch['channel_id'].numpy()[0]
            items_correspond = items[channel_id]
            rep = get_rep(model.model, batch['inputs'])

            dists, samples = retrieve_examples(
                args, rep, items_correspond,
                pool_number=len(items_correspond),
                topk=args.num_closest_samples,
                query_series=batch['inputs'].cpu().numpy()
            )
            selected_dists = dists[samples]
            selected_dists = selected_dists / np.sum(selected_dists)  # normalize to sum=1

            ex_list = [items_correspond[k] for k in samples]
            icv = obtain_icv(ex_list, rank=1, weights=selected_dists)
            icv = icv[1:]  # keep in sync with your pipeline
            icv = icv.to(device=device, dtype=model.dtype)

            # inject ICV (your util expects .model)
            add_icv_layers(model.model, torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1), [args.lam])

            preds, labels = model.predict(batch)

            remove_icv_layers(model.model)

            for metric in metric_list:
                metric.push(preds, labels)
                print(f'Metric {metric.name}: {metric.value}')

            print(preds)
            quit()
            
            acc_count += count_num_tensor_elements(preds)

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
