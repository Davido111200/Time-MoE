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
import random
import glob

from transformers import AutoModelForCausalLM

from data_provider.data_factory import data_provider
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, BenchmarkEvalDatasetTrain, GeneralEvalDataset
from utils.forward_tracer import ForwardTracer, ForwardTrace
from util import obtain_icv_new, add_icv_layers, remove_icv_layers, retrieve_examples_new, cap_each_id

torch.multiprocessing.set_sharing_strategy("file_system")


SCRATCH_DIR = "/scratch/s223540177/Time-Moe/cache_data_finetuned"

def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def flush_gt_part(buffers_gt, parts_gt, ch, scratch_dir, data_tag):
    os.makedirs(os.path.join(scratch_dir, data_tag), exist_ok=True)
    part_path = os.path.join(scratch_dir, f"{data_tag}/gt_cache_ch{ch}_part{parts_gt[ch]}.pt")
    torch.save(buffers_gt[ch], part_path)
    buffers_gt[ch].clear()
    parts_gt[ch] += 1

def finalize_gt_cache(scratch_dir, data_tag, dec_in, final_gt_path, delete_parts=True):
    """
    Load all per-channel part files in order, concatenate, and write a single gt file:
      final structure: {channel_id: [(idx, rep, hs, raw_seq), ...], ...}
    """
    gt_final = {}
    base_dir = os.path.join(scratch_dir, data_tag)

    for ch in range(dec_in):
        pattern = os.path.join(base_dir, f"gt_cache_ch{ch}_part*.pt")
        part_files = sorted(glob.glob(pattern), key=lambda p: int(os.path.splitext(p)[0].split("part")[-1]))
        merged = []
        for pf in part_files:
            chunk = torch.load(pf, map_location="cpu")
            merged.extend(chunk)
        gt_final[ch] = merged

        if delete_parts:
            for pf in part_files:
                try:
                    os.remove(pf)
                except OSError:
                    pass

    # save single final file
    os.makedirs(os.path.dirname(final_gt_path), exist_ok=True)
    torch.save(gt_final, final_gt_path)
    return gt_final  # in case you want it in-memory too



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

@torch.no_grad()
def get_rep_with_hidden_states(model, series_1d: torch.Tensor):
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
                preds, labels = model.predict(batch)
                ip_and_preds = torch.cat([batch['inputs'].to("cuda"), preds], dim=1).squeeze(-1)  # [1, D, T1+T2]
                r, hs = get_rep_with_hidden_states(model.model, ip_and_preds)

                pred[channel_id].append((idx, r, hs, batch['inputs']))
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

                # concatenate x and y along time dimension
                ip_and_gt = torch.cat([x, y], dim=1)  # [D, T1+T2]
                r, hs = get_rep_with_hidden_states(model.model, ip_and_gt)

                gt[channel_id].append((idx, r, hs, x))

                pbar.update(1)
            pbar.close()

        # create directory if not exist
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        # save gt to gt_path
        torch.save(gt, gt_path)
        print(f'Saved gt to {gt_path}')

    for i in range(args.dec_in):
        # randomly select args.pool_number samples
        indices = random.sample(range(len(pred[i])), min(args.pool_number, len(pred[i])))
        pred[i] = [pred[i][j] for j in indices]
        gt[i] = [gt[i][j] for j in indices]


    # NOTE: Select the best steering weight, based on vaidation performance
    # if args.data_path.endswith('.csv'):
    #     dataset_valid = BenchmarkEvalDatasetValid(
    #         args.data_path,
    #         seq_len=seq_len,
    #         pred_len=pred_len,
    #     )
    # else:
    #     raise ValueError("Only csv data is supported for training gathering.")

    # if torch.cuda.is_available() and dist.is_initialized():
    #     sampler = DistributedSampler(dataset=dataset_valid, shuffle=False)
    # else:
    #     sampler = None

    # valid_dl = DataLoader(
    #     dataset=dataset_valid,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     shuffle=False,
    #     num_workers=2,
    #     prefetch_factor=2,
    #     drop_last=False,
    # )

    # lamb_range = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]

    # print("SELECTING LAMBDA")
    # for lam in lamb_range:
    #     acc_count_eval = 0
    #     with torch.no_grad():
    #         for idx, batch in enumerate(tqdm(valid_dl)):
    #             channel_id = batch['channel_id'].numpy()[0]
    #             gt_correspond = gt[channel_id]
    #             pred_correspond = pred[channel_id]

    #             rep = get_rep(model.model, batch['inputs'])

    #             dists, samples = retrieve_examples_new(
    #                 args, rep, gt_correspond,
    #                 pool_number=args.pool_number,
    #                 topk=args.num_closest_samples,
    #                 query_series=batch['inputs'].cpu().numpy()
    #             )

    #             selected_similarities = 1 - dists[samples]

    #             # print(selected_similarities)
    #             selected_similarities = selected_similarities / np.sum(selected_similarities)  # normalize to sum=1

                
    #             gt_list = [gt_correspond[k] for k in samples]
    #             pred_list = [pred_correspond[k] for k in samples]

    #             icv = obtain_icv_new(gt_list, pred_list, rank=1, weights=selected_similarities)
    #             icv = icv[1:]  # keep in sync with your pipeline
    #             icv = icv.to(device=device, dtype=model.dtype)

    #             # inject ICV
    #             add_icv_layers(model.model, torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1), [lam])

    #             preds, labels = model.predict(batch)
    #             remove_icv_layers(model.model)

    #             for metric in metric_list_eval:
    #                 metric.push(preds, labels)

    #             acc_count_eval += count_num_tensor_elements(preds)
    #     print(acc_count_eval)
    #     ret_metric_eval = {}
    #     for metric in metric_list_eval:
    #         ret_metric_eval[metric.name] = metric.value / acc_count_eval

    #     # based on mse to select
    #     if rank == 0:
    #         print(f'lam {lam} - {ret_metric_eval}')
    #         if lam == lamb_range[0]:
    #             best_lam = lam
    #             best_mse = ret_metric_eval['mse']
    #         else:
    #             if ret_metric_eval['mse'] < best_mse:
    #                 best_mse = ret_metric_eval['mse']
    #                 best_lam = lam

    # print(f'Best lam: {best_lam} with mse {best_mse}')
    # # delay for 5 seconds to see the print
    # time.sleep(5)



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

            if args.retrieval == 'euclidean':
                normalized_dists = dists / np.max(dists)
                selected_similarities = 1 - normalized_dists[samples]
            elif args.retrieval == 'cosine':
                selected_similarities = 1 - dists[samples]
                selected_similarities = selected_similarities / np.sum(selected_similarities)  # normalize to sum=1
            else:   
                normalized_dists = dists / np.max(dists)
                selected_similarities = 1 - normalized_dists[samples]

            gt_list = [gt_correspond[k] for k in samples]
            pred_list = [pred_correspond[k] for k in samples]

            icv = obtain_icv_new(gt_list, pred_list, rank=1, weights=selected_similarities)
            icv = icv[1:]  # keep in sync with your pipeline
            icv = icv.to(device=device, dtype=model.dtype)

            # inject ICV
            add_icv_layers(model.model, torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1), [args.lam])

            preds, labels = model.predict(batch)
            remove_icv_layers(model.model)

            for metric in metric_list:
                metric.push(preds, labels)

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
    parser.add_argument('--num_closest_samples', type=int, default=16)
    parser.add_argument('--lam', type=float, default=0.5)

    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--pool_number', type=int, default=10000)
    parser.add_argument('--retrieval', type=str, default='cosine')
    parser.add_argument('--tail_n', default=None)

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
