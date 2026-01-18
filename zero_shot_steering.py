from contextlib import contextmanager
from tqdm import tqdm
import numpy as np
import argparse
import os
import torch
from data_provider.data_factory import data_provider
from utils.metrics import metric
import time
import itertools
import matplotlib.pyplot as plt
import wandb
import math, heapq, time

from transformers import AutoModelForCausalLM
from util import obtain_icv, add_icv_layers, remove_icv_layers, retrieve_examples, visualize_data

from utils.forward_tracer import ForwardTracer, ForwardTrace

# ---------------------------
# Thin wrapper to mimic ChatTime API with TimeMoE


import torch


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


# ---------------------------
# Seed & Data helpers
# ---------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def build_incontext_map(dataset_size: int, num_examples: int):
    all_indices = range(dataset_size)
    perms = itertools.permutations(all_indices, num_examples)
    return {i: p for i, p in enumerate(perms)}

# ---------------------------
# Argparse: kept identical
# ---------------------------
parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
parser.add_argument('--is_training', type=int, required=True, default=1)
parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--model', type=str, required=True, default='Autoformer')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1')
parser.add_argument('--root_path', type=str, default='./data/ETT/')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
parser.add_argument('--inverse', action='store_true', default=False)

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25)

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25)

# model define (kept for compatibility)
parser.add_argument('--expand', type=int, default=2)
parser.add_argument('--d_conv', type=int, default=4)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--num_kernels', type=int, default=6)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--distil', action='store_false', default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--channel_independence', type=int, default=1)
parser.add_argument('--decomp_method', type=str, default='moving_avg')
parser.add_argument('--use_norm', type=int, default=1)
parser.add_argument('--down_sampling_layers', type=int, default=0)
parser.add_argument('--down_sampling_window', type=int, default=1)
parser.add_argument('--down_sampling_method', type=str, default=None)
parser.add_argument('--seg_len', type=int, default=96)

# optimization
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--use_amp', action='store_true', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_type', type=str, default='cuda')
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
parser.add_argument('--p_hidden_layers', type=int, default=2)

# metrics (dtw)
parser.add_argument('--use_dtw', type=bool, default=False)

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--jitter', default=False, action="store_true")
parser.add_argument('--scaling', default=False, action="store_true")
parser.add_argument('--permutation', default=False, action="store_true")
parser.add_argument('--randompermutation', default=False, action="store_true")
parser.add_argument('--magwarp', default=False, action="store_true")
parser.add_argument('--timewarp', default=False, action="store_true")
parser.add_argument('--windowslice', default=False, action="store_true")
parser.add_argument('--windowwarp', default=False, action="store_true")
parser.add_argument('--rotation', default=False, action="store_true")
parser.add_argument('--spawner', default=False, action="store_true")
parser.add_argument('--dtwwarp', default=False, action="store_true")
parser.add_argument('--shapedtwwarp', default=False, action="store_true")
parser.add_argument('--wdba', default=False, action="store_true")
parser.add_argument('--discdtw', default=False, action="store_true")
parser.add_argument('--discsdtw', default=False, action="store_true")
parser.add_argument('--extra_tag', type=str, default="")

# TimeXer
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--weighted_loss', type=int, default=0)

# RL params (kept, unused in this script)
parser.add_argument('--num_processes', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--clip_param', type=float, default=0.2)
parser.add_argument('--ppo_epoch', type=int, default=4)
parser.add_argument('--num_mini_batch', type=int, default=4)
parser.add_argument('--value_loss_coef', type=float, default=0.5)
parser.add_argument('--entropy_coef', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=0.5)
parser.add_argument('--use_gae', action='store_true')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gae_lambda', type=float, default=0.95)
parser.add_argument('--use_proper_time_limits', action='store_true')
parser.add_argument('--num_env_steps', type=int, default=10000)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='/home/s223540177/ICL4TS/saves/trained_models/')
parser.add_argument('--num_examples', type=int, default=4)
parser.add_argument('--num_closest_samples', type=int, default=4)
parser.add_argument('--retrieval', type=str, default='euclidean')
parser.add_argument('--tail_n', type=int, default=8)

# ICV
parser.add_argument('--subsample_size', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.1)

args = parser.parse_args()
setup_seed(args.seed)

# ---- Weights & Biases ----
run = wandb.init(
    project="iclts-steering",
    name=f"{args.model_id}",
    config=vars(args),
)
wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")

# ---------------------------
# Data
# ---------------------------
train_data, train_loader = get_data(args, flag='train')
vali_data, vali_loader = get_data(args, flag='val')
test_data, test_loader = get_data(args, flag='test')


# ---------------------------
# Model: TimeMoE with ChatTime-like API
# ---------------------------
model_path = "Maple728/TimeMoE-50M"
device = args.gpu_type if (args.use_gpu and torch.cuda.is_available() and args.gpu_type == 'cuda') else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(args.gpu)

# model = TimeMoEWrapper(args=args, hist_len=args.seq_len, pred_len=args.pred_len, model_path=model_path, device=device)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    trust_remote_code=True,
)
print('test_data: ', len(test_data))

# ---------------------------
# Build vector DB using K lowest-entropy examples
# ---------------------------
K = 100
train_subset = torch.utils.data.Subset(train_data, range(K))
train_loader_small = torch.utils.data.DataLoader(
    dataset=train_subset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=True
)

# train_subset = train_data
# train_loader_small = train_loader

k = len(train_loader_small)
items = []  # (entropy, idx, r, hs, d)

pbar = tqdm(total=len(train_loader_small), desc='Training', ncols=100)
with torch.no_grad():
    for i, (x, y, x_mark, y_mark) in enumerate(train_loader_small):
        x = x.transpose(1, 2).squeeze(0)  # [D, T]
        x = x[-1].unsqueeze(0)  # use only target variable for retrieval, [T]
        r, hs = get_rep_with_hidden_states(model, x)
        items.append((i, r, hs, x))
        pbar.update(1)
pbar.close()

# select k lowest-entropy (same as your code intent)
# randomly select k indices
N = len(items)
k = min(1000, N)                       # guard if N < 1000
idx = torch.randperm(N)[:k].tolist()   # Python ints for list indexing
best_list = [items[i] for i in idx]    # subset of your list

vector_db = {}
raw_series = []
for idx, (i, r, hs, d) in enumerate(best_list):
    vector_db[idx] = (r, hs, d)
    raw_series.append(d)

# ---------------------------
# Evaluation with steering (ICV injection)
# ---------------------------
num_eval_samples = len(test_data)  # set to -1 to run full test set
preds = []
trues = []

print("TEST loader length: ", len(test_loader))

for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
    batch_x = batch_x.transpose(1, 2).squeeze(0)  # [D, T]
    batch_y = batch_y.transpose(1, 2).squeeze(0)  # [D, pred_len]

    batch_x = batch_x[-1].unsqueeze(0)  # use only target variable for retrieval, [T]
    batch_y = batch_y[-1].unsqueeze(0)  # [pred_len]

    mean, std = batch_x.mean(dim=-1, keepdim=True), batch_x.std(dim=-1, keepdim=True)
    batch_x = (batch_x - mean) / std

    rep = get_rep(model, batch_x)
    
    # retrieve examples
    dists, samples = retrieve_examples(
        args, rep, vector_db,
        pool_number=len(vector_db),
        topk=args.num_closest_samples,
        query_series=batch_x.cpu().numpy()
    )

    selected_dists = dists[samples]
    selected_dists = selected_dists / np.sum(selected_dists)  # normalize to sum=1

    ex_list = [vector_db[k] for k in samples]
    icv = obtain_icv(ex_list, rank=1, weights=selected_dists)
    icv = icv[1:]  # keep in sync with your pipeline

    # inject ICV (your util expects .model)
    add_icv_layers(model, torch.stack([icv], dim=1).cuda() if device == 'cuda' else torch.stack([icv], dim=1), [args.lam])

    hist_data = batch_x  # [T]
    output = model.generate(batch_x.to(dtype=model.dtype), max_new_tokens=args.pred_len)  # shape is [batch_size, 12 + 6]
    normed_predictions = output[:, -args.pred_len:]  # shape is [batch_size, 6]    outputs = np.expand_dims(output, axis=0)  # [1, pred_len]

    predictions = normed_predictions * std + mean

    trues.append(batch_y[-1:].numpy())  # [1, pred_len]
    preds.append(predictions)               # [1, pred_len]

    # remove ICV after each prediction
    remove_icv_layers(model)

    # per-iter metrics (kept consistent with your earlier print style)
    
    mse = torch.mean((predictions - batch_y[-1:].numpy()) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - batch_y[-1:].numpy()))
    print(f'Iter: {i}, mse: {mse:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}')

    if i == num_eval_samples:
        break

trues = np.concatenate(trues, axis=0)
preds = np.concatenate(preds, axis=0)
mae, mse, rmse, mape, mspe = metric(preds, trues)

folder_path = 'test_results/zeroshot/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
np.save(folder_path + 'pred.npy', preds)
np.save(folder_path + 'true.npy', trues)

print('mse: ', mse, 'rmse: ', rmse, 'mae: ', mae, 'mape: ', mape, 'mspe: ', mspe)
