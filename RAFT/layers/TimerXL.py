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
import torch.nn as nn
from transformers import AutoModelForCausalLM

from data_provider.data_factory import data_provider

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


def get_rep_with_hidden_states(model, series_1d_input_only, series_1d: torch.Tensor):
    """
    Returns:
        rep: [B, H]      mean over time of the final layer
        hs:  [B, L*H]    concat of last-token states from all layers
    """
    params = next(model.parameters())
    device = params.device
    dtype  = params.dtype

    x = series_1d.to(device=device, dtype=dtype)

    x_input = series_1d_input_only.to(device=device, dtype=dtype)

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

    # Use the model's own hidden_states (list/tuple of length L), each [B, T, H]
    hidden_list = list(out.hidden_states)
    h_final_input_only = list(out_inp_only.hidden_states)[-1]

    # Last token from each layer -> [B, H], then stack along a new L-dim -> [B, L, H]
    last_tokens = [h[:, -1, :] for h in hidden_list]          # L × [B, H]
    hs_layers   = torch.stack(last_tokens, dim=1).contiguous() # [B, L, H]

    # Flatten layers into features -> [B, L*H]
    hs = hs_layers.view(hs_layers.size(0), -1).contiguous()

    # Final layer mean over time -> [B, H]
    # h_final = hidden_list[-1]                                  # [B, T, H]
    rep = h_final_input_only.mean(dim=1).contiguous()                     # [B, H]

    return rep, hs


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
    

import torch
import torch.nn.functional as F

def fill_nans(x: torch.Tensor, window: int = 2) -> torch.Tensor:
    if not torch.isnan(x).any():
        return x

    B, T = x.shape
    device = x.device
    dtype = x.dtype

    valid = torch.isfinite(x)

    x0 = torch.where(valid, x, torch.zeros_like(x))

    k = 2 * window + 1
    kernel = torch.ones(1, 1, k, device=device, dtype=dtype)

    # [B, 1, T]
    x0 = x0.unsqueeze(1)
    valid = valid.unsqueeze(1).to(dtype)

    sum_vals = F.conv1d(x0, kernel, padding=window)
    count    = F.conv1d(valid, kernel, padding=window)

    mean = sum_vals / count.clamp_min(1.0)

    out = x.clone()
    nan_mask = torch.isnan(out)

    out[nan_mask] = mean.squeeze(1)[nan_mask]

    # fallback if *all* neighbors are NaN
    out[torch.isnan(out)] = 0.0

    return out



class TimerXL(nn.Module):
    def __init__(self, model_path, device, seq_len, pred_len, **kwargs):
        super(TimerXL, self).__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.model.to(device)

        print(f'>>> Model dtype: {self.model.dtype}; Attention:{self.model.config._attn_implementation}')

        
        self.device = device
        self.pred_len = pred_len
        self.dtype = self.model.dtype
        # self.model#\.eval()
        self.model.config._attn_implementation = "eager"


    def predict(self, batch):
        pred_len = self.pred_len

        
        # print('batch: ', torch.sum(batch))
        # print("input shape: ", batch.shape)
        # print(batch.dtype, batch.min().item(), batch.max().item(), torch.isfinite(batch).all().item())

        outputs = self.model.generate(
            batch.to(self.device).to(self.dtype),
            max_new_tokens=pred_len,
        )
        
        preds = outputs[:, -pred_len:]

        preds = fill_nans(preds, window=2)
        # print('preds: ', torch.sum(preds))
        # labels = batch['labels'].to(device)
        # if len(preds.shape) > len(labels.shape):d
            # labels = labels[..., None]
        return preds

