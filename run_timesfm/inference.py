#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate TimesFM using the SAME data loading path as your Time-MoE evaluator:
  - BenchmarkEvalDataset / GeneralEvalDataset
  - DataLoader + (optional) DistributedSampler
  - NCCL init from env vars

This file supports 2 backends:
  1) backend="timesfm": uses the google-research timesfm package (2.5 style torch wrapper)
  2) backend="transformers": uses Hugging Face TimesFmModelForPrediction (works even if timesfm 2.5 wrapper is missing)

Important assumption:
- batch["inputs"] and batch["labels"] are real-valued time series tensors.
If they are token ids (dtype long, small vocab-like values), TimesFM cannot be used unless you de-tokenize.

Example:
  torchrun --nproc_per_node=1 eval_timesfm.py \
    --backend transformers \
    --model google/timesfm-2.0-500m-pytorch \
    --data ETTh1 \
    --data_path /scratch/.../ETTh1.csv \
    --seq_len 512 \
    --pred_len 96 \
    --batch_size 16
"""

import os
import argparse
import logging
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

import sys
sys.path.append("/home/s223540177/Time-MoE")
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset

logging.basicConfig(level=logging.INFO)


def setup_nccl(rank, world_size, master_addr="127.0.0.1", master_port=9899):
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def is_token_like(x: torch.Tensor) -> bool:
    # Heuristic: integer dtype + relatively small max value
    if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
        return False
    mx = int(x.max().item()) if x.numel() > 0 else 0
    return mx <= 65536


def make_dataloader(args, dataset):
    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
    else:
        sampler = None

    dl = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )
    return dl


def load_dataset(args):
    if args.data_path.endswith(".csv"):
        return BenchmarkEvalDataset(args.data_path, seq_len=args.seq_len, pred_len=args.pred_len)
    return GeneralEvalDataset(args.data_path, seq_len=args.seq_len, pred_len=args.pred_len)


def ddp_env() -> Tuple[int, int, int]:
    world_size = int(os.getenv("WORLD_SIZE") or 1)
    rank = int(os.getenv("RANK") or 0)
    local_rank = int(os.getenv("LOCAL_RANK") or 0)
    return world_size, rank, local_rank


def pick_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


@torch.no_grad()
def forecast_timesfm_backend(
    model,
    inputs: torch.Tensor,   # [B, seq_len]
    pred_len: int,
) -> np.ndarray:
    """
    timesfm backend expects list of 1D numpy arrays (float32).
    Returns point forecast as numpy array [B, pred_len].
    """
    x_np = inputs.detach().cpu().float().numpy()
    batch_inputs = [x_np[i] for i in range(x_np.shape[0])]
    point_fcst, _ = model.forecast(horizon=pred_len, inputs=batch_inputs)
    point_fcst = np.asarray(point_fcst, dtype=np.float32)
    return point_fcst


@torch.no_grad()
def forecast_transformers_backend(
    model,
    inputs: torch.Tensor,   # [B, seq_len]
    pred_len: int,
    freq_bucket: int,
    device: torch.device,
) -> torch.Tensor:
    """
    transformers backend expects list[1D tensors] + freq [B].
    Returns point forecast tensor [B, pred_len] on device.
    """
    from transformers import TimesFmModelForPrediction

    if not isinstance(model, TimesFmModelForPrediction):
        raise TypeError("Expected TimesFmModelForPrediction for transformers backend")

    past_list = [inputs[i].to(device).float() for i in range(inputs.shape[0])]
    freq = torch.full((inputs.shape[0],), int(freq_bucket), dtype=torch.long, device=device)

    out = model(past_values=past_list, freq=freq, return_dict=True)
    preds = out.mean_predictions  # typically [B, H_model]

    # Slice to pred_len if needed
    if preds.ndim == 2 and preds.shape[1] != pred_len:
        preds = preds[:, :pred_len]
    return preds


def build_timesfm_model(args, device: torch.device):
    """
    Build timesfm model (google-research timesfm) if available.
    If your timesfm package does not include the 2.5 torch wrapper, this will raise with a clear message.
    """
    import timesfm

    # Find a torch wrapper class in the module
    candidates = [n for n in dir(timesfm) if n.startswith("TimesFM_") and n.endswith("_torch")]
    if not candidates:
        raise RuntimeError(
            "Your imported 'timesfm' module has no TimesFM_*_torch classes.\n"
            "This usually means you installed an older PyPI timesfm or you are shadowing the package.\n"
            "Fix (recommended):\n"
            "  pip uninstall -y timesfm\n"
            "  pip install \"timesfm[torch] @ git+https://github.com/google-research/timesfm.git\"\n"
            "Or use --backend transformers instead."
        )

    # Prefer 2p5 if present
    prefer = [c for c in candidates if "2p5" in c.lower()]
    cls_name = (prefer[0] if prefer else candidates[0])
    cls = getattr(timesfm, cls_name)

    # Load and compile
    try:
        model = cls.from_pretrained(args.model, torch_compile=args.torch_compile)
    except TypeError:
        model = cls.from_pretrained(args.model)

    # Compile config: must cover your seq_len and pred_len
    max_context = max(args.max_context, args.seq_len)
    max_horizon = max(args.max_horizon, args.pred_len)

    # Be defensive: ForecastConfig fields can vary slightly by version
    cfg_kwargs = dict(
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=args.normalize_inputs,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=args.infer_is_positive,
        fix_quantile_crossing=True,
    )
    fields = getattr(timesfm.ForecastConfig, "__dataclass_fields__", None)
    if fields:
        cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in fields}
    cfg = timesfm.ForecastConfig(**cfg_kwargs)
    model.compile(cfg)

    return model, cls_name


def build_transformers_model(args, device: torch.device):
    import timesfm

    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model).to(device).to(dtype)

    model.eval()
    return model


def evaluate(args):
    world_size, rank, local_rank = ddp_env()

    # DDP init
    is_dist = False
    device = pick_device(local_rank)
    if device.type == "cuda":
        try:
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
            master_port = int(os.getenv("MASTER_PORT", "9899"))
            setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
            is_dist = True
        except Exception as e:
            logging.warning(f"NCCL init failed, falling back to single-process eval: {e}")
            is_dist = False

    # Dataset + loader
    dataset = load_dataset(args)
    test_dl = make_dataloader(args, dataset)

    # Model
    if args.backend == "timesfm":
        model, cls_name = build_timesfm_model(args, device)
        if rank == 0:
            logging.info(f"Using timesfm backend class: {cls_name}")
    else:
        model = build_transformers_model(args, device)
        if rank == 0:
            logging.info("Using transformers backend TimesFmModelForPrediction")

    # Metric sums (sum over all elements)
    se_sum = 0.0
    ae_sum = 0.0
    count = 0

    pbar = tqdm(test_dl, disable=(rank != 0))
    for idx, batch in enumerate(pbar):
        inputs = batch["inputs"]
        labels = batch["labels"]

        # Basic validation
        if is_token_like(inputs) or is_token_like(labels):
            raise RuntimeError(
                "Your dataset appears tokenized (integer ids). TimesFM expects real-valued time series.\n"
                "You need to modify BenchmarkEvalDataset/GeneralEvalDataset to output float values, not tokens."
            )

        # Shape handling
        if inputs.ndim == 3 and inputs.shape[-1] == 1:
            inputs = inputs.squeeze(-1)
        if labels.ndim == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)

        if inputs.ndim != 2 or labels.ndim != 2:
            raise ValueError(f"Expected inputs/labels to be 2D: got {tuple(inputs.shape)} and {tuple(labels.shape)}")

        # Ensure correct lengths
        if inputs.shape[1] != args.seq_len:
            # Some datasets may return variable length; slice last seq_len
            inputs = inputs[:, -args.seq_len:]
        if labels.shape[1] != args.pred_len:
            labels = labels[:, : args.pred_len]

        if args.backend == "timesfm":
            # timesfm forecast returns numpy on CPU
            point_fcst = forecast_timesfm_backend(model, inputs, args.pred_len)  # [B, pred_len]
            preds = torch.from_numpy(point_fcst).to(device)
        else:
            preds = forecast_transformers_backend(model, inputs, args.pred_len, args.freq, device)

        y = labels.to(device).float()
        preds = preds.to(device).float()

        # Match horizon if backend returns longer
        H = min(preds.shape[1], y.shape[1])
        preds = preds[:, :H]
        y = y[:, :H]

        err = preds - y
        se_sum += float((err * err).sum().item())
        ae_sum += float(err.abs().sum().item())
        count += int(y.numel())

        if rank == 0 and (idx % args.log_every == 0):
            mse_step = float((err * err).mean().item())
            pbar.set_postfix(mse=mse_step)

    # Reduce across ranks
    if is_dist and dist.is_initialized():
        stats = torch.tensor([se_sum, ae_sum, float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        se_sum, ae_sum, count_f = stats.tolist()
        count = int(count_f)

    mse = se_sum / max(count, 1)
    mae = ae_sum / max(count, 1)

    if rank == 0:
        item = {
            "backend": args.backend,
            "model": args.model,
            "data": args.data,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "mse": float(mse),
            "mae": float(mae),
        }
        logging.info(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TimesFM Evaluate (Time-MoE-style data loader)")
    parser.add_argument("--backend", type=str, default="transformers", choices=["timesfm", "transformers"])
    parser.add_argument("--model", "-m", type=str, default="google/timesfm-2.0-500m-pytorch", help="Model id or path")
    parser.add_argument("--data_path", "-d", type=str, required=True, help="Benchmark data path")
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--seq_len", "-c", type=int, default=None)
    parser.add_argument("--pred_len", "-p", type=int, default=96)

    # transformers backend only
    parser.add_argument("--freq", type=int, default=0, help="TimesFM frequency bucket for transformers backend")

    # timesfm backend only
    parser.add_argument("--max_context", type=int, default=1024)
    parser.add_argument("--max_horizon", type=int, default=256)
    parser.add_argument("--normalize_inputs", action="store_true", help="timesfm compile option")
    parser.add_argument("--infer_is_positive", action="store_true", help="timesfm compile option")
    parser.add_argument("--torch_compile", action="store_true", help="timesfm from_pretrained option (if supported)")

    # runtime
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", help="transformers backend dtype")
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()

    if args.seq_len is None:
        if args.pred_len in (96, 192, 336, 720):
            args.seq_len = 512
        else:
            args.seq_len = args.pred_len * 4

    evaluate(args)
