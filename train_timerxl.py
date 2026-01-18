#!/usr/bin/env python
# -*- coding:utf-8 _*-

import os
import argparse
import logging

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDatasetTrain

logging.basicConfig(level=logging.INFO)


def setup_nccl(rank, world_size, master_addr="127.0.0.1", master_port=9899):
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def unwrap_batch(batch):
    """
    Handles:
        - (x, y)
        - (x, y, *extra)
        - { 'x': ..., 'y': ... }
        - { 'past_target': ..., 'future_target': ... }
        - { 'inputs': ..., 'labels': ... }
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(f"Cannot unwrap batch length {len(batch)}")
        return batch[0], batch[1]

    if isinstance(batch, dict):
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        if "past_target" in batch and "future_target" in batch:
            return batch["past_target"], batch["future_target"]
        if "inputs" in batch and "labels" in batch:
            return batch["inputs"], batch["labels"]
        raise KeyError(f"Unsupported batch keys: {list(batch.keys())}")

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def flatten_channels(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Train on all channels independently (shared weights).

    Input:
      x: [B,T,C] or [B,T] or [T,C] or [T]
      y: [B,H,C] or [B,H] or [H,C] or [H]

    Output:
      x2: [B*C,T] or [B,T]
      y2: [B*C,H] or [B,H]
    """
    # normalize shapes: add batch dim if missing
    if x.ndim == 1:              # [T]
        x = x.unsqueeze(0)       # [1,T]
    elif x.ndim == 2:
        # could be [B,T] or [T,C]; assume [T,C] if second dim small
        if x.shape[1] <= 64 and x.shape[0] > 64:
            x = x.unsqueeze(0)   # [1,T,C]

    if y.ndim == 1:              # [H]
        y = y.unsqueeze(0)       # [1,H]
    elif y.ndim == 2:
        # could be [B,H] or [H,C]; assume [H,C] if second dim small
        if y.shape[1] <= 64 and y.shape[0] > 64:
            y = y.unsqueeze(0)   # [1,H,C]

    if x.ndim == 3:
        B, T, C = x.shape
        x2 = x.transpose(1, 2).contiguous().view(B * C, T)  # [B,C,T] -> [B*C,T]
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

    if y.ndim == 3:
        B, H, C = y.shape
        y2 = y.transpose(1, 2).contiguous().view(B * C, H)  # [B,C,H] -> [B*C,H]
    elif y.ndim == 2:
        y2 = y
    else:
        raise ValueError(f"Unexpected y shape: {tuple(y.shape)}")

    return x2, y2


def trim_to_multiple(x_bt: torch.Tensor, multiple: int) -> torch.Tensor:
    """
    x_bt: [B,T]. Trim T down to nearest multiple of `multiple` (keep most recent).
    """
    if x_bt.ndim != 2:
        raise ValueError(f"Expected [B,T], got {tuple(x_bt.shape)}")
    B, T = x_bt.shape
    T2 = (T // multiple) * multiple
    if T2 < multiple:
        raise ValueError(f"Need at least {multiple} points, got {T}.")
    if T2 != T:
        x_bt = x_bt[:, -T2:]
    return x_bt


class SumEvalMetric:
    def __init__(self, name: str):
        self.name = name
        self.value = 0.0

    def reset(self):
        self.value = 0.0

    def push(self, preds: torch.Tensor, labels: torch.Tensor):
        self.value += float(self._calculate(preds, labels).item())

    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(preds - labels))


@torch.no_grad()
def eval_on_loader(model, loader, device, pred_len: int, patch_len: int):
    model.eval()
    mse_metric = MSEMetric("mse")
    mae_metric = MAEMetric("mae")
    n_points = 0

    for batch in loader:
        bx, by = unwrap_batch(batch)
        bx = bx.to(device=device, dtype=torch.float32)
        by = by.to(device=device, dtype=torch.float32)

        series_in, target = flatten_channels(bx, by)     # [B*C,T], [B*C,H]
        series_in = trim_to_multiple(series_in, patch_len)

        out = model(
            series_in,
            use_cache=False,
            max_output_length=pred_len,
            return_dict=True,
        )
        preds = out.logits  # [B*C, pred_len] for TimerForPrediction

        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if target.shape[1] != pred_len:
            target = target[:, :pred_len]

        mse_metric.push(preds, target)
        mae_metric.push(preds, target)
        n_points += int(target.numel())

    mse = mse_metric.value / max(n_points, 1)
    mae = mae_metric.value / max(n_points, 1)
    return {"mse": mse, "mae": mae}


def evaluate(args):
    # DDP env (optional)
    world_size = int(os.getenv("WORLD_SIZE") or 1)
    rank = int(os.getenv("RANK") or 0)
    local_rank = int(os.getenv("LOCAL_RANK") or 0)

    # Device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Init distributed if requested
    is_dist = False
    if args.use_multi_gpu and torch.cuda.is_available() and world_size > 1:
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.getenv("MASTER_PORT", "9899"))
        setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
        is_dist = dist.is_initialized()
        logging.info(f"DDP initialized: {is_dist}, rank={rank}, world_size={world_size}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        "thuml/timer-base-84m",
        trust_remote_code=True,
    ).to(device)


    # Timer patches along the last dimension; enforce seq_len multiple of patch_len
    patch_len = int(getattr(model.config, "input_token_len", 512))
    logging.info(f"Timer patch_len (config.input_token_len): {patch_len}")

    # Default seq_len if not given
    if args.seq_len is None:
        # common safe default: 30 * patch_len (like 2880 when patch_len=96)
        args.seq_len = 30 * patch_len
    else:
        args.seq_len = (args.seq_len // patch_len) * patch_len
        if args.seq_len < patch_len:
            args.seq_len = patch_len

    logging.info(f"Using seq_len={args.seq_len}, pred_len={args.pred_len}")

    # Dataset
    if not args.data_path.endswith(".csv"):
        raise ValueError("Only CSV data is supported for this training script.")

    dataset_train = BenchmarkEvalDatasetTrain(
        args,
        args.data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        max_train_samples=args.pool_number,
    )


    # Loader
    sampler = DistributedSampler(dataset_train, shuffle=True) if (is_dist and device.type == "cuda") else None
    train_dl = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=max(0, int(args.num_workers)),
        prefetch_factor=2 if args.num_workers and args.num_workers > 0 else None,
        drop_last=False,
    )


    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", disable=(rank != 0))

        for step, batch in enumerate(pbar, start=1):
            bx, by = unwrap_batch(batch)
            bx = bx.to(device=device, dtype=torch.float32)
            by = by.to(device=device, dtype=torch.float32)

            # Train on ALL channels independently: [B,T,C]->[B*C,T], [B,H,C]->[B*C,H]
            series_in, target = flatten_channels(bx, by)
            series_in = trim_to_multiple(series_in, patch_len)

            out = model(
                series_in,
                use_cache=False,
                max_output_length=args.pred_len,
                return_dict=True,
            )
            preds = out.logits

            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            if target.ndim == 1:
                target = target.unsqueeze(0)
            if target.shape[1] != args.pred_len:
                target = target[:, : args.pred_len]

            loss = loss_fn(preds, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            running_loss += float(loss.item())
            if (step % args.log_interval) == 0:
                avg_loss = running_loss / args.log_interval
                running_loss = 0.0
                if rank == 0:
                    pbar.set_postfix(loss=f"{avg_loss:.6f}")

        # Quick train-set metrics
        metrics = eval_on_loader(model, train_dl, device, args.pred_len, patch_len)
        if rank == 0:
            logging.info(f"[Epoch {epoch}] train_MSE={metrics['mse']:.6f}, train_MAE={metrics['mae']:.6f}")

        # Save checkpoint
        if (not is_dist) or dist.get_rank() == 0:
            ckpt_dir = os.path.join(args.output_dir, f"{args.data}/epoch-{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            logging.info(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Timer Finetune (all channels independently)")

    parser.add_argument("--data_path", "-d", type=str, required=True, help="Benchmark CSV path")
    parser.add_argument("--data", type=str, required=True)

    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--seq_len", "-c", type=int, default=None, help="Context length (will be trimmed to patch multiple)")
    parser.add_argument("--pred_len", "-p", type=int, default=96, help="Prediction length")

    # Kept for compatibility with BenchmarkEvalDatasetTrain signature
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--root_path", type=str, default="./data/ETT/")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--dec_in", type=int, required=True)
    parser.add_argument("--c_out", type=int, required=True)

    # training configs
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="(unused for now) Model path",
    )
    parser.add_argument(
        '--label_len',
        type=int,
        default=0,
        help='Label length'
    )

    # misc
    parser.add_argument("--pool_number", type=int, default=10000)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/s223540177/Time-MoE/checkpoints/timer_finetuned_all_channels",
    )
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)

    args = parser.parse_args()
    evaluate(args)
