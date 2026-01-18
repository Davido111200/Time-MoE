#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune TimesFM on a CSV time series dataset using sliding windows.

Key fix for your crashes:
- DO NOT pass `future_values` into the model forward, because your checkpoint predicts a fixed
  horizon (often 128) and will crash if your labels are 96.
- DO NOT force `model.config.horizon_length = 96`, because the checkpoint still produces 128
  internally and then the model reshape breaks.
- Instead: run forward WITHOUT future_values, then slice predictions to your label horizon,
  and compute MSE loss yourself.

Example:
  python finetune.py \
    --csv /path/to/ETTh2.csv \
    --out /path/to/out_dir \
    --checkpoint google/timesfm-2.0-500m-pytorch \
    --context_len 512 --horizon_len 96 --freq 0 --bf16
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import TimesFmModelForPrediction


def _infer_target_col(df: pd.DataFrame, target_col: Optional[str]) -> str:
    if target_col is not None and target_col in df.columns:
        return target_col
    if "OT" in df.columns:
        return "OT"
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found to use as target.")
    return numeric_cols[-1]


def load_csv_as_matrix(
    csv_path: str,
    *,
    timestamp_col: Optional[str] = None,
    target_col: Optional[str] = None,
    multivariate: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      data: float32 array of shape [T, N_series]
      series_names: list length N_series
    """
    df = pd.read_csv(csv_path)

    # Drop timestamp column if present or infer it
    if timestamp_col is not None and timestamp_col in df.columns:
        df = df.drop(columns=[timestamp_col])
    else:
        for cand in ["date", "datetime", "time", "timestamp"]:
            if cand in df.columns and not pd.api.types.is_numeric_dtype(df[cand]):
                df = df.drop(columns=[cand])
                break

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"No numeric columns found in {csv_path}")

    if multivariate:
        use_cols = numeric_cols
    else:
        use_cols = [_infer_target_col(df, target_col)]

    sub = df[use_cols].copy()
    for c in use_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub[c] = sub[c].interpolate(limit_direction="both").ffill().bfill()

    data = sub.to_numpy(dtype=np.float32)  # [T, N]
    return data, use_cols


def time_split(data: np.ndarray, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test


class SlidingWindowMultiSeries(Dataset):
    """
    Treat each column as an independent univariate series.
    """
    def __init__(
        self,
        data: np.ndarray,            # [T, N]
        context_len: int,
        horizon_len: int,
        freq: int,
        stride: int = 1,
    ):
        assert data.ndim == 2
        self.data = data
        self.context_len = int(context_len)
        self.horizon_len = int(horizon_len)
        self.total_len = self.context_len + self.horizon_len
        self.freq = int(freq)
        self.stride = int(stride)

        T, N = data.shape
        max_start = T - self.total_len
        if max_start < 0:
            raise ValueError(f"Series too short: need at least {self.total_len} points, got {T}")

        self.positions_per_series = (max_start // self.stride) + 1
        self.num_series = N

    def __len__(self) -> int:
        return self.num_series * self.positions_per_series

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series_idx = idx // self.positions_per_series
        pos_idx = idx % self.positions_per_series
        start = pos_idx * self.stride

        x = self.data[start : start + self.context_len, series_idx]
        y = self.data[start + self.context_len : start + self.context_len + self.horizon_len, series_idx]

        return {
            "past_values": torch.tensor(x, dtype=torch.float32),     # [context_len]
            "future_values": torch.tensor(y, dtype=torch.float32),   # [horizon_len]
            "freq": torch.tensor(self.freq, dtype=torch.long),       # scalar
        }


def collate_timesfm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # TimesFM supports list[1D] inputs; keep it as list for compatibility.
    past_values = [b["past_values"] for b in batch]  # list length B, each [context_len]
    future_values = torch.stack([b["future_values"] for b in batch], dim=0)  # [B, H_label]
    freq = torch.stack([b["freq"] for b in batch], dim=0).view(-1)  # [B]
    return {"past_values": past_values, "future_values": future_values, "freq": freq}


def _slice_to_label_horizon(preds: torch.Tensor, fut: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    preds expected shape [B, H_model] (most common). Also supports [B, *, H_model] by slicing last dim.
    fut shape [B, H_label]
    Returns preds_sliced, fut (unchanged).
    """
    if fut.ndim != 2:
        raise ValueError(f"Expected fut shape [B, H], got {tuple(fut.shape)}")

    H = fut.shape[1]
    if preds.ndim == 2:
        return preds[:, :H], fut
    if preds.ndim >= 3:
        # slice last dimension
        return preds[..., :H], fut
    raise ValueError(f"Unexpected preds shape: {tuple(preds.shape)}")


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    count = 0

    for batch in loader:
        past = [t.to(device) for t in batch["past_values"]]
        fut = batch["future_values"].to(device)
        freq = batch["freq"].to(device)

        out = model(past_values=past, freq=freq, return_dict=True)
        preds = out.mean_predictions

        preds, fut = _slice_to_label_horizon(preds, fut)

        err = preds - fut
        se_sum += float((err * err).sum().item())
        ae_sum += float(err.abs().sum().item())
        count += int(fut.numel())

    denom = max(count, 1)
    mse = se_sum / denom
    mae = ae_sum / denom
    return {"loss": mse, "mse": mse, "mae": mae}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True, help="Path to dataset CSV (e.g., ETTh2.csv)")
    ap.add_argument("--out", type=str, required=True, help="Output directory to save fine-tuned model")
    ap.add_argument("--checkpoint", type=str, default="google/timesfm-2.0-500m-pytorch")

    ap.add_argument("--target_col", type=str, default=None)
    ap.add_argument("--timestamp_col", type=str, default=None)
    ap.add_argument("--multivariate", action="store_true")

    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--horizon_len", type=int, default=96, help="Label forecast length.")
    ap.add_argument("--pred_len", type=int, default=None, help="Alias for horizon_len.")
    ap.add_argument("--freq", type=int, default=0, help="TimesFM frequency bucket (0/1/2).")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--train_head_only", action="store_true")
    ap.add_argument("--num_workers", type=int, default=1)

    args = ap.parse_args()
    if args.pred_len is not None:
        args.horizon_len = args.pred_len

    os.makedirs(args.out, exist_ok=True)

    data, series_names = load_csv_as_matrix(
        args.csv,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        multivariate=args.multivariate,
    )
    train_data, val_data, test_data = time_split(data, args.train_ratio, args.val_ratio)

    train_ds = SlidingWindowMultiSeries(train_data, args.context_len, args.horizon_len, args.freq, stride=args.stride)
    val_ds = SlidingWindowMultiSeries(val_data, args.context_len, args.horizon_len, args.freq, stride=args.stride)
    test_ds = SlidingWindowMultiSeries(test_data, args.context_len, args.horizon_len, args.freq, stride=args.stride)

    prefetch = 2 if args.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=prefetch,
        collate_fn=collate_timesfm, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, prefetch_factor=prefetch,
        collate_fn=collate_timesfm, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, prefetch_factor=prefetch,
        collate_fn=collate_timesfm, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one: --fp16 or --bf16")

    if args.fp16 and device.type == "cuda":
        dtype = torch.float16
    elif args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = TimesFmModelForPrediction.from_pretrained(
        args.checkpoint,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)

    model_h = getattr(model.config, "horizon_length", None)
    patch_len = getattr(model.config, "patch_length", None)

    print("Device:", device)
    print("Label horizon_len:", args.horizon_len)
    print("Model config horizon_length:", model_h)
    print("Model patch_length:", patch_len)

    if model_h is not None and args.horizon_len > int(model_h):
        raise ValueError(
            f"horizon_len={args.horizon_len} is larger than model horizon_length={model_h}. "
            f"Set --horizon_len {model_h} or use a different checkpoint."
        )

    if args.train_head_only:
        for n, p in model.named_parameters():
            p.requires_grad = any(k in n.lower() for k in ["head", "prediction", "quantile", "proj", "lm_head"])
        if not any(p.requires_grad for p in model.parameters()):
            for p in model.parameters():
                p.requires_grad = True

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = (device.type == "cuda") and (dtype in (torch.float16, torch.bfloat16))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    def autocast_ctx():
        if device.type != "cuda":
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=dtype, enabled=use_amp)

    best_val = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        optim.zero_grad(set_to_none=True)

        for it, batch in enumerate(pbar, start=1):
            past = [t.to(device) for t in batch["past_values"]]
            fut = batch["future_values"].to(device)
            freq = batch["freq"].to(device)

            with autocast_ctx():
                # IMPORTANT: no future_values passed, we compute loss ourselves
                out = model(past_values=past, freq=freq, return_dict=True)
                preds = out.mean_predictions
                preds, fut = _slice_to_label_horizon(preds, fut)
                loss = F.mse_loss(preds, fut, reduction="mean") / max(args.grad_accum, 1)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it % args.grad_accum) == 0:
                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)

            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=(running / max(steps, 1)))

        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        print(
            f"[epoch {epoch}] val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} val_mse={val_metrics['mse']:.6f} | "
            f"test_mae={test_metrics['mae']:.6f} test_mse={test_metrics['mse']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            model.save_pretrained(args.out)
            with open(os.path.join(args.out, "series_columns.txt"), "w", encoding="utf-8") as f:
                for s in series_names:
                    f.write(s + "\n")
            with open(os.path.join(args.out, "train_args.txt"), "w", encoding="utf-8") as f:
                f.write(str(vars(args)))

    print(f"Done. Best val_loss={best_val:.6f}. Model saved to: {args.out}")


if __name__ == "__main__":
    main()
