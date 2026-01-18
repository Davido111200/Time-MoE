import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from chronos.chronos2.model import Chronos2Model
import matplotlib.pyplot as plt
import types

from scipy.signal import welch as _welch
import math 
import random
from typing import Dict, List, Any, Tuple, Union

from typing import Dict, List, Tuple, Any, Optional


# each item is: (sample_id, rep, hs, raw_series)
Item = Tuple[Any, Any, Any, Any]
DemoTuple = Tuple[Any, Any, Any, Any]  # (idx, rep, hidden_states, raw_seq)

Tensor = torch.Tensor
Demo = Tuple[int, Tensor, Tensor, Tensor]  # (idx, rep, hidden_states, raw_sequence)

def cap_each_id(
    gt: Dict[Any, List[Item]],
    preds: Dict[Any, List[Item]],
    max_per: int = 100,
    seed: int = 0,
    prefer_order: str = "gt",   # "gt" | "preds" — whose order to respect when aligning
):
    """
    For every id (top-level key), keep at most `max_per` tuples in GT and Preds.
    - If id exists in both: align by sample_id (tuple[0]), sample from the intersection,
      and keep in the chosen source order (gt or preds), filtered to sampled ids.
    - If id exists only in one: sample up to `max_per` from that side (preserving order);
      the other side remains as-is (empty if missing).
    - Returns new_gt, new_preds with the same key structure (union of input keys).
    """
    rng = random.Random(seed)
    all_ids = list(set(gt.keys()) | set(preds.keys()))
    new_gt, new_preds = {}, {}

    for key in all_ids:
        gt_list = gt.get(key, [])
        pr_list = preds.get(key, [])

        # helpers
        def _to_id_map(lst: List[Item]):
            return { itm[0]: itm for itm in lst }  # assumes unique sample_id

        def _ordered_ids(lst: List[Item]):
            return [itm[0] for itm in lst]

        # both sides present → align by sample_id
        if gt_list and pr_list:
            gt_map, pr_map = _to_id_map(gt_list), _to_id_map(pr_list)
            # choose an ordering source to preserve relative order
            base_order = _ordered_ids(gt_list if prefer_order == "gt" else pr_list)
            common_ids_in_order = [sid for sid in base_order if sid in gt_map and sid in pr_map]

            if len(common_ids_in_order) <= max_per:
                keep_ids = set(common_ids_in_order)
            else:
                # random subset, but keep original order afterwards
                keep_ids = set(rng.sample(common_ids_in_order, max_per))

            kept_ids_ordered = [sid for sid in common_ids_in_order if sid in keep_ids]
            new_gt[key]    = [gt_map[sid] for sid in kept_ids_ordered]
            new_preds[key] = [pr_map[sid] for sid in kept_ids_ordered]

        # only GT present
        elif gt_list and not pr_list:
            if len(gt_list) <= max_per:
                keep_idx = list(range(len(gt_list)))
            else:
                # sample indices, then sort to preserve original order
                idx = rng.sample(range(len(gt_list)), max_per)
                keep_idx = sorted(idx)
            new_gt[key] = [gt_list[i] for i in keep_idx]
            new_preds[key] = preds.get(key, [])  # empty if missing

        # only Preds present
        elif pr_list and not gt_list:
            if len(pr_list) <= max_per:
                keep_idx = list(range(len(pr_list)))
            else:
                idx = rng.sample(range(len(pr_list)), max_per)
                keep_idx = sorted(idx)
            new_preds[key] = [pr_list[i] for i in keep_idx]
            new_gt[key] = gt.get(key, [])  # empty if missing

        else:
            # neither side has entries for this key
            new_gt[key], new_preds[key] = [], []

    return new_gt, new_preds



def visualize_data(
    args,
    pred,
    list_of_examples,
    align="end",
    feature_index=-1,   # which feature/column to plot if data is >1D (use -1 for last column)
):
    out_path=f"/home/s223540177/ICL4TS/img/{args.retrieval}_{args.num_closest_samples}_visualization.png"
    def to1d(x):
        # unwrap torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)

        # If already 1D, done
        if x.ndim == 1:
            return x.astype(float, copy=False)

        # Reduce to a single feature: flatten all leading axes, then pick one column
        # Works for [L, D], [B, L, D], [*, D], [L, 1], etc.
        x2d = x.reshape(-1, x.shape[-1])         # [N, D]
        col = x2d[:, feature_index]              # pick feature
        return col.astype(float, copy=False)

    pred = to1d(pred)
    examples = [to1d(e) for e in list_of_examples]

    # longest length for padding/align
    Lmax = max([len(pred)] + [len(e) for e in examples])

    def pad(a):
        pad_len = Lmax - len(a)
        if pad_len <= 0:
            return a
        pad_block = np.full(pad_len, np.nan)
        return np.concatenate([pad_block, a]) if align == "end" else np.concatenate([a, pad_block])

    x_axis = np.arange(Lmax)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, pad(pred), label=f"Prediction (L={len(pred)})", linewidth=2)
    for i, e in enumerate(examples):
        plt.plot(x_axis, pad(e), "--", label=f"Example {i+1} (L={len(e)})")
    plt.title("Prediction vs Retrieved Examples")
    plt.xlabel(f"Time step (aligned by {align})")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v



class ICVLayer(nn.Module):
    def __init__(self, icv, lam, dtype):
        super(ICVLayer, self).__init__()
        self.icv = icv
        self.lam = lam
        self.dtype = dtype

    def forward(self, x):
        if self.icv is not None:
            x = x.float()
            original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            y = 0
            for i in range(len(self.icv)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x, -self.icv[i][None,None,:], dim=-1)).unsqueeze(-1)
                y += self.lam[i] * lambda_sim * F.normalize(self.icv[i], dim=-1).repeat(1,x.shape[1],1)
            y = y/len(self.icv)
            x = F.normalize(F.normalize(x.float(), p=2, dim=-1) + y, p=2, dim=-1) * original_norm
            x = x.to(self.dtype)
            return x
        else:
            return x



class ICVBetaTail(nn.Module):
    """
    One tail that unifies:
      - Plain (your ICVLayer):   y_plain = mean_i( lam[i]*(1+ReLU(cos(x,-d_i)) ) * norm(d_i) )
                                 out_plain = norm( norm(x) + y_plain ) * ||x||
      - Confidence (your ICVLayerConfidence): legacy gate on cos(x,-d), schedule, stretch/keep

    beta in [0,1]: 0 -> plain; 1 -> confidence; interpolate otherwise.
    icv: [H] or [K,H]
    lam: float | list[float] | 1D tensor [K]
    """

    def __init__(
        self,
        icv: torch.Tensor,
        lam,
        dtype=torch.float32,
        *,
        beta: float = 0.5,              # 0=plain, 1=confidence
        # confidence-mode knobs (match your ICVLayerConfidence)
        baseline: float = 0.25,
        margin: float = 0.10,
        power: float = 1.25,
        renorm_conf: str = "stretch",   # "keep" | "stretch"
        stretch: float = 0.10,
        max_frac: float | None = None,
        schedule: dict | None = None,   # {'tau','warmup','cutoff','scale'}
        global_gain: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert 0.0 <= beta <= 1.0, "beta must be in [0,1]"
        self.beta = float(beta)
        self.dtype = dtype
        self.eps = float(eps)

        # icv buffer -> [K,H]
        if not isinstance(icv, torch.Tensor):
            raise ValueError("icv must be a torch.Tensor")
        if icv.dim() == 1:
            icv = icv.unsqueeze(0)
        assert icv.dim() == 2, f"icv must be [K,H] or [H], got {tuple(icv.shape)}"
        self.register_buffer("icv", icv)

        # lam buffer -> [K]
        if isinstance(lam, (list, tuple)):
            lam = torch.tensor(lam, dtype=torch.float32)
        elif not isinstance(lam, torch.Tensor):
            lam = torch.tensor([float(lam)], dtype=torch.float32)
        if lam.dim() == 0:
            lam = lam.view(1)
        self.register_buffer("lam", lam)

        # confidence params
        self.baseline = float(baseline)
        self.margin   = float(margin)
        self.power    = float(power)
        self.renorm_conf = renorm_conf
        self.stretch  = float(stretch)
        self.max_frac = max_frac
        self.schedule = schedule or {}
        self.register_buffer("step_ctr", torch.zeros((), dtype=torch.long))
        self.global_gain = None if global_gain is None else float(global_gain)

    def reset(self):
        self.step_ctr.zero_()

    @staticmethod
    def _ensure_bth(x: torch.Tensor):
        if x.dim() == 2:
            B, H = x.shape
            return x.unsqueeze(1), True, (B, 1, H)
        elif x.dim() == 3:
            return x, False, tuple(x.shape)
        else:
            raise ValueError(f"x must be [B,H] or [B,T,H], got {tuple(x.shape)}")

    def _sched_gain_scalar(self, step: int) -> float:
        tau    = int(self.schedule.get('tau', 0) or 0)
        warmup = int(self.schedule.get('warmup', 0) or 0)
        cutoff = int(self.schedule.get('cutoff', 10**9))
        scale  = float(self.schedule.get('scale', 1.0))

        if step >= cutoff:
            return 0.0
        if step < warmup:
            base = (step + 1) / max(1, warmup)   # linear warmup
        else:
            base = 1.0 if tau <= 0 else math.exp(-(step - warmup) / float(tau))
        return scale * base

    def forward(self, x: torch.Tensor, gain_map: torch.Tensor | None = None):
        if self.icv is None or self.icv.numel() == 0:
            print("ICV is None or empty, returning input as-is.")
            return x

        x32 = x.float()
        xb, squeezed, (B, T, H) = self._ensure_bth(x32)
        orig_norm = xb.norm(dim=-1, keepdim=True) + self.eps

        # normalized directions [K,H]
        dir = self.icv.float()
        dir = dir / (dir.norm(dim=-1, keepdim=True) + self.eps)
        K = dir.shape[0]

        # cosines
        xk   = xb.unsqueeze(0).expand(K, -1, -1, -1)        # [K,B,T,H]
        dirk = dir.view(K, 1, 1, H)
        cos  = F.cosine_similarity(xk, dirk.expand(-1, B, T, -1), dim=-1)  # [K,B,T]
        cos_neg = -cos  # cos(x, -dir)

        # ===== Plain branch: EXACT math as your ICVLayer =====
        # y_plain = mean_i( lam[i] * (1 + ReLU(cos_neg_i)) * dir_i )
        lam_plain = self.lam.to(xb.device, dtype=torch.float32).view(K, 1, 1)
        lambda_sim = 1.0 + F.relu(cos_neg)                              # [K,B,T]
        y_plain = (lam_plain * lambda_sim).unsqueeze(-1) * dirk.expand(-1, B, T, -1)  # [K,B,T,H]
        y_plain = y_plain.mean(dim=0)                                   # [B,T,H]

        # out_plain = normalize( normalize(x) + y_plain ) * ||x||
        x_unit = F.normalize(xb, dim=-1)
        out_plain = F.normalize(x_unit + y_plain, dim=-1) * orig_norm   # [B,T,H]

        if self.beta <= 0.0:
            out = out_plain
            return out.squeeze(1).to(self.dtype) if squeezed else out.to(self.dtype)

        # ===== Confidence branch: EXACT ICVLayerConfidence =====
        gate_raw = F.relu(cos_neg + self.margin)                        # [K,B,T]
        gate = self.baseline + gate_raw.pow(self.power)                 # [K,B,T]

        w = self.lam.to(xb.device, dtype=torch.float32).view(K)
        if K == 1:
            global_gain = float(w.item())
            w = torch.ones_like(w)
        else:
            global_gain = 1.0 if self.global_gain is None else float(self.global_gain)
            w = w / (w.sum() + self.eps)

        gate = gate * w.view(K, 1, 1)                                   # [K,B,T]
        delta_conf = (gate.unsqueeze(-1) * dirk.expand(-1, B, T, -1)).sum(dim=0)  # [B,T,H]

        if self.schedule:
            # 1 -> 0 linearly across the current T positions, single call only
            scale = float(self.schedule.get("scale", 1.0))
            if T == 1:
                sched_map = torch.ones(1, T, 1, device=xb.device, dtype=xb.dtype) * scale
            else:
                sched_map = torch.linspace(1.0, 0.0, steps=T, device=xb.device, dtype=xb.dtype).view(1, T, 1)
                sched_map = scale * sched_map
            delta_conf = delta_conf * sched_map
            # Do NOT advance any counters; no step/gen state for this mode.
        else:
            delta_conf = delta_conf 

        if gain_map is not None:
            delta_conf = delta_conf * (gain_map if gain_map.dim() else gain_map.to(delta_conf))
        delta_conf = delta_conf * global_gain

        # optional clamp
        if self.max_frac is not None:
            max_delta = self.max_frac * orig_norm
            delta_conf = torch.clamp(delta_conf, min=-max_delta, max=max_delta)

        out_conf = xb + delta_conf
        if self.renorm_conf == "keep":
            out_conf = F.normalize(out_conf, dim=-1) * orig_norm
        elif self.renorm_conf == "stretch":
            gmean = gate.mean(dim=0, keepdim=False).unsqueeze(-1)  # [B,T,1]
            factor = (1.0 + self.stretch * gmean).clamp(min=0.0)
            out_conf = out_conf * factor

        if self.beta >= 1.0:
            out = out_conf
            return out.squeeze(1).to(self.dtype) if squeezed else out.to(self.dtype)

        # ===== Interpolate outputs =====
        out = (1.0 - self.beta) * out_plain + self.beta * out_conf
        return out.squeeze(1).to(self.dtype) if squeezed else out.to(self.dtype)



class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


def _mmr_select(sim_q, V, k, lam=0.7, candidate_idx=None):
    """
    MMR: score = lam * sim(query, i) - (1-lam) * max_{j in S} cos(i, j)
    sim_q is higher=better. Redundancy uses cosine on V.
    """
    if candidate_idx is None:
        candidate_idx = np.arange(V.shape[0])
    cand = list(candidate_idx)
    selected = []

    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    while cand and len(selected) < k:
        if not selected:
            i_local = int(np.argmax(sim_q[cand]))
            selected.append(cand.pop(i_local))
            continue
        S = np.array(selected, dtype=int)
        # redundancy = max cosine sim to any already-selected item
        red = (Vn[cand] @ Vn[S].T).max(axis=1)
        score = lam * sim_q[cand] - (1.0 - lam) * red
        i_local = int(np.argmax(score))
        selected.append(cand.pop(i_local))

    return selected

def select_mmr(sim_q, V, k, lam=0.7, shortlist_factor=8, best_last=True):
    """
    Two-stage: shortlist top-M by similarity, then MMR for diversity.
    If best_last=True, move the most similar of the selected to the end.
    """
    M = min(V.shape[0], max(k, k * shortlist_factor))
    short = np.argpartition(sim_q, -M)[-M:]
    selected = _mmr_select(sim_q, V, k=k, lam=lam, candidate_idx=short)

    if best_last and len(selected) > 1:
        sims = sim_q[selected]
        best_pos = int(np.argmax(sims))
        ordered = [selected[i] for i in range(len(selected)) if i != best_pos] + [selected[best_pos]]
        return ordered
    return selected

def stack_vector_db(vector_db):
    """
    vector_db: {idx(int) -> rep (np.ndarray[d] or torch.Tensor[d])}
    Returns:
      V: (N, d) float32 matrix
      id2row: list of original train indices aligned with rows of V
    """
    id2row = sorted(vector_db.keys())
    # infer dim and type from first item
    first = vector_db[id2row[0]]
    if hasattr(first, "detach"):  # torch.Tensor
        first = first.detach().cpu().numpy()
    d = int(first.shape[0])

    V = np.empty((len(id2row), d), dtype=np.float32)
    for r, idx in enumerate(id2row):
        v = vector_db[idx]
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        V[r] = v.astype(np.float32, copy=False)
    return V, id2row

def stack_dist_db(vector_db):
    """
    vector_db: {idx(int) -> dist (np.ndarray[h] or torch.Tensor[h])}
    Returns:
      D: (N, H) float32 matrix  (H = max length across dists; shorter rows are padded)
      id2row: list of original train indices aligned with rows of D
    """
    eps = 1e-8
    id2row = sorted(vector_db.keys())

    # 1) First pass: convert to 1-D numpy arrays, record lengths
    dists_list = []
    max_len = 0
    for idx in id2row:
        dist = vector_db[idx]
        if isinstance(dist, torch.Tensor):
            dist = dist.detach().cpu().numpy()
        dist = np.asarray(dist)
        dist = np.squeeze(dist)  # handles (1,V), (V,1), (1,)
        if dist.ndim == 2:
            # pick a 1-D view if one dim is 1; else flatten
            if dist.shape[0] == 1:
                dist = dist[0]
            elif dist.shape[1] == 1:
                dist = dist[:, 0]
            else:
                dist = dist.reshape(-1)
        elif dist.ndim != 1:
            dist = dist.reshape(-1)

        dist = dist.astype(np.float32, copy=False)
        dists_list.append(dist)
        if dist.shape[0] > max_len:
            max_len = dist.shape[0]

    # 2) Allocate and fill with epsilon padding; renormalize each row
    N = len(id2row)
    D = np.full((N, max_len), eps, dtype=np.float32)

    for r, vec in enumerate(dists_list):
        L = vec.shape[0]
        if L >= max_len:
            row = vec[:max_len]
        else:
            row = D[r]
            row[:L] = vec
        # renormalize to sum=1 (avoids issues in KL/JS)
        s = row.sum()
        if s > 0:
            D[r] = row / s
        else:
            D[r] = row  # all eps already

    return D, id2row

def stack_vector_db_with_dist(vector_db):
    """
    vector_db: {idx(int) -> (rep (np.ndarray[d] or torch.Tensor[d]), dist (np.ndarray[h] or torch.Tensor[h]))}
    Returns:
      V: (N, d) float32 matrix
      D: (N, h) float32 matrix
      id2row: list of original train indices aligned with rows of V and D
    """
    id2row = sorted(vector_db.keys())
    # infer dim and type from first item
    first_rep, first_dist = vector_db[id2row[0]]
    if hasattr(first_rep, "detach"):  # torch.Tensor
        first_rep = first_rep.detach().cpu().numpy()
    if hasattr(first_dist, "detach"):  # torch.Tensor
        first_dist = first_dist.detach().cpu().numpy()
    d = int(first_rep.shape[0])
    h = int(first_dist.shape[0])

    V = np.empty((len(id2row), d), dtype=np.float32)
    D = np.empty((len(id2row), h), dtype=np.float32)
    for r, idx in enumerate(id2row):
        v, dist = vector_db[idx]
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        if hasattr(dist, "detach"):
            dist = dist.detach().cpu().numpy()
        V[r] = v.astype(np.float32, copy=False)
        D[r] = dist.astype(np.float32, copy=False)
    return V, D, id2row

def kl_divergence_batch(p, Q, eps=1e-8):
    """
    KL(P || Q_i) for each row Q_i in Q.
    p: (V,)
    Q: (N, V)
    Returns: (N,)
    """
    p = p.astype(np.float32, copy=False)
    Q = Q.astype(np.float32, copy=False)

    # normalize & clip for numerical stability
    p = p / (p.sum() + eps)
    Q = Q / (Q.sum(axis=1, keepdims=True) + eps)

    p = np.clip(p, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)

    logp = np.log(p)
    logQ = np.log(Q)
    # KL(P||Q) = sum_j p_j [log p_j - log q_j]
    return (p * (logp - logQ)).sum(axis=1)

def select_topk_smallest(values, k):
    """Return indices of k smallest values, sorted ascending by value."""
    k = int(min(k, len(values)))
    part = np.argpartition(values, k-1)[:k]
    return part[np.argsort(values[part])].tolist()


def obtain_icv(hidden_states, rank=1, weights=None):
    # hidden_states[i][1] is used as in your original code
    num_demonstration = len(hidden_states)
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][2].view(-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all)  # [N, D]

    # --- weighted direction (no edge-case handling) ---
    w = torch.as_tensor(weights, device=fit_data.device, dtype=fit_data.dtype)
    w = w / w.sum()  # normalize to sum=1
    direction_vec = (w[:, None] * fit_data).sum(dim=0)  # [D]
    direction = direction_vec.view(hidden_states[0][2].shape)

    return direction

def obtain_icv_new(gts, preds, hidden_shape, rank=1, weights=None):
    # gts[i][1] is used as in your original code
    num_demonstration = len(gts)
    neg_all = []
    pos_all = []

    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = gts[demonstration_id][2].view(-1) - preds[demonstration_id][2].view(-1)
        hidden_states_all.append(h)
        neg_all.append(preds[demonstration_id][2].view(-1))
        pos_all.append(gts[demonstration_id][2].view(-1))
    fit_data = torch.stack(hidden_states_all)

    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=0,keepdim=True) + pca.mean_).mean(0).view(preds[demonstration_id][2].size(0), preds[demonstration_id][2].size(1))

    L = preds[0][2].shape[1] // hidden_shape

    direction = direction.view(L, hidden_shape)

    return direction


@torch.no_grad()
def obtain_icv_unified(
    args, hidden_tf_list, hidden_fr_list,
    errors_list=None, eps: float = 1e-6,
    whiten: bool = True, normalize_per_layer: bool = False,
    hidden_size: int | None = None,
):
    """
    One ICV builder that works for ETTh1 and ETTh2:
    - If [L,T,H], builds BOTH (a) discounted/time-weighted delta and (b) last-step delta, then blends them 50/50.
    - If [L,H], uses the direct delta.
    - Whitening is directional only (layer energy restored).
    Returns V: [L,H].
    """
    assert len(hidden_tf_list) == len(hidden_fr_list) and len(hidden_fr_list) > 0
    h0 = hidden_fr_list[0][2]
    device = h0.device

    # infer L,H
    if h0.dim() == 2:
        L, H = h0.shape
        if L == 1 and hidden_size is not None and (h0.shape[1] % hidden_size == 0):
            L = h0.shape[1] // hidden_size; H = hidden_size
    elif h0.dim() == 3:
        L, _, H = h0.shape
    else:
        raise ValueError(f"hidden_states must be [L,H] or [L,T,H], got {tuple(h0.shape)}")
    if hidden_size is not None:
        assert H == hidden_size

    def _collapse_to_LH(h, err):
        if h.dim() == 2:
            return h.to(device=device, dtype=torch.float32).view(L, H)

    def _last_to_LH(h):
        if h.dim() == 2:
            return h.to(device=device, dtype=torch.float32).view(L, H)
        return h.to(device=device, dtype=torch.float32)[:, -1, :]  # [L,H]

    # accumulators
    Vc = torch.zeros(L, H, device=device)   # collapsed
    Vl = torch.zeros(L, H, device=device)   # last-step
    if whiten:
        fr_m = torch.zeros(L, H, device=device)
        fr_e2 = torch.zeros(L, H, device=device)

    for i in range(len(hidden_fr_list)):
        h_tf = hidden_tf_list[i][2]
        h_fr = hidden_fr_list[i][2]
        err  = (errors_list[i] if (errors_list is not None and i < len(errors_list)) else None)

        tf_c = _collapse_to_LH(h_tf, err)
        fr_c = _collapse_to_LH(h_fr, err)

        tf_l = _last_to_LH(h_tf)
        fr_l = _last_to_LH(h_fr)

        Vc += (tf_c - fr_c)
        Vl += (tf_l - fr_l)

        if whiten:
            fr_m  += fr_c
            fr_e2 += fr_c * fr_c

    # blend (fixed 50/50 works well across ETTh1/ETTh2)
    V = args.collapse_weight * Vc + (1 - args.collapse_weight) * Vl

    # sign stabilize against pre-whiten mean
    V_avg = V.clone() / max(1, len(hidden_fr_list))

    # preserve per-layer energy through whitening
    pre_E = V.view(L, -1).norm(dim=1, keepdim=True) + eps
    if whiten:
        fr_mean = fr_m / float(len(hidden_fr_list))
        fr_var  = torch.clamp(fr_e2 / float(len(hidden_fr_list)) - fr_mean * fr_mean, min=0.0)
        V = V / torch.sqrt(fr_var + eps)              # directional whitening
        post_E = V.view(L, -1).norm(dim=1, keepdim=True) + eps
        V = V * (pre_E / post_E)                      # restore energy

    dots = (V * V_avg).view(L, -1).sum(dim=1)
    flip = dots < 0
    if flip.any():
        V[flip] = -V[flip]

    if normalize_per_layer:
        V = V / (V.norm(dim=1, keepdim=True) + eps)

    return V


# @torch.no_grad()
# def obtain_icv_interpolate(
#     args,
#     hidden_tf_list: List[DemoTuple],
#     hidden_fr_list: List[DemoTuple],
#     errors_list: Optional[List[torch.Tensor]] = None,
#     *,
#     beta: float = 0.0,                 # 0 -> plain; 1 -> confidence; blend otherwise
#     collapse_weight: float = 0.5,      # blend collapsed vs last-step
#     eps: float = 1e-6,
#     # whitening
#     whiten: bool = True,
#     # plain branch knobs (mirror your 'unified')
#     plain_energy_restore: bool = False,
#     plain_normalize_per_layer: bool = False,
#     # confidence branch knobs
#     conf_normalize_per_layer: bool = True,   # direction-only for confidence tail
#     hidden_size: Optional[int] = None,       # sanity guard
# ) -> torch.Tensor:
#     """
#     Returns V: [L,H].
#       - If beta == 0: EXACT plain behavior (matches obtain_icv_unified when knobs match).
#       - If beta == 1: Confidence-oriented vector (direction-only; scale handled by tail).
#       - Else: convex blend (1-beta)*V_plain + beta*V_conf after independent sign-stabilization.
#     """
#     # print(len(hidden_tf_list), len(hidden_fr_list))
#     assert len(hidden_tf_list) == len(hidden_fr_list) and len(hidden_fr_list) > 0
#     assert 0.0 <= beta <= 1.0
#     h0 = hidden_fr_list[0][2]
#     device = h0.device
#     dtype = torch.float32


#     if h0.dim() == 2:
#         L, H = h0.shape
#         # packed case [1, L*H]
#         if L == 1 and hidden_size is not None and (H % hidden_size == 0):
#             L = H // hidden_size; H = hidden_size
#     elif h0.dim() == 3:
#         L, _, H = h0.shape
#     else:
#         raise ValueError(f"hidden must be [L,H] or [L,T,H], got {tuple(h0.shape)}")
#     if hidden_size is not None:
#         assert H == hidden_size, f"H={H} != hidden_size={hidden_size}"

#     def _collapse_to_LH(h: torch.Tensor, err: Optional[torch.Tensor]) -> torch.Tensor:
#         if h.dim() == 2:
#             return h.to(device=device, dtype=dtype).view(L, H)
#         else:
#             raise ValueError

#     def _last_to_LH(h: torch.Tensor) -> torch.Tensor:
#         if h.dim() == 2:
#             return h.to(device=device, dtype=dtype).view(L, H)
#         else:
#             raise ValueError

#     # accumulate per-branch raw deltas
#     Vc = torch.zeros(L, H, device=device, dtype=dtype)
#     Vl = torch.zeros(L, H, device=device, dtype=dtype)
#     if whiten:
#         fr_m  = torch.zeros(L, H, device=device, dtype=dtype)
#         fr_e2 = torch.zeros(L, H, device=device, dtype=dtype)

#     N = len(hidden_fr_list)
#     for i in range(N):
#         h_tf = hidden_tf_list[i][2]
#         h_fr = hidden_fr_list[i][2]
#         err  = None
#         if errors_list is not None and i < len(errors_list) and errors_list[i] is not None:
#             err = errors_list[i]

#         tf_c = _collapse_to_LH(h_tf, err)   # [L,H]
#         fr_c = _collapse_to_LH(h_fr, err)   # [L,H]
#         tf_l = _last_to_LH(h_tf)            # [L,H]
#         fr_l = _last_to_LH(h_fr)            # [L,H]

#         Vc += (tf_c - fr_c)
#         Vl += (tf_l - fr_l)

#         if whiten:
#             fr_m  += fr_c
#             fr_e2 += fr_c * fr_c

#     # common blend pre-whitening
#     cw = float(collapse_weight)
#     V_raw = cw * Vc + (1.0 - cw) * Vl            # [L,H]
#     V_avg = V_raw / max(1, N)                    # for sign stabilization

#     # whitening stats
#     if whiten:
#         fr_mean = fr_m / float(N)
#         fr_var  = torch.clamp(fr_e2 / float(N) - fr_mean * fr_mean, min=0.0)

#     # plain branch
#     V_plain = V_raw.clone()
#     if whiten:
#         pre_E = V_plain.view(L, -1).norm(dim=1, keepdim=True) + eps
#         V_plain = V_plain / torch.sqrt(fr_var + eps)          # directional whitening
#         if plain_energy_restore:
#             post_E = V_plain.view(L, -1).norm(dim=1, keepdim=True) + eps
#             V_plain = V_plain * (pre_E / post_E)              # restore layer energy
#     # sign stabilization
#     dots = (V_plain * V_avg).view(L, -1).sum(dim=1)
#     flip = dots < 0
#     if flip.any():
#         V_plain[flip] = -V_plain[flip]
#     if plain_normalize_per_layer:
#         V_plain = V_plain / (V_plain.norm(dim=1, keepdim=True) + eps)

#     if beta == 0.0:
#         return V_plain  # EXACT plain endpoint

#     # ===== Confidence branch =====
#     V_conf = V_raw.clone()
#     if whiten:
#         V_conf = V_conf / torch.sqrt(fr_var + eps)            # directional whitening
#         # no energy restore here; confidence tail handles gain externally
#     # sign stabilization (independent of plain)
#     dots_c = (V_conf * V_avg).view(L, -1).sum(dim=1)
#     flip_c = dots_c < 0
#     if flip_c.any():
#         V_conf[flip_c] = -V_conf[flip_c]
#     if conf_normalize_per_layer:
#         V_conf = V_conf / (V_conf.norm(dim=1, keepdim=True) + eps)

#     if beta == 1.0:
#         return V_conf  # EXACT confidence endpoint

#     # ===== Interpolate =====
#     V_mix = (1.0 - beta) * V_plain + beta * V_conf

#     if hidden_size is not None:
#         assert V_mix.shape == (L, hidden_size), f"V has shape {tuple(V_mix.shape)}, expected [L,{hidden_size}]"
#     return V_mix


@torch.no_grad()
def obtain_icv_interpolate(
    args,
    hidden_tf_list: List[DemoTuple],
    hidden_fr_list: List[DemoTuple],
    errors_list: Optional[List[torch.Tensor]] = None,
    *,
    beta: float = 0.0,
    collapse_weight: float = 0.5,
    eps: float = 1e-6,
    whiten: bool = True,
    plain_energy_restore: bool = False,
    plain_normalize_per_layer: bool = False,
    conf_normalize_per_layer: bool = True,
    hidden_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Robustly computes an interpolation control vector V of shape [L, hidden_size],
    even when hidden-state tensors are packed or flattened in various ways.
    """

    assert len(hidden_tf_list) == len(hidden_fr_list) and len(hidden_fr_list) > 0, f"Mismatched or empty hidden-state lists. Found {len(hidden_tf_list)} TF and {len(hidden_fr_list)} FR."
    assert 0.0 <= beta <= 1.0

    h0 = hidden_fr_list[0][2]  
    h0_flat = h0.view(-1)
    assert hidden_size is not None, "hidden_size must be provided."
    total_elems = h0_flat.numel()
    assert total_elems % hidden_size == 0, (
        f"Packed hidden length {total_elems} is not divisible by hidden_size={hidden_size}"
    )
    L = total_elems // hidden_size  # total number of transformer layers
    H = hidden_size

    # Helper to reshape any hidden tensor into [L, H], padding or cropping rows as needed.
    def _to_LH(h: torch.Tensor) -> torch.Tensor:
        h = h.to(dtype=torch.float32)
        flat = h.view(-1)
        assert flat.numel() % H == 0, (
            f"Hidden tensor with {flat.numel()} elements cannot be reshaped with H={H}"
        )
        Li = flat.numel() // H
        flat_reshaped = flat.view(Li, H)
        if Li == L:
            return flat_reshaped
        elif Li < L:
            pad = torch.zeros(L - Li, H, device=flat_reshaped.device, dtype=flat_reshaped.dtype)
            return torch.cat([flat_reshaped, pad], dim=0)
        else:
            return flat_reshaped[:L]

    device = h0.device  # use the same device as the incoming tensors

    # Initialize accumulators
    Vc = torch.zeros(L, H, device=device)
    Vl = torch.zeros(L, H, device=device)
    if whiten:
        fr_m  = torch.zeros(L, H, device=device)
        fr_e2 = torch.zeros(L, H, device=device)

    N = len(hidden_fr_list)
    for i in range(N):
        h_tf = hidden_tf_list[i][2]
        h_fr = hidden_fr_list[i][2]

        # Reinterpret each hidden-state tensor as [L, H]
        tf_c = _to_LH(h_tf)  # collapsed view (identical to last-token view here)
        fr_c = _to_LH(h_fr)
        tf_l = _to_LH(h_tf)  # last-token view
        fr_l = _to_LH(h_fr)

        Vc += (tf_c - fr_c)
        Vl += (tf_l - fr_l)

        if whiten:
            fr_m  += fr_c
            fr_e2 += fr_c * fr_c

    # Blend collapsed vs last-token contributions
    cw = float(collapse_weight)
    V_raw = cw * Vc + (1.0 - cw) * Vl
    V_avg = V_raw / max(1, N)  # for sign stabilization

    if whiten:
        fr_mean = fr_m / float(N)
        fr_var  = torch.clamp(fr_e2 / float(N) - fr_mean * fr_mean, min=0.0)

    # Plain branch
    V_plain = V_raw.clone()
    if whiten:
        pre_E = V_plain.view(L, -1).norm(dim=1, keepdim=True) + eps
        V_plain = V_plain / torch.sqrt(fr_var + eps)
        if plain_energy_restore:
            post_E = V_plain.view(L, -1).norm(dim=1, keepdim=True) + eps
            V_plain = V_plain * (pre_E / post_E)
    dots = (V_plain * V_avg).view(L, -1).sum(dim=1)
    flip = dots < 0
    if flip.any():
        V_plain[flip] = -V_plain[flip]
    if plain_normalize_per_layer:
        V_plain = V_plain / (V_plain.norm(dim=1, keepdim=True) + eps)


    if beta == 0.0:
        return V_plain

    # Confidence branch
    V_conf = V_raw.clone()
    if whiten:
        V_conf = V_conf / torch.sqrt(fr_var + eps)
    dots_c = (V_conf * V_avg).view(L, -1).sum(dim=1)
    flip_c = dots_c < 0
    if flip_c.any():
        V_conf[flip_c] = -V_conf[flip_c]
    if conf_normalize_per_layer:
        V_conf = V_conf / (V_conf.norm(dim=1, keepdim=True) + eps)

    if beta == 1.0:
        return V_conf

    # Interpolate the two branches
    V_mix = (1.0 - beta) * V_plain + beta * V_conf
    return V_mix



@torch.no_grad()
def obtain_icv_dtf(
    hidden_tf_list: List[DemoTuple],      # each: (idx, rep, h_tf, raw_seq_tf)
    hidden_fr_list: List[DemoTuple],      # each: (idx, rep, h_fr, raw_seq_fr)
    eps: float = 1e-6,
    whiten: bool = True,                  # whitening from collapsed free-run variance
    normalize_per_layer: bool = True,     # L2-normalize each layer’s vector
    hidden_size: Optional[int] = 384,  # if known, can be used to verify H
):
    """
    Returns a steering vector V with shape [L, H].

    Inputs (slot 2 of each tuple) may be:
      - [L, H]         (already collapsed)
      - [L, T, H]      (collapsed to [L, H] with discount+optional error weights)
    """

    assert len(hidden_tf_list) == len(hidden_fr_list), "Mismatch in number of demos"
    N = len(hidden_fr_list)
    assert N > 0, "Empty input lists"

    # infer device / L,H from the first free-run item
    h0 = hidden_fr_list[0][2]
    device = h0.device

    # --- helpers ---
    def _collapse_to_LH(h: torch.Tensor, err: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Ensure shape [L, H]. If h is [L, T, H], form a normalized weight per time step:
            w_t ∝ gamma^(T-1-t) * normed_error_t    (if error provided)
        and compute: sum_t w_t * h[:, t, :]

        err can be:
          - None
          - [T] or [T, C]  (the latter is averaged over C)
        """
        if h.dim() == 2:
            # already [L, H]
            return h.to(device=device, dtype=torch.float32)

    # probe L,H
    h_fr0 = hidden_fr_list[0][2]
    H_probe = h_fr0.shape[-1]
    if h_fr0.dim() == 2:
        L = h_fr0.shape[0]
        H = h_fr0.shape[1]
    elif h_fr0.dim() == 3:
        L = h_fr0.shape[0]
        H = h_fr0.shape[2]
    else:
        raise ValueError(f"hidden_states must be [L,H] or [L,T,H], got {tuple(h_fr0.shape)}")
    assert H == H_probe, "Hidden size mismatch"

    # Accumulators over demos (already [L,H] after collapse)
    V = torch.zeros(L, H, device=device)          # sum of deltas per layer
    avg_delta = torch.zeros(L, H, device=device)  # for sign stabilization
    denom_for_avg = 0.0

    if whiten:
        fr_first_moment  = torch.zeros(L, H, device=device)
        fr_second_moment = torch.zeros(L, H, device=device)

    for i in range(N):
        h_tf_raw = hidden_tf_list[i][2]
        h_fr_raw = hidden_fr_list[i][2]

        err_i = None
        # collapse each to [L,H]
        h_tf = _collapse_to_LH(h_tf_raw, err_i)  # [L,H]
        h_fr = _collapse_to_LH(h_fr_raw, err_i)  # [L,H]

        # delta in [L,H]
        delta = h_tf - h_fr

        # accumulate
        V += delta
        avg_delta += delta
        denom_for_avg += 1.0

        if whiten:
            fr_first_moment  += h_fr
            fr_second_moment += h_fr.pow(2)

    # average delta for sign stabilization
    if denom_for_avg > 0:
        avg_delta = avg_delta / denom_for_avg  # [L,H]

    # whitening from collapsed free-run stats (per-dimension)
    if whiten:
        fr_mean = fr_first_moment / max(N, 1)
        fr_e2   = fr_second_moment / max(N, 1)
        fr_var  = torch.clamp(fr_e2 - fr_mean.pow(2), min=0.0)  # [L,H]
        V = V / torch.sqrt(fr_var + eps)

    # sign stabilization per layer
    dots = (V * avg_delta).view(L, -1).sum(dim=1)  # [L]
    flip_mask = dots < 0
    if flip_mask.any():
        V[flip_mask] = -V[flip_mask]

    # per-layer normalization
    if normalize_per_layer:
        norms = V.norm(dim=1, keepdim=True) + eps  # [L,1]
        V = V / norms

    L = V.shape[1] // hidden_size 

    if V.shape[1] == hidden_size:
        pass
    else:
        V = V.view(L, hidden_size)

    return V  # [L,H]


@torch.no_grad()
def obtain_icv_dtf_test(
    hidden_tf_list: List,    # [(idx, rep, h_tf, raw_seq_tf), ...] for teacher-forcing mode
    hidden_fr_list: List,    # [(idx, rep, h_fr, raw_seq_fr), ...] for free-running mode
    errors_list: Optional[List] = None, 
    eps: float = 1e-6,
    whiten: bool = True, 
    normalize_per_layer: bool = False,  # set False to preserve actual magnitudes
    hidden_size: Optional[int] = None   # if known, for assertion
):
    """
    Compute a steering vector V of shape [L, H] (L layers, H hidden size) that, when added to the model's
    hidden states, nudges them from free-running behavior toward teacher-forced behavior.
    """
    assert len(hidden_tf_list) == len(hidden_fr_list), "Mismatch in number of demos"
    N = len(hidden_fr_list)
    assert N > 0, "Empty input lists."
    # Determine device and dimensions from the first free-run sample
    h0 = hidden_fr_list[0][2]  # hidden states tensor from first free-run sample
    device = h0.device
    if h0.dim() == 2:
        # hidden states already collapsed: shape [L, H]
        L, H = h0.shape
        if L == 1:
            L = h0.shape[1] // hidden_size 
            H = hidden_size
            h0 = h0.view(L, H)
    elif h0.dim() == 3:
        # hidden states with time dimension: shape [L, T, H]
        L, T, H = h0.shape
    else:
        raise ValueError(f"Expected hidden_states shape [L,H] or [L,T,H], got {tuple(h0.shape)}")
    if hidden_size is not None:
        assert H == hidden_size, f"Hidden size mismatch: expected {hidden_size}, got {H}"
    # Helper to collapse [L, T, H] to [L, H] with discounting and optional error weighting
    def _collapse_to_LH(h: torch.Tensor, err: Optional[torch.Tensor]) -> torch.Tensor:
        if h.dim() == 2:
            # Already [L, H]
            h = h.view(L, H)
            return h.to(device=device, dtype=torch.float32)
        
    # Accumulators for differences and (optionally) free-run stats
    V_sum = torch.zeros(L, H, device=device)     # sum of deltas (teacher_forced - free_run) per layer
    fr_mean_acc = torch.zeros(L, H, device=device) if whiten else None
    fr_var_acc  = torch.zeros(L, H, device=device) if whiten else None
    # Compute layer-wise differences for each demo sequence
    for i in range(N):
        h_tf = hidden_tf_list[i][2]
        h_fr = hidden_fr_list[i][2]
        err_i = errors_list[i] if (errors_list is not None and i < len(errors_list)) else None
        # Collapse to [L, H]
        h_tf_LH = _collapse_to_LH(h_tf, err_i)
        h_fr_LH = _collapse_to_LH(h_fr, err_i)
        # Accumulate difference
        V_sum += (h_tf_LH - h_fr_LH)
        # Accumulate free-run stats for whitening
        if whiten:
            fr_mean_acc += h_fr_LH
            fr_var_acc  += h_fr_LH ** 2
    # Average difference (for sign stabilization reference)
    V_avg = V_sum / float(N)
    V = V_sum.clone()
    # Whitening: normalize differences by free-run variance per dimension
    if whiten:
        fr_mean = fr_mean_acc / float(N)
        fr_mean_sq = fr_var_acc / float(N)
        fr_var = fr_mean_sq - fr_mean ** 2
        fr_var = torch.clamp(fr_var, min=eps)            # avoid zero variance
        V = V / torch.sqrt(fr_var + eps)                 # scale by inverse std-dev
    # Sign stabilization: ensure each layer's vector points in the same general direction as the average delta
    dot_sign = (V * V_avg).sum(dim=1)                    # dot product per layer
    flip_mask = (dot_sign < 0)                           # if negative, flip the vector
    V[flip_mask] *= -1
    # Optional per-layer normalization (disabled by default)
    if normalize_per_layer:
        layer_norms = V.norm(dim=1, keepdim=True) + eps
        V = V / layer_norms
    return V  # shape [L, H]


# @torch.no_grad()
# def obtain_icv_dtf_test(
#     hidden_tf_list: List,  # [(idx, rep, h_tf, raw_seq_tf), ...]
#     hidden_fr_list: List,  # [(idx, rep, h_fr, raw_seq_fr), ...]
#     errors_list: Optional[List] = None,
#     gamma: float = 0.99,
#     eps: float = 1e-6,
#     use_median: bool = False,
#     whiten: bool = False,
#     whiten_source: str = "fr",  # "fr" or "tf"
#     scale: float = 1.0,
#     hidden_size: Optional[int] = None
# ):
#     """
#     Compute an intervention vector for steering TSFM hidden activations.

#     Compared to the original implementation, this version:
#       - optionally disables whitening or uses ridge‑regularised whitening;
#       - supports median or robust mean of differences across demos;
#       - exposes a global scaling factor to control the intervention strength;
#       - allows dynamic scaling based on current activations (e.g. error).
#     """
#     assert len(hidden_tf_list) == len(hidden_fr_list) > 0
#     N = len(hidden_fr_list)
#     h0 = hidden_fr_list[0][2]
#     device = h0.device
#     # Determine layer and hidden dimensions
#     if h0.dim() == 2:
#         L, H = h0.shape
#         if L == 1 and hidden_size is not None:
#             L = h0.shape[1] // hidden_size
#             H = hidden_size
#             h0 = h0.view(L, H)
#     elif h0.dim() == 3:
#         L, T, H = h0.shape
#     else:
#         raise ValueError("Unexpected hidden state shape")
#     # Collapse function
#     def collapse(h: torch.Tensor, err: Optional[torch.Tensor]) -> torch.Tensor:
#         if h.dim() == 2:
#             return h.view(L, H)
#         L_l, T_l, H_l = h.shape
#         weights = torch.ones(T_l, device=device)
#         # Discount
#         if gamma < 1.0:
#             t_idx = torch.arange(T_l - 1, -1, -1, device=device, dtype=torch.float32)
#             weights = (gamma ** t_idx)
#         # Error weighting
#         if err is not None:
#             e = err if err.dim() == 1 else err.abs().mean(dim=1)
#             e = e[:T_l]
#             e = e / (e.mean() + eps) if e.numel() > 0 else 1.0
#             weights = weights[:T_l] * e
#         weights = weights / (weights.sum() + eps)
#         return (h[:, :T_l, :] * weights.view(1, T_l, 1)).sum(dim=1)
#     # Collect per‑demo differences
#     diffs = []
#     fr_acts = []
#     tf_acts = []
#     for i in range(N):
#         h_tf = collapse(hidden_tf_list[i][2], errors_list[i] if errors_list else None)
#         h_fr = collapse(hidden_fr_list[i][2], errors_list[i] if errors_list else None)
#         diffs.append(h_tf - h_fr)
#         fr_acts.append(h_fr)
#         tf_acts.append(h_tf)
#     # Aggregate the difference (mean or median)
#     diffs_stack = torch.stack(diffs)
#     if use_median:
#         V = torch.median(diffs_stack, dim=0).values
#     else:
#         V = diffs_stack.mean(dim=0)
#     # Optional whitening (with ridge regularisation)
#     if whiten:
#         acts = fr_acts if whiten_source == "fr" else tf_acts
#         acts_stack = torch.stack(acts)
#         mean = acts_stack.mean(dim=0)
#         centered = acts_stack - mean
#         # Estimate covariance with ridge to avoid singularity
#         cov = (centered.reshape(-1, H).T @ centered.reshape(-1, H)) / (L * N - 1)
#         ridge = eps * torch.eye(H, device=device)
#         inv_std = torch.linalg.inv(torch.linalg.cholesky(cov + ridge))
#         # Whiten each layer independently
#         V = (inv_std @ V.T).T
#     # Sign stabilisation
#     V_avg = V.mean(dim=0)
#     signs = torch.sign((V * V_avg).sum(dim=1))
#     V = V * signs.unsqueeze(-1)
#     # Apply global scaling
#     V = V * scale
#     return V


def get_chronos2_encoder_mlp_layers(model):
    """
    Return the list of (name, mlp_module) for the 12 encoder FFN MLPs.

    Works if you pass either:
      - Chronos2Pipeline, or
      - Chronos2Model directly.
    """
    # Unwrap pipeline -> underlying HF model
    if hasattr(model, "model") and hasattr(model, "forecast_type"):
        # Chronos2Pipeline
        model = model.model

    # At this point `model` should be Chronos2Model
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "block"):
        raise ValueError("Expected Chronos2Model with .encoder.block, got: "
                         f"{type(model)}")

    layers = []
    for b_idx, block in enumerate(encoder.block):
        # Each block has: layer[0]=TimeSelfAttention, [1]=GroupSelfAttention, [2]=FeedForward
        ff = block.layer[2]
        mlp = ff.mlp
        name = f"encoder.block.{b_idx}.layer.2.mlp"
        layers.append((name, mlp))

    return layers

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def get_layers_path(model):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    if isinstance(block, tuple):
        _, block = block

    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")

def find_modulelist(root, keywords, *, exact=False):
    """
    Find a nn.ModuleList under `root` whose name matches any keyword.

    Works for:
      find_modulelist(m.model, ["layers"])  -> ModuleList(...)
      find_modulelist(m, ["layers"])        -> ModuleList(...)
    """
    if isinstance(root, tuple):
        _, root = root
    if not isinstance(root, nn.Module):
        raise TypeError(f"`root` must be nn.Module (or (idx, nn.Module)), got {type(root)}")

    def match(name: str) -> bool:
        if exact:
            return any(name == k for k in keywords)
        return any(k in name for k in keywords)

    # 1) Fast path: check direct attributes (common for ModuleList fields)
    for attr_name, attr_val in vars(root).items():
        if isinstance(attr_val, nn.ModuleList) and match(attr_name):
            return attr_val

    # 2) General path: search the whole tree, but only accept ModuleList
    for name, module in root.named_modules():
        if isinstance(module, nn.ModuleList) and match(name):
            return module

    # Helpful error
    available = [name for name, module in root.named_modules() if isinstance(module, nn.ModuleList)]
    raise ValueError(
        f"Could not find ModuleList matching {keywords}. "
        f"Available ModuleLists: {available}"
    )

# NOTE: this works previously
def add_icv_layers(model, icv, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["ffn_layer"]
    assert len(icv) == len(layers)

    for i, layer in enumerate(layers):
        ffn = find_module(layer, mlp_keywords)           # TimeMoeSparseExpertsLayer

        # match device/dtype with FFN params (to avoid cuda/cpu mismatch)
        p = next(ffn.parameters())
        dev, dt = p.device, p.dtype

        # register ICV inside the ffn layer
        ffn.icv_tail = ICVLayer(icv[i].to(device=dev, dtype=dt), alpha, dtype=dt).to(device=dev, dtype=dt)
        # monkey-patch forward: original returns (hidden_states, router_logits)
        ffn._orig_forward = ffn.forward
        def _forward_with_icv(self, *args, **kwargs):
            y, router_logits = self._orig_forward(*args, **kwargs)
            y = self.icv_tail(y)                         # apply ICV to hidden states only
            return y, router_logits
        ffn.forward = types.MethodType(_forward_with_icv, ffn)


def add_icv_layers_new(model, icv, alpha: list):
    layers = get_layers(model)
    mlp_keywords = ["ffn_layer"]
    assert len(icv) == len(layers)

    for i, layer in enumerate(layers):
        # optional: skip if this layer's vector is (near) zero
        icv_i = icv[i]
        if isinstance(icv_i, torch.Tensor) and icv_i.abs().max().item() < 1e-8:
            continue

        ffn = find_module(layer, mlp_keywords)  # TimeMoeSparseExpertsLayer
        p = next(ffn.parameters())
        dev, dt = p.device, p.dtype

        # NOTE: pass per-ICV weights in 'alpha' (e.g., [1.0]) — layer scaling should
        # be baked into the icv[i] tensor beforehand (mask/taper outside).
        ffn.icv_tail = ICVLayer(icv_i.to(device=dev, dtype=dt), alpha, dtype=dt).to(device=dev, dtype=dt)

        if not hasattr(ffn, "_orig_forward"):
            ffn._orig_forward = ffn.forward

        def _forward_with_icv(self, *args, **kwargs):
            y, router_logits = self._orig_forward(*args, **kwargs)  # y: [B,T,H]
            if hasattr(self, "icv_tail") and self.icv_tail is not None:
                y = self.icv_tail(y)
            return y, router_logits

        ffn.forward = types.MethodType(_forward_with_icv, ffn)



def remove_icv_layers(model):
    layers = get_layers(model)
    mlp_keywords = ["ffn_layer"]
    for i, layer in enumerate(layers):
        ffn = find_module(layer, mlp_keywords)           # TimeMoeSparseExpertsLayer
        ffn.forward = ffn._orig_forward
        del ffn._orig_forward
        del ffn.icv_tail


def retrieve_examples(args, rep, vector_db, pool_number, topk, query_series=None):
    """
    Minimal modification for list-backed DB:
      - vector_db is a list of (orig_idx, rep_tensor, hs, series_tensor[1,T])
      - Returns:
          dists   : np.array of distances aligned to the FIRST `pool_number` candidates
                    (in the same order as the selected slice of vector_db)
          samples : list of selected original indices (orig_idx) length = topk
    Other notes unchanged from your original version.
    """
    import numpy as np

    # ---------- tiny helpers ----------
    def _to_np(x):
        # works for torch.Tensor or np.ndarray / list
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=float)

    def _series1d(x):
        # series can be torch [1,T] or [T]; make it 1D np
        s = _to_np(x)
        return s.reshape(-1)

    def _znorm(x, eps=1e-8):
        x = np.asarray(x, float)
        std = x.std()
        return (x - x.mean()) / (std + eps)

    def _pair_last(q, r, tail_n=None):
        q = np.asarray(q, float); r = np.asarray(r, float)
        if q.size == 0 or r.size == 0:
            return q[:0], r[:0]
        if tail_n is None:
            L = min(q.size, r.size)
        else:
            L = min(int(tail_n), q.size, r.size)
        return q[-L:], r[-L:]

    def _dtw(a, b, window=None):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 1e9
        if window is None: window = max(n, m)
        window = max(window, abs(n - m))
        INF = 1e18
        D = np.full((n+1, m+1), INF, dtype=float); D[0,0] = 0.0
        for i in range(1, n+1):
            j0, j1 = max(1, i-window), min(m, i+window)
            ai = a[i-1]
            for j in range(j0, j1+1):
                cost = (ai - b[j-1])**2
                D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return float(D[n,m]**0.5)

    def _xcorr_dist(a, b, max_lag=None):
        a, b = _znorm(a), _znorm(b)
        L = min(len(a), len(b))
        a, b = a[:L], b[:L]
        if L == 0:
            return 1.0
        if max_lag is None: max_lag = L - 1
        max_lag = int(max(0, max_lag))
        best = -1.0
        for lag in range(-max_lag, max_lag+1):
            if lag >= 0:
                aa, bb = a[lag:], b[:L-lag]
            else:
                aa, bb = a[:L+lag], b[-lag:]
            if len(aa) == 0:
                continue
            denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-8
            c = float(np.dot(aa, bb)) / denom
            if c > best: best = c
        return 1.0 - best

    def _fft_feat(x, k=8):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.zeros(k, dtype=float)
        X = np.fft.rfft(x)
        mag = np.abs(X)[1:k+1]
        nrm = np.linalg.norm(mag)
        return (mag / nrm) if nrm > 0 else mag

    def _cosine_dist(u, v, eps=1e-8):
        u = np.asarray(u, float); v = np.asarray(v, float)
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0: return 1.0
        return 1.0 - float(np.dot(u, v) / (nu * nv + eps))

    def _pearson_dist(a, b, eps=1e-8):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size == 0 or b.size == 0:
            return 1.0
        a = a - a.mean(); b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
        r = float(np.dot(a, b) / denom)
        return 1.0 - abs(r)

    def _zscore_safe(x):
        x = np.asarray(x, float)
        finite = np.isfinite(x)
        if not np.any(finite):
            return np.zeros_like(x)
        mu = np.nanmean(x[finite]); sd = np.nanstd(x[finite])
        sd = 1.0 if (not np.isfinite(sd) or sd == 0) else sd
        z = (x - mu) / sd
        z[~finite] = np.max(z[finite]) + 5.0
        return z

    # ---------- config / candidates ----------
    if vector_db is None or len(vector_db) == 0:
        return np.array([]), []

    tail_n = getattr(args, "tail_n", None)
    mode   = getattr(args, "retrieval", "euclidean").lower()

    # Use the first pool_number candidates from this channel's list
    pool_n = min(int(pool_number), len(vector_db))
    idxs   = list(range(pool_n))  # indices into vector_db
    # Original IDs for alignment/return
    orig_ids = [vector_db[j][0] for j in idxs]

    # Query representations
    rep_np = _to_np(rep)
    q_series_np = None if query_series is None else _series1d(query_series)

    # Shortcuts to pull fields
    def _rep_of(j):
        return _to_np(vector_db[j][1])
    def _raw_of(j):
        return _series1d(vector_db[j][3])

    # ---------- distances by mode ----------
    if mode == "euclidean":
        d = [np.linalg.norm(rep_np - _rep_of(j)) for j in idxs]
    elif mode == "cosine":
        d = [_cosine_dist(rep_np, _rep_of(j)) for j in idxs]
    elif mode == "z_euclidean":
        if q_series_np is None:
            raise ValueError("z_euclidean requires query_series (raw 1D query).")
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            d.append(np.linalg.norm(_znorm(qt) - _znorm(rt)))
    elif mode == "pearson":
        if q_series_np is None:
            raise ValueError("pearson requires query_series (raw 1D query).")
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            d.append(_pearson_dist(qt, rt))
    elif mode == "dtw":
        if q_series_np is None:
            raise ValueError("dtw requires query_series (raw 1D query).")
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            d.append(_dtw(qt, rt))
    elif mode == "xcorr":
        if q_series_np is None:
            raise ValueError("xcorr requires query_series (raw 1D query).")
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            max_lag = getattr(args, "max_lag", None)
            if max_lag is None:
                max_lag = max(0, min(len(qt), len(rt)) - 1)
            d.append(_xcorr_dist(qt, rt, max_lag=int(max_lag)))
    elif mode == "fft":
        if q_series_np is None:
            raise ValueError("fft requires query_series (raw 1D query).")
        k_bins = int(getattr(args, "fft_k", 8))
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            d.append(np.linalg.norm(_fft_feat(qt, k=k_bins) - _fft_feat(rt, k=k_bins)))
    elif mode == "hybrid":
        alpha = float(getattr(args, "alpha", 0.5))
        sec   = getattr(args, "hybrid_sec", "dtw").lower()
        if sec not in {"dtw", "xcorr", "fft", "z_euclidean", "pearson"}:
            sec = "dtw"

        rep_d = [np.linalg.norm(rep_np - _rep_of(j)) for j in idxs]

        if q_series_np is None:
            sec_d = [_cosine_dist(rep_np, _rep_of(j)) for j in idxs]
        else:
            vals = []
            for j in idxs:
                r = _raw_of(j)
                qt, rt = _pair_last(q_series_np, r, tail_n)
                if sec == "dtw":
                    vals.append(_dtw(qt, rt))
                elif sec == "xcorr":
                    max_lag = getattr(args, "max_lag", None)
                    if max_lag is None:
                        max_lag = max(0, min(len(qt), len(rt)) - 1)
                    vals.append(_xcorr_dist(qt, rt, max_lag=int(max_lag)))
                elif sec == "fft":
                    k_bins = int(getattr(args, "fft_k", 8))
                    vals.append(np.linalg.norm(_fft_feat(qt, k=k_bins) - _fft_feat(rt, k=k_bins)))
                elif sec == "z_euclidean":
                    vals.append(np.linalg.norm(_znorm(qt) - _znorm(rt)))
                else:
                    vals.append(_pearson_dist(qt, rt))
            sec_d = vals

        rep_z = _zscore_safe(rep_d)
        sec_z = _zscore_safe(sec_d)
        d = alpha * rep_z + (1.0 - alpha) * sec_z
    else:
        raise ValueError(f"Unknown retrieval mode: {args.retrieval}")

    # ---------- selection / return ----------
    dists = np.asarray(d, dtype=float)
    k = min(int(topk), len(dists))
    order = np.argsort(dists)[:k]


    # samples should be the indices
    samples = [orig_ids[i] for i in order]

    return dists, samples


def retrieve_examples_new(args, rep, vector_db, pool_number, topk, query_series=None):
    """
    Minimal modification for list-backed DB:
      - vector_db is a list of (orig_idx, rep_tensor, hs, series_tensor[1,T])
      - Returns:
          dists   : np.array of distances aligned to the FIRST `pool_number` candidates
                    (in the same order as the selected slice of vector_db)
          samples : list of selected original indices (orig_idx) length = topk
    Other notes unchanged from your original version.
    """
    import numpy as np

    # ---------- tiny helpers ----------
    def _to_np(x):
        # works for torch.Tensor or np.ndarray / list
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=float)

    def _series1d(x):
        # series can be torch [1,T] or [T]; make it 1D np
        s = _to_np(x)
        return s.reshape(-1)

    def _znorm(x, eps=1e-8):
        x = np.asarray(x, float)
        std = x.std()
        return (x - x.mean()) / (std + eps)

    def _pair_last(q, r, tail_n=None):
        q = np.asarray(q, float); r = np.asarray(r, float)
        if q.size == 0 or r.size == 0:
            return q[:0], r[:0]
        if tail_n is None:
            L = min(q.size, r.size)
        else:
            L = min(int(tail_n), q.size, r.size)
        return q[-L:], r[-L:]

    def _dtw(a, b, window=None):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 1e9
        if window is None: window = max(n, m)
        window = max(window, abs(n - m))
        INF = 1e18
        D = np.full((n+1, m+1), INF, dtype=float); D[0,0] = 0.0
        for i in range(1, n+1):
            j0, j1 = max(1, i-window), min(m, i+window)
            ai = a[i-1]
            for j in range(j0, j1+1):
                cost = (ai - b[j-1])**2
                D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return float(D[n,m]**0.5)

    def _xcorr_dist(a, b, max_lag=None):
        a, b = _znorm(a), _znorm(b)
        L = min(len(a), len(b))
        a, b = a[:L], b[:L]
        if L == 0:
            return 1.0
        if max_lag is None: max_lag = L - 1
        max_lag = int(max(0, max_lag))
        best = -1.0
        for lag in range(-max_lag, max_lag+1):
            if lag >= 0:
                aa, bb = a[lag:], b[:L-lag]
            else:
                aa, bb = a[:L+lag], b[-lag:]
            if len(aa) == 0:
                continue
            denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-8
            c = float(np.dot(aa, bb)) / denom
            if c > best: best = c
        return 1.0 - best

    def _fft_feat(x, k=8):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.zeros(k, dtype=float)
        X = np.fft.rfft(x)
        mag = np.abs(X)[1:k+1]
        nrm = np.linalg.norm(mag)
        return (mag / nrm) if nrm > 0 else mag

    def _cosine_dist(u, v, eps=1e-8):
        # ensure 1-D float arrays
        u = np.asarray(u, dtype=np.float32).reshape(-1)
        v = np.asarray(v, dtype=np.float32).reshape(-1)

        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0:
            return 1.0  # maximum distance if one is zero vector

        return 1.0 - float(np.dot(u, v) / (nu * nv + eps))

    def _pearson_dist(a, b, eps=1e-8):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size == 0 or b.size == 0:
            return 1.0
        a = a - a.mean(); b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
        r = float(np.dot(a, b) / denom)
        return 1.0 - abs(r)

    def _zscore_safe(x):
        x = np.asarray(x, float)
        finite = np.isfinite(x)
        if not np.any(finite):
            return np.zeros_like(x)
        mu = np.nanmean(x[finite]); sd = np.nanstd(x[finite])
        sd = 1.0 if (not np.isfinite(sd) or sd == 0) else sd
        z = (x - mu) / sd
        z[~finite] = np.max(z[finite]) + 5.0
        return z

    def _welch_psd_prob(x, fs=1.0, nperseg=None, window="hann"):
        x = np.asarray(x, float)
        if x.size == 0:
            return np.array([1.0], dtype=float)
        x = (x - x.mean()) / (x.std() + 1e-8)  # stabilise scale
        if nperseg is None:
            nperseg = max(32, min(256, x.size // 4))
        nperseg = min(nperseg, x.size)  # SciPy will warn otherwise
        f, Pxx = _welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=nperseg//2, detrend="constant", scaling="density")
        Pxx = np.maximum(Pxx, 0.0)
        s = Pxx.sum()
        return (Pxx / s) if s > 0 else np.ones_like(Pxx) / len(Pxx)

    def _hellinger(p, q):
        p = np.asarray(p, float); q = np.asarray(q, float)
        L = min(p.size, q.size)
        if L == 0: return 1.0
        p = p[:L]; q = q[:L]
        p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
        return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2)))  # ∈ [0,1]

    # ---------- config / candidates ----------
    if vector_db is None or len(vector_db) == 0:
        return np.array([]), []

    tail_n = getattr(args, "tail_n", None)
    mode   = getattr(args, "retrieval", "euclidean").lower()

    # Use the last pool_number candidates from this channel's list
    if pool_number is not None:
        pool_n = min(int(pool_number), len(vector_db))
    else:
        pool_n = len(vector_db)
    idxs = list(range(len(vector_db) - pool_n, len(vector_db)))  # indices into vector_db

    # Query representations
    rep_np = _to_np(rep)
    q_series_np = None if query_series is None else _series1d(query_series)

    # Shortcuts to pull fields
    def _rep_of(j):
        return _to_np(vector_db[j][1])
    def _raw_of(j):
        return _series1d(vector_db[j][3])

    # ---------- distances by mode ----------
    if mode == "euclidean":
        d = [np.linalg.norm(rep_np - _rep_of(j)) for j in idxs]
    elif mode == "cosine":
        d = [_cosine_dist(rep_np, _rep_of(j)) for j in idxs]
    elif mode == "z_euclidean":
        if q_series_np is None:
            raise ValueError("z_euclidean requires query_series (raw 1D query).")
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            d.append(np.linalg.norm(_znorm(qt) - _znorm(rt)))
    elif mode == "pearson":
        if q_series_np is None:
            raise ValueError("pearson requires query_series (raw 1D query).")
        d = []
    elif mode == "dtw":
        if q_series_np is None:
            raise ValueError("dtw requires query_series (raw 1D query).")
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            d.append(_dtw(qt, rt))
    elif mode == "xcorr":
        if q_series_np is None:
            raise ValueError("xcorr requires query_series (raw 1D query).")
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            max_lag = getattr(args, "max_lag", None)
            if max_lag is None:
                max_lag = max(0, min(len(qt), len(rt)) - 1)
            d.append(_xcorr_dist(qt, rt, max_lag=int(max_lag)))
    elif mode == "fft":
        if q_series_np is None:
            raise ValueError("fft requires query_series (raw 1D query).")
        k_bins = int(getattr(args, "fft_k", 8))
        d = []
        for j in idxs:
            r = _raw_of(j)
            qt, rt = _pair_last(q_series_np, r, tail_n)
            d.append(np.linalg.norm(_fft_feat(qt, k=k_bins) - _fft_feat(rt, k=k_bins)))

    elif mode == "psd_hellinger":
        if q_series_np is None:
            raise ValueError("psd_hellinger requires query_series (raw 1D query).")
        fs = float(getattr(args, "psd_fs", 1.0))
        nperseg = getattr(args, "psd_nperseg", None)
        window = getattr(args, "psd_window", "hann")
        d = []
        for j in idxs:
            qt, rt = _pair_last(q_series_np, _raw_of(j), tail_n)
            px = _welch_psd_prob(qt, fs=fs, nperseg=nperseg, window=window)
            py = _welch_psd_prob(rt, fs=fs, nperseg=nperseg, window=window)
            d.append(_hellinger(px, py))  # bounded [0,1]

    elif mode == "hybrid":
        alpha = float(getattr(args, "alpha", 0.5))
        sec   = getattr(args, "hybrid_sec", "dtw").lower()
        if sec not in {"dtw", "xcorr", "fft", "z_euclidean", "pearson"}:
            sec = "dtw"

        rep_d = [np.linalg.norm(rep_np - _rep_of(j)) for j in idxs]

        if q_series_np is None:
            sec_d = [_cosine_dist(rep_np, _rep_of(j)) for j in idxs]
        else:
            vals = []
            for j in idxs:
                r = _raw_of(j)
                qt, rt = _pair_last(q_series_np, r, tail_n)
                if sec == "dtw":
                    vals.append(_dtw(qt, rt))
                elif sec == "xcorr":
                    max_lag = getattr(args, "max_lag", None)
                    if max_lag is None:
                        max_lag = max(0, min(len(qt), len(rt)) - 1)
                    vals.append(_xcorr_dist(qt, rt, max_lag=int(max_lag)))
                elif sec == "fft":
                    k_bins = int(getattr(args, "fft_k", 8))
                    vals.append(np.linalg.norm(_fft_feat(qt, k=k_bins) - _fft_feat(rt, k=k_bins)))
                elif sec == "z_euclidean":
                    vals.append(np.linalg.norm(_znorm(qt) - _znorm(rt)))
                else:
                    vals.append(_pearson_dist(qt, rt))
            sec_d = vals

        rep_z = _zscore_safe(rep_d)
        sec_z = _zscore_safe(sec_d)
        d = alpha * rep_z + (1.0 - alpha) * sec_z
    else:
        raise ValueError(f"Unknown retrieval mode: {args.retrieval}")

    # ---------- selection / return ----------
    dists = np.asarray(d, dtype=float)
    k = min(int(topk), len(dists))
    order = np.argsort(dists)[:k]

    # print("k", k)
    # print("order", order)

    return dists, order


def add_icv_layers_interpolate(
    model,
    icv_per_layer,                 # list[Tensor], len == #layers
    *,
    lam,                   
    beta: float = 0.5,            
    mlp_keywords=("ffn_layer", "mlp", "feedforward"),
    skip_zero: bool = True,
    # confidence knobs
    baseline=0.25, margin=0.10, power=1.25,
    renorm_conf="stretch", stretch=0.10,
    max_frac=None, schedule=None, global_gain=None,
):
    layers = get_layers(model)
    assert len(icv_per_layer) == len(layers), "Length mismatch between icv_per_layer and model layers."


    for i, layer in enumerate(layers):
        ffn = find_module(layer, mlp_keywords)
        if ffn is None:
            continue

        v = icv_per_layer[i]
        if not isinstance(v, torch.Tensor):
            continue
        if skip_zero and v.abs().max().item() < 1e-8:
            continue

        p = next(ffn.parameters())
        dev, dt = p.device, p.dtype

        # per-layer lam allowed
        lam_i = lam[i] if isinstance(lam, (list, tuple)) and i < len(lam) else lam

        ffn.icv_tail = ICVBetaTail(
            v.to(device=dev, dtype=dt),
            lam_i,
            dtype=dt,
            beta=beta,
            baseline=baseline, margin=margin, power=power,
            renorm_conf=renorm_conf, stretch=stretch,
            max_frac=max_frac, schedule=schedule, global_gain=global_gain
        ).to(device=dev, dtype=dt)

        if hasattr(ffn.icv_tail, "reset"):
            ffn.icv_tail.reset()

        if not hasattr(ffn, "_orig_forward"):
            ffn._orig_forward = ffn.forward
            def _forward_with_icv(self, *args, **kwargs):
                out = self._orig_forward(*args, **kwargs)
                if isinstance(out, tuple):
                    y, router_logits = out
                else:
                    y, router_logits = out, None
                if hasattr(self, "icv_tail") and self.icv_tail is not None:
                    y = self.icv_tail(y, gain_map=None)
                return (y, router_logits) if router_logits is not None else y
            ffn.forward = types.MethodType(_forward_with_icv, ffn)


def get_timer_mlp_layers(model_or_wrapper):
    """
    Returns a list of TimerMLP modules (one per decoder layer).
    Accepts either TimerForPrediction or TimerModel.
    """
    # unwrap TimerForPrediction -> TimerModel
    if hasattr(model_or_wrapper, "model") and isinstance(model_or_wrapper.model, nn.Module):
        core = model_or_wrapper.model
    else:
        core = model_or_wrapper

    layers = []
    assert hasattr(core, "layers"), "Expected .layers (ModuleList of TimerDecoderLayer)"
    for layer in core.layers:
        assert hasattr(layer, "ffn_layer"), "TimerDecoderLayer lacks ffn_layer"
        layers.append(layer.ffn_layer)
    return layers




class ICVBetaTailTimerXL(nn.Module):
    """
    Timer-XL friendly steering:
      - Applies delta to x (FFN output)
      - Computes gating cosines on gate_x (FFN input)
      - Supports x layouts: [B,H], [B,T,H], [T,B,H]
      - Supports gain_map layouts: [B,T], [T,B], [B,T,1], [T,B,1]
    """

    def __init__(
        self,
        icv: torch.Tensor,             # [K,H] or [H]
        lam,                           # scalar or [K]
        *,
        beta: float = 0.5,
        baseline: float = 0.25,
        margin: float = 0.10,
        power: float = 1.25,
        renorm_conf: str = "keep",     # "keep" or "stretch"
        stretch: float = 0.10,
        max_frac: float | None = 0.05,
        schedule: dict | None = None,
        global_gain: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert 0.0 <= beta <= 1.0
        self.beta = float(beta)
        self.baseline = float(baseline)
        self.margin = float(margin)
        self.power = float(power)
        self.renorm_conf = str(renorm_conf)
        self.stretch = float(stretch)
        self.max_frac = max_frac
        self.schedule = schedule or {}
        self.global_gain = None if global_gain is None else float(global_gain)
        self.eps = float(eps)

        if icv.dim() == 1:
            icv = icv.unsqueeze(0)
        if icv.dim() != 2:
            raise ValueError(f"icv must be [K,H] or [H], got {tuple(icv.shape)}")
        self.register_buffer("icv", icv)

        if isinstance(lam, (list, tuple)):
            lam = torch.tensor(lam, dtype=torch.float32)
        elif not isinstance(lam, torch.Tensor):
            lam = torch.tensor([float(lam)], dtype=torch.float32)
        if lam.dim() == 0:
            lam = lam.view(1)
        self.register_buffer("lam", lam.float())

    @staticmethod
    def _to_bth(x: torch.Tensor):
        """
        Normalize to [B,T,H]. Return xb, meta for restoring.
        meta = (squeezed_2d, was_tbh)
        """
        if x.dim() == 2:
            # [B,H] -> [B,1,H]
            return x.unsqueeze(1), (True, False)

        if x.dim() == 3:
            a, b, h = x.shape
            # Heuristic: treat [T,B,H] if second dim looks like batch (small)
            # and first dim looks like sequence (larger).
            if b <= 64 and a >= b * 4:
                return x.transpose(0, 1).contiguous(), (False, True)  # [B,T,H], was_tbh=True
            return x, (False, False)  # already [B,T,H]

        raise ValueError(f"x must be [B,H], [B,T,H], or [T,B,H], got {tuple(x.shape)}")

    @staticmethod
    def _from_bth(x_bth: torch.Tensor, meta):
        squeezed_2d, was_tbh = meta
        y = x_bth
        if squeezed_2d:
            y = y.squeeze(1)  # [B,H]
        if was_tbh:
            y = y.transpose(0, 1).contiguous()  # back to [T,B,H]
        return y

    @staticmethod
    def _align_gain_map(gain_map: torch.Tensor, B: int, T: int, device, dtype):
        gm = gain_map
        if gm.dim() == 2:
            if gm.shape == (B, T):
                gm = gm
            elif gm.shape == (T, B):
                gm = gm.transpose(0, 1)
            else:
                raise ValueError(f"gain_map {tuple(gm.shape)} not compatible with B={B},T={T}")
            gm = gm.unsqueeze(-1)  # [B,T,1]
        elif gm.dim() == 3:
            if gm.shape == (B, T, 1):
                gm = gm
            elif gm.shape == (T, B, 1):
                gm = gm.transpose(0, 1)
            else:
                raise ValueError(f"gain_map {tuple(gm.shape)} not compatible with B={B},T={T}")
        else:
            raise ValueError(f"gain_map must be 2D/3D, got {gm.dim()}D")
        return gm.to(device=device, dtype=dtype)

    def _broadcast_lam(self, K: int, device):
        lam = self.lam.to(device=device, dtype=torch.float32)
        if lam.numel() == 1 and K > 1:
            lam = lam.repeat(K)
        if lam.numel() != K:
            raise ValueError(f"lam has {lam.numel()} elems but icv has K={K}")
        return lam

    def forward(self, x: torch.Tensor, *, gate_x: torch.Tensor | None = None, gain_map: torch.Tensor | None = None):
        if self.icv is None or self.icv.numel() == 0:
            return x

        # Keep output dtype identical to input dtype (do not force a fixed dtype).
        out_dtype = x.dtype

        xb_apply, meta_apply = self._to_bth(x.float())  # [B,T,H]
        B, T, H = xb_apply.shape
        orig_norm = xb_apply.norm(dim=-1, keepdim=True) + self.eps

        if gate_x is None:
            xb_gate = xb_apply
        else:
            xb_gate, meta_gate = self._to_bth(gate_x.float())
            if xb_gate.shape != xb_apply.shape:
                raise ValueError(f"gate_x normalized shape {tuple(xb_gate.shape)} != x normalized shape {tuple(xb_apply.shape)}")

        # Directions
        d = self.icv.float()  # [K,H]
        d = d / (d.norm(dim=-1, keepdim=True) + self.eps)
        K = d.shape[0]
        lamK = self._broadcast_lam(K, xb_apply.device)  # [K]

        # Cosines: compute without expanding to [K,B,T,H]
        g = F.normalize(xb_gate, dim=-1)  # [B,T,H]
        cos = torch.einsum("bth,kh->kbt", g, d)  # [K,B,T]
        cos_neg = -cos

        # ---------- plain branch ----------
        lam_plain = lamK.view(K, 1, 1)                       # [K,1,1]
        lambda_sim = 1.0 + F.relu(cos_neg)                   # [K,B,T]
        coeff_plain = lam_plain * lambda_sim                 # [K,B,T]
        # mean over K to match your previous behavior
        y_plain = torch.einsum("kbt,kh->bth", coeff_plain, d) / max(1, K)  # [B,T,H]

        x_unit = F.normalize(xb_apply, dim=-1)
        out_plain = F.normalize(x_unit + y_plain, dim=-1) * orig_norm

        if self.beta <= 0.0:
            return self._from_bth(out_plain, meta_apply).to(out_dtype)

        # ---------- confidence branch ----------
        gate_raw = F.relu(cos_neg + self.margin)             # [K,B,T]
        gate = self.baseline + gate_raw.pow(self.power)      # [K,B,T]

        # Normalize lam weights across K (unless K==1)
        w = lamK.clone()
        if K == 1:
            global_gain = float(w.item())
            w = torch.ones_like(w)
        else:
            global_gain = 1.0 if self.global_gain is None else float(self.global_gain)
            w = w / (w.sum() + self.eps)

        gate = gate * w.view(K, 1, 1)                        # [K,B,T]
        delta_conf = torch.einsum("kbt,kh->bth", gate, d)     # [B,T,H]

        # Optional schedule over time
        if self.schedule:
            scale = float(self.schedule.get("scale", 1.0))
            if T > 1:
                sched = torch.linspace(1.0, 0.0, steps=T, device=delta_conf.device, dtype=delta_conf.dtype).view(1, T, 1)
            else:
                sched = torch.ones(1, T, 1, device=delta_conf.device, dtype=delta_conf.dtype)
            delta_conf = delta_conf * (scale * sched)

        # Apply external gain map correctly
        if gain_map is not None:
            gm = self._align_gain_map(gain_map, B=B, T=T, device=delta_conf.device, dtype=delta_conf.dtype)  # [B,T,1]
            delta_conf = delta_conf * gm

        delta_conf = delta_conf * global_gain

        # Clamp magnitude relative to original norm
        if self.max_frac is not None:
            max_delta = self.max_frac * orig_norm
            delta_conf = torch.clamp(delta_conf, min=-max_delta, max=max_delta)

        out_conf = xb_apply + delta_conf

        if self.renorm_conf == "keep":
            out_conf = F.normalize(out_conf, dim=-1) * orig_norm
        elif self.renorm_conf == "stretch":
            gmean = gate.mean(dim=0).unsqueeze(-1)           # [B,T,1]
            factor = (1.0 + self.stretch * gmean).clamp(min=0.0)
            out_conf = out_conf * factor

        out = (1.0 - self.beta) * out_plain + self.beta * out_conf
        return self._from_bth(out, meta_apply).to(out_dtype)


def add_icv_layers_timer(
    model,
    icv_per_layer,                 # list[Tensor], len == #layers
    *,
    lam,                   
    beta: float = 0.5,            
    mlp_keywords=("ffn_layer", "mlp", "feedforward"),
    skip_zero: bool = False,
    # confidence knobs
    baseline=0.25, margin=0.10, power=1.25,
    renorm_conf="stretch", stretch=0.10,
    max_frac=None, schedule=None, global_gain=None,
):
    layers = get_layers(model)
    assert len(icv_per_layer) == len(layers), "Length mismatch between icv_per_layer and model layers."
    layers_snapshot = list(layers)  # freeze ModuleList so we never mutate what we iterate

    for i, layer in enumerate(layers_snapshot):
        # get the FFN inside this decoder layer (TimerDecoderLayer has .ffn_layer)
        ffn = getattr(layer, "ffn_layer", None)
        if ffn is None:
            raise RuntimeError(f"Could not find ffn_layer in TimerDecoderLayer at index {i}.")

        if i >= len(icv_per_layer):
            break

        v = icv_per_layer[i]
        if not torch.is_tensor(v):
            raise ValueError(f"icv_per_layer[{i}] is not a tensor.")
        if skip_zero and v.abs().max().item() < 1e-8:
            raise RuntimeError(f"Skipping zero icv at layer {i}.")

        p = next(ffn.parameters(), None)
        if p is None:
            raise RuntimeError(f"FFN at layer {i} has no parameters.")
        dev, dt = p.device, p.dtype

        lam_i = lam[i] if isinstance(lam, (list, tuple)) and i < len(lam) else lam

        # attach to the *FFN module*, not the ModuleList
        ffn.icv_tail = ICVBetaTail(
            v.to(device=dev, dtype=dt),
            lam_i,
            dtype=dt,
            beta=beta,
            baseline=baseline,
            margin=margin,
            power=power,
            renorm_conf=renorm_conf,
            stretch=stretch,
            max_frac=max_frac,
            schedule=schedule,
            global_gain=global_gain,
        ).to(device=dev, dtype=dt)

        if hasattr(ffn.icv_tail, "reset"):
            ffn.icv_tail.reset()

        # wrap forward once per FFN
        if not hasattr(ffn, "_orig_forward"):
            ffn._orig_forward = ffn.forward

            def _forward_with_icv(self, *args, **kwargs):
                out = self._orig_forward(*args, **kwargs)

                if isinstance(out, tuple):
                    y, *rest = out
                else:
                    y, rest = out, []

                tail = getattr(self, "icv_tail", None)
                if tail is not None:
                    y = tail(y, gain_map=None)

                return (y, *rest) if rest else y

            ffn.forward = types.MethodType(_forward_with_icv, ffn)


def remove_icv_layers_timer(model: nn.Module):
    hs = getattr(model, "_timerxl_icv_handles", None)
    if hs:
        for h in hs:
            try:
                h.remove()
            except Exception:
                pass
    if hasattr(model, "_timerxl_icv_handles"):
        delattr(model, "_timerxl_icv_handles")
    if hasattr(model, "_timerxl_icv_modules"):
        delattr(model, "_timerxl_icv_modules")
