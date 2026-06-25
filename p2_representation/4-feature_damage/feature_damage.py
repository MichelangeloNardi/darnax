"""Diagnostic #5, refined at the POOLED-FEATURE level (+ logit damage, rescue/lesion).

Per-spin readout importance is block-constant under 8x8 pooling (all 64 spins in a
block share the same W_out row), so the per-spin flip correlation is too coarse. Here
we work in the 256-dim pooled-feature space where the readout actually acts.

For each model (A local-rule, B BPTT CE_D, C BPTT+align) x seed, using the serialized
models from ../3-CD_diagnostics/models/:

  1. Train a linear probe on pooled C  ->  W_probe_C (256 x 10).
     Feature importance = ||W_probe_C[f, :]||_2.
  2. Feature-level C->D change  d[f] = pool(C)[f] - pool(D)[f].
     Correlate importance with change (feature-level and per (feature,example)).
  3. Logit/margin damage  damage[ex,f] = W_probe_C[f, y] * (pool(C)[f] - pool(D)[f]).
     Total correct-class logit drop C->D = sum_f damage. Per-feature mean damage.
  4. Rescue: replace the top-k most-damaged pooled features of D with their C values,
     evaluate under W_probe_C -> does accuracy climb from acc(D) toward acc(C)?
     Lesion: replace those same top-k features of C with D values -> does C collapse?
     (top-k chosen per example by signed damage, i.e. correct-class contribution lost.)

All accuracies are measured with the SAME C-probe (W_probe_C), so the rescue/lesion
curves interpolate between acc(C-probe on D) [k=0 rescue] and acc(C-probe on C)
[k=0 lesion]. Test set used for analysis; a train split fits the probe.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/4-feature_damage/feature_damage.py
Smoke:  python p2_representation/4-feature_damage/feature_damage.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE.parent / "3-CD_diagnostics" / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_GRID = [0, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]


def make_roller(warmup, clamped, free):
    roll = cm.make_rollout(warmup, clamped, free)

    @eqx.filter_jit
    def f(orch, state, key):
        return roll(orch, state, key)[0]
    return f


def load_model(seed, name, cfg):
    _, template = cm.build_model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{name}_seed{seed}.eqx", template)


def collect_pooled(orch, state_tmpl, it, roller, key, max_b):
    X, Y = [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        s = roller(orch, state_tmpl.init(cm.to_hwc(xb), yb), key)
        X.append(np.asarray(cm.pool_j1(np.asarray(s[1])))); Y.append(np.asarray(yb))
    return np.concatenate(X), np.argmax(np.concatenate(Y), 1)


def fit_probe(X, y_idx, epochs):
    """Adam linear probe on pooled reps; returns W (256,10) numpy."""
    probe = nn.Linear(X.shape[1], 10, bias=False).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=cm.PROBE_WD)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X).float(); yt = torch.from_numpy(y_idx).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
    return probe.weight.detach().cpu().numpy().T  # (256,10)


def acc(X, W, y):
    return float(((X @ W).argmax(1) == y).mean())


def pearson(a, b):
    a = a.ravel().astype(np.float64) - a.mean(); b = b.ravel().astype(np.float64) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def analyze(orch, cfg, ds, state_tmpl, roll_C, roll_D, args):
    key = jax.random.PRNGKey(0)
    # probe trained on C (train split)
    Xc_tr, ytr = collect_pooled(orch, state_tmpl, ds, roll_C, key, args.probe_train_batches)
    W = fit_probe(Xc_tr, ytr, args.probe_epochs)                     # (256,10)

    # test split: pooled C and D
    Xc, yte = collect_pooled(orch, state_tmpl, ds.iter_test(), roll_C, key, args.max_batches)
    Xd, _ = collect_pooled(orch, state_tmpl, ds.iter_test(), roll_D, key, args.max_batches)
    N = Xc.shape[0]

    imp_feat = np.linalg.norm(W, axis=1)                             # (256,)
    delta = Xc - Xd                                                  # (N,256) feature change
    change_feat = np.sqrt((delta ** 2).mean(0))                      # RMS change per feature
    Wy = W[:, yte].T                                                 # (N,256) probe weight for true class
    damage = Wy * delta                                             # (N,256) correct-class logit damage

    out = {
        "acc_C": acc(Xc, W, yte),                  # C-probe on C
        "acc_D": acc(Xd, W, yte),                  # C-probe on D (rescue k=0)
        "corr_imp_change_feat": pearson(imp_feat, change_feat),     # 256 features
        "corr_imp_change_fe": pearson(np.abs(Wy), np.abs(delta)),   # per (feature,example)
        "total_logit_damage": float(damage.sum(1).mean()),          # mean correct-class logit drop C->D
        "k_grid": K_GRID,
    }

    # rank features per example by signed damage (most correct-class contribution lost first)
    n_feat = Xc.shape[1]
    order = np.argsort(-damage, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.tile(np.arange(n_feat), (N, 1)), axis=1)

    rescue, lesion = [], []
    for k in K_GRID:
        mask = rank < k                                             # (N,256) top-k damaged
        Xd_resc = np.where(mask, Xc, Xd)                           # restore C values into D
        Xc_les = np.where(mask, Xd, Xc)                            # corrupt C with D values
        rescue.append(acc(Xd_resc, W, yte))
        lesion.append(acc(Xc_les, W, yte))
    out["rescue_curve"] = rescue
    out["lesion_curve"] = lesion
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--probe-train-batches", type=int, default=800)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 6; args.probe_epochs = 3; args.max_batches = 6

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll_C = make_roller(warmup, clamped, free)
    roll_D = make_roller(warmup, 0, free)

    t0 = time.time()
    results = {"config": "best_channel_entropy", "seeds": args.seeds, "k_grid": K_GRID, "models": {}}
    for name in ["A", "B", "C"]:
        per_seed = []
        for seed in args.seeds:
            orch = load_model(seed, name, cfg)
            d = analyze(orch, cfg, ds, state_tmpl, roll_C, roll_D, args)
            per_seed.append(d)
            print(f"  [{name} s{seed}] accC={d['acc_C']:.3f} accD(Cprobe)={d['acc_D']:.3f} "
                  f"corr_imp_change={d['corr_imp_change_feat']:.3f} "
                  f"logit_damage={d['total_logit_damage']:.3f} "
                  f"rescue@32={d['rescue_curve'][K_GRID.index(32)]:.3f} ({cm.fmt(time.time()-t0)})")
        scal = ["acc_C", "acc_D", "corr_imp_change_feat", "corr_imp_change_fe", "total_logit_damage"]
        results["models"][name] = {
            "per_seed": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in scal},
            "std": {k: float(np.std([p[k] for p in per_seed])) for k in scal},
            "rescue_mean": np.mean([p["rescue_curve"] for p in per_seed], 0).tolist(),
            "lesion_mean": np.mean([p["lesion_curve"] for p in per_seed], 0).tolist(),
        }

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "feature_damage.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
