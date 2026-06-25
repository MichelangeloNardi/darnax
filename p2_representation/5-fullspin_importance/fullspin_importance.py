"""Matei-style FULL-SPIN importance: are C->D flips concentrated on the spins that
carry C's class information?

Per-spin readout importance from the pooled W_out is block-constant (8x8 pooling), so
exp 3/4 couldn't resolve it. Here we train a regularized linear probe directly on the
16384 raw spins of C (no pooling), and use ITS weights as genuine per-spin importance.

For A/B/C (serialized exp-3 models) x seeds:
  - collect hard-sign C and D hidden states (N,32,32,16) -> flatten to 16384 spins.
  - probe_C: regularized linear probe trained on full-spin C_train. Report acc on
    C_test (probe_C_on_C) and D_test (probe_C_on_D).
  - per test example, true label y, strongest wrong class k = argmax_{c!=y} (C@W)_c:
      importance_i = |W[i,y] - W[i,k]|            (contribution to the y-vs-k margin)
      flip_i       = 1[C_i != D_i]
      damage_i     = (C_i - D_i) * (W[i,y] - W[i,k])
  - report corr(flip, importance), corr(flip, damage); mean imp/damage flipped vs
    stable; top-5% important n flipped enrichment; flip rate by importance decile.
  - random-flip controls (evaluate the C-probe):
      uniform   : flip the same #spins per example, chosen uniformly at random in C
      empirical : flip each spin with its empirical per-spin flip rate p_i
  - field diagnostics: |field_C| flipped vs stable; field_C*C on flipped spins.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/5-fullspin_importance/fullspin_importance.py
Smoke:  python p2_representation/5-fullspin_importance/fullspin_importance.py --smoke
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

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE.parent / "3-CD_diagnostics" / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DSPIN = cm.H * cm.W * cm.C  # 16384


def make_roller(warmup, clamped, free):
    roll = cm.make_rollout(warmup, clamped, free)

    @eqx.filter_jit
    def f(orch, state, key):
        return roll(orch, state, key)[0]
    return f


def load_model(seed, name, cfg):
    _, template = cm.build_model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{name}_seed{seed}.eqx", template)


def collect_fullspin(orch, state_tmpl, it, roller, key, max_b, want_field=False):
    """Return flattened hard-sign spins (N,16384), labels (N,), and optionally the
    flattened field (N,16384)."""
    S, F, Y = [], [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        s = roller(orch, state_tmpl.init(cm.to_hwc(xb), yb), key)
        S.append(np.asarray(s[1]).reshape(s[1].shape[0], -1))
        if want_field:
            F.append(np.asarray(s.fields[1]).reshape(s[1].shape[0], -1))
        Y.append(np.argmax(np.asarray(yb), 1))
    S = np.concatenate(S); Y = np.concatenate(Y)
    F = np.concatenate(F) if want_field else None
    return S, Y, F


def fit_probe(X, y, epochs, wd):
    """Regularized (L2) linear probe on full-spin X; returns W (16384,10)."""
    probe = nn.Linear(X.shape[1], 10, bias=False).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
        batch_size=256, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
    return probe.weight.detach().cpu().numpy().T  # (16384,10)


def acc(X, W, y):
    return float(((X @ W).argmax(1) == y).mean())


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def analyze(orch, ds, state_tmpl, roll_C, roll_D, args):
    key = jax.random.PRNGKey(0)
    rng = np.random.default_rng(0)

    # full-spin probe trained on C
    Xc_tr, ytr, _ = collect_fullspin(orch, state_tmpl, ds, roll_C, key, args.probe_train_batches)
    W = fit_probe(Xc_tr, ytr, args.probe_epochs, args.weight_decay)        # (16384,10)
    del Xc_tr

    # test C, D (+ C field)
    Xc, yte, Cf = collect_fullspin(orch, state_tmpl, ds.iter_test(), roll_C, key, args.max_batches, want_field=True)
    Xd, _, _ = collect_fullspin(orch, state_tmpl, ds.iter_test(), roll_D, key, args.max_batches)
    N = Xc.shape[0]

    out = {"probe_C_on_C": acc(Xc, W, yte), "probe_C_on_D": acc(Xd, W, yte)}

    # per-example strongest wrong class k under C logits
    logitsC = Xc @ W                                       # (N,10)
    masked = logitsC.copy(); masked[np.arange(N), yte] = -np.inf
    kcls = masked.argmax(1)                                # (N,)

    Wy = W[:, yte].T                                       # (N,16384)
    Wk = W[:, kcls].T                                      # (N,16384)
    margin_w = Wy - Wk                                     # (N,16384) per-spin y-vs-k weight
    importance = np.abs(margin_w)
    flip = (Xc != Xd)                                      # (N,16384) bool (hard +-1)
    damage = (Xc - Xd) * margin_w                          # (N,16384)

    out["corr_flip_importance"] = pearson(flip.astype(np.float64), importance)
    out["corr_flip_damage"] = pearson(flip.astype(np.float64), damage)
    fl, st = flip, ~flip
    out["imp_flipped"] = float(importance[fl].mean()); out["imp_stable"] = float(importance[st].mean())
    out["damage_flipped"] = float(damage[fl].mean()); out["damage_stable"] = float(damage[st].mean())

    # top-5% important ∩ flipped enrichment (per example, then averaged)
    thr = np.quantile(importance, 0.95, axis=1, keepdims=True)
    top = importance >= thr
    base_rate = float(flip.mean())
    rate_in_top = float(flip[top].mean())
    out["flip_rate_overall"] = base_rate
    out["flip_rate_in_top5pct"] = rate_in_top
    out["top5pct_enrichment"] = rate_in_top / base_rate if base_rate > 0 else 0.0

    # flip rate by global importance decile
    edges = np.quantile(importance, np.linspace(0, 1, 11))
    dec = np.clip(np.digitize(importance, edges[1:-1]), 0, 9)
    out["flip_rate_by_decile"] = [float(flip[dec == d].mean()) for d in range(10)]

    # random-flip controls (evaluate C-probe)
    kper = flip.reshape(N, -1).sum(1)
    rand_u = np.zeros_like(flip)
    for i in range(N):
        if kper[i] > 0:
            rand_u[i, rng.choice(DSPIN, size=int(kper[i]), replace=False)] = True
    Xc_u = Xc.copy(); Xc_u[rand_u] *= -1                  # flip spins in C (hard +-1)
    out["acc_rand_uniform"] = acc(Xc_u, W, yte)

    p_spin = flip.mean(0)                                  # empirical per-spin flip rate (16384,)
    rand_e = rng.random((N, DSPIN)) < p_spin[None, :]
    Xc_e = Xc.copy(); Xc_e[rand_e] *= -1
    out["acc_rand_empirical"] = acc(Xc_e, W, yte)

    # field diagnostics (field at C)
    out["absfield_flipped"] = float(np.abs(Cf)[fl].mean()); out["absfield_stable"] = float(np.abs(Cf)[st].mean())
    out["fieldC_dot_C_flipped"] = float((Cf * Xc)[fl].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--probe-train-batches", type=int, default=400)   # ~12.8k imgs
    ap.add_argument("--probe-epochs", type=int, default=25)
    ap.add_argument("--weight-decay", type=float, default=1e-3)       # regularized probe
    ap.add_argument("--max-batches", type=int, default=63)            # ~2000 test imgs
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 8; args.probe_epochs = 3
        args.weight_decay = 1e-3; args.max_batches = 6

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll_C = make_roller(warmup, clamped, free)
    roll_D = make_roller(warmup, 0, free)

    t0 = time.time()
    scal = ["probe_C_on_C", "probe_C_on_D", "corr_flip_importance", "corr_flip_damage",
            "imp_flipped", "imp_stable", "damage_flipped", "damage_stable",
            "flip_rate_overall", "flip_rate_in_top5pct", "top5pct_enrichment",
            "acc_rand_uniform", "acc_rand_empirical",
            "absfield_flipped", "absfield_stable", "fieldC_dot_C_flipped"]
    results = {"config": "best_channel_entropy", "seeds": args.seeds, "models": {}}
    for name in ["A", "B", "C"]:
        per_seed = []
        for seed in args.seeds:
            orch = load_model(seed, name, cfg)
            d = analyze(orch, ds, state_tmpl, roll_C, roll_D, args)
            per_seed.append(d)
            print(f"  [{name} s{seed}] pC->C={d['probe_C_on_C']:.3f} pC->D={d['probe_C_on_D']:.3f} "
                  f"corr(flip,imp)={d['corr_flip_importance']:.3f} enrich={d['top5pct_enrichment']:.2f} "
                  f"accD~ rand_u={d['acc_rand_uniform']:.3f} ({cm.fmt(time.time()-t0)})")
        results["models"][name] = {
            "per_seed": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in scal},
            "std": {k: float(np.std([p[k] for p in per_seed])) for k in scal},
            "flip_rate_by_decile_mean": np.mean([p["flip_rate_by_decile"] for p in per_seed], 0).tolist(),
        }

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "fullspin_importance.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
