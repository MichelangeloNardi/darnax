"""Does the C->D flip phenomenon exist in the classical (non-conv) FC net?

Train the FC asymmetric recurrent net (fc_common) with the local rule, then run the
exp-5 full-spin importance analysis on its 256 hidden spins:
  - probe_C: regularized linear probe trained on the clamped state C; report acc on
    C_test (probe_C_on_C) and D_test (probe_C_on_D).
  - decisive control: flip the SAME number of spins per example AT RANDOM in C and
    re-evaluate the C-probe (acc_rand_uniform). If the actual C->D flips crater the
    probe while random flips do not, the flips target the class-carrying spins.
  - per-spin: corr(flip, importance/damage), top-5% importance enrichment, flip rate
    by importance decile, |field_C| flipped vs stable.

Smoke (local CPU):  uv run python p3_spin_analysis/1-fc_phenomenon/phenomenon.py --smoke
Cluster: XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python \
  p3_spin_analysis/1-fc_phenomenon/phenomenon.py [--train-win]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p3_spin_analysis"))

import fc_common as fc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_probe(X, y, epochs, wd):
    """Regularized (L2) linear probe on full-spin X; returns W (D, 10)."""
    probe = nn.Linear(X.shape[1], fc.N_CLASSES, bias=False).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X).float(),
                                        torch.from_numpy(y).long()),
        batch_size=256, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
    return probe.weight.detach().cpu().numpy().T


def acc(X, W, y):
    return float(((X @ W).argmax(1) == y).mean())


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def train_backbone(cfg, seed, epochs, train_batches):
    key = jax.random.PRNGKey(seed)
    state, orch = fc.build_model(cfg, key)
    opt, opt_state = fc.make_optimizer(orch, cfg, win=cfg.get("train_win", False))
    trainer = fc.make_trainer(orch, state, opt, opt_state, cfg)
    ds = fc.get_dataset(batch_size=32)
    key = jax.random.PRNGKey(seed + 1)
    for ep in range(epochs):
        if train_batches is None:
            trainer, key = fc.train_epoch(trainer, ds, key)
        else:
            for i, (xb, yb) in enumerate(ds):
                if i >= train_batches:
                    break
                key = trainer.train_step(fc.downsample(xb), yb, key)
        head, key = fc.eval_head(trainer, ds, key)
        print(f"    [s{seed}] ep{ep} head_D={head:.3f} ({fc.fmt(time.time()-T0)})")
    return trainer.orchestrator, state, ds


def analyze(orch, state, ds, cfg, args):
    key = jax.random.PRNGKey(0)
    rng = np.random.default_rng(0)
    roll_C, roll_D = fc.rollers(cfg)
    nspin = cfg.get("n_spins", fc.N_SPINS)

    Xc_tr, ytr = fc.collect_spins(orch, state, ds, roll_C, key, args.probe_train_batches)
    W = fit_probe(Xc_tr, ytr, args.probe_epochs, args.weight_decay)
    del Xc_tr

    Xc, yte, Cf = fc.collect_spins(orch, state, ds.iter_test(), roll_C, key,
                                   args.max_batches, want_field=True)
    Xd, _ = fc.collect_spins(orch, state, ds.iter_test(), roll_D, key, args.max_batches)
    Wd = fit_probe(*fc.collect_spins(orch, state, ds, roll_D, key, args.probe_train_batches),
                   args.probe_epochs, args.weight_decay)
    N = Xc.shape[0]

    out = {"probe_C_on_C": acc(Xc, W, yte), "probe_C_on_D": acc(Xd, W, yte),
           "probe_D_on_D": acc(Xd, Wd, yte)}

    logitsC = Xc @ W
    masked = logitsC.copy(); masked[np.arange(N), yte] = -np.inf
    kcls = masked.argmax(1)
    margin_w = W[:, yte].T - W[:, kcls].T
    importance = np.abs(margin_w)
    flip = (Xc != Xd)
    damage = (Xc - Xd) * margin_w

    out["corr_flip_importance"] = pearson(flip.astype(np.float64), importance)
    out["corr_flip_damage"] = pearson(flip.astype(np.float64), damage)
    fl, st = flip, ~flip
    out["imp_flipped"] = float(importance[fl].mean()); out["imp_stable"] = float(importance[st].mean())

    thr = np.quantile(importance, 0.95, axis=1, keepdims=True)
    top = importance >= thr
    base_rate = float(flip.mean())
    out["flip_rate_overall"] = base_rate
    out["flip_rate_in_top5pct"] = float(flip[top].mean())
    out["top5pct_enrichment"] = float(flip[top].mean()) / base_rate if base_rate > 0 else 0.0

    edges = np.quantile(importance, np.linspace(0, 1, 11))
    dec = np.clip(np.digitize(importance, edges[1:-1]), 0, 9)
    out["flip_rate_by_decile"] = [float(flip[dec == d].mean()) for d in range(10)]

    kper = flip.sum(1)
    rand_u = np.zeros_like(flip)
    for i in range(N):
        if kper[i] > 0:
            rand_u[i, rng.choice(nspin, size=int(kper[i]), replace=False)] = True
    Xc_u = Xc.copy(); Xc_u[rand_u] *= -1
    out["acc_rand_uniform"] = acc(Xc_u, W, yte)

    p_spin = flip.mean(0)
    rand_e = rng.random((N, nspin)) < p_spin[None, :]
    Xc_e = Xc.copy(); Xc_e[rand_e] *= -1
    out["acc_rand_empirical"] = acc(Xc_e, W, yte)

    out["overlap_CD"] = float((Xc == Xd).mean())
    out["absfield_flipped"] = float(np.abs(Cf)[fl].mean()); out["absfield_stable"] = float(np.abs(Cf)[st].mean())
    out["fieldC_dot_C_flipped"] = float((Cf * Xc)[fl].mean())
    return out


T0 = time.time()


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-win", action="store_true", help="train W_in (sparse) instead of frozen")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-batches", type=int, default=None)
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--probe-epochs", type=int, default=25)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=63)
    ap.add_argument("--cfg", type=str, default=str(REPO / "p3_spin_analysis" / "fc_start_cfg.json"))
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 2; args.train_batches = 40
        args.probe_train_batches = 12; args.probe_epochs = 4; args.max_batches = 6

    cfg = fc.load_cfg(args.cfg)
    if args.train_win:
        cfg["train_win"] = True

    scal = ["probe_C_on_C", "probe_C_on_D", "probe_D_on_D", "overlap_CD",
            "corr_flip_importance", "corr_flip_damage", "imp_flipped", "imp_stable",
            "flip_rate_overall", "flip_rate_in_top5pct", "top5pct_enrichment",
            "acc_rand_uniform", "acc_rand_empirical",
            "absfield_flipped", "absfield_stable", "fieldC_dot_C_flipped"]
    per_seed = []
    T0 = time.time()
    for seed in args.seeds:
        orch, state, ds = train_backbone(cfg, seed, args.epochs, args.train_batches)
        d = analyze(orch, state, ds, cfg, args)
        per_seed.append(d)
        print(f"  [s{seed}] pC->C={d['probe_C_on_C']:.3f} pC->D={d['probe_C_on_D']:.3f} "
              f"pD->D={d['probe_D_on_D']:.3f} flip={d['flip_rate_overall']:.3f} "
              f"rand_u={d['acc_rand_uniform']:.3f} enrich={d['top5pct_enrichment']:.2f} "
              f"({fc.fmt(time.time()-T0)})")

    results = {
        "config": Path(args.cfg).name, "train_win": cfg.get("train_win", False),
        "seeds": args.seeds, "per_seed": per_seed,
        "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in scal},
        "std": {k: float(np.std([p[k] for p in per_seed])) for k in scal},
        "flip_rate_by_decile_mean": np.mean([p["flip_rate_by_decile"] for p in per_seed], 0).tolist(),
    }
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else
                          ("phenomenon_trainwin.json" if cfg.get("train_win") else "phenomenon.json"))
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {fc.fmt(time.time()-T0)})")


if __name__ == "__main__":
    main()
