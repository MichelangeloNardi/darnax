"""Exp p3-3 — dangerous-spin gated rule: per-position×channel kappa in the conv C=16 model.

The local rule updates J only where the margin is below threshold:
  J += eta * s_i s_j * 1[s_i (J s)_i < kappa].
Diagnostics: the units that lose class info C->D are the weakly field-pinned ones — the rule
isn't stabilising them enough. Idea: make kappa PER-POSITION×CHANNEL and raise it on the
dangerous (persistently low mean |field_C|) units so the rule fires on a wider margin window
there. Variant (a) only: dangerous = low mean |field_C| (label-free during training; task-1
showed polarization gives no label-free signal, so variant (b) is dropped).

Mechanism (no darnax-core edit): the conv gate is `(y*y_hat < threshold)` and an ARRAY threshold
broadcasts over (N,H,W,C); `threshold` is not touched by the optimizer (backward returns zero for
it). So we set J1.threshold to a (H,W,C) kappa map each epoch:
  meanfield_i = mean_n |field_C_i|  (over a subset, current weights)
  dangerous_i = meanfield_i < quantile(meanfield, q)   (bottom-q fraction)
  kappa_i     = threshold_j + boost * dangerous_i
boost=0 recovers the standard rule (uniform threshold_j) = baseline. Sweep (boost, q).

Baseline: conv C=16 at best_channel_entropy_cfg (probe_D ~0.45; gradient reference ~0.51).
Metric: probe_D (Adam linear probe on pooled D). Per the standing rule, the modified rule gets
its own tuning (tune_gated.py); this script is the functional sweep + 3-seed confirm.

Smoke:  uv run python p3_spin_analysis/3-gated_rule/gated_kappa.py --smoke
Cluster: XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python \
  p3_spin_analysis/3-gated_rule/gated_kappa.py
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
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
H, W, C = cm.H, cm.W, cm.C


def compute_kappa_map(orch, state_tmpl, ds, cfg, key, boost, q, n_batches):
    """(H,W,C) kappa map: threshold_j + boost on the bottom-q-quantile mean|field_C| units."""
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll = eqx.filter_jit(cm.make_rollout(warmup, clamped, free))
    acc = np.zeros((H, W, C), np.float64); nb = 0
    k = key
    for i, (xb, yb) in enumerate(ds):
        if i >= n_batches:
            break
        s, k = roll(orch, state_tmpl.init(cm.to_hwc(xb), yb), k)
        acc += np.abs(np.asarray(s.fields[1])).mean(0); nb += 1
    meanfield = acc / max(nb, 1)                          # (H,W,C)
    base = float(cfg["threshold_j"])
    if boost == 0.0:
        return jnp.full((H, W, C), base, jnp.float32)
    thr = np.quantile(meanfield, q)
    kappa = base + boost * (meanfield < thr).astype(np.float32)
    return jnp.asarray(kappa, jnp.float32)


def set_kappa(trainer, kmap):
    trainer.orchestrator = eqx.tree_at(lambda o: o.lmap[1][1].threshold, trainer.orchestrator, kmap)
    return trainer


def measure(orch, state_tmpl, ds, cfg, key, probe_epochs, train_batches, ov_batches=63):
    """probe_D + head_D + head_C + Omega_CD (C<->D overlap = mean(sign C · sign D)).
    Omega_CD is the diagnostic for the head-accuracy explanation: the W_out head trains on C and
    evals on D, so a higher C<->D overlap makes the head transfer (head_D -> head_C) even if the
    pooled representation (probe_D) is unchanged."""
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    rD = eqx.filter_jit(cm.make_rollout(warmup, 0, free))
    rC = eqx.filter_jit(cm.make_rollout(warmup, clamped, free))
    Wout = np.asarray(orch.lmap[2][1].W)

    def grab_pooled(it, mb, k):
        X, Y = [], []
        for i, (xb, yb) in enumerate(it):
            if mb is not None and i >= mb:
                break
            s, k = rD(orch, state_tmpl.init(cm.to_hwc(xb), yb), k)
            X.append(np.asarray(cm.pool_j1(np.asarray(s[1])))); Y.append(np.asarray(yb))
        return np.concatenate(X), np.concatenate(Y), k
    Xtr, Ytr, key = grab_pooled(ds, train_batches, key)
    Xte, Yte, key = grab_pooled(ds.iter_test(), None, key)
    ytr, yte = np.argmax(Ytr, 1), np.argmax(Yte, 1)
    pD = max(bc.offline_probe(Xtr, ytr, Xte, yte, probe_epochs))
    head_D = float((Xte @ Wout).argmax(1).__eq__(yte).mean())

    ov, hc, k = [], [], key
    for i, (xb, yb) in enumerate(ds.iter_test()):
        if i >= ov_batches:
            break
        y = np.argmax(np.asarray(yb), 1)
        sC, k = rC(orch, state_tmpl.init(cm.to_hwc(xb), yb), k)
        sD, k = rD(orch, state_tmpl.init(cm.to_hwc(xb), yb), k)
        Cs, Ds = np.asarray(sC[1]), np.asarray(sD[1])
        ov.append(float((np.sign(Cs) * np.sign(Ds)).mean()))
        hc.append(float((np.asarray(cm.pool_j1(Cs)) @ Wout).argmax(1).__eq__(y).mean()))
    return pD, head_D, float(np.mean(hc)), float(np.mean(ov))


def run(cfg, ds, boost, q, seed, args):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = cm.build_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)
    for _ in range(args.epochs):
        kmap = compute_kappa_map(trainer.orchestrator, state, ds, cfg, key, boost, q, args.kappa_batches)
        trainer = set_kappa(trainer, kmap)
        trainer, key = _train_epoch(trainer, ds, key, cfg["kernel_decay_rate"], args.max_batches)
    return measure(trainer.orchestrator, state, ds, cfg, key, args.probe_epochs, args.probe_train_batches)


def _train_epoch(trainer, ds, key, dr, max_batches):
    for i, (xb, yb) in enumerate(ds):
        if max_batches is not None and i >= max_batches:
            break
        key = trainer.train_step(cm.to_hwc(xb), yb, key)
    if dr > 0:
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(path, trainer.orchestrator, path(trainer.orchestrator) * (1 - dr))
    return trainer, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--boosts", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--qs", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--kappa-batches", type=int, default=32)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.boosts = [0.0, 1.0]; args.qs = [0.5]; args.epochs = 2
        args.kappa_batches = 3; args.probe_epochs = 3; args.probe_train_batches = 6; args.max_batches = 6

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    # boost=0 is the baseline regardless of q -> run once
    combos = [(0.0, args.qs[0])] + [(b, q) for b in args.boosts if b != 0.0 for q in args.qs]
    results = {"config": "best_channel_entropy", "seeds": args.seeds, "cells": {}}
    for boost, q in combos:
        tag = "baseline" if boost == 0.0 else f"boost{boost}_q{q}"
        pDs, hDs, hCs, ovs = [], [], [], []
        for seed in args.seeds:
            pD, hD, hC, ov = run(cfg, ds, boost, q, seed, args)
            pDs.append(pD); hDs.append(hD); hCs.append(hC); ovs.append(ov)
        results["cells"][tag] = {"boost": boost, "q": q,
                                 "probe_D_mean": float(np.mean(pDs)), "probe_D_std": float(np.std(pDs)),
                                 "head_D_mean": float(np.mean(hDs)), "head_C_mean": float(np.mean(hCs)),
                                 "omega_CD_mean": float(np.mean(ovs)), "omega_CD_std": float(np.std(ovs)),
                                 "probe_D_seeds": pDs}
        print(f"  [{tag:16s}] probe_D={np.mean(pDs):.3f} head_D={np.mean(hDs):.3f} "
              f"head_C={np.mean(hCs):.3f} Omega_CD={np.mean(ovs):.3f} ({cm.fmt(time.time()-t0)})")

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "gated_kappa.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}. Reference: baseline probe_D ~0.45, BPTT ~0.51. (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
