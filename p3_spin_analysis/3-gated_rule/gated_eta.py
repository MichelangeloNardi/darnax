"""Exp p3-3 fallback — per-channel learning-rate (eta) boost on dangerous channels.

The kappa sweep (gated_kappa.py) left probe_D flat: raising the gate threshold changes how
OFTEN the rule fires, not the update MAGNITUDE. This fallback boosts the update magnitude on
dangerous channels instead. The recurrent kernel is shared across spatial positions, so eta is
naturally PER-OUTPUT-CHANNEL (16), not per-spin: EtaConv2DRecurrentDiscrete scales its kernel
update by a per-output-channel eta_map; effective lr on channel c = lr_j * eta_map[c].

Each epoch: meanfield_c = mean over (n,H,W) of |field_C| for channel c; dangerous = bottom-q
channels; eta_map[c] = 1 + boost on dangerous, else 1. boost=0 recovers the standard rule.
eta_map is set as a leaf each epoch (the optimizer doesn't touch it: backward returns 0 for it).

Baseline conv C=16 at best_channel_entropy_cfg; metric probe_D (~0.45; ref ~0.51). Sweep (boost, q).

Smoke:  uv run python p3_spin_analysis/3-gated_rule/gated_eta.py --smoke
Cluster: XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python \
  p3_spin_analysis/3-gated_rule/gated_eta.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
H, W, C, KSIZE, POOL = cm.H, cm.W, cm.C, cm.KSIZE, cm.POOL


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


GK = _load(HERE / "gated_kappa.py", "gated_kappa")  # reuse probe_D, _train_epoch


class EtaConv2DRecurrentDiscrete(Conv2DRecurrentDiscrete):
    """Recurrent conv whose kernel UPDATE is scaled per output channel by eta_map (lr boost)."""

    eta_map: Array  # (channels,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_map = jnp.ones(self.channels, dtype=self.kernel.dtype)

    def backward(self, x, y, y_hat, gate=None):
        upd = super().backward(x, y, y_hat, gate)
        scaled = upd.kernel * self.eta_map[None, None, None, :]
        return eqx.tree_at(lambda m: m.kernel, upd, scaled)


def build_eta_model(cfg, key):
    keys = jax.random.split(key, 5)
    j1 = EtaConv2DRecurrentDiscrete(
        channels=C, kernel_size=KSIZE, groups=1, j_d=cfg["j_d"], threshold=cfg["threshold_j"],
        key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
        entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0)
    layer_map = LayerMap.from_dict({
        1: {0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE, threshold=cfg["threshold_win"],
                      strength=1.0, key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.0),
            1: j1,
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2])},
        2: {1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10, strength=1.0,
                               threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer()},
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=layer_map)


def compute_eta_map(orch, state_tmpl, ds, cfg, key, boost, q, n_batches):
    """(C,) per-channel lr multiplier: 1 + boost on bottom-q-quantile mean|field_C| channels."""
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll = eqx.filter_jit(cm.make_rollout(warmup, clamped, free))
    acc = np.zeros(C, np.float64); k = key
    for i, (xb, yb) in enumerate(ds):
        if i >= n_batches:
            break
        s, k = roll(orch, state_tmpl.init(cm.to_hwc(xb), yb), k)
        acc += np.abs(np.asarray(s.fields[1])).mean(axis=(0, 1, 2))   # per channel
    meanfield = acc / max(n_batches, 1)                               # (C,)
    if boost == 0.0:
        return jnp.ones(C, jnp.float32)
    thr = np.quantile(meanfield, q)
    return jnp.asarray(1.0 + boost * (meanfield < thr).astype(np.float32), jnp.float32)


def run(cfg, ds, boost, q, seed, args):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_eta_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)
    for _ in range(args.epochs):
        emap = compute_eta_map(trainer.orchestrator, state, ds, cfg, key, boost, q, args.kappa_batches)
        trainer.orchestrator = eqx.tree_at(lambda o: o.lmap[1][1].eta_map, trainer.orchestrator, emap)
        trainer, key = GK._train_epoch(trainer, ds, key, cfg["kernel_decay_rate"], args.max_batches)
    return GK.measure(trainer.orchestrator, state, ds, cfg, key, args.probe_epochs, args.probe_train_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--boosts", type=float, nargs="+", default=[0.0, 1.0, 3.0, 6.0])
    ap.add_argument("--qs", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--kappa-batches", type=int, default=32)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.boosts = [0.0, 3.0]; args.qs = [0.5]; args.epochs = 2
        args.kappa_batches = 3; args.probe_epochs = 3; args.probe_train_batches = 6; args.max_batches = 6

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    combos = [(0.0, args.qs[0])] + [(b, q) for b in args.boosts if b != 0.0 for q in args.qs]
    results = {"config": "best_channel_entropy", "seeds": args.seeds, "cells": {}}
    for boost, q in combos:
        tag = "baseline" if boost == 0.0 else f"eta{boost}_q{q}"
        pDs, hDs, hCs, ovs = [], [], [], []
        for seed in args.seeds:
            pD, hD, hC, ov = run(cfg, ds, boost, q, seed, args)
            pDs.append(pD); hDs.append(hD); hCs.append(hC); ovs.append(ov)
        results["cells"][tag] = {"boost": boost, "q": q, "probe_D_mean": float(np.mean(pDs)),
                                 "probe_D_std": float(np.std(pDs)), "head_D_mean": float(np.mean(hDs)),
                                 "head_C_mean": float(np.mean(hCs)), "omega_CD_mean": float(np.mean(ovs)),
                                 "probe_D_seeds": pDs}
        print(f"  [{tag:14s}] probe_D={np.mean(pDs):.3f} head_D={np.mean(hDs):.3f} "
              f"head_C={np.mean(hCs):.3f} Omega_CD={np.mean(ovs):.3f} ({cm.fmt(time.time()-t0)})")

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("eta_smoke.json" if args.smoke else "gated_eta.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}. Reference: baseline probe_D ~0.45, BPTT ~0.51. (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
