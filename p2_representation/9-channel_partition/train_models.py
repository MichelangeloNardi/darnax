"""Train and SERIALIZE the channel-partition models (exp 9).

For each partition tag (baseline + 4 disjoint splits) and seed, train the standard
local-rule backbone (DynamicalTrainer; perceptron + entropy rules) with the W_in /
W_back channel masking from partition.py, exactly matching the exp-3 model-A recipe
(same epochs, same kernel_decay_rate) so the BASELINE reproduces exp-3 A (~0.45 D-probe)
as a within-run control. Then diagnostics.py / group_diagnostics.py / fullspin_importance.py
load the models and compute everything offline.

Saves models/<tag>_seed<seed>.eqx (+ meta.json with the param-parity report).

Run (cluster):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/9-channel_partition/train_models.py
Smoke:  python p2_representation/9-channel_partition/train_models.py --smoke
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import partition as P

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"


def _train_epoch(trainer, ds, key, decay_rate, max_batches):
    """cm.train_epoch with an optional per-epoch batch cap (smoke only)."""
    for i, (xb, yb) in enumerate(ds):
        if max_batches is not None and i >= max_batches:
            break
        key = trainer.train_step(cm.to_hwc(xb), yb, key)
    if decay_rate > 0:
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay_rate))
    return trainer, key


def train_partition(cfg, ds, tag, seed, args):
    """Standard local-rule training for one partition (clamped rollout = C)."""
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = P.build_partitioned_model(cfg, mk, tag)
    opt, opt_state = cm.make_optimizer(orch, cfg)         # W_back frozen; win/j1/wout trained
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)  # clamped = cfg default -> C
    for _ in range(args.epochs):
        trainer, key = _train_epoch(trainer, ds, key, cfg["kernel_decay_rate"], args.max_batches)
    return trainer.orchestrator


def assert_masks_intact(orch, tag):
    """The masked W_in / W_back channels must still be exactly zero after training."""
    I, L, _ = P.group_indices(tag)
    win = np.asarray(orch.lmap[1][0].kernel)
    wb = np.asarray(orch.lmap[1][2].W)
    nonI = [c for c in range(P.C) if c not in set(I.tolist())]
    nonL = [c for c in range(P.C) if c not in set(L.tolist())]
    if nonI:
        assert np.abs(win[:, :, :, nonI]).sum() == 0.0, f"{tag}: W_in leaked into non-I channels"
    if nonL:
        assert np.abs(wb[:, nonL]).sum() == 0.0, f"{tag}: W_back leaked into non-L channels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--tags", type=str, nargs="+", default=P.TAGS)
    ap.add_argument("--epochs", type=int, default=20)         # matches exp-3 model A
    ap.add_argument("--max-batches", type=int, default=None)  # smoke trims the dataset
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 1; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # param parity report on a fresh build of each tag
    print("Parameter accounting (nominal must equal baseline):")
    _, orch0 = P.build_partitioned_model(cfg, jax.random.PRNGKey(0), "baseline")
    base_nom = P.nominal_trainable_params(orch0)
    param_meta = {}
    for tag in args.tags:
        _, o = P.build_partitioned_model(cfg, jax.random.PRNGKey(0), tag)
        param_meta[tag] = P.param_report(o, tag, base_nom)

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        for tag in args.tags:
            orch = train_partition(cfg, ds, tag, seed, args)
            assert_masks_intact(orch, tag)
            eqx.tree_serialise_leaves(MODELS_DIR / f"{tag}_seed{seed}.eqx", orch)
            print(f"  [{tag:9s}] trained + masks intact + serialized  ({cm.fmt(time.time() - t0)})")

    meta = {"config": "best_channel_entropy", "seeds": args.seeds, "tags": args.tags,
            "partitions": {t: P.PARTITIONS[t] for t in args.tags},
            "epochs": args.epochs, "baseline_nominal_trainable": base_nom,
            "param_report": param_meta}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(args.seeds) * len(args.tags)} models to {MODELS_DIR}  "
          f"(total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
