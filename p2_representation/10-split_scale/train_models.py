"""Train + serialize the split-vs-scale models (exp 10).

For each config in arch.CONFIGS and seed, train the standard local-rule backbone
(DynamicalTrainer; perceptron + entropy rules) with the W_in/W_back channel masking,
matching the exp-3 model-A recipe (20 epochs, kernel_decay_rate). Logs nominal and
effective active params per config; asserts the masked channels stay exactly zero after
training. Serializes models/<name>_seed<seed>.eqx (+ meta.json).

Run (cluster):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/10-split_scale/train_models.py
Smoke:  python p2_representation/10-split_scale/train_models.py --smoke
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
sys.path.insert(0, str(REPO / "p2_representation" / "10-split_scale"))

import common as cm
import arch as A

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"


def _train_epoch(trainer, ds, key, decay_rate, max_batches):
    for i, (xb, yb) in enumerate(ds):
        if max_batches is not None and i >= max_batches:
            break
        key = trainer.train_step(cm.to_hwc(xb), yb, key)
    if decay_rate > 0:
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay_rate))
    return trainer, key


def train_config(cfg, ds, name, seed, args):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = A.build_model(cfg, mk, name)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)  # clamped = cfg default -> C
    for _ in range(args.epochs):
        trainer, key = _train_epoch(trainer, ds, key, cfg["kernel_decay_rate"], args.max_batches)
    return trainer.orchestrator


def assert_masks_intact(orch, name):
    C = A.channels_of(name)
    I, L, _ = A.group_indices(name)
    win = np.asarray(orch.lmap[1][0].kernel); wb = np.asarray(orch.lmap[1][2].W)
    nonI = [c for c in range(C) if c not in set(I.tolist())]
    nonL = [c for c in range(C) if c not in set(L.tolist())]
    if nonI:
        assert np.abs(win[:, :, :, nonI]).sum() == 0.0, f"{name}: W_in leaked into non-I channels"
    if nonL:
        assert np.abs(wb[:, nonL]).sum() == 0.0, f"{name}: W_back leaked into non-L channels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--names", type=str, nargs="+", default=A.NAMES)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 1; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("Parameter accounting (nominal must match within each C):")
    param_meta = {}
    for name in args.names:
        _, o = A.build_model(cfg, jax.random.PRNGKey(0), name)
        param_meta[name] = A.param_report(o, name)

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        for name in args.names:
            orch = train_config(cfg, ds, name, seed, args)
            assert_masks_intact(orch, name)
            eqx.tree_serialise_leaves(MODELS_DIR / f"{name}_seed{seed}.eqx", orch)
            print(f"  [{name:21s}] trained + masks intact + serialized  ({cm.fmt(time.time() - t0)})")

    meta = {"config": "best_channel_entropy", "seeds": args.seeds, "names": args.names,
            "configs": {n: A.CONFIGS[n] for n in args.names},
            "epochs": args.epochs, "param_report": param_meta}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(args.seeds) * len(args.names)} models to {MODELS_DIR}  "
          f"(total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
