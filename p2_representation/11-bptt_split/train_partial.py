"""Exp 11 part 2 — partial-overlap split architecture (local rule).

C=24, three 8-channel groups: `both` (input + label), `input_only` (W_in only),
`label_only` (W_back only). W_in reaches both+input_only (16 channels, original input
capacity); W_back reaches both+label_only (16 channels). All channels participate in J1
and W_out. Trained with the standard local rule (DynamicalTrainer), exp-3 model-A recipe.
Serializes models/partial_C24_seed<s>.eqx. diagnostics.py compares it to standard_C24 and
the strict split_C24 (loaded from ../10-split_scale/models).

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/11-bptt_split/train_partial.py
Smoke:  python p2_representation/11-bptt_split/train_partial.py --smoke
"""
from __future__ import annotations

import argparse
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
import splitarch as S

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


def assert_masks_intact(orch):
    C = S.PARTIAL["C"]
    win = np.asarray(orch.lmap[1][0].kernel); wb = np.asarray(orch.lmap[1][2].W)
    nonI = [c for c in range(C) if c not in set(S.PARTIAL["I_idx"])]
    nonL = [c for c in range(C) if c not in set(S.PARTIAL["L_idx"])]
    assert np.abs(win[:, :, :, nonI]).sum() == 0.0, "W_in leaked outside I_idx"
    assert np.abs(wb[:, nonL]).sum() == 0.0, "W_back leaked outside L_idx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 1; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"partial_C24 groups: both={S.PARTIAL['groups']['both']} "
          f"input_only={S.PARTIAL['groups']['input_only']} "
          f"label_only={S.PARTIAL['groups']['label_only']} | "
          f"W_in->{len(S.PARTIAL['I_idx'])}ch  W_back->{len(S.PARTIAL['L_idx'])}ch")

    for seed in args.seeds:
        key = jax.random.PRNGKey(seed)
        key, mk = jax.random.split(key)
        state, orch = S.build_partial(cfg, mk)
        opt, opt_state = cm.make_optimizer(orch, cfg)
        trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)
        for _ in range(args.epochs):
            trainer, key = _train_epoch(trainer, ds, key, cfg["kernel_decay_rate"], args.max_batches)
        assert_masks_intact(trainer.orchestrator)
        eqx.tree_serialise_leaves(MODELS_DIR / f"partial_C24_seed{seed}.eqx", trainer.orchestrator)
        print(f"  [partial_C24 s{seed}] trained + masks intact + serialized  ({cm.fmt(time.time() - t0)})")
    print(f"\nSaved {len(args.seeds)} partial_C24 models  (total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
