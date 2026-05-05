#!/usr/bin/env python3
"""CIFAR10 training with the sparse FC architecture (adapted from MNIST tutorial).

Replaces the conv architecture with SparseFullyConnected + SparseRecurrentDiscrete
(2000 hidden units, 99% sparsity) — the same architecture that achieves SOTA on MNIST.

Key differences from our failed conv run:
  - dim_data=3072 (32×32×3 flattened, no conv)
  - weight decay built into every module (prevents weight explosion)
  - optimizer lr=1.0, modules handle their own lr/wd internally
  - full test-set evaluation via ds.iter_test()

Run on w01:
  python cifar10_sparse_fc.py --rule current
  python cifar10_sparse_fc.py --rule long_warmup
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.fully_connected import (
    FrozenRescaledFullyConnected,
    FullyConnected,
    SparseFullyConnected,
)
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import SparseRecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer


# ---------------------------------------------------------------------------
# Default hyperparameters (mirrors MNIST tutorial, adapted for CIFAR10)
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "dim_hidden": 2000,
    "sparsity": 0.99,
    "sparsity_win": 0.9,
    "strength_forth": 5.0,
    "strength_back": 1.3,
    "j_d": 0.95,
    "threshold_in": 1.78,
    "threshold_j": 1.78,
    "threshold_out": 7.0,
    "threshold_back": 0.0,
    # optimizer — same as tutorial; lr rescaled by sparsity inside build_model
    "lr_win_raw": 0.159,
    "lr_j_raw": 0.058,
    "lr_wout": 0.17,
    "wd_win": 0.01,
    "wd_j": 6e-5,
    "wd_wout": 0.02,
    # training
    "batch_size": 16,
    "n_epochs": 20,
    "warmup_n_iter": 1,
    "train_clamped_n_iter": 5,
    "train_free_n_iter": 5,
    "eval_n_iter": 5,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(
    seed: int,
    dim_hidden: int,
    sparsity: float,
    sparsity_win: float,
    strength_forth: float,
    strength_back: float,
    j_d: float,
    threshold_in: float,
    threshold_j: float,
    threshold_out: float,
    threshold_back: float,
    lr_win: float,
    lr_j: float,
    lr_wout: float,
    wd_win: float,
    wd_j: float,
    wd_wout: float,
    dim_data: int = 3072,
    num_labels: int = 10,
) -> tuple[SequentialState, SequentialOrchestrator]:
    state = SequentialState((dim_data, dim_hidden, num_labels))
    keys = jax.random.split(jax.random.key(seed), 4)

    layer_map = LayerMap.from_dict({
        1: {
            0: SparseFullyConnected(
                in_features=dim_data,
                out_features=dim_hidden,
                strength=strength_forth,
                threshold=threshold_in,
                sparsity=sparsity_win,
                key=keys[0],
                lr=lr_win,
                weight_decay=wd_win,
            ),
            1: SparseRecurrentDiscrete(
                features=dim_hidden,
                j_d=j_d,
                sparsity=sparsity,
                threshold=threshold_j,
                key=keys[1],
                lr=lr_j,
                weight_decay=wd_j,
            ),
            2: FrozenRescaledFullyConnected(
                in_features=num_labels,
                out_features=dim_hidden,
                strength=strength_back,
                threshold=threshold_back,
                key=keys[2],
            ),
        },
        2: {
            1: FullyConnected(
                in_features=dim_hidden,
                out_features=num_labels,
                strength=1.0,
                threshold=threshold_out,
                key=keys[3],
                lr=lr_wout,
                weight_decay=wd_wout,
            ),
            2: OutputLayer(),
        },
    })
    return state, SequentialOrchestrator(layers=layer_map)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(DEFAULTS)
    # CLI overrides
    for k in ["dim_hidden", "n_epochs", "batch_size", "threshold_in", "threshold_j",
              "threshold_out", "lr_win_raw", "lr_j_raw", "lr_wout",
              "wd_win", "wd_j", "wd_wout", "warmup_n_iter"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    if args.rule == "long_warmup":
        cfg["warmup_n_iter"] = args.warmup_n_iter if args.warmup_n_iter is not None else 10

    # Rescale lr by sparsity (same as tutorial)
    lr_win = cfg["lr_win_raw"] / float(jnp.sqrt(1.0 - cfg["sparsity_win"]))
    lr_j   = cfg["lr_j_raw"]   / float(jnp.sqrt(1.0 - cfg["sparsity"]))

    state, orch = build_model(
        seed=args.seed,
        dim_hidden=cfg["dim_hidden"],
        sparsity=cfg["sparsity"],
        sparsity_win=cfg["sparsity_win"],
        strength_forth=cfg["strength_forth"],
        strength_back=cfg["strength_back"],
        j_d=cfg["j_d"],
        threshold_in=cfg["threshold_in"],
        threshold_j=cfg["threshold_j"],
        threshold_out=cfg["threshold_out"],
        threshold_back=cfg["threshold_back"],
        lr_win=lr_win,
        lr_j=lr_j,
        lr_wout=cfg["lr_wout"],
        wd_win=cfg["wd_win"],
        wd_j=cfg["wd_j"],
        wd_wout=cfg["wd_wout"],
    )

    optimizer = optax.sgd(learning_rate=1.0)
    opt_state = optimizer.init(eqx.filter(orch, eqx.is_inexact_array))

    trainer = DynamicalTrainer(
        orchestrator=orch,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        warmup_n_iter=cfg["warmup_n_iter"],
        train_clamped_n_iter=cfg["train_clamped_n_iter"],
        train_free_n_iter=cfg["train_free_n_iter"],
        eval_n_iter=cfg["eval_n_iter"],
    )

    ds = Cifar10(
        batch_size=cfg["batch_size"],
        linear_projection=None,
        label_mode="pm1",
        x_transform="identity",
    )
    ds.build(key=jax.random.key(args.seed))

    key = jax.random.key(args.seed + 1)
    log: list[dict] = []
    t0 = time.time()

    print(f"Rule: {args.rule}  warmup={cfg['warmup_n_iter']}  "
          f"thr_in={cfg['threshold_in']}  thr_j={cfg['threshold_j']}  "
          f"dim_hidden={cfg['dim_hidden']}  epochs={cfg['n_epochs']}")
    print(f"lr_win={lr_win:.4f}  lr_j={lr_j:.4f}  lr_wout={cfg['lr_wout']}")

    for epoch in range(cfg["n_epochs"] + 1):
        # Train (skip epoch 0 — just evaluate initial state)
        if epoch > 0:
            for xb, yb in ds:
                key = trainer.train_step(xb, yb, key)

        # Full test-set eval
        test_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(xb, yb, key)
            test_accs.append(float(metrics["accuracy"]))

        # Full train-set eval (expensive but honest)
        train_accs = []
        for xb, yb in ds:
            key, metrics = trainer.eval_step(xb, yb, key)
            train_accs.append(float(metrics["accuracy"]))

        entry = {
            "epoch": epoch,
            "train_acc": float(np.mean(train_accs)) if train_accs else float("nan"),
            "test_acc":  float(np.mean(test_accs))  if test_accs  else float("nan"),
            "elapsed_s": time.time() - t0,
        }
        log.append(entry)
        print(f"  epoch={epoch:3d}  train={entry['train_acc']:.4f}  "
              f"test={entry['test_acc']:.4f}  t={entry['elapsed_s']:.0f}s")

    return {
        "rule": args.rule,
        "seed": args.seed,
        "config": {**cfg, "lr_win_scaled": lr_win, "lr_j_scaled": lr_j},
        "log": log,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR10 with sparse FC architecture.")
    p.add_argument("--rule",         choices=["current", "long_warmup"], required=True)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--n-epochs",     type=int,   default=None, dest="n_epochs")
    p.add_argument("--batch-size",   type=int,   default=None, dest="batch_size")
    p.add_argument("--dim-hidden",   type=int,   default=None, dest="dim_hidden")
    p.add_argument("--threshold-in", type=float, default=None, dest="threshold_in")
    p.add_argument("--threshold-j",  type=float, default=None, dest="threshold_j")
    p.add_argument("--threshold-out",type=float, default=None, dest="threshold_out")
    p.add_argument("--lr-win",       type=float, default=None, dest="lr_win_raw")
    p.add_argument("--lr-j",         type=float, default=None, dest="lr_j_raw")
    p.add_argument("--lr-wout",      type=float, default=None, dest="lr_wout")
    p.add_argument("--wd-win",       type=float, default=None, dest="wd_win")
    p.add_argument("--wd-j",         type=float, default=None, dest="wd_j")
    p.add_argument("--wd-wout",      type=float, default=None, dest="wd_wout")
    p.add_argument("--warmup-n-iter",type=int,   default=None, dest="warmup_n_iter")
    p.add_argument("--output",       type=str,   default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.output) if args.output else \
        Path(f"results/cifar10_sparse_fc/cifar10_sparse_fc_{args.rule}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_training(args)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
