#!/usr/bin/env python3
"""Full CIFAR10 training run.

Trains the convolutional Darnax network on the full CIFAR10 dataset using
the DynamicalTrainer (warmup → clamped → free, local perceptron rule).

Compares current rule (n_warmup=1) vs long_warmup rule (n_warmup=10).
Logs train accuracy, test accuracy every --eval-every batches.

Designed to run on w01 (GPU). Uses batch training, not single-image.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.pooling import GlobalMajorityPooling, GlobalUnpooling
from darnax.modules.fully_connected import FullyConnected, FrozenRescaledFullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer


# ---------------------------------------------------------------------------
# Model (same architecture as gap experiments)
# ---------------------------------------------------------------------------

def build_model(
    seed: int,
    threshold: float = 1.7,
    in_channels: int = 3,
    spatial_size: int = 32,
    n_channels: int = 64,
    num_labels: int = 10,
    input_kernel_size: int = 7,
    recur_kernel_size: int = 7,
    j_d_conv: float = 0.9,
    j_d_fc: float = 0.9,
):
    keys = jax.random.split(jax.random.key(seed), 5)
    s = spatial_size
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(
                in_channels=in_channels, out_channels=n_channels,
                kernel_size=input_kernel_size, threshold=threshold,
                strength=1.0, key=keys[0], padding_mode="constant",
            ),
            1: Conv2DRecurrentDiscrete(
                channels=n_channels, kernel_size=recur_kernel_size,
                groups=n_channels, j_d=j_d_conv, threshold=threshold,
                padding_mode="constant", key=keys[1],
                lr=1.0, weight_decay=0.0,
            ),
            2: GlobalUnpooling(strength=1.0, axis=(1, 2)),
        },
        2: {
            1: GlobalMajorityPooling(strength=1.0, axis=(1, 2)),
            2: RecurrentDiscrete(features=n_channels, j_d=j_d_fc, threshold=threshold, key=keys[2]),
            3: FrozenRescaledFullyConnected(
                in_features=num_labels, out_features=n_channels,
                strength=1.0, threshold=0.0, key=keys[3],
            ),
        },
        3: {
            2: FullyConnected(
                in_features=n_channels, out_features=num_labels,
                strength=1.0, threshold=threshold, key=keys[4],
            ),
            3: OutputLayer(),
        },
    })
    state = SequentialState([
        (s, s, in_channels),
        (s, s, n_channels),
        (n_channels,),
        (num_labels,),
    ])
    orch = SequentialOrchestrator(layers=layer_map)
    return state, orch


def build_optimizer(orch, lr):
    params, _ = eqx.partition(orch, eqx.is_inexact_array)
    labels = jtu.tree_map(lambda _: "frozen", params, is_leaf=eqx.is_array)

    def label_module(m, lbl):
        return jtu.tree_map(lambda _: lbl, m, is_leaf=eqx.is_array)

    for (i, j), lbl in [((1, 0), "w_in"), ((1, 1), "j_conv"), ((2, 2), "j_fc"), ((3, 2), "w_out")]:
        labels = eqx.tree_at(
            lambda m, _i=i, _j=j: m.lmap[_i][_j], labels,
            replace=label_module(params.lmap[i][j], lbl),
        )
    optimizer = optax.multi_transform(
        {"frozen": optax.sgd(0.0), "w_in": optax.sgd(lr), "j_conv": optax.sgd(lr),
         "j_fc": optax.sgd(lr), "w_out": optax.sgd(lr)},
        labels,
    )
    opt_state = optimizer.init(eqx.filter(orch, eqx.is_inexact_array))
    return optimizer, opt_state


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    rule_name: str,
    seed: int,
    threshold: float,
    learning_rate: float,
    n_warmup: int,         # 1 for current, 10 for long_warmup
    n_clamped: int,
    n_free: int,
    n_eval_free: int,
    batch_size: int,
    n_epochs: int,
    eval_every: int,       # eval every N batches
    dataset_batch_size: int,
) -> dict[str, Any]:

    state_template, orch = build_model(seed=seed, threshold=threshold)
    optimizer, opt_state = build_optimizer(orch, lr=learning_rate)

    trainer = DynamicalTrainer(
        orchestrator=orch,
        state=state_template,
        optimizer=optimizer,
        optimizer_state=opt_state,
        warmup_n_iter=n_warmup,
        train_clamped_n_iter=n_clamped,
        train_free_n_iter=n_free,
        eval_n_iter=n_eval_free,
    )

    # Dataset
    data_train = Cifar10(batch_size=batch_size, linear_projection=None,
                         label_mode="pm1", x_transform="identity")
    data_train.build(key=jax.random.PRNGKey(seed))
    data_test = Cifar10(batch_size=batch_size, linear_projection=None,
                        label_mode="pm1", x_transform="identity")
    data_test.build(key=jax.random.PRNGKey(seed + 100))

    def reshape_x(x):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 32, 32, 3)
        return x

    rng = jax.random.PRNGKey(seed)
    log: list[dict] = []
    batch_idx = 0
    t0 = time.time()

    for epoch in range(n_epochs):
        for x, y in data_train:
            x = reshape_x(x)
            # Train one image at a time — local plasticity rules are attractor-based
            # and cannot meaningfully average gradients across multiple patterns at once.
            for i in range(x.shape[0]):
                rng = trainer.train_step(x[i:i+1], y[i:i+1], rng)
            batch_idx += 1

            if batch_idx % eval_every == 0:
                # Evaluate on a few test batches
                test_accs = []
                train_accs = []
                for x_e, y_e in data_test:
                    x_e = reshape_x(x_e)
                    for i in range(x_e.shape[0]):
                        rng, metrics = trainer.eval_step(x_e[i:i+1], y_e[i:i+1], rng)
                        test_accs.append(float(metrics["accuracy"]))
                    if len(test_accs) >= 80:
                        break
                for x_e, y_e in data_train:
                    x_e = reshape_x(x_e)
                    for i in range(x_e.shape[0]):
                        rng, metrics = trainer.eval_step(x_e[i:i+1], y_e[i:i+1], rng)
                        train_accs.append(float(metrics["accuracy"]))
                    if len(train_accs) >= 80:
                        break

                entry = {
                    "batch": batch_idx,
                    "epoch": epoch,
                    "train_acc": float(np.mean(train_accs)),
                    "test_acc":  float(np.mean(test_accs)),
                    "elapsed_s": time.time() - t0,
                }
                log.append(entry)
                print(f"  [{rule_name}] epoch={epoch} batch={batch_idx:4d}  "
                      f"train={entry['train_acc']:.3f}  test={entry['test_acc']:.3f}  "
                      f"t={entry['elapsed_s']:.0f}s")

    return {
        "rule": rule_name,
        "seed": seed,
        "config": {
            "threshold": threshold,
            "learning_rate": learning_rate,
            "n_warmup": n_warmup,
            "n_clamped": n_clamped,
            "n_free": n_free,
            "n_eval_free": n_eval_free,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        },
        "log": log,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Full CIFAR10 training: current vs long_warmup.")
    p.add_argument("--rule",          choices=["current", "long_warmup"], required=True)
    p.add_argument("--threshold",     type=float, default=1.7)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-warmup",      type=int,   default=None,
                   help="Warmup steps. Default: 1 for current, 10 for long_warmup.")
    p.add_argument("--n-clamped",     type=int,   default=5)
    p.add_argument("--n-free",        type=int,   default=5)
    p.add_argument("--n-eval-free",   type=int,   default=10)
    p.add_argument("--batch-size",    type=int,   default=32,
                   help="Training batch size. Default 32.")
    p.add_argument("--n-epochs",      type=int,   default=3)
    p.add_argument("--eval-every",    type=int,   default=100,
                   help="Evaluate every N weight updates.")
    p.add_argument("--dataset-batch-size", type=int, default=32)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--output",        type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Default warmup by rule
    if args.n_warmup is None:
        args.n_warmup = 10 if args.rule == "long_warmup" else 1

    out_path = Path(args.output) if args.output else \
        Path(f"results/cifar10/cifar10_{args.rule}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rule: {args.rule}  warmup={args.n_warmup}  lr={args.learning_rate}  "
          f"threshold={args.threshold}  batch={args.batch_size}  epochs={args.n_epochs}")

    result = run_training(
        rule_name=args.rule,
        seed=args.seed,
        threshold=args.threshold,
        learning_rate=args.learning_rate,
        n_warmup=args.n_warmup,
        n_clamped=args.n_clamped,
        n_free=args.n_free,
        n_eval_free=args.n_eval_free,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_every=args.eval_every,
        dataset_batch_size=args.dataset_batch_size,
    )

    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
