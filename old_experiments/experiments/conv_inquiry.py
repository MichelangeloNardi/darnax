"""
Conv Inquiry Experiment: Single-Layer Convolutional Input & Recurrent.

Four conditions:
1. Conv Win + FC J (frozen Win/J: lr_win=0, lr_j=0)
2. Conv Win + Conv J (frozen Win/J: lr_win=0, lr_j=0)
3. Conv Win + FC J (learning Win: lr_win>0, lr_j=0)
4. Conv Win + Conv J (learning both: lr_win>0, lr_j>0)

Vary kernel_size in [3, 5, 7] and collect test accuracy + param counts.
Hyperparameters stay as close to FC baseline as possible.
"""

import sys
from pathlib import Path
import os
import json
import time
from datetime import datetime
from pathlib import Path as _Path
from pathlib import Path as _Path

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import copy
from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pprint

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.fully_connected import (
    FullyConnected,
    FrozenRescaledFullyConnected,
    SparseFullyConnected,
)
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import SparseRecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer
from darnax.utils.typing import PyTree
from darnax.modules.conv.conv_adapters import ConvAdapter, ConvRecurrentDiscrete


# ============================================================================
# Base Configuration (close to tutorial 09 FC baseline)
# ============================================================================

BASE_PARAMS = {
    "master_seed": 0,
    "epochs":7,
    "model": {
        "kwargs": {
            "seed": 44,
            "dim_data": 3072,  # CIFAR10: 32x32x3
            "dim_hidden": (32, 32, 8),  # H, W, C for conv; flat = 8192
            "num_labels": 10,
            "sparsity": 0.99,
            "sparsity_win": 0.9,
            "strength_forth": 5.0,
            "strength_back": 1.3,
            "j_d": 0.95,
            "threshold_in": 1.78,
            "threshold_out": 7.0,
            "threshold_j": 1.78,
            "threshold_back": 0.0,
            "win_type": "conv",
            "j_type": "conv",
            "kernel_size": 3,  # Will be overridden
        }
    },
    "data": {
        "kwargs": {
            "batch_size": 16,
            "linear_projection": None,  # CIFAR10: no flatten param
            "num_images_per_class": None,
            "label_mode": "pm1",
            "x_transform": "identity",
        },
    },
    "optimizer": {
        "learning_rate_win": 0.159,
        "learning_rate_j": 0.058,
        "learning_rate_wout": 0.17,
        "weight_decay_win": 0.01,
        "weight_decay_j": 0.00006,
        "weight_decay_wout": 0.02,
    },
    "trainer": {
        "kwargs": {
            "warmup_n_iter": 1,
            "train_clamped_n_iter": 5,
            "train_free_n_iter": 5,
            "eval_n_iter": 5,
        },
    },
}


# ============================================================================
# Model Builder (Supports Conv Adapters)
# ============================================================================

import math


def build_model(
    seed: int,
    dim_data: int,
    dim_hidden,
    sparsity: float,
    sparsity_win: float,
    num_labels: int,
    strength_forth: float,
    strength_back: float,
    threshold_in: float,
    threshold_out: float,
    threshold_j: float,
    threshold_back: float,
    j_d: float,
    win_type: str = "conv",
    j_type: str = "conv",
    kernel_size: int = 3,
    c_in: int = 1,  # Input channels (1 for MNIST, 3 for CIFAR10)
):
    """Build model with optional conv adapters."""
    if isinstance(dim_hidden, (tuple, list)):
        h_hidden, w_hidden, c_hidden = dim_hidden
        dim_hidden_flat = h_hidden * w_hidden * c_hidden
    else:
        dim_hidden_flat = dim_hidden
        side = int(math.sqrt(dim_hidden_flat))
        h_hidden, w_hidden, c_hidden = side, side, 1

    state = SequentialState((dim_data, dim_hidden_flat, num_labels))
    master_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(master_key, num=5)

    # Win
    if win_type == "fc":
        win = SparseFullyConnected(
            in_features=dim_data,
            out_features=dim_hidden_flat,
            strength=strength_forth,
            threshold=threshold_in,
            sparsity=sparsity_win,
            key=keys[0],
        )
    else:
        # Compute spatial dimensions from dim_data and c_in
        spatial_dim = int((dim_data / c_in) ** 0.5)
        win = ConvAdapter(
            h_in=spatial_dim,
            w_in=spatial_dim,
            c_in=c_in,
            c_out=c_hidden,
            kernel_size=kernel_size,
            h_out=h_hidden,
            w_out=w_hidden,
            strength=strength_forth,
            threshold=threshold_in,
            key=keys[0],
        )

    # J
    if j_type == "fc":
        j_mod = SparseRecurrentDiscrete(
            features=dim_hidden_flat,
            j_d=j_d,
            sparsity=sparsity,
            threshold=threshold_j,
            key=keys[1],
        )
    else:
        j_mod = ConvRecurrentDiscrete(
            h=h_hidden,
            w=w_hidden,
            channels=c_hidden,
            kernel_size=kernel_size,
            j_d=j_d,
            threshold=threshold_j,
            key=keys[1],
        )

    feedback = FrozenRescaledFullyConnected(
        in_features=num_labels,
        out_features=dim_hidden_flat,
        strength=strength_back,
        threshold=threshold_back,
        key=keys[2],
    )
    output = FullyConnected(
        in_features=dim_hidden_flat,
        out_features=num_labels,
        strength=1.0,
        threshold=threshold_out,
        key=keys[3],
    )

    layer_map = {
        1: {0: win, 1: j_mod, 2: feedback},
        2: {1: output, 2: OutputLayer()},
    }
    lmap = LayerMap.from_dict(layer_map)
    orch = SequentialOrchestrator(lmap)
    return state, orch


# ============================================================================
# Learning Rate Mapping
# ============================================================================


def make_lr_map_v2(
    model: SequentialOrchestrator,
    overrides: Mapping[tuple[int, int], str] | None = None,
    default_label: str = "default",
) -> PyTree:
    """Build parameter labels for optax.multi_transform."""
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    def like(tree, value: str):
        return jtu.tree_map(lambda _: value, tree, is_leaf=eqx.is_array)

    labels = jtu.tree_map(lambda _: default_label, params, is_leaf=eqx.is_array)

    if overrides:
        for (i, j), label in overrides.items():
            labels = eqx.tree_at(
                lambda m: m.lmap[i][j],
                labels,
                replace=like(params.lmap[i][j], label),
            )

    return labels


# ============================================================================
# Weight Decay (Handles Conv Kernels)
# ============================================================================


def decay(orchestrator: SequentialOrchestrator, cfg: dict):
    """Apply weight decay to Win, J, W_out (handles conv and FC)."""
    new_orch = orchestrator
    dim_hidden = cfg["model"]["kwargs"]["dim_hidden"]
    if isinstance(dim_hidden, (tuple, list)):
        h, w, c = dim_hidden
        dim_hidden_flat = h * w * c
    else:
        dim_hidden_flat = dim_hidden

    def _get_kernel_and_mask(module):
        if hasattr(module, "kernel"):
            return module.kernel, getattr(module, "update_mask", None)
        if hasattr(module, "W"):
            return module.W, getattr(module, "_mask", None)
        return None, None

    # Input (Win)
    win_mod = new_orch.lmap[1][0]
    kernel_win, mask_win = _get_kernel_and_mask(win_mod)
    if kernel_win is not None:
        rescale = (
            cfg["optimizer"]["weight_decay_win"]
            * cfg["optimizer"]["learning_rate_win"]
            / (dim_hidden_flat ** 0.5)
        )
        dW = kernel_win * rescale
        if mask_win is not None:
            dW = dW * mask_win
        new_orch = eqx.tree_at(lambda m: m.lmap[1][0].kernel, new_orch, kernel_win + dW)
    else:
        W_in = new_orch.lmap[1][0].W
        rescale = (
            cfg["optimizer"]["weight_decay_win"]
            * cfg["optimizer"]["learning_rate_win"]
            / (dim_hidden_flat ** 0.5)
        )
        new_orch = eqx.tree_at(lambda m: m.lmap[1][0].W, new_orch, W_in + W_in * rescale)

    # Recurrent (J)
    j_mod = new_orch.lmap[1][1]
    kernel_j, mask_j = _get_kernel_and_mask(j_mod)
    if kernel_j is not None:
        rescale = (
            cfg["optimizer"]["weight_decay_j"]
            * cfg["optimizer"]["learning_rate_j"]
            / (dim_hidden_flat ** 0.5)
        )
        dW = kernel_j * rescale
        if mask_j is not None:
            dW = dW * mask_j
        new_orch = eqx.tree_at(lambda m: m.lmap[1][1].kernel, new_orch, kernel_j + dW)
    else:
        J = new_orch.lmap[1][1].J if hasattr(new_orch.lmap[1][1], "J") else None
        if J is not None:
            rescale = (
                cfg["optimizer"]["weight_decay_j"]
                * cfg["optimizer"]["learning_rate_j"]
                / (dim_hidden_flat ** 0.5)
            )
            new_orch = eqx.tree_at(lambda m: m.lmap[1][1].J, new_orch, J + J * rescale)

    # Output (W_out)
    W_out = new_orch.lmap[2][1].W
    rescale = (
        cfg["optimizer"]["weight_decay_wout"]
        * cfg["optimizer"]["learning_rate_wout"]
        / (dim_hidden_flat ** 0.5)
    )
    new_orch = eqx.tree_at(lambda m: m.lmap[2][1].W, new_orch, W_out + W_out * rescale)

    return new_orch


# ============================================================================
# Training Loop (One Full Experiment)
# ============================================================================


def run_experiment(
    condition_name: str,
    win_type: str,
    j_type: str,
    lr_win: float,
    lr_j: float,
    kernel_size: int,
    cfg: dict,
):
    """Run one condition and return per-epoch train/test accuracies, params, and runtime."""
    print(f"\n{'='*70}")
    print(f"Condition: {condition_name}")
    print(f"Win: {win_type}, J: {j_type}, LR_win: {lr_win}, LR_j: {lr_j}, kernel: {kernel_size}")
    print(f"{'='*70}")

    cfg = copy.deepcopy(cfg)
    cfg["model"]["kwargs"].update(
        {
            "win_type": win_type,
            "j_type": j_type,
            "kernel_size": kernel_size,
            "c_in": 3,  # CIFAR10 has 3 color channels
        }
    )
    cfg["optimizer"]["learning_rate_win"] = lr_win
    cfg["optimizer"]["learning_rate_j"] = lr_j

    key = jax.random.PRNGKey(cfg["master_seed"])
    state, orch = build_model(**cfg["model"]["kwargs"])

    ds = Cifar10(**cfg["data"]["kwargs"])
    key, data_key = jax.random.split(key)
    ds.build(data_key)

    # Learning rate map
    lr_map = make_lr_map_v2(orch, overrides={(1, 0): "w_in", (1, 1): "j", (2, 1): "w_out"})

    optimizer = optax.multi_transform(
        {
            "default": optax.sgd(learning_rate=0.0),
            "w_in": optax.sgd(learning_rate=lr_win),
            "j": optax.sgd(learning_rate=lr_j),
            "w_out": optax.sgd(learning_rate=cfg["optimizer"]["learning_rate_wout"]),
        },
        lr_map,
    )
    opt_state = optimizer.init(eqx.filter(orch, eqx.is_inexact_array))

    trainer = DynamicalTrainer(
        orchestrator=orch,
        state=state,
        optimizer=optimizer,
        optimizer_state=opt_state,
        **cfg["trainer"]["kwargs"],
    )

    # Collect per-epoch metrics
    train_accs = []
    test_accs = []
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        if epoch != 0:
            for xb, yb in ds:
                key = trainer.train_step(xb, yb, key)
                trainer.orchestrator = decay(trainer.orchestrator, cfg)

        # Eval on test split
        accs_test = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_test.append(metrics["accuracy"])
        acc_test = float(jnp.mean(jnp.array(accs_test))) if accs_test else float("nan")

        # Eval on train split
        accs_train = []
        for xb, yb in ds:
            key, metrics = trainer.eval_step(xb, yb, key)
            accs_train.append(metrics["accuracy"])
        acc_train = float(jnp.mean(jnp.array(accs_train))) if accs_train else float("nan")

        train_accs.append(acc_train)
        test_accs.append(acc_test)

        print(f"Epoch {epoch:03d} | train_acc={acc_train:.4f} | eval_acc={acc_test:.4f}")

    elapsed = time.time() - start_time

    # Count params
    n_params = sum(
        p.size for p in jtu.tree_leaves(eqx.filter(trainer.orchestrator, eqx.is_inexact_array))
    )

    print(f"Final Test Accuracy: {test_accs[-1]:.4f}")
    print(f"Total Parameters: {n_params}")
    print(f"Elapsed (s): {elapsed:.2f}")

    return {
        "train_accs": train_accs,
        "test_accs": test_accs,
        "n_params": int(n_params),
        "time_sec": float(elapsed),
    }


# ============================================================================
# Main Experiment Driver
# ============================================================================


def main():
    """Run full 4-condition × kernel-size inquiry."""
    kernel_sizes = [3, 5, 7]
    conditions = [
        {
            "name": "Conv-Win + FC-J (frozen)",
            "win_type": "conv",
            "j_type": "fc",
            "lr_win": 0.0,
            "lr_j": 0.0,
        },
        {
            "name": "Conv-Win + Conv-J (frozen)",
            "win_type": "conv",
            "j_type": "conv",
            "lr_win": 0.0,
            "lr_j": 0.0,
        },
        {
            "name": "Conv-Win (learn) + FC-J (frozen)",
            "win_type": "conv",
            "j_type": "fc",
            "lr_win": 0.159,
            "lr_j": 0.058,
        },
        {
            "name": "Conv-Win + Conv-J (both learn)",
            "win_type": "conv",
            "j_type": "conv",
            "lr_win": 0.159,
            "lr_j": 0.058,
        },
    ]

    results = []

    for kernel_size in kernel_sizes:
        for cond in conditions:
            res = run_experiment(
                condition_name=cond["name"],
                win_type=cond["win_type"],
                j_type=cond["j_type"],
                lr_win=cond["lr_win"],
                lr_j=cond["lr_j"],
                kernel_size=kernel_size,
                cfg=BASE_PARAMS,
            )

            entry = {
                "kernel_size": int(kernel_size),
                "condition": cond["name"],
                "lr_win": float(cond["lr_win"]),
                "lr_j": float(cond["lr_j"]),
                "train_accs": res["train_accs"],
                "test_accs": res["test_accs"],
                "n_params": int(res["n_params"]),
                "time_sec": float(res["time_sec"]),
            }
            results.append(entry)

    # Summary print
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Kernel':<6} | {'Condition':<40} | {'Last Test Acc':<12} | {'Params':<12} | {'Time(s)':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['kernel_size']:<6} | {r['condition']:<40} | {r['test_accs'][-1]:<12.4f} | {r['n_params']:<12} | {r['time_sec']:<8.2f}")

    # Save full results as JSON (one file)
    out_dir = _Path(__file__).resolve().parents[0] / "results"
    os.makedirs(out_dir, exist_ok=True)
    json_path = out_dir / f"conv_inquiry_results.json"
    with open(json_path, "w") as f:
        json.dump({"created": datetime.utcnow().isoformat() + "Z", "results": results}, f, indent=2)

    print(f"Saved JSON results to: {json_path}")


if __name__ == "__main__":
    main()
