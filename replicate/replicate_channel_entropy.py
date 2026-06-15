"""replicate_channel_entropy.py

Trains the channel_entropy model from scratch on CIFAR-10 using darnax
(MateiCosa/darnax, conv branch).

Architecture: ChannelWBack + Conv2DRecurrentDiscrete(entropy_beta, lambda_entropy=1)
  Conv2DRecurrentDiscrete has entropy modulation built in via entropy_beta and
  lambda_entropy constructor parameters.
  ChannelWBack and PooledFlattenFC are imported from darnax.modules.conv.spatial_fc.

Hyperparameters are loaded from best_channel_entropy_cfg.json in the same directory.

Usage:
  conda activate darnn
  python replicate_channel_entropy.py
"""

from __future__ import annotations

import jax
jax.config.update("jax_default_matmul_precision", "highest")

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE       = Path(__file__).resolve().parent        # replicate/
DARNAX_SRC = HERE.parent / "src"                   # src/
CFG_PATH   = HERE / "matei_W_out_cfg.json"

sys.path.insert(0, str(DARNAX_SRC))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C, KSIZE     = 16, 5
H_IMG, W_IMG = 32, 32
POOL         = 8
EPOCHS       = 20
SEEDS        = [0, 42, 123, 7, 999]
PROBE_EPOCHS = 20
PROBE_WD     = 1.433e-4   # from best trial (trial 14, probe_acc=0.4501)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: dict, key: jax.Array):
    """Build SequentialOrchestrator for channel_entropy architecture.

    Uses Conv2DRecurrentDiscrete directly with entropy_beta and lambda_entropy
    (built into darnax_update) instead of the EntropyJ1 subclass.
    """
    keys = jax.random.split(key, 5)

    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(
                in_channels=3, out_channels=C, kernel_size=KSIZE,
                threshold=cfg["threshold_win"], strength=1.0,
                key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.0,
            ),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H_IMG, W_IMG, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(
                pool=POOL, H=H_IMG, W=W_IMG, C_in=C, n_classes=10,
                strength=1.0, threshold=5.0,
                key=keys[3], lr=1.0, weight_decay=0.0,
            ),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H_IMG, W_IMG, 3), (H_IMG, W_IMG, C), 10])
    return state, SequentialOrchestrator(layers=layer_map)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_optimizer(orchestrator, cfg: dict):
    mom    = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(
            lambda m, r=i, c=j: m.lmap[r][c], labels,
            replace=like(params.lmap[i][j], lbl),
        )

    # W_back at (1,2) stays "default" → lr=0 (frozen)
    opt = optax.multi_transform({
        "default": optax.sgd(0.0),
        "win":     sgd(-cfg["lr_win"]),
        "j1":      sgd(-cfg["lr_j"]),
        "wout":    sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------

def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_probe(trainer, ds: Cifar10, key: jax.Array) -> dict:
    """Train a linear probe on pooled J1 representations.

    Mirrors the original T21 evaluation: 8×8 avg-pool → 256-dim → Linear(256, 10).
    """
    def pool_j1(h):
        N = h.shape[0]
        return h.reshape(N, H_IMG // POOL, POOL, W_IMG // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)

    reps_tr, lbl_tr, reps_te, lbl_te = [], [], [], []

    for xb, yb in ds:
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_tr.append(pool_j1(np.array(trainer.state[1])))
        lbl_tr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_te.append(pool_j1(np.array(trainer.state[1])))
        lbl_te.append(np.argmax(np.array(yb), axis=-1))

    X_tr = torch.from_numpy(np.concatenate(reps_tr)).float()
    y_tr = torch.from_numpy(np.concatenate(lbl_tr)).long()
    X_te = torch.from_numpy(np.concatenate(reps_te)).float()
    y_te = torch.from_numpy(np.concatenate(lbl_te)).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe  = nn.Linear(256, 10, bias=False).to(device)
    opt_p  = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    crit   = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb_t, yb_t in loader:
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            opt_p.zero_grad(); crit(probe(xb_t), yb_t).backward(); opt_p.step()
        probe.eval()
        with torch.no_grad():
            tr = (probe(X_tr.to(device)).argmax(1) == y_tr.to(device)).float().mean().item()
            te = (probe(X_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
        train_accs.append(tr); test_accs.append(te)

    return {"train": train_accs, "test": test_accs,
            "best_train": max(train_accs), "best_test": max(test_accs)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_seed(cfg: dict, ds: Cifar10, seed: int) -> tuple[list[float], list[float]]:
    """Train for EPOCHS epochs from a single seed.

    Returns (head_accs, probe_accs) — both lists of length EPOCHS.
    probe_accs[i] is the best probe test accuracy after epoch i+1.
    """
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(orch, cfg)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    decay = cfg["kernel_decay_rate"]
    head_accs: list[float] = []
    probe_accs: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
            # Per-batch W_in column-normalisation was originally included here but
            # suppresses W_in growth and causes probe accuracy to drop from ~0.46 to
            # ~0.43.  Without it, W_in and J settle into a roughly equal contribution
            # (~50/50), which is the regime that matches the reference (0.4501).
            # Kept commented out for reference; the correct loop applies decay only.
            # win_k = trainer.orchestrator.lmap[1][0].kernel  # (kh, kw, 3, C)
            # kh, kw, ci, co = win_k.shape
            # flat   = win_k.reshape(-1, co)
            # normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            # trainer.orchestrator = eqx.tree_at(
            #     lambda o: o.lmap[1][0].kernel,
            #     trainer.orchestrator,
            #     normed.reshape(kh, kw, ci, co),
            # )
        # kernel decay at end of epoch
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay)
            )

        # --- evaluate (model head) ---
        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        epoch_acc = float(np.mean(batch_accs))
        head_accs.append(epoch_acc)

        # --- linear probe on current representations ---
        probe_res = run_probe(trainer, ds, key)
        probe_accs.append(probe_res["best_test"])

        print(f"  seed={seed}  epoch={epoch:2d}/{EPOCHS}"
              f"  head={epoch_acc:.4f}  probe={probe_res['best_test']:.4f}")

    return head_accs, probe_accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)

    # strip metadata keys that are not model hyperparameters
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    all_accs:       list[list[float]] = []
    all_probe_accs: list[list[float]] = []
    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print(f"{'='*50}")
        head_curve, probe_curve = train_one_seed(cfg, ds, seed)
        all_accs.append(head_curve)
        all_probe_accs.append(probe_curve)

    all_accs_np    = np.array(all_accs)        # (n_seeds, EPOCHS)
    all_probes_np  = np.array(all_probe_accs)  # (n_seeds, EPOCHS)
    mean_acc       = all_accs_np.mean(axis=0)
    std_acc        = all_accs_np.std(axis=0)
    mean_probe     = all_probes_np.mean(axis=0)
    std_probe      = all_probes_np.std(axis=0)
    epochs         = np.arange(1, EPOCHS + 1)

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    colours = plt.cm.tab10(np.linspace(0, 0.5, len(SEEDS)))

    # left: model head accuracy over epochs
    for (seed, curve), col in zip(zip(SEEDS, all_accs), colours):
        ax1.plot(epochs, curve, color=col, alpha=0.45, linewidth=1.2,
                 label=f"seed {seed}")
    ax1.plot(epochs, mean_acc, "k-", linewidth=2.5, label="mean")
    ax1.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                     alpha=0.18, color="black", label="±1 std")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Model head accuracy")
    ax1.legend(fontsize=8, ncol=2)

    # right: linear probe test accuracy over epochs
    for (seed, curve), col in zip(zip(SEEDS, all_probe_accs), colours):
        ax2.plot(epochs, curve, color=col, alpha=0.45, linewidth=1.2,
                 label=f"seed {seed}")
    ax2.plot(epochs, mean_probe, "k-", linewidth=2.5, label="mean")
    ax2.fill_between(epochs, mean_probe - std_probe, mean_probe + std_probe,
                     alpha=0.18, color="black", label="±1 std")
    ax2.axhline(0.4501, color="grey", linewidth=1.5, linestyle=":",
                label="T21 best (0.4501)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Probe test accuracy")
    ax2.set_title(f"Linear probe ({PROBE_EPOCHS} ep) per epoch")
    ax2.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"channel_entropy — {len(SEEDS)} seeds, {EPOCHS} epochs  "
        f"(ChannelWBack + Conv2DRecurrentDiscrete with entropy, darnax_update)",
        fontsize=10,
    )
    fig.tight_layout()
    out_png = HERE / "accuracy_curves.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nPlot saved to {out_png}")

    print(f"\nModel head test accuracy (epoch {EPOCHS}):")
    for seed, curve in zip(SEEDS, all_accs):
        print(f"  seed {seed:4d}: {curve[-1]:.4f}")
    print(f"  mean:       {mean_acc[-1]:.4f} ± {std_acc[-1]:.4f}")

    print(f"\nLinear probe best test accuracy (epoch {EPOCHS}):")
    for seed, curve in zip(SEEDS, all_probe_accs):
        print(f"  seed {seed:4d}: {curve[-1]:.4f}")
    print(f"  mean:       {mean_probe[-1]:.4f} ± {std_probe[-1]:.4f}")
    print(f"  T21 reference: 0.4501")


if __name__ == "__main__":
    main()
