"""training_diagnostics.py

Trains the entropy architecture with full instrumentation. Three diagnostics:

  1. Weight dynamics  — |ΔW| per module per epoch (Win, J1, W_out).
                        Tells us: are weights still learning at epoch 10-20,
                        or have they frozen?

  2. CD overlap       — measured on 20 fixed training images after each epoch.
                        C = end of free phase (training dynamics, no weight update).
                        D = end of inference dynamics.
                        Tells us: does the train/inference gap open during full training?

  3. Filter analysis  — after training: visualize all 16 Win filters as 5x5 RGB patches;
                        plot SVD spectrum of J1 kernel (how diverse are the 16 channels?);
                        plot J1 channel-connectivity heatmap (mean |weight| per in→out channel pair).

Single seed (seed=0), 20 epochs, same hyperparams as replicate_channel_entropy.py.

Run from repo root:
  python experiments4/training_diagnostics.py
"""

from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"

sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.states.sequential import SequentialState

sys.path.insert(0, str(HERE))
from entropy_gap_experiment import (
    build_entropy_model, build_optimizer, to_hwc, _cd_overlap,
)

C_CH, KSIZE = 16, 5
H_IMG, W_IMG = 32, 32
EPOCHS     = 20
SEED       = 0
N_CD       = 20   # training images used for CD measurement each epoch


# ---------------------------------------------------------------------------
# Phase primitives (no weight update)
# ---------------------------------------------------------------------------

def _run_phases(orch, state, image, label, cfg, key):
    """Run warmup→clamped→free without touching weights. Returns (C, D, key)."""
    # ---- state C: training dynamics ----
    s = state.replace_val(0, image)

    # warmup: forward messages only, output free
    for _ in range(1):
        s, key = orch.step(s, rng=key, filter_messages="inference")

    # clamped: clamp output layer
    s = s.replace_val(2, label)
    for _ in range(cfg["clamped_n_iter"]):
        s, key = orch.step(s, rng=key, filter_messages="all")

    # free: release output (restore to zero so it can evolve)
    s = s.replace_val(2, jnp.zeros_like(s.states[2]))
    for _ in range(cfg["free_n_iter"]):
        s, key = orch.step(s, rng=key, filter_messages="inference")

    C = np.array(s.states[1])  # (1, H, W, 16)

    # ---- state D: inference dynamics ----
    s2 = state.replace_val(0, image)
    for _ in range(5):  # eval_n_iter
        s2, key = orch.step(s2, rng=key, filter_messages="inference")

    D = np.array(s2.states[1])
    return C, D, key


# ---------------------------------------------------------------------------
# Diagnostic 1: weight delta norm per module
# ---------------------------------------------------------------------------

def _param_norm(module) -> float:
    # Only use the kernel — scalar hyperparams like anti_threshold=-1e9
    # would otherwise dominate the norm completely.
    return float(jnp.sum(module.kernel ** 2) ** 0.5)


def _param_delta(mod_before, mod_after) -> float:
    return float(jnp.sum((mod_before.kernel - mod_after.kernel) ** 2) ** 0.5)


def snapshot_modules(orch):
    return {
        "win":  orch.lmap[1][0],
        "j1":   orch.lmap[1][1],
        "wout": orch.lmap[2][1],
    }


# ---------------------------------------------------------------------------
# Diagnostic 2: CD overlap on fixed sample
# ---------------------------------------------------------------------------

def measure_cd(orch, state_template, images, labels, cfg, key):
    cds = []
    for img, lbl in zip(images, labels):
        C, D, key = _run_phases(orch, state_template, img, lbl, cfg, key)
        cds.append(_cd_overlap(C, D))
    return float(np.mean(cds)), key


# ---------------------------------------------------------------------------
# Diagnostic 3: filter analysis
# ---------------------------------------------------------------------------

def plot_win_filters(kernel, out_path):
    """kernel: (5,5,3,16) → 16 RGB 5x5 filters in a 4x4 grid."""
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for idx, ax in enumerate(axes.flat):
        f = np.array(kernel[:, :, :, idx])          # (5,5,3)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f)
        ax.axis("off")
        ax.set_title(f"f{idx}", fontsize=7)
    fig.suptitle("Win filters (16 output channels, 5×5 RGB)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_j1_svd(kernel, out_path):
    """kernel: (5,5,16,16) → reshape to (5*5*16, 16), plot singular values."""
    kh, kw, cin_g, cout = kernel.shape
    mat = np.array(kernel).reshape(kh * kw * cin_g, cout)
    sv = np.linalg.svd(mat, compute_uv=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(sv)), sv, color="steelblue")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value")
    ax.set_title("J1 kernel SVD spectrum\n(higher = more diverse output filters)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return sv


def plot_j1_channel_heatmap(kernel, out_path):
    """kernel: (5,5,16,16) → mean |weight| per (in_channel, out_channel) pair."""
    # sum over spatial dims → (cin_g=16, cout=16), mean abs weight
    heatmap = np.abs(np.array(kernel)).mean(axis=(0, 1))  # (16, 16)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(heatmap, cmap="viridis", aspect="auto")
    ax.set_xlabel("Output channel")
    ax.set_ylabel("Input channel")
    ax.set_title("J1 mean |weight| per channel pair\n(diagonal = self-connections fixed at j_d)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_training_curves(log, out_path):
    epochs = [e["epoch"] for e in log]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    # panel 1: |ΔW| per module
    ax = axes[0]
    for module in ["win", "j1", "wout"]:
        ax.plot(epochs, [e["dW"][module] for e in log], label=module, marker="o", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("|ΔW|")
    ax.set_title("Weight update magnitude per epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # panel 2: weight norms
    ax = axes[1]
    for module in ["win", "j1", "wout"]:
        ax.plot(epochs, [e["W_norm"][module] for e in log], label=module, marker="o", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("|W|")
    ax.set_title("Weight magnitude per epoch")
    ax.legend(); ax.grid(alpha=0.3)

    # panel 3: CD overlap + head accuracy
    ax = axes[2]
    ax.plot(epochs, [e["cd"] for e in log], "b-o", ms=3, label="CD overlap")
    ax2 = ax.twinx()
    ax2.plot(epochs, [e["head_acc"] for e in log], "r--s", ms=3, label="head acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CD overlap", color="b")
    ax2.set_ylabel("Head test acc", color="r")
    ax.set_title("CD overlap + head accuracy")
    ax.set_ylim(0, 1); ax2.set_ylim(0, 1)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Training diagnostics — entropy rule, seed={SEED}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    # fixed CD sample: first N_CD batches of size 1 from training set
    ds_single = Cifar10(batch_size=2, x_transform="identity", label_mode="pm1",
                        linear_projection=None, rescale=True)
    ds_single.build(jax.random.PRNGKey(0))
    cd_images, cd_labels = [], []
    for i, (xb, yb) in enumerate(ds_single):
        if i >= N_CD: break
        cd_images.append(to_hwc(xb[0:1]))
        cd_labels.append(yb[0:1])

    state_template, orch = build_entropy_model(cfg, SEED)
    opt, opt_state = build_optimizer(orch, cfg)

    from darnax.trainers.dynamical import DynamicalTrainer
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state_template,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    key = jax.random.PRNGKey(SEED)
    log = []

    results_dir = HERE / "results"
    figs_dir    = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    print(f"Training {EPOCHS} epochs — entropy rule, seed={SEED}")
    print(f"{'epoch':>6}  {'dW_win':>9}  {'dW_j1':>9}  {'dW_wout':>9}  "
          f"{'|W|_j1':>9}  {'CD':>7}  {'head':>7}")

    for epoch in range(1, EPOCHS + 1):

        # snapshot weights before epoch
        snap_before = snapshot_modules(trainer.orchestrator)

        # --- train ---
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
            # Win normalization (same as replicate script)
            win_k = trainer.orchestrator.lmap[1][0].kernel
            kh, kw, ci, co = win_k.shape
            flat   = win_k.reshape(-1, co)
            normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            trainer.orchestrator = eqx.tree_at(
                lambda o: o.lmap[1][0].kernel,
                trainer.orchestrator,
                normed.reshape(kh, kw, ci, co),
            )

        # kernel decay (same as replicate script)
        decay = cfg["kernel_decay_rate"]
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay)
            )

        # --- diagnostic 1: weight deltas ---
        snap_after = snapshot_modules(trainer.orchestrator)
        dW = {m: _param_delta(snap_before[m], snap_after[m]) for m in snap_before}
        W_norm = {m: _param_norm(snap_after[m]) for m in snap_after}

        # --- diagnostic 2: CD overlap ---
        # Use a clean zero state (not the last training state) so the
        # dynamics start from the same neutral point every time.
        cd_mean, key = measure_cd(
            trainer.orchestrator, state_template,
            cd_images, cd_labels, cfg, key
        )

        # --- head accuracy on test set ---
        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        head_acc = float(np.mean(batch_accs))

        entry = {
            "epoch": epoch,
            "dW": dW, "W_norm": W_norm,
            "cd": cd_mean, "head_acc": head_acc,
        }
        log.append(entry)

        print(f"{epoch:>6}  {dW['win']:>9.4f}  {dW['j1']:>9.4f}  {dW['wout']:>9.4f}  "
              f"{W_norm['j1']:>9.4f}  {cd_mean:>7.4f}  {head_acc:>7.4f}")

    # --- diagnostic 3: filter analysis ---
    print("\nFilter analysis:")
    win_kernel = trainer.orchestrator.lmap[1][0].kernel   # (5,5,3,16)
    j1_kernel  = trainer.orchestrator.lmap[1][1].kernel   # (5,5,16,16)

    plot_win_filters(win_kernel,           figs_dir / "diag_win_filters.png")
    sv = plot_j1_svd(j1_kernel,            figs_dir / "diag_j1_svd.png")
    plot_j1_channel_heatmap(j1_kernel,     figs_dir / "diag_j1_heatmap.png")
    plot_training_curves(log,              figs_dir / "diag_training_curves.png")

    print(f"\nJ1 singular values: {np.round(sv, 3)}")
    print(f"J1 effective rank (sv[0]/sv[-1] ratio): {sv[0]/sv[-1]:.1f}x")

    # save log
    out_path = results_dir / "training_diagnostics.json"
    out_path.write_text(json.dumps({"config": cfg, "seed": SEED, "log": log}, indent=2))
    print(f"\nLog saved to {out_path}")


if __name__ == "__main__":
    main()
