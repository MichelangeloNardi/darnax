"""experiments4/diag_C_matrix.py

Single-matrix autocorrelation diagnostic, on Config 3 with short dynamics
(warmup=1, clamped=5, free=5).

At selected training checkpoints, we capture the J1 state at EVERY inner step
of the dynamics for a fixed measurement batch:
    W1 (after the 1 warmup step)
    C1, C2, C3, C4, C5  (after each clamped step)
    F1, F2, F3, F4, F5  (after each free step)
giving 11 J1 states total per checkpoint. We then compute the 11×11
dot-product overlap matrix between all pairs and plot it.

Reads off-diagonally as the autocorrelation of the dynamics: cells close to
the diagonal show how much one step shifts the state; far-off-diagonal cells
show how much the trajectory drifts across the whole sequence.

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/diag_C_matrix.py
"""

from __future__ import annotations

import json
import sys
import time
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
sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

C_CH, KSIZE = 16, 5
H, W = 32, 32
POOL = 8
SEED = 0
BATCH_SIZE = 32

# Short dynamics (Config 3 hyperparams + short iteration counts)
WARMUP_N = 1
CLAMPED_N = 5
FREE_N = 5

CHECKPOINTS = [0, 50, 200, 1000, 3000]

BASE_CFG = {
    "lr_win":            0.0014704900068512122,
    "lr_j":              0.0001640857799594289,
    "lr_wout":           0.0046199902170752346,
    "threshold_win":     0.3020596991424575,
    "threshold_j":       0.831821321919564,
    "j_d":               0.8928072670420131,
    "entropy_beta":      0.0160087816929895,
    "momentum":          0.6668784969214014,
    "kernel_decay_rate": 0.0017937722033317504,
    "strength_back":     0.1680366998579321,
}


def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C_CH, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0,
                      key=keys[0], padding_mode="constant",
                      lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(channels=C_CH, kernel_size=KSIZE, groups=1,
                                       j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                                       key=keys[1], padding_mode="constant",
                                       lr=1.0, weight_decay=0.0,
                                       entropy_beta=cfg["entropy_beta"],
                                       lambda_entropy=1.0),
            2: ChannelWBack(10, H, W, C_CH, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C_CH, n_classes=10,
                               strength=1.0, threshold=5.0,
                               key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C_CH), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def make_train_optimizer(orch, cfg):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orch, eqx.is_inexact_array)

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
    opt = optax.multi_transform({
        "default": optax.sgd(0.0),
        "win":     sgd(-cfg["lr_win"]),
        "j1":      sgd(-cfg["lr_j"]),
        "wout":    sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orch, eqx.is_inexact_array))


def normalize_win(orch):
    win_k = orch.lmap[1][0].kernel
    kh, kw, ci, co = win_k.shape
    flat = win_k.reshape(-1, co)
    normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
    return eqx.tree_at(
        lambda o: o.lmap[1][0].kernel, orch, normed.reshape(kh, kw, ci, co),
    )


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def capture_trajectory(orch, state_template, x, y, rng):
    """Run dynamics step by step and return the J1 state after each inner step.

    Returns a list of 11 numpy arrays, one per inner step:
      [W1,  C1, C2, C3, C4, C5,  F1, F2, F3, F4, F5]
    """
    state = state_template.init(x, y)
    out = []

    # 1 warmup step (forward only)
    state, rng = orch.step(state, rng=rng, filter_messages="forward")
    out.append(np.asarray(state[1]))

    # 5 clamped steps (all)
    for _ in range(CLAMPED_N):
        state, rng = orch.step(state, rng=rng, filter_messages="all")
        out.append(np.asarray(state[1]))

    # 5 free steps (forward only)
    for _ in range(FREE_N):
        state, rng = orch.step(state, rng=rng, filter_messages="forward")
        out.append(np.asarray(state[1]))

    return out


def autocorr_matrix(states):
    """Pairwise dot-product overlap between binary states."""
    n = len(states)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            M[i, j] = float(np.mean(np.sign(states[i]) * np.sign(states[j])))
    return M


def plot_matrices(per_checkpoint, fig_path, step_labels):
    cps = sorted(per_checkpoint.keys())
    fig, axes = plt.subplots(1, len(cps),
                             figsize=(4.5 * len(cps), 4.5), sharey=True)
    if len(cps) == 1:
        axes = [axes]

    # Common vmin
    all_vals = np.concatenate([per_checkpoint[cp].ravel() for cp in cps])
    off_diag = all_vals[all_vals < 0.9999]
    vmin = float(off_diag.min()) if off_diag.size else 0.0

    for ax, cp in zip(axes, cps):
        M = per_checkpoint[cp]
        im = ax.imshow(M, vmin=vmin - 0.01, vmax=1.0, cmap="viridis")
        for i in range(len(step_labels)):
            for j in range(len(step_labels)):
                v = M[i, j]
                colour = "white" if v < (vmin + 1.0) / 2 + 0.02 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=colour, fontsize=6.5)
        # Phase dividers: after W1 (index 0) and after C5 (index 5)
        for pos in (0.5, 5.5):
            ax.axhline(pos, color="white", lw=1.2)
            ax.axvline(pos, color="white", lw=1.2)
        ax.set_xticks(range(len(step_labels)))
        ax.set_yticks(range(len(step_labels)))
        ax.set_xticklabels(step_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(step_labels, fontsize=8)
        ax.set_title(f"batch {cp}", fontsize=11)

    axes[0].set_ylabel("step within dynamics")
    fig.suptitle(
        "Within-dynamics J1 autocorrelation matrix  "
        "(Config 3 hyperparams, short dynamics 1+5+5 = 11 steps)",
        fontsize=11,
    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="dot product")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = BASE_CFG
    step_labels = (["W1"]
                   + [f"C{i}" for i in range(1, CLAMPED_N + 1)]
                   + [f"F{i}" for i in range(1, FREE_N + 1)])

    print("Building dataset...", flush=True)
    ds = Cifar10(batch_size=BATCH_SIZE, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True, shuffle=True)
    ds.build(jax.random.PRNGKey(0))

    print("Building model...", flush=True)
    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_train_optimizer(orch, cfg)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=WARMUP_N,
        train_clamped_n_iter=CLAMPED_N,
        train_free_n_iter=FREE_N,
        eval_n_iter=FREE_N,
    )

    # Fixed measurement batch
    x_meas = y_meas = None
    for xb, yb in ds:
        x_meas, y_meas = to_hwc(xb), yb
        break

    per_checkpoint: dict = {}

    def measure(batch_count):
        meas_rng = jax.random.PRNGKey(3_000 + batch_count)
        states = capture_trajectory(
            trainer.orchestrator, trainer.state, x_meas, y_meas, meas_rng,
        )
        M = autocorr_matrix(states)
        per_checkpoint[batch_count] = M
        # Quick textual summary
        adj = np.diagonal(M, offset=1).mean()   # mean step-to-step (adjacent)
        far = M[0, -1]                          # W1 vs F5
        print(f"  batch {batch_count:5d}  ⟨adj⟩={adj:.4f}  W1↔F5={far:.4f}",
              flush=True)

    pending = list(CHECKPOINTS)
    if pending[0] == 0:
        print("\nMeasuring at batch 0 (random init)...", flush=True)
        measure(0)
        pending.pop(0)

    print("\nTraining (short dynamics 1-5-5)...", flush=True)
    t_train = time.time()
    batch_count = 0
    train_key = jax.random.PRNGKey(SEED + 1)
    decay = cfg["kernel_decay_rate"]
    done = False
    while not done:
        for xb, yb in ds:
            train_key = trainer.train_step(to_hwc(xb), yb, train_key)
            trainer.orchestrator = normalize_win(trainer.orchestrator)
            batch_count += 1
            if pending and batch_count >= pending[0]:
                # one-time kernel decay before measuring (matches training-loop behavior)
                measure(batch_count)
                pending.pop(0)
                if not pending:
                    done = True
                    break
    print(f"\nDone in {time.time() - t_train:.1f}s", flush=True)

    # Save raw and figure
    results_dir = HERE / "results"
    figs_dir = HERE / "meeting"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    raw = {
        "config": cfg,
        "warmup_n": WARMUP_N,
        "clamped_n": CLAMPED_N,
        "free_n": FREE_N,
        "step_labels": step_labels,
        "matrices": {str(cp): M.tolist() for cp, M in per_checkpoint.items()},
    }
    out_json = results_dir / "diag_C_matrix.json"
    out_json.write_text(json.dumps(raw, indent=2, default=float))
    print(f"Saved JSON to {out_json}")

    out_png = figs_dir / "diag_autocorr_matrix.png"
    plot_matrices(per_checkpoint, out_png, step_labels)
    print(f"Saved figure to {out_png}")

    # Remove the stale per-phase autocorrelation figure
    stale = figs_dir / "diag_autocorrelation.png"
    if stale.exists():
        stale.unlink()
        print(f"Removed stale {stale}")


if __name__ == "__main__":
    main()
