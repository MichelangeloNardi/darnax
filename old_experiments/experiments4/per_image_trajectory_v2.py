"""experiments4/per_image_trajectory_v2.py

Per-image trajectory tracking (v2). Extends the agent's original by adding:

  (1) Per-probe J1 stability against a "post-first-seen baseline".
      The agent's run showed B-C and C-D overlaps stay at 1.0 for every probe
      across training — but that doesn't tell us whether the J1 state for one
      probe is itself stable, or whether J1 is drifting in lockstep across all
      phases (so within-batch overlaps stay 1.0 by coincidence).
      For each probe we capture its J1_C and J1_D at the first measurement
      taken AT OR AFTER the probe's first-seen batch (i.e., the network has
      had at least one opportunity to learn on it). We then compare every
      subsequent J1_C/J1_D for that probe to that baseline. If the overlap
      decays, the per-probe representation is genuinely drifting.

  (2) W_out and J1-kernel drift from initialization.
      Single scalars per checkpoint: ||W_out(t) - W_out(0)||_F and
      ||J(t) - J(0)||_F. Tells us how much each matrix has moved by training
      step t — directly answers "is the forgetting driven by W_out drift?".

  (3) W_out direction drift in the SAME probe.
      Tracks the cosine similarity of W_out's per-class rows over time —
      complementary to the L2 norm above.

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/per_image_trajectory_v2.py
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
from darnax.trainers.utils import scan_n

C_CH, KSIZE = 16, 5
H, W = 32, 32
POOL = 8
SEED = 0
BATCH_SIZE = 32

N_EPOCHS = 2
MEASURE_EVERY = 50
PROBES_PER_CLASS = 5
N_CLASSES = 10

CONFIG = {
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
    "warmup_n_iter":     6,
    "clamped_n_iter":    11,
    "free_n_iter":       14,
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


def compute_permutations(ds, n_epochs):
    assert ds._train_epoch_key is not None
    n_train = ds.x_train.shape[0]
    epoch_key = ds._train_epoch_key
    perms = []
    for _ in range(n_epochs):
        key_epoch, epoch_key = jax.random.split(epoch_key)
        perms.append(np.asarray(jax.random.permutation(key_epoch, n_train)))
    return perms


def pick_probes_earliest_per_class(y_train, perm, probes_per_class):
    y_class = np.asarray(jnp.argmax(y_train, axis=-1))
    picked: dict[int, list[int]] = {c: [] for c in range(N_CLASSES)}
    n_remaining = N_CLASSES * probes_per_class
    for oi in perm:
        c = int(y_class[oi])
        if len(picked[c]) < probes_per_class:
            picked[c].append(int(oi))
            n_remaining -= 1
            if n_remaining == 0:
                break
    indices = []
    for c in range(N_CLASSES):
        indices.extend(picked[c])
    return indices


def first_seen_batches(perms, probe_indices, batch_size):
    perm0 = perms[0]
    return [int(np.where(perm0 == oi)[0][0] // batch_size) for oi in probe_indices]


def make_measure_fn(cfg):
    warmup_n = cfg["warmup_n_iter"]
    clamped_n = cfg["clamped_n_iter"]
    free_n = cfg["free_n_iter"]

    @eqx.filter_jit
    def measure(orch, state_template, x, y, rng):
        state0 = state_template.init(x, y)
        (state_a, rng_a), _ = scan_n(orch.step, (state0, rng),
                                     n_iter=warmup_n, filter_messages="forward")
        (state_b, rng_b), _ = scan_n(orch.step, (state_a, rng_a),
                                     n_iter=clamped_n, filter_messages="all")
        (state_c, rng_c), _ = scan_n(orch.step, (state_b, rng_b),
                                     n_iter=free_n, filter_messages="forward")
        (state_d, rng_d), _ = scan_n(orch.step, (state_a, rng_a),
                                     n_iter=free_n, filter_messages="forward")
        state_c_pred, _ = orch.predict(state_c, rng_c)
        state_d_pred, _ = orch.predict(state_d, rng_d)
        return {
            "j1_A": state_a[1], "j1_B": state_b[1],
            "j1_C": state_c[1], "j1_D": state_d[1],
            "margin_C": soft_margin(state_c_pred.readout, y),
            "margin_D": soft_margin(state_d_pred.readout, y),
        }

    return measure


def soft_margin(logits, y):
    correct = jnp.argmax(y, axis=-1)
    correct_logit = jnp.take_along_axis(logits, correct[:, None], axis=-1)[:, 0]
    mask = jax.nn.one_hot(correct, num_classes=logits.shape[-1], dtype=bool)
    wrong_max = jnp.max(jnp.where(mask, -jnp.inf, logits), axis=-1)
    return correct_logit - wrong_max


def pairwise_match_per_image(s1, s2):
    axes = tuple(range(1, s1.ndim))
    return jnp.mean((jnp.sign(s1) == jnp.sign(s2)).astype(jnp.float32), axis=axes)


def weight_drift_metrics(orch, initial_W_out, initial_J1_kernel, initial_W_in_kernel):
    """Compute aggregate drift metrics from initialization."""
    W_out = np.array(orch.lmap[2][1].W)
    J1_k = np.array(orch.lmap[1][1].kernel)
    W_in_k = np.array(orch.lmap[1][0].kernel)

    # L2 distances from init
    wout_drift = float(np.linalg.norm(W_out - initial_W_out))
    j1_drift = float(np.linalg.norm(J1_k - initial_J1_kernel))
    win_drift = float(np.linalg.norm(W_in_k - initial_W_in_kernel))

    # Cosine similarity of W_out's per-class rows to init
    n_classes = W_out.shape[1]
    wout_cos = []
    for c in range(n_classes):
        v_init = initial_W_out[:, c]
        v_now = W_out[:, c]
        denom = float(np.linalg.norm(v_init) * np.linalg.norm(v_now) + 1e-8)
        wout_cos.append(float(np.dot(v_init, v_now) / denom))

    return {
        "wout_l2_drift": wout_drift,
        "j1_l2_drift": j1_drift,
        "win_l2_drift": win_drift,
        "wout_class_cosine": wout_cos,
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

CMAP_CYCLE = plt.cm.tab10.colors


def _plot_overlap_panel(ax, aligned_x, data, title, probe_classes, ylim=(0.45, 1.02)):
    for pi in range(data.shape[0]):
        mask = aligned_x[pi] >= 0
        if mask.any():
            ax.plot(aligned_x[pi][mask], data[pi][mask], "-",
                    color=CMAP_CYCLE[probe_classes[pi]], alpha=0.35, lw=0.8)
    _plot_mean_curve(ax, aligned_x, data, label="mean over all probes")
    ax.set_title(title)
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("match rate")
    ax.set_ylim(*ylim)
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def _plot_margin_panel(ax, aligned_x, margin, probe_classes, title):
    for pi in range(margin.shape[0]):
        mask = aligned_x[pi] >= 0
        if mask.any():
            ax.plot(aligned_x[pi][mask], margin[pi][mask], "-",
                    color=CMAP_CYCLE[probe_classes[pi]], alpha=0.35, lw=0.8)
    _plot_mean_curve(ax, aligned_x, margin, label="mean over all probes")
    ax.set_title(title)
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("margin (correct − max wrong)")
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def _plot_mean_curve(ax, aligned_x, data, label):
    valid = (aligned_x >= 0) & np.isfinite(data)
    if not valid.any():
        return
    xs_pos = aligned_x[valid]
    xmin = int(xs_pos.min()); xmax = int(xs_pos.max())
    step = max(1, (xmax - xmin) // 60 or 1)
    grid = np.arange(xmin, xmax + 1, step)
    means = np.full_like(grid, np.nan, dtype=float)
    for gi, g in enumerate(grid):
        vals = []
        for pi in range(data.shape[0]):
            mask = (aligned_x[pi] >= 0) & np.isfinite(data[pi])
            if mask.any():
                xs_p = aligned_x[pi][mask]
                ys_p = data[pi][mask]
                if xs_p.min() <= g <= xs_p.max():
                    vals.append(float(np.interp(g, xs_p, ys_p)))
        if vals:
            means[gi] = float(np.mean(vals))
    valid_grid = ~np.isnan(means)
    if valid_grid.any():
        ax.plot(grid[valid_grid], means[valid_grid], "-",
                color="black", lw=2.0, label=label)


def plot_main_figure(results, probe_classes, fig_path):
    """Main figure: original 9 panels + 3 new panels for J1 stability."""
    aligned_x = np.array(results["aligned_batches"])

    fig, axes = plt.subplots(4, 3, figsize=(16, 17), sharex=False)

    # Row 0: A-B, B-C, C-D pairwise overlaps (same-checkpoint, within-probe)
    for ax, key in zip(axes[0], ["A_B", "B_C", "C_D"]):
        _plot_overlap_panel(
            ax, aligned_x, np.array(results[f"overlap_{key}"]),
            f"{key.replace('_', '-')} (same-checkpoint within probe)",
            probe_classes,
        )

    # Row 1: A-C, A-D, B-D
    for ax, key in zip(axes[1], ["A_C", "A_D", "B_D"]):
        _plot_overlap_panel(
            ax, aligned_x, np.array(results[f"overlap_{key}"]),
            f"{key.replace('_', '-')} (same-checkpoint within probe)",
            probe_classes,
        )

    # Row 2: margins
    margin_c = np.array(results["margin_C"])
    margin_d = np.array(results["margin_D"])
    _plot_margin_panel(axes[2, 0], aligned_x, margin_c, probe_classes,
                       "Soft margin from state C (training protocol)")
    _plot_margin_panel(axes[2, 1], aligned_x, margin_d, probe_classes,
                       "Soft margin from state D (inference protocol)")

    # C-vs-D margin scatter
    ax = axes[2, 2]
    for pi in range(margin_c.shape[0]):
        mask = aligned_x[pi] >= 0
        if mask.any():
            ax.scatter(margin_c[pi][mask], margin_d[pi][mask],
                       color=CMAP_CYCLE[probe_classes[pi]],
                       s=12, alpha=0.5, edgecolors="none")
    for c in range(N_CLASSES):
        ax.scatter([], [], color=CMAP_CYCLE[c], s=20, label=f"class {c}")
    finite_c = margin_c[(aligned_x >= 0) & np.isfinite(margin_c)]
    finite_d = margin_d[(aligned_x >= 0) & np.isfinite(margin_d)]
    if finite_c.size and finite_d.size:
        lim = float(max(np.abs(finite_c).max(), np.abs(finite_d).max())) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6, alpha=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("margin from C"); ax.set_ylabel("margin from D")
    ax.set_title("C-vs-D margin (each point = one probe × checkpoint)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # Row 3: NEW — J1 stability per probe (cross-time overlap vs baseline)
    _plot_overlap_panel(
        axes[3, 0], aligned_x, np.array(results["stability_C"]),
        "J1_C(probe, t) vs J1_C(probe, baseline)\n(per-probe drift)",
        probe_classes,
    )
    _plot_overlap_panel(
        axes[3, 1], aligned_x, np.array(results["stability_D"]),
        "J1_D(probe, t) vs J1_D(probe, baseline)\n(per-probe drift)",
        probe_classes,
    )

    # Weight drift (single curve, batch on x)
    ax = axes[3, 2]
    batches = np.array(results["measure_batches"])
    ax.plot(batches, results["wout_l2_drift"], "-o", color="tab:red",
            label="||W_out(t) - W_out(0)||")
    ax.plot(batches, results["j1_l2_drift"], "-s", color="tab:blue",
            label="||J(t) - J(0)||")
    ax.plot(batches, results["win_l2_drift"], "-^", color="tab:green",
            label="||W_in(t) - W_in(0)||")
    ax.set_xlabel("training batch"); ax.set_ylabel("L2 drift from init")
    ax.set_title("Weight matrix drift from init")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Per-image trajectories (v2): J1 stability + W_out drift  "
        f"({PROBES_PER_CLASS} probes/class × {N_CLASSES} = "
        f"{PROBES_PER_CLASS * N_CLASSES} probes, {N_EPOCHS} epochs)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_wout_cosine(results, fig_path):
    batches = np.array(results["measure_batches"])
    cos = np.array(results["wout_class_cosine"])  # (N_meas, 10)
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in range(N_CLASSES):
        ax.plot(batches, cos[:, c], "-o", color=CMAP_CYCLE[c],
                markersize=4, label=f"class {c}")
    ax.set_xlabel("training batch")
    ax.set_ylabel("cosine(W_out_c(t), W_out_c(0))")
    ax.set_title("Per-class W_out direction drift from initialization")
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Building dataset...", flush=True)
    ds = Cifar10(
        batch_size=BATCH_SIZE, x_transform="identity", label_mode="pm1",
        linear_projection=None, rescale=True, shuffle=True,
    )
    ds.build(jax.random.PRNGKey(0))

    results_dir = HERE / "results"
    figs_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    perms = compute_permutations(ds, N_EPOCHS)
    probe_indices = pick_probes_earliest_per_class(
        ds.y_train, perms[0], PROBES_PER_CLASS,
    )
    first_seen = first_seen_batches(perms, probe_indices, BATCH_SIZE)
    n_batches_per_epoch = len(ds._train_bounds)
    total_batches = N_EPOCHS * n_batches_per_epoch

    probe_x_flat = ds.x_train[jnp.asarray(probe_indices)]
    probe_y = ds.y_train[jnp.asarray(probe_indices)]
    probe_x = to_hwc(probe_x_flat)
    probe_classes = np.asarray(jnp.argmax(probe_y, axis=-1))
    n_probes = len(probe_indices)

    print(f"Probes per class       : {PROBES_PER_CLASS} (total {n_probes})", flush=True)
    print(f"First-seen batch range : "
          f"{min(first_seen)}..{max(first_seen)}", flush=True)
    print(f"Total batches planned  : {total_batches} "
          f"({N_EPOCHS} × {n_batches_per_epoch})", flush=True)

    print("\nBuilding model...", flush=True)
    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(CONFIG, mk)
    opt, opt_state = make_train_optimizer(orch, CONFIG)

    # Capture initial weight matrices for drift tracking
    initial_W_out = np.array(orch.lmap[2][1].W)
    initial_J1_kernel = np.array(orch.lmap[1][1].kernel)
    initial_W_in_kernel = np.array(orch.lmap[1][0].kernel)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=CONFIG["warmup_n_iter"],
        train_clamped_n_iter=CONFIG["clamped_n_iter"],
        train_free_n_iter=CONFIG["free_n_iter"],
        eval_n_iter=CONFIG["free_n_iter"],
    )

    measure = make_measure_fn(CONFIG)
    meas_rng = jax.random.PRNGKey(7777)

    # Storage
    measure_batches: list[int] = []
    overlap_data = {k: [] for k in ["A_B", "A_C", "A_D", "B_C", "B_D", "C_D"]}
    margins_C: list[np.ndarray] = []
    margins_D: list[np.ndarray] = []

    # NEW: per-probe J1 baselines (J1 state at first measurement after first-seen)
    # Stored as numpy arrays once set.
    baseline_J1_C: list[np.ndarray | None] = [None] * n_probes
    baseline_J1_D: list[np.ndarray | None] = [None] * n_probes

    # NEW: per-probe per-checkpoint stability (overlap vs baseline)
    # NaN until baseline is captured (for batches before probe's first-seen).
    stability_C_per_probe: list[list[float]] = [[] for _ in range(n_probes)]
    stability_D_per_probe: list[list[float]] = [[] for _ in range(n_probes)]

    # NEW: weight drift metrics per checkpoint
    wout_l2_drift: list[float] = []
    j1_l2_drift: list[float] = []
    win_l2_drift: list[float] = []
    wout_class_cosine: list[list[float]] = []

    def record_measurement(batch_count: int) -> None:
        out = measure(trainer.orchestrator, trainer.state, probe_x, probe_y, meas_rng)
        j1_A = np.asarray(out["j1_A"])
        j1_B = np.asarray(out["j1_B"])
        j1_C = np.asarray(out["j1_C"])
        j1_D = np.asarray(out["j1_D"])

        pairs = {
            "A_B": pairwise_match_per_image(out["j1_A"], out["j1_B"]),
            "A_C": pairwise_match_per_image(out["j1_A"], out["j1_C"]),
            "A_D": pairwise_match_per_image(out["j1_A"], out["j1_D"]),
            "B_C": pairwise_match_per_image(out["j1_B"], out["j1_C"]),
            "B_D": pairwise_match_per_image(out["j1_B"], out["j1_D"]),
            "C_D": pairwise_match_per_image(out["j1_C"], out["j1_D"]),
        }
        for k, v in pairs.items():
            overlap_data[k].append(np.asarray(v))
        margins_C.append(np.asarray(out["margin_C"]))
        margins_D.append(np.asarray(out["margin_D"]))
        measure_batches.append(batch_count)

        # Per-probe stability: capture baseline at first measurement after
        # probe's first-seen batch, then track overlap to baseline.
        for pi in range(n_probes):
            if baseline_J1_C[pi] is None and batch_count >= first_seen[pi]:
                baseline_J1_C[pi] = j1_C[pi].copy()
                baseline_J1_D[pi] = j1_D[pi].copy()
                stability_C_per_probe[pi].append(1.0)  # by definition
                stability_D_per_probe[pi].append(1.0)
            elif baseline_J1_C[pi] is not None:
                sc = float(np.mean(
                    np.sign(j1_C[pi]) == np.sign(baseline_J1_C[pi])
                ).astype(np.float32))
                sd = float(np.mean(
                    np.sign(j1_D[pi]) == np.sign(baseline_J1_D[pi])
                ).astype(np.float32))
                stability_C_per_probe[pi].append(sc)
                stability_D_per_probe[pi].append(sd)
            else:
                stability_C_per_probe[pi].append(np.nan)
                stability_D_per_probe[pi].append(np.nan)

        # Weight drift
        drift = weight_drift_metrics(
            trainer.orchestrator,
            initial_W_out, initial_J1_kernel, initial_W_in_kernel,
        )
        wout_l2_drift.append(drift["wout_l2_drift"])
        j1_l2_drift.append(drift["j1_l2_drift"])
        win_l2_drift.append(drift["win_l2_drift"])
        wout_class_cosine.append(drift["wout_class_cosine"])

        n_baselined = sum(1 for b in baseline_J1_C if b is not None)
        print(f"  batch {batch_count:5d}  "
              f"⟨B-C⟩={float(np.mean(pairs['B_C'])):.4f}  "
              f"⟨C-D⟩={float(np.mean(pairs['C_D'])):.4f}  "
              f"⟨margin_D⟩={float(np.mean(margins_D[-1])):.3f}  "
              f"||W_out drift||={drift['wout_l2_drift']:.3f}  "
              f"baselined={n_baselined}/{n_probes}", flush=True)

    print("\nInitial measurement (batch 0)...", flush=True)
    record_measurement(0)

    print("\nTraining...", flush=True)
    t_train = time.time()
    batch_count = 0
    train_key = jax.random.PRNGKey(SEED + 1)
    for epoch in range(N_EPOCHS):
        for xb, yb in ds:
            train_key = trainer.train_step(to_hwc(xb), yb, train_key)
            trainer.orchestrator = normalize_win(trainer.orchestrator)
            batch_count += 1
            if batch_count % MEASURE_EVERY == 0 or batch_count == total_batches:
                record_measurement(batch_count)
        print(f"  --- epoch {epoch + 1}/{N_EPOCHS} done ({batch_count} batches) ---",
              flush=True)
    print(f"\nTraining loop done in {time.time() - t_train:.1f}s.", flush=True)

    measure_batches_arr = np.asarray(measure_batches)
    aligned_batches = (
        measure_batches_arr[None, :] - np.asarray(first_seen)[:, None]
    )

    # Stack per-probe stability arrays — length is consistent (all probes
    # measured at all checkpoints).
    stability_C = np.array(stability_C_per_probe)   # (n_probes, n_meas)
    stability_D = np.array(stability_D_per_probe)

    results = {
        "config": CONFIG,
        "probes_per_class": PROBES_PER_CLASS,
        "n_epochs": N_EPOCHS,
        "probe_indices": probe_indices,
        "probe_classes": probe_classes.tolist(),
        "first_seen": list(map(int, first_seen)),
        "measure_batches": list(map(int, measure_batches_arr)),
        "aligned_batches": aligned_batches.astype(int).tolist(),
        "margin_C": np.stack(margins_C, axis=1).tolist(),
        "margin_D": np.stack(margins_D, axis=1).tolist(),
        "stability_C": stability_C.tolist(),
        "stability_D": stability_D.tolist(),
        "wout_l2_drift": wout_l2_drift,
        "j1_l2_drift": j1_l2_drift,
        "win_l2_drift": win_l2_drift,
        "wout_class_cosine": wout_class_cosine,
    }
    for k in overlap_data:
        results[f"overlap_{k}"] = np.stack(overlap_data[k], axis=1).tolist()

    out_path = results_dir / "per_image_trajectory_v2.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved trajectories to {out_path}")

    # Plots
    plot_data = {
        "aligned_batches": aligned_batches,
        "margin_C": np.stack(margins_C, axis=1),
        "margin_D": np.stack(margins_D, axis=1),
        "stability_C": stability_C,
        "stability_D": stability_D,
        "measure_batches": measure_batches_arr,
        "wout_l2_drift": wout_l2_drift,
        "j1_l2_drift": j1_l2_drift,
        "win_l2_drift": win_l2_drift,
    }
    for k in overlap_data:
        plot_data[f"overlap_{k}"] = np.stack(overlap_data[k], axis=1)

    main_path = figs_dir / "per_image_trajectory_v2.png"
    plot_main_figure(plot_data, probe_classes, main_path)
    print(f"Saved main plot to {main_path}")

    wout_path = figs_dir / "per_image_trajectory_v2_wout_cosine.png"
    plot_wout_cosine({
        "measure_batches": measure_batches_arr,
        "wout_class_cosine": np.array(wout_class_cosine),
    }, wout_path)
    print(f"Saved W_out cosine plot to {wout_path}")

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    bc = np.stack(overlap_data["B_C"], axis=1)
    cd = np.stack(overlap_data["C_D"], axis=1)
    md = np.stack(margins_D, axis=1)
    print(f"  {'batch':>6}  {'⟨B-C⟩':>8}  {'⟨C-D⟩':>8}  "
          f"{'⟨stab_C⟩':>9}  {'⟨stab_D⟩':>9}  {'⟨margin_D⟩':>11}  {'W_out drift':>11}")
    for ti, bc_t in enumerate(measure_batches_arr):
        # mean stability over probes that have been baselined
        stab_c_t = stability_C[:, ti]; stab_d_t = stability_D[:, ti]
        stab_c_mean = float(np.nanmean(stab_c_t)) if np.isfinite(stab_c_t).any() else np.nan
        stab_d_mean = float(np.nanmean(stab_d_t)) if np.isfinite(stab_d_t).any() else np.nan
        print(f"  {int(bc_t):>6}  {float(np.mean(bc[:, ti])):>8.4f}  "
              f"{float(np.mean(cd[:, ti])):>8.4f}  "
              f"{stab_c_mean:>9.4f}  {stab_d_mean:>9.4f}  "
              f"{float(np.mean(md[:, ti])):>11.4f}  "
              f"{wout_l2_drift[ti]:>11.4f}")


if __name__ == "__main__":
    main()
