"""experiments4/per_image_trajectory.py

Per-image fixed-point tracking during training, time-aligned by 'first seen'.

Why this is interesting
-----------------------
The aggregate C-D overlap measured in overlap_experiment.py is ≈ 1.0, but that
is local in time: it's measured right after each batch's training update, when
the "hole" the network just dug at state C is still fresh. Mattia's hypothesis
is that C and D drift apart for image X as the network trains on subsequent
images on top of X — a forgetting effect that is averaged out by aggregate
measurements but might be visible per-image when aligned to first-seen.

The agreed-on next test from the meeting (per the follow-up brief):
> pick 10 images (one per class) and, during training, periodically measure
> A, B, C, D, the soft margin, and B-C / C-D overlaps. Plot the curves with
> the x-axis aligned to "when this image was first seen" so all 10 curves can
> be compared on the same timescale.

If B-C and C-D start near 1 and decay over time, the gap is real (forgetting).
If they stay near 1, the 44% accuracy ceiling is a capacity limit, not a
train/inference protocol mismatch.

Outputs
-------
results/per_image_trajectory.json  — per-probe trajectories (overlaps & margins)
figures/per_image_trajectory.png   — multi-panel plot aligned by first-seen

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/per_image_trajectory.py
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

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

C_CH, KSIZE = 16, 5
H, W = 32, 32
POOL = 8
SEED = 0
BATCH_SIZE = 32

N_EPOCHS = 2
MEASURE_EVERY = 50          # batches between measurements
PROBES_PER_CLASS = 5        # 5 × 10 = 50 probes total
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


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# probe selection & first-seen prediction
# ---------------------------------------------------------------------------

def compute_permutations(ds, n_epochs):
    """Replicate Cifar10.__iter__'s key splits to predict the shuffled order.

    Must be called BEFORE any dataset iteration — otherwise ``_train_epoch_key``
    has already advanced and the prediction will be off-by-one. Reads but does
    not mutate the dataset state.
    """
    assert ds._train_epoch_key is not None, "expected shuffle=True with a built key"
    n_train = ds.x_train.shape[0]
    epoch_key = ds._train_epoch_key
    perms = []
    for _ in range(n_epochs):
        key_epoch, epoch_key = jax.random.split(epoch_key)
        perms.append(np.asarray(jax.random.permutation(key_epoch, n_train)))
    return perms


def pick_probes_earliest_per_class(y_train, perm, probes_per_class):
    """For each class, return the indices of its ``probes_per_class`` earliest
    occurrences in ``perm``.

    Walking the permutation in order and taking the first K instances per class
    guarantees that all probes are first-seen within the first ~K * 10 / 32 ≈
    K/3 batches (one per class per ~32 images). This keeps post-first-seen
    trajectories long and comparable in length, and averaging over K per class
    smooths out per-instance idiosyncrasies.

    Returns indices in class-major order:
        [class0_probe0, class0_probe1, ..., class1_probe0, ...]
    so probe[k * 10 + c] is the k-th probe of class c (if class-major) — but
    here we return class-major: probe[c * K + k] is the k-th probe of class c.
    """
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
    """For each probe, find the batch index of its first appearance (epoch 0)."""
    perm0 = perms[0]
    return [int(np.where(perm0 == oi)[0][0] // batch_size) for oi in probe_indices]


# ---------------------------------------------------------------------------
# per-image measurement (JIT'd)
# ---------------------------------------------------------------------------

def make_measure_fn(cfg):
    warmup_n = cfg["warmup_n_iter"]
    clamped_n = cfg["clamped_n_iter"]
    free_n = cfg["free_n_iter"]

    @eqx.filter_jit
    def measure(orch, state_template, x, y, rng):
        """Run all four trajectories and predict W_out logits at C and D.

        Returns a dict of per-image arrays:
          j1_{A,B,C,D}      : (B, H, W, C)        — J1 states
          margin_{C,D}      : (B,)                — soft margins from W_out
        """
        state0 = state_template.init(x, y)

        # A: warmup (forward filter)
        (state_a, rng_a), _ = scan_n(
            orch.step, (state0, rng), n_iter=warmup_n, filter_messages="forward",
        )
        # B: clamped from A (all filter, label fed back)
        (state_b, rng_b), _ = scan_n(
            orch.step, (state_a, rng_a), n_iter=clamped_n, filter_messages="all",
        )
        # C: free from B (forward filter, label removed)
        (state_c, rng_c), _ = scan_n(
            orch.step, (state_b, rng_b), n_iter=free_n, filter_messages="forward",
        )
        # D: free from A (forward filter, skip clamped)
        (state_d, rng_d), _ = scan_n(
            orch.step, (state_a, rng_a), n_iter=free_n, filter_messages="forward",
        )

        # W_out predictions from C and D
        state_c_pred, _ = orch.predict(state_c, rng_c)
        state_d_pred, _ = orch.predict(state_d, rng_d)

        margin_c = soft_margin(state_c_pred.readout, y)
        margin_d = soft_margin(state_d_pred.readout, y)

        return {
            "j1_A": state_a[1], "j1_B": state_b[1],
            "j1_C": state_c[1], "j1_D": state_d[1],
            "margin_C": margin_c, "margin_D": margin_d,
        }

    return measure


def soft_margin(logits, y):
    """Per-image margin = correct_class_logit - max(wrong_class_logits).

    ``y`` is in pm1 encoding with one +1 per row; ``logits`` is W_out's output.
    """
    correct = jnp.argmax(y, axis=-1)
    correct_logit = jnp.take_along_axis(logits, correct[:, None], axis=-1)[:, 0]
    mask = jax.nn.one_hot(correct, num_classes=logits.shape[-1], dtype=bool)
    wrong_max = jnp.max(jnp.where(mask, -jnp.inf, logits), axis=-1)
    return correct_logit - wrong_max


def pairwise_match_per_image(s1, s2):
    """Sign-match rate per image (averaged over spatial+channel dims)."""
    axes = tuple(range(1, s1.ndim))
    return jnp.mean((jnp.sign(s1) == jnp.sign(s2)).astype(jnp.float32), axis=axes)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

CMAP_CYCLE = plt.cm.tab10.colors  # 10 distinct colors, one per class


def _plot_overlap_panel(ax, aligned_x, data, key, probe_classes):
    """One panel: per-probe lines colored by class + mean curve over all probes."""
    for pi in range(data.shape[0]):
        mask = aligned_x[pi] >= 0
        if mask.any():
            ax.plot(aligned_x[pi][mask], data[pi][mask], "-",
                    color=CMAP_CYCLE[probe_classes[pi]], alpha=0.35, lw=0.8)
    _plot_mean_curve(ax, aligned_x, data, label="mean over all probes")
    ax.set_title(f"{key.replace('_', '-')} sign-match rate")
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("match rate")
    ax.set_ylim(0.45, 1.02)
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


def plot_trajectories(results, probe_classes, fig_path):
    """Plot per-probe trajectories aligned by first-seen batch.

    Lines are clipped to x >= 0 (post-first-seen). Colors are per class — the
    K probes per class share a color.
    """
    aligned_x = np.array(results["aligned_batches"])  # (N_probes, N_meas)

    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharex=False)

    for ax, key in zip(axes[0], ["A_B", "B_C", "C_D"]):
        _plot_overlap_panel(ax, aligned_x, np.array(results[f"overlap_{key}"]),
                            key, probe_classes)
    for ax, key in zip(axes[1], ["A_C", "A_D", "B_D"]):
        _plot_overlap_panel(ax, aligned_x, np.array(results[f"overlap_{key}"]),
                            key, probe_classes)

    margin_c = np.array(results["margin_C"])
    margin_d = np.array(results["margin_D"])
    _plot_margin_panel(axes[2, 0], aligned_x, margin_c, probe_classes,
                       "Soft margin from state C (training protocol)")
    _plot_margin_panel(axes[2, 1], aligned_x, margin_d, probe_classes,
                       "Soft margin from state D (inference protocol)")

    # margin_C vs margin_D scatter (post-first-seen only, colored by class)
    ax = axes[2, 2]
    for pi in range(margin_c.shape[0]):
        mask = aligned_x[pi] >= 0
        if mask.any():
            ax.scatter(margin_c[pi][mask], margin_d[pi][mask],
                       color=CMAP_CYCLE[probe_classes[pi]],
                       s=12, alpha=0.5, edgecolors="none")
    # Build a class-legend (one entry per class).
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

    fig.suptitle(
        f"Per-image fixed-point trajectories aligned by first-seen batch  "
        f"({margin_c.shape[0]} probes = {PROBES_PER_CLASS}/class, "
        f"{N_EPOCHS} epoch{'s' if N_EPOCHS != 1 else ''}, "
        f"CIFAR-10, entropy rule, trial 34 config)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_curve(ax, aligned_x, data, label):
    """Plot mean curve over all probes, interpolated onto a shared aligned grid."""
    valid = (aligned_x >= 0) & np.isfinite(data)
    if not valid.any():
        return
    xs_pos = aligned_x[valid]
    xmin = int(xs_pos.min())
    xmax = int(xs_pos.max())
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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

    # Replicate the dataset's shuffle ahead of iteration so we know exactly
    # which batches will contain each probe. Pick PROBES_PER_CLASS earliest
    # examples per class — keeps all probes first-seen in the first ~few
    # batches with comparable-length post-first-seen trajectories.
    perms = compute_permutations(ds, N_EPOCHS)
    probe_indices = pick_probes_earliest_per_class(
        ds.y_train, perms[0], PROBES_PER_CLASS,
    )
    first_seen = first_seen_batches(perms, probe_indices, BATCH_SIZE)
    n_batches_per_epoch = len(ds._train_bounds)
    total_batches = N_EPOCHS * n_batches_per_epoch

    probe_x_flat = ds.x_train[jnp.asarray(probe_indices)]   # (P, 3072) in [0, 1]
    probe_y      = ds.y_train[jnp.asarray(probe_indices)]   # (P, 10) pm1
    probe_x      = to_hwc(probe_x_flat)                     # (P, 32, 32, 3) in [-1, +1]
    probe_classes = np.asarray(jnp.argmax(probe_y, axis=-1))  # (P,) class ids

    print(f"Probes per class       : {PROBES_PER_CLASS} (total {len(probe_indices)})",
          flush=True)
    print(f"First-seen batch range : "
          f"{min(first_seen)}..{max(first_seen)}  (mean {np.mean(first_seen):.1f})",
          flush=True)
    print(f"Total batches planned  : {total_batches} "
          f"({N_EPOCHS} × {n_batches_per_epoch})", flush=True)

    print("\nBuilding model...", flush=True)
    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(CONFIG, mk)
    opt, opt_state = make_train_optimizer(orch, CONFIG)

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

    measure_batches: list[int] = []
    overlap_data = {k: [] for k in ["A_B", "A_C", "A_D", "B_C", "B_D", "C_D"]}
    margins_C: list[np.ndarray] = []
    margins_D: list[np.ndarray] = []

    def record_measurement(batch_count: int) -> None:
        out = measure(trainer.orchestrator, trainer.state, probe_x, probe_y, meas_rng)
        # per-image pairwise sign-match (over spatial+channel)
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
        # Brief progress line.
        print(f"  batch {batch_count:5d}  "
              f"⟨B-C⟩={float(np.mean(pairs['B_C'])):.4f}  "
              f"⟨C-D⟩={float(np.mean(pairs['C_D'])):.4f}  "
              f"⟨margin_D⟩={float(np.mean(margins_D[-1])):.3f}",
              flush=True)

    # Initial measurement (batch 0, random init).
    print("\nInitial measurement (random init)...", flush=True)
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

    # Stack into (N_probes, N_meas) arrays.
    measure_batches_arr = np.asarray(measure_batches)               # (N_meas,)
    # aligned_batches[pi, t] = measure_batches[t] - first_seen[pi]
    aligned_batches = (
        measure_batches_arr[None, :] - np.asarray(first_seen)[:, None]
    )

    results = {
        "config": CONFIG,
        "probes_per_class": PROBES_PER_CLASS,
        "n_epochs": N_EPOCHS,
        "probe_indices": probe_indices,
        "probe_classes": probe_classes.tolist(),
        "first_seen": list(map(int, first_seen)),
        "measure_batches": list(map(int, measure_batches_arr)),
        "aligned_batches": aligned_batches.astype(int).tolist(),
        "margin_C": np.stack(margins_C, axis=1).tolist(),  # (N_probes, N_meas)
        "margin_D": np.stack(margins_D, axis=1).tolist(),
    }
    for k in overlap_data:
        results[f"overlap_{k}"] = np.stack(overlap_data[k], axis=1).tolist()

    out_path = results_dir / "per_image_trajectory.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved trajectories to {out_path}")

    plot_path = figs_dir / "per_image_trajectory.png"
    results_for_plot = {
        "aligned_batches": aligned_batches,
        "margin_C": np.stack(margins_C, axis=1),
        "margin_D": np.stack(margins_D, axis=1),
    }
    for k in overlap_data:
        results_for_plot[f"overlap_{k}"] = np.stack(overlap_data[k], axis=1)
    plot_trajectories(results_for_plot, probe_classes, plot_path)
    print(f"Saved plot to {plot_path}")

    # Brief summary table.
    print(f"\n{'=' * 78}")
    print("SUMMARY  (mean over probes of B-C, C-D, margin_D)")
    print(f"{'=' * 78}")
    bc = np.stack(overlap_data["B_C"], axis=1)
    cd = np.stack(overlap_data["C_D"], axis=1)
    md = np.stack(margins_D, axis=1)
    print(f"  {'batch':>6}  {'⟨B-C⟩':>8}  {'⟨C-D⟩':>8}  {'⟨margin_D⟩':>11}")
    for ti, bc_t in enumerate(measure_batches_arr):
        print(f"  {int(bc_t):>6}  {float(np.mean(bc[:, ti])):>8.4f}  "
              f"{float(np.mean(cd[:, ti])):>8.4f}  "
              f"{float(np.mean(md[:, ti])):>11.4f}")


if __name__ == "__main__":
    main()
