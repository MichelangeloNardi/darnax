"""experiments4/cross_time_cd.py

Explicit per-image cross-time, cross-protocol measurement:
    overlap( J1_C(probe, t_baseline),  J1_D(probe, t_later) )

where t_baseline is the first measurement at/after the probe's first-seen
batch, and t_later runs across the rest of training.

Mattia's hypothesis test: the "hole" we dug at C when we learned the probe —
does it survive later updates on other batches? If yes, then for every later
checkpoint the inference fixed point D should land at the same place as the
original training C. If no, D drifts away and we'd see this curve decay.

Same setup as per_image_trajectory_v2.py — 50 probes (5/class) first-seen
early, 2 epochs, every 50 batches a measurement. Also reports the *same-
protocol* baselines (stability_C and stability_D) for comparison, and a
per-class soft-margin breakdown.

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/cross_time_cd.py
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


def dot_per_image(s1, s2):
    """Dot-product overlap per image (averaged over spatial+channel dims)."""
    axes = tuple(range(1, s1.ndim))
    return jnp.mean(s1 * s2, axis=axes)


CMAP_CYCLE = plt.cm.tab10.colors


def _mean_curve(aligned_x, data, label, ax, color="black", lw=2.0):
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
                color=color, lw=lw, label=label)


def plot_results(results, probe_classes, fig_path):
    aligned = np.array(results["aligned_batches"])
    cross = np.array(results["cross_C0_to_D_t"])     # (probes, n_meas) dot product
    stab_C = np.array(results["stability_C_dot"])
    stab_D = np.array(results["stability_D_dot"])
    margin_D = np.array(results["margin_D"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) cross-time C-vs-D'
    ax = axes[0, 0]
    for pi in range(cross.shape[0]):
        mask = aligned[pi] >= 0
        if mask.any():
            ax.plot(aligned[pi][mask], cross[pi][mask],
                    color=CMAP_CYCLE[probe_classes[pi]], alpha=0.4, lw=0.8)
    _mean_curve(aligned, cross, "mean over probes", ax)
    ax.set_title("⟨J1_C(probe, t_baseline) · J1_D(probe, t)⟩\n"
                 "(Mattia's hypothesis: cross-time, cross-protocol)")
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("dot product")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) same-protocol baselines for comparison
    ax = axes[0, 1]
    _mean_curve(aligned, stab_C, "stability_C (same protocol)", ax, color="tab:blue", lw=2)
    _mean_curve(aligned, stab_D, "stability_D (same protocol)", ax, color="tab:red", lw=2)
    _mean_curve(aligned, cross,  "cross C(0) ↔ D(t)",           ax, color="black", lw=2.5)
    ax.set_title("Mean curves: same- vs cross-protocol")
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("dot product")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (c) per-class mean soft margin (D)
    ax = axes[1, 0]
    for c in range(N_CLASSES):
        class_probes = np.where(probe_classes == c)[0]
        if len(class_probes) == 0:
            continue
        class_margin = margin_D[class_probes]   # (probes_in_class, n_meas)
        class_aligned = aligned[class_probes]
        # mean curve per class
        _mean_curve(class_aligned, class_margin, f"class {c}", ax,
                    color=CMAP_CYCLE[c], lw=1.6)
    ax.set_title("Per-class mean soft margin (state D)")
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("margin (correct − max wrong)")
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)

    # (d) per-probe terminal margin (final checkpoint)
    ax = axes[1, 1]
    final_margin = margin_D[:, -1]   # (probes,)
    # group bars by class
    bar_data = []
    for c in range(N_CLASSES):
        mask = probe_classes == c
        if mask.any():
            bar_data.append((c, final_margin[mask]))
    positions = []
    margins_flat = []
    colors_flat = []
    x = 0
    xticks = []
    xticklabels = []
    for c, margs in bar_data:
        for m in margs:
            positions.append(x); margins_flat.append(m); colors_flat.append(CMAP_CYCLE[c])
            x += 1
        xticks.append(x - len(margs) / 2 - 0.5)
        xticklabels.append(f"cl {c}")
        x += 0.5   # gap
    ax.bar(positions, margins_flat, color=colors_flat, width=0.85)
    ax.axhline(0.0, color="k", lw=0.6, ls="-")
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylabel("final margin (state D)")
    ax.set_title("Final margin per probe, grouped by class\n"
                 "(bars above 0 = classified correctly)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Cross-time C↔D' + per-class margin breakdown  "
        f"({PROBES_PER_CLASS} probes/class, {N_EPOCHS} epochs)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Building dataset...", flush=True)
    ds = Cifar10(batch_size=BATCH_SIZE, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True, shuffle=True)
    ds.build(jax.random.PRNGKey(0))

    results_dir = HERE / "results"
    figs_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    perms = compute_permutations(ds, N_EPOCHS)
    probe_indices = pick_probes_earliest_per_class(
        ds.y_train, perms[0], PROBES_PER_CLASS)
    first_seen = first_seen_batches(perms, probe_indices, BATCH_SIZE)
    n_batches_per_epoch = len(ds._train_bounds)
    total_batches = N_EPOCHS * n_batches_per_epoch

    probe_x = to_hwc(ds.x_train[jnp.asarray(probe_indices)])
    probe_y = ds.y_train[jnp.asarray(probe_indices)]
    probe_classes = np.asarray(jnp.argmax(probe_y, axis=-1))
    n_probes = len(probe_indices)

    print(f"Probes per class       : {PROBES_PER_CLASS} (total {n_probes})", flush=True)
    print(f"First-seen batch range : {min(first_seen)}..{max(first_seen)}", flush=True)
    print(f"Total batches planned  : {total_batches}", flush=True)

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
    margin_C_list: list[np.ndarray] = []
    margin_D_list: list[np.ndarray] = []
    stability_C_list: list[list[float]] = [[] for _ in range(n_probes)]
    stability_D_list: list[list[float]] = [[] for _ in range(n_probes)]
    cross_C0_D_list: list[list[float]] = [[] for _ in range(n_probes)]

    baseline_J1_C: list[np.ndarray | None] = [None] * n_probes
    baseline_J1_D: list[np.ndarray | None] = [None] * n_probes

    def record(batch_count):
        out = measure(trainer.orchestrator, trainer.state, probe_x, probe_y, meas_rng)
        j1_C = np.asarray(out["j1_C"])
        j1_D = np.asarray(out["j1_D"])
        margin_C_list.append(np.asarray(out["margin_C"]))
        margin_D_list.append(np.asarray(out["margin_D"]))
        measure_batches.append(batch_count)
        for pi in range(n_probes):
            if baseline_J1_C[pi] is None and batch_count >= first_seen[pi]:
                baseline_J1_C[pi] = j1_C[pi].copy()
                baseline_J1_D[pi] = j1_D[pi].copy()
                stability_C_list[pi].append(1.0)   # dot product (binary)
                stability_D_list[pi].append(1.0)
                cross_C0_D_list[pi].append(float(np.mean(
                    np.sign(j1_D[pi]) * np.sign(baseline_J1_C[pi]))))
            elif baseline_J1_C[pi] is not None:
                stability_C_list[pi].append(float(np.mean(
                    np.sign(j1_C[pi]) * np.sign(baseline_J1_C[pi]))))
                stability_D_list[pi].append(float(np.mean(
                    np.sign(j1_D[pi]) * np.sign(baseline_J1_D[pi]))))
                cross_C0_D_list[pi].append(float(np.mean(
                    np.sign(j1_D[pi]) * np.sign(baseline_J1_C[pi]))))
            else:
                stability_C_list[pi].append(np.nan)
                stability_D_list[pi].append(np.nan)
                cross_C0_D_list[pi].append(np.nan)
        baselined = sum(1 for b in baseline_J1_C if b is not None)
        print(f"  batch {batch_count:5d}  "
              f"⟨stab_C⟩={float(np.nanmean([sc[-1] for sc in stability_C_list])):.3f}  "
              f"⟨stab_D⟩={float(np.nanmean([sd[-1] for sd in stability_D_list])):.3f}  "
              f"⟨C0↔D(t)⟩={float(np.nanmean([cc[-1] for cc in cross_C0_D_list])):.3f}  "
              f"⟨margin_D⟩={float(np.mean(margin_D_list[-1])):.3f}  "
              f"baselined={baselined}/{n_probes}", flush=True)

    print("\nInitial measurement...", flush=True)
    record(0)

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
                record(batch_count)
        print(f"  --- epoch {epoch + 1}/{N_EPOCHS} done ---", flush=True)
    print(f"\nDone in {time.time() - t_train:.1f}s", flush=True)

    measure_batches_arr = np.asarray(measure_batches)
    aligned_batches = (measure_batches_arr[None, :]
                       - np.asarray(first_seen)[:, None])

    results = {
        "config": CONFIG,
        "probes_per_class": PROBES_PER_CLASS,
        "n_epochs": N_EPOCHS,
        "probe_indices": probe_indices,
        "probe_classes": probe_classes.tolist(),
        "first_seen": list(map(int, first_seen)),
        "measure_batches": list(map(int, measure_batches_arr)),
        "aligned_batches": aligned_batches.astype(int).tolist(),
        "margin_C": np.stack(margin_C_list, axis=1).tolist(),
        "margin_D": np.stack(margin_D_list, axis=1).tolist(),
        "stability_C_dot": [sc for sc in stability_C_list],
        "stability_D_dot": [sd for sd in stability_D_list],
        "cross_C0_to_D_t": [cc for cc in cross_C0_D_list],
    }
    out_path = results_dir / "cross_time_cd.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved to {out_path}")

    plot_data = {
        "aligned_batches": aligned_batches,
        "margin_C": np.stack(margin_C_list, axis=1),
        "margin_D": np.stack(margin_D_list, axis=1),
        "stability_C_dot": np.array(stability_C_list),
        "stability_D_dot": np.array(stability_D_list),
        "cross_C0_to_D_t": np.array(cross_C0_D_list),
    }
    fig_path = figs_dir / "cross_time_cd.png"
    plot_results(plot_data, probe_classes, fig_path)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
