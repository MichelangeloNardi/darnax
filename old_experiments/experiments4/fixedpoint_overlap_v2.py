"""experiments4/fixedpoint_overlap_v2.py

Experiment 1 (v2) — same 16×16 A/B/C/D × {none, win, j, both} matrix as
fixedpoint_overlap.py, with two changes per Michelangelo's request:

  (Q2) Use the CURRENT batch (the one the trainer is about to consume at
       this checkpoint) instead of a fixed first batch. At every checkpoint
       we pull a fresh batch from the dataset, measure the 16-point matrix
       on it, then train on it. The measurement and the training therefore
       see the same images.

  Finer early checkpoints: the agent's run showed the matrix collapses from
       block-structured to uniformly 1.0 somewhere between batch 0 and 100.
       New checkpoint grid {0, 1, 5, 10, 25, 50, 100, 250, 500, 1000} pins
       down where that transition happens.

Update is still K = 1 per the meeting; we'll add a K > 1 burst in a follow-up
run if K = 1 stays below noise.

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/fixedpoint_overlap_v2.py
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

CHECKPOINTS = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
ABLATIONS = ["none", "win", "j", "both"]
PHASE_LABELS = ["A", "B", "C", "D"]

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


def _label_tree(orch):
    params, _ = eqx.partition(orch, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(
            lambda m, r=i, c=j: m.lmap[r][c], labels,
            replace=like(params.lmap[i][j], lbl),
        )
    return labels


def make_train_optimizer(orch, cfg):
    mom = cfg["momentum"]

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = _label_tree(orch)
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


def apply_ablated_update(orch, grads, ablation, cfg, labels):
    """K=1 SGD step with fresh momentum-free optimizer, mask by ablation type."""
    if ablation == "none":
        return orch

    lr_win = -cfg["lr_win"] if ablation in ("win", "both") else 0.0
    lr_j = -cfg["lr_j"] if ablation in ("j", "both") else 0.0

    opt = optax.multi_transform({
        "default": optax.sgd(0.0),
        "win":     optax.sgd(lr_win),
        "j1":      optax.sgd(lr_j),
        "wout":    optax.sgd(0.0),
    }, labels)

    params = eqx.filter(orch, eqx.is_inexact_array)
    opt_state = opt.init(params)
    grads_filtered = eqx.filter(grads, eqx.is_inexact_array)
    updates, _ = opt.update(grads_filtered, opt_state, params=params)
    new_orch = eqx.apply_updates(orch, updates)

    if ablation in ("win", "both"):
        new_orch = normalize_win(new_orch)
    return new_orch


def make_measure_fns(cfg):
    warmup_n = cfg["warmup_n_iter"]
    clamped_n = cfg["clamped_n_iter"]
    free_n = cfg["free_n_iter"]

    @eqx.filter_jit
    def measure_abcd_with_grads(orch, state_template, x, y, rng):
        state0 = state_template.init(x, y)
        (state_a, rng_a), _ = scan_n(orch.step, (state0, rng),
                                     n_iter=warmup_n, filter_messages="forward")
        (state_b, rng_b), _ = scan_n(orch.step, (state_a, rng_a),
                                     n_iter=clamped_n, filter_messages="all")
        (state_c, rng_c), _ = scan_n(orch.step, (state_b, rng_b),
                                     n_iter=free_n, filter_messages="forward")
        (state_d, _), _ = scan_n(orch.step, (state_a, rng_a),
                                 n_iter=free_n, filter_messages="forward")
        grads = orch.backward(state_c, rng=rng_c)
        return (state_a[1], state_b[1], state_c[1], state_d[1]), grads

    @eqx.filter_jit
    def measure_abcd(orch, state_template, x, y, rng):
        state0 = state_template.init(x, y)
        (state_a, rng_a), _ = scan_n(orch.step, (state0, rng),
                                     n_iter=warmup_n, filter_messages="forward")
        (state_b, rng_b), _ = scan_n(orch.step, (state_a, rng_a),
                                     n_iter=clamped_n, filter_messages="all")
        (state_c, _), _ = scan_n(orch.step, (state_b, rng_b),
                                 n_iter=free_n, filter_messages="forward")
        (state_d, _), _ = scan_n(orch.step, (state_a, rng_a),
                                 n_iter=free_n, filter_messages="forward")
        return state_a[1], state_b[1], state_c[1], state_d[1]

    return measure_abcd_with_grads, measure_abcd


def overlap_matrices(j1_list):
    n = len(j1_list)
    match = np.zeros((n, n), dtype=np.float32)
    dot = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = j1_list[i]
        for j in range(n):
            sj = j1_list[j]
            match[i, j] = float(jnp.mean((jnp.sign(si) == jnp.sign(sj)).astype(jnp.float32)))
            dot[i, j] = float(jnp.mean(si * sj))
    return match, dot


def plot_heatmaps(all_results, fig_path):
    cps = sorted(all_results.keys())
    n_cp = len(cps)
    n_cols = min(5, n_cp)
    n_rows = 2 * int(np.ceil(n_cp / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 4.0 * n_rows))
    if n_cp == 1:
        axes = axes.reshape(2, 1)
    axes = np.atleast_2d(axes)
    block_lines = [4, 8, 12]

    for ci, cp in enumerate(cps):
        col = ci % n_cols
        row_block = (ci // n_cols) * 2
        res = all_results[cp]
        labels_ = res["labels"]
        match = np.array(res["match"])
        dot = np.array(res["dot"])

        for offset, (data, title, vmin) in enumerate([
            (match, f"sign-match @ b{cp}", 0.4),
            (dot,   f"dot @ b{cp}",        -0.2),
        ]):
            ax = axes[row_block + offset, col]
            im = ax.imshow(data, vmin=vmin, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(labels_)))
            ax.set_yticks(range(len(labels_)))
            ax.set_xticklabels(labels_, rotation=90, fontsize=6)
            ax.set_yticklabels(labels_, fontsize=6)
            for bl in block_lines:
                ax.axhline(bl - 0.5, color="white", lw=0.5, alpha=0.5)
                ax.axvline(bl - 0.5, color="white", lw=0.5, alpha=0.5)
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.045)

    # Hide unused cells (in case n_cp not a multiple of n_cols)
    used = n_cp
    for ci in range(used, n_cols * (n_rows // 2)):
        col = ci % n_cols
        row_block = (ci // n_cols) * 2
        for offset in (0, 1):
            axes[row_block + offset, col].axis("off")

    fig.suptitle(
        "Experiment 1 v2 — 16×16 J1 overlap matrix per checkpoint  "
        "(current-batch measurement, K=1 ablations)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_saturation_curve(all_results, fig_path):
    """Trace key off-diagonal overlaps (A-D, A-C, A-B, B-C, C-D) vs checkpoint
    to pin down where the matrix saturates."""
    cps = sorted(all_results.keys())
    keys = [("A↔B", 0, 1), ("A↔C", 0, 2), ("A↔D", 0, 3),
            ("B↔C", 1, 2), ("B↔D", 1, 3), ("C↔D", 2, 3),
            ("none↔win (A)", 0, 4), ("none↔j (A)", 0, 8), ("none↔both (A)", 0, 12)]
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, i, j in keys:
        ys = [np.array(all_results[cp]["match"])[i, j] for cp in cps]
        ax.plot(cps, ys, "-o", label=label, alpha=0.85, markersize=4)
    ax.set_xlabel("training batch")
    ax.set_ylabel("sign-match rate")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.set_title("Saturation of the 16×16 J1 overlap matrix during early training")
    ax.grid(alpha=0.3)
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

    print("Building model...", flush=True)
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

    measure_with_grads, measure_only = make_measure_fns(CONFIG)
    labels = _label_tree(orch)

    point_labels = [f"{p}_{ab}" for ab in ABLATIONS for p in PHASE_LABELS]
    all_results: dict[int, dict] = {}

    def measure_on_batch(batch_count: int, x_meas, y_meas) -> None:
        t0 = time.time()
        meas_rng = jax.random.PRNGKey(1_000 + batch_count)

        (j_a, j_b, j_c, j_d), grads = measure_with_grads(
            trainer.orchestrator, trainer.state, x_meas, y_meas, meas_rng,
        )
        all_j1 = [j_a, j_b, j_c, j_d]

        for ab in ABLATIONS[1:]:
            new_orch = apply_ablated_update(
                trainer.orchestrator, grads, ab, CONFIG, labels,
            )
            j_a2, j_b2, j_c2, j_d2 = measure_only(
                new_orch, trainer.state, x_meas, y_meas, meas_rng,
            )
            all_j1.extend([j_a2, j_b2, j_c2, j_d2])

        match, dot = overlap_matrices(all_j1)
        all_results[batch_count] = {
            "match": match.tolist(),
            "dot": dot.tolist(),
            "labels": point_labels,
        }
        print(f"  batch {batch_count:5d}  "
              f"A↔D={match[0, 3]:.3f}  C↔D={match[2, 3]:.3f}  "
              f"none↔both={match[0, 12]:.3f}  ({time.time() - t0:.1f}s)",
              flush=True)

    pending = list(CHECKPOINTS)
    train_key = jax.random.PRNGKey(SEED + 1)
    batch_count = 0

    print("\nTraining + measuring...", flush=True)
    t_train = time.time()
    for xb, yb in ds:
        x_hwc = to_hwc(xb)

        # Batch 0 = pre-training measurement, using THIS batch.
        if pending and pending[0] == 0 and batch_count == 0:
            measure_on_batch(0, x_hwc, yb)
            pending.pop(0)

        # Train on this batch.
        train_key = trainer.train_step(x_hwc, yb, train_key)
        trainer.orchestrator = normalize_win(trainer.orchestrator)
        batch_count += 1

        # Post-train measurement at checkpoint batches — use the batch we just
        # trained on (current batch, per Q2).
        if pending and batch_count >= pending[0]:
            measure_on_batch(batch_count, x_hwc, yb)
            pending.pop(0)

        if not pending:
            break

    print(f"\nLoop done in {time.time() - t_train:.1f}s "
          f"({batch_count} batches).", flush=True)

    out_path = results_dir / "fixedpoint_overlap_v2.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"Saved metrics to {out_path}")

    fig_path = figs_dir / "fixedpoint_overlap_v2.png"
    plot_heatmaps(all_results, fig_path)
    print(f"Saved heatmaps to {fig_path}")

    fig2_path = figs_dir / "fixedpoint_overlap_v2_saturation.png"
    plot_saturation_curve(all_results, fig2_path)
    print(f"Saved saturation curve to {fig2_path}")

    print(f"\n{'=' * 78}")
    print("SUMMARY  (key off-diagonal entries of the 4×4 baseline block)")
    print(f"{'=' * 78}")
    print(f"  {'batch':>6}  {'A↔B':>7}  {'A↔C':>7}  {'A↔D':>7}  "
          f"{'B↔C':>7}  {'B↔D':>7}  {'C↔D':>7}  {'none↔both':>10}")
    for cp in sorted(all_results):
        m = np.array(all_results[cp]["match"])
        print(f"  {cp:>6}  {m[0, 1]:>7.4f}  {m[0, 2]:>7.4f}  {m[0, 3]:>7.4f}  "
              f"{m[1, 2]:>7.4f}  {m[1, 3]:>7.4f}  {m[2, 3]:>7.4f}  "
              f"{m[0, 12]:>10.4f}")


if __name__ == "__main__":
    main()
