"""experiments4/fixedpoint_overlap.py

Experiment 1 — pairwise overlap of (A, B, C, D) J1 fixed points before and
after one learning step, then extended to 16x16 with update ablations.

J1 fixed points (state[1] read at four distinct moments for the same (x, y)):

    A = state after warmup        (filter="forward"; label inactive — backward
                                   edge from output to J1 excluded)
    B = state after clamped from A (filter="all";    label fed back via W_back)
    C = state after free from B   (filter="forward"; label removed —
                                   the negative-phase state the local rule
                                   actually consumes during training)
    D = state after free from A   (filter="forward"; skip clamped entirely —
                                   the pure inference trajectory)

If C ≈ D, the clamped phase imprint is washed out by the free relaxation —
i.e. the network already produces the correct fixed point under inference.

For the 16-point analysis we apply the local rule once with four ablations:
    "none" -> no update                (gives back the same A, B, C, D)
    "win"  -> only W_in (lmap[1][0]) is updated
    "j"    -> only J     (lmap[1][1]) is updated
    "both" -> W_in + J both updated    (W_out is downstream of J1 → never
                                       updated in these ablations)

Each ablation step uses a *fresh* momentum-free optimizer, so we measure the
pure effect of one gradient on the J1 fixed-point geometry — no carryover from
prior training momentum.

Repeated at training checkpoints {0, 100, 500, 1000} batches to see how the
geometry evolves with learning.

Run on w01 (heavy: ~1000 training batches + ~16 fixed-point computations per
checkpoint, expect ~5-15 min on a single GPU):
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/fixedpoint_overlap.py
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

CHECKPOINTS = [0, 100, 500, 1000]      # batches at which to run 16x16 measurement
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


# ---------------------------------------------------------------------------
# model / training optimizer
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


def _label_tree(orch):
    """Per-leaf string labels: 'win' for lmap[1][0], 'j1' for lmap[1][1],
    'wout' for lmap[2][1], 'default' for everything else."""
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
    """Training optimizer matching replicate_channel_entropy / overlap_experiment."""
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
    """L2-normalize Win kernel per output channel (matches replicate train loop)."""
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
# ablation step (fresh optimizer, no momentum carryover — option a)
# ---------------------------------------------------------------------------

def apply_ablated_update(orch, grads, ablation, cfg, labels):
    """Apply one SGD step to ``orch`` using ``grads`` (output of orch.backward),
    but only update the matrices selected by ``ablation``.

    Uses a brand-new optimizer state every call so there is no momentum
    carryover from previous training steps — this isolates the effect of a
    *single gradient* on the J1 representation, rather than the trainer's
    cumulative trajectory.

    Parameters
    ----------
    orch : SequentialOrchestrator
        Pre-update orchestrator.
    grads : SequentialOrchestrator
        Per-edge local updates (output of ``orch.backward(state, rng)``).
    ablation : str
        One of ``"none"``, ``"win"``, ``"j"``, ``"both"``.
    cfg : dict
        Config (uses ``lr_win`` and ``lr_j``).
    labels : pytree
        Label tree built once via ``_label_tree(orch)``.

    Returns
    -------
    SequentialOrchestrator
        New orchestrator after the masked update. Win kernel is L2-renormalized
        per output channel if Win was updated (matches training-loop behavior).
    """
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


# ---------------------------------------------------------------------------
# measurement (JIT'd; static phase counts captured via closure)
# ---------------------------------------------------------------------------

def make_measure_fns(cfg):
    warmup_n = cfg["warmup_n_iter"]
    clamped_n = cfg["clamped_n_iter"]
    free_n = cfg["free_n_iter"]

    @eqx.filter_jit
    def measure_abcd_with_grads(orch, state_template, x, y, rng):
        """Return (A, B, C, D) J1 states and the grads at state C."""
        state0 = state_template.init(x, y)
        (state_a, rng_a), _ = scan_n(
            orch.step, (state0, rng), n_iter=warmup_n, filter_messages="forward",
        )
        (state_b, rng_b), _ = scan_n(
            orch.step, (state_a, rng_a), n_iter=clamped_n, filter_messages="all",
        )
        (state_c, rng_c), _ = scan_n(
            orch.step, (state_b, rng_b), n_iter=free_n, filter_messages="forward",
        )
        (state_d, _), _ = scan_n(
            orch.step, (state_a, rng_a), n_iter=free_n, filter_messages="forward",
        )
        grads = orch.backward(state_c, rng=rng_c)
        return (state_a[1], state_b[1], state_c[1], state_d[1]), grads

    @eqx.filter_jit
    def measure_abcd(orch, state_template, x, y, rng):
        """Return (A, B, C, D) J1 states (no grads — used on ablated orchs)."""
        state0 = state_template.init(x, y)
        (state_a, rng_a), _ = scan_n(
            orch.step, (state0, rng), n_iter=warmup_n, filter_messages="forward",
        )
        (state_b, rng_b), _ = scan_n(
            orch.step, (state_a, rng_a), n_iter=clamped_n, filter_messages="all",
        )
        (state_c, _), _ = scan_n(
            orch.step, (state_b, rng_b), n_iter=free_n, filter_messages="forward",
        )
        (state_d, _), _ = scan_n(
            orch.step, (state_a, rng_a), n_iter=free_n, filter_messages="forward",
        )
        return state_a[1], state_b[1], state_c[1], state_d[1]

    return measure_abcd_with_grads, measure_abcd


def overlap_matrices(j1_list):
    """Pairwise sign-match-rate and mean-dot for J1 states.

    For sign-valued states in {-1, +1}, dot = 2 * match_rate - 1, so the two
    metrics are linearly related. We report both for sanity / readability.

    Parameters
    ----------
    j1_list : list[Array]
        Length-n list of arrays with identical shape (B, ...).

    Returns
    -------
    match : np.ndarray, shape (n, n)
    dot   : np.ndarray, shape (n, n)
    """
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


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_heatmaps(all_results, fig_path):
    cps = sorted(all_results.keys())
    n_cp = len(cps)
    fig, axes = plt.subplots(2, n_cp, figsize=(4.7 * n_cp, 9.4))
    if n_cp == 1:
        axes = axes.reshape(2, 1)

    # Group block boundaries in the 16x16 (every 4 entries is one ablation)
    block_lines = [4, 8, 12]

    for ci, cp in enumerate(cps):
        res = all_results[cp]
        labels_ = res["labels"]
        match = np.array(res["match"])
        dot = np.array(res["dot"])

        for row, (data, title, vmin) in enumerate([
            (match, f"Sign-match rate  @ batch {cp}", 0.4),
            (dot,   f"Mean dot ⟨s_i·s_j⟩ @ batch {cp}", -0.2),
        ]):
            ax = axes[row, ci]
            im = ax.imshow(data, vmin=vmin, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(labels_)))
            ax.set_yticks(range(len(labels_)))
            ax.set_xticklabels(labels_, rotation=90, fontsize=7)
            ax.set_yticklabels(labels_, fontsize=7)
            for bl in block_lines:
                ax.axhline(bl - 0.5, color="white", lw=0.6, alpha=0.5)
                ax.axvline(bl - 0.5, color="white", lw=0.6, alpha=0.5)
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.045)

    fig.suptitle(
        "(A, B, C, D) J1 overlap × {none, win, j, both} update ablations  "
        "(CIFAR-10, entropy rule, trial 34 config)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Building dataset...", flush=True)
    ds = Cifar10(
        batch_size=BATCH_SIZE, x_transform="identity", label_mode="pm1",
        linear_projection=None, rescale=True,
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

    # Fixed measurement batch (first batch of the dataset).
    x_meas = y_meas = None
    for xb, yb in ds:
        x_meas, y_meas = to_hwc(xb), yb
        break

    # Stable labels for the 16 measurement points: A_none, B_none, ..., D_both.
    point_labels = [f"{p}_{ab}" for ab in ABLATIONS for p in PHASE_LABELS]

    all_results: dict[int, dict] = {}
    pending_checkpoints = list(CHECKPOINTS)

    def measure_now(batch_count: int) -> None:
        print(f"\n=== Measuring at batch {batch_count} ===", flush=True)
        t0 = time.time()
        meas_rng = jax.random.PRNGKey(1_000 + batch_count)  # reproducible per-cp

        # Baseline (A, B, C, D) on current orch, plus the local update grads.
        (j_a, j_b, j_c, j_d), grads = measure_with_grads(
            trainer.orchestrator, trainer.state, x_meas, y_meas, meas_rng,
        )
        all_j1 = [j_a, j_b, j_c, j_d]  # ABLATIONS[0] = "none" → baseline values

        for ab in ABLATIONS[1:]:  # win, j, both
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
        elapsed = time.time() - t0
        print(f"  16×16 in {elapsed:.1f}s  "
              f"(A vs D match = {match[0, 3]:.3f},  "
              f"C vs D match = {match[2, 3]:.3f})", flush=True)

    # Initial measurement at batch 0 (random init).
    if pending_checkpoints and pending_checkpoints[0] == 0:
        measure_now(0)
        pending_checkpoints.pop(0)

    print("\nTraining...", flush=True)
    t_train = time.time()
    batch_count = 0
    train_key = jax.random.PRNGKey(SEED + 1)
    for xb, yb in ds:
        train_key = trainer.train_step(to_hwc(xb), yb, train_key)
        trainer.orchestrator = normalize_win(trainer.orchestrator)
        batch_count += 1

        if pending_checkpoints and batch_count >= pending_checkpoints[0]:
            measure_now(batch_count)
            pending_checkpoints.pop(0)
            if not pending_checkpoints:
                break

    print(f"\nTraining loop done in {time.time() - t_train:.1f}s "
          f"({batch_count} batches).", flush=True)

    out_path = results_dir / "fixedpoint_overlap.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"Saved metrics to {out_path}")

    fig_path = figs_dir / "fixedpoint_overlap.png"
    plot_heatmaps(all_results, fig_path)
    print(f"Saved heatmaps to {fig_path}")

    # Brief summary table.
    print(f"\n{'=' * 78}")
    print("SUMMARY  (key off-diagonal entries of the 4×4 baseline block)")
    print(f"{'=' * 78}")
    print(f"  {'batch':>6}  {'A↔B':>7}  {'A↔C':>7}  {'A↔D':>7}  "
          f"{'B↔C':>7}  {'B↔D':>7}  {'C↔D':>7}")
    for cp in sorted(all_results):
        m = np.array(all_results[cp]["match"])
        print(f"  {cp:>6}  {m[0, 1]:>7.4f}  {m[0, 2]:>7.4f}  {m[0, 3]:>7.4f}  "
              f"{m[1, 2]:>7.4f}  {m[1, 3]:>7.4f}  {m[2, 3]:>7.4f}")


if __name__ == "__main__":
    main()
