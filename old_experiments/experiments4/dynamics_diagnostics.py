"""experiments4/dynamics_diagnostics.py

Three diagnostics on the trained Config 3 (Matei W_out-tuned) network:

  (B) h_i decomposition by source matrix
      For one fixed measurement batch, at multiple training checkpoints, compute
      the local field at every J1 neuron as
          h_i  =  h_i^{W_in}  +  h_i^{J}  +  h_i^{W_back}
      using each module's forward pass evaluated on the appropriate state slice.
      We then track the mean magnitude of each source contribution over training
      — this directly answers Mattia's "does W_in dominate over J?" question.

  (C) Autocorrelation across dynamics steps
      For one batch at multiple training checkpoints, run the warmup / clamped /
      free phases STEP BY STEP (not via scan_n, which would erase intermediate
      states), capturing s(t) at every inner step. Then plot dot-product overlap
      ⟨s(t)·s(t+1)⟩ vs step number within each phase. Tells us how fast each
      phase converges — and whether Config 3's long phase counts (6, 11, 14)
      are over-specified.

  (A) Continuous h_i distribution (pre-sign)
      Same h_i values as in (B), but viewed as a histogram. Are most neurons
      far from zero (confident commitments) or near zero (borderline)?
      How does the distribution shape evolve with training?

All three share the same model and training loop. Measurements run every few
hundred batches on a fixed measurement batch (so the comparison across time is
controlled). Single seed, Config 3 hyperparameters.

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/dynamics_diagnostics.py
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

# Checkpoints at which to take the three diagnostics
CHECKPOINTS = [0, 50, 200, 1000, 3000]

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


# ---------------------------------------------------------------------------
# Diagnostic measurements
# ---------------------------------------------------------------------------

@eqx.filter_jit
def compute_C_state(orch, state_template, x, y, rng, warmup_n, clamped_n, free_n):
    """Run warmup → clamped → free, return the full state at C (and B)."""
    state0 = state_template.init(x, y)
    (state_a, rng_a), _ = scan_n(orch.step, (state0, rng),
                                 n_iter=warmup_n, filter_messages="forward")
    (state_b, rng_b), _ = scan_n(orch.step, (state_a, rng_a),
                                 n_iter=clamped_n, filter_messages="all")
    (state_c, rng_c), _ = scan_n(orch.step, (state_b, rng_b),
                                 n_iter=free_n, filter_messages="forward")
    return state_a, state_b, state_c


def decompose_h_at_state(orch, state):
    """Decompose the J1 local field into its three source contributions.

    Returns three arrays of shape (B, H, W, C_CH):
      h_win    — Conv2D(state[0])              (input image contribution)
      h_J      — Conv2DRecurrentDiscrete(state[1])  (recurrent self contribution)
      h_wback  — ChannelWBack(state[2])         (label feedback contribution)
    """
    h_win   = orch.lmap[1][0](state[0])
    h_J     = orch.lmap[1][1](state[1])
    h_wback = orch.lmap[1][2](state[2])
    return h_win, h_J, h_wback


def run_phase_step_by_step(orch, state_init, rng, n_steps, filter_messages):
    """Run a phase one step at a time, returning the J1 state at every step."""
    states = [np.asarray(state_init[1])]
    state, key = state_init, rng
    for _ in range(n_steps):
        state, key = orch.step(state, rng=key, filter_messages=filter_messages)
        states.append(np.asarray(state[1]))
    return states, state, key


def overlap_dot(s_a, s_b):
    """Dot-product overlap for sign-valued states (= 2·match − 1)."""
    return float(np.mean(s_a * s_b))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diagnostic_B(checkpoints, contrib_means, contrib_stds, fig_path):
    """Mean |h| contribution per source vs training batch."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sources = ["W_in", "J", "W_back"]
    colors = ["tab:green", "tab:blue", "tab:red"]
    markers = ["^", "s", "o"]
    for src, col, mk in zip(sources, colors, markers):
        means = np.array([contrib_means[cp][src] for cp in checkpoints])
        stds  = np.array([contrib_stds[cp][src]  for cp in checkpoints])
        ax.errorbar(checkpoints, means, yerr=stds, marker=mk,
                    color=col, label=f"⟨|h^{{{src}}}|⟩", capsize=3)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("training batch")
    ax.set_ylabel("mean |contribution to h_i|")
    ax.set_title("(B) Decomposition of J1 local field by source matrix  "
                 "(measured at the C state)")
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_C(checkpoints, autocorr_data, fig_path):
    """Within-phase autocorrelation curves at every checkpoint, one panel per phase."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    phase_names = ["warmup", "clamped", "free"]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(checkpoints)))

    for ax, phase in zip(axes, phase_names):
        for cp, col in zip(checkpoints, cmap):
            curve = autocorr_data[cp][phase]   # list of <s(t), s(t+1)>
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, "-o", color=col, label=f"batch {cp}",
                    markersize=4)
        ax.set_title(f"{phase} phase")
        ax.set_xlabel("step t (within phase)")
        ax.set_ylabel("⟨s(t-1) · s(t)⟩")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("(C) Step-to-step J1 autocorrelation within each phase",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_A(checkpoints, h_total_per_cp, fig_path):
    """Histograms of total h_i (pre-sign) at each checkpoint, overlaid."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(checkpoints)))
    bins = np.linspace(-3.0, 3.0, 80)

    for cp, col in zip(checkpoints, cmap):
        h_flat = h_total_per_cp[cp].ravel()
        ax.hist(h_flat, bins=bins, histtype="step", lw=2.0,
                color=col, label=f"batch {cp}", density=True)
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("total h_i (J1 local field, pre-sign)")
    ax.set_ylabel("density")
    ax.set_title("(A) Distribution of the J1 local field over training")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
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

    # Fixed measurement batch
    x_meas = y_meas = None
    for xb, yb in ds:
        x_meas, y_meas = to_hwc(xb), yb
        break
    assert x_meas is not None

    contrib_means: dict = {}
    contrib_stds: dict = {}
    autocorr_data: dict = {}
    h_total_per_cp: dict = {}

    def measure_at(batch_count):
        meas_rng = jax.random.PRNGKey(2_000 + batch_count)

        # (B) + (A): decomposition at the C state
        state_a, state_b, state_c = compute_C_state(
            trainer.orchestrator, trainer.state, x_meas, y_meas, meas_rng,
            CONFIG["warmup_n_iter"], CONFIG["clamped_n_iter"],
            CONFIG["free_n_iter"],
        )
        h_win, h_J, h_wback = decompose_h_at_state(trainer.orchestrator, state_c)
        h_total = h_win + h_J + h_wback

        contrib_means[batch_count] = {
            "W_in":   float(jnp.mean(jnp.abs(h_win))),
            "J":      float(jnp.mean(jnp.abs(h_J))),
            "W_back": float(jnp.mean(jnp.abs(h_wback))),
        }
        contrib_stds[batch_count] = {
            "W_in":   float(jnp.std(jnp.abs(h_win))),
            "J":      float(jnp.std(jnp.abs(h_J))),
            "W_back": float(jnp.std(jnp.abs(h_wback))),
        }
        h_total_per_cp[batch_count] = np.asarray(h_total)

        # (C) Autocorrelation: rerun each phase step by step
        state0 = trainer.state.init(x_meas, y_meas)

        states_w, state_a2, key2 = run_phase_step_by_step(
            trainer.orchestrator, state0, meas_rng,
            CONFIG["warmup_n_iter"], "forward",
        )
        states_c, state_b2, key3 = run_phase_step_by_step(
            trainer.orchestrator, state_a2, key2,
            CONFIG["clamped_n_iter"], "all",
        )
        states_f, _, _ = run_phase_step_by_step(
            trainer.orchestrator, state_b2, key3,
            CONFIG["free_n_iter"], "forward",
        )

        autocorr_data[batch_count] = {
            "warmup":  [overlap_dot(states_w[t], states_w[t-1])
                        for t in range(1, len(states_w))],
            "clamped": [overlap_dot(states_c[t], states_c[t-1])
                        for t in range(1, len(states_c))],
            "free":    [overlap_dot(states_f[t], states_f[t-1])
                        for t in range(1, len(states_f))],
        }

        print(f"  batch {batch_count:5d}  "
              f"|h_win|={contrib_means[batch_count]['W_in']:.3f}  "
              f"|h_J|={contrib_means[batch_count]['J']:.3f}  "
              f"|h_back|={contrib_means[batch_count]['W_back']:.3f}  "
              f"warmup_step1_corr={autocorr_data[batch_count]['warmup'][0]:.3f}",
              flush=True)

    pending = list(CHECKPOINTS)

    print("\nInitial diagnostic at batch 0...", flush=True)
    if pending[0] == 0:
        measure_at(0)
        pending.pop(0)

    print("\nTraining + diagnostics...", flush=True)
    t_train = time.time()
    batch_count = 0
    train_key = jax.random.PRNGKey(SEED + 1)
    done = False
    while not done:
        for xb, yb in ds:
            train_key = trainer.train_step(to_hwc(xb), yb, train_key)
            trainer.orchestrator = normalize_win(trainer.orchestrator)
            batch_count += 1
            if pending and batch_count >= pending[0]:
                measure_at(batch_count)
                pending.pop(0)
                if not pending:
                    done = True
                    break

    print(f"\nLoop done in {time.time() - t_train:.1f}s "
          f"({batch_count} batches).", flush=True)

    # Save raw data (drop the full h arrays — they're large)
    raw = {
        "config": CONFIG,
        "checkpoints": list(contrib_means.keys()),
        "contrib_means": contrib_means,
        "contrib_stds":  contrib_stds,
        "autocorrelation": autocorr_data,
    }
    out_path = results_dir / "dynamics_diagnostics.json"
    out_path.write_text(json.dumps(raw, indent=2, default=float))
    print(f"\nSaved metrics to {out_path}")

    cps = sorted(contrib_means.keys())
    plot_diagnostic_B(cps, contrib_means, contrib_stds,
                      figs_dir / "diag_B_decomposition.png")
    print(f"Saved (B) to {figs_dir / 'diag_B_decomposition.png'}")

    plot_diagnostic_C(cps, autocorr_data,
                      figs_dir / "diag_C_autocorrelation.png")
    print(f"Saved (C) to {figs_dir / 'diag_C_autocorrelation.png'}")

    plot_diagnostic_A(cps, h_total_per_cp,
                      figs_dir / "diag_A_h_distribution.png")
    print(f"Saved (A) to {figs_dir / 'diag_A_h_distribution.png'}")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"{'batch':>6}  {'|h_win|':>8}  {'|h_J|':>8}  {'|h_back|':>8}  "
          f"{'W_in/J':>7}")
    for cp in cps:
        cm = contrib_means[cp]
        ratio = cm["W_in"] / max(cm["J"], 1e-8)
        print(f"{cp:>6}  {cm['W_in']:>8.4f}  {cm['J']:>8.4f}  "
              f"{cm['W_back']:>8.4f}  {ratio:>7.3f}")


if __name__ == "__main__":
    main()
