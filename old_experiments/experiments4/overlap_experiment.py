"""overlap_experiment.py

Measures the overlap between the clamped fixed point (C) and the free fixed
point (D) of the J1 state throughout one training pass on CIFAR-10.

For each regime we re-build the model with the same hyper-parameters but
different (warmup, clamped, free) iteration counts:
  - current     : short warmup, moderate clamped/free
  - long_warmup : long warmup, moderate clamped/free
  - contrastive : short warmup, long matched clamped/free

A measurement at training step t consists of:
  state₀ = state.init(x_meas, y_meas)
  state_warm = run warmup phase (forward only) from state₀
  state_C    = run clamped phase ("all" messages, label fed back) from state_warm
  state_D    = run free  phase ("forward") from state_warm
  overlap_match = mean( sign(state_C[1]) == sign(state_D[1]) )
  overlap_dot   = mean( state_C[1] * state_D[1] )         # in [-1, +1]

Run on w01 (single epoch, ~5-10 minutes):
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/overlap_experiment.py
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
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer
from darnax.trainers.utils import scan_n

C, KSIZE = 16, 5
H, W = 32, 32
POOL = 8
SEED = 0
MEASURE_EVERY = 100  # batches

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
}

REGIMES = {
    "current":     {"warmup": 1,  "clamped": 2,  "free": 2},
    "long_warmup": {"warmup": 15, "clamped": 2,  "free": 2},
    "contrastive": {"warmup": 1,  "clamped": 10, "free": 10},
}


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------

def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0,
                      key=keys[0], padding_mode="constant",
                      lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(channels=C, kernel_size=KSIZE, groups=1,
                                       j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                                       key=keys[1], padding_mode="constant",
                                       lr=1.0, weight_decay=0.0,
                                       entropy_beta=cfg["entropy_beta"],
                                       lambda_entropy=1.0),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                               strength=1.0, threshold=5.0,
                               key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def make_optimizer(orch, cfg):
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


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Overlap measurement (JIT-compiled per regime via static n_iter args)
# ---------------------------------------------------------------------------

def make_overlap_fn(warmup_n: int, clamped_n: int, free_n: int):
    """Build a JIT'd measurement function with the given static iteration counts."""

    @eqx.filter_jit
    def overlap_fn(orch, state, x, y, rng):
        state = state.init(x, y)

        # warmup phase (forward only)
        (state, rng), _ = scan_n(
            orch.step, (state, rng), n_iter=warmup_n, filter_messages="forward",
        )
        state_warm = state

        # clamped phase (label feedback) → state C
        (state_c, _), _ = scan_n(
            orch.step, (state_warm, rng), n_iter=clamped_n, filter_messages="all",
        )

        # free phase (no feedback) → state D
        (state_d, _), _ = scan_n(
            orch.step, (state_warm, rng), n_iter=free_n, filter_messages="forward",
        )

        s_c = state_c[1]  # J1 state, (B, 32, 32, 16)
        s_d = state_d[1]

        match_rate = jnp.mean((jnp.sign(s_c) == jnp.sign(s_d)).astype(jnp.float32))
        dot = jnp.mean(s_c * s_d)
        return match_rate, dot

    return overlap_fn


# ---------------------------------------------------------------------------
# One regime: full training pass + periodic overlap measurement
# ---------------------------------------------------------------------------

def run_regime(cfg: dict, regime_name: str, regime_cfg: dict, ds: Cifar10):
    print(f"\n=== {regime_name}  (warmup={regime_cfg['warmup']}, "
          f"clamped={regime_cfg['clamped']}, free={regime_cfg['free']}) ===", flush=True)
    t0 = time.time()

    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(orch, cfg)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=regime_cfg["warmup"],
        train_clamped_n_iter=regime_cfg["clamped"],
        train_free_n_iter=regime_cfg["free"],
        eval_n_iter=5,
    )

    overlap_fn = make_overlap_fn(
        regime_cfg["warmup"], regime_cfg["clamped"], regime_cfg["free"],
    )

    # Fixed measurement batch (first batch of the dataset)
    x_meas = y_meas = None
    for xb, yb in ds:
        x_meas, y_meas = to_hwc(xb), yb
        break

    batch_idx, ov_match, ov_dot = [], [], []

    # Initial measurement at step 0 (random init)
    m, d = overlap_fn(trainer.orchestrator, trainer.state, x_meas, y_meas, key)
    batch_idx.append(0); ov_match.append(float(m)); ov_dot.append(float(d))
    print(f"  batch    0 (init)  match={float(m):.4f}  dot={float(d):.4f}", flush=True)

    decay = cfg["kernel_decay_rate"]
    batch_count = 0
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)

        # Normalize Win filters per output channel
        win_k = trainer.orchestrator.lmap[1][0].kernel
        kh, kw, ci, co = win_k.shape
        flat = win_k.reshape(-1, co)
        normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[1][0].kernel,
            trainer.orchestrator,
            normed.reshape(kh, kw, ci, co),
        )
        batch_count += 1

        if batch_count % MEASURE_EVERY == 0:
            m, d = overlap_fn(trainer.orchestrator, trainer.state, x_meas, y_meas, key)
            batch_idx.append(batch_count); ov_match.append(float(m)); ov_dot.append(float(d))
            print(f"  batch {batch_count:4d}            match={float(m):.4f}  "
                  f"dot={float(d):.4f}", flush=True)

    # Final kernel decay (matches replicate)
    for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
        trainer.orchestrator = eqx.tree_at(
            path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay),
        )

    # Final measurement
    m, d = overlap_fn(trainer.orchestrator, trainer.state, x_meas, y_meas, key)
    batch_idx.append(batch_count); ov_match.append(float(m)); ov_dot.append(float(d))

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s   final match={float(m):.4f}  dot={float(d):.4f}",
          flush=True)
    return batch_idx, ov_match, ov_dot, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    results_dir = HERE / "results"
    figs_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    results = {}
    for name, regime_cfg in REGIMES.items():
        bidx, om, od, elapsed = run_regime(CONFIG, name, regime_cfg, ds)
        results[name] = {
            "regime_cfg": regime_cfg,
            "batch_idx": bidx,
            "overlap_match": om,
            "overlap_dot": od,
            "elapsed_s": elapsed,
        }

    out_path = results_dir / "overlap_experiment.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved to {out_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"current": "steelblue", "long_warmup": "green", "contrastive": "darkorange"}

    for name, res in results.items():
        cfg = res["regime_cfg"]
        label = f"{name} (w={cfg['warmup']}, c={cfg['clamped']}, f={cfg['free']})"
        ax1.plot(res["batch_idx"], res["overlap_match"], "-o",
                 color=colors[name], label=label, markersize=4)
        ax2.plot(res["batch_idx"], res["overlap_dot"], "-o",
                 color=colors[name], label=label, markersize=4)

    ax1.set_title("Sign match rate: mean(sign(C) == sign(D))")
    ax1.set_xlabel("Training batch"); ax1.set_ylabel("Match rate")
    ax1.set_ylim(0.45, 1.02); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.set_title("Mean overlap: mean(C · D)")
    ax2.set_xlabel("Training batch"); ax2.set_ylabel("Mean C·D")
    ax2.set_ylim(-0.1, 1.05); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle("Clamped vs free fixed-point overlap during 1 training epoch "
                 "(CIFAR-10, entropy rule, trial 34 config)", fontsize=11)
    fig.tight_layout()
    fig_path = figs_dir / "overlap_experiment.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {fig_path}")

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY (final overlap values after 1 epoch)")
    print(f"{'='*65}")
    for name, res in results.items():
        print(f"  {name:13s}: match={res['overlap_match'][-1]:.4f}   "
              f"dot={res['overlap_dot'][-1]:.4f}   "
              f"(time: {res['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
