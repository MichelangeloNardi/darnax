"""experiments4/short_dynamics_test.py

Does Config 3 (Matei W_out-tuned hyperparameters) still work if we strip the
dynamics down to a classic 1-5-5 (warmup-clamped-free)?

Same lr's, thresholds, decay, momentum, j_d, entropy_beta as Config 3 — only
the iteration counts change. Trains for 5 epochs, reports head accuracy.

If accuracy stays >40% we have a "short-dynamics Config 3" that's *much* more
informative for the A/B/C/D geometry experiments (the long-dynamics version
saturates all overlaps to 1.0 within 100 batches, leaving nothing to study).

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/short_dynamics_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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
EPOCHS = 5

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

DYNAMICS = {
    "config3_long_(6,11,14)":  {"warmup": 6, "clamped": 11, "free": 14},
    "config3_short_(1,5,5)":   {"warmup": 1, "clamped": 5,  "free": 5},
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


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def run_one(label, dyn, ds):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
    cfg = BASE_CFG | dyn
    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_train_optimizer(orch, cfg)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=dyn["warmup"],
        train_clamped_n_iter=dyn["clamped"],
        train_free_n_iter=dyn["free"],
        eval_n_iter=dyn["free"],
    )

    decay = cfg["kernel_decay_rate"]
    head_accs = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
            win_k = trainer.orchestrator.lmap[1][0].kernel
            kh, kw, ci, co = win_k.shape
            flat = win_k.reshape(-1, co)
            normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            trainer.orchestrator = eqx.tree_at(
                lambda o: o.lmap[1][0].kernel,
                trainer.orchestrator,
                normed.reshape(kh, kw, ci, co),
            )
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay),
            )
        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        head = float(np.mean(batch_accs))
        head_accs.append(head)
        print(f"  epoch {epoch}/{EPOCHS}  head={head:.4f}  "
              f"elapsed={time.time() - t0:.1f}s", flush=True)
    return head_accs


def main():
    ds = Cifar10(batch_size=BATCH_SIZE, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    results = {}
    for label, dyn in DYNAMICS.items():
        results[label] = run_one(label, dyn, ds)

    print(f"\n{'=' * 60}\nFINAL HEAD ACCURACY ({EPOCHS} epochs)\n{'=' * 60}")
    for label, accs in results.items():
        print(f"  {label:30s}  epoch1={accs[0]:.4f}  "
              f"epoch{EPOCHS}={accs[-1]:.4f}  best={max(accs):.4f}")


if __name__ == "__main__":
    main()
