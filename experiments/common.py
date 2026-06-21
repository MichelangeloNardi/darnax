"""Shared building blocks for the entropy CIFAR-10 experiments.

Import this from every experiment so the model, optimizer, dataset and training
loop live in ONE place. Each experiment script only sets the few things it tweaks.

The network (fixed across experiments):
  (1,0) Conv2D                  W_in    3->16, 5x5            feedforward, threshold-Hebb
  (1,1) Conv2DRecurrentDiscrete J1      16ch, 5x5            recurrent core + entropy rule
  (1,2) ChannelWBack            W_back  10->16 broadcast     label feedback, FROZEN
  (2,1) PooledFlattenFC         W_out   8x8 pool->256->10    readout, perceptron rule

Training rollout per batch (DynamicalTrainer): warmup -> clamped -> free, then the
local rule updates the weights at the REACHED state.
  - clamped phase ("all" messages) injects the label y through W_back   -> state B
  - free phase  (forward-only) relaxes it                               -> state C
So a readout trained inside this rollout is trained on C. Setting clamped_n_iter=0
removes the label-injection phase, so the rollout is warmup -> free = D (the
inference state). That single switch is how we "train W_out on D instead of C".

Evaluation never has the label, so eval_step is always warmup -> free = D.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

# ── architecture constants (fixed across all experiments) ─────────────────────
C, KSIZE = 16, 5
H, W, POOL = 32, 32, 8


# ── config / data ─────────────────────────────────────────────────────────────

def load_cfg(path) -> dict:
    """Load a replicate/ JSON config. Extra metadata keys are harmless (we only
    index the hyperparameters we need)."""
    return json.load(open(path))


def get_dataset(batch_size: int = 32) -> Cifar10:
    ds = Cifar10(batch_size=batch_size, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))
    return ds


def to_hwc(xb):
    """Flat CIFAR batch (N, 3072) in [0,1] -> (N, 32, 32, 3) in [-1, 1]."""
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    """8x8 avg-pool the J1 activation (N, 32, 32, 16) -> flat (N, 256)."""
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, key):
    """Build the (state, orchestrator) for the channel-entropy architecture."""
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0, key=keys[0],
                      padding_mode="constant", lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                               strength=1.0, threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=layer_map)


def reinit_wout(orch, key):
    """Return a copy of the orchestrator with a freshly initialised W_out."""
    new_wout = PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                               strength=1.0, threshold=5.0, key=key, lr=1.0, weight_decay=0.0)
    return eqx.tree_at(lambda o: o.lmap[2][1], orch, new_wout)


# ── optimizer ─────────────────────────────────────────────────────────────────

def make_optimizer(orch, cfg: dict, win=True, j1=True, wout=True):
    """SGD with per-edge learning rates. Set any of win/j1/wout=False to freeze it.
    W_back is always frozen. (W_in and J1 descend their margin -> negative lr.)"""
    mom = cfg["momentum"]
    params, _ = eqx.partition(orch, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, jx), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(lambda m, r=i, c=jx: m.lmap[r][c], labels,
                             replace=like(params.lmap[i][jx], lbl))

    opt = optax.multi_transform({
        "default": optax.set_to_zero(),
        "win":  sgd(-cfg["lr_win"]) if win else optax.set_to_zero(),
        "j1":   sgd(-cfg["lr_j"]) if j1 else optax.set_to_zero(),
        "wout": sgd(cfg["lr_wout"]) if wout else optax.set_to_zero(),
    }, labels)
    return opt, opt.init(eqx.filter(orch, eqx.is_inexact_array))


# ── trainer ───────────────────────────────────────────────────────────────────

def make_trainer(orch, state, opt, opt_state, cfg: dict, clamped_n=None):
    """Build a DynamicalTrainer.

    clamped_n controls which state the weights are trained on:
      None  -> cfg["clamped_n_iter"]  (warmup->clamped->free = C, the default)
      0     -> no clamped phase        (warmup->free = D, the inference state)

    eval_n_iter is set to free_n_iter so the evaluated state (D) uses the same
    number of free steps as the state the readout was trained on.
    """
    warmup = cfg.get("warmup_n_iter", 1)
    clamped = cfg["clamped_n_iter"] if clamped_n is None else clamped_n
    free = cfg["free_n_iter"]
    return DynamicalTrainer(
        orchestrator=orch, state=state, optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=warmup, train_clamped_n_iter=clamped,
        train_free_n_iter=free, eval_n_iter=free,
    )


# ── training / eval loops ─────────────────────────────────────────────────────

def train_epoch(trainer, ds, key, decay_rate=0.0):
    """One pass over the training set. decay_rate>0 shrinks the W_in and J1
    kernels at the end of the epoch (skip it when those are frozen)."""
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)
    if decay_rate > 0:
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay_rate)
            )
    return trainer, key


def eval_head(trainer, ds, key):
    """Mean test accuracy of the model's own W_out head (eval = D)."""
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def fit_wout(orch, state, ds, cfg, clamped_n, key, epochs):
    """Freeze W_in/J1, re-init W_out, and train ONLY W_out for `epochs` passes.

    clamped_n selects the state W_out is trained on (see make_trainer):
      cfg["clamped_n_iter"] -> C (warmup->clamped->free)
      0                     -> D (warmup->free, the inference state)
    Always evaluated on D. Returns (test_accuracy, key).
    """
    key, wk = jax.random.split(key)
    orch = reinit_wout(orch, wk)
    opt, opt_state = make_optimizer(orch, cfg, win=False, j1=False, wout=True)
    trainer = make_trainer(orch, state, opt, opt_state, cfg, clamped_n=clamped_n)
    for _ in range(epochs):
        trainer, key = train_epoch(trainer, ds, key, decay_rate=0.0)  # backbone frozen
    return eval_head(trainer, ds, key)


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"
