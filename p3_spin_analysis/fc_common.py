"""Shared building blocks for the CLASSICAL (non-conv) FC asymmetric recurrent net.

This is the fully-connected translation of the conv entropy architecture used in
p1/p2, built to check whether the C->D flip phenomenon (a linear probe trained on
the clamped state C collapses on the inference state D via a few class-targeted
spin flips) also appears in the paper's base dense recurrent net -- and to study it
where every spin is a distinct unit (no conv weight-sharing / spatial pooling).

Topology (mirrors p1/common.py edge-for-edge; same DynamicalTrainer + rollout):
  (1,0) W_in    d_in -> N     FrozenFullyConnected (random) OR SparseFullyConnected
  (1,1) J       N recurrent   SparseRecurrentDiscrete (sign activation, j_d diagonal)
  (1,2) W_back  10  -> N      FrozenRescaledFullyConnected (frozen label feedback)
  (2,1) W_out   N   -> 10     FullyConnected (perceptron readout)
  (2,2)         OutputLayer

State = [d_in, N, 10]. C = warmup->clamped(all)->free(forward); D = warmup->free.
Setting clamped_n_iter=0 removes the label-injection phase (-> D). W_back at (1,2)
is a right-going edge, so it only fires under filter_messages="all" (clamped phase).

Input: CIFAR (N,3072) in [0,1] -> 2x2 avg-pool to 16x16x3 = 768 -> [-1,1].
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
from darnax.modules.fully_connected import (
    FrozenFullyConnected,
    FrozenRescaledFullyConnected,
    FullyConnected,
    SparseFullyConnected,
)
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import SparseRecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

# ── architecture constants ─────────────────────────────────────────────────────
N_SPINS = 256                 # hidden units (= number of spins, one distinct unit each)
DS = 2                        # input spatial downsample factor (32 -> 16)
D_IN = (32 // DS) * (32 // DS) * 3   # 768
N_CLASSES = 10
PROBE_WD = 1.433e-4           # Adam linear-probe weight decay (shared with p1/p2)


# ── config / data ──────────────────────────────────────────────────────────────

def load_cfg(path) -> dict:
    return json.load(open(path))


def get_dataset(batch_size: int = 32) -> Cifar10:
    ds = Cifar10(batch_size=batch_size, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))
    return ds


def downsample(xb):
    """Flat CIFAR batch (N,3072) in [0,1] -> 2x2 avg-pooled (N, D_IN) in [-1,1]."""
    x = jnp.asarray(xb).reshape(-1, 32, 32, 3)
    h = 32 // DS
    x = x.reshape(-1, h, DS, h, DS, 3).mean(axis=(2, 4))   # (N,16,16,3)
    return x.reshape(-1, D_IN) * 2.0 - 1.0


# ── model ──────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, key):
    """Build (state, orchestrator) for the FC asymmetric recurrent net.

    cfg["train_win"] (default False): if False, W_in is a FROZEN random projection
    (input featurizer, no updates); if True, W_in is a trainable SparseFullyConnected
    at the same sparsity as J.
    """
    keys = jax.random.split(key, 5)
    N = cfg.get("n_spins", N_SPINS)
    sparsity = cfg["sparsity"]
    train_win = bool(cfg.get("train_win", False))

    if train_win:
        w_in = SparseFullyConnected(
            in_features=D_IN, out_features=N, strength=cfg.get("strength_win", 1.0),
            threshold=cfg["threshold_win"], sparsity=sparsity, key=keys[0],
            lr=1.0, weight_decay=0.0)
    else:
        w_in = FrozenFullyConnected(
            in_features=D_IN, out_features=N, strength=cfg.get("strength_win", 1.0),
            threshold=cfg["threshold_win"], key=keys[0], lr=1.0, weight_decay=0.0)

    layer_map = LayerMap.from_dict({
        1: {
            0: w_in,
            1: SparseRecurrentDiscrete(
                features=N, j_d=cfg["j_d"], sparsity=sparsity,
                threshold=cfg["threshold_j"], key=keys[1], strength=1.0),
            2: FrozenRescaledFullyConnected(
                in_features=N_CLASSES, out_features=N, strength=cfg["strength_back"],
                threshold=0.0, key=keys[2]),
        },
        2: {
            1: FullyConnected(
                in_features=N, out_features=N_CLASSES, strength=1.0,
                threshold=cfg.get("threshold_wout", 5.0), key=keys[3],
                lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    return SequentialState([D_IN, N, N_CLASSES]), SequentialOrchestrator(layers=layer_map)


# ── optimizer (SGD with per-edge lr; W_in/J descend their margin -> negative lr) ─

def make_optimizer(orch, cfg: dict, win=True, j1=True, wout=True):
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


# ── trainer ────────────────────────────────────────────────────────────────────

def make_trainer(orch, state, opt, opt_state, cfg: dict, clamped_n=None):
    warmup = cfg.get("warmup_n_iter", 1)
    clamped = cfg["clamped_n_iter"] if clamped_n is None else clamped_n
    free = cfg["free_n_iter"]
    return DynamicalTrainer(
        orchestrator=orch, state=state, optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=warmup, train_clamped_n_iter=clamped,
        train_free_n_iter=free, eval_n_iter=free,
    )


def train_epoch(trainer, ds, key):
    for xb, yb in ds:
        key = trainer.train_step(downsample(xb), yb, key)
    return trainer, key


def eval_head(trainer, ds, key):
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(downsample(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ── representation rollouts (C = clamped, D = free) ─────────────────────────────

def make_rollout(warmup_n, clamped_n, free_n):
    """warmup -> clamped -> free; clamped_n>0 reaches C, clamped_n=0 reaches D."""
    def rollout(orch, state, key):
        for _ in range(warmup_n):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        for _ in range(clamped_n):
            state, key = orch.step(state, rng=key, filter_messages="all")
        for _ in range(free_n):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        return state, key
    return rollout


def rollers(cfg):
    """(roll_C, roll_D) jitted single-state rollers returning state only."""
    warmup = cfg.get("warmup_n_iter", 1)
    clamped, free = cfg["clamped_n_iter"], cfg["free_n_iter"]
    rC, rD = make_rollout(warmup, clamped, free), make_rollout(warmup, 0, free)

    @eqx.filter_jit
    def fC(orch, state, key):
        return rC(orch, state, key)[0]

    @eqx.filter_jit
    def fD(orch, state, key):
        return rD(orch, state, key)[0]
    return fC, fD


def collect_spins(orch, state_tmpl, ds_iter, roller, key, max_b=None, want_field=False):
    """Roll every batch to its fixed point; return hard-sign spins (N, N_SPINS),
    label indices (N,), and optionally the field (N, N_SPINS). Reps + labels come
    from the SAME pass (the train set reshuffles every iteration)."""
    S, F, Y = [], [], []
    for i, (xb, yb) in enumerate(ds_iter):
        if max_b is not None and i >= max_b:
            break
        s = roller(orch, state_tmpl.init(downsample(xb), yb), key)
        S.append(np.asarray(s[1]))
        if want_field:
            F.append(np.asarray(s.fields[1]))
        Y.append(np.argmax(np.asarray(yb), 1))
    S = np.concatenate(S); Y = np.concatenate(Y)
    F = np.concatenate(F) if want_field else None
    return (S, Y, F) if want_field else (S, Y)


# ── quick closed-form ridge probe (smoke / fast sanity) ─────────────────────────

def ridge_probe_acc(Xtr, ytr, Xte, yte, lam=1.0):
    """Closed-form ridge on ±1 spins -> one-hot; argmax accuracy on the test set."""
    Xtr, Xte = jnp.asarray(Xtr), jnp.asarray(Xte)
    Yoh = jax.nn.one_hot(jnp.asarray(ytr), N_CLASSES)
    d = Xtr.shape[1]
    W = jnp.linalg.solve(Xtr.T @ Xtr + lam * jnp.eye(d), Xtr.T @ Yoh)
    return float(jnp.mean((Xte @ W).argmax(1) == jnp.asarray(yte)))
