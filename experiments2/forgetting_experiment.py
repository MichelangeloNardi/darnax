#!/usr/bin/env python3
"""Sequential forgetting experiment for the simple convolutional Darnax model.

This script is a refactor/extension of the logic from the notebook `3. conv2.ipynb`.
It trains on image A until memorized, then trains on image B until memorized, while
tracking both readout-level and internal dynamics metrics.

Main outputs:
- steps required to memorize A
- steps required to memorize B
- how many B-phase steps both A and B are classified correctly
- trajectories for soft margin, accuracy, alignment, convergence/flips,
  clamp-free distance, and W_back overlap

Notes
-----
- This script assumes it is run inside an environment where the `darnax` package is
  importable and the CIFAR10 dataset helper used in the notebook is available.
- The script uses the same model architecture and helper logic as the notebook.
- A true hidden-margin metric s_i h_i is not included here because the notebook did
  not expose a stable public API for extracting local fields from the modules. The
  internal metrics included here are the ones already supported by the notebook:
  clamp/free distance, flip fraction, W_back overlap, and state overlap across steps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from sklearn.metrics.pairwise import cosine_similarity

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.pooling import GlobalMajorityPooling, GlobalUnpooling
from darnax.modules.fully_connected import FullyConnected, FrozenRescaledFullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState


# -----------------------------------------------------------------------------
# Model + optimization (ported from the notebook)
# -----------------------------------------------------------------------------

def build_conv_model(
    seed: int,
    in_channels: int = 3,
    spatial_size: int = 32,
    n_channels: int = 64,
    num_labels: int = 10,
    input_kernel_size: int = 7,
    recur_kernel_size: int = 7,
    threshold_conv: float = 1.7,
    threshold_fc: float = 1.7,
    threshold_out: float = 1.7,
    threshold_back: float = 0.0,
    j_d_conv: float = 0.9,
    j_d_fc: float = 0.9,
    strength_in: float = 1.0,
    strength_pool: float = 1.0,
    strength_unpool: float = 1.0,
    strength_wout: float = 1.0,
    strength_wback: float = 1.0,
    lr_conv: float = 1.0,
    weight_decay_conv: float = 0.0,
    mask_prob: float = 0.0,
    field_momentum: float = 0.0,
) -> tuple[SequentialState, SequentialOrchestrator]:
    keys = jax.random.split(jax.random.key(seed), 5)
    s = spatial_size

    layer_map = LayerMap.from_dict(
        {
            1: {
                0: Conv2D(
                    in_channels=in_channels,
                    out_channels=n_channels,
                    kernel_size=input_kernel_size,
                    threshold=threshold_conv,
                    strength=strength_in,
                    key=keys[0],
                    padding_mode="constant",
                ),
                1: Conv2DRecurrentDiscrete(
                    channels=n_channels,
                    kernel_size=recur_kernel_size,
                    groups=n_channels,
                    j_d=j_d_conv,
                    threshold=threshold_conv,
                    padding_mode="constant",
                    key=keys[1],
                    lr=lr_conv,
                    weight_decay=weight_decay_conv,
                ),
                2: GlobalUnpooling(strength=strength_unpool, axis=(1, 2)),
            },
            2: {
                1: GlobalMajorityPooling(strength=strength_pool, axis=(1, 2)),
                2: RecurrentDiscrete(
                    features=n_channels,
                    j_d=j_d_fc,
                    threshold=threshold_fc,
                    key=keys[2],
                ),
                3: FrozenRescaledFullyConnected(
                    in_features=num_labels,
                    out_features=n_channels,
                    strength=strength_wback,
                    threshold=threshold_back,
                    key=keys[3],
                ),
            },
            3: {
                2: FullyConnected(
                    in_features=n_channels,
                    out_features=num_labels,
                    strength=strength_wout,
                    threshold=threshold_out,
                    key=keys[4],
                ),
                3: OutputLayer(),
            },
        }
    )

    state = SequentialState(
        [
            (s, s, in_channels),
            (s, s, n_channels),
            (n_channels,),
            (num_labels,),
        ]
    )
    orchestrator = SequentialOrchestrator(
        layers=layer_map,
        field_momentum=field_momentum,
        mask_prob=mask_prob,
    )
    return state, orchestrator


def run_dynamics(
    orchestrator: SequentialOrchestrator,
    state: SequentialState,
    rng,
    n_warmup: int = 1,
    n_clamped: int = 5,
    n_free: int = 5,
) -> tuple[SequentialState, list[SequentialState]]:
    history: list[SequentialState] = []

    for _ in range(n_warmup):
        state, rng = orchestrator.step(state, rng=rng, filter_messages="inference")
        history.append(state)

    for _ in range(n_clamped):
        state, rng = orchestrator.step(state, rng=rng, filter_messages="all")
        history.append(state)

    for _ in range(n_free):
        state, rng = orchestrator.step(state, rng=rng, filter_messages="inference")
        history.append(state)

    return state, history


def build_optimizer(
    orchestrator: SequentialOrchestrator,
    lr_w_in: float = 0.1,
    lr_j_conv: float = 0.05,
    lr_j_fc: float = 0.05,
    lr_w_out: float = 0.1,
    trainable_overrides: dict[tuple[int, int], str] | None = None,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    if trainable_overrides is None:
        trainable_overrides = {
            (1, 0): "w_in",
            (1, 1): "j_conv",
            (2, 2): "j_fc",
            (3, 2): "w_out",
        }

    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)
    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)

    def like(tree, value: str):
        return jtu.tree_map(lambda _: value, tree, is_leaf=eqx.is_array)

    for (i, j), label in trainable_overrides.items():
        labels = eqx.tree_at(
            lambda m, _i=i, _j=j: m.lmap[_i][_j],
            labels,
            replace=like(params.lmap[i][j], label),
        )

    lr_map = {
        "w_in": lr_w_in,
        "j_conv": lr_j_conv,
        "j_fc": lr_j_fc,
        "w_out": lr_w_out,
    }
    transforms = {"default": optax.sgd(learning_rate=0.0)}
    transforms.update({lbl: optax.sgd(learning_rate=lr) for lbl, lr in lr_map.items()})

    optimizer = optax.multi_transform(transforms, labels)
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))
    return optimizer, opt_state


def apply_update(
    orchestrator: SequentialOrchestrator,
    state: SequentialState,
    rng,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[SequentialOrchestrator, optax.OptState]:
    grads = orchestrator.backward(state, rng=rng)
    grads_filtered = eqx.filter(grads, eqx.is_inexact_array)
    params_filtered = eqx.filter(orchestrator, eqx.is_inexact_array)
    updates, new_opt_state = optimizer.update(
        grads_filtered, opt_state, params=params_filtered
    )
    new_orchestrator = eqx.apply_updates(orchestrator, updates)
    return new_orchestrator, new_opt_state


# -----------------------------------------------------------------------------
# Metrics (ported/extended from the notebook)
# -----------------------------------------------------------------------------

def extract_layer(history: list[SequentialState], layer_idx: int) -> list[np.ndarray]:
    return [np.array(h.states[layer_idx]) for h in history]


def flatten_layer(activations: list[np.ndarray]) -> list[np.ndarray]:
    return [a.reshape(a.shape[0], -1) for a in activations]


def compute_neuron_flips(
    history: list[SequentialState],
    layer_idx: int = 1,
    percentage: bool = True,
) -> np.ndarray:
    acts = flatten_layer(extract_layer(history, layer_idx))
    flips = np.zeros(len(acts) - 1)
    for t in range(len(acts) - 1):
        diff = (acts[t] != acts[t + 1]).astype(float)
        if percentage:
            flips[t] = diff.mean() * 100.0
        else:
            flips[t] = diff.mean(axis=1).sum()
    return flips


def clamped_free_distance(
    history: list[SequentialState],
    layer_idx: int,
    n_warmup: int = 1,
    n_clamped: int = 5,
) -> float:
    acts = flatten_layer(extract_layer(history, layer_idx))
    idx_clamped_end = n_warmup + n_clamped - 1
    idx_free_end = len(acts) - 1
    sims = [
        cosine_similarity([acts[idx_clamped_end][b]], [acts[idx_free_end][b]])[0, 0]
        for b in range(acts[idx_clamped_end].shape[0])
    ]
    return float(1.0 - np.nanmean(sims))


def compute_wback_overlap(
    history: list[SequentialState],
    wback_module,
    layer_idx: int = 2,
    output_idx: int = 3,
) -> np.ndarray:
    overlaps = np.zeros(len(history))
    for t in range(len(history)):
        output = np.array(history[t].states[output_idx])
        projected = np.array(wback_module(output))
        layer_act = np.array(history[t].states[layer_idx])
        sims = []
        for b in range(layer_act.shape[0]):
            p, l = projected[b], layer_act[b]
            norm = np.linalg.norm(p) * np.linalg.norm(l) + 1e-8
            sims.append(float(np.dot(p, l) / norm))
        overlaps[t] = np.mean(sims)
    return overlaps


def compute_free_metrics(
    orchestrator: SequentialOrchestrator,
    state_final: SequentialState,
    y_true,
    rng,
) -> tuple[float, float, np.ndarray, Any]:
    rng, pred_rng = jax.random.split(rng)
    pred_state, _ = orchestrator.predict(state_final, pred_rng)
    predicted = np.array(pred_state.states[-1])
    y_true_np = np.array(y_true)

    true_cls = np.argmax(y_true_np, axis=-1)
    pred_cls = np.argmax(predicted, axis=-1)
    accuracy = float(np.mean(pred_cls == true_cls))
    alignment = float(np.mean(np.sign(predicted) * y_true_np))
    return accuracy, alignment, predicted, rng


def compute_soft_margin(
    orchestrator: SequentialOrchestrator,
    state_final: SequentialState,
    y_true,
    rng,
) -> tuple[float, Any]:
    accuracy, alignment, logits, rng = compute_free_metrics(
        orchestrator, state_final, y_true, rng
    )
    del accuracy, alignment
    y_true_np = np.array(y_true)
    true_cls = np.argmax(y_true_np, axis=-1)

    margins = []
    for b in range(logits.shape[0]):
        tc = int(true_cls[b])
        correct_logit = logits[b, tc]
        wrong_logits = np.concatenate([logits[b, :tc], logits[b, tc + 1 :]])
        margins.append(float(correct_logit - wrong_logits.max()))
    return float(np.mean(margins)), rng


def compute_hidden_margin_fc(
    orchestrator: SequentialOrchestrator,
    state: SequentialState,
) -> float:
    """Compute the true hidden margin mean_i(s_i * h_i) for the FC recurrent layer (level 2).

    The full local field is the sum of three message contributions:
      - feedforward from conv layer via GlobalMajorityPooling (lmap[2][1])
      - recurrent self-coupling via RecurrentDiscrete (lmap[2][2]):  h = s @ J
      - feedback from output via FrozenRescaledFullyConnected (lmap[2][3])

    The hidden margin measures how well the current state is a stable fixed
    point of the combined dynamics: a positive value means neurons are on
    average aligned with their incoming field (stable / memorized).

    Parameters
    ----------
    orchestrator : SequentialOrchestrator
    state        : SequentialState  (typically after dynamics have converged)

    Returns
    -------
    float  — mean over batch and neurons of  s_i * h_i
    """
    s1 = jnp.array(state.states[1])   # (batch, H, W, C) conv activations
    s2 = jnp.array(state.states[2])   # (batch, N)       FC activations
    s3 = jnp.array(state.states[3])   # (batch, K)       output activations

    # Feedforward: GlobalMajorityPooling applied to conv layer
    pool_mod   = orchestrator.lmap[2][1]   # GlobalMajorityPooling
    h_ff       = pool_mod(s1)              # (batch, N)

    # Recurrent: s2 @ J  (RecurrentDiscrete.__call__ returns x @ J)
    recur_mod  = orchestrator.lmap[2][2]   # RecurrentDiscrete
    h_rec      = recur_mod(s2)             # (batch, N)

    # Feedback: FrozenRescaledFullyConnected applied to output layer
    wback_mod  = orchestrator.lmap[2][3]   # FrozenRescaledFullyConnected
    h_back     = wback_mod(s3)             # (batch, N)

    h_total    = h_ff + h_rec + h_back     # (batch, N)
    margin     = float(jnp.mean(s2 * h_total))
    return margin


def compute_hidden_margin_conv(
    orchestrator: SequentialOrchestrator,
    state: SequentialState,
) -> float:
    """Compute the true hidden margin mean_i(s_i * h_i) for the conv recurrent layer (level 1).

    The full local field is the sum of three message contributions:
      - feedforward from input via Conv2D (lmap[1][0])
      - recurrent self-coupling via Conv2DRecurrentDiscrete (lmap[1][1])
      - feedback from FC layer via GlobalUnpooling (lmap[1][2])

    Returns
    -------
    float  — mean over batch and all spatial/channel neurons of  s_i * h_i
    """
    s0 = jnp.array(state.states[0])   # (batch, H, W, in_channels)  input
    s1 = jnp.array(state.states[1])   # (batch, H, W, C)  conv activations
    s2 = jnp.array(state.states[2])   # (batch, N)        FC activations

    # Feedforward from input: Conv2D
    conv_in_mod = orchestrator.lmap[1][0]   # Conv2D
    h_ff        = conv_in_mod(s0)           # (batch, H, W, C)

    # Recurrent: Conv2DRecurrentDiscrete applied to s1
    conv_rec_mod = orchestrator.lmap[1][1]  # Conv2DRecurrentDiscrete
    h_rec        = conv_rec_mod(s1)         # (batch, H, W, C)

    # Feedback from FC layer: GlobalUnpooling
    unpool_mod  = orchestrator.lmap[1][2]   # GlobalUnpooling
    h_back      = unpool_mod(s2)            # (batch, H, W, C)

    h_total     = h_ff + h_rec + h_back     # (batch, H, W, C)
    margin      = float(jnp.mean(s1 * h_total))
    return margin


def final_state_overlap(
    state_a: SequentialState,
    state_b: SequentialState,
    layer_idx: int = 2,
) -> float:
    a = np.array(state_a.states[layer_idx]).reshape(np.array(state_a.states[layer_idx]).shape[0], -1)
    b = np.array(state_b.states[layer_idx]).reshape(np.array(state_b.states[layer_idx]).shape[0], -1)
    sims = [cosine_similarity([a[i]], [b[i]])[0, 0] for i in range(a.shape[0])]
    return float(np.mean(sims))


@dataclass
class EvalMetrics:
    accuracy: float
    soft_margin: float
    alignment: float
    converged: bool
    flip_fraction_last: float
    flip_fraction_mean: float
    clamp_free_distance_conv: float
    clamp_free_distance_fc: float
    wback_overlap_last: float
    wback_overlap_mean: float
    hidden_margin_fc: float
    hidden_margin_conv: float
    free_steps: int


def evaluate_sample(
    orchestrator: SequentialOrchestrator,
    state_template: SequentialState,
    x_data,
    y_data,
    rng,
    *,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    conv_layer_idx: int = 1,
    fc_layer_idx: int = 2,
) -> tuple[EvalMetrics, SequentialState, list[SequentialState], Any]:
    s0 = state_template.init(x_data, y_data)
    rng, dyn_rng = jax.random.split(rng)
    state_final, history = run_dynamics(
        orchestrator,
        s0,
        rng=dyn_rng,
        n_warmup=n_warmup,
        n_clamped=n_clamped,
        n_free=n_free,
    )

    flips = compute_neuron_flips(history, layer_idx=conv_layer_idx, percentage=True)
    flip_last = float(flips[-1]) if len(flips) else 0.0
    flip_mean = float(np.mean(flips)) if len(flips) else 0.0
    conv_dist = clamped_free_distance(
        history, layer_idx=conv_layer_idx, n_warmup=n_warmup, n_clamped=n_clamped
    )
    fc_dist = clamped_free_distance(
        history, layer_idx=fc_layer_idx, n_warmup=n_warmup, n_clamped=n_clamped
    )

    wback_module = orchestrator.lmap[2][3]
    overlaps = compute_wback_overlap(history, wback_module, layer_idx=fc_layer_idx, output_idx=3)
    overlap_last = float(overlaps[-1]) if len(overlaps) else float("nan")
    overlap_mean = float(np.mean(overlaps)) if len(overlaps) else float("nan")

    accuracy, alignment, _, rng = compute_free_metrics(orchestrator, state_final, y_data, rng)
    soft_margin, rng = compute_soft_margin(orchestrator, state_final, y_data, rng)

    hm_fc   = compute_hidden_margin_fc(orchestrator, state_final)
    hm_conv = compute_hidden_margin_conv(orchestrator, state_final)

    metrics = EvalMetrics(
        accuracy=accuracy,
        soft_margin=soft_margin,
        alignment=alignment,
        converged=bool(flip_last == 0.0),
        flip_fraction_last=flip_last,
        flip_fraction_mean=flip_mean,
        clamp_free_distance_conv=conv_dist,
        clamp_free_distance_fc=fc_dist,
        wback_overlap_last=overlap_last,
        wback_overlap_mean=overlap_mean,
        hidden_margin_fc=hm_fc,
        hidden_margin_conv=hm_conv,
        free_steps=n_free,
    )
    return metrics, state_final, history, rng


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------

def load_cifar_samples(batch_size: int = 32):
    data = Cifar10(
        batch_size=batch_size,
        linear_projection=None,
        label_mode="pm1",
        x_transform="identity",
    )

    key = jax.random.PRNGKey(0)
    key, data_key = jax.random.split(key)

    data.build(key=data_key)

    x, y = next(iter(data))
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 32, 32, 3)
    return x, y


def load_samples(
    idx_a: int,
    idx_b: int,
    dataset_batch_size: int,
    pair_file: str | None,
):
    if pair_file is not None:
        arr = np.load(pair_file)
        x_a, y_a = arr["x_a"], arr["y_a"]
        x_b, y_b = arr["x_b"], arr["y_b"]
        if x_a.ndim == 3:
            x_a = x_a[None, ...]
        if y_a.ndim == 1:
            y_a = y_a[None, ...]
        if x_b.ndim == 3:
            x_b = x_b[None, ...]
        if y_b.ndim == 1:
            y_b = y_b[None, ...]
        return x_a, y_a, x_b, y_b

    x, y = load_cifar_samples(batch_size=dataset_batch_size)
    if idx_a >= len(x) or idx_b >= len(x):
        raise IndexError(f"Requested indices ({idx_a}, {idx_b}) exceed loaded batch size {len(x)}")
    return x[idx_a : idx_a + 1], y[idx_a : idx_a + 1], x[idx_b : idx_b + 1], y[idx_b : idx_b + 1]


# -----------------------------------------------------------------------------
# Experiment core
# -----------------------------------------------------------------------------

def memorized(metrics_window: list[EvalMetrics], consecutive_k: int) -> bool:
    if len(metrics_window) < consecutive_k:
        return False
    recent = metrics_window[-consecutive_k:]
    return all(m.accuracy >= 1.0 and m.soft_margin > 0.0 and m.converged for m in recent)


def metrics_dict(metrics: EvalMetrics) -> dict[str, Any]:
    return asdict(metrics)


def run_single_seed_forgetting(
    *,
    x_a,
    y_a,
    x_b,
    y_b,
    threshold: float,
    learning_rate: float,
    seed: int,
    momentum: float,
    mask_prob: float,
    max_updates_a: int,
    max_updates_b: int,
    consecutive_k: int,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
) -> dict[str, Any]:
    state_template, orch = build_conv_model(
        seed=seed,
        threshold_conv=threshold,
        threshold_fc=threshold,
        threshold_out=threshold,
        mask_prob=mask_prob,
        field_momentum=momentum,
    )
    opt, opt_state = build_optimizer(
        orch,
        lr_w_in=learning_rate,
        lr_j_conv=learning_rate,
        lr_j_fc=learning_rate,
        lr_w_out=learning_rate,
    )
    rng = jax.random.PRNGKey(seed)

    phase_a: list[dict[str, Any]] = []
    phase_b_a: list[dict[str, Any]] = []
    phase_b_b: list[dict[str, Any]] = []

    last_a_state = None
    steps_to_memorize_a: int | None = None
    steps_to_memorize_b: int | None = None

    # Phase A: train on A until memorized
    recent_a: list[EvalMetrics] = []
    for step in range(max_updates_a + 1):
        m_a, state_a, _, rng = evaluate_sample(
            orch,
            state_template,
            x_a,
            y_a,
            rng,
            n_warmup=n_warmup,
            n_clamped=n_clamped,
            n_free=n_free,
        )
        record = metrics_dict(m_a)
        if last_a_state is not None:
            record["state_overlap_prev_fc"] = final_state_overlap(last_a_state, state_a, layer_idx=2)
            record["state_overlap_prev_conv"] = final_state_overlap(last_a_state, state_a, layer_idx=1)
        else:
            record["state_overlap_prev_fc"] = None
            record["state_overlap_prev_conv"] = None
        record["update_step"] = step
        phase_a.append(record)
        recent_a.append(m_a)
        last_a_state = state_a

        if memorized(recent_a, consecutive_k):
            steps_to_memorize_a = step
            break

        if step < max_updates_a:
            rng, upd_rng = jax.random.split(rng)
            orch, opt_state = apply_update(orch, state_a, upd_rng, opt, opt_state)

    if steps_to_memorize_a is None:
        steps_to_memorize_a = max_updates_a + 1

    # Snapshot A just before B training, for before/after comparison
    m_a_before_b, state_a_before_b, _, rng = evaluate_sample(
        orch,
        state_template,
        x_a,
        y_a,
        rng,
        n_warmup=n_warmup,
        n_clamped=n_clamped,
        n_free=n_free,
    )

    # Phase B: evaluate A and B at every step, update on B until memorized
    recent_b: list[EvalMetrics] = []
    for step in range(max_updates_b + 1):
        m_a, state_a, _, rng = evaluate_sample(
            orch,
            state_template,
            x_a,
            y_a,
            rng,
            n_warmup=n_warmup,
            n_clamped=n_clamped,
            n_free=n_free,
        )
        rec_a = metrics_dict(m_a)
        rec_a["update_step"] = step
        rec_a["state_overlap_to_preB_fc"] = final_state_overlap(state_a_before_b, state_a, layer_idx=2)
        rec_a["state_overlap_to_preB_conv"] = final_state_overlap(state_a_before_b, state_a, layer_idx=1)
        phase_b_a.append(rec_a)

        m_b, state_b, _, rng = evaluate_sample(
            orch,
            state_template,
            x_b,
            y_b,
            rng,
            n_warmup=n_warmup,
            n_clamped=n_clamped,
            n_free=n_free,
        )
        rec_b = metrics_dict(m_b)
        rec_b["update_step"] = step
        phase_b_b.append(rec_b)
        recent_b.append(m_b)

        if memorized(recent_b, consecutive_k):
            steps_to_memorize_b = step
            break

        if step < max_updates_b:
            rng, upd_rng = jax.random.split(rng)
            orch, opt_state = apply_update(orch, state_b, upd_rng, opt, opt_state)

    if steps_to_memorize_b is None:
        steps_to_memorize_b = max_updates_b + 1

    both_correct = sum(
        int(a["accuracy"] >= 1.0 and b["accuracy"] >= 1.0)
        for a, b in zip(phase_b_a, phase_b_b)
    )
    both_margin_positive = sum(
        int(a["soft_margin"] > 0.0 and b["soft_margin"] > 0.0)
        for a, b in zip(phase_b_a, phase_b_b)
    )

    return {
        "seed": seed,
        "steps_to_memorize_a": steps_to_memorize_a,
        "steps_to_memorize_b": steps_to_memorize_b,
        "n_steps_both_correct_during_B": both_correct,
        "n_steps_both_margin_positive_during_B": both_margin_positive,
        "phase_a": phase_a,
        "phase_b_eval_a": phase_b_a,
        "phase_b_eval_b": phase_b_b,
        "summary_before_B": metrics_dict(m_a_before_b),
    }


# -----------------------------------------------------------------------------
# Aggregation / CLI
# -----------------------------------------------------------------------------

def pad_and_mean(list_of_lists: list[list[dict[str, Any]]], key: str) -> list[float]:
    max_len = max(len(xs) for xs in list_of_lists)
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, xs in enumerate(list_of_lists):
        for j, item in enumerate(xs):
            val = item.get(key)
            if val is None:
                continue
            arr[i, j] = float(val)
    return np.nanmean(arr, axis=0).tolist()


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "steps_to_memorize_a_mean": float(np.mean([r["steps_to_memorize_a"] for r in results])),
        "steps_to_memorize_a_std": float(np.std([r["steps_to_memorize_a"] for r in results])),
        "steps_to_memorize_b_mean": float(np.mean([r["steps_to_memorize_b"] for r in results])),
        "steps_to_memorize_b_std": float(np.std([r["steps_to_memorize_b"] for r in results])),
        "n_steps_both_correct_during_B_mean": float(np.mean([r["n_steps_both_correct_during_B"] for r in results])),
        "n_steps_both_correct_during_B_std": float(np.std([r["n_steps_both_correct_during_B"] for r in results])),
        "n_steps_both_margin_positive_during_B_mean": float(np.mean([r["n_steps_both_margin_positive_during_B"] for r in results])),
        "n_steps_both_margin_positive_during_B_std": float(np.std([r["n_steps_both_margin_positive_during_B"] for r in results])),
        "phase_a_soft_margin_mean_traj": pad_and_mean([r["phase_a"] for r in results], "soft_margin"),
        "phase_a_flip_last_mean_traj": pad_and_mean([r["phase_a"] for r in results], "flip_fraction_last"),
        "phase_b_a_soft_margin_mean_traj": pad_and_mean([r["phase_b_eval_a"] for r in results], "soft_margin"),
        "phase_b_b_soft_margin_mean_traj": pad_and_mean([r["phase_b_eval_b"] for r in results], "soft_margin"),
        "phase_b_a_clamp_free_fc_mean_traj": pad_and_mean([r["phase_b_eval_a"] for r in results], "clamp_free_distance_fc"),
        "phase_b_b_clamp_free_fc_mean_traj": pad_and_mean([r["phase_b_eval_b"] for r in results], "clamp_free_distance_fc"),
        "phase_a_hidden_margin_fc_mean_traj":   pad_and_mean([r["phase_a"]       for r in results], "hidden_margin_fc"),
        "phase_a_hidden_margin_conv_mean_traj": pad_and_mean([r["phase_a"]       for r in results], "hidden_margin_conv"),
        "phase_b_a_hidden_margin_fc_mean_traj": pad_and_mean([r["phase_b_eval_a"] for r in results], "hidden_margin_fc"),
        "phase_b_b_hidden_margin_fc_mean_traj": pad_and_mean([r["phase_b_eval_b"] for r in results], "hidden_margin_fc"),
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential forgetting experiment for the conv Darnax model.")
    p.add_argument("--threshold", type=float, required=True, help="Common threshold for conv/fc/out layers.")
    p.add_argument("--learning-rate", type=float, required=True, help="Learning rate used for all trainable modules.")
    p.add_argument("--momentum", type=float, default=0.5, help="Field momentum. Default: 0.5")
    p.add_argument("--mask", type=float, default=0.2, help="Mask probability. Default: 0.2")
    p.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds. Default: 3")
    p.add_argument("--idx-a", type=int, default=0, help="Index of image A inside the loaded CIFAR batch.")
    p.add_argument("--idx-b", type=int, default=1, help="Index of image B inside the loaded CIFAR batch.")
    p.add_argument("--pair-file", type=str, default=None, help="Optional .npz file with x_a,y_a,x_b,y_b arrays.")
    p.add_argument("--dataset-batch-size", type=int, default=32, help="How many CIFAR samples to load when using indices.")
    p.add_argument("--max-updates-a", type=int, default=100, help="Maximum A updates before giving up.")
    p.add_argument("--max-updates-b", type=int, default=100, help="Maximum B updates before giving up.")
    p.add_argument("--memorization-k", type=int, default=3, help="Consecutive successful evals required for memorization.")
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-clamped", type=int, default=5)
    p.add_argument("--n-free", type=int, default=5)
    p.add_argument("--output", type=str, default="forgetting_results.json", help="Output JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_a, y_a, x_b, y_b = load_samples(args.idx_a, args.idx_b, args.dataset_batch_size, args.pair_file)

    results = []
    for seed in range(args.n_seeds):
        result = run_single_seed_forgetting(
            x_a=x_a,
            y_a=y_a,
            x_b=x_b,
            y_b=y_b,
            threshold=args.threshold,
            learning_rate=args.learning_rate,
            seed=seed,
            momentum=args.momentum,
            mask_prob=args.mask,
            max_updates_a=args.max_updates_a,
            max_updates_b=args.max_updates_b,
            consecutive_k=args.memorization_k,
            n_warmup=args.n_warmup,
            n_clamped=args.n_clamped,
            n_free=args.n_free,
        )
        results.append(result)
        print(
            f"seed={seed}  steps_A={result['steps_to_memorize_a']}  "
            f"steps_B={result['steps_to_memorize_b']}  both_correct_B={result['n_steps_both_correct_during_B']}"
        )

    payload = {
        "config": vars(args),
        "image_a_class": int(np.argmax(np.array(y_a)[0])),
        "image_b_class": int(np.argmax(np.array(y_b)[0])),
        "summary": build_summary(results),
        "per_seed": results,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
