#!/usr/bin/env python3
"""Train/inference gap experiment.

Tracks the geometry of states A, B, C, D over K weight updates for four learning rules.
The goal is to measure whether training and inference visit compatible regions of state space.

State definitions:
  A = state after warmup (forward-only messages, same for all rules)
  B = state after the second phase
  C = state after the third phase — the state that gets stabilized by the update
  D = inference state: short warmup + free dynamics, no clamping ever

  Rule 1 (current):       warmup → clamped → free;   stabilize C (free)
  Rule 2 (approximate):   warmup → free → clamped;   stabilize C (clamped)
  Rule 3 (ep):            warmup → free → clamped;   stabilize C - de-stabilize B
                          (mimics Equilibrium Propagation: backward(C) - backward(B))
  Rule 4 (long_warmup):   long warmup until convergence → clamped → free; stabilize C

Key insight: in rules 1/4, D and B both start from A and run free dynamics, so measuring
C↔D directly tells us how far the stabilized free state is from the inference endpoint.
In rule 2, B is produced by free dynamics from A, making B ≡ D by construction — so C↔D
tells us how far clamping takes us from the inference endpoint.

Overlap metric: mean(s_X · s_Y) over neurons, computed at the FC hidden layer (index 2).
For ±1 binary vectors this equals the cosine similarity and is the standard spin-glass overlap.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.pooling import GlobalMajorityPooling, GlobalUnpooling
from darnax.modules.fully_connected import FullyConnected, FrozenRescaledFullyConnected
from darnax.modules.input_output import OutputLayer
from darnax.modules.recurrent import RecurrentDiscrete
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState


# ---------------------------------------------------------------------------
# Rule enum
# ---------------------------------------------------------------------------

class Rule(str, Enum):
    CURRENT = "current"
    APPROXIMATE = "approximate"
    EP = "ep"
    LONG_WARMUP = "long_warmup"


# ---------------------------------------------------------------------------
# Model construction (same architecture as experiments2)
# ---------------------------------------------------------------------------

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
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(
                in_channels=in_channels, out_channels=n_channels,
                kernel_size=input_kernel_size, threshold=threshold_conv,
                strength=strength_in, key=keys[0], padding_mode="constant",
            ),
            1: Conv2DRecurrentDiscrete(
                channels=n_channels, kernel_size=recur_kernel_size,
                groups=n_channels, j_d=j_d_conv, threshold=threshold_conv,
                padding_mode="constant", key=keys[1],
                lr=lr_conv, weight_decay=weight_decay_conv,
            ),
            2: GlobalUnpooling(strength=strength_unpool, axis=(1, 2)),
        },
        2: {
            1: GlobalMajorityPooling(strength=strength_pool, axis=(1, 2)),
            2: RecurrentDiscrete(features=n_channels, j_d=j_d_fc, threshold=threshold_fc, key=keys[2]),
            3: FrozenRescaledFullyConnected(
                in_features=num_labels, out_features=n_channels,
                strength=strength_wback, threshold=threshold_back, key=keys[3],
            ),
        },
        3: {
            2: FullyConnected(
                in_features=n_channels, out_features=num_labels,
                strength=strength_wout, threshold=threshold_out, key=keys[4],
            ),
            3: OutputLayer(),
        },
    })
    state = SequentialState([
        (s, s, in_channels),
        (s, s, n_channels),
        (n_channels,),
        (num_labels,),
    ])
    orchestrator = SequentialOrchestrator(
        layers=layer_map, field_momentum=field_momentum, mask_prob=mask_prob,
    )
    return state, orchestrator


def build_optimizer(
    orchestrator: SequentialOrchestrator,
    lr: float,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)
    labels = jtu.tree_map(lambda _: "frozen", params, is_leaf=eqx.is_array)

    def label_module(m, lbl):
        return jtu.tree_map(lambda _: lbl, m, is_leaf=eqx.is_array)

    for (i, j), lbl in [((1, 0), "w_in"), ((1, 1), "j_conv"), ((2, 2), "j_fc"), ((3, 2), "w_out")]:
        labels = eqx.tree_at(
            lambda m, _i=i, _j=j: m.lmap[_i][_j], labels,
            replace=label_module(params.lmap[i][j], lbl),
        )

    optimizer = optax.multi_transform(
        {"frozen": optax.sgd(0.0), "w_in": optax.sgd(lr), "j_conv": optax.sgd(lr),
         "j_fc": optax.sgd(lr), "w_out": optax.sgd(lr)},
        labels,
    )
    opt_state = optimizer.init(eqx.filter(orchestrator, eqx.is_inexact_array))
    return optimizer, opt_state


# ---------------------------------------------------------------------------
# Phase primitives
# ---------------------------------------------------------------------------

def run_warmup(
    orch: SequentialOrchestrator,
    state: SequentialState,
    rng,
    n_steps: int,
) -> tuple[SequentialState, Any]:
    for _ in range(n_steps):
        state, rng = orch.step(state, rng=rng, filter_messages="inference")
    return state, rng


def run_clamped(
    orch: SequentialOrchestrator,
    state: SequentialState,
    rng,
    n_steps: int,
) -> tuple[SequentialState, Any]:
    for _ in range(n_steps):
        state, rng = orch.step(state, rng=rng, filter_messages="all")
    return state, rng


def run_free(
    orch: SequentialOrchestrator,
    state: SequentialState,
    rng,
    n_steps: int,
) -> tuple[SequentialState, Any]:
    for _ in range(n_steps):
        state, rng = orch.step(state, rng=rng, filter_messages="inference")
    return state, rng


# ---------------------------------------------------------------------------
# Per-rule: extract A, B, C, D and compute local gradients
# ---------------------------------------------------------------------------

@dataclass
class PhaseStates:
    A: SequentialState
    B: SequentialState
    C: SequentialState
    D: SequentialState


def _get_inference_state(
    orch: SequentialOrchestrator,
    state_template: SequentialState,
    x, y, rng,
    n_warmup: int,
    n_free: int,
) -> tuple[SequentialState, Any]:
    """State D: warmup + free, no clamping."""
    s0 = state_template.init(x, y)
    s0, rng = run_warmup(orch, s0, rng, n_warmup)
    s0, rng = run_free(orch, s0, rng, n_free)
    return s0, rng


def extract_states_rule1(
    orch: SequentialOrchestrator,
    state_template: SequentialState,
    x, y, rng,
    n_warmup: int, n_clamped: int, n_free: int,
) -> tuple[PhaseStates, Any]:
    """warmup → A, clamped → B, free → C; D = warmup + free."""
    s = state_template.init(x, y)
    s, rng = run_warmup(orch, s, rng, n_warmup);  A = s
    s, rng = run_clamped(orch, s, rng, n_clamped); B = s
    s, rng = run_free(orch, s, rng, n_free);       C = s
    D, rng = _get_inference_state(orch, state_template, x, y, rng, n_warmup, n_free)
    return PhaseStates(A, B, C, D), rng


def extract_states_rule2(
    orch: SequentialOrchestrator,
    state_template: SequentialState,
    x, y, rng,
    n_warmup: int, n_clamped: int, n_free: int,
) -> tuple[PhaseStates, Any]:
    """warmup → A, free → B, clamped → C; D = warmup + free (= B by construction)."""
    s = state_template.init(x, y)
    s, rng = run_warmup(orch, s, rng, n_warmup);  A = s
    s, rng = run_free(orch, s, rng, n_free);       B = s
    s, rng = run_clamped(orch, s, rng, n_clamped); C = s
    # D is obtained independently (same warmup seed would give same A→free = B,
    # but we re-derive it cleanly for consistent measurement)
    D, rng = _get_inference_state(orch, state_template, x, y, rng, n_warmup, n_free)
    return PhaseStates(A, B, C, D), rng


def extract_states_rule4(
    orch: SequentialOrchestrator,
    state_template: SequentialState,
    x, y, rng,
    n_warmup_long: int, n_clamped: int, n_free: int, n_warmup_short: int,
) -> tuple[PhaseStates, Any]:
    """Long warmup → A, clamped → B, free → C; D uses short warmup + free."""
    s = state_template.init(x, y)
    s, rng = run_warmup(orch, s, rng, n_warmup_long); A = s
    s, rng = run_clamped(orch, s, rng, n_clamped);    B = s
    s, rng = run_free(orch, s, rng, n_free);           C = s
    D, rng = _get_inference_state(orch, state_template, x, y, rng, n_warmup_short, n_free)
    return PhaseStates(A, B, C, D), rng


def compute_grads_rule1(orch, states: PhaseStates, rng):
    """Stabilize C (free state)."""
    grads = orch.backward(states.C, rng=rng)
    return eqx.filter(grads, eqx.is_inexact_array), rng


def compute_grads_rule2(orch, states: PhaseStates, rng):
    """Stabilize C (clamped state)."""
    grads = orch.backward(states.C, rng=rng)
    return eqx.filter(grads, eqx.is_inexact_array), rng


def compute_rads_rule3(orch, states: PhaseStates, rng, ep_alpha: float = 1.0):
    """EP-like: stabilize C (clamped) and de-stabilize B (free).
    Combined update = backward(C) - ep_alpha * backward(B).
    ep_alpha=1.0 is the full contrastive case; smaller values soften de-stabilization.
    """
    rng, rng_c, rng_b = jax.random.split(rng, 3)
    grads_c = eqx.filter(orch.backward(states.C, rng=rng_c), eqx.is_inexact_array)
    grads_b = eqx.filter(orch.backward(states.B, rng=rng_b), eqx.is_inexact_array)
    grads_ep = jtu.tree_map(lambda c, b: c - ep_alpha * b, grads_c, grads_b)
    return grads_ep, rng


def apply_grads(
    orch: SequentialOrchestrator,
    grads_filtered,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[SequentialOrchestrator, optax.OptState]:
    params_filtered = eqx.filter(orch, eqx.is_inexact_array)
    updates, new_opt_state = optimizer.update(grads_filtered, opt_state, params=params_filtered)
    return eqx.apply_updates(orch, updates), new_opt_state


# ---------------------------------------------------------------------------
# Overlap and soft-margin metrics
# ---------------------------------------------------------------------------

def state_overlap(sa: SequentialState, sb: SequentialState, layer_idx: int = 2) -> float:
    """Overlap m = mean(s_A * s_B) for ±1 binary vectors. Equals cosine similarity."""
    a = np.array(sa.states[layer_idx]).ravel()
    b = np.array(sb.states[layer_idx]).ravel()
    return float(np.mean(a * b))


def compute_quadrilateral(states: PhaseStates, layer_idx: int = 2) -> dict[str, float]:
    return {
        "AB": state_overlap(states.A, states.B, layer_idx),
        "BC": state_overlap(states.B, states.C, layer_idx),
        "CD": state_overlap(states.C, states.D, layer_idx),
        "AD": state_overlap(states.A, states.D, layer_idx),
        "AC": state_overlap(states.A, states.C, layer_idx),
        "BD": state_overlap(states.B, states.D, layer_idx),
    }


def soft_margin(orch: SequentialOrchestrator, state: SequentialState, y) -> float:
    pred_state, _ = orch.predict(state, jax.random.PRNGKey(0))
    logits = np.array(pred_state.states[-1])
    y_np = np.array(y)
    true_cls = int(np.argmax(y_np))
    correct = logits[0, true_cls]
    wrong = np.concatenate([logits[0, :true_cls], logits[0, true_cls + 1:]])
    return float(correct - wrong.max())


def accuracy(orch: SequentialOrchestrator, state: SequentialState, y) -> float:
    pred_state, _ = orch.predict(state, jax.random.PRNGKey(0))
    logits = np.array(pred_state.states[-1])
    y_np = np.array(y)
    return float(np.argmax(logits[0]) == np.argmax(y_np[0]))


# ---------------------------------------------------------------------------
# Single-rule experiment loop
# ---------------------------------------------------------------------------

def run_rule_experiment(
    rule: Rule,
    x, y,
    seed: int,
    threshold: float,
    learning_rate: float,
    momentum: float,
    mask_prob: float,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
    n_updates: int,
    ep_alpha: float = 1.0,
    layer_idx: int = 2,
) -> dict[str, Any]:
    state_template, orch = build_conv_model(
        seed=seed, threshold_conv=threshold, threshold_fc=threshold,
        threshold_out=threshold, mask_prob=mask_prob, field_momentum=momentum,
    )
    optimizer, opt_state = build_optimizer(orch, lr=learning_rate)
    rng = jax.random.PRNGKey(seed)

    trajectory = []

    for k in range(n_updates):
        rng, step_rng = jax.random.split(rng)

        if rule == Rule.CURRENT:
            states, step_rng = extract_states_rule1(
                orch, state_template, x, y, step_rng, n_warmup, n_clamped, n_free)
            grads, step_rng = compute_grads_rule1(orch, states, step_rng)

        elif rule == Rule.APPROXIMATE:
            states, step_rng = extract_states_rule2(
                orch, state_template, x, y, step_rng, n_warmup, n_clamped, n_free)
            grads, step_rng = compute_grads_rule2(orch, states, step_rng)

        elif rule == Rule.EP:
            states, step_rng = extract_states_rule2(
                orch, state_template, x, y, step_rng, n_warmup, n_clamped, n_free)
            grads, step_rng = compute_grads_rule3(orch, states, step_rng, ep_alpha=ep_alpha)

        elif rule == Rule.LONG_WARMUP:
            states, step_rng = extract_states_rule4(
                orch, state_template, x, y, step_rng,
                n_warmup_long, n_clamped, n_free, n_warmup)
            grads, step_rng = compute_grads_rule1(orch, states, step_rng)

        overlaps_fc   = compute_quadrilateral(states, layer_idx=2)
        overlaps_conv = compute_quadrilateral(states, layer_idx=1)
        sm = soft_margin(orch, states.C, y)
        acc = accuracy(orch, states.D, y)

        trajectory.append({
            "update": k,
            "overlaps_fc": overlaps_fc,
            "overlaps_conv": overlaps_conv,
            "soft_margin_C": sm,
            "accuracy_D": acc,
        })

        orch, opt_state = apply_grads(orch, grads, optimizer, opt_state)

        if k % 10 == 0:
            print(f"  [{rule.value}] step {k:3d}  CD_fc={overlaps_fc['CD']:.3f}  "
                  f"margin_C={sm:.3f}  acc_D={acc:.1f}")

    return {"rule": rule.value, "seed": seed, "ep_alpha": ep_alpha, "trajectory": trajectory}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_single_image(image_idx: int = 0, dataset_batch_size: int = 32):
    data = Cifar10(batch_size=dataset_batch_size, linear_projection=None,
                   label_mode="pm1", x_transform="identity")
    data.build(key=jax.random.PRNGKey(0))
    x, y = next(iter(data))
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 32, 32, 3)
    return x[image_idx:image_idx+1], y[image_idx:image_idx+1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/inference gap experiment.")
    p.add_argument("--rules", nargs="+",
                   choices=[r.value for r in Rule], default=[r.value for r in Rule],
                   help="Which rules to run. Default: all four.")
    p.add_argument("--threshold",     type=float, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--momentum",      type=float, default=0.0)
    p.add_argument("--mask",          type=float, default=0.0)
    p.add_argument("--n-warmup",      type=int,   default=1)
    p.add_argument("--n-warmup-long", type=int,   default=20,
                   help="Warmup steps for rule 4 (long_warmup). Default: 20.")
    p.add_argument("--n-clamped",     type=int,   default=5)
    p.add_argument("--n-free",        type=int,   default=5)
    p.add_argument("--ep-alpha",       type=float, default=1.0,
                   help="De-stabilization strength for EP rule: update = backward(C) - alpha*backward(B). Default: 1.0")
    p.add_argument("--n-updates",     type=int,   default=50,
                   help="Number of weight updates (K). Default: 50.")
    p.add_argument("--n-seeds",       type=int,   default=3)
    p.add_argument("--image-idx",     type=int,   default=0,
                   help="Which CIFAR10 image to use.")
    p.add_argument("--output",        type=str,   default="gap_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x, y = load_single_image(args.image_idx)
    print(f"Image class: {int(np.argmax(np.array(y)[0]))}")

    all_results = []
    for rule_str in args.rules:
        rule = Rule(rule_str)
        for seed in range(args.n_seeds):
            print(f"\n--- Rule: {rule.value}  Seed: {seed} ---")
            result = run_rule_experiment(
                rule=rule, x=x, y=y, seed=seed,
                threshold=args.threshold,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                mask_prob=args.mask,
                n_warmup=args.n_warmup,
                n_clamped=args.n_clamped,
                n_free=args.n_free,
                n_warmup_long=args.n_warmup_long,
                n_updates=args.n_updates,
                ep_alpha=args.ep_alpha,
            )
            all_results.append(result)

    payload = {"config": vars(args), "results": all_results}
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
