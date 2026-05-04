#!/usr/bin/env python3
"""Stability under interference: two-pattern sequential training.

Phase A (weight updates 0..n_updates_a-1): train on pattern A only.
Phase B (weight updates n_updates_a..n_updates_a+n_updates_b-1): train on pattern B only.

At every weight update we measure:
  - flip_rate_C: fraction of neurons still flipping at end of free phase (fixed-point proximity)
  - state_drift_fc: L2 distance of state C from previous update (learning convergence)
  - weight_delta_norm: |ΔW| total
  - soft_margin_A, soft_margin_B: how well both patterns are still correct
  - accuracy_A, accuracy_B: binary correctness

Key question: does training on B destabilise the fixed point found for A?
Does long_warmup remain more stable during interference than current?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gap_experiment import (
    Rule, build_conv_model, build_optimizer, apply_grads,
    run_warmup, run_free,
    load_single_image,
    compute_grads_rule1, compute_grads_rule2, compute_grads_rule3,
    extract_states_rule1, extract_states_rule2, extract_states_rule4,
    soft_margin, accuracy,
)
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState


def neuron_flip_rate(s_prev, s_next, layer_idx=2):
    a = np.array(s_prev.states[layer_idx]).ravel()
    b = np.array(s_next.states[layer_idx]).ravel()
    return float(np.mean(a != b))


def state_l2_change(s_prev, s_next, layer_idx=2):
    a = np.array(s_prev.states[layer_idx]).ravel().astype(float)
    b = np.array(s_next.states[layer_idx]).ravel().astype(float)
    return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


def weight_delta_norm(orch_before, orch_after):
    pb = eqx.filter(orch_before, eqx.is_inexact_array)
    pa = eqx.filter(orch_after,  eqx.is_inexact_array)
    delta = jtu.tree_map(lambda a, b: np.array(b - a), pb, pa)
    total = 0.0
    for (i, j) in [(1, 0), (1, 1), (2, 2), (3, 2)]:
        leaves = jtu.tree_leaves(delta.lmap[i][j])
        total += sum(np.linalg.norm(l.ravel()) ** 2 for l in leaves)
    return float(total ** 0.5)


def free_phase_flip_rate(orch, state_template, x, y, rng, n_warmup, n_free_steps=20, layer_idx=2):
    s = state_template.init(x, y)
    s, rng = run_warmup(orch, s, rng, n_warmup)
    flip_rates = []
    s_prev = s
    for _ in range(n_free_steps):
        s_next, rng = orch.step(s_prev, rng=rng, filter_messages="inference")
        flip_rates.append(neuron_flip_rate(s_prev, s_next, layer_idx))
        s_prev = s_next
    return flip_rates


def do_weight_update(rule, orch, state_template, x, y, step_rng,
                     n_warmup, n_clamped, n_free, n_warmup_long, ep_alpha, optimizer, opt_state):
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
            orch, state_template, x, y, step_rng, n_warmup_long, n_clamped, n_free, n_warmup)
        grads, step_rng = compute_grads_rule1(orch, states, step_rng)
    orch_new, opt_state_new = apply_grads(orch, grads, optimizer, opt_state)
    return orch_new, opt_state_new, states, step_rng


def measure_pattern(orch, state_template, x, y, rng, n_warmup, n_free):
    """Soft margin and accuracy for a pattern, using inference state D."""
    s = state_template.init(x, y)
    s, rng = run_warmup(orch, s, rng, n_warmup)
    s, rng = run_free(orch, s, rng, n_free)
    sm = soft_margin(orch, s, y)
    acc = accuracy(orch, s, y)
    return sm, acc


def run_interference_seed(
    rule: Rule, xa, ya, xb, yb, seed: int,
    threshold: float, learning_rate: float,
    n_warmup: int, n_clamped: int, n_free: int,
    n_warmup_long: int, n_updates_a: int, n_updates_b: int,
    ep_alpha: float = 0.3, n_free_conv_curve: int = 20,
) -> dict[str, Any]:
    state_template, orch = build_conv_model(
        seed=seed, threshold_conv=threshold, threshold_fc=threshold, threshold_out=threshold,
    )
    optimizer, opt_state = build_optimizer(orch, lr=learning_rate)
    rng = jax.random.PRNGKey(seed)
    trajectory = []
    prev_C = None

    total_updates = n_updates_a + n_updates_b
    for k in range(total_updates):
        rng, step_rng = jax.random.split(rng)
        phase = "A" if k < n_updates_a else "B"
        x_cur, y_cur = (xa, ya) if phase == "A" else (xb, yb)

        orch_before = orch
        orch, opt_state, states, step_rng = do_weight_update(
            rule, orch, state_template, x_cur, y_cur, step_rng,
            n_warmup, n_clamped, n_free, n_warmup_long, ep_alpha, optimizer, opt_state,
        )

        # Flip rate within free phase (using current training pattern)
        rng, curve_rng = jax.random.split(rng)
        n_wu = n_warmup_long if rule == Rule.LONG_WARMUP else n_warmup
        flip_curve = free_phase_flip_rate(
            orch, state_template, x_cur, y_cur, curve_rng, n_wu, n_free_conv_curve)

        # State drift
        drift_fc = state_l2_change(prev_C, states.C) if prev_C is not None else None
        prev_C = states.C

        # Weight delta
        w_norm = weight_delta_norm(orch_before, orch)

        # Both pattern margins (using inference state)
        rng, rA, rB = jax.random.split(rng, 3)
        sm_a, acc_a = measure_pattern(orch, state_template, xa, ya, rA, n_wu, n_free)
        sm_b, acc_b = measure_pattern(orch, state_template, xb, yb, rB, n_wu, n_free)

        rec = {
            "update": k,
            "phase": phase,
            "flip_rate_final": flip_curve[-1],
            "flip_rate_mean":  float(np.mean(flip_curve)),
            "state_drift_fc":  drift_fc,
            "weight_delta_norm": w_norm,
            "soft_margin_A": sm_a,
            "soft_margin_B": sm_b,
            "accuracy_A":    acc_a,
            "accuracy_B":    acc_b,
        }
        trajectory.append(rec)

        if k % 10 == 0:
            print(f"  [{rule.value}] k={k:3d} ({phase})  "
                  f"flip={flip_curve[-1]:.3f}  |ΔW|={w_norm:.2f}  "
                  f"margin_A={sm_a:.2f}  margin_B={sm_b:.2f}")

    return {"rule": rule.value, "seed": seed, "ep_alpha": ep_alpha, "trajectory": trajectory}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rules", nargs="+", choices=[r.value for r in Rule], default=[r.value for r in Rule])
    p.add_argument("--threshold",     type=float, default=1.7)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-warmup",      type=int,   default=1)
    p.add_argument("--n-warmup-long", type=int,   default=10)
    p.add_argument("--n-clamped",     type=int,   default=5)
    p.add_argument("--n-free",        type=int,   default=5)
    p.add_argument("--n-free-curve",  type=int,   default=20)
    p.add_argument("--ep-alpha",      type=float, default=0.3)
    p.add_argument("--n-updates-a",   type=int,   default=40)
    p.add_argument("--n-updates-b",   type=int,   default=40)
    p.add_argument("--idx-a",         type=int,   default=0)
    p.add_argument("--idx-b",         type=int,   default=1)
    p.add_argument("--n-seeds",       type=int,   default=3)
    p.add_argument("--output",        type=str,   default="results/interference_stability/interference_stability_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    xa, ya = load_single_image(args.idx_a)
    xb, yb = load_single_image(args.idx_b)
    print(f"Pattern A idx={args.idx_a}, Pattern B idx={args.idx_b}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = []

    for rule_str in args.rules:
        rule = Rule(rule_str)
        for seed in range(args.n_seeds):
            print(f"\n--- Rule: {rule.value}  Seed: {seed} ---")
            r = run_interference_seed(
                rule=rule, xa=xa, ya=ya, xb=xb, yb=yb, seed=seed,
                threshold=args.threshold, learning_rate=args.learning_rate,
                n_warmup=args.n_warmup, n_clamped=args.n_clamped, n_free=args.n_free,
                n_warmup_long=args.n_warmup_long,
                n_updates_a=args.n_updates_a, n_updates_b=args.n_updates_b,
                ep_alpha=args.ep_alpha, n_free_conv_curve=args.n_free_curve,
            )
            all_results.append(r)

    out_path.write_text(json.dumps({"config": vars(args), "results": all_results}, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
