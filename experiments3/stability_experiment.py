#!/usr/bin/env python3
"""Stability analysis experiment.

Measures three distinct notions of stability over K weight updates:

1. NEURON FLIP RATE — within a single dynamic run (e.g. the free phase), what fraction
   of neurons change their sign between consecutive dynamic steps?
   A network approaching a fixed point will have flip_rate → 0.

2. STATE CONVERGENCE — how much does the final free-phase state change between consecutive
   weight updates? If states[k] ≈ states[k+1] the weight landscape has converged.

3. WEIGHT UPDATE NORM — the L2 norm of the weight delta applied at each step.
   Tracks whether learning slows down / converges.

All three metrics are measured for each of the four rules (current, approximate, long_warmup, EP)
across K weight update steps and averaged over multiple seeds.

Additionally, we measure the within-free-phase flip rate at every dynamic step to produce
a "convergence curve" (how fast states stabilise within a single free run).
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
    Rule,
    build_conv_model,
    build_optimizer,
    apply_grads,
    run_warmup,
    run_clamped,
    run_free,
    load_single_image,
    compute_grads_rule1,
    compute_grads_rule2,
    compute_grads_rule3,
    extract_states_rule1,
    extract_states_rule2,
    extract_states_rule4,
)
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def neuron_flip_rate(s_prev: SequentialState, s_next: SequentialState, layer_idx: int = 2) -> float:
    """Fraction of neurons that flipped sign between two consecutive dynamic steps."""
    a = np.array(s_prev.states[layer_idx]).ravel()
    b = np.array(s_next.states[layer_idx]).ravel()
    return float(np.mean(a != b))


def state_change(s_prev: SequentialState, s_next: SequentialState, layer_idx: int = 2) -> float:
    """L2 distance between two states at a given layer (normalised by sqrt(N))."""
    a = np.array(s_prev.states[layer_idx]).ravel().astype(float)
    b = np.array(s_next.states[layer_idx]).ravel().astype(float)
    return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


def weight_update_norm(orch_before: SequentialOrchestrator, orch_after: SequentialOrchestrator) -> dict[str, float]:
    """L2 norm of ΔW for each trainable weight matrix."""
    params_before = eqx.filter(orch_before, eqx.is_inexact_array)
    params_after  = eqx.filter(orch_after,  eqx.is_inexact_array)
    delta = jtu.tree_map(lambda a, b: np.array(b - a), params_before, params_after)

    norms: dict[str, float] = {}
    # access by (row, col) keys in lmap
    for (i, j), name in [((1, 0), "w_in"), ((1, 1), "j_conv"), ((2, 2), "j_fc"), ((3, 2), "w_out")]:
        mod_delta = delta.lmap[i][j]
        leaves = jtu.tree_leaves(mod_delta)
        total = float(sum(np.linalg.norm(l.ravel()) ** 2 for l in leaves if l is not None) ** 0.5)
        norms[name] = total
    norms["total"] = float(sum(v ** 2 for v in norms.values()) ** 0.5)
    return norms


def free_phase_convergence_curve(
    orch: SequentialOrchestrator,
    state_template: SequentialState,
    x, y, rng,
    n_warmup: int,
    n_free_steps: int,
    layer_idx: int = 2,
) -> list[float]:
    """Run warmup + free phase step-by-step; record flip_rate at each free step."""
    s = state_template.init(x, y)
    s, rng = run_warmup(orch, s, rng, n_warmup)
    flip_rates = []
    s_prev = s
    for _ in range(n_free_steps):
        s_next, rng = orch.step(s_prev, rng=rng, filter_messages="inference")
        flip_rates.append(neuron_flip_rate(s_prev, s_next, layer_idx))
        s_prev = s_next
    return flip_rates


# ---------------------------------------------------------------------------
# Per-update-step stability record
# ---------------------------------------------------------------------------

def run_stability_seed(
    rule: Rule,
    x, y,
    seed: int,
    threshold: float,
    learning_rate: float,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
    n_updates: int,
    ep_alpha: float = 0.3,
    layer_idx: int = 2,
    n_free_conv_curve: int = 20,  # how many free steps for the convergence curve
) -> dict[str, Any]:
    state_template, orch = build_conv_model(
        seed=seed, threshold_conv=threshold, threshold_fc=threshold, threshold_out=threshold,
    )
    optimizer, opt_state = build_optimizer(orch, lr=learning_rate)
    rng = jax.random.PRNGKey(seed)

    trajectory = []
    prev_C: SequentialState | None = None

    for k in range(n_updates):
        rng, step_rng = jax.random.split(rng)
        orch_before = orch

        # --- Extract states and compute grads ---
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

        # --- Apply update ---
        orch, opt_state = apply_grads(orch, grads, optimizer, opt_state)

        # --- Flip rate WITHIN free phase at state C ---
        # Re-derive state C step-by-step to get per-dynamic-step flip rates
        rng, curve_rng = jax.random.split(rng)
        n_warmup_for_curve = n_warmup_long if rule == Rule.LONG_WARMUP else n_warmup
        flip_curve = free_phase_convergence_curve(
            orch, state_template, x, y, curve_rng,
            n_warmup=n_warmup_for_curve, n_free_steps=n_free_conv_curve, layer_idx=layer_idx,
        )

        # --- State C change across weight updates ---
        state_drift_fc  = state_change(prev_C, states.C, layer_idx=2) if prev_C is not None else None
        state_drift_conv = state_change(prev_C, states.C, layer_idx=1) if prev_C is not None else None
        prev_C = states.C

        # --- Weight update norms ---
        w_norms = weight_update_norm(orch_before, orch)

        rec = {
            "update": k,
            "flip_curve": flip_curve,          # list[float] length n_free_conv_curve
            "flip_rate_final": flip_curve[-1],  # flip rate at last free step
            "flip_rate_mean":  float(np.mean(flip_curve)),
            "state_drift_fc":  state_drift_fc,
            "state_drift_conv": state_drift_conv,
            "weight_norms":    w_norms,
        }
        trajectory.append(rec)

        if k % 10 == 0:
            print(f"  [{rule.value}] step {k:3d}  "
                  f"flip_final={flip_curve[-1]:.3f}  "
                  f"drift={'N/A' if state_drift_fc is None else f'{state_drift_fc:.3f}'}  "
                  f"|ΔW|={w_norms['total']:.4f}")

    return {
        "rule": rule.value, "seed": seed, "ep_alpha": ep_alpha,
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_stability_experiment(
    rule: Rule,
    x, y,
    seeds: list[int],
    threshold: float,
    learning_rate: float,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
    n_updates: int,
    ep_alpha: float = 0.3,
    n_free_conv_curve: int = 20,
) -> list[dict]:
    results = []
    for seed in seeds:
        print(f"\n--- Rule: {rule.value}  Seed: {seed} ---")
        r = run_stability_seed(
            rule=rule, x=x, y=y, seed=seed,
            threshold=threshold, learning_rate=learning_rate,
            n_warmup=n_warmup, n_clamped=n_clamped, n_free=n_free,
            n_warmup_long=n_warmup_long, n_updates=n_updates,
            ep_alpha=ep_alpha, n_free_conv_curve=n_free_conv_curve,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stability analysis: flip rates, drift, weight norms.")
    p.add_argument("--rules", nargs="+", choices=[r.value for r in Rule],
                   default=[r.value for r in Rule])
    p.add_argument("--threshold",        type=float, default=1.7)
    p.add_argument("--learning-rate",    type=float, default=0.05)
    p.add_argument("--n-warmup",         type=int,   default=1)
    p.add_argument("--n-warmup-long",    type=int,   default=10)
    p.add_argument("--n-clamped",        type=int,   default=5)
    p.add_argument("--n-free",           type=int,   default=5)
    p.add_argument("--n-free-curve",     type=int,   default=20,
                   help="Free-phase steps used for the convergence curve (default 20).")
    p.add_argument("--ep-alpha",         type=float, default=0.3)
    p.add_argument("--n-updates",        type=int,   default=60)
    p.add_argument("--n-seeds",          type=int,   default=3)
    p.add_argument("--image-idx",        type=int,   default=0)
    p.add_argument("--output",           type=str,   default="results/stability/stability_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x, y = load_single_image(args.image_idx)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for rule_str in args.rules:
        rule = Rule(rule_str)
        results = run_stability_experiment(
            rule=rule, x=x, y=y,
            seeds=list(range(args.n_seeds)),
            threshold=args.threshold,
            learning_rate=args.learning_rate,
            n_warmup=args.n_warmup,
            n_clamped=args.n_clamped,
            n_free=args.n_free,
            n_warmup_long=args.n_warmup_long,
            n_updates=args.n_updates,
            ep_alpha=args.ep_alpha,
            n_free_conv_curve=args.n_free_curve,
        )
        all_results.extend(results)

    payload = {"config": vars(args), "results": all_results}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
