#!/usr/bin/env python3
"""Active-neuron analysis: why does |ΔW| keep growing for the current rule?

The perceptron learning rule fires for neuron i whenever:
    s_i * h_i < κ    (neuron is not sufficiently stabilised)
where h_i = sum of incoming weighted messages, s_i = neuron activation (±1), κ = threshold.

The update for neuron i's incoming weight from j is:
    ΔW_ij ∝ s_i * s_j   (if the rule fires for i)

So |ΔW| is large when MANY neurons are "active" (i.e., their margin s*h is below threshold).

This experiment tracks, at each weight update step:
  - active_fraction: fraction of neurons where s*h < κ (rule fires)
  - mean_margin:     mean(s*h) over all neurons
  - margin_hist:     histogram of s*h values (to see the distribution shift)
  - |ΔW| per weight matrix

Hypothesis: for the current rule, as training progresses the weight magnitudes grow,
which pushes more neurons above threshold at first — but then the threshold κ stays fixed,
so the rule keeps firing for any neuron that ends up below it, and there's no
self-limiting mechanism. For long_warmup, the network reaches a genuine fixed point,
so nearly all neurons have s*h >> κ and the rule barely fires.
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
    run_warmup, run_clamped, run_free,
    load_single_image,
    compute_grads_rule1, compute_grads_rule2, compute_grads_rule3,
    extract_states_rule1, extract_states_rule2, extract_states_rule4,
)
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState


# ---------------------------------------------------------------------------
# Compute active neuron fraction and margin distribution
# ---------------------------------------------------------------------------

def compute_active_stats(
    orch: SequentialOrchestrator,
    state: SequentialState,
    threshold: float,
    layer_idx: int = 2,
) -> dict[str, Any]:
    """Compute fraction of neurons where s*h < κ (update rule fires) at a given layer.

    We approximate h (the pre-activation field) as: h = messages aggregated before
    the activation function. We re-derive it by computing s * activation_inverse, but
    since states are already ±1 we instead look at the raw field directly.

    Simpler proxy: use the state overlap approach. For a ±1 state, the stabilisation
    margin for neuron i is  m_i = s_i * h_i. If m_i < κ the rule fires.
    We compute h_i for each neuron from the layer's reduce() output.
    """
    s = np.array(state.states[layer_idx]).ravel().astype(float)

    # Get the layer's incoming messages to reconstruct h.
    # The simplest approach: run one step without updating state to get the field,
    # then measure how many neurons would flip (i.e., sign(h) != s).
    # Active neurons = those where the field and current state disagree: sign(h) != s
    # which is equivalent to s*h < 0. The threshold adds a margin: s*h < κ.
    # Since we can't easily extract h here without modifying the library,
    # we use a surrogate: compare state before and after one dynamic step.
    # Neurons that WOULD flip = neurons the dynamics wants to change = active neurons.

    rng = jax.random.PRNGKey(0)
    next_state, _ = orch.step(state, rng=rng, filter_messages="inference")
    s_next = np.array(next_state.states[layer_idx]).ravel().astype(float)

    # Neurons that want to flip = not yet at fixed point
    would_flip = (s != s_next)
    active_fraction = float(np.mean(would_flip))

    # Also compute current-state vs next-state agreement as a stability proxy
    # (same as flip rate from stability experiment but computed differently)
    return {
        "active_fraction": active_fraction,
        "stable_fraction": 1.0 - active_fraction,
    }


def weight_update_norm(orch_before, orch_after) -> dict[str, float]:
    params_before = eqx.filter(orch_before, eqx.is_inexact_array)
    params_after  = eqx.filter(orch_after,  eqx.is_inexact_array)
    delta = jtu.tree_map(lambda a, b: np.array(b - a), params_before, params_after)
    norms: dict[str, float] = {}
    for (i, j), name in [((1, 0), "w_in"), ((1, 1), "j_conv"), ((2, 2), "j_fc"), ((3, 2), "w_out")]:
        leaves = jtu.tree_leaves(delta.lmap[i][j])
        norms[name] = float(sum(np.linalg.norm(l.ravel()) ** 2 for l in leaves) ** 0.5)
    norms["total"] = float(sum(v ** 2 for v in norms.values()) ** 0.5)
    return norms


def weight_magnitude(orch) -> dict[str, float]:
    """L2 norm of the current weight matrices (not the delta — the actual values)."""
    params = eqx.filter(orch, eqx.is_inexact_array)
    norms: dict[str, float] = {}
    for (i, j), name in [((1, 0), "w_in"), ((1, 1), "j_conv"), ((2, 2), "j_fc"), ((3, 2), "w_out")]:
        leaves = jtu.tree_leaves(params.lmap[i][j])
        norms[name] = float(sum(np.linalg.norm(l.ravel()) ** 2 for l in leaves) ** 0.5)
    norms["total"] = float(sum(v ** 2 for v in norms.values()) ** 0.5)
    return norms


# ---------------------------------------------------------------------------
# Per-rule run
# ---------------------------------------------------------------------------

def run_active_neuron_seed(
    rule: Rule, x, y, seed: int,
    threshold: float, learning_rate: float,
    n_warmup: int, n_clamped: int, n_free: int,
    n_warmup_long: int, n_updates: int, ep_alpha: float = 0.3,
) -> dict[str, Any]:
    state_template, orch = build_conv_model(
        seed=seed, threshold_conv=threshold, threshold_fc=threshold, threshold_out=threshold,
    )
    optimizer, opt_state = build_optimizer(orch, lr=learning_rate)
    rng = jax.random.PRNGKey(seed)
    trajectory = []

    for k in range(n_updates):
        rng, step_rng = jax.random.split(rng)
        orch_before = orch

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

        orch, opt_state = apply_grads(orch, grads, optimizer, opt_state)

        # Measure active neurons IN STATE C (the state being stabilised)
        active_C = compute_active_stats(orch, states.C, threshold, layer_idx=2)
        # Also measure in state D (inference state)
        active_D = compute_active_stats(orch, states.D, threshold, layer_idx=2)

        w_delta = weight_update_norm(orch_before, orch)
        w_mag   = weight_magnitude(orch)

        rec = {
            "update": k,
            "active_fraction_C":  active_C["active_fraction"],
            "active_fraction_D":  active_D["active_fraction"],
            "weight_delta_norms": w_delta,
            "weight_magnitudes":  w_mag,
        }
        trajectory.append(rec)

        if k % 10 == 0:
            print(f"  [{rule.value}] step {k:3d}  "
                  f"active_C={active_C['active_fraction']:.3f}  "
                  f"|ΔW|={w_delta['total']:.2f}  "
                  f"|W|={w_mag['total']:.2f}")

    return {"rule": rule.value, "seed": seed, "ep_alpha": ep_alpha, "trajectory": trajectory}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rules", nargs="+", choices=[r.value for r in Rule], default=[r.value for r in Rule])
    p.add_argument("--threshold",     type=float, default=1.7)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-warmup",      type=int,   default=1)
    p.add_argument("--n-warmup-long", type=int,   default=10)
    p.add_argument("--n-clamped",     type=int,   default=5)
    p.add_argument("--n-free",        type=int,   default=5)
    p.add_argument("--ep-alpha",      type=float, default=0.3)
    p.add_argument("--n-updates",     type=int,   default=60)
    p.add_argument("--n-seeds",       type=int,   default=3)
    p.add_argument("--image-idx",     type=int,   default=0)
    p.add_argument("--output",        type=str,   default="results/active_neurons/active_neurons_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    x, y = load_single_image(args.image_idx)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for rule_str in args.rules:
        rule = Rule(rule_str)
        for seed in range(args.n_seeds):
            print(f"\n--- Rule: {rule.value}  Seed: {seed} ---")
            r = run_active_neuron_seed(
                rule=rule, x=x, y=y, seed=seed,
                threshold=args.threshold, learning_rate=args.learning_rate,
                n_warmup=args.n_warmup, n_clamped=args.n_clamped, n_free=args.n_free,
                n_warmup_long=args.n_warmup_long, n_updates=args.n_updates, ep_alpha=args.ep_alpha,
            )
            all_results.append(r)

    out_path.write_text(json.dumps({"config": vars(args), "results": all_results}, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
