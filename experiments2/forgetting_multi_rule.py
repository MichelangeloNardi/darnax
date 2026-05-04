#!/usr/bin/env python3
"""Point 4: Hyperparameter tuning via forgetting crossing point.

Runs the two-image forgetting experiment for each of the four learning rules,
sweeping over (threshold, learning_rate) grids.  For each configuration the
key metric is the crossing-point window: how many B-phase steps both images
are simultaneously classified correctly (n_steps_both_correct_during_B).

The tuning criterion is: maximize soft-margin at the crossing point.
We define this as: max over B-phase steps of min(margin_A, margin_B)
when both are positive — i.e. the best simultaneous margin.

Rules:
  current      warmup → clamped → free;   stabilize C (free)
  approximate  warmup → free → clamped;   stabilize C (clamped)
  long_warmup  long-warmup → clamped → free; stabilize C (free)
  ep           warmup → free → clamped;   stabilize C - alpha*B (EP)

Usage — run one rule at a time (recommended for HPC):
  python forgetting_multi_rule.py --rule current \\
      --threshold-grid 1.3 1.5 1.7 1.9 \\
      --lr-grid 0.01 0.03 0.05 0.1 \\
      --n-seeds 3 --output forgetting_current.json

Then load all four JSONs in the analysis notebook to find best hyperparams
and pick the winner per rule.

Output JSON schema:
  {
    "config": { ... },
    "grid_results": [
      {
        "threshold": float, "learning_rate": float,
        "rule": str,
        "seeds": [
          {
            "seed": int,
            "steps_to_memorize_a": int,
            "steps_to_memorize_b": int,
            "n_steps_both_correct": int,
            "best_simultaneous_margin": float,   ← key tuning criterion
            "crossing_window_margins_a": [...],  ← margin_A at steps where both correct
            "crossing_window_margins_b": [...]
          }
        ],
        "mean_n_both_correct": float,
        "mean_best_sim_margin": float,
      }
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments3"))

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
import optax

from forgetting_experiment import (
    build_conv_model,
    build_optimizer as build_optimizer_forgetting,
    evaluate_sample,
    load_samples,
    memorized,
    metrics_dict,
    final_state_overlap,
)
from gap_experiment import (
    Rule,
    build_optimizer as build_optimizer_gap,
    run_warmup,
    run_clamped,
    run_free,
    apply_grads,
)
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState


# ---------------------------------------------------------------------------
# Rule-aware single weight update
# ---------------------------------------------------------------------------

def _ep_update(orch, state_template, x, y, rng,
               n_warmup, n_clamped, n_free, ep_alpha):
    """EP: backward(C_clamped) - alpha * backward(B_free)."""
    s = state_template.init(x, y)
    s, rng = _run(orch, s, rng, n_warmup, "inference")   # warmup
    B = s
    s, rng = _run(orch, s, rng, n_free, "inference")     # free → B
    # NOTE: B is the free state; C is the clamped state
    B = s
    s, rng = _run(orch, s, rng, n_clamped, "all")        # clamped → C
    C = s
    rng, rng_c, rng_b = jax.random.split(rng, 3)
    gc = eqx.filter(orch.backward(C, rng=rng_c), eqx.is_inexact_array)
    gb = eqx.filter(orch.backward(B, rng=rng_b), eqx.is_inexact_array)
    grads = jtu.tree_map(lambda c, b: c - ep_alpha * b, gc, gb)
    return grads, rng


def _run(orch, state, rng, n_steps, filter_messages):
    for _ in range(n_steps):
        state, rng = orch.step(state, rng=rng, filter_messages=filter_messages)
    return state, rng


def _get_stabilize_state(orch, state_template, x, y, rng,
                         rule: str, n_warmup, n_clamped, n_free, n_warmup_long):
    """Return (grads_filtered, rng) for the given rule."""
    s = state_template.init(x, y)

    if rule == "current":
        s, rng = _run(orch, s, rng, n_warmup, "inference")
        s, rng = _run(orch, s, rng, n_clamped, "all")
        s, rng = _run(orch, s, rng, n_free, "inference")   # C = free
        grads = eqx.filter(orch.backward(s, rng=rng), eqx.is_inexact_array)

    elif rule == "approximate":
        s, rng = _run(orch, s, rng, n_warmup, "inference")
        s, rng = _run(orch, s, rng, n_free, "inference")
        s, rng = _run(orch, s, rng, n_clamped, "all")      # C = clamped
        grads = eqx.filter(orch.backward(s, rng=rng), eqx.is_inexact_array)

    elif rule == "long_warmup":
        s, rng = _run(orch, s, rng, n_warmup_long, "inference")
        s, rng = _run(orch, s, rng, n_clamped, "all")
        s, rng = _run(orch, s, rng, n_free, "inference")   # C = free
        grads = eqx.filter(orch.backward(s, rng=rng), eqx.is_inexact_array)

    elif rule == "ep":
        raise ValueError("Use _ep_update() directly for EP rule")
    else:
        raise ValueError(f"Unknown rule: {rule}")

    return grads, rng


def apply_update_rule(
    orch, opt_state, optimizer,
    state_template, x, y, rng,
    rule: str,
    n_warmup: int, n_clamped: int, n_free: int,
    n_warmup_long: int, ep_alpha: float,
):
    """One weight update for the given rule. Returns (new_orch, new_opt_state, rng)."""
    if rule == "ep":
        grads, rng = _ep_update(
            orch, state_template, x, y, rng,
            n_warmup, n_clamped, n_free, ep_alpha,
        )
    else:
        grads, rng = _get_stabilize_state(
            orch, state_template, x, y, rng,
            rule, n_warmup, n_clamped, n_free, n_warmup_long,
        )

    params_filtered = eqx.filter(orch, eqx.is_inexact_array)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=params_filtered)
    new_orch = eqx.apply_updates(orch, updates)
    return new_orch, new_opt_state, rng


# ---------------------------------------------------------------------------
# Core forgetting loop (rule-aware)
# ---------------------------------------------------------------------------

def run_forgetting_seed(
    *,
    x_a, y_a, x_b, y_b,
    rule: str,
    threshold: float,
    learning_rate: float,
    seed: int,
    momentum: float,
    mask_prob: float,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
    ep_alpha: float,
    max_updates_a: int,
    max_updates_b: int,
    consecutive_k: int,
) -> dict[str, Any]:
    state_template, orch = build_conv_model(
        seed=seed, threshold_conv=threshold, threshold_fc=threshold,
        threshold_out=threshold, mask_prob=mask_prob, field_momentum=momentum,
    )
    # Use gap-style optimizer (single lr for all params)
    optimizer, opt_state = build_optimizer_gap(orch, lr=learning_rate)
    rng = jax.random.PRNGKey(seed)

    # --- Phase A: train on A until memorized ---
    recent_a = []
    steps_to_memorize_a = None
    last_state_a = None
    for step in range(max_updates_a + 1):
        m_a, state_a, _, rng = evaluate_sample(
            orch, state_template, x_a, y_a, rng,
            n_warmup=n_warmup, n_clamped=n_clamped, n_free=n_free,
        )
        recent_a.append(m_a)
        last_state_a = state_a
        if memorized(recent_a, consecutive_k):
            steps_to_memorize_a = step
            break
        if step < max_updates_a:
            orch, opt_state, rng = apply_update_rule(
                orch, opt_state, optimizer,
                state_template, x_a, y_a, rng,
                rule=rule,
                n_warmup=n_warmup, n_clamped=n_clamped, n_free=n_free,
                n_warmup_long=n_warmup_long, ep_alpha=ep_alpha,
            )
    if steps_to_memorize_a is None:
        steps_to_memorize_a = max_updates_a + 1

    # --- Phase B: evaluate A+B at every step, train on B ---
    recent_b = []
    steps_to_memorize_b = None
    margins_a_both, margins_b_both = [], []
    n_both_correct = 0

    for step in range(max_updates_b + 1):
        m_a, _, _, rng = evaluate_sample(
            orch, state_template, x_a, y_a, rng,
            n_warmup=n_warmup, n_clamped=n_clamped, n_free=n_free,
        )
        m_b, _, _, rng = evaluate_sample(
            orch, state_template, x_b, y_b, rng,
            n_warmup=n_warmup, n_clamped=n_clamped, n_free=n_free,
        )
        recent_b.append(m_b)

        both_correct = m_a.accuracy >= 1.0 and m_b.accuracy >= 1.0
        if both_correct:
            n_both_correct += 1
            margins_a_both.append(m_a.soft_margin)
            margins_b_both.append(m_b.soft_margin)

        if memorized(recent_b, consecutive_k):
            steps_to_memorize_b = step
            break
        if step < max_updates_b:
            orch, opt_state, rng = apply_update_rule(
                orch, opt_state, optimizer,
                state_template, x_b, y_b, rng,
                rule=rule,
                n_warmup=n_warmup, n_clamped=n_clamped, n_free=n_free,
                n_warmup_long=n_warmup_long, ep_alpha=ep_alpha,
            )

    if steps_to_memorize_b is None:
        steps_to_memorize_b = max_updates_b + 1

    # Best simultaneous margin: max over crossing steps of min(margin_A, margin_B)
    if margins_a_both:
        sim_margins = [min(a, b) for a, b in zip(margins_a_both, margins_b_both)]
        best_sim_margin = float(max(sim_margins))
    else:
        best_sim_margin = float("-inf")

    return {
        "seed": seed,
        "steps_to_memorize_a": steps_to_memorize_a,
        "steps_to_memorize_b": steps_to_memorize_b,
        "n_steps_both_correct": n_both_correct,
        "best_simultaneous_margin": best_sim_margin,
        "crossing_window_margins_a": margins_a_both,
        "crossing_window_margins_b": margins_b_both,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid(
    rule: str,
    x_a, y_a, x_b, y_b,
    threshold_grid: list[float],
    lr_grid: list[float],
    n_seeds: int,
    momentum: float,
    mask_prob: float,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
    ep_alpha: float,
    max_updates_a: int,
    max_updates_b: int,
    consecutive_k: int,
) -> list[dict[str, Any]]:
    grid_results = []
    total = len(threshold_grid) * len(lr_grid) * n_seeds
    done = 0

    for threshold in threshold_grid:
        for lr in lr_grid:
            seed_results = []
            for seed in range(n_seeds):
                done += 1
                print(f"[{done}/{total}]  rule={rule}  threshold={threshold}  "
                      f"lr={lr}  seed={seed}")
                res = run_forgetting_seed(
                    x_a=x_a, y_a=y_a, x_b=x_b, y_b=y_b,
                    rule=rule,
                    threshold=threshold,
                    learning_rate=lr,
                    seed=seed,
                    momentum=momentum,
                    mask_prob=mask_prob,
                    n_warmup=n_warmup,
                    n_clamped=n_clamped,
                    n_free=n_free,
                    n_warmup_long=n_warmup_long,
                    ep_alpha=ep_alpha,
                    max_updates_a=max_updates_a,
                    max_updates_b=max_updates_b,
                    consecutive_k=consecutive_k,
                )
                seed_results.append(res)
                print(f"    steps_A={res['steps_to_memorize_a']}  "
                      f"steps_B={res['steps_to_memorize_b']}  "
                      f"both_correct={res['n_steps_both_correct']}  "
                      f"best_sim_margin={res['best_simultaneous_margin']:.3f}")

            mean_both = float(np.mean([r["n_steps_both_correct"] for r in seed_results]))
            mean_margin = float(np.nanmean([r["best_simultaneous_margin"] for r in seed_results
                                           if r["best_simultaneous_margin"] > float("-inf")]) if
                                any(r["best_simultaneous_margin"] > float("-inf") for r in seed_results)
                                else float("nan"))
            grid_results.append({
                "threshold": threshold,
                "learning_rate": lr,
                "rule": rule,
                "seeds": seed_results,
                "mean_n_both_correct": mean_both,
                "mean_best_sim_margin": mean_margin,
            })

    # Print best configs
    valid = [r for r in grid_results if not np.isnan(r["mean_best_sim_margin"])]
    if valid:
        best = max(valid, key=lambda r: r["mean_best_sim_margin"])
        print(f"\nBest config for rule={rule}:")
        print(f"  threshold={best['threshold']}  lr={best['learning_rate']}")
        print(f"  mean_n_both_correct={best['mean_n_both_correct']:.1f}  "
              f"mean_best_sim_margin={best['mean_best_sim_margin']:.3f}")
    else:
        best_by_window = max(grid_results, key=lambda r: r["mean_n_both_correct"])
        print(f"\nNo crossing found. Best by window length: "
              f"threshold={best_by_window['threshold']}  lr={best_by_window['learning_rate']}  "
              f"mean_n_both_correct={best_by_window['mean_n_both_correct']:.1f}")

    return grid_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-rule forgetting experiment with hyperparameter grid search."
    )
    p.add_argument("--rule",
                   choices=["current", "approximate", "long_warmup", "ep"],
                   required=True)
    p.add_argument("--threshold-grid", nargs="+", type=float,
                   default=[1.3, 1.5, 1.7, 1.9],
                   help="Threshold values to sweep. Default: 1.3 1.5 1.7 1.9")
    p.add_argument("--lr-grid", nargs="+", type=float,
                   default=[0.01, 0.03, 0.05, 0.1],
                   help="Learning rate values to sweep. Default: 0.01 0.03 0.05 0.1")
    p.add_argument("--ep-alpha",      type=float, default=0.3,
                   help="EP de-stabilization strength. Default: 0.3 (sweet spot from single-pattern study).")
    p.add_argument("--momentum",      type=float, default=0.0)
    p.add_argument("--mask",          type=float, default=0.0)
    p.add_argument("--n-seeds",       type=int,   default=3)
    p.add_argument("--n-warmup",      type=int,   default=1)
    p.add_argument("--n-warmup-long", type=int,   default=30)
    p.add_argument("--n-clamped",     type=int,   default=5)
    p.add_argument("--n-free",        type=int,   default=5)
    p.add_argument("--max-updates-a", type=int,   default=150)
    p.add_argument("--max-updates-b", type=int,   default=150)
    p.add_argument("--memorization-k",type=int,   default=3)
    p.add_argument("--idx-a",         type=int,   default=0)
    p.add_argument("--idx-b",         type=int,   default=1)
    p.add_argument("--dataset-batch-size", type=int, default=32)
    p.add_argument("--output",        type=str,
                   default=None,
                   help="Output JSON path. Default: forgetting_{rule}.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_a, y_a, x_b, y_b = load_samples(
        args.idx_a, args.idx_b, args.dataset_batch_size, pair_file=None
    )
    print(f"Rule: {args.rule}")
    print(f"Threshold grid: {args.threshold_grid}")
    print(f"LR grid: {args.lr_grid}")
    print(f"Total configs: {len(args.threshold_grid) * len(args.lr_grid)} × {args.n_seeds} seeds")

    grid_results = run_grid(
        rule=args.rule,
        x_a=x_a, y_a=y_a, x_b=x_b, y_b=y_b,
        threshold_grid=args.threshold_grid,
        lr_grid=args.lr_grid,
        n_seeds=args.n_seeds,
        momentum=args.momentum,
        mask_prob=args.mask,
        n_warmup=args.n_warmup,
        n_clamped=args.n_clamped,
        n_free=args.n_free,
        n_warmup_long=args.n_warmup_long,
        ep_alpha=args.ep_alpha,
        max_updates_a=args.max_updates_a,
        max_updates_b=args.max_updates_b,
        consecutive_k=args.memorization_k,
    )

    out_path = Path(args.output or f"forgetting_{args.rule}.json")
    out_path.write_text(json.dumps({"config": vars(args), "grid_results": grid_results}, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
