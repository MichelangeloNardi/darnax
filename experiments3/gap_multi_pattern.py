#!/usr/bin/env python3
"""Multi-pattern train/inference gap experiment.

Extends gap_experiment.py to study how the C↔D gap evolves as more patterns are
trained sequentially.

Protocol for each rule:
  For pattern_idx in 1..max_patterns:
    - Train on pattern[pattern_idx] for n_updates weight updates
    - After training, measure CD overlap for all patterns seen so far
      (to detect cross-pattern interference)

Rules compared: current (rule 1) vs long_warmup (rule 4).
These are the two most interesting ones from the single-pattern study.

Output JSON schema:
  {
    "config": { ... },
    "results": [
      {
        "rule": str,
        "seed": int,
        "pattern_sequence": [   # one entry per pattern added
          {
            "n_patterns_trained": int,   # how many patterns trained so far (1-indexed)
            "pattern_idx": int,          # CIFAR10 image index just trained
            "final_CD_per_pattern": {    # CD_fc after training, measured per pattern
              "0": float,
              "1": float,
              ...
            },
            "final_margin_per_pattern": {  # soft_margin for C state, per pattern
              "0": float,
              ...
            },
            "final_acc_per_pattern": {     # accuracy for D state, per pattern
              "0": float,
              ...
            },
          }
        ]
      }
    ]
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

# Re-use all primitives from gap_experiment.py
from gap_experiment import (
    Rule,
    PhaseStates,
    build_conv_model,
    build_optimizer,
    run_warmup,
    run_clamped,
    run_free,
    extract_states_rule1,
    extract_states_rule4,
    compute_grads_rule1,
    apply_grads,
    compute_quadrilateral,
    soft_margin,
    accuracy,
)
from darnax.datasets.classification.cifar10 import Cifar10


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_images(image_indices: list[int], dataset_batch_size: int = 64):
    data = Cifar10(batch_size=dataset_batch_size, linear_projection=None,
                   label_mode="pm1", x_transform="identity")
    data.build(key=jax.random.PRNGKey(0))
    x_all, y_all = next(iter(data))
    if x_all.ndim == 2:
        x_all = x_all.reshape(x_all.shape[0], 32, 32, 3)
    xs = [x_all[i:i+1] for i in image_indices]
    ys = [y_all[i:i+1] for i in image_indices]
    return xs, ys


# ---------------------------------------------------------------------------
# Measurement: CD overlap for a single pattern given current weights
# ---------------------------------------------------------------------------

def measure_pattern(
    rule: Rule,
    orch,
    state_template,
    x, y,
    rng,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
) -> tuple[float, float, float, Any]:
    """Return (CD_fc, soft_margin_C, acc_D, new_rng)."""
    rng, step_rng = jax.random.split(rng)
    if rule == Rule.CURRENT:
        states, step_rng = extract_states_rule1(
            orch, state_template, x, y, step_rng, n_warmup, n_clamped, n_free)
    elif rule == Rule.LONG_WARMUP:
        states, step_rng = extract_states_rule4(
            orch, state_template, x, y, step_rng,
            n_warmup_long, n_clamped, n_free, n_warmup)
    else:
        raise ValueError(f"Unsupported rule for multi-pattern: {rule}")

    overlaps_fc = compute_quadrilateral(states, layer_idx=2)
    sm = soft_margin(orch, states.C, y)
    acc = accuracy(orch, states.D, y)
    return overlaps_fc["CD"], sm, acc, rng


# ---------------------------------------------------------------------------
# One training step for a single pattern (returns updated orch + opt_state)
# ---------------------------------------------------------------------------

def train_step_on_pattern(
    rule: Rule,
    orch,
    opt_state,
    optimizer,
    state_template,
    x, y,
    rng,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
) -> tuple[Any, Any, Any]:
    """One weight update on (x, y). Returns (new_orch, new_opt_state, new_rng)."""
    rng, step_rng = jax.random.split(rng)
    if rule == Rule.CURRENT:
        states, step_rng = extract_states_rule1(
            orch, state_template, x, y, step_rng, n_warmup, n_clamped, n_free)
        grads, step_rng = compute_grads_rule1(orch, states, step_rng)
    elif rule == Rule.LONG_WARMUP:
        states, step_rng = extract_states_rule4(
            orch, state_template, x, y, step_rng,
            n_warmup_long, n_clamped, n_free, n_warmup)
        grads, step_rng = compute_grads_rule1(orch, states, step_rng)
    else:
        raise ValueError(f"Unsupported rule: {rule}")
    orch, opt_state = apply_grads(orch, grads, optimizer, opt_state)
    return orch, opt_state, rng


# ---------------------------------------------------------------------------
# Full multi-pattern experiment for one (rule, seed)
# ---------------------------------------------------------------------------

def run_multi_pattern_experiment(
    rule: Rule,
    xs: list,
    ys: list,
    seed: int,
    threshold: float,
    learning_rate: float,
    momentum: float,
    mask_prob: float,
    n_warmup: int,
    n_clamped: int,
    n_free: int,
    n_warmup_long: int,
    n_updates_per_pattern: int,
) -> dict[str, Any]:
    state_template, orch = build_conv_model(
        seed=seed, threshold_conv=threshold, threshold_fc=threshold,
        threshold_out=threshold, mask_prob=mask_prob, field_momentum=momentum,
    )
    optimizer, opt_state = build_optimizer(orch, lr=learning_rate)
    rng = jax.random.PRNGKey(seed)

    pattern_sequence = []

    for pat_idx, (x, y) in enumerate(zip(xs, ys)):
        n_trained = pat_idx + 1
        print(f"\n  [{rule.value}] seed={seed}  training on pattern {pat_idx} "
              f"(n_patterns so far: {n_trained})")

        # Train n_updates_per_pattern steps on this pattern
        for k in range(n_updates_per_pattern):
            orch, opt_state, rng = train_step_on_pattern(
                rule, orch, opt_state, optimizer, state_template,
                x, y, rng, n_warmup, n_clamped, n_free, n_warmup_long,
            )
            if k % 10 == 0 or k == n_updates_per_pattern - 1:
                print(f"    step {k+1}/{n_updates_per_pattern}", end="\r", flush=True)
        print()

        # After training, measure CD / margin / acc for all patterns seen so far
        cd_per_pat, margin_per_pat, acc_per_pat = {}, {}, {}
        for prev_idx in range(n_trained):
            cd, sm, acc_val, rng = measure_pattern(
                rule, orch, state_template,
                xs[prev_idx], ys[prev_idx], rng,
                n_warmup, n_clamped, n_free, n_warmup_long,
            )
            cd_per_pat[str(prev_idx)] = cd
            margin_per_pat[str(prev_idx)] = sm
            acc_per_pat[str(prev_idx)] = acc_val

        print(f"  After {n_trained} pattern(s): "
              f"CD={[f'{v:.3f}' for v in cd_per_pat.values()]}  "
              f"margin={[f'{v:.2f}' for v in margin_per_pat.values()]}")

        pattern_sequence.append({
            "n_patterns_trained": n_trained,
            "pattern_idx": pat_idx,
            "final_CD_per_pattern": cd_per_pat,
            "final_margin_per_pattern": margin_per_pat,
            "final_acc_per_pattern": acc_per_pat,
        })

    return {
        "rule": rule.value,
        "seed": seed,
        "pattern_sequence": pattern_sequence,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-pattern train/inference gap experiment.")
    p.add_argument("--rules", nargs="+",
                   choices=["current", "long_warmup"],
                   default=["current", "long_warmup"],
                   help="Rules to run. Default: current long_warmup.")
    p.add_argument("--max-patterns",   type=int,   default=10,
                   help="How many patterns to train sequentially. Default: 10.")
    p.add_argument("--threshold",      type=float, required=True)
    p.add_argument("--learning-rate",  type=float, required=True)
    p.add_argument("--momentum",       type=float, default=0.0)
    p.add_argument("--mask",           type=float, default=0.0)
    p.add_argument("--n-warmup",       type=int,   default=1)
    p.add_argument("--n-warmup-long",  type=int,   default=30,
                   help="Warmup steps for long_warmup rule. Default: 30.")
    p.add_argument("--n-clamped",      type=int,   default=5)
    p.add_argument("--n-free",         type=int,   default=5)
    p.add_argument("--n-updates",      type=int,   default=60,
                   help="Weight updates per pattern. Default: 60.")
    p.add_argument("--n-seeds",        type=int,   default=3)
    p.add_argument("--image-start",    type=int,   default=0,
                   help="First CIFAR10 image index. Default: 0.")
    p.add_argument("--output",         type=str,   default="gap_multi_pattern_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_indices = list(range(args.image_start, args.image_start + args.max_patterns))
    xs, ys = load_images(image_indices)
    print(f"Loaded {args.max_patterns} patterns (CIFAR10 images {image_indices[0]}–{image_indices[-1]})")

    all_results = []
    for rule_str in args.rules:
        rule = Rule(rule_str)
        for seed in range(args.n_seeds):
            print(f"\n=== Rule: {rule.value}  Seed: {seed} ===")
            result = run_multi_pattern_experiment(
                rule=rule,
                xs=xs, ys=ys,
                seed=seed,
                threshold=args.threshold,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                mask_prob=args.mask,
                n_warmup=args.n_warmup,
                n_clamped=args.n_clamped,
                n_free=args.n_free,
                n_warmup_long=args.n_warmup_long,
                n_updates_per_pattern=args.n_updates,
            )
            all_results.append(result)

    payload = {"config": vars(args), "results": all_results}
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
