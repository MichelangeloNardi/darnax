#!/usr/bin/env python3
"""Point 2: How long does the warmup need to be?

Sweeps n_warmup_long ∈ {1, 3, 5, 10, 20, 30} for the long_warmup rule and
measures CD_fc, soft_margin, and accuracy after K weight updates.

For each warmup length the script runs n_seeds full training runs (same as
gap_experiment.py) and reports the trajectory.  The analysis notebook plots
CD vs warmup length at various training steps, making it easy to find the
minimum warmup that achieves CD ≈ 1.

Output JSON schema:
  {
    "config": { ... },
    "results": [
      {
        "n_warmup_long": int,
        "seed": int,
        "trajectory": [
          { "update": int,
            "overlaps_fc": {...}, "overlaps_conv": {...},
            "soft_margin_C": float, "accuracy_D": float }
        ]
      }
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from gap_experiment import (
    Rule,
    build_conv_model,
    build_optimizer,
    extract_states_rule4,
    compute_grads_rule1,
    apply_grads,
    compute_quadrilateral,
    soft_margin,
    accuracy,
    load_single_image,
)
import jax
import numpy as np


# ---------------------------------------------------------------------------
# Single (warmup_len, seed) run
# ---------------------------------------------------------------------------

def run_warmup_length(
    n_warmup_long: int,
    x, y,
    seed: int,
    threshold: float,
    learning_rate: float,
    momentum: float,
    mask_prob: float,
    n_warmup_short: int,
    n_clamped: int,
    n_free: int,
    n_updates: int,
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
        states, step_rng = extract_states_rule4(
            orch, state_template, x, y, step_rng,
            n_warmup_long, n_clamped, n_free, n_warmup_short,
        )
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

        if k % 10 == 0 or k == n_updates - 1:
            print(f"  [warmup={n_warmup_long:2d}] seed={seed} step={k:3d}"
                  f"  CD_fc={overlaps_fc['CD']:.3f}  margin={sm:.2f}  acc={acc:.1f}")

    return {"n_warmup_long": n_warmup_long, "seed": seed, "trajectory": trajectory}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warmup length sweep for the long_warmup rule.")
    p.add_argument("--warmup-lengths", nargs="+", type=int,
                   default=[1, 3, 5, 10, 20, 30],
                   help="Values of n_warmup_long to sweep. Default: 1 3 5 10 20 30.")
    p.add_argument("--threshold",      type=float, required=True)
    p.add_argument("--learning-rate",  type=float, required=True)
    p.add_argument("--momentum",       type=float, default=0.0)
    p.add_argument("--mask",           type=float, default=0.0)
    p.add_argument("--n-warmup",       type=int,   default=1,
                   help="Short warmup for inference state D. Default: 1.")
    p.add_argument("--n-clamped",      type=int,   default=5)
    p.add_argument("--n-free",         type=int,   default=5)
    p.add_argument("--n-updates",      type=int,   default=60)
    p.add_argument("--n-seeds",        type=int,   default=3)
    p.add_argument("--image-idx",      type=int,   default=0)
    p.add_argument("--output",         type=str,   default="gap_warmup_sweep_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x, y = load_single_image(args.image_idx)
    print(f"Image class: {int(np.argmax(np.array(y)[0]))}")
    print(f"Sweeping n_warmup_long ∈ {args.warmup_lengths}")

    all_results = []
    for wl in args.warmup_lengths:
        for seed in range(args.n_seeds):
            print(f"\n=== n_warmup_long={wl}  seed={seed} ===")
            result = run_warmup_length(
                n_warmup_long=wl, x=x, y=y, seed=seed,
                threshold=args.threshold,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                mask_prob=args.mask,
                n_warmup_short=args.n_warmup,
                n_clamped=args.n_clamped,
                n_free=args.n_free,
                n_updates=args.n_updates,
            )
            all_results.append(result)

    payload = {"config": vars(args), "results": all_results}
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
