# Exp p3-3 — dangerous-spin gated rule (per-position×channel κ), conv C=16

The local rule updates J only where the margin is below threshold:
`J += η·s_i s_j·1[s_i (Js)_i < κ]`. Diagnostics: the units that lose class info C→D are the
weakly field-pinned ones — the rule doesn't stabilise them enough. Idea: make **κ
per-position×channel** and **raise it on dangerous units** (persistently low mean |field_C|) so
the rule fires on a wider margin window there.

**Variant (a) only** — dangerous = low mean |field_C| (label-free during training). Variant (b)
(polarization) dropped: exp p3-2 showed polarization gives no label-free danger signal.

## Mechanism (no darnax-core edit)
The conv gate is `(y*y_hat < threshold)` and an **array threshold broadcasts** over (N,H,W,C);
`threshold` is not touched by the optimizer (backward returns zero for it). So we set the J1
`threshold` leaf to a (H,W,C) κ map each epoch:
```
meanfield_i = mean_n |field_C_i|              (subset, current weights)
dangerous_i = meanfield_i < quantile(meanfield, q)   (bottom-q fraction)
κ_i         = threshold_j + boost·dangerous_i
```
`boost=0` → uniform threshold_j → the standard rule (baseline). The "unit" is per
position×channel (matches the per-spin diagnosis; the gate is per-activation).

## Setup
- Baseline: conv C=16 at `replicate/best_channel_entropy_cfg.json` (probe_D ~0.45; gradient
  reference ~0.51). Metric: probe_D (Adam linear probe on pooled D) + head_D.
- `gated_kappa.py` — functional sweep over (boost, q), 3 seeds × 20 ep; boost=0 = baseline.
  `--smoke` for a tiny CPU run.
- Per the standing per-rule-tuning rule: if any cell beats baseline, `tune_gated.py` (Optuna
  over boost, q, threshold_j, lr_j) follows before a final comparison.
- Fallback (if κ alone doesn't move probe_D): per-**channel** learning-rate η boost on dangerous
  channels (the recurrent kernel is shared across positions, so η is per-channel not per-spin).
- `.gitignore`: `results/smoke.json`, `models/`.

Cluster: `XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python
p3_spin_analysis/3-gated_rule/gated_kappa.py`.

## Results — κ functional sweep (3 seeds × 20 ep; boost=0 = baseline)

| cell | probe_D | head_D |
|---|---|---|
| baseline (boost 0) | 0.438 ± 0.006 | 0.241 |
| boost 0.5, q 0.25 | 0.439 ± 0.002 | 0.309 |
| boost 0.5, q 0.5 | 0.441 ± 0.004 | 0.339 |
| boost 1.0, q 0.25 | 0.435 ± 0.004 | 0.343 |
| boost 1.0, q 0.5 | 0.436 ± 0.005 | 0.381 |
| boost 2.0, q 0.25 | 0.433 ± 0.003 | 0.404 |
| boost 2.0, q 0.5 | 0.432 ± 0.005 | 0.414 |

**Factual observations (numbers only):**
- probe_D (target): 0.432–0.441 across all cells, vs baseline 0.438 — flat, no cell above
  baseline within std; the largest boost (2.0, q 0.5) is 0.432 (≤ baseline). Reference: gradient
  ceiling ~0.51.
- head_D (model's own W_out): rises monotonically with boost, 0.241 (baseline) → 0.414
  (boost 2.0, q 0.5).

Result: `results/gated_kappa.json`. Per the standing rule, κ's new knobs (boost, q) were swept
at the tuned baseline config; probe_D did not move → the per-channel η fallback is run next.

## Results — η fallback (per-channel lr boost on dangerous channels; 3 seeds × 20 ep)

| cell | probe_D | head_D |
|---|---|---|
| baseline (boost 0) | 0.439 ± 0.006 | 0.241 |
| eta 1.0, q 0.25 | 0.440 ± 0.000 | 0.266 |
| eta 1.0, q 0.5 | 0.443 ± 0.001 | 0.251 |
| eta 3.0, q 0.25 | 0.442 ± 0.004 | 0.239 |
| eta 3.0, q 0.5 | 0.439 ± 0.007 | 0.274 |
| eta 6.0, q 0.25 | 0.440 ± 0.004 | 0.269 |
| eta 6.0, q 0.5 | 0.444 ± 0.003 | 0.297 |

**Factual observations (numbers only):**
- probe_D: 0.439–0.444 across all η cells vs baseline 0.439 — flat (within std), up to boost 6×.
- head_D: 0.239–0.297 (baseline 0.241) — small, non-monotonic; smaller than the κ head_D lift
  (which reached 0.414). Reference: gradient ceiling ~0.51.

Result: `results/gated_eta.json`. Both gated-rule knobs (κ firing-frequency, η update-magnitude)
leave probe_D at the ~0.44 baseline.
