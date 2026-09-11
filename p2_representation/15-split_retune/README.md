# Experiment 15 — split re-tune with phase-step counts + injection strength

Closes the split question rigorously. Exp 13 tuned 6 HPs per target but **held
`clamped_n_iter`, `warmup_n_iter`, `strength_back` at baseline** — and those are the direct
levers on the split's failure mode (exp 10: with the label confined to the L group, C ≡ D — the
label couldn't move the state). This re-tune adds those three (9 HPs total) and re-tunes the
splits **and** their dense controls over the same space, so the comparison is fair.

## Setup
- **Objective:** `probe_D` (Adam linear probe on pooled D), as in exp 13.
- **9 tuned HPs:** exp-13's `lr_j`, `lr_win`, `j_d`, `threshold_j`, `free_n_iter`,
  `entropy_beta` **+ `clamped_n_iter` ∈ {3,5,8,12,16} + `warmup_n_iter` ∈ {1,2,4} +
  `strength_back` ∈ [0.5, 6.0]**. (Held: `lr_wout`, `threshold_win`, `momentum`,
  `kernel_decay_rate`.)
- Note: D = warmup→free does not use the clamped phase or W_back directly, but **training**
  (warmup→clamped→free) does, so these three shape the learned W_in/J1 → D indirectly.
- **Budget:** Optuna TPE, 40 trials × 8-ep screen × 1 seed → `replicate/tuned9_<target>.json`;
  study → `results/study9_<target>.json`. Then `final.py` re-runs at 3 seeds × 20 ep vs the
  baseline config.

## Targets (5) — split vs dense at matched C
| target | architecture | machine |
|---|---|---|
| standard_c24 | dense C24 (all channels get input+label) | w01 |
| split_c24 | split_C24_I16_L8_N0 | w01 |
| partial_c24 | partial-overlap C24 | w02 |
| split_c32 | split_C32_I16_L8_N8 | w02 |
| standard_c32 | dense C32 | w03 |

**Comparison:** `split_cX` vs `standard_cX` at the **same C** = identical **nominal** params
(masking, not shrinking, so weight tensors are the same size). The split has fewer **effective**
(nonzero) `W_in` params — reported by arch.param_report — which handicaps it; so a split loss is
a strong negative, a split win despite fewer effective params means routing helps.

## Files
- `tune.py` — one Optuna study per target (9-HP `sample_cfg`); reuses exp-13 tune.py
  build/train/measure. `--target`, `--trials`, `--smoke`.
- `final.py` — 3-seed × 20-ep re-run of `tuned9_<target>` vs baseline; reuses exp-13
  final.py::run_cfg. `--targets`, `--out` (per-machine parallel), `--smoke`.
- `.gitignore` — `results/*_smoke.json`, `models/`.

Reference: exp-13 6-HP tuned numbers in `../13-hp_tuning/results/final.json`.

## Results

_Pending sweep._
