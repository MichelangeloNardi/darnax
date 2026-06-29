# Experiment 13 — per-target hyperparameter tuning

Each new rule / architecture is tuned to its OWN optimum before comparison, because
`best_channel_entropy_cfg` is tuned for the DynamicalTrainer on the C=16 standard architecture
(reusing it for a new rule/architecture is unfair — see exp 12, where CHL is unstable at that
config). One Optuna study per target.

## Setup
- **Objective:** `probe_D` (Adam linear probe on pooled D). The probe does not use W_out, so
  `lr_wout` is irrelevant and not tuned.
- **Tuned HPs (6 main config knobs driving D):** `lr_j`, `lr_win`, `j_d`, `threshold_j`,
  `free_n_iter`, `entropy_beta`. Ranges: lr_j ∈ [1e-4, 5e-3] log, lr_win ∈ [2e-3, 8e-2] log,
  j_d ∈ [0.5, 1.25], threshold_j ∈ [1.0, 3.2], free_n_iter ∈ {4,6,8,12}, entropy_beta ∈ [0.1, 0.9].
  Held at baseline: `lr_wout`, `threshold_win`, `clamped_n_iter`, `momentum`,
  `kernel_decay_rate`, `strength_back`.
- **Budget:** Optuna TPE, 20 trials × 8-epoch screen × 1 seed. Best config per target written to
  `replicate/tuned_<target>.json`; full study to `results/study_<target>.json`.
- **Final:** after tuning, `final.py` re-runs each target's best config at 3 seeds × 20 epochs
  (the fair number), comparing tuned vs the baseline-config result.

## Targets (architecture + rule)
| target | architecture | rule | machine |
|---|---|---|---|
| chl_c16 | standard C16 | contrastive (CHL) | w01 |
| partial_c24 | partial-overlap C24 | DynamicalTrainer | w01 |
| standard_c24 | standard C24 | DynamicalTrainer | w02 |
| split_c24 | split_C24_I16_L8_N0 | DynamicalTrainer | w02 |
| standard_c32 | standard C32 | DynamicalTrainer | w03 |
| split_c32 | split_C32_I16_L8_N8 | DynamicalTrainer | w03 |

(2 targets per machine, run sequentially; w01/w02/w03 in parallel.)

## Files
- `tune.py` — `--target <name>` runs one Optuna study; reuses exp-10 `arch`, exp-11 `splitarch`,
  exp-12 CHL loop, and `bc.offline_probe`. `--smoke` for a tiny run.
- `final.py` — re-runs the tuned configs at 3 seeds × 20 epochs (added after tuning).
- `.gitignore` — `results/*_smoke.json`, `models/`.

Run one target (cluster): `XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/13-hp_tuning/tune.py --target standard_c24`

## Results

_Pending tuning sweep._
