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

Tuned HPs per target (8-ep screen optimum):

| target | lr_j | lr_win | j_d | threshold_j | free_n_iter | entropy_beta | screen probe_D |
|---|---|---|---|---|---|---|---|
| baseline (ref) | 9.7e-4 | 2.3e-2 | 0.895 | 1.98 | 6 | 0.345 | — |
| chl_c16 | 3.2e-4 | 7.7e-2 | 1.129 | 1.263 | 8 | 0.786 | 0.400 |
| standard_c24 | 8.6e-4 | 2.8e-2 | 0.952 | 2.199 | 12 | 0.871 | 0.461 |
| split_c24 | 5.1e-4 | 2.5e-3 | 1.019 | 2.247 | 12 | 0.843 | 0.430 |
| standard_c32 | 4.0e-3 | 1.4e-2 | 0.811 | 1.582 | 4 | 0.594 | 0.475 |
| split_c32 | 4.0e-3 | 1.4e-2 | 0.811 | 1.582 | 4 | 0.594 | 0.438 |
| partial_c24 | 7.5e-4 | 6.6e-3 | 0.751 | 1.853 | 4 | 0.485 | 0.442 |

Confirmed re-run (`final.py`, 3 seeds × 20 epochs), baseline-config vs tuned-config probe_D:

| target | baseline-cfg probe_D | tuned probe_D | tuned head_D |
|---|---|---|---|
| standard_c24 | 0.464 ± 0.001 | 0.461 ± 0.005 | 0.321 |
| split_c24 | 0.420 ± 0.014 | 0.433 ± 0.002 | 0.377 |
| partial_c24 | 0.433 ± 0.009 | 0.438 ± 0.007 | 0.349 |
| standard_c32 | 0.478 ± 0.002 | 0.475 ± 0.004 | 0.224 |
| split_c32 | 0.431 ± 0.012 | 0.421 ± 0.013 | 0.365 |
| chl_c16 | 0.186 ± 0.043 | 0.397 ± 0.014 | 0.100 |

(For `chl_c16`, "baseline-cfg" = the contrastive rule at `best_channel_entropy_cfg`, "tuned" =
the contrastive rule at its tuned config. The DynamicalTrainer reference on standard C16 is
probe_D ~0.44, exp 12.)

**Factual observations (numbers only):**
- chl_c16: tuned 0.397 ± 0.014 vs baseline-cfg 0.186 ± 0.043. Tuned lr_j is 3× lower than the
  baseline (3.2e-4 vs 9.7e-4).
- standard_c24/c32: tuned ≈ baseline-cfg (0.461 vs 0.464 ; 0.475 vs 0.478).
- split_c24: tuned 0.433 vs baseline-cfg 0.420; standard_c24 tuned 0.461. split_c32: tuned
  0.421 vs baseline-cfg 0.431; standard_c32 tuned 0.475. partial_c24: tuned 0.438 vs
  baseline-cfg 0.433; standard_c24 tuned 0.461.
- The 8-ep/1-seed screen optimum did not always transfer to 3-seed/20-ep: split_c32 tuned
  (0.421) is below its baseline-cfg (0.431), and below its own screen value (0.438).

Tuned configs: `replicate/tuned_<target>.json`. Studies: `results/study_<target>.json`. Merged
3-seed results: `results/final.json`.
