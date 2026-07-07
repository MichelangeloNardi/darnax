# Experiment 1 — C→D flip phenomenon in a classical (non-conv) FC net

Does the C→D flip phenomenon (a linear probe trained on the clamped state **C**
collapses on the inference state **D**, via a subset of sign-flipping spins that a
random-flip control shows are class-targeted) also occur in the paper's base **dense
fully-connected** asymmetric recurrent net — where every spin is a distinct unit (no
conv weight-sharing / spatial pooling)?

## Model (`../fc_common.py`)

Edge-for-edge FC translation of the conv entropy architecture (same `DynamicalTrainer`,
same warmup→clamped→free rollout):

| edge | module | rule |
|---|---|---|
| (1,0) W_in  | `FrozenFullyConnected` (frozen) **or** `SparseFullyConnected` (trained) 768→256 | perceptron / none |
| (1,1) J     | `SparseRecurrentDiscrete` 256, sparsity 0.9, sign activation, `j_d` diagonal frozen | perceptron |
| (1,2) W_back| `FrozenRescaledFullyConnected` 10→256 (frozen label feedback) | none |
| (2,1) W_out | `FullyConnected` 256→10 | perceptron |
| (2,2)       | `OutputLayer` | — |

- Input: CIFAR (N,3072)∈[0,1] → 2×2 avg-pool to 16×16×3 = 768 → [−1,1].
- Hidden: N = 256 spins (±1). State = [768, 256, 10].
- **C** = warmup→clamped(`all` messages, W_back injects label)→free; **D** = warmup→free
  (`clamped_n_iter=0`). W_back at (1,2) is a right-going edge, so it fires only in the
  clamped phase.
- Two W_in variants: **frozen** (random projection, trainable J + W_out ≈ 9k params) and
  **trainwin** (sparse trainable W_in, ≈ 15.6k params). Both use sparse J.

## Method

1. **HP-tune** each variant to its own optimum (`tune.py`, Optuna 40 trials, objective =
   closed-form ridge probe_D on the 256 D-spins, 8-epoch/1-seed screen). Tuned:
   `lr_j, lr_win, j_d, threshold_j, threshold_win, strength_back, free_n_iter,
   clamped_n_iter`. Best configs → `replicate/tuned_fc_{frozen,trainwin}.json`.
2. **Phenomenon analysis** (`phenomenon.py`, 3 seeds × 20 epochs at the tuned config).
   Full-spin analysis on the 256 hidden spins (exp-5 protocol):
   - `probe_C`: L2-regularized Adam linear probe trained on C; acc on C_test
     (`probe_C_on_C`) and D_test (`probe_C_on_D`); `probe_D_on_D` = a probe trained on D.
   - Per test example (true `y`, strongest wrong class `k`): `importance_i=|W[i,y]−W[i,k]|`,
     `flip_i=1[C_i≠D_i]`, `damage_i=(C_i−D_i)(W[i,y]−W[i,k])`.
   - **Random-flip controls**: flip the same #spins per example at random (uniform /
     empirical per-spin rate) in C, re-evaluate the C-probe.
   - Field: `|field_C|` flipped vs stable.

## Tuned configs (probe_D, 8-ep/1-seed screen)

| variant | probe_D | lr_j | j_d | threshold_j | strength_back | free | clmp |
|---|---|---|---|---|---|---|---|
| trainwin | 0.381 | 1.03e-3 | 0.874 | 2.84 | 2.37 | 12 | 5 |
| frozen   | 0.348 | 2.36e-3 | 0.792 | 3.14 | 1.98 | 12 | 5 |

## Results (3 seeds × 20 epochs, tuned config)

| metric | frozen | trainwin |
|---|---|---|
| probe_C on C | 1.000 ± 0.000 | 1.000 ± 0.000 |
| probe_C on D | 0.153 ± 0.017 | 0.037 ± 0.003 |
| probe_D on D | 0.340 ± 0.005 | 0.383 ± 0.012 |
| **acc random-flip (uniform)** | 0.924 ± 0.006 | 0.631 ± 0.017 |
| acc random-flip (empirical) | 0.920 ± 0.008 | 0.621 ± 0.014 |
| overlap(C,D) | 0.771 ± 0.002 | 0.651 ± 0.002 |
| C→D flip rate | 0.229 ± 0.002 | 0.349 ± 0.002 |
| top-5% importance enrichment | 1.264 ± 0.018 | 1.309 ± 0.006 |
| corr(flip, damage) | 0.396 ± 0.001 | 0.390 ± 0.007 |
| corr(flip, importance) | 0.037 ± 0.003 | 0.064 ± 0.003 |
| \|field_C\| flipped / stable | 0.838 / 1.084 | 0.834 / 0.958 |
| head_D (own W_out) | 0.147 | 0.033 |

Flip rate by C-probe importance decile (1→10):
- frozen:   0.224, 0.223, 0.223, 0.216, 0.214, 0.209, 0.219, 0.228, 0.250, 0.280
- trainwin: 0.329, 0.336, 0.335, 0.323, 0.324, 0.330, 0.334, 0.354, 0.389, 0.439

Reference (conv model A, exp-5, best_channel_entropy): probe_C_on_C 0.973, probe_C_on_D
0.261, acc random-flip 0.972, flip rate 0.052, overlap 0.896, top-5% enrichment 1.60,
corr(flip,damage) 0.31, |field_C| flipped/stable 3.49/10.31.

## Files
- `../fc_common.py` — FC model, optimizer, rollout, spin collection, ridge probe.
- `../fc_start_cfg.json` — starting config for tuning.
- `tune.py` — per-variant Optuna tune → `replicate/tuned_fc_{frozen,trainwin}.json`.
- `phenomenon.py` — full-spin C/D analysis → `results/phenomenon{,_trainwin}.json`.
- `plot.py` → `figures/phenomenon.png` (regenerate from JSON).
- `results/study_fc_{frozen,trainwin}.json` — Optuna trial logs.
- `_launch_{tune,phen}.sh` — cluster tmux launchers.

Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. `models/` and
`results/smoke.json` are git-ignored; figures are globally git-ignored.
