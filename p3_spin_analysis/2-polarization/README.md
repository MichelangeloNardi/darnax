# Exp p3-2 — polarization / selectivity vs C→D flip danger (FC model)

Tests whether the "dangerous" hidden units (those that flip sign C→D and carry class info;
exp p3-1 / exp-5) can be identified from **polarization** descriptors.

## Per-unit descriptors (256 FC units), states C and D
- polarization `p_i = mean_n s_i` ∈ [−1, 1] (per unit, over the dataset)
- per-class polarization `p_i^c = mean_{n: y=c} s_i`
- selectivity `S_i = std_c(p_i^c)`

Computed on **D** (warmup→free, no label — inference state) and **C** (warmup→clamped→free).

## Per-unit danger quantities (correlated against, Pearson across the 256 units)
- `flip_rate_i` = mean_n 1[sign C ≠ sign D]
- `absfield_C_i` = mean_n |field at C|
- `importance_i` = mean_n |W[i,y] − W[i,k]|  (full-spin C-probe W; k = strongest wrong class)
- `damage_i` = mean_n (s^C_i − s^D_i)(W[i,y] − W[i,k])
- `wnorm_i` = ‖W[i,:]‖₂

## Hypothesis under test (not assumed)
Dangerous units have **low |p_i|** and **high S_i**, i.e. `corr(|p|, flip_rate) < 0`,
`corr(S, flip_rate) > 0`, `corr(|p|, absfield_C) > 0`, `corr(S, importance) > 0`,
`corr(S, damage) > 0`.

## Setup
- Model: FC asymmetric recurrent net (`fc_common.py`), tuned configs
  `replicate/tuned_fc_{frozen,trainwin}.json`. 256 hidden units. 3 seeds, 20 epochs.
- Reuses exp p3-1 `phenomenon.py` (`train_backbone`, `fit_probe`, `pearson`) and `fc_common`
  (`build_model`, `rollers`, `collect_spins`). Full-spin C-probe: Adam linear, wd 1e-3.
- `polarization.py --cfg <tuned config>` (train_win read from the config). `--smoke` for a
  tiny CPU run. `plot.py` → `figures/polarization.png`. Cluster:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python
  p3_spin_analysis/2-polarization/polarization.py --cfg replicate/tuned_fc_frozen.json`.
- `.gitignore`: `results/*_smoke.json`, `models/`.

## Results

_Pending run._
