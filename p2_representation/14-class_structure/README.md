# Experiment 14 — class structure of the inference state D

`old_experiments/experiments4/attractor_geometry.py` measured the **geometry** of the
attractor states: per-instance within-class cosine 0.067 ± 0.185 (≈ between-class 0.019),
but class **centroids** with structure — animals mutually positive 0.4–0.9
(cat↔frog 0.88, deer↔frog 0.89), auto↔truck 0.61, airplane↔ship 0.72,
airplane↔cat −0.60, ship↔dog −0.45. No accuracy was attached to that geometry.

This experiment attaches accuracy: how well does D support **coarse** (superclass)
discrimination compared with the 10-way task, and which groupings of the 10 CIFAR-10
classes does it encode?

## Setup

- Architecture / config: standard `channel_entropy` C=16, `best_channel_entropy_cfg.json`,
  `DynamicalTrainer` (perceptron + entropy rules), 20 epochs, 3 seeds (0, 42, 123).
- States: **D** = warmup → free (`clamped_n_iter=0`), **C** = warmup → clamped → free.
  Reps are the 8×8-pooled J1 activation (256-d), collected with `cm.collect_reps` in a
  single pass over train and test (`bc.collect_D` / local `collect_C`).
- Readouts:
  - 10-way Adam linear probe (`bc.offline_probe`, 20 epochs, `cm.PROBE_WD`) → per-class
    accuracy, 10×10 confusion;
  - the model's own `W_out` head → same, for contrast;
  - closed-form ridge (λ = 1.0, bias column) for every binary partition.

## Measurements

1. **Per-class accuracy + confusion** on D, probe and head.
2. **Error decomposition.** Of the 10-way errors, the fraction landing inside the true
   class's group vs crossing the boundary, compared against the rate expected under
   uniform confusion — `(|group|−1)/9` per example.
3. **Named groupings** (binary ridge, balanced accuracy):
   | name | +1 side | scored on |
   |---|---|---|
   | `vehicle_vs_animal` | airplane auto ship truck | all 10 |
   | `road_vs_skywater` | auto truck | vehicles only |
   | `mammal_vs_other_animal` | cat deer dog horse | animals only |
   | `flies_vs_not` | airplane bird | all 10 |
   Plus the 4-way `FOURWAY` grouping (road / sky-water / mammal / non-mammal) suggested
   by the centroid heatmap, scored one-vs-all.
4. **Exhaustive 511-partition scan.** Every non-trivial dichotomy of the 10 classes
   (deduplicated by complement), ranked by balanced accuracy. A partition target is
   constant within a class, so with `U[:,c]` the sum of training reps of class c and
   `A = XᵀX + λI`, the ridge solution for a sign vector `s ∈ {−1,+1}¹⁰` is `A⁻¹U s` —
   all 511 fits are signed sums of 10 precomputed vectors, two matmuls total.
5. **Average-linkage clustering** of the 10 class centroids in D (cosine distance,
   implemented in-file; no scipy/sklearn dependency), cut at k = 2…5, each cut scored
   one-vs-all.

All of 1–5 are also computed on **C**.

## Controls

Every partition score is reported alongside two controls, since a dichotomy is only
informative about the representation if D decodes it better than these do:

- **`pixels`** — ridge on the raw 3072-d image (`x·2−1`);
- **`randD`** — the same D rollout through an **untrained** W_in / J1 (random init),
  256-d, i.e. dimension-matched to D. Same idea as `replicate/random/`.

Console output and panel (d) report `lift = balanced(D) − max(balanced(randD), balanced(pixels))`.

## Files

- `run.py` — trains, collects C / D / random-init reps, runs every analysis, writes
  `results/class_structure.json`. Reps cached in `reps/seed<N>.npz` (gitignored);
  `--reuse-reps` skips training and re-runs the analysis only.
- `plot.py` — `figures/class_structure.png`, four panels: (a) per-class accuracy,
  (b) confusion reordered vehicles-first, (c) centroid cosine similarity,
  (d) the 511-partition scan against the controls.
- `.gitignore` — `reps/`, `results/*_smoke.json`.

Run (cluster):
```
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/14-class_structure/run.py
python p2_representation/14-class_structure/plot.py
```
Smoke: `python p2_representation/14-class_structure/run.py --smoke` (1 epoch, 6 batches,
1 seed — trains essentially nothing; checks the pipeline only).

## Results

Pending — not yet run at the full budget.
