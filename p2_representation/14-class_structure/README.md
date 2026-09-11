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
  `results/class_structure.json`. Reps cached in `reps/seed<N>_ep<E>.npz` (gitignored);
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

## Results (3 seeds x 20 epochs, w02, 11m45s)

**10-way, same closed-form ridge readout on every representation**

| representation | 10-way acc |
|---|---|
| C (clamped) | 0.9306 ± 0.0290 |
| D | 0.4508 ± 0.0063 |
| randD (untrained W_in/J1) | 0.4245 ± 0.0077 |
| pixels (3072-d) | 0.3718 |

Adam probe on D 0.4570 ± 0.0057; W_out head on D 0.2421 ± 0.0270.

**Error decomposition (10-way probe on D).** Errors landing inside the true class's
vehicle/animal group: 0.704 / 0.704 / 0.709 per seed. Rate expected under uniform
confusion: 0.481 / 0.480 / 0.481.

**Named dichotomies — balanced accuracy, mean ± std over seeds**

| grouping | D | C | randD | pixels | lift (D − best control) |
|---|---|---|---|---|---|
| `vehicle_vs_animal` | 0.8203 ± 0.0005 | 0.9568 | 0.8199 | 0.7932 | +0.0004 ± 0.0100 |
| `road_vs_skywater` (vehicles only) | 0.8103 ± 0.0059 | 0.9930 | 0.8048 | 0.7650 | +0.0055 ± 0.0107 |
| `mammal_vs_other_animal` (animals only) | 0.6445 ± 0.0069 | 0.8901 | 0.6205 | 0.6004 | +0.0240 ± 0.0111 |
| `flies_vs_not` | 0.5725 ± 0.0063 | 0.8599 | 0.5449 | 0.5465 | +0.0256 ± 0.0058 |

**4-way (centroid-suggested).** D 0.6419 ± 0.0038, randD 0.6289 ± 0.0057, pixels 0.5937.
Per-group on D: mammal 0.813, road_vehicle 0.668, sky_water_vehicle 0.652,
non_mammal_animal 0.264.

**511-partition scan, ranked by mean lift over the best control**

| dichotomy | lift |
|---|---|
| ship | +0.0473 ± 0.0027 |
| auto+ship | +0.0439 ± 0.0044 |
| airplane+auto | +0.0411 ± 0.0084 |
| airplane+auto+bird | +0.0351 ± 0.0103 |
| airplane+cat+frog | +0.0329 ± 0.0066 |
| … | |
| horse+truck | −0.0177 ± 0.0070 |
| auto+horse | −0.0217 ± 0.0063 |
| horse | −0.0228 ± 0.0051 |

Over all 511: mean lift +0.0127, median +0.0124, max +0.0473; 64.6 % have lift > 0.01.
`vehicle_vs_animal` ranks near the bottom of that distribution at +0.0004.

**Average-linkage cuts on the D class centroids (seed 0)**

| k | groups | one-vs-all acc |
|---|---|---|
| 2 | {airplane, ship} · {auto, bird, cat, deer, dog, frog, horse, truck} | 0.8772 |
| 3 | {airplane, ship} · {auto, truck} · {6 animals} | 0.7871 |
| 4 | {airplane, ship} · {auto} · {6 animals} · {truck} | 0.7521 |
| 5 | {airplane} · {auto} · {6 animals} · {ship} · {truck} | 0.7321 |

Figure: `figures/class_structure.png`. Raw JSON: `results/class_structure.json`.
