# Experiment 6 — C-basin diagnostics (local-rule model A)

Do the high-accuracy clamped states **C** form real attractor basins under the
hard-sign **free dynamics**, how class-structured are they, and how close is the
inference trajectory **D** to falling into the C basin? Model A (local rule), 3 seeds,
loaded from `../3-CD_diagnostics/models/`. Relaxation = iterate `s ← sign(W_in(x) +
J1(s))` for **30** free steps. Reference: acc_C 0.96, acc_D 0.27, overlap(C,D) 0.90.

## Results (3 seeds)

**1. C basin radius** (flip p% of spins, relax):

| p% | 0 | 1 | 2 | 5 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|
| overlap w/ C | 0.999 | 0.99 | 0.98 | 0.97 | 0.95 | 0.91 | 0.86 | 0.81 |
| return rate (ov≥0.95) | 1.0 | 1.0 | 1.0 | 1.0 | **0.51** | 0.00 | 0.00 | 0.00 |
| C-probe acc | 0.96 | 0.96 | 0.95 | 0.94 | 0.92 | 0.88 | 0.80 | 0.61 |

**2. C→D boundary** (apply k% of the real C→D flips, relax):

| k% | 0 | 10 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|
| overlap w/ C | 0.999 | 0.99 | 0.98 | 0.95 | 0.92 | 0.90 |
| overlap w/ D | 0.90 | 0.90 | 0.91 | 0.94 | 0.97 | 0.99 |
| returns to C (vs D) | 1.0 | 1.0 | 1.0 | **0.89** | 0.00 | 0.00 |

**3. D-trajectory rescue** (pin/replace top-k damaged spins to C, relax; final-D acc):

| k spins | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| pin, C-probe acc | 0.34 | 0.41 | 0.52 | 0.71 | 0.89 | **0.96** |
| replace, C-probe acc | 0.30 | 0.33 | 0.40 | 0.53 | 0.76 | 0.94 |

(pin overlap-with-C rises 0.90→0.96; rescuing earlier in the trajectory is marginally
better than later, e.g. pin k=128: t=1 0.75 → t=7 0.71.)

**4. C class geometry** (within / between prototype distance; nearest-class acc):

| space | within | between | proto acc |
|---|---|---|---|
| full Hamming | 0.387 | 0.384 | 0.58 |
| pooled L2 | 6.43 | 6.55 | 0.67 |
| probe-logit L2 | **3.67** | **7.98** | **0.98** |

## Findings

1. **C is a genuine but small attractor basin.** Perturbing ≤5% of spins reliably
   relaxes back (return rate ~1.0, overlap ≥0.97); the exact-return edge is ~10% (rate
   0.51); beyond 20% it never returns to C. *But* class info is robust far past the
   basin — C-probe acc is still 0.88 at 20% and 0.80 at 30% — so relaxation drifts to
   *nearby class-consistent* states, not to garbage.
2. **D lives across a separatrix ~60–70% along the C→D flip set.** Applying ≤50% of the
   real flips still relaxes back to C (returns-to-C 1.0 at 25%, 0.89 at 50%); ≥75% falls
   to D. So C and D are in **different basins**, separated by a *majority* of the
   class-carrying flips — D isn't a small perturbation of C's basin.
3. **C's class structure is a thin linear direction, not a spin-space clustering.**
   within ≈ between for Hamming (0.39/0.38) and pooled L2 (6.4/6.6) — C states are *not*
   geometrically clustered by class (per-instance dispersion; matches the attractor-
   geometry finding). Only in **probe-logit space** is there clean separation (within
   3.67 ≪ between 7.98, prototype accuracy **0.98**). The label identity is a low-dim
   readout projection, not a basin-level cluster.
4. **D is rescuable by anchoring a few hundred class-carrying spins.** Pinning the
   top-k damaged spins to C and relaxing recovers C-probe acc from 0.27 (D) toward 0.96
   (C): k=128 → 0.71, 256 → 0.89, **512 (3% of spins) → 0.96**. Replace-once is weaker
   (needs ~512 to reach 0.94). This complements exp 5 (C→D flips *target* the
   class-carrying spins): holding those same spins to C drags D back into the C basin.

**Bottom line:** C is a real attractor with a narrow exact-return radius (~5–10% of
spins) but a wide class-consistent neighborhood; D sits in a neighboring basin a
majority of flips away; class identity is a linear readout direction (not spin
geometry); and pinning ~3% of spins (the class-carriers) to C relaxes D fully back into
the C basin.

## Files
- `c_basin.py` — diagnostics (loads exp-3 model). `--model A|B|C`, `--smoke`.
- `plot.py` → `figures/c_basin_<model>.png`.
- `results/c_basin_A.json` — per-seed curves + aggregated geometry scalars.

Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. B/C can be run with
`--model B|C` for comparison (uses the same serialized exp-3 models).
