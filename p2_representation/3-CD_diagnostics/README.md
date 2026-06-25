# Experiment 3 — C/D diagnostics across three models

Compare the clamped state **C** (warmup→clamped+W_back→free) and the inference
state **D** (warmup→free) for three trained models, to understand *how* BPTT and the
alignment regularizer reshape the dynamics — not just the final accuracy.

- **A** standard local-rule (DynamicalTrainer; perceptron + entropy rules)
- **B** BPTT CE_D (exp 1; gradients into W_in/J1/W_out, no regularizer)
- **C** BPTT CE_D + α·align(D,C) (exp 2 `ce_D_reg_pool`, α=0.3)

`train_models.py` (re)trains and **serializes** the three models per seed (exp 1/2
saved only scalar curves, never the weights — so 3 short trainings, not the sweep).
`diagnostics.py` then loads them and computes everything **offline** (no training);
re-runnable in ~2 min. The orchestrator stores the pre-activation **field**
(`state.fields[1]`), so margins / field-projections are read directly.

## Results (3 seeds, best_channel_entropy)

| diagnostic | A (local) | B (BPTT) | C (+align) |
|---|---|---|---|
| 1. C probe acc | 0.977 | 0.711 | 0.750 |
| 2. D probe acc | 0.451 | 0.498 | 0.497 |
| 3. C–D flip rate | 0.052 | 0.236 | 0.219 |
| 4. overlap(C,D) | 0.896 | 0.528 | 0.563 |
| 6. margin flipped / stable | 3.49 / 10.31 | 4.08 / 6.12 | 3.34 / 5.28 |
| 5. corr(readout-imp, flip) | 0.014 | −0.003 | 0.002 |
| 5. corr(\|field\|, flip) | −0.196 | −0.234 | −0.249 |
| 8. \|field\| flipped vs random | 3.49 / 9.86 | 4.08 / 5.61 | 3.34 / 4.84 |
| 8. corr(\|field\|,flip) random | −0.003 | −0.005 | −0.003 |
| 9. C-free overlap / flip rate | 0.999 / 0.000 | 0.596 / 0.202 | 0.705 / 0.147 |

(probe = Adam linear probe; margins/fields are at C; see `figures/diagnostics.png`.)

## Findings

1. **Local rule lives in a C≈D regime; BPTT does not.** For A, clamped≈free
   (overlap 0.90, 5% flips) and **C is a near-perfect fixed point of the free
   dynamics** (overlap 0.999, 0% flips): removing the label leaves the state put.
   BPTT B/C reach a *more separable D* (0.50 vs 0.45) but C and D genuinely diverge
   (overlap ~0.53; C is unstable under free relaxation, ~15–20% flips).
2. **A's C is almost perfectly label-imprinted (0.98)**; BPTT's C much less (0.71–
   0.75) — B/C never trained with W_back, so their clamped state isn't tuned to the
   injection.
3. **Flips are dynamically, not task, selective.** flip↔readout-importance ≈ 0 for
   all models; flip↔|field| strongly negative (−0.20…−0.25): the spins that flip are
   the **weakly-pinned, low-field** ones. Random-flip control confirms it — really-
   flipped spins have far lower |field| than random (A 3.5 vs 9.9) and the
   correlation vanishes under random flips.
4. **The alignment regularizer does what it says, modestly.** C vs B: higher C–D
   overlap (0.563 vs 0.528), more C-free stability (0.705 vs 0.596), fewer flips — D
   is pulled toward C — but **D separability is unchanged** (0.497 vs 0.498).
   Aligning the states makes them mutually consistent without making D better
   (matches exp 2's "modest knob").

## Files
- `train_models.py` — trains + serializes A/B/C per seed → `models/<name>_seed<s>.eqx`.
- `diagnostics.py` — loads models, computes the 9 diagnostics → `results/diagnostics.json`.
- `plot.py` — regenerates `figures/diagnostics.png`.

`models/` is git-ignored (regenerate with `train_models.py`). Caveat: diagnostics #6
and #7 coincide in this implementation (both `C·field_C`); a distinct #7
(`field_D·C` on flipped spins) can be added cheaply since the models are serialized.
Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
