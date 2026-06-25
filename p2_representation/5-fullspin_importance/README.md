# Experiment 5 — Matei-style full-spin importance

**Question.** Are the individual C→D spin flips concentrated on the full-spin features
that actually carry C's class information? (The pooled `W_out` importance is block-
constant under 8×8 pooling, so exp 3/4 couldn't resolve this per spin.)

**Method.** For A/B/C (serialized exp-3 models) × 3 seeds: collect hard-sign C and D
hidden states (N,32,32,16) flattened to 16384 spins. Train a **regularized linear
probe on full-spin C** (no pooling) → `W` (16384×10); report acc on C_test
(`probe_C_on_C`) and D_test (`probe_C_on_D`). Per test example (true label `y`,
strongest wrong class `k = argmax_{c≠y}(C@W)_c`):
`importance_i = |W[i,y]−W[i,k]|`, `flip_i = 1[C_i≠D_i]`, `damage_i = (C_i−D_i)(W[i,y]−W[i,k])`.

## Results (3 seeds, best_channel_entropy)

| metric | A (local) | B (BPTT) | C (+align) |
|---|---|---|---|
| probe_C on C | 0.973 | 0.734 | 0.801 |
| probe_C on D (actual flips) | 0.261 | 0.222 | 0.245 |
| **C, uniform random flips** | **0.972** | 0.570 | 0.662 |
| **C, empirical-rate flips** | 0.962 | 0.572 | 0.653 |
| flip rate (overall) | 0.052 | 0.236 | 0.219 |
| corr(flip, importance) | 0.041 | 0.008 | 0.009 |
| **corr(flip, damage)** | **0.310** | 0.015 | 0.025 |
| top-5% importance enrichment | **1.60** | 1.05 | 1.06 |
| flip rate decile-1 → decile-10 | 0.044 → **0.072** | 0.233 → 0.244 | 0.216 → 0.228 |
| \|field_C\| flipped / stable | 3.49 / 10.31 | 4.08 / 6.12 | 3.34 / 5.28 |

## Findings

**The local rule (A): YES — flips are concentrated on the class-carrying spins.**
- The decisive control: flipping the *same number* of spins **at random** (uniform or
  empirical-per-spin-rate) leaves the C-probe at **0.97**, but the *actual* C→D flips
  crater it to **0.26**. A's ~5% flips are placed where they destroy the C-readout.
- Corroborated per-spin: `corr(flip, damage)=0.31`, top-5% important spins are **1.6×**
  enriched in flips, and flip rate rises monotonically across importance deciles
  (4.4%→7.2%). This matches Matei's summary: flips sit on C's class-carrying features.
- Why: A's C is ~98% label-imprinted (W_back), so its class info concentrates on spins
  that are exactly the ones that move going to the unlabeled D.

**The BPTT models (B/C): NO — flips are importance-blind / structural.**
- `corr(flip, importance)≈0.01`, `corr(flip, damage)≈0.02`, enrichment ≈1.05, flat
  deciles. Per-spin importance does not predict flipping.
- Their actual D still damages the C-probe more than random (0.22 vs 0.57 for B) — but
  because ~24% of spins flip in a *coordinated* way that moves the rep off the C-probe's
  manifold, not because the flips target individually class-important spins. Their C is
  far less label-imprinted, so class info isn't concentrated on the spins that flip.

**Common to all:** flips hit **weakly-pinned** spins — `|field_C|` is far smaller on
flipped (3.3–4.1) than stable (5.3–10.3) spins. (`fieldC·C` equals `|field_C|` here
because C is a sign fixed point, `C=sign(field_C)`.)

**Bottom line:** Matei's "flips land on the class-carrying spins" picture holds for the
**local-rule** model but **not** for the BPTT models — a mechanistic distinction
between how the two regimes reach D.

## Files
- `fullspin_importance.py` — analysis (loads exp-3 models, trains 16384-dim C-probes).
- `plot.py` → `figures/fullspin_importance.png`.
- `results/fullspin_importance.json` — per (model,seed) scalars + decile curves.

Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. (Rescue/lesion-style
controls here use the true label only via the wrong-class `k`; the random-flip
controls are label-free.)
