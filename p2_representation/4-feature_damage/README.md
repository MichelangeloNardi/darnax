# Experiment 4 — pooled-feature damage, rescue & lesion (refined diagnostic #5)

Exp-3 diagnostic #5 (importance-vs-flip) was too coarse: the readout acts *after* 8×8
average pooling, so all 64 spins in a block share one `W_out` row and the per-spin
readout importance is block-constant. Here we work in the **256-dim pooled-feature
space** where the readout actually lives, using the serialized exp-3 models
(`../3-CD_diagnostics/models/`).

For each model (A local-rule, B BPTT CE_D, C BPTT+align) × 3 seeds:
1. Train a linear probe on **pooled C** → `W_probe_C` (256×10); importance = `‖W_probe_C[f,:]‖₂`.
2. Feature-level change `Δ[f] = pool(C)[f] − pool(D)[f]`; correlate with importance.
3. **Logit damage** `damage[ex,f] = W_probe_C[f, y] · Δ[ex,f]` — per-feature correct-class
   logit lost going C→D under the C-readout.
4. **Rescue**: restore the top-k most-damaged features of D to their C values; **Lesion**:
   corrupt those top-k features of C to D values. Both scored under `W_probe_C`, k swept
   0→256. (Top-k chosen per example by signed damage — uses the true label, so the curves
   are an *oracle* concentration bound, not a label-free operation.)

## Results (3 seeds)

| | A (local) | B (BPTT) | C (+align) |
|---|---|---|---|
| acc C (C-probe on C) | 0.977 | 0.712 | 0.750 |
| acc D (C-probe on D) | 0.248 | 0.225 | 0.268 |
| corr(importance, C→D change), feature-level | **+0.567** | −0.140 | −0.107 |
| corr(importance, C→D change), per (f,ex) | +0.397 | −0.006 | −0.001 |
| total correct-class logit damage | **6.80** | 2.40 | 1.84 |
| rescue acc @ k=32 / k=64 | 0.94 / 0.98 | 0.88 / 0.97 | 0.88 / 0.97 |
| lesion acc @ k=16 / k=32 | 0.75 / 0.52 | 0.10 / 0.02 | 0.15 / 0.05 |

## Findings

1. **The C-trained readout collapses on D** (acc 0.71–0.98 → ~0.25) for all models —
   the C-readout doesn't transfer to D (C and D are different states; cf. exp 3). Note
   D's *own* probe reaches ~0.50; this is specifically the C→D transfer failure.
2. **The damage is concentrated in a few pooled features.** Rescue: restoring just the
   top-**32** of 256 damaged features (to C) lifts the C-probe on D from ~0.25 to
   ~0.88–0.94; top-64 reaches/exceeds acc_C. Lesion: corrupting the top-**16–32**
   damaged features collapses C accuracy (B/C → <0.10). A small feature set carries the
   whole C↔D readout gap. (Oracle caveat: top-k uses labels.)
3. **For the local rule (A), the damaged features ARE the important ones** (corr
   **+0.57**) and the total logit damage is ~3× larger (6.8 vs 1.8–2.4): A's C-readout
   leans on the **label-imprinted** features that W_back injects into C and that vanish
   at D (A's C is ~98% label-decodable, exp 3). For BPTT (B/C) importance and C→D change
   are **decoupled** (corr ≈ 0) — their C is far less label-imprinted, so less of the
   readout rides on features that die at D.
4. **A's C-readout is more redundant/robust to lesioning** (lesion@16 still 0.75) than
   B/C (→0.10): A's 98%-decodable C spreads label info across more features, whereas
   B/C concentrate it, so removing the top-damaged few collapses them faster.
5. For BPTT, a **hybrid** (top-damaged features from C, rest from D) scores ~0.98 under
   the C-probe — above both acc_C and acc_D — but this is the label-aware oracle
   selection, so it bounds recoverability rather than being deployable.

## Files
- `feature_damage.py` — analysis (loads exp-3 models). `--smoke` for a tiny run.
- `plot.py` — regenerates `figures/feature_damage.png`.
- `results/feature_damage.json` — per (model,seed) scalars + rescue/lesion curves.

Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
