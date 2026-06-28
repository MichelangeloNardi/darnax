# Experiment 10 — split routing at variable channel count C

**Goal (per spec).** Test whether split input/label routing helps once the original input
capacity (|I| = 16 input-driven channels) is preserved, by scaling total channels instead
of holding C = 16 fixed (exp 9 held C = 16, which reduced input capacity in the splits).

## Configs (3 seeds each)

Masking, not shrinking: tensors have the standard shapes for each C; external inputs are
routed only to their assigned groups. `MaskedConv2D` confines W_in to group I and masks its
Hebbian updates (the j_d-diagonal pattern); the frozen W_back is confined to group L by
zeroing its columns. J1 (full groups=1 recurrence over all C) and W_out (reads all C pooled)
are unchanged. Contiguous channel ranges I = `[0:|I|]`, L = `[|I|:|I|+|L|]`, N = rest.

| name | C | I | L | N | role |
|---|---|---|---|---|---|
| `baseline_C16` | 16 | 16 | 16 | 0 | current baseline (all channels get input + label) |
| `standard_C24` | 24 | 24 | 24 | 0 | scale-only control |
| `split_C24_I16_L8_N0` | 24 | 16 | 8 | 0 | disjoint I/L, input capacity preserved |
| `split_C32_I16_L8_N8` | 32 | 16 | 8 | 8 | disjoint, N = 8 (neither directly) |
| `standard_C32` | 32 | 32 | 32 | 0 | scale-only control for split_C32 |

`standard`/`baseline` configs have |I| = |L| = C (masks all-ones → `MaskedConv2D` is a
bit-identical no-op → the standard model at that C). Nominal trainable params (W_in.kernel +
J1 − diagonal + W_out.W) match **within each C** (asserted): C16 = 10144, C24 = 20016,
C32 = 33088. Effective active params (W_in: 5·5·3·|I|; W_back frozen: 10·|L|) are logged per
config; the splits keep W_in effective = 1200 (|I| = 16), the same input capacity as baseline.

## Metrics

**Global** (per config, 3 seeds): `probe_C`, `probe_D` (pooled Adam probe), `head_acc_D`
(model's own W_out on D), `probe_C_transfer_D` (probe trained on C, evaluated on D),
`flip_rate` and `overlap_CD` (C↔D), `fullspin_pC_on_C` / `fullspin_pC_on_D` (raw 32·32·C
C-probe on C / on D = after the actual C→D flips), `rand_flip_acc` (same C-probe after
flipping the same number of spins at random).

**Per-group (I / L / N):** `probe_C_group` / `probe_D_group` (pooled probe restricted to the
group's channels — `probe_D_group` answers "does the group decode class at D"),
`flip_rate_group`, field margin `margin_flipped` / `margin_stable` (C·field_C at C),
`absfield_D_group` (mean |field| on the group's channels at D — at D, W_in is masked off L/N
and there is no label, so this is the J recurrence drive), `fieldD_margin_group`
(mean(field_D · D) on the group's channels at D).

### Interpretation criteria (per spec — recorded verbatim, not asserted here)
- If `split_C24` beats `baseline_C16` but not `standard_C24`, the gain is just scale.
- If `split_C24` beats or matches `standard_C24` with better C→D behavior, routing helps.
- If L/N remain chance at D, recurrence is still failing to transfer input information into
  the label/associative channels.

## Files
- `arch.py` — variable-C `MaskedConv2D`, `build_model`, config registry, C-aware `pool`,
  param accounting.
- `train_models.py` — trains all configs × 3 seeds (exp-3 model-A recipe: DynamicalTrainer,
  `kernel_decay_rate`, 20 epochs), asserts masks intact, serializes `models/<name>_seed<s>.eqx`.
- `diagnostics.py` — C-aware global + per-group diagnostics (self-contained; reuses only the
  C-agnostic `bc.offline_probe`). Writes `results/diagnostics.json`.
- `plot.py` → `figures/global.png`, `figures/groups.png`.
- `.gitignore` — `models/`, `results/smoke.json` (figures globally git-ignored).

Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; `--smoke` on each for a tiny
CPU run.

## Results (3 seeds, best_channel_entropy; means)

**Global**

| config | C/I/L/N | probe_C | probe_D | head_D | C→D transfer | flip | overlap | fs pC→C | fs pC→D | rand-flip |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_C16 | 16/16/16/0 | 0.962 | 0.440 | 0.247 | 0.319 | 0.051 | 0.898 | 0.963 | 0.259 | 0.960 |
| standard_C24 | 24/24/24/0 | 0.965 | 0.464 | 0.278 | 0.367 | 0.035 | 0.930 | 0.971 | 0.302 | 0.970 |
| split_C24_I16_L8_N0 | 24/16/8/0 | 0.407 | 0.408 | 0.279 | 0.407 | 0.000 | 1.000 | 0.353 | 0.353 | 0.353 |
| split_C32_I16_L8_N8 | 32/16/8/8 | 0.430 | 0.427 | 0.371 | 0.428 | 0.000 | 1.000 | 0.381 | 0.382 | 0.381 |
| standard_C32 | 32/32/32/0 | 0.972 | 0.473 | 0.284 | 0.366 | 0.032 | 0.935 | 0.966 | 0.303 | 0.964 |

(fs = full-spin C-probe; "fs pC→D" = after the actual C→D flips; "rand-flip" = after the same
number of random flips. For the split configs flip = 0.000, so fs pC→D = rand-flip = fs pC→C.)

**Per-group (I / L / N), means:** probe_C / probe_D / flip / |field|@D / fieldD·D@D

| config | I | L | N |
|---|---|---|---|
| baseline_C16 | 0.962 / 0.440 / 0.051 / 10.1 / 10.1 | 0.962 / 0.440 / 0.051 / 10.1 / 10.1 | – |
| standard_C24 | 0.965 / 0.463 / 0.035 / 11.4 / 11.4 | 0.964 / 0.464 / 0.035 / 11.4 / 11.4 | – |
| split_C24_I16_L8_N0 | 0.406 / 0.410 / 0.000 / 10.8 / 10.8 | 0.100 / 0.100 / 0.000 / 11.9 / 11.9 | – |
| split_C32_I16_L8_N8 | 0.432 / 0.430 / 0.000 / 15.0 / 15.0 | 0.100 / 0.100 / 0.000 / 14.6 / 14.6 | 0.100 / 0.100 / 0.000 / 16.3 / 16.3 |
| standard_C32 | 0.973 / 0.474 / 0.032 / 12.3 / 12.3 | 0.972 / 0.473 / 0.032 / 12.3 / 12.3 | – |

**Factual observations (numbers only):**
- `baseline_C16` matches exp-3/5 model A (probe_C 0.96, probe_D 0.44, flip 0.05, fs pC→C 0.96 /
  pC→D 0.26 / rand 0.96).
- probe_D: baseline_C16 0.440; standard_C24 0.464; standard_C32 0.473; split_C24 0.408;
  split_C32 0.427.
- split vs scale-only at equal C: probe_D split_C24 0.408 vs standard_C24 0.464; split_C32
  0.427 vs standard_C32 0.473.
- head_acc_D: split_C32 0.371; standard_C32 0.284; standard_C24 0.278; split_C24 0.279;
  baseline_C16 0.247. probe_C_transfer_D: split_C32 0.428, split_C24 0.407, standard_C24 0.367,
  standard_C32 0.366, baseline_C16 0.319.
- All split configs: C→D flip 0.000, overlap 1.000, probe_C ≈ probe_D (0.41–0.43); the
  full-spin actual-flip and random-flip accuracies coincide (no flips). standard/baseline
  configs: flip 0.03–0.05, overlap 0.90–0.94, probe_C ≈ 0.96–0.97 ≫ probe_D ≈ 0.44–0.47.
- Per-group probe in the split configs: group I probe_C/probe_D 0.41–0.43; groups L and N
  probe_C/probe_D 0.100. In standard/baseline configs every channel-group probe ≈ 0.96 (C) /
  0.44–0.47 (D). |field|@D is similar across groups within each config (10–16).

Figures: `figures/global.png`, `figures/groups.png` (regenerate from the JSON via `plot.py`).
