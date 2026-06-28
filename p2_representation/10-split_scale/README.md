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

## Results

_Pending cluster run._
