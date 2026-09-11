# Experiment 16 — associative augmentation (fair split), BPTT-screened

**Fairness fix over exps 9/10/15.** Those compared a split against a dense model of the same
*total* size, which gave the split *less* input/label fan-in — unfair. Here we instead **fix
the input fan-in at 16** (matching the classic C16 baseline) and *add* capacity:
- **I** — 16 input-only neurons (see `W_in`, not `W_back`) — same input capacity as classic C16
- **L** — a **distinct** label-only subset (see `W_back`, not `W_in`) — disjoint from I
- **N** — recurrent-only "associative middle" (see neither directly)
- all disjoint; `J1` recurrence and `W_out` read **all** neurons.

Baseline = **classic_c16** (16 neurons, each sees input AND label = the standard tuned model).
Question: on top of a full 16-input core, does adding a distinct label group + recurrent-only
associative neurons help?

**Weights.** vs a dense model of the same total M, this keeps `W_in` at the 16-input fan-in
(1200) instead of growing to `input·M`, while `J1` (~M²) and `W_out` (~M) grow the same — a
*cheaper* way to add recurrent capacity. Confirmed by `arch16.param_report`:
`W_in` 1200 for every config; `J1`/`W_out` grow with C (trainable 10144 → 88504 for N=0…32).

## Configs (fixed I=16, L=8 distinct; N swept)
| name | C | I | L | N |
|---|---|---|---|---|
| classic_c16 (baseline) | 16 | 16 | 16 (=all) | 0 |
| aug_L8_N0 | 24 | 16 | 8 | 0 |
| aug_L8_N8 | 32 | 16 | 8 | 8 |
| aug_L8_N16 | 40 | 16 | 8 | 16 |
| aug_L8_N32 | 56 | 16 | 8 | 32 |

## Plan
1. **BPTT ceiling screen** (`bptt_screen.py`): tanh-BPTT on `CE(W_out·pool(D), y)` with W_in's
   gradient masked to I (j_d frozen, W_out free), measure the true hard-sign probe_D. Does the
   ceiling rise as N grows, above classic_c16 (~0.51)? If not, the architecture doesn't benefit
   even with gradients → stop.
2. **Local rule** (only for configs the screen shows headroom): per-config HP tune + 3-seed
   final vs classic_c16, reusing exp-13/15 machinery.

Caveat (per prior findings): the BPTT–local gap is one data point and may vary by architecture;
BPTT is an upper-bound screen, not a guarantee for the local rule.

## Files
- `arch16.py` — config registry, builders (reuse exp-11 `splitarch.build_explicit`), win-mask,
  param report.
- `bptt_screen.py` — step 1 (reuses exp-11 `make_split_bptt_step` + exp-13 `measure_probeD`).
  `--name <config>`, `--seeds`, `--smoke`.
- `.gitignore` — `results/*_smoke.json`, `models/`.

Run one (cluster): `XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/16-associative_aug/bptt_screen.py --name aug_L8_N16`

## Results

_Pending BPTT screen._
