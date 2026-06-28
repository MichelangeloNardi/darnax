# Experiment 11 — BPTT split diagnostic + partial-overlap split

Follow-up to exp 10 (strict disjoint split underperformed the scale-only control; in the
splits L/N channels probed at chance at D under the local rule). Two parts.

## Part 1 — BPTT split-routing diagnostic

Same split architectures `split_C24_I16_L8_N0` and `split_C32_I16_L8_N8` (from exp-10
`arch.py`). Train with the tanh-BPTT inference objective `CE(W_out · pool(D), y)` (the exp-1
ceiling method), enforcing the split wiring during BPTT: **W_in's gradient is masked to group
I** (non-I kernel columns start at 0 and never move), J1's `j_d` diagonal is frozen, W_out is
free. tanh surrogate (β 1→4), best backbone checkpointed by a C-aware hard-sign separability
proxy. The optimised kernels are written into a real split orchestrator and measured on the
**true hard-sign D**.

Reports (hard-sign): `probe_D`, `head_acc_D`, per-group probes (I / L / N), C/D overlap,
whether L/N become class-informative at D (= per-group `probe_D_group`).

Question: can real gradients make the L/N channels useful at D, given W_in only feeds I (so
L/N can receive input information only through the J1 recurrence)?

## Part 2 — partial-overlap split (local rule)

C=24 with three 8-channel groups: `both` (input + label), `input_only` (W_in only),
`label_only` (W_back only). W_in reaches `both`+`input_only` (16 channels = original input
capacity); W_back reaches `both`+`label_only` (16 channels). All channels participate in J1
and W_out. Trained with the standard local rule (exp-3 model-A recipe). Compared to
`standard_C24` and the strict `split_C24_I16_L8_N0` (exp-10 local-rule models, loaded from
`../10-split_scale/models`). Nominal trainable params equal across all C=24 configs (20016).

Reports the same global metrics + per-group probes (`both` / `input_only` / `label_only`).

## Main questions (per spec — recorded verbatim, not asserted here)
- Can BPTT make L/N channels useful in the strict split?
- Does partial overlap preserve label supervision while still reducing C→D mismatch?
- If L/N remain chance even under BPTT, strict split routing is probably not useful.

## Files
- `splitarch.py` — `build_explicit` (explicit, possibly-overlapping I/L channel sets),
  the partial-overlap config, C-aware hard-sign reps for the BPTT proxy. Reuses exp-10
  `arch.MaskedConv2D` / `arch.build_model` and `../bptt_common.py`.
- `bptt_train.py` — part 1: tanh-BPTT on the split architectures with W_in grad masked to I;
  serializes `models/<name>_bptt_seed<s>.eqx`.
- `train_partial.py` — part 2: local-rule training of `partial_C24`; serializes
  `models/partial_C24_seed<s>.eqx`.
- `diagnostics.py` — C-aware diagnostics with explicit channel groups; evaluates the part-1
  BPTT models, the part-2 partial model, and the exp-10 local references. Writes
  `results/diagnostics.json`.
- `plot.py` → `figures/global.png`. `.gitignore` — `models/`, `results/smoke.json`.

Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; `--smoke` on each for a tiny
CPU run. (Part-2 references require exp-10's `../10-split_scale/models/` to be present.)

## Results

_Pending cluster run._
