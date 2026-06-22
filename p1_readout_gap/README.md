# Problem 1 — the W_out ↔ probe readout gap

Experiments on **why the model's own head (W_out, perceptron rule) underperforms a
linear probe (Adam)**, and how to close it.

**Conclusion:** the gap is mostly a **train/inference state mismatch**, not the rule.
The online W_out trains on the clamped state **C** (label-contaminated when
`strength_back` is high) but is evaluated on the inference state **D**. Train/fit the
readout on **D** (or use a weak `W_back`) and W_out reaches ~0.42–0.44, within ~2 pts
of the Adam probe (~0.46). The remaining ~2 pts is the only true rule penalty. The
~0.46 representation ceiling is **problem 2** (see `../p2_representation/`).

## Layout
- `common.py` — shared building blocks (model, optimizer, dataset, trainer, rollouts,
  rep collection, offline W_out). **Reused by problem 2 too** (`sys.path` → this folder).
- `1-Probe_Wout_on_C_vs_D/` — probe fit on C vs D; ABCD diagnostics.
- `2-Wout_on_C_vs_D/` — W_out fit on C vs D, across configs.
- `3-Wout_C_vs_D_random/` — same, trained vs random backbone.
- `4-Probe_vs_Wout_full/` — full 8-cell grid {W_out,probe}×{online,offline}×{C,D}, eval on D and C.

Each experiment: `python p1_readout_gap/<n>-.../run.py` then `plot.py`. Configs in `../replicate/`.
