# 1 — Readouts on C vs D

**Question:** does fitting the readouts (linear probe + `W_out`) on the
clamped-consolidated representation **C** beat fitting them on the inference
representation **D**?

## ABCD

| State | How it's reached | Uses label? |
|-------|------------------|-------------|
| A | warmup (forward-only, `W_in` only) | no |
| B | A → clamped (`all` messages, `W_back` injects `y`) | **yes** |
| C | B → free (forward-only) — what the **online** rule trains on | yes (via B) |
| D | A → free (forward-only, skip clamped) = **inference** / `eval_step` | no |

The online `W_out` already trains on **C** but is evaluated on **D**; the linear
probe currently trains *and* tests on **D**.

## Constraint

C needs the label (clamped phase). At test time there are no labels, so test
reps can only be **D**. The valid comparison is **fit-on-C(train) → eval-on-D(test)**.

## What the script reports (per seed, per epoch)

- `head` — online `W_out` (C-trained), eval D — reference
- `probe_D` — Adam probe fit on D(train) → D(test) — baseline
- `probe_C` — Adam probe fit on C(train) → D(test) — **hypothesis**
- `probe_C_leaky` — Adam probe fit on C(train) → C(test) — ceiling only
  (uses test labels to *form* the rep → not a valid accuracy, diagnostic)

Final epoch, offline `W_out` (Win/J frozen, `W_out` re-init, perceptron rule):
- `wout_C` (clamped_n = cfg) vs `wout_D` (clamped_n = 0), both eval on D(test)

## Run (cluster)

```bash
~/miniforge3/envs/darnax_hpc/bin/python experiments/1-Probe_Wout_on_C_vs_D/probe_wout_c_vs_d.py
```

Config: `replicate/best_channel_entropy_cfg.json`. Defaults: 3 seeds × 10 epochs.
Outputs: `results/probe_wout_c_vs_d.json`, `figures/probe_wout_c_vs_d.png`.
