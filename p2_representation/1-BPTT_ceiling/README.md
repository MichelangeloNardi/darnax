# Experiment 1 — BPTT ceiling on the inference state D

**Question.** Problem 1 showed the readout plateaus at the ~0.46 ceiling of the
inference state D. How high *could* D go if the hidden recurrent representation
(`W_in` / `J1`) were optimised with real gradients instead of the gradient-free
local rule? This is a **diagnostic upper bound**, explicitly not darnax-faithful.

**Method.** Backprop-through-time a differentiable rollout to D
(warmup→free, forward messages only — exactly `orch.step(filter_messages="forward")`,
so no `W_back`, no `W_out` feedback), with the non-differentiable `jnp.sign`
activation replaced by a surrogate. Optimise `{W_in.kernel, J1.kernel, W_out.W}`
with `jax.grad` + Adam to minimise `CE(W_out·pool(D), y)`. `J1`'s `j_d` diagonal is
kept frozen (gradient masked by `update_mask`). Random init, best_channel_entropy
cfg, 3 seeds.

The **ceiling is always measured on the true hard-sign dynamics**: after BPTT we
rebuild a real `SequentialOrchestrator` with the optimised kernels, collect D reps
with the actual `jnp.sign` rollout, and fit a perceptron `W_out` + an Adam linear
probe on them (eval on the CIFAR test set).

Two surrogates as a bracket:
- **tanh**: `tanh(β·x)`, β annealed 1→4 (capped — ramping to a hard sign
  destabilises the rollout).
- **STE**: straight-through estimator (hard sign forward, identity grad).

Because annealing β over-sharpens late epochs, we **checkpoint the best backbone**
by a per-epoch hard-sign separability proxy (`sep`: true sign dynamics → pooled D →
closed-form ridge readout on a held-out *train* subset) and measure the ceiling on
that checkpoint, not the final weights.

## Results (3 seeds, eval on D, hard sign)

| | W_out (perceptron) | probe (Adam) | best ckpt |
|---|---|---|---|
| **BPTT (tanh)** | **0.480 ± 0.002** | **0.506 ± 0.003** | ep 13–14 |
| BPTT (STE) | 0.247 ± 0.016 | 0.272 ± 0.012 | ep 1–3 |
| gradient-free (p1) | ~0.44 | ~0.46 | — |

- **tanh BPTT clears the gradient-free ceiling by ~4–5 pts** (probe 0.51 vs 0.46),
  reproducible across all 3 seeds → there is real headroom in D's representation;
  the local rule sits ~5 pts below what D can support. This ~0.51 is the **target
  ceiling for an improved learning rule**.
- The absolute number stays ~0.5 because the *architecture* is the cap (1 conv +
  16-ch binary recurrent state + 8×8 pool + linear readout), not the learning rule.
- **STE is not a usable surrogate here**: it degrades the backbone from epoch 1
  (best ckpt ep 1–3) down to ~0.27, *below* the ~0.40 random-init level — the
  identity gradient through 7 hard-sign steps is too crude. The tanh soft-relaxation
  arm is the trustworthy ceiling.
- Both `sep` and soft-acc peak at β≈1.5–2.5 and decline toward β=4 (see right panel),
  confirming the over-sharpening; checkpointing absorbs it.

## Files
- `run.py` — BPTT optimisation + hard-sign ceiling measurement. `--smoke` for a
  tiny CPU run. Knobs: `--bptt-epochs --beta-max --lr --proxy-batches`.
- `plot.py` — regenerates `figures/ceiling.png` from `results/ceiling.json`.
- `results/ceiling.json` — per-seed curves (`sep_curve`, `soft_acc_curve`,
  `wout_curve`, `probe_curve`, `best_epoch`) + means/stds.
- `results/run.log` — full training log.
- `figures/ceiling.png` — left: ceiling bars vs gradient-free; right: tanh
  checkpoint-selection diagnostic.

Shared infra in `../bptt_common.py` (differentiable rollout, surrogates, hard-reps
+ ridge proxy, offline readout fits). Run from repo root:
`XLA_PYTHON_CLIENT_PREALLOCATE=false python p2_representation/1-BPTT_ceiling/run.py`.
