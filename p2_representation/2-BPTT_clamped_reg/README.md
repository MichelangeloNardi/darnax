# Experiment 2 — BPTT with a clamped-distance regularizer

**Question.** Exp 1 showed the plain BPTT ceiling on the inference state D is
~0.51. Here we ask whether *pulling D toward the clamped state C* (the label-injected
fixed point) during BPTT yields a better/more-separable D. C is the state the
dynamics reach when the label is fed through `W_back`; the idea is to distill that
label-shaped fixed point into the free inference dynamics.

**Loss.** Same differentiable-rollout diagnostic as exp 1, with an added distance
term (C is a **stop-gradient target** — only D is pulled toward C):

```
loss = CE(W_out · pool(<ce_state>), y) + alpha · MSE( <space>(D), stopgrad(<space>(C)) )
```

Four variants = {CE on **D** vs **C**} × {regularize **pooled** 256-dim vs **full**
32×32×16 spins}:

| variant | CE term | distance term |
|---|---|---|
| `ce_D_reg_pool` | CE(W_out·pool(D), y) | α‖pool(D) − sg(pool(C))‖² |
| `ce_C_reg_pool` | CE(W_out·pool(C), y) | α‖pool(D) − sg(pool(C))‖² |
| `ce_D_reg_full` | CE(W_out·pool(D), y) | α‖D − sg(C)‖² |
| `ce_C_reg_full` | CE(W_out·pool(C), y) | α‖D − sg(C)‖² |

The distance is a **per-element MSE** (mean over batch *and* features) so `alpha` is
comparable across the pooled and full-spin variants (raw summed ‖·‖² would differ
~64× between 256 and 16384 dims).

**Setup.** New differentiable rollout `diff_forward_CD` produces both states from one
input, **sharing the warmup phase**: D = warmup→free (forward messages), C =
warmup→clamped(+`W_back(y)`)→free. `W_back` is frozen (constant message, computed
once). tanh surrogate only (STE collapsed in exp 1), β annealed 1→4, `J1` `j_d`
frozen. Random init, best_channel_entropy cfg, 3 seeds, α ∈ {0.1, 0.3, 1.0}.

As in exp 1, the **ceiling is measured on the true hard-sign D dynamics** (perceptron
`W_out` + Adam probe on hard D reps), and the best backbone is checkpointed by the
per-epoch hard-sign separability proxy.

## Results

_(filled in on completion — see `results/clamped_reg.json` and `figures/clamped_reg.png`.)_

References: exp-1 plain BPTT (tanh) probe **0.506** / W_out 0.480; gradient-free ~0.46.

## Files
- `run.py` — sweep (4 variants × α × seeds). `--smoke` for a tiny CPU run; knobs
  `--alphas --bptt-epochs --beta-max --lr`.
- `plot.py` — regenerates `figures/clamped_reg.png` from `results/clamped_reg.json`.
- `results/clamped_reg.json` — per (variant, α): per-seed curves + means/stds.

Shared infra in `../bptt_common.py` (`diff_forward_CD`, `_reg_loss`,
`make_reg_bptt_step`, plus the exp-1 hard-sign measurement helpers). Run from repo
root: `XLA_PYTHON_CLIENT_PREALLOCATE=false python p2_representation/2-BPTT_clamped_reg/run.py`.
