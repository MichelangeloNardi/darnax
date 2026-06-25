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

## Results (3 seeds, probe accuracy on hard-sign D)

| variant | α=0.1 | α=0.3 | α=1.0 |
|---|---|---|---|
| `ce_D_reg_pool` | 0.504 ± 0.003 | 0.512 ± 0.001 | 0.495 ± 0.004 |
| `ce_C_reg_pool` | 0.395 ± 0.014 | 0.401 ± 0.002 | 0.397 ± 0.007 |
| `ce_D_reg_full` | 0.509 ± 0.004 | **0.516 ± 0.005** | 0.471 ± 0.003 |
| `ce_C_reg_full` | 0.393 ± 0.006 | 0.390 ± 0.005 | 0.399 ± 0.012 |

(W_out perceptron tracks ~2 pts below in every cell.) References: exp-1 plain BPTT
(tanh) probe **0.506** / W_out 0.480; gradient-free ~0.46.

**Findings:**
- **CE on D + a mild clamped pull helps a little.** Best is `ce_D_reg_full` α=0.3 →
  **0.516**, ~1 pt over plain BPTT (0.506) and ~5.5 pts over gradient-free. Pooled
  alignment (`ce_D_reg_pool` α=0.3) gives 0.512. The gain is small but consistent
  across seeds.
- **α matters and is mild.** α=0.3 > α=0.1 ≳ plain BPTT; α=1.0 over-constrains and
  *hurts* (0.471–0.495) — forcing D≈C too hard sacrifices the CE objective.
- **CE on C always collapses (~0.39–0.40).** The clamped state is trivially label-
  separable (the label is injected through W_back), so CE→0 almost immediately
  without ever pressuring D to be good; the distance term alone can't rescue D, and
  the checkpoint proxy bails out at early epochs. CE on C is the wrong objective.
- **Full-spin vs pooled alignment** behave almost identically for CE-on-D (full
  marginally better at α=0.3); the regularizer space is a second-order knob.

Takeaway: the clamped-distance regularizer is a modest improvement (~+1 pt over
plain BPTT) only in the CE-on-D + mild-α regime. The BPTT representation ceiling is
still ~0.51–0.52; pulling D toward C nudges it but does not unlock a new regime.

## Files
- `run.py` — sweep (4 variants × α × seeds). `--smoke` for a tiny CPU run; knobs
  `--alphas --bptt-epochs --beta-max --lr`.
- `plot.py` — regenerates `figures/clamped_reg.png` from `results/clamped_reg.json`.
- `results/clamped_reg.json` — per (variant, α): per-seed curves + means/stds.

Shared infra in `../bptt_common.py` (`diff_forward_CD`, `_reg_loss`,
`make_reg_bptt_step`, plus the exp-1 hard-sign measurement helpers). Run from repo
root: `XLA_PYTHON_CLIENT_PREALLOCATE=false python p2_representation/2-BPTT_clamped_reg/run.py`.
