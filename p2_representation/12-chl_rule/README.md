# Experiment 12 — contrastive (CHL / EP-style) local rule

Motivated by exp 11: under BPTT the split L/N channels become class-informative at D, but the
gradient-free clamped-only local rule (`DynamicalTrainer`) leaves them at chance. The
contrastive rule is the local, gradient-free approximation of that BPTT credit assignment.

## Rule

Per batch, mirroring the working `DynamicalTrainer` plumbing (`orchestrator.step` forward/all +
`orchestrator.backward` + `make_optimizer`):

```
s0 = warmup (forward)                        # shared warmup
A  = s0 -> free   (forward)                   # free state (no label)
B  = s0 -> clamped (all, label via W_back)    # clamped/nudged state
grad = backward(B) - backward(A)              # contrastive difference
optimizer step (make_optimizer signed lrs)
```

The difference propagates the label signal through the recurrence to every channel (the signal
the clamped-only rule lacks). darnax's `ContrastiveHebbianTrainer` is **not** drop-in here (it
uses `filter_messages="left"` and omits the conv step's `t_win`/`t_back`), so the loop is
implemented in `run.py`.

Smoke (3 epochs, 30 batches, CPU): the default sign `backward(clamped) − backward(free)` trains
(probe_D 0.380 vs DynamicalTrainer 0.405); `--flip` is the opposite sign.

## Important: this is a FUNCTIONAL test, not the fair comparison

This run uses `best_channel_entropy_cfg` as a **starting point only**. That config was tuned for
the `DynamicalTrainer`; per the per-rule HP-tuning rule, a fair CHL-vs-baseline comparison needs
**CHL-specific HP tuning** (its phase lengths `clamped_n`/`free_n`, lr) as a follow-up. This
script answers: does the contrastive rule run, train, and move probe_D relative to the
clamped-only rule at the same config?

## Files
- `run.py` — CHL training loop + DynamicalTrainer baseline (same config/seed), per-epoch
  probe_D (Adam linear probe on pooled D) and head_acc_D. `--flip`, `--no-baseline`, `--smoke`.
- `plot.py` → `figures/curves.png` (probe_D / head_acc_D per epoch, CHL vs baseline).
- `.gitignore` — `models/`, `results/smoke.json`.

Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Results (3 seeds × 20 epochs, functional test at the baseline-tuned config)

| rule | best probe_D (mean) | final probe_D (mean) | head_D (mean) |
|---|---|---|---|
| CHL (contrastive) | 0.289 | 0.225 ± 0.023 | 0.107 |
| DynamicalTrainer (baseline) | 0.449 | 0.442 ± 0.003 | 0.252 |

Per seed (CHL best / final ; baseline final): seed 0 0.393 / 0.241 ; 0.440. seed 42 0.258 /
0.242 ; 0.446. seed 123 0.217 / 0.193 ; 0.440.

**Factual observations (numbers only):**
- At the baseline config, CHL's probe_D **peaks then decays** (best 0.289 mean vs final 0.225
  mean); the peak height is seed-dependent (0.393 / 0.258 / 0.217). The DynamicalTrainer
  baseline is stable (final 0.442 ± 0.003).
- This config is `best_channel_entropy_cfg`, tuned for the DynamicalTrainer (not for CHL). A
  CHL-specific HP tuning (its lr / phase lengths) is the follow-up (exp 13) before any fair
  CHL-vs-baseline comparison.

Figure: `figures/curves.png` (regenerate from the JSON via `plot.py`).
