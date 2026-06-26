# Experiment 6 — predictive-coding error-driven clamped feedback

**Question.** Exp 3–5 showed the local rule lives in a **C≈D** regime where C is ~98%
label-imprinted by the static clamp `W_back(y)`, and its ~5% C→D flips are concentrated
on the label-carrying spins (random-flip control: 0.97 → actual 0.26). That imprint is
*artificial* — injected by an open-loop label field that keeps pushing even once the
state already encodes the class. **Can an error-driven (predictive-coding) clamp build a
C that is still class-informative but less artificially label-imprinted, so D reaches /
preserves it better?**

## Method — closed-loop feedback during the clamped phase

Replace the static label clamp with a feedback field that depends on the **current
hidden state** `s_t` (Mattia's formulation):

```
eps_t = y01 - softmax( decode(s_t) )        # y01 = (y+1)/2, one-hot {0,1}
b_t   = g * project(eps_t)                    # g = beta * sqrt(H*W) = 32*beta  (N=H*W=1024)
field = W_in x + J s_t + b_t                  # faithful: J1.reduce == sum
s_t+1 = sign(field)
```

As `s_t` learns to predict `y`, `eps_t → 0` and the field **self-limits** instead of
hard-imprinting the label. Only the clamped phase changes; warmup→free (= **D**, the
inference state) is untouched, and the perceptron/entropy local rules still train on the
reached **C**.

**Injection point.** The orchestrator's `step` builds the row-1 field from
`messages[sender]` that read only `state[sender]`, so the static `W_back(state[2])` path
can't see `s_t`. We instead run a custom clamped step (`pc_feedback.pc_clamped_step`) that
reproduces the faithful row-1 aggregation (`W_in·x + J·s_t`, `J1.reduce` is a sum) and
adds `b_t` before `sign` (via the orchestrator's own `_safe_activate`). `y` is read from
`state[2]` (set by `state.init`), so PC rollers keep the `(orch, state, key)` signature
and are **drop-in** for the exp-3 `diagnose()` / exp-5 `analyze()` C-roller slot.

**Two instant variants** (leaky error neurons deferred to a follow-up):
- **`wback`** (tied): `decode = mean_HW(s) @ W_back.Wᵀ` (16→10; mean is the normalized
  adjoint of the spatial broadcast), `project = eps @ W_back.W` broadcast over H×W. Uses
  the **raw** `W_back` matrix (skips ChannelWBack's ±1 `(a,b)` rescaling, which is for
  labels not errors).
- **`wout`**: `decode = pool8×8(s) @ W_out.W` (256→10), `project = unpool(eps @ W_out.Wᵀ)`
  — each pooled feature's feedback spread uniformly over its 8×8 block. Uses the **online**
  `W_out` (a second, co-evolving closed loop).

**Not backprop.** `b_t` is forward + transpose ops of existing frozen/online weights,
local in time, gradient-free. Unlike the exp 1–2 BPTT *ceiling*, these are **deployable
darnax local-rule variants** — a real candidate rule, not just an upper bound.

`g = β·√(H·W) = 32β`. β is swept `{0.1, 0.3, 1.0}`; `train_models.py` logs the t=0 RMS of
`b_t` vs the raw clamp (`strength_back=1.47`) for calibration, and selects the best β per
variant by a quick hard-sign D-ridge proxy.

## Files
- `pc_feedback.py` — the PC clamped step, rollers, custom train loop, calibration helper.
- `train_models.py` — trains raw baseline + `wback`/`wout` × β grid × 3 seeds →
  `models/<tag>_seed<s>.eqx` + `select.json` (best β per variant by D-proxy).
- `diagnostics.py` — reuses exp-3 `diagnose()` with PC C-rollers: C/D probe, C–D flip /
  overlap, C-free stability, fields.
- `fullspin_importance.py` — reuses exp-5 `analyze()`: the decisive random-flip control
  (does C→D flip on class-carrying spins?).
- `plot.py` → `figures/pc_feedback.png`. `--smoke` on each for a tiny CPU run.

`models/` and `figures/` are git-ignored. Run from repo root with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; `diagnostics.py` / `fullspin_importance.py` default
to the baseline + selected-β models (`--all-betas` for the full grid).

## Results

_(pending first cluster run)_

**Hypothesis / what success looks like:** vs the raw clamp (C-probe 0.98, D-probe 0.45,
flip 0.05 targeting class spins), a good PC β gives a C that is **less label-imprinted**
(C-probe below 0.98 but not collapsed), C→D flips **importance-blind** (random-flip
control ≈ actual flips, BPTT-like), higher overlap(C,D) / C-free stability, and ideally
D-probe lifted toward the ~0.51 BPTT ceiling.
