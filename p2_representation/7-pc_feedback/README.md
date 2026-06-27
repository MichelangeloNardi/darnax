# Experiment 7 — predictive-coding error-driven clamped feedback

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

## Results (3 seeds, best_channel_entropy)

| model | C-probe | D-probe | C–D flip | overlap(C,D) | fullspin pC→D (actual) | pC→D **random** | enrich |
|---|---|---|---|---|---|---|---|
| **raw** (static clamp) | 0.964 | **0.453** | 0.047 | 0.907 | 0.258 | 0.956 | 1.59 |
| wback_b0.05 | 0.930 | 0.444 | 0.027 | 0.947 | 0.294 | 0.913 | 1.68 |
| wback_b0.1 | 0.999 | 0.441 | 0.097 | 0.807 | 0.241 | 0.999 | 1.55 |
| wback_b0.3 | 1.000 | 0.422 | 0.218 | 0.565 | 0.212 | 1.000 | 1.41 |
| wback_b1.0 | 1.000 | 0.383 | 0.388 | 0.225 | 0.126 | 0.983 | 1.16 |
| wout_b0.05 | 0.996 | 0.416 | 0.028 | 0.943 | 0.334 | 0.986 | 2.32 |
| wout_b0.1 | 0.998 | 0.411 | 0.048 | 0.905 | 0.293 | 0.995 | 2.22 |
| wout_b0.3 | 0.999 | 0.359 | 0.272 | 0.455 | 0.206 | 0.998 | 1.27 |
| wout_b1.0 | 1.000 | 0.329 | 0.459 | 0.082 | 0.132 | 0.879 | 1.00 |

(β=0.05 ≈ the raw-clamp field magnitude; β=1.0 ≈ 16×. fullspin: a C-probe trained on the
16384 raw spins, evaluated on D under the **actual** C→D flips vs **random** flips of the
same count — if actual ≪ random, the flips target the class-carrying spins.)

### The hypothesis is refuted (robustly, across the whole β range)

Error-driven feedback did the **opposite** of the goal on every axis:

1. **C gets *more* label-imprinted, not less.** Every PC variant at β≥0.1 has C-probe
   ≥0.996 → **1.000** (vs raw 0.964). The clamp decodes/projects the error through
   `W_back`/`W_out` — label-aligned directions — so it injects label-correlated drive
   *precisely onto the label-carrying spins* each step. "Self-limiting" (ε→0) doesn't
   de-imprint: by the time ε is small, C is already fully label-aligned, and the local
   rule then trains on an *even more* imprinted C. (Only wback_b0.05, weaker than the raw
   clamp, dips to 0.930 — that's just under-injection, and D doesn't benefit.)
2. **D never improves.** D-probe decreases monotonically with β and never beats raw 0.453
   (best PC = wback_b0.05 at 0.444 ≈ raw). No β reaches toward the ~0.51 BPTT ceiling.
3. **The flips still target the class-carrying spins.** The decisive random-flip control
   stays ≈ C-probe-on-C (green ≈ left bar) while the actual flips crater it — i.e. the
   exp-5 signature is *intact*, not removed. Strong β does drive C and D apart (flip↑,
   overlap↓ — superficially BPTT-like geometry), but the divergence is **destructive**:
   C stays fully imprinted *and* D gets worse. Only the most violent setting
   (wout_b1.0, 46% flips) finally goes importance-blind (enrich→1.0, random control drops
   to 0.88) — but it does so by shredding the representation (D-probe 0.329, the worst).

**Bottom line.** Closing the feedback loop through the label readouts reinforces the
artificial imprint rather than relaxing it; there is **no β where C de-imprints while D is
preserved**. To de-imprint C you likely need a feedback signal *not* aligned with the
label readout (BPTT reorganizes broadly, importance-blind, from random directions).
Natural next steps: **leaky error neurons** (the deferred α-variant) and/or a feedback
projector decoupled from `W_back`/`W_out`.

See `figures/pc_feedback.png` (regenerate from the JSONs via `plot.py`).
