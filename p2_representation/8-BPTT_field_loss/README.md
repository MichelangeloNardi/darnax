# Experiment 8 — BPTT teacher-field loss (can we enlarge the C basin to lift D?)

Exp 6 showed that *pinning* teacher A's class-carrying C_A spins to C rescues its
inference state D (acc 0.27→0.96). This experiment asks whether BPTT can **bake that
into the dynamics**: finetune A so the student's input-only field
`h_t = W_in·x + J1·s_t` *supports* C_A on the top-k class-carrying spins, with no
pinning at test time — does D then fall into the high-accuracy C basin?

**Setup.** Teacher = frozen model A (exp-3 `models/A_seed*.eqx`). Student **initialized
from A** (so the per-spin target C_A[i] is coordinate-aligned), BPTT, tanh surrogate
(β 1→4), j_d frozen, best checkpoint by hard-sign `sep`. Per batch (no grad): teacher
C_A (warmup→clamp→free), D_A (warmup→free), and M = top-k=256 spins by damage
`(C_A−D_A)(W_A[:,y]−W_A[:,k])` under A's full-spin C-probe.

- **Exp 1:** `L = CE(Wout·pool(D), y) + γ·mean_{i∈M} ReLU(κ − C_A[i]·h_D[i])`
- **Exp 2:** same but `mean_t mean_{i∈M} ReLU(κ − C_A[i]·h_t[i])`

κ=1, γ∈{1,3}, k=256, 3 seeds (minimal probe). All measured on the TRUE hard-sign D.

## Results (3 seeds)

| variant | D probe | D W_out | teacher C-probe on D | overlap(D,C_A) | field margin C_A·h on M | flip rate on M |
|---|---|---|---|---|---|---|
| exp1 γ=1 | 0.502 ± 0.004 | 0.485 | 0.298 | 0.558 | 0.246 | 0.446 |
| exp1 γ=3 | 0.486 ± 0.010 | 0.467 | 0.229 | 0.486 | 0.234 | 0.421 |
| exp2 γ=1 | 0.500 ± 0.007 | 0.477 | 0.283 | 0.567 | 0.213 | 0.455 |
| exp2 γ=3 | 0.490 ± 0.004 | 0.474 | 0.255 | 0.512 | 0.263 | 0.436 |

Refs: plain BPTT probe **0.506**; teacher A C-probe-on-D **0.26**, acc_C **0.96**, κ=1.

## Findings — hypothesis REFUTED

1. **No lift above the ceiling.** Every variant lands at D-probe ≈ 0.49–0.51 = plain
   BPTT (0.506); γ=3 slightly *hurts* (over-constraining, like the clamped-distance
   reg in exp 2). The field loss buys nothing.
2. **D does not fall into the C basin.** Teacher C-probe transfer on D stays ~0.23–0.30
   — essentially teacher A's baseline 0.26, nowhere near acc_C 0.96. Overlap(D,C_A)
   only ~0.5–0.6 (C would be ~0.9+). D remains in its own basin.
3. **The field loss can't even satisfy its own margin.** It nudged `C_A·h` on M from
   negative up to only ~+0.25 — far below κ=1 — and **~43% of the M spins still flip**
   C_A→D. A soft field bias conflicts with the free dynamics' fixed point: you cannot
   make the input-only field strongly support C_A on the class spins while staying a
   free attractor.

**Why this fails where exp-6 pinning succeeded.** Pinning *hard-clamps* the spins
throughout relaxation; the field loss only *tilts* the field and lets the dynamics
relax freely. Exp 6's separatrix needs ~60–70% of the class-flips actually suppressed,
but the loss leaves ~43% flipping. A gentle field tilt doesn't cross the basin
boundary. Consistent with exp 7 (PC feedback) and exp 2 (clamped-reg): nudging D toward
C — whether by distance, error feedback, or field support — does not unlock a new
regime. **The ~0.51 BPTT ceiling holds.**

## Files
- `run.py` — sweep (loads teacher A, init student from A). `--smoke`, `--gammas --k`.
- `plot.py` → `figures/field_loss.png`.
- `results/field_loss.json` — per (exp,γ): per-seed metrics + mean/std.

Shared infra added to `../bptt_common.py`: `diff_forward_traj`, `field_penalty`,
`make_field_step`. Run from repo root with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
Wider sweep (γ, k, κ) is cheap to extend, but the minimal probe already shows no signal.
