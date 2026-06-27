# Experiment 9 — channel partition by direct wiring (equal nominal capacity)

**Question.** Exp 7 tried to *de-imprint* the clamped state C with an error-driven
feedback clamp and failed because the error was routed onto **every** label-carrying
neuron (through `W_back`/`W_out`), reinforcing the imprint. The lesson: a de-imprinting
mechanism must be **decoupled from the label-readout direction**. This experiment is the
**structural** version of that idea — instead of changing the *signal*, we change the
*wiring*: confine the label injection to a dedicated subset of channels by construction,
so most channels form a representation the label never touches directly.

**Hypothesis.** Split the 16 hidden channels into three disjoint groups:
- **I** — receives the input message (`W_in`) directly,
- **L** — receives the label message (`W_back`) directly,
- **N** — receives **neither** directly (only the `J1` recurrence).

`J1` (full `groups=1` recurrence) and `W_out` (reads all 16 pooled) are **unchanged**, so
the groups communicate through recurrence and the readout still sees every channel. The
prediction: C becomes **less artificially label-imprinted on I and N** (the label can't
inject there), so the inference state D preserves those channels better — possibly lifting
D toward the ~0.51 BPTT ceiling, or at least changing the C-imprint / flip-targeting
signature relative to the baseline.

## Method — mask, not shrink (`partition.py`)

We keep `W_in` as the full 3→16 conv and `W_back` as the full 10→16 map but apply a fixed
**channel-output mask**:
- **`W_in` is Hebb-trained**, so its **updates are masked too** (`MaskedConv2D`, the same
  pattern `Conv2DRecurrentDiscrete` uses for its frozen `j_d` diagonal): the kernel columns
  of non-`I` channels start at zero and `backward()` multiplies `dW` by the channel mask, so
  they never move — verified post-training by `assert_masks_intact`.
- **`W_back` is frozen** (`backward`→zeros), so zeroing its non-`L` columns once at build is
  enough.

Every weight **tensor** keeps its baseline shape — a pure rewiring. No darnax-core edits;
`MaskedConv2D` and the masked builder live in `partition.py`.

### ⚠️ Capacity parity — what is and isn't matched
`build_model`/`train_models.py` print and **assert** the **nominal** trainable-param count
(`W_in.kernel` + `J1`−diagonal + `W_out.W` tensor elements) equals the baseline. This holds
**by construction** because masking does not change tensor shapes. **But the *effective*
(nonzero, actually-trained) connectivity does change**: `W_in` trains only `5·5·3·|I|` of its
1200 entries and `W_back`'s active fan-in drops to `10·|L|` (frozen regardless). So a
partition is a **strict constraint / subset** of the baseline's connectivity — **the baseline
is the capacity upper bound, not an equal-DOF twin.** Consequently:
- an **improvement** (or an equal-D with a changed C-imprint / flip signature) is **clean
  evidence for routing**;
- a **degradation** may partly reflect the **reduced effective `W_in`/`W_back` capacity**, not
  routing alone.
`param_report()` prints both numbers (nominal asserted-equal; effective varying).

## Partitions (3 seeds each)

Contiguous channel ranges `I=[0:|I|]`, `L=[|I|:|I|+|L|]`, `N=` rest.

| tag | I | L | N | isolates |
|---|---|---|---|---|
| **baseline** | 16 | 16 | 0 | matched control — every channel gets **both** (non-disjoint = current model) |
| `L8_N0` | 8 | 8 | 0 | label on half |
| `L4_N0` | 12 | 4 | 0 | small label footprint, no associative group |
| `L4_N4` | 8 | 4 | 4 | same \|L\|, **plus** a recurrence-only group N |
| `L2_N0` | 14 | 2 | 0 | tiny label footprint |

Two axes: **\|L\| at N=0** (8→4→2, "shrink the label's direct surface") and **adding N**
(`L4_N0` vs `L4_N4`, "purely-associative-through-recurrence channels"). `baseline` is trained
here (within-run control) and should reproduce exp-3 model A (~0.45 D-probe), since at
all-ones masks `MaskedConv2D` is a no-op.

## Files
- `partition.py` — `MaskedConv2D`, `build_partitioned_model`, partition registry, param accounting.
- `train_models.py` — trains baseline + 4 partitions × 3 seeds (exp-3 model-A recipe:
  DynamicalTrainer, `kernel_decay_rate`, 20 epochs), asserts masks intact, serializes
  `models/<tag>_seed<s>.eqx` (+ `meta.json` with the param report).
- `diagnostics.py` — **reuses exp-3 `diagnose()`** (stock C/D rollers) → global C/D probe,
  flip rate, overlap(C,D), C-free stability, field margins, importance-vs-flip, random-flip
  control. Directly comparable to exp-3 model A.
- `fullspin_importance.py` — **reuses exp-5 `analyze()`** → the decisive random-flip
  flip-targeting control on the 16384 raw spins.
- `group_diagnostics.py` — **per-group (I/L/N)** breakdown: C/D probe (pooled, group-restricted),
  flip rate / overlap, field margins, `W_out` reliance, and a per-group random-flip control
  (does group g's flipping target class-carriers, under a global full-spin C-probe).
- `plot.py` → `figures/global.png`, `figures/groups.png` (regenerate from the JSONs).
- `.gitignore` — `models/`, `results/{smoke,group_smoke}.json` (figures globally git-ignored).

`models/` and `figures/` are git-ignored. Run from repo root with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; `--smoke` on each for a tiny CPU run.

## Results

_Pending cluster run._ Reference baselines (exp-3/5 model A): C-probe 0.977, D-probe 0.451,
C→D flip 0.052, overlap(C,D) 0.896; full-spin C-probe-on-C 0.973, on-D (actual flips) 0.261,
on-D (random flips) 0.972 (enrichment 1.60). The BPTT ceiling target is D-probe ~0.51.
