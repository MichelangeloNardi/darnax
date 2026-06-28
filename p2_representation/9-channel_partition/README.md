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

## Results (3 seeds, best_channel_entropy)

| tag | I | L | N | C-probe | D-probe | C→D flip | overlap(C,D) | C-free ov |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 16 | 16 | 0 | **0.972** | **0.452** | 0.051 | 0.898 | 0.999 |
| L8_N0 | 8 | 8 | 0 | 0.362 | 0.360 | **0.000** | **1.000** | 1.000 |
| L4_N0 | 12 | 4 | 0 | 0.385 | 0.382 | 0.000 | 1.000 | 1.000 |
| L4_N4 | 8 | 4 | 4 | 0.361 | 0.362 | 0.000 | 1.000 | 1.000 |
| L2_N0 | 14 | 2 | 0 | 0.402 | 0.403 | 0.000 | 1.000 | 1.000 |

Per-group probe (C ≡ D, so one number per group) and W_out reliance (‖W_out[feat,·]‖):

| tag | I probe | L probe | N probe | W_out reliance I / L / N |
|---|---|---|---|---|
| baseline | 0.96 | 0.96 | – | 2.75 / 2.75 / – |
| L8_N0 | 0.35 | **0.10** | – | 0.17 / 0.26 / – |
| L4_N0 | 0.37 | 0.10 | – | 0.10 / 0.34 / – |
| L4_N4 | 0.35 | 0.10 | **0.10** | 0.16 / 0.24 / 0.26 |
| L2_N0 | 0.39 | 0.10 | – | 0.06 / 0.55 / – |

### Findings

1. **Within-run control validates.** `baseline` reproduces exp-3/5 model A almost exactly
   (C-probe 0.97, D-probe 0.45, flip 0.05, overlap 0.90, full-spin pC→C 0.96 / pC→D 0.26 /
   random 0.96, enrichment 1.53). So `MaskedConv2D` at all-ones masks is a faithful no-op and
   the comparison is clean.

2. **Structural de-imprinting WORKS — but by disconnecting the label, not cleaning it.**
   Every partition de-imprints C hard: C-probe 0.97 → 0.36–0.40 (full-spin pC→C 0.96 → 0.28–
   0.32). This is the *opposite* of exp 7, where routing the label *signal* through W_back/W_out
   only **reinforced** the imprint. **But the partitions all collapse to C ≡ D** (flip = 0.000,
   overlap = 1.000): confining the label to a channel subset injects too little drive to move
   the sign fixed point *at all*, so there is no imprint to begin with — the label is
   effectively disconnected, not "more cleanly" imprinted.

3. **D does NOT rise toward 0.51 — it falls, and the fall is pure |I| (capacity), not routing.**
   D-probe tracks the input-channel count: L2_N0 (I=14) 0.403 > L4_N0 (I=12) 0.382 > L8_N0 /
   L4_N4 (I=8) 0.360. At fixed |I|=8, L8_N0 ≈ L4_N4 (0.360 vs 0.362) — the |L| / |N| split is
   irrelevant. So D is governed by how many channels receive **direct input**; this is exactly
   the **effective-capacity caveat** above realised, not a routing effect.

4. **Channels without direct input are class-DEAD; recurrence does not carry class info.**
   The per-group probe is the decisive result: in every partition, groups **L and N decode at
   chance (0.10)** while only group **I** carries class signal (0.35–0.39). Removing W_in from a
   channel makes it class-uninformative — the J1 recurrence does *not* propagate class structure
   into the label-only (L) or associative (N) channels. The associative group N is dead weight:
   `L4_N0` (I=12, N=0) 0.382 **>** `L4_N4` (I=8, N=4) 0.362 — converting input channels into
   recurrence-only channels strictly **hurts**. (W_out still places weight on the dead L/N
   channels — reliance even exceeds I — but those features carry no class signal, so it is
   wasted/bias capacity.)

**Bottom line.** Partitioning is a clean **negative for lifting D** but a sharp mechanistic
result: (i) the baseline's C-imprint is an *all-or-nothing broadcast* — the label must hit
(nearly) all channels to imprint C; confine it and C de-imprints to C ≡ D. (ii) The
representation's class information lives **entirely in the input-driven channels**; J1
recurrence alone builds nothing class-relevant, so D-probe is bottlenecked by |I| and
recurrence-only channels are dead weight. This rules out "de-imprint C structurally to free
D": the imprint was not what capped D — input-channel count is. Reference target unmet:
D-probe ~0.51 (BPTT ceiling) vs best partition 0.40 < baseline 0.45.

**Methodology fairness.** As flagged, the partitions are NOT effective-DOF-matched to baseline;
the result confirms that is exactly what dominates (D ∝ |I|). The fair claim is therefore:
*routing provides no benefit, and reducing the input-driven channel count hurts D while
recurrence-only channels contribute nothing* — not "routing hurts D".
