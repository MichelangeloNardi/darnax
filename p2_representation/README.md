# Problem 2 — improving the hidden recurrent representation

Problem 1 showed the readout can be brought to ~0.44 (W_out) / ~0.46 (probe), but
**both plateau at the ~0.46 ceiling of the inference state D**. Problem 2 attacks that
ceiling: make the hidden recurrent (J1 / W_in) representation more separable.

## First planned experiment — BPTT ceiling (diagnostic)

Backprop-through-time the rollout to **D**, loss `CE(W_out · pool(D), y)`, real
gradients into `W_in`/`J1` (and `W_out`). This is **not** a darnax-faithful method —
it's a **diagnostic upper bound**: the best representation D can reach. Then measure
W_out (perceptron) and the probe on that representation to see how far the
gradient-free dynamics are from the ceiling.

## Reusing shared code

Model / optimizer / dataset / rollouts / rep-collection live in
`../p1_readout_gap/common.py`. Import it:

```python
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))   # shared experiment helpers
import common as cm
```

Note: `common.py` builds the model with the gradient-free local rules. BPTT needs a
**differentiable rollout** (gradients through `orch.step`), which is new infra to add
here — likely a thin differentiable forward + `optax` on the conv kernels via real
`jax.grad`, separate from `cm.make_optimizer` (which is for the local rule).
