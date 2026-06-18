"""bptt/bptt_run.py

BPTT + cross-entropy loss baseline — replicating Matei's ~50% target.

WHAT CHANGED vs standard local-learning training
-------------------------------------------------
1. STEConv2DRecurrentDiscrete (line ~80):
     The only architectural change. Overrides Conv2DRecurrentDiscrete.activation()
     to use the straight-through estimator:

         forward:  sign(x)          (same discrete value as before)
         backward: dy/dx = 1        (gradient passes through as if identity)

     Implementation: x + stop_gradient(sign(x) - x)
       - forward:  x + (sign(x)-x) = sign(x)  ✓
       - backward: d/dx[x] + d/dx[stop_grad(...)] = 1 + 0 = 1  ✓

     Without STE, sign(x) has zero gradient everywhere → no learning signal.

2. loss_and_grad() (line ~140):
     Replaces the local perceptron/Hebb backward() with standard JAX autograd.
     Runs the full forward trajectory (warmup → clamped → free), reads out the
     Wout logits, computes cross-entropy, and differentiates through everything.

3. Optimizer: Adam on all trainable params (Win, J1, Wout) jointly.
   WBack remains frozen (never updated), same as standard training.

WHAT DID NOT CHANGE
-------------------
- Architecture: same Conv2D + Conv2DRecurrentDiscrete + ChannelWBack + PooledFlattenFC
- Forward phases: warmup(1) → clamped(5, WBack on) → free(6)
- WBack is still active during the clamped phase — labels are still injected
- POOL, KSIZE, C, H, W: identical
- Dataset, preprocessing: identical

TARGET: ~50% test accuracy (Matei's reported BPTT result)
BASELINE: Kassym local learning = 0.274, Matei local learning = 0.430 (20ep)
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import jax
jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.utils import scan_n

EPOCHS    = 20
SEED      = 0
LR        = 3e-4          # Adam learning rate (same for all params)
C, KSIZE  = 16, 5
H, W, POOL = 32, 32, 8
_STRIP = {"wback_type","j1_window_hebb","j1_entropy","trial_number","probe_acc","c05_j1"}


# ── CHANGE 1: STE activation ─────────────────────────────────────────────────

class STEConv2DRecurrentDiscrete(Conv2DRecurrentDiscrete):
    """Conv2DRecurrentDiscrete with straight-through estimator for sign().

    Only activation() is overridden. Every other method (forward conv,
    backward / Hebb rule, reduce) is inherited unchanged and unused here —
    we never call backward() in the BPTT regime.
    """

    def activation(self, x: jax.Array) -> jax.Array:
        # STE: forward = sign(x), backward = identity (gradient passes through)
        return x + jax.lax.stop_gradient(jnp.sign(x) - x)


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, key: jax.Array):
    keys = jax.random.split(key, 5)
    lm = LayerMap.from_dict({
        1: {
            # Win: unchanged Conv2D
            0: Conv2D(3, C, KSIZE, threshold=cfg["threshold_win"], strength=1.0,
                      key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.0),
            # J1: STE version — the only architectural change
            1: STEConv2DRecurrentDiscrete(
                C, KSIZE, groups=1, j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            # WBack: unchanged, still frozen (never in the optimizer)
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            # Wout: unchanged PooledFlattenFC — but now trained via CE gradient
            1: PooledFlattenFC(POOL, H, W, C, 10, strength=1.0, threshold=5.0,
                      key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    return state, SequentialOrchestrator(layers=lm)


# ── CHANGE 2: forward pass + CE loss ─────────────────────────────────────────

def forward(orch, state_tmpl, x, y, warmup_n, clamped_n, free_n, rng):
    """Unrolled forward pass: warmup → clamped → free → logits.

    Same phases as DynamicalTrainer._train_step_impl, but:
    - No local backward() called
    - Returns logits (N,10) for CE loss
    - Fully differentiable through STE
    """
    s = state_tmpl.init(x, y)
    # warmup: forward-only (Win + J1, no WBack)
    (s, rng), _ = scan_n(orch.step, (s, rng), warmup_n, filter_messages="forward")
    # clamped: all messages (WBack injects label)
    (s, rng), _ = scan_n(orch.step, (s, rng), clamped_n, filter_messages="all")
    # free: forward-only (Win + J1, WBack off)
    (s, rng), _ = scan_n(orch.step, (s, rng), free_n, filter_messages="forward")
    # logits via Wout: pool(J1) @ W
    logits = orch.lmap[2][1](s[1])   # PooledFlattenFC.__call__  → (N,10)
    return logits


def ce_loss(logits: jax.Array, y_pm1: jax.Array) -> jax.Array:
    y_int = jnp.argmax(y_pm1, axis=-1)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y_int))


# ── CHANGE 3: Adam optimizer on all trainable params ─────────────────────────

def make_optimizer(orch):
    """Single Adam optimizer on Win + J1 + Wout. WBack is excluded (frozen)."""
    # Partition: trainable = Win kernel + J1 kernel + Wout W
    # WBack has no trainable params in the standard perceptron setup either.
    def is_trainable(x):
        return eqx.is_inexact_array(x)

    # We exclude WBack by tagging it as non-trainable.
    # WBack params are in lmap[1][2]; we zero its grad with a filter.
    params, static = eqx.partition(orch, eqx.is_inexact_array)

    # Zero out WBack so Adam never touches it (same as local learning: WBack frozen)
    def zero_wback(tree):
        return eqx.tree_at(lambda m: m.lmap[1][2], tree,
                           jtu.tree_map(jnp.zeros_like, tree.lmap[1][2],
                                        is_leaf=eqx.is_array),
                           is_leaf=eqx.is_array)

    opt = optax.adam(LR)
    opt_state = opt.init(params)
    return opt, opt_state, params, static


# ── JIT-compiled train step ───────────────────────────────────────────────────

@eqx.filter_jit
def train_step(orch, opt_state, opt, state_tmpl, x, y, warmup_n, clamped_n, free_n, rng):
    params, static = eqx.partition(orch, eqx.is_inexact_array)

    def loss_fn(params):
        orch_ = eqx.combine(params, static)
        logits = forward(orch_, state_tmpl, x, y, warmup_n, clamped_n, free_n, rng)
        return ce_loss(logits, y)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Zero WBack gradients so it is never updated
    grads = eqx.tree_at(
        lambda m: m.lmap[1][2], grads,
        jtu.tree_map(jnp.zeros_like, grads.lmap[1][2], is_leaf=eqx.is_array),
        is_leaf=eqx.is_array,
    )

    updates, new_opt_state = opt.update(grads, opt_state, params=params)
    new_orch = eqx.apply_updates(orch, updates)
    return new_orch, new_opt_state, loss


@eqx.filter_jit
def eval_step(orch, state_tmpl, x, y, warmup_n, free_n, rng):
    """Evaluate on D state (no label injection, same as real inference)."""
    s = state_tmpl.init(x, y)
    (s, rng), _ = scan_n(orch.step, (s, rng), warmup_n, filter_messages="forward")
    (s, rng), _ = scan_n(orch.step, (s, rng), free_n,   filter_messages="forward")
    logits = orch.lmap[2][1](s[1])
    y_int  = jnp.argmax(y, axis=-1)
    acc    = jnp.mean(jnp.argmax(logits, axis=-1) == y_int)
    return acc


def to_hwc(xb): return xb.reshape(-1, H, W, 3) * 2.0 - 1.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with open(CFG_PATH) as f:
        cfg = {k: v for k, v in json.load(f).items() if k not in _STRIP}

    warmup_n  = 1
    clamped_n = cfg["clamped_n_iter"]
    free_n    = cfg["free_n_iter"]
    print(f"Config: warmup={warmup_n}  clamped={clamped_n}  free={free_n}  lr_adam={LR}")
    print(f"Epochs: {EPOCHS}  Seed: {SEED}", flush=True)

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state_tmpl, orch = build_model(cfg, mk)

    params, static = eqx.partition(orch, eqx.is_inexact_array)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    head_accs, losses = [], []

    for epoch in range(1, EPOCHS + 1):
        ep_losses = []
        for xb, yb in ds:
            x = to_hwc(xb)
            key, rng = jax.random.split(key)
            orch, opt_state, loss = train_step(
                orch, opt_state, opt, state_tmpl, x, yb,
                warmup_n, clamped_n, free_n, rng)
            ep_losses.append(float(loss))

        # eval on D (honest inference, no label)
        accs = []
        for xb, yb in ds.iter_test():
            key, rng = jax.random.split(key)
            acc = eval_step(orch, state_tmpl, to_hwc(xb), yb, warmup_n, free_n, rng)
            accs.append(float(acc))

        mean_acc  = float(np.mean(accs))
        mean_loss = float(np.mean(ep_losses))
        head_accs.append(mean_acc)
        losses.append(mean_loss)
        print(f"  epoch {epoch:2d}/{EPOCHS}  CE_loss={mean_loss:.4f}  head(D)={mean_acc:.4f}", flush=True)

    print(f"\nFinal head accuracy (D state): {head_accs[-1]:.4f}")
    print(f"Best head accuracy:            {max(head_accs):.4f}")

    # save
    results_dir = HERE / "results"; results_dir.mkdir(exist_ok=True)
    figures_dir = HERE / "figures"; figures_dir.mkdir(exist_ok=True)
    out = {"epochs": EPOCHS, "lr": LR, "seed": SEED,
           "warmup_n": warmup_n, "clamped_n": clamped_n, "free_n": free_n,
           "head_accs": head_accs, "losses": losses,
           "final_head": head_accs[-1], "best_head": max(head_accs)}
    (results_dir / "run.json").write_text(json.dumps(out, indent=2))

    # plot
    ep = np.arange(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep, losses, "-o", markersize=3, color="#2563EB")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CE loss"); ax1.set_title("Training loss (BPTT)")
    ax1.grid(alpha=0.3)
    ax2.plot(ep, head_accs, "-o", markersize=3, color="#16A34A", label="BPTT+CE (D eval)")
    ax2.axhline(0.274, color="#DC2626", linestyle="--", label="Kassym local (0.274)")
    ax2.axhline(0.430, color="#EA580C", linestyle="--", label="Matei local 20ep (0.430)")
    ax2.axhline(0.500, color="#7C3AED", linestyle=":", label="BPTT target (0.500)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Test acc"); ax2.set_title("Head accuracy on D state")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle(f"BPTT + CE loss + STE  (Adam lr={LR}, Kassym config)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "bptt_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved results/run.json and figures/bptt_curves.png")


if __name__ == "__main__":
    main()
