"""contribution_analysis.py

Mattia's question: "have we ever taken a look at the distribution of the margin
contributions at a fixed point for the entropy rule vs the old rule, both after training?"

For each rule (entropy / current), after training:
  1. Run N_ANALYSIS test images to their fixed point (inference dynamics).
  2. For each output neuron (h,w,j) in J1, compute all 400 contributions:
       c_{dh,dw,i→j}(h,w) = s_out[h,w,j] * kernel[dh,dw,i,j] * s_in[h+dh,w+dw,i]
     These are the individual terms that sum to give neuron j's total recurrent input.
  3. Compare distributions: histogram, per-neuron entropy, Gini coefficient.

Hypothesis:
  - Entropy rule: contributions more uniform (high entropy, low Gini)
  - Current rule: a few inputs dominate (low entropy, high Gini, heavy tail)

Run from repo root:
  python experiments4/contribution_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"

sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

C, KSIZE    = 16, 5
H, W        = 32, 32
POOL        = 8
TRAIN_EPOCHS = 10    # enough to see the effect; full 20 would take twice as long
SEED        = 0
N_ANALYSIS  = 200    # test images to analyze fixed-point contributions on

RULES = {
    "entropy": {"lambda_entropy": 1.0, "warmup_n_iter": 1},
    "current": {"lambda_entropy": 0.0, "warmup_n_iter": 1},
}


# ---------------------------------------------------------------------------
# Model / optimizer (same as replicate)
# ---------------------------------------------------------------------------

def build_model(cfg, key, lambda_entropy):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(
                in_channels=3, out_channels=C, kernel_size=KSIZE,
                threshold=cfg["threshold_win"], strength=1.0,
                key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.0,
            ),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=lambda_entropy,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(
                pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                strength=1.0, threshold=5.0,
                key=keys[3], lr=1.0, weight_decay=0.0,
            ),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def build_optimizer(orchestrator, cfg):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(
            lambda m, r=i, c=j: m.lmap[r][c], labels,
            replace=like(params.lmap[i][j], lbl),
        )

    opt = optax.multi_transform({
        "default": optax.sgd(0.0),
        "win":     sgd(-cfg["lr_win"]),
        "j1":      sgd(-cfg["lr_j"]),
        "wout":    sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg, ds, lambda_entropy, rule_name):
    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk, lambda_entropy)
    opt, opt_state = build_optimizer(orch, cfg)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    decay = cfg["kernel_decay_rate"]
    for epoch in range(1, TRAIN_EPOCHS + 1):
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
            win_k = trainer.orchestrator.lmap[1][0].kernel
            kh, kw, ci, co = win_k.shape
            flat   = win_k.reshape(-1, co)
            normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            trainer.orchestrator = eqx.tree_at(
                lambda o: o.lmap[1][0].kernel,
                trainer.orchestrator,
                normed.reshape(kh, kw, ci, co),
            )
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay)
            )

        # quick head acc
        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        print(f"  {rule_name}  epoch={epoch:2d}/{TRAIN_EPOCHS}  "
              f"head={np.mean(batch_accs):.4f}", flush=True)

    return trainer, key


# ---------------------------------------------------------------------------
# Contribution computation
# ---------------------------------------------------------------------------

def extract_patches(s_in, kh, kw):
    """Extract sliding-window patches from s_in.

    s_in: (1, H, W, C_in)
    Returns: (H, W, kh*kw*C_in) — for each output position, all input values
             in the receptive field, in kernel order.
    """
    pad_h, pad_w = kh // 2, kw // 2
    s_pad = jnp.pad(s_in, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)), mode="constant")

    patches = lax.conv_general_dilated_patches(
        s_pad,
        filter_shape=(kh, kw),
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )  # (1, H, W, kh*kw*C_in)
    return patches[0]  # (H, W, kh*kw*C_in)


def compute_contributions(kernel, s_in, s_out):
    """Compute per-input contributions to each output neuron.

    kernel : (kh, kw, cin_g, cout)  — J1 kernel (groups=1 so cin_g=C)
    s_in   : (1, H, W, C)           — input state (layer 1 activations)
    s_out  : (1, H, W, C)           — output state (same layer, fixed point)

    Returns: contributions of shape (H, W, cout, kh*kw*cin_g)
      contributions[h,w,j,:] = s_out[h,w,j] * kernel[:,:,:,j].flatten() * patches[h,w,:]
    """
    kh, kw, cin_g, cout = kernel.shape
    patches = extract_patches(s_in, kh, kw)  # (H, W, kh*kw*cin_g)

    # kernel reshaped: (kh*kw*cin_g, cout)
    k_flat = kernel.reshape(-1, cout)  # (400, 16)

    # contributions[h,w,j,input_idx] = s_out[h,w,j] * k_flat[input_idx,j] * patches[h,w,input_idx]
    # patches: (H,W,400), k_flat: (400,16), s_out[0]: (H,W,16)
    # raw[h,w,input_idx,j] = patches[h,w,input_idx] * k_flat[input_idx,j]
    raw = patches[:, :, :, None] * k_flat[None, None, :, :]  # (H,W,400,16)
    # multiply by s_out sign
    s_out_hw = s_out[0][:, :, None, :]  # (H,W,1,16)
    contributions = raw * s_out_hw       # (H,W,400,16)
    # reorder to (H,W,cout,n_inputs)
    contributions = contributions.transpose(0, 1, 3, 2)  # (H,W,16,400)
    return np.array(contributions)


def boltzmann_entropy(contribs, beta):
    """Entropy of Boltzmann distribution over contributions.

    contribs: (..., n_inputs)
    Returns entropy of same shape without last dim.
    """
    logits = beta * contribs
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs  = np.exp(logits)
    probs  = probs / probs.sum(axis=-1, keepdims=True)
    # entropy
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    return ent


def gini(contribs):
    """Gini coefficient of |contributions|. 0=uniform, 1=maximally concentrated."""
    x = np.abs(contribs)
    x = np.sort(x, axis=-1)
    n = x.shape[-1]
    idx = np.arange(1, n + 1)
    return (2 * (idx * x).sum(axis=-1) / (n * x.sum(axis=-1) + 1e-12) - (n + 1) / n)


def collect_contributions(trainer, ds_test, key, cfg):
    """Run N_ANALYSIS test images to fixed point, collect J1 contributions."""
    kernel = np.array(trainer.orchestrator.lmap[1][1].kernel)  # (5,5,16,16)
    all_contribs = []
    count = 0

    for xb, yb in ds_test.iter_test():
        if count >= N_ANALYSIS:
            break
        bs = xb.shape[0]
        for b in range(min(bs, N_ANALYSIS - count)):
            img = to_hwc(xb[b:b+1])
            key, _ = trainer.eval_step(img, yb[b:b+1], key)
            s1 = trainer.state[1]  # (1,H,W,C) — fixed-point J1 state
            c = compute_contributions(kernel, s1, s1)  # (H,W,16,400)
            all_contribs.append(c)
            count += 1

    return np.stack(all_contribs)  # (N_ANALYSIS, H, W, 16, 400)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_analysis(results: dict, out_path: Path, cfg):
    """Four-panel comparison figure."""
    beta = cfg["entropy_beta"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    colours = {"entropy": "steelblue", "current": "tomato"}

    # ---- panel 1: histogram of all contributions ----
    ax = axes[0, 0]
    for rule, contribs in results.items():
        flat = contribs.flatten()
        ax.hist(flat, bins=200, alpha=0.5, density=True,
                color=colours[rule], label=rule)
    ax.set_xlabel("Contribution c = s_out · J_ij · s_in")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of all contributions\n(all neurons, all images)")
    ax.legend(); ax.grid(alpha=0.3)

    # ---- panel 2: histogram of |contributions| ----
    ax = axes[0, 1]
    for rule, contribs in results.items():
        flat = np.abs(contribs).flatten()
        ax.hist(flat, bins=150, alpha=0.5, density=True,
                color=colours[rule], label=rule)
    ax.set_xlabel("|Contribution|")
    ax.set_ylabel("Density")
    ax.set_title("|Contributions| — does entropy rule suppress large values?")
    ax.legend(); ax.grid(alpha=0.3)

    # ---- panel 3: per-neuron Boltzmann entropy distribution ----
    ax = axes[1, 0]
    for rule, contribs in results.items():
        # contribs: (N, H, W, 16, 400) → entropy: (N, H, W, 16)
        ent = boltzmann_entropy(contribs, beta)  # (N,H,W,16)
        flat = ent.flatten()
        ax.hist(flat, bins=100, alpha=0.5, density=True,
                color=colours[rule], label=f"{rule}  μ={flat.mean():.2f}")
    ax.axvline(np.log(400), color="k", ls=":", lw=1.5, label="max entropy (uniform)")
    ax.set_xlabel(f"Boltzmann entropy (β={beta:.2f})")
    ax.set_ylabel("Density")
    ax.set_title("Per-neuron contribution entropy\n(higher = more uniform inputs)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- panel 4: Gini coefficient distribution ----
    ax = axes[1, 1]
    for rule, contribs in results.items():
        g = gini(contribs)  # (N,H,W,16)
        flat = g.flatten()
        ax.hist(flat, bins=100, alpha=0.5, density=True,
                color=colours[rule], label=f"{rule}  μ={flat.mean():.3f}")
    ax.set_xlabel("Gini coefficient of |contributions|")
    ax.set_ylabel("Density")
    ax.set_title("Gini coefficient per neuron\n(lower = more uniform, higher = more concentrated)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"Margin contribution distributions at fixed point\n"
        f"entropy vs current rule — {TRAIN_EPOCHS} epochs, seed={SEED}, "
        f"N={N_ANALYSIS} test images",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    results_dir = HERE / "results"
    figs_dir    = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    trained = {}
    keys    = {}

    for rule_name, rule_cfg in RULES.items():
        print(f"\n{'='*55}")
        print(f"Training: {rule_name}  (lambda_entropy={rule_cfg['lambda_entropy']})")
        print(f"{'='*55}")
        trainer, key = train(cfg, ds, rule_cfg["lambda_entropy"], rule_name)
        trained[rule_name] = trainer
        keys[rule_name] = key

    print(f"\nCollecting fixed-point contributions on {N_ANALYSIS} test images...")
    contribs = {}
    for rule_name, trainer in trained.items():
        print(f"  {rule_name}...", flush=True)
        contribs[rule_name] = collect_contributions(trainer, ds, keys[rule_name], cfg)
        print(f"    shape: {contribs[rule_name].shape}  "
              f"mean|c|={np.abs(contribs[rule_name]).mean():.4f}  "
              f"std={contribs[rule_name].std():.4f}")

    # summary stats
    print("\nSummary:")
    beta = cfg["entropy_beta"]
    for rule_name, c in contribs.items():
        ent  = boltzmann_entropy(c, beta).mean()
        g    = gini(c).mean()
        frac_large = (np.abs(c) > 0.5).mean()
        print(f"  {rule_name:>10}:  mean_entropy={ent:.3f}  mean_gini={g:.3f}  "
              f"frac|c|>0.5={frac_large:.4f}")

    # save summary
    summary = {}
    for rule_name, c in contribs.items():
        ent = boltzmann_entropy(c, beta)
        g   = gini(c)
        summary[rule_name] = {
            "mean_entropy": float(ent.mean()),
            "std_entropy":  float(ent.std()),
            "mean_gini":    float(g.mean()),
            "std_gini":     float(g.std()),
            "mean_abs_contrib": float(np.abs(c).mean()),
            "std_contrib":  float(c.std()),
            "frac_large":   float((np.abs(c) > 0.5).mean()),
        }
    out = results_dir / "contribution_analysis.json"
    out.write_text(json.dumps({"config": cfg, "seed": SEED,
                               "train_epochs": TRAIN_EPOCHS,
                               "n_analysis": N_ANALYSIS,
                               "summary": summary}, indent=2))
    print(f"\nSaved summary to {out}")

    plot_analysis(contribs, figs_dir / "contribution_analysis.png", cfg)
    print("Done.")


if __name__ == "__main__":
    main()
