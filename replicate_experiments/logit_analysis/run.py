"""logit_analysis/run.py

Matei's C-vs-D sensitivity experiment.

We train the Kassym model for TRAIN_EPOCHS epochs with the standard local
learning rule, then — with the trained weights frozen — analyse how much
the C→D transition changes the logits and which spins are responsible.

KEY QUANTITIES (per test image)
--------------------------------
h_C : 256-dim pooled J1 after warmup → clamped (WBack on) → free
h_D : 256-dim pooled J1 after warmup → free only (no label injection)
logits_C = h_C @ W        (W is the trained Wout matrix, 256×10)
logits_D = h_D @ W
Δlogits  = logits_C - logits_D   (which class scores changed?)
flipped  = sign(h_C) ≠ sign(h_D) (which of the 256 spins flipped direction?)

SENSITIVITY ANALYSIS
---------------------
For each spin i its contribution to logit k is |W[i,k]|.
Matei's hypothesis: flipped spins have disproportionately HIGH |W[i,k*]|
for the class k* that changed most — i.e. the free phase acts almost
adversarially by resetting exactly the class-relevant spins.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import jax
jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent / "1-Ablation_runs"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer
from darnax.trainers.utils import scan_n

TRAIN_EPOCHS = 10
SEED         = 0
C, KSIZE     = 16, 5
H, W, POOL   = 32, 32, 8
_STRIP = {"wback_type","j1_window_hebb","j1_entropy","trial_number","probe_acc","c05_j1"}


# ── build / train helpers (identical to standard_run) ─────────────────────────

def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    lm = LayerMap.from_dict({
        1: {0: Conv2D(3, C, KSIZE, threshold=cfg["threshold_win"], strength=1.0,
                      key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(C, KSIZE, groups=1, j_d=cfg["j_d"],
                      threshold=cfg["threshold_j"], key=keys[1], padding_mode="constant",
                      lr=1.0, weight_decay=0.0, entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2])},
        2: {1: PooledFlattenFC(POOL, H, W, C, 10, strength=1.0, threshold=5.0,
                      key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer()},
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=lm)


def make_optimizer(orch, cfg):
    import jax.tree_util as jtu
    mom = cfg["momentum"]
    params, _ = eqx.partition(orch, eqx.is_inexact_array)
    def like(t, v): return jtu.tree_map(lambda _: v, t, is_leaf=eqx.is_array)
    def sgd(lr): return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)
    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i,j), lbl in [((1,0),"win"),((1,1),"j1"),((2,1),"wout")]:
        labels = eqx.tree_at(lambda m,r=i,c=j: m.lmap[r][c], labels,
                             replace=like(params.lmap[i][j], lbl))
    opt = optax.multi_transform({"default": optax.set_to_zero(),
                                  "win": sgd(-cfg["lr_win"]),
                                  "j1":  sgd(-cfg["lr_j"]),
                                  "wout": sgd(cfg["lr_wout"])}, labels)
    return opt, opt.init(eqx.filter(orch, eqx.is_inexact_array))


def to_hwc(xb): return xb.reshape(-1, H, W, 3) * 2.0 - 1.0
def pool_np(h):
    N = h.shape[0]
    return h.reshape(N, H//POOL, POOL, W//POOL, POOL, C).mean(axis=(2,4)).reshape(N,-1)


# ── collect C and D states ────────────────────────────────────────────────────

@eqx.filter_jit
def get_C_and_D(orch, state_tmpl, x, y, rng, warmup_n, clamped_n, free_n):
    """Returns (h_C, h_D): J1 tensors (N,H,W,C) for both states."""
    # --- state C: warmup → clamped (WBack on) → free ---
    s = state_tmpl.init(x, y)
    (s, rng), _ = scan_n(orch.step, (s, rng), warmup_n, filter_messages="forward")
    (s, rng), _ = scan_n(orch.step, (s, rng), clamped_n, filter_messages="all")
    (s, rng), _ = scan_n(orch.step, (s, rng), free_n,    filter_messages="forward")
    h_C = s[1]

    # --- state D: warmup → free only (no label injection) ---
    s2 = state_tmpl.init(x, y)
    (s2, rng), _ = scan_n(orch.step, (s2, rng), warmup_n, filter_messages="forward")
    (s2, rng), _ = scan_n(orch.step, (s2, rng), free_n,   filter_messages="forward")
    h_D = s2[1]

    return h_C, h_D, rng


def collect_all(trainer, ds, key, warmup_n, clamped_n, free_n):
    rows = []
    for xb, yb in ds.iter_test():
        x = to_hwc(xb)
        h_C, h_D, key = get_C_and_D(
            trainer.orchestrator, trainer.state, x, yb, key,
            warmup_n, clamped_n, free_n)
        hC = pool_np(np.array(h_C))   # (N,256)
        hD = pool_np(np.array(h_D))
        y_int = np.argmax(np.array(yb), axis=-1)
        rows.append((hC, hD, y_int))
    hC_all  = np.concatenate([r[0] for r in rows])
    hD_all  = np.concatenate([r[1] for r in rows])
    y_all   = np.concatenate([r[2] for r in rows])
    return hC_all, hD_all, y_all


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(hC, hD, y_true, W):
    """
    W : (256, 10) Wout weight matrix.
    Returns a dict of per-image statistics.
    """
    N = len(y_true)

    logits_C = hC @ W                          # (N,10)
    logits_D = hD @ W                          # (N,10)
    delta_logits = logits_C - logits_D         # (N,10)
    delta_h      = hC - hD                     # (N,256)
    flipped      = (np.sign(hC) != np.sign(hD))  # (N,256) bool

    pred_C = np.argmax(logits_C, axis=1)       # (N,)
    pred_D = np.argmax(logits_D, axis=1)

    correct_C = (pred_C == y_true)
    correct_D = (pred_D == y_true)

    # for each image, the class dimension most disturbed by C→D
    k_star = np.argmax(np.abs(delta_logits), axis=1)  # (N,)

    # sensitivity: |W[i, k*]| for each spin i
    # W[:,k_star[n]] gives the column of W for image n's most-affected class
    sens = np.abs(W[:, k_star].T)   # (N, 256)

    # per-image: Pearson correlation between sensitivity and |Δh|
    abs_dh = np.abs(delta_h)
    corr_per_image = np.array([
        np.corrcoef(sens[n], abs_dh[n])[0, 1] for n in range(N)
    ])

    # aggregate: mean |W| for flipped vs non-flipped spins
    # W is (256,10), flipped is (N,256) — need per-spin global sensitivity
    global_sens = np.abs(W).mean(axis=1)   # (256,) mean |weight| per spin across classes
    w_flipped     = global_sens[flipped.any(axis=0)].mean()   if flipped.any() else np.nan
    w_non_flipped = global_sens[~flipped.any(axis=0)].mean()  if (~flipped).any() else np.nan

    return {
        "acc_C":            correct_C.mean(),
        "acc_D":            correct_D.mean(),
        "correct_C_wrong_D": (correct_C & ~correct_D).mean(),
        "wrong_C_correct_D": (~correct_C & correct_D).mean(),
        "mean_flipped_frac": flipped.mean(axis=1).mean(),
        "mean_abs_delta_logit": np.abs(delta_logits).mean(),
        "corr_sens_dh_mean": np.nanmean(corr_per_image),
        "w_flipped_mean":   w_flipped,
        "w_non_flipped_mean": w_non_flipped,
        "flip_sensitivity_ratio": w_flipped / w_non_flipped,
        # arrays for plotting
        "logits_C": logits_C, "logits_D": logits_D,
        "delta_h": delta_h, "flipped": flipped,
        "sens": sens, "abs_dh": abs_dh,
        "correct_C": correct_C, "correct_D": correct_D,
        "corr_per_image": corr_per_image,
    }


def plot(res, W, figures_dir):
    figures_dir.mkdir(exist_ok=True)
    flipped = res["flipped"]
    abs_dh  = res["abs_dh"]
    sens    = res["sens"]

    # ── 1. W magnitude: flipped vs non-flipped spins ─────────────────────────
    # Use mean |W| across all classes for each spin (global sensitivity)
    global_sens = np.abs(W).mean(axis=1)   # (256,) mean |weight| per spin
    fig, ax = plt.subplots(figsize=(7, 4))
    ever_flipped = flipped.any(axis=0)     # (256,) which spins flip in at least one image
    s_flip  = global_sens[ever_flipped]
    s_nflip = global_sens[~ever_flipped]
    ax.violinplot([s_nflip, s_flip], positions=[0, 1], showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-flipped spins", "Flipped spins (C→D)"])
    ax.set_ylabel("|Wout weight| (mean over classes)")
    ax.set_title(f"Flipped spins have {'higher' if s_flip.mean()>s_nflip.mean() else 'lower'} "
                 f"Wout sensitivity\n"
                 f"mean flipped={s_flip.mean():.4f}  non-flipped={s_nflip.mean():.4f}  "
                 f"ratio={s_flip.mean()/s_nflip.mean():.2f}×")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "wout_sensitivity_violin.png", dpi=150)
    plt.close(fig)

    # ── 2. Scatter: |Δh_i| vs |W[i, k*]| (per-spin, random subset of images) ─
    idx = np.random.choice(len(abs_dh), min(500, len(abs_dh)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(abs_dh[idx].reshape(-1), sens[idx].reshape(-1),
               alpha=0.05, s=2, color="#2563EB")
    ax.set_xlabel("|h_C - h_D| per spin")
    ax.set_ylabel("|W[spin, k*]| (sensitivity to most-changed class)")
    ax.set_title(f"Sensitivity vs spin change magnitude\n"
                 f"mean corr per image = {res['corr_sens_dh_mean']:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_scatter.png", dpi=150)
    plt.close(fig)

    # ── 3. Accuracy breakdown bar ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    cats  = ["Correct C\n& D", "Correct C\nwrong D\n(supervision fail)",
             "Wrong C\ncorrect D", "Wrong C\n& D"]
    cC, cD = res["correct_C"], res["correct_D"]
    vals = [(cC & cD).mean(), (cC & ~cD).mean(), (~cC & cD).mean(), (~cC & ~cD).mean()]
    colors = ["#16A34A", "#DC2626", "#EA580C", "#6B7280"]
    ax.bar(cats, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Fraction of test images")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title(f"C (cheat) acc={res['acc_C']:.3f}  D (honest) acc={res['acc_D']:.3f}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_breakdown.png", dpi=150)
    plt.close(fig)

    # ── 4. |Δlogit| histogram per class ───────────────────────────────────────
    delta_logits = res["logits_C"] - res["logits_D"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([np.abs(delta_logits[:, k]) for k in range(10)],
               labels=[str(k) for k in range(10)])
    ax.set_xlabel("Class index")
    ax.set_ylabel("|logit_C - logit_D|")
    ax.set_title("Which classes are most disturbed by C→D transition?")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "delta_logit_per_class.png", dpi=150)
    plt.close(fig)

    print(f"  Figures saved to {figures_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with open(CFG_PATH) as f:
        cfg = {k: v for k, v in json.load(f).items() if k not in _STRIP}

    warmup_n  = 1
    clamped_n = cfg["clamped_n_iter"]
    free_n    = cfg["free_n_iter"]
    print(f"Config: warmup={warmup_n}  clamped={clamped_n}  free={free_n}", flush=True)

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(orch, cfg)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=warmup_n,
        train_clamped_n_iter=clamped_n,
        train_free_n_iter=free_n,
        eval_n_iter=free_n,
    )

    print(f"Training {TRAIN_EPOCHS} epochs...", flush=True)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
        accs = []
        for xb, yb in ds.iter_test():
            key, m = trainer.eval_step(to_hwc(xb), yb, key)
            accs.append(float(m["accuracy"]))
        print(f"  epoch {epoch}/{TRAIN_EPOCHS}  head={np.mean(accs):.4f}", flush=True)

    # ── analysis ──────────────────────────────────────────────────────────────
    print("\nCollecting C and D states on test set...", flush=True)
    hC, hD, y_true = collect_all(trainer, ds, key, warmup_n, clamped_n, free_n)

    # Wout weight matrix: shape (256, 10)
    W = np.array(trainer.orchestrator.lmap[2][1].W)
    print(f"  hC: {hC.shape}  hD: {hD.shape}  W: {W.shape}")

    res = analyse(hC, hD, y_true, W)
    print(f"\n=== Results ===")
    print(f"  Acc on C (cheat — label injected): {res['acc_C']:.4f}")
    print(f"  Acc on D (honest — no label):      {res['acc_D']:.4f}")
    print(f"  Correct C, wrong D (supervision fail): {res['correct_C_wrong_D']:.4f}")
    print(f"  Wrong C, correct D:                    {res['wrong_C_correct_D']:.4f}")
    print(f"  Mean fraction of spins flipped C→D: {res['mean_flipped_frac']:.4f}")
    print(f"  Mean |Δlogit|:                       {res['mean_abs_delta_logit']:.4f}")
    print(f"  Mean corr(sensitivity, |Δh|):        {res['corr_sens_dh_mean']:.4f}")
    print(f"  Mean |W| flipped spins:     {res['w_flipped_mean']:.5f}")
    print(f"  Mean |W| non-flipped spins: {res['w_non_flipped_mean']:.5f}")
    print(f"  Ratio (flipped / non-flipped): {res['flip_sensitivity_ratio']:.3f}×")

    out_dir = HERE / "figures"
    plot(res, W, out_dir)

    summary = {k: float(v) for k, v in res.items()
               if isinstance(v, (float, np.floating))}
    (HERE / "results.json").write_text(json.dumps(summary, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
