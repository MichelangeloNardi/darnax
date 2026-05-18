"""attractor_geometry.py

Measures the geometry of the network's attractor configurations in J1
space (theme 2). Tests whether the Representation Manifold described
in the paper manifests as clean class clusters in the trained model.

After training the standard channel-entropy model, collects final
state[1] (32×32×16 = 8192-d, ±1 binary) for N_PER_CLASS=100 test
images per class (1000 total). Then:

  - Within-class vs between-class cosine similarity distributions
  - Per-class centroid pairwise similarity matrix (10×10 heatmap)
  - Per-class PCA spectrum → intrinsic dimensionality
  - 2D t-SNE embedding of all states, colored by class

If within-class sim ≫ between-class sim, the RM has clean class
clusters → scaling = more / sharper clusters.
If they overlap, classes are not separated in attractor space and
scaling channels won't help until the architecture is fixed.

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/attractor_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
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

C, KSIZE = 16, 5
H, W = 32, 32
EPOCHS = 20
SEED = 0
N_PER_CLASS = 100
N_CLASSES = 10
PCA_VAR_TARGET = 0.9  # for intrinsic dim


# ---------------------------------------------------------------------------
# Model / optimizer / preprocessing (mirrors replicate_channel_entropy.py)
# ---------------------------------------------------------------------------

def build_model(cfg: dict, key: jax.Array):
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
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(
                pool=8, H=H, W=W, C_in=C, n_classes=10,
                strength=1.0, threshold=5.0,
                key=keys[3], lr=1.0, weight_decay=0.0,
            ),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def make_optimizer(orchestrator, cfg: dict):
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

def train_model(cfg: dict, ds: Cifar10, seed: int):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(orch, cfg)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    decay = cfg["kernel_decay_rate"]
    head_accs: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
            win_k = trainer.orchestrator.lmap[1][0].kernel
            kh, kw, ci, co = win_k.shape
            flat = win_k.reshape(-1, co)
            normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            trainer.orchestrator = eqx.tree_at(
                lambda o: o.lmap[1][0].kernel,
                trainer.orchestrator,
                normed.reshape(kh, kw, ci, co),
            )

        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay),
            )

        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        head_acc = float(np.mean(batch_accs))
        head_accs.append(head_acc)
        print(f"  epoch={epoch:2d}/{EPOCHS}  head={head_acc:.4f}", flush=True)

    return trainer, head_accs, key


# ---------------------------------------------------------------------------
# Collect attractor states (N_PER_CLASS per class from test set)
# ---------------------------------------------------------------------------

def collect_class_attractors(trainer, ds: Cifar10, key: jax.Array):
    """Return (X, y, key) where:
      X : (N_PER_CLASS * N_CLASSES, 8192) — flattened final state[1]
      y : (N_PER_CLASS * N_CLASSES,)      — class labels in 0..9
    """
    per_class: dict[int, list[np.ndarray]] = {c: [] for c in range(N_CLASSES)}

    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        h = np.array(trainer.state[1])               # (B, 32, 32, 16)
        h_flat = h.reshape(h.shape[0], -1)           # (B, 8192)
        lbls = np.argmax(np.array(yb), axis=-1)
        for i, lbl in enumerate(lbls):
            lbl = int(lbl)
            if len(per_class[lbl]) < N_PER_CLASS:
                per_class[lbl].append(h_flat[i])
        if all(len(v) >= N_PER_CLASS for v in per_class.values()):
            break

    X = np.concatenate([np.stack(per_class[c][:N_PER_CLASS]) for c in range(N_CLASSES)])
    y = np.concatenate([np.full(N_PER_CLASS, c) for c in range(N_CLASSES)])
    return X, y, key


# ---------------------------------------------------------------------------
# Geometry analysis
# ---------------------------------------------------------------------------

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine similarity (N, N). For ±1 vectors this equals dot/D."""
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norm
    return Xn @ Xn.T


def analyze_geometry(X: np.ndarray, y: np.ndarray) -> dict:
    N, D = X.shape
    sim = cosine_sim_matrix(X)

    same_class = (y[:, None] == y[None, :])
    diag = np.eye(N, dtype=bool)
    within = sim[same_class & ~diag]
    between = sim[~same_class]

    centroids = np.stack([X[y == c].mean(axis=0) for c in range(N_CLASSES)])
    centroid_sim = cosine_sim_matrix(centroids)  # (10, 10)

    # Per-class intrinsic dim via PCA on covariance.
    # Using the dual N×N Gram matrix since N=100 << D=8192.
    intrinsic_dims: list[int] = []
    for c in range(N_CLASSES):
        Xc = X[y == c]
        Xc_centered = Xc - Xc.mean(axis=0, keepdims=True)
        gram = Xc_centered @ Xc_centered.T / max(Xc.shape[0] - 1, 1)
        eigs = np.linalg.eigvalsh(gram)[::-1]
        eigs = np.clip(eigs, 0, None)
        if eigs.sum() < 1e-12:
            intrinsic_dims.append(0)
            continue
        cumvar = np.cumsum(eigs) / eigs.sum()
        k = int(np.searchsorted(cumvar, PCA_VAR_TARGET) + 1)
        intrinsic_dims.append(k)

    return {
        "within_mean": float(within.mean()),
        "within_std": float(within.std()),
        "between_mean": float(between.mean()),
        "between_std": float(between.std()),
        "separation": float(within.mean() - between.mean()),
        "within_sim": within,
        "between_sim": between,
        "centroid_sim": centroid_sim,
        "intrinsic_dims": intrinsic_dims,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "auto", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def plot_geometry(metrics: dict, X: np.ndarray, y: np.ndarray, fig_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    ax_hist, ax_cent, ax_dim, ax_tsne = axes.flat

    # 1. Within vs between similarity histogram
    bins = np.linspace(min(metrics["between_sim"].min(), metrics["within_sim"].min()),
                       max(metrics["between_sim"].max(), metrics["within_sim"].max()), 60)
    ax_hist.hist(metrics["within_sim"], bins=bins, density=True, alpha=0.6,
                 label=f"within (μ={metrics['within_mean']:.3f})", color="steelblue")
    ax_hist.hist(metrics["between_sim"], bins=bins, density=True, alpha=0.6,
                 label=f"between (μ={metrics['between_mean']:.3f})", color="orange")
    ax_hist.axvline(metrics["within_mean"], color="steelblue", linestyle="--", linewidth=1)
    ax_hist.axvline(metrics["between_mean"], color="orange", linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Cosine similarity between attractors")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Within vs between class similarity\n"
                      f"separation = {metrics['separation']:.3f}")
    ax_hist.legend()

    # 2. Centroid similarity heatmap
    csim = metrics["centroid_sim"]
    im = ax_cent.imshow(csim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_cent.set_xticks(range(N_CLASSES))
    ax_cent.set_yticks(range(N_CLASSES))
    ax_cent.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax_cent.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax_cent.set_title("Class centroid cosine similarity")
    fig.colorbar(im, ax=ax_cent, fraction=0.046)

    # 3. Per-class intrinsic dimension
    ax_dim.bar(range(N_CLASSES), metrics["intrinsic_dims"], color="steelblue", alpha=0.85)
    ax_dim.set_xticks(range(N_CLASSES))
    ax_dim.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax_dim.set_ylabel(f"PCs to {int(PCA_VAR_TARGET*100)}% var")
    ax_dim.set_title(f"Per-class intrinsic dim (D_ambient={X.shape[1]})")
    for i, d in enumerate(metrics["intrinsic_dims"]):
        ax_dim.text(i, d + 0.5, str(d), ha="center", fontsize=8)

    # 4. t-SNE
    from sklearn.manifold import TSNE
    print("  Running t-SNE (this may take ~30s)...", flush=True)
    Z = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(X)
    cmap = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
    for c in range(N_CLASSES):
        m = (y == c)
        ax_tsne.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.7,
                        label=CIFAR10_CLASSES[c], color=cmap[c])
    ax_tsne.set_title("t-SNE of attractor states (color = class)")
    ax_tsne.legend(fontsize=7, ncol=2, markerscale=1.5, loc="best")
    ax_tsne.set_xticks([]); ax_tsne.set_yticks([])

    fig.suptitle(f"Attractor geometry — C={C}, seed={SEED}, {EPOCHS} epochs, "
                 f"{N_PER_CLASS}/class", fontsize=11)
    fig.tight_layout()
    fig_path = fig_dir / "attractor_geometry.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {fig_path.name}")
    return Z


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
    figs_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    print(f"Training seed={SEED} for {EPOCHS} epochs ...")
    trainer, head_accs, key = train_model(cfg, ds, SEED)
    final_head = head_accs[-1]

    print(f"\nCollecting {N_PER_CLASS} attractors per class ...")
    X, y, key = collect_class_attractors(trainer, ds, key)
    print(f"  X.shape={X.shape}  y.shape={y.shape}")
    binary_frac = float(np.mean(np.isclose(np.abs(X), 1.0, atol=0.05)))
    print(f"  fraction of entries that are ±1: {binary_frac:.4f}")

    print("\nAnalyzing geometry ...")
    metrics = analyze_geometry(X, y)

    print(f"  Within-class mean cosine sim:  {metrics['within_mean']:.4f} "
          f"± {metrics['within_std']:.4f}")
    print(f"  Between-class mean cosine sim: {metrics['between_mean']:.4f} "
          f"± {metrics['between_std']:.4f}")
    print(f"  Separation:                    {metrics['separation']:.4f}")
    print(f"  Per-class intrinsic dim @ {int(PCA_VAR_TARGET*100)}% var:")
    for c, d in enumerate(metrics["intrinsic_dims"]):
        print(f"    {CIFAR10_CLASSES[c]:>10s}: {d}")

    Z = plot_geometry(metrics, X, y, figs_dir)

    save_metrics = {
        "head_final": float(final_head),
        "head_curve": head_accs,
        "within_mean": metrics["within_mean"],
        "within_std": metrics["within_std"],
        "between_mean": metrics["between_mean"],
        "between_std": metrics["between_std"],
        "separation": metrics["separation"],
        "intrinsic_dims": metrics["intrinsic_dims"],
        "centroid_sim": metrics["centroid_sim"].tolist(),
        "tsne_xy": Z.tolist(),
        "tsne_labels": y.tolist(),
        "binary_fraction": binary_frac,
    }
    out_json = results_dir / "attractor_geometry.json"
    out_json.write_text(json.dumps(save_metrics, indent=2))
    print(f"\nResults saved to {out_json}")

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"  Head accuracy:      {final_head:.4f}")
    print(f"  Within sim:         {metrics['within_mean']:.4f}")
    print(f"  Between sim:        {metrics['between_mean']:.4f}")
    print(f"  Separation:         {metrics['separation']:.4f}")
    print(f"  Mean intrinsic dim: {np.mean(metrics['intrinsic_dims']):.1f}")


if __name__ == "__main__":
    main()
