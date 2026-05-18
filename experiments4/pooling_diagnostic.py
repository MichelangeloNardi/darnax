"""pooling_diagnostic.py

Tests whether the 8×8 avg-pool in the head is the bottleneck behind
the head vs probe gap (~22% head vs ~45% probe).

After training the standard channel-entropy model, collects the J1
representation state[1] of shape (N, 32, 32, 16). Trains an Adam
linear probe at each pool level POOL ∈ {1, 2, 4, 8}:

  POOL=8 →  4×4×16 =   256d   (current head input)
  POOL=4 →  8×8×16 =  1024d
  POOL=2 → 16×16×16 = 4096d
  POOL=1 → 32×32×16 = 16384d  (no spatial pooling)

If POOL=1 probe ≫ POOL=8 probe, the head's avg-pool is the
bottleneck — and keeping spatial info beats scaling channels.

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/pooling_diagnostic.py
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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
POOL_HEAD = 8
EPOCHS = 20
SEED = 0
PROBE_EPOCHS = 20
PROBE_WD = 1.433e-4
POOLS = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Model / optimizer / preprocessing
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
                pool=POOL_HEAD, H=H, W=W, C_in=C, n_classes=10,
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


def pool_features(h: np.ndarray, pool: int) -> np.ndarray:
    """h: (N, 32, 32, 16) → (N, (32/pool)² × 16) features.

    pool=1 returns the raw flattened state. Larger pool = more spatial averaging.
    """
    N = h.shape[0]
    if pool == 1:
        return h.reshape(N, -1)
    return h.reshape(N, H // pool, pool, W // pool, pool, C).mean(axis=(2, 4)).reshape(N, -1)


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
# Representation collection + probes
# ---------------------------------------------------------------------------

def collect_full_states(trainer, ds: Cifar10, key: jax.Array):
    """Return (X_tr, y_tr, X_te, y_te, key) where X is the full (N, 32, 32, 16) state[1]."""
    reps_tr, lbl_tr, reps_te, lbl_te = [], [], [], []
    for xb, yb in ds:
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_tr.append(np.array(trainer.state[1]))
        lbl_tr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_te.append(np.array(trainer.state[1]))
        lbl_te.append(np.argmax(np.array(yb), axis=-1))
    return (
        np.concatenate(reps_tr), np.concatenate(lbl_tr),
        np.concatenate(reps_te), np.concatenate(lbl_te),
        key,
    )


def train_probe(X_tr: np.ndarray, y_tr: np.ndarray,
                X_te: np.ndarray, y_te: np.ndarray,
                n_epochs: int = PROBE_EPOCHS) -> list[float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = X_tr.shape[1]
    probe = nn.Linear(d, 10, bias=False).to(device)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    X_te_t = torch.from_numpy(X_te).float().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)
    crit = nn.CrossEntropyLoss()

    accs: list[float] = []
    for _ in range(n_epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            te = (probe(X_te_t).argmax(1) == y_te_t).float().mean().item()
        accs.append(te)
    return accs


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

    print("\nCollecting J1 states ...")
    X_tr_full, y_tr, X_te_full, y_te, key = collect_full_states(trainer, ds, key)
    print(f"  train: {X_tr_full.shape}  test: {X_te_full.shape}")

    results: dict = {
        "head_final": float(final_head),
        "head_curve": head_accs,
        "probe_per_pool": {},
    }

    for pool in POOLS:
        X_tr = pool_features(X_tr_full, pool)
        X_te = pool_features(X_te_full, pool)
        print(f"\nProbe at POOL={pool}  (dim={X_tr.shape[1]}) ...")
        accs = train_probe(X_tr, y_tr, X_te, y_te)
        best = max(accs)
        results["probe_per_pool"][str(pool)] = {
            "dim": int(X_tr.shape[1]),
            "best": float(best),
            "curve": accs,
        }
        print(f"  best probe accuracy: {best:.4f}")

    out_json = results_dir / "pooling_diagnostic.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_json}")

    # --- plot ---
    pool_values = sorted(POOLS, reverse=True)  # 8, 4, 2, 1
    dims = [results["probe_per_pool"][str(p)]["dim"] for p in pool_values]
    bests = [results["probe_per_pool"][str(p)]["best"] for p in pool_values]
    labels = [f"POOL={p}\n({d}d)" for p, d in zip(pool_values, dims)]

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(13, 5))

    ax_bar.bar(range(len(pool_values)), bests, color="steelblue", alpha=0.85)
    ax_bar.axhline(final_head, color="red", linestyle="--",
                   label=f"head ({final_head:.3f})")
    ax_bar.set_xticks(range(len(pool_values)))
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("Test accuracy")
    ax_bar.set_title("Probe accuracy vs pool level")
    for i, b in enumerate(bests):
        ax_bar.text(i, b + 0.005, f"{b:.3f}", ha="center", fontsize=10)
    ax_bar.legend()

    for p in pool_values:
        r = results["probe_per_pool"][str(p)]
        ax_curve.plot(range(1, PROBE_EPOCHS + 1), r["curve"],
                      label=f"POOL={p} ({r['dim']}d)", linewidth=1.5)
    ax_curve.set_xlabel("Probe epoch")
    ax_curve.set_ylabel("Test accuracy")
    ax_curve.set_title("Probe training curves")
    ax_curve.legend(fontsize=9)

    fig.suptitle(f"Pooling diagnostic — C={C}, seed={SEED}, {EPOCHS} epochs", fontsize=11)
    fig.tight_layout()
    fig_path = figs_dir / "pooling_diagnostic.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {fig_path}")

    # --- summary ---
    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"  Head accuracy:        {final_head:.4f}")
    for p in pool_values:
        r = results["probe_per_pool"][str(p)]
        print(f"  Probe POOL={p}  ({r['dim']:>5}d): {r['best']:.4f}")


if __name__ == "__main__":
    main()
