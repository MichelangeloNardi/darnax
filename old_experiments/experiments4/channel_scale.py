"""channel_scale.py

Scale the number of J1 channels (16 → 32 → 64) to improve representation quality.
Identical architecture and training loop to replicate_channel_entropy.py, but with
a --channels argument. Probe accuracy (not W_out head) is the primary metric.

Usage:
  python experiments4/channel_scale.py --channels 32
  python experiments4/channel_scale.py --channels 64
"""

from __future__ import annotations

import argparse
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

KSIZE        = 5
H_IMG, W_IMG = 32, 32
POOL         = 8
EPOCHS       = 20
SEEDS        = [0, 42, 123]   # 3 seeds for speed
PROBE_EPOCHS = 20
PROBE_WD     = 1.433e-4


def build_model(cfg: dict, key: jax.Array, C: int):
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
            2: ChannelWBack(10, H_IMG, W_IMG, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(
                pool=POOL, H=H_IMG, W=W_IMG, C_in=C, n_classes=10,
                strength=1.0, threshold=5.0,
                key=keys[3], lr=1.0, weight_decay=0.0,
            ),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H_IMG, W_IMG, 3), (H_IMG, W_IMG, C), 10])
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


def run_probe(trainer, ds: Cifar10, key: jax.Array, C: int) -> dict:
    n_pool = (H_IMG // POOL) * (W_IMG // POOL)  # 4*4 = 16 tiles
    probe_dim = n_pool * C

    def pool_j1(h):
        N = h.shape[0]
        return h.reshape(N, H_IMG // POOL, POOL, W_IMG // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)

    reps_tr, lbl_tr, reps_te, lbl_te = [], [], [], []
    for xb, yb in ds:
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_tr.append(pool_j1(np.array(trainer.state[1])))
        lbl_tr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_te.append(pool_j1(np.array(trainer.state[1])))
        lbl_te.append(np.argmax(np.array(yb), axis=-1))

    X_tr = torch.from_numpy(np.concatenate(reps_tr)).float()
    y_tr = torch.from_numpy(np.concatenate(lbl_tr)).long()
    X_te = torch.from_numpy(np.concatenate(reps_te)).float()
    y_te = torch.from_numpy(np.concatenate(lbl_te)).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe  = nn.Linear(probe_dim, 10, bias=False).to(device)
    opt_p  = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    crit   = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb_t, yb_t in loader:
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            opt_p.zero_grad(); crit(probe(xb_t), yb_t).backward(); opt_p.step()
        probe.eval()
        with torch.no_grad():
            tr = (probe(X_tr.to(device)).argmax(1) == y_tr.to(device)).float().mean().item()
            te = (probe(X_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
        train_accs.append(tr); test_accs.append(te)

    return {"train": train_accs, "test": test_accs,
            "best_train": max(train_accs), "best_test": max(test_accs)}


def train_one_seed(cfg: dict, ds: Cifar10, seed: int, C: int):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk, C)
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
    head_accs, probe_accs = [], []

    for epoch in range(1, EPOCHS + 1):
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

        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        head_acc = float(np.mean(batch_accs))
        head_accs.append(head_acc)

        probe_res = run_probe(trainer, ds, key, C)
        probe_accs.append(probe_res["best_test"])

        print(f"  C={C}  seed={seed}  epoch={epoch:2d}/{EPOCHS}"
              f"  head={head_acc:.4f}  probe={probe_res['best_test']:.4f}",
              flush=True)

    return head_accs, probe_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, required=True,
                        help="Number of J1 channels (e.g. 32 or 64)")
    args = parser.parse_args()
    C = args.channels

    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    print(f"Channel scaling experiment: C={C}, seeds={SEEDS}, epochs={EPOCHS}")
    print(f"Probe dim: {(H_IMG // POOL) * (W_IMG // POOL) * C} ({(H_IMG//POOL)*(W_IMG//POOL)}×{C})")

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    results_dir = HERE / "results"
    figs_dir    = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    all_head, all_probe = [], []
    for seed in SEEDS:
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
        h, p = train_one_seed(cfg, ds, seed, C)
        all_head.append(h)
        all_probe.append(p)

    all_head_np  = np.array(all_head)
    all_probe_np = np.array(all_probe)
    mean_h  = all_head_np.mean(axis=0);  std_h  = all_head_np.std(axis=0)
    mean_p  = all_probe_np.mean(axis=0); std_p  = all_probe_np.std(axis=0)
    epochs  = np.arange(1, EPOCHS + 1)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS  C={C}")
    print(f"  head  epoch {EPOCHS}: {mean_h[-1]:.4f} ± {std_h[-1]:.4f}")
    print(f"  probe epoch {EPOCHS}: {mean_p[-1]:.4f} ± {std_p[-1]:.4f}")
    print(f"  probe best:          {all_probe_np.max(axis=1).mean():.4f}")
    print(f"  baseline (C=16):     0.4501 probe (T21 reference)")

    # save results
    out = results_dir / f"channel_scale_C{C}.json"
    out.write_text(json.dumps({
        "channels": C, "seeds": SEEDS, "epochs": EPOCHS, "config": cfg,
        "head_accs":  all_head,
        "probe_accs": all_probe,
        "mean_probe_final": float(mean_p[-1]),
        "std_probe_final":  float(std_p[-1]),
    }, indent=2))
    print(f"\nSaved to {out}")

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colours = plt.cm.tab10(np.linspace(0, 0.5, len(SEEDS)))
    for (seed, curve), col in zip(zip(SEEDS, all_head), colours):
        ax1.plot(epochs, curve, color=col, alpha=0.5, label=f"seed {seed}")
    ax1.plot(epochs, mean_h, "k-", lw=2, label="mean")
    ax1.fill_between(epochs, mean_h - std_h, mean_h + std_h, alpha=0.15, color="k")
    ax1.set_title(f"Head accuracy  (C={C})"); ax1.set_xlabel("Epoch"); ax1.legend(fontsize=8)

    for (seed, curve), col in zip(zip(SEEDS, all_probe), colours):
        ax2.plot(epochs, curve, color=col, alpha=0.5, label=f"seed {seed}")
    ax2.plot(epochs, mean_p, "k-", lw=2, label="mean")
    ax2.fill_between(epochs, mean_p - std_p, mean_p + std_p, alpha=0.15, color="k")
    ax2.axhline(0.4501, color="grey", ls=":", lw=1.5, label="C=16 ref (0.45)")
    ax2.set_title(f"Probe accuracy  (C={C})"); ax2.set_xlabel("Epoch"); ax2.legend(fontsize=8)

    fig.suptitle(f"Channel scale C={C} — entropy rule, {len(SEEDS)} seeds", fontsize=10)
    fig.tight_layout()
    fig_path = figs_dir / f"channel_scale_C{C}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
