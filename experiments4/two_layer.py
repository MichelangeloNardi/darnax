"""two_layer.py

Two-layer conv architecture:
  - Layer 1: Conv2D(3→C1) + Conv2DRecurrentDiscrete(C1) — spatial 32×32
  - Layer 2: Conv2D(C1→C2) + Conv2DRecurrentDiscrete(C2) — spatial 16×16 (downsampled)
  - Inter-layer: MajorityPooling(stride=2) forward, ConstantUnpooling backward (both frozen)
  - Feedback: ChannelWBack(10→C2) only to layer 2
  - Output: PooledFlattenFC on layer 2 representations

Baseline: C1=C2=16. Probe reads from layer 2.

Run from repo root:
  python experiments4/two_layer.py
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
from darnax.modules.conv.pooling import MajorityPooling, ConstantUnpooling
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

C1, C2   = 16, 16     # channels in layer 1 and layer 2
KSIZE    = 5
H1, W1   = 32, 32     # layer 1 spatial size
H2, W2   = 16, 16     # layer 2 spatial size (downsampled 2x)
POOL     = 4          # avg-pool for probe: 16/4 = 4x4 tiles → 4*4*C2 features
EPOCHS   = 20
SEEDS    = [0, 42, 123]
PROBE_EPOCHS = 20
PROBE_WD     = 1.433e-4


def build_model(cfg: dict, key: jax.Array):
    keys = jax.random.split(key, 8)

    layer_map = LayerMap.from_dict({
        1: {
            # feedforward from input
            0: Conv2D(
                in_channels=3, out_channels=C1, kernel_size=KSIZE,
                threshold=cfg["threshold_win"], strength=1.0,
                key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.0,
            ),
            # recurrent self-connections
            1: Conv2DRecurrentDiscrete(
                channels=C1, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            # feedback from layer 2 (upsampled, frozen)
            2: ConstantUnpooling(kernel_size=2, strength=cfg["strength_back"]),
        },
        2: {
            # feedforward from layer 1 (downsampled, frozen)
            1: MajorityPooling(kernel_size=2, strength=1.0, key=keys[2], stride=2),
            # recurrent self-connections
            2: Conv2DRecurrentDiscrete(
                channels=C2, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[3], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            # feedback from output layer
            3: ChannelWBack(10, H2, W2, C2, cfg["strength_back"], keys[4]),
        },
        3: {
            # readout from layer 2
            2: PooledFlattenFC(
                pool=POOL, H=H2, W=W2, C_in=C2, n_classes=10,
                strength=1.0, threshold=5.0,
                key=keys[5], lr=1.0, weight_decay=0.0,
            ),
            3: OutputLayer(),
        },
    })

    state = SequentialState([
        (H1, W1, 3),    # layer 0: input
        (H1, W1, C1),   # layer 1: first hidden
        (H2, W2, C2),   # layer 2: second hidden
        10,             # layer 3: output
    ])
    return state, SequentialOrchestrator(layers=layer_map)


def make_optimizer(orchestrator, cfg: dict):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 2), "j2"), ((3, 2), "wout")]:
        labels = eqx.tree_at(
            lambda m, r=i, c=j: m.lmap[r][c], labels,
            replace=like(params.lmap[i][j], lbl),
        )

    opt = optax.multi_transform({
        "default": optax.sgd(0.0),   # frozen (ChannelWBack, pooling)
        "win":     sgd(-cfg["lr_win"]),
        "j1":      sgd(-cfg["lr_j"]),
        "j2":      sgd(-cfg["lr_j"]),
        "wout":    sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def run_probe(trainer, ds: Cifar10, key: jax.Array) -> dict:
    probe_dim = (H2 // POOL) * (W2 // POOL) * C2  # 4*4*16 = 256

    def pool_j2(h):
        N = h.shape[0]
        return h.reshape(N, H2 // POOL, POOL, W2 // POOL, POOL, C2).mean(axis=(2, 4)).reshape(N, -1)

    reps_tr, lbl_tr, reps_te, lbl_te = [], [], [], []
    for xb, yb in ds:
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_tr.append(pool_j2(np.array(trainer.state[2])))
        lbl_tr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_te.append(pool_j2(np.array(trainer.state[2])))
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


def train_one_seed(cfg: dict, ds: Cifar10, seed: int):
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
    head_accs, probe_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
            # normalize Win filters
            win_k = trainer.orchestrator.lmap[1][0].kernel
            kh, kw, ci, co = win_k.shape
            flat   = win_k.reshape(-1, co)
            normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            trainer.orchestrator = eqx.tree_at(
                lambda o: o.lmap[1][0].kernel,
                trainer.orchestrator,
                normed.reshape(kh, kw, ci, co),
            )

        # kernel decay on Win, J1, J2
        for path in [
            lambda o: o.lmap[1][0].kernel,
            lambda o: o.lmap[1][1].kernel,
            lambda o: o.lmap[2][2].kernel,
        ]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay)
            )

        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        head_acc = float(np.mean(batch_accs))
        head_accs.append(head_acc)

        probe_res = run_probe(trainer, ds, key)
        probe_accs.append(probe_res["best_test"])

        print(f"  2L  seed={seed}  epoch={epoch:2d}/{EPOCHS}"
              f"  head={head_acc:.4f}  probe={probe_res['best_test']:.4f}", flush=True)

    return head_accs, probe_accs


def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    print(f"Two-layer architecture: C1={C1}, C2={C2}, seeds={SEEDS}, epochs={EPOCHS}")
    print(f"Layer 1: (32,32,{C1})  Layer 2: (16,16,{C2})  Probe dim: {(H2//POOL)*(W2//POOL)*C2}")

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
        h, p = train_one_seed(cfg, ds, seed)
        all_head.append(h)
        all_probe.append(p)

    all_head_np  = np.array(all_head)
    all_probe_np = np.array(all_probe)
    mean_h = all_head_np.mean(axis=0);  std_h = all_head_np.std(axis=0)
    mean_p = all_probe_np.mean(axis=0); std_p = all_probe_np.std(axis=0)
    epochs = np.arange(1, EPOCHS + 1)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — 2-layer C1={C1} C2={C2}")
    print(f"  probe epoch {EPOCHS}: {mean_p[-1]:.4f} ± {std_p[-1]:.4f}")
    print(f"  probe best:          {all_probe_np.max(axis=1).mean():.4f}")
    print(f"  1-layer baseline:    0.4501 (C=16, T21 reference)")

    out = results_dir / "two_layer.json"
    out.write_text(json.dumps({
        "C1": C1, "C2": C2, "seeds": SEEDS, "epochs": EPOCHS, "config": cfg,
        "head_accs": all_head, "probe_accs": all_probe,
        "mean_probe_final": float(mean_p[-1]),
        "std_probe_final":  float(std_p[-1]),
    }, indent=2))
    print(f"Saved to {out}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colours = plt.cm.tab10(np.linspace(0, 0.5, len(SEEDS)))
    for (seed, curve), col in zip(zip(SEEDS, all_head), colours):
        ax1.plot(epochs, curve, color=col, alpha=0.5, label=f"seed {seed}")
    ax1.plot(epochs, mean_h, "k-", lw=2, label="mean")
    ax1.fill_between(epochs, mean_h - std_h, mean_h + std_h, alpha=0.15, color="k")
    ax1.set_title(f"Head accuracy — 2L C1={C1} C2={C2}"); ax1.set_xlabel("Epoch"); ax1.legend(fontsize=8)

    for (seed, curve), col in zip(zip(SEEDS, all_probe), colours):
        ax2.plot(epochs, curve, color=col, alpha=0.5, label=f"seed {seed}")
    ax2.plot(epochs, mean_p, "k-", lw=2, label="mean")
    ax2.fill_between(epochs, mean_p - std_p, mean_p + std_p, alpha=0.15, color="k")
    ax2.axhline(0.4501, color="grey", ls=":", lw=1.5, label="1L C=16 ref (0.45)")
    ax2.set_title(f"Probe accuracy — 2L C1={C1} C2={C2}"); ax2.set_xlabel("Epoch"); ax2.legend(fontsize=8)

    fig.suptitle(f"Two-layer conv — entropy rule, {len(SEEDS)} seeds", fontsize=10)
    fig.tight_layout()
    fig_path = figs_dir / "two_layer.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
