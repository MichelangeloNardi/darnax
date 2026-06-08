"""standard_run.py

Standard channel-entropy training on CIFAR-10.
Win, J1, Wout all train online. 5 seeds x 10 epochs, best config.

Per-epoch: head accuracy (Wout perceptron rule) + linear probe on J1 reps.

Saves to:
  experiments5/results/standard.json   — per-seed and mean±std results
  experiments5/figures/standard.png    — mean±std accuracy curves

Run on cluster:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments5/standard_run.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import equinox as eqx
import jax, jax.numpy as jnp, jax.tree_util as jtu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax, torch
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

SEEDS        = [0, 42, 123, 7, 999]
EPOCHS       = 10
C, KSIZE     = 16, 5
H, W, POOL   = 32, 32, 8
PROBE_EPOCHS = 20
PROBE_WD     = 1.433e-4


def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0, key=keys[0],
                      padding_mode="constant", lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                               strength=1.0, threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=layer_map)


def make_optimizer(orchestrator, cfg):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(lambda m, r=i, c=j: m.lmap[r][c], labels,
                             replace=like(params.lmap[i][j], lbl))

    opt = optax.multi_transform({
        "default": optax.set_to_zero(),
        "win":  sgd(-cfg["lr_win"]),
        "j1":   sgd(-cfg["lr_j"]),
        "wout": sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


def run_probe(trainer, ds, key):
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
    probe  = nn.Linear(256, 10, bias=False).to(device)
    opt_p  = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    crit   = nn.CrossEntropyLoss()

    best_test = 0.0
    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb_t, yb_t in loader:
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            opt_p.zero_grad()
            crit(probe(xb_t), yb_t).backward()
            opt_p.step()
        probe.eval()
        with torch.no_grad():
            te = (probe(X_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
        best_test = max(best_test, te)

    return best_test


def train_epoch(trainer, ds, cfg, key):
    decay = cfg["kernel_decay_rate"]
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)
    for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
        trainer.orchestrator = eqx.tree_at(
            path, trainer.orchestrator,
            path(trainer.orchestrator) * (1.0 - decay),
        )
    return trainer, key


def eval_head(trainer, ds, key):
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def run_one_seed(seed, cfg, ds):
    print(f"\n{'='*50}\nSeed {seed}\n{'='*50}", flush=True)
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
    head_accs, probe_accs = [], []
    for epoch in range(1, EPOCHS + 1):
        trainer, key = train_epoch(trainer, ds, cfg, key)
        head_acc, key = eval_head(trainer, ds, key)
        probe_acc = run_probe(trainer, ds, key)
        head_accs.append(head_acc)
        probe_accs.append(probe_acc)
        print(f"  seed={seed}  epoch={epoch:2d}/{EPOCHS}  head={head_acc:.4f}  probe={probe_acc:.4f}", flush=True)
    return {"seed": seed, "head_accs": head_accs, "probe_accs": probe_accs}


def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    print("Config:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cfg.items()}, flush=True)

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    per_seed = [run_one_seed(s, cfg, ds) for s in SEEDS]

    head_mat  = np.array([r["head_accs"]  for r in per_seed])  # (n_seeds, epochs)
    probe_mat = np.array([r["probe_accs"] for r in per_seed])

    out = {
        "seeds":       SEEDS,
        "epochs":      EPOCHS,
        "per_seed":    per_seed,
        "head_mean":   head_mat.mean(0).tolist(),
        "head_std":    head_mat.std(0).tolist(),
        "probe_mean":  probe_mat.mean(0).tolist(),
        "probe_std":   probe_mat.std(0).tolist(),
    }

    print(f"\nHead  final: mean={head_mat[:, -1].mean():.4f}  std={head_mat[:, -1].std():.4f}", flush=True)
    print(f"Probe final: mean={probe_mat[:, -1].mean():.4f}  std={probe_mat[:, -1].std():.4f}", flush=True)

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    out_path = results_dir / "standard.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {out_path}", flush=True)

    ep = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    hm, hs = head_mat.mean(0), head_mat.std(0)
    pm, ps = probe_mat.mean(0), probe_mat.std(0)
    ax.plot(ep, hm, "-o", color="#2563EB", label="Head (Wout, perceptron rule)")
    ax.fill_between(ep, hm - hs, hm + hs, alpha=0.2, color="#2563EB")
    ax.plot(ep, pm, "-s", color="#EA580C", label="Linear probe (J1, Adam)")
    ax.fill_between(ep, pm - ps, pm + ps, alpha=0.2, color="#EA580C")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_title(f"Standard training — {len(SEEDS)} seeds ± 1 std")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = figures_dir / "standard.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {fig_path}", flush=True)


if __name__ == "__main__":
    main()
