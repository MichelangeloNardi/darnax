"""standard_run.py

Standard channel-entropy training on CIFAR-10.
Win, J1, Wout all train online. Single seed, 10 epochs, best config.

Per-epoch: head accuracy (Wout perceptron rule) + linear probe on J1 reps.

Saves to:
  experiments5/results/standard.json   — per-epoch head_accs and probe_accs
  experiments5/figures/standard.png    — accuracy curves

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

SEED         = 0
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


def make_optimizer(orchestrator, cfg, lr_win=None, lr_j=None, lr_wout=None):
    """Build multi-transform SGD. Pass 0.0 to freeze a module, None to use cfg default."""
    lr_win  = cfg["lr_win"]  if lr_win  is None else lr_win
    lr_j    = cfg["lr_j"]    if lr_j    is None else lr_j
    lr_wout = cfg["lr_wout"] if lr_wout is None else lr_wout

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

    # optax.set_to_zero() is memory-efficient for frozen modules (no momentum state)
    def make_tx(lr, sign):
        if lr == 0.0:
            return optax.set_to_zero()
        return sgd(sign * lr)

    opt = optax.multi_transform({
        "default": optax.set_to_zero(),
        "win":  make_tx(lr_win,  -1.0),   # local rules return positive updates; SGD subtracts
        "j1":   make_tx(lr_j,   -1.0),
        "wout": make_tx(lr_wout, +1.0),   # perceptron rule already returns negative update
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


def run_probe(trainer, ds, key):
    """Collect J1 reps on full train+test set, train linear probe, return best test acc."""
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


def train_epoch(trainer, ds, cfg, key, update_win=True, update_j1=True):
    """One training epoch. Only applies Win normalization and kernel decay for active modules."""
    decay = cfg["kernel_decay_rate"]
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)
    if update_win:
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[1][0].kernel,
            trainer.orchestrator,
            trainer.orchestrator.lmap[1][0].kernel * (1.0 - decay),
        )
    if update_j1:
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[1][1].kernel,
            trainer.orchestrator,
            trainer.orchestrator.lmap[1][1].kernel * (1.0 - decay),
        )
    return trainer, key


def eval_head(trainer, ds, key):
    """Evaluate model head (Wout) accuracy on the test set."""
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    print("Config:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cfg.items()})

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(orch, cfg)  # all modules active

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
        trainer, key = train_epoch(trainer, ds, cfg, key, update_win=True, update_j1=True)
        head_acc, key = eval_head(trainer, ds, key)
        probe_acc     = run_probe(trainer, ds, key)
        head_accs.append(head_acc)
        probe_accs.append(probe_acc)
        print(f"epoch={epoch:2d}  head={head_acc:.4f}  probe={probe_acc:.4f}", flush=True)

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    out = {"seed": SEED, "epochs": EPOCHS, "head_accs": head_accs, "probe_accs": probe_accs}
    out_path = results_dir / "standard.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")

    ep = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, head_accs,  "-o", label="W_out (head, perceptron rule)")
    ax.plot(ep, probe_accs, "-s", label="Linear probe (J1 reps, Adam)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy")
    ax.set_title(f"Standard training — seed={SEED}, {EPOCHS} epochs")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = figures_dir / "standard.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
