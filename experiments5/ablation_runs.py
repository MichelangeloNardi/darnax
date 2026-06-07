"""ablation_runs.py

Three ablation training variants for the channel-entropy CIFAR-10 architecture.
Single seed (default: 0), 10 epochs, best config.

Modes
-----
  offline_wout    — Win + J1 train normally; Wout is frozen (lr=0) throughout
                    the 10 epochs. After training, Wout is optimised offline via
                    Adam+CE on the fixed final J1 representations and the model
                    is evaluated with the new weights. Per-epoch head_acc will be
                    near-random (frozen Wout); per-epoch probe_acc tracks J1 quality.

  random_baseline — Win and J1 frozen at random initialisation; only Wout trains
                    (perceptron rule, online). Proper random-feature baseline: how
                    well can a linear head do on random convolutional features?

  j_only          — Win frozen at random init; J1 + Wout train normally.
                    Measures the marginal benefit of learning J1 when Win is random.

NEW CODE vs standard_run.py
---------------------------
  make_optimizer()  — same function as in standard_run.py but lr_win / lr_j /
                      lr_wout are set to 0.0 for frozen modules, which routes
                      them through optax.set_to_zero() (no weight update, no
                      momentum state allocated).

  train_epoch()     — unchanged helper from standard_run.py; the update_win /
                      update_j1 flags skip Win normalisation and kernel decay for
                      frozen modules so they stay exactly at their random init.

  The combination of lr=0.0 in the optimizer AND update_win/update_j1=False in
  train_epoch is what implements true module freezing: the optimizer touches
  nothing, and the out-of-optimizer operations (normalisation, decay) are also
  skipped.

  For offline_wout: after the training loop, run_probe() is called once on the
  final J1 representations and the resulting linear-layer weights are written
  back into trainer.orchestrator.lmap[2][1].W (the actual Wout matrix in the
  JAX model). The model is then re-evaluated to confirm the transferred accuracy.

Results saved to:
  experiments5/results/ablation_{mode}.json

Usage:
  python experiments5/ablation_runs.py --mode offline_wout
  python experiments5/ablation_runs.py --mode random_baseline
  python experiments5/ablation_runs.py --mode j_only
  python experiments5/ablation_runs.py --mode all   # run all three sequentially
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import equinox as eqx
import jax, jax.numpy as jnp, jax.tree_util as jtu
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

# --- per-mode flags -----------------------------------------------------------
# update_win  : apply Win normalisation per-batch and Win kernel decay per-epoch
# update_j1   : apply J1 kernel decay per-epoch
# lr_wout=0   : Wout frozen via optimizer (set_to_zero)
MODE_FLAGS = {
    #                       lr_win  lr_j   lr_wout  update_win  update_j1
    "offline_wout":    dict(win=1,  j1=1,  wout=0,  norm_win=True,  decay_j1=True),
    "random_baseline": dict(win=0,  j1=0,  wout=1,  norm_win=False, decay_j1=False),
    "j_only":          dict(win=0,  j1=1,  wout=1,  norm_win=False, decay_j1=True),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Optimizer — the key difference from standard_run.py
# ---------------------------------------------------------------------------

def make_optimizer(orchestrator, cfg, lr_win_active, lr_j_active, lr_wout_active):
    """Build multi-transform SGD optimizer with module-level freeze control.

    Passing a *_active=0 routes that module through optax.set_to_zero() so
    no update is applied and no momentum state is allocated.  Passing 1 uses
    the learning rate from cfg as normal.
    """
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

    def tx_win():
        return sgd(-cfg["lr_win"]) if lr_win_active else optax.set_to_zero()

    def tx_j1():
        return sgd(-cfg["lr_j"]) if lr_j_active else optax.set_to_zero()

    def tx_wout():
        return sgd(cfg["lr_wout"]) if lr_wout_active else optax.set_to_zero()

    opt = optax.multi_transform({
        "default": optax.set_to_zero(),
        "win":  tx_win(),
        "j1":   tx_j1(),
        "wout": tx_wout(),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


# ---------------------------------------------------------------------------
# Training utilities (identical to standard_run.py)
# ---------------------------------------------------------------------------

def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


def run_probe(trainer, ds, key, return_weights=False):
    """Collect J1 reps, train linear probe (Adam+CE), return best test acc.

    If return_weights=True, also returns the trained weight matrix (256, 10)
    so it can be written back into the JAX Wout.
    """
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

    best_test, best_epoch_W = 0.0, None
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
        if te > best_test:
            best_test = te
            best_epoch_W = probe.weight.detach().cpu().numpy()  # (10, 256)

    if return_weights:
        W_jax = best_epoch_W.T  # (256, 10) — matches FullyConnected.W shape
        return best_test, W_jax
    return best_test


def train_epoch(trainer, ds, cfg, key, norm_win=True, decay_j1=True):
    """One training epoch with conditional Win normalisation and kernel decay.

    norm_win=False : skip per-batch Win normalisation and per-epoch Win decay
                     (used when Win is frozen at random init)
    decay_j1=False : skip per-epoch J1 kernel decay (used when J1 is frozen)
    """
    decay = cfg["kernel_decay_rate"]
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)
        if norm_win:
            win_k = trainer.orchestrator.lmap[1][0].kernel
            kh, kw, ci, co = win_k.shape
            flat   = win_k.reshape(-1, co)
            normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            trainer.orchestrator = eqx.tree_at(
                lambda o: o.lmap[1][0].kernel,
                trainer.orchestrator,
                normed.reshape(kh, kw, ci, co),
            )
    if norm_win:
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[1][0].kernel,
            trainer.orchestrator,
            trainer.orchestrator.lmap[1][0].kernel * (1.0 - decay),
        )
    if decay_j1:
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[1][1].kernel,
            trainer.orchestrator,
            trainer.orchestrator.lmap[1][1].kernel * (1.0 - decay),
        )
    return trainer, key


def eval_head(trainer, ds, key):
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


# ---------------------------------------------------------------------------
# Per-mode training
# ---------------------------------------------------------------------------

def run_mode(mode, cfg, ds):
    flags = MODE_FLAGS[mode]
    print(f"\n{'='*60}")
    print(f"MODE: {mode}")
    print(f"  Win trains: {bool(flags['win'])}  |  J1 trains: {bool(flags['j1'])}  "
          f"|  Wout trains: {bool(flags['wout'])}")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(
        orch, cfg,
        lr_win_active=flags["win"],
        lr_j_active=flags["j1"],
        lr_wout_active=flags["wout"],
    )
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
        trainer, key = train_epoch(
            trainer, ds, cfg, key,
            norm_win=flags["norm_win"],
            decay_j1=flags["decay_j1"],
        )
        head_acc, key = eval_head(trainer, ds, key)
        probe_acc     = run_probe(trainer, ds, key)
        head_accs.append(head_acc)
        probe_accs.append(probe_acc)
        print(f"  epoch={epoch:2d}  head={head_acc:.4f}  probe={probe_acc:.4f}", flush=True)

    result = {
        "mode": mode,
        "seed": SEED,
        "epochs": EPOCHS,
        "head_accs": head_accs,
        "probe_accs": probe_accs,
    }

    # For offline_wout: after training, optimise Wout on fixed final J1 reps,
    # write the weights back into the JAX model, and re-evaluate the head.
    if mode == "offline_wout":
        print("\n  [offline_wout] Training Wout offline on final J1 representations...")
        offline_acc, W_opt = run_probe(trainer, ds, key, return_weights=True)
        # Transfer trained weights into the JAX Wout (PooledFlattenFC.W)
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[2][1].W,
            trainer.orchestrator,
            jnp.array(W_opt),
        )
        offline_head_acc, _ = eval_head(trainer, ds, key)
        print(f"  offline_wout probe acc={offline_acc:.4f}  "
              f"model head after transfer={offline_head_acc:.4f}")
        result["offline_wout_probe_acc"]  = offline_acc
        result["offline_wout_head_acc"]   = offline_head_acc

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=list(MODE_FLAGS) + ["all"],
                        help="Which ablation to run (default: all)")
    args = parser.parse_args()

    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    modes = list(MODE_FLAGS) if args.mode == "all" else [args.mode]
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    for mode in modes:
        result = run_mode(mode, cfg, ds)
        out_path = results_dir / f"ablation_{mode}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
