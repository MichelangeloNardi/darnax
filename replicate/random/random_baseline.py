"""win_init/random_baseline.py

Sanity-check baseline: Win and J fixed at random initialisation (never
updated).  Only the linear readout is trained.

Two eval protocols:
  standard  — Win active throughout (warmup + free)
  win_init  — Win active at t=0 only; free dynamics use Win-disabled orch

For each: train a linear probe (cross-entropy, Adam) and a Wout perceptron
(interleaved: trained on D) for N_EPOCHS epochs on fixed features.

Adapted from Matei's script for our repo layout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent.parent
CFG_PATH = ROOT / "replicate" / "best_channel_entropy_cfg.json"
sys.path.insert(0, str(ROOT / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.utils import scan_n

C, KSIZE     = 16, 5
H, W         = 32, 32
POOL         = 8
D_FEAT       = (H // POOL) * (W // POOL) * C   # 256
N_CLASSES    = 10
WARMUP_N     = 1
EVAL_N       = 5
MARGIN       = 5.0
N_EPOCHS     = 20          # Wout training epochs on fixed features
PROBE_EPOCHS = 50
PROBE_WD     = 1e-4
SEED         = 0

_STRIP = {"wback_type", "j1_window_hebb", "j1_entropy", "trial_number",
          "probe_acc", "c05_j1"}


def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0, key=keys[0],
                      padding_mode="constant", lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(channels=C, kernel_size=KSIZE, groups=1,
                      j_d=cfg["j_d"], threshold=cfg["threshold_j"], key=keys[1],
                      padding_mode="constant", lr=1.0, weight_decay=0.0,
                      entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                      strength=1.0, threshold=MARGIN, key=keys[3],
                      lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def to_hwc(xb): return jnp.asarray(xb.reshape(-1, H, W, 3) * 2.0 - 1.0)

def pool_np(j1):
    N = j1.shape[0]
    return (j1.reshape(N, H//POOL, POOL, W//POOL, POOL, C)
              .mean(axis=(2, 4)).reshape(N, -1))

def _zero_win(orch):
    return eqx.tree_at(lambda o: o.lmap[1][0].kernel, orch,
                       jnp.zeros_like(orch.lmap[1][0].kernel))


@eqx.filter_jit
def _get_D_std(orch, state_tmpl, x, y, key):
    """D state: warmup + free, Win active throughout, no label injection."""
    state = state_tmpl.init(x, y)
    (state, key), _ = scan_n(orch.step, (state, key), WARMUP_N,
                             filter_messages="forward")
    (state, key), _ = scan_n(orch.step, (state, key), EVAL_N,
                             filter_messages="forward")
    return state[1], key


@eqx.filter_jit
def _get_D_win_init(orch, state_tmpl, x, y, key):
    """D state: warmup with Win, then free with Win zeroed (pure J1 dynamics)."""
    state = state_tmpl.init(x, y)
    (state, key), _ = scan_n(orch.step, (state, key), WARMUP_N,
                             filter_messages="forward")
    nw = _zero_win(orch)
    (state, key), _ = scan_n(nw.step, (state, key), EVAL_N,
                             filter_messages="forward")
    return state[1], key


def collect_features(get_D, orch, state_tmpl, ds, key):
    rtr, ltr, rte, lte = [], [], [], []
    for xb, yb in ds:
        j1, key = get_D(orch, state_tmpl, to_hwc(xb), yb, key)
        rtr.append(pool_np(np.array(j1)))
        ltr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        j1, key = get_D(orch, state_tmpl, to_hwc(xb), yb, key)
        rte.append(pool_np(np.array(j1)))
        lte.append(np.argmax(np.array(yb), axis=-1))
    return (np.concatenate(rtr), np.concatenate(ltr),
            np.concatenate(rte), np.concatenate(lte)), key


def run_probe(X_tr, y_tr, X_te, y_te):
    dev = torch.device("cpu")
    Xtr_t = torch.from_numpy(X_tr).float().to(dev)
    ytr_t = torch.from_numpy(y_tr).long().to(dev)
    Xte_t = torch.from_numpy(X_te).float().to(dev)
    yte_t = torch.from_numpy(y_te).long().to(dev)
    probe = nn.Linear(D_FEAT, 10, bias=False).to(dev)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(Xtr_t.cpu(), ytr_t.cpu()),
                        batch_size=256, shuffle=True)
    crit  = nn.CrossEntropyLoss()
    best  = 0.0
    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            acc = (probe(Xte_t).argmax(1) == yte_t).float().mean().item()
        best = max(best, acc)
    return best


def run_wout_perceptron(X_tr, y_tr_oh, X_te, y_te, n_epochs, lr):
    """Train a ±1 perceptron on fixed features for n_epochs."""
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((N_CLASSES, D_FEAT)).astype(np.float32) * 0.01
    n_batches = max(1, len(X_tr) // 32)
    best = 0.0
    for ep in range(n_epochs):
        idx = rng.permutation(len(X_tr))
        for b in range(n_batches):
            sl  = idx[b*32:(b+1)*32]
            xp  = X_tr[sl]; y = y_tr_oh[sl]
            yh  = xp @ W.T
            err = (y * yh <= MARGIN).astype(np.float32)
            grad = -(err * y).T @ xp / np.sqrt(len(sl) * D_FEAT)
            W   -= lr * grad
        pred = (X_te @ W.T).argmax(1)
        acc  = (pred == y_te).mean()
        best = max(best, float(acc))
        if (ep + 1) % 5 == 0:
            print(f"    wout ep {ep+1:2d}/{n_epochs}  acc={acc:.4f}  best={best:.4f}", flush=True)
    return best


def main():
    with open(CFG_PATH) as f:
        cfg = {k: v for k, v in json.load(f).items() if k not in _STRIP}

    print("Config:", {k: round(v,4) if isinstance(v,float) else v for k,v in cfg.items()})

    ds = Cifar10(batch_size=32, x_transform="identity",
                 label_mode="pm1", linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state_tmpl, orch = build_model(cfg, mk)   # random init, never updated

    optuna_overrides = {"j_d": 0.852, "strength_back": 0.839}
    cfg_optuna = {**cfg, **optuna_overrides}
    _, orch_optuna = build_model(cfg_optuna, mk)   # same kernel init, different j_d/sb

    results = {}

    conditions = [
        ("standard",     _get_D_std,      orch,        cfg),
        ("win_init",     _get_D_win_init, orch,        cfg),
        ("win_init_opt", _get_D_win_init, orch_optuna, cfg_optuna),
    ]

    for name, get_D, orch_c, cfg_c in conditions:
        print(f"\n── {name}  (j_d={cfg_c['j_d']:.3f}  sb={cfg_c['strength_back']:.3f}) ──", flush=True)

        print("  collecting features...", flush=True)
        (X_tr, y_tr, X_te, y_te), key = collect_features(
            get_D, orch_c, state_tmpl, ds, key)

        y_tr_oh = np.full((len(y_tr), N_CLASSES), -1.0, dtype=np.float32)
        y_tr_oh[np.arange(len(y_tr)), y_tr] = 1.0

        print("  running probe...", flush=True)
        probe_acc = run_probe(X_tr, y_tr, X_te, y_te)

        print("  running wout perceptron...", flush=True)
        wout_acc  = run_wout_perceptron(X_tr, y_tr_oh, X_te, y_te,
                                        N_EPOCHS, lr=float(cfg_c["lr_wout"]))

        print(f"  probe = {probe_acc:.4f}", flush=True)
        print(f"  wout  = {wout_acc:.4f}", flush=True)
        results[name] = {"probe": probe_acc, "wout": wout_acc,
                         "j_d": cfg_c["j_d"], "strength_back": cfg_c["strength_back"]}

    print(f"\n{'='*55}")
    print(f"  {'protocol':14s}  {'probe':>8}  {'wout':>8}")
    for name, r in results.items():
        print(f"  {name:14s}  {r['probe']:8.4f}  {r['wout']:8.4f}")
    print(f"{'='*55}")

    out = HERE / "random_baseline_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nJSON → {out}")


if __name__ == "__main__":
    main()
