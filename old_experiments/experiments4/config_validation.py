"""config_validation.py

Quick head-accuracy validation of three candidate configs (Matei probe-tuned,
Kassym probe-tuned, Matei W_out-tuned). Runs 3 epochs, single seed, no probe.

  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/config_validation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
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
POOL = 8
EPOCHS = 3
SEED = 0

CONFIGS = {
    "1_matei_probe": {
        "lr_j": 0.001353138590536092,
        "lr_wout": 0.7400321650885163,
        "lr_win": 0.06079550019033205,
        "kernel_decay_rate": 0.01438343649118462,
        "threshold_j": 3.3222342758629004,
        "threshold_win": 1.4037416864067074,
        "j_d": 0.7492062110663401,
        "entropy_beta": 0.7312213411315246,
        "momentum": 0.6062485647259589,
        "wback_type": "conv1x1",
        "wback_strength": 0.8973904603571834,
        "clamped_n_iter": 4,
        "free_n_iter": 13,
        "warmup_n_iter": 1,
    },
    "2_kassym_probe": {
        "lr_j": 0.0009740402572938124,
        "lr_win": 0.02261680399040041,
        "lr_wout": 0.04321887433959399,
        "kernel_decay_rate": 0.0007847404026606365,
        "threshold_j": 1.9796175431992729,
        "threshold_win": 0.8510825401105292,
        "j_d": 0.8953837335689401,
        "entropy_beta": 0.3449405979902655,
        "momentum": 0.3162081708642795,
        "clamped_n_iter": 5,
        "free_n_iter": 6,
        "strength_back": 1.4711531428803912,
        "wback_type": "channel",
        "warmup_n_iter": 1,
    },
    "3_matei_wout": {
        "lr_win": 0.0014704900068512122,
        "lr_j": 0.0001640857799594289,
        "lr_wout": 0.0046199902170752346,
        "threshold_win": 0.3020596991424575,
        "threshold_j": 0.831821321919564,
        "j_d": 0.8928072670420131,
        "entropy_beta": 0.0160087816929895,
        "momentum": 0.6668784969214014,
        "kernel_decay_rate": 0.0017937722033317504,
        "strength_back": 0.1680366998579321,
        "clamped_n_iter": 11,
        "free_n_iter": 14,
        "warmup_n_iter": 6,
        "wback_type": "channel",
    },
}


def build_model(cfg, key):
    # `conv1x1` is Matei's old name for what is now ChannelWBack — same op.
    keys = jax.random.split(key, 5)
    strength_back = cfg.get("strength_back", cfg.get("wback_strength", 1.0))
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0,
                      key=keys[0], padding_mode="constant",
                      lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(channels=C, kernel_size=KSIZE, groups=1,
                                       j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                                       key=keys[1], padding_mode="constant",
                                       lr=1.0, weight_decay=0.0,
                                       entropy_beta=cfg["entropy_beta"],
                                       lambda_entropy=1.0),
            2: ChannelWBack(10, H, W, C, strength_back, keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                               strength=1.0, threshold=5.0,
                               key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def make_optimizer(orch, cfg):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orch, eqx.is_inexact_array)

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
    return opt, opt.init(eqx.filter(orch, eqx.is_inexact_array))


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def run_one(name, cfg, ds):
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    print(f"  warmup={cfg['warmup_n_iter']}, clamped={cfg['clamped_n_iter']}, "
          f"free={cfg['free_n_iter']}", flush=True)

    try:
        key = jax.random.PRNGKey(SEED)
        key, mk = jax.random.split(key)
        state, orch = build_model(cfg, mk)
    except NotImplementedError as e:
        print(f"  SKIPPED: {e}", flush=True)
        return None

    opt, opt_state = make_optimizer(orch, cfg)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=cfg["warmup_n_iter"],
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    decay = cfg["kernel_decay_rate"]
    head_accs = []
    t0 = time.time()
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
        print(f"  epoch {epoch}/{EPOCHS}  head={head_acc:.4f}  "
              f"elapsed={time.time()-t0:.1f}s", flush=True)

    return head_accs


def main():
    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = run_one(name, cfg, ds)

    print(f"\n{'='*60}\nSUMMARY (head accuracy after {EPOCHS} epochs)\n{'='*60}")
    for name, accs in results.items():
        if accs is None:
            print(f"  {name:20s}: SKIPPED")
        else:
            print(f"  {name:20s}: epoch1={accs[0]:.4f}  "
                  f"epoch{EPOCHS}={accs[-1]:.4f}  best={max(accs):.4f}")


if __name__ == "__main__":
    main()
