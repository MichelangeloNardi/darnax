"""entropy_gap_experiment.py

Single-image train/inference gap and forgetting experiment using the
channel_entropy architecture (Conv2D + Conv2DRecurrentDiscrete(entropy) +
ChannelWBack + PooledFlattenFC).

Two differences from replicate_channel_entropy.py:
  - No kernel normalization; Conv2D (Win) uses weight_decay=0.001 instead.
  - Single-image training: no DataLoader, no epochs, N weight updates per image.

Run from repo root:
  python experiments4/entropy_gap_experiment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"

sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

C, KSIZE = 16, 5
H_IMG, W_IMG = 32, 32
POOL = 8


# ---------------------------------------------------------------------------
# Model and optimizer
# ---------------------------------------------------------------------------

def build_entropy_model(cfg: dict, seed: int) -> tuple[SequentialState, SequentialOrchestrator]:
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(
                in_channels=3, out_channels=C, kernel_size=KSIZE,
                threshold=cfg["threshold_win"], strength=1.0,
                key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.001,
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


def build_optimizer(
    orchestrator: SequentialOrchestrator, cfg: dict
) -> tuple[optax.GradientTransformation, optax.OptState]:
    mom = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr: float) -> optax.GradientTransformation:
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(
            lambda m, r=i, c=j: m.lmap[r][c], labels,
            replace=like(params.lmap[i][j], lbl),
        )

    # W_back at (1,2) stays "default" → lr=0 (frozen)
    opt = optax.multi_transform({
        "default": optax.sgd(0.0),
        "win":     sgd(-cfg["lr_win"]),
        "j1":      sgd(-cfg["lr_j"]),
        "wout":    sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _soft_margin(orch: SequentialOrchestrator, state: SequentialState, y) -> float:
    """Correct logit minus max wrong logit, after calling predict."""
    pred_state, _ = orch.predict(state, jax.random.PRNGKey(0))
    logits = np.array(pred_state.states[-1])[0]   # (10,)
    y_np = np.array(y)[0]
    true_cls = int(np.argmax(y_np))
    correct = logits[true_cls]
    wrong = np.concatenate([logits[:true_cls], logits[true_cls + 1:]])
    return float(correct - wrong.max())


def _cd_overlap(C, D) -> float:
    """mean(C * D) over a pair of ±1 conv-map arrays."""
    return float(jnp.mean(jnp.asarray(C) * jnp.asarray(D)))


# ---------------------------------------------------------------------------
# Core experiment functions
# ---------------------------------------------------------------------------

def run_gap_experiment(
    cfg: dict,
    image,
    label,
    n_updates: int,
    seed: int,
) -> list[dict]:
    """Run N weight updates on one image; track CD overlap and soft margin."""
    state_template, orch = build_entropy_model(cfg, seed)
    opt, opt_state = build_optimizer(orch, cfg)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state_template,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    key = jax.random.PRNGKey(seed)
    results = []

    for k in range(n_updates):
        # weight update: warmup → clamped → free
        key = trainer.train_step(image, label, key)
        C = np.array(trainer.state[1])   # (1, 32, 32, 16) conv map after free phase
        sm = _soft_margin(trainer.orchestrator, trainer.state, label)

        # inference pass: warmup only (eval_step = warmup + free eval dynamics)
        key, _ = trainer.eval_step(image, label, key)
        D = np.array(trainer.state[1])

        cd = _cd_overlap(C, D)
        results.append({"update": k, "cd_overlap": cd, "soft_margin": sm})

        if k % 10 == 0:
            print(f"  [gap] step {k:3d}  CD={cd:.3f}  margin={sm:.3f}")

    return results


def run_forgetting_experiment(
    cfg: dict,
    image_A,
    label_A,
    image_B,
    label_B,
    n_phase: int,
    seed: int,
) -> dict:
    """Memorize A for n_phase updates, then switch to B for n_phase updates."""
    state_template, orch = build_entropy_model(cfg, seed)
    opt, opt_state = build_optimizer(orch, cfg)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state_template,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    key = jax.random.PRNGKey(seed)
    phase_A = []

    # Phase A: train on image A
    for k in range(n_phase):
        key = trainer.train_step(image_A, label_A, key)
        C_A = np.array(trainer.state[1])
        sm_A = _soft_margin(trainer.orchestrator, trainer.state, label_A)

        key, _ = trainer.eval_step(image_A, label_A, key)
        D_A = np.array(trainer.state[1])

        cd_A = _cd_overlap(C_A, D_A)
        phase_A.append({"update": k, "cd_overlap": cd_A, "soft_margin_A": sm_A})

        if k % 10 == 0:
            print(f"  [phase_A] step {k:3d}  CD={cd_A:.3f}  margin_A={sm_A:.3f}")

    phase_B = []

    # Phase B: train on image B, eval A without weight update
    for k in range(n_phase):
        key = trainer.train_step(image_B, label_B, key)
        C_B = np.array(trainer.state[1])
        sm_B = _soft_margin(trainer.orchestrator, trainer.state, label_B)

        key, _ = trainer.eval_step(image_B, label_B, key)
        D_B = np.array(trainer.state[1])
        cd_B = _cd_overlap(C_B, D_B)

        # eval A: inference only, no weight update
        key, _ = trainer.eval_step(image_A, label_A, key)
        sm_A = _soft_margin(trainer.orchestrator, trainer.state, label_A)

        phase_B.append({
            "update": k,
            "cd_overlap_B": cd_B,
            "soft_margin_A": sm_A,
            "soft_margin_B": sm_B,
        })

        if k % 10 == 0:
            print(f"  [phase_B] step {k:3d}  CD_B={cd_B:.3f}  margin_A={sm_A:.3f}  margin_B={sm_B:.3f}")

    return {"phase_A": phase_A, "phase_B": phase_B}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=2, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    it = iter(ds)
    xb0, yb0 = next(it)   # shape: (2, ...) — take first sample from each
    image_A = to_hwc(xb0[0:1])
    image_B = to_hwc(xb0[1:2])
    label_A = yb0[0:1]
    label_B = yb0[1:2]

    print(f"Image A class: {int(np.argmax(np.array(label_A)[0]))}")
    print(f"Image B class: {int(np.argmax(np.array(label_B)[0]))}")

    print("\n=== Gap experiment (image A, 60 updates) ===")
    gap_results = run_gap_experiment(cfg, image_A, label_A, n_updates=60, seed=0)

    print("\n=== Forgetting experiment (60 + 60 updates) ===")
    forgetting_results = run_forgetting_experiment(
        cfg, image_A, label_A, image_B, label_B, n_phase=60, seed=0,
    )

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "entropy_gap_results.json"
    out_path.write_text(json.dumps(
        {"config": cfg, "gap_experiment": gap_results, "forgetting_experiment": forgetting_results},
        indent=2,
    ))
    print(f"\nSaved to {out_path}")

    print("\n--- Gap experiment (every 10 steps) ---")
    print(f"{'step':>5}  {'CD_overlap':>10}  {'soft_margin':>12}")
    for r in gap_results:
        if r["update"] % 10 == 0:
            print(f"{r['update']:>5}  {r['cd_overlap']:>10.3f}  {r['soft_margin']:>12.3f}")

    print("\n--- Forgetting experiment (every 10 steps) ---")
    print(f"{'phase':>7}  {'step':>5}  {'CD':>8}  {'margin_A':>10}  {'margin_B':>10}")
    for r in forgetting_results["phase_A"]:
        if r["update"] % 10 == 0:
            print(f"{'A':>7}  {r['update']:>5}  {r['cd_overlap']:>8.3f}  {r['soft_margin_A']:>10.3f}  {'—':>10}")
    for r in forgetting_results["phase_B"]:
        if r["update"] % 10 == 0:
            print(f"{'B':>7}  {r['update']:>5}  {r['cd_overlap_B']:>8.3f}  {r['soft_margin_A']:>10.3f}  {r['soft_margin_B']:>10.3f}")


if __name__ == "__main__":
    main()
