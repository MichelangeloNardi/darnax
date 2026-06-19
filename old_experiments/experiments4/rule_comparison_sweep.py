"""rule_comparison_sweep.py

Compare forgetting resistance of three learning rules on Matei's architecture:
  - current:     lambda_entropy=0, warmup_n_iter=1   (standard perceptron rule)
  - long_warmup: lambda_entropy=0, warmup_n_iter=10  (perceptron + long warmup)
  - entropy:     lambda_entropy=1, warmup_n_iter=1   (entropy modulation)

Architecture is identical across all three rules (Conv2D 3→16ch + Conv2DRecurrentDiscrete
+ ChannelWBack frozen + PooledFlattenFC). The only things that change are lambda_entropy
and warmup_n_iter — isolating the rule effect from architecture effects.

Sweep: 5 seeds x 5 image pairs, phase A = 60 updates, phase B = 150 updates.

Run from repo root:
  python experiments4/rule_comparison_sweep.py
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

sys.path.insert(0, str(HERE))
from old_experiments.experiments4.entropy_gap_experiment import to_hwc, _soft_margin, _cd_overlap

C, KSIZE = 16, 5
H_IMG, W_IMG = 32, 32
POOL = 8

RULES = {
    "current":     {"lambda_entropy": 0.0, "warmup_n_iter": 1},
    "long_warmup": {"lambda_entropy": 0.0, "warmup_n_iter": 10},
    "entropy":     {"lambda_entropy": 1.0, "warmup_n_iter": 1},
}

SEEDS     = [0, 42, 123, 7, 999]
N_PAIRS   = 5
N_PHASE_A = 60
N_PHASE_B = 150


# ---------------------------------------------------------------------------
# Model builder — parameterized by lambda_entropy
# ---------------------------------------------------------------------------

def build_model(cfg: dict, seed: int, lambda_entropy: float
                ) -> tuple[SequentialState, SequentialOrchestrator]:
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
                entropy_beta=cfg["entropy_beta"], lambda_entropy=lambda_entropy,
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


def build_optimizer(orchestrator, cfg):
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


# ---------------------------------------------------------------------------
# Single forgetting run
# ---------------------------------------------------------------------------

def run_one(cfg, image_A, label_A, image_B, label_B,
            seed: int, lambda_entropy: float, warmup_n_iter: int) -> dict:
    state_template, orch = build_model(cfg, seed, lambda_entropy)
    opt, opt_state = build_optimizer(orch, cfg)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state_template,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=warmup_n_iter,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    key = jax.random.PRNGKey(seed)
    phase_A, phase_B = [], []

    for k in range(N_PHASE_A):
        key = trainer.train_step(image_A, label_A, key)
        C = np.array(trainer.state[1])
        sm_A = _soft_margin(trainer.orchestrator, trainer.state, label_A)
        key, _ = trainer.eval_step(image_A, label_A, key)
        D = np.array(trainer.state[1])
        phase_A.append({"update": k, "cd": float(_cd_overlap(C, D)), "margin_A": sm_A})

    for k in range(N_PHASE_B):
        key = trainer.train_step(image_B, label_B, key)
        C_B = np.array(trainer.state[1])
        sm_B = _soft_margin(trainer.orchestrator, trainer.state, label_B)
        key, _ = trainer.eval_step(image_B, label_B, key)
        D_B = np.array(trainer.state[1])
        cd_B = float(_cd_overlap(C_B, D_B))

        key, _ = trainer.eval_step(image_A, label_A, key)
        sm_A = _soft_margin(trainer.orchestrator, trainer.state, label_A)

        phase_B.append({"update": k, "cd_B": cd_B, "margin_A": sm_A, "margin_B": sm_B})

    return {"phase_A": phase_A, "phase_B": phase_B}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=2 * N_PAIRS, x_transform="identity",
                 label_mode="pm1", linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    xb, yb = next(iter(ds))
    images = [to_hwc(xb[2*i : 2*i+1]) for i in range(N_PAIRS)]
    labels = [yb[2*i : 2*i+1]         for i in range(N_PAIRS)]
    classes = [int(np.argmax(np.array(labels[i])[0])) for i in range(N_PAIRS)]
    print(f"Image classes: {classes}")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "rule_comparison_sweep.json"

    all_results = []
    total = len(RULES) * N_PAIRS * len(SEEDS)
    done = 0

    for rule_name, rule_cfg in RULES.items():
        lambda_e  = rule_cfg["lambda_entropy"]
        warmup_n  = rule_cfg["warmup_n_iter"]

        for pair_idx in range(N_PAIRS):
            img_A, lbl_A = images[pair_idx], labels[pair_idx]
            img_B, lbl_B = images[(pair_idx + 1) % N_PAIRS], labels[(pair_idx + 1) % N_PAIRS]
            cls_A = classes[pair_idx]
            cls_B = classes[(pair_idx + 1) % N_PAIRS]

            for seed in SEEDS:
                done += 1
                print(f"\n[{done}/{total}] rule={rule_name}  pair={pair_idx} "
                      f"(cls {cls_A}→{cls_B})  seed={seed}")

                run = run_one(cfg, img_A, lbl_A, img_B, lbl_B,
                              seed, lambda_e, warmup_n)

                final_margin_A = run["phase_B"][-1]["margin_A"]
                min_margin_A   = min(r["margin_A"] for r in run["phase_B"])
                steps_positive = sum(1 for r in run["phase_B"] if r["margin_A"] > 0)
                memorized_A    = run["phase_A"][-1]["margin_A"] > 0

                print(f"  End phase A: margin_A={run['phase_A'][-1]['margin_A']:.2f}")
                print(f"  End phase B: margin_A={final_margin_A:.2f}  "
                      f"min={min_margin_A:.2f}  steps_positive={steps_positive}/{N_PHASE_B}")

                all_results.append({
                    "rule": rule_name,
                    "pair_idx": pair_idx,
                    "cls_A": cls_A, "cls_B": cls_B,
                    "seed": seed,
                    "memorized_A": memorized_A,
                    "final_margin_A_phaseB": final_margin_A,
                    "min_margin_A_phaseB": min_margin_A,
                    "steps_positive_A_phaseB": steps_positive,
                    "phase_A": run["phase_A"],
                    "phase_B": run["phase_B"],
                })

                out_path.write_text(json.dumps(
                    {"config": cfg, "n_phase_A": N_PHASE_A, "n_phase_B": N_PHASE_B,
                     "seeds": SEEDS, "n_pairs": N_PAIRS, "rules": list(RULES.keys()),
                     "results": all_results},
                    indent=2,
                ))

    # summary table grouped by rule
    print("\n" + "="*80)
    print("SUMMARY — steps where margin_A > 0 during phase B (out of 150)")
    print(f"{'rule':>12}  {'pair':>5}  {'clsA→B':>8}  {'seed':>6}  "
          f"{'steps+':>8}  {'min_mA':>8}  {'final_mA':>10}")
    print("-"*80)
    for r in all_results:
        print(f"{r['rule']:>12}  {r['pair_idx']:>5}  "
              f"{r['cls_A']}→{r['cls_B']:>6}  {r['seed']:>6}  "
              f"{r['steps_positive_A_phaseB']:>7}/{N_PHASE_B}  "
              f"{r['min_margin_A_phaseB']:>8.2f}  "
              f"{r['final_margin_A_phaseB']:>10.2f}")

    # per-rule aggregate
    print("\n" + "="*80)
    print("AGGREGATE — mean steps_positive across all pairs and seeds")
    for rule_name in RULES:
        runs = [r for r in all_results if r["rule"] == rule_name]
        mean_steps = np.mean([r["steps_positive_A_phaseB"] for r in runs])
        mean_min   = np.mean([r["min_margin_A_phaseB"] for r in runs])
        frac_full  = np.mean([r["steps_positive_A_phaseB"] == N_PHASE_B for r in runs])
        print(f"  {rule_name:>12}:  mean_steps_positive={mean_steps:.1f}/{N_PHASE_B}  "
              f"mean_min_margin={mean_min:.2f}  frac_full_survival={frac_full:.2f}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
