"""entropy_forgetting_sweep.py

Forgetting sweep for the entropy architecture:
  - Phase A: 60 weight updates on image A (memorize)
  - Phase B: 150 weight updates on image B (does A survive?)
  - 5 seeds × 5 image pairs = 25 runs total

Key question: does margin_A stay positive throughout phase B,
or does it eventually collapse? Does this hold across seeds and pairs?

Run from repo root:
  python experiments4/entropy_forgetting_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"

sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.trainers.dynamical import DynamicalTrainer

# import shared helpers from the first script
sys.path.insert(0, str(HERE))
from old_experiments.experiments4.entropy_gap_experiment import (
    build_entropy_model,
    build_optimizer,
    to_hwc,
    _soft_margin,
    _cd_overlap,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS       = [0, 42, 123, 7, 999]
N_PAIRS     = 5        # image pairs (pair k = images 2k and 2k+1 from dataset)
N_PHASE_A   = 60       # weight updates to memorize A
N_PHASE_B   = 150      # weight updates on B — the key question


# ---------------------------------------------------------------------------
# Single forgetting run
# ---------------------------------------------------------------------------

def run_one(cfg, image_A, label_A, image_B, label_B, seed: int) -> dict:
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

def main() -> None:
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=2 * N_PAIRS, x_transform="identity",
                 label_mode="pm1", linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    # grab first batch: 2*N_PAIRS images, split into N_PAIRS pairs
    xb, yb = next(iter(ds))
    images = [to_hwc(xb[2*i : 2*i+1]) for i in range(N_PAIRS)]
    labels = [yb[2*i : 2*i+1]         for i in range(N_PAIRS)]
    classes = [int(np.argmax(np.array(labels[i])[0])) for i in range(N_PAIRS)]
    print(f"Image classes: {classes}")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = []
    total = len(SEEDS) * N_PAIRS
    done = 0

    for pair_idx in range(N_PAIRS):
        img_A, lbl_A = images[pair_idx], labels[pair_idx]
        img_B, lbl_B = images[(pair_idx + 1) % N_PAIRS], labels[(pair_idx + 1) % N_PAIRS]
        cls_A, cls_B = classes[pair_idx], classes[(pair_idx + 1) % N_PAIRS]

        for seed in SEEDS:
            done += 1
            print(f"\n[{done}/{total}] pair={pair_idx} (cls {cls_A}→{cls_B})  seed={seed}")
            run = run_one(cfg, img_A, lbl_A, img_B, lbl_B, seed)

            # summary stats for this run
            final_margin_A = run["phase_B"][-1]["margin_A"]
            min_margin_A   = min(r["margin_A"] for r in run["phase_B"])
            steps_positive = sum(1 for r in run["phase_B"] if r["margin_A"] > 0)
            memorized_A    = run["phase_A"][-1]["margin_A"] > 0

            print(f"  End of phase A: margin_A={run['phase_A'][-1]['margin_A']:.2f}")
            print(f"  End of phase B: margin_A={final_margin_A:.2f}  "
                  f"min={min_margin_A:.2f}  steps_positive={steps_positive}/{N_PHASE_B}")

            all_results.append({
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

            # save incrementally so we don't lose data if interrupted
            out_path = results_dir / "entropy_forgetting_sweep.json"
            out_path.write_text(json.dumps(
                {"config": cfg, "n_phase_A": N_PHASE_A, "n_phase_B": N_PHASE_B,
                 "seeds": SEEDS, "n_pairs": N_PAIRS, "results": all_results},
                indent=2,
            ))

    # final summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print(f"{'pair':>5}  {'clsA→B':>8}  {'seed':>6}  {'min_mA':>8}  {'steps+':>8}  {'final_mA':>10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['pair_idx']:>5}  {r['cls_A']}→{r['cls_B']:>6}  {r['seed']:>6}  "
              f"{r['min_margin_A_phaseB']:>8.2f}  "
              f"{r['steps_positive_A_phaseB']:>7}/{N_PHASE_B}  "
              f"{r['final_margin_A_phaseB']:>10.2f}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
