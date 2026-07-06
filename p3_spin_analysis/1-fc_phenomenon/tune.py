"""HP-tune the FC asymmetric recurrent net to ITS OWN optimum (per hp-tune-per-rule).

One Optuna study per W_in variant (frozen | trainwin). Objective = probe_D: a
closed-form ridge linear probe on the 256 hidden spins of the free/inference state D
(fast, deterministic, no W_out involved). After tuning, the phenomenon analysis
(phenomenon.py --cfg replicate/tuned_fc_<variant>.json) is run at 3 seeds.

Tunes: lr_j, lr_win, j_d, threshold_j, threshold_win, strength_back, free_n_iter,
clamped_n_iter (lr_win/threshold_win are inert for the frozen variant). Screening:
8 epochs, 1 seed, capped rep-collection batches.

Cluster:
  XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python \
    p3_spin_analysis/1-fc_phenomenon/tune.py --variant trainwin --trials 30
Smoke: uv run python p3_spin_analysis/1-fc_phenomenon/tune.py --variant frozen --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")

import numpy as np
import optuna

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p3_spin_analysis"))

import fc_common as fc

CFG_PATH = REPO / "p3_spin_analysis" / "fc_start_cfg.json"
TUNED_DIR = REPO / "replicate"


def measure_probeD(orch, state, ds, cfg, key, train_batches, test_batches):
    _, roll_D = fc.rollers(cfg)
    Xtr, ytr = fc.collect_spins(orch, state, ds, roll_D, key, train_batches)
    Xte, yte = fc.collect_spins(orch, state, ds.iter_test(), roll_D, key, test_batches)
    return fc.ridge_probe_acc(Xtr, ytr, Xte, yte)


def train(cfg, ds, seed, epochs, max_batches):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = fc.build_model(cfg, mk)
    opt, opt_state = fc.make_optimizer(orch, cfg, win=cfg.get("train_win", False))
    trainer = fc.make_trainer(orch, state, opt, opt_state, cfg)
    for _ in range(epochs):
        for i, (xb, yb) in enumerate(ds):
            if max_batches is not None and i >= max_batches:
                break
            key = trainer.train_step(fc.downsample(xb), yb, key)
    return trainer.orchestrator, state, key


def sample_cfg(base, trial):
    cfg = dict(base)
    cfg["lr_j"] = trial.suggest_float("lr_j", 1e-4, 5e-3, log=True)
    cfg["lr_win"] = trial.suggest_float("lr_win", 1e-3, 5e-2, log=True)
    cfg["j_d"] = trial.suggest_float("j_d", 0.5, 1.25)
    cfg["threshold_j"] = trial.suggest_float("threshold_j", 1.0, 3.2)
    cfg["threshold_win"] = trial.suggest_float("threshold_win", 0.3, 2.0)
    cfg["strength_back"] = trial.suggest_float("strength_back", 0.5, 4.0)
    cfg["free_n_iter"] = trial.suggest_categorical("free_n_iter", [4, 6, 8, 12])
    cfg["clamped_n_iter"] = trial.suggest_categorical("clamped_n_iter", [3, 5, 8])
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["frozen", "trainwin"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--test-batches", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.trials = 2; args.epochs = 1; args.train_batches = 6
        args.test_batches = 4; args.max_batches = 4

    base = fc.load_cfg(CFG_PATH)
    base["train_win"] = (args.variant == "trainwin")
    ds = fc.get_dataset(batch_size=32)
    t0 = time.time()
    print(f"=== tuning FC variant={args.variant} train_win={base['train_win']} "
          f"trials={args.trials} epochs/trial={args.epochs} ===")

    def objective(trial):
        cfg = sample_cfg(base, trial)
        try:
            orch, state, key = train(cfg, ds, args.seed, args.epochs, args.max_batches)
            pD = float(measure_probeD(orch, state, ds, cfg, key, args.train_batches, args.test_batches))
        except Exception as e:  # noqa: BLE001 — a diverged config scores 0, not crash the study
            print(f"  trial {trial.number} FAILED: {type(e).__name__}: {e}")
            return 0.0
        if not np.isfinite(pD):
            pD = 0.0
        print(f"  trial {trial.number:2d} probe_D={pD:.4f} | lr_j={cfg['lr_j']:.2e} "
              f"lr_win={cfg['lr_win']:.2e} j_d={cfg['j_d']:.3f} thr_j={cfg['threshold_j']:.2f} "
              f"thr_w={cfg['threshold_win']:.2f} s_back={cfg['strength_back']:.2f} "
              f"free={cfg['free_n_iter']} clmp={cfg['clamped_n_iter']}  ({fc.fmt(time.time()-t0)})")
        return pD

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)

    best = dict(base)
    best.update(study.best_params)
    best["_tuned_variant"] = args.variant
    best["_tuned_probe_D"] = study.best_value
    best["_tuned_params"] = study.best_params
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    (out_dir / f"study_fc_{args.variant}{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(
        {"variant": args.variant, "best_value": study.best_value, "best_params": study.best_params,
         "trials": [{"number": t.number, "value": t.value, "params": t.params}
                    for t in study.trials]}, indent=2))
    if not args.smoke:
        cfg_path = TUNED_DIR / f"tuned_fc_{args.variant}.json"
        cfg_path.write_text(json.dumps(best, indent=2))
        print(f"wrote tuned config -> {cfg_path}")
    print(f"\nBEST {args.variant}: probe_D={study.best_value:.4f}  params={study.best_params}  "
          f"(total {fc.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
