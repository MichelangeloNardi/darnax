"""Exp 15 — split re-tune with the phase-step counts + injection strength (close the splits
rigorously).

Exp 13 tuned 6 HPs (lr_j, lr_win, j_d, threshold_j, free_n_iter, entropy_beta) but HELD
clamped_n_iter, warmup_n_iter and strength_back at baseline. Those three are the direct levers
on the split's failure mode: in a split the label is confined to the L group and (exp 10)
couldn't move the state (C≡D). D=warmup->free doesn't use the clamped phase or W_back directly,
but TRAINING rolls warmup->clamped->free and the local rule learns from it, so clamped_n_iter /
warmup_n_iter / strength_back shape the learned W_in/J1 and thus D indirectly. This re-tune adds
those three -> 9 HPs, so each split (and its dense control) is compared near its true optimum.

Reuses exp-13 tune.py (build / train / measure_probeD) via importlib; only sample_cfg (9 params)
and the target set (the 5 architecture configs; no CHL) differ. Objective = probe_D. Writes the
best config to replicate/tuned9_<target>.json (does NOT clobber exp-13's tuned_<target>.json) and
the study to results/study9_<target>.json. final.py then re-runs at 3 seeds x 20 ep.

Run one target (cluster):
  XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python \
    p2_representation/15-split_retune/tune.py --target split_c24 --trials 40
Smoke:  python p2_representation/15-split_retune/tune.py --target split_c24 --smoke
"""
from __future__ import annotations

import argparse
import importlib.util
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
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# reuse exp-13 build / train / measure_probeD / collect_D verbatim
T = _load(REPO / "p2_representation" / "13-hp_tuning" / "tune.py", "exp13_tune")
import common as cm  # noqa: E402

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
TUNED_DIR = REPO / "replicate"

# the 5 architecture targets (splits + dense controls); CHL is a separate thread
TARGETS = {k: T.TARGETS[k] for k in
           ["standard_c24", "split_c24", "partial_c24", "standard_c32", "split_c32"]}


def sample_cfg(base, trial):
    """9-param space: the exp-13 six + clamped_n_iter, warmup_n_iter, strength_back."""
    cfg = dict(base)
    cfg["lr_j"] = trial.suggest_float("lr_j", 1e-4, 5e-3, log=True)
    cfg["lr_win"] = trial.suggest_float("lr_win", 2e-3, 8e-2, log=True)
    cfg["j_d"] = trial.suggest_float("j_d", 0.5, 1.25)
    cfg["threshold_j"] = trial.suggest_float("threshold_j", 1.0, 3.2)
    cfg["free_n_iter"] = trial.suggest_categorical("free_n_iter", [4, 6, 8, 12])
    cfg["entropy_beta"] = trial.suggest_float("entropy_beta", 0.1, 0.9)
    cfg["clamped_n_iter"] = trial.suggest_categorical("clamped_n_iter", [3, 5, 8, 12, 16])
    cfg["warmup_n_iter"] = trial.suggest_categorical("warmup_n_iter", [1, 2, 4])
    cfg["strength_back"] = trial.suggest_float("strength_back", 0.5, 6.0)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--test-batches", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.trials = 2; args.epochs = 1; args.probe_epochs = 2
        args.train_batches = 6; args.test_batches = 4; args.max_batches = 4

    base = cm.load_cfg(CFG_PATH)
    archkey, rule = TARGETS[args.target]
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    print(f"=== exp15 re-tune target={args.target} arch={archkey} rule={rule} "
          f"trials={args.trials} epochs/trial={args.epochs} (9 HPs) ===")

    def objective(trial):
        cfg = sample_cfg(base, trial)
        try:
            orch, state_tmpl, key = T.train(cfg, archkey, rule, ds, args.seed, args.epochs, args.max_batches)
            pD = T.measure_probeD(orch, state_tmpl, ds, cfg, key, args.probe_epochs,
                                  args.train_batches, args.test_batches)
        except Exception as e:  # noqa: BLE001 — a diverged config should score 0, not crash the study
            print(f"  trial {trial.number} FAILED: {type(e).__name__}: {e}")
            return 0.0
        pD = float(pD)
        if not np.isfinite(pD):
            pD = 0.0
        print(f"  trial {trial.number:2d} probe_D={pD:.4f} | lr_j={cfg['lr_j']:.2e} "
              f"lr_win={cfg['lr_win']:.2e} j_d={cfg['j_d']:.2f} thr_j={cfg['threshold_j']:.2f} "
              f"free={cfg['free_n_iter']} clamp={cfg['clamped_n_iter']} warm={cfg['warmup_n_iter']} "
              f"sback={cfg['strength_back']:.2f} ent={cfg['entropy_beta']:.2f}  ({cm.fmt(time.time()-t0)})")
        return pD

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)

    best = dict(base); best.update(study.best_params)
    best["_tuned_target"] = args.target
    best["_tuned_probe_D"] = study.best_value
    best["_tuned_params"] = study.best_params
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    (out_dir / (f"study9_{args.target}{'_smoke' if args.smoke else ''}.json")).write_text(json.dumps(
        {"target": args.target, "arch": archkey, "rule": rule, "n_hp": 9,
         "best_value": study.best_value, "best_params": study.best_params,
         "trials": [{"number": t.number, "value": t.value, "params": t.params}
                    for t in study.trials]}, indent=2))
    if not args.smoke:
        cfg_path = TUNED_DIR / f"tuned9_{args.target}.json"
        cfg_path.write_text(json.dumps(best, indent=2))
        print(f"wrote tuned config -> {cfg_path}")
    print(f"\nBEST {args.target}: probe_D={study.best_value:.4f}  params={study.best_params}  "
          f"(total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
