"""Exp 13 — per-target hyperparameter tuning (Optuna), so each new rule / architecture is
compared near ITS OWN optimum (best_channel_entropy_cfg is tuned for the DynamicalTrainer on
the C=16 standard architecture; reusing it for a new rule/architecture is unfair).

One Optuna study per target. Objective = probe_D (Adam linear probe on pooled D — the metric
all p2 comparisons use; it does not use W_out, so lr_wout is irrelevant and not tuned).

Tunes the 6 main config HPs that drive the D representation:
  lr_j, lr_win, j_d, threshold_j, free_n_iter, entropy_beta
(others kept at the baseline config: lr_wout, threshold_win, clamped_n_iter, momentum,
kernel_decay_rate, strength_back). Screening: 8 epochs, 1 seed, capped batches. After tuning,
re-run the best config at 3 seeds x 20 epochs (final.py) for the fair number.

Targets (architecture + rule):
  chl_c16        standard C16,  contrastive (CHL) rule        (exp 12)
  standard_c24   standard C24,  DynamicalTrainer              (exp 10 scale control)
  standard_c32   standard C32,  DynamicalTrainer              (exp 10 scale control)
  split_c24      split_C24_I16_L8_N0, DynamicalTrainer        (exp 10)
  split_c32      split_C32_I16_L8_N8, DynamicalTrainer        (exp 10)
  partial_c24    partial-overlap C24, DynamicalTrainer        (exp 11)

Run one target (cluster):
  XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python \
    p2_representation/13-hp_tuning/tune.py --target standard_c24 --trials 20
Smoke:  python p2_representation/13-hp_tuning/tune.py --target chl_c16 --smoke
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

import equinox as eqx
import numpy as np
import optuna

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))
sys.path.insert(0, str(REPO / "p2_representation" / "10-split_scale"))
sys.path.insert(0, str(REPO / "p2_representation" / "11-bptt_split"))

import common as cm
import bptt_common as bc
import arch as A
import splitarch as S

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
TUNED_DIR = REPO / "replicate"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


e12 = _load(REPO / "p2_representation" / "12-chl_rule" / "run.py", "exp12_chl")

# target -> (architecture key, rule)
TARGETS = {
    "chl_c16": ("standard_c16", "chl"),
    "standard_c24": ("standard_C24", "dynamical"),
    "standard_c32": ("standard_C32", "dynamical"),
    "split_c24": ("split_C24_I16_L8_N0", "dynamical"),
    "split_c32": ("split_C32_I16_L8_N8", "dynamical"),
    "partial_c24": ("partial", "dynamical"),
}


def build(cfg, key, archkey):
    if archkey == "standard_c16":
        return cm.build_model(cfg, key)
    if archkey == "partial":
        return S.build_partial(cfg, key)
    return A.build_model(cfg, key, archkey)


def pool_caware(h):
    N, Hh, Ww, Cc = h.shape
    p = cm.POOL
    return np.asarray(h).reshape(N, Hh // p, p, Ww // p, p, Cc).mean(axis=(2, 4)).reshape(N, -1)


def collect_D(orch, state_tmpl, it, roll, key, max_b):
    X, Y = [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        s, key = roll(orch, state_tmpl.init(cm.to_hwc(xb), yb), key)
        X.append(pool_caware(np.asarray(s[1]))); Y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(Y), key


def measure_probeD(orch, state_tmpl, ds, cfg, key, probe_epochs, train_batches, test_batches):
    warmup, free = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"]
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, free))
    Xtr, Ytr, key = collect_D(orch, state_tmpl, ds, roll, key, train_batches)
    Xte, Yte, key = collect_D(orch, state_tmpl, ds.iter_test(), roll, key, test_batches)
    return max(bc.offline_probe(Xtr, np.argmax(Ytr, 1), Xte, np.argmax(Yte, 1), probe_epochs))


def train(cfg, archkey, rule, ds, seed, epochs, max_batches):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state_tmpl, orch = build(cfg, mk, archkey)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    warmup, free, clamped = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"], cfg["clamped_n_iter"]
    dr = cfg["kernel_decay_rate"]
    if rule == "dynamical":
        trainer = cm.make_trainer(orch, state_tmpl, opt, opt_state, cfg)
        for _ in range(epochs):
            for i, (xb, yb) in enumerate(ds):
                if max_batches is not None and i >= max_batches:
                    break
                key = trainer.train_step(cm.to_hwc(xb), yb, key)
            if dr > 0:
                for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
                    trainer.orchestrator = eqx.tree_at(path, trainer.orchestrator,
                                                       path(trainer.orchestrator) * (1 - dr))
        orch = trainer.orchestrator
    else:  # chl
        for _ in range(epochs):
            for i, (xb, yb) in enumerate(ds):
                if max_batches is not None and i >= max_batches:
                    break
                state = state_tmpl.init(cm.to_hwc(xb), yb)
                grads, key = e12.chl_grads(orch, state, key, warmup, free, clamped, False)
                params = eqx.filter(orch, eqx.is_inexact_array)
                updates, opt_state = opt.update(eqx.filter(grads, eqx.is_inexact_array), opt_state, params=params)
                orch = eqx.apply_updates(orch, updates)
            if dr > 0:
                for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
                    orch = eqx.tree_at(path, orch, path(orch) * (1 - dr))
    return orch, state_tmpl, key


def sample_cfg(base, trial):
    cfg = dict(base)
    cfg["lr_j"] = trial.suggest_float("lr_j", 1e-4, 5e-3, log=True)
    cfg["lr_win"] = trial.suggest_float("lr_win", 2e-3, 8e-2, log=True)
    cfg["j_d"] = trial.suggest_float("j_d", 0.5, 1.25)
    cfg["threshold_j"] = trial.suggest_float("threshold_j", 1.0, 3.2)
    cfg["free_n_iter"] = trial.suggest_categorical("free_n_iter", [4, 6, 8, 12])
    cfg["entropy_beta"] = trial.suggest_float("entropy_beta", 0.1, 0.9)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--train-batches", type=int, default=400)   # rep collection cap for the probe
    ap.add_argument("--test-batches", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)    # training batch cap (None=full)
    args = ap.parse_args()
    if args.smoke:
        args.trials = 2; args.epochs = 1; args.probe_epochs = 2
        args.train_batches = 6; args.test_batches = 4; args.max_batches = 4

    base = cm.load_cfg(CFG_PATH)
    archkey, rule = TARGETS[args.target]
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    print(f"=== tuning target={args.target} arch={archkey} rule={rule} "
          f"trials={args.trials} epochs/trial={args.epochs} ===")

    def objective(trial):
        cfg = sample_cfg(base, trial)
        try:
            orch, state_tmpl, key = train(cfg, archkey, rule, ds, args.seed, args.epochs, args.max_batches)
            pD = measure_probeD(orch, state_tmpl, ds, cfg, key, args.probe_epochs,
                                args.train_batches, args.test_batches)
        except Exception as e:  # noqa: BLE001 — a diverged config should score 0, not crash the study
            print(f"  trial {trial.number} FAILED: {type(e).__name__}: {e}")
            return 0.0
        pD = float(pD)
        if not np.isfinite(pD):
            pD = 0.0
        print(f"  trial {trial.number:2d} probe_D={pD:.4f} | lr_j={cfg['lr_j']:.2e} lr_win={cfg['lr_win']:.2e} "
              f"j_d={cfg['j_d']:.3f} thr_j={cfg['threshold_j']:.2f} free={cfg['free_n_iter']} "
              f"ent={cfg['entropy_beta']:.3f}  ({cm.fmt(time.time()-t0)})")
        return pD

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)

    best = dict(base)
    best.update(study.best_params)
    best["_tuned_target"] = args.target
    best["_tuned_probe_D"] = study.best_value
    best["_tuned_params"] = study.best_params
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    (out_dir / (f"study_{args.target}{'_smoke' if args.smoke else ''}.json")).write_text(json.dumps(
        {"target": args.target, "arch": archkey, "rule": rule,
         "best_value": study.best_value, "best_params": study.best_params,
         "baseline_probe_D_ref": base.get("probe_acc"),
         "trials": [{"number": t.number, "value": t.value, "params": t.params}
                    for t in study.trials]}, indent=2))
    if not args.smoke:
        cfg_path = TUNED_DIR / f"tuned_{args.target}.json"
        cfg_path.write_text(json.dumps(best, indent=2))
        print(f"wrote tuned config -> {cfg_path}")
    print(f"\nBEST {args.target}: probe_D={study.best_value:.4f}  params={study.best_params}  "
          f"(total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
