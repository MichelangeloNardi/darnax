"""Exp 13 final — re-run each tuned config at 3 seeds x 20 epochs (the fair number), next to
the baseline-config result for the same target. Run AFTER tune.py has written
replicate/tuned_<target>.json.

probe_D (+ head_acc_D) reported per target for: tuned config vs baseline_channel_entropy config.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/13-hp_tuning/final.py
Smoke:  python p2_representation/13-hp_tuning/final.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))
sys.path.insert(0, str(REPO / "p2_representation" / "13-hp_tuning"))

import common as cm
import tune as T  # reuse build/train/measure

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
TUNED_DIR = REPO / "replicate"


def measure_both(orch, state_tmpl, ds, cfg, key, probe_epochs, train_b, test_b):
    pD = T.measure_probeD(orch, state_tmpl, ds, cfg, key, probe_epochs, train_b, test_b)
    # head_acc_D on the model's own W_out
    warmup, free = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"]
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, free))
    Xte, Yte, key = T.collect_D(orch, state_tmpl, ds.iter_test(), roll, key, test_b)
    Wout = np.asarray(orch.lmap[2][1].W)
    head = float((Xte @ Wout).argmax(1).__eq__(np.argmax(Yte, 1)).mean())
    return pD, head


def run_cfg(cfg, archkey, rule, ds, seeds, epochs, max_batches, probe_epochs, train_b, test_b):
    pDs, heads = [], []
    for seed in seeds:
        orch, state_tmpl, key = T.train(cfg, archkey, rule, ds, seed, epochs, max_batches)
        pD, head = measure_both(orch, state_tmpl, ds, cfg, key, probe_epochs, train_b, test_b)
        pDs.append(float(pD)); heads.append(head)
    return {"probe_D_mean": float(np.mean(pDs)), "probe_D_std": float(np.std(pDs)),
            "head_D_mean": float(np.mean(heads)), "probe_D_seeds": pDs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--targets", nargs="+", default=list(T.TARGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--test-batches", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="output filename (for per-machine parallel runs)")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 1; args.probe_epochs = 2
        args.train_batches = 6; args.test_batches = 4; args.max_batches = 4

    base = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    results = {"seeds": args.seeds, "epochs": args.epochs, "models": {}}
    for target in args.targets:
        archkey, rule = T.TARGETS[target]
        tuned_path = TUNED_DIR / f"tuned_{target}.json"
        entry = {"arch": archkey, "rule": rule}
        # baseline config
        entry["baseline_cfg"] = run_cfg(base, archkey, rule, ds, args.seeds, args.epochs,
                                        args.max_batches, args.probe_epochs, args.train_batches, args.test_batches)
        # tuned config (if present)
        if tuned_path.exists():
            tuned = cm.load_cfg(tuned_path)
            entry["tuned_cfg"] = run_cfg(tuned, archkey, rule, ds, args.seeds, args.epochs,
                                         args.max_batches, args.probe_epochs, args.train_batches, args.test_batches)
            entry["tuned_params"] = tuned.get("_tuned_params")
        else:
            print(f"  [{target}] no tuned config at {tuned_path}; baseline only")
        results["models"][target] = entry
        b = entry["baseline_cfg"]["probe_D_mean"]
        tt = entry.get("tuned_cfg", {}).get("probe_D_mean")
        print(f"  [{target:13s}] baseline_cfg probe_D={b:.3f}"
              + (f"  tuned probe_D={tt:.3f}" if tt is not None else "")
              + f"  ({cm.fmt(time.time()-t0)})")

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_name = args.out or ("final_smoke.json" if args.smoke else "final.json")
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
