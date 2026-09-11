"""Exp 15 final — re-run each 9-HP tuned config at 3 seeds x 20 epochs vs the baseline config,
for the split re-tune. Run AFTER tune.py wrote replicate/tuned9_<target>.json.

Reuses exp-13 final.py::run_cfg (which reuses exp-13 tune.py build/train/measure). Reports
probe_D + head_D per target for baseline-cfg and the 9-HP tuned config. The split-vs-dense
comparison (split_cX tuned9 vs standard_cX tuned9, same C = same nominal params) is assembled in
the README. The exp-13 6-HP tuned numbers live in ../13-hp_tuning/results/final.json.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/15-split_retune/final.py --targets split_c24
Smoke:  python p2_representation/15-split_retune/final.py --targets split_c24 --smoke
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


F = _load(REPO / "p2_representation" / "13-hp_tuning" / "final.py", "exp13_final")
T = F.T
import common as cm  # noqa: E402

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
TUNED_DIR = REPO / "replicate"
TARGETS = ["standard_c24", "split_c24", "partial_c24", "standard_c32", "split_c32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--test-batches", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 1; args.probe_epochs = 2
        args.train_batches = 6; args.test_batches = 4; args.max_batches = 4

    base = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    results = {"seeds": args.seeds, "epochs": args.epochs, "n_hp": 9, "models": {}}
    for target in args.targets:
        archkey, rule = T.TARGETS[target]
        entry = {"arch": archkey, "rule": rule}
        entry["baseline_cfg"] = F.run_cfg(base, archkey, rule, ds, args.seeds, args.epochs,
                                          args.max_batches, args.probe_epochs, args.train_batches, args.test_batches)
        tuned_path = TUNED_DIR / f"tuned9_{target}.json"
        if tuned_path.exists():
            tuned = cm.load_cfg(tuned_path)
            entry["tuned9_cfg"] = F.run_cfg(tuned, archkey, rule, ds, args.seeds, args.epochs,
                                            args.max_batches, args.probe_epochs, args.train_batches, args.test_batches)
            entry["tuned9_params"] = tuned.get("_tuned_params")
        else:
            print(f"  [{target}] no tuned9 config at {tuned_path}; baseline only")
        results["models"][target] = entry
        b = entry["baseline_cfg"]["probe_D_mean"]
        tt = entry.get("tuned9_cfg", {}).get("probe_D_mean")
        print(f"  [{target:13s}] baseline_cfg probe_D={b:.3f}"
              + (f"  tuned9 probe_D={tt:.3f}" if tt is not None else "")
              + f"  ({cm.fmt(time.time()-t0)})")

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_name = args.out or ("final_smoke.json" if args.smoke else "final.json")
    (out_dir / out_name).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_dir / out_name}  (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
