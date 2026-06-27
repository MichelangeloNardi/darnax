"""Global C/D diagnostics for the channel-partition models, REUSING the exp-3
diagnose() so the numbers are directly comparable to the raw-clamp baseline (exp-3
model A) and across partitions.

Unlike exp 7 (which changed the C-roller), exp 9 changes only the ARCHITECTURE: the
rollers are the STANDARD warmup->clamped->free (C) and warmup->free (D). So we hand
exp-3 diagnose() the stock rollers and the partitioned orchestrator -- fully drop-in.
The only wrinkle is the deserialise template, which must be built per-tag with
partition.build_partitioned_model so the MaskedConv2D out_mask leaf round-trips.

Diagnostics: C/D probe acc, C-D flip rate, overlap(C,D), C-free stability, field
margins, importance-vs-flip, random-flip control. Per-group (I/L/N) versions live in
group_diagnostics.py.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/9-channel_partition/diagnostics.py
Smoke:  python p2_representation/9-channel_partition/diagnostics.py --smoke
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import partition as P

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse exp-3 diagnose() + its rollers verbatim
d3 = _load(REPO / "p2_representation" / "3-CD_diagnostics" / "diagnostics.py", "exp3_diag")


def load_model(seed, tag, cfg):
    _, template = P.build_partitioned_model(cfg, jax.random.PRNGKey(0), tag)
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{tag}_seed{seed}.eqx", template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--tags", type=str, nargs="+", default=P.TAGS)
    ap.add_argument("--probe-train-batches", type=int, default=800)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--diag-examples", type=int, default=2000)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 4; args.probe_epochs = 2
        args.diag_examples = 128; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]

    roll_D = d3.make_roller(warmup, 0, free)
    roll_C = d3.make_roller(warmup, clamped, free)
    free_cont = d3.make_free_cont(free)

    t0 = time.time()
    results = {"config": "best_channel_entropy", "seeds": args.seeds,
               "tags": args.tags, "partitions": {t: P.PARTITIONS[t] for t in args.tags},
               "models": {}}
    for tag in args.tags:
        per_seed = []
        for seed in args.seeds:
            orch = load_model(seed, tag, cfg)
            d = d3.diagnose(orch, cfg, ds, state_tmpl, roll_D, roll_C, free_cont, args)
            per_seed.append(d)
            print(f"  [{tag:9s} s{seed}] probeC={d['probe_C']:.3f} probeD={d['probe_D']:.3f} "
                  f"flip={d['flip_rate']:.3f} ov={d['overlap_CD']:.3f} "
                  f"Cfree_ov={d['C_free_overlap']:.3f} ({cm.fmt(time.time() - t0)})")
        keys = per_seed[0].keys()
        results["models"][tag] = {
            "per_seed": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in keys},
            "std": {k: float(np.std([p[k] for p in per_seed])) for k in keys},
        }

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "diagnostics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time() - t0)})")
    print("Reference: exp-3 model A probeC=0.977 probeD=0.451 flip=0.052 ov=0.896 Cfree_ov=0.999.")


if __name__ == "__main__":
    main()
