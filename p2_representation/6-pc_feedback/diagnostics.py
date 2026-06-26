"""C/D diagnostics for the PC-feedback models, REUSING the exp-3 diagnose() so the
numbers are directly comparable to the raw-clamp baseline (exp-3 model A).

The only change vs exp-3 is the C-roller: for a PC model, C = warmup -> PC-clamped ->
free (the clamped phase uses the closed-loop error field, not the static W_back(y)).
D = warmup -> free is identical for every model. exp-3 diagnose() takes roll_C / roll_D
as arguments, so we just hand it a PC roller.

Models: the raw-clamp baseline + the best-beta wback / wout from train_models' select.json
(use --all-betas to diagnose the whole grid). Diagnostics computed: C/D probe acc,
C-D flip rate, overlap(C,D), C-free stability, margins/fields, importance-vs-flip.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/6-pc_feedback/diagnostics.py
Smoke:  python p2_representation/6-pc_feedback/diagnostics.py --smoke
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
import pc_feedback as pcf

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse exp-3 diagnose() + standard rollers verbatim
d3 = _load(REPO / "p2_representation" / "3-CD_diagnostics" / "diagnostics.py", "exp3_diag")


def load_model(seed, tag, cfg):
    _, template = cm.build_model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{tag}_seed{seed}.eqx", template)


def beta_of(tag):
    return float(tag.split("_b")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--all-betas", action="store_true",
                    help="diagnose the full beta grid (default: select.json best per variant)")
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

    roll_D = d3.make_roller(warmup, 0, free)              # inference state (all models)
    roll_C_raw = d3.make_roller(warmup, clamped, free)    # static-clamp C (baseline)
    free_cont = d3.make_free_cont(free)

    sel = json.loads((MODELS_DIR / "select.json").read_text())
    if args.all_betas:
        tags = ["raw"] + [f"{v}_b{b}" for v in sel["variants"] for b in sel["betas"]]
    else:
        tags = ["raw"] + [sel["selected"][v] for v in sel["variants"]]

    t0 = time.time()
    results = {"config": "best_channel_entropy", "seeds": args.seeds,
               "selected": sel["selected"], "tags": tags, "models": {}}
    for tag in tags:
        if tag == "raw":
            roll_C = roll_C_raw
        else:
            variant = "wback" if tag.startswith("wback") else "wout"
            g = beta_of(tag) * (pcf.N_GAIN ** 0.5)
            roll_C = pcf.make_pc_roller(warmup, clamped, free, variant, g)
        per_seed = []
        for seed in args.seeds:
            orch = load_model(seed, tag, cfg)
            d = d3.diagnose(orch, cfg, ds, state_tmpl, roll_D, roll_C, free_cont, args)
            per_seed.append(d)
            print(f"  [{tag} seed {seed}] probeC={d['probe_C']:.3f} probeD={d['probe_D']:.3f} "
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


if __name__ == "__main__":
    main()
