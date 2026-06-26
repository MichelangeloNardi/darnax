"""Full-spin importance / flip diagnostic for the PC-feedback models, REUSING exp-5
analyze() so the decisive random-flip control is identical to the raw-clamp baseline.

The question (Matei-style): are the C->D spin flips concentrated on the full-spin
features that carry C's class info? For the raw clamp (exp-5 model A) the answer was a
sharp YES (random-flip control 0.97 vs actual flips 0.26). The PC hypothesis is that
error-driven feedback makes C LESS artificially label-imprinted, so its flips become
importance-blind (BPTT-like) while C stays class-informative.

Only the C-roller changes (PC clamped phase); analyze() takes roll_C / roll_D as args.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/6-pc_feedback/fullspin_importance.py
Smoke:  python p2_representation/6-pc_feedback/fullspin_importance.py --smoke
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


fs5 = _load(REPO / "p2_representation" / "5-fullspin_importance" / "fullspin_importance.py", "exp5_fs")


def load_model(seed, tag, cfg):
    _, template = cm.build_model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{tag}_seed{seed}.eqx", template)


def beta_of(tag):
    return float(tag.split("_b")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--all-betas", action="store_true")
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--probe-epochs", type=int, default=25)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=63)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 8; args.probe_epochs = 3
        args.max_batches = 6

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]

    roll_D = fs5.make_roller(warmup, 0, free)
    roll_C_raw = fs5.make_roller(warmup, clamped, free)

    sel = json.loads((MODELS_DIR / "select.json").read_text())
    if args.all_betas:
        tags = ["raw"] + [f"{v}_b{b}" for v in sel["variants"] for b in sel["betas"]]
    else:
        tags = ["raw"] + [sel["selected"][v] for v in sel["variants"]]

    t0 = time.time()
    scal = ["probe_C_on_C", "probe_C_on_D", "corr_flip_importance", "corr_flip_damage",
            "imp_flipped", "imp_stable", "damage_flipped", "damage_stable",
            "flip_rate_overall", "flip_rate_in_top5pct", "top5pct_enrichment",
            "acc_rand_uniform", "acc_rand_empirical",
            "absfield_flipped", "absfield_stable", "fieldC_dot_C_flipped"]
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
            d = fs5.analyze(orch, ds, state_tmpl, roll_C, roll_D, args)
            per_seed.append(d)
            print(f"  [{tag} s{seed}] pC->C={d['probe_C_on_C']:.3f} pC->D={d['probe_C_on_D']:.3f} "
                  f"corr(flip,imp)={d['corr_flip_importance']:.3f} enrich={d['top5pct_enrichment']:.2f} "
                  f"rand_u={d['acc_rand_uniform']:.3f} ({cm.fmt(time.time() - t0)})")
        results["models"][tag] = {
            "per_seed": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in scal},
            "std": {k: float(np.std([p[k] for p in per_seed])) for k in scal},
            "flip_rate_by_decile_mean": np.mean(
                [p["flip_rate_by_decile"] for p in per_seed], 0).tolist(),
        }

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "fullspin_importance.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
