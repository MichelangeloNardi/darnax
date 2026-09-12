"""Exp p3-2 — polarization / selectivity of FC hidden units vs the C->D flip danger.

For each of the 256 FC hidden units i, compute label-free-ish descriptors and test whether
they identify the "dangerous" units (the ones that flip sign C->D and carry class info):
  polarization      p_i   = mean_n s_i        (over the dataset; per state C and D)
  per-class pol.    p_i^c = mean_{n: y=c} s_i
  selectivity       S_i   = std_c(p_i^c)
Then correlate (Pearson across the 256 units) each of {|p_i|, S_i} against the per-unit
danger quantities from exp-5 / the phenomenon analysis:
  flip_rate_i   = mean_n 1[s^C_i != s^D_i]
  absfield_C_i  = mean_n |field^C_i|                        (weak field-pinning = danger)
  importance_i  = mean_n |W[i,y] - W[i,k]|                  (full-spin C-probe W; k=wrong class)
  damage_i      = mean_n (s^C_i - s^D_i)(W[i,y] - W[i,k])   (readout logit lost C->D)
  wnorm_i       = ||W[i,:]||_2

Hypothesis to TEST (not assume): dangerous units have low |p_i| and high S_i, i.e.
  corr(|p|, flip_rate) < 0, corr(S, flip_rate) > 0, corr(|p|, absfield_C) > 0,
  corr(S, importance) > 0, corr(S, damage) > 0.
Descriptors on both states: D (label-free at inference, no clamp) and C (label-clamped).

Reuses fc_common (build/rollers/collect_spins) + exp-1 phenomenon.py (train_backbone/fit_probe/
pearson). Configs: replicate/tuned_fc_{frozen,trainwin}.json. 3 seeds.

Smoke:  uv run python p3_spin_analysis/2-polarization/polarization.py --smoke
Cluster: XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniforge3/envs/darnax_hpc/bin/python \
  p3_spin_analysis/2-polarization/polarization.py --cfg replicate/tuned_fc_frozen.json
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p3_spin_analysis"))

import fc_common as fc


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


P1 = _load(REPO / "p3_spin_analysis" / "1-fc_phenomenon" / "phenomenon.py", "p3_phenomenon")

# descriptor x danger-quantity correlations to report
DESC = ["absp_C", "absp_D", "S_C", "S_D"]
DANGER = ["flip_rate", "absfield_C", "importance", "damage", "wnorm"]
CORR_KEYS = [f"corr_{d}_{t}" for d in DESC for t in DANGER]


def per_class_pol(X, y, nclass=fc.N_CLASSES):
    """(nclass, N_units) per-class mean activation; selectivity = std over classes (N_units,)."""
    pc = np.stack([X[y == c].mean(0) for c in range(nclass)])
    return pc, pc.std(0)


def analyze(orch, state, ds, cfg, args):
    key = jax.random.PRNGKey(0)
    roll_C, roll_D = fc.rollers(cfg)

    # full-spin C-probe (exp-5): trained on clamped state C
    Xc_tr, ytr = fc.collect_spins(orch, state, ds, roll_C, key, args.probe_train_batches)
    W = P1.fit_probe(Xc_tr, ytr, args.probe_epochs, args.weight_decay)   # (N_units, 10)
    del Xc_tr

    # test states: C (+field) and D, with labels from the SAME pass
    Xc, yte, Cf = fc.collect_spins(orch, state, ds.iter_test(), roll_C, key, args.max_batches, want_field=True)
    Xd, _ = fc.collect_spins(orch, state, ds.iter_test(), roll_D, key, args.max_batches)
    N = Xc.shape[0]

    # per-unit descriptors
    pC, pD = Xc.mean(0), Xd.mean(0)                # polarization in [-1,1] (per unit)
    _, S_C = per_class_pol(Xc, yte)
    _, S_D = per_class_pol(Xd, yte)
    desc = {"absp_C": np.abs(pC), "absp_D": np.abs(pD), "S_C": S_C, "S_D": S_D}

    # per-unit danger quantities
    flip = (Xc != Xd)
    flip_rate = flip.mean(0)
    absfield_C = np.abs(Cf).mean(0)
    logitsC = Xc @ W; masked = logitsC.copy(); masked[np.arange(N), yte] = -np.inf
    kcls = masked.argmax(1)
    margin_w = W[:, yte].T - W[:, kcls].T           # (N, N_units)
    importance = np.abs(margin_w).mean(0)
    damage = ((Xc - Xd) * margin_w).mean(0)
    wnorm = np.linalg.norm(W, axis=1)
    danger = {"flip_rate": flip_rate, "absfield_C": absfield_C,
              "importance": importance, "damage": damage, "wnorm": wnorm}

    out = {f"corr_{d}_{t}": P1.pearson(desc[d], danger[t]) for d in DESC for t in DANGER}
    # context scalars
    out["mean_absp_C"] = float(np.abs(pC).mean()); out["mean_absp_D"] = float(np.abs(pD).mean())
    out["mean_S_C"] = float(S_C.mean()); out["mean_S_D"] = float(S_D.mean())
    out["overall_flip_rate"] = float(flip_rate.mean())
    # per-unit arrays (for plotting; kept per analyze call, saved for seed 0 only by caller)
    arrays = {k: np.asarray(v).tolist() for k, v in {**desc, **danger}.items()}
    return out, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cfg", type=str, default=str(REPO / "replicate" / "tuned_fc_frozen.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-batches", type=int, default=None)
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--probe-epochs", type=int, default=25)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=63)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 2; args.train_batches = 40
        args.probe_train_batches = 12; args.probe_epochs = 4; args.max_batches = 6

    cfg = fc.load_cfg(args.cfg)
    tag = "trainwin" if cfg.get("train_win", False) else "frozen"
    P1.T0 = time.time()
    per_seed, arrays0 = [], None
    for seed in args.seeds:
        orch, state, ds = P1.train_backbone(cfg, seed, args.epochs, args.train_batches)
        d, arrays = analyze(orch, state, ds, cfg, args)
        per_seed.append(d)
        if arrays0 is None:
            arrays0 = arrays
        print(f"  [{tag} s{seed}] corr(|p_D|,flip)={d['corr_absp_D_flip_rate']:+.3f} "
              f"corr(S_D,flip)={d['corr_S_D_flip_rate']:+.3f} "
              f"corr(|p_D|,field)={d['corr_absp_D_absfield_C']:+.3f} "
              f"corr(S_D,imp)={d['corr_S_D_importance']:+.3f} ({fc.fmt(time.time()-P1.T0)})")

    keys = list(per_seed[0].keys())
    results = {"config": Path(args.cfg).name, "train_win": cfg.get("train_win", False),
               "seeds": args.seeds, "per_seed": per_seed,
               "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in keys},
               "std": {k: float(np.std([p[k] for p in per_seed])) for k in keys},
               "arrays_seed0": arrays0}
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (f"polarization_{tag}{'_smoke' if args.smoke else ''}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {fc.fmt(time.time()-P1.T0)})")


if __name__ == "__main__":
    main()
