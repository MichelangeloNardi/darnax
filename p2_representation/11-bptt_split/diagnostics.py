"""Exp 11 diagnostics — C-aware, explicit channel groups.

Evaluates, on the TRUE hard-sign dynamics:
  part 1 (BPTT split):  split_C24_bptt, split_C32_bptt   groups I / L / N
  part 2 (partial):     partial_C24                       groups both / input_only / label_only
  references (exp-10 local-rule models, loaded from ../10-split_scale/models):
                        standard_C24, split_C24_I16_L8_N0  groups I / L (/ N)

Reuses exp-10 diagnostics' C-aware collection helpers (A.pool-based) + bc.offline_probe.
Global: probe_C, probe_D, probe_C_transfer_D, head_acc_D, flip_rate, overlap_CD, full-spin
pC_on_C / pC_on_D / rand_flip_acc. Per-group: probe_C_group, probe_D_group (decode at C / D),
flip_rate_group, absfield_D_group, fieldD_margin_group.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/11-bptt_split/diagnostics.py
Smoke:  python p2_representation/11-bptt_split/diagnostics.py --smoke
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
sys.path.insert(0, str(REPO / "p2_representation" / "10-split_scale"))

import common as cm
import bptt_common as bc
import arch as A
import splitarch as S
from darnax.states.sequential import SequentialState

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"
EXP10_MODELS = REPO / "p2_representation" / "10-split_scale" / "models"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


D10 = _load(REPO / "p2_representation" / "10-split_scale" / "diagnostics.py", "exp10_diag")

GLOBAL_KEYS = ["probe_C", "probe_D", "probe_C_transfer_D", "head_acc_D", "flip_rate",
               "overlap_CD", "fullspin_pC_on_C", "fullspin_pC_on_D", "rand_flip_acc"]
GROUP_KEYS = ["probe_C_group", "probe_D_group", "flip_rate_group",
              "absfield_D_group", "fieldD_margin_group"]


def diagnose(orch, groups, ds, state_tmpl, roll_C, roll_D, args):
    """groups: dict {gname: channel-index array}. Mirrors exp-10 diagnose with explicit groups."""
    C = int(orch.lmap[1][1].channels)
    key = jax.random.PRNGKey(0); rng = np.random.default_rng(0); ep = args.probe_epochs

    Xc_tr, yc_tr = D10.collect_pooled(orch, state_tmpl, ds, roll_C, key, args.probe_train_batches)
    Xd_tr, yd_tr = D10.collect_pooled(orch, state_tmpl, ds, roll_D, key, args.probe_train_batches)
    Xc_te, yc_te = D10.collect_pooled(orch, state_tmpl, ds.iter_test(), roll_C, key, args.max_batches)
    Xd_te, yd_te = D10.collect_pooled(orch, state_tmpl, ds.iter_test(), roll_D, key, args.max_batches)

    out = {}
    out["probe_C"] = max(bc.offline_probe(Xc_tr, yc_tr, Xc_te, yc_te, ep))
    out["probe_D"] = max(bc.offline_probe(Xd_tr, yd_tr, Xd_te, yd_te, ep))
    out["probe_C_transfer_D"] = max(bc.offline_probe(Xc_tr, yc_tr, Xd_te, yd_te, ep))
    out["head_acc_D"] = D10.acc(Xd_te, np.asarray(orch.lmap[2][1].W), yd_te)

    Cs, ys, Cf = D10.collect_spins(orch, state_tmpl, ds.iter_test(), roll_C, key, args.spin_batches, True)
    Ds, _, Df = D10.collect_spins(orch, state_tmpl, ds.iter_test(), roll_D, key, args.spin_batches, True)
    flip = np.sign(Cs) != np.sign(Ds)
    out["flip_rate"] = float(flip.mean()); out["overlap_CD"] = float((Cs * Ds).mean())

    Xcf_tr, ycf_tr, _ = D10.collect_spins(orch, state_tmpl, ds, roll_C, key, args.probe_train_batches)
    Xcf_tr = Xcf_tr.reshape(Xcf_tr.shape[0], -1)
    Wfs = D10.fit_fullspin_probe(Xcf_tr, ycf_tr, ep, args.weight_decay); del Xcf_tr
    Xcf = Cs.reshape(Cs.shape[0], -1); Xdf = Ds.reshape(Ds.shape[0], -1)
    out["fullspin_pC_on_C"] = D10.acc(Xcf, Wfs, ys)
    out["fullspin_pC_on_D"] = D10.acc(Xdf, Wfs, ys)
    flip_flat = (Xcf != Xdf); kper = flip_flat.sum(1); M = Xcf.shape[1]
    rand = np.zeros_like(flip_flat)
    for i in range(flip_flat.shape[0]):
        if kper[i] > 0:
            rand[i, rng.choice(M, size=int(kper[i]), replace=False)] = True
    Xrand = Xcf.copy(); Xrand[rand] *= -1
    out["rand_flip_acc"] = D10.acc(Xrand, Wfs, ys)

    out["groups"] = {}
    for gname, chans in groups.items():
        ch = np.asarray(chans)
        if ch.size == 0:
            out["groups"][gname] = None
            continue
        cols = A.pooled_cols_for_group(C, ch)
        gpC = max(bc.offline_probe(Xc_tr[:, cols], yc_tr, Xc_te[:, cols], yc_te, ep))
        gpD = max(bc.offline_probe(Xd_tr[:, cols], yd_tr, Xd_te[:, cols], yd_te, ep))
        out["groups"][gname] = {
            "n_channels": int(ch.size),
            "probe_C_group": float(gpC), "probe_D_group": float(gpD),
            "flip_rate_group": float(flip[..., ch].mean()),
            "absfield_D_group": float(np.abs(Df[..., ch]).mean()),
            "fieldD_margin_group": float((Df[..., ch] * Ds[..., ch]).mean()),
        }
    return out


def targets():
    """(tag, model_dir, prefix, template_builder, groups_dict)."""
    out = []
    # part 1 — BPTT split models (this folder)
    for name in S.BPTT_CONFIGS:
        I, L, Nn = A.group_indices(name)
        g = {"I": I, "L": L, "N": Nn}
        out.append((f"{name}__bptt", MODELS_DIR, f"{name}_bptt",
                    lambda c, n=name: A.build_model(c, jax.random.PRNGKey(0), n)[1], g))
    # part 2 — partial-overlap (this folder)
    pg = {k: np.asarray(v) for k, v in S.PARTIAL["groups"].items()}
    out.append(("partial_C24__local", MODELS_DIR, "partial_C24",
                lambda c: S.build_partial(c, jax.random.PRNGKey(0))[1], pg))
    # references — exp-10 local-rule models
    for name in ["standard_C24", "split_C24_I16_L8_N0"]:
        I, L, Nn = A.group_indices(name)
        g = {"I": I, "L": L, "N": Nn}
        out.append((f"{name}__local_ref", EXP10_MODELS, name,
                    lambda c, n=name: A.build_model(c, jax.random.PRNGKey(0), n)[1], g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--spin-batches", type=int, default=63)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 6; args.probe_epochs = 2
        args.spin_batches = 4; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll_C = D10.make_roller(warmup, clamped, free)
    roll_D = D10.make_roller(warmup, 0, free)

    t0 = time.time()
    results = {"config": "best_channel_entropy", "seeds": args.seeds, "models": {}}
    for tag, mdir, prefix, build_tmpl, groups in targets():
        tmpl = build_tmpl(cfg)
        C = int(tmpl.lmap[1][1].channels)
        state_tmpl = SequentialState([(cm.H, cm.W, 3), (cm.H, cm.W, C), 10])
        per_seed = []
        for seed in args.seeds:
            path = mdir / f"{prefix}_seed{seed}.eqx"
            if not path.exists():
                print(f"  [skip {tag} s{seed}] missing {path}")
                continue
            orch = eqx.tree_deserialise_leaves(path, tmpl)
            d = diagnose(orch, groups, ds, state_tmpl, roll_C, roll_D, args)
            per_seed.append(d)
            gp = " ".join(f"{g}D={d['groups'][g]['probe_D_group']:.2f}"
                          for g in d["groups"] if d["groups"][g] is not None)
            print(f"  [{tag:28s} s{seed}] pD={d['probe_D']:.3f} head={d['head_acc_D']:.3f} "
                  f"flip={d['flip_rate']:.3f} ov={d['overlap_CD']:.3f} | {gp} ({cm.fmt(time.time() - t0)})")
        if not per_seed:
            continue
        gnames = list(per_seed[0]["groups"].keys())
        agg_g = {}
        for g in gnames:
            present = [p["groups"][g] for p in per_seed if p["groups"][g] is not None]
            agg_g[g] = None if not present else {
                k: {"mean": float(np.mean([q[k] for q in present])),
                    "std": float(np.std([q[k] for q in present]))} for k in GROUP_KEYS}
        results["models"][tag] = {
            "per_seed": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in GLOBAL_KEYS},
            "std": {k: float(np.std([p[k] for p in per_seed])) for k in GLOBAL_KEYS},
            "group_agg": agg_g,
        }

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "diagnostics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
