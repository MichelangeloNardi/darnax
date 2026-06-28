"""C-aware global + per-group diagnostics for the split-vs-scale models (exp 10).

Self-contained (NOT reusing exp-3 diagnose / exp-5 analyze, which hardwire C=16 via
cm.pool_j1 / FEAT_IDX / DSPIN). Uses the C-agnostic torch probe fitter (bc.offline_probe)
and arch.pool (infers C from the array). Rollers are the stock C/D rollers.

States: C = warmup->clamped->free (label injected via W_back), D = warmup->free (inference).
Every rep is paired with the labels from its OWN collection pass (train reshuffles).

Global metrics (per config, 3 seeds):
  probe_C, probe_D            pooled Adam probe (train C/eval C ; train D/eval D)
  probe_C_transfer_D         pooled probe trained on C, evaluated on D
  head_acc_D                 model's own W_out on pooled D
  flip_rate, overlap_CD      C->D spin flip rate / mean(C*D)
  fullspin_pC_on_C / _on_D   full-spin (raw 32*32*C) C-probe on C / on D
  rand_flip_acc              full-spin C-probe after flipping the SAME #spins at random
                             (vs fullspin_pC_on_D = after the ACTUAL C->D flips)

Per-group (I / L / N) metrics:
  probe_C_group, probe_D_group   pooled probe restricted to the group's channels (decode at C / at D)
  flip_rate_group
  margin_flipped / margin_stable group field margin (C*field_C) at C
  absfield_D_group               mean |field| on the group's channels at D (= J recurrence drive,
                                 since W_in is masked off L/N and no label at D)
  fieldD_margin_group            mean(field_D * D) on the group's channels at D

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/10-split_scale/diagnostics.py
Smoke:  python p2_representation/10-split_scale/diagnostics.py --smoke
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
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))
sys.path.insert(0, str(REPO / "p2_representation" / "10-split_scale"))

import common as cm
import bptt_common as bc
import arch as A

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_roller(warmup, clamped, free):
    roll = cm.make_rollout(warmup, clamped, free)

    @eqx.filter_jit
    def f(orch, state, key):
        return roll(orch, state, key)[0]
    return f


def load_model(seed, name, cfg):
    _, template = A.build_model(cfg, jax.random.PRNGKey(0), name)
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{name}_seed{seed}.eqx", template)


# ── collection (C-aware) ──────────────────────────────────────────────────────
def collect_pooled(orch, state_tmpl, it, roller, key, max_b):
    X, Y = [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        s = roller(orch, state_tmpl.init(cm.to_hwc(xb), yb), key)
        X.append(A.pool(np.asarray(s[1]))); Y.append(np.argmax(np.asarray(yb), 1))
    return np.concatenate(X), np.concatenate(Y)


def collect_spins(orch, state_tmpl, it, roller, key, max_b, want_field=False):
    S, F, Y = [], [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        s = roller(orch, state_tmpl.init(cm.to_hwc(xb), yb), key)
        S.append(np.asarray(s[1]))
        if want_field:
            F.append(np.asarray(s.fields[1]))
        Y.append(np.argmax(np.asarray(yb), 1))
    S = np.concatenate(S); Y = np.concatenate(Y)
    return S, Y, (np.concatenate(F) if want_field else None)


def fit_fullspin_probe(X, y, epochs, wd):
    """L2 linear probe on raw spins; returns W (D,10). C-agnostic."""
    probe = nn.Linear(X.shape[1], 10, bias=False).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(np.ascontiguousarray(X)).float(),
                                       torch.from_numpy(y).long()), batch_size=256, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
    return probe.weight.detach().cpu().numpy().T


def acc(X, Wp, y):
    return float((X @ Wp).argmax(1).__eq__(y).mean())


def diagnose(orch, name, cfg, ds, state_tmpl, roll_C, roll_D, args):
    C = A.channels_of(name)
    key = jax.random.PRNGKey(0); rng = np.random.default_rng(0)
    ep = args.probe_epochs

    # pooled reps (train/test, C/D), labels paired per pass
    Xc_tr, yc_tr = collect_pooled(orch, state_tmpl, ds, roll_C, key, args.probe_train_batches)
    Xd_tr, yd_tr = collect_pooled(orch, state_tmpl, ds, roll_D, key, args.probe_train_batches)
    Xc_te, yc_te = collect_pooled(orch, state_tmpl, ds.iter_test(), roll_C, key, args.max_batches)
    Xd_te, yd_te = collect_pooled(orch, state_tmpl, ds.iter_test(), roll_D, key, args.max_batches)

    out = {}
    out["probe_C"] = max(bc.offline_probe(Xc_tr, yc_tr, Xc_te, yc_te, ep))
    out["probe_D"] = max(bc.offline_probe(Xd_tr, yd_tr, Xd_te, yd_te, ep))
    out["probe_C_transfer_D"] = max(bc.offline_probe(Xc_tr, yc_tr, Xd_te, yd_te, ep))
    Wout = np.asarray(orch.lmap[2][1].W)
    out["head_acc_D"] = acc(Xd_te, Wout, yd_te)

    # spin subset (C, D, fields)
    Cs, ys, Cf = collect_spins(orch, state_tmpl, ds.iter_test(), roll_C, key, args.spin_batches, True)
    Ds, _, Df = collect_spins(orch, state_tmpl, ds.iter_test(), roll_D, key, args.spin_batches, True)
    flip = np.sign(Cs) != np.sign(Ds)
    out["flip_rate"] = float(flip.mean()); out["overlap_CD"] = float((Cs * Ds).mean())

    # full-spin C-probe + random-flip control
    Xcf_tr, ycf_tr, _ = collect_spins(orch, state_tmpl, ds, roll_C, key, args.probe_train_batches)
    Xcf_tr = Xcf_tr.reshape(Xcf_tr.shape[0], -1)
    Wfs = fit_fullspin_probe(Xcf_tr, ycf_tr, ep, args.weight_decay); del Xcf_tr
    Xcf = Cs.reshape(Cs.shape[0], -1); Xdf = Ds.reshape(Ds.shape[0], -1)
    out["fullspin_pC_on_C"] = acc(Xcf, Wfs, ys)
    out["fullspin_pC_on_D"] = acc(Xdf, Wfs, ys)              # = after ACTUAL C->D flips
    flip_flat = (Xcf != Xdf); kper = flip_flat.sum(1); M = Xcf.shape[1]
    rand = np.zeros_like(flip_flat)
    for i in range(flip_flat.shape[0]):
        if kper[i] > 0:
            rand[i, rng.choice(M, size=int(kper[i]), replace=False)] = True
    Xrand = Xcf.copy(); Xrand[rand] *= -1
    out["rand_flip_acc"] = acc(Xrand, Wfs, ys)

    # per-group
    I, L, Nn = A.group_indices(name)
    out["groups"] = {}
    for gname, chans in [("I", I), ("L", L), ("N", Nn)]:
        if chans.size == 0:
            out["groups"][gname] = None
            continue
        ch = np.asarray(chans)
        cols = A.pooled_cols_for_group(C, ch)
        gpC = max(bc.offline_probe(Xc_tr[:, cols], yc_tr, Xc_te[:, cols], yc_te, ep))
        gpD = max(bc.offline_probe(Xd_tr[:, cols], yd_tr, Xd_te[:, cols], yd_te, ep))
        fg = flip[..., ch]; fl, st = fg, ~fg
        mC = (Cs * Cf)[..., ch]
        out["groups"][gname] = {
            "n_channels": int(chans.size),
            "probe_C_group": float(gpC), "probe_D_group": float(gpD),
            "flip_rate_group": float(fg.mean()),
            "margin_flipped": float(mC[fl].mean()) if fl.any() else 0.0,
            "margin_stable": float(mC[st].mean()) if st.any() else 0.0,
            "absfield_D_group": float(np.abs(Df[..., ch]).mean()),
            "fieldD_margin_group": float((Df[..., ch] * Ds[..., ch]).mean()),
        }
    return out


GLOBAL_KEYS = ["probe_C", "probe_D", "probe_C_transfer_D", "head_acc_D", "flip_rate",
               "overlap_CD", "fullspin_pC_on_C", "fullspin_pC_on_D", "rand_flip_acc"]
GROUP_KEYS = ["probe_C_group", "probe_D_group", "flip_rate_group", "margin_flipped",
              "margin_stable", "absfield_D_group", "fieldD_margin_group"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--names", type=str, nargs="+", default=A.NAMES)
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
    roll_C = make_roller(warmup, clamped, free)
    roll_D = make_roller(warmup, 0, free)

    t0 = time.time()
    results = {"config": "best_channel_entropy", "seeds": args.seeds, "names": args.names,
               "configs": {n: A.CONFIGS[n] for n in args.names}, "models": {}}
    for name in args.names:
        state_tmpl, _ = A.build_model(cfg, jax.random.PRNGKey(0), name)
        per_seed = []
        for seed in args.seeds:
            orch = load_model(seed, name, cfg)
            d = diagnose(orch, name, cfg, ds, state_tmpl, roll_C, roll_D, args)
            per_seed.append(d)
            print(f"  [{name:21s} s{seed}] pC={d['probe_C']:.3f} pD={d['probe_D']:.3f} "
                  f"head={d['head_acc_D']:.3f} transfer={d['probe_C_transfer_D']:.3f} "
                  f"flip={d['flip_rate']:.3f} ov={d['overlap_CD']:.3f} "
                  f"rand={d['rand_flip_acc']:.3f} ({cm.fmt(time.time() - t0)})")
        gnames = ["I", "L", "N"]
        agg_g = {}
        for g in gnames:
            present = [p["groups"][g] for p in per_seed if p["groups"][g] is not None]
            agg_g[g] = None if not present else {
                k: {"mean": float(np.mean([q[k] for q in present])),
                    "std": float(np.std([q[k] for q in present]))} for k in GROUP_KEYS}
        results["models"][name] = {
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
