"""Per-group (I / L / N) diagnostics for the channel-partition models.

This is the structural test the partition is FOR: the static label clamp imprints C;
if we confine the label to group L, the prediction is that L stays label-imprinted
while the input-driven (I) and associative-only (N) channels carry a representation the
label never touches directly -- so C de-imprints on I/N and D preserves those channels
better. The global diagnostics (diagnostics.py / fullspin_importance.py) can't see this;
here we break every metric down by channel group.

For each tag and seed, per group g in {I, L, N} (channels = partition.group_indices):
  - C-probe / D-probe by group   : pooled linear probe restricted to g's channels
  - C->D flip rate / overlap(C,D) by group
  - field margin (C*field_C) flipped/stable, |field_C| flipped/stable, by group
  - W_out readout reliance by group : mean ||W_out[pooled-feature, :]|| over g's features
  - random-flip control by group : flip g's ACTUAL C->D flips vs random same-count flips
    within g, scored under a single global full-spin C-probe (does g's flipping target
    class-carrying spins?)

Channel<->feature maps: pool_j1 flattens (4,4,16) C-order, so pooled feature f has
channel f%16; full-spin index s (in 32*32*16, C-order) has channel s%16.

Reuses the exp-5 full-spin probe (fit_probe/acc) and bc.offline_probe for the pooled
group probes. Pairs every rep with the labels from its OWN collection pass (the train
set reshuffles each iteration).

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/9-channel_partition/group_diagnostics.py
Smoke:  python p2_representation/9-channel_partition/group_diagnostics.py --smoke
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
import bptt_common as bc
import partition as P

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"
C, H, W = cm.C, cm.H, cm.W


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d3 = _load(REPO / "p2_representation" / "3-CD_diagnostics" / "diagnostics.py", "exp3_diag")
fs5 = _load(REPO / "p2_representation" / "5-fullspin_importance" / "fullspin_importance.py", "exp5_fs")


def load_model(seed, tag, cfg):
    _, template = P.build_partitioned_model(cfg, jax.random.PRNGKey(0), tag)
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{tag}_seed{seed}.eqx", template)


def collect_spins(orch, state_tmpl, it, roller, key, max_b, want_field=False):
    """Hard-sign spins (N,H,W,C), labels (N,), optional field (N,H,W,C)."""
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
    F = np.concatenate(F) if want_field else None
    return S, Y, F


def group_pooled_probe(orch, state_tmpl, ds, roller, chans, args, key):
    """Pooled linear probe restricted to group `chans` -> test acc (best over epochs)."""
    cols = np.array([f for f in range(256) if (f % C) in set(chans.tolist())])
    Xtr, Ytr = d3.collect_pooled(orch, state_tmpl, ds, roller, key, args.probe_train_batches)
    Xte, Yte = d3.collect_pooled(orch, state_tmpl, ds.iter_test(), roller, key, args.max_batches)
    ytr, yte = np.argmax(Ytr, 1), np.argmax(Yte, 1)
    return max(bc.offline_probe(Xtr[:, cols], ytr, Xte[:, cols], yte, args.probe_epochs))


def analyze_groups(orch, cfg, ds, state_tmpl, roll_C, roll_D, args):
    key = jax.random.PRNGKey(0)
    rng = np.random.default_rng(0)
    I, L, Nn = P.group_indices(args._tag)
    groups = {"I": I, "L": L, "N": Nn}

    # spin-level subset (C, D, C-field)
    Cs, ys, Cf = collect_spins(orch, state_tmpl, ds.iter_test(), roll_C, key,
                               args.spin_batches, want_field=True)
    Ds, _, _ = collect_spins(orch, state_tmpl, ds.iter_test(), roll_D, key, args.spin_batches)
    flip = np.sign(Cs) != np.sign(Ds)                        # (N,H,W,C)
    margin = Cs * Cf                                          # (N,H,W,C)
    absfield = np.abs(Cf)

    # global full-spin C-probe (for the per-group random-flip control)
    Xc_tr, ytr, _ = fs5.collect_fullspin(orch, state_tmpl, ds, roll_C, key, args.probe_train_batches)
    Wfs = fs5.fit_probe(Xc_tr, ytr, args.probe_epochs, args.weight_decay)   # (16384,10)
    del Xc_tr
    Xc = Cs.reshape(Cs.shape[0], -1); Xd = Ds.reshape(Ds.shape[0], -1)
    flip_flat = (Xc != Xd)
    spin_chan = np.arange(Xc.shape[1]) % C                   # channel of each flat spin
    yte_fs = ys
    acc_C_global = fs5.acc(Xc, Wfs, yte_fs)

    # W_out reliance per pooled feature
    Wout = np.asarray(orch.lmap[2][1].W)                     # (256,10)
    fnorm = np.linalg.norm(Wout, axis=1)                    # (256,)

    out = {"acc_C_fullspin_global": float(acc_C_global), "groups": {}}
    for gname, chans in groups.items():
        if chans.size == 0:
            out["groups"][gname] = None
            continue
        chans_set = set(chans.tolist())
        chan_sl = np.asarray(chans)

        # pooled per-group probes (C and D)
        probe_C = group_pooled_probe(orch, state_tmpl, ds, roll_C, chan_sl, args, key)
        probe_D = group_pooled_probe(orch, state_tmpl, ds, roll_D, chan_sl, args, key)

        # spin metrics restricted to group channels (last axis)
        fg = flip[..., chan_sl]; mg = margin[..., chan_sl]; afg = absfield[..., chan_sl]
        Cg, Dg = Cs[..., chan_sl], Ds[..., chan_sl]
        fl, st = fg, ~fg
        g_pool_cols = [f for f in range(256) if (f % C) in chans_set]

        # random-flip control within this group's spins, under the global full-spin probe
        gmask_flat = np.isin(spin_chan, chan_sl)             # (16384,)
        flip_g = flip_flat & gmask_flat[None, :]
        Xc_act = Xc.copy(); Xc_act[flip_g] *= -1
        acc_actual = fs5.acc(Xc_act, Wfs, yte_fs)
        kper = flip_g.sum(1)
        g_spin_idx = np.flatnonzero(gmask_flat)
        rand_g = np.zeros_like(flip_g)
        for i in range(flip_g.shape[0]):
            if kper[i] > 0:
                rand_g[i, rng.choice(g_spin_idx, size=int(kper[i]), replace=False)] = True
        Xc_rand = Xc.copy(); Xc_rand[rand_g] *= -1
        acc_random = fs5.acc(Xc_rand, Wfs, yte_fs)

        out["groups"][gname] = {
            "n_channels": int(chans.size),
            "probe_C": float(probe_C), "probe_D": float(probe_D),
            "flip_rate": float(fg.mean()), "overlap_CD": float((Cg * Dg).mean()),
            "margin_flipped": float(mg[fl].mean()) if fl.any() else 0.0,
            "margin_stable": float(mg[st].mean()) if st.any() else 0.0,
            "absfield_flipped": float(afg[fl].mean()) if fl.any() else 0.0,
            "absfield_stable": float(afg[st].mean()) if st.any() else 0.0,
            "wout_reliance": float(fnorm[g_pool_cols].mean()),
            "randctrl_acc_actual_flips": float(acc_actual),
            "randctrl_acc_random_flips": float(acc_random),
        }
    return out


def _agg(per_seed, gname, field):
    vals = [d["groups"][gname][field] for d in per_seed if d["groups"][gname] is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--tags", type=str, nargs="+", default=P.TAGS)
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--spin-batches", type=int, default=63)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 8; args.probe_epochs = 2
        args.spin_batches = 6; args.max_batches = 6; args.tags = P.TAGS

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll_C = d3.make_roller(warmup, clamped, free)
    roll_D = d3.make_roller(warmup, 0, free)

    t0 = time.time()
    results = {"config": "best_channel_entropy", "seeds": args.seeds,
               "tags": args.tags, "partitions": {t: P.PARTITIONS[t] for t in args.tags},
               "models": {}}
    fields = ["probe_C", "probe_D", "flip_rate", "overlap_CD", "margin_flipped",
              "margin_stable", "absfield_flipped", "absfield_stable", "wout_reliance",
              "randctrl_acc_actual_flips", "randctrl_acc_random_flips"]
    for tag in args.tags:
        args._tag = tag
        per_seed = []
        for seed in args.seeds:
            orch = load_model(seed, tag, cfg)
            d = analyze_groups(orch, cfg, ds, state_tmpl, roll_C, roll_D, args)
            per_seed.append(d)
            msg = " ".join(
                f"{g}(pC={d['groups'][g]['probe_C']:.2f},pD={d['groups'][g]['probe_D']:.2f},"
                f"flip={d['groups'][g]['flip_rate']:.2f})"
                for g in ["I", "L", "N"] if d["groups"][g] is not None)
            print(f"  [{tag:9s} s{seed}] {msg} ({cm.fmt(time.time() - t0)})")
        agg = {}
        for g in ["I", "L", "N"]:
            agg[g] = {f: {"mean": _agg(per_seed, g, f)[0], "std": _agg(per_seed, g, f)[1]}
                      for f in fields}
        results["models"][tag] = {"per_seed": per_seed, "group_agg": agg}

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("group_smoke.json" if args.smoke else "group_diagnostics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
