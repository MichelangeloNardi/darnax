"""C-basin diagnostics for the local-rule model A.

Do the high-accuracy clamped states C form real attractor basins under the (hard-sign)
free dynamics, how class-structured are they, and how close is the normal inference
trajectory D to falling into the C basin?

Relaxation primitive (free dynamics only, deterministic): from a hidden state s and the
example's input x, iterate  s <- sign(W_in(x) + J1(s))  (zeros -> +1, as the
orchestrator does) for N_RELAX steps. Built directly from model A's W_in / J1 modules.

Diagnostics (model A, 3 seeds, serialized in ../3-CD_diagnostics/models/):
  1. C basin radius   : flip p% random spins of C, relax, report overlap-with-C,
                        return rate (overlap >= RET_THR) and pooled C-probe acc vs p.
  2. C->D boundary    : S_k = C with a random k% of the REAL C->D flip set set to their
                        D values; relax; report overlap-with-C, overlap-with-D and
                        return-to-C vs fall-to-D, vs k.
  3. D-traj rescue    : for each state s_t of the warmup->free (D) trajectory, set the
                        top-k damaged spins to their C values and relax. Two modes:
                        pin (clamp top-k to C throughout) and replace (set once, free).
                        Report final overlap-with-C and pooled C-probe acc, by (k,t).
  4. Class geometry   : within- vs between-class distances of C (full Hamming, pooled
                        L2, probe-logit L2) via class prototypes; prototype nearest-class
                        accuracy.

Probes: pooled 256-d C-probe for all "C-probe accuracy"; full-spin 16384-d C-probe only
to rank spin damage (importance |W[i,y]-W[i,k]|, damage (C-D)(W[y]-W[k]); k = strongest
wrong class), reusing exp 5's definition.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/6-C_basin/c_basin.py
Smoke:  python p2_representation/6-C_basin/c_basin.py --smoke
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
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE.parent / "3-CD_diagnostics" / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W, C = cm.H, cm.W, cm.C
DSPIN = H * W * C            # 16384
N_RELAX = 30
RET_THR = 0.95
P_FLIP = [0, 1, 2, 5, 10, 20, 30, 40]      # diag 1 (percent)
K_BND = [0, 10, 25, 50, 75, 100]           # diag 2 (percent of real flip set)
K_RES = [16, 32, 64, 128, 256, 512]        # diag 3 (number of spins)


def load_model(seed, name, cfg):
    _, template = cm.build_model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{name}_seed{seed}.eqx", template)


# ── jitted dynamics built from model A's W_in / J1 ────────────────────────────

def make_dynamics(orch):
    win, j1 = orch.lmap[1][0], orch.lmap[1][1]

    def _sign(f):
        s = jnp.sign(f)
        return jnp.where(s == 0, 1.0, s)

    @eqx.filter_jit
    def relax_free(x, h0):
        wm = win(x)
        h = h0
        for _ in range(N_RELAX):
            h = _sign(wm + j1(h))
        return h

    @eqx.filter_jit
    def relax_pin(x, h0, mask, vals):
        wm = win(x)
        h = jnp.where(mask, vals, h0)
        for _ in range(N_RELAX):
            h = _sign(wm + j1(h))
            h = jnp.where(mask, vals, h)
        return h

    @eqx.filter_jit
    def d_trajectory(x, h0, n_total):
        # warmup+free are all forward steps; capture hidden state after each
        def body(h, _):
            h2 = _sign(win(x) + j1(h))
            return h2, h2
        _, traj = jax.lax.scan(body, h0, None, length=n_total)
        return traj  # (n_total, N, H, W, C)

    return relax_free, relax_pin, d_trajectory


# ── pooled / full-spin C-probes ───────────────────────────────────────────────

def fit_probe(X, y, epochs, wd):
    probe = nn.Linear(X.shape[1], 10, bias=False).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
        batch_size=256, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
    return probe.weight.detach().cpu().numpy().T


def pool_np(h):  # (N,H,W,C)->(N,256)
    return np.asarray(cm.pool_j1(jnp.asarray(h)))


def probe_acc(h, Wp, y):
    return float((pool_np(h) @ Wp).argmax(1).__eq__(y).mean())


def overlap(a, b):  # (N,...) +-1 -> per-example mean over spins
    return (a.reshape(a.shape[0], -1) * b.reshape(b.shape[0], -1)).mean(1)


# ── collection ────────────────────────────────────────────────────────────────

def collect_C(orch, state_tmpl, it, cfg, key, max_b):
    """Hidden C states (warmup->clamped->free) + inputs + labels."""
    roll = eqx.filter_jit(cm.make_rollout(cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]))
    Cs, Xs, Ys = [], [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        x = cm.to_hwc(xb)
        s, key = roll(orch, state_tmpl.init(x, yb), key)
        Cs.append(np.asarray(s[1])); Xs.append(np.asarray(x)); Ys.append(np.argmax(np.asarray(yb), 1))
    return np.concatenate(Cs), np.concatenate(Xs), np.concatenate(Ys)


def rand_subset_mask(flipset, frac, rng):
    N, M = flipset.shape
    mask = np.zeros((N, M), bool)
    for i in range(N):
        idx = np.where(flipset[i])[0]
        m = int(round(frac * len(idx)))
        if m > 0:
            mask[i, rng.choice(idx, size=m, replace=False)] = True
    return mask


def analyze(orch, ds, state_tmpl, cfg, args):
    key = jax.random.PRNGKey(0); rng = np.random.default_rng(0)
    relax_free, relax_pin, d_traj = make_dynamics(orch)
    warmup, free = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"]
    n_total = warmup + free

    # probes on train C
    Ctr, _, ytr = collect_C(orch, state_tmpl, ds, cfg, key, args.probe_train_batches)
    Ctr_flat = Ctr.reshape(Ctr.shape[0], -1)
    Wp = fit_probe(pool_np(Ctr), ytr, args.probe_epochs, cm.PROBE_WD)      # (256,10)
    Wf = fit_probe(Ctr_flat, ytr, args.probe_epochs, args.fullspin_wd)     # (16384,10)

    # class geometry (diag 4) uses train C prototypes
    out = {}
    out.update(class_geometry(Ctr, ytr, Wp))

    # test subset: C, D-trajectory
    Cte, Xte, yte = collect_C(orch, state_tmpl, ds.iter_test(), cfg, key, args.max_batches)
    N = Cte.shape[0]
    Xj = jnp.asarray(Xte)
    h0 = jnp.zeros((N, H, W, C))            # hidden init = zeros, as in state.init
    traj = np.asarray(d_traj(Xj, h0, n_total))          # (n_total,N,H,W,C); traj[-1]=D
    Dte = traj[-1]
    Cf = jnp.asarray(Cte)

    # --- diag 1: C basin radius ---
    d1 = {"p": P_FLIP, "overlap": [], "return_rate": [], "probe_acc": []}
    Cflat = Cte.reshape(N, -1)
    for p in P_FLIP:
        if p == 0:
            relaxed = np.asarray(relax_free(Xj, Cf))
        else:
            mask = rand_subset_mask(np.ones((N, DSPIN), bool), p / 100.0, rng).reshape(N, H, W, C)
            pert = np.where(mask, -Cte, Cte)
            relaxed = np.asarray(relax_free(Xj, jnp.asarray(pert)))
        ov = overlap(relaxed, Cte)
        d1["overlap"].append(float(ov.mean()))
        d1["return_rate"].append(float((ov >= RET_THR).mean()))
        d1["probe_acc"].append(probe_acc(relaxed, Wp, yte))
    out["basin_radius"] = d1

    # --- diag 2: C->D boundary ---
    flipset = (Cflat != Dte.reshape(N, -1))
    d2 = {"k": K_BND, "overlap_C": [], "overlap_D": [], "return_to_C_rate": []}
    for kp in K_BND:
        if kp == 0:
            Sk = Cte
        else:
            m = rand_subset_mask(flipset, kp / 100.0, rng).reshape(N, H, W, C)
            Sk = np.where(m, Dte, Cte)
        relaxed = np.asarray(relax_free(Xj, jnp.asarray(Sk)))
        ovC = overlap(relaxed, Cte); ovD = overlap(relaxed, Dte)
        d2["overlap_C"].append(float(ovC.mean())); d2["overlap_D"].append(float(ovD.mean()))
        d2["return_to_C_rate"].append(float((ovC > ovD).mean()))
    out["cd_boundary"] = d2

    # --- diag 3: D-trajectory rescue (top-k damaged spins -> C; pin & replace) ---
    logitsC = Cflat @ Wf
    masked = logitsC.copy(); masked[np.arange(N), yte] = -np.inf
    kw = masked.argmax(1)
    margin_w = Wf[:, yte].T - Wf[:, kw].T               # (N,16384)
    damage = (Cflat - Dte.reshape(N, -1)) * margin_w
    order = np.argsort(-damage, axis=1)                 # most-damaged first
    d3 = {"k": K_RES, "t": list(range(1, n_total + 1)), "pin": {}, "replace": {}}
    for k in K_RES:
        topk = np.zeros((N, DSPIN), bool)
        np.put_along_axis(topk, order[:, :k], True, axis=1)
        topk_hwc = topk.reshape(N, H, W, C); topk_j = jnp.asarray(topk_hwc)
        ov_pin, ac_pin, ov_rep, ac_rep = [], [], [], []
        for t in range(n_total):
            st = traj[t]
            st_resc = np.where(topk_hwc, Cte, st)        # set top-k spins to C
            r_pin = np.asarray(relax_pin(Xj, jnp.asarray(st_resc), topk_j, Cf))
            r_rep = np.asarray(relax_free(Xj, jnp.asarray(st_resc)))
            ov_pin.append(float(overlap(r_pin, Cte).mean())); ac_pin.append(probe_acc(r_pin, Wp, yte))
            ov_rep.append(float(overlap(r_rep, Cte).mean())); ac_rep.append(probe_acc(r_rep, Wp, yte))
        d3["pin"][str(k)] = {"overlap_C": ov_pin, "probe_acc": ac_pin}
        d3["replace"][str(k)] = {"overlap_C": ov_rep, "probe_acc": ac_rep}
    out["d_rescue"] = d3

    # reference scalars
    out["probe_C_on_C"] = probe_acc(Cte, Wp, yte)
    out["probe_C_on_D"] = probe_acc(Dte, Wp, yte)
    out["overlap_CD"] = float(overlap(Cte, Dte).mean())
    return out


def class_geometry(C, y, Wp):
    """within/between class distances (full Hamming, pooled L2, probe-logit L2) via
    prototypes, + prototype nearest-class accuracy."""
    N = C.shape[0]; Cflat = C.reshape(N, -1); Cpool = pool_np(C); Clog = Cpool @ Wp
    protos_full, protos_pool, protos_log = [], [], []
    for c in range(10):
        m = (y == c)
        protos_full.append(np.sign(Cflat[m].mean(0)))   # +-1 prototype for Hamming
        protos_pool.append(Cpool[m].mean(0))
        protos_log.append(Clog[m].mean(0))
    PF = np.stack(protos_full); PP = np.stack(protos_pool); PL = np.stack(protos_log)

    def stats(X, P, hamming=False):
        # distance of each example to every prototype
        if hamming:
            d = 0.5 * (1 - (X @ P.T) / X.shape[1])       # fraction differing (X,P in +-1)
        else:
            d = np.linalg.norm(X[:, None, :] - P[None, :, :], axis=2)
        within = d[np.arange(len(X)), y].mean()
        other = d.copy(); other[np.arange(len(X)), y] = np.inf
        between = other.min(1).mean()                    # nearest other-class prototype
        acc = float((d.argmin(1) == y).mean())
        return float(within), float(between), acc

    wf, bf, af = stats(Cflat, PF, hamming=True)
    wp, bp, ap = stats(Cpool, PP)
    wl, bl, al = stats(Clog, PL)
    return {
        "hamming_within": wf, "hamming_between": bf, "proto_acc_hamming": af,
        "pooled_within": wp, "pooled_between": bp, "proto_acc_pooled": ap,
        "logit_within": wl, "logit_between": bl, "proto_acc_logit": al,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--model", default="A")
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--fullspin-wd", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=47)   # ~1500 test imgs
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 6; args.probe_epochs = 3; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))

    t0 = time.time()
    scal = ["probe_C_on_C", "probe_C_on_D", "overlap_CD",
            "hamming_within", "hamming_between", "proto_acc_hamming",
            "pooled_within", "pooled_between", "proto_acc_pooled",
            "logit_within", "logit_between", "proto_acc_logit"]
    results = {"config": "best_channel_entropy", "model": args.model, "seeds": args.seeds,
               "N_RELAX": N_RELAX, "per_seed": []}
    for seed in args.seeds:
        orch = load_model(seed, args.model, cfg)
        d = analyze(orch, ds, state_tmpl, cfg, args)
        results["per_seed"].append(d)
        print(f"  [{args.model} s{seed}] pC->C={d['probe_C_on_C']:.3f} pC->D={d['probe_C_on_D']:.3f} "
              f"proto_acc(pool)={d['proto_acc_pooled']:.3f} "
              f"basin_ret@5%={d['basin_radius']['return_rate'][P_FLIP.index(5)]:.3f} "
              f"bnd_retC@50%={d['cd_boundary']['return_to_C_rate'][K_BND.index(50)]:.3f} "
              f"({cm.fmt(time.time()-t0)})")

    # aggregate scalars
    results["mean"] = {k: float(np.mean([p[k] for p in results["per_seed"]])) for k in scal}
    results["std"] = {k: float(np.std([p[k] for p in results["per_seed"]])) for k in scal}
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else f"c_basin_{args.model}.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
