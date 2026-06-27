"""BPTT field-loss experiments: can we push D above the ~0.51 ceiling by forcing the
student's input-only field to support teacher A's class-carrying C_A spins (enlarging
the C basin so D falls into it)?

Teacher = frozen local-rule model A (exp-3 models/A_seed*.eqx). Per batch (no grad):
  C_A = teacher warmup->clamp->free (hard sign);  D_A = teacher warmup->free
  M   = top-k spins by damage (C_A - D_A)*(W_A[:,y] - W_A[:,k]) under teacher's full-spin
        C-probe W_A (k = strongest wrong class). k = 256.
Student = BPTT, INITIALIZED FROM TEACHER A (so the per-spin target C_A[i] is coordinate-
aligned), tanh surrogate (beta 1->4), j_d frozen, checkpoint by hard-sign sep.

  Exp 1: L = CE(Wout.pool(D), y) + gamma * mean_{i in M} ReLU(kappa - C_A[i]*h_D[i])
  Exp 2: L = CE(Wout.pool(D), y) + gamma * mean_t mean_{i in M} ReLU(kappa - C_A[i]*h_t[i])
where h_t = W_in x + J1 s_t (student input-only field). kappa = 1.0.

Always measured on the TRUE hard-sign D. Reports: D probe acc, perceptron W_out acc,
D<->C_A overlap, teacher C-probe transfer on D, field margin C_A*h on M, flip rate on M.

Minimal probe: gamma in {1,3}, k=256, both experiments, 3 seeds.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/8-BPTT_field_loss/run.py
Smoke:  python p2_representation/8-BPTT_field_loss/run.py --smoke
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
import optax
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE.parent / "3-CD_diagnostics" / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W, C = cm.H, cm.W, cm.C
DSPIN = H * W * C
KAPPA = 1.0


def load_teacher(seed, cfg):
    _, t = cm.build_model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"A_seed{seed}.eqx", t)


def fit_probe(X, y, epochs, wd):
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


def make_teacher_targets(teacher, state_tmpl, cfg, k):
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    rollC = cm.make_rollout(warmup, clamped, free)
    rollD = cm.make_rollout(warmup, 0, free)

    def targets(W_A, x, yb, y_idx, key):
        sC, _ = rollC(teacher, state_tmpl.init(x, yb), key)
        sD, _ = rollD(teacher, state_tmpl.init(x, yb), key)
        C_A, D_A = sC[1], sD[1]                               # (N,H,W,C) +-1
        n = C_A.shape[0]
        Cf, Df = C_A.reshape(n, -1), D_A.reshape(n, -1)
        logits = Cf @ W_A
        masked = logits.at[jnp.arange(n), y_idx].set(-jnp.inf)
        kw = masked.argmax(1)
        mw = W_A[:, y_idx].T - W_A[:, kw].T
        damage = (Cf - Df) * mw
        _, idx = jax.lax.top_k(damage, k)                     # (n,k)
        M = jnp.zeros((n, DSPIN), bool).at[jnp.arange(n)[:, None], idx].set(True)
        return C_A, M.reshape(n, H, W, C)

    return eqx.filter_jit(targets)


def collect_teacher_C(teacher, state_tmpl, it, cfg, key, max_b, full=False):
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll = eqx.filter_jit(cm.make_rollout(warmup, clamped, free))
    Xs, Cs, Ys = [], [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        x = cm.to_hwc(xb)
        s, key = roll(teacher, state_tmpl.init(x, yb), key)
        Cs.append(np.asarray(s[1])); Ys.append(np.argmax(np.asarray(yb), 1))
        if full:
            Xs.append(np.asarray(x))
    C = np.concatenate(Cs); Y = np.concatenate(Ys)
    return (C, Y, np.concatenate(Xs)) if full else (C, Y)


def overlap(a, b):
    return (a.reshape(a.shape[0], -1) * b.reshape(b.shape[0], -1)).mean(1)


def limited(it, n):
    for i, b in enumerate(it):
        if n is not None and i >= n:
            return
        yield b


def run_student(teacher, params0, W_A, Wp_A, tt, cfg, ds, exp1, gamma, seed, args, t0, tag):
    win, j1, wout = teacher.lmap[1][0], teacher.lmap[1][1], teacher.lmap[2][1]
    warmup, free = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"]
    n_steps = warmup + free
    params = {k: jnp.array(v) for k, v in params0.items()}

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)
    step = bc.make_field_step(win, j1, wout, n_steps, "tanh", exp1, opt, KAPPA)
    hard_reps = bc.make_hard_reps(win, j1, n_steps)
    betas = np.geomspace(1.0, args.beta_max, args.bptt_epochs).astype(np.float32)
    key = jax.random.PRNGKey(seed)

    # checkpoint proxy subset
    Xf, Yf = [], []
    for xb, yb in limited(ds, args.proxy_batches if args.max_batches is None else args.max_batches):
        Xf.append(cm.to_hwc(xb)); Yf.append(np.asarray(yb))
    Xf = jnp.asarray(np.concatenate(Xf)); Yf = np.concatenate(Yf)
    half = len(Xf) // 2
    Xfa, Xfe = Xf[:half], Xf[half:]
    Yoh = jnp.asarray((Yf[:half] > 0).astype(np.float32)); ye = jnp.asarray(np.argmax(Yf[half:], 1))

    best_sep, best_params = -1.0, params
    for ep in range(args.bptt_epochs):
        beta = jnp.asarray(betas[ep]); g = jnp.asarray(gamma)
        for xb, yb in limited(ds, args.max_batches):
            x = cm.to_hwc(xb); yj = jnp.asarray(np.asarray(yb))
            yidx = jnp.asarray(np.argmax(np.asarray(yb), 1))
            C_A, M = tt(W_A, x, yj, yidx, key)
            params, opt_state, _ = step(params, opt_state, x, yidx, C_A, M, beta, g)
        sep = float(bc.ridge_acc(hard_reps(params, Xfa), Yoh, hard_reps(params, Xfe), ye))
        if sep > best_sep:
            best_sep, best_params = sep, params
    params = best_params

    # ---- hard-sign measurement ----
    state_tmpl, orch = cm.build_model(cfg, jax.random.PRNGKey(0))
    orch = bc.orch_with_kernels(orch, params)
    orch = eqx.tree_at(lambda o: o.lmap[2][1].W, orch, params["wout"])
    if args.max_batches is not None:
        Xtr, Ytr, Xte, Yte = _collectD_lim(orch, state_tmpl, ds, cfg, args, key)
    else:
        Xtr, Ytr, Xte, Yte, key = bc.collect_D(orch, state_tmpl, ds, cfg, key)
    ytr, yte = np.argmax(Ytr, 1), np.argmax(Yte, 1)
    W0 = np.asarray(orch.lmap[2][1].W)
    wout_acc = bc.offline_wout(cfg, Xtr, Ytr, W0, Xte, yte, args.readout_epochs)[-1]
    probe_acc = max(bc.offline_probe(Xtr, ytr, Xte, yte, args.probe_epochs))
    cprobe_transfer = float((Xte @ Wp_A).argmax(1).__eq__(yte).mean())   # teacher pooled C-probe on student D

    # C_A-dependent spin metrics on a fixed test subset
    win_s, j1_s = orch.lmap[1][0], orch.lmap[1][1]
    rollD = eqx.filter_jit(cm.make_rollout(warmup, 0, free))
    ov, fmarg, fliprate = [], [], []
    for xb, yb in limited(ds.iter_test(), args.subset_batches if args.max_batches is None else args.max_batches):
        x = cm.to_hwc(xb); yj = jnp.asarray(np.asarray(yb)); yidx = jnp.asarray(np.argmax(np.asarray(yb), 1))
        sD, _ = rollD(orch, state_tmpl.init(x, yj), key)
        D = np.asarray(sD[1])
        C_A, M = tt(W_A, x, yj, yidx, key); C_A = np.asarray(C_A); M = np.asarray(M)
        hfield = np.asarray(win_s(x) + j1_s(jnp.asarray(D)))
        ov.append(overlap(D, C_A))
        fmarg.append((C_A * hfield)[M])
        fliprate.append((np.sign(D) != C_A)[M])
    res = {
        "probe_acc": probe_acc, "wout_acc": float(wout_acc),
        "cprobe_transfer_D": cprobe_transfer,
        "overlap_D_CA": float(np.concatenate(ov).mean()),
        "field_margin_on_M": float(np.concatenate(fmarg).mean()),
        "flip_rate_on_M": float(np.concatenate(fliprate).mean()),
        "best_sep": best_sep,
    }
    print(f"    [{tag} g{gamma} s{seed}] probeD={probe_acc:.3f} wout={wout_acc:.3f} "
          f"Ctransfer={cprobe_transfer:.3f} ov(D,CA)={res['overlap_D_CA']:.3f} "
          f"margin={res['field_margin_on_M']:.2f} flipM={res['flip_rate_on_M']:.3f} "
          f"({cm.fmt(time.time()-t0)})")
    return res


def _collectD_lim(orch, state_tmpl, ds, cfg, args, key):
    warmup = cfg.get("warmup_n_iter", 1)
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, cfg["free_n_iter"]))

    def grab(it):
        X, Y, k = [], [], key
        for xb, yb in limited(it, args.max_batches):
            s, k = roll(orch, state_tmpl.init(cm.to_hwc(xb), yb), k)
            X.append(np.asarray(cm.pool_j1(np.asarray(s[1])))); Y.append(np.asarray(yb))
        return np.concatenate(X), np.concatenate(Y)

    Xtr, Ytr = grab(ds); Xte, Yte = grab(ds.iter_test())
    return Xtr, Ytr, Xte, Yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--bptt-epochs", type=int, default=30)
    ap.add_argument("--beta-max", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--readout-epochs", type=int, default=10)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--fullspin-wd", type=float, default=1e-3)
    ap.add_argument("--proxy-batches", type=int, default=128)
    ap.add_argument("--probe-train-batches", type=int, default=400)
    ap.add_argument("--subset-batches", type=int, default=63)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.gammas = [1.0]; args.bptt_epochs = 2
        args.readout_epochs = 2; args.probe_epochs = 2; args.probe_train_batches = 6
        args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    t0 = time.time()

    results = {"config": "best_channel_entropy", "seeds": args.seeds, "k": args.k,
               "kappa": KAPPA, "experiments": {"exp1_endpoint": {}, "exp2_trajectory": {}}}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        teacher = load_teacher(seed, cfg)
        params0 = {"win": np.asarray(teacher.lmap[1][0].kernel),
                   "j1": np.asarray(teacher.lmap[1][1].kernel),
                   "wout": np.asarray(teacher.lmap[2][1].W)}
        key = jax.random.PRNGKey(seed)
        Ctr, ytr_t = collect_teacher_C(teacher, state_tmpl, ds, cfg, key, args.probe_train_batches)
        W_A = jnp.asarray(fit_probe(Ctr.reshape(Ctr.shape[0], -1), ytr_t, args.probe_epochs, args.fullspin_wd))
        Wp_A = fit_probe(np.asarray(cm.pool_j1(jnp.asarray(Ctr))), ytr_t, args.probe_epochs, cm.PROBE_WD)
        print(f"  teacher probes fit ({cm.fmt(time.time()-t0)})")
        tt = make_teacher_targets(teacher, state_tmpl, cfg, args.k)
        for exp1, name in [(True, "exp1_endpoint"), (False, "exp2_trajectory")]:
            for g in args.gammas:
                r = run_student(teacher, params0, W_A, Wp_A, tt, cfg, ds, exp1, g, seed, args, t0, name)
                results["experiments"][name].setdefault(str(g), []).append(r)

    # aggregate
    for name, gd in results["experiments"].items():
        for g, lst in gd.items():
            ks = lst[0].keys()
            gd[g] = {"per_seed": lst,
                     "mean": {k: float(np.mean([p[k] for p in lst])) for k in ks},
                     "std": {k: float(np.std([p[k] for p in lst])) for k in ks}}
    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "field_loss.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time()-t0)})")
    print("Reference: plain BPTT probe 0.506; teacher A C-probe-on-D ~0.26, acc_C ~0.96.")


if __name__ == "__main__":
    main()
