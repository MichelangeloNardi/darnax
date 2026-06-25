"""p2_representation/2-BPTT_clamped_reg/run.py

BPTT ceiling on D with a clamped-distance regularizer. Same diagnostic as exp 1
(BPTT a differentiable rollout, measure the hard-sign ceiling on D), but the loss
adds a term pulling the free inference state D toward the clamped (label-injected)
state C, with C as a stop-gradient target:

    loss = CE(W_out . pool(<ce_state>), y) + alpha * MSE( <space>(D), sg(<space>(C)) )

Four variants (ce_state x reg_space):
    ce_D_reg_pool : CE on D + alpha || pool(D) - sg(pool(C)) ||^2
    ce_C_reg_pool : CE on C + alpha || pool(D) - sg(pool(C)) ||^2
    ce_D_reg_full : CE on D + alpha || D - sg(C) ||^2        (full 32x32x16 spins)
    ce_C_reg_full : CE on C + alpha || D - sg(C) ||^2

Distance is per-element MSE (mean over batch and features) so alpha is comparable
across pooled and full-spin variants. tanh surrogate only (STE collapsed in exp 1);
beta annealed 1->4; best backbone checkpointed by the hard-sign separability proxy;
ceiling always measured on the TRUE hard-sign D dynamics (perceptron W_out + Adam
probe). Random init, best_channel_entropy cfg, 3 seeds, alpha in {0.1,0.3,1.0}.

Run (cluster, from repo root):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/2-BPTT_clamped_reg/run.py
Smoke:
  python p2_representation/2-BPTT_clamped_reg/run.py --smoke
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
KIND = "tanh"
VARIANTS = [
    ("ce_D_reg_pool", "D", "pool"),
    ("ce_C_reg_pool", "C", "pool"),
    ("ce_D_reg_full", "D", "full"),
    ("ce_C_reg_full", "C", "full"),
]


def beta_schedule(epochs, beta_max):
    return np.geomspace(1.0, beta_max, epochs).astype(np.float32)


def limited(it, n):
    if n is None:
        yield from it
        return
    for i, b in enumerate(it):
        if i >= n:
            return
        yield b


def materialize(it, n):
    """Pull n batches into stacked (X_hwc, Y_pm1) arrays."""
    X, Y = [], []
    for xb, yb in limited(it, n):
        X.append(cm.to_hwc(xb)); Y.append(np.asarray(yb))
    return jnp.asarray(np.concatenate(X)), np.concatenate(Y)


def run_one(cfg, variant, ce_on, reg_space, alpha, seed, ds, args, t0):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state_tmpl, orch = cm.build_model(cfg, mk)  # RANDOM init
    win, j1, wout, params = bc.extract_kernels(orch)
    wback = orch.lmap[1][2]

    warmup = cfg.get("warmup_n_iter", 1)
    clamped = cfg["clamped_n_iter"]
    free = cfg["free_n_iter"]
    n_steps_D = warmup + free

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)
    step = bc.make_reg_bptt_step(win, j1, wback, wout, warmup, clamped, free,
                                 KIND, ce_on, reg_space, opt)
    soft_logits = bc.make_soft_eval(win, j1, wout, n_steps_D, KIND)  # soft acc on D
    hard_reps = bc.make_hard_reps(win, j1, n_steps_D)

    betas = beta_schedule(args.bptt_epochs, args.beta_max)

    # fixed TRAIN subset for the checkpoint-selection separability proxy
    n_sub = args.max_batches if args.max_batches is not None else args.proxy_batches
    Xsub, Ysub = materialize(ds, n_sub)
    half = len(Xsub) // 2
    Xf, Xe = Xsub[:half], Xsub[half:]
    Yf_oh = jnp.asarray((np.asarray(Ysub[:half]) > 0).astype(np.float32))
    ye_idx = jnp.asarray(np.argmax(np.asarray(Ysub[half:]), axis=1))

    Xte_soft, Yte_soft = materialize(ds.iter_test(), args.max_batches)
    yte_soft = np.argmax(np.asarray(Yte_soft), axis=1)

    sep_curve, soft_curve = [], []
    best_sep, best_params, best_epoch = -1.0, params, 0
    alpha_j = jnp.asarray(alpha)
    for ep in range(args.bptt_epochs):
        beta = jnp.asarray(betas[ep])
        ep_loss = []
        for xb, yb in limited(ds, args.max_batches):
            x = cm.to_hwc(xb)
            y = jnp.asarray(np.asarray(yb))
            y_idx = jnp.asarray(np.argmax(np.asarray(yb), axis=1))
            params, opt_state, loss = step(params, opt_state, x, y, y_idx, beta, alpha_j)
            ep_loss.append(float(loss))
        sep = float(bc.ridge_acc(hard_reps(params, Xf), Yf_oh, hard_reps(params, Xe), ye_idx))
        sep_curve.append(sep)
        if sep > best_sep:
            best_sep, best_params, best_epoch = sep, params, ep + 1
        soft = float((np.asarray(soft_logits(params, Xte_soft, beta)).argmax(1) == yte_soft).mean())
        soft_curve.append(soft)
        print(f"      [{variant} a={alpha} s{seed}] ep {ep + 1:2d}/{args.bptt_epochs}  "
              f"b={float(betas[ep]):.2f}  loss={np.mean(ep_loss):.4f}  "
              f"sep={sep:.4f}  softD={soft:.4f}  ({cm.fmt(time.time() - t0)})")

    # ── hard-sign ceiling on D, on the best checkpoint ────────────────────────
    params = best_params
    orch_opt = bc.orch_with_kernels(orch, params)
    if args.max_batches is not None:
        Xtr, Ytr, Xte, Yte = _collect_D_limited(orch_opt, state_tmpl, ds, cfg, args, key)
    else:
        Xtr, Ytr, Xte, Yte, key = bc.collect_D(orch_opt, state_tmpl, ds, cfg, key)
    ytr_idx, yte_idx = np.argmax(Ytr, axis=1), np.argmax(Yte, axis=1)
    W0 = np.asarray(orch_opt.lmap[2][1].W)
    wout_curve = bc.offline_wout(cfg, Xtr, Ytr, W0, Xte, yte_idx, args.readout_epochs)
    probe_curve = bc.offline_probe(Xtr, ytr_idx, Xte, yte_idx, args.probe_epochs)

    print(f"    [{variant} a={alpha} s{seed}]  ckpt ep {best_epoch}  "
          f"HARD D: W_out={wout_curve[-1]:.4f}  probe={max(probe_curve):.4f}")
    return {
        "best_epoch": best_epoch, "best_sep": best_sep,
        "sep_curve": sep_curve, "soft_curve": soft_curve,
        "wout_curve": wout_curve, "probe_curve": probe_curve,
        "wout_final": wout_curve[-1], "probe_best": max(probe_curve),
    }


def _collect_D_limited(orch, state_tmpl, ds, cfg, args, key):
    warmup = cfg.get("warmup_n_iter", 1)
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, cfg["free_n_iter"]))

    def grab(it):
        X, Y, k = [], [], key
        for xb, yb in limited(it, args.max_batches):
            s = state_tmpl.init(cm.to_hwc(xb), yb)
            s, k = roll(orch, s, k)
            X.append(np.asarray(cm.pool_j1(np.asarray(s[1])))); Y.append(np.asarray(yb))
        return np.concatenate(X), np.concatenate(Y)

    Xtr, Ytr = grab(ds)
    Xte, Yte = grab(ds.iter_test())
    return Xtr, Ytr, Xte, Yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--bptt-epochs", type=int, default=30)
    ap.add_argument("--readout-epochs", type=int, default=10)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-max", type=float, default=4.0)
    ap.add_argument("--proxy-batches", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [0]; args.alphas = [0.3]; args.bptt_epochs = 2
        args.readout_epochs = 2; args.probe_epochs = 2; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()

    results = {"config": "best_channel_entropy", "kind": KIND, "seeds": args.seeds,
               "alphas": args.alphas, "args": vars(args), "variants": {}}
    for variant, ce_on, reg_space in VARIANTS:
        print(f"\n{'=' * 64}\nVariant: {variant}  (CE on {ce_on}, reg on {reg_space})\n{'=' * 64}")
        results["variants"][variant] = {}
        for alpha in args.alphas:
            per_seed = [run_one(cfg, variant, ce_on, reg_space, alpha, s, ds, args, t0)
                        for s in args.seeds]
            wf = [r["wout_final"] for r in per_seed]
            pb = [r["probe_best"] for r in per_seed]
            results["variants"][variant][str(alpha)] = {
                "per_seed": per_seed,
                "wout_mean": float(np.mean(wf)), "wout_std": float(np.std(wf)),
                "probe_mean": float(np.mean(pb)), "probe_std": float(np.std(pb)),
            }
            print(f"  >>> {variant} a={alpha}: W_out {np.mean(wf):.4f}±{np.std(wf):.4f}  "
                  f"probe {np.mean(pb):.4f}±{np.std(pb):.4f}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "clamped_reg.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}   (total {cm.fmt(time.time() - t0)})")
    print("Reference: BPTT no-reg (exp 1) tanh probe 0.506 / W_out 0.480; grad-free ~0.46.")


if __name__ == "__main__":
    main()
