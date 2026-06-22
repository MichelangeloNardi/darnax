"""p2_representation/1-BPTT_ceiling/run.py

Diagnostic ceiling for the inference state D via backprop-through-time.

We backprop a differentiable rollout to D (warmup -> free, forward messages only,
the sign activation replaced by a surrogate) and optimise W_in / J1 / W_out with
real gradients (jax.grad + optax Adam) to minimise CE(W_out . pool(D), y). This is
NOT darnax-faithful — it is an UPPER BOUND on what D's representation can reach.

Two surrogates are run as a bracket (see bptt_common.apply_phi):
  - "tanh": tanh(beta*x), beta annealed up over epochs.
  - "ste" : straight-through estimator (hard sign forward, identity grad).

Backbone starts from RANDOM init (pure architectural ceiling, independent of the
local rule). The ceiling is always reported on the TRUE hard-sign dynamics: after
BPTT we rebuild a real orchestrator with the optimised kernels, collect D reps,
and fit the perceptron W_out and an Adam probe on them. The soft-rollout accuracy
is logged as an internal sanity check only (not a headline number).

Compare the resulting probe/W_out to the ~0.46 gradient-free ceiling (p1).

Run (cluster, from repo root):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/1-BPTT_ceiling/run.py

Smoke (tiny CPU, no GPU):
  python p2_representation/1-BPTT_ceiling/run.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")  # TF32 corrupts sign decisions

import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))   # p1 common.py
sys.path.insert(0, str(REPO / "p2_representation"))  # bptt_common.py

import common as cm
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
SURROGATES = ["tanh", "ste"]


# ── beta annealing schedule for the tanh surrogate ────────────────────────────

def beta_schedule(kind: str, epochs: int, beta_max: float) -> np.ndarray:
    """tanh: geometric ramp 1 -> beta_max (soft -> moderately sign-like). Capped at
    a moderate value: ramping all the way to a hard sign destabilises the rollout
    (loss rises, accuracy falls). ste: beta unused (1.0)."""
    if kind == "tanh":
        return np.geomspace(1.0, beta_max, epochs).astype(np.float32)
    return np.ones(epochs, dtype=np.float32)


# ── limited iterator for the smoke test ───────────────────────────────────────

def limited(it, n):
    if n is None:
        yield from it
        return
    for i, b in enumerate(it):
        if i >= n:
            return
        yield b


# ── one (surrogate, seed) ─────────────────────────────────────────────────────

def run_one(cfg, kind, seed, ds, args, t0):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state_tmpl, orch = cm.build_model(cfg, mk)  # RANDOM init
    win, j1, wout, params = bc.extract_kernels(orch)

    n_steps = cfg.get("warmup_n_iter", 1) + cfg["free_n_iter"]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)
    step = bc.make_bptt_step(win, j1, wout, n_steps, kind, opt)
    soft_logits = bc.make_soft_eval(win, j1, wout, n_steps, kind)
    hard_reps = bc.make_hard_reps(win, j1, n_steps)

    betas = beta_schedule(kind, args.bptt_epochs, args.beta_max)

    # fixed TRAIN subset for the checkpoint-selection proxy (split fit/eval halves).
    # Selecting on train avoids leaking the test set into model selection.
    Xsub, Ysub = [], []
    n_sub = args.max_batches if args.max_batches is not None else args.proxy_batches
    for xb, yb in limited(ds, n_sub):
        Xsub.append(cm.to_hwc(xb))
        Ysub.append(np.asarray(yb))
    Xsub = jnp.asarray(np.concatenate(Xsub))
    Ysub = np.concatenate(Ysub)
    half = len(Xsub) // 2
    Xf, Xe = Xsub[:half], Xsub[half:]
    Yf_oh = jnp.asarray((Ysub[:half] > 0).astype(np.float32))   # onehot fit targets
    ye_idx = jnp.asarray(np.argmax(Ysub[half:], axis=1))

    # fixed test subset for the soft-accuracy sanity check (informational only)
    Xte_soft, yte_soft = [], []
    for xb, yb in limited(ds.iter_test(), args.max_batches):
        Xte_soft.append(cm.to_hwc(xb))
        yte_soft.append(np.argmax(np.asarray(yb), axis=1))
    Xte_soft = jnp.asarray(np.concatenate(Xte_soft))
    yte_soft = np.concatenate(yte_soft)

    soft_acc_curve, sep_curve = [], []
    best_sep, best_params, best_epoch = -1.0, params, 0
    for ep in range(args.bptt_epochs):
        beta = jnp.asarray(betas[ep])
        ep_loss = []
        for xb, yb in limited(ds, args.max_batches):
            x = cm.to_hwc(xb)
            y_idx = jnp.asarray(np.argmax(np.asarray(yb), axis=1))
            params, opt_state, loss = step(params, opt_state, x, y_idx, beta)
            ep_loss.append(float(loss))
        # checkpoint-selection proxy: hard-sign linear separability on the train subset
        sep = float(bc.ridge_acc(hard_reps(params, Xf), Yf_oh, hard_reps(params, Xe), ye_idx))
        sep_curve.append(sep)
        if sep > best_sep:
            best_sep, best_params, best_epoch = sep, params, ep + 1  # params leaves are immutable
        # soft sanity check (on the surrogate's own forward)
        logits = soft_logits(params, Xte_soft, beta)
        soft_acc = float((np.asarray(logits).argmax(1) == yte_soft).mean())
        soft_acc_curve.append(soft_acc)
        print(f"    [{kind} seed {seed}] ep {ep + 1:2d}/{args.bptt_epochs}  "
              f"beta={float(betas[ep]):.2f}  loss={np.mean(ep_loss):.4f}  "
              f"sep={sep:.4f}  soft_acc={soft_acc:.4f}  ({cm.fmt(time.time() - t0)})")

    print(f"  [{kind} seed {seed}]  best checkpoint: epoch {best_epoch} (sep={best_sep:.4f})")

    # ── hard-sign ceiling (the headline) — measured on the BEST checkpoint ─────
    params = best_params
    orch_opt = bc.orch_with_kernels(orch, params)
    if args.max_batches is not None:
        # smoke: cap rep collection by slicing a tiny ds-like loop inline
        Xtr, Ytr, Xte, Yte = _collect_D_limited(orch_opt, state_tmpl, ds, cfg, args, key)
    else:
        Xtr, Ytr, Xte, Yte, key = bc.collect_D(orch_opt, state_tmpl, ds, cfg, key)

    ytr_idx = np.argmax(Ytr, axis=1)
    yte_idx = np.argmax(Yte, axis=1)
    W0 = np.asarray(orch_opt.lmap[2][1].W)

    wout_curve = bc.offline_wout(cfg, Xtr, Ytr, W0, Xte, yte_idx, args.readout_epochs)
    probe_curve = bc.offline_probe(Xtr, ytr_idx, Xte, yte_idx, args.probe_epochs)

    print(f"  [{kind} seed {seed}]  HARD-SIGN ceiling on D:  "
          f"W_out={wout_curve[-1]:.4f}  probe={max(probe_curve):.4f}")

    return {
        "soft_acc_curve": soft_acc_curve,
        "sep_curve": sep_curve,
        "best_epoch": best_epoch,
        "best_sep": best_sep,
        "wout_curve": wout_curve,
        "probe_curve": probe_curve,
        "wout_final": wout_curve[-1],
        "probe_best": max(probe_curve),
    }


def _collect_D_limited(orch, state_tmpl, ds, cfg, args, key):
    """Smoke-only: hard-sign D reps over a few batches (train + test)."""
    import equinox as eqx
    warmup = cfg.get("warmup_n_iter", 1)
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, cfg["free_n_iter"]))

    def grab(it):
        X, Y = [], []
        k = key
        for xb, yb in limited(it, args.max_batches):
            s = state_tmpl.init(cm.to_hwc(xb), yb)
            s, k = roll(orch, s, k)
            X.append(np.asarray(cm.pool_j1(np.asarray(s[1]))))
            Y.append(np.asarray(yb))
        return np.concatenate(X), np.concatenate(Y)

    Xtr, Ytr = grab(ds)
    Xte, Yte = grab(ds.iter_test())
    return Xtr, Ytr, Xte, Yte


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--bptt-epochs", type=int, default=30)
    ap.add_argument("--readout-epochs", type=int, default=10)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-max", type=float, default=4.0,
                    help="tanh surrogate: final (capped) inverse temperature")
    ap.add_argument("--proxy-batches", type=int, default=128,
                    help="train batches for the checkpoint-selection separability proxy")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches per pass (smoke only)")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [0]
        args.bptt_epochs = 2
        args.readout_epochs = 2
        args.probe_epochs = 2
        args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)

    t0 = time.time()
    results: dict = {"config": "best_channel_entropy", "seeds": args.seeds,
                     "args": vars(args), "surrogates": {}}
    for kind in SURROGATES:
        print(f"\n{'=' * 60}\nSurrogate: {kind}\n{'=' * 60}")
        per_seed = []
        for seed in args.seeds:
            per_seed.append(run_one(cfg, kind, seed, ds, args, t0))
        wout_finals = [r["wout_final"] for r in per_seed]
        probe_bests = [r["probe_best"] for r in per_seed]
        results["surrogates"][kind] = {
            "per_seed": per_seed,
            "wout_mean": float(np.mean(wout_finals)),
            "wout_std": float(np.std(wout_finals)),
            "probe_mean": float(np.mean(probe_bests)),
            "probe_std": float(np.std(probe_bests)),
        }
        print(f"\n  >>> {kind}: W_out {np.mean(wout_finals):.4f} ± {np.std(wout_finals):.4f}"
              f"   probe {np.mean(probe_bests):.4f} ± {np.std(probe_bests):.4f}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "ceiling.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}   (total {cm.fmt(time.time() - t0)})")
    print("\nReference (p1, gradient-free): W_out ~0.44, probe ~0.46 on D")


if __name__ == "__main__":
    main()
