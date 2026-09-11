"""Exp 16 step 1 — BPTT ceiling screen for the associative-augmentation architectures.

Screen each config's REPRESENTATIONAL CEILING with BPTT (real gradients) before investing in
local-rule tuning: if adding recurrent-only ("associative middle") neurons on top of a fixed
16-input core does not raise the achievable probe_D even with gradients, the architecture does
not benefit and the local rule won't either.

For each config (arch16): tanh-BPTT on CE(W_out.pool(D), y) with W_in's gradient masked to the
input-only group I (j_d frozen, W_out free), best-checkpointed by a C-aware hard-sign proxy;
then measure the TRUE hard-sign probe_D. Reuses exp-11 make_split_bptt_step + exp-13
measure_probeD.

Question: does the BPTT probe_D rise as N (recurrent-only) grows, above classic_c16's ceiling?

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/16-associative_aug/bptt_screen.py --name aug_L8_N16
Smoke:  python p2_representation/16-associative_aug/bptt_screen.py --name classic_c16 --smoke
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
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in ["src", "p1_readout_gap", "p2_representation", "p2_representation/10-split_scale",
          "p2_representation/11-bptt_split", "p2_representation/16-associative_aug"]:
    sys.path.insert(0, str(REPO / p))

import common as cm
import bptt_common as bc
import splitarch as S
import arch16 as A16

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


B11 = _load(REPO / "p2_representation" / "11-bptt_split" / "bptt_train.py", "exp11_bptt")
T = _load(REPO / "p2_representation" / "13-hp_tuning" / "tune.py", "exp13_tune")


def screen_one(cfg, name, seed, args):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    _, orch = A16.build(cfg, mk, name)
    win, j1, wout, params = bc.extract_kernels(orch)      # win.kernel already masked to I
    warmup, free = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"]
    n_steps = warmup + free
    wmask = A16.win_mask(name)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)
    step = B11.make_split_bptt_step(win, j1, wout, n_steps, "tanh", opt, wmask)
    hard_reps = S.make_hard_reps_caware(win, j1, n_steps)

    n_sub = args.max_batches if args.max_batches is not None else 128
    Xs, Ys = [], []
    ds = args._ds
    for i, (xb, yb) in enumerate(ds):
        if i >= n_sub:
            break
        Xs.append(cm.to_hwc(xb)); Ys.append(np.asarray(yb))
    Xsub = jnp.asarray(np.concatenate(Xs)); Ysub = np.concatenate(Ys)
    half = len(Xsub) // 2
    Xf, Xe = Xsub[:half], Xsub[half:]
    Yf_oh = jnp.asarray((np.asarray(Ysub[:half]) > 0).astype(np.float32))
    ye_idx = jnp.asarray(np.argmax(np.asarray(Ysub[half:]), axis=1))

    betas = np.geomspace(1.0, args.beta_max, args.bptt_epochs).astype(np.float32)
    best_sep, best_params = -1.0, params
    for ep in range(args.bptt_epochs):
        beta = jnp.asarray(betas[ep])
        for i, (xb, yb) in enumerate(ds):
            if args.max_batches is not None and i >= args.max_batches:
                break
            params, opt_state, _ = step(params, opt_state, cm.to_hwc(xb),
                                        jnp.asarray(np.argmax(np.asarray(yb), 1)), beta)
        sep = float(bc.ridge_acc(hard_reps(params, Xf), Yf_oh, hard_reps(params, Xe), ye_idx))
        if sep > best_sep:
            best_sep, best_params = sep, params

    # hard-sign ceiling: real orchestrator with optimised kernels, C-aware probe_D
    state_tmpl, orch = A16.build(cfg, jax.random.PRNGKey(0), name)
    orch = bc.orch_with_kernels(orch, best_params)
    orch = eqx.tree_at(lambda o: o.lmap[2][1].W, orch, best_params["wout"])
    probe_D = T.measure_probeD(orch, state_tmpl, ds, cfg, key, args.probe_epochs,
                               args.train_batches, args.test_batches)
    return float(probe_D), best_sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=A16.NAMES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--bptt-epochs", type=int, default=30)
    ap.add_argument("--beta-max", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--test-batches", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.bptt_epochs = 2; args.probe_epochs = 2
        args.train_batches = 6; args.test_batches = 4; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    args._ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    A16.param_report(args.name)
    pDs, seps = [], []
    for seed in args.seeds:
        pD, sep = screen_one(cfg, args.name, seed, args)
        pDs.append(pD); seps.append(sep)
        print(f"  [{args.name} bptt s{seed}] probe_D={pD:.4f} best_sep={sep:.4f} ({cm.fmt(time.time()-t0)})")

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out = {"name": args.name, "C": A16.channels_of(args.name), "seeds": args.seeds,
           "probe_D_mean": float(np.mean(pDs)), "probe_D_std": float(np.std(pDs)),
           "probe_D_seeds": pDs, "best_sep_seeds": seps}
    (out_dir / (f"screen_{args.name}{'_smoke' if args.smoke else ''}.json")).write_text(json.dumps(out, indent=2))
    print(f"\nBPTT ceiling {args.name}: probe_D={np.mean(pDs):.4f}  (ref classic_c16 BPTT ~0.51) "
          f"(total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
