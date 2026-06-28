"""Exp 11 part 1 — BPTT split-routing diagnostic.

Train the split_C24_I16_L8_N0 / split_C32_I16_L8_N8 architectures with the tanh-BPTT
inference objective CE(W_out . pool(D), y) (the exp-1 ceiling method), enforcing the split
wiring: W_in's gradient is masked to group I (only I channels receive input), J1's j_d
diagonal is frozen, W_out is free. tanh surrogate (beta 1->4), best backbone checkpointed
by a C-aware hard-sign separability proxy. Serializes the optimised split orchestrators;
diagnostics.py then measures the TRUE hard-sign D (probe_D, head_D, per-group I/L/N probes,
C/D overlap).

Question: can real gradients make the L/N channels class-informative at D, given W_in only
feeds I (so L/N can only get input information THROUGH the J1 recurrence)?

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/11-bptt_split/bptt_train.py
Smoke:  python p2_representation/11-bptt_split/bptt_train.py --smoke
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
sys.path.insert(0, str(REPO / "p2_representation" / "10-split_scale"))

import common as cm
import bptt_common as bc
import arch as A
import splitarch as S

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"


def make_split_bptt_step(win, j1, wout, n_steps, kind, opt, win_mask):
    """bc.make_bptt_step + a W_in channel-output gradient mask (keeps W_in confined to
    group I; non-I columns start at 0 and never move)."""
    grad_fn = jax.value_and_grad(bc._ce_loss)
    j1_mask = j1.update_mask

    @eqx.filter_jit
    def step(params, opt_state, x, y_idx, beta):
        loss, grads = grad_fn(params, win, j1, wout, x, y_idx, n_steps, kind, beta)
        grads = {**grads, "j1": grads["j1"] * j1_mask, "win": grads["win"] * win_mask}
        upd, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, upd)
        params = {**params, "j1": j1._apply_jd_constraint(params["j1"])}
        return params, opt_state, loss

    return step


def _materialize(it, n):
    X, Y = [], []
    for i, (xb, yb) in enumerate(it):
        if n is not None and i >= n:
            break
        X.append(cm.to_hwc(xb)); Y.append(np.asarray(yb))
    return jnp.asarray(np.concatenate(X)), np.concatenate(Y)


def train_one(cfg, ds, name, seed, args):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    _, orch = A.build_model(cfg, mk, name)
    win, j1, wout, params = bc.extract_kernels(orch)        # win.kernel already masked to I
    warmup, free = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"]
    n_steps = warmup + free
    win_mask = S.win_mask_for(name)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)
    step = make_split_bptt_step(win, j1, wout, n_steps, "tanh", opt, win_mask)
    hard_reps = S.make_hard_reps_caware(win, j1, n_steps)

    n_sub = args.max_batches if args.max_batches is not None else 128
    Xsub, Ysub = _materialize(ds, n_sub)
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

    # write optimised kernels into a real split orchestrator (W_in stays masked to I)
    orch = bc.orch_with_kernels(orch, best_params)
    orch = eqx.tree_at(lambda o: o.lmap[2][1].W, orch, best_params["wout"])
    return orch, best_sep


def assert_win_masked(orch, name):
    C = A.channels_of(name)
    I, _, _ = A.group_indices(name)
    win = np.asarray(orch.lmap[1][0].kernel)
    nonI = [c for c in range(C) if c not in set(I.tolist())]
    if nonI:
        assert np.abs(win[:, :, :, nonI]).sum() == 0.0, f"{name}: BPTT leaked W_in into non-I"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--names", type=str, nargs="+", default=S.BPTT_CONFIGS)
    ap.add_argument("--bptt-epochs", type=int, default=30)
    ap.add_argument("--beta-max", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.bptt_epochs = 2; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    meta = {"config": "best_channel_entropy", "seeds": args.seeds, "names": args.names,
            "objective": "CE(Wout.pool(D),y) tanh-BPTT, W_in masked to I, j_d frozen",
            "best_sep": {}}
    for name in args.names:
        for seed in args.seeds:
            orch, sep = train_one(cfg, ds, name, seed, args)
            assert_win_masked(orch, name)
            eqx.tree_serialise_leaves(MODELS_DIR / f"{name}_bptt_seed{seed}.eqx", orch)
            meta["best_sep"].setdefault(name, {})[str(seed)] = sep
            print(f"  [{name:21s} bptt s{seed}] best_sep={sep:.4f} masks_ok  ({cm.fmt(time.time() - t0)})")

    (MODELS_DIR / "bptt_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(args.seeds) * len(args.names)} BPTT split models  (total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
