"""Train and SERIALIZE the three models for the C/D diagnostics.

The exp 1/2 runs only saved scalar curves, not weights, so we (re)train one
instance of each model per seed and dump the full orchestrator with
eqx.tree_serialise_leaves. This is 3 short trainings per seed, NOT the 36-run
sweep. After this, diagnostics.py loads the models and computes everything offline.

  A = standard local-rule        (DynamicalTrainer; perceptron + entropy rules)
  B = BPTT CE_D                  (exp 1; no regularizer)
  C = BPTT CE_D + alpha*align    (exp 2 winner-style: ce_D_reg_pool, alpha=0.3)

Saves models/<A|B|C>_seed<seed>.eqx (+ a small meta.json).

Run (cluster):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/3-CD_diagnostics/train_models.py
Smoke:  python p2_representation/3-CD_diagnostics/train_models.py --smoke
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
MODELS_DIR = HERE / "models"

# model C = the exp-2 alignment variant chosen for the diagnostics
C_CE_ON, C_REG_SPACE, C_ALPHA = "D", "pool", 0.3


def set_all_kernels(orch, params):
    """Write BPTT-optimised W_in/J1/W_out back into an orchestrator."""
    orch = bc.orch_with_kernels(orch, params)              # win, j1
    return eqx.tree_at(lambda o: o.lmap[2][1].W, orch, params["wout"])


def _materialize(it, n):
    X, Y = [], []
    for i, (xb, yb) in enumerate(it):
        if n is not None and i >= n:
            break
        X.append(cm.to_hwc(xb)); Y.append(np.asarray(yb))
    return jnp.asarray(np.concatenate(X)), np.concatenate(Y)


def train_local(cfg, ds, seed, args):
    """Model A: standard local-rule training (DynamicalTrainer, clamped rollout)."""
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = cm.build_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)  # clamped = cfg default
    for _ in range(args.epochs_a):
        trainer, key = cm.train_epoch(trainer, ds, key, decay_rate=cfg["kernel_decay_rate"])
    return trainer.orchestrator


def train_bptt(cfg, ds, seed, args, reg):
    """Models B (reg=None) and C (reg=(ce_on,reg_space,alpha)). tanh surrogate,
    beta annealed 1->4, best backbone checkpointed by hard-sign separability."""
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    _, orch = cm.build_model(cfg, mk)
    win, j1, wout, params = bc.extract_kernels(orch)
    wback = orch.lmap[1][2]
    warmup = cfg.get("warmup_n_iter", 1)
    clamped, free = cfg["clamped_n_iter"], cfg["free_n_iter"]
    n_steps = warmup + free

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)
    if reg is None:
        step = bc.make_bptt_step(win, j1, wout, n_steps, "tanh", opt)
    else:
        ce_on, reg_space, _ = reg
        step = bc.make_reg_bptt_step(win, j1, wback, wout, warmup, clamped, free,
                                     "tanh", ce_on, reg_space, opt)
    hard_reps = bc.make_hard_reps(win, j1, n_steps)

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
            x = cm.to_hwc(xb)
            y = jnp.asarray(np.asarray(yb))
            y_idx = jnp.asarray(np.argmax(np.asarray(yb), axis=1))
            if reg is None:
                params, opt_state, _ = step(params, opt_state, x, y_idx, beta)
            else:
                params, opt_state, _ = step(params, opt_state, x, y, y_idx, beta,
                                            jnp.asarray(reg[2]))
        sep = float(bc.ridge_acc(hard_reps(params, Xf), Yf_oh, hard_reps(params, Xe), ye_idx))
        if sep > best_sep:
            best_sep, best_params = sep, params
    return set_all_kernels(orch, best_params), best_sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs-a", type=int, default=20)
    ap.add_argument("--bptt-epochs", type=int, default=30)
    ap.add_argument("--beta-max", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs_a = 2; args.bptt_epochs = 2; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        orchA = train_local(cfg, ds, seed, args)
        print(f"  A (local-rule) trained  ({cm.fmt(time.time() - t0)})")
        orchB, sepB = train_bptt(cfg, ds, seed, args, reg=None)
        print(f"  B (BPTT CE_D) trained, best sep {sepB:.4f}  ({cm.fmt(time.time() - t0)})")
        orchC, sepC = train_bptt(cfg, ds, seed, args, reg=(C_CE_ON, C_REG_SPACE, C_ALPHA))
        print(f"  C (BPTT CE_D+align) trained, best sep {sepC:.4f}  ({cm.fmt(time.time() - t0)})")
        for name, orch in [("A", orchA), ("B", orchB), ("C", orchC)]:
            eqx.tree_serialise_leaves(MODELS_DIR / f"{name}_seed{seed}.eqx", orch)
        print(f"  serialized A/B/C for seed {seed}")

    meta = {"config": "best_channel_entropy", "seeds": args.seeds,
            "model_C": {"ce_on": C_CE_ON, "reg_space": C_REG_SPACE, "alpha": C_ALPHA},
            "args": vars(args)}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(args.seeds)*3} models to {MODELS_DIR}  (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
