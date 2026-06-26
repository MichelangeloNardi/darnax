"""Train + SERIALIZE the exp-6 models: the raw-clamp baseline and the two instant
PC-feedback variants (wback, wout) over a small beta grid, per seed.

  raw            standard local rule, static W_back(y) clamp     (== exp-3 model A)
  wback_b<beta>  PC feedback, W_back-tied (instant)
  wout_b<beta>   PC feedback, W_out-based (instant)

All are trained with the SAME local rules (perceptron + entropy) and per-edge lrs from
best_channel_entropy; only the clamped-phase external field differs. Serializes
models/<tag>_seed<seed>.eqx, logs a t=0 field-scale calibration vs the raw clamp, and
writes select.json picking the best beta per variant by a quick hard-sign D-ridge proxy
(so diagnostics.py can default to baseline + the selected betas, ~exp-3 scale).

Run (cluster):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/6-pc_feedback/train_models.py
Smoke:  python p2_representation/6-pc_feedback/train_models.py --smoke
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import pc_feedback as pcf

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"
VARIANTS = ["wback", "wout"]


@eqx.filter_jit
def ridge_acc(Xf, Yf_oh, Xe, ye_idx, lam=1.0):
    """Closed-form ridge readout fit on (Xf, Yf_oh), accuracy on (Xe, ye_idx).
    Inlined from bptt_common to keep training free of the torch import."""
    d = Xf.shape[1]
    W = jnp.linalg.solve(Xf.T @ Xf + lam * jnp.eye(d), Xf.T @ Yf_oh)
    return jnp.mean((Xe @ W).argmax(1) == ye_idx)


def train_raw(cfg, ds, seed, epochs):
    """Baseline: standard local-rule training with the static W_back(y) clamp."""
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = cm.build_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)   # clamped = cfg default
    for _ in range(epochs):
        trainer, key = cm.train_epoch(trainer, ds, key, decay_rate=cfg["kernel_decay_rate"])
    return trainer.orchestrator


def collect_pooled_D(orch, ds_iter, state_tmpl, cfg, key, max_b):
    """Pooled D reps (warmup->free, hard sign) + pm1 labels, from limited batches."""
    warmup = cfg.get("warmup_n_iter", 1)
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, cfg["free_n_iter"]))
    X, Y = [], []
    for i, (xb, yb) in enumerate(ds_iter):
        if i >= max_b:
            break
        s, key = roll(orch, state_tmpl.init(cm.to_hwc(xb), yb), key)
        X.append(np.asarray(cm.pool_j1(np.asarray(s[1]))))
        Y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(Y), key


def d_proxy(orch, ds, state_tmpl, cfg, max_b):
    """Quick hard-sign D separability: closed-form ridge readout on pooled D."""
    key = jax.random.PRNGKey(0)
    Xtr, Ytr, key = collect_pooled_D(orch, ds, state_tmpl, cfg, key, max_b)
    Xte, Yte, key = collect_pooled_D(orch, ds.iter_test(), state_tmpl, cfg, key, max_b // 2)
    return float(ridge_acc(
        jnp.asarray(Xtr), jnp.asarray((Ytr > 0).astype(np.float32)),
        jnp.asarray(Xte), jnp.asarray(np.argmax(Yte, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    # beta grid brackets the raw-clamp field magnitude: smoke calibration showed
    # rms(b_t) ~= 16.3*beta, raw clamp ~= 0.91, so beta~0.06 matches; the grid spans
    # ~0.8x (b0.05) -> ~16x (b1.0) the static-clamp strength.
    ap.add_argument("--betas", type=float, nargs="+", default=[0.05, 0.1, 0.3, 1.0])
    ap.add_argument("--proxy-batches", type=int, default=64)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 2; args.betas = [0.3]; args.proxy_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    MODELS_DIR.mkdir(exist_ok=True)
    xb0, yb0 = next(iter(ds))
    x0, y0 = cm.to_hwc(xb0), jnp.asarray(np.asarray(yb0))
    t0 = time.time()

    proxies: dict[str, dict[str, float]] = {}  # tag -> {seed: D-proxy}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        models = {"raw": train_raw(cfg, ds, seed, args.epochs)}
        print(f"  raw (static clamp) trained  ({cm.fmt(time.time() - t0)})")
        for variant in VARIANTS:
            for beta in args.betas:
                g = beta * (pcf.N_GAIN ** 0.5)
                orch = pcf.train_pc_model(cfg, ds, seed, variant, beta, args.epochs)
                tag = f"{variant}_b{beta}"
                models[tag] = orch
                rb, rraw = pcf.field_scales(orch, state_tmpl, x0, y0, variant, g)
                print(f"  {tag} trained  (g={g:.1f}; rms b_t={rb:.3f} vs raw clamp "
                      f"{rraw:.3f})  ({cm.fmt(time.time() - t0)})")
        for tag, orch in models.items():
            eqx.tree_serialise_leaves(MODELS_DIR / f"{tag}_seed{seed}.eqx", orch)
            proxies.setdefault(tag, {})[str(seed)] = d_proxy(
                orch, ds, state_tmpl, cfg, args.proxy_batches)
        print(f"  serialized {len(models)} models; D-proxy(raw)="
              f"{proxies['raw'][str(seed)]:.3f}  ({cm.fmt(time.time() - t0)})")

    # select best beta per variant by mean-over-seeds D-proxy
    proxy_mean = {tag: float(np.mean(list(d.values()))) for tag, d in proxies.items()}
    selected = {}
    for variant in VARIANTS:
        cands = {tag: m for tag, m in proxy_mean.items() if tag.startswith(variant + "_b")}
        selected[variant] = max(cands, key=cands.get)
    (MODELS_DIR / "select.json").write_text(json.dumps({
        "config": "best_channel_entropy", "seeds": args.seeds, "betas": args.betas,
        "variants": VARIANTS, "selected": selected,
        "proxy_mean": proxy_mean, "proxy_per_seed": proxies,
    }, indent=2))
    print(f"\nselected: {selected}")
    print(f"proxy_mean: {json.dumps(proxy_mean, indent=2)}")
    print(f"Saved models + select.json to {MODELS_DIR}  (total {cm.fmt(time.time() - t0)})")


if __name__ == "__main__":
    main()
