"""Exp 12 — Contrastive Hebbian / EP-style local rule on the standard entropy architecture.

Motivation (exp 11): under BPTT the split L/N channels become class-informative at D, but the
gradient-free clamped-only local rule (DynamicalTrainer) leaves them at chance. The contrastive
rule is the local, gradient-free approximation of that BPTT credit assignment: it computes the
local update at BOTH a clamped (label-on) state and a free state and uses the DIFFERENCE, which
propagates the label signal through the recurrence to every channel.

The darnax ContrastiveHebbianTrainer is not drop-in here (it uses filter_messages="left" and
omits the conv step's t_win/t_back), so we implement the contrastive loop directly, mirroring
the working DynamicalTrainer plumbing (orchestrator.step forward/all + orchestrator.backward +
make_optimizer):

  per batch:
    s0 = warmup (forward)                       # shared warmup
    A  = s0 -> free   (forward)                 # free state (no label)
    B  = s0 -> clamped (all, label via W_back)  # clamped/nudged state
    grad = backward(B) - backward(A)            # contrastive difference  (--flip swaps)
    optimizer step (make_optimizer signed lrs)

NOTE (per the per-rule HP-tuning rule): this run uses best_channel_entropy_cfg as a STARTING
point only. That config is tuned for the DynamicalTrainer; a fair CHL vs baseline comparison
needs CHL-specific HP tuning (phase lengths + lr) as a follow-up. This script is the functional
test: does the contrastive rule run, train, and lift D-probe / make channels informative?

Reports per epoch: head_acc_D (model W_out on D) and probe_D (Adam linear probe on pooled D),
for CHL and (for reference, same config/seed) the DynamicalTrainer baseline.

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/12-chl_rule/run.py
Smoke:  python p2_representation/12-chl_rule/run.py --smoke
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"


def _roll(orch, state, key, n, filt):
    for _ in range(n):
        state, key = orch.step(state, rng=key, filter_messages=filt)
    return state, key


@eqx.filter_jit
def chl_grads(orch, state, key, warmup, free, clamped, flip):
    """Contrastive grad = backward(clamped) - backward(free) (flip swaps), shaped like orch."""
    s0, key = _roll(orch, state, key, warmup, "forward")
    sA, kA = _roll(orch, s0, key, free, "forward")           # free state
    sB, kB = _roll(orch, s0, key, clamped, "all")            # clamped/nudged state
    dA = orch.backward(sA, rng=kA)
    dB = orch.backward(sB, rng=kB)
    lo, hi = (dA, dB) if not flip else (dB, dA)
    grads = jax.tree_util.tree_map(lambda b, a: b - a, hi, lo)
    return grads, kB


def chl_epoch(orch, opt, opt_state, state_tmpl, ds, key, cfg, flip, max_batches):
    warmup, free, clamped = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"], cfg["clamped_n_iter"]
    for i, (xb, yb) in enumerate(ds):
        if max_batches is not None and i >= max_batches:
            break
        state = state_tmpl.init(cm.to_hwc(xb), yb)
        grads, key = chl_grads(orch, state, key, warmup, free, clamped, flip)
        params = eqx.filter(orch, eqx.is_inexact_array)
        grads = eqx.filter(grads, eqx.is_inexact_array)
        updates, opt_state = opt.update(grads, opt_state, params=params)
        orch = eqx.apply_updates(orch, updates)
    # kernel decay (as in cm.train_epoch)
    dr = cfg["kernel_decay_rate"]
    if dr > 0:
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            orch = eqx.tree_at(path, orch, path(orch) * (1.0 - dr))
    return orch, opt_state, key


def _collect_capped(orch, state_tmpl, it, roll, key, max_b):
    X, Y = [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        state = state_tmpl.init(cm.to_hwc(xb), yb)
        state, key = roll(orch, state, key)
        X.append(np.asarray(cm.pool_j1(np.asarray(state[1])))); Y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(Y), key


def measure(orch, state_tmpl, ds, cfg, key, probe_epochs, probe_train_batches=400):
    """probe_D (Adam linear probe on pooled D) and head_acc_D (model W_out on D)."""
    warmup, free = cfg.get("warmup_n_iter", 1), cfg["free_n_iter"]
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, free))
    Xtr, Ytr, key = _collect_capped(orch, state_tmpl, ds, roll, key, probe_train_batches)
    Xte, Yte, key = _collect_capped(orch, state_tmpl, ds.iter_test(), roll, key, None)
    ytr, yte = np.argmax(Ytr, 1), np.argmax(Yte, 1)
    probe_D = max(bc.offline_probe(Xtr, ytr, Xte, yte, probe_epochs))
    Wout = np.asarray(orch.lmap[2][1].W)
    head = float((Xte @ Wout).argmax(1).__eq__(yte).mean())
    return probe_D, head, key


def train_chl(cfg, ds, seed, args):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state_tmpl, orch = cm.build_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    hist = []
    for ep in range(args.epochs):
        orch, opt_state, key = chl_epoch(orch, opt, opt_state, state_tmpl, ds, key, cfg,
                                         args.flip, args.max_batches)
        pD, head, key = measure(orch, state_tmpl, ds, cfg, key, args.probe_epochs)
        hist.append({"epoch": ep, "probe_D": pD, "head_acc_D": head})
    return orch, hist, key


def train_baseline(cfg, ds, seed, args):
    """DynamicalTrainer (current rule) at the same config/seed, for reference."""
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state_tmpl, orch = cm.build_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    trainer = cm.make_trainer(orch, state_tmpl, opt, opt_state, cfg)
    hist = []
    for ep in range(args.epochs):
        for i, (xb, yb) in enumerate(ds):
            if args.max_batches is not None and i >= args.max_batches:
                break
            key = trainer.train_step(cm.to_hwc(xb), yb, key)
        if cfg["kernel_decay_rate"] > 0:
            for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
                trainer.orchestrator = eqx.tree_at(
                    path, trainer.orchestrator, path(trainer.orchestrator) * (1 - cfg["kernel_decay_rate"]))
        pD, head, key = measure(trainer.orchestrator, state_tmpl, ds, cfg, key, args.probe_epochs)
        hist.append({"epoch": ep, "probe_D": pD, "head_acc_D": head})
    return trainer.orchestrator, hist, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--flip", action="store_true", help="grad = backward(free) - backward(clamped)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 3; args.probe_epochs = 3; args.max_batches = 30

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    results = {"config": "best_channel_entropy (CHL functional test; NOT CHL-tuned)",
               "flip": args.flip, "seeds": args.seeds, "chl": {}, "baseline": {}}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        _, hc, _ = train_chl(cfg, ds, seed, args)
        results["chl"][str(seed)] = hc
        print(f"  CHL      final: probe_D={hc[-1]['probe_D']:.3f} head={hc[-1]['head_acc_D']:.3f} "
              f"(best probe_D {max(h['probe_D'] for h in hc):.3f})  ({cm.fmt(time.time()-t0)})")
        if not args.no_baseline:
            _, hb, _ = train_baseline(cfg, ds, seed, args)
            results["baseline"][str(seed)] = hb
            print(f"  baseline final: probe_D={hb[-1]['probe_D']:.3f} head={hb[-1]['head_acc_D']:.3f} "
                  f"({cm.fmt(time.time()-t0)})")

    def agg(d):
        if not d:
            return {}
        finals = [v[-1]["probe_D"] for v in d.values()]
        bests = [max(h["probe_D"] for h in v) for v in d.values()]
        heads = [v[-1]["head_acc_D"] for v in d.values()]
        return {"final_probe_D_mean": float(np.mean(finals)), "final_probe_D_std": float(np.std(finals)),
                "best_probe_D_mean": float(np.mean(bests)), "head_D_mean": float(np.mean(heads))}
    results["chl_agg"] = agg(results["chl"]); results["baseline_agg"] = agg(results["baseline"])

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "chl.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nCHL {results['chl_agg']}\nbaseline {results['baseline_agg']}")
    print(f"Saved {out_path}  (total {cm.fmt(time.time()-t0)})")
    print("Reference: DynamicalTrainer (model A) D-probe ~0.44-0.46; BPTT ceiling ~0.51.")


if __name__ == "__main__":
    main()
