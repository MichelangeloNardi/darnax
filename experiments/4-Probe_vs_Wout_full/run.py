"""experiments/4-Probe_vs_Wout_full/run.py

Full probe-vs-W_out comparison: 8 cells x 2 eval states = 16 numbers per config.

8 cells = readout {W_out (perceptron), probe (Adam)}
        x training {online, offline}
        x fit-state {C, D}

  W_out online  C : the model's own head (trained per-batch at C, the default)
  W_out online  D : per-batch at D, while W_in/J1 still train at C (two-rollout)
  W_out offline C : backbone frozen at end, fresh W_out trained (perceptron) at C
  W_out offline D : backbone frozen at end, fresh W_out trained (perceptron) at D
  probe online  C : one carried Adam probe, warm-started, updated each EPOCH on C reps
  probe online  D : same, on D reps
  probe offline C : fresh Adam probe trained at the end on final C reps
  probe offline D : fresh Adam probe trained at the end on final D reps

Each cell is evaluated on BOTH:
  eval_D : classic, valid (inference state, no label)
  eval_C : cheating (C built with the test labels) -- tells us if C is separable

Per-epoch curves are stored in the JSON (online cells over EPOCHS, offline cells
over their own training epochs); the plot uses the final-epoch value.

All model/optimizer/dataset/rollout machinery is in experiments/common.py. The
two genuinely new pieces here: the two-rollout online step (W_out on D) and the
online/offline probe + eval-on-C helpers.

Run on cluster (from repo root):
  ~/miniforge3/envs/darnax_hpc/bin/python experiments/4-Probe_vs_Wout_full/run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")  # TF32 corrupts sign decisions

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import common as cm
from darnax.utils.perceptron_rule import perceptron_rule_backward

# ── knobs ─────────────────────────────────────────────────────────────────────
SEEDS = [0, 42, 123]
EPOCHS = 8            # online backbone + readout training
ONLINE_PROBE_PASSES = 5   # Adam passes per epoch for the carried online probes
WOUT_OFFLINE_EPOCHS = 10  # perceptron passes for offline W_out
PROBE_OFFLINE_EPOCHS = 20 # Adam passes for offline probe
WOUT_THRESHOLD = 5.0      # matches PooledFlattenFC threshold in common.build_model

CONFIGS = {
    "best_channel_entropy": REPO / "replicate" / "best_channel_entropy_cfg.json",
    "matei_cgf":            REPO / "replicate" / "matei_cgf.json",
    "matei_W_out":          REPO / "replicate" / "matei_W_out_cfg.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CELLS = ["wout_online_C", "wout_online_D", "wout_offline_C", "wout_offline_D",
         "probe_online_C", "probe_online_D", "probe_offline_C", "probe_offline_D"]


# ── small eval helpers ────────────────────────────────────────────────────────

def idx(Ypm1):
    return np.argmax(Ypm1, axis=1)


def wout_acc(W, X, y_idx):
    """Accuracy of a linear W_out (256,10) on pooled reps X."""
    return float(((X @ np.asarray(W)).argmax(1) == y_idx).mean())


def new_probe():
    return nn.Linear(256, 10, bias=False).to(DEVICE)


def probe_update(probe, opt, X, y_idx, passes):
    """A few Adam passes (warm-started) on reps X."""
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float(),
                                      torch.from_numpy(y_idx).long()),
                        batch_size=256, shuffle=True)
    crit = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(passes):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(probe(xb), yb).backward()
            opt.step()


def probe_acc(probe, X, y_idx):
    probe.eval()
    with torch.no_grad():
        pred = probe(torch.from_numpy(X).float().to(DEVICE)).argmax(1).cpu().numpy()
    return float((pred == y_idx).mean())


# ── the two-rollout online step (W_out on C AND a separate W_out on D) ─────────

def make_online_step(cfg, opt, optd, roll_C, roll_D):
    """One training step. Updates the orchestrator (W_in+J1+W_out, all at C, exactly
    like the standard trainer) AND a separate W_out `woutD` trained at D.
    Returns the jitted step function."""
    @eqx.filter_jit
    def step(orch, woutD, state_tmpl, x, y, key, opt_state, optd_state):
        state = state_tmpl.init(x, y)
        sC, key = roll_C(orch, state, key)   # warmup->clamped->free = C
        sD, key = roll_D(orch, state, key)   # warmup->free          = D

        # backbone + the model's W_out: gradients at C (identical to common's trainer)
        gC = orch.backward(sC, rng=key)
        params = eqx.filter(orch, eqx.is_inexact_array)
        grads = eqx.filter(gC, eqx.is_inexact_array)
        upd, opt_state = opt.update(grads, opt_state, params=params)
        orch = eqx.apply_updates(orch, upd)

        # separate W_out trained at D (same perceptron rule)
        gD = woutD.backward(x=sD[1], y=y, y_hat=woutD(sD[1]))
        pw = eqx.filter(woutD, eqx.is_inexact_array)
        gw = eqx.filter(gD, eqx.is_inexact_array)
        uw, optd_state = optd.update(gw, optd_state, params=pw)
        woutD = eqx.apply_updates(woutD, uw)
        return orch, woutD, opt_state, optd_state, key
    return step


# ── offline W_out: perceptron rule on FIXED pooled reps (faithful to the model) ─

def offline_wout(cfg, X, Ypm1, W0, evals, epochs, key):
    """Train a linear W_out with the perceptron rule on fixed reps X (pm1 labels
    Ypm1). evals = {"eval_D": (Xte, idx), "eval_C": (Xte, idx)}. Returns curves."""
    mom = cfg["momentum"]
    opt = optax.sgd(cfg["lr_wout"], momentum=mom) if mom > 0 else optax.sgd(cfg["lr_wout"])
    W = jnp.asarray(W0)
    opt_state = opt.init(W)
    th = jnp.asarray(WOUT_THRESHOLD)

    @eqx.filter_jit
    def pstep(W, opt_state, Xb, Yb):
        grad = perceptron_rule_backward(Xb, Yb, Xb @ W, th)
        upd, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(W, upd), opt_state

    Xj, Yj = jnp.asarray(X), jnp.asarray(Ypm1)
    N, bs = X.shape[0], 32
    curves = {k: [] for k in evals}
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            W, opt_state = pstep(W, opt_state, Xj[b], Yj[b])
        for name, (Xe, ye) in evals.items():
            curves[name].append(wout_acc(W, Xe, ye))
    return curves


def offline_probe(X, y_idx, evals, epochs):
    """Fresh Adam probe trained on fixed reps X. evals as above. Returns curves."""
    probe = new_probe()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=cm.PROBE_WD)
    curves = {k: [] for k in evals}
    for _ in range(epochs):
        probe_update(probe, opt, X, y_idx, 1)
        for name, (Xe, ye) in evals.items():
            curves[name].append(probe_acc(probe, Xe, ye))
    return curves


# ── one (config, seed) ────────────────────────────────────────────────────────

def run_seed(cfg, seed, ds, t0):
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = cm.build_model(cfg, mk)
    key, wk = jax.random.split(key)
    woutD = cm.reinit_wout(orch, wk).lmap[2][1]   # standalone W_out for the D path

    opt, opt_state = cm.make_optimizer(orch, cfg)           # W_in+J1+W_out all train
    mom = cfg["momentum"]
    optd = optax.sgd(cfg["lr_wout"], momentum=mom) if mom > 0 else optax.sgd(cfg["lr_wout"])
    optd_state = optd.init(eqx.filter(woutD, eqx.is_inexact_array))

    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll_C_p, roll_D_p = cm.make_rollout(warmup, clamped, free), cm.make_rollout(warmup, 0, free)
    roll_C, roll_D = eqx.filter_jit(roll_C_p), eqx.filter_jit(roll_D_p)
    step = make_online_step(cfg, opt, optd, roll_C_p, roll_D_p)

    # carried online probes (warm-started across epochs)
    probeC, probeD = new_probe(), new_probe()
    optPC = torch.optim.Adam(probeC.parameters(), lr=1e-3, weight_decay=cm.PROBE_WD)
    optPD = torch.optim.Adam(probeD.parameters(), lr=1e-3, weight_decay=cm.PROBE_WD)

    curves = {c: {"eval_D": [], "eval_C": []} for c in CELLS}
    Xtr_C = Xtr_D = Ytr = Xte_C = Xte_D = Yte = None

    for epoch in range(1, EPOCHS + 1):
        for xb, yb in ds:
            orch, woutD, opt_state, optd_state, key = step(
                orch, woutD, state, cm.to_hwc(xb), yb, key, opt_state, optd_state)
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            orch = eqx.tree_at(path, orch, path(orch) * (1.0 - cfg["kernel_decay_rate"]))

        # reps at C and D (train for probe training, test for evaluation).
        # NOTE: the train set reshuffles every pass, so each collect_reps call
        # returns its OWN matching labels — Ytr_C and Ytr_D are different orderings
        # and must not be mixed. (The test set iterates in order, so Yte matches
        # both Xte_C and Xte_D.)
        Xtr_C, Ytr_C, key = cm.collect_reps(orch, state, ds, roll_C, key)
        Xtr_D, Ytr_D, key = cm.collect_reps(orch, state, ds, roll_D, key)
        Xte_C, Yte,   key = cm.collect_reps(orch, state, ds.iter_test(), roll_C, key)
        Xte_D, _,     key = cm.collect_reps(orch, state, ds.iter_test(), roll_D, key)
        ytrC_i, ytrD_i, yte_i = idx(Ytr_C), idx(Ytr_D), idx(Yte)

        # online probes: warm-start update on this epoch's reps (matched labels)
        probe_update(probeC, optPC, Xtr_C, ytrC_i, ONLINE_PROBE_PASSES)
        probe_update(probeD, optPD, Xtr_D, ytrD_i, ONLINE_PROBE_PASSES)

        Wc, Wd = orch.lmap[2][1].W, woutD.W
        for cell, (kind, obj) in {
            "wout_online_C":  ("wout", Wc),
            "wout_online_D":  ("wout", Wd),
            "probe_online_C": ("probe", probeC),
            "probe_online_D": ("probe", probeD),
        }.items():
            if kind == "wout":
                curves[cell]["eval_D"].append(wout_acc(obj, Xte_D, yte_i))
                curves[cell]["eval_C"].append(wout_acc(obj, Xte_C, yte_i))
            else:
                curves[cell]["eval_D"].append(probe_acc(obj, Xte_D, yte_i))
                curves[cell]["eval_C"].append(probe_acc(obj, Xte_C, yte_i))

        print(f"  seed={seed} ep={epoch:2d}/{EPOCHS}  "
              f"woutC[D]={curves['wout_online_C']['eval_D'][-1]:.3f} "
              f"woutD[D]={curves['wout_online_D']['eval_D'][-1]:.3f} "
              f"probeC[D]={curves['probe_online_C']['eval_D'][-1]:.3f} "
              f"probeD[D]={curves['probe_online_D']['eval_D'][-1]:.3f}  "
              f"[{cm.fmt(time.time()-t0)}]", flush=True)

    # ── offline cells on the FINAL frozen backbone (reuse last-epoch reps) ──
    yte_i = idx(Yte)
    W0 = cm.reinit_wout(orch, jax.random.PRNGKey(seed + 1)).lmap[2][1].W
    evals = {"eval_D": (Xte_D, yte_i), "eval_C": (Xte_C, yte_i)}  # eval states are fixed
    for cell, Xtr_S, Ytr_S in [("wout_offline_C", Xtr_C, Ytr_C),
                               ("wout_offline_D", Xtr_D, Ytr_D)]:
        curves[cell] = offline_wout(cfg, Xtr_S, Ytr_S, W0, evals, WOUT_OFFLINE_EPOCHS, key)
    for cell, Xtr_S, Ytr_S in [("probe_offline_C", Xtr_C, Ytr_C),
                               ("probe_offline_D", Xtr_D, Ytr_D)]:
        curves[cell] = offline_probe(Xtr_S, idx(Ytr_S), evals, PROBE_OFFLINE_EPOCHS)

    print(f"  seed={seed} offline:  "
          f"woutC[D]={curves['wout_offline_C']['eval_D'][-1]:.3f} "
          f"woutD[D]={curves['wout_offline_D']['eval_D'][-1]:.3f} "
          f"probeC[D]={curves['probe_offline_C']['eval_D'][-1]:.3f} "
          f"probeD[D]={curves['probe_offline_D']['eval_D'][-1]:.3f}", flush=True)
    return curves


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    results = []
    for name, path in CONFIGS.items():
        cfg = cm.load_cfg(path)
        print(f"\n{'='*64}\nCONFIG: {name}  (strength_back={cfg['strength_back']:.3f}, "
              f"clamped={cfg['clamped_n_iter']}, free={cfg['free_n_iter']})\n{'='*64}", flush=True)
        per_seed = []
        for seed in SEEDS:
            curves = run_seed(cfg, seed, ds, t0)
            per_seed.append({"seed": seed, "curves": curves})
        results.append({"config": name, "per_seed": per_seed})

    # ── summary: final-epoch mean over seeds, both eval states ──
    print(f"\n{'='*78}\nSUMMARY (final epoch, mean over {len(SEEDS)} seeds)\n{'='*78}", flush=True)
    print(f"{'config':<22}{'cell':<18}{'eval_D':>9}{'eval_C':>9}", flush=True)
    summary = {}
    for r in results:
        summary[r["config"]] = {}
        for cell in CELLS:
            d = np.mean([s["curves"][cell]["eval_D"][-1] for s in r["per_seed"]])
            c = np.mean([s["curves"][cell]["eval_C"][-1] for s in r["per_seed"]])
            summary[r["config"]][cell] = {"eval_D": float(d), "eval_C": float(c)}
            print(f"{r['config']:<22}{cell:<18}{d:>9.4f}{c:>9.4f}", flush=True)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = {"seeds": SEEDS, "epochs": EPOCHS, "cells": CELLS,
           "summary": summary, "results": results}
    (out_dir / "probe_vs_wout.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_dir / 'probe_vs_wout.json'}  (total {cm.fmt(time.time()-t0)})", flush=True)


if __name__ == "__main__":
    main()
