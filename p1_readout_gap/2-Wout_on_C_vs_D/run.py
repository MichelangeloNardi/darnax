"""experiments/2-Wout_on_C_vs_D/run.py

Quick test: for each config, does training W_out on D beat training it on C?

For every config + seed:
  1. Train the full network normally (W_in + J1 + W_out, online). The online
     W_out is trained on C -> this gives `head_online` (the C number, as shipped).
  2. Freeze W_in + J1. Re-init W_out and train it (same perceptron rule) on:
        wout_C : clamped_n = cfg  (rollout warmup->clamped->free = C)
        wout_D : clamped_n = 0    (rollout warmup->free          = D)
     Both are evaluated on D (eval never has the label).

All the model/optimizer/training machinery lives in experiments/common.py; this
script only loops over configs and flips clamped_n between C and D.

Run on cluster (from repo root):
  ~/miniforge3/envs/darnax_hpc/bin/python experiments/2-Wout_on_C_vs_D/run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")  # TF32 corrupts sign decisions

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))

import common as cm

# ── knobs ─────────────────────────────────────────────────────────────────────
SEEDS = [0, 42]
EPOCHS = 8          # backbone (W_in+J1+W_out) training epochs
WOUT_EPOCHS = 8     # offline W_out re-training epochs

CONFIGS = {
    "best_channel_entropy": REPO / "replicate" / "best_channel_entropy_cfg.json",
    "matei_cgf":            REPO / "replicate" / "matei_cgf.json",
    "matei_W_out":          REPO / "replicate" / "matei_W_out_cfg.json",
}


def fit_wout(orch_trained, state, ds, cfg, clamped_n, key):
    """Freeze W_in/J1, re-init W_out, train it on C (clamped_n=cfg) or D
    (clamped_n=0). Returns test accuracy (eval = D)."""
    key, wk = jax.random.split(key)
    orch = cm.reinit_wout(orch_trained, wk)
    opt, opt_state = cm.make_optimizer(orch, cfg, win=False, j1=False, wout=True)
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg, clamped_n=clamped_n)
    for _ in range(WOUT_EPOCHS):
        trainer, key = cm.train_epoch(trainer, ds, key, decay_rate=0.0)  # backbone frozen
    acc, key = cm.eval_head(trainer, ds, key)
    return acc, key


def run_config(name, cfg_path, ds, t0):
    cfg = cm.load_cfg(cfg_path)
    print(f"\n{'='*60}\nCONFIG: {name}  "
          f"(strength_back={cfg['strength_back']:.3f}, clamped={cfg['clamped_n_iter']}, "
          f"free={cfg['free_n_iter']}, warmup={cfg.get('warmup_n_iter', 1)})\n{'='*60}", flush=True)
    rows = []
    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        key, mk = jax.random.split(key)
        state, orch = cm.build_model(cfg, mk)
        opt, opt_state = cm.make_optimizer(orch, cfg)            # all three train
        trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)  # clamped=cfg -> W_out on C

        for _ in range(EPOCHS):
            trainer, key = cm.train_epoch(trainer, ds, key, decay_rate=cfg["kernel_decay_rate"])
        head_online, key = cm.eval_head(trainer, ds, key)        # online W_out (trained on C)

        wout_C, key = fit_wout(trainer.orchestrator, trainer.state, ds, cfg,
                               cfg["clamped_n_iter"], key)        # offline W_out on C
        wout_D, key = fit_wout(trainer.orchestrator, trainer.state, ds, cfg, 0, key)  # on D

        rows.append({"seed": seed, "head_online": head_online,
                     "wout_C": wout_C, "wout_D": wout_D})
        print(f"  seed={seed}  head_online(C)={head_online:.4f}  "
              f"wout_C={wout_C:.4f}  wout_D={wout_D:.4f}  "
              f"gain(D-C)={wout_D - wout_C:+.4f}  [elapsed={cm.fmt(time.time()-t0)}]", flush=True)
    return {"config": name, "strength_back": cfg["strength_back"], "per_seed": rows}


def main():
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    results = [run_config(name, path, ds, t0) for name, path in CONFIGS.items()]

    # ── summary table ──
    print(f"\n{'='*60}\nSUMMARY (mean over {len(SEEDS)} seeds, eval on D)\n{'='*60}", flush=True)
    print(f"{'config':<22}{'wout_C':>9}{'wout_D':>9}{'gain':>9}", flush=True)
    for r in results:
        c = np.mean([s["wout_C"] for s in r["per_seed"]])
        d = np.mean([s["wout_D"] for s in r["per_seed"]])
        print(f"{r['config']:<22}{c:>9.4f}{d:>9.4f}{d - c:>+9.4f}", flush=True)
        r["wout_C_mean"], r["wout_D_mean"] = float(c), float(d)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = {"seeds": SEEDS, "epochs": EPOCHS, "wout_epochs": WOUT_EPOCHS, "results": results}
    (out_dir / "wout_c_vs_d.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_dir / 'wout_c_vs_d.json'}  (total {cm.fmt(time.time()-t0)})", flush=True)


if __name__ == "__main__":
    main()
