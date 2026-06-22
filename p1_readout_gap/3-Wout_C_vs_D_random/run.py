"""experiments/3-Wout_C_vs_D_random/run.py

Same W_out-on-C-vs-D test as experiment 2, but comparing two backbones:

  trained : W_in + J1 trained normally, then W_out re-fit on C vs D   (= exp 2)
  random  : W_in + J1 FROZEN at random init, only W_out is fit on C vs D

For each config × mode × seed: get the backbone (train it, or leave it random),
then freeze W_in/J1, re-init W_out and train it (perceptron rule) on:
    wout_C : clamped_n = cfg  (rollout warmup->clamped->free = C)
    wout_D : clamped_n = 0    (rollout warmup->free          = D)
both evaluated on D. The only difference from exp 2 is the `random` mode, which
skips backbone training. All machinery is in experiments/common.py.

Run on cluster (from repo root):
  ~/miniforge3/envs/darnax_hpc/bin/python experiments/3-Wout_C_vs_D_random/run.py
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
SEEDS = [0, 42, 123]
EPOCHS = 8          # backbone training epochs (trained mode only)
WOUT_EPOCHS = 8     # W_out re-training epochs (both modes)
MODES = ["trained", "random"]

CONFIGS = {
    "best_channel_entropy": REPO / "replicate" / "best_channel_entropy_cfg.json",
    "matei_cgf":            REPO / "replicate" / "matei_cgf.json",
    "matei_W_out":          REPO / "replicate" / "matei_W_out_cfg.json",
}


def get_backbone(cfg, seed, mode, ds, key):
    """Return (orchestrator, state) for the chosen backbone.

    trained : train W_in+J1+W_out normally for EPOCHS, return the trained model.
    random  : build the model and return it untouched (W_in/J1 random, frozen).
    """
    key, mk = jax.random.split(key)
    state, orch = cm.build_model(cfg, mk)
    if mode == "random":
        return orch, state, key
    opt, opt_state = cm.make_optimizer(orch, cfg)              # all three train
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)  # clamped=cfg -> W_out on C
    for _ in range(EPOCHS):
        trainer, key = cm.train_epoch(trainer, ds, key, decay_rate=cfg["kernel_decay_rate"])
    return trainer.orchestrator, trainer.state, key


def run_cell(name, cfg, mode, ds, t0):
    rows = []
    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        orch, state, key = get_backbone(cfg, seed, mode, ds, key)
        wout_C, key = cm.fit_wout(orch, state, ds, cfg, cfg["clamped_n_iter"], key, WOUT_EPOCHS)
        wout_D, key = cm.fit_wout(orch, state, ds, cfg, 0, key, WOUT_EPOCHS)
        rows.append({"seed": seed, "wout_C": wout_C, "wout_D": wout_D})
        print(f"  [{name:<20} {mode:<7}] seed={seed}  wout_C={wout_C:.4f}  "
              f"wout_D={wout_D:.4f}  gain(D-C)={wout_D - wout_C:+.4f}  "
              f"[elapsed={cm.fmt(time.time()-t0)}]", flush=True)
    return rows


def main():
    ds = cm.get_dataset(batch_size=32)
    t0 = time.time()
    results = []
    for name, path in CONFIGS.items():
        cfg = cm.load_cfg(path)
        print(f"\n{'='*64}\nCONFIG: {name}  (strength_back={cfg['strength_back']:.3f}, "
              f"clamped={cfg['clamped_n_iter']}, free={cfg['free_n_iter']})\n{'='*64}", flush=True)
        for mode in MODES:
            rows = run_cell(name, cfg, mode, ds, t0)
            results.append({"config": name, "mode": mode,
                            "strength_back": cfg["strength_back"], "per_seed": rows})

    # ── summary table ──
    print(f"\n{'='*64}\nSUMMARY (mean over {len(SEEDS)} seeds, eval on D)\n{'='*64}", flush=True)
    print(f"{'config':<22}{'mode':<9}{'wout_C':>9}{'wout_D':>9}{'gain':>9}", flush=True)
    for r in results:
        c = float(np.mean([s["wout_C"] for s in r["per_seed"]]))
        d = float(np.mean([s["wout_D"] for s in r["per_seed"]]))
        r["wout_C_mean"], r["wout_D_mean"] = c, d
        print(f"{r['config']:<22}{r['mode']:<9}{c:>9.4f}{d:>9.4f}{d - c:>+9.4f}", flush=True)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = {"seeds": SEEDS, "epochs": EPOCHS, "wout_epochs": WOUT_EPOCHS,
           "modes": MODES, "results": results}
    (out_dir / "wout_c_vs_d_random.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_dir / 'wout_c_vs_d_random.json'}  (total {cm.fmt(time.time()-t0)})", flush=True)


if __name__ == "__main__":
    main()
