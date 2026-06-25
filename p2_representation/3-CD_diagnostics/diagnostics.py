"""C/D diagnostics on the three trained models (A local-rule, B BPTT CE_D,
C BPTT CE_D + alpha*align). Loads the serialized orchestrators from train_models.py
and computes 9 diagnostics offline (no training). Results per seed -> mean +- std.

States (per the ABCD convention):
  D = warmup -> free                 (inference; forward messages only)
  C = warmup -> clamped -> free      (clamped phase injects the label via W_back)
The orchestrator stores the pre-activation *field* (state.fields[1]) of the last
step, so margins / field-projections are read directly.

Diagnostics:
  1 C probe acc          Adam linear probe on pooled C
  2 D probe acc          Adam linear probe on pooled D
  3 C-D flip rate        mean[sign(C) != sign(D)]
  4 overlap(C,D)         mean(C * D)            (both +-1)
  5 importance-vs-flip   point-biserial corr(importance, flip), for BOTH
                         importances: readout contribution |W_out[feat,y]|/P^2
                         and local field magnitude |field_C|
  6 margin flipped/stable  margin = C * field_C, mean for flipped vs stable spins
  7 field.C on flips     mean(C * field_C) over flipped spins (vs stable)
  8 random-flip control  mean importance of really-flipped spins vs random spins
                         (matched per-example count) + corr under random flips
  9 C stability          start from C, run `free` more forward steps -> C';
                         overlap(C, C') and flip rate (= does C survive free dyn.)

Run (cluster): XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/3-CD_diagnostics/diagnostics.py
Smoke:  python p2_representation/3-CD_diagnostics/diagnostics.py --smoke
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
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
MODELS_DIR = HERE / "models"
H, W, C, P = cm.H, cm.W, cm.C, cm.POOL


# ── rollers (deterministic: conv forward ignores rng, sign is deterministic) ──

def make_roller(warmup, clamped, free):
    roll = cm.make_rollout(warmup, clamped, free)

    @eqx.filter_jit
    def f(orch, state, key):
        state, _ = roll(orch, state, key)
        return state
    return f


def make_free_cont(free):
    @eqx.filter_jit
    def f(orch, state, key):
        for _ in range(free):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        return state
    return f


def load_model(seed, name, cfg):
    _, template = cm.build_model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(MODELS_DIR / f"{name}_seed{seed}.eqx", template)


# ── feature-index map: spin (h,w,c) -> pooled W_out row ───────────────────────
# pool_j1 flattens (H//P, W//P, C) in C-order, so feat = ((h//P)*(W//P)+(w//P))*C + c
def feat_index_map():
    hb = (np.arange(H) // P)[:, None, None]
    wb = (np.arange(W) // P)[None, :, None]
    cc = np.arange(C)[None, None, :]
    return ((hb * (W // P) + wb) * C + cc).astype(np.int64)  # (H,W,C)


FEAT_IDX = feat_index_map()


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


# ── pooled reps for the probe (diag 1,2) ──────────────────────────────────────

def collect_pooled(orch, state_tmpl, it, roller, key, max_b):
    X, Y = [], []
    for i, (xb, yb) in enumerate(it):
        if max_b is not None and i >= max_b:
            break
        s = roller(orch, state_tmpl.init(cm.to_hwc(xb), yb), key)
        X.append(np.asarray(cm.pool_j1(np.asarray(s[1])))); Y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(Y)


# ── one model/seed ────────────────────────────────────────────────────────────

def diagnose(orch, cfg, ds, state_tmpl, roll_D, roll_C, free_cont, args):
    key = jax.random.PRNGKey(0)
    out = {}

    # --- diag 1,2: probe acc on C and D ---
    for tag, roller in [("D", roll_D), ("C", roll_C)]:
        Xtr, Ytr = collect_pooled(orch, state_tmpl, ds, roller, key, args.probe_train_batches)
        Xte, Yte = collect_pooled(orch, state_tmpl, ds.iter_test(), roller, key, args.max_batches)
        acc = bc.offline_probe(Xtr, np.argmax(Ytr, 1), Xte, np.argmax(Yte, 1), args.probe_epochs)
        out[f"probe_{tag}"] = max(acc)

    # --- spin-level subset (batched -> host numpy) ---
    Cs, Cf, Ds, Cps, Ys = [], [], [], [], []
    nb = 0
    for xb, yb in ds.iter_test():
        if (args.max_batches is not None and nb >= args.max_batches) or \
           (nb * xb.shape[0] >= args.diag_examples):
            break
        x, y = cm.to_hwc(xb), yb
        sC = roll_C(orch, state_tmpl.init(x, y), key)
        sD = roll_D(orch, state_tmpl.init(x, y), key)
        sCp = free_cont(orch, sC, key)
        Cs.append(np.asarray(sC[1])); Cf.append(np.asarray(sC.fields[1]))
        Ds.append(np.asarray(sD[1])); Cps.append(np.asarray(sCp[1]))
        Ys.append(np.argmax(np.asarray(yb), 1)); nb += 1
    Cs = np.concatenate(Cs); Cf = np.concatenate(Cf)
    Ds = np.concatenate(Ds); Cps = np.concatenate(Cps); Ys = np.concatenate(Ys)
    N = Cs.shape[0]

    flip = (np.sign(Cs) != np.sign(Ds))                 # (N,H,W,C) bool
    out["flip_rate"] = float(flip.mean())                                  # diag 3
    out["overlap_CD"] = float((Cs * Ds).mean())                            # diag 4

    margin_C = Cs * Cf                                  # diag 6/7 base
    fl, st = flip, ~flip
    out["margin_flipped"] = float(margin_C[fl].mean()) if fl.any() else 0.0
    out["margin_stable"] = float(margin_C[st].mean()) if st.any() else 0.0  # diag 6
    out["fieldC_on_flipped"] = float((Cf * Cs)[fl].mean()) if fl.any() else 0.0  # diag 7
    out["fieldC_on_stable"] = float((Cf * Cs)[st].mean()) if st.any() else 0.0

    # importances (diag 5)
    absW = np.abs(np.asarray(orch.lmap[2][1].W))        # (256,10)
    WF = absW[FEAT_IDX]                                 # (H,W,C,10)
    imp_read = np.moveaxis(WF[..., Ys], -1, 0) / (P * P)   # (N,H,W,C)
    imp_field = np.abs(Cf)                              # (N,H,W,C)
    out["corr_readout_flip"] = pearson(imp_read, flip.astype(np.float64))
    out["corr_field_flip"] = pearson(imp_field, flip.astype(np.float64))

    # diag 8: random-flip control (match per-example flip count)
    rng = np.random.default_rng(0)
    kper = flip.reshape(N, -1).sum(1)                   # flips per example
    M = H * W * C
    rand_flip = np.zeros((N, M), bool)
    for i in range(N):
        if kper[i] > 0:
            rand_flip[i, rng.choice(M, size=int(kper[i]), replace=False)] = True
    rand_flip = rand_flip.reshape(N, H, W, C)
    out["mean_imp_read_flipped"] = float(imp_read[fl].mean()) if fl.any() else 0.0
    out["mean_imp_read_random"] = float(imp_read[rand_flip].mean()) if rand_flip.any() else 0.0
    out["mean_imp_field_flipped"] = float(imp_field[fl].mean()) if fl.any() else 0.0
    out["mean_imp_field_random"] = float(imp_field[rand_flip].mean()) if rand_flip.any() else 0.0
    out["corr_readout_flip_random"] = pearson(imp_read, rand_flip.astype(np.float64))
    out["corr_field_flip_random"] = pearson(imp_field, rand_flip.astype(np.float64))

    # diag 9: C stability under free dynamics
    out["C_free_overlap"] = float((Cs * Cps).mean())
    out["C_free_flip_rate"] = float((np.sign(Cs) != np.sign(Cps)).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--probe-train-batches", type=int, default=800)  # ~25k imgs
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--diag-examples", type=int, default=2000)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.probe_train_batches = 4; args.probe_epochs = 2
        args.diag_examples = 128; args.max_batches = 4

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    state_tmpl, _ = cm.build_model(cfg, jax.random.PRNGKey(0))
    warmup, clamped, free = cfg.get("warmup_n_iter", 1), cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll_D = make_roller(warmup, 0, free)
    roll_C = make_roller(warmup, clamped, free)
    free_cont = make_free_cont(free)

    t0 = time.time()
    results = {"config": "best_channel_entropy", "seeds": args.seeds, "models": {}}
    for name in ["A", "B", "C"]:
        per_seed = []
        for seed in args.seeds:
            orch = load_model(seed, name, cfg)
            d = diagnose(orch, cfg, ds, state_tmpl, roll_D, roll_C, free_cont, args)
            per_seed.append(d)
            print(f"  [{name} seed {seed}] probeD={d['probe_D']:.3f} probeC={d['probe_C']:.3f} "
                  f"flip={d['flip_rate']:.3f} ov={d['overlap_CD']:.3f} "
                  f"corrR={d['corr_readout_flip']:.3f} Cfree_ov={d['C_free_overlap']:.3f} "
                  f"({cm.fmt(time.time()-t0)})")
        keys = per_seed[0].keys()
        results["models"][name] = {
            "per_seed": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in keys},
            "std": {k: float(np.std([p[k] for p in per_seed])) for k in keys},
        }

    out_dir = HERE / "results"; out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("smoke.json" if args.smoke else "diagnostics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}  (total {cm.fmt(time.time()-t0)})")


if __name__ == "__main__":
    main()
