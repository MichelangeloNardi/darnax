"""experiments/1-Probe_Wout_on_C_vs_D/probe_wout_c_vs_d.py

Does fitting the readouts (linear probe + W_out) on the CLAMPED-consolidated
representation C beat fitting them on the inference representation D?

ABCD recap (matches replicate_experiments/.../diagnostics.py::_run_abcd):
  A = after warmup            (forward-only, input W_in only, no label)
  B = after clamped (A -> B)  ("all" messages; W_back injects the label y)
  C = after free   (B -> C)   (forward-only) -- the state the ONLINE rule trains on
  D = after free   (A -> D)   (forward-only, skip clamped) == inference / eval_step

The online W_out is already trained on C; the linear probe is currently trained
AND tested on D (via eval_step). Here we fit the readouts on C instead.

HARD CONSTRAINT: C needs the label (clamped phase reads y through W_back), so at
test time -- where labels are unavailable -- representations can only be D.
The valid comparison is therefore fit-on-C(train) -> evaluate-on-D(test).

Per seed, per epoch we report:
  head            : online W_out (trained on C), eval on D            [reference]
  probe_D         : Adam probe fit on D(train), eval on D(test)       [baseline]
  probe_C         : Adam probe fit on C(train), eval on D(test)       [hypothesis]
  probe_C_leaky   : Adam probe fit on C(train), eval on C(test)       [ceiling; uses
                    test labels to FORM the representation -> NOT a valid accuracy,
                    diagnostic only: shows how separable C is at all]

At the final epoch we also run an offline W_out comparison (Win/J frozen, W_out
re-initialised and trained with the perceptron rule):
  wout_C : W_out trained on C (clamped_n = cfg), eval on D(test)
  wout_D : W_out trained on D (clamped_n = 0),   eval on D(test)

Run on cluster (from repo root):
  ~/miniforge3/envs/darnax_hpc/bin/python experiments/1-Probe_Wout_on_C_vs_D/probe_wout_c_vs_d.py
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
import jax.tree_util as jtu
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

# ── experiment knobs ──────────────────────────────────────────────────────────
SEEDS = [0, 42, 123]
EPOCHS = 10
WOUT_OFFLINE_EPOCHS = 10

# ── architecture constants ────────────────────────────────────────────────────
C, KSIZE = 16, 5
H, W, POOL = 32, 32, 8
PROBE_EPOCHS = 20
PROBE_WD = 1.433e-4


# ── model / optimizer ─────────────────────────────────────────────────────────

def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0, key=keys[0],
                      padding_mode="constant", lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                               strength=1.0, threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=layer_map)


def make_optimizer(orchestrator, cfg, win=True, j1=True, wout=True):
    """Multi-transform SGD. Inactive groups are set_to_zero (frozen)."""
    mom = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, jx), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(lambda m, r=i, c=jx: m.lmap[r][c], labels,
                             replace=like(params.lmap[i][jx], lbl))

    opt = optax.multi_transform({
        "default": optax.set_to_zero(),
        "win":  sgd(-cfg["lr_win"]) if win else optax.set_to_zero(),
        "j1":   sgd(-cfg["lr_j"]) if j1 else optax.set_to_zero(),
        "wout": sgd(cfg["lr_wout"]) if wout else optax.set_to_zero(),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def reinit_wout(orch, key):
    new_wout = PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                              strength=1.0, threshold=5.0, key=key, lr=1.0, weight_decay=0.0)
    return eqx.tree_at(lambda o: o.lmap[2][1], orch, new_wout)


# ── data helpers ──────────────────────────────────────────────────────────────

def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


# ── representation rollouts (jitted) ──────────────────────────────────────────

def make_rollout(warmup_n, clamped_n, free_n):
    """Return a jitted fn rolling a state through warmup->clamped->free.

    clamped_n == 0 gives D (warmup->free, == eval_step). clamped_n > 0 gives C.
    warmup/clamped/free counts are Python ints -> static, loops unroll under jit.
    """
    @eqx.filter_jit
    def rollout(orch, state, key):
        for _ in range(warmup_n):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        for _ in range(clamped_n):
            state, key = orch.step(state, rng=key, filter_messages="all")
        for _ in range(free_n):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        return state, key

    return rollout


def collect_reps(rollout, orch, state_template, iterator, key):
    """Roll every batch to its fixed point; return pooled J1 reps + label indices."""
    reps, lbls = [], []
    for xb, yb in iterator:
        state = state_template.init(to_hwc(xb), yb)
        state, key = rollout(orch, state, key)
        reps.append(pool_j1(np.asarray(state[1])))
        lbls.append(np.argmax(np.asarray(yb), axis=-1))
    return np.concatenate(reps), np.concatenate(lbls), key


# ── ABCD attractor diagnostics ────────────────────────────────────────────────

ABCD_LABELS = ["A", "B", "C", "D"]


def make_abcd(warmup_n, clamped_n, free_n):
    """Jitted fn returning the J1 buffers (B,H,W,C) at A, B, C, D for one batch.

    A = warmup, B = A→clamped, C = B→free, D = A→free (skip clamped). Faithful to
    replicate_experiments/.../diagnostics.py::_run_abcd (D uses A's rng path; rng
    is inert here since the conv/wback modules ignore it).
    """
    @eqx.filter_jit
    def fn(orch, state, key):
        sA = state
        for _ in range(warmup_n):
            sA, key = orch.step(sA, rng=key, filter_messages="forward")
        key_a = key
        sB = sA
        for _ in range(clamped_n):
            sB, key = orch.step(sB, rng=key, filter_messages="all")
        sC = sB
        for _ in range(free_n):
            sC, key = orch.step(sC, rng=key, filter_messages="forward")
        sD, kd = sA, key_a
        for _ in range(free_n):
            sD, kd = orch.step(sD, rng=kd, filter_messages="forward")
        return sA[1], sB[1], sC[1], sD[1]

    return fn


def _cos_per_image(a, b):
    na = np.linalg.norm(a, axis=1) + 1e-8
    nb = np.linalg.norm(b, axis=1) + 1e-8
    return (a * b).sum(axis=1) / (na * nb)


def abcd_grid(abcd_fn, orch, state_template, x, y, key):
    """4×4 mean per-image cosine-similarity matrix between A, B, C, D."""
    state = state_template.init(x, y)
    bufs = abcd_fn(orch, state, key)
    N = x.shape[0]
    flat = [np.asarray(b).reshape(N, -1).astype(np.float32) for b in bufs]
    M = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            M[i, j] = float(_cos_per_image(flat[i], flat[j]).mean())
    return M


# ── linear probe (Adam) ───────────────────────────────────────────────────────

def fit_adam_probe(X_tr, y_tr, X_te, y_te):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = torch.from_numpy(X_tr).float()
    y_tr = torch.from_numpy(y_tr).long()
    X_te = torch.from_numpy(X_te).float().to(device)
    y_te = torch.from_numpy(y_te).long().to(device)

    probe = nn.Linear(256, 10, bias=False).to(device)
    opt_p = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb_t, yb_t in loader:
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            opt_p.zero_grad()
            crit(probe(xb_t), yb_t).backward()
            opt_p.step()
        probe.eval()
        with torch.no_grad():
            te = (probe(X_te).argmax(1) == y_te).float().mean().item()
        best = max(best, te)
    return best


# ── training / eval ───────────────────────────────────────────────────────────

def train_epoch(trainer, ds, cfg, key, decay=True):
    d = cfg["kernel_decay_rate"]
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)
    if decay:
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - d)
            )
    return trainer, key


def eval_head(trainer, ds, key):
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def fmt(s):
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s"


# ── offline W_out comparison (final epoch) ────────────────────────────────────

def offline_wout(orch_trained, state_template, ds, cfg, clamped_n, key, tag):
    """Freeze Win/J, re-init W_out, train it with the perceptron rule, eval on D.

    clamped_n = cfg gives W_out-on-C; clamped_n = 0 gives W_out-on-D.
    """
    key, wk = jax.random.split(key)
    orch = reinit_wout(orch_trained, wk)
    opt, opt_state = make_optimizer(orch, cfg, win=False, j1=False, wout=True)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state_template,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1, train_clamped_n_iter=clamped_n,
        train_free_n_iter=cfg["free_n_iter"], eval_n_iter=5,
    )
    for _ in range(WOUT_OFFLINE_EPOCHS):
        trainer, key = train_epoch(trainer, ds, cfg, key, decay=False)
    head, key = eval_head(trainer, ds, key)
    print(f"    [offline W_out on {tag}] head(D)={head:.4f}", flush=True)
    return head, key


# ── single seed ───────────────────────────────────────────────────────────────

def run_one_seed(seed, cfg, ds, t0_all):
    print(f"\n{'='*54}\nSeed {seed}\n{'='*54}", flush=True)
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(orch, cfg, win=True, j1=True, wout=True)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1, train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"], eval_n_iter=5,
    )

    warmup_n, clamped_n, free_n = 1, cfg["clamped_n_iter"], cfg["free_n_iter"]
    roll_C = make_rollout(warmup_n, clamped_n, free_n)  # C: warmup -> clamped -> free
    roll_D = make_rollout(warmup_n, 0, free_n)          # D: warmup -> free (== eval)
    abcd_fn = make_abcd(warmup_n, clamped_n, free_n)    # A, B, C, D buffers

    rep_rng = jax.random.PRNGKey(7)  # fixed: reps are deterministic anyway
    diag_rng = jax.random.PRNGKey(42)

    # Fixed diagnostic batch (first test batch), same across all epochs for this seed
    diag_x, diag_y = next(iter(ds.iter_test()))
    diag_x = to_hwc(diag_x)

    rec = {"head": [], "probe_D": [], "probe_C": [], "probe_C_leaky": [],
           "cd_sim": [], "abcd_grids": []}
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        trainer, key = train_epoch(trainer, ds, cfg, key, decay=True)

        head, key = eval_head(trainer, ds, key)

        o = trainer.orchestrator
        Xc_tr, yc_tr, rep_rng = collect_reps(roll_C, o, trainer.state, ds, rep_rng)
        Xd_tr, yd_tr, rep_rng = collect_reps(roll_D, o, trainer.state, ds, rep_rng)
        Xc_te, yc_te, rep_rng = collect_reps(roll_C, o, trainer.state, ds.iter_test(), rep_rng)
        Xd_te, yd_te, rep_rng = collect_reps(roll_D, o, trainer.state, ds.iter_test(), rep_rng)

        probe_D = fit_adam_probe(Xd_tr, yd_tr, Xd_te, yd_te)
        probe_C = fit_adam_probe(Xc_tr, yc_tr, Xd_te, yd_te)
        probe_C_leaky = fit_adam_probe(Xc_tr, yc_tr, Xc_te, yc_te)

        M = abcd_grid(abcd_fn, o, trainer.state, diag_x, diag_y, diag_rng)
        cd_sim = float(M[2, 3])  # C vs D

        rec["head"].append(head)
        rec["probe_D"].append(probe_D)
        rec["probe_C"].append(probe_C)
        rec["probe_C_leaky"].append(probe_C_leaky)
        rec["cd_sim"].append(cd_sim)
        rec["abcd_grids"].append(M.tolist())

        elapsed = time.time() - t0_all
        print(f"  seed={seed} ep={epoch:2d}/{EPOCHS}  head={head:.4f}  "
              f"probe_D={probe_D:.4f}  probe_C={probe_C:.4f}  "
              f"probe_C_leaky={probe_C_leaky:.4f}  cd_sim={cd_sim:.3f}  "
              f"[{fmt(time.time()-t0)}/ep  elapsed={fmt(elapsed)}]", flush=True)

    # offline W_out: C vs D, final representations
    wout_C, key = offline_wout(trainer.orchestrator, trainer.state, ds, cfg, clamped_n, key, "C")
    wout_D, key = offline_wout(trainer.orchestrator, trainer.state, ds, cfg, 0, key, "D")

    rec["wout_C"] = wout_C
    rec["wout_D"] = wout_D
    rec["seed"] = seed
    return rec


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}
    print("Config:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cfg.items()},
          flush=True)

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    t0_all = time.time()
    per_seed = [run_one_seed(s, cfg, ds, t0_all) for s in SEEDS]
    print(f"\nTotal runtime: {fmt(time.time()-t0_all)}", flush=True)

    def stack(k):
        return np.array([r[k] for r in per_seed])  # (n_seeds, EPOCHS)

    # ABCD grids: (n_seeds, EPOCHS, 4, 4) -> mean over seeds (EPOCHS, 4, 4)
    abcd_all = np.array([r["abcd_grids"] for r in per_seed])
    abcd_mean = abcd_all.mean(0)

    out = {
        "seeds": SEEDS, "epochs": EPOCHS,
        "per_seed": per_seed,
        "head_mean": stack("head").mean(0).tolist(),
        "probe_D_mean": stack("probe_D").mean(0).tolist(),
        "probe_C_mean": stack("probe_C").mean(0).tolist(),
        "probe_C_leaky_mean": stack("probe_C_leaky").mean(0).tolist(),
        "cd_sim_mean": stack("cd_sim").mean(0).tolist(),
        "cd_sim_std": stack("cd_sim").std(0).tolist(),
        "abcd_labels": ABCD_LABELS,
        "abcd_grid_mean": abcd_mean.tolist(),
        "wout_C_mean": float(np.mean([r["wout_C"] for r in per_seed])),
        "wout_D_mean": float(np.mean([r["wout_D"] for r in per_seed])),
        "wout_C_std": float(np.std([r["wout_C"] for r in per_seed])),
        "wout_D_std": float(np.std([r["wout_D"] for r in per_seed])),
    }

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "probe_wout_c_vs_d.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {results_dir / 'probe_wout_c_vs_d.json'}", flush=True)

    # ── summary ──
    print("\n=== Final epoch (mean over seeds) ===", flush=True)
    print(f"  head           : {out['head_mean'][-1]:.4f}", flush=True)
    print(f"  probe_D (base) : {out['probe_D_mean'][-1]:.4f}", flush=True)
    print(f"  probe_C (hyp)  : {out['probe_C_mean'][-1]:.4f}", flush=True)
    print(f"  probe_C_leaky  : {out['probe_C_leaky_mean'][-1]:.4f}  (ceiling, uses test labels)", flush=True)
    print(f"  W_out on C     : {out['wout_C_mean']:.4f} ± {out['wout_C_std']:.4f}", flush=True)
    print(f"  W_out on D     : {out['wout_D_mean']:.4f} ± {out['wout_D_std']:.4f}", flush=True)
    print(f"  C–D similarity : {out['cd_sim_mean'][-1]:.3f} (final epoch)", flush=True)

    # ── plots ──
    figures_dir = HERE / "figures"
    figures_dir.mkdir(exist_ok=True)
    ep = np.arange(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for k, lbl, c in [("probe_C_mean", "probe C (fit C → test D)", "#16A34A"),
                      ("probe_D_mean", "probe D (fit D → test D)", "#EA580C"),
                      ("probe_C_leaky_mean", "probe C leaky (fit C → test C)", "#9CA3AF"),
                      ("head_mean", "head (online W_out)", "#2563EB")]:
        ax1.plot(ep, out[k], "-o", color=c, label=lbl)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test accuracy")
    ax1.set_title("Probe / head: fit on C vs D")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    bars = ["W_out\nC", "W_out\nD", "probe\nC", "probe\nD"]
    vals = [out["wout_C_mean"], out["wout_D_mean"],
            out["probe_C_mean"][-1], out["probe_D_mean"][-1]]
    errs = [out["wout_C_std"], out["wout_D_std"], 0, 0]
    ax2.bar(bars, vals, yerr=errs, color=["#16A34A", "#EA580C", "#16A34A", "#EA580C"], alpha=0.85)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax2.set_ylabel("Test accuracy (eval on D)")
    ax2.set_title("Readouts fit on C vs D (final)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Readouts on C vs D — {len(SEEDS)} seeds × {EPOCHS} epochs", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "probe_wout_c_vs_d.png", dpi=120)
    print(f"Plot saved {figures_dir / 'probe_wout_c_vs_d.png'}", flush=True)

    # ── ABCD diagnostics figure ──
    abcd_mean = np.array(out["abcd_grid_mean"])  # (EPOCHS, 4, 4)
    pairs = [(2, 3, "C–D"), (0, 2, "A–C"), (0, 3, "A–D"),
             (1, 2, "B–C"), (0, 1, "A–B"), (1, 3, "B–D")]
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    cd = np.array(out["cd_sim_mean"]); cds = np.array(out["cd_sim_std"])
    for i, j, lbl in pairs:
        axes[0].plot(ep, abcd_mean[:, i, j], "-o", markersize=3, label=lbl)
    axes[0].fill_between(ep, cd - cds, cd + cds, alpha=0.15, color="#2563EB")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Mean per-image cosine sim")
    axes[0].set_title("ABCD pairwise similarity over epochs")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3)

    for ax, e_idx, ttl in [(axes[1], 0, f"ABCD grid — epoch 1"),
                           (axes[2], EPOCHS - 1, f"ABCD grid — epoch {EPOCHS}")]:
        M = abcd_mean[e_idx]
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(4)); ax.set_xticklabels(ABCD_LABELS)
        ax.set_yticks(range(4)); ax.set_yticklabels(ABCD_LABELS)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
        ax.set_title(ttl)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.suptitle(f"ABCD attractor geometry — mean over {len(SEEDS)} seeds", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(figures_dir / "abcd_grids.png", dpi=120)
    print(f"Plot saved {figures_dir / 'abcd_grids.png'}", flush=True)


if __name__ == "__main__":
    main()
