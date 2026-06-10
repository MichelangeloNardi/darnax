"""experiments6/lambda_sweep.py

Sweep lambda_win = lambda_back over 5 values for the channel-entropy CIFAR-10
architecture.  3 seeds × 10 epochs per lambda.  Best config throughout.

Lambda values: [0.0, 0.5, 0.7, 0.9, 1.0]
  1.0  = standard baseline (no decay, identical to standard_run)
  0.0  = Win/WBack contribute only at t=0 (first warmup step), zeroed after

What runs per seed:
  - Online training: Win + J1 + Wout all train (perceptron rule, same as standard_run)
  - Offline Wout:    after 10 ep, Win+J1 frozen, Wout fine-tuned 10 ep (perceptron)

Per-epoch data collected on a fixed held-out diagnostic batch:
  weight norms (Win, J1, WBack Frobenius), field contributions (Win/J1/WBack
  fractional |h|), ABCD attractor sims (6 pairs), ABCD 8x8 grid (before vs
  after one training step), autocorrelation matrix (step-to-step cosine sims).

Output files:
  experiments6/results/sweep.json        -- accuracy + offline results
  experiments6/results/diagnostics.json  -- per-epoch diagnostics (large)
  experiments6/figures/accuracy.png      -- bar chart: final-epoch accuracy vs lambda
  experiments6/figures/accuracy_curves.png
  experiments6/figures/weight_norms.png
  experiments6/figures/field_fractions.png
  experiments6/figures/abcd.png          -- 8x8 grids, final epoch, mean seeds
  experiments6/figures/autocorr.png      -- autocorr matrices, final epoch, seed 0

Run from repo root on cluster:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments6/lambda_sweep.py
"""

from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path

import equinox as eqx
import jax, jax.numpy as jnp, jax.tree_util as jtu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE     = Path(__file__).resolve().parent
REPO     = HERE.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments5"))  # for diagnostics module

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

from diagnostics import (
    collect_autocorr_matrix,
    collect_weight_norms,
    collect_field_contributions,
    collect_abcd_states,
    collect_abcd_8x8,
)

# ── sweep parameters ──────────────────────────────────────────────────────────
LAMBDA_VALUES  = [0.0, 0.5, 0.7, 0.9, 1.0]
SEEDS          = [0, 42, 123]
EPOCHS         = 10
OFFLINE_EPOCHS = 10

# ── architecture constants ────────────────────────────────────────────────────
C, KSIZE     = 16, 5
H, W, POOL   = 32, 32, 8
PROBE_EPOCHS = 20
PROBE_WD     = 1.433e-4


# ── model / optimizer ─────────────────────────────────────────────────────────

def build_model(cfg: dict, key: jax.Array, lam: float):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(
                in_channels=3, out_channels=C, kernel_size=KSIZE,
                threshold=cfg["threshold_win"], strength=1.0, key=keys[0],
                padding_mode="constant", lr=1.0, weight_decay=0.0,
            ),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(
                pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                strength=1.0, threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0,
            ),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    orch  = SequentialOrchestrator(layers=layer_map, lambda_win=lam, lambda_back=lam)
    return state, orch


def make_optimizer(orch, cfg: dict, *, lr_win=True, lr_j=True, lr_wout=True):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orch, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(
            lambda m, r=i, c=j: m.lmap[r][c], labels,
            replace=like(params.lmap[i][j], lbl),
        )

    opt = optax.multi_transform({
        "default": optax.set_to_zero(),
        "win":  sgd(-cfg["lr_win"]) if lr_win  else optax.set_to_zero(),
        "j1":   sgd(-cfg["lr_j"])   if lr_j    else optax.set_to_zero(),
        "wout": sgd(cfg["lr_wout"]) if lr_wout else optax.set_to_zero(),
    }, labels)
    return opt, opt.init(eqx.filter(orch, eqx.is_inexact_array))


# ── data / probe helpers ──────────────────────────────────────────────────────

def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


def run_probe(trainer, ds, key) -> float:
    reps_tr, lbl_tr, reps_te, lbl_te = [], [], [], []
    for xb, yb in ds:
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_tr.append(pool_j1(np.array(trainer.state[1])))
        lbl_tr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_te.append(pool_j1(np.array(trainer.state[1])))
        lbl_te.append(np.argmax(np.array(yb), axis=-1))

    X_tr = torch.from_numpy(np.concatenate(reps_tr)).float()
    y_tr = torch.from_numpy(np.concatenate(lbl_tr)).long()
    X_te = torch.from_numpy(np.concatenate(reps_te)).float()
    y_te = torch.from_numpy(np.concatenate(lbl_te)).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe  = nn.Linear(256, 10, bias=False).to(device)
    opt_p  = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    crit   = nn.CrossEntropyLoss()

    best_test = 0.0
    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb_t, yb_t in loader:
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            opt_p.zero_grad()
            crit(probe(xb_t), yb_t).backward()
            opt_p.step()
        probe.eval()
        with torch.no_grad():
            te = (probe(X_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
        best_test = max(best_test, te)
    return best_test


# ── training helpers ──────────────────────────────────────────────────────────

def train_epoch(trainer, ds, cfg, key):
    decay = cfg["kernel_decay_rate"]
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)
    for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
        trainer.orchestrator = eqx.tree_at(
            path, trainer.orchestrator,
            path(trainer.orchestrator) * (1.0 - decay),
        )
    return trainer, key


def eval_head(trainer, ds, key) -> tuple[float, jax.Array]:
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def fmt(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s"


# ── per-seed run ──────────────────────────────────────────────────────────────

def run_one_seed(
    lam: float, seed: int, cfg: dict, ds,
    diag_rng, t_start: float, epoch_times_all: list,
) -> dict:
    print(f"\n  --- λ={lam:.2f}  seed={seed} ---", flush=True)
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)

    state, orch = build_model(cfg, mk, lam)
    opt, opt_state = make_optimizer(orch, cfg)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )
    warmup_n  = 1
    clamped_n = cfg["clamped_n_iter"]
    free_n    = cfg["free_n_iter"]

    # Fixed diagnostic batch: first test batch, same across all epochs for this seed
    diag_x, diag_y = next(iter(ds.iter_test()))
    diag_x = to_hwc(diag_x)

    head_accs: list[float] = []
    probe_accs: list[float] = []
    per_epoch_diag: list[dict] = []

    t_seed_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        trainer, key = train_epoch(trainer, ds, cfg, key)
        head_acc, key = eval_head(trainer, ds, key)
        probe_acc     = run_probe(trainer, ds, key)

        # --- diagnostics on fixed batch ---
        wn   = collect_weight_norms(trainer.orchestrator)
        ff   = collect_field_contributions(
            trainer.orchestrator, trainer.state,
            diag_x, diag_y, diag_rng, warmup_n,
        )
        abcd = collect_abcd_states(
            trainer.orchestrator, trainer.state,
            diag_x, diag_y, diag_rng, warmup_n, clamped_n, free_n,
        )
        _, orch_after, _, _ = eqx.filter_jit(DynamicalTrainer._train_step_impl)(
            diag_x, diag_y, diag_rng,
            trainer.orchestrator, trainer.state, trainer.ctx,
        )
        abcd_8x8 = collect_abcd_8x8(
            trainer.orchestrator, orch_after, trainer.state,
            diag_x, diag_y, diag_rng, warmup_n, clamped_n, free_n,
        )
        sim, ac_labels, _ = collect_autocorr_matrix(
            trainer.orchestrator, trainer.state,
            diag_x, diag_y, diag_rng, warmup_n, clamped_n, free_n,
        )

        per_epoch_diag.append({
            "epoch":           epoch,
            "weight_norms":    wn,
            "field_fractions": ff,
            "abcd":            abcd,
            "abcd_8x8":        abcd_8x8,
            "autocorr_matrix": sim.tolist(),
            "autocorr_labels": ac_labels,
        })

        t_epoch = time.time() - t0
        epoch_times_all.append(t_epoch)
        elapsed = time.time() - t_start
        print(
            f"  λ={lam:.2f}  seed={seed}  ep={epoch:2d}/{EPOCHS}"
            f"  head={head_acc:.4f}  probe={probe_acc:.4f}"
            f"  cd={abcd['cd_sim_mean']:.3f}"
            f"  [{fmt(t_epoch)}/ep  elapsed={fmt(elapsed)}]",
            flush=True,
        )
        head_accs.append(head_acc)
        probe_accs.append(probe_acc)

    # --- offline Wout: Win + J1 frozen, Wout trains for OFFLINE_EPOCHS epochs ---
    print(f"  [offline Wout] training {OFFLINE_EPOCHS} ep (Win+J1 frozen)...", flush=True)
    opt2, opt2_state = make_optimizer(
        trainer.orchestrator, cfg, lr_win=False, lr_j=False, lr_wout=True,
    )
    trainer2 = DynamicalTrainer(
        orchestrator=trainer.orchestrator, state=trainer.state,
        optimizer=opt2, optimizer_state=opt2_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )
    for ep in range(1, OFFLINE_EPOCHS + 1):
        for xb, yb in ds:
            key = trainer2.train_step(to_hwc(xb), yb, key)
        h, key = eval_head(trainer2, ds, key)
        print(f"    offline ep {ep:2d}/{OFFLINE_EPOCHS}  head={h:.4f}", flush=True)
    offline_head_acc, key = eval_head(trainer2, ds, key)
    print(
        f"  [offline done]  head={offline_head_acc:.4f}"
        f"  total seed: {fmt(time.time() - t_seed_start)}",
        flush=True,
    )

    return {
        "seed":             seed,
        "head_accs":        head_accs,
        "probe_accs":       probe_accs,
        "offline_head_acc": offline_head_acc,
        "per_epoch_diag":   per_epoch_diag,
    }


# ── per-lambda run ────────────────────────────────────────────────────────────

def run_one_lambda(
    lam: float, cfg: dict, ds,
    diag_rng, t_start: float, epoch_times_all: list,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"LAMBDA = {lam:.2f}  ({len(SEEDS)} seeds × {EPOCHS} epochs)", flush=True)
    print(f"{'='*60}", flush=True)

    per_seed = [
        run_one_seed(lam, s, cfg, ds, diag_rng, t_start, epoch_times_all)
        for s in SEEDS
    ]

    head_mat    = np.array([r["head_accs"]        for r in per_seed])
    probe_mat   = np.array([r["probe_accs"]       for r in per_seed])
    offline_arr = np.array([r["offline_head_acc"] for r in per_seed])

    print(
        f"\n  λ={lam:.2f}  head(final)={head_mat[:,-1].mean():.4f}±{head_mat[:,-1].std():.4f}"
        f"  probe(final)={probe_mat[:,-1].mean():.4f}±{probe_mat[:,-1].std():.4f}"
        f"  offline={offline_arr.mean():.4f}±{offline_arr.std():.4f}",
        flush=True,
    )

    return {
        "lambda":            lam,
        "seeds":             SEEDS,
        "epochs":            EPOCHS,
        "per_seed":          per_seed,
        "head_mean":         head_mat.mean(0).tolist(),
        "head_std":          head_mat.std(0).tolist(),
        "probe_mean":        probe_mat.mean(0).tolist(),
        "probe_std":         probe_mat.std(0).tolist(),
        "offline_head_mean": float(offline_arr.mean()),
        "offline_head_std":  float(offline_arr.std()),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results: list[dict], figures_dir: Path) -> None:
    epochs = list(range(1, EPOCHS + 1))
    lams   = [r["lambda"] for r in all_results]
    colors = plt.cm.viridis(np.linspace(0.05, 0.90, len(lams)))
    n_seeds = len(SEEDS)

    # 1. bar chart: final-epoch accuracy (online head, probe, offline head) vs lambda
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(lams))
    width = 0.25
    head_f    = [r["head_mean"][-1]       for r in all_results]
    head_s    = [r["head_std"][-1]        for r in all_results]
    probe_f   = [r["probe_mean"][-1]      for r in all_results]
    probe_s   = [r["probe_std"][-1]       for r in all_results]
    offline_f = [r["offline_head_mean"]   for r in all_results]
    offline_s = [r["offline_head_std"]    for r in all_results]

    ax.bar(x - width, head_f,    width, yerr=head_s,    label="Wout online (perceptron)",
           color="#2563EB", alpha=0.85, capsize=4)
    ax.bar(x,         probe_f,   width, yerr=probe_s,   label="Linear probe (J1, Adam)",
           color="#EA580C", alpha=0.85, capsize=4)
    ax.bar(x + width, offline_f, width, yerr=offline_s, label="Wout offline (frozen J1, 10ep)",
           color="#16A34A", alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={l:.1f}" for l in lams], fontsize=11)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy vs lambda — epoch {EPOCHS}, mean±std over {n_seeds} seeds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy.png", dpi=150)
    plt.close(fig)
    print("saved accuracy.png", flush=True)

    # 2. per-epoch accuracy curves per lambda
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r, col in zip(all_results, colors):
        lbl = f"λ={r['lambda']:.1f}"
        hm, hs = np.array(r["head_mean"]),  np.array(r["head_std"])
        pm, ps = np.array(r["probe_mean"]), np.array(r["probe_std"])
        ax1.plot(epochs, hm, "-o", color=col, label=lbl, markersize=4, linewidth=1.8)
        ax1.fill_between(epochs, hm - hs, hm + hs, alpha=0.12, color=col)
        ax2.plot(epochs, pm, "-s", color=col, label=lbl, markersize=4, linewidth=1.8)
        ax2.fill_between(epochs, pm - ps, pm + ps, alpha=0.12, color=col)
    for ax, title in [(ax1, "Head accuracy (Wout online)"), (ax2, "Linear probe (J1, Adam)")]:
        ax.set_xlabel("Epoch"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Training curves — lambda sweep ({n_seeds} seeds ±1 std)")
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_curves.png", dpi=150)
    plt.close(fig)
    print("saved accuracy_curves.png", flush=True)

    # 3. weight norms over epochs per lambda (mean over seeds)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for r, col in zip(all_results, colors):
        lbl = f"λ={r['lambda']:.1f}"
        ns  = len(r["per_seed"])
        win_m = np.zeros((ns, EPOCHS))
        j1_m  = np.zeros((ns, EPOCHS))
        wb_m  = np.zeros((ns, EPOCHS))
        for si, sr in enumerate(r["per_seed"]):
            for ei, ed in enumerate(sr["per_epoch_diag"]):
                win_m[si, ei] = ed["weight_norms"]["win"]
                j1_m[si, ei]  = ed["weight_norms"]["j1"]
                wb_m[si, ei]  = ed["weight_norms"]["wback"]
        axes[0].plot(epochs, win_m.mean(0), color=col, label=lbl, linewidth=1.8)
        axes[1].plot(epochs, j1_m.mean(0),  color=col, label=lbl, linewidth=1.8)
        axes[2].plot(epochs, wb_m.mean(0),  color=col, label=lbl, linewidth=1.8, linestyle="--")
    titles = ["Win  ‖W_in‖_F", "J1  ‖J‖_F", "WBack  ‖W_b‖_F  (frozen)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t); ax.set_xlabel("Epoch"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"Weight norms over training (mean over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(figures_dir / "weight_norms.png", dpi=150)
    plt.close(fig)
    print("saved weight_norms.png", flush=True)

    # 4. field contributions at final epoch — grouped bar per lambda
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x     = np.arange(len(lams))
    width = 0.25
    win_f, j1_f, wb_f = [], [], []
    for r in all_results:
        wf_s, j1f_s, wbf_s = [], [], []
        for sr in r["per_seed"]:
            ff = sr["per_epoch_diag"][-1]["field_fractions"]
            wf_s.append(ff["win"]); j1f_s.append(ff["j1"]); wbf_s.append(ff["wback"])
        win_f.append(np.mean(wf_s)); j1_f.append(np.mean(j1f_s)); wb_f.append(np.mean(wbf_s))
    ax.bar(x - width, win_f, width, label="Win",   color="#16A34A", alpha=0.85)
    ax.bar(x,         j1_f,  width, label="J1",    color="#2563EB", alpha=0.85)
    ax.bar(x + width, wb_f,  width, label="WBack", color="#DC2626", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={l:.1f}" for l in lams], fontsize=11)
    ax.set_ylabel("Fractional |h| contribution")
    ax.set_title(f"J1 field contributions at epoch {EPOCHS} (mean over {n_seeds} seeds)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "field_fractions.png", dpi=150)
    plt.close(fig)
    print("saved field_fractions.png", flush=True)

    # 5. ABCD 8×8 grids — final epoch, mean over seeds, one panel per lambda
    labels_8x8 = ["A", "B", "C", "D", "A'", "B'", "C'", "D'"]
    mats_8x8   = []
    all_off_vals: list[float] = []
    for r in all_results:
        mats = []
        for sr in r["per_seed"]:
            ed = sr["per_epoch_diag"][-1]
            if "abcd_8x8" in ed:
                mats.append(np.array(ed["abcd_8x8"]["matrix"]))
        if mats:
            m = np.nanmean(mats, axis=0)
            mats_8x8.append(m)
            off = m[~np.eye(8, dtype=bool) & ~np.isnan(m)]
            all_off_vals.extend(off.tolist())
        else:
            mats_8x8.append(np.full((8, 8), np.nan))
    vmin_8x8 = float(np.percentile(all_off_vals, 2)) if all_off_vals else 0.85
    cmap_8x8 = plt.cm.viridis.copy(); cmap_8x8.set_bad("#cccccc")

    fig, axes = plt.subplots(1, len(lams), figsize=(5.5 * len(lams), 5.2))
    for ax, mat, lam in zip(axes, mats_8x8, lams):
        im = ax.imshow(np.ma.masked_invalid(mat), vmin=vmin_8x8, vmax=1.0, cmap=cmap_8x8)
        ax.set_xticks(range(8)); ax.set_xticklabels(labels_8x8, fontsize=8)
        ax.set_yticks(range(8)); ax.set_yticklabels(labels_8x8, fontsize=8)
        ax.axhline(3.5, color="white", lw=2); ax.axvline(3.5, color="white", lw=2)
        for i in range(8):
            for j in range(8):
                v = mat[i, j]
                if not np.isnan(v):
                    tc = "white" if v < (vmin_8x8 + 1.0) / 2 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=tc)
        ax.set_title(f"λ={lam:.1f}")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"ABCD × A'B'C'D' cosine similarity — epoch {EPOCHS}, mean {n_seeds} seeds\n"
        "Top-left 4×4 = ABCD (before update)   Bottom-right = A'B'C'D' (after one step)\n"
        "Off-diagonal blocks = how much fixed points shift after one training step",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "abcd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved abcd.png", flush=True)

    # 6. autocorrelation matrices — final epoch, seed 0, one panel per lambda
    fig, axes = plt.subplots(1, len(lams), figsize=(5 * len(lams), 4.5))
    for ax, r, lam in zip(axes, all_results, lams):
        ed         = r["per_seed"][0]["per_epoch_diag"][-1]
        sim_all    = np.array(ed["autocorr_matrix"])
        labels_all = ed["autocorr_labels"]
        idx    = [i for i, lb in enumerate(labels_all) if lb != "init"]
        sim    = sim_all[np.ix_(idx, idx)]
        lbls   = [labels_all[i] for i in idx]
        off    = sim[~np.eye(len(sim), dtype=bool)]
        vmin   = float(np.percentile(off, 2))
        im = ax.imshow(sim, vmin=vmin, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, fontsize=7, rotation=90)
        ax.set_yticks(range(len(lbls))); ax.set_yticklabels(lbls, fontsize=7)
        n_w = sum(1 for lb in lbls if lb.startswith("W"))
        n_c = sum(1 for lb in lbls if lb.startswith("C"))
        for b in [n_w - 0.5, n_w + n_c - 0.5]:
            ax.axhline(b, color="white", lw=1.2); ax.axvline(b, color="white", lw=1.2)
        ax.set_title(f"λ={lam:.1f}")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"J1 step-to-step cosine similarity — epoch {EPOCHS}, seed 0  (init excluded)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "autocorr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved autocorr.png", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    print("Config:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cfg.items()},
          flush=True)
    print(f"\nLambda values: {LAMBDA_VALUES}", flush=True)
    print(f"Seeds: {SEEDS}   Epochs: {EPOCHS}   Offline epochs: {OFFLINE_EPOCHS}", flush=True)

    ds = Cifar10(
        batch_size=32, x_transform="identity", label_mode="pm1",
        linear_projection=None, rescale=True,
    )
    ds.build(jax.random.PRNGKey(0))

    # Fixed RNG for all diagnostic calls — reproducible across epochs, seeds, lambdas
    diag_rng = jax.random.PRNGKey(42)

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    t_start = time.time()
    epoch_times_all: list[float] = []

    all_results = [
        run_one_lambda(lam, cfg, ds, diag_rng, t_start, epoch_times_all)
        for lam in LAMBDA_VALUES
    ]

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {fmt(total_time)}", flush=True)

    # --- summary table ---
    print(f"\n{'Lambda':>8}  {'Head(final)':>12}  {'Probe(final)':>13}  {'Offline':>10}", flush=True)
    print("-" * 52, flush=True)
    for r in all_results:
        print(
            f"  {r['lambda']:>5.2f}  "
            f"  {r['head_mean'][-1]:.4f}±{r['head_std'][-1]:.4f}  "
            f"  {r['probe_mean'][-1]:.4f}±{r['probe_std'][-1]:.4f}  "
            f"  {r['offline_head_mean']:.4f}±{r['offline_head_std']:.4f}",
            flush=True,
        )

    # --- save accuracy results (compact) ---
    sweep_out = {
        "lambda_values": LAMBDA_VALUES,
        "seeds":         SEEDS,
        "epochs":        EPOCHS,
        "offline_epochs": OFFLINE_EPOCHS,
        "per_lambda": [
            {
                "lambda":            r["lambda"],
                "per_seed": [
                    {
                        "seed":             s["seed"],
                        "head_accs":        s["head_accs"],
                        "probe_accs":       s["probe_accs"],
                        "offline_head_acc": s["offline_head_acc"],
                    }
                    for s in r["per_seed"]
                ],
                "head_mean":         r["head_mean"],
                "head_std":          r["head_std"],
                "probe_mean":        r["probe_mean"],
                "probe_std":         r["probe_std"],
                "offline_head_mean": r["offline_head_mean"],
                "offline_head_std":  r["offline_head_std"],
            }
            for r in all_results
        ],
    }
    sweep_path = results_dir / "sweep.json"
    sweep_path.write_text(json.dumps(sweep_out, indent=2))
    print(f"\nAccuracy results saved to {sweep_path}", flush=True)

    # --- save diagnostics (per-epoch, separate file) ---
    diag_out = {
        "lambda_values": LAMBDA_VALUES,
        "seeds":         SEEDS,
        "epochs":        EPOCHS,
        "warmup_n":      1,
        "clamped_n":     cfg["clamped_n_iter"],
        "free_n":        cfg["free_n_iter"],
        "per_lambda": [
            {
                "lambda":   r["lambda"],
                "per_seed": [
                    {
                        "seed":           s["seed"],
                        "per_epoch_diag": s["per_epoch_diag"],
                    }
                    for s in r["per_seed"]
                ],
            }
            for r in all_results
        ],
    }
    diag_path = results_dir / "diagnostics.json"
    diag_path.write_text(json.dumps(diag_out, indent=2))
    print(f"Diagnostic results saved to {diag_path}", flush=True)

    # --- plots ---
    print("\nGenerating figures...", flush=True)
    plot_results(all_results, figures_dir)
    print(f"\nAll done. Total: {fmt(total_time)}", flush=True)


if __name__ == "__main__":
    main()
