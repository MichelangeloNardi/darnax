"""experiments6/plot_sweep.py

Regenerate all figures from saved sweep + diagnostics JSON files.
Run after lambda_sweep.py has completed (or partially completed).

Usage:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments6/plot_sweep.py
  # or locally:
  uv run python experiments6/plot_sweep.py
"""

from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE        = Path(__file__).resolve().parent
RESULTS     = HERE / "results"
FIGURES     = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

sweep_path = RESULTS / "sweep.json"
diag_path  = RESULTS / "diagnostics.json"

if not sweep_path.exists():
    print(f"ERROR: {sweep_path} not found. Run lambda_sweep.py first.")
    sys.exit(1)
if not diag_path.exists():
    print(f"ERROR: {diag_path} not found. Run lambda_sweep.py first.")
    sys.exit(1)

with open(sweep_path) as f:
    sweep = json.load(f)
with open(diag_path) as f:
    diag = json.load(f)

LAMBDA_VALUES = sweep["lambda_values"]
SEEDS         = sweep["seeds"]
EPOCHS        = sweep["epochs"]
OFFLINE_EPOCHS = sweep["offline_epochs"]
n_seeds       = len(SEEDS)
n_lams        = len(LAMBDA_VALUES)
warmup_n      = diag["warmup_n"]
clamped_n     = diag["clamped_n"]
free_n        = diag["free_n"]

epochs  = list(range(1, EPOCHS + 1))
colors  = plt.cm.viridis(np.linspace(0.05, 0.90, n_lams))


# ── helpers ───────────────────────────────────────────────────────────────────

def sweep_rec(lam: float) -> dict:
    for r in sweep["per_lambda"]:
        if abs(r["lambda"] - lam) < 1e-9:
            return r
    raise KeyError(lam)


def diag_rec(lam: float) -> dict:
    for r in diag["per_lambda"]:
        if abs(r["lambda"] - lam) < 1e-9:
            return r
    raise KeyError(lam)


def get_epoch_diag(lam: float, seed_idx: int, epoch: int) -> dict:
    dr = diag_rec(lam)
    return dr["per_seed"][seed_idx]["per_epoch_diag"][epoch - 1]


# ── 1. accuracy bar chart (final epoch) ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(n_lams)
width = 0.25
head_f, head_s, probe_f, probe_s, offline_f, offline_s = [], [], [], [], [], []
for lam in LAMBDA_VALUES:
    r = sweep_rec(lam)
    head_f.append(r["head_mean"][-1]);    head_s.append(r["head_std"][-1])
    probe_f.append(r["probe_mean"][-1]);  probe_s.append(r["probe_std"][-1])
    offline_f.append(r["offline_head_mean"]); offline_s.append(r["offline_head_std"])

ax.bar(x - width, head_f,    width, yerr=head_s,    label="Wout online (perceptron)",
       color="#2563EB", alpha=0.85, capsize=4)
ax.bar(x,         probe_f,   width, yerr=probe_s,   label="Linear probe (J1, Adam)",
       color="#EA580C", alpha=0.85, capsize=4)
ax.bar(x + width, offline_f, width, yerr=offline_s, label=f"Wout offline (frozen J1, {OFFLINE_EPOCHS}ep)",
       color="#16A34A", alpha=0.85, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels([f"λ={l:.1f}" for l in LAMBDA_VALUES], fontsize=11)
ax.set_ylabel("Test accuracy")
ax.set_title(f"Accuracy vs lambda — epoch {EPOCHS}, mean±std over {n_seeds} seeds")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(FIGURES / "accuracy.png", dpi=150)
plt.close(fig)
print("saved accuracy.png")


# ── 2. per-epoch accuracy curves ──────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for lam, col in zip(LAMBDA_VALUES, colors):
    r   = sweep_rec(lam)
    lbl = f"λ={lam:.1f}"
    hm, hs = np.array(r["head_mean"]),  np.array(r["head_std"])
    pm, ps = np.array(r["probe_mean"]), np.array(r["probe_std"])
    ax1.plot(epochs, hm, "-o", color=col, label=lbl, markersize=4, linewidth=1.8)
    ax1.fill_between(epochs, hm - hs, hm + hs, alpha=0.12, color=col)
    ax2.plot(epochs, pm, "-s", color=col, label=lbl, markersize=4, linewidth=1.8)
    ax2.fill_between(epochs, pm - ps, pm + ps, alpha=0.12, color=col)
for ax, title in [(ax1, "Head accuracy (Wout online)"), (ax2, "Linear probe (J1, Adam)")]:
    ax.set_xlabel("Epoch"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle(f"Training curves — lambda sweep ({n_seeds} seeds ±1 std)")
fig.tight_layout()
fig.savefig(FIGURES / "accuracy_curves.png", dpi=150)
plt.close(fig)
print("saved accuracy_curves.png")


# ── 3. weight norms ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for lam, col in zip(LAMBDA_VALUES, colors):
    dr  = diag_rec(lam)
    lbl = f"λ={lam:.1f}"
    win_m = np.zeros((n_seeds, EPOCHS))
    j1_m  = np.zeros((n_seeds, EPOCHS))
    wb_m  = np.zeros((n_seeds, EPOCHS))
    for si, sr in enumerate(dr["per_seed"]):
        for ei, ed in enumerate(sr["per_epoch_diag"]):
            win_m[si, ei] = ed["weight_norms"]["win"]
            j1_m[si, ei]  = ed["weight_norms"]["j1"]
            wb_m[si, ei]  = ed["weight_norms"]["wback"]
    axes[0].plot(epochs, win_m.mean(0), color=col, label=lbl, linewidth=1.8)
    axes[1].plot(epochs, j1_m.mean(0),  color=col, label=lbl, linewidth=1.8)
    axes[2].plot(epochs, wb_m.mean(0),  color=col, label=lbl, linewidth=1.8, linestyle="--")
for ax, t in zip(axes, ["Win ‖W_in‖_F", "J1 ‖J‖_F", "WBack ‖W_b‖_F (frozen)"]):
    ax.set_title(t); ax.set_xlabel("Epoch"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle(f"Weight norms over training (mean over {n_seeds} seeds)")
fig.tight_layout()
fig.savefig(FIGURES / "weight_norms.png", dpi=150)
plt.close(fig)
print("saved weight_norms.png")


# ── 4. field contributions bar (final epoch) ─────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4.5))
x     = np.arange(n_lams); width = 0.25
win_f, j1_f, wb_f = [], [], []
for lam in LAMBDA_VALUES:
    dr = diag_rec(lam)
    wf_s, j1f_s, wbf_s = [], [], []
    for sr in dr["per_seed"]:
        ff = sr["per_epoch_diag"][-1]["field_fractions"]
        wf_s.append(ff["win"]); j1f_s.append(ff["j1"]); wbf_s.append(ff["wback"])
    win_f.append(np.mean(wf_s)); j1_f.append(np.mean(j1f_s)); wb_f.append(np.mean(wbf_s))
ax.bar(x - width, win_f, width, label="Win",   color="#16A34A", alpha=0.85)
ax.bar(x,         j1_f,  width, label="J1",    color="#2563EB", alpha=0.85)
ax.bar(x + width, wb_f,  width, label="WBack", color="#DC2626", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"λ={l:.1f}" for l in LAMBDA_VALUES], fontsize=11)
ax.set_ylabel("Fractional |h| contribution")
ax.set_title(f"J1 field contributions at epoch {EPOCHS} (mean over {n_seeds} seeds)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "field_fractions.png", dpi=150)
plt.close(fig)
print("saved field_fractions.png")


# ── 5. field contributions over epochs (line curves per lambda) ───────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
comp_names = ["win", "j1", "wback"]
comp_cols  = ["#16A34A", "#2563EB", "#DC2626"]
comp_titles = ["Win fraction", "J1 fraction", "WBack fraction"]
for lam, col in zip(LAMBDA_VALUES, colors):
    dr  = diag_rec(lam)
    lbl = f"λ={lam:.1f}"
    for ci, comp in enumerate(comp_names):
        mat = np.zeros((n_seeds, EPOCHS))
        for si, sr in enumerate(dr["per_seed"]):
            for ei, ed in enumerate(sr["per_epoch_diag"]):
                mat[si, ei] = ed["field_fractions"][comp]
        axes[ci].plot(epochs, mat.mean(0), color=col, label=lbl, linewidth=1.8)
for ax, t in zip(axes, comp_titles):
    ax.set_title(t); ax.set_xlabel("Epoch"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
fig.suptitle(f"J1 field contributions over training (mean over {n_seeds} seeds)")
fig.tight_layout()
fig.savefig(FIGURES / "field_fractions_curves.png", dpi=150)
plt.close(fig)
print("saved field_fractions_curves.png")


# ── 6. ABCD 8×8 grids — final epoch, mean seeds ──────────────────────────────

labels_8x8 = ["A", "B", "C", "D", "A'", "B'", "C'", "D'"]
mats_8x8   = []
all_off_vals: list[float] = []
for lam in LAMBDA_VALUES:
    dr = diag_rec(lam)
    mats = []
    for sr in dr["per_seed"]:
        ed = sr["per_epoch_diag"][-1]
        if "abcd_8x8" in ed:
            mats.append(np.array(ed["abcd_8x8"]["matrix"]))
    if mats:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(mats, axis=0)
        mats_8x8.append(m)
        off = m[~np.eye(8, dtype=bool) & ~np.isnan(m)]
        all_off_vals.extend(off.tolist())
    else:
        mats_8x8.append(np.full((8, 8), np.nan))

vmin_8x8 = float(np.percentile(all_off_vals, 2)) if all_off_vals else 0.85
cmap_8x8 = plt.cm.viridis.copy(); cmap_8x8.set_bad("#cccccc")
fig, axes = plt.subplots(1, n_lams, figsize=(5.5 * n_lams, 5.2))
for ax, mat, lam in zip(axes, mats_8x8, LAMBDA_VALUES):
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
    "Top-left 4×4 = ABCD (before update)   Bottom-right = A'B'C'D' (after one step)",
    y=1.04,
)
fig.tight_layout()
fig.savefig(FIGURES / "abcd.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved abcd.png")


# ── 7. autocorrelation matrices — final epoch, seed 0 ────────────────────────

fig, axes = plt.subplots(1, n_lams, figsize=(5 * n_lams, 4.5))
for ax, lam in zip(axes, LAMBDA_VALUES):
    ed         = get_epoch_diag(lam, 0, EPOCHS)
    sim_all    = np.array(ed["autocorr_matrix"])
    labels_all = ed["autocorr_labels"]
    idx  = [i for i, lb in enumerate(labels_all) if lb != "init"]
    sim  = sim_all[np.ix_(idx, idx)]
    lbls = [labels_all[i] for i in idx]
    off  = sim[~np.eye(len(sim), dtype=bool)]
    vmin = float(np.percentile(off, 2))
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
fig.savefig(FIGURES / "autocorr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved autocorr.png")


# ── 8. CD similarity over epochs per lambda ───────────────────────────────────
# This shows the training/inference gap trajectory across training.

fig, ax = plt.subplots(figsize=(8, 4))
for lam, col in zip(LAMBDA_VALUES, colors):
    dr   = diag_rec(lam)
    lbl  = f"λ={lam:.1f}"
    cd_m = np.zeros((n_seeds, EPOCHS))
    for si, sr in enumerate(dr["per_seed"]):
        for ei, ed in enumerate(sr["per_epoch_diag"]):
            cd_m[si, ei] = ed["abcd"]["cd_sim_mean"]
    ax.plot(epochs, cd_m.mean(0), "-o", color=col, label=lbl, markersize=4, linewidth=1.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("C-D cosine similarity (mean)")
ax.set_title(f"Train/inference gap (C-D sim) over epochs — {n_seeds} seeds")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "cd_similarity.png", dpi=150)
plt.close(fig)
print("saved cd_similarity.png")


# ── summary table ─────────────────────────────────────────────────────────────

print(f"\n{'Lambda':>8}  {'Head(final)':>12}  {'Probe(final)':>13}  {'Offline Wout':>13}")
print("-" * 55)
for lam in LAMBDA_VALUES:
    r = sweep_rec(lam)
    print(
        f"  {lam:>5.2f}  "
        f"  {r['head_mean'][-1]:.4f}±{r['head_std'][-1]:.4f}  "
        f"  {r['probe_mean'][-1]:.4f}±{r['probe_std'][-1]:.4f}  "
        f"  {r['offline_head_mean']:.4f}±{r['offline_head_std']:.4f}"
    )
print("\nAll figures saved to experiments6/figures/")
