"""experiments7/plot_sigma03.py

Detailed diagnostic plots for σ=0.3, with mean ± std across 3 seeds.
Selected epoch snapshots: 1, 3, 5, 7, 10.

Loads from experiments7/results/diagnostics_merged.json.
Saves to experiments7/figures/sigma03_*.png.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

TARGET_SIGMA    = 0.3
SELECTED_EPOCHS = [1, 3, 5, 7, 10]
SELECTED_IDX    = [e - 1 for e in SELECTED_EPOCHS]

# ── load ──────────────────────────────────────────────────────────────────────

with open(RESULTS / "diagnostics_merged.json") as f:
    d = json.load(f)

warmup_n  = d["warmup_n"]
clamped_n = d["clamped_n"]
free_n    = d["free_n"]

sigma03 = next(e for e in d["per_noise"] if abs(e["noise_std"] - TARGET_SIGMA) < 0.01)
seeds_data = sigma03["per_seed"]   # list of 3 dicts, each with per_epoch_diag
N_SEEDS = len(seeds_data)
SEEDS   = [s["seed"] for s in seeds_data]
EPOCHS  = len(seeds_data[0]["per_epoch_diag"])
all_epochs = list(range(1, EPOCHS + 1))

print(f"σ={TARGET_SIGMA}  seeds={SEEDS}  epochs={EPOCHS}")
print(f"Selected epochs for matrix plots: {SELECTED_EPOCHS}")


# ── helpers ───────────────────────────────────────────────────────────────────

ABCD_PAIRS  = [("a","b"),("a","c"),("a","d"),("b","c"),("b","d"),("c","d")]
PAIR_LABELS = ["A-B","A-C","A-D","B-C","B-D","C-D"]
PAIR_COLORS = ["#2563EB","#7C3AED","#DB2777","#EA580C","#16A34A","#DC2626"]


def _get_scalar_trace(key_fn, seeds_data=seeds_data):
    """Extract (mean, std) per epoch from a scalar diagnostic across seeds."""
    mat = np.array([[key_fn(ep) for ep in sd["per_epoch_diag"]] for sd in seeds_data])
    return mat.mean(0), mat.std(0)


def _shade(ax, epochs, mean, std, color, label=None, lw=1.8, ls="-", marker="o", ms=4):
    ax.plot(epochs, mean, ls + marker, color=color, linewidth=lw, markersize=ms, label=label)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.18, color=color)


# ── Figure 1: ABCD 8×8 grids — mean across 3 seeds, 5 selected epochs ───────

labels_8x8 = ["A","B","C","D","A'","B'","C'","D'"]

fig, axes = plt.subplots(1, len(SELECTED_EPOCHS), figsize=(5.5 * len(SELECTED_EPOCHS), 5.5))
all_off = []
for ep_idx in SELECTED_IDX:
    for sd in seeds_data:
        m = np.array(sd["per_epoch_diag"][ep_idx]["abcd_8x8"]["matrix"])
        all_off.extend(m[~np.eye(8, dtype=bool)].tolist())
vmin = float(np.percentile(all_off, 2))
cmap = plt.cm.viridis

for ax, ep_idx, ep in zip(axes, SELECTED_IDX, SELECTED_EPOCHS):
    mats = np.array([sd["per_epoch_diag"][ep_idx]["abcd_8x8"]["matrix"] for sd in seeds_data])
    m = mats.mean(0)
    im = ax.imshow(m, vmin=vmin, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(8)); ax.set_xticklabels(labels_8x8, fontsize=8)
    ax.set_yticks(range(8)); ax.set_yticklabels(labels_8x8, fontsize=8)
    ax.axhline(3.5, color="white", lw=2); ax.axvline(3.5, color="white", lw=2)
    for i in range(8):
        for j in range(8):
            v = m[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v < (vmin + 1.0) / 2 else "black")
    ax.set_title(f"Epoch {ep}", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.75)

fig.suptitle(
    f"ABCD × A'B'C'D' cosine similarity — σ={TARGET_SIGMA}  mean over {N_SEEDS} seeds\n"
    f"(rows=before update, cols=after update; white line separates before/after blocks)",
    fontsize=11, y=1.02
)
fig.tight_layout()
fig.savefig(FIGURES / "sigma03_abcd_8x8.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved sigma03_abcd_8x8.png")


# ── Figure 2: Autocorrelation — mean across seeds, 5 selected epochs ─────────

labels_ac_full = seeds_data[0]["per_epoch_diag"][0]["autocorr_labels"]
drop_init = [i for i, lb in enumerate(labels_ac_full) if lb != "init"]
labels_ac = [labels_ac_full[i] for i in drop_init]
n_w = sum(1 for lb in labels_ac if lb.startswith("W"))
n_c = sum(1 for lb in labels_ac if lb.startswith("C"))

fig, axes = plt.subplots(1, len(SELECTED_EPOCHS), figsize=(5.0 * len(SELECTED_EPOCHS), 5.0))
for ax, ep_idx, ep in zip(axes, SELECTED_IDX, SELECTED_EPOCHS):
    sims = np.array([
        np.array(sd["per_epoch_diag"][ep_idx]["autocorr_matrix"])[np.ix_(drop_init, drop_init)]
        for sd in seeds_data
    ])
    m = sims.mean(0)
    off = m[~np.eye(len(m), dtype=bool)]
    im = ax.imshow(m, vmin=float(np.percentile(off, 2)), vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels_ac))); ax.set_xticklabels(labels_ac, fontsize=7, rotation=90)
    ax.set_yticks(range(len(labels_ac))); ax.set_yticklabels(labels_ac, fontsize=7)
    for b in [n_w - 0.5, n_w + n_c - 0.5]:
        ax.axhline(b, color="white", lw=1.2); ax.axvline(b, color="white", lw=1.2)
    ax.set_title(f"Epoch {ep}", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(
    f"J1 autocorrelation — σ={TARGET_SIGMA}  mean over {N_SEEDS} seeds  (init excluded)\n"
    f"W=warmup, C=clamped (label on), F=free (label off)",
    fontsize=11, y=1.02
)
fig.tight_layout()
fig.savefig(FIGURES / "sigma03_autocorr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved sigma03_autocorr.png")


# ── Figure 3: Autocorrelation deep — full 13×13 + step-to-step bar ───────────

labels_full = seeds_data[0]["per_epoch_diag"][0]["autocorr_labels"]
n_init = 1; n_w2 = n_init + warmup_n; n_c2 = n_w2 + clamped_n

fig, axes = plt.subplots(2, len(SELECTED_EPOCHS), figsize=(5.5 * len(SELECTED_EPOCHS), 9))
for col_i, (ep_idx, ep) in enumerate(zip(SELECTED_IDX, SELECTED_EPOCHS)):
    sims_full = np.array([
        np.array(sd["per_epoch_diag"][ep_idx]["autocorr_matrix"])
        for sd in seeds_data
    ])
    m = sims_full.mean(0)
    off = m[~np.eye(len(m), dtype=bool)]

    ax_top = axes[0, col_i]
    im = ax_top.imshow(m, vmin=float(np.percentile(off, 2)), vmax=1.0, cmap="viridis")
    ax_top.set_xticks(range(len(labels_full)))
    ax_top.set_xticklabels(labels_full, fontsize=6.5, rotation=90)
    ax_top.set_yticks(range(len(labels_full)))
    ax_top.set_yticklabels(labels_full, fontsize=6.5)
    for b in [n_init - 0.5, n_w2 - 0.5, n_c2 - 0.5]:
        ax_top.axhline(b, color="white", lw=1.2); ax_top.axvline(b, color="white", lw=1.2)
    ax_top.set_title(f"Epoch {ep}", fontsize=11)
    plt.colorbar(im, ax=ax_top, shrink=0.75)

    ax_bot = axes[1, col_i]
    # step-to-step sim: mean and std across seeds
    step_means = np.array([
        [sims_full[si, i, i-1] for i in range(1, len(labels_full))]
        for si in range(N_SEEDS)
    ])
    sm = step_means.mean(0); ss = step_means.std(0)
    x = range(len(sm))
    ax_bot.bar(x, sm, color="#2563EB", alpha=0.75)
    ax_bot.errorbar(x, sm, yerr=ss, fmt="none", color="#1e3a8a", capsize=3, linewidth=1.2)
    ax_bot.set_xticks(list(x))
    ax_bot.set_xticklabels(labels_full[1:], fontsize=7, rotation=90)
    for b in [warmup_n - 0.5, warmup_n + clamped_n - 0.5]:
        ax_bot.axvline(b, color="#DC2626", lw=1.5, linestyle="--", alpha=0.7)
    ax_bot.set_ylim(0.85, 1.01)
    ax_bot.set_ylabel("Cosine sim with prev step")
    ax_bot.set_title(f"Epoch {ep} — step-to-step", fontsize=11)
    ax_bot.grid(axis="y", alpha=0.3)

fig.suptitle(
    f"Autocorrelation deep dive — σ={TARGET_SIGMA}  mean ± std over {N_SEEDS} seeds\n"
    f"Top: full 13×13 (init→W→C→F). Bottom: J1 sim with previous step (red = phase boundary)",
    fontsize=11, y=1.01
)
fig.tight_layout()
fig.savefig(FIGURES / "sigma03_autocorr_deep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved sigma03_autocorr_deep.png")


# ── Figure 4: ABCD pairwise sims — all 6 pairs, mean ± std over epochs ───────

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, pair, lbl, col in zip(axes.ravel(), ABCD_PAIRS, PAIR_LABELS, PAIR_COLORS):
    p, q = pair
    mean, std = _get_scalar_trace(lambda ep, p=p, q=q: ep["abcd"][f"{p}{q}_sim_mean"])
    _shade(ax, all_epochs, mean, std, col, label=lbl)
    for ep in SELECTED_EPOCHS:
        ax.axvline(ep, color="gray", lw=0.7, linestyle="--", alpha=0.5)
    ax.set_title(lbl, fontsize=12, color=col)
    ax.set_ylim(0.85, 1.01)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cosine similarity")
    ax.grid(alpha=0.3); ax.set_xticks(all_epochs)

fig.suptitle(
    f"ABCD pairwise cosine sims — σ={TARGET_SIGMA}  mean ± 1 std over {N_SEEDS} seeds\n"
    "Gray dashed = epochs shown in matrix plots",
    fontsize=12
)
fig.tight_layout()
fig.savefig(FIGURES / "sigma03_abcd_pairs.png", dpi=150)
plt.close(fig)
print("saved sigma03_abcd_pairs.png")


# ── Figure 5: All pairs + weight norms on one figure ─────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: all 6 pairs overlaid
for pair, lbl, col in zip(ABCD_PAIRS, PAIR_LABELS, PAIR_COLORS):
    p, q = pair
    mean, std = _get_scalar_trace(lambda ep, p=p, q=q: ep["abcd"][f"{p}{q}_sim_mean"])
    lw = 2.5 if pair == ("c","d") else 1.2
    ls = "-" if pair == ("c","d") else "--"
    _shade(ax1, all_epochs, mean, std, col, label=lbl, lw=lw, ls=ls)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cosine similarity")
ax1.set_title("All ABCD pairs (C-D bold)")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.set_xticks(all_epochs); ax1.set_ylim(0.85, 1.01)

# Right: weight norms
for key, label, col, ls in [("win","Win ‖W‖_F","#16A34A","-"),
                              ("j1","J1 ‖J‖_F","#2563EB","-"),
                              ("wback","WBack ‖W‖_F (frozen)","#DC2626","--")]:
    mean, std = _get_scalar_trace(lambda ep, k=key: ep["weight_norms"][k])
    _shade(ax2, all_epochs, mean, std, col, label=label, ls=ls)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Frobenius norm")
ax2.set_title("Weight norms over training")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_xticks(all_epochs)

fig.suptitle(f"σ={TARGET_SIGMA}  mean ± 1 std over {N_SEEDS} seeds — dynamics and norms", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "sigma03_dynamics.png", dpi=150)
plt.close(fig)
print("saved sigma03_dynamics.png")


# ── Figure 6: Field contributions — fractional and absolute ──────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for key, label, col in [("win","Win","#16A34A"),("j1","J1","#2563EB"),("wback","WBack","#DC2626")]:
    frac_m, frac_s = _get_scalar_trace(lambda ep, k=key: ep["field_fractions"][k])
    abs_m,  abs_s  = _get_scalar_trace(lambda ep, k=key: ep["field_fractions"][f"{k}_abs"])
    _shade(ax1, all_epochs, frac_m, frac_s, col, label=label)
    _shade(ax2, all_epochs, abs_m,  abs_s,  col, label=f"{label} |h|")

ax1.set_xlabel("Epoch"); ax1.set_ylabel("Fractional |h| contribution")
ax1.set_title("J1 field contributions (fractional)")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(all_epochs)

ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean |h| (absolute)")
ax2.set_title("J1 field contributions (absolute)")
ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xticks(all_epochs)

fig.suptitle(f"σ={TARGET_SIGMA}  mean ± 1 std over {N_SEEDS} seeds — J1 field contributions", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "sigma03_field_fractions.png", dpi=150)
plt.close(fig)
print("saved sigma03_field_fractions.png")

print("\nAll done.")
