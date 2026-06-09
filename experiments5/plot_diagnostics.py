"""Generate figures from standard_diagnostics.json."""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS = pathlib.Path(__file__).parent / "results"
FIGURES = pathlib.Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

with open(RESULTS / "standard_diagnostics.json") as f:
    data = json.load(f)

seeds     = data["seeds"]
epochs    = list(range(1, data["epochs"] + 1))
warmup_n  = data["warmup_n"]
clamped_n = data["clamped_n"]
free_n    = data["free_n"]
n_seeds   = len(seeds)
n_epochs  = data["epochs"]


# ── helpers ─────────────────────────────────────────────────────────────────

def get_per_epoch(key_path):
    """Return (n_seeds, n_epochs) array for a dotted key path into per_epoch_diag."""
    keys = key_path.split(".")
    out = np.zeros((n_seeds, n_epochs))
    for si, seed_rec in enumerate(data["per_seed"]):
        for ep_rec in seed_rec["per_epoch_diag"]:
            ei = ep_rec["epoch"] - 1
            val = ep_rec
            for k in keys:
                val = val[k]
            out[si, ei] = float(val)
    return out


def plot_band(ax, xs, mat, color, label, ls="-"):
    m, s = mat.mean(0), mat.std(0)
    ax.plot(xs, m, color=color, label=label, linewidth=2, linestyle=ls)
    ax.fill_between(xs, m - s, m + s, alpha=0.15, color=color)


# ── 1. autocorr heatmaps (epoch 1 / 5 / 10, seed 0) ─────────────────────────

def get_autocorr(seed_idx, epoch):
    ep_rec = data["per_seed"][seed_idx]["per_epoch_diag"][epoch - 1]
    return np.array(ep_rec["autocorr_matrix"]), ep_rec["autocorr_labels"]

show_epochs = [1, 5, 10]
show_epochs = [e for e in show_epochs if e <= n_epochs]

fig, axes = plt.subplots(1, len(show_epochs), figsize=(5 * len(show_epochs), 4.5))
if len(show_epochs) == 1:
    axes = [axes]
for ax, ep in zip(axes, show_epochs):
    sim, labels = get_autocorr(seed_idx=0, epoch=ep)
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Epoch {ep}")
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("J1 autocorrelation (pairwise cosine sim) — seed 0")
fig.tight_layout()
fig.savefig(FIGURES / "diag_autocorr.png", dpi=150); plt.close(fig)
print("saved diag_autocorr.png")


# ── 2. weight norm curves ────────────────────────────────────────────────────

win_norms  = get_per_epoch("weight_norms.win")
j1_norms   = get_per_epoch("weight_norms.j1")
wback_norms = get_per_epoch("weight_norms.wback")

fig, ax = plt.subplots(figsize=(7, 4))
plot_band(ax, epochs, win_norms,   "#16A34A", "Win  (||W_in||_F)")
plot_band(ax, epochs, j1_norms,    "#2563EB", "J1   (||J||_F)")
plot_band(ax, epochs, wback_norms, "#DC2626", "WBack (frozen, constant)",  "--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Frobenius norm")
ax.set_title(f"Weight norms over training ({n_seeds} seeds ± 1 std)")
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(FIGURES / "diag_weight_norms.png", dpi=150); plt.close(fig)
print("saved diag_weight_norms.png")


# ── 3. field contribution fractions (area chart, mean over seeds) ─────────────

win_frac  = get_per_epoch("field_fractions.win")
j1_frac   = get_per_epoch("field_fractions.j1")
wb_frac   = get_per_epoch("field_fractions.wback")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: fractions (stacked area, mean over seeds)
ax = axes[0]
win_m  = win_frac.mean(0)
j1_m   = j1_frac.mean(0)
wb_m   = wb_frac.mean(0)
ax.stackplot(epochs, win_m, j1_m, wb_m,
             labels=["Win", "J1", "WBack"],
             colors=["#16A34A", "#2563EB", "#DC2626"],
             alpha=0.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Fractional |h| contribution")
ax.set_title("Field contributions (mean over seeds)")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Right: absolute magnitudes with bands
ax = axes[1]
win_abs  = get_per_epoch("field_fractions.win_abs")
j1_abs   = get_per_epoch("field_fractions.j1_abs")
wb_abs   = get_per_epoch("field_fractions.wback_abs")
plot_band(ax, epochs, win_abs,  "#16A34A", "Win ⟨|h^Win|⟩")
plot_band(ax, epochs, j1_abs,   "#2563EB", "J1  ⟨|h^J|⟩")
plot_band(ax, epochs, wb_abs,   "#DC2626", "WBack ⟨|h^B|⟩", "--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |field|")
ax.set_title(f"Absolute field magnitudes ({n_seeds} seeds ± 1 std)")
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig.tight_layout()
fig.savefig(FIGURES / "diag_field_contributions.png", dpi=150); plt.close(fig)
print("saved diag_field_contributions.png")


# ── 4. ABCD overlap curve (per epoch) ────────────────────────────────────────

cd_mean = get_per_epoch("abcd.cd_sim_mean")
bc_mean = get_per_epoch("abcd.bc_sim_mean")

fig, ax = plt.subplots(figsize=(7, 4))
plot_band(ax, epochs, cd_mean, "#9333EA", "C-D cosine sim (attractor autonomy)")
plot_band(ax, epochs, bc_mean, "#EA580C", "B-C cosine sim (clamped→free convergence)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean per-image cosine similarity")
ax.set_title(f"ABCD attractor similarity over training ({n_seeds} seeds ± 1 std)")
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.1, 1.05)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(FIGURES / "diag_abcd.png", dpi=150); plt.close(fig)
print("saved diag_abcd.png")


# ── 5. Within-epoch ABCD C-D similarity (seed 0) ────────────────────────────

seed0_within = data["per_seed"][0]["within_epoch_abcd"]
if seed0_within:
    fig, ax = plt.subplots(figsize=(9, 4))
    # Group by epoch and plot each as a separate line segment
    by_epoch: dict[int, list] = {}
    for rec in seed0_within:
        by_epoch.setdefault(rec["epoch"], []).append(rec)

    cmap = plt.cm.viridis
    ep_vals = sorted(by_epoch)
    for ep in ep_vals:
        recs = sorted(by_epoch[ep], key=lambda r: r["batch_idx"])
        xs = [r["batch_idx"] for r in recs]
        ys = [r["cd_sim_mean"] for r in recs]
        color = cmap((ep - 1) / max(len(ep_vals) - 1, 1))
        ax.plot(xs, ys, "-o", color=color, label=f"ep {ep}", markersize=4)

    ax.set_xlabel("Batch index within epoch")
    ax.set_ylabel("C-D cosine similarity")
    ax.set_title("Within-epoch ABCD evolution — seed 0 (each line = one epoch)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "diag_within_epoch_abcd.png", dpi=150); plt.close(fig)
    print("saved diag_within_epoch_abcd.png")
else:
    print("skipped diag_within_epoch_abcd.png (no within-epoch data)")
