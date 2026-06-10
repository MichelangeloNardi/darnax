"""Generate figures from standard_diagnostics.json.

Trajectory plots (autocorr heatmaps, within-epoch ABCD) use seed 0 only.
Scalar curves (weight norms, field fractions) show mean over all seeds without
error bands — diagnostics are about shape, not significance.
"""

from __future__ import annotations

import json
import pathlib
import warnings

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
n_seeds   = len(data["per_seed"])
n_epochs  = data["epochs"]


# ── helpers ─────────────────────────────────────────────────────────────────

def get_per_epoch_mean(key_path):
    """(n_epochs,) mean over seeds for a dotted key path into per_epoch_diag."""
    keys = key_path.split(".")
    mat = np.zeros((n_seeds, n_epochs))
    for si, seed_rec in enumerate(data["per_seed"]):
        for ep_rec in seed_rec["per_epoch_diag"]:
            ei = ep_rec["epoch"] - 1
            val = ep_rec
            for k in keys:
                val = val[k]
            mat[si, ei] = float(val)
    return mat.mean(0)


def get_seed0_epoch(epoch):
    return data["per_seed"][0]["per_epoch_diag"][epoch - 1]


# ── 1. autocorr heatmaps (epochs 1 / mid / last, seed 0) ─────────────────────
# Drop the "init" row/col (all-zero state, uninformative).
# Use vmin from actual data minimum so the colour range is informative.

show_epochs = sorted({1, n_epochs // 2 or 1, n_epochs})
show_epochs = sorted(set(show_epochs))

fig, axes = plt.subplots(1, len(show_epochs), figsize=(5 * len(show_epochs), 4.5))
if len(show_epochs) == 1:
    axes = [axes]
for ax, ep in zip(axes, show_epochs):
    ep_rec  = get_seed0_epoch(ep)
    sim_all = np.array(ep_rec["autocorr_matrix"])
    labels_all = ep_rec["autocorr_labels"]
    # drop "init" (index 0)
    idx    = [i for i, l in enumerate(labels_all) if l != "init"]
    sim    = sim_all[np.ix_(idx, idx)]
    labels = [labels_all[i] for i in idx]
    vmin   = float(np.percentile(sim[~np.eye(len(sim), dtype=bool)], 2))
    im = ax.imshow(sim, vmin=vmin, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    # phase boundary lines: after W steps and after C steps
    n_w = sum(1 for l in labels if l.startswith("W"))
    n_c = sum(1 for l in labels if l.startswith("C"))
    for boundary in [n_w - 0.5, n_w + n_c - 0.5]:
        ax.axhline(boundary, color="white", lw=1.2)
        ax.axvline(boundary, color="white", lw=1.2)
    ax.set_title(f"Epoch {ep}")
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("J1 step-to-step cosine similarity — seed 0  (init excluded)", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES / "diag_autocorr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved diag_autocorr.png")


# ── 2. weight norm curves (mean over seeds, no band) ─────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, get_per_epoch_mean("weight_norms.win"),  color="#16A34A", linewidth=2, label="Win  ||W_in||_F")
ax.plot(epochs, get_per_epoch_mean("weight_norms.j1"),   color="#2563EB", linewidth=2, label="J1   ||J||_F")
ax.plot(epochs, get_per_epoch_mean("weight_norms.wback"), color="#DC2626", linewidth=2, label="WBack (frozen)", linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Frobenius norm")
ax.set_title(f"Weight norms over training (mean of {n_seeds} seeds)")
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(FIGURES / "diag_weight_norms.png", dpi=150)
plt.close(fig)
print("saved diag_weight_norms.png")


# ── 3. field contribution fractions (mean over seeds) ─────────────────────────

win_frac  = get_per_epoch_mean("field_fractions.win")
j1_frac   = get_per_epoch_mean("field_fractions.j1")
wb_frac   = get_per_epoch_mean("field_fractions.wback")
win_abs   = get_per_epoch_mean("field_fractions.win_abs")
j1_abs    = get_per_epoch_mean("field_fractions.j1_abs")
wb_abs    = get_per_epoch_mean("field_fractions.wback_abs")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.stackplot(epochs, win_frac, j1_frac, wb_frac,
             labels=["Win", "J1", "WBack"],
             colors=["#16A34A", "#2563EB", "#DC2626"], alpha=0.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Fractional |h| contribution")
ax.set_title(f"Field contributions (mean of {n_seeds} seeds)")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax = axes[1]
ax.plot(epochs, win_abs, color="#16A34A", linewidth=2, label="Win  mean|h^Win|")
ax.plot(epochs, j1_abs,  color="#2563EB", linewidth=2, label="J1   mean|h^J|")
ax.plot(epochs, wb_abs,  color="#DC2626", linewidth=2, label="WBack mean|h^B|", linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |field|")
ax.set_title(f"Absolute field magnitudes (mean of {n_seeds} seeds)")
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig.tight_layout()
fig.savefig(FIGURES / "diag_field_contributions.png", dpi=150)
plt.close(fig)
print("saved diag_field_contributions.png")


# ── 4. ABCD 8×8 overlap grid — before update (ABCD) vs after update (A'B'C'D') ──
# Each cell (i,j) = mean per-image cosine similarity.
# Top-left 4×4 block: within-ABCD; bottom-right: within-A'B'C'D';
# off-diagonal blocks: how much the fixed points shift after one training step.
# Grey cells appear when "abcd_8x8" key is missing (old data without orch_after).

show_epochs = sorted({1, n_epochs // 2 or 1, n_epochs})
show_epochs = sorted(set(show_epochs))

def get_8x8(ep):
    """Return (8×8 matrix, labels) averaged over seeds; NaN for missing data."""
    mats = []
    labels_ref = None
    for seed_rec in data["per_seed"]:
        ep_rec = seed_rec["per_epoch_diag"][ep - 1]
        if "abcd_8x8" in ep_rec:
            mats.append(np.array(ep_rec["abcd_8x8"]["matrix"]))
            labels_ref = ep_rec["abcd_8x8"]["labels"]
    if not mats:
        return np.full((8, 8), np.nan), ["A","B","C","D","A'","B'","C'","D'"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(mats, axis=0), labels_ref

epoch_8x8 = {ep: get_8x8(ep) for ep in show_epochs}

# Colour range from available off-diagonal values
all_vals_8x8 = np.concatenate([
    m[~np.eye(8, dtype=bool) & ~np.isnan(m)].ravel()
    for m, _ in epoch_8x8.values()
    if not np.all(np.isnan(m))
]) if any(not np.all(np.isnan(m)) for m, _ in epoch_8x8.values()) else np.array([0.85, 1.0])
vmin_8x8 = float(np.percentile(all_vals_8x8, 2)) if len(all_vals_8x8) else 0.85

cmap_8x8 = plt.cm.viridis.copy()
cmap_8x8.set_bad(color="#cccccc")

fig, axes = plt.subplots(1, len(show_epochs), figsize=(5.5 * len(show_epochs), 5.2))
if len(show_epochs) == 1:
    axes = [axes]

for ax, ep in zip(axes, show_epochs):
    mat, lbl = epoch_8x8[ep]
    im = ax.imshow(np.ma.masked_invalid(mat), vmin=vmin_8x8, vmax=1.0, cmap=cmap_8x8)
    ax.set_xticks(range(8)); ax.set_xticklabels(lbl, fontsize=9)
    ax.set_yticks(range(8)); ax.set_yticklabels(lbl, fontsize=9)
    # block separator between before/after halves
    ax.axhline(3.5, color="white", lw=2)
    ax.axvline(3.5, color="white", lw=2)
    # annotate non-NaN cells
    for i in range(8):
        for j in range(8):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v < (vmin_8x8 + 1.0) / 2 else "black")
    ax.set_title(f"Epoch {ep}")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(
    f"ABCD × A'B'C'D' cosine similarity (mean of {n_seeds} seeds)\n"
    "Left/top = before update   Right/bottom = after one training step   grey = not recorded",
    y=1.02,
)
fig.tight_layout()
fig.savefig(FIGURES / "diag_abcd.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved diag_abcd.png")


# ── 5. within-epoch ABCD evolution — seed 0 only ─────────────────────────────

seed0_within = data["per_seed"][0]["within_epoch_abcd"]
if seed0_within:
    fig, ax = plt.subplots(figsize=(9, 4))
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
    ax.set_title("Within-epoch ABCD evolution — seed 0 (one line per epoch)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "diag_within_epoch_abcd.png", dpi=150)
    plt.close(fig)
    print("saved diag_within_epoch_abcd.png")
else:
    print("skipped diag_within_epoch_abcd.png (no within-epoch data)")
