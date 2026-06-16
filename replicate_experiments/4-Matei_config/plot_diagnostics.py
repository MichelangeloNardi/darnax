"""4-Matei_config/plot_diagnostics.py

Diagnostic plots for the Matei config run (matei_W_out_cfg.json).
Mean ± std across 3 seeds, selected epochs: 1, 3, 5, 7, 10.

Saves to replicate_experiments/4-Matei_config/figures/.
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

SELECTED_EPOCHS = [1, 5, 10, 15, 20]
SELECTED_IDX    = [e - 1 for e in SELECTED_EPOCHS]

# ── load ──────────────────────────────────────────────────────────────────────

with open(RESULTS / "diagnostics.json") as f:
    d = json.load(f)

warmup_n  = d["warmup_n"]
clamped_n = d["clamped_n"]
free_n    = d["free_n"]

seeds_data = d["per_seed"]
N_SEEDS    = len(seeds_data)
SEEDS      = [s["seed"] for s in seeds_data]
EPOCHS     = len(seeds_data[0]["per_epoch_diag"])
all_epochs = list(range(1, EPOCHS + 1))

print(f"Seeds={SEEDS}  epochs={EPOCHS}  warmup={warmup_n}  clamped={clamped_n}  free={free_n}")
print(f"Autocorr steps: {1 + warmup_n + clamped_n + free_n}")

ABCD_PAIRS  = [("a","b"),("a","c"),("a","d"),("b","c"),("b","d"),("c","d")]
PAIR_LABELS = ["A-B","A-C","A-D","B-C","B-D","C-D"]
PAIR_COLORS = ["#2563EB","#7C3AED","#DB2777","#EA580C","#16A34A","#DC2626"]


def _trace(key_fn):
    mat = np.array([[key_fn(ep) for ep in sd["per_epoch_diag"]] for sd in seeds_data])
    return mat.mean(0), mat.std(0)


def _shade(ax, epochs, mean, std, color, label=None, lw=1.8, ls="-", marker="o", ms=4):
    ax.plot(epochs, mean, ls + marker, color=color, linewidth=lw, markersize=ms, label=label)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.18, color=color)


# ── Figure 1: ABCD 8×8 grids at 5 epochs (mean over seeds) ──────────────────

labels_8x8 = ["A","B","C","D","A'","B'","C'","D'"]
all_off = []
for ep_idx in SELECTED_IDX:
    for sd in seeds_data:
        m = np.array(sd["per_epoch_diag"][ep_idx]["abcd_8x8"]["matrix"])
        all_off.extend(m[~np.eye(8, dtype=bool)].tolist())
vmin = float(np.percentile(all_off, 2))
cmap = plt.cm.viridis

fig, axes = plt.subplots(1, len(SELECTED_EPOCHS), figsize=(5.5 * len(SELECTED_EPOCHS), 5.5))
for ax, ep_idx, ep in zip(axes, SELECTED_IDX, SELECTED_EPOCHS):
    m = np.array([sd["per_epoch_diag"][ep_idx]["abcd_8x8"]["matrix"] for sd in seeds_data]).mean(0)
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
    f"ABCD × A'B'C'D' cosine similarity — Matei config  mean {N_SEEDS} seeds\n"
    "(rows=before update, cols=after update; white line = before/after boundary)",
    fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "abcd_8x8.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved abcd_8x8.png")


# ── Figure 2: Autocorrelation matrices at 5 epochs (mean, drop init) ─────────

labels_full = seeds_data[0]["per_epoch_diag"][0]["autocorr_labels"]
drop_init   = [i for i, lb in enumerate(labels_full) if lb != "init"]
labels_ac   = [labels_full[i] for i in drop_init]
n_w = sum(1 for lb in labels_ac if lb.startswith("W"))
n_c = sum(1 for lb in labels_ac if lb.startswith("C"))

fig, axes = plt.subplots(1, len(SELECTED_EPOCHS), figsize=(5.5 * len(SELECTED_EPOCHS), 5.5))
for ax, ep_idx, ep in zip(axes, SELECTED_IDX, SELECTED_EPOCHS):
    sims = np.array([
        np.array(sd["per_epoch_diag"][ep_idx]["autocorr_matrix"])[np.ix_(drop_init, drop_init)]
        for sd in seeds_data
    ]).mean(0)
    off = sims[~np.eye(len(sims), dtype=bool)]
    im = ax.imshow(sims, vmin=float(np.percentile(off, 2)), vmax=1.0, cmap="viridis")
    tick_step = 2
    ticks = list(range(0, len(labels_ac), tick_step))
    ax.set_xticks(ticks); ax.set_xticklabels([labels_ac[i] for i in ticks], fontsize=7, rotation=90)
    ax.set_yticks(ticks); ax.set_yticklabels([labels_ac[i] for i in ticks], fontsize=7)
    for b in [n_w - 0.5, n_w + n_c - 0.5]:
        ax.axhline(b, color="white", lw=1.2); ax.axvline(b, color="white", lw=1.2)
    ax.set_title(f"Epoch {ep}", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(
    f"J1 autocorrelation — Matei config  mean {N_SEEDS} seeds  (init excluded)\n"
    f"W=warmup (1 step), C=clamped ({clamped_n} steps, label on), F=free ({free_n} steps, label off)",
    fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "autocorr.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved autocorr.png")


# ── Figure 3: Step-to-step sim bars at 5 epochs (mean ± std) ────────────────

fig, axes = plt.subplots(1, len(SELECTED_EPOCHS), figsize=(5.5 * len(SELECTED_EPOCHS), 4.5))
for ax, ep_idx, ep in zip(axes, SELECTED_IDX, SELECTED_EPOCHS):
    step_mat = np.array([
        [np.array(sd["per_epoch_diag"][ep_idx]["autocorr_matrix"])[i, i-1]
         for i in range(1, len(labels_full))]
        for sd in seeds_data
    ])
    sm, ss = step_mat.mean(0), step_mat.std(0)
    x = range(len(sm))
    ax.bar(x, sm, color="#2563EB", alpha=0.75)
    ax.errorbar(x, sm, yerr=ss, fmt="none", color="#1e3a8a", capsize=3, linewidth=1.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_full[1:], fontsize=6, rotation=90)
    for b in [warmup_n - 0.5, warmup_n + clamped_n - 0.5]:
        ax.axvline(b, color="#DC2626", lw=1.5, linestyle="--", alpha=0.8)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Cosine sim with prev step")
    ax.set_title(f"Epoch {ep}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle(
    f"Step-to-step J1 similarity — Matei config  mean ± std {N_SEEDS} seeds\n"
    "Red dashed = phase boundary (W|C, C|F)",
    fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "step_to_step.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved step_to_step.png")


# ── Figure 4: ABCD pairwise sims over epochs (all 6 pairs, mean ± std) ───────

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, pair, lbl, col in zip(axes.ravel(), ABCD_PAIRS, PAIR_LABELS, PAIR_COLORS):
    p, q = pair
    mean, std = _trace(lambda ep, p=p, q=q: ep["abcd"][f"{p}{q}_sim_mean"])
    _shade(ax, all_epochs, mean, std, col, label=lbl)
    for ep in SELECTED_EPOCHS:
        ax.axvline(ep, color="gray", lw=0.7, linestyle="--", alpha=0.5)
    ax.set_title(lbl, fontsize=12, color=col)
    ax.set_ylim(0.85, 1.01)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cosine similarity")
    ax.grid(alpha=0.3); ax.set_xticks(all_epochs)
fig.suptitle(
    f"ABCD pairwise sims — Matei config  mean ± 1 std  {N_SEEDS} seeds\n"
    "Gray dashed = epochs shown in matrix plots",
    fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "abcd_pairs.png", dpi=150)
plt.close(fig)
print("saved abcd_pairs.png")


# ── Figure 5: All pairs overlaid + weight norms ───────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for pair, lbl, col in zip(ABCD_PAIRS, PAIR_LABELS, PAIR_COLORS):
    p, q = pair
    mean, std = _trace(lambda ep, p=p, q=q: ep["abcd"][f"{p}{q}_sim_mean"])
    lw = 2.5 if pair == ("c","d") else 1.2
    ls = "-" if pair == ("c","d") else "--"
    _shade(ax1, all_epochs, mean, std, col, label=lbl, lw=lw, ls=ls)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cosine similarity")
ax1.set_title("All ABCD pairs (C-D bold)")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.set_xticks(all_epochs); ax1.set_ylim(0.85, 1.01)

for key, label, col, ls in [("win","Win ‖W‖_F","#16A34A","-"),
                              ("j1","J1 ‖J‖_F","#2563EB","-"),
                              ("wback","WBack ‖W‖_F (frozen)","#DC2626","--")]:
    mean, std = _trace(lambda ep, k=key: ep["weight_norms"][k])
    _shade(ax2, all_epochs, mean, std, col, label=label, ls=ls)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Frobenius norm")
ax2.set_title("Weight norms over training")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_xticks(all_epochs)

fig.suptitle(f"Matei config — mean ± 1 std  {N_SEEDS} seeds", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "dynamics.png", dpi=150)
plt.close(fig)
print("saved dynamics.png")


# ── Figure 6: Field contributions (fractional + absolute) ────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for key, label, col in [("win","Win","#16A34A"),("j1","J1","#2563EB"),("wback","WBack","#DC2626")]:
    fm, fs = _trace(lambda ep, k=key: ep["field_fractions"][k])
    am, as_ = _trace(lambda ep, k=key: ep["field_fractions"][f"{k}_abs"])
    _shade(ax1, all_epochs, fm, fs, col, label=label)
    _shade(ax2, all_epochs, am, as_, col, label=f"{label} |h|")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Fractional |h| contribution")
ax1.set_title("J1 field contributions (fractional)")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xticks(all_epochs)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean |h| (absolute)")
ax2.set_title("J1 field contributions (absolute)")
ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xticks(all_epochs)
fig.suptitle(f"Matei config — mean ± 1 std  {N_SEEDS} seeds", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "field_fractions.png", dpi=150)
plt.close(fig)
print("saved field_fractions.png")


print("\nAll done.")
