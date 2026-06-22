"""5-Matei_Random_W_back/plot_autocorr.py

Step-to-step autocorrelation plots for the noise sweep.
Data is already in results/diagnostics.json — this just visualises it.

Figures saved to figures/:
  - autocorr_final_epoch.png   : step-to-step bars at epoch 10, one panel per noise level
  - autocorr_C1_over_epochs.png: how C1 step-to-step sim evolves during training per noise
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

with open(RESULTS / "diagnostics.json") as f:
    d = json.load(f)

NOISE_VALUES = d["noise_values"]
EPOCHS       = d["epochs"]
warmup_n     = d["warmup_n"]
clamped_n    = d["clamped_n"]
free_n       = d["free_n"]
N_SEEDS      = len(d["per_noise"][0]["per_seed"])

labels_full = d["per_noise"][0]["per_seed"][0]["per_epoch_diag"][0]["autocorr_labels"]
# step-to-step: pair (labels[i-1], labels[i]) for i in 1..n
step_labels = [f"{labels_full[i-1]}→{labels_full[i]}" for i in range(1, len(labels_full))]
n_steps = len(step_labels)

# phase boundary positions (for vertical red lines)
# warmup ends after index warmup_n-1 in step_labels (0-based)
# clamped ends after warmup_n + clamped_n - 1
wb_boundary = warmup_n - 0.5          # between W1 and C1
cb_boundary = warmup_n + clamped_n - 0.5  # between C11 and F1

colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(NOISE_VALUES)))


# ── Figure 1: step-to-step bars at final epoch, one panel per noise ───────────

fig, axes = plt.subplots(1, len(NOISE_VALUES), figsize=(5.5 * len(NOISE_VALUES), 4.5),
                         sharey=True)

for ax, noise_entry, col in zip(axes, d["per_noise"], colors):
    ns = noise_entry["noise_std"]
    # gather step-to-step sims across seeds at final epoch
    step_mat = np.array([
        [np.array(sd["per_epoch_diag"][-1]["autocorr_matrix"])[i, i - 1]
         for i in range(1, len(labels_full))]
        for sd in noise_entry["per_seed"]
    ])  # (n_seeds, n_steps)
    sm, ss = step_mat.mean(0), step_mat.std(0)

    ax.bar(range(n_steps), sm, color=col, alpha=0.78)
    ax.errorbar(range(n_steps), sm, yerr=ss, fmt="none",
                color="black", capsize=2.5, linewidth=0.9)
    ax.axvline(wb_boundary, color="#DC2626", lw=1.5, linestyle="--", alpha=0.85)
    ax.axvline(cb_boundary, color="#DC2626", lw=1.5, linestyle="--", alpha=0.85)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(step_labels, fontsize=5.5, rotation=90)
    ax.set_ylim(0.80, 1.005)
    ax.set_title(f"σ={ns:.1f}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Step-to-step cosine sim")
fig.suptitle(
    f"Step-to-step J1 similarity — Matei config + W_back noise  (epoch {EPOCHS}, mean±std {N_SEEDS} seeds)\n"
    f"warmup={warmup_n}  clamped={clamped_n}  free={free_n}   red dashed = phase boundaries",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(FIGURES / "autocorr_final_epoch.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved autocorr_final_epoch.png")


# ── Figure 2: C1 step-to-step sim over training epochs, per noise value ───────
# C1 is the first clamped step (index warmup_n in step_labels, i.e. row warmup_n+1 in matrix)
C1_row = warmup_n + 1   # row index in the full autocorr matrix for C1
F1_row = warmup_n + clamped_n + 1   # F1

epochs_x = list(range(1, EPOCHS + 1))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_C1, ax_F1 = axes

for noise_entry, col in zip(d["per_noise"], colors):
    ns = noise_entry["noise_std"]
    # C1 step-to-step: autocorr_matrix[C1_row][C1_row - 1]
    c1_mat = np.array([
        [np.array(sd["per_epoch_diag"][ep]["autocorr_matrix"])[C1_row][C1_row - 1]
         for ep in range(EPOCHS)]
        for sd in noise_entry["per_seed"]
    ])
    f1_mat = np.array([
        [np.array(sd["per_epoch_diag"][ep]["autocorr_matrix"])[F1_row][F1_row - 1]
         for ep in range(EPOCHS)]
        for sd in noise_entry["per_seed"]
    ])
    lbl = f"σ={ns:.1f}"
    for ax, mat in [(ax_C1, c1_mat), (ax_F1, f1_mat)]:
        m, s = mat.mean(0), mat.std(0)
        ax.plot(epochs_x, m, "-o", color=col, label=lbl, markersize=4, linewidth=1.8)
        ax.fill_between(epochs_x, m - s, m + s, alpha=0.14, color=col)

for ax, title, step in [
    (ax_C1, "C1 (first clamped step) sim", "W1→C1"),
    (ax_F1, "F1 (first free step) sim",    "C11→F1"),
]:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Step-to-step cosine sim")
    ax.set_title(f"{title}\n({step})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.80, 1.005)
    ax.set_xticks(epochs_x)

fig.suptitle(
    f"Key step-to-step sims over training — Matei config + W_back noise  ({N_SEEDS} seeds ±1 std)\n"
    f"A value < 1 means the dynamics take more than 1 step to converge (non-trivial)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(FIGURES / "autocorr_C1_F1_over_epochs.png", dpi=150)
plt.close(fig)
print("saved autocorr_C1_F1_over_epochs.png")

print("\nAll done.")
