"""experiments4/meeting_figures.py

Re-render existing v2 results into two minimal meeting-ready figures, using
the dot-product overlap (= 2·match − 1) instead of the sign-match rate.

Inputs (already on disk):
  experiments4/results/fixedpoint_overlap_v2.json
  experiments4/results/per_image_trajectory_v2.json

Outputs:
  experiments4/figures/meeting_exp1.png   — saturation of the 16-point geometry
  experiments4/figures/meeting_exp2.png   — per-image trajectories + drift

No training, no JAX, no GPU — just numpy + matplotlib on cached results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"


def match_to_dot(m):
    """Binary {-1, +1} states: dot = 2·match − 1."""
    return 2.0 * np.asarray(m) - 1.0


def fig_exp1():
    """Single panel: saturation of key dot-product overlaps vs training batch."""
    with open(RESULTS / "fixedpoint_overlap_v2.json") as f:
        d = json.load(f)
    cps = sorted(int(k) for k in d.keys())
    # match arrays per checkpoint
    M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}

    # Labels in stored order: A_none, B_none, C_none, D_none, A_win, B_win, ...
    # Within-baseline-block indices: A=0, B=1, C=2, D=3
    keys = [
        ("A↔B (warmup → clamped)", 0, 1),
        ("A↔C (warmup → clamped → free)", 0, 2),
        ("A↔D (warmup → free, inference)", 0, 3),
        ("B↔C (clamped → free)", 1, 2),
        ("B↔D (clamped vs inference)", 1, 3),
        ("C↔D (train-free vs inference)", 2, 3),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for (label, i, j), col in zip(keys, colors):
        ys = [match_to_dot(M[cp][i, j]) for cp in cps]
        ax.plot(cps, ys, "-o", label=label, color=col, lw=1.8, markersize=5)

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("training batch")
    ax.set_ylabel("dot-product overlap  ⟨s_i · s_j⟩")
    ax.set_ylim(0.85, 1.005)
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_title("J1 dynamics overlap saturates by ~batch 100  (Config 3)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = FIGS / "meeting_exp1.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_exp2():
    """2×2 panel: C-D dot, J1 stability, soft margin, weight drift."""
    with open(RESULTS / "per_image_trajectory_v2.json") as f:
        d = json.load(f)

    aligned = np.array(d["aligned_batches"])
    cd_match = np.array(d["overlap_C_D"])            # (probes, n_meas)
    stab_C   = np.array(d["stability_C"])            # (probes, n_meas)
    margin_D = np.array(d["margin_D"])               # (probes, n_meas)
    batches  = np.array(d["measure_batches"])
    wout_drift = np.array(d["wout_l2_drift"])
    j_drift    = np.array(d["j1_l2_drift"])
    win_drift  = np.array(d["win_l2_drift"])

    cd_dot   = match_to_dot(cd_match)
    stab_dot = match_to_dot(stab_C)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

    # (a) per-probe C-D dot product trajectory
    ax = axes[0, 0]
    for pi in range(cd_dot.shape[0]):
        mask = aligned[pi] >= 0
        if mask.any():
            ax.plot(aligned[pi][mask], cd_dot[pi][mask], color="tab:gray",
                    alpha=0.25, lw=0.7)
    ax.plot([], [], color="tab:gray", alpha=0.5, label="per probe (n=50)")
    # mean
    valid = aligned >= 0
    if valid.any():
        # crude mean along checkpoints
        xs = np.unique(aligned[valid])
        means = []
        for x in xs:
            mask = aligned == x
            if mask.any():
                means.append(np.nanmean(cd_dot[mask]))
            else:
                means.append(np.nan)
        ax.plot(xs, means, "k-", lw=2, label="mean")
    ax.set_title("Same-checkpoint C↔D overlap (train vs inference state)")
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("dot-product ⟨s_C · s_D⟩")
    ax.set_ylim(0.9, 1.005)
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) cross-time J1 stability per probe (vs post-first-seen baseline)
    ax = axes[0, 1]
    for pi in range(stab_dot.shape[0]):
        mask = aligned[pi] >= 0
        if mask.any():
            ax.plot(aligned[pi][mask], stab_dot[pi][mask], color="tab:gray",
                    alpha=0.25, lw=0.7)
    ax.plot([], [], color="tab:gray", alpha=0.5, label="per probe (n=50)")
    # mean curve
    if valid.any():
        xs = np.unique(aligned[valid])
        means = []
        for x in xs:
            mask = aligned == x
            if mask.any():
                means.append(np.nanmean(stab_dot[mask]))
            else:
                means.append(np.nan)
        ax.plot(xs, means, "k-", lw=2, label="mean")
    ax.set_title("J1_C(probe, t) vs J1_C(probe, baseline)  (cross-time)")
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("dot-product ⟨s_t · s_baseline⟩")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.axhline(0.0, color="k", lw=0.4, ls=":")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (c) soft margin from D (inference)
    ax = axes[1, 0]
    for pi in range(margin_D.shape[0]):
        mask = aligned[pi] >= 0
        if mask.any():
            ax.plot(aligned[pi][mask], margin_D[pi][mask], color="tab:gray",
                    alpha=0.25, lw=0.7)
    ax.plot([], [], color="tab:gray", alpha=0.5, label="per probe (n=50)")
    if valid.any():
        xs = np.unique(aligned[valid])
        means = []
        for x in xs:
            mask = aligned == x
            if mask.any():
                means.append(np.nanmean(margin_D[mask]))
            else:
                means.append(np.nan)
        ax.plot(xs, means, "k-", lw=2, label="mean")
    ax.set_title("Soft margin from inference state (= correct − max wrong)")
    ax.set_xlabel("batches since first seen")
    ax.set_ylabel("margin")
    ax.axhline(0.0, color="k", lw=0.4, ls=":")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # (d) weight drift from init
    ax = axes[1, 1]
    ax.plot(batches, wout_drift, "-o", color="tab:red",
            markersize=3, label=r"$\|W_{out}(t)-W_{out}(0)\|$")
    ax.plot(batches, j_drift, "-s", color="tab:blue",
            markersize=3, label=r"$\|J(t)-J(0)\|$")
    ax.plot(batches, win_drift, "-^", color="tab:green",
            markersize=3, label=r"$\|W_{in}(t)-W_{in}(0)\|$")
    ax.set_title("Weight matrices' L2 drift from initialization")
    ax.set_xlabel("training batch")
    ax.set_ylabel("L2 distance from init")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Per-image trajectories (Config 3, entropy rule, 50 probes × 2 epochs)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGS / "meeting_exp2.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fig_exp1()
    fig_exp2()
