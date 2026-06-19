"""experiments4/build_meeting_pack.py

Re-render all experimental results into a clean meeting/ folder.

Inputs (already on disk):
  results/fixedpoint_overlap_v2.json   — long-dynamics 16-point matrix
  results/fixedpoint_overlap_v3.json   — short-dynamics 16-point matrix
  results/per_image_trajectory_v2.json — per-probe trajectories
  results/cross_time_cd.json           — explicit cross-time C↔D' measurement
  results/dynamics_diagnostics.json    — (B) decomposition + (C) autocorrelation

Outputs (figures only, all dot-product):
  meeting/exp1_geometry.png         — 6-line saturation curve (long dyn)
  meeting/exp1_ablation_shifts.png  — how each update type shifts A/B/C/D
  meeting/exp1_matrix_batch0.png    — small batch-0 heatmap, dot product only
  meeting/exp2_stability_cohort.png — per-cohort J1 stability + cross-protocol
  meeting/exp2_margin_dynamics.png  — soft-margin evolution per class + final bars
  meeting/diag_field_decomposition.png  — h_i = h_W_in + h_J + h_W_back
  meeting/diag_autocorrelation.png      — within-phase step-to-step overlap
  meeting/diag_h_distribution.png       — pre-sign field histogram
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MEET = HERE / "meeting"
MEET.mkdir(exist_ok=True)


def match_to_dot(m):
    return 2.0 * np.asarray(m) - 1.0


# ----- Experiment 1 -----------------------------------------------------------

def exp1_geometry():
    """Six within-phase overlaps over training, dot product."""
    with open(RES / "fixedpoint_overlap_v2.json") as f:
        d = json.load(f)
    cps = sorted(int(k) for k in d.keys())
    M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}

    keys = [
        ("A↔B", 0, 1),
        ("A↔C", 0, 2),
        ("A↔D", 0, 3),
        ("B↔C", 1, 2),
        ("B↔D", 1, 3),
        ("C↔D", 2, 3),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, i, j), col in zip(keys, colors):
        ys = [match_to_dot(M[cp][i, j]) for cp in cps]
        ax.plot(cps, ys, "-o", label=label, color=col, lw=2.0, markersize=6)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("training batch")
    ax.set_ylabel("dot-product overlap  ⟨s · s⟩")
    ax.set_ylim(0.82, 1.005)
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.set_title("Experiment 1 — geometry of A, B, C, D during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(MEET / "exp1_geometry.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def exp1_ablation_shifts():
    """How much one K=1 update of W_in / J / both shifts each phase's J1 state."""
    with open(RES / "fixedpoint_overlap_v2.json") as f:
        d = json.load(f)
    cps = sorted(int(k) for k in d.keys())
    M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}
    # Labels: A_none=0, B_none=1, C_none=2, D_none=3,
    #         A_win=4, B_win=5, C_win=6, D_win=7,
    #         A_j=8,   B_j=9,   C_j=10,  D_j=11,
    #         A_both=12, B_both=13, C_both=14, D_both=15

    phases = ["A", "B", "C", "D"]
    ablations = {
        "W_in only": [(0, 4), (1, 5), (2, 6), (3, 7)],   # none vs win, per phase
        "J only":    [(0, 8), (1, 9), (2, 10), (3, 11)],
        "both":      [(0, 12), (1, 13), (2, 14), (3, 15)],
    }
    colors = {"W_in only": "tab:green", "J only": "tab:blue", "both": "tab:red"}

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, phase, pairs_for_phase in zip(axes, phases,
                                          zip(*ablations.values())):
        for ablation_name, (i, j) in zip(ablations.keys(), pairs_for_phase):
            ys = [match_to_dot(M[cp][i, j]) for cp in cps]
            ax.plot(cps, ys, "-o", label=ablation_name,
                    color=colors[ablation_name], lw=1.8, markersize=5)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(f"phase {phase}:  ⟨no-update · ablation⟩")
        ax.set_xlabel("training batch")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.6, 1.02)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "Experiment 1 — how much one update of each kind shifts the J1 state "
        "(dot product between pre-update and post-update state)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(MEET / "exp1_ablation_shifts.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def exp1_batch0_matrix():
    """Small annotated dot-product matrix at batch 0 (only baseline 4×4 block)."""
    with open(RES / "fixedpoint_overlap_v2.json") as f:
        d = json.load(f)
    M = np.array(d["0"]["match"])
    # Use only the baseline 4×4 (no ablations needed in the headline figure)
    sub = match_to_dot(M[:4, :4])
    labels = ["A", "B", "C", "D"]

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    vmin = float(np.min(sub[~np.eye(4, dtype=bool)]))   # exclude diagonal in vmin
    im = ax.imshow(sub, vmin=vmin - 0.005, vmax=1.0, cmap="viridis")
    for i in range(4):
        for j in range(4):
            v = sub[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v < (vmin + 1.0) / 2 + 0.005 else "black",
                    fontsize=11)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_title("Dot-product overlap matrix  (batch 0, no updates)")
    plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(MEET / "exp1_matrix_batch0.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----- Experiment 2 -----------------------------------------------------------

def _cohort_mean_curve(ax, x_mat, y_mat, mask, label, color):
    """Mean ± std over probes in the cohort, on the aligned x-axis."""
    if not mask.any():
        return
    sub_x = x_mat[mask]
    sub_y = y_mat[mask]
    # interpolate to a common grid
    valid = (sub_x >= 0) & np.isfinite(sub_y)
    if not valid.any():
        return
    xmin = int(sub_x[valid].min())
    xmax = int(sub_x[valid].max())
    step = max(1, (xmax - xmin) // 80 or 1)
    grid = np.arange(xmin, xmax + 1, step)
    means = np.full_like(grid, np.nan, dtype=float)
    stds  = np.full_like(grid, np.nan, dtype=float)
    for gi, g in enumerate(grid):
        vals = []
        for pi in range(sub_y.shape[0]):
            m = (sub_x[pi] >= 0) & np.isfinite(sub_y[pi])
            if m.any():
                xs_p, ys_p = sub_x[pi][m], sub_y[pi][m]
                if xs_p.min() <= g <= xs_p.max():
                    vals.append(float(np.interp(g, xs_p, ys_p)))
        if vals:
            means[gi] = float(np.mean(vals))
            stds[gi]  = float(np.std(vals))
    valid_grid = ~np.isnan(means)
    if valid_grid.any():
        ax.plot(grid[valid_grid], means[valid_grid], "-",
                color=color, lw=2.2, label=label)
        ax.fill_between(grid[valid_grid],
                        means[valid_grid] - stds[valid_grid],
                        means[valid_grid] + stds[valid_grid],
                        color=color, alpha=0.15)


def exp2_stability_cohort():
    """Stability_C, Stability_D, Cross-protocol, split by baseline cohort."""
    with open(RES / "cross_time_cd.json") as f:
        d = json.load(f)
    first_seen = np.array(d["first_seen"])
    measure_batches = np.array(d["measure_batches"])
    aligned = np.array(d["aligned_batches"])
    stab_C = np.array(d["stability_C_dot"])
    stab_D = np.array(d["stability_D_dot"])
    cross  = np.array(d["cross_C0_to_D_t"])

    # Identify cohorts
    baseline_b = np.full(len(first_seen), -1)
    for pi in range(len(first_seen)):
        cands = measure_batches[measure_batches >= first_seen[pi]]
        if len(cands):
            baseline_b[pi] = cands.min()
    mask_late = baseline_b == 50
    mask_init = baseline_b == 0
    n_late = int(mask_late.sum())
    n_init = int(mask_init.sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    panels = [
        ("stability_C  (same protocol)", stab_C),
        ("stability_D  (same protocol)", stab_D),
        ("C(t=baseline) ↔ D(t)  (cross protocol)", cross),
    ]
    for ax, (title, mat) in zip(axes, panels):
        _cohort_mean_curve(ax, aligned, mat, mask_late,
                           f"baselined at batch 50 (n={n_late})", "tab:blue")
        _cohort_mean_curve(ax, aligned, mat, mask_init,
                           f"baselined at batch 0   (n={n_init})", "tab:orange")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.axhline(0.0, color="k", lw=0.4, ls=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("batches since first-seen")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].legend(fontsize=9, loc="center right")
    fig.suptitle(
        "Experiment 2 — per-probe J1 stability split by baseline-capture batch",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(MEET / "exp2_stability_cohort.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def exp2_margin_dynamics():
    """Soft-margin evolution: per-class mean curves + per-probe final margin."""
    with open(RES / "per_image_trajectory_v2.json") as f:
        d = json.load(f)
    aligned = np.array(d["aligned_batches"])
    margin_D = np.array(d["margin_D"])
    probe_classes = np.array(d["probe_classes"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Per-class mean margin curves
    ax = axes[0]
    for c in range(10):
        m = probe_classes == c
        if not m.any():
            continue
        _cohort_mean_curve(ax, aligned, margin_D, m,
                           f"class {c}", plt.cm.tab10.colors[c])
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("Per-class mean soft margin (state D)")
    ax.set_xlabel("batches since first-seen")
    ax.set_ylabel("margin (correct − max wrong)")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)

    # Per-probe final margin bar chart, grouped by class
    ax = axes[1]
    final_margin = margin_D[:, -1]
    bar_x = []
    bar_y = []
    bar_c = []
    pos = 0
    xticks = []; xticklabels = []
    for c in range(10):
        m = probe_classes == c
        if not m.any():
            continue
        idxs = np.where(m)[0]
        for i in idxs:
            bar_x.append(pos); bar_y.append(final_margin[i])
            bar_c.append(plt.cm.tab10.colors[c])
            pos += 1
        xticks.append(pos - len(idxs) / 2 - 0.5)
        xticklabels.append(f"cl{c}")
        pos += 0.5
    ax.bar(bar_x, bar_y, color=bar_c, width=0.85)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylabel("final margin (state D)")
    ax.set_title("Final margin per probe, grouped by class\n"
                 "(bars above zero = classified correctly)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Experiment 2 — soft-margin dynamics  (50 probes = 5 per class, 2 epochs)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(MEET / "exp2_margin_dynamics.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----- Diagnostics (copy/regenerate) ------------------------------------------

def diagnostic_figures():
    """Three diagnostic panels, regenerated from JSON for consistency."""
    with open(RES / "dynamics_diagnostics.json") as f:
        d = json.load(f)
    cps = sorted(int(k) for k in d["contrib_means"].keys())
    contrib_means = d["contrib_means"]
    autocorr = d["autocorrelation"]

    # (B) Decomposition: bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sources = ["W_in", "J", "W_back"]
    colors = ["tab:green", "tab:blue", "tab:red"]
    markers = ["^", "s", "o"]
    for src, col, mk in zip(sources, colors, markers):
        ys = [contrib_means[str(cp)][src] for cp in cps]
        ax.plot(cps, ys, "-" + mk, color=col, label=f"⟨|h^{{{src}}}|⟩", lw=1.8)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("training batch")
    ax.set_ylabel("mean |contribution to h_i|")
    ax.set_title("Diagnostic B — decomposition of the J1 local field by source")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEET / "diag_field_decomposition.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # (C) Autocorrelation: 3 phases
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    phase_names = ["warmup", "clamped", "free"]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(cps)))
    for ax, phase in zip(axes, phase_names):
        for cp, col in zip(cps, cmap):
            curve = autocorr[str(cp)][phase]
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, "-o", color=col,
                    label=f"batch {cp}", markersize=4)
        ax.set_title(f"{phase} phase")
        ax.set_xlabel("step within phase")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("⟨s(t-1) · s(t)⟩")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Diagnostic C — step-to-step J1 autocorrelation within each phase",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(MEET / "diag_autocorrelation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def diag_h_distribution_copy():
    """Just copy the existing histogram figure into the meeting folder."""
    src = HERE / "figures" / "diag_A_h_distribution.png"
    dst = MEET / "diag_h_distribution.png"
    if src.exists():
        dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    exp1_geometry()
    exp1_ablation_shifts()
    exp1_batch0_matrix()
    exp2_stability_cohort()
    exp2_margin_dynamics()
    diagnostic_figures()
    diag_h_distribution_copy()
    print(f"Built meeting pack at {MEET}")
    for p in sorted(MEET.iterdir()):
        print(f"  {p.name}")
