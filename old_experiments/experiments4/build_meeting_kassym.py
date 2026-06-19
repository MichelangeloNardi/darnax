"""experiments4/build_meeting_kassym.py

Build the Kassym-config equivalents of the meeting figures, plus four
side-by-side comparison figures (Config 3 vs Kassym).

Reads from:
  results/fixedpoint_overlap_v2.json        (Config 3 Exp 1)
  results/cross_time_cd_v2.json             (Config 3 Exp 2)
  results/fixedpoint_overlap_kassym.json    (Kassym Exp 1)
  results/cross_time_cd_kassym.json         (Kassym Exp 2)

Writes to:
  meeting_kassym/  — Kassym-only figures (same naming as meeting/)
  meeting/exp_compare_*.png  — side-by-side comparison figures
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
MEET_K = HERE / "meeting_kassym"
MEET_K.mkdir(exist_ok=True)


def match_to_dot(m):
    return 2.0 * np.asarray(m) - 1.0


# ---------- Exp 1 single-config plots (Kassym) -------------------------------

def exp1_geometry_kassym():
    with open(RES / "fixedpoint_overlap_kassym.json") as f:
        d = json.load(f)
    cps = sorted(int(k) for k in d.keys())
    M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}
    keys = [("A↔B", 0, 1), ("A↔C", 0, 2), ("A↔D", 0, 3),
            ("B↔C", 1, 2), ("B↔D", 1, 3), ("C↔D", 2, 3)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, i, j), col in zip(keys, colors):
        ys = [match_to_dot(M[cp][i, j]) for cp in cps]
        ax.plot(cps, ys, "-o", label=label, color=col, lw=2.0, markersize=6)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("training batch")
    ax.set_ylabel("dot-product overlap")
    ax.set_ylim(0.35, 1.005)
    ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.set_title("Kassym config — geometry of A, B, C, D during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(MEET_K / "exp1_geometry.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def exp1_ablation_kassym():
    with open(RES / "fixedpoint_overlap_kassym.json") as f:
        d = json.load(f)
    cps = sorted(int(k) for k in d.keys())
    M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}
    phases = ["A", "B", "C", "D"]
    ablations = {
        "W_in only": [(0, 4), (1, 5), (2, 6), (3, 7)],
        "J only":    [(0, 8), (1, 9), (2, 10), (3, 11)],
        "both":      [(0, 12), (1, 13), (2, 14), (3, 15)],
    }
    colors = {"W_in only": "tab:green", "J only": "tab:blue", "both": "tab:red"}

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, phase, pairs in zip(axes, phases, zip(*ablations.values())):
        for name, (i, j) in zip(ablations.keys(), pairs):
            ys = [match_to_dot(M[cp][i, j]) for cp in cps]
            ax.plot(cps, ys, "-o", color=colors[name], lw=1.8, markersize=5, label=name)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(f"phase {phase}:  ⟨no-update · ablation⟩")
        ax.set_xlabel("training batch")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Kassym config — K=1 update shifts per phase", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEET_K / "exp1_ablation_shifts.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def exp1_4x16_kassym():
    with open(RES / "fixedpoint_overlap_kassym.json") as f:
        d = json.load(f)
    checkpoints = [0, 1, 5, 25, 100]
    col_labels = ["A", "B", "C", "D",
                  "A'_W_in", "B'_W_in", "C'_W_in", "D'_W_in",
                  "A'_J", "B'_J", "C'_J", "D'_J",
                  "A'_both", "B'_both", "C'_both", "D'_both"]
    row_labels = ["A", "B", "C", "D"]

    submats = []
    for cp in checkpoints:
        M = np.array(d[str(cp)]["match"])
        submats.append(match_to_dot(M[:4, :]))
    diag_mask = np.eye(4, 16, dtype=bool)
    all_off = np.concatenate([s[~diag_mask].ravel() for s in submats])
    vmin = float(all_off.min())

    fig, axes = plt.subplots(1, len(checkpoints),
                             figsize=(5.5 * len(checkpoints), 4.0), sharey=True)
    for ax, cp, sub in zip(axes, checkpoints, submats):
        im = ax.imshow(sub, vmin=vmin - 0.01, vmax=1.0, cmap="viridis", aspect="auto")
        for i in range(4):
            for j in range(16):
                v = sub[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < (vmin + 1.0) / 2 + 0.05 else "black",
                        fontsize=7)
        for x in (3.5, 7.5, 11.5):
            ax.axvline(x, color="white", lw=1.2)
        ax.set_xticks(range(16))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(f"batch {cp}", fontsize=11)
    axes[0].set_ylabel("baseline phase")
    fig.suptitle("Kassym config — 4×16 matrix at 5 checkpoints", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="dot product")
    fig.savefig(MEET_K / "exp1_matrix_4x16.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------- Exp 2 single-config plots (Kassym) -------------------------------

def _mean_band(ax, x, mat, label, color):
    means = np.nanmean(mat, axis=0)
    stds  = np.nanstd(mat,  axis=0)
    ax.plot(x, means, "-", color=color, lw=2.4, label=label)
    ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.18)


def exp2_stability_kassym():
    with open(RES / "cross_time_cd_kassym.json") as f:
        d = json.load(f)
    baseline = d["baseline_batch"]
    mb = np.array(d["measure_batches"])
    stab_C = np.array(d["stability_C_dot"])
    stab_D = np.array(d["stability_D_dot"])
    cross  = np.array(d["cross_C0_to_D_t"])

    keep = mb >= baseline
    x = mb[keep] - baseline
    stab_C = stab_C[:, keep]; stab_D = stab_D[:, keep]; cross = cross[:, keep]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (mat, title) in zip(axes, [
        (stab_C, "stability_C  (same protocol)"),
        (stab_D, "stability_D  (same protocol)"),
        (cross,  f"C(t={baseline}) ↔ D(t)  (cross protocol)"),
    ]):
        for pi in range(mat.shape[0]):
            ax.plot(x, mat[pi], "-", color="tab:gray", alpha=0.20, lw=0.7)
        _mean_band(ax, x, mat, "mean ± std", "tab:red")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"batches since baseline (batch {baseline})")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.axhline(0.0, color="k", lw=0.4, ls=":")
        ax.set_ylim(-0.5, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Kassym config — per-probe J1 stability (all 50 probes baselined at batch 50)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(MEET_K / "exp2_stability_unified.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def exp2_margin_kassym():
    with open(RES / "cross_time_cd_kassym.json") as f:
        d = json.load(f)
    mb = np.array(d["measure_batches"])
    margin_D = np.array(d["margin_D"])
    probe_classes = np.array(d["probe_classes"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for c in range(10):
        m = probe_classes == c
        if not m.any():
            continue
        cls_margin = margin_D[m].mean(axis=0)
        ax.plot(mb, cls_margin, "-", color=plt.cm.tab10.colors[c], lw=1.6, label=f"class {c}")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("Per-class mean soft margin (state D) — Kassym")
    ax.set_xlabel("training batch")
    ax.set_ylabel("margin")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    final_margin = margin_D[:, -1]
    bar_x = []; bar_y = []; bar_c = []
    pos = 0; xticks = []; xticklabels = []
    for c in range(10):
        m = probe_classes == c
        if not m.any():
            continue
        idxs = np.where(m)[0]
        for i in idxs:
            bar_x.append(pos); bar_y.append(final_margin[i])
            bar_c.append(plt.cm.tab10.colors[c]); pos += 1
        xticks.append(pos - len(idxs) / 2 - 0.5)
        xticklabels.append(f"cl{c}")
        pos += 0.5
    ax.bar(bar_x, bar_y, color=bar_c, width=0.85)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylabel("final margin (state D)")
    ax.set_title("Final margin per probe, grouped by class — Kassym")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Kassym config — soft-margin dynamics", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEET_K / "exp2_margin_unified.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------- Side-by-side comparison figures ---------------------------------

def compare_exp1_geometry():
    """Two-panel side-by-side: Config 3 vs Kassym, 6-line geometry."""
    with open(RES / "fixedpoint_overlap_v2.json") as f:
        d3 = json.load(f)
    with open(RES / "fixedpoint_overlap_kassym.json") as f:
        dk = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    keys = [("A↔B", 0, 1), ("A↔C", 0, 2), ("A↔D", 0, 3),
            ("B↔C", 1, 2), ("B↔D", 1, 3), ("C↔D", 2, 3)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for ax, d, title in [(axes[0], d3, "Config 3 (long dynamics 6-11-14)"),
                         (axes[1], dk, "Kassym (short dynamics 1-5-6)")]:
        cps = sorted(int(k) for k in d.keys())
        M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}
        for (label, i, j), col in zip(keys, colors):
            ys = [match_to_dot(M[cp][i, j]) for cp in cps]
            ax.plot(cps, ys, "-o", label=label, color=col, lw=2.0, markersize=5)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training batch")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].set_ylim(0.35, 1.005)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Experiment 1 comparison — A/B/C/D geometry over training",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(MEET / "exp_compare_exp1_geometry.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def compare_exp2_stability():
    """Two-panel: Config 3 vs Kassym, cross-protocol C(50)↔D(t) trajectory."""
    with open(RES / "cross_time_cd_v2.json") as f:
        d3 = json.load(f)
    with open(RES / "cross_time_cd_kassym.json") as f:
        dk = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, d, title, color in [
        (axes[0], d3, "Config 3 (long dynamics)", "tab:blue"),
        (axes[1], dk, "Kassym (short dynamics)", "tab:red"),
    ]:
        baseline = d["baseline_batch"]
        mb = np.array(d["measure_batches"])
        cross = np.array(d["cross_C0_to_D_t"])
        keep = mb >= baseline
        x = mb[keep] - baseline
        cross = cross[:, keep]
        for pi in range(cross.shape[0]):
            ax.plot(x, cross[pi], "-", color="tab:gray", alpha=0.18, lw=0.6)
        _mean_band(ax, x, cross, "mean ± std", color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"batches since baseline (batch {baseline})")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.axhline(0.0, color="k", lw=0.4, ls=":")
        ax.set_ylim(-0.5, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("dot-product overlap")
    fig.suptitle("Experiment 2 comparison — cross-protocol C(t=50) ↔ D(t)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(MEET / "exp_compare_exp2_stability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def compare_exp2_margin():
    """Two-panel: Config 3 vs Kassym, per-class mean soft margin."""
    with open(RES / "cross_time_cd_v2.json") as f:
        d3 = json.load(f)
    with open(RES / "cross_time_cd_kassym.json") as f:
        dk = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, d, title in [(axes[0], d3, "Config 3"), (axes[1], dk, "Kassym")]:
        mb = np.array(d["measure_batches"])
        margin_D = np.array(d["margin_D"])
        probe_classes = np.array(d["probe_classes"])
        for c in range(10):
            m = probe_classes == c
            if not m.any():
                continue
            cls = margin_D[m].mean(axis=0)
            ax.plot(mb, cls, "-", color=plt.cm.tab10.colors[c], lw=1.6,
                    label=f"class {c}")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training batch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-class mean soft margin (state D)")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("Experiment 2 comparison — per-class soft margin",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(MEET / "exp_compare_exp2_margin.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def compare_exp1_ablation():
    """Side-by-side ablation shifts for phase A only (simpler)."""
    with open(RES / "fixedpoint_overlap_v2.json") as f:
        d3 = json.load(f)
    with open(RES / "fixedpoint_overlap_kassym.json") as f:
        dk = json.load(f)
    colors = {"W_in only": "tab:green", "J only": "tab:blue", "both": "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, d, title in [(axes[0], d3, "Config 3"), (axes[1], dk, "Kassym")]:
        cps = sorted(int(k) for k in d.keys())
        M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}
        for name, idx in [("W_in only", 4), ("J only", 8), ("both", 12)]:
            ys = [match_to_dot(M[cp][0, idx]) for cp in cps]
            ax.plot(cps, ys, "-o", color=colors[name], lw=1.8, markersize=5,
                    label=name)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training batch")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.set_ylim(0.55, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("⟨A_none · A_ablation⟩  (shift per K=1 update)")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Experiment 1 comparison — how much each K=1 update shifts state A",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(MEET / "exp_compare_exp1_ablation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Per-config Kassym figures
    exp1_geometry_kassym()
    exp1_ablation_kassym()
    exp1_4x16_kassym()
    exp2_stability_kassym()
    exp2_margin_kassym()

    # Side-by-side comparison figures
    compare_exp1_geometry()
    compare_exp1_ablation()
    compare_exp2_stability()
    compare_exp2_margin()

    print(f"\nKassym-only figures in {MEET_K}:")
    for p in sorted(MEET_K.iterdir()):
        print(f"  {p.name}")
    print(f"\nComparison figures in {MEET}:")
    for p in sorted(MEET.glob("exp_compare_*.png")):
        print(f"  {p.name}")
