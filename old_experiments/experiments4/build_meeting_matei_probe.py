"""experiments4/build_meeting_matei_probe.py

Build Matei-probe (trial 13) meeting figures, the full 16x16 matrices for
BOTH Matei-probe and Kassym configs, and three-way comparison figures
(Config 3 vs Kassym vs Matei-probe).

Reads:
  results/fixedpoint_overlap_v2.json          (Config 3 Exp 1)
  results/cross_time_cd_v2.json               (Config 3 Exp 2)
  results/fixedpoint_overlap_kassym.json      (Kassym Exp 1)
  results/cross_time_cd_kassym.json           (Kassym Exp 2)
  results/fixedpoint_overlap_matei_probe.json (Matei-probe Exp 1)
  results/cross_time_cd_matei_probe.json      (Matei-probe Exp 2)

Writes:
  meeting_matei_probe/   — Matei-probe-only figures
  meeting/exp1_matrix_16x16_<config>.png  — full 16x16 matrices
  meeting/exp_compare3_*.png              — three-way comparisons
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
MEET_M = HERE / "meeting_matei_probe"
MEET_M.mkdir(exist_ok=True)

CONFIGS = {
    "config3": {
        "exp1": "fixedpoint_overlap_v2.json",
        "exp2": "cross_time_cd_v2.json",
        "label": "Config 3 (Matei W_out-tuned, 6-11-14)",
    },
    "kassym": {
        "exp1": "fixedpoint_overlap_kassym.json",
        "exp2": "cross_time_cd_kassym.json",
        "label": "Kassym (probe-tuned, 1-5-6)",
    },
    "matei_probe": {
        "exp1": "fixedpoint_overlap_matei_probe.json",
        "exp2": "cross_time_cd_matei_probe.json",
        "label": "Matei probe-tuned (trial 13, 1-4-13)",
    },
}

POINT_LABELS = ["A", "B", "C", "D",
                "A'win", "B'win", "C'win", "D'win",
                "A'J", "B'J", "C'J", "D'J",
                "A'both", "B'both", "C'both", "D'both"]


def match_to_dot(m):
    return 2.0 * np.asarray(m) - 1.0


# ---------- Full 16x16 matrix at selected checkpoints ------------------------

def full_16x16(config_key):
    cfg = CONFIGS[config_key]
    with open(RES / cfg["exp1"]) as f:
        d = json.load(f)
    checkpoints = [0, 5, 50, 1000]
    submats = [match_to_dot(np.array(d[str(cp)]["match"])) for cp in checkpoints]
    # common vmin over off-diagonal cells
    off = []
    for s in submats:
        off.append(s[~np.eye(16, dtype=bool)].ravel())
    vmin = float(np.concatenate(off).min())

    fig, axes = plt.subplots(1, len(checkpoints),
                             figsize=(5.6 * len(checkpoints), 5.2), sharey=True)
    for ax, cp, sub in zip(axes, checkpoints, submats):
        im = ax.imshow(sub, vmin=vmin - 0.01, vmax=1.0, cmap="viridis")
        for i in range(16):
            for j in range(16):
                v = sub[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < (vmin + 1.0) / 2 + 0.03 else "black",
                        fontsize=4.3)
        for pos in (3.5, 7.5, 11.5):
            ax.axhline(pos, color="white", lw=1.0)
            ax.axvline(pos, color="white", lw=1.0)
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(POINT_LABELS, rotation=90, fontsize=6)
        ax.set_yticklabels(POINT_LABELS, fontsize=6)
        ax.set_title(f"batch {cp}", fontsize=11)
    fig.suptitle(f"Full 16×16 dot-product matrix — {cfg['label']}", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="dot product")
    out = MEET / f"exp1_matrix_16x16_{config_key}.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---------- Matei-probe single-config figures --------------------------------

def matei_geometry():
    with open(RES / CONFIGS["matei_probe"]["exp1"]) as f:
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
    ax.set_xlabel("training batch"); ax.set_ylabel("dot-product overlap")
    ax.set_ylim(0.5, 1.01); ax.axhline(1.0, color="k", lw=0.4, ls=":")
    ax.set_title("Matei probe-tuned — geometry of A, B, C, D")
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(MEET_M / "exp1_geometry.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved matei exp1_geometry")


def _mean_band(ax, x, mat, label, color):
    means = np.nanmean(mat, axis=0); stds = np.nanstd(mat, axis=0)
    ax.plot(x, means, "-", color=color, lw=2.4, label=label)
    ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.18)


def matei_stability():
    with open(RES / CONFIGS["matei_probe"]["exp2"]) as f:
        d = json.load(f)
    baseline = d["baseline_batch"]
    mb = np.array(d["measure_batches"])
    stab_C = np.array(d["stability_C_dot"])
    stab_D = np.array(d["stability_D_dot"])
    cross = np.array(d["cross_C0_to_D_t"])
    keep = mb >= baseline
    x = mb[keep] - baseline
    stab_C, stab_D, cross = stab_C[:, keep], stab_D[:, keep], cross[:, keep]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (mat, title) in zip(axes, [
        (stab_C, "stability_C (same protocol)"),
        (stab_D, "stability_D (same protocol)"),
        (cross, f"C(t={baseline}) ↔ D(t) (cross protocol)"),
    ]):
        for pi in range(mat.shape[0]):
            ax.plot(x, mat[pi], "-", color="tab:gray", alpha=0.20, lw=0.7)
        _mean_band(ax, x, mat, "mean ± std", "tab:purple")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"batches since baseline (batch {baseline})")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.axhline(0.0, color="k", lw=0.4, ls=":")
        ax.set_ylim(-1.0, 1.05); ax.grid(alpha=0.3)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Matei probe-tuned — per-probe J1 stability (baseline batch 50)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(MEET_M / "exp2_stability_unified.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved matei exp2_stability")


def matei_margin():
    with open(RES / CONFIGS["matei_probe"]["exp2"]) as f:
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
        ax.plot(mb, margin_D[m].mean(axis=0), "-",
                color=plt.cm.tab10.colors[c], lw=1.6, label=f"class {c}")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("Per-class mean soft margin (state D) — Matei probe")
    ax.set_xlabel("training batch"); ax.set_ylabel("margin")
    ax.legend(fontsize=8, ncol=2, loc="lower left"); ax.grid(alpha=0.3)
    ax = axes[1]
    final = margin_D[:, -1]
    bx, by, bc = [], [], []
    pos = 0; xt, xtl = [], []
    for c in range(10):
        m = probe_classes == c
        if not m.any():
            continue
        idxs = np.where(m)[0]
        for i in idxs:
            bx.append(pos); by.append(final[i]); bc.append(plt.cm.tab10.colors[c]); pos += 1
        xt.append(pos - len(idxs) / 2 - 0.5); xtl.append(f"cl{c}"); pos += 0.5
    ax.bar(bx, by, color=bc, width=0.85)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(xt); ax.set_xticklabels(xtl, fontsize=8)
    ax.set_ylabel("final margin (state D)")
    ax.set_title("Final margin per probe, grouped by class — Matei probe")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Matei probe-tuned — soft-margin dynamics", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEET_M / "exp2_margin_unified.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved matei exp2_margin")


# ---------- Three-way comparisons --------------------------------------------

def compare3_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    keys = [("A↔B", 0, 1), ("A↔C", 0, 2), ("A↔D", 0, 3),
            ("B↔C", 1, 2), ("B↔D", 1, 3), ("C↔D", 2, 3)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for ax, key in zip(axes, ["config3", "kassym", "matei_probe"]):
        with open(RES / CONFIGS[key]["exp1"]) as f:
            d = json.load(f)
        cps = sorted(int(k) for k in d.keys())
        M = {cp: np.array(d[str(cp)]["match"]) for cp in cps}
        for (label, i, j), col in zip(keys, colors):
            ys = [match_to_dot(M[cp][i, j]) for cp in cps]
            ax.plot(cps, ys, "-o", label=label, color=col, lw=1.8, markersize=4)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(CONFIGS[key]["label"], fontsize=10)
        ax.set_xlabel("training batch")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].set_ylim(0.5, 1.01)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Experiment 1 — A/B/C/D geometry across three configs", fontsize=12)
    fig.tight_layout()
    fig.savefig(MEET / "exp_compare3_exp1_geometry.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("saved compare3 exp1 geometry")


def compare3_stability():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cols = {"config3": "tab:blue", "kassym": "tab:red", "matei_probe": "tab:purple"}
    for ax, key in zip(axes, ["config3", "kassym", "matei_probe"]):
        with open(RES / CONFIGS[key]["exp2"]) as f:
            d = json.load(f)
        baseline = d["baseline_batch"]
        mb = np.array(d["measure_batches"])
        cross = np.array(d["cross_C0_to_D_t"])
        keep = mb >= baseline
        x = mb[keep] - baseline
        cross = cross[:, keep]
        for pi in range(cross.shape[0]):
            ax.plot(x, cross[pi], "-", color="tab:gray", alpha=0.15, lw=0.6)
        _mean_band(ax, x, cross, "mean ± std", cols[key])
        ax.set_title(CONFIGS[key]["label"], fontsize=10)
        ax.set_xlabel(f"batches since baseline ({baseline})")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.axhline(0.0, color="k", lw=0.4, ls=":")
        ax.set_ylim(-1.0, 1.05); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("cross-protocol C(t=50) ↔ D(t)")
    fig.suptitle("Experiment 2 — cross-protocol J1 drift across three configs",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(MEET / "exp_compare3_exp2_stability.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("saved compare3 exp2 stability")


def compare3_margin():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, key in zip(axes, ["config3", "kassym", "matei_probe"]):
        with open(RES / CONFIGS[key]["exp2"]) as f:
            d = json.load(f)
        mb = np.array(d["measure_batches"])
        margin_D = np.array(d["margin_D"])
        probe_classes = np.array(d["probe_classes"])
        for c in range(10):
            m = probe_classes == c
            if not m.any():
                continue
            ax.plot(mb, margin_D[m].mean(axis=0), "-",
                    color=plt.cm.tab10.colors[c], lw=1.4, label=f"class {c}")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_title(CONFIGS[key]["label"], fontsize=10)
        ax.set_xlabel("training batch"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-class mean soft margin (state D)")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Experiment 2 — per-class soft margin across three configs",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(MEET / "exp_compare3_exp2_margin.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("saved compare3 exp2 margin")


if __name__ == "__main__":
    # Matei-probe single-config
    matei_geometry()
    matei_stability()
    matei_margin()
    # Full 16x16 matrices for matei_probe and kassym
    full_16x16("matei_probe")
    full_16x16("kassym")
    full_16x16("config3")
    # Three-way comparisons
    compare3_geometry()
    compare3_stability()
    compare3_margin()
    print("\nDone.")
