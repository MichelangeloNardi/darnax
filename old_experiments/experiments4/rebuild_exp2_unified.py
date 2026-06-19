"""experiments4/rebuild_exp2_unified.py

Rebuild the Experiment-2 stability figure from cross_time_cd_v2.json
(all probes baselined at batch 50, no cohort split needed).

Also regenerate the per-class margin figure from cross_time_cd_v2.json
(margin_D over time + final bars), keeping it identical in spirit to
the earlier exp2_margin_dynamics.png but with the same probe data as the
new run.
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


def _mean_band(ax, x, mat, label, color):
    """Mean ± std over probes, shared x-axis."""
    means = np.nanmean(mat, axis=0)
    stds  = np.nanstd(mat,  axis=0)
    ax.plot(x, means, "-", color=color, lw=2.4, label=label)
    ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.18)


def exp2_stability_unified():
    with open(RES / "cross_time_cd_v2.json") as f:
        d = json.load(f)
    baseline_batch = d["baseline_batch"]
    measure_batches = np.array(d["measure_batches"])
    stab_C = np.array(d["stability_C_dot"])
    stab_D = np.array(d["stability_D_dot"])
    cross  = np.array(d["cross_C0_to_D_t"])

    # Only show measurements at or after the baseline
    keep = measure_batches >= baseline_batch
    x = measure_batches[keep] - baseline_batch    # x = "batches since baseline"
    stab_C = stab_C[:, keep]
    stab_D = stab_D[:, keep]
    cross  = cross[:, keep]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    # Per-probe + mean for each metric
    for ax, (mat, title) in zip(axes, [
        (stab_C, "stability_C  (same protocol)"),
        (stab_D, "stability_D  (same protocol)"),
        (cross,  f"C(t={baseline_batch}) ↔ D(t)   (cross protocol)"),
    ]):
        for pi in range(mat.shape[0]):
            ax.plot(x, mat[pi], "-", color="tab:gray", alpha=0.25, lw=0.7)
        _mean_band(ax, x, mat, "mean ± std", "tab:blue")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"batches since baseline (batch {baseline_batch})")
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
        ax.axhline(0.0, color="k", lw=0.4, ls=":")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("dot-product overlap")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(
        f"Experiment 2 — per-probe J1 stability (all 50 probes baselined at batch {baseline_batch})",
        fontsize=11,
    )
    fig.tight_layout()
    out = MEET / "exp2_stability_unified.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def exp2_margin_unified():
    with open(RES / "cross_time_cd_v2.json") as f:
        d = json.load(f)
    measure_batches = np.array(d["measure_batches"])
    margin_D = np.array(d["margin_D"])       # (probes, n_meas)
    probe_classes = np.array(d["probe_classes"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # (a) Per-class mean margin curves
    ax = axes[0]
    for c in range(10):
        m = probe_classes == c
        if not m.any():
            continue
        cls_margin = margin_D[m].mean(axis=0)
        ax.plot(measure_batches, cls_margin, "-",
                color=plt.cm.tab10.colors[c], lw=1.6, label=f"class {c}")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("Per-class mean soft margin (state D)")
    ax.set_xlabel("training batch")
    ax.set_ylabel("margin (correct − max wrong)")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)

    # (b) Per-probe final margin bar chart
    ax = axes[1]
    final_margin = margin_D[:, -1]
    bar_x = []; bar_y = []; bar_c = []
    pos = 0
    xticks = []; xticklabels = []
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
    ax.set_title("Final margin per probe, grouped by class\n"
                 "(bars above zero = classified correctly)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Experiment 2 — soft-margin dynamics  (50 probes, 2 epochs)",
                 fontsize=11)
    fig.tight_layout()
    out = MEET / "exp2_margin_unified.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    exp2_stability_unified()
    exp2_margin_unified()
    # Clean up old (cohort-split) figures so they don't get mistaken
    for stale in ["exp2_stability_cohort.png", "exp2_margin_dynamics.png"]:
        p = MEET / stale
        if p.exists():
            p.unlink()
            print(f"removed stale {p}")
