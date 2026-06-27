"""Regenerate exp-9 figures from the result JSONs (figures/ is git-ignored).

  figures/global.png   <- results/diagnostics.json + results/fullspin_importance.json
  figures/groups.png   <- results/group_diagnostics.json

Run: python p2_representation/9-channel_partition/plot.py
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
FIG = HERE / "figures"

# exp-3/5 model-A references (the baseline target)
REF = {"probe_C": 0.977, "probe_D": 0.451, "flip_rate": 0.052, "overlap_CD": 0.896,
       "pC_on_C": 0.973, "pC_on_D": 0.261, "rand_u": 0.972}


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def plot_global():
    diag = _load("diagnostics.json")
    fs = _load("fullspin_importance.json")
    if diag is None and fs is None:
        return
    base = diag or fs
    tags = base["tags"]
    x = np.arange(len(tags))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    if diag is not None:
        m = diag["models"]
        for ax, (mc, ml, ref_c, ref_l, title) in zip(
            axes[:2],
            [("probe_C", "probe_D", REF["probe_C"], REF["probe_D"], "Probe acc (pooled)"),
             ("flip_rate", "overlap_CD", REF["flip_rate"], REF["overlap_CD"], "C->D geometry")]):
            a = [m[t]["mean"][mc] for t in tags]
            ae = [m[t]["std"][mc] for t in tags]
            b = [m[t]["mean"][ml] for t in tags]
            be = [m[t]["std"][ml] for t in tags]
            ax.bar(x - 0.2, a, 0.4, yerr=ae, label=mc, capsize=3)
            ax.bar(x + 0.2, b, 0.4, yerr=be, label=ml, capsize=3)
            ax.axhline(ref_c, ls="--", c="C0", lw=1, alpha=0.6)
            ax.axhline(ref_l, ls="--", c="C1", lw=1, alpha=0.6)
            ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right")
            ax.set_title(title); ax.legend(fontsize=8)

    if fs is not None:
        m = fs["models"]; ax = axes[2]
        cc = [m[t]["mean"]["probe_C_on_C"] for t in tags]
        cd = [m[t]["mean"]["probe_C_on_D"] for t in tags]
        rr = [m[t]["mean"]["acc_rand_uniform"] for t in tags]
        ax.bar(x - 0.27, cc, 0.27, label="C-probe on C")
        ax.bar(x, cd, 0.27, label="on D (actual flips)")
        ax.bar(x + 0.27, rr, 0.27, label="on D (random flips)")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_title("Flip-targeting (full-spin C-probe)"); ax.legend(fontsize=8)

    fig.suptitle("Exp 9 — channel partition: global C/D diagnostics", fontsize=13)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "global.png", dpi=130)
    print("wrote", FIG / "global.png")


def plot_groups():
    g = _load("group_diagnostics.json")
    if g is None:
        return
    tags = g["tags"]; m = g["models"]
    groups = ["I", "L", "N"]
    colors = {"I": "C0", "L": "C3", "N": "C2"}
    metrics = [("probe_C", "C-probe by group"), ("probe_D", "D-probe by group"),
               ("flip_rate", "C->D flip rate by group")]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    x = np.arange(len(tags))
    for ax, (field, title) in zip(axes[:3], metrics):
        for gi, gn in enumerate(groups):
            vals = [m[t]["group_agg"][gn][field]["mean"] for t in tags]
            errs = [m[t]["group_agg"][gn][field]["std"] or 0 for t in tags]
            vals = [v if v is not None else np.nan for v in vals]
            ax.bar(x + (gi - 1) * 0.27, vals, 0.27, yerr=errs, label=gn,
                   color=colors[gn], capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_title(title); ax.legend(title="group", fontsize=8)

    # random-flip control by group: actual vs random (paired), averaged over groups present
    ax = axes[3]
    for gi, gn in enumerate(groups):
        act = [m[t]["group_agg"][gn]["randctrl_acc_actual_flips"]["mean"] for t in tags]
        ran = [m[t]["group_agg"][gn]["randctrl_acc_random_flips"]["mean"] for t in tags]
        act = [v if v is not None else np.nan for v in act]
        ran = [v if v is not None else np.nan for v in ran]
        ax.plot(x, act, "o-", color=colors[gn], label=f"{gn} actual")
        ax.plot(x, ran, "o--", color=colors[gn], alpha=0.5, label=f"{gn} random")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_title("Flip-targeting by group\n(global full-spin C-probe acc)")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Exp 9 — channel partition: per-group (I / L / N) diagnostics", fontsize=13)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "groups.png", dpi=130)
    print("wrote", FIG / "groups.png")


if __name__ == "__main__":
    plot_global()
    plot_groups()
