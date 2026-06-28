"""Regenerate exp-10 figures from results/diagnostics.json (figures/ is git-ignored).
  figures/global.png  — global metrics per config
  figures/groups.png  — per-group (I/L/N) metrics per config
Run: python p2_representation/10-split_scale/plot.py
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


def _load():
    p = RES / "diagnostics.json"
    return json.loads(p.read_text()) if p.exists() else None


def plot_global(d):
    names = d["names"]; m = d["models"]; x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))

    a = axes[0]
    for i, k in enumerate(["probe_C", "probe_D", "head_acc_D", "probe_C_transfer_D"]):
        a.bar(x + (i - 1.5) * 0.2, [m[n]["mean"][k] for n in names], 0.2,
              yerr=[m[n]["std"][k] for n in names], capsize=2, label=k)
    a.set_title("probe / head accuracy"); a.legend(fontsize=8)

    a = axes[1]
    for i, k in enumerate(["flip_rate", "overlap_CD"]):
        a.bar(x + (i - 0.5) * 0.35, [m[n]["mean"][k] for n in names], 0.35,
              yerr=[m[n]["std"][k] for n in names], capsize=2, label=k)
    a.set_title("C->D geometry"); a.legend(fontsize=8)

    a = axes[2]
    for i, k in enumerate(["fullspin_pC_on_C", "fullspin_pC_on_D", "rand_flip_acc"]):
        a.bar(x + (i - 1) * 0.27, [m[n]["mean"][k] for n in names], 0.27,
              yerr=[m[n]["std"][k] for n in names], capsize=2, label=k)
    a.set_title("full-spin C-probe: actual vs random flips"); a.legend(fontsize=8)

    for a in axes:
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, ha="right")
    fig.suptitle("Exp 10 — split routing at variable C: global metrics", fontsize=13)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "global.png", dpi=130); print("wrote", FIG / "global.png")


def plot_groups(d):
    names = d["names"]; m = d["models"]; x = np.arange(len(names))
    groups = ["I", "L", "N"]; colors = {"I": "C0", "L": "C3", "N": "C2"}
    metrics = [("probe_C_group", "group C-probe"), ("probe_D_group", "group D-probe"),
               ("flip_rate_group", "group C->D flip rate"), ("absfield_D_group", "|field| at D by group")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    for a, (k, title) in zip(axes, metrics):
        for gi, g in enumerate(groups):
            vals, errs = [], []
            for n in names:
                ga = m[n]["group_agg"][g]
                vals.append(ga[k]["mean"] if ga else np.nan)
                errs.append(ga[k]["std"] if ga else 0)
            a.bar(x + (gi - 1) * 0.27, vals, 0.27, yerr=errs, label=g, color=colors[g], capsize=2)
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, ha="right")
        a.set_title(title); a.legend(title="group", fontsize=8)
    fig.suptitle("Exp 10 — split routing at variable C: per-group (I/L/N) metrics", fontsize=13)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "groups.png", dpi=130); print("wrote", FIG / "groups.png")


if __name__ == "__main__":
    d = _load()
    if d is not None:
        plot_global(d); plot_groups(d)
