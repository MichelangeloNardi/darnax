"""Regenerate exp-12 figure from results/chl.json (figures/ git-ignored).
  figures/curves.png — probe_D and head_acc_D per epoch, CHL vs DynamicalTrainer baseline.
Run: python p2_representation/12-chl_rule/plot.py
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


def _curve(d):
    seeds = list(d)
    eps = [h["epoch"] for h in d[seeds[0]]]
    pD = np.array([[h["probe_D"] for h in d[s]] for s in seeds])
    hd = np.array([[h["head_acc_D"] for h in d[s]] for s in seeds])
    return np.array(eps), pD, hd


def main():
    p = RES / "chl.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for d, lbl, c in [(r["chl"], "CHL", "C0"), (r["baseline"], "DynamicalTrainer", "C1")]:
        if not d:
            continue
        eps, pD, hd = _curve(d)
        axes[0].plot(eps, pD.mean(0), "-o", color=c, label=lbl)
        axes[0].fill_between(eps, pD.mean(0) - pD.std(0), pD.mean(0) + pD.std(0), color=c, alpha=0.2)
        axes[1].plot(eps, hd.mean(0), "-o", color=c, label=lbl)
    axes[0].axhline(0.51, ls="--", c="grey", lw=1, label="BPTT ceiling")
    axes[0].set_title("probe_D"); axes[1].set_title("head_acc_D")
    for a in axes:
        a.set_xlabel("epoch"); a.legend(fontsize=8)
    fig.suptitle("Exp 12 — contrastive (CHL) vs clamped-only (DynamicalTrainer) local rule", fontsize=12)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "curves.png", dpi=130); print("wrote", FIG / "curves.png")


if __name__ == "__main__":
    main()
