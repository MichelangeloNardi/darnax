"""Regenerate figures/phenomenon.png from results/phenomenon*.json.

Compares the two W_in variants (frozen vs trainable) on the FC C->D flip phenomenon:
  (a) probe accuracies: C-probe on C, C-probe on D, D-probe on D, random-flip control
  (b) C->D flip rate and C/D overlap
  (c) flip rate by C-probe importance decile (targeting signal)

Run: uv run python p3_spin_analysis/1-fc_phenomenon/plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

FILES = [("frozen", RES / "phenomenon.json"), ("trainwin", RES / "phenomenon_trainwin.json")]


def load():
    out = {}
    for name, p in FILES:
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def main():
    data = load()
    if not data:
        print("no results yet"); return
    names = list(data)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) probe accuracies
    keys = ["probe_C_on_C", "probe_C_on_D", "probe_D_on_D", "acc_rand_uniform"]
    labels = ["C-probe/C", "C-probe/D", "D-probe/D", "rand-flip ctrl"]
    x = np.arange(len(keys)); w = 0.8 / len(names)
    for i, n in enumerate(names):
        m = [data[n]["mean"][k] for k in keys]
        s = [data[n]["std"][k] for k in keys]
        axes[0].bar(x + i * w, m, w, yerr=s, capsize=3, label=n)
    axes[0].axhline(0.1, ls="--", c="gray", lw=1, label="chance")
    axes[0].set_xticks(x + w * (len(names) - 1) / 2); axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("test accuracy"); axes[0].set_title("probe accuracies"); axes[0].legend()

    # (b) flip rate + overlap
    keys2 = ["flip_rate_overall", "overlap_CD", "top5pct_enrichment"]
    x2 = np.arange(len(keys2))
    for i, n in enumerate(names):
        m = [data[n]["mean"][k] for k in keys2]
        axes[1].bar(x2 + i * w, m, w, label=n)
    axes[1].set_xticks(x2 + w * (len(names) - 1) / 2)
    axes[1].set_xticklabels(["C->D flip", "overlap(C,D)", "top5% enrich"], rotation=20)
    axes[1].set_title("flip geometry"); axes[1].legend()

    # (c) flip rate by importance decile
    for n in names:
        axes[2].plot(range(1, 11), data[n]["flip_rate_by_decile_mean"], "o-", label=n)
    axes[2].set_xlabel("C-probe importance decile"); axes[2].set_ylabel("flip rate")
    axes[2].set_title("flip rate vs importance"); axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIG / "phenomenon.png", dpi=130)
    print(f"wrote {FIG / 'phenomenon.png'}")


if __name__ == "__main__":
    main()
