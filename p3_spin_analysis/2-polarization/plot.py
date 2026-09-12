"""Regenerate exp p3-2 figures from results/polarization_{frozen,trainwin}.json.
  figures/polarization.png — per-unit scatters (|p_D|, S_D vs flip_rate, colored by importance)
                             + key correlation bars (mean±std over seeds), both configs.
Run: python p3_spin_analysis/2-polarization/plot.py
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

KEY = ["corr_absp_D_flip_rate", "corr_S_D_flip_rate", "corr_absp_D_absfield_C",
       "corr_S_D_importance", "corr_S_D_damage"]


def main():
    cfgs = [(t, RES / f"polarization_{t}.json") for t in ("frozen", "trainwin")]
    cfgs = [(t, p) for t, p in cfgs if p.exists()]
    if not cfgs:
        return
    fig, axes = plt.subplots(len(cfgs), 3, figsize=(16, 4.6 * len(cfgs)), squeeze=False)
    for r, (tag, p) in enumerate(cfgs):
        d = json.loads(p.read_text()); a = d["arrays_seed0"]
        fr = np.array(a["flip_rate"]); imp = np.array(a["importance"])
        for c, (xk, xlab) in enumerate([("absp_D", "|p_i|  (polarization, state D)"),
                                        ("S_D", "S_i  (selectivity, state D)")]):
            ax = axes[r][c]
            sc = ax.scatter(np.array(a[xk]), fr, c=imp, s=14, cmap="viridis")
            ax.set_xlabel(xlab); ax.set_ylabel("flip rate"); ax.set_title(f"{tag}: {xk} vs flip")
            plt.colorbar(sc, ax=ax, label="importance")
        ax = axes[r][2]
        x = np.arange(len(KEY))
        means = [d["mean"][k] for k in KEY]; stds = [d["std"][k] for k in KEY]
        ax.bar(x, means, yerr=stds, capsize=3, color="C0")
        ax.axhline(0, c="grey", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([k.replace("corr_", "") for k in KEY], rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{tag}: key correlations (mean±std, 3 seeds)"); ax.set_ylabel("Pearson r")
    fig.suptitle("Exp p3-2 — polarization / selectivity vs C→D flip danger (256 FC units)", fontsize=13)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "polarization.png", dpi=130); print("wrote", FIG / "polarization.png")


if __name__ == "__main__":
    main()
