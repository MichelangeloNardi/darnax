"""Plot the C/D diagnostics across the 3 models (A local-rule, B BPTT CE_D,
C BPTT CE_D+align). Reads results/diagnostics.json.

Usage:  python p2_representation/3-CD_diagnostics/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "diagnostics.json"
FIGDIR = HERE / "figures"
MODELS = ["A", "B", "C"]
LABELS = ["A\nlocal-rule", "B\nBPTT CE_D", "C\nBPTT +align"]
COL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def main():
    d = json.loads(RESULTS.read_text())
    M = d["models"]

    def mean(k):
        return [M[m]["mean"][k] for m in MODELS]

    def std(k):
        return [M[m]["std"][k] for m in MODELS]

    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x = np.arange(3)
    cols = [COL[m] for m in MODELS]

    # 1: probe acc C vs D
    ax = axes[0, 0]
    ax.bar(x - 0.2, mean("probe_C"), 0.4, yerr=std("probe_C"), capsize=3, label="C", color="#999")
    ax.bar(x + 0.2, mean("probe_D"), 0.4, yerr=std("probe_D"), capsize=3, label="D", color="#333")
    ax.set_title("1,2: probe accuracy on C and D"); ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.axhline(0.46, ls=":", color="gray"); ax.legend(fontsize=8)

    # 2: flip rate + overlap
    ax = axes[0, 1]
    ax.bar(x - 0.2, mean("flip_rate"), 0.4, yerr=std("flip_rate"), capsize=3, label="C-D flip rate", color="#C44")
    ax.bar(x + 0.2, mean("overlap_CD"), 0.4, yerr=std("overlap_CD"), capsize=3, label="overlap(C,D)", color="#48C")
    ax.set_title("3,4: C-D flip rate & overlap"); ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.legend(fontsize=8)

    # 3: C stability under free dynamics
    ax = axes[0, 2]
    ax.bar(x - 0.2, mean("C_free_overlap"), 0.4, yerr=std("C_free_overlap"), capsize=3,
           label="overlap(C, free(C))", color="#7A4")
    ax.bar(x + 0.2, mean("C_free_flip_rate"), 0.4, yerr=std("C_free_flip_rate"), capsize=3,
           label="flip rate under free", color="#A47")
    ax.set_title("9: C stability under free dynamics"); ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.legend(fontsize=8)

    # 4: margins flipped vs stable
    ax = axes[1, 0]
    ax.bar(x - 0.2, mean("margin_flipped"), 0.4, yerr=std("margin_flipped"), capsize=3, label="flipped", color="#E88")
    ax.bar(x + 0.2, mean("margin_stable"), 0.4, yerr=std("margin_stable"), capsize=3, label="stable", color="#4A8")
    ax.set_title("6: margin (C·field_C) flipped vs stable"); ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.legend(fontsize=8)

    # 5: correlations
    ax = axes[1, 1]
    w = 0.2
    ax.bar(x - 1.5 * w, mean("corr_readout_flip"), w, yerr=std("corr_readout_flip"), capsize=2, label="readout·flip", color="#88C")
    ax.bar(x - 0.5 * w, mean("corr_field_flip"), w, yerr=std("corr_field_flip"), capsize=2, label="|field|·flip", color="#C58")
    ax.bar(x + 0.5 * w, mean("corr_readout_flip_random"), w, label="readout·flip (rand)", color="#CCD")
    ax.bar(x + 1.5 * w, mean("corr_field_flip_random"), w, label="|field|·flip (rand)", color="#ECC")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("5,8: importance-vs-flip correlation (+random control)")
    ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.legend(fontsize=7)

    # 6: |field| flipped vs random (control)
    ax = axes[1, 2]
    ax.bar(x - 0.2, mean("mean_imp_field_flipped"), 0.4, yerr=std("mean_imp_field_flipped"), capsize=3, label="really flipped", color="#C66")
    ax.bar(x + 0.2, mean("mean_imp_field_random"), 0.4, yerr=std("mean_imp_field_random"), capsize=3, label="random (matched count)", color="#66C")
    ax.set_title("8: |field| of flipped vs random spins"); ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.legend(fontsize=8)

    fig.suptitle("C/D diagnostics — A local-rule vs B BPTT CE_D vs C BPTT CE_D+align "
                 "(best_channel_entropy, 3 seeds)", fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "diagnostics.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
