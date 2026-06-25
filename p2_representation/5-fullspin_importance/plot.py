"""Plot the full-spin importance results (exp 5).

Usage:  python p2_representation/5-fullspin_importance/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "fullspin_importance.json"
FIGDIR = HERE / "figures"
MODELS = ["A", "B", "C"]
LAB = {"A": "A local-rule", "B": "B BPTT CE_D", "C": "C BPTT +align"}
COL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def main():
    d = json.loads(RESULTS.read_text())
    M = d["models"]
    x = np.arange(3)
    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    def m(k):
        return [M[z]["mean"][k] for z in MODELS]

    # 1: the concentration test — C-probe acc on C, D, random-flip controls
    a = ax[0, 0]
    w = 0.2
    a.bar(x - 1.5 * w, m("probe_C_on_C"), w, label="C (actual)", color="#777")
    a.bar(x - 0.5 * w, m("probe_C_on_D"), w, label="D (actual flips)", color="#C33")
    a.bar(x + 0.5 * w, m("acc_rand_uniform"), w, label="C, uniform random flips", color="#39C")
    a.bar(x + 1.5 * w, m("acc_rand_empirical"), w, label="C, empirical-rate flips", color="#9C3")
    a.set_title("C-probe accuracy: actual D vs random-flip controls\n(D << random  =>  flips are concentrated)")
    a.set_xticks(x); a.set_xticklabels([LAB[z] for z in MODELS], fontsize=8)
    a.set_ylabel("accuracy"); a.legend(fontsize=8); a.set_ylim(0, 1.02)

    # 2: flip rate by importance decile
    a = ax[0, 1]
    for z in MODELS:
        dec = M[z]["flip_rate_by_decile_mean"]
        a.plot(np.arange(1, 11), np.array(dec) / dec[0], "-o", ms=3, color=COL[z], label=LAB[z])
    a.set_title("flip rate by importance decile (normalized to decile 1)")
    a.set_xlabel("importance decile (10 = most important)"); a.set_ylabel("relative flip rate")
    a.legend(fontsize=8)

    # 3: correlations + enrichment
    a = ax[1, 0]
    a.bar(x - w, m("corr_flip_importance"), w, label="corr(flip, importance)", color="#88C")
    a.bar(x, m("corr_flip_damage"), w, label="corr(flip, damage)", color="#C58")
    a.axhline(0, color="k", lw=0.6)
    a2 = a.twinx()
    a2.plot(x, m("top5pct_enrichment"), "k^--", label="top-5% enrichment")
    a2.axhline(1.0, color="gray", lw=0.6, ls=":")
    a2.set_ylabel("top-5% enrichment (×)")
    a.set_title("per-spin: flip↔importance / flip↔damage / top-5% enrichment")
    a.set_xticks(x); a.set_xticklabels([LAB[z] for z in MODELS], fontsize=8)
    a.legend(fontsize=8, loc="upper left")

    # 4: |field| flipped vs stable
    a = ax[1, 1]
    a.bar(x - 0.2, m("absfield_flipped"), 0.4, label="flipped", color="#E88")
    a.bar(x + 0.2, m("absfield_stable"), 0.4, label="stable", color="#4A8")
    a.set_title("|field_C| flipped vs stable (flips hit weakly-pinned spins)")
    a.set_xticks(x); a.set_xticklabels([LAB[z] for z in MODELS], fontsize=8); a.legend(fontsize=8)

    fig.suptitle("Full-spin importance: are C→D flips concentrated on C's class-carrying spins? "
                 "(best_channel_entropy, 3 seeds)", fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "fullspin_importance.png"
    fig.savefig(out, dpi=120); print(f"saved {out}")


if __name__ == "__main__":
    main()
