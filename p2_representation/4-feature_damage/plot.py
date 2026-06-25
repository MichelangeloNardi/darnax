"""Plot the pooled-feature damage / rescue / lesion results (exp 4).

Usage:  python p2_representation/4-feature_damage/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "feature_damage.json"
FIGDIR = HERE / "figures"
MODELS = ["A", "B", "C"]
LAB = {"A": "A local-rule", "B": "B BPTT CE_D", "C": "C BPTT +align"}
COL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def main():
    d = json.loads(RESULTS.read_text())
    M = d["models"]; kg = np.array(d["k_grid"])
    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # rescue
    a = ax[0, 0]
    for m in MODELS:
        a.plot(kg, M[m]["rescue_mean"], "-o", ms=3, color=COL[m], label=LAB[m])
        a.axhline(M[m]["mean"]["acc_C"], ls=":", lw=0.8, color=COL[m])
    a.set_title("Rescue: C-probe acc on D with top-k damaged features restored to C")
    a.set_xlabel("k (pooled features restored, of 256)"); a.set_ylabel("accuracy"); a.legend(fontsize=8)
    a.set_ylim(0, 1.02)

    # lesion
    a = ax[0, 1]
    for m in MODELS:
        a.plot(kg, M[m]["lesion_mean"], "-o", ms=3, color=COL[m], label=LAB[m])
    a.set_title("Lesion: C-probe acc on C with top-k damaged features corrupted to D")
    a.set_xlabel("k (pooled features corrupted, of 256)"); a.set_ylabel("accuracy"); a.legend(fontsize=8)
    a.set_ylim(0, 1.02)

    x = np.arange(3); cols = [COL[m] for m in MODELS]
    # corr(importance, C->D change)
    a = ax[1, 0]
    a.bar(x - 0.2, [M[m]["mean"]["corr_imp_change_feat"] for m in MODELS], 0.4,
          yerr=[M[m]["std"]["corr_imp_change_feat"] for m in MODELS], capsize=3, label="feature-level", color="#88C")
    a.bar(x + 0.2, [M[m]["mean"]["corr_imp_change_fe"] for m in MODELS], 0.4,
          yerr=[M[m]["std"]["corr_imp_change_fe"] for m in MODELS], capsize=3, label="per (feature,example)", color="#C58")
    a.axhline(0, color="k", lw=0.6)
    a.set_title("corr( probe importance , C→D feature change )")
    a.set_xticks(x); a.set_xticklabels([LAB[m] for m in MODELS], fontsize=8); a.legend(fontsize=8)

    # acc_C / acc_D and total logit damage
    a = ax[1, 1]
    a.bar(x - 0.2, [M[m]["mean"]["acc_C"] for m in MODELS], 0.4, label="acc C (C-probe)", color="#777")
    a.bar(x + 0.2, [M[m]["mean"]["acc_D"] for m in MODELS], 0.4, label="acc D (C-probe)", color="#333")
    a.set_ylabel("accuracy"); a.set_xticks(x); a.set_xticklabels([LAB[m] for m in MODELS], fontsize=8)
    a2 = a.twinx()
    a2.plot(x, [M[m]["mean"]["total_logit_damage"] for m in MODELS], "r^-", label="total logit damage")
    a2.set_ylabel("total correct-class logit damage", color="r")
    a.set_title("C-probe acc on C vs D  +  total logit damage")
    a.legend(fontsize=8, loc="upper right")

    fig.suptitle("Pooled-feature damage / rescue / lesion (C-probe; best_channel_entropy, 3 seeds)", fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "feature_damage.png"
    fig.savefig(out, dpi=120); print(f"saved {out}")


if __name__ == "__main__":
    main()
