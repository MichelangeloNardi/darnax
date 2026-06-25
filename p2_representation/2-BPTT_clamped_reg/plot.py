"""Plot the clamped-distance-regularizer sweep.

Two panels (Adam probe, perceptron W_out). Each: 4 variants on the x-axis, one bar
per alpha, error bars = std over seeds. Reference lines: exp-1 plain BPTT and the
gradient-free p1 ceiling.

Usage:  python p2_representation/2-BPTT_clamped_reg/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "clamped_reg.json"
FIGDIR = HERE / "figures"

# references
BPTT_PROBE, BPTT_WOUT = 0.506, 0.480     # exp 1, plain BPTT (tanh)
GF_PROBE, GF_WOUT = 0.46, 0.44           # gradient-free (p1)

VARIANT_ORDER = ["ce_D_reg_pool", "ce_C_reg_pool", "ce_D_reg_full", "ce_C_reg_full"]


def main():
    d = json.loads(RESULTS.read_text())
    V = d["variants"]
    variants = [v for v in VARIANT_ORDER if v in V]
    alphas = [str(a) for a in d["alphas"]]

    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    panels = [("probe", "probe (Adam)", BPTT_PROBE, GF_PROBE),
              ("wout", "W_out (perceptron)", BPTT_WOUT, GF_WOUT)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))

    for ax, (key, title, bptt_ref, gf_ref) in zip(axes, panels):
        x = np.arange(len(variants))
        w = 0.8 / len(alphas)
        for ai, a in enumerate(alphas):
            m = [V[v][a][f"{key}_mean"] for v in variants]
            s = [V[v][a][f"{key}_std"] for v in variants]
            off = (ai - (len(alphas) - 1) / 2) * w
            ax.bar(x + off, m, w, yerr=s, capsize=3, color=colors[ai], label=f"α={a}")
            for xi, mi in zip(x, m):
                ax.text(xi + off, mi + 0.008, f"{mi:.2f}", ha="center", fontsize=7)
        ax.axhline(bptt_ref, ls="--", lw=1.4, color="crimson",
                   label=f"BPTT no-reg ~{bptt_ref:.2f}")
        ax.axhline(gf_ref, ls=":", lw=1.4, color="gray",
                   label=f"grad-free ~{gf_ref:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Test accuracy on D (hard sign)")
        ax.set_ylim(0, 0.62)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2, loc="lower right")

    fig.suptitle("BPTT + clamped-distance regularizer — pull D toward C "
                 "(best_channel_entropy, 3 seeds)", fontsize=11)
    fig.tight_layout()
    out = FIGDIR / "clamped_reg.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
