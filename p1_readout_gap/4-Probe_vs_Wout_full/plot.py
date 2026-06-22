"""experiments/4-Probe_vs_Wout_full/plot.py

Plot the 8-cell probe-vs-W_out comparison (final-epoch value).
Reads results/probe_vs_wout.json, writes figures/probe_vs_wout.png.
Two rows: eval on D (valid) and eval on C (cheating). x-axis = 8 cells, bars per config.

Run:  python experiments/4-Probe_vs_Wout_full/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROBE_CEILING = 0.46


def main():
    data = json.load(open(HERE / "results" / "probe_vs_wout.json"))
    cells = data["cells"]
    results = data["results"]
    configs = [r["config"] for r in results]
    seeds = data["seeds"]

    # value[(config, cell, eval)] = (mean, std) over seeds, final epoch
    def stat(r, cell, ev):
        vals = [s["curves"][cell][ev][-1] for s in r["per_seed"]]
        return float(np.mean(vals)), float(np.std(vals))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(cells))
    width = 0.8 / len(configs)
    colors = plt.cm.tab10(np.linspace(0, 0.3, len(configs)))

    for ax, ev, title in [(axes[0], "eval_D", "Evaluated on D (valid — inference state)"),
                          (axes[1], "eval_C", "Evaluated on C (cheating — is C separable?)")]:
        for ci, r in enumerate(results):
            means = [stat(r, c, ev)[0] for c in cells]
            stds = [stat(r, c, ev)[1] for c in cells]
            off = (ci - (len(configs) - 1) / 2) * width
            bars = ax.bar(x + off, means, width, yerr=stds, capsize=3,
                          color=colors[ci], label=r["config"])
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, m + 0.008, f"{m:.2f}",
                        ha="center", fontsize=6.5, rotation=90)
        if ev == "eval_D":
            ax.axhline(PROBE_CEILING, color="grey", ls=":", lw=1.3,
                       label=f"probe ref (~{PROBE_CEILING:.2f})")
        ax.set_ylabel("Test accuracy")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(0, 1.02)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace("_", "\n") for c in cells], fontsize=8)
    fig.suptitle(f"Probe vs W_out — 8 cells, final epoch, {len(seeds)} seeds", fontsize=12)
    fig.tight_layout()
    out = HERE / "figures" / "probe_vs_wout.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
