"""experiments/3-Wout_C_vs_D_random/plot.py

Plot the W_out-on-C-vs-D results (trained vs random backbone).
Reads results/wout_c_vs_d_random.json, writes figures/comparison.png.

Run:  python experiments/3-Wout_C_vs_D_random/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROBE_CEILING = 0.46  # linear-probe reference on D


def mean_std(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    data = json.load(open(HERE / "results" / "wout_c_vs_d_random.json"))
    results = data["results"]
    configs = list(dict.fromkeys(r["config"] for r in results))  # preserve order
    modes = data["modes"]

    # lookup[(config, mode)] = (C_mean, C_std, D_mean, D_std)
    lookup = {}
    for r in results:
        cm_, cs = mean_std(r["per_seed"], "wout_C")
        dm, ds_ = mean_std(r["per_seed"], "wout_D")
        lookup[(r["config"], r["mode"])] = (cm_, cs, dm, ds_)

    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.5), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    x = np.arange(len(configs))
    width = 0.38
    for ax, mode in zip(axes, modes):
        c_means = [lookup[(c, mode)][0] for c in configs]
        c_stds = [lookup[(c, mode)][1] for c in configs]
        d_means = [lookup[(c, mode)][2] for c in configs]
        d_stds = [lookup[(c, mode)][3] for c in configs]

        b1 = ax.bar(x - width / 2, c_means, width, yerr=c_stds, capsize=4,
                    color="#EA580C", label="W_out on C")
        b2 = ax.bar(x + width / 2, d_means, width, yerr=d_stds, capsize=4,
                    color="#16A34A", label="W_out on D")
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.3f}", ha="center", fontsize=8)

        ax.axhline(PROBE_CEILING, color="grey", ls=":", lw=1.3,
                   label=f"probe ceiling (~{PROBE_CEILING:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
        ax.set_title(f"{mode} backbone")
        ax.set_ylabel("Test accuracy (eval on D)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"W_out trained on C vs D — {len(data['seeds'])} seeds "
        f"(strong→weak W_back: best_channel_entropy > matei_cgf > matei_W_out)",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "figures" / "comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
