"""Regenerate figures/pc_feedback.png from the exp-6 result JSONs (png/log gitignored).

Left   : C/D probe accuracy per model (raw clamp vs PC-feedback variants).
Middle : C->D flip rate, overlap(C,D), C-free stability.
Right  : full-spin control -- C-probe under the ACTUAL C->D flips vs random flips of
         the same count. If PC makes C less label-imprinted, the gap (random - actual)
         shrinks toward the BPTT regime while the C-probe itself stays informative.

Run:  python p2_representation/6-pc_feedback/plot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)


def _m(model, k):
    return model["mean"].get(k, np.nan), model["std"].get(k, np.nan)


def main():
    diag_p = RES / "diagnostics.json"
    fs_p = RES / "fullspin_importance.json"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if diag_p.exists():
        diag = json.loads(diag_p.read_text())
        tags = diag["tags"]
        x = np.arange(len(tags))
        # left: probe acc
        ax = axes[0]
        for off, (k, lbl) in zip((-0.2, 0.2), [("probe_C", "C probe"), ("probe_D", "D probe")]):
            ms = [_m(diag["models"][t], k) for t in tags]
            ax.bar(x + off, [m for m, _ in ms], 0.4, yerr=[s for _, s in ms], label=lbl, capsize=3)
        ax.axhline(0.46, ls="--", c="grey", lw=1, label="D ceiling ~0.46")
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_ylabel("probe accuracy"); ax.set_title("C/D separability"); ax.legend()
        # middle: flip / overlap / C-free
        ax = axes[1]
        keys = [("flip_rate", "C-D flip"), ("overlap_CD", "overlap(C,D)"),
                ("C_free_overlap", "C-free overlap")]
        for off, (k, lbl) in zip((-0.27, 0.0, 0.27), keys):
            ms = [_m(diag["models"][t], k) for t in tags]
            ax.bar(x + off, [m for m, _ in ms], 0.27, yerr=[s for _, s in ms], label=lbl, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_title("C vs D geometry"); ax.legend()

    if fs_p.exists():
        fs = json.loads(fs_p.read_text())
        tags = fs["tags"]
        x = np.arange(len(tags))
        ax = axes[2]
        keys = [("probe_C_on_C", "C-probe on C"),
                ("probe_C_on_D", "on D (actual flips)"),
                ("acc_rand_uniform", "on C, random flips")]
        for off, (k, lbl) in zip((-0.27, 0.0, 0.27), keys):
            ms = [_m(fs["models"][t], k) for t in tags]
            ax.bar(x + off, [m for m, _ in ms], 0.27, yerr=[s for _, s in ms], label=lbl, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_ylabel("accuracy"); ax.set_title("full-spin flip targeting (random control)")
        ax.legend()

    fig.tight_layout()
    out = FIG / "pc_feedback.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
