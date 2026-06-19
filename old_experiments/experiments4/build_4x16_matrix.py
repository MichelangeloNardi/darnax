"""experiments4/build_4x16_matrix.py

Build the 4×16 matrix the user actually wanted:
  rows    : A_none, B_none, C_none, D_none           (baseline ABCD, no update)
  columns : the 16 points (4 baseline + 4 win + 4 j + 4 both)
  values  : dot-product overlap

Generate it at multiple checkpoints to show the saturation progression.
All in dot product (no match-rate).
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
MEET = HERE / "meeting"


def match_to_dot(m):
    return 2.0 * np.asarray(m) - 1.0


def build():
    with open(RES / "fixedpoint_overlap_v2.json") as f:
        d = json.load(f)

    # Checkpoints to show: random init, early, mid, saturated
    checkpoints = [0, 1, 5, 25, 100]
    col_labels = ["A", "B", "C", "D",
                  "A'_W_in", "B'_W_in", "C'_W_in", "D'_W_in",
                  "A'_J",    "B'_J",    "C'_J",    "D'_J",
                  "A'_both", "B'_both", "C'_both", "D'_both"]
    row_labels = ["A", "B", "C", "D"]

    fig, axes = plt.subplots(1, len(checkpoints),
                             figsize=(5.5 * len(checkpoints), 4.0),
                             sharey=True)

    # Common colour scale across panels so they're directly comparable
    submats = []
    for cp in checkpoints:
        M = np.array(d[str(cp)]["match"])
        sub = match_to_dot(M[:4, :])    # 4 rows × 16 cols
        submats.append(sub)
    # vmin: min off-diagonal value across all checkpoints (excluding diagonal)
    diag_mask = np.eye(4, 16, dtype=bool)
    all_offdiag = np.concatenate([s[~diag_mask].ravel() for s in submats])
    vmin = float(all_offdiag.min())
    print(f"vmin (off-diagonal): {vmin:.3f}")

    for ax, cp, sub in zip(axes, checkpoints, submats):
        im = ax.imshow(sub, vmin=vmin - 0.01, vmax=1.0, cmap="viridis",
                       aspect="auto")
        for i in range(4):
            for j in range(16):
                v = sub[i, j]
                colour = "white" if v < (vmin + 1.0) / 2 + 0.05 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=colour, fontsize=7)
        # Block dividers (between baseline / win / j / both)
        for x in (3.5, 7.5, 11.5):
            ax.axvline(x, color="white", lw=1.2)
        ax.set_xticks(range(16))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(f"batch {cp}", fontsize=11)

    axes[0].set_ylabel("baseline phase")
    fig.suptitle(
        "4×16 dot-product matrix:  baseline ABCD (no update) ↔ ABCD after each K=1 update",
        fontsize=12,
    )
    # Shared colorbar
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="dot product")
    out = MEET / "exp1_matrix_4x16.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    build()
