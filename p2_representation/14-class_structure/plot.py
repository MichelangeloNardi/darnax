"""Exp 14 figures — class structure of the D representation.

  figures/class_structure.png
    (a) per-class accuracy on D, probe vs W_out head, vehicles and animals separated
    (b) row-normalised 10x10 confusion matrix on D, classes reordered vehicles-first
    (c) class-centroid cosine similarity on D, same ordering
    (d) the 511-partition scan: D vs the stronger of the two controls

Run:  python p2_representation/14-class_structure/plot.py
      python p2_representation/14-class_structure/plot.py --results results/xyz.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent

# categorical hues (validated: lightness band, chroma floor, CVD dE 10.9 protan,
# normal-vision dE 24.2, contrast >= 3:1 on a light surface)
BLUE, ORANGE, GREEN = "#2a6fd4", "#d2660b", "#0e8f77"
INK, MUTED, GRID = "#1b1f24", "#5c656d", "#dcdcd4"

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f7fc", BLUE, "#0f2f5e"])
DIV = LinearSegmentedColormap.from_list("div_bo", [BLUE, "#eceae6", ORANGE])

VEHICLES = [0, 1, 8, 9]
ANIMALS = [2, 3, 4, 5, 6, 7]
ORDER = VEHICLES + ANIMALS


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "grid.color": GRID, "grid.linewidth": 0.6,
    })


def panel_per_class(ax, res, classes):
    """(a) per-class accuracy, probe vs head, mean +- std over seeds."""
    seeds = list(res["seeds"].values())
    probe = np.array([[s["D"]["tenway"]["per_class"][classes[c]] for c in ORDER]
                      for s in seeds])
    head = np.array([[s["head_D"]["per_class"][classes[c]] for c in ORDER] for s in seeds])
    x = np.arange(10)
    w = 0.38
    kw = dict(width=w, edgecolor="white", linewidth=0.8)
    ax.bar(x - w / 2, probe.mean(0), yerr=probe.std(0) if len(seeds) > 1 else None,
           color=BLUE, label="Adam probe", error_kw=dict(lw=0.9, ecolor=MUTED), **kw)
    ax.bar(x + w / 2, head.mean(0), yerr=head.std(0) if len(seeds) > 1 else None,
           color=ORANGE, label="W_out head", error_kw=dict(lw=0.9, ecolor=MUTED), **kw)

    ax.axvline(3.5, color=MUTED, lw=1, ls="-", alpha=0.55)
    ax.axhline(0.1, color=MUTED, lw=0.9, ls=":")
    ax.text(0.015, 0.105, "chance", transform=ax.get_yaxis_transform(),
            fontsize=7.5, color=MUTED, va="bottom")
    top = max(probe.mean(0).max(), head.mean(0).max())
    ax.text(1.5, top * 1.27, "VEHICLES", ha="center", fontsize=8, color=MUTED,
            fontweight="semibold")
    ax.text(6.5, top * 1.27, "ANIMALS", ha="center", fontsize=8, color=MUTED,
            fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels([classes[c] for c in ORDER], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("test accuracy on D")
    ax.set_ylim(0, top * 1.36)
    ax.set_title("(a) Per-class accuracy")
    ax.grid(axis="y", alpha=0.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def panel_confusion(ax, res, classes, fig):
    """(b) row-normalised confusion, reordered so the two groups form blocks."""
    seeds = list(res["seeds"].values())
    M = np.mean([np.asarray(s["D"]["tenway"]["confusion"], float) for s in seeds], axis=0)
    M = M / M.sum(1, keepdims=True)
    M = M[np.ix_(ORDER, ORDER)]
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=M.max())
    for i in range(10):
        for j in range(10):
            if M[i, j] >= 0.02:
                ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=6.6,
                        color="white" if M[i, j] > 0.55 * M.max() else INK)
    for p in (3.5,):
        ax.axhline(p, color=ORANGE, lw=1.6)
        ax.axvline(p, color=ORANGE, lw=1.6)
    lab = [classes[c] for c in ORDER]
    ax.set_xticks(range(10)); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(10)); ax.set_yticklabels(lab, fontsize=7.5)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    t = np.mean([s["D"]["tenway"]["frac_errors_within_group"] for s in seeds])
    e = np.mean([s["D"]["tenway"]["expected_frac_uniform"] for s in seeds])
    ax.set_title(f"(b) Confusion on D  ·  {t:.0%} of errors stay inside the group "
                 f"(uniform: {e:.0%})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("row share", fontsize=8)


def panel_centroids(ax, res, classes, fig):
    """(c) class-centroid cosine similarity — diverging, symmetric, gray midpoint."""
    S = np.asarray(res["seeds"][list(res["seeds"])[0]]["D"]["centroid_sim"], float)
    S = S[np.ix_(ORDER, ORDER)]
    off = S.copy(); np.fill_diagonal(off, np.nan)
    v = float(np.nanmax(np.abs(off)))
    cmap = DIV.copy(); cmap.set_bad("#f4f3f0")
    im = ax.imshow(off, cmap=cmap, vmin=-v, vmax=v)
    ax.axhline(3.5, color=INK, lw=1.4); ax.axvline(3.5, color=INK, lw=1.4)
    lab = [classes[c] for c in ORDER]
    ax.set_xticks(range(10)); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(10)); ax.set_yticklabels(lab, fontsize=7.5)
    ax.set_title("(c) Class-centroid cosine similarity on D")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("cosine", fontsize=8)


def panel_scan(ax, res, named):
    """(d) all 511 dichotomies: D vs the better control. Above the line = D adds signal."""
    seeds = list(res["seeds"].values())
    d_by, r_by = {}, {}
    for s in seeds:
        for x in s["D"]["scan"]:
            d_by.setdefault(tuple(x["classes"]), []).append(x["balanced"])
        for x in s["randD"]["scan"]:
            r_by.setdefault(tuple(x["classes"]), []).append(x["balanced"])
    p_by = {tuple(x["classes"]): x["balanced"] for x in res["pixels"]["scan"]}
    keys = list(d_by)
    D = np.array([np.mean(d_by[k]) for k in keys])
    ctrl = np.array([max(np.mean(r_by[k]), p_by[k]) for k in keys])

    ax.scatter(ctrl, D, s=9, color=MUTED, alpha=0.30, linewidths=0, label="all 511 dichotomies")
    lo = min(ctrl.min(), D.min()) - 0.02
    hi = max(ctrl.max(), D.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", alpha=0.6)
    ax.text(hi, hi, "  D = control", fontsize=7.5, color=MUTED, va="center")

    lift = D - ctrl
    top = [i for i in np.argsort(-lift)[:5] if lift[i] > 0.005]
    if top:
        ax.scatter(ctrl[top], D[top], s=52, facecolor="none", edgecolor=GREEN,
                   linewidths=1.6, label="largest lift over control", zorder=3)
        for r, i in enumerate(top):
            ax.annotate("+".join(res["classes"][c] for c in keys[i]),
                        (ctrl[i], D[i]), textcoords="offset points",
                        xytext=(9, 7 - 13 * (r % 3)), fontsize=7, color=INK)

    idx = {k: i for i, k in enumerate(keys)}
    for j, (nm, cls) in enumerate(named.items()):
        k = tuple(sorted(cls))
        if k in idx:
            i = idx[k]
            ax.scatter([ctrl[i]], [D[i]], s=58, color=ORANGE, zorder=4,
                       edgecolor="white", linewidths=1.1,
                       label="named grouping" if nm == list(named)[0] else None)
            ax.annotate(nm, (ctrl[i], D[i]), textcoords="offset points",
                        xytext=(9, 9 + 11 * (j % 2)),
                        fontsize=7.5, color=INK, fontweight="semibold")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("balanced accuracy, best control (random-init D or raw pixels)")
    ax.set_ylabel("balanced accuracy on D")
    ax.set_title("(d) Which dichotomies does D encode beyond the controls?")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results/class_structure.json")
    ap.add_argument("--out", type=str, default="figures/class_structure.png")
    args = ap.parse_args()

    path = HERE / args.results if not Path(args.results).is_absolute() else Path(args.results)
    res = json.loads(Path(path).read_text())
    classes = res["classes"]
    # panel (d) scores every dichotomy on all 10 classes, so groupings that were
    # scored on a restricted subset are not comparable on those axes
    d0 = res["seeds"][list(res["seeds"])[0]]["D"]["named"]
    named = {k: set(v) for k, v in res["groupings"].items()
             if len(v) > 1 and "restricted_to" not in d0.get(k, {})}

    style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 12.5))
    panel_per_class(axes[0, 0], res, classes)
    panel_confusion(axes[0, 1], res, classes, fig)
    panel_centroids(axes[1, 0], res, classes, fig)
    panel_scan(axes[1, 1], res, named)

    n = len(res["seeds"])
    fig.suptitle(f"Class structure of the inference state D  ·  channel_entropy, "
                 f"{res['epochs']} epoch{'s' if res['epochs'] > 1 else ''}, "
                 f"{n} seed{'s' if n > 1 else ''}",
                 fontsize=13, fontweight="semibold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
