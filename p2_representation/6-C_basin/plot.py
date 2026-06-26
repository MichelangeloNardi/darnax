"""Plot C-basin diagnostics (exp 6). Reads results/c_basin_<model>.json.

Usage:  python p2_representation/6-C_basin/plot.py [--model A]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="A"); args = ap.parse_args()
    d = json.loads((HERE / "results" / f"c_basin_{args.model}.json").read_text())
    ps = d["per_seed"]

    def avg(path):
        out = []
        for s in ps:
            o = s
            for k in path:
                o = o[k]
            out.append(o)
        return np.mean(out, 0)

    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    # diag 1
    p = ps[0]["basin_radius"]["p"]
    a = ax[0, 0]
    a.plot(p, avg(["basin_radius", "overlap"]), "-o", ms=3, label="overlap with C")
    a.plot(p, avg(["basin_radius", "return_rate"]), "-s", ms=3, label="return rate (ov≥0.95)")
    a.plot(p, avg(["basin_radius", "probe_acc"]), "-^", ms=3, label="C-probe acc")
    a.set_title("1: C basin radius"); a.set_xlabel("% spins randomly flipped"); a.legend(fontsize=8); a.set_ylim(0, 1.02)

    # diag 2
    k = ps[0]["cd_boundary"]["k"]
    a = ax[0, 1]
    a.plot(k, avg(["cd_boundary", "overlap_C"]), "-o", ms=3, label="overlap with C")
    a.plot(k, avg(["cd_boundary", "overlap_D"]), "-o", ms=3, label="overlap with D")
    a.plot(k, avg(["cd_boundary", "return_to_C_rate"]), "-s", ms=3, label="returns to C (vs D)")
    a.set_title("2: C→D boundary"); a.set_xlabel("% of real C→D flips applied"); a.legend(fontsize=8); a.set_ylim(0, 1.02)

    # diag 4 distances
    a = ax[0, 2]
    spaces = ["hamming", "pooled", "logit"]
    x = np.arange(3); wn = [d["mean"][f"{s}_within"] for s in spaces]; bt = [d["mean"][f"{s}_between"] for s in spaces]
    # normalize each space to its between for comparability
    wn_n = [w / b for w, b in zip(wn, bt)]
    a.bar(x - 0.2, wn_n, 0.4, label="within / between", color="#C66")
    a.bar(x + 0.2, [1.0] * 3, 0.4, label="between (=1)", color="#6AC")
    a.set_title("4: C class geometry — within/between distance")
    a.set_xticks(x); a.set_xticklabels(spaces); a.legend(fontsize=8)

    # diag 3 rescue vs k at final t
    kk = ps[0]["d_rescue"]["k"]; tt = ps[0]["d_rescue"]["t"]; tlast = len(tt) - 1
    a = ax[1, 0]
    a.plot(kk, [avg(["d_rescue", "pin", str(K), "probe_acc"])[tlast] for K in kk], "-o", ms=4, label="pin")
    a.plot(kk, [avg(["d_rescue", "replace", str(K), "probe_acc"])[tlast] for K in kk], "-s", ms=4, label="replace")
    a.axhline(d["mean"]["probe_C_on_C"], ls=":", color="g", label="acc C")
    a.axhline(d["mean"]["probe_C_on_D"], ls=":", color="r", label="acc D (no rescue)")
    a.set_title("3: D rescue — C-probe acc vs k (final state D)")
    a.set_xlabel("k spins pinned/replaced to C"); a.legend(fontsize=8); a.set_ylim(0, 1.02)

    # diag 3 rescue vs t for k=128
    a = ax[1, 1]
    for K in [64, 128, 256]:
        if K in kk:
            a.plot(tt, avg(["d_rescue", "pin", str(K), "probe_acc"]), "-o", ms=3, label=f"pin k={K}")
    a.axhline(d["mean"]["probe_C_on_D"], ls=":", color="r")
    a.set_title("3: D rescue (pin) — C-probe acc by trajectory step t")
    a.set_xlabel("trajectory step t (7 = final D)"); a.legend(fontsize=8); a.set_ylim(0, 1.02)

    # diag 4 prototype nearest-class accuracy
    a = ax[1, 2]
    a.bar(x, [d["mean"][f"proto_acc_{s}"] for s in spaces],
          yerr=[d["std"][f"proto_acc_{s}"] for s in spaces], capsize=3, color="#7A6")
    a.set_title("4: prototype nearest-class accuracy")
    a.set_xticks(x); a.set_xticklabels(spaces); a.set_ylim(0, 1.02)

    fig.suptitle(f"C-basin diagnostics — model {args.model} (local rule, {len(ps)} seeds, "
                 f"{d['N_RELAX']}-step free relaxation)", fontsize=12)
    fig.tight_layout()
    out = FIGDIR / f"c_basin_{args.model}.png"
    fig.savefig(out, dpi=120); print(f"saved {out}")


if __name__ == "__main__":
    main()
