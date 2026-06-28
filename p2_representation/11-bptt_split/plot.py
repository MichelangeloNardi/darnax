"""Regenerate exp-11 figures from results/diagnostics.json (figures/ git-ignored).
  figures/global.png — global metrics per model
  figures/groups.png — per-group probe_D per model
Run: python p2_representation/11-bptt_split/plot.py
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
FIG = HERE / "figures"


def main():
    p = RES / "diagnostics.json"
    if not p.exists():
        return
    d = json.loads(p.read_text()); m = d["models"]; tags = list(m)
    x = np.arange(len(tags))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    a = axes[0]
    for i, k in enumerate(["probe_D", "head_acc_D", "probe_C_transfer_D"]):
        a.bar(x + (i - 1) * 0.27, [m[t]["mean"][k] for t in tags], 0.27,
              yerr=[m[t]["std"][k] for t in tags], capsize=2, label=k)
    a.set_title("probe_D / head_D / transfer"); a.legend(fontsize=8)

    a = axes[1]
    for i, k in enumerate(["flip_rate", "overlap_CD"]):
        a.bar(x + (i - 0.5) * 0.35, [m[t]["mean"][k] for t in tags], 0.35,
              yerr=[m[t]["std"][k] for t in tags], capsize=2, label=k)
    a.set_title("C->D geometry"); a.legend(fontsize=8)

    # per-group probe_D (variable group names per model)
    a = axes[2]
    width = 0.8
    for ti, t in enumerate(tags):
        ga = m[t]["group_agg"]; gn = [g for g in ga if ga[g] is not None]
        n = len(gn)
        for gi, g in enumerate(gn):
            off = (gi - (n - 1) / 2) * (width / max(n, 1))
            v = ga[g]["probe_D_group"]["mean"]
            a.bar(ti + off, v, width / max(n, 1),
                  color=plt.cm.tab10(gi), label=g if ti == 0 else None)
            a.text(ti + off, v + 0.01, g[:3], ha="center", va="bottom", fontsize=6, rotation=90)
    a.axhline(0.10, ls="--", c="grey", lw=1, label="chance")
    a.set_title("per-group probe_D (decode at D)"); a.legend(fontsize=7)

    for a in axes:
        a.set_xticks(x); a.set_xticklabels(tags, rotation=35, ha="right", fontsize=7)
    fig.suptitle("Exp 11 — BPTT split diagnostic + partial-overlap split", fontsize=13)
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "global.png", dpi=130); print("wrote", FIG / "global.png")


if __name__ == "__main__":
    main()
