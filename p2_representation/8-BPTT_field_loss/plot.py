"""Plot BPTT field-loss results (exp 8). Reads results/field_loss.json.

Usage:  python p2_representation/8-BPTT_field_loss/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "field_loss.json"
FIGDIR = HERE / "figures"
BPTT_PROBE = 0.506
TEACHER_CTRANS = 0.26
TEACHER_ACC_C = 0.96


def main():
    d = json.loads(RESULTS.read_text())
    cells, labels = [], []
    for exp in ["exp1_endpoint", "exp2_trajectory"]:
        for g in sorted(d["experiments"][exp], key=float):
            cells.append(d["experiments"][exp][g])
            labels.append(f"{'E1' if '1' in exp else 'E2'}\nγ={g}")
    x = np.arange(len(cells))

    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    def m(key):
        return [c["mean"][key] for c in cells]

    def s(key):
        return [c["std"][key] for c in cells]

    a = ax[0]
    w = 0.27
    a.bar(x - w, m("probe_acc"), w, yerr=s("probe_acc"), capsize=3, label="D probe acc", color="#4C72B0")
    a.bar(x, m("wout_acc"), w, label="D W_out acc", color="#DD8452")
    a.bar(x + w, m("cprobe_transfer_D"), w, label="teacher C-probe on D", color="#55A868")
    a.axhline(BPTT_PROBE, ls="--", color="crimson", lw=1.4, label=f"plain BPTT probe {BPTT_PROBE}")
    a.axhline(TEACHER_CTRANS, ls=":", color="gray", lw=1.4, label=f"teacher C-on-D {TEACHER_CTRANS}")
    a.axhline(TEACHER_ACC_C, ls=":", color="green", lw=1.2, label=f"teacher acc_C {TEACHER_ACC_C}")
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylabel("accuracy"); a.set_ylim(0, 1.0)
    a.set_title("Accuracy: D stays at the BPTT ceiling, no fall into C basin")
    a.legend(fontsize=7, ncol=2)

    a = ax[1]
    a.bar(x - w, m("overlap_D_CA"), w, label="overlap(D, C_A)", color="#8172B3")
    a.bar(x, m("field_margin_on_M"), w, label="field margin C_A·h on M", color="#C44E52")
    a.bar(x + w, m("flip_rate_on_M"), w, label="flip rate on M", color="#937860")
    a.axhline(1.0, ls=":", color="r", lw=1.0, label="κ target (=1)")
    a.set_xticks(x); a.set_xticklabels(labels)
    a.set_title("Mechanism on top-k spins M: field weakly tilted, ~43% still flip")
    a.legend(fontsize=8)

    fig.suptitle("BPTT teacher-field loss — does supporting C_A's class spins pull D into the C basin? (3 seeds)",
                 fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "field_loss.png"
    fig.savefig(out, dpi=120); print(f"saved {out}")


if __name__ == "__main__":
    main()
