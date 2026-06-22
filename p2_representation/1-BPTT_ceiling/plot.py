"""Plot the BPTT ceiling results.

Left  : bar chart of the hard-sign ceiling (W_out perceptron + Adam probe) for each
        surrogate, vs the gradient-free p1 reference. Error bars = std over seeds.
Right : per-epoch training diagnostic for the tanh arm (one seed) — the soft-rollout
        accuracy and the hard-sign separability proxy `sep`, with the selected best
        checkpoint marked. Shows the over-sharpening as beta is annealed up.

Usage:  python p2_representation/1-BPTT_ceiling/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "ceiling.json"
FIGDIR = HERE / "figures"

# gradient-free reference (p1_readout_gap, best_channel_entropy, eval on D)
GF_WOUT, GF_PROBE = 0.44, 0.46


def main():
    d = json.loads(RESULTS.read_text())
    surrs = d["surrogates"]
    order = [k for k in ("tanh", "ste") if k in surrs]

    FIGDIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── left: ceiling bars ────────────────────────────────────────────────────
    labels, wout_m, wout_s, probe_m, probe_s = [], [], [], [], []
    for k in order:
        labels.append(f"BPTT\n({k})")
        wout_m.append(surrs[k]["wout_mean"]); wout_s.append(surrs[k]["wout_std"])
        probe_m.append(surrs[k]["probe_mean"]); probe_s.append(surrs[k]["probe_std"])

    x = np.arange(len(labels))
    w = 0.36
    ax1.bar(x - w / 2, wout_m, w, yerr=wout_s, capsize=4, label="W_out (perceptron)",
            color="#4C72B0")
    ax1.bar(x + w / 2, probe_m, w, yerr=probe_s, capsize=4, label="probe (Adam)",
            color="#DD8452")
    ax1.axhline(GF_PROBE, ls="--", lw=1.4, color="#DD8452",
                label=f"grad-free probe ~{GF_PROBE:.2f}")
    ax1.axhline(GF_WOUT, ls="--", lw=1.4, color="#4C72B0",
                label=f"grad-free W_out ~{GF_WOUT:.2f}")
    for xi, (wm, pm) in enumerate(zip(wout_m, probe_m)):
        ax1.text(xi - w / 2, wm + 0.012, f"{wm:.3f}", ha="center", fontsize=8)
        ax1.text(xi + w / 2, pm + 0.012, f"{pm:.3f}", ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Test accuracy on D (hard sign)")
    ax1.set_ylim(0, 0.62)
    ax1.set_title("BPTT ceiling vs gradient-free (best_channel_entropy, 3 seeds)")
    ax1.legend(fontsize=8, loc="upper right")

    # ── right: tanh training diagnostic (seed 0) ───────────────────────────────
    ps = surrs["tanh"]["per_seed"][0]
    ep = np.arange(1, len(ps["sep_curve"]) + 1)
    ax2.plot(ep, ps["soft_acc_curve"], "-o", ms=3, color="#DD8452",
             label="soft-rollout acc (test)")
    ax2.plot(ep, ps["sep_curve"], "-s", ms=3, color="#55A868",
             label="hard-sep proxy (ridge, train)")
    be = ps["best_epoch"]
    ax2.axvline(be, color="k", ls=":", lw=1.4, label=f"best ckpt (ep {be})")
    ax2.set_xlabel("BPTT epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("tanh seed 0 — checkpoint selection (beta annealed 1→4)")
    ax2.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    out = FIGDIR / "ceiling.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
