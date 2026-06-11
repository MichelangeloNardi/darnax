"""experiments7/merge_and_plot.py

Merge sweep.json (seed 0) + sweep_extra.json (seeds 42, 123) into
sweep_merged.json / diagnostics_merged.json, recompute means/stds,
and regenerate all figures.

Run locally after pulling both result files from the cluster:
  uv run python experiments7/merge_and_plot.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────

with open(RESULTS / "sweep.json") as f:
    s0 = json.load(f)
with open(RESULTS / "sweep_extra.json") as f:
    sx = json.load(f)
with open(RESULTS / "diagnostics.json") as f:
    d0 = json.load(f)
with open(RESULTS / "diagnostics_extra.json") as f:
    dx = json.load(f)

NOISE_VALUES = s0["noise_values"]
EPOCHS       = s0["epochs"]
OFFLINE_EPOCHS = s0["offline_epochs"]

assert s0["noise_values"] == sx["noise_values"], "noise_values mismatch"

# ── merge ─────────────────────────────────────────────────────────────────────

def merge_sweep(base, extra):
    merged_per_noise = []
    for r0, rx in zip(base["per_noise"], extra["per_noise"]):
        assert abs(r0["noise_std"] - rx["noise_std"]) < 1e-9
        per_seed = r0["per_seed"] + rx["per_seed"]
        all_seeds = [s0["seed"] for s0 in base["per_noise"][0]["per_seed"]] + \
                    [sx["seed"] for sx in extra["per_noise"][0]["per_seed"]]

        head_mat    = np.array([s["head_accs"]        for s in per_seed])
        probe_mat   = np.array([s["probe_accs"]       for s in per_seed])
        offline_arr = np.array([s["offline_head_acc"] for s in per_seed])

        merged_per_noise.append({
            "noise_std":         r0["noise_std"],
            "per_seed":          per_seed,
            "head_mean":         head_mat.mean(0).tolist(),
            "head_std":          head_mat.std(0).tolist(),
            "probe_mean":        probe_mat.mean(0).tolist(),
            "probe_std":         probe_mat.std(0).tolist(),
            "offline_head_mean": float(offline_arr.mean()),
            "offline_head_std":  float(offline_arr.std()),
        })

    all_seeds = ([s["seed"] for s in base["per_noise"][0]["per_seed"]] +
                 [s["seed"] for s in extra["per_noise"][0]["per_seed"]])
    return {
        "noise_values":   NOISE_VALUES,
        "seeds":          all_seeds,
        "epochs":         EPOCHS,
        "offline_epochs": OFFLINE_EPOCHS,
        "per_noise":      merged_per_noise,
    }


def merge_diag(base, extra):
    merged_per_noise = []
    for r0, rx in zip(base["per_noise"], extra["per_noise"]):
        assert abs(r0["noise_std"] - rx["noise_std"]) < 1e-9
        merged_per_noise.append({
            "noise_std": r0["noise_std"],
            "per_seed":  r0["per_seed"] + rx["per_seed"],
        })
    all_seeds = ([s["seed"] for s in base["per_noise"][0]["per_seed"]] +
                 [s["seed"] for s in extra["per_noise"][0]["per_seed"]])
    return {
        "noise_values": NOISE_VALUES,
        "seeds":        all_seeds,
        "epochs":       EPOCHS,
        "warmup_n":     base["warmup_n"],
        "clamped_n":    base["clamped_n"],
        "free_n":       base["free_n"],
        "per_noise":    merged_per_noise,
    }


merged_sweep = merge_sweep(s0, sx)
merged_diag  = merge_diag(d0, dx)

(RESULTS / "sweep_merged.json").write_text(json.dumps(merged_sweep, indent=2))
(RESULTS / "diagnostics_merged.json").write_text(json.dumps(merged_diag, indent=2))
print(f"Saved sweep_merged.json  ({len(merged_sweep['seeds'])} seeds)")
print(f"Saved diagnostics_merged.json")

# ── summary table ─────────────────────────────────────────────────────────────

n_seeds = len(merged_sweep["seeds"])
print(f"\nMerged results ({n_seeds} seeds: {merged_sweep['seeds']})")
print(f"\n{'Noise':>7}  {'Head(final)':>14}  {'Probe(final)':>14}  {'Offline Wout':>14}")
print("-" * 58)
for r in merged_sweep["per_noise"]:
    print(f"  {r['noise_std']:>5.2f}"
          f"  {r['head_mean'][-1]:.4f} ± {r['head_std'][-1]:.4f}"
          f"  {r['probe_mean'][-1]:.4f} ± {r['probe_std'][-1]:.4f}"
          f"  {r['offline_head_mean']:.4f} ± {r['offline_head_std']:.4f}")

# ── replot ─────────────────────────────────────────────────────────────────────
# Import plot_results from noise_sweep and run it on merged data.

sys.path.insert(0, str(HERE))
from noise_sweep import plot_results, EPOCHS as EP

# Reconstruct the all_results list that plot_results expects
all_results = []
for rn, rd in zip(merged_sweep["per_noise"], merged_diag["per_noise"]):
    # Attach per_epoch_diag back onto per_seed entries
    per_seed_full = []
    for sn, sd in zip(rn["per_seed"], rd["per_seed"]):
        per_seed_full.append({**sn, "per_epoch_diag": sd["per_epoch_diag"]})
    all_results.append({**rn, "per_seed": per_seed_full})

print("\nRegenerating figures...")
plot_results(all_results, FIGURES)
print("Done.")
