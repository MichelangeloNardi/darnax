"""Generate figures for experiments5 results (multi-seed format)."""
import json
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS = pathlib.Path(__file__).parent / "results"
FIGURES = pathlib.Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

std   = load("standard.json")
off   = load("ablation_offline_wout.json")
rand  = load("ablation_random_baseline.json")
jonly = load("ablation_j_only.json")

epochs = list(range(1, std["epochs"] + 1))

COLORS = {
    "standard":        "#2563EB",
    "offline_wout":    "#16A34A",
    "random_baseline": "#DC2626",
    "j_only":          "#9333EA",
}
LABELS = {
    "standard":        "Standard (Win+J1+Wout)",
    "offline_wout":    "Offline Wout (Win+J train, Wout offline perceptron)",
    "random_baseline": "Random baseline (Wout only)",
    "j_only":          "J-only (Win primes once, λ=0 after)",
}


def plot_band(ax, epochs, mean, std, color, label, ls="-"):
    m, s = np.array(mean), np.array(std)
    ax.plot(epochs, m, color=color, label=label, linewidth=2, linestyle=ls)
    ax.fill_between(epochs, m - s, m + s, alpha=0.15, color=color)


# ── 1. comparison_head.png ───────────────────────────────────────────────────
# offline_wout per-epoch values are meaningless (Wout frozen at random init during
# those epochs). Show only the final offline head accuracy as a dashed reference line.
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_band(ax, epochs, std["head_mean"],  std["head_std"],  COLORS["standard"],        LABELS["standard"])
plot_band(ax, epochs, rand["head_mean"], rand["head_std"], COLORS["random_baseline"], LABELS["random_baseline"], ":")
plot_band(ax, epochs, jonly["head_mean"],jonly["head_std"],COLORS["j_only"],          LABELS["j_only"],          "-.")
ax.axhline(off["offline_head_mean"], color=COLORS["offline_wout"], linestyle="--", linewidth=2,
           label=f"Offline Wout (perceptron, final) = {off['offline_head_mean']:.3f}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Head accuracy (test)")
ax.set_title(f"Head accuracy — ablation comparison ({len(std['seeds'])} seeds ± 1 std)")
ax.legend(fontsize=8, loc="lower left")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylim(0, 0.55); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "comparison_head.png", dpi=150); plt.close(fig)
print("saved comparison_head.png")

# ── 2. comparison_probe.png ──────────────────────────────────────────────────
# offline_wout trains Win+J identically to standard, so its probe is redundant.
fig, ax = plt.subplots(figsize=(8, 4.5))
plot_band(ax, epochs, std["probe_mean"],  std["probe_std"],  COLORS["standard"],        LABELS["standard"])
plot_band(ax, epochs, rand["probe_mean"], rand["probe_std"], COLORS["random_baseline"], LABELS["random_baseline"], ":")
plot_band(ax, epochs, jonly["probe_mean"],jonly["probe_std"],COLORS["j_only"],          LABELS["j_only"],          "-.")
ax.set_xlabel("Epoch"); ax.set_ylabel("Linear probe accuracy (test)")
ax.set_title(f"Linear probe accuracy — ablation comparison ({len(std['seeds'])} seeds ± 1 std)")
ax.legend(fontsize=8, loc="lower right")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.set_ylim(0.38, 0.50); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "comparison_probe.png", dpi=150); plt.close(fig)
print("saved comparison_probe.png")

# ── 3. summary_bar.png ───────────────────────────────────────────────────────
labels_short = ["Standard", "Offline\nWout\n(perceptron)", "Random\nbaseline", "J-only\n(Win primes\nonce)"]
head_means = [
    np.array(std["head_mean"])[-1],
    off["offline_head_mean"],
    np.array(rand["head_mean"])[-1],
    np.array(jonly["head_mean"])[-1],
]
head_stds = [
    np.array(std["head_std"])[-1],
    off["offline_head_std"],
    np.array(rand["head_std"])[-1],
    np.array(jonly["head_std"])[-1],
]
probe_means = [
    np.array(std["probe_mean"])[-1],
    np.array(off["probe_mean"])[-1],
    np.array(rand["probe_mean"])[-1],
    np.array(jonly["probe_mean"])[-1],
]
probe_stds = [
    np.array(std["probe_std"])[-1],
    np.array(off["probe_std"])[-1],
    np.array(rand["probe_std"])[-1],
    np.array(jonly["probe_std"])[-1],
]

x = np.arange(len(labels_short)); width = 0.35
fig, ax = plt.subplots(figsize=(8, 4.5))
bh = ax.bar(x - width/2, head_means,  width, yerr=head_stds,  capsize=4,
            label="Head accuracy",  color="#2563EB", alpha=0.85)
bp = ax.bar(x + width/2, probe_means, width, yerr=probe_stds, capsize=4,
            label="Linear probe",   color="#EA580C", alpha=0.85)
ax.bar_label(bh, labels=[f"{v:.3f}" for v in head_means],  fontsize=8, padding=4)
ax.bar_label(bp, labels=[f"{v:.3f}" for v in probe_means], fontsize=8, padding=4)
ax.set_xticks(x); ax.set_xticklabels(labels_short)
ax.set_ylabel("Test accuracy")
ax.set_title(f"Final accuracy — ablation summary ({len(std['seeds'])} seeds, mean ± std)")
ax.legend(); ax.set_ylim(0, 0.60); ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIGURES / "summary_bar.png", dpi=150); plt.close(fig)
print("saved summary_bar.png")
