"""wout_improvement.py

Diagnoses and reduces the W_out vs probe accuracy gap on CIFAR-10.

Standard setup:
  W_out (PooledFlattenFC + perceptron rule, threshold=5) → ~22% test accuracy
  Linear probe on same pooled J1 representations → ~45%

Both classifiers see identical 256-dim inputs (8×8 avg-pool of the 32×32×16 J1
fixed-point state). The gap is entirely due to the training algorithm.

Three approaches compared per seed:
  A. Baseline  — perceptron rule during 20-epoch joint training
  B. CE rule   — cross-entropy gradient for W_out, same 20-epoch joint training
  C. Adam w.s. — after standard training, freeze J1 and fine-tune W_out
                 with Adam+CE on fixed representations (warm start from W_out init)

Diagnostics:
  - Alignment: mean cosine similarity between W_out cols and probe cols
  - Head accuracy curve and probe accuracy curve per epoch

Run on w01:
  ~/miniforge3/envs/darnax_hpc/bin/python experiments4/wout_improvement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"

sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

C, KSIZE = 16, 5
H, W = 32, 32
POOL = 8
PROBE_DIM = (H // POOL) * (W // POOL) * C  # 256
EPOCHS = 20
SEEDS = [0, 42, 123]
PROBE_EPOCHS = 20
PROBE_WD = 1.433e-4
FINETUNE_EPOCHS = 30  # Adam fine-tune passes after standard training


# ---------------------------------------------------------------------------
# CE variant of PooledFlattenFC
# Uses softmax CE gradient instead of the perceptron rule.
# ---------------------------------------------------------------------------

class CEPooledFlattenFC(PooledFlattenFC):
    """PooledFlattenFC with softmax cross-entropy backward."""

    def backward(self, x, y, y_hat, gate=None):
        x_p = self._pool(x)                        # (B, 256)
        B = y_hat.shape[0]
        H_in = self.W.shape[0]                     # 256
        probs = jax.nn.softmax(y_hat, axis=-1)     # (B, 10)
        dL_dz = probs - (y + 1.0) / 2.0           # (B, 10); y in {-1,+1} → target in {0,1}
        dL_dW = x_p.T @ dL_dz / B                 # (256, 10)
        dW = dL_dW / (H_in ** 0.5)                # scale like perceptron rule
        dW = self.lr * dW + self.lr * self.weight_decay * self.W
        zero = jax.tree.map(jnp.zeros_like, self)
        return eqx.tree_at(lambda m: m.W, zero, dW)


# ---------------------------------------------------------------------------
# Model / optimizer builders
# ---------------------------------------------------------------------------

def build_model(cfg: dict, key: jax.Array, use_ce_wout: bool = False):
    keys = jax.random.split(key, 5)
    wout_cls = CEPooledFlattenFC if use_ce_wout else PooledFlattenFC
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(
                in_channels=3, out_channels=C, kernel_size=KSIZE,
                threshold=cfg["threshold_win"], strength=1.0,
                key=keys[0], padding_mode="constant", lr=1.0, weight_decay=0.0,
            ),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: wout_cls(
                pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                strength=1.0, threshold=5.0,
                key=keys[3], lr=1.0, weight_decay=0.0,
            ),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def make_optimizer(orchestrator, cfg: dict):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(
            lambda m, r=i, c=j: m.lmap[r][c], labels,
            replace=like(params.lmap[i][j], lbl),
        )

    opt = optax.multi_transform({
        "default": optax.sgd(0.0),
        "win":     sgd(-cfg["lr_win"]),
        "j1":      sgd(-cfg["lr_j"]),
        "wout":    sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


# ---------------------------------------------------------------------------
# Representation collection
# ---------------------------------------------------------------------------

def collect_reps(trainer, ds: Cifar10):
    key = jax.random.PRNGKey(999)
    reps_tr, lbl_tr, reps_te, lbl_te = [], [], [], []
    for xb, yb in ds:
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_tr.append(pool_j1(np.array(trainer.state[1])))
        lbl_tr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_te.append(pool_j1(np.array(trainer.state[1])))
        lbl_te.append(np.argmax(np.array(yb), axis=-1))
    return (np.concatenate(reps_tr), np.concatenate(lbl_tr),
            np.concatenate(reps_te), np.concatenate(lbl_te))


# ---------------------------------------------------------------------------
# Torch probe / fine-tune
# ---------------------------------------------------------------------------

def run_probe(X_tr, y_tr, X_te, y_te,
              n_epochs: int = PROBE_EPOCHS,
              init_W: np.ndarray | None = None) -> tuple[list[float], np.ndarray]:
    """Train a linear probe (Adam + CE) on fixed representations.

    Returns (test_acc_per_epoch, final_W) where final_W has shape (256, 10).
    If init_W is provided, warm-start from those weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = nn.Linear(PROBE_DIM, 10, bias=False).to(device)
    if init_W is not None:
        probe.weight.data = torch.from_numpy(init_W.T).float().to(device)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    X_te_t = torch.from_numpy(X_te).float().to(device)
    y_te_t = torch.from_numpy(y_te).long().to(device)

    opt_p = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)
    crit = nn.CrossEntropyLoss()

    test_accs = []
    for _ in range(n_epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_p.zero_grad()
            crit(probe(xb), yb).backward()
            opt_p.step()
        probe.eval()
        with torch.no_grad():
            te = (probe(X_te_t).argmax(1) == y_te_t).float().mean().item()
        test_accs.append(te)

    final_W = probe.weight.data.T.cpu().numpy()  # (256, 10)
    return test_accs, final_W


def eval_wout(W: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> float:
    """Direct evaluation of a weight matrix on test representations."""
    preds = np.argmax(X_te @ W, axis=1)
    return float((preds == y_te).mean())


def cosine_alignment(W1: np.ndarray, W2: np.ndarray) -> float:
    """Mean cosine similarity between corresponding columns of W1, W2 (256, 10)."""
    sims = []
    for k in range(W1.shape[1]):
        v1, v2 = W1[:, k], W2[:, k]
        sims.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)))
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_seed(cfg: dict, ds: Cifar10, seed: int, use_ce_wout: bool = False):
    tag = "CE  " if use_ce_wout else "perc"
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk, use_ce_wout=use_ce_wout)
    opt, opt_state = make_optimizer(orch, cfg)

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    decay = cfg["kernel_decay_rate"]
    head_accs, probe_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
            win_k = trainer.orchestrator.lmap[1][0].kernel
            kh, kw, ci, co = win_k.shape
            flat = win_k.reshape(-1, co)
            normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
            trainer.orchestrator = eqx.tree_at(
                lambda o: o.lmap[1][0].kernel,
                trainer.orchestrator,
                normed.reshape(kh, kw, ci, co),
            )

        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator, path(trainer.orchestrator) * (1.0 - decay),
            )

        batch_accs = []
        for xb, yb in ds.iter_test():
            key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
            batch_accs.append(float(metrics["accuracy"]))
        head_acc = float(np.mean(batch_accs))
        head_accs.append(head_acc)

        reps_tr, lbl_tr, reps_te, lbl_te = collect_reps(trainer, ds)
        probe_curve, _ = run_probe(reps_tr, lbl_tr, reps_te, lbl_te)
        probe_accs.append(max(probe_curve))

        print(f"  [{tag}] seed={seed}  epoch={epoch:2d}/{EPOCHS}"
              f"  head={head_acc:.4f}  probe={probe_accs[-1]:.4f}", flush=True)

    return trainer, head_accs, probe_accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    results_dir = HERE / "results"
    figs_dir    = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    all_results: dict = {}

    for seed in SEEDS:
        print(f"\n{'='*65}")
        print(f"SEED {seed}")
        print(f"{'='*65}")

        # --- A: baseline perceptron ---
        print("\n[A] Perceptron rule")
        trainer_p, head_p, probe_p = train_one_seed(cfg, ds, seed, use_ce_wout=False)

        reps_tr, lbl_tr, reps_te, lbl_te = collect_reps(trainer_p, ds)
        W_p = np.array(trainer_p.orchestrator.lmap[2][1].W)  # (256, 10)
        wout_acc_p = eval_wout(W_p, reps_te, lbl_te)

        probe_scratch_curve, probe_W = run_probe(reps_tr, lbl_tr, reps_te, lbl_te)
        probe_scratch = max(probe_scratch_curve)

        # C: Adam warm-start from W_out weights
        print(f"  [post] Adam warm-start from W_out...")
        adam_ws_curve, _ = run_probe(
            reps_tr, lbl_tr, reps_te, lbl_te,
            n_epochs=FINETUNE_EPOCHS, init_W=W_p,
        )

        align_p = cosine_alignment(W_p, probe_W)

        print(f"\n  [seed={seed}] Perceptron post-training:")
        print(f"    W_out direct:      {wout_acc_p:.4f}")
        print(f"    Adam warm-start:   {max(adam_ws_curve):.4f}")
        print(f"    Probe scratch:     {probe_scratch:.4f}")
        print(f"    Alignment:         {align_p:.4f}")

        # --- B: CE rule ---
        print("\n[B] CE rule")
        trainer_ce, head_ce, probe_ce = train_one_seed(cfg, ds, seed, use_ce_wout=True)

        reps_tr2, lbl_tr2, reps_te2, lbl_te2 = collect_reps(trainer_ce, ds)
        W_ce = np.array(trainer_ce.orchestrator.lmap[2][1].W)
        wout_acc_ce = eval_wout(W_ce, reps_te2, lbl_te2)

        probe_ce_scratch_curve, probe_W_ce = run_probe(reps_tr2, lbl_tr2, reps_te2, lbl_te2)
        probe_ce_scratch = max(probe_ce_scratch_curve)
        align_ce = cosine_alignment(W_ce, probe_W_ce)

        print(f"\n  [seed={seed}] CE post-training:")
        print(f"    W_out direct:      {wout_acc_ce:.4f}")
        print(f"    Probe scratch:     {probe_ce_scratch:.4f}")
        print(f"    Alignment:         {align_ce:.4f}")

        all_results[str(seed)] = {
            "perc": {
                "head_accs":      head_p,
                "probe_accs":     probe_p,
                "wout_acc":       float(wout_acc_p),
                "probe_scratch":  float(probe_scratch),
                "adam_ws_best":   float(max(adam_ws_curve)),
                "adam_ws_curve":  adam_ws_curve,
                "alignment":      float(align_p),
            },
            "ce": {
                "head_accs":      head_ce,
                "probe_accs":     probe_ce,
                "wout_acc":       float(wout_acc_ce),
                "probe_scratch":  float(probe_ce_scratch),
                "alignment":      float(align_ce),
            },
        }

    # --- save ---
    out_path = results_dir / "wout_improvement.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}")

    # --- plot ---
    epochs = np.arange(1, EPOCHS + 1)
    ft_epochs = np.arange(1, FINETUNE_EPOCHS + 1)
    colours = plt.cm.tab10(np.linspace(0, 0.5, len(SEEDS)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_h, ax_p, ax_ft, ax_bar = axes.flat

    # top-left: head accuracy
    for col, seed in zip(colours, SEEDS):
        ax_h.plot(epochs, all_results[str(seed)]["perc"]["head_accs"],
                  color=col, ls="-", alpha=0.8, label=f"perc s{seed}")
        ax_h.plot(epochs, all_results[str(seed)]["ce"]["head_accs"],
                  color=col, ls="--", alpha=0.8, label=f"CE s{seed}")
    ax_h.set_title("Head accuracy  (— perc, -- CE)"); ax_h.set_xlabel("Epoch")
    ax_h.legend(fontsize=7, ncol=2)

    # top-right: probe accuracy
    for col, seed in zip(colours, SEEDS):
        ax_p.plot(epochs, all_results[str(seed)]["perc"]["probe_accs"],
                  color=col, ls="-", alpha=0.8)
        ax_p.plot(epochs, all_results[str(seed)]["ce"]["probe_accs"],
                  color=col, ls="--", alpha=0.8)
    ax_p.set_title("Probe accuracy during training  (— perc, -- CE)")
    ax_p.set_xlabel("Epoch")

    # bottom-left: Adam warm-start curve
    for col, seed in zip(colours, SEEDS):
        ax_ft.plot(ft_epochs, all_results[str(seed)]["perc"]["adam_ws_curve"],
                   color=col, label=f"seed {seed}")
    mean_wout = np.mean([all_results[str(s)]["perc"]["wout_acc"] for s in SEEDS])
    mean_probe = np.mean([all_results[str(s)]["perc"]["probe_scratch"] for s in SEEDS])
    ax_ft.axhline(mean_wout,  color="red",   ls=":", lw=1.5, label=f"W_out perc ({mean_wout:.3f})")
    ax_ft.axhline(mean_probe, color="green", ls=":", lw=1.5, label=f"probe scratch ({mean_probe:.3f})")
    ax_ft.set_title("Adam warm-start fine-tune (W_out init, fixed reps)")
    ax_ft.set_xlabel("Fine-tune epoch"); ax_ft.legend(fontsize=8)

    # bottom-right: summary bar
    methods = ["Perceptron\nW_out", "CE rule\nW_out", "Adam\nwarm-start", "Probe\nscratch"]
    bar_means = [
        np.mean([all_results[str(s)]["perc"]["wout_acc"] for s in SEEDS]),
        np.mean([all_results[str(s)]["ce"]["wout_acc"] for s in SEEDS]),
        np.mean([all_results[str(s)]["perc"]["adam_ws_best"] for s in SEEDS]),
        np.mean([all_results[str(s)]["perc"]["probe_scratch"] for s in SEEDS]),
    ]
    bar_stds = [
        np.std([all_results[str(s)]["perc"]["wout_acc"] for s in SEEDS]),
        np.std([all_results[str(s)]["ce"]["wout_acc"] for s in SEEDS]),
        np.std([all_results[str(s)]["perc"]["adam_ws_best"] for s in SEEDS]),
        np.std([all_results[str(s)]["perc"]["probe_scratch"] for s in SEEDS]),
    ]
    ax_bar.bar(methods, bar_means, yerr=bar_stds, capsize=5,
               color=["steelblue", "orange", "green", "red"], alpha=0.8)
    ax_bar.set_title("W_out accuracy: method comparison")
    ax_bar.set_ylabel("Test accuracy")
    for i, (m, s) in enumerate(zip(bar_means, bar_stds)):
        ax_bar.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)

    fig.suptitle("W_out improvement — C=16, CIFAR-10, entropy rule", fontsize=12)
    fig.tight_layout()
    fig_path = figs_dir / "wout_improvement.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to {fig_path}")

    # --- final summary ---
    print(f"\n{'='*65}")
    print(f"SUMMARY (mean ± std over {len(SEEDS)} seeds)")
    print(f"{'='*65}")
    for method, key_path in [
        ("Perceptron W_out",   ("perc", "wout_acc")),
        ("CE rule W_out",      ("ce",   "wout_acc")),
        ("Adam warm-start",    ("perc", "adam_ws_best")),
        ("Probe (scratch)",    ("perc", "probe_scratch")),
    ]:
        vals = [all_results[str(s)][key_path[0]][key_path[1]] for s in SEEDS]
        print(f"  {method:22s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\nAlignment (perc):  {np.mean([all_results[str(s)]['perc']['alignment'] for s in SEEDS]):.4f}")
    print(f"Alignment (CE):    {np.mean([all_results[str(s)]['ce']['alignment'] for s in SEEDS]):.4f}")


if __name__ == "__main__":
    main()
