"""4-Matei_config/matei_config_run.py

Standard diagnostic run using matei_W_out_cfg.json instead of Kassym's config.
3 seeds × 20 epochs, same diagnostic collection as 1-Ablation_runs/standard_run.py.

Key differences from Kassym's best config:
  - clamped_n_iter=11, free_n_iter=14  (vs 5/6)
  - strength_back=0.168                (vs 1.471 — much weaker WBack signal)
  - Different lr_win, lr_j, lr_wout, momentum, entropy_beta

Head accuracy target: ~0.43 (matching Matei's reference 0.445).
jax_default_matmul_precision="high" required on NVIDIA GPUs to avoid TF32
corrupting the perceptron rule sign decisions.

Run on cluster:
  ~/miniforge3/envs/darnax_hpc/bin/python replicate_experiments/4-Matei_config/matei_config_run.py
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path

import jax
jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import jax.numpy as jnp, jax.tree_util as jtu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE     = Path(__file__).resolve().parent
REPO     = HERE.parent.parent
CFG_PATH = REPO / "replicate" / "matei_W_out_cfg.json"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent / "1-Ablation_runs"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.layer_maps.sparse import LayerMap
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer

from diagnostics import (
    collect_autocorr_matrix,
    collect_weight_norms,
    collect_field_contributions,
    collect_abcd_states,
    collect_abcd_8x8,
)

SEEDS        = [0, 42, 123]
EPOCHS       = 10
C, KSIZE     = 16, 5
H, W, POOL   = 32, 32, 8
PROBE_EPOCHS = 20
PROBE_WD     = 1.433e-4


def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0, key=keys[0],
                      padding_mode="constant", lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(
                channels=C, kernel_size=KSIZE, groups=1,
                j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
                entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
            ),
            2: ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                               strength=1.0, threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=layer_map)


def make_optimizer(orchestrator, cfg):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orchestrator, eqx.is_inexact_array)

    def like(tree, val):
        return jtu.tree_map(lambda _: val, tree, is_leaf=eqx.is_array)

    def sgd(lr):
        return optax.sgd(lr, momentum=mom) if mom > 0 else optax.sgd(lr)

    labels = jtu.tree_map(lambda _: "default", params, is_leaf=eqx.is_array)
    for (i, j), lbl in [((1, 0), "win"), ((1, 1), "j1"), ((2, 1), "wout")]:
        labels = eqx.tree_at(lambda m, r=i, c=j: m.lmap[r][c], labels,
                             replace=like(params.lmap[i][j], lbl))

    opt = optax.multi_transform({
        "default": optax.set_to_zero(),
        "win":  sgd(-cfg["lr_win"]),
        "j1":   sgd(-cfg["lr_j"]),
        "wout": sgd(cfg["lr_wout"]),
    }, labels)
    return opt, opt.init(eqx.filter(orchestrator, eqx.is_inexact_array))


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def pool_j1(h):
    N = h.shape[0]
    return h.reshape(N, H // POOL, POOL, W // POOL, POOL, C).mean(axis=(2, 4)).reshape(N, -1)


def run_probe(trainer, ds, key):
    reps_tr, lbl_tr, reps_te, lbl_te = [], [], [], []
    for xb, yb in ds:
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_tr.append(pool_j1(np.array(trainer.state[1])))
        lbl_tr.append(np.argmax(np.array(yb), axis=-1))
    for xb, yb in ds.iter_test():
        key, _ = trainer.eval_step(to_hwc(xb), yb, key)
        reps_te.append(pool_j1(np.array(trainer.state[1])))
        lbl_te.append(np.argmax(np.array(yb), axis=-1))

    X_tr = torch.from_numpy(np.concatenate(reps_tr)).float()
    y_tr = torch.from_numpy(np.concatenate(lbl_tr)).long()
    X_te = torch.from_numpy(np.concatenate(reps_te)).float()
    y_te = torch.from_numpy(np.concatenate(lbl_te)).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe  = nn.Linear(256, 10, bias=False).to(device)
    opt_p  = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=PROBE_WD)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    crit   = nn.CrossEntropyLoss()

    best_test = 0.0
    for _ in range(PROBE_EPOCHS):
        probe.train()
        for xb_t, yb_t in loader:
            xb_t, yb_t = xb_t.to(device), yb_t.to(device)
            opt_p.zero_grad()
            crit(probe(xb_t), yb_t).backward()
            opt_p.step()
        probe.eval()
        with torch.no_grad():
            te = (probe(X_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
        best_test = max(best_test, te)
    return best_test


def eval_head(trainer, ds, key):
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def fmt(s):
    m, sec = divmod(int(s), 60)
    return f"{m}m{sec:02d}s"


def run_one_seed(seed, cfg, ds, seed_idx, n_seeds, t_start, epoch_times_all, diag_rng):
    print(f"\n{'='*50}\nSeed {seed}  ({seed_idx+1}/{n_seeds})\n{'='*50}", flush=True)
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk)
    opt, opt_state = make_optimizer(orch, cfg)

    warmup_n  = 1
    clamped_n = cfg["clamped_n_iter"]
    free_n    = cfg["free_n_iter"]

    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=warmup_n,
        train_clamped_n_iter=clamped_n,
        train_free_n_iter=free_n,
        eval_n_iter=5,
    )

    diag_x, diag_y = next(iter(ds.iter_test()))
    diag_x = to_hwc(diag_x)

    decay = cfg["kernel_decay_rate"]
    head_accs, probe_accs, per_epoch_diag = [], [], []

    t_seed = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # train
        for xb, yb in ds:
            key = trainer.train_step(to_hwc(xb), yb, key)
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            trainer.orchestrator = eqx.tree_at(
                path, trainer.orchestrator,
                path(trainer.orchestrator) * (1.0 - decay),
            )

        head_acc, key = eval_head(trainer, ds, key)
        probe_acc     = run_probe(trainer, ds, key)

        # diagnostics
        t_diag = time.time()
        wn   = collect_weight_norms(trainer.orchestrator)
        ff   = collect_field_contributions(
            trainer.orchestrator, trainer.state, diag_x, diag_y, diag_rng, warmup_n)
        abcd = collect_abcd_states(
            trainer.orchestrator, trainer.state, diag_x, diag_y, diag_rng,
            warmup_n, clamped_n, free_n)
        _, orch_after, _, _ = eqx.filter_jit(DynamicalTrainer._train_step_impl)(
            diag_x, diag_y, diag_rng, trainer.orchestrator, trainer.state, trainer.ctx)
        abcd_8x8 = collect_abcd_8x8(
            trainer.orchestrator, orch_after, trainer.state,
            diag_x, diag_y, diag_rng, warmup_n, clamped_n, free_n)
        sim, labels, _ = collect_autocorr_matrix(
            trainer.orchestrator, trainer.state, diag_x, diag_y, diag_rng,
            warmup_n, clamped_n, free_n)
        t_diag = time.time() - t_diag

        per_epoch_diag.append({
            "epoch":           epoch,
            "weight_norms":    wn,
            "field_fractions": ff,
            "abcd":            abcd,
            "abcd_8x8":        abcd_8x8,
            "autocorr_matrix": sim.tolist(),
            "autocorr_labels": labels,
        })

        t_epoch = time.time() - t0
        epoch_times_all.append(t_epoch)
        elapsed = time.time() - t_start
        epochs_done  = seed_idx * EPOCHS + epoch
        epochs_total = n_seeds * EPOCHS
        eta = (sum(epoch_times_all) / len(epoch_times_all)) * (epochs_total - epochs_done)

        head_accs.append(head_acc)
        probe_accs.append(probe_acc)
        print(
            f"  seed={seed}  epoch={epoch:2d}/{EPOCHS}  head={head_acc:.4f}  probe={probe_acc:.4f}"
            f"  cd={abcd['cd_sim_mean']:.3f}  diag={fmt(t_diag)}"
            f"  [{fmt(t_epoch)}/ep  elapsed={fmt(elapsed)}  eta={fmt(eta)}]",
            flush=True,
        )
    print(f"  seed={seed} done in {fmt(time.time()-t_seed)}", flush=True)
    return {
        "seed":           seed,
        "head_accs":      head_accs,
        "probe_accs":     probe_accs,
        "per_epoch_diag": per_epoch_diag,
    }


def main():
    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"head_acc", "trial_number", "wback_type",
                        "j1_window_hebb", "j1_entropy", "probe_acc", "c05_j1"}}

    print("Config:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cfg.items()}, flush=True)
    print(f"Seeds: {SEEDS}  Epochs: {EPOCHS}", flush=True)

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))
    diag_rng = jax.random.PRNGKey(42)

    results_dir = HERE / "results"
    figures_dir = HERE / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    t_start = time.time()
    epoch_times_all = []
    per_seed = [
        run_one_seed(s, cfg, ds, i, len(SEEDS), t_start, epoch_times_all, diag_rng)
        for i, s in enumerate(SEEDS)
    ]
    print(f"\nTotal: {fmt(time.time()-t_start)}", flush=True)

    head_mat  = np.array([r["head_accs"]  for r in per_seed])
    probe_mat = np.array([r["probe_accs"] for r in per_seed])
    print(f"\nHead  final: {head_mat[:,-1].mean():.4f} ± {head_mat[:,-1].std():.4f}", flush=True)
    print(f"Probe final: {probe_mat[:,-1].mean():.4f} ± {probe_mat[:,-1].std():.4f}", flush=True)

    # save accuracy results
    out = {
        "seeds": SEEDS, "epochs": EPOCHS,
        "per_seed":   [{"seed": r["seed"], "head_accs": r["head_accs"], "probe_accs": r["probe_accs"]}
                       for r in per_seed],
        "head_mean":  head_mat.mean(0).tolist(),
        "head_std":   head_mat.std(0).tolist(),
        "probe_mean": probe_mat.mean(0).tolist(),
        "probe_std":  probe_mat.std(0).tolist(),
    }
    (results_dir / "run.json").write_text(json.dumps(out, indent=2))
    print("Saved results/run.json", flush=True)

    # save diagnostics
    diag_out = {
        "seeds": SEEDS, "epochs": EPOCHS,
        "warmup_n": 1, "clamped_n": cfg["clamped_n_iter"], "free_n": cfg["free_n_iter"],
        "per_seed": [{"seed": r["seed"], "per_epoch_diag": r["per_epoch_diag"]}
                     for r in per_seed],
    }
    (results_dir / "diagnostics.json").write_text(json.dumps(diag_out, indent=2))
    print("Saved results/diagnostics.json", flush=True)

    # plot accuracy curves
    ep = np.arange(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for r, col in zip(per_seed, ["#2563EB", "#EA580C", "#16A34A"]):
        ax1.plot(ep, r["head_accs"],  "-o", color=col, label=f"seed={r['seed']}", markersize=3)
        ax2.plot(ep, r["probe_accs"], "-s", color=col, label=f"seed={r['seed']}", markersize=3)
    hm, hs = head_mat.mean(0),  head_mat.std(0)
    pm, ps = probe_mat.mean(0), probe_mat.std(0)
    ax1.plot(ep, hm, "k-", linewidth=2, label="mean")
    ax1.fill_between(ep, hm-hs, hm+hs, alpha=0.15, color="k")
    ax2.plot(ep, pm, "k-", linewidth=2, label="mean")
    ax2.fill_between(ep, pm-ps, pm+ps, alpha=0.15, color="k")
    for ax, title in [(ax1, "Head accuracy (Wout, perceptron rule)"),
                      (ax2, "Linear probe (J1, Adam 20ep)")]:
        ax.set_xlabel("Epoch"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Matei config — {len(SEEDS)} seeds  (jax_matmul_precision='high')", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_curves.png", dpi=130)
    plt.close(fig)
    print("Saved figures/accuracy_curves.png", flush=True)


if __name__ == "__main__":
    main()
