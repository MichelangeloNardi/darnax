"""ablation_runs.py

Three ablation training variants for the channel-entropy CIFAR-10 architecture.
5 seeds x 10 epochs each, best config.

Modes
-----
  offline_wout    — Win + J1 train normally; Wout is frozen (lr=0) throughout
                    the 10 epochs. After training, Wout is optimised offline via
                    Adam+CE on the fixed final J1 representations and the model
                    is evaluated with the new weights.

  random_baseline — Win and J1 frozen at random initialisation; only Wout trains
                    online with the perceptron rule.

  j_only          — Win frozen at random init; J1 + Wout train normally.

Freezing is implemented by two independent mechanisms:
  1. optax.set_to_zero() in the optimizer  — no weight update applied
  2. decay_win/decay_j1=False in train_epoch — skips per-epoch kernel decay

Results saved to:
  experiments5/results/ablation_{mode}.json  — per-seed and mean±std

Usage:
  python experiments5/ablation_runs.py --mode all
  python experiments5/ablation_runs.py --mode offline_wout
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import jax
jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import jax.numpy as jnp, jax.tree_util as jtu
import numpy as np
import optax, torch
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

SEEDS             = [0, 42, 123, 7, 999]
EPOCHS            = 10
WOUT_OFFLINE_EPOCHS = 10   # epochs to train Wout offline with perceptron rule
C, KSIZE          = 16, 5
H, W, POOL        = 32, 32, 8
PROBE_EPOCHS      = 20
PROBE_WD          = 1.433e-4

# lambda_win=0.0 means Win message is zeroed out at every step (extreme decay)
MODE_FLAGS = {
    #                       lr_win  lr_j   lr_wout  decay_win  decay_j1  lambda_win
    "offline_wout":    dict(win=1,  j1=1,  wout=0,  decay_win=True,  decay_j1=True,  lambda_win=1.0),
    "random_baseline": dict(win=0,  j1=0,  wout=1,  decay_win=False, decay_j1=False, lambda_win=1.0),
    "j_only":          dict(win=0,  j1=1,  wout=1,  decay_win=False, decay_j1=True,  lambda_win=0.0),
}


def build_model(cfg, key, lambda_win=1.0):
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
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(
        layers=layer_map, lambda_win=lambda_win
    )


def make_optimizer(orchestrator, cfg, lr_win_active, lr_j_active, lr_wout_active):
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
        "win":  sgd(-cfg["lr_win"]) if lr_win_active else optax.set_to_zero(),
        "j1":   sgd(-cfg["lr_j"])   if lr_j_active  else optax.set_to_zero(),
        "wout": sgd(cfg["lr_wout"]) if lr_wout_active else optax.set_to_zero(),
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


def train_epoch(trainer, ds, cfg, key, decay_win=False, decay_j1=True):
    decay = cfg["kernel_decay_rate"]
    for xb, yb in ds:
        key = trainer.train_step(to_hwc(xb), yb, key)
    if decay_win:
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[1][0].kernel, trainer.orchestrator,
            trainer.orchestrator.lmap[1][0].kernel * (1.0 - decay),
        )
    if decay_j1:
        trainer.orchestrator = eqx.tree_at(
            lambda o: o.lmap[1][1].kernel, trainer.orchestrator,
            trainer.orchestrator.lmap[1][1].kernel * (1.0 - decay),
        )
    return trainer, key


def eval_head(trainer, ds, key):
    accs = []
    for xb, yb in ds.iter_test():
        key, metrics = trainer.eval_step(to_hwc(xb), yb, key)
        accs.append(float(metrics["accuracy"]))
    return float(np.mean(accs)), key


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def run_one_seed(mode, seed, cfg, ds, seed_idx, n_seeds, t_script_start, epoch_times_all):
    flags = MODE_FLAGS[mode]
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = build_model(cfg, mk, lambda_win=flags["lambda_win"])
    opt, opt_state = make_optimizer(
        orch, cfg,
        lr_win_active=flags["win"],
        lr_j_active=flags["j1"],
        lr_wout_active=flags["wout"],
    )
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=1,
        train_clamped_n_iter=cfg["clamped_n_iter"],
        train_free_n_iter=cfg["free_n_iter"],
        eval_n_iter=5,
    )

    n_modes = len(MODE_FLAGS)
    modes_list = list(MODE_FLAGS)
    mode_idx = modes_list.index(mode)
    epochs_total = n_modes * n_seeds * EPOCHS
    epochs_before_this_mode = mode_idx * n_seeds * EPOCHS

    head_accs, probe_accs = [], []
    t_seed_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        trainer, key = train_epoch(
            trainer, ds, cfg, key,
            decay_win=flags["decay_win"],
            decay_j1=flags["decay_j1"],
        )
        head_acc, key = eval_head(trainer, ds, key)
        probe_acc = run_probe(trainer, ds, key)
        t_epoch = time.time() - t0
        epoch_times_all.append(t_epoch)

        elapsed = time.time() - t_script_start
        epochs_done = epochs_before_this_mode + seed_idx * EPOCHS + epoch
        avg_epoch = sum(epoch_times_all) / len(epoch_times_all)
        eta = avg_epoch * (epochs_total - epochs_done)

        head_accs.append(head_acc)
        probe_accs.append(probe_acc)
        print(f"  seed={seed}  epoch={epoch:2d}/{EPOCHS}  head={head_acc:.4f}  probe={probe_acc:.4f}"
              f"  [{fmt(t_epoch)}/epoch  elapsed={fmt(elapsed)}  eta={fmt(eta)}]", flush=True)
    print(f"  seed={seed} done in {fmt(time.time()-t_seed_start)}", flush=True)

    result = {"seed": seed, "head_accs": head_accs, "probe_accs": probe_accs}

    if mode == "offline_wout":
        # Train Wout offline with the perceptron rule (Win+J frozen, same rule as online)
        # This answers: "how well can Wout learn given fixed final representations?"
        print(f"  [offline Wout] training with perceptron rule for {WOUT_OFFLINE_EPOCHS} epochs", flush=True)
        t_offline_start = time.time()
        opt2, opt2_state = make_optimizer(
            trainer.orchestrator, cfg,
            lr_win_active=False, lr_j_active=False, lr_wout_active=True,
        )
        trainer2 = DynamicalTrainer(
            orchestrator=trainer.orchestrator, state=trainer.state,
            optimizer=opt2, optimizer_state=opt2_state,
            warmup_n_iter=1,
            train_clamped_n_iter=cfg["clamped_n_iter"],
            train_free_n_iter=cfg["free_n_iter"],
            eval_n_iter=5,
        )
        for ep in range(1, WOUT_OFFLINE_EPOCHS + 1):
            trainer2, key = train_epoch(trainer2, ds, cfg, key, decay_win=False, decay_j1=False)
            h, key = eval_head(trainer2, ds, key)
            print(f"    offline epoch {ep:2d}/{WOUT_OFFLINE_EPOCHS}  head={h:.4f}", flush=True)
        offline_head_acc, key = eval_head(trainer2, ds, key)
        print(f"  [offline Wout done] head={offline_head_acc:.4f}  ({fmt(time.time()-t_offline_start)})", flush=True)
        result["offline_head_acc"] = offline_head_acc

    return result


def run_mode(mode, cfg, ds, t_script_start, epoch_times_all):
    print(f"\n{'='*60}", flush=True)
    print(f"MODE: {mode}  ({len(SEEDS)} seeds)", flush=True)
    flags = MODE_FLAGS[mode]
    print(f"  Win trains: {bool(flags['win'])}  |  J1 trains: {bool(flags['j1'])}  "
          f"|  Wout trains: {bool(flags['wout'])}", flush=True)
    print(f"{'='*60}", flush=True)

    per_seed = [run_one_seed(mode, s, cfg, ds, i, len(SEEDS), t_script_start, epoch_times_all)
                for i, s in enumerate(SEEDS)]

    head_mat  = np.array([r["head_accs"]  for r in per_seed])
    probe_mat = np.array([r["probe_accs"] for r in per_seed])

    result = {
        "mode":        mode,
        "seeds":       SEEDS,
        "epochs":      EPOCHS,
        "per_seed":    per_seed,
        "head_mean":   head_mat.mean(0).tolist(),
        "head_std":    head_mat.std(0).tolist(),
        "probe_mean":  probe_mat.mean(0).tolist(),
        "probe_std":   probe_mat.std(0).tolist(),
    }

    if mode == "offline_wout":
        offline_heads = [r["offline_head_acc"] for r in per_seed]
        result["offline_head_mean"] = float(np.mean(offline_heads))
        result["offline_head_std"]  = float(np.std(offline_heads))
        print(f"\n  offline Wout (perceptron rule) — head: "
              f"{result['offline_head_mean']:.4f} ± {result['offline_head_std']:.4f}", flush=True)

    print(f"\n  Head  final: mean={head_mat[:,-1].mean():.4f}  std={head_mat[:,-1].std():.4f}", flush=True)
    print(f"  Probe final: mean={probe_mat[:,-1].mean():.4f}  std={probe_mat[:,-1].std():.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=list(MODE_FLAGS) + ["all"])
    args = parser.parse_args()

    with open(CFG_PATH) as f:
        full_cfg = json.load(f)
    cfg = {k: v for k, v in full_cfg.items()
           if k not in {"wback_type", "j1_window_hebb", "j1_entropy",
                        "trial_number", "probe_acc", "c05_j1"}}

    ds = Cifar10(batch_size=32, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True)
    ds.build(jax.random.PRNGKey(0))

    modes = list(MODE_FLAGS) if args.mode == "all" else [args.mode]
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    t_script_start = time.time()
    epoch_times_all = []
    for mode in modes:
        result = run_mode(mode, cfg, ds, t_script_start, epoch_times_all)
        out_path = results_dir / f"ablation_{mode}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved to {out_path}", flush=True)
    print(f"\nTotal runtime: {fmt(time.time()-t_script_start)}", flush=True)


if __name__ == "__main__":
    main()
