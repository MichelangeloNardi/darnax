"""experiments4/cross_time_cd_kassym.py

Experiment 2 (per-image trajectory, unified baseline at batch 50) on Kassym's
tuned config (trial 14). Same structure as cross_time_cd_v2.py.
"""

from __future__ import annotations

import json
import sys
import time
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from darnax.datasets.classification.cifar10 import Cifar10
from darnax.layer_maps.sparse import LayerMap
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC
from darnax.modules.input_output import OutputLayer
from darnax.orchestrators.sequential import SequentialOrchestrator
from darnax.states.sequential import SequentialState
from darnax.trainers.dynamical import DynamicalTrainer
from darnax.trainers.utils import scan_n

C_CH, KSIZE = 16, 5
H, W = 32, 32
POOL = 8
SEED = 0
BATCH_SIZE = 32

N_EPOCHS = 2
MEASURE_EVERY = 50
BASELINE_BATCH = 50
PROBES_PER_CLASS = 5
N_CLASSES = 10

CONFIG = {
    "lr_j":              0.0009740402572938124,
    "lr_win":            0.02261680399040041,
    "lr_wout":           0.04321887433959399,
    "kernel_decay_rate": 0.0007847404026606365,
    "threshold_j":       1.9796175431992729,
    "threshold_win":     0.8510825401105292,
    "j_d":               0.8953837335689401,
    "entropy_beta":      0.3449405979902655,
    "momentum":          0.3162081708642795,
    "strength_back":     1.4711531428803912,
    "warmup_n_iter":     1,
    "clamped_n_iter":    5,
    "free_n_iter":       6,
}


def build_model(cfg, key):
    keys = jax.random.split(key, 5)
    layer_map = LayerMap.from_dict({
        1: {
            0: Conv2D(in_channels=3, out_channels=C_CH, kernel_size=KSIZE,
                      threshold=cfg["threshold_win"], strength=1.0,
                      key=keys[0], padding_mode="constant",
                      lr=1.0, weight_decay=0.0),
            1: Conv2DRecurrentDiscrete(channels=C_CH, kernel_size=KSIZE, groups=1,
                                       j_d=cfg["j_d"], threshold=cfg["threshold_j"],
                                       key=keys[1], padding_mode="constant",
                                       lr=1.0, weight_decay=0.0,
                                       entropy_beta=cfg["entropy_beta"],
                                       lambda_entropy=1.0),
            2: ChannelWBack(10, H, W, C_CH, cfg["strength_back"], keys[2]),
        },
        2: {
            1: PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C_CH, n_classes=10,
                               strength=1.0, threshold=5.0,
                               key=keys[3], lr=1.0, weight_decay=0.0),
            2: OutputLayer(),
        },
    })
    state = SequentialState([(H, W, 3), (H, W, C_CH), 10])
    return state, SequentialOrchestrator(layers=layer_map)


def make_train_optimizer(orch, cfg):
    mom = cfg["momentum"]
    params, _ = eqx.partition(orch, eqx.is_inexact_array)

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
    return opt, opt.init(eqx.filter(orch, eqx.is_inexact_array))


def normalize_win(orch):
    win_k = orch.lmap[1][0].kernel
    kh, kw, ci, co = win_k.shape
    flat = win_k.reshape(-1, co)
    normed = flat / (jnp.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
    return eqx.tree_at(
        lambda o: o.lmap[1][0].kernel, orch, normed.reshape(kh, kw, ci, co),
    )


def to_hwc(xb):
    return xb.reshape(-1, 32, 32, 3) * 2.0 - 1.0


def compute_permutations(ds, n_epochs):
    assert ds._train_epoch_key is not None
    n_train = ds.x_train.shape[0]
    epoch_key = ds._train_epoch_key
    perms = []
    for _ in range(n_epochs):
        key_epoch, epoch_key = jax.random.split(epoch_key)
        perms.append(np.asarray(jax.random.permutation(key_epoch, n_train)))
    return perms


def pick_probes_earliest_per_class(y_train, perm, probes_per_class):
    y_class = np.asarray(jnp.argmax(y_train, axis=-1))
    picked: dict[int, list[int]] = {c: [] for c in range(N_CLASSES)}
    n_remaining = N_CLASSES * probes_per_class
    for oi in perm:
        c = int(y_class[oi])
        if len(picked[c]) < probes_per_class:
            picked[c].append(int(oi))
            n_remaining -= 1
            if n_remaining == 0:
                break
    indices = []
    for c in range(N_CLASSES):
        indices.extend(picked[c])
    return indices


def first_seen_batches(perms, probe_indices, batch_size):
    perm0 = perms[0]
    return [int(np.where(perm0 == oi)[0][0] // batch_size) for oi in probe_indices]


def make_measure_fn(cfg):
    warmup_n = cfg["warmup_n_iter"]
    clamped_n = cfg["clamped_n_iter"]
    free_n = cfg["free_n_iter"]

    @eqx.filter_jit
    def measure(orch, state_template, x, y, rng):
        state0 = state_template.init(x, y)
        (state_a, rng_a), _ = scan_n(orch.step, (state0, rng),
                                     n_iter=warmup_n, filter_messages="forward")
        (state_b, rng_b), _ = scan_n(orch.step, (state_a, rng_a),
                                     n_iter=clamped_n, filter_messages="all")
        (state_c, rng_c), _ = scan_n(orch.step, (state_b, rng_b),
                                     n_iter=free_n, filter_messages="forward")
        (state_d, rng_d), _ = scan_n(orch.step, (state_a, rng_a),
                                     n_iter=free_n, filter_messages="forward")
        state_c_pred, _ = orch.predict(state_c, rng_c)
        state_d_pred, _ = orch.predict(state_d, rng_d)
        return {
            "j1_C": state_c[1], "j1_D": state_d[1],
            "margin_C": soft_margin(state_c_pred.readout, y),
            "margin_D": soft_margin(state_d_pred.readout, y),
        }
    return measure


def soft_margin(logits, y):
    correct = jnp.argmax(y, axis=-1)
    correct_logit = jnp.take_along_axis(logits, correct[:, None], axis=-1)[:, 0]
    mask = jax.nn.one_hot(correct, num_classes=logits.shape[-1], dtype=bool)
    wrong_max = jnp.max(jnp.where(mask, -jnp.inf, logits), axis=-1)
    return correct_logit - wrong_max


def main():
    print("Building dataset...", flush=True)
    ds = Cifar10(batch_size=BATCH_SIZE, x_transform="identity", label_mode="pm1",
                 linear_projection=None, rescale=True, shuffle=True)
    ds.build(jax.random.PRNGKey(0))

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    perms = compute_permutations(ds, N_EPOCHS)
    probe_indices = pick_probes_earliest_per_class(
        ds.y_train, perms[0], PROBES_PER_CLASS)
    first_seen = first_seen_batches(perms, probe_indices, BATCH_SIZE)
    n_batches_per_epoch = len(ds._train_bounds)
    total_batches = N_EPOCHS * n_batches_per_epoch

    assert max(first_seen) < BASELINE_BATCH

    probe_x = to_hwc(ds.x_train[jnp.asarray(probe_indices)])
    probe_y = ds.y_train[jnp.asarray(probe_indices)]
    probe_classes = np.asarray(jnp.argmax(probe_y, axis=-1))
    n_probes = len(probe_indices)

    print(f"Probes per class    : {PROBES_PER_CLASS} (total {n_probes})", flush=True)
    print(f"First-seen range    : {min(first_seen)}..{max(first_seen)}", flush=True)
    print(f"BASELINE_BATCH      : {BASELINE_BATCH}", flush=True)
    print(f"Total batches       : {total_batches}", flush=True)

    print("\nBuilding model (Kassym config)...", flush=True)
    key = jax.random.PRNGKey(SEED)
    key, mk = jax.random.split(key)
    state, orch = build_model(CONFIG, mk)
    opt, opt_state = make_train_optimizer(orch, CONFIG)
    trainer = DynamicalTrainer(
        orchestrator=orch, state=state,
        optimizer=opt, optimizer_state=opt_state,
        warmup_n_iter=CONFIG["warmup_n_iter"],
        train_clamped_n_iter=CONFIG["clamped_n_iter"],
        train_free_n_iter=CONFIG["free_n_iter"],
        eval_n_iter=CONFIG["free_n_iter"],
    )

    measure = make_measure_fn(CONFIG)
    meas_rng = jax.random.PRNGKey(7777)

    measure_batches: list[int] = []
    margin_C_list: list[np.ndarray] = []
    margin_D_list: list[np.ndarray] = []
    stability_C_list: list[list[float]] = [[] for _ in range(n_probes)]
    stability_D_list: list[list[float]] = [[] for _ in range(n_probes)]
    cross_C0_D_list: list[list[float]] = [[] for _ in range(n_probes)]
    overlap_CD: list[np.ndarray] = []

    baseline_J1_C: np.ndarray | None = None
    baseline_J1_D: np.ndarray | None = None

    def record(batch_count):
        nonlocal baseline_J1_C, baseline_J1_D
        out = measure(trainer.orchestrator, trainer.state, probe_x, probe_y, meas_rng)
        j1_C = np.asarray(out["j1_C"])
        j1_D = np.asarray(out["j1_D"])
        margin_C_list.append(np.asarray(out["margin_C"]))
        margin_D_list.append(np.asarray(out["margin_D"]))
        measure_batches.append(batch_count)
        cd_per_img = np.mean(np.sign(j1_C) * np.sign(j1_D), axis=(1, 2, 3))
        overlap_CD.append(cd_per_img)

        if batch_count == BASELINE_BATCH:
            baseline_J1_C = j1_C.copy()
            baseline_J1_D = j1_D.copy()
            for pi in range(n_probes):
                stability_C_list[pi].append(1.0)
                stability_D_list[pi].append(1.0)
                cross_C0_D_list[pi].append(float(np.mean(
                    np.sign(j1_D[pi]) * np.sign(baseline_J1_C[pi]))))
        elif baseline_J1_C is not None:
            for pi in range(n_probes):
                stability_C_list[pi].append(float(np.mean(
                    np.sign(j1_C[pi]) * np.sign(baseline_J1_C[pi]))))
                stability_D_list[pi].append(float(np.mean(
                    np.sign(j1_D[pi]) * np.sign(baseline_J1_D[pi]))))
                cross_C0_D_list[pi].append(float(np.mean(
                    np.sign(j1_D[pi]) * np.sign(baseline_J1_C[pi]))))
        else:
            for pi in range(n_probes):
                stability_C_list[pi].append(np.nan)
                stability_D_list[pi].append(np.nan)
                cross_C0_D_list[pi].append(np.nan)

        if baseline_J1_C is not None:
            sc = float(np.nanmean([s[-1] for s in stability_C_list]))
            sd = float(np.nanmean([s[-1] for s in stability_D_list]))
            cr = float(np.nanmean([s[-1] for s in cross_C0_D_list]))
        else:
            sc = sd = cr = float("nan")
        print(f"  batch {batch_count:5d}  "
              f"⟨stab_C⟩={sc:.3f}  ⟨stab_D⟩={sd:.3f}  ⟨C0↔D⟩={cr:.3f}  "
              f"⟨margin_D⟩={float(np.mean(margin_D_list[-1])):.3f}",
              flush=True)

    print("\nInitial measurement (batch 0)...", flush=True)
    record(0)

    print("\nTraining (Kassym config)...", flush=True)
    t_train = time.time()
    batch_count = 0
    train_key = jax.random.PRNGKey(SEED + 1)
    for epoch in range(N_EPOCHS):
        for xb, yb in ds:
            train_key = trainer.train_step(to_hwc(xb), yb, train_key)
            trainer.orchestrator = normalize_win(trainer.orchestrator)
            batch_count += 1
            if batch_count % MEASURE_EVERY == 0 or batch_count == total_batches:
                record(batch_count)
        print(f"  --- epoch {epoch + 1}/{N_EPOCHS} done ---", flush=True)
    print(f"\nDone in {time.time() - t_train:.1f}s", flush=True)

    measure_batches_arr = np.asarray(measure_batches)

    results = {
        "config": CONFIG,
        "config_label": "kassym (trial 14)",
        "probes_per_class": PROBES_PER_CLASS,
        "n_epochs": N_EPOCHS,
        "baseline_batch": BASELINE_BATCH,
        "probe_indices": probe_indices,
        "probe_classes": probe_classes.tolist(),
        "first_seen": list(map(int, first_seen)),
        "measure_batches": list(map(int, measure_batches_arr)),
        "margin_C": np.stack(margin_C_list, axis=1).tolist(),
        "margin_D": np.stack(margin_D_list, axis=1).tolist(),
        "stability_C_dot": [sc for sc in stability_C_list],
        "stability_D_dot": [sd for sd in stability_D_list],
        "cross_C0_to_D_t": [cc for cc in cross_C0_D_list],
        "overlap_CD": np.stack(overlap_CD, axis=1).tolist(),
    }
    out_path = results_dir / "cross_time_cd_kassym.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved to {out_path}")

    print(f"\n{'=' * 60}\nSUMMARY  (Kassym config, baseline {BASELINE_BATCH})\n{'=' * 60}")
    stab_C = np.array(stability_C_list)
    cross = np.array(cross_C0_D_list)
    margin = np.stack(margin_D_list, axis=1)
    print(f"{'batch':>6}  {'mean stab_C':>11}  {'mean cross':>10}  {'mean margin_D':>13}")
    for ti, b in enumerate(measure_batches_arr):
        sc = float(np.nanmean(stab_C[:, ti]))
        cr = float(np.nanmean(cross[:, ti]))
        md = float(np.mean(margin[:, ti]))
        print(f"{int(b):>6}  {sc:>11.4f}  {cr:>10.4f}  {md:>13.4f}")


if __name__ == "__main__":
    main()
