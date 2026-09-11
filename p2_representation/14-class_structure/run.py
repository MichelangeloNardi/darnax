"""Exp 14 — what class structure does D actually encode?

`old_experiments/experiments4/attractor_geometry.py` found that the class CENTROIDS of
the attractor states carry semantic structure (animals mutually positive 0.4-0.9,
airplane<->ship 0.72, auto<->truck 0.61, animals vs vehicles negative) while per-instance
attractors are nearly orthogonal (within-class cosine 0.067). That was a geometry
measurement only: no accuracy was ever attached to it.

This experiment attaches accuracy. On the inference state D (and on C for contrast):

  1. per-class accuracy + the 10x10 confusion matrix (Adam probe and the W_out head);
  2. error decomposition across a named grouping: what fraction of the 10-way errors
     stay INSIDE the group vs cross the boundary, against the rate expected if the
     confusion were uniform;
  3. accuracy of a binary readout for named groupings (vehicle/animal, road/sky-water,
     mammal/non-mammal, and the 4-way centroid clustering);
  4. an EXHAUSTIVE scan of all 511 non-trivial binary partitions of the 10 classes,
     ranked by balanced accuracy -> which dichotomies the representation encodes best,
     without deciding in advance which ones are interesting;
  5. average-linkage clustering of the 10 class centroids in D, cut at k = 2..5, with
     the induced partitions scored the same way.

Controls for every partition score (a dichotomy is only interesting if D decodes it
better than these do):
  - `pixels` : ridge on the raw 3072-d image;
  - `randD`  : the same D rollout through an UNTRAINED W_in / J1 (random init), which
               is dimension-matched to D at 256-d (cf. replicate/random/).

The 511-partition scan uses a closed-form ridge readout and the fact that a partition
target is constant within a class: with u_c the sum of training reps of class c and
A = X^T X + lam I, the ridge solution for sign vector s in {-1,+1}^10 is
W(s) = A^-1 U s, so all 511 fits are signed sums of 10 precomputed vectors. The whole
scan is two matmuls, not 511 fits.

Run (cluster):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ~/miniforge3/envs/darnax_hpc/bin/python p2_representation/14-class_structure/run.py
Reuse cached reps (skips training, analysis only):
  python p2_representation/14-class_structure/run.py --reuse-reps
Smoke:  python p2_representation/14-class_structure/run.py --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_default_matmul_precision", "high")

import equinox as eqx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))

import common as cm
import bptt_common as bc

CFG_PATH = REPO / "replicate" / "best_channel_entropy_cfg.json"
REPS_DIR = HERE / "reps"
RESULTS_DIR = HERE / "results"

CLASSES = ["airplane", "auto", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Named groupings. Each maps a name -> the set of class indices on the +1 side.
GROUPINGS = {
    "vehicle_vs_animal": {0, 1, 8, 9},                 # plane auto ship truck
    "road_vs_skywater": {1, 9},                        # within vehicles only (see below)
    "mammal_vs_other_animal": {3, 4, 5, 7},            # cat deer dog horse (within animals)
    "flies_vs_not": {0, 2},                            # airplane bird -- a shape/context cut
}
# Groupings scored on a SUBSET of classes only (the rest are dropped from the fit).
GROUPING_RESTRICT = {
    "road_vs_skywater": {0, 1, 8, 9},
    "mammal_vs_other_animal": {2, 3, 4, 5, 6, 7},
}

# The 4-way partition the attractor_geometry centroid heatmap suggests.
FOURWAY = {
    "road_vehicle": [1, 9],
    "sky_water_vehicle": [0, 8],
    "mammal": [3, 4, 5, 7],
    "non_mammal_animal": [2, 6],
}


# ── rep collection ────────────────────────────────────────────────────────────

def collect_C(orch, state_tmpl, ds, cfg, key):
    """Roll warmup -> clamped -> free = C (label injected through W_back)."""
    warmup = cfg.get("warmup_n_iter", 1)
    roll = eqx.filter_jit(cm.make_rollout(warmup, cfg["clamped_n_iter"], cfg["free_n_iter"]))
    Xtr, Ytr, key = cm.collect_reps(orch, state_tmpl, ds, roll, key)
    Xte, Yte, key = cm.collect_reps(orch, state_tmpl, ds.iter_test(), roll, key)
    return Xtr, Ytr, Xte, Yte, key


def collect_pixels(ds):
    """Raw 3072-d images for the pixel control, train and test."""
    def grab(it):
        X, Y = [], []
        for xb, yb in it:
            X.append(np.asarray(xb).reshape(len(xb), -1) * 2.0 - 1.0)
            Y.append(np.asarray(yb))
        return np.concatenate(X), np.concatenate(Y)
    return grab(ds), grab(ds.iter_test())


def train_local(cfg, ds, seed, epochs, max_batches=None):
    """Standard local-rule training (DynamicalTrainer; perceptron + entropy)."""
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = cm.build_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    trainer = cm.make_trainer(orch, state, opt, opt_state, cfg)
    for _ in range(epochs):
        if max_batches is None:
            trainer, key = cm.train_epoch(trainer, ds, key,
                                          decay_rate=cfg["kernel_decay_rate"])
        else:
            for i, (xb, yb) in enumerate(ds):
                if i >= max_batches:
                    break
                key = trainer.train_step(cm.to_hwc(xb), yb, key)
    return trainer.orchestrator, state, key


# ── ridge machinery (shared by every partition score) ─────────────────────────

def _bias(X):
    return np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)


class PartitionScorer:
    """Closed-form ridge scorer for every binary partition of the class set.

    Precomputes A^-1 U (d x K) on the training reps, where U[:, c] is the sum of the
    training reps of class c. The ridge solution for a sign vector s is then A^-1 U s,
    and the test scores are (Xte A^-1 U) s -- one (Nte x K) matrix, reused for all
    partitions.
    """

    def __init__(self, Xtr, ytr, Xte, yte, classes, lam=1.0):
        self.classes = list(classes)
        K = len(self.classes)
        Xtr, Xte = _bias(np.asarray(Xtr, np.float64)), _bias(np.asarray(Xte, np.float64))
        d = Xtr.shape[1]
        U = np.zeros((d, K))
        for j, c in enumerate(self.classes):
            U[:, j] = Xtr[ytr == c].sum(axis=0)
        A = Xtr.T @ Xtr + lam * np.eye(d)
        self.Ste = Xte @ np.linalg.solve(A, U)          # (Nte, K)
        self.yte = np.asarray(yte)
        self.col = {c: j for j, c in enumerate(self.classes)}
        self.te_col = np.array([self.col[c] for c in self.yte])
        self.n_per_class = np.array([(self.yte == c).sum() for c in self.classes])

    def score(self, pos):
        """Balanced accuracy of the ridge readout for the dichotomy `pos` vs rest.

        Returns balanced accuracy, plain accuracy, and the majority-class baseline
        (plain accuracy of always predicting the larger side).
        """
        s = np.array([1.0 if c in pos else -1.0 for c in self.classes])
        pred = np.sign(self.Ste @ s)
        pred[pred == 0] = 1.0
        truth = s[self.te_col]
        hit = pred == truth
        tpr = hit[truth > 0].mean() if (truth > 0).any() else 0.0
        tnr = hit[truth < 0].mean() if (truth < 0).any() else 0.0
        n_pos = self.n_per_class[s > 0].sum()
        maj = max(n_pos, len(truth) - n_pos) / len(truth)
        return dict(balanced=float(0.5 * (tpr + tnr)), acc=float(hit.mean()),
                    majority=float(maj), tpr=float(tpr), tnr=float(tnr))

    def multiway(self, groups):
        """One-vs-all ridge over a K-way grouping; returns accuracy + confusion."""
        names = list(groups)
        S = np.stack([np.array([1.0 if c in groups[g] else -1.0 for c in self.classes])
                      for g in names], axis=1)                       # (K, G)
        pred = (self.Ste @ S).argmax(1)
        gid = {c: names.index(g) for g in names for c in groups[g]}
        truth = np.array([gid[c] for c in self.yte])
        G = len(names)
        conf = np.zeros((G, G), int)
        for t, p in zip(truth, pred):
            conf[t, p] += 1
        per = {names[g]: float((pred[truth == g] == g).mean()) for g in range(G)}
        return dict(names=names, acc=float((pred == truth).mean()),
                    balanced=float(np.mean(list(per.values()))),
                    per_group=per, confusion=conf.tolist())


def all_partitions():
    """All 511 non-trivial dichotomies of 10 classes, deduplicated by complement."""
    out = []
    for k in range(1, 6):
        for S in itertools.combinations(range(10), k):
            if k == 5 and 0 not in S:
                continue
            out.append(frozenset(S))
    return out


# ── clustering (average linkage on cosine distance, no scipy) ─────────────────

def average_linkage(centroids):
    """Agglomerate 10 class centroids; return the merge order and the cut at each k."""
    Z = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - Z @ Z.T
    np.fill_diagonal(D, np.inf)
    clusters = {i: [i] for i in range(len(Z))}
    cuts = {len(clusters): [sorted(v) for v in clusters.values()]}
    merges = []
    while len(clusters) > 1:
        ks = list(clusters)
        best, bi, bj = np.inf, None, None
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                ia, ib = ks[a], ks[b]
                d = float(np.mean(D[np.ix_(clusters[ia], clusters[ib])]))
                if d < best:
                    best, bi, bj = d, ia, ib
        merges.append(dict(merged=[sorted(clusters[bi]), sorted(clusters[bj])],
                           distance=best))
        clusters[bi] = clusters[bi] + clusters[bj]
        del clusters[bj]
        cuts[len(clusters)] = [sorted(v) for v in clusters.values()]
    return merges, cuts


# ── per-class / confusion for a 10-way readout ───────────────────────────────

def tenway_report(pred, yte, pos_group):
    """Per-class accuracy, 10x10 confusion, and the in-group / cross-group split of
    the errors against the rate expected under uniform confusion."""
    conf = np.zeros((10, 10), int)
    for t, p in zip(yte, pred):
        conf[t, p] += 1
    per_class = {CLASSES[c]: float((pred[yte == c] == c).mean()) for c in range(10)}

    err = pred != yte
    same = np.array([(p in pos_group) == (t in pos_group) for t, p in zip(yte, pred)])
    n_err = int(err.sum())
    stay = int((err & same).sum())
    # under uniform confusion, an error from class t lands in t's own group with
    # probability (|group(t)| - 1) / 9
    exp_stay = float(np.mean([
        (len(pos_group) - 1) / 9 if t in pos_group else (10 - len(pos_group) - 1) / 9
        for t in yte[err]
    ])) if n_err else float("nan")
    return dict(acc=float((pred == yte).mean()), per_class=per_class,
                confusion=conf.tolist(), n_errors=n_err,
                errors_within_group=stay,
                frac_errors_within_group=stay / n_err if n_err else float("nan"),
                expected_frac_uniform=exp_stay)


# ── one seed ──────────────────────────────────────────────────────────────────

def analyse(Xtr, ytr, Xte, yte, args, probe_pred=None):
    """All partition analyses for one representation."""
    sc = PartitionScorer(Xtr, ytr, Xte, yte, range(10), lam=args.lam)
    out = {"ridge_10way": sc.multiway({CLASSES[c]: {c} for c in range(10)})["acc"]}

    out["named"] = {}
    for name, pos in GROUPINGS.items():
        keep = GROUPING_RESTRICT.get(name)
        if keep is None:
            out["named"][name] = sc.score(pos)
        else:
            m_tr = np.isin(ytr, list(keep))
            m_te = np.isin(yte, list(keep))
            sub = PartitionScorer(np.asarray(Xtr)[m_tr], ytr[m_tr],
                                  np.asarray(Xte)[m_te], yte[m_te],
                                  sorted(keep), lam=args.lam)
            out["named"][name] = sub.score(pos)
            out["named"][name]["restricted_to"] = sorted(keep)

    out["fourway"] = sc.multiway({k: set(v) for k, v in FOURWAY.items()})

    scan = []
    for S in all_partitions():
        r = sc.score(S)
        r["classes"] = sorted(S)
        r["names"] = [CLASSES[c] for c in sorted(S)]
        scan.append(r)
    scan.sort(key=lambda r: -r["balanced"])
    out["scan"] = scan

    cent = np.stack([np.asarray(Xte)[yte == c].mean(0) for c in range(10)])
    cn = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12)
    out["centroid_sim"] = (cn @ cn.T).tolist()
    merges, cuts = average_linkage(cent)
    out["linkage"] = merges
    out["cuts"] = {
        str(k): dict(groups=[[CLASSES[c] for c in g] for g in v],
                     score=sc.multiway({f"g{i}": set(g) for i, g in enumerate(v)}))
        for k, v in cuts.items() if 2 <= k <= 5
    }

    if probe_pred is not None:
        out["tenway"] = tenway_report(probe_pred, yte, GROUPINGS["vehicle_vs_animal"])
    return out


def run_seed(cfg, ds, seed, args):
    t0 = time.time()
    # the epoch count is part of the name so a --smoke cache can never be picked up
    # by a full --reuse-reps run
    cache = REPS_DIR / f"seed{seed}_ep{args.epochs}.npz"
    if args.reuse_reps and cache.exists():
        z = np.load(cache)
        print(f"  [seed {seed}] reusing cached reps {cache.name}")
        D = {k: z[k] for k in z.files}
    else:
        print(f"  [seed {seed}] training {args.epochs} epochs ...")
        orch, state, key = train_local(cfg, ds, seed, args.epochs, args.max_batches)
        Xtr_D, Ytr_D, Xte_D, Yte_D, key = bc.collect_D(orch, state, ds, cfg, key)
        Xtr_C, Ytr_C, Xte_C, Yte_C, key = collect_C(orch, state, ds, cfg, key)

        rk = jax.random.PRNGKey(10_000 + seed)
        rk, mk = jax.random.split(rk)
        state_r, orch_r = cm.build_model(cfg, mk)
        Xtr_R, Ytr_R, Xte_R, Yte_R, rk = bc.collect_D(orch_r, state_r, ds, cfg, rk)

        W = np.asarray(orch.lmap[2][1].W)
        D = dict(
            Xtr_D=np.asarray(Xtr_D, np.float32), ytr_D=np.argmax(Ytr_D, 1),
            Xte_D=np.asarray(Xte_D, np.float32), yte_D=np.argmax(Yte_D, 1),
            Xtr_C=np.asarray(Xtr_C, np.float32), ytr_C=np.argmax(Ytr_C, 1),
            Xte_C=np.asarray(Xte_C, np.float32), yte_C=np.argmax(Yte_C, 1),
            Xtr_R=np.asarray(Xtr_R, np.float32), ytr_R=np.argmax(Ytr_R, 1),
            Xte_R=np.asarray(Xte_R, np.float32), yte_R=np.argmax(Yte_R, 1),
            wout=W,
        )
        REPS_DIR.mkdir(exist_ok=True)
        np.savez_compressed(cache, **D)
        print(f"  [seed {seed}] reps cached ({cm.fmt(time.time() - t0)})")

    res = {}
    # 10-way Adam probe on D -> the per-class / confusion numbers
    curve = bc.offline_probe(D["Xtr_D"], D["ytr_D"], D["Xte_D"], D["yte_D"],
                             args.probe_epochs)
    res["probe_D_curve"] = curve
    probe_pred = _probe_predictions(D["Xtr_D"], D["ytr_D"], D["Xte_D"], args.probe_epochs)
    res["head_D"] = tenway_report(
        (D["Xte_D"] @ D["wout"]).argmax(1), D["yte_D"], GROUPINGS["vehicle_vs_animal"])

    res["D"] = analyse(D["Xtr_D"], D["ytr_D"], D["Xte_D"], D["yte_D"], args,
                       probe_pred=probe_pred)
    res["C"] = analyse(D["Xtr_C"], D["ytr_C"], D["Xte_C"], D["yte_C"], args)
    res["randD"] = analyse(D["Xtr_R"], D["ytr_R"], D["Xte_R"], D["yte_R"], args)
    res["seconds"] = time.time() - t0
    return res


def _probe_predictions(Xtr, ytr, Xte, epochs):
    """Same Adam probe as bc.offline_probe, but returning test predictions."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    dev = bc.DEVICE
    probe = nn.Linear(Xtr.shape[1], 10, bias=False).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=cm.PROBE_WD)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(),
                                      torch.from_numpy(ytr).long()),
                        batch_size=256, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        return probe(torch.from_numpy(Xte).float().to(dev)).argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--reuse-reps", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.epochs = 1; args.probe_epochs = 2; args.max_batches = 6

    cfg = cm.load_cfg(CFG_PATH)
    ds = cm.get_dataset(batch_size=32)
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # pixel control, computed once (independent of seed)
    print("pixel control ...")
    (Xtr_p, Ytr_p), (Xte_p, Yte_p) = collect_pixels(ds)
    if args.max_batches is not None:
        n = args.max_batches * 32
        Xtr_p, Ytr_p, Xte_p, Yte_p = Xtr_p[:n], Ytr_p[:n], Xte_p[:n], Yte_p[:n]
    pixels = analyse(Xtr_p, np.argmax(Ytr_p, 1), Xte_p, np.argmax(Yte_p, 1), args)

    out = {"config": "best_channel_entropy", "epochs": args.epochs, "lam": args.lam,
           "classes": CLASSES, "groupings": {k: sorted(v) for k, v in GROUPINGS.items()},
           "fourway": FOURWAY, "pixels": pixels, "seeds": {}}
    for seed in args.seeds:
        out["seeds"][str(seed)] = run_seed(cfg, ds, seed, args)
        print(f"  [seed {seed}] done ({cm.fmt(time.time() - t0)})")

    name = args.out or ("class_structure_smoke.json" if args.smoke
                        else "class_structure.json")
    (RESULTS_DIR / name).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS_DIR / name}  (total {cm.fmt(time.time() - t0)})")

    # console summary
    for seed, r in out["seeds"].items():
        print(f"\n=== seed {seed} ===")
        print(f"  probe_D (10-way, Adam)   {r['probe_D_curve'][-1]:.4f}")
        print(f"  head_D  (10-way, W_out)  {r['head_D']['acc']:.4f}")
        t = r["D"]["tenway"]
        print(f"  errors staying inside the vehicle/animal group: "
              f"{t['frac_errors_within_group']:.3f} "
              f"(uniform-confusion expectation {t['expected_frac_uniform']:.3f})")
        print("  named dichotomies (balanced acc on D | randD | pixels):")
        for k in GROUPINGS:
            print(f"    {k:26s} {r['D']['named'][k]['balanced']:.4f} | "
                  f"{r['randD']['named'][k]['balanced']:.4f} | "
                  f"{out['pixels']['named'][k]['balanced']:.4f}")
        print("  top 8 of the 511-partition scan on D (balanced | randD | pixels | lift):")
        rand_by = {tuple(x["classes"]): x["balanced"] for x in r["randD"]["scan"]}
        pix_by = {tuple(x["classes"]): x["balanced"] for x in out["pixels"]["scan"]}
        for x in r["D"]["scan"][:8]:
            k = tuple(x["classes"])
            print(f"    {'+'.join(x['names']):38s} {x['balanced']:.4f} | "
                  f"{rand_by[k]:.4f} | {pix_by[k]:.4f} | "
                  f"{x['balanced'] - max(rand_by[k], pix_by[k]):+.4f}")


if __name__ == "__main__":
    main()
