"""Shared infra for the problem-2 BPTT ceiling experiments.

Problem 1 (../p1_readout_gap/) established that the readout can be brought to
~0.44 (W_out) / ~0.46 (probe), but both plateau at the ~0.46 ceiling of the
*inference state D*. Problem 2 asks how high that ceiling could be if the hidden
recurrent representation (W_in / J1) were optimised with REAL gradients instead
of the gradient-free local rule.

This module is the analogue of p1's common.py, but for a **differentiable**
forward. It is explicitly NOT darnax-faithful — it is a diagnostic upper bound.

Key pieces
----------
- `differentiable rollout to D`: warmup -> free (forward messages only; no W_back,
  no W_out feedback) reproducing orch.step(filter_messages="forward"), but with the
  non-differentiable `jnp.sign` activation replaced by a surrogate φ.
- two surrogates (a bracket): `tanh(beta*x)` (β annealed up) and a straight-through
  estimator (`sign` forward, identity-gradient backward).
- the **ceiling is always measured on the true hard-sign dynamics**: after BPTT we
  rebuild a real SequentialOrchestrator with the optimised kernels and collect D
  reps with cm.make_rollout / cm.collect_reps, then fit the perceptron W_out and an
  Adam probe on those reps (offline_wout / offline_probe, ported from p1 exp 4).

The model / optimiser / dataset / rollout / rep-collection live in p1's common.py;
the caller is responsible for the sys.path inserts (src + p1_readout_gap).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import common as cm  # p1_readout_gap/common.py (caller puts it on sys.path)
from darnax.utils.perceptron_rule import perceptron_rule_backward

WOUT_THRESHOLD = 5.0  # matches PooledFlattenFC threshold in cm.build_model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── surrogate activations ─────────────────────────────────────────────────────
# `kind` is static (baked into the jitted step's closure); `beta` is a traced
# scalar so annealing it does not trigger recompiles.

def apply_phi(kind: str, beta, x):
    """Surrogate for the recurrent layer's sign activation.

    kind="tanh": soft relaxation tanh(beta*x), used in BOTH passes.
    kind="ste" : straight-through estimator — forward is the true hard sign
                 (zeros -> +1, matching the orchestrator's zero_sign_value),
                 backward gradient is the identity.
    """
    if kind == "tanh":
        return jnp.tanh(beta * x)
    if kind == "ste":
        hard = jnp.sign(x)
        hard = jnp.where(hard == 0, 1.0, hard)
        return jax.lax.stop_gradient(hard - x) + x
    raise ValueError(f"unknown surrogate kind: {kind}")


# ── differentiable rollout to D ───────────────────────────────────────────────

def extract_kernels(orch):
    """Pull the three trainable tensors out of a (random-init) orchestrator.
    Returns (win, j1, wout) modules and the params dict we optimise."""
    win = orch.lmap[1][0]
    j1 = orch.lmap[1][1]
    wout = orch.lmap[2][1]
    params = {"win": win.kernel, "j1": j1.kernel, "wout": wout.W}
    return win, j1, wout, params


def diff_forward(params, win, j1, wout, x, n_steps, kind, beta):
    """Differentiable rollout to D. Returns (logits, h) where h is the final
    (soft) hidden state. W_in(x) is constant across forward steps (state[0]=x is
    never updated), so it is computed once."""
    win_ = eqx.tree_at(lambda m: m.kernel, win, params["win"])
    j1_ = eqx.tree_at(lambda m: m.kernel, j1, params["j1"])
    wout_ = eqx.tree_at(lambda m: m.W, wout, params["wout"])

    win_msg = win_(x)                       # (N, H, W, C), constant across steps
    h = jnp.zeros_like(win_msg)             # state[1] init = zeros
    for _ in range(n_steps):
        field = win_msg + j1_(h)            # reduce = sum of forward messages
        h = apply_phi(kind, beta, field)
    return wout_(h), h


def _ce_loss(params, win, j1, wout, x, y_idx, n_steps, kind, beta):
    logits, _ = diff_forward(params, win, j1, wout, x, n_steps, kind, beta)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y_idx).mean()


def make_bptt_step(win, j1, wout, n_steps, kind, opt):
    """Build a jitted BPTT step. The J1 gradient is masked by update_mask and the
    j_d constraint re-asserted each step, so the frozen self-coupling cannot move.
    `kind` is static; `beta` is a traced argument."""
    grad_fn = jax.value_and_grad(_ce_loss)
    j1_mask = j1.update_mask

    @eqx.filter_jit
    def step(params, opt_state, x, y_idx, beta):
        loss, grads = grad_fn(params, win, j1, wout, x, y_idx, n_steps, kind, beta)
        grads = {**grads, "j1": grads["j1"] * j1_mask}  # keep j_d diagonal frozen
        upd, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, upd)
        params = {**params, "j1": j1._apply_jd_constraint(params["j1"])}  # safety net
        return params, opt_state, loss

    return step


def make_soft_eval(win, j1, wout, n_steps, kind):
    """Jitted soft-rollout logits (for the internal soft-accuracy sanity check)."""
    @eqx.filter_jit
    def soft_logits(params, x, beta):
        logits, _ = diff_forward(params, win, j1, wout, x, n_steps, kind, beta)
        return logits
    return soft_logits


# ── checkpoint-selection proxy: hard-sign linear separability ─────────────────
# Per-epoch metric for picking the best backbone. Rolls the TRUE hard-sign
# dynamics (independent of W_out), then scores linear separability with a quick
# closed-form ridge readout on a held-out TRAIN subset. This tracks the reported
# Adam-probe ceiling far better than the soft-rollout accuracy, which at low beta
# overestimates how separable the *binary* state actually is.

def make_hard_reps(win, j1, n_steps):
    """Jitted pooled D reps from the true hard-sign dynamics (zeros -> +1, matching
    the orchestrator). W_out-independent; identical dynamics to collect_D."""
    @eqx.filter_jit
    def reps(params, x):
        win_ = eqx.tree_at(lambda m: m.kernel, win, params["win"])
        j1_ = eqx.tree_at(lambda m: m.kernel, j1, params["j1"])
        win_msg = win_(x)
        h = jnp.zeros_like(win_msg)
        for _ in range(n_steps):
            hard = jnp.sign(win_msg + j1_(h))
            h = jnp.where(hard == 0, 1.0, hard)
        return cm.pool_j1(h)  # (N, 256)
    return reps


@eqx.filter_jit
def ridge_acc(Xf, Yf_oh, Xe, ye_idx, lam=1.0):
    """Closed-form ridge readout fit on (Xf, Yf_oh), accuracy on (Xe, ye_idx)."""
    d = Xf.shape[1]
    W = jnp.linalg.solve(Xf.T @ Xf + lam * jnp.eye(d), Xf.T @ Yf_oh)
    return jnp.mean((Xe @ W).argmax(1) == ye_idx)


# ── hard-sign ceiling measurement (the headline) ──────────────────────────────

def orch_with_kernels(orch, params):
    """Return a copy of `orch` with W_in/J1 replaced by the BPTT-optimised
    kernels. W_out is left as-is (it is refit by offline_wout / offline_probe)."""
    orch = eqx.tree_at(lambda o: o.lmap[1][0].kernel, orch, params["win"])
    orch = eqx.tree_at(lambda o: o.lmap[1][1].kernel, orch, params["j1"])
    return orch


def collect_D(orch, state_tmpl, ds, cfg, key):
    """Roll the TRUE hard-sign dynamics (warmup -> free = D) over train and test.
    Reps and labels come from the SAME pass (the train set reshuffles every
    iteration — see the p1 exp-4 label-mismatch bug)."""
    warmup = cfg.get("warmup_n_iter", 1)
    free = cfg["free_n_iter"]
    roll = eqx.filter_jit(cm.make_rollout(warmup, 0, free))  # clamped_n=0 -> D
    Xtr, Ytr, key = cm.collect_reps(orch, state_tmpl, ds, roll, key)
    Xte, Yte, key = cm.collect_reps(orch, state_tmpl, ds.iter_test(), roll, key)
    return Xtr, Ytr, Xte, Yte, key


# ── readout fits on collected reps (ported from p1 exp 4) ──────────────────────

def wout_acc(W, X, y_idx):
    return float(((X @ np.asarray(W)).argmax(1) == y_idx).mean())


def offline_wout(cfg, X, Ypm1, W0, Xte, yte_idx, epochs):
    """Perceptron-rule W_out on fixed reps X (pm1 labels). Returns acc curve on D."""
    mom = cfg["momentum"]
    opt = optax.sgd(cfg["lr_wout"], momentum=mom) if mom > 0 else optax.sgd(cfg["lr_wout"])
    W = jnp.asarray(W0)
    opt_state = opt.init(W)
    th = jnp.asarray(WOUT_THRESHOLD)

    @eqx.filter_jit
    def pstep(W, opt_state, Xb, Yb):
        grad = perceptron_rule_backward(Xb, Yb, Xb @ W, th)
        upd, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(W, upd), opt_state

    Xj, Yj = jnp.asarray(X), jnp.asarray(Ypm1)
    N, bs = X.shape[0], 32
    curve = []
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            W, opt_state = pstep(W, opt_state, Xj[b], Yj[b])
        curve.append(wout_acc(W, Xte, yte_idx))
    return curve


def offline_probe(X, y_idx, Xte, yte_idx, epochs):
    """Fresh Adam linear probe on fixed reps X. Returns acc curve on D."""
    probe = nn.Linear(X.shape[1], 10, bias=False).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=cm.PROBE_WD)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(X)).float(),
                      torch.from_numpy(np.asarray(y_idx)).long()),
        batch_size=256, shuffle=True,
    )
    Xte_t = torch.from_numpy(np.asarray(Xte)).float().to(DEVICE)
    curve = []
    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(probe(xb), yb).backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(Xte_t).argmax(1).cpu().numpy()
        curve.append(float((pred == yte_idx).mean()))
    return curve
