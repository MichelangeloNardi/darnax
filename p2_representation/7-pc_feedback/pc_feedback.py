"""Predictive-coding (PC) error-driven clamped feedback — shared infra for exp 6.

Mattia's idea: replace the STATIC label clamp  W_back(y)  with a CLOSED-LOOP error
field that depends on the current hidden state s_t. During the clamped phase:

    eps_t  = y01 - softmax( decode(s_t) )          # y01 = (y+1)/2, one-hot in {0,1}
    b_t    = g * project(eps_t)                      # g = beta * sqrt(H*W) = 32*beta
    field  = W_in x + J s_t + b_t                    # faithful: J1.reduce == sum
    s_t+1  = sign(field)                             # via orch._safe_activate (zero->+1)

As s_t comes to predict the label, eps_t -> 0 and the field vanishes, so the feedback
self-limits instead of hard-imprinting the label like the raw clamp.

Two INSTANT variants (leaky deferred):
  "wback"  tied. decode = mean_HW(s) @ W_back.W^T  (16->10, mean is the normalized
           adjoint of the spatial broadcast); project = eps @ W_back.W broadcast over
           H,W. Uses the RAW W_back matrix (skips ChannelWBack's +-1 (a,b) rescaling,
           which is meant for labels, not continuous errors).
  "wout"   decode = pool8x8(s) @ W_out.W  (256->10); project = unpool(eps @ W_out.W^T),
           i.e. each pooled feature's feedback spread uniformly over its 8x8 block.
           Uses the ONLINE W_out (a second, co-evolving closed loop).

NOT backprop: b_t is forward + transpose ops of existing frozen/online weights, local
in time, gradient-free. The perceptron/entropy local rules still train on the reached
C. Only the clamped phase changes; D = warmup->free is untouched, so it stays the same
inference state as every other experiment.

The gain uses N = H*W = 1024 (hidden spatial size per sample), so g = beta*sqrt(1024).
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import common as cm

H, W, C, P = cm.H, cm.W, cm.C, cm.POOL
N_GAIN = H * W  # mean-field N for the feedback gain: g = beta * sqrt(H*W)


# ── the closed-loop error field ───────────────────────────────────────────────

def _unpool(fb, n):
    """Spread each of the 256 pooled features uniformly over its 8x8 block.
    Inverse layout of cm.pool_j1: 256 == (H//P, W//P, C) flattened in C-order."""
    g = fb.reshape(n, H // P, W // P, C)                       # (N,4,4,16)
    g = jnp.broadcast_to(g[:, :, None, :, None, :],
                         (n, H // P, P, W // P, P, C))         # (N,4,8,4,8,16)
    return g.reshape(n, H, W, C)


def pc_field(orch, s, y, variant, g):
    """Closed-loop error field b_t = g * project( y01 - softmax(decode(s)) ).

    s : (N,H,W,C) current hidden spins;  y : (N,10) pm1 labels. Returns (N,H,W,C)."""
    y01 = (y + 1.0) * 0.5                              # one-hot in {0,1}
    if variant == "wback":
        Wb = orch.lmap[1][2].W                         # (10,16) raw projection matrix
        chan = s.mean(axis=(1, 2))                     # (N,16) mean-pool decode
        logits = chan @ Wb.T                           # (N,10)
        eps = y01 - jax.nn.softmax(logits, axis=-1)    # (N,10)
        b_ch = eps @ Wb                                # (N,16)
        b = jnp.broadcast_to(b_ch[:, None, None, :], s.shape)
    elif variant == "wout":
        Wo = orch.lmap[2][1].W                         # (256,10) online readout
        logits = cm.pool_j1(s) @ Wo                    # (N,10)
        eps = y01 - jax.nn.softmax(logits, axis=-1)    # (N,10)
        fb = eps @ Wo.T                                # (N,256) pooled-feature feedback
        b = _unpool(fb, s.shape[0])                    # (N,H,W,C)
    else:
        raise ValueError(f"unknown variant: {variant}")
    return g * b


def pc_clamped_step(orch, state, key, *, variant, g):
    """One clamped step with PC feedback replacing W_back(y).

    Faithful field aggregation (sum, == J1.reduce) + sign via the orchestrator's own
    helpers. y is read from state[2] (populated by state.init), so the roller keeps the
    (orch, state, key) -> (state, key) signature and is drop-in for the diagnostics."""
    s = state[1]
    y = state[2]
    win = orch.lmap[1][0](state[0])                    # W_in x
    j = orch.lmap[1][1](s)                             # J s_t  (incl. j_d self-term)
    b = pc_field(orch, s, y, variant, g)              # closed-loop error field
    field = win + j + b                                # == J1.reduce({win, j, b})
    field = orch._apply_field_momentum(state.fields[1], field)
    s_next = orch._safe_activate(1, field)            # sign + zero->zero_sign_value
    state = state.replace_field(1, field).replace_val(1, s_next)
    return state, key


# ── rollers / training ────────────────────────────────────────────────────────

def make_pc_roller(warmup, clamped, free, variant, g):
    """Jitted (orch, state, key) -> state rollout with a PC clamped phase. warmup/free
    use the standard forward step; only the clamped phase is PC. Drop-in for the C-roller
    slot of exp-3 diagnose() and exp-5 analyze()."""
    @eqx.filter_jit
    def roll(orch, state, key):
        for _ in range(warmup):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        for _ in range(clamped):
            state, key = pc_clamped_step(orch, state, key, variant=variant, g=g)
        for _ in range(free):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        return state
    return roll


def make_pc_train_step(state_tmpl, warmup, clamped, free, variant, g, optimizer):
    """Replicates DynamicalTrainer._train_step_impl with a PC clamped phase spliced in:
    warmup(forward) -> PC-clamped -> free(forward) -> orch.backward -> optax. The local
    rules (perceptron/entropy) train on the PC-reached C exactly as the baseline."""
    @eqx.filter_jit
    def step(orch, opt_state, x, y, key):
        state = state_tmpl.init(x, y)
        for _ in range(warmup):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        for _ in range(clamped):
            state, key = pc_clamped_step(orch, state, key, variant=variant, g=g)
        for _ in range(free):
            state, key = orch.step(state, rng=key, filter_messages="forward")
        grads = orch.backward(state, rng=key)           # default filter_messages="inference"
        params = eqx.filter(orch, eqx.is_inexact_array)
        gflt = eqx.filter(grads, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(gflt, opt_state, params=params)
        orch = eqx.apply_updates(orch, updates)
        return orch, opt_state, key
    return step


def train_pc_model(cfg, ds, seed, variant, beta, epochs):
    """Train one backbone with PC clamped feedback. Mirrors cm.train_epoch's per-epoch
    kernel decay; W_in/J1/W_out all trained by their local rules (W_out co-evolves and
    is read by the 'wout' feedback)."""
    key = jax.random.PRNGKey(seed)
    key, mk = jax.random.split(key)
    state, orch = cm.build_model(cfg, mk)
    opt, opt_state = cm.make_optimizer(orch, cfg)
    warmup = cfg.get("warmup_n_iter", 1)
    clamped, free = cfg["clamped_n_iter"], cfg["free_n_iter"]
    g = beta * (N_GAIN ** 0.5)
    step = make_pc_train_step(state, warmup, clamped, free, variant, g, opt)
    decay = cfg["kernel_decay_rate"]
    for _ in range(epochs):
        for xb, yb in ds:
            orch, opt_state, key = step(
                orch, opt_state, cm.to_hwc(xb), jnp.asarray(np.asarray(yb)), key)
        for path in [lambda o: o.lmap[1][0].kernel, lambda o: o.lmap[1][1].kernel]:
            orch = eqx.tree_at(path, orch, path(orch) * (1.0 - decay))
    return orch


# ── calibration helper ────────────────────────────────────────────────────────

def field_scales(orch, state_tmpl, x, y, variant, g):
    """RMS magnitude of the PC field b_t at t=0 (after one warmup step) vs the raw
    W_back(y) clamp, on one batch. Lets us see how beta*sqrt(N) compares to the
    label-injection strength the raw clamp uses (strength_back=1.47)."""
    state = state_tmpl.init(x, y)
    state, _ = orch.step(state, rng=jax.random.PRNGKey(0), filter_messages="forward")
    b = pc_field(orch, state[1], state[2], variant, g)
    raw = orch.lmap[1][2](state[2])                    # ChannelWBack(y), (N,H,W,C)
    rms = lambda a: float(jnp.sqrt(jnp.mean(a ** 2)))
    return rms(b), rms(raw)
