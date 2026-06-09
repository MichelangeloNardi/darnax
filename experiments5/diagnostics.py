"""experiments5/diagnostics.py

Diagnostic collection functions for the channel-entropy CIFAR-10 architecture.
All functions call orchestrator.step() directly (no lax.scan) so intermediate
states are accessible at every step.

When step() is called without t_win/t_back, defaults are 0, so lambda_win**0=1.0
— no decay is applied during diagnostic collection. This is intentional: we
measure the full-contribution architecture regardless of training mode.
"""

from __future__ import annotations

import numpy as np


def collect_autocorr_matrix(orch, state_template, x, y, rng, warmup_n, clamped_n, free_n):
    """Step warmup→clamped→free one step at a time and compute pairwise J1 cosine sims.

    Returns
    -------
    sim : np.ndarray, shape (n_steps, n_steps)
        Pairwise cosine similarities. n_steps = 1 + warmup_n + clamped_n + free_n.
    labels : list[str]
        Step labels, e.g. ["init", "W1", "C1", ..., "C5", "F1", ..., "F6"].
    states : list[np.ndarray]
        Raw J1 activation arrays (B, H, W, C) at each step.
    """
    state = state_template.init(x, y)
    states = [np.asarray(state[1])]
    labels = ["init"]

    for i in range(warmup_n):
        state, rng = orch.step(state, rng=rng, filter_messages="forward")
        states.append(np.asarray(state[1]))
        labels.append(f"W{i + 1}")

    for i in range(clamped_n):
        state, rng = orch.step(state, rng=rng, filter_messages="all")
        states.append(np.asarray(state[1]))
        labels.append(f"C{i + 1}")

    for i in range(free_n):
        state, rng = orch.step(state, rng=rng, filter_messages="forward")
        states.append(np.asarray(state[1]))
        labels.append(f"F{i + 1}")

    n = len(states)
    flat = [s.reshape(s.shape[0], -1).astype(np.float32) for s in states]  # (B, D)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            a, b = flat[i], flat[j]
            na = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
            nb = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
            sim[i, j] = float(((a / na) * (b / nb)).sum(axis=1).mean())

    return sim, labels, states


def collect_weight_norms(orch):
    """Frobenius norms of Win, J1, and W_back weight matrices.

    Returns
    -------
    dict with keys "win", "j1", "wback" (float each).
    """
    def fnorm(a):
        return float(np.linalg.norm(np.asarray(a).ravel()))

    win_k = orch.lmap[1][0].kernel   # (kH, kW, C_in, C_out)
    j1_k  = orch.lmap[1][1].kernel   # (kH, kW, C, C)
    wback_w = orch.lmap[1][2].W      # (C_out, C) — ChannelWBack inherits W from FullyConnected

    return {"win": fnorm(win_k), "j1": fnorm(j1_k), "wback": fnorm(wback_w)}


def collect_field_contributions(orch, state_template, x, y, rng, warmup_n):
    """Measure fractional Win/J/W_back contributions to the J1 field.

    Runs warmup so J1 has a non-trivial state, then evaluates each source
    module independently at that state.

    Returns
    -------
    dict with:
      "win", "j1", "wback" : float in [0, 1] — fractional contribution to mean |h|
      "win_abs", "j1_abs", "wback_abs" : float — mean absolute field magnitudes
    """
    state = state_template.init(x, y)
    for _ in range(warmup_n):
        state, rng = orch.step(state, rng=rng, filter_messages="forward")

    h_win   = np.asarray(orch.lmap[1][0](state[0]))   # (B, H, W, C)
    h_j     = np.asarray(orch.lmap[1][1](state[1]))   # (B, H, W, C)
    h_wback = np.asarray(orch.lmap[1][2](state[2]))   # (B, H, W, C)

    m_win   = float(np.abs(h_win).mean())
    m_j     = float(np.abs(h_j).mean())
    m_wback = float(np.abs(h_wback).mean())
    total   = m_win + m_j + m_wback + 1e-12

    return {
        "win":       m_win / total,
        "j1":        m_j / total,
        "wback":     m_wback / total,
        "win_abs":   m_win,
        "j1_abs":    m_j,
        "wback_abs": m_wback,
    }


def collect_abcd_states(orch, state_template, x, y, rng, warmup_n, clamped_n, free_n):
    """Collect ABCD attractor states and per-image C-D cosine similarity.

    A = after warmup (no label feedback yet)
    B = after clamped (label-informed)
    C = free from B  (correct attractor path)
    D = free from A  (autonomous path, no clamping)

    C-D similarity: did the network converge to the same fixed point autonomously?
    High C-D → class attractors exist independently of label feedback.

    Returns
    -------
    dict with:
      "cd_sim_mean", "cd_sim_std" : float — mean/std of per-image cosine sim(C, D)
      "bc_sim_mean", "bc_sim_std" : float — mean/std of per-image cosine sim(B, C)
    """
    state_0 = state_template.init(x, y)

    # A: after warmup
    state_a, rng_a = state_0, rng
    for _ in range(warmup_n):
        state_a, rng_a = orch.step(state_a, rng=rng_a, filter_messages="forward")

    # B: clamped from A
    state_b, rng_b = state_a, rng_a
    for _ in range(clamped_n):
        state_b, rng_b = orch.step(state_b, rng=rng_b, filter_messages="all")

    # C: free from B
    state_c, rng_c = state_b, rng_b
    for _ in range(free_n):
        state_c, rng_c = orch.step(state_c, rng=rng_c, filter_messages="forward")

    # D: free from A (skip clamping) — reuse rng_a for a fair comparison
    state_d, rng_d = state_a, rng_a
    for _ in range(free_n):
        state_d, rng_d = orch.step(state_d, rng=rng_d, filter_messages="forward")

    B = x.shape[0]
    j1_c = np.asarray(state_c[1]).reshape(B, -1).astype(np.float32)
    j1_d = np.asarray(state_d[1]).reshape(B, -1).astype(np.float32)
    j1_b = np.asarray(state_b[1]).reshape(B, -1).astype(np.float32)

    def cosine_per_image(a, b):
        na = np.linalg.norm(a, axis=1) + 1e-8
        nb = np.linalg.norm(b, axis=1) + 1e-8
        return (a * b).sum(axis=1) / (na * nb)

    cd = cosine_per_image(j1_c, j1_d)
    bc = cosine_per_image(j1_b, j1_c)

    return {
        "cd_sim_mean": float(cd.mean()),
        "cd_sim_std":  float(cd.std()),
        "bc_sim_mean": float(bc.mean()),
        "bc_sim_std":  float(bc.std()),
    }
