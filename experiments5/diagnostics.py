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


# ── internal helpers ──────────────────────────────────────────────────────────

def _cosine_per_image(a, b):
    na = np.linalg.norm(a, axis=1) + 1e-8
    nb = np.linalg.norm(b, axis=1) + 1e-8
    return (a * b).sum(axis=1) / (na * nb)


def _run_abcd(orch, state_template, x, y, rng, warmup_n, clamped_n, free_n):
    """Run warmup→clamped→free and free-from-A, return (j1_a, j1_b, j1_c, j1_d).

    Each array has shape (N, D) float32 where D = H*W*C (J1 flattened).
    """
    state_0 = state_template.init(x, y)

    state_a, rng_a = state_0, rng
    for _ in range(warmup_n):
        state_a, rng_a = orch.step(state_a, rng=rng_a, filter_messages="forward")

    state_b, rng_b = state_a, rng_a
    for _ in range(clamped_n):
        state_b, rng_b = orch.step(state_b, rng=rng_b, filter_messages="all")

    state_c, rng_c = state_b, rng_b
    for _ in range(free_n):
        state_c, rng_c = orch.step(state_c, rng=rng_c, filter_messages="forward")

    state_d, rng_d = state_a, rng_a
    for _ in range(free_n):
        state_d, rng_d = orch.step(state_d, rng=rng_d, filter_messages="forward")

    N = x.shape[0]
    return tuple(
        np.asarray(s[1]).reshape(N, -1).astype(np.float32)
        for s in (state_a, state_b, state_c, state_d)
    )


def _pairwise_cosine_matrix(j1_list):
    """(n, n) mean per-image cosine similarity matrix over a list of (N, D) arrays."""
    n = len(j1_list)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            mat[i, j] = float(_cosine_per_image(j1_list[i], j1_list[j]).mean())
    return mat


# ── public API ────────────────────────────────────────────────────────────────

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

    flat = [s.reshape(s.shape[0], -1).astype(np.float32) for s in states]
    sim = _pairwise_cosine_matrix(flat)
    return sim, labels, states


def collect_weight_norms(orch):
    """Frobenius norms of Win, J1, and W_back weight matrices.

    Returns
    -------
    dict with keys "win", "j1", "wback" (float each).
    """
    def fnorm(a):
        return float(np.linalg.norm(np.asarray(a).ravel()))

    return {
        "win":   fnorm(orch.lmap[1][0].kernel),
        "j1":    fnorm(orch.lmap[1][1].kernel),
        "wback": fnorm(orch.lmap[1][2].W),
    }


def collect_field_contributions(orch, state_template, x, y, rng, warmup_n):
    """Fractional Win/J/W_back contributions to the J1 field after warmup.

    Returns
    -------
    dict with:
      "win", "j1", "wback"           : float in [0, 1] — fractional |h| contribution
      "win_abs", "j1_abs", "wback_abs" : float — mean absolute field magnitudes
    """
    state = state_template.init(x, y)
    for _ in range(warmup_n):
        state, rng = orch.step(state, rng=rng, filter_messages="forward")

    h_win   = np.asarray(orch.lmap[1][0](state[0]))
    h_j     = np.asarray(orch.lmap[1][1](state[1]))
    h_wback = np.asarray(orch.lmap[1][2](state[2]))

    m_win, m_j, m_wback = map(lambda h: float(np.abs(h).mean()), (h_win, h_j, h_wback))
    total = m_win + m_j + m_wback + 1e-12

    return {
        "win":       m_win / total,
        "j1":        m_j / total,
        "wback":     m_wback / total,
        "win_abs":   m_win,
        "j1_abs":    m_j,
        "wback_abs": m_wback,
    }


def collect_abcd_states(orch, state_template, x, y, rng, warmup_n, clamped_n, free_n):
    """All 6 pairwise cosine similarities between ABCD attractor states.

    A = after warmup, B = after clamped, C = free(B), D = free(A).

    Returns
    -------
    dict with "{xy}_sim_mean" and "{xy}_sim_std" for xy in {ab,ac,ad,bc,bd,cd}.
    """
    j1_a, j1_b, j1_c, j1_d = _run_abcd(orch, state_template, x, y, rng,
                                         warmup_n, clamped_n, free_n)
    j1 = {"a": j1_a, "b": j1_b, "c": j1_c, "d": j1_d}
    result = {}
    for p, q in [("a","b"),("a","c"),("a","d"),("b","c"),("b","d"),("c","d")]:
        sims = _cosine_per_image(j1[p], j1[q])
        result[f"{p}{q}_sim_mean"] = float(sims.mean())
        result[f"{p}{q}_sim_std"]  = float(sims.std())
    return result


def collect_abcd_8x8(orch_before, orch_after, state_template, x, y, rng,
                     warmup_n, clamped_n, free_n):
    """8×8 pairwise cosine similarity matrix: ABCD (before update) × A'B'C'D' (after).

    The cross-block (rows A-D vs cols A'-D') shows how much the network's fixed
    points shift after one training step on this batch.

    Returns
    -------
    dict with:
      "matrix" : list[list[float]], shape 8×8
      "labels" : ["A","B","C","D","A'","B'","C'","D'"]
    """
    j1_before = _run_abcd(orch_before, state_template, x, y, rng, warmup_n, clamped_n, free_n)
    j1_after  = _run_abcd(orch_after,  state_template, x, y, rng, warmup_n, clamped_n, free_n)
    mat = _pairwise_cosine_matrix(list(j1_before) + list(j1_after))
    return {
        "matrix": mat.tolist(),
        "labels": ["A", "B", "C", "D", "A'", "B'", "C'", "D'"],
    }
