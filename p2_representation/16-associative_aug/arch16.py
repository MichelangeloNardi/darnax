"""Exp 16 — associative augmentation (fair split): config registry + builders.

Fairness fix over exps 9/10/15: those compared a split against a dense model of the SAME total
size, which gave the split LESS input/label fan-in -> unfair. Here we instead FIX the input
fan-in at 16 (matching the classic C16 baseline) and ADD capacity:
  - I : 16 input-only neurons (see W_in, not W_back) -- same input capacity as classic C16
  - L : a DISTINCT label-only subset (see W_back, not W_in)   [distinct from I]
  - N : recurrent-only "associative middle" (see neither directly)
  - all disjoint; J1 recurrence + W_out read ALL neurons.
Baseline = classic_c16 (16 neurons, each sees input AND label = the standard tuned model).

Question: on top of a full 16-input core, does adding a distinct label group + recurrent-only
associative neurons help? Swept at fixed I=16, L=8, N in {0,8,16,32}.

Weights note: vs a dense model of the same total M, this saves the W_in (and frozen W_back)
fan-in of the extra neurons (W_in = input_dim x 16, not x M) but J1 (~M^2) and W_out (~M) grow
the same -- so it is a cheaper way to add recurrent capacity.

Reuses exp-11 splitarch.build_explicit (arch.MaskedConv2D) + exp-10 arch for pooling/params.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))
sys.path.insert(0, str(REPO / "p2_representation" / "10-split_scale"))
sys.path.insert(0, str(REPO / "p2_representation" / "11-bptt_split"))

import common as cm  # noqa: E402
import splitarch as S  # exp-11: build_explicit  # noqa: E402

KSIZE = cm.KSIZE


def _cfg(C, nL):
    """I=16 input-only [0:16], L=nL distinct label-only [16:16+nL], N=rest recurrent-only."""
    return {"C": C, "I": list(range(16)), "L": list(range(16, 16 + nL))}


# name -> {C, I (input-only idx), L (label-only idx, distinct)}; N = the rest (recurrent-only)
CONFIGS = {
    "classic_c16": {"C": 16, "I": list(range(16)), "L": list(range(16))},  # all input+label (baseline)
    "aug_L8_N0":  _cfg(24, 8),   # I16 L8 N0
    "aug_L8_N8":  _cfg(32, 8),   # I16 L8 N8
    "aug_L8_N16": _cfg(40, 8),   # I16 L8 N16
    "aug_L8_N32": _cfg(56, 8),   # I16 L8 N32
}
NAMES = list(CONFIGS)


def channels_of(name):
    return CONFIGS[name]["C"]


def groups(name):
    """(I_idx, L_idx, N_idx) arrays; N = channels in neither I nor L."""
    c = CONFIGS[name]; C = c["C"]
    I, L = np.array(c["I"]), np.array(c["L"])
    N = np.array([k for k in range(C) if k not in set(I.tolist()) | set(L.tolist())])
    return I, L, N


def build(cfg, key, name):
    c = CONFIGS[name]
    return S.build_explicit(cfg, key, c["C"], c["I"], c["L"])


def win_mask(name):
    """(1,1,1,C) W_in channel-output mask (1 at input-only channels I)."""
    c = CONFIGS[name]; C = c["C"]
    m = jnp.zeros(C).at[jnp.asarray(c["I"])].set(1.0)
    return m[None, None, None, :]


def param_report(name):
    """Print nominal + effective trainable weight counts (W_in + J1-diag + W_out; W_back frozen).
    Highlights that W_in stays at |I|=16 fan-in while J1/W_out grow with total C."""
    c = CONFIGS[name]; C = c["C"]
    I, L, N = groups(name)
    kk = KSIZE * KSIZE
    win_eff = kk * 3 * len(I)                 # only |I| input channels
    j1 = kk * C * C - C                        # full recurrence minus frozen j_d diagonal
    wout = (cm.H // cm.POOL) * (cm.W // cm.POOL) * C * 10
    dense_win = kk * 3 * C                      # a dense model of the same total would have this
    print(f"  [{name:12s}] C={C:2d} I={len(I):2d} L={len(L):2d} N={len(N):2d} | "
          f"trainable={win_eff + j1 + wout} (W_in {win_eff} vs dense {dense_win}; "
          f"J1 {j1}; W_out {wout})")
    return {"name": name, "C": C, "sizes": [len(I), len(L), len(N)],
            "win_eff": win_eff, "j1": j1, "wout": wout,
            "trainable": win_eff + j1 + wout, "dense_win": dense_win}
