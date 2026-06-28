"""Split-routing at VARIABLE channel count C (exp 10).

Generalizes exp-9's channel partition to arbitrary C so we can preserve the original
input capacity (|I|=16 input-driven channels) while scaling total channels. Same masking
(not shrink): W_in confined to group I via a masked Hebbian update (MaskedConv2D, the
j_d-diagonal pattern); frozen W_back confined to group L by zeroing its columns. J1
(full groups=1 recurrence over all C) and W_out (reads all C pooled) are unchanged.

Configs (name -> C, |I|, |L|; group N = channels in neither I nor L):
  baseline_C16          C16  I16 L16   all channels get input+label (current baseline)
  standard_C24          C24  I24 L24   scale-only control
  split_C24_I16_L8_N0   C24  I16 L8    disjoint I/L, N=0
  split_C32_I16_L8_N8   C32  I16 L8    disjoint, N=8 (neither directly)
  standard_C32          C32  I32 L32   scale-only control for split_C32

"all-both" configs (|I|=|L|=C) reproduce the standard model at that C (masks all ones ->
MaskedConv2D is a bit-identical no-op). No darnax-core edits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))

import common as cm  # noqa: E402
from darnax.layer_maps.sparse import LayerMap  # noqa: E402
from darnax.modules.conv.conv import Conv2D, Conv2DRecurrentDiscrete  # noqa: E402
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC  # noqa: E402
from darnax.modules.input_output import OutputLayer  # noqa: E402
from darnax.orchestrators.sequential import SequentialOrchestrator  # noqa: E402
from darnax.states.sequential import SequentialState  # noqa: E402

KSIZE, H, W, POOL = cm.KSIZE, cm.H, cm.W, cm.POOL


# ── config registry: name -> (C, |I|, |L|) ────────────────────────────────────
CONFIGS: dict[str, tuple[int, int, int]] = {
    "baseline_C16":        (16, 16, 16),
    "standard_C24":        (24, 24, 24),
    "split_C24_I16_L8_N0": (24, 16, 8),
    "split_C32_I16_L8_N8": (32, 16, 8),
    "standard_C32":        (32, 32, 32),
}
NAMES = list(CONFIGS)


def channels_of(name: str) -> int:
    return CONFIGS[name][0]


def is_all_both(name: str) -> bool:
    C, nI, nL = CONFIGS[name]
    return nI == C and nL == C


def group_indices(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(I_idx, L_idx, N_idx). all-both configs: I=L=all channels, N=empty (non-disjoint).
    split configs: I=[0:|I|], L=[|I|:|I|+|L|], N=rest -- disjoint."""
    C, nI, nL = CONFIGS[name]
    if is_all_both(name):
        allc = np.arange(C)
        return allc, allc, np.array([], dtype=int)
    assert nI + nL <= C, f"{name}: |I|+|L|={nI + nL} > C={C}"
    I = np.arange(0, nI)
    L = np.arange(nI, nI + nL)
    N = np.arange(nI + nL, C)
    return I, L, N


def group_masks(name: str) -> tuple[Array, Array]:
    C = channels_of(name)
    I, L, _ = group_indices(name)
    I_mask = jnp.zeros(C).at[jnp.asarray(I)].set(1.0)
    L_mask = jnp.zeros(C).at[jnp.asarray(L)].set(1.0)
    return I_mask, L_mask


# ── masked W_in (channel-output mask on a full 3->C conv) ─────────────────────
class MaskedConv2D(Conv2D):
    """Conv2D whose output is confined to a fixed channel subset. Non-target kernel
    columns start at zero and their Hebbian updates are masked, so they stay zero
    (mirrors Conv2DRecurrentDiscrete's j_d update_mask)."""

    out_mask: Array  # (out_channels,)

    def __init__(self, *args, out_mask, **kwargs):
        super().__init__(*args, **kwargs)
        m = jnp.asarray(out_mask, dtype=self.kernel.dtype)
        self.out_mask = m
        self.kernel = self.kernel * m[None, None, None, :]

    def backward(self, x, y, y_hat, gate=None):
        upd = super().backward(x, y, y_hat, gate)
        masked_k = upd.kernel * self.out_mask[None, None, None, :]
        return eqx.tree_at(lambda mm: mm.kernel, upd, masked_k)


# ── model builder (variable C, mirrors cm.build_model) ────────────────────────
def build_model(cfg: dict, key, name: str):
    C = channels_of(name)
    I_mask, L_mask = group_masks(name)
    keys = jax.random.split(key, 5)

    win = MaskedConv2D(
        in_channels=3, out_channels=C, kernel_size=KSIZE,
        threshold=cfg["threshold_win"], strength=1.0, key=keys[0],
        padding_mode="constant", lr=1.0, weight_decay=0.0, out_mask=I_mask,
    )
    j1 = Conv2DRecurrentDiscrete(
        channels=C, kernel_size=KSIZE, groups=1,
        j_d=cfg["j_d"], threshold=cfg["threshold_j"],
        key=keys[1], padding_mode="constant", lr=1.0, weight_decay=0.0,
        entropy_beta=cfg["entropy_beta"], lambda_entropy=1.0,
    )
    wback = ChannelWBack(10, H, W, C, cfg["strength_back"], keys[2])
    wback = eqx.tree_at(lambda m: m.W, wback, wback.W * L_mask[None, :])

    wout = PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                           strength=1.0, threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0)

    layer_map = LayerMap.from_dict({
        1: {0: win, 1: j1, 2: wback},
        2: {1: wout, 2: OutputLayer()},
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=layer_map)


# ── C-aware pooling (infers C from the array; cm.pool_j1 is hardwired to C=16) ─
def pool(h):
    """8x8 avg-pool (N,H,W,C) -> (N, (H/8)*(W/8)*C)."""
    N, Hh, Ww, Cc = h.shape
    p = POOL
    return np.asarray(h).reshape(N, Hh // p, p, Ww // p, p, Cc).mean(axis=(2, 4)).reshape(N, -1)


def pooled_cols_for_group(C: int, chans) -> np.ndarray:
    """Pooled-feature columns (in (H/8,W/8,C) C-order flatten) belonging to channels `chans`."""
    s = set(np.asarray(chans).tolist())
    nfeat = (H // POOL) * (W // POOL) * C
    return np.array([f for f in range(nfeat) if (f % C) in s])


# ── parameter accounting ──────────────────────────────────────────────────────
def nominal_trainable_params(orch) -> int:
    """W_in.kernel + (J1.kernel - C frozen diag) + W_out.W tensor elements (W_back frozen)."""
    C = int(orch.lmap[1][1].channels)
    win = int(np.asarray(orch.lmap[1][0].kernel).size)
    j1 = int(np.asarray(orch.lmap[1][1].kernel).size) - C
    wout = int(np.asarray(orch.lmap[2][1].W).size)
    return win + j1 + wout


def effective_params(orch, name: str) -> dict:
    C, nI, nL = CONFIGS[name]
    kk = KSIZE * KSIZE
    win_eff = kk * 3 * nI
    j1_eff = int(np.asarray(orch.lmap[1][1].kernel).size) - C
    wout_eff = int(np.asarray(orch.lmap[2][1].W).size)
    wback_eff = 10 * nL
    return {"win_eff": win_eff, "j1_eff": j1_eff, "wout_eff": wout_eff,
            "wback_eff_frozen": wback_eff, "trainable_eff": win_eff + j1_eff + wout_eff}


def param_report(orch, name: str) -> dict:
    C, nI, nL = CONFIGS[name]
    _, nN = (None, C - nI - nL if not is_all_both(name) else 0)
    nom = nominal_trainable_params(orch)
    eff = effective_params(orch, name)
    print(f"  [{name:21s}] C={C:2d} I={nI:2d} L={nL:2d} N={nN:2d} | "
          f"nominal_trainable={nom:5d} | effective_trainable={eff['trainable_eff']:5d} "
          f"(W_in {eff['win_eff']}/{KSIZE * KSIZE * 3 * C}) | "
          f"W_back_frozen_active={eff['wback_eff_frozen']}/{10 * C}")
    return {"name": name, "C": C, "sizes": [nI, nL, nN], "nominal_trainable": nom, **eff}
