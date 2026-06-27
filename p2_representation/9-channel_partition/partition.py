"""Channel-partition architecture: confine the input (W_in) and label (W_back)
fan-in to disjoint channel groups, at EQUAL NOMINAL CAPACITY.

The entropy baseline gives every hidden channel BOTH the input message (W_in) and
the label message (W_back). Here we split the 16 hidden channels into three disjoint
groups:
  - group I: receives the input message  (W_in)   directly
  - group L: receives the label message  (W_back)  directly
  - group N: receives NEITHER directly (only the J1 recurrence)
J1 (full groups=1 recurrence) and W_out (reads all 16 pooled) are UNCHANGED, so the
groups communicate through recurrence and the readout still sees every channel.

MASK, not shrink. We keep W_in as the full 3->16 conv and W_back as the full 10->16
map, but apply a fixed channel-OUTPUT mask so connections to non-target channels are
zero and STAY zero:
  - W_in is Hebb-trained, so its UPDATES are masked too (the j_d-diagonal pattern):
    MaskedConv2D multiplies dW by the channel mask in backward(); kernel columns that
    start at 0 therefore never move (the decay term is masked as well).
  - W_back is FROZEN (backward -> zeros), so zeroing its columns once at build is
    enough.
This keeps every weight TENSOR identical in shape/count to the baseline -- a pure
rewiring. NOMINAL trainable params are therefore equal across all partitions BY
CONSTRUCTION; the EFFECTIVE (nonzero/trainable) W_in fan-in and W_back fan-in shrink
with |I| and |L|. See param_report() and the README for the fairness caveat.

This file is the shared infra for exp 9 (train_models / diagnostics / group_diagnostics
/ fullspin_importance all import it). No darnax-core edits.
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

C, KSIZE, H, W, POOL = cm.C, cm.KSIZE, cm.H, cm.W, cm.POOL


# ── partition registry: tag -> (|I|, |L|, |N|), contiguous channel ranges ─────
# baseline is the matched control: every channel receives BOTH (I=L=all 16, N=0),
# i.e. the current model (the only non-disjoint entry).
PARTITIONS: dict[str, tuple[int, int, int]] = {
    "baseline": (16, 16, 0),   # all channels get input AND label (current model)
    "L8_N0":    (8, 8, 0),     # label on half, input on the other half
    "L4_N0":    (12, 4, 0),    # small label footprint, no associative group
    "L4_N4":    (8, 4, 4),     # same |L|, plus an associative (recurrence-only) group
    "L2_N0":    (14, 2, 0),    # tiny label footprint
}
TAGS = list(PARTITIONS)


def group_indices(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel index arrays (I_idx, L_idx, N_idx). For 'baseline' I=L=all 16, N=empty
    (non-disjoint control). Otherwise I=[0:|I|], L=[|I|:|I|+|L|], N=rest -- disjoint."""
    nI, nL, nN = PARTITIONS[tag]
    if tag == "baseline":
        allc = np.arange(C)
        return allc, allc, np.array([], dtype=int)
    assert nI + nL + nN == C, f"{tag}: I+L+N={nI + nL + nN} != {C}"
    I = np.arange(0, nI)
    L = np.arange(nI, nI + nL)
    N = np.arange(nI + nL, C)
    return I, L, N


def group_masks(tag: str) -> tuple[Array, Array]:
    """(I_mask, L_mask) as float (16,) channel vectors: 1 where the group receives
    the input / label message directly."""
    I, L, _ = group_indices(tag)
    I_mask = jnp.zeros(C).at[jnp.asarray(I)].set(1.0)
    L_mask = jnp.zeros(C).at[jnp.asarray(L)].set(1.0)
    return I_mask, L_mask


# ── masked W_in: full 3->16 conv with a fixed channel-output mask ─────────────
class MaskedConv2D(Conv2D):
    """Conv2D whose output is confined to a fixed subset of channels. The kernel
    columns of non-target channels are zeroed at init and their Hebbian updates are
    masked, so they remain exactly zero throughout training (mirrors the j_d-diagonal
    update_mask of Conv2DRecurrentDiscrete)."""

    out_mask: Array  # (out_channels,) 1 = trainable/active channel, 0 = silenced

    def __init__(self, *args, out_mask, **kwargs):
        super().__init__(*args, **kwargs)
        m = jnp.asarray(out_mask, dtype=self.kernel.dtype)
        self.out_mask = m
        # silence non-target output channels from the start
        self.kernel = self.kernel * m[None, None, None, :]

    def backward(self, x, y, y_hat, gate=None):
        upd = super().backward(x, y, y_hat, gate)
        masked_k = upd.kernel * self.out_mask[None, None, None, :]
        return eqx.tree_at(lambda m: m.kernel, upd, masked_k)


# ── model builder (mirrors cm.build_model, swaps W_in / masks W_back) ─────────
def build_partitioned_model(cfg: dict, key, tag: str):
    """Build (state, orchestrator) for partition `tag`. baseline (all-ones masks)
    is structurally identical to cm.build_model except W_in is a MaskedConv2D with an
    all-ones mask (a no-op) -- so the baseline reproduces the standard model."""
    I_mask, L_mask = group_masks(tag)
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
    # confine the label injection to group L (frozen module: mask once, stays put)
    wback = eqx.tree_at(lambda m: m.W, wback, wback.W * L_mask[None, :])

    wout = PooledFlattenFC(pool=POOL, H=H, W=W, C_in=C, n_classes=10,
                           strength=1.0, threshold=5.0, key=keys[3], lr=1.0, weight_decay=0.0)

    layer_map = LayerMap.from_dict({
        1: {0: win, 1: j1, 2: wback},
        2: {1: wout, 2: OutputLayer()},
    })
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=layer_map)


# ── parameter accounting ──────────────────────────────────────────────────────
def _jd_frozen_count() -> int:
    """Number of J1 kernel entries fixed at j_d (one self-connection per channel)."""
    return C  # groups=1: central-element diagonal, one per channel


def nominal_trainable_params(orch) -> int:
    """Tensor-element count of the trainable kernels (W_in + J1-minus-diagonal + W_out).
    W_back is frozen and excluded. Identical across all partitions (masking, not shrink)."""
    win = int(np.asarray(orch.lmap[1][0].kernel).size)
    j1 = int(np.asarray(orch.lmap[1][1].kernel).size) - _jd_frozen_count()
    wout = int(np.asarray(orch.lmap[2][1].W).size)
    return win + j1 + wout


def effective_params(orch, tag: str) -> dict:
    """Nonzero/active connections actually used by each weight. W_in keeps only its
    |I| output channels; W_back keeps |L| (frozen). J1/W_out are unchanged."""
    I, L, _ = group_indices(tag)
    nI, nL = (C, C) if tag == "baseline" else (len(I), len(L))
    kk = KSIZE * KSIZE
    win_eff = kk * 3 * nI                      # W_in active kernel entries
    j1_eff = int(np.asarray(orch.lmap[1][1].kernel).size) - _jd_frozen_count()
    wout_eff = int(np.asarray(orch.lmap[2][1].W).size)
    wback_eff = 10 * nL                        # frozen, but report active fan-in
    return {"win_eff": win_eff, "j1_eff": j1_eff, "wout_eff": wout_eff,
            "wback_eff_frozen": wback_eff,
            "trainable_eff": win_eff + j1_eff + wout_eff}


def param_report(orch, tag: str, baseline_nominal: int) -> dict:
    """Print + assert nominal-param parity; return the per-partition accounting."""
    nom = nominal_trainable_params(orch)
    eff = effective_params(orch, tag)
    nI, nL, nN = PARTITIONS[tag]
    assert nom == baseline_nominal, (
        f"{tag}: nominal trainable params {nom} != baseline {baseline_nominal} "
        f"(masking must NOT change tensor shapes)")
    print(f"  [{tag:9s}] I={nI:2d} L={nL:2d} N={nN:2d} | "
          f"nominal_trainable={nom} (==baseline) | "
          f"effective_trainable={eff['trainable_eff']} "
          f"(W_in {eff['win_eff']}/{KSIZE * KSIZE * 3 * C}) | "
          f"W_back_frozen_active={eff['wback_eff_frozen']}/{10 * C}")
    return {"tag": tag, "sizes": [nI, nL, nN], "nominal_trainable": nom, **eff}
