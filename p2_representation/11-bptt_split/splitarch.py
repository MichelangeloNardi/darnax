"""Shared infra for exp 11 (BPTT split diagnostic + partial-overlap split).

Reuses exp-10's arch (MaskedConv2D, build_model, C-aware pool, param accounting) and
the BPTT machinery in ../bptt_common.py. Adds:
  - build_explicit(): build a model from EXPLICIT W_in (group I) and W_back (group L)
    channel index sets, allowing OVERLAP (needed for the partial-overlap config).
  - the partial-overlap C=24 config (both / input_only / label_only, 8 channels each).
  - make_hard_reps_caware(): C-aware pooled hard-sign reps for the BPTT checkpoint proxy
    (bc.make_hard_reps pools via cm.pool_j1, which is hardwired to C=16).
"""
from __future__ import annotations

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "p1_readout_gap"))
sys.path.insert(0, str(REPO / "p2_representation"))
sys.path.insert(0, str(REPO / "p2_representation" / "10-split_scale"))

import common as cm  # noqa: E402
import arch as A  # exp-10 arch  # noqa: E402
from darnax.layer_maps.sparse import LayerMap  # noqa: E402
from darnax.modules.conv.conv import Conv2DRecurrentDiscrete  # noqa: E402
from darnax.modules.conv.spatial_fc import ChannelWBack, PooledFlattenFC  # noqa: E402
from darnax.modules.input_output import OutputLayer  # noqa: E402
from darnax.orchestrators.sequential import SequentialOrchestrator  # noqa: E402
from darnax.states.sequential import SequentialState  # noqa: E402

KSIZE, H, W, POOL = cm.KSIZE, cm.H, cm.W, cm.POOL

# part-1 BPTT split configs (reuse exp-10 split architectures)
BPTT_CONFIGS = ["split_C24_I16_L8_N0", "split_C32_I16_L8_N8"]

# part-2 partial-overlap config: C=24, three 8-channel groups; W_in -> both+input_only,
# W_back -> both+label_only (overlap on `both`). Preserves input capacity |I_win|=16.
PARTIAL = {
    "C": 24,
    "groups": {  # channel index ranges
        "both": list(range(0, 8)),
        "input_only": list(range(8, 16)),
        "label_only": list(range(16, 24)),
    },
}
PARTIAL["I_idx"] = PARTIAL["groups"]["both"] + PARTIAL["groups"]["input_only"]   # W_in
PARTIAL["L_idx"] = PARTIAL["groups"]["both"] + PARTIAL["groups"]["label_only"]   # W_back


def build_explicit(cfg, key, C: int, I_idx, L_idx):
    """Build (state, orch) with W_in confined to channels I_idx and W_back to L_idx
    (which may overlap). Mirrors arch.build_model; reuses arch.MaskedConv2D."""
    I_mask = jnp.zeros(C).at[jnp.asarray(I_idx)].set(1.0)
    L_mask = jnp.zeros(C).at[jnp.asarray(L_idx)].set(1.0)
    keys = jax.random.split(key, 5)

    win = A.MaskedConv2D(
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
    lmap = LayerMap.from_dict({1: {0: win, 1: j1, 2: wback}, 2: {1: wout, 2: OutputLayer()}})
    return SequentialState([(H, W, 3), (H, W, C), 10]), SequentialOrchestrator(layers=lmap)


def build_partial(cfg, key):
    return build_explicit(cfg, key, PARTIAL["C"], PARTIAL["I_idx"], PARTIAL["L_idx"])


def make_hard_reps_caware(win, j1, n_steps):
    """C-aware pooled hard-sign D reps for the BPTT checkpoint proxy (arch.pool, not
    cm.pool_j1). Returns (N, (H/8)*(W/8)*C)."""
    @eqx.filter_jit
    def reps(params, x):
        win_ = eqx.tree_at(lambda m: m.kernel, win, params["win"])
        j1_ = eqx.tree_at(lambda m: m.kernel, j1, params["j1"])
        win_msg = win_(x)
        h = jnp.zeros_like(win_msg)
        for _ in range(n_steps):
            hard = jnp.sign(win_msg + j1_(h))
            h = jnp.where(hard == 0, 1.0, hard)
        N, Hh, Ww, Cc = h.shape
        return h.reshape(N, Hh // POOL, POOL, Ww // POOL, POOL, Cc).mean(axis=(2, 4)).reshape(N, -1)
    return reps


def win_mask_for(name: str):
    """(1,1,1,C) W_in channel-output mask for a split config (group I only)."""
    I_mask, _ = A.group_masks(name)
    return I_mask[None, None, None, :]
