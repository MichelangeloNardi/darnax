"""Flat-vector convolution adapters wrapping professor's conv utils.

These adapters accept and return flattened tensors ``(B, H*W*C)`` and
internally reshape to NHWC for convolution operations. They expose a
``kernel`` parameter and implement ``backward`` returning a PyTree with a
``kernel`` update so they integrate with the existing training pipeline.
"""
from __future__ import annotations

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from darnax.modules.interfaces import Adapter, Layer
import operator
from jax.tree_util import tree_reduce
from darnax.modules.conv.utils import conv_backward_with_threshold

KeyArray = Array


class ConvAdapter(Adapter):
    """Feedforward conv adapter that accepts flattened inputs.

    Forward: (B, H_in*W_in*C_in) -> reshape(B,H_in,W_in,C_in) -> conv -> flatten
    Backward: uses conv_backward_with_threshold to compute kernel gradient and
    returns an update PyTree with the same kernel shape.
    """

    kernel: Array
    strength: Array
    threshold: Array
    lr: Array
    weight_decay: Array

    h_in: int = eqx.field(static=True)
    w_in: int = eqx.field(static=True)
    c_in: int = eqx.field(static=True)
    h_out: int = eqx.field(static=True)
    w_out: int = eqx.field(static=True)
    c_out: int = eqx.field(static=True)
    kernel_size: tuple[int, int] = eqx.field(static=True)
    padding_mode: str | None = eqx.field(static=True)

    def __init__(
        self,
        h_in: int,
        w_in: int,
        c_in: int,
        c_out: int,
        kernel_size: int | tuple[int, int],
        h_out: int | None = None,
        w_out: int | None = None,
        padding_mode: str | None = "constant",
        strength: float = 1.0,
        threshold: float = 0.0,
        key: KeyArray | None = None,
        dtype: DTypeLike = jnp.float32,
        *,
        lr: float = 1.0,
        weight_decay: float = 0.0,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        self.h_in, self.w_in, self.c_in = int(h_in), int(w_in), int(c_in)
        self.c_out = int(c_out)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        self.padding_mode = padding_mode

        if h_out is None:
            h_out = self.h_in
        if w_out is None:
            w_out = self.w_in
        self.h_out, self.w_out = int(h_out), int(w_out)

        kh, kw = self.kernel_size
        fan_in = kh * kw * self.c_in
        key, init_key = jax.random.split(key)
        self.kernel = jax.random.normal(init_key, shape=(kh, kw, self.c_in, self.c_out), dtype=dtype) / jnp.sqrt(
            jnp.asarray(fan_in, dtype=dtype)
        )

        self.strength = jnp.asarray(strength, dtype=dtype)
        self.threshold = jnp.asarray(threshold, dtype=dtype)
        self.lr = jnp.asarray(lr, dtype=dtype)
        self.weight_decay = jnp.asarray(weight_decay, dtype=dtype)

    @property
    def has_state(self) -> bool:
        return False

    def __call__(self, x_flat: Array, rng: KeyArray | None = None) -> Array:
        batch = x_flat.shape[0]
        x_nhwc = x_flat.reshape(batch, self.h_in, self.w_in, self.c_in)

        kh, kw = self.kernel_size
        pad_h, pad_w = kh // 2, kw // 2
        if self.padding_mode is not None:
            x_nhwc = jnp.pad(x_nhwc, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=self.padding_mode)

        y_nhwc = jax.lax.conv_general_dilated(
            lhs=x_nhwc,
            rhs=self.kernel,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        y_nhwc = y_nhwc * self.strength
        y_flat = y_nhwc.reshape(batch, self.h_out * self.w_out * self.c_out)
        return y_flat

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        # Accept named args (x, y, y_hat, gate) to match orchestrator calls.
        x_flat = x
        y_flat = y
        y_hat_flat = y_hat
        batch = x_flat.shape[0]
        x_nhwc = x_flat.reshape(batch, self.h_in, self.w_in, self.c_in)
        y_nhwc = y_flat.reshape(batch, self.h_out, self.w_out, self.c_out)
        y_hat_nhwc = y_hat_flat.reshape(batch, self.h_out, self.w_out, self.c_out)

        kh, kw = self.kernel_size
        dw = conv_backward_with_threshold(
            x=x_nhwc,
            y=y_nhwc,
            y_hat=y_hat_nhwc,
            threshold=self.threshold,
            kernel_shape=(kh, kw),
            groups=1,
            strides=(1, 1),
            padding_mode=self.padding_mode,
        )

        n, ho, wo, _ = y_nhwc.shape
        decay_scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=dw.dtype))
        dw = self.lr * dw + self.weight_decay * decay_scale * self.kernel

        zero_update: Self = jax.tree_util.tree_map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        update: Self = eqx.tree_at(lambda m: m.kernel, zero_update, dw)
        return update


class ConvRecurrentDiscrete(Layer):
    """Depthwise / grouped recurrent conv adapter on flat vectors.

    It creates a kernel with shape (Kh, Kw, 1, channels) and uses
    feature_group_count=channels for depthwise behaviour.
    """

    kernel: Array
    threshold: Array
    j_d: Array
    lr: Array
    weight_decay: Array
    update_mask: Array

    h: int = eqx.field(static=True)
    w: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    kernel_size: tuple[int, int] = eqx.field(static=True)
    padding_mode: str | None = eqx.field(static=True)
    central_element: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        h: int,
        w: int,
        channels: int,
        kernel_size: int | tuple[int, int],
        j_d: float,
        threshold: float = 0.0,
        padding_mode: str | None = "constant",
        key: KeyArray | None = None,
        dtype: DTypeLike = jnp.float32,
        *,
        lr: float = 1.0,
        weight_decay: float = 0.0,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        self.h, self.w, self.channels = int(h), int(w), int(channels)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        kh, kw = self.kernel_size
        if (kh % 2) == 0 or (kw % 2) == 0:
            raise ValueError("kernel_size must be odd for same padding")

        self.central_element = (kh // 2, kw // 2)
        self.padding_mode = padding_mode

        fan_in = kh * kw * 1
        key, ik = jax.random.split(key)
        self.kernel = jax.random.normal(ik, shape=(kh, kw, 1, self.channels), dtype=dtype) / jnp.sqrt(
            jnp.asarray(fan_in, dtype=dtype)
        )

        self.threshold = jnp.asarray(threshold, dtype=dtype)
        self.j_d = jnp.asarray(j_d, dtype=dtype)
        self.lr = jnp.asarray(lr, dtype=dtype)
        self.weight_decay = jnp.asarray(weight_decay, dtype=dtype)

        # build mask: 1 everywhere except central diagonal
        kh, kw = self.kernel_size
        ch, cw = self.central_element
        mask = jnp.ones((kh, kw, 1, self.channels), dtype=self.kernel.dtype)
        mask = mask.at[ch, cw, 0, :].set(0.0)
        self.update_mask = mask

        # set diagonal center to j_d
        self.kernel = self.kernel.at[ch, cw, 0, :].set(self.j_d)


    # inherits has_state=True from Layer

    def reduce(self, h):
        """Aggregate incoming messages by summation (works for PyTree of arrays)."""
        return jnp.asarray(tree_reduce(operator.add, h))

    def activation(self, x: Array) -> Array:
        """Discrete activation: sign (±1)."""
        return jnp.sign(x)

    def __call__(self, x_flat: Array, rng: KeyArray | None = None) -> Array:
        batch = x_flat.shape[0]
        x_nhwc = x_flat.reshape(batch, self.h, self.w, self.channels)

        kh, kw = self.kernel_size
        pad_h, pad_w = kh // 2, kw // 2
        if self.padding_mode is not None:
            x_nhwc = jnp.pad(x_nhwc, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=self.padding_mode)

        # Compute depthwise (per-channel) convolution explicitly to avoid grouped-conv
        # dimension-mismatch issues across different JAX versions / kernels.
        outputs = []
        for c in range(self.channels):
            lhs = x_nhwc[..., c : c + 1]  # (B, H, W, 1)
            rhs = self.kernel[:, :, 0:1, c : c + 1]  # (Kh, Kw, 1, 1)
            y_ch = jax.lax.conv_general_dilated(
                lhs=lhs,
                rhs=rhs,
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            outputs.append(y_ch)

        y_nhwc = jnp.concatenate(outputs, axis=-1)
        y_flat = y_nhwc.reshape(batch, self.h * self.w * self.channels)
        return y_flat

    def backward(self, x: Array, y: Array, y_hat: Array, gate: Array | None = None) -> Self:
        # Accept named args (x, y, y_hat, gate) to match orchestrator calls.
        x_flat = x
        y_flat = y
        y_hat_flat = y_hat
        batch = x_flat.shape[0]
        x_nhwc = x_flat.reshape(batch, self.h, self.w, self.channels)
        y_nhwc = y_flat.reshape(batch, self.h, self.w, self.channels)
        y_hat_nhwc = y_hat_flat.reshape(batch, self.h, self.w, self.channels)

        kh, kw = self.kernel_size

        # Compute per-channel kernel gradients and stack into (Kh, Kw, 1, channels)
        grads = []
        for c in range(self.channels):
            x_ch = x_nhwc[..., c : c + 1]
            y_ch = y_nhwc[..., c : c + 1]
            yhat_ch = y_hat_nhwc[..., c : c + 1]
            dw_ch = conv_backward_with_threshold(
                x=x_ch,
                y=y_ch,
                y_hat=yhat_ch,
                threshold=self.threshold,
                kernel_shape=(kh, kw),
                groups=1,
                strides=(1, 1),
                padding_mode=self.padding_mode,
            )
            # dw_ch shape (Kh, Kw, 1, 1)
            grads.append(dw_ch)

        dw = jnp.concatenate(grads, axis=-1)  # (Kh, Kw, 1, channels)

        n, ho, wo, _ = y_nhwc.shape
        decay_scale = 1.0 / jnp.sqrt(jnp.asarray(n * ho * wo, dtype=dw.dtype))
        dw = self.lr * dw + self.weight_decay * decay_scale * self.kernel

        dw = dw * self.update_mask

        zero_update: Self = jax.tree_util.tree_map(jnp.zeros_like, self, is_leaf=eqx.is_inexact_array)
        update: Self = eqx.tree_at(lambda m: m.kernel, zero_update, dw)
        return update
