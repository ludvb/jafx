from typing import Callable, Optional

import jax
import jax.numpy as jnp

from .. import param, prng


def conv(
    x: jnp.ndarray,
    dim_out: int,
    kernel_size: int,
    window_strides: Optional[tuple[int]] = None,
    padding: str = "SAME",
    lhs_dilation: Optional[tuple[int]] = None,
    rhs_dilation: Optional[tuple[int]] = None,
    bias: bool = True,
    weight_init: Optional[jnp.ndarray] = None,
    bias_init: Optional[jnp.ndarray] = None,
):
    if window_strides is None:
        window_strides = (1, 1)

    if lhs_dilation is None:
        lhs_dilation = (1, 1)

    if rhs_dilation is None:
        rhs_dilation = (1, 1)

    if weight_init is None:
        weight_init = jax.random.normal(
            prng.nextkey(), shape=(dim_out, x.shape[-1], kernel_size, kernel_size)
        )
        weight_init = weight_init / jnp.sqrt(x.shape[-1] * kernel_size**2)

    if bias_init is None:
        bias_init = jnp.zeros((dim_out,))

    W = param("W", weight_init)
    y = jax.lax.conv_general_dilated(
        x[None],
        W,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )[0]
    if bias:
        b = param("b", bias_init)
        y = y + b
    return y


def linear(
    x: jnp.ndarray,
    dim_out: int,
    bias: bool = True,
    weight_init: Optional[jnp.ndarray] = None,
    bias_init: Optional[jnp.ndarray] = None,
):
    if weight_init is None:
        weight_init = jax.random.normal(prng.nextkey(), shape=(x.shape[-1], dim_out))
        weight_init = weight_init / jnp.sqrt(x.shape[-1])

    if bias_init is None:
        bias_init = jnp.zeros((dim_out,))

    W = param("W", weight_init)
    y = x @ W
    if bias:
        b = param("b", bias_init)
        y = y + b
    return y


def sequential(*layers):
    def forward(x):
        for layer in layers:
            x = layer(x)
        return x

    return forward
