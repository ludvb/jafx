import jax
import jax.numpy as jnp


def mish(x: jnp.ndarray):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function
    (https://arxiv.org/abs/1908.08681)"""
    return x * jnp.tanh(jax.nn.softplus(x))
