import jax.numpy as jnp


def group_norm(x: jnp.ndarray, group_size: int = 8, eps: float = 1e-5):
    *shape, C = x.shape
    if C % group_size != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by group size ({group_size})"
        )
    x = x.reshape(-1, C // group_size)
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(*shape, C)
    return x
