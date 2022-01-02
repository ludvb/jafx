import warnings
from contextlib import contextmanager
from functools import reduce

import jax

from . import batch_axes, state


def pnextkey():
    prng_key = currentkey()
    max_idx = reduce(lambda a, x: a * x, batch_axes().values(), 1)
    prng_keys = jax.random.split(prng_key, max_idx + 1)
    set_prng_key(prng_keys[0])
    return prng_keys[jax.lax.axis_index(list(batch_axes()))]


def nextkey():
    prng_key = currentkey()
    (prng_key,) = jax.random.split(prng_key, 1)
    set_prng_key(prng_key)
    return prng_key


def currentkey():
    try:
        with state.namespace(["global"]):
            return state.get("prng_key")
    except state.StateException:
        warnings.warn("No PRNG key has been set; seeding generator with 0")
        seed(0)
        return currentkey()


def set_prng_key(key):
    with state.namespace(["global"]):
        state.set("prng_key", key)


def seed(seed: int):
    set_prng_key(jax.random.PRNGKey(seed))


@contextmanager
def temp_prng_key(key):
    old_prng_key = currentkey()
    try:
        set_prng_key(key)
        yield
    finally:
        set_prng_key(old_prng_key)
