import warnings
from contextlib import contextmanager

import jax

from . import state
from .namespace import Namespace


def next_prng_key():
    prng_key = current_prng_key()
    (prng_key,) = jax.random.split(prng_key, 1)
    set_prng_key(prng_key)
    return prng_key


def current_prng_key():
    try:
        with Namespace(["global"]):
            return state.get("prng_key")
    except state.StateException:
        warnings.warn("No PRNG key has been set; seeding generator with 0")
        seed(0)
        return current_prng_key()


def set_prng_key(key):
    with Namespace(["global"]):
        state.set("prng_key", key)


def seed(seed: int):
    set_prng_key(jax.random.PRNGKey(seed))


@contextmanager
def temp_prng_key(key):
    old_prng_key = current_prng_key()
    try:
        set_prng_key(key)
        yield
    finally:
        set_prng_key(old_prng_key)
