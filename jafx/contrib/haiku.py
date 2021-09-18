from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp

from .. import Namespace, param, state
from ..prng import next_prng_key


def wrap_haiku(name: str, fn: Callable[..., Any]) -> Callable[..., jnp.ndarray]:
    """Convenience wrapper for computations using haiku modules"""

    module_init, module_apply = hk.transform_with_state(fn)

    def _run_module(*args, **kwargs):
        module_param_default, module_state_default = module_init(
            next_prng_key(), *args, **kwargs
        )

        module_param_default, module_param_treedef = jax.tree_util.tree_flatten(
            module_param_default
        )
        module_param = param(name, module_param_default)
        module_param = jax.tree_util.tree_unflatten(module_param_treedef, module_param)

        with Namespace(scope=name):
            try:
                module_state = state.get("haiku_state")
            except state.StateException:
                module_state = module_state_default

            result, new_state = module_apply(
                module_param, module_state, next_prng_key(), *args, **kwargs
            )

            new_state = jax.lax.stop_gradient(new_state)
            state.set("haiku_state", new_state)

        return result

    return _run_module
