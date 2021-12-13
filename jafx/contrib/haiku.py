from typing import Any, Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from .. import batch_axes, param, state
from ..params import NoParamException
from ..prng import next_prng_key


def wrap_haiku(
    name: str,
    fn: Callable[..., Any],
    state_batch_reduction: Optional[Callable] = None,
) -> Callable[..., jnp.ndarray]:
    """Convenience wrapper for computations using haiku modules"""

    if state_batch_reduction is None:
        state_batch_reduction = jax.lax.pmean

    module_init, module_apply = hk.transform_with_state(fn)

    def _run_module(*args, **kwargs):
        try:
            with state.scope(name):
                module_state = state.get("haiku_state")
            module_param = param(name)

        except (state.StateException, NoParamException):
            with state.scope(name):
                module_param_default, module_state = module_init(
                    next_prng_key(), *args, **kwargs
                )
            module_param = param(name, module_param_default)

        with state.scope(name):
            result, new_state = module_apply(
                module_param, module_state, next_prng_key(), *args, **kwargs
            )
            new_state = jax.lax.stop_gradient(new_state)
            for batch_axis in batch_axes():
                new_state = state_batch_reduction(new_state, batch_axis)
            state.set("haiku_state", new_state)

        return result

    return _run_module
