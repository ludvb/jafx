from typing import Any, Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from jax.interpreters.batching import BatchTracer

from .. import Namespace, param, state
from ..params import NoParamException
from ..prng import next_prng_key


def state_batch_reduction_mean(x: BatchTracer) -> jnp.ndarray:
    return x.val.mean(x.batch_dim)


def state_batch_reduction_sum(x: BatchTracer) -> jnp.ndarray:
    return x.val.sum(x.batch_dim)


def wrap_haiku(
    name: str,
    fn: Callable[..., Any],
    state_batch_reduction: Optional[Callable[[BatchTracer], jnp.ndarray]] = None,
) -> Callable[..., jnp.ndarray]:
    """Convenience wrapper for computations using haiku modules"""

    if state_batch_reduction is None:
        state_batch_reduction = state_batch_reduction_mean

    def _state_batch_reduction_mapper(x: Any) -> Any:
        if isinstance(x, BatchTracer):
            return state_batch_reduction(x)
        return x

    module_init, module_apply = hk.transform_with_state(fn)

    def _run_module(*args, **kwargs):
        try:
            with Namespace(scope=name):
                module_state = state.get("haiku_state")
            module_param = param(name)

        except (state.StateException, NoParamException):
            module_param_default, module_state = module_init(
                next_prng_key(), *args, **kwargs
            )
            module_param = param(name, module_param_default)

        with Namespace(scope=name):
            result, new_state = module_apply(
                module_param, module_state, next_prng_key(), *args, **kwargs
            )
            new_state = jax.lax.stop_gradient(new_state)
            new_state = jax.tree_util.tree_map(_state_batch_reduction_mapper, new_state)
            state.set("haiku_state", new_state)

        return result

    return _run_module
