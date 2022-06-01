from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from .. import batch_axes, param, prng, state
from ..params import NoParamException


def wrap_flax(
    name: str,
    module: nn.Module,
    state_batch_reduction: Optional[Callable] = None,
) -> Callable[..., jnp.ndarray]:
    """Convenience wrapper for computations using flax modules"""

    if state_batch_reduction is None:
        state_batch_reduction = jax.lax.pmean

    def _run_module(*args, **kwargs):
        try:
            with state.scope(name):
                module_state = state.get("flax_state")
            module_param = param(name)

        except (state.StateException, NoParamException):
            with state.scope(name):
                module_state = module.init(prng.nextkey(), *args, **kwargs)
            module_state = module_state.unfreeze()
            module_param = param(name, module_state.pop("params"))

        with state.scope(name):
            result, new_state = module.apply(
                module_state | {"params": module_param},
                *args,
                **kwargs,
                mutable=list(module_state)
            )
            new_state = new_state.unfreeze()
            new_state = jax.lax.stop_gradient(new_state)
            new_state = state_batch_reduction(new_state, list(batch_axes()))
            state.set("flax_state", new_state)

        return result

    return _run_module
