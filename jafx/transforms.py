# TODO: implement remaining transforms: xmap, loops, etc.

from typing import Optional

import jax

from . import state
from .handler import NoHandlerError
from .intercept import Intercept
from .namespace import Namespace


def __raise_no_handler_error(_):
    raise NoHandlerError()


_DYNAMIC_STATE_BLOCKER = Intercept(
    fn=__raise_no_handler_error,
    predicate=lambda x: isinstance(x, state.StateMessage) and not x.static,
)


def _with_lazy_initialization(*args, **kwargs):
    def _decorator(fun, identifier: Optional[str] = None):
        if identifier is None:
            identifier = str(hash(fun))
        try:
            with Namespace(["global", "functions", identifier]):
                _ = state.get("is_initialized", static=True)
        except state.StateException:
            _ = fun(*args, **kwargs)
            with Namespace(["global", "functions", identifier]):
                state.set("is_initialized", True, static=True)
        return fun

    return _decorator


def grad(fun, *grad_args, **grad_kwargs):
    try:
        has_aux = grad_kwargs.pop("has_aux")
    except KeyError:
        has_aux = False

    def _wrapped_fun(state_param, state_nonparam, *args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(state_nonparam) as ds_nonparam:
                with state.DynamicState({"param_state": state_param}) as ds_param:
                    result = fun(*args, **kwargs)

        if has_aux:
            result, extra = result
        else:
            extra = None

        return result, (extra, {**ds_nonparam.state, **ds_param.state})

    def _wrapped_grad_fun(*args, **kwargs):
        _ = _with_lazy_initialization(*args, **kwargs)(fun)

        cur_state = state.full().copy()
        state_param = cur_state.pop("param_state")

        result, (extra, new_state) = jax.grad(
            _wrapped_fun, *grad_args, **grad_kwargs, has_aux=True
        )(state_param, cur_state, *args, **kwargs)

        state.update(new_state)

        if has_aux:
            return result, extra

        return result

    return _wrapped_grad_fun


def pmap(fun, axis_name=None, *, in_axes=0, out_axes=0, **pmap_kwargs):
    def _wrapped_fun(cur_state, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(cur_state) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_pmap_fun(*args, **kwargs):
        _ = _with_lazy_initialization(*args, **kwargs)(
            lambda *args, **kwargs: jax.vmap(
                fun, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes
            )(*args, **kwargs),
            identifier=str(hash(fun)),
        )

        cur_state = state.full()

        result, new_state = jax.pmap(
            _wrapped_fun,
            axis_name,
            in_axes=(None, in_axes),
            out_axes=(out_axes, None),
            **pmap_kwargs,
        )(cur_state, args, **kwargs)

        state.update(new_state)

        return result

    return _wrapped_pmap_fun


vmap = jax.vmap


def jit(fun, **jit_kwargs):
    def _wrapped_fun(state_dynamic, *args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(state_dynamic) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_jit_fun(*args, **kwargs):
        _ = _with_lazy_initialization(*args, **kwargs)(fun)

        result, new_state = jax.jit(_wrapped_fun, **jit_kwargs)(
            state.full(), *args, **kwargs
        )

        state.update(new_state)

        return result

    return _wrapped_jit_fun
