# TODO: implement remaining transforms: xmap, loops, etc.
# TODO: this module is somewhat messy and we have a lot of repeating patterns,
#       consider refactoring
# TODO: needs testing

import warnings
from functools import partial, wraps
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
    # TODO: condition initialization on static state
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


def _patch_defaut(transform):
    @wraps(transform)
    def _patched_transform(fn, *transform_args, **transform_kwargs):
        def _wrapped_fun(*args, _cur_state, **kwargs):
            with _DYNAMIC_STATE_BLOCKER:
                with state.DynamicState(_cur_state) as ds:
                    result = fn(*args, **kwargs)

            return result, ds.state

        def _wrapped_transform(*args, **kwargs):
            _ = _with_lazy_initialization(*args, **kwargs)(fn)

            result, new_state = transform(
                _wrapped_fun, *transform_args, **transform_kwargs
            )(*args, _cur_state=state.full(), **kwargs)

            state.update(new_state)

            return result

        return _wrapped_transform

    return _patched_transform


def value_and_grad(fun, *grad_args, **grad_kwargs):
    try:
        has_aux = grad_kwargs.pop("has_aux")
    except KeyError:
        has_aux = False

    def _wrapped_fun(*args, _cur_state, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(_cur_state) as ds:
                result = fun(*args, **kwargs)

        if has_aux:
            result, extra = result
        else:
            extra = None

        return result, (extra, ds.state)

    def _wrapped_value_and_grad_fun(*args, **kwargs):
        _ = _with_lazy_initialization(*args, **kwargs)(fun)

        cur_state = state.full()

        (result, (extra, new_state)), grad = jax.value_and_grad(
            _wrapped_fun, *grad_args, **grad_kwargs, has_aux=True
        )(*args, _cur_state=cur_state, **kwargs)

        state.update(new_state)

        if has_aux:
            result = (result, extra)

        return result, grad

    return _wrapped_value_and_grad_fun


def grad(fun, *grad_args, **grad_kwargs):
    has_aux = "has_aux" in grad_kwargs and grad_kwargs["has_aux"]
    _value_and_grad = value_and_grad(fun, *grad_args, **grad_kwargs)

    def _wrapped_grad_fun(*args, **kwargs):
        result = _value_and_grad(*args, **kwargs)

        if has_aux:
            (_, extra), grad = result
            return grad, extra

        _, grad = result
        return grad

    return _wrapped_grad_fun


def value_and_param_grad(fun, *grad_args, **grad_kwargs):
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

    def _wrapped_value_and_grad_fun(*args, **kwargs):
        _ = _with_lazy_initialization(*args, **kwargs)(fun)

        cur_state = state.full().copy()
        state_param = cur_state.pop("param_state")

        (result, (extra, new_state)), grad = jax.value_and_grad(
            _wrapped_fun, *grad_args, **grad_kwargs, has_aux=True
        )(state_param, cur_state, *args, **kwargs)

        state.update(new_state)

        if has_aux:
            result = (result, extra)

        return result, grad

    return _wrapped_value_and_grad_fun


def param_grad(fun, *grad_args, **grad_kwargs):
    has_aux = "has_aux" in grad_kwargs and grad_kwargs["has_aux"]
    _value_and_grad = value_and_param_grad(fun, *grad_args, **grad_kwargs)

    def _wrapped_param_grad_fun(*args, **kwargs):
        result = _value_and_grad(*args, **kwargs)

        if has_aux:
            (_, extra), grad = result
            return grad, extra

        _, grad = result
        return grad

    return _wrapped_param_grad_fun


def vjp(fn, *vjp_args, **vjp_kwargs):
    try:
        has_aux = vjp_kwargs.pop("has_aux")
    except KeyError:
        has_aux = False

    def _wrapped_fun(cur_state, *args):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(cur_state) as ds:
                result = fn(*args)

        if has_aux:
            result, extra = result
        else:
            extra = None

        return result, (extra, ds.state)

    result, f_vjp, (extra, new_state) = jax.vjp(
        partial(_wrapped_fun, state.full()), *vjp_args, **vjp_kwargs, has_aux=True
    )

    state.update(new_state)

    if has_aux:
        result = (result, extra)

    return result, f_vjp


def param_vjp(fn, **vjp_kwargs):
    try:
        has_aux = vjp_kwargs.pop("has_aux")
    except KeyError:
        has_aux = False

    def _wrapped_fun(cur_state, cur_param_state):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(cur_state) as ds:
                with state.DynamicState({"param_state": cur_param_state}) as ds_param:
                    result = fn()

        if has_aux:
            result, extra = result
        else:
            extra = None

        return result, (extra, {**ds.state, **ds_param.state})

    cur_state = state.full().copy()
    state_param = cur_state.pop("param_state")

    result, f_vjp, (extra, new_state) = jax.vjp(
        partial(_wrapped_fun, cur_state), state_param, **vjp_kwargs, has_aux=True
    )

    state.update(new_state)

    if has_aux:
        result = (result, extra)

    return result, f_vjp


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


def soft_pmap(fun, axis_name=None, *, in_axes=0, **soft_pmap_kwargs):
    warnings.warn("jafx.soft_pmap is buggy and not advisable to use")

    def _wrapped_fun(cur_state, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(cur_state) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_soft_pmap_fun(*args, **kwargs):
        _ = _with_lazy_initialization(*args, **kwargs)(
            lambda *args, **kwargs: jax.vmap(fun, axis_name=axis_name, in_axes=in_axes)(
                *args, **kwargs
            ),
            identifier=str(hash(fun)),
        )

        cur_state = state.full()

        result, new_state = jax.soft_pmap(
            _wrapped_fun,
            axis_name,
            in_axes=(None, in_axes),
            **soft_pmap_kwargs,
        )(cur_state, args, **kwargs)

        # XXX: jax.soft_pmap (and jax.experimental.maps.xmap, which soft_pmap is
        #      based on) does not allow `None` out_axes. As a workaround, we
        #      manually gather state here. This is not a good solution: it does
        #      not check if arrays have been reduced and introduces unnecessary
        #      overhead.
        new_state = jax.tree_map(lambda x: x[0], new_state)

        state.update(new_state)

        return result

    return _wrapped_soft_pmap_fun


def vmap(fun, in_axes=0, out_axes=0, axis_name=None):
    def _wrapped_fun(state_dynamic, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(state_dynamic) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_vmap_fun(*args, **kwargs):
        _ = _with_lazy_initialization(*args, **kwargs)(
            lambda *args, **kwargs: jax.vmap(
                fun, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes
            )(*args, **kwargs),
            identifier=str(hash(fun)),
        )

        cur_state = state.full()

        result, new_state = jax.vmap(
            _wrapped_fun,
            in_axes=(None, in_axes),
            out_axes=(out_axes, None),
            axis_name=axis_name,
        )(cur_state, args, **kwargs)

        state.update(new_state)

        return result

    return _wrapped_vmap_fun


checkpoint = _patch_defaut(jax.checkpoint)


jit = _patch_defaut(jax.jit)
