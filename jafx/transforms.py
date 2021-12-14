# TODO: implement remaining transforms: xmap, loops, etc.
# TODO: this module is somewhat messy and we have a lot of repeating patterns,
#       consider refactoring
# TODO: needs testing

import operator as op
import warnings
from contextlib import contextmanager
from functools import partial, wraps
from typing import Hashable, Optional

import jax

from . import state
from .handler import NoHandlerError
from .intercept import Intercept


def __raise_no_handler_error(_):
    raise NoHandlerError()


_DYNAMIC_STATE_BLOCKER = Intercept(
    fn=__raise_no_handler_error,
    predicate=lambda x: isinstance(x, state.StateMessage) and not x.static,
)


def _lazy_initialization(fun, initializer=None, identifier: Optional[str] = None):
    if initializer is None:
        initializer = fun

    def _safe_hash(x):
        try:
            return hash(x)
        except TypeError:
            # TODO: how should we deal with unhashable static state?
            return 0

    def _get_identifier():
        if identifier is None:
            identifier_ = str(hash(fun))
        else:
            identifier_ = identifier

        ss = state.full(static=True)
        try:
            _ = ss.pop("function")
        except KeyError:
            pass
        state_hash = jax.tree_util.tree_reduce(
            op.xor,
            jax.tree_util.tree_map(_safe_hash, ss),
            0,
        )
        identifier_ = identifier_ + "-" + str(abs(state_hash))

        return identifier_

    try:
        return state.get("function", static=True, namespace=[_get_identifier()])

    except state.StateException:
        pass

    _ = initializer()

    @wraps(fun)
    def _fun(*args, **kwargs):
        # NOTE: This closure is used to identify the current invocation
        #       of fun with the current static state. Without it, JAX
        #       would reuse the same trace for all static states.
        return fun(*args, **kwargs)

    state.set("function", _fun, static=True, namespace=[_get_identifier()])

    return _fun


def _patch_default(transform):
    @wraps(transform)
    def _patched_transform(fn, *transform_args, identifier=None, **transform_kwargs):
        if identifier is None:
            identifier = str(hash(fn))

        def _wrapped_fun(*args, _cur_state, **kwargs):
            with _DYNAMIC_STATE_BLOCKER:
                with state.DynamicState(_cur_state) as ds:
                    result = fn(*args, **kwargs)

            return result, ds.state

        def _wrapped_transform(*args, **kwargs):
            fun = _lazy_initialization(
                transform(_wrapped_fun, *transform_args, **transform_kwargs),
                initializer=lambda: fn(*args, **kwargs),
                identifier=identifier,
            )

            result, new_state = fun(*args, _cur_state=state.full(), **kwargs)

            state.update(new_state)

            return result

        return _wrapped_transform

    return _patched_transform


@contextmanager
def _track_batch_axis(axis_name: Hashable):
    current_batch_axes = batch_axes()
    with state.namespace(["batch_axes"]):
        state.set(group="global", value=current_batch_axes + [axis_name], static=True)
    try:
        yield
    finally:
        with state.namespace(["batch_axes"]):
            state.set(group="global", value=current_batch_axes, static=True)


class _BatchAxis:
    pass


def batch_axes():
    try:
        with state.namespace(["batch_axes"]):
            return state.get("global", static=True)
    except state.StateException:
        return []


def value_and_grad(fun, *grad_args, identifier=None, **grad_kwargs):
    if identifier is None:
        identifier = str(hash(fun))

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
        fun_ = _lazy_initialization(
            jax.value_and_grad(_wrapped_fun, *grad_args, **grad_kwargs, has_aux=True),
            initializer=lambda: fun(*args, **kwargs),
            identifier=identifier,
        )

        (result, (extra, new_state)), grad = fun_(
            *args, _cur_state=state.full(), **kwargs
        )

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


def value_and_param_grad(fun, *grad_args, identifier=None, **grad_kwargs):
    if identifier is None:
        identifier = str(hash(fun))

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
        fun_ = _lazy_initialization(
            jax.value_and_grad(_wrapped_fun, *grad_args, **grad_kwargs, has_aux=True),
            initializer=lambda: fun(*args, **kwargs),
            identifier=identifier,
        )

        cur_state = state.full().copy()
        state_param = cur_state.pop("param_state")

        (result, (extra, new_state)), grad = fun_(
            state_param, cur_state, *args, **kwargs
        )

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


def pmap(fun, axis_name=None, *, in_axes=0, out_axes=0, identifier=None, **pmap_kwargs):
    if identifier is None:
        identifier = str(hash(fun))

    if axis_name is None:
        axis_name = _BatchAxis()

    def _wrapped_fun(cur_state, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(cur_state) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_pmap_fun(*args, **kwargs):
        with _track_batch_axis(axis_name=axis_name):
            fun_ = _lazy_initialization(
                jax.pmap(
                    _wrapped_fun,
                    axis_name,
                    in_axes=(None, in_axes),
                    out_axes=(out_axes, None),
                    **pmap_kwargs,
                ),
                initializer=lambda: jax.vmap(
                    fun,
                    axis_name=axis_name,
                    in_axes=in_axes,
                    out_axes=out_axes,
                )(*args, **kwargs),
                identifier=identifier,
            )

            result, new_state = fun_(state.full(), args, **kwargs)

            state.update(new_state)

            return result

    return _wrapped_pmap_fun


def soft_pmap(fun, axis_name=None, *, in_axes=0, identifier=None, **soft_pmap_kwargs):
    warnings.warn("jafx.soft_pmap is buggy and not advisable to use")

    if identifier is None:
        identifier = str(hash(fun))

    if axis_name is None:
        axis_name = _BatchAxis()

    def _wrapped_fun(cur_state, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(cur_state) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_soft_pmap_fun(*args, **kwargs):
        with _track_batch_axis(axis_name=axis_name):
            fun_ = _lazy_initialization(*args, **kwargs)(
                jax.soft_pmap(
                    _wrapped_fun,
                    axis_name,
                    in_axes=(None, in_axes),
                    **soft_pmap_kwargs,
                ),
                initializer=lambda: jax.vmap(fun, axis_name=axis_name, in_axes=in_axes)(
                    *args, **kwargs
                ),
                identifier=identifier,
            )

            result, new_state = fun_(state.full(), args, **kwargs)

            # XXX: jax.soft_pmap (and jax.experimental.maps.xmap, which soft_pmap is
            #      based on) does not allow `None` out_axes. As a workaround, we
            #      manually gather state here. This is not a good solution: it does
            #      not check if arrays have been reduced and introduces unnecessary
            #      overhead.
            new_state = jax.tree_util.tree_map(lambda x: x[0], new_state)

            state.update(new_state)

            return result

    return _wrapped_soft_pmap_fun


def vmap(fun, axis_name=None, *, in_axes=0, out_axes=0, identifier=None):
    if identifier is None:
        identifier = str(hash(fun))

    if axis_name is None:
        axis_name = _BatchAxis()

    def _wrapped_fun(state_dynamic, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(state_dynamic) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_vmap_fun(*args, **kwargs):
        with _track_batch_axis(axis_name=axis_name):
            fun_ = _lazy_initialization(
                jax.vmap(
                    _wrapped_fun,
                    in_axes=(None, in_axes),
                    out_axes=(out_axes, None),
                    axis_name=axis_name,
                ),
                initializer=lambda: jax.vmap(
                    fun, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes
                )(*args, **kwargs),
                identifier=identifier,
            )

            result, new_state = fun_(state.full(), args, **kwargs)

            state.update(new_state)

            return result

    return _wrapped_vmap_fun


def scan(fun, init, xs, *scan_args, identifier=None, **scan_kwargs):
    if identifier is None:
        identifier = str(hash(fun))

    def _wrapped_fun(state, x):
        jafx_state, fn_state = state
        with _DYNAMIC_STATE_BLOCKER, state.DynamicState(jafx_state) as ds:
            fn_state, y = fun(state, x)
        return (ds.state, fn_state), y

    fun = _lazy_initialization(
        _wrapped_fun,
        initializer=lambda: fun(init, xs[0]),
        identifier=identifier,
    )

    (jafx_state, fn_state), ys = jax.lax.scan(
        _wrapped_fun,
        (state.full(), init),
        xs,
        *scan_args,
        **scan_kwargs,
    )
    state.update(jafx_state)

    return fn_state, ys


def cond(pred, true_fun, false_fun, *operands):
    def _wrap_fun(fun):
        def _wrapped_fun(v):
            _cur_state, *operands = v
            with _DYNAMIC_STATE_BLOCKER:
                with state.DynamicState(_cur_state) as ds:
                    result = fun(*operands)
            return result, ds.state

        return _wrapped_fun

    true_fun_ = _wrap_fun(true_fun)
    false_fun_ = _wrap_fun(false_fun)

    true_fun_ = _lazy_initialization(
        true_fun_,
        initializer=lambda: true_fun(*operands),
        identifier=str(hash(true_fun)),
    )
    false_fun_ = _lazy_initialization(
        false_fun_,
        initializer=lambda: false_fun(*operands),
        identifier=str(hash(true_fun)),
    )

    result, new_state = jax.lax.cond(
        pred,
        true_fun_,
        false_fun_,
        (state.full(), *operands),
    )
    state.update(new_state)

    return result


checkpoint = _patch_default(jax.checkpoint)


jit = _patch_default(jax.jit)
