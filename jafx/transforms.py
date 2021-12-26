# TODO: implement remaining transforms: xmap, loops, etc.
# TODO: this module is very messy. when
#       https://github.com/google/jax/issues/7155 is fixed, we can probably
#       simplify substantially

import operator as op
from contextlib import contextmanager
from functools import partial, wraps
from typing import Hashable, Optional

import jax
import jax.numpy as jnp

from . import state
from .handler import NoHandlerError
from .intercept import Intercept


def __raise_no_handler_error(_):
    raise NoHandlerError()


_DYNAMIC_STATE_BLOCKER = Intercept(
    fn=__raise_no_handler_error,
    predicate=lambda x: isinstance(x, state.StateMessage) and not x.static,
)


def _hash_state(state):
    def _safe_hash(x):
        try:
            return hash(x)
        except TypeError:
            # TODO: how should we deal with unhashable state?
            return 0

    return jax.tree_util.tree_reduce(
        op.xor,
        jax.tree_util.tree_map(_safe_hash, state),
        0,
    )


def _lazy_initialization(fun, initializer=None, identifier: Optional[str] = None):
    if initializer is None:
        initializer = fun

    def _get_identifier():
        if identifier is None:
            identifier_ = str(hash(fun))
        else:
            identifier_ = identifier

        ss = state.full(static=True)
        try:
            _ = ss.pop("isinit")
        except KeyError:
            pass

        return identifier_ + "-" + str(_hash_state(ss))

    try:
        noinit = state.get("noinit", static=True, namespace=["value"])
        if noinit:
            return fun
    except state.StateException:
        pass

    try:
        isinit = state.get("isinit", static=True, namespace=[_get_identifier()])
        if isinit:

            @wraps(fun)
            def _fun(*args, **kwargs):
                with state.temp("noinit", True, static=True, namespace=["value"]):
                    return fun(*args, **kwargs)

            return _fun

    except state.StateException:
        pass

    def _initializer(*args, **kwargs):
        initializer_result = initializer(*args, **kwargs)
        state.set("isinit", True, static=True, namespace=[_get_identifier()])
        return initializer_result

    return _initializer


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
                initializer=lambda *args, _cur_state=None, **kwargs: (
                    fn(*args, **kwargs),
                    state.full(),
                ),
                identifier=identifier,
            )

            result, new_state = fun(*args, _cur_state=state.full(), **kwargs)

            state.update(new_state, add_missing=True)

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
    def __init__(self, identifier):
        self.identifier = identifier

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        return self.identifier


def batch_axes():
    try:
        with state.namespace(["batch_axes"]):
            return state.get("global", static=True)
    except state.StateException:
        return []


def value_and_grad(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    allow_int=False,
    reduce_axes=(),
    identifier=None,
):
    if identifier is None:
        identifier = str(hash(fun))

    def _wrapped_fun(*args, _cur_state, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(_cur_state) as ds:
                result = fun(*args, **kwargs)

        if has_aux:
            result, extra = result
        else:
            extra = None

        return result, (extra, ds.state)

    def _run_initializer(*args, _cur_state, **kwargs):
        result = fun(*args, **kwargs)

        if has_aux:
            result, extra = result
        else:
            extra = None

        s = state.full()
        return (
            (result, (extra, s)),
            jnp.zeros_like(args[argnums])
            if isinstance(argnums, int)
            else [jnp.zeros_like(args[i]) for i in argnums],
        )

    def _wrapped_value_and_grad_fun(*args, **kwargs):
        fun_ = _lazy_initialization(
            jax.value_and_grad(
                _wrapped_fun,
                argnums=argnums,
                has_aux=True,
                holomorphic=holomorphic,
                allow_int=allow_int,
                reduce_axes=reduce_axes,
            ),
            initializer=_run_initializer,
            identifier=identifier,
        )

        (result, (extra, new_state)), grad = fun_(
            *args, _cur_state=state.full(), **kwargs
        )

        state.update(new_state, add_missing=True)

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

    def _run_transform(*args, **kwargs):
        cur_state = state.full().copy()
        state_param = cur_state.pop("param_state")
        return jax.value_and_grad(
            _wrapped_fun,
            *grad_args,
            **grad_kwargs,
            has_aux=True,
        )(state_param, cur_state, *args, **kwargs)

    def _run_initializer(*args, **kwargs):
        result = fun(*args, **kwargs)

        if has_aux:
            result, extra = result
        else:
            extra = None

        s = state.full()
        return (
            (result, (extra, s)),
            jax.tree_util.tree_map(jnp.zeros_like, s["param_state"]),
        )

    def _wrapped_value_and_grad_fun(*args, **kwargs):
        fun_ = _lazy_initialization(
            _run_transform,
            initializer=_run_initializer,
            identifier=identifier,
        )

        (result, (extra, new_state)), grad = fun_(*args, **kwargs)

        state.update(new_state, add_missing=True)

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

    state.update(new_state, add_missing=True)

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

    state.update(new_state, add_missing=True)

    if has_aux:
        result = (result, extra)

    return result, f_vjp


def _make_pmap_initializer(fun, axis_name, in_axes, out_axes, identifier):
    def _initializer(_, args, **kwargs):
        def _fun(state_dynamic, args, **kwargs):
            with _DYNAMIC_STATE_BLOCKER:
                with state.DynamicState(state_dynamic) as ds:
                    result = fun(*args, **kwargs)

            mapped_state_axes = jax.tree_util.tree_map(
                lambda x: 0 if hasattr(x, "batch_dim") else None, ds.state
            )
            state.set(
                "mapped_state_axes",
                mapped_state_axes,
                static=True,
                namespace=[identifier + f"-{_hash_state(batch_axes())}"],
            )
            # TODO: ^ reduce mapped_state_axes to minimal prefix tree

            # new_state = jax.tree_util.tree_multimap(
            #     lambda x, axis: jax.lax.all_gather(x, axis_name=axis_name, axis=axis)
            #     if axis is not None
            #     else x,
            #     new_state,
            #     mapped_state_axes,
            # )
            # TODO: ^ look into why this doesn't work
            new_state = jax.tree_util.tree_map(
                lambda x: jax.lax.all_gather(x, axis_name=axis_name, axis=0)
                if hasattr(x, "batch_dim")
                else x,
                ds.state,
            )

            return result, new_state

        return jax.vmap(
            _fun,
            in_axes=(None, in_axes),
            out_axes=(out_axes, None),
            axis_name=axis_name,
        )(state.full(), args, **kwargs)

    return _initializer


def _get_pmap_axes(identifier):
    def _tree_fill(t1, t2, fill_value=None):
        if not isinstance(t1, dict) or not isinstance(t2, dict):
            return t1
        return {
            k: _tree_fill(t1[k], t2[k])
            if k in t1 and k in t2
            else t1[k]
            if k in t1 and k not in t2
            else fill_value
            for k in set(t1) | set(t2)
        }

    def _tree_trim(t1, t2):
        if not isinstance(t1, dict) or not isinstance(t2, dict):
            return t1
        return {k: _tree_trim(t1[k], t2[k]) for k in t2}

    try:
        mapped_state_axes = state.get(
            "mapped_state_axes",
            static=True,
            namespace=[identifier + f"-{_hash_state(batch_axes())}"],
        )
    except state.StateException:
        mapped_state_axes = None

    cur_state = state.full()
    state_out_axes = _tree_fill(mapped_state_axes, cur_state)
    state_in_axes = _tree_trim(state_out_axes, cur_state)

    return state_in_axes, state_out_axes


def pmap(fun, axis_name=None, *, in_axes=0, out_axes=0, identifier=None, **pmap_kwargs):
    if identifier is None:
        identifier = str(hash(fun))

    if axis_name is None:
        axis_name = _BatchAxis(identifier)

    def _wrapped_fun(cur_state, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(cur_state) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_pmap_fun(*args, **kwargs):
        with _track_batch_axis(axis_name=axis_name):
            state_in_axes, state_out_axes = _get_pmap_axes(identifier)
            fun_ = _lazy_initialization(
                jax.pmap(
                    _wrapped_fun,
                    axis_name,
                    in_axes=(state_in_axes, in_axes),
                    out_axes=(out_axes, state_out_axes),
                    **pmap_kwargs,
                ),
                initializer=_make_pmap_initializer(
                    fun,
                    axis_name,
                    in_axes,
                    out_axes,
                    identifier,
                ),
                identifier=identifier,
            )

            result, new_state = fun_(state.full(), args, **kwargs)

            state.update(new_state, add_missing=True)

            return result

    return _wrapped_pmap_fun


def vmap(fun, axis_name=None, *, in_axes=0, out_axes=0, identifier=None):
    if identifier is None:
        identifier = str(hash(fun))

    if axis_name is None:
        axis_name = _BatchAxis(identifier)

    def _wrapped_fun(state_dynamic, args, **kwargs):
        with _DYNAMIC_STATE_BLOCKER:
            with state.DynamicState(state_dynamic) as ds:
                result = fun(*args, **kwargs)

        return result, ds.state

    def _wrapped_vmap_fun(*args, **kwargs):
        with _track_batch_axis(axis_name=axis_name):
            state_in_axes, state_out_axes = _get_pmap_axes(identifier)
            fun_ = _lazy_initialization(
                jax.vmap(
                    _wrapped_fun,
                    in_axes=(state_in_axes, in_axes),
                    out_axes=(out_axes, state_out_axes),
                    axis_name=axis_name,
                ),
                initializer=_make_pmap_initializer(
                    fun, axis_name, in_axes, out_axes, identifier
                ),
                identifier=identifier,
            )

            result, new_state = fun_(state.full(), args, **kwargs)

            state.update(new_state, add_missing=True)

            return result

    return _wrapped_vmap_fun


def scan(fun, init, xs, *scan_args, identifier=None, **scan_kwargs):
    if identifier is None:
        identifier = str(hash(fun))

    def _wrapped_fun(cur_state, x):
        jafx_state, fn_state = cur_state
        with _DYNAMIC_STATE_BLOCKER, state.DynamicState(jafx_state) as ds:
            fn_state, y = fun(fn_state, x)
        return (ds.state, fn_state), y

    def _run_scan():
        return jax.lax.scan(
            _wrapped_fun,
            (state.full(), init),
            xs,
            *scan_args,
            **scan_kwargs,
        )

    def _initializer():
        (new_state, fn_state), y = _wrapped_fun((state.full(), init), xs[0])
        y = jax.tree_util.tree_map(lambda x: jnp.repeat(x[None], xs.shape[0], 0), y)
        return (new_state, fn_state), y

    (jafx_state, fn_state), ys = _lazy_initialization(
        _run_scan,
        initializer=_initializer,
        identifier=identifier,
    )()

    state.update(jafx_state, add_missing=True)

    return fn_state, ys


def cond(pred, true_fun, false_fun, *operands, identifier=None):
    if identifier == False:
        identifier = str(hash(true_fun) ^ hash(false_fun))

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

    def _initializer():
        _ = true_fun(*operands)
        return false_fun(*operands)

    true_fun_ = _lazy_initialization(
        lambda: true_fun_,
        initializer=lambda: true_fun(*operands),
        identifier=str(hash(true_fun)),
    )()
    false_fun_ = _lazy_initialization(
        lambda: false_fun_,
        initializer=lambda: false_fun(*operands),
        identifier=str(hash(true_fun)),
    )()

    result, new_state = _lazy_initialization(
        lambda: jax.lax.cond(
            pred,
            true_fun_,
            false_fun_,
            (state.full(), *operands),
        ),
        initializer=_initializer,
        identifier=identifier,
    )()
    state.update(new_state, add_missing=True)

    return result


checkpoint = _patch_default(jax.checkpoint)


jit = _patch_default(jax.jit)
