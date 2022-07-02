# TODO: implement remaining transforms: xmap, loops, etc.
# TODO: this module is very messy. when
#       https://github.com/google/jax/issues/7155 is fixed, we can probably
#       simplify substantially

import itertools as it
import operator as op
from contextlib import contextmanager
from functools import partial, total_ordering, wraps
from typing import Any, Hashable, Optional

import attr
import jax

from . import state
from .intercept import Intercept

_DYNAMIC_STATE_BLOCKER = Intercept(
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


def _get_identifier(fun, identifier=None):
    if identifier is None:
        identifier = str(hash(fun))

    ss = state.full(static=True)
    try:
        _ = ss.pop("isinit")
    except KeyError:
        pass

    return identifier + "-" + str(_hash_state(ss))


def _is_initialized(fun, identifier=None):
    try:
        return state.get(
            "isinit", static=True, namespace=[_get_identifier(fun, identifier)]
        )
    except state.StateException:
        return False


def _set_initialized(fun, value, identifier=None):
    state.set(
        "isinit", value, static=True, namespace=[_get_identifier(fun, identifier)]
    )


def _lazy_initialization(
    fun,
    initializer=None,
    identifier: Optional[str] = None,
    use_init_return_value: bool = True,
):
    if initializer is None:
        initializer = fun

    try:
        noinit = state.get("noinit", static=True, namespace=["value"])
        if noinit:
            return fun
    except state.StateException:
        pass

    @wraps(fun)
    def _wrapped_fun(*args, **kwargs):
        with state.temp("noinit", True, static=True, namespace=["value"]):
            return fun(*args, **kwargs)

    if _is_initialized(fun, identifier=identifier):
        return _wrapped_fun

    def _initializer(*args, **kwargs):
        ss = state.full(static=True)
        initializer_result = initializer(*args, **kwargs)

        _set_initialized(fun, True, identifier=identifier)

        if use_init_return_value:
            return initializer_result

        # When not using initializer return value, we reset the static state to
        # its initial state. Otherwise, counters etc. could get updated twice.
        state.update(ss, add_missing=False, static=True)

        return _wrapped_fun(*args, **kwargs)

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
def _track_batch_axis(axis_name: Hashable, axis_size: int):
    current_batch_axes = batch_axes()
    with state.namespace(["batch_axes"]):
        state.set(
            group="global",
            value=current_batch_axes | {axis_name: axis_size},
            static=True,
        )
    try:
        yield
    finally:
        with state.namespace(["batch_axes"]):
            state.set(group="global", value=current_batch_axes, static=True)


@total_ordering
class _BatchAxis:
    def __init__(self, identifier):
        self.identifier = identifier

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.identifier == other.identifier
        return False

    def __lt__(self, other):
        if isinstance(other, type(self)):
            return self.identifier < other.identifier
        return False

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        return self.identifier


def batch_axes() -> dict[str, int]:
    try:
        with state.namespace(["batch_axes"]):
            return state.get("global", static=True)
    except state.StateException:
        return {}


def value_and_grad(
    fun,
    argnums=0,
    has_aux=False,
    holomorphic=False,
    allow_int=False,
    reduce_axes=(),
    *,
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
            identifier=identifier,
            use_init_return_value=False,
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


def vjp(fn, *primals, **vjp_kwargs):
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
        partial(_wrapped_fun, state.full()),
        *primals,
        **vjp_kwargs,
        has_aux=True,
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
                namespace=[identifier + f"-{_hash_state(list(batch_axes()))}"],
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
            namespace=[identifier + f"-{_hash_state(list(batch_axes()))}"],
        )
    except state.StateException:
        mapped_state_axes = None

    cur_state = state.full()
    state_out_axes = _tree_fill(mapped_state_axes, cur_state)
    state_in_axes = _tree_trim(state_out_axes, cur_state)

    return state_in_axes, state_out_axes


@attr.define
class _LeafNode:
    # NOTE: this is used as an indicator to avoid leaf checking using a more
    #       brittle is_leaf function
    value: Any


def _parse_axes(axes):
    def _is_leaf(node):
        if isinstance(node, int):
            return True
        if isinstance(node, dict):
            return all(
                isinstance(k, int) and isinstance(v, Hashable) for k, v in node.items()
            )
        if isinstance(node, list):
            try:
                return node[-1] is ...
            except IndexError:
                return False
        return False

    def _parse_axes(in_axis):
        if isinstance(in_axis, int):
            return _LeafNode([(None, in_axis)])
        if isinstance(in_axis, dict):
            return _LeafNode([(k, i) for i, k in in_axis.items()])
        if isinstance(in_axis, list):
            return _LeafNode([(k, i) for i, k in enumerate(in_axis[:-1])])
        raise RuntimeError()

    return jax.tree_util.tree_map(_parse_axes, axes, is_leaf=_is_leaf)


def _get_axis_size(pytree, axes):
    def _size_of_subtree(axes: _LeafNode, tree):
        def _size_of_axis_in_subtree(i):
            nodes = jax.tree_util.tree_leaves(tree)
            return jax.tree_util.tree_map(lambda node: node.shape[i], nodes)

        return [
            _LeafNode((name, s))
            for name, pos in axes.value
            for s in _size_of_axis_in_subtree(pos)
        ]

    axes = _parse_axes(axes)
    subtree_sizes = jax.tree_util.tree_map(_size_of_subtree, axes, pytree)
    subtree_sizes = [x.value for x in jax.tree_util.tree_leaves(subtree_sizes)]

    def _collapse_sizes(k, sizes):
        size, *rest = sizes
        if not all(x == size for x in rest):
            sizes_str = " ,".join(map(str, sizes))
            if k is None:
                raise ValueError(f"Mismatched sizes ({sizes_str})")
            else:
                raise ValueError(f'Mismatched sizes of axis "{k}" ({sizes_str})')
        return size

    subtree_sizes_dict = {
        k: _collapse_sizes(k, [x for _, x in groups])
        for k, groups in it.groupby(sorted(subtree_sizes), key=lambda x: x[0])
    }

    return subtree_sizes_dict


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
        with _track_batch_axis(
            axis_name=axis_name, axis_size=_get_axis_size(args, in_axes)[None]
        ):
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
        with _track_batch_axis(
            axis_name=axis_name, axis_size=_get_axis_size(args, in_axes)[None]
        ):
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
        (new_state, _), _ = _wrapped_fun((state.full(), init), xs[0])
        state.update(new_state, add_missing=True)

    (jafx_state, fn_state), ys = _lazy_initialization(
        _run_scan,
        initializer=_initializer,
        identifier=identifier,
        use_init_return_value=False,
    )()

    state.update(jafx_state, add_missing=True)

    return fn_state, ys


def while_loop(cond_fun, body_fun, init_val, identifier=None):
    if identifier is None:
        identifier = str(hash(body_fun))

    def _wrapped_body_fun(cur_state):
        jafx_state, a = cur_state
        with _DYNAMIC_STATE_BLOCKER, state.DynamicState(jafx_state) as ds:
            a = body_fun(a)
        return ds.state, a

    def _wrapped_cond_fun(cur_state):
        _, a = cur_state
        return cond_fun(a)

    def _run_while_loop():
        return jax.lax.while_loop(
            _wrapped_cond_fun,
            _wrapped_body_fun,
            (state.full(), init_val),
        )

    def _initializer():
        new_state, _ = _wrapped_body_fun((state.full(), init_val))
        state.update(new_state, add_missing=True)

    jafx_state, a = _lazy_initialization(
        _run_while_loop,
        initializer=_initializer,
        identifier=identifier,
        use_init_return_value=False,
    )()

    state.update(jafx_state, add_missing=True)

    return a


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

    def _run_cond():
        return jax.lax.cond(
            pred,
            true_fun_,
            false_fun_,
            (state.full(), *operands),
        )

    def _initializer():
        _, new_state1 = true_fun_((state.full(), *operands))
        _, new_state2 = false_fun_((state.full(), *operands))
        state.update(new_state1, add_missing=True)
        state.update(new_state2, add_missing=True)

    result, new_state = _lazy_initialization(
        _run_cond,
        initializer=_initializer,
        identifier=identifier,
        use_init_return_value=False,
    )()
    state.update(new_state, add_missing=True)

    return result


checkpoint = _patch_default(jax.checkpoint)


jit = _patch_default(jax.jit)
