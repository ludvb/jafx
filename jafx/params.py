"""Utilities for gradient-based optimization
"""

from typing import Any, NamedTuple, Optional

import attr
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import (
    Optimizer,
    OptimizerState,
    adam,
    pack_optimizer_state,
    unpack_optimizer_state,
)

from . import state, transforms
from .handler import Handler, Message, NoHandlerError, ReturnValue, send
from .hparams import get_hparam
from .intercept import Intercept
from .io import LoadStateMessage, SaveStateMessage, StateIOMessage
from .transforms import _is_initialized, _set_initialized


class NoParamException(Exception):
    def __init__(self, param_name):
        super().__init__()
        self._param_name = param_name

    def __repr__(self) -> str:
        return f'No parameter named "{self._param_name}"'


@attr.define
class ParamMessage(Message):
    pass


@attr.define
class GetParam(ParamMessage):
    name: str
    default_value: Any = None


@attr.define
class UpdateParams(ParamMessage):
    grads: Any


def _get_param(name: str, default_value: Any) -> jnp.ndarray:
    with state.scope(name):
        try:
            param = state.get("param_state")
        except state.StateException:
            optimizer_constructor = get_hparam("optimizer", adam, warn_if_unset=True)
            lr = get_hparam("learning_rate", 1e-3, warn_if_unset=True)

            optimizer = optimizer_constructor(lr)
            state.set("opt", optimizer, static=True)

            try:
                opt_state = state.get("opt_state")
                _ = state.get("param_step")
                param = jax.tree_map(jnp.array, optimizer.params_fn(opt_state))
                state.set("param_state", param)

            except state.StateException:
                if default_value is None:
                    raise NoParamException(name)

                param = default_value
                opt_state = optimizer.init_fn(param)
                state.set("opt_state", opt_state)
                state.set("param_state", param)
                state.set("param_step", 0)

    return param


def _update_params(grads) -> None:
    opt_state = state.full()["opt_state"]
    opt = state.full(static=True)["opt"]
    param_step = state.full()["param_step"]

    opt_state_new = jax.tree_map(
        lambda opt, opt_state, param_step, g: opt.update_fn(param_step, g, opt_state),
        opt,
        opt_state,
        param_step,
        grads,
        is_leaf=lambda x: isinstance(x, Optimizer),
    )
    param_state_new = jax.tree_map(
        lambda opt, opt_state: opt.params_fn(opt_state),
        opt,
        opt_state_new,
        is_leaf=lambda x: isinstance(x, Optimizer),
    )
    param_step_new = jax.tree_map(lambda x: x + 1, param_step)

    state.update(
        {
            "opt_state": opt_state_new,
            "param_state": param_state_new,
            "param_step": param_step_new,
        }
    )
    update_global_step()


class ParamHandler(Handler):
    def _handle(self, message: Message) -> Any:
        match message:
            case GetParam( name=name, default_value=default_value):
                param = _get_param(name, default_value)
                return ReturnValue(param)

            case UpdateParams(grads=grads):
                _update_params(grads)
                return


def param(
    name: str,
    default_value: Any = None,
) -> Any:
    return send( message=GetParam( name=name, default_value=default_value))


def update_params(grads: Any) -> None:
    try:
        send(message=UpdateParams(grads=grads))
    except NoHandlerError:
        pass


def get_global_step() -> int:
    with state.namespace(["global_step"]):
        try:
            return state.get("global")
        except state.StateException:
            return 0


def update_global_step(new_step: Optional[int] = None) -> None:
    if new_step is None:
        new_step = get_global_step() + 1
    with state.namespace(["global_step"]):
        state.set("global", new_step)


def _patch_grad_transform(transform):
    def _wrapped_transform(fun, *args, identifier=None, **kwargs):
        def _wrapped_fun(param_state, *args_, **kwargs_):
            with state.DynamicState({"param_state": param_state.copy()}) as ds:
                return fun(*args_, **kwargs_)

        def _run_transform(*args_, **kwargs_):
            # Run initialization separately, as we need to know the
            # post-initialization "param_state" before function call.
            if not _is_initialized(fun, identifier=identifier):
                _ = fun(*args_, **kwargs_)
                _set_initialized(fun, True, identifier=identifier)

            try:
                param_state = state.full()["param_state"]
            except KeyError:
                param_state = {}
            return transform(_wrapped_fun, *args, **kwargs)(
                param_state, *args_, **kwargs_
            )

        return _run_transform

    return _wrapped_transform


value_and_param_grad = _patch_grad_transform(transforms.value_and_grad)
param_grad = _patch_grad_transform(transforms.grad)


def param_vjp(fun, *, identifier=None, **vjp_kwargs):
    result, f_vjp = _patch_grad_transform(
        lambda wrapped_fun: lambda param_state: transforms.vjp(
            wrapped_fun, param_state, **vjp_kwargs
        )
    )(fun, identifier=identifier)()
    f_vjp_unpacked = lambda *args, **kwargs: f_vjp(*args, **kwargs)[0]
    return result, f_vjp_unpacked


class UnpackedOptimizerState(NamedTuple):
    data: Any


def _opt_state_io_packer_fn(message: Message):
    if isinstance(message, SaveStateMessage):
        try:
            packed_opt_state = message.state["opt_state"]
        except KeyError:
            return send(message=message, interpret_final=False)
        unpacked_opt_state = jax.tree_map(
            lambda x: UnpackedOptimizerState(data=unpack_optimizer_state(x)),
            packed_opt_state,
            is_leaf=lambda x: isinstance(x, OptimizerState),
        )
        message.state["opt_state"] = unpacked_opt_state
        try:
            del message.state["param_state"]
        except KeyError:
            pass
        return send(message=message, interpret_final=False)

    if isinstance(message, LoadStateMessage):
        send(message=message, interpret_final=False)
        try:
            unpacked_opt_state = state.full()["opt_state"]
        except KeyError:
            return
        packed_opt_state = jax.tree_map(
            lambda x: pack_optimizer_state(x.data),
            unpacked_opt_state,
            is_leaf=lambda x: isinstance(x, UnpackedOptimizerState),
        )
        state.update({"opt_state": packed_opt_state})
        return ReturnValue(None)


opt_state_io_packer = Intercept(
    _opt_state_io_packer_fn, predicate=lambda x: isinstance(x, StateIOMessage)
)
