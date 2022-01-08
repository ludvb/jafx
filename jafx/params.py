from typing import Any, Callable, NamedTuple, Optional

import jax.numpy as jnp
from jax.example_libraries.optimizers import (
    Optimizer,
    OptimizerState,
    adam,
    pack_optimizer_state,
    unpack_optimizer_state,
)
from jax.tree_util import tree_map, tree_multimap

from . import state
from .global_step import update_global_step
from .handler import Message, send
from .hparams import get_hparam
from .intercept import Intercept
from .io import LoadStateMessage, SaveStateMessage, StateIOMessage


class NoParamException(Exception):
    def __init__(self, param_name):
        super().__init__()
        self._param_name = param_name

    def __repr__(self) -> str:
        return f'No parameter named "{self._param_name}"'


def param(
    name: str,
    default_value: jnp.ndarray = None,
    optimizer_constructor: Optional[Callable[[float], Optimizer]] = None,
    lr: Optional[float] = None,
    lr_multiplier: float = 1.0,
) -> Any:
    with state.scope(name):
        try:
            param = state.get("param_state")
        except state.StateException:

            if optimizer_constructor is None:
                optimizer_constructor = get_hparam(
                    "optimizer", adam, warn_if_unset=True
                )

            if lr is None:
                lr = get_hparam("learning_rate", 1e-3, warn_if_unset=True)
            lr = lr * lr_multiplier

            optimizer = optimizer_constructor(lr)
            state.set("opt", optimizer, static=True)

            try:
                opt_state = state.get("opt_state")
                _ = state.get("param_step")
                param = optimizer.params_fn(opt_state)
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


def update_params(grads: Any) -> None:
    opt_state = state.full()["opt_state"]
    opt = state.full(static=True)["opt"]
    param_step = state.full()["param_step"]

    opt_state_new = tree_multimap(
        lambda opt, opt_state, param_step, g: opt.update_fn(param_step, g, opt_state),
        opt,
        opt_state,
        param_step,
        grads,
        is_leaf=lambda x: isinstance(x, Optimizer),
    )
    param_state_new = tree_multimap(
        lambda opt, opt_state: opt.params_fn(opt_state),
        opt,
        opt_state_new,
        is_leaf=lambda x: isinstance(x, Optimizer),
    )
    param_step_new = tree_map(lambda x: x + 1, param_step)

    state.update(
        {
            "opt_state": opt_state_new,
            "param_state": param_state_new,
            "param_step": param_step_new,
        }
    )
    update_global_step()


class UnpackedOptimizerState(NamedTuple):
    data: Any


def _opt_state_io_packer_fn(message: Message):
    if isinstance(message, SaveStateMessage):
        try:
            packed_opt_state = message.state["opt_state"]
        except KeyError:
            return send(message=message, interpret_final=False)
        unpacked_opt_state = tree_map(
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
        packed_opt_state = tree_map(
            lambda x: pack_optimizer_state(x.data),
            unpacked_opt_state,
            is_leaf=lambda x: isinstance(x, UnpackedOptimizerState),
        )
        state.update({"opt_state": packed_opt_state})
        return


opt_state_io_packer = Intercept(
    _opt_state_io_packer_fn, predicate=lambda x: isinstance(x, StateIOMessage)
)
