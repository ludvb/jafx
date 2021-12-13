# TODO: store hparams as static state

from typing import Any, Callable, Optional

import attr
from jax.example_libraries.optimizers import Optimizer, adam

from .handler import Handler, Message, send


@attr.define
class GetHParam(Message):
    name: str


class HParams(Handler):
    def __init__(
        self,
        default_optimizer: Optional[Callable[[float], Optimizer]] = None,
        default_lr: Optional[float] = None,
        **kwargs: Any,
    ):
        self.default_optimizer = default_optimizer
        self.default_lr = default_lr
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _handle(self, message: Message) -> Any:
        if isinstance(message, GetHParam):
            try:
                hparam = getattr(self, message.name)
            except AttributeError:
                hparam = None
            if hparam is not None:
                return hparam
            return send(message=message, interpret_final=False)
        raise RuntimeError()

    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, GetHParam)


def get_hparam(name: str) -> Any:
    return send(GetHParam(name))


def default_optimizer() -> Callable[[float], Optimizer]:
    return get_hparam("default_optimizer")


def default_lr() -> float:
    return get_hparam("default_lr")


default_hparams = HParams(
    default_optimizer=lambda lr: adam(lr),
    default_lr=1e-3,
)
