import attr
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any


@attr.define
class Message:
    pass


@attr.define
class ReturnValue:
    value: Any


class NoHandlerError(RuntimeError):
    pass


class Handler(metaclass=ABCMeta):
    @abstractmethod
    def _handle(self, message: Message) -> Any:
        pass

    def __enter__(self):
        _STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # assert _STACK.pop() == self
        x = _STACK.pop()
        assert x == self, (x, self)


_STACK: list[Handler] = []
_STACK_PTR: list[int] = []


def send(message: Message, interpret_final: bool = True) -> Any:
    if interpret_final or _STACK_PTR == []:
        _STACK_PTR.append(len(_STACK) - 1)
    else:
        _STACK_PTR.append(_STACK_PTR[-1])

    try:
        while _STACK_PTR[-1] >= 0:
            stack_ptr = _STACK_PTR[-1]
            _STACK_PTR[-1] -= 1
            handler = _STACK[stack_ptr]
            match handler._handle(message):
                case None:
                    pass
                case ReturnValue(value):
                    return value
                case value:
                    warnings.warn(
                        "Handler return value was not wrapped in `ReturnValue`."
                        " This is discouraged, since it can introduce bugs if the return value is `None`."
                    )
                    return value
        else:
            raise NoHandlerError("Unhandled message: " + type(message).__name__)
    finally:
        _STACK_PTR.pop()


def get_instances(h: type, interpret_final: bool = True) -> list[Handler]:
    hs = []
    if interpret_final:
        stack_ptr = len(_STACK) - 1
    else:
        stack_ptr = _STACK_PTR[-1]
    while stack_ptr >= 0:
        h_inst = _STACK[stack_ptr]
        if isinstance(h_inst, h):
            hs.append(h_inst)
    return hs
