from abc import ABCMeta, abstractmethod
from typing import Any


class Message:
    pass


class NoHandlerError(RuntimeError):
    pass


class Handler(metaclass=ABCMeta):
    @abstractmethod
    def _handle(self, message: Message) -> Any:
        pass

    @abstractmethod
    def _is_handler_for(self, message: Message) -> bool:
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
            if handler._is_handler_for(message):
                return handler._handle(message)
        else:
            raise NoHandlerError("Unhandled message: " + str(message))
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
