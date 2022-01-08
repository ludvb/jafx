from typing import Any, Callable, Optional

from .handler import Handler, Message, NoHandlerError


def _raise_no_handler_error(_):
    raise NoHandlerError()


class Intercept(Handler):
    def __init__(
        self,
        fn: Optional[Callable[[Message], Any]] = None,
        predicate: Optional[Callable[[Message], bool]] = None,
    ):
        if fn is None:
            fn = _raise_no_handler_error
        if predicate is None:
            predicate = lambda _: True

        super().__init__()

        self.fn = fn
        self.predicate = predicate

    def _handle(self, message: Message) -> Any:
        return self.fn(message)

    def _is_handler_for(self, message: Message) -> bool:
        return self.predicate(message)
