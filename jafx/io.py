import pickle
from typing import Any

import attr

from . import state
from .handler import Handler, Message, NoHandlerError, send


@attr.define
class SaveStateMessage(Message):
    filename: str
    state: Any


@attr.define
class LoadStateMessage(Message):
    filename: str


StateIOMessage = (SaveStateMessage, LoadStateMessage)


class StateIO(Handler):
    def _handle(self, message: Message) -> Any:
        if isinstance(message, SaveStateMessage):
            with open(message.filename, "wb") as f:
                pickle.dump(message.state, f)
            try:
                send(message=message, interpret_final=False)
            except NoHandlerError:
                pass
            return

        if isinstance(message, LoadStateMessage):
            try:
                send(message=message, interpret_final=False)
            except NoHandlerError:
                pass
            with open(message.filename, "rb") as f:
                s = pickle.load(f)
            state.update(s, add_missing=True)
            return

        raise RuntimeError()

    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, StateIOMessage)


def save_dynamic_state(filename: str) -> None:
    send(SaveStateMessage(filename=filename, state=state.full()))


def load_dynamic_state(filename: str) -> Any:
    send(LoadStateMessage(filename=filename))
