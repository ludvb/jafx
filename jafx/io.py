import pickle
from typing import Any

import attr
import jax
import jax.numpy as jnp
import numpy as np

from . import state
from .handler import Handler, Message, NoHandlerError, ReturnValue, send


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
            return ReturnValue(None)

        if isinstance(message, LoadStateMessage):
            try:
                send(message=message, interpret_final=False)
            except NoHandlerError:
                pass
            with open(message.filename, "rb") as f:
                s = pickle.load(f)

            # jnp.ndarrays are pickled as np.ndarrays, thus need to convert
            # them back
            # TODO: how should we deal with sharded arrays?
            s = jax.tree_map(
                lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, s
            )

            state.update(s, add_missing=True)
            return ReturnValue(None)


def save_dynamic_state(filename: str) -> None:
    send(SaveStateMessage(filename=filename, state=state.full()))


def load_dynamic_state(filename: str) -> Any:
    send(LoadStateMessage(filename=filename))
