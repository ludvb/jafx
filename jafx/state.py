from typing import Any, Optional

import attr

from .handler import Handler, Message, NoHandlerError, send
from .namespace import current_namespace
from .util import tree_merge, tree_update


@attr.define
class FullStateMessage(Message):
    static: bool


@attr.define
class GetStateMessage(Message):
    group: str
    static: bool


@attr.define
class ListGroupsMessage(Message):
    static: bool


@attr.define
class SetStateMessage(Message):
    group: str
    value: Any
    static: bool


@attr.define
class UpdateStateMessage(Message):
    state: dict[str, Any]
    add_missing: bool
    static: bool


StateMessage = (
    FullStateMessage,
    GetStateMessage,
    ListGroupsMessage,
    SetStateMessage,
    UpdateStateMessage,
)


class StateException(Exception):
    pass


class _State(Handler):
    def __init__(self, initial_state: Optional[Any] = None):
        if initial_state is None:
            initial_state = {}
        super().__init__()
        self._state = initial_state

    @property
    def state(self):
        return self._state

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self._state))

    def _set_in_current_scope(self, group: str, value: Any):
        *prefix, k = current_namespace()
        x = self._state
        for path in [group, *prefix]:
            if path not in x:
                x[path] = {}
            x = x[path]
        x[k] = value

    def _get_in_current_scope(self, group: str):
        x = self._state[group]
        for path in current_namespace():
            x = x[path]
        return x

    def _handle(self, message: Message) -> Any:
        if isinstance(message, SetStateMessage):
            try:
                send(message, interpret_final=False)
            except (NoHandlerError, StateException):
                pass
            self._set_in_current_scope(message.group, message.value)
            return

        if isinstance(message, UpdateStateMessage):
            try:
                send(message, interpret_final=False)
            except (NoHandlerError, StateException):
                pass
            if message.add_missing:
                self._state = tree_merge(self._state, message.state)
            else:
                self._state = tree_update(self._state, message.state)
            return

        if isinstance(message, GetStateMessage):
            try:
                return self._get_in_current_scope(message.group)
            except KeyError:
                try:
                    return send(message, interpret_final=False)
                except NoHandlerError:
                    raise StateException(
                        'No state for group "{}" in namespace /{}'.format(
                            message.group, "/".join(current_namespace())
                        )
                    )

        if isinstance(message, FullStateMessage):
            try:
                result = send(message, interpret_final=False)
            except NoHandlerError:
                result = {}
            return tree_merge(result, self._state)

        raise NotImplementedError()


class DynamicState(_State):
    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, StateMessage) and not message.static


class StaticState(_State):
    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, StateMessage) and message.static


def set(group: str, value: Any, static: bool = False) -> None:
    return send(SetStateMessage(group=group, value=value, static=static))


def get(group: str, static: bool = False) -> Any:
    return send(GetStateMessage(static=static, group=group))


def update(
    state: dict[str, Any], add_missing: bool = False, static: bool = False
) -> None:
    return send(UpdateStateMessage(state=state, add_missing=add_missing, static=static))


def full(static: bool = False) -> dict[str, Any]:
    return send(FullStateMessage(static=static))
