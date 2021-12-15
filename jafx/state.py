from contextlib import contextmanager
from typing import Any, Optional

import attr
import jax

from .handler import Handler, Message, NoHandlerError, send
from .util import tree_merge, tree_update


@attr.define
class FullStateMessage(Message):
    static: bool


@attr.define
class GetStateMessage(Message):
    group: str
    namespace: list[str]
    static: bool


@attr.define
class RmStateMessage(Message):
    group: str
    namespace: list[str]
    static: bool


@attr.define
class ListGroupsMessage(Message):
    static: bool


@attr.define
class SetStateMessage(Message):
    group: str
    namespace: list[str]
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
    RmStateMessage,
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

    def _set_state(self, group: str, value: Any, namespace: list[str]):
        *prefix, k = [group, *namespace]
        x = self._state
        for path in prefix:
            if path not in x:
                x[path] = {}
            x = x[path]
        x[k] = value

    def _get_state(self, group: str, namespace: list[str]):
        x = self._state[group]
        for path in namespace:
            x = x[path]
        return x

    def _rm_state(self, group: str, namespace: list[str]):
        def _rm(d, subpaths):
            cur_path, *next_paths = subpaths
            if next_paths == []:
                del d[cur_path]
            else:
                _rm(d[cur_path], next_paths)
                if len(d[cur_path]) == 0:
                    del d[cur_path]

        _rm(self._state, [group, *namespace])

    def _handle(self, message: Message) -> Any:
        if isinstance(message, SetStateMessage):
            try:
                send(message, interpret_final=False)
            except (NoHandlerError, StateException):
                pass
            self._set_state(message.group, message.value, message.namespace)
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
                return self._get_state(message.group, message.namespace)
            except KeyError:
                try:
                    return send(message, interpret_final=False)
                except NoHandlerError:
                    raise StateException(
                        'No state for group "{}" in namespace /{}'.format(
                            message.group, "/".join(message.namespace)
                        )
                    )

        if isinstance(message, RmStateMessage):
            self._rm_state(message.group, message.namespace)
            try:
                return send(message, interpret_final=False)
            except NoHandlerError:
                pass
            return

        if isinstance(message, FullStateMessage):
            try:
                result = send(message, interpret_final=False)
            except NoHandlerError:
                result = {}
            state = jax.tree_util.tree_map(lambda x: x, self._state)
            # ^ NOTE: This identity map is used to recursively copy the
            #         self._state PyTree, thereby disallowing state
            #         modifications by mutating the return value.
            return tree_merge(result, state)

        raise NotImplementedError()


class DynamicState(_State):
    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, StateMessage) and not message.static


class StaticState(_State):
    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, StateMessage) and message.static


def set(
    group: str,
    value: Any,
    static: bool = False,
    namespace: Optional[list[str]] = None,
) -> None:
    if namespace is None:
        namespace = get_namespace()
    return send(
        SetStateMessage(
            group=group,
            value=value,
            namespace=namespace,
            static=static,
        )
    )


def get(
    group: str,
    static: bool = False,
    namespace: Optional[list[str]] = None,
) -> Any:
    if namespace is None:
        namespace = get_namespace()
    return send(
        GetStateMessage(
            static=static,
            group=group,
            namespace=namespace,
        )
    )


def rm(
    group: str,
    static: bool = False,
    namespace: Optional[list[str]] = None,
) -> Any:
    if namespace is None:
        namespace = get_namespace()
    return send(
        RmStateMessage(
            static=static,
            group=group,
            namespace=namespace,
        )
    )


@contextmanager
def temp(
    group: str,
    value: Any,
    static: bool = False,
    namespace: Optional[list[str]] = None,
):
    if namespace is None:
        namespace = get_namespace()

    class NotSet:
        pass

    try:
        cur_value = get(group, static=static, namespace=namespace)
    except StateException:
        cur_value = NotSet()

    try:
        set(group, value, static=static, namespace=namespace)
        yield
    finally:
        if isinstance(cur_value, NotSet):
            rm(group, static=static, namespace=namespace)
        else:
            set(group, cur_value, static=static, namespace=namespace)


def update(
    state: dict[str, Any], add_missing: bool = False, static: bool = False
) -> None:
    return send(
        UpdateStateMessage(
            state=state,
            add_missing=add_missing,
            static=static,
        )
    )


def full(static: bool = False) -> dict[str, Any]:
    return send(FullStateMessage(static=static))


def namespace(namespace: list[str]):
    return temp(
        group="namespace",
        value=namespace,
        static=True,
        namespace=[],
    )


def scope(scope: str):
    try:
        ns = get("namespace", static=True, namespace=[])
    except StateException:
        ns = []
    return namespace(ns + [scope])


def get_namespace() -> list[str]:
    try:
        return get("namespace", static=True, namespace=[])
    except StateException:
        return []
