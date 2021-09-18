from typing import Any, Optional

import attr
from haiku import data_structures as ds

from .handler import Handler, Message, NoHandlerError, send


@attr.define
class GetNamespaceMessage(Message):
    pass


class Namespace(Handler):
    def __init__(
        self, namespace: Optional[list[str]] = None, scope: Optional[str] = None
    ):
        super().__init__()
        self.namespace = namespace
        self.scope = scope

    def _handle(self, message: Message) -> Any:
        if isinstance(message, GetNamespaceMessage):
            if self.namespace is not None:
                return self.namespace
            prefix = current_namespace(interpret_final=False)
            if self.scope is not None:
                prefix.append(self.scope)
            return prefix

        raise NotImplementedError()

    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, (GetNamespaceMessage,))


def current_namespace(interpret_final=True) -> list[str]:
    try:
        return send(message=GetNamespaceMessage(), interpret_final=interpret_final)
    except NoHandlerError:
        return []
