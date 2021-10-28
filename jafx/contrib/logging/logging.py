import os
import warnings
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar, get_args

import attr
from pendulum.datetime import DateTime

from ...handler import Handler, Message, NoHandlerError, send


class LogLevel(Enum):
    DEBUG = 0
    INFO = 10
    WARNING = 50
    ERROR = 100


MessageType = TypeVar("MessageType")


@attr.define
class LogMessage(Message, Generic[MessageType]):
    message: MessageType
    date: DateTime
    pid: int
    level: LogLevel


class Logger(Handler, Generic[MessageType], metaclass=ABCMeta):
    def __init__(self, log_level: LogLevel = LogLevel.INFO):
        super().__init__()
        self.log_level = log_level

    @abstractmethod
    def _handle_log_message(self, log_message: LogMessage[MessageType]) -> None:
        raise NotImplementedError()

    def _is_handler_for(self, message: Message) -> bool:
        return isinstance(message, LogMessage) and isinstance(
            message.message, get_args(self.__orig_bases__[0])[0]  # type: ignore
        )

    def _handle(self, message: LogMessage[MessageType]):
        if message.level.value >= self.log_level.value:
            self._handle_log_message(message)
        try:
            send(message, interpret_final=False)
        except NoHandlerError:
            pass


def log(level: LogLevel, message: Any) -> None:
    try:
        send(
            LogMessage(
                message=message,
                date=DateTime.now(),
                pid=os.getpid(),
                level=level,
            )
        )
    except NoHandlerError:
        warnings.warn(f'No handler for log message of type "{type(message).__name__}"')


def log_debug(message: Any) -> None:
    return log(LogLevel.DEBUG, message)


def log_info(message: Any) -> None:
    return log(LogLevel.INFO, message)


def log_warning(message: Any) -> None:
    return log(LogLevel.WARNING, message)


def log_error(message: Any) -> None:
    return log(LogLevel.ERROR, message)
