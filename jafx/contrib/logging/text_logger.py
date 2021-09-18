from typing import Any, Callable, NamedTuple, TextIO

import attr
from colors import color
from jax.experimental.host_callback import id_tap

from ...handler import send
from ...intercept import Intercept
from .logging import Logger, LogLevel, LogMessage


def default_log_message_formatter(message: LogMessage[str]) -> str:
    if message.level == LogLevel.DEBUG:
        level_str = color("DEBUG", fg="#555555")
    elif message.level == LogLevel.INFO:
        level_str = "INFO"
    elif message.level == LogLevel.WARNING:
        level_str = color("WARNING", fg="#ffff00")
    elif message.level == LogLevel.ERROR:
        level_str = color("ERROR", fg="#ff0000", style="bold")
    else:
        raise NotImplementedError()
    return "{}  {}  {}  {}\n".format(
        "[ {} ]".format(message.date.to_datetime_string()),
        level_str,
        color("({})".format(str(message.pid)), fg="Gray"),
        message.message,
    )


class TextLogger(Logger[str]):
    def __init__(
        self,
        output_devices: list[TextIO],
        formatter: Callable[[LogMessage[str]], str] = default_log_message_formatter,
        log_level: LogLevel = LogLevel.INFO,
    ):
        super().__init__(log_level=log_level)
        self.output_devices = output_devices
        self.formatter = formatter

    def _handle_log_message(self, log_message: LogMessage[str]):
        formatted_message = self.formatter(log_message)
        for output_device in self.output_devices:
            output_device.write(formatted_message)


class FormatString(NamedTuple):
    fmt: str
    fmt_args: list[Any] = []
    fmt_kwargs: dict[str, Any] = {}


def format_string(fmt, *args: Any, **kwargs) -> FormatString:
    return FormatString(fmt, list(args), kwargs)


def _reify_formatted_string(log_message: LogMessage[FormatString]) -> None:
    def _send_reified(x: tuple[list[Any], dict[str, Any]], _):
        fmt_args, fmt_kwargs = x
        log_message_ = attr.evolve(
            log_message,
            message=log_message.message.fmt.format(*fmt_args, **fmt_kwargs),
        )
        send(log_message_)

    id_tap(
        _send_reified, (log_message.message.fmt_args, log_message.message.fmt_kwargs)
    )


def make_string_formatter():
    return Intercept(
        fn=_reify_formatted_string,
        predicate=lambda msg: isinstance(msg, LogMessage)
        and isinstance(msg.message, FormatString),
    )
