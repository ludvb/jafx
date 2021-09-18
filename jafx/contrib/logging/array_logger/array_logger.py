from typing import Any, Callable, NamedTuple, Optional, Union

import attr
import jax
from jax.experimental.host_callback import barrier_wait, id_tap

from ....global_step import get_global_step
from ..logging import Logger, LogLevel, LogMessage, log
from .jaxboard import SummaryWriter


class LogArrayHistogramOptions(NamedTuple):
    bins: int


class LogArrayImageOptions(NamedTuple):
    pass


class LogArrayImagesOptions(NamedTuple):
    num_rows: Optional[int]
    num_cols: Optional[int]


class LogArrayScalarOptions(NamedTuple):
    pass


@attr.define
class LogArray:
    tag: str
    data: Any
    options: Union[
        LogArrayImageOptions,
        LogArrayImagesOptions,
        LogArrayScalarOptions,
        LogArrayHistogramOptions,
    ]
    transformation: Callable[[Any], Any]
    log_frequency: int


class TensorboardLogger(Logger[LogArray]):
    def __init__(
        self,
        save_path: str = "tb_logs",
        log_level: LogLevel = LogLevel.INFO,
    ):
        super().__init__(log_level=log_level)
        self._summary_writer = SummaryWriter(save_path, enable=jax.process_index() == 0)

    def _handle_log_message(self, log_message: LogMessage[LogArray]):
        def _do_log(data, _):
            data, global_step = data

            if global_step % log_message.message.log_frequency != 0:
                return

            data = log_message.message.transformation(data)
            options = log_message.message.options

            if isinstance(options, LogArrayHistogramOptions):
                self._summary_writer.histogram(
                    log_message.message.tag,
                    data,
                    bins=options.bins,
                    step=global_step,
                )
                return

            if isinstance(options, LogArrayImageOptions):
                self._summary_writer.image(
                    log_message.message.tag, data, step=global_step
                )
                return

            if isinstance(options, LogArrayImagesOptions):
                self._summary_writer.images(
                    log_message.message.tag,
                    data,
                    rows=options.num_rows,
                    cols=options.num_cols,
                    step=global_step,
                )
                return

            if isinstance(options, LogArrayScalarOptions):
                self._summary_writer.scalar(
                    log_message.message.tag, data, step=global_step
                )
                return

            raise NotImplementedError()

        id_tap(_do_log, (log_message.message.data, get_global_step()))

    def __exit__(self, exc_type, exc_value, exc_tb):
        barrier_wait()
        self._summary_writer.flush()
        self._summary_writer.close()
        return super().__exit__(exc_type, exc_value, exc_tb)


def log_histogram(
    tag: str,
    data: Any,
    bins: int = 50,
    transformation: Optional[Callable[[Any], Any]] = None,
    log_frequency: int = 1,
    level: LogLevel = LogLevel.INFO,
) -> None:
    if transformation is None:
        transformation = lambda x: x

    return log(
        level,
        LogArray(
            tag=tag,
            data=data,
            options=LogArrayHistogramOptions(bins=bins),
            transformation=transformation,
            log_frequency=log_frequency,
        ),
    )


def log_image(
    tag: str,
    data: Any,
    transformation: Optional[Callable[[Any], Any]] = None,
    log_frequency: int = 1,
    level: LogLevel = LogLevel.INFO,
) -> None:
    if transformation is None:
        transformation = lambda x: x

    return log(
        level,
        LogArray(
            tag=tag,
            data=data,
            options=LogArrayImageOptions(),
            transformation=transformation,
            log_frequency=log_frequency,
        ),
    )


def log_images(
    tag: str,
    data: Any,
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    transformation: Optional[Callable[[Any], Any]] = None,
    log_frequency: int = 1,
    level: LogLevel = LogLevel.INFO,
) -> None:
    if transformation is None:
        transformation = lambda x: x

    return log(
        level,
        LogArray(
            tag=tag,
            data=data,
            options=LogArrayImagesOptions(num_rows=num_rows, num_cols=num_cols),
            transformation=transformation,
            log_frequency=log_frequency,
        ),
    )


def log_scalar(
    tag: str,
    data: Any,
    log_frequency: int = 1,
    transformation: Optional[Callable[[Any], Any]] = None,
    level: LogLevel = LogLevel.INFO,
) -> None:
    if transformation is None:
        transformation = lambda x: x

    return log(
        level,
        LogArray(
            tag=tag,
            data=data,
            options=LogArrayScalarOptions(),
            transformation=transformation,
            log_frequency=log_frequency,
        ),
    )
