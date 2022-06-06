from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Union

import attr
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import barrier_wait, id_tap

from .... import get_namespace
from ....global_step import get_global_step
from ..logging import Logger, LogLevel, LogMessage, log
from .jaxboard import SummaryWriter


class LogArrayImageOptions(NamedTuple):
    num_rows: Optional[int]
    num_cols: Optional[int]
    padding: Optional[float]
    normalize: bool


class LogArrayScalarOptions(NamedTuple):
    bins: Optional[int]


class LogArrayVideoOptions(NamedTuple):
    num_rows: Optional[int]
    num_cols: Optional[int]
    padding: Optional[float]
    normalize: bool
    fps: Optional[float]
    duration: Optional[float]


@attr.define
class LogArray:
    tag: str
    data: Any
    transformation: Callable[[Any], jnp.ndarray]
    options: Union[
        LogArrayImageOptions,
        LogArrayScalarOptions,
        LogArrayVideoOptions,
    ]
    log_frequency: int


def make_grid(
    images: jnp.ndarray,
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    padding: Optional[float] = None,
    normalize: bool = True,
) -> jnp.ndarray:
    if images.ndim == 3:
        images = images[None]

    if images.ndim > 4:
        images = images.reshape(-1, *images.shape[-3:])

    if normalize:
        images = images - images.min()
        images = images / images.max()

    if padding is not None and images.shape[0] > 1:
        padding_px = int(np.round(padding * max(images.shape[1:3])))
        images = jnp.pad(
            images,
            (
                (0, 0),
                (padding_px, padding_px),
                (padding_px, padding_px),
                (0, 0),
            ),
        )

    n = images.shape[0]

    if num_cols is not None:
        num_cols = num_cols
        num_rows = int(np.ceil(n / num_cols))
    elif num_rows is not None:
        num_rows = num_cols
        num_cols = int(np.ceil(n / num_rows))
    else:
        num_cols = int(np.ceil(np.sqrt(n)))
        num_rows = int(np.ceil(n / num_cols))

    n_padded = num_rows * num_cols
    images = jnp.pad(images, ((0, n_padded - n), (0, 0), (0, 0), (0, 0)))

    _, H, W, C = images.shape
    images = images.reshape(num_rows, num_cols, H, W, C)
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape(num_rows * H, num_cols * W, C)

    return images


class TensorboardLogger(Logger[LogArray]):
    def __init__(
        self,
        save_path: str = "tb_logs",
        log_level: LogLevel = LogLevel.INFO,
    ):
        super().__init__(log_level=log_level)
        self._summary_writer = SummaryWriter(save_path, enable=jax.process_index() == 0)

    def _handle_log_message(self, log_message: LogMessage[LogArray]):
        ns = get_namespace()

        def _do_log(data, _):
            data, global_step = data

            if global_step % log_message.message.log_frequency != 0:
                return

            options = log_message.message.options
            data = log_message.message.transformation(data)
            tag = "/".join([*ns, log_message.message.tag])

            if isinstance(options, LogArrayScalarOptions):
                if data.ndim == 0:
                    self._summary_writer.scalar(tag, data, step=global_step)
                else:
                    self._summary_writer.histogram(
                        tag, data, bins=options.bins or 10, step=global_step
                    )
                return

            if isinstance(options, LogArrayImageOptions):
                self._summary_writer.image(
                    tag,
                    make_grid(
                        data,
                        num_rows=options.num_rows,
                        num_cols=options.num_cols,
                        padding=options.padding,
                        normalize=options.normalize,
                    ),
                    step=global_step,
                )
                return

            if isinstance(options, LogArrayVideoOptions):
                self._summary_writer.video(
                    tag,
                    jax.vmap(
                        partial(
                            make_grid,
                            num_rows=options.num_rows,
                            num_cols=options.num_cols,
                            padding=options.padding,
                            normalize=options.normalize,
                        ),
                        in_axes=-4,
                        out_axes=-4,
                    )(data),
                    step=global_step,
                    fps=options.fps,
                    duration=options.duration,
                )
                return

            raise NotImplementedError()

        id_tap(_do_log, (log_message.message.data, get_global_step()))

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            barrier_wait()
            self._summary_writer.flush()
            self._summary_writer.close()
        except:
            pass
        return super().__exit__(exc_type, exc_value, exc_tb)


def log_scalar(
    tag: str,
    data: Any,
    bins: int = 50,
    transformation: Optional[Callable[[Any], jnp.ndarray]] = None,
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
            options=LogArrayScalarOptions(bins=bins),
            transformation=transformation,
            log_frequency=log_frequency,
        ),
    )


def log_image(
    tag: str,
    data: Any,
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    padding: Optional[float] = None,
    normalize: bool = True,
    transformation: Optional[Callable[[Any], jnp.ndarray]] = None,
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
            options=LogArrayImageOptions(
                num_rows=num_rows,
                num_cols=num_cols,
                padding=padding,
                normalize=normalize,
            ),
            transformation=transformation,
            log_frequency=log_frequency,
        ),
    )


def log_video(
    tag: str,
    data: Any,
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    padding: Optional[float] = None,
    normalize: bool = True,
    fps: Optional[float] = None,
    duration: Optional[float] = None,
    transformation: Optional[Callable[[Any], jnp.ndarray]] = None,
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
            options=LogArrayVideoOptions(
                num_rows=num_rows,
                num_cols=num_cols,
                padding=padding,
                normalize=normalize,
                fps=fps,
                duration=duration,
            ),
            transformation=transformation,
            log_frequency=log_frequency,
        ),
    )
