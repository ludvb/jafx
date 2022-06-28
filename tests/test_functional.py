import sys
from functools import partial
from glob import glob

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map, tree_reduce

from jafx import default, param, state, transforms
from jafx.contrib.haiku import wrap_haiku
from jafx.contrib.logging import (
    LogLevel,
    TensorboardLogger,
    TextLogger,
    log_image,
    log_info,
    log_scalar,
)
from jafx.contrib.logging.text_logger import format_string, make_string_formatter
from jafx.hparams import hparams
from jafx.io import load_dynamic_state, save_dynamic_state
from jafx.params import update_params


def _compute_loss(z):
    x = param("x", jnp.ones((10, 10, 1)))
    conv = wrap_haiku("conv", lambda x: hk.Conv2D(3, 3)(x))
    x = conv(x)
    log_info(format_string("x.sum() = {}", x.sum()))
    log_scalar("x_sum", x.sum(), log_frequency=5)
    log_image("x_image", x, log_frequency=5)
    x = (z * x**2).sum()
    return x


@partial(transforms.pmap, axis_name="num_devices")
def _update_state(z):
    loss, grad = transforms.value_and_param_grad(_compute_loss)(z)
    grad = jax.lax.pmean(grad, axis_name="num_devices")
    loss = jax.lax.pmean(loss, axis_name="num_devices")
    update_params(grad)
    return loss


def test_functional(tmp_path):
    loss = float("inf")
    with TextLogger([sys.stderr], log_level=LogLevel.DEBUG), make_string_formatter():
        with TensorboardLogger(tmp_path / "tb_logs"):
            with default.handlers(), hparams(learning_rate=0.01):
                for _ in range(100):
                    loss = _update_state(jnp.ones(jax.device_count()))
                    log_info("loss = " + str(loss.mean()))

    assert loss < 1.0
    assert len(glob(str(tmp_path / "tb_logs" / "*"))) == 1


def test_save_load(tmp_path):
    with default.handlers(), hparams(learning_rate=0.01):
        old_losses = []
        for _ in range(5):
            old_losses.append(_update_state(jnp.ones(jax.device_count())))
        save_dynamic_state(tmp_path / "state.pkl")

        old_opt_state = state.full()["opt_state"]

    with default.handlers(), hparams(learning_rate=0.01):
        load_dynamic_state(tmp_path / "state.pkl")

        new_opt_state = state.full()["opt_state"]
        assert tree_reduce(
            lambda a, x: a and x,
            tree_map(
                lambda a, b: (a == b).all(),
                old_opt_state,
                new_opt_state,
            ),
        ), "\n".join(
            [
                "Restored opt_state is different.",
                f" Old: {str(old_opt_state)}",
                f" New: {str(new_opt_state)}",
            ]
        )

        new_losses = []
        for _ in range(5):
            new_losses.append(_update_state(jnp.ones(jax.device_count())))

    assert np.mean(new_losses) < np.min(old_losses)
