from jax.example_libraries.optimizers import adam

from .io import StateIO
from .params import ParamHandler, opt_state_io_packer
from .state import DynamicState, StaticState
from .util import StackedContext
from . import hparams


def handlers():
    return StackedContext(
        state_io=StateIO(),
        opt_state_packer=opt_state_io_packer,
        static_state=StaticState(),
        dynamic_state=DynamicState(),
        hparams=hparams(optimizer=adam, learning_rate=1e-3),
        param_store=ParamHandler(),
    )
