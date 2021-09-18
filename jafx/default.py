from .hparams import default_hparams
from .io import StateIO
from .params import opt_state_io_packer
from .state import DynamicState, StaticState
from .util import StackedContext


def handlers():
    return StackedContext(
        state_io=StateIO(),
        opt_state_packer=opt_state_io_packer,
        hparam=default_hparams,
        static_state=StaticState(),
        dynamic_state=DynamicState(),
    )
