import warnings
from typing import Optional, TypeVar

from . import state
from .util import StackedContext

HparamType = TypeVar("HparamType")


def get_hparam(
    name: str,
    default: Optional[HparamType] = None,
    set_default: bool = True,
    warn_if_unset: bool = False,
) -> HparamType:
    try:
        return state.get("hparams", static=True, namespace=[name])
    except state.StateException:
        if default is not None:
            if warn_if_unset:
                warnings.warn(f'Hparam "{name}" is unset, using default.')
            if set_default:
                set_hparam(name, default)
            return default
        raise RuntimeError(f'Hparam "{name}" is unset with no default')


def set_hparam(name: str, value: HparamType) -> HparamType:
    state.set("hparams", value, static=True, namespace=[name])
    return value


def hparams(**hparams):
    return StackedContext(
        *(
            state.temp("hparams", value, static=True, namespace=[name])
            for name, value in hparams.items()
        )
    )
