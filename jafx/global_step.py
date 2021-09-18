from typing import Optional

from . import Namespace, state


def get_global_step() -> int:
    with Namespace(["global_step"]):
        try:
            return state.get("global")
        except state.StateException:
            return 0


def update_global_step(new_step: Optional[int] = None) -> None:
    if new_step is None:
        new_step = get_global_step() + 1
    with Namespace(["global_step"]):
        state.set("global", new_step)
