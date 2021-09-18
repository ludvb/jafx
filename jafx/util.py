from typing import Any


def tree_merge(tree_a: Any, tree_b: Any) -> Any:
    if isinstance(tree_a, dict) and isinstance(tree_b, dict):
        result = {k: None for k in list(tree_a) + list(tree_b)}
        for k in result:
            if k in tree_a and k not in tree_b:
                result[k] = tree_a[k]
            elif k in tree_b and k not in tree_a:
                result[k] = tree_b[k]
            else:
                result[k] = tree_merge(tree_a[k], tree_b[k])
        return result
    return tree_b


def tree_update(tree_a: Any, tree_b: Any) -> Any:
    if isinstance(tree_a, dict) and isinstance(tree_b, dict):
        result = {}
        for k in tree_a:
            if k in tree_b:
                result[k] = tree_update(tree_a[k], tree_b[k])
            else:
                result[k] = tree_a[k]
        return result
    else:
        return tree_b


class StackedContext:
    def __init__(self, *ctx, **named_ctx):
        self._ctx = ctx
        self._named_ctx = named_ctx

    def __dict__(self):
        return self._named_ctx

    def __enter__(self):
        for ctx in [*self._ctx, *self._named_ctx.values()]:
            ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        for ctx in reversed([*self._ctx, *self._named_ctx.values()]):
            ctx.__exit__(exc_type, exc_value, exc_tb)
