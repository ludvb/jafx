import sys
from functools import partial, wraps
from typing import Any, Callable, Generator


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


class ContextManager:
    """Creates a reusable context manager based on given generator function."""

    def __init__(self, fun: Callable[..., Generator[Any, None, None]], *args, **kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self._generators = []

    def __enter__(self):
        g = self.fun(*self.args, **self.kwargs)
        self._generators.append(g)
        value = next(g)
        return value

    def __exit__(self, exc_type, exc_value, traceback):
        g = self._generators.pop()
        if exc_type is None:
            try:
                next(g)
            except StopIteration:
                return False
            raise RuntimeError("Generator didn't stop")
        try:
            g.throw(exc_type, exc_value, traceback)
            raise RuntimeError("Generator didn't stop")
        except StopIteration:
            return True
        except:
            _, exc_value_, _ = sys.exc_info()
            if exc_value is not exc_value_:
                raise


def contextmanager(
    fun: Callable[..., Generator[Any, None, None]]
) -> Callable[..., ContextManager]:
    return partial(ContextManager, fun)


def as_decorator(context_manager):
    """Converts a context manager into a decorator."""

    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            if callable(context_manager):
                ctx = context_manager()
            else:
                ctx = context_manager
            with ctx:
                return fun(*args, **kwargs)

        return wrapper

    return decorator


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
