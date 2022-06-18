import jafx


def test_while_loop():
    def cond_fn(x):
        return x < 10

    def body_fn(x):
        return x + 1

    with jafx.default.handlers():
        assert jafx.while_loop(cond_fn, body_fn, 0) == 10
