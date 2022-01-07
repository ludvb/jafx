# JaFx: Effectful JAX

**WARNING**:
JaFx is experimental.
Expect bugs, breaking API changes, and missing documentation.
Do not use in production.

JaFx provides effect handlers for machine learning models written in [JAX](https://github.com/google/jax).
A design goal of JaFx is to separate training from model logic.
New model components can be introduced without changing other parts of the code, making it trivial to revise or extend existing designs.
JaFx is best suited as a playground to iterate new ideas for machine learning models.


## Installation

JaFx can be installed using `pip` by running the command

``` sh
pip install --user --upgrade git+https://github.com/ludvb/jafx
```

JAX needs to be installed separately, since it has different releases depending on your CUDA version.
Follow the instructions [here](https://github.com/google/jax#installation) to install JAX.


## Examples

### Ordinary least squares in JAX and JaFx

``` python
import jax
import jafx
import jax.numpy as jnp
import numpy as np
from jax.example_libraries.optimizers import adam

X = np.linspace(0, 10, num=50)
Y = -3.0 + 1.5 * X + np.random.normal(size=50)


## Pure JAX:
params = {"b0": jnp.array(0.0), "b1": jnp.array(0.0)}
opt = adam(step_size=0.01)
opt_state = opt.init_fn(params)

def jax_loss(params):
    b0, b1 = params["b0"], params["b1"]
    y = b0 + b1 * X
    loss = ((y - Y) ** 2).sum()
    return loss

@jax.jit
def jax_step(opt_state, step):
    params = opt.params_fn(opt_state)
    grad = jax.grad(jax_loss)(params)
    opt_state = opt.update_fn(step, grad, opt_state)
    return opt_state

for step in range(1000):
    opt_state = jax_step(opt_state, step)
print("Result: " + str(opt.params_fn(opt_state)))


## JaFx style:
def jafx_loss():
    # Parameters are defined where used in model code and
    # initialized implicitly
    b0 = jafx.param("b0", jnp.array(0.0))
    b1 = jafx.param("b1", jnp.array(0.0))
    y = b0 + b1 * X
    loss = ((y - Y) ** 2).sum()
    return loss

@jafx.jit
def jafx_step():
    grad = jafx.param_grad(jafx_loss)()
    jafx.update_params(grad)

with jafx.default.handlers(), jafx.hparams(learning_rate=0.01):
    for _ in range(1000):
        jafx_step()
    print("Result: " + str(jafx.state.full()["param_state"]))
```


### Wrapping Haiku modules in JaFx

[Haiku](https://github.com/deepmind/dm-haiku) modules can be wrapped inside JaFx models for additional expressivity:

``` python
import haiku as hk
import jax
import jafx
import jax.numpy as jnp
import numpy as np
from jafx.contrib.haiku import wrap_haiku

X = np.linspace(0, 10, num=50)
Y = -3.0 + 1.5 * X + np.random.normal(size=50)

def model(X):
    X = X[:, None]
    X = hk.Linear(5)(X)
    X = jax.nn.tanh(X)
    X = hk.Linear(1)(X)
    X = X.flatten()
    return X

def loss():
    predictor = wrap_haiku("model", model)
    y = predictor(X)
    loss = ((y - Y) ** 2).sum()
    return loss

@jafx.jit
def step():
    grad = jafx.param_grad(loss)()
    jafx.update_params(grad)

with jafx.default.handlers(), jafx.hparams(learning_rate=0.01):
    for _ in range(1000):
        step()
    print("Data:       " + str(Y))
    print("Prediction: " + str(wrap_haiku("model", model)(X)))
```


### Tensorboard logging

JaFx comes with effect handlers for logging in [Tensorboard](https://github.com/tensorflow/tensorboard) using Jaxboard from [Trax](https://github.com/google/trax):

``` python
import jafx
import jax.numpy as jnp
import numpy as np
from jafx.contrib.logging import TensorboardLogger, log_scalar

X = np.linspace(0, 10, num=50)
Y = -3.0 + 1.5 * X + np.random.normal(size=50)

def loss():
    b0 = jafx.param("b0", jnp.array(0.0))
    b1 = jafx.param("b1", jnp.array(0.0))
    y = b0 + b1 * X
    loss = ((y - Y) ** 2).sum()
    log_scalar("loss", loss)
    return loss

@jafx.jit
def step():
    grad = jafx.param_grad(loss)()
    jafx.update_params(grad)

with TensorboardLogger("./tb-logs"):
    with jafx.default.handlers(), jafx.hparams(learning_rate=0.01):
        for _ in range(1000):
            step()
```


## Related projects

- [NumPyro](https://github.com/pyro-ppl/numpyro/): Probabilistic programming in JAX using extensible effects
