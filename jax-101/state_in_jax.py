"""
Changing the state is considered a side-effect, resulting in non-pure
functions in functional programming.

This is critical - if changing the state of any mutable is a side effect,
how can we do anything? change model parameters, optimiser state?
"""
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
import matplotlib.pyplot as plt

# A simple counter example

print("\nnormal counter!\n")

class Counter:
    def __init__(self):
        self.n = 0

    def count(self) -> int:
        self.n += 1
        return self.n

    def reset(self):
        self.n = 0

counter = Counter()

for __ in range(3):
    print(counter.count())

# now CONSIDER: the n attribute maintains the counters STATE
# between iterations. Changing this state is a SIDE EFFECT
# if we jit it....

print("\njitted counter!\n")

counter_jitted = jax.jit(counter.count)  # this is cool, we can jit class methods!
counter.reset()

for __ in range(3):
    print(counter_jitted())

# this carefully what is happening here. JAX traces the function and compiles it,
# ignoring any side-effects. We init n at 0, it runs once when Jax traces, so iterate
# to n. but then it doesn't change again.

# The solution - explicit state
# ----------------------------------------------------------------------------------------------------------------------

CounterState = int

class CounterV2:

    def count(self, n: CounterState) -> Tuple[int, CounterState]:
        """
        You could just return n+1, but here we separate its role as the
        output and as the counter state for learning purposes
        """
        return n + 1, n + 1

    def reset(self) -> CounterState:  # so the variable becomes a type?
        return 0

counter = CounterV2()
state = counter.reset()

for __ in range(3):
    value, state = counter.count(state)
    print(value)

# So, we need to track the state explicitly in a variable that the function
# outputs. Now we have a pure functional programming function!

# A general strategy - check the docs to see common functional programing
# pattern to convert a class to stateless class. The theory is the same.

# Simple worked stratergy - linear regression
# ----------------------------------------------------------------------------------------------------------------------

# Note this example is pretty much identical to the one in getting_started_with_jax_numpy.py,
# but has some nice examples of using random key to init params and storing in a class

class Params(NamedTuple):  # Named Tuple: Intersectio of dictionary and key https://towardsdatascience.com/what-are-named-tuples-in-python-59dc7bd15680
                           # Access elements via name OR index! cool
    weight: jnp.ndarray
    bias: jnp.ndarray


def init_params(rng) -> Params:
    weights_key, bias_key = jax.random.split(rng)
    weight = jax.random.normal(weights_key, ())
    bias = jax.random.normal(bias_key, ())
    return Params(weight, bias)

def loss(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = params.weight * x + params.bias
    loss = jnp.mean((pred - y)**2)
    return loss

LEARNING_RATE = 0.005

@jax.jit
def update(params: Params, x: jnp.ndarray, y:jnp.ndarray) -> Params:
    grad = jax.grad(loss)(params, x, y)

    # If we were using Adam or another stateful optimizer, we would also
    # do something like:
    #   updates, new optimizer_state = optimizer(grad, optimizer_state)
    # and then use updates instead of grad to actually update the params.

    new_params = lambda params, grad: params - LEARNING_RATE * grad

    # so now we map over our NamedTuple (a pytree with leafs containing weight array
    # and bias array
    return jax.tree_map(new_params, params, grad)

# WE MANUALLY PIPE PARAMS IN AND OUT OF THE UPDATE FUNCTION

rng = jax.random.PRNGKey(42)

params = init_params(rng)

x_rng, noise_rng = jax.random.split(rng)

true_weights, true_bias = 2, -1

xs = jax.random.normal(x_rng, (128, 1))
noise = jax.random.normal(noise_rng, (128, 1)) * 0.5
ys = true_weights * xs + true_bias + noise

max_iter = 1000
for __ in range(max_iter):
    params = update(params, xs, ys)

plt.scatter(xs, ys)
plt.plot(xs, xs * params.weight + params.bias, "r")

plt.show()

# The strategy described above is how any (jitted) JAX program must handle state.!!!
# The details can become confusing in the case of many states. See https://jax.readthedocs.io/en/latest/jax-101/07-state.html