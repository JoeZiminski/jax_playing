"""
Excellent overview

https://ericmjl.github.io/essays-on-data-science/machine-learning/differential-computing-jax/

What is differential computing?

I think at its core, we can think of differential computing as
"computing where derivatives (from calculus) are a first-class citizen".

This is analogous to probabilistic computing, in which probabilistic constructs
(such as probability distributions and MCMC samplers) are given "first class"
status in a language.

By "first class" status, I mean that the relevant computing
constructs are a well-developed category of constructs in the language, with
clearly defined interfaces.

With deep learning, the core technical problem that needs to be solved is o
ptimizing parameters of a model to minimize some loss function.
It's here where the full set of partial derivatives of the loss function
w.r.t. each parameter in the model can be automatically calculated using an
AD system, and these partial derivatives can be used to update their
model parameters in the direction that minimizes loss.

For example, AD is used in the Bayesian statistical modelling world.
Hamiltonian Monte Carlo samplers use AD to help the sampler program identify
the direction in which its next MCMC step should be taken.
"""
import jax
import numpy as np
import jax.numpy as jnp
from timeit import timeit
import matplotlib.pyplot as plt
import time

# Many common NumPy programs would run just as well in JAX if you substitute np for jnp.

# but

# 1) Jax can run on CPU, GPU or TPU. So already improvement if we run on GPU
# 2) Jax can be just-in-time (JIT) compiled under some restrictions - so huge speedup!
# timing Jax functions
# 3) Jax can also transform functions (particular emphasis on differentiation, see link above)

# (Technical detail: when a JAX function is called (including jnp.array creation),
# the corresponding operation is dispatched to an accelerator to be computed
# asynchronously when possible. The returned array is therefore not necessarily
# ‘filled in’ as soon as the function returns. Thus, if we don’t require the result immediately,
# the computation won’t block Python execution. Therefore, unless we block_until_ready or
# convert the array to a regular Python type, we will only time the dispatch, not the actual
# computation. See Asynchronous dispatch in the JAX docs.)

long_vector = jnp.arange(int(1e7))

print(timeit(lambda: np.dot(long_vector, long_vector), number=100))
print(timeit(lambda: jnp.dot(long_vector, long_vector).block_until_ready(), number=100))


# Grad
# ----------------------------------------------------------------------------------------------------------------------

def sum_of_squares(x):
    return jnp.sum(x**2)

def check_sum_of_squares_grad(x):
    return np.sum(2*x)

sum_of_squares_dx = jax.grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
print("\nGradients!\n")
print(f" sum of squares: {sum_of_squares(x)}")
print(f" sanity check sum of squared derivitive (summed): {check_sum_of_squares_grad(x)}")
print(f" jax grad(sum_of_squares)(x): {sum_of_squares_dx(x)}")  # not 100% sure why the output is not summed?

# You can think of jax.grad() as the derivative operator from vector calculus
# Taking the derivative of the loss function can be used to determine the direction
# which will decrease loss maximally. For example:

def sum_squared_error(x, y):
    return jnp.sum((x-y)**2)

sum_squared_error_dx = jax.grad(sum_squared_error)

y = jnp.asarray([1.1, 2.1, 3.1, 4.1])

print(f"\n sum of squared error: {sum_squared_error(x, y)}")
print(f" diff sum of squared error w.r.t x: {sum_squared_error_dx(x, y)}")

# To find the gradient with respect to a differnt argument (or several) (e.g. w.r.t. y),
# you can set argnums

# argnums (Union[int, Sequence[int]]) – Optional, integer or sequence of integers.
# Specifies which positional argument(s) to differentiate with respect to (default 0).
# if a tuple is specified, it is the dimensions to diff w.r.t
# otherwise, pass the dimension exactly

def sanity_check_grad(x, y):
    """(x-y)**2 = x^2 - 2xy + y^2"""
    return 2*(y - x)

print(f"\n sanity check gradient w.r.t. y: {sanity_check_grad(x, y)}")

gradient_wrt_y = jax.grad(sum_squared_error, argnums=1)
gradient_wrt_xy = jax.grad(sum_squared_error, argnums=(0, 1))
print(f" jax gradient w.r.t. y: {gradient_wrt_y(x, y)}")
print(f" jax gradient w.r.y x and y {gradient_wrt_xy(x, y)}")

# Pytrees (coming up soon) can be used to reduce the argument load when we
# have a lot of parameters.

# Handy Gradient Fucntions
# ----------------------------------------------------------------------------------------------------------------------

# There is a nice convenient function jax.value_and_grad() which returns both the
# value and the graduient, handy!
# jax.value_and_grad(f)(x) == (f(x), jax.grad(f)(x))

# If we want to return Auxillary results (i.e. intermediate), we can use
# has_aux argument.

# Differences between Numpy and Jax
# ----------------------------------------------------------------------------------------------------------------------

# Jax follows a functional programming ethos,
# https://codeburst.io/a-beginner-friendly-intro-to-functional-programming-4f69aa109569

# A key feature of functional programming is that functions have no side effects, e.g.
# they do not do anything outside of what they return, they have no effects except
# for returning their value.

# This is why jax arrays cannot be modified in place (see
# /thinking-in-jax/jax_vs_numpy.py. The alternative is to use
# jax.at[idx].set(num) which returns a copy. This may seem
# wasteful but compiler optimisations usually address it.

# Your first Jax training loop
# ----------------------------------------------------------------------------------------------------------------------

xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise

plt.scatter(xs, ys)
plt.show()

# We use a single array theta = [m, c], to house the parameters
theta = [3, -1]

def model(theta, x):
    """computers mx + c on a batch of inputs"""
    m, c, d = theta
    return m * x + c + d

def loss_function(theta, x, y):
    return jnp.mean( (y - model(theta, x))**2 )

def update(theta, x, y, lr=0.1):
    """ small step in the direction of the steepest decent"""
    gradient = jax.grad(loss_function)(theta, x, y)
    new_theta = theta - lr * gradient  # gradient = (dθ1, dθ2) i.e. size = θ.size (I assume order). 
    return new_theta

# In JAX, it’s common to define an update() function that is called every step,
# taking the current parameters as input and returning the new parameters.

theta = jnp.array([-1.0, 1.0, 2.0])  # NOTE: this will error if int. TypeError: grad requires real- or complex-valued inputs
                                     # (input dtype that is a sub-dtype of np.inexact), but got int32.
num_iter = 1000

t = time.time()
for __ in range(num_iter):
    theta = update(theta, xs, ys)
print(f"non-jit estimation: {time.time() - t}")

plt.scatter(xs, ys)
plt.plot(xs, model(theta, xs))
plt.title("Model")
plt.show()

# Test time with JIT
# ----------------------------------------------------------------------------------------------------------------------

xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))

@jax.jit
def update_jit(theta, x, y, lr=0.1):
    """ small step in the direction of the steepest decent"""
    gradient = jax.grad(loss_function)(theta, x, y)
    new_theta = theta - lr * gradient  # gradient = (dθ1, dθ2) i.e. size = θ.size (I assume order).
    return new_theta


t = time.time()
for __ in range(num_iter):
    theta = update_jit(theta, xs, ys)
print(f"jitted estimation: {time.time() - t}")  # frik me! 3.77 vs. 0.04 s!
