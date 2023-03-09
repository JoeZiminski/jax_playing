# JAX allows us to transform Python functions.
# IMPORTANT: This is done by first converting the Python function into a simple intermediate language called jaxpr

# see thinking-in-jax/static_vs_traced_operations.py for details on make_jaxpr
# also https://jax.readthedocs.io/en/latest/jaxpr.html

# CRITICAL: jaxpr does not check for functional side effects - it assumes pure functions
# if functions are not pure, they can still run and MAY CAUSE BUGS / ERRORS.
# Due to tracing, side effects will only be run once, during generation of jaxpr.

# How Tracing Works CRITICAL
# 1) on a single function call, jax will wrap ARGUMENTS with tracer object.
# 2) jax records all JAX operations performed on them during the function call
#    the output of this is jaxpr
# 3) jax uses jaxpr to trace to reconstruct the entire function. Critically,
#    side effects are not traced and so are not applied in compiled code! Will
#    only occur once, during tracing!!

# also, recall conditional behaviour!

# Jitting, a nice example with SELU

# ----------------------------------------------------------------------------------------------------------------------
# Scaled Exponential Linear Unit

# Activation function: map Y = (weights * input) + bias, to the range [0, 1]
# https://www.superannotate.com/blog/activation-functions-in-neural-networks

# note non-binary activation functions mean the neuron can have non-binary firing state
# "So unlike biological neurons, artificial neurons don’t just “fire”: they send continuous values instead of binary signals."
# https://towardsdatascience.com/the-differences-between-artificial-and-biological-neural-networks-a8b46db828b7

# SULE as an activation function: https://towardsdatascience.com/gentle-introduction-to-selus-b19943068cd9
import jax
import jax.numpy as jnp
from timeit import timeit

def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)  # cool! didn't know this, operate on the chosen

selu_jit = jax.jit(selu)

x = jnp.arange(1000000)

# Run once to allow Jax to do it's tracing
selu_jit(x).block_until_ready()

print(f"SELU: {timeit(lambda: selu(x).block_until_ready(), number=1000)}")
print(f"SELU jit: {timeit(lambda: selu_jit(x).block_until_ready(), number=1000)}")

# NOTE: didn't see this in previous examples:
# (If we didn’t include the warm-up call separately, everything would still work,
# but then the compilation time would be included in the benchmark. It would still be faster,
# because we run many loops in the benchmark, but it wouldn’t be a fair comparison.)

# Why can’t we just JIT everything?
# ----------------------------------------------------------------------------------------------------------------------

# JIT doesn't work with conditionals on x
# recall all Jax tracers save is the shape and dtype. Thus we can only condition
# on these values, not the actual values.

# The best way to handle this is to avoid conditionals - keep to pure functional programming.
# There are jax control flow operators, but they should be avoided if possible.
# see https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html# for details.


# Caching
# ----------------------------------------------------------------------------------------------------------------------

# JAX will run on first function call and CACHE COMPILED CODE for the dtype, size of x to avoid re-use.
# If x size or dtype chanegs, JAX will need to compile again.
# This is related to the static_argnums value in @partial(jax.jit, static_argnames)
# see static_vs_traced_oeprations.py
