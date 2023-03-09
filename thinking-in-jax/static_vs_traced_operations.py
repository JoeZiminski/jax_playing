"""
JIT mechanics: tracing and static variables

JIT and other JAX transforms work by tracing a function to determine
its effect on inputs of a specific shape and type.

Variables that you don’t want to be traced can be marked as static
"""
import numpy as np
from jax import jit, make_jaxpr
from jax.errors import ConcretizationTypeError
import jax.numpy as jnp

@jit
def f(x, y):
    print("Running f():")
    print(f" x = {x}")
    print(f" y = {y}")
    result = jnp.dot(x + 1, y + 1)
    print(f" result = {result}")
    return result

x = np.random.randn(3, 4)
y = np.random.randn(4)
print(f(x, y))

# Everything executes, but what is printed are the traced values.
# These encode the SHAPE and DTYPE of arrays (agnostic to values).

# IMPORTANT: This recorded sequence of computations can then be efficiently
# applied within XLA to new inputs with the same shape and dtype, without
# having to re-execute the Python code.

# When we call the compiled function again on matching inputs, no-recomputation
# is required and the result is computed in compiled XLA rather than in Python.
print(f(x, y))  # wow, nothing is printed now because it is running straight
                # from the compiled code!?!?!

x2 = np.random.rand(3, 4)
y2 = np.random.rand(4)
print(f(x2, y2))

print("\nThis is really cool, we can see the extracted sequence of operations\n")
print(make_jaxpr(f)(x, y))

print("\nTRACING MEANS FLOW CONTROL STATEMENTS IN "
      "THE FUNCTION CANNOT DEPEND ON TRACED VALUES !!\n")

@jit
def f(x, neg):
    return -x if neg else x

try:
    f(1, True)
except ConcretizationTypeError as e:
    print(e)

# If there are variables that you would not like to be traced,
# they can be marked as static for the purposes of JIT compilation:
# This presumably has important implications as will break
# trace flow if interspersed with non-trace items

from functools import partial
    # Return a new partial object which when called will behave like func
    # called with the positional arguments args and keyword arguments keywords.
    # If more arguments are supplied to the call, they are appended to args.


@partial(jit, static_argnums=(1,))
def f(x, neg):
    return -x if neg else x

f(1, True)

"""
Understanding which values and operations will be static and which will be traced is a key part of using jax.jit effectively.

Just as values can be either static or traced, operations can be static or traced.

KEY TO UNDERSTANDING:
Static operations are evaluated at compile-time in Python; traced operations are compiled & evaluated at run-time in XLA.

KEY TO IMPLEMENTATION
Use numpy for operations that you want to be static; use jax.numpy for operations that you want to be traced.
"""

# This distinction between static and traced values makes it important to
# think about how to keep a static value static. Consider this function:

try:
    @jit
    def f(x):
        return x.reshape(jnp.array(x.shape).prod())

except ConcretizationTypeError as e:
    print(e)

# This fails because a tracer was found instead of a 1D sequence of concrete
# values of integer type. Let's add some print statements to the function
# to understand why this is happening:

@jit
def f(x):
    print(f"1. x = {x}")
    print(f"2. x.shape = {x.shape}")
    print(f"3. jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
    # comment this out to avoid the error:
    # return x.reshape(jnp.array(x.shape).prod())

f(x)

# although x is a traced value, x.shape is a static value. When tracing occurs,
# jnp.array() turns this into a traced value.  However,
# FUNCTIONS REQUIRE STATIC INPUTS. ARRAY SHAPES MUST BE STATIC.

# a useful pattern is to use numpy for operations that should be static
# (i.e. done at compile-time), and use jax.numpy for operations that should be traced
# (i.e.compiled and executed at run-time). For this function, it might
# look like this:

# THINK:
# STATIC - DONE AT COMPILE TIM -  NUMPY
# TRACED - COMPILED AND EXECUTED AT RUN-TIME - JAX JNP


















