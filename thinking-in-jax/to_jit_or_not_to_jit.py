"""
By default JAX executes operations one at a time, in sequence.
Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once.
Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.

NOTE THIS: 'requires array shapes to be static & known at compile time.'

The fact that all JAX operations are expressed in terms of XLA allows
JAX to use the XLA compiler to execute blocks of code very efficiently.
    XLA (accelerated linear algebra) is a compiler-based linear algebra execution engine.
    It is the backend that powers machine learning frameworks such as TensorFlow and JAX at Google, on a variety of devices including CPUs, GPUs, and TPUs
    https://www.tensorflow.org/xla
"""
import numpy as np
import jax.numpy as jnp
import timeit
import time

def norm(X):
    X = X - X.mean(axis=0)
    return X / X.std(axis=0)

# JIT compiled version of this function can be created
from jax import jit
norm_compiled = jit(norm)

# This function returns the same results as the original, up to standard floating-point accuracy:
np.random.seed(1701)
X = jnp.array(np.random.rand(100000, 10))
np.allclose(norm(X), norm_compiled(X), atol=1e-6)

# Why isn't this much faster? is this what the docs are trying to show?
# This is similar results to ipython %timeit
# See getting_started_with_jax_numpy.py for notes on timing
print(f"Numpy: {timeit.timeit(lambda: norm(X).block_until_ready(), number=1000)}")
print(f"Jax: {timeit.timeit(lambda: norm_compiled(X).block_until_ready(), number=1000)}")

# That said, jax.jit does have limitations: in particular, it requires all arrays to have static shapes.
# That means that some JAX operations are incompatible with JIT compilation.
def get_negatives(x):
    return x[x < 0]

x = jnp.array(np.random.randn(10))
get_negatives(x)

print("A jit-able function must generate an array whose shape is KNOWN at compile time...\n")
try:
    jit(get_negatives)(x)
except IndexError as e:
    print(e)