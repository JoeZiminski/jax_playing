"""
https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html

JAX provides a NumPy-inspired interface for convenience.

Through duck-typing, JAX arrays can often be used as drop-in replacements of NumPy arrays.
Python’s duck-typing allows JAX arrays and NumPy arrays to be used interchangeably in many places.
    It is a term used in dynamic languages that do not have strong typing.
    The idea is that you don't need to specify a type in order to invoke an
    existing method on an object - if a method is defined on it, you can invoke it.

Unlike NumPy arrays, JAX arrays are always immutable.

Almost anything that can be done with numpy can be done with jax.numpy:
"""
import matplotlib.pyplot as plt
import numpy as np
show_plots = False

# Max a sine x cosine function in NUmPy
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np * 2 * np.pi) * np.cos(x_np * 2 * np.pi)

if show_plots:
    plt.plot(x_np, y_np)
    plt.show()

# Now do it in Jax
import jax.numpy as jnp

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp * 2 * jnp.pi) * jnp.cos(x_jnp * 2 * jnp.pi)

if show_plots:
    plt.plot(x_jnp, y_jnp)
    plt.show()

assert jnp.allclose(y_np, y_jnp, rtol=0, atol=1e-5)  # what is going on here? 1e-6 is 'standard precision' according
                                                     # to JIT or not to JIT section. See https://stackoverflow.com/questions/13542944/how-many-significant-digits-do-floats-and-doubles-have-in-java
                                                     # but 7 digits is 32-bit float. Aha! jax 32-bit by default, numpy is 64-bit.
                                                     # Oh yes, this is covered in https://jax.readthedocs.io/en/latest/faq.html
print("y_np and y_jnp are only the same to 5dp")

print(f"Numpy Type: {type(x_np)}")
print(f"Jax Type: {type(x_jnp)}\n")

jax_immutable = jnp.arange(10)

print("Trying to assign to jax array...")
try:
    jax_immutable[5] = 101
except TypeError as e:
    print(e)

# For updating individual elements, JAX provides an indexed update syntax that returns an updated copy:
changed_jax = jax_immutable.at[0].set(101)

print(f"Changed jax with .at[0].set(101): {changed_jax}\n")

# NumPy, lax & XLA: JAX API layering
# jax.numpy is a high-level wrapper that provides a familiar interface.
# jax.lax is a lower-level API that is stricter and often more powerful.
# All JAX operations are implemented in terms of operations in XLA – the Accelerated Linear Algebra compiler.

from jax import lax

print("Jax numpy will allow mixed types...")
jnp.add(1, 1.0)


print("lax will not...\n")
try:
    lax.add(1, 1.0)
except TypeError as e:
    print(e)

print("\nUnder the hood implementations in lax are much more general, take convolution")

x = jnp.arange(10)  # note output type is int
y = jnp.cos(x[:3])
y_conv_jnp = jnp.convolve(x, y)

y_conv_lax = lax.conv_general_dilated(x.reshape(1, 1, x.size).astype(float),   # note the reshaping
                                      y.reshape(1, 1, y.size),
                                      window_strides=(1,),
                                      padding=[(len(y) - 1, len(y) - 1)]
                                      )
"Standard numpy API for jnp, but requires reshape x and y as below for lax"

print(x.reshape(1, 1, x.size))
print(y.reshape(1, 1, y.size))
