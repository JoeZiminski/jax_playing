import jax
# Now this... is frikkin.... AWESOME!!

# avoid manual vectorisation re-writes with JAX automatic vectorisation!?

# ... in retrospect, it is very nice for well-defined vectorizations, but
# not a golden bullet

# Manual Vectorisation
# ----------------------------------------------------------------------------------------------------------------------
import jax.numpy as jnp

# Let's say we have 1D convolution

x = jnp.arange(5)
w = jnp.array([2.0, 3.0, 4.0])

def convolve(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

print(f"convolution output: \n{convolve(x, w)}")

# and we have a load of 1d vectors and set of weights we want to
# apply (these are independent, repeated just for example).
# We can do this in loop

xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

def manually_batched_convolve(xs, ws):
    output = []
    for i in range(xs.ndim):
        output.append(convolve(xs[i], ws[i]))
    return jnp.stack(output)

print(f" manually batched convolution: \n{manually_batched_convolve(xs, ws)}")

# We could manually vectorise

def manually_vectorized_convolve(xs, ws):
    output = []
    for i in range(1, xs.shape[-1] - 1):  # this is only vectorised across the batch dimension not the convolution
        conv = jnp.sum(xs[:, i-1:i+2] * ws, axis=1)
        output.append(conv)
    return output

manually_vectorized_convolve(xs, ws)

# BUT WHY DO THAT WHEN WE CAN TAKE ADVANTAGE OF AUTOMATIC VECTORISATION
# with jax.vmap

# jax.vmap generates a vectorised implementation of a function automatically.
# It does this by tracing a function simila to JIT, 'and automatically adding
# batch axes at the begginign of each input'. IF THE BATCH DIMENSION IS NOT
# THE FIRST, YOU MAY USE IN_AXES AND OUT_AXES ARGUMENTS TO SPECIFY THE LOCATION OF
# BATCH DIMENSION INPUTS AND OUTPUTS.
# (this is key for ongoing PR https://github.com/pyro-ppl/numpyro/pull/1529)

auto_batch_convolve = jax.vmap(convolve)
auto_batch_convolve(xs, ws)

auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)  # we must transpose to keep input size the same!
wst = jnp.transpose(ws)

print(f"output_batched convolve: \n{auto_batch_convolve_v2(xst, wst)}")

auto_batch_convolve_v3 = jax.vmap(convolve, in_axes=1, out_axes=0)
print(f" mapping convolution along second axis, "
      f"out second axis on transposed xs,ws : \n{auto_batch_convolve_v3(xst, wst)}")

# jit and vmap are composable

jitted_batch_convolve = jax.jit(auto_batch_convolve)

print(f" mapping convol out along first axis and output on first: \n{jitted_batch_convolve(xs, ws)}")




