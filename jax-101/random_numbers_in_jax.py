"""
This is one area where JAX divergences from Numpy.

pseudorandom number generators are determined by their initial
value, the seed. Each step of random number generation depends on a
hidden state that is carried over from generation of one sample to the next
"""
import numpy as np
from jax import random

def print_truncated_random_state():
    full_random_state = np.random.get_state()  # "Return a tuple representing the internal state of the generator."
                                               # https://numpy.org/doc/stable/reference/random/generated/numpy.random.set_state.html#numpy.random.set_state
    print(str(full_random_state)[:460])

# the state is updated by each call to a random function

np.random.seed(0)

print_truncated_random_state()

__ = np.random.uniform()

print_truncated_random_state()

__ = np.random.normal()  # why doesn't second call update? https://en.wikipedia.org/wiki/Mersenne_Twister for details

print_truncated_random_state()

# Random Numbers in Jax
# ----------------------------------------------------------------------------------------------------------------------

# Numpy's random number generator algorithm is difficult to implement to be JAX compliant. This is due to the
# global state

# consider:

np.random.seed(0)

def bar():
    return np.random.uniform()

def baz():
    return np.random.uniform()

def foo():
    return bar() + 2 * baz()

print(foo())

# Note, numpy provides sequential equivalence e.g. if we
# take random uniform vector length N vs. N sequential samples from random uniform calls,
# the two output vectors should be the same.,

# Why is this not Jax compatible? Well, bar() and baz() do not depend on eachother
# and so their order of execution should be viable to be changed for optimisation (e.g. parallelisation).
# for Python / Numpy, the order of execution is standard. However, due to the random state
# this code will not be reproducible unless the order is maintained!

# In JAX, 'RANDOM FUNCTIONS EXPLICITLY CONSUME THE STATE'
# ----------------------------------------------------------------------------------------------------------------------

# in Jax, we must use Key which are similar to seeds), to pass to random number generator each
# time we want to use it. NEVER RE-USE KEYS (feeding the same key to random functions can result in correlated outputs)

key = random.PRNGKey(42)
print(key)

new_key, sub_key = random.split(key)

# use the sub_key to generate the random sample,
# and new key if we want to create another key.
# Discard the old key and sub-key after use!

del key
sample = random.normal(sub_key)
del sub_key

new_key2, sub_key2 = random.split(new_key)  # ....

# It doesn’t matter which part of the output of split(key)
# we call key, and which we call subkey. They are all
# pseudorandom numbers with equal status. The reason we
# use the key/subkey convention is to keep track of how
# they’re consumed down the road.

# NOTE 1: that split can create many keys are once:

key, *nintey_nine_other_keys = random.split(new_key, num=100)

# NOTE 2: sequential equivilence is not ensured

key = random.PRNGKey(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(sub_key) for sub_key in subkeys])
print(f"random samples in sequence {sequence}")

key = random.PRNGKey(42)
all_at_once = random.normal(key, shape=(3,))
print(f"random samples all at once: {all_at_once}")













