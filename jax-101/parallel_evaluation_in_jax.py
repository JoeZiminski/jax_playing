"""
single-program, multiple data (SPMD) is where the same algorithm is run on
different batches of data, in parallel on separate devices

Conceptually, this is not very different to vectorisation, where the same oeprations
occur in parallel in differnt parts of memory on the same device.

Vectorisation is supposed in jax with jax.vmap

Jax supports device parallelalism similarly, with jax.pmap
"""

# Won't complete this now as not totally relevant (AFAIK) but the take home message is:

# use jax.pamp instead of jax.vmap to map over multiple deices (e.g. tensor-processing units, TPUs)

# https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html