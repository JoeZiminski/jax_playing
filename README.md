This repo contains walkthrough and note for the [JAX
documentation](https://jax.readthedocs.io/en/latest/), in particular
"How to think in Jax" and "Tutorial: Jax 101".

In general the docs are very nice. See code for additional notes.
In general some key points

# Jax

## Key features of Jax
- jax can run on CPU, GPU, TPU (tensor processing unit)
- jax is jittable
- jax is supreme auto-differentiator (jax.grad() a function). Maybes computing derives up to arbitary order extremely convenient
- jax is google, based on XLA that is google compile optimiser build for tensorflow (that is google)
- because of this, jax is super super fast!

## Key restrictions of Jax

1) Jax works by running through all functions once to create a jaxpr (jax expression) that
tracks the variable through the function flow. Based on this code is compile and the compiled
version is run. Only the data type and shape are traced - this is agnostic to _values_ of input.

    KEY: This assumes [funnctionally pure code]( https://codeburst.io/a-beginner-friendly-intro-to-functional-programming-4f69aa109569) i.e.
    all changes to variables occur in function output. Any side-effects (i.e. effects that
    occur not in function outputs) are executed once during tracing, then ignored 
    as they are not captured by tracing and so do not appear in the compiled code that is run
    
    Note because input values are not captured in the tracing, **conditions on the input values cannot be used**

IMPORTANT: Jax won't necessarily complain if you provide functionally impure code, and could lead
to hidden bugs. As such, think hard and test!

2) Pytrees are jax way of handling nested structures (e.g. list of dicts). It is pretty neat and provides
intuitive API to handle this.

3) Random number generation is handled with keys as RNG generators depend on global state variable
and known order of expression execution. The use is fairly intuitive, just need to remember to do it.

4) state variables can never be implit e.g. held as a class variable and updated output of
function output. All state variables must be handled explitily (i.e. input into and out of a function).

5) Jax has two API, numpy and lax. Lax is for lower-level use.

# Machine Learning

These docs are also a nice intro to machine learning. 

1) define a model that takes some params (be it regression beta, neural net weights and biases)
2) define a loss function model(params, x) - y
3) Take the gradient of the loss function evaluated at params and x w.r.t params
   shift the params slightly in the negative gradient direction i.e. minimize the loss function
4) init some params
5) iteratively update the params using the gradient decent method
6) done!

Note that nodes in neuronal nets are typically not binary!

for each neuron, as expected:
    `Y = weights * inputs + bias`

but the 'activation function' mapping Y to the range [0, 1] is
typically not binary. See RELU, SELU, sigmoud.