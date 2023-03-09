"""
Pytrees - note, this is critical for understanding

Pytrees refers too.... nested data structures! :) :( :)

So at it's heart, ML (at least in this introduction) is
really the steps of

1) define a model that takes some params (be it regression beta, neural net weights and biases)
2) define a loss function model(params, x) - y
3) Take the gradient of the loss function evaluated at params and x w.r.t params
   shift the params slightly in the negative gradient direction i.e. minimize the loss function
4) init some params
5) iteratively update the params using the gradient decent method
6) done!

"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Iterable

show_plots = False

# We have branches of pytrees (i.e. containers),
# a leaf is anything that is not a pytree.
# There is no need for recursion to define a pytree,
# just a stack of containers.
# If nested, container types do not need to be matched.



example_trees = [
    [1, "a", object()],
    (1, (2, 3), ()),
    [1, {"k1": 2, "k2": (3, 4)}, 5],
    {"a": 2, "b": (2, 3)},
    jnp.array([1, 2, 3]),
]

# This is nice - lets see how many leaves we have
for pytree in example_trees:
    leaves = jax.tree_util.tree_leaves(pytree)
    print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

print("\nor, printing the leaves all in one!:\n")
print(jax.tree_util.tree_leaves(example_trees))

# Common Pytree Functions
# ----------------------------------------------------------------------------------------------------------------------

# jax.tree.map works like pythons map on entire pytrees

print("mapping a function to pytrees")
trees = [
    [1, 2, 3],
    [1, [2, 3, 4]],
    {"a": 1, "b": 2, "c": 3}
]
print(jax.tree_map(lambda x: x * 2, trees))  # hell yea, that's cool

print("\nit also works with multiple arguments")
print(jax.tree_map(lambda x, y: x + y, trees, trees)) # hell, YEA!

# Example: ML learning parameters for a multi-layer perceptron

def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
        dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
             biases=np.ones(shape=(n_out,)))
    )
    return params
params = init_mlp_params([1, 128, 128, 1])

# jax.tree_map checks the shapes of our parameters contained
# in the list of dicts!

# [{'biases': (128,), 'weights': (1, 128)},
#  {'biases': (128,), 'weights': (128, 128)},
#  {'biases': (1,), 'weights': (128, 1)}]

print(f"\nList of dicts \n{jax.tree_map(lambda x: x.shape, params)}")

# Training

def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu( x @ layer["weights"] + layer["biases"])  # i.e. rectified activation function on weights * input + bias
    return x @ last["weights"] + last["biases"]

def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)  # so just like a regression, now we optimise the loss of the net weights and biases

LEARNING_RATE = 0.0001  # this is interesting to play with

@jax.jit
def update(params, x, y):
    # 1) take the model with the different params
    # 2) diff w.r.t. params
    # 3) move in the direction of negative gradient
    grads = jax.grad(loss_fn)(params, x, y)  # why do we need to init this on each function call?

    # NOTE: grads is a pytree (dict) of weights (128) and biases (120). This is because
    # we take the gradient function of our loss. Then we feed into it all our weights and biases,
    # returning the grad. We have 3 outputs, one w.r.t each layer. So this is the gradient
    # out of loss function w.r.t each layer. The weights and biases contain the derivatives
    # of the loss function in each direction (?)

    # This is useful because we can apply the stochastic gradient decent update using
    # tree utilities
    descent_function = lambda params, grads: params - LEARNING_RATE * grads

    return jax.tree_map(descent_function, params, grads)  # so now we return each param with a small step
                                                          # in the negative gradient direction

xs = np.random.normal(size=(128, 1))
ys = xs ** 2

max_iter = 1000
for __ in range(max_iter):
    params = update(params, xs, ys)
#    print(loss_fn(params, xs, ys))

if show_plots:
    plt.scatter(xs, ys, label="original data")
    plt.scatter(xs, forward(params, xs), label="model prediction")
    plt.legend()
    plt.show()

# Custom PyTree Nodes
# ----------------------------------------------------------------------------------------------------------------------

# IF YOU DEFINE YOUR OWN CONTAINER CLASS, IT WILL BE CONSIDERED A LEAF (i.. not a pytree).
# as such, we need to register our container with jax by telling it how to flatten and unflatten it:

class MyContainer:
    def __init__(self, name: str, a: int, b: int, c: int):
        self.name = name
        self.a = a
        self.b = b
        self.c = c

jax.tree_util.tree_leaves([
    MyContainer("Alice", 1, 2, 3),  # MyContainer is considered a leaf
    MyContainer("Bob", 4, 5, 6)
])

def flatten_MyContainer(container) -> Tuple[Iterable[int], str]:
    """
    Returns an iterable over container contents, and aux data.
    """
    flat_contents = [container.a, container.b, container.c]

    # we don't want the name to appear as a child, so it is auxiliary data.
    # auxiliary data is usually a description of the structure of a node,
    # e.g. the keys of a dict -- anything that ISN'T THE NODES CHILDREN
    aux_data = container.name

    return flat_contents, aux_data

def unflatten_MyContainer(
        aux_data: str, flat_contents: Iterable[int]) -> MyContainer:
    """
    Converts aux data and the flat contents into a MyContainer
    """
    return MyContainer(aux_data, *flat_contents)

# See the online version for example where Python in-build
# methods can be Jax-compliant, but I see no reason why not
# to explicitly define containers, unless it becomes cumbersome.

jax.tree_util.register_pytree_node(MyContainer,
                                   flatten_MyContainer,
                                   unflatten_MyContainer)

new_container_list = [
    MyContainer("Alice", 1, 2, 3),
    MyContainer("Bob", 4, 5, 6)
]
jax.tree_util.tree_leaves(new_container_list)

new_first_contents = jax.tree_util.tree_map(lambda x: x * 2, new_container_list)[0].a
print(f"new first contents after mapping (I guess we need to define repr as "
      f"it doesnt print full class to console): {new_first_contents}")

# Common pytree cotchas and patterns
# ----------------------------------------------------------------------------------------------------------------------

# Gotchas

# Mistaking Nodes for leaves - accidently introducting three nodes instead of leaves:

a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]

# Try to make another tree with ones instead of zeros
shapes = jax.tree_map(lambda x: x.shape, a_tree)

print(f"\nMistaking nodes for leaves, shapes: \n{shapes}\n")

print(f"\nMistaking nodes for leaves:\n {jax.tree_map(jnp.ones, shapes)}")

# see this prints ones of length [2, 3, 3, 4] NOT [(2,3), (3,4)]
# this is because the shape of an array is a TUPLE, with is a pytree node NOT a leave
# its elements are leaves. So map, whihc oeprates on leaves, is called on the x.shape
# elements NOT x.shape

# however, np.array or jnp array are LEAVES and so if we convert the shape to an array it will work

shapes = jax.tree_map(lambda x: np.array(x.shape), a_tree)
print(f"\nWorking nodes for leaves:\n {jax.tree_map(jnp.ones, shapes)}")

# Gotchas - NONE is a node without children, not a leaf!

# Gotchas - Jax also allows transposing pytrees (turn a list of trees into a tree of lists).


def tree_transpose(list_of_trees):
  """
  Convert a list of trees of identical structure into a single tree of lists.
  """
  return jax.tree_map(lambda *xs: list(xs), *list_of_trees)

# Convert a dataset from row-major to column-major:
episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
tree_transpose(episode_steps)










