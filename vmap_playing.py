import jax.numpy as jnp
from jax import vmap
import numpy as np

x = jnp.array([1.0, 2.0, 3])
y = jnp.array([4.0, 5.0, 6.0])

def f(x, y):
    return jnp.dot(x, y)

print(f(x, y))
map_to_first = vmap(f, (0, None), 0)(x, y)

# this will the first dimension over the len(first_dimension) and run in parallel
# e.g. [1, 2, 3] becomes [1], [2], [3] and the output
# [[1.y],
#  [2.y],
#  3.y]]

# One Dimensional ------------------------------------------------------------------------------------------------------

print("One Dimensional\n")

print(f"\nMapped along first {map_to_first}")

print(f"\nCustom map along first: {np.array([np.dot(x[0], y), np.dot(x[1], y), np.dot(x[2], y)])}")

map_to_second = vmap(f, (None, 0), 0)(x, y)
print(f"\nMapped along second: {map_to_second}")
print(f"\nCustom map along first: {np.array([np.dot(x, y[0]), np.dot(x, y[1]), np.dot(x, y[2])])}")

# oaky this makes sense, we just split both and do element by element
map_both = vmap(f, (0, 0), 0)(x, y)
print(f"\nMap both {map_both}")

# Two Dimensional ------------------------------------------------------------------------------------------------------

print("\n Lets map a (4,) array by a 4x3 array \n")

x = jnp.array([1.0, 2.0, 3.0, 4.0])

y = jnp.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0], [14.0, 15.0, 16.0]])

# this will map the first dimension as 1, 2, 3 4 separately and
# perform separate dot products with y

print(f"\nvmapping inputs (0, None) will vectorise over the 1st argument, with axis 4,\n "
      f"resulting in 4 separate dot prodcuts (scalar vs. matrix): "
      f"{vmap(f, (0, None), 0)(x, y)}")

print(f"\n recapitulating in numpy:\n {np.array([[np.dot(x[0], y)], [np.dot(x[1], y)], [np.dot(x[2], y)], [np.dot(x[3], y)]])}")

# Now lets just map over the first axis of y (i.e. 4). This will split
# y into 4 separate arrays and map over them with the separate x

print(f"\nNow mapping over both arguments using the first dim (4 and 4 respectively.\n"
      f"This will use 4 different scalars (x) on 4 different rows (y) and output result per row\n"
      f"{vmap(f, (0, 0))(x, y)}")

print(f"\n recapitulating in numpy:\n {np.array([np.dot(x[0], y[0]), np.dot(x[1], y[1]), np.dot(x[2], y[2]), np.dot(x[3], y[3])])}")

print(f"\nby default this maps the OUTPUT (above) along the first axis (4 rows). "
      f"But we can make it map along the second axis (3 rows) "
      f"{vmap(f, (0, 0), 1)(x, y)}")