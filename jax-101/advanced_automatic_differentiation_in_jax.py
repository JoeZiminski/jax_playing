"""
Advanced Automatic Differentiation in JAX

https://www.youtube.com/watch?v=wG_nF1awSSY
https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

see also notes on getting_started_with_jax_numpy.py

# Notes on the video:

Pretty interesting stuff!

We need derivatives for gradient based optimisation
Automatic differentiation allows us to efficiently compute
derivatives. Other methods:
- manual differentation: can do but long, we want to automate
- numerical methods: error prone due to truncation / floating-point error https://www.uio.no/studier/emner/matnat/math/MAT-INF1100/h10/kompendiet/kap11.pdf
  event worse, is O(n)
- symbolic methods - terms quicklu blow up (e.g. product rule exansion)

auto-diff has the same accuracy as symbolic methods. auto-diff works by
decomposing expressions into primitives of which we know the derivatives,
and using the chain rule to calculate them.

Forward mode - replace values vi with tuple (vi, deriv(vi))
                e.g. [sin(x1/x2)-e^{x2}] can be decomposed to x1 = v0, x2 = v1
                    v2 = v0/v1, sin(v2), exp(v1). This is similar to dummy-variable
                    scheme for differential equations
                now we have these primitives, we also track the flow through the function
                at the same time,w e compute the tangent values also tracking flow through
                the function. We can get partial derivatives for each input in this manner.
                Each pass of forward-mode auto-diff produces a column in the jacobian of the system
                (so we need repeat passes per input variable partial derivative)
                more details on initalisation in video
                ideal when n << m

Implementation - operator overloading. We can overload primite operators (e.g. +)
                 to take values and derivatives.
                 course code transformation - input source code is transformed
                                              to calculate coded values and implicit
                                              derivatives at each step.

Reverse mode - propagate derivatves backwards from output.
              1) forward pass through function
              2) store dependencies of the expression tree in memory
              3) conmpute partial derivatives w.r.t intermediate variables (adjoints)
              ' the gradient is computed with one reverse pass'
              ' back propagation may be veiwed as a special case of reverse-mode auto-diff
"""
import jax
import jax.numpy as jnp
import numpy as np

# Higher order derivatives are trivially computed. I think this is okay, we have two variables
# but each is 1d and so it is just implicit differentiation not multivariable calculus (the JAX
# example uses 1 input)
f = lambda x, y: y * jnp.sin(x)**2 - 101 * y/4 * jnp.exp(-x/y)

dfdx = jax.grad(f)
d2f2dx2 = jax.grad(dfdx)
d3f2dx3 = jax.grad(d2f2dx2)
d4f2dx4 = jax.grad(d3f2dx3)
d5f2dx5 = jax.grad(d4f2dx4)

x = 1.01
y = 2.02

print(f"first derivative at x, y: {dfdx(x, y)}")
print(f"second derivative at x, y: {d2f2dx2(x, y)}")
print(f"fifth! derivative  at x, y: {d5f2dx5(x, y)}")

# Multivariate Derivatives
# ----------------------------------------------------------------------------------------------------------------------

# for partial derivatives of multivariate functions, we need to use different functions
# like jacobian is first partial derivative matrix, hessian is second partial derivative matrix

def f(x):
    return jnp.dot(x, x)

multivariate_output = jax.jacfwd(jax.grad(f))(jnp.array([1.0, 2.0, 3.0]))
print(f"\njacfwd output:\n {multivariate_output}")

# The docs cover the following use cases, not currently relevant:
# Stopping gradients
# Straight-through estimator using stop_gradient
# Per-example gradients