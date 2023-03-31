import time
import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from frozendict import frozendict
from jax import vmap, lax
from jraph._src import graph as gn_graph
from jraph._src import utils


def get_zdot_lambda(N, Dim, hamiltonian, drag=None, constraints=None, external_force=None):
    dim = N*Dim
    I = jnp.eye(dim)
    J = jnp.zeros((2*dim, 2*dim))
    J = J.at[:dim, dim:].set(I)
    J = J.at[dim:, :dim].set(-I)

    J2 = jnp.zeros((2*dim, 2*dim))
    J2 = J2.at[:dim, :dim].set(I)
    J2 = J2.at[dim:, dim:].set(I)

    def dH_dz(x, p, params):
        dH_dx = jax.grad(hamiltonian, 0)(x, p, params)
        dH_dp = jax.grad(hamiltonian, 1)(x, p, params)
        return jnp.hstack([dH_dx.flatten(), dH_dp.flatten()])

    if drag is None:
        def drag(x, p, params):
            return 0.0

    def dD_dz(x, p, params):
        dD_dx = jax.grad(drag, 0)(x, p, params)
        dD_dp = jax.grad(drag, 1)(x, p, params)
        return jnp.hstack([dD_dx.flatten(), dD_dp.flatten()])

    if external_force is None:
        def external_force(x, p, params):
            return 0.0*p

    if constraints is None:
        def constraints(x, p, params):
            return jnp.zeros((1, 2*dim))

    def fn_zdot(x, p, params):
        dH = dH_dz(x, p, params)
        dD = J2 @ dD_dz(x, p, params)
        dD = - J @ dD
        F = jnp.hstack(
            [jnp.zeros(dim), external_force(x, p, params).flatten()])
        F = -J @ F
        S = dH + J2 @ dD + F
        A = constraints(x, p, params).reshape(-1, 2*dim)
        Aᵀ = A.T
        INV = jnp.linalg.pinv(A @ J @ Aᵀ)
        λ = -INV @ A @ J @ S
        zdot = J @ (S + Aᵀ @ λ)
        return zdot.reshape(2*N, Dim)

    def lambda_force(x, p, params):
        dH = dH_dz(x, p, params)
        dD = J2 @ dD_dz(x, p, params)
        dD = - J @ dD
        F = jnp.hstack(
            [jnp.zeros(dim), external_force(x, p, params).flatten()])
        F = -J @ F
        S = dH + J2 @ dD + F
        A = constraints(x, p, params).reshape(-1, 2*dim)
        Aᵀ = A.T
        INV = jnp.linalg.pinv(A @ J @ Aᵀ)
        λ = -INV @ A @ J @ S
        return (J @ Aᵀ @ λ).reshape(2*N, Dim)
    return fn_zdot, lambda_force
