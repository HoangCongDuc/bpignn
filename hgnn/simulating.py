import time
import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jax.experimental import ode
import numpy as np
from frozendict import frozendict
from jax import vmap, lax
from jraph._src import graph as gn_graph
from jraph._src import utils


def z0(x, p):
    return jnp.vstack([x, p])

def get_forward_sim(params=None, zdot_model=None, runs=10, stride=1000, dt=1e-5, tol=1.4e-7):
    
    def zdot_model_func(z, t, params):
        x, p = jnp.split(z, 2)
        return zdot_model(x, p, params)
    
    def fn(R, V):
        t = jnp.linspace(0.0, runs*stride*dt, runs*stride)
        _z_out = ode.odeint(zdot_model_func, z0(R, V), t, params, atol=tol, rtol=tol)
        return _z_out[0::stride]
    
    return fn


def get_forward_sim_noparam(zdot_model=None, runs=10, stride=1000, dt=1e-5, tol=1.4e-7):
    
    def zdot_model_func(z, t, params):
        x, p = jnp.split(z, 2)
        return zdot_model(x, p, params)
    
    def fn(R, V, params):
        t = jnp.linspace(0.0, runs*stride*dt, runs*stride)
        _z_out = ode.odeint(zdot_model_func, z0(R, V), t, params, atol=tol, rtol=tol)
        return _z_out[0::stride]
    
    return fn