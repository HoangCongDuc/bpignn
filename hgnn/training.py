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


def MSE(y_act, y_pred):
    return jnp.mean(jnp.square(y_pred - y_act))


def generate_loss_fn(v_zdot_model):

    @jax.jit
    def loss_fn_(params, Rs, Vs, Zs_dot):
        pred = v_zdot_model(Rs, Vs, params)
        return MSE(pred, Zs_dot)
    
    return loss_fn_


def generate_gloss(loss_fn):
    
    def gloss(*args):
        return jax.value_and_grad(loss_fn)(*args)
    
    return gloss


def generate_opt_update_wrapper(opt_update_):

    @jax.jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        grads_ = jax.tree_map(functools.partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)
    
    return opt_update


def generate_update_fn(gloss, opt_update, get_params):

    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value

    @jax.jit
    def step(i, ps, *args):
        return update(i, *ps, *args)
    
    return step


def batching(*args, size=None):
    L = len(args[0])
    if size != None:
        nbatches1 = int((L - 0.5) // size) + 1
        nbatches2 = max(1, nbatches1 - 1)
        size1 = int(L/nbatches1)
        size2 = int(L/nbatches2)
        if size1*nbatches1 > size2*nbatches2:
            size = size1
            nbatches = nbatches1
        else:
            size = size2
            nbatches = nbatches2
    else:
        nbatches = 1
        size = L

    newargs = []
    for arg in args:
        newargs += [jnp.array([arg[i*size:(i+1)*size]
                            for i in range(nbatches)])]
    return newargs
