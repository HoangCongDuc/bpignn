import jax
from jax import vmap, lax
import jax.numpy as jnp
from jax.example_libraries import optimizers

import flax
from flax import linen as nn
import optax
from frozendict import frozendict

import numpyro
import numpyro.distributions as dist


from .model import get_layers


def initialize_mlp_prob(name, sizes, affine=[False], scale=1.0):
    """ Initialize the weights of all layers of a linear layer network """
    # Initialize a single layer with Gaussian weights -  helper function
    if len(affine) != len(sizes):
        affine = [affine[0]]*len(sizes)
    affine[-1] = True

    def initialize_layer(name_, m, n, affine=True, scale=1e-2):
        ws = numpyro.sample(f'{name_}_W', dist.Normal(scale=scale), sample_shape=(n, m))
#         bs = numpyro.sample(f'{name_}_b', dist.Normal(scale=(0. if affine else scale)), sample_shape=(n,))
        bs = numpyro.sample(f'{name_}_b', dist.Normal(scale=scale), sample_shape=(n,))
        return ws, bs
        
    return [initialize_layer(f'{name}_{i}', m, n, affine=a, scale=scale) 
            for i, (m, n, a) in enumerate(zip(sizes[:-1], sizes[1:], affine))]


def mlp_prob(name, in_, out_, hidden, nhidden, **kwargs):
    return initialize_mlp_prob(name, get_layers(in_, out_, hidden, nhidden), **kwargs)


def generate_HGNN_params_prob(Oh, Nei, Ef, Eei, dim, hidden, nhidden):
    
    fneke_params = initialize_mlp_prob('fneke', [Oh, Nei])
    fne_params = initialize_mlp_prob('fne', [Oh, Nei])

    fb_params = mlp_prob('fb', Ef, Eei, hidden, nhidden)
    fv_params = mlp_prob('fv', Nei + Eei, Nei, hidden, nhidden)
    fe_params = mlp_prob('fe', Nei, Eei, hidden, nhidden)

    ff1_params = mlp_prob('ff1', Eei, 1, hidden, nhidden)
    ff2_params = mlp_prob('ff2', Nei, 1, hidden, nhidden)
    ff3_params = mlp_prob('ff3', dim + Nei, 1, hidden, nhidden)
    ke_params = initialize_mlp_prob('ke', [1 + Nei, 10, 10, 1], affine=[True])

    Hparams = dict(
        fb=fb_params,
        fv=fv_params,
        fe=fe_params,
        ff1=ff1_params,
        ff2=ff2_params,
        ff3=ff3_params,
        fne=fne_params,
        fneke=fneke_params,
        ke=ke_params
    )
    
    return {"H": Hparams}


def generate_model(Oh, Nei, Ef, Eei, dim, hidden, nhidden, v_zdot_model):

    def model(Rs_, Vs_, Zs_dot_=None, norm_scale=1e-4, subsample_size=2000):
        p_ = generate_HGNN_params_prob(Oh, Nei, Ef, Eei, dim, hidden, nhidden)
        with numpyro.plate("data", len(Rs_), dim=-1, subsample_size=subsample_size) as ind:
            R = Rs_[ind]
            V = Vs_[ind]
            Zdot = None if (Zs_dot_ is None) else Zs_dot_[ind]
            pred = v_zdot_model(R, V, p_)
        numpyro.sample('zs_dot', dist.Normal(pred, scale=norm_scale), obs=Zdot)
        return pred

    return model


def construct_param_from_samples(samples, i):
    
    d = dict()
    p_names = set([n.split('_')[0] for n in samples.keys()])
    
    for n in p_names:
        k = 0
        l = []
        while f'{n}_{k}_W' in samples:
            ws = samples[f'{n}_{k}_W'][i]
            bs = samples[f'{n}_{k}_b'][i]
            l.append((ws, bs))
            k += 1
        d[n] = l
        
    return {'H': d}
