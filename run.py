from functools import partial
import os
import pickle as pkl
from collections.abc import MutableMapping
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Union
import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import tqdm
from IPython.display import HTML

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
from numpyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO

import jraph
from jraph._src import graph as gn_graph
from jraph._src import utils

print(f'Jax: CPUs={jax.local_device_count("cpu")} - GPUs={jax.local_device_count("gpu")}')

from hgnn.noisify import add_noise_and_truncate
from hgnn.model import *
from hgnn.model_vi import *
from hgnn.hamiltonian import *
from hgnn.training import *
from hgnn.simulating import *

import argparse

parser = argparse.ArgumentParser(description='GNN_dropout')
parser.add_argument('--exp', type=str, default='pendulum-n2',  
                    help='experiment name')
parser.add_argument('--method', type=str, default='dropout', 
                    help='Use dropout, vi, or none')
parser.add_argument('--dropout_rate', type=float, default=0.5, 
                    help='dropout rate. Not used if method is vi')
parser.add_argument('--tol', type=float, default=1e-5, 
                    help='simulation tolerance')
parser.add_argument('--test_count', type=int, default=10, 
                    help='number of test cases to run')
parser.add_argument('--runs', type=int, default=100, 
                    help='number of simulation runs')
parser.add_argument('--add_noise', type=bool, default=True, 
                    help='whether to add noise in simulation')
# parser.add_argument('--save_path', type=str, default='denoised/',  
#                     help='folder containing denoised results')

# parser.add_argument('--beta', type=int, default=10, metavar='N',
#                     help='neighbours coupling constant')
# parser.add_argument('--std_const', type=int, default=1, metavar='N',
#                     help='coefficient for std dependence on r')

args = parser.parse_args()

exp = args.exp
method = args.method 
dropout_rate = args.dropout_rate
add_noise = args.add_noise

# setting dropout rate and save dir
if method == 'dropout':
    dropout_rate = args.dropout_rate
    save_dir = f'./results/{exp}/{method}_{dropout_rate}/'
else:
    args.dropout_rate = 0.
    dropout_rate = 0.
    save_dir = f'./results/{exp}/{method}/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


noise_scale = 0.01
truncate_decimal = 2

num_samples = 100

# we won't use all tests
test_count = args.test_count

# Running for more periods than provided dataset
runs = args.runs

# Setting simulation tolerance parameter
tol = args.tol

print('Args =', vars(args))
print('Out folder =', save_dir)

# Loading files
Zs_train = jnp.load(f'./data/{exp}/Zs_train.npy')
Zs_dot_train = jnp.load(f'./data/{exp}/Zs_dot_train.npy')
Zs_test = jnp.load(f'./data/{exp}/Zs_test.npy')
Zs_dot_test = jnp.load(f'./data/{exp}/Zs_dot_test.npy')

Zs_train, Zs_dot_train = add_noise_and_truncate(Zs_train, Zs_dot_train, 
                                                scale=noise_scale, 
                                                decimals=truncate_decimal)

N2, dim = Zs_train.shape[-2:]
N = N2 // 2
species = jnp.zeros(N, dtype=int)
masses = jnp.ones(N)

Zs = Zs_train.reshape(-1, N2, dim)
Zs_dot = Zs_dot_train.reshape(-1, N2, dim)

Zst = Zs_test.reshape(-1, N2, dim)
Zst_dot = Zs_dot_test.reshape(-1, N2, dim)

with open(f'./data/{exp}/param.json', 'r') as f:
    d = json.load(f)
    stride = d['stride']
    dt = d['dt']
    lr = d['lr']
    batch_size = d['batch_size']
    epochs = d['epochs']


if 'pendulum' in exp:

    def pendulum_connections(P):
        return (jnp.array([i for i in range(P-1)] + [i for i in range(1, P)], dtype=int),
                jnp.array([i for i in range(1, P)] + [i for i in range(P-1)], dtype=int))

    def edge_order(P):
        N = (P-1)
        return jnp.array(jnp.hstack([jnp.array(range(N, 2*N)), jnp.array(range(N))]), dtype=int)

    senders, receivers = pendulum_connections(N)
    eorder = edge_order(N)
    
elif 'nbody' in exp:
    
    def get_fully_connected_senders_and_receivers(num_particles: int, self_edges: bool = False):
        """Returns senders and receivers for fully connected particles."""
        particle_indices = np.arange(num_particles)
        senders, receivers = np.meshgrid(particle_indices, particle_indices)
        senders, receivers = senders.flatten(), receivers.flatten()
        if not self_edges:
            mask = senders != receivers
            senders, receivers = senders[mask], receivers[mask]
        return senders, receivers

    def get_fully_edge_order(N):
        out = []
        for j in range(N):
            for i in range(N):
                if i == j:
                    pass
                else:
                    if j > i:
                        out += [i*(N-1) + j-1]
                    else:
                        out += [i*(N-1) + j]
        return np.array(out)

    senders, receivers = get_fully_connected_senders_and_receivers(N)
    eorder = get_fully_edge_order(N)
    
else:
    raise ValueError(f'Invalid exp: {exp}')

# ----
key = jax.random.PRNGKey(42)

Ef = 1  # eij dim
Oh = 1

Eei = 5
Nei = 5

hidden = 5
nhidden = 2
# --

dropout_rate = args.dropout_rate
R, V = jnp.split(Zs[0], 2, axis=0)

# change dropout_rate=0. to turn dropout off
apply_fn = energy_fn(
    senders=senders, receivers=receivers, species=species, R=R, V=V, eorder=eorder, dropout_rate=dropout_rate)
Hmodel = generate_Hmodel(apply_fn)

def phi(x):
    X = jnp.vstack([x[:1, :]*0, x])
    return jnp.square(X[:-1, :] - X[1:, :]).sum(axis=1) - 1.0

constraints = get_constraints(N, dim, phi)

zdot_model, lamda_force_model = get_zdot_lambda(
    N, dim, hamiltonian=Hmodel, drag=None, 
    constraints=constraints, 
    external_force=None)

v_zdot_model = vmap(zdot_model, in_axes=(0, 0, None))

# Training phase

if method == 'vi':
    
    print(f"Training with VI ...")
    
    Rs, Vs = jnp.split(Zs, 2, axis=1)
    Rst, Vst = jnp.split(Zst, 2, axis=1)
    
    model = generate_model(Oh, Nei, Ef, Eei, dim, hidden, nhidden, v_zdot_model)
    key, rng_key_ = random.split(key)

    guide = numpyro.infer.autoguide.AutoNormal(model, init_scale=0.1)
    optimizer = numpyro.optim.ClippedAdam(step_size=1e-3, clip_norm=1e-6)
    svi = SVI(model, guide=guide, optim=optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key_, 100000, Rs, Vs, Zs_dot, norm_scale=1e-4, subsample_size=2500, stable_update=True)
    
    key, rng_key_ = random.split(key)
    predictive = numpyro.infer.Predictive(guide, params=svi_result.params, num_samples=num_samples)
    samples = predictive(rng_key_, Rs, Vs, Zs_dot)
    
    with open(save_dir + 'params.pkl', 'wb+') as f:
        pkl.dump(samples, f)

else:
    
    params = generate_HGNN_params(Oh, Nei, Ef, Eei, dim, hidden, nhidden, key)

    loss_fn = generate_loss_fn(v_zdot_model=v_zdot_model)
    gloss = generate_gloss(loss_fn=loss_fn)

    opt_init, opt_update_, get_params = optimizers.adam(lr)

    opt_update = generate_opt_update_wrapper(opt_update_=opt_update_)

    step = generate_update_fn(gloss=gloss, opt_update=opt_update, get_params=get_params)

    Rs, Vs = jnp.split(Zs, 2, axis=1)
    Rst, Vst = jnp.split(Zst, 2, axis=1)

    bRs, bVs, bZs_dot = batching(Rs, Vs, Zs_dot,
                                size=min(len(Rs), batch_size))

    print(f"Training ...")

    # opt_state = optimiser.init(params)
    opt_state = opt_init(params)

    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []

    last_loss = 1000

    for epoch in tqdm.trange(epochs):
        l = 0.0
        for data in zip(bRs, bVs, bZs_dot):
            change_RNG()
            optimizer_step += 1
            opt_state, params, l_ = step(optimizer_step, (opt_state, params, 0), *data)
            l += l_
        l = l/len(bRs)
        if (epoch + 1) % (epochs // 20) == 0:
            # opt_state, params, l = step(
            #     optimizer_step, (opt_state, params, 0), Rs, Vs, Zs_dot)
            larray += [l]
            ltarray += [loss_fn(params, Rst, Vst, Zst_dot)]
            print(f"Epoch: {epoch + 1}/{epochs} Loss (MSE):  train={larray[-1]}, test={ltarray[-1]}")
        
    params = get_params(opt_state)

    fig, axs = plt.subplots(1, 1)
    plt.semilogy(larray, label="Training")
    plt.semilogy(ltarray, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    ## save model and loss plot
    plt.savefig(save_dir + "loss.png")
    np.save(save_dir + "params.npy", params)

# ---------
# Simulation
# ---------
all_traj = []

sim_model = get_forward_sim_noparam(
    zdot_model=zdot_model, runs=runs, stride=stride, dt=dt, tol=1e-5)

test_count = min(test_count, Zs_test.shape[0])
pbar = tqdm.trange(test_count * num_samples)

for idx in range(test_count):
    
    pbar.set_description(f'Test {idx+1} of {test_count}')

    z_actual_out = Zs_test[idx]
    x_act_out, p_act_out = jnp.split(z_actual_out, 2, axis=1)

    Zs_init = Zs_test[idx:idx+1, 0:1]

    with jax.default_device(jax.devices('cpu')[0]):
        trajectories = {
            'pred_pos': [],
            'pred_vel': [],
            'actual_pos': jnp.array(x_act_out),
            'actual_vel': jnp.array(p_act_out),
        }

    for i in range(num_samples):

        if dropout_rate > 0.:
            change_RNG()

        if add_noise:
            Zs_noisy = add_noise_and_truncate(Zs_init, Zs_init, scale=noise_scale)[0].squeeze((0, 1))
            R_noisy = Zs_noisy[:N]
            V_noisy = Zs_noisy[N:]
        else:
            R_noisy = Zs_init.squeeze((0, 1))[:N]
            V_noisy = Zs_init.squeeze((0, 1))[N:]
            
        if method == 'vi':
            p_ = construct_param_from_samples(samples, i)
        else:
            p_ = params

        z_pred_out = sim_model(R_noisy, V_noisy, params=p_)
        x_pred_out, p_pred_out = jnp.split(z_pred_out, 2, axis=1)

        with jax.default_device(jax.devices('cpu')[0]):
            trajectories['pred_pos'].append(x_pred_out)
            trajectories['pred_vel'].append(p_pred_out)
            
        pbar.update()
        
    with jax.default_device(jax.devices('cpu')[0]):
        trajectories['pred_pos'] = jnp.array(trajectories['pred_pos'])
        trajectories['pred_vel'] = jnp.array(trajectories['pred_vel'])
        trajectories['pred_pos_avg'] = jnp.mean(trajectories['pred_pos'], axis=0)
        trajectories['pred_vel_avg'] = jnp.mean(trajectories['pred_vel'], axis=0)
    
    all_traj.append(trajectories)

with open(save_dir + f'all_traj.pkl', 'wb+') as f:
    pkl.dump(all_traj, f)

# ---------
