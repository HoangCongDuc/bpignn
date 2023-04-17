import numpy as np
import jax.numpy as jnp


def truncate(Zs, Zs_dot, decimals=2):
    N = Zs.shape[2] // 2
    xs = jnp.round(Zs[:, :, :N], decimals=decimals)
    vs = jnp.round(Zs[:, :, N:], decimals=9*decimals)
    Zs_new = jnp.concatenate([xs, vs], axis=2)
    return Zs_new, jnp.round(Zs_dot, decimals=9*decimals)
    
    # return jnp.round(Zs, decimals=decimals), jnp.round(Zs_dot, decimals=decimals)


def add_noise(Zs, Zs_dot, scale=0.05):
    
    N_samples = Zs.shape[0]
    N_t = Zs.shape[1]
    N = Zs.shape[2] // 2
    d = Zs.shape[3]
    
    # noise = jnp.array(scale * np.random.randn(N_samples, N_t, 3 * N, d))
    # return Zs + noise[:, :, :2*N], Zs_dot + noise[:, :, N:]
    
    noise = scale * np.random.randn(N_samples, N_t, N, d)
    new_Zs = np.array(Zs)
    new_Zs[:, :, :N] += noise
    new_Zs = jnp.array(new_Zs)
    return new_Zs, Zs_dot



def add_noise_and_truncate(Zs, Zs_dot, scale=0.05, decimals=2):
    if scale == 0:
        return truncate(Zs=Zs, Zs_dot=Zs_dot, decimals=decimals)
    else:
        return truncate(*add_noise(Zs=Zs, Zs_dot=Zs_dot, scale=scale), decimals=decimals)
