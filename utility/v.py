import jax
import jax.numpy as jnp
from jax.scipy.special import j1, h

def construct_G_S_jax(pos_D, pos_S, k, n):
    M = pos_S.shape[0]
    N = pos_D.shape[0]
    a = (n**2 / jnp.pi)**0.5

    # Compute pairwise distances (broadcasting)
    pos_S_exp = pos_S[:, None, :]  # shape (M, 1, dim)
    pos_D_exp = pos_D[None, :, :]  # shape (1, N, dim)
    rho_ij = jnp.linalg.norm(pos_S_exp - pos_D_exp, axis=2)  # shape (M, N)

    G_S = -1j * 0.5 * jnp.pi * k * a * j1(k * a) * hankel2(0, k * rho_ij)
    return G_S