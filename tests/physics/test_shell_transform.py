"""Analytic controls for the CLM shell transform (get_Pkz.get_uk_clm_shell).

Reproduces the GODMAX-independent checks of Shivam's notebooks/debug_tsv/
compare_clm_shell_fourier.ipynb (commit 14ae4e4): shell masses telescope, the closed-form
windows agree with direct quadrature, u(k=0) = 1 exactly, the low-k limit has the right
curvature, and the transform is differentiable.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from godmax.get_Pkzs import get_Pkz
from godmax.get_radial_profiles import Profiles

R = jnp.geomspace(0.005, 48.0, 23)                                   # Mpc/h


def synthetic_mass(r, scale=0.7):
    """Smooth, monotonic, saturating enclosed mass [Msun/h], shape (nr, 1, 1)."""
    return (3e14 * (r / scale) ** 1.7 / (1.0 + (r / scale) ** 1.7))[:, None, None]


def shell_u(k, M, r=R):
    """get_uk_clm_shell on an arbitrary (r, M) without building a full model."""
    fake = SimpleNamespace(r_array=r, Mclm_mat=M, get_Mclm_shell_masses=Profiles.get_Mclm_shell_masses,
                           _clm_inner_window=get_Pkz._clm_inner_window, _clm_shell_window=get_Pkz._clm_shell_window)
    return get_Pkz.get_uk_clm_shell(fake, k, M)


def test_shell_masses_telescope():
    M = synthetic_mass(R)
    dm = Profiles.get_Mclm_shell_masses(M)
    assert float(jnp.max(jnp.abs(jnp.cumsum(dm, axis=0) / M - 1.0))) < 1e-12


def test_inner_window_matches_quadrature():
    """Inner cell M(<r) ~ r^2:  W(x) = 2 (1 - cos x) / x^2 = int_0^1 sinc(x sqrt(y)) dy."""
    xs = jnp.linspace(0.0, 0.75, 21)
    ys = jnp.linspace(0.0, 1.0, 50001)
    dense = jax.vmap(lambda x: jnp.trapezoid(jnp.sinc(x * jnp.sqrt(ys) / jnp.pi), x=ys))(xs)
    assert float(jnp.max(jnp.abs(get_Pkz._clm_inner_window(xs) - dense))) < 1e-8


def test_shell_window_matches_quadrature():
    """Constant-density shell [0.3, 0.9] Mpc/h, including the high-k sign changes."""
    k = jnp.linspace(0.0, 150.0, 51)
    rin, rout = 0.3, 0.9
    rr = jnp.linspace(rin, rout, 100001)
    dense = jax.vmap(lambda kk: 3.0 * jnp.trapezoid(rr ** 2 * jnp.sinc(kk * rr / jnp.pi), x=rr) / (rout ** 3 - rin ** 3))(k)
    closed = get_Pkz._clm_shell_window(k, jnp.array([rin, rout]))[:, 0]
    assert float(jnp.max(jnp.abs(closed - dense))) < 1e-8


def test_shell_u_is_one_at_k0():
    u0 = shell_u(jnp.array([0.0]), synthetic_mass(R))
    assert float(jnp.max(jnp.abs(u0 - 1.0))) < 1e-14


def test_shell_low_k_curvature():
    """u(k) = 1 - k^2 <r^2> / 6 + O(k^4), with <r^2> from the same shells."""
    M = synthetic_mass(R)
    dm = np.asarray(Profiles.get_Mclm_shell_masses(M))[:, 0, 0]
    r = np.asarray(R)
    rin, rout = r[:-1], r[1:]
    r2_shell = 0.6 * (rout ** 5 - rin ** 5) / (rout ** 3 - rin ** 3)      # <r^2> of a constant-density shell
    r2_inner = 0.5 * r[0] ** 2                                            # <r^2> of the rho ~ 1/r inner cell
    mean_r2 = (dm[0] * r2_inner + np.sum(dm[1:] * r2_shell)) / float(M[-1, 0, 0])
    k = 1e-3
    u = float(shell_u(jnp.array([k]), M)[0, 0, 0])
    assert abs((1.0 - u) / (k ** 2 * mean_r2 / 6.0) - 1.0) < 1e-4


def test_shell_transform_gradient_matches_finite_difference():
    shape = jnp.linspace(-0.2, 0.2, R.size)[:, None, None]
    k = jnp.geomspace(1e-3, 50.0, 40)

    def objective(theta):
        return jnp.mean(shell_u(k, synthetic_mass(R) * jnp.exp(theta * shape)))

    g_ad = float(jax.grad(objective)(0.0))
    eps = 1e-5
    g_fd = float((objective(eps) - objective(-eps)) / (2 * eps))
    assert np.isfinite(g_ad) and abs(g_ad / g_fd - 1.0) < 1e-6
