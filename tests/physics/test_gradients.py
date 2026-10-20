"""Gradients that HMC relies on.

1. Halofit non-linear scale: d k_nl / d Omega_m at the default num_points vs a grid-free
   reference (bisection + implicit-function theorem). At 64 nodes the error was 59%.
2. End-to-end: jax.grad of P_mm and P_yy through model construction vs a central finite
   difference, for a cosmology (Om0) and a baryonic (theta_ej_0) parameter.

Tolerances (set before running):
  KNL_GRAD_TOL   relative error of d k_nl / d Om vs the reference, pipeline redshifts
  E2E_GRAD_TOL   relative AD vs FD agreement
"""
import copy
import contextlib
import io
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest

from godmax import validation as V
from godmax.get_Pkzs import get_Pkz
from godmax.helpers.jax_cosmo_power import halofit_parameters

from conftest import base_dicts, CONFIGS

KNL_GRAD_TOL = 0.05
E2E_GRAD_TOL = 0.02


def _cosmo_fn(cp):
    def f(theta):
        return jc.Cosmology(Omega_c=theta[0] - cp["Ob0"], Omega_b=cp["Ob0"], h=cp["H0"] / 100., sigma8=theta[1],
                            n_s=cp["ns"], Omega_k=0., w0=cp.get("w0", -1.0), wa=0.)
    return f


@pytest.mark.slow
@pytest.mark.parametrize("z", [0.1, 0.3, 0.7, 1.2])
def test_halofit_knl_derivative(z, model):
    num_points = model("on", cls=get_Pkz).num_points_halofit
    cp = base_dicts()[0]["cosmo"]
    cosmo = _cosmo_fn(cp)
    theta = jnp.array([cp["Om0"], cp["sigma8"]])
    a = 1.0 / (1.0 + z)
    jac = np.asarray(jax.jacfwd(lambda th: halofit_parameters(cosmo(th), a, num_points=num_points))(theta)).reshape(3, 2)
    _, dknl_ref = V.knl_reference(cosmo, theta, a)
    err = np.abs(jac[0] / dknl_ref - 1.0)
    assert np.all(err < KNL_GRAD_TOL), f"num_points={num_points}: rel err d k_nl/d(Om, s8) = {err}"


def _observable(name, ik, jz):
    sim0, halo0, ana0, other0 = base_dicts()
    ana0.update(CONFIGS["on"])

    def f(par, value):
        sim = copy.deepcopy(sim0)
        if par == "Om0":
            sim["cosmo"] = dict(sim["cosmo"], Om0=value)
        else:
            sim[par] = value
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            obj = get_Pkz(sim, copy.deepcopy(halo0), copy.deepcopy(ana0), copy.deepcopy(other0))
        return jnp.log(getattr(obj, name)[ik, jz])
    return f, sim0


@pytest.mark.slow
@pytest.mark.parametrize("par", ["Om0", "theta_ej_0"])
@pytest.mark.parametrize("name", ["Pmm_tot_mat", "Pyy_tot_kz_mat"])
def test_end_to_end_gradient(par, name, model):
    o = model("on", cls=get_Pkz)
    ik = int(np.argmin(np.abs(np.asarray(o.kPk_array) - 1.0)))
    jz = 1
    f, sim0 = _observable(name, ik, jz)
    x0 = float(sim0["cosmo"]["Om0"] if par == "Om0" else sim0[par])
    g_ad = float(jax.grad(partial(f, par))(x0))
    h = 1e-3 * x0
    g_fd = float((f(par, x0 + h) - f(par, x0 - h)) / (2 * h))
    assert np.isfinite(g_ad), "gradient is not finite"
    assert abs(g_ad - g_fd) <= E2E_GRAD_TOL * max(abs(g_fd), 1e-3), f"d ln {name}/d {par}: AD {g_ad:.5g}, FD {g_fd:.5g}"
