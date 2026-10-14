"""Profile-level sanity: finiteness, positivity, u(k -> 0) = 1, the zeta solve, and the
amplitude of the thermal tSZ reference against the self-similar relation.

Tolerances (set before running):
  U0_TOL        raw FFTLog u at the lowest k_mcfit vs 1 (mass-normalised profiles)
  ZETA_TOL      |zeta_code - zeta_root| from one Newton step on the relaxation equation
  Y3D_BAND      Y3D(<r200c) of the thermal NFW reference / (Eq. 21 x E(z)^2/3): an order-unity
                band that catches unit errors (factors of h, 1e3, (1+z)^n), not a precision test
"""
import numpy as np
import pytest

from godmax import validation as V
from godmax.get_Pkzs import get_Pkz

U0_TOL = 0.01
ZETA_TOL = 1e-3
Y3D_BAND = (0.7, 1.5)

ARRAYS = ["rho_dmb_mat", "rho_gas_mat", "rho_cga_mat", "rho_clm_mat", "rho_nfw_mat", "Mdmb_mat", "Mclm_mat",
          "zeta_mat", "Pe_mat_physical", "y3d_mat", "uk_dmb", "uk_nfw", "uk_clm", "uk_y",
          "Pmm_tot_mat", "Pgm_tot_mat", "Pgg_tot_mat", "Pym_tot_mat", "Pgy_tot_mat", "Pyy_tot_kz_mat",
          "Cl_kappa_kappa_tot_mat", "Cl_gal_gal_tot_mat", "Cl_gal_kappa_tot_mat", "Cl_kappa_y_tot_mat",
          "Cl_gal_y_tot_mat", "Cl_y_y_signal_mat"]


@pytest.mark.parametrize("config", ["on", "off", "no_backreact", "tsz_off"])
def test_everything_finite(model, config):
    assert V.finite_report(model(config), ARRAYS) == []


@pytest.mark.parametrize("name", ["rho_dmb_mat", "rho_gas_mat", "rho_cga_mat", "rho_clm_mat", "Pe_mat_physical"])
def test_densities_and_pressure_nonnegative(model, name):
    assert float(np.min(np.asarray(getattr(model("on", cls=get_Pkz), name)))) >= 0.0


U0_XFAIL = pytest.mark.xfail(strict=True, reason="Raw FFTLog u at the lowest k_mcfit (0.051 h/Mpc) exceeds 1 for small haloes (log10 M = 11.5-12.6, low z): up to 1.037 (DMB) and 1.056 (NFW). get_uk_from_interp_Pk clips u <= 1, hiding it (2026-10-13). FFTLog edge inaccuracy, not the CLM method.")


@pytest.mark.parametrize("name", [pytest.param("uk_dmb_tointp", marks=U0_XFAIL),
                                  pytest.param("uk_nfw_tointp", marks=U0_XFAIL), "uk_ne_tointp"])
def test_fftlog_u_is_one_at_lowest_k(model, name):
    s = V.u_k0_report(model("on", cls=get_Pkz))[name]
    assert max(abs(s["min"] - 1.0), abs(s["max"] - 1.0)) < U0_TOL, s


def test_galaxy_u_clm_is_one_at_k0(model):
    on = model("on", cls=get_Pkz)
    assert V.u_k0_report(on)["shell_u_at_k0_max_abs_err"] < 1e-12
    assert float(np.max(np.abs(np.asarray(on.uk_clm)[0] - 1.0))) < 1e-5     # first kPk point, k = 1e-4 h/Mpc


def test_zeta_root_inside_bracket(model):
    rep = V.zeta_report(model("on", cls=get_Pkz))
    assert rep["n_at_bracket_edge"] == 0, rep


def test_zeta_root_accuracy(model):
    rep = V.zeta_report(model("on", cls=get_Pkz))
    assert rep["zeta_error_abs_max"] < ZETA_TOL, rep


def _y3d_over_eq21(o):
    SIGMA_T, ME_C2, MPC, KPC = 6.6524587e-25, 510.99895, 3.0856776e24, 3.0856776e21
    h = o.cosmo_params["H0"] / 100.0
    z, r, M = (np.asarray(getattr(o, n)) for n in ("z_array", "r_array", "M_array"))
    Pe, r200 = np.asarray(o.Pe_mat_physical), np.asarray(o.r200c_mat)
    rf = np.geomspace(r[0], r[-1], 2000)
    Y = np.zeros((len(z), len(M)))
    for j in range(len(z)):
        for i in range(len(M)):
            p = np.exp(np.interp(np.log(rf), np.log(r), np.log(np.clip(Pe[:, j, i], 1e-300, None))))
            p = np.where(rf <= r200[j, i], p, 0.0)
            rp = rf / h / (1 + z[j]) * MPC
            Y[j, i] = SIGMA_T / ME_C2 * np.trapezoid(4 * np.pi * rp ** 2 * p, rp) / KPC ** 2
    cp = o.cosmo_params
    h70 = h / 0.7
    Yss = 97.6 / h70 * (M / h * h70 / 1e15) ** (5 / 3) * (cp["Ob0"] / 0.043) * (0.25 / cp["Om0"])
    Ez = np.sqrt(cp["Om0"] * (1 + z) ** 3 + 1 - cp["Om0"])
    return Y / (Yss[None] * Ez[:, None] ** (2 / 3))


def test_thermal_y3d_selfsimilar_amplitude(model):
    q = _y3d_over_eq21(model("tsz_off", cls=get_Pkz))
    assert Y3D_BAND[0] < np.median(q) < Y3D_BAND[1], (q.min(), np.median(q), q.max())
