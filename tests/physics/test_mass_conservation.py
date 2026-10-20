"""Mass conservation of the collisionless matter (CLM) and radial-grid coverage.

The CLM is defined by its enclosed mass M_clm(<r) = f_clm M_nfw(r / zeta(r)). Every density
or Fourier profile built from it must give that mass back. Two paths use it:
  * galaxy satellites: shell transform of Mclm_mat (exact by construction);
  * total matter (rho_dmb -> u_dmb, P_mm) and the HSE enclosed mass Mdmb_mat (-> pressure,
    tSZ): Profiles.get_rho_clm (central log-derivative, approximately conservative).
The exact density (godmax.validation.exact_rho_clm) is the reference for the second path.

Tolerances (set before running, on the 64-point test grid):
  SHELL_NEG_TOL    shells more negative than this fraction of M(r_max) mean M_clm is not monotonic
  CLOSURE_TOL      rho_clm on r_array must recover the CLM mass to this fraction
  CARRIED_TOL      CLM mass inside Mdmb_mat(<r) vs the model, beyond 0.1 r200c
  SPECTRA_TOL      get_rho_clm vs the exact density, in every spectrum that uses it
  COVERAGE_TOL     mass on the radial grid vs its target, per component
"""
import numpy as np
import pytest

from godmax import validation as V
from godmax.get_Pkzs import get_Pkz

SHELL_NEG_TOL = 1e-4
CLOSURE_TOL = 0.01
CARRIED_TOL = 0.01
SPECTRA_TOL = 0.01
COVERAGE_TOL = 0.02


def _zero_rho_clm(self, jz, jM, r_array_here=None):
    import jax.numpy as jnp
    r = self.r_array if r_array_here is None else r_array_here
    return jnp.zeros_like(r)


def test_shell_masses_telescope(model):
    rep = V.shell_report(model("on", cls=get_Pkz))
    assert rep["telescoping_max_rel_err"] < 1e-12


def test_clm_enclosed_mass_monotonic(model):
    rep = V.shell_report(model("on", cls=get_Pkz))
    assert rep["min_shell_over_Mend"] > -SHELL_NEG_TOL, rep


def test_rho_clm_recovers_clm_mass(model):
    c = V.rho_clm_closure(model("on", cls=get_Pkz))
    assert np.max(np.abs(c - 1.0)) < CLOSURE_TOL, (c.min(), c.max())


@pytest.mark.xfail(strict=True, reason="get_rho_clm uses the central log-derivative (B7), which is approximate: on this test grid it carries +1.3% CLM mass into Mdmb_mat and gives C_yy +1.3..+2.3%, P_yy up to +1.4% vs the exact density (2026-10-13). Fix: exact density or a shell treatment of the DMB path.")
def test_clm_mass_carried_into_hse_mass(model):
    """Mdmb_mat re-integrates rho_dmb on its own grid (calling get_rho_clm there). Its CLM part,
    Mdmb[native] - Mdmb[zero CLM], must equal the model's M_clm(<r) - M_clm(<r_0)."""
    on = model("on", cls=get_Pkz)
    zero = model("on", cls=get_Pkz, rho_clm=_zero_rho_clm)
    M = np.asarray(on.Mclm_mat)
    carried = np.asarray(on.Mdmb_mat) - np.asarray(zero.Mdmb_mat)
    ratio = (carried - carried[:1])[1:] / (M - M[:1])[1:]
    x = (np.asarray(on.r_array)[:, None, None] / np.asarray(on.r200c_mat)[None])[1:]
    sel = x >= 0.1
    assert np.max(np.abs(ratio[sel] - 1.0)) < CARRIED_TOL, (ratio[sel].min(), ratio[sel].max())


B7_XFAIL = pytest.mark.xfail(strict=True, reason="get_rho_clm uses the central log-derivative (B7), which is approximate: on this test grid it carries +1.3% CLM mass into Mdmb_mat and gives C_yy +1.3..+2.3%, P_yy up to +1.4% vs the exact density (2026-10-13). Fix: exact density or a shell treatment of the DMB path.")


@pytest.mark.parametrize("name,kind", [("Pmm_tot_mat", "P"), ("Pgm_tot_mat", "P"),
                                       pytest.param("Pyy_tot_kz_mat", "P", marks=B7_XFAIL),
                                       ("Cl_kappa_kappa_tot_mat", "C"),
                                       pytest.param("Cl_y_y_signal_mat", "C", marks=B7_XFAIL),
                                       ("Cl_kappa_y_tot_mat", "C"), ("Cl_gal_y_tot_mat", "C")])
def test_get_rho_clm_matches_exact_density(model, name, kind):
    """Spectra built with the pipeline's get_rho_clm vs the exact CLM density.
    P(k): k <= 10 h/Mpc, all z.  C(ell): all ell."""
    native, exact = model("on"), model("on", rho_clm=V.exact_rho_clm)
    a, b = np.asarray(getattr(native, name)), np.asarray(getattr(exact, name))
    if kind == "P":
        sel = np.asarray(native.kPk_array) <= 10.0
        a, b = a[sel], b[sel]
    dev = np.max(np.abs(a / b - 1.0))
    assert dev < SPECTRA_TOL, f"{name}: max |native/exact - 1| = {dev:.4f}"


GAS_NORM_TOL = 0.005     # integral of rho_gas to infinity vs f_gas M_tot
GAS_EXCESS_TOL = 0.005   # gas on the radial grid may not exceed the budget by more than the grid quadrature error
PGE_EJECTION = {"nu_theta_ej_M": -0.33}   # mass-dependent ejection radius (tSZxEuclid Pge): r_ej up to ~60 r200c


def _gas_to_infinity_over_budget(obj):
    """Independent check: integrate the model's gas profile, rebuilt from its parameters
    (beta, r_co, r_ej, gamma, delta, normalisation), from 0 to effectively infinity with numpy."""
    A = lambda n: np.asarray(getattr(obj, n))
    beta, rco, rej, norm = A("beta_mat"), A("r_co_mat"), A("r_ej_mat"), A("rho_gas_norm_mat")
    g, d = float(obj.gamma_rhogas), float(obj.delta_rhogas)
    out = np.zeros_like(beta)
    for j, i in np.ndindex(beta.shape):
        r = np.geomspace(1e-6 * rco[j, i], 1e3 * rej[j, i], 200001)
        rho = norm[j, i] / ((1 + r / rco[j, i]) ** beta[j, i] * (1 + (r / rej[j, i]) ** g) ** ((d - beta[j, i]) / g))
        out[j, i] = np.trapezoid(4 * np.pi * r ** 2 * rho, r)
    return out / (A("fgas_mat") * A("Mtot_mat"))


@pytest.mark.parametrize("sim_over", [{}, PGE_EJECTION], ids=["default", "pge_ejection"])
def test_gas_normalised_to_infinity(model, sim_over):
    """M_gas(<inf) = f_gas M_tot (Schneider & Teyssier 2019 Eq. 2.13; GODMAX paper Sec. 2.1.2)."""
    q = _gas_to_infinity_over_budget(model("on", cls=get_Pkz, sim_over=sim_over))
    assert np.max(np.abs(q - 1.0)) < GAS_NORM_TOL, (q.min(), q.max())


@pytest.mark.parametrize("sim_over", [{}, PGE_EJECTION], ids=["default", "pge_ejection"])
def test_gas_on_grid_never_exceeds_budget(model, sim_over):
    """The grid is part of all space, so it cannot hold more gas than f_gas M_tot. The old
    normalisation (within 8 r200c) put up to 20x the budget on the grid with the Pge ejection radius."""
    o = model("on", cls=get_Pkz, sim_over=sim_over)
    gas = V.coverage_report(o)["gas_on_grid"]
    j, i = np.unravel_index(np.argmax(gas), gas.shape)
    assert gas.max() < 1.0 + GAS_EXCESS_TOL, f"max {gas.max():.3f} at log10M={np.log10(np.asarray(o.M_array)[i]):.2f}"


COV_XFAIL = pytest.mark.xfail(strict=True, reason="The radial grid (rmax = 16 Mpc/h) does not reach RMAX_R200c x r200c = 8 r200c, the radius M_tot is integrated to, for the 10^15.5 Msun/h halo at z = 1.6: CLM endpoint 0.955, DMB 0.964 of target (2026-10-13). All other (z, M) are within 1.1%.")


@pytest.mark.parametrize("component", [pytest.param("clm_endpoint", marks=COV_XFAIL),
                                       pytest.param("dmb_on_grid", marks=COV_XFAIL), "gas_on_grid"])
def test_radial_grid_contains_halo_mass(model, component):
    cov = V.coverage_report(model("on", cls=get_Pkz))[component]
    dev = np.abs(cov - 1.0)
    i = np.unravel_index(np.argmax(dev), dev.shape)
    M = np.asarray(model("on", cls=get_Pkz).M_array)
    assert dev.max() < COVERAGE_TOL, (f"{component}: worst {cov[i]:.3f} at (jz={i[0]}, log10M={np.log10(M[i[1]]):.2f}); "
                                      f"range {cov.min():.3f}-{cov.max():.3f}")
