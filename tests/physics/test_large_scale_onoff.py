"""Large-scale invariant: baryonification only moves mass inside haloes, so switching it on
or off must leave mass-normalised spectra unchanged on large scales (k <= 0.01 h/Mpc, and the
lowest multipole). tSZ is exempt (pressure, not mass); its ratios are only required to be finite.

ON  = DMB matter, backreaction, baryonified tSZ.
OFF = halofit matter, no backreaction, thermal NFW tSZ.
Also checked: backreaction alone (galaxies through the CLM), and CMB lensing (one source bin).

Tolerances (set before running):
  LOWK_TOL     max |ON/OFF - 1| at k <= 0.01 h/Mpc, all z
  ELL_TOL      |ON/OFF - 1| at the lowest ell
  SHELL_TOL    galaxy power with the shell transform, backreaction ON/OFF (Shivam: 1.5e-7)
"""
import numpy as np
import pytest

from godmax import validation as V

LOWK_TOL = 0.01
ELL_TOL = 0.01
SHELL_TOL = 1e-5


@pytest.mark.parametrize("name", ["Pmm_tot_mat", "Pgm_tot_mat", "Pgg_tot_mat"])
def test_baryonification_on_off_low_k(model, name):
    dev = V.max_lowk_deviation(model("on"), model("off"), name)
    assert dev < LOWK_TOL, f"{name}: {dev:.4g}"


@pytest.mark.parametrize("name", ["Pmm_tot_mat", "Pgm_tot_mat"])
def test_backreaction_on_off_low_k(model, name):
    dev = V.max_lowk_deviation(model("on"), model("no_backreact"), name)
    assert dev < LOWK_TOL, f"{name}: {dev:.4g}"


def test_galaxy_power_backreaction_shell_exact(model):
    """Shivam's test: P_gg backreaction ON/OFF -> 1 to ~1e-7 with the shell transform."""
    dev = V.max_lowk_deviation(model("on"), model("no_backreact"), "Pgg_tot_mat")
    assert dev < SHELL_TOL, f"{dev:.3g}"


@pytest.mark.parametrize("name", ["Cl_kappa_kappa_tot_mat", "Cl_gal_kappa_tot_mat", "Cl_gal_gal_tot_mat"])
def test_baryonification_on_off_lowest_ell(model, name):
    dev = V.lowest_ell_deviation(model("on"), model("off"), name)
    assert dev < ELL_TOL, f"{name}: {dev:.4g}"


def test_cmb_lensing_on_off_lowest_ell(model):
    on = model("on", ana_over={"is_cmb_lensing": True})
    off = model("off", ana_over={"is_cmb_lensing": True})
    dev = V.lowest_ell_deviation(on, off, "Cl_kappa_kappa_tot_mat")
    assert dev < ELL_TOL, f"{dev:.4g}"


@pytest.mark.parametrize("name", ["Pyy_tot_kz_mat", "Pym_tot_mat", "Pgy_tot_mat"])
def test_tsz_on_off_finite(model, name):
    """Not required to be 1 (integrated Compton-Y differs between branches), only finite and positive."""
    r = np.asarray(getattr(model("on"), name)) / np.asarray(getattr(model("off"), name))
    assert np.all(np.isfinite(r)) and np.all(r > 0)
