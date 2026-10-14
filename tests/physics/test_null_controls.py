"""Null controls: changing one switch must leave everything that does not depend on it
bit-identical. These catch accidental coupling between sectors."""
import numpy as np
import pytest

from godmax import validation as V
from godmax.get_Pkzs import get_Pkz

GALAXY_ONLY = ["uk_clm"]
MATTER_SIDE = ["Mclm_mat", "rho_clm_mat", "rho_dmb_mat", "uk_dmb", "Pmm_dmb_tot_mat", "Pmm_nfw_tot_mat",
               "Pmm_tot_mat", "phfit_kz_mat", "hmf_Mz_mat", "bias_Mz_mat", "Ncen_mat", "Nsat_mat"]


def _same(a, b, names):
    return [n for n in names if not np.array_equal(np.asarray(getattr(a, n)), np.asarray(getattr(b, n)), equal_nan=True)]


def test_clm_fourier_selector_only_changes_galaxies(model):
    """Shivam's test: legacy_fftlog vs direct_shell must leave the matter side identical."""
    shell = model("on", cls=get_Pkz)
    legacy = model("on", cls=get_Pkz, ana_over={"clm_fourier_transform_method": "legacy_fftlog"})
    assert _same(shell, legacy, MATTER_SIDE) == []
    assert _same(shell, legacy, GALAXY_ONLY) == GALAXY_ONLY      # and it does change the galaxy profile


def test_rho_clm_method_does_not_touch_galaxies_or_halofit(model):
    native, exact = model("on"), model("on", rho_clm=V.exact_rho_clm)
    names = ["uk_clm", "Pgg_tot_mat", "Cl_gal_gal_tot_mat", "phfit_kz_mat", "hmf_Mz_mat", "Mclm_mat", "zeta_mat",
             "rho_gas_mat", "rho_cga_mat"]
    assert _same(native, exact, names) == []


def test_rho_clm_method_irrelevant_without_backreaction(model):
    """Without backreaction rho_clm = f_clm rho_nfw. get_rho_dmb still evaluates get_rho_clm (both
    branches of a jnp.where), so replace it by zero: nothing may change."""
    from test_mass_conservation import _zero_rho_clm
    a, b = model("no_backreact"), model("no_backreact", rho_clm=_zero_rho_clm)
    assert _same(a, b, ["rho_dmb_mat", "Pmm_tot_mat", "Pyy_tot_kz_mat", "Pgg_tot_mat", "Cl_y_y_signal_mat"]) == []


def test_tsz_toggle_does_not_touch_matter_or_galaxies(model):
    a, b = model("on"), model("tsz_off")
    assert _same(a, b, ["Pmm_tot_mat", "Pgg_tot_mat", "Pgm_tot_mat", "Cl_kappa_kappa_tot_mat",
                        "Cl_gal_gal_tot_mat", "Cl_gal_kappa_tot_mat"]) == []
