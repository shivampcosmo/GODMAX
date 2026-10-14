"""Validation tests for the single baryonification_tSZ switch.

ON  (True):  full baryonic tSZ model — baryonified gas density, DMB
             gravitational potential, nonthermal pressure included
             (P_th = P_tot * (1 - R_nt)).
OFF (False): fully thermal NFW reference — gas traces NFW, NFW potential,
             P_th = P_tot.

The branches must be independent: no Y3D or low-k matching between them, and
the OFF branch must never evaluate the nonthermal prescription.
"""
import copy
import pathlib

import jax.numpy as jnp
import pytest
import yaml

from godmax.get_radial_profiles import Profiles

PARAMS_PATH = pathlib.Path(__file__).resolve().parents[1] / "param_files" / "params_default.yaml"


def load_param_dicts():
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)
    sim = copy.deepcopy(params["sim_params"])
    halo = copy.deepcopy(params["halo_params"])
    ana = copy.deepcopy(params["analysis"])
    other = copy.deepcopy(params["other_params"])
    # Tiny grids: these tests check branch logic, not accuracy.
    halo.update(nr=16, nz=4, nM=6, nk=16)
    ana.update(model_galaxies=False, backreaction=False, model_tSZ=True)
    ana["num_points_trapz_int"] = 16
    # YAML 1.1 reads bare "5e-4" as a string; base_class needs a float scalar
    # here (a length-1 list would not broadcast onto the redshift grid).
    ana["nbar_gal_comoving_val"] = float(ana["nbar_gal_comoving_val"][0])
    return sim, halo, ana, other


def make_profiles(baryonification_tSZ, alpha_nt=None):
    sim, halo, ana, other = load_param_dicts()
    ana["baryonification_tSZ"] = baryonification_tSZ
    if alpha_nt is not None:
        sim["alpha_nt"] = alpha_nt
    return Profiles(sim, halo, ana, other)


@pytest.fixture(scope="module")
def profiles_on():
    return make_profiles(True)


@pytest.fixture(scope="module")
def profiles_off():
    return make_profiles(False)


def test_off_is_fully_thermal(profiles_off):
    """OFF: P_e = P_tot / 1.932 with no nonthermal factor."""
    assert jnp.allclose(
        profiles_off.Pe_mat_physical,
        profiles_off.Ptot_mat_physical / 1.932,
    )


def test_on_includes_nonthermal(profiles_on):
    """ON: P_e = P_tot * max(0, 1 - R_nt) / 1.932."""
    expected = (
        profiles_on.Ptot_mat_physical
        * jnp.maximum(0.0, 1.0 - profiles_on.Pnt_fac_bary_mat)
        / 1.932
    )
    assert jnp.allclose(profiles_on.Pe_mat_physical, expected)
    # Nonthermal support must actually do something somewhere on the grid.
    assert float(jnp.max(profiles_on.Pnt_fac_bary_mat)) > 0.0


def test_branch_independence(profiles_on, profiles_off):
    """Each run computes only its own branch."""
    # OFF: y3d comes straight from run_pressure_calc_nfw; bary branch never ran.
    assert not hasattr(profiles_off, "y3d_bary_mat")
    assert jnp.array_equal(profiles_off.y3d_mat, profiles_off.y3d_nfw_mat)
    # ON: NFW reference never ran.
    assert not hasattr(profiles_on, "y3d_nfw_mat")
    assert jnp.array_equal(profiles_on.y3d_mat, profiles_on.y3d_bary_mat)


def test_no_forced_normalization(profiles_on, profiles_off):
    """Integrated Compton-Y must not be equal by construction between branches."""
    assert not jnp.allclose(profiles_on.Y3D_active_mat, profiles_off.Y3D_active_mat)


def test_off_insensitive_to_alpha_nt():
    """OFF never evaluates the nonthermal prescription, so alpha_nt is inert."""
    y3d_a = make_profiles(False, alpha_nt=0.18).y3d_mat
    y3d_b = make_profiles(False, alpha_nt=0.36).y3d_mat
    assert jnp.array_equal(y3d_a, y3d_b)


def test_on_sensitive_to_alpha_nt():
    """ON includes nonthermal support, so alpha_nt must change the result."""
    y3d_a = make_profiles(True, alpha_nt=0.18).y3d_mat
    y3d_b = make_profiles(True, alpha_nt=0.36).y3d_mat
    assert not jnp.allclose(y3d_a, y3d_b)
