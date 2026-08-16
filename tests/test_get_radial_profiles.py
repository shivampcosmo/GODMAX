"""Regression coverage for the core HOD stellar-calculation path."""

from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from get_radial_profiles import Profiles


FORMULA_RTOL = 1.0e-12


class _SyntheticGalaxyProfiles(Profiles):
    """Exercise live HOD methods while replacing unrelated SHMR root finding."""

    def get_Mthresh(self, jz):
        return self.synthetic_thresholds[jz]

    def get_Mstar_Mh(self, jz, jM):
        return self.synthetic_stellar_masses[jz, jM]

    def get_Mh_Mstar(self, jz, jM, Mstar_array=None):
        return self.synthetic_halo_at_threshold[jz]


def _synthetic_galaxy_profiles():
    profiles = object.__new__(_SyntheticGalaxyProfiles)
    profiles.model_galaxies = True
    profiles.nz = 2
    profiles.nM = 3
    profiles.num_points_gal_cal = 12
    profiles.M_array = jnp.asarray([1.0e12, 3.0e12, 1.0e13], dtype=jnp.float64)
    profiles.Mtot_mat = jnp.broadcast_to(profiles.M_array, (profiles.nz, profiles.nM))
    profiles.synthetic_thresholds = jnp.asarray([1.0e10, 2.0e10], dtype=jnp.float64)
    profiles.synthetic_stellar_masses = jnp.asarray(
        [[0.5e10, 1.0e10, 2.0e10], [1.0e10, 2.0e10, 4.0e10]],
        dtype=jnp.float64,
    )
    profiles.synthetic_halo_at_threshold = jnp.asarray([8.0e11, 1.2e12], dtype=jnp.float64)
    profiles.siglogMstar_Ncen_z = jnp.asarray([0.25, 0.30], dtype=jnp.float64)
    profiles.fcen_z = jnp.asarray([0.8, 0.6], dtype=jnp.float64)
    profiles.h = 0.7
    profiles.Bsat_Nsat_z = jnp.asarray([9.01, 8.5], dtype=jnp.float64)
    profiles.Bcut_Nsat_z = jnp.asarray([1.69, 1.5], dtype=jnp.float64)
    profiles.betasat_Nsat_z = jnp.asarray([0.74, 0.70], dtype=jnp.float64)
    profiles.betacut_Nsat_z = jnp.asarray([0.60, 0.55], dtype=jnp.float64)
    profiles.alphasat_Nsat_z = jnp.asarray([1.0, 1.1], dtype=jnp.float64)
    profiles.Ob0 = 0.049
    profiles.Om0 = 0.31
    return profiles


def _expected_occupations(profiles):
    log10_threshold = jnp.log10(profiles.synthetic_thresholds)[:, None]
    log10_stellar_mass = jnp.log10(profiles.synthetic_stellar_masses)
    ncen = profiles.fcen_z[:, None] * 0.5 * (
        1
        - jax.lax.erf(
            (log10_threshold - log10_stellar_mass)
            / (jnp.sqrt(2) * profiles.siglogMstar_Ncen_z[:, None])
        )
    )

    halo_at_threshold = profiles.synthetic_halo_at_threshold[:, None]
    msat = (
        (1.0e12 * profiles.h)
        * profiles.Bsat_Nsat_z[:, None]
        * (halo_at_threshold / 1.0e12) ** profiles.betasat_Nsat_z[:, None]
    )
    mcut = (
        (1.0e12 * profiles.h)
        * profiles.Bcut_Nsat_z[:, None]
        * (halo_at_threshold / 1.0e12) ** profiles.betacut_Nsat_z[:, None]
    )
    nsat = (
        ncen / jnp.maximum(profiles.fcen_z[:, None], 1.0e-10)
        * (profiles.M_array[None, :] / msat) ** profiles.alphasat_Nsat_z[:, None]
        * jnp.exp(-mcut / profiles.M_array[None, :])
    )
    return ncen, nsat


def test_run_stars_calc_builds_galaxy_occupations_and_stellar_fractions():
    profiles = _synthetic_galaxy_profiles()

    profiles.run_stars_calc()

    expected_ncen, expected_nsat = _expected_occupations(profiles)
    np.testing.assert_allclose(profiles.Ncen_mat, expected_ncen, rtol=FORMULA_RTOL, atol=0.0)
    np.testing.assert_allclose(profiles.Nsat_mat, expected_nsat, rtol=FORMULA_RTOL, atol=0.0)

    expected_shape = (profiles.nz, profiles.nM)
    for name in ("Ncen_mat", "Nsat_mat", "fstar_cen_mat", "fstar_sat_mat", "fstar_tot_mat"):
        value = np.asarray(getattr(profiles, name))
        assert value.shape == expected_shape
        assert np.all(np.isfinite(value))

    np.testing.assert_allclose(
        profiles.fstar_tot_mat,
        profiles.fstar_cen_mat + profiles.fstar_sat_mat,
        rtol=FORMULA_RTOL,
        atol=0.0,
    )
    component_cap = 0.49 * profiles.Ob0 / profiles.Om0
    assert np.all(np.asarray(profiles.fstar_cen_mat) >= 0.0)
    assert np.all(np.asarray(profiles.fstar_sat_mat) >= 0.0)
    assert np.all(np.asarray(profiles.fstar_cen_mat) <= component_cap)
    assert np.all(np.asarray(profiles.fstar_sat_mat) <= component_cap)

    fgas = profiles.Ob0 / profiles.Om0 - profiles.fstar_tot_mat
    fclm = 1.0 - profiles.Ob0 / profiles.Om0 + profiles.fstar_sat_mat
    np.testing.assert_allclose(
        fgas + fclm + profiles.fstar_cen_mat,
        jnp.ones(expected_shape, dtype=jnp.float64),
        rtol=FORMULA_RTOL,
        atol=0.0,
    )


def test_hod_occupations_preserve_finite_nonzero_gradients():
    def occupations(log10_threshold):
        profiles = _synthetic_galaxy_profiles()
        profiles.Mthresh_array = profiles.synthetic_thresholds.at[0].set(10**log10_threshold)
        return jnp.stack((profiles.get_Ncen(0, 1), profiles.get_Nsat(0, 1)))

    jacobian = jax.jacrev(occupations)(jnp.asarray(10.1, dtype=jnp.float64))

    assert np.all(np.isfinite(np.asarray(jacobian)))
    assert np.all(np.asarray(jnp.abs(jacobian)) > 0.0)


def test_run_stars_calc_without_galaxies_is_unchanged_power_law_branch():
    profiles = object.__new__(Profiles)
    profiles.model_galaxies = False
    profiles.M_array = jnp.asarray([1.0e12, 2.0e12], dtype=jnp.float64)
    profiles.nz = 2
    profiles.A_starcga = 0.09
    profiles.M1_starcga = 10**11.4
    profiles.eta_cga = 0.6
    profiles.eta_star = 0.3

    profiles.run_stars_calc()

    expected_central = profiles.A_starcga * (
        profiles.M1_starcga / profiles.M_array
    ) ** profiles.eta_cga
    expected_total = profiles.A_starcga * (
        profiles.M1_starcga / profiles.M_array
    ) ** profiles.eta_star
    expected_central = jnp.broadcast_to(expected_central, (profiles.nz, profiles.M_array.size))
    expected_total = jnp.broadcast_to(expected_total, (profiles.nz, profiles.M_array.size))

    np.testing.assert_allclose(
        profiles.fstar_cen_mat, expected_central, rtol=FORMULA_RTOL, atol=0.0
    )
    np.testing.assert_allclose(
        profiles.fstar_tot_mat, expected_total, rtol=FORMULA_RTOL, atol=0.0
    )
    np.testing.assert_allclose(
        profiles.fstar_sat_mat,
        expected_total - expected_central,
        rtol=FORMULA_RTOL,
        atol=0.0,
    )
