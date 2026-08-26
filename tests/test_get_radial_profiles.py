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

from get_radial_profiles import Profiles, _nonthermal_redshift_factor
from get_Pkzs import get_Pkz


FORMULA_RTOL = 1.0e-12


def _nonthermal_pressure_fraction(alpha_nt, z, r_ratio=0.7):
    n_nt = jnp.asarray(0.3, dtype=jnp.float64)
    beta_nt = jnp.asarray(0.5, dtype=jnp.float64)
    return (
        alpha_nt
        * _nonthermal_redshift_factor(alpha_nt, n_nt, beta_nt, z)
        * r_ratio**n_nt
    )


def _legacy_positive_nonthermal_pressure_fraction(alpha_nt, z, r_ratio=0.7):
    n_nt = jnp.asarray(0.3, dtype=jnp.float64)
    beta_nt = jnp.asarray(0.5, dtype=jnp.float64)
    fmax = 8.0 ** (-n_nt) / alpha_nt
    fz = jnp.minimum(
        (1 + z) ** beta_nt,
        (fmax - 1) * jnp.tanh(beta_nt * z) + 1,
    )
    return alpha_nt * fz * r_ratio**n_nt


def test_nonthermal_pressure_alpha_zero_has_finite_right_limit_gradient():
    alpha_zero = jnp.asarray(0.0, dtype=jnp.float64)
    z = jnp.asarray(0.5, dtype=jnp.float64)

    value = _nonthermal_pressure_fraction(alpha_zero, z)
    gradient = jax.grad(_nonthermal_pressure_fraction)(alpha_zero, z)
    expected_gradient = (1 + z) ** 0.5 * 0.7**0.3

    np.testing.assert_allclose(value, 0.0, rtol=0.0, atol=0.0)
    assert np.isfinite(float(gradient))
    np.testing.assert_allclose(
        gradient,
        expected_gradient,
        rtol=FORMULA_RTOL,
        atol=0.0,
    )


def test_nonthermal_pressure_positive_alpha_preserves_value_and_gradient():
    alphas = jnp.asarray([0.01, 0.05, 0.18, 0.5], dtype=jnp.float64)
    redshifts = jnp.asarray([0.0, 0.5, 1.0, 2.0], dtype=jnp.float64)

    for alpha_nt, z in zip(alphas, redshifts):
        value = _nonthermal_pressure_fraction(alpha_nt, z)
        legacy_value = _legacy_positive_nonthermal_pressure_fraction(alpha_nt, z)
        gradient = jax.grad(_nonthermal_pressure_fraction)(alpha_nt, z)
        legacy_gradient = jax.grad(
            _legacy_positive_nonthermal_pressure_fraction
        )(alpha_nt, z)
        np.testing.assert_allclose(
            value,
            legacy_value,
            rtol=FORMULA_RTOL,
            atol=0.0,
        )
        np.testing.assert_allclose(
            gradient,
            legacy_gradient,
            rtol=FORMULA_RTOL,
            atol=0.0,
        )


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


def _synthetic_clm_shell_profiles():
    profiles = object.__new__(get_Pkz)
    profiles.r_array = jnp.geomspace(0.005, 48.0, 23)
    r = profiles.r_array[:, None]
    scale = jnp.asarray([1.0e10, 1.0e16])[None, :]
    radius = jnp.asarray([0.2, 5.0])[None, :]
    enclosed = scale * (r / radius) ** 1.8 / (1.0 + (r / radius) ** 1.8)
    profiles.Mclm_mat = enclosed[:, None, :]
    return profiles


def test_clm_shell_telescope_zero_mode_and_extreme_mass_scales():
    profiles = _synthetic_clm_shell_profiles()
    shell_mass = profiles.get_Mclm_shell_masses(profiles.Mclm_mat)
    reconstructed = jnp.cumsum(shell_mass, axis=0)
    uk_zero = profiles.get_uk_clm_shell(jnp.asarray([0.0]))

    np.testing.assert_allclose(
        reconstructed, profiles.Mclm_mat, rtol=FORMULA_RTOL, atol=1.0e-15
    )
    np.testing.assert_allclose(uk_zero, 1.0, rtol=FORMULA_RTOL, atol=1.0e-15)
    assert uk_zero.shape == (1, 1, 2)
    assert np.all(np.isfinite(np.asarray(uk_zero)))

    k_target = jnp.geomspace(1.0e-4, 20.0, 13)
    k_raw = jnp.geomspace(2.0e-4, 40.0, 9)
    combined = profiles.get_uk_clm_shell(jnp.concatenate((k_target, k_raw)))
    np.testing.assert_allclose(
        combined[:k_target.size],
        profiles.get_uk_clm_shell(k_target),
        rtol=FORMULA_RTOL,
        atol=0.0,
    )
    np.testing.assert_allclose(
        combined[k_target.size:],
        profiles.get_uk_clm_shell(k_raw),
        rtol=FORMULA_RTOL,
        atol=0.0,
    )


def test_clm_shell_low_k_moment_has_the_physical_sign_and_curvature():
    profiles = _synthetic_clm_shell_profiles()
    r = profiles.r_array
    shell_mass = profiles.get_Mclm_shell_masses(profiles.Mclm_mat)
    rin, rout = r[:-1], r[1:]
    shell_r2 = (3.0 / 5.0) * (rout**5 - rin**5) / (rout**3 - rin**3)
    mean_r2 = (
        profiles.Mclm_mat[0] * r[0]**2 / 2.0
        + jnp.einsum('r,rzm->zm', shell_r2, shell_mass[1:])
    ) / profiles.Mclm_mat[-1]

    k = jnp.asarray([1.0e-3 / r[-1]])
    uk = profiles.get_uk_clm_shell(k)[0]
    measured = (1.0 - uk) / k[0]**2

    assert np.all(np.asarray(uk) < 1.0)
    assert np.all(np.asarray(measured) > 0.0)
    np.testing.assert_allclose(measured, mean_r2 / 6.0, rtol=2.0e-6, atol=0.0)


def test_clm_shell_keeps_signed_high_k_windows_and_negative_input_shells():
    profiles = object.__new__(get_Pkz)
    profiles.r_array = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    profiles.Mclm_mat = jnp.asarray([0.0, 1.0], dtype=jnp.float64)[:, None, None]
    k = jnp.linspace(0.1, 30.0, 512)
    uk = profiles.get_uk_clm_shell(k)[:, 0, 0]
    index = int(jnp.argmin(uk))

    radius = jnp.linspace(1.0, 2.0, 20001)
    direct = jnp.trapezoid(
        3.0 * radius**2 / (2.0**3 - 1.0**3)
        * jnp.sinc(k[index] * radius / jnp.pi),
        x=radius,
    )
    assert float(uk[index]) < 0.0
    np.testing.assert_allclose(uk[index], direct, rtol=2.0e-8, atol=2.0e-10)

    nonmonotone = jnp.asarray([1.0, 3.0, 2.0, 5.0])[:, None, None]
    shell_mass = profiles.get_Mclm_shell_masses(nonmonotone)
    assert float(shell_mass[2, 0, 0]) < 0.0
    np.testing.assert_allclose(
        jnp.cumsum(shell_mass, axis=0), nonmonotone, rtol=FORMULA_RTOL, atol=0.0
    )

    invalid_endpoint = jnp.asarray([1.0, 0.0])[:, None, None]
    assert np.all(np.isnan(np.asarray(profiles.get_uk_clm_shell(k[:1], invalid_endpoint))))


def test_clm_shell_is_jittable_and_has_a_finite_nonzero_shape_gradient():
    profiles = _synthetic_clm_shell_profiles()
    base_mass = profiles.Mclm_mat
    weights = jnp.linspace(0.5, 1.5, 17)[:, None, None]
    k = jnp.geomspace(1.0e-3, 30.0, 17)

    def objective(theta):
        deformation = jnp.linspace(1.0, 0.0, base_mass.shape[0])[:, None, None]
        varied_mass = base_mass * (1.0 + theta * deformation)
        return jnp.sum(weights * profiles.get_uk_clm_shell(k, varied_mass))

    theta = jnp.asarray(0.03, dtype=jnp.float64)
    value, gradient = jax.jit(jax.value_and_grad(objective))(theta)
    step = jnp.asarray(1.0e-5, dtype=jnp.float64)
    finite_difference = (objective(theta + step) - objective(theta - step)) / (2.0 * step)

    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert abs(float(gradient)) > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-8, atol=1.0e-10)
