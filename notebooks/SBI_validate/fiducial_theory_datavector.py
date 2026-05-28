"""Build the fiducial GODMAX theory datavector for SBI validation."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Dict, Mapping, Tuple

import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.interpolate import RegularGridInterpolator

from gaussian_covariance import (
    TARGET_SPECTRA,
    build_datavector,
    build_gaussian_covariance,
    covariance_quality_checks,
    invert_covariance,
    regularize_covariance,
)
from survey_defaults import SurveyDefaults


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
PASTING_DIR = REPO_ROOT / "notebooks" / "pasting"
OUTPUT_DIR = THIS_DIR / "outputs"
DEFAULT_OUTPUT = OUTPUT_DIR / "fiducial_theory_datavector.npz"
DEFAULT_PAINT_R200C_FACTOR = 8.0


def ensure_repo_paths() -> None:
    """Put GODMAX source and pasting utilities on sys.path."""

    for path in (SRC_DIR, PASTING_DIR, REPO_ROOT):
        spath = str(path)
        if spath not in sys.path:
            sys.path.insert(0, spath)


def lsst_like_source_nz(z: np.ndarray, z0: float = 0.64,
                        alpha: float = 1.5) -> np.ndarray:
    """Return a normalized single-bin LSST-like source n(z)."""

    z = np.asarray(z, dtype=float)
    nz = z ** 2 * np.exp(-((z / z0) ** alpha))
    norm = np.trapz(nz, z)
    return nz / norm if norm > 0 else nz


def configure_kappa_source(analysis_dict: dict, other_params_dict: dict,
                           kappa_source: str) -> None:
    """Configure the source kernel used by get_Cl."""

    if kappa_source == "cmb":
        analysis_dict["is_cmb_lensing"] = True
        analysis_dict["nz_source_info_dict"] = {
            "z_array_source": jnp.ones(1),
            "nbins": 1,
            "nz0": jnp.ones(1),
        }
    elif kappa_source == "lsst":
        z_source = np.linspace(0.01, 3.0, 300)
        analysis_dict["is_cmb_lensing"] = False
        analysis_dict["nz_source_info_dict"] = {
            "z_array_source": z_source,
            "nbins": 1,
            "nz0": lsst_like_source_nz(z_source),
        }
    else:
        raise ValueError("kappa_source must be either 'cmb' or 'lsst'")

    other_params_dict["Delta_z_bias_array"] = jnp.zeros(1)
    other_params_dict["mult_shear_bias_array"] = jnp.zeros(1)


def compute_ne0_cm3(cosmo_params: Mapping[str, float],
                    helium_fraction: float = 0.24) -> float:
    """Mean z=0 electron density in cm^-3."""

    from astropy import constants as const
    import astropy.units as u

    h = cosmo_params["H0"] / 100.0
    rho_crit_0 = 1.878e-29 * h ** 2
    mp = const.m_p.to(u.g).value
    return rho_crit_0 * cosmo_params["Ob0"] * (1.0 - helium_fraction / 2.0) / mp


def angular_galaxy_density_sr(analysis_dict: Mapping[str, np.ndarray],
                              cosmo_jax) -> float:
    """Compute angular galaxy density per steradian from nbar(z)."""

    ensure_repo_paths()
    from paste_backlight_utils import compute_dV_dz_per_sr

    z = np.asarray(analysis_dict["nbar_gal_comoving_zarray"], dtype=float)
    nbar = np.asarray(analysis_dict["nbar_gal_comoving_val"], dtype=float)
    dV_dz = compute_dV_dz_per_sr(cosmo_jax, z)
    return float(np.trapz(nbar * dV_dz, z))


def build_notebook_matched_config(
    gal_zmin: float = 0.4,
    gal_zmax: float = 0.6,
    nbar_comoving: float = 1.0e-4,
    kappa_source: str = "cmb",
):
    """Load the same base config used by paste_backlight_maps_analytic_test."""

    ensure_repo_paths()
    from paste_backlight_utils import build_config, get_project_paths

    paths = get_project_paths()
    config = build_config(
        paths["params"],
        paths["data"],
        nbar_comoving=nbar_comoving,
        gal_zmin=gal_zmin,
        gal_zmax=gal_zmax,
    )
    (
        sim_params_dict,
        halo_params_dict,
        analysis_dict,
        other_params_dict,
        cosmo_jax,
        zarray_lens,
        nz_lens,
        gal_zrange,
    ) = config

    configure_kappa_source(analysis_dict, other_params_dict, kappa_source)
    return (
        sim_params_dict,
        halo_params_dict,
        analysis_dict,
        other_params_dict,
        cosmo_jax,
        zarray_lens,
        nz_lens,
        gal_zrange,
    )


def build_theory_objects(
    gal_zmin: float = 0.4,
    gal_zmax: float = 0.6,
    nbar_comoving: float = 1.0e-4,
    hod_mass_cut: float = 1.0e13,
    kappa_source: str = "cmb",
    remove_galaxy_baryon_suppression: bool = True,
    sim_param_overrides: Mapping[str, float] | None = None,
    other_param_overrides: Mapping[str, float] | None = None,
):
    """Build base/profile/Pk/Cl objects for the fiducial validation point."""

    ensure_repo_paths()
    from base_class import base_class
    from get_radial_profiles import Profiles
    from get_Pkzs import get_Pkz
    from get_Cls import get_Cl

    config = build_notebook_matched_config(
        gal_zmin=gal_zmin,
        gal_zmax=gal_zmax,
        nbar_comoving=nbar_comoving,
        kappa_source=kappa_source,
    )
    (
        sim_params_dict,
        halo_params_dict,
        analysis_dict,
        other_params_dict,
        cosmo_jax,
        zarray_lens,
        nz_lens,
        gal_zrange,
    ) = config

    sim_params_dict = copy.deepcopy(sim_params_dict)
    halo_params_dict = copy.deepcopy(halo_params_dict)
    analysis_dict = copy.deepcopy(analysis_dict)
    other_params_dict = copy.deepcopy(other_params_dict)

    if sim_param_overrides:
        for name, value in sim_param_overrides.items():
            if name.startswith("cosmo."):
                cosmo_name = name.split(".", 1)[1]
                sim_params_dict["cosmo"][cosmo_name] = value
            else:
                sim_params_dict[name] = value

    if other_param_overrides:
        for name, value in other_param_overrides.items():
            other_params_dict[name] = value

    base_obj = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    profiles = Profiles(
        sim_params_dict,
        halo_params_dict,
        analysis_dict,
        other_params_dict,
        base_class_obj=base_obj,
    )

    mass_mask = jnp.where(profiles.M_array > hod_mass_cut, 1.0, 0.0)
    mass_mask_2d = jnp.tile(mass_mask, (halo_params_dict["nz"], 1))
    profiles.Ncen_mat = profiles.Ncen_mat * mass_mask_2d
    profiles.Nsat_mat = profiles.Nsat_mat * mass_mask_2d

    pkz = get_Pkz(
        sim_params_dict,
        halo_params_dict,
        analysis_dict,
        other_params_dict,
        Profiles_obj=profiles,
    )

    if remove_galaxy_baryon_suppression:
        pkz.Pgg_tot_mat = pkz.Pgg_1h_kz_mat + pkz.Pgg_2h_kz_mat
        pkz.Pgm_tot_mat = pkz.Pgm_1h_kz_mat + pkz.Pgm_2h_kz_mat
        pkz.Pgm_nfw_tot_mat = pkz.Pgm_nfw_1h_kz_mat + pkz.Pgm_nfw_2h_kz_mat

    cls = get_Cl(
        sim_params_dict,
        halo_params_dict,
        analysis_dict,
        other_params_dict,
        Pkz_obj=pkz,
    )

    context = {
        "sim_params_dict": sim_params_dict,
        "halo_params_dict": halo_params_dict,
        "analysis_dict": analysis_dict,
        "other_params_dict": other_params_dict,
        "cosmo_jax": cosmo_jax,
        "zarray_lens": zarray_lens,
        "nz_lens": nz_lens,
        "gal_zrange": gal_zrange,
        "base": base_obj,
        "profiles": profiles,
        "pkz": pkz,
        "cls": cls,
        "hod_mass_cut": hod_mass_cut,
        "nbar_comoving": nbar_comoving,
        "kappa_source": kappa_source,
        "remove_galaxy_baryon_suppression": remove_galaxy_baryon_suppression,
        "sim_param_overrides": dict(sim_param_overrides or {}),
        "other_param_overrides": dict(other_param_overrides or {}),
    }
    return context


def _bias_for_probe(pkz, probe: int) -> np.ndarray:
    if probe == 0:
        return np.asarray(pkz.bm_dmb_kz_mat, dtype=float)
    if probe == 1:
        return np.asarray(pkz.bm_nfw_kz_mat, dtype=float)
    if probe == 2:
        return np.asarray(pkz.bg_kz_mat, dtype=float)
    if probe == 3:
        return np.asarray(pkz.by_kz_mat, dtype=float)
    if probe == 4:
        return np.asarray(pkz.be_kz_mat, dtype=float)
    raise ValueError(f"Unknown probe code {probe}")


def _trapz_lnm(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Integrate an array over the last axis in dlnM."""

    return np.trapz(values, x=np.log(np.asarray(mass, dtype=float)), axis=-1)


def _interp_transform_to_kpk(k_src: np.ndarray, uk_src: np.ndarray,
                             k_dst: np.ndarray) -> np.ndarray:
    """Interpolate profile transforms from mcfit k to the halo-model k grid."""

    k_src = np.asarray(k_src, dtype=float)
    k_dst = np.asarray(k_dst, dtype=float)
    uk_src = np.asarray(uk_src, dtype=float)
    out = np.empty((len(k_dst), uk_src.shape[1], uk_src.shape[2]), dtype=float)
    log_k_src = np.log(k_src)
    log_k_dst = np.log(k_dst)
    for iz in range(uk_src.shape[1]):
        for im in range(uk_src.shape[2]):
            vals = np.clip(uk_src[:, iz, im], 1.0e-300, np.inf)
            out[:, iz, im] = np.exp(np.interp(log_k_dst, log_k_src, np.log(vals)))
    return out


def truncated_profile_transforms(pkz, paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR) -> Dict[str, np.ndarray]:
    """Return 3D Fourier profiles truncated at the map-painting radius.

    The pasted maps include only pixels within ``paint_r200c_factor * R200c`` of
    each halo.  This function applies the same support cut in the analytic
    profile transforms.  It is an analytic approximation to the map operation,
    which is a projected-radius cut after line-of-sight projection.
    """

    ensure_repo_paths()
    from mcfitjax.cosmology_jax import xi2P

    r_comoving = np.asarray(pkz.r_array, dtype=float)[:, None, None]
    r200c_comoving = np.asarray(pkz.r200c_mat, dtype=float)[None, :, :]
    mask = r_comoving <= float(paint_r200c_factor) * r200c_comoving
    mtot = np.asarray(pkz.Mtot_mat, dtype=float)

    xi2p = xi2P(pkz.r_array, nx=pkz.nr, lowring=True)

    k_src, uk_y_src = xi2p(jnp.asarray(np.asarray(pkz.y3d_mat, dtype=float) * mask), axis=0, extrap=False)
    _, uk_ne_src = xi2p(jnp.asarray(np.asarray(pkz.ne_mat, dtype=float) * mask), axis=0, extrap=False)
    _, uk_dmb_src = xi2p(
        jnp.asarray(np.asarray(pkz.rho_dmb_mat, dtype=float) * mask / mtot[None, :, :]),
        axis=0,
        extrap=False,
    )

    k_dst = np.asarray(pkz.kPk_array, dtype=float)
    return {
        "uk_y": _interp_transform_to_kpk(k_src, uk_y_src, k_dst),
        "uk_ne": _interp_transform_to_kpk(k_src, uk_ne_src, k_dst),
        "uk_dmb": _interp_transform_to_kpk(k_src, uk_dmb_src, k_dst),
        "paint_r200c_factor": np.asarray(float(paint_r200c_factor)),
    }


def beam_window_for_ell(context: Mapping[str, object]) -> np.ndarray:
    """Return the Gaussian profile-smoothing beam used by the pasted maps."""

    cls = context["cls"]
    sig_beam = float(cls.sig_beam)
    ell = np.asarray(cls.ell_array, dtype=float)
    return np.exp(-0.5 * ell * (ell + 1.0) * sig_beam ** 2)


def single_halo_tau_profile_normalization_check(
    context: Mapping[str, object],
    halo_mass: float = 1.0e14,
    halo_z: float = 0.5,
    paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR,
    nside: int = 512,
) -> Dict[str, float]:
    """Compare map-painter projected tau support to a 3D truncated profile.

    This is a unit diagnostic for the tau normalization path.  A value different
    from unity is expected because the maps use a projected-radius aperture,
    while the map-matched theory uses a 3D support cut as a fast approximation.
    """

    ensure_repo_paths()
    from get_sim_maps import setup_sim_map

    mock_params_setup = {
        "nside": int(nside),
        "get_ymap": False,
        "get_kSZmap": False,
        "get_taumap": True,
        "get_kappamap": False,
        "get_galmap": False,
        "smooth_profiles": True,
    }
    prof = setup_sim_map(
        context["sim_params_dict"],
        context["halo_params_dict"],
        context["analysis_dict"],
        context["other_params_dict"],
        mock_params_setup,
        Profiles_obj=context["profiles"],
    )

    z_grid = np.asarray(prof.z_array, dtype=float)
    m_grid = np.asarray(prof.M_array, dtype=float)
    iz = int(np.argmin(np.abs(z_grid - halo_z)))
    im = int(np.argmin(np.abs(np.log(m_grid) - np.log(halo_mass))))
    z_val = float(z_grid[iz])
    r200c_phys = float(np.asarray(prof.r200c_mat)[iz, im] / (1.0 + z_val))
    rmax_phys = float(paint_r200c_factor) * r200c_phys

    rp = np.asarray(prof.rp_array, dtype=float)
    ne2d = np.asarray(prof.ne2D_mat_physical, dtype=float)[:, iz, im]
    sel_rp = (rp > 0) & (rp <= rmax_phys) & np.isfinite(ne2d)
    projected_aperture = np.nan
    if np.count_nonzero(sel_rp) > 1:
        projected_aperture = float(2.0 * np.pi * np.trapz(rp[sel_rp] * ne2d[sel_rp], rp[sel_rp]))

    r_phys = np.asarray(prof.r_array, dtype=float) / (1.0 + z_val)
    ne3d = np.asarray(prof.ne_mat_physical, dtype=float)[:, iz, im]
    sel_r = (r_phys > 0) & (r_phys <= rmax_phys) & np.isfinite(ne3d)
    spherical_truncated = np.nan
    if np.count_nonzero(sel_r) > 1:
        spherical_truncated = float(4.0 * np.pi * np.trapz(r_phys[sel_r] ** 2 * ne3d[sel_r], r_phys[sel_r]))

    ratio = projected_aperture / spherical_truncated if spherical_truncated > 0 else np.nan
    return {
        "halo_mass_requested": float(halo_mass),
        "halo_z_requested": float(halo_z),
        "halo_mass_grid": float(m_grid[im]),
        "halo_z_grid": z_val,
        "paint_r200c_factor": float(paint_r200c_factor),
        "r200c_physical_mpc_over_h": r200c_phys,
        "projected_aperture_internal": projected_aperture,
        "spherical_truncated_internal": spherical_truncated,
        "projected_over_spherical": float(ratio),
    }


def power_for_probe_pair(pkz, probe1: int, probe2: int) -> np.ndarray:
    """Construct P(k,z) for a probe pair not cached by get_Cl."""

    vmapped = vmap(vmap(lambda jk, jz: pkz.get_P_1h(jk, jz, probe1, probe2),
                        in_axes=(0, None)),
                   in_axes=(None, 0))
    p1h = np.asarray(vmapped(jnp.arange(pkz.nk), jnp.arange(pkz.nz)).T, dtype=float)
    b1 = _bias_for_probe(pkz, probe1)
    b2 = _bias_for_probe(pkz, probe2)
    p2h = b1 * b2 * np.asarray(pkz.plin_kz_mat, dtype=float)
    return p1h + p2h


def build_map_matched_power_components(
    context: Mapping[str, object],
    paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR,
) -> Dict[str, np.ndarray]:
    """Build resolved-halo power spectra matched to the pasted-map support."""

    pkz = context["pkz"]
    tr = truncated_profile_transforms(pkz, paint_r200c_factor=paint_r200c_factor)

    hmf = np.asarray(pkz.hmf_Mz_mat, dtype=float)
    bias = np.asarray(pkz.bias_Mz_mat, dtype=float)
    mass = np.asarray(pkz.M_array, dtype=float)
    mtot = np.asarray(pkz.Mtot_mat, dtype=float)
    plin = np.asarray(pkz.plin_kz_mat, dtype=float)
    rhom0 = float(pkz.rhom_0)

    hmf_3 = hmf[None, :, :]
    bias_3 = bias[None, :, :]
    mtot_3 = mtot[None, :, :]

    ukg = np.asarray(pkz.ukg_cross, dtype=float)
    ukg_auto = np.asarray(pkz.ukg_auto_sqr, dtype=float)
    uk_y = np.asarray(tr["uk_y"], dtype=float)
    uk_ne = np.asarray(tr["uk_ne"], dtype=float)
    uk_dmb = np.asarray(tr["uk_dmb"], dtype=float)

    y_field = uk_y
    # Tau maps are painted from absolute electron number density.  Use the same
    # absolute comoving electron-density transform here; do not apply the legacy
    # GODMAX probe-4 Mtot/rhom0 normalization or any map-derived calibration.
    e_field = uk_ne
    m_field = mtot_3 * uk_dmb / rhom0

    pgg_1h = _trapz_lnm(ukg_auto * hmf_3, mass)
    pgg_2h = np.asarray(pkz.bg_kz_mat, dtype=float) ** 2 * plin

    pgy_1h = _trapz_lnm(ukg * y_field * hmf_3, mass)
    pge_1h = _trapz_lnm(ukg * e_field * hmf_3, mass)
    pgm_1h = _trapz_lnm(ukg * m_field * hmf_3, mass)

    by = _trapz_lnm(y_field * hmf_3 * bias_3, mass)
    be = _trapz_lnm(e_field * hmf_3 * bias_3, mass)
    bm = _trapz_lnm(m_field * hmf_3 * bias_3, mass)
    bg = np.asarray(pkz.bg_kz_mat, dtype=float)

    pgy_2h = bg * by * plin
    pge_2h = bg * be * plin
    pgm_2h = bg * bm * plin

    pyy_1h = _trapz_lnm(y_field * y_field * hmf_3, mass)
    pye_1h = _trapz_lnm(y_field * e_field * hmf_3, mass)
    pee_1h = _trapz_lnm(e_field * e_field * hmf_3, mass)
    pym_1h = _trapz_lnm(y_field * m_field * hmf_3, mass)
    pem_1h = _trapz_lnm(e_field * m_field * hmf_3, mass)
    pmm_1h = _trapz_lnm(m_field * m_field * hmf_3, mass)

    pyy_2h = by * by * plin
    pye_2h = by * be * plin
    pee_2h = be * be * plin
    pym_2h = by * bm * plin
    pem_2h = be * bm * plin
    pmm_2h = bm * bm * plin

    return {
        "Pgg_1h": pgg_1h,
        "Pgg_2h": pgg_2h,
        "Pgg_resolved": pgg_1h + pgg_2h,
        "Pgy_1h": pgy_1h,
        "Pgy_2h": pgy_2h,
        "Pgy_resolved": pgy_1h + pgy_2h,
        "Pge_1h": pge_1h,
        "Pge_2h": pge_2h,
        "Pge_resolved": pge_1h + pge_2h,
        "Pgm_1h": pgm_1h,
        "Pgm_2h": pgm_2h,
        "Pgm_resolved": pgm_1h + pgm_2h,
        "Pyy_resolved": pyy_1h + pyy_2h,
        "Pye_resolved": pye_1h + pye_2h,
        "Pee_resolved": pee_1h + pee_2h,
        "Pym_resolved": pym_1h + pym_2h,
        "Pem_resolved": pem_1h + pem_2h,
        "Pmm_resolved": pmm_1h + pmm_2h,
        "by_resolved": by,
        "be_resolved": be,
        "bm_resolved": bm,
        "paint_r200c_factor": np.asarray(float(paint_r200c_factor)),
    }


def _prefactor_for_probe(cls, probe: int) -> np.ndarray:
    chi = np.asarray(cls.chi_array_for_Cls, dtype=float)
    dchi_dz = np.asarray(cls.dchi_dz_array_for_Cls, dtype=float)
    if probe == 0:
        return np.asarray(cls.Wk_mat[0], dtype=float) / chi ** 2
    if probe == 2:
        return np.asarray(cls.Wg_mat[0], dtype=float) / (dchi_dz * chi ** 2)
    if probe == 3:
        return np.asarray(cls.Wy_array, dtype=float) / chi ** 2
    if probe == 4:
        return np.asarray(cls.Wtau_array, dtype=float) / chi ** 2
    raise ValueError(f"Unsupported projected probe {probe}")


def _absolute_tau_prefactor(cls) -> np.ndarray:
    """Optical-depth kernel for absolute comoving electron density profiles."""

    z_for = np.asarray(cls.z_array_for_Cls, dtype=float)
    chi = np.asarray(cls.chi_array_for_Cls, dtype=float)
    return float(cls.const_coeff_tau) * (1.0 + z_for) ** 2 / chi ** 2


def project_power_to_cl(cls, power_kz: np.ndarray, probe1: int, probe2: int,
                        prefactor1: np.ndarray | None = None,
                        prefactor2: np.ndarray | None = None) -> np.ndarray:
    """Project a custom P(k,z) to C_l using the same Limber convention as get_Cl."""

    ell = np.asarray(cls.ell_array, dtype=float)
    z_grid = np.asarray(cls.z_array, dtype=float)
    z_for = np.asarray(cls.z_array_for_Cls, dtype=float)
    chi = np.asarray(cls.chi_array_for_Cls, dtype=float)
    dchi_dz = np.asarray(cls.dchi_dz_array_for_Cls, dtype=float)
    logk = np.log(np.asarray(cls.kPk_array, dtype=float))

    log_power = np.log(np.clip(np.asarray(power_kz, dtype=float), 1.0e-300, np.inf))
    interp = RegularGridInterpolator(
        (logk, z_grid),
        log_power,
        bounds_error=False,
        fill_value=None,
    )

    pref1 = _prefactor_for_probe(cls, probe1) if prefactor1 is None else np.asarray(prefactor1, dtype=float)
    pref2 = _prefactor_for_probe(cls, probe2) if prefactor2 is None else np.asarray(prefactor2, dtype=float)
    cl = np.empty_like(ell)
    for i, ell_val in enumerate(ell):
        k_for = (ell_val + 0.5) / np.clip(chi, 1.0, np.inf)
        points = np.column_stack([np.log(k_for), z_for])
        pk_for = np.exp(interp(points))
        integrand = pref1 * pref2 * chi ** 2 * dchi_dz * pk_for
        cl[i] = np.trapz(integrand, z_for)
    return cl


def build_full_signal_cls(context: Mapping[str, object],
                          apply_tau_physical_correction: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Build all signal spectra needed by the datavector and covariance."""

    cls = context["cls"]
    pkz = context["pkz"]
    gal_zmin, gal_zmax = context["gal_zrange"]
    z_eff = 0.5 * (gal_zmin + gal_zmax)
    tau_correction = 1.0
    if apply_tau_physical_correction:
        tau_correction = compute_ne0_cm3(context["sim_params_dict"]["cosmo"]) * (1.0 + z_eff) ** 3

    cl_signal = {
        "gg": np.asarray(cls.Cl_gal_gal_tot_mat[:, 0, 0], dtype=float),
        "gy": np.asarray(cls.Cl_gal_y_tot_mat[:, 0], dtype=float),
        "gtau": np.asarray(cls.Cl_gal_tau_tot_mat[:, 0], dtype=float) * tau_correction,
        "gkappa": np.asarray(cls.Cl_gal_kappa_tot_mat[:, 0, 0], dtype=float),
        "kappakappa": np.asarray(cls.Cl_kappa_kappa_tot_mat[:, 0, 0], dtype=float),
        "ykappa": np.asarray(cls.Cl_kappa_y_tot_mat[:, 0], dtype=float),
    }

    yy_power = power_for_probe_pair(pkz, 3, 3)
    ytau_power = power_for_probe_pair(pkz, 3, 4)
    tautau_power = power_for_probe_pair(pkz, 4, 4)
    taukappa_power = power_for_probe_pair(pkz, 0, 4)

    cl_signal["yy"] = project_power_to_cl(cls, yy_power, 3, 3)
    cl_signal["ytau"] = project_power_to_cl(cls, ytau_power, 3, 4) * tau_correction
    cl_signal["tautau"] = project_power_to_cl(cls, tautau_power, 4, 4) * tau_correction ** 2
    cl_signal["taukappa"] = project_power_to_cl(cls, taukappa_power, 4, 0) * tau_correction

    correction_meta = {
        "apply_tau_physical_correction": bool(apply_tau_physical_correction),
        "tau_physical_correction_factor": float(tau_correction),
        "tau_correction_z_eff": float(z_eff),
    }
    return cl_signal, correction_meta


def build_map_matched_signal_cls(
    context: Mapping[str, object],
    apply_tau_physical_correction: bool = True,
    paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Build resolved-halo spectra matched to the pasted-map validation.

    The target ``gg/gy/gtau/gkappa`` vector is the resolved branch.  Full-theory
    spectra are also saved under ``*_full`` keys for diagnostics.
    """

    full_signal, full_correction_meta = build_full_signal_cls(
        context,
        apply_tau_physical_correction=apply_tau_physical_correction,
    )
    cls = context["cls"]

    components = build_map_matched_power_components(
        context,
        paint_r200c_factor=paint_r200c_factor,
    )
    beam = beam_window_for_ell(context)
    tau_prefactor = _absolute_tau_prefactor(cls)

    gg_1h = project_power_to_cl(cls, components["Pgg_1h"], 2, 2)
    gg_2h = project_power_to_cl(cls, components["Pgg_2h"], 2, 2)
    gy_1h = project_power_to_cl(cls, components["Pgy_1h"], 2, 3) * beam
    gy_2h = project_power_to_cl(cls, components["Pgy_2h"], 2, 3) * beam
    gtau_1h = project_power_to_cl(
        cls, components["Pge_1h"], 2, 4, prefactor2=tau_prefactor
    ) * beam
    gtau_2h = project_power_to_cl(
        cls, components["Pge_2h"], 2, 4, prefactor2=tau_prefactor
    ) * beam
    gkappa_1h = project_power_to_cl(cls, components["Pgm_1h"], 2, 0) * beam
    gkappa_2h = project_power_to_cl(cls, components["Pgm_2h"], 2, 0) * beam

    yy = project_power_to_cl(cls, components["Pyy_resolved"], 3, 3) * beam ** 2
    ytau = project_power_to_cl(
        cls, components["Pye_resolved"], 3, 4, prefactor2=tau_prefactor
    ) * beam ** 2
    tautau = project_power_to_cl(
        cls, components["Pee_resolved"], 4, 4,
        prefactor1=tau_prefactor,
        prefactor2=tau_prefactor,
    ) * beam ** 2
    ykappa = project_power_to_cl(cls, components["Pym_resolved"], 3, 0) * beam ** 2
    taukappa = project_power_to_cl(
        cls, components["Pem_resolved"], 4, 0, prefactor1=tau_prefactor
    ) * beam ** 2
    kappakappa = project_power_to_cl(cls, components["Pmm_resolved"], 0, 0) * beam ** 2

    gtau = gtau_1h + gtau_2h

    cl_signal = dict(full_signal)
    for key, value in full_signal.items():
        cl_signal[f"{key}_full"] = value

    cl_signal.update({
        "gg": gg_1h + gg_2h,
        "gy": gy_1h + gy_2h,
        "gtau": gtau,
        "gkappa": gkappa_1h + gkappa_2h,
        "yy": yy,
        "ytau": ytau,
        "tautau": tautau,
        "ykappa": ykappa,
        "taukappa": taukappa,
        "kappakappa": kappakappa,
        "gg_1h_resolved_8r200c": gg_1h,
        "gg_2h_resolved_8r200c": gg_2h,
        "gy_1h_resolved_8r200c": gy_1h,
        "gy_2h_resolved_8r200c": gy_2h,
        "gy_resolved_8r200c": gy_1h + gy_2h,
        "gtau_1h_resolved_8r200c": gtau_1h,
        "gtau_2h_resolved_8r200c": gtau_2h,
        "gtau_resolved_8r200c": gtau,
        "gkappa_1h_resolved_8r200c": gkappa_1h,
        "gkappa_2h_resolved_8r200c": gkappa_2h,
        "gkappa_resolved_8r200c": gkappa_1h + gkappa_2h,
        "beam_window": beam,
    })

    correction_meta = {
        "theory_mode": "map_matched_resolved",
        "paint_r200c_factor": float(paint_r200c_factor),
        "resolved_tau_unit_convention": (
            "absolute comoving electron number density from truncated uk_ne"
        ),
        "resolved_tau_kernel": (
            "W_tau = sigma_T * (Mpc/h in cm) * (1 + z)^2 for absolute comoving ne"
        ),
        "map_derived_calibrations_applied": False,
        "full_theory_diagnostic_corrections": full_correction_meta,
        "profile_truncation_note": (
            "Analytic profiles are truncated at the same 8 R200c support used "
            "by the pasted-map pixel finder. This is a 3D support approximation "
            "to the map's projected-radius cut."
        ),
        "missing_field_note": (
            "The target gtau spectrum excludes the be_kz low-mass/missing-electron "
            "2-halo completion. Full-theory spectra are retained under *_full "
            "for diagnostics and are not used as the SBI target."
        ),
    }
    return cl_signal, correction_meta


def build_signal_cls(
    context: Mapping[str, object],
    apply_tau_physical_correction: bool = True,
    theory_mode: str = "map_matched_resolved",
    paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Build signal spectra for the requested theory mode."""

    if theory_mode == "full":
        cl_signal, meta = build_full_signal_cls(
            context,
            apply_tau_physical_correction=apply_tau_physical_correction,
        )
        meta["theory_mode"] = "full"
        return cl_signal, meta
    if theory_mode == "map_matched_resolved":
        return build_map_matched_signal_cls(
            context,
            apply_tau_physical_correction=apply_tau_physical_correction,
            paint_r200c_factor=paint_r200c_factor,
        )
    raise ValueError("theory_mode must be 'full' or 'map_matched_resolved'")


def save_validation_product(path: pathlib.Path, ell: np.ndarray, delta_ell: np.ndarray,
                            data_vector: np.ndarray, cov: np.ndarray, corr: np.ndarray,
                            precision: np.ndarray, labels, cl_signal, noise,
                            metadata: Mapping[str, object]) -> None:
    """Save the full theory/covariance product as an npz file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ell": np.asarray(ell, dtype=float),
        "delta_ell": np.asarray(delta_ell, dtype=float),
        "data_vector": np.asarray(data_vector, dtype=float),
        "cov": np.asarray(cov, dtype=float),
        "corr": np.asarray(corr, dtype=float),
        "precision": np.asarray(precision, dtype=float),
        "labels": np.asarray(labels, dtype=object),
        "spectra_order": np.asarray(TARGET_SPECTRA, dtype=object),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    }
    for key, value in cl_signal.items():
        payload[f"cl_{key}"] = np.asarray(value, dtype=float)
    for key, value in noise.items():
        payload[f"noise_{key}"] = np.asarray(value, dtype=float)
    np.savez_compressed(path, **payload)


def build_and_save_fiducial(
    output_path: pathlib.Path | str = DEFAULT_OUTPUT,
    gal_zmin: float = 0.4,
    gal_zmax: float = 0.6,
    nbar_comoving: float = 1.0e-4,
    hod_mass_cut: float = 1.0e13,
    kappa_source: str = "cmb",
    apply_tau_physical_correction: bool = True,
    theory_mode: str = "map_matched_resolved",
    paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR,
    sim_param_overrides: Mapping[str, float] | None = None,
    other_param_overrides: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Build theory, covariance, and save the validation product."""

    context = build_theory_objects(
        gal_zmin=gal_zmin,
        gal_zmax=gal_zmax,
        nbar_comoving=nbar_comoving,
        hod_mass_cut=hod_mass_cut,
        kappa_source=kappa_source,
        sim_param_overrides=sim_param_overrides,
        other_param_overrides=other_param_overrides,
    )
    cls = context["cls"]
    ell = np.asarray(cls.ell_array, dtype=float)
    delta_ell = np.asarray(context["analysis_dict"]["dl_array_survey"], dtype=float)
    nbar_gal_sr = angular_galaxy_density_sr(context["analysis_dict"], context["cosmo_jax"])
    cl_signal, correction_meta = build_signal_cls(
        context,
        apply_tau_physical_correction=apply_tau_physical_correction,
        theory_mode=theory_mode,
        paint_r200c_factor=paint_r200c_factor,
    )
    data_vector, labels = build_datavector(cl_signal, TARGET_SPECTRA)
    survey = SurveyDefaults(
        beam_fwhm_arcmin=float(context["analysis_dict"].get("beam_fwhm_arcmin", 6.87))
    )
    cov, corr, noise, cov_meta = build_gaussian_covariance(
        ell,
        delta_ell,
        cl_signal,
        nbar_gal_sr=nbar_gal_sr,
        survey=survey,
        spectra_order=TARGET_SPECTRA,
    )
    cov, jitter = regularize_covariance(cov)
    diag = np.diag(cov)
    corr = cov / np.sqrt(np.clip(np.outer(diag, diag), 1.0e-300, np.inf))
    precision, _ = invert_covariance(cov)
    checks = covariance_quality_checks(cov)
    metadata = {
        "repo_root": str(REPO_ROOT),
        "gal_zmin": float(gal_zmin),
        "gal_zmax": float(gal_zmax),
        "nbar_comoving": float(nbar_comoving),
        "hod_mass_cut": float(hod_mass_cut),
        "kappa_source": kappa_source,
        "ell_grid": "paste_backlight_utils.build_config",
        "data_vector_order": list(TARGET_SPECTRA),
        "covariance": cov_meta,
        "corrections": correction_meta,
        "theory_mode": theory_mode,
        "paint_r200c_factor": float(paint_r200c_factor),
        "sim_param_overrides": {
            key: float(value) for key, value in (sim_param_overrides or {}).items()
        },
        "other_param_overrides": {
            key: float(value) for key, value in (other_param_overrides or {}).items()
        },
        "map_derived_calibrations_applied": False,
        "precision_jitter_added": float(jitter),
        "quality_checks": checks,
        "remove_galaxy_baryon_suppression": bool(context["remove_galaxy_baryon_suppression"]),
    }

    output_path = pathlib.Path(output_path)
    save_validation_product(
        output_path,
        ell,
        delta_ell,
        data_vector,
        cov,
        corr,
        precision,
        labels,
        cl_signal,
        noise,
        metadata,
    )
    return {
        "output_path": output_path,
        "ell": ell,
        "delta_ell": delta_ell,
        "data_vector": data_vector,
        "cov": cov,
        "corr": corr,
        "precision": precision,
        "labels": labels,
        "cl_signal": cl_signal,
        "noise": noise,
        "metadata": metadata,
    }


def parse_key_value_overrides(values: list[str] | None) -> Dict[str, float]:
    """Parse command-line overrides of the form ``name=value``."""

    overrides: Dict[str, float] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Override must have the form name=value: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = float(value)
    return overrides


def load_validation_product(path: pathlib.Path | str = DEFAULT_OUTPUT) -> Dict[str, object]:
    """Load a saved fiducial theory product."""

    path = pathlib.Path(path)
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    cl_signal = {
        key[3:]: data[key]
        for key in data.files
        if key.startswith("cl_")
    }
    noise = {
        key[6:]: data[key]
        for key in data.files
        if key.startswith("noise_")
    }
    return {
        "path": path,
        "ell": data["ell"],
        "delta_ell": data["delta_ell"],
        "data_vector": data["data_vector"],
        "cov": data["cov"],
        "corr": data["corr"],
        "precision": data["precision"],
        "labels": data["labels"].tolist(),
        "spectra_order": data["spectra_order"].tolist(),
        "metadata": metadata,
        "cl_signal": cl_signal,
        "noise": noise,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--kappa-source", choices=("cmb", "lsst"), default="cmb")
    parser.add_argument("--gal-zmin", type=float, default=0.4)
    parser.add_argument("--gal-zmax", type=float, default=0.6)
    parser.add_argument("--nbar-comoving", type=float, default=1.0e-4)
    parser.add_argument("--hod-mass-cut", type=float, default=1.0e13)
    parser.add_argument("--theory-mode", choices=("full", "map_matched_resolved"),
                        default="map_matched_resolved")
    parser.add_argument("--paint-r200c-factor", type=float, default=DEFAULT_PAINT_R200C_FACTOR)
    parser.add_argument("--sim-param", action="append", default=[],
                        help="Override a simulation parameter as name=value. Use cosmo.NAME for cosmology.")
    parser.add_argument("--other-param", action="append", default=[],
                        help="Override an other_params_dict entry as name=value.")
    args = parser.parse_args()

    result = build_and_save_fiducial(
        output_path=args.output,
        gal_zmin=args.gal_zmin,
        gal_zmax=args.gal_zmax,
        nbar_comoving=args.nbar_comoving,
        hod_mass_cut=args.hod_mass_cut,
        kappa_source=args.kappa_source,
        theory_mode=args.theory_mode,
        paint_r200c_factor=args.paint_r200c_factor,
        sim_param_overrides=parse_key_value_overrides(args.sim_param),
        other_param_overrides=parse_key_value_overrides(args.other_param),
    )
    print(f"Saved fiducial theory product to {result['output_path']}")


if __name__ == "__main__":
    main()
