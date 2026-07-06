"""Diagnostics for xDESI Abacus pasted galaxy clustering.

This file is local to ``notebooks/xDESI``.  It only reads existing catalogs and
maps; cache/output paths are constructed by the notebook under the xDESI output
tree.
"""

from __future__ import annotations

import copy
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import healpy as hp
import numpy as np
from astropy import constants as astro_const
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import eval_legendre
from scipy.spatial import cKDTree
from scipy.stats import poisson

import abacus_pasting_helpers as aph


@dataclass
class TheoryBundle:
    name: str
    sim_params: dict
    halo_params: dict
    analysis: dict
    other_params: dict
    base: object
    profiles: object
    pkz: object
    cls: object


def normalize_nz(z: np.ndarray, nz: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    nz = np.nan_to_num(np.asarray(nz, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    nz = np.maximum(nz, float(floor))
    norm = np.trapezoid(nz, z)
    return nz / norm if norm > 0 else nz


def clone_lens_info_with_bin(
    lens_info: Mapping[str, object],
    compare_bin: int,
    z: np.ndarray,
    nz: np.ndarray,
    *,
    z_edges: Optional[np.ndarray] = None,
    floor: float = 0.0,
) -> dict:
    """Clone GODMAX lens n(z) info and replace one bin."""

    out = copy.deepcopy(dict(lens_info))
    out["z_array_lens"] = np.asarray(z, dtype=np.float64)
    out[f"nz{compare_bin}"] = normalize_nz(z, nz, floor=floor)
    if z_edges is not None:
        edges = np.asarray(out["z_edges_bins_lens"], dtype=np.float64).copy()
        edges[int(compare_bin)] = np.asarray(z_edges, dtype=np.float64)
        out["z_edges_bins_lens"] = edges
    return out


def measured_galaxy_nz(
    galaxies: np.ndarray,
    z_grid: np.ndarray,
    *,
    z_range: Optional[Tuple[float, float]] = None,
    central_flag: Optional[bool] = None,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Histogram generated galaxies onto the requested z grid."""

    valid = galaxies[:, 5] > 0.5
    if z_range is not None:
        valid &= (galaxies[:, 2] >= z_range[0]) & (galaxies[:, 2] < z_range[1])
    if central_flag is not None:
        valid &= (galaxies[:, 4] > 0.5) if central_flag else (galaxies[:, 4] < 0.5)
    z = np.asarray(galaxies[valid, 2], dtype=np.float64)
    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=np.float64)[valid]

    dz = np.median(np.diff(z_grid))
    edges = np.concatenate(([z_grid[0] - 0.5 * dz], 0.5 * (z_grid[1:] + z_grid[:-1]), [z_grid[-1] + 0.5 * dz]))
    hist, _ = np.histogram(z, bins=edges, weights=w)
    nz = normalize_nz(z_grid, hist, floor=0.0)
    return {"z": z_grid, "nz": nz, "count": int(len(z)), "hist": hist, "edges": edges}


def make_tophat_nz(z_grid: np.ndarray, z_range: Tuple[float, float]) -> np.ndarray:
    nz = ((z_grid >= z_range[0]) & (z_grid < z_range[1])).astype(np.float64)
    return normalize_nz(z_grid, nz)


def reweight_to_target_nz(
    galaxies: np.ndarray,
    z_grid: np.ndarray,
    target_nz: np.ndarray,
    *,
    max_weight: float = 20.0,
) -> np.ndarray:
    """Return per-galaxy weights that approximately transform measured dN/dz to target dN/dz."""

    measured = measured_galaxy_nz(galaxies, z_grid)["nz"]
    measured_at_gal = np.interp(galaxies[:, 2], z_grid, measured, left=0.0, right=0.0)
    target_at_gal = np.interp(galaxies[:, 2], z_grid, normalize_nz(z_grid, target_nz), left=0.0, right=0.0)
    weights = np.zeros(len(galaxies), dtype=np.float64)
    mask = (galaxies[:, 5] > 0.5) & (measured_at_gal > 0)
    weights[mask] = target_at_gal[mask] / measured_at_gal[mask]
    weights = np.clip(weights, 0.0, float(max_weight))
    return weights


def make_weighted_delta_map(
    galaxies: np.ndarray,
    nside: int,
    *,
    weights: Optional[np.ndarray] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Make a weighted galaxy overdensity map and its Poisson shot-noise level."""

    valid = galaxies[:, 5] > 0.5
    if z_range is not None:
        valid &= (galaxies[:, 2] >= z_range[0]) & (galaxies[:, 2] < z_range[1])
    if weights is None:
        w = np.ones(np.count_nonzero(valid), dtype=np.float64)
    else:
        w_all = np.asarray(weights, dtype=np.float64)
        valid &= w_all > 0
        w = w_all[valid]

    gals = galaxies[valid]
    npix = hp.nside2npix(nside)
    count_map = np.zeros(npix, dtype=np.float64)
    if len(gals):
        pix = hp.ang2pix(nside, gals[:, 0], gals[:, 1], lonlat=True)
        np.add.at(count_map, pix, w)
    mean = float(np.mean(count_map))
    delta = count_map / mean - 1.0 if mean > 0 else count_map
    sumw = float(np.sum(w))
    sumw2 = float(np.sum(w**2))
    shot = 4.0 * math.pi * sumw2 / max(sumw**2, 1.0e-30)
    return {
        "delta": delta.astype(np.float32),
        "count_map": count_map.astype(np.float32),
        "n_gal": int(len(gals)),
        "sumw": sumw,
        "sumw2": sumw2,
        "mean_per_pix": mean,
        "shot": shot,
    }


def measure_weighted_cls(
    maps: Mapping[str, np.ndarray],
    galaxies: np.ndarray,
    nside: int,
    *,
    weights: Optional[np.ndarray] = None,
    z_range: Optional[Tuple[float, float]] = None,
    pixwin_correction: bool = True,
) -> dict:
    """Measure gg and galaxy-field cross-spectra for a weighted galaxy map."""

    lmax = 3 * int(nside) - 1
    g = make_weighted_delta_map(galaxies, nside, weights=weights, z_range=z_range)
    pixwin2 = np.ones(lmax + 1, dtype=np.float64)
    if pixwin_correction:
        pixwin2 = np.maximum(hp.pixwin(nside, lmax=lmax) ** 2, 1.0e-30)
    out = {
        "ell": np.arange(lmax + 1),
        "delta_g": g["delta"],
        "n_gal": g["n_gal"],
        "sumw": g["sumw"],
        "sumw2": g["sumw2"],
        "mean_g_per_pix": g["mean_per_pix"],
        "shot_gg": g["shot"],
    }
    gg_raw = hp.anafast(g["delta"], lmax=lmax)
    gg_with = gg_raw / pixwin2
    shot_deconvolved = g["shot"] / pixwin2
    gg_without = (gg_raw - g["shot"]) / pixwin2
    if not pixwin_correction:
        shot_deconvolved = np.full(lmax + 1, g["shot"], dtype=np.float64)
    out["gg_with_shot"] = gg_with
    out["gg_without_shot"] = gg_without
    out["gg"] = out["gg_without_shot"]
    for field, key in (
        ("gy", "map_ymap"),
        ("gtau", "map_tau"),
        ("gksz", "map_ksz"),
        ("gkappa_cmb", "map_kappa_cmb"),
        ("gkappa_wl", "map_kappa_wl"),
    ):
        if key in maps:
            out[field] = hp.anafast(g["delta"], np.asarray(maps[key], dtype=np.float32), lmax=lmax) / pixwin2
    out["gg_raw"] = gg_raw
    out["shot_gg_deconvolved"] = shot_deconvolved
    out["pixwin2"] = pixwin2
    return out


def hod_expectation_weights(catalog: Mapping[str, np.ndarray], bundle: TheoryBundle) -> dict:
    """Interpolate GODMAX HOD expectation values onto the input halo catalog."""

    z = np.asarray(catalog["z"], dtype=np.float64)
    logm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)
    ncen = np.nan_to_num(_interp_profile_matrix(bundle, bundle.profiles.Ncen_mat, z, logm), nan=0.0, posinf=0.0, neginf=0.0)
    nsat = np.nan_to_num(_interp_profile_matrix(bundle, bundle.profiles.Nsat_mat, z, logm), nan=0.0, posinf=0.0, neginf=0.0)
    ncen = np.clip(ncen, 0.0, 1.0)
    nsat = np.maximum(nsat, 0.0)
    return {"ncen": ncen, "nsat": nsat, "ntot": ncen + nsat}


def make_catalog_weighted_delta_map(
    catalog: Mapping[str, np.ndarray],
    nside: int,
    weights: np.ndarray,
    *,
    z_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Make a weighted halo-catalog overdensity map with a weighted point self-term."""

    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0.0)
    if z_range is not None:
        z = np.asarray(catalog["z"], dtype=np.float64)
        valid &= (z >= z_range[0]) & (z < z_range[1])

    npix = hp.nside2npix(nside)
    count_map = np.zeros(npix, dtype=np.float64)
    if np.any(valid):
        pix = hp.ang2pix(
            int(nside),
            np.asarray(catalog["ra_deg"], dtype=np.float64)[valid],
            np.asarray(catalog["dec_deg"], dtype=np.float64)[valid],
            lonlat=True,
        )
        np.add.at(count_map, pix, weights[valid])

    mean = float(np.mean(count_map))
    delta = count_map / mean - 1.0 if mean > 0.0 else count_map
    sumw = float(np.sum(weights[valid]))
    sumw2 = float(np.sum(weights[valid] ** 2))
    shot = 4.0 * math.pi * sumw2 / max(sumw**2, 1.0e-30)
    return {
        "delta": delta.astype(np.float32),
        "count_map": count_map.astype(np.float32),
        "n_obj": int(np.count_nonzero(valid)),
        "sumw": sumw,
        "sumw2": sumw2,
        "mean_per_pix": mean,
        "shot": shot,
    }


def measure_catalog_weighted_cls(
    catalog: Mapping[str, np.ndarray],
    nside: int,
    weights: np.ndarray,
    *,
    z_range: Optional[Tuple[float, float]] = None,
    lmax: Optional[int] = None,
    pixwin_correction: bool = True,
) -> dict:
    """Measure an auto-spectrum for a deterministic weighted halo-catalog map."""

    if lmax is None:
        lmax = 3 * int(nside) - 1
    else:
        lmax = min(int(lmax), 3 * int(nside) - 1)
    g = make_catalog_weighted_delta_map(catalog, nside, weights, z_range=z_range)
    pixwin2 = np.ones(lmax + 1, dtype=np.float64)
    if pixwin_correction:
        pixwin2 = np.maximum(hp.pixwin(nside, lmax=lmax) ** 2, 1.0e-30)
    raw = hp.anafast(g["delta"], lmax=lmax)
    shot_deconvolved = g["shot"] / pixwin2
    out = {
        "ell": np.arange(lmax + 1),
        "delta": g["delta"],
        "weighted_map": g["count_map"],
        "n_obj": g["n_obj"],
        "sumw": g["sumw"],
        "sumw2": g["sumw2"],
        "mean_per_pix": g["mean_per_pix"],
        "shot": g["shot"],
        "shot_deconvolved": shot_deconvolved,
        "cl_raw": raw,
        "cl_with_shot": raw / pixwin2,
        "cl_without_shot": (raw - g["shot"]) / pixwin2,
        "pixwin2": pixwin2,
    }
    return out


def measure_catalog_weighted_cross_cls(
    catalog: Mapping[str, np.ndarray],
    nside: int,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    *,
    z_range: Optional[Tuple[float, float]] = None,
    lmax: Optional[int] = None,
    pixwin_correction: bool = True,
) -> dict:
    """Measure a weighted parent-halo cross-spectrum with point self-term subtraction."""

    if lmax is None:
        lmax = 3 * int(nside) - 1
    else:
        lmax = min(int(lmax), 3 * int(nside) - 1)
    ga = make_catalog_weighted_delta_map(catalog, nside, weights_a, z_range=z_range)
    gb = make_catalog_weighted_delta_map(catalog, nside, weights_b, z_range=z_range)
    pixwin2 = np.ones(lmax + 1, dtype=np.float64)
    if pixwin_correction:
        pixwin2 = np.maximum(hp.pixwin(nside, lmax=lmax) ** 2, 1.0e-30)

    wa = np.asarray(weights_a, dtype=np.float64)
    wb = np.asarray(weights_b, dtype=np.float64)
    valid = np.isfinite(wa) & np.isfinite(wb) & (wa > 0.0) & (wb > 0.0)
    if z_range is not None:
        z = np.asarray(catalog["z"], dtype=np.float64)
        valid &= (z >= z_range[0]) & (z < z_range[1])
    shot = 4.0 * math.pi * float(np.sum(wa[valid] * wb[valid])) / max(ga["sumw"] * gb["sumw"], 1.0e-30)
    raw = hp.anafast(ga["delta"], gb["delta"], lmax=lmax)
    return {
        "ell": np.arange(lmax + 1),
        "delta_a": ga["delta"],
        "delta_b": gb["delta"],
        "sumw_a": ga["sumw"],
        "sumw_b": gb["sumw"],
        "n_obj_a": ga["n_obj"],
        "n_obj_b": gb["n_obj"],
        "shot": shot,
        "shot_deconvolved": shot / pixwin2,
        "cl_raw": raw,
        "cl_with_shot": raw / pixwin2,
        "cl_without_shot": (raw - shot) / pixwin2,
        "pixwin2": pixwin2,
    }


def gaussian_ratio_error(ell: np.ndarray, *, delta_ell: int, fsky: float) -> np.ndarray:
    """Approximate fractional Gaussian auto-spectrum error for binned ratios."""

    ell = np.asarray(ell, dtype=np.float64)
    denom = np.maximum((2.0 * ell + 1.0) * max(float(delta_ell), 1.0) * max(float(fsky), 1.0e-6), 1.0)
    return np.sqrt(2.0 / denom)


def sky_coverage_by_nside(ra_deg: np.ndarray, dec_deg: np.ndarray, nsides: Sequence[int] = (8, 16, 32, 64)) -> list:
    rows = []
    for nside in nsides:
        pix = hp.ang2pix(int(nside), ra_deg, dec_deg, lonlat=True)
        rows.append({"nside": int(nside), "n_occupied": int(len(np.unique(pix))), "npix": int(hp.nside2npix(nside)), "fsky_occupied": float(len(np.unique(pix)) / hp.nside2npix(nside))})
    return rows


def flat_lcdm_comoving_distance_hmpc(z: np.ndarray, omega_m: float, *, n_grid: int = 8192) -> np.ndarray:
    """Comoving distance in Mpc/h for a flat LCDM cosmology."""

    z = np.asarray(z, dtype=np.float64)
    zmax = max(float(np.nanmax(z)), 1.0e-6)
    grid = np.linspace(0.0, zmax, int(n_grid))
    ez = np.sqrt(float(omega_m) * (1.0 + grid) ** 3 + (1.0 - float(omega_m)))
    integ = np.zeros_like(grid)
    integ[1:] = np.cumsum(0.5 * (1.0 / ez[1:] + 1.0 / ez[:-1]) * np.diff(grid))
    return (astro_const.c.to("km/s").value / 100.0) * np.interp(z, grid, integ)


def full_sky_shell_volume_hmpc3(z_lo: float, z_hi: float, omega_m: float, *, fsky: float = 1.0) -> float:
    chi = flat_lcdm_comoving_distance_hmpc(np.asarray([z_lo, z_hi]), omega_m)
    return float(fsky * (4.0 * math.pi / 3.0) * abs(chi[1] ** 3 - chi[0] ** 3))


def catalog_hmf_points(
    catalog: Mapping[str, np.ndarray],
    *,
    z_slices: Sequence[Tuple[float, float]],
    logm_edges: np.ndarray,
    omega_m: float,
    fsky: float = 1.0,
) -> list:
    """Return catalog dn/dlnM points in M200c_hMsun bins."""

    z = np.asarray(catalog["z"], dtype=np.float64)
    logm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)
    rows = []
    for zlo, zhi in z_slices:
        volume = full_sky_shell_volume_hmpc3(zlo, zhi, omega_m, fsky=fsky)
        for mlo, mhi in zip(logm_edges[:-1], logm_edges[1:]):
            mask = (z >= zlo) & (z < zhi) & (logm >= mlo) & (logm < mhi)
            count = int(np.count_nonzero(mask))
            dlnm = math.log(10.0) * (float(mhi) - float(mlo))
            rows.append(
                {
                    "z_lo": float(zlo),
                    "z_hi": float(zhi),
                    "z_mid": 0.5 * (float(zlo) + float(zhi)),
                    "logM_lo": float(mlo),
                    "logM_hi": float(mhi),
                    "logM_mid": 0.5 * (float(mlo) + float(mhi)),
                    "count": count,
                    "volume_hMpc3": volume,
                    "dndlnM": count / max(volume * dlnm, 1.0e-300),
                    "poisson_err": math.sqrt(count) / max(volume * dlnm, 1.0e-300) if count > 0 else np.nan,
                    "mass_definition": "M200c_hMsun",
                    "hmf_units": "(h/Mpc)^3 per dlnM",
                }
            )
    return rows


def theory_hmf_curve(bundle: TheoryBundle, z_eval: float) -> dict:
    """Interpolate GODMAX dn/dlnM theory at one redshift."""

    z_grid = np.asarray(bundle.base.z_array, dtype=np.float64)
    m_grid = np.asarray(bundle.base.M_array, dtype=np.float64)
    hmf = np.asarray(bundle.pkz.hmf_Mz_mat, dtype=np.float64)
    hmf_z = np.asarray([np.interp(float(z_eval), z_grid, hmf[:, im]) for im in range(len(m_grid))])
    return {
        "z": float(z_eval),
        "M200c_hMsun": m_grid,
        "log10M200c_hMsun": np.log10(m_grid),
        "dndlnM": hmf_z,
        "mass_definition": f"M{int(getattr(bundle.base, 'mdef_Delta', 200))}c_hMsun",
        "hmf_model": getattr(bundle.base, "hmf_model", ""),
        "hmf_units": "(h/Mpc)^3 per dlnM",
    }


def build_theory_bundle(
    config_path: Path | str,
    catalog_key: str,
    *,
    name: str,
    lens_info: Optional[Mapping[str, object]] = None,
    is_cmb_lensing: bool = True,
    symbolic: bool = False,
    z_max: float = 0.5,
    log10_mass_min: float = 14.0,
    include_ia: bool = False,
) -> TheoryBundle:
    """Build GODMAX theory objects with optional lens n(z) and HMF/PK mode overrides."""

    from base_class import base_class
    from get_Cls import get_Cl
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    config = aph.load_config(config_path)
    _, attrs = aph.load_halo_catalog(aph.catalog_path(config, catalog_key), indices=np.asarray([], dtype=np.int64))
    sim_params, halo_params, analysis, other_params = aph.prepare_godmax_config(
        config,
        attrs,
        is_cmb_lensing=is_cmb_lensing,
        z_max=z_max,
        log10_mass_min=log10_mass_min,
    )
    if lens_info is not None:
        analysis["nz_lens_info_dict"] = copy.deepcopy(dict(lens_info))
    analysis["symbolic_pk"] = bool(symbolic)
    analysis["symbolic_hmf"] = bool(symbolic)
    if not include_ia:
        other_params["A_IA"] = 0.0
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    aph.apply_hod_mass_cut(profiles, log10_mass_min)
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)
    cls = get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)
    return TheoryBundle(name, sim_params, halo_params, analysis, other_params, base, profiles, pkz, cls)


def extract_theory_curves(bundle_cmb: TheoryBundle, bundle_wl: Optional[TheoryBundle], compare_bin: int, z_mean: float) -> dict:
    cls_cmb = bundle_cmb.cls
    curves = {
        "ell": np.asarray(cls_cmb.ell_array, dtype=np.float64),
        "gg": np.asarray(cls_cmb.Cl_gal_gal_tot_mat[:, compare_bin, compare_bin], dtype=np.float64),
        "gy": np.asarray(cls_cmb.Cl_gal_y_tot_mat[:, compare_bin], dtype=np.float64),
        "gtau": np.asarray(cls_cmb.Cl_gal_tau_tot_mat[:, compare_bin], dtype=np.float64)
        * aph.tau_theory_conversion(cls_cmb.cosmo_params, z_mean),
        "gkappa_cmb": np.asarray(cls_cmb.Cl_gal_kappa_tot_mat[:, compare_bin, 0], dtype=np.float64),
    }
    if bundle_wl is not None:
        curves["gkappa_wl"] = np.asarray(bundle_wl.cls.Cl_gal_kappa_tot_mat[:, compare_bin, 0], dtype=np.float64)
    return curves


def bin_pair(ell: np.ndarray, measured: np.ndarray, theory: np.ndarray, *, ell_min: int, ell_max: int, delta_ell: int) -> dict:
    import abacus_particle_shell_helpers as psh

    return psh.bin_spectrum_pair(ell, measured, theory, ell_min=ell_min, ell_max=ell_max, delta_ell=delta_ell)


def load_fit_clgg_reference(config: Mapping[str, object], compare_bin: int) -> dict:
    with open(config["godmax"]["xdesi_fit_summary"], "rb") as handle:
        fit = pickle.load(handle)
    z_group = fit["zvals"][int(compare_bin)]
    key = f"z{z_group[0]:.3f}_{z_group[-1]:.3f}"
    cl = fit["Cl_gg_all"]
    out = {
        "key": key,
        "ell": np.asarray(cl["l_array"], dtype=np.float64),
        "cl": np.asarray(cl[key], dtype=np.float64),
    }
    if "Cl_gg_all_std" in fit and key in fit["Cl_gg_all_std"]:
        out["std"] = np.asarray(fit["Cl_gg_all_std"][key], dtype=np.float64)
    return out


def infer_catalog_fsky(ra_deg: np.ndarray, dec_deg: np.ndarray, *, nside: int = 64) -> float:
    pix = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True)
    return float(len(np.unique(pix)) / hp.nside2npix(nside))


def hmf_count_comparison(
    catalog: Mapping[str, np.ndarray],
    bundle: TheoryBundle,
    *,
    z_edges: np.ndarray,
    logm_edges: np.ndarray,
    fsky: Optional[float] = None,
) -> dict:
    """Compare observed halo counts to theory HMF counts in z/logM bins."""

    if fsky is None:
        fsky = infer_catalog_fsky(catalog["ra_deg"], catalog["dec_deg"])
    z = np.asarray(catalog["z"], dtype=np.float64)
    logm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)
    observed, _, _ = np.histogram2d(z, logm, bins=[z_edges, logm_edges])

    z_grid = np.asarray(bundle.base.z_array, dtype=np.float64)
    m_grid = np.asarray(bundle.base.M_array, dtype=np.float64)
    logm_grid = np.log10(m_grid)
    hmf = np.asarray(bundle.pkz.hmf_Mz_mat, dtype=np.float64)
    chi_grid = np.asarray(bundle.base.chi_array, dtype=np.float64)
    chi_of_z = interp1d(z_grid, chi_grid, bounds_error=False, fill_value="extrapolate")

    expected = np.zeros_like(observed, dtype=np.float64)
    rows = []
    for iz, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        zmid = 0.5 * (zlo + zhi)
        hmf_z = np.asarray([np.interp(zmid, z_grid, hmf[:, im]) for im in range(len(m_grid))])
        chi_lo = float(chi_of_z(zlo))
        chi_hi = float(chi_of_z(zhi))
        volume = fsky * (4.0 * math.pi / 3.0) * abs(chi_hi**3 - chi_lo**3)
        for im, (mlo, mhi) in enumerate(zip(logm_edges[:-1], logm_edges[1:])):
            logm_samples = np.linspace(float(mlo), float(mhi), 96)
            hmf_samples = np.interp(logm_samples, logm_grid, hmf_z, left=np.nan, right=np.nan)
            valid = np.isfinite(hmf_samples)
            if np.count_nonzero(valid) >= 2:
                n_density = np.trapezoid(hmf_samples[valid], x=np.log(10.0) * logm_samples[valid])
            else:
                n_density = np.nan
            expected[iz, im] = volume * n_density
            rows.append(
                {
                    "z_lo": zlo,
                    "z_hi": zhi,
                    "logM_lo": mlo,
                    "logM_hi": mhi,
                    "observed": observed[iz, im],
                    "expected": expected[iz, im],
                    "obs_over_exp": observed[iz, im] / expected[iz, im] if np.isfinite(expected[iz, im]) and expected[iz, im] > 0 else np.nan,
                }
            )
    return {"observed": observed, "expected": expected, "rows": rows, "fsky": fsky}


def _interp_profile_matrix(bundle: TheoryBundle, matrix: np.ndarray, z: np.ndarray, logm: np.ndarray) -> np.ndarray:
    interp = RegularGridInterpolator(
        (np.asarray(bundle.base.z_array, dtype=np.float64), np.log10(np.asarray(bundle.base.M_array, dtype=np.float64))),
        np.asarray(matrix, dtype=np.float64),
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(np.column_stack([z, logm]))


def hod_occupation_comparison(
    catalog: Mapping[str, np.ndarray],
    galaxies: np.ndarray,
    bundle: TheoryBundle,
    *,
    z_edges: np.ndarray,
    logm_edges: np.ndarray,
) -> list:
    """Compare actual generated counts to summed HOD expectations in z/logM bins."""

    hz = np.asarray(catalog["z"], dtype=np.float64)
    hm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)
    exp_ncen = _interp_profile_matrix(bundle, bundle.profiles.Ncen_mat, hz, hm)
    exp_nsat = _interp_profile_matrix(bundle, bundle.profiles.Nsat_mat, hz, hm)

    valid_g = galaxies[:, 5] > 0.5
    gz = galaxies[valid_g, 2]
    gm = np.log10(galaxies[valid_g, 3])
    is_cen = galaxies[valid_g, 4] > 0.5
    rows = []
    for iz, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        for im, (mlo, mhi) in enumerate(zip(logm_edges[:-1], logm_edges[1:])):
            hmask = (hz >= zlo) & (hz < zhi) & (hm >= mlo) & (hm < mhi)
            gmask = (gz >= zlo) & (gz < zhi) & (gm >= mlo) & (gm < mhi)
            actual_cen = int(np.count_nonzero(gmask & is_cen))
            actual_sat = int(np.count_nonzero(gmask & ~is_cen))
            pred_cen = float(np.nansum(exp_ncen[hmask]))
            pred_sat = float(np.nansum(exp_nsat[hmask]))
            rows.append(
                {
                    "z_lo": zlo,
                    "z_hi": zhi,
                    "logM_lo": mlo,
                    "logM_hi": mhi,
                    "n_halos": int(np.count_nonzero(hmask)),
                    "actual_cen": actual_cen,
                    "pred_cen": pred_cen,
                    "actual_sat": actual_sat,
                    "pred_sat": pred_sat,
                    "actual_tot": actual_cen + actual_sat,
                    "pred_tot": pred_cen + pred_sat,
                    "cen_over_pred": actual_cen / pred_cen if pred_cen > 0 else np.nan,
                    "sat_over_pred": actual_sat / pred_sat if pred_sat > 0 else np.nan,
                    "tot_over_pred": (actual_cen + actual_sat) / (pred_cen + pred_sat) if (pred_cen + pred_sat) > 0 else np.nan,
                }
            )
    return rows


def hod_pair_count_comparison(
    catalog: Mapping[str, np.ndarray],
    galaxies: np.ndarray,
    bundle: TheoryBundle,
    *,
    z_edges: np.ndarray,
    logm_edges: np.ndarray,
) -> list:
    """Compare realized and expected one-halo galaxy pair moments.

    The theory Pgg one-halo term uses central-satellite and satellite-satellite
    moments proportional to ``2*Ncen*Nsat`` and ``Nsat**2``.  For a realized
    catalog, the corresponding unordered pair moments are
    ``2*ncen*nsat`` and ``nsat*(nsat-1)``.
    """

    hz = np.asarray(catalog["z"], dtype=np.float64)
    hm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)
    exp_ncen = _interp_profile_matrix(bundle, bundle.profiles.Ncen_mat, hz, hm)
    exp_nsat = _interp_profile_matrix(bundle, bundle.profiles.Nsat_mat, hz, hm)

    valid_g = galaxies[:, 5] > 0.5
    gm = np.log10(np.asarray(galaxies[valid_g, 3], dtype=np.float64))
    gz = np.asarray(galaxies[valid_g, 2], dtype=np.float64)
    is_cen = np.asarray(galaxies[valid_g, 4] > 0.5)
    gal_bin = np.full(len(galaxies[valid_g]), -1, dtype=np.int64)

    rows = []
    for iz, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        for im, (mlo, mhi) in enumerate(zip(logm_edges[:-1], logm_edges[1:])):
            hmask = (hz >= zlo) & (hz < zhi) & (hm >= mlo) & (hm < mhi)
            gmask = (gz >= zlo) & (gz < zhi) & (gm >= mlo) & (gm < mhi)
            gal_bin[gmask] = iz * (len(logm_edges) - 1) + im

            n_halos = int(np.count_nonzero(hmask))
            actual_cen = int(np.count_nonzero(gmask & is_cen))
            actual_sat = int(np.count_nonzero(gmask & ~is_cen))
            pred_cen = float(np.nansum(exp_ncen[hmask]))
            pred_sat = float(np.nansum(exp_nsat[hmask]))

            pred_cs = float(np.nansum(2.0 * exp_ncen[hmask] * exp_nsat[hmask]))
            pred_ss_poisson_ordered = float(np.nansum(exp_nsat[hmask] ** 2))
            pred_ss_same_gal_included = float(np.nansum(exp_nsat[hmask] ** 2 + exp_nsat[hmask]))

            rows.append(
                {
                    "z_lo": float(zlo),
                    "z_hi": float(zhi),
                    "logM_lo": float(mlo),
                    "logM_hi": float(mhi),
                    "n_halos": n_halos,
                    "actual_cen": actual_cen,
                    "actual_sat": actual_sat,
                    "pred_cen": pred_cen,
                    "pred_sat": pred_sat,
                    "actual_cs_ordered": 0.0,
                    "pred_cs_ordered": pred_cs,
                    "actual_ss_ordered": 0.0,
                    "pred_ss_poisson_ordered": pred_ss_poisson_ordered,
                    "pred_ss_same_gal_included": pred_ss_same_gal_included,
                }
            )

    # Realized pair counts need per-parent-halo grouping.  The generated catalog
    # stores parent halo mass and redshift; for this lightcone these are unique
    # enough when rounded to float32 precision, which is how they are stored.
    parent_dtype = np.dtype([("z", np.float32), ("m", np.float32)])
    parent_keys = np.empty(len(galaxies[valid_g]), dtype=parent_dtype)
    parent_keys["z"] = np.asarray(galaxies[valid_g, 2], dtype=np.float32)
    parent_keys["m"] = np.asarray(galaxies[valid_g, 3], dtype=np.float32)
    unique_keys, inv = np.unique(parent_keys, return_inverse=True)
    n_parent = len(unique_keys)
    ncen_parent = np.bincount(inv, weights=is_cen.astype(np.float64), minlength=n_parent)
    nsat_parent = np.bincount(inv, weights=(~is_cen).astype(np.float64), minlength=n_parent)
    parent_bin = np.full(n_parent, -1, dtype=np.int64)
    order = np.argsort(inv)
    parent_id, first_pos = np.unique(inv[order], return_index=True)
    parent_bin[parent_id] = gal_bin[order[first_pos]]

    for row_index, row in enumerate(rows):
        psel = parent_bin == row_index
        actual_cs = float(np.sum(2.0 * ncen_parent[psel] * nsat_parent[psel]))
        actual_ss = float(np.sum(nsat_parent[psel] * np.maximum(nsat_parent[psel] - 1.0, 0.0)))
        row["actual_cs_ordered"] = actual_cs
        row["actual_ss_ordered"] = actual_ss
        row["actual_pair_total"] = actual_cs + actual_ss
        row["pred_pair_total_poisson"] = row["pred_cs_ordered"] + row["pred_ss_poisson_ordered"]
        row["pred_pair_total_same_gal_included"] = row["pred_cs_ordered"] + row["pred_ss_same_gal_included"]
        row["cs_over_pred"] = actual_cs / row["pred_cs_ordered"] if row["pred_cs_ordered"] > 0 else np.nan
        row["ss_over_pred_poisson"] = actual_ss / row["pred_ss_poisson_ordered"] if row["pred_ss_poisson_ordered"] > 0 else np.nan
        row["pair_over_pred_poisson"] = row["actual_pair_total"] / row["pred_pair_total_poisson"] if row["pred_pair_total_poisson"] > 0 else np.nan
        row["pair_over_pred_same_gal_included"] = (
            row["actual_pair_total"] / row["pred_pair_total_same_gal_included"]
            if row["pred_pair_total_same_gal_included"] > 0
            else np.nan
        )
    return rows


def satellite_truncation_summary(bundle: TheoryBundle) -> dict:
    nsat = np.asarray(bundle.profiles.Nsat_mat, dtype=np.float64)
    max_mean = float(np.nanmax(nsat))
    max_gals = int(np.ceil(max_mean + np.sqrt(max_mean))) + 2
    safe_max_gals = int(np.ceil(max_mean + 10.0 * np.sqrt(max_mean))) + 1
    tail = poisson.sf(max_gals - 1, nsat)
    safe_tail = poisson.sf(safe_max_gals - 1, nsat)
    return {
        "max_mean_nsat": max_mean,
        "max_gals_per_halo": max_gals,
        "safe_max_gals_per_halo": safe_max_gals,
        "max_tail_probability": float(np.nanmax(tail)),
        "mean_tail_probability": float(np.nanmean(tail)),
        "safe_max_tail_probability": float(np.nanmax(safe_tail)),
        "safe_mean_tail_probability": float(np.nanmean(safe_tail)),
        "tail_grid": tail,
        "safe_tail_grid": safe_tail,
    }


def project_pk_component_to_cl(bundle: TheoryBundle, pk_component: np.ndarray, compare_bin: int) -> np.ndarray:
    """Project a P(k,z) component using the same galaxy kernel as get_Cl."""

    cls = bundle.cls
    k_grid = np.asarray(cls.kPk_array, dtype=np.float64)
    z_grid = np.asarray(cls.z_array, dtype=np.float64)
    pk = np.maximum(np.asarray(pk_component, dtype=np.float64), 1.0e-300)
    interp = RegularGridInterpolator(
        (np.log(k_grid), z_grid),
        np.log(pk),
        bounds_error=False,
        fill_value=None,
    )

    ell = np.asarray(cls.ell_array, dtype=np.float64)
    zc = np.asarray(cls.z_array_for_Cls, dtype=np.float64)
    chi = np.asarray(cls.chi_array_for_Cls, dtype=np.float64)
    dchi_dz = np.asarray(cls.dchi_dz_array_for_Cls, dtype=np.float64)
    wg = np.asarray(cls.Wg_mat[compare_bin], dtype=np.float64)
    prefac = wg / np.maximum(dchi_dz * chi**2, 1.0e-30)
    out = np.zeros_like(ell)
    for i, el in enumerate(ell):
        k_eval = (el + 0.5) / np.maximum(chi, 1.0)
        points = np.column_stack([np.log(k_eval), zc])
        pk_lz = np.exp(interp(points))
        fx = prefac * prefac * chi**2 * dchi_dz * pk_lz
        out[i] = np.trapezoid(fx, x=zc)
    return out


def project_pk_component_to_cl_with_window(
    bundle: TheoryBundle,
    pk_component: np.ndarray,
    z_window: np.ndarray,
    *,
    z_window_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project a P(k,z) component with an explicit normalized dN/dz window."""

    cls = bundle.cls
    k_grid = np.asarray(cls.kPk_array, dtype=np.float64)
    z_grid = np.asarray(cls.z_array, dtype=np.float64)
    pk = np.maximum(np.asarray(pk_component, dtype=np.float64), 1.0e-300)
    interp = RegularGridInterpolator(
        (np.log(k_grid), z_grid),
        np.log(pk),
        bounds_error=False,
        fill_value=None,
    )

    ell = np.asarray(cls.ell_array, dtype=np.float64)
    zc = np.asarray(cls.z_array_for_Cls, dtype=np.float64)
    chi = np.asarray(cls.chi_array_for_Cls, dtype=np.float64)
    dchi_dz = np.asarray(cls.dchi_dz_array_for_Cls, dtype=np.float64)
    if z_window_grid is None:
        z_window_grid = zc
    wg = normalize_nz(np.asarray(z_window_grid, dtype=np.float64), np.asarray(z_window, dtype=np.float64))
    if len(wg) != len(zc) or np.max(np.abs(np.asarray(z_window_grid) - zc)) > 1.0e-10:
        wg = np.interp(zc, np.asarray(z_window_grid, dtype=np.float64), wg, left=0.0, right=0.0)
        wg = normalize_nz(zc, wg)

    prefac = wg / np.maximum(dchi_dz * chi**2, 1.0e-30)
    out = np.zeros_like(ell)
    for i, el in enumerate(ell):
        k_eval = (el + 0.5) / np.maximum(chi, 1.0)
        points = np.column_stack([np.log(k_eval), zc])
        pk_lz = np.exp(interp(points))
        fx = prefac * prefac * chi**2 * dchi_dz * pk_lz
        out[i] = np.trapezoid(fx, x=zc)
    return out


def galaxy_matter_bias_curves_with_window(
    bundle: TheoryBundle,
    bias_dict: Mapping[str, np.ndarray],
    z_window: np.ndarray,
    *,
    z_window_grid: Optional[np.ndarray] = None,
    matter_power: str = "total",
) -> dict:
    """Return projected ``C_hm/C_mm`` curves for an explicit radial window."""

    pkz = bundle.pkz
    if matter_power == "linear":
        pmm = np.asarray(pkz.plin_kz_mat, dtype=np.float64)
    elif matter_power == "halofit":
        pmm = np.asarray(pkz.phfit_kz_mat, dtype=np.float64)
    elif matter_power == "total":
        pmm = np.asarray(pkz.Pmm_tot_mat, dtype=np.float64)
    else:
        raise ValueError(f"Unknown matter_power={matter_power!r}. Use 'linear', 'halofit', or 'total'.")

    cls_mm = project_pk_component_to_cl_with_window(bundle, pmm, z_window, z_window_grid=z_window_grid)
    out_cls = {"mm": cls_mm}
    out_bias = {}
    key_map = {
        "cen": "cen_self",
        "sat_u": "sat_u_self",
        "sat_u1": "sat_u1_self",
        "tot_std": "tot_std",
        "tot_u1": "tot_u1",
    }
    for out_key, bias_key in key_map.items():
        cls_hm = project_pk_component_to_cl_with_window(
            bundle,
            np.asarray(bias_dict[bias_key], dtype=np.float64) * pmm,
            z_window,
            z_window_grid=z_window_grid,
        )
        out_cls[f"hm_{out_key}"] = cls_hm
        out_bias[out_key] = cls_hm / np.maximum(cls_mm, 1.0e-300)
    return {
        "ell": np.asarray(bundle.cls.ell_array, dtype=np.float64),
        "cls": out_cls,
        "bias": out_bias,
        "matter_power": matter_power,
    }


def pgg_decomposition_cls(bundle: TheoryBundle, compare_bin: int) -> dict:
    """Project raw and response-suppressed galaxy 1h/2h theory components."""

    pkz = bundle.pkz
    p1 = np.asarray(pkz.Pgg_1h_kz_mat, dtype=np.float64)
    p2 = np.asarray(pkz.Pgg_2h_kz_mat, dtype=np.float64)
    response = np.asarray(getattr(pkz, "Pmm_sup_tot_mat", np.ones_like(p1)), dtype=np.float64)
    hmf = np.asarray(pkz.hmf_Mz_mat, dtype=np.float64)
    ncen = np.asarray(pkz.Ncen_mat, dtype=np.float64)
    nsat = np.asarray(pkz.Nsat_mat, dtype=np.float64)
    uk = np.asarray(pkz.uk_clm, dtype=np.float64)
    nbar = np.maximum(np.asarray(pkz.nbarz, dtype=np.float64), 1.0e-30)
    logm = np.log(np.asarray(pkz.M_array, dtype=np.float64))
    p1_cs = np.trapezoid(hmf[None, :, :] * (2.0 * ncen[None, :, :] * nsat[None, :, :] * uk) / nbar[None, :, None] ** 2, x=logm, axis=-1)
    p1_ss = np.trapezoid(hmf[None, :, :] * (nsat[None, :, :] * uk) ** 2 / nbar[None, :, None] ** 2, x=logm, axis=-1)
    p1_point_cs = np.trapezoid(hmf * (2.0 * ncen * nsat) / nbar[:, None] ** 2, x=logm, axis=-1)
    p1_point_ss = np.trapezoid(hmf * nsat**2 / nbar[:, None] ** 2, x=logm, axis=-1)
    p1_point_cs = np.broadcast_to(p1_point_cs[None, :], p1.shape)
    p1_point_ss = np.broadcast_to(p1_point_ss[None, :], p1.shape)
    alpha = float(getattr(pkz, "alpha_gg", 1.0))
    raw_alpha = (np.maximum(p1, 0.0) ** alpha + np.maximum(p2, 0.0) ** alpha) ** (1.0 / alpha)
    components = {
        "1h_raw": p1,
        "1h_cs_raw": p1_cs,
        "1h_ss_raw": p1_ss,
        "1h_point_cs": p1_point_cs,
        "1h_point_ss": p1_point_ss,
        "1h_point_total": p1_point_cs + p1_point_ss,
        "2h_raw": p2,
        "sum_raw": p1 + p2,
        "alpha_raw": raw_alpha,
        "1h_response": p1 * response,
        "1h_cs_response": p1_cs * response,
        "1h_ss_response": p1_ss * response,
        "2h_response": p2 * response,
        "sum_response": (p1 + p2) * response,
        "stored_total": np.asarray(pkz.Pgg_tot_mat, dtype=np.float64),
        "response_factor": response,
    }
    cls = {
        key: project_pk_component_to_cl(bundle, value, compare_bin)
        for key, value in components.items()
        if key != "response_factor"
    }
    return {
        "ell": np.asarray(bundle.cls.ell_array, dtype=np.float64),
        "cls": cls,
        "pk": components,
        "alpha_gg": alpha,
        "gg_transition_model": getattr(pkz, "gg_transition_model", ""),
    }


def same_halo_pair_cls(
    galaxies: np.ndarray,
    ell_values: np.ndarray,
    *,
    z_range: Optional[Tuple[float, float]] = None,
    max_groups: Optional[int] = None,
) -> dict:
    """Direct one-halo angular Cl from generated same-parent galaxy pairs.

    For l>0 and a full-sky point catalog, the shot-subtracted same-parent
    contribution is ``4*pi/N^2 * sum_{i != j, same parent} P_l(cos theta_ij)``.
    The returned central-satellite and satellite-satellite terms use ordered
    pairs, matching the usual HOD convention.
    """

    ell_values = np.asarray(ell_values, dtype=np.int64)
    valid = galaxies[:, 5] > 0.5
    if z_range is not None:
        valid &= (galaxies[:, 2] >= float(z_range[0])) & (galaxies[:, 2] < float(z_range[1]))
    g = np.asarray(galaxies[valid], dtype=np.float64)
    n_gal = int(len(g))
    out_zero = np.zeros(len(ell_values), dtype=np.float64)
    if n_gal == 0:
        return {
            "ell": ell_values,
            "cl_1h": out_zero.copy(),
            "cl_cs": out_zero.copy(),
            "cl_ss": out_zero.copy(),
            "cl_1h_point": out_zero.copy(),
            "cl_cs_point": out_zero.copy(),
            "cl_ss_point": out_zero.copy(),
            "n_gal": 0,
            "n_groups": 0,
            "ordered_pairs": 0.0,
        }

    parent_dtype = np.dtype([("z", np.float32), ("m", np.float32)])
    keys = np.empty(n_gal, dtype=parent_dtype)
    keys["z"] = np.asarray(g[:, 2], dtype=np.float32)
    keys["m"] = np.asarray(g[:, 3], dtype=np.float32)
    unique_keys, inv = np.unique(keys, return_inverse=True)
    order = np.argsort(inv)
    starts = np.searchsorted(inv[order], np.arange(len(unique_keys)), side="left")
    ends = np.searchsorted(inv[order], np.arange(len(unique_keys)), side="right")

    cs_sum = np.zeros(len(ell_values), dtype=np.float64)
    ss_sum = np.zeros(len(ell_values), dtype=np.float64)
    ordered_cs = 0.0
    ordered_ss = 0.0
    n_groups_used = 0
    for ig, (start, end) in enumerate(zip(starts, ends)):
        if max_groups is not None and ig >= int(max_groups):
            break
        idx = order[start:end]
        if len(idx) < 2:
            continue
        is_cen = g[idx, 4] > 0.5
        vec = unit_vectors_from_radec(g[idx, 0], g[idx, 1])
        n = len(idx)
        n_groups_used += 1
        for ia in range(n - 1):
            dots = np.clip(vec[ia + 1 :] @ vec[ia], -1.0, 1.0)
            pair_is_cs = is_cen[ia] != is_cen[ia + 1 :]
            pair_is_ss = (~is_cen[ia]) & (~is_cen[ia + 1 :])
            for jell, ell in enumerate(ell_values):
                p = eval_legendre(int(ell), dots)
                if np.any(pair_is_cs):
                    cs_sum[jell] += 2.0 * float(np.sum(p[pair_is_cs]))
                if np.any(pair_is_ss):
                    ss_sum[jell] += 2.0 * float(np.sum(p[pair_is_ss]))
            ordered_cs += 2.0 * float(np.count_nonzero(pair_is_cs))
            ordered_ss += 2.0 * float(np.count_nonzero(pair_is_ss))

    norm = 4.0 * math.pi / max(float(n_gal) ** 2, 1.0e-30)
    cs = norm * cs_sum
    ss = norm * ss_sum
    cs_point = np.full(len(ell_values), norm * ordered_cs, dtype=np.float64)
    ss_point = np.full(len(ell_values), norm * ordered_ss, dtype=np.float64)
    return {
        "ell": ell_values.astype(np.float64),
        "cl_1h": cs + ss,
        "cl_cs": cs,
        "cl_ss": ss,
        "cl_1h_point": cs_point + ss_point,
        "cl_cs_point": cs_point,
        "cl_ss_point": ss_point,
        "n_gal": n_gal,
        "n_groups": int(n_groups_used),
        "ordered_pairs": float(ordered_cs + ordered_ss),
        "ordered_cs": float(ordered_cs),
        "ordered_ss": float(ordered_ss),
        "shot": 4.0 * math.pi / max(float(n_gal), 1.0),
    }


def response_decomposed_pgg_cls(bundle: TheoryBundle, compare_bin: int) -> dict:
    pkz = bundle.pkz
    p1 = np.asarray(pkz.Pgg_1h_kz_mat, dtype=np.float64)
    p2 = np.asarray(pkz.Pgg_2h_kz_mat, dtype=np.float64)
    if getattr(pkz, "gg_transition_model", "") == "response":
        response = np.asarray(pkz.Pmm_sup_tot_mat, dtype=np.float64)
        p1 = p1 * response
        p2 = p2 * response
    return {
        "ell": np.asarray(bundle.cls.ell_array, dtype=np.float64),
        "gg_1h": project_pk_component_to_cl(bundle, p1, compare_bin),
        "gg_2h": project_pk_component_to_cl(bundle, p2, compare_bin),
        "gg_sum": project_pk_component_to_cl(bundle, p1 + p2, compare_bin),
    }


def galaxy_2h_variant_cls(
    bundle: TheoryBundle,
    compare_bin: int,
    *,
    apply_response: bool = True,
    logm_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Project galaxy 2-halo variants that isolate central/satellite and u(k) effects."""

    pkz = bundle.pkz
    m = np.asarray(bundle.base.M_array, dtype=np.float64)
    z_grid = np.asarray(bundle.base.z_array, dtype=np.float64)
    x = np.log(m)
    hmf = np.asarray(pkz.hmf_Mz_mat, dtype=np.float64)
    bias = np.asarray(pkz.bias_Mz_mat, dtype=np.float64)
    ncen = np.asarray(pkz.Ncen_mat, dtype=np.float64)
    nsat = np.asarray(pkz.Nsat_mat, dtype=np.float64)
    uk = np.asarray(pkz.uk_clm, dtype=np.float64)
    plin = np.asarray(pkz.plin_kz_mat, dtype=np.float64)
    response = np.ones_like(plin)
    if apply_response and getattr(pkz, "gg_transition_model", "") == "response":
        response = np.asarray(pkz.Pmm_sup_tot_mat, dtype=np.float64)

    if logm_range is not None:
        logm = np.log10(m)
        mmask = ((logm >= float(logm_range[0])) & (logm < float(logm_range[1]))).astype(np.float64)
        ncen = ncen * mmask[None, :]
        nsat = nsat * mmask[None, :]
    if z_range is not None:
        zmask = ((z_grid >= float(z_range[0])) & (z_grid < float(z_range[1]))).astype(np.float64)
        ncen = ncen * zmask[:, None]
        nsat = nsat * zmask[:, None]

    nbar_cen = np.maximum(np.trapezoid(hmf * ncen, x=x, axis=-1), 1.0e-30)
    nbar_sat = np.maximum(np.trapezoid(hmf * nsat, x=x, axis=-1), 1.0e-30)
    nbar_tot = np.maximum(nbar_cen + nbar_sat, 1.0e-30)

    def bias_from_weight(weight: np.ndarray, denom: np.ndarray) -> np.ndarray:
        if weight.ndim == 2:
            num = np.trapezoid(hmf * bias * weight, x=x, axis=-1)
            b_1d = num / denom
            return np.broadcast_to(b_1d[None, :], plin.shape)
        else:
            num = np.trapezoid(hmf[None, :, :] * bias[None, :, :] * weight, x=x, axis=-1)
            return num / denom[None, :]

    b_cen_self = bias_from_weight(ncen, nbar_cen)
    b_sat_u_self = bias_from_weight(nsat[None, :, :] * uk, nbar_sat)
    b_sat_u1_self = bias_from_weight(nsat, nbar_sat)
    b_tot_std = bias_from_weight(ncen[None, :, :] + nsat[None, :, :] * uk, nbar_tot)
    b_tot_u1 = bias_from_weight(ncen + nsat, nbar_tot)

    pk_components = {
        "cen_cen_u1": b_cen_self * b_cen_self * plin * response,
        "sat_sat_u": b_sat_u_self * b_sat_u_self * plin * response,
        "sat_sat_u1": b_sat_u1_self * b_sat_u1_self * plin * response,
        "cen_sat_u": b_cen_self * b_sat_u_self * plin * response,
        "cen_sat_u1": b_cen_self * b_sat_u1_self * plin * response,
        "tot_std": b_tot_std * b_tot_std * plin * response,
        "tot_u1": b_tot_u1 * b_tot_u1 * plin * response,
        "stored_2h": np.asarray(pkz.Pgg_2h_kz_mat, dtype=np.float64) * response,
    }
    cls_components = {
        key: project_pk_component_to_cl(bundle, value, compare_bin)
        for key, value in pk_components.items()
    }
    return {
        "ell": np.asarray(bundle.cls.ell_array, dtype=np.float64),
        "cls": cls_components,
        "pk": pk_components,
        "bias": {
            "cen_self": b_cen_self,
            "sat_u_self": b_sat_u_self,
            "sat_u1_self": b_sat_u1_self,
            "tot_std": b_tot_std,
            "tot_u1": b_tot_u1,
        },
        "nbar": {
            "cen": nbar_cen,
            "sat": nbar_sat,
            "tot": nbar_tot,
        },
        "apply_response": bool(apply_response),
    }


def galaxy_matter_2h_variant_cls(
    bundle: TheoryBundle,
    compare_bin: int,
    *,
    matter_power: str = "total",
    apply_response: bool = False,
) -> dict:
    """Project galaxy-matter 2-halo variants and the matching matter auto.

    This is a diagnostic complement to :func:`galaxy_2h_variant_cls`.  It uses
    the same lens-bin projection kernel for both fields, so the ratios
    ``C_hm/C_mm`` can be compared to empirical biases measured against Abacus
    particle-shell matter maps built with the same broad redshift window.
    """

    pkz = bundle.pkz
    if matter_power == "linear":
        pmm = np.asarray(pkz.plin_kz_mat, dtype=np.float64)
    elif matter_power == "halofit":
        pmm = np.asarray(pkz.phfit_kz_mat, dtype=np.float64)
    elif matter_power == "total":
        pmm = np.asarray(pkz.Pmm_tot_mat, dtype=np.float64)
    else:
        raise ValueError(f"Unknown matter_power={matter_power!r}. Use 'linear', 'halofit', or 'total'.")

    response = np.ones_like(pmm)
    if apply_response and getattr(pkz, "gg_transition_model", "") == "response":
        response = np.asarray(pkz.Pmm_sup_tot_mat, dtype=np.float64)

    gg_variants = galaxy_2h_variant_cls(bundle, compare_bin, apply_response=False)
    bias = gg_variants["bias"]
    pk_components = {
        "mm": pmm,
        "hm_cen": bias["cen_self"] * pmm * response,
        "hm_sat_u": bias["sat_u_self"] * pmm * response,
        "hm_sat_u1": bias["sat_u1_self"] * pmm * response,
        "hm_tot_std": bias["tot_std"] * pmm * response,
        "hm_tot_u1": bias["tot_u1"] * pmm * response,
    }
    cls_components = {
        key: project_pk_component_to_cl(bundle, value, compare_bin)
        for key, value in pk_components.items()
    }
    mm = np.maximum(cls_components["mm"], 1.0e-300)
    return {
        "ell": np.asarray(bundle.cls.ell_array, dtype=np.float64),
        "cls": cls_components,
        "bias": {
            "cen": cls_components["hm_cen"] / mm,
            "sat_u": cls_components["hm_sat_u"] / mm,
            "sat_u1": cls_components["hm_sat_u1"] / mm,
            "tot_std": cls_components["hm_tot_std"] / mm,
            "tot_u1": cls_components["hm_tot_u1"] / mm,
        },
        "matter_power": matter_power,
        "apply_response": bool(apply_response),
    }


def measure_delta_map_cls(
    delta_a: np.ndarray,
    nside: int,
    *,
    delta_b: Optional[np.ndarray] = None,
    lmax: Optional[int] = None,
    pixwin_correction: bool = True,
) -> dict:
    """Measure an auto- or cross-spectrum of already normalized delta maps."""

    if lmax is None:
        lmax = 3 * int(nside) - 1
    else:
        lmax = min(int(lmax), 3 * int(nside) - 1)
    pixwin2 = np.ones(lmax + 1, dtype=np.float64)
    if pixwin_correction:
        pixwin2 = np.maximum(hp.pixwin(nside, lmax=lmax) ** 2, 1.0e-30)
    if delta_b is None:
        raw = hp.anafast(np.asarray(delta_a, dtype=np.float32), lmax=lmax)
    else:
        raw = hp.anafast(np.asarray(delta_a, dtype=np.float32), np.asarray(delta_b, dtype=np.float32), lmax=lmax)
    return {
        "ell": np.arange(lmax + 1),
        "cl_raw": raw,
        "cl": raw / pixwin2,
        "pixwin2": pixwin2,
    }


def select_halos(
    catalog: Mapping[str, np.ndarray],
    *,
    logm_range: Tuple[float, float],
    z_range: Tuple[float, float],
    max_halos: int,
    seed: int = 1234,
) -> np.ndarray:
    logm = np.asarray(catalog["log10M200c_hMsun"])
    z = np.asarray(catalog["z"])
    mask = (logm >= logm_range[0]) & (logm < logm_range[1]) & (z >= z_range[0]) & (z < z_range[1])
    idx = np.where(mask)[0]
    if len(idx) > max_halos:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_halos, replace=False)
    return np.asarray(idx, dtype=np.int64)


def stack_map_around_halos(
    field_map: np.ndarray,
    catalog: Mapping[str, np.ndarray],
    nside: int,
    halo_indices: np.ndarray,
    radial_edges_r200: np.ndarray,
) -> dict:
    sums = np.zeros(len(radial_edges_r200) - 1, dtype=np.float64)
    counts = np.zeros(len(radial_edges_r200) - 1, dtype=np.int64)
    for ih in halo_indices:
        ra = float(catalog["ra_deg"][ih])
        dec = float(catalog["dec_deg"][ih])
        r200 = float(catalog["R200c_hMpc"][ih])
        da = float(catalog["DA_hMpc"][ih])
        max_angle = float(radial_edges_r200[-1]) * r200 / max(da, 1.0e-8)
        vec = hp.ang2vec(ra, dec, lonlat=True)
        pix = hp.query_disc(nside, vec, max_angle, inclusive=True)
        if len(pix) == 0:
            continue
        pra, pdec = hp.pix2ang(nside, pix, lonlat=True)
        theta = angular_separation_rad(ra, dec, pra, pdec)
        rr200 = theta * da / max(r200, 1.0e-8)
        bin_id = np.searchsorted(radial_edges_r200, rr200, side="right") - 1
        valid = (bin_id >= 0) & (bin_id < len(sums))
        np.add.at(sums, bin_id[valid], np.asarray(field_map)[pix[valid]])
        np.add.at(counts, bin_id[valid], 1)
    mean = np.full_like(sums, np.nan, dtype=np.float64)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return {"r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]), "mean": mean, "counts": counts}


def angular_separation_rad(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)
    a = np.sin((dec1 - dec2) / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra1 - ra2) / 2.0) ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def unit_vectors_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def stack_satellites_projected_r200(
    galaxies: np.ndarray,
    catalog: Mapping[str, np.ndarray],
    halo_indices: np.ndarray,
    radial_edges_r200: np.ndarray,
    *,
    max_match_r200: Optional[float] = None,
) -> dict:
    """Approximate satellite projected R/R200 by matching to the nearest selected halo."""

    if max_match_r200 is None:
        max_match_r200 = float(radial_edges_r200[-1])
    if len(halo_indices) == 0:
        return {"r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]), "counts": np.zeros(len(radial_edges_r200) - 1)}
    hvec = unit_vectors_from_radec(catalog["ra_deg"][halo_indices], catalog["dec_deg"][halo_indices])
    tree = cKDTree(hvec)
    hlogm = np.asarray(catalog["log10M200c_hMsun"])[halo_indices]
    hz = np.asarray(catalog["z"])[halo_indices]
    zlo, zhi = float(np.min(hz) - 1.0e-4), float(np.max(hz) + 1.0e-4)
    mlo, mhi = float(np.min(hlogm) - 1.0e-4), float(np.max(hlogm) + 1.0e-4)
    valid = (
        (galaxies[:, 5] > 0.5)
        & (galaxies[:, 4] < 0.5)
        & (galaxies[:, 2] >= zlo)
        & (galaxies[:, 2] < zhi)
        & (np.log10(galaxies[:, 3]) >= mlo)
        & (np.log10(galaxies[:, 3]) < mhi)
    )
    sats = galaxies[valid]
    if len(sats) == 0:
        return {"r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]), "counts": np.zeros(len(radial_edges_r200) - 1)}
    svec = unit_vectors_from_radec(sats[:, 0], sats[:, 1])
    _, nearest = tree.query(svec, k=1)
    matched = halo_indices[nearest]
    theta = angular_separation_rad(catalog["ra_deg"][matched], catalog["dec_deg"][matched], sats[:, 0], sats[:, 1])
    rr200 = theta * catalog["DA_hMpc"][matched] / np.maximum(catalog["R200c_hMpc"][matched], 1.0e-8)
    keep = rr200 <= max_match_r200
    hist, _ = np.histogram(rr200[keep], bins=radial_edges_r200)
    return {
        "r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]),
        "counts": hist,
        "n_sat_matched": int(np.count_nonzero(keep)),
        "n_halos": int(len(halo_indices)),
    }


def parent_key_collision_summary(catalog: Mapping[str, np.ndarray], galaxies: np.ndarray) -> dict:
    """Summarize uniqueness of the float32 parent key stored in the galaxy catalog."""

    halo_dtype = np.dtype([("z", np.float32), ("m", np.float32)])
    halo_keys = np.empty(len(catalog["z"]), dtype=halo_dtype)
    halo_keys["z"] = np.asarray(catalog["z"], dtype=np.float32)
    halo_keys["m"] = np.asarray(catalog["M200c_hMsun"], dtype=np.float32)
    unique_halo_keys, halo_counts = np.unique(halo_keys, return_counts=True)

    valid_sat = (galaxies[:, 5] > 0.5) & (galaxies[:, 4] < 0.5)
    sat_keys = np.empty(np.count_nonzero(valid_sat), dtype=halo_dtype)
    sat_keys["z"] = np.asarray(galaxies[valid_sat, 2], dtype=np.float32)
    sat_keys["m"] = np.asarray(galaxies[valid_sat, 3], dtype=np.float32)
    sat_unique = np.unique(sat_keys)
    pos = np.searchsorted(unique_halo_keys, sat_unique)
    valid_pos = pos < len(unique_halo_keys)
    matched = np.zeros(len(sat_unique), dtype=bool)
    matched[valid_pos] = unique_halo_keys[pos[valid_pos]] == sat_unique[valid_pos]
    matched_counts = halo_counts[pos[matched]] if np.any(matched) else np.asarray([], dtype=np.int64)
    return {
        "n_halos": int(len(halo_keys)),
        "n_unique_halo_keys": int(len(unique_halo_keys)),
        "n_colliding_halo_keys": int(np.count_nonzero(halo_counts > 1)),
        "max_halos_per_key": int(np.max(halo_counts)) if len(halo_counts) else 0,
        "n_satellites": int(np.count_nonzero(valid_sat)),
        "n_unique_satellite_keys": int(len(sat_unique)),
        "n_unmatched_satellite_keys": int(np.count_nonzero(~matched)),
        "n_ambiguous_satellite_keys": int(np.count_nonzero(matched_counts > 1)),
        "max_halos_per_satellite_key": int(np.max(matched_counts)) if len(matched_counts) else 0,
    }


def match_satellites_to_parent_halos(galaxies: np.ndarray, catalog: Mapping[str, np.ndarray]) -> dict:
    """Assign satellites to parent halos using the stored float32 parent ``(z, mass)`` key."""

    halo_dtype = np.dtype([("z", np.float32), ("m", np.float32)])
    halo_keys = np.empty(len(catalog["z"]), dtype=halo_dtype)
    halo_keys["z"] = np.asarray(catalog["z"], dtype=np.float32)
    halo_keys["m"] = np.asarray(catalog["M200c_hMsun"], dtype=np.float32)
    unique_halo_keys, inv_halo = np.unique(halo_keys, return_inverse=True)
    halo_order = np.argsort(inv_halo)
    halo_start = np.searchsorted(inv_halo[halo_order], np.arange(len(unique_halo_keys)), side="left")
    halo_end = np.searchsorted(inv_halo[halo_order], np.arange(len(unique_halo_keys)), side="right")

    valid_sat = np.where((galaxies[:, 5] > 0.5) & (galaxies[:, 4] < 0.5))[0]
    sat_keys = np.empty(len(valid_sat), dtype=halo_dtype)
    sat_keys["z"] = np.asarray(galaxies[valid_sat, 2], dtype=np.float32)
    sat_keys["m"] = np.asarray(galaxies[valid_sat, 3], dtype=np.float32)
    key_pos = np.searchsorted(unique_halo_keys, sat_keys)
    valid_pos = key_pos < len(unique_halo_keys)
    key_match = np.zeros(len(sat_keys), dtype=bool)
    key_match[valid_pos] = unique_halo_keys[key_pos[valid_pos]] == sat_keys[valid_pos]

    matched_halo = np.full(len(valid_sat), -1, dtype=np.int64)
    ambiguous = np.zeros(len(valid_sat), dtype=bool)
    if np.any(key_match):
        for pos in np.unique(key_pos[key_match]):
            sat_local = np.where(key_match & (key_pos == pos))[0]
            candidates = halo_order[halo_start[pos]:halo_end[pos]]
            if len(candidates) == 1:
                matched_halo[sat_local] = candidates[0]
                continue
            ambiguous[sat_local] = True
            theta = angular_separation_rad(
                np.asarray(catalog["ra_deg"], dtype=np.float64)[candidates][None, :],
                np.asarray(catalog["dec_deg"], dtype=np.float64)[candidates][None, :],
                np.asarray(galaxies[valid_sat[sat_local], 0], dtype=np.float64)[:, None],
                np.asarray(galaxies[valid_sat[sat_local], 1], dtype=np.float64)[:, None],
            )
            matched_halo[sat_local] = candidates[np.argmin(theta, axis=1)]

    ok = matched_halo >= 0
    rr200 = np.full(len(valid_sat), np.nan, dtype=np.float64)
    if np.any(ok):
        h = matched_halo[ok]
        theta = angular_separation_rad(
            np.asarray(catalog["ra_deg"], dtype=np.float64)[h],
            np.asarray(catalog["dec_deg"], dtype=np.float64)[h],
            np.asarray(galaxies[valid_sat[ok], 0], dtype=np.float64),
            np.asarray(galaxies[valid_sat[ok], 1], dtype=np.float64),
        )
        rr200[ok] = theta * np.asarray(catalog["DA_hMpc"], dtype=np.float64)[h] / np.maximum(
            np.asarray(catalog["R200c_hMpc"], dtype=np.float64)[h], 1.0e-8
        )
    return {
        "satellite_indices": valid_sat,
        "matched_halo_indices": matched_halo,
        "rr200": rr200,
        "matched": ok,
        "ambiguous": ambiguous,
        "n_satellites": int(len(valid_sat)),
        "n_matched": int(np.count_nonzero(ok)),
        "n_unmatched": int(np.count_nonzero(~ok)),
        "n_ambiguous": int(np.count_nonzero(ambiguous)),
    }


def stack_satellites_by_parent_key(
    galaxies: np.ndarray,
    catalog: Mapping[str, np.ndarray],
    match: Mapping[str, np.ndarray],
    radial_edges_r200: np.ndarray,
    *,
    logm_range: Tuple[float, float],
    z_range: Tuple[float, float],
) -> dict:
    """Stack satellite projected radii after robust parent-key matching."""

    matched = np.asarray(match["matched"], dtype=bool)
    halo_idx = np.asarray(match["matched_halo_indices"], dtype=np.int64)
    keep = matched.copy()
    hsel = halo_idx[keep]
    if len(hsel):
        hz = np.asarray(catalog["z"], dtype=np.float64)[hsel]
        hm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)[hsel]
        keep_idx = np.where(keep)[0]
        keep[keep_idx] &= (hz >= z_range[0]) & (hz < z_range[1]) & (hm >= logm_range[0]) & (hm < logm_range[1])
    rr = np.asarray(match["rr200"], dtype=np.float64)[keep]
    rr = rr[np.isfinite(rr)]
    hist, _ = np.histogram(rr, bins=radial_edges_r200)
    dr = np.diff(radial_edges_r200)
    pdf = hist / np.maximum(np.sum(hist) * dr, 1.0)
    return {
        "r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]),
        "counts": hist,
        "pdf": pdf,
        "rr200": rr,
        "n_sat_matched": int(len(rr)),
    }


def model_geometry_for_catalog(bundle: TheoryBundle, catalog: Mapping[str, np.ndarray]) -> dict:
    """Interpolate GODMAX geometry and R200c onto catalog halo positions."""

    z = np.asarray(catalog["z"], dtype=np.float64)
    logm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)
    model_r200_comoving = _interp_profile_matrix(bundle, bundle.profiles.r200c_mat, z, logm)
    model_r200_physical = model_r200_comoving / (1.0 + z)
    model_da = np.interp(
        z,
        np.asarray(bundle.base.z_array, dtype=np.float64),
        np.asarray(bundle.base.DA_array, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    catalog_r200 = np.asarray(catalog["R200c_hMpc"], dtype=np.float64)
    catalog_da = np.asarray(catalog["DA_hMpc"], dtype=np.float64)
    return {
        "R200c_model_hMpc": np.asarray(model_r200_comoving, dtype=np.float64),
        "R200c_model_comoving_hMpc": np.asarray(model_r200_comoving, dtype=np.float64),
        "R200c_model_physical_hMpc": np.asarray(model_r200_physical, dtype=np.float64),
        "DA_model_hMpc": np.asarray(model_da, dtype=np.float64),
        "R200c_catalog_hMpc": catalog_r200,
        "DA_catalog_hMpc": catalog_da,
        "R200c_catalog_over_model": catalog_r200 / np.maximum(model_r200_comoving, 1.0e-30),
        "R200c_catalog_physical_over_model_comoving": catalog_r200 / np.maximum(model_r200_comoving, 1.0e-30),
        "R200c_catalog_physical_over_model_physical": catalog_r200 / np.maximum(model_r200_physical, 1.0e-30),
        "DA_catalog_over_model": catalog_da / np.maximum(model_da, 1.0e-30),
    }


def satellite_rr200_with_geometry(
    galaxies: np.ndarray,
    catalog: Mapping[str, np.ndarray],
    match: Mapping[str, np.ndarray],
    da_hMpc: np.ndarray,
    r200_hMpc: np.ndarray,
) -> np.ndarray:
    """Recompute matched satellite projected R/R200 with supplied halo geometry."""

    sat_idx = np.asarray(match["satellite_indices"], dtype=np.int64)
    halo_idx = np.asarray(match["matched_halo_indices"], dtype=np.int64)
    matched = np.asarray(match["matched"], dtype=bool)
    da = np.asarray(da_hMpc, dtype=np.float64)
    r200 = np.asarray(r200_hMpc, dtype=np.float64)
    rr200 = np.full(len(sat_idx), np.nan, dtype=np.float64)
    if not np.any(matched):
        return rr200
    h = halo_idx[matched]
    theta = angular_separation_rad(
        np.asarray(catalog["ra_deg"], dtype=np.float64)[h],
        np.asarray(catalog["dec_deg"], dtype=np.float64)[h],
        np.asarray(galaxies[sat_idx[matched], 0], dtype=np.float64),
        np.asarray(galaxies[sat_idx[matched], 1], dtype=np.float64),
    )
    rr200[matched] = theta * da[h] / np.maximum(r200[h], 1.0e-30)
    return rr200


def stack_satellite_rr200_by_parent(
    catalog: Mapping[str, np.ndarray],
    match: Mapping[str, np.ndarray],
    rr200: np.ndarray,
    radial_edges_r200: np.ndarray,
    *,
    logm_range: Tuple[float, float],
    z_range: Tuple[float, float],
) -> dict:
    """Stack a precomputed satellite R/R200 array using matched parent bins."""

    matched = np.asarray(match["matched"], dtype=bool)
    halo_idx = np.asarray(match["matched_halo_indices"], dtype=np.int64)
    rr = np.asarray(rr200, dtype=np.float64)
    keep = matched & np.isfinite(rr)
    hsel = halo_idx[keep]
    if len(hsel):
        hz = np.asarray(catalog["z"], dtype=np.float64)[hsel]
        hm = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)[hsel]
        keep_idx = np.where(keep)[0]
        keep[keep_idx] &= (hz >= z_range[0]) & (hz < z_range[1]) & (hm >= logm_range[0]) & (hm < logm_range[1])
    rr_sel = rr[keep]
    hist, _ = np.histogram(rr_sel, bins=radial_edges_r200)
    dr = np.diff(radial_edges_r200)
    pdf = hist / np.maximum(np.sum(hist) * dr, 1.0)
    return {
        "r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]),
        "counts": hist,
        "pdf": pdf,
        "rr200": rr_sel,
        "n_sat_matched": int(len(rr_sel)),
    }


def expected_projected_satellite_hist(
    bundle: TheoryBundle,
    *,
    logm: float,
    z: float,
    radial_edges_r200: np.ndarray,
    n_samples: int = 200_000,
    seed: int = 12345,
    include_samples: bool = False,
) -> dict:
    """Monte-Carlo the projected satellite R/R200 distribution used by get_sim_maps."""

    rng = np.random.default_rng(seed)
    z_grid = np.asarray(bundle.base.z_array, dtype=np.float64)
    logm_grid = np.log10(np.asarray(bundle.base.M_array, dtype=np.float64))
    iz = int(np.argmin(np.abs(z_grid - z)))
    im = int(np.argmin(np.abs(logm_grid - logm)))
    r_comoving_kpc = np.asarray(bundle.base.r_array, dtype=np.float64) * 1000.0
    rho = np.asarray(bundle.profiles.rho_clm_mat[:, iz, im], dtype=np.float64)
    a = 1.0 / (1.0 + z_grid[iz])
    pdf = np.maximum(rho / a**3, 0.0) * r_comoving_kpc**2
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(r_comoving_kpc))])
    if cdf[-1] <= 0:
        return {"r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]), "pdf": np.full(len(radial_edges_r200) - 1, np.nan)}
    cdf /= cdf[-1]
    r3d_comoving = np.interp(rng.random(n_samples), cdf, r_comoving_kpc)
    r3d_physical = r3d_comoving / (1.0 + z_grid[iz])
    sintheta = np.sqrt(1.0 - rng.uniform(-1.0, 1.0, n_samples) ** 2)
    rproj = r3d_physical * sintheta
    r200_kpc = float(np.asarray(bundle.profiles.r200c_mat, dtype=np.float64)[iz, im] * 1000.0)
    rr200 = rproj / max(r200_kpc, 1.0e-8)
    hist, _ = np.histogram(rr200, bins=radial_edges_r200, density=True)
    out = {"r_mid": 0.5 * (radial_edges_r200[:-1] + radial_edges_r200[1:]), "pdf": hist, "z_grid": z_grid[iz], "logm_grid": logm_grid[im]}
    if include_samples:
        out["rr200"] = rr200
    return out


def radial_distribution_metrics(observed_rr200: np.ndarray, expected_rr200: np.ndarray) -> dict:
    """Return compact projected-radius agreement metrics."""

    obs = np.asarray(observed_rr200, dtype=np.float64)
    exp = np.asarray(expected_rr200, dtype=np.float64)
    obs = obs[np.isfinite(obs)]
    exp = exp[np.isfinite(exp)]
    if len(obs) == 0 or len(exp) == 0:
        return {
            "n_obs": int(len(obs)),
            "median_ratio": np.nan,
            "q68_ratio": np.nan,
            "ks_distance": np.nan,
        }
    grid = np.unique(np.concatenate([obs, exp]))
    obs_cdf = np.searchsorted(np.sort(obs), grid, side="right") / len(obs)
    exp_cdf = np.searchsorted(np.sort(exp), grid, side="right") / len(exp)
    return {
        "n_obs": int(len(obs)),
        "median_ratio": float(np.nanmedian(obs) / max(np.nanmedian(exp), 1.0e-30)),
        "q68_ratio": float(np.nanquantile(obs, 0.68) / max(np.nanquantile(exp, 0.68), 1.0e-30)),
        "ks_distance": float(np.nanmax(np.abs(obs_cdf - exp_cdf))),
    }



def paint_single_halo_maps(
    config: Mapping[str, object],
    bundle: TheoryBundle,
    *,
    mass_hMsun: float,
    z: float,
    nside: int,
    ra_deg: float = 180.0,
    dec_deg: float = 0.0,
    max_paint_r200: Optional[float] = None,
    random_seed: int = 12345,
) -> dict:
    """Paint deterministic maps for one synthetic halo with existing GODMAX machinery."""

    import jax
    import jax.numpy as jnp
    from get_sim_maps import get_sim_map, setup_sim_map

    logm = float(np.log10(mass_hMsun))
    z_grid = np.asarray(bundle.base.z_array, dtype=np.float64)
    logm_grid = np.log10(np.asarray(bundle.base.M_array, dtype=np.float64))
    r200_interp = RegularGridInterpolator(
        (z_grid, logm_grid),
        np.asarray(bundle.profiles.r200c_mat, dtype=np.float64),
        bounds_error=False,
        fill_value=None,
    )
    da = float(np.interp(z, z_grid, np.asarray(bundle.base.DA_array, dtype=np.float64)))
    r200 = float(r200_interp([[z, logm]])[0])
    catalog = {
        "ra_deg": np.asarray([ra_deg], dtype=np.float64),
        "dec_deg": np.asarray([dec_deg], dtype=np.float64),
        "z": np.asarray([z], dtype=np.float64),
        "M200c_hMsun": np.asarray([mass_hMsun], dtype=np.float64),
        "log10M200c_hMsun": np.asarray([logm], dtype=np.float64),
        "vlos_kms": np.asarray([0.0], dtype=np.float64),
        "R200c_hMpc": np.asarray([r200], dtype=np.float64),
        "DA_hMpc": np.asarray([da], dtype=np.float64),
    }
    setup_params = {
        "nside": int(nside),
        "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
        "profile_timing": False,
        "get_galmap": False,
        "get_ymap": True,
        "get_kSZmap": True,
        "get_taumap": True,
        "get_kappamap": True,
        "get_baryonifiedmap": True,
        "kappa_source_bin": 0,
    }
    setup = setup_sim_map(
        bundle.sim_params,
        bundle.halo_params,
        bundle.analysis,
        bundle.other_params,
        setup_params,
        Profiles_obj=bundle.profiles,
    )
    pixels = aph.build_pixel_work_package(
        catalog,
        int(nside),
        float(max_paint_r200 or config["pasting"].get("max_paint_R200c_factor", 3.0)),
        int(config["pasting"].get("pixel_batch_size", 2000)),
    )
    mock_params = dict(setup_params)
    mock_params.update(
        {
            "halo_z": jnp.array(catalog["z"], dtype=jnp.float32),
            "halo_ra": jnp.array(catalog["ra_deg"], dtype=jnp.float32),
            "halo_dec": jnp.array(catalog["dec_deg"], dtype=jnp.float32),
            "halo_M": jnp.array(catalog["M200c_hMsun"], dtype=jnp.float64),
            "halo_vlos": jnp.array(catalog["vlos_kms"], dtype=jnp.float32),
            "nearby_pix_all": jnp.array(pixels["nearby_pix_all"]),
            "pix_prop_all": jnp.array(
                [np.log(pixels["distances"]), pixels["z"], pixels["logM"], pixels["vlos"]],
                dtype=jnp.float32,
            ).T,
            "start_ind": jnp.array(pixels["start_ind"], dtype=jnp.int32),
            "end_ind": jnp.array(pixels["end_ind"], dtype=jnp.int32),
            "ang_distance_all": jnp.array(pixels["ang_distance_all"], dtype=jnp.float32),
            "rp_max_all": jnp.array(pixels["rp_max_all"], dtype=jnp.float32),
            "random_seed": int(random_seed),
        }
    )
    mock = get_sim_map(bundle.sim_params, bundle.halo_params, bundle.analysis, bundle.other_params, mock_params, Profiles_obj=setup)
    npix = hp.nside2npix(nside)
    maps = {
        "map_ymap": np.zeros(npix, dtype=np.float32),
        "map_ksz": np.zeros(npix, dtype=np.float32),
        "map_tau": np.zeros(npix, dtype=np.float32),
        "map_kappa": np.zeros(npix, dtype=np.float32),
        "map_rhom": np.zeros(npix, dtype=np.float32),
        "map_rhom_dmo": np.zeros(npix, dtype=np.float32),
    }
    attr_map = {
        "map_ymap": "ymap_final",
        "map_ksz": "kszmap_final",
        "map_tau": "taumap_final",
        "map_kappa": "kappamap_final",
        "map_rhom": "rhommap_final",
        "map_rhom_dmo": "rhom_dmo_map_final",
    }
    for key, attr in attr_map.items():
        if hasattr(mock, attr):
            maps[key] = np.asarray(np.nan_to_num(getattr(mock, attr)), dtype=np.float32)
    jax.clear_caches()
    return {"maps": maps, "catalog": catalog, "pixels": pixels}
