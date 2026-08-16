#!/usr/bin/env python
"""Re-execute matched 3D/projected profiles before any map-level comparison."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from common import (
    REPO_ROOT,
    WORKSPACE_ROOT,
    assert_map_contract_unchanged,
    canonical_json,
    comparison_source_manifest,
    effective_godmax_config_manifest,
    git_is_dirty,
    git_revision,
    godmax_profiles_class_path,
    load_config,
    load_yaml,
    resolve_path,
    runtime_version_manifest,
    profile_integration_contract,
    projected_profile_contract,
    sha256_file,
    sha256_json,
    validate_parameter_crosswalk,
)
from paint_baryonforge import (
    _scientific_imports,
    build_ccl_cosmology,
    build_direct_models,
    make_cmb_convergence_class,
    tabulate_projected_model,
)


SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
XDESI_DIR = REPO_ROOT / "notebooks" / "xDESI"
if str(XDESI_DIR) not in sys.path:
    sys.path.insert(0, str(XDESI_DIR))


def _godmax_profiles(config: Mapping[str, Any]):
    import jax

    jax.config.update("jax_enable_x64", True)
    from abacus_pasting_helpers import prepare_godmax_config
    from base_class import base_class
    from get_radial_profiles import Profiles as NativeProfiles
    from get_sim_maps import get_sim_map, setup_sim_map

    profiles_class = NativeProfiles
    profiles_path = godmax_profiles_class_path(config)
    if profiles_path is not None:
        module_name, _, class_name = profiles_path.rpartition(".")
        profiles_class = getattr(importlib.import_module(module_name), class_name)
        if not issubclass(profiles_class, NativeProfiles):
            raise TypeError(f"{profiles_path} is not a native GODMAX Profiles subclass.")

    catalog_path = resolve_path(config["catalog"]["output_h5"], config.get("_config_path"))
    with h5py.File(catalog_path, "r") as catalog_handle:
        catalog_attrs = dict(catalog_handle.attrs)
        catalog_redshift = np.asarray(catalog_handle["z"][:], dtype=np.float64)
    sim_params, halo_params, analysis, other_params = prepare_godmax_config(
        config,
        catalog_attrs,
        is_cmb_lensing=True,
        z_max=float(np.max(catalog_redshift)),
        log10_mass_min=float(catalog_attrs["log10_m_min_hmsun"]),
    )
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = profiles_class(
        sim_params,
        halo_params,
        analysis,
        other_params,
        base_class_obj=base,
    )
    setup_params = {
        "nside": int(config["validation"]["smoke_nside"]),
        "smooth_profiles": False,
        "profile_timing": False,
        "use_fused_profile_maps": True,
        "return_sparse_maps": False,
        "store_projected_matter_maps": False,
        "get_galmap": False,
        "get_ymap": True,
        "get_kSZmap": False,
        "get_taumap": False,
        "get_kappamap": True,
        "get_multi_kappamap": False,
        "get_baryonifiedmap": False,
        "kappa_source_bin": 0,
    }
    projected = setup_sim_map(
        sim_params,
        halo_params,
        analysis,
        other_params,
        setup_params,
        Profiles_obj=profiles,
    )
    # ``setup_sim_map`` creates the projected matter table but the map-side
    # lensing kernel normally lives on ``get_sim_map``.  Reuse that exact
    # implementation here so the profile diagnostic and native painter use
    # the same CMB-kappa convention.
    wkappa, source_label = get_sim_map._compute_wkappa_array(
        projected,
        kappa_source_bin=0,
        is_cmb_lensing=True,
    )
    projected.Wkappa_array_for_map = np.asarray(wkappa, dtype=np.float64)
    projected.kappa_source_label = source_label
    return profiles, projected, {
        "sim_params": sim_params,
        "halo_params": halo_params,
        "analysis": analysis,
        "other_params": other_params,
        "profiles_class_fqname": (
            f"{profiles_class.__module__}.{profiles_class.__qualname__}"
        ),
    }


def _interp3_positive(values, radii, redshifts, masses, query_r, query_z, query_m):
    values = np.asarray(values, dtype=np.float64)
    expected_shape = (len(radii), len(redshifts), len(masses))
    if values.shape != expected_shape:
        raise ValueError(f"Profile array has shape {values.shape}; expected {expected_shape}.")
    floor = max(float(np.nanmax(values)) * 1.0e-300, 1.0e-300)
    interpolator = RegularGridInterpolator(
        (np.log(radii), redshifts, np.log(masses)),
        np.log(np.maximum(values, floor)),
        bounds_error=True,
    )
    points = np.column_stack(
        [
            np.log(np.asarray(query_r, dtype=np.float64)),
            np.full(len(query_r), float(query_z)),
            np.full(len(query_r), np.log(float(query_m))),
        ]
    )
    return np.exp(interpolator(points))


def _interp2(values, redshifts, masses, query_z, query_m, *, positive: bool = False):
    values = np.asarray(values, dtype=np.float64)
    expected_shape = (len(redshifts), len(masses))
    if values.shape != expected_shape:
        raise ValueError(f"Grid has shape {values.shape}; expected {expected_shape}.")
    grid_values = np.log(np.maximum(values, 1.0e-300)) if positive else values
    interpolator = RegularGridInterpolator(
        (redshifts, np.log(masses)), grid_values, bounds_error=True
    )
    value = float(interpolator([[float(query_z), np.log(float(query_m))]])[0])
    return float(np.exp(value)) if positive else value


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(np.asarray(numerator, dtype=np.float64), np.nan)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    out[valid] = np.asarray(numerator, dtype=np.float64)[valid] / denominator[valid]
    return out


def _ratio_summary(numerator: np.ndarray, denominator: np.ndarray) -> dict:
    ratio = _ratio(numerator, denominator)
    finite = np.isfinite(ratio)
    if not np.any(finite):
        raise ValueError("Profile ratio has no finite samples.")
    denominator = np.asarray(denominator, dtype=np.float64)
    amplitude_floor = max(float(np.nanmax(np.abs(denominator))) * 1.0e-10, 1.0e-300)
    support = finite & (np.abs(denominator) > amplitude_floor)
    if not np.any(support):
        raise ValueError("Profile ratio has no samples on non-negligible reference support.")
    raw = ratio[finite]
    supported = ratio[support]
    return {
        "median": float(np.median(raw)),
        "p16": float(np.percentile(raw, 16.0)),
        "p84": float(np.percentile(raw, 84.0)),
        "support_median": float(np.median(supported)),
        "support_p16": float(np.percentile(supported, 16.0)),
        "support_p84": float(np.percentile(supported, 84.0)),
        "support_fraction": float(np.count_nonzero(support) / ratio.size),
        "support_relative_floor": 1.0e-10,
    }


def _godmax_component_mass_quadrature(godmax, n_radial: int) -> dict:
    """Integrate every GODMAX grid point on its effective normalization domain."""

    rmin_r200c = float(getattr(godmax, "integration_rmin_r200c", 0.01))
    rmax_r200c = float(getattr(godmax, "integration_rmax_r200c", 8.0))
    scaled_radius = np.geomspace(
        rmin_r200c,
        rmax_r200c,
        int(n_radial),
        dtype=np.float64,
    )
    log_scaled_radius = np.log(scaled_radius)
    closure = np.empty((len(godmax.z_array), len(godmax.M_array)), dtype=np.float64)
    redistribution_closure = np.empty_like(closure)
    dmo_target_error = np.empty_like(closure)
    component_over_input_m200c = np.empty_like(closure)
    dmo_over_input_m200c = np.empty_like(closure)

    for redshift_index in range(closure.shape[0]):
        radius_200c = np.asarray(godmax.r200c_mat[redshift_index], dtype=np.float64)
        radius = scaled_radius[:, None] * radius_200c[None, :]
        core_radius = (
            np.asarray(godmax.theta_co[redshift_index], dtype=np.float64) * radius_200c
        )[None, :]
        ejection_radius = (
            np.asarray(godmax.theta_ej[redshift_index], dtype=np.float64) * radius_200c
        )[None, :]
        beta = np.asarray(godmax.beta_mat[redshift_index], dtype=np.float64)[None, :]
        gas_shape = 1.0 / (
            (1.0 + radius / core_radius) ** beta
            * (1.0 + (radius / ejection_radius) ** float(godmax.gamma_rhogas))
            ** ((float(godmax.delta_rhogas) - beta) / float(godmax.gamma_rhogas))
        )
        rho_gas = (
            np.asarray(godmax.rho_gas_norm_mat[redshift_index], dtype=np.float64)[None, :]
            * gas_shape
        )

        concentration = np.asarray(
            godmax.conc_Mz_mat[redshift_index], dtype=np.float64
        )[None, :]
        scale_radius = radius_200c[None, :] / concentration
        truncation_radius = np.asarray(
            godmax.rt_mat[redshift_index], dtype=np.float64
        )[None, :]
        x_nfw = radius / scale_radius
        rho_nfw = 1.0 / (x_nfw * (1.0 + x_nfw) ** 2)
        if bool(godmax.nfw_trunc):
            rho_nfw /= (1.0 + (radius / truncation_radius) ** 2) ** 2
        rho_nfw *= np.asarray(
            godmax.rho_nfw_norm_mat[redshift_index], dtype=np.float64
        )[None, :]
        rho_collisionless = rho_nfw * np.asarray(
            godmax.fclm_mat[redshift_index], dtype=np.float64
        )[None, :]

        stellar_scale = np.asarray(
            godmax.Rh_mat[redshift_index], dtype=np.float64
        )[None, :]
        stellar_mass = (
            np.asarray(godmax.fstar_cen_mat[redshift_index], dtype=np.float64)
            * np.asarray(godmax.Mtot_mat[redshift_index], dtype=np.float64)
        )[None, :]
        rho_stars = (
            stellar_mass
            / (4.0 * np.pi**1.5 * stellar_scale * radius**2)
            * np.exp(-(0.5 * radius / stellar_scale) ** 2)
        )

        volume_weight = 4.0 * np.pi * radius**3
        component_mass = np.trapz(
            volume_weight * (rho_gas + rho_stars + rho_collisionless),
            x=log_scaled_radius,
            axis=0,
        )
        dmo_mass = np.trapz(
            volume_weight * rho_nfw,
            x=log_scaled_radius,
            axis=0,
        )
        native_total = np.asarray(
            godmax.Mtot_mat[redshift_index], dtype=np.float64
        )
        input_m200c = np.asarray(godmax.M_array, dtype=np.float64)
        closure[redshift_index] = component_mass / native_total - 1.0
        redistribution_closure[redshift_index] = component_mass / dmo_mass - 1.0
        dmo_target_error[redshift_index] = dmo_mass / native_total - 1.0
        component_over_input_m200c[redshift_index] = component_mass / input_m200c
        dmo_over_input_m200c[redshift_index] = dmo_mass / input_m200c

    worst = np.unravel_index(int(np.nanargmax(np.abs(closure))), closure.shape)
    return {
        "domain": (
            "effective GODMAX normalization domain in comoving R200c "
            f"[{rmin_r200c:.8g}, {rmax_r200c:.8g}]"
        ),
        "reference": (
            "effective GODMAX Mtot: truncated DMO mass integrated over the recorded "
            "comparison domain; this is the normalization target used by gas and stars"
        ),
        "n_redshift": int(closure.shape[0]),
        "n_mass": int(closure.shape[1]),
        "n_grid_points": int(closure.size),
        "n_radial": int(n_radial),
        "max_abs_relative_error": float(np.nanmax(np.abs(closure))),
        "median_abs_relative_error": float(np.nanmedian(np.abs(closure))),
        "component_over_dmo_max_abs_relative_error": float(
            np.nanmax(np.abs(redistribution_closure))
        ),
        "dmo_quadrature_vs_native_Mtot_max_abs_relative_error": float(
            np.nanmax(np.abs(dmo_target_error))
        ),
        "extended_component_mass_over_input_M200c_range": (
            float(np.nanmin(component_over_input_m200c)),
            float(np.nanmax(component_over_input_m200c)),
        ),
        "extended_dmo_mass_over_input_M200c_range": (
            float(np.nanmin(dmo_over_input_m200c)),
            float(np.nanmax(dmo_over_input_m200c)),
        ),
        "input_M200c_interpretation": (
            "M200c is enclosed mass at R200c, not the integral of the extended "
            "truncated profile over the effective normalization domain"
        ),
        "worst_z": float(np.asarray(godmax.z_array)[worst[0]]),
        "worst_mass_hMsun": float(np.asarray(godmax.M_array)[worst[1]]),
        "worst_signed_relative_error": float(closure[worst]),
    }


def _baryonforge_component_mass_quadrature(
    models: Mapping[str, Any],
    cosmo,
    params: Mapping[str, Any],
    n_radial: int,
) -> dict:
    """Integrate every BaryonForge production-table mass/redshift node."""

    tabulation = params["tabulation"]
    numerics = params["numerics"]
    redshifts = np.linspace(
        float(tabulation["z_min"]),
        float(tabulation["z_max"]),
        int(tabulation["n_z"]),
    )
    masses = np.geomspace(
        10.0 ** float(tabulation["log10_M_min_Msun"]),
        10.0 ** float(tabulation["log10_M_max_Msun"]),
        int(tabulation["n_M"]),
    )
    radius = np.geomspace(
        float(numerics["r_min_int_Mpc"]),
        float(numerics["r_max_int_Mpc"]),
        int(n_radial),
    )
    log_radius = np.log(radius)
    volume_weight = 4.0 * np.pi * radius[None, :] ** 3
    closure = np.empty((redshifts.size, masses.size), dtype=np.float64)
    component_over_input_m200c = np.empty_like(closure)
    dmo_over_input_m200c = np.empty_like(closure)

    for redshift_index, redshift in enumerate(redshifts):
        scale_factor = 1.0 / (1.0 + float(redshift))
        rho_gas = np.asarray(
            models["gas_direct"].real(cosmo, radius, masses, scale_factor),
            dtype=np.float64,
        )
        rho_stars = np.asarray(
            models["stars_direct"].real(cosmo, radius, masses, scale_factor),
            dtype=np.float64,
        )
        collisionless = models["collisionless_direct"]
        rho_collisionless = np.asarray(
            collisionless.real(cosmo, radius, masses, scale_factor),
            dtype=np.float64,
        )
        rho_dmo = np.asarray(
            collisionless.DarkMatter.real(cosmo, radius, masses, scale_factor),
            dtype=np.float64,
        )
        component_mass = np.trapz(
            volume_weight * (rho_gas + rho_stars + rho_collisionless),
            x=log_radius,
            axis=-1,
        )
        dmo_mass = np.trapz(
            volume_weight * rho_dmo,
            x=log_radius,
            axis=-1,
        )
        closure[redshift_index] = component_mass / dmo_mass - 1.0
        component_over_input_m200c[redshift_index] = component_mass / masses
        dmo_over_input_m200c[redshift_index] = dmo_mass / masses

    worst = np.unravel_index(int(np.nanargmax(np.abs(closure))), closure.shape)
    return {
        "domain": (
            "common native BaryonForge r_min_int--r_max_int comoving Mpc "
            f"[{radius[0]:.8g}, {radius[-1]:.8g}]"
        ),
        "reference": (
            "native BaryonForge M_tot: truncated DMO mass integrated over "
            "r_min_int--r_max_int; this is the normalization target used by gas and stars"
        ),
        "n_redshift": int(closure.shape[0]),
        "n_mass": int(closure.shape[1]),
        "n_grid_points": int(closure.size),
        "n_radial": int(n_radial),
        "max_abs_relative_error": float(np.nanmax(np.abs(closure))),
        "median_abs_relative_error": float(np.nanmedian(np.abs(closure))),
        "extended_component_mass_over_input_M200c_range": (
            float(np.nanmin(component_over_input_m200c)),
            float(np.nanmax(component_over_input_m200c)),
        ),
        "extended_dmo_mass_over_input_M200c_range": (
            float(np.nanmin(dmo_over_input_m200c)),
            float(np.nanmax(dmo_over_input_m200c)),
        ),
        "input_M200c_interpretation": (
            "M200c is enclosed mass at R200c, not the integral of the extended "
            "truncated profile over r_min_int--r_max_int"
        ),
        "worst_z": float(redshifts[worst[0]]),
        "worst_mass_Msun": float(masses[worst[1]]),
        "worst_signed_relative_error": float(closure[worst]),
    }


def _profile_input_contract(config: Mapping[str, Any]) -> dict:
    """Freeze every local input that can affect the profile comparison."""

    config_path = resolve_path(config["_config_path"])
    bparams_path = resolve_path(config["profiles"]["baryonforge_params"], config_path)
    gparams_path = resolve_path(config["profiles"]["godmax_params"], config_path)
    catalog_path = resolve_path(config["catalog"]["output_h5"], config_path)
    with h5py.File(catalog_path, "r") as catalog_handle:
        catalog_attrs = dict(catalog_handle.attrs)
        catalog_redshift = np.asarray(catalog_handle["z"][:], dtype=np.float64)
    source_manifest = comparison_source_manifest()
    profile_source = Path(__file__).resolve()
    source_manifest[
        profile_source.relative_to(WORKSPACE_ROOT).as_posix()
    ] = sha256_file(profile_source)
    source_manifest = dict(sorted(source_manifest.items()))
    effective_godmax = effective_godmax_config_manifest(
        config,
        catalog_attrs,
        catalog_redshift,
        is_cmb_lensing=True,
        log10_mass_min=float(catalog_attrs["log10_m_min_hmsun"]),
    )
    return {
        "comparison_config_path": str(config_path),
        "comparison_config_sha256": sha256_file(config_path),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_selection_predicate": str(config["catalog"]["predicate"]),
        "catalog_halo_count": int(catalog_redshift.size),
        "catalog_z_min": float(np.min(catalog_redshift)),
        "catalog_z_max": float(np.max(catalog_redshift)),
        "godmax_params": str(gparams_path),
        "godmax_params_sha256": sha256_file(gparams_path),
        "baryonforge_params": str(bparams_path),
        "baryonforge_params_sha256": sha256_file(bparams_path),
        "effective_godmax_config_manifest": effective_godmax,
        "effective_godmax_config_sha256": effective_godmax["sha256"],
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256_json(source_manifest),
        "godmax_git_sha": git_revision(WORKSPACE_ROOT / "GODMAX"),
        "baryonforge_git_sha": git_revision(WORKSPACE_ROOT / "BaryonForge"),
        "godmax_git_dirty": git_is_dirty(WORKSPACE_ROOT / "GODMAX"),
        "baryonforge_git_dirty": git_is_dirty(WORKSPACE_ROOT / "BaryonForge"),
        "runtime_versions": runtime_version_manifest(),
        "profile_integration_contract": profile_integration_contract(config),
        "projected_profile_contract": projected_profile_contract(config),
    }


def compare(
    config: Mapping[str, Any],
    output: Path,
    overwrite: bool,
    *,
    frozen_inputs: Mapping[str, Any] | None = None,
) -> dict:
    frozen_inputs = dict(
        frozen_inputs
        if frozen_inputs is not None
        else _profile_input_contract(config)
    )
    assert_map_contract_unchanged(
        frozen_inputs,
        _profile_input_contract(config),
        context="Profile-comparison pre-input-load validation",
    )
    crosswalk = validate_parameter_crosswalk(config)
    if not crosswalk["ok"]:
        raise ValueError(f"Parameter crosswalk failed: {crosswalk['failed']}")
    bparams_path = Path(frozen_inputs["baryonforge_params"])
    bparams = load_yaml(bparams_path)
    assert_map_contract_unchanged(
        frozen_inputs,
        _profile_input_contract(config),
        context="Profile-comparison post-input-load validation",
    )
    cosmo = build_ccl_cosmology(bparams["cosmology"])
    bmodels = build_direct_models(bparams, cosmo)
    y_tabulated, _ = tabulate_projected_model(
        bmodels["y_direct"], cosmo, bparams, smoke=False, verbose=False
    )
    sigma_tabulated, _ = tabulate_projected_model(
        bmodels["matter_direct"], cosmo, bparams, smoke=False, verbose=False
    )
    _, _, ccl = _scientific_imports()
    CMBConvergence = make_cmb_convergence_class(ccl)
    kappa_tabulated = CMBConvergence(
        sigma_tabulated,
        source_redshift=float(bparams["adapter"]["cmb_source_redshift"]),
    )
    godmax, projected, _ = _godmax_profiles(config)

    h = float(bparams["cosmology"]["h"])
    g_r = np.asarray(godmax.r_array, dtype=np.float64)
    g_z = np.asarray(godmax.z_array, dtype=np.float64)
    g_m = np.asarray(godmax.M_array, dtype=np.float64)
    g_rp = np.asarray(projected.rp_array, dtype=np.float64)
    x = np.geomspace(
        float(config["profiles"]["radius_min_R200c"]),
        float(config["profiles"]["radius_max_R200c"]),
        int(config["profiles"]["n_radius"]),
    )

    # In this simple-stellar baseline, the painted one-halo components are gas
    # + central stars + collisionless matter, where collisionless matter also
    # carries the satellite-stellar fraction.  Verify the component-fraction
    # budget independently of any finite-radius density quadrature.
    godmax_budget = (
        np.asarray(godmax.fgas_mat, dtype=np.float64)
        + np.asarray(godmax.fstar_cen_mat, dtype=np.float64)
        + np.asarray(godmax.fclm_mat, dtype=np.float64)
    )
    baryonforge_budget_samples = []
    baryon_fraction = float(bparams["cosmology"]["Omega_b"]) / float(
        bparams["cosmology"]["Omega_m"]
    )
    for mass_h in config["profiles"]["masses_hMsun"]:
        mass_physical = np.asarray([float(mass_h) / h], dtype=np.float64)
        for redshift in config["profiles"]["redshifts"]:
            scale_factor = 1.0 / (1.0 + float(redshift))
            f_gas = float(bmodels["gas_direct"].get_f_gas(mass_physical, scale_factor, cosmo)[0])
            f_central = float(
                bmodels["gas_direct"].get_f_star_cen(mass_physical, scale_factor, cosmo)[0]
            )
            f_satellite = float(
                bmodels["gas_direct"].get_f_star_sat(mass_physical, scale_factor, cosmo)[0]
            )
            f_collisionless = 1.0 - baryon_fraction + f_satellite
            baryonforge_budget_samples.append(
                {
                    "mass_hMsun": float(mass_h),
                    "z": float(redshift),
                    "sum": f_gas + f_central + f_collisionless,
                }
            )
    baryonforge_budget_error = max(
        abs(sample["sum"] - 1.0) for sample in baryonforge_budget_samples
    )
    component_conservation = {
        "assembly": "gas + central stars + collisionless matter (DM + satellite stars)",
        "fraction_reference": "each backend's native extended DMO mass Mtot",
        "godmax_grid_max_abs_fraction_error": float(np.max(np.abs(godmax_budget - 1.0))),
        "baryonforge_profile_points_max_abs_fraction_error": float(
            baryonforge_budget_error
        ),
        "tolerance": 1.0e-12,
        "ok": bool(
            np.max(np.abs(godmax_budget - 1.0)) <= 1.0e-12
            and baryonforge_budget_error <= 1.0e-12
        ),
        "baryonforge_samples": baryonforge_budget_samples,
    }
    quadrature_tolerance = float(
        config["validation"]["native_redistribution_max_relative_error"]
    )
    quadrature_radial_samples = int(
        config["validation"]["native_redistribution_radial_samples"]
    )
    component_conservation["density_quadrature"] = {
        "godmax": _godmax_component_mass_quadrature(
            godmax,
            quadrature_radial_samples,
        ),
        "baryonforge": _baryonforge_component_mass_quadrature(
            bmodels,
            cosmo,
            bparams,
            quadrature_radial_samples,
        ),
    }
    component_conservation["density_quadrature_tolerance"] = quadrature_tolerance
    component_conservation["density_quadrature_ok"] = all(
        result["max_abs_relative_error"] <= quadrature_tolerance
        for result in component_conservation["density_quadrature"].values()
    )
    component_conservation["ok"] = bool(
        component_conservation["ok"]
        and component_conservation["density_quadrature_ok"]
    )
    if not component_conservation["ok"]:
        raise RuntimeError(
            "Native extended-profile component conservation failed: "
            f"{component_conservation}"
        )

    records = []
    summaries = {}
    for mass_h in config["profiles"]["masses_hMsun"]:
        mass_h = float(mass_h)
        mass_physical = mass_h / h
        for redshift in config["profiles"]["redshifts"]:
            redshift = float(redshift)
            a = 1.0 / (1.0 + redshift)
            r200_g = _interp2(godmax.r200c_mat, g_z, g_m, redshift, mass_h, positive=True)
            r200_b = float(
                np.asarray(bmodels["mass_def"].get_radius(cosmo, mass_physical, a))
            ) / a
            radius_g = x * r200_g
            radius_b = radius_g / h
            # GODMAX's projected tables are queried with the physical
            # transverse separation DA*theta in Mpc/h, while BaryonForge's
            # profile API is queried with the corresponding comoving Mpc.
            radius_g_projected = a * radius_g

            gm = {
                "rho_gas": _interp3_positive(
                    godmax.rho_gas_mat, g_r, g_z, g_m, radius_g, redshift, mass_h
                ),
                "rho_stars": _interp3_positive(
                    godmax.rho_cga_mat, g_r, g_z, g_m, radius_g, redshift, mass_h
                ),
                "rho_collisionless": _interp3_positive(
                    godmax.rho_clm_mat, g_r, g_z, g_m, radius_g, redshift, mass_h
                ),
                "rho_matter": _interp3_positive(
                    godmax.rho_dmb_mat, g_r, g_z, g_m, radius_g, redshift, mass_h
                ),
                "y_projected": _interp3_positive(
                    projected.y2D_mat_physical,
                    g_rp,
                    g_z,
                    g_m,
                    radius_g_projected,
                    redshift,
                    mass_h,
                ),
            }
            sigma_phys = _interp3_positive(
                projected.rhom2D_mat_physical,
                g_rp,
                g_z,
                g_m,
                radius_g_projected,
                redshift,
                mass_h,
            )
            gm["sigma_matter_physical_Msun_Mpc2"] = sigma_phys * h
            wkappa = float(np.interp(redshift, g_z, np.asarray(projected.Wkappa_array_for_map)))
            gm["kappa_cmb"] = wkappa * a**2 * sigma_phys / float(godmax.rho_m_bar)

            bf = {
                "rho_gas": np.asarray(
                    bmodels["gas_direct"].real(cosmo, radius_b, mass_physical, a), dtype=np.float64
                )
                / h**2,
                "rho_stars": np.asarray(
                    bmodels["stars_direct"].real(cosmo, radius_b, mass_physical, a), dtype=np.float64
                )
                / h**2,
                "rho_collisionless": np.asarray(
                    bmodels["collisionless_direct"].real(
                        cosmo, radius_b, mass_physical, a
                    ),
                    dtype=np.float64,
                )
                / h**2,
                "rho_matter": np.asarray(
                    bmodels["matter_direct"].real(cosmo, radius_b, mass_physical, a),
                    dtype=np.float64,
                )
                / h**2,
                "y_projected": np.asarray(
                    bmodels["y_direct"].projected(cosmo, radius_b, mass_physical, a),
                    dtype=np.float64,
                ),
                "kappa_cmb": np.asarray(
                    bmodels["kappa_direct"].projected(
                        cosmo, radius_b, mass_physical, a
                    ),
                    dtype=np.float64,
                ),
                "sigma_matter_physical_Msun_Mpc2": np.asarray(
                    bmodels["matter_direct"].projected(
                        cosmo, radius_b, mass_physical, a
                    ),
                    dtype=np.float64,
                )
                / a**2,
            }
            bf_tabulated = {
                "y_projected": np.asarray(
                    y_tabulated.projected(cosmo, radius_b, mass_physical, a),
                    dtype=np.float64,
                ),
                "kappa_cmb": np.asarray(
                    kappa_tabulated.projected(cosmo, radius_b, mass_physical, a),
                    dtype=np.float64,
                ),
                "sigma_matter_physical_Msun_Mpc2": np.asarray(
                    sigma_tabulated.projected(cosmo, radius_b, mass_physical, a),
                    dtype=np.float64,
                )
                / a**2,
            }
            fstar_g = _interp2(godmax.fstar_tot_mat, g_z, g_m, redshift, mass_h)
            fcga_g = _interp2(godmax.fstar_cen_mat, g_z, g_m, redshift, mass_h)
            fstar_b = float(
                np.asarray(bmodels["gas_direct"].get_f_star(np.asarray([mass_physical]), a, cosmo))[0]
            )
            fcga_b = float(
                np.asarray(
                    bmodels["gas_direct"].get_f_star_cen(np.asarray([mass_physical]), a, cosmo)
                )[0]
            )
            key = f"log10M{np.log10(mass_h):.2f}_z{redshift:.3f}"
            summary = {
                "mass_hMsun": mass_h,
                "mass_Msun": mass_physical,
                "z": redshift,
                "R200c_godmax_comoving_hMpc": r200_g,
                "R200c_baryonforge_comoving_Mpc": r200_b,
                # This is not a unit round-trip test: each backend has already
                # recomputed R200c from its own critical-density constants.
                # Keep the native definition difference distinct from the
                # exact M/h and r/h adapter identities checked elsewhere.
                "R200c_baryonforge_times_h_over_godmax": r200_b * h / r200_g,
                "R200c_native_relative_difference_after_unit_conversion": (
                    abs(r200_b * h - r200_g) / r200_g
                ),
                "fstar_godmax": fstar_g,
                "fstar_baryonforge": fstar_b,
                "fcga_godmax": fcga_g,
                "fcga_baryonforge": fcga_b,
            }
            for field in gm:
                for suffix, value in _ratio_summary(bf[field], gm[field]).items():
                    summary[f"{field}_direct_ratio_{suffix}"] = value
            for field in bf_tabulated:
                for suffix, value in _ratio_summary(bf_tabulated[field], gm[field]).items():
                    summary[f"{field}_tabulated_ratio_{suffix}"] = value
                for suffix, value in _ratio_summary(bf_tabulated[field], bf[field]).items():
                    summary[f"{field}_tabulated_over_direct_{suffix}"] = value
            summaries[key] = summary
            records.append(
                (key, radius_g, radius_g_projected, radius_b, gm, bf, bf_tabulated)
            )

    assert_map_contract_unchanged(
        frozen_inputs,
        _profile_input_contract(config),
        context="Profile-comparison pre-publication validation",
    )
    provenance = {
        "schema": "baryonforge_godmax_profile_comparison_v1",
        "status": "exploratory_native_numerics",
        **frozen_inputs,
        "unit_boundary": {
            "catalog_R200c_hMpc": "physical/proper Mpc/h",
            "catalog_DA_hMpc": "physical/proper angular-diameter distance Mpc/h",
            "M_baryonforge_Msun": "M_godmax_hMsun / h",
            "r_godmax_3d": "comoving Mpc/h",
            "r_godmax_projected": "physical Mpc/h",
            "r_baryonforge_comoving_Mpc": "r_godmax_comoving_hMpc / h",
            "rho_baryonforge_to_godmax": "rho_baryonforge / h**2",
            "y_projected": "dimensionless Compton-y",
            "kappa_cmb": "dimensionless halo-only CMB convergence",
        },
        "component_conservation_check": component_conservation,
        "known_non_equivalences": [
            "mass-scaled versus absolute integration bounds",
            "different hydrostatic integration grids",
            "BaryonForge a=0 still uses cumulative integration and spline differentiation",
            "PyCCL and jax_cosmo use slightly different native CMB distance kernels",
        ],
        "summaries": summaries,
    }
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite explicitly.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace existing staging file {temporary}.")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = provenance["schema"]
            handle.attrs["comparison_config_sha256"] = provenance[
                "comparison_config_sha256"
            ]
            handle.attrs["catalog_sha256"] = provenance["catalog_sha256"]
            handle.attrs["effective_godmax_config_sha256"] = provenance[
                "effective_godmax_config_sha256"
            ]
            handle.attrs["source_manifest_sha256"] = provenance[
                "source_manifest_sha256"
            ]
            handle.attrs["provenance_json"] = canonical_json(provenance)
            for key, radius_g, radius_g_projected, radius_b, gm, bf, bf_tabulated in records:
                group = handle.create_group(key)
                group.create_dataset("radius_R200c", data=x)
                group.create_dataset("radius_godmax_comoving_hMpc", data=radius_g)
                group.create_dataset(
                    "radius_godmax_projected_physical_hMpc",
                    data=radius_g_projected,
                )
                group.create_dataset("radius_baryonforge_comoving_Mpc", data=radius_b)
                for backend, fields in (("godmax", gm), ("baryonforge", bf)):
                    target = group.create_group(backend)
                    for name, values in fields.items():
                        target.create_dataset(name, data=np.asarray(values, dtype=np.float64))
                tabulated = group.create_group("baryonforge_tabulated_for_painter")
                for name, values in bf_tabulated.items():
                    tabulated.create_dataset(name, data=np.asarray(values, dtype=np.float64))
                ratios = group.create_group("baryonforge_direct_over_godmax")
                for name in gm:
                    ratios.create_dataset(name, data=_ratio(bf[name], gm[name]))
                ratios = group.create_group("baryonforge_tabulated_over_godmax")
                for name in bf_tabulated:
                    ratios.create_dataset(name, data=_ratio(bf_tabulated[name], gm[name]))
                ratios = group.create_group("baryonforge_tabulated_over_direct")
                for name in bf_tabulated:
                    ratios.create_dataset(name, data=_ratio(bf_tabulated[name], bf[name]))
                for attr, value in summaries[key].items():
                    group.attrs[attr] = value
        assert_map_contract_unchanged(
            frozen_inputs,
            _profile_input_contract(config),
            context="Profile-comparison pre-publication validation",
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    provenance["output_h5"] = str(output)
    return provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    initial_config = load_config(args.config)
    frozen_inputs = _profile_input_contract(initial_config)
    config = load_config(args.config)
    if canonical_json(initial_config) != canonical_json(config):
        raise RuntimeError(
            "Comparison configuration changed while profile inputs were being frozen."
        )
    assert_map_contract_unchanged(
        frozen_inputs,
        _profile_input_contract(config),
        context="Initial profile-input validation",
    )
    output = resolve_path(args.output or config["profiles"]["output_h5"], config["_config_path"])
    report = compare(
        config,
        output,
        bool(args.overwrite),
        frozen_inputs=frozen_inputs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
