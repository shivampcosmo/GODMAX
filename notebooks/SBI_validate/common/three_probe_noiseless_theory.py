#!/usr/bin/env python3
"""Standalone map-matched resolved theory for the noiseless three-probe paste.

This module intentionally does not use :class:`get_Cl`: the production map is a
resolved-halo realization and its continuous fields are painted from smoothed,
line-of-sight projected profile tables through a transverse 8 R200c aperture.
The functions below reconstruct that operator, assemble the corresponding
resolved 1h+2h power, and perform the Limber projection.  HEALPix and estimator
windows are deliberately left to the comparison stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Mapping

os.environ.setdefault("JAX_ENABLE_X64", "True")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import h5py
import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "notebooks" / "xDESI", THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from three_probe_fast_paste import (  # noqa: E402
    _catalog_attrs,
    prepare_fast_paste_godmax_config,
)
from three_probe_mock_contract import sha256_array, sha256_file  # noqa: E402
from three_probe_projected_operator import (  # noqa: E402
    painter_log_interpolate,
    projected_painter_transform,
)
from three_probe_resolved_theory import (  # noqa: E402
    ResolvedSupport,
    assemble_resolved_power,
    fields_from_godmax,
    map_matched_profile_transforms,
    validate_resolved_inputs,
)


FIELD_TABLES = {
    "y": "y2D_mat_physical",
    "e": "ne2D_mat_physical",
    "m": "rhom2D_mat_physical",
}
REQUIRED_KERNELS = (
    "realized_hod_galaxy_redshift",
    "realized_hod_galaxy_nz",
    "halo_redshift",
    "hod_target_nbar_on_halo_redshift_h3mpc3",
    "cmb_lensing_efficiency_Wkappa_hmpc",
    "profile_smoothing_ell",
    "profile_smoothing_Bell",
)
PROVENANCE_PATHS = {
    "config_file_sha256": lambda config, theory: config,
    "config_factory_sha256": lambda config, theory: THIS_DIR / "three_probe_fast_paste.py",
    "pasting_helper_sha256": lambda config, theory: REPO_ROOT / "notebooks" / "xDESI" / "abacus_pasting_helpers.py",
    "get_sim_maps_sha256": lambda config, theory: REPO_ROOT / "src" / "get_sim_maps.py",
    "get_radial_profiles_sha256": lambda config, theory: REPO_ROOT / "src" / "get_radial_profiles.py",
    "base_class_sha256": lambda config, theory: REPO_ROOT / "src" / "base_class.py",
    "godmax_default_params_sha256": lambda config, theory: pathlib.Path(theory["default_params_path"]),
}


def validate_frozen_file_hashes(
    attrs: Mapping[str, Any], paths_by_attribute: Mapping[str, pathlib.Path | str]
) -> dict[str, str]:
    """Hash every local dependency and require equality with frozen map attrs."""

    observed: dict[str, str] = {}
    for attribute, path_input in paths_by_attribute.items():
        if attribute not in attrs:
            raise ValueError(f"Final map is missing frozen provenance attribute {attribute}")
        path = pathlib.Path(path_input)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen dependency does not exist: {path}")
        digest = sha256_file(path)
        if digest != str(attrs[attribute]):
            raise ValueError(
                f"Frozen dependency hash mismatch for {attribute}: {digest} != {attrs[attribute]}"
            )
        observed[attribute] = digest
    return observed


def validate_frozen_kernel_hashes(
    attrs: Mapping[str, Any], kernels: Mapping[str, np.ndarray]
) -> dict[str, str]:
    """Require every saved kernel to match the final product's byte contract."""

    encoded = attrs.get("kernel_dataset_sha256_json")
    if encoded is None:
        raise ValueError("Final map is missing kernel_dataset_sha256_json")
    try:
        expected = json.loads(str(encoded))
    except json.JSONDecodeError as error:
        raise ValueError("Final map kernel hash manifest is invalid JSON") from error
    missing = [name for name in REQUIRED_KERNELS if name not in kernels or name not in expected]
    if missing:
        raise ValueError(f"Final map kernel hash manifest is incomplete: {missing}")
    observed = {name: sha256_array(np.asarray(kernels[name])) for name in REQUIRED_KERNELS}
    mismatched = [name for name, digest in observed.items() if digest != str(expected[name])]
    if mismatched:
        raise ValueError(f"Final map kernel content hash mismatch: {mismatched}")
    return observed


def override_analysis_with_map_kernels(
    analysis: dict[str, Any], kernels: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Install the realized galaxy n(z) and exact HOD-consumed nbar anchors."""

    missing = [name for name in REQUIRED_KERNELS if name not in kernels]
    if missing:
        raise ValueError(f"Final map is missing required kernels: {missing}")
    z_g = np.asarray(kernels["realized_hod_galaxy_redshift"], dtype=np.float64)
    nz_g = np.asarray(kernels["realized_hod_galaxy_nz"], dtype=np.float64)
    z_h = np.asarray(kernels["halo_redshift"], dtype=np.float64)
    nbar = np.asarray(
        kernels["hod_target_nbar_on_halo_redshift_h3mpc3"], dtype=np.float64
    )
    wkappa = np.asarray(kernels["cmb_lensing_efficiency_Wkappa_hmpc"], dtype=np.float64)
    if z_g.shape != nz_g.shape or z_g.ndim != 1 or z_g.size < 3:
        raise ValueError("Realized HOD galaxy n(z) arrays are not aligned one-dimensional arrays")
    if z_h.shape != nbar.shape or z_h.shape != wkappa.shape or z_h.ndim != 1:
        raise ValueError("Halo-redshift, HOD nbar, and CMB-efficiency arrays are not aligned")
    arrays = (z_g, nz_g, z_h, nbar, wkappa)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("A required final-map kernel contains non-finite values")
    if np.any(np.diff(z_g) <= 0.0) or np.any(np.diff(z_h) <= 0.0):
        raise ValueError("Final-map redshift anchors must be strictly increasing")
    norm = float(np.trapz(nz_g, z_g))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"Realized HOD galaxy n(z) is not normalized: {norm}")
    if np.any(nz_g < 0.0) or np.any(nbar <= 0.0) or np.any(wkappa < 0.0):
        raise ValueError("n(z), HOD nbar, and CMB efficiency must be non-negative/positive")
    if z_g[0] != 0.3 or z_g[-1] != 0.5:
        raise ValueError("Realized galaxy kernel does not have the frozen [0.3, 0.5] support")

    analysis["nz_lens_info_dict"] = {
        "nbins_lens": 1,
        "z_edges_bins_lens": [[0.3, 0.5]],
        "z_array_lens": z_g.tolist(),
        "nz0": nz_g.tolist(),
    }
    analysis["nbar_gal_comoving_zarray"] = z_h.tolist()
    analysis["nbar_gal_comoving_val"] = nbar.tolist()
    return {
        "realized_nz_normalization": norm,
        "realized_nz_sha256": sha256_array(z_g, nz_g),
        "hod_nbar_sha256": sha256_array(z_h, nbar),
        "cmb_efficiency_sha256": sha256_array(z_h, wkappa),
    }


def projected_smoothed_profile_transforms(
    setup: Any,
    k_hmpc: np.ndarray,
    *,
    paint_r200c_factor: float = 8.0,
    n_projected_radius: int = 256,
) -> dict[str, np.ndarray]:
    """Transform the actual smoothed painter tables through their 8R aperture.

    The returned arrays have shape ``(nk,nz,nM)`` and match the field basis used
    by :func:`assemble_resolved_power`: y3d volume, absolute comoving electron
    number density, and matter density divided by present mean matter density.
    Smoothing is already present in the input tables and is not applied again.
    """

    k = np.asarray(k_hmpc, dtype=np.float64)
    rp_nodes = np.asarray(setup.rp_array, dtype=np.float64)
    z = np.asarray(setup.z_array, dtype=np.float64)
    mass = np.asarray(setup.M_array, dtype=np.float64)
    r200 = np.asarray(setup.r200c_mat, dtype=np.float64)
    if k.ndim != 1 or np.any(k < 0.0) or np.any(np.diff(k) <= 0.0):
        raise ValueError("k_hmpc must be a strictly increasing non-negative 1D grid")
    if n_projected_radius < 32:
        raise ValueError("Projected-radius quadrature requires at least 32 nodes")
    if r200.shape != (z.size, mass.size):
        raise ValueError("R200c table is not aligned with the paste z/M grid")
    if not bool(getattr(setup, "smooth_profiles", False)):
        raise ValueError("Projected theory requires the actual smoothed painter tables")

    tables = {
        field: np.asarray(getattr(setup, name), dtype=np.float64)
        for field, name in FIELD_TABLES.items()
    }
    expected_shape = (rp_nodes.size, z.size, mass.size)
    for field, table in tables.items():
        if table.shape != expected_shape or not np.all(np.isfinite(table)) or np.any(table <= 0.0):
            raise ValueError(f"Smoothed {field} painter table is invalid: {table.shape}")

    output = {
        field: np.empty((k.size, z.size, mass.size), dtype=np.float64)
        for field in FIELD_TABLES
    }
    rho_m = float(np.asarray(setup.rhom_0))
    if not np.isfinite(rho_m) or rho_m <= 0.0:
        raise ValueError("Present-day mean matter density must be finite and positive")
    for iz, z_value in enumerate(z):
        a = 1.0 / (1.0 + z_value)
        for im in range(mass.size):
            aperture_phys = float(paint_r200c_factor) * r200[iz, im] * a
            rp_dense = np.geomspace(
                max(aperture_phys * 1.0e-7, np.finfo(np.float64).tiny),
                aperture_phys,
                int(n_projected_radius),
            )
            factors = {"y": a ** -3, "e": 1.0, "m": 1.0 / rho_m}
            for field, table in tables.items():
                sigma = painter_log_interpolate(
                    rp_nodes, table[:, iz, im], rp_dense
                )
                output[field][:, iz, im] = projected_painter_transform(
                    k,
                    rp_dense,
                    sigma,
                    z_value,
                    aperture_phys,
                    physical_to_theory_volume_factor=factors[field],
                )
    return output


def project_resolved_power_to_intrinsic_cls(
    ell: np.ndarray,
    k_hmpc: np.ndarray,
    redshift: np.ndarray,
    chi_hmpc: np.ndarray,
    dchi_dz_hmpc: np.ndarray,
    powers: Mapping[str, np.ndarray],
    realized_galaxy_nz: np.ndarray,
    cmb_efficiency_hmpc: np.ndarray,
    tau_constant_mpc3_h3: float,
) -> dict[str, np.ndarray]:
    """Project resolved powers to intrinsic smooth Cls without map windows."""

    ell = np.asarray(ell, dtype=np.float64)
    k = np.asarray(k_hmpc, dtype=np.float64)
    z = np.asarray(redshift, dtype=np.float64)
    chi = np.asarray(chi_hmpc, dtype=np.float64)
    dchi = np.asarray(dchi_dz_hmpc, dtype=np.float64)
    nz = np.asarray(realized_galaxy_nz, dtype=np.float64)
    wkappa = np.asarray(cmb_efficiency_hmpc, dtype=np.float64)
    if ell.ndim != 1 or ell.size < 1 or np.any(ell < 0.0):
        raise ValueError("ell must be a non-negative one-dimensional grid")
    if not (z.shape == chi.shape == dchi.shape == nz.shape == wkappa.shape):
        raise ValueError("All Limber redshift kernels must have one common shape")
    if np.any(chi <= 0.0) or np.any(dchi <= 0.0):
        raise ValueError("chi and dchi/dz must be positive")
    if abs(float(np.trapz(nz, z)) - 1.0) > 1.0e-6:
        raise ValueError("Realized galaxy n(z) is not normalized for Limber projection")

    required = {
        "gg": "Pgg_resolved",
        "gy": "Pgy_resolved",
        "gtau": "Pge_resolved",
        "gkappa": "Pgm_resolved",
    }
    interpolators = {}
    for spectrum, key in required.items():
        if key not in powers:
            raise ValueError(f"Resolved power is missing {key}")
        value = np.asarray(powers[key], dtype=np.float64)
        if value.shape != (k.size, z.size) or not np.all(np.isfinite(value)):
            raise ValueError(f"Resolved power {key} is invalid")
        interpolators[spectrum] = RegularGridInterpolator(
            (np.log(k), z), value, bounds_error=True
        )

    result = {name: np.empty(ell.size, dtype=np.float64) for name in required}
    tau_kernel = float(tau_constant_mpc3_h3) * (1.0 + z) ** 2
    y_kernel = 1.0 / (1.0 + z)
    kernels = {"gy": y_kernel, "gtau": tau_kernel, "gkappa": wkappa}
    for index, ell_value in enumerate(ell):
        k_limber = (ell_value + 0.5) / chi
        if np.any(k_limber < k[0]) or np.any(k_limber > k[-1]):
            raise ValueError(
                f"Limber k lies outside resolved grid at ell={ell_value}: "
                f"[{k_limber.min()}, {k_limber.max()}] vs [{k[0]}, {k[-1]}]"
            )
        points = np.column_stack((np.log(k_limber), z))
        pgg = interpolators["gg"](points)
        result["gg"][index] = np.trapz(nz * nz * pgg / (dchi * chi * chi), z)
        for spectrum in ("gy", "gtau", "gkappa"):
            power = interpolators[spectrum](points)
            result[spectrum][index] = np.trapz(
                nz * kernels[spectrum] * power / (chi * chi), z
            )
    return result


def _read_final_map(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
        if "kernels" not in handle:
            raise ValueError("Final map has no kernels group")
        group = handle["kernels"]
        kernels = {name: np.asarray(group[name]) for name in group}
        kernel_attrs = dict(group.attrs)
    return attrs, kernels, kernel_attrs


def build_noiseless_intrinsic_theory(
    config_path: pathlib.Path | str,
    map_path: pathlib.Path | str,
    *,
    verify_catalog_sha: bool = True,
    n_projected_radius: int = 256,
    ell_max: int = 1535,
) -> dict[str, Any]:
    """Construct the current-code, final-H5-bound intrinsic theory product."""

    from base_class import base_class
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles
    from get_sim_maps import setup_sim_map

    config_path = pathlib.Path(config_path).resolve()
    map_path = pathlib.Path(map_path).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    theory = config["resolved_theory"]
    attrs, kernels, kernel_attrs = _read_final_map(map_path)

    paths = {
        attribute: resolver(config_path, theory)
        for attribute, resolver in PROVENANCE_PATHS.items()
    }
    frozen_hashes = validate_frozen_file_hashes(attrs, paths)
    kernel_hashes = validate_frozen_kernel_hashes(attrs, kernels)
    for key in ("catalog_file_sha256", "catalog_cosmology_sha256"):
        expected = theory["cosmology_sha256" if key == "catalog_cosmology_sha256" else key]
        if str(attrs.get(key, "")) != str(expected):
            raise ValueError(f"Final map {key} differs from experiment config")
    catalog_path = pathlib.Path(theory["catalog_path"])
    if verify_catalog_sha and sha256_file(catalog_path) != str(theory["catalog_file_sha256"]):
        raise ValueError("Catalog content hash differs from frozen experiment config")
    map_nside = int(attrs.get("nside", -1))
    if map_nside not in (512, 1024) or int(attrs.get("comparison_lmax", -1)) != 1535:
        raise ValueError("Final map does not use a frozen supported nside/lmax")
    if (int(attrs.get("halo_grid_nr", -1)), int(attrs.get("halo_grid_nM", -1)), int(attrs.get("halo_grid_nz", -1))) != (48, 24, 48):
        raise ValueError("Final map does not use the frozen 48/24/48 paste grid")
    if str(attrs.get("projected_profile_integration_method", "")) != "physical_table_cosh" or int(attrs.get("num_points_projected_profile", -1)) != 32:
        raise ValueError("Final map does not use physical_table_cosh with 32 LOS nodes")
    if not bool(attrs.get("profile_smoothing_applied", False)) or str(attrs.get("profile_smoothing_method", "")) != "real_space_gaussian":
        raise ValueError("Final map does not use the frozen real-space smoothing")
    smoothing_contract = (
        float(attrs.get("profile_smoothing_fwhm_pixel_fraction", -1.0)),
        int(attrs.get("profile_smoothing_quadrature_points", -1)),
        float(attrs.get("profile_smoothing_radial_sigma_cutoff", -1.0)),
    )
    if smoothing_contract != (0.5, 64, 10.0):
        raise ValueError(f"Final map smoothing contract is not frozen: {smoothing_contract}")

    catalog_attrs = _catalog_attrs(catalog_path)
    sim, halo, analysis, other = prepare_fast_paste_godmax_config(
        config, catalog_attrs, config_path=config_path
    )
    kernel_provenance = override_analysis_with_map_kernels(analysis, kernels)
    base = base_class(sim, halo, analysis, other)
    profiles = Profiles(sim, halo, analysis, other, base_class_obj=base)
    pkz = get_Pkz(sim, halo, analysis, other, Profiles_obj=profiles)
    setup_params = {
        "nside": map_nside,
        "smooth_profiles": True,
        "profile_smoothing_fwhm_pixel_fraction": 0.5,
        "profile_smoothing_method": "real_space_gaussian",
        "profile_smoothing_quadrature_points": 64,
        "profile_smoothing_radial_sigma_cutoff": 10.0,
        "projected_profile_integration_method": "physical_table_cosh",
        "num_points_projected_profile": 32,
        "profile_timing": False,
        "get_ymap": True,
        "get_kSZmap": False,
        "get_taumap": True,
        "get_kappamap": True,
        "get_galmap": False,
        "get_baryonifiedmap": False,
    }
    setup = setup_sim_map(sim, halo, analysis, other, setup_params, Profiles_obj=profiles)

    k = np.asarray(pkz.kPk_array, dtype=np.float64)
    projected = projected_smoothed_profile_transforms(
        setup, k, n_projected_radius=n_projected_radius
    )
    fields = {"g": np.asarray(pkz.ukg_cross, dtype=np.float64), **projected}
    galaxy_auto = np.asarray(pkz.ukg_auto_sqr, dtype=np.float64)
    support = ResolvedSupport()
    validate_resolved_inputs(
        pkz.M_array,
        pkz.z_array,
        pkz.kPk_array,
        pkz.hmf_Mz_mat,
        pkz.bias_Mz_mat,
        pkz.plin_kz_mat,
        fields,
        galaxy_auto,
        support,
    )
    powers_jax = assemble_resolved_power(
        pkz.M_array,
        pkz.hmf_Mz_mat,
        pkz.bias_Mz_mat,
        pkz.plin_kz_mat,
        fields,
        galaxy_auto,
    )
    powers = {name: np.asarray(value, dtype=np.float64) for name, value in powers_jax.items()}

    if int(ell_max) < 2 or int(ell_max) > 3 * map_nside - 1:
        raise ValueError("Requested theory ell_max is outside the HEALPix support")
    ell = np.arange(int(ell_max) + 1, dtype=np.float64)
    z = np.asarray(pkz.z_array, dtype=np.float64)
    realized_nz = np.interp(
        z,
        np.asarray(kernels["realized_hod_galaxy_redshift"], dtype=np.float64),
        np.asarray(kernels["realized_hod_galaxy_nz"], dtype=np.float64),
    )
    realized_nz /= np.trapz(realized_nz, z)
    wkappa = np.asarray(kernels["cmb_lensing_efficiency_Wkappa_hmpc"], dtype=np.float64)
    if not np.array_equal(z.astype(np.float32), np.asarray(kernels["halo_redshift"], dtype=np.float32)):
        raise ValueError("Runtime GODMAX redshift nodes differ from the exact map interpolation anchors")
    cls = project_resolved_power_to_intrinsic_cls(
        ell,
        k,
        z,
        np.asarray(pkz.chi_array, dtype=np.float64),
        np.asarray(pkz.dchi_dz_array, dtype=np.float64),
        powers,
        realized_nz,
        wkappa,
        float(setup.const_coeff_tau),
    )

    spherical_fields, spherical_auto = fields_from_godmax(
        pkz, map_matched_profile_transforms(pkz)
    )
    spherical_powers_jax = assemble_resolved_power(
        pkz.M_array,
        pkz.hmf_Mz_mat,
        pkz.bias_Mz_mat,
        pkz.plin_kz_mat,
        spherical_fields,
        spherical_auto,
    )
    spherical_powers = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in spherical_powers_jax.items()
    }
    spherical_cls = project_resolved_power_to_intrinsic_cls(
        ell,
        k,
        z,
        np.asarray(pkz.chi_array, dtype=np.float64),
        np.asarray(pkz.dchi_dz_array, dtype=np.float64),
        spherical_powers,
        realized_nz,
        wkappa,
        float(setup.const_coeff_tau),
    )
    saved_bell_ell = np.asarray(kernels["profile_smoothing_ell"], dtype=np.float64)
    saved_bell = np.asarray(kernels["profile_smoothing_Bell"], dtype=np.float64)
    sigma_rad = float(kernel_attrs["profile_smoothing_sigma_rad"])
    bell = np.exp(-0.5 * (ell * sigma_rad) ** 2)
    np.testing.assert_allclose(
        bell[: saved_bell.size], saved_bell, rtol=2.0e-15, atol=0.0
    )
    for spectrum in ("gy", "gtau", "gkappa"):
        spherical_cls[spectrum] *= bell

    provenance = {
        "config": str(config_path),
        "map": str(map_path),
        "catalog": str(catalog_path.resolve()),
        "frozen_file_hashes": frozen_hashes,
        "frozen_kernel_hashes": kernel_hashes,
        "get_Pkzs_sha256": sha256_file(REPO_ROOT / "src" / "get_Pkzs.py"),
        "theory_module_sha256": sha256_file(pathlib.Path(__file__)),
        "kernel_provenance": kernel_provenance,
        "kernel_attrs": {key: str(value) for key, value in kernel_attrs.items()},
        "support": {"mass_min_hmsun": 5.0e11, "mass_max_hmsun": 1.0e16, "z_min": 0.3, "z_max": 0.5},
        "profile_operator": "actual_smoothed_projected_table_then_transverse_8R200c_aperture",
        "transfer_applied": "profile smoothing embedded; no HEALPix or estimator window",
        "spherical_diagnostic": "spherical_8R200c_then_saved_Bell_once_on_continuous_cross_leg",
        "n_projected_radius": int(n_projected_radius),
        "nside": map_nside,
        "ell_max": int(ell_max),
    }
    return {
        "ell": ell,
        "cls": cls,
        "spherical_8r_cls": spherical_cls,
        "powers": powers,
        "redshift": z,
        "k_hmpc": k,
        "chi_hmpc": np.asarray(pkz.chi_array, dtype=np.float64),
        "dchi_dz_hmpc": np.asarray(pkz.dchi_dz_array, dtype=np.float64),
        "tau_constant_mpc3_h3": float(setup.const_coeff_tau),
        "realized_nz_on_theory_grid": realized_nz,
        "cmb_efficiency_hmpc": wkappa,
        "provenance": provenance,
    }


def save_theory_product(path: pathlib.Path | str, product: Mapping[str, Any]) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "ell": np.asarray(product["ell"]),
        "redshift": np.asarray(product["redshift"]),
        "k_hmpc": np.asarray(product["k_hmpc"]),
        "realized_nz_on_theory_grid": np.asarray(product["realized_nz_on_theory_grid"]),
        "cmb_efficiency_hmpc": np.asarray(product["cmb_efficiency_hmpc"]),
        "provenance_json": np.asarray(json.dumps(product["provenance"], indent=2, sort_keys=True)),
    }
    for name, value in product["cls"].items():
        payload[f"cl_{name}_intrinsic_smooth"] = np.asarray(value)
    for name, value in product["spherical_8r_cls"].items():
        payload[f"cl_{name}_spherical_8r_diagnostic"] = np.asarray(value)
    for name, value in product["powers"].items():
        payload[f"power_{name}"] = np.asarray(value)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--map", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--skip-catalog-sha", action="store_true")
    parser.add_argument("--n-projected-radius", type=int, default=256)
    parser.add_argument("--ell-max", type=int, default=1535)
    args = parser.parse_args()
    product = build_noiseless_intrinsic_theory(
        args.config,
        args.map,
        verify_catalog_sha=not args.skip_catalog_sha,
        n_projected_radius=args.n_projected_radius,
        ell_max=args.ell_max,
    )
    output = save_theory_product(args.output, product)
    print(output)


if __name__ == "__main__":
    main()
