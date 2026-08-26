#!/usr/bin/env python3
"""Fail-closed configuration and cheap validation for the fast SBI paste."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Mapping

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.special import j0

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "notebooks" / "xDESI", THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from three_probe_mock_contract import canonical_json_sha256, sha256_array, sha256_file  # noqa: E402
from three_probe_projected_operator import (  # noqa: E402
    painter_log_interpolate,
    painter_rp_nodes,
    project_physical_profile_cosh,
    projected_painter_transform,
)
from validate_three_probe_projected_operator import selected_profiles  # noqa: E402
from validate_three_probe_resolved_theory import catalog_nbar, load_contract  # noqa: E402


GRID = {"nr": 48, "nM": 24, "nz": 48}

# The five sampled gas parameters, with the contract's prior bounds.  These are the
# ONLY keys a paste run may override, and they all live in ``sim_params``.
SAMPLED_GAS_PARAMETERS = {
    "theta_ej_0": (0.5, 8.0),
    "alpha_nt": (0.0, 0.5),
    "mu_beta": (0.005, 1.5),
    "theta_co_0": (0.001, 0.5),
    "nu_theta_ej_M": (-1.0, 1.0),
}
MAP_DATASETS = ("map_ymap", "map_tau", "map_kappa_cmb")
TARGET_Z = (0.3, 0.4, 0.5)
TARGET_LOGM = (np.log10(5.0e11), 13.0, 14.0, 15.0)


def _provenance(config_path: pathlib.Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "config_sha256": sha256_file(config_path),
        "validation_script_sha256": sha256_file(pathlib.Path(__file__)),
        "projected_operator_sha256": sha256_file(THIS_DIR / "three_probe_projected_operator.py"),
        "pasting_helper_sha256": sha256_file(REPO_ROOT / "notebooks" / "xDESI" / "abacus_pasting_helpers.py"),
    }


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def validate_fast_paste_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen fast settings without constructing GODMAX."""

    theory = config["resolved_theory"]
    paste = config["pasting"]
    exact = {
        "mode": (theory.get("mode"), "map_matched_resolved"),
        "unresolved_completion": (theory.get("unresolved_completion"), False),
        "projector": (paste.get("projected_profile_integration_method"), "physical_table_cosh"),
        "n_los": (int(paste.get("num_points_projected_profile", -1)), 32),
        "max_paint": (float(paste.get("max_paint_R200c_factor", -1)), 8.0),
        "smooth_profiles": (paste.get("smooth_profiles"), True),
        "smoothing_kernel": (paste.get("profile_smoothing_kernel"), "gaussian"),
        "smoothing_pixel_fraction": (float(paste.get("profile_smoothing_fwhm_pixel_fraction", -1.0)), 0.5),
        "smoothing_method": (paste.get("profile_smoothing_method"), "real_space_gaussian"),
        "save_projection_kernels": (paste.get("save_projection_kernels"), True),
        "preallocate": (paste.get("jax", {}).get("preallocate"), False),
        "strict_combine": (paste.get("require_strict_combine_contract"), True),
        "freeze_galaxy_catalog": (paste.get("freeze_galaxy_catalog"), True),
        "selective_allocation": (paste.get("allocate_only_requested_maps"), True),
    }
    failed = {key: values for key, values in exact.items() if values[0] != values[1]}
    if failed:
        raise ValueError(f"Fast-paste contract mismatch: {failed}")
    observed_grid = {key: int(paste[f"halo_profile_{key}"]) for key in GRID}
    if observed_grid != GRID:
        raise ValueError(f"Fast-paste grid differs from frozen baseline: {observed_grid} != {GRID}")
    requested = tuple(
        key for key, enabled in (
            ("map_ymap", paste.get("get_ymap")),
            ("map_tau", paste.get("get_taumap")),
            ("map_kappa_cmb", paste.get("get_kappa_cmb")),
        ) if enabled is True
    )
    forbidden_flags = (
        "get_kszmap", "get_kappa_wl", "get_baryonifiedmap", "store_projected_matter_maps"
    )
    if requested != MAP_DATASETS or any(paste.get(key) is not False for key in forbidden_flags):
        raise ValueError("Fast paste must allocate only y, tau, and CMB-kappa maps")
    if tuple(int(value) for value in paste["supported_nside"]) != (512, 1024):
        raise ValueError("Only the frozen nside=512 development and nside=1024 control are supported")
    if int(paste["comparison_lmax"]) != 1535:
        raise ValueError("The frozen comparison lmax must be 1535")
    return {"grid": observed_grid, "map_datasets": list(requested)}


def prepare_fast_paste_godmax_config(
    config: Mapping[str, Any],
    catalog_attrs: Mapping[str, Any] | None,
    *,
    config_path: pathlib.Path | None,
    is_cmb_lensing: bool = False,
    z_max: float | None = None,
    log10_mass_min: float | None = None,
):
    """Return catalog-bound dictionaries before any GODMAX constructor runs."""

    del is_cmb_lensing, z_max, log10_mass_min
    if config_path is None:
        raise ValueError("The fast-paste factory requires the experiment config path")
    validate_fast_paste_config(config)
    _, theory, _, _, support, cosmology = load_contract(pathlib.Path(config_path), False)
    if catalog_attrs is None:
        raise ValueError("Catalog attrs are required for fail-closed cosmology matching")
    for key in ("H0", "Omega_M", "Omega_b", "sigma8", "ns", "w0"):
        if key not in catalog_attrs:
            raise ValueError(f"Missing catalog cosmology attribute {key}")

    params_path = pathlib.Path(theory["default_params_path"])
    params = _load_yaml(params_path)
    sim = copy.deepcopy(params["sim_params"])
    gas_override = resolve_gas_parameter_override(config, sim)
    halo = copy.deepcopy(params["halo_params"])
    analysis = copy.deepcopy(params["analysis"])
    other = copy.deepcopy(params["other_params"])
    sim.update(gas_override["applied"])
    sim["cosmo"] = dict(cosmology)
    sim["init_power"] = True

    paste = config["pasting"]
    halo.update({
        "lg10_Mmin": float(np.log10(support.mass_min_hmsun)),
        "lg10_Mmax": float(np.log10(support.mass_max_hmsun)),
        "nM": int(paste["halo_profile_nM"]),
        "zmin": float(support.z_min),
        "zmax": float(support.z_max),
        "nz": int(paste["halo_profile_nz"]),
        "nr": int(paste["halo_profile_nr"]),
        "mdef_Delta": 200,
        "hmf_model": "T10",
    })
    analysis.update({
        "symbolic_hmf": False,
        "symbolic_pk": False,
        "zmin_for_Cls": float(support.z_min),
        "zmax_for_Cls": float(support.z_max),
        "nz_for_Cls": int(paste["halo_profile_nz"]),
        "projected_profile_integration_method": "physical_table_cosh",
        "num_points_projected_profile": 32,
        "projected_profile_los_max_comoving_mpc": None,
    })

    kernel_path = pathlib.Path(theory["validation_output_dir"]) / theory["lens_kernel"]["output_name"]
    with h5py.File(kernel_path, "r") as handle:
        if str(handle.attrs["catalog_file_sha256"]) != str(theory["catalog_file_sha256"]):
            raise ValueError("Lens kernel catalog SHA differs from the paste contract")
        if str(handle.attrs["cosmology_sha256"]) != canonical_json_sha256(cosmology):
            raise ValueError("Lens kernel cosmology differs from the paste contract")
        group = handle["primary"]
        z_lens = np.asarray(group["z"], dtype=np.float64)
        nz_lens = np.asarray(group["nz"], dtype=np.float64)
        if str(group.attrs["kernel_array_sha256"]) != sha256_array(z_lens, nz_lens):
            raise ValueError("Lens-kernel array hash mismatch")
    if abs(float(np.trapz(nz_lens, z_lens)) - 1.0) > 1.0e-6:
        raise ValueError("Lens kernel is not normalized to 1e-6")
    if z_lens[0] != support.z_min or z_lens[-1] != support.z_max:
        raise ValueError("Lens-kernel and halo supports differ")
    analysis["nz_lens_info_dict"] = {
        "nbins_lens": 1,
        "z_edges_bins_lens": [[support.z_min, support.z_max]],
        "z_array_lens": z_lens.tolist(),
        "nz0": nz_lens.tolist(),
    }
    z_nbar, nbar = catalog_nbar(kernel_path, cosmology)
    analysis["nbar_gal_comoving_zarray"] = z_nbar.tolist()
    analysis["nbar_gal_comoving_val"] = nbar.tolist()
    return sim, halo, analysis, other


def resolve_gas_parameter_override(
    config: Mapping[str, Any], default_sim_params: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and resolve the per-run gas-parameter override, fail-closed.

    Before this existed there was no way to paste at any parameter point other
    than whatever sat in ``params_default.yaml`` -- and those values are exactly
    the point the frozen mock was painted at.  A parameter scan launched without
    an override would therefore have silently pasted N copies of the same map,
    with nothing in the output to reveal it.  Hence:

    * only the five sampled keys are accepted, by name;
    * every value must lie inside the contract's prior bounds;
    * ``pasting.require_gas_parameter_overrides: true`` makes omission an error,
      so a generated campaign config cannot fall back to the defaults;
    * the resolved point and its canonical hash are returned for stamping into
      the paste product, so "which theta produced this map" is answerable from
      the map alone.

    Absent the flag and the override this is a no-op, which is what the theory
    forward model and the all-sky yy builder need: they carry their own theta.
    """

    paste = dict(config.get("pasting", {}) or {})
    required = bool(paste.get("require_gas_parameter_overrides", False))
    raw = paste.get("gas_parameter_overrides", None)

    if raw is None:
        if required:
            raise ValueError(
                "pasting.require_gas_parameter_overrides is true but no "
                "pasting.gas_parameter_overrides block was supplied; refusing to "
                "paste at the params_default.yaml point by omission"
            )
        baseline = {name: float(default_sim_params[name]) for name in SAMPLED_GAS_PARAMETERS}
        return {
            "applied": {},
            "resolved": baseline,
            "source": "params_default.yaml (no override requested)",
            "sha256": canonical_json_sha256(baseline),
            "is_override": False,
        }

    if not isinstance(raw, Mapping):
        raise ValueError("pasting.gas_parameter_overrides must be a mapping")
    unknown = sorted(set(map(str, raw)) - set(SAMPLED_GAS_PARAMETERS))
    if unknown:
        raise ValueError(
            f"pasting.gas_parameter_overrides contains non-sampled key(s) {unknown}; "
            f"only {sorted(SAMPLED_GAS_PARAMETERS)} may be overridden"
        )
    missing = sorted(set(SAMPLED_GAS_PARAMETERS) - set(map(str, raw)))
    if missing:
        raise ValueError(
            f"pasting.gas_parameter_overrides must declare all five sampled "
            f"parameters explicitly; missing {missing}"
        )

    applied: dict[str, float] = {}
    for name, (low, high) in SAMPLED_GAS_PARAMETERS.items():
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"gas_parameter_overrides[{name}] must be a real number, got {value!r}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"gas_parameter_overrides[{name}] is not finite")
        if not (low <= value <= high):
            raise ValueError(
                f"gas_parameter_overrides[{name}]={value!r} is outside the contract "
                f"prior [{low}, {high}]"
            )
        if name not in default_sim_params:
            raise ValueError(f"{name} is absent from params_default sim_params; cannot override")
        applied[name] = value

    return {
        "applied": applied,
        "resolved": dict(applied),
        "source": "pasting.gas_parameter_overrides",
        "sha256": canonical_json_sha256(applied),
        "is_override": True,
    }


def preflight(config_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    config = _load_yaml(config_path)
    static = validate_fast_paste_config(config)
    _, theory, catalog_path, source_files, support, cosmology = load_contract(config_path, True)
    _, _, analysis, _ = prepare_fast_paste_godmax_config(
        config, _catalog_attrs(catalog_path), config_path=config_path
    )
    payload = {
        "status": "PASS",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "catalog": str(catalog_path.resolve()),
        "catalog_sha256": theory["catalog_file_sha256"],
        "source_files": source_files,
        "support": support.__dict__,
        "cosmology": cosmology,
        "cosmology_sha256": canonical_json_sha256(cosmology),
        "static": static,
        "projector": analysis["projected_profile_integration_method"],
        "n_los": analysis["num_points_projected_profile"],
        "provenance": _provenance(config_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fast_paste_preflight.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _catalog_attrs(path: pathlib.Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return dict(handle.attrs)


def validate_projection(config_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    """Compare 32 and 64 LOS points on identical baseline GODMAX profiles."""

    from base_class import base_class
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    config = _load_yaml(config_path)
    catalog_path = pathlib.Path(config["resolved_theory"]["catalog_path"])
    sim, halo, analysis, other = prepare_fast_paste_godmax_config(
        config, _catalog_attrs(catalog_path), config_path=config_path
    )
    base = base_class(sim, halo, analysis, other)
    profiles = Profiles(sim, halo, analysis, other, base_class_obj=base)
    pkz = get_Pkz(sim, halo, analysis, other, Profiles_obj=profiles)
    radius = np.asarray(pkz.r_array, dtype=np.float64)
    source_rp = painter_rp_nodes(radius)
    mass_grid = np.asarray(pkz.M_array, dtype=np.float64)
    z_grid = np.asarray(pkz.z_array, dtype=np.float64)
    k = np.concatenate(([0.0], np.geomspace(1.0e-4, 2.0, 64)))
    records = []
    all_delta = []
    for target_z in TARGET_Z:
        iz = int(np.argmin(np.abs(z_grid - target_z)))
        z = float(z_grid[iz])
        a = 1.0 / (1.0 + z)
        for target_logm in TARGET_LOGM:
            im = int(np.argmin(np.abs(np.log10(mass_grid) - target_logm)))
            r200 = float(pkz.r200c_mat[iz, im])
            aperture = 8.0 * r200 * a
            rp = np.geomspace(max(aperture * 1.0e-7, 1.0e-10), aperture, 192)
            for field, (_, physical, factor) in selected_profiles(pkz, iz, im).items():
                transforms = {}
                for n_los in (32, 64):
                    nodes = project_physical_profile_cosh(radius, physical, z, source_rp, n_los=n_los)
                    sigma = painter_log_interpolate(source_rp, nodes, rp)
                    transforms[n_los] = projected_painter_transform(
                        k, rp, sigma, z, aperture, physical_to_theory_volume_factor=factor
                    )
                scale = max(abs(float(transforms[64][0])), np.finfo(np.float64).tiny)
                delta = np.abs(transforms[32] - transforms[64]) / scale
                all_delta.append(delta)
                records.append({
                    "field": field, "z": z, "mass_hmsun": float(mass_grid[im]),
                    "median_normalized_delta": float(np.median(delta)),
                    "max_normalized_delta": float(np.max(delta)),
                })
    combined = np.concatenate(all_delta)
    median = float(np.median(combined))
    maximum = float(np.max(combined))
    passed = bool(np.all(np.isfinite(combined)) and median < 0.005 and maximum < 0.02)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "metric": "abs(T32-T64)/abs(T64(k=0))",
        "k_max_hmpc": 2.0,
        "preregistered_median_limit": 0.005,
        "preregistered_max_limit": 0.02,
        "median": median,
        "max": maximum,
        "records": records,
        "grid": GRID,
        "provenance": _provenance(config_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fast_paste_los32_vs64.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(combined, bins=60)
    ax.axvline(0.02, color="crimson", linestyle="--", label="pre-registered max")
    ax.set_xlabel(r"$|T_{32}-T_{64}|/|T_{64}(k=0)|$")
    ax.set_ylabel("samples")
    ax.set_title("physical_table_cosh LOS convergence; k <= 2 h/Mpc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fast_paste_los32_vs64.png", dpi=180)
    plt.close(fig)
    if not passed:
        raise RuntimeError(f"32-point LOS validation failed: median={median:.3e}, max={maximum:.3e}")
    return json_path


def _build_pkz_with_grid(config_path: pathlib.Path, grid: Mapping[str, int]):
    from base_class import base_class
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    config = _load_yaml(config_path)
    catalog_path = pathlib.Path(config["resolved_theory"]["catalog_path"])
    sim, halo, analysis, other = prepare_fast_paste_godmax_config(
        config, _catalog_attrs(catalog_path), config_path=config_path
    )
    halo.update({key: int(value) for key, value in grid.items()})
    analysis["nz_for_Cls"] = int(grid["nz"])
    base = base_class(sim, halo, analysis, other)
    profiles = Profiles(sim, halo, analysis, other, base_class_obj=base)
    return get_Pkz(sim, halo, analysis, other, Profiles_obj=profiles)


def _common_target_transforms(pkz) -> dict[str, np.ndarray]:
    radius = np.asarray(pkz.r_array, dtype=np.float64)
    z_grid = np.asarray(pkz.z_array, dtype=np.float64)
    logm_grid = np.log10(np.asarray(pkz.M_array, dtype=np.float64))
    source_rp = painter_rp_nodes(radius)
    k = np.concatenate(([0.0], np.geomspace(1.0e-4, 2.0, 64)))
    r200_interp = RegularGridInterpolator(
        (z_grid, logm_grid), np.asarray(pkz.r200c_mat, dtype=np.float64), bounds_error=True
    )
    field_arrays = {
        "y": np.asarray(pkz.y3d_mat, dtype=np.float64),
        "e": np.asarray(pkz.ne_mat_physical, dtype=np.float64),
        "m": np.asarray(pkz.rho_dmb_mat, dtype=np.float64)
        * (1.0 + z_grid)[None, :, None] ** 3 / float(pkz.rhom_0),
    }
    interpolators = {
        field: RegularGridInterpolator(
            (np.log(radius), z_grid, logm_grid), np.log(values), bounds_error=True
        )
        for field, values in field_arrays.items()
    }
    outputs = {field: [] for field in field_arrays}
    for z in TARGET_Z:
        a = 1.0 / (1.0 + z)
        for logm in TARGET_LOGM:
            r200 = float(r200_interp((z, logm)))
            aperture = 8.0 * r200 * a
            rp = np.geomspace(max(aperture * 1.0e-7, 1.0e-10), aperture, 192)
            points = np.column_stack((np.log(radius), np.full(radius.size, z), np.full(radius.size, logm)))
            for field, interpolator in interpolators.items():
                physical = np.exp(interpolator(points))
                nodes = project_physical_profile_cosh(radius, physical, z, source_rp, n_los=64)
                sigma = painter_log_interpolate(source_rp, nodes, rp)
                factor = a ** -3 if field == "y" else 1.0
                outputs[field].append(projected_painter_transform(
                    k, rp, sigma, z, aperture, physical_to_theory_volume_factor=factor
                ))
    return {field: np.stack(values) for field, values in outputs.items()}


def validate_grid(config_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    variants = {
        "mass_48": {"nr": 48, "nM": 48, "nz": 48},
        "redshift_96": {"nr": 48, "nM": 24, "nz": 96},
        "radial_96": {"nr": 96, "nM": 24, "nz": 48},
    }
    baseline = _common_target_transforms(_build_pkz_with_grid(config_path, GRID))
    records = []
    passed = True
    for label, grid in variants.items():
        transformed = _common_target_transforms(_build_pkz_with_grid(config_path, grid))
        for field in baseline:
            scale = np.maximum(np.abs(transformed[field][:, :1]), np.finfo(np.float64).tiny)
            delta = np.abs(transformed[field] - baseline[field]) / scale
            maximum = float(np.max(delta))
            records.append({
                "variant": label, "grid": grid, "field": field,
                "median_normalized_delta": float(np.median(delta)),
                "max_normalized_delta": maximum,
            })
            passed = passed and bool(np.all(np.isfinite(delta)) and maximum < 0.05)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "metric": "abs(T_variant-T_baseline)/abs(T_variant(k=0))",
        "baseline_grid": GRID,
        "preregistered_max_limit": 0.05,
        "records": records,
        "provenance": _provenance(config_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fast_paste_grid_convergence.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{item['variant']}:{item['field']}" for item in records]
    values = [item["max_normalized_delta"] for item in records]
    ax.bar(np.arange(len(values)), values)
    ax.axhline(0.05, color="crimson", linestyle="--", label="pre-registered max")
    ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("maximum normalized transform change")
    ax.set_title("One-axis-at-a-time profile-grid convergence; k <= 2 h/Mpc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fast_paste_grid_convergence.png", dpi=180)
    plt.close(fig)
    if not passed:
        raise RuntimeError("Fast-paste profile grid failed the pre-registered 5% gate")
    return path


def validate_paste_contract(config_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    """Construct the exact smoothed painter tables and save its theory kernels."""

    from base_class import base_class
    from get_radial_profiles import Profiles
    from get_sim_maps import get_sim_map, setup_sim_map
    from notebooks.xDESI.abacus_pasting_helpers import _paste_kernel_bundle

    config = _load_yaml(config_path)
    catalog_path = pathlib.Path(config["resolved_theory"]["catalog_path"])
    sim, halo, analysis, other = prepare_fast_paste_godmax_config(
        config, _catalog_attrs(catalog_path), config_path=config_path
    )
    base = base_class(sim, halo, analysis, other)
    profiles = Profiles(sim, halo, analysis, other, base_class_obj=base)
    paste = config["pasting"]
    setup_params = {
        "nside": int(paste["nside"]),
        "smooth_profiles": True,
        "profile_smoothing_fwhm_pixel_fraction": float(
            paste["profile_smoothing_fwhm_pixel_fraction"]
        ),
        "profile_smoothing_method": str(paste["profile_smoothing_method"]),
        "profile_smoothing_quadrature_points": int(
            paste["profile_smoothing_quadrature_points"]
        ),
        "profile_smoothing_radial_sigma_cutoff": float(
            paste["profile_smoothing_radial_sigma_cutoff"]
        ),
        "profile_timing": True,
        "use_fused_profile_maps": True,
        "return_sparse_maps": True,
        "store_projected_matter_maps": False,
        "get_galmap": True,
        "get_ymap": True,
        "get_kSZmap": False,
        "get_taumap": True,
        "get_kappamap": False,
        "get_multi_kappamap": True,
        "multi_kappa_source_bins": [],
        "multi_kappa_include_cmb": True,
        "get_baryonifiedmap": False,
    }
    setup = setup_sim_map(sim, halo, analysis, other, setup_params, Profiles_obj=profiles)
    hod_smoke = get_sim_map.get_hod_params(
        setup,
        jnp.asarray(setup.M_array[len(setup.M_array) // 2], dtype=jnp.float64),
        jnp.asarray(setup.z_array[len(setup.z_array) // 2], dtype=jnp.float32),
    )
    hod_smoke = np.asarray(hod_smoke, dtype=np.float64)
    if hod_smoke.shape != (2,) or not np.all(np.isfinite(hod_smoke)):
        raise ValueError(f"Mixed-precision HOD interpolation smoke failed: {hod_smoke}")
    arrays, attrs = _paste_kernel_bundle(setup, analysis, config)
    required_smoothing_steps = (
        "ymap_profile_smoothing",
        "ne_map_profile_smoothing",
        "kappa_map_profile_smoothing",
    )
    missing = [key for key in required_smoothing_steps if key not in setup.timing_results]
    if missing:
        raise ValueError(f"Requested profile smoothing did not execute for every field: {missing}")
    for name in ("y2D_mat_physical", "ne2D_mat_physical", "rhom2D_mat_physical"):
        value = np.asarray(getattr(setup, name), dtype=np.float64)
        n_nonfinite = int(np.count_nonzero(~np.isfinite(value)))
        n_zero = int(np.count_nonzero(value == 0.0))
        n_negative = int(np.count_nonzero(value < 0.0))
        if n_nonfinite or n_zero or n_negative:
            finite = value[np.isfinite(value)]
            finite_min = float(np.min(finite)) if finite.size else float("nan")
            finite_max = float(np.max(finite)) if finite.size else float("nan")
            raise ValueError(
                f"Smoothed painter table {name} is not finite and positive: "
                f"nonfinite={n_nonfinite}, zero={n_zero}, negative={n_negative}, "
                f"finite_min={finite_min:.9e}, finite_max={finite_max:.9e}"
            )

    refined_params = dict(setup_params)
    refined_params["profile_timing"] = False
    refined_params["profile_smoothing_quadrature_points"] = 2 * int(
        setup_params["profile_smoothing_quadrature_points"]
    )
    refined = setup_sim_map(sim, halo, analysis, other, refined_params, Profiles_obj=profiles)
    field_tables = {
        "y": ("y2D_mat_physical_orig", "y2D_mat_physical"),
        "tau": ("ne2D_mat_physical_orig", "ne2D_mat_physical"),
        "kappa_cmb": ("rhom2D_mat_physical_orig", "rhom2D_mat_physical"),
    }
    convergence = {}
    flux = {}
    transfer = {}
    transfer_ell = np.asarray([0, 100, 250, 500, 750, 1000, 1250, 1535], dtype=np.int32)
    transfer_target = np.exp(
        -0.5 * (transfer_ell.astype(np.float64) * float(setup.sigma_val)) ** 2
    )
    sample_z = (0, len(setup.z_array) // 2, len(setup.z_array) - 1)
    sample_m = (0, len(setup.M_array) // 2, len(setup.M_array) - 1)
    for field, (original_name, smooth_name) in field_tables.items():
        original = np.asarray(getattr(setup, original_name), dtype=np.float64)
        smooth = np.asarray(getattr(setup, smooth_name), dtype=np.float64)
        smooth_refined = np.asarray(getattr(refined, smooth_name), dtype=np.float64)
        scale = np.maximum(np.max(np.abs(smooth_refined), axis=0, keepdims=True), 1.0e-300)
        normalized_delta = np.abs(smooth - smooth_refined) / scale
        convergence[field] = {
            "max_normalized_abs_64_vs_128": float(np.max(normalized_delta)),
            "median_normalized_abs_64_vs_128": float(np.median(normalized_delta)),
        }
        flux_errors = []
        transfer_errors = []
        for iz in range(len(setup.z_array)):
            theta = np.asarray(setup.rp_array / setup.DA_array[iz], dtype=np.float64)
            flux_original = np.trapz(2.0 * np.pi * theta[:, None] * original[:, iz, :], theta, axis=0)
            flux_smooth = np.trapz(2.0 * np.pi * theta[:, None] * smooth[:, iz, :], theta, axis=0)
            flux_errors.extend(np.abs(flux_smooth / flux_original - 1.0).tolist())
        for iz in sample_z:
            theta_nodes = np.asarray(setup.rp_array / setup.DA_array[iz], dtype=np.float64)
            theta_dense = np.geomspace(theta_nodes[0], theta_nodes[-1], 4096)
            for im in sample_m:
                input_dense = np.exp(np.interp(
                    np.log(theta_dense), np.log(theta_nodes),
                    np.log(np.maximum(original[:, iz, im], np.finfo(np.float32).tiny))
                ))
                smooth_dense = np.exp(np.interp(
                    np.log(theta_dense), np.log(theta_nodes), np.log(smooth[:, iz, im])
                ))
                phase = transfer_ell[:, None] * theta_dense[None, :]
                fin = 2.0 * np.pi * np.trapz(
                    theta_dense[None, :] * input_dense[None, :] * j0(phase),
                    theta_dense,
                    axis=-1,
                )
                fout = 2.0 * np.pi * np.trapz(
                    theta_dense[None, :] * smooth_dense[None, :] * j0(phase),
                    theta_dense,
                    axis=-1,
                )
                transfer_errors.append(np.abs(fout - transfer_target * fin) / abs(fin[0]))
        transfer_errors = np.asarray(transfer_errors)
        flux[field] = {
            "max_abs_fractional_change_within_table": float(np.max(flux_errors)),
            "median_abs_fractional_change_within_table": float(np.median(flux_errors)),
        }
        transfer[field] = {
            "ell": transfer_ell.tolist(),
            "max_zero_mode_normalized_residual_by_ell": np.max(transfer_errors, axis=0).tolist(),
            "max_zero_mode_normalized_residual_ell_le_1000": float(
                np.max(transfer_errors[:, transfer_ell <= 1000])
            ),
        }
    if max(item["max_normalized_abs_64_vs_128"] for item in convergence.values()) > 0.01:
        raise ValueError(f"Gaussian smoothing quadrature is not converged to 1%: {convergence}")
    if max(item["max_abs_fractional_change_within_table"] for item in flux.values()) > 0.02:
        raise ValueError(f"Gaussian smoothing changes in-table flux by more than 2%: {flux}")
    if max(item["max_zero_mode_normalized_residual_ell_le_1000"] for item in transfer.values()) > 0.05:
        raise ValueError(f"Gaussian smoothing transfer closure exceeds 5% through ell=1000: {transfer}")

    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "fast_paste_projection_contract.h5"
    with h5py.File(h5_path.with_suffix(".h5.tmp"), "w") as handle:
        group = handle.create_group("kernels")
        for key, value in arrays.items():
            group.create_dataset(key, data=value)
        for key, value in attrs.items():
            group.attrs[key] = value
        handle.attrs["config_sha256"] = sha256_file(config_path)
        handle.attrs["validation_script_sha256"] = sha256_file(pathlib.Path(__file__))
        handle.attrs["pasting_helper_sha256"] = sha256_file(
            REPO_ROOT / "notebooks" / "xDESI" / "abacus_pasting_helpers.py"
        )
        handle.attrs["get_sim_maps_sha256"] = sha256_file(REPO_ROOT / "src" / "get_sim_maps.py")
        handle.attrs["status"] = "PASS"
    os.replace(h5_path.with_suffix(".h5.tmp"), h5_path)

    payload = {
        "status": "PASS",
        "projection_contract_h5": str(h5_path.resolve()),
        "projection_contract_h5_sha256": sha256_file(h5_path),
        "nside": int(paste["nside"]),
        "profile_smoothing_fwhm_arcmin": attrs["profile_smoothing_fwhm_arcmin"],
        "profile_smoothing_pixel_resolution_arcmin": attrs["profile_smoothing_pixel_resolution_arcmin"],
        "profile_smoothing_fwhm_pixel_fraction": attrs["profile_smoothing_fwhm_pixel_fraction"],
        "profile_smoothing_sigma_rad": attrs["profile_smoothing_sigma_rad"],
        "lens_nz_normalization": attrs["lens_nz_normalization"],
        "kernel_dataset_sha256_json": attrs["kernel_dataset_sha256_json"],
        "smoothing_timings_s": {
            key: float(setup.timing_results[key]) for key in required_smoothing_steps
        },
        "mixed_precision_hod_smoke": {
            "mean_ncen": float(hod_smoke[0]),
            "mean_nsat": float(hod_smoke[1]),
            "status": "PASS",
        },
        "smoothing_quadrature_convergence": convergence,
        "smoothing_flux_conservation": flux,
        "smoothing_transfer_closure": transfer,
        "provenance": _provenance(config_path),
        "get_sim_maps_sha256": sha256_file(REPO_ROOT / "src" / "get_sim_maps.py"),
    }
    json_path = output_dir / "fast_paste_projection_contract.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(arrays["lens_redshift"], arrays["lens_nz"])
    axes[0].set(xlabel="lens z", ylabel="normalized dN/dz", title="Catalog lens kernel")
    axes[1].plot(arrays["halo_redshift"], arrays["cmb_lensing_efficiency_Wkappa_hmpc"])
    axes[1].set(xlabel="lens z", ylabel="Wkappa [h/Mpc]", title="CMB lensing efficiency")
    axes[2].plot(arrays["profile_smoothing_ell"], arrays["profile_smoothing_Bell"])
    axes[2].set(xlabel="ell", ylabel="B_ell", title="Half-pixel Gaussian")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fast_paste_projection_contract.png", dpi=180)
    plt.close(fig)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "validate-projection", "validate-grid", "validate-paste-contract"),
    )
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    output = args.output_dir or pathlib.Path(config["resolved_theory"]["validation_output_dir"]) / "fast_paste"
    if args.command == "preflight":
        result = preflight(args.config, output)
    elif args.command == "validate-projection":
        result = validate_projection(args.config, output)
    elif args.command == "validate-grid":
        result = validate_grid(args.config, output)
    else:
        result = validate_paste_contract(args.config, output)
    print(result)


if __name__ == "__main__":
    main()
