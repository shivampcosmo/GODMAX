"""Shared, dependency-light helpers for the BaryonForge--GODMAX comparison."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml


os.environ.setdefault("JAX_ENABLE_X64", "True")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

MAP_KEYS = ("map_ymap", "map_kappa_cmb")
MAP_PRODUCT_SCHEMA = "baryonforge_godmax_native_maps_v1"
MAP_SEMANTICS = "halo-only y and halo-only CMB kappa; no two-halo or unbound matter field"
NOISE_POLICY = "none; deterministic noiseless halo-profile maps"
MASS_PROXY_SEMANTICS = "Interpolated_N * ParticleMassHMsun, treated as M200c"
PROVISIONAL_STATUS = "provisional_mass_proxy"
PROVISIONAL_REASONS = (
    "catalog halo mass is Interpolated_N * ParticleMassHMsun treated as M200c",
    "native GODMAX and BaryonForge five-R200c footprints are nominally, not bitwise, matched",
)
CATALOG_COSMOLOGY_KEYS = {
    "H0": "H0",
    "Omega_M": "Omega_M",
    "Omega_b": "Omega_b",
    "sigma8": "sigma8",
    "ns": "ns",
    "w0": "w0",
}
NATIVE_NORMALIZATION_VARIANT = "native_8r_v1"
ASYMPTOTIC_NORMALIZATION_VARIANT = "asymptotic_total_mass_v1"
ASYMPTOTIC_PROFILES_CLASS_PATH = (
    "matched_godmax_profiles.AsymptoticNormalizationProfiles"
)
ASYMPTOTIC_INTEGRATION_METHOD = "gauss_legendre_log"
ASYMPTOTIC_INTEGRATION_POINTS = 64
NATIVE_CORE_INTEGRATION_POINTS = 64
NATIVE_PROJECTION_VARIANT = "native_legacy_projection_v1"
MATCHED_PROJECTION_VARIANT = "physical_table_cosh_100mpc_v1"


def load_yaml(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data).__name__}.")
    return data


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    config = load_yaml(config_path)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(path: str | Path, config_path: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if config_path is not None:
        beside_config = Path(config_path).expanduser().resolve().parent / candidate
        if beside_config.exists():
            return beside_config.resolve()
    return (REPO_ROOT / candidate).resolve()


def godmax_normalization_variant(config: Mapping[str, Any]) -> str:
    """Return the explicit GODMAX normalization path selected by the run."""

    variant = str(
        config.get("profiles", {}).get(
            "normalization_variant", NATIVE_NORMALIZATION_VARIANT
        )
    )
    supported = {
        NATIVE_NORMALIZATION_VARIANT,
        ASYMPTOTIC_NORMALIZATION_VARIANT,
    }
    if variant not in supported:
        raise ValueError(
            f"Unsupported profiles.normalization_variant={variant!r}; "
            f"expected one of {sorted(supported)}."
        )
    return variant


def godmax_profiles_class_path(config: Mapping[str, Any]) -> str | None:
    """Return the optional comparison subclass without importing JAX."""

    variant = godmax_normalization_variant(config)
    if variant == NATIVE_NORMALIZATION_VARIANT:
        return None
    return ASYMPTOTIC_PROFILES_CLASS_PATH


def profile_integration_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe both finite proxies for the Schneider zero-to-infinity domain."""

    config_path = config.get("_config_path")
    godmax = load_yaml(resolve_path(config["profiles"]["godmax_params"], config_path))
    baryonforge = load_yaml(
        resolve_path(config["profiles"]["baryonforge_params"], config_path)
    )
    analysis = godmax["analysis"]
    numerics = baryonforge["numerics"]
    variant = godmax_normalization_variant(config)
    if variant == NATIVE_NORMALIZATION_VARIANT:
        rmax_r200c = 8.0
        class_name = "get_radial_profiles.Profiles"
        extended_method = "uniform_log_trapezoid"
        extended_points = int(analysis["num_points_trapz_int"])
    else:
        rmax_r200c = float(analysis["comparison_extended_profile_rmax_r200c"])
        class_name = ASYMPTOTIC_PROFILES_CLASS_PATH
        extended_method = str(
            analysis["comparison_extended_profile_integration_method"]
        )
        extended_points = int(analysis["comparison_extended_profile_num_points"])
    core_points = int(analysis["num_points_trapz_int"])
    return {
        "physical_definition": (
            "Schneider component masses at r -> infinity and HSE P(r -> infinity)=0"
        ),
        "godmax": {
            "normalization_variant": variant,
            "profiles_class_fqname": class_name,
            "r_min_R200c": 0.01,
            "r_max_R200c": rmax_r200c,
            "core_integration_method": "uniform_log_trapezoid",
            "core_num_points": core_points,
            "num_points_trapz_int": core_points,
            "extended_integration_method": extended_method,
            "extended_num_points": extended_points,
            "max_simultaneous_integration_nodes": max(
                core_points, extended_points
            ),
            "quadrature_rule_storage_bytes": (
                2 * extended_points * np.dtype(np.float64).itemsize
                if extended_method == ASYMPTOTIC_INTEGRATION_METHOD
                else 0
            ),
            "radius_unit": "comoving Mpc/h after multiplication by R200c",
        },
        "baryonforge": {
            "r_min_Mpc": float(numerics["r_min_int_Mpc"]),
            "r_max_Mpc": float(numerics["r_max_int_Mpc"]),
            "r_steps": int(numerics["r_steps"]),
            "radius_unit": "comoving Mpc",
        },
    }


def godmax_projection_variant(config: Mapping[str, Any]) -> str:
    """Return the explicit projected-profile numerical path selected by a run."""

    variant = str(
        config.get("profiles", {}).get(
            "projection_variant", NATIVE_PROJECTION_VARIANT
        )
    )
    supported = {NATIVE_PROJECTION_VARIANT, MATCHED_PROJECTION_VARIANT}
    if variant not in supported:
        raise ValueError(
            f"Unsupported profiles.projection_variant={variant!r}; "
            f"expected one of {sorted(supported)}."
        )
    return variant


def projected_profile_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the finite support and quadrature used by both projectors."""

    config_path = config.get("_config_path")
    godmax = load_yaml(resolve_path(config["profiles"]["godmax_params"], config_path))
    baryonforge = load_yaml(
        resolve_path(config["profiles"]["baryonforge_params"], config_path)
    )
    halo = godmax["halo_params"]
    analysis = godmax["analysis"]
    numerics = baryonforge["numerics"]
    adapter = baryonforge.get("adapter", {})
    h = float(godmax["sim_params"]["cosmo"]["H0"]) / 100.0
    godmax_support_hmpc = float(halo["rmax"])
    return {
        "projection_variant": godmax_projection_variant(config),
        "physical_definition": (
            "2 times the line-of-sight integral of the physical 3D profile, "
            "with no extrapolation beyond the finite comoving profile support"
        ),
        "godmax": {
            "method": str(
                analysis.get(
                    "projected_profile_integration_method", "legacy_log_radius"
                )
            ),
            "num_points": int(analysis.get("num_points_projected_profile", 32)),
            "los_max_comoving_Mpc": (
                None
                if analysis.get("projected_profile_los_max_comoving_mpc") is None
                else float(analysis["projected_profile_los_max_comoving_mpc"])
            ),
            "table_rmin_comoving_hMpc": float(halo["rmin"]),
            "table_rmax_comoving_hMpc": godmax_support_hmpc,
            "table_rmax_comoving_Mpc": godmax_support_hmpc / h,
            "table_nr": int(halo["nr"]),
            "transverse_radius_unit": "physical Mpc/h",
        },
        "baryonforge": {
            "method": str(
                adapter.get(
                    "projected_profile_integration_method",
                    "BaseBFGProfiles._projected_realspace",
                )
            ),
            "num_points": (
                None
                if adapter.get("projected_profile_num_points") is None
                else int(adapter["projected_profile_num_points"])
            ),
            "los_max_comoving_Mpc": (
                None
                if adapter.get("projected_profile_los_max_comoving_Mpc") is None
                else float(adapter["projected_profile_los_max_comoving_Mpc"])
            ),
            "proj_cutoff_comoving_Mpc": float(numerics["proj_cutoff_Mpc"]),
            "n_per_decade": int(numerics["n_per_decade_proj"]),
            "transverse_radius_unit": "comoving Mpc",
        },
    }


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(path),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__array__"):
        return np.asarray(value).tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_is_dirty(path: str | Path) -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(path),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def comparison_source_manifest() -> dict[str, str]:
    """Hash all local Python sources that can affect either native map product."""

    relative_paths = {
        "GODMAX/notebooks/xDESI/abacus_lightcone_catalog.py",
        "GODMAX/notebooks/xDESI/abacus_pasting_helpers.py",
        "GODMAX/notebooks/xDESI/baryonforge_compare/common.py",
        "GODMAX/notebooks/xDESI/baryonforge_compare/measure_statistics.py",
        "GODMAX/notebooks/xDESI/baryonforge_compare/matched_godmax_profiles.py",
        "GODMAX/notebooks/xDESI/baryonforge_compare/paint_godmax.py",
        "GODMAX/notebooks/xDESI/baryonforge_compare/paint_baryonforge.py",
    }
    for path in (WORKSPACE_ROOT / "GODMAX" / "src").rglob("*.py"):
        relative = path.relative_to(WORKSPACE_ROOT).as_posix()
        if "arxiv" not in path.relative_to(WORKSPACE_ROOT / "GODMAX" / "src").parts:
            relative_paths.add(relative)
    for path in (WORKSPACE_ROOT / "BaryonForge" / "BaryonForge").rglob("*.py"):
        relative_paths.add(path.relative_to(WORKSPACE_ROOT).as_posix())
    manifest = {}
    for relative in sorted(relative_paths):
        path = WORKSPACE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison source is missing: {path}")
        manifest[relative] = sha256_file(path)
    return manifest


_RUNTIME_VERSION_CACHE: dict[str, Any] | None = None


def runtime_version_manifest() -> dict[str, Any]:
    """Bind provenance to the modules Python actually imports.

    Distribution metadata alone is not sufficient: an environment may contain
    duplicate or stale ``*.dist-info`` directories whose reported version does
    not match the import selected by ``sys.path``.  Probe imports in an isolated
    child process so importing JAX here cannot initialize the parent process
    before its configured platform is applied.  The distribution version is
    retained only as auxiliary diagnostic information.
    """

    global _RUNTIME_VERSION_CACHE
    if _RUNTIME_VERSION_CACHE is not None:
        return copy.deepcopy(_RUNTIME_VERSION_CACHE)

    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "h5py": "h5py",
        "astropy": "astropy",
        "healpy": "healpy",
        "pyccl": "pyccl",
        "jax": "jax",
        "jaxlib": "jaxlib",
        "jax-cosmo": "jax_cosmo",
        "interpax": "interpax",
        "joblib": "joblib",
        "numba": "numba",
        "pymaster": "pymaster",
        "PyYAML": "yaml",
        "tqdm": "tqdm",
    }
    probe = r"""
import importlib
import json
import pathlib
import sys

modules = json.loads(sys.argv[1])
result = {}
for distribution, module_name in modules.items():
    try:
        module = importlib.import_module(module_name)
        raw_version = getattr(module, "__version__", None)
        if raw_version is None:
            raw_version = getattr(module, "VERSION", None)
        module_file = getattr(module, "__file__", None)
        result[distribution] = {
            "import_status": "ok",
            "module": module_name,
            "imported_version": None if raw_version is None else str(raw_version),
            "resolved_file": None if module_file is None else str(pathlib.Path(module_file).resolve()),
        }
    except Exception as exc:
        result[distribution] = {
            "import_status": "error",
            "module": module_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
print("__RUNTIME_IMPORT_PROBE__" + json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(distributions, sort_keys=True)],
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "__RUNTIME_IMPORT_PROBE__"
    payload_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
        None,
    )
    if payload_line is None:
        raise RuntimeError(
            "Runtime import probe did not return its JSON payload; stdout was "
            f"{completed.stdout!r}, stderr was {completed.stderr!r}."
        )
    imported = json.loads(payload_line[len(marker) :])

    manifest: dict[str, Any] = {
        "python": {
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": sha256_file(Path(sys.executable).resolve()),
        }
    }
    for distribution, module_name in distributions.items():
        record = dict(imported[distribution])
        try:
            record["distribution_metadata_version"] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            record["distribution_metadata_version"] = None
        resolved_file = record.get("resolved_file")
        record["resolved_file_sha256"] = (
            sha256_file(resolved_file)
            if resolved_file is not None and Path(resolved_file).is_file()
            else None
        )
        imported_version = record.get("imported_version")
        metadata_version = record.get("distribution_metadata_version")
        record["metadata_matches_import"] = (
            None
            if imported_version is None or metadata_version is None
            else str(imported_version) == str(metadata_version)
        )
        record.setdefault("module", module_name)
        manifest[distribution] = record

    _RUNTIME_VERSION_CACHE = copy.deepcopy(manifest)
    return copy.deepcopy(manifest)


def effective_godmax_config_manifest(
    config: Mapping[str, Any],
    catalog_attrs: Mapping[str, Any],
    selected_redshift: np.ndarray,
    *,
    is_cmb_lensing: bool = False,
    log10_mass_min: float | None = None,
) -> dict:
    """Record the exact expanded dictionaries used to construct GODMAX arrays."""

    default_path = resolve_path(config["godmax"]["default_params"], config.get("_config_path"))
    override_path = resolve_path(config["godmax"]["xdesi_params"], config.get("_config_path"))
    redshift = np.asarray(selected_redshift, dtype=np.float64)
    if redshift.ndim != 1 or redshift.size == 0:
        raise ValueError("selected_redshift must be a non-empty one-dimensional array.")
    xdesi_dir = REPO_ROOT / "notebooks" / "xDESI"
    if str(xdesi_dir) not in sys.path:
        sys.path.insert(0, str(xdesi_dir))
    from abacus_pasting_helpers import (
        effective_grid_canonicalization_contract,
        prepare_godmax_config,
    )

    sim_params, halo_params, analysis, other_params = prepare_godmax_config(
        config,
        catalog_attrs,
        is_cmb_lensing=bool(is_cmb_lensing),
        z_max=float(np.max(redshift)),
        log10_mass_min=float(
            catalog_attrs["log10_m_min_hmsun"]
            if log10_mass_min is None
            else log10_mass_min
        ),
    )

    external_inputs = {}
    for label, key in (
        ("xdesi_fit_summary", "xdesi_fit_summary"),
        ("source_nz_fits", "source_nz_fits"),
    ):
        path = resolve_path(config["godmax"][key], config.get("_config_path"))
        external_inputs[label] = {"path": str(path), "sha256": sha256_file(path)}
    external_inputs["source_nz_selection"] = {
        "hdu": config["godmax"]["source_nz_hdu"],
        "z_column": config["godmax"]["source_nz_z_column"],
        "bin_column": config["godmax"]["source_nz_bin_column"],
        "floor": float(config["godmax"]["source_nz_floor"]),
    }
    manifest = {
        "effective_dictionaries": jsonable(
            {
                "sim_params": sim_params,
                "halo_params": halo_params,
                "analysis": analysis,
                "other_params": other_params,
            }
        ),
        "default_params_path": str(default_path),
        "default_params_sha256": sha256_file(default_path),
        "override_params_path": str(override_path),
        "override_params_sha256": sha256_file(override_path),
        "selected_z_min": float(np.min(redshift)),
        "selected_z_max": float(np.max(redshift)),
        "is_cmb_lensing": bool(is_cmb_lensing),
        "log10_mass_min": float(
            catalog_attrs["log10_m_min_hmsun"]
            if log10_mass_min is None
            else log10_mass_min
        ),
        "external_inputs": external_inputs,
        "runtime_map_flags": jsonable(config["pasting"]),
        "generated_grid_canonicalization": (
            effective_grid_canonicalization_contract()
        ),
    }
    manifest["sha256"] = sha256_json(manifest)
    return manifest


def shared_map_contract(
    config: Mapping[str, Any],
    catalog_attrs: Mapping[str, Any],
    selected_redshift: np.ndarray,
) -> dict:
    """Build provenance that must be byte-identical across both map producers."""

    config_path = resolve_path(config["_config_path"])
    godmax_params = resolve_path(config["profiles"]["godmax_params"], config_path)
    baryonforge_params = resolve_path(config["profiles"]["baryonforge_params"], config_path)
    catalog_path = resolve_path(config["catalog"]["output_h5"], config_path)
    redshift = np.asarray(selected_redshift, dtype=np.float64)
    if redshift.ndim != 1 or redshift.size == 0:
        raise ValueError("selected_redshift must be a non-empty one-dimensional array.")
    halo_count = int(catalog_attrs.get("selection_rows", redshift.size))
    if halo_count <= 0:
        raise ValueError(f"Catalog selection must contain halos, got {halo_count}.")
    source_manifest = comparison_source_manifest()
    effective_godmax = effective_godmax_config_manifest(
        config, catalog_attrs, redshift
    )
    runtime_versions = runtime_version_manifest()
    integration_contract = profile_integration_contract(config)
    projection_contract = projected_profile_contract(config)
    return {
        "schema": MAP_PRODUCT_SCHEMA,
        "comparison_config_path": str(config_path),
        "comparison_config_sha256": sha256_file(config_path),
        "godmax_params_path": str(godmax_params),
        "godmax_params_sha256": sha256_file(godmax_params),
        "baryonforge_params_path": str(baryonforge_params),
        "baryonforge_params_sha256": sha256_file(baryonforge_params),
        "effective_godmax_config_sha256": effective_godmax["sha256"],
        "effective_godmax_config_manifest": effective_godmax,
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256_json(source_manifest),
        "godmax_git_sha": git_revision(WORKSPACE_ROOT / "GODMAX"),
        "baryonforge_git_sha": git_revision(WORKSPACE_ROOT / "BaryonForge"),
        "godmax_git_dirty": git_is_dirty(WORKSPACE_ROOT / "GODMAX"),
        "baryonforge_git_dirty": git_is_dirty(WORKSPACE_ROOT / "BaryonForge"),
        "runtime_versions": runtime_versions,
        "profile_integration_contract": integration_contract,
        "projected_profile_contract": projection_contract,
        "smoke_table": False,
        "max_halos": None,
        "baryonforge_splitjoin_n_jobs": int(config["baryonforge"]["n_jobs"]),
        "godmax_pixel_workers": int(config["pasting"]["pixel_workers"]),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "selection_predicate": str(config["catalog"]["predicate"]),
        "mass_cut_predicate": str(config["catalog"]["predicate"]),
        "halo_count": halo_count,
        "n_halos_painted": halo_count,
        "complete_catalog_paint": True,
        "nside": int(config["pasting"]["nside"]),
        "ordering": str(config["sky_patch"]["ordering"]).upper(),
        "max_paint_R200c_factor": float(
            config["pasting"]["max_paint_R200c_factor"]
        ),
        "smooth_profiles": bool(config["pasting"]["smooth_profiles"]),
        "halo_only": True,
        "z_min": float(np.min(redshift)),
        "z_max": float(np.max(redshift)),
        "h": float(catalog_attrs["h"]),
        "H0": float(catalog_attrs["H0"]),
        "Omega_M": float(catalog_attrs["Omega_M"]),
        "Omega_b": float(catalog_attrs["Omega_b"]),
        "map_semantics": MAP_SEMANTICS,
        "noise_policy": NOISE_POLICY,
        "mass_proxy_semantics": MASS_PROXY_SEMANTICS,
        "provisional_status": PROVISIONAL_STATUS,
        "provisional_reasons": PROVISIONAL_REASONS,
        "analysis_mask_policy": "none in map product; one inner-cap mask is applied by measure_statistics.py",
        "cmb_source_redshift": 1100.0,
    }


def current_map_contract(config: Mapping[str, Any]) -> dict:
    """Hash the current catalog and all other inputs into one map contract."""

    catalog_path = resolve_path(
        config["catalog"]["output_h5"], config.get("_config_path")
    )
    with h5py.File(catalog_path, "r") as handle:
        catalog_attrs = dict(handle.attrs)
        selected_redshift = np.asarray(handle["z"][:], dtype=np.float64)
    return shared_map_contract(config, catalog_attrs, selected_redshift)


def assert_map_contract_unchanged(
    frozen: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject publication when any hashed run input changed during painting."""

    if canonical_json(frozen) == canonical_json(current):
        return
    keys = sorted(set(frozen).union(current))
    changed = [
        key
        for key in keys
        if canonical_json(frozen.get(key)) != canonical_json(current.get(key))
    ]
    raise RuntimeError(
        f"{context}: comparison inputs changed after the map contract was frozen; "
        f"refusing to publish the product. Changed contract keys: {changed}"
    )


def load_config_and_freeze_map_contract(
    path: str | Path,
) -> tuple[dict, dict]:
    """Load one self-consistent config/input snapshot for a long map run."""

    first_config = load_config(path)
    frozen = current_map_contract(first_config)
    verified_config = load_config(path)
    if canonical_json(first_config) != canonical_json(verified_config):
        raise RuntimeError(
            "Comparison configuration changed while the initial map contract was "
            "being frozen; refusing to start the painter."
        )
    assert_map_contract_unchanged(
        frozen,
        current_map_contract(verified_config),
        context="Initial map-contract validation",
    )
    return verified_config, frozen


def read_h5_attrs(handle: h5py.File | h5py.Group) -> dict:
    return {str(key): jsonable(value) for key, value in handle.attrs.items()}


def catalog_cosmology(attrs: Mapping[str, Any]) -> dict:
    missing = [source for source in CATALOG_COSMOLOGY_KEYS.values() if source not in attrs]
    if missing:
        raise KeyError(f"Catalog is missing cosmology attribute(s): {missing}")
    return {
        "H0": float(attrs["H0"]),
        "Omega_m": float(attrs["Omega_M"]),
        "Omega_b": float(attrs["Omega_b"]),
        "h": float(attrs["H0"]) / 100.0,
        "sigma8": float(attrs["sigma8"]),
        "n_s": float(attrs["ns"]),
        "w0": float(attrs["w0"]),
        "wa": 0.0,
    }


def cap_mask(nside: int, ra_deg: float, dec_deg: float, radius_deg: float) -> np.ndarray:
    import healpy as hp

    mask = np.zeros(hp.nside2npix(int(nside)), dtype=np.float64)
    vector = hp.ang2vec(float(ra_deg), float(dec_deg), lonlat=True)
    pixels = hp.query_disc(
        int(nside), vector, math.radians(float(radius_deg)), inclusive=False, nest=False
    )
    mask[np.asarray(pixels, dtype=np.int64)] = 1.0
    return mask


def read_map_file(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read either native output using the common map dataset names."""

    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        group = handle["maps"] if "maps" in handle else handle
        maps: dict[str, np.ndarray] = {}
        aliases = {
            "map_ymap": ("map_ymap", "y", "ymap"),
            "map_kappa_cmb": ("map_kappa_cmb", "kappa_cmb", "kappa"),
        }
        for canonical, candidates in aliases.items():
            for candidate in candidates:
                if candidate in group:
                    maps[canonical] = np.asarray(group[candidate][:], dtype=np.float64)
                    break
            if canonical not in maps:
                raise KeyError(f"{path} lacks maps/{canonical}; tried aliases {candidates}.")
        attrs = read_h5_attrs(handle)
        provenance: dict[str, Any] = {}
        for group_name in ("provenance", "comparison_provenance"):
            if group_name not in handle:
                continue
            group_attrs = read_h5_attrs(handle[group_name])
            encoded = group_attrs.pop("json", None)
            if encoded is not None:
                try:
                    decoded = json.loads(str(encoded))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid provenance JSON in {path}:{group_name}."
                    ) from exc
                if isinstance(decoded, Mapping):
                    provenance.update(jsonable(decoded))
            provenance.update(group_attrs)
        if provenance:
            attrs["provenance"] = provenance
        attrs["path"] = str(path)
    sizes = {array.size for array in maps.values()}
    if len(sizes) != 1:
        raise ValueError(f"Map fields in {path} have inconsistent lengths: {sorted(sizes)}")
    return maps, attrs


def baryonforge_profile_kwargs(params: Mapping[str, Any]) -> dict:
    profile = copy.deepcopy(dict(params["profile_parameters"]))
    numerics = params["numerics"]
    profile.update(
        {
            "cutoff": float(numerics["cutoff_Mpc"]),
            "proj_cutoff": float(numerics["proj_cutoff_Mpc"]),
            "r_min_int": float(numerics["r_min_int_Mpc"]),
            "r_max_int": float(numerics["r_max_int_Mpc"]),
            "r_steps": int(numerics["r_steps"]),
            "padding_lo_proj": float(numerics["padding_lo_proj"]),
            "padding_hi_proj": float(numerics["padding_hi_proj"]),
            "n_per_decade_proj": int(numerics["n_per_decade_proj"]),
            "use_fftlog_projection": bool(numerics["use_fftlog_projection"]),
        }
    )
    return profile


def _close(actual: float, expected: float, *, rtol: float = 2.0e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=rtol)


def validate_parameter_crosswalk(config: Mapping[str, Any]) -> dict:
    """Validate only identities that are controllable from the two parameter files."""

    config_path = config.get("_config_path")
    gpath = resolve_path(config["profiles"]["godmax_params"], config_path)
    bpath = resolve_path(config["profiles"]["baryonforge_params"], config_path)
    godmax_runtime_path = resolve_path(config["godmax"]["xdesi_params"], config_path)
    baryonforge_runtime_path = resolve_path(
        config["baryonforge"]["params"], config_path
    )
    godmax = load_yaml(gpath)
    bforge = load_yaml(bpath)
    gsim = godmax["sim_params"]
    ghalo = godmax["halo_params"]
    ganalysis = godmax["analysis"]
    bp = bforge["profile_parameters"]
    bc = bforge["cosmology"]
    h = float(bc["h"])

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "ok": bool(condition),
                "actual": jsonable(actual),
                "expected": jsonable(expected),
            }
        )

    normalization_variant = str(
        config.get("profiles", {}).get(
            "normalization_variant", NATIVE_NORMALIZATION_VARIANT
        )
    )
    supported_normalization_variants = {
        NATIVE_NORMALIZATION_VARIANT,
        ASYMPTOTIC_NORMALIZATION_VARIANT,
    }
    check(
        "godmax.normalization_variant_supported",
        normalization_variant in supported_normalization_variants,
        normalization_variant,
        sorted(supported_normalization_variants),
    )
    if normalization_variant == ASYMPTOTIC_NORMALIZATION_VARIANT:
        check(
            "godmax.asymptotic_rmax_R200c",
            ganalysis.get("comparison_extended_profile_rmax_r200c") is not None
            and _close(
                ganalysis["comparison_extended_profile_rmax_r200c"], 128.0
            ),
            ganalysis.get("comparison_extended_profile_rmax_r200c"),
            128.0,
        )
        check(
            "godmax.native_core_num_points",
            int(ganalysis.get("num_points_trapz_int", -1))
            == NATIVE_CORE_INTEGRATION_POINTS,
            ganalysis.get("num_points_trapz_int"),
            NATIVE_CORE_INTEGRATION_POINTS,
        )
        check(
            "godmax.asymptotic_integration_method",
            ganalysis.get("comparison_extended_profile_integration_method")
            == ASYMPTOTIC_INTEGRATION_METHOD,
            ganalysis.get("comparison_extended_profile_integration_method"),
            ASYMPTOTIC_INTEGRATION_METHOD,
        )
        check(
            "godmax.asymptotic_num_points",
            int(ganalysis.get("comparison_extended_profile_num_points", -1))
            == ASYMPTOTIC_INTEGRATION_POINTS,
            ganalysis.get("comparison_extended_profile_num_points"),
            ASYMPTOTIC_INTEGRATION_POINTS,
        )
        check(
            "baryonforge.asymptotic_rmin_Mpc",
            _close(bforge["numerics"]["r_min_int_Mpc"], 1.0e-6),
            bforge["numerics"]["r_min_int_Mpc"],
            1.0e-6,
        )
        check(
            "baryonforge.asymptotic_rmax_Mpc",
            _close(bforge["numerics"]["r_max_int_Mpc"], 100.0),
            bforge["numerics"]["r_max_int_Mpc"],
            100.0,
        )
        check(
            "baryonforge.asymptotic_r_steps",
            int(bforge["numerics"]["r_steps"]) == 512,
            bforge["numerics"]["r_steps"],
            512,
        )

    projection_variant = godmax_projection_variant(config)
    if projection_variant == MATCHED_PROJECTION_VARIANT:
        projected = projected_profile_contract(config)
        godmax_projection = projected["godmax"]
        baryonforge_projection = projected["baryonforge"]
        check(
            "godmax.projected_profile_integration_method",
            godmax_projection["method"] == "physical_table_cosh",
            godmax_projection["method"],
            "physical_table_cosh",
        )
        check(
            "godmax.projected_profile_points",
            int(godmax_projection["num_points"]) == 128,
            godmax_projection["num_points"],
            128,
        )
        check(
            "godmax.projected_profile_table_nr",
            int(godmax_projection["table_nr"]) == 128,
            godmax_projection["table_nr"],
            128,
        )
        check(
            "godmax.projected_profile_table_rmax_comoving_hMpc",
            _close(godmax_projection["table_rmax_comoving_hMpc"], 70.0),
            godmax_projection["table_rmax_comoving_hMpc"],
            70.0,
        )
        check(
            "projection.common_los_max_comoving_Mpc",
            _close(
                godmax_projection["los_max_comoving_Mpc"],
                baryonforge_projection["los_max_comoving_Mpc"],
            ),
            godmax_projection["los_max_comoving_Mpc"],
            baryonforge_projection["los_max_comoving_Mpc"],
        )
        check(
            "baryonforge.projected_profile_integration_method",
            baryonforge_projection["method"] == "nonsingular_gauss_legendre",
            baryonforge_projection["method"],
            "nonsingular_gauss_legendre",
        )
        check(
            "baryonforge.projected_profile_points",
            int(baryonforge_projection["num_points"]) == 128,
            baryonforge_projection["num_points"],
            128,
        )
        check(
            "baryonforge.projected_profile_native_cutoff",
            _close(baryonforge_projection["proj_cutoff_comoving_Mpc"], 100.0),
            baryonforge_projection["proj_cutoff_comoving_Mpc"],
            100.0,
        )

    # The top-level profile paths define the hashed comparison contract while
    # the backend-specific aliases are consumed by the actual model builders.
    # Refuse a future YAML edit that makes those two views point at different
    # files, even if the headline parameter crosswalk still happens to pass.
    check(
        "paths.godmax_runtime_params",
        godmax_runtime_path == gpath,
        str(godmax_runtime_path),
        str(gpath),
    )
    check(
        "paths.baryonforge_runtime_params",
        baryonforge_runtime_path == bpath,
        str(baryonforge_runtime_path),
        str(bpath),
    )

    gc = gsim["cosmo"]
    check("cosmology.flat", gc["flat"] is True, gc["flat"], True)
    for gkey, bkey, scale in (
        ("H0", "h", 100.0),
        ("Om0", "Omega_m", 1.0),
        ("Ob0", "Omega_b", 1.0),
        ("sigma8", "sigma8", 1.0),
        ("ns", "n_s", 1.0),
        ("w0", "w0", 1.0),
    ):
        check(
            f"cosmology.{gkey}",
            _close(gc[gkey], float(bc[bkey]) * scale),
            gc[gkey],
            float(bc[bkey]) * scale,
        )

    direct = {
        "theta_ej_0": "theta_ej",
        "nu_theta_ej_M": "mu_theta_ej",
        "nu_theta_ej_z": "nu_theta_ej",
        "theta_co_0": "theta_co",
        "nu_theta_co_M": "mu_theta_co",
        "nu_theta_co_z": "nu_theta_co",
        "mu_beta": "mu_beta",
        "nu_z": "nu_M_c",
        "gamma_rhogas": "gamma",
        "delta_rhogas": "delta",
        "epsilon_rt": "epsilon",
        "alpha_nt": "alpha_nt",
        "beta_nt": "nu_nt",
        "n_nt": "gamma_nt",
        "A_starcga": "A",
        "eta_star": "eta",
    }
    for gkey, bkey in direct.items():
        check(f"profile.{gkey}->{bkey}", _close(gsim[gkey], bp[bkey]), bp[bkey], gsim[gkey])

    for gkey, bkey in (
        ("log10_Mstar0_theta_ej", "M_theta_ej"),
        ("log10_Mstar0_theta_co", "M_theta_co"),
        ("log10_Mc0", "M_c"),
        ("log10_M1_starcga", "M1"),
    ):
        expected = 10.0 ** float(gsim[gkey]) / h
        check(f"mass_pivot.{gkey}->{bkey}", _close(bp[bkey], expected), bp[bkey], expected)

    eta_delta = float(gsim["eta_cga"]) - float(gsim["eta_star"])
    for key, expected in (
        ("tau", gsim["eta_star"]),
        ("eta_delta", eta_delta),
        ("tau_delta", eta_delta),
        ("epsilon_h", 0.015),
    ):
        check(f"simple_stars.{key}", _close(bp[key], expected), bp[key], expected)

    check("godmax.model_galaxies", ganalysis["model_galaxies"] is False, ganalysis["model_galaxies"], False)
    check("godmax.backreaction", ganalysis["backreaction"] is False, ganalysis["backreaction"], False)
    check("godmax.nfw_trunc", gsim["nfw_trunc"] is True, gsim["nfw_trunc"], True)
    check("godmax.model_tSZ", ganalysis["model_tSZ"] is True, ganalysis["model_tSZ"], True)
    check(
        "godmax.is_cmb_lensing",
        ganalysis["is_cmb_lensing"] is True,
        ganalysis["is_cmb_lensing"],
        True,
    )
    check("baryonforge.no_backreaction_limit", _close(bp["a"], 0.0), bp["a"], 0.0)
    check("mass_definition", int(ghalo["mdef_Delta"]) == 200 and bforge["mass_definition"] == "200c", [ghalo["mdef_Delta"], bforge["mass_definition"]], [200, "200c"])
    check("concentration", ghalo["conc_model"] == "Duffy08" and bforge["concentration_model"] == "Duffy08", [ghalo["conc_model"], bforge["concentration_model"]], ["Duffy08", "Duffy08"])
    check("electron_pressure_factor", _close(bforge["adapter"]["electron_pressure_factor"], 1.0 / 1.932), bforge["adapter"]["electron_pressure_factor"], 1.0 / 1.932)
    check("baryonforge.profile_family", bforge["profile_family"] == "Schneider19", bforge["profile_family"], "Schneider19")
    check("baryonforge.mass_input_unit", bforge["mass_input_unit"] == "Msun", bforge["mass_input_unit"], "Msun")
    check("baryonforge.radius_input_unit", bforge["radius_input_unit"] == "comoving_Mpc", bforge["radius_input_unit"], "comoving_Mpc")
    check("baryonforge.cosmology.wa", _close(bc["wa"], 0.0), bc["wa"], 0.0)

    # Degrees of freedom that must remain disabled for the simple analytic
    # crosswalk.  Checking only the headline parameters would allow future
    # config edits to reintroduce unmatched mass/redshift/concentration terms.
    check("godmax.nu_M", _close(gsim["nu_M"], 0.0), gsim["nu_M"], 0.0)
    for key in ("nu_theta_ej_c", "nu_theta_co_c"):
        check(f"godmax.{key}", _close(gsim[key], 0.0), gsim[key], 0.0)
    for key in ("zeta_M_c", "zeta_theta_ej", "zeta_theta_co"):
        check(f"baryonforge.{key}", _close(bp[key], 0.0), bp[key], 0.0)
    for family in ("gamma", "delta"):
        for prefix in ("mu", "nu", "zeta"):
            key = f"{prefix}_{family}"
            check(f"baryonforge.{key}", _close(bp[key], 0.0), bp[key], 0.0)
    stellar_zero_keys = (
        "nu_A",
        "nu_M1",
        "nu_eta",
        "nu_eta_delta",
        "nu_tau",
        "nu_tau_delta",
        "zeta_A",
        "zeta_M1",
        "zeta_eta",
        "zeta_eta_delta",
        "zeta_tau",
        "zeta_tau_delta",
        "mu_epsilon_h",
        "nu_epsilon_h",
        "zeta_epsilon_h",
    )
    for key in stellar_zero_keys:
        check(f"baryonforge.{key}", _close(bp[key], 0.0), bp[key], 0.0)
    check("baryonforge.cdelta", bp["cdelta"] is None, bp["cdelta"], None)
    adapter = bforge["adapter"]
    check(
        "adapter.one_halo_matter_assembly",
        adapter["matter_assembly"] == "gas_plus_stars_plus_collisionless_one_halo",
        adapter["matter_assembly"],
        "gas_plus_stars_plus_collisionless_one_halo",
    )
    check(
        "adapter.no_two_halo",
        adapter["include_two_halo"] is False,
        adapter["include_two_halo"],
        False,
    )
    check(
        "adapter.no_global_renormalization",
        adapter["include_darkmatterbaryon_global_renormalization"] is False,
        adapter["include_darkmatterbaryon_global_renormalization"],
        False,
    )
    check(
        "adapter.cmb_source_redshift",
        _close(adapter["cmb_source_redshift"], 1100.0),
        adapter["cmb_source_redshift"],
        1100.0,
    )
    check(
        "baryonforge.realspace_projection",
        bforge["numerics"]["use_fftlog_projection"] is False,
        bforge["numerics"]["use_fftlog_projection"],
        False,
    )
    realspace_projection = not bool(bforge["numerics"]["use_fftlog_projection"])
    expected_remove_extra_a = not realspace_projection
    check(
        "adapter.thermal_sz_projected_a_dispatch",
        bool(bforge["adapter"]["remove_thermal_sz_extra_projected_a"])
        is expected_remove_extra_a,
        bforge["adapter"]["remove_thermal_sz_extra_projected_a"],
        expected_remove_extra_a,
    )
    check(
        "adapter.undo_tabulated_projected_a",
        bforge["adapter"]["undo_tabulated_profile_projected_a"] is True,
        bforge["adapter"]["undo_tabulated_profile_projected_a"],
        True,
    )

    pasting = config["pasting"]
    check("paint_cutoff", _close(pasting["max_paint_R200c_factor"], 5.0), pasting["max_paint_R200c_factor"], 5.0)
    check("internal_smoothing", pasting["smooth_profiles"] is False and bforge["adapter"]["internal_pixel_smoothing"] is False, [pasting["smooth_profiles"], bforge["adapter"]["internal_pixel_smoothing"]], [False, False])
    check("ordering", config["sky_patch"]["ordering"] == "RING", config["sky_patch"]["ordering"], "RING")
    check("strict_mass_predicate", config["catalog"]["predicate"] == "M200c_hMsun > 1.0e13", config["catalog"]["predicate"], "M200c_hMsun > 1.0e13")
    check(
        "map_fields",
        pasting["get_ymap"] is True
        and pasting["get_kappa_cmb"] is True
        and pasting["get_kappa_wl"] is False,
        [pasting["get_ymap"], pasting["get_kappa_cmb"], pasting["get_kappa_wl"]],
        [True, True, False],
    )
    check(
        "godmax.active_painter_path",
        pasting["use_fused_profile_maps"] is True
        and pasting["use_multi_kappa_maps"] is True
        and pasting["return_sparse_maps"] is True
        and pasting["store_projected_matter_maps"] is False
        and pasting["get_baryonifiedmap"] is False
        and pasting["get_galmap"] is False,
        [
            pasting["use_fused_profile_maps"],
            pasting["use_multi_kappa_maps"],
            pasting["return_sparse_maps"],
            pasting["store_projected_matter_maps"],
            pasting["get_baryonifiedmap"],
            pasting["get_galmap"],
        ],
        [True, True, True, False, False, False],
    )
    validation = config["validation"]
    for key in (
        "require_catalog_hash_match",
        "require_no_internal_smoothing",
        "require_ring_ordering",
        "require_complete_catalog_paint",
        "require_current_contract_match",
        "require_nonzero_production_maps",
    ):
        check(f"validation.{key}", validation[key] is True, validation[key], True)
    check(
        "execution.baryonforge_splitjoin_n_jobs",
        int(config["baryonforge"]["n_jobs"]) == 8,
        config["baryonforge"]["n_jobs"],
        8,
    )
    check(
        "execution.godmax_pixel_workers",
        int(pasting["pixel_workers"]) == 1,
        pasting["pixel_workers"],
        1,
    )
    production_splits = pasting["num_splits_by_nside"]
    check(
        "execution.godmax_num_splits_nside1024",
        int(production_splits.get(1024, production_splits.get("1024"))) == 1,
        production_splits.get(1024, production_splits.get("1024")),
        1,
    )

    failed = [item for item in checks if not item["ok"]]
    return {
        "ok": not failed,
        "godmax_params": str(gpath),
        "baryonforge_params": str(bpath),
        "checks": checks,
        "failed": failed,
    }
