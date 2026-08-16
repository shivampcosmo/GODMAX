"""Fast, tSZ-only GODMAX pasting for the DES cluster halo lightcone.

The public entry points are :func:`preflight_catalog`, :func:`run_tsz_paste`,
and :func:`load_tsz_map`.  JAX and GODMAX imports are deliberately deferred so
that x64 can be enabled before either package creates an array.

The input catalog does not record an SO mass definition.  This module therefore
never presents ``M_interp`` as a measured M200c: it is a provisional mass proxy
that the configured profile calculation conditionally treats as M200c.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import h5py
import numpy as np
import yaml


# This executes before any deferred JAX/GODMAX import in this module.
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-des-cluster")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path(__file__).resolve()
RHO_CRIT_0_HUNITS = 2.77536627245708e11
C_KMS = 299792.458
SCHEMA = "des_cluster_tsz_paste_v1"
MAP_DATASET = "maps/map_ymap"
EXPECTED_C000_COSMOLOGY = {
    "H0": 67.36,
    "Om0": 0.315192,
    "Ob0": 0.049301692328524445,
    "sigma8": 0.807952,
    "ns": 0.9649,
    "w0": -1.0,
}
EXPECTED_C000_OBSERVER_HMPC = np.array([-990.0, -990.0, -990.0], dtype=np.float64)


def _deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return value


def _read_config_tree(path: str | Path, stack: tuple[Path, ...] = ()) -> tuple[dict, list[str]]:
    """Read a complete config or a small ``base_params`` override file."""

    resolved = Path(path).resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"Circular base_params chain: {cycle}")
    cfg = _read_yaml(resolved)
    base_ref = cfg.pop("base_params", None)
    if base_ref is None:
        return cfg, [str(resolved)]
    base_path = Path(str(base_ref))
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    base_cfg, sources = _read_config_tree(base_path, (*stack, resolved))
    return _deep_update(base_cfg, cfg), [*sources, str(resolved)]


def load_params(
    path: str | Path = Path(__file__).with_name("params_tsz.yaml"),
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Read, optionally override, and validate the complete run configuration."""

    cfg, config_sources = _read_config_tree(path)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    cfg["_config_path"] = str(Path(path).resolve())
    cfg["_config_sources"] = config_sources
    validate_params(cfg)
    return cfg


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{label} is missing required keys: {missing}")


def validate_params(cfg: Mapping[str, Any]) -> None:
    """Fail closed on settings that would silently change units or map physics."""

    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema={SCHEMA!r}; got {cfg.get('schema')!r}.")
    _require_keys(
        cfg,
        ("catalog", "cosmology", "profiles", "map", "runtime", "output", "validation"),
        "configuration",
    )
    catalog = cfg["catalog"]
    _require_keys(
        catalog,
        ("path", "dataset", "fields", "selection", "observer_xyz_hmpc", "position_unit", "mass_unit"),
        "catalog",
    )
    if str(catalog["position_unit"]) != "comoving Mpc/h":
        raise ValueError("catalog.position_unit must be exactly 'comoving Mpc/h'.")
    if str(catalog["mass_unit"]) != "Msun/h":
        raise ValueError("catalog.mass_unit must be exactly 'Msun/h'.")
    origin = np.asarray(catalog["observer_xyz_hmpc"], dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("catalog.observer_xyz_hmpc must contain three finite values.")
    if not np.array_equal(origin, EXPECTED_C000_OBSERVER_HMPC):
        raise ValueError(
            "catalog.observer_xyz_hmpc must match the c000 lightcone origin "
            f"{EXPECTED_C000_OBSERVER_HMPC.tolist()}."
        )
    selection = catalog["selection"]
    if selection.get("operator") != ">":
        raise ValueError("Only the strict mass predicate '>' is supported.")
    if not np.isfinite(float(selection["mass_min_hmsun"])):
        raise ValueError("catalog.selection.mass_min_hmsun must be finite.")
    redshift_max = selection.get("redshift_max")
    if selection.get("redshift_max_operator", "<=") != "<=":
        raise ValueError("Only the inclusive redshift predicate '<=' is supported.")
    if redshift_max is not None and (
        not np.isfinite(float(redshift_max)) or float(redshift_max) < 0.0
    ):
        raise ValueError("catalog.selection.redshift_max must be null or finite and nonnegative.")
    if not bool(catalog.get("mass_definition_is_provisional", False)):
        raise ValueError(
            "The source file does not prove an SO mass definition; "
            "catalog.mass_definition_is_provisional must remain true."
        )

    cosmo = cfg["cosmology"]
    _require_keys(cosmo, ("flat", "H0", "Om0", "Ob0", "sigma8", "ns", "w0"), "cosmology")
    if not bool(cosmo["flat"]):
        raise ValueError("Only the configured flat c000 cosmology is supported.")
    if not (0.0 < float(cosmo["Ob0"]) < float(cosmo["Om0"]) < 1.0):
        raise ValueError("Require 0 < Ob0 < Om0 < 1.")
    if float(cosmo["H0"]) <= 0.0:
        raise ValueError("cosmology.H0 must be positive.")
    for key, expected in EXPECTED_C000_COSMOLOGY.items():
        if not np.isclose(float(cosmo[key]), expected, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                f"cosmology.{key}={cosmo[key]!r} does not match the c000 value {expected!r}."
            )

    profiles = cfg["profiles"]
    if profiles.get("class") != "matched_godmax_profiles.AsymptoticNormalizationProfiles":
        raise ValueError("Only the validated AsymptoticNormalizationProfiles class is allowed.")
    if profiles.get("normalization_variant") != "asymptotic_total_mass_v1":
        raise ValueError("Only the validated asymptotic_total_mass_v1 normalization is allowed.")
    analysis = profiles["overrides"]["analysis"]
    if analysis.get("projected_profile_integration_method") != "physical_table_cosh":
        raise ValueError("The tSZ map requires the unit-consistent physical_table_cosh projector.")
    if bool(analysis.get("model_galaxies", True)) or not bool(analysis.get("model_tSZ", False)):
        raise ValueError("This helper is tSZ-only: model_galaxies=false and model_tSZ=true are required.")
    configured_cosmo = profiles["overrides"]["sim_params"].get("cosmo")
    if configured_cosmo is not None:
        for key in ("flat", "H0", "Om0", "Ob0", "sigma8", "ns", "w0"):
            if configured_cosmo[key] != cosmo[key]:
                raise ValueError(f"profiles.overrides.sim_params.cosmo.{key} must match cosmology.{key}.")

    map_cfg = cfg["map"]
    nside = int(map_cfg["nside"])
    if nside <= 0 or nside & (nside - 1):
        raise ValueError("map.nside must be a positive power of two.")
    if str(map_cfg.get("ordering", "")).upper() != "RING":
        raise ValueError("Only HEALPix RING ordering is supported.")
    if float(map_cfg["max_paint_R200c_factor"]) <= 0.0:
        raise ValueError("map.max_paint_R200c_factor must be positive.")
    if bool(map_cfg.get("smooth_profiles", True)):
        raise ValueError("The validated profile path requires map.smooth_profiles=false.")
    if float(map_cfg["pressure_amplitude"]) < 0.0:
        raise ValueError("map.pressure_amplitude must be nonnegative.")
    if map_cfg.get("central_radius_policy") != "extended_projected_grid_no_extrapolation":
        raise ValueError(
            "map.central_radius_policy must be 'extended_projected_grid_no_extrapolation'."
        )
    if map_cfg.get("beam") != "none" or map_cfg.get("noise") != "none":
        raise ValueError("This halo-only product must not silently add a beam or noise.")

    runtime = cfg["runtime"]
    for key in ("halo_chunk_size", "pixel_batch_size", "pair_batch_size", "pixel_workers"):
        if int(runtime[key]) <= 0:
            raise ValueError(f"runtime.{key} must be positive.")
    if runtime.get("pixel_backend") != "healpy_ring":
        raise ValueError("The optimized path currently requires runtime.pixel_backend=healpy_ring.")
    if not bool(runtime.get("jax_enable_x64", False)):
        raise ValueError("runtime.jax_enable_x64 must be true.")
    projected_rmin = float(profiles["projected_radius_min_hmpc"])
    projected_ncentral = int(profiles["projected_radius_num_central_points"])
    if not np.isfinite(projected_rmin) or projected_rmin <= 0.0:
        raise ValueError("profiles.projected_radius_min_hmpc must be finite and positive.")
    if projected_ncentral < 8:
        raise ValueError("profiles.projected_radius_num_central_points must be at least 8.")


def _field_names(cfg: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    fields = cfg["catalog"]["fields"]
    return (
        str(fields["mass"]),
        str(fields["x"]),
        str(fields["y"]),
        str(fields["z_position"]),
        str(fields["redshift"]),
    )


def _expansion_rate(z: np.ndarray, cosmology: Mapping[str, Any]) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    om = float(cosmology["Om0"])
    w0 = float(cosmology["w0"])
    ode = 1.0 - om
    return np.sqrt(om * (1.0 + z) ** 3 + ode * (1.0 + z) ** (3.0 * (1.0 + w0)))


def comoving_distance_hmpc(z: np.ndarray, cosmology: Mapping[str, Any]) -> np.ndarray:
    """Flat-wCDM radial distance in comoving Mpc/h.

    The h factor cancels: chi[Mpc/h] = c/100 integral dz/E(z).
    A dense deterministic table is sufficient for the catalog-frame guard; the
    catalog's measured observer-relative radius is used for actual painting.
    """

    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.empty_like(z)
    if np.any(z < 0.0) or not np.all(np.isfinite(z)):
        raise ValueError("Redshift values must be finite and nonnegative.")
    zmax = float(np.max(z))
    # Fix the table over the complete configured lightcone range so that the
    # answer does not depend on HDF5 chunk boundaries.
    grid_max = max(4.0, zmax * (1.0 + 1.0e-10) + 1.0e-12)
    ngrid = max(32768, int(np.ceil(grid_max * 32768)))
    grid = np.linspace(0.0, grid_max, ngrid)
    inv_e = 1.0 / _expansion_rate(grid, cosmology)
    increments = 0.5 * (inv_e[1:] + inv_e[:-1]) * np.diff(grid)
    integral = np.concatenate(([0.0], np.cumsum(increments)))
    return (C_KMS / 100.0) * np.interp(z, grid, integral)


def _selection_mask(
    mass: np.ndarray,
    redshift: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    apply_mass_cut: bool = True,
    apply_redshift_cut: bool = True,
) -> np.ndarray:
    selection = cfg["catalog"]["selection"]
    keep = np.ones(len(mass), dtype=bool)
    if apply_mass_cut:
        keep &= mass > float(selection["mass_min_hmsun"])
    redshift_max = selection.get("redshift_max")
    if apply_redshift_cut and redshift_max is not None:
        keep &= redshift <= float(redshift_max)
    return keep


def adapt_records(
    records: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    apply_cut: bool = True,
    apply_redshift_cut: bool = True,
) -> dict:
    """Convert one compound HDF5 slice into the canonical painter boundary."""

    mass_name, x_name, y_name, zp_name, redshift_name = _field_names(cfg)
    mass = np.asarray(records[mass_name], dtype=np.float64)
    xyz = np.column_stack((records[x_name], records[y_name], records[zp_name])).astype(np.float64)
    redshift = np.asarray(records[redshift_name], dtype=np.float64)
    finite = np.isfinite(mass) & np.isfinite(redshift) & np.all(np.isfinite(xyz), axis=1)
    if not np.all(finite):
        raise ValueError(f"Catalog chunk contains {int(np.count_nonzero(~finite))} nonfinite rows.")
    keep = _selection_mask(
        mass,
        redshift,
        cfg,
        apply_mass_cut=apply_cut,
        apply_redshift_cut=apply_redshift_cut,
    )
    mass = mass[keep]
    xyz = xyz[keep]
    redshift = redshift[keep]

    origin = np.asarray(cfg["catalog"]["observer_xyz_hmpc"], dtype=np.float64)
    delta = xyz - origin[None, :]
    chi = np.linalg.norm(delta, axis=1)
    if np.any(chi <= 0.0):
        raise ValueError("A halo lies at the observer position; angular coordinates are undefined.")
    ra = np.mod(np.degrees(np.arctan2(delta[:, 1], delta[:, 0])), 360.0)
    dec = np.degrees(np.arcsin(np.clip(delta[:, 2] / chi, -1.0, 1.0)))
    da = chi / (1.0 + redshift)
    ez = _expansion_rate(redshift, cfg["cosmology"])
    r200c = (
        3.0 * mass / (4.0 * np.pi * 200.0 * RHO_CRIT_0_HUNITS * ez**2)
    ) ** (1.0 / 3.0)
    return {
        "ra_deg": ra,
        "dec_deg": dec,
        "z": redshift,
        "M200c_hMsun": mass,
        "log10M200c_hMsun": np.log10(mass),
        "vlos_kms": np.zeros(len(mass), dtype=np.float32),
        "R200c_hMpc": r200c,
        "DA_hMpc": da,
        "chi_hMpc": chi,
        "keep_mask": keep,
    }


def _catalog_attrs(handle: h5py.File) -> dict:
    out: dict[str, Any] = {}
    for key, value in handle.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, np.generic):
            value = value.item()
        out[str(key)] = value
    return out


def preflight_catalog(cfg: Mapping[str, Any]) -> dict:
    """Stream the complete source once and prove its selection/frame contract."""

    path = Path(cfg["catalog"]["path"])
    dataset_name = str(cfg["catalog"]["dataset"])
    mass_name, x_name, y_name, zp_name, redshift_name = _field_names(cfg)
    requested_fields = (mass_name, x_name, y_name, zp_name, redshift_name)
    threshold = float(cfg["catalog"]["selection"]["mass_min_hmsun"])
    scan_chunk = int(cfg["validation"]["preflight_chunk_size"])
    minimum_mass = np.inf
    maximum_mass = -np.inf
    minimum_z = np.inf
    maximum_z = -np.inf
    maximum_distance_error = 0.0
    selected = 0
    rows = 0
    all_source_rows_pass = True
    all_source_rows_pass_mass_cut = True
    selected_index_hasher = hashlib.sha256()

    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"Dataset {dataset_name!r} not found in {path}.")
        dataset = handle[dataset_name]
        if dataset.dtype.names is None:
            raise TypeError(f"{dataset_name!r} must be a compound dataset.")
        missing = sorted(set(requested_fields) - set(dataset.dtype.names))
        if missing:
            raise KeyError(f"Catalog is missing fields: {missing}")
        attrs = _catalog_attrs(handle)
        if bool(cfg["validation"]["require_complete_input"]) and not bool(attrs.get("complete", False)):
            raise ValueError("Input catalog does not declare complete=True.")
        source_threshold = attrs.get("mass_threshold")
        if source_threshold is not None and threshold < float(source_threshold):
            raise ValueError(
                f"Requested mass cut {threshold:g} is below the already-filtered source "
                f"cut {float(source_threshold):g}; the missing halos cannot be recovered."
            )
        rows = int(dataset.shape[0])
        fields_view = dataset.fields(requested_fields)
        for start in range(0, rows, scan_chunk):
            records = fields_view[start : min(start + scan_chunk, rows)]
            canonical = adapt_records(
                records,
                cfg,
                apply_cut=False,
                apply_redshift_cut=False,
            )
            mass = canonical["M200c_hMsun"]
            redshift = canonical["z"]
            mass_keep = mass > threshold
            keep = _selection_mask(mass, redshift, cfg)
            all_source_rows_pass_mass_cut &= bool(np.all(mass_keep))
            all_source_rows_pass &= bool(np.all(keep))
            selected += int(np.count_nonzero(keep))
            selected_indices = (start + np.flatnonzero(keep)).astype("<i8", copy=False)
            selected_index_hasher.update(selected_indices.tobytes())
            if np.any(keep):
                minimum_mass = min(minimum_mass, float(np.min(mass[keep])))
                maximum_mass = max(maximum_mass, float(np.max(mass[keep])))
                minimum_z = min(minimum_z, float(np.min(redshift[keep])))
                maximum_z = max(maximum_z, float(np.max(redshift[keep])))
            expected_chi = comoving_distance_hmpc(redshift, cfg["cosmology"])
            relative = np.abs(canonical["chi_hMpc"] - expected_chi) / np.maximum(expected_chi, 1.0)
            maximum_distance_error = max(maximum_distance_error, float(np.max(relative, initial=0.0)))

    if selected == 0:
        minimum_mass = maximum_mass = minimum_z = maximum_z = np.nan
    validation = cfg["validation"]
    if bool(validation["require_strict_mass_cut"]) and bool(
        cfg["catalog"]["selection"].get("source_is_prefiltered", False)
    ) and threshold == float(attrs.get("mass_threshold", threshold)) and not all_source_rows_pass_mass_cut:
        raise ValueError("The prefiltered catalog contains rows that fail the strict mass cut.")
    expected_rows = int(cfg["catalog"]["selection"].get("expected_rows", -1))
    if (
        bool(validation["require_expected_rows"])
        and threshold == float(attrs.get("mass_threshold", threshold))
        and expected_rows >= 0
        and selected != expected_rows
    ):
        raise ValueError(f"Expected {expected_rows:,} selected rows; found {selected:,}.")
    max_allowed = float(validation["max_distance_redshift_relative_error"])
    if maximum_distance_error > max_allowed:
        raise ValueError(
            "Observer/cosmology radial-distance check failed: "
            f"max relative error {maximum_distance_error:.6g} > {max_allowed:.6g}."
        )

    halo_grid = cfg["profiles"]["overrides"]["halo_params"]
    if selected:
        if minimum_z < float(halo_grid["zmin"]) or maximum_z > float(halo_grid["zmax"]):
            raise ValueError("Catalog redshift range lies outside the configured profile grid.")
        if np.log10(minimum_mass) < float(halo_grid["lg10_Mmin"]) or np.log10(maximum_mass) > float(
            halo_grid["lg10_Mmax"]
        ):
            raise ValueError("Catalog mass range lies outside the configured profile grid.")
    return {
        "source_path": str(path),
        "dataset": dataset_name,
        "source_rows": rows,
        "selected_rows": selected,
        "all_source_rows_pass_cut": all_source_rows_pass,
        "all_source_rows_pass_mass_cut": all_source_rows_pass_mass_cut,
        "selected_row_index_sha256": selected_index_hasher.hexdigest(),
        "mass_min_hmsun": minimum_mass,
        "mass_max_hmsun": maximum_mass,
        "z_min": minimum_z,
        "z_max": maximum_z,
        "max_distance_redshift_relative_error": maximum_distance_error,
        "source_attrs": attrs,
    }


def stratified_row_indices(cfg: Mapping[str, Any], n_halos: int = 64) -> np.ndarray:
    """Choose deterministic marginal mass/redshift quantiles for validation."""

    requested = int(n_halos)
    if requested <= 0:
        return np.empty(0, dtype=np.int64)
    path = Path(cfg["catalog"]["path"])
    dataset_name = str(cfg["catalog"]["dataset"])
    mass_name, _, _, _, redshift_name = _field_names(cfg)
    with h5py.File(path, "r") as handle:
        rows = handle[dataset_name].fields((mass_name, redshift_name))[:]
    mass = np.asarray(rows[mass_name], dtype=np.float64)
    redshift = np.asarray(rows[redshift_name], dtype=np.float64)
    valid = np.flatnonzero(_selection_mask(mass, redshift, cfg))
    if requested >= len(valid):
        return valid.astype(np.int64)
    half = requested // 2
    z_order = valid[np.argsort(redshift[valid], kind="stable")]
    mass_order = valid[np.argsort(mass[valid], kind="stable")]
    picked = list(z_order[np.linspace(0, len(z_order) - 1, half, dtype=np.int64)])
    picked.extend(
        mass_order[np.linspace(0, len(mass_order) - 1, requested - half, dtype=np.int64)]
    )
    unique = list(dict.fromkeys(int(value) for value in picked))
    if len(unique) < requested:
        for value in valid[np.linspace(0, len(valid) - 1, requested * 2, dtype=np.int64)]:
            if int(value) not in unique:
                unique.append(int(value))
            if len(unique) == requested:
                break
    return np.sort(np.asarray(unique[:requested], dtype=np.int64))


def _iter_selected_chunks(
    cfg: Mapping[str, Any],
    max_halos: int | None,
    row_indices: np.ndarray | None = None,
) -> Iterator[dict]:
    path = Path(cfg["catalog"]["path"])
    dataset_name = str(cfg["catalog"]["dataset"])
    names = _field_names(cfg)
    chunk_size = int(cfg["runtime"]["halo_chunk_size"])
    emitted = 0
    with h5py.File(path, "r") as handle:
        parent = handle[dataset_name]
        row_count = int(parent.shape[0])
        dataset = parent.fields(names)
        source_threshold = handle.attrs.get("mass_threshold")
        requested_threshold = float(cfg["catalog"]["selection"]["mass_min_hmsun"])
        use_prefiltered_rows = bool(
            cfg["catalog"]["selection"].get("source_is_prefiltered", False)
            and source_threshold is not None
            and requested_threshold == float(source_threshold)
        )
        if row_indices is not None:
            indices = np.asarray(row_indices, dtype=np.int64)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= row_count):
                raise ValueError("row_indices must be a one-dimensional in-range array.")
            if len(indices) and np.any(np.diff(indices) <= 0):
                raise ValueError("row_indices must be strictly increasing and unique.")
            for start in range(0, len(indices), chunk_size):
                batch_indices = indices[start : start + chunk_size]
                records = dataset[batch_indices]
                canonical = adapt_records(records, cfg, apply_cut=not use_prefiltered_rows)
                if len(canonical["z"]) != len(batch_indices):
                    raise ValueError("One or more requested validation rows fail the configured selection.")
                selected_indices = batch_indices[np.asarray(canonical["keep_mask"], dtype=bool)]
                chunk = {key: value for key, value in canonical.items() if key != "keep_mask"}
                chunk["source_row_index"] = selected_indices
                yield chunk
            return
        for start in range(0, row_count, chunk_size):
            records = dataset[start : min(start + chunk_size, row_count)]
            canonical = adapt_records(records, cfg, apply_cut=not use_prefiltered_rows)
            available = len(canonical["z"])
            if max_halos is not None:
                available = min(available, max(0, int(max_halos) - emitted))
            if available:
                source_indices = np.arange(start, min(start + chunk_size, row_count), dtype=np.int64)
                source_indices = source_indices[np.asarray(canonical["keep_mask"], dtype=bool)][:available]
                chunk = {
                    key: value[:available]
                    for key, value in canonical.items()
                    if key != "keep_mask"
                }
                chunk["source_row_index"] = source_indices
                yield chunk
                emitted += available
            if max_halos is not None and emitted >= int(max_halos):
                break


def _configure_jax(cfg: Mapping[str, Any]):
    platforms = str(cfg["runtime"].get("jax_platforms", "auto"))
    if platforms.lower() != "auto":
        os.environ.setdefault("JAX_PLATFORMS", platforms)
    os.environ["JAX_ENABLE_X64"] = "True"
    jax_was_imported = "jax" in sys.modules
    import jax

    if jax_was_imported and not bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "JAX was already imported with x64 disabled. Restart the kernel and run "
            "the notebook's first cell before importing tsz_pasting."
        )
    jax.config.update("jax_enable_x64", True)
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("JAX x64 is disabled. Restart the kernel and run the first notebook cell again.")
    if platforms.lower() != "auto":
        allowed = {value.strip() for value in platforms.lower().split(",")}
        if "cuda" in allowed:
            allowed.add("gpu")
        if jax.default_backend().lower() not in allowed:
            raise RuntimeError(
                f"Requested runtime.jax_platforms={platforms!r}, but the initialized "
                f"JAX backend is {jax.default_backend()!r}; restart the kernel."
            )
    return jax


def _profile_parameter_dicts(cfg: Mapping[str, Any]) -> tuple[dict, dict, dict, dict]:
    base = _read_yaml(cfg["profiles"]["default_params"])
    merged = _deep_update(base, cfg["profiles"]["overrides"])
    merged["sim_params"]["cosmo"] = copy.deepcopy(cfg["cosmology"])
    return tuple(copy.deepcopy(merged[name]) for name in ("sim_params", "halo_params", "analysis", "other_params"))  # type: ignore[return-value]


def build_profile_setup(cfg: Mapping[str, Any]):
    """Build the validated asymptotic profile and tSZ projection exactly once."""

    jax = _configure_jax(cfg)
    for path in (
        REPO_ROOT / "src",
        REPO_ROOT / "notebooks" / "xDESI",
        REPO_ROOT / "notebooks" / "xDESI" / "baryonforge_compare",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from base_class import base_class
    from get_radial_profiles import Profiles as NativeProfiles
    from get_sim_maps import setup_sim_map

    class_path = str(cfg["profiles"]["class"])
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError("profiles.class must be a fully qualified module.Class name.")
    profiles_class = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(profiles_class, type) or not issubclass(profiles_class, NativeProfiles):
        raise TypeError(f"{class_path} is not a GODMAX Profiles subclass.")

    sim_params, halo_params, analysis, other_params = _profile_parameter_dicts(cfg)
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = profiles_class(
        sim_params,
        halo_params,
        analysis,
        other_params,
        base_class_obj=base,
    )
    setup_params = {
        "nside": int(cfg["map"]["nside"]),
        "smooth_profiles": False,
        "profile_timing": False,
        "use_fused_profile_maps": True,
        "return_sparse_maps": True,
        "store_projected_matter_maps": False,
        "get_galmap": False,
        "get_ymap": False,
        "get_kSZmap": False,
        "get_kszmap": False,
        "get_taumap": False,
        "get_kappamap": False,
        "get_multi_kappamap": False,
        "get_baryonifiedmap": False,
    }
    setup = setup_sim_map(
        sim_params,
        halo_params,
        analysis,
        other_params,
        setup_params,
        Profiles_obj=profiles,
    )
    pressure = np.asarray(setup.Pe_mat_physical)
    if not np.all(np.isfinite(pressure)) or np.any(pressure < 0.0):
        raise ValueError("GODMAX electron pressure table is nonfinite or negative.")
    native_rp = np.asarray(setup.rp_array, dtype=np.float64)
    setup.native_projected_rp_min_hmpc = float(native_rp[0])
    requested_rp_min = float(cfg["profiles"]["projected_radius_min_hmpc"])
    if requested_rp_min >= float(native_rp[0]):
        raise ValueError(
            "profiles.projected_radius_min_hmpc must be below the native projected-grid minimum "
            f"{float(native_rp[0]):.8g} Mpc/h."
        )
    central_rp = np.geomspace(
        requested_rp_min,
        float(native_rp[0]),
        num=int(cfg["profiles"]["projected_radius_num_central_points"]),
        endpoint=False,
        dtype=np.float64,
    )
    import jax.numpy as jnp

    setup.rp_array = jnp.asarray(np.concatenate((central_rp, native_rp)), dtype=jnp.float32)
    setup._setup_ymap()
    jax.block_until_ready(setup.y2D_mat_physical)
    projected_y = np.asarray(setup.y2D_mat_physical)
    if not np.all(np.isfinite(projected_y)) or np.any(projected_y < 0.0):
        raise ValueError("GODMAX projected-y table is nonfinite or negative.")
    return setup


def make_pair_evaluator(setup, pair_batch_size: int):
    """Create one pure fixed-shape JIT kernel and reuse it for the whole map."""

    import jax
    import jax.numpy as jnp

    interpolator = setup.log_y2D_interp
    rp_min = float(np.asarray(setup.rp_array)[0])
    rp_max = float(np.asarray(setup.rp_array)[-1])

    @jax.jit
    def evaluate(props):
        return jax.vmap(lambda prop: jnp.exp(interpolator(prop[0], prop[1], prop[2])))(props)

    safe_log_radius = float(np.log(np.asarray(setup.rp_array)[0]))
    safe_z = float(np.asarray(setup.z_array)[0])
    safe_log_mass = float(np.log(np.asarray(setup.M_array)[0]))
    warm = np.empty((int(pair_batch_size), 3), dtype=np.float32)
    warm[:] = (safe_log_radius, safe_z, safe_log_mass)
    jax.block_until_ready(evaluate(jnp.asarray(warm)))
    evaluate.godmax_rp_min = rp_min
    evaluate.godmax_rp_max = rp_max
    evaluate.godmax_native_rp_min = float(
        getattr(setup, "native_projected_rp_min_hmpc", rp_min)
    )
    return evaluate


def evaluate_pairs_fixed(
    evaluator,
    pixel_work: Mapping[str, np.ndarray],
    pair_batch_size: int,
) -> np.ndarray:
    """Evaluate pairs without compiling a new final-batch shape."""

    import jax
    import jax.numpy as jnp

    n_pairs = len(pixel_work["distances"])
    output = np.empty(n_pairs, dtype=np.float32)
    if n_pairs == 0:
        return output
    rp_max = float(getattr(evaluator, "godmax_rp_max", np.inf))
    rp_min = float(getattr(evaluator, "godmax_rp_min", 0.0))
    minimum_distance = float(np.min(pixel_work["distances"], initial=np.inf))
    maximum_distance = float(np.max(pixel_work["distances"], initial=0.0))
    if minimum_distance < rp_min * (1.0 - 1.0e-6):
        raise ValueError(
            "Pixel transverse radius lies below the extended projected profile grid: "
            f"{minimum_distance:.8g} < {rp_min:.8g} Mpc/h."
        )
    if maximum_distance > rp_max * (1.0 + 1.0e-6):
        raise ValueError(
            "Pixel transverse radius lies above the projected profile grid: "
            f"{maximum_distance:.8g} > {rp_max:.8g} Mpc/h."
        )
    batch_size = int(pair_batch_size)
    padded = np.empty((batch_size, 3), dtype=np.float32)
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        count = end - start
        padded[:count, 0] = np.log(np.maximum(pixel_work["distances"][start:end], 1.0e-7))
        padded[:count, 1] = pixel_work["z"][start:end]
        padded[:count, 2] = pixel_work["logM"][start:end]
        if count < batch_size:
            padded[count:] = padded[count - 1]
        values = evaluator(jnp.asarray(padded))
        jax.block_until_ready(values)
        output[start:end] = np.asarray(values[:count], dtype=np.float32)
    if not np.all(np.isfinite(output)) or np.any(output < 0.0):
        raise ValueError("Pair evaluator produced nonfinite or negative Compton-y.")
    return output


def _pixel_helper():
    path = REPO_ROOT / "notebooks" / "xDESI"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from abacus_pasting_helpers import build_pixel_work_package

    return build_pixel_work_package


def _accumulate_pixel_work(output_map: np.ndarray, pixel_work: Mapping[str, np.ndarray], values: np.ndarray) -> None:
    sort_idx = np.asarray(pixel_work["sort_idx"], dtype=np.int64)
    boundaries = np.asarray(pixel_work["boundaries"], dtype=np.int64)
    pix_unique = np.asarray(pixel_work["pix_unique"], dtype=np.int64)
    sums = np.add.reduceat(values[sort_idx].astype(np.float64), boundaries[:-1])
    output_map[pix_unique] += sums


def validate_pair_kernel_against_reference(
    params_path: str | Path = Path(__file__).with_name("params_tsz.yaml"),
    *,
    overrides: Mapping[str, Any] | None = None,
    max_halos: int = 8,
    alternate_pair_batch_size: int | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-12,
) -> dict:
    """Compare the thin evaluator with GODMAX's established ``get_sim_map``.

    This is a bounded validation utility, not part of production painting.  It
    constructs one shared profile setup and evaluates the identical pixel-pair
    package through both assembly paths.
    """

    if int(max_halos) <= 0:
        raise ValueError("max_halos must be positive for evaluator validation.")
    cfg = load_params(params_path, overrides)
    preflight_catalog(cfg)
    setup = build_profile_setup(cfg)
    chunk = next(_iter_selected_chunks(cfg, int(max_halos)))
    build_pixels = _pixel_helper()
    pixel_work = build_pixels(
        chunk,
        nside=int(cfg["map"]["nside"]),
        max_paint=float(cfg["map"]["max_paint_R200c_factor"]),
        batch_size=int(cfg["runtime"]["pixel_batch_size"]),
        workers=int(cfg["runtime"]["pixel_workers"]),
        single_pixel_angle_factor=float(cfg["runtime"]["single_pixel_angle_factor"]),
        verbose=bool(cfg["runtime"]["verbose"]),
        log_batches=False,
        pixel_backend=str(cfg["runtime"]["pixel_backend"]),
        include_legacy_pixel_arrays=False,
        precompute_pixel_groups=True,
    )
    if pixel_work is None:
        raise RuntimeError("Reference validation produced no pixel pairs.")
    evaluator = make_pair_evaluator(setup, int(cfg["runtime"]["pair_batch_size"]))
    thin_pairs = evaluate_pairs_fixed(evaluator, pixel_work, int(cfg["runtime"]["pair_batch_size"]))
    rp_min = float(evaluator.godmax_rp_min)
    n_below_projected_grid = int(np.count_nonzero(pixel_work["distances"] < rp_min * (1.0 - 1.0e-6)))
    alternate_size = (
        max(1, int(cfg["runtime"]["pair_batch_size"]) // 2 + 1)
        if alternate_pair_batch_size is None
        else int(alternate_pair_batch_size)
    )
    if alternate_size <= 0:
        raise ValueError("alternate_pair_batch_size must be positive.")
    alternate_pairs = evaluate_pairs_fixed(evaluator, pixel_work, alternate_size)
    batch_size_invariant = bool(np.array_equal(thin_pairs, alternate_pairs))
    thin_map = np.zeros(12 * int(cfg["map"]["nside"]) ** 2, dtype=np.float64)
    _accumulate_pixel_work(thin_map, pixel_work, thin_pairs)

    import jax
    import jax.numpy as jnp
    from get_sim_maps import get_sim_map

    sim_params, halo_params, analysis, other_params = _profile_parameter_dicts(cfg)
    props = np.column_stack(
        (
            np.log(np.maximum(pixel_work["distances"], 1.0e-7)),
            pixel_work["z"],
            pixel_work["logM"],
            pixel_work["vlos"],
        )
    ).astype(np.float32)
    mock_params = {
        "nside": int(cfg["map"]["nside"]),
        "smooth_profiles": False,
        "profile_timing": False,
        "use_fused_profile_maps": True,
        "return_sparse_maps": True,
        "store_projected_matter_maps": False,
        "get_galmap": False,
        "get_ymap": True,
        "get_kSZmap": False,
        "get_taumap": False,
        "get_kappamap": False,
        "get_multi_kappamap": False,
        "get_baryonifiedmap": False,
        "nearby_pix_all": pixel_work["nearby_pix_all"],
        "pix_unique": pixel_work["pix_unique"],
        "sort_idx": pixel_work["sort_idx"],
        "boundaries": pixel_work["boundaries"],
        "pix_prop_all": jnp.asarray(props),
    }
    reference = get_sim_map(
        sim_params,
        halo_params,
        analysis,
        other_params,
        mock_params,
        Profiles_obj=setup,
    )
    reference_pixels, reference_values = reference.ymap_final
    jax.block_until_ready(reference_values)
    thin_values = thin_map[np.asarray(reference_pixels, dtype=np.int64)].astype(np.float32)
    reference_values = np.asarray(reference_values, dtype=np.float32)
    absolute = np.abs(thin_values - reference_values)
    relative = absolute / np.maximum(np.abs(reference_values), float(atol))
    same_footprint = np.array_equal(
        np.flatnonzero(thin_map), np.asarray(reference_pixels, dtype=np.int64)
    )
    passed = bool(
        same_footprint
        and batch_size_invariant
        and n_below_projected_grid == 0
        and np.allclose(thin_values, reference_values, rtol=float(rtol), atol=float(atol))
    )
    return {
        "passed": passed,
        "n_halos": int(len(chunk["z"])),
        "n_pairs": int(len(thin_pairs)),
        "n_active_pixels": int(len(reference_pixels)),
        "same_nonzero_footprint": same_footprint,
        "pair_batch_size": int(cfg["runtime"]["pair_batch_size"]),
        "alternate_pair_batch_size": alternate_size,
        "pair_batch_size_bitwise_invariant": batch_size_invariant,
        "projected_radius_min_hmpc": rp_min,
        "n_pairs_below_projected_grid": n_below_projected_grid,
        "max_absolute_difference": float(np.max(absolute, initial=0.0)),
        "max_relative_difference": float(np.max(relative, initial=0.0)),
        "rtol": float(rtol),
        "atol": float(atol),
    }


def _sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _configuration_hash(cfg: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _default_output_path(
    cfg: Mapping[str, Any],
    config_hash: str,
    max_halos: int | None,
    row_indices: np.ndarray | None,
) -> Path:
    if row_indices is not None:
        sample_hash = hashlib.sha256(np.asarray(row_indices, dtype=np.int64).tobytes()).hexdigest()[:8]
        suffix = f"sample{len(row_indices)}-{sample_hash}"
    else:
        suffix = "all" if max_halos is None else str(int(max_halos))
    filename = (
        f"{cfg['output']['run_name']}_nside{int(cfg['map']['nside'])}_"
        f"halos{suffix}_{config_hash[:12]}.h5"
    )
    return Path(cfg["output"]["directory"]) / filename


def _write_output(
    path: Path,
    output_map: np.ndarray,
    cfg: Mapping[str, Any],
    preflight: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    overwrite: bool,
) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass overwrite=True explicitly.")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Refusing to replace pre-existing staging file {staging}.")
    commit, dirty = _git_state()
    compression = cfg["output"].get("compression")
    compression_opts = int(cfg["output"].get("compression_level", 1)) if compression == "gzip" else None
    config_hash = _configuration_hash(cfg)
    selection = cfg["catalog"]["selection"]
    selection_predicate = (
        f"{cfg['catalog']['fields']['mass']} > "
        f"{float(selection['mass_min_hmsun']):.17g} Msun/h"
    )
    redshift_max = selection.get("redshift_max")
    if redshift_max is not None:
        selection_predicate += (
            f" and {cfg['catalog']['fields']['redshift']} <= {float(redshift_max):.12g}"
        )
    try:
        with h5py.File(staging, "w") as handle:
            maps = handle.create_group("maps")
            maps.create_dataset(
                "map_ymap",
                data=np.asarray(output_map, dtype=np.float32),
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
            )
            attrs = handle.attrs
            attrs["schema"] = "godmax_des_cluster_tsz_map_v1"
            attrs["map_dataset"] = MAP_DATASET
            attrs["map_units"] = "dimensionless Compton-y"
            attrs["ordering"] = "RING"
            attrs["nside"] = int(cfg["map"]["nside"])
            attrs["npix"] = int(len(output_map))
            attrs["product"] = "halo-only one-halo Compton-y"
            attrs["contains_diffuse_or_unbound_gas"] = False
            attrs["contains_analytic_two_halo"] = False
            attrs["beam"] = "none"
            attrs["noise"] = "none"
            attrs["pressure_amplitude"] = float(cfg["map"]["pressure_amplitude"])
            attrs["max_paint_R200c_factor"] = float(cfg["map"]["max_paint_R200c_factor"])
            attrs["central_radius_policy"] = str(cfg["map"]["central_radius_policy"])
            attrs["mass_definition"] = str(cfg["catalog"]["mass_definition"])
            attrs["mass_definition_is_provisional"] = True
            attrs["mass_assumption"] = "M_interp conditionally treated as M200c in Msun/h"
            attrs["position_unit"] = "comoving Mpc/h"
            attrs["observer_xyz_hmpc"] = np.asarray(cfg["catalog"]["observer_xyz_hmpc"], dtype=np.float64)
            attrs["angular_coordinate_formula"] = "RA=atan2(dY,dX), Dec=asin(dZ/|d|), d=XYZ-observer"
            attrs["distance_formula"] = "DA_hMpc=|XYZ-observer|/(1+z)"
            attrs["radius_formula"] = "R200c=[3M/(4pi*200*rho_crit0*E(z)^2)]^(1/3), proper Mpc/h"
            attrs["velocity_policy"] = "source missing; vlos=0 placeholder unused by tSZ"
            attrs["selection_predicate"] = selection_predicate
            attrs["selection_is_strict"] = True
            attrs["mass_selection_is_strict"] = True
            attrs["redshift_max"] = np.nan if redshift_max is None else float(redshift_max)
            attrs["redshift_max_is_inclusive"] = bool(redshift_max is not None)
            attrs["selection_redshift_max"] = (
                np.nan if redshift_max is None else float(redshift_max)
            )
            attrs["selection_redshift_max_inclusive"] = bool(redshift_max is not None)
            attrs["source_catalog"] = str(cfg["catalog"]["path"])
            attrs["source_dataset"] = str(cfg["catalog"]["dataset"])
            attrs["source_rows"] = int(preflight["source_rows"])
            attrs["selected_rows_available"] = int(preflight["selected_rows"])
            attrs["selected_z_min"] = float(preflight["z_min"])
            attrs["selected_z_max"] = float(preflight["z_max"])
            attrs["selected_row_index_sha256"] = str(
                preflight["selected_row_index_sha256"]
            )
            attrs["painted_row_index_sha256"] = str(
                diagnostics["painted_row_index_sha256"]
            )
            attrs["n_halos_painted"] = int(diagnostics["n_halos_painted"])
            attrs["n_pairs_below_projected_grid"] = int(
                diagnostics["n_pairs_below_projected_grid"]
            )
            if diagnostics.get("projected_radius_min_hmpc") is not None:
                attrs["projected_radius_min_hmpc"] = float(
                    diagnostics["projected_radius_min_hmpc"]
                )
                attrs["projected_radius_max_hmpc"] = float(
                    diagnostics["projected_radius_max_hmpc"]
                )
            attrs["complete_selected_catalog_painted"] = bool(
                int(diagnostics["n_halos_painted"]) == int(preflight["selected_rows"])
                and str(diagnostics["painted_row_index_sha256"])
                == str(preflight["selected_row_index_sha256"])
            )
            for key, value in cfg["cosmology"].items():
                attrs[f"cosmology_{key}"] = value
            attrs["profile_class"] = str(cfg["profiles"]["class"])
            attrs["normalization_variant"] = str(cfg["profiles"]["normalization_variant"])
            attrs["projection_variant"] = str(
                cfg["profiles"]["overrides"]["analysis"]["projected_profile_integration_method"]
            )
            attrs["config_path"] = str(cfg["_config_path"])
            attrs["config_sha256"] = config_hash
            attrs["config_file_sha256"] = _sha256_file(cfg["_config_path"])
            attrs["config_sources_json"] = json.dumps(
                [
                    {"path": source, "sha256": _sha256_file(source)}
                    for source in cfg.get("_config_sources", [cfg["_config_path"]])
                ],
                sort_keys=True,
            )
            attrs["catalog_sha256"] = _sha256_file(cfg["catalog"]["path"])
            attrs["helper_sha256"] = _sha256_file(MODULE_PATH)
            attrs["pixel_helper_sha256"] = _sha256_file(
                REPO_ROOT / "notebooks" / "xDESI" / "abacus_pasting_helpers.py"
            )
            attrs["godmax_map_source_sha256"] = _sha256_file(REPO_ROOT / "src" / "get_sim_maps.py")
            attrs["profiles_source_sha256"] = _sha256_file(
                REPO_ROOT
                / "notebooks"
                / "xDESI"
                / "baryonforge_compare"
                / "matched_godmax_profiles.py"
            )
            attrs["default_params_sha256"] = _sha256_file(cfg["profiles"]["default_params"])
            attrs["profile_grid_json"] = json.dumps(
                cfg["profiles"]["overrides"]["halo_params"], sort_keys=True
            )
            attrs["runtime_json"] = json.dumps(cfg["runtime"], sort_keys=True)
            attrs["git_commit"] = commit
            attrs["git_dirty"] = dirty
            attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            attrs["config_yaml"] = yaml.safe_dump(
                {key: value for key, value in cfg.items() if not str(key).startswith("_")}, sort_keys=True
            )
            attrs["diagnostics_json"] = json.dumps(dict(diagnostics), sort_keys=True)
            handle.flush()
        if overwrite:
            os.replace(staging, path)
        else:
            # ``os.replace`` would silently clobber a product created after
            # the initial exists check.  A same-filesystem hard link is an
            # atomic no-replace publication: exactly one concurrent writer
            # can claim ``path``.
            try:
                os.link(staging, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite concurrently-created {path}."
                ) from exc
            staging.unlink()
    except Exception:
        if staging.exists():
            staging.unlink()
        raise
    return path


def run_tsz_paste(
    params_path: str | Path = Path(__file__).with_name("params_tsz.yaml"),
    *,
    overrides: Mapping[str, Any] | None = None,
    max_halos: int | None = None,
    row_indices: np.ndarray | None = None,
    output_path: str | Path | None = None,
    overwrite: bool | None = None,
    dry_run: bool = False,
) -> dict:
    """Preflight, paint, and atomically write a single Compton-y HEALPix map.

    ``max_halos`` is for bounded smoke/convergence runs.  Leave it as ``None``
    to paint every halo passing the configured strict mass cut.
    """

    cfg = load_params(params_path, overrides)
    if max_halos is not None and int(max_halos) < 0:
        raise ValueError("max_halos must be nonnegative or None.")
    if row_indices is not None and max_halos is not None:
        raise ValueError("Use either max_halos or row_indices, not both.")
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
    preflight = preflight_catalog(cfg)
    if row_indices is not None:
        target_halos = int(len(row_indices))
    else:
        target_halos = int(preflight["selected_rows"]) if max_halos is None else min(
            int(max_halos), int(preflight["selected_rows"])
        )
    if dry_run:
        return {"config": cfg, "preflight": preflight, "target_halos": target_halos}

    nside = int(cfg["map"]["nside"])
    output_map = np.zeros(12 * nside * nside, dtype=np.float64)
    pressure_amplitude = float(cfg["map"]["pressure_amplitude"])
    setup = evaluator = None
    projected_radius_min_hmpc = None
    projected_radius_max_hmpc = None
    if target_halos > 0 and pressure_amplitude > 0.0:
        setup = build_profile_setup(cfg)
        evaluator = make_pair_evaluator(setup, int(cfg["runtime"]["pair_batch_size"]))
        projected_radius_min_hmpc = float(evaluator.godmax_rp_min)
        projected_radius_max_hmpc = float(evaluator.godmax_rp_max)
    build_pixels = _pixel_helper() if evaluator is not None else None

    n_halos_painted = 0
    n_pairs = 0
    n_pairs_below_projected_grid = 0
    chunks = 0
    painted_index_hasher = hashlib.sha256()
    started = time.perf_counter()
    if evaluator is not None and build_pixels is not None:
        for chunk in _iter_selected_chunks(cfg, target_halos, row_indices=row_indices):
            painted_index_hasher.update(
                np.asarray(chunk["source_row_index"], dtype="<i8").tobytes()
            )
            pixel_work = build_pixels(
                chunk,
                nside=nside,
                max_paint=float(cfg["map"]["max_paint_R200c_factor"]),
                batch_size=int(cfg["runtime"]["pixel_batch_size"]),
                workers=int(cfg["runtime"]["pixel_workers"]),
                single_pixel_angle_factor=float(cfg["runtime"]["single_pixel_angle_factor"]),
                verbose=bool(cfg["runtime"]["verbose"]),
                log_batches=False,
                pixel_backend=str(cfg["runtime"]["pixel_backend"]),
                include_legacy_pixel_arrays=False,
                precompute_pixel_groups=True,
            )
            if pixel_work is not None:
                n_pairs_below_projected_grid += int(
                    np.count_nonzero(
                        pixel_work["distances"]
                        < float(evaluator.godmax_rp_min) * (1.0 - 1.0e-6)
                    )
                )
                values = evaluate_pairs_fixed(
                    evaluator, pixel_work, int(cfg["runtime"]["pair_batch_size"])
                )
                _accumulate_pixel_work(output_map, pixel_work, values)
                n_pairs += len(values)
            n_halos_painted += len(chunk["z"])
            chunks += 1
            if bool(cfg["runtime"]["verbose"]):
                print(
                    f"[tsz] painted {n_halos_painted:,}/{target_halos:,} halos; "
                    f"pairs={n_pairs:,}; elapsed={time.perf_counter() - started:.1f}s"
                )
    elif pressure_amplitude == 0.0 and target_halos > 0:
        # Preserve the exact null while still validating and counting the
        # selected input rows (especially explicit validation indices).
        for chunk in _iter_selected_chunks(cfg, target_halos, row_indices=row_indices):
            painted_index_hasher.update(
                np.asarray(chunk["source_row_index"], dtype="<i8").tobytes()
            )
            n_halos_painted += len(chunk["z"])
            chunks += 1

    # Applying this once after native y evaluation makes A=0 an exact null and
    # avoids sending zero pressure through the logarithmic interpolator.
    output_map *= pressure_amplitude
    if not np.all(np.isfinite(output_map)) or np.any(output_map < 0.0):
        raise ValueError("Final Compton-y map is nonfinite or negative.")
    if n_halos_painted != target_halos:
        raise RuntimeError(f"Expected to paint {target_halos:,} halos; painted {n_halos_painted:,}.")
    painted_row_index_sha256 = painted_index_hasher.hexdigest()
    if max_halos is None and row_indices is None:
        expected_index_sha256 = str(preflight["selected_row_index_sha256"])
        if painted_row_index_sha256 != expected_index_sha256:
            raise RuntimeError(
                "Painted-row identity does not match the preflight selection: "
                f"{painted_row_index_sha256} != {expected_index_sha256}."
            )
    diagnostics = {
        "n_halos_painted": int(n_halos_painted),
        "n_pairs": int(n_pairs),
        "n_pairs_below_projected_grid": int(n_pairs_below_projected_grid),
        "projected_radius_min_hmpc": projected_radius_min_hmpc,
        "projected_radius_max_hmpc": projected_radius_max_hmpc,
        "n_chunks": int(chunks),
        "elapsed_s": float(time.perf_counter() - started),
        "map_min": float(np.min(output_map, initial=0.0)),
        "map_max": float(np.max(output_map, initial=0.0)),
        "map_sum": float(np.sum(output_map, dtype=np.float64)),
        "painted_row_index_sha256": painted_row_index_sha256,
        "catalog_sampling": "explicit_row_indices" if row_indices is not None else (
            "complete_selection" if max_halos is None else "first_selected_rows"
        ),
    }
    if "jax" in sys.modules:
        import jax

        diagnostics.update(
            {
                "jax_backend": str(jax.default_backend()),
                "jax_devices": [str(device) for device in jax.devices()],
                "jax_x64": bool(jax.config.jax_enable_x64),
            }
        )
    else:
        diagnostics.update(
            {"jax_backend": "not_initialized_exact_null", "jax_devices": [], "jax_x64": None}
        )
    config_hash = _configuration_hash(cfg)
    path = Path(output_path) if output_path is not None else _default_output_path(
        cfg, config_hash, max_halos, row_indices
    )
    actual_overwrite = bool(cfg["output"]["overwrite"]) if overwrite is None else bool(overwrite)
    written = _write_output(path, output_map, cfg, preflight, diagnostics, overwrite=actual_overwrite)
    return {"path": written, "config": cfg, "preflight": preflight, "diagnostics": diagnostics}


def load_tsz_map(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a saved map and its provenance without importing JAX or GODMAX."""

    with h5py.File(path, "r") as handle:
        if MAP_DATASET not in handle:
            raise KeyError(f"{path} does not contain {MAP_DATASET!r}.")
        ymap = handle[MAP_DATASET][:]
        attrs = _catalog_attrs(handle)
    return ymap, attrs


__all__ = [
    "MAP_DATASET",
    "adapt_records",
    "build_profile_setup",
    "comoving_distance_hmpc",
    "evaluate_pairs_fixed",
    "load_params",
    "load_tsz_map",
    "make_pair_evaluator",
    "preflight_catalog",
    "run_tsz_paste",
    "stratified_row_indices",
    "validate_pair_kernel_against_reference",
    "validate_params",
]
