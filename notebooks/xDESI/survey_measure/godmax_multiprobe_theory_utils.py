"""Helpers for GODMAX theory comparisons to xDESI multi-probe measurements."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml


SCHEMA_MEASUREMENT = "xdesi_multiprobe_measurement_v1"
DESI_TRUE_NZ_REDSHIFT_KIND = "spectroscopic_calibrated_true_redshift"
HALO_MASS_FLOOR_TOL = 5.0e-7
HOD_ARRAY_BASE_NAMES = (
    "log10M1_fshmr",
    "log10M1_a_fshmr",
    "log10Mstar0_fshmr",
    "log10Mstar0_a_fshmr",
    "beta_fshmr",
    "beta_a_fshmr",
    "delta_fshmr",
    "delta_a_fshmr",
    "gamma_fshmr",
    "gamma_a_fshmr",
    "siglogMstar_Ncen",
    "alphasat_Nsat",
    "Bcut_Nsat",
    "Bsat_Nsat",
    "betacut_Nsat",
    "betasat_Nsat",
    "fcen",
)

MEASUREMENT_FAMILY_ORDER: Tuple[str, ...] = (
    "des_shear_EE",
    "act_y_des_shear_E",
    "desi_g_auto",
    "desi_g_act_y",
    "desi_g_des_shear_E",
    "desi_g_act_kappa",
    "desi_pi_act_T",
)


@dataclass(frozen=True)
class MeasurementData:
    path: Path
    names: List[str]
    ell: np.ndarray
    data_vector: np.ndarray
    covariance: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    families: Dict[str, str]
    labels: Dict[str, str]
    theory_keys: Dict[str, str]
    ell_left: Optional[np.ndarray] = None
    ell_right: Optional[np.ndarray] = None
    transfer_null_from: Dict[str, float] = field(default_factory=dict)
    data_vector_valid: Optional[np.ndarray] = None
    archive_indices: Optional[np.ndarray] = None
    archive_data_vector_size: Optional[int] = None
    galaxy_auto_view: str = "total"


def measurement_data_identity_sha256(
    *,
    names: Sequence[str],
    ell: object,
    data_vector: object,
    covariance: object,
    starts: object,
    stops: object,
    data_vector_valid: Optional[object] = None,
    archive_indices: Optional[object] = None,
) -> str:
    """Content fingerprint for a cached vector's exact measurement basis."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(names), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    digest.update(b"\0")
    for label, value in (
        ("ell", ell),
        ("data_vector", data_vector),
        ("covariance", covariance),
        ("starts", starts),
        ("stops", stops),
        ("data_vector_valid", np.ones(np.asarray(data_vector).shape, dtype=bool) if data_vector_valid is None else data_vector_valid),
        ("archive_indices", np.arange(np.asarray(data_vector).size, dtype=np.int64) if archive_indices is None else archive_indices),
    ):
        array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {"label": label, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(header + b"\0" + array.tobytes(order="C") + b"\0")
    return digest.hexdigest()


def measurement_identity_sha256(measurement: MeasurementData) -> str:
    return measurement_data_identity_sha256(
        names=measurement.names,
        ell=measurement.ell,
        data_vector=measurement.data_vector,
        covariance=measurement.covariance,
        starts=measurement.starts,
        stops=measurement.stops,
        data_vector_valid=measurement.data_vector_valid,
        archive_indices=measurement.archive_indices,
    )


def theory_vector_cache_fields(
    theory_vector: object,
    measurement_identity_sha256: str,
    generation_metadata: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    """Return self-verifying NPZ fields for a cached theory vector."""

    generation = dict(generation_metadata)
    generation["measurement_identity_sha256"] = str(measurement_identity_sha256)
    generation_json = json.dumps(
        generation,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    array = np.ascontiguousarray(np.asarray(theory_vector))
    digest = hashlib.sha256()
    digest.update(str(measurement_identity_sha256).encode("utf-8") + b"\0")
    digest.update(generation_json.encode("utf-8") + b"\0")
    digest.update(
        json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\0"
    )
    digest.update(array.tobytes(order="C"))
    return {
        "theory_vector_generation_json": np.asarray(generation_json),
        "theory_vector_identity_sha256": np.asarray(digest.hexdigest()),
    }


def comparison_config_identity_sha256(config: Mapping[str, object]) -> str:
    """Fingerprint the materialized inputs that define a comparison theory.

    Output locations and accumulated warning text are deliberately excluded: neither
    changes the computed theory.  The fingerprint does include the merged parameter
    dictionaries, materialized n(z)/number-density metadata, raw wrapper options, and
    the exact source-product paths.
    """

    paths = config.get("paths", {})
    raw = dict(config.get("raw", {}))
    raw.pop("output_dir", None)
    payload = {
        "raw": raw,
        "params": config.get("params", {}),
        "metadata": config.get("metadata", {}),
        "source_paths": {
            str(key): value
            for key, value in paths.items()
            if str(key) != "output_dir"
        },
    }
    canonical = json.dumps(
        _serializable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def theory_response_identity_sha256(config: Mapping[str, object]) -> str:
    """Fingerprint the effective saved response used to window a theory vector."""

    ensure_godmax_import_paths(Path(config["repo_root"]))
    from multiprobe_namaster import (
        DESI_GALAXY_AUTO_MEAN_CONVENTION,
        _load_default_transfers,
        validate_measurement_product_identity,
    )

    digest = hashlib.sha256()

    def update_json(label: str, value: object) -> None:
        payload = json.dumps(
            _serializable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(label.encode("utf-8") + b"\0" + payload + b"\0")

    def update_array(label: str, value: object) -> None:
        array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(label.encode("utf-8") + b"\0" + header + b"\0")
        digest.update(array.tobytes(order="C") + b"\0")

    raw_wrapper = config.get("raw", {}).get("theory_to_data_vector", {})
    include_pixel_windows = bool(raw_wrapper.get("include_default_pixel_windows", True))
    include_act_beams = bool(raw_wrapper.get("include_default_act_beams", True))
    measurement_path = Path(config["paths"]["measurement_h5"])
    with h5py.File(measurement_path, "r") as h5:
        validate_measurement_product_identity(
            h5,
            allow_legacy_product=allow_legacy_product_from_config(config),
        )
        measurement_config = json.loads(str(h5.attrs["config_json"]))
        lmax = int(measurement_config["lmax"])
        field_metadata = _json_load_attr(h5["fields"].attrs, "metadata_json", {})
        transfers = _load_default_transfers(
            h5,
            lmax,
            include_pixel_windows=include_pixel_windows,
            include_act_beams=include_act_beams,
        )
        update_json("contract", "xdesi_theory_response_v1")
        update_json(
            "wrapper_options",
            {
                "include_default_pixel_windows": include_pixel_windows,
                "include_default_act_beams": include_act_beams,
                "theory_shear_e_is_positive_kappa": bool(
                    raw_wrapper.get("theory_shear_e_is_positive_kappa", True)
                ),
                "ksz_velocity_correlation": float(
                    raw_wrapper.get("ksz_velocity_correlation", 0.3)
                ),
                "shear_m_bias_means": config.get("metadata", {}).get(
                    "shear_m_bias_means", {}
                ),
            },
        )
        update_json(
            "measurement_config",
            {"nside": int(measurement_config["nside"]), "lmax": lmax},
        )
        update_json("field_metadata", field_metadata)
        for field_name in sorted(transfers):
            update_array(f"transfer[{field_name}]", transfers[field_name])

        spectrum_names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in h5["joint/spectrum_names"][:]
        ]
        update_json("spectrum_names", spectrum_names)
        if "joint/data_vector_valid" in h5:
            update_array("data_vector_valid", h5["joint/data_vector_valid"][:])
        for index, name in enumerate(spectrum_names):
            group = h5[f"spectra/{name}"]
            fields = json.loads(str(group.attrs["fields"]))
            family = str(group.attrs["family"])
            component = int(group.attrs.get("component", 0))
            update_json(
                f"spectrum[{index}].metadata",
                {
                    "name": name,
                    "fields": fields,
                    "family": family,
                    "theory_key": str(group.attrs["theory_key"]),
                    "metadata": json.loads(str(group.attrs["metadata_json"])),
                    "component": component,
                    "cl_convention": str(group.attrs.get("cl_convention", "")),
                },
            )
            update_array(
                f"spectrum[{index}].bandpower_window_selected",
                group["bandpower_window_selected"][:],
            )
            if family == "desi_g_auto":
                convention = str(group.attrs.get("cl_convention", ""))
                if convention == DESI_GALAXY_AUTO_MEAN_CONVENTION:
                    if "noise_decoupled_all_components" not in group:
                        raise ValueError(
                            f"DESI galaxy auto {name!r} has no saved shot-noise template."
                        )
                    noise = np.asarray(group["noise_decoupled_all_components"][:])
                    if not 0 <= component < noise.shape[0]:
                        raise ValueError(
                            f"DESI galaxy auto {name!r} component {component} is outside "
                            f"saved noise shape {noise.shape}."
                        )
                    update_array(f"spectrum[{index}].shot_noise_template", noise[component])
    return digest.hexdigest()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: Mapping[str, object], override: Mapping[str, object]) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_repo_path(path: str | Path, root: Optional[Path] = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (root or repo_root()) / path


def _json_load_attr(attrs: h5py.AttributeManager, key: str, default):
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def allow_legacy_product_from_config(config: Mapping[str, object]) -> bool:
    """Return the explicit historical-product opt-in from a comparison config."""

    raw = config.get("raw", {})
    wrapper = raw.get("theory_to_data_vector", {}) if isinstance(raw, Mapping) else {}
    return bool(wrapper.get("allow_legacy_product", False)) if isinstance(wrapper, Mapping) else False


def validate_measurement_map_identity(
    measurement_h5: h5py.File,
    map_h5: h5py.File,
    *,
    allow_legacy_product: bool = False,
) -> str:
    """Require a measurement and n(z) map file from the same pipeline-v2 map product."""

    from multiprobe_namaster import (
        MEASUREMENT_PIPELINE_VERSION,
        validate_map_product_hdf_identity,
        validate_measurement_product_identity,
    )

    measurement_id = validate_measurement_product_identity(
        measurement_h5,
        allow_legacy_product=allow_legacy_product,
    )
    map_id = validate_map_product_hdf_identity(
        map_h5,
        allow_legacy_product=allow_legacy_product,
    )
    if measurement_id and map_id and measurement_id != map_id:
        raise ValueError(
            "Measurement and map_h5 have different map_product_id values; their masks, "
            "n(z), windows or estimator settings may not describe the same data vector."
        )
    both_current = (
        str(measurement_h5.attrs.get("pipeline_version", "")) == MEASUREMENT_PIPELINE_VERSION
        and str(map_h5.attrs.get("pipeline_version", "")) == MEASUREMENT_PIPELINE_VERSION
    )
    if both_current and (not measurement_id or not map_id):
        raise ValueError("Pipeline-v2 measurement/map products require non-empty map_product_id values.")
    return measurement_id or map_id


def _validate_theory_nz_content(
    measurement_h5: h5py.File,
    map_h5: h5py.File,
) -> None:
    """Bind the HDF5 n(z) arrays consumed by theory to the embedded map metadata."""

    raw = measurement_h5.attrs.get("map_metadata_json")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    metadata = json.loads(str(raw))

    def check_dataset(h5: h5py.File, dataset: str, expected: object, label: str) -> None:
        if dataset not in h5:
            raise ValueError(f"Pipeline-v2 product is missing {label} dataset {dataset!r}.")
        actual = np.asarray(h5[dataset][:])
        expected_array = np.asarray(expected)
        if actual.shape != expected_array.shape or not np.array_equal(
            actual,
            expected_array,
            equal_nan=True,
        ):
            raise ValueError(
                f"{label} dataset {dataset!r} does not match the content-addressed map metadata."
            )

    des_source = metadata.get("des_y3_source_nz")
    if isinstance(des_source, Mapping):
        for key in ("z_mid", "dndz_by_bin"):
            if key in des_source:
                check_dataset(measurement_h5, f"nz/des_shear/{key}", des_source[key], "DES source n(z)")
    desi = metadata.get("desi_summary")
    if isinstance(desi, Mapping):
        for key in ("z_mid", "z_edges", "nz_dndz_by_pz"):
            if key in desi:
                check_dataset(map_h5, f"nz/desi/{key}", desi[key], "DESI lens n(z)")


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def _normalize_rows(
    z: np.ndarray,
    values: np.ndarray,
    label: str,
    warnings: List[str],
    rtol: float = 5.0e-4,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {arr.shape}.")
    norms = _trapz(arr, z, axis=1)
    for i, norm in enumerate(norms, start=1):
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"{label} bin {i} has invalid normalization {norm}.")
        if not np.isclose(norm, 1.0, rtol=rtol, atol=rtol):
            warnings.append(f"{label} bin {i} renormalized from integral {norm:.8g}.")
            arr[i - 1] /= norm
    return arr


def _support_edges(z_mid: np.ndarray, z_edges: Optional[np.ndarray], dndz: np.ndarray) -> np.ndarray:
    edges = []
    for row in np.asarray(dndz):
        support = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if support.size == 0:
            raise ValueError("DESI lens n(z) has an empty pz-bin support.")
        first = int(support[0])
        last = int(support[-1])
        if z_edges is not None and len(z_edges) == len(z_mid) + 1:
            edges.append([float(z_edges[first]), float(z_edges[last + 1])])
        else:
            edges.append([float(z_mid[first]), float(z_mid[last])])
    return np.asarray(edges, dtype=np.float64)


def _priors_from_measurement(measurement_path: Path) -> dict:
    with h5py.File(measurement_path, "r") as h5:
        if "theory_interface" not in h5:
            return {}
        raw = h5["theory_interface"].attrs.get("des_y3_gaussian_priors_json", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _shear_m_means(priors: Mapping[str, object]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i in range(1, 5):
        entry = priors.get(f"mult_shear_bias_bin{i}", {})
        out[i] = float(entry.get("mu", 0.0)) if isinstance(entry, Mapping) else 0.0
    return out


def _delta_z_means(priors: Mapping[str, object]) -> List[float]:
    out: List[float] = []
    for i in range(1, 5):
        entry = priors.get(f"Delta_z_bias_bin{i}", {})
        out.append(float(entry.get("mu", 0.0)) if isinstance(entry, Mapping) else 0.0)
    return out


def _serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(key): _serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(value) for value in obj]
    return obj


def to_jsonable(obj):
    return _serializable(obj)


def validate_comparison_halo_mass_floor(params: Mapping[str, object], raw: Mapping[str, object]) -> dict:
    required = raw.get("minimum_halo_log10_m200c_hmsun")
    halo_params = params.get("halo_params", {})
    actual = halo_params.get("lg10_Mmin")
    if required is None:
        return {
            "required": None,
            "actual": float(actual) if actual is not None else None,
            "enforced": False,
        }
    if actual is None:
        raise ValueError(
            "Comparison config requires minimum_halo_log10_m200c_hmsun="
            f"{float(required):.6f}, but merged halo_params.lg10_Mmin is missing."
        )
    actual_float = float(actual)
    required_float = float(required)
    if not math.isfinite(actual_float):
        raise ValueError(f"halo_params.lg10_Mmin is not finite: {actual!r}.")
    if abs(actual_float - required_float) > HALO_MASS_FLOOR_TOL:
        raise ValueError(
            "Backlight-compatible theory comparisons require the GODMAX halo mass grid "
            f"to start at log10(M200c/[Msun/h])={required_float:.6f}; "
            f"merged halo_params.lg10_Mmin={actual_float:.6f}."
        )
    return {
        "required": required_float,
        "actual": actual_float,
        "enforced": True,
        "mass_definition": "M200c",
        "mass_units": "Msun/h",
    }


def load_comparison_config(config_path: str | Path) -> dict:
    """Load and merge the path-backed comparison YAML into GODMAX parameters."""

    root = repo_root()
    config_path = resolve_repo_path(config_path, root)
    raw = read_yaml(config_path)
    base_path = resolve_repo_path(raw["base_params"], root)
    xdesi_path = resolve_repo_path(raw["xdesi_params"], root)
    params = deep_update(read_yaml(base_path), read_yaml(xdesi_path))
    if "cosmology" in raw:
        params.setdefault("sim_params", {}).setdefault("cosmo", {}).update(copy.deepcopy(raw["cosmology"]))
    if "overrides" in raw:
        params = deep_update(params, raw["overrides"])
    halo_mass_floor = validate_comparison_halo_mass_floor(params, raw)

    paths = {
        "config": config_path,
        "base_params": base_path,
        "xdesi_params": xdesi_path,
        "measurement_h5": resolve_repo_path(raw["measurement_h5"], root),
        "map_h5": resolve_repo_path(raw["map_h5"], root),
        "output_dir": resolve_repo_path(raw["output_dir"], root),
    }
    return {
        "repo_root": root,
        "raw": raw,
        "params": params,
        "paths": paths,
        "warnings": [],
        "metadata": {"halo_mass_floor": halo_mass_floor},
    }


def load_measurement_data(
    measurement_path: str | Path,
    *,
    include_invalid_placeholders: bool = False,
    allow_legacy_product: bool = False,
    galaxy_auto_view: str = "total",
) -> MeasurementData:
    path = resolve_repo_path(measurement_path)
    with h5py.File(path, "r") as h5:
        from multiprobe_namaster import (
            DESI_GALAXY_AUTO_PRIMARY_VIEW,
            DESI_GALAXY_AUTO_SUBTRACTED_VIEW,
            validate_galaxy_auto_views,
            validate_measurement_product_identity,
        )

        validate_measurement_product_identity(
            h5,
            allow_legacy_product=allow_legacy_product,
        )
        aliases = {
            "total": DESI_GALAXY_AUTO_PRIMARY_VIEW,
            "cl_gg_plus_sn": DESI_GALAXY_AUTO_PRIMARY_VIEW,
            "weighted_poisson_subtracted": DESI_GALAXY_AUTO_SUBTRACTED_VIEW,
            "shot_subtracted": DESI_GALAXY_AUTO_SUBTRACTED_VIEW,
        }
        requested_view = aliases.get(str(galaxy_auto_view).strip().lower())
        if requested_view is None:
            raise ValueError(
                "galaxy_auto_view must be 'total' or 'weighted_poisson_subtracted'."
            )
        if requested_view == DESI_GALAXY_AUTO_PRIMARY_VIEW:
            data_path = "joint/data_vector"
            covariance_path = "joint/cov"
        else:
            validate_galaxy_auto_views(h5, require=True)
            view_root = f"joint/views/{DESI_GALAXY_AUTO_SUBTRACTED_VIEW}"
            data_path = f"{view_root}/data_vector"
            covariance_path = f"{view_root}/cov"
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["joint/spectrum_names"][:]]
        starts = h5["joint/slice_start"][:].astype(int)
        stops = h5["joint/slice_stop"][:].astype(int)
        families: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        theory_keys: Dict[str, str] = {}
        for name in names:
            group = h5[f"spectra/{name}"]
            families[name] = str(group.attrs.get("family", "unknown"))
            labels[name] = str(group.attrs.get("label", name))
            theory_keys[name] = str(group.attrs.get("theory_key", name))
        transfer_null_from: Dict[str, float] = {}
        kappa_curve_path = "transfer_functions/act_kappa_filter_baseline"
        if kappa_curve_path in h5:
            curve = np.asarray(h5[kappa_curve_path][:], dtype=np.float64)
            if curve.ndim == 2 and curve.shape[1] >= 2 and curve.shape[0] > 1:
                amplitude = np.abs(curve[:, 1])
                scale = float(np.max(amplitude))
                nonzero = amplitude > max(np.finfo(np.float64).tiny, scale * 1.0e-12)
                nonzero_indices = np.flatnonzero(nonzero)
                if nonzero_indices.size:
                    first_trailing = int(nonzero_indices[-1]) + 1
                    if first_trailing < curve.shape[0] and not np.any(nonzero[first_trailing:]):
                        transfer_null_from["desi_g_act_kappa"] = float(curve[first_trailing, 0])
        archive_data = np.asarray(h5[data_path][:], dtype=np.float64)
        archive_cov = np.asarray(h5[covariance_path][:], dtype=np.float64)
        archive_valid = (
            np.asarray(h5["joint/data_vector_valid"][:], dtype=bool)
            if "joint/data_vector_valid" in h5
            else np.ones(archive_data.size, dtype=bool)
        )
        if archive_valid.shape != archive_data.shape:
            raise ValueError("Saved data-vector validity mask has the wrong shape.")
        archive_indices = np.arange(archive_data.size, dtype=np.int64)
        ell_common = np.asarray(h5["joint/ell"][:], dtype=np.float64)
        ell_left_common = np.asarray(h5["ell_left"][:]) if "ell_left" in h5 else None
        ell_right_common = np.asarray(h5["ell_right"][:]) if "ell_right" in h5 else None
        if include_invalid_placeholders or np.all(archive_valid):
            data_vector = archive_data
            covariance = archive_cov
            ell_out = ell_common
            starts_out = starts
            stops_out = stops
            valid_out = archive_valid
            ell_left_out = ell_left_common
            ell_right_out = ell_right_common
            selected_archive_indices = archive_indices
        else:
            selected_archive_indices = archive_indices[archive_valid]
            data_vector = archive_data[archive_valid]
            covariance = archive_cov[np.ix_(archive_valid, archive_valid)]
            starts_active: List[int] = []
            stops_active: List[int] = []
            ell_chunks: List[np.ndarray] = []
            left_chunks: List[np.ndarray] = []
            right_chunks: List[np.ndarray] = []
            cursor = 0
            for start, stop in zip(starts, stops):
                local_valid = archive_valid[int(start) : int(stop)]
                if not np.any(local_valid):
                    raise ValueError("A saved spectrum has no statistically valid bandpowers.")
                starts_active.append(cursor)
                cursor += int(np.count_nonzero(local_valid))
                stops_active.append(cursor)
                ell_chunks.append(ell_common[local_valid])
                if ell_left_common is not None and ell_right_common is not None:
                    left_chunks.append(ell_left_common[local_valid])
                    right_chunks.append(ell_right_common[local_valid])
            ell_out = np.concatenate(ell_chunks)
            starts_out = np.asarray(starts_active, dtype=int)
            stops_out = np.asarray(stops_active, dtype=int)
            valid_out = np.ones(data_vector.size, dtype=bool)
            ell_left_out = np.concatenate(left_chunks) if left_chunks else None
            ell_right_out = np.concatenate(right_chunks) if right_chunks else None
        return MeasurementData(
            path=path,
            names=names,
            ell=ell_out,
            data_vector=data_vector,
            covariance=covariance,
            starts=starts_out,
            stops=stops_out,
            families=families,
            labels=labels,
            theory_keys=theory_keys,
            ell_left=ell_left_out,
            ell_right=ell_right_out,
            transfer_null_from=transfer_null_from,
            data_vector_valid=valid_out,
            archive_indices=selected_archive_indices,
            archive_data_vector_size=int(archive_data.size),
            galaxy_auto_view=requested_view,
        )


def materialize_nz_inputs(config: Mapping[str, object]) -> dict:
    """Read HDF5 n(z) products and inject GODMAX-ready n(z) dictionaries."""

    cfg = copy.deepcopy(dict(config))
    raw = cfg["raw"]
    datasets = raw["datasets"]
    paths = cfg["paths"]
    measurement_path = Path(paths["measurement_h5"])
    map_path = Path(paths["map_h5"])
    if not measurement_path.exists():
        raise FileNotFoundError(f"Missing measurement HDF5: {measurement_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"Missing map HDF5 with DESI weighted n(z): {map_path}")

    warnings = list(cfg.get("warnings", []))
    with h5py.File(measurement_path, "r") as mh5, h5py.File(map_path, "r") as map_h5:
        allow_legacy_product = allow_legacy_product_from_config(cfg)
        map_product_id = validate_measurement_map_identity(
            mh5,
            map_h5,
            allow_legacy_product=allow_legacy_product,
        )
        if not allow_legacy_product:
            _validate_theory_nz_content(mh5, map_h5)
        measurement_config = json.loads(str(mh5.attrs["config_json"]))
        field_meta = _json_load_attr(mh5["fields"].attrs, "metadata_json", {})
        priors = _priors_from_measurement(measurement_path)

        des_cfg = datasets["des_source_nz"]
        src_file = mh5 if des_cfg["file"] == "measurement_h5" else map_h5
        z_source = np.asarray(src_file[des_cfg["z_mid"]][:], dtype=np.float64)
        dndz_source = _normalize_rows(
            z_source,
            np.asarray(src_file[des_cfg["dndz_by_bin"]][:], dtype=np.float64),
            "DES source n(z)",
            warnings,
        )

        lens_cfg = datasets["desi_lens_nz"]
        lens_file = map_h5 if lens_cfg["file"] == "map_h5" else mh5
        z_lens = np.asarray(lens_file[lens_cfg["z_mid"]][:], dtype=np.float64)
        z_edges = np.asarray(lens_file[lens_cfg["z_edges"]][:], dtype=np.float64)
        lens_group_path = str(Path(lens_cfg["dndz_by_pz"]).parent)
        lens_group = lens_file[lens_group_path] if lens_group_path in lens_file else None
        lens_redshift_kind = str(lens_group.attrs.get("redshift_kind", "")) if lens_group is not None else ""
        if lens_redshift_kind != DESI_TRUE_NZ_REDSHIFT_KIND:
            raise ValueError(
                "DESI lens n(z) is not the calibrated true-redshift kernel. "
                f"Read redshift_kind={lens_redshift_kind!r} from {lens_cfg['dndz_by_pz']}; "
                "rerun the fast1024 measurement into the true-n(z) output directory."
            )
        dndz_lens = _normalize_rows(
            z_lens,
            np.asarray(lens_file[lens_cfg["dndz_by_pz"]][:], dtype=np.float64),
            "DESI calibrated true lens n(z)",
            warnings,
        )
        lens_edges = _support_edges(z_lens, z_edges, dndz_lens)
        lens_nz_provenance = {}
        if lens_group is not None:
            for key, value in lens_group.attrs.items():
                lens_nz_provenance[str(key)] = value.decode("utf-8") if isinstance(value, bytes) else value
        photoz_diagnostic = None
        if "nz/desi_photoz_diagnostic" in lens_file:
            pg = lens_file["nz/desi_photoz_diagnostic"]
            photoz_diagnostic = {
                "z_mid": np.asarray(pg["z_mid"][:], dtype=np.float64) if "z_mid" in pg else None,
                "z_edges": np.asarray(pg["z_edges"][:], dtype=np.float64) if "z_edges" in pg else None,
                "dndz_by_pz": np.asarray(pg["nz_dndz_by_pz"][:], dtype=np.float64) if "nz_dndz_by_pz" in pg else None,
                "description": str(pg.attrs.get("description", "")),
                "redshift_kind": str(pg.attrs.get("redshift_kind", "")),
            }

        ksz_default_a_v = (
            mh5["theory_interface/ksz_default_A_v_by_pz"][:]
            if "theory_interface/ksz_default_A_v_by_pz" in mh5
            else np.full(4, np.nan)
        )
        ksz_sigma_true = (
            mh5["theory_interface/ksz_sigma_true_gas_over_c_by_pz"][:]
            if "theory_interface/ksz_sigma_true_gas_over_c_by_pz" in mh5
            else np.full(4, np.nan)
        )

    source_info = {
        "nbins": int(dndz_source.shape[0]),
        "z_array_source": z_source.tolist(),
    }
    for i in range(dndz_source.shape[0]):
        source_info[f"nz{i}"] = dndz_source[i].tolist()

    lens_info = {
        "nbins_lens": int(dndz_lens.shape[0]),
        "z_array_lens": z_lens.tolist(),
        "z_edges_bins_lens": lens_edges.tolist(),
    }
    for i in range(dndz_lens.shape[0]):
        lens_info[f"nz{i}"] = dndz_lens[i].tolist()

    params = cfg["params"]
    params.setdefault("analysis", {})["nz_source_info_dict"] = source_info
    params.setdefault("analysis", {})["nz_lens_info_dict"] = lens_info
    other_params = params.setdefault("other_params", {})
    if not bool(other_params.get("sampled_des_y3_nuisance", False)):
        other_params["Delta_z_bias_array"] = _delta_z_means(priors)

    metadata = dict(cfg.get("metadata", {}))
    metadata.update(
        {
            "measurement_config": measurement_config,
            "field_metadata": field_meta,
            "des_y3_gaussian_priors": priors,
            "shear_m_bias_means": _shear_m_means(priors),
            "source_z_mid": z_source,
            "source_dndz": dndz_source,
            "lens_z_mid": z_lens,
            "lens_z_edges": z_edges,
            "lens_dndz": dndz_lens,
            "lens_edges": lens_edges,
            "lens_redshift_kind": lens_redshift_kind,
            "lens_nz_provenance": lens_nz_provenance,
            "lens_photoz_diagnostic": photoz_diagnostic,
            "ksz_default_A_v_by_pz": np.asarray(ksz_default_a_v, dtype=np.float64),
            "ksz_sigma_true_gas_over_c_by_pz": np.asarray(ksz_sigma_true, dtype=np.float64),
            "lmax": int(measurement_config["lmax"]),
            "map_product_id": map_product_id,
        }
    )
    cfg["metadata"] = metadata
    cfg["warnings"] = warnings
    return cfg


def _jax_cosmo_distances(cosmo_params: Mapping[str, float], z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_cosmo.background as bkgrd
    from astropy import constants as const
    from jax_cosmo import Cosmology

    cosmo = Cosmology(
        Omega_c=float(cosmo_params["Om0"]) - float(cosmo_params["Ob0"]),
        Omega_b=float(cosmo_params["Ob0"]),
        h=float(cosmo_params["H0"]) / 100.0,
        sigma8=float(cosmo_params["sigma8"]),
        n_s=float(cosmo_params["ns"]),
        Omega_k=0.0,
        w0=float(cosmo_params["w0"]),
        wa=0.0,
    )
    a = jnp.asarray(1.0 / (1.0 + np.asarray(z, dtype=np.float64)))
    chi = np.asarray(bkgrd.radial_comoving_distance(cosmo, a), dtype=np.float64)
    dchi_dz = np.asarray((const.c.value * 1.0e-3) / bkgrd.H(cosmo, a), dtype=np.float64)
    return chi, dchi_dz


def _hod_bin_indices_for_z(z: np.ndarray, z_edges_bins_lens: np.ndarray) -> np.ndarray:
    """Match Profiles.setup_hod_params bin assignment on an arbitrary z grid.

    Returned indices are 0 for no lens bin and 1..nbin for populated bins.
    In overlapping support regions, the highest matching bin index wins, which
    matches the jnp.max(where(mask, indices + 1, 0)) convention in Profiles.
    """

    z = np.asarray(z, dtype=np.float64)
    edges = np.asarray(z_edges_bins_lens, dtype=np.float64)
    out = np.zeros(z.shape, dtype=np.int64)
    for i, (lo, hi) in enumerate(edges, start=1):
        out[(z > lo) & (z < hi)] = i
    return out


def compute_desi_nbar_comoving(config: Mapping[str, object]) -> dict:
    """Compute per-photo-z-bin HOD nbar(z) from measured angular densities and true DESI n(z)."""

    cfg = copy.deepcopy(dict(config))
    metadata = cfg.get("metadata", {})
    if "lens_z_mid" not in metadata or "lens_dndz" not in metadata:
        cfg = materialize_nz_inputs(cfg)
        metadata = cfg["metadata"]

    analysis = cfg.get("params", {}).get("analysis", {})
    cached_keys = (
        "nbar_per_sr_by_pz",
        "nbar_comoving_by_pz",
        "nbar_comoving_total",
        "nbar_comoving_target_by_pz",
        "nbar_physical_by_pz",
        "chi_lens_hmpc",
        "dchi_dz_lens_hmpc",
    )
    if all(key in metadata for key in cached_keys) and {
        "nbar_gal_comoving_zarray",
        "nbar_gal_comoving_val",
    }.issubset(analysis):
        return cfg

    z = np.asarray(metadata["lens_z_mid"], dtype=np.float64)
    dndz = np.asarray(metadata["lens_dndz"], dtype=np.float64)
    field_meta = metadata["field_metadata"]
    nbar_per_sr = []
    for i in range(1, 5):
        meta_outer = field_meta.get(f"g{i}", {})
        meta = meta_outer.get("metadata", {}) if isinstance(meta_outer, Mapping) else {}
        if "nbar_per_sr" not in meta:
            raise KeyError(f"fields metadata for g{i} lacks nbar_per_sr.")
        nbar_per_sr.append(float(meta["nbar_per_sr"]))
    nbar_per_sr_arr = np.asarray(nbar_per_sr, dtype=np.float64)

    chi, dchi_dz = _jax_cosmo_distances(cfg["params"]["sim_params"]["cosmo"], z)
    denom = chi**2 * dchi_dz
    nbar_by_pz = np.divide(
        nbar_per_sr_arr[:, None] * dndz,
        denom[None, :],
        out=np.zeros_like(dndz, dtype=np.float64),
        where=np.isfinite(denom[None, :]) & (denom[None, :] > 0.0),
    )
    nbar_sum = np.sum(nbar_by_pz, axis=0)
    lens_edges = np.asarray(metadata["lens_edges"], dtype=np.float64)
    hod_bin_indices = _hod_bin_indices_for_z(z, lens_edges)
    floor = float(cfg["raw"].get("nbar", {}).get("floor_comoving_hmpc3", 1.0e-10))
    nbar_sum = np.maximum(np.nan_to_num(nbar_sum, nan=0.0, posinf=0.0, neginf=0.0), floor)
    nbar_by_pz = np.maximum(np.nan_to_num(nbar_by_pz, nan=0.0, posinf=0.0, neginf=0.0), floor)
    nbar_physical_by_pz = nbar_by_pz * (1.0 + z[None, :]) ** 3
    nbar_physical_sum = nbar_sum * (1.0 + z) ** 3

    cfg["params"]["analysis"]["nbar_gal_comoving_zarray"] = z.tolist()
    cfg["params"]["analysis"]["nbar_gal_comoving_val"] = nbar_sum.tolist()
    cfg["metadata"] = dict(metadata)
    cfg["metadata"].update(
        {
            "nbar_per_sr_by_pz": nbar_per_sr_arr,
            "nbar_comoving_by_pz": nbar_by_pz,
            "nbar_comoving_total": nbar_sum,
            "nbar_comoving_sum_by_pz": nbar_sum,
            "nbar_comoving_target": nbar_sum,
            "nbar_comoving_target_by_pz": nbar_by_pz,
            "nbar_physical_by_pz": nbar_physical_by_pz,
            "nbar_physical_target": nbar_physical_sum,
            "nbar_physical_target_by_pz": nbar_physical_by_pz,
            "nbar_hod_bin_indices": hod_bin_indices,
            "nbar_target_convention": "per_photoz_bin_single_godmax_models",
            "chi_lens_hmpc": chi,
            "dchi_dz_lens_hmpc": dchi_dz,
        }
    )
    return cfg


def validation_summary(config: Mapping[str, object]) -> dict:
    metadata = config["metadata"]
    return {
        "measurement_h5": config["paths"]["measurement_h5"],
        "map_h5": config["paths"]["map_h5"],
        "lmax": metadata["lmax"],
        "des_source_nz_shape": np.asarray(metadata["source_dndz"]).shape,
        "desi_lens_nz_shape": np.asarray(metadata["lens_dndz"]).shape,
        "desi_lens_redshift_kind": metadata.get("lens_redshift_kind", ""),
        "desi_lens_nz_provenance": metadata.get("lens_nz_provenance", {}),
        "desi_lens_edges": np.asarray(metadata["lens_edges"]),
        "ksz_default_A_v_by_pz": np.asarray(metadata["ksz_default_A_v_by_pz"]),
        "ksz_sigma_true_gas_over_c_by_pz": np.asarray(metadata["ksz_sigma_true_gas_over_c_by_pz"]),
        "shear_m_bias_means": metadata["shear_m_bias_means"],
        "warnings": config.get("warnings", []),
    }


def ensure_godmax_import_paths(root: Optional[Path] = None) -> None:
    root = root or repo_root()
    for path in (root / "src", root / "notebooks" / "xDESI" / "survey_measure"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _collapse_hod_array_for_pz(values: object, pz_bin: int):
    import jax.numpy as jnp

    arr = jnp.atleast_1d(jnp.asarray(values, dtype=jnp.float64))
    if arr.shape[0] == 1:
        return arr
    if arr.shape[0] <= int(pz_bin):
        raise ValueError(f"HOD array has length {arr.shape[0]}, cannot select pz bin {pz_bin}.")
    return jnp.stack([arr[0], arr[int(pz_bin)]])


def _collapse_hod_arrays_for_single_pz(params: dict, pz_bin: int) -> None:
    sim_params = params["sim_params"]
    for base in HOD_ARRAY_BASE_NAMES:
        key = f"{base}_array"
        if key in sim_params:
            sim_params[key] = _collapse_hod_array_for_pz(sim_params[key], pz_bin)


def config_without_galaxies(config: Mapping[str, object]) -> dict:
    cfg = copy.deepcopy(dict(config))
    cfg = compute_desi_nbar_comoving(cfg)
    cfg["params"] = copy.deepcopy(cfg["params"])
    cfg["params"].setdefault("analysis", {})["model_galaxies"] = False
    return cfg


def config_for_single_desi_pz(config: Mapping[str, object], pz_bin: int) -> dict:
    """Return a GODMAX config whose lone lens bin is one photometric DESI pz bin."""

    pz_bin = int(pz_bin)
    if pz_bin < 1 or pz_bin > 4:
        raise ValueError(f"pz_bin must be 1..4, got {pz_bin}.")
    cfg = copy.deepcopy(dict(config))
    cfg = compute_desi_nbar_comoving(cfg)
    metadata = cfg["metadata"]
    z_lens = np.asarray(metadata["lens_z_mid"], dtype=np.float64)
    dndz = np.asarray(metadata["lens_dndz"], dtype=np.float64)[pz_bin - 1]
    z_edges = np.asarray(metadata["lens_edges"], dtype=np.float64)[pz_bin - 1 : pz_bin]
    nbar = np.asarray(metadata["nbar_comoving_by_pz"], dtype=np.float64)[pz_bin - 1]

    params = copy.deepcopy(cfg["params"])
    params["analysis"]["nz_lens_info_dict"] = {
        "nbins_lens": 1,
        "z_array_lens": z_lens.tolist(),
        "z_edges_bins_lens": z_edges.tolist(),
        "nz0": dndz.tolist(),
    }
    params["analysis"]["nbar_gal_comoving_zarray"] = z_lens.tolist()
    params["analysis"]["nbar_gal_comoving_val"] = nbar.tolist()
    params["analysis"]["model_galaxies"] = True
    params["analysis"]["hod_params_model"] = "perbin"
    params["analysis"]["single_photometric_pz_bin"] = pz_bin
    _collapse_hod_arrays_for_single_pz(params, pz_bin)

    cfg["params"] = params
    cfg["metadata"] = dict(metadata)
    cfg["metadata"].update(
        {
            "single_photometric_pz_bin": pz_bin,
            "single_pz_lens_dndz": dndz,
            "single_pz_nbar_comoving": nbar,
        }
    )
    return cfg


def _params_for_model(config: Mapping[str, object], *, is_cmb_lensing: bool) -> Tuple[dict, dict, dict, dict]:
    import jax.numpy as jnp

    cfg = compute_desi_nbar_comoving(config)
    params = copy.deepcopy(cfg["params"])
    lmax = int(cfg["metadata"]["lmax"])
    # Optional sparse compute grid: evaluate the (expensive) halo model + C(l) on a
    # log-spaced subset of integer multipoles and interpolate back to the dense
    # integer grid downstream (see _densify_theory_cls_to_full_lmax in the HMC
    # driver). This slashes the reverse-mode memory of the per-leapfrog gradient
    # (dense l=2..lmax keeps a ~60 GiB/4-chain tape) at <0.01% accuracy cost.
    # Default (nell_compute unset) preserves the exact dense behavior.
    nell_compute = params["halo_params"].pop("nell_compute", None)
    if nell_compute is not None and 0 < int(nell_compute) < (lmax - 1):
        ell_sparse = np.unique(
            np.geomspace(2.0, float(lmax), int(nell_compute)).round().astype(np.int64)
        ).astype(np.float64)
        params["halo_params"]["ell_array"] = jnp.asarray(ell_sparse, dtype=jnp.float64)
    else:
        params["halo_params"]["ell_array"] = jnp.arange(2, lmax + 1, dtype=jnp.float64)
    params["analysis"]["is_cmb_lensing"] = bool(is_cmb_lensing)
    params["analysis"]["symbolic_pk"] = False
    params["analysis"]["symbolic_hmf"] = False
    if is_cmb_lensing:
        z_source = params["analysis"]["nz_source_info_dict"]["z_array_source"]
        params["analysis"]["nz_source_info_dict"] = {
            "nbins": 1,
            "z_array_source": z_source,
            "nz0": np.ones(len(z_source), dtype=np.float64).tolist(),
        }
        params["other_params"]["Delta_z_bias_array"] = [0.0]
        params["other_params"]["mult_shear_bias_array"] = [0.0]
    return (
        params["sim_params"],
        params["halo_params"],
        params["analysis"],
        params["other_params"],
    )


def build_one_godmax_model(config: Mapping[str, object], *, is_cmb_lensing: bool):
    ensure_godmax_import_paths(Path(config["repo_root"]))
    from base_class import base_class
    from get_Cls import get_Cl
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    sim_params, halo_params, analysis, other_params = _params_for_model(config, is_cmb_lensing=is_cmb_lensing)
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)
    cls = get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)
    return cls


def build_godmax_models(config: Mapping[str, object]) -> Dict[str, object]:
    """Build GODMAX theory objects for a photometric-bin-consistent comparison.

    The shared ``wl`` object has ``model_galaxies=False`` and is used for pure
    DES shear and ACT-y x shear spectra.  DESI spectra are assembled from one
    single-lens-bin WL and CMB-lensing object per pz bin, so overlapping
    calibrated true-redshift kernels never force HOD parameters onto true-z
    support intervals.
    """

    cfg = compute_desi_nbar_comoving(config)
    wl_nongal = build_one_godmax_model(config_without_galaxies(cfg), is_cmb_lensing=False)
    gal_wl_by_pz = {}
    gal_cmb_by_pz = {}
    for pz_bin in range(1, 5):
        pz_cfg = config_for_single_desi_pz(cfg, pz_bin)
        gal_wl_by_pz[pz_bin] = build_one_godmax_model(pz_cfg, is_cmb_lensing=False)
        gal_cmb_by_pz[pz_bin] = build_one_godmax_model(pz_cfg, is_cmb_lensing=True)
    return {
        "wl": wl_nongal,
        "cmb": gal_cmb_by_pz[1],
        "gal_wl_by_pz": gal_wl_by_pz,
        "gal_cmb_by_pz": gal_cmb_by_pz,
        "modeling": "single_godmax_object_per_photometric_pz_bin",
    }


def ne0_cm3(cosmo_params: Mapping[str, float], helium_mass_fraction: float = 0.24) -> float:
    from astropy import constants as const

    h = float(cosmo_params["H0"]) / 100.0
    rho_crit_0 = 1.878e-29 * h**2
    return float(rho_crit_0 * float(cosmo_params["Ob0"]) * (1.0 - helium_mass_fraction / 2.0) / const.m_p.to("g").value)


def corrected_gal_tau_cls_zdependent(cls_obj) -> np.ndarray:
    """Apply physical ne0*(1+z)^3 tau normalization inside the Limber integral."""

    z = np.asarray(cls_obj.z_array_for_Cls, dtype=np.float64)
    chi = np.asarray(cls_obj.chi_array_for_Cls, dtype=np.float64)
    dchi_dz = np.asarray(cls_obj.dchi_dz_array_for_Cls, dtype=np.float64)
    wg = np.asarray(cls_obj.Wg_mat, dtype=np.float64)
    wtau = np.asarray(cls_obj.Wtau_array, dtype=np.float64)
    pge = np.asarray(cls_obj.cached_power_spectra[2, 4], dtype=np.float64)
    wtau_corrected = wtau * ne0_cm3(cls_obj.cosmo_params) * (1.0 + z) ** 3
    prefac_tau = wtau_corrected / chi**2
    out = np.zeros((len(cls_obj.ell_array), wg.shape[0]), dtype=np.float64)
    for j in range(wg.shape[0]):
        prefac_g = wg[j] / (dchi_dz * chi**2)
        integrand = pge * prefac_g[None, :] * prefac_tau[None, :] * chi[None, :] ** 2 * dchi_dz[None, :]
        out[:, j] = _trapz(integrand, z, axis=1)
    return out


def corrected_gal_tau_cls_effz(cls_obj, lens_mean_z: Sequence[float]) -> np.ndarray:
    raw = np.asarray(cls_obj.Cl_gal_tau_tot_mat, dtype=np.float64).copy()
    out = np.zeros_like(raw)
    ne0 = ne0_cm3(cls_obj.cosmo_params)
    for j, z_mean in enumerate(lens_mean_z):
        out[:, j] = raw[:, j] * ne0 * (1.0 + float(z_mean)) ** 3
    return out


def lens_mean_z_from_metadata(metadata: Mapping[str, object]) -> np.ndarray:
    field_meta = metadata["field_metadata"]
    vals = []
    for i in range(1, 5):
        meta_outer = field_meta.get(f"g{i}", {})
        meta = meta_outer.get("metadata", {}) if isinstance(meta_outer, Mapping) else {}
        vals.append(float(meta.get("mean_true_z", meta.get("mean_z", np.nan))))
    return np.asarray(vals, dtype=np.float64)


def model_ell_array(models: Mapping[str, object]) -> np.ndarray:
    return np.asarray(models["wl"].ell_array, dtype=np.float64)


def extract_theory_cls_from_models(
    models: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    return_diagnostics: bool = False,
):
    """Extract smooth spectra from the pz-specific GODMAX model bundle."""

    cls_wl = models["wl"]
    gal_wl_by_pz = models["gal_wl_by_pz"]
    gal_cmb_by_pz = models["gal_cmb_by_pz"]
    theory: Dict[str, np.ndarray] = {}
    for i in range(4):
        for j in range(i, 4):
            theory[f"des_shear_EE_tomo{i + 1}_tomo{j + 1}"] = np.asarray(
                cls_wl.Cl_kappa_kappa_tot_mat[:, i, j], dtype=np.float64
            )
    for i in range(4):
        theory[f"act_y_des_shear_E_tomo{i + 1}"] = np.asarray(cls_wl.Cl_kappa_y_tot_mat[:, i], dtype=np.float64)

    tau_zdep_cols = []
    tau_effz_cols = []
    lens_mean_z = lens_mean_z_from_metadata(metadata)
    for pz_bin in range(1, 5):
        pz_wl = gal_wl_by_pz[pz_bin]
        pz_cmb = gal_cmb_by_pz[pz_bin]
        theory[f"desi_g_auto_pz{pz_bin}"] = np.asarray(pz_wl.Cl_gal_gal_tot_mat[:, 0, 0], dtype=np.float64)
        theory[f"desi_g_act_y_pz{pz_bin}"] = np.asarray(pz_wl.Cl_gal_y_tot_mat[:, 0], dtype=np.float64)
        theory[f"desi_g_act_kappa_pz{pz_bin}"] = np.asarray(pz_cmb.Cl_gal_kappa_tot_mat[:, 0, 0], dtype=np.float64)
        for tomo in range(1, 5):
            theory[f"desi_g_des_shear_E_pz{pz_bin}_tomo{tomo}"] = np.asarray(
                pz_wl.Cl_gal_kappa_tot_mat[:, 0, tomo - 1], dtype=np.float64
            )
        tau_zdep = corrected_gal_tau_cls_zdependent(pz_wl)[:, 0]
        tau_effz = corrected_gal_tau_cls_effz(pz_wl, [lens_mean_z[pz_bin - 1]])[:, 0]
        theory[f"desi_g_tau_pz{pz_bin}"] = tau_zdep
        tau_zdep_cols.append(tau_zdep)
        tau_effz_cols.append(tau_effz)

    if not return_diagnostics:
        return theory

    tau_zdep_mat = np.stack(tau_zdep_cols, axis=1)
    tau_effz_mat = np.stack(tau_effz_cols, axis=1)
    diagnostics = {
        "ell": np.asarray(cls_wl.ell_array, dtype=np.float64),
        "lens_mean_z": lens_mean_z,
        "gal_tau_zdependent": tau_zdep_mat,
        "gal_tau_effective_z_approx": tau_effz_mat,
        "gal_tau_effective_over_zdependent": np.divide(
            tau_effz_mat,
            tau_zdep_mat,
            out=np.full_like(tau_zdep_mat, np.nan),
            where=tau_zdep_mat != 0.0,
        ),
        "modeling": models.get("modeling", ""),
    }
    return theory, diagnostics


def extract_theory_cls(
    cls_wl,
    cls_cmb,
    metadata: Mapping[str, object],
    *,
    return_diagnostics: bool = False,
):
    """Extract smooth GODMAX spectra keyed by the measurement theory interface."""

    if isinstance(cls_wl, Mapping) and "gal_wl_by_pz" in cls_wl:
        return extract_theory_cls_from_models(cls_wl, metadata, return_diagnostics=return_diagnostics)

    theory: Dict[str, np.ndarray] = {}
    for i in range(4):
        for j in range(i, 4):
            theory[f"des_shear_EE_tomo{i + 1}_tomo{j + 1}"] = np.asarray(
                cls_wl.Cl_kappa_kappa_tot_mat[:, i, j], dtype=np.float64
            )
    for i in range(4):
        theory[f"act_y_des_shear_E_tomo{i + 1}"] = np.asarray(cls_wl.Cl_kappa_y_tot_mat[:, i], dtype=np.float64)
        theory[f"desi_g_auto_pz{i + 1}"] = np.asarray(cls_wl.Cl_gal_gal_tot_mat[:, i, i], dtype=np.float64)
        theory[f"desi_g_act_y_pz{i + 1}"] = np.asarray(cls_wl.Cl_gal_y_tot_mat[:, i], dtype=np.float64)
        theory[f"desi_g_act_kappa_pz{i + 1}"] = np.asarray(cls_cmb.Cl_gal_kappa_tot_mat[:, i, 0], dtype=np.float64)
    for i in range(4):
        for j in range(4):
            theory[f"desi_g_des_shear_E_pz{i + 1}_tomo{j + 1}"] = np.asarray(
                cls_wl.Cl_gal_kappa_tot_mat[:, i, j], dtype=np.float64
            )

    tau_zdep = corrected_gal_tau_cls_zdependent(cls_wl)
    for i in range(4):
        theory[f"desi_g_tau_pz{i + 1}"] = tau_zdep[:, i]

    if not return_diagnostics:
        return theory

    lens_mean_z = lens_mean_z_from_metadata(metadata)
    tau_effz = corrected_gal_tau_cls_effz(cls_wl, lens_mean_z)
    diagnostics = {
        "ell": np.asarray(cls_wl.ell_array, dtype=np.float64),
        "lens_mean_z": lens_mean_z,
        "gal_tau_zdependent": tau_zdep,
        "gal_tau_effective_z_approx": tau_effz,
        "gal_tau_effective_over_zdependent": np.divide(
            tau_effz,
            tau_zdep,
            out=np.full_like(tau_zdep, np.nan),
            where=tau_zdep != 0.0,
        ),
    }
    return theory, diagnostics


def validate_theory_keys(
    measurement_path: str | Path,
    theory_cls: Mapping[str, np.ndarray],
    *,
    allow_legacy_product: bool = False,
) -> List[str]:
    missing = []
    with h5py.File(resolve_repo_path(measurement_path), "r") as h5:
        from multiprobe_namaster import validate_measurement_product_identity

        validate_measurement_product_identity(
            h5,
            allow_legacy_product=allow_legacy_product,
        )
        for raw_name in h5["joint/spectrum_names"][:]:
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            group = h5[f"spectra/{name}"]
            theory_key = str(group.attrs.get("theory_key", name))
            family = str(group.attrs.get("family", ""))
            if name not in theory_cls and not (family == "desi_pi_act_T" and theory_key in theory_cls):
                missing.append(f"{name} or {theory_key}")
    return missing


def desi_galaxy_shot_noise_amplitudes_from_config(config: Mapping[str, object]) -> object:
    """Resolve one unambiguous fixed or saved/best-fit shot-amplitude source."""

    raw = config.get("raw", {}).get("theory_to_data_vector", {})
    has_explicit_wrapper_value = "desi_galaxy_shot_noise_amplitudes" in raw
    other_params = config.get("params", {}).get("other_params", {})
    sampled_or_saved = {
        pz_bin: other_params[f"desi_galaxy_shot_noise_amplitude_pz{pz_bin}"]
        for pz_bin in range(1, 5)
        if f"desi_galaxy_shot_noise_amplitude_pz{pz_bin}" in other_params
    }
    if has_explicit_wrapper_value and sampled_or_saved:
        raise ValueError(
            "Ambiguous DESI galaxy shot-noise amplitudes: remove either "
            "raw.theory_to_data_vector.desi_galaxy_shot_noise_amplitudes or the "
            "params.other_params.desi_galaxy_shot_noise_amplitude_pz{1..4} values. "
            "Sampled/saved amplitudes and a fixed wrapper override cannot coexist."
        )
    if has_explicit_wrapper_value:
        return raw["desi_galaxy_shot_noise_amplitudes"]
    if sampled_or_saved:
        return {
            pz_bin: sampled_or_saved.get(pz_bin, 1.0)
            for pz_bin in range(1, 5)
        }
    return 1.0


def theory_data_vector(config: Mapping[str, object], theory_cls: Mapping[str, np.ndarray], ell: np.ndarray):
    ensure_godmax_import_paths(Path(config["repo_root"]))
    from multiprobe_namaster import theory_to_data_vector

    raw = config["raw"].get("theory_to_data_vector", {})
    metadata = config.get("metadata", {})
    other_params = config.get("params", {}).get("other_params", {})
    shear_m_bias = config["metadata"]["shear_m_bias_means"]
    if bool(metadata.get("sampled_des_shear_m_bias_in_model", False)) or bool(
        other_params.get("sampled_des_shear_m_bias_in_model", False)
    ):
        shear_m_bias = None
    return theory_to_data_vector(
        config["paths"]["measurement_h5"],
        theory_cls,
        ell=np.asarray(ell, dtype=np.float64),
        shear_m_bias=shear_m_bias,
        ksz_velocity_correlation=float(raw.get("ksz_velocity_correlation", 0.3)),
        desi_galaxy_shot_noise_amplitudes=desi_galaxy_shot_noise_amplitudes_from_config(config),
        include_default_pixel_windows=bool(raw.get("include_default_pixel_windows", True)),
        include_default_act_beams=bool(raw.get("include_default_act_beams", True)),
        theory_shear_e_is_positive_kappa=bool(raw.get("theory_shear_e_is_positive_kappa", True)),
        allow_legacy_product=bool(raw.get("allow_legacy_product", False)),
    )


def comparison_statistics(measurement: MeasurementData, theory_vector: np.ndarray) -> dict:
    if theory_vector.shape != measurement.data_vector.shape:
        raise ValueError(
            f"Theory vector shape {theory_vector.shape} does not match data shape {measurement.data_vector.shape}."
        )
    valid = (
        np.ones(measurement.data_vector.size, dtype=bool)
        if measurement.data_vector_valid is None
        else np.asarray(measurement.data_vector_valid, dtype=bool)
    )
    if valid.shape != measurement.data_vector.shape:
        raise ValueError("Measurement validity mask does not match its data-vector shape.")
    resid_full = measurement.data_vector - theory_vector
    resid = resid_full[valid]
    covariance = measurement.covariance[np.ix_(valid, valid)]
    try:
        alpha = np.linalg.solve(covariance, resid)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(covariance) @ resid
    stats = {
        "full": {
            "chi2": float(resid @ alpha),
            "ndof": int(resid.size),
        },
        "families": {},
    }
    for family in sorted(set(measurement.families.values())):
        idx_chunks = []
        for name, start, stop in zip(measurement.names, measurement.starts, measurement.stops):
            if measurement.families[name] == family:
                indices = np.arange(start, stop, dtype=int)
                idx_chunks.append(indices[valid[indices]])
        if not idx_chunks:
            continue
        idx = np.concatenate(idx_chunks)
        sub_resid = resid_full[idx]
        sub_cov = measurement.covariance[np.ix_(idx, idx)]
        try:
            sub_alpha = np.linalg.solve(sub_cov, sub_resid)
        except np.linalg.LinAlgError:
            sub_alpha = np.linalg.pinv(sub_cov) @ sub_resid
        stats["families"][family] = {
            "chi2": float(sub_resid @ sub_alpha),
            "ndof": int(idx.size),
        }
    return stats


def dell_factor(ell: np.ndarray) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) / (2.0 * math.pi)


def measurement_ell_slice(measurement: MeasurementData, start: int, stop: int) -> np.ndarray:
    ell = np.asarray(measurement.ell, dtype=np.float64)
    n_band = int(stop) - int(start)
    if ell.size == n_band:
        return ell
    if ell.size == measurement.data_vector.size:
        return ell[int(start) : int(stop)]
    if ell.size >= n_band:
        return ell[:n_band]
    raise ValueError(f"Cannot match ell array of length {ell.size} to band count {n_band}.")


def measurement_ell_edge_slice(
    measurement: MeasurementData,
    start: int,
    stop: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if measurement.ell_left is None or measurement.ell_right is None:
        return None, None
    left = np.asarray(measurement.ell_left, dtype=np.float64)
    right = np.asarray(measurement.ell_right, dtype=np.float64)
    n_band = int(stop) - int(start)
    if left.size == n_band and right.size == n_band:
        return left, right
    if left.size == measurement.data_vector.size and right.size == measurement.data_vector.size:
        return left[int(start) : int(stop)], right[int(start) : int(stop)]
    if left.size >= n_band and right.size >= n_band:
        return left[:n_band], right[:n_band]
    return None, None


def _safe_family_png(output_dir, filename_prefix: str, family: str, max_basename: int = 200):
    """Per-family PNG path whose basename stays under the filesystem limit (~255).
    No-op for normal-length names; for very long prefixes it truncates the prefix
    and appends a short hash of the full prefix to keep names unique."""
    base = f"{filename_prefix}_{family}.png"
    if len(base) <= max_basename:
        return output_dir / base
    import hashlib
    h = hashlib.md5(filename_prefix.encode("utf-8")).hexdigest()[:8]
    keep = max(max_basename - len(family) - len(h) - len(".png") - 2, 8)
    return output_dir / f"{filename_prefix[:keep]}{h}_{family}.png"


def _configure_ell_axis(
    ax,
    ell: np.ndarray,
    *,
    ell_left: Optional[np.ndarray],
    xscale: str,
    ell_max: Optional[float],
    xlim: Optional[Tuple[float, float]],
) -> None:
    """Apply an ell-axis scale and limits using positive saved band support."""

    scale = str(xscale)
    if scale != "linear":
        ax.set_xscale(scale)

    if xlim is not None:
        left, right = float(xlim[0]), float(xlim[1])
        if not np.isfinite(left) or not np.isfinite(right) or left >= right:
            raise ValueError(f"xlim must be finite and increasing, got {xlim!r}.")
        if scale == "log" and left <= 0.0:
            raise ValueError(f"A logarithmic ell axis requires xlim[0] > 0, got {left!r}.")
        ax.set_xlim(left, right)
    elif ell_max is not None:
        right = float(ell_max)
        if not np.isfinite(right) or (scale == "log" and right <= 0.0):
            raise ValueError(
                f"ell_max must be finite and positive for a logarithmic ell axis, got {ell_max!r}."
            )
        if scale == "log":
            support = np.asarray(
                ell if ell_left is None else ell_left,
                dtype=np.float64,
            )
            positive = support[np.isfinite(support) & (support > 0.0)]
            if positive.size == 0:
                raise ValueError(
                    "A logarithmic ell axis requires at least one finite positive band support value."
                )
            left = float(np.min(positive))
            if left >= right:
                raise ValueError(
                    f"Logarithmic ell limits must be increasing, got {(left, right)!r}."
                )
            ax.set_xlim(left, right)
        else:
            ax.set_xlim(right=right)
    elif scale == "log":
        support = np.asarray(
            ell if ell_left is None else ell_left,
            dtype=np.float64,
        )
        positive = support[np.isfinite(support) & (support > 0.0)]
        if positive.size == 0:
            raise ValueError(
                "A logarithmic ell axis requires at least one finite positive band support value."
            )
        ax.set_xlim(left=float(np.min(positive)))

    if scale == "log":
        left, right = (float(value) for value in ax.get_xlim())
        if not np.isfinite(left) or not np.isfinite(right) or left <= 0.0 or left >= right:
            raise ValueError(
                "A logarithmic ell axis did not resolve to finite, positive, increasing limits: "
                f"{(left, right)!r}."
            )


def validate_measurement_plot_family_coverage(measurement: MeasurementData) -> None:
    """Refuse plots that would silently omit or misclassify a saved spectrum."""

    missing_metadata = [name for name in measurement.names if name not in measurement.families]
    if missing_metadata:
        raise ValueError(f"Measurement spectra lack family metadata: {missing_metadata}")
    actual = {measurement.families[name] for name in measurement.names}
    allowed = set(MEASUREMENT_FAMILY_ORDER)
    unknown = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    if unknown or missing:
        raise ValueError(
            "Measurement family coverage is incomplete or unknown: "
            f"missing={missing}, unknown={unknown}."
        )


def _shade_transfer_null_region(ax, measurement: MeasurementData, family: str) -> None:
    cutoff = measurement.transfer_null_from.get(family)
    if cutoff is None:
        return
    if measurement.ell_right is not None and np.asarray(measurement.ell_right).size:
        right = float(np.max(measurement.ell_right))
    else:
        right = float(np.max(measurement.ell))
    if not np.isfinite(cutoff) or not np.isfinite(right) or cutoff >= right:
        return
    ax.axvspan(
        float(cutoff),
        right,
        color="#c9b36a",
        alpha=0.20,
        lw=0,
        zorder=0,
        label=r"ACT $\kappa$ transfer = 0",
    )


def plot_family_comparisons(
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                data_cl = measurement.data_vector[start:stop]
                theory_cl = theory_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                if family == "desi_g_auto":
                    y_data = data_cl
                    y_theory = theory_cl
                    y_err = err
                    ylabel = r"$C_\ell$ (signal + shot noise)"
                else:
                    fac = dell_factor(ell)
                    sign = -1.0 if family == "desi_pi_act_T" else 1.0
                    scale = 1.0e3 if family == "desi_pi_act_T" else 1.0
                    y_data = sign * scale * fac * data_cl
                    y_theory = sign * scale * fac * theory_cl
                    y_err = scale * fac * err
                    ylabel = r"$D_\ell$"
                    if family == "desi_pi_act_T":
                        ylabel = r"$-10^3 D_\ell^{\pi T}$"
                ax.errorbar(ell, y_data, yerr=y_err, fmt="o", ms=3.2, lw=1.0, color=colors.get(family, "#333333"), label="measurement")
                ax.plot(ell, y_theory, "-", lw=1.6, color="#111111", label="GODMAX windowed")
                ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.55)
                if family == "desi_g_auto" and np.all(y_data > 0.0) and np.all(y_theory > 0.0):
                    ax.set_yscale("log")
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: measurement vs GODMAX"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            fig.suptitle(title, fontsize=13)
            out = output_dir / f"fast1024_godmax_comparison_{family}.png"
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def plot_family_dell_comparisons(
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: str = "godmax_dell_comparison",
    ell_max: Optional[float] = None,
    ksz_ylim: Optional[Tuple[float, float]] = None,
    ksz_scale: float = 1.0e3,
    active_band_indices: Optional[Mapping[str, Sequence[int]]] = None,
    total_reduced_chi2: Optional[float] = None,
    chi2_dof: Optional[int] = None,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if theory_vector.shape != measurement.data_vector.shape:
        raise ValueError(
            f"Theory vector shape {theory_vector.shape} does not match data shape {measurement.data_vector.shape}."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                ell_left, ell_right = measurement_ell_edge_slice(measurement, start, stop)
                data_cl = measurement.data_vector[start:stop]
                theory_cl = theory_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                local_index = np.arange(ell.size, dtype=int)
                if ell_max is not None:
                    keep = ell <= float(ell_max)
                    local_index = local_index[keep]
                    ell = ell[keep]
                    if ell_left is not None and ell_right is not None:
                        ell_left = ell_left[keep]
                        ell_right = ell_right[keep]
                    data_cl = data_cl[keep]
                    theory_cl = theory_cl[keep]
                    err = err[keep]
                if active_band_indices is not None and name in active_band_indices:
                    active = set(int(x) for x in active_band_indices[name])
                    excluded = np.asarray([int(i) not in active for i in local_index], dtype=bool)
                    inactive_spans = (
                        [(float(lo), float(hi)) for lo, hi in zip(ell_left[excluded], ell_right[excluded])]
                        if np.any(excluded) and ell_left is not None and ell_right is not None
                        else []
                    )
                else:
                    inactive_spans = []
                fac = dell_factor(ell)
                sign = -1.0 if family == "desi_pi_act_T" else 1.0
                scale = float(ksz_scale) if family == "desi_pi_act_T" else 1.0
                y_data = sign * scale * fac * data_cl
                y_theory = sign * scale * fac * theory_cl
                y_err = scale * fac * err
                ylabel = r"$D_\ell$"
                if family == "desi_pi_act_T":
                    ylabel = r"$-D_\ell^{\pi T}$" if np.isclose(scale, 1.0) else r"$-10^3 D_\ell^{\pi T}$"
                if inactive_spans:
                    first_inactive = True
                    for lo, hi in inactive_spans:
                        ax.fill_between(
                            [lo, hi],
                            [0.0, 0.0],
                            [1.0, 1.0],
                            transform=ax.get_xaxis_transform(),
                            color="#b8bcc5",
                            alpha=0.28,
                            lw=0,
                            zorder=0,
                            label="not in likelihood" if first_inactive else None,
                        )
                        first_inactive = False
                ax.errorbar(ell, y_data, yerr=y_err, fmt="o", ms=3.2, lw=1.0, color=colors.get(family, "#333333"), label="data", zorder=2)
                ax.plot(ell, y_theory, "-", lw=1.6, color="#111111", label="bestfit theory", zorder=3)
                ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.55)
                if family == "desi_pi_act_T" and ksz_ylim is not None:
                    ax.set_ylim(float(ksz_ylim[0]), float(ksz_ylim[1]))
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                _configure_ell_axis(
                    ax,
                    ell,
                    ell_left=ell_left,
                    xscale=xscale,
                    ell_max=ell_max,
                    xlim=xlim,
                )
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: data vs bestfit theory in D_ell"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            if total_reduced_chi2 is not None:
                title += "\n" + rf"best-fit reduced $\chi^2={float(total_reduced_chi2):.2f}$"
                if chi2_dof is not None:
                    title += f" ({int(chi2_dof)} dof)"
            fig.suptitle(title, fontsize=12)
            out = _safe_family_png(output_dir, filename_prefix, family)
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def plot_family_dell_residual_comparisons(
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: str = "godmax_dell_residual",
    ell_max: Optional[float] = None,
    ksz_scale: float = 1.0,
    active_band_indices: Optional[Mapping[str, Sequence[int]]] = None,
    total_reduced_chi2: Optional[float] = None,
    chi2_dof: Optional[int] = None,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if theory_vector.shape != measurement.data_vector.shape:
        raise ValueError(
            f"Theory vector shape {theory_vector.shape} does not match data shape {measurement.data_vector.shape}."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                ell_left, ell_right = measurement_ell_edge_slice(measurement, start, stop)
                data_cl = measurement.data_vector[start:stop]
                theory_cl = theory_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                local_index = np.arange(ell.size, dtype=int)
                if ell_max is not None:
                    keep = ell <= float(ell_max)
                    local_index = local_index[keep]
                    ell = ell[keep]
                    if ell_left is not None and ell_right is not None:
                        ell_left = ell_left[keep]
                        ell_right = ell_right[keep]
                    data_cl = data_cl[keep]
                    theory_cl = theory_cl[keep]
                    err = err[keep]
                if active_band_indices is not None and name in active_band_indices:
                    active = set(int(x) for x in active_band_indices[name])
                    excluded = np.asarray([int(i) not in active for i in local_index], dtype=bool)
                    inactive_spans = (
                        [(float(lo), float(hi)) for lo, hi in zip(ell_left[excluded], ell_right[excluded])]
                        if np.any(excluded) and ell_left is not None and ell_right is not None
                        else []
                    )
                else:
                    inactive_spans = []
                fac = dell_factor(ell)
                sign = -1.0 if family == "desi_pi_act_T" else 1.0
                scale = float(ksz_scale) if family == "desi_pi_act_T" else 1.0
                y_data = sign * scale * fac * data_cl
                y_theory = sign * scale * fac * theory_cl
                y_err = scale * fac * err
                residual = np.full_like(y_data, np.nan, dtype=np.float64)
                valid = np.isfinite(y_data) & np.isfinite(y_theory) & np.isfinite(y_err) & (y_err > 0.0)
                residual[valid] = (y_theory[valid] - y_data[valid]) / y_err[valid]
                if inactive_spans:
                    first_inactive = True
                    for lo, hi in inactive_spans:
                        ax.fill_between(
                            [lo, hi],
                            [0.0, 0.0],
                            [1.0, 1.0],
                            transform=ax.get_xaxis_transform(),
                            color="#b8bcc5",
                            alpha=0.28,
                            lw=0,
                            zorder=0,
                            label="not in likelihood" if first_inactive else None,
                        )
                        first_inactive = False
                ax.axhline(0.0, color="#555555", lw=0.8, alpha=0.75, zorder=1)
                ax.axhline(1.0, color="#4f4f4f", lw=1.6, ls="--", alpha=0.95, zorder=1, label=r"$\pm 1\sigma$")
                ax.axhline(-1.0, color="#4f4f4f", lw=1.6, ls="--", alpha=0.95, zorder=1)
                ax.plot(ell, residual, "o-", ms=3.2, lw=1.1, color=colors.get(family, "#333333"), label=r"$(\mathrm{bestfit}-\mathrm{data})/\sigma$", zorder=3)
                _configure_ell_axis(
                    ax,
                    ell,
                    ell_left=ell_left,
                    xscale=xscale,
                    ell_max=ell_max,
                    xlim=xlim,
                )
                if ylim is not None:
                    ax.set_ylim(float(ylim[0]), float(ylim[1]))
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(r"$(D_\ell^\mathrm{bf}-D_\ell^\mathrm{data})/\sigma(D_\ell)$")
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: bestfit residuals"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            if total_reduced_chi2 is not None:
                title += "\n" + rf"best-fit reduced $\chi^2={float(total_reduced_chi2):.2f}$"
                if chi2_dof is not None:
                    title += f" ({int(chi2_dof)} dof)"
            fig.suptitle(title, fontsize=12)
            out = _safe_family_png(output_dir, filename_prefix, family)
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def measurement_plot_values(
    ell: np.ndarray,
    cl: np.ndarray,
    err: np.ndarray,
    *,
    family: str,
    quantity: str,
    ksz_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return plotting values without changing the saved measurement convention.

    The HDF5 product always stores raw ``C_ell`` values, including raw
    ``C_ell^{pi,T}`` for kSZ.  The paper-style kSZ sign flip is therefore a
    display convention applied only to ``D_ell`` plots.  Means and one-sigma
    errors receive the same positive ``C_ell -> D_ell`` scale factor.
    """

    quantity = str(quantity).lower()
    if quantity not in {"cl", "dell"}:
        raise ValueError(f"quantity must be 'cl' or 'dell', got {quantity!r}.")
    ell = np.asarray(ell, dtype=np.float64)
    cl = np.asarray(cl, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    if ell.shape != cl.shape or cl.shape != err.shape:
        raise ValueError("ell, cl and err must have identical shapes.")

    is_ksz = family == "desi_pi_act_T"
    scale = float(ksz_scale) if is_ksz else 1.0
    if is_ksz and (not np.isfinite(scale) or scale <= 0.0):
        raise ValueError(f"ksz_scale must be finite and positive, got {ksz_scale!r}.")
    if np.isclose(scale, 1.0):
        scale_prefix = ""
    else:
        exponent = int(round(math.log10(scale)))
        if np.isclose(scale, 10.0**exponent):
            scale_prefix = rf"10^{{{exponent}}}\,"
        else:
            scale_prefix = rf"{scale:g}\,"
    if quantity == "cl":
        factor = np.ones_like(ell)
        sign = 1.0
        ylabel = r"$C_\ell$"
        if is_ksz:
            ylabel = rf"${scale_prefix}C_\ell^{{\pi T}}$"
    else:
        factor = dell_factor(ell)
        sign = -1.0 if is_ksz else 1.0
        ylabel = r"$D_\ell$"
        if is_ksz:
            ylabel = rf"$-{scale_prefix}D_\ell^{{\pi T}}$"
    return sign * scale * factor * cl, abs(scale) * factor * err, ylabel


def plot_measurement_bandpowers(
    measurement: MeasurementData,
    output_dir: str | Path,
    *,
    quantity: str,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: Optional[str] = None,
    ell_max: Optional[float] = None,
    ksz_ylim: Optional[Tuple[float, float]] = None,
    ksz_scale: float = 1.0,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    quantity = str(quantity).lower()
    if quantity not in {"cl", "dell"}:
        raise ValueError(f"quantity must be 'cl' or 'dell', got {quantity!r}.")
    if filename_prefix is None:
        filename_prefix = f"measurement_{quantity}"
    validate_measurement_plot_family_coverage(measurement)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in MEASUREMENT_FAMILY_ORDER:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                ell_left, ell_right = measurement_ell_edge_slice(
                    measurement,
                    start,
                    stop,
                )
                data_cl = measurement.data_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                if ell_max is not None:
                    keep = ell <= float(ell_max)
                    ell = ell[keep]
                    if ell_left is not None and ell_right is not None:
                        ell_left = ell_left[keep]
                        ell_right = ell_right[keep]
                    data_cl = data_cl[keep]
                    err = err[keep]
                y_data, y_err, ylabel = measurement_plot_values(
                    ell,
                    data_cl,
                    err,
                    family=family,
                    quantity=quantity,
                    ksz_scale=ksz_scale,
                )
                if family == "desi_g_auto":
                    if measurement.galaxy_auto_view == "total":
                        ylabel = (
                            r"$C_\ell^{gg}+N_\ell^{\rm shot}$"
                            if quantity == "cl"
                            else r"$\ell(\ell+1)(C_\ell^{gg}+N_\ell^{\rm shot})/(2\pi)$"
                        )
                    else:
                        ylabel = (
                            r"$C_\ell^{gg}\;(\widehat N_\ell^{\rm P}\ {\rm subtracted})$"
                            if quantity == "cl"
                            else r"$D_\ell^{gg}\;(\widehat N_\ell^{\rm P}\ {\rm subtracted})$"
                        )
                _shade_transfer_null_region(ax, measurement, family)
                ax.errorbar(ell, y_data, yerr=y_err, fmt="o", ms=3.2, lw=1.0, color=colors.get(family, "#333333"), label="measurement")
                ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.55)
                if family == "desi_pi_act_T" and ksz_ylim is not None:
                    ax.set_ylim(float(ksz_ylim[0]), float(ksz_ylim[1]))
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                _configure_ell_axis(
                    ax,
                    ell,
                    ell_left=ell_left,
                    xscale=xscale,
                    ell_max=ell_max,
                    xlim=xlim,
                )
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            quantity_title = r"$C_\ell$" if quantity == "cl" else r"$D_\ell$"
            title = f"{family}: measurement in {quantity_title}"
            if family == "desi_g_auto":
                if measurement.galaxy_auto_view == "total":
                    title += " (total clustering + weighted-Poisson shot noise)"
                else:
                    title += " (weighted-Poisson template subtracted)"
            if family == "desi_pi_act_T" and quantity == "dell":
                title += " (positive kSZ convention)"
            if family in measurement.transfer_null_from:
                title += rf" (shaded: transfer null from $\ell={measurement.transfer_null_from[family]:g}$)"
            fig.suptitle(title, fontsize=13)
            out = _safe_family_png(output_dir, filename_prefix, family)
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def plot_measurement_dell(
    measurement: MeasurementData,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: str = "measurement_dell",
    ell_max: Optional[float] = None,
    ksz_ylim: Optional[Tuple[float, float]] = None,
    ksz_scale: float = 1.0,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    return plot_measurement_bandpowers(
        measurement,
        output_dir,
        quantity="dell",
        pdf_path=pdf_path,
        filename_prefix=filename_prefix,
        ell_max=ell_max,
        ksz_ylim=ksz_ylim,
        ksz_scale=ksz_scale,
        xscale=xscale,
        xlim=xlim,
    )


def plot_measurement_cl(
    measurement: MeasurementData,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: str = "measurement_cl",
    ell_max: Optional[float] = None,
    ksz_scale: float = 1.0,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    return plot_measurement_bandpowers(
        measurement,
        output_dir,
        quantity="cl",
        pdf_path=pdf_path,
        filename_prefix=filename_prefix,
        ell_max=ell_max,
        ksz_ylim=None,
        ksz_scale=ksz_scale,
        xscale=xscale,
        xlim=xlim,
    )


def save_outputs(
    config: Mapping[str, object],
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    theory_names: Sequence[str],
    theory_cls: Mapping[str, np.ndarray],
    ell_theory: np.ndarray,
    stats: Mapping[str, object],
    tau_diagnostics: Optional[Mapping[str, np.ndarray]] = None,
) -> dict:
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "theory_data_vector_fast1024.npz"
    measurement_identity = measurement_identity_sha256(measurement)
    comparison_config_identity = comparison_config_identity_sha256(config)
    theory_response_identity = theory_response_identity_sha256(config)
    vector_cache_fields = theory_vector_cache_fields(
        theory_vector,
        measurement_identity,
        {
            "product_kind": "configured_theory_vector",
            "config_path": str(config["paths"]["config"]),
            "comparison_config_identity_sha256": comparison_config_identity,
            "theory_response_identity_sha256": theory_response_identity,
            "theory_names": list(theory_names),
        },
    )
    np.savez_compressed(
        npz_path,
        ell_band=measurement.ell,
        data_vector=measurement.data_vector,
        theory_vector=theory_vector,
        covariance=measurement.covariance,
        spectrum_names=np.asarray(measurement.names),
        slice_start=np.asarray(measurement.starts, dtype=np.int64),
        slice_stop=np.asarray(measurement.stops, dtype=np.int64),
        measurement_identity_sha256=np.asarray(measurement_identity),
        theory_response_identity_sha256=np.asarray(theory_response_identity),
        theory_names=np.asarray(list(theory_names)),
        ell_theory=np.asarray(ell_theory, dtype=np.float64),
        theory_cls_keys=np.asarray(sorted(theory_cls)),
        ksz_default_A_v_by_pz=np.asarray(config["metadata"]["ksz_default_A_v_by_pz"], dtype=np.float64),
        **vector_cache_fields,
    )
    summary_path = output_dir / "comparison_summary_fast1024.json"
    summary = {
        "config_path": config["paths"]["config"],
        "measurement_path": measurement.path,
        "map_path": config["paths"]["map_h5"],
        "npz_path": npz_path,
        "measurement_identity_sha256": measurement_identity,
        "comparison_config_identity_sha256": comparison_config_identity,
        "theory_response_identity_sha256": theory_response_identity,
        "stats": stats,
        "validation": validation_summary(config),
    }
    if tau_diagnostics is not None:
        ratio = np.asarray(tau_diagnostics["gal_tau_effective_over_zdependent"], dtype=np.float64)
        summary["tau_effective_z_over_zdependent_median_by_pz"] = np.nanmedian(ratio, axis=0)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_serializable(summary), handle, indent=2)
    return {"npz": npz_path, "summary": summary_path}
