"""Measure matched BaryonForge--GODMAX map statistics on one common cap.

The two input products must contain RING-ordered HEALPix ``map_ymap`` and
``map_kappa_cmb`` arrays.  This driver deliberately applies one mask, one
mean-subtraction convention, one NaMaster workspace, and one bandpower
definition to both backends.  It measures deterministic code differences;
it does not attach a Gaussian cosmic-variance covariance to them.

Radial stacks are intentionally outside this first measurement product.  The
output records that omission explicitly so that a spectra-only product cannot
be mistaken for the full comparison requested by the project charter.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import healpy as hp
import numpy as np
import pymaster as nmt


THIS_DIR = Path(__file__).resolve().parent
XDESI_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    MAP_PRODUCT_SCHEMA,
    assert_map_contract_unchanged,
    cap_mask,
    current_map_contract,
    load_config,
    load_config_and_freeze_map_contract,
    read_map_file,
    resolve_path,
    sha256_file,
    sha256_json,
)


SCHEMA = "baryonforge_godmax_common_mask_statistics_v1"
MAP_KEYS = ("map_ymap", "map_kappa_cmb")
QUANTILE_PROBABILITIES = np.asarray(
    [0.0, 0.1, 1.0, 5.0, 16.0, 50.0, 84.0, 95.0, 99.0, 99.9, 100.0],
    dtype=np.float64,
)

# All fields are scalar and share one mask.  ``residual`` always means
# BaryonForge minus GODMAX.
SPECTRUM_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("godmax_yy", "godmax_y", "godmax_y"),
    ("godmax_kk", "godmax_kappa", "godmax_kappa"),
    ("godmax_yk", "godmax_y", "godmax_kappa"),
    ("baryonforge_yy", "baryonforge_y", "baryonforge_y"),
    ("baryonforge_kk", "baryonforge_kappa", "baryonforge_kappa"),
    ("baryonforge_yk", "baryonforge_y", "baryonforge_kappa"),
    ("cross_backend_yy", "godmax_y", "baryonforge_y"),
    ("cross_backend_kk", "godmax_kappa", "baryonforge_kappa"),
    ("godmax_y_baryonforge_k", "godmax_y", "baryonforge_kappa"),
    ("baryonforge_y_godmax_k", "baryonforge_y", "godmax_kappa"),
    ("residual_yy", "residual_y", "residual_y"),
    ("residual_kk", "residual_kappa", "residual_kappa"),
    ("residual_yk", "residual_y", "residual_kappa"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=True)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}.")
    return value


def _read_required_maps(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    maps, attrs = read_map_file(path)
    maps = dict(_require_mapping(maps, f"maps in {path}"))
    attrs = dict(_require_mapping(attrs, f"attributes in {path}"))
    nested_provenance = attrs.get("provenance")
    if isinstance(nested_provenance, Mapping):
        # Native and adapter products may place the same provenance keys at
        # the root or in a dedicated group.  Preserve the nested record while
        # exposing missing root keys to the strict shared-object checks below.
        for key, value in nested_provenance.items():
            attrs.setdefault(str(key), value)
    missing = [key for key in MAP_KEYS if key not in maps]
    if missing:
        raise KeyError(f"{path} is missing required map dataset(s): {missing}")

    out: Dict[str, np.ndarray] = {}
    for key in MAP_KEYS:
        values = np.asarray(maps[key])
        if values.ndim != 1:
            raise ValueError(f"{path}:{key} must be one-dimensional, got shape {values.shape}.")
        finite = np.isfinite(values)
        if not bool(np.all(finite)):
            raise ValueError(
                f"{path}:{key} contains {int(values.size - np.count_nonzero(finite))} non-finite pixels."
            )
        out[key] = values
    return out, attrs


def _infer_and_validate_nside(
    godmax_maps: Mapping[str, np.ndarray],
    baryonforge_maps: Mapping[str, np.ndarray],
    godmax_attrs: Mapping[str, Any],
    baryonforge_attrs: Mapping[str, Any],
    expected_nside: Optional[int],
) -> int:
    lengths = {
        "godmax_y": int(np.asarray(godmax_maps["map_ymap"]).size),
        "godmax_kappa": int(np.asarray(godmax_maps["map_kappa_cmb"]).size),
        "baryonforge_y": int(np.asarray(baryonforge_maps["map_ymap"]).size),
        "baryonforge_kappa": int(np.asarray(baryonforge_maps["map_kappa_cmb"]).size),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input map lengths differ: {lengths}")
    npix = next(iter(lengths.values()))
    try:
        inferred = int(hp.npix2nside(npix))
    except ValueError as exc:
        raise ValueError(f"Map length {npix} is not a valid full-sky HEALPix size.") from exc

    if expected_nside is not None and inferred != int(expected_nside):
        raise ValueError(f"Maps imply NSIDE={inferred}, but configuration/CLI requests {expected_nside}.")
    for backend, attrs in (("GODMAX", godmax_attrs), ("BaryonForge", baryonforge_attrs)):
        if "nside" in attrs and int(attrs["nside"]) != inferred:
            raise ValueError(f"{backend} metadata NSIDE={attrs['nside']} does not match map NSIDE={inferred}.")
        ordering = attrs.get("ordering")
        if ordering is not None and str(ordering).upper() != "RING":
            raise ValueError(f"{backend} map ordering must be RING, got {ordering!r}.")
    return inferred


def _scalar_equal(left: Any, right: Any) -> bool:
    left = _jsonable(left)
    right = _jsonable(right)
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        if isinstance(left, bool) or isinstance(right, bool):
            return bool(left) == bool(right)
        return bool(np.isclose(float(left), float(right), rtol=1.0e-12, atol=0.0, equal_nan=True))
    return left == right


def _check_shared_provenance(
    godmax_attrs: Mapping[str, Any],
    baryonforge_attrs: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Require both native products to identify the same comparison object."""

    required_shared_keys = (
        "schema",
        "comparison_config_path",
        "comparison_config_sha256",
        "godmax_params_path",
        "godmax_params_sha256",
        "baryonforge_params_path",
        "baryonforge_params_sha256",
        "effective_godmax_config_sha256",
        "source_manifest_sha256",
        "godmax_git_sha",
        "baryonforge_git_sha",
        "godmax_git_dirty",
        "baryonforge_git_dirty",
        "runtime_versions",
        "smoke_table",
        "max_halos",
        "baryonforge_splitjoin_n_jobs",
        "godmax_pixel_workers",
        "catalog_sha256",
        "catalog_path",
        "selection_predicate",
        "halo_count",
        "n_halos_painted",
        "complete_catalog_paint",
        "nside",
        "ordering",
        "max_paint_R200c_factor",
        "smooth_profiles",
        "halo_only",
        "z_min",
        "z_max",
        "h",
        "H0",
        "Omega_M",
        "Omega_b",
        "map_semantics",
        "noise_policy",
        "mass_proxy_semantics",
        "provisional_status",
        "provisional_reasons",
        "analysis_mask_policy",
        "cmb_source_redshift",
    )
    optional_shared_keys = (
        "catalog_selection_sha256",
        "mass_cut_predicate",
    )
    shared_keys = required_shared_keys + optional_shared_keys
    report: Dict[str, Dict[str, Any]] = {}
    mismatches = []
    for key in shared_keys:
        has_g = key in godmax_attrs
        has_b = key in baryonforge_attrs
        if has_g and has_b:
            equal = _scalar_equal(godmax_attrs[key], baryonforge_attrs[key])
            status = "equal" if equal else "mismatch"
            if not equal:
                mismatches.append(key)
        else:
            status = "missing_one" if has_g or has_b else "missing_both"
            if key in required_shared_keys:
                mismatches.append(f"{key}.{status}")
        report[key] = {
            "status": status,
            "godmax": _jsonable(godmax_attrs.get(key)),
            "baryonforge": _jsonable(baryonforge_attrs.get(key)),
        }

    for backend, attrs in (("GODMAX", godmax_attrs), ("BaryonForge", baryonforge_attrs)):
        if "smooth_profiles" in attrs and bool(attrs["smooth_profiles"]):
            mismatches.append(f"{backend}.smooth_profiles=true")
        expected_backend = backend.lower()
        if attrs.get("backend") != expected_backend:
            mismatches.append(
                f"{backend}.backend={attrs.get('backend')!r}, expected {expected_backend!r}"
            )
        if attrs.get("schema") != MAP_PRODUCT_SCHEMA:
            mismatches.append(
                f"{backend}.schema={attrs.get('schema')!r}, expected {MAP_PRODUCT_SCHEMA!r}"
            )
        if attrs.get("complete_catalog_paint") is not True:
            mismatches.append(f"{backend}.complete_catalog_paint is not true")
        if attrs.get("smoke_table") is not False:
            mismatches.append(f"{backend}.smoke_table is not false")
        if attrs.get("max_halos") is not None:
            mismatches.append(f"{backend}.max_halos is not null")
        if attrs.get("n_halos_painted") != attrs.get("halo_count"):
            mismatches.append(f"{backend}.n_halos_painted != halo_count")
        if backend == "GODMAX" and (
            attrs.get("split_index") != 0 or attrs.get("num_splits") != 1
        ):
            mismatches.append("GODMAX map is not the complete one-split product")
        if backend == "BaryonForge" and attrs.get("n_jobs") != attrs.get(
            "baryonforge_splitjoin_n_jobs"
        ):
            mismatches.append(
                "BaryonForge n_jobs differs from the configured SplitJoin geometry"
            )

        unit_boundary = attrs.get("unit_boundary")
        required_unit_keys = {
            "catalog_mass",
            "catalog_radius",
            "catalog_distance",
            "map_ymap",
            "map_kappa_cmb",
        }
        if not isinstance(unit_boundary, Mapping):
            mismatches.append(f"{backend}.unit_boundary missing")
        else:
            missing_units = sorted(required_unit_keys.difference(unit_boundary))
            if missing_units:
                mismatches.append(f"{backend}.unit_boundary missing {missing_units}")
            if unit_boundary.get("map_ymap") != "dimensionless Compton-y":
                mismatches.append(f"{backend}.unit_boundary.map_ymap invalid")
            if (
                unit_boundary.get("map_kappa_cmb")
                != "dimensionless halo-only CMB convergence"
            ):
                mismatches.append(f"{backend}.unit_boundary.map_kappa_cmb invalid")

        source_manifest = attrs.get("source_manifest")
        if not isinstance(source_manifest, Mapping):
            mismatches.append(f"{backend}.source_manifest missing")
        elif sha256_json(source_manifest) != attrs.get("source_manifest_sha256"):
            mismatches.append(f"{backend}.source_manifest_sha256 does not match manifest")

        effective_manifest = attrs.get("effective_godmax_config_manifest")
        if not isinstance(effective_manifest, Mapping):
            mismatches.append(f"{backend}.effective_godmax_config_manifest missing")
        else:
            manifest_without_digest = dict(effective_manifest)
            embedded_digest = manifest_without_digest.pop("sha256", None)
            recomputed_digest = sha256_json(manifest_without_digest)
            if embedded_digest != recomputed_digest:
                mismatches.append(
                    f"{backend}.effective_godmax_config_manifest embedded digest invalid"
                )
            if recomputed_digest != attrs.get("effective_godmax_config_sha256"):
                mismatches.append(
                    f"{backend}.effective_godmax_config_sha256 does not match manifest"
                )

    if expected_contract is not None:
        for backend, attrs in (
            ("GODMAX", godmax_attrs),
            ("BaryonForge", baryonforge_attrs),
        ):
            for key, expected in expected_contract.items():
                if key in {"source_manifest", "effective_godmax_config_manifest"}:
                    continue
                if key not in attrs or not _scalar_equal(attrs[key], expected):
                    mismatches.append(f"{backend}.{key} differs from current config/source")
    if mismatches:
        raise ValueError(
            "The two map products do not describe the same comparison object: " + ", ".join(mismatches)
        )
    return report


def _current_expected_map_contract(config: Mapping[str, Any]) -> dict:
    """Rebuild the intended shared contract before accepting production maps."""

    return current_map_contract(config)


def _strict_integer_edges(raw_edges: np.ndarray, ell_min: int, lmax: int, n_bins: int) -> np.ndarray:
    edges = np.asarray(raw_edges, dtype=np.int64).copy()
    if edges.size != int(n_bins) + 1:
        raise ValueError(f"Expected {n_bins + 1} band edges, got {edges.size}.")
    edges[0] = int(ell_min)
    edges[-1] = int(lmax) + 1
    for index in range(1, edges.size):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1
    if edges[-1] != int(lmax) + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"Cannot form {n_bins} non-empty integer bands over inclusive ell range [{ell_min}, {lmax}]."
        )
    return edges


def _make_bins(ell_min: int, lmax: int, n_bins: int, binning: str) -> Tuple[nmt.NmtBin, np.ndarray, np.ndarray]:
    if ell_min < 0 or lmax < ell_min:
        raise ValueError(f"Invalid ell range [{ell_min}, {lmax}].")
    if n_bins <= 0 or n_bins > lmax - ell_min + 1:
        raise ValueError(f"n_bins={n_bins} is incompatible with ell range [{ell_min}, {lmax}].")
    mode = str(binning).lower()
    if mode == "linear":
        raw = np.ceil(np.linspace(ell_min, lmax + 1, n_bins + 1)).astype(np.int64)
    elif mode == "sqrt":
        raw = np.rint(np.linspace(math.sqrt(ell_min), math.sqrt(lmax), n_bins + 1) ** 2).astype(np.int64)
    elif mode == "log":
        if ell_min <= 0:
            raise ValueError("Logarithmic ell binning requires ell_min > 0.")
        raw = np.rint(np.geomspace(ell_min, lmax + 1, n_bins + 1)).astype(np.int64)
    else:
        raise ValueError(f"Unsupported statistics.binning={binning!r}; expected linear, sqrt, or log.")
    edges = _strict_integer_edges(raw, ell_min, lmax, n_bins)
    left = edges[:-1].astype(np.int32)
    right = edges[1:].astype(np.int32)
    return nmt.NmtBin.from_edges(left, right), left, right


def _masked_map(values: np.ndarray, mask: np.ndarray, subtract_weighted_mean: bool) -> Tuple[np.ndarray, float]:
    positive = np.asarray(mask) > 0.0
    if not bool(np.any(positive)):
        raise ValueError("Analysis mask has no positive pixels.")
    weights = np.asarray(mask[positive], dtype=np.float64)
    selected = np.asarray(values[positive], dtype=np.float64)
    mean = float(np.sum(weights * selected, dtype=np.float64) / np.sum(weights, dtype=np.float64))
    out = np.zeros(np.asarray(values).shape, dtype=np.float32)
    if subtract_weighted_mean:
        out[positive] = np.asarray(selected - mean, dtype=np.float32)
    else:
        out[positive] = np.asarray(selected, dtype=np.float32)
        mean = 0.0
    return out, mean


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(numerator).shape, np.nan, dtype=np.float64)
    out[valid] = np.asarray(numerator, dtype=np.float64)[valid] / np.asarray(denominator, dtype=np.float64)[valid]
    return out


def _field_summary(values: np.ndarray, pixel_area_sr: float, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Map-summary input must be a non-empty one-dimensional array.")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("Map-summary input contains non-finite values.")
    if weights is None:
        w = np.ones(arr.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != arr.shape or np.any(~np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("Summary weights must be finite, non-negative, and match the map shape.")
    wsum = float(np.sum(w, dtype=np.float64))
    if wsum <= 0.0:
        raise ValueError("Summary weights sum to zero.")
    mean = float(np.sum(w * arr, dtype=np.float64) / wsum)
    rms = float(np.sqrt(np.sum(w * arr**2, dtype=np.float64) / wsum))
    std = float(np.sqrt(np.sum(w * (arr - mean) ** 2, dtype=np.float64) / wsum))
    return {
        "n_pixels": int(arr.size),
        "weight_sum": wsum,
        "mean": mean,
        "std": std,
        "rms": rms,
        "l1_mean": float(np.sum(w * np.abs(arr), dtype=np.float64) / wsum),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
        "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
        "sum_unweighted": float(np.sum(arr, dtype=np.float64)),
        "integral_sr_unweighted": float(np.sum(arr, dtype=np.float64) * pixel_area_sr),
        "quantile_probability_percent": QUANTILE_PROBABILITIES.copy(),
        "quantile": np.percentile(arr, QUANTILE_PROBABILITIES),
    }


def _pair_summary(reference: np.ndarray, candidate: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    test = np.asarray(candidate, dtype=np.float64)
    if ref.shape != test.shape or ref.ndim != 1 or ref.size == 0:
        raise ValueError("Pair-summary arrays must be non-empty, one-dimensional, and shape-matched.")
    if weights is None:
        w = np.ones(ref.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != ref.shape or np.any(~np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("Pair-summary weights must be finite, non-negative, and shape-matched.")
    wsum = float(np.sum(w, dtype=np.float64))
    if wsum <= 0.0:
        raise ValueError("Pair-summary weights sum to zero.")
    delta = test - ref
    mean_ref = float(np.sum(w * ref, dtype=np.float64) / wsum)
    mean_test = float(np.sum(w * test, dtype=np.float64) / wsum)
    ref_centered = ref - mean_ref
    test_centered = test - mean_test
    ref_var = float(np.sum(w * ref_centered**2, dtype=np.float64))
    test_var = float(np.sum(w * test_centered**2, dtype=np.float64))
    covariance = float(np.sum(w * ref_centered * test_centered, dtype=np.float64))
    ref_norm2 = float(np.sum(w * ref**2, dtype=np.float64))
    test_norm2 = float(np.sum(w * test**2, dtype=np.float64))
    dot = float(np.sum(w * ref * test, dtype=np.float64))
    ref_l1 = float(np.sum(w * np.abs(ref), dtype=np.float64) / wsum)
    ref_rms = float(np.sqrt(ref_norm2 / wsum))
    mae = float(np.sum(w * np.abs(delta), dtype=np.float64) / wsum)
    rmse = float(np.sqrt(np.sum(w * delta**2, dtype=np.float64) / wsum))
    ref_sum = float(np.sum(w * ref, dtype=np.float64))
    test_sum = float(np.sum(w * test, dtype=np.float64))
    return {
        "n_pixels": int(ref.size),
        "weight_sum": wsum,
        "reference": "GODMAX",
        "candidate": "BaryonForge",
        "difference_convention": "BaryonForge minus GODMAX",
        "reference_mean": mean_ref,
        "candidate_mean": mean_test,
        "difference_mean": float(np.sum(w * delta, dtype=np.float64) / wsum),
        "mae": mae,
        "rmse": rmse,
        "relative_l1_to_godmax": float(mae / ref_l1) if ref_l1 > 0.0 else float("nan"),
        "relative_rmse_to_godmax": float(rmse / ref_rms) if ref_rms > 0.0 else float("nan"),
        "pearson_r": float(covariance / math.sqrt(ref_var * test_var))
        if ref_var > 0.0 and test_var > 0.0
        else float("nan"),
        "cosine_similarity": float(dot / math.sqrt(ref_norm2 * test_norm2))
        if ref_norm2 > 0.0 and test_norm2 > 0.0
        else float("nan"),
        "gain_through_origin": float(dot / ref_norm2) if ref_norm2 > 0.0 else float("nan"),
        "weighted_sum_ratio": float(test_sum / ref_sum) if ref_sum != 0.0 else float("nan"),
        "difference_quantile_probability_percent": QUANTILE_PROBABILITIES.copy(),
        "difference_quantile": np.percentile(delta, QUANTILE_PROBABILITIES),
    }


def _write_metric_group(parent: h5py.Group, name: str, values: Mapping[str, Any]) -> h5py.Group:
    group = parent.create_group(name)
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, compression="lzf")
        elif isinstance(value, (str, bytes, bool, int, float, np.generic)):
            group.attrs[key] = _jsonable(value)
        else:
            group.attrs[f"{key}_json"] = _json_dumps(value)
    return group


def _diagnostics_from_spectra(spectra: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for field, prefix in (("y", "yy"), ("kappa", "kk")):
        gm = np.asarray(spectra[f"godmax_{prefix}"]["cl"], dtype=np.float64)
        bf = np.asarray(spectra[f"baryonforge_{prefix}"]["cl"], dtype=np.float64)
        cross = np.asarray(spectra[f"cross_backend_{prefix}"]["cl"], dtype=np.float64)
        residual = np.asarray(spectra[f"residual_{prefix}"]["cl"], dtype=np.float64)
        gm_nonzero = np.isfinite(gm) & (gm != 0.0)
        positive_autos = np.isfinite(gm) & np.isfinite(bf) & (gm > 0.0) & (bf > 0.0)
        amplitude = np.full(gm.shape, np.nan, dtype=np.float64)
        amplitude[positive_autos] = np.sqrt(bf[positive_autos] / gm[positive_autos])
        coherence = np.full(gm.shape, np.nan, dtype=np.float64)
        coherence[positive_autos] = cross[positive_autos] / np.sqrt(gm[positive_autos] * bf[positive_autos])
        out[field] = {
            "gain_cross_over_godmax_auto": _safe_ratio(cross, gm, gm_nonzero),
            "amplitude_sqrt_auto_ratio": amplitude,
            "coherence": coherence,
            "residual_fraction_of_godmax_auto": _safe_ratio(residual, gm, gm_nonzero),
            "valid_gain_and_residual": gm_nonzero.astype(np.uint8),
            "valid_amplitude_and_coherence": positive_autos.astype(np.uint8),
        }

    gm_yk = np.asarray(spectra["godmax_yk"]["cl"], dtype=np.float64)
    bf_yk = np.asarray(spectra["baryonforge_yk"]["cl"], dtype=np.float64)
    valid_yk = np.isfinite(gm_yk) & (gm_yk != 0.0)
    out["yk"] = {
        "baryonforge_over_godmax": _safe_ratio(bf_yk, gm_yk, valid_yk),
        "valid": valid_yk.astype(np.uint8),
    }
    return out


def _closure_diagnostics(spectra: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
    expected_yy = (
        spectra["baryonforge_yy"]["cl"]
        + spectra["godmax_yy"]["cl"]
        - 2.0 * spectra["cross_backend_yy"]["cl"]
    )
    expected_kk = (
        spectra["baryonforge_kk"]["cl"]
        + spectra["godmax_kk"]["cl"]
        - 2.0 * spectra["cross_backend_kk"]["cl"]
    )
    expected_yk = (
        spectra["baryonforge_yk"]["cl"]
        - spectra["baryonforge_y_godmax_k"]["cl"]
        - spectra["godmax_y_baryonforge_k"]["cl"]
        + spectra["godmax_yk"]["cl"]
    )
    definitions = {
        "yy": (
            np.asarray(spectra["residual_yy"]["cl"], dtype=np.float64),
            np.asarray(expected_yy, dtype=np.float64),
            "C[(BF_y-GM_y),(BF_y-GM_y)] = C[BF_y,BF_y] + C[GM_y,GM_y] - 2 C[GM_y,BF_y]",
        ),
        "kk": (
            np.asarray(spectra["residual_kk"]["cl"], dtype=np.float64),
            np.asarray(expected_kk, dtype=np.float64),
            "C[(BF_k-GM_k),(BF_k-GM_k)] = C[BF_k,BF_k] + C[GM_k,GM_k] - 2 C[GM_k,BF_k]",
        ),
        "yk": (
            np.asarray(spectra["residual_yk"]["cl"], dtype=np.float64),
            np.asarray(expected_yk, dtype=np.float64),
            "C[(BF_y-GM_y),(BF_k-GM_k)] = C[BF_y,BF_k] - C[BF_y,GM_k] - C[GM_y,BF_k] + C[GM_y,GM_k]",
        ),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, (direct, expected, formula) in definitions.items():
        difference = direct - expected
        scale = float(max(np.max(np.abs(direct)), np.max(np.abs(expected)))) if direct.size else 0.0
        max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        out[name] = {
            "formula": formula,
            "direct": direct,
            "expected_from_component_spectra": expected,
            "difference": difference,
            "max_abs_difference": max_abs,
            "rms_difference": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
            "reference_scale_max_abs": scale,
            "max_abs_difference_over_reference_scale": float(max_abs / scale)
            if scale > 0.0
            else (0.0 if max_abs == 0.0 else float("inf")),
            "pass_fail_policy": "diagnostic_only; no exploratory tolerance is promoted to a gate",
        }
    return out


def _write_product(
    output: Path,
    *,
    overwrite: bool,
    metadata: Mapping[str, Any],
    binary_mask: np.ndarray,
    apodized_mask: np.ndarray,
    ell: np.ndarray,
    ell_left: np.ndarray,
    ell_right: np.ndarray,
    bandpower_window: np.ndarray,
    spectra: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Mapping[str, np.ndarray]],
    closure: Mapping[str, Mapping[str, Any]],
    map_summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pair_summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    null_metadata: Mapping[str, Any],
    pre_publish: Callable[[], None] | None = None,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with h5py.File(tmp, "w", track_order=True) as h5:
            h5.attrs["schema"] = SCHEMA
            h5.attrs["created_utc"] = utc_now()
            h5.attrs["metadata_json"] = _json_dumps(metadata)
            h5.attrs["ordering"] = "RING"
            h5.attrs["noise_policy"] = str(metadata["noise_policy"])
            h5.attrs["provisional_status"] = str(metadata["provisional_status"])
            h5.attrs["comparison_config_sha256"] = str(
                metadata["comparison_config_sha256"]
            )
            h5.attrs["input_source_manifest_sha256"] = str(
                metadata["input_source_manifest_sha256"]
            )
            h5.attrs["radial_stacks_included"] = False
            h5.attrs["radial_stacks_note"] = (
                "This product contains common-mask map summaries and NaMaster spectra only. "
                "Equal-halo-weighted radial stacks must be produced by a separate, explicitly labelled stage."
            )

            h5.create_dataset("ell", data=np.asarray(ell, dtype=np.float64))
            h5.create_dataset("ell_left", data=np.asarray(ell_left, dtype=np.int32))
            h5.create_dataset("ell_right", data=np.asarray(ell_right, dtype=np.int32))
            h5.create_dataset("bandpower_window", data=np.asarray(bandpower_window, dtype=np.float64), compression="lzf")

            mask_group = h5.create_group("mask")
            binary_pixels = np.flatnonzero(np.asarray(binary_mask) > 0.0).astype(np.int64)
            apodized_pixels = np.flatnonzero(np.asarray(apodized_mask) > 0.0).astype(np.int64)
            mask_group.create_dataset("binary_pixel_index", data=binary_pixels, compression="lzf")
            mask_group.create_dataset("apodized_pixel_index", data=apodized_pixels, compression="lzf")
            mask_group.create_dataset(
                "apodized_weight",
                data=np.asarray(apodized_mask[apodized_pixels], dtype=np.float32),
                compression="lzf",
            )
            mask_group.attrs["binary_weights_are_unity"] = True
            mask_group.attrs["fsky_binary"] = float(np.mean(binary_mask))
            mask_group.attrs["fsky_apodized_weight"] = float(np.mean(apodized_mask))
            mask_group.attrs["fsky_apodized_weight_squared"] = float(np.mean(np.asarray(apodized_mask) ** 2))

            spectra_group = h5.create_group("spectra")
            for name, result in spectra.items():
                group = spectra_group.create_group(name)
                group.attrs["field_a"] = str(result["fields"][0])
                group.attrs["field_b"] = str(result["fields"][1])
                group.attrs["spin_a"] = 0
                group.attrs["spin_b"] = 0
                group.attrs["bandpower_window_ref"] = "/bandpower_window"
                group.attrs["pixel_window_deconvolved"] = False
                group.create_dataset("cl", data=np.asarray(result["cl"], dtype=np.float64))
                group.create_dataset("dell", data=np.asarray(result["dell"], dtype=np.float64))
                group.create_dataset("pcl", data=np.asarray(result["pcl"], dtype=np.float64), compression="lzf")

            diagnostic_group = h5.create_group("diagnostics")
            for name, values in diagnostics.items():
                group = diagnostic_group.create_group(name)
                for key, value in values.items():
                    group.create_dataset(key, data=np.asarray(value), compression="lzf")

            null_group = h5.create_group("null_tests")
            null_group.attrs["metadata_json"] = _json_dumps(null_metadata)
            closure_group = null_group.create_group("linear_residual_closure")
            for name, values in closure.items():
                group = closure_group.create_group(name)
                for key, value in values.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value, compression="lzf")
                    else:
                        group.attrs[key] = _jsonable(value)

            summary_group = h5.create_group("map_summaries")
            for field_name, variants in map_summaries.items():
                field_group = summary_group.create_group(field_name)
                for variant, values in variants.items():
                    _write_metric_group(field_group, variant, values)

            pair_group = h5.create_group("pair_summaries")
            for field_name, variants in pair_summaries.items():
                field_group = pair_group.create_group(field_name)
                for variant, values in variants.items():
                    _write_metric_group(field_group, variant, values)

            stacks_group = h5.create_group("radial_stacks")
            stacks_group.attrs["included"] = False
            stacks_group.attrs["reason"] = str(h5.attrs["radial_stacks_note"])
        if pre_publish is not None:
            pre_publish()
        os.replace(tmp, output)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def measure(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    allow_synthetic_provenance = bool(
        getattr(args, "_allow_synthetic_provenance", False)
    )
    if allow_synthetic_provenance:
        config = dict(load_config(config_path))
        frozen_contract = None
        frozen_config_sha256 = sha256_file(config_path)
    else:
        config, frozen_contract = load_config_and_freeze_map_contract(config_path)
        frozen_config_sha256 = frozen_contract["comparison_config_sha256"]
    stats = dict(_require_mapping(config.get("statistics", {}), "statistics"))
    sky = dict(_require_mapping(config.get("sky_patch", {}), "sky_patch"))

    godmax_path = Path(resolve_path(args.godmax_maps, config_path))
    baryonforge_path = Path(resolve_path(args.baryonforge_maps, config_path))
    output = Path(resolve_path(args.output, config_path))
    if output in {godmax_path, baryonforge_path}:
        raise ValueError("Statistics output must differ from both input map paths.")
    for path in (godmax_path, baryonforge_path):
        if not path.exists():
            raise FileNotFoundError(path)

    frozen_input_hashes = {
        "config": frozen_config_sha256,
        "godmax_map": sha256_file(godmax_path),
        "baryonforge_map": sha256_file(baryonforge_path),
    }

    def assert_statistics_inputs_unchanged(context: str) -> None:
        current_hashes = {
            "config": sha256_file(config_path),
            "godmax_map": sha256_file(godmax_path),
            "baryonforge_map": sha256_file(baryonforge_path),
        }
        changed = [
            key
            for key, expected in frozen_input_hashes.items()
            if current_hashes[key] != expected
        ]
        if changed:
            raise RuntimeError(
                f"{context}: statistics inputs changed after their hashes were frozen; "
                f"refusing to publish. Changed files: {changed}"
            )
        if frozen_contract is not None:
            assert_map_contract_unchanged(
                frozen_contract,
                current_map_contract(config),
                context=context,
            )

    godmax_maps, godmax_attrs = _read_required_maps(godmax_path)
    baryonforge_maps, baryonforge_attrs = _read_required_maps(baryonforge_path)
    assert_statistics_inputs_unchanged("Statistics post-input-read validation")

    configured_nside = args.nside
    if configured_nside is None:
        configured_nside = stats.get("nside", config.get("pasting", {}).get("nside"))
    nside = _infer_and_validate_nside(
        godmax_maps,
        baryonforge_maps,
        godmax_attrs,
        baryonforge_attrs,
        None if configured_nside is None else int(configured_nside),
    )
    expected_contract = frozen_contract
    provenance_report = _check_shared_provenance(
        godmax_attrs,
        baryonforge_attrs,
        expected_contract=expected_contract,
    )

    try:
        center_ra = float(sky["center_ra_deg"])
        center_dec = float(sky["center_dec_deg"])
        radius_deg = float(sky["radius_deg"])
    except KeyError as exc:
        raise KeyError(f"sky_patch is missing required cap geometry key {exc.args[0]!r}.") from exc
    configured_ordering = str(sky.get("ordering", "RING")).upper()
    if configured_ordering != "RING":
        raise ValueError(f"sky_patch.ordering must be RING for this comparison, got {configured_ordering!r}.")
    binary_mask = np.asarray(cap_mask(nside, center_ra, center_dec, radius_deg), dtype=np.float64)
    if binary_mask.shape != (hp.nside2npix(nside),):
        raise ValueError(f"cap_mask returned shape {binary_mask.shape}, expected {(hp.nside2npix(nside),)}.")
    if np.any(~np.isfinite(binary_mask)) or np.any(binary_mask < 0.0) or not np.any(binary_mask > 0.0):
        raise ValueError("cap_mask returned a non-finite, negative, or empty mask.")
    binary_mask = (binary_mask > 0.0).astype(np.float64)

    apodization_deg = float(stats.get("apodization_deg", 0.0))
    apodization_type = str(stats.get("apodization_type", "C2"))
    if apodization_deg < 0.0:
        raise ValueError("statistics.apodization_deg must be non-negative.")
    if apodization_deg > 0.0:
        pixel_resolution_deg = float(np.degrees(hp.nside2resol(nside)))
        if apodization_deg < pixel_resolution_deg:
            raise ValueError(
                "statistics.apodization_deg is smaller than the characteristic HEALPix pixel size "
                f"({apodization_deg:g} < {pixel_resolution_deg:g} deg at NSIDE={nside}). "
                "NaMaster cannot resolve that apodization scale safely."
            )
        if apodization_type not in {"C1", "C2", "Smooth"}:
            raise ValueError("statistics.apodization_type must be C1, C2, or Smooth.")
        apodized_mask = np.asarray(
            nmt.mask_apodization(binary_mask, apodization_deg, apotype=apodization_type),
            dtype=np.float64,
        )
    else:
        apodized_mask = binary_mask.copy()
    if np.any(~np.isfinite(apodized_mask)) or np.any(apodized_mask < 0.0) or not np.any(apodized_mask > 0.0):
        raise ValueError("Apodization produced a non-finite, negative, or empty mask.")

    deconvolve_pixel_window = bool(stats.get("deconvolve_pixel_window", False))
    if deconvolve_pixel_window:
        raise ValueError(
            "statistics.deconvolve_pixel_window=true is unsupported here. "
            "The matched comparison keeps the common HEALPix pixel window in both backends."
        )
    subtract_weighted_mean = bool(stats.get("subtract_weighted_mean", True))
    ell_min = int(stats.get("ell_min", 8))
    lmax = int(stats.get("lmax", min(nside, 3 * nside - 1)))
    n_bins = int(stats.get("n_bins", 10))
    binning = str(stats.get("binning", "linear"))
    if lmax > 3 * nside - 1:
        raise ValueError(f"statistics.lmax={lmax} exceeds the HEALPix limit {3 * nside - 1}.")
    bins, ell_left, ell_right = _make_bins(ell_min, lmax, n_bins, binning)

    binary_pixels = np.flatnonzero(binary_mask > 0.0)
    apodized_pixels = np.flatnonzero(apodized_mask > 0.0)
    pixel_area_sr = float(hp.nside2pixarea(nside))
    raw_selected = {
        "godmax_y": np.asarray(godmax_maps["map_ymap"][binary_pixels], dtype=np.float64),
        "godmax_kappa": np.asarray(godmax_maps["map_kappa_cmb"][binary_pixels], dtype=np.float64),
        "baryonforge_y": np.asarray(baryonforge_maps["map_ymap"][binary_pixels], dtype=np.float64),
        "baryonforge_kappa": np.asarray(baryonforge_maps["map_kappa_cmb"][binary_pixels], dtype=np.float64),
    }
    require_nonzero = bool(
        config.get("validation", {}).get("require_nonzero_production_maps", True)
    )
    if require_nonzero and not allow_synthetic_provenance:
        empty_fields = [
            name for name, values in raw_selected.items() if np.count_nonzero(values) == 0
        ]
        if empty_fields:
            raise ValueError(
                "Production map fields are identically zero on the binary analysis cap: "
                + ", ".join(empty_fields)
            )
    raw_selected["residual_y"] = raw_selected["baryonforge_y"] - raw_selected["godmax_y"]
    raw_selected["residual_kappa"] = raw_selected["baryonforge_kappa"] - raw_selected["godmax_kappa"]

    processed_maps: Dict[str, np.ndarray] = {}
    subtracted_means: Dict[str, float] = {}
    for field_name, values in (
        ("godmax_y", godmax_maps["map_ymap"]),
        ("godmax_kappa", godmax_maps["map_kappa_cmb"]),
        ("baryonforge_y", baryonforge_maps["map_ymap"]),
        ("baryonforge_kappa", baryonforge_maps["map_kappa_cmb"]),
    ):
        processed_maps[field_name], subtracted_means[field_name] = _masked_map(
            values, apodized_mask, subtract_weighted_mean
        )
    processed_maps["residual_y"] = processed_maps["baryonforge_y"] - processed_maps["godmax_y"]
    processed_maps["residual_kappa"] = processed_maps["baryonforge_kappa"] - processed_maps["godmax_kappa"]
    subtracted_means["residual_y"] = subtracted_means["baryonforge_y"] - subtracted_means["godmax_y"]
    subtracted_means["residual_kappa"] = (
        subtracted_means["baryonforge_kappa"] - subtracted_means["godmax_kappa"]
    )

    processed_selected = {
        name: np.asarray(values[apodized_pixels], dtype=np.float64) for name, values in processed_maps.items()
    }
    apodized_weights = np.asarray(apodized_mask[apodized_pixels], dtype=np.float64)

    map_summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in raw_selected:
        map_summaries[name] = {
            "raw_binary_cap": _field_summary(raw_selected[name], pixel_area_sr),
            "processed_apodized_cap": _field_summary(
                processed_selected[name], pixel_area_sr, weights=apodized_weights
            ),
        }
        map_summaries[name]["processed_apodized_cap"]["weighted_mean_subtracted"] = subtract_weighted_mean
        map_summaries[name]["processed_apodized_cap"]["subtracted_mean"] = subtracted_means[name]

    pair_summaries = {
        "y": {
            "raw_binary_cap": _pair_summary(raw_selected["godmax_y"], raw_selected["baryonforge_y"]),
            "processed_apodized_cap": _pair_summary(
                processed_selected["godmax_y"], processed_selected["baryonforge_y"], apodized_weights
            ),
        },
        "kappa": {
            "raw_binary_cap": _pair_summary(
                raw_selected["godmax_kappa"], raw_selected["baryonforge_kappa"]
            ),
            "processed_apodized_cap": _pair_summary(
                processed_selected["godmax_kappa"],
                processed_selected["baryonforge_kappa"],
                apodized_weights,
            ),
        },
    }

    fields: Dict[str, nmt.NmtField] = {}
    for name, values in processed_maps.items():
        fields[name] = nmt.NmtField(
            apodized_mask,
            [values],
            spin=0,
            n_iter=int(stats.get("n_iter", 0)),
            n_iter_mask=int(stats.get("n_iter_mask", 0)),
            lmax=lmax,
            lmax_mask=lmax,
            lite=True,
        )

    workspace = nmt.NmtWorkspace.from_fields(fields["godmax_y"], fields["godmax_y"], bins)
    window = np.asarray(workspace.get_bandpower_windows()[0, :, 0, :], dtype=np.float64)
    ell = np.asarray(bins.get_effective_ells(), dtype=np.float64)
    spectra: Dict[str, Dict[str, Any]] = {}
    for name, field_a, field_b in SPECTRUM_SPECS:
        pcl = np.asarray(nmt.compute_coupled_cell(fields[field_a], fields[field_b])[0], dtype=np.float64)
        cl = np.asarray(workspace.decouple_cell(pcl[None, :])[0], dtype=np.float64)
        spectra[name] = {
            "fields": (field_a, field_b),
            "pcl": pcl,
            "cl": cl,
            "dell": ell * (ell + 1.0) * cl / (2.0 * np.pi),
        }

    diagnostics = _diagnostics_from_spectra(spectra)
    closure = _closure_diagnostics(spectra)
    input_equal_y = bool(np.array_equal(raw_selected["godmax_y"], raw_selected["baryonforge_y"]))
    input_equal_kappa = bool(
        np.array_equal(raw_selected["godmax_kappa"], raw_selected["baryonforge_kappa"])
    )
    null_metadata = {
        "residual_definition": "BaryonForge minus GODMAX",
        "identical_input_y_applicable": input_equal_y,
        "identical_input_kappa_applicable": input_equal_kappa,
        "godmax_y_zero_on_binary_cap": bool(np.count_nonzero(raw_selected["godmax_y"]) == 0),
        "godmax_kappa_zero_on_binary_cap": bool(np.count_nonzero(raw_selected["godmax_kappa"]) == 0),
        "baryonforge_y_zero_on_binary_cap": bool(np.count_nonzero(raw_selected["baryonforge_y"]) == 0),
        "baryonforge_kappa_zero_on_binary_cap": bool(
            np.count_nonzero(raw_selected["baryonforge_kappa"]) == 0
        ),
        "identical_input_y_max_abs_residual_cl": float(np.max(np.abs(spectra["residual_yy"]["cl"])))
        if input_equal_y
        else None,
        "identical_input_kappa_max_abs_residual_cl": float(
            np.max(np.abs(spectra["residual_kk"]["cl"]))
        )
        if input_equal_kappa
        else None,
        "linear_closure_is_diagnostic_not_tolerance_gate": True,
    }

    metadata = {
        "config_path": str(config_path),
        "config_sha256": frozen_input_hashes["config"],
        "godmax_map_path": str(godmax_path),
        "godmax_map_sha256": frozen_input_hashes["godmax_map"],
        "baryonforge_map_path": str(baryonforge_path),
        "baryonforge_map_sha256": frozen_input_hashes["baryonforge_map"],
        "input_hashes_frozen_before_read": True,
        "godmax_map_attrs": _jsonable(godmax_attrs),
        "baryonforge_map_attrs": _jsonable(baryonforge_attrs),
        "shared_provenance_check": provenance_report,
        "comparison_config_sha256": godmax_attrs["comparison_config_sha256"],
        "input_source_manifest_sha256": godmax_attrs["source_manifest_sha256"],
        "noise_policy": godmax_attrs["noise_policy"],
        "provisional_status": godmax_attrs["provisional_status"],
        "provisional_reasons": godmax_attrs["provisional_reasons"],
        "nside": nside,
        "ordering": "RING",
        "map_units": {"y": "dimensionless Compton-y", "kappa": "dimensionless halo-only CMB convergence"},
        "mask": {
            "kind": "inner angular cap",
            "center_ra_deg": center_ra,
            "center_dec_deg": center_dec,
            "radius_deg": radius_deg,
            "n_binary_pixels": int(binary_pixels.size),
            "area_deg2": float(binary_pixels.size * hp.nside2pixarea(nside, degrees=True)),
            "apodization_deg": apodization_deg,
            "apodization_type": apodization_type,
        },
        "spectra": {
            "ell_min": ell_min,
            "lmax": lmax,
            "n_bins": n_bins,
            "binning": binning,
            "subtract_weighted_mean": subtract_weighted_mean,
            "subtracted_means": subtracted_means,
            "deconvolve_pixel_window": False,
            "compute_covariance": False,
            "covariance_note": (
                "Both maps are deterministic transforms of one halo catalog. No Gaussian cosmic-variance "
                "covariance is assigned to the backend difference."
            ),
            "workspace_note": "One spin-0 workspace is reused for all fields because every field has the same mask.",
            "spectrum_names": [name for name, _, _ in SPECTRUM_SPECS],
        },
        "radial_stacks": {
            "included": False,
            "reason": "Omitted from the first common-mask statistics driver; requires a separate equal-halo-weighted implementation.",
        },
    }

    _write_product(
        output,
        overwrite=bool(args.overwrite),
        metadata=metadata,
        binary_mask=binary_mask,
        apodized_mask=apodized_mask,
        ell=ell,
        ell_left=ell_left,
        ell_right=ell_right,
        bandpower_window=window,
        spectra=spectra,
        diagnostics=diagnostics,
        closure=closure,
        map_summaries=map_summaries,
        pair_summaries=pair_summaries,
        null_metadata=null_metadata,
        pre_publish=lambda: assert_statistics_inputs_unchanged(
            "Statistics pre-publication validation"
        ),
    )

    # Release full-sky arrays before returning the small command summary.
    del fields, processed_maps, godmax_maps, baryonforge_maps
    gc.collect()
    return {
        "output": str(output),
        "schema": SCHEMA,
        "nside": nside,
        "n_binary_cap_pixels": int(binary_pixels.size),
        "cap_area_deg2": metadata["mask"]["area_deg2"],
        "n_spectra": len(SPECTRUM_SPECS),
        "spectra": [name for name, _, _ in SPECTRUM_SPECS],
        "radial_stacks_included": False,
        "closure_max_abs_difference": {
            name: float(values["max_abs_difference"]) for name, values in closure.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Matched comparison YAML.")
    parser.add_argument("--godmax-maps", required=True, help="GODMAX HDF5 map product.")
    parser.add_argument("--baryonforge-maps", required=True, help="BaryonForge HDF5 map product.")
    parser.add_argument("--output", required=True, help="Output common-mask statistics HDF5.")
    parser.add_argument("--nside", type=int, default=None, help="Optional strict NSIDE assertion.")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace an existing output file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the final JSON summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = measure(args)
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
