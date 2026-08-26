"""Fail-closed contract helpers for the three-probe Abacus mock.

The catalog mass is an explicitly provisional particle-count proxy.  These
helpers prevent it, its cosmology, or its strict redshift support from being
silently replaced by defaults when constructing validation theory products.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

import h5py
import numpy as np


CATALOG_COSMOLOGY_KEYS = ("H0", "Omega_M", "Omega_b", "sigma8", "ns", "w0", "h")
SOURCE_COSMOLOGY_KEYS = (
    "H0",
    "Omega_M",
    "Omega_DE",
    "Omega_K",
    "CAMB_Omega_b",
    "CAMB_sigma8",
    "CAMB_ns",
    "w0",
)


@dataclass(frozen=True)
class ResolvedSupport:
    z_min: float
    z_max: float
    mass_min_hmsun: float
    mass_max_hmsun: float


def sha256_file(path: pathlib.Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def require_true(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not True:
        raise ValueError(f"Contract key {key!r} must be literally true")


def _finite_float(mapping: Mapping[str, Any], key: str, context: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing required {context} key {key!r}")
    value = float(mapping[key])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {context} key {key!r}: {value}")
    return value


def canonical_cosmology(
    catalog_attrs: Mapping[str, Any], source_header: Mapping[str, Any]
) -> dict[str, float | bool]:
    """Return the exact GODMAX cosmology after cross-checking source and catalog."""

    catalog = {key: _finite_float(catalog_attrs, key, "catalog cosmology") for key in CATALOG_COSMOLOGY_KEYS}
    source = {key: _finite_float(source_header, key, "source cosmology") for key in SOURCE_COSMOLOGY_KEYS}
    comparisons = {
        "H0": (catalog["H0"], source["H0"]),
        "Omega_M": (catalog["Omega_M"], source["Omega_M"]),
        "Omega_b": (catalog["Omega_b"], source["CAMB_Omega_b"]),
        "sigma8": (catalog["sigma8"], source["CAMB_sigma8"]),
        "ns": (catalog["ns"], source["CAMB_ns"]),
        "w0": (catalog["w0"], source["w0"]),
    }
    for name, (left, right) in comparisons.items():
        if left != right:
            raise ValueError(f"Catalog/source cosmology mismatch for {name}: {left!r} != {right!r}")
    if catalog["H0"] != 100.0 * catalog["h"]:
        raise ValueError("Catalog H0 is not exactly 100*h")
    if source["Omega_K"] != 0.0:
        raise ValueError(f"GODMAX validation requires a flat source cosmology, Omega_K={source['Omega_K']}")
    if not np.isclose(source["Omega_DE"], 1.0 - source["Omega_M"], rtol=0.0, atol=2e-8):
        raise ValueError("Source Omega_DE is inconsistent with flatness")
    return {
        "flat": True,
        "H0": catalog["H0"],
        "Om0": catalog["Omega_M"],
        "Ob0": catalog["Omega_b"],
        "sigma8": catalog["sigma8"],
        "ns": catalog["ns"],
        "w0": catalog["w0"],
    }


def validate_catalog_contract(
    catalog_path: pathlib.Path,
    theory_config: Mapping[str, Any],
    source_header: Mapping[str, Any],
    *,
    verify_file_sha: bool,
) -> tuple[dict[str, Any], ResolvedSupport, dict[str, float | bool]]:
    """Validate identity, cosmology, units, and common resolved support."""

    require_true(theory_config, "override_cosmology_from_catalog")
    require_true(theory_config, "require_complete_source_cosmology")
    if theory_config.get("mode") != "map_matched_resolved":
        raise ValueError("resolved_theory.mode must be map_matched_resolved")
    if theory_config.get("unresolved_completion") is not False:
        raise ValueError("unresolved_completion must be literally false")

    expected_size = int(theory_config["catalog_file_size_bytes"])
    if catalog_path.stat().st_size != expected_size:
        raise ValueError("Catalog file size does not match the frozen contract")
    if verify_file_sha:
        observed_sha = sha256_file(catalog_path)
        if observed_sha != theory_config["catalog_file_sha256"]:
            raise ValueError("Catalog file SHA-256 does not match the frozen contract")

    with h5py.File(catalog_path, "r") as handle:
        attrs = {key: handle.attrs[key] for key in handle.attrs}
        required_datasets = {
            "z",
            "M200c_hMsun",
            "N_interp",
            "source_file_index",
            "source_row_index",
            "halo_timeslice_index",
            "ra_deg",
            "dec_deg",
        }
        missing = sorted(required_datasets.difference(handle.keys()))
        if missing:
            raise ValueError(f"Catalog is missing required datasets: {missing}")
        lengths = {int(handle[key].shape[0]) for key in required_datasets}
        if lengths != {int(attrs["n_halos"])}:
            raise ValueError(f"Catalog datasets do not share n_halos: {lengths}")

    source_files = json.loads(str(attrs["source_files_json"]))
    coverage = json.loads(str(attrs["source_coverage_report_json"]))
    if not source_files or coverage.get("status") != "passed":
        raise ValueError("Catalog source-shell coverage is absent or did not pass")
    source_dirs = [pathlib.Path(path).parent.name for path in source_files]
    if source_dirs != list(coverage.get("expected_source_dirs", [])):
        raise ValueError("Catalog source files disagree with the frozen shell-coverage report")
    for boundary in coverage.get("boundaries", []):
        if any(int(value) != 0 for value in boundary.get("overlap_counts", {}).values()):
            raise ValueError("A declared boundary-null shell contributes selected rows")

    identity_checks = {
        "row_identity_sha256": "catalog_row_identity_sha256",
        "selection_contract_sha256": "catalog_selection_contract_sha256",
        "source_content_manifest_sha256": "catalog_source_content_manifest_sha256",
    }
    for attr_key, config_key in identity_checks.items():
        if str(attrs.get(attr_key, "")) != str(theory_config[config_key]):
            raise ValueError(f"Catalog identity mismatch for {attr_key}")

    common = theory_config["common_support"]
    support = ResolvedSupport(
        z_min=float(common["z_min"]),
        z_max=float(common["z_max"]),
        mass_min_hmsun=float(common["mass_min_hmsun"]),
        mass_max_hmsun=float(common["mass_max_hmsun"]),
    )
    exact_pairs = {
        "z_min": support.z_min,
        "z_max": support.z_max,
        "mass_min_hmsun": support.mass_min_hmsun,
        "mass_max_hmsun": support.mass_max_hmsun,
    }
    for attr_key, expected in exact_pairs.items():
        if float(attrs[attr_key]) != expected:
            raise ValueError(f"Catalog/config support mismatch for {attr_key}")
    boolean_pairs = {
        "z_min_exclusive": True,
        "z_max_exclusive": True,
        "mass_min_inclusive": True,
        "mass_max_exclusive": True,
    }
    for attr_key, expected in boolean_pairs.items():
        if bool(attrs[attr_key]) is not expected or bool(common[attr_key]) is not expected:
            raise ValueError(f"Required selection convention is not frozen for {attr_key}")
    if str(attrs["mass_unit"]) != "Msun/h":
        raise ValueError(f"Expected catalog mass_unit='Msun/h', found {attrs['mass_unit']!r}")
    if "provisional" not in str(attrs["mass_definition_status"]):
        raise ValueError("The provisional mass-definition status must be explicit")

    cosmology = canonical_cosmology(attrs, source_header)
    return attrs, support, cosmology


def make_normalized_kernel(counts: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert fixed-bin counts into a boundary-zero, trapezoid-normalized dN/dz."""

    counts = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if counts.ndim != 1 or edges.shape != (counts.size + 1,):
        raise ValueError("Kernel counts/edges shape mismatch")
    if np.any(counts < 0) or not np.all(np.diff(edges) > 0) or counts.sum() <= 0:
        raise ValueError("Kernel counts and edges must be non-negative, ordered, and non-empty")
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths)
    z_grid = np.concatenate(([edges[0]], centers, [edges[-1]]))
    nz_grid = np.concatenate(([0.0], density, [0.0]))
    integral = float(np.trapz(nz_grid, z_grid))
    nz_grid /= integral
    integral = float(np.trapz(nz_grid, z_grid))
    return z_grid, nz_grid, integral


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
