#!/usr/bin/env python3
"""Prepare the native-replica 256-point paired gX/g2X mock-SBI product.

This revision uses a separately named joint-builder adapter whose returned
metadata includes both row-level and point-level proposal densities.

The round-2 HDF5 bundles store unmasked signal-plus-noise maps.  Both gX and g2X
are therefore remeasured from each same saved bundle with the current estimator;
the mask recorded by the earlier products.npz measurement is audit information,
not an admission gate for using the maps.

This is a new schema.  It deliberately does not reinterpret the earlier uniform
five-replica pilot files.

Workflow
--------
1. Build the original all-128, five-replica g2X child with the existing validated
   round-1 builder.
2. Run ``init-round2`` once to cache the frozen squared-galaxy harmonic field.
3. Run ``measure-round2`` over all 128 campaign points.  Both gX and g2X are
   measured from each exact saved ``noisy_XX.h5`` bundle using one shared masked
   harmonic transform per field.
4. Run ``assemble`` to replay the exact old gX noise seeds, join the two cohorts,
   and write one flat, paired master product.

The master has 128*5 + 128*10 = 1,920 native rows.  It uses CSR-like point row
offsets and a per-point replica-count vector; there is no global replica count or
rectangular point-by-replica table.  Old and new source snapshots remain labelled,
so any pooled inference is explicitly exploratory.
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys

_SBI_ROOT = _pl.Path(__file__).resolve().parents[1]
for _directory in (_SBI_ROOT, _SBI_ROOT / "common"):
    if str(_directory) not in _sys.path:
        _sys.path.insert(0, str(_directory))

import argparse
import hashlib
import json
import os
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

import build_mock_sbi_g2x_pilot_contract as g2contract
import build_mock_sbi_gx_g2x_pilot as joint_builder

import build_mock_sbi_g2x_trainingset as g2build

CACHE_SCHEMA = "godmax.mock_sbi.g2x_fixed_estimator_cache.v1"
SHARD_SCHEMA = "godmax.mock_sbi.round2_native_gx_g2x_shard.v2"
MASTER_SCHEMA = "godmax.mock_sbi.gx_g2x_native_heterogeneous.v3"
ROUND2_CAMPAIGN_NAME = "current_code_original_defaults_round2_20260911"
ROUND2_COMPLETION_AGGREGATE = (
    "973022ddd4f803060965e7ea539f356deadad3ea5f894a585c4735a305d636a4"
)
N_OLD_POINTS = 128
N_NEW_POINTS = 128
OLD_REPLICAS = 5
NEW_REPLICAS = 10
N_POINTS = N_OLD_POINTS + N_NEW_POINTS
N_ROWS = N_OLD_POINTS * OLD_REPLICAS + N_NEW_POINTS * NEW_REPLICAS
COVARIANCE_DOF = N_OLD_POINTS * (OLD_REPLICAS - 1) + N_NEW_POINTS * (
    NEW_REPLICAS - 1
)
N_VALIDATION_PER_COHORT = 13
PARAMETER_NAMES = g2build.PARAMETER_NAMES
PRIOR_LOW = g2contract.PRIOR_LOW
PRIOR_HIGH = g2contract.PRIOR_HIGH
GX_VECTOR_ORDER = "spectrum-major gy[14],gkappa[14],gtau[14]"
G2X_VECTOR_ORDER = g2build.VECTOR_ORDER
MAP_DATASETS = {
    "y": "maps/map_ymap",
    "tau": "maps/map_tau",
    "kappa": "maps/map_kappa_cmb",
}


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_savez(path: pathlib.Path, arrays: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _json_from_array(value: np.ndarray) -> dict[str, Any]:
    result = json.loads(str(np.asarray(value).item()))
    if not isinstance(result, dict):
        raise ValueError("manifest_json is not an object")
    return result


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _strings(value: np.ndarray) -> tuple[str, ...]:
    return tuple(_text(item) for item in np.asarray(value).reshape(-1))


def _canonical_seed_rows(
    seeds: np.ndarray,
    saved_order: Sequence[str],
    canonical_order: Sequence[str],
) -> np.ndarray:
    saved = tuple(str(name) for name in saved_order)
    canonical = tuple(str(name) for name in canonical_order)
    if len(saved) != len(set(saved)) or set(saved) != set(canonical):
        raise ValueError(f"seed field order {saved} is not a permutation of {canonical}")
    values = np.asarray(seeds, dtype=np.uint32)
    if values.ndim != 2 or values.shape[1] != len(saved):
        raise ValueError(f"noise seed table has invalid shape {values.shape}")
    return values[:, [saved.index(name) for name in canonical]]


def _replica_table(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 2):
        raise ValueError("replica_count must be one-dimensional and at least two")
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
    )
    point_index = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    replica = np.concatenate(
        [np.arange(count, dtype=np.int64) for count in counts]
    )
    return offsets, point_index, replica


def _validate_flat_rows(
    point_index: np.ndarray,
    replica: np.ndarray,
    replica_count: np.ndarray,
    row_offsets: np.ndarray,
) -> int:
    counts = np.asarray(replica_count, dtype=np.int64)
    expected_offsets, expected_points, expected_replicas = _replica_table(counts)
    if not np.array_equal(np.asarray(row_offsets, dtype=np.int64), expected_offsets):
        raise ValueError("point_row_offsets is not the cumulative replica_count table")
    if not np.array_equal(np.asarray(point_index, dtype=np.int64), expected_points):
        raise ValueError("point_index is not in canonical point-major order")
    if not np.array_equal(np.asarray(replica, dtype=np.int64), expected_replicas):
        raise ValueError("replica is not the canonical local 0..R_point-1 sequence")
    return int(np.sum(counts - 1))


def _assemble_proposal_density(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    point_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Join the two point-level proposal-density tables and expand to rows.

    The old joint-builder adapter must expose both the row and point versions
    stored by the round-1 g2X builder.  Keeping this check at the assembly
    boundary prevents a loader from silently dropping ``point_log_q`` again.
    """

    old_point = np.asarray(old["point_log_q"], dtype=np.float64)
    new_point = np.asarray(new["point_log_q"], dtype=np.float64)
    old_rows = np.asarray(old["log_q"], dtype=np.float64)
    old_point_index = np.asarray(old["point_index"], dtype=np.int64)
    combined_point_index = np.asarray(point_index, dtype=np.int64)

    if old_point.shape != (N_OLD_POINTS,):
        raise ValueError(
            f"old point_log_q shape {old_point.shape}; expected {(N_OLD_POINTS,)}"
        )
    if new_point.shape != (N_NEW_POINTS,):
        raise ValueError(
            f"new point_log_q shape {new_point.shape}; expected {(N_NEW_POINTS,)}"
        )
    if old_rows.shape != (N_OLD_POINTS * OLD_REPLICAS,):
        raise ValueError(
            f"old row log_q shape {old_rows.shape}; "
            f"expected {(N_OLD_POINTS * OLD_REPLICAS,)}"
        )
    if old_point_index.shape != (N_OLD_POINTS * OLD_REPLICAS,):
        raise ValueError(
            f"old point_index shape {old_point_index.shape}; "
            f"expected {(N_OLD_POINTS * OLD_REPLICAS,)}"
        )
    if np.any(old_point_index < 0) or np.any(old_point_index >= N_OLD_POINTS):
        raise ValueError("old point_index is outside 0..127")
    if not np.array_equal(
        old_rows, old_point[old_point_index], equal_nan=True
    ):
        raise ValueError("old row log_q does not match old point_log_q")
    if combined_point_index.shape != (N_ROWS,):
        raise ValueError(
            f"combined point_index shape {combined_point_index.shape}; "
            f"expected {(N_ROWS,)}"
        )
    if np.any(combined_point_index < 0) or np.any(combined_point_index >= N_POINTS):
        raise ValueError("combined point_index is outside 0..255")

    point_log_q = np.concatenate([old_point, new_point])
    return point_log_q, point_log_q[combined_point_index]


def _make_point_split(
    point_theta: np.ndarray,
    point_cohort: np.ndarray,
    validation_per_cohort: int = N_VALIDATION_PER_COHORT,
) -> np.ndarray:
    theta = np.asarray(point_theta, dtype=np.float64)
    cohort = np.asarray(point_cohort).astype("U")
    if theta.shape != (cohort.size, len(PARAMETER_NAMES)):
        raise ValueError("point theta/cohort shapes disagree")
    split = np.zeros(cohort.size, dtype=np.int8)
    for label in ("old", "new"):
        indices = np.flatnonzero(cohort == label)
        if indices.size <= validation_per_cohort:
            raise ValueError(f"cohort {label!r} is too small for the frozen split")
        ranked: list[tuple[bytes, int]] = []
        for index in indices:
            digest = hashlib.sha256()
            digest.update(b"godmax.mock_sbi.point_split.20260913\0")
            digest.update(label.encode("ascii"))
            digest.update(np.ascontiguousarray(theta[index]).tobytes())
            ranked.append((digest.digest(), int(index)))
        for _, index in sorted(ranked)[:validation_per_cohort]:
            split[index] = 1
    return split


def _find_declared_digest(value: Any, expected: str) -> bool:
    if isinstance(value, Mapping):
        return any(_find_declared_digest(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_find_declared_digest(item, expected) for item in value)
    return str(value) == expected


def _validate_theta_u(theta: np.ndarray, u: np.ndarray, label: str) -> None:
    """Require the saved physical/probit coordinates to be one exact prior map."""

    from scipy.special import ndtr

    theta_values = np.asarray(theta, dtype=np.float64)
    u_values = np.asarray(u, dtype=np.float64)
    if theta_values.shape != u_values.shape or theta_values.shape[-1:] != (5,):
        raise ValueError(f"{label}: theta/u shapes disagree")
    expected = PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * ndtr(u_values)
    absolute_tolerance = 32.0 * np.finfo(np.float64).eps * float(
        np.max(PRIOR_HIGH - PRIOR_LOW)
    )
    if not np.allclose(
        theta_values, expected, rtol=0.0, atol=absolute_tolerance
    ):
        maximum_error = float(np.max(np.abs(theta_values - expected)))
        raise ValueError(
            f"{label}: theta is not the registered prior transform of u; "
            f"maximum absolute error {maximum_error:.17g}"
        )


def _load_cache(path: pathlib.Path, context: Any, msc: Any) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fixed-g2 cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    manifest = _json_from_array(arrays["manifest_json"])
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"unexpected cache schema {manifest.get('schema_version')!r}")
    identity = {
        "noise_contract_sha256": context.contract_sha256,
        "workspace_sha256": context.workspace_sha256,
        "mask_array_sha256": context.mask_sha256,
        "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
    }
    for name, expected in identity.items():
        if str(manifest.get("estimator", {}).get(name)) != str(expected):
            raise ValueError(f"fixed-g2 cache estimator {name} changed")
    g2_alm = np.asarray(arrays.get("g2_alm"))
    if g2_alm.ndim != 1 or not np.all(np.isfinite(g2_alm)):
        raise ValueError("cached g2_alm is malformed")
    if msc.sha256_array(g2_alm) != str(manifest.get("g2_alm_sha256")):
        raise ValueError("cached g2_alm hash differs from its manifest")
    if not np.array_equal(arrays.get("band_edges"), context.band_edges):
        raise ValueError("fixed-g2 cache band edges differ from the estimator")
    return {"path": path, "arrays": arrays, "manifest": manifest, "g2_alm": g2_alm}


def _init_round2(args: argparse.Namespace) -> int:
    import healpy as hp
    import mock_sbi_common as msc

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"cache exists; pass --overwrite: {output}")
    remaps = g2build._parse_path_remaps(args.path_remap)
    context = g2build._load_estimator_context(msc, remaps)
    map_path, map_sha, frozen = g2build._resolve_and_verify_frozen_galaxy_map(
        msc, remaps
    )
    g2_alm, galaxy = g2build._build_fixed_g2_alm(
        context, map_path, map_sha, frozen, hp, msc
    )
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "purpose": "reuse the exact frozen squared-galaxy alm across round-2 shards",
        "estimator": {
            "noise_contract_sha256": context.contract_sha256,
            "workspace_sha256": context.workspace_sha256,
            "mask_array_sha256": context.mask_sha256,
            "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
        },
        "galaxy": galaxy,
        "g2_alm_sha256": msc.sha256_array(g2_alm),
        "source_code": {
            "this_file": _sha256_file(pathlib.Path(__file__).resolve()),
            "g2_builder": _sha256_file(pathlib.Path(g2build.__file__).resolve()),
            "mock_sbi_common": _sha256_file(pathlib.Path(msc.__file__).resolve()),
        },
        "path_remaps": list(args.path_remap),
    }
    print(f"fixed g2 alm sha256 {manifest['g2_alm_sha256']}")
    if args.preflight:
        print(f"PREFLIGHT PASS; would create {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        output,
        {
            "g2_alm": g2_alm,
            "band_edges": np.asarray(context.band_edges, dtype=np.int64),
            "effective_ell": np.asarray(context.effective_ell, dtype=np.float64),
            "manifest_json": json.dumps(manifest, sort_keys=True),
        },
    )
    _atomic_write_text(output.with_suffix(".json"), json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    return 0


def _load_round2_batch(
    batch_dir: pathlib.Path, context: Any | None = None
) -> dict[str, Any]:
    batch_dir = batch_dir.expanduser().resolve()
    if batch_dir.name != ROUND2_CAMPAIGN_NAME:
        raise ValueError(
            f"round-2 batch directory must be {ROUND2_CAMPAIGN_NAME!r}; "
            f"got {batch_dir.name!r}"
        )
    campaign_path = batch_dir / "campaign.json"
    complete_path = batch_dir / "campaign_complete.json"
    training_path = batch_dir / "training_set.npz"
    campaign = _load_json(campaign_path)
    completion = _load_json(complete_path)
    if not _find_declared_digest(completion, ROUND2_COMPLETION_AGGREGATE):
        raise ValueError(
            "campaign_complete.json does not contain the registered aggregate "
            f"SHA256 {ROUND2_COMPLETION_AGGREGATE}"
        )
    entries = campaign.get("entries")
    if not isinstance(entries, list) or len(entries) != N_NEW_POINTS:
        raise ValueError("round-2 campaign must contain exactly 128 entries")
    with np.load(training_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"x", "mu", "theta", "u", "log_q", "noise_seeds"}
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"round-2 batch training set is missing {missing}")
    shapes = {
        "x": (N_NEW_POINTS, NEW_REPLICAS, 42),
        "mu": (N_NEW_POINTS, 42),
        "theta": (N_NEW_POINTS, 5),
        "u": (N_NEW_POINTS, 5),
        "log_q": (N_NEW_POINTS,),
        "noise_seeds": (N_NEW_POINTS, NEW_REPLICAS, 3),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"round-2 {name} shape {arrays[name].shape}; expected {shape}")
        if arrays[name].dtype.kind in "fc" and not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"round-2 {name} contains non-finite values")
    if "parameter_names" in arrays and _strings(arrays["parameter_names"]) != PARAMETER_NAMES:
        raise ValueError("round-2 parameter order differs from the registered order")
    _validate_theta_u(arrays["theta"], arrays["u"], "round-2 training set")
    if context is not None:
        for name, expected in {
            "noise contract": context.contract_sha256,
            "workspace": context.workspace_sha256,
        }.items():
            if not _find_declared_digest(campaign, str(expected)):
                raise ValueError(
                    f"round-2 campaign does not pin the current {name} SHA256 "
                    f"{expected}"
                )
    return {
        "dir": batch_dir,
        "campaign_path": campaign_path,
        "campaign": campaign,
        "complete_path": complete_path,
        "completion": completion,
        "training_path": training_path,
        "arrays": arrays,
        "entries": entries,
    }


def _entry_theta(entry: Mapping[str, Any]) -> np.ndarray | None:
    value = entry.get("theta")
    if isinstance(value, Mapping):
        if not all(name in value for name in PARAMETER_NAMES):
            return None
        return np.asarray([value[name] for name in PARAMETER_NAMES], dtype=np.float64)
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    return result if result.shape == (5,) else None


def _entry_vector(entry: Mapping[str, Any], name: str) -> np.ndarray | None:
    value = entry.get(name)
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    return result if result.shape == (5,) else None


def _seeds_from_hdf5_attr(value: Any, order: Sequence[str]) -> np.ndarray:
    parsed = json.loads(_text(value))
    if isinstance(parsed, Mapping):
        if not all(name in parsed for name in order):
            raise ValueError("field_seeds_json is missing a declared seed field")
        return np.asarray([parsed[name] for name in order], dtype=np.uint32)
    result = np.asarray(parsed, dtype=np.uint32)
    if result.shape != (len(order),):
        raise ValueError("field_seeds_json is neither a seed mapping nor a seed row")
    return result


def _declared_path_matches(
    raw_path: str,
    expected: pathlib.Path,
    roots: Sequence[pathlib.Path],
) -> bool:
    declared = pathlib.Path(raw_path)
    candidates = [declared] if declared.is_absolute() else [root / declared for root in roots]
    if any(candidate.expanduser().resolve() == expected for candidate in candidates):
        return True
    # Registered products may move with the checkout.  Admit relocation only
    # when the complete repository-relative data suffix is identical.
    try:
        declared_suffix = declared.parts[declared.parts.index("data") :]
        expected_suffix = expected.parts[expected.parts.index("data") :]
    except ValueError:
        return False
    return declared_suffix == expected_suffix


def _load_round2_point(
    batch: Mapping[str, Any],
    point: int,
    context: Any,
    canonical_seed_order: Sequence[str],
) -> dict[str, Any]:
    import h5py

    if not 0 <= point < N_NEW_POINTS:
        raise IndexError(f"round-2 point {point} is outside 0..127")
    entry = batch["entries"][point]
    if not isinstance(entry, Mapping):
        raise ValueError(f"campaign entry {point} is not an object")
    if int(entry.get("index", point)) != point:
        raise ValueError(f"campaign entry {point} has a different index")
    run_name = str(entry.get("run_name", ""))
    if not run_name:
        raise ValueError(f"campaign entry {point} has no run_name")
    theta = np.asarray(batch["arrays"]["theta"][point], dtype=np.float64)
    u = np.asarray(batch["arrays"]["u"][point], dtype=np.float64)
    declared_theta = _entry_theta(entry)
    if declared_theta is not None and not np.array_equal(theta, declared_theta):
        raise ValueError(f"campaign and training theta differ at point {point}")
    declared_u = _entry_vector(entry, "u")
    if declared_u is not None and not np.array_equal(u, declared_u):
        raise ValueError(f"campaign and training u differ at point {point}")

    point_dir = pathlib.Path(batch["dir"]) / "noisy_maps" / run_name
    signal_dir = pathlib.Path(batch["dir"]) / "maps" / run_name
    signal_candidates = sorted(
        path.resolve()
        for path in signal_dir.glob("*.h5")
        if "_split" not in path.name
    )
    if len(signal_candidates) != 1:
        raise ValueError(
            f"round-2 point {point} has {len(signal_candidates)} combined signal-map "
            f"candidates under {signal_dir}; expected exactly one"
        )
    signal_path = signal_candidates[0]
    products_path = point_dir / "products.npz"
    completion_path = point_dir / "completion.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"round-2 point completion is missing: {completion_path}")
    with np.load(products_path, allow_pickle=False) as archive:
        products = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "x", "mu", "theta", "u", "bank_indices", "noise_seeds",
        "seed_field_order", "map_filenames", "parameter_names",
        "provenance_json", "diagnostic_json",
    }
    missing = sorted(required - set(products))
    if missing:
        raise KeyError(f"{products_path} is missing {missing}")
    expected_shapes = {
        "x": (NEW_REPLICAS, 42),
        "mu": (42,),
        "theta": (5,),
        "u": (5,),
        "bank_indices": (NEW_REPLICAS,),
        "noise_seeds": (NEW_REPLICAS, 3),
        "map_filenames": (NEW_REPLICAS,),
    }
    for name, shape in expected_shapes.items():
        if products[name].shape != shape:
            raise ValueError(f"{products_path.name}:{name} has shape {products[name].shape}, expected {shape}")
    exact_pairs = {
        "x": batch["arrays"]["x"][point],
        "mu": batch["arrays"]["mu"][point],
        "theta": theta,
        "u": u,
        "noise_seeds": batch["arrays"]["noise_seeds"][point],
    }
    for name, expected in exact_pairs.items():
        if not np.array_equal(products[name], expected):
            raise ValueError(f"batch and per-point {name} differ at point {point}")
    if not np.all(np.isfinite(products["x"])) or not np.all(np.isfinite(products["mu"])):
        raise ValueError(f"round-2 products contain non-finite vectors at point {point}")
    if _strings(products["parameter_names"]) != PARAMETER_NAMES:
        raise ValueError(f"round-2 products have the wrong parameter order at point {point}")
    provenance = _json_from_array(products["provenance_json"])
    _json_from_array(products["diagnostic_json"])
    bank_indices = np.asarray(products["bank_indices"], dtype=np.int64)
    if np.any(bank_indices < 0) or np.unique(bank_indices).size != NEW_REPLICAS:
        raise ValueError(f"point {point} does not have ten distinct non-negative bank indices")
    declared_bank_indices = entry.get("bank_indices")
    if declared_bank_indices is None:
        raise KeyError(f"campaign entry {point} has no bank_indices table")
    if not np.array_equal(
        bank_indices, np.asarray(declared_bank_indices, dtype=np.int64)
    ):
        raise ValueError(f"campaign and products bank indices differ at point {point}")
    saved_seed_order = _strings(products["seed_field_order"])
    canonical_seeds = _canonical_seed_rows(
        products["noise_seeds"], saved_seed_order, canonical_seed_order
    )

    map_paths: list[pathlib.Path] = []
    map_metadata: list[dict[str, Any]] = []
    for replica, raw_name in enumerate(products["map_filenames"]):
        declared = pathlib.Path(_text(raw_name))
        path = declared if declared.is_absolute() else point_dir / declared
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"round-2 noisy bundle is missing: {path}")
        with h5py.File(path, "r") as handle:
            for name in ("replica", "bank_index", "field_seeds_json", "signal_path"):
                if name not in handle.attrs:
                    raise KeyError(f"{path} is missing HDF5 attribute {name!r}")
            if int(handle.attrs["replica"]) != replica:
                raise ValueError(f"{path} has the wrong local replica")
            if int(handle.attrs["bank_index"]) != int(bank_indices[replica]):
                raise ValueError(f"{path} has the wrong bank index")
            stored_seed_row = _seeds_from_hdf5_attr(
                handle.attrs["field_seeds_json"], saved_seed_order
            )
            stored_canonical = _canonical_seed_rows(
                stored_seed_row[None, :], saved_seed_order, canonical_seed_order
            )[0]
            if not np.array_equal(stored_canonical, canonical_seeds[replica]):
                raise ValueError(f"{path} has the wrong field seed triplet")
            declared_signal = _text(handle.attrs["signal_path"])
            if not _declared_path_matches(
                declared_signal,
                signal_path,
                (path.parent, pathlib.Path(batch["dir"])),
            ):
                raise ValueError(
                    f"{path} points at signal_path={declared_signal!r}, not the "
                    f"campaign signal map {signal_path}"
                )
            for field, dataset in MAP_DATASETS.items():
                if dataset not in handle:
                    raise KeyError(f"{path} has no dataset {dataset!r}")
                stored = handle[dataset]
                if tuple(stored.shape) != tuple(context.mask.shape):
                    raise ValueError(
                        f"{path}:{dataset} shape {stored.shape}; expected "
                        f"{tuple(context.mask.shape)}"
                    )
                if np.dtype(stored.dtype) != np.dtype(np.float32):
                    raise TypeError(f"{path}:{dataset} dtype {stored.dtype}; expected float32")
            map_metadata.append(
                {
                    "replica": replica,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "bank_index": int(bank_indices[replica]),
                    "noise_seeds": canonical_seeds[replica].tolist(),
                    "signal_path": str(signal_path),
                }
            )
        map_paths.append(path)
    return {
        "point": point,
        "run_name": run_name,
        "point_dir": point_dir,
        "signal_path": signal_path,
        "products_path": products_path,
        "completion_path": completion_path,
        "products": products,
        "theta": theta,
        "u": u,
        "log_q": float(batch["arrays"]["log_q"][point]),
        "bank_indices": bank_indices,
        "noise_seeds": canonical_seeds,
        "seed_field_order_saved": saved_seed_order,
        "map_paths": map_paths,
        "map_metadata": map_metadata,
        "provenance": provenance,
    }


def _measure_saved_bundle_pair(
    path: pathlib.Path,
    context: Any,
    g2_alm: np.ndarray,
    hp: Any,
    msc: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure paired gX and g2X from one unmasked three-field map bundle."""

    import h5py

    gx_blocks: list[np.ndarray] = []
    g2x_blocks: list[np.ndarray] = []
    removed: list[float] = []
    with h5py.File(path, "r") as handle:
        for _, field in g2build.G2X_SPECTRA:
            total = np.asarray(handle[MAP_DATASETS[field]], dtype=np.float64)
            if total.shape != context.mask.shape or not np.all(np.isfinite(total)):
                raise ValueError(f"{path}:{field} contains an invalid map")
            centered, mean = msc.subtract_weighted_mean(total, context.mask)
            total_alm = hp.map2alm(
                context.mask * centered,
                lmax=msc.LMAX,
                iter=0,
            )
            coupled_gx = hp.alm2cl(
                context.fixed_galaxy_alm, total_alm, lmax=msc.LMAX
            )
            coupled_g2x = hp.alm2cl(
                g2_alm, total_alm, lmax=msc.LMAX
            )
            gx_bandpowers = np.asarray(
                context.workspace.decouple_cell(coupled_gx[None, :]),
                dtype=np.float64,
            )[0]
            g2x_bandpowers = np.asarray(
                context.workspace.decouple_cell(coupled_g2x[None, :]),
                dtype=np.float64,
            )[0]
            if gx_bandpowers.shape != (14,) or not np.all(
                np.isfinite(gx_bandpowers)
            ):
                raise RuntimeError(f"{path}:g{field} produced invalid bandpowers")
            if g2x_bandpowers.shape != (14,) or not np.all(
                np.isfinite(g2x_bandpowers)
            ):
                raise RuntimeError(f"{path}:g2{field} produced invalid bandpowers")
            gx_blocks.append(gx_bandpowers)
            g2x_blocks.append(g2x_bandpowers)
            removed.append(float(mean))
            del total, centered, total_alm, coupled_gx, coupled_g2x
    return (
        np.concatenate(gx_blocks),
        np.concatenate(g2x_blocks),
        np.asarray(removed, dtype=np.float64),
    )


def _measure_round2(args: argparse.Namespace) -> int:
    import healpy as hp
    import mock_sbi_common as msc

    if not 0 <= args.point_start < args.point_stop <= N_NEW_POINTS:
        raise ValueError("require 0 <= --point-start < --point-stop <= 128")
    remaps = g2build._parse_path_remaps(args.path_remap)
    context = g2build._load_estimator_context(msc, remaps)
    cache = _load_cache(args.g2_cache, context, msc)
    batch = _load_round2_batch(args.batch_dir, context)
    output_dir = args.output_dir.expanduser().resolve()
    prepared: list[dict[str, Any]] = []
    for point in range(args.point_start, args.point_stop):
        prepared.append(
            _load_round2_point(batch, point, context, msc.NOISE_FIELDS)
        )
    print(
        f"validated round-2 points {args.point_start}:{args.point_stop}; "
        f"{len(prepared) * NEW_REPLICAS} saved paired bundles"
    )
    print("source mask hashes ignored; both gX and g2X will be remeasured")
    if args.preflight:
        print(f"PREFLIGHT PASS; would write shards under {output_dir}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for record in prepared:
        point = int(record["point"])
        output = output_dir / f"round2_point_{point:03d}.npz"
        sidecar = output.with_suffix(".json")
        if (output.exists() or sidecar.exists()) and args.skip_existing:
            print(f"skip existing {output.name}")
            continue
        if (output.exists() or sidecar.exists()) and not args.overwrite:
            raise FileExistsError(f"round-2 shard exists; pass --overwrite or --skip-existing: {output}")
        gx_array = np.empty((NEW_REPLICAS, 42), dtype=np.float64)
        g2x = np.empty((NEW_REPLICAS, 42), dtype=np.float64)
        removed = np.empty((NEW_REPLICAS, 3), dtype=np.float64)
        for replica, path in enumerate(record["map_paths"]):
            gx_array[replica], g2x[replica], removed[replica] = (
                _measure_saved_bundle_pair(
                    path, context, cache["g2_alm"], hp, msc
                )
            )
            print(
                f"point {point:03d} replica {replica:02d}: {path.name}",
                flush=True,
            )
        gx_signal, _, signal_removed = _measure_saved_bundle_pair(
            record["signal_path"], context, cache["g2_alm"], hp, msc
        )
        gx_products_reference = np.asarray(
            record["products"]["x"], dtype=np.float64
        )
        gx_signal_products_reference = np.asarray(
            record["products"]["mu"], dtype=np.float64
        )
        manifest = {
            "schema_version": SHARD_SCHEMA,
            "point_index": point,
            "run_name": record["run_name"],
            "replicas": NEW_REPLICAS,
            "gx_vector_order": GX_VECTOR_ORDER,
            "g2x_vector_order": G2X_VECTOR_ORDER,
            "seed_field_order_saved": list(record["seed_field_order_saved"]),
            "seed_field_order_canonical": list(msc.NOISE_FIELDS),
            "products": {
                "path": str(record["products_path"]),
                "sha256": _sha256_file(record["products_path"]),
            },
            "completion": {
                "path": str(record["completion_path"]),
                "sha256": _sha256_file(record["completion_path"]),
            },
            "campaign": {
                "path": str(batch["campaign_path"]),
                "sha256": _sha256_file(batch["campaign_path"]),
            },
            "fixed_g2_cache": {
                "path": str(cache["path"]),
                "sha256": _sha256_file(cache["path"]),
                "g2_alm_sha256": cache["manifest"]["g2_alm_sha256"],
            },
            "estimator": {
                "noise_contract_sha256": context.contract_sha256,
                "workspace_sha256": context.workspace_sha256,
                "mask_array_sha256": context.mask_sha256,
                "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
            },
            "map_bundles": record["map_metadata"],
            "signal_map": {
                "path": str(record["signal_path"]),
                "sha256": _sha256_file(record["signal_path"]),
            },
            "source_measurement_provenance_policy": (
                "not used as an estimator-identity gate; saved HDF5 fields are "
                "unmasked and both statistics are remeasured with this estimator"
            ),
            "gx_source": "remeasured directly from the saved noisy HDF5 bundle",
            "g2x_source": "remeasured from the same saved noisy HDF5 bundle and same field alm",
            "gx_products_reference_sha256": msc.sha256_array(
                gx_products_reference
            ),
            "gx_remeasurement_minus_products_max_abs": float(
                np.max(np.abs(gx_array - gx_products_reference))
            ),
            "gx_signal_products_reference_sha256": msc.sha256_array(
                gx_signal_products_reference
            ),
            "gx_signal_remeasurement_minus_products_max_abs": float(
                np.max(np.abs(gx_signal - gx_signal_products_reference))
            ),
            "gx_sha256": msc.sha256_array(gx_array),
            "g2x_sha256": msc.sha256_array(g2x),
            "elapsed_seconds": time.time() - started,
        }
        arrays = {
            "gx": gx_array,
            "g2x": g2x,
            "gx_point_signal": gx_signal,
            "gx_products_reference": gx_products_reference,
            "gx_point_signal_products_reference": gx_signal_products_reference,
            "theta": np.asarray(record["theta"], dtype=np.float64),
            "u": np.asarray(record["u"], dtype=np.float64),
            "log_q": np.asarray(record["log_q"], dtype=np.float64),
            "replica": np.arange(NEW_REPLICAS, dtype=np.int64),
            "bank_indices": np.asarray(record["bank_indices"], dtype=np.int64),
            "noise_seeds": np.asarray(record["noise_seeds"], dtype=np.uint32),
            "removed_field_means_g2x": removed,
            "removed_field_means_signal": signal_removed,
            "map_filenames": np.asarray([str(path) for path in record["map_paths"]]),
            "manifest_json": json.dumps(manifest, sort_keys=True),
        }
        _atomic_savez(output, arrays)
        _atomic_write_text(sidecar, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output}")
    return 0


def _load_round2_shards(
    shard_dir: pathlib.Path,
    batch: Mapping[str, Any],
    msc: Any,
    context: Any,
    cache: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    shard_dir = shard_dir.expanduser().resolve()
    result: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "gx", "g2x", "gx_point_signal", "theta", "u", "log_q",
            "bank_indices", "noise_seeds",
        )
    }
    for point in range(N_NEW_POINTS):
        source = _load_round2_point(batch, point, context, msc.NOISE_FIELDS)
        path = shard_dir / f"round2_point_{point:03d}.npz"
        sidecar = path.with_suffix(".json")
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"round-2 shard or sidecar is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        manifest = _json_from_array(arrays["manifest_json"])
        if _load_json(sidecar) != manifest:
            raise ValueError(f"round-2 shard sidecar differs: {path}")
        if manifest.get("schema_version") != SHARD_SCHEMA or int(manifest.get("point_index", -1)) != point:
            raise ValueError(f"round-2 shard identity is invalid: {path}")
        if manifest.get("gx_vector_order") != GX_VECTOR_ORDER or manifest.get(
            "g2x_vector_order"
        ) != G2X_VECTOR_ORDER:
            raise ValueError(f"round-2 shard vector order changed: {path}")
        if tuple(manifest.get("seed_field_order_canonical", ())) != tuple(
            msc.NOISE_FIELDS
        ):
            raise ValueError(f"round-2 shard canonical seed order changed: {path}")
        if str(manifest.get("campaign", {}).get("sha256")) != _sha256_file(
            batch["campaign_path"]
        ):
            raise ValueError(f"round-2 shard campaign identity changed: {path}")
        if str(manifest.get("products", {}).get("sha256")) != _sha256_file(
            source["products_path"]
        ):
            raise ValueError(f"round-2 shard products identity changed: {path}")
        if str(manifest.get("completion", {}).get("sha256")) != _sha256_file(
            source["completion_path"]
        ):
            raise ValueError(f"round-2 shard completion identity changed: {path}")
        if str(manifest.get("fixed_g2_cache", {}).get("sha256")) != _sha256_file(
            cache["path"]
        ) or str(manifest.get("fixed_g2_cache", {}).get("g2_alm_sha256")) != str(
            cache["manifest"]["g2_alm_sha256"]
        ):
            raise ValueError(f"round-2 shard fixed-g2 cache identity changed: {path}")
        expected_estimator = {
            "noise_contract_sha256": context.contract_sha256,
            "workspace_sha256": context.workspace_sha256,
            "mask_array_sha256": context.mask_sha256,
            "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
        }
        if manifest.get("estimator") != expected_estimator:
            raise ValueError(f"round-2 shard estimator identity changed: {path}")
        expected = {
            "gx": (10, 42), "g2x": (10, 42), "gx_point_signal": (42,),
            "gx_products_reference": (10, 42),
            "gx_point_signal_products_reference": (42,),
            "theta": (5,), "u": (5,), "log_q": (), "bank_indices": (10,),
            "noise_seeds": (10, 3), "replica": (10,),
        }
        for name, shape in expected.items():
            if name not in arrays or arrays[name].shape != shape:
                observed = None if name not in arrays else arrays[name].shape
                raise ValueError(f"{path}:{name} shape {observed}; expected {shape}")
        if not np.array_equal(arrays["replica"], np.arange(10, dtype=np.int64)):
            raise ValueError(f"{path} has noncanonical replica IDs")
        if not np.array_equal(
            arrays["gx_products_reference"], batch["arrays"]["x"][point]
        ):
            raise ValueError(f"{path} changed its saved products.npz gX reference")
        if not np.array_equal(
            arrays["gx_point_signal_products_reference"],
            batch["arrays"]["mu"][point],
        ):
            raise ValueError(
                f"{path} changed its saved products.npz noiseless gX reference"
            )
        if not np.array_equal(arrays["theta"], batch["arrays"]["theta"][point]) or not np.array_equal(
            arrays["u"], batch["arrays"]["u"][point]
        ):
            raise ValueError(f"{path} changed the round-2 parameter row")
        if not np.array_equal(
            arrays["log_q"], batch["arrays"]["log_q"][point], equal_nan=True
        ):
            raise ValueError(f"{path} changed the round-2 point log_q")
        if not np.array_equal(arrays["bank_indices"], source["bank_indices"]) or not np.array_equal(
            arrays["noise_seeds"], source["noise_seeds"]
        ):
            raise ValueError(f"{path} changed the round-2 bank/seed row key")
        if msc.sha256_array(arrays["gx"]) != str(manifest.get("gx_sha256")):
            raise ValueError(f"{path} gX hash mismatch")
        if msc.sha256_array(arrays["g2x"]) != str(manifest.get("g2x_sha256")):
            raise ValueError(f"{path} g2X hash mismatch")
        for name in result:
            result[name].append(np.asarray(arrays[name]))
    return {
        "gx": np.concatenate(result["gx"], axis=0),
        "g2x": np.concatenate(result["g2x"], axis=0),
        "gx_point_signal": np.stack(result["gx_point_signal"]),
        "point_theta": np.stack(result["theta"]),
        "point_u": np.stack(result["u"]),
        "point_log_q": np.asarray(result["log_q"], dtype=np.float64),
        "bank_indices": np.concatenate(result["bank_indices"]),
        "noise_seeds": np.concatenate(result["noise_seeds"], axis=0),
    }


def _bank_overlap(
    bank_indices: np.ndarray,
    noise_seeds: np.ndarray,
    point_index: np.ndarray,
    point_split: np.ndarray,
) -> dict[str, int]:
    train_rows = point_split[point_index] == 0
    validation_rows = ~train_rows
    train_bank = set(np.asarray(bank_indices[train_rows], dtype=np.int64).tolist())
    validation_bank = set(np.asarray(bank_indices[validation_rows], dtype=np.int64).tolist())
    train_seeds = {tuple(int(x) for x in row) for row in noise_seeds[train_rows]}
    validation_seeds = {tuple(int(x) for x in row) for row in noise_seeds[validation_rows]}
    return {
        "train_unique_bank_indices": len(train_bank),
        "validation_unique_bank_indices": len(validation_bank),
        "shared_bank_indices": len(train_bank & validation_bank),
        "train_unique_seed_triplets": len(train_seeds),
        "validation_unique_seed_triplets": len(validation_seeds),
        "shared_seed_triplets": len(train_seeds & validation_seeds),
    }


def _assemble(args: argparse.Namespace) -> int:
    import healpy as hp
    import mock_sbi_common as msc
    from three_probe_inference_contract import MOCK_CONTRACT_PATH, VECTOR_ORDER

    output = args.output.expanduser().resolve()
    sidecar = output.with_suffix(".json")
    if (output.exists() or sidecar.exists()) and not (args.overwrite or args.preflight):
        raise FileExistsError(f"master output exists; pass --overwrite: {output}")
    remaps = g2build._parse_path_remaps(args.path_remap)
    context = g2build._load_estimator_context(msc, remaps)
    cache = _load_cache(args.g2_cache, context, msc)
    old = joint_builder._load_g2x_rows(args.old_g2x_training_set, msc)
    if int(old["n_points"]) != N_OLD_POINTS or int(old["replicas"]) != OLD_REPLICAS:
        raise ValueError("old child must be the complete 128-point, five-replica product")
    if int(old["n_rows"]) != N_OLD_POINTS * OLD_REPLICAS:
        raise ValueError("old child row count is not 640")
    joint_builder._check_estimator_identity(old, context)
    plan, prepared = joint_builder._validate_plan_and_sources(
        args.old_plan, old, context, remaps, msc
    )
    if len(prepared) != N_OLD_POINTS or len(plan["entries"]) != N_OLD_POINTS:
        raise ValueError("old plan/source match does not cover all 128 original points")
    batch = _load_round2_batch(args.batch_dir, context)
    new = _load_round2_shards(
        args.round2_shard_dir, batch, msc, context, cache
    )

    # Build and validate every cheap cross-child assembly table before the
    # preflight exit.  This makes preflight exercise the complete old/new key,
    # shape and row-index contract without repeating the expensive map transforms.
    point_theta = np.concatenate([old["point_theta"], new["point_theta"]], axis=0)
    point_u = np.concatenate([old["point_u"], new["point_u"]], axis=0)
    if point_theta.shape != (N_POINTS, 5) or point_u.shape != (N_POINTS, 5):
        raise ValueError(
            f"combined point theta/u shapes are {point_theta.shape}/{point_u.shape}; "
            f"expected {(N_POINTS, 5)}"
        )
    if not np.all(np.isfinite(point_theta)) or not np.all(np.isfinite(point_u)):
        raise ValueError("all 256 design points require finite theta and u")
    if len({np.ascontiguousarray(row).tobytes() for row in point_theta}) != N_POINTS:
        raise ValueError("the combined cohorts do not contain 256 distinct theta rows")
    _validate_theta_u(point_theta, point_u, "combined 256-point design")

    replica_count = np.concatenate(
        [np.full(N_OLD_POINTS, OLD_REPLICAS), np.full(N_NEW_POINTS, NEW_REPLICAS)]
    ).astype(np.int64)
    row_offsets, point_index, replica = _replica_table(replica_count)
    dof = _validate_flat_rows(point_index, replica, replica_count, row_offsets)
    if dof != COVARIANCE_DOF or int(row_offsets[-1]) != N_ROWS:
        raise AssertionError("heterogeneous replica algebra changed")

    point_log_q, log_q = _assemble_proposal_density(old, new, point_index)

    theta = point_theta[point_index]
    u = point_u[point_index]
    bank_indices = np.concatenate([old["bank_indices"], new["bank_indices"]])
    noise_seeds = np.concatenate([old["noise_seeds"], new["noise_seeds"]], axis=0)
    if bank_indices.shape != (N_ROWS,) or noise_seeds.shape != (N_ROWS, 3):
        raise ValueError(
            f"combined bank/seed shapes are {bank_indices.shape}/{noise_seeds.shape}; "
            f"expected {(N_ROWS,)}/{(N_ROWS, 3)}"
        )
    point_cohort = np.asarray(["old"] * N_OLD_POINTS + ["new"] * N_NEW_POINTS)
    cohort = point_cohort[point_index]
    source_point_index = np.concatenate(
        [old["point_plan_position"], np.arange(N_NEW_POINTS, dtype=np.int64)]
    )
    if source_point_index.shape != (N_POINTS,):
        raise ValueError("combined source-point table has the wrong shape")
    point_split = _make_point_split(point_theta, point_cohort)
    if np.sum(point_split[:N_OLD_POINTS]) != N_VALIDATION_PER_COHORT or np.sum(
        point_split[N_OLD_POINTS:]
    ) != N_VALIDATION_PER_COHORT:
        raise AssertionError("frozen point split lost cohort stratification")

    print("validated old 128x5 child and all 128x10 round-2 shards")
    if args.preflight:
        print(
            f"PREFLIGHT PASS; validated all child interfaces and would assemble "
            f"{N_POINTS} points / {N_ROWS} rows into {output}"
        )
        return 0

    print("measuring the 128 old noiseless gX signals", flush=True)
    old_signal = np.empty((N_OLD_POINTS, 42), dtype=np.float64)
    for point, (_, info, record) in enumerate(prepared):
        maps = joint_builder._load_signal_maps(info, record, context, msc)
        old_signal[point] = msc.measure_map_vector(maps, context)
        print(f"old point {point + 1:03d}/{N_OLD_POINTS}", flush=True)
        del maps

    replayed = 0

    def noise_provider(key: tuple[int, int, int]) -> np.ndarray:
        nonlocal replayed
        seeds = dict(zip(msc.NOISE_FIELDS, key, strict=True))
        value = msc.noise_vector(seeds, context)
        replayed += 1
        if replayed % 32 == 0:
            print(f"replayed {replayed} unique old gX noise triplets", flush=True)
        return value

    old_gx, old_gx_noise, old_unique_noise = joint_builder._assemble_gx_rows(
        old_signal, old["point_index"], old["noise_seeds"], noise_provider
    )
    if old_gx.shape != (640, 42) or new["gx"].shape != (1280, 42):
        raise AssertionError("child gX shapes changed")

    gx = np.concatenate([old_gx, new["gx"]], axis=0)
    g2x = np.concatenate([old["x"], new["g2x"]], axis=0)
    gx_point_signal = np.concatenate([old_signal, new["gx_point_signal"]], axis=0)

    print("loading the registered noiseless observations", flush=True)
    mock_contract, mock_contract_source_policy = joint_builder._load_mock_contract_without_source_provenance(
        MOCK_CONTRACT_PATH, msc
    )
    if VECTOR_ORDER != GX_VECTOR_ORDER or msc.VECTOR_ORDER != GX_VECTOR_ORDER:
        raise ValueError("registered gX vector order changed")
    gx_observation = np.asarray(mock_contract.data_vector, dtype=np.float64)
    reference_map_path, reference_sha, _ = g2build._resolve_and_verify_frozen_galaxy_map(
        msc, remaps
    )
    reference_maps = msc.read_paste_maps(reference_map_path)
    g2x_observation, observation_removed_means = g2contract._measure_noiseless_g2x(
        reference_maps, context, cache["g2_alm"], hp, msc
    )
    del reference_maps
    generating_path = g2build._resolve_path(
        args.generating_point, msc.REPO_ROOT, remaps
    )
    theta_reference, u_reference, generating_payload = g2contract._load_generating_point(
        generating_path
    )

    arrays: dict[str, Any] = {
        "gx": gx,
        "g2x": g2x,
        "gx_point_signal": gx_point_signal,
        "gx_old_noise": old_gx_noise,
        "gx_data_vector": gx_observation,
        "g2x_data_vector": g2x_observation,
        "theta": theta,
        "u": u,
        "log_q": log_q,
        "point_theta": point_theta,
        "point_u": point_u,
        "point_log_q": point_log_q,
        "point_index": point_index,
        "replica": replica,
        "replica_count": replica_count,
        "point_row_offsets": row_offsets,
        "cohort": cohort,
        "point_cohort": point_cohort,
        "source_point_index": source_point_index,
        "bank_indices": bank_indices,
        "noise_seeds": noise_seeds,
        "point_split": point_split,
        "theta_reference": theta_reference,
        "u_reference": u_reference,
        "band_edges": np.asarray(context.band_edges, dtype=np.int64),
        "effective_ell": np.asarray(context.effective_ell, dtype=np.float64),
    }
    expected_master_shapes = {
        "gx": (N_ROWS, 42),
        "g2x": (N_ROWS, 42),
        "gx_point_signal": (N_POINTS, 42),
        "gx_old_noise": (N_OLD_POINTS * OLD_REPLICAS, 42),
        "gx_data_vector": (42,),
        "g2x_data_vector": (42,),
        "theta": (N_ROWS, 5),
        "u": (N_ROWS, 5),
        "log_q": (N_ROWS,),
        "point_theta": (N_POINTS, 5),
        "point_u": (N_POINTS, 5),
        "point_log_q": (N_POINTS,),
        "point_index": (N_ROWS,),
        "replica": (N_ROWS,),
        "replica_count": (N_POINTS,),
        "point_row_offsets": (N_POINTS + 1,),
        "cohort": (N_ROWS,),
        "point_cohort": (N_POINTS,),
        "source_point_index": (N_POINTS,),
        "bank_indices": (N_ROWS,),
        "noise_seeds": (N_ROWS, 3),
        "point_split": (N_POINTS,),
        "theta_reference": (5,),
        "u_reference": (5,),
        "band_edges": (15,),
        "effective_ell": (14,),
    }
    for name, expected_shape in expected_master_shapes.items():
        observed_shape = np.asarray(arrays[name]).shape
        if observed_shape != expected_shape:
            raise ValueError(
                f"master array {name!r} shape {observed_shape}; "
                f"expected {expected_shape}"
            )
    if not np.array_equal(
        np.asarray(arrays["log_q"]),
        np.asarray(arrays["point_log_q"])[np.asarray(arrays["point_index"])],
        equal_nan=True,
    ):
        raise ValueError("master row log_q does not match point_log_q")
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            if name not in {"log_q", "point_log_q"}:
                raise ValueError(f"master array {name} contains non-finite values")
    inventory = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": msc.sha256_array(np.asarray(value)),
        }
        for name, value in arrays.items()
    }
    manifest = {
        "schema_version": MASTER_SCHEMA,
        "analysis_status": "exploratory_two_snapshot_native_replica_rerun",
        "n_points": N_POINTS,
        "n_rows": N_ROWS,
        "replica_count_histogram": {"5": N_OLD_POINTS, "10": N_NEW_POINTS},
        "covariance_degrees_of_freedom_all_points": COVARIANCE_DOF,
        "covariance_dof_interpretation": (
            "algebraic within-point residual df; finite-bank reuse reduces effective independence"
        ),
        "parameter_names": list(PARAMETER_NAMES),
        "gx_vector_order": GX_VECTOR_ORDER,
        "g2x_vector_order": G2X_VECTOR_ORDER,
        "seed_field_order": list(msc.NOISE_FIELDS),
        "row_order": "cohort old then new; within cohort point-major then local replica",
        "cohorts": {
            "old": {
                "points": N_OLD_POINTS,
                "replicas_per_point": OLD_REPLICAS,
                "source_snapshot": "original round-1 plan/runtime",
                "g2x_training_set": str(old["path"]),
                "g2x_training_set_sha256": old["file_sha256"],
                "paste_plan": str(args.old_plan.expanduser().resolve()),
                "paste_plan_sha256": msc.sha256_file(args.old_plan.expanduser().resolve()),
                "gx_noise_route": "direct replay of each saved g2X seed triplet",
                "unique_gx_noise_triplets_replayed": old_unique_noise,
            },
            "new": {
                "points": N_NEW_POINTS,
                "replicas_per_point": NEW_REPLICAS,
                "source_snapshot": "round-2 frozen runtime in the campaign directory",
                "campaign": str(batch["campaign_path"]),
                "campaign_sha256": _sha256_file(batch["campaign_path"]),
                "campaign_complete": str(batch["complete_path"]),
                "campaign_complete_sha256": _sha256_file(batch["complete_path"]),
                "registered_completion_aggregate_sha256": ROUND2_COMPLETION_AGGREGATE,
                "training_set": str(batch["training_path"]),
                "training_set_sha256": _sha256_file(batch["training_path"]),
                "gx_route": (
                    "direct remeasurement of each saved unmasked noisy HDF5 bundle "
                    "with the current estimator"
                ),
                "g2x_route": (
                    "paired remeasurement from the same bundle and same masked field alm"
                ),
            },
        },
        "source_snapshot_warning": (
            "old and new scientific source snapshots differ; pooled inference is exploratory"
        ),
        "estimator": {
            "noise_contract_sha256": context.contract_sha256,
            "workspace_sha256": context.workspace_sha256,
            "mask_array_sha256": context.mask_sha256,
            "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
            "g2_alm_sha256": cache["manifest"]["g2_alm_sha256"],
        },
        "observation": {
            "gx_kind": "registered noiseless pasted-map measurement",
            "gx_sha256": msc.sha256_array(gx_observation),
            "g2x_kind": "same reference paste measured with fixed squared galaxy field",
            "g2x_sha256": msc.sha256_array(g2x_observation),
            "g2x_removed_field_means": observation_removed_means.tolist(),
            "reference_map_path": str(reference_map_path),
            "reference_map_sha256": reference_sha,
            "gx_contract_source_policy": mock_contract_source_policy,
        },
        "generating_point": {
            "path": str(generating_path),
            "sha256": msc.sha256_file(generating_path),
            "source_schema": generating_payload.get("schema"),
        },
        "point_split": {
            "method": "13 lowest SHA256(theta,cohort) per cohort held out",
            "training_points": int(np.sum(point_split == 0)),
            "validation_points": int(np.sum(point_split == 1)),
            "validation_old": int(np.sum(point_split[:128] == 1)),
            "validation_new": int(np.sum(point_split[128:] == 1)),
            "sha256": msc.sha256_array(point_split),
            "bank_overlap": _bank_overlap(
                bank_indices, noise_seeds, point_index, point_split
            ),
        },
        "arrays": inventory,
        "source_code": {
            "this_file": _sha256_file(pathlib.Path(__file__).resolve()),
            "g2_builder": _sha256_file(pathlib.Path(g2build.__file__).resolve()),
            "g2_contract": _sha256_file(pathlib.Path(g2contract.__file__).resolve()),
            "joint_builder": _sha256_file(pathlib.Path(joint_builder.__file__).resolve()),
            "mock_sbi_common": _sha256_file(pathlib.Path(msc.__file__).resolve()),
        },
        "null_controls": {
            "old_child_modified": False,
            "round2_saved_map_bundles_modified": False,
            "round2_products_gx_used_for_training": False,
            "round2_gx_and_g2x_remeasured_from_same_field_alm": True,
            "covariance_repair": None,
        },
    }
    arrays["manifest_json"] = json.dumps(manifest, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output, arrays)
    _atomic_write_text(sidecar, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    print(f"points/rows/complete-data covariance df: {N_POINTS}/{N_ROWS}/{COVARIANCE_DOF}")
    print(f"point split: 230 train / 26 held out; {manifest['point_split']['bank_overlap']}")
    return 0


def _run_self_test() -> int:
    counts = np.concatenate([np.full(128, 5), np.full(128, 10)]).astype(np.int64)
    offsets, points, replicas = _replica_table(counts)
    assert offsets[-1] == 1920
    assert _validate_flat_rows(points, replicas, counts, offsets) == 1664
    old_point_log_q = np.linspace(-8.0, -4.0, N_OLD_POINTS)
    new_point_log_q = np.linspace(-3.0, -1.0, N_NEW_POINTS)
    old_point_index = np.repeat(
        np.arange(N_OLD_POINTS, dtype=np.int64), OLD_REPLICAS
    )
    old = {
        "point_log_q": old_point_log_q,
        "log_q": old_point_log_q[old_point_index],
        "point_index": old_point_index,
    }
    new = {"point_log_q": new_point_log_q}
    point_log_q, row_log_q = _assemble_proposal_density(old, new, points)
    np.testing.assert_array_equal(
        point_log_q, np.concatenate([old_point_log_q, new_point_log_q])
    )
    np.testing.assert_array_equal(row_log_q, point_log_q[points])
    missing_point_log_q = dict(old)
    del missing_point_log_q["point_log_q"]
    try:
        _assemble_proposal_density(missing_point_log_q, new, points)
    except KeyError:
        pass
    else:
        raise AssertionError("missing old point_log_q was not rejected")
    corrupted_old = dict(old)
    corrupted_old["log_q"] = np.asarray(old["log_q"]).copy()
    corrupted_old["log_q"][0] += 1.0
    try:
        _assemble_proposal_density(corrupted_old, new, points)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted old row/point log_q mapping was accepted")
    theta = np.arange(256 * 5, dtype=np.float64).reshape(256, 5)
    cohort = np.asarray(["old"] * 128 + ["new"] * 128)
    first = _make_point_split(theta, cohort)
    second = _make_point_split(theta, cohort)
    assert np.array_equal(first, second)
    assert np.sum(first[:128]) == 13 and np.sum(first[128:]) == 13
    seeds = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    reordered = _canonical_seed_rows(seeds, ("y", "tau", "kappa"), ("y", "kappa", "tau"))
    np.testing.assert_array_equal(reordered, [[1, 3, 2], [4, 6, 5]])
    u = np.asarray([[0.0] * 5, [-1.0, -0.5, 0.25, 0.75, 1.5]])
    from scipy.special import ndtr

    theta_from_u = PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * ndtr(u)
    _validate_theta_u(theta_from_u, u, "self-test")
    corrupted_theta = theta_from_u.copy()
    corrupted_theta[0, 0] += 1.0e-8
    try:
        _validate_theta_u(corrupted_theta, u, "corruption null")
    except ValueError:
        pass
    else:
        raise AssertionError("theta/u corruption null was not rejected")
    broken = replicas.copy()
    broken[5] = 99
    try:
        _validate_flat_rows(points, broken, counts, offsets)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted replica table was accepted")
    assert not _find_declared_digest({"x": ["wrong"]}, ROUND2_COMPLETION_AGGREGATE)
    assert _find_declared_digest(
        {"nested": {"aggregate_sha256": ROUND2_COMPLETION_AGGREGATE}},
        ROUND2_COMPLETION_AGGREGATE,
    )
    print(
        "SELF-TEST PASS: 256 points, heterogeneous rows/df, old+new "
        "proposal-density interface, split, seed order, and corruption nulls"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("self-test", help="run dependency-free schema/null checks")

    init = sub.add_parser("init-round2", help="build the fixed g2-alm cache once")
    init.add_argument("--output", type=pathlib.Path, required=True)
    init.add_argument("--path-remap", action="append", default=[], metavar="OLD=NEW")
    init.add_argument("--preflight", action="store_true")
    init.add_argument("--overwrite", action="store_true")

    measure = sub.add_parser("measure-round2", help="measure one contiguous range of round-2 points")
    measure.add_argument("--batch-dir", type=pathlib.Path, required=True)
    measure.add_argument("--g2-cache", type=pathlib.Path, required=True)
    measure.add_argument("--output-dir", type=pathlib.Path, required=True)
    measure.add_argument("--point-start", type=int, required=True)
    measure.add_argument("--point-stop", type=int, required=True)
    measure.add_argument("--path-remap", action="append", default=[], metavar="OLD=NEW")
    measure.add_argument("--preflight", action="store_true")
    measure.add_argument("--overwrite", action="store_true")
    measure.add_argument("--skip-existing", action="store_true")

    assemble = sub.add_parser("assemble", help="join old 5-row and new 10-row cohorts")
    assemble.add_argument("--old-g2x-training-set", type=pathlib.Path, required=True)
    assemble.add_argument("--old-plan", type=pathlib.Path, required=True)
    assemble.add_argument("--batch-dir", type=pathlib.Path, required=True)
    assemble.add_argument("--round2-shard-dir", type=pathlib.Path, required=True)
    assemble.add_argument("--g2-cache", type=pathlib.Path, required=True)
    assemble.add_argument("--generating-point", type=pathlib.Path, default=g2contract.GENERATING_POINT_PATH)
    assemble.add_argument("--output", type=pathlib.Path, required=True)
    assemble.add_argument("--path-remap", action="append", default=[], metavar="OLD=NEW")
    assemble.add_argument("--preflight", action="store_true")
    assemble.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "self-test":
        return _run_self_test()
    if args.command == "init-round2":
        return _init_round2(args)
    if args.command == "measure-round2":
        return _measure_round2(args)
    if args.command == "assemble":
        return _assemble(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

