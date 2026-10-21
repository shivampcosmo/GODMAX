#!/usr/bin/env python3
"""Build matched gX + g2X products with complete assembly metadata.

No two-point training NPZ and no stored augmented maps are assumed. The existing
g2X training NPZ is the authoritative row table. For its exact 104 points and
five saved noise-seed triplets, this script measures ``gy/gkappa/gtau`` from the
same pasted maps and replays the same field noise through ``mock_sbi_common``.

The output contains the new gX rows and score contracts for three paired vectors:
``gy[14],g2y[14]``, ``gkappa[14],g2kappa[14]``, and
``gtau[14],g2tau[14]``. Each full 28 x 28 within-point covariance retains the
measured gX--g2X cross block. No covariance repair is permitted.

Example
-------

python notebooks/SBI_validate/07_mock_sbi_snle/build_mock_sbi_gx_g2x_pilot_assembly_contractfix_20260914.py \\
  --plan data/SBI_validate/mock_sbi/paste_plan/round_1/paste_plan.json \\
  --g2x-training-set data/SBI_validate/mock_sbi/paste_plan/round_1/training_set_g2x_rep5_104points.npz \\
  --g2x-contract data/SBI_validate/mock_sbi/paste_plan/round_1/g2x_pilot_contract_rep5.npz \\
  --output data/SBI_validate/mock_sbi/gx_g2x_pilot_rep5/gx_g2x_pilot_rep5_104points.npz \\
  --path-remap /mnt/ceph/users/spandey/ltu-godmax/GODMAX=/work/hdd/bdne/aacharya2/GODMAX
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys

_SBI_ROOT = _pl.Path(__file__).resolve().parents[1]
for _directory in (_SBI_ROOT, _SBI_ROOT / "common"):
    if str(_directory) not in _sys.path:
        _sys.path.insert(0, str(_directory))

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import yaml

import build_mock_sbi_g2x_pilot_contract as g2pilot
import run_mock_sbi_g2x_snle_pilot as g2run

import build_mock_sbi_g2x_trainingset as g2build

OUTPUT_SCHEMA = "godmax.mock_sbi.gx_g2x_pilot.v1"
ANALYSIS_STATUS = "exploratory_five_replica_joint_pilot"
PARAMETER_NAMES = g2pilot.PARAMETER_NAMES
EXPECTED_REPLICAS = g2pilot.EXPECTED_REPLICAS
GX_VECTOR_ORDER = "spectrum-major gy[14],gkappa[14],gtau[14]"
G2X_VECTOR_ORDER = g2pilot.VECTOR_ORDER
PAIR_SPECS: dict[str, dict[str, Any]] = {
    "gy_g2y": {
        "gx_name": "gy",
        "g2x_name": "g2y",
        "gx_indices": np.arange(0, 14, dtype=np.int64),
        "g2x_indices": np.arange(0, 14, dtype=np.int64),
    },
    "gkappa_g2kappa": {
        "gx_name": "gkappa",
        "g2x_name": "g2kappa",
        "gx_indices": np.arange(14, 28, dtype=np.int64),
        "g2x_indices": np.arange(14, 28, dtype=np.int64),
    },
    "gtau_g2tau": {
        "gx_name": "gtau",
        "g2x_name": "g2tau",
        "gx_indices": np.arange(28, 42, dtype=np.int64),
        "g2x_indices": np.arange(28, 42, dtype=np.int64),
    },
}


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Match the canonical contract identity used by the common loader."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_mock_contract_without_source_provenance(
    contract_path: pathlib.Path,
    msc: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """Load numerical mock inputs without checking historical source files.

    The supplied contract records source paths from a different SBI_validate
    layout (including modules now under ``common``).  Those files are not
    executed by this builder.  Validate the original contract's pinned file
    identity, clear ``source_hashes`` in an ephemeral copy, and pass that copy
    through the unchanged common loader.  This retains every numerical array,
    shape, ordering, prior, covariance-rank and Cholesky check while deliberately
    omitting all historical source-file checks.

    Neither the original contract nor any common module is modified.  The
    returned policy record is written into the output manifest.
    """

    import three_probe_inference_contract as registry

    contract_path = contract_path.expanduser().resolve()
    registered_path = pathlib.Path(registry.MOCK_CONTRACT_PATH).resolve()
    if contract_path != registered_path:
        raise ValueError(
            "the compatibility path is restricted to the registered mock contract"
        )

    registered_contract_sha = str(registry.MOCK_CONTRACT_SHA256)
    observed_contract_sha = msc.sha256_file(contract_path)
    if observed_contract_sha != registered_contract_sha:
        raise ValueError(
            "mock contract file hash changed before applying the source waiver: "
            f"{observed_contract_sha} != {registered_contract_sha}"
        )

    with contract_path.open() as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("mock contract must decode to a mapping")
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("mock contract source_hashes must be a mapping")
    compatible_payload = copy.deepcopy(dict(payload))
    compatible_payload["source_hashes"] = {}

    original_registry_path = registry.MOCK_CONTRACT_PATH
    original_registry_sha = registry.MOCK_CONTRACT_SHA256
    with tempfile.TemporaryDirectory(prefix="godmax_mock_contract_") as directory:
        temporary_contract = pathlib.Path(directory) / contract_path.name
        temporary_contract.write_text(
            yaml.safe_dump(compatible_payload, sort_keys=True)
        )
        registry.MOCK_CONTRACT_PATH = temporary_contract
        registry.MOCK_CONTRACT_SHA256 = msc.sha256_file(temporary_contract)
        try:
            checked = registry.load_training_contract(temporary_contract)
        finally:
            registry.MOCK_CONTRACT_PATH = original_registry_path
            registry.MOCK_CONTRACT_SHA256 = original_registry_sha

    checked = dataclasses.replace(
        checked,
        contract_path=contract_path,
        contract_sha256=_canonical_json_sha256(payload),
    )
    exception = {
        "status": "historical_source_file_checks_skipped_by_user_request",
        "analysis_scope": "exploratory_five_replica_joint_pilot_only",
        "contract_path": str(contract_path),
        "contract_file_sha256": observed_contract_sha,
        "strict_source_validation": "not_attempted_by_user_request",
        "declared_sources_not_checked": {
            str(name): str(digest) for name, digest in source_hashes.items()
        },
        "declared_source_files_were_executed": False,
        "waived_checks": [
            "existence_and_content_hashes_of_all_declared_historical_sources"
        ],
        "checks_retained": [
            "registered_mock_contract_file_sha256",
            "observation_and_covariance_array_hashes",
            "array_shapes_and_vector_order",
            "production_covariance_cholesky_window_identity",
            "covariance_rank_and_cholesky_reconstruction",
            "sampled_and_fixed_parameter_contracts",
        ],
        "validation_route": (
            "unchanged common.three_probe_inference_contract.load_training_contract "
            "on an ephemeral payload with source_hashes cleared"
        ),
    }
    print(
        "      WARNING: historical source-file checks are disabled for this "
        "exploratory pilot; the pinned contract and all numerical checks passed",
        flush=True,
    )
    return checked, exception


def _atomic_write_text(path: pathlib.Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_savez(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _json_from_npz(value: np.ndarray) -> dict[str, Any]:
    parsed = json.loads(str(np.asarray(value).item()))
    if not isinstance(parsed, dict):
        raise ValueError("manifest_json did not decode to an object")
    return parsed


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: Any | None = None,
    finite: bool = True,
) -> np.ndarray:
    if name not in arrays:
        raise KeyError(f"g2X training set is missing array {name!r}")
    value = np.asarray(arrays[name], dtype=dtype)
    if value.shape != shape:
        raise ValueError(
            f"g2X training array {name!r} has shape {value.shape}; expected {shape}"
        )
    if finite and value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
        raise ValueError(f"g2X training array {name!r} is non-finite")
    return value


def _load_g2x_rows(path: pathlib.Path, msc: Any) -> dict[str, Any]:
    """Load the strict g2X product plus its saved row/source metadata."""

    training = g2pilot._load_training_set(path, msc)
    with np.load(training["path"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    n_rows = int(training["n_rows"])
    n_points = int(training["n_points"])
    replicas = int(training["replicas"])
    plan_position = _require_array(
        arrays, "plan_position", (n_rows,), dtype=np.int64
    )
    plan_entry_index = _require_array(
        arrays, "plan_entry_index", (n_rows,), dtype=np.int64
    )
    noise_seeds = _require_array(
        arrays, "noise_seeds", (n_rows, 3), dtype=np.uint32
    )
    row_log_q = _require_array(
        arrays, "log_q", (n_rows,), dtype=np.float64, finite=False
    )
    point_log_q = _require_array(
        arrays, "point_log_q", (n_points,), dtype=np.float64, finite=False
    )
    point_bank_indices = _require_array(
        arrays,
        "point_bank_indices",
        (n_points, replicas),
        dtype=np.int64,
    )

    if np.any(training["bank_indices"] < 0):
        raise ValueError("g2X row bank_indices contains a negative value")
    if not np.array_equal(
        np.unique(training["replica"]), np.arange(replicas, dtype=np.int64)
    ):
        raise ValueError("g2X row replicas are not exactly 0..4")

    point_plan_position = np.empty(n_points, dtype=np.int64)
    point_entry_index = np.empty(n_points, dtype=np.int64)
    for point in range(n_points):
        rows = np.flatnonzero(training["point_index"] == point)
        ordered_rows = rows[np.argsort(training["replica"][rows])]
        if not np.array_equal(
            training["replica"][ordered_rows],
            np.arange(replicas, dtype=np.int64),
        ):
            raise ValueError(f"g2X point {point} has a noncanonical replica table")
        if np.unique(plan_position[rows]).size != 1:
            raise ValueError(f"g2X point {point} spans multiple plan positions")
        if np.unique(plan_entry_index[rows]).size != 1:
            raise ValueError(f"g2X point {point} spans multiple entry indices")
        point_plan_position[point] = plan_position[rows[0]]
        point_entry_index[point] = plan_entry_index[rows[0]]
        if not np.array_equal(
            training["bank_indices"][ordered_rows], point_bank_indices[point]
        ):
            raise ValueError(
                f"g2X row and point bank-index tables disagree at point {point}"
            )
        expected_log_q = np.full(replicas, point_log_q[point], dtype=np.float64)
        if not np.array_equal(
            row_log_q[ordered_rows], expected_log_q, equal_nan=True
        ):
            raise ValueError(
                f"g2X row and point log_q tables disagree at point {point}"
            )
    if np.unique(point_plan_position).size != n_points:
        raise ValueError("two g2X points refer to the same full-plan position")

    bank = training["manifest"].get("bank", {})
    bank_size = int(bank.get("size", -1))
    if bank_size <= 0 or np.any(training["bank_indices"] >= bank_size):
        raise ValueError("g2X bank-size/index contract is invalid")
    if tuple(bank.get("seed_namespace", ())) != tuple(msc.NAMESPACE_TRAINING):
        raise ValueError("g2X seed namespace differs from mock_sbi_common")
    if tuple(bank.get("seed_field_order", ())) != tuple(msc.NOISE_FIELDS):
        raise ValueError("g2X noise-seed field order differs from mock_sbi_common")

    seed_records = msc.noise_bank_seeds(bank_size, msc.NAMESPACE_TRAINING)
    seed_bank = np.asarray(
        [[record[name] for name in msc.NOISE_FIELDS] for record in seed_records],
        dtype=np.uint32,
    )
    expected_seeds = seed_bank[training["bank_indices"]]
    if not np.array_equal(noise_seeds, expected_seeds):
        bad = int(np.flatnonzero(np.any(noise_seeds != expected_seeds, axis=1))[0])
        raise ValueError(
            f"g2X row {bad} noise_seeds do not match its saved bank index"
        )

    source_records = training["manifest"].get("source_maps")
    if not isinstance(source_records, list) or len(source_records) != n_points:
        raise ValueError("g2X manifest has no one-to-one source_maps table")
    source_by_point: dict[int, dict[str, Any]] = {}
    for record in source_records:
        if not isinstance(record, dict):
            raise ValueError("g2X source_maps contains a non-object record")
        point = int(record.get("point_index", -1))
        if point in source_by_point or not 0 <= point < n_points:
            raise ValueError("g2X source_maps has a duplicate/invalid point_index")
        source_by_point[point] = record
    if set(source_by_point) != set(range(n_points)):
        raise ValueError("g2X source_maps does not cover every output point")

    return {
        **training,
        "arrays": arrays,
        "plan_position": plan_position,
        "plan_entry_index": plan_entry_index,
        "point_plan_position": point_plan_position,
        "point_entry_index": point_entry_index,
        "noise_seeds": noise_seeds,
        "log_q": row_log_q,
        "point_log_q": point_log_q,
        "point_bank_indices": point_bank_indices,
        "source_by_point": source_by_point,
    }


def _check_estimator_identity(
    training: Mapping[str, Any], context: Any
) -> dict[str, str]:
    identity = {
        "noise_contract_sha256": str(context.contract_sha256),
        "workspace_sha256": str(context.workspace_sha256),
        "mask_array_sha256": str(context.mask_sha256),
        "fixed_galaxy_alm_sha256": str(context.galaxy_alm_sha256),
    }
    registered = training["manifest"].get("estimator", {})
    for name, value in identity.items():
        if str(registered.get(name)) != value:
            raise ValueError(
                f"g2X training estimator {name}={registered.get(name)!r}, "
                f"but the current frozen estimator gives {value!r}"
            )
    if not np.array_equal(training["band_edges"], context.band_edges):
        raise ValueError("g2X training and frozen estimator band edges differ")
    if not np.array_equal(training["effective_ell"], context.effective_ell):
        raise ValueError("g2X training and frozen estimator effective ell differ")
    return identity


def _validate_plan_and_sources(
    plan_path: pathlib.Path,
    training: Mapping[str, Any],
    context: Any,
    remaps: Sequence[tuple[pathlib.Path, pathlib.Path]],
    msc: Any,
) -> tuple[dict[str, Any], list[tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]]]]:
    """Match every saved output point to its original full-plan map source."""

    plan_path = plan_path.expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"paste plan does not exist: {plan_path}")
    plan = g2build._load_and_validate_plan(plan_path)
    source_plan = training["manifest"].get("source_plan", {})
    if msc.sha256_file(plan_path) != str(source_plan.get("sha256")):
        raise ValueError("--plan is not the exact plan that built the g2X training set")
    if int(plan.get("nside", -1)) != int(msc.NSIDE):
        raise ValueError(
            f"plan nside {plan.get('nside')} differs from frozen nside {msc.NSIDE}"
        )
    if len(plan["entries"]) != int(training["manifest"].get("n_plan_points", -1)):
        raise ValueError("full plan length differs from the g2X training manifest")

    prepared: list[
        tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    expected_shape = tuple(context.mask.shape)
    for point in range(int(training["n_points"])):
        position = int(training["point_plan_position"][point])
        if not 0 <= position < len(plan["entries"]):
            raise ValueError(f"g2X point {point} has invalid plan position {position}")
        entry = plan["entries"][position]
        record = training["source_by_point"][point]
        if int(training["point_entry_index"][point]) != int(entry["index"]):
            raise ValueError(f"g2X point {point} entry index differs from its plan row")
        expected_theta = np.asarray(
            [entry["theta"][name] for name in PARAMETER_NAMES], dtype=np.float64
        )
        if not np.array_equal(training["point_theta"][point], expected_theta):
            raise ValueError(f"g2X point {point} theta differs from its plan row")
        exact_fields = {
            "point_index": point,
            "plan_position": position,
            "entry_index": int(entry["index"]),
            "run_name": str(entry["run_name"]),
            "config_sha256": str(entry["config_sha256"]),
            "theta_sha256": str(entry["theta_sha256"]),
        }
        for name, value in exact_fields.items():
            if record.get(name) != value:
                raise ValueError(
                    f"g2X source record {point} {name}={record.get(name)!r}; "
                    f"expected {value!r}"
                )

        info = g2build._entry_map_info(
            entry, msc.REPO_ROOT, remaps, msc
        )
        if int(info["nside"]) != int(msc.NSIDE):
            raise ValueError(f"point {point} map nside differs from the estimator")
        source_kind = str(record.get("source_kind"))
        if source_kind == "combined_hdf5":
            path = pathlib.Path(info["combined_map_path"])
            if not g2build._is_nonempty_file(path):
                raise FileNotFoundError(f"saved combined source is missing: {path}")
            g2build._validate_combined_map_layout(info, expected_shape, msc)
            observed_sha = msc.sha256_file(path)
            if observed_sha != str(record.get("map_sha256")):
                raise ValueError(
                    f"combined source changed for point {point}: {observed_sha} "
                    f"!= {record.get('map_sha256')}"
                )
            info["source_kind"] = "combined"
        elif source_kind == "strict_split_sum_in_memory":
            _, metadata = g2build._process_split_set(
                info,
                expected_shape,
                msc,
                load_maps=False,
                # The callee remains fail-closed: True admits only the exact
                # registered combine-only helper hash pair.
                allow_known_helper_sha_mix=True,
            )
            for name in (
                "source_kind",
                "num_splits",
                "split_strategy",
                "n_input_halos",
                "pasting_helper_sha256",
            ):
                if metadata.get(name) != record.get(name):
                    raise ValueError(
                        f"split source metadata {name!r} changed at point {point}"
                    )
            info["source_kind"] = "splits"
        else:
            raise ValueError(
                f"g2X point {point} has unsupported saved source kind {source_kind!r}"
            )
        prepared.append((entry, info, record))
    return plan, prepared


def _load_signal_maps(
    info: Mapping[str, Any],
    record: Mapping[str, Any],
    context: Any,
    msc: Any,
) -> dict[str, np.ndarray]:
    """Reload exactly the source representation recorded by the g2X builder."""

    if info["source_kind"] == "combined":
        maps = msc.read_paste_maps(pathlib.Path(info["combined_map_path"]))
    elif info["source_kind"] == "splits":
        maps, _ = g2build._process_split_set(
            info,
            tuple(context.mask.shape),
            msc,
            load_maps=True,
            # The callee rejects every pair except the registered combine-only
            # helper hashes, so this does not waive arbitrary source changes.
            allow_known_helper_sha_mix=True,
        )
        assert maps is not None
    else:
        raise ValueError(f"unsupported source kind {info['source_kind']!r}")
    observed = msc.sha256_array(
        *(np.asarray(maps[field], dtype=np.float64) for field in msc.NOISE_FIELDS)
    )
    expected = str(record.get("signal_map_arrays_float64_sha256"))
    if observed != expected:
        raise ValueError(
            f"signal-map arrays changed for {record.get('run_name')}: "
            f"{observed} != {expected}"
        )
    return {
        field: np.asarray(maps[field], dtype=np.float64)
        for field in msc.NOISE_FIELDS
    }


def _assemble_gx_rows(
    point_signal: np.ndarray,
    point_index: np.ndarray,
    noise_seeds: np.ndarray,
    noise_provider: Callable[[tuple[int, int, int]], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Add one cached, seed-addressed gX noise vector to each point signal."""

    point_signal = np.asarray(point_signal, dtype=np.float64)
    point_index = np.asarray(point_index, dtype=np.int64)
    noise_seeds = np.asarray(noise_seeds, dtype=np.uint32)
    if point_signal.ndim != 2 or point_signal.shape[1] != 42:
        raise ValueError("point_signal must have shape (n_points, 42)")
    if point_index.shape != (noise_seeds.shape[0],) or noise_seeds.shape[1:] != (3,):
        raise ValueError("row point/seed tables have inconsistent shapes")
    if np.any(point_index < 0) or np.any(point_index >= point_signal.shape[0]):
        raise ValueError("row point_index is outside point_signal")

    cache: dict[tuple[int, int, int], np.ndarray] = {}
    noise_rows = np.empty((noise_seeds.shape[0], 42), dtype=np.float64)
    for row, raw in enumerate(noise_seeds):
        key = tuple(int(value) for value in raw)
        if key not in cache:
            value = np.asarray(noise_provider(key), dtype=np.float64)
            if value.shape != (42,) or not np.all(np.isfinite(value)):
                raise RuntimeError(
                    f"noise provider returned invalid vector {value.shape} for {key}"
                )
            cache[key] = value
        noise_rows[row] = cache[key]
    rows = point_signal[point_index] + noise_rows
    if not np.all(np.isfinite(rows)):
        raise RuntimeError("assembled gX rows contain non-finite values")
    return rows, noise_rows, len(cache)


def _load_noise_vector_bank(
    path: pathlib.Path,
    training: Mapping[str, Any],
    context: Any,
    msc: Any,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Load an optional Stage-5 gX bank, accepting it only after full identity checks."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"requested two-point noise bank is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"vectors", "seeds", "seed_field_order", "report_json"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"two-point noise bank is missing arrays {missing}")
        vectors = np.asarray(archive["vectors"], dtype=np.float64)
        seeds = np.asarray(archive["seeds"], dtype=np.uint32)
        order = tuple(str(value) for value in archive["seed_field_order"])
        report = json.loads(str(np.asarray(archive["report_json"]).item()))
    bank_size = int(training["manifest"]["bank"]["size"])
    if vectors.shape != (bank_size, 42) or seeds.shape != (bank_size, 3):
        raise ValueError(
            f"two-point noise bank has shapes {vectors.shape}, {seeds.shape}; "
            f"expected {(bank_size, 42)}, {(bank_size, 3)}"
        )
    if order != tuple(msc.NOISE_FIELDS):
        raise ValueError(f"two-point bank seed order {order} != {msc.NOISE_FIELDS}")
    expected_records = msc.noise_bank_seeds(bank_size, msc.NAMESPACE_TRAINING)
    expected_seeds = np.asarray(
        [[record[field] for field in msc.NOISE_FIELDS] for record in expected_records],
        dtype=np.uint32,
    )
    if not np.array_equal(seeds, expected_seeds):
        raise ValueError("two-point noise-bank seeds differ from deterministic Stage-5 seeds")
    required_report = {
        "namespace": "training",
        "namespace_entropy": list(msc.NAMESPACE_TRAINING),
        "noise_contract_sha256": str(context.contract_sha256),
        "workspace_sha256": str(context.workspace_sha256),
        "mask_array_sha256": str(context.mask_sha256),
        "vector_order": str(msc.VECTOR_ORDER),
        "vectors_sha256": msc.sha256_array(vectors),
    }
    for name, value in required_report.items():
        if report.get(name) != value:
            raise ValueError(
                f"two-point noise-bank report {name}={report.get(name)!r}; "
                f"expected {value!r}"
            )
    report_path = path.with_suffix(".report.json")
    if not report_path.is_file():
        raise FileNotFoundError(f"two-point noise-bank report is missing: {report_path}")
    if json.loads(report_path.read_text()) != report:
        raise ValueError("two-point noise-bank NPZ and report sidecar differ")
    return {
        tuple(int(value) for value in seeds[index]): vectors[index]
        for index in range(bank_size)
    }


def _joint_payload(
    name: str,
    gx_rows: np.ndarray,
    g2x_rows: np.ndarray,
    point_index: np.ndarray,
    point_u: np.ndarray,
    gx_observation: np.ndarray,
    g2x_observation: np.ndarray,
    u_reference: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one paired 28-vector covariance, affine response and score."""

    if name not in PAIR_SPECS:
        raise ValueError(f"unknown pair {name!r}")
    spec = PAIR_SPECS[name]
    gx_indices = np.asarray(spec["gx_indices"], dtype=np.int64)
    g2x_indices = np.asarray(spec["g2x_indices"], dtype=np.int64)
    rows = np.concatenate(
        [gx_rows[:, gx_indices], g2x_rows[:, g2x_indices]], axis=1
    )
    observation = np.concatenate(
        [gx_observation[gx_indices], g2x_observation[g2x_indices]]
    )
    n_points = int(np.max(point_index)) + 1
    point_means, residuals, covariance, dof = g2pilot._within_point_covariance(
        rows, point_index, n_points
    )
    cholesky = np.linalg.cholesky(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    response = g2pilot._fit_affine_response(point_u, point_means)
    intercept = np.asarray(response["intercept"], dtype=np.float64)
    jacobian = np.asarray(response["jacobian"], dtype=np.float64)
    reference = intercept + jacobian @ np.asarray(u_reference, dtype=np.float64)
    score = g2pilot._score_payload(
        covariance, jacobian, reference, np.arange(28, dtype=np.int64)
    )
    whitened_observation = np.linalg.solve(
        np.asarray(score["cholesky"], dtype=np.float64), observation - reference
    )
    observed_score = np.asarray(score["operator"]) @ whitened_observation
    cross_covariance = np.asarray(covariance[:14, 14:], dtype=np.float64)
    scale = np.sqrt(np.diag(covariance))
    cross_correlation = cross_covariance / np.outer(scale[:14], scale[14:])
    group_sums = np.stack(
        [residuals[point_index == point].sum(axis=0) for point in range(n_points)]
    )
    fit_residual = np.asarray(response["fit_residual"], dtype=np.float64)
    fit_whitened = np.linalg.solve(cholesky, fit_residual.T).T

    arrays: dict[str, np.ndarray] = {
        "x": rows,
        "data_vector": observation,
        "point_means": point_means,
        "within_point_residuals": residuals,
        "covariance": covariance,
        "cholesky": cholesky,
        "cross_covariance": cross_covariance,
        "cross_correlation": cross_correlation,
        "affine_intercept": intercept,
        "affine_jacobian": jacobian,
        "affine_reference_prediction": reference,
        "affine_fit_residual": fit_residual,
        "finite_point_mask": np.asarray(response["finite_point_mask"], dtype=bool),
        "indices": np.arange(28, dtype=np.int64),
    }
    for key in (
        "covariance",
        "cholesky",
        "jacobian",
        "whitened_jacobian",
        "gram",
        "posterior_metric",
        "operator",
        "noise_covariance",
        "reference_prediction",
    ):
        arrays[f"score_{key}"] = np.asarray(score[key])
    arrays["observed_score"] = np.asarray(observed_score, dtype=np.float64)

    metadata = {
        "gx_name": str(spec["gx_name"]),
        "g2x_name": str(spec["g2x_name"]),
        "vector_order": f"{spec['gx_name']}[14],{spec['g2x_name']}[14]",
        "gx_source_indices": gx_indices.tolist(),
        "g2x_source_indices": g2x_indices.tolist(),
        "dimension": 28,
        "covariance_degrees_of_freedom": int(dof),
        "positive_definite_without_repair": True,
        "jitter": None,
        "eigenvalue_floor": None,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "maximum_abs_group_residual_sum": float(np.max(np.abs(group_sums))),
        "maximum_abs_cross_correlation": float(np.max(np.abs(cross_correlation))),
        "response_model": "global affine OLS of per-point joint mean on [1,u]",
        "response_design_rank": int(response["rank"]),
        "response_singular_values": np.asarray(response["singular_values"]).tolist(),
        "response_rank_after_whitening": int(score["response_rank"]),
        "whitened_fit_residual_rms_per_component": float(
            np.sqrt(np.mean(fit_whitened * fit_whitened))
        ),
        "observed_score_norm": float(np.linalg.norm(observed_score)),
    }
    return arrays, metadata


def _run_self_test() -> int:
    rng = np.random.default_rng(20260904)
    n_points, replicas = 40, EXPECTED_REPLICAS
    point_u = rng.normal(size=(n_points, 5))
    gx_intercept = rng.normal(size=42)
    gx_jacobian = rng.normal(size=(42, 5))
    g2_intercept = rng.normal(size=42)
    g2_jacobian = rng.normal(size=(42, 5))
    gx_signal = gx_intercept[None, :] + point_u @ gx_jacobian.T
    g2_signal = g2_intercept[None, :] + point_u @ g2_jacobian.T
    point_index = np.repeat(np.arange(n_points), replicas)
    replica = np.tile(np.arange(replicas), n_points)
    noise_seeds = np.column_stack(
        [
            np.arange(1000, 1000 + n_points * replicas, dtype=np.uint32),
            np.arange(2000, 2000 + n_points * replicas, dtype=np.uint32),
            np.arange(3000, 3000 + n_points * replicas, dtype=np.uint32),
        ]
    )
    # Exercise cache reuse across points without introducing a within-point duplicate.
    noise_seeds[replicas] = noise_seeds[0]

    def vector_from_seed(key: tuple[int, int, int], offset: int = 0) -> np.ndarray:
        local = np.random.default_rng(np.random.SeedSequence((*key, offset)))
        return local.normal(size=42)

    gx_rows, gx_noise, unique = _assemble_gx_rows(
        gx_signal, point_index, noise_seeds, vector_from_seed
    )
    assert unique == n_points * replicas - 1
    np.testing.assert_array_equal(gx_rows, gx_signal[point_index] + gx_noise)
    assert np.array_equal(gx_noise[0], gx_noise[replicas])

    independent = np.stack(
        [vector_from_seed(tuple(int(v) for v in row), 1) for row in noise_seeds]
    )
    g2_rows = g2_signal[point_index] + 0.4 * gx_noise + independent
    gx_observation = gx_intercept.copy()
    g2_observation = g2_intercept.copy()
    for name in PAIR_SPECS:
        arrays, metadata = _joint_payload(
            name,
            gx_rows,
            g2_rows,
            point_index,
            point_u,
            gx_observation,
            g2_observation,
            np.zeros(5),
        )
        assert arrays["x"].shape == (n_points * replicas, 28)
        assert arrays["covariance"].shape == (28, 28)
        assert arrays["score_operator"].shape == (5, 28)
        assert arrays["observed_score"].shape == (5,)
        assert metadata["covariance_degrees_of_freedom"] == n_points * (replicas - 1)
        np.linalg.cholesky(arrays["covariance"])
        if not np.any(np.abs(arrays["cross_covariance"]) > 0.0):
            raise AssertionError(f"{name} synthetic cross block vanished")
        gx_indices = np.asarray(PAIR_SPECS[name]["gx_indices"])
        g2_indices = np.asarray(PAIR_SPECS[name]["g2x_indices"])
        np.testing.assert_array_equal(
            arrays["x"],
            np.concatenate(
                [gx_rows[:, gx_indices], g2_rows[:, g2_indices]], axis=1
            ),
        )
    if not np.array_equal(replica[:replicas], np.arange(replicas)):
        raise AssertionError("synthetic row table lost canonical replica order")

    # Regression for the all-256 assembler interface: the strict pilot loader
    # intentionally returns only the fields it uses, so this adapter must copy
    # the proposal-density arrays from the original training NPZ itself.
    interface_points = 3
    interface_rows = interface_points * replicas
    interface_point_index = np.repeat(
        np.arange(interface_points, dtype=np.int64), replicas
    )
    interface_replica = np.tile(
        np.arange(replicas, dtype=np.int64), interface_points
    )
    interface_bank_indices = np.arange(interface_rows, dtype=np.int64)
    interface_point_log_q = np.asarray([-3.0, -2.0, -1.0])

    class _FakeMSC:
        NAMESPACE_TRAINING = (11, 22)
        NOISE_FIELDS = ("y", "tau", "kappa")

        @staticmethod
        def noise_bank_seeds(size: int, namespace: Sequence[int]) -> list[dict[str, int]]:
            assert tuple(namespace) == _FakeMSC.NAMESPACE_TRAINING
            return [
                {
                    field: int(1000 + 10 * index + column)
                    for column, field in enumerate(_FakeMSC.NOISE_FIELDS)
                }
                for index in range(size)
            ]

    seed_records = _FakeMSC.noise_bank_seeds(32, _FakeMSC.NAMESPACE_TRAINING)
    seed_bank = np.asarray(
        [
            [record[field] for field in _FakeMSC.NOISE_FIELDS]
            for record in seed_records
        ],
        dtype=np.uint32,
    )
    with tempfile.TemporaryDirectory(prefix="godmax_g2x_loader_selftest_") as directory:
        training_path = pathlib.Path(directory) / "training.npz"
        _atomic_savez(
            training_path,
            {
                "plan_position": interface_point_index,
                "plan_entry_index": interface_point_index,
                "noise_seeds": seed_bank[interface_bank_indices],
                "log_q": interface_point_log_q[interface_point_index],
                "point_log_q": interface_point_log_q,
                "point_bank_indices": interface_bank_indices.reshape(
                    interface_points, replicas
                ),
            },
        )
        synthetic_training = {
            "path": training_path,
            "n_rows": interface_rows,
            "n_points": interface_points,
            "replicas": replicas,
            "bank_indices": interface_bank_indices,
            "replica": interface_replica,
            "point_index": interface_point_index,
            "manifest": {
                "bank": {
                    "size": 32,
                    "seed_namespace": list(_FakeMSC.NAMESPACE_TRAINING),
                    "seed_field_order": list(_FakeMSC.NOISE_FIELDS),
                },
                "source_maps": [
                    {"point_index": point} for point in range(interface_points)
                ],
            },
        }
        original_loader = g2pilot._load_training_set
        g2pilot._load_training_set = lambda path, msc: synthetic_training
        try:
            loaded = _load_g2x_rows(training_path, _FakeMSC)
        finally:
            g2pilot._load_training_set = original_loader
        np.testing.assert_array_equal(loaded["point_log_q"], interface_point_log_q)
        np.testing.assert_array_equal(
            loaded["log_q"], interface_point_log_q[interface_point_index]
        )
    print(
        "SELF-TEST PASS: exact seed-keyed row pairing, cache reuse, canonical "
        "gy/g2y-gkappa/g2kappa-gtau/g2tau slices, full cross covariance, "
        "unrepaired Cholesky factors, affine responses, score shapes, and the "
        "old row/point proposal-density loader interface"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", type=pathlib.Path)
    parser.add_argument("--g2x-training-set", type=pathlib.Path)
    parser.add_argument("--g2x-contract", type=pathlib.Path)
    parser.add_argument(
        "--noise-bank",
        type=pathlib.Path,
        default=None,
        help=(
            "optional existing noise_bank_training.npz; it is used only after "
            "its seeds, vector hash, estimator and JSON report all validate. "
            "Without it, the exact saved seeds are replayed through mock_sbi_common"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        metavar="OLD_PREFIX=NEW_PREFIX",
        help="rewrite absolute paths embedded in the plan and frozen contracts",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate identities and source layouts, then exit without measurements/writes",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    for name in ("plan", "g2x_training_set", "g2x_contract", "output"):
        if getattr(args, name) is None:
            parser.error(
                f"--{name.replace('_', '-')} is required unless --self-test is used"
            )
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    sidecar = output.with_suffix(".json")
    if not args.preflight and not args.overwrite:
        existing = [path for path in (output, sidecar) if path.exists()]
        if existing:
            parser.error(
                "output already exists; pass --overwrite to replace it atomically: "
                + ", ".join(str(path) for path in existing)
            )

    started = time.time()
    import mock_sbi_common as msc
    from three_probe_inference_contract import (
        MOCK_CONTRACT_PATH,
        VECTOR_ORDER as REGISTERED_GX_VECTOR_ORDER,
    )

    remaps = g2build._parse_path_remaps(args.path_remap)
    print("[1/6] loading the saved g2X rows and their pilot contract ...", flush=True)
    training = _load_g2x_rows(args.g2x_training_set, msc)
    g2_contract = g2run._load_contract(args.g2x_contract, training, msc)
    g2_arrays = g2_contract["arrays"]
    registered_common_sha = str(
        g2_contract["manifest"]
        .get("source_code", {})
        .get("mock_sbi_common", {})
        .get("sha256", "")
    )
    current_common_sha = msc.sha256_file(pathlib.Path(msc.__file__).resolve())
    common_source_validation = {
        "status": "not_enforced_by_user_request",
        "g2x_contract_declared_sha256": registered_common_sha,
        "current_sha256": current_common_sha,
        "hashes_match": registered_common_sha == current_common_sha,
    }
    if not common_source_validation["hashes_match"]:
        print(
            "      WARNING: ignoring the g2X contract's historical "
            "mock_sbi_common.py source hash",
            flush=True,
        )
    if int(training["replicas"]) != EXPECTED_REPLICAS:
        raise ValueError("this joint pilot requires exactly five saved replicas")
    print(
        f"      {training['n_points']} points x {training['replicas']} replicas "
        f"= {training['n_rows']} exactly paired rows",
        flush=True,
    )

    print("[2/6] loading the frozen estimator and registered observations ...", flush=True)
    context = g2build._load_estimator_context(msc, remaps)
    estimator_identity = _check_estimator_identity(training, context)
    mock_contract, mock_contract_source_exception = (
        _load_mock_contract_without_source_provenance(MOCK_CONTRACT_PATH, msc)
    )
    if REGISTERED_GX_VECTOR_ORDER != GX_VECTOR_ORDER or msc.VECTOR_ORDER != GX_VECTOR_ORDER:
        raise ValueError("registered gX vector order differs from this joint builder")
    if tuple(item["name"] for item in mock_contract.sampled_parameters) != PARAMETER_NAMES:
        raise ValueError("mock observation parameter order differs from g2X")
    prior_low = np.asarray(
        [item["prior"]["low"] for item in mock_contract.sampled_parameters],
        dtype=np.float64,
    )
    prior_high = np.asarray(
        [item["prior"]["high"] for item in mock_contract.sampled_parameters],
        dtype=np.float64,
    )
    if not np.array_equal(prior_low, g2pilot.PRIOR_LOW) or not np.array_equal(
        prior_high, g2pilot.PRIOR_HIGH
    ):
        raise ValueError("mock observation and g2X pilot prior boxes differ")
    if not np.array_equal(mock_contract.band_edges, training["band_edges"]):
        raise ValueError("mock observation and g2X training band edges differ")
    if not np.array_equal(mock_contract.effective_ell, training["effective_ell"]):
        raise ValueError("mock observation and g2X effective ell differ")
    if g2_contract["manifest"].get("observation", {}).get("kind") != (
        "frozen_reference_paste_noiseless_g2x"
    ):
        raise ValueError("g2X contract observation is not the registered noiseless paste")
    g2_estimator = g2_contract["manifest"].get("estimator", {})
    for name, value in estimator_identity.items():
        if str(g2_estimator.get(name)) != value:
            raise ValueError(f"g2X pilot contract estimator {name} differs")
    reference_path, reference_sha, _ = g2build._resolve_and_verify_frozen_galaxy_map(
        msc, remaps
    )
    if str(g2_contract["manifest"]["observation"].get("map_sha256")) != reference_sha:
        raise ValueError("g2X observation does not use the frozen reference paste")
    gx_observation = np.asarray(mock_contract.data_vector, dtype=np.float64)
    g2x_observation = np.asarray(g2_arrays["data_vector"], dtype=np.float64)
    theta_reference = np.asarray(g2_arrays["theta_reference"], dtype=np.float64)
    u_reference = np.asarray(g2_arrays["u_reference"], dtype=np.float64)
    if gx_observation.shape != (42,) or g2x_observation.shape != (42,):
        raise ValueError("registered observations are not both 42-vectors")
    print(
        "      registered noiseless gX and g2X observations; common 14-band estimator",
        flush=True,
    )

    print("[3/6] matching all saved points to their exact plan/map sources ...", flush=True)
    plan, prepared = _validate_plan_and_sources(
        args.plan, training, context, remaps, msc
    )
    n_combined = sum(info["source_kind"] == "combined" for _, info, _ in prepared)
    n_split_sets = len(prepared) - n_combined
    bank_vectors = None
    if args.noise_bank is not None:
        bank_vectors = _load_noise_vector_bank(
            args.noise_bank, training, context, msc
        )
        noise_route = f"verified Stage-5 vector bank {args.noise_bank.expanduser().resolve()}"
    else:
        noise_route = "direct replay through mock_sbi_common.noise_vector"
    print(
        f"      {n_combined} combined maps, {n_split_sets} complete split sets; "
        f"gX noise route: {noise_route}",
        flush=True,
    )

    if args.preflight:
        print("\nPREFLIGHT PASS")
        print(f"      plan: {args.plan.expanduser().resolve()}")
        print(f"      g2X training set: {training['path']}")
        print(f"      g2X pilot contract: {g2_contract['path']}")
        print(f"      points/rows: {training['n_points']}/{training['n_rows']}")
        print(f"      map sources: {n_combined} combined, {n_split_sets} split sets")
        print(f"      noise route: {noise_route}")
        print(f"      output to create: {output}")
        print(f"      manifest to create: {sidecar}")
        print("      no files were written")
        return 0

    print("[4/6] measuring noiseless gy/gkappa/gtau once per saved point ...", flush=True)
    point_signal = np.empty((training["n_points"], 42), dtype=np.float64)
    for point, (entry, info, record) in enumerate(prepared):
        maps = _load_signal_maps(info, record, context, msc)
        point_signal[point] = msc.measure_map_vector(maps, context)
        if not np.all(np.isfinite(point_signal[point])):
            raise RuntimeError(f"point {point} produced a non-finite gX signal vector")
        print(
            f"      point {point + 1:3d}/{training['n_points']}: {entry['run_name']}",
            flush=True,
        )
        del maps
    if len({msc.sha256_array(row) for row in point_signal}) != training["n_points"]:
        raise RuntimeError("two distinct g2X design points produced the same gX signal row")

    print("[5/6] replaying matched noise and building three full joint contracts ...", flush=True)
    replayed = 0

    def noise_provider(key: tuple[int, int, int]) -> np.ndarray:
        nonlocal replayed
        if bank_vectors is not None:
            try:
                return bank_vectors[key]
            except KeyError as error:
                raise KeyError(f"saved g2X seed triplet is absent from noise bank: {key}") from error
        seeds = dict(zip(msc.NOISE_FIELDS, key, strict=True))
        vector = msc.noise_vector(seeds, context)
        replayed += 1
        if replayed % 32 == 0:
            print(f"      replayed {replayed} unique field-noise triplets", flush=True)
        return vector

    gx_rows, gx_noise_rows, n_unique_noise = _assemble_gx_rows(
        point_signal,
        training["point_index"],
        training["noise_seeds"],
        noise_provider,
    )
    pairs: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for name in PAIR_SPECS:
        pair_arrays, pair_metadata = _joint_payload(
            name,
            gx_rows,
            training["x"],
            training["point_index"],
            training["point_u"],
            gx_observation,
            g2x_observation,
            u_reference,
        )
        expected_dof = training["n_points"] * (training["replicas"] - 1)
        if pair_metadata["covariance_degrees_of_freedom"] != expected_dof:
            raise AssertionError(f"{name} covariance degrees of freedom changed")
        pairs[name] = (pair_arrays, pair_metadata)
        print(
            f"      {name}: 28 -> 5; dof {expected_dof}; "
            f"response rank {pair_metadata['response_rank_after_whitening']}; "
            "full cross block retained",
            flush=True,
        )

    output_arrays: dict[str, Any] = {
        "gx": gx_rows,
        "gx_noise": gx_noise_rows,
        "gx_point_signal": point_signal,
        "gx_data_vector": gx_observation,
        "g2x_data_vector": g2x_observation,
        "theta": training["theta"],
        "u": training["u"],
        "point_theta": training["point_theta"],
        "point_u": training["point_u"],
        "point_index": training["point_index"],
        "replica": training["replica"],
        "plan_position": training["plan_position"],
        "plan_entry_index": training["plan_entry_index"],
        "bank_indices": training["bank_indices"],
        "noise_seeds": training["noise_seeds"],
        "point_bank_indices": training["point_bank_indices"],
        "theta_reference": theta_reference,
        "u_reference": u_reference,
        "band_edges": training["band_edges"],
        "effective_ell": training["effective_ell"],
    }
    pair_manifest: dict[str, Any] = {}
    for pair_name, (pair_arrays, pair_metadata) in pairs.items():
        pair_manifest[pair_name] = pair_metadata
        for array_name, value in pair_arrays.items():
            output_arrays[f"{array_name}_{pair_name}"] = value

    manifest: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "scientific_scope": (
            "exploratory matched five-replica gX+g2X check only; not a validated "
            "or quotable Stage-7 posterior"
        ),
        "construction": (
            "gX signal measured once from each exact saved g2X map source; the g2X "
            "row's own field-noise seed triplet supplies the additive gX noise vector"
        ),
        "cross_covariance_policy": (
            "each 28x28 covariance is estimated from paired within-point residuals; "
            "the off-diagonal gX-g2X block is retained"
        ),
        "covariance_repair": "none; Cholesky failure is fatal",
        "parameter_names": list(PARAMETER_NAMES),
        "prior_low": prior_low.tolist(),
        "prior_high": prior_high.tolist(),
        "gx_vector_order": GX_VECTOR_ORDER,
        "g2x_vector_order": G2X_VECTOR_ORDER,
        "n_band": 14,
        "n_plan_points": len(plan["entries"]),
        "n_points": int(training["n_points"]),
        "replicas": int(training["replicas"]),
        "n_rows": int(training["n_rows"]),
        "n_unique_noise_triplets_used": int(n_unique_noise),
        "noise_route": noise_route,
        "source_plan": {
            "path": str(args.plan.expanduser().resolve()),
            "sha256": msc.sha256_file(args.plan.expanduser().resolve()),
        },
        "g2x_training_set": {
            "path": str(training["path"]),
            "sha256": training["file_sha256"],
            "schema_version": training["manifest"]["schema_version"],
            "output_vector_sha256": training["manifest"]["output_vector_sha256"],
        },
        "g2x_pilot_contract": {
            "path": str(g2_contract["path"]),
            "sha256": g2_contract["file_sha256"],
            "schema_version": g2_contract["manifest"]["schema_version"],
        },
        "gx_mock_contract": {
            "path": str(mock_contract.contract_path),
            "file_sha256": msc.sha256_file(mock_contract.contract_path),
            "canonical_contract_sha256": mock_contract.contract_sha256,
            "observation_kind": "pasted_map_measurement_noiseless",
            "data_vector_sha256": msc.sha256_array(gx_observation),
            "source_validation": (
                {"status": "strict_all_declared_sources_verified"}
                if mock_contract_source_exception is None
                else mock_contract_source_exception
            ),
        },
        "source_code_validation": {
            "status": "historical_code_hashes_not_enforced_by_user_request",
            "g2x_mock_sbi_common": common_source_validation,
            "numerical_and_map_integrity_checks_retained": True,
        },
        "reference_paste": {
            "path": str(reference_path),
            "sha256": reference_sha,
        },
        "estimator": {
            **estimator_identity,
            "nside": int(msc.NSIDE),
            "lmax": int(msc.LMAX),
            "band_edges": np.asarray(context.band_edges).tolist(),
            "effective_ell": np.asarray(context.effective_ell).tolist(),
            "coupling": "frozen scalar NaMaster workspace; decouple_cell",
        },
        "map_sources": {
            "combined": n_combined,
            "strict_split_sum_in_memory": n_split_sets,
            "records_in_g2x_training_manifest": int(training["n_points"]),
        },
        "pairs": pair_manifest,
        "path_remaps": [f"{old}={new}" for old, new in remaps],
        "source_code": {
            "builder": {
                "path": str(pathlib.Path(__file__).resolve()),
                "sha256": msc.sha256_file(pathlib.Path(__file__).resolve()),
            },
            "g2x_training_builder": {
                "path": str(pathlib.Path(g2build.__file__).resolve()),
                "sha256": msc.sha256_file(pathlib.Path(g2build.__file__).resolve()),
            },
            "g2x_contract_builder": {
                "path": str(pathlib.Path(g2pilot.__file__).resolve()),
                "sha256": msc.sha256_file(pathlib.Path(g2pilot.__file__).resolve()),
            },
            "mock_sbi_common": {
                "path": str(pathlib.Path(msc.__file__).resolve()),
                "sha256": msc.sha256_file(pathlib.Path(msc.__file__).resolve()),
            },
        },
        "output_arrays": {},
        "elapsed_seconds": time.time() - started,
    }
    manifest["output_arrays"] = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": msc.sha256_array(np.asarray(value)),
        }
        for name, value in output_arrays.items()
    }
    output_arrays["manifest_json"] = json.dumps(manifest, sort_keys=True)

    print("[6/6] writing the matched gX+g2X pilot product ...", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output, output_arrays)
    _atomic_write_text(sidecar, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {output}")
    print(f"      manifest {sidecar}")
    print(f"      analysis status {ANALYSIS_STATUS}")
    print(
        f"      gx rows {gx_rows.shape}; three paired 28D contracts; "
        f"{n_unique_noise} unique noise triplets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
