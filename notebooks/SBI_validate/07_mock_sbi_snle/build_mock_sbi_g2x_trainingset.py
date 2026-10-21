#!/usr/bin/env python3
"""Build a five-realisation :math:`g^2 X` training set from a paste plan.

This script starts from the products that actually exist after the mock-SBI paste
stage:

* ``paste_plan.json`` and its ``mocksbi_*.yaml`` configurations; and
* either the combined noiseless HDF5 map for each point or its complete set of
  ``split000of032`` through ``split031of032`` products.

It does **not** require ``training_set_round1.npz`` or an ``augmented_maps``
directory.  Those are optional products of a later Stage-5 job and may never have
been created.  A combined signal map is also optional: complete split sets are
strictly validated and summed in memory in the same float32 split order as the
registered combiner, without writing another map file.

For every available point in the plan, the script reconstructs the same deterministic
training noise-bank seeds and the 64-wide point-to-bank assignment used by
``16_measure_augment_round1.sbatch`` and ``build_mock_sbi_training_set.py``.  It selects
the first ``--replicas`` columns of that assignment (five by default), generates y, tau
and CMB-kappa noise maps, adds them to the corresponding noiseless pasted maps in memory,
and measures the skew spectra

    C_ell^{g^2,y}, C_ell^{g^2,kappa}, C_ell^{g^2,tau}.

The fixed galaxy overdensity is reconstructed from the frozen catalog, squared before
the common mask is applied, and mask-weight centred.  Spectra use the frozen scalar
NaMaster workspace and 14 inference bands.  The noisy maps are discarded after each
measurement; only the 42-component vectors, labels, seeds and provenance are saved.

Example
-------
Run from the repository root.  This command creates the explicitly named output
``training_set_g2x_rep5.npz`` (and a JSON sidecar) beside the plan::

    python notebooks/SBI_validate/07_mock_sbi_snle/build_mock_sbi_g2x_training_set_all128_v2.py \\
      --plan data/SBI_validate/mock_sbi/paste_plan/round_1/paste_plan.json \\
      --output data/SBI_validate/mock_sbi/paste_plan/round_1/training_set_g2x_rep5.npz \\
      --replicas 5

If ``--output`` is omitted, that same filename is derived from ``--replicas`` and
placed beside the plan.  Use ``--preflight`` first to validate every path, frozen
contract, YAML hash, pasted-map dataset shape, and fixed g2 reconstruction without
creating the output.

The registered round-1 helper-SHA incident can be accepted with
``--allow-known-combine-only-helper-sha-mix``.  The option recognizes only the exact
pre/post hash pair from the edit confined to ``combine_partial_maps``; every other
contract mismatch remains fatal.  Accepted points and per-hash split counts are written
to the output provenance.  Alternatively, pass ``helper_sha_repaste.json`` as
``--exclude-manifest`` to omit each affected point in full.  Exclusions retain the
original full-plan positions used for noise assignment.  For an un-repaired round-1
plan, add::

      --exclude-manifest data/SBI_validate/mock_sbi/paste_plan/round_1/helper_sha_repaste.json

If absolute paths embedded in the plan/YAML files point at another checkout, add for
example::

      --path-remap /old/GODMAX=/new/GODMAX

The remap is handled inside this script, including paths stored in the frozen HDF5
contract; ``common/mock_sbi_common.py`` does not need to be edited.  The galaxy catalog
is always the exact whole-file-SHA-verified map registered in that contract; an arbitrary
replacement is not accepted because equality of truncated linear galaxy alms would not
prove equality after squaring the pixel-space field.

The output ordering is ``g2y[14],g2kappa[14],g2tau[14]``.

Do **not** pass this file to the current ``run_mock_sbi_snle.py`` unchanged.  That runner
loads the existing two-point ``gy/gkappa/gtau`` observation, covariance and score operator;
the equal vector width (42) would not make those objects compatible.  A separately frozen
g2X observation, covariance/compressor and schema-checking runner are required first.
"""

from __future__ import annotations

# Keep imports working when this file is run directly from the stage folder.
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


PLAN_SCHEMA = "godmax.mock_sbi.paste_plan.v1"
OUTPUT_SCHEMA = "godmax.mock_sbi.g2x_training_set.v1"
DEFAULT_REPLICAS = 5
DEFAULT_STAGE5_REPLICAS = 64
DEFAULT_BANK_SIZE = 2048
ASSIGNMENT_NAMESPACE = (20260825, 205)
# Registered incident: these are the helper-file hashes immediately before and
# after an edit confined to ``combine_partial_maps``.  ``run_paste_split`` was
# unchanged, so the split map contents are scientifically equivalent.  Acceptance
# remains explicit, exact-pair only, and is recorded in the output manifest.
KNOWN_COMBINE_ONLY_HELPER_SHA_MIX = frozenset(
    {
        "2044f85df2206c9b53bb9fa2ab5916dbdc0d8297d169ea27c073196640ea773e",
        "121c9783447948149ed5cd7d53fbe272dd503fcf2f1b53d09268a04a3042dd3d",
    }
)
PARAMETER_NAMES = (
    "theta_ej_0",
    "alpha_nt",
    "mu_beta",
    "theta_co_0",
    "nu_theta_ej_M",
)
G2X_SPECTRA = (
    ("g2y", "y"),
    ("g2kappa", "kappa"),
    ("g2tau", "tau"),
)
VECTOR_ORDER = "spectrum-major g2y[14],g2kappa[14],g2tau[14]"
SPLIT_COMMON_ATTRS = (
    "catalog_key",
    "catalog_path",
    "nside",
    "n_input_halos",
    "num_splits",
    "split_strategy",
    "split_block_halos",
    "max_paint_R200c_factor",
    "smooth_profiles",
    "map_datasets_json",
    "random_seed_base",
    "random_seed_rule",
    "projected_profile_integration_method",
    "num_points_projected_profile",
    "halo_grid_nr",
    "halo_grid_nM",
    "halo_grid_nz",
    "catalog_file_sha256",
    "config_file_sha256",
    "catalog_row_identity_sha256",
    "catalog_selection_contract_sha256",
    "catalog_source_content_manifest_sha256",
    "catalog_cosmology_sha256",
    "pasting_helper_sha256",
    "get_sim_maps_sha256",
    "get_radial_profiles_sha256",
    "base_class_sha256",
    "config_factory_sha256",
    "godmax_default_params_sha256",
    "pasting_contract_version",
    "development_lmax",
    "comparison_lmax",
    "profile_smoothing_applied",
    "profile_smoothing_kernel",
    "profile_smoothing_fwhm_arcmin",
    "profile_smoothing_pixel_resolution_arcmin",
    "profile_smoothing_fwhm_pixel_fraction",
    "profile_smoothing_sigma_rad",
    "profile_smoothing_method",
    "profile_smoothing_quadrature_points",
    "profile_smoothing_radial_sigma_cutoff",
    "healpix_pixel_window_applied_during_paste",
    "theory_transfer_policy",
    "kernel_dataset_sha256_json",
    "profiles_class_fqname",
    "galaxy_catalog_columns_json",
)


def _atomic_write_text(path: pathlib.Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_bank_indices(
    round_label: str,
    n_points: int,
    n_replicas: int,
    bank_size: int,
) -> np.ndarray:
    """Reproduce Stage 5's deterministic point/replica-to-bank assignment."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if n_replicas <= 0:
        raise ValueError("n_replicas must be positive")
    if bank_size <= 0:
        raise ValueError("bank_size must be positive")
    if n_replicas > bank_size:
        raise ValueError(
            f"{n_replicas} replicas requested from a bank of {bank_size}"
        )
    label_word = int.from_bytes(
        hashlib.sha256(str(round_label).encode("utf-8")).digest()[:4],
        "big",
    )
    entropy = tuple(ASSIGNMENT_NAMESPACE) + (label_word,)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return np.stack(
        [
            rng.choice(bank_size, size=n_replicas, replace=False)
            for _ in range(n_points)
        ]
    ).astype(np.int64, copy=False)


def select_stage5_bank_indices(
    round_label: str,
    n_points: int,
    selected_replicas: int,
    stage5_replicas: int,
    bank_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Stage 5's full assignment, then select its written-map prefix."""

    if selected_replicas <= 0:
        raise ValueError("selected_replicas must be positive")
    if stage5_replicas < selected_replicas:
        raise ValueError("stage5_replicas must be >= selected_replicas")
    full = assign_bank_indices(
        round_label,
        n_points,
        stage5_replicas,
        bank_size,
    )
    return full, full[:, :selected_replicas]


def _verify_stage5_assignment_parity(
    round_label: str,
    n_points: int,
    n_replicas: int,
    bank_size: int,
    observed: np.ndarray,
) -> dict[str, Any]:
    """Require exact parity with Stage 5's checked-in assignment function."""

    import importlib.util

    source = _SBI_ROOT / "05_pasting" / "build_mock_sbi_training_set.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"authoritative Stage-5 assignment implementation is missing: {source}"
        )
    spec = importlib.util.spec_from_file_location(
        "_godmax_stage5_training_builder_for_parity",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import Stage-5 assignment code from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = np.asarray(
        module.assign_bank_indices(
            round_label,
            n_points,
            n_replicas,
            bank_size,
        ),
        dtype=np.int64,
    )
    if not np.array_equal(np.asarray(observed, dtype=np.int64), expected):
        raise RuntimeError(
            "point-to-noise-bank assignments differ from "
            "05_pasting/build_mock_sbi_training_set.py"
        )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "implementation_path": str(source),
        "implementation_sha256": source_sha,
        "assignment_namespace": [
            int(value) for value in module.ASSIGNMENT_NAMESPACE
        ],
        "parity_checked": True,
    }


def _parse_path_remaps(
    values: Sequence[str],
) -> list[tuple[pathlib.Path, pathlib.Path]]:
    rules: list[tuple[pathlib.Path, pathlib.Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"path remap {value!r} must have the form OLD_PREFIX=NEW_PREFIX"
            )
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"path remap {value!r} has an empty prefix")
        old_path = pathlib.Path(old).expanduser()
        new_path = pathlib.Path(new).expanduser()
        if not old_path.is_absolute() or not new_path.is_absolute():
            raise ValueError(
                f"path remap {value!r} must use absolute OLD and NEW prefixes"
            )
        rules.append((old_path.resolve(), new_path.resolve()))
    return rules


def _resolve_path(
    value: str | pathlib.Path,
    repo_root: pathlib.Path,
    remaps: Sequence[tuple[pathlib.Path, pathlib.Path]],
) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    for old, new in remaps:
        try:
            relative = path.relative_to(old)
        except ValueError:
            continue
        path = new / relative
        break
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _training_output_path(
    plan_path: pathlib.Path,
    requested: pathlib.Path | None,
    replicas: int,
) -> pathlib.Path:
    """Return the explicit output or the deterministic plan-local default."""

    if requested is not None:
        return requested.expanduser().resolve()
    return plan_path.parent / f"training_set_g2x_rep{replicas}.npz"


def _load_and_validate_plan(path: pathlib.Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict):
        raise ValueError("paste plan did not decode to an object")
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(
            f"plan schema {plan.get('schema_version')!r} != {PLAN_SCHEMA!r}"
        )
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("paste plan has no entries")
    if int(plan.get("count", -1)) != len(entries):
        raise ValueError("paste plan count differs from len(entries)")
    if tuple(plan.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError(
            f"plan parameter order {plan.get('parameter_names')} != "
            f"{list(PARAMETER_NAMES)}"
        )
    if "round" not in plan:
        raise KeyError("paste plan is missing its round label")

    required = {
        "index",
        "run_name",
        "config_path",
        "config_sha256",
        "theta",
        "theta_sha256",
        "u",
        "log_q",
        "importance_eligible",
    }
    seen_indices: set[int] = set()
    seen_runs: set[str] = set()
    seen_theta: set[str] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"plan entry {position} is not an object")
        missing = sorted(required - set(entry))
        if missing:
            raise KeyError(f"plan entry {position} is missing {missing}")
        index = entry["index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"plan entry {position} has a non-integer index")
        if index != position:
            raise ValueError(
                f"plan entry at list position {position} declares index {index}. "
                "This is a compacted or reordered plan, so its original Stage-5 "
                "noise-assignment positions cannot be recovered from this file alone. "
                "Use the original paste_plan.json and pass helper_sha_repaste.json "
                "through --exclude-manifest instead."
            )
        if index in seen_indices:
            raise ValueError(f"plan has duplicate entry index {index}")
        seen_indices.add(index)
        run_name = str(entry["run_name"])
        if not run_name or run_name in seen_runs:
            raise ValueError(f"plan has an empty or duplicate run name {run_name!r}")
        seen_runs.add(run_name)

        theta = entry["theta"]
        if not isinstance(theta, dict):
            raise ValueError(f"plan entry {position} theta is not an object")
        missing_theta = [name for name in PARAMETER_NAMES if name not in theta]
        if missing_theta:
            raise KeyError(
                f"plan entry {position} theta is missing {missing_theta}"
            )
        theta_values = np.asarray(
            [theta[name] for name in PARAMETER_NAMES], dtype=np.float64
        )
        if not np.all(np.isfinite(theta_values)):
            raise ValueError(f"plan entry {position} theta is non-finite")
        theta_sha = _canonical_json_sha256(
            {name: float(theta[name]) for name in PARAMETER_NAMES}
        )
        if theta_sha != str(entry["theta_sha256"]):
            raise ValueError(
                f"plan entry {position} theta hash {theta_sha} != "
                f"{entry['theta_sha256']}"
            )
        if theta_sha in seen_theta:
            raise ValueError(
                f"plan contains the same theta more than once: {theta_sha}"
            )
        seen_theta.add(theta_sha)

        if entry["u"] is not None:
            u = np.asarray(entry["u"], dtype=np.float64)
            if u.shape != (len(PARAMETER_NAMES),) or not np.all(np.isfinite(u)):
                raise ValueError(f"plan entry {position} has an invalid u vector")
        if entry["log_q"] is not None and not np.isfinite(float(entry["log_q"])):
            raise ValueError(f"plan entry {position} has an invalid log_q")
        if not isinstance(entry["importance_eligible"], bool):
            raise ValueError(
                f"plan entry {position} importance_eligible is not boolean"
            )
        if "cached_map" in entry:
            cached_map = entry["cached_map"]
            cached_sha = entry.get("cached_map_sha256")
            if not isinstance(cached_map, str) or not cached_map:
                raise ValueError(
                    f"plan entry {position} has an invalid cached_map path"
                )
            if (
                not isinstance(cached_sha, str)
                or len(cached_sha) != 64
                or any(character not in "0123456789abcdef" for character in cached_sha)
            ):
                raise ValueError(
                    f"plan entry {position} uses cached_map but has no valid "
                    "lowercase SHA-256 in cached_map_sha256"
                )

    return plan


def _load_exclusion_manifest(
    path: pathlib.Path,
    plan: Mapping[str, Any],
    repo_root: pathlib.Path,
    remaps: Sequence[tuple[pathlib.Path, pathlib.Path]],
    msc,
) -> tuple[set[int], dict[str, Any]]:
    """Resolve a stage exclusion manifest without compacting the paste plan.

    Stage 17 identifies the helper-SHA incident by the basename of each
    ``config_path`` in ``helper_sha_repaste.json``.  Reproduce that matching rule,
    but additionally require a one-to-one match against the current plan and verify
    the current config hash before excluding it.
    """

    manifest_path = _resolve_path(path, repo_root, remaps)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"exclusion manifest does not exist: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("exclusion manifest did not decode to an object")
    manifest_entries = payload.get("entries")
    if not isinstance(manifest_entries, list) or not manifest_entries:
        raise ValueError("exclusion manifest has no non-empty entries list")

    plan_by_config_name: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for position, entry in enumerate(plan["entries"]):
        name = pathlib.Path(str(entry["config_path"])).name
        if not name:
            raise ValueError(f"plan position {position} has an empty config basename")
        if name in plan_by_config_name:
            raise ValueError(
                f"plan config basenames are not unique; duplicate {name!r}"
            )
        plan_by_config_name[name] = (position, entry)

    excluded_positions: set[int] = set()
    records: list[dict[str, Any]] = []
    for manifest_position, item in enumerate(manifest_entries):
        if not isinstance(item, dict) or "config_path" not in item:
            raise ValueError(
                f"exclusion entry {manifest_position} has no config_path"
            )
        config_name = pathlib.Path(str(item["config_path"])).name
        if config_name not in plan_by_config_name:
            raise ValueError(
                f"exclusion entry {manifest_position} ({config_name}) is not in "
                "the supplied paste plan"
            )
        plan_position, plan_entry = plan_by_config_name[config_name]
        if plan_position in excluded_positions:
            raise ValueError(
                f"exclusion manifest lists {config_name!r} more than once"
            )

        config_path = _resolve_path(
            plan_entry["config_path"], repo_root, remaps
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"excluded point's paste config does not exist: {config_path}"
            )
        config_sha = msc.sha256_file(config_path)
        if config_sha != str(plan_entry["config_sha256"]):
            raise ValueError(
                f"excluded point's paste config changed: {config_path}; "
                f"{config_sha} != {plan_entry['config_sha256']}"
            )

        excluded_positions.add(plan_position)
        records.append(
            {
                "manifest_position": manifest_position,
                "plan_position": plan_position,
                "entry_index": int(plan_entry["index"]),
                "run_name": str(plan_entry["run_name"]),
                "config_path": str(config_path),
                "config_sha256": config_sha,
                "manifest_entry": item,
            }
        )

    metadata = {
        "path": str(manifest_path),
        "sha256": msc.sha256_file(manifest_path),
        "matching_rule": "unique config_path basename, as in 17_trial_snle_partial.sbatch",
        "n_excluded": len(records),
        "records": records,
    }
    return excluded_positions, metadata


def _entry_map_info(
    entry: Mapping[str, Any],
    repo_root: pathlib.Path,
    remaps: Sequence[tuple[pathlib.Path, pathlib.Path]],
    msc,
) -> dict[str, Any]:
    """Resolve one plan entry and both its combined and split map locations."""

    import yaml

    config_path = _resolve_path(entry["config_path"], repo_root, remaps)
    if not config_path.is_file():
        raise FileNotFoundError(f"paste config does not exist: {config_path}")
    config_sha = msc.sha256_file(config_path)
    if config_sha != str(entry["config_sha256"]):
        raise ValueError(
            f"paste config changed: {config_path}; "
            f"{config_sha} != {entry['config_sha256']}"
        )
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"paste config did not decode to an object: {config_path}")
    try:
        paste = config["pasting"]
        project = config["project"]
        catalog_key = config["resolved_theory"]["catalog_key"]
    except (KeyError, TypeError) as error:
        raise KeyError(f"paste config is missing its map-location contract: {config_path}") from error

    if str(paste.get("run_name")) != str(entry["run_name"]):
        raise ValueError(
            f"config run name {paste.get('run_name')!r} != "
            f"plan run name {entry['run_name']!r}"
        )
    if "nside" not in paste:
        raise KeyError(f"paste config has no pasting.nside: {config_path}")
    overrides = paste.get("gas_parameter_overrides")
    if not isinstance(overrides, dict):
        raise ValueError(f"config has no gas_parameter_overrides: {config_path}")
    for name in PARAMETER_NAMES:
        if name not in overrides or not np.isclose(
            float(overrides[name]),
            float(entry["theta"][name]),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"config override {name} does not match the plan for "
                f"{entry['run_name']}"
            )

    if entry.get("cached_map"):
        combined_path = _resolve_path(entry["cached_map"], repo_root, remaps)
        split_paths: tuple[pathlib.Path, ...] = ()
        num_splits = 0
        expected_map_sha = str(entry.get("cached_map_sha256", ""))
    else:
        output_root = _resolve_path(project["output_root"], repo_root, remaps)
        map_root = output_root / str(project["map_subdir"]) / str(paste["run_name"])
        stem = f"abacus_pasted_maps_{catalog_key}_nside{int(paste['nside'])}"
        combined_path = map_root / f"{stem}.h5"
        num_splits = int(paste.get("num_splits", 0))
        if num_splits <= 0:
            raise ValueError(
                f"config has invalid pasting.num_splits={num_splits}: {config_path}"
            )
        split_paths = tuple(
            map_root / f"{stem}_split{split:03d}of{num_splits:03d}.h5"
            for split in range(num_splits)
        )
        expected_map_sha = ""

    return {
        "config_path": config_path,
        "config_sha256": config_sha,
        "is_cached_map": bool(entry.get("cached_map")),
        "combined_map_path": combined_path.resolve(),
        "split_paths": tuple(path.resolve() for path in split_paths),
        "num_splits": num_splits,
        "expected_map_sha256": expected_map_sha,
        "nside": int(paste["nside"]),
        "catalog_key": str(catalog_key),
        "catalog_file_sha256": str(
            config["resolved_theory"].get("catalog_file_sha256", "")
        ),
        "strict_combine": bool(paste.get("require_strict_combine_contract", False)),
        "galaxies_expected": bool(paste.get("get_galmap", True)),
        "save_projection_kernels": bool(
            paste.get("save_projection_kernels", False)
        ),
    }


def _timing_path(split_path: pathlib.Path) -> pathlib.Path:
    return split_path.with_suffix(split_path.suffix + ".timing.json")


def _is_nonempty_file(path: pathlib.Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _attribute_equal(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _attribute_text(value: Any) -> str:
    """Return a stable text representation for a scalar HDF5 attribute."""

    scalar = np.asarray(value)
    if scalar.shape != ():
        raise TypeError(f"expected a scalar HDF5 attribute, got shape {scalar.shape}")
    item = scalar.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _is_registered_combine_only_helper_mix(values: Sequence[str]) -> bool:
    """Return true only for the registered pre/post helper hash pair."""

    return frozenset(str(value) for value in values) == (
        KNOWN_COMBINE_ONLY_HELPER_SHA_MIX
    )


def _attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    decode = getattr(value, "decode", None)
    if callable(decode):
        return str(decode("utf-8"))
    return str(value)


def _expected_block_striped_counts(
    n_input: int,
    num_splits: int,
    block_size: int,
) -> list[int]:
    counts = [0] * num_splits
    for block_index, start in enumerate(range(0, n_input, max(1, block_size))):
        stop = min(start + max(1, block_size), n_input)
        counts[block_index % num_splits] += stop - start
    return counts


def _validate_combined_map_layout(
    info: Mapping[str, Any],
    expected_shape: tuple[int, ...],
    msc,
) -> None:
    import h5py

    map_path = pathlib.Path(info["combined_map_path"])
    expected_file_sha = str(info["expected_map_sha256"])
    if expected_file_sha:
        observed_file_sha = msc.sha256_file(map_path)
        if observed_file_sha != expected_file_sha:
            raise ValueError(
                f"cached map hash {observed_file_sha} != {expected_file_sha}: "
                f"{map_path}"
            )

    with h5py.File(map_path, "r") as handle:
        missing = [
            dataset for dataset in msc.MAP_DATASETS.values() if dataset not in handle
        ]
        if missing:
            raise KeyError(f"{map_path} is missing dataset(s) {missing}")
        observed_map_hashes: dict[str, str] = {}
        for field, dataset in msc.MAP_DATASETS.items():
            stored = handle[dataset]
            if tuple(stored.shape) != expected_shape:
                raise ValueError(
                    f"{map_path} {field} dataset has shape {stored.shape}; "
                    f"expected {expected_shape}"
                )
            if np.dtype(stored.dtype) != np.dtype(np.float32):
                raise TypeError(
                    f"{map_path} {field} dataset has dtype {stored.dtype}; "
                    "expected float32 from the registered combiner"
                )
            values = np.asarray(stored, dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"combined map {map_path} {field} contains non-finite values"
                )
            observed_map_hashes[pathlib.PurePosixPath(dataset).name] = (
                msc.sha256_array(values)
            )
            del values

        if info["is_cached_map"]:
            return

        required_attrs = (
            "schema_version",
            "ordering",
            "catalog_key",
            "nside",
            "n_input_halos",
            "combined_from_num_splits",
            "combined_split_halo_sum",
            "config_file_sha256",
            "catalog_file_sha256",
            "map_datasets_json",
            "map_dataset_sha256_json",
        )
        missing_attrs = [key for key in required_attrs if key not in handle.attrs]
        if missing_attrs:
            raise ValueError(
                f"combined map {map_path} is missing contract attrs: {missing_attrs}"
            )
        if str(handle.attrs["schema_version"]) != "sbi_three_probe_signal_v1":
            raise ValueError(f"combined map {map_path} has an unexpected schema")
        if str(handle.attrs["ordering"]) != "RING":
            raise ValueError(f"combined map {map_path} is not in RING ordering")
        if int(handle.attrs["nside"]) != int(info["nside"]):
            raise ValueError(f"combined map {map_path} nside differs from its YAML")
        if str(handle.attrs["catalog_key"]) != str(info["catalog_key"]):
            raise ValueError(
                f"combined map {map_path} catalog_key differs from its YAML"
            )
        if str(handle.attrs["config_file_sha256"]) != str(info["config_sha256"]):
            raise ValueError(
                f"combined map {map_path} was not produced by the current YAML"
            )
        if str(handle.attrs["catalog_file_sha256"]) != str(
            info["catalog_file_sha256"]
        ):
            raise ValueError(
                f"combined map {map_path} does not reference the frozen catalog SHA"
            )
        if int(handle.attrs["combined_from_num_splits"]) != int(
            info["num_splits"]
        ):
            raise ValueError(
                f"combined map {map_path} split count differs from its YAML"
            )
        if int(handle.attrs["combined_split_halo_sum"]) != int(
            handle.attrs["n_input_halos"]
        ):
            raise ValueError(
                f"combined map {map_path} split halo sum does not cover its catalog"
            )
        required_map_keys = {
            pathlib.PurePosixPath(dataset).name
            for dataset in msc.MAP_DATASETS.values()
        }
        declared_map_keys = set(json.loads(str(handle.attrs["map_datasets_json"])))
        stored_map_keys = set(handle["maps"].keys()) if "maps" in handle else set()
        if declared_map_keys != required_map_keys or stored_map_keys != required_map_keys:
            raise ValueError(
                f"combined map {map_path} map-key contract differs: declared "
                f"{sorted(declared_map_keys)}, stored {sorted(stored_map_keys)}, "
                f"required {sorted(required_map_keys)}"
            )
        declared_hashes = json.loads(
            str(handle.attrs["map_dataset_sha256_json"])
        )
        if not isinstance(declared_hashes, dict):
            raise ValueError(
                f"combined map {map_path} map hashes are not a JSON object"
            )
        if set(declared_hashes) != required_map_keys:
            raise ValueError(
                f"combined map {map_path} map hash keys differ from its datasets"
            )
        if observed_map_hashes != {
            str(key): str(value) for key, value in declared_hashes.items()
        }:
            mismatched_hashes = sorted(
                key
                for key in required_map_keys
                if observed_map_hashes.get(key) != str(declared_hashes.get(key))
            )
            raise ValueError(
                f"combined map {map_path} dataset content hashes differ for "
                f"{mismatched_hashes}"
            )


def _process_split_set(
    info: Mapping[str, Any],
    expected_shape: tuple[int, ...],
    msc,
    *,
    load_maps: bool,
    allow_known_helper_sha_mix: bool = False,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Validate and optionally sum a complete split set exactly in split order."""

    import h5py

    if not info["strict_combine"]:
        raise ValueError(
            "direct split loading requires pasting.require_strict_combine_contract=true"
        )
    if info["galaxies_expected"]:
        raise ValueError(
            "direct split loading is registered only for mock-SBI configs with "
            "pasting.get_galmap=false; combine this point with the official combiner"
        )

    paths = tuple(pathlib.Path(path) for path in info["split_paths"])
    num_splits = int(info["num_splits"])
    if len(paths) != num_splits:
        raise ValueError(f"resolved {len(paths)} split paths; expected {num_splits}")

    required_map_keys = {
        pathlib.PurePosixPath(dataset).name
        for dataset in msc.MAP_DATASETS.values()
    }
    reference_attrs: dict[str, Any] | None = None
    reference_kernel_attrs: dict[str, Any] | None = None
    observed_indices: list[int] = []
    observed_halo_counts = [0] * num_splits
    helper_sha_counts: dict[str, int] = {}
    split_records: list[dict[str, Any]] = []
    maps_sum = (
        {
            field: np.zeros(expected_shape, dtype=np.float32)
            for field in msc.NOISE_FIELDS
        }
        if load_maps
        else None
    )

    for expected_index, path in enumerate(paths):
        timing_path = _timing_path(path)
        if not _is_nonempty_file(path):
            raise FileNotFoundError(f"split file is missing or empty: {path}")
        if not _is_nonempty_file(timing_path):
            raise FileNotFoundError(
                f"split has no completed timing sidecar: {timing_path}"
            )

        record: dict[str, Any] = {
            "split_index": expected_index,
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "timing_path": str(timing_path),
        }
        with h5py.File(path, "r") as handle:
            missing_attrs = [key for key in SPLIT_COMMON_ATTRS if key not in handle.attrs]
            if missing_attrs:
                raise ValueError(f"split {path} is missing contract attrs: {missing_attrs}")
            part_attrs = {key: handle.attrs[key] for key in SPLIT_COMMON_ATTRS}
            helper_sha = _attribute_text(part_attrs["pasting_helper_sha256"])
            helper_sha_counts[helper_sha] = helper_sha_counts.get(helper_sha, 0) + 1
            split_index = int(handle.attrs.get("split_index", -1))
            if split_index != expected_index:
                raise ValueError(
                    f"split filename position {expected_index} declares split_index "
                    f"{split_index}: {path}"
                )
            observed_indices.append(split_index)
            observed_halo_counts[split_index] = int(
                handle.attrs.get("n_split_halos", -1)
            )
            if observed_halo_counts[split_index] < 0:
                raise ValueError(f"split {path} has no valid n_split_halos")

            if reference_attrs is None:
                reference_attrs = part_attrs
            else:
                mismatched = [
                    key
                    for key in SPLIT_COMMON_ATTRS
                    if not _attribute_equal(part_attrs[key], reference_attrs[key])
                ]
                if mismatched:
                    reference_helper_sha = _attribute_text(
                        reference_attrs["pasting_helper_sha256"]
                    )
                    observed_helper_values = {helper_sha, reference_helper_sha}
                    known_helper_only_mix = (
                        mismatched == ["pasting_helper_sha256"]
                        and _is_registered_combine_only_helper_mix(
                            observed_helper_values
                        )
                    )
                    if not (
                        allow_known_helper_sha_mix and known_helper_only_mix
                    ):
                        detail = ""
                        if mismatched == ["pasting_helper_sha256"]:
                            detail = (
                                "; observed helper hashes "
                                f"{sorted(observed_helper_values)}"
                            )
                        raise ValueError(
                            f"split {path} has incompatible contract attrs: "
                            f"{mismatched}{detail}"
                        )

            if "maps" not in handle:
                raise KeyError(f"split {path} has no maps group")
            actual_map_keys = set(handle["maps"].keys())
            declared_map_keys = set(
                json.loads(str(handle.attrs["map_datasets_json"]))
            )
            if actual_map_keys != declared_map_keys:
                raise ValueError(
                    f"split {path} map datasets differ from its declared schema"
                )
            if actual_map_keys != required_map_keys:
                raise ValueError(
                    f"split {path} has map keys {sorted(actual_map_keys)}; expected "
                    f"{sorted(required_map_keys)}"
                )
            if "galaxies" not in handle or int(handle["galaxies"].shape[0]) != 0:
                raise ValueError(
                    f"split {path} violates the frozen get_galmap=false contract"
                )

            map_hashes: dict[str, str] = {}
            for field, dataset in msc.MAP_DATASETS.items():
                stored = handle[dataset]
                if tuple(stored.shape) != expected_shape:
                    raise ValueError(
                        f"split {path} {field} shape {stored.shape}; "
                        f"expected {expected_shape}"
                    )
                if np.dtype(stored.dtype) != np.dtype(np.float32):
                    raise TypeError(
                        f"split {path} {field} dtype {stored.dtype}; expected float32"
                    )
                if load_maps:
                    values = np.asarray(stored, dtype=np.float32)
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"split {path} {field} contains non-finite values")
                    map_hashes[field] = msc.sha256_array(values)
                    assert maps_sum is not None
                    maps_sum[field] += values
                    del values

            if info["save_projection_kernels"]:
                if "kernels" not in handle:
                    raise KeyError(f"split {path} has no registered kernels group")
                declared_kernel_hashes = json.loads(
                    str(handle.attrs["kernel_dataset_sha256_json"])
                )
                observed_kernel_hashes = {
                    key: msc.sha256_array(np.asarray(handle[f"kernels/{key}"]))
                    for key in handle["kernels"]
                }
                if observed_kernel_hashes != declared_kernel_hashes:
                    raise ValueError(
                        f"split {path} kernel arrays do not match their declared hashes"
                    )
                kernel_attrs = dict(handle["kernels"].attrs)
                if reference_kernel_attrs is None:
                    reference_kernel_attrs = kernel_attrs
                elif set(kernel_attrs) != set(reference_kernel_attrs) or any(
                    not _attribute_equal(kernel_attrs[key], reference_kernel_attrs[key])
                    for key in reference_kernel_attrs
                ):
                    raise ValueError(f"split {path} kernel attrs differ across splits")

        if load_maps:
            record["timing_sha256"] = msc.sha256_file(timing_path)
            record["map_dataset_sha256"] = map_hashes
        split_records.append(record)

    if observed_indices != list(range(num_splits)):
        raise ValueError(
            f"split IDs are not exactly 0..{num_splits - 1}: {observed_indices}"
        )
    assert reference_attrs is not None
    observed_helper_shas = frozenset(helper_sha_counts)
    if len(observed_helper_shas) > 1:
        if not allow_known_helper_sha_mix:
            raise ValueError(
                "split set has mixed pasting_helper_sha256 values: "
                f"{dict(sorted(helper_sha_counts.items()))}"
            )
        if not _is_registered_combine_only_helper_mix(observed_helper_shas):
            raise ValueError(
                "--allow-known-combine-only-helper-sha-mix accepts exactly the "
                "registered helper hash pair; observed "
                f"{dict(sorted(helper_sha_counts.items()))}"
            )
    if int(reference_attrs["nside"]) != int(info["nside"]):
        raise ValueError("split nside differs from the requested map resolution")
    if int(reference_attrs["num_splits"]) != num_splits:
        raise ValueError("split num_splits differs from the YAML contract")
    if str(reference_attrs["catalog_key"]) != str(info["catalog_key"]):
        raise ValueError("split catalog_key differs from the YAML contract")
    if str(reference_attrs["config_file_sha256"]) != str(info["config_sha256"]):
        raise ValueError("split maps were not produced by the current YAML config")
    if str(reference_attrs["catalog_file_sha256"]) != str(
        info["catalog_file_sha256"]
    ):
        raise ValueError("split maps do not reference the frozen catalog SHA")

    n_input = int(reference_attrs["n_input_halos"])
    if sum(observed_halo_counts) != n_input:
        raise ValueError(
            "split halo counts do not cover the catalog exactly: "
            f"{sum(observed_halo_counts)} != {n_input}"
        )
    strategy = str(reference_attrs["split_strategy"])
    if strategy in {"block_striped", "striped", "block-stripe", "block_stripe"}:
        expected_counts = _expected_block_striped_counts(
            n_input,
            num_splits,
            int(reference_attrs["split_block_halos"]),
        )
        if observed_halo_counts != expected_counts:
            raise ValueError(
                "block-striped split counts differ from the exact partition: "
                f"{observed_halo_counts}"
            )

    metadata: dict[str, Any] = {
        "source_kind": "strict_split_sum_in_memory",
        "num_splits": num_splits,
        "split_strategy": strategy,
        "n_input_halos": n_input,
        # Keep the scalar field for compatibility with the joint-product builder.
        "pasting_helper_sha256": _attribute_text(
            reference_attrs["pasting_helper_sha256"]
        ),
        "pasting_helper_sha256_counts": dict(sorted(helper_sha_counts.items())),
        "split_records": split_records,
    }
    if len(observed_helper_shas) > 1:
        metadata["accepted_contract_exception"] = {
            "attribute": "pasting_helper_sha256",
            "observed_sha256_counts": dict(sorted(helper_sha_counts.items())),
            "accepted_sha256_pair": sorted(KNOWN_COMBINE_ONLY_HELPER_SHA_MIX),
            "incident": "combine_partial_maps-only helper edit",
            "scientific_scope": (
                "registered helper edit did not alter run_paste_split or map content"
            ),
            "opt_in_flag": "--allow-known-combine-only-helper-sha-mix",
        }
    if not load_maps:
        return None, metadata

    assert maps_sum is not None
    if any(not np.all(np.isfinite(value)) for value in maps_sum.values()):
        raise ValueError("in-memory combined maps contain non-finite values")
    metadata["combined_map_dataset_sha256"] = {
        field: msc.sha256_array(maps_sum[field]) for field in msc.NOISE_FIELDS
    }
    metadata["combined_map_arrays_sha256"] = msc.sha256_array(
        *(maps_sum[field] for field in msc.NOISE_FIELDS)
    )
    maps_float64 = {
        field: np.asarray(maps_sum[field], dtype=np.float64)
        for field in msc.NOISE_FIELDS
    }
    return maps_float64, metadata


def _validate_map_layouts(
    ready: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    expected_shape: tuple[int, ...],
    msc,
    *,
    allow_known_helper_sha_mix: bool = False,
) -> list[dict[str, Any]]:
    """Fail fast on malformed combined products or incomplete split sets."""

    accepted_exceptions: list[dict[str, Any]] = []
    for position, entry, info in ready:
        try:
            if info["source_kind"] == "combined":
                _validate_combined_map_layout(
                    info, expected_shape, msc
                )
            elif info["source_kind"] == "splits":
                _, metadata = _process_split_set(
                    info,
                    expected_shape,
                    msc,
                    load_maps=False,
                    allow_known_helper_sha_mix=allow_known_helper_sha_mix,
                )
                if "accepted_contract_exception" in metadata:
                    accepted_exceptions.append(
                        {
                            "plan_position": int(position),
                            "entry_index": int(entry["index"]),
                            "run_name": str(entry["run_name"]),
                            **metadata["accepted_contract_exception"],
                        }
                    )
            else:
                raise ValueError(f"unknown source kind {info['source_kind']!r}")
        except Exception as error:
            raise RuntimeError(
                f"plan position {position} ({entry['run_name']}) failed "
                f"{type(error).__name__}: {error}"
            ) from error
    return accepted_exceptions


def _load_estimator_context(
    msc,
    remaps: Sequence[tuple[pathlib.Path, pathlib.Path]],
):
    """Load the frozen context without requiring changes to mock_sbi_common.

    With no relocation rules, use the repository's canonical loader directly.  If
    the HDF5 contract contains paths from another checkout, reproduce that loader
    locally while remapping only its registered workspace and map paths.  All
    scientific identity and content-hash checks remain enabled.
    """

    if not remaps:
        return msc.load_estimator_context()

    import h5py
    import pymaster as nmt

    contract_path = pathlib.Path(msc.NOISE_CONTRACT_PATH)
    contract_sha = msc.sha256_file(contract_path)
    with h5py.File(contract_path, "r") as handle:
        schema = str(handle.attrs["schema_version"])
        if schema not in msc.ACCEPTED_CONTRACT_SCHEMAS:
            raise ValueError(
                f"unexpected noise-contract schema {schema!r}; expected one of "
                f"{msc.ACCEPTED_CONTRACT_SCHEMAS}"
            )
        if str(handle.attrs["vector_order"]) != msc.VECTOR_ORDER:
            raise ValueError(
                "noise-contract vector ordering differs from the frozen inference order"
            )
        if (
            int(handle.attrs["nside"]) != msc.NSIDE
            or int(handle.attrs["lmax"]) != msc.LMAX
        ):
            raise ValueError(
                "noise-contract nside/lmax differ from the frozen inference resolution"
            )

        stored_mask = np.asarray(handle["mask"], dtype=np.float64)
        galaxy_alm = np.asarray(handle["fixed_masked_alm/g"])
        noise_cls = {
            name: np.asarray(
                handle[f"noise_cls/{'y_effective' if name == 'y' else name}"],
                dtype=np.float64,
            )
            for name in msc.NOISE_FIELDS
        }
        expected_noise = json.loads(
            str(handle.attrs["noise_dataset_sha256_json"])
        )
        band_edges = np.asarray(handle["band_edges"], dtype=np.int64)
        effective_ell = np.asarray(handle["effective_ell"], dtype=np.float64)
        window = np.asarray(handle["window"], dtype=np.float64)
        pixel_window_g = np.asarray(handle["pixel_window_g"], dtype=np.float64)
        covariance = np.asarray(handle["hmc/covariance"], dtype=np.float64)
        cholesky = np.asarray(handle["hmc/cholesky"], dtype=np.float64)
        signal_cls = {
            key: np.asarray(handle[f"signal_cls/{key}"], dtype=np.float64)
            for key in handle["signal_cls"]
        }
        fixed_bandpowers = {
            spectrum: np.asarray(
                handle[f"fixed_bandpowers/{spectrum}"], dtype=np.float64
            )
            for spectrum in msc.SPECTRA
        }
        theory_bandpowers = {
            spectrum: np.asarray(
                handle[f"theory_bandpowers/{spectrum}"], dtype=np.float64
            )
            for spectrum in msc.SPECTRA
        }
        workspace_path = _resolve_path(
            str(handle.attrs["workspace_path"]), msc.REPO_ROOT, remaps
        )
        workspace_sha = str(handle.attrs["workspace_sha256"])
        registered_map_path = _resolve_path(
            str(handle.attrs["map_path"]), msc.REPO_ROOT, remaps
        )

    if not np.array_equal(band_edges, msc.INFERENCE_EDGES):
        raise ValueError(
            "noise-contract band edges are not the frozen 14 inference bands"
        )
    for name, value in noise_cls.items():
        if msc.sha256_array(value) != expected_noise[name]:
            raise ValueError(f"noise curve hash mismatch for {name}")
    if not workspace_path.is_file():
        raise FileNotFoundError(
            f"remapped NaMaster workspace does not exist: {workspace_path}"
        )
    if msc.sha256_file(workspace_path) != workspace_sha:
        raise ValueError(
            f"NaMaster workspace hash mismatch after remapping: {workspace_path}"
        )
    expected_map_path = _resolve_path(
        msc.FROZEN_MAP_PATH, msc.REPO_ROOT, remaps
    )
    if registered_map_path != expected_map_path:
        raise ValueError(
            "noise contract points at an unexpected map after remapping: "
            f"{registered_map_path} != {expected_map_path}"
        )
    if covariance.shape != (msc.VECTOR_SIZE, msc.VECTOR_SIZE) or not np.array_equal(
        covariance, covariance.T
    ):
        raise ValueError(
            f"frozen covariance is not a symmetric {msc.VECTOR_SIZE}x"
            f"{msc.VECTOR_SIZE} matrix"
        )
    if window.shape != (msc.N_BAND, msc.LMAX + 1):
        raise ValueError(f"unexpected bandpower window shape {window.shape}")

    mask, mask_metadata = msc.canonical_mask(stored_mask)
    workspace = nmt.NmtWorkspace.from_file(str(workspace_path))
    return msc.EstimatorContext(
        mask=mask,
        mask_sum=float(np.sum(mask)),
        stored_mask_float32=stored_mask,
        mask_metadata=mask_metadata,
        fixed_galaxy_alm=galaxy_alm,
        workspace=workspace,
        noise_cls=noise_cls,
        band_edges=band_edges,
        effective_ell=effective_ell,
        window=window,
        pixel_window_g=pixel_window_g,
        covariance=covariance,
        cholesky=cholesky,
        signal_cls=signal_cls,
        fixed_bandpowers=fixed_bandpowers,
        theory_bandpowers=theory_bandpowers,
        contract_sha256=contract_sha,
        workspace_sha256=workspace_sha,
        mask_sha256=msc.sha256_array(mask),
        stored_mask_sha256=msc.sha256_array(stored_mask),
        galaxy_alm_sha256=msc.sha256_array(galaxy_alm),
    )


def _load_or_reconstruct_training_seed_bank(
    msc,
    count: int,
    requested_path: pathlib.Path | None,
    remaps: Sequence[tuple[pathlib.Path, pathlib.Path]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the Stage-5 seed table when present and prove reconstruction parity."""

    seed_records = msc.noise_bank_seeds(count, msc.NAMESPACE_TRAINING)
    reconstructed = np.asarray(
        [
            [record[name] for name in msc.NOISE_FIELDS]
            for record in seed_records
        ],
        dtype=np.uint32,
    )
    if reconstructed.shape != (count, len(msc.NOISE_FIELDS)):
        raise RuntimeError(
            f"reconstructed seed bank has invalid shape {reconstructed.shape}"
        )
    if np.unique(reconstructed.ravel()).size != reconstructed.size:
        raise RuntimeError("reconstructed seed bank contains a cross-field collision")
    leaked = sorted(
        msc.reserved_observation_seeds().intersection(
            reconstructed.ravel().tolist()
        )
    )
    if leaked:
        raise RuntimeError(f"training seed bank reuses observation seeds: {leaked}")

    official = msc.REPO_ROOT / "data/SBI_validate/mock_sbi/noise_bank_training.npz"
    bank_path = (
        _resolve_path(requested_path, msc.REPO_ROOT, remaps)
        if requested_path is not None
        else official.resolve()
    )
    if not bank_path.is_file():
        if requested_path is not None:
            raise FileNotFoundError(f"requested noise bank does not exist: {bank_path}")
        return reconstructed, {
            "source_kind": "deterministic_reconstruction",
            "expected_official_path": str(bank_path),
            "official_file_present": False,
            "namespace": list(msc.NAMESPACE_TRAINING),
            "field_order": list(msc.NOISE_FIELDS),
        }

    with np.load(bank_path, allow_pickle=False) as payload:
        required = {"seeds", "seed_field_order"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise KeyError(f"noise bank {bank_path} is missing arrays {missing}")
        stored = np.asarray(payload["seeds"], dtype=np.uint32)
        field_order = tuple(str(value) for value in payload["seed_field_order"])
    if stored.shape != reconstructed.shape:
        raise ValueError(
            f"noise bank seed shape {stored.shape} != requested/reconstructed "
            f"shape {reconstructed.shape}; set --bank-size to the stored bank size"
        )
    if field_order != tuple(msc.NOISE_FIELDS):
        raise ValueError(
            f"noise bank field order {field_order} != {tuple(msc.NOISE_FIELDS)}"
        )
    if not np.array_equal(stored, reconstructed):
        raise RuntimeError(
            "stored noise-bank seeds differ from mock_sbi_common.noise_bank_seeds"
        )
    return stored, {
        "source_kind": "stored_bank_verified_against_reconstruction",
        "path": str(bank_path),
        "sha256": msc.sha256_file(bank_path),
        "official_file_present": True,
        "namespace": list(msc.NAMESPACE_TRAINING),
        "field_order": list(msc.NOISE_FIELDS),
    }


def _resolve_and_verify_frozen_galaxy_map(
    msc,
    remaps: Sequence[tuple[pathlib.Path, pathlib.Path]],
) -> tuple[pathlib.Path, str, dict[str, Any]]:
    """Resolve the exact catalog and its frozen pixel-field scalars."""

    import h5py

    contract_path = pathlib.Path(msc.NOISE_CONTRACT_PATH)
    with h5py.File(contract_path, "r") as handle:
        required_attrs = {
            "map_path",
            "map_sha256",
            "galaxy_report_json",
            "galaxy_mean_count_per_masked_pixel",
            "field_weighted_means_json",
        }
        missing_attrs = sorted(required_attrs - set(handle.attrs))
        if missing_attrs:
            raise ValueError(
                f"frozen noise contract {contract_path} is missing attrs "
                f"{missing_attrs}"
            )
        registered_path = _resolve_path(
            _attribute_text(handle.attrs["map_path"]), msc.REPO_ROOT, remaps
        )
        expected_sha = _attribute_text(handle.attrs["map_sha256"])
        try:
            expected_report = json.loads(
                _attribute_text(handle.attrs["galaxy_report_json"])
            )
            expected_means = json.loads(
                _attribute_text(handle.attrs["field_weighted_means_json"])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"frozen noise contract {contract_path} has invalid galaxy JSON"
            ) from exc
        expected_mean_count = float(
            handle.attrs["galaxy_mean_count_per_masked_pixel"]
        )

    if not isinstance(expected_report, Mapping) or not expected_report:
        raise ValueError("noise contract galaxy_report_json is not a non-empty object")
    if not isinstance(expected_means, Mapping) or "g_overdensity_removed" not in expected_means:
        raise ValueError(
            "noise contract field_weighted_means_json has no "
            "g_overdensity_removed"
        )
    expected_removed_mean = float(expected_means["g_overdensity_removed"])
    if not np.isfinite(expected_mean_count) or expected_mean_count <= 0.0:
        raise ValueError(
            "noise contract has invalid galaxy_mean_count_per_masked_pixel "
            f"{expected_mean_count}"
        )
    if not np.isfinite(expected_removed_mean):
        raise ValueError(
            "noise contract has non-finite g_overdensity_removed "
            f"{expected_removed_mean}"
        )

    expected_path = _resolve_path(msc.FROZEN_MAP_PATH, msc.REPO_ROOT, remaps)
    if registered_path != expected_path:
        raise ValueError(
            "frozen noise contract points at an unexpected galaxy catalog after "
            f"path remapping: {registered_path} != {expected_path}"
        )
    if not registered_path.is_file():
        raise FileNotFoundError(
            f"frozen galaxy catalog does not exist: {registered_path}"
        )
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError(
            f"noise contract has an invalid frozen-map SHA-256: {expected_sha!r}"
        )
    observed_sha = msc.sha256_file(registered_path)
    if observed_sha != expected_sha:
        raise ValueError(
            f"frozen galaxy catalog SHA-256 {observed_sha} != contract "
            f"{expected_sha}: {registered_path}"
        )
    return registered_path, observed_sha, {
        "galaxy_report": {
            str(key): int(value) for key, value in expected_report.items()
        },
        "mean_count_per_masked_pixel": expected_mean_count,
        "removed_delta_g_mean": expected_removed_mean,
        "source": str(contract_path),
    }


def _build_fixed_g2_alm(
    ctx,
    galaxy_map_path: pathlib.Path,
    galaxy_map_sha256: str,
    frozen_galaxy_contract: Mapping[str, Any],
    hp,
    msc,
):
    from three_probe_noiseless_estimator import build_galaxy_count_map

    counts, galaxy_report = build_galaxy_count_map(
        galaxy_map_path,
        nside=msc.NSIDE,
    )
    expected_report = dict(frozen_galaxy_contract["galaxy_report"])
    observed_report = {str(key): int(value) for key, value in galaxy_report.items()}
    if observed_report != expected_report:
        raise RuntimeError(
            "reconstructed galaxy selection report differs from the frozen "
            f"contract: {observed_report} != {expected_report}"
        )

    # These are the exact scalars used when the frozen galaxy overdensity was
    # created.  Reusing them avoids turning a platform-dependent reduction in a
    # regenerated float64 mask into a different galaxy normalization.
    mean_count = float(frozen_galaxy_contract["mean_count_per_masked_pixel"])
    removed_delta_mean = float(frozen_galaxy_contract["removed_delta_g_mean"])
    delta_g = _overdensity_from_frozen_scalars(
        counts,
        mean_count,
        removed_delta_mean,
    )

    # Diagnostic only.  The exact input catalog is SHA-pinned above, but the
    # final float64 bits of libsharp's map2alm are not portable across execution
    # environments.  This transform is not used to construct g2 below.
    reconstructed_g_alm = hp.map2alm(
        ctx.mask * delta_g,
        lmax=msc.LMAX,
        iter=0,
    )
    reconstructed_sha = msc.sha256_array(reconstructed_g_alm)
    if reconstructed_g_alm.shape != ctx.fixed_galaxy_alm.shape:
        raise RuntimeError(
            f"reconstructed fixed galaxy alm shape {reconstructed_g_alm.shape} "
            f"!= frozen {ctx.fixed_galaxy_alm.shape}"
        )
    if not np.all(np.isfinite(reconstructed_g_alm)):
        raise RuntimeError("diagnostic reconstructed galaxy alm is non-finite")
    maximum_difference = float(
        np.max(np.abs(reconstructed_g_alm - ctx.fixed_galaxy_alm))
    )
    bitwise_match = reconstructed_sha == ctx.galaxy_alm_sha256

    g2 = np.square(np.asarray(delta_g, dtype=np.float64))
    g2_centered, removed_g2_mean = msc.subtract_weighted_mean(g2, ctx.mask)
    g2_alm = hp.map2alm(
        ctx.mask * g2_centered,
        lmax=msc.LMAX,
        iter=0,
    )
    if not np.all(np.isfinite(g2_alm)):
        raise RuntimeError("fixed squared-galaxy alm contains non-finite values")

    return g2_alm, {
        "catalog_path": str(galaxy_map_path),
        "catalog_sha256": str(galaxy_map_sha256),
        "catalog_identity": "whole-file SHA-256 verified against noise contract",
        "galaxy_report": observed_report,
        "galaxy_report_identity": "exact equality with noise-contract JSON",
        "galaxy_count_map_sha256": msc.sha256_array(counts),
        "mean_count_per_masked_pixel": float(mean_count),
        "removed_delta_g_mean": float(removed_delta_mean),
        "galaxy_normalization_source": (
            "frozen noise-contract galaxy_mean_count_per_masked_pixel and "
            "field_weighted_means_json[g_overdensity_removed]"
        ),
        "removed_delta_g_squared_mean": float(removed_g2_mean),
        "reference_galaxy_alm_sha256": ctx.galaxy_alm_sha256,
        "diagnostic_reconstructed_galaxy_alm_sha256": reconstructed_sha,
        "diagnostic_reconstructed_galaxy_alm_bitwise_match": bitwise_match,
        "diagnostic_reconstructed_galaxy_alm_max_abs_difference": maximum_difference,
        "diagnostic_reconstructed_galaxy_alm_policy": (
            "reported only; exact catalog SHA, selection report, and frozen "
            "normalization scalars define identity because libsharp float64 "
            "last bits are execution-environment dependent"
        ),
        "g2_alm_sha256": msc.sha256_array(g2_alm),
        "construction": (
            "delta_g normalized and demeaned on the common mask; delta_g squared "
            "before masking; mask-weighted mean of delta_g_squared removed; mask "
            "applied once before map2alm"
        ),
        "shot_noise_subtraction": "none; fixed realized galaxy field",
    }


def _overdensity_from_frozen_scalars(
    counts: np.ndarray,
    mean_count: float,
    removed_mean: float,
) -> np.ndarray:
    """Rebuild the contract's pixel overdensity without a new mask reduction."""

    count_array = np.asarray(counts)
    if count_array.ndim != 1 or np.any(count_array < 0):
        raise ValueError("galaxy counts must be a non-negative one-dimensional array")
    mean_count = float(mean_count)
    removed_mean = float(removed_mean)
    if not np.isfinite(mean_count) or mean_count <= 0.0:
        raise ValueError(f"invalid frozen galaxy mean count {mean_count}")
    if not np.isfinite(removed_mean):
        raise ValueError(f"invalid frozen removed galaxy mean {removed_mean}")
    result = (
        np.asarray(count_array, dtype=np.float64) / mean_count
        - 1.0
        - removed_mean
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("reconstructed frozen galaxy overdensity is non-finite")
    return result


def _measure_noisy_g2x(
    signal_maps: Mapping[str, np.ndarray],
    seed_row: np.ndarray,
    ctx,
    g2_alm: np.ndarray,
    hp,
    msc,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = {
        name: int(seed_row[position])
        for position, name in enumerate(msc.NOISE_FIELDS)
    }
    blocks: list[np.ndarray] = []
    removed_means: list[float] = []

    for _, field in G2X_SPECTRA:
        signal = np.asarray(signal_maps[field], dtype=np.float64)
        if signal.shape != ctx.mask.shape or not np.all(np.isfinite(signal)):
            raise ValueError(
                f"{field} signal map has invalid shape/content: {signal.shape}"
            )
        noise_alm = msc.synalm_seeded(ctx.noise_cls[field], seeds[field])
        noise_map = hp.alm2map(
            noise_alm,
            nside=msc.NSIDE,
            lmax=msc.LMAX,
        )
        if noise_map.shape != ctx.mask.shape or not np.all(np.isfinite(noise_map)):
            raise RuntimeError(
                f"{field} noise map has invalid shape/content: {noise_map.shape}"
            )
        total_map = signal + noise_map
        centered, removed_mean = msc.subtract_weighted_mean(total_map, ctx.mask)
        total_masked_alm = hp.map2alm(
            ctx.mask * centered,
            lmax=msc.LMAX,
            iter=0,
        )
        coupled = hp.alm2cl(g2_alm, total_masked_alm, lmax=msc.LMAX)
        bandpowers = np.asarray(
            ctx.workspace.decouple_cell(coupled[None, :]),
            dtype=np.float64,
        )[0]
        if bandpowers.shape != (ctx.effective_ell.size,):
            raise RuntimeError(
                f"{field} spectrum has shape {bandpowers.shape}; expected "
                f"{(ctx.effective_ell.size,)}"
            )
        if not np.all(np.isfinite(bandpowers)):
            raise RuntimeError(f"g2{field} bandpowers contain non-finite values")
        blocks.append(bandpowers)
        removed_means.append(float(removed_mean))

        del noise_alm, noise_map, total_map, centered, total_masked_alm, coupled

    vector = np.concatenate(blocks)
    expected_size = len(G2X_SPECTRA) * int(ctx.effective_ell.size)
    if vector.shape != (expected_size,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(
            f"g2X vector has invalid shape/content {vector.shape}; expected "
            f"{(expected_size,)}"
        )
    return vector, np.asarray(removed_means, dtype=np.float64)


def _run_self_test() -> int:
    import tempfile

    if KNOWN_COMBINE_ONLY_HELPER_SHA_MIX != {
        "2044f85df2206c9b53bb9fa2ab5916dbdc0d8297d169ea27c073196640ea773e",
        "121c9783447948149ed5cd7d53fbe272dd503fcf2f1b53d09268a04a3042dd3d",
    }:
        raise AssertionError("registered helper-SHA pair changed")
    if not _is_registered_combine_only_helper_mix(
        tuple(KNOWN_COMBINE_ONLY_HELPER_SHA_MIX)
    ):
        raise AssertionError("registered helper-SHA pair was not recognized")
    if _is_registered_combine_only_helper_mix(
        (next(iter(KNOWN_COMBINE_ONLY_HELPER_SHA_MIX)),)
    ):
        raise AssertionError("one helper SHA was incorrectly accepted as a mix")
    if _is_registered_combine_only_helper_mix(("unknown-a", "unknown-b")):
        raise AssertionError("an unknown helper-SHA pair was incorrectly accepted")

    frozen_delta = _overdensity_from_frozen_scalars(
        np.asarray([0, 2, 5], dtype=np.int64),
        mean_count=2.0,
        removed_mean=0.25,
    )
    np.testing.assert_array_equal(
        frozen_delta,
        np.asarray([-1.25, -0.25, 1.25], dtype=np.float64),
    )
    try:
        _overdensity_from_frozen_scalars(
            np.asarray([0, 1], dtype=np.int64),
            mean_count=0.0,
            removed_mean=0.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid frozen galaxy normalization did not fail")

    first = assign_bank_indices("round_1", 4, 5, 32)
    second = assign_bank_indices("round_1", 4, 5, 32)
    np.testing.assert_array_equal(first, second)
    if any(np.unique(row).size != row.size for row in first):
        raise AssertionError("assignment self-test produced a within-point duplicate")
    if np.array_equal(first, assign_bank_indices("round_2", 4, 5, 32)):
        raise AssertionError("round-label separation self-test failed")

    stage5_full, selected = select_stage5_bank_indices(
        "round_1",
        n_points=4,
        selected_replicas=5,
        stage5_replicas=64,
        bank_size=2048,
    )
    np.testing.assert_array_equal(selected, stage5_full[:, :5])
    if selected.shape != (4, 5):
        raise AssertionError(
            f"Stage-5 prefix selection has shape {selected.shape}; expected (4, 5)"
        )
    direct_five = assign_bank_indices("round_1", 4, 5, 2048)
    if np.array_equal(selected, direct_five):
        raise AssertionError(
            "self-test failed to distinguish the first five columns of a "
            "64-wide Stage-5 assignment from a directly generated five-wide assignment"
        )

    rules = _parse_path_remaps(["/old/GODMAX=/new/GODMAX"])
    mapped = _resolve_path(
        "/old/GODMAX/data/map.h5",
        pathlib.Path("/unused"),
        rules,
    )
    if mapped != pathlib.Path("/new/GODMAX/data/map.h5"):
        raise AssertionError(f"path-remap self-test returned {mapped}")
    default_output = _training_output_path(
        pathlib.Path("/repo/data/round_1/paste_plan.json"), None, 5
    )
    if default_output != pathlib.Path(
        "/repo/data/round_1/training_set_g2x_rep5.npz"
    ):
        raise AssertionError(f"default-output self-test returned {default_output}")
    timing = _timing_path(pathlib.Path("map_split031of032.h5"))
    if timing != pathlib.Path("map_split031of032.h5.timing.json"):
        raise AssertionError(f"split-sidecar self-test returned {timing}")
    if _expected_block_striped_counts(10, 3, 2) != [4, 4, 2]:
        raise AssertionError("block-striped split partition self-test failed")

    class ManifestTestMsc:
        @staticmethod
        def sha256_file(path):
            return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = pathlib.Path(temporary_directory)
        config = root / "mocksbi_test.yaml"
        config.write_text("pasting: {}\n")
        config_sha = ManifestTestMsc.sha256_file(config)
        manifest = root / "helper_sha_repaste.json"
        manifest.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "config_path": "/another/checkout/mocksbi_test.yaml",
                            "repaste_splits": [1],
                        }
                    ]
                }
            )
        )
        test_plan = {
            "entries": [
                {
                    "index": 7,
                    "run_name": "mocksbi_test",
                    "config_path": str(config),
                    "config_sha256": config_sha,
                }
            ]
        }
        excluded, metadata = _load_exclusion_manifest(
            manifest,
            test_plan,
            root,
            (),
            ManifestTestMsc,
        )
        if excluded != {0} or metadata["records"][0]["entry_index"] != 7:
            raise AssertionError("exclusion-manifest matching self-test failed")

    theta = {name: float(index + 1) for index, name in enumerate(PARAMETER_NAMES)}
    entry = {
        "index": 0,
        "run_name": "mocksbi_test",
        "config_path": "config.yaml",
        "config_sha256": "config-sha",
        "theta": theta,
        "theta_sha256": _canonical_json_sha256(theta),
        "u": [0.0] * len(PARAMETER_NAMES),
        "log_q": -1.0,
        "importance_eligible": True,
    }
    temporary_plan = {
        "schema_version": PLAN_SCHEMA,
        "round": "1",
        "count": 1,
        "parameter_names": list(PARAMETER_NAMES),
        "entries": [entry],
    }
    # Exercise the validator without writing a temporary file.
    decoded = json.loads(json.dumps(temporary_plan))
    if decoded["entries"][0]["theta_sha256"] != entry["theta_sha256"]:
        raise AssertionError("plan JSON round-trip changed theta provenance")

    compacted_plan = dict(temporary_plan)
    compacted_entry = dict(entry)
    compacted_entry["index"] = 7
    compacted_plan["entries"] = [compacted_entry]
    with tempfile.TemporaryDirectory() as temporary_directory:
        compacted_path = pathlib.Path(temporary_directory) / "ready_plan.json"
        compacted_path.write_text(json.dumps(compacted_plan))
        try:
            _load_and_validate_plan(compacted_path)
        except ValueError as error:
            if "compacted or reordered plan" not in str(error):
                raise AssertionError(
                    "compacted-plan self-test raised the wrong error"
                ) from error
        else:
            raise AssertionError("compacted-plan self-test did not fail closed")

    class FakeWorkspace:
        @staticmethod
        def decouple_cell(coupled):
            return np.asarray(coupled)

    class FakeContext:
        mask = np.asarray([1.0, 0.5])
        effective_ell = np.asarray([10.0, 20.0])
        workspace = FakeWorkspace()
        noise_cls = {
            "y": np.asarray([0.1, 0.2]),
            "tau": np.asarray([0.3, 0.4]),
            "kappa": np.asarray([0.5, 0.6]),
        }

    class FakeHealpy:
        @staticmethod
        def alm2map(alm, *, nside, lmax):
            del nside, lmax
            return np.asarray(alm)

        @staticmethod
        def map2alm(values, *, lmax, iter):
            del lmax, iter
            return np.asarray(values)

        @staticmethod
        def alm2cl(left, right, *, lmax):
            del lmax
            return np.asarray(left) * np.asarray(right)

    class FakeMsc:
        NOISE_FIELDS = ("y", "tau", "kappa")
        NSIDE = 1
        LMAX = 1

        @staticmethod
        def synalm_seeded(noise_cl, seed):
            return np.asarray(noise_cl) + 0.01 * int(seed)

        @staticmethod
        def subtract_weighted_mean(values, mask):
            mean = float(np.sum(values * mask) / np.sum(mask))
            return np.asarray(values) - mean, mean

    signal_maps = {
        "y": np.asarray([1.0, 2.0]),
        "tau": np.asarray([3.0, 5.0]),
        "kappa": np.asarray([7.0, 11.0]),
    }
    seed_row = np.asarray([11, 22, 33], dtype=np.uint32)
    g2_alm = np.asarray([2.0, -1.0])
    vector, means = _measure_noisy_g2x(
        signal_maps,
        seed_row,
        FakeContext(),
        g2_alm,
        FakeHealpy(),
        FakeMsc(),
    )
    expected_blocks = []
    expected_means = []
    seed_by_field = dict(zip(FakeMsc.NOISE_FIELDS, seed_row, strict=True))
    for _, field in G2X_SPECTRA:
        total = (
            signal_maps[field]
            + FakeContext.noise_cls[field]
            + 0.01 * int(seed_by_field[field])
        )
        centered, mean = FakeMsc.subtract_weighted_mean(total, FakeContext.mask)
        expected_blocks.append(g2_alm * FakeContext.mask * centered)
        expected_means.append(mean)
    np.testing.assert_allclose(vector, np.concatenate(expected_blocks))
    np.testing.assert_allclose(means, expected_means)

    print(
        "SELF-TEST PASS: frozen galaxy normalization, assignment construction, "
        "path remapping, map-level addition, split partitioning, exclusion "
        "matching, default output, and g2y/g2kappa/g2tau ordering"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plan",
        type=pathlib.Path,
        help="existing paste_plan.json containing the mocksbi YAML entries",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help=(
            "new NPZ file to create; default is training_set_g2x_repK.npz "
            "beside the plan"
        ),
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=DEFAULT_REPLICAS,
        help=(
            "number of Stage-5 written-map replicas to measure per available "
            "point; selects columns 0..K-1 from the full Stage-5 assignment"
        ),
    )
    parser.add_argument(
        "--stage5-replicas",
        type=int,
        default=DEFAULT_STAGE5_REPLICAS,
        help=(
            "width used when reproducing Stage 5's assignment; default 64 "
            "matches 16_measure_augment_round1.sbatch, whose first five columns "
            "were the intended saved maps"
        ),
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=DEFAULT_BANK_SIZE,
        help="training seed-bank size; 2048 reproduces 00_noise_bank.sbatch",
    )
    parser.add_argument(
        "--noise-bank",
        type=pathlib.Path,
        default=None,
        help=(
            "optional existing noise_bank_training.npz; when omitted, the "
            "official path is verified if present, otherwise its seed table is "
            "reconstructed from mock_sbi_common"
        ),
    )
    parser.add_argument(
        "--assignment-label",
        default=None,
        help="assignment label; default is the plan directory name (for example round_1)",
    )
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        metavar="OLD_PREFIX=NEW_PREFIX",
        help=(
            "rewrite absolute paths embedded in the plan, YAML and frozen "
            "contract; repeat as needed"
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "measure only points with a combined map or complete split set; "
            "assignments remain tied to full-plan positions"
        ),
    )
    parser.add_argument(
        "--allow-known-combine-only-helper-sha-mix",
        action="store_true",
        help=(
            "accept only the registered pre/post helper SHA pair whose edit "
            "was confined to combine_partial_maps; all other contract "
            "mismatches remain fatal and accepted points are recorded"
        ),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=pathlib.Path,
        help=(
            "JSON manifest whose entries[*].config_path names whole plan points "
            "to omit; use the existing helper_sha_repaste.json for the known "
            "mixed-helper incident instead of weakening split validation"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output atomically",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "validate paths, hashes, the frozen estimator, HDF5 map layouts and "
            "fixed g2 reconstruction; print the output target and exit without writing"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run dependency-free contract checks and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    if args.plan is None:
        parser.error("--plan is required unless --self-test is used")
    if args.replicas <= 0:
        parser.error("--replicas must be positive")
    if args.stage5_replicas <= 0:
        parser.error("--stage5-replicas must be positive")
    if args.replicas > args.stage5_replicas:
        parser.error("--replicas cannot exceed --stage5-replicas")
    if args.bank_size <= 0:
        parser.error("--bank-size must be positive")
    if args.stage5_replicas > args.bank_size:
        parser.error("--stage5-replicas cannot exceed --bank-size")

    plan_path = args.plan.expanduser().resolve()
    output_path = _training_output_path(plan_path, args.output, args.replicas)
    sidecar_path = output_path.with_suffix(".json")
    if not plan_path.is_file():
        raise FileNotFoundError(f"paste plan does not exist: {plan_path}")
    if output_path == plan_path:
        raise ValueError("--output must differ from --plan")
    existing = [path for path in (output_path, sidecar_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {existing[0]}; pass --overwrite to replace it"
        )
    remaps = _parse_path_remaps(args.path_remap)
    started = time.time()

    # Scientific imports are lazy so --help and --self-test work on login nodes.
    import healpy as hp
    import mock_sbi_common as msc

    print("[1/6] loading the paste plan and locating map products ...", flush=True)
    plan = _load_and_validate_plan(plan_path)
    if int(plan.get("nside", -1)) != int(msc.NSIDE):
        raise ValueError(
            f"plan nside {plan.get('nside')} != frozen estimator nside {msc.NSIDE}"
        )
    entries = plan["entries"]
    excluded_positions: set[int] = set()
    exclusion_metadata: dict[str, Any] | None = None
    if args.exclude_manifest is not None:
        excluded_positions, exclusion_metadata = _load_exclusion_manifest(
            args.exclude_manifest,
            plan,
            msc.REPO_ROOT,
            remaps,
            msc,
        )
        print(
            f"      excluding {len(excluded_positions)} whole point(s) from "
            f"{exclusion_metadata['path']}",
            flush=True,
        )
    ready: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    missing_maps: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        if position in excluded_positions:
            continue
        info = _entry_map_info(entry, msc.REPO_ROOT, remaps, msc)
        if info["nside"] != msc.NSIDE:
            raise ValueError(
                f"entry {position} nside {info['nside']} != {msc.NSIDE}"
            )
        combined_path = pathlib.Path(info["combined_map_path"])
        split_paths = tuple(pathlib.Path(path) for path in info["split_paths"])
        complete_splits = sum(
            _is_nonempty_file(path) and _is_nonempty_file(_timing_path(path))
            for path in split_paths
        )
        present_split_h5 = sum(_is_nonempty_file(path) for path in split_paths)
        if _is_nonempty_file(combined_path):
            info["source_kind"] = "combined"
        elif split_paths and complete_splits == int(info["num_splits"]):
            info["source_kind"] = "splits"
        else:
            missing_maps.append(
                {
                    "plan_position": position,
                    "entry_index": int(entry["index"]),
                    "run_name": str(entry["run_name"]),
                    "expected_combined_map": str(combined_path),
                    "present_split_h5": present_split_h5,
                    "completed_splits": complete_splits,
                    "required_splits": int(info["num_splits"]),
                }
            )
            continue
        ready.append((position, entry, info))

    if missing_maps and not args.allow_missing:
        raise FileNotFoundError(
            f"{len(missing_maps)} map point(s) have neither a combined file nor a "
            f"complete split set. First missing point expects "
            f"{missing_maps[0]['expected_combined_map']} or "
            f"{missing_maps[0]['required_splits']} completed splits, but has "
            f"{missing_maps[0]['completed_splits']}. Use --allow-missing only if "
            "a partial training set is intentional."
        )
    if not ready:
        raise FileNotFoundError(
            "none of the plan points has a combined map or complete split set"
        )
    n_combined = sum(info["source_kind"] == "combined" for _, _, info in ready)
    n_split_sets = sum(info["source_kind"] == "splits" for _, _, info in ready)
    print(
        f"      {len(ready)} of {len(entries)} points ready "
        f"({len(excluded_positions)} excluded): "
        f"{n_combined} combined, {n_split_sets} complete split sets",
        flush=True,
    )

    print("[2/6] loading/verifying Stage-5 seeds and assignments ...", flush=True)
    bank_seeds, seed_bank_provenance = _load_or_reconstruct_training_seed_bank(
        msc,
        args.bank_size,
        args.noise_bank,
        remaps,
    )
    assignment_label = args.assignment_label or plan_path.parent.name
    stage5_full_bank_indices, selected_full_bank_indices = (
        select_stage5_bank_indices(
            assignment_label,
            len(entries),
            args.replicas,
            args.stage5_replicas,
            args.bank_size,
        )
    )
    assignment_provenance = _verify_stage5_assignment_parity(
        assignment_label,
        len(entries),
        args.stage5_replicas,
        args.bank_size,
        stage5_full_bank_indices,
    )
    ready_positions = np.asarray([position for position, _, _ in ready], dtype=np.int64)
    point_bank_indices = selected_full_bank_indices[ready_positions]
    print(
        f"      label {assignment_label!r}; {len(ready)} points x "
        f"{args.replicas} selected replicas from the {args.stage5_replicas}-wide "
        f"Stage-5 assignment; seed order {msc.NOISE_FIELDS}",
        flush=True,
    )

    print("[3/6] loading and verifying the frozen estimator ...", flush=True)
    ctx = _load_estimator_context(msc, remaps)
    n_band = int(ctx.effective_ell.size)
    vector_size = len(G2X_SPECTRA) * n_band
    if n_band != 14 or vector_size != 42:
        raise RuntimeError(
            f"frozen estimator has {n_band} bands ({vector_size} g2X entries); "
            "this product requires 14 bands and 42 entries"
        )
    identity = {
        "noise_contract_sha256": ctx.contract_sha256,
        "workspace_sha256": ctx.workspace_sha256,
        "mask_array_sha256": ctx.mask_sha256,
        "fixed_galaxy_alm_sha256": ctx.galaxy_alm_sha256,
    }

    (
        galaxy_map,
        galaxy_map_sha256,
        frozen_galaxy_contract,
    ) = _resolve_and_verify_frozen_galaxy_map(
        msc,
        remaps,
    )

    print("      checking combined-map and split-set contracts ...", flush=True)
    accepted_contract_exceptions = _validate_map_layouts(
        ready,
        tuple(ctx.mask.shape),
        msc,
        allow_known_helper_sha_mix=(
            args.allow_known_combine_only_helper_sha_mix
        ),
    )
    if accepted_contract_exceptions:
        print(
            "      WARNING: accepted the registered combine-only helper-SHA "
            f"mix for {len(accepted_contract_exceptions)} point(s)",
            flush=True,
        )

    print("[4/6] reconstructing and squaring the fixed galaxy field ...", flush=True)
    g2_alm, galaxy_metadata = _build_fixed_g2_alm(
        ctx,
        galaxy_map,
        galaxy_map_sha256,
        frozen_galaxy_contract,
        hp,
        msc,
    )
    print(
        f"      selected {galaxy_metadata['galaxy_report']['rows_selected']:,} galaxies; "
        f"g2 alm {galaxy_metadata['g2_alm_sha256'][:16]}",
        flush=True,
    )
    if not galaxy_metadata["diagnostic_reconstructed_galaxy_alm_bitwise_match"]:
        print(
            "      diagnostic: regenerated g alm differs from the stored "
            "libsharp result by max |delta|="
            f"{galaxy_metadata['diagnostic_reconstructed_galaxy_alm_max_abs_difference']:.6e}; "
            "the exact catalog/report/frozen-normalization gates passed",
            flush=True,
        )

    if args.preflight:
        print("\nPREFLIGHT PASS")
        print(f"      plan: {plan_path}")
        print(
            f"      map sources: {n_combined} combined, "
            f"{n_split_sets} complete split sets"
        )
        print(f"      excluded whole points: {len(excluded_positions)}")
        print(
            "      accepted registered helper-SHA exceptions: "
            f"{len(accepted_contract_exceptions)}"
        )
        print(f"      rows to measure: {len(ready) * args.replicas}")
        print(f"      fixed g2 alm sha256: {galaxy_metadata['g2_alm_sha256']}")
        print(f"      output to create: {output_path}")
        print(f"      manifest to create: {sidecar_path}")
        print("      no files were written")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("[5/6] generating noisy maps in memory and measuring g2X ...", flush=True)
    n_points = len(ready)
    n_rows = n_points * args.replicas
    vectors = np.empty((n_rows, vector_size), dtype=np.float64)
    removed_means = np.empty((n_rows, len(G2X_SPECTRA)), dtype=np.float64)
    row_theta = np.empty((n_rows, len(PARAMETER_NAMES)), dtype=np.float64)
    row_u = np.empty_like(row_theta)
    row_log_q = np.empty(n_rows, dtype=np.float64)
    row_eligible = np.empty(n_rows, dtype=bool)
    row_point = np.empty(n_rows, dtype=np.int64)
    row_replica = np.empty(n_rows, dtype=np.int64)
    row_plan_position = np.empty(n_rows, dtype=np.int64)
    row_entry_index = np.empty(n_rows, dtype=np.int64)
    row_bank_index = np.empty(n_rows, dtype=np.int64)
    row_noise_seeds = np.empty((n_rows, len(msc.NOISE_FIELDS)), dtype=np.uint32)
    point_theta = np.empty((n_points, len(PARAMETER_NAMES)), dtype=np.float64)
    point_u = np.empty_like(point_theta)
    point_log_q = np.empty(n_points, dtype=np.float64)
    point_eligible = np.empty(n_points, dtype=bool)
    source_map_records: list[dict[str, Any]] = []
    seen_map_hashes: dict[str, str] = {}

    output_row = 0
    for output_point, (plan_position, entry, info) in enumerate(ready):
        theta = np.asarray(
            [entry["theta"][name] for name in PARAMETER_NAMES],
            dtype=np.float64,
        )
        u = (
            np.full(len(PARAMETER_NAMES), np.nan, dtype=np.float64)
            if entry["u"] is None
            else np.asarray(entry["u"], dtype=np.float64)
        )
        log_q = np.nan if entry["log_q"] is None else float(entry["log_q"])
        eligible = bool(entry["importance_eligible"])
        point_theta[output_point] = theta
        point_u[output_point] = u
        point_log_q[output_point] = log_q
        point_eligible[output_point] = eligible

        source_kind = str(info["source_kind"])
        if source_kind == "combined":
            map_path = pathlib.Path(info["combined_map_path"])
            map_file_sha = msc.sha256_file(map_path)
            expected_map_sha = str(info["expected_map_sha256"])
            if expected_map_sha and map_file_sha != expected_map_sha:
                raise ValueError(
                    f"cached map changed for {entry['run_name']}: "
                    f"{map_file_sha} != {expected_map_sha}"
                )
            signal_maps = msc.read_paste_maps(map_path)
            source_metadata: dict[str, Any] = {
                "source_kind": "combined_hdf5",
                "map_path": str(map_path),
                "map_sha256": map_file_sha,
            }
        elif source_kind == "splits":
            signal_maps, split_metadata = _process_split_set(
                info,
                tuple(ctx.mask.shape),
                msc,
                load_maps=True,
                allow_known_helper_sha_mix=(
                    args.allow_known_combine_only_helper_sha_mix
                ),
            )
            assert signal_maps is not None
            source_metadata = dict(split_metadata)
            source_metadata["combined_map_path_not_present"] = str(
                info["combined_map_path"]
            )
        else:
            raise RuntimeError(f"unsupported map source kind {source_kind!r}")

        signal_map_sha = msc.sha256_array(
            *(
                np.asarray(signal_maps[field], dtype=np.float64)
                for field in msc.NOISE_FIELDS
            )
        )
        previous_theta = seen_map_hashes.get(signal_map_sha)
        if previous_theta is not None and previous_theta != entry["theta_sha256"]:
            raise RuntimeError(
                "one summed signal-map triplet is shared by distinct theta values: "
                f"{signal_map_sha}"
            )
        seen_map_hashes[signal_map_sha] = str(entry["theta_sha256"])
        source_map_records.append(
            {
                "point_index": output_point,
                "plan_position": plan_position,
                "entry_index": int(entry["index"]),
                "run_name": str(entry["run_name"]),
                "config_path": str(info["config_path"]),
                "config_sha256": str(info["config_sha256"]),
                **source_metadata,
                "signal_map_arrays_float64_sha256": signal_map_sha,
                "theta_sha256": str(entry["theta_sha256"]),
            }
        )

        for replica in range(args.replicas):
            bank_index = int(selected_full_bank_indices[plan_position, replica])
            seeds = bank_seeds[bank_index]
            vector, means = _measure_noisy_g2x(
                signal_maps,
                seeds,
                ctx,
                g2_alm,
                hp,
                msc,
            )
            vectors[output_row] = vector
            removed_means[output_row] = means
            row_theta[output_row] = theta
            row_u[output_row] = u
            row_log_q[output_row] = log_q
            row_eligible[output_row] = eligible
            row_point[output_row] = output_point
            row_replica[output_row] = replica
            row_plan_position[output_row] = plan_position
            row_entry_index[output_row] = int(entry["index"])
            row_bank_index[output_row] = bank_index
            row_noise_seeds[output_row] = seeds
            output_row += 1

        elapsed = time.time() - started
        rate = output_row / max(elapsed, 1.0e-12)
        eta = (n_rows - output_row) / max(rate, 1.0e-12)
        print(
            f"      point {output_point + 1:4d}/{n_points}: "
            f"{entry['run_name']}  rows {output_row:5d}/{n_rows}  "
            f"eta {eta / 60:.1f} min",
            flush=True,
        )
        del signal_maps

    if output_row != n_rows:
        raise AssertionError(f"measured {output_row} rows, expected {n_rows}")
    if vectors.shape != (n_points * args.replicas, 42):
        raise AssertionError(
            f"final vector shape {vectors.shape} != {(n_rows, 42)}"
        )
    if not np.all(np.isfinite(vectors)):
        bad = np.argwhere(~np.isfinite(vectors))[0].tolist()
        raise RuntimeError(f"non-finite final g2X value at row,column {bad}")
    unique_row_hashes = {msc.sha256_array(row) for row in vectors}
    if len(unique_row_hashes) != n_rows:
        raise RuntimeError(
            f"only {len(unique_row_hashes)} of {n_rows} g2X rows are distinct"
        )

    print("[6/6] writing the g2X training product ...", flush=True)
    output_manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "summary": "masked NaMaster skew spectra of squared galaxy overdensity",
        "product_scope": (
            "g2X training set only; a matching g2X observation, covariance and "
            "compression operator are separate prerequisites for SNLE"
        ),
        "incompatible_existing_runner": (
            "07_mock_sbi_snle/run_mock_sbi_snle.py uses the two-point "
            "gy/gkappa/gtau observation, covariance and score operator"
        ),
        "vector_order": VECTOR_ORDER,
        "spectra": [name for name, _ in G2X_SPECTRA],
        "field_order": [field for _, field in G2X_SPECTRA],
        "n_band": n_band,
        "n_plan_points": len(entries),
        "n_points": n_points,
        "replicas": int(args.replicas),
        "stage5_assignment_replicas": int(args.stage5_replicas),
        "n_rows": n_rows,
        "n_unique_rows": len(unique_row_hashes),
        "parameter_names": list(PARAMETER_NAMES),
        "source_plan": {
            "path": str(plan_path),
            "sha256": msc.sha256_file(plan_path),
            "schema_version": plan["schema_version"],
            "round": plan["round"],
        },
        "missing_maps": missing_maps,
        "exclusion_manifest": exclusion_metadata,
        "accepted_contract_exceptions": accepted_contract_exceptions,
        "partial": bool(missing_maps or excluded_positions),
        "bank": {
            "size": int(args.bank_size),
            "seed_namespace": list(msc.NAMESPACE_TRAINING),
            "assignment_label": str(assignment_label),
            "seed_field_order": list(msc.NOISE_FIELDS),
            "selected_unique_indices": int(np.unique(row_bank_index).size),
            "reuse_factor": float(n_rows / args.bank_size),
            "selection": (
                "columns 0..K-1 of the full Stage-5 assignment at each "
                "full-plan position"
            ),
            "seed_bank": seed_bank_provenance,
            **assignment_provenance,
        },
        "noise_generation": (
            "per field: synalm on frozen integer-ell noise Cl -> alm2map -> "
            "add to float64 pasted signal map -> common-mask weighted centring -> "
            "mask -> map2alm"
        ),
        "noisy_map_storage": "none; maps generated and discarded field by field",
        "estimator": {
            **identity,
            "nside": int(msc.NSIDE),
            "lmax": int(msc.LMAX),
            "band_edges": np.asarray(ctx.band_edges, dtype=np.int64).tolist(),
            "effective_ell": np.asarray(
                ctx.effective_ell, dtype=np.float64
            ).tolist(),
            "coupling": "frozen scalar NaMaster workspace; decouple_cell",
            "pixel_window_correction": "none in measured simulator summary",
        },
        "galaxy": galaxy_metadata,
        "source_maps": source_map_records,
        "path_remaps": [f"{old}={new}" for old, new in remaps],
        "output_vector_sha256": msc.sha256_array(vectors),
        "elapsed_seconds": time.time() - started,
    }

    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            x=vectors,
            theta=row_theta,
            u=row_u,
            log_q=row_log_q,
            importance_eligible=row_eligible,
            point_index=row_point,
            replica=row_replica,
            plan_position=row_plan_position,
            plan_entry_index=row_entry_index,
            noise_seeds=row_noise_seeds,
            bank_indices=row_bank_index,
            removed_field_means=removed_means,
            point_bank_indices=point_bank_indices,
            point_theta=point_theta,
            point_u=point_u,
            point_log_q=point_log_q,
            point_importance_eligible=point_eligible,
            band_edges=np.asarray(ctx.band_edges, dtype=np.int64),
            effective_ell=np.asarray(ctx.effective_ell, dtype=np.float64),
            manifest_json=json.dumps(output_manifest, sort_keys=True),
        )
    os.replace(temporary, output_path)
    _atomic_write_text(
        sidecar_path,
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
    )

    print(f"\nwrote {output_path}")
    print(
        f"      shape {vectors.shape}; vector sha256 "
        f"{output_manifest['output_vector_sha256']}"
    )
    print(f"      manifest {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

