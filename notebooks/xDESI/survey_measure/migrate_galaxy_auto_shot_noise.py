#!/usr/bin/env python3
"""Losslessly migrate a signal-only v2 measurement to the `_gshot` mean convention.

The source file is never modified. The destination keeps an exact copy of each
old galaxy-auto signal dataset before replacing the public mean with signal plus
the saved estimator-matched shot-noise bandpower template. Covariance datasets
and all non-galaxy-auto spectra are copied without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from multiprobe_namaster import (
    DESI_GALAXY_AUTO_MEAN_CONVENTION,
    DESI_GALAXY_AUTO_THEORY_INTERFACE_NOTE,
    SCHEMA_MEASUREMENT,
    covariance_input_noise_policy,
    theory_to_data_vector,
)


LEGACY_SIGNAL_ONLY_CONVENTION = "shot_noise_subtracted_signal"
MIGRATION_RULE = (
    "galaxy_auto_total = shot_noise_subtracted_signal + "
    "noise_decoupled_all_components[component]"
)
SHOT_NOISE_PLOTTING_NOTE = (
    "The saved spectra/<name>/cl values already contain clustering signal plus "
    "weighted Poisson shot noise exactly once. The archived "
    "cl_shot_noise_subtracted_signal dataset preserves the source mean exactly; "
    "do not add noise_decoupled_all_components[component] to the measurement again."
)
JOINT_DATA_VECTOR_CONVENTION = (
    "joint/data_vector is assembled from spectra/<name>/cl. DESI galaxy auto entries "
    "contain clustering signal plus weighted Poisson shot noise exactly once. Theory "
    "windows the clustering signal and then adds an amplitude times the saved, "
    "already-decoupled shot-noise bandpower template."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(dataset: h5py.Dataset) -> List[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[...]
    ]


def default_destination(source: str | Path) -> Path:
    source = Path(source)
    suffix = "_pipev2.h5"
    if not source.name.endswith(suffix):
        raise ValueError(
            f"Cannot infer `_gshot` destination from {source.name!r}; pass --destination."
        )
    return source.with_name(source.name[: -len(suffix)] + "_pipev2_gshot.h5")


def _dataset_paths(h5: h5py.File) -> List[str]:
    paths: List[str] = []

    def collect(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    h5.visititems(collect)
    return sorted(paths)


def _assert_attrs_equal(
    old_obj: object,
    new_obj: object,
    path: str,
    *,
    allowed_changes: tuple[str, ...] = (),
) -> None:
    old_attrs = old_obj.attrs
    new_attrs = new_obj.attrs
    allowed = set(allowed_changes)
    old_keys = set(old_attrs.keys()) - allowed
    new_keys = set(new_attrs.keys()) - allowed
    if old_keys != new_keys:
        raise AssertionError(f"Protected attributes changed at {path}")
    for key in old_keys:
        if not np.array_equal(np.asarray(old_attrs[key]), np.asarray(new_attrs[key])):
            raise AssertionError(f"Protected attribute changed at {path}:{key}")


def _object_paths(h5: h5py.File) -> List[str]:
    paths = [""]
    h5.visititems(lambda name, _obj: paths.append(name))
    return sorted(paths)


def _is_galaxy_auto_covariance_input(dataset: h5py.Dataset) -> bool:
    return (
        str(dataset.attrs.get("kind_a", "")) == "desi_galaxy"
        and str(dataset.attrs.get("kind_b", "")) == "desi_galaxy"
        and str(dataset.attrs.get("field_a", ""))
        == str(dataset.attrs.get("field_b", ""))
    )


def _expected_galaxy_auto_covariance_policy(
    dataset: h5py.Dataset,
    input_mode: str,
) -> str:
    field_a = str(dataset.attrs["field_a"])
    field_b = str(dataset.attrs["field_b"])
    field_metadata = {
        field_a: {"kind": str(dataset.attrs["kind_a"])},
        field_b: {"kind": str(dataset.attrs["kind_b"])},
    }
    return covariance_input_noise_policy(
        field_a,
        field_b,
        field_metadata,
        input_mode=input_mode,
    )


def _refresh_galaxy_auto_covariance_provenance_in_h5(h5: h5py.File) -> List[str]:
    """Update convention text only; the covariance inputs themselves stay byte-identical."""

    group = h5["input_cls_for_covariance"]
    input_mode = str(group.attrs.get("mode", "inka_data"))
    updated: List[str] = []
    for name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset) or not _is_galaxy_auto_covariance_input(dataset):
            continue
        dataset.attrs["noise_policy"] = _expected_galaxy_auto_covariance_policy(
            dataset,
            input_mode,
        )
        updated.append(f"input_cls_for_covariance/{name}")
    return sorted(updated)


def _resolve_legacy_config_defaults_in_h5(h5: h5py.File) -> Dict[str, object]:
    """Materialize only execution defaults required by the current identity check."""

    resolved: Dict[str, object] = {}
    if "config_json" in h5.attrs:
        config = json.loads(str(h5.attrs["config_json"]))
        if "lmax_mask" not in config:
            if "lmax" not in config:
                raise ValueError("Legacy config has neither lmax_mask nor lmax.")
            config["lmax_mask"] = int(config["lmax"])
            resolved["lmax_mask"] = int(config["lmax"])
        if resolved:
            h5.attrs["config_json"] = json.dumps(config, indent=2, sort_keys=True)
            h5.attrs["galaxy_auto_mean_migration_resolved_config_defaults_json"] = json.dumps(
                resolved, sort_keys=True
            )
    if "theory_interface" in h5:
        theory_interface = h5["theory_interface"]
        theory_interface.attrs[
            "desi_galaxy_auto_mean_convention"
        ] = DESI_GALAXY_AUTO_MEAN_CONVENTION
        theory_interface.attrs[
            "desi_galaxy_auto_shot_noise_nuisance"
        ] = DESI_GALAXY_AUTO_THEORY_INTERFACE_NOTE
    _refresh_galaxy_auto_covariance_provenance_in_h5(h5)
    return resolved


def resolve_legacy_config_defaults(path: str | Path) -> Dict[str, object]:
    """Resolve compatible metadata defaults in an existing migrated copy only."""

    path = Path(path).resolve()
    with h5py.File(path, "r+") as h5:
        if str(h5.attrs.get("desi_galaxy_auto_mean_convention", "")) != (
            DESI_GALAXY_AUTO_MEAN_CONVENTION
        ):
            raise ValueError(f"Refusing to modify a non-`_gshot` product: {path}")
        resolved = _resolve_legacy_config_defaults_in_h5(h5)
        h5.flush()
    return resolved


def audit_migration(source: str | Path, destination: str | Path) -> Dict[str, object]:
    """Assert the exact allowed delta between a legacy source and `_gshot` copy."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    source_sha256 = _sha256(source)
    with h5py.File(source, "r") as old, h5py.File(destination, "r") as new:
        if str(new.attrs.get("desi_galaxy_auto_mean_convention", "")) != (
            DESI_GALAXY_AUTO_MEAN_CONVENTION
        ):
            raise AssertionError("Destination root galaxy-auto convention is missing or wrong.")
        if str(new.attrs.get("galaxy_auto_mean_migration_source", "")) != str(source):
            raise AssertionError("Destination migration source path is wrong.")
        if str(new.attrs.get("galaxy_auto_mean_migration_source_sha256", "")) != source_sha256:
            raise AssertionError("Destination migration source hash is wrong.")
        if str(new.attrs.get("galaxy_auto_mean_migration_rule", "")) != MIGRATION_RULE:
            raise AssertionError("Destination migration rule is wrong.")
        if "theory_interface" in old:
            theory_interface = new["theory_interface"]
            if str(theory_interface.attrs.get("desi_galaxy_auto_mean_convention", "")) != (
                DESI_GALAXY_AUTO_MEAN_CONVENTION
            ):
                raise AssertionError("Theory-interface galaxy-auto convention is wrong.")
            if str(
                theory_interface.attrs.get("desi_galaxy_auto_shot_noise_nuisance", "")
            ) != DESI_GALAXY_AUTO_THEORY_INTERFACE_NOTE:
                raise AssertionError("Theory-interface shot-noise nuisance contract is wrong.")
        resolved_config_defaults: Dict[str, object] = {}
        if "config_json" in old.attrs:
            old_config = json.loads(str(old.attrs["config_json"]))
            new_config = json.loads(str(new.attrs["config_json"]))
            for key, value in old_config.items():
                if new_config.get(key) != value:
                    raise AssertionError(f"Existing config value changed during migration: {key}")
            if "lmax_mask" not in old_config:
                expected = int(old_config["lmax"])
                if int(new_config.get("lmax_mask", -1)) != expected:
                    raise AssertionError("Implicit legacy lmax_mask was not materialized correctly.")
                resolved_config_defaults["lmax_mask"] = expected

        names = _strings(old["joint/spectrum_names"])
        starts = np.asarray(old["joint/slice_start"][:], dtype=np.int64)
        stops = np.asarray(old["joint/slice_stop"][:], dtype=np.int64)
        galaxy_names = [
            name
            for name in names
            if str(old[f"spectra/{name}"].attrs.get("family", "")) == "desi_g_auto"
        ]
        galaxy_auto_covariance_paths = set()
        covariance_input_group = new["input_cls_for_covariance"]
        covariance_input_mode = str(covariance_input_group.attrs.get("mode", ""))
        for dataset_name, dataset in covariance_input_group.items():
            if not isinstance(dataset, h5py.Dataset) or not _is_galaxy_auto_covariance_input(dataset):
                continue
            path = f"input_cls_for_covariance/{dataset_name}"
            expected_policy = _expected_galaxy_auto_covariance_policy(
                dataset,
                covariance_input_mode,
            )
            if str(dataset.attrs.get("noise_policy", "")) != expected_policy:
                raise AssertionError(f"Stale galaxy-auto covariance provenance at {path}")
            galaxy_auto_covariance_paths.add(path)
        allowed_changes = {"joint/data_vector"}
        expected_extras = set()
        for name in galaxy_names:
            allowed_changes.update(
                {f"spectra/{name}/cl", f"spectra/{name}/cl_all_components"}
            )
            expected_extras.update(
                {
                    f"spectra/{name}/cl_shot_noise_subtracted_signal",
                    f"spectra/{name}/cl_all_components_shot_noise_subtracted_signal",
                }
            )

        old_paths = set(_dataset_paths(old))
        new_paths = set(_dataset_paths(new))
        if new_paths - old_paths != expected_extras:
            raise AssertionError(
                "Destination has unexpected added/omitted datasets: "
                f"added={sorted(new_paths - old_paths)}, omitted={sorted(old_paths - new_paths)}"
            )
        exact_unchanged = 0
        covariance_datasets = 0
        covariance_input_mode = str(old["input_cls_for_covariance"].attrs.get("mode", ""))
        covariance_estimator_version = str(
            old.attrs.get("covariance_estimator_version", "")
        )
        if covariance_input_mode != str(
            new["input_cls_for_covariance"].attrs.get("mode", "")
        ):
            raise AssertionError("Covariance input mode changed during migration.")
        if covariance_estimator_version != str(
            new.attrs.get("covariance_estimator_version", "")
        ):
            raise AssertionError("Covariance estimator version changed during migration.")
        for path in sorted(old_paths - allowed_changes):
            if not np.array_equal(old[path][...], new[path][...]):
                raise AssertionError(f"Protected dataset changed: {path}")
            exact_unchanged += 1
            if (
                path.startswith("covariance_blocks/")
                or path.startswith("input_cls_for_covariance/")
                or path in {"joint/cov", "joint/corr"}
                or path.endswith("/cov")
                or path.endswith("/err")
            ):
                covariance_datasets += 1

        attribute_objects_checked = 0
        for path in _object_paths(old):
            if path and path not in new:
                raise AssertionError(f"Protected HDF5 object omitted: {path}")
            allowed_attribute_changes: tuple[str, ...] = ()
            if path == "":
                allowed_attribute_changes = (
                    "config_json",
                    "desi_galaxy_auto_mean_convention",
                    "galaxy_auto_mean_migration_source",
                    "galaxy_auto_mean_migration_source_sha256",
                    "galaxy_auto_mean_migration_utc",
                    "galaxy_auto_mean_migration_rule",
                    "galaxy_auto_mean_migration_resolved_config_defaults_json",
                )
            elif path == "joint":
                allowed_attribute_changes = ("data_vector_convention",)
            elif path == "theory_interface":
                allowed_attribute_changes = (
                    "desi_galaxy_auto_mean_convention",
                    "desi_galaxy_auto_shot_noise_nuisance",
                )
            elif path in {f"spectra/{name}" for name in galaxy_names}:
                allowed_attribute_changes = ("cl_convention", "shot_noise_plotting_note")
            elif path in galaxy_auto_covariance_paths:
                allowed_attribute_changes = ("noise_policy",)
            _assert_attrs_equal(
                old[path] if path else old,
                new[path] if path else new,
                path or "/",
                allowed_changes=allowed_attribute_changes,
            )
            attribute_objects_checked += 1

        expected_vector = np.asarray(old["joint/data_vector"][:], dtype=np.float64)
        changed_elements = 0
        for name in galaxy_names:
            index = names.index(name)
            start, stop = int(starts[index]), int(stops[index])
            old_group = old[f"spectra/{name}"]
            new_group = new[f"spectra/{name}"]
            component = int(old_group.attrs["component"])
            old_cl = np.asarray(old_group["cl"][:], dtype=np.float64)
            old_all = np.asarray(old_group["cl_all_components"][:], dtype=np.float64)
            noise_all = np.asarray(
                old_group["noise_decoupled_all_components"][:], dtype=np.float64
            )
            expected_cl = old_cl + noise_all[component]
            if not np.array_equal(new_group["cl"][:], expected_cl):
                raise AssertionError(f"Wrong migrated selected mean: {name}")
            if not np.array_equal(new_group["cl_all_components"][:], old_all + noise_all):
                raise AssertionError(f"Wrong migrated component means: {name}")
            if not np.array_equal(
                new_group["cl_shot_noise_subtracted_signal"][:], old_cl
            ):
                raise AssertionError(f"Signal-only archive is not exact: {name}")
            if not np.array_equal(
                new_group["cl_all_components_shot_noise_subtracted_signal"][:], old_all
            ):
                raise AssertionError(f"Component archive is not exact: {name}")
            if str(new_group.attrs.get("cl_convention", "")) != (
                DESI_GALAXY_AUTO_MEAN_CONVENTION
            ):
                raise AssertionError(f"Wrong migrated spectrum convention: {name}")
            if str(new_group.attrs.get("shot_noise_plotting_note", "")) != (
                SHOT_NOISE_PLOTTING_NOTE
            ):
                raise AssertionError(f"Wrong migrated plotting convention: {name}")
            expected_vector[start:stop] = expected_cl
            changed_elements += stop - start

        if not np.array_equal(new["joint/data_vector"][:], expected_vector):
            raise AssertionError("Destination joint vector is not the exact expected vector.")
        if str(new["joint"].attrs.get("data_vector_convention", "")) != (
            JOINT_DATA_VECTOR_CONVENTION
        ):
            raise AssertionError("Destination joint-vector convention is missing or wrong.")

    return {
        "source": str(source),
        "source_sha256": source_sha256,
        "destination": str(destination),
        "destination_sha256": _sha256(destination),
        "galaxy_auto_spectra": galaxy_names,
        "changed_data_vector_elements": changed_elements,
        "exactly_unchanged_source_datasets": exact_unchanged,
        "exactly_unchanged_covariance_related_datasets": covariance_datasets,
        "protected_attribute_objects_checked": attribute_objects_checked,
        "updated_galaxy_auto_covariance_provenance": sorted(galaxy_auto_covariance_paths),
        "covariance_input_mode": covariance_input_mode,
        "covariance_estimator_version": covariance_estimator_version,
        "resolved_config_defaults": resolved_config_defaults,
        "status": "PASS",
    }


def migrate_product(source: str | Path, destination: str | Path) -> Dict[str, object]:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("Source and destination must be different; the source is immutable.")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256(source)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary path already exists: {temporary}")

    changed_spectra: List[str] = []
    changed_elements = 0
    try:
        shutil.copy2(source, temporary)
        with h5py.File(temporary, "r+") as h5:
            if str(h5.attrs.get("schema", "")) != SCHEMA_MEASUREMENT:
                raise ValueError(f"Unexpected measurement schema in {source}")
            root_convention = str(h5.attrs.get("desi_galaxy_auto_mean_convention", ""))
            if root_convention not in {"", LEGACY_SIGNAL_ONLY_CONVENTION}:
                raise ValueError(
                    "Source is not a legacy signal-only product: "
                    f"desi_galaxy_auto_mean_convention={root_convention!r}"
                )

            joint = h5["joint"]
            names = _strings(joint["spectrum_names"])
            starts = np.asarray(joint["slice_start"][:], dtype=np.int64)
            stops = np.asarray(joint["slice_stop"][:], dtype=np.int64)
            if not (len(names) == starts.size == stops.size):
                raise ValueError("Joint spectrum names/slices have inconsistent lengths.")
            data_vector = np.asarray(joint["data_vector"][:], dtype=np.float64)

            for index, name in enumerate(names):
                group = h5[f"spectra/{name}"]
                if str(group.attrs.get("family", "")) != "desi_g_auto":
                    continue
                convention = str(group.attrs.get("cl_convention", ""))
                if convention not in {"", LEGACY_SIGNAL_ONLY_CONVENTION}:
                    raise ValueError(
                        f"Galaxy auto {name!r} is not signal-only: cl_convention={convention!r}"
                    )
                if "noise_decoupled_all_components" not in group:
                    raise ValueError(f"Galaxy auto {name!r} has no saved shot-noise template.")
                if "cl_shot_noise_subtracted_signal" in group:
                    raise ValueError(f"Galaxy auto {name!r} already has a migration archive.")

                component = int(group.attrs["component"])
                old_cl = np.asarray(group["cl"][:], dtype=np.float64)
                old_all = np.asarray(group["cl_all_components"][:], dtype=np.float64)
                noise_all = np.asarray(
                    group["noise_decoupled_all_components"][:], dtype=np.float64
                )
                if old_all.shape != noise_all.shape:
                    raise ValueError(
                        f"Galaxy auto {name!r} component/noise shapes differ: "
                        f"{old_all.shape} versus {noise_all.shape}."
                    )
                if not (0 <= component < noise_all.shape[0]):
                    raise ValueError(
                        f"Galaxy auto {name!r} component {component} is outside {noise_all.shape}."
                    )
                template = noise_all[component]
                if old_cl.shape != template.shape:
                    raise ValueError(
                        f"Galaxy auto {name!r} selected/noise shapes differ: "
                        f"{old_cl.shape} versus {template.shape}."
                    )
                start, stop = int(starts[index]), int(stops[index])
                if stop - start != old_cl.size:
                    raise ValueError(
                        f"Galaxy auto {name!r} joint slice {start}:{stop} does not match "
                        f"its {old_cl.size} bandpowers."
                    )
                if not np.array_equal(data_vector[start:stop], old_cl):
                    raise ValueError(
                        f"Galaxy auto {name!r} joint slice is not identical to spectra/{name}/cl."
                    )

                group.create_dataset("cl_shot_noise_subtracted_signal", data=old_cl)
                group.create_dataset(
                    "cl_all_components_shot_noise_subtracted_signal", data=old_all
                )
                new_cl = old_cl + template
                group["cl"][...] = new_cl
                group["cl_all_components"][...] = old_all + noise_all
                data_vector[start:stop] = new_cl
                group.attrs["cl_convention"] = DESI_GALAXY_AUTO_MEAN_CONVENTION
                group.attrs["shot_noise_plotting_note"] = SHOT_NOISE_PLOTTING_NOTE
                changed_spectra.append(name)
                changed_elements += old_cl.size

            if not changed_spectra:
                raise ValueError("Source contains no DESI galaxy-auto spectra to migrate.")
            joint["data_vector"][...] = data_vector
            joint.attrs["data_vector_convention"] = JOINT_DATA_VECTOR_CONVENTION
            h5.attrs[
                "desi_galaxy_auto_mean_convention"
            ] = DESI_GALAXY_AUTO_MEAN_CONVENTION
            h5.attrs["galaxy_auto_mean_migration_source"] = str(source)
            h5.attrs["galaxy_auto_mean_migration_source_sha256"] = source_sha256
            h5.attrs["galaxy_auto_mean_migration_utc"] = datetime.now(timezone.utc).isoformat()
            h5.attrs["galaxy_auto_mean_migration_rule"] = MIGRATION_RULE
            resolved_config_defaults = _resolve_legacy_config_defaults_in_h5(h5)
            h5.flush()

        audit = audit_migration(source, temporary)
        if destination.exists():
            raise FileExistsError(f"Destination appeared during migration: {destination}")
        os.replace(temporary, destination)
        audit["destination"] = str(destination)
        audit["destination_sha256"] = _sha256(destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "source": str(source),
        "source_sha256": source_sha256,
        "destination": str(destination),
        "destination_sha256": _sha256(destination),
        "changed_spectra": changed_spectra,
        "changed_data_vector_elements": changed_elements,
        "mean_convention": DESI_GALAXY_AUTO_MEAN_CONVENTION,
        "covariance_policy": (
            "numerical covariance and iNKA inputs copied unchanged; galaxy-auto noise-policy "
            "text updated because raw-map iNKA inputs and saved means both contain shot noise"
        ),
        "resolved_config_defaults": resolved_config_defaults,
        "audit": audit,
    }


def audit_zero_signal_theory_wrapper(measurement_path: str | Path) -> Dict[str, object]:
    """Verify the saved shot template is the wrapper's only zero-signal output."""

    measurement_path = Path(measurement_path).resolve()
    with h5py.File(measurement_path, "r") as h5:
        config = json.loads(str(h5.attrs["config_json"]))
        lmax = int(config["lmax"])
        names = _strings(h5["joint/spectrum_names"])
        zeros = {name: np.zeros(lmax + 1, dtype=np.float64) for name in names}
        expected_chunks: List[np.ndarray] = []
        galaxy_elements = 0
        for name in names:
            group = h5[f"spectra/{name}"]
            n_band = int(group["cl"].shape[0])
            expected = np.zeros(n_band, dtype=np.float64)
            if str(group.attrs.get("family", "")) == "desi_g_auto":
                component = int(group.attrs["component"])
                expected = np.asarray(
                    group["noise_decoupled_all_components"][component],
                    dtype=np.float64,
                )
                galaxy_elements += n_band
            expected_chunks.append(expected)
        expected_unit = np.concatenate(expected_chunks)

    zero, zero_names = theory_to_data_vector(
        measurement_path,
        zeros,
        desi_galaxy_shot_noise_amplitudes=0.0,
    )
    unit, unit_names = theory_to_data_vector(
        measurement_path,
        zeros,
        desi_galaxy_shot_noise_amplitudes=1.0,
    )
    double, double_names = theory_to_data_vector(
        measurement_path,
        zeros,
        desi_galaxy_shot_noise_amplitudes=2.0,
    )
    if zero_names != names or unit_names != names or double_names != names:
        raise AssertionError("Theory wrapper changed the saved spectrum order.")
    if not np.array_equal(zero, np.zeros_like(expected_unit)):
        raise AssertionError("A_shot=0 did not remove the saved shot templates exactly.")
    if not np.array_equal(unit, expected_unit):
        raise AssertionError("A_shot=1 did not reproduce the saved shot templates exactly.")
    if not np.array_equal(double, 2.0 * expected_unit):
        raise AssertionError("A_shot=2 did not reproduce twice the saved templates exactly.")
    nonzero = int(np.count_nonzero(unit))
    if nonzero != galaxy_elements:
        raise AssertionError(
            f"Zero-signal wrapper has {nonzero} nonzero elements, expected {galaxy_elements}."
        )
    return {
        "measurement": str(measurement_path),
        "measurement_sha256": _sha256(measurement_path),
        "n_spectra": len(names),
        "galaxy_auto_nonzero_elements": nonzero,
        "zero_max_abs_err": float(np.max(np.abs(zero))),
        "unit_max_abs_err": float(np.max(np.abs(unit - expected_unit))),
        "double_max_abs_err": float(np.max(np.abs(double - 2.0 * expected_unit))),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="Audit an existing destination instead of creating it.",
    )
    parser.add_argument(
        "--resolve-legacy-config-defaults",
        action="store_true",
        help="Materialize compatible implicit config defaults in an existing migrated copy.",
    )
    parser.add_argument(
        "--audit-theory-wrapper",
        action="store_true",
        help="Check zero-signal A_shot=1/2 outputs on the existing destination.",
    )
    args = parser.parse_args()
    destination = args.destination or default_destination(args.source)
    modes = sum(
        bool(value)
        for value in (
            args.audit_existing,
            args.resolve_legacy_config_defaults,
            args.audit_theory_wrapper,
        )
    )
    if modes > 1:
        parser.error(
            "Choose only one of --audit-existing, --resolve-legacy-config-defaults, "
            "and --audit-theory-wrapper."
        )
    if args.audit_theory_wrapper:
        report = audit_zero_signal_theory_wrapper(destination)
    elif args.audit_existing:
        report = audit_migration(args.source, destination)
    elif args.resolve_legacy_config_defaults:
        resolved = resolve_legacy_config_defaults(destination)
        report = audit_migration(args.source, destination)
        report["resolved_config_defaults_now"] = resolved
    else:
        report = migrate_product(args.source, destination)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
