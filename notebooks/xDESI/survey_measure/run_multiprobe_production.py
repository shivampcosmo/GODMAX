#!/usr/bin/env python
"""Split production driver for xDESI multi-probe NaMaster measurements."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
import pymaster as nmt

from multiprobe_namaster import (
    COVARIANCE_ESTIMATOR_VERSION,
    DESI_GALAXY_AUTO_MEAN_CONVENTION,
    DESI_GALAXY_AUTO_PRIMARY_VIEW,
    DESI_GALAXY_AUTO_SUBTRACTED_VIEW,
    DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION,
    MAP_CONSTRUCTION_VERSION,
    MEASUREMENT_PIPELINE_VERSION,
    SCHEMA_MAPS,
    SCHEMA_MEASUREMENT,
    SCHEMA_MEASUREMENT_VALIDITY_MASK,
    SPECTRUM_ESTIMATOR_VERSION,
    SurveyBundle,
    MeasurementConfig,
    SpectrumSpec,
    _corr_from_cov,
    _covariance_workspace_from_fields,
    _json_dumps,
    _string_array,
    _write_dataset,
    add_common_cli_args,
    build_nmt_fields,
    build_probe_maps,
    compute_covariance_block_with_workspace,
    config_from_args,
    covariance_diagnostics,
    covariance_group_key_for_specs,
    covariance_input_noise_policy,
    default_spectrum_specs,
    load_map_product,
    make_bandpower_edges,
    make_bins,
    measurement_schema_for_config,
    pack_joint_data_vector,
    validate_map_metadata_identity,
    validate_galaxy_auto_views,
    validate_measurement_product_identity,
    save_map_product,
    save_measurement_product,
    utc_now,
)


def _config_from_map_metadata(config: MeasurementConfig, map_metadata: Mapping[str, object]) -> MeasurementConfig:
    _required_map_product_id(map_metadata)
    map_config = map_metadata.get("config", {})
    if not isinstance(map_config, Mapping):
        raise ValueError("Map product metadata has no valid config mapping.")
    required_keys = (
        "pipeline_version",
        "stage",
        "nside",
        "lmax_mask",
        "act_downgrade",
        "shear_e_to_kappa_sign",
        "shear_mask_dataset",
        "shear_noise_attr",
        "subtract_masked_mean",
        "mask_apodization_deg",
        "mask_apodization_type",
        "pair_overlap_mean_subtract",
    )
    if str(config.stage) == "highres4096":
        required_keys += (
            "kappa_cmb_lmax",
            "act_cmb_temperature_units_confirmed",
            "minimum_desi_random_realizations",
        )
    for key in required_keys:
        if key not in map_config:
            raise ValueError(f"Map product is missing construction config key {key!r}; regenerate it.")
        expected = _resolved_config_value(config, key)
        actual = map_config[key]
        if key == "lmax_mask" and actual is None:
            actual = map_config.get("lmax")
        if not _config_value_matches(actual, expected):
            raise ValueError(
                f"Requested {key}={expected!r} does not match cached map value "
                f"{map_config[key]!r}. Use the matching stage/options or regenerate maps."
            )
    if "lmax" not in map_config:
        raise ValueError("Map product is missing construction config key 'lmax'; regenerate it.")
    if int(config.lmax) > int(map_config["lmax"]):
        raise ValueError(f"Requested lmax={config.lmax} exceeds cached-map lmax={map_config['lmax']}.")
    config.validate()
    return config


def spectra_path(config: MeasurementConfig) -> Path:
    return config.output_root / f"xdesi_multiprobe_spectra_{config.product_tag}.h5"


def manifest_path(config: MeasurementConfig) -> Path:
    return config.output_root / f"covariance_manifest_{config.covariance_product_tag}.json"


def covariance_work_plan_path(config: MeasurementConfig) -> Path:
    return config.output_root / f"covariance_work_plan_{config.covariance_product_tag}.json"


def block_dir(config: MeasurementConfig) -> Path:
    return config.output_root / f"covariance_blocks_{config.covariance_product_tag}"


def block_shard_path(config: MeasurementConfig, group: Mapping[str, object]) -> Path:
    return block_dir(config) / f"cov_group_{int(group['index']):04d}_{str(group['class'])}.h5"


def cov_workspace_cache_dir(config: MeasurementConfig) -> Path:
    return block_dir(config) / "cov_workspaces"


COVARIANCE_CONFIG_KEYS = (
    "pipeline_version",
    "stage",
    "nside",
    "lmax",
    "lmax_mask",
    "ell_min",
    "n_bins",
    "binning",
    "act_downgrade",
    "shear_mask_dataset",
    "shear_noise_attr",
    "shear_e_to_kappa_sign",
    "subtract_masked_mean",
    "mask_apodization_deg",
    "mask_apodization_type",
    "pair_overlap_mean_subtract",
    "n_iter",
    "n_iter_mask",
    "covariance_l_toeplitz",
    "covariance_l_exact",
    "covariance_dl_band",
    "covariance_input_mode",
    "covariance_input_smooth_bandpowers",
    "covariance_input_smooth_window",
    "covariance_zero_parity_odd_inputs",
)

SPECTRUM_CONFIG_KEYS = (
    "pipeline_version",
    "stage",
    "nside",
    "lmax",
    "lmax_mask",
    "ell_min",
    "n_bins",
    "binning",
    "kappa_cmb_lmax",
    "act_downgrade",
    "shear_mask_dataset",
    "shear_noise_attr",
    "shear_e_to_kappa_sign",
    "subtract_masked_mean",
    "mask_apodization_deg",
    "mask_apodization_type",
    "pair_overlap_mean_subtract",
    "n_iter",
    "n_iter_mask",
    "include_ksz_velocity_shuffle",
    "ksz_shuffle_seed",
)


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def covariance_config_payload(config: MeasurementConfig | Mapping[str, object]) -> Dict[str, object]:
    if isinstance(config, Mapping):
        source = config
        payload = {key: source.get(key) for key in COVARIANCE_CONFIG_KEYS}
        if payload.get("lmax_mask") is None:
            payload["lmax_mask"] = source.get("lmax")
        return payload
    return {key: _resolved_config_value(config, key) for key in COVARIANCE_CONFIG_KEYS}


def covariance_config_digest(config: MeasurementConfig | Mapping[str, object]) -> str:
    return _sha256_json(covariance_config_payload(config))


def _group_digest(group: Mapping[str, object]) -> str:
    return _sha256_json(group)


def _required_map_product_id(map_metadata: Mapping[str, object]) -> str:
    pipeline = str(map_metadata.get("pipeline_version", ""))
    construction = str(map_metadata.get("map_construction_version", ""))
    if pipeline != MEASUREMENT_PIPELINE_VERSION:
        raise ValueError(
            f"Map product pipeline_version={pipeline!r}; expected {MEASUREMENT_PIPELINE_VERSION!r}. "
            "Regenerate maps with the current estimator."
        )
    if construction != MAP_CONSTRUCTION_VERSION:
        raise ValueError(
            f"Map construction version={construction!r}; expected {MAP_CONSTRUCTION_VERSION!r}. "
            "Regenerate maps with the current estimator."
        )
    return validate_map_metadata_identity(map_metadata)


def _array_digest(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def _group_mask_digest_from_fields(
    group: Mapping[str, object],
    fields: Mapping[str, object],
) -> str:
    payload = [
        {
            "field": str(name),
            "mask_digest": _array_digest(fields[str(name)].mask),
        }
        for name in group.get("representative_fields", [])
    ]
    return _sha256_json(payload)


def _group_mask_digest_from_metadata(
    group: Mapping[str, object],
    map_metadata: Mapping[str, object],
) -> str:
    content = map_metadata.get("map_content_digests", {})
    if not isinstance(content, Mapping):
        raise ValueError("Map metadata has no content digests for covariance assembly.")
    field_mask_names = content.get("field_mask_names", {})
    masks = content.get("masks", {})
    if not isinstance(field_mask_names, Mapping) or not isinstance(masks, Mapping):
        raise ValueError("Map metadata mask-content identities are incomplete.")
    payload = []
    for name_raw in group.get("representative_fields", []):
        name = str(name_raw)
        if name not in field_mask_names:
            raise ValueError(f"Map metadata has no mask identity for field {name!r}.")
        mask_name = str(field_mask_names[name])
        if mask_name not in masks:
            raise ValueError(f"Map metadata has no content digest for mask {mask_name!r}.")
        payload.append({"field": name, "mask_digest": str(masks[mask_name])})
    return _sha256_json(payload)


def _field_names_for_groups(groups: Iterable[Mapping[str, object]]) -> Set[str]:
    """Return only the field names a set of covariance groups actually reference.

    A group's covariance only needs the fields that appear in its blocks (plus the
    representative fields used for the covariance workspace). Building just these instead
    of all ~15 probe fields cuts per-process memory by ~3-5x (the spin-2 alms dominate),
    which lets many more single-threaded groups pack onto one node.
    """

    names: Set[str] = set()
    for group in groups:
        names.update(str(n) for n in group.get("representative_fields", []))
        for block in group.get("blocks", []):
            names.update(str(n) for n in block.get("fields_i", []))
            names.update(str(n) for n in block.get("fields_j", []))
    return names


def _build_cov_fields(
    map_fields: Mapping[str, object],
    config: MeasurementConfig,
    groups: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    """Build NaMaster fields for only the probes referenced by ``groups``."""

    needed = _field_names_for_groups(groups)
    subset = {name: fmap for name, fmap in map_fields.items() if name in needed}
    missing = sorted(needed - set(subset))
    if missing:
        raise KeyError(f"Covariance group references field(s) absent from the map product: {missing}")
    return build_nmt_fields(subset, config)


def _cov_workspace_cache_path(
    config: MeasurementConfig,
    group: Mapping[str, object],
    fields: Mapping[str, object],
) -> Path:
    """Path of the on-disk covariance workspace for a group's mask/spin signature.

    The covariance workspace depends only on the four masks (the alias key), the spins,
    and lmax/Toeplitz settings -- never on the field data or the noise model. The actual
    mask bytes are part of the signature so an estimator change cannot accidentally reuse
    a workspace built from a stale mask under the same human-readable tag.
    """

    signature = {
        "key": [str(k) for k in group.get("key", [])],
        "spins": [int(s) for s in group.get("spins", [])],
        "lmax": int(config.lmax),
        "lmax_mask": int(config.effective_lmax_mask),
        "l_toeplitz": int(config.covariance_l_toeplitz),
        "l_exact": int(config.covariance_l_exact),
        "dl_band": int(config.covariance_dl_band),
        "pipeline_version": MEASUREMENT_PIPELINE_VERSION,
        "covariance_estimator_version": COVARIANCE_ESTIMATOR_VERSION,
        "mask_digests": [
            _array_digest(fields[name].mask) for name in group.get("representative_fields", [])
        ],
    }
    digest = _sha256_json(signature)[:20]
    return cov_workspace_cache_dir(config) / f"cw_{digest}.fits"


def _get_or_build_cov_workspace(
    group: Mapping[str, object],
    fields: Mapping[str, object],
    config: MeasurementConfig,
    *,
    use_cache: bool = True,
) -> object:
    """Load the group's covariance workspace from disk if cached, else build and cache it."""

    representatives = list(group["representative_fields"])
    path = _cov_workspace_cache_path(config, group, fields)
    if use_cache and path.exists():
        try:
            cw = nmt.NmtCovarianceWorkspace.from_file(str(path))
            print(f"[{utc_now()}] group {group['index']} reused cached covariance workspace {path.name}", flush=True)
            return cw
        except Exception as exc:  # pragma: no cover - corrupt/old cache, rebuild
            print(f"[{utc_now()}] group {group['index']} cached workspace {path.name} unreadable ({exc}); rebuilding", flush=True)
    cw = _covariance_workspace_from_fields(
        fields[representatives[0]].cov_field,
        fields[representatives[1]].cov_field,
        fields[representatives[2]].cov_field,
        fields[representatives[3]].cov_field,
        config,
    )
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".fits.tmp")
            if tmp.exists():
                tmp.unlink()
            cw.write_to(str(tmp))
            os.replace(tmp, path)
        except Exception as exc:  # pragma: no cover - cache write best effort
            print(f"[{utc_now()}] group {group['index']} could not cache workspace ({exc})", flush=True)
    return cw


def _config_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return bool(actual) == expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == int(expected)
        except Exception:
            return False
    if isinstance(expected, float):
        try:
            return float(actual) == float(expected)
        except Exception:
            return False
    return str(actual) == str(expected)


def _resolved_config_value(config: MeasurementConfig, key: str) -> object:
    """Return the value actually executed for a possibly defaulted option."""

    if key == "lmax_mask":
        return int(config.effective_lmax_mask)
    return getattr(config, key)


def _existing_product_matches_config(
    path: Path,
    schema: str,
    config: MeasurementConfig,
    *,
    expected_map_product_id: Optional[str] = None,
) -> Tuple[bool, str]:
    if not path.exists():
        return False, "file does not exist"
    stored_ell_left: Optional[np.ndarray] = None
    stored_ell_right: Optional[np.ndarray] = None
    try:
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("schema") != schema:
                return False, f"schema is {h5.attrs.get('schema')!r}, expected {schema!r}"
            pipeline_version = str(h5.attrs.get("pipeline_version", ""))
            if pipeline_version != MEASUREMENT_PIPELINE_VERSION:
                return False, (
                    f"pipeline_version is {pipeline_version!r}, expected "
                    f"{MEASUREMENT_PIPELINE_VERSION!r}"
                )
            if schema == SCHEMA_MAPS:
                construction_version = str(h5.attrs.get("map_construction_version", ""))
                if construction_version != MAP_CONSTRUCTION_VERSION:
                    return False, (
                        f"map_construction_version is {construction_version!r}, expected "
                        f"{MAP_CONSTRUCTION_VERSION!r}"
                    )
                if not str(h5.attrs.get("map_product_id", "")):
                    return False, "map_product_id is missing"
                metadata = json.loads(h5.attrs["metadata_json"])
                try:
                    metadata_product_id = _required_map_product_id(metadata)
                except ValueError as exc:
                    return False, str(exc)
                if str(h5.attrs.get("map_product_id", "")) != metadata_product_id:
                    return False, "map_product_id attribute does not match content-addressed metadata"
                cfg = metadata.get("config", {})
            else:
                spectrum_version = str(h5.attrs.get("spectrum_estimator_version", ""))
                if spectrum_version != SPECTRUM_ESTIMATOR_VERSION:
                    return False, (
                        f"spectrum_estimator_version is {spectrum_version!r}, expected "
                        f"{SPECTRUM_ESTIMATOR_VERSION!r}"
                    )
                mean_convention = str(h5.attrs.get("desi_galaxy_auto_mean_convention", ""))
                if mean_convention != DESI_GALAXY_AUTO_MEAN_CONVENTION:
                    return False, (
                        "desi_galaxy_auto_mean_convention is "
                        f"{mean_convention!r}, expected {DESI_GALAXY_AUTO_MEAN_CONVENTION!r}"
                    )
                stored_map_product_id = str(h5.attrs.get("map_product_id", ""))
                if not stored_map_product_id:
                    return False, "measurement map_product_id is missing"
                embedded_map_metadata = json.loads(h5.attrs["map_metadata_json"])
                try:
                    embedded_map_product_id = _required_map_product_id(embedded_map_metadata)
                except ValueError as exc:
                    return False, f"embedded map metadata is incompatible: {exc}"
                if stored_map_product_id != embedded_map_product_id:
                    return False, "measurement map_product_id does not match embedded map metadata"
                if expected_map_product_id is not None and stored_map_product_id != str(expected_map_product_id):
                    return False, (
                        f"map_product_id is {stored_map_product_id!r}, expected "
                        f"{str(expected_map_product_id)!r}"
                    )
                cfg = json.loads(h5.attrs["config_json"])
                if "ell_left" not in h5 or "ell_right" not in h5:
                    return False, "measurement bandpower-edge arrays are missing"
                stored_ell_left = np.asarray(h5["ell_left"][:], dtype=np.int64)
                stored_ell_right = np.asarray(h5["ell_right"][:], dtype=np.int64)
    except Exception as exc:
        return False, f"could not read product metadata: {exc}"
    config_keys = SPECTRUM_CONFIG_KEYS
    if str(config.stage) == "highres4096":
        config_keys += (
            "act_cmb_temperature_units_confirmed",
            "minimum_desi_random_realizations",
        )
    for key in config_keys:
        expected = _resolved_config_value(config, key)
        if key not in cfg:
            if key == "kappa_cmb_lmax" and expected is None:
                continue
            return False, f"missing config key {key!r}"
        actual = cfg[key]
        if key == "lmax_mask" and actual is None:
            actual = cfg.get("lmax")
        if not _config_value_matches(actual, expected):
            return False, f"config {key}={cfg[key]!r}, expected {expected!r}"
    if schema in {SCHEMA_MEASUREMENT, SCHEMA_MEASUREMENT_VALIDITY_MASK}:
        expected_left, expected_right = make_bandpower_edges(config)
        if not np.array_equal(stored_ell_left, np.asarray(expected_left, dtype=np.int64)):
            return False, "measurement ell_left does not match the current exact edge table"
        if not np.array_equal(stored_ell_right, np.asarray(expected_right, dtype=np.int64)):
            return False, "measurement ell_right does not match the current exact edge table"
    return True, "compatible"


def _field_spin_from_name(name: str) -> int:
    return 2 if str(name).startswith("s") else 0


def build_covariance_manifest(config: MeasurementConfig) -> Dict[str, object]:
    config.validate()
    ell_left, ell_right = make_bandpower_edges(config)
    specs = default_spectrum_specs()
    groups: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for i, spec_i in enumerate(specs):
        for j, spec_j in enumerate(specs[i:], start=i):
            key = covariance_group_key_for_specs(spec_i, spec_j)
            if key not in groups:
                representative_fields = [spec_i.fields[0], spec_i.fields[1], spec_j.fields[0], spec_j.fields[1]]
                spins = [_field_spin_from_name(name) for name in representative_fields]
                groups[key] = {
                    "key": list(key),
                    "representative_fields": representative_fields,
                    "spins": spins,
                    "class": "scalar" if all(spin == 0 for spin in spins) else "spin2",
                    "blocks": [],
                }
            groups[key]["blocks"].append(
                {
                    "spec_i": spec_i.name,
                    "spec_j": spec_j.name,
                    "spec_i_index": i,
                    "spec_j_index": j,
                    "fields_i": list(spec_i.fields),
                    "fields_j": list(spec_j.fields),
                }
            )
    out_groups = []
    for index, group in enumerate(groups.values()):
        group = dict(group)
        group["index"] = index
        group["n_blocks"] = len(group["blocks"])
        out_groups.append(group)
    n_by_class = {
        "scalar": int(sum(1 for group in out_groups if group["class"] == "scalar")),
        "spin2": int(sum(1 for group in out_groups if group["class"] == "spin2")),
    }
    manifest = {
        "created_utc": utc_now(),
        "pipeline_version": MEASUREMENT_PIPELINE_VERSION,
        "covariance_estimator_version": COVARIANCE_ESTIMATOR_VERSION,
        "covariance_config_digest": covariance_config_digest(config),
        "stage": config.stage,
        "config": covariance_config_payload(config),
        "ell_left": np.asarray(ell_left, dtype=np.int64).tolist(),
        "ell_right": np.asarray(ell_right, dtype=np.int64).tolist(),
        "n_spectra": len(specs),
        "n_covariance_blocks": len(specs) * (len(specs) + 1) // 2,
        "n_covariance_groups": len(out_groups),
        "n_covariance_groups_by_class": n_by_class,
        "spectrum_names": [spec.name for spec in specs],
        "groups": out_groups,
    }
    digest_payload = {key: value for key, value in manifest.items() if key != "created_utc"}
    manifest["manifest_digest"] = _sha256_json(digest_payload)
    return manifest


def validate_covariance_manifest(manifest: Mapping[str, object], config: MeasurementConfig) -> None:
    expected_config_digest = covariance_config_digest(config)
    if str(manifest.get("pipeline_version", "")) != MEASUREMENT_PIPELINE_VERSION:
        raise ValueError("Covariance manifest was built by a different measurement pipeline version.")
    if str(manifest.get("covariance_estimator_version", "")) != COVARIANCE_ESTIMATOR_VERSION:
        raise ValueError("Covariance manifest was built for a different covariance estimator version.")
    if str(manifest.get("covariance_config_digest", "")) != expected_config_digest:
        raise ValueError("Covariance manifest config does not match the requested covariance configuration.")
    expected_left, expected_right = make_bandpower_edges(config)
    if not np.array_equal(
        np.asarray(manifest.get("ell_left", []), dtype=np.int64),
        np.asarray(expected_left, dtype=np.int64),
    ):
        raise ValueError("Covariance manifest ell_left does not match the requested exact edge table.")
    if not np.array_equal(
        np.asarray(manifest.get("ell_right", []), dtype=np.int64),
        np.asarray(expected_right, dtype=np.int64),
    ):
        raise ValueError("Covariance manifest ell_right does not match the requested exact edge table.")
    digest_payload = {
        key: value for key, value in manifest.items() if key not in {"created_utc", "manifest_digest"}
    }
    expected_manifest_digest = _sha256_json(digest_payload)
    if str(manifest.get("manifest_digest", "")) != expected_manifest_digest:
        raise ValueError("Covariance manifest digest is missing or does not match its contents.")
    expected_manifest = build_covariance_manifest(config)
    semantic_payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "manifest_digest"}
    }
    expected_semantic_payload = {
        key: value
        for key, value in expected_manifest.items()
        if key not in {"created_utc", "manifest_digest"}
    }
    if semantic_payload != expected_semantic_payload:
        raise ValueError(
            "Covariance manifest does not match the canonical covariance-group contract."
        )
    specs = default_spectrum_specs()
    expected_blocks = len(specs) * (len(specs) + 1) // 2
    if int(manifest.get("n_spectra", -1)) != len(specs):
        raise ValueError("Covariance manifest spectrum count is inconsistent with the current inventory.")
    if int(manifest.get("n_covariance_blocks", -1)) != expected_blocks:
        raise ValueError("Covariance manifest block count is inconsistent with the current inventory.")


def write_covariance_manifest(path: Path, config: MeasurementConfig, overwrite: bool = False) -> Dict[str, object]:
    manifest = build_covariance_manifest(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text())
        validate_covariance_manifest(existing, config)
        return existing
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return manifest


def load_covariance_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing covariance manifest: {path}")
    return json.loads(path.read_text())


COVARIANCE_WORK_PLAN_VERSION = "xdesi_covariance_work_plan_v1"
HIGHRES_RESOURCE_STRESS_GROUP_INDICES = (
    29,
    47,
    65,
    157,
    218,
    219,
    223,
    224,
    226,
    230,
    231,
)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _map_metadata_for_work_plan(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared map product: {path}")
    with h5py.File(path, "r") as h5:
        if str(h5.attrs.get("schema", "")) != SCHEMA_MAPS:
            raise ValueError(f"Prepared map {path} has the wrong schema.")
        if "metadata_json" not in h5.attrs:
            raise ValueError(f"Prepared map {path} has no embedded metadata_json.")
        raw_metadata = h5.attrs["metadata_json"]
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(str(raw_metadata))
        if str(h5.attrs.get("map_product_id", "")) != validate_map_metadata_identity(metadata):
            raise ValueError(f"Prepared map {path} has inconsistent map-product identity metadata.")
    return metadata


def _balanced_covariance_bundles(
    groups: Sequence[Mapping[str, object]],
    groups_per_bundle: int,
    *,
    stress_group_indices: Sequence[int] = (),
) -> List[List[Mapping[str, object]]]:
    if groups_per_bundle <= 0:
        raise ValueError("groups_per_bundle must be positive.")
    if not groups:
        return []
    by_index = {int(group["index"]): group for group in groups}
    stress = [by_index[index] for index in stress_group_indices if index in by_index]
    stress_ids = {int(group["index"]) for group in stress}
    remaining = [group for group in groups if int(group["index"]) not in stress_ids]
    if stress and len(stress) < groups_per_bundle:
        risk_sorted = sorted(
            remaining,
            key=lambda group: (
                -sum(str(value).startswith("s") for value in group.get("representative_fields", [])),
                -int(group.get("n_blocks", 0)),
                int(group["index"]),
            ),
        )
        fill = risk_sorted[: groups_per_bundle - len(stress)]
        stress.extend(fill)
        fill_ids = {int(group["index"]) for group in fill}
        remaining = [group for group in remaining if int(group["index"]) not in fill_ids]

    n_regular_bundles = (
        (len(remaining) + groups_per_bundle - 1) // groups_per_bundle if remaining else 0
    )
    ordered = sorted(
        remaining,
        key=lambda group: (
            -int(group.get("n_blocks", 0)),
            -len(set(str(value) for value in group.get("representative_fields", []))),
            int(group["index"]),
        ),
    )
    bundles: List[List[Mapping[str, object]]] = [stress] if stress else []
    regular_bundles: List[List[Mapping[str, object]]] = [
        [] for _ in range(n_regular_bundles)
    ]
    # Round-robin after sorting by a deterministic cost proxy spreads expensive
    # groups while keeping bundle sizes within one of each other.
    for position, group in enumerate(ordered):
        regular_bundles[position % n_regular_bundles].append(group)
    bundles.extend(regular_bundles)
    if max(len(bundle) for bundle in bundles) > groups_per_bundle:
        raise AssertionError("Balanced covariance plan exceeded its per-node group cap.")
    regular_sizes = [len(bundle) for bundle in regular_bundles]
    if regular_sizes and max(regular_sizes) - min(regular_sizes) > 1:
        raise AssertionError("Balanced covariance plan produced avoidably uneven bundle sizes.")
    return bundles


def build_covariance_work_plan(
    config: MeasurementConfig,
    manifest: Mapping[str, object],
    maps: Path,
    spectra: Path,
    *,
    groups_per_bundle: int,
) -> Dict[str, object]:
    """Build a deterministic resume plan containing only missing covariance groups."""

    validate_covariance_manifest(manifest, config)
    map_ok, map_reason = _existing_product_matches_config(maps, SCHEMA_MAPS, config)
    if not map_ok:
        raise ValueError(f"Prepared map {maps} is incompatible ({map_reason}).")
    map_metadata = _map_metadata_for_work_plan(maps)
    config = _config_from_map_metadata(config, map_metadata)
    map_product_id = _required_map_product_id(map_metadata)
    spectra_ok, spectra_reason = _existing_product_matches_config(
        spectra,
        measurement_schema_for_config(config),
        config,
        expected_map_product_id=map_product_id,
    )
    if not spectra_ok:
        raise ValueError(f"Pilot spectra {spectra} are incompatible ({spectra_reason}).")

    compatible: List[Dict[str, object]] = []
    missing: List[Mapping[str, object]] = []
    for group in manifest["groups"]:
        shard = block_shard_path(config, group)
        group_mask_digest = _group_mask_digest_from_metadata(group, map_metadata)
        if not shard.exists():
            missing.append(group)
            continue
        ok, reason = _covariance_shard_compatibility(
            shard,
            group,
            manifest,
            config,
            map_product_id,
            group_mask_digest,
        )
        if not ok:
            raise ValueError(
                f"Existing covariance shard {shard} is incompatible ({reason}); "
                "refusing to overwrite or hide it in a resume plan."
            )
        compatible.append(
            {
                "group_index": int(group["index"]),
                "group_digest": _group_digest(group),
                "path": str(shard),
                "sha256": _sha256_file(shard),
            }
        )

    stress_indices = (
        HIGHRES_RESOURCE_STRESS_GROUP_INDICES
        if str(config.stage) == "highres4096"
        else ()
    )
    bundles = _balanced_covariance_bundles(
        missing,
        int(groups_per_bundle),
        stress_group_indices=stress_indices,
    )
    bundle_payload = []
    for index, bundle in enumerate(bundles):
        bundle_payload.append(
            {
                "bundle_index": index,
                "group_indices": [int(group["index"]) for group in bundle],
                "group_digests": [_group_digest(group) for group in bundle],
                "n_blocks": int(sum(int(group["n_blocks"]) for group in bundle)),
            }
        )
    payload: Dict[str, object] = {
        "created_utc": utc_now(),
        "version": COVARIANCE_WORK_PLAN_VERSION,
        "stage": str(config.stage),
        "pipeline_version": MEASUREMENT_PIPELINE_VERSION,
        "covariance_estimator_version": COVARIANCE_ESTIMATOR_VERSION,
        "covariance_config_digest": covariance_config_digest(config),
        "manifest_path": str(manifest_path(config)),
        "manifest_digest": str(manifest["manifest_digest"]),
        "maps_path": str(maps),
        "map_product_id": map_product_id,
        "map_size_bytes": int(maps.stat().st_size),
        "spectra_path": str(spectra),
        "spectra_size_bytes": int(spectra.stat().st_size),
        "spectra_sha256": _sha256_file(spectra),
        "groups_per_bundle": int(groups_per_bundle),
        "n_manifest_groups": int(len(manifest["groups"])),
        "n_reused_groups": int(len(compatible)),
        "n_missing_groups": int(len(missing)),
        "n_bundles": int(len(bundle_payload)),
        "stress_bundle_index": 0 if bundle_payload else None,
        "stress_group_indices": (
            bundle_payload[0]["group_indices"] if bundle_payload else []
        ),
        "reused_groups": compatible,
        "bundles": bundle_payload,
    }
    digest_payload = {
        key: value for key, value in payload.items() if key not in {"created_utc", "plan_digest"}
    }
    payload["plan_digest"] = _sha256_json(digest_payload)
    validate_covariance_work_plan(payload, manifest, config)
    return payload


def validate_covariance_work_plan(
    plan: Mapping[str, object],
    manifest: Mapping[str, object],
    config: MeasurementConfig,
) -> None:
    validate_covariance_manifest(manifest, config)
    if str(plan.get("version", "")) != COVARIANCE_WORK_PLAN_VERSION:
        raise ValueError("Covariance work plan has the wrong version.")
    if str(plan.get("stage", "")) != str(config.stage):
        raise ValueError("Covariance work plan is bound to a different measurement stage.")
    if str(plan.get("pipeline_version", "")) != MEASUREMENT_PIPELINE_VERSION:
        raise ValueError("Covariance work plan has the wrong measurement-pipeline version.")
    if str(plan.get("covariance_estimator_version", "")) != COVARIANCE_ESTIMATOR_VERSION:
        raise ValueError("Covariance work plan has the wrong covariance-estimator version.")
    if str(plan.get("manifest_digest", "")) != str(manifest["manifest_digest"]):
        raise ValueError("Covariance work plan is bound to a different manifest.")
    if str(plan.get("covariance_config_digest", "")) != covariance_config_digest(config):
        raise ValueError("Covariance work plan is bound to a different covariance config.")
    digest_payload = {
        key: value for key, value in plan.items() if key not in {"created_utc", "plan_digest"}
    }
    if str(plan.get("plan_digest", "")) != _sha256_json(digest_payload):
        raise ValueError("Covariance work-plan digest does not match its contents.")
    canonical_groups = {int(group["index"]): group for group in manifest["groups"]}
    if int(plan.get("n_manifest_groups", -1)) != len(canonical_groups):
        raise ValueError("Covariance work-plan manifest-group count is inconsistent.")
    if int(plan.get("groups_per_bundle", 0)) <= 0:
        raise ValueError("Covariance work plan has a non-positive per-bundle group cap.")
    reused = list(plan.get("reused_groups", []))
    bundles = list(plan.get("bundles", []))
    reused_indices = [int(item["group_index"]) for item in reused]
    missing_indices: List[int] = []
    for expected_bundle_index, bundle in enumerate(bundles):
        if int(bundle.get("bundle_index", -1)) != expected_bundle_index:
            raise ValueError("Covariance work-plan bundle indices are not canonical and contiguous.")
        indices = [int(value) for value in bundle.get("group_indices", [])]
        digests = [str(value) for value in bundle.get("group_digests", [])]
        if not indices or len(indices) != len(digests):
            raise ValueError("Covariance work-plan bundle inventory is empty or malformed.")
        if len(indices) > int(plan["groups_per_bundle"]):
            raise ValueError("Covariance work-plan bundle exceeds its resource cap.")
        for index, digest in zip(indices, digests):
            if index not in canonical_groups or digest != _group_digest(canonical_groups[index]):
                raise ValueError("Covariance work-plan group identity disagrees with the manifest.")
        missing_indices.extend(indices)
    for item in reused:
        index = int(item["group_index"])
        if index not in canonical_groups or str(item["group_digest"]) != _group_digest(
            canonical_groups[index]
        ):
            raise ValueError("Covariance work-plan reused-group identity disagrees with the manifest.")
    all_indices = reused_indices + missing_indices
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Covariance work plan assigns a group more than once.")
    if set(all_indices) != set(canonical_groups):
        raise ValueError("Covariance work plan does not cover every manifest group exactly once.")
    if int(plan.get("n_reused_groups", -1)) != len(reused_indices):
        raise ValueError("Covariance work-plan reused-group count is inconsistent.")
    if int(plan.get("n_missing_groups", -1)) != len(missing_indices):
        raise ValueError("Covariance work-plan missing-group count is inconsistent.")
    if int(plan.get("n_bundles", -1)) != len(bundles):
        raise ValueError("Covariance work-plan bundle count is inconsistent.")
    if bundles:
        if int(plan.get("stress_bundle_index", -1)) != 0:
            raise ValueError("Covariance work plan must place its production stress bundle first.")
        if [int(value) for value in plan.get("stress_group_indices", [])] != [
            int(value) for value in bundles[0]["group_indices"]
        ]:
            raise ValueError("Covariance work-plan stress inventory disagrees with bundle zero.")


def validate_covariance_work_plan_frozen_inputs(
    plan: Mapping[str, object],
    manifest: Mapping[str, object],
    config: MeasurementConfig,
) -> None:
    """Re-attest immutable inputs immediately before final assembly."""

    validate_covariance_work_plan(plan, manifest, config)
    maps = Path(str(plan["maps_path"]))
    spectra = Path(str(plan["spectra_path"]))
    if not maps.exists() or int(maps.stat().st_size) != int(plan["map_size_bytes"]):
        raise ValueError("Frozen covariance work-plan map input is missing or changed size.")
    map_metadata = _map_metadata_for_work_plan(maps)
    if _required_map_product_id(map_metadata) != str(plan["map_product_id"]):
        raise ValueError("Frozen covariance work-plan map identity changed before assembly.")
    if not spectra.exists() or int(spectra.stat().st_size) != int(plan["spectra_size_bytes"]):
        raise ValueError("Frozen covariance work-plan spectra input is missing or changed size.")
    if _sha256_file(spectra) != str(plan["spectra_sha256"]):
        raise ValueError("Frozen covariance work-plan spectra SHA256 changed before assembly.")
    for item in plan.get("reused_groups", []):
        shard = Path(str(item["path"]))
        if not shard.exists() or _sha256_file(shard) != str(item["sha256"]):
            raise ValueError(
                f"Frozen reused covariance shard for group {item['group_index']} changed."
            )


def _groups_for_class(manifest: Mapping[str, object], cov_class: str) -> List[Mapping[str, object]]:
    groups = list(manifest["groups"])
    if cov_class == "all":
        return groups
    return [group for group in groups if group["class"] == cov_class]


def _read_string_dataset(ds: h5py.Dataset) -> List[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in ds[:]]


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - 3600 * hours - 60 * minutes
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:04.1f}s"
    if minutes:
        return f"{minutes:d}m{secs:04.1f}s"
    return f"{secs:.1f}s"


def _proc_cpu_seconds(pid: int) -> float:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        close = text.rfind(")")
        if close < 0:
            return 0.0
        fields = text[close + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        return float(ticks) / float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except Exception:
        return 0.0


def _proc_status(pid: int) -> Dict[str, object]:
    out: Dict[str, object] = {"state": "unknown", "threads": np.nan, "rss_gb": np.nan, "vms_gb": np.nan}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                out["state"] = line.split(":", 1)[1].strip()
            elif line.startswith("Threads:"):
                out["threads"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("VmRSS:"):
                out["rss_gb"] = float(line.split()[1]) / 1024.0**2
            elif line.startswith("VmSize:"):
                out["vms_gb"] = float(line.split()[1]) / 1024.0**2
    except Exception:
        pass
    return out


@contextmanager
def heartbeat(label: str, interval: float = 120.0, allocated_cpus: Optional[int] = None):
    """Print process CPU/memory status periodically while inside long C calls."""

    interval = float(interval)
    if interval <= 0.0:
        yield
        return
    pid = os.getpid()
    stop = threading.Event()
    start_wall = time.monotonic()
    start_cpu = _proc_cpu_seconds(pid)
    last_wall = start_wall
    last_cpu = start_cpu

    def run() -> None:
        nonlocal last_wall, last_cpu
        while not stop.wait(interval):
            now = time.monotonic()
            cpu = _proc_cpu_seconds(pid)
            recent_cores = (cpu - last_cpu) / max(now - last_wall, 1.0e-9)
            avg_cores = (cpu - start_cpu) / max(now - start_wall, 1.0e-9)
            status = _proc_status(pid)
            alloc = ""
            if allocated_cpus and allocated_cpus > 0:
                alloc = f", recent_alloc={100.0 * recent_cores / allocated_cpus:.1f}%"
            print(
                f"[{utc_now()}] heartbeat {label}: elapsed={_format_seconds(now - start_wall)}, "
                f"cpu_recent={recent_cores:.2f} cores, cpu_avg={avg_cores:.2f} cores{alloc}, "
                f"rss={float(status['rss_gb']):.2f} GB, vms={float(status['vms_gb']):.2f} GB, "
                f"threads={status['threads']}, state={status['state']}",
                flush=True,
            )
            last_wall = now
            last_cpu = cpu

    thread = threading.Thread(target=run, name=f"heartbeat:{label}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


@contextmanager
def timed_step(label: str):
    start = time.monotonic()
    print(f"[{utc_now()}] START {label}", flush=True)
    try:
        yield
    finally:
        print(f"[{utc_now()}] DONE  {label} in {_format_seconds(time.monotonic() - start)}", flush=True)


def _read_spectra_product(path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    with h5py.File(path, "r") as h5:
        schema = str(h5.attrs.get("schema", ""))
        if schema not in {SCHEMA_MEASUREMENT, SCHEMA_MEASUREMENT_VALIDITY_MASK}:
            raise ValueError(f"{path} has unsupported measurement schema {schema!r}.")
        mean_convention = str(h5.attrs.get("desi_galaxy_auto_mean_convention", ""))
        if mean_convention != DESI_GALAXY_AUTO_MEAN_CONVENTION:
            raise ValueError(
                f"{path} has DESI galaxy-auto mean convention {mean_convention!r}; "
                f"expected {DESI_GALAXY_AUTO_MEAN_CONVENTION!r}. Refusing to relabel "
                "an old shot-noise-subtracted spectra cache during assembly."
            )
        map_metadata = json.loads(h5.attrs["map_metadata_json"])
        config = json.loads(h5.attrs["config_json"])
        spectra: Dict[str, Dict[str, object]] = {}
        for name in h5["spectra"]:
            g = h5[f"spectra/{name}"]
            spectra[name] = {
                "name": name,
                "family": str(g.attrs["family"]),
                "label": str(g.attrs["label"]),
                "theory_key": str(g.attrs["theory_key"]),
                "component_label": str(g.attrs["component_label"]),
                "fields": tuple(json.loads(g.attrs["fields"])),
                "component": int(g.attrs["component"]),
                "component_labels": json.loads(g.attrs["component_labels"]),
                "metadata": json.loads(g.attrs["metadata_json"]),
                "ell": g["ell"][:],
                "cl": g["cl"][:],
                "cl_all_components": g["cl_all_components"][:],
                "pcl_all_components": g["pcl_all_components"][:],
                "bandpower_window_selected": g["bandpower_window_selected"][:],
                "noise_decoupled_all_components": (
                    None if "noise_decoupled_all_components" not in g else g["noise_decoupled_all_components"][:]
                ),
            }
            if "pair_overlap_mean_subtraction_json" in g.attrs:
                spectra[name]["pair_overlap_mean_subtraction"] = json.loads(g.attrs["pair_overlap_mean_subtraction_json"])
        null_tests: Dict[str, Dict[str, object]] = {}
        if "null_tests" in h5:
            for name in h5["null_tests"]:
                g = h5[f"null_tests/{name}"]
                item = {"ell": g["ell"][:], "cl": g["cl"][:]}
                for key, value in g.attrs.items():
                    if isinstance(value, str):
                        try:
                            item[key] = json.loads(value)
                        except json.JSONDecodeError:
                            item[key] = value
                    else:
                        item[key] = value
                null_tests[name] = item
        field_metadata = {}
        if "fields" in h5 and "metadata_json" in h5["fields"].attrs:
            field_metadata = json.loads(h5["fields"].attrs["metadata_json"])
        result = {
            "schema": schema,
            "created_utc": utc_now(),
            "config": config,
            "ell": h5["ell"][:],
            "ell_left": h5["ell_left"][:],
            "ell_right": h5["ell_right"][:],
            "binning": str(h5.attrs.get("binning", config.get("binning", "sqrt"))),
            "ell_max_inclusive": int(h5.attrs.get("ell_max_inclusive", config.get("lmax", 0))),
            "spectra": spectra,
            "covariance_blocks": {},
            "joint": None,
            "null_tests": null_tests,
            "input_cls_for_covariance": {},
            "workspace_keys": [],
            "covariance_workspace_keys": [],
            "field_metadata": field_metadata,
        }
    return result, map_metadata


def _write_input_cls_group(parent: h5py.Group, input_cl_cache: Mapping[Tuple[str, ...], np.ndarray]) -> None:
    group = parent.create_group("input_cls_for_covariance")
    for key, cl in input_cl_cache.items():
        if len(key) == 3:
            input_mode, a, b = key
            dataset_name = f"{input_mode}__{a}__x__{b}"
        else:
            input_mode = "legacy"
            a, b = key
            dataset_name = f"{a}__x__{b}"
        ds = _write_dataset(group, dataset_name, np.asarray(cl), dtype="f8")
        ds.attrs["input_mode"] = str(input_mode)
        ds.attrs["field_a"] = str(a)
        ds.attrs["field_b"] = str(b)


def _read_input_cls_group(parent: h5py.Group) -> Dict[Tuple[str, ...], np.ndarray]:
    out: Dict[Tuple[str, ...], np.ndarray] = {}
    if "input_cls_for_covariance" not in parent:
        return out
    for name in parent["input_cls_for_covariance"]:
        ds = parent[f"input_cls_for_covariance/{name}"]
        key = (str(ds.attrs["input_mode"]), str(ds.attrs["field_a"]), str(ds.attrs["field_b"]))
        out[key] = ds[:]
    return out


def _expected_block_dataset_names(group: Mapping[str, object]) -> Set[str]:
    return {
        f"{str(block['spec_i'])}__x__{str(block['spec_j'])}"
        for block in group.get("blocks", [])
    }


def _covariance_shard_compatibility(
    path: Path,
    group: Mapping[str, object],
    manifest: Mapping[str, object],
    config: MeasurementConfig,
    map_product_id: str,
    group_mask_digest: str,
) -> Tuple[bool, str]:
    if not path.exists():
        return False, "file does not exist"
    try:
        with h5py.File(path, "r") as h5:
            expected_attrs = {
                "pipeline_version": MEASUREMENT_PIPELINE_VERSION,
                "covariance_estimator_version": COVARIANCE_ESTIMATOR_VERSION,
                "covariance_config_digest": covariance_config_digest(config),
                "manifest_digest": str(manifest["manifest_digest"]),
                "group_digest": _group_digest(group),
                "map_product_id": map_product_id,
                "group_mask_digest": group_mask_digest,
            }
            for key, expected in expected_attrs.items():
                actual = str(h5.attrs.get(key, ""))
                if actual != str(expected):
                    return False, f"{key}={actual!r}, expected {expected!r}"
            if int(h5.attrs.get("group_index", -1)) != int(group["index"]):
                return False, "group index does not match"
            if "covariance_blocks" not in h5:
                return False, "covariance_blocks group is missing"
            actual_names = set(h5["covariance_blocks"].keys())
            expected_names = _expected_block_dataset_names(group)
            if actual_names != expected_names:
                return False, "covariance block inventory does not match manifest group"
            expected_shape = (int(config.n_bins), int(config.n_bins))
            for name in actual_names:
                if h5[f"covariance_blocks/{name}"].shape != expected_shape:
                    return False, f"block {name!r} has the wrong shape"
            if "input_cls_for_covariance" not in h5:
                return False, "input covariance spectra are missing"
    except Exception as exc:
        return False, f"could not validate shard: {exc}"
    return True, "compatible"


def run_prepare(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    bundle = SurveyBundle.from_root(args.survey_root)
    output = Path(args.maps_out).resolve() if args.maps_out else config.default_maps_path
    if output.exists() and not args.force:
        ok, reason = _existing_product_matches_config(output, SCHEMA_MAPS, config)
        if ok:
            print(f"[{utc_now()}] Reusing existing compatible map product: {output}", flush=True)
            return
        raise FileExistsError(f"{output} exists but is not compatible ({reason}); pass --force to replace it.")
    print(f"[{utc_now()}] Preparing maps for {config.stage}: {output}", flush=True)
    fields, metadata = build_probe_maps(bundle, config)
    save_map_product(output, fields, metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote {output}", flush=True)


def run_spectra(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path

    if getattr(args, "patch_shear_only", False):
        raise ValueError(
            "--patch-shear-only is unsafe for pipeline v2: corrected masks and mode-coupling windows "
            "change every spectrum with a shear endpoint, not only the four autos. Run a full spectra "
            "measurement (use --force only to replace an incompatible v2 output)."
        )

    print(f"[{utc_now()}] Loading maps for spectra: {maps}", flush=True)
    map_fields, map_metadata = load_map_product(maps)
    config = _config_from_map_metadata(config, map_metadata)
    config.output_dir = args.output_dir
    output = Path(args.spectra_out).resolve() if args.spectra_out else spectra_path(config)
    if output.exists() and not args.force:
        ok, reason = _existing_product_matches_config(
            output,
            measurement_schema_for_config(config),
            config,
            expected_map_product_id=_required_map_product_id(map_metadata),
        )
        if ok:
            print(f"[{utc_now()}] Reusing existing compatible spectra product: {output}", flush=True)
            return
        raise FileExistsError(f"{output} exists but is not compatible ({reason}); pass --force to replace it.")
    config.compute_covariance = False
    from multiprobe_namaster import measure_all

    result = measure_all(map_fields, config, verbose=not args.quiet)
    save_measurement_product(output, result, map_metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote spectra product {output}", flush=True)


def run_make_cov_manifest(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    # Construct the exact NaMaster bin object locally so invalid edge tables
    # fail before any scheduler jobs are submitted.
    make_bins(config)
    output = Path(args.manifest_out).resolve() if args.manifest_out else manifest_path(config)
    manifest = write_covariance_manifest(output, config, overwrite=args.force)
    print(
        f"[{utc_now()}] Wrote covariance manifest {output} "
        f"({manifest['n_covariance_groups']} groups; {manifest['n_covariance_blocks']} blocks)",
        flush=True,
    )


def run_make_cov_work_plan(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    manifest = load_covariance_manifest(manifest_file)
    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path
    spectra = Path(args.spectra_path).resolve() if args.spectra_path else spectra_path(config)
    output = Path(args.plan_out).resolve() if args.plan_out else covariance_work_plan_path(config)
    plan = build_covariance_work_plan(
        config,
        manifest,
        maps,
        spectra,
        groups_per_bundle=int(args.groups_per_bundle),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, output)
    print(
        f"[{utc_now()}] Wrote covariance work plan {output}: "
        f"reused={plan['n_reused_groups']} missing={plan['n_missing_groups']} "
        f"bundles={plan['n_bundles']} digest={plan['plan_digest']}",
        flush=True,
    )


def run_show_cov_work_bundle(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    manifest = load_covariance_manifest(manifest_file)
    plan_file = Path(args.plan_path).resolve() if args.plan_path else covariance_work_plan_path(config)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_covariance_work_plan(plan, manifest, config)
    batch_id = int(
        args.batch_id
        if args.batch_id is not None
        else os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    )
    bundles = list(plan["bundles"])
    if not (0 <= batch_id < len(bundles)):
        raise ValueError(f"Covariance bundle {batch_id} is outside 0..{len(bundles) - 1}.")
    # Machine-readable stdout for the Slurm bundle worker.  Validation emits no
    # other stdout, so shell read/mapfile remains fail-closed.
    print(" ".join(str(value) for value in bundles[batch_id]["group_indices"]))


def run_cov_key(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    manifest = load_covariance_manifest(manifest_file)
    validate_covariance_manifest(manifest, config)
    groups = _groups_for_class(manifest, args.cov_class)
    task_id = int(args.task_id if args.task_id is not None else os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_id >= len(groups):
        print(f"[{utc_now()}] task_id={task_id} outside {args.cov_class} group count={len(groups)}; skipping.", flush=True)
        return
    group = groups[task_id]

    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path
    # Resume jobs must not spend ~25 minutes loading/building a nside=4096
    # field merely to discover that its immutable shard already exists.  The
    # content-addressed map metadata is sufficient to validate the shard's mask
    # identity; missing work still takes the full map-loading path below.
    map_metadata_preflight = _map_metadata_for_work_plan(maps)
    config = _config_from_map_metadata(config, map_metadata_preflight)
    config.output_dir = args.output_dir
    map_product_id_preflight = _required_map_product_id(map_metadata_preflight)
    output_preflight = block_shard_path(config, group)
    if output_preflight.exists() and not args.force:
        compatible, reason = _covariance_shard_compatibility(
            output_preflight,
            group,
            manifest,
            config,
            map_product_id_preflight,
            _group_mask_digest_from_metadata(group, map_metadata_preflight),
        )
        if compatible:
            print(
                f"[{utc_now()}] Reusing compatible covariance shard without loading maps: "
                f"{output_preflight}",
                flush=True,
            )
            return
        raise FileExistsError(
            f"Existing covariance shard {output_preflight} is incompatible ({reason}); "
            "pass --force to replace it."
        )
    print(
        f"[{utc_now()}] Computing covariance group {group['index']} "
        f"({group['class']}, {group['n_blocks']} blocks) from {maps}",
        flush=True,
    )
    allocated_cpus = int(os.environ.get("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    with heartbeat(
        f"cov-key group={group['index']} class={group['class']}",
        interval=float(args.heartbeat_interval),
        allocated_cpus=allocated_cpus,
    ):
        with timed_step(f"group {group['index']} load map product"):
            map_fields, map_metadata = load_map_product(maps, field_names=_field_names_for_groups([group]))
        config = _config_from_map_metadata(config, map_metadata)
        config.output_dir = args.output_dir
        map_product_id = _required_map_product_id(map_metadata)
        group_mask_digest = _group_mask_digest_from_fields(group, map_fields)
        output = block_shard_path(config, group)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.force:
            compatible, reason = _covariance_shard_compatibility(
                output, group, manifest, config, map_product_id, group_mask_digest
            )
            if compatible:
                print(f"[{utc_now()}] Reusing compatible covariance shard {output}", flush=True)
                return
            raise FileExistsError(
                f"Existing covariance shard {output} is incompatible ({reason}); pass --force to replace it."
            )
        with timed_step(f"group {group['index']} make bins"):
            bins = make_bins(config)
        with timed_step(f"group {group['index']} build NaMaster fields"):
            fields = _build_cov_fields(map_fields, config, [group])
        with timed_step(
            f"group {group['index']} get/build covariance workspace "
            f"({','.join(group['representative_fields'])})"
        ):
            cw = _get_or_build_cov_workspace(
                group, fields, config, use_cache=not getattr(args, "no_cov_workspace_cache", False)
            )
        specs = {spec.name: spec for spec in default_spectrum_specs()}
        workspace_cache = {}
        input_cl_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        blocks: Dict[Tuple[str, str], np.ndarray] = {}
        for block_info in group["blocks"]:
            spec_i = specs[str(block_info["spec_i"])]
            spec_j = specs[str(block_info["spec_j"])]
            with timed_step(f"group {group['index']} block {spec_i.name} x {spec_j.name}"):
                blocks[(spec_i.name, spec_j.name)] = compute_covariance_block_with_workspace(
                    spec_i,
                    spec_j,
                    fields,
                    bins,
                    workspace_cache,
                    cw,
                    input_cl_cache,
                    config,
                )

    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w", track_order=True) as h5:
        h5.attrs["created_utc"] = utc_now()
        h5.attrs["stage"] = config.stage
        h5.attrs["config_json"] = _json_dumps(config.to_dict())
        h5.attrs["group_json"] = json.dumps(group)
        h5.attrs["group_index"] = int(group["index"])
        h5.attrs["group_class"] = str(group["class"])
        h5.attrs["pipeline_version"] = MEASUREMENT_PIPELINE_VERSION
        h5.attrs["covariance_estimator_version"] = COVARIANCE_ESTIMATOR_VERSION
        h5.attrs["covariance_config_digest"] = covariance_config_digest(config)
        h5.attrs["manifest_digest"] = str(manifest["manifest_digest"])
        h5.attrs["group_digest"] = _group_digest(group)
        h5.attrs["map_product_id"] = map_product_id
        h5.attrs["group_mask_digest"] = group_mask_digest
        bg = h5.create_group("covariance_blocks")
        for (name_i, name_j), block in blocks.items():
            ds = _write_dataset(bg, f"{name_i}__x__{name_j}", block, dtype="f8")
            ds.attrs["spectrum_i"] = name_i
            ds.attrs["spectrum_j"] = name_j
        _write_input_cls_group(h5, input_cl_cache)
    os.replace(tmp, output)
    print(f"[{utc_now()}] Wrote covariance shard {output}", flush=True)


def _compute_covariance_group(
    group: Mapping[str, object],
    manifest: Mapping[str, object],
    fields: Mapping[str, object],
    bins: object,
    config: MeasurementConfig,
    map_product_id: str,
    *,
    force: bool,
    use_cache: bool = True,
) -> Path:
    group_mask_digest = _group_mask_digest_from_fields(group, fields)
    output = block_shard_path(config, group)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        compatible, reason = _covariance_shard_compatibility(
            output, group, manifest, config, map_product_id, group_mask_digest
        )
        if compatible:
            print(f"[{utc_now()}] Reusing compatible covariance shard {output}", flush=True)
            return output
        raise FileExistsError(
            f"Existing covariance shard {output} is incompatible ({reason}); pass --force to replace it."
        )
    with timed_step(
        f"group {group['index']} get/build covariance workspace "
        f"({','.join(group['representative_fields'])})"
    ):
        cw = _get_or_build_cov_workspace(group, fields, config, use_cache=use_cache)
    specs = {spec.name: spec for spec in default_spectrum_specs()}
    workspace_cache = {}
    input_cl_cache: Dict[Tuple[str, ...], np.ndarray] = {}
    blocks: Dict[Tuple[str, str], np.ndarray] = {}
    for block_info in group["blocks"]:
        spec_i = specs[str(block_info["spec_i"])]
        spec_j = specs[str(block_info["spec_j"])]
        with timed_step(f"group {group['index']} block {spec_i.name} x {spec_j.name}"):
            blocks[(spec_i.name, spec_j.name)] = compute_covariance_block_with_workspace(
                spec_i,
                spec_j,
                fields,
                bins,
                workspace_cache,
                cw,
                input_cl_cache,
                config,
            )
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w", track_order=True) as h5:
        h5.attrs["created_utc"] = utc_now()
        h5.attrs["stage"] = config.stage
        h5.attrs["config_json"] = _json_dumps(config.to_dict())
        h5.attrs["group_json"] = json.dumps(group)
        h5.attrs["group_index"] = int(group["index"])
        h5.attrs["group_class"] = str(group["class"])
        h5.attrs["pipeline_version"] = MEASUREMENT_PIPELINE_VERSION
        h5.attrs["covariance_estimator_version"] = COVARIANCE_ESTIMATOR_VERSION
        h5.attrs["covariance_config_digest"] = covariance_config_digest(config)
        h5.attrs["manifest_digest"] = str(manifest["manifest_digest"])
        h5.attrs["group_digest"] = _group_digest(group)
        h5.attrs["map_product_id"] = map_product_id
        h5.attrs["group_mask_digest"] = group_mask_digest
        bg = h5.create_group("covariance_blocks")
        for (name_i, name_j), block in blocks.items():
            ds = _write_dataset(bg, f"{name_i}__x__{name_j}", block, dtype="f8")
            ds.attrs["spectrum_i"] = name_i
            ds.attrs["spectrum_j"] = name_j
        _write_input_cls_group(h5, input_cl_cache)
    os.replace(tmp, output)
    print(f"[{utc_now()}] Wrote covariance shard {output}", flush=True)
    return output


def run_cov_batch(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    manifest = load_covariance_manifest(manifest_file)
    validate_covariance_manifest(manifest, config)
    groups = _groups_for_class(manifest, args.cov_class)
    batch_id = int(args.batch_id if args.batch_id is not None else os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    start = batch_id * batch_size
    stop = min(start + batch_size, len(groups))
    if start >= len(groups):
        print(f"[{utc_now()}] batch_id={batch_id} outside {args.cov_class} group count={len(groups)}; skipping.", flush=True)
        return
    selected_groups = groups[start:stop]
    parallel_groups = int(getattr(args, "parallel_groups", 1))
    if parallel_groups > 1 and len(selected_groups) > 1:
        env_base = os.environ.copy()
        omp_threads = max(1, int(getattr(args, "omp_threads_per_group", 1)))
        env_base["OMP_NUM_THREADS"] = str(omp_threads)
        env_base["OMP_PROC_BIND"] = "spread"
        env_base["OMP_PLACES"] = "cores"
        env_base["OMP_MAX_ACTIVE_LEVELS"] = "1"
        env_base["MKL_NUM_THREADS"] = "1"
        env_base["OPENBLAS_NUM_THREADS"] = "1"
        env_base["NUMEXPR_NUM_THREADS"] = "1"
        common = [
            sys.executable,
            str(Path(__file__).resolve()),
            "cov-key",
            "--stage",
            str(config.stage),
            "--output-dir",
            str(args.output_dir),
            "--cov-class",
            str(args.cov_class),
            "--manifest-path",
            str(manifest_file),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
        ]
        if args.maps_path:
            common.extend(["--maps-path", str(args.maps_path)])
        if args.force:
            common.append("--force")
        if getattr(args, "no_cov_workspace_cache", False):
            common.append("--no-cov-workspace-cache")
        running = []
        for group in selected_groups:
            task_id = groups.index(group)
            cmd = [*common, "--task-id", str(task_id)]
            print(
                f"[{utc_now()}] Launching group {group['index']} as subprocess "
                f"(task_id={task_id}, OMP_NUM_THREADS={omp_threads})",
                flush=True,
            )
            running.append(subprocess.Popen(cmd, env=env_base))
            while len(running) >= parallel_groups:
                proc = running.pop(0)
                ret = proc.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, proc.args)
        for proc in running:
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, proc.args)
        return

    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path
    print(
        f"[{utc_now()}] Computing covariance batch {batch_id} "
        f"({args.cov_class} groups {start}..{stop - 1}) from {maps}",
        flush=True,
    )
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1")))
    with heartbeat(
        f"cov-batch id={batch_id} class={args.cov_class} groups={start}..{stop - 1}",
        interval=float(args.heartbeat_interval),
        allocated_cpus=allocated_cpus,
    ):
        with timed_step(f"batch {batch_id} load map product"):
            map_fields, map_metadata = load_map_product(maps, field_names=_field_names_for_groups(selected_groups))
        config = _config_from_map_metadata(config, map_metadata)
        config.output_dir = args.output_dir
        map_product_id = _required_map_product_id(map_metadata)
        with timed_step(f"batch {batch_id} make bins"):
            bins = make_bins(config)
        with timed_step(f"batch {batch_id} build NaMaster fields"):
            fields = _build_cov_fields(map_fields, config, selected_groups)
        use_cache = not getattr(args, "no_cov_workspace_cache", False)
        for group in selected_groups:
            print(
                f"[{utc_now()}] Group {group['index']} ({group['class']}, {group['n_blocks']} blocks)",
                flush=True,
            )
            _compute_covariance_group(
                group,
                manifest,
                fields,
                bins,
                config,
                map_product_id,
                force=args.force,
                use_cache=use_cache,
            )


def run_assemble(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    spec_file = Path(args.spectra_path).resolve() if args.spectra_path else spectra_path(config)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    result, map_metadata = _read_spectra_product(spec_file)
    config = _config_from_map_metadata(config, map_metadata)
    config.output_dir = args.output_dir
    map_product_id = _required_map_product_id(map_metadata)
    output = Path(args.measurement_out).resolve() if args.measurement_out else config.default_measurement_path
    spectra_ok, spectra_reason = _existing_product_matches_config(
        spec_file,
        measurement_schema_for_config(config),
        config,
        expected_map_product_id=map_product_id,
    )
    if not spectra_ok:
        raise ValueError(f"Spectra product {spec_file} is incompatible ({spectra_reason}).")
    manifest = load_covariance_manifest(manifest_file)
    validate_covariance_manifest(manifest, config)
    if covariance_config_digest(result["config"]) != covariance_config_digest(config):
        raise ValueError("Spectra product covariance-relevant config does not match the assembly config.")
    specs = default_spectrum_specs()
    ell = np.asarray(result["ell"], dtype=np.float64)
    n_per = ell.size
    n_data = n_per * len(specs)
    cov = np.zeros((n_data, n_data), dtype=np.float64)
    slices = {spec.name: (i * n_per, (i + 1) * n_per) for i, spec in enumerate(specs)}
    covariance_blocks: Dict[Tuple[str, str], np.ndarray] = {}
    input_cls: Dict[Tuple[str, ...], np.ndarray] = {}
    expected_blocks = {
        (str(block["spec_i"]), str(block["spec_j"]))
        for group in manifest["groups"]
        for block in group["blocks"]
    }
    if len(expected_blocks) != len(specs) * (len(specs) + 1) // 2:
        raise ValueError("Manifest contains missing or duplicate upper-triangle covariance blocks.")

    for group in manifest["groups"]:
        shard = block_shard_path(config, group)
        if not shard.exists():
            raise FileNotFoundError(f"Missing covariance shard for group {group['index']}: {shard}")
        compatible, reason = _covariance_shard_compatibility(
            shard,
            group,
            manifest,
            config,
            map_product_id,
            _group_mask_digest_from_metadata(group, map_metadata),
        )
        if not compatible:
            raise ValueError(f"Covariance shard {shard} is incompatible ({reason}).")
        with h5py.File(shard, "r") as h5:
            for key, values in _read_input_cls_group(h5).items():
                if key in input_cls and not np.array_equal(input_cls[key], values):
                    raise ValueError(f"Conflicting repeated covariance input spectrum {key!r} across shards.")
                input_cls[key] = values
            for name in h5["covariance_blocks"]:
                ds = h5[f"covariance_blocks/{name}"]
                name_i = str(ds.attrs["spectrum_i"])
                name_j = str(ds.attrs["spectrum_j"])
                block = ds[:]
                key = (name_i, name_j)
                if key not in expected_blocks:
                    raise ValueError(f"Unexpected covariance block {key!r} in {shard}.")
                if key in covariance_blocks:
                    raise ValueError(f"Duplicate covariance block {key!r} across shards.")
                if block.shape != (n_per, n_per) or not np.all(np.isfinite(block)):
                    raise ValueError(f"Covariance block {key!r} has invalid shape or non-finite values.")
                covariance_blocks[(name_i, name_j)] = block
                si = slice(*slices[name_i])
                sj = slice(*slices[name_j])
                cov[si, sj] = block
                if name_i != name_j:
                    cov[sj, si] = block.T

    if set(covariance_blocks) != expected_blocks:
        missing = sorted(expected_blocks - set(covariance_blocks))
        raise ValueError(f"Assembled covariance is incomplete; missing {len(missing)} block(s): {missing[:5]}")
    if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1.0e-8, atol=1.0e-20):
        raise ValueError("Assembled covariance is non-finite or non-symmetric.")
    if np.any(np.diag(cov) <= 0.0):
        raise ValueError("Assembled covariance has a non-positive diagonal; refusing to save clipped errors.")

    for spec in specs:
        name = spec.name
        start, stop = slices[name]
        block = cov[start:stop, start:stop]
        result["spectra"][name]["cov"] = block
        result["spectra"][name]["err"] = np.sqrt(np.diag(block))

    packed = pack_joint_data_vector(
        specs,
        result["spectra"],
        config,
        np.asarray(result["ell_left"]),
        np.asarray(result["ell_right"]),
    )
    result["schema"] = measurement_schema_for_config(config)
    result["covariance_blocks"] = covariance_blocks
    result["input_cls_for_covariance"] = input_cls
    result["covariance_workspace_keys"] = [group["key"] for group in manifest["groups"]]
    # The spectra intermediate deliberately records compute_covariance=False.  The
    # assembled product is a different contract: it contains the complete covariance
    # and must carry the authoritative production configuration rather than inheriting
    # that execution-only spectra flag.
    result["config"] = config.to_dict()
    result["config"]["compute_covariance"] = True
    result["joint"] = {
        "spectrum_names": [spec.name for spec in specs],
        "ell": ell,
        "data_vector": packed["data_vector"],
        "data_vector_raw": packed["data_vector_raw"],
        "data_vector_valid": packed["data_vector_valid"],
        "data_vector_weighted_poisson_subtracted": packed[
            "data_vector_weighted_poisson_subtracted"
        ],
        "data_vector_raw_weighted_poisson_subtracted": packed[
            "data_vector_raw_weighted_poisson_subtracted"
        ],
        "galaxy_auto_weighted_poisson_template": packed[
            "galaxy_auto_weighted_poisson_template"
        ],
        "spectrum_validity": packed["spectrum_validity"],
        "cov": cov,
        "corr": _corr_from_cov(cov),
        "slices": slices,
        "diagnostics": covariance_diagnostics(cov, compute_eig=not args.skip_cov_eig),
    }
    save_measurement_product(output, result, map_metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote assembled measurement {output}", flush=True)


def run_validate(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    path = Path(args.measurement_path).resolve() if args.measurement_path else config.default_measurement_path
    compatible, reason = _existing_product_matches_config(
        path, measurement_schema_for_config(config), config
    )
    if not compatible:
        raise ValueError(f"Measurement product {path} is incompatible ({reason}).")
    with h5py.File(path, "r") as h5:
        validate_measurement_product_identity(h5)
        if str(h5.attrs.get("covariance_estimator_version", "")) != COVARIANCE_ESTIMATOR_VERSION:
            raise ValueError("Measurement covariance estimator version is missing or stale.")
        cov = h5["joint/cov"][:]
        data = h5["joint/data_vector"][:]
        raw = h5["joint/data_vector_raw"][:] if "joint/data_vector_raw" in h5 else data.copy()
        valid = (
            h5["joint/data_vector_valid"][:].astype(bool)
            if "joint/data_vector_valid" in h5
            else np.ones(data.size, dtype=bool)
        )
        names = _read_string_dataset(h5["joint/spectrum_names"])
        diagnostics = json.loads(str(h5["joint"].attrs.get("diagnostics_json", "{}")))
        mean_convention = str(h5.attrs["desi_galaxy_auto_mean_convention"])
        galaxy_auto_views = validate_galaxy_auto_views(h5, require=True)
    expected = 46 * int(config.n_bins)
    if cov.shape != (expected, expected):
        raise ValueError(f"Covariance shape {cov.shape} does not match expected {(expected, expected)}.")
    if data.shape != (expected,):
        raise ValueError(f"Data vector shape {data.shape} does not match expected {(expected,)}.")
    if raw.shape != (expected,) or valid.shape != (expected,):
        raise ValueError("Raw data vector and validity mask must match the archive data-vector shape.")
    if not np.array_equal(data[valid], raw[valid]) or not np.all(data[~valid] == 0.0):
        raise ValueError("Packed data vector does not obey its validity mask and raw archive vector.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Data vector contains non-finite values.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance contains non-finite values.")
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-20):
        raise ValueError("Covariance is not symmetric.")
    diag = np.diag(cov)
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        raise ValueError("Covariance diagonal is not strictly positive and finite.")
    if len(names) != 46:
        raise ValueError(f"Expected 46 spectra, found {len(names)}.")
    sigma = np.sqrt(diag)
    corr = cov / np.outer(sigma, sigma)
    corr = 0.5 * (corr + corr.T)
    corr_eig = np.linalg.eigvalsh(corr)
    cov_eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    if not np.all(np.isfinite(corr_eig)) or not np.all(np.isfinite(cov_eig)):
        raise ValueError("Covariance/correlation eigenvalues contain non-finite values.")
    corr_threshold = float(args.corr_eigen_threshold)
    rank = int(np.sum(corr_eig > corr_threshold))
    if rank <= 0:
        raise ValueError(f"Correlation eigencut threshold {corr_threshold:g} retains zero modes.")
    if float(np.min(corr_eig)) < -1.0e-6:
        raise ValueError(f"Correlation matrix has a strongly negative eigenvalue: {np.min(corr_eig):.6e}.")
    active_cov = cov[np.ix_(valid, valid)]
    active_data = data[valid]
    active_sigma = np.sqrt(np.diag(active_cov))
    active_corr = active_cov / np.outer(active_sigma, active_sigma)
    active_corr = 0.5 * (active_corr + active_corr.T)
    active_corr_eig = np.linalg.eigvalsh(active_corr)
    active_cov_eig = np.linalg.eigvalsh(0.5 * (active_cov + active_cov.T))
    if not np.all(np.isfinite(active_data)) or not np.all(np.isfinite(active_cov)):
        raise ValueError("Active data vector or active covariance submatrix is non-finite.")
    if float(np.min(active_corr_eig)) < -1.0e-6:
        raise ValueError(
            f"Active correlation matrix has a strongly negative eigenvalue: {np.min(active_corr_eig):.6e}."
        )
    active_rank = int(np.sum(active_corr_eig > corr_threshold))
    expected_invalid = 28 if config.kappa_cmb_lmax is not None else 0
    if int(np.count_nonzero(~valid)) != expected_invalid:
        raise ValueError(
            f"Validity mask has {np.count_nonzero(~valid)} placeholders, expected {expected_invalid}."
        )
    report = {
        "measurement_path": str(path),
        "n_spectra": len(names),
        "n_bins": int(config.n_bins),
        "desi_galaxy_auto_mean_convention": mean_convention,
        "desi_galaxy_auto_views_contract_version": (
            DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION
        ),
        "desi_galaxy_auto_primary_hmc_view": DESI_GALAXY_AUTO_PRIMARY_VIEW,
        "desi_galaxy_auto_subtracted_view": DESI_GALAXY_AUTO_SUBTRACTED_VIEW,
        "desi_galaxy_auto_views": galaxy_auto_views,
        "data_vector_size": int(data.size),
        "archive_data_vector_size": int(data.size),
        "active_data_vector_size": int(active_data.size),
        "zero_placeholder_count": int(np.count_nonzero(~valid)),
        "covariance_shape": list(cov.shape),
        "data_finite": bool(np.all(np.isfinite(data))),
        "covariance_finite": bool(np.all(np.isfinite(cov))),
        "covariance_symmetric": bool(np.allclose(cov, cov.T, rtol=1e-8, atol=1e-20)),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "cov_eigen_min": float(np.min(cov_eig)),
        "cov_eigen_max": float(np.max(cov_eig)),
        "corr_eigen_min": float(np.min(corr_eig)),
        "corr_eigen_max": float(np.max(corr_eig)),
        "corr_eigen_threshold": corr_threshold,
        "corr_eigencut_rank": rank,
        "corr_eigencut_dropped_modes": int(corr_eig.size - rank),
        "active_covariance_shape": list(active_cov.shape),
        "active_cov_eigen_min": float(np.min(active_cov_eig)),
        "active_cov_eigen_max": float(np.max(active_cov_eig)),
        "active_corr_eigen_min": float(np.min(active_corr_eig)),
        "active_corr_eigen_max": float(np.max(active_corr_eig)),
        "active_corr_eigencut_rank": active_rank,
        "active_corr_eigencut_dropped_modes": int(active_corr_eig.size - active_rank),
        "hdf5_diagnostics": diagnostics,
        "submission_runtime_source_sha256": os.environ.get("XDESI_RUNTIME_SOURCE_SHA256", ""),
    }
    report_path = path.with_name(f"measurement_validation_{config.product_tag}.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[{utc_now()}] Validation passed for {path}: shape={cov.shape}, "
        f"diag=[{diag.min():.3e}, {diag.max():.3e}], "
        f"corr_eig=[{corr_eig.min():.3e}, {corr_eig.max():.3e}], "
        f"archive_rank@{corr_threshold:g}={rank}/{corr_eig.size}, "
        f"active_rank={active_rank}/{active_corr_eig.size}, report={report_path}",
        flush=True,
    )


def _parse_ksz_ylim(value: object) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if str(value).strip().lower() in {"auto", "none", ""}:
        return None
    parts = str(value).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Expected two values for --plot-ksz-ylim, got {value!r}.")
    return float(parts[0]), float(parts[1])


def run_plot_measurement_dell(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    path = Path(args.measurement_path).resolve() if args.measurement_path else config.default_measurement_path
    output_dir = Path(args.plot_dir).resolve() if args.plot_dir else config.output_root / "plots"
    pdf = Path(args.pdf_out).resolve() if args.pdf_out else output_dir / f"measurement_dell_{config.product_tag}.pdf"
    ell_max = None if args.plot_ell_max is not None and float(args.plot_ell_max) <= 0.0 else args.plot_ell_max
    ksz_ylim = _parse_ksz_ylim(args.plot_ksz_ylim)
    xscale = "log" if str(config.binning).lower() == "log" else "linear"
    compatible, reason = _existing_product_matches_config(
        path, measurement_schema_for_config(config), config
    )
    if not compatible:
        raise ValueError(f"Measurement product {path} is incompatible ({reason}).")

    import godmax_multiprobe_theory_utils as gmt

    measurement = gmt.load_measurement_data(path)
    outputs = gmt.plot_measurement_dell(
        measurement,
        output_dir,
        pdf_path=pdf,
        filename_prefix=f"measurement_dell_{config.product_tag}",
        ell_max=ell_max,
        ksz_ylim=ksz_ylim,
        ksz_scale=float(args.plot_ksz_scale),
        xscale=xscale,
    )
    summary = {
        "measurement_h5": str(path),
        "pdf": str(pdf),
        "pngs": [str(p) for p in outputs],
        "ell_max": ell_max,
        "ksz_ylim": ksz_ylim,
        "ksz_scale": float(args.plot_ksz_scale),
        "xscale": xscale,
        "transfer_null_from": measurement.transfer_null_from,
    }
    summary_path = output_dir / f"measurement_dell_{config.product_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{utc_now()}] Wrote measurement D_ell plot {pdf}", flush=True)
    print(f"[{utc_now()}] Wrote measurement D_ell plot summary {summary_path}", flush=True)


def run_plot_measurement_cl_dell(args: argparse.Namespace) -> None:
    """Save complete C_ell and D_ell views from one validated HDF5 product."""

    config = config_from_args(args)
    path = Path(args.measurement_path).resolve() if args.measurement_path else config.default_measurement_path
    output_dir = Path(args.plot_dir).resolve() if args.plot_dir else config.output_root / "plots"
    ell_max = None if args.plot_ell_max is not None and float(args.plot_ell_max) <= 0.0 else args.plot_ell_max
    ksz_ylim = _parse_ksz_ylim(args.plot_ksz_ylim)
    xscale = "log" if str(config.binning).lower() == "log" else "linear"
    compatible, reason = _existing_product_matches_config(
        path, measurement_schema_for_config(config), config
    )
    if not compatible:
        raise ValueError(f"Measurement product {path} is incompatible ({reason}).")

    import godmax_multiprobe_theory_utils as gmt

    measurement = gmt.load_measurement_data(path)
    cl_pdf = output_dir / f"measurement_cl_{config.product_tag}.pdf"
    dell_pdf = output_dir / f"measurement_dell_{config.product_tag}.pdf"
    cl_outputs = gmt.plot_measurement_cl(
        measurement,
        output_dir,
        pdf_path=cl_pdf,
        filename_prefix=f"measurement_cl_{config.product_tag}",
        ell_max=ell_max,
        ksz_scale=float(args.plot_ksz_scale),
        xscale=xscale,
    )
    dell_outputs = gmt.plot_measurement_dell(
        measurement,
        output_dir,
        pdf_path=dell_pdf,
        filename_prefix=f"measurement_dell_{config.product_tag}",
        ell_max=ell_max,
        ksz_ylim=ksz_ylim,
        ksz_scale=float(args.plot_ksz_scale),
        xscale=xscale,
    )
    summary = {
        "measurement_h5": str(path),
        "n_spectra": len(measurement.names),
        "cl_pdf": str(cl_pdf),
        "cl_pngs": [str(p) for p in cl_outputs],
        "dell_pdf": str(dell_pdf),
        "dell_pngs": [str(p) for p in dell_outputs],
        "ell_max": ell_max,
        "dell_ksz_ylim": ksz_ylim,
        "ksz_scale": float(args.plot_ksz_scale),
        "xscale": xscale,
        "transfer_null_from": measurement.transfer_null_from,
        "cl_ksz_sign_convention": "raw_C_ell_piT",
        "dell_ksz_sign_convention": "minus_D_ell_piT_paper_display",
        "galaxy_auto_view": measurement.galaxy_auto_view,
        "error_source": "sqrt_diagonal_of_saved_joint_covariance",
        "submission_runtime_source_sha256": os.environ.get("XDESI_RUNTIME_SOURCE_SHA256", ""),
    }
    summary_path = output_dir / f"measurement_cl_dell_{config.product_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{utc_now()}] Wrote C_ell and D_ell plots for {len(measurement.names)} spectra: "
        f"{cl_pdf}, {dell_pdf}",
        flush=True,
    )
    print(f"[{utc_now()}] Wrote measurement plot summary {summary_path}", flush=True)


def run_finalize(args: argparse.Namespace) -> None:
    """Assemble, validate, attest HMC inputs, and plot in one small allocation."""

    config = config_from_args(args)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    manifest = load_covariance_manifest(manifest_file)
    plan_file = (
        Path(args.plan_path).resolve()
        if args.plan_path
        else covariance_work_plan_path(config)
    )
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_covariance_work_plan_frozen_inputs(plan, manifest, config)
    output = (
        Path(args.measurement_out).resolve()
        if args.measurement_out
        else config.default_measurement_path
    )
    if output.exists() and not args.force:
        compatible, reason = _existing_product_matches_config(
            output,
            measurement_schema_for_config(config),
            config,
        )
        if not compatible:
            raise FileExistsError(
                f"Existing assembled measurement {output} is incompatible ({reason}); "
                "refusing to replace it implicitly."
            )
        with h5py.File(output, "r") as h5:
            validate_measurement_product_identity(h5)
            validate_galaxy_auto_views(h5, require=True)
            if "joint/cov" not in h5:
                raise ValueError("Existing assembled measurement has no full covariance.")
        print(f"[{utc_now()}] Reusing compatible assembled measurement {output}", flush=True)
    else:
        run_assemble(args)

    args.measurement_path = str(output)
    run_validate(args)

    import godmax_multiprobe_theory_utils as gmt

    total = gmt.load_measurement_data(output, galaxy_auto_view="total")
    subtracted = gmt.load_measurement_data(
        output,
        galaxy_auto_view="weighted_poisson_subtracted",
    )
    expected_archive = 46 * int(config.n_bins)
    expected_invalid = 28 if config.kappa_cmb_lmax is not None else 0
    expected_active = expected_archive - expected_invalid
    if len(total.names) != 46 or total.archive_data_vector_size != expected_archive:
        raise ValueError("HMC readiness check found a non-canonical spectrum/archive layout.")
    if total.data_vector.shape != (expected_active,) or total.covariance.shape != (
        expected_active,
        expected_active,
    ):
        raise ValueError("HMC primary view has the wrong active vector/covariance shape.")
    if not np.array_equal(total.covariance, subtracted.covariance):
        raise ValueError("Galaxy-auto total and subtracted HMC views do not share one covariance.")
    if not np.array_equal(total.archive_indices, subtracted.archive_indices):
        raise ValueError("Galaxy-auto HMC views do not share the same active archive indices.")
    if not np.all(np.isfinite(total.data_vector)) or not np.all(np.isfinite(total.covariance)):
        raise ValueError("HMC primary vector or covariance contains non-finite values.")
    if total.galaxy_auto_view != DESI_GALAXY_AUTO_PRIMARY_VIEW:
        raise ValueError("HMC loader did not select the total galaxy-auto view by default.")

    readiness = {
        "created_utc": utc_now(),
        "measurement_path": str(output),
        "measurement_sha256": _sha256_file(output),
        "primary_hmc_view": DESI_GALAXY_AUTO_PRIMARY_VIEW,
        "alternate_view": DESI_GALAXY_AUTO_SUBTRACTED_VIEW,
        "n_spectra": len(total.names),
        "archive_data_vector_size": int(total.archive_data_vector_size),
        "active_data_vector_size": int(total.data_vector.size),
        "invalid_kappa_placeholders": int(expected_invalid),
        "active_covariance_shape": list(total.covariance.shape),
        "active_total_data_vector_sha256": _array_digest(total.data_vector),
        "active_subtracted_data_vector_sha256": _array_digest(subtracted.data_vector),
        "active_covariance_sha256": _array_digest(total.covariance),
        "active_archive_indices_sha256": _array_digest(total.archive_indices),
        "covariance_views_array_equal": True,
        "shot_noise_likelihood_rule": (
            "Fit the primary total C_ell^gg+SN vector with clustering theory plus a free "
            "amplitude times the saved already-decoupled weighted-Poisson template."
        ),
    }
    readiness_path = output.with_name(f"hmc_input_readiness_{config.product_tag}.json")
    readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[{utc_now()}] Wrote HMC-input readiness attestation {readiness_path}", flush=True)

    run_plot_measurement_cl_dell(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        add_common_cli_args(subparser)

    p = sub.add_parser("prepare")
    add_common(p)
    p.add_argument("--maps-out", default=None)
    p.set_defaults(func=run_prepare)

    p = sub.add_parser("spectra")
    add_common(p)
    p.add_argument("--maps-path", default=None)
    p.add_argument("--spectra-out", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--patch-shear-only", action="store_true",
                   help="Deprecated safety trap. Pipeline v2 requires a full spectra recompute and "
                        "will reject this option.")
    p.set_defaults(func=run_spectra)

    p = sub.add_parser("make-cov-manifest")
    add_common(p)
    p.add_argument("--manifest-out", default=None)
    p.set_defaults(func=run_make_cov_manifest)

    p = sub.add_parser("make-cov-work-plan")
    add_common(p)
    p.add_argument("--maps-path", default=None)
    p.add_argument("--spectra-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--plan-out", default=None)
    p.add_argument("--groups-per-bundle", type=int, default=8)
    p.set_defaults(func=run_make_cov_work_plan)

    p = sub.add_parser("show-cov-work-bundle")
    add_common(p)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--plan-path", default=None)
    p.add_argument("--batch-id", type=int, default=None)
    p.set_defaults(func=run_show_cov_work_bundle)

    p = sub.add_parser("cov-key")
    add_common(p)
    p.add_argument("--maps-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--cov-class", choices=["all", "scalar", "spin2"], default="all")
    p.add_argument("--heartbeat-interval", type=float, default=120.0)
    p.add_argument("--no-cov-workspace-cache", action="store_true",
                   help="Do not read/write the on-disk covariance-workspace cache (rebuild every time).")
    p.set_defaults(func=run_cov_key)

    p = sub.add_parser("cov-batch")
    add_common(p)
    p.add_argument("--maps-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--batch-id", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--parallel-groups", type=int, default=1)
    p.add_argument("--omp-threads-per-group", type=int, default=1)
    p.add_argument("--cov-class", choices=["all", "scalar", "spin2"], default="all")
    p.add_argument("--heartbeat-interval", type=float, default=120.0)
    p.add_argument("--no-cov-workspace-cache", action="store_true",
                   help="Do not read/write the on-disk covariance-workspace cache (rebuild every time).")
    p.set_defaults(func=run_cov_batch)

    p = sub.add_parser("assemble")
    add_common(p)
    p.add_argument("--spectra-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--measurement-out", default=None)
    p.add_argument("--skip-cov-eig", action="store_true")
    p.set_defaults(func=run_assemble)

    p = sub.add_parser("validate")
    add_common(p)
    p.add_argument("--measurement-path", default=None)
    p.add_argument("--corr-eigen-threshold", type=float, default=1.0e-8)
    p.set_defaults(func=run_validate)

    p = sub.add_parser("plot-measurement-dell")
    add_common(p)
    p.add_argument("--measurement-path", default=None)
    p.add_argument("--plot-dir", default=None)
    p.add_argument("--pdf-out", default=None)
    p.add_argument("--plot-ell-max", type=float, default=0.0)
    p.add_argument("--plot-ksz-ylim", default="auto")
    p.add_argument("--plot-ksz-scale", type=float, default=1.0)
    p.set_defaults(func=run_plot_measurement_dell)

    p = sub.add_parser("plot-measurement-cl-dell")
    add_common(p)
    p.add_argument("--measurement-path", default=None)
    p.add_argument("--plot-dir", default=None)
    p.add_argument("--plot-ell-max", type=float, default=0.0)
    p.add_argument("--plot-ksz-ylim", default="auto")
    p.add_argument("--plot-ksz-scale", type=float, default=1.0)
    p.set_defaults(func=run_plot_measurement_cl_dell)

    p = sub.add_parser("finalize")
    add_common(p)
    p.add_argument("--spectra-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--plan-path", default=None)
    p.add_argument("--measurement-out", default=None)
    p.add_argument("--measurement-path", default=None)
    p.add_argument("--plot-dir", default=None)
    p.add_argument("--plot-ell-max", type=float, default=0.0)
    p.add_argument("--plot-ksz-ylim", default="auto")
    p.add_argument("--plot-ksz-scale", type=float, default=1.0)
    p.add_argument("--corr-eigen-threshold", type=float, default=1.0e-8)
    p.set_defaults(func=run_finalize, skip_cov_eig=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
