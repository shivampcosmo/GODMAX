#!/usr/bin/env python3
"""Fail-closed validator for the frozen three-probe inference manifest."""

from __future__ import annotations

# --- keep imports working from a theme subfolder: common/ holds the
# --- modules shared by more than one stage.
import pathlib as _pl, sys as _sys
_ROOT = _pl.Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "common"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np
import yaml

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_three_probe_inference_manifest as contract
from three_probe_inference_contract import load_training_contract


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(manifest_path: pathlib.Path) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    with manifest_path.open() as handle:
        manifest = yaml.safe_load(handle)
    _require(manifest["schema_version"] == "godmax.sbi.three_probe_inference_manifest.v1", "Manifest schema mismatch")
    _require(manifest["analysis"]["vector_size"] == 42, "Inference vector must have 42 entries")
    _require(manifest["analysis"]["probes"] == list(contract.SPECTRA), "Probe ordering mismatch")
    _require(manifest["analysis"]["n_bands_per_probe"] == 14, "Expected 14 bands per probe")
    _require(manifest["analysis"]["band_edges"][-1] == 2010, "Partial final band entered inference")
    _require(manifest["analysis"]["partial_final_band_policy"] == "diagnostic_only_excluded_from_inference", "Partial-band policy mismatch")
    try:
        contract.validate_seed_namespaces(manifest["seed_namespaces"])
    except ValueError as error:
        raise ValueError(str(error)) from error

    observation_path = manifest_path.parents[3] / manifest["observation"]["path"]
    _require(observation_path.is_file(), f"Observation product missing: {observation_path}")
    for source in manifest["observation"]["source_files"].values():
        path = manifest_path.parents[3] / source["path"]
        _require(path.is_file(), f"Frozen source missing: {path}")
        _require(contract.sha256_file(path) == source["sha256"], f"Source hash mismatch: {path}")
    for source in manifest["implementation_source_files"].values():
        path = manifest_path.parents[3] / source["path"]
        _require(path.is_file(), f"Implementation source missing: {path}")
        _require(contract.sha256_file(path) == source["sha256"], f"Implementation source hash mismatch: {path}")

    with h5py.File(observation_path, "r") as handle:
        vector = np.asarray(handle["data_vector"], dtype=np.float64)
        covariance = np.asarray(handle["covariance"], dtype=np.float64)
        cholesky = np.asarray(handle["cholesky"], dtype=np.float64)
        window = np.asarray(handle["window"], dtype=np.float64)
        pixel_window_g = np.asarray(handle["pixel_window_g"], dtype=np.float64)
        profile_smoothing_bell = np.asarray(handle["profile_smoothing_bell"], dtype=np.float64)
        edges = np.asarray(handle["band_edges"], dtype=np.int64)
        _require(str(handle.attrs["vector_order"]) == contract.VECTOR_ORDER, "Observation vector-order mismatch")
        _require(str(handle.attrs["manifest_sha256"]) == contract.canonical_json_sha256(manifest), "Observation manifest digest mismatch")
    _require(vector.shape == (42,) and np.all(np.isfinite(vector)), "Observation vector is invalid")
    _require(contract.sha256_array(vector) == manifest["observation"]["vector_sha256"], "Observation vector hash mismatch")
    _require(covariance.shape == (42, 42) and np.array_equal(covariance, covariance.T), "Covariance is invalid")
    _require(np.linalg.matrix_rank(covariance) == 42, "Covariance rank is not 42")
    _require(
        cholesky.shape == (42, 42)
        and contract.cholesky_correlation_reconstruction_error(covariance, cholesky) <= 1.0e-13,
        "Cholesky replay mismatch",
    )
    _require(contract.sha256_array(covariance) == manifest["covariance"]["covariance_sha256"], "Covariance hash mismatch")
    _require(contract.sha256_array(cholesky) == manifest["covariance"]["cholesky_sha256"], "Cholesky hash mismatch")
    _require(window.shape == (14, 2049) and edges.shape == (15,), "Window or band-edge shape mismatch")
    _require(pixel_window_g.shape == (2049,) and profile_smoothing_bell.shape == (2049,), "Transfer shape mismatch")
    _require(contract.sha256_array(pixel_window_g) == manifest["covariance"]["pixel_window_g_sha256"], "Galaxy pixel-window hash mismatch")
    _require(contract.sha256_array(profile_smoothing_bell) == manifest["covariance"]["profile_smoothing_bell_sha256"], "Profile Bell hash mismatch")

    inputs = contract.load_frozen_inputs(observation_index=0)
    _require(np.array_equal(vector, inputs["vector"]), "Observation does not replay frozen realization 000")
    _require(np.array_equal(covariance, inputs["covariance"]), "Observation covariance differs from frozen contract")
    _require(np.array_equal(cholesky, inputs["cholesky"]), "Observation Cholesky differs from frozen contract")
    _require(np.array_equal(window, inputs["window"]), "Observation windows differ from frozen contract")
    _require(np.array_equal(pixel_window_g, inputs["pixel_window_g"]), "Observation galaxy pixel window differs from frozen contract")
    _require(np.array_equal(profile_smoothing_bell, inputs["profile_smoothing_bell"]), "Observation profile Bell differs from frozen map")
    training = load_training_contract(manifest_path.with_name("inference_contract.yaml"))
    _require(np.array_equal(training.data_vector, vector), "Training contract data vector differs from audit replay")
    _require(np.array_equal(training.covariance, covariance), "Training contract covariance differs from audit replay")
    _require(np.array_equal(training.cholesky, cholesky), "Training contract Cholesky differs from audit replay")
    _require(np.array_equal(training.window, window), "Training contract window differs from audit replay")
    _require(np.array_equal(training.pixel_window_g, pixel_window_g), "Training contract galaxy pixel window differs from audit replay")
    _require(np.array_equal(training.profile_smoothing_bell, profile_smoothing_bell), "Training contract profile Bell differs from audit replay")
    return {
        "manifest": str(manifest_path),
        "observation": str(observation_path),
        "vector_size": int(vector.size),
        "covariance_rank": int(np.linalg.matrix_rank(covariance)),
        "manifest_sha256": contract.canonical_json_sha256(manifest),
        "training_contract_sha256": training.contract_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
