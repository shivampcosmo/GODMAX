#!/usr/bin/env python3
"""Freeze the common 42-vector observation for three-probe posterior inference.

This script intentionally consumes the revised tau realization together with the
parent gy/gkappa realization.  It writes only content-addressed products and does
not run inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
from typing import Any, Mapping

import h5py
import numpy as np
import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
NOISE_ROOT = REPO_ROOT / "data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048"
MAP_PATH = REPO_ROOT / (
    "data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/"
    "abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5"
)
CONTRACT_PATH = NOISE_ROOT / "noise_contract_tau_snrmatch_gkappa.h5"
ENSEMBLE_PATH = NOISE_ROOT / "noisy_ensemble_tau_snrmatch_gkappa.h5"
SPECTRA = ("gy", "gkappa", "gtau")
VECTOR_ORDER = "spectrum-major gy[14],gkappa[14],gtau[14]"
PARAMETERS = (
    ("theta_ej_0", r"\theta_{\rm ej,0}", 2.0, 0.5, 8.0),
    ("alpha_nt", r"\alpha_{\rm nt}", 0.18, 0.0, 0.5),
    ("mu_beta", r"\mu_\beta", 0.6, 0.005, 1.5),
    ("theta_co_0", r"\theta_{\rm co,0}", 0.05, 0.001, 0.5),
    ("nu_theta_ej_M", r"\nu^M_{\theta_{\rm ej}}", 0.0, -1.0, 1.0),
)
FIXED_PARAMETERS = {
    "log10_Mstar0_theta_ej": 16.0,
    "nu_theta_ej_z": 0.0,
    "log10_Mc0": 14.83,
    "delta_rhogas": 7.0,
    "gamma_rhogas": 2.0,
}


def seed_namespaces() -> dict[str, Any]:
    """The immutable PRNG domains for this experiment identity."""

    return {
        "observation": {"kind": "frozen_map_realization", "realization_index": 0},
        "theory_sbi": {"kind": "SeedSequence", "entropy": [20260821, 101]},
        "mock_sbi_training": {"kind": "SeedSequence", "entropy": [20260821, 201]},
        "mock_sbi_holdout": {"kind": "SeedSequence", "entropy": [20260821, 301]},
        "network_initialization": {"kind": "SeedSequence", "entropy": [20260821, 401]},
        "policy": "all namespace entropy tuples must be distinct; observation seeds are excluded from every training and holdout stream",
    }


def validate_seed_namespaces(value: Mapping[str, Any]) -> None:
    if dict(value) != seed_namespaces():
        raise ValueError("Seed namespaces differ from the pre-registered experiment contract")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relative_to_repo(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def source_worktree_identity() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()

    return {
        "git_head": run("git", "rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(("git", "diff", "--binary"), cwd=REPO_ROOT)
        ).hexdigest(),
    }


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")


def cholesky_correlation_reconstruction_error(covariance: np.ndarray, cholesky: np.ndarray) -> float:
    """Maximum reconstruction error after removing heterogeneous covariance units."""

    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    normalized_factor = cholesky / scale[:, None]
    return float(np.max(np.abs(normalized_factor @ normalized_factor.T - correlation)))


def load_frozen_inputs(observation_index: int) -> dict[str, Any]:
    if observation_index != 0:
        raise ValueError("The pre-registered observation is realization index 0")
    required = (MAP_PATH, CONTRACT_PATH, ENSEMBLE_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen input(s): " + ", ".join(missing))
    parent_realization = NOISE_ROOT / f"realizations/noise_realization_{observation_index:03d}.h5"
    tau_realization = NOISE_ROOT / "tau_snrmatch_gkappa_realizations" / (
        f"tau_noise_realization_{observation_index:03d}.h5"
    )
    if not parent_realization.is_file() or not tau_realization.is_file():
        raise FileNotFoundError("The selected parent or revised-tau realization is absent")

    with h5py.File(CONTRACT_PATH, "r") as handle:
        if str(handle.attrs["vector_order"]) != VECTOR_ORDER:
            raise ValueError("Frozen contract vector ordering mismatch")
        band_edges = np.asarray(handle["band_edges"], dtype=np.int64)
        effective_ell = np.asarray(handle["effective_ell"], dtype=np.float64)
        window = np.asarray(handle["window"], dtype=np.float64)
        pixel_window_g = np.asarray(handle["pixel_window_g"], dtype=np.float64)
        covariance = np.asarray(handle["hmc/covariance"], dtype=np.float64)
        cholesky = np.asarray(handle["hmc/cholesky"], dtype=np.float64)
        noise_hashes = json.loads(str(handle.attrs["noise_dataset_sha256_json"]))
        contract_attrs = {key: handle.attrs[key] for key in handle.attrs}
    _require_shape("band_edges", band_edges, (15,))
    _require_shape("effective_ell", effective_ell, (14,))
    _require_shape("window", window, (14, 2049))
    _require_shape("pixel_window_g", pixel_window_g, (2049,))
    _require_shape("covariance", covariance, (42, 42))
    _require_shape("cholesky", cholesky, (42, 42))
    if not np.array_equal(band_edges, np.asarray([80, 101, 127, 160, 201, 253, 319, 401, 505, 636, 801, 1008, 1268, 1597, 2010])):
        raise ValueError("Inference band edges are not the frozen 14 complete bands")
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("Frozen covariance is not exactly symmetric")
    if cholesky_correlation_reconstruction_error(covariance, cholesky) > 1.0e-13:
        raise ValueError("Stored Cholesky fails the normalized double-precision reconstruction check")
    if np.linalg.matrix_rank(covariance) != 42:
        raise ValueError("Frozen inference covariance is not full rank")

    with h5py.File(MAP_PATH, "r") as handle:
        smoothing_ell = np.asarray(handle["kernels/profile_smoothing_ell"], dtype=np.int64)
        saved_profile_smoothing_bell = np.asarray(handle["kernels/profile_smoothing_Bell"], dtype=np.float64)
        smoothing_sigma_rad = float(handle["kernels"].attrs["profile_smoothing_sigma_rad"])
    if not np.array_equal(smoothing_ell, np.arange(saved_profile_smoothing_bell.size, dtype=np.int64)):
        raise ValueError("Saved profile-smoothing Bell multipoles are not a contiguous grid")
    if saved_profile_smoothing_bell.shape != (1536,) or not np.isfinite(smoothing_sigma_rad):
        raise ValueError("Frozen map does not contain the expected ell<=1535 Bell anchor")
    profile_smoothing_bell = np.exp(-0.5 * (np.arange(2049, dtype=np.float64) * smoothing_sigma_rad) ** 2)
    np.testing.assert_allclose(
        profile_smoothing_bell[: saved_profile_smoothing_bell.size],
        saved_profile_smoothing_bell,
        rtol=2.0e-15,
        atol=0.0,
    )

    with h5py.File(parent_realization, "r") as handle:
        if int(handle.attrs["realization"]) != observation_index:
            raise ValueError("Parent realization index mismatch")
        parent_contract_sha = str(handle.attrs["contract_sha256"])
        parent_subseeds = json.loads(str(handle.attrs["field_subseeds_json"]))
        gy = np.asarray(handle["bandpowers/gy"], dtype=np.float64)
        gkappa = np.asarray(handle["bandpowers/gkappa"], dtype=np.float64)
    with h5py.File(tau_realization, "r") as handle:
        if int(handle.attrs["realization"]) != observation_index:
            raise ValueError("Revised tau realization index mismatch")
        if str(handle.attrs["contract_sha256"]) != sha256_file(CONTRACT_PATH):
            raise ValueError("Revised tau realization does not use the frozen contract")
        tau_subseed = int(handle.attrs["tau_subseed"])
        gtau = np.asarray(handle["bandpowers/gtau"], dtype=np.float64)
    vector = np.concatenate((gy, gkappa, gtau))
    _require_shape("observation vector", vector, (42,))

    with h5py.File(ENSEMBLE_PATH, "r") as handle:
        ensemble_vector = np.concatenate(
            [np.asarray(handle[f"draws/{name}"][observation_index], dtype=np.float64) for name in SPECTRA]
        )
        ensemble_covariance = np.asarray(handle["hmc_covariance"], dtype=np.float64)
        ensemble_summary = json.loads(str(handle.attrs["summary_json"]))
    if not np.array_equal(vector, ensemble_vector):
        raise ValueError("Assembled observation does not equal the selected ensemble row")
    if not np.array_equal(covariance, ensemble_covariance):
        raise ValueError("Ensemble and frozen-contract HMC covariance differ")

    return {
        "band_edges": band_edges,
        "effective_ell": effective_ell,
        "window": window,
        "pixel_window_g": pixel_window_g,
        "profile_smoothing_bell": profile_smoothing_bell,
        "covariance": covariance,
        "cholesky": cholesky,
        "vector": vector,
        "contract_attrs": contract_attrs,
        "noise_hashes": noise_hashes,
        "parent_contract_sha256": parent_contract_sha,
        "parent_subseeds": parent_subseeds,
        "tau_subseed": tau_subseed,
        "ensemble_summary": ensemble_summary,
        "source_paths": {
            "signal_map": MAP_PATH,
            "noise_contract": CONTRACT_PATH,
            "noisy_ensemble": ENSEMBLE_PATH,
            "parent_realization": parent_realization,
            "tau_realization": tau_realization,
        },
    }


def build_manifest_payload(inputs: Mapping[str, Any], observation_path: pathlib.Path) -> dict[str, Any]:
    source_paths: Mapping[str, pathlib.Path] = inputs["source_paths"]
    source_files = {
        name: {"path": relative_to_repo(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    implementation_files = {
        "manifest_builder": pathlib.Path(__file__).resolve(),
        "manifest_validator": pathlib.Path(__file__).with_name("validate_three_probe_inference_manifest.py"),
        "noise_contract_builder": pathlib.Path(__file__).with_name("three_probe_noise_contract.py"),
        "tau_noise_revision": pathlib.Path(__file__).with_name("rerun_three_probe_tau_noise.py"),
    }
    return {
        "schema_version": "godmax.sbi.three_probe_inference_manifest.v1",
        "observation": {
            "path": relative_to_repo(observation_path),
            "selection_rule": "lowest_valid_realization_index_predeclared",
            "realization_index": 0,
            "vector_order": VECTOR_ORDER,
            "vector_sha256": sha256_array(inputs["vector"]),
            "source_files": source_files,
            "field_subseeds": {
                "y": int(inputs["parent_subseeds"]["y"]),
                "kappa": int(inputs["parent_subseeds"]["kappa"]),
                "tau": int(inputs["tau_subseed"]),
            },
        },
        "analysis": {
            "probes": list(SPECTRA),
            "n_bands_per_probe": 14,
            "vector_size": 42,
            "band_edges": inputs["band_edges"].tolist(),
            "harmonic_lmax": 2048,
            "inference_ell_max": 2009,
            "partial_final_band_policy": "diagnostic_only_excluded_from_inference",
            "nside": int(inputs["contract_attrs"]["nside"]),
            "ordering": "RING",
        },
        "covariance": {
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "covariance_sha256": sha256_array(inputs["covariance"]),
            "cholesky_sha256": sha256_array(inputs["cholesky"]),
            "window_sha256": sha256_array(inputs["window"]),
            "pixel_window_g_sha256": sha256_array(inputs["pixel_window_g"]),
            "profile_smoothing_bell_sha256": sha256_array(inputs["profile_smoothing_bell"]),
            "rank": 42,
            "sample_covariance_policy": "diagnostic_only_never_used_for_inference",
            "noise_curve_sha256": inputs["noise_hashes"],
        },
        "implementation_source_files": {
            name: {"path": relative_to_repo(path), "sha256": sha256_file(path)}
            for name, path in implementation_files.items()
        },
        "parameters": {
            "sampled": [
                {"name": name, "latex": latex, "truth": truth, "prior": {"kind": "uniform", "low": low, "high": high}}
                for name, latex, truth, low, high in PARAMETERS
            ],
            "fixed": FIXED_PARAMETERS,
        },
        "seed_namespaces": seed_namespaces(),
        "worktree": source_worktree_identity(),
    }


def build_training_contract_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the oracle-free input permitted to posterior runners.

    The audit manifest deliberately retains truth values for post-run coverage checks.
    This projection is the only YAML contract a sampler may load.
    """

    observation = manifest["observation"]
    return {
        "schema_version": "godmax.sbi.three_probe_training_contract.v1",
        "observation": {
            "path": observation["path"],
            "vector_order": observation["vector_order"],
            "vector_sha256": observation["vector_sha256"],
        },
        "analysis": manifest["analysis"],
        "covariance": manifest["covariance"],
        "source_hashes": {
            name: source["sha256"] for name, source in observation["source_files"].items()
        },
        "parameters": {
            "sampled": [
                {key: value for key, value in parameter.items() if key != "truth"}
                for parameter in manifest["parameters"]["sampled"]
            ],
            "fixed": manifest["parameters"]["fixed"],
        },
        "seed_namespaces": {
            name: value
            for name, value in manifest["seed_namespaces"].items()
            if name in {"theory_sbi", "network_initialization", "policy"}
        },
    }


def atomic_yaml(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=True)
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def atomic_observation(path: pathlib.Path, inputs: Mapping[str, Any], manifest_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = "godmax.sbi.three_probe_observation.v1"
        handle.attrs["manifest_sha256"] = manifest_sha256
        handle.attrs["vector_order"] = VECTOR_ORDER
        handle.attrs["observation_realization_index"] = 0
        handle.attrs["source_file_sha256_json"] = json.dumps(
            {name: sha256_file(path) for name, path in inputs["source_paths"].items()}, sort_keys=True
        )
        handle.attrs["field_subseeds_json"] = json.dumps({
            "y": int(inputs["parent_subseeds"]["y"]),
            "kappa": int(inputs["parent_subseeds"]["kappa"]),
            "tau": int(inputs["tau_subseed"]),
        }, sort_keys=True)
        handle.create_dataset("data_vector", data=inputs["vector"])
        handle.create_dataset("covariance", data=inputs["covariance"])
        handle.create_dataset("cholesky", data=inputs["cholesky"])
        handle.create_dataset("band_edges", data=inputs["band_edges"])
        handle.create_dataset("effective_ell", data=inputs["effective_ell"])
        handle.create_dataset("window", data=inputs["window"])
        handle.create_dataset("pixel_window_g", data=inputs["pixel_window_g"])
        handle.create_dataset("profile_smoothing_bell", data=inputs["profile_smoothing_bell"])
    os.replace(temporary, path)


def build(output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    inputs = load_frozen_inputs(observation_index=0)
    output_dir = output_dir.resolve()
    observation_path = output_dir / "observation.h5"
    manifest_path = output_dir / "experiment_manifest.yaml"
    training_contract_path = output_dir / "inference_contract.yaml"
    payload = build_manifest_payload(inputs, observation_path)
    manifest_sha256 = canonical_json_sha256(payload)
    atomic_yaml(manifest_path, payload)
    atomic_yaml(training_contract_path, build_training_contract_payload(payload))
    atomic_observation(observation_path, inputs, manifest_sha256)
    return manifest_path, observation_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest, observation = build(args.output_dir)
    print(json.dumps({
        "manifest": str(manifest),
        "training_contract": str(manifest.with_name("inference_contract.yaml")),
        "observation": str(observation),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
