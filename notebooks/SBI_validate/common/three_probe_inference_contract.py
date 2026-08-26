"""Read the oracle-free common input for all three-probe posterior runners.

This module intentionally accepts only ``inference_contract.yaml``.  The separate
audit manifest may contain a truth point for post-run diagnostics and is never a
valid sampler input.
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/inference_contract.yaml"
# A second, explicitly-registered contract whose observation is the forward
# model's OWN prediction at a chosen parameter point, with the production
# covariance, window and Cholesky reused byte-for-byte.  Its purpose is to make
# HMC-versus-SBI agreement decidable: on the production contract the observation
# is a pasted-map measurement that the analytic model cannot reproduce, so the
# chi-square floor is ~161 for 42-5 = 37 and any disagreement is entangled with
# model misspecification.  Here chi2 at the generating point is exactly zero, so
# a disagreement can only be the inference machinery.
#
# Identity is pinned by ONE hash -- the contract file's own -- and the contract
# then declares the hash of every array it admits, so the chain
# pinned sha -> contract -> array hashes -> arrays is closed.  The generating
# parameter point is deliberately NOT in the contract (nor readable through it):
# it lives in a sibling file that no sampler reads, so these inputs stay
# oracle-free exactly like the production ones.
SELFCONSISTENT_CONTRACT_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/inference_contract_selfconsistent.yaml"
SELFCONSISTENT_CONTRACT_SHA256 = "036e8f373c326509d0bf736eba7bca0c7adaa58633d76d3f5dc31c981a5c635a"

# A third registered contract, for the mock-SBI campaign.  Its observation is the
# NOISELESS measurement of the archived pasted maps at the reference parameter
# point -- the pasted analogue of the self-consistent theory observation above.
# The pasted simulator reproduces it exactly, so chi2 at the generating point is
# zero for the mock forward model just as it is for the analytic one, and the
# three posteriors (theory HMC, theory SBI, mock SBI) become comparable: each is
# conditioned on an observation its own forward model can reach.
#
# The production contract's observation is the same paste PLUS one noise draw.
# That draw displaces the posterior from the generating point by about one sigma
# in a random direction, which would show up as a mock-versus-theory offset that
# is the noise realization and nothing else.  The covariance-side arrays are the
# production ones byte for byte, enforced below.
MOCK_CONTRACT_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/inference_contract_mock.yaml"
MOCK_CONTRACT_SHA256 = "99ee9bf7c6e150787134fe769a1e2114de3a4d358e44e1aed9b9e757616c1930"
SCHEMA_VERSION = "godmax.sbi.three_probe_training_contract.v1"
FORBIDDEN_ORACLE_TOKENS = ("truth", "fiducial", "hmc")
VECTOR_ORDER = "spectrum-major gy[14],gkappa[14],gtau[14]"
EXPECTED_SAMPLED_PARAMETERS = (
    {"name": "theta_ej_0", "latex": r"\theta_{\rm ej,0}", "prior": {"kind": "uniform", "low": 0.5, "high": 8.0}},
    {"name": "alpha_nt", "latex": r"\alpha_{\rm nt}", "prior": {"kind": "uniform", "low": 0.0, "high": 0.5}},
    {"name": "mu_beta", "latex": r"\mu_\beta", "prior": {"kind": "uniform", "low": 0.005, "high": 1.5}},
    {"name": "theta_co_0", "latex": r"\theta_{\rm co,0}", "prior": {"kind": "uniform", "low": 0.001, "high": 0.5}},
    {"name": "nu_theta_ej_M", "latex": r"\nu^M_{\theta_{\rm ej}}", "prior": {"kind": "uniform", "low": -1.0, "high": 1.0}},
)
EXPECTED_FIXED_PARAMETERS = {
    "log10_Mstar0_theta_ej": 16.0,
    "nu_theta_ej_z": 0.0,
    "log10_Mc0": 14.83,
    "delta_rhogas": 7.0,
    "gamma_rhogas": 2.0,
}
EXPECTED_SEED_NAMESPACES = {
    "theory_sbi": {"kind": "SeedSequence", "entropy": [20260821, 101]},
    "network_initialization": {"kind": "SeedSequence", "entropy": [20260821, 401]},
    "policy": "all namespace entropy tuples must be distinct; observation seeds are excluded from every training and holdout stream",
}
EXPECTED_ARRAY_HASHES = {
    "data_vector": "a02d13698cc7f57c88e749f173aabfbf5d3ff98789fbf9f8f6282b6d50b037ad",
    "covariance": "e2211c44a196ec750effd3d652291110652a7686da1fc0e93f4c68628ee3fda6",
    "cholesky": "eaeeea732e8d241bc717cf50241c7cb0100f9ae8392dcd4bc204b3e7f637cd25",
    "window": "f3ec2c4a8e93ac9ecc1683cf94e041974cc847897b71e1bc06010c4da357d18c",
    "pixel_window_g": "ced14b134a9c7ed78ca820809ed23ff0366639f6c9003ce1af502fd363cb1243",
    "profile_smoothing_bell": "44b98a684a4a8c865f77c428d854737951c8ab2a8137c4e81f109c750c8fdfd0",
}
EXPECTED_SOURCE_PATHS = {
    "signal_map": REPO_ROOT / "data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5",
    "noise_contract": REPO_ROOT / "data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/noise_contract_tau_snrmatch_gkappa.h5",
    "noisy_ensemble": REPO_ROOT / "data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/noisy_ensemble_tau_snrmatch_gkappa.h5",
    "parent_realization": REPO_ROOT / "data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/realizations/noise_realization_000.h5",
    "tau_realization": REPO_ROOT / "data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/tau_snrmatch_gkappa_realizations/tau_noise_realization_000.h5",
}


@dataclass(frozen=True)
class ThreeProbeInferenceContract:
    """Immutable numeric inputs and prior metadata allowed to a posterior runner."""

    contract_path: pathlib.Path
    contract_sha256: str
    data_vector: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    window: np.ndarray
    pixel_window_g: np.ndarray
    profile_smoothing_bell: np.ndarray
    band_edges: np.ndarray
    effective_ell: np.ndarray
    sampled_parameters: tuple[Mapping[str, Any], ...]
    fixed_parameters: Mapping[str, Any]
    seed_namespaces: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_oracle_free(value: Any, path: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if any(token in key_text for token in FORBIDDEN_ORACLE_TOKENS):
                raise ValueError(f"Forbidden oracle field at {path}.{key}")
            _assert_oracle_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_oracle_free(child, f"{path}[{index}]")
    elif isinstance(value, str) and any(token in value.casefold() for token in FORBIDDEN_ORACLE_TOKENS):
        raise ValueError(f"Forbidden oracle value at {path}")


def _sha256_file(path: pathlib.Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    import hashlib

    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _cholesky_reconstruction_error(covariance: np.ndarray, cholesky: np.ndarray) -> float:
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    normalized_factor = cholesky / scale[:, None]
    return float(np.max(np.abs(normalized_factor @ normalized_factor.T - correlation)))


def _resolve_observation(path_value: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(path_value)
    _require(not relative.is_absolute() and ".." not in relative.parts, "Observation path must be repository-relative")
    observation = (REPO_ROOT / relative).resolve()
    _require(observation.is_relative_to(REPO_ROOT), "Observation path escapes the repository")
    _require(observation.is_file(), f"Observation product missing: {observation}")
    return observation


def load_training_contract(contract_path: pathlib.Path) -> ThreeProbeInferenceContract:
    """Load and verify the exact 42-vector inference input without audit metadata."""

    contract_path = contract_path.resolve()
    is_production = contract_path == DEFAULT_CONTRACT_PATH.resolve()
    is_selfconsistent = contract_path == SELFCONSISTENT_CONTRACT_PATH.resolve()
    is_mock = contract_path == MOCK_CONTRACT_PATH.resolve()
    _require(is_production or is_selfconsistent or is_mock,
             "Only a registered inference contract is a valid sampler input")
    if is_selfconsistent:
        _require(SELFCONSISTENT_CONTRACT_SHA256 is not None,
                 "The self-consistent contract has no pinned identity in this module; "
                 "run build_three_probe_selfconsistent_observation.py, which stamps it")
        _require(_sha256_file(contract_path) == SELFCONSISTENT_CONTRACT_SHA256,
                 "Self-consistent contract file hash does not match its pinned identity")
    if is_mock:
        _require(MOCK_CONTRACT_SHA256 is not None,
                 "The mock contract has no pinned identity in this module; "
                 "run build_mock_sbi_observation.py --stamp, which stamps it")
        _require(_sha256_file(contract_path) == MOCK_CONTRACT_SHA256,
                 "Mock contract file hash does not match its pinned identity")
    with contract_path.open() as handle:
        payload = yaml.safe_load(handle)
    _require(isinstance(payload, Mapping), "Training contract must be a mapping")
    _assert_oracle_free(payload)
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Training-contract schema mismatch")

    analysis = payload["analysis"]
    covariance_meta = payload["covariance"]
    observation_meta = payload["observation"]
    _require(analysis["vector_size"] == 42, "Inference vector must have 42 entries")
    _require(analysis["probes"] == ["gy", "gkappa", "gtau"], "Probe order mismatch")
    _require(analysis["n_bands_per_probe"] == 14, "Expected 14 bands per probe")
    _require(analysis["partial_final_band_policy"] == "diagnostic_only_excluded_from_inference", "Partial band entered inference")
    _require(tuple(payload["parameters"]["sampled"]) == EXPECTED_SAMPLED_PARAMETERS, "Sampled-prior contract mismatch")
    _require(payload["parameters"]["fixed"] == EXPECTED_FIXED_PARAMETERS, "Fixed-parameter contract mismatch")
    _require(payload["seed_namespaces"] == EXPECTED_SEED_NAMESPACES, "Seed-namespace contract mismatch")
    declared_hashes = {
        "data_vector": observation_meta["vector_sha256"],
        "covariance": covariance_meta["covariance_sha256"],
        "cholesky": covariance_meta["cholesky_sha256"],
        "window": covariance_meta["window_sha256"],
        "pixel_window_g": covariance_meta["pixel_window_g_sha256"],
        "profile_smoothing_bell": covariance_meta["profile_smoothing_bell_sha256"],
    }
    if is_production:
        # The production contract's identity is hardcoded here as well as declared,
        # so its behaviour is unchanged by the registry.
        for name, expected in EXPECTED_ARRAY_HASHES.items():
            _require(declared_hashes[name] == expected, f"Pinned {name} identity mismatch")
        _require(set(payload["source_hashes"]) == set(EXPECTED_SOURCE_PATHS), "Frozen-source set mismatch")
        for name, path in EXPECTED_SOURCE_PATHS.items():
            _require(path.is_file() and _sha256_file(path) == payload["source_hashes"][name], f"Frozen-source hash mismatch: {name}")
    else:
        # Covariance-side arrays must be the production ones, byte for byte: the
        # two contracts differ in the observation and in nothing else.
        for name in ("covariance", "cholesky", "window", "pixel_window_g", "profile_smoothing_bell"):
            _require(declared_hashes[name] == EXPECTED_ARRAY_HASHES[name],
                     f"Self-consistent contract must reuse the production {name}")
        _require(declared_hashes["data_vector"] != EXPECTED_ARRAY_HASHES["data_vector"],
                 "A derived contract declares the production data vector; it is not distinct "
                 "from production and would silently duplicate that run")
        for name, digest in payload["source_hashes"].items():
            path = (REPO_ROOT / name) if (REPO_ROOT / name).exists() else None
            _require(path is not None and _sha256_file(path) == digest,
                     f"Frozen-source hash mismatch: {name}")

    observation_path = _resolve_observation(observation_meta["path"])
    with h5py.File(observation_path, "r") as handle:
        _require(str(handle.attrs["vector_order"]) == VECTOR_ORDER, "Observation vector-order mismatch")
        data_vector = np.asarray(handle["data_vector"], dtype=np.float64)
        covariance = np.asarray(handle["covariance"], dtype=np.float64)
        cholesky = np.asarray(handle["cholesky"], dtype=np.float64)
        window = np.asarray(handle["window"], dtype=np.float64)
        pixel_window_g = np.asarray(handle["pixel_window_g"], dtype=np.float64)
        profile_smoothing_bell = np.asarray(handle["profile_smoothing_bell"], dtype=np.float64)
        band_edges = np.asarray(handle["band_edges"], dtype=np.int64)
        effective_ell = np.asarray(handle["effective_ell"], dtype=np.float64)

    _require(data_vector.shape == (42,) and np.all(np.isfinite(data_vector)), "Invalid data vector")
    _require(covariance.shape == (42, 42) and np.array_equal(covariance, covariance.T), "Invalid covariance")
    _require(cholesky.shape == (42, 42), "Invalid Cholesky factor")
    _require(window.shape == (14, 2049) and pixel_window_g.shape == (2049,) and profile_smoothing_bell.shape == (2049,) and band_edges.shape == (15,) and effective_ell.shape == (14,), "Invalid window inputs")
    _require(np.linalg.matrix_rank(covariance) == 42, "Covariance rank is not 42")
    _require(
        _cholesky_reconstruction_error(covariance, cholesky) <= 1.0e-13,
        "Cholesky replay mismatch",
    )
    for name, array in (("data_vector", data_vector), ("covariance", covariance),
                        ("cholesky", cholesky), ("window", window),
                        ("pixel_window_g", pixel_window_g),
                        ("profile_smoothing_bell", profile_smoothing_bell)):
        _require(_sha256_array(array) == declared_hashes[name], f"{name} hash mismatch")
    _require(np.array_equal(band_edges, np.asarray(analysis["band_edges"], dtype=np.int64)), "Band-edge mismatch")

    return ThreeProbeInferenceContract(
        contract_path=contract_path,
        contract_sha256=_canonical_json_sha256(payload),
        data_vector=data_vector,
        covariance=covariance,
        cholesky=cholesky,
        window=window,
        pixel_window_g=pixel_window_g,
        profile_smoothing_bell=profile_smoothing_bell,
        band_edges=band_edges,
        effective_ell=effective_ell,
        sampled_parameters=tuple(payload["parameters"]["sampled"]),
        fixed_parameters=payload["parameters"]["fixed"],
        seed_namespaces=payload["seed_namespaces"],
    )
