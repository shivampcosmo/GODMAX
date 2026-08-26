#!/usr/bin/env python3
"""Shared, fail-closed estimator context for mock SBI on pasted Abacus maps.

Every mock-SBI product must be measured with the *same* mask, NaMaster workspace,
fixed galaxy alm and band definition that produced the frozen inference
observation.  This module is the single place that loads them, verifies their
hashes, and turns maps into the 42-vector.  Nothing here rebuilds a workspace,
a mask, or a covariance: rebuilding is how mock SBI would silently stop being
comparable to the theory runs.

Two facts this module relies on, both verified (kb.sbi.mock-sbi-pasted-response-plan
section 1.3 and 1.4):

*   The galaxy field is frozen across the five sampled gas parameters, so the
    estimator is linear in the y/tau/kappa alm and

        x(theta, seed) = mu_paste(theta) + nu(seed)

    holds to machine precision with ``nu`` independent of ``theta``.
*   The frozen contract covariance is reproduced by y/tau/kappa noise draws
    alone, so it is used unchanged.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

NOISE_ROOT = REPO_ROOT / "data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048"
NOISE_CONTRACT_PATH = NOISE_ROOT / "noise_contract_tau_snrmatch_gkappa.h5"
FROZEN_MAP_PATH = REPO_ROOT / (
    "data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/"
    "abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5"
)
INFERENCE_CONTRACT_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/inference_contract.yaml"

# The production contract is the tau-SNR-matched revision (v2); the parent v1
# contract is accepted so the same loader can read the pre-revision products.
ACCEPTED_CONTRACT_SCHEMAS = (
    "sbi_three_probe_noise_contract_tau_effective_v2",
    "sbi_three_probe_noise_contract_v1",
)

NSIDE = 1024
LMAX = 2048
SPECTRA = ("gy", "gkappa", "gtau")
VECTOR_ORDER = "spectrum-major gy[14],gkappa[14],gtau[14]"
N_BAND = 14
VECTOR_SIZE = 42
PAIR_FIELDS = {"gy": ("g", "y"), "gkappa": ("g", "kappa"), "gtau": ("g", "tau")}
NOISE_FIELDS = ("y", "tau", "kappa")
MAP_DATASETS = {"y": "maps/map_ymap", "tau": "maps/map_tau", "kappa": "maps/map_kappa_cmb"}

INFERENCE_EDGES = np.asarray(
    [80, 101, 127, 160, 201, 253, 319, 401, 505, 636, 801, 1008, 1268, 1597, 2010],
    dtype=np.int64,
)

# Seeds already consumed by the frozen observation and the 12-realization diagnostic
# ensemble.  The noise bank must not reuse any of them: the observation's noise may
# never appear in a training or holdout stream.
OBSERVATION_BASE_SEED = 2026082000
OBSERVATION_N_REALIZATIONS = 12
OBSERVATION_FIELD_OFFSETS = {"y": 100_000, "tau": 200_000, "kappa": 300_000}

# Pre-registered PRNG namespaces (build_three_probe_inference_manifest.seed_namespaces).
NAMESPACE_TRAINING = (20260821, 201)
NAMESPACE_HOLDOUT = (20260821, 301)

_UINT32 = 1 << 32


def reserved_observation_seeds() -> frozenset[int]:
    """Every numpy seed the frozen observation and diagnostic ensemble consumed."""

    reserved: set[int] = set()
    for realization in range(OBSERVATION_N_REALIZATIONS):
        base = OBSERVATION_BASE_SEED + realization
        reserved.add(base)
        for offset in OBSERVATION_FIELD_OFFSETS.values():
            reserved.add(base + offset)
    return frozenset(reserved)


def sha256_file(path: pathlib.Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_mask(stored: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Regenerate the exact float64 mask and prove it is the stored one.

    ``build_contract`` computed every bandpower, window and covariance with the
    float64 mask returned by ``solve_common_c2_cap`` but stored the array as
    float32 (12.6M pixels).  Measuring with the lossily stored array costs
    7.6e-11 relative on the 42-vector, worth 4.8e-17 in whitened chi-square --
    negligible, but not float64 round-off, and it would mask a real regression.

    So the float64 mask is regenerated analytically and required to round
    *bitwise* to the stored array.  That makes the recovery a verified identity
    rather than a reconstruction, and measurement then reproduces the contract's
    own bandpowers to 8.6e-15.
    """

    from three_probe_noiseless_estimator import solve_common_c2_cap

    exact, metadata = solve_common_c2_cap(nside=NSIDE)
    exact = np.asarray(exact, dtype=np.float64)
    stored = np.asarray(stored, dtype=np.float64)
    if exact.shape != stored.shape:
        raise ValueError(f"Regenerated mask shape {exact.shape} != stored {stored.shape}")
    if not np.array_equal(exact.astype(np.float32), stored.astype(np.float32)):
        raise ValueError(
            "Regenerated float64 mask does not round to the stored float32 mask; "
            "the mask solver or its parameters have changed"
        )
    return exact, {
        "source": "solve_common_c2_cap(nside=1024) regenerated float64",
        "rounds_to_stored_float32": True,
        "max_abs_difference_vs_stored": float(np.max(np.abs(exact - stored))),
        "mean_mask_squared": float(np.mean(exact * exact)),
        "solver_metadata": {str(k): (float(v) if isinstance(v, (int, float)) else str(v))
                            for k, v in dict(metadata).items()},
    }


@dataclass(frozen=True)
class EstimatorContext:
    """The frozen estimator: mask, workspace, fixed galaxy alm, bands, covariance."""

    mask: np.ndarray
    mask_sum: float
    stored_mask_float32: np.ndarray
    mask_metadata: Mapping[str, Any]
    fixed_galaxy_alm: np.ndarray
    workspace: Any
    noise_cls: Mapping[str, np.ndarray]
    band_edges: np.ndarray
    effective_ell: np.ndarray
    window: np.ndarray
    pixel_window_g: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray
    signal_cls: Mapping[str, np.ndarray]
    fixed_bandpowers: Mapping[str, np.ndarray]
    theory_bandpowers: Mapping[str, np.ndarray]
    contract_sha256: str
    workspace_sha256: str
    mask_sha256: str
    stored_mask_sha256: str
    galaxy_alm_sha256: str

    def whiten(self, residual: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.cholesky, np.asarray(residual, dtype=np.float64))

    def chi2(self, residual: np.ndarray) -> float:
        w = self.whiten(residual)
        return float(w @ w)


def load_estimator_context(*, require_hashes: bool = True) -> EstimatorContext:
    """Load the frozen estimator from the noise contract, verifying every hash."""

    import pymaster as nmt

    contract_sha = sha256_file(NOISE_CONTRACT_PATH)
    with h5py.File(NOISE_CONTRACT_PATH, "r") as handle:
        schema = str(handle.attrs["schema_version"])
        if schema not in ACCEPTED_CONTRACT_SCHEMAS:
            raise ValueError(
                f"Unexpected noise-contract schema {schema!r}; "
                f"registered schemas are {ACCEPTED_CONTRACT_SCHEMAS}"
            )
        if str(handle.attrs["vector_order"]) != VECTOR_ORDER:
            raise ValueError("Noise contract vector ordering differs from the inference order")
        if int(handle.attrs["nside"]) != NSIDE or int(handle.attrs["lmax"]) != LMAX:
            raise ValueError("Noise contract nside/lmax differ from the inference resolution")
        stored_mask = np.asarray(handle["mask"], dtype=np.float64)
        galaxy_alm = np.asarray(handle["fixed_masked_alm/g"])
        noise_cls = {
            name: np.asarray(handle[f"noise_cls/{'y_effective' if name == 'y' else name}"], dtype=np.float64)
            for name in NOISE_FIELDS
        }
        expected_noise = json.loads(str(handle.attrs["noise_dataset_sha256_json"]))
        band_edges = np.asarray(handle["band_edges"], dtype=np.int64)
        effective_ell = np.asarray(handle["effective_ell"], dtype=np.float64)
        window = np.asarray(handle["window"], dtype=np.float64)
        pixel_window_g = np.asarray(handle["pixel_window_g"], dtype=np.float64)
        covariance = np.asarray(handle["hmc/covariance"], dtype=np.float64)
        cholesky = np.asarray(handle["hmc/cholesky"], dtype=np.float64)
        signal_cls = {key: np.asarray(handle[f"signal_cls/{key}"], dtype=np.float64)
                      for key in handle["signal_cls"]}
        fixed_bandpowers = {s: np.asarray(handle[f"fixed_bandpowers/{s}"], dtype=np.float64) for s in SPECTRA}
        theory_bandpowers = {s: np.asarray(handle[f"theory_bandpowers/{s}"], dtype=np.float64) for s in SPECTRA}
        workspace_path = pathlib.Path(str(handle.attrs["workspace_path"]))
        workspace_sha = str(handle.attrs["workspace_sha256"])
        map_path = pathlib.Path(str(handle.attrs["map_path"]))

    if not np.array_equal(band_edges, INFERENCE_EDGES):
        raise ValueError("Noise contract band edges are not the frozen 14 inference bands")
    for name, value in noise_cls.items():
        if sha256_array(value) != expected_noise[name]:
            raise ValueError(f"Noise curve hash mismatch for {name}")
    if require_hashes:
        if sha256_file(workspace_path) != workspace_sha:
            raise ValueError("NaMaster workspace hash mismatch")
        if map_path.resolve() != FROZEN_MAP_PATH.resolve():
            raise ValueError(f"Noise contract points at an unexpected map: {map_path}")
    if covariance.shape != (VECTOR_SIZE, VECTOR_SIZE) or not np.array_equal(covariance, covariance.T):
        raise ValueError("Frozen covariance is not a symmetric 42x42 matrix")
    if window.shape != (N_BAND, LMAX + 1):
        raise ValueError(f"Unexpected bandpower window shape {window.shape}")

    mask, mask_metadata = canonical_mask(stored_mask)
    workspace = nmt.NmtWorkspace.from_file(str(workspace_path))
    return EstimatorContext(
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
        mask_sha256=sha256_array(mask),
        stored_mask_sha256=sha256_array(stored_mask),
        galaxy_alm_sha256=sha256_array(galaxy_alm),
    )


def subtract_weighted_mean(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """The contract's mask-weighted centring (three_probe_noiseless_estimator)."""

    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    mean = float(np.sum(mask * values) / np.sum(mask))
    return values - mean, mean


def cross_bandpowers(field_alms: Mapping[str, np.ndarray], context: EstimatorContext) -> np.ndarray:
    """Decoupled 42-vector from y/tau/kappa alms crossed with the fixed galaxy alm."""

    import healpy as hp

    blocks = []
    for spectrum in SPECTRA:
        _, right = PAIR_FIELDS[spectrum]
        coupled = hp.alm2cl(context.fixed_galaxy_alm, field_alms[right], lmax=LMAX)
        blocks.append(np.asarray(context.workspace.decouple_cell(coupled[None, :]))[0])
    vector = np.concatenate(blocks)
    if vector.shape != (VECTOR_SIZE,) or not np.all(np.isfinite(vector)):
        raise RuntimeError("Cross bandpower assembly produced an invalid 42-vector")
    return vector


def masked_alms_from_maps(maps: Mapping[str, np.ndarray], context: EstimatorContext) -> dict[str, np.ndarray]:
    """Centre each field on the mask and transform, exactly as the contract does."""

    import healpy as hp

    if set(maps) != set(NOISE_FIELDS):
        raise ValueError(f"Expected exactly {NOISE_FIELDS}, got {tuple(maps)}")
    out: dict[str, np.ndarray] = {}
    for name in NOISE_FIELDS:
        values = np.asarray(maps[name], dtype=np.float64)
        if values.shape != context.mask.shape:
            raise ValueError(f"{name} map shape {values.shape} does not match the mask")
        centered, _ = subtract_weighted_mean(values, context.mask)
        out[name] = hp.map2alm(context.mask * centered, lmax=LMAX, iter=0)
    return out


def measure_map_vector(maps: Mapping[str, np.ndarray], context: EstimatorContext) -> np.ndarray:
    """Signal 42-vector for one set of y/tau/kappa maps (no noise added)."""

    return cross_bandpowers(masked_alms_from_maps(maps, context), context)


def read_paste_maps(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read the three theta-dependent maps from a combined paste product."""

    path = pathlib.Path(path)
    with h5py.File(path, "r") as handle:
        missing = [key for key in MAP_DATASETS.values() if key not in handle]
        if missing:
            raise KeyError(f"{path} is missing dataset(s) {missing}")
        return {name: np.asarray(handle[dataset], dtype=np.float64)
                for name, dataset in MAP_DATASETS.items()}


def measure_paste_file(path: pathlib.Path, context: EstimatorContext) -> np.ndarray:
    return measure_map_vector(read_paste_maps(path), context)


def synalm_seeded(cl: np.ndarray, seed: int) -> np.ndarray:
    """``three_probe_noise_contract._synalm_seeded``, reproduced exactly."""

    import healpy as hp

    seed = int(seed)
    if not 0 <= seed < _UINT32:
        raise ValueError(f"numpy legacy seed out of range: {seed}")
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return hp.synalm(np.asarray(cl, dtype=np.float64), lmax=LMAX, new=True)
    finally:
        np.random.set_state(state)


def noise_vector(field_seeds: Mapping[str, int], context: EstimatorContext, *,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """One theta-independent noise 42-vector from field-level harmonic draws.

    This is the only permitted noise route.  Drawing ``L @ epsilon`` in bandpower
    space is forbidden for this experiment: it bypasses the mask coupling and the
    estimator, and it cannot be validated against the map products.
    """

    import healpy as hp

    if set(field_seeds) != set(NOISE_FIELDS):
        raise ValueError(f"Expected seeds for exactly {NOISE_FIELDS}, got {tuple(field_seeds)}")
    # ``mask`` exists only so the frozen observation can be reproduced bitwise:
    # ``three_probe_noise_contract.realize`` read the lossily stored float32 mask,
    # while ``build_contract`` used the float64 one for the signal alms.  Production
    # noise draws use the canonical float64 mask.
    weight = context.mask if mask is None else np.asarray(mask, dtype=np.float64)
    alms: dict[str, np.ndarray] = {}
    for name in NOISE_FIELDS:
        alm = synalm_seeded(context.noise_cls[name], field_seeds[name])
        noise_map = hp.alm2map(alm, nside=NSIDE, lmax=LMAX)
        centered, _ = subtract_weighted_mean(noise_map, weight)
        alms[name] = hp.map2alm(weight * centered, lmax=LMAX, iter=0)
    return cross_bandpowers(alms, context)


def noise_bank_seeds(count: int, namespace: Sequence[int]) -> list[dict[str, int]]:
    """Deterministic, globally unique per-draw field seeds for a noise bank.

    Uniqueness across *all* fields and draws is required, not merely within a
    field.  ``healpy.synalm`` reseeds numpy's legacy generator, so two draws that
    share a seed produce perfectly correlated deviates: the same seed reused for
    the same field duplicates a realization, and reused across two fields
    manufactures a spurious cross-correlation between them.  At 3 x 2048 draws
    from a 32-bit space a birthday collision is a ~0.4% event, and the
    ``(20260821, 201)`` namespace does in fact collide once (value 2823752942 at
    flat positions 1592 and 3128, both kappa).

    So rather than accept collisions, over-generate and take the first
    ``3 * count`` values that are distinct and not reserved by the frozen
    observation.  Deterministic, reproducible, and collision-free by
    construction.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    needed = 3 * int(count)
    reserved = reserved_observation_seeds()
    sequence = np.random.SeedSequence(tuple(int(v) for v in namespace))
    # 4x headroom: the expected number of rejections is O(needed^2 / 2^32) ~ 0.005,
    # so this never iterates more than once in practice, but the loop is honest.
    pool_size = 4 * needed
    accepted: list[int] = []
    seen: set[int] = set()
    rejected = 0
    while len(accepted) < needed:
        raw = sequence.generate_state(pool_size, dtype=np.uint32)
        for value in raw.tolist():
            if value in seen or value in reserved:
                rejected += 1
                continue
            seen.add(value)
            accepted.append(value)
            if len(accepted) == needed:
                break
        if len(accepted) < needed:
            pool_size *= 2

    seeds: list[dict[str, int]] = []
    for index in range(int(count)):
        seeds.append({name: int(accepted[3 * index + offset])
                      for offset, name in enumerate(NOISE_FIELDS)})
    flat = [value for entry in seeds for value in entry.values()]
    if len(set(flat)) != len(flat):
        raise AssertionError("noise_bank_seeds produced a duplicate after rejection")
    if reserved.intersection(flat):
        raise AssertionError("noise_bank_seeds produced a reserved observation seed")
    return seeds


def load_inference_observation() -> tuple[np.ndarray, dict[str, Any]]:
    """The frozen 42-vector observation the theory runs used, with its provenance."""

    import yaml

    with INFERENCE_CONTRACT_PATH.open() as handle:
        contract = yaml.safe_load(handle)
    observation = contract["observation"]
    if str(observation["vector_order"]) != VECTOR_ORDER:
        raise ValueError("Inference contract vector ordering differs")
    path = REPO_ROOT / str(observation["path"])
    with h5py.File(path, "r") as handle:
        vector = np.asarray(handle["data_vector"], dtype=np.float64)
    if vector.shape != (VECTOR_SIZE,):
        raise ValueError(f"Observation has shape {vector.shape}, expected (42,)")
    if sha256_array(vector) != str(observation["vector_sha256"]):
        raise ValueError("Observation vector hash does not match the inference contract")
    return vector, {
        "path": str(observation["path"]),
        "vector_sha256": str(observation["vector_sha256"]),
        "contract_sha256": sha256_file(INFERENCE_CONTRACT_PATH),
    }
