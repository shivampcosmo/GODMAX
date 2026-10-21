#!/usr/bin/env python3
"""Run one all-256 mock-SBI arm from the paired native gX/g2X master.

Eleven canonical probes are supported.  A single-statistic probe uses its
usual five-dimensional normalized score.  Every multi-statistic probe fits a
separate 14-dimensional whitening/score contract for each named component and
concatenates the resulting five-dimensional scores in ``component_order``.
The existing gX+g2X probes therefore remain 10D, while the all-field 2-point
and all-field 3-point probes are 15D.  Components are never added or jointly
whitened; their dependence is retained because SNLE sees the paired rows.

Old, new, or combined training cohorts may be selected.  ``native`` uses five
old and ten new replicas; ``first5`` is the matched-replica control.

The master contains a frozen cohort-stratified point split.  Held-out points are
excluded from both covariance/score fitting and network fitting.  SBI's own
validation fraction is retained only for early stopping inside the training
points.  Each network is sampled twice with independent MCMC seeds, so sampler
repeat stability and between-network stability are reported separately.  These
diagnostics are evidence, not a mathematical convergence guarantee.
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
from collections.abc import Mapping
from typing import Any

import numpy as np

import build_mock_sbi_g2x_pilot_contract as scorelib
import prepare_mock_sbi_all256_native as prep

POSTERIOR_SCHEMA = "godmax.mock_sbi.all256_native_snle_posterior.v1"
SEPARATE_SCORE_SCHEMA = (
    "godmax.mock_sbi.all256_native_snle_separate_score_concat10.v1"
)
SEPARATE_SCORE_15_SCHEMA = (
    "godmax.mock_sbi.all256_native_snle_separate_score_concat15.v1"
)
PARAMETER_NAMES = prep.PARAMETER_NAMES
PRIOR_LOW = scorelib.PRIOR_LOW
PRIOR_HIGH = scorelib.PRIOR_HIGH
TRAINING = {
    "training_batch_size": 512,
    "learning_rate": 5.0e-4,
    "validation_fraction": 0.10,
    "stop_after_epochs": 25,
    "max_num_epochs": 500,
}
GY_TRAINING = {
    "training_batch_size": 64,
    "learning_rate": 2.0e-4,
    "validation_fraction": 0.10,
    "stop_after_epochs": 100,
    "max_num_epochs": 2000,
}
MCMC = {
    "method": "slice_np_vectorized",
    "num_chains": 20,
    "warmup_steps": 250,
    "thin": 1,
    "independent_repeats": 2,
}
GY_MCMC = {
    "method": "slice_np_vectorized",
    "num_chains": 20,
    "warmup_steps": 1000,
    "thin": 1,
    "independent_repeats": 2,
}
DEFAULT_POSTERIOR_SAMPLES = 20_000
GY_DEFAULT_POSTERIOR_SAMPLES = 100_000
NAMESPACE = (20260913, 2901)


def _block(start: int) -> np.ndarray:
    return np.arange(start, start + 14, dtype=np.int64)


PROBE_SPECS: dict[str, dict[str, Any]] = {
    "gy": {"gx": _block(0), "g2x": None, "dimension": 14},
    "g2y": {"gx": None, "g2x": _block(0), "dimension": 14},
    "gy_g2y": {"gx": _block(0), "g2x": _block(0), "dimension": 28},
    "gkappa": {"gx": _block(14), "g2x": None, "dimension": 14},
    "g2kappa": {"gx": None, "g2x": _block(14), "dimension": 14},
    "gkappa_g2kappa": {"gx": _block(14), "g2x": _block(14), "dimension": 28},
    "gtau": {"gx": _block(28), "g2x": None, "dimension": 14},
    "g2tau": {"gx": None, "g2x": _block(28), "dimension": 14},
    "gtau_g2tau": {"gx": _block(28), "g2x": _block(28), "dimension": 28},
    # Append the new probes so Slurm indices 0--8 retain their established
    # meanings.  The stored 42-vectors are ordered y, kappa, tau.
    "gy_gkappa_gtau": {
        "gx": np.arange(42, dtype=np.int64),
        "g2x": None,
        "dimension": 42,
    },
    "g2y_g2kappa_g2tau": {
        "gx": None,
        "g2x": np.arange(42, dtype=np.int64),
        "dimension": 42,
    },
}
PROBE_CODES = {name: index + 1 for index, name in enumerate(PROBE_SPECS)}
JOINT_COMPONENTS = {
    "gy_g2y": ("gy", "g2y"),
    "gkappa_g2kappa": ("gkappa", "g2kappa"),
    "gtau_g2tau": ("gtau", "g2tau"),
    "gy_gkappa_gtau": ("gy", "gkappa", "gtau"),
    "g2y_g2kappa_g2tau": ("g2y", "g2kappa", "g2tau"),
}
COHORT_CODES = {"old": 1, "new": 2, "combined": 3}
REPLICA_POLICY_CODES = {"native": 1, "first5": 2}


def _atomic_savez(path: pathlib.Path, arrays: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _json_from_array(value: np.ndarray) -> dict[str, Any]:
    result = json.loads(str(np.asarray(value).item()))
    if not isinstance(result, dict):
        raise ValueError("manifest_json is not an object")
    return result


def _load_master(path: pathlib.Path, msc: Any) -> dict[str, Any]:
    path = path.expanduser().resolve()
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"master or sidecar is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    manifest = _json_from_array(arrays.pop("manifest_json"))
    if json.loads(sidecar.read_text()) != manifest:
        raise ValueError("master NPZ manifest and JSON sidecar differ")
    if manifest.get("schema_version") != prep.MASTER_SCHEMA:
        raise ValueError(f"unexpected master schema {manifest.get('schema_version')!r}")
    if int(manifest.get("n_points", -1)) != prep.N_POINTS or int(
        manifest.get("n_rows", -1)
    ) != prep.N_ROWS:
        raise ValueError("master does not declare 256 points and 1,920 rows")
    if tuple(manifest.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("master parameter order changed")
    if manifest.get("gx_vector_order") != prep.GX_VECTOR_ORDER or manifest.get(
        "g2x_vector_order"
    ) != prep.G2X_VECTOR_ORDER:
        raise ValueError("master vector ordering changed")

    expected_shapes = {
        "gx": (1920, 42), "g2x": (1920, 42), "gx_point_signal": (256, 42),
        "gx_old_noise": (640, 42),
        "gx_data_vector": (42,), "g2x_data_vector": (42,), "theta": (1920, 5),
        "u": (1920, 5), "log_q": (1920,),
        "point_theta": (256, 5), "point_u": (256, 5), "point_log_q": (256,),
        "point_index": (1920,), "replica": (1920,), "replica_count": (256,),
        "point_row_offsets": (257,), "cohort": (1920,), "point_cohort": (256,),
        "source_point_index": (256,), "bank_indices": (1920,),
        "noise_seeds": (1920, 3), "point_split": (256,),
        "theta_reference": (5,), "u_reference": (5,), "band_edges": (15,),
        "effective_ell": (14,),
    }
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            observed = None if name not in arrays else arrays[name].shape
            raise ValueError(f"master {name!r} shape {observed}; expected {shape}")
    inventory = manifest.get("arrays", {})
    if set(inventory) != set(arrays):
        raise ValueError("master array inventory differs from its NPZ arrays")
    for name, value in arrays.items():
        record = inventory[name]
        if list(value.shape) != list(record.get("shape", [])) or str(value.dtype) != str(
            record.get("dtype")
        ):
            raise ValueError(f"master array metadata changed for {name!r}")
        if msc.sha256_array(value) != str(record.get("sha256")):
            raise ValueError(f"master array hash changed for {name!r}")
        if value.dtype.kind in "fc" and name not in {"log_q", "point_log_q"} and not np.all(
            np.isfinite(value)
        ):
            raise ValueError(f"master array {name!r} contains non-finite values")

    dof = prep._validate_flat_rows(
        arrays["point_index"], arrays["replica"], arrays["replica_count"],
        arrays["point_row_offsets"],
    )
    if dof != prep.COVARIANCE_DOF:
        raise ValueError(f"master complete-data covariance df {dof} != {prep.COVARIANCE_DOF}")
    expected_counts = np.concatenate([np.full(128, 5), np.full(128, 10)])
    if not np.array_equal(arrays["replica_count"], expected_counts):
        raise ValueError("master replica-count vector is not 128x5 followed by 128x10")
    point_index = np.asarray(arrays["point_index"], dtype=np.int64)
    if not np.array_equal(arrays["theta"], arrays["point_theta"][point_index]) or not np.array_equal(
        arrays["u"], arrays["point_u"][point_index]
    ):
        raise ValueError("master row parameters do not match their point parameters")
    if not np.array_equal(
        arrays["log_q"], arrays["point_log_q"][point_index], equal_nan=True
    ):
        raise ValueError("master row log_q does not match point_log_q")
    prep._validate_theta_u(
        arrays["point_theta"], arrays["point_u"], "master 256-point design"
    )
    expected_cohort = np.asarray(["old"] * 128 + ["new"] * 128)
    if not np.array_equal(arrays["point_cohort"].astype("U"), expected_cohort) or not np.array_equal(
        arrays["cohort"].astype("U"), expected_cohort[point_index]
    ):
        raise ValueError("master cohort labels changed")
    split = np.asarray(arrays["point_split"], dtype=np.int8)
    if set(split.tolist()) != {0, 1} or np.sum(split[:128]) != 13 or np.sum(split[128:]) != 13:
        raise ValueError("master point split is not the frozen 115/13 split per cohort")
    if msc.sha256_array(split) != str(manifest["point_split"]["sha256"]):
        raise ValueError("master point-split hash changed")
    return {
        "path": path,
        "sidecar": sidecar,
        "file_sha256": msc.sha256_file(path),
        "manifest": manifest,
        "arrays": arrays,
    }


def _select_vector(gx: np.ndarray, g2x: np.ndarray, probe: str) -> np.ndarray:
    if probe not in PROBE_SPECS:
        raise ValueError(f"unknown probe {probe!r}")
    spec = PROBE_SPECS[probe]
    left = None if spec["gx"] is None else np.asarray(gx)[..., spec["gx"]]
    right = None if spec["g2x"] is None else np.asarray(g2x)[..., spec["g2x"]]
    if left is None:
        result = right
    elif right is None:
        result = left
    else:
        result = np.concatenate([left, right], axis=-1)
    assert result is not None
    if result.shape[-1] != int(spec["dimension"]):
        raise AssertionError(f"probe {probe} selection dimension changed")
    return np.asarray(result, dtype=np.float64)


def _policy_row_mask(arrays: Mapping[str, np.ndarray], policy: str) -> np.ndarray:
    if policy == "native":
        return np.ones(arrays["point_index"].shape, dtype=bool)
    if policy == "first5":
        return np.asarray(arrays["replica"], dtype=np.int64) < 5
    raise ValueError(f"unknown replica policy {policy!r}")


def _cohort_row_mask(arrays: Mapping[str, np.ndarray], cohort: str) -> np.ndarray:
    if cohort == "combined":
        return np.ones(arrays["point_index"].shape, dtype=bool)
    if cohort not in {"old", "new"}:
        raise ValueError(f"unknown cohort {cohort!r}")
    return arrays["cohort"].astype("U") == cohort


def _build_contract(
    arrays: Mapping[str, np.ndarray],
    probe: str,
    replica_policy: str,
) -> dict[str, Any]:
    """Fit one common score contract using only frozen-split training points."""

    point_split = np.asarray(arrays["point_split"], dtype=np.int8)
    point_index = np.asarray(arrays["point_index"], dtype=np.int64)
    training_points = np.flatnonzero(point_split == 0)
    row_mask = (point_split[point_index] == 0) & _policy_row_mask(arrays, replica_policy)
    selected = _select_vector(arrays["gx"], arrays["g2x"], probe)[row_mask]
    old_to_new = np.full(prep.N_POINTS, -1, dtype=np.int64)
    old_to_new[training_points] = np.arange(training_points.size, dtype=np.int64)
    local_point_index = old_to_new[point_index[row_mask]]
    if np.any(local_point_index < 0):
        raise AssertionError("held-out point entered the score contract")
    point_means, residuals, covariance, dof = scorelib._within_point_covariance(
        selected, local_point_index, training_points.size
    )
    cholesky = np.linalg.cholesky(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    response = scorelib._fit_affine_response(
        np.asarray(arrays["point_u"], dtype=np.float64)[training_points], point_means
    )
    intercept = np.asarray(response["intercept"], dtype=np.float64)
    jacobian = np.asarray(response["jacobian"], dtype=np.float64)
    reference = intercept + jacobian @ np.asarray(arrays["u_reference"], dtype=np.float64)
    score = scorelib._score_payload(
        covariance, jacobian, reference, np.arange(selected.shape[1], dtype=np.int64)
    )
    observation = _select_vector(
        arrays["gx_data_vector"], arrays["g2x_data_vector"], probe
    )
    whitened_observation = np.linalg.solve(cholesky, observation - reference)
    observed_score = np.asarray(score["operator"], dtype=np.float64) @ whitened_observation
    expected_dof = int(
        np.sum(
            np.minimum(
                np.asarray(arrays["replica_count"], dtype=np.int64)[training_points],
                5 if replica_policy == "first5" else 10,
            )
            - 1
        )
    )
    if replica_policy == "native":
        expected_dof = int(
            np.sum(np.asarray(arrays["replica_count"])[training_points] - 1)
        )
    if dof != expected_dof:
        raise AssertionError(f"contract covariance df {dof} != {expected_dof}")
    return {
        "row_mask": row_mask,
        "training_points": training_points,
        "point_means": point_means,
        "residuals": residuals,
        "covariance": covariance,
        "cholesky": cholesky,
        "eigenvalues": eigenvalues,
        "dof": dof,
        "intercept": intercept,
        "jacobian": jacobian,
        "reference": reference,
        "operator": np.asarray(score["operator"], dtype=np.float64),
        "observed_score": observed_score,
        "observation": observation,
        "response_rank": int(response["rank"]),
        "score_response_rank": int(score["response_rank"]),
    }


def _summarise(vectors: np.ndarray, contract: Mapping[str, Any], compression: str) -> np.ndarray:
    values = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    whitened = np.linalg.solve(
        np.asarray(contract["cholesky"], dtype=np.float64),
        (values - np.asarray(contract["reference"], dtype=np.float64)[None, :]).T,
    ).T
    if compression == "raw":
        return whitened
    if compression == "score":
        return (np.asarray(contract["operator"], dtype=np.float64) @ whitened.T).T
    raise ValueError(f"unknown compression {compression!r}")


def _components_for_probe(probe: str) -> tuple[str, ...]:
    """Return the independently normalized component order for one probe."""

    if probe not in PROBE_SPECS:
        raise ValueError(f"unknown probe {probe!r}")
    return JOINT_COMPONENTS.get(probe, (probe,))


def _build_component_contracts(
    arrays: Mapping[str, np.ndarray],
    probe: str,
    replica_policy: str,
) -> dict[str, dict[str, Any]]:
    """Build one 14D score contract per component of a joint probe."""

    components = _components_for_probe(probe)
    contracts = {
        component: _build_contract(arrays, component, replica_policy)
        for component in components
    }
    reference = contracts[components[0]]
    for component in components[1:]:
        candidate = contracts[component]
        if not np.array_equal(candidate["row_mask"], reference["row_mask"]):
            raise AssertionError("component score contracts use different rows")
        if not np.array_equal(
            candidate["training_points"], reference["training_points"]
        ):
            raise AssertionError("component score contracts use different points")
        if int(candidate["dof"]) != int(reference["dof"]):
            raise AssertionError("component score contracts have different covariance df")
    return contracts


def _summarise_probe(
    gx: np.ndarray,
    g2x: np.ndarray,
    probe: str,
    contracts: Mapping[str, Mapping[str, Any]],
    compression: str,
) -> np.ndarray:
    """Score each component separately, then concatenate without adding."""

    components = _components_for_probe(probe)
    pieces = []
    for component in components:
        vectors = _select_vector(gx, g2x, component)
        pieces.append(_summarise(vectors, contracts[component], compression))
    result = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
    expected = (
        5 * len(components)
        if compression == "score"
        else int(PROBE_SPECS[probe]["dimension"])
    )
    if result.ndim != 2 or result.shape[1] != expected:
        raise AssertionError(
            f"probe {probe} produced summary shape {result.shape}; expected (*,{expected})"
        )
    return np.asarray(result, dtype=np.float64)


def _raw_joint_covariance(
    contracts: Mapping[str, Mapping[str, Any]], probe: str
) -> np.ndarray | None:
    """Return the paired raw covariance for diagnostics, never for whitening."""

    if probe not in JOINT_COMPONENTS:
        return None
    components = JOINT_COMPONENTS[probe]
    residuals = np.concatenate(
        [np.asarray(contracts[name]["residuals"], dtype=np.float64) for name in components],
        axis=1,
    )
    dof = int(contracts[components[0]]["dof"])
    covariance = residuals.T @ residuals / float(dof)
    covariance = 0.5 * (covariance + covariance.T)
    raw_dimension = sum(
        int(np.asarray(contracts[name]["residuals"]).shape[1])
        for name in components
    )
    if covariance.shape != (raw_dimension, raw_dimension) or not np.all(
        np.isfinite(covariance)
    ):
        raise RuntimeError("raw joint diagnostic covariance is invalid")
    np.linalg.cholesky(covariance)
    return covariance


def _theta_from_u(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    values = np.asarray(u, dtype=np.float64)
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * ndtr(values)


def _settings_for_probe(
    probe: str, requested_samples: int | None
) -> tuple[dict[str, Any], dict[str, Any], int]:
    if probe == "gy":
        training = dict(GY_TRAINING)
        mcmc = dict(GY_MCMC)
        default_samples = GY_DEFAULT_POSTERIOR_SAMPLES
    else:
        training = dict(TRAINING)
        mcmc = dict(MCMC)
        default_samples = DEFAULT_POSTERIOR_SAMPLES
    samples = default_samples if requested_samples is None else requested_samples
    return training, mcmc, samples


def _seeded(
    probe: str,
    cohort: str,
    replica_policy: str,
    network_seed: int,
    stage: int,
    repeat: int = 0,
) -> int:
    return int(
        np.random.SeedSequence(
            tuple(NAMESPACE)
            + (
                PROBE_CODES[probe], COHORT_CODES[cohort],
                REPLICA_POLICY_CODES[replica_policy], int(network_seed),
                int(stage), int(repeat),
            )
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _estimator_log_prob(estimator: Any, x: Any, theta: Any) -> tuple[np.ndarray | None, str | None]:
    attempts = (
        lambda: estimator.log_prob(x, context=theta),
        lambda: estimator.log_prob(inputs=x, context=theta),
        lambda: estimator.log_prob(x, condition=theta),
        lambda: estimator.log_prob(inputs=x, condition=theta),
        lambda: estimator.log_prob(x, theta),
    )
    errors: list[str] = []
    for attempt in attempts:
        try:
            value = attempt()
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            result = np.asarray(value, dtype=np.float64).reshape(-1)
            if result.size != int(x.shape[0]) or not np.all(np.isfinite(result)):
                errors.append(f"invalid result shape/content {result.shape}")
                continue
            return result, None
        except (TypeError, ValueError, RuntimeError) as error:
            errors.append(f"{type(error).__name__}: {error}")
    return None, " | ".join(errors)


def _component_contract_record(
    contract: Mapping[str, Any], msc: Any
) -> dict[str, Any]:
    eigenvalues = np.asarray(contract["eigenvalues"], dtype=np.float64)
    return {
        "input_dimension": int(np.asarray(contract["covariance"]).shape[0]),
        "summary_dimension": int(np.asarray(contract["operator"]).shape[0]),
        "covariance_degrees_of_freedom": int(contract["dof"]),
        "covariance_repair": None,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "affine_response_rank": int(contract["response_rank"]),
        "whitened_response_rank": int(contract["score_response_rank"]),
        "covariance_sha256": msc.sha256_array(contract["covariance"]),
        "operator_sha256": msc.sha256_array(contract["operator"]),
        "reference_prediction_sha256": msc.sha256_array(contract["reference"]),
        "observation_sha256": msc.sha256_array(contract["observation"]),
    }


def _run_self_test() -> int:
    rng = np.random.default_rng(20260913)
    gx = rng.normal(size=(40, 42))
    g2x = rng.normal(size=(40, 42))
    for probe, spec in PROBE_SPECS.items():
        selected = _select_vector(gx, g2x, probe)
        assert selected.shape == (40, spec["dimension"])
        changed_gx = gx.copy()
        changed_g2x = g2x.copy()
        used_gx = np.asarray([], dtype=np.int64) if spec["gx"] is None else spec["gx"]
        used_g2x = np.asarray([], dtype=np.int64) if spec["g2x"] is None else spec["g2x"]
        changed_gx[:, np.setdiff1d(np.arange(42), used_gx)] += 1.0e12
        changed_g2x[:, np.setdiff1d(np.arange(42), used_g2x)] -= 1.0e12
        np.testing.assert_array_equal(
            selected, _select_vector(changed_gx, changed_g2x, probe)
        )
    identity_contract = {
        "cholesky": np.eye(14),
        "reference": np.zeros(14),
        "operator": np.eye(5, 14),
    }
    for joint, components in JOINT_COMPONENTS.items():
        contracts = {name: identity_contract for name in components}
        summary = _summarise_probe(gx, g2x, joint, contracts, "score")
        assert summary.shape == (40, 5 * len(components))
        for index, component in enumerate(components):
            component_summary = _summarise_probe(
                gx, g2x, component, contracts, "score"
            )
            np.testing.assert_array_equal(
                summary[:, 5 * index : 5 * (index + 1)], component_summary
            )
    seeds = {
        _seeded(probe, cohort, policy, network, stage, repeat)
        for probe in PROBE_SPECS
        for cohort in COHORT_CODES
        for policy in REPLICA_POLICY_CODES
        for network in range(4)
        for stage in (1, 2)
        for repeat in range(2)
    }
    assert len(seeds) == len(PROBE_SPECS) * 3 * 2 * 4 * 2 * 2
    u = np.zeros((3, 5))
    expected = np.repeat((0.5 * (PRIOR_LOW + PRIOR_HIGH))[None, :], 3, axis=0)
    np.testing.assert_allclose(_theta_from_u(u), expected)
    gy_training, gy_mcmc, gy_samples = _settings_for_probe("gy", None)
    assert gy_training == GY_TRAINING
    assert gy_mcmc == GY_MCMC
    assert gy_samples == GY_DEFAULT_POSTERIOR_SAMPLES
    standard_training, standard_mcmc, standard_samples = _settings_for_probe(
        "g2y", None
    )
    assert standard_training == TRAINING
    assert standard_mcmc == MCMC
    assert standard_samples == DEFAULT_POSTERIOR_SAMPLES
    print(
        "SELF-TEST PASS: eleven probes, separate component-score "
        "concatenation (10D/15D), component nulls, unique seeds, prior transform"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", type=pathlib.Path)
    parser.add_argument("--probe", choices=tuple(PROBE_SPECS))
    parser.add_argument("--cohort", choices=("old", "new", "combined"), default="combined")
    parser.add_argument("--training-replicas", choices=("native", "first5"), default="native")
    parser.add_argument("--contract-replicas", choices=("native", "first5"), default="native")
    parser.add_argument("--compression", choices=("score", "raw"), default="score")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=None,
        help=(
            "samples per network; defaults to 100000 for gy and 20000 for "
            "all other probes"
        ),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    for name in ("master", "probe", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required unless --self-test is used")
    if args.seeds != 4:
        parser.error("--seeds must remain 4 for the registered network ensemble")
    if args.probe in JOINT_COMPONENTS and args.compression != "score":
        joint_dimension = 5 * len(JOINT_COMPONENTS[args.probe])
        parser.error(
            "multi-statistic probes require --compression score: their summary "
            f"is the {joint_dimension}D concatenation of separately normalized "
            "5D component scores"
        )
    training_settings, mcmc_settings, posterior_samples = _settings_for_probe(
        args.probe, args.posterior_samples
    )
    if posterior_samples <= 0 or posterior_samples % mcmc_settings["independent_repeats"]:
        parser.error("--posterior-samples must be positive and divisible by two MCMC repeats")
    per_repeat = posterior_samples // mcmc_settings["independent_repeats"]
    if per_repeat % mcmc_settings["num_chains"]:
        parser.error("samples per MCMC repeat must be divisible by 20 chains")
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if output.exists() and not (args.overwrite or args.preflight):
        parser.error(f"output exists; pass --overwrite: {output}")

    import mock_sbi_common as msc

    started = time.time()
    print("[1/5] loading the paired heterogeneous-replica master", flush=True)
    master = _load_master(args.master, msc)
    arrays = master["arrays"]
    components = _components_for_probe(args.probe)
    contracts = _build_component_contracts(
        arrays, args.probe, args.contract_replicas
    )
    raw_joint_covariance = _raw_joint_covariance(contracts, args.probe)
    point_index = np.asarray(arrays["point_index"], dtype=np.int64)
    point_split = np.asarray(arrays["point_split"], dtype=np.int8)
    base_mask = _cohort_row_mask(arrays, args.cohort) & _policy_row_mask(
        arrays, args.training_replicas
    )
    train_mask = base_mask & (point_split[point_index] == 0)
    heldout_mask = base_mask & (point_split[point_index] == 1)
    if not np.any(train_mask) or not np.any(heldout_mask):
        raise ValueError("selected cohort/policy has no training or held-out rows")
    train_u = np.asarray(arrays["u"], dtype=np.float64)[train_mask]
    all_summary = _summarise_probe(
        arrays["gx"], arrays["g2x"], args.probe, contracts, args.compression
    )
    train_summary = all_summary[train_mask]
    heldout_u = np.asarray(arrays["u"], dtype=np.float64)[heldout_mask]
    heldout_summary = all_summary[heldout_mask]
    observed = _summarise_probe(
        arrays["gx_data_vector"], arrays["g2x_data_vector"],
        args.probe, contracts, args.compression,
    )[0]
    if args.compression == "score":
        expected_observed = np.concatenate(
            [np.asarray(contracts[name]["observed_score"]) for name in components]
        )
        if not np.allclose(observed, expected_observed, rtol=0.0, atol=1.0e-13):
            raise RuntimeError("observed component-score concatenation changed")
    train_points = np.unique(point_index[train_mask])
    heldout_points = np.unique(point_index[heldout_mask])
    if np.intersect1d(train_points, heldout_points).size:
        raise AssertionError("point-level training/held-out leakage detected")
    train_banks = set(np.asarray(arrays["bank_indices"])[train_mask].tolist())
    heldout_banks = set(np.asarray(arrays["bank_indices"])[heldout_mask].tolist())
    print(
        f"      {args.probe}, cohort={args.cohort}, training={args.training_replicas}, "
        f"contract={args.contract_replicas}"
    )
    print(
        f"      train {train_points.size} points/{train_mask.sum()} rows; held out "
        f"{heldout_points.size} points/{heldout_mask.sum()} rows"
    )
    contract_dof = {name: int(contracts[name]["dof"]) for name in components}
    print(f"      component order {components}; summary dimension {train_summary.shape[1]}")
    print(
        f"      component covariance df {contract_dof}; shared train/held-out "
        f"bank indices {len(train_banks & heldout_banks)}"
    )
    print(
        f"      training settings {training_settings}; MCMC settings "
        f"{mcmc_settings}; samples/network {posterior_samples}"
    )
    if args.preflight:
        print("PREFLIGHT PASS")
        print(f"      output to create: {output}")
        print("      no files were written")
        return 0

    print("[2/5] training four independently initialized SNLE likelihoods", flush=True)
    import torch
    from sbi.inference import SNLE
    from torch.distributions import Independent, Normal

    prior = Independent(Normal(torch.zeros(5), torch.ones(5)), 1)
    theta_torch = torch.as_tensor(train_u, dtype=torch.float32)
    summary_torch = torch.as_tensor(train_summary, dtype=torch.float32)
    heldout_theta_torch = torch.as_tensor(heldout_u, dtype=torch.float32)
    heldout_summary_torch = torch.as_tensor(heldout_summary, dtype=torch.float32)
    observed_torch = torch.as_tensor(observed, dtype=torch.float32)
    per_network: list[np.ndarray] = []
    network_records: list[dict[str, Any]] = []
    repeat_arrays: dict[str, np.ndarray] = {}
    heldout_log_prob_arrays: dict[str, np.ndarray] = {}

    for network in range(args.seeds):
        network_started = time.time()
        training_seed = _seeded(
            args.probe, args.cohort, args.training_replicas, network, 1
        )
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        inference = SNLE(
            prior=prior,
            density_estimator="nsf",
            device="cpu",
            show_progress_bars=False,
        )
        inference.append_simulations(theta_torch, summary_torch)
        estimator = inference.train(**training_settings, show_train_summary=True)
        with torch.no_grad():
            heldout_log_prob, heldout_error = _estimator_log_prob(
                estimator, heldout_summary_torch, heldout_theta_torch
            )
        if heldout_log_prob is None:
            raise RuntimeError(
                f"network {network}: could not evaluate the held-out likelihood; "
                f"refusing to write a successful result ({heldout_error})"
            )
        heldout_log_prob_arrays[f"heldout_log_prob_seed_{network}"] = heldout_log_prob
        posterior = inference.build_posterior(
            estimator,
            prior=prior,
            sample_with="mcmc",
            mcmc_method=mcmc_settings["method"],
            mcmc_parameters={
                "num_chains": mcmc_settings["num_chains"],
                "warmup_steps": mcmc_settings["warmup_steps"],
                "thin": mcmc_settings["thin"],
            },
        )
        repeats: list[np.ndarray] = []
        repeat_records: list[dict[str, Any]] = []
        for repeat in range(mcmc_settings["independent_repeats"]):
            sampling_seed = _seeded(
                args.probe, args.cohort, args.training_replicas,
                network, 2, repeat,
            )
            torch.manual_seed(sampling_seed)
            np.random.seed(sampling_seed)
            draws = np.asarray(
                posterior.sample(
                    (per_repeat,), x=observed_torch, show_progress_bars=False
                ),
                dtype=np.float64,
            )
            if draws.shape != (per_repeat, 5) or not np.all(np.isfinite(draws)):
                raise RuntimeError(
                    f"network {network} repeat {repeat} returned invalid draws {draws.shape}"
                )
            repeats.append(draws)
            repeat_arrays[f"u_seed_{network}_repeat_{repeat}"] = draws
            repeat_records.append(
                {
                    "repeat": repeat,
                    "sampling_seed": sampling_seed,
                    "samples": per_repeat,
                    "mean_u": draws.mean(axis=0).tolist(),
                    "sd_u": draws.std(axis=0, ddof=1).tolist(),
                }
            )
        network_draws = np.concatenate(repeats, axis=0)
        per_network.append(network_draws)
        repeat_means = np.asarray([record["mean_u"] for record in repeat_records])
        repeat_scale = network_draws.std(axis=0, ddof=1)
        repeat_shift = float(
            np.max(np.abs(repeat_means[0] - repeat_means[1]) / repeat_scale)
        )
        training_summary = getattr(inference, "summary", getattr(inference, "_summary", None))
        record = {
            "network_seed": network,
            "training_seed": training_seed,
            "seconds": time.time() - network_started,
            "mean_u": network_draws.mean(axis=0).tolist(),
            "sd_u": network_draws.std(axis=0, ddof=1).tolist(),
            "correlation_u": np.corrcoef(network_draws, rowvar=False).tolist(),
            "mcmc_repeats": repeat_records,
            "max_repeat_mean_shift_sigma": repeat_shift,
            "heldout_mean_log_prob": float(np.mean(heldout_log_prob)),
            "heldout_log_prob_error": None,
            "sbi_training_summary": _jsonable(training_summary),
        }
        network_records.append(record)
        print(
            f"      network {network}: mean {np.round(network_draws.mean(axis=0), 3)} "
            f"sd {np.round(network_draws.std(axis=0, ddof=1), 3)} "
            f"MCMC-repeat shift {repeat_shift:.3f} sigma "
            f"heldout mean logp {record['heldout_mean_log_prob']} "
            f"({record['seconds']:.0f}s)",
            flush=True,
        )

    print("[3/5] separating sampler-repeat and network stability", flush=True)
    pooled_u = np.concatenate(per_network, axis=0)
    pooled_theta = _theta_from_u(pooled_u)
    pooled_mean = pooled_u.mean(axis=0)
    pooled_sd = pooled_u.std(axis=0, ddof=1)
    network_means = np.asarray([record["mean_u"] for record in network_records])
    max_network_departure = float(
        np.max(np.abs(network_means - pooled_mean[None, :]) / pooled_sd[None, :])
    )
    max_repeat_shift = float(
        max(record["max_repeat_mean_shift_sigma"] for record in network_records)
    )
    print(f"      pooled mean_u {np.round(pooled_mean, 4)}")
    print(f"      pooled sd_u   {np.round(pooled_sd, 4)}")
    print(f"      max independent-MCMC-repeat mean shift {max_repeat_shift:.3f} pooled sigma")
    print(f"      max network-mean departure {max_network_departure:.3f} pooled sigma")

    print("[4/5] constructing reproducibility and held-out diagnostics", flush=True)
    component_records = {
        name: _component_contract_record(contracts[name], msc)
        for name in components
    }
    is_separate_joint = args.probe in JOINT_COMPONENTS
    if is_separate_joint:
        component_count = len(components)
        concatenated_dimension = 5 * component_count
        if component_count == 2:
            schema_version = SEPARATE_SCORE_SCHEMA
            analysis_status = "exploratory_two_snapshot_all256_separate_score_concat10"
        elif component_count == 3:
            schema_version = SEPARATE_SCORE_15_SCHEMA
            analysis_status = "exploratory_two_snapshot_all256_separate_score_concat15"
        else:
            raise AssertionError(
                f"unsupported joint component count {component_count}"
            )
        summary_construction = (
            "separately whitened and normalized 5D component scores, "
            "concatenated in component_order without addition"
        )
        score_contract_record: dict[str, Any] = {
            "kind": "separate_component_score_concatenation",
            "fit_cohort": "combined training points only",
            "replica_policy": args.contract_replicas,
            "component_order": list(components),
            "component_summary_dimensions": [5] * component_count,
            "literal_component_addition": False,
            "raw_joint_covariance_used_for_whitening": False,
            "cross_dependence_handling": (
                f"paired {concatenated_dimension}D component-score rows are retained "
                "and their joint conditional density is learned by SNLE"
            ),
            "components": component_records,
            "raw_joint_covariance_role": "diagnostic only; not used for whitening",
            "raw_joint_covariance_sha256": msc.sha256_array(raw_joint_covariance),
        }
    else:
        schema_version = POSTERIOR_SCHEMA
        analysis_status = "exploratory_two_snapshot_all256_snle"
        summary_construction = f"single {args.compression} summary"
        score_contract_record = {
            "fit_cohort": "combined training points only",
            "replica_policy": args.contract_replicas,
            **component_records[components[0]],
        }
    raw_observation = _select_vector(
        arrays["gx_data_vector"], arrays["g2x_data_vector"], args.probe
    )
    manifest = {
        "schema_version": schema_version,
        "analysis_status": analysis_status,
        "probe": args.probe,
        "cohort": args.cohort,
        "training_replica_policy": args.training_replicas,
        "contract_replica_policy": args.contract_replicas,
        "source_snapshot_warning": master["manifest"]["source_snapshot_warning"],
        "compression": args.compression,
        "summary_construction": summary_construction,
        "component_order": list(components),
        "input_dimension": int(PROBE_SPECS[args.probe]["dimension"]),
        "summary_dimension": int(train_summary.shape[1]),
        "parameter_names": list(PARAMETER_NAMES),
        "n_master_points": prep.N_POINTS,
        "n_master_native_rows": prep.N_ROWS,
        "n_training_points": int(train_points.size),
        "n_training_rows": int(train_mask.sum()),
        "n_heldout_points": int(heldout_points.size),
        "n_heldout_rows": int(heldout_mask.sum()),
        "point_split_sha256": master["manifest"]["point_split"]["sha256"],
        "shared_train_heldout_bank_indices": len(train_banks & heldout_banks),
        "score_contract": score_contract_record,
        "training": training_settings,
        "training_validation_policy": (
            "frozen point holdout is excluded from score/network fitting; SBI's internal 10% split is early stopping only"
        ),
        "mcmc": mcmc_settings,
        "network_seeds": args.seeds,
        "posterior_samples_per_network": posterior_samples,
        "pooled_samples": int(pooled_u.shape[0]),
        "per_network": network_records,
        "max_independent_mcmc_repeat_mean_shift_sigma": max_repeat_shift,
        "max_network_mean_departure_sigma": max_network_departure,
        "convergence_interpretation": (
            "independent sampler-repeat agreement tests MCMC stability; between-network agreement tests learned-likelihood stability; neither alone certifies posterior correctness"
        ),
        "master": str(master["path"]),
        "master_sha256": master["file_sha256"],
        "observation_sha256": msc.sha256_array(raw_observation),
        "observed_summary": observed.tolist(),
        "training_summary_sha256": msc.sha256_array(train_summary),
        "heldout_summary_sha256": msc.sha256_array(heldout_summary),
        "theta_reference": np.asarray(arrays["theta_reference"]).tolist(),
        "u_reference": np.asarray(arrays["u_reference"]).tolist(),
        "namespace": list(NAMESPACE),
        "elapsed_seconds": time.time() - started,
    }

    print("[5/5] writing posterior and diagnostics", flush=True)
    payload: dict[str, Any] = {
        "u": pooled_u,
        "theta": pooled_theta,
        "theta_reference": np.asarray(arrays["theta_reference"], dtype=np.float64),
        "u_reference": np.asarray(arrays["u_reference"], dtype=np.float64),
        "observed_summary": np.asarray(observed, dtype=np.float64),
        "component_order": np.asarray(components),
        "point_split": np.asarray(arrays["point_split"], dtype=np.int8),
        "training_point_indices": train_points,
        "heldout_point_indices": heldout_points,
        "manifest_json": json.dumps(manifest, sort_keys=True),
        **repeat_arrays,
        **heldout_log_prob_arrays,
    }
    if is_separate_joint:
        payload["raw_joint_covariance_diagnostic"] = np.asarray(
            raw_joint_covariance, dtype=np.float64
        )
        for name in components:
            component = contracts[name]
            payload[f"covariance_{name}"] = np.asarray(
                component["covariance"], dtype=np.float64
            )
            payload[f"score_cholesky_{name}"] = np.asarray(
                component["cholesky"], dtype=np.float64
            )
            payload[f"score_operator_{name}"] = np.asarray(
                component["operator"], dtype=np.float64
            )
            payload[f"score_reference_prediction_{name}"] = np.asarray(
                component["reference"], dtype=np.float64
            )
            payload[f"observed_score_{name}"] = np.asarray(
                component["observed_score"], dtype=np.float64
            )
    else:
        component = contracts[components[0]]
        payload["covariance"] = np.asarray(component["covariance"], dtype=np.float64)
        payload["score_operator"] = np.asarray(component["operator"], dtype=np.float64)
        payload["score_reference_prediction"] = np.asarray(
            component["reference"], dtype=np.float64
        )
    for network, draws in enumerate(per_network):
        payload[f"u_seed_{network}"] = draws
        payload[f"theta_seed_{network}"] = _theta_from_u(draws)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output, payload)
    _atomic_write_text(output.with_suffix(".json"), json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

