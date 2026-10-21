#!/usr/bin/env python3
"""Build the explicitly exploratory five-replica g-squared SNLE contract.

This script supplies the pieces that the g2X training NPZ deliberately does not
contain:

* a NOISELESS ``g2y/g2kappa/g2tau`` observation measured from the frozen
  reference paste with the same squared-galaxy field and NaMaster workspace;
* a 42 x 42 empirical noise covariance estimated only from within-point
  residuals of the five map-level noise replicas; and
* four normalized score-compression contracts for ``g2y``, ``g2kappa``,
  ``g2tau`` and their joint vector.

The response Jacobian is an exploratory global affine fit to the per-point mean
g2X vector in standard-normal probit coordinates.  This is not the analytic
two-point score operator and this script never loads or reuses the two-point
observation/covariance/operator.  The product is labelled as a five-replica
pilot throughout.  It is useful for a first end-to-end check, but it does not
inherit the Stage-7 agreement result, which required 64 noise replicas and 512
distinct simulator points.

No covariance repair is permitted.  If the within-point covariance is not
positive definite, the script stops rather than adding jitter or flooring an
eigenvalue.

Example
-------

python notebooks/SBI_validate/07_mock_sbi_snle/build_mock_sbi_g2x_pilot_contract.py \\
  --training-set data/SBI_validate/mock_sbi/paste_plan/round_1/training_set_g2x_rep5_104points.npz \\
  --output data/SBI_validate/mock_sbi/paste_plan/round_1/g2x_pilot_contract_rep5.npz \\
  --path-remap /mnt/ceph/users/spandey/ltu-godmax/GODMAX=/work/hdd/bdne/aacharya2/GODMAX
"""

from __future__ import annotations

# Keep imports working from the Stage-7 theme folder.  Shared project modules
# stay in common/; this script does not modify them.
import pathlib as _pl
import sys as _sys

_SBI_ROOT = _pl.Path(__file__).resolve().parents[1]
for _directory in (_SBI_ROOT, _SBI_ROOT / "common"):
    if str(_directory) not in _sys.path:
        _sys.path.insert(0, str(_directory))

import argparse
import json
import os
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

import build_mock_sbi_g2x_trainingset as g2build


CONTRACT_SCHEMA = "godmax.mock_sbi.g2x_pilot_contract.v1"
TRAINING_SCHEMA = g2build.OUTPUT_SCHEMA
ANALYSIS_STATUS = "exploratory_five_replica_pilot"
PARAMETER_NAMES = g2build.PARAMETER_NAMES
VECTOR_ORDER = g2build.VECTOR_ORDER
EXPECTED_REPLICAS = 5
PRIOR_LOW = np.asarray([0.5, 0.0, 0.005, 0.001, -1.0], dtype=np.float64)
PRIOR_HIGH = np.asarray([8.0, 0.5, 1.5, 0.5, 1.0], dtype=np.float64)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GENERATING_POINT_PATH = REPO_ROOT / (
    "data/SBI_validate/three_probe_inference/"
    "observation_mock_generating_point.json"
)
PROBE_INDICES = {
    "g2y": np.arange(0, 14, dtype=np.int64),
    "g2kappa": np.arange(14, 28, dtype=np.int64),
    "g2tau": np.arange(28, 42, dtype=np.int64),
    "all": np.arange(0, 42, dtype=np.int64),
}


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
    payload: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    if name not in payload:
        raise KeyError(f"training set is missing array {name!r}")
    value = np.asarray(payload[name], dtype=dtype)
    if not np.all(np.isfinite(value)) and value.dtype.kind in "fc":
        # Row-level u can contain a forced diagnostic anchor.  It is checked
        # separately and never used for the affine response.
        if name not in {"u", "point_u", "log_q", "point_log_q"}:
            raise ValueError(f"training array {name!r} contains non-finite values")
    return value


def _load_training_set(path: pathlib.Path, msc_module: Any) -> dict[str, Any]:
    """Load and strictly validate the g2X training product."""

    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"training set does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    manifest = _json_from_npz(arrays["manifest_json"])

    if manifest.get("schema_version") != TRAINING_SCHEMA:
        raise ValueError(
            f"training schema {manifest.get('schema_version')!r} != "
            f"{TRAINING_SCHEMA!r}"
        )
    if manifest.get("vector_order") != VECTOR_ORDER:
        raise ValueError(
            f"training vector order {manifest.get('vector_order')!r} != "
            f"{VECTOR_ORDER!r}"
        )
    if tuple(manifest.get("spectra", ())) != tuple(
        name for name, _ in g2build.G2X_SPECTRA
    ):
        raise ValueError("training spectrum inventory differs from the g2X contract")
    if tuple(manifest.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("training parameter order differs from the Stage-7 order")

    n_points = int(manifest["n_points"])
    replicas = int(manifest["replicas"])
    n_rows = int(manifest["n_rows"])
    n_band = int(manifest["n_band"])
    if replicas != EXPECTED_REPLICAS:
        raise ValueError(
            f"this pilot requires exactly {EXPECTED_REPLICAS} replicas, but the "
            f"training set declares {replicas}"
        )
    if n_band != 14 or n_rows != n_points * replicas:
        raise ValueError(
            f"inconsistent training dimensions: {n_points} points x {replicas} "
            f"replicas, {n_rows} rows, {n_band} bands"
        )

    x = _require_array(arrays, "x", dtype=np.float64)
    u = _require_array(arrays, "u", dtype=np.float64)
    theta = _require_array(arrays, "theta", dtype=np.float64)
    point_index = _require_array(arrays, "point_index", dtype=np.int64)
    replica = _require_array(arrays, "replica", dtype=np.int64)
    point_u = _require_array(arrays, "point_u", dtype=np.float64)
    point_theta = _require_array(arrays, "point_theta", dtype=np.float64)
    bank_indices = _require_array(arrays, "bank_indices", dtype=np.int64)
    band_edges = _require_array(arrays, "band_edges", dtype=np.int64)
    effective_ell = _require_array(arrays, "effective_ell", dtype=np.float64)

    expected_shapes = {
        "x": (n_rows, 42),
        "u": (n_rows, 5),
        "theta": (n_rows, 5),
        "point_index": (n_rows,),
        "replica": (n_rows,),
        "point_u": (n_points, 5),
        "point_theta": (n_points, 5),
        "bank_indices": (n_rows,),
        "band_edges": (15,),
        "effective_ell": (14,),
    }
    observed = {
        "x": x.shape,
        "u": u.shape,
        "theta": theta.shape,
        "point_index": point_index.shape,
        "replica": replica.shape,
        "point_u": point_u.shape,
        "point_theta": point_theta.shape,
        "bank_indices": bank_indices.shape,
        "band_edges": band_edges.shape,
        "effective_ell": effective_ell.shape,
    }
    for name, expected in expected_shapes.items():
        if observed[name] != expected:
            raise ValueError(
                f"training array {name!r} has shape {observed[name]}, expected "
                f"{expected}"
            )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(theta)):
        raise ValueError("training x/theta arrays contain non-finite values")
    if not np.all(np.isfinite(point_theta)):
        raise ValueError("training point_theta contains non-finite values")
    if not np.array_equal(band_edges, msc_module.INFERENCE_EDGES):
        raise ValueError("training band edges differ from the frozen estimator")
    if msc_module.sha256_array(x) != str(manifest.get("output_vector_sha256")):
        raise ValueError("training x array hash differs from its embedded manifest")

    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"the builder's JSON sidecar is missing: {sidecar}"
        )
    sidecar_manifest = json.loads(sidecar.read_text())
    if sidecar_manifest != manifest:
        raise ValueError("training NPZ manifest and JSON sidecar differ")

    expected_points = np.arange(n_points, dtype=np.int64)
    if not np.array_equal(np.unique(point_index), expected_points):
        raise ValueError("point_index is not the contiguous range 0..n_points-1")
    expected_replicas = np.arange(replicas, dtype=np.int64)
    for point in range(n_points):
        rows = np.flatnonzero(point_index == point)
        if rows.size != replicas:
            raise ValueError(
                f"point {point} has {rows.size} rows; expected {replicas}"
            )
        if not np.array_equal(np.sort(replica[rows]), expected_replicas):
            raise ValueError(
                f"point {point} does not contain replicas 0..{replicas - 1}"
            )
        if not np.array_equal(theta[rows], np.repeat(point_theta[point][None, :], replicas, axis=0)):
            raise ValueError(f"row theta values do not match point_theta at point {point}")
        if np.all(np.isfinite(point_u[point])):
            if not np.array_equal(
                u[rows], np.repeat(point_u[point][None, :], replicas, axis=0)
            ):
                raise ValueError(f"row u values do not match point_u at point {point}")
        elif not np.all(np.isnan(u[rows])):
            raise ValueError(
                f"non-finite point_u at point {point} is not represented by all-NaN rows"
            )

    return {
        "path": path,
        "sidecar": sidecar,
        "file_sha256": msc_module.sha256_file(path),
        "manifest": manifest,
        "x": x,
        "u": u,
        "theta": theta,
        "point_index": point_index,
        "replica": replica,
        "point_u": point_u,
        "point_theta": point_theta,
        "bank_indices": bank_indices,
        "band_edges": band_edges,
        "effective_ell": effective_ell,
        "n_points": n_points,
        "replicas": replicas,
        "n_rows": n_rows,
    }


def _within_point_covariance(
    x: np.ndarray,
    point_index: np.ndarray,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return point means, residuals and their unbiased common covariance."""

    x = np.asarray(x, dtype=np.float64)
    point_index = np.asarray(point_index, dtype=np.int64)
    point_means = np.empty((n_points, x.shape[1]), dtype=np.float64)
    residuals = np.empty_like(x)
    degrees_of_freedom = 0
    for point in range(n_points):
        rows = np.flatnonzero(point_index == point)
        if rows.size < 2:
            raise ValueError(f"point {point} has fewer than two noise replicas")
        mean = np.mean(x[rows], axis=0, dtype=np.float64)
        point_means[point] = mean
        residuals[rows] = x[rows] - mean[None, :]
        degrees_of_freedom += int(rows.size - 1)

    if degrees_of_freedom <= x.shape[1]:
        raise RuntimeError(
            f"within-point covariance has only {degrees_of_freedom} degrees of "
            f"freedom for dimension {x.shape[1]}"
        )
    covariance = residuals.T @ residuals / float(degrees_of_freedom)
    # This is an exact symmetrisation of the same empirical estimator, not a
    # covariance repair or regularisation.
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError("within-point covariance contains non-finite values")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        eigenvalues = np.linalg.eigvalsh(covariance)
        raise RuntimeError(
            "the five-replica within-point covariance is not positive definite; "
            "this pilot forbids jitter/eigenvalue flooring. Generate more map-level "
            f"noise replicas instead. Eigenvalue range {eigenvalues.min():.6e} .. "
            f"{eigenvalues.max():.6e}"
        ) from error
    return point_means, residuals, covariance, degrees_of_freedom


def _fit_affine_response(
    point_u: np.ndarray,
    point_means: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    """Fit mean g2X = intercept + Jacobian @ u on finite design points."""

    point_u = np.asarray(point_u, dtype=np.float64)
    point_means = np.asarray(point_means, dtype=np.float64)
    finite = np.all(np.isfinite(point_u), axis=1) & np.all(
        np.isfinite(point_means), axis=1
    )
    u_fit = point_u[finite]
    y_fit = point_means[finite]
    if u_fit.shape[0] <= u_fit.shape[1] + 1:
        raise RuntimeError(
            f"only {u_fit.shape[0]} finite points are available for a five-parameter "
            "affine response"
        )
    design = np.column_stack([np.ones(u_fit.shape[0], dtype=np.float64), u_fit])
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, y_fit, rcond=None
    )
    if int(rank) != design.shape[1]:
        raise RuntimeError(
            f"affine response design has rank {rank}; expected {design.shape[1]}"
        )
    prediction = design @ coefficients
    residual = y_fit - prediction
    return {
        "intercept": np.asarray(coefficients[0], dtype=np.float64),
        "jacobian": np.asarray(coefficients[1:].T, dtype=np.float64),
        "fit_prediction": np.asarray(prediction, dtype=np.float64),
        "fit_residual": np.asarray(residual, dtype=np.float64),
        "finite_point_mask": finite,
        "rank": int(rank),
        "singular_values": np.asarray(singular_values, dtype=np.float64),
    }


def _score_payload(
    covariance: np.ndarray,
    jacobian: np.ndarray,
    reference_prediction: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    """Build the Stage-7 normalized score for one selected vector block."""

    indices = np.asarray(indices, dtype=np.int64)
    selected_covariance = np.asarray(
        covariance[np.ix_(indices, indices)], dtype=np.float64
    )
    selected_jacobian = np.asarray(jacobian[indices], dtype=np.float64)
    selected_reference = np.asarray(reference_prediction[indices], dtype=np.float64)
    try:
        cholesky = np.linalg.cholesky(selected_covariance)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            f"selected covariance for indices {indices.tolist()} is not positive "
            "definite; no repair is permitted"
        ) from error

    whitened_jacobian = np.linalg.solve(cholesky, selected_jacobian)
    gram = whitened_jacobian.T @ whitened_jacobian
    gram = 0.5 * (gram + gram.T)
    posterior_metric = gram + np.eye(len(PARAMETER_NAMES), dtype=np.float64)
    metric_values, metric_vectors = np.linalg.eigh(posterior_metric)
    if np.any(metric_values <= 0.0) or not np.all(np.isfinite(metric_values)):
        raise RuntimeError("Fisher-plus-prior score metric is not positive definite")
    inverse_sqrt = (
        metric_vectors
        @ np.diag(metric_values ** -0.5)
        @ metric_vectors.T
    )
    operator = inverse_sqrt @ whitened_jacobian.T
    noise_covariance = inverse_sqrt @ gram @ inverse_sqrt
    singular_values = np.linalg.svd(whitened_jacobian, compute_uv=False)
    return {
        "indices": indices,
        "covariance": selected_covariance,
        "cholesky": cholesky,
        "jacobian": selected_jacobian,
        "whitened_jacobian": whitened_jacobian,
        "gram": gram,
        "posterior_metric": posterior_metric,
        "operator": operator,
        "noise_covariance": noise_covariance,
        "reference_prediction": selected_reference,
        "response_singular_values": singular_values,
        "response_rank": int(np.linalg.matrix_rank(whitened_jacobian)),
        "metric_eigenvalues": metric_values,
    }


def _load_generating_point(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.special import ndtri

    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"mock generating-point JSON is missing: {path}")
    payload = json.loads(path.read_text())
    if tuple(payload.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("mock generating-point parameter order is inconsistent")
    theta = np.asarray(payload.get("theta"), dtype=np.float64)
    if theta.shape != (5,) or not np.all(np.isfinite(theta)):
        raise ValueError("mock generating-point theta is not a finite five-vector")
    fraction = (theta - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)
    if not np.all((fraction > 0.0) & (fraction < 1.0)):
        raise ValueError("mock generating point is not strictly inside the prior")
    u = np.asarray(ndtri(fraction), dtype=np.float64)
    return theta, u, payload


def _measure_noiseless_g2x(
    signal_maps: Mapping[str, np.ndarray],
    context: Any,
    g2_alm: np.ndarray,
    hp: Any,
    msc: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the noiseless reference through the training builder's estimator."""

    blocks: list[np.ndarray] = []
    removed_means: list[float] = []
    for _, field in g2build.G2X_SPECTRA:
        signal = np.asarray(signal_maps[field], dtype=np.float64)
        if signal.shape != context.mask.shape or not np.all(np.isfinite(signal)):
            raise ValueError(
                f"reference {field} map has invalid shape/content {signal.shape}"
            )
        centered, removed_mean = msc.subtract_weighted_mean(signal, context.mask)
        masked_alm = hp.map2alm(
            context.mask * centered,
            lmax=msc.LMAX,
            iter=0,
        )
        coupled = hp.alm2cl(g2_alm, masked_alm, lmax=msc.LMAX)
        bandpowers = np.asarray(
            context.workspace.decouple_cell(coupled[None, :]), dtype=np.float64
        )[0]
        if bandpowers.shape != (14,) or not np.all(np.isfinite(bandpowers)):
            raise RuntimeError(
                f"reference g2{field} block has invalid shape/content "
                f"{bandpowers.shape}"
            )
        blocks.append(bandpowers)
        removed_means.append(float(removed_mean))
    vector = np.concatenate(blocks)
    if vector.shape != (42,) or not np.all(np.isfinite(vector)):
        raise RuntimeError("noiseless g2X observation is not a finite 42-vector")
    return vector, np.asarray(removed_means, dtype=np.float64)


def _run_self_test() -> int:
    rng = np.random.default_rng(20260902)
    n_points, replicas, dimension = 24, 5, 9
    point_u = rng.normal(size=(n_points, 5))
    intercept = rng.normal(size=dimension)
    jacobian = rng.normal(size=(dimension, 5))
    base = rng.normal(size=(dimension, dimension))
    noise_factor = np.linalg.cholesky(base @ base.T + np.eye(dimension))
    point_index = np.repeat(np.arange(n_points), replicas)
    signal = intercept[None, :] + point_u @ jacobian.T
    x = np.repeat(signal, replicas, axis=0)
    x += rng.normal(size=x.shape) @ noise_factor.T

    point_means, residuals, covariance, dof = _within_point_covariance(
        x, point_index, n_points
    )
    assert point_means.shape == (n_points, dimension)
    assert residuals.shape == x.shape
    assert dof == n_points * (replicas - 1)
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=0.0)
    np.linalg.cholesky(covariance)

    response = _fit_affine_response(point_u, point_means)
    assert response["jacobian"].shape == (dimension, 5)
    assert response["rank"] == 6
    reference = np.asarray(response["intercept"]) + np.asarray(
        response["jacobian"]
    ) @ np.zeros(5)
    selection = np.arange(0, 3, dtype=np.int64)
    score = _score_payload(
        covariance,
        np.asarray(response["jacobian"]),
        reference,
        selection,
    )
    assert score["operator"].shape == (5, 3)
    assert score["cholesky"].shape == (3, 3)

    # Null control: changing unselected columns cannot change this summary.
    vectors = rng.normal(size=(7, dimension))
    changed = vectors.copy()
    changed[:, 3:] += 1.0e6
    selected_reference = np.asarray(score["reference_prediction"])
    selected_cholesky = np.asarray(score["cholesky"])
    selected_operator = np.asarray(score["operator"])

    def summarise(values: np.ndarray) -> np.ndarray:
        whitened = np.linalg.solve(
            selected_cholesky,
            (values[:, selection] - selected_reference[None, :]).T,
        )
        return (selected_operator @ whitened).T

    np.testing.assert_array_equal(summarise(vectors), summarise(changed))
    print(
        "SELF-TEST PASS: within-point covariance, affine response, score shapes, "
        "and unselected-block null control"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--training-set",
        type=pathlib.Path,
        help="existing five-replica g2X training NPZ",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="explicit new .npz pilot contract to create",
    )
    parser.add_argument(
        "--generating-point",
        type=pathlib.Path,
        default=GENERATING_POINT_PATH,
        help="frozen mock generating-point JSON",
    )
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        metavar="OLD_PREFIX=NEW_PREFIX",
        help="rewrite absolute paths embedded in the frozen contract; repeat as needed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing contract and JSON sidecar",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="perform every calculation and validation but write no files",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run dependency-free synthetic checks and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    if args.training_set is None:
        parser.error("--training-set is required unless --self-test is used")
    if args.output is None:
        parser.error("--output is required unless --self-test is used")

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

    remaps = g2build._parse_path_remaps(args.path_remap)
    generating_point_path = g2build._resolve_path(
        args.generating_point, msc.REPO_ROOT, remaps
    )

    print("[1/5] loading the five-replica g2X training set ...", flush=True)
    training = _load_training_set(args.training_set, msc)
    print(
        f"      {training['n_points']} points x {training['replicas']} replicas "
        f"= {training['n_rows']} rows; order {VECTOR_ORDER}",
        flush=True,
    )

    print("[2/5] estimating within-point noise covariance ...", flush=True)
    point_means, residuals, covariance, covariance_dof = _within_point_covariance(
        training["x"], training["point_index"], training["n_points"]
    )
    cholesky = np.linalg.cholesky(covariance)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    residual_group_sums = np.stack(
        [
            residuals[training["point_index"] == point].sum(axis=0)
            for point in range(training["n_points"])
        ]
    )
    print(
        f"      dof {covariance_dof}; eigenvalue range "
        f"{covariance_eigenvalues.min():.6e} .. "
        f"{covariance_eigenvalues.max():.6e}; no repair applied",
        flush=True,
    )

    print("[3/5] measuring the noiseless frozen-reference g2X observation ...", flush=True)
    import healpy as hp

    context = g2build._load_estimator_context(msc, remaps)
    training_estimator = training["manifest"].get("estimator", {})
    current_identity = {
        "noise_contract_sha256": context.contract_sha256,
        "workspace_sha256": context.workspace_sha256,
        "mask_array_sha256": context.mask_sha256,
        "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
    }
    for key, value in current_identity.items():
        if str(training_estimator.get(key)) != str(value):
            raise ValueError(
                f"training estimator {key}={training_estimator.get(key)!r}, but "
                f"the current frozen estimator gives {value!r}"
            )
    if not np.array_equal(training["band_edges"], context.band_edges):
        raise ValueError("training and current estimator band edges differ")
    if not np.array_equal(training["effective_ell"], context.effective_ell):
        raise ValueError("training and current estimator effective ell differ")

    (
        reference_map_path,
        reference_map_sha256,
        frozen_galaxy_contract,
    ) = g2build._resolve_and_verify_frozen_galaxy_map(msc, remaps)
    g2_alm, galaxy_metadata = g2build._build_fixed_g2_alm(
        context,
        reference_map_path,
        reference_map_sha256,
        frozen_galaxy_contract,
        hp,
        msc,
    )
    training_galaxy = training["manifest"].get("galaxy", {})
    if str(training_galaxy.get("g2_alm_sha256")) != str(
        galaxy_metadata["g2_alm_sha256"]
    ):
        raise ValueError(
            "the reconstructed squared-galaxy alm differs from the one that "
            "measured the training set"
        )
    signal_maps = msc.read_paste_maps(reference_map_path)
    observation, observation_removed_means = _measure_noiseless_g2x(
        signal_maps, context, g2_alm, hp, msc
    )
    del signal_maps
    theta_reference, u_reference, generating_payload = _load_generating_point(
        generating_point_path
    )
    print(
        f"      observation sha256 {msc.sha256_array(observation)}; "
        f"reference u {np.round(u_reference, 4)}",
        flush=True,
    )

    print("[4/5] fitting the affine response and four score contracts ...", flush=True)
    response = _fit_affine_response(training["point_u"], point_means)
    intercept = np.asarray(response["intercept"], dtype=np.float64)
    jacobian = np.asarray(response["jacobian"], dtype=np.float64)
    reference_prediction = intercept + jacobian @ u_reference
    fit_residual = np.asarray(response["fit_residual"], dtype=np.float64)
    fit_whitened = np.linalg.solve(cholesky, fit_residual.T).T
    scores = {
        name: _score_payload(
            covariance,
            jacobian,
            reference_prediction,
            indices,
        )
        for name, indices in PROBE_INDICES.items()
    }
    observed_score_norms: dict[str, float] = {}
    for name, score in scores.items():
        indices = np.asarray(score["indices"], dtype=np.int64)
        whitened_observation = np.linalg.solve(
            np.asarray(score["cholesky"]),
            observation[indices] - np.asarray(score["reference_prediction"]),
        )
        observed_score = np.asarray(score["operator"]) @ whitened_observation
        score["observed_score"] = observed_score
        observed_score_norms[name] = float(np.linalg.norm(observed_score))
        print(
            f"      {name:8s} dim {indices.size:2d} -> score 5; "
            f"response rank {score['response_rank']}; "
            f"|s_obs| {observed_score_norms[name]:.4f}",
            flush=True,
        )

    manifest: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "scientific_scope": (
            "exploratory end-to-end check only; not covered by the validated "
            "64-replica/512-point Stage-7 agreement result"
        ),
        "vector_order": VECTOR_ORDER,
        "spectra": [name for name, _ in g2build.G2X_SPECTRA],
        "n_band": 14,
        "parameter_names": list(PARAMETER_NAMES),
        "prior_low": PRIOR_LOW.tolist(),
        "prior_high": PRIOR_HIGH.tolist(),
        "training_set": {
            "path": str(training["path"]),
            "sha256": training["file_sha256"],
            "schema_version": training["manifest"]["schema_version"],
            "n_points": training["n_points"],
            "replicas": training["replicas"],
            "n_rows": training["n_rows"],
            "selected_unique_bank_indices": int(
                np.unique(training["bank_indices"]).size
            ),
        },
        "observation": {
            "kind": "frozen_reference_paste_noiseless_g2x",
            "map_path": str(reference_map_path),
            "map_sha256": reference_map_sha256,
            "data_vector_sha256": msc.sha256_array(observation),
            "noise": "none",
            "removed_field_means": observation_removed_means.tolist(),
        },
        "generating_point": {
            "path": str(generating_point_path),
            "sha256": msc.sha256_file(generating_point_path),
            "source_schema": generating_payload.get("schema"),
            "theta": theta_reference.tolist(),
            "u": u_reference.tolist(),
        },
        "covariance": {
            "estimator": (
                "pooled unbiased within-point residual covariance: "
                "sum_(point,replica) r r^T / sum_point(R_point-1)"
            ),
            "degrees_of_freedom": covariance_dof,
            "degrees_of_freedom_interpretation": (
                "algebraic residual degrees of freedom; repeated noise-bank "
                "indices reduce the effective statistical precision"
            ),
            "dimension": 42,
            "positive_definite_without_repair": True,
            "jitter": None,
            "eigenvalue_floor": None,
            "minimum_eigenvalue": float(covariance_eigenvalues.min()),
            "maximum_eigenvalue": float(covariance_eigenvalues.max()),
            "condition_number": float(
                covariance_eigenvalues.max() / covariance_eigenvalues.min()
            ),
            "maximum_abs_group_residual_sum": float(
                np.max(np.abs(residual_group_sums))
            ),
        },
        "response": {
            "model": "global affine OLS of per-point mean g2X on [1,u]",
            "exploratory_approximation": True,
            "n_finite_points": int(np.sum(response["finite_point_mask"])),
            "design_rank": int(response["rank"]),
            "design_singular_values": np.asarray(
                response["singular_values"]
            ).tolist(),
            "whitened_fit_residual_rms_per_component": float(
                np.sqrt(np.mean(fit_whitened * fit_whitened))
            ),
            "reference": "affine response evaluated at generating-point u",
        },
        "score_compression": {
            "normalization": "posterior_metric_fisher_plus_standard_normal_prior",
            "formula": "H^-1/2 (L^-1 J)^T L^-1 (x-mu_ref), H=J^T C^-1 J+I",
            "probes": {
                name: {
                    "indices": np.asarray(score["indices"]).tolist(),
                    "input_dimension": int(np.asarray(score["indices"]).size),
                    "output_dimension": 5,
                    "response_rank": int(score["response_rank"]),
                    "response_singular_values": np.asarray(
                        score["response_singular_values"]
                    ).tolist(),
                    "metric_eigenvalues": np.asarray(
                        score["metric_eigenvalues"]
                    ).tolist(),
                    "observed_score_norm": observed_score_norms[name],
                }
                for name, score in scores.items()
            },
        },
        "estimator": {
            **current_identity,
            "nside": int(msc.NSIDE),
            "lmax": int(msc.LMAX),
            "band_edges": np.asarray(context.band_edges).tolist(),
            "effective_ell": np.asarray(context.effective_ell).tolist(),
            "coupling": "frozen scalar NaMaster workspace; decouple_cell",
            "galaxy": galaxy_metadata,
        },
        "path_remaps": [f"{old}={new}" for old, new in remaps],
        "source_code": {
            "builder": {
                "path": str(pathlib.Path(__file__).resolve()),
                "sha256": msc.sha256_file(pathlib.Path(__file__).resolve()),
            },
            "g2_training_builder": {
                "path": str(pathlib.Path(g2build.__file__).resolve()),
                "sha256": msc.sha256_file(pathlib.Path(g2build.__file__).resolve()),
            },
            "mock_sbi_common": {
                "path": str(pathlib.Path(msc.__file__).resolve()),
                "sha256": msc.sha256_file(pathlib.Path(msc.__file__).resolve()),
            },
        },
        "output_arrays": {},
        "elapsed_seconds": time.time() - started,
    }

    output_arrays: dict[str, Any] = {
        "data_vector": observation,
        "covariance": covariance,
        "cholesky": cholesky,
        "point_means": point_means,
        "within_point_residuals": residuals,
        "affine_intercept": intercept,
        "affine_jacobian": jacobian,
        "affine_reference_prediction": reference_prediction,
        "affine_fit_residual": fit_residual,
        "finite_point_mask": np.asarray(response["finite_point_mask"], dtype=bool),
        "theta_reference": theta_reference,
        "u_reference": u_reference,
        "band_edges": np.asarray(context.band_edges, dtype=np.int64),
        "effective_ell": np.asarray(context.effective_ell, dtype=np.float64),
    }
    for name, score in scores.items():
        for key in (
            "indices",
            "covariance",
            "cholesky",
            "jacobian",
            "whitened_jacobian",
            "gram",
            "posterior_metric",
            "operator",
            "noise_covariance",
            "reference_prediction",
            "observed_score",
        ):
            output_arrays[f"{key}_{name}"] = np.asarray(score[key])

    manifest["output_arrays"] = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": msc.sha256_array(np.asarray(value)),
        }
        for name, value in output_arrays.items()
    }
    output_arrays["manifest_json"] = json.dumps(manifest, sort_keys=True)

    print("[5/5] validating output target ...", flush=True)
    if args.preflight:
        print("\nPREFLIGHT PASS")
        print(f"      analysis status: {ANALYSIS_STATUS}")
        print(f"      training set: {training['path']}")
        print(f"      covariance dof: {covariance_dof}")
        print(f"      output to create: {output}")
        print(f"      manifest to create: {sidecar}")
        print("      no files were written")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output, output_arrays)
    _atomic_write_text(sidecar, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {output}")
    print(f"      manifest {sidecar}")
    print(f"      analysis status {ANALYSIS_STATUS}")
    print(
        f"      covariance {covariance.shape}, dof {covariance_dof}; "
        "four five-dimensional score contracts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
