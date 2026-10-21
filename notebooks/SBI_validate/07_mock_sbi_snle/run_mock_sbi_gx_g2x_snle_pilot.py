#!/usr/bin/env python3
"""Run one matched two-point + three-point exploratory mock-SBI SNLE posterior.

The input is made by ``build_mock_sbi_gx_g2x_pilot.py`` from the existing
five-replica g2X rows.  The accepted probes are ``gy_g2y``,
``gkappa_g2kappa`` and ``gtau_g2tau``.  Each uses a full 28 x 28 covariance,
including the measured two-point--three-point cross block.  This script never
multiplies separate posteriors and never substitutes a block-diagonal covariance.

The NSF/SNLE optimizer, standard-normal probit prior, MCMC settings and four
network initialisations match ``run_mock_sbi_g2x_snle_pilot.py``.  The result is
still an explicitly exploratory 104-point, five-noise-replica pilot.

Example
-------

python notebooks/SBI_validate/07_mock_sbi_snle/run_mock_sbi_gx_g2x_snle_pilot.py \
  --g2x-training-set data/SBI_validate/mock_sbi/paste_plan/round_1/training_set_g2x_rep5_104points.npz \
  --g2x-contract data/SBI_validate/mock_sbi/paste_plan/round_1/g2x_pilot_contract_rep5.npz \
  --joint-product data/SBI_validate/mock_sbi/paste_plan/round_1/gx_g2x_pilot_rep5_104points.npz \
  --probe gy_g2y \
  --output data/SBI_validate/mock_sbi/gx_g2x_pilot_rep5/mock_snle_gx_g2x_pilot5_gy_g2y_posterior.npz
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys

_SBI_ROOT = _pl.Path(__file__).resolve().parents[1]
for _directory in (_SBI_ROOT, _SBI_ROOT / "common"):
    if str(_directory) not in _sys.path:
        _sys.path.insert(0, str(_directory))

import argparse
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

import build_mock_sbi_g2x_pilot_contract as g2pilot
import build_mock_sbi_gx_g2x_pilot as joint_builder
import run_mock_sbi_g2x_snle_pilot as g2runner


POSTERIOR_SCHEMA = "godmax.mock_sbi.gx_g2x_snle_pilot_posterior.v1"
ANALYSIS_STATUS = joint_builder.ANALYSIS_STATUS
PARAMETER_NAMES = joint_builder.PARAMETER_NAMES
PAIR_SPECS = joint_builder.PAIR_SPECS
PRIOR_LOW = g2pilot.PRIOR_LOW
PRIOR_HIGH = g2pilot.PRIOR_HIGH
TRAINING = g2runner.TRAINING
MCMC = g2runner.MCMC
NAMESPACE = (20260904, 1801)
PROBE_CODES = {"gy_g2y": 1, "gkappa_g2kappa": 2, "gtau_g2tau": 3}


def _seeded(probe: str, network_seed: int, stage: int) -> int:
    return int(
        np.random.SeedSequence(
            tuple(NAMESPACE)
            + (PROBE_CODES[probe], int(network_seed), int(stage))
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _cholesky_reconstruction_error(
    covariance: np.ndarray, cholesky: np.ndarray
) -> float:
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    normalized_factor = cholesky / scale[:, None]
    return float(
        np.max(np.abs(normalized_factor @ normalized_factor.T - correlation))
    )


def _load_joint_product(
    path: pathlib.Path,
    training: Mapping[str, Any],
    g2_contract: Mapping[str, Any],
    msc: Any,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"matched gX+g2X product does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "manifest_json" not in arrays:
        raise KeyError("matched gX+g2X product has no manifest_json")
    manifest = joint_builder._json_from_npz(arrays["manifest_json"])
    if manifest.get("schema_version") != joint_builder.OUTPUT_SCHEMA:
        raise ValueError(
            f"joint schema {manifest.get('schema_version')!r} != "
            f"{joint_builder.OUTPUT_SCHEMA!r}"
        )
    if manifest.get("analysis_status") != ANALYSIS_STATUS:
        raise ValueError("joint product is not labelled as the five-replica pilot")
    if tuple(manifest.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("joint product parameter order differs")
    if manifest.get("gx_vector_order") != joint_builder.GX_VECTOR_ORDER:
        raise ValueError("joint product gX order differs")
    if manifest.get("g2x_vector_order") != joint_builder.G2X_VECTOR_ORDER:
        raise ValueError("joint product g2X order differs")
    for name, expected in (
        ("n_points", training["n_points"]),
        ("replicas", training["replicas"]),
        ("n_rows", training["n_rows"]),
    ):
        if int(manifest.get(name, -1)) != int(expected):
            raise ValueError(f"joint product {name} differs from g2X training")
    if str(manifest.get("g2x_training_set", {}).get("sha256")) != str(
        training["file_sha256"]
    ):
        raise ValueError("joint product was built from a different g2X training NPZ")
    if str(manifest.get("g2x_pilot_contract", {}).get("sha256")) != str(
        g2_contract["file_sha256"]
    ):
        raise ValueError("joint product was built from a different g2X pilot contract")

    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"joint-product JSON sidecar is missing: {sidecar}")
    if json.loads(sidecar.read_text()) != manifest:
        raise ValueError("joint-product NPZ manifest and JSON sidecar differ")
    registered = manifest.get("output_arrays", {})
    if not isinstance(registered, dict) or not registered:
        raise ValueError("joint product has no registered output arrays")
    if set(arrays) - {"manifest_json"} != set(registered):
        raise ValueError("joint-product arrays differ from its registered inventory")
    for name, record in registered.items():
        value = np.asarray(arrays[name])
        if list(value.shape) != list(record.get("shape", [])):
            raise ValueError(f"joint array {name!r} shape differs from manifest")
        if str(value.dtype) != str(record.get("dtype")):
            raise ValueError(f"joint array {name!r} dtype differs from manifest")
        if msc.sha256_array(value) != str(record.get("sha256")):
            raise ValueError(f"joint array {name!r} hash differs from manifest")

    n_rows = int(training["n_rows"])
    n_points = int(training["n_points"])
    expected_shapes = {
        "gx": (n_rows, 42),
        "gx_noise": (n_rows, 42),
        "gx_point_signal": (n_points, 42),
        "gx_data_vector": (42,),
        "g2x_data_vector": (42,),
        "theta_reference": (5,),
        "u_reference": (5,),
        "band_edges": (15,),
        "effective_ell": (14,),
    }
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            observed = None if name not in arrays else arrays[name].shape
            raise ValueError(f"joint array {name!r} has shape {observed}; expected {shape}")
        if arrays[name].dtype.kind in "fc" and not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"joint array {name!r} contains non-finite values")

    exact_row_arrays = {
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
        "band_edges": training["band_edges"],
        "effective_ell": training["effective_ell"],
    }
    for name, expected in exact_row_arrays.items():
        if name not in arrays or not np.array_equal(arrays[name], expected, equal_nan=True):
            raise ValueError(f"joint product changed the g2X row array {name!r}")
    if not np.array_equal(
        arrays["gx"], arrays["gx_point_signal"][training["point_index"]] + arrays["gx_noise"]
    ):
        raise ValueError("joint product gX rows are not signal plus their saved noise rows")
    for bank_index in np.unique(training["bank_indices"]):
        rows = np.flatnonzero(training["bank_indices"] == bank_index)
        if rows.size > 1 and not np.array_equal(
            arrays["gx_noise"][rows],
            np.repeat(arrays["gx_noise"][rows[:1]], rows.size, axis=0),
        ):
            raise ValueError(
                f"reused bank index {bank_index} maps to different gX noise vectors"
            )
    g2_arrays = g2_contract["arrays"]
    for name in ("g2x_data_vector", "theta_reference", "u_reference"):
        source_name = "data_vector" if name == "g2x_data_vector" else name
        if not np.array_equal(arrays[name], g2_arrays[source_name]):
            raise ValueError(f"joint product changed g2X contract array {source_name!r}")

    pairs_manifest = manifest.get("pairs", {})
    if set(pairs_manifest) != set(PAIR_SPECS):
        raise ValueError("joint product pair inventory is not canonical")
    for probe, spec in PAIR_SPECS.items():
        record = pairs_manifest[probe]
        gx_indices = np.asarray(spec["gx_indices"], dtype=np.int64)
        g2_indices = np.asarray(spec["g2x_indices"], dtype=np.int64)
        if record.get("gx_source_indices") != gx_indices.tolist():
            raise ValueError(f"{probe} gX source indices are not canonical")
        if record.get("g2x_source_indices") != g2_indices.tolist():
            raise ValueError(f"{probe} g2X source indices are not canonical")
        if int(record.get("covariance_degrees_of_freedom", -1)) != (
            n_points * (int(training["replicas"]) - 1)
        ):
            raise ValueError(f"{probe} has the wrong within-point covariance dof")
        if record.get("positive_definite_without_repair") is not True:
            raise ValueError(f"{probe} is not marked positive definite without repair")
        if record.get("jitter") is not None or record.get("eigenvalue_floor") is not None:
            raise ValueError(f"{probe} declares a forbidden covariance repair")
        required_pair = {
            f"x_{probe}": (n_rows, 28),
            f"data_vector_{probe}": (28,),
            f"covariance_{probe}": (28, 28),
            f"cholesky_{probe}": (28, 28),
            f"cross_covariance_{probe}": (14, 14),
            f"cross_correlation_{probe}": (14, 14),
            f"score_cholesky_{probe}": (28, 28),
            f"score_operator_{probe}": (5, 28),
            f"score_reference_prediction_{probe}": (28,),
            f"observed_score_{probe}": (5,),
        }
        for name, shape in required_pair.items():
            if name not in arrays or arrays[name].shape != shape:
                observed = None if name not in arrays else arrays[name].shape
                raise ValueError(f"joint array {name!r} has shape {observed}; expected {shape}")
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"joint array {name!r} contains non-finite values")
        expected_x = np.concatenate(
            [arrays["gx"][:, gx_indices], training["x"][:, g2_indices]], axis=1
        )
        if not np.array_equal(arrays[f"x_{probe}"], expected_x):
            raise ValueError(f"{probe} rows are not the exact paired gX/g2X rows")
        expected_observation = np.concatenate(
            [arrays["gx_data_vector"][gx_indices], arrays["g2x_data_vector"][g2_indices]]
        )
        if not np.array_equal(arrays[f"data_vector_{probe}"], expected_observation):
            raise ValueError(f"{probe} observation is not the canonical concatenation")
        covariance = np.asarray(arrays[f"covariance_{probe}"], dtype=np.float64)
        cholesky = np.asarray(arrays[f"cholesky_{probe}"], dtype=np.float64)
        if not np.array_equal(covariance, covariance.T):
            raise ValueError(f"{probe} covariance is not exactly symmetric")
        if not np.array_equal(arrays[f"score_covariance_{probe}"], covariance):
            raise ValueError(f"{probe} score covariance differs from the full covariance")
        if not np.array_equal(arrays[f"score_cholesky_{probe}"], cholesky):
            raise ValueError(f"{probe} score Cholesky differs from the saved Cholesky")
        if not np.array_equal(arrays[f"cross_covariance_{probe}"], covariance[:14, 14:]):
            raise ValueError(f"{probe} cross-covariance block is inconsistent")
        if _cholesky_reconstruction_error(covariance, cholesky) > 1.0e-13:
            raise ValueError(f"{probe} Cholesky replay mismatch")
    return {
        "path": path,
        "sidecar": sidecar,
        "file_sha256": msc.sha256_file(path),
        "manifest": manifest,
        "arrays": arrays,
    }


def _run_self_test() -> int:
    import hashlib
    import tempfile

    rng = np.random.default_rng(20260904)
    vectors = rng.normal(size=(13, 28))
    cholesky = np.diag(np.linspace(1.0, 2.0, 28))
    operator = rng.normal(size=(5, 28))
    reference = rng.normal(size=28)
    for compression, dimension in (("score", 5), ("raw", 28)):
        result = g2runner._summarise(
            vectors,
            indices=np.arange(28),
            cholesky=cholesky,
            operator=operator,
            reference_prediction=reference,
            compression=compression,
        )
        assert result.shape == (13, dimension)
    for probe, spec in PAIR_SPECS.items():
        assert np.asarray(spec["gx_indices"]).shape == (14,)
        assert np.asarray(spec["g2x_indices"]).shape == (14,)
        assert len({_seeded(probe, seed, stage) for seed in range(4) for stage in (1, 2)}) == 8
    all_seeds = {
        _seeded(probe, seed, stage)
        for probe in PAIR_SPECS
        for seed in range(4)
        for stage in (1, 2)
    }
    assert len(all_seeds) == len(PAIR_SPECS) * 4 * 2
    midpoint = g2runner._theta_from_u(np.zeros((2, 5), dtype=np.float64))
    np.testing.assert_allclose(
        midpoint,
        np.repeat((0.5 * (PRIOR_LOW + PRIOR_HIGH))[None, :], 2, axis=0),
    )
    covariance = cholesky @ cholesky.T
    assert _cholesky_reconstruction_error(covariance, cholesky) <= 1.0e-13

    # Exercise the strict file loader, including a row-swap negative control.
    n_points, replicas = 36, 5
    n_rows = n_points * replicas
    point_u = rng.normal(size=(n_points, 5))
    point_theta = rng.normal(size=(n_points, 5))
    point_index = np.repeat(np.arange(n_points), replicas)
    replica = np.tile(np.arange(replicas), n_points)
    gx_point_signal = rng.normal(size=(n_points, 42))
    g2_point_signal = rng.normal(size=(n_points, 42))
    gx_noise = rng.normal(size=(n_rows, 42))
    g2_noise = 0.35 * gx_noise + rng.normal(size=(n_rows, 42))
    gx = gx_point_signal[point_index] + gx_noise
    g2x = g2_point_signal[point_index] + g2_noise
    u = np.repeat(point_u, replicas, axis=0)
    theta = np.repeat(point_theta, replicas, axis=0)
    bank_indices = np.arange(n_rows, dtype=np.int64)
    noise_seeds = (
        np.arange(n_rows * 3, dtype=np.uint32).reshape(n_rows, 3) + 100
    )
    plan_position = np.repeat(np.arange(n_points, dtype=np.int64), replicas)
    gx_observation = rng.normal(size=42)
    g2x_observation = rng.normal(size=42)
    theta_reference = np.zeros(5)
    u_reference = np.zeros(5)
    training = {
        "n_rows": n_rows,
        "n_points": n_points,
        "replicas": replicas,
        "file_sha256": "synthetic-training-sha",
        "x": g2x,
        "u": u,
        "theta": theta,
        "point_u": point_u,
        "point_theta": point_theta,
        "point_index": point_index,
        "replica": replica,
        "plan_position": plan_position,
        "plan_entry_index": plan_position.copy(),
        "bank_indices": bank_indices,
        "noise_seeds": noise_seeds,
        "point_bank_indices": bank_indices.reshape(n_points, replicas),
        "band_edges": np.arange(15, dtype=np.int64),
        "effective_ell": np.arange(14, dtype=np.float64),
    }
    fake_g2_contract = {
        "file_sha256": "synthetic-contract-sha",
        "arrays": {
            "data_vector": g2x_observation,
            "theta_reference": theta_reference,
            "u_reference": u_reference,
        },
    }
    product_arrays: dict[str, Any] = {
        "gx": gx,
        "gx_noise": gx_noise,
        "gx_point_signal": gx_point_signal,
        "gx_data_vector": gx_observation,
        "g2x_data_vector": g2x_observation,
        "theta": theta,
        "u": u,
        "point_theta": point_theta,
        "point_u": point_u,
        "point_index": point_index,
        "replica": replica,
        "plan_position": plan_position,
        "plan_entry_index": plan_position.copy(),
        "bank_indices": bank_indices,
        "noise_seeds": noise_seeds,
        "point_bank_indices": bank_indices.reshape(n_points, replicas),
        "theta_reference": theta_reference,
        "u_reference": u_reference,
        "band_edges": training["band_edges"],
        "effective_ell": training["effective_ell"],
    }
    pair_manifest: dict[str, Any] = {}
    for probe in PAIR_SPECS:
        pair_arrays, pair_metadata = joint_builder._joint_payload(
            probe,
            gx,
            g2x,
            point_index,
            point_u,
            gx_observation,
            g2x_observation,
            u_reference,
        )
        pair_manifest[probe] = pair_metadata
        for name, value in pair_arrays.items():
            product_arrays[f"{name}_{probe}"] = value

    class FakeMsc:
        @staticmethod
        def sha256_array(value: np.ndarray) -> str:
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256()
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
            return digest.hexdigest()

        @staticmethod
        def sha256_file(path: pathlib.Path) -> str:
            return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

    product_manifest: dict[str, Any] = {
        "schema_version": joint_builder.OUTPUT_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "parameter_names": list(PARAMETER_NAMES),
        "gx_vector_order": joint_builder.GX_VECTOR_ORDER,
        "g2x_vector_order": joint_builder.G2X_VECTOR_ORDER,
        "n_points": n_points,
        "replicas": replicas,
        "n_rows": n_rows,
        "g2x_training_set": {"sha256": training["file_sha256"]},
        "g2x_pilot_contract": {"sha256": fake_g2_contract["file_sha256"]},
        "pairs": pair_manifest,
        "output_arrays": {},
    }

    def write_product(
        path: pathlib.Path,
        values: Mapping[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        manifest["output_arrays"] = {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "sha256": FakeMsc.sha256_array(np.asarray(value)),
            }
            for name, value in values.items()
        }
        with path.open("wb") as handle:
            np.savez_compressed(
                handle, **values, manifest_json=json.dumps(manifest, sort_keys=True)
            )
        path.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    with tempfile.TemporaryDirectory() as directory:
        product_path = pathlib.Path(directory) / "joint.npz"
        write_product(product_path, product_arrays, product_manifest)
        loaded = _load_joint_product(
            product_path, training, fake_g2_contract, FakeMsc
        )
        assert loaded["arrays"]["x_gy_g2y"].shape == (n_rows, 28)

        broken_arrays = dict(product_arrays)
        broken_rows = np.asarray(broken_arrays["x_gy_g2y"]).copy()
        broken_rows[[0, replicas]] = broken_rows[[replicas, 0]]
        broken_arrays["x_gy_g2y"] = broken_rows
        broken_manifest = json.loads(json.dumps(product_manifest))
        write_product(product_path, broken_arrays, broken_manifest)
        try:
            _load_joint_product(
                product_path, training, fake_g2_contract, FakeMsc
            )
        except ValueError as error:
            if "exact paired gX/g2X rows" not in str(error):
                raise AssertionError("row-swap guard raised the wrong error") from error
        else:
            raise AssertionError("row-swap negative control did not fail")
    print(
        "SELF-TEST PASS: paired 28D score/raw summaries, canonical pair slices, "
        "unique training/sampling seeds, prior transform, Cholesky replay, "
        "strict synthetic product loading, and row-swap rejection"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--g2x-training-set", type=pathlib.Path)
    parser.add_argument("--g2x-contract", type=pathlib.Path)
    parser.add_argument("--joint-product", type=pathlib.Path)
    parser.add_argument("--probe", choices=tuple(PAIR_SPECS))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="must be 4, matching the registered pilot network ensemble",
    )
    parser.add_argument("--compression", choices=("score", "raw"), default="score")
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    for name in (
        "g2x_training_set",
        "g2x_contract",
        "joint_product",
        "probe",
        "output",
    ):
        if getattr(args, name) is None:
            parser.error(
                f"--{name.replace('_', '-')} is required unless --self-test is used"
            )
    if args.seeds != 4:
        parser.error("this pilot requires --seeds 4")
    if args.posterior_samples <= 0:
        parser.error("--posterior-samples must be positive")
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if output.exists() and not (args.overwrite or args.preflight):
        parser.error(f"output exists; pass --overwrite to replace it: {output}")

    started = time.time()
    import mock_sbi_common as msc

    print("[1/5] loading and matching all frozen inputs ...", flush=True)
    training = joint_builder._load_g2x_rows(args.g2x_training_set, msc)
    g2_contract = g2runner._load_contract(args.g2x_contract, training, msc)
    product = _load_joint_product(args.joint_product, training, g2_contract, msc)
    arrays = product["arrays"]
    probe = str(args.probe)
    print(
        f"      {probe}: {training['n_points']} points x "
        f"{training['replicas']} exactly paired replicas",
        flush=True,
    )

    print("[2/5] applying the registered full joint compression ...", flush=True)
    joint_x = np.asarray(arrays[f"x_{probe}"], dtype=np.float64)
    finite = np.all(np.isfinite(training["u"]), axis=1) & np.all(
        np.isfinite(joint_x), axis=1
    )
    if not np.all(finite):
        print(
            f"      dropping {int(np.sum(~finite))} diagnostic row(s) with no "
            "finite probit coordinate",
            flush=True,
        )
    u_rows = np.asarray(training["u"][finite], dtype=np.float64)
    x_rows = joint_x[finite]
    indices = np.arange(28, dtype=np.int64)
    cholesky = np.asarray(arrays[f"score_cholesky_{probe}"], dtype=np.float64)
    operator = np.asarray(arrays[f"score_operator_{probe}"], dtype=np.float64)
    reference = np.asarray(
        arrays[f"score_reference_prediction_{probe}"], dtype=np.float64
    )
    summaries = g2runner._summarise(
        x_rows,
        indices=indices,
        cholesky=cholesky,
        operator=operator,
        reference_prediction=reference,
        compression=args.compression,
    )
    observation = np.asarray(arrays[f"data_vector_{probe}"], dtype=np.float64)
    observed = g2runner._summarise(
        observation[None, :],
        indices=indices,
        cholesky=cholesky,
        operator=operator,
        reference_prediction=reference,
        compression=args.compression,
    )[0]
    if not np.all(np.isfinite(summaries)) or not np.all(np.isfinite(observed)):
        raise RuntimeError("compressed joint training set or observation is non-finite")
    if args.compression == "score":
        if summaries.shape[1] != 5:
            raise RuntimeError(f"score summary has unexpected shape {summaries.shape}")
        scale = max(1.0, float(np.max(np.abs(observed))))
        error = float(np.max(np.abs(observed - arrays[f"observed_score_{probe}"])))
        if error > 1.0e-13 * scale:
            raise RuntimeError("replayed observed joint score differs from its contract")
    print(
        f"      {x_rows.shape[0]} rows: full 28D covariance -> "
        f"{summaries.shape[1]}D {args.compression} summary",
        flush=True,
    )

    if args.preflight:
        print("\nPREFLIGHT PASS")
        print(f"      analysis status: {ANALYSIS_STATUS}")
        print(f"      probe: {probe}")
        print("      covariance: full paired 28x28 including cross block")
        print(f"      compression: {args.compression}")
        print(f"      training rows: {x_rows.shape[0]}")
        print(f"      network seeds: {args.seeds}")
        print(f"      posterior samples per seed: {args.posterior_samples}")
        print(f"      output to create: {output}")
        print("      no files were written")
        return 0

    print(f"[3/5] training {args.seeds} SNLE network replicate(s) ...", flush=True)
    import torch
    from sbi.inference import SNLE
    from torch.distributions import Independent, Normal

    prior = Independent(Normal(torch.zeros(5), torch.ones(5)), 1)
    theta_torch = torch.as_tensor(u_rows, dtype=torch.float32)
    summary_torch = torch.as_tensor(summaries, dtype=torch.float32)
    observed_torch = torch.as_tensor(observed, dtype=torch.float32)
    per_seed: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        seed_started = time.time()
        training_seed = _seeded(probe, seed, 1)
        sampling_seed = _seeded(probe, seed, 2)
        torch.manual_seed(training_seed)
        inference = SNLE(
            prior=prior,
            density_estimator="nsf",
            device="cpu",
            show_progress_bars=False,
        )
        inference.append_simulations(theta_torch, summary_torch)
        estimator = inference.train(**TRAINING, show_train_summary=False)
        posterior = inference.build_posterior(
            estimator,
            prior=prior,
            sample_with="mcmc",
            mcmc_method=MCMC["method"],
            mcmc_parameters={
                "num_chains": MCMC["num_chains"],
                "warmup_steps": MCMC["warmup_steps"],
                "thin": MCMC["thin"],
            },
        )
        torch.manual_seed(sampling_seed)
        draws = np.asarray(
            posterior.sample(
                (args.posterior_samples,),
                x=observed_torch,
                show_progress_bars=False,
            ),
            dtype=np.float64,
        )
        if draws.shape != (args.posterior_samples, 5) or not np.all(np.isfinite(draws)):
            raise RuntimeError(f"seed {seed} returned invalid draws {draws.shape}")
        per_seed.append(draws)
        records.append(
            {
                "seed_index": seed,
                "training_seed": training_seed,
                "sampling_seed": sampling_seed,
                "seconds": time.time() - seed_started,
                "mean_u": draws.mean(axis=0).tolist(),
                "sd_u": draws.std(axis=0, ddof=1).tolist(),
                "q05_u": np.percentile(draws, 5.0, axis=0).tolist(),
                "median_u": np.median(draws, axis=0).tolist(),
                "q95_u": np.percentile(draws, 95.0, axis=0).tolist(),
            }
        )
        print(
            f"      seed {seed}: mean {np.round(draws.mean(axis=0), 3)}  "
            f"sd {np.round(draws.std(axis=0, ddof=1), 3)}  "
            f"({time.time() - seed_started:.0f}s)",
            flush=True,
        )

    print("[4/5] pooling the four network posteriors ...", flush=True)
    pooled_u = np.concatenate(per_seed, axis=0)
    pooled_theta = g2runner._theta_from_u(pooled_u)
    seed_means = np.asarray([record["mean_u"] for record in records])
    pooled_sd = pooled_u.std(axis=0, ddof=1)
    if np.any(pooled_sd <= 0.0) or not np.all(np.isfinite(pooled_sd)):
        raise RuntimeError(f"pooled posterior has invalid probit widths {pooled_sd}")
    max_seed_departure = float(
        np.max(
            np.abs(seed_means - pooled_u.mean(axis=0)[None, :])
            / pooled_sd[None, :]
        )
    )
    print(f"      pooled mean_u {np.round(pooled_u.mean(axis=0), 4)}")
    print(f"      pooled sd_u   {np.round(pooled_sd, 4)}")
    print(
        "      largest seed-mean departure from pooled mean: "
        f"{max_seed_departure:.3f} pooled sigma"
    )

    print("[5/5] writing the explicitly labelled joint pilot posterior ...", flush=True)
    pair_record = product["manifest"]["pairs"][probe]
    manifest = {
        "schema_version": POSTERIOR_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "scientific_scope": (
            "exploratory 104-point/five-replica matched gX+g2X check only; "
            "not a validated or quotable Stage-7 posterior"
        ),
        "probe": probe,
        "input_vector_order": pair_record["vector_order"],
        "input_dimension": 28,
        "covariance_policy": (
            "full paired within-point 28x28 covariance including the measured "
            "gX-g2X cross block"
        ),
        "covariance_degrees_of_freedom": pair_record[
            "covariance_degrees_of_freedom"
        ],
        "covariance_repair": "none",
        "maximum_abs_cross_correlation": pair_record[
            "maximum_abs_cross_correlation"
        ],
        "compression": args.compression,
        "summary_dimension": int(summaries.shape[1]),
        "estimator": "SNLE (NLE, one round, NSF)",
        "proposal_correction_required": False,
        "network_seeds": args.seeds,
        "posterior_samples_per_seed": args.posterior_samples,
        "pooled_samples": int(pooled_u.shape[0]),
        "per_seed": records,
        "max_seed_departure_sigma": max_seed_departure,
        "n_pasted_points": int(training["n_points"]),
        "noise_replicas_per_point": int(training["replicas"]),
        "n_training_rows_before_finite_filter": int(training["n_rows"]),
        "n_training_rows": int(x_rows.shape[0]),
        "g2x_training_set": str(training["path"]),
        "g2x_training_set_sha256": training["file_sha256"],
        "g2x_pilot_contract": str(g2_contract["path"]),
        "g2x_pilot_contract_sha256": g2_contract["file_sha256"],
        "joint_product": str(product["path"]),
        "joint_product_sha256": product["file_sha256"],
        "observation_kind": "paired_noiseless_pasted_gx_and_g2x",
        "observation_vector_sha256": msc.sha256_array(observation),
        "observed_summary": observed.tolist(),
        "observed_summary_norm": float(np.linalg.norm(observed)),
        "training_summary_sha256": msc.sha256_array(summaries),
        "parameter_names": list(PARAMETER_NAMES),
        "theta_reference": np.asarray(arrays["theta_reference"]).tolist(),
        "u_reference": np.asarray(arrays["u_reference"]).tolist(),
        "training": TRAINING,
        "mcmc": MCMC,
        "device": "cpu",
        "namespace": list(NAMESPACE),
        "elapsed_seconds": time.time() - started,
    }
    payload: dict[str, Any] = {
        "u": pooled_u,
        "theta": pooled_theta,
        "theta_reference": np.asarray(arrays["theta_reference"], dtype=np.float64),
        "u_reference": np.asarray(arrays["u_reference"], dtype=np.float64),
        "manifest_json": json.dumps(manifest, sort_keys=True),
    }
    for seed, draws in enumerate(per_seed):
        payload[f"u_seed_{seed}"] = draws
        payload[f"theta_seed_{seed}"] = g2runner._theta_from_u(draws)
    output.parent.mkdir(parents=True, exist_ok=True)
    g2runner._atomic_savez(output, payload)
    print(f"\nwrote {output}")
    print(f"      analysis status {ANALYSIS_STATUS}")
    print(
        f"      {training['n_points']} pasted points, {x_rows.shape[0]} rows, "
        f"{args.seeds} pooled networks, {pooled_u.shape[0]} posterior draws"
    )
    print("\nposterior in physical units (exploratory pooled ensemble):")
    for index, name in enumerate(PARAMETER_NAMES):
        q05, median, q95 = np.percentile(pooled_theta[:, index], [5, 50, 95])
        print(f"   {name:16s} {median:9.4f}   90% CI [{q05:9.4f}, {q95:9.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

