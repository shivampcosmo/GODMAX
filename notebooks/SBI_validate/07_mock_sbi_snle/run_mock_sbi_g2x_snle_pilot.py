#!/usr/bin/env python3
"""Run one explicitly exploratory g-squared mock-SBI SNLE posterior.

This is the schema-checking g2X counterpart of ``run_mock_sbi_snle.py``.  It
accepts only the dedicated five-replica pilot contract made by
``build_mock_sbi_g2x_pilot_contract.py`` and exactly one of four selections:

``g2y``
    columns 0:14;
``g2kappa``
    columns 14:28;
``g2tau``
    columns 28:42; or
``all``
    all 42 columns in the frozen order g2y, g2kappa, g2tau.

The default estimator/training/MCMC configuration matches the existing Stage-7
SNLE runner: NSF neural likelihood, score compression, four network seeds, and
the standard-normal prior in probit coordinates.  The scientific interpretation
does not match the validated production arm: 104 distinct points with only five
noise replicas are an exploratory end-to-end check, not a quotable posterior.

The output filename is always explicit.  Example for one probe:

python notebooks/SBI_validate/07_mock_sbi_snle/run_mock_sbi_g2x_snle_pilot.py \\
  --training-set data/SBI_validate/mock_sbi/paste_plan/round_1/training_set_g2x_rep5_104points.npz \\
  --contract data/SBI_validate/mock_sbi/paste_plan/round_1/g2x_pilot_contract_rep5.npz \\
  --probe g2y \\
  --output data/SBI_validate/mock_sbi/g2x_pilot_rep5/mock_snle_g2x_pilot5_g2y_posterior.npz
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
import os
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

import build_mock_sbi_g2x_pilot_contract as pilot


POSTERIOR_SCHEMA = "godmax.mock_sbi.g2x_snle_pilot_posterior.v1"
ANALYSIS_STATUS = pilot.ANALYSIS_STATUS
PARAMETER_NAMES = pilot.PARAMETER_NAMES
PRIOR_LOW = pilot.PRIOR_LOW
PRIOR_HIGH = pilot.PRIOR_HIGH
PROBE_INDICES = pilot.PROBE_INDICES

# Identical optimizer and MCMC settings to the established Stage-7 runner.
TRAINING = {
    "training_batch_size": 512,
    "learning_rate": 5.0e-4,
    "validation_fraction": 0.10,
    "stop_after_epochs": 25,
    "max_num_epochs": 500,
}
MCMC = {
    "method": "slice_np_vectorized",
    "num_chains": 20,
    "warmup_steps": 250,
    "thin": 1,
}
NAMESPACE = (20260902, 1701)
PROBE_CODES = {"g2y": 1, "g2kappa": 2, "g2tau": 3, "all": 4}


def _seeded(probe: str, network_seed: int, stage: int) -> int:
    return int(
        np.random.SeedSequence(
            tuple(NAMESPACE)
            + (PROBE_CODES[probe], int(network_seed), int(stage))
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _theta_from_u(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    values = np.asarray(u, dtype=np.float64)
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * ndtr(values)


def _json_from_npz(value: np.ndarray) -> dict[str, Any]:
    parsed = json.loads(str(np.asarray(value).item()))
    if not isinstance(parsed, dict):
        raise ValueError("manifest_json did not decode to an object")
    return parsed


def _atomic_savez(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _load_contract(
    path: pathlib.Path,
    training: Mapping[str, Any],
    msc_module: Any,
) -> dict[str, Any]:
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"g2X pilot contract does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "manifest_json" not in arrays:
        raise KeyError("g2X pilot contract has no manifest_json")
    manifest = _json_from_npz(arrays["manifest_json"])
    if manifest.get("schema_version") != pilot.CONTRACT_SCHEMA:
        raise ValueError(
            f"contract schema {manifest.get('schema_version')!r} != "
            f"{pilot.CONTRACT_SCHEMA!r}"
        )
    if manifest.get("analysis_status") != ANALYSIS_STATUS:
        raise ValueError("contract is not explicitly labelled as the five-replica pilot")
    if manifest.get("vector_order") != pilot.VECTOR_ORDER:
        raise ValueError("contract vector order differs from the g2X training order")
    if tuple(manifest.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("contract parameter order differs from the Stage-7 order")
    contract_training = manifest.get("training_set", {})
    if str(contract_training.get("sha256")) != str(training["file_sha256"]):
        raise ValueError(
            "pilot contract was built from a different training NPZ: "
            f"{contract_training.get('sha256')} != {training['file_sha256']}"
        )
    if int(contract_training.get("replicas", -1)) != pilot.EXPECTED_REPLICAS:
        raise ValueError("pilot contract does not declare exactly five replicas")
    if int(contract_training.get("n_points", -1)) != int(training["n_points"]):
        raise ValueError("pilot contract and training set disagree on n_points")

    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"pilot contract JSON sidecar is missing: {sidecar}")
    if json.loads(sidecar.read_text()) != manifest:
        raise ValueError("pilot contract NPZ manifest and JSON sidecar differ")

    registered_arrays = manifest.get("output_arrays", {})
    if not isinstance(registered_arrays, dict) or not registered_arrays:
        raise ValueError("pilot contract has no registered output-array hashes")
    for name, record in registered_arrays.items():
        if name not in arrays:
            raise KeyError(f"pilot contract is missing registered array {name!r}")
        value = np.asarray(arrays[name])
        if list(value.shape) != list(record.get("shape", [])):
            raise ValueError(f"contract array {name!r} shape differs from manifest")
        if str(value.dtype) != str(record.get("dtype")):
            raise ValueError(f"contract array {name!r} dtype differs from manifest")
        if msc_module.sha256_array(value) != str(record.get("sha256")):
            raise ValueError(f"contract array {name!r} hash differs from manifest")

    required_global = {
        "data_vector": (42,),
        "covariance": (42, 42),
        "cholesky": (42, 42),
        "theta_reference": (5,),
        "u_reference": (5,),
        "effective_ell": (14,),
    }
    for name, shape in required_global.items():
        if name not in arrays or arrays[name].shape != shape:
            observed = None if name not in arrays else arrays[name].shape
            raise ValueError(
                f"contract array {name!r} has shape {observed}; expected {shape}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"contract array {name!r} contains non-finite values")

    for probe, canonical_indices in PROBE_INDICES.items():
        indices_name = f"indices_{probe}"
        if indices_name not in arrays or not np.array_equal(
            arrays[indices_name], canonical_indices
        ):
            raise ValueError(f"contract indices for {probe} are not canonical")
        dimension = int(canonical_indices.size)
        required_probe = {
            f"covariance_{probe}": (dimension, dimension),
            f"cholesky_{probe}": (dimension, dimension),
            f"operator_{probe}": (5, dimension),
            f"reference_prediction_{probe}": (dimension,),
            f"observed_score_{probe}": (5,),
        }
        for name, shape in required_probe.items():
            if name not in arrays or arrays[name].shape != shape:
                observed = None if name not in arrays else arrays[name].shape
                raise ValueError(
                    f"contract array {name!r} has shape {observed}; expected {shape}"
                )
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"contract array {name!r} contains non-finite values")

        selected = arrays["covariance"][np.ix_(canonical_indices, canonical_indices)]
        if not np.array_equal(arrays[f"covariance_{probe}"], selected):
            raise ValueError(f"saved {probe} covariance is not the canonical submatrix")

    return {
        "path": path,
        "sidecar": sidecar,
        "file_sha256": msc_module.sha256_file(path),
        "manifest": manifest,
        "arrays": arrays,
    }


def _summarise(
    vectors: np.ndarray,
    *,
    indices: np.ndarray,
    cholesky: np.ndarray,
    operator: np.ndarray,
    reference_prediction: np.ndarray,
    compression: str,
) -> np.ndarray:
    values = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    indices = np.asarray(indices, dtype=np.int64)
    selected = values[:, indices]
    whitened = np.linalg.solve(
        np.asarray(cholesky, dtype=np.float64),
        (selected - np.asarray(reference_prediction)[None, :]).T,
    ).T
    if compression == "raw":
        return whitened
    if compression == "score":
        return (np.asarray(operator, dtype=np.float64) @ whitened.T).T
    raise ValueError(f"unsupported compression {compression!r}")


def _run_self_test() -> int:
    rng = np.random.default_rng(20260902)
    vectors = rng.normal(size=(11, 42))
    for probe, indices in PROBE_INDICES.items():
        dimension = indices.size
        cholesky = np.diag(np.linspace(1.0, 2.0, dimension))
        operator = rng.normal(size=(5, dimension))
        reference = rng.normal(size=dimension)
        score = _summarise(
            vectors,
            indices=indices,
            cholesky=cholesky,
            operator=operator,
            reference_prediction=reference,
            compression="score",
        )
        raw = _summarise(
            vectors,
            indices=indices,
            cholesky=cholesky,
            operator=operator,
            reference_prediction=reference,
            compression="raw",
        )
        assert score.shape == (11, 5)
        assert raw.shape == (11, dimension)

        # Null control for every single-probe selection: perturbing columns that
        # are not selected must leave the summary bit-for-bit unchanged.
        if probe != "all":
            changed = vectors.copy()
            unselected = np.setdiff1d(np.arange(42), indices)
            changed[:, unselected] += 1.0e12
            changed_score = _summarise(
                changed,
                indices=indices,
                cholesky=cholesky,
                operator=operator,
                reference_prediction=reference,
                compression="score",
            )
            np.testing.assert_array_equal(score, changed_score)

    u = np.zeros((2, 5), dtype=np.float64)
    theta = _theta_from_u(u)
    expected_midpoint = np.repeat(
        (0.5 * (PRIOR_LOW + PRIOR_HIGH))[None, :], 2, axis=0
    )
    np.testing.assert_allclose(theta, expected_midpoint)
    assert len({_seeded(probe, seed, 1) for probe in PROBE_INDICES for seed in range(4)}) == 16
    print(
        "SELF-TEST PASS: canonical probe slices, score/raw shapes, unselected-block "
        "null controls, prior transform, and unique run seeds"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--training-set", type=pathlib.Path)
    parser.add_argument("--contract", type=pathlib.Path)
    parser.add_argument("--probe", choices=tuple(PROBE_INDICES))
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="explicit new posterior .npz file to create",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="must be 4: the registered pilot pools four network initialisations",
    )
    parser.add_argument("--compression", choices=("score", "raw"), default="score")
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate inputs and summaries, print the output path, and exit",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    for name in ("training_set", "contract", "probe", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --self-test is used")
    if args.seeds != 4:
        parser.error(
            "this pilot requires --seeds 4 so every output has the registered "
            "four-network ensemble and matches the overview notebook"
        )
    if args.posterior_samples <= 0:
        parser.error("--posterior-samples must be positive")

    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if output.exists() and not (args.overwrite or args.preflight):
        parser.error(f"output exists; pass --overwrite to replace it: {output}")

    started = time.time()
    import mock_sbi_common as msc

    print("[1/5] loading and matching the training set and pilot contract ...", flush=True)
    training = pilot._load_training_set(args.training_set, msc)
    contract = _load_contract(args.contract, training, msc)
    arrays = contract["arrays"]
    indices = np.asarray(arrays[f"indices_{args.probe}"], dtype=np.int64)
    cholesky = np.asarray(arrays[f"cholesky_{args.probe}"], dtype=np.float64)
    operator = np.asarray(arrays[f"operator_{args.probe}"], dtype=np.float64)
    reference_prediction = np.asarray(
        arrays[f"reference_prediction_{args.probe}"], dtype=np.float64
    )
    print(
        f"      {args.probe}: columns {indices.tolist()}, "
        f"{training['n_points']} points x {training['replicas']} replicas",
        flush=True,
    )

    print("[2/5] applying the registered g2X compression ...", flush=True)
    finite = np.all(np.isfinite(training["u"]), axis=1) & np.all(
        np.isfinite(training["x"]), axis=1
    )
    if not np.all(finite):
        print(
            f"      dropping {int(np.sum(~finite))} diagnostic row(s) with no "
            "finite probit coordinate",
            flush=True,
        )
    u_rows = np.asarray(training["u"][finite], dtype=np.float64)
    x_rows = np.asarray(training["x"][finite], dtype=np.float64)
    summaries = _summarise(
        x_rows,
        indices=indices,
        cholesky=cholesky,
        operator=operator,
        reference_prediction=reference_prediction,
        compression=args.compression,
    )
    observation = np.asarray(arrays["data_vector"], dtype=np.float64)
    observed = _summarise(
        observation[None, :],
        indices=indices,
        cholesky=cholesky,
        operator=operator,
        reference_prediction=reference_prediction,
        compression=args.compression,
    )[0]
    if not np.all(np.isfinite(summaries)) or not np.all(np.isfinite(observed)):
        raise RuntimeError("compressed training set or observation is non-finite")
    if args.compression == "score" and summaries.shape[1] != 5:
        raise RuntimeError(f"score summary has unexpected shape {summaries.shape}")
    print(
        f"      {x_rows.shape[0]} rows -> {summaries.shape[1]} dimensions; "
        f"|summary_obs| {np.linalg.norm(observed):.4f}",
        flush=True,
    )

    if args.preflight:
        print("\nPREFLIGHT PASS")
        print(f"      analysis status: {ANALYSIS_STATUS}")
        print(f"      probe: {args.probe}")
        print(f"      compression: {args.compression}")
        print(f"      training rows: {x_rows.shape[0]}")
        print(f"      network seeds: {args.seeds}")
        print(f"      samples per seed: {args.posterior_samples}")
        print(f"      output to create: {output}")
        print("      no files were written")
        return 0

    print(f"[3/5] training {args.seeds} SNLE network replicate(s) ...", flush=True)
    import torch
    from torch.distributions import Independent, Normal
    from sbi.inference import SNLE

    prior = Independent(Normal(torch.zeros(5), torch.ones(5)), 1)
    theta_torch = torch.as_tensor(u_rows, dtype=torch.float32)
    summary_torch = torch.as_tensor(summaries, dtype=torch.float32)
    observed_torch = torch.as_tensor(observed, dtype=torch.float32)
    per_seed: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        t0 = time.time()
        train_seed = _seeded(args.probe, seed, 1)
        sample_seed = _seeded(args.probe, seed, 2)
        torch.manual_seed(train_seed)
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
        torch.manual_seed(sample_seed)
        draws = np.asarray(
            posterior.sample(
                (args.posterior_samples,),
                x=observed_torch,
                show_progress_bars=False,
            ),
            dtype=np.float64,
        )
        if draws.shape != (args.posterior_samples, 5) or not np.all(
            np.isfinite(draws)
        ):
            raise RuntimeError(
                f"seed {seed} returned invalid posterior draws {draws.shape}"
            )
        per_seed.append(draws)
        records.append(
            {
                "seed_index": seed,
                "training_seed": train_seed,
                "sampling_seed": sample_seed,
                "seconds": time.time() - t0,
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
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    print("[4/5] pooling the four network posteriors ...", flush=True)
    pooled_u = np.concatenate(per_seed, axis=0)
    pooled_theta = _theta_from_u(pooled_u)
    seed_means = np.asarray([record["mean_u"] for record in records])
    pooled_sd = pooled_u.std(axis=0, ddof=1)
    if np.any(pooled_sd <= 0.0) or not np.all(np.isfinite(pooled_sd)):
        raise RuntimeError(f"pooled posterior has invalid probit widths {pooled_sd}")
    max_seed_departure = (
        float(
            np.max(
                np.abs(seed_means - pooled_u.mean(axis=0)[None, :])
                / pooled_sd[None, :]
            )
        )
        if args.seeds > 1
        else float("nan")
    )
    print(f"      pooled mean_u {np.round(pooled_u.mean(axis=0), 4)}")
    print(f"      pooled sd_u   {np.round(pooled_sd, 4)}")
    print(
        "      largest seed-mean departure from pooled mean: "
        f"{max_seed_departure:.3f} pooled sigma"
    )

    print("[5/5] writing the explicitly labelled pilot posterior ...", flush=True)
    manifest = {
        "schema_version": POSTERIOR_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "scientific_scope": (
            "exploratory end-to-end check only; not a validated or quotable "
            "Stage-7 posterior"
        ),
        "probe": args.probe,
        "probe_indices": indices.tolist(),
        "input_vector_order": pilot.VECTOR_ORDER,
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
        "training_set": str(training["path"]),
        "training_set_sha256": training["file_sha256"],
        "pilot_contract": str(contract["path"]),
        "pilot_contract_sha256": contract["file_sha256"],
        "observation_kind": contract["manifest"]["observation"]["kind"],
        "observation_vector_sha256": msc.sha256_array(observation),
        "observed_summary": observed.tolist(),
        "observed_summary_norm": float(np.linalg.norm(observed)),
        "training_summary_sha256": msc.sha256_array(summaries),
        "parameter_names": list(PARAMETER_NAMES),
        "theta_reference": np.asarray(arrays["theta_reference"]).tolist(),
        "u_reference": np.asarray(arrays["u_reference"]).tolist(),
        "training": TRAINING,
        "mcmc": MCMC,
        "namespace": list(NAMESPACE),
        "elapsed_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_payload: dict[str, Any] = {
        "u": pooled_u,
        "theta": pooled_theta,
        "theta_reference": np.asarray(arrays["theta_reference"], dtype=np.float64),
        "u_reference": np.asarray(arrays["u_reference"], dtype=np.float64),
        "manifest_json": json.dumps(manifest, sort_keys=True),
    }
    for seed, draws in enumerate(per_seed):
        save_payload[f"u_seed_{seed}"] = draws
        save_payload[f"theta_seed_{seed}"] = _theta_from_u(draws)
    _atomic_savez(output, save_payload)

    print(f"\nwrote {output}")
    print(f"      analysis status {ANALYSIS_STATUS}")
    print(
        f"      {training['n_points']} pasted points, {x_rows.shape[0]} rows, "
        f"{args.seeds} pooled networks, {pooled_u.shape[0]} posterior draws"
    )
    print("\nposterior in physical units (exploratory pooled ensemble):")
    for index, name in enumerate(PARAMETER_NAMES):
        q05, median, q95 = np.percentile(pooled_theta[:, index], [5, 50, 95])
        print(
            f"   {name:16s} {median:9.4f}   90% CI "
            f"[{q05:9.4f}, {q95:9.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
