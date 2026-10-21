#!/usr/bin/env python3
r"""Run one gX-only exploratory mock-SBI SNLE posterior.

This runner uses the exact two-point rows already saved by the matched
``gx_g2x_pilot_rep5_104points.npz`` product.  It does not rebuild maps, draw new
noise, or load the historical frozen-source registry.  The accepted probes are
``gy``, ``gkappa``, and ``gtau``; each selects one canonical 14-bin block from
the saved 42-vector ordered ``gy[14],gkappa[14],gtau[14]``.

For a gX-only posterior, the covariance is the 14 x 14 gX principal block of
the corresponding saved paired covariance.  The affine response is the gX
block of the same paired response.  A new normalized five-parameter score is
then constructed from those gX-only objects.  The g2X block and the measured
gX--g2X cross block are deliberately not used.

The g2X training set and pilot contract are read only to let the existing
joint-product loader verify exact row order, seed assignment, NPZ array hashes,
and JSON-sidecar consistency.  These checks are numerical/product-integrity
checks, not historical code-source provenance checks.

Because the original ``gy`` pilot did not appear converged, that probe alone
uses a larger training and MCMC budget: smaller optimizer batches, longer
early-stopping patience, more maximum epochs, more MCMC warm-up, and 100000
retained samples per network by default.  ``gkappa`` and ``gtau`` retain the
original pilot settings.  These larger budgets do not themselves certify
convergence; the four independently trained network posteriors must still be
compared after the run.

Example
-------

python notebooks/SBI_validate/07_mock_sbi_snle/run_mock_sbi_gx_only_snle_pilot_gy_longer_20260910.py \
  --g2x-training-set data/SBI_validate/mock_sbi/paste_plan/round_1/training_set_g2x_rep5_104points.npz \
  --g2x-contract data/SBI_validate/mock_sbi/paste_plan/round_1/g2x_pilot_contract_rep5.npz \
  --joint-product data/SBI_validate/mock_sbi/gx_g2x_pilot_rep5/gx_g2x_pilot_rep5_104points.npz \
  --probe gy \
  --output data/SBI_validate/mock_sbi/gx_g2x_pilot_rep5/mock_snle_gx_pilot5_gy_posterior.npz
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
import run_mock_sbi_gx_g2x_snle_pilot as joint_runner


POSTERIOR_SCHEMA = "godmax.mock_sbi.gx_only_snle_pilot_posterior.v1"
ANALYSIS_STATUS = "exploratory_five_replica_gx_only_pilot"
PARAMETER_NAMES = joint_builder.PARAMETER_NAMES
PRIOR_LOW = g2pilot.PRIOR_LOW
PRIOR_HIGH = g2pilot.PRIOR_HIGH
TRAINING = g2runner.TRAINING
MCMC = g2runner.MCMC
GY_TRAINING = {
    "training_batch_size": 64,
    "learning_rate": 2.0e-4,
    "validation_fraction": 0.10,
    "stop_after_epochs": 100,
    "max_num_epochs": 2000,
}
GY_MCMC = {
    "method": "slice_np_vectorized",
    "num_chains": 20,
    "warmup_steps": 1000,
    "thin": 1,
}
DEFAULT_POSTERIOR_SAMPLES = 20_000
GY_DEFAULT_POSTERIOR_SAMPLES = 100_000
NAMESPACE = (20260910, 2601)
EXPECTED_POINTS = 104

GX_SPECS: dict[str, dict[str, Any]] = {
    "gy": {
        "indices": np.arange(0, 14, dtype=np.int64),
        "paired_probe": "gy_g2y",
    },
    "gkappa": {
        "indices": np.arange(14, 28, dtype=np.int64),
        "paired_probe": "gkappa_g2kappa",
    },
    "gtau": {
        "indices": np.arange(28, 42, dtype=np.int64),
        "paired_probe": "gtau_g2tau",
    },
}
PROBE_CODES = {"gy": 1, "gkappa": 2, "gtau": 3}


def _settings_for_probe(
    probe: str, requested_posterior_samples: int | None
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Return effective inference settings, with a longer budget only for gy."""

    if probe not in GX_SPECS:
        raise ValueError(f"unknown gX probe {probe!r}")
    if (
        requested_posterior_samples is not None
        and requested_posterior_samples <= 0
    ):
        raise ValueError("posterior sample count must be positive")
    if probe == "gy":
        training = dict(GY_TRAINING)
        mcmc = dict(GY_MCMC)
        default_samples = GY_DEFAULT_POSTERIOR_SAMPLES
    else:
        training = dict(TRAINING)
        mcmc = dict(MCMC)
        default_samples = DEFAULT_POSTERIOR_SAMPLES
    posterior_samples = (
        default_samples
        if requested_posterior_samples is None
        else int(requested_posterior_samples)
    )
    return training, mcmc, posterior_samples


def _seeded(probe: str, network_seed: int, stage: int) -> int:
    """Return a deterministic, probe-separated torch seed."""

    return int(
        np.random.SeedSequence(
            tuple(NAMESPACE)
            + (PROBE_CODES[probe], int(network_seed), int(stage))
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _require_finite_shape(
    arrays: Mapping[str, Any], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    if name not in arrays:
        raise KeyError(f"joint product is missing array {name!r}")
    value = np.asarray(arrays[name])
    if value.shape != shape:
        raise ValueError(
            f"joint array {name!r} has shape {value.shape}; expected {shape}"
        )
    if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
        raise ValueError(f"joint array {name!r} contains non-finite values")
    return value


def _cholesky_reconstruction_error(
    covariance: np.ndarray, cholesky: np.ndarray
) -> float:
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    normalized_factor = cholesky / scale[:, None]
    return float(
        np.max(np.abs(normalized_factor @ normalized_factor.T - correlation))
    )


def _build_gx_score_contract(
    probe: str,
    product: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract one gX block and construct its gX-only score compressor."""

    if probe not in GX_SPECS:
        raise ValueError(f"unsupported gX probe {probe!r}")
    spec = GX_SPECS[probe]
    indices = np.asarray(spec["indices"], dtype=np.int64)
    paired_probe = str(spec["paired_probe"])
    arrays = product["arrays"]
    pair_records = product["manifest"].get("pairs", {})
    if paired_probe not in pair_records:
        raise KeyError(f"joint manifest has no pair record {paired_probe!r}")
    pair_record = pair_records[paired_probe]

    n_rows = int(training["n_rows"])
    n_points = int(training["n_points"])
    replicas = int(training["replicas"])
    expected_dof = n_points * (replicas - 1)
    if int(pair_record.get("covariance_degrees_of_freedom", -1)) != expected_dof:
        raise ValueError(f"{paired_probe} has the wrong covariance degrees of freedom")
    if pair_record.get("positive_definite_without_repair") is not True:
        raise ValueError(f"{paired_probe} is not marked positive definite without repair")
    if pair_record.get("jitter") is not None:
        raise ValueError(f"{paired_probe} declares forbidden covariance jitter")
    if pair_record.get("eigenvalue_floor") is not None:
        raise ValueError(f"{paired_probe} declares a forbidden eigenvalue floor")
    if pair_record.get("gx_source_indices") != indices.tolist():
        raise ValueError(f"{paired_probe} does not use the canonical {probe} columns")

    gx = _require_finite_shape(arrays, "gx", (n_rows, 42))
    gx_observation = _require_finite_shape(arrays, "gx_data_vector", (42,))
    paired_rows = _require_finite_shape(
        arrays, f"x_{paired_probe}", (n_rows, 28)
    )
    paired_observation = _require_finite_shape(
        arrays, f"data_vector_{paired_probe}", (28,)
    )
    paired_covariance = np.asarray(
        _require_finite_shape(
            arrays, f"covariance_{paired_probe}", (28, 28)
        ),
        dtype=np.float64,
    )
    paired_jacobian = np.asarray(
        _require_finite_shape(
            arrays, f"affine_jacobian_{paired_probe}", (28, 5)
        ),
        dtype=np.float64,
    )
    paired_reference = np.asarray(
        _require_finite_shape(
            arrays, f"affine_reference_prediction_{paired_probe}", (28,)
        ),
        dtype=np.float64,
    )

    rows = np.asarray(gx[:, indices], dtype=np.float64)
    observation = np.asarray(gx_observation[indices], dtype=np.float64)
    if not np.array_equal(rows, paired_rows[:, :14]):
        raise ValueError(
            f"saved {probe} rows differ from the gX half of {paired_probe}"
        )
    if not np.array_equal(observation, paired_observation[:14]):
        raise ValueError(
            f"saved {probe} observation differs from the gX half of {paired_probe}"
        )

    covariance = np.asarray(paired_covariance[:14, :14], dtype=np.float64)
    if not np.array_equal(covariance, covariance.T):
        raise ValueError(f"{probe} covariance is not exactly symmetric")
    jacobian = np.asarray(paired_jacobian[:14], dtype=np.float64)
    reference = np.asarray(paired_reference[:14], dtype=np.float64)
    score = g2pilot._score_payload(
        covariance,
        jacobian,
        reference,
        np.arange(14, dtype=np.int64),
    )
    cholesky = np.asarray(score["cholesky"], dtype=np.float64)
    if _cholesky_reconstruction_error(covariance, cholesky) > 1.0e-13:
        raise ValueError(f"{probe} Cholesky reconstruction failed")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{probe} covariance is not positive definite")
    if int(score["response_rank"]) < 1:
        raise ValueError(f"{probe} whitened affine response has zero rank")

    observed_score = g2runner._summarise(
        observation[None, :],
        indices=np.arange(14, dtype=np.int64),
        cholesky=cholesky,
        operator=np.asarray(score["operator"], dtype=np.float64),
        reference_prediction=np.asarray(
            score["reference_prediction"], dtype=np.float64
        ),
        compression="score",
    )[0]
    if observed_score.shape != (5,) or not np.all(np.isfinite(observed_score)):
        raise RuntimeError(f"{probe} observed score is invalid")

    return {
        "probe": probe,
        "paired_probe": paired_probe,
        "indices": indices,
        "rows": rows,
        "observation": observation,
        "covariance": covariance,
        "cholesky": cholesky,
        "operator": np.asarray(score["operator"], dtype=np.float64),
        "reference_prediction": np.asarray(
            score["reference_prediction"], dtype=np.float64
        ),
        "observed_score": observed_score,
        "covariance_degrees_of_freedom": expected_dof,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "response_rank": int(score["response_rank"]),
        "response_singular_values": np.asarray(
            score["response_singular_values"], dtype=np.float64
        ),
    }


def _run_self_test() -> int:
    rng = np.random.default_rng(20260910)
    n_points, replicas = 40, 5
    n_rows = n_points * replicas
    point_u = rng.normal(size=(n_points, 5))
    point_index = np.repeat(np.arange(n_points), replicas)

    gx_intercept = rng.normal(size=42)
    gx_jacobian = rng.normal(size=(42, 5))
    g2_intercept = rng.normal(size=42)
    g2_jacobian = rng.normal(size=(42, 5))
    gx_signal = gx_intercept[None, :] + point_u @ gx_jacobian.T
    g2_signal = g2_intercept[None, :] + point_u @ g2_jacobian.T
    common_noise = rng.normal(scale=0.3, size=(n_rows, 42))
    gx_rows = gx_signal[point_index] + common_noise
    g2_rows = (
        g2_signal[point_index]
        + 0.35 * common_noise
        + rng.normal(scale=0.3, size=(n_rows, 42))
    )
    gx_observation = gx_intercept.copy()
    g2_observation = g2_intercept.copy()
    u_reference = np.zeros(5, dtype=np.float64)

    arrays: dict[str, Any] = {
        "gx": gx_rows,
        "gx_data_vector": gx_observation,
    }
    pair_manifest: dict[str, Any] = {}
    for paired_probe in joint_builder.PAIR_SPECS:
        pair_arrays, pair_metadata = joint_builder._joint_payload(
            paired_probe,
            gx_rows,
            g2_rows,
            point_index,
            point_u,
            gx_observation,
            g2_observation,
            u_reference,
        )
        pair_manifest[paired_probe] = pair_metadata
        for name, value in pair_arrays.items():
            arrays[f"{name}_{paired_probe}"] = value

    training = {
        "n_rows": n_rows,
        "n_points": n_points,
        "replicas": replicas,
    }
    product = {"arrays": arrays, "manifest": {"pairs": pair_manifest}}
    for probe, spec in GX_SPECS.items():
        contract = _build_gx_score_contract(probe, product, training)
        assert contract["rows"].shape == (n_rows, 14)
        assert contract["observation"].shape == (14,)
        assert contract["covariance"].shape == (14, 14)
        assert contract["operator"].shape == (5, 14)
        assert contract["observed_score"].shape == (5,)
        assert contract["covariance_degrees_of_freedom"] == n_points * (
            replicas - 1
        )

        summaries = g2runner._summarise(
            contract["rows"],
            indices=np.arange(14, dtype=np.int64),
            cholesky=contract["cholesky"],
            operator=contract["operator"],
            reference_prediction=contract["reference_prediction"],
            compression="score",
        )
        assert summaries.shape == (n_rows, 5)

        # Null control: changing either unselected gX block cannot affect the
        # selected rows or their score summary.
        changed_arrays = dict(arrays)
        changed_gx = gx_rows.copy()
        unselected = np.setdiff1d(np.arange(42), spec["indices"])
        changed_gx[:, unselected] += 1.0e12
        changed_arrays["gx"] = changed_gx
        changed_product = {
            "arrays": changed_arrays,
            "manifest": {"pairs": pair_manifest},
        }
        changed = _build_gx_score_contract(probe, changed_product, training)
        np.testing.assert_array_equal(contract["rows"], changed["rows"])
        np.testing.assert_array_equal(
            contract["observed_score"], changed["observed_score"]
        )

    # Negative control: selected-row corruption must be rejected against the
    # saved paired vector.
    broken_arrays = dict(arrays)
    broken_gx = gx_rows.copy()
    broken_gx[[0, replicas], 0] = broken_gx[[replicas, 0], 0]
    broken_arrays["gx"] = broken_gx
    try:
        _build_gx_score_contract(
            "gy",
            {"arrays": broken_arrays, "manifest": {"pairs": pair_manifest}},
            training,
        )
    except ValueError as error:
        if "gX half" not in str(error):
            raise AssertionError("row-corruption guard raised the wrong error") from error
    else:
        raise AssertionError("selected-row corruption was not rejected")

    all_seeds = {
        _seeded(probe, seed, stage)
        for probe in GX_SPECS
        for seed in range(4)
        for stage in (1, 2)
    }
    assert len(all_seeds) == len(GX_SPECS) * 4 * 2

    gy_training, gy_mcmc, gy_samples = _settings_for_probe("gy", None)
    assert gy_training == GY_TRAINING
    assert gy_mcmc == GY_MCMC
    assert gy_samples == GY_DEFAULT_POSTERIOR_SAMPLES
    for unchanged_probe in ("gkappa", "gtau"):
        training, mcmc, samples = _settings_for_probe(unchanged_probe, None)
        assert training == TRAINING
        assert mcmc == MCMC
        assert samples == DEFAULT_POSTERIOR_SAMPLES
    _, _, explicit_samples = _settings_for_probe("gy", 12_345)
    assert explicit_samples == 12_345

    midpoint = g2runner._theta_from_u(np.zeros((2, 5), dtype=np.float64))
    np.testing.assert_allclose(
        midpoint,
        np.repeat((0.5 * (PRIOR_LOW + PRIOR_HIGH))[None, :], 2, axis=0),
    )
    print(
        "SELF-TEST PASS: canonical gX slices, 14D covariance/5D score, "
        "unselected-block null controls, row-corruption rejection, unique "
        "training/sampling seeds, gy-only longer budget, unchanged kappa/tau "
        "budgets, and prior transform"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--g2x-training-set", type=pathlib.Path)
    parser.add_argument("--g2x-contract", type=pathlib.Path)
    parser.add_argument("--joint-product", type=pathlib.Path)
    parser.add_argument("--probe", choices=tuple(GX_SPECS))
    parser.add_argument(
        "--output", type=pathlib.Path, help="explicit new posterior .npz to create"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="must be 4, matching the g2X and paired pilot ensemble",
    )
    parser.add_argument("--compression", choices=("score", "raw"), default="score")
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=None,
        help=(
            "retained samples TOTAL PER NETWORK; defaults to 100000 for gy "
            "and 20000 for gkappa/gtau"
        ),
    )
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
    if args.posterior_samples is not None and args.posterior_samples <= 0:
        parser.error("--posterior-samples must be positive")

    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if output.exists() and not (args.overwrite or args.preflight):
        parser.error(f"output exists; pass --overwrite to replace it: {output}")

    started = time.time()
    import mock_sbi_common as msc

    print("[1/5] loading the exact saved gX rows and row contract ...", flush=True)
    training = joint_builder._load_g2x_rows(args.g2x_training_set, msc)
    if int(training["n_points"]) != EXPECTED_POINTS:
        raise ValueError(
            f"this dated pilot requires {EXPECTED_POINTS} points; "
            f"the training product contains {training['n_points']}"
        )
    g2_contract = g2runner._load_contract(args.g2x_contract, training, msc)
    product = joint_runner._load_joint_product(
        args.joint_product, training, g2_contract, msc
    )
    probe = str(args.probe)
    training_settings, mcmc_settings, posterior_samples = _settings_for_probe(
        probe, args.posterior_samples
    )
    gx_contract = _build_gx_score_contract(probe, product, training)
    print(
        f"      {probe}: columns {gx_contract['indices'].tolist()}; "
        f"{training['n_points']} points x {training['replicas']} exact replicas",
        flush=True,
    )

    print("[2/5] applying the gX-only covariance and score compression ...", flush=True)
    gx_rows = np.asarray(gx_contract["rows"], dtype=np.float64)
    finite = np.all(np.isfinite(training["u"]), axis=1) & np.all(
        np.isfinite(gx_rows), axis=1
    )
    if not np.all(finite):
        print(
            f"      dropping {int(np.sum(~finite))} diagnostic row(s) with no "
            "finite probit coordinate",
            flush=True,
        )
    u_rows = np.asarray(training["u"][finite], dtype=np.float64)
    x_rows = gx_rows[finite]
    local_indices = np.arange(14, dtype=np.int64)
    summaries = g2runner._summarise(
        x_rows,
        indices=local_indices,
        cholesky=gx_contract["cholesky"],
        operator=gx_contract["operator"],
        reference_prediction=gx_contract["reference_prediction"],
        compression=args.compression,
    )
    observation = np.asarray(gx_contract["observation"], dtype=np.float64)
    observed = g2runner._summarise(
        observation[None, :],
        indices=local_indices,
        cholesky=gx_contract["cholesky"],
        operator=gx_contract["operator"],
        reference_prediction=gx_contract["reference_prediction"],
        compression=args.compression,
    )[0]
    if not np.all(np.isfinite(summaries)) or not np.all(np.isfinite(observed)):
        raise RuntimeError("compressed gX training set or observation is non-finite")
    expected_dimension = 5 if args.compression == "score" else 14
    if summaries.shape != (x_rows.shape[0], expected_dimension):
        raise RuntimeError(f"gX summary has unexpected shape {summaries.shape}")
    if args.compression == "score" and not np.array_equal(
        observed, gx_contract["observed_score"]
    ):
        raise RuntimeError("replayed observed gX score differs from the runtime contract")
    print(
        f"      {x_rows.shape[0]} rows: gX-only 14D covariance -> "
        f"{summaries.shape[1]}D {args.compression} summary",
        flush=True,
    )

    if args.preflight:
        print("\nPREFLIGHT PASS")
        print(f"      analysis status: {ANALYSIS_STATUS}")
        print(f"      probe: {probe}")
        print(f"      source columns: {gx_contract['indices'].tolist()}")
        print("      input: exact saved gX rows; no map/noise regeneration")
        print("      historical code-source hash checks: not invoked")
        print("      covariance: gX-only 14x14 principal block; no repair")
        print(
            "      covariance degrees of freedom: "
            f"{gx_contract['covariance_degrees_of_freedom']}"
        )
        print(f"      compression: {args.compression}")
        print(f"      training rows: {x_rows.shape[0]}")
        print(f"      network seeds: {args.seeds}")
        print(f"      training settings: {training_settings}")
        print(f"      MCMC settings: {mcmc_settings}")
        print(f"      posterior samples per seed: {posterior_samples}")
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
        estimator = inference.train(
            **training_settings, show_train_summary=(probe == "gy")
        )
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
        torch.manual_seed(sampling_seed)
        draws = np.asarray(
            posterior.sample(
                (posterior_samples,),
                x=observed_torch,
                show_progress_bars=False,
            ),
            dtype=np.float64,
        )
        if draws.shape != (posterior_samples, 5) or not np.all(
            np.isfinite(draws)
        ):
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

    print("[5/5] writing the explicitly labelled gX-only pilot ...", flush=True)
    manifest = {
        "schema_version": POSTERIOR_SCHEMA,
        "analysis_status": ANALYSIS_STATUS,
        "scientific_scope": (
            "exploratory 104-point/five-replica gX-only check; not a "
            "validated or quotable Stage-7 posterior"
        ),
        "probe": probe,
        "paired_product_probe": gx_contract["paired_probe"],
        "probe_indices": gx_contract["indices"].tolist(),
        "input_vector_order": f"{probe}[14]",
        "input_dimension": 14,
        "row_policy": (
            "exact gX rows already saved by the matched product; no new noise "
            "realizations and no map regeneration"
        ),
        "covariance_policy": (
            "gX-only 14x14 principal block of the saved paired within-point "
            "covariance; g2X and the gX-g2X cross block excluded"
        ),
        "covariance_degrees_of_freedom": gx_contract[
            "covariance_degrees_of_freedom"
        ],
        "covariance_repair": "none",
        "minimum_covariance_eigenvalue": gx_contract["minimum_eigenvalue"],
        "maximum_covariance_eigenvalue": gx_contract["maximum_eigenvalue"],
        "covariance_condition_number": gx_contract["condition_number"],
        "response_rank_after_whitening": gx_contract["response_rank"],
        "compression": args.compression,
        "summary_dimension": int(summaries.shape[1]),
        "estimator": "SNLE (NLE, one round, NSF)",
        "proposal_correction_required": False,
        "network_seeds": args.seeds,
        "posterior_samples_per_seed": posterior_samples,
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
        "historical_code_source_hashes_enforced": False,
        "numerical_product_integrity_checks_enforced": True,
        "observation_kind": "pasted_map_measurement_noiseless_gx",
        "observation_vector_sha256": msc.sha256_array(observation),
        "covariance_sha256": msc.sha256_array(gx_contract["covariance"]),
        "observed_summary": observed.tolist(),
        "observed_summary_norm": float(np.linalg.norm(observed)),
        "training_rows_sha256": msc.sha256_array(x_rows),
        "training_summary_sha256": msc.sha256_array(summaries),
        "parameter_names": list(PARAMETER_NAMES),
        "theta_reference": np.asarray(product["arrays"]["theta_reference"]).tolist(),
        "u_reference": np.asarray(product["arrays"]["u_reference"]).tolist(),
        "prior_low": PRIOR_LOW.tolist(),
        "prior_high": PRIOR_HIGH.tolist(),
        "training": training_settings,
        "mcmc": mcmc_settings,
        "device": "cpu",
        "namespace": list(NAMESPACE),
        "elapsed_seconds": time.time() - started,
    }
    payload: dict[str, Any] = {
        "u": pooled_u,
        "theta": pooled_theta,
        "theta_reference": np.asarray(
            product["arrays"]["theta_reference"], dtype=np.float64
        ),
        "u_reference": np.asarray(
            product["arrays"]["u_reference"], dtype=np.float64
        ),
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
