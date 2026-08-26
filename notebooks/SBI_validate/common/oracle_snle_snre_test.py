#!/usr/bin/env python3
"""Oracle test of SNLE and SNRE on mock simulations: no proposal correction at all.

Direct NPE failed here in two distinct ways, both traced to the proposal correction:

*   SNPE-C's atomic loss produced ``NaN/Inf in the evaluation of the MoG proposal
    posterior`` at round 2 in both compressions -- the same failure the theory campaign
    hit at round 3 with 65,536 simulations per round, so not a count problem.
*   Replacing it with explicit ``p0/q`` importance reweighting ran, but the correction
    degraded exactly as the rounds sharpened: effective sample size 70% -> 1.3% and
    Pareto k +0.51 -> +1.09 between rounds 2 and 3, which makes the round-3 posterior an
    artifact rather than a posterior.

SNLE and SNRE remove that failure mode structurally rather than by tuning.  They learn
``p(x|theta)`` or the likelihood ratio, and the prior is applied analytically when the
posterior is sampled.  sbi's ``append_simulations`` for these methods takes no
``proposal`` argument at all: the design distribution affects only *where the
likelihood is accurate*, never the correctness of the target.  So sequential rounds can
sharpen without invalidating anything.

The price is that sampling needs MCMC rather than a direct draw, which in five
dimensions is cheap.

Everything else is held fixed against the other arms: same stand-in simulator, same
observation (the production contract's data vector), same exact reference (the archived
10,000-sample NUTS chain), same capped x2 proposal for round 1, same gates.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import torch

_PROCESS_STARTED = time.time()
from torch.distributions import Independent, Normal

THIS_DIR = pathlib.Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[2], THIS_DIR.parents[2] / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mock_sbi_common as msc
from oracle_direct_npe_test import inflate_covariance, seeded

NAMESPACE = (20260824, 903, 0)
from oracle_paste_budget_test import GATE, batched_predict, compare_posteriors, posterior_summary
from three_probe_agreement_common import (
    GRID, backend_manifest, build_problem, compress, score_operator, sha256_array,
)

NAMESPACE_BASE = (20260824, 903)
MCMC = dict(method="slice_np_vectorized", num_chains=20, warmup_steps=250, thin=1)


def robust_scale(samples: np.ndarray) -> np.ndarray:
    """Normalized IQR per parameter: a width statistic with bounded tail sensitivity.

    A plain standard deviation on these posteriors is not reproducible -- the emulator
    arm gave width changes of +4.356 and +0.300 at the same N with different sampler
    seeds, because rare tail excursions dominate the second moment.  The interquartile
    range divided by 1.349 equals the standard deviation for a Gaussian and is
    insensitive to the tails, so it is reported alongside.
    """

    q75, q25 = np.percentile(samples, [75, 25], axis=0)
    return (q75 - q25) / 1.3489795


def compare_robust(reference: np.ndarray, trial: np.ndarray) -> dict:
    ref_scale, trial_scale = robust_scale(reference), robust_scale(trial)
    ref_med, trial_med = np.median(reference, axis=0), np.median(trial, axis=0)
    return {
        "median_drift_robust_sigma": np.abs((trial_med - ref_med) / ref_scale).tolist(),
        "max_median_drift_robust_sigma": float(np.max(np.abs((trial_med - ref_med) / ref_scale))),
        "robust_width_relative_change": ((trial_scale - ref_scale) / ref_scale).tolist(),
        "max_abs_robust_width_relative_change": float(
            np.max(np.abs((trial_scale - ref_scale) / ref_scale))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("snle", "snre"), required=True)
    parser.add_argument("--compression", choices=("score", "raw"), default="score")
    parser.add_argument("--theta-per-round", type=int, nargs="+", default=[96, 96, 96])
    parser.add_argument("--noise-per-theta", type=int, default=64)
    parser.add_argument("--inflation", type=float, default=2.0)
    parser.add_argument("--posterior-samples", type=int, default=20000)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi")
    parser.add_argument("--max-wall-seconds", type=float, default=None,
                        help="Hard budget for this arm.  A round cannot be stopped part "
                             "way, so the runner declines to START a round it projects "
                             "cannot finish, and exits cleanly with the ladder it did "
                             "complete.  This makes the job's wall time a guarantee "
                             "rather than an estimate.")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Shift every PRNG stream. Repeating an arm with a different "
                             "offset measures whether its result is reproducible -- which "
                             "the emulator arm was not (width +4.356 vs +0.300 at the same "
                             "N on different sampler seeds).")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global NAMESPACE
    NAMESPACE = NAMESPACE_BASE + (args.seed_offset,)
    tag = args.tag or f"{args.estimator}_{args.compression}_s{args.seed_offset}"
    started = time.time()

    reference_chain = np.load(
        msc.REPO_ROOT / "data/SBI_validate/three_probe_inference/hmc_v2/run01/hmc_samples.npz"
    )["u"].reshape(-1, 5)
    proposal_mean = reference_chain.mean(axis=0)
    proposal_covariance = inflate_covariance(np.cov(reference_chain, rowvar=False),
                                             args.inflation, "capped")
    print(f"[1/5] {args.estimator.upper()} on {args.compression}; round-1 design = theory "
          f"posterior inflated x{args.inflation} (capped at the prior)")
    print(f"      proposal sd {np.round(np.sqrt(np.diag(proposal_covariance)), 3)}")
    print(f"      no proposal correction is needed: the prior is applied at sampling time")

    print("[2/5] building the stand-in simulator ...", flush=True)
    problem = build_problem(contract_path=msc.INFERENCE_CONTRACT_PATH)
    cholesky = problem.cholesky
    reference_point = json.loads(
        (msc.REPO_ROOT / "data/SBI_validate/three_probe_inference/reference_point_v2.json").read_text())
    payload = score_operator(problem, np.asarray(reference_point["u_map"], dtype=np.float64))
    operator = np.asarray(payload["operator"], dtype=np.float64)
    reference_prediction = np.asarray(payload["reference_prediction"], dtype=np.float64)

    def summarise(vectors: np.ndarray) -> np.ndarray:
        if args.compression == "raw":
            return np.linalg.solve(cholesky, (vectors - reference_prediction[None, :]).T).T
        return compress(operator, cholesky, reference_prediction, vectors)

    observed = summarise(problem.observation[None, :])[0]
    print(f"      summary dimension {observed.size}")

    def simulate(u_theta: np.ndarray, noise_seed: int):
        prediction = batched_predict(problem, u_theta, chunk=16)
        rng = np.random.default_rng(noise_seed)
        whitened_signal = np.linalg.solve(
            cholesky, (prediction - reference_prediction[None, :]).T).T
        repeats = args.noise_per_theta
        rows = np.repeat(whitened_signal, repeats, axis=0)
        rows = rows + rng.standard_normal(rows.shape)
        theta_rows = np.repeat(u_theta, repeats, axis=0)
        return theta_rows, (rows if args.compression == "raw" else (operator @ rows.T).T)

    prior = Independent(Normal(torch.zeros(5), torch.ones(5)), 1)
    if args.estimator == "snle":
        from sbi.inference import SNLE
        inference = SNLE(prior=prior, density_estimator="nsf", device="cpu",
                         show_progress_bars=False)
    else:
        from sbi.inference import SNRE_B
        inference = SNRE_B(prior=prior, classifier="resnet", device="cpu",
                           show_progress_bars=False)

    observed_torch = torch.as_tensor(observed, dtype=torch.float32)
    posterior = None
    rounds = []
    total = 0
    stopped_early = None
    # Finalisation (writing the JSON) is cheap, but leave room so a truncated arm still
    # records what it measured.
    reserve = 60.0 if args.max_wall_seconds is None else min(60.0, 0.05 * args.max_wall_seconds)
    for index, count in enumerate(args.theta_per_round):
        number = index + 1
        if args.max_wall_seconds is not None and rounds:
            elapsed = time.time() - _PROCESS_STARTED
            last = rounds[-1]
            # Cost is dominated by training, which scales with the cumulative row count.
            growth = (total + count) / max(total, 1)
            projected = elapsed + last["round_seconds"] * growth + reserve
            if projected > args.max_wall_seconds:
                stopped_early = {
                    "before_round": number,
                    "elapsed_seconds": elapsed,
                    "projected_seconds": projected,
                    "budget_seconds": args.max_wall_seconds,
                    "last_round_seconds": last["round_seconds"],
                    "assumed_growth": growth,
                }
                print(f"      stopping before round {number}: {elapsed:.0f}s elapsed, "
                      f"projected {projected:.0f}s against a {args.max_wall_seconds:.0f}s "
                      f"budget", flush=True)
                break
        round_started = time.time()
        print(f"[3/5] round {number}: {count} distinct expensive points "
              f"x {args.noise_per_theta} noise draws", flush=True)
        if number == 1:
            u_theta = np.random.default_rng(seeded(NAMESPACE, 1, 1)).multivariate_normal(
                proposal_mean, proposal_covariance, size=count)
        else:
            u_theta = np.asarray(posterior.sample((count,), x=observed_torch,
                                                  show_progress_bars=False), dtype=np.float64)
        total += count
        theta_rows, x_rows = simulate(u_theta, seeded(NAMESPACE, number, 2))
        if not np.all(np.isfinite(x_rows)):
            raise RuntimeError(f"round {number}: non-finite summaries")

        # No `proposal` argument exists for SNLE/SNRE: the design distribution changes
        # only where the likelihood is accurate, not what is being estimated.
        inference.append_simulations(torch.as_tensor(theta_rows, dtype=torch.float32),
                                     torch.as_tensor(x_rows, dtype=torch.float32),
                                     from_round=index)
        torch.manual_seed(seeded(NAMESPACE, number, 3))
        try:
            estimator = inference.train(training_batch_size=512, learning_rate=5e-4,
                                        validation_fraction=0.10, stop_after_epochs=25,
                                        max_num_epochs=500, show_train_summary=False)
        except (AssertionError, ValueError) as error:
            print(f"      round {number} TRAINING FAILED: {error!r}")
            rounds.append({"round": number, "status": "training_failed", "error": repr(error),
                           "cumulative_expensive_points": total})
            break

        posterior = inference.build_posterior(
            estimator, prior=prior, sample_with="mcmc", mcmc_method=MCMC["method"],
            mcmc_parameters=dict(num_chains=MCMC["num_chains"],
                                 warmup_steps=MCMC["warmup_steps"], thin=MCMC["thin"]))
        posterior.set_default_x(observed_torch)
        t0 = time.time()
        samples = np.asarray(posterior.sample((args.posterior_samples,), x=observed_torch,
                                              show_progress_bars=False), dtype=np.float64)
        mcmc_seconds = time.time() - t0

        moment = compare_posteriors(reference_chain, samples)
        robust = compare_robust(reference_chain, samples)
        record = {
            "round": number, "status": "trained",
            "round_seconds": time.time() - round_started,
            "distinct_expensive_points_this_round": count,
            "cumulative_expensive_points": total,
            "training_rows_cumulative": int(sum(
                r.get("training_rows_this_round", 0) for r in rounds) + theta_rows.shape[0]),
            "training_rows_this_round": int(theta_rows.shape[0]),
            "mcmc_seconds": mcmc_seconds,
            "posterior": moment, "posterior_robust": robust,
            "posterior_summary": posterior_summary(samples),
            "posterior_robust_scale": robust_scale(samples).tolist(),
        }
        record["gate"] = {
            "posterior_mean_drift_sigma": moment["max_mean_drift_sigma"] <= GATE["posterior_mean_drift_sigma"],
            "posterior_width_relative_change": moment["max_abs_width_relative_change"] <= GATE["posterior_width_relative_change"],
            "posterior_correlation_change": moment["max_abs_correlation_change"] <= GATE["posterior_correlation_change"],
        }
        record["gate_robust"] = {
            "median_drift": robust["max_median_drift_robust_sigma"] <= GATE["posterior_mean_drift_sigma"],
            "robust_width": robust["max_abs_robust_width_relative_change"] <= GATE["posterior_width_relative_change"],
        }
        record["gate_passed"] = all(record["gate"].values())
        record["gate_robust_passed"] = all(record["gate_robust"].values())
        rounds.append(record)
        print(f"      cumulative {total:4d} pts | MOMENT drift {moment['max_mean_drift_sigma']:6.3f}s "
              f"width {moment['max_abs_width_relative_change']:+7.3f} corr {moment['max_abs_correlation_change']:6.3f}"
              f"  | ROBUST drift {robust['max_median_drift_robust_sigma']:6.3f}s "
              f"width {robust['max_abs_robust_width_relative_change']:+7.3f}"
              f"  round {record['round_seconds']:.0f}s (mcmc {mcmc_seconds:.0f}s)  "
              f"{'PASS' if record['gate_passed'] else 'fail'}"
              f"/{'PASS' if record['gate_robust_passed'] else 'fail'}", flush=True)
        print(f"        sd        {[round(v,3) for v in record['posterior_summary']['sd']]}"
              f"  vs exact {[round(v,3) for v in reference_chain.std(0,ddof=1)]}")
        print(f"        robust sd {[round(v,3) for v in record['posterior_robust_scale']]}"
              f"  vs exact {[round(v,3) for v in robust_scale(reference_chain)]}")

    passing = [r["cumulative_expensive_points"] for r in rounds if r.get("gate_passed")]
    robust_passing = [r["cumulative_expensive_points"] for r in rounds if r.get("gate_robust_passed")]
    out_payload = {
        "status": "PASS" if passing else "FAIL",
        "estimator": args.estimator, "compression": args.compression,
        "smallest_passing_expensive_budget": min(passing) if passing else None,
        "smallest_robust_passing_expensive_budget": min(robust_passing) if robust_passing else None,
        "proposal_correction_required": False,
        "proposal": {"source": "archived theory NUTS chain hmc_v2/run01",
                     "inflation_sd": args.inflation, "mode": "capped",
                     "sd": np.sqrt(np.diag(proposal_covariance)).tolist(),
                     "volume_relative_to_prior": float(np.sqrt(np.linalg.det(proposal_covariance)))},
        "gate_thresholds": {k: GATE[k] for k in
                            ("posterior_mean_drift_sigma", "posterior_width_relative_change",
                             "posterior_correlation_change")},
        "stopped_early": stopped_early,
        "max_wall_seconds": args.max_wall_seconds,
        "mcmc": MCMC, "noise_per_theta": args.noise_per_theta,
        "seed_offset": args.seed_offset, "namespace": list(NAMESPACE),
        "theta_per_round": args.theta_per_round, "rounds": rounds,
        "identity": {"grid": list(GRID), "backend": backend_manifest(),
                     "contract_sha256": problem.contract.contract_sha256,
                     "observed_summary_sha256": sha256_array(observed)},
        "elapsed_seconds": time.time() - started,
    }
    out = args.output_dir / f"oracle_{tag}.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"\nstatus {out_payload['status']}  moment-gate budget "
          f"{out_payload['smallest_passing_expensive_budget']}  robust-gate budget "
          f"{out_payload['smallest_robust_passing_expensive_budget']}  wrote {out}")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
