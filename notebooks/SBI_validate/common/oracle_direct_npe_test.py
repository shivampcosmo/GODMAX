#!/usr/bin/env python3
"""Oracle test of DIRECT sequential NPE on mock simulations, with no emulator.

The question: can a few hundred pasted parameter points, each augmented with many
free noise draws, give the mock posterior directly through NPE -- no transfer
function, no GP, no emulator uncertainty?

This tests the architecture for free, using the analytic theory model (21 ms) as a
stand-in for the paste.  The observation is the production contract's data vector, so
the exact target posterior is the archived 10,000-sample NUTS chain of that same
stand-in model on that same observation.  Any disagreement is therefore the inference
machinery, not model misspecification.

Why the inflated theory proposal is load-bearing, quantitatively.  In probit
coordinates the prior is exactly N(0, I).  The theory posterior's covariance has
eigen-standard-deviations [0.044, 0.197, 0.952, 1.111, 1.205]: only two of five
directions are constrained, and the tightest is 23x narrower than the prior.  Volume
alone understates the effect -- x2 inflation shrinks the volume only 2.8x -- but the
*yield* is what matters: about 1.8% of prior draws land inside the posterior's 2-sigma
region, so a 288-point prior design would contain roughly 5 useful points against
roughly 288 from the inflated proposal.

Prior handling.  The inflated proposal is installed as sbi's ``prior``, so round 1 uses
the standard loss and rounds 2+ use SNPE-C's atomic correction toward that same
reference measure.  The reported posterior is then reweighted analytically from the
inflated proposal to the true N(0, I) prior, which is exact because both densities are
known in closed form.  This deliberately avoids passing a hand-built Gaussian as a
``proposal`` object: sbi types that argument as a DirectPosterior, and the atomic loss
was already the fragile part of the theory campaign (deterministic NaN at round 3 with
65,536 simulations per round).
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

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.distributions import Independent, MultivariateNormal, Normal

THIS_DIR = pathlib.Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[2], THIS_DIR.parents[2] / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mock_sbi_common as msc
from oracle_paste_budget_test import (
    GATE, batched_predict, compare_posteriors, posterior_summary,
)
from three_probe_agreement_common import (
    GRID, backend_manifest, build_problem, compress, importance_diagnostics, pareto_k,
    score_operator, sha256_array,
)

PARAMETER_NAMES = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")
TRAINING = dict(hidden_features=128, num_components=10, training_batch_size=512,
                max_num_epochs=500, stop_after_epochs=25,
                validation_fraction=0.10, learning_rate=5e-4)
NAMESPACE = (20260824, 901)


def seeded(entropy, *spawn) -> int:
    return int(np.random.SeedSequence(tuple(int(v) for v in entropy) + tuple(int(v) for v in spawn))
               .generate_state(1, dtype=np.uint32)[0])


def log_normal_density(u: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    delta = u - np.asarray(mean, dtype=np.float64)[None, :]
    factor = np.linalg.cholesky(covariance)
    solved = np.linalg.solve(factor, delta.T)
    quad = np.einsum("ij,ij->j", solved, solved)
    log_det = 2.0 * np.sum(np.log(np.diag(factor)))
    return -0.5 * quad - 0.5 * log_det - 0.5 * u.shape[1] * np.log(2.0 * np.pi)


def inflate_covariance(covariance: np.ndarray, inflation: float, mode: str) -> np.ndarray:
    """Inflate a posterior covariance for use as a design proposal.

    ``naive`` scales every direction, which is what one writes first but which pushes
    the proposal past the prior wherever the posterior was already prior-width.  Here
    the theory posterior's eigen-standard-deviations are
    [0.044, 0.197, 0.952, 1.111, 1.205] against a prior sd of 1, so a x2 naive
    inflation exceeds the prior in three of five directions.  Two costs follow: draws
    land outside the prior, and the p0/q importance weights collapse there (measured
    5.1% effective sample size).

    ``capped`` inflates in the eigenbasis and clips each eigen-sd at the prior's 1.0.
    It keeps the full x2 safety margin exactly where the posterior is informative and
    falls back to the prior where it is not.  Volume goes from 0.355 to about 0.035 of
    the prior, so it is also the tighter design.
    """

    values, vectors = np.linalg.eigh(covariance)
    sd = np.sqrt(np.maximum(values, 0.0)) * float(inflation)
    if mode == "capped":
        sd = np.minimum(sd, 1.0)
    return vectors @ np.diag(sd ** 2) @ vectors.T


def log_mixture_density(u: np.ndarray, components) -> np.ndarray:
    """log of a normalized Gaussian mixture, for the pooled proposal."""

    parts = np.stack([np.log(w) + log_normal_density(u, m, c) for w, m, c in components])
    top = parts.max(axis=0)
    return top + np.log(np.exp(parts - top[None, :]).sum(axis=0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi")
    parser.add_argument("--theta-per-round", type=int, nargs="+", default=[96, 96, 96],
                        help="distinct EXPENSIVE parameter points per round")
    parser.add_argument("--noise-per-theta", type=int, default=64,
                        help="free noise draws per pasted point; the training set is the product")
    parser.add_argument("--inflation", type=float, default=2.0, help="proposal sd inflation")
    parser.add_argument("--proposal-mode", choices=("naive", "capped"), default="capped",
                        help="'naive' inflates every direction by --inflation. 'capped' does "
                             "the same in the covariance eigenbasis but never exceeds the "
                             "prior's own sd of 1: inflating past the prior wastes draws "
                             "outside it AND collapses the p0/q importance weights, which "
                             "measured 5.1% ESS and turned a too-wide posterior into a "
                             "too-narrow one.")
    parser.add_argument("--mode", choices=("snpec", "pooled"), default="pooled",
                        help="'snpec' uses sbi's sequential atomic loss (which failed with "
                             "NaN/Inf at round 2 in both compressions). 'pooled' trains plain "
                             "NPE on all rounds pooled and reweights from the analytic pooled "
                             "proposal mixture to the true prior -- no atomic loss anywhere.")
    parser.add_argument("--compression", choices=("score", "raw"), default="score")
    parser.add_argument("--posterior-samples", type=int, default=20000)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.compression}_{'x'.join(map(str, args.theta_per_round))}_m{args.noise_per_theta}"
    started = time.time()

    reference_chain = np.load(
        msc.REPO_ROOT / "data/SBI_validate/three_probe_inference/hmc_v2/run01/hmc_samples.npz"
    )["u"].reshape(-1, 5)
    proposal_mean = reference_chain.mean(axis=0)
    proposal_covariance = inflate_covariance(np.cov(reference_chain, rowvar=False),
                                             args.inflation, args.proposal_mode)
    print(f"[1/6] proposal = theory posterior inflated x{args.inflation} "
          f"({args.proposal_mode}), estimator mode '{args.mode}'")
    print(f"      mean {np.round(proposal_mean, 3)}")
    print(f"      sd   {np.round(np.sqrt(np.diag(proposal_covariance)), 3)}  (prior sd is 1.0 per axis)")
    volume = float(np.sqrt(np.linalg.det(proposal_covariance)))
    print(f"      volume relative to the N(0,I) prior = {volume:.4f}")

    print("[2/6] building the stand-in simulator and the score compression ...", flush=True)
    problem = build_problem(contract_path=msc.INFERENCE_CONTRACT_PATH)
    cholesky = problem.cholesky
    observation = problem.observation
    reference_point = json.loads(
        (msc.REPO_ROOT / "data/SBI_validate/three_probe_inference/reference_point_v2.json").read_text())
    u_map = np.asarray(reference_point["u_map"], dtype=np.float64)
    operator_payload = score_operator(problem, u_map)
    operator = np.asarray(operator_payload["operator"], dtype=np.float64)
    reference_prediction = np.asarray(operator_payload["reference_prediction"], dtype=np.float64)

    def summarise(vectors: np.ndarray) -> np.ndarray:
        if args.compression == "raw":
            return np.linalg.solve(cholesky, (vectors - reference_prediction[None, :]).T).T
        return compress(operator, cholesky, reference_prediction, vectors)

    observed_summary = summarise(observation[None, :])[0]
    dim_x = observed_summary.size
    print(f"      summary dimension {dim_x}  |s| = {np.linalg.norm(observed_summary):.4f}")

    def simulate(u_theta: np.ndarray, noise_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """One expensive evaluation per row, then ``noise_per_theta`` free noise draws.

        This is the whole reason a few hundred pastes can train a network: the noise is
        free and adds exactly linearly, so each expensive point yields as many training
        rows as wanted.  Only the number of DISTINCT rows is budget-limited.
        """
        prediction = batched_predict(problem, u_theta, chunk=16)
        rng = np.random.default_rng(noise_seed)
        repeats = args.noise_per_theta
        whitened_signal = np.linalg.solve(
            cholesky, (prediction - reference_prediction[None, :]).T).T
        theta_rows = np.repeat(u_theta, repeats, axis=0)
        noise = rng.standard_normal((u_theta.shape[0] * repeats, whitened_signal.shape[1]))
        whitened = np.repeat(whitened_signal, repeats, axis=0) + noise
        if args.compression == "raw":
            return theta_rows, whitened
        return theta_rows, (operator @ whitened.T).T

    from sbi.inference import SNPE
    from sbi.utils import posterior_nn

    torch.manual_seed(seeded(NAMESPACE, 0))
    # The inflated proposal is installed as the reference measure; the reported
    # posterior is reweighted back to N(0, I) analytically at the end.
    sbi_prior = MultivariateNormal(torch.as_tensor(proposal_mean, dtype=torch.float32),
                                   covariance_matrix=torch.as_tensor(proposal_covariance,
                                                                     dtype=torch.float32))
    builder = posterior_nn(model="mdn", hidden_features=TRAINING["hidden_features"],
                           num_components=TRAINING["num_components"],
                           z_score_theta="independent", z_score_x="independent")
    inference = SNPE(prior=sbi_prior, density_estimator=builder, device="cpu",
                     show_progress_bars=False)
    observed_torch = torch.as_tensor(observed_summary, dtype=torch.float32)

    # In 'pooled' mode sbi's prior is the TRUE prior and every append passes
    # proposal=None, so sbi always uses the plain NPE loss.  The learned density then
    # approximates p(x|theta) * q_pool(theta), where q_pool is the analytic mixture of
    # every round's proposal; reweighting by p0/q_pool recovers the true posterior
    # exactly.  No atomic loss is ever evaluated.
    if args.mode == "pooled":
        inference = SNPE(prior=Independent(Normal(torch.zeros(5), torch.ones(5)), 1),
                         density_estimator=builder, device="cpu", show_progress_bars=False)

    posterior = None
    rounds = []
    total_expensive = 0
    components = []          # (weight, mean, covariance) of the pooled proposal
    pooled_theta, pooled_x = [], []
    for index, count in enumerate(args.theta_per_round):
        number = index + 1
        print(f"[3/6] round {number}: {count} distinct expensive points "
              f"x {args.noise_per_theta} noise draws", flush=True)
        seed = seeded(NAMESPACE, number, 1)
        if number == 1:
            round_mean, round_cov = proposal_mean, proposal_covariance
        else:
            # Fit the previous round's reweighted posterior and inflate it the same way,
            # so the next round's proposal density stays analytic and normalized.
            previous = np.asarray(rounds[-1]["_samples"], dtype=np.float64)
            round_mean = previous.mean(axis=0)
            round_cov = inflate_covariance(np.cov(previous, rowvar=False),
                                           args.inflation, args.proposal_mode)
        u_theta = np.random.default_rng(seed).multivariate_normal(round_mean, round_cov,
                                                                 size=count)
        components.append((float(count), round_mean, round_cov))
        total_expensive += count

        theta_rows, x_rows = simulate(u_theta, seeded(NAMESPACE, number, 2))
        if not np.all(np.isfinite(x_rows)):
            raise RuntimeError(f"round {number}: non-finite summaries")
        pooled_theta.append(theta_rows)
        pooled_x.append(x_rows)

        if args.mode == "pooled":
            # Retrain from scratch on the pooled set each round: cheap, and it keeps the
            # reference measure exactly the pooled mixture rather than a running mixture
            # that sbi would have to be told about.
            inference = SNPE(prior=Independent(Normal(torch.zeros(5), torch.ones(5)), 1),
                             density_estimator=builder, device="cpu",
                             show_progress_bars=False)
            inference.append_simulations(
                torch.as_tensor(np.concatenate(pooled_theta), dtype=torch.float32),
                torch.as_tensor(np.concatenate(pooled_x), dtype=torch.float32),
                proposal=None)
        else:
            inference.append_simulations(
                torch.as_tensor(theta_rows, dtype=torch.float32),
                torch.as_tensor(x_rows, dtype=torch.float32),
                proposal=None if number == 1 else posterior)

        torch.manual_seed(seeded(NAMESPACE, number, 3))
        try:
            estimator = inference.train(
                training_batch_size=TRAINING["training_batch_size"],
                learning_rate=TRAINING["learning_rate"],
                validation_fraction=TRAINING["validation_fraction"],
                stop_after_epochs=TRAINING["stop_after_epochs"],
                max_num_epochs=TRAINING["max_num_epochs"],
                show_train_summary=False)
        except AssertionError as error:
            print(f"      round {number} TRAINING FAILED: {error!r}")
            rounds.append({"round": number, "status": "training_failed", "error": repr(error),
                           "cumulative_expensive_points": total_expensive})
            break
        posterior = inference.build_posterior(estimator)
        posterior.set_default_x(observed_torch)

        raw = np.asarray(posterior.sample((args.posterior_samples,), x=observed_torch,
                                          show_progress_bars=False), dtype=np.float64)
        total_weight = sum(w for w, _, _ in components)
        mixture = [(w / total_weight, m, c) for w, m, c in components]
        reference_measure = (log_mixture_density(raw, mixture) if args.mode == "pooled"
                             else log_normal_density(raw, proposal_mean, proposal_covariance))
        log_w = log_normal_density(raw, np.zeros(5), np.eye(5)) - reference_measure
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights /= weights.sum()
        effective = float(1.0 / np.sum(weights ** 2))
        pick = np.random.default_rng(seeded(NAMESPACE, number, 4)).choice(
            raw.shape[0], size=min(args.posterior_samples, 20000), replace=True, p=weights)
        reweighted = raw[pick]

        record = {
            "round": number,
            "status": "trained",
            "distinct_expensive_points_this_round": count,
            "cumulative_expensive_points": total_expensive,
            "training_rows_used": int(np.concatenate(pooled_theta).shape[0]
                                      if args.mode == "pooled" else theta_rows.shape[0]),
            "proposal_mean": round_mean.tolist(),
            "proposal_sd": np.sqrt(np.diag(round_cov)).tolist(),
            "proposal_volume_vs_prior": float(np.sqrt(np.linalg.det(round_cov))),
            "reweighting_pareto_k": float(pareto_k(log_w)),
            "reweighting_effective_sample_size": effective,
            "reweighting_ess_fraction": effective / raw.shape[0],
            "posterior_before_reweighting": compare_posteriors(reference_chain, raw),
            "posterior": compare_posteriors(reference_chain, reweighted),
            "posterior_summary": posterior_summary(reweighted),
            "_samples": reweighted,
        }
        p_ = record["posterior"]
        record["gate"] = {
            "posterior_mean_drift_sigma": p_["max_mean_drift_sigma"] <= GATE["posterior_mean_drift_sigma"],
            "posterior_width_relative_change": p_["max_abs_width_relative_change"] <= GATE["posterior_width_relative_change"],
            "posterior_correlation_change": p_["max_abs_correlation_change"] <= GATE["posterior_correlation_change"],
        }
        record["gate_passed"] = all(record["gate"].values())
        rounds.append(record)
        print(f"      cumulative {total_expensive:4d} pts | drift {p_['max_mean_drift_sigma']:6.3f}s  "
              f"width {p_['max_abs_width_relative_change']:+7.3f}  "
              f"corr {p_['max_abs_correlation_change']:6.3f}  "
              f"reweight ESS {100*record['reweighting_ess_fraction']:5.1f}% (k={record['reweighting_pareto_k']:+.2f})  "
              f"{'PASS' if record['gate_passed'] else 'fail'}", flush=True)
        print(f"        posterior sd {[round(v,3) for v in record['posterior_summary']['sd']]}"
              f"  vs exact {[round(v,3) for v in reference_chain.std(0,ddof=1)]}")
        print(f"        width change {[round(v,3) for v in p_['width_relative_change']]}")
        print(f"        mean drift   {[round(v,3) for v in p_['mean_drift_sigma']]}")

    for r in rounds:
        r.pop("_samples", None)

    passing = [r["cumulative_expensive_points"] for r in rounds if r.get("gate_passed")]
    payload = {
        "status": "PASS" if passing else "FAIL",
        "smallest_passing_expensive_budget": min(passing) if passing else None,
        "architecture": "direct sequential NPE on mock simulations; no emulator, no transfer",
        "compression": args.compression,
        "estimator_mode": args.mode,
        "proposal_mode": args.proposal_mode,
        "summary_dimension": int(dim_x),
        "noise_per_theta": args.noise_per_theta,
        "theta_per_round": args.theta_per_round,
        "gate_thresholds": {k: GATE[k] for k in
                            ("posterior_mean_drift_sigma", "posterior_width_relative_change",
                             "posterior_correlation_change")},
        "proposal": {
            "source": "archived theory NUTS chain hmc_v2/run01",
            "inflation_sd": args.inflation,
            "mean": proposal_mean.tolist(),
            "covariance": proposal_covariance.tolist(),
            "volume_relative_to_prior": volume,
            "installed_as": "sbi prior; reported posterior reweighted to N(0,I) analytically",
        },
        "rounds": rounds,
        "reference_posterior": {"source": "hmc_v2/run01", "n_samples": int(reference_chain.shape[0])},
        "identity": {"grid": list(GRID), "backend": backend_manifest(),
                     "contract_sha256": problem.contract.contract_sha256,
                     "observed_summary_sha256": sha256_array(observed_summary)},
        "elapsed_seconds": time.time() - started,
    }
    out = args.output_dir / f"oracle_direct_npe_{tag}.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(f"\nstatus {payload['status']}   smallest passing budget "
          f"{payload['smallest_passing_expensive_budget']}   wrote {out}")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
