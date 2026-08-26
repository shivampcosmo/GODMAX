#!/usr/bin/env python3
"""Score-compressed sequential NPE for the three-probe contract, with an exact
likelihood cross-check computed inside the same job (v2).

Why the previous SBI campaign could not work
--------------------------------------------
The absolute misfit of this observation is large: the whitened chi-square at the
best fit is of order 168 for a nominal ``42 - 5 = 37``.  Conditional NPE
simulates ``x = L^-1(mu(theta) - d) + eps`` and conditions at ``x = 0``, so
reaching the observation requires a ~13-sigma noise cancellation.  The network is
asked for a density where the simulator essentially never lands.  That is why ten
rounds, MDN and NSF, signed ``asinh``, exact probit coordinates and 15-dimensional
PCA all failed: none of them changes the geometry.

The fix used here
-----------------
Compress to the exact 5-dimensional normalized score at the pinned MAP,

    s(x) = G^-1/2 A^T L^-1 (x - mu(u_map)),    A = L^-1 J,  G = A^T A.

At an interior MAP the stationarity condition is ``A^T L^-1 (d - mu) = u_map``,
so the entire absolute misfit lies in the 37-dimensional orthogonal complement
and is projected out.  The observed summary becomes a typical simulator draw and
the noise term has exactly the identity covariance.  The summary dimension now
equals the parameter dimension, which is the natural target for NPE.  PCA failed
at this because it retains high-variance *data* directions -- which is where the
misfit lives -- whereas the score retains parameter-information directions.

Standard-normal probit coordinates additionally mean the prior is unbounded, so
there is no NPE leakage correction and no prior-wall pathology.

Self-validation without touching the HMC
----------------------------------------
After the final round the job builds a Student-t(nu=4) proposal from the NPE
posterior's own moments, evaluates the *exact* likelihood on draws from it, and
forms self-normalized importance weights against the exact target.  A heavy-tailed
analytic proposal is used deliberately: the previous campaign reached Pareto
k = 1.016 because a flow fitted to the target was used as the proposal, and a
fitted flow has lighter tails than its target -- the one property a proposal may
not have.  If Pareto k < 0.7, the reweighted sample *is* the exact posterior, and
the NPE-versus-exact comparison is the agreement gate.  No HMC sample, no
generating parameter, and no audit manifest is read anywhere in this file.
"""

from __future__ import annotations

# --- keep imports working from a theme subfolder: common/ holds the
# --- modules shared by more than one stage.
import pathlib as _pl, sys as _sys
_ROOT = _pl.Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "common"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import json
import pathlib
import time

# Stamped at import so a wall-clock budget counts JAX import and problem
# construction, not just the round loop.
_PROCESS_STARTED = time.time()

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.distributions import Independent, Normal

from three_probe_agreement_common import (
    GRID, PARAMETER_NAMES, PARITY_RELATIVE_TOLERANCE, REFERENCE_POINT_PATH,
    atomic_json, atomic_npz, backend_manifest, build_problem,
    credible_interval_summary, environment_manifest, importance_diagnostics,
    numerical_source_manifest, seed_from_entropy, sha256_file, theta_from_probit,
)

GATE_FAILURE_EXIT_CODE = 3
ROUNDS = (65_536, 65_536, 65_536, 65_536)      # 262,144 total, 2x the previous campaign
# Beyond this radius the probit map saturates in float64 -- Phi(u) == 1.0 exactly
# for u >= 8.30 -- so theta pins to the prior edge and the forward model becomes
# *exactly* constant.  That is an infinite likelihood plateau carrying no
# information.  The N(0,1) prior suppresses it (exp(-8**2/2) ~ 1e-14 of the peak)
# and the HMC potential contains that prior term explicitly, so NUTS can never
# reach it.  An NPE density has no prior inside it, so in job 6928795 the fitted
# tails walked out onto the plateau across rounds -- max|u| 6.6 -> 44 -> 18201 --
# and round 4's SNPE-C atomic loss evaluated a non-finite MDN log-prob at an atom
# out there and aborted the run after 48 minutes.
#
# Restricting *proposal* draws to this region is statistically free: sbi's
# _log_prob_proposal_posterior_atomic never evaluates the proposal density, only
# prior.log_prob(atoms) and the net at batch atoms, so the estimated target is
# unchanged and only the proposal's efficiency moves.
#
# Restricting *posterior* draws is a genuine support statement and is recorded as
# such in the artifact.  It is not a loosened tolerance: the excluded prior mass
# is below 1e-14 of the peak, and without it np.cov(posterior_samples) is inflated
# by four orders of magnitude, which makes the Student-t importance proposal --
# the job's only independent check -- useless.
IDENTIFIABLE_U_RADIUS = 8.0
MIN_REGION_ACCEPTANCE = 0.02
MIN_ROUNDS_FOR_A_RESULT = 2
# sbi asserts finiteness inside its atomic loss and aborts the run.  Production
# sbi_sc/run01 hit it in round 3 with proposal acceptance 1.0, so the probit
# plateau was not the cause, and a stub with the same posterior conditioning
# (eigenvalues [7.9e-4 ... 1.27], condition 1601, thin ROTATED direction) trained
# all four rounds cleanly -- so the mechanism is not simply thin-ridge float32
# underflow either.  With no established mechanism, the honest response is a
# bounded retry rather than an invented fix: the atomic loss draws its contrastive
# atoms at random, so an unlucky batch is cleared by a reseed, and a deterministic
# failure still falls back to the graceful stop.  Every attempt is recorded.
TRAIN_ATTEMPTS = 3
POSTERIOR_DRAWS = 40_000
VALIDATION_DRAWS = 20_000
STUDENT_T_DEGREES_OF_FREEDOM = 4.0
STUDENT_T_SCALE_INFLATION = 1.5
PROPOSAL_LAPLACE_WEIGHT = 0.30      # MAP-anchored Laplace component
PROPOSAL_BROAD_WEIGHT = 0.10        # deliberately over-dispersed safety net
PROPOSAL_BROAD_INFLATION = 9.0
PROPOSAL_CLUSTERS = 8               # k-means components tracking the NPE posterior
PROPOSAL_MIN_CLUSTER_MULTIPLE = 5   # a cluster needs 5*dim points to get a covariance
# Measured on the production contract: a two-component elliptical mixture (Laplace
# at the MAP plus one Gaussian fitted to the NPE moments) gave Pareto k = 0.935 and
# an importance ESS of 25 out of 20,000 -- the exact reference could not adjudicate
# the thing it exists to adjudicate.  The exact profile likelihood is sharply peaked
# with broad shoulders (delta chi2 = 10 at one Laplace sigma, then flattening to 14
# at two), which no single ellipse covers.  A k-means mixture follows an arbitrary
# shape, and the broad component guarantees tail coverage even if every cluster is
# misplaced, so the proposal degrades gracefully instead of collapsing.
TRAINING = dict(hidden_features=128, num_components=10, training_batch_size=512,
                max_num_epochs=500, stop_after_epochs=25,
                validation_fraction=0.10, learning_rate=5e-4)
# Frozen before the run.  Round stability is measured in units of the pooled
# standard deviation; agreement is measured against the job's own exact-likelihood
# reference, never against the HMC.
GATES = dict(
    max_round_mean_shift=0.30,
    round_width_ratio=(0.85, 1.18),
    min_importance_ess=1000.0,
    max_importance_weight=0.02,
    max_pareto_k=0.70,
    max_exact_vs_npe_mean_shift=0.30,
    exact_vs_npe_width_ratio=(0.85, 1.18),
)


def sample_in_region(posterior, count: int, observed_torch, seed: int,
                     radius: float, label: str):
    """Draw ``count`` posterior samples restricted to ``|u|_inf <= radius``.

    Re-draws rather than clipping: clipping would pile mass onto the boundary and
    silently distort the density.  Returns the draws plus an acceptance record so
    the artifact shows how much of the fitted density fell outside the
    identifiable region -- that fraction is the diagnostic for NPE tail quality.
    """

    torch.manual_seed(seed)
    kept, total_kept, total_drawn, attempts = [], 0, 0, 0
    while total_kept < count and attempts < 20:
        attempts += 1
        request = int(min(max(2 * (count - total_kept), 4096), 200_000))
        raw = posterior.sample((request,), x=observed_torch,
                               show_progress_bars=False).numpy().astype(np.float64)
        total_drawn += len(raw)
        good = raw[np.all(np.isfinite(raw), axis=1)
                   & (np.max(np.abs(raw), axis=1) <= radius)]
        kept.append(good)
        total_kept += len(good)
    acceptance = float(total_kept) / float(max(total_drawn, 1))
    if total_kept < count:
        raise RuntimeError(
            f"{label}: only {total_kept} of {count} requested draws fell inside "
            f"|u|_inf <= {radius} after {attempts} attempts (acceptance "
            f"{acceptance:.4f}). The NPE density is not concentrated in the "
            f"identifiable region; the fit is unusable, not merely inefficient.")
    if acceptance < MIN_REGION_ACCEPTANCE:
        raise RuntimeError(
            f"{label}: acceptance inside the identifiable region is "
            f"{acceptance:.4f} < {MIN_REGION_ACCEPTANCE}. Refusing to continue on "
            f"a density that places over 98 percent of its mass on the "
            f"likelihood plateau.")
    out = np.concatenate(kept, axis=0)[:count]
    return out, dict(acceptance=acceptance, drawn=int(total_drawn),
                     attempts=int(attempts), radius=float(radius))


def build_defensive_proposal(samples: np.ndarray, u_map: np.ndarray,
                             laplace_covariance: np.ndarray, seed: int):
    """Student-t mixture that follows the NPE posterior's shape, not just its moments.

    Three kinds of component, all Student-t(nu=4) and exactly normalised so the
    importance weights stay valid:

    * the Laplace approximation at the pinned MAP -- network-independent and known
      good, so the proposal is never worse than a Laplace proposal;
    * one deliberately over-dispersed copy of it, which alone bounds the weight of
      any point the other components misplace;
    * up to ``PROPOSAL_CLUSTERS`` k-means components fitted to the NPE draws, which
      is what lets the proposal follow a ridge or any other non-elliptical shape.
    """

    from scipy.cluster.vq import kmeans2

    dimension = samples.shape[1]
    components = [
        (np.asarray(u_map, dtype=np.float64),
         np.linalg.cholesky(laplace_covariance * STUDENT_T_SCALE_INFLATION)),
        (np.asarray(u_map, dtype=np.float64),
         np.linalg.cholesky(laplace_covariance * PROPOSAL_BROAD_INFLATION)),
    ]
    weights = [PROPOSAL_LAPLACE_WEIGHT, PROPOSAL_BROAD_WEIGHT]

    # A floor keeps a thin or tiny cluster from producing a singular covariance,
    # which would make its density infinite and the weights meaningless.
    floor = 1.0e-3 * np.diag(np.diag(np.cov(samples, rowvar=False)))
    centroids, labels = kmeans2(samples, PROPOSAL_CLUSTERS, minit="++", seed=int(seed))
    kept, populations = [], []
    for index in range(len(centroids)):
        member = samples[labels == index]
        if len(member) < PROPOSAL_MIN_CLUSTER_MULTIPLE * dimension:
            continue
        covariance = np.cov(member, rowvar=False) * STUDENT_T_SCALE_INFLATION + floor
        try:
            factor = np.linalg.cholesky(0.5 * (covariance + covariance.T))
        except np.linalg.LinAlgError:
            continue
        kept.append((member.mean(axis=0), factor))
        populations.append(len(member))

    cluster_weight = 1.0 - PROPOSAL_LAPLACE_WEIGHT - PROPOSAL_BROAD_WEIGHT
    if kept:
        share = np.asarray(populations, dtype=np.float64)
        share /= share.sum()
        components.extend(kept)
        weights.extend((cluster_weight * share).tolist())
    else:
        # No usable cluster: hand the cluster weight to the broad component rather
        # than silently renormalising onto the narrow one.
        weights[1] += cluster_weight
    weights = (np.asarray(weights, dtype=np.float64) / np.sum(weights)).tolist()
    detail = dict(n_components=len(components), n_clusters_kept=len(kept),
                  clusters_requested=PROPOSAL_CLUSTERS,
                  cluster_populations=[int(v) for v in populations],
                  laplace_weight=PROPOSAL_LAPLACE_WEIGHT,
                  broad_weight=float(weights[1]),
                  broad_inflation=PROPOSAL_BROAD_INFLATION,
                  degrees_of_freedom=STUDENT_T_DEGREES_OF_FREEDOM)
    return components, weights, detail


def student_t_mixture_logpdf(u: np.ndarray, components, weights, df: float) -> np.ndarray:
    """Exactly normalised log density of a Student-t mixture."""

    terms = np.stack([np.log(w) + student_t_logpdf(u, loc, chol, df)
                      for (loc, chol), w in zip(components, weights)])
    return np.logaddexp.reduce(terms, axis=0)


def student_t_mixture_sample(rng, n: int, components, weights, df: float) -> np.ndarray:
    counts = rng.multinomial(n, np.asarray(weights, dtype=np.float64))
    draws = [student_t_sample(rng, int(k), loc, chol, df)
             for (loc, chol), k in zip(components, counts) if k > 0]
    out = np.concatenate(draws, axis=0)
    rng.shuffle(out, axis=0)
    return out


def student_t_logpdf(u: np.ndarray, loc: np.ndarray, scale_chol: np.ndarray, df: float) -> np.ndarray:
    from scipy.special import gammaln

    dimension = loc.size
    delta = np.linalg.solve(scale_chol, (u - loc[None, :]).T).T
    quadratic = np.sum(delta * delta, axis=1)
    log_determinant = 2.0 * np.sum(np.log(np.diag(scale_chol)))
    return (gammaln(0.5 * (df + dimension)) - gammaln(0.5 * df)
            - 0.5 * dimension * np.log(df * np.pi) - 0.5 * log_determinant
            - 0.5 * (df + dimension) * np.log1p(quadratic / df))


def student_t_sample(rng: np.random.Generator, n: int, loc: np.ndarray,
                     scale_chol: np.ndarray, df: float) -> np.ndarray:
    normal = rng.standard_normal((n, loc.size))
    chi = rng.chisquare(df, size=n)
    return loc[None, :] + (normal @ scale_chol.T) / np.sqrt(chi / df)[:, None]


def batched_apply(function, values: np.ndarray, batch_size: int, label: str) -> np.ndarray:
    """Apply a vmapped JAX function in chunks, halving the chunk on OOM.

    A single evaluation of this forward carries large intermediates -- the
    aperture Hankel transform is (n_z, n_M, n_aperture) and the Limber
    projection is (n_ell, n_z) -- so vmapping thousands at once is not viable.
    A smoke run with a chunk of 4096 asked XLA for 102 GiB on an 80 GiB H100.
    The chunk is therefore adaptive rather than a guessed constant.
    """

    outputs = []
    start = 0
    current = int(batch_size)
    while start < len(values):
        chunk = values[start:start + current]
        try:
            outputs.append(np.asarray(function(jnp.asarray(chunk, dtype=jnp.float64)),
                                      dtype=np.float64))
        except Exception as exc:  # noqa: BLE001 - XLA raises several OOM types
            if "RESOURCE_EXHAUSTED" not in str(exc) and "Out of memory" not in str(exc):
                raise
            if current <= 1:
                raise RuntimeError(f"{label}: out of memory even at chunk size 1") from exc
            current = max(current // 2, 1)
            print(f"{label}: OOM, retrying with chunk size {current}", flush=True)
            continue
        start += len(chunk)
    return np.concatenate(outputs, axis=0)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    return float(np.interp(quantile, cumulative, values[order]))


def weighted_summary(samples: np.ndarray, weights: np.ndarray, names: tuple[str, ...]) -> dict:
    weights = weights / weights.sum()
    out = {}
    order = np.argsort(samples, axis=0)
    for index, name in enumerate(names):
        column = samples[:, index]
        mean = float(np.sum(weights * column))
        variance = float(np.sum(weights * (column - mean) ** 2))
        sorted_index = order[:, index]
        cumulative = np.cumsum(weights[sorted_index])
        low = float(np.interp(0.05, cumulative, column[sorted_index]))
        high = float(np.interp(0.95, cumulative, column[sorted_index]))
        out[name] = dict(mean=mean, std=float(np.sqrt(variance)),
                         median=float(np.interp(0.5, cumulative, column[sorted_index])),
                         q05=low, q95=high, width90=high - low)
    return out


def compare_summaries(a: dict, b: dict, names: tuple[str, ...]) -> dict:
    """Mean shift in pooled sigma and 90% width ratio, per parameter (a versus b)."""

    out = {}
    for name in names:
        pooled = np.sqrt(0.5 * (a[name]["std"] ** 2 + b[name]["std"] ** 2))
        out[name] = dict(
            mean_shift_pooled_sigma=float(abs(a[name]["mean"] - b[name]["mean"]) / pooled),
            width90_ratio=float(a[name]["width90"] / b[name]["width90"]),
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wall-seconds", type=float, default=None,
                        help="Stop starting new rounds once the next one would "
                             "cross this many seconds from process start, then run "
                             "the exact-likelihood validation on the rounds "
                             "completed. A round cannot be stopped part-way, so this "
                             "trades campaign length for a guaranteed finish.")
    parser.add_argument("--contract", type=pathlib.Path, default=None,
                        help="Registered inference contract supplying the "
                             "observation. Default: the production pasted-map "
                             "contract. The loader admits only registered "
                             "contracts, so this selects between audited inputs.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reference-point", type=pathlib.Path, default=REFERENCE_POINT_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-nonportable-forward", action="store_true",
                        help="Sample anyway on a forward model that fails CPU/GPU "
                             "parity; the artifact is labelled accordingly.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Accept a non-converged reference grid and label the artifact SMOKE.")
    parser.add_argument("--rounds-override", type=int, nargs="*", default=None)
    parser.add_argument("--validation-draws", type=int, default=VALIDATION_DRAWS)
    parser.add_argument("--posterior-draws", type=int, default=POSTERIOR_DRAWS)
    parser.add_argument("--forward-batch-size", type=int, default=16,
                        help="Starting vmap chunk; halved automatically on OOM.")
    parser.add_argument("--resume", action="store_true",
                        help="Re-append saved round_*_simulations.npz from the "
                             "output directory instead of re-simulating them. "
                             "Retrains on CPU only; refuses across a changed "
                             "contract, reference point or numerical source set.")
    args = parser.parse_args()

    started = time.time()
    environment = environment_manifest()
    sources = numerical_source_manifest()
    backend = backend_manifest()

    reference = json.loads(args.reference_point.read_text())
    if reference["schema"] != "godmax.sbi.three_probe_reference_point.v2":
        raise RuntimeError("Unexpected reference-point schema")
    # Three states: True (stage 1 verified CPU/GPU parity), False (stage 1
    # measured a disagreement), None (never checked).  Only True samples freely;
    # None is treated as unverified rather than as permission.
    portable = reference.get("backend_portable")
    if portable is False and not args.allow_nonportable_forward:
        raise RuntimeError(
            "The pinned reference point records that the forward model is NOT "
            "backend-portable. Refusing to sample: a posterior from a forward "
            "model that disagrees with itself across backends is not a result. "
            "Re-run stage 1 after fixing the compiler toolchain, or pass "
            "--allow-nonportable-forward for an explicitly-labelled diagnostic.")
    if portable is None and not (args.smoke_test or args.allow_nonportable_forward):
        raise RuntimeError(
            "The pinned reference point carries no backend-parity verdict. "
            "Run stage 1 (which stamps it) before sampling, or pass "
            "--smoke-test / --allow-nonportable-forward deliberately.")
    grid = tuple(reference["grid"])
    if grid != GRID and not args.smoke_test:
        raise RuntimeError(f"Reference point grid {grid} is not the converged grid {GRID}")
    reference_sha256 = sha256_file(args.reference_point)
    rounds = tuple(args.rounds_override) if args.rounds_override else ROUNDS
    validation_draws = int(args.validation_draws)
    posterior_draws = int(args.posterior_draws)

    problem = build_problem(grid, jit_compile=False, contract_path=args.contract)
    if reference["contract_sha256"] != problem.contract.contract_sha256:
        raise RuntimeError("Reference point was built against a different inference contract")

    batch_predict = jax.jit(jax.vmap(problem.predict_u))
    single_chi2 = jax.jit(problem.chi2_u)
    batch_chi2 = jax.jit(jax.vmap(problem.chi2_u))

    parity_points = np.asarray(reference["parity"]["probit_points"], dtype=np.float64)
    parity_expected = np.asarray(reference["parity"]["vectors"], dtype=np.float64)
    parity_here = batched_apply(batch_predict, parity_points, 1, "parity")
    parity_relative = np.abs(parity_here / parity_expected - 1.0)
    parity = dict(
        reference_backend=reference["backend"]["default_backend"],
        reference_device_kind=reference["backend"]["device_kind"],
        this_backend=backend["default_backend"], this_device_kind=backend["device_kind"],
        max_relative_difference=float(parity_relative.max()),
        median_relative_difference=float(np.median(parity_relative)),
        tolerance=PARITY_RELATIVE_TOLERANCE,
        passed=bool(parity_relative.max() <= PARITY_RELATIVE_TOLERANCE),
    )
    print("backend parity:", json.dumps(parity, sort_keys=True), flush=True)

    u_map = np.asarray(reference["u_map"], dtype=np.float64)
    operator = np.asarray(reference["score_operator"], dtype=np.float64)
    reference_prediction = np.asarray(reference["score_reference_prediction"], dtype=np.float64)
    cholesky = problem.cholesky
    observed_score = operator @ np.linalg.solve(cholesky, problem.observation - reference_prediction)

    theory_entropy = tuple(problem.contract.seed_namespaces["theory_sbi"]["entropy"])
    network_entropy = tuple(problem.contract.seed_namespaces["network_initialization"]["entropy"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A resume replays saved simulations, so it must refuse across any change that
    # would have altered those simulations.  The runner's own sha is deliberately
    # NOT part of this check: the saved (u, summary) pairs depend on the forward
    # model, the reference point and the seeds, not on the training code below.
    resume_state = dict(requested=bool(args.resume), reused_rounds=[], dropped_rows={})
    if args.resume:
        # The provenance to check against is the ORIGINAL run's preflight, which
        # this run is about to overwrite.  Snapshot it once, so the check cannot be
        # made vacuous by the two-invocation (--preflight-only then real) pattern
        # in the sbatch script.
        previous_path = args.output_dir / "resume_provenance.json"
        if not previous_path.exists() and (args.output_dir / "preflight.json").exists():
            previous_path.write_text((args.output_dir / "preflight.json").read_text())
        saved_rounds = sorted(args.output_dir.glob("round_*_simulations.npz"))
        if not saved_rounds:
            # --resume on a directory with nothing to resume is a fresh start, not an
            # error.  The sbatch scripts pass --resume unconditionally so that a
            # requeue reuses whatever the first attempt got; on the first attempt
            # there is nothing, and refusing there would kill the submission before
            # it began.
            resume_state["fresh_start"] = True
            print("--resume: no saved simulations in the output directory; "
                  "starting a fresh campaign", flush=True)
            args.resume = False
        elif not previous_path.exists():
            raise RuntimeError(
                f"--resume found {len(saved_rounds)} saved round file(s) but no "
                f"preflight.json or resume_provenance.json to check them against. "
                f"Refusing to reuse simulations of unknown provenance.")
    if args.resume:
        previous = json.loads(previous_path.read_text())
        mismatch = {
            key: (previous.get(key), current)
            for key, current in (
                ("contract_sha256", problem.contract.contract_sha256),
                ("reference_point_sha256", reference_sha256),
            ) if previous.get(key) != current
        }
        if previous.get("numerical_sources", {}).get("aggregate_sha256") != \
                sources["aggregate_sha256"]:
            mismatch["numerical_sources_aggregate"] = (
                previous.get("numerical_sources", {}).get("aggregate_sha256"),
                sources["aggregate_sha256"])
        if tuple(previous.get("grid", ())) != tuple(grid):
            mismatch["grid"] = (previous.get("grid"), list(grid))
        if mismatch:
            raise RuntimeError(
                "Refusing to resume across changed provenance: "
                + json.dumps(mismatch, sort_keys=True))
        resume_state["previous_runner_source_sha256"] = previous.get("runner_source_sha256")

    preflight = dict(
        schema="godmax.sbi.three_probe_sbi_v2_preflight.v1", status="PASS",
        grid=list(grid), grid_is_converged=bool(grid == GRID), smoke_test=bool(args.smoke_test), rounds=list(rounds), total_simulations=sum(rounds),
        contract_sha256=problem.contract.contract_sha256,
        reference_point_sha256=reference_sha256,
        runner_source_sha256=sha256_file(pathlib.Path(__file__)),
        numerical_sources=sources, environment=environment, backend=backend,
        backend_parity=parity,
        summary="exact_5d_normalized_score_at_pinned_map",
        parameterization="exact_box_to_standard_normal_probit",
        density_estimator="mdn_10_components",
        gates=GATES,
        observed_score=observed_score.tolist(),
        observed_score_norm=float(np.linalg.norm(observed_score)),
        fisher_condition_number=reference["fisher_condition_number"],
        chi2_at_map=reference["chi2_at_map"],
        comparison_inputs_used_during_training=False,
    )
    atomic_json(args.output_dir / "preflight.json", preflight)
    print("observed 5d score:", np.round(observed_score, 5),
          "  |s| =", round(float(np.linalg.norm(observed_score)), 4),
          "(a typical draw has |s| ~ sqrt(5) = 2.24)", flush=True)
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return

    from sbi.inference import SNPE
    from sbi.utils import posterior_nn

    network_seed = seed_from_entropy(network_entropy, sum(rounds))
    torch.manual_seed(network_seed)
    prior = Independent(Normal(torch.zeros(5), torch.ones(5)), 1)
    builder = posterior_nn(model="mdn", hidden_features=TRAINING["hidden_features"],
                           num_components=TRAINING["num_components"],
                           z_score_theta="none", z_score_x="independent")
    inference = SNPE(prior=prior, density_estimator=builder, device="cpu")
    observed_torch = torch.as_tensor(observed_score, dtype=torch.float32)

    def simulate(u: np.ndarray, seed: int) -> np.ndarray:
        """Simulate data at ``u``, add the exact Gaussian noise, and compress."""

        prediction = batched_apply(batch_predict, u, args.forward_batch_size, "simulate")
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("Non-finite forward prediction in the SBI simulator")
        noise = np.random.default_rng(seed).standard_normal(prediction.shape)
        whitened = np.linalg.solve(cholesky, (prediction - reference_prediction[None, :]).T).T + noise
        return (operator @ whitened.T).T

    # Enough time for the exact-likelihood validation plus artifact writing.
    validation_reserve_seconds = 1200.0
    round_seconds = []
    train_retries = []
    budget_stop = None
    round_records = []
    posterior = None
    posterior_samples = None
    posterior_draw_info = None
    training_failure = None
    for index, count in enumerate(rounds):
        round_number = index + 1
        round_started = time.time()
        if (args.max_wall_seconds is not None and round_records
                and len(round_records) >= MIN_ROUNDS_FOR_A_RESULT):
            elapsed = time.time() - _PROCESS_STARTED
            # Rounds get slower as the training set grows, so scale the last
            # round's cost by how much more data the next one will train on.
            growth = sum(rounds[:round_number + 1]) / max(sum(rounds[:round_number]), 1)
            projected = elapsed + round_seconds[-1] * growth + validation_reserve_seconds
            if projected > args.max_wall_seconds:
                budget_stop = dict(
                    reason="max_wall_seconds", limit_seconds=float(args.max_wall_seconds),
                    elapsed_seconds=float(elapsed),
                    projected_after_next_round=float(projected),
                    rounds_completed=len(round_records),
                    rounds_requested=len(rounds),
                    simulations_completed=int(sum(rounds[:round_number])),
                    simulations_requested=int(sum(rounds)),
                    last_round_seconds=float(round_seconds[-1]),
                    assumed_growth=float(growth),
                    validation_reserve_seconds=float(validation_reserve_seconds))
                print(f"[stop] wall budget {args.max_wall_seconds:.0f} s would be "
                      f"exceeded by round {round_number} "
                      f"(elapsed {elapsed:.0f} s, last round {round_seconds[-1]:.0f} s, "
                      f"growth x{growth:.2f}). Proceeding to the exact-likelihood "
                      f"validation with {len(round_records)} rounds and "
                      f"{sum(rounds[:round_number])} simulations.", flush=True)
                break
        proposal_seed = seed_from_entropy(theory_entropy, sum(rounds), round_number, 1)
        noise_seed = seed_from_entropy(theory_entropy, sum(rounds), round_number, 2)
        saved_simulations = args.output_dir / f"round_{round_number}_simulations.npz"
        saved_record = args.output_dir / f"round_{round_number}.ready.json"
        proposal_info = None
        reused = False
        if args.resume and saved_simulations.exists() and saved_record.exists():
            stored = json.loads(saved_record.read_text())
            if (stored.get("proposal_seed"), stored.get("noise_seed")) != \
                    (int(proposal_seed), int(noise_seed)):
                raise RuntimeError(
                    f"round {round_number}: saved seeds "
                    f"{(stored.get('proposal_seed'), stored.get('noise_seed'))} do not "
                    f"match the recomputed {(int(proposal_seed), int(noise_seed))}; "
                    f"the seed derivation changed, so the saved simulations are not "
                    f"the ones this configuration would have produced.")
            payload_npz = np.load(saved_simulations)
            u_round = np.asarray(payload_npz["u"], dtype=np.float64)
            summaries = np.asarray(payload_npz["summary"], dtype=np.float64)
            inside = np.max(np.abs(u_round), axis=1) <= IDENTIFIABLE_U_RADIUS
            dropped = int((~inside).sum())
            u_round, summaries = u_round[inside], summaries[inside]
            # Dropping plateau rows is the same proposal redefinition applied to
            # fresh draws above, applied retroactively to a saved proposal.  It is
            # self-consistent because the atomic loss never evaluates the proposal
            # density, only the round index.
            resume_state["reused_rounds"].append(round_number)
            resume_state["dropped_rows"][str(round_number)] = dropped
            reused = True
            print(f"round {round_number}: reusing {len(u_round)} saved simulations "
                  f"({dropped} plateau rows dropped)", flush=True)
        elif posterior is None:
            u_round = np.random.default_rng(proposal_seed).standard_normal((count, 5))
            summaries = simulate(u_round, noise_seed)
        else:
            u_round, proposal_info = sample_in_region(
                posterior, count, observed_torch, proposal_seed,
                IDENTIFIABLE_U_RADIUS, f"round {round_number} proposal")
            print(f"round {round_number}: proposal acceptance inside "
                  f"|u|<={IDENTIFIABLE_U_RADIUS} = {proposal_info['acceptance']:.4f}",
                  flush=True)
            summaries = simulate(u_round, noise_seed)
        # BUG 2 fixed: rounds after the first must be appended with the actual
        # proposal, so SNPE-C applies its atomic loss correction.  Forcing the
        # first-round loss on proposal-drawn simulations would have fitted the
        # proposal-weighted density instead of the posterior.
        inference.append_simulations(torch.as_tensor(u_round, dtype=torch.float32),
                                     torch.as_tensor(summaries, dtype=torch.float32),
                                     proposal=posterior)
        # Pin the training RNG per round so a round's network is a deterministic
        # function of (its data, its round index) and nothing else.  Without this
        # the state entering train() depends on how many draws earlier rounds
        # happened to consume, so a --resume that replays saved simulations
        # produces a different network than the original run and any round it
        # regenerates is unreproducible.  Verified by
        # tests/three_probe_v2/test_sbi_region_and_resume.py [4].  Attempt 1 keeps
        # index 4 so that reproducibility survives the addition of retries.
        estimator, attempts = None, []
        for attempt in range(1, TRAIN_ATTEMPTS + 1):
            torch.manual_seed(seed_from_entropy(
                network_entropy, sum(rounds), round_number,
                4 if attempt == 1 else 1000 + attempt))
            try:
                estimator = inference.train(
                    training_batch_size=TRAINING["training_batch_size"],
                    learning_rate=TRAINING["learning_rate"],
                    validation_fraction=TRAINING["validation_fraction"],
                    stop_after_epochs=TRAINING["stop_after_epochs"],
                    max_num_epochs=TRAINING["max_num_epochs"],
                    # A failed train() can leave the net in the very state that
                    # produced the non-finite value; continuing from those weights
                    # would most likely fail the same way.
                    **({} if attempt == 1 else {"retrain_from_scratch": True}),
                )
                attempts.append(dict(attempt=attempt, status="trained"))
                break
            except AssertionError as error:
                attempts.append(dict(attempt=attempt, status="assertion_error",
                                     error=repr(error)))
                print(f"round {round_number}: training attempt {attempt} of "
                      f"{TRAIN_ATTEMPTS} failed ({error})"
                      + ("; retrying from a fresh network"
                         if attempt < TRAIN_ATTEMPTS else ""), flush=True)
        if estimator is None:
            # Losing every earlier round to a late training failure is a worse
            # outcome than stopping with a clearly-labelled shorter campaign, so
            # record it, keep the last good posterior, and go on to the
            # exact-likelihood reference -- which is independent of the NPE and is
            # the check that actually adjudicates.
            training_failure = dict(round=round_number, attempts=attempts,
                                    error=attempts[-1]["error"],
                                    completed_rounds=len(round_records))
            print(f"round {round_number}: TRAINING FAILED after {len(attempts)} "
                  f"attempts; continuing with the round-{len(round_records)} "
                  f"posterior", flush=True)
            if len(round_records) < MIN_ROUNDS_FOR_A_RESULT:
                raise RuntimeError(
                    f"Training failed in round {round_number} with only "
                    f"{len(round_records)} completed round(s); "
                    f"{MIN_ROUNDS_FOR_A_RESULT} are needed for a result. "
                    f"Attempts: {json.dumps(attempts)}")
            break
        if len(attempts) > 1:
            train_retries.append(dict(round=round_number, attempts=attempts))
        posterior = inference.build_posterior(estimator)
        # sbi requires a proposal handed to append_simulations to carry its own
        # x_o; sampling with x=... does not set it, and the next round's
        # append_simulations rejects the proposal without it.
        posterior.set_default_x(observed_torch)
        posterior_samples, posterior_draw_info = sample_in_region(
            posterior, posterior_draws, observed_torch,
            seed_from_entropy(network_entropy, sum(rounds), round_number, 3),
            IDENTIFIABLE_U_RADIUS, f"round {round_number} posterior")
        chi2_draws = batched_apply(batch_chi2, posterior_samples[:512],
                                   args.forward_batch_size, f"round {round_number} chi2")
        record = dict(
            round=round_number, simulations=int(count),
            cumulative_simulations=int(sum(rounds[:round_number])),
            proposal_seed=int(proposal_seed), noise_seed=int(noise_seed),
            simulations_used=int(len(u_round)), reused_saved_simulations=bool(reused),
            proposal_region=proposal_info,
            posterior_draw_region=posterior_draw_info,
            u_summary=credible_interval_summary(posterior_samples,
                                                tuple(f"u_{n}" for n in PARAMETER_NAMES)),
            theta_summary=credible_interval_summary(
                theta_from_probit(posterior_samples, problem.low, problem.high), PARAMETER_NAMES),
            exact_chi2_first_512=dict(minimum=float(chi2_draws.min()),
                                      median=float(np.median(chi2_draws)),
                                      q95=float(np.percentile(chi2_draws, 95.0))),
        )
        round_records.append(record)
        atomic_npz(args.output_dir / f"round_{round_number}_simulations.npz",
                   u=u_round, summary=summaries)
        atomic_npz(args.output_dir / f"posterior_samples_round_{round_number}.npz",
                   u=posterior_samples,
                   theta=theta_from_probit(posterior_samples, problem.low, problem.high))
        atomic_json(args.output_dir / f"round_{round_number}.ready.json", record)
        round_seconds.append(time.time() - round_started)
        record["wall_seconds"] = round_seconds[-1]
        print(f"round {round_number}: exact chi2 of posterior draws "
              f"min {chi2_draws.min():.2f} median {np.median(chi2_draws):.2f}"
              f"   ({round_seconds[-1]:.0f} s)", flush=True)

    npe_u_summary = round_records[-1]["u_summary"]
    round_stability = compare_summaries(npe_u_summary, round_records[-2]["u_summary"],
                                        tuple(f"u_{n}" for n in PARAMETER_NAMES))

    # ---- exact-likelihood reference, computed here, HMC never consulted -------
    # Two-component defensive proposal.  Anchoring only on the NPE posterior makes
    # the exact reference useless exactly when it is most needed: a real run at
    # 768 simulations produced an NPE posterior 18-53x too wide, and the resulting
    # NPE-derived proposal gave Pareto k = 15.0 and ESS 1.66 of 384 -- so the
    # reference could not adjudicate the very thing it exists to adjudicate.  The
    # first component is the Laplace approximation at the pinned MAP, which is
    # independent of the network and known good; the second tracks the NPE
    # posterior so the proposal stays efficient once NPE is right.  The mixture is
    # exactly normalised, so the importance weights remain valid.
    laplace_covariance = np.asarray(reference["laplace_covariance"], dtype=np.float64)
    laplace_covariance = 0.5 * (laplace_covariance + laplace_covariance.T)
    components, mixture_weights, proposal_detail = build_defensive_proposal(
        posterior_samples, u_map, laplace_covariance,
        seed_from_entropy(theory_entropy, sum(rounds), 98, 5))
    print("defensive proposal:", json.dumps(proposal_detail, sort_keys=True), flush=True)
    rng = np.random.default_rng(seed_from_entropy(theory_entropy, sum(rounds), 99, 7))
    u_validation = student_t_mixture_sample(rng, validation_draws, components,
                                            mixture_weights, STUDENT_T_DEGREES_OF_FREEDOM)
    chi2_validation = batched_apply(batch_chi2, u_validation, args.forward_batch_size,
                                    "exact-likelihood validation")
    log_target = -0.5 * chi2_validation - 0.5 * np.sum(u_validation ** 2, axis=1)
    log_proposal = student_t_mixture_logpdf(u_validation, components, mixture_weights,
                                            STUDENT_T_DEGREES_OF_FREEDOM)
    log_weights = log_target - log_proposal
    diagnostics = importance_diagnostics(log_weights)
    weights = np.exp(log_weights - np.max(log_weights))
    exact_u_summary = weighted_summary(u_validation, weights,
                                       tuple(f"u_{n}" for n in PARAMETER_NAMES))
    exact_theta_summary = weighted_summary(
        theta_from_probit(u_validation, problem.low, problem.high), weights, PARAMETER_NAMES)
    exact_vs_npe = compare_summaries(npe_u_summary, exact_u_summary,
                                     tuple(f"u_{n}" for n in PARAMETER_NAMES))

    gate_items = dict(
        round_stability_mean=all(v["mean_shift_pooled_sigma"] <= GATES["max_round_mean_shift"]
                                 for v in round_stability.values()),
        round_stability_width=all(GATES["round_width_ratio"][0] <= v["width90_ratio"]
                                  <= GATES["round_width_ratio"][1] for v in round_stability.values()),
        importance_ess=diagnostics["ess"] >= GATES["min_importance_ess"],
        importance_max_weight=diagnostics["max_weight"] <= GATES["max_importance_weight"],
        pareto_k=bool(np.isfinite(diagnostics["pareto_k"]) and diagnostics["pareto_k"] <= GATES["max_pareto_k"]),
        exact_vs_npe_mean=all(v["mean_shift_pooled_sigma"] <= GATES["max_exact_vs_npe_mean_shift"]
                              for v in exact_vs_npe.values()),
        exact_vs_npe_width=all(GATES["exact_vs_npe_width_ratio"][0] <= v["width90_ratio"]
                               <= GATES["exact_vs_npe_width_ratio"][1] for v in exact_vs_npe.values()),
    )
    gate_items["all_rounds_trained"] = training_failure is None
    gate_items["completed_requested_rounds"] = budget_stop is None
    gate = all(gate_items.values())

    atomic_npz(args.output_dir / "exact_likelihood_validation.npz",
               u=u_validation, chi2=chi2_validation, log_weights=log_weights,
               theta=theta_from_probit(u_validation, problem.low, problem.high))
    payload = dict(
        schema="godmax.sbi.three_probe_sbi_v2.v1",
        status=("SMOKE_" if args.smoke_test else "")
               + ("PASS" if (gate and training_failure is None) else "COMPLETED_REJECTED"),
        training_failure=training_failure,
        training_retries=train_retries,
        budget_stop=budget_stop,
        rounds_completed=len(round_records),
        simulations_completed=int(sum(r["simulations"] for r in round_records)),
        round_wall_seconds=round_seconds,
        identifiable_region=dict(
            radius=IDENTIFIABLE_U_RADIUS,
            statement="proposal and posterior draws restricted to |u|_inf <= 8.0, "
                      "beyond which Phi(u) saturates in float64 and the forward "
                      "model is exactly constant",
            excluded_prior_mass_upper_bound=float(np.exp(-0.5 * IDENTIFIABLE_U_RADIUS ** 2)),
            proposal_restriction_changes_target=False,
            posterior_restriction_is_a_support_statement=True,
        ),
        resume=resume_state,
        gate=gate_items, gates=GATES,
        grid=list(grid), grid_is_converged=bool(grid == GRID), smoke_test=bool(args.smoke_test), rounds=list(rounds), total_simulations=sum(rounds),
        contract_sha256=problem.contract.contract_sha256,
        reference_point_sha256=reference_sha256,
        runner_source_sha256=sha256_file(pathlib.Path(__file__)),
        numerical_sources=sources, environment=environment, backend=backend,
        backend_parity=parity, wall_seconds=time.time() - started,
        summary="exact_5d_normalized_score_at_pinned_map",
        observed_score=observed_score.tolist(),
        rounds_detail=round_records,
        round_stability=round_stability,
        npe_u_summary=npe_u_summary,
        npe_theta_summary=round_records[-1]["theta_summary"],
        exact_reference=dict(
            method="student_t_mixture_defensive_importance_sampling",
            components=["laplace_at_pinned_map", "laplace_over_dispersed",
                        "kmeans_clusters_of_npe_draws"],
            proposal_detail=proposal_detail,
            component_weights=mixture_weights,
            degrees_of_freedom=STUDENT_T_DEGREES_OF_FREEDOM,
            scale_inflation=STUDENT_T_SCALE_INFLATION,
            draws=validation_draws,
            diagnostics=diagnostics,
            u_summary=exact_u_summary, theta_summary=exact_theta_summary,
            chi2=dict(minimum=float(chi2_validation.min()),
                      weighted_median=float(_weighted_quantile(
                          chi2_validation, weights, 0.5)),
                      weighted_q95=float(_weighted_quantile(
                          chi2_validation, weights, 0.95)),
                      posterior_weighted_mean=float(np.sum(weights * chi2_validation) / weights.sum())),
        ),
        exact_vs_npe=exact_vs_npe,
        chi2_reference=dict(retained_rank=42, n_varied=5, expected=37, expected_scatter=8.6),
        comparison_inputs_used_during_training=False,
    )
    atomic_json(args.output_dir / "diagnostics.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "numerical_sources"}, sort_keys=True))
    print("\nimportance diagnostics:", diagnostics)
    print("NPE versus exact posterior (pooled sigma / width ratio):")
    for name, value in exact_vs_npe.items():
        print(f"   {name:20s} shift {value['mean_shift_pooled_sigma']:.3f}  "
              f"width {value['width90_ratio']:.3f}")
    print(f"\nABSOLUTE FIT: posterior-weighted exact chi2 "
          f"{payload['exact_reference']['chi2']['posterior_weighted_mean']:.2f} "
          f"against the nominal reference 42-5 = 37 +- 8.6.")
    if not gate:
        print(f"SBI completed but FAILED the gate: "
              f"{[k for k, v in gate_items.items() if not v]}", flush=True)
        raise SystemExit(GATE_FAILURE_EXIT_CODE)


if __name__ == "__main__":
    main()
