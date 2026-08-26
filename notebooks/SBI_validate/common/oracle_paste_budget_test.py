#!/usr/bin/env python3
"""Stage 0: decide the paste budget before spending any GPU time on pastes.

The expensive unknown in mock SBI is the pasted response ``mu_paste(theta)``.
Everything about how many pastes we need is a statement about how well a smooth
5-parameter, 42-output function can be emulated from N design points -- and that
question can be answered for free, because the analytic theory model is the same
kind of function and costs 21 ms per evaluation.

So: treat ``mu_theory(theta)`` as a stand-in expensive simulator, emulate it from
N points drawn by the section-5 mixture recipe, and measure

  * held-out accuracy in whitened chi-square units (the only units that matter,
    because that is what moves a posterior), and
  * the posterior actually obtained through the emulator, against the exact
    posterior of the same problem.

This is the *conservative* target: emulating the full response, with no help from
the transfer factorization.  Whatever N passes here is an upper bound on the real
requirement, because production emulates ``r(theta) = mu_paste/mu_theory``, whose
variation over the design is smaller.  The script therefore also reports the
achievable *relative* accuracy per N, so Stage 2 can convert its measured
r-variation amplitude into a required N without rerunning this.

The exact reference posterior is computed two independent ways -- emcee on the
true forward model, and the archived production NUTS chain -- so a disagreement
between the emulator and the reference cannot be blamed on the reference.
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

import emcee
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

THIS_DIR = pathlib.Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[2], THIS_DIR.parents[2] / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mock_sbi_common as msc
from three_probe_agreement_common import (
    GRID, backend_manifest, build_problem, credible_interval_summary,
    numerical_source_manifest, sha256_array,
)

PARAMETER_NAMES = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")

# Section-5 round-1 mixture, in probit coordinates.
MIXTURE_RECIPES = {
    # The pre-registered recipe from the deprecated three-way plan.
    "defensive": {"weights": {"guide": 0.50, "broadened": 0.30, "prior": 0.20},
                  "broaden_covariance_factor": 4.0},
    # Tighter insurance, defensible now that the transfer-corrected guide has been
    # shown to put the generating point at the 45th percentile rather than the
    # 99.999th: the prior component no longer has to carry the whole risk.
    "concentrated": {"weights": {"guide": 0.60, "broadened": 0.30, "prior": 0.10},
                     "broaden_covariance_factor": 2.25},
    # Diagnostic only, never a production design.  The measured failure mode is that
    # honest GP uncertainty becomes O(1) in the prior tails, where training coverage is
    # thin, and emcee then finds an inflated likelihood there and grows a broad shoulder.
    # This recipe removes the prior component entirely to test whether the posterior gate
    # is recoverable once the design covers the posterior -- which is what the sequential
    # rounds are supposed to achieve.
    "guide_only": {"weights": {"guide": 1.0, "broadened": 0.0, "prior": 0.0},
                   "broaden_covariance_factor": 2.25},
}

BUDGET_LADDER = (24, 48, 96, 144, 200, 288, 384)

# The likelihood-error gate is meaningful only where the posterior has support.
# The section-5 mixture deliberately puts 20% of its draws in the prior, and those
# reach chi2 ~ 1.3e5 against a minimum of ~160; a p95 taken over all of them
# measures emulator behaviour in regions no chain ever visits.  So the gate is
# applied to the posterior-relevant subset -- points within this much of the
# design's minimum chi2 -- while the all-holdout numbers are reported alongside so
# far-tail behaviour stays visible.  For 5 parameters the 99.99% point of a
# chi2_5 is 25.7, so 30 is comfortably outside any credible region.
DELTA_CHI2_RELEVANT = 30.0

# Stage-0 gate (the deprecated three-way plan's "normal path" thresholds, kept
# verbatim so this campaign is judged by a pre-existing bar, not a new one).
GATE = {
    "p95_abs_delta_chi2": 0.10,
    "max_abs_delta_chi2": 0.50,
    "posterior_mean_drift_sigma": 0.10,
    "posterior_width_relative_change": 0.10,
    "posterior_correlation_change": 0.10,
}
N_EMULATOR_SEEDS = 3


def draw_mixture(n: int, u_map, covariance, rng, recipe) -> tuple[np.ndarray, np.ndarray]:
    """IID draws from the frozen round-1 mixture, with their component labels.

    Drawn directly, with no candidate ranking, minimum-separation rejection or
    post-draw selection: any of those would change the sampling density and
    invalidate the stored proposal log-density.
    """

    weights = recipe["weights"]
    labels = rng.choice(list(weights), size=n, p=[weights[k] for k in weights])
    draws = np.empty((n, u_map.size), dtype=np.float64)
    for name, cov in (("guide", covariance),
                      ("broadened", recipe["broaden_covariance_factor"] * covariance),
                      ("prior", np.eye(u_map.size))):
        mask = labels == name
        if not np.any(mask):
            continue
        mean = np.zeros(u_map.size) if name == "prior" else u_map
        draws[mask] = rng.multivariate_normal(mean, cov, size=int(mask.sum()))
    return draws, labels


def mixture_log_prob(u: np.ndarray, u_map, covariance, recipe) -> np.ndarray:
    """Normalized log-density of the mixture, needed for any importance use."""

    from scipy.stats import multivariate_normal

    u = np.atleast_2d(u)
    components = [
        (recipe["weights"]["guide"], u_map, covariance),
        (recipe["weights"]["broadened"], u_map,
         recipe["broaden_covariance_factor"] * covariance),
        (recipe["weights"]["prior"], np.zeros(u_map.size), np.eye(u_map.size)),
    ]
    densities = np.stack([
        weight * multivariate_normal(mean=mean, cov=cov, allow_singular=False).pdf(u)
        for weight, mean, cov in components
    ])
    return np.log(densities.sum(axis=0))


_PREDICTOR_CACHE: dict[int, object] = {}


def _vmapped_predictor(problem):
    """Cache the jitted vmap per problem.

    Building ``jax.jit(jax.vmap(...))`` inside the call retraces and recompiles on
    every invocation.  That is invisible for a one-shot design evaluation and
    catastrophic inside an MCMC likelihood, where it is called once per step: the
    factorized arm's emcee run did not finish in 10 minutes with the function
    rebuilt each time.
    """

    key = id(problem)
    if key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = jax.jit(jax.vmap(problem.predict_u))
    return _PREDICTOR_CACHE[key]


def batched_predict(problem, u: np.ndarray, chunk: int = 16) -> np.ndarray:
    """Forward evaluations in fixed-size chunks.

    A single vmap over the whole design OOMs: 161 evaluations at this grid asked
    for 70 GB.  The chunk size matches the campaign's batched_apply.
    """

    fn = _vmapped_predictor(problem)
    out = np.empty((u.shape[0], msc.VECTOR_SIZE), dtype=np.float64)
    for start in range(0, u.shape[0], chunk):
        block = u[start:start + chunk]
        if block.shape[0] == chunk:
            out[start:start + chunk] = np.asarray(fn(jnp.asarray(block)), dtype=np.float64)
        else:
            for offset, row in enumerate(block):
                out[start + offset] = np.asarray(problem.predict_u(jnp.asarray(row)),
                                                 dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Forward model returned non-finite predictions on the design")
    return out


class LogResponseEmulator:
    """PCA + independent-GP emulator of ``log mu(theta)``, with uncertainty.

    Emulating the response linearly does not work and it is worth recording why:
    over the section-5 mixture the per-band raw response spans up to a factor 105
    (median 3.8), so a stationary GP cannot fit both the guide core and the prior
    tail.  Measured at N=48, a linear-space emulator gives p95 |delta chi2| of
    ~8.2e3 and a degenerate fit (every length scale pinned at its upper bound).

    In log space the same target has a per-band standard deviation of 0.16
    (max 0.71) across the identical design.  That is the well-conditioned
    quantity, and it is also the physically natural one: these amplitudes respond
    multiplicatively to the gas parameters.  The bands are strictly positive
    everywhere on the prior, so the transform is safe.
    """

    # The response is a smooth 5-parameter family, so log mu lives on a low-dimensional
    # manifold: measured on this design the PCA spectrum is 0.91, 0.063, 0.025, 1.8e-3,
    # 2.0e-4, ... and 9 components already carry 1 - 1e-6 of the variance.  Keeping
    # components down to 1e-8 added four GPs fitting numerical noise and cost ~40% of
    # the runtime for nothing.
    def __init__(self, u_train, mu_train, *, seed: int,
                 variance_target: float = 1.0 - 1.0e-6, max_components_cap: int = 12,
                 max_component_fraction: float = 1.0 / 3.0):
        mu_train = np.asarray(mu_train, dtype=np.float64)
        if np.any(mu_train <= 0.0):
            raise ValueError("Log-space emulation requires strictly positive training responses")
        self.log_train = np.log(mu_train)
        self.u_mean = u_train.mean(axis=0)
        self.u_scale = np.where(u_train.std(axis=0) > 0, u_train.std(axis=0), 1.0)
        x = (u_train - self.u_mean) / self.u_scale

        max_components = max(1, min(int(max_component_fraction * u_train.shape[0]),
                                    u_train.shape[0] - 1, mu_train.shape[1],
                                    int(max_components_cap)))
        self.pca = PCA(n_components=max_components, svd_solver="full", random_state=seed)
        scores = self.pca.fit_transform(self.log_train)
        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components = int(min(max_components, np.searchsorted(cumulative, variance_target) + 1))
        self.score_scale = np.where(scores[:, :self.n_components].std(axis=0) > 0,
                                    scores[:, :self.n_components].std(axis=0), 1.0)
        self.models = []
        for index in range(self.n_components):
            kernel = (ConstantKernel(1.0, (1e-4, 1e4))
                      * Matern(length_scale=np.ones(x.shape[1]),
                               length_scale_bounds=(1e-2, 1e2), nu=2.5)
                      + WhiteKernel(1e-8, (1e-12, 1e-2)))
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                             n_restarts_optimizer=2, random_state=seed + index)
            model.fit(x, scores[:, index] / self.score_scale[index])
            self.models.append(model)

    def _scores(self, u, with_std: bool):
        x = (np.atleast_2d(u) - self.u_mean) / self.u_scale
        means, stds = [], []
        for index, model in enumerate(self.models):
            if with_std:
                mean, std = model.predict(x, return_std=True)
                stds.append(std * self.score_scale[index])
            else:
                mean = model.predict(x)
            means.append(mean * self.score_scale[index])
        return np.stack(means, axis=1), (np.stack(stds, axis=1) if with_std else None)

    def predict(self, u, *, with_variance: bool = False):
        """Return mu, and optionally the GP score variances and the PCA basis.

        The emulator covariance is rank-k, not diagonal.  In log space
        ``Cov(log mu) = B^T diag(sigma^2) B`` with ``B`` the retained PCA
        components; propagating only its diagonal through the whitening is not an
        approximation of that matrix but a different object, and measured here it
        inflated the posterior width by up to 5.4x.  So the factors are returned
        and the likelihood uses the exact low-rank form via Woodbury.
        """

        means, stds = self._scores(u, with_variance)
        padded = np.zeros((means.shape[0], self.pca.n_components_), dtype=np.float64)
        padded[:, :self.n_components] = means
        log_mu = self.pca.inverse_transform(padded)
        mu = np.exp(log_mu)
        if not with_variance:
            return mu
        return mu, stds ** 2, self.pca.components_[:self.n_components]   # (n,42), (n,k), (k,42)


def low_rank_gaussian_log_likelihood(residual, factors, score_variance):
    """log N(0; I + M S M^T) for whitened residuals, exactly, at rank k.

    ``factors`` is M with shape (n, 42, k) and ``score_variance`` is S's diagonal
    with shape (n, k).  Woodbury gives

        (I + M S M^T)^-1 = I - M (S^-1 + M^T M)^-1 M^T
        logdet(I + M S M^T) = logdet(S^-1 + M^T M) + logdet(S)

    which costs a k x k solve per sample instead of a 42 x 42 one, and is exact
    rather than a diagonal stand-in.
    """

    n, _, k = factors.shape
    s_inv = 1.0 / np.maximum(score_variance, 1e-300)
    gram = np.einsum("nik,nil->nkl", factors, factors)
    middle = gram + np.stack([np.diag(row) for row in s_inv])
    projected = np.einsum("nik,ni->nk", factors, residual)
    solved = np.linalg.solve(middle, projected[..., None])[..., 0]
    quadratic = np.einsum("ni,ni->n", residual, residual) - np.einsum("nk,nk->n", projected, solved)
    sign, logabsdet = np.linalg.slogdet(middle)
    if np.any(sign <= 0):
        return np.full(n, -np.inf)
    log_det = logabsdet + np.sum(np.log(np.maximum(score_variance, 1e-300)), axis=1)
    return -0.5 * quadratic - 0.5 * log_det


def guarded(log_prob):
    """emcee must never receive a NaN: a single one poisons every walker silently."""

    def wrapped(u_batch):
        value = np.asarray(log_prob(u_batch), dtype=np.float64)
        return np.where(np.isfinite(value), value, -np.inf)

    return wrapped


def run_emcee(log_prob, dim, *, seed, n_walkers=40, n_steps=4000, burn=None, start=None,
              scale=0.5, vectorize=True):
    # burn must scale with n_steps: a fixed 1000 silently discards the whole chain
    # for any short smoke run, and get_chain then returns an empty array.
    burn = max(1, n_steps // 4) if burn is None else int(burn)
    if burn >= n_steps:
        raise ValueError(f"burn={burn} must be smaller than n_steps={n_steps}")
    rng = np.random.default_rng(seed)
    origin = np.zeros(dim) if start is None else np.asarray(start, dtype=np.float64)
    p0 = origin[None, :] + scale * rng.normal(size=(n_walkers, dim))
    safe = guarded(log_prob)
    initial = safe(p0)
    if not np.all(np.isfinite(initial)):
        raise RuntimeError(
            f"{int(np.sum(~np.isfinite(initial)))}/{n_walkers} emcee start points have "
            f"non-finite log-probability; the chain would never move"
        )
    sampler = emcee.EnsembleSampler(n_walkers, dim, safe, vectorize=vectorize)
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    if chain.size == 0 or not np.all(np.isfinite(chain)):
        raise RuntimeError("emcee produced an empty or non-finite chain")
    return chain, {
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorrelation_time": [float(v) for v in sampler.get_autocorr_time(quiet=True)],
        "n_walkers": n_walkers, "n_steps": n_steps, "burn": burn,
        "n_samples": int(chain.shape[0]),
    }


def posterior_summary(chain: np.ndarray) -> dict:
    return {
        "mean": chain.mean(axis=0).tolist(),
        "sd": chain.std(axis=0, ddof=1).tolist(),
        "correlation": np.corrcoef(chain, rowvar=False).tolist(),
    }


def compare_posteriors(reference: np.ndarray, trial: np.ndarray) -> dict:
    ref_mean, ref_sd = reference.mean(axis=0), reference.std(axis=0, ddof=1)
    trial_mean, trial_sd = trial.mean(axis=0), trial.std(axis=0, ddof=1)
    ref_corr = np.corrcoef(reference, rowvar=False)
    trial_corr = np.corrcoef(trial, rowvar=False)
    off = ~np.eye(reference.shape[1], dtype=bool)
    return {
        "mean_drift_sigma": np.abs((trial_mean - ref_mean) / ref_sd).tolist(),
        "max_mean_drift_sigma": float(np.max(np.abs((trial_mean - ref_mean) / ref_sd))),
        "width_relative_change": ((trial_sd - ref_sd) / ref_sd).tolist(),
        "max_abs_width_relative_change": float(np.max(np.abs((trial_sd - ref_sd) / ref_sd))),
        "max_abs_correlation_change": float(np.max(np.abs(trial_corr[off] - ref_corr[off]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi")
    parser.add_argument("--guide", type=pathlib.Path, default=None,
                        help="transfer_and_guide.json; defaults to the canonical mock_sbi directory "
                             "so that redirecting --output-dir does not silently lose the input")
    parser.add_argument("--recipe", choices=sorted(MIXTURE_RECIPES), default="defensive")
    parser.add_argument("--design-from-chain", type=pathlib.Path, default=None,
                        help="DIAGNOSTIC ONLY. Draw the design by resampling an existing "
                             "posterior chain instead of the mixture, to isolate 'is the "
                             "architecture capable when the design covers the posterior?' "
                             "from 'is the design right?'.  Circular as a production "
                             "design -- it uses the answer -- but the production route to "
                             "the same coverage is the sequential rounds.")
    parser.add_argument("--target", choices=("full", "factorized"), default="full",
                        help="'full' emulates the whole response (conservative). "
                             "'factorized' emulates the ratio to a cheap coarse-grid "
                             "baseline and multiplies it back, which is the production "
                             "architecture (r = mu_paste / mu_theory).")
    parser.add_argument("--baseline-grid", type=int, nargs=4, default=(64, 48, 22, 64),
                        help="Coarse forward grid used as the factorized arm's cheap "
                             "baseline. The default is this project's previous production "
                             "grid, documented FAIL at 1e-2 median against the converged "
                             "grid, so the ratio has a realistic amplitude and real "
                             "theta-dependence rather than an invented synthetic form.")
    parser.add_argument("--n-design", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--emcee-steps", type=int, default=4000)
    parser.add_argument("--skip-exact-emcee", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    guide_path = args.guide or (msc.REPO_ROOT / "data/SBI_validate/mock_sbi/transfer_and_guide.json")
    guide_payload = json.loads(guide_path.read_text())
    # Centre the oracle design on the RAW theory guide, because the exact posterior of
    # the raw theory model is the one we already have an independent NUTS chain for.
    # The recipe is the section-5 mixture; only its centre differs from production.
    u_map = np.asarray(guide_payload["raw_theory_guide"]["u_map"], dtype=np.float64)
    covariance = np.asarray(guide_payload["raw_theory_guide"]["covariance"], dtype=np.float64)

    print("[1/6] building the forward model ...", flush=True)
    problem = build_problem(contract_path=msc.INFERENCE_CONTRACT_PATH)
    cholesky = problem.cholesky
    observation = problem.observation
    whitened_observation = np.linalg.solve(cholesky, observation)
    # Whitening a covariance is L^-1 C L^-T; the emulator covariance is carried in
    # its exact low-rank factored form (see low_rank_gaussian_log_likelihood).
    cholesky_inverse = np.linalg.inv(cholesky)

    print(f"[2/6] drawing {args.n_design} design points from the '{args.recipe}' mixture ...", flush=True)
    rng = np.random.default_rng(args.seed)
    recipe = MIXTURE_RECIPES[args.recipe]
    if args.design_from_chain is not None:
        chain = np.load(args.design_from_chain)["u"].reshape(-1, 5)
        pick = rng.choice(chain.shape[0], size=args.n_design, replace=False)
        u_design = chain[pick]
        labels = np.full(args.n_design, "chain", dtype=object)
        print(f"      design resampled from {args.design_from_chain.name} "
              f"({chain.shape[0]} samples)")
    else:
        u_design, labels = draw_mixture(args.n_design, u_map, covariance, rng, recipe)
    counts = ({"chain": int(args.n_design)} if args.design_from_chain is not None
              else {name: int(np.sum(labels == name)) for name in recipe["weights"]})
    print(f"      component counts {counts}")

    print(f"[3/6] evaluating the stand-in simulator at {args.n_design} points ...", flush=True)
    t0 = time.time()
    predictions = batched_predict(problem, u_design)
    eval_seconds = time.time() - t0
    whitened = np.linalg.solve(cholesky, predictions.T).T
    print(f"      {eval_seconds:.1f}s total = {eval_seconds/args.n_design*1000:.1f} ms/eval")

    baseline_problem = None
    baseline_design = None
    if args.target == "factorized":
        print(f"      building the coarse baseline at grid {tuple(args.baseline_grid)} ...", flush=True)
        baseline_problem = build_problem(grid=tuple(args.baseline_grid),
                                         contract_path=msc.INFERENCE_CONTRACT_PATH)
        t0 = time.time()
        baseline_design = batched_predict(baseline_problem, u_design)
        print(f"      baseline evaluated in {time.time()-t0:.1f}s; "
              f"median |ratio-1| = {np.median(np.abs(predictions/baseline_design - 1.0)):.3e}",
              flush=True)

    def exact_log_prob(u_batch):
        u_batch = np.atleast_2d(np.asarray(u_batch, dtype=np.float64))
        preds = batched_predict(problem, u_batch, chunk=u_batch.shape[0])
        residual = whitened_observation[None, :] - np.linalg.solve(cholesky, preds.T).T
        return -0.5 * np.einsum("ij,ij->i", residual, residual) - 0.5 * np.einsum("ij,ij->i", u_batch, u_batch)

    reference_chain = None
    reference_info = {}
    if not args.skip_exact_emcee:
        print("[4/6] exact reference posterior by emcee on the true model ...", flush=True)
        t0 = time.time()
        reference_chain, reference_info = run_emcee(exact_log_prob, 5, seed=args.seed + 1,
                                                   n_steps=args.emcee_steps, start=u_map)
        reference_info["wall_seconds"] = time.time() - t0
        print(f"      {reference_info['n_samples']} samples, acceptance "
              f"{reference_info['acceptance_fraction']:.3f}, {reference_info['wall_seconds']/60:.1f} min")

    print("[5/6] cross-checking the reference against the archived NUTS chain ...", flush=True)
    nuts_path = msc.REPO_ROOT / "data/SBI_validate/three_probe_inference/hmc_v2/run01/hmc_samples.npz"
    nuts_comparison = None
    if nuts_path.is_file():
        nuts = np.load(nuts_path)["u"].reshape(-1, 5)
        if reference_chain is not None:
            nuts_comparison = compare_posteriors(nuts, reference_chain)
            print(f"      emcee vs NUTS: max mean drift {nuts_comparison['max_mean_drift_sigma']:.3f} sigma, "
                  f"max width change {nuts_comparison['max_abs_width_relative_change']:+.3f}, "
                  f"max corr change {nuts_comparison['max_abs_correlation_change']:.3f}")
        if reference_chain is None:
            reference_chain = nuts
            reference_info = {"source": "archived production NUTS chain", "n_samples": int(nuts.shape[0])}
    elif reference_chain is None:
        raise RuntimeError("No reference posterior available")

    print("[6/6] budget ladder ...", flush=True)
    design_chi2 = np.einsum("ij,ij->i", whitened_observation[None, :] - whitened,
                            whitened_observation[None, :] - whitened)
    chi2_floor = float(design_chi2.min())
    print(f"      design chi2: min {chi2_floor:.1f}  median {np.median(design_chi2):.1f}  "
          f"max {design_chi2.max():.3g}")
    print(f"      posterior-relevant subset = chi2 <= {chi2_floor + DELTA_CHI2_RELEVANT:.1f}")

    eligible = [n for n in BUDGET_LADDER if n < args.n_design]
    largest_n = max(eligible) if eligible else None
    results = []
    for n_train in eligible:
        if args.target == "factorized":
            train_target = predictions[:n_train] / baseline_design[:n_train]
            hold_baseline = baseline_design[n_train:]
        else:
            train_target = predictions[:n_train]
            hold_baseline = None
        u_train, mu_train = u_design[:n_train], train_target
        u_hold, w_hold = u_design[n_train:], whitened[n_train:]
        chi2_hold = design_chi2[n_train:]
        relevant = chi2_hold <= chi2_floor + DELTA_CHI2_RELEVANT
        if relevant.sum() < 8:
            print(f"      N={n_train:4d}  skipped: only {int(relevant.sum())} "
                  f"posterior-relevant holdout points")
            continue

        seed_records = []
        emulators = []
        for seed_index in range(N_EMULATOR_SEEDS):
            emulator = LogResponseEmulator(u_train, mu_train, seed=args.seed + 100 * seed_index)
            raw_prediction = emulator.predict(u_hold)
            if hold_baseline is not None:
                raw_prediction = raw_prediction * hold_baseline
            predicted = np.linalg.solve(cholesky, raw_prediction.T).T
            residual_true = whitened_observation[None, :] - w_hold
            residual_emu = whitened_observation[None, :] - predicted
            delta = np.abs(np.einsum("ij,ij->i", residual_emu, residual_emu)
                           - np.einsum("ij,ij->i", residual_true, residual_true))
            seed_records.append({
                "seed_index": seed_index,
                "n_pca_components": int(emulator.n_components),
                "relevant": {
                    "n_points": int(relevant.sum()),
                    "p95_abs_delta_chi2": float(np.percentile(delta[relevant], 95)),
                    "median_abs_delta_chi2": float(np.median(delta[relevant])),
                    "max_abs_delta_chi2": float(delta[relevant].max()),
                },
                "all_holdout": {
                    "n_points": int(delta.size),
                    "p95_abs_delta_chi2": float(np.percentile(delta, 95)),
                    "median_abs_delta_chi2": float(np.median(delta)),
                    "max_abs_delta_chi2": float(delta.max()),
                },
                "median_relative_response_error": float(
                    np.median(np.abs(raw_prediction - predictions[n_train:])
                              / np.abs(predictions[n_train:]))),
            })
            emulators.append(emulator)

        worst = {
            "p95_abs_delta_chi2": max(r["relevant"]["p95_abs_delta_chi2"] for r in seed_records),
            "max_abs_delta_chi2": max(r["relevant"]["max_abs_delta_chi2"] for r in seed_records),
            "all_holdout_p95_abs_delta_chi2": max(r["all_holdout"]["p95_abs_delta_chi2"] for r in seed_records),
        }
        holdout_ok = (worst["p95_abs_delta_chi2"] <= GATE["p95_abs_delta_chi2"]
                      and worst["max_abs_delta_chi2"] <= GATE["max_abs_delta_chi2"])
        # The posterior comparison is the expensive half (two emcee runs, plus a forward
        # evaluation per step in the factorized arm).  Run it where it can change the
        # decision -- where the cheap holdout gate passes -- and always at the largest N,
        # so the ladder never ends without at least one measured posterior.
        evaluate_posterior = holdout_ok or n_train == largest_n
        entry = {"n_train": n_train, "n_holdout": int(u_hold.shape[0]),
                 "n_relevant_holdout": int(relevant.sum()), "seeds": seed_records,
                 "worst_over_seeds": worst, "holdout_gate_passed": bool(holdout_ok)}

        if not evaluate_posterior:
            entry.update({"gate": None, "gate_passed": False,
                          "posterior_skipped": "holdout gate failed and this is not the largest N"})
            results.append(entry)
            print(f"      N={n_train:4d}  p95|dchi2| {worst['p95_abs_delta_chi2']:9.4f}  "
                  f"max {worst['max_abs_delta_chi2']:9.4f}  "
                  f"relerr {seed_records[0]['median_relative_response_error']:.2e}  "
                  f"(posterior not evaluated)", flush=True)
            continue

        emulator = emulators[0]

        def emulator_log_prob(u_batch, emulator=emulator):
            u_batch = np.atleast_2d(np.asarray(u_batch, dtype=np.float64))
            mu, score_variance, basis = emulator.predict(u_batch, with_variance=True)
            if baseline_problem is not None:
                mu = mu * batched_predict(baseline_problem, u_batch, chunk=u_batch.shape[0])
            residual = np.linalg.solve(cholesky, (observation[None, :] - mu).T).T
            # M = L^-1 diag(mu) B^T is the whitened square root of C_emu.  log mu =
            # log ratio + log baseline and the baseline is deterministic, so the same
            # factorization holds for both arms with the full mu.
            factors = np.einsum("ij,nj,kj->nik", cholesky_inverse, mu, basis)
            return (low_rank_gaussian_log_likelihood(residual, factors, score_variance)
                    - 0.5 * np.sum(u_batch ** 2, axis=1))

        def emulator_log_prob_no_cemu(u_batch, emulator=emulator):
            u_batch = np.atleast_2d(np.asarray(u_batch, dtype=np.float64))
            mu = emulator.predict(u_batch)
            if baseline_problem is not None:
                mu = mu * batched_predict(baseline_problem, u_batch, chunk=u_batch.shape[0])
            residual = np.linalg.solve(cholesky, (observation[None, :] - mu).T).T
            return -0.5 * np.sum(residual ** 2, axis=1) - 0.5 * np.sum(u_batch ** 2, axis=1)

        chain, info = run_emcee(emulator_log_prob, 5, seed=args.seed + 2,
                                n_steps=args.emcee_steps, start=u_map)
        seed_records[0]["posterior"] = compare_posteriors(reference_chain, chain)
        seed_records[0]["posterior_info"] = info
        seed_records[0]["posterior_summary"] = posterior_summary(chain)
        # Null control on the emulator-uncertainty term: how much of the posterior is
        # set by C_emu rather than by the emulated mean.  It is a diagnostic, so it runs
        # only where it is most informative -- at the largest N and at any passing N.
        if n_train == largest_n or all(
            worst[key] <= GATE[key] for key in ("p95_abs_delta_chi2", "max_abs_delta_chi2")
        ):
            chain_no_cemu, _ = run_emcee(emulator_log_prob_no_cemu, 5, seed=args.seed + 3,
                                         n_steps=args.emcee_steps, start=u_map)
            seed_records[0]["posterior_without_emulator_covariance"] = compare_posteriors(
                reference_chain, chain_no_cemu)
        posterior = seed_records[0]["posterior"]

        gate = {
            "p95_abs_delta_chi2": worst["p95_abs_delta_chi2"] <= GATE["p95_abs_delta_chi2"],
            "max_abs_delta_chi2": worst["max_abs_delta_chi2"] <= GATE["max_abs_delta_chi2"],
            "posterior_mean_drift_sigma": posterior["max_mean_drift_sigma"] <= GATE["posterior_mean_drift_sigma"],
            "posterior_width_relative_change": posterior["max_abs_width_relative_change"] <= GATE["posterior_width_relative_change"],
            "posterior_correlation_change": posterior["max_abs_correlation_change"] <= GATE["posterior_correlation_change"],
        }
        entry.update({"gate": gate, "gate_passed": all(gate.values())})
        results.append(entry)
        nc = seed_records[0].get("posterior_without_emulator_covariance")
        null_note = (f"[no-Cemu width {nc['max_abs_width_relative_change']:+7.3f}]  "
                     if nc is not None else "")
        print(f"      N={n_train:4d}  p95|dchi2| {worst['p95_abs_delta_chi2']:9.4f}  "
              f"max {worst['max_abs_delta_chi2']:9.4f}  drift {posterior['max_mean_drift_sigma']:6.3f}s  "
              f"width {posterior['max_abs_width_relative_change']:+7.3f}  "
              f"corr {posterior['max_abs_correlation_change']:6.3f}  "
              f"relerr {seed_records[0]['median_relative_response_error']:.2e}  "
              f"{null_note}{'PASS' if entry['gate_passed'] else 'fail'}", flush=True)

    passing = [entry["n_train"] for entry in results if entry["gate_passed"]]
    recommended = min(passing) if passing else None
    payload = {
        "status": "PASS" if recommended is not None else "FAIL",
        "recommended_n_train": recommended,
        "interpretation": (
            "Conservative upper bound: this emulates the FULL response with no transfer "
            "factorization. Production emulates r(theta) = mu_paste/mu_theory, whose variation "
            "over the design is smaller, so the real requirement is at most this."
        ),
        "gate_thresholds": GATE,
        "delta_chi2_relevant": DELTA_CHI2_RELEVANT,
        "design_chi2_floor": chi2_floor,
        "ladder": results,
        "design": {
            "n_design": args.n_design,
            "mixture_recipe_name": args.recipe,
            "mixture_recipe": recipe,
            "component_counts": counts,
            "centre": "raw theory Laplace guide (so an independent exact NUTS chain exists)",
            "u_map": u_map.tolist(),
            "seed": args.seed,
            "u_design_sha256": sha256_array(u_design),
            "prediction_sha256": sha256_array(predictions),
            "forward_eval_seconds_total": eval_seconds,
            "forward_ms_per_eval": 1000.0 * eval_seconds / args.n_design,
        },
        "guide_path": str(guide_path),
        "guide_sha256": msc.sha256_file(guide_path),
        "reference_posterior": reference_info,
        "reference_vs_archived_nuts": nuts_comparison,
        "identity": {
            "grid": list(GRID),
            "contract_sha256": problem.contract.contract_sha256,
            "numerical_sources": numerical_source_manifest(),
            "backend": backend_manifest(),
        },
        "elapsed_seconds": time.time() - started,
    }
    tag = "chaindesign" if args.design_from_chain is not None else args.recipe
    out = args.output_dir / f"oracle_budget_test_{tag}_{args.target}.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    np.savez_compressed(args.output_dir / f"oracle_budget_design_{tag}_{args.target}.npz",
                        u_design=u_design, labels=labels.astype("U10"),
                        predictions=predictions, whitened=whitened,
                        reference_chain=reference_chain)
    print(f"\nstatus {payload['status']}   recommended N = {recommended}   wrote {out}")
    return 0 if recommended is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
