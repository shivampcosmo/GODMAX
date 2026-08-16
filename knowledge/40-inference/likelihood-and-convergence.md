---
id: kb.inference.likelihood-and-convergence
title: Whitened likelihood, retained rank, and the diagnostics required before quoting a result
layer: 40-inference
owner: inference-statistician
status: verified
confidence: medium
scope:
  - notebooks/xDESI/survey_measure/godmax_multiprobe_hmc_stage31.py
  - notebooks/xDESI/survey_measure/combine_godmax_hmc_stage31_workers.py
  - notebooks/xDESI/survey_measure/godmax_multiprobe_map_stage31.py
  - notebooks/xDESI/survey_measure/optimize_godmax_multiprobe_stage31.py
  - notebooks/xDESI/survey_measure/plot_godmax_hmc_bestfit_residuals.py
  - notebooks/xDESI/survey_measure/plot_stage31_bestfit_vs_fiducial_cls.py
  - notebooks/xDESI/survey_measure/prune_stage31_cov_for_shearfix.py
  - run_scripts/pge/sample_params_v5.py
invariants:
  - INV-WHITEN-RANK-01
  - INV-CHI2-HONEST-01
  - INV-MCMC-TREEDEPTH-01
  - INV-MCMC-CONVERGENCE-01
  - INV-PRIOR-DESY3-01
checks:
  - "TODO(inference-statistician): assert the varied-parameter list is 31 with HOD slices [1:5]"
verified_at_commit: cf72943
verified_on: 2026-08-05
see_also: [kb.xdesi.analysis-state, kb.numerics.jax-contract]
scope_digest: sha256:1884ea9f987e9b619e3536d45105cd6b
---

## Claim

The likelihood is a full-covariance whitened chi2 whose degrees of freedom are set by an
eigenvalue cut, so **no chi2 is interpretable without its retained rank**, and no posterior is
quotable without r_hat, ESS, divergences and tree-depth saturation.

## Why it is true

**Whitening** (`INV-WHITEN-RANK-01`). From `BACKLIGHT_PASTE_HANDOFF_SUMMARY.md`:
`chi2 = || W (data − theory) ||^2`, with `W` built from the eigendecomposition of the
covariance **correlation** matrix at eigenvalue threshold 1e-8. For `fast1024` this retains
rank 459 of 460 in the recorded legacy-v1 product. The corrected `_pipev2_gshot` product
retains 460/460 at the same threshold and has not yet been fit. The threshold sets the degrees of freedom, so two chi2 values computed at
different retained ranks are not comparable — and lowering the threshold to admit more modes
inflates chi2 through noise-dominated directions rather than revealing a worse fit.

This interacts with precision: a 1e-8 cut is not float32-safe, so a dropped rank is often a
`INV-JAX-X64-01` violation rather than a covariance problem.

For schema-v2 high-resolution measurements, whitening first intersects any user ell cuts
with the saved `joint/data_vector_valid` mask. The requested archive has 920 slots, but the
four galaxy x ACT-kappa spectra each exclude seven transfer-null bands above ell 3000, so
the no-additional-cut analysis basis has 892 entries. Its covariance is the ordinary
principal submatrix `cov[np.ix_(valid, valid)]`; using the inverse of the full covariance
with zero placeholders would condition on invalid estimators and is forbidden. The retained
whitening rank remains product-specific and must be measured after assembly rather than
inherited from fast1024 or midres2048.

**Goodness of fit** (`INV-CHI2-HONEST-01`). Judge against `retained rank − n_varied`. For
Stage-31 `fast1024`: `459 − 31 = 428`, scatter of order `sqrt(2 × 428) ≈ 29`. The v1 best fit
at 7346.23 is not a good fit; the misfit is concentrated in `desi_g_auto` (6411 of the total).
See `kb.xdesi.analysis-state` for the per-family breakdown.

**Samplers in use.** NumPyro NUTS for Stage-31, with `chain_method: vectorized` fanned across
up to 16 GPU workers (`400x16`, `1000x2000` configurations exist). The repository also keeps
the NumPyro sampler entry point `run_scripts/pge/sample_params_v5.py`; no current tracked
blackjax Stage-31 entry point is in this document's scope.

**Multi-worker pooling** (`INV-MCMC-CONVERGENCE-01`) is the highest-risk operation here.
Pooling non-converged or mutually disagreeing workers manufactures a narrow, smooth posterior
that looks better than any individual worker. `combine_godmax_hmc_stage31_workers.py` performs
the pooling; `monitor_godmax_hmc_stage31_checkpoints.py` tracks progress.

**Tree depth** (`INV-MCMC-TREEDEPTH-01`). The v2 configuration uses `max_tree_depth: 4` with
`target_accept_prob: 0.85`, `num_warmup: 800`, `num_samples: 8000`, `num_chains: 4`. Depth 4
is low for 31 correlated parameters: a saturated tree truncates the trajectory before it
decorrelates, so NUTS degenerates toward a short-step random walk — while r_hat and
acceptance rate both look acceptable and the marginals come out too narrow.

**Priors** (`INV-PRIOR-DESY3-01`). DES Y3 Gaussian priors are fixed:
`Delta_z_bias_bin{1..4}` sigma `[0.018, 0.015, 0.011, 0.017]`; `mult_shear_bias_bin{1..4}`
(mean, sigma) `[(-0.006, 0.009), (-0.020, 0.008), (-0.024, 0.008), (-0.037, 0.008)]`.

**Galaxy shot-noise nuisance.** Current `_pipev2_gshot` likelihood products contain
signal-plus-shot-noise galaxy auto bandpowers. The JAX theory path windows the signal-only
GODMAX clustering spectrum first and then adds the saved decoupled bandpower template with a
per-photo-z amplitude. It recognizes optional sampled scalars
`desi_galaxy_shot_noise_amplitude_pz{1..4}`; if none are declared, it uses the fixed
`theory_to_data_vector.desi_galaxy_shot_noise_amplitudes` setting (default 1). No production
prior for these four amplitudes has been selected: adding them to a fit requires an explicit
prior and a corresponding update to the expected varied-parameter count and goodness-of-fit
degrees of freedom. Saved/best-fit values in `params.other_params` are honored by later
theory-vector consumers. A configuration containing both those values and an explicit fixed
`theory_to_data_vector` amplitude is rejected as ambiguous, preventing a sampled chain and
its saved best-fit file from producing different theory vectors.

**Cache identity.** New Stage-31 chains use contract
`stage31_chain_v4_gshot_validitymask`. Every worker NPZ records an SHA256 identity over the
exact selected data vector, covariance, whitener, bandpower responses, transfers and shot
templates, together with the archive-validity selection, measurement path, `map_product_id`,
galaxy-auto mean convention and ordered parameter names. A separate parameter-contract SHA256 covers each target,
fiducial and prior kind/bound/mean/sigma, so workers with the same names but different prior
files cannot be pooled. The combiner and MAP HMC-start reader fail closed on missing or
mismatched identities; the combiner also recomputes the selected minimum-chi2 sample under
the current likelihood before saving it.
Cached fiducial/best-fit theory vectors similarly carry an exact measurement-basis
fingerprint plus a second hash over the theory vector and its generation contract. That
contract includes an SHA256 over the materialized comparison configuration (merged params,
n(z), number-density/provenance metadata, wrapper options and source-product paths); best-fit
vectors additionally bind the saved best sample, likelihood identity and chi2. Plotting
also requires a content identity over the effective saved theory response (field metadata,
resolved transfers, bandpower windows and selected galaxy-shot templates). It rejects
unversioned, altered, stale-config, stale-response, stale-prior or stale-likelihood vectors
rather than comparing arrays with a loose tolerance. Cached vectors written before this
contract must be regenerated; relabelling an old NPZ is not valid provenance.

## The diagnostics required before quoting anything

1. r_hat per parameter, and the worst one named.
2. ESS per parameter, bulk and tail. Low ESS with r_hat ≈ 1 is the tree-depth signature.
3. Divergence count. Nonzero divergences with healthy acceptance usually means a non-finite
   gradient in a prior corner (`INV-JAX-GRAD-FINITE-01`) — route to `jax-numerics`.
4. Tree-depth saturation fraction.
5. Step size at the end of adaptation; pinned at the ceiling is a warning.
6. Per-worker agreement: best-fit chi2 spread and pairwise marginal overlap.
7. Absolute chi2 with retained rank and parameter count, plus the per-family breakdown.
8. Posterior predictive residuals per family. A family-localised misfit is a pipeline problem,
   not a model problem — route to `xdesi-lead`.

## How to verify

```bash
python tools/kb/kb.py invariants --check --layer inference
grep -n "max_tree_depth\|target_accept_prob\|num_samples\|chain_method" \
  param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml
```

On the cluster, read `fit_summary_stage31_multigpu.json` and report chi2, retained rank and
n_varied together — never separately.

## Failure modes

- **chi2 without retained rank.** Uninterpretable, and silently incomparable to any other
  chi2 in the project.
- **Relative improvement quoted as success.** 1.65e6 → 7.3e3 sounds decisive and the fit is
  still 17× its expectation.
- **Saturated tree depth read as convergence.** Narrow marginals, r_hat ≈ 1, very low ESS,
  step size pinned at the adaptation ceiling.
- **Pooled workers without a consistency check.** Pooled contours tighter than any individual
  worker's; multimodal pooled marginals from workers stuck in different basins.
- **Calibration absorbing a model failure.** Shear calibration parameters pulled many sigma
  from their prior means while chi2 improves. Report this; do not accept it.
- **Covariance surgery without a rank comparison.** `prune_stage31_cov_for_shearfix.py`
  modifies the object defining every uncertainty in the analysis. Any change needs retained
  rank before and after, the per-family chi2 effect, and `physics-referee` sign-off.
- **Widening a prior or lowering the eigenvalue cut to improve a fit.**
  `INV-PROC-NOTOLERANCE-01` — this converts a detected error into an undetected one.
- **Flat shot noise passed through the signal window.** The saved shot template is already
  the estimator's decoupled bandpower response; applying the galaxy pixel/mask response to it
  again biases both its amplitude and scale dependence.

## Open questions

- The v2 chains' saturation fraction, r_hat and ESS are not recorded. Until they are, **no v2
  posterior is quotable**. Owner: `inference-statistician`. Blocking.
- Whether `max_tree_depth: 4` was a deliberate cost trade-off or an oversight is not recorded.
  If deliberate, the justification belongs in the journal; if not, it needs raising before the
  next production run. Owner: `inference-statistician`.
- Derived from the handoff summary; not verified against
  `godmax_multiprobe_hmc_stage31.py` at line level. `confidence: medium`.
