---
id: kb.sbi.three-probe-hmc-sbi-verification-summary
title: Three-probe HMC and theory-SBI verification experiments
layer: 60-projects
owner: inference-statistician
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/diagnose_theory_sbi_preprocessing.py
  - notebooks/SBI_validate/pilot_theory_sbi_estimators.py
  - notebooks/SBI_validate/run_theory_sbi_three_probe_calibrated.py
  - notebooks/SBI_validate/pilot_theory_neural_importance.py
  - notebooks/SBI_validate/run_theory_neural_importance_50k.py
  - notebooks/SBI_validate/validate_theory_neural_importance_5k.py
  - notebooks/SBI_validate/run_theory_neural_importance_tail_refinement.py
  - notebooks/SBI_validate/run_theory_sbi_probit_mdn_65k.py
  - notebooks/SBI_validate/plot_hmc_probit_mdn_65k.py
invariants:
  - INV-JAX-SEED-01
  - INV-WHITEN-RANK-01
  - INV-MCMC-TREEDEPTH-01
  - INV-MCMC-CONVERGENCE-01
  - INV-CHI2-HONEST-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_calibrated.py
see_also:
  - kb.sbi.three-probe-posterior-comparison-plan
  - kb.sbi.three-probe-posterior-comparison-execution
  - kb.sbi.analytical-hmc-sbi
supersedes: []
scope_digest: sha256:6db8884e12129341655e1a490ae6c6d7
verified_at_commit: 29c3a27
verified_on: 2026-08-21
---

# Three-probe HMC and theory-SBI verification experiments

## Executive conclusion

No theory-SBI posterior tested in this campaign is certified as converged or as agreeing with
the current HMC posterior. The experiments did establish the following:

1. HMC and SBI use the same frozen observation, covariance, lower Cholesky factor, parameter
   priors, forward model, Bell/window treatment, and numerical grid.
2. The HMC and SBI Gaussian likelihoods are algebraically identical. HMC evaluates
   `||L^-1(d-mu(theta))||^2`; SBI simulates `x=L^-1(mu(theta)-d)+epsilon` and conditions at
   `x=0`. The sign change disappears under the squared norm.
3. The SBI inputs are the signed, 42-dimensional Cholesky-whitened residuals. No logarithm is
   applied to the spectra. SBI's learned per-feature affine standardization is the only default
   input scaling.
4. The best-supported diagnosis for conditional-NPE failure is that the observed zero residual
   requires unusually strong noise cancellation: the exact posterior occupies a rare tail relative
   to ordinary noisy simulator draws. This is not proven to be the unique cause. Changing MDN/NSF
   architecture, applying signed `asinh`, PCA compression, adding rounds, or using exact probit
   parameter coordinates did not solve the observed failure.
5. Exact-likelihood-corrected neural importance inference reaches the correct local chi-square
   region, but its importance tail remains unreliable. The final targeted tail acquisition had
   Pareto `k=1.016`, so it was stopped before refitting.
6. The current HMC artifact is itself rejected: 15 divergences and 22.883% depth-6 saturation.
   Its median chi-square is 271.0 for nominal `rank-n_varied=42-5=37`, so agreement with it would
   not certify an acceptable absolute fit.

The current generating parameter and audit manifest were excluded from every SBI training and
proposal path. The current depth-6 HMC samples were excluded from training, proposal construction,
and every blind numerical stopping gate. The likelihood-corrected campaign froze its methods
before opening that HMC. The later probit-MDN transplant was instead motivated by inspecting a
legacy HMC/SBI comparison; it remained oracle-free in training and stopping, but its architecture
selection was HMC-comparison-informed and is therefore only a diagnostic method test.

## Frozen inference problem

| Quantity | Frozen value |
|---|---|
| probes | `gy`, `gkappa`, `gtau` |
| data-vector length | 42: 14 bands per probe |
| varied parameters | `theta_ej_0`, `alpha_nt`, `mu_beta`, `theta_co_0`, `nu_theta_ej_M` |
| forward grid | aperture/profile-r/profile-z/Limber-ell = `64/48/22/64` |
| contract hash | `9f9c92ed806a88dc6b989612feb6d9dc159c2b48bf49dfbc28cff7693cf54547` |
| forward hash | `88b048368377395202a85beeb945b54baa2d99fc234f2ae06ee6204a51b41768` |
| likelihood | Gaussian with the frozen 42x42 covariance and lower Cholesky factor |

The training contract deliberately rejects fields or values containing generating-point, audit,
or HMC information. Observation seeds are disjoint from all training and holdout seed namespaces.

## HMC reference

The current diagnostic HMC is:

`data/SBI_validate/three_probe_inference/hmc_depth6/job_6917288`

Configuration: 1,000 warm-up steps, 3,000 samples in each of four vectorized chains, dense mass,
maximum tree depth 6. It completed, but the convergence gate rejected it:

- 15 divergences;
- 22.883% transitions at maximum depth;
- chi-square min/median/95th percentile = 265.64/271.02/276.64;
- nominal expected goodness-of-fit reference = `42-5=37`, with scatter about 8.6.

The HMC posterior may be used only as a diagnostic contour target. It is not a certified chain
and its absolute fit is unacceptable.

## Experiment inventory

### 1. Original sequential conditional NPE: ten rounds, 130,000 simulations

Artifact:

`data/SBI_validate/three_probe_inference/theory_sbi_5round/job_6919080`

The initial five rounds used 80,000 simulations; five additional rounds added 50,000. Each round
saved 30,000 posterior samples. The estimator was an MDN conditioned on the full whitened
42-vector. It was blind to the generating point.

Result: not converged. The largest round-9-to-10 displacement was 0.50 final-posterior standard
deviations. Posterior-predictive whitened chi-square among 128 saved draws had minimum/median
270.2/364.7. More rounds did not cure the rare-conditioning problem.

### 2. Fixed-data estimator and preprocessing pilots

Controlled pilots used identical authenticated simulation rows and seeds to test:

- raw whitened residual + MDN;
- signed `asinh` whitened residual + MDN;
- raw whitened residual + NSF/atomic SNPE-C;
- signed `asinh` residual + NSF/atomic SNPE-C;
- 15-dimensional PCA/score compression + NSF.

No spectrum was logged because the cross spectra and whitened residuals can be signed. `asinh`
was the only nonlinear input transform, fixed before training and preserving zero.

Result: every candidate failed the frozen observation-local exact-likelihood gate. Posterior draws
did not reliably occupy the exact low-chi-square region. PCA also failed to preserve local relative
chi-square accurately enough. No candidate was promoted to a 50,000-simulation conditional-NPE
production run.

### 3. Likelihood-corrected neural-importance pilot

Artifact:

`data/SBI_validate/three_probe_inference/calibrated_50k/job_6921423`

This method used the exact blind target weight

`log w(theta) = -chi2(theta)/2 - log q(theta)`

on a hash-pinned proposal artifact. It then fit an exactly bounded flow using a logit
change-of-variables with an analytic Jacobian.

Result: pilot passed. Importance ESS was 207.17 and exact chi-square min/median/95th percentile
was 266.46/270.73/277.58. This demonstrated that Rao-Blackwellizing the known Gaussian noise
could reach the correct likelihood region with far fewer simulations than conditional NPE.

### 4. Five-round likelihood-corrected production run: 50,000 simulations

Artifact:

`data/SBI_validate/three_probe_inference/calibrated_50k/job_6921482`

Budget: `20000,10000,8000,6000,6000`; every proposal retained a 10% box-prior component. Exact
multiple-proposal deterministic-mixture weights were used. Twenty percent of each round was held
out. The job completed all forward evaluations in 75:03 and intentionally exited nonzero because
the frozen blind gates failed.

Final diagnostics:

| diagnostic | result | gate | status |
|---|---:|---:|---|
| training ESS | 1989.2 | >=1000 | pass |
| maximum training weight | 0.0138 | <=0.02 | pass |
| held-out ESS | 96.75 | >=200 | fail |
| maximum held-out weight | 0.0946 | <=0.05 | fail |

Round-4-to-5 means and widths were stable, but held-out weights showed that rare tail points still
dominated. The flow was especially too narrow in `theta_ej_0` and `nu_theta_ej_M`.

### 5. Independent 5,000-simulation validation extension

Artifact:

`data/SBI_validate/three_probe_inference/calibrated_50k/job_6922972`

The round-5 flow was not retrained. Five thousand new rows were drawn from the frozen 90% flow +
10% prior proposal, taking the total to 55,000 evaluations.

- ESS = 342.73: pass;
- maximum weight = 0.0432: pass;
- all mean shifts <=0.25 pooled sigma: pass;
- 90% width ratios for `theta_ej_0` and `nu_theta_ej_M` = 0.896 and 0.888: fail against the
  unchanged `[0.9,1.1]` interval.

The exact 55,000-row balance diagnostic had ESS 1290.76, maximum weight 0.02454, and Pareto
`k=0.781`. Since `k>0.7`, the tail estimate was not reliable even though conventional ESS looked
adequate.

### 6. Reconstruction of the favorable legacy 65,536-simulation plot

Legacy plot:

`notebooks/SBI_validate/outputs/theory_sbi/sbi_five_parameter_probe_sequential_3round/comparison_plots/final_round_only/gy_gkappa_gtau_hmc_latest_vs_sbi_round3.pdf`

The legacy run used three rounds `16384,16384,32768`, an exact box-to-standard-Normal probit
parameterization, a 10-component MDN, and the uncompressed 51-dimensional whitened vector.

The plot looked close in location, but recomputation showed SBI/HMC 90% width ratios of
1.29--1.44. Its HMC had two divergences and 74.825% depth-5 saturation. It also used a different
51-element idealized analytical observation and different fixed physics. Thus it was a useful
method clue, not evidence that the current problem had previously been solved. Because this plot
motivated the subsequent architecture choice, the transplant was not a fully HMC-blind
method-selection experiment, although its runner never accessed any HMC sample or generating
parameter and its numerical gates did not depend on HMC.

### 7. Controlled legacy-style probit-MDN transplant: 65,536 simulations

Artifact:

`data/SBI_validate/three_probe_inference/probit_mdn_65k/job_6923231`

Job 6923231 copied only the legacy neural choices onto the current sealed 42-vector contract:
three rounds `16384,16384,32768`, exact probit coordinates, full whitened input, and a
10-component MDN. Sources, transitive numerical dependencies, catalog, data products, package
versions, seeds, proposals, and per-round outputs were content-addressed. It completed in 674.59
seconds.

Blind result:

- round-3 chi-square min/median/95th percentile = 267.21/318.56/604.98;
- round-2-to-3 shifts exceeded the stability gate for `alpha_nt` and `mu_beta`;
- round-2-to-3 widths failed for three parameters.

Diagnostic comparison to HMC:

| parameter | mean shift, pooled sigma | SBI/HMC 90% width |
|---|---:|---:|
| `theta_ej_0` | 0.915 | 2.250 |
| `alpha_nt` | 0.249 | 1.685 |
| `mu_beta` | 0.449 | 1.066 |
| `theta_co_0` | 0.126 | 1.033 |
| `nu_theta_ej_M` | 0.717 | 1.590 |

Frozen comparison gates were mean shifts <=0.5 and widths in `[0.8,1.25]`. The method therefore
failed, particularly in the two target tail directions.

Diagnostic plot:

`data/SBI_validate/three_probe_inference/probit_mdn_65k/job_6923231/hmc_vs_probit_mdn_round3_diagnostic.pdf`

PDF SHA-256: `1fb46927aacbfe1ecc9b6b529fc9c06db485b6ffddbc975dd94814ec27360e92`.
An independent referee verified the input hashes, statistics, parameter ordering, rendering,
rejection labels, and absence of a generating-parameter marker.

### 8. Targeted likelihood-corrected tail acquisition

Artifact:

`data/SBI_validate/three_probe_inference/calibrated_50k/job_6923255`

The preregistered acquisition proposal was 50% round-5 bounded flow, 30% an exactly normalized
logit-space broadened component with scale `[2,1,1,1,2]`, and 20% box prior. The broadened
directions were selected from blind validation failures, not HMC. Five thousand new simulations
were evaluated.

Result:

| diagnostic | result | gate | status |
|---|---:|---:|---|
| pooled ESS | 1290.31 | >=1000 | pass |
| maximum weight | 0.01881 | <=0.025 | pass |
| Pareto `k` | 1.0159 | <=0.7 | fail |

The job stopped before flow refitting and before the independent 5,000-row validation set, as
required. `k>1` indicates a severely heavy or effectively unbounded importance-weight tail; ESS
alone was misleading. No posterior or updated triangle plot was published from this run.

## What caused the apparent HMC/SBI discrepancy

The discrepancy is not explained by different injected noise, covariance, Bell correction,
window projection, parameter prior, grid, or a log transform. Those identities were explicitly
checked and content-addressed.

The best-supported diagnosis is observation-local rarity; the experiments do not prove it is the
unique cause. For much of the proposal mass,
`||L^-1(mu(theta)-d)||^2` is hundreds, while the conditional network is asked to learn the density
at zero after adding a typical unit-Normal noise vector. Simulations close to zero require rare
noise cancellation. Sequential conditional NPE therefore learns a smooth conditional density
from almost no examples in the relevant region. Exact likelihood weighting removes simulator
noise variance and finds that region, but current proposals still undersample the correlated
tails, particularly those involving `theta_ej_0` and `nu_theta_ej_M`.

There is a second limitation: the HMC reference is not converged and has an unacceptable absolute
chi-square. Consequently, contour agreement with it cannot be treated as ground truth, and some
HMC/SBI width disagreement may be HMC bias from divergences and depth saturation.

## Methods falsified by these experiments

- Adding more MDN active-learning rounds without changing the rare-conditioning geometry.
- Raw versus signed-`asinh` input scaling as the primary fix.
- NSF instead of MDN on the same fixed-data conditional problem.
- Fifteen-dimensional PCA/score compression under the tested preservation gate.
- Exact probit coordinates plus the legacy three-round MDN recipe on the current observation.
- Treating conventional importance ESS as sufficient when Pareto `k` is unreliable.
- Interpreting the visually favorable legacy plot as a calibrated comparison.

## Remaining scientifically defensible options

The preregistered refinement route is exhausted and its thresholds must not be changed after the
failure. A new campaign requires a new, independently reviewed charter. Defensible options are:

1. obtain a genuinely converged reference posterior using deeper, better-adapted HMC or another
   exact-likelihood sampler, then reassess whether the present HMC widths were biased;
2. use an exact-likelihood SMC/tempered sampler or a proposal specifically designed from a
   deterministic optimization/Laplace approximation, with tail reliability validated before
   neural fitting;
3. train a likelihood estimator rather than a posterior estimator, then sample the learned
   likelihood with a separately validated MCMC/SMC algorithm;
4. investigate why the fixed noisy observation has chi-square about 270 against a rank-42 model,
   because neither HMC nor SBI can provide a scientifically acceptable fit while that absolute
   discrepancy remains.

None of these options authorizes use of the generating parameter in training or selection.

## Reproduction and evidence

Primary command-bound evidence ledger:

`knowledge/.kb/ledgers/2026-08-21-three-probe-sbi-calibrated-50k.md`

Core checks:

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  tests/test_sbi_three_probe_calibrated.py

python -m json.tool \
  data/SBI_validate/three_probe_inference/probit_mdn_65k/job_6923231/diagnostics.json

python -m json.tool \
  data/SBI_validate/three_probe_inference/calibrated_50k/job_6923255/tail_refinement.failed.json

sha256sum \
  data/SBI_validate/three_probe_inference/probit_mdn_65k/job_6923231/\
hmc_vs_probit_mdn_round3_diagnostic.pdf
```

The independent referee reproduced the posterior summaries, chi-square diagnostics, artifact
hashes, agreement-gate failures, and final plot. All posterior results in this document remain
labeled rejected or uncertified.

## Failure modes for future work

- A run is called successful because it completed, despite failing blind calibration gates.
- HMC contour smoothness is mistaken for convergence despite divergences or depth saturation.
- ESS is quoted without maximum weight and Pareto `k`.
- HMC is consulted while choosing an SBI proposal, transform, round count, or stopping point.
- The generating parameter enters the training contract, proposal, or plot.
- A favorable contour overlay is accepted without exact-likelihood and round-stability checks.
- Thresholds or broadening scales are changed after observing a failure.

## Open questions

- Why does the frozen noisy observation have chi-square near 270 for a rank-42 likelihood?
- How much of the narrow HMC tail is caused by divergences and depth-6 saturation?
- Can an exact-likelihood tempered sampler cover the correlated tail with reliable Pareto
  diagnostics at a practical simulation budget?
