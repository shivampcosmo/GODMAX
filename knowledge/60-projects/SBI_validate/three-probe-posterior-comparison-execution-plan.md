---
id: kb.sbi.three-probe-posterior-comparison-plan
title: Execution plan for the controlled three-probe posterior comparison
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/build_three_probe_inference_manifest.py
  - notebooks/SBI_validate/validate_three_probe_inference_manifest.py
  - notebooks/SBI_validate/three_probe_inference_contract.py
  - notebooks/SBI_validate/three_probe_jax_forward_model.py
  - notebooks/SBI_validate/validate_three_probe_jax_hmc_forward.py
  - notebooks/SBI_validate/run_hmc_three_probe_contract.py
  - notebooks/SBI_validate/submit_hmc_three_probe_contract.sbatch
  - notebooks/SBI_validate/plot_hmc_three_probe_contract.py
  - notebooks/SBI_validate/plot_hmc_sbi_three_probe_comparison.py
  - notebooks/SBI_validate/run_theory_sbi_three_probe_contract.py
  - notebooks/SBI_validate/submit_theory_sbi_three_probe_contract.sbatch
  - notebooks/SBI_validate/diagnose_theory_sbi_preprocessing.py
  - notebooks/SBI_validate/build_theory_sbi_pca_compression.py
  - notebooks/SBI_validate/pilot_theory_sbi_estimators.py
  - notebooks/SBI_validate/select_theory_sbi_pilot.py
  - notebooks/SBI_validate/pilot_theory_neural_importance.py
  - notebooks/SBI_validate/submit_theory_neural_importance_pilot.sbatch
  - notebooks/SBI_validate/run_theory_neural_importance_50k.py
  - notebooks/SBI_validate/submit_theory_neural_importance_50k.sbatch
  - notebooks/SBI_validate/validate_theory_neural_importance_5k.py
  - notebooks/SBI_validate/submit_theory_neural_importance_5k_validation.sbatch
  - notebooks/SBI_validate/plot_hmc_neural_importance_50k.py
  - notebooks/SBI_validate/run_theory_neural_importance_tail_refinement.py
  - notebooks/SBI_validate/submit_theory_neural_importance_tail_refinement.sbatch
  - notebooks/SBI_validate/run_theory_sbi_probit_mdn_65k.py
  - notebooks/SBI_validate/submit_theory_sbi_probit_mdn_65k.sbatch
  - notebooks/SBI_validate/plot_hmc_probit_mdn_65k.py
  - notebooks/SBI_validate/run_theory_sbi_three_probe_calibrated.py
  - notebooks/SBI_validate/submit_theory_sbi_three_probe_calibrated.sbatch
  - notebooks/SBI_validate/submit_theory_sbi_estimator_pilot.sbatch
  - tests/test_sbi_three_probe_inference_manifest.py
  - tests/test_sbi_three_probe_inference_contract.py
  - tests/test_sbi_three_probe_jax_forward_model.py
  - tests/test_sbi_three_probe_theory_sbi_contract.py
  - tests/test_sbi_three_probe_calibrated.py
invariants:
  - INV-WINDOW-CMP-01
  - INV-BEAM-01
  - INV-NMT-COUPLED-01
  - INV-NMT-BANDMAJOR-01
  - INV-PRODUCT-PROV-01
  - INV-JAX-SEED-01
  - INV-WHITEN-RANK-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_inference_manifest.py
  - "[needs-data] /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/SBI_validate/build_three_probe_inference_manifest.py --output-dir data/SBI_validate/three_probe_inference"
  - "[needs-data] /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/SBI_validate/validate_three_probe_inference_manifest.py --manifest data/SBI_validate/three_probe_inference/experiment_manifest.yaml"
  - "[needs-data] /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_inference_contract.py"
  - "[needs-data] /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_jax_forward_model.py"
  - "[needs-data] JAX_PLATFORMS=cpu /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/SBI_validate/validate_three_probe_jax_hmc_forward.py --output-dir data/SBI_validate/three_probe_inference/validation/jax_hmc_forward_vs_mock"
verified_at_commit: 29c3a27
verified_on: 2026-08-21
see_also:
  - kb.sbi.three-probe-posterior-comparison-execution
  - kb.sbi.three-probe-noisy-mock-covariance
  - kb.sbi.analytical-hmc-sbi
supersedes: []
scope_digest: sha256:42cec6bd6a53ad7919284a1d768369fc
---

## Claim

The controlled comparison proceeds only after one content-addressed 42-element observation
and its frozen noise/covariance contract have been built and validated.  Phase 1 does not
authorize existing legacy samplers: the new theory-HMC, theory-SBI, and mock-SBI consumers
must each validate and require this identity before they can run.  Their training-noise streams
are disjoint from it and from each other.

## Why it is true

The execution handoff fixes the common 14-band `gy,gkappa,gtau` vector, the revised
S/N-matched noise contract, the five-parameter box prior, and a single immutable noisy
observation.  Its older HMC/SBI and map-SBI scripts use incompatible vector sizes or map
resolution, so Phase 1 creates a fail-closed adapter before any sampler is launched.

### Execution sequence

1. Build and validate the manifest and observation.  Select realization 000 by the
   predeclared lowest-valid-index rule; combine its gy/gkappa parent realization with its
   revised gtau realization, and require exact equality to ensemble row 000.
2. Implement a JAX-native, map-matched five-parameter 42-vector evaluator; prove fiducial,
   scalar/batch, JIT, gradient, finite-difference, support, and grid-stability checks.
3. Run and certify four-chain direct theory HMC against the frozen Cholesky likelihood, but
   seal its samples and likelihood products from the theory-SBI team.
4. Train and certify blind theory SBI from `mu_theory(theta) + L_C epsilon`, without truth,
   HMC samples, HMC likelihood values, or HMC-driven architecture/acquisition/stopping choices.
5. Only after theory SBI is frozen, reveal the independently converged HMC product and run the
   pre-registered HMC-versus-SBI agreement report.  A failed HMC or SBI convergence/calibration
   gate blocks mock-SBI proposal construction; it is not repaired by tuning to agreement.
6. Convert the accepted blind theory-SBI posterior into a density-evaluable
   core/broadened/prior/tail proposal, then establish parameter-indexed nside-1024 pasted-map
   replay in a 60-point pilot.
7. Run one fixed tail-safe proposal round followed by exactly three mock-SBI active-learning
   rounds, splitting by theta and reserving map-noise holdouts.
8. Independently refute accepted products and make one hash-bound posterior comparison.

### Noise contract

`same noise` has three distinct, required meanings:

- all methods condition on the identical frozen observed vector;
- theory HMC and theory SBI use the exact same analytic covariance and Cholesky factor;
- mock SBI uses map-level noise drawn from the same dense curves, field conventions, mask,
  workspace, and bandpower estimator that generated that covariance.

Fresh theory-SBI and mock-SBI training draws are required.  Reusing the observation draw, or
one noise draw at every theta, would make the comparison statistically invalid.

### Theory-SBI information firewall

Theory SBI is a genuine inference exercise, not a recovery of a supplied truth. Before Phase C,
produce a training-facing inference view of the frozen manifest that contains only the observed
42-vector, prior bounds, fixed model settings, covariance/Cholesky, windows, source hashes, and
SBI seed namespaces. The five fiducial/truth values remain in a separate audit-only product for
post-run coverage and plotting; they must not be an argument, configuration field, proposal
seed, simulator centre, loss target, acquisition input, or stopping signal of the SBI runner.

The new theory-SBI runner must have no HMC-chain, HMC-checkpoint, or HMC-likelihood input.
Round 0 samples the original box prior. Later rounds may sample the SBI estimator's own posterior
conditioned on the frozen observation, provided every row records the normalized proposal density
and SNPE receives that proposal for correction. This is data-adaptive sequential SBI, not access
to the truth. HMC samples become visible only after theory-SBI architecture, seed ensemble, number
of simulations, and stopping rule are frozen.

Acceptance requires a static no-oracle test (the training contract contains no truth field and no
HMC artifact path), a run-log audit of the exact inputs, and held-out SBC/coverage plus posterior
predictive checks. Contour agreement with HMC is a final validation only.

### Phase-2 contract-loader prediction

- **Direction:** the builder emits a separate `inference_contract.yaml`; it contains no
  truth/fiducial value, audit-file path, or HMC artifact path, while preserving the exact
  observation/covariance/window identities required by all inference consumers.
- **Affected products:** the training-facing YAML and its read-only loader only. The existing
  audit manifest remains provenance metadata and never enters a posterior runner.
- **Null control:** mutating an audit-only truth value does not change the loaded inference
  arrays or the training contract digest. A training contract containing a recursive
  truth/fiducial/HMC-artifact field is rejected before it opens the observation.
- **Falsifier:** a training loader that admits a forbidden oracle field, fails to verify the
  pinned observation/covariance/Cholesky/window hashes, or accepts the audit manifest as an
  inference contract blocks Phase 2 and every posterior runner.

### Phase-2B exact-forward-model prediction

- **Direction:** the new x64 JAX evaluator reproduces the physical-table-cosh, real-space
  smoothed, transverse-8R200c operator. Because that smoothing is embedded in its projected
  y/e/m tables, it applies the frozen galaxy pixel window once, but no additional Bell, before
  the saved 14-band NaMaster window. A Bell is allowed only for an explicitly unsmoothed
  harmonic-theory branch, never in combination with embedded smoothing. It returns exactly 42 entries in
  `gy,gkappa,gtau` order and accepts neither a truth/fiducial point nor a noise seed.
- **Magnitude:** no model-to-map agreement threshold is predeclared here; forward agreement,
  AD-versus-finite-difference, support/grid, and posterior predictive limits must be measured
  before an HMC job is eligible.
- **Null control:** changing an observation or training noise seed cannot change
  `mu_theory(theta)`; the legacy spherical operator must fail the projected-operator equality
  test rather than be silently substituted.
- **Falsifier:** float32 table construction, a trace concretisation, non-finite/zero gradient,
  missing or doubled smoothing/transfer/window, an admitted partial band, or any host-only operation in
  the traced likelihood blocks HMC submission.

### Theory-HMC/SBI unlock gate

The two estimators are trained independently, then compared only after both are frozen. HMC must
pass its four-chain convergence gate; theory SBI must pass its held-out coverage/SBC, posterior
predictive, ensemble-seed, and added-simulation stability gates. The comparison report then gives
per-parameter mean/median shifts in a declared joint covariance, interval-width ratios, correlation
differences, and a classifier two-sample or Jensen--Shannon diagnostic. It is an unlock gate for
mock SBI, not a tuning objective: neither method may be rerun with information extracted from the
other merely to improve agreement.

### Five-round blind theory-SBI production identity

The production theory-SBI run uses SNPE-C for exactly five rounds with simulation counts
`20000,12000,14000,16000,18000` (80,000 total). Round 1 draws from the original five-dimensional box prior;
each later round draws from the immediately preceding SBI posterior conditioned on the
frozen observation, and passes that density to SNPE as the proposal for correction. The
simulator returns `L^{-1}(mu(theta)-x_obs) + epsilon`, so the network observation is exactly
zero and every simulation uses a fresh standard-normal draw from the theory-SBI seed namespace.

The forward-model discretisation is identical to the approved approximate HMC run: 64 aperture
nodes, 48 profile-radius nodes, 22 profile-redshift nodes, 24 mass nodes, 48 k nodes, and 64
requested Limber ell nodes (51 unique integer nodes), followed by interpolation to ell 0--2048
and the frozen transfer/window operator. This is an explicitly approximate common analysis,
not an exact replacement for the former high-resolution model. The runner may load only the
pinned training contract; a different round count, budget, grid, covariance/observation hash,
or seed namespace is a production-identity failure.

The frozen density estimator is a 10-component MDN with 128 hidden features, independent
standardization of theta and x, training batches of 256, learning rate `5e-4`, 10% validation,
at most 300 epochs, and early stopping after 30 epochs without validation improvement. Forward
simulation batches contain 16 rows. These values and the one-CUDA-device requirement are
fail-closed production settings, not runtime tuning knobs.

SNPE/MDN training is CPU-resident while the JAX forward simulator remains on the allocated GPU.
This separation is required because `sbi==0.21.0` constructs CPU-side tensors in its non-atomic
SNPE-C mixture correction; mixed CUDA/CPU training is a hard failure. A continuation after round 1
must CPU-map the saved posterior, replay its normalized proposal density, reuse the already saved
round-2 simulations exactly, and then continue rounds 2--5 without resimulating earlier rows.

Immediately after each round trains, the runner draws and atomically saves 30,000 bounded
posterior samples with normalized log density, parameter order, prior bounds, frozen observation,
contract digest, round number, and cumulative simulation count. Only after that file is complete
and hashed may the corresponding `round_N.ready.json` marker appear. These per-round files are
the contour interface while later active-learning rounds remain in progress.

After inspection of the five-round result, the user authorized exactly five additional active
rounds with 50,000 new simulations: rounds 6--10 use 10,000 simulations each. Round 6 proposes
from the frozen round-5 posterior. The continuation reconstructs the five historical proposal
objects and SNPE round indices from hash-bound artifacts, deep-copies the round-5 estimator for
training, and never resimulates or overwrites rounds 1--5. The forward grid, observation,
covariance, seeds, MDN architecture, and per-round 30,000-draw contour interface remain unchanged.

Pre-registered prediction: all five round arrays are finite and inside the prior, saved proposal
log densities are finite, identical derived round seeds replay the noise exactly, and changing
only the training-noise seed changes `epsilon` but not `mu(theta)`. Completion alone does not
establish convergence: round stability and held-out posterior-predictive diagnostics are required
before comparison with the sealed HMC product.

The rounds 6--10 extension completed on 2026-08-21 with 130,000 cumulative simulations and
30,000 saved posterior draws per round. It is **not certified converged**. The largest
round-9-to-10 mean displacement is 0.50 final-posterior standard deviations, and the 128 saved
posterior-predictive whitened chi-squares have minimum/median 270.2/364.7 versus the declared
`rank - n_varied = 42 - 5 = 37` reference. A diagnostic round-10 overlay with the rejected HMC
may be inspected, but neither its overlap nor its disagreement is an acceptance test. Held-out
coverage/SBC and independent-seed stability remain required before this theory posterior can
unlock mock SBI.

### Pre-registered 50k theory-SBI recovery campaign

The covariance/noise audit precedes any estimator change. HMC and SBI must load the same pinned
contract and lower Cholesky factor. The HMC residual is
`w_H = L^{-1}(x_obs - mu(theta))`; the SBI simulator is
`x = L^{-1}(mu(theta) - x_obs) + epsilon`, with `epsilon ~ N(0,I)`, and the NPE is conditioned
at zero. Their signs differ but their Gaussian likelihood norms are identical. A contract/hash,
triangular-solve, or unit-noise replay mismatch blocks all estimator experiments.

Raw cross spectra are signed and therefore must never be logged. The existing input is already
the full 42-dimensional Cholesky-whitened residual, followed by SBI's learned per-feature affine
standardisation. The bounded alternatives are: (A) the existing raw-whitened input with MDN,
(B) raw-whitened input with NSF/atomic SNPE-C, (C) elementwise `asinh(x)` with MDN, and
(D) elementwise `asinh(x)` with NSF/atomic SNPE-C. `asinh` is fixed before training, signed,
bijective, maps the zero observation to zero, and uses the unit whitened-noise scale; clipping,
absolute values, unsigned logs, data-dependent transforms, and HMC-informed compression are
prohibited. Each pilot uses identical train/validation rows and seeds.

Method selection is blind to HMC and uses held-out simulations. The selected estimator must have
finite in-prior samples/log densities, deterministic seed replay, and improve held-out local SBC
rank uniformity and empirical 68%/95% marginal coverage relative to the existing method. No
tolerance is adjusted after seeing a result. The fresh production budget is exactly 50,000
simulations, allocated `20000,10000,8000,6000,6000` across five sequential rounds; every later
round uses the immediately preceding posterior and the actual normalized proposal in SNPE-C.

Before spending that budget, a fixed-data pilot trains raw-MDN, signed-asinh-MDN, raw-NSF,
signed-asinh-NSF, and score-NSF on the same authenticated 16,000 prior rows and evaluates them on
the same disjoint 4,000 rows. Selection first requires finite held-out conditional log density and
atomic NSF (the MDNs are diagnostic controls), then the highest mean held-out `log q(theta|x)`.
As an observation-local safety gate, 256 blind
posterior draws must have exact-likelihood median chi-square no more than 10 above the blind
reference minimum and 95th percentile no more than 25 above it. The score basis is accepted only
if a dimension below 42 preserves local (`Delta chi2 <= 25`) relative chi-square with maximum
absolute error at most 0.5 and RMS error at most 0.1. These criteria are frozen before the pilot.

The fixed-data raw/asinh and 15-dimensional PCA-score pilots failed that local-likelihood gate;
therefore they cannot authorize the 50,000-simulation run. The final bounded recovery route
Rao--Blackwellizes the known Gaussian simulation noise instead of asking a conditional NPE to
learn the unobserved cancellation tail. For a proposal row it computes the exact blind target
weight `log w = -chi2(theta)/2 - log q(theta)`, where `chi2` uses only the frozen observation,
forward model, and Cholesky factor. A pilot fits an unconditional NSF to an importance resample
of the independently hash-pinned round-8 artifact, chosen before training because it has the
largest effective sample size among the inspected blind rounds. The pilot requires importance
ESS at least 100 and retains the same frozen local-likelihood gate: 256 independent flow draws
must have exact chi-square median no more than 10 above the input minimum and 95th percentile no
more than 25 above it. Failure ends this route; neither threshold may change.
The round-7 proposal pickle is independently hash-pinned and its normalized log density is
replayed. Because SBI's serialized leakage correction can change every row by one common additive
constant, the stored and replayed relative log densities must agree to `1e-5` after removing that
constant; the replayed values are then used in the weights. This does not relax the target because
the common constant cancels identically when weights are normalized.

This route is **likelihood-corrected neural importance inference**, not likelihood-free NPE. Its
production implementation, if the pilot passes, retains the frozen 50,000 forward-evaluation
budget and five-round split, uses an exact normalized 90% current-flow plus 10% box-prior proposal,
records proposal densities and target weights for every row, and publishes contour samples only
after hash-bound round completion. Truth parameters, fiducial vectors, mock products, audit
manifests, and HMC files are forbidden in pilot, training, selection, and stopping. A static
source scan and a pinned simulation SHA are submission blockers. The sealed diagnostic HMC may
be opened only after the final method, seeds, simulation budget, and stopping rules are frozen.

If the 50,000-evaluation run completes but the fixed 20% held-out subset has ESS below 200, no
threshold changes and the held-out rows are not recycled into a passing claim. One final frozen
validation-only extension may draw exactly 5,000 new rows from the round-5 90% bounded-flow plus
10% box-prior proposal, bringing the total to 55,000 (a 10% validation extension). The round-5
flow is not retrained. This independent set must satisfy the unchanged gates: ESS at least 200,
maximum normalized weight at most 0.05, all flow/importance mean shifts at most 0.25 pooled sigma,
and all 90% width ratios in `[0.9,1.1]`. Failure ends the blind recovery rather than changing the
split, proposal, or tolerances.

After that validation failed only its unchanged width floor for `theta_ej_0` (0.896) and
`nu_theta_ej_M` (0.888), the user explicitly authorized continued refinement. The exact 55,000-row
deterministic-mixture diagnostic has Pareto `k=0.781`, above the fixed reliability standard
`k<=0.7`; the tail estimate is therefore not reliable. This authorizes one final blind
tail-acquisition lap, not a tolerance change. The proposal is fixed before execution as 50%
round-5 bounded flow, 30% an exactly normalized logit-space broadened copy with scale vector
`[2,1,1,1,2]`, and 20% original box prior. Its centre is the componentwise median of the frozen
round-5 flow samples. The two broadened directions are selected solely by the failed blind width
diagnostics, not by the comparison chain.

This final lap draws exactly 5,000 active rows, recomputes the exact balance-mixture weights over
all 60,000 rows, and requires Pareto `k<=0.7`, ESS>=1,000, and maximum weight<=0.025 before fitting
a new bounded NSF with systematic resampling. It then draws a separate 5,000-row validation set
from the same 50/30/20 form built around the new flow, bringing the total to 65,000 evaluations.
Certification retains the unchanged independent gates: ESS>=200, maximum weight<=0.05, all mean
shifts<=0.25 pooled sigma, and all 90% width ratios in `[0.9,1.1]`. No further simulation
extension, broadening-scale/direction adjustment, or tolerance change is allowed if this fails.

After the method and final round are frozen, comparison to the sealed diagnostic HMC requires all
five marginal mean differences to be at most 0.5 pooled posterior standard deviations, all 90%
interval-width ratios to lie in `[0.8,1.25]`, and the last-round SBI mean shifts to be at most
0.25 final-posterior standard deviations with 90% width ratios in `[0.9,1.1]`. The final SBI must
also place its retained posterior draws in the same exact-likelihood chi-square region as HMC;
matching contours while assigning poor exact likelihood is a falsifier. Because the current HMC
has divergences and depth saturation, satisfying these gates establishes agreement with that
diagnostic target, not independent HMC certification.

### Blind tail-preserving mock-SBI campaign

Mock SBI receives only the accepted theory-SBI posterior density `q(theta)`, its normalized
proposal components, the training-facing inference contract, and mock simulations. It has no
truth/audit product, no HMC product, and no exact theory likelihood access. The initial mock
proposal is the normalized mixture already specified in the handoff: 40% theory-SBI posterior
core, 25% broadened theory-SBI component, 25% original box prior, and 10% explicitly normalized
theory-SBI tail/boundary component. Each row records `log p0`, `log q`, component, round, and
seed; sequential SNPE receives the actual proposal for density correction.

The default 480-paste budget is fixed as follows:

| use | unique pastes | role |
|---|---:|---|
| pilot plus immutable holdout | 60 | replay, domain, scaling, and coverage controls |
| round 0 | 180 | fixed tail-safe theory-SBI proposal mixture |
| active round 1 | 80 | mock posterior mass times ensemble disagreement |
| active round 2 | 80 | same acquisition rule, after held-out round-1 review |
| active round 3 | 80 | same acquisition rule, after held-out round-2 review |
| total | 480 | hard budget before a separately approved extension |

Every active round reserves at least 25% of unique theta locations for the original-prior and
theory-SBI-tail/boundary components. The tail definition and component normalisation are fixed
before any mock result is inspected. All three active rounds run even if an early contour appears
stable; a failed held-out gate labels the final mock posterior unconverged rather than changing
the tail fraction, coverage criterion, or round count.

### Historical precedent

The August `gy_gkappa_gtau_hmc_latest_vs_sbi_round3.pdf` is evidence that the repository has used
the intended **sequential SNPE mechanism**: its first round samples the prior, and later proposals
come from its own posterior at the observation. It is not a production template for this plan:
it uses a legacy 51-element fiducial theory product, requires an HMC artifact for contract
validation, and the overlaid HMC chains failed their convergence gate. It also lacks the held-out
coverage/SBC acceptance required here.

The saved arrays sharpen that interpretation. That run used 65,536 simulations
(`16384,16384,32768`), a 51-element idealized analytical observation, an exact box prior in a
standard-Normal probit basis, and a 10-component MDN on the full Cholesky-whitened vector. Its
SBI/HMC standardized marginal-mean shifts are small, but its SBI 90% widths are 1.29--1.44 times
the plotted HMC widths; the reference HMC also has two divergences and 74.825% maximum-depth
saturation. The plot is therefore a method clue rather than calibrated-equality evidence.

One controlled transplant is pre-registered. It keeps the sealed 42-element noisy observation,
covariance, Cholesky, Bell, windows, five priors, fixed physics, forward grid `(64,48,22,64)`, and
seed namespaces. It changes only the neural method to exact probit coordinates, a 10-component
MDN, the uncompressed whitened residual, and three rounds of `16384,16384,32768`. The neural
observation is zero and simulations are `L^-1(mu(theta)-d)+epsilon`, `epsilon~N(0,I)`; no signed
spectrum is logged or clipped.

Generating parameters, audit products, comparison chains, mock metadata, and legacy posterior
artifacts are prohibited from proposals, training, selection, and stopping. The blind prediction
is reduced boundary/tail error in `theta_ej_0` and `nu_theta_ej_M` relative to the physical-space
MDN. Non-finite replay, failure to reach the current exact-likelihood region, or unstable
round-2-to-3 means/widths falsifies it. Only after round 3 is frozen may the rejected diagnostic
comparison chain be opened. The existing gates remain unchanged: mean shifts <=0.5 pooled sigma
and 90% width ratios in `[0.8,1.25]` for all parameters.

Resource scaling is anchored to completed job 6921482, which evaluated 50,000 rows with the same
forward hash and `(64,48,22,64)` grid in 75:03 wall time. Linear forward scaling predicts about
98 minutes for 65,536 rows. The legacy 65,536-row MDN run recorded 18:33 total runtime, so one
H100, 15 CPUs, 120 GB, and a conservative three-hour wall limit leaves margin without changing
the numerical grid. This is a capacity estimate, not a posterior-validity claim.

### Pre-registered Phase-1 prediction

- **Direction:** realization 000 is selected before any posterior is available; its assembled
  42-vector equals ensemble row 000 bit-for-bit.
- **Affected products:** only the new manifest and observation product.
- **Null control:** changing no input but rebuilding preserves the vector, covariance,
  Cholesky, windows, and manifest digest; changing only a declared seed-domain label leaves
  those numerical arrays unchanged but makes validation fail.
- **Falsifier:** any ordering/hash mismatch, non-positive-definite covariance, normalized
  double-precision Cholesky reconstruction error above `1e-13`, or overlapping seed namespace
  blocks all samplers.  The stored covariance and Cholesky hashes remain exact identities.

## How to verify

Run the declared checks.  The data-backed validator must report a 42-element vector in
probe-major `gy,gkappa,gtau` order, 14 complete windows, a full-rank covariance, exact
Cholesky reconstruction, and source hashes matching the revised contract.

## Failure modes

- A parent tau realization is silently used after the S/N-matched revision.
- The 12-realization sample covariance replaces the analytic covariance.
- A 36- or 51-element legacy vector enters a new sampler.
- Observation, training, and holdout seed namespaces overlap.
- The incomplete `[2010, 2049)` band is admitted as an inference datum.

## Open questions

- The exact JAX-native evaluator cost and full-prior gradient behavior remain blocking Phase 2.
- The pilot determines the safe paste-array resources; no SLURM submission is authorized here.
