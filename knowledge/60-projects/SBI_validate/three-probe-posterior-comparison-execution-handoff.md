---
id: kb.sbi.three-probe-posterior-comparison-execution
title: Execution handoff for theory HMC, theory SBI, and pasted-mock SBI
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/three_probe_noise_contract.py
  - notebooks/SBI_validate/three_probe_noiseless_estimator.py
  - notebooks/SBI_validate/three_probe_noiseless_theory.py
  - notebooks/SBI_validate/run_hmc_five_parameter_probe_scan.py
  - notebooks/SBI_validate/run_sbi_five_parameter_probe_sequential.py
  - notebooks/SBI_validate/run_simulator_native_active_sbi.py
  - notebooks/SBI_validate/theory_sbi_utils.py
  - notebooks/SBI_validate/map_sbi_pasted_utils.py
  - notebooks/SBI_validate/run_map_sbi_pasted_worker.py
  - notebooks/SBI_validate/three_probe_fast_paste.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-PHYS-UNITS-01
  - INV-NZ-NORM-01
  - INV-WINDOW-CMP-01
  - INV-BEAM-01
  - INV-SHOTNOISE-01
  - INV-NMT-BANDMAJOR-01
  - INV-NMT-COUPLED-01
  - INV-PRODUCT-PROV-01
  - INV-JAX-X64-01
  - INV-JAX-GRAD-FINITE-01
  - INV-JAX-SEED-01
  - INV-MCMC-CONVERGENCE-01
  - INV-MCMC-TREEDEPTH-01
  - INV-WHITEN-RANK-01
  - INV-CHI2-HONEST-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - python tools/kb/kb.py invariants --check --id INV-JAX-X64-01 --id INV-JAX-SEED-01 --id INV-PROC-NOTOLERANCE-01
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_noise_contract.py tests/test_sbi_three_probe_noiseless_estimator.py tests/test_sbi_three_probe_noiseless_theory.py
  - "[needs-data] /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/SBI_validate/validate_three_probe_inference_manifest.py --manifest data/SBI_validate/three_probe_inference/experiment_manifest.yaml"
verified_at_commit: 29c3a27
verified_on: 2026-08-20
see_also:
  - kb.sbi.three-probe-noisy-mock-covariance
  - kb.sbi.three-probe-noiseless-cl-validation
  - kb.sbi.three-probe-fast-paste
  - kb.sbi.three-probe-resolved-theory
  - kb.sbi.simulator-native-efficient
  - kb.inference.likelihood-and-convergence
supersedes:
  - kb.sbi.three-way-mock-comparison-plan
scope_digest: sha256:c8495968e482f4d7be22c0bce5d43510
---

## Claim

The next production objective is one controlled comparison of three posterior estimators:
direct theory HMC, theory SBI, and pasted-mock SBI. All three must use the same five-parameter
box prior, truth, fixed parameters, 42-element `gy,gkappa,gtau` observation, exact NaMaster
windows, nside-1024 transfer convention, and frozen analytic covariance. The mock-SBI branch
may spend only a few hundred unique signal pastes and therefore uses a theory-SBI-informed,
tail-preserving proposal followed by two or three active-learning rounds.

The validated inputs exist, but none of the three production posteriors exists yet. In
particular, the older 51-element/17-band and 36-element/nside-512 drivers are templates only;
they must not be launched unchanged. A posterior is physically quotable only after the common
manifest, differentiable theory path, sampler convergence, proposal correction, held-out SBI
coverage, and final independent comparison gates below have passed.

## Why it is true

### Current state: what is complete

The following work is complete and is the starting point for every new session.

1. **Selected halo lightcone.** The frozen parent is
   `AbacusBacklight_base_c0000_ph000`, with strict `0.3 < z < 0.5`,
   `5e11 <= M_particle_proxy/(Msun/h) < 1e16`, and the proxy provisionally identified with
   M200c. The catalog selection contains 66,159,463 halos. This is an operational mass
   convention, not a validated spherical-overdensity mass calibration.
2. **Fiducial signal maps.** The accepted signal product is
   `data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/`
   `abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5`, SHA256
   `86a26333e3336fef8ff85c0b1511533e9ec422d576d9600ba611638763e449f9`.
   It uses nside 1024, RING ordering, `physical_table_cosh` with 32 line-of-sight nodes,
   an 8 R200c transverse aperture, `(nr,nM,nz)=(48,24,48)`, and a Gaussian profile smoothing
   FWHM equal to half a pixel: 1.7177432 arcmin. The Gaussian belongs to the pasted continuous
   profiles and must be represented once, never twice, in any theory-to-map response.
3. **Angular basis.** Dense theory, maps, masks, and coupling are carried through
   `ell_max=2048`. Inference retains the 14 complete native bands with right-exclusive edges
   `[80,101,127,160,201,253,319,401,505,636,801,1008,1268,1597,2010]`.
   The vector is 42 elements in the exact order
   `gy[0:14], gkappa[14:28], gtau[28:42]`. Modes 2010--2048 enter the harmonic coupling
   calculation, but the incomplete native band `[2010,2049)` is diagnostic only and is not an
   inference datum. This is the frozen meaning of “up to ell_max 2048.”
4. **Noise and covariance.** The production input is
   `data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/`
   `noise_contract_tau_snrmatch_gkappa.h5`, SHA256
   `ecbd23f63dc96a8cd77910f96a62d0568ac6b6a1e928f12d7cb66bb724b9681`.
   It contains the exact dense noise curves, common mask, NaMaster response, 42 by 42 analytic
   covariance and Cholesky factor. The effective tau white-noise depth is
   `0.023266988679843306 tau arcmin`; it was selected to match the forecast gtau amplitude
   S/N to gkappa. Forecast amplitude S/N values are 87.3635 for gy and 81.9036 for both
   gkappa and gtau. The y noise includes official SO Deproj-2 noise plus the Gaussian
   all-halo-minus-pasted-slice yy term; kappa uses official SO iterative MV reconstruction
   noise. The same dense curves must generate mock noise and define theory covariance.
5. **Noise replay evidence.** Twelve deterministic noisy realizations and their measurements
   exist in `noisy_ensemble_tau_snrmatch_gkappa.h5`, SHA256
   `c3ecbd4e8035217a508b2778b56a1febc280aeec8d955207d3f42d33e565ba70`.
   Their sample covariance has rank at most 11 and is diagnostic only. It must never replace
   the analytic 42 by 42 HMC/SBI covariance.
6. **Noiseless paste-versus-theory work.** The nside-1024 comparison through ell 2048,
   cosmology, realized HOD n(z), CMB lensing efficiency, smoothing, aperture, and large- and
   small-scale mismatch hypotheses have been recorded in the linked noiseless, resolved,
   projected-operator, and fast-paste documents. These validate the fiducial experiment
   inputs; they do not validate parameter derivatives or a posterior.

### Current state: what is not complete

No production HMC, theory-SBI, or mock-SBI posterior has been run for this 42-vector setup.
The following are blockers, not optional refinements.

- There is no content-addressed experiment manifest selecting one observed 42-vector and one
  truth/noise seed for all three methods.
- The existing five-parameter HMC and sequential-SBI scripts target older products and vector
  lengths. Their sampler structure and parameter declarations may be reused, but their cached
  theory products, observation, covariance, response, and assertions may not.
- The exact parameter-varying nside-1024 projected-painter theory has not yet been proven to
  be JAX differentiable through all five parameters. Host NumPy/SciPy profile transforms are
  acceptable for evidence generation or theory SBI, but they cannot sit inside NUTS.
- No accepted theory-SBI posterior or normalized tail proposal exists.
- No parameter-indexed paste bank, mock-SBI active-learning rounds, or final GetDist triangle
  exists.

### Frozen common statistical experiment

Create `data/SBI_validate/three_probe_inference/experiment_manifest.yaml` and a machine-readable
HDF5/NPZ observation product before any sampler run. The manifest must hash every dependency
and freeze all choices below.

| contract item | required value |
|---|---|
| probes/order | `gy`, `gkappa`, `gtau`; 14 bands each; 42-vector probe-major order |
| harmonic support | dense `0 <= ell <= 2048`; inference bands end at ell 2009 |
| map geometry | nside 1024, RING, the saved common mask and exact NaMaster workspace/windows |
| signal transfer | saved half-pixel profile smoothing and map/pixel convention, applied once |
| halo support | strict `0.3<z<0.5`, `5e11<=Mproxy<1e16 Msun/h`, unresolved completion off |
| cosmology | exact c0000 catalog cosmology and its saved hash |
| kernels | realized HOD galaxy n(z), consumed HOD nbar, and saved CMB Wkappa arrays/hashes |
| covariance | the frozen analytic 42 by 42 covariance and Cholesky from the S/N-matched contract |
| observed vector | one immutable noisy pasted fiducial vector, with the chosen noise seed saved |
| truth | the five fiducial values below |

All methods must condition on the same immutable observed vector. The recommended choice is
one predeclared realization from the accepted noisy pasted ensemble. Do not choose the seed
after inspecting posterior agreement. The observation product must save the raw maps or exact
map hashes, measured bandpowers, 42-vector, covariance, windows, noise seed, estimator code
hash, and manifest SHA256.

The shared Gaussian likelihood is

`log L(theta) = -0.5 * ||L_C^{-1} [d_obs - mu(theta)]||^2 + constant`,

where `L_C` is the frozen Cholesky factor. No covariance parameter is varied in this first
five-parameter comparison. The retained rank must be measured and reported; the expected
value is 42 if the frozen covariance remains full rank. With five varied parameters, an
absolute best-fit goodness-of-fit comparison uses `rank - 5`, not an improvement relative to
the fiducial.

### Frozen five-parameter model

Parameter order, labels, fiducials, and original box priors are inherited from
`run_hmc_five_parameter_probe_scan.py:41-60` and must be identical in every product.

| index | name | fiducial | prior |
|---:|---|---:|---:|
| 0 | `theta_ej_0` | 2.0 | Uniform(0.5, 8.0) |
| 1 | `alpha_nt` | 0.05 | Uniform(0.0, 0.5) |
| 2 | `mu_beta` | 0.5 | Uniform(0.005, 1.5) |
| 3 | `theta_co_0` | 0.05 | Uniform(0.001, 0.5) |
| 4 | `nu_theta_ej_M` | -0.1 | Uniform(-1.0, 1.0) |

The fixed complement is
`log10_Mstar0_theta_ej=15`, `nu_theta_ej_z=0`, `log10_Mc0=13.75`,
`delta_rhogas=7`, and `gamma_rhogas=2`. Cosmology, HOD, mass selection, smoothing, noise,
mask, and kernels are also fixed. Any later change is a new experiment identity, not a resume.

### Phase A — build one inference-ready theory evaluator

Implement one 42-vector function `mu_theory(theta)` that consumes the exact resolved
map-matched support and projects dense Cls through the saved transfers and bandpower windows.
Both HMC and theory SBI must call this same public evaluator or demonstrably equivalent pure
and batched wrappers.

For HMC, x64 must be enabled before any JAX array is created. Static input validation and
content hashing occur outside tracing. Every parameter must have a finite, nonzero expected
gradient for at least one affected band, both eagerly and under `jax.jit`; finite differences
at interior points must agree with autodiff at the pre-registered numerical accuracy. Test
prior-center and near-boundary points. A host conversion, zero gradient, NaN, or support/cache
mismatch blocks HMC. If the exact projected-table construction cannot be differentiated,
build a separate JAX-native transform or a differentiable emulator validated over the whole
prior and tails. Do not call a host NumPy bridge from inside NumPyro.

Required nulls are: the fiducial output reproduces the frozen theory vector; changing only the
noise seed cannot change `mu_theory`; changing only a parameter changes the expected probe
families; batch and scalar evaluation agree; and the evaluator never consumes low-mass
completion or the partial final band.

### Phase B — theory HMC

Run direct NumPyro NUTS against the common Gaussian likelihood and original box prior. Start
with at least four genuinely independent chains, x64, a benchmarked target acceptance near
0.9, and a tree depth high enough not to truncate trajectories. A reasonable first production
allocation is at least 2,000 warmup and 4,000 retained draws per chain; extend rather than thin
if the diagnostics fail.

The posterior is not accepted until all of the following hold:

- rank and absolute chi2 are reported with `n_varied=5` and per-probe residuals;
- rank-normalized split R-hat is at most 1.01 for every parameter;
- bulk and tail ESS are each at least 1,000 for every parameter for the pooled production
  result;
- divergences are zero, E-BFMI is above 0.3 per chain, and tree-depth saturation is below 1%;
- chain marginals and per-chain best-fit chi2 agree, and posterior predictive residuals show
  no unreported probe-localized failure.

If a gate fails, increase warmup/samples/depth or reparameterize after diagnosing gradients.
Never widen a prior, drop a band, regularize the covariance, or loosen a gate to obtain a pass.

### Phase C — theory SBI

Theory SBI targets exactly the same posterior as HMC. Generate simulations as
`x = mu_theory(theta) + L_C epsilon`, with `theta` drawn from the original five-dimensional
box prior and `epsilon` drawn from a versioned standard-normal seed stream. Whiten with the
same Cholesky used by HMC. Because theory calls are cheap relative to pasting, use tens of
thousands of simulations rather than imposing the mock budget; a practical starting design is
50,000 prior simulations followed by one or two 25,000-simulation focused rounds if coverage
requires them.

Train an ensemble of neural posterior estimators with independent initialization and data
seeds. Split train/validation/test by unique `theta`, not by noise realization. Acceptance
requires simulation-based calibration or empirical rank coverage over held-out observations,
posterior predictive checks, stability across network seeds and added simulations, and no
boundary leakage. HMC samples must remain hidden while choosing the architecture, simulations,
or stopping point; HMC-versus-theory-SBI agreement is a final validation, not an acquisition
signal.

### Phase D — convert theory SBI into a tail-safe mock proposal

The accepted theory-SBI posterior is a proposal `q(theta)`, never a replacement prior. Every
expensive parameter row must save `log p0(theta)`, `log q(theta)`, proposal component, round,
and random seed. Sequential SNPE must receive the actual proposal for correction, or an
importance-aware NLE/NRE route must apply the equivalent density ratio. A posterior sample
cloud relabelled as a prior is invalid.

Use a normalized mixture with known density. The initial default allocation for unique paste
locations is:

- 40% accepted theory-SBI posterior core;
- 25% a normalized broadened posterior component in a bounded/probit parameterization;
- 25% original box prior;
- 10% an explicitly normalized tail/boundary component.

The broadened and tail components must have evaluable normalized densities. Deterministic
corner points may be used as diagnostics, but must be excluded from density-corrected training
unless generated by a valid continuous proposal. Require material occupancy in the outer
posterior quantiles for every parameter; do not define “tail” after seeing the mock result.

### Phase E — pasted-mock simulator and active learning

The target is the common-random-numbers, fixed-phase response conditional on c0000/ph000 and
the frozen galaxy realization. For reference point `theta_ref`, define

`mu_mock(theta) = mu_theory(theta_ref) + [b_paste(theta) - b_paste(theta_ref)]`,

where `b_paste` is the exact 42-vector measured from nside-1024 pasted maps with the frozen
estimator. This anchoring prevents the known absolute fiducial paste-versus-theory offset from
being mistaken for parameter response, but it does not turn one phase into an ensemble mean.
The final mock posterior must be labelled conditional on this phase and HOD realization.

The five varied parameters affect the continuous y/tau/kappa profiles, while the frozen
galaxy catalog, halo catalog, geometry, cosmology, kernels, smoothing, aperture, and HOD remain
unchanged. Paste each unique `theta` once, save all signal maps, and reuse that signal for all
noise realizations. Add y, kappa, and tau noise at map level using exactly the dense curves and
field conventions in the frozen contract, then run the exact same NaMaster estimator. Do not
inject independent bandpower noise as a substitute for the map pipeline.

Use 8 training noise realizations per unique signal paste and reserve 4 additional seeds for
held-out validation. Repeated noise draws at one `theta` are conditionally useful but are not
new signal simulations; report both unique-paste and returned-vector counts. Keep the number
of draws balanced per training theta and split train/validation/test by theta so noise siblings
cannot leak across splits.

The default budget is 480 unique nside-1024 signal pastes, with a hard stop at 600 unless the
user explicitly approves a new budget:

| use | unique pastes | purpose |
|---|---:|---|
| pilot and immutable holdout | 60 | replay, response, scaling, and never-trained coverage tests |
| round 0 | 180 | tail-safe theory-SBI mixture |
| round 1 | 140 | posterior mass times ensemble disagreement, retaining prior/tail fraction |
| round 2 | 100 | only if the pre-registered round-1 stability gate fails |
| total default | 480 | remains a few hundred; 3,840 training noise vectors at 8 per paste |

Round 1 and round 2 acquisition may use only mock simulations, estimator ensembles, proposal
densities, and held-out mock diagnostics. It must not query HMC likelihood values or exact
theory residuals to choose points. Each new round keeps at least 25% of its locations in the
original-prior or tail components, preventing posterior collapse and preserving coverage of
the tails requested by the user.

Stop after round 2 or 3 only when independently held-out conditional coverage, posterior
predictive checks, ensemble-to-ensemble stability, and marginal/contour stability pass. If the
hard paste cap is reached without those gates, report the mock posterior as unconverged; do not
lower the coverage criterion.

### Product and replay contract

Every inference or simulation product must be content-addressed and atomic. The minimum
per-paste record is:

- ordered physical `theta`, prior bounds, truth, fixed parameters, and parameter hash;
- `log_p0`, `log_q`, proposal component, round, role, and acquisition score;
- halo/catalog, cosmology, n(z), nbar, Wkappa, noise, mask, workspace, transfer, configuration,
  code, and source-worktree identities;
- signal seed/common phase, galaxy catalog hash, noise seeds, and PRNG implementation;
- complete y/tau/kappa signal maps, noisy maps or replayable noise-alm hashes, measured dense
  and bandpower spectra, 42-vector, run timing, peak memory, and success/failure state.

Use a manifest table with one row per unique theta and a child table with one row per noise
realization. Cache identity must include parameter values and all physics inputs so two rounds
cannot silently reuse a stale map. A rerun at identical theta and seeds must be bitwise equal;
changing only a noise seed must leave the signal maps bitwise unchanged.

### Phase F — final posterior comparison

Only after all three methods pass their own gates, make one GetDist triangle plot with the
same parameter order, LaTeX labels, prior ranges, truth lines, and clearly distinguished
weights. Save the underlying GetDist chains and a comparison JSON/HDF5 manifest.

Report, for each method, posterior means, medians, 68% and 95% intervals, MAP or minimum-chi2
sample where meaningful, covariance/correlation, effective sample size, and method-specific
validation metrics. Report pairwise posterior shifts normalized by a declared joint covariance
and a distributional comparison such as classifier two-sample performance or Jensen-Shannon
diagnostics. Do not claim equivalence from overlapping 68% contours alone.

The headline plot compares:

1. direct theory HMC;
2. theory SBI trained on the same theory likelihood simulations;
3. pasted-mock SBI for the theory-anchored, fixed-phase conditional response.

Also save posterior predictive bandpower plots for all three probes and the absolute chi2/rank
of the common observation under the direct theory posterior. The already-saved first
paste-versus-theory comparison remains immutable provenance and must not be overwritten.

### Required execution order and ownership

1. **xDESI lead / measurement:** freeze and validate the manifest, observation, exact 42-vector,
   windows, covariance, transfer, noise, and ordering.
2. **JAX numerics / halo-model physics:** implement and validate the five-parameter JAX theory
   response, gradients, units, support, and grid stability.
3. **Inference statistician:** run and certify theory HMC.
4. **Inference statistician:** train and certify theory SBI without consulting HMC for tuning.
5. **Paste validator:** build the parameter-indexed nside-1024 signal simulator, run a small
   pilot, and prove deterministic replay/noise separation before any large array.
6. **Inference statistician:** execute mock rounds and held-out coverage under the paste cap.
7. **Physics referee:** independently attempt to refute each posterior and the final comparison.
8. **xDESI lead / KB curator:** run the gate, save the triangle and ledgers, verify documents,
   and journal the result.

No production SLURM submission is authorized by this document alone. Before each submission,
state the node/GPU count, wall time, array concurrency, expected evidence, output collision
policy, and total node-hours, then obtain user approval as required by `AGENTS.md`.

## How to verify

Current knowledge and invariant checks:

```bash
python tools/kb/kb.py status
python tools/kb/kb.py stale
python tools/kb/kb.py invariants --check \
  --id INV-JAX-X64-01 --id INV-JAX-SEED-01 \
  --id INV-WINDOW-CMP-01 --id INV-PROC-NOTOLERANCE-01
```

Current input-product checks:

```bash
sha256sum \
  data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5 \
  data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/noise_contract_tau_snrmatch_gkappa.h5 \
  data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/noisy_ensemble_tau_snrmatch_gkappa.h5
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  tests/test_sbi_three_probe_noise_contract.py \
  tests/test_sbi_three_probe_noiseless_estimator.py \
  tests/test_sbi_three_probe_noiseless_theory.py
```

Before sampler submission, the new manifest validator must additionally assert: exact hashes,
42-vector shape/order, 14 complete windows, zero inference use of the partial band, covariance
symmetry/finiteness/positive definiteness/rank, Cholesky replay, normalized realized n(z), exact
c0000 cosmology, exact five-parameter contract, and distinct observation/training/holdout seeds.

Before accepting HMC, archive the full ArviZ diagnostics and an evidence ledger containing the
exact command, environment, job IDs, output hashes, R-hat, bulk/tail ESS, divergences, E-BFMI,
tree-depth saturation, retained rank, absolute chi2, and per-probe residuals.

Before accepting either SBI posterior, archive exact training rows/proposals/weights/seeds,
loss curves, network ensemble, held-out coverage/SBC, posterior predictive checks, and round
stability. For mock SBI, additionally report unique pastes separately from noise-expanded
vectors and prove the holdout split is by theta.

## Failure modes

- **Old vector silently reused.** A smooth 36- or 51-vector contour is produced with the wrong
  bands. Hard shape/order/hash checks must fail before inference.
- **The partial final band is treated as complete.** The advertised ell support becomes
  estimator-dependent and cannot use the frozen covariance.
- **Gaussian smoothing applied twice or omitted.** High-ell gy/gtau/gkappa move coherently;
  saved transfer identities no longer reproduce the fiducial vector.
- **Host theory bridge inside NUTS.** One or more parameter gradients are zero/NaN and the
  posterior looks prior-dominated or spuriously narrow.
- **Twelve-realization sample covariance used.** The 42-vector covariance is rank deficient;
  uncertainties and whitening become meaningless.
- **Theory posterior relabelled as mock prior.** Tail mass is removed and the final mock
  posterior targets a different distribution even if the triangle plot looks excellent.
- **Noise siblings split across train and validation.** Validation appears artificially good
  because the same signal realization is present on both sides.
- **Noise draws counted as new pastes.** Simulation efficiency is overstated and signal-space
  coverage is much lower than claimed.
- **Absolute pasted model compared as if ensemble calibrated.** The known c0000 fixed-phase
  and fiducial operator offset is absorbed into physical parameters. Keep the response-anchored
  target and label it conditional.
- **Acquisition consults HMC or exact theory likelihood.** Mock-SBI agreement becomes circular.
- **Stopping gates changed after seeing contours.** This violates the no-tolerance rule; an
  unconverged result must remain labelled unconverged.
- **Worktree or dependency changes during an array.** Split maps can mix physics code. The
  manifest must reject combination and affected rows must be rerun.
- **Only the triangle plot is saved.** Without chains, proposals, diagnostics, maps, and hashes,
  the result cannot be reproduced or independently refuted.

## Open questions

- **Blocking:** can the exact five-parameter projected-painter theory be made fully JAX-native
  and gradient-finite at acceptable HMC cost, or is a separately validated differentiable
  emulator required? Owner: `jax-numerics` with `halo-model-physicist`.
- **Blocking:** which predeclared noisy realization becomes the single common observed vector?
  Freeze it before any posterior inspection. Owner: `xdesi-lead`.
- **Blocking:** does the response-anchored mock simulator remain stable over the full original
  prior and deliberately oversampled tails? The 60-point pilot decides; failures are model
  domain evidence, not grounds to narrow the prior. Owner: `abacus-paste-validator`.
- **Non-blocking until resource planning:** the exact wall time and safe GPU concurrency for
  480 nside-1024 unique pastes must be measured from the pilot before requesting the production
  array. Owner: `abacus-paste-validator`.
- **Blocking for final publication language:** one c0000/ph000 phase only validates a
  conditional response. Ensemble-calibrated simulation constraints require additional phases
  or a separate phase-variance model and are outside the present few-hundred-paste budget.

