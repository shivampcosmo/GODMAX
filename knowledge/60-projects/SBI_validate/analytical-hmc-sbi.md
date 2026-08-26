---
id: kb.sbi.analytical-hmc-sbi
title: Analytical HMC and SBI comparison contract
layer: 60-projects
owner: inference-statistician
status: verified
confidence: medium
scope:
  - notebooks/SBI_validate/fiducial_theory_datavector.py
  - notebooks/SBI_validate/gaussian_covariance.py
  - notebooks/SBI_validate/survey_defaults.py
  - notebooks/SBI_validate/theory_sbi_utils.py
  - notebooks/SBI_validate/run_hmc_theory_cls.py
  - notebooks/SBI_validate/run_sbi_theory_cls.py
  - notebooks/SBI_validate/compare_hmc_sbi_full_theory_pairs.py
  - notebooks/SBI_validate/run_hmc_gas_parameter_scan.py
  - notebooks/SBI_validate/analyze_hmc_gas_parameter_scan.py
  - notebooks/SBI_validate/plot_failed_hmc_gas_pair_diagnostics.py
  - notebooks/SBI_validate/run_hmc_five_parameter_probe_scan.py
  - notebooks/SBI_validate/plot_hmc_five_parameter_probe_scan_getdist.py
  - notebooks/SBI_validate/submit_hmc_five_parameter_probe_scan.sbatch
  - notebooks/SBI_validate/run_hmc_five_parameter_probe_checkpointed.py
  - notebooks/SBI_validate/monitor_hmc_five_parameter_checkpoints.py
  - notebooks/SBI_validate/submit_hmc_five_parameter_probe_checkpointed.sbatch
  - notebooks/SBI_validate/submit_hmc_five_parameter_checkpoint_monitor.sbatch
  - notebooks/SBI_validate/submit_hmc_five_parameter_checkpointed.sh
  - notebooks/SBI_validate/run_sbi_five_parameter_probe_sequential.py
  - notebooks/SBI_validate/monitor_hmc_sbi_five_parameter_sequential.py
  - notebooks/SBI_validate/plot_hmc_sbi_five_parameter_final_only.py
  - notebooks/SBI_validate/plot_hmc_depth5_vs_depth6_getdist.py
  - notebooks/SBI_validate/run_hmc_five_parameter_depth6_continuation.py
  - notebooks/SBI_validate/submit_hmc_five_parameter_depth6_continuation.sbatch
  - notebooks/SBI_validate/submit_hmc_five_parameter_depth6_continuation.sh
  - notebooks/SBI_validate/run_sbi_five_parameter_gp_efficiency.py
  - notebooks/SBI_validate/validate_sbi_emulator_architectures.py
  - notebooks/SBI_validate/submit_sbi_five_parameter_gp_efficiency.sbatch
  - notebooks/SBI_validate/submit_sbi_five_parameter_gp_efficiency.sh
  - notebooks/SBI_validate/submit_sbi_five_parameter_probe_sequential.sbatch
  - notebooks/SBI_validate/submit_hmc_sbi_five_parameter_monitor.sbatch
  - notebooks/SBI_validate/submit_sbi_five_parameter_sequential.sh
  - notebooks/SBI_validate/submit_hmc_gas_parameter_scan.sbatch
  - notebooks/SBI_validate/validate_full_theory_gas_profile_nulls.py
  - notebooks/SBI_validate/validate_full_theory_covariance.py
  - notebooks/SBI_validate/validate_full_theory_covariance_resolution.py
  - notebooks/SBI_validate/noise_curves/simons_observatory/PROVENANCE.md
  - notebooks/SBI_validate/noise_curves/simons_observatory/LAT_lensing_noise_README.md
  - notebooks/SBI_validate/noise_curves/simons_observatory/LAT_comp_sep_noise_README.txt
  - notebooks/SBI_validate/noise_curves/simons_observatory/LICENSE
  - notebooks/SBI_validate/noise_curves/simons_observatory/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt
  - notebooks/SBI_validate/noise_curves/simons_observatory/nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat
  - tests/test_sbi_so_noise_covariance.py
  - notebooks/SBI_validate/03_compare_hmc_sbi_theory_cls_gy_thetaej.ipynb
invariants:
  - INV-JAX-X64-01
  - INV-JAX-GRAD-FINITE-01
  - INV-JAX-SEED-01
  - INV-MCMC-CONVERGENCE-01
  - INV-MCMC-TREEDEPTH-01
  - INV-PROC-EVIDENCE-01
checks:
  - python -m py_compile notebooks/SBI_validate/survey_defaults.py notebooks/SBI_validate/fiducial_theory_datavector.py notebooks/SBI_validate/gaussian_covariance.py notebooks/SBI_validate/theory_sbi_utils.py notebooks/SBI_validate/run_hmc_theory_cls.py notebooks/SBI_validate/run_sbi_theory_cls.py notebooks/SBI_validate/compare_hmc_sbi_full_theory_pairs.py notebooks/SBI_validate/validate_full_theory_gas_profile_nulls.py notebooks/SBI_validate/validate_full_theory_covariance.py notebooks/SBI_validate/validate_full_theory_covariance_resolution.py notebooks/SBI_validate/run_sbi_five_parameter_probe_sequential.py notebooks/SBI_validate/monitor_hmc_sbi_five_parameter_sequential.py notebooks/SBI_validate/plot_hmc_sbi_five_parameter_final_only.py notebooks/SBI_validate/run_hmc_five_parameter_depth6_continuation.py notebooks/SBI_validate/run_sbi_five_parameter_gp_efficiency.py notebooks/SBI_validate/validate_sbi_emulator_architectures.py
  - python -m pytest -q tests/test_sbi_so_noise_covariance.py
  - bash -n notebooks/SBI_validate/submit_hmc_five_parameter_depth6_continuation.sbatch notebooks/SBI_validate/submit_hmc_five_parameter_depth6_continuation.sh notebooks/SBI_validate/submit_sbi_five_parameter_gp_efficiency.sbatch notebooks/SBI_validate/submit_sbi_five_parameter_gp_efficiency.sh
verified_at_commit: 29c3a27
verified_on: 2026-08-17
see_also: [kb.inference.likelihood-and-convergence, kb.numerics.jax-contract]
supersedes: []
scope_digest: sha256:645662e424975a279740838c48ef3391
---

## Claim

The analytical HMC and SBI comparison runners can obtain their predictions through the
same selected GODMAX theory-vector callable and persist enough immutable sample metadata
to prove which backend, data, priors, fixed values, seeds, and sampler budgets were used.
The dedicated gas-profile comparison consists of two conditional fits:

- `theta_ej_0 + delta_rhogas`, fixing `gamma_rhogas=2` and `nu_theta_ej_M=-0.1`;
- `theta_ej_0 + gamma_rhogas`, fixing `delta_rhogas=7` and `nu_theta_ej_M=-0.1`.

The earlier comparison used 80 native finite-grid entries, but those sampler products are
stale because their covariance failed the y-beam and CMB-kappa noise contracts. The current
SO/DESI full-theory product has 68 `gg/gy/gtau/gkappa` entries: 17 native bins whose complete
integer-ell support lies within both official SO curves. HMC and SBI have not been rerun on
this product.

The intended dedicated full-theory covariance uses those same four target spectra plus six
full-theory field-pair auxiliaries. The covariance-only
`yy/ytau/tautau/ykappa/taukappa/kappakappa` spectra now use one robust full-`P(k,z)`
projector, and covariance factorization/inversion scale to correlation space so no common
absolute floor overwrites blocks with different units. The corrected product uses
beam-deconvolved y theory with official SO tSZ Deproj-2 noise and CMB-convergence theory
with official iterative baseline SO MV convergence noise. Noise is averaged directly over
supported integer multipoles, with no interpolation, extrapolation, or partial bands.
DESI-like galaxies use `1/nbar` Poisson shot noise. Tau noise remains a provisional
effective white-noise assumption, not an SO forecast.

A factor-two ell refinement over the same complete integer support confirms the
DESI `gg` and official-SO `gy+gkappa` Gaussian diagonals and SNR at the recorded
resolution level. It does not confirm `gtau`: one tau diagonal error changes by
17.35%, and no physical tau-reconstruction noise curve has been selected. The
four-probe total is therefore an explicitly provisional algebraic diagnostic,
not a validated SO four-probe forecast.

The separate three-probe conditional HMC scan pairs `theta_ej_0` with each of
the other nine Stage-31 baryon/gas scalars while using only `gy`, `gtau`, and
`gkappa`. Seven pairs pass the pre-registered convergence gate. Within that
converged subset, `alpha_nt` is uniquely lowest-correlated: the median
within-chain block-bootstrap absolute Pearson correlation is 0.0670 with a 95%
interval `[0.0313,0.1043]`, compared with 0.6981 `[0.6779,0.7159]` for the
next pair, `mu_beta`; absolute Spearman gives the same ordering. The
prior-normalized ellipse axis-ratio cross-check instead ranks `nu_theta_ej_M`
first, narrowly followed by `mu_beta` and `alpha_nt`, so the conservative
all-metric Pareto set contains those three pairs and no unique all-metric winner
is claimed. `log10_Mstar0_theta_ej` and `nu_theta_ej_z` fail R-hat, ESS, and
divergence checks, so no all-nine winner is claimed and no scientific contours
are emitted for them. This is a conditional ranking under the fixed complement,
priors, three-probe covariance, and implemented finite-grid theory; it is not a
prior-independent baryonic-identifiability result.

## Why it is true

`make_inference_theory_vector_function` selects the requested backend. Its direct branch
returns an empty auxiliary-information mapping, while `theory_mode="full"` selects the
native `get_Cl` totals in `notebooks/SBI_validate/theory_sbi_utils.py`. HMC calls that
function inside each NumPyro likelihood evaluation. NUTS differentiates the current exact
nonlinear likelihood along each trajectory; it does not replace theory with one fixed
Jacobian. SBI selects its Python per-row direct simulation loop because no Jacobian exists
in `theory_info`, then trains on the complete selected Cholesky-whitened vector.

Both runners persist the selected data vector, covariance, selection indices, parameter
specifications, fixed parameter specifications, backend, offset policy, and random seed or
sampler settings alongside their samples. `compare_hmc_sbi_full_theory_pairs.py` refuses
to plot unless these arrays equal each other and the current dedicated fiducial exactly,
no forbidden affine/compression products exist, the flattened HMC draws equal the saved
chain arrays, and the comparison-specific HMC convergence gate passes. It recomputes
unrounded per-parameter ArviZ R-hat and bulk/tail ESS from the saved chains before plotting.

The gas-scan runner additionally validates finite direct predictions, Jacobians,
and likelihood gradients at the fiducial, midpoint, and four exact prior
corners before each HMC run. Its analyzer requires identical executed-source,
product, data, covariance, Cholesky, and selection hashes across all nine jobs;
checks exact chain/flattened-array linkage; and ranks only pairs whose saved
convergence gate passes. Failed pairs are shown as diagnostic text rather than
posterior contours.

The SBI network's internal independent theta/x z-scoring is invertible preprocessing; it
preserves all selected components and is not theory linearization or summary compression. A
single SBI seed supports this requested descriptive HMC comparison, not a calibration or
simulation-based-calibration claim.

The five-parameter sequential comparison uses three independent probe-selection jobs and
three SNPE-C rounds. It trains in the exact componentwise probit image of each HMC uniform
box prior, so a prior-dominated physical coordinate maps to a standard Normal rather than
being approximated across a hard density-estimator boundary. This transform preserves the
physical prior and is neither data compression nor theory linearization. Every round saves
physical posterior samples plus the direct full-theory simulations and a SHA-bound ready
marker. The plot monitor requires bitwise equality of the HMC and SBI data vector,
covariance, Cholesky factor, selections, prior bounds, and fiducial values before it emits a
configuration-specific GetDist triangle. Interim HMC overlays are explicitly diagnostic
unless the existing HMC convergence gate has produced its final validated artifact.
The final-only plotting helper reuses the same SHA-chain and exact-array checks, then draws
only the latest HMC checkpoint and SBI round 3. It records the source and PDF hashes and
keeps a failed HMC convergence gate visible in both the figure title and plot manifest.
The completed 3,000-draw HMC products for all three probe selections fail their unchanged
convergence gates. Consequently, the final-only figures are diagnostic comparisons, not
quotable HMC posteriors; each PDF includes per-parameter R-hat and bulk/tail ESS together
with total/per-chain divergences and depth-5 saturation.

The depth-6 continuation workflow treats those final checkpoint states as immutable parent
inputs. It preserves each chain's RNG, position, gradient, potential, adapted step size,
and dense inverse mass matrix; performs no new warm-up; and collects a separate 2,000-draw
segment with `max_tree_depth=6`. The new segment receives the unchanged convergence gate.
A mechanically linked 5,000-draw mixed-depth artifact is saved only as a descriptive
history: inherited divergences and depth-5 saturation prevent it from being relabelled as
a newly converged posterior.

The dedicated depth-comparison plotter authenticates and overlays only the separate
3,000-draw depth-5 segment and 2,000-draw depth-6 segment for each of `gy`,
`gy+gkappa`, and `gy+gkappa+gtau`; it never uses the mixed-depth 5,000-draw history.
All draws from all four chains are retained without thinning or diagnostic pruning. The
three GetDist figures are diagnostic-only because both segments failed their unchanged
convergence gates. Each figure visibly reports per-parameter R-hat and bulk/tail ESS,
total/per-chain divergences, and total/per-chain depth saturation, while a manifest binds
the exact sampler, gate, likelihood-contract, and figure hashes.

The first efficient GP-SBI benchmark was correctly rejected after 548 unique direct
theta points: its global prior/fiducial design missed the narrow relevant region, fresh
posterior-anchor likelihood errors were large, and the surrogate chains diverged. Its
contours are diagnostic failures. A conditional architecture screen nevertheless shows
that the plain vector ARD squared-exponential GP is accurate when its simulations occupy
the posterior region; scalar-likelihood and quadratic-mean alternatives are worse.

The second, pre-registered lap used the user-designated current HMC as a
nonconverged placement benchmark, not as a theory target. One shared 384-point design
mixes the three HMC empirical Gaussians in probit coordinates with a covariance-inflated
pooled component. Nested 128/256/384 rounds reserve 16/32/48 exact holdouts and add eight
fresh direct/full posterior anchors per probe selection, capping cumulative unique theta
points at 152/304/456. The one- and two-probe residuals remain exact prefix/covariance
transformations of the same three-probe evaluations. Surrogate NUTS uses a Normal-CDF
parameterization of the unchanged physical Uniform priors. Existing likelihood-error,
Jacobian and stability tolerances were unchanged. It stopped before posterior sampling:
the duplicated JAX and scikit GP evaluators disagreed above their strict implementation
threshold on both GPU and CPU, although the physical discrepancy was small. That
threshold was not relaxed and no lap-2 contour was accepted.

The final workflow retains the same local design and exact-theory gates but uses scikit
as the single GP implementation and four independently seeded affine-invariant `emcee`
ensembles in the exact prior-probit coordinates. Its convergence gate uses cross-ensemble
R-hat/agreement, integrated autocorrelation time, autocorrelation ESS, chain length,
and walker acceptance fractions. Divergence and tree-depth statistics are explicitly
not applicable rather than being fabricated. The analytic derivative of the same GP
kernel is compared with the independently differentiated direct full theory. Explicit
HMC mean/width/correlation agreement remains an additional algorithmic gate. Each round
writes hash-bound color-blind-safe GetDist comparisons, and any failed exact-theory or
sampler gate leaves them diagnostic.

Here, `full` means the standard native finite-grid GODMAX branch. It does not establish
infinite radial support. The production radial grid ends at 10 Mpc/h although the nominal
profile normalization support can extend beyond it. The comparison is therefore a valid
algorithmic comparison of HMC and SBI applied to the identical implemented likelihood;
physical interpretation remains conditional on the open radial-closure/support-convergence
thread owned by `kb.physics.halo-model-ingredients`.

## How to verify

```bash
python -m py_compile \
  notebooks/SBI_validate/fiducial_theory_datavector.py \
  notebooks/SBI_validate/gaussian_covariance.py \
  notebooks/SBI_validate/theory_sbi_utils.py \
  notebooks/SBI_validate/run_hmc_theory_cls.py \
  notebooks/SBI_validate/run_sbi_theory_cls.py \
  notebooks/SBI_validate/compare_hmc_sbi_full_theory_pairs.py \
  notebooks/SBI_validate/validate_full_theory_gas_profile_nulls.py \
  notebooks/SBI_validate/validate_full_theory_covariance.py

PYTHONPATH=notebooks/SBI_validate MPLCONFIGDIR=/tmp/matplotlib python \
  notebooks/SBI_validate/validate_full_theory_covariance.py \
  --product notebooks/SBI_validate/outputs/theory_sbi/fiducial_full_thetaej2_nuejm_minus0p1_delta7_gamma2.npz \
  --output-dir notebooks/SBI_validate/outputs/theory_sbi/full_theory_covariance_validation

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=notebooks/SBI_validate:src python \
  notebooks/SBI_validate/validate_full_theory_covariance_resolution.py \
  --product notebooks/SBI_validate/outputs/theory_sbi/fiducial_full_thetaej2_nuejm_minus0p1_delta7_gamma2.npz \
  --output notebooks/SBI_validate/outputs/theory_sbi/full_theory_covariance_validation/ell_resolution_validation.json \
  --refinement-factor 2

JAX_PLATFORMS=cpu python \
  notebooks/SBI_validate/validate_full_theory_gas_profile_nulls.py \
  --output /tmp/profile_null_audit.json

python notebooks/SBI_validate/compare_hmc_sbi_full_theory_pairs.py \
  --stage validate --profile production --pairs theta_delta,theta_gamma

MPLCONFIGDIR=/tmp/mpl-hmc-gas-scan \
PYTHONPATH=notebooks/SBI_validate:src \
python notebooks/SBI_validate/analyze_hmc_gas_parameter_scan.py \
  --scan-root \
  notebooks/SBI_validate/outputs/theory_sbi/hmc_gas_parameter_scan_1000x4

python -m py_compile \
  notebooks/SBI_validate/run_sbi_five_parameter_probe_sequential.py \
  notebooks/SBI_validate/monitor_hmc_sbi_five_parameter_sequential.py \
  notebooks/SBI_validate/plot_hmc_sbi_five_parameter_final_only.py \
  notebooks/SBI_validate/run_hmc_five_parameter_depth6_continuation.py

bash -n \
  notebooks/SBI_validate/submit_sbi_five_parameter_probe_sequential.sbatch \
  notebooks/SBI_validate/submit_hmc_sbi_five_parameter_monitor.sbatch \
  notebooks/SBI_validate/submit_sbi_five_parameter_sequential.sh \
  notebooks/SBI_validate/submit_hmc_five_parameter_depth6_continuation.sbatch \
  notebooks/SBI_validate/submit_hmc_five_parameter_depth6_continuation.sh \
  notebooks/SBI_validate/submit_sbi_five_parameter_gp_efficiency.sbatch \
  notebooks/SBI_validate/submit_sbi_five_parameter_gp_efficiency.sh
```

Expected result: exit status 0. Full verification additionally requires re-executing the
production HMC/SBI stages, their sampler/convergence contracts, the two 2x2 plot products,
and the independent S6 refutation recorded in the linked evidence ledger. Stored notebook
output is not evidence.

## Failure modes

- A direct run that saves a Jacobian or evaluates an affine batch response has silently
  reverted to linearization while its plots can still look plausible.
- Different saved data vectors, covariance matrices, selection indices, priors, or fixed
  parameters make the HMC and SBI contours incomparable.
- Native cached `kappa-y` or `kappa-kappa` interpolation spikes make the Gaussian
  covariance indefinite. Adding one absolute jitter to every diagonal can hide the
  failure while reducing the full-vector SNR to order unity. The covariance validator
  rejects missing auxiliary provenance and any nonzero correlation-space jitter.
- Beam-convolved y signal cannot be combined with beam-deconvolved y noise. A CMB-lensing
  kappa target also cannot use the LSST `sigma_e^2/n_eff` shape-noise prescription. Either
  mismatch invalidates plotted error bars and covariance-weighted SNR even when the matrix
  is positive definite.
- Using a noise value interpolated at an effective ell, extrapolating beyond an SO table,
  or retaining a partially covered band violates the SO curve contract.
- Missing HMC rank-normalized R-hat, bulk/tail ESS, divergence count, or tree-depth
  saturation makes a visually smooth contour unquotable.
- Treating the two panels as marginals of one three-parameter posterior is wrong; each is
  a separate conditional two-parameter fit with the other gas slope fixed.
- Agreement between methods cannot validate a common finite-grid physics error. The raw
  radial closure and support-refinement diagnostic must remain visible rather than being
  hidden by the identical-simulator comparison.
- All HMC chains currently initialize at the fiducial. Strong convergence and independent
  SBI agreement mitigate but do not eliminate weak multimodality detection.
- A visually plausible gas-scan contour is not evidence when its convergence
  gate fails. The `log10_Mstar0_theta_ej` and `nu_theta_ej_z` chains are retained
  only as failure diagnostics until an explicitly authorized follow-up run
  passes the unchanged gate.

## Open questions

- The production simulation budget needed for stable direct-theory SBI contours has not
  been established across independent SBI seeds. Owner: inference-statistician. Blocking
  general SBI-calibration claims: yes; blocking this labelled single-seed comparison: no.
- Native-full radial support and raw integrated component-mass convergence are open at the
  production grid. Owner: halo-model-physicist. Blocking physical posterior claims: yes;
  blocking the identical-simulator algorithm comparison: no.
- A physically grounded tau-reconstruction noise curve has not been selected. Owner:
  inference-statistician. Blocking the displayed `gtau` errors and any four-probe survey
  forecast claim: yes; blocking separately reported SO `gy+gkappa` and DESI `gg` Gaussian
  diagnostics: no.
- The `log10_Mstar0_theta_ej` and `nu_theta_ej_z` three-probe scans did not
  converge at 1000 warm-up plus 1000 retained draws per chain. Owner:
  inference-statistician. Blocking an all-nine least-degeneracy ranking: yes;
  blocking the explicitly labelled seven-pair converged-subset ranking: no.
