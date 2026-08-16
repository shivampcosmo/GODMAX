---
id: kb.xdesi.analysis-state
title: xDESI analysis state — what is measured, what is fitted, what it means
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: high
scope:
  - notebooks/xDESI/survey_measure/godmax_multiprobe_hmc_stage31.py
  - notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py
  - notebooks/xDESI/survey_measure/BACKLIGHT_PASTE_HANDOFF_SUMMARY.md
  - param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml
  - param_files/xDESI/priors_multiprobe_fast1024_hmc_stage31.yaml
invariants:
  - INV-CHI2-HONEST-01
  - INV-WHITEN-RANK-01
  - INV-WINDOW-CMP-01
  - INV-NZ-TRUEZ-01
  - INV-HOD-PZBIN-01
  - INV-HOD-ARRAY0-01
  - INV-MCMC-TREEDEPTH-01
checks:
  - python tools/kb/kb.py invariants --check --id INV-DV-SHAPE-01
verified_at_commit: cf72943
verified_on: 2026-08-16
see_also: [kb.measurement.multiprobe-product, kb.xdesi.ksz-conventions, kb.xdesi.abacus-paste]
scope_digest: sha256:625b23455de3cf3d71a02721f55d94c3
---

## Claim

The recorded legacy-v1 Stage-31 fit varies 31 fixed-cosmology astrophysical and HOD parameters against the
460-element `fast1024` data vector. The v1 best fit reaches whitened chi2 = 7346.23 against
an expectation near 428 — **it is not a good fit**, and is usable only as an operational
starting point for Abacus map-pasting work. A corrected pipeline-v2 fast measurement now
exists and its covariance retains rank 460/460 at the fixed 1e-8 correlation cut, but no v2
likelihood fit or goodness-of-fit result has been run. A corrected pipeline-v2
`midres2048` measurement also now exists at 16 bands through ell 4096; its covariance retains
rank 736/736 at the same fixed cut. It too has no theory fit or goodness-of-fit result. The
20-band nside-4096 measurement through ell 8192 and its Gaussian/iNKA covariance are now
complete. The 920-slot archive retains rank 920/920 and the 892-element active selection
retains rank 892/892 at the same fixed cut. This is a structurally HMC-ready saved input;
no high-resolution theory vector, S/N, chi2, sampler run, or posterior is currently available.

## Why it is true

From `BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` (dated 2026-06-04):

**Legacy-v1 likelihood.** `chi2 = || W (data − theory) ||^2`, with `W` built from the covariance
correlation-matrix eigendecomposition at eigenvalue threshold 1e-8. For `fast1024` this
retained rank 459 of 460 (`INV-WHITEN-RANK-01`). The regenerated pipeline-v2 fast covariance
retains 460 of 460 at the same threshold (`corr_eig_min = 0.0657707658`) and has no negative
covariance or correlation eigenmodes. The ranks are product-specific. No v2 theory fit has
been run, so v2 goodness of fit remains unverified.

**Goodness of fit.** With 460 elements, rank 459, and 31 varied parameters, a good fit would
sit near `459 − 31 = 428`, with scatter of order `sqrt(2 × 428) ≈ 29`.

```text
fiducial whitened chi2 = 1646728.163018249
v1 best-fit whitened chi2 =    7346.232641444647
```

Per-family v1 block chi2 (fiducial → best):

```text
des_shear_EE:          132.52  ->   124.93
act_y_des_shear_E:     159.23  ->    60.01
desi_g_auto:        1623261.22  ->  6411.27
desi_g_act_y:         14287.50  ->   177.10
desi_g_des_shear_E:    6521.06  ->   420.76
desi_g_act_kappa:      2711.17  ->   141.51
desi_pi_act_T:            43.18 ->    21.72
```

The misfit is **localised in `desi_g_auto`** (6411 of 7346), not diffuse. That points at the
galaxy sector — HOD, lens kernel, shot noise, or scale cuts — rather than at a global
calibration problem. The 224× improvement is not evidence of a good model
(`INV-CHI2-HONEST-01`).

**The 31 varied parameters** (`INV-HOD-ARRAY0-01`). Seven global baryonic scalars:
`log10_Mstar0_theta_ej`, `theta_ej_0`, `nu_theta_ej_M`, `nu_theta_ej_z`, `log10_Mc0`,
`mu_beta`, `alpha_nt`. Per-pz-bin HOD entries `[1:5]` (array entry 0 fixed) for
`log10M1_fshmr`, `log10M1_a_fshmr`, `delta_fshmr`, `gamma_fshmr`, `siglogMstar_Ncen`,
`alphasat_Nsat`. Fixed: cosmology, DES shear calibration and source-z shifts, IA defaults,
omitted HOD arrays, zero HOD evolution arrays, and analysis settings apart from explicit
comparison overrides.

**The photometric-bin HOD structure** (`INV-HOD-PZBIN-01`). Because the calibrated true-z
distributions of the four photometric bins **overlap**, assigning per-pz HOD parameters to
disjoint true-z support intervals is wrong. The implemented fix: one shared non-galaxy
WL/CMB theory block for shear- and y-only spectra, plus a separate galaxy theory block per
photometric bin using that bin's HOD parameters and its own true-n(z)/nbar(z); the
46-spectrum vector is assembled from these blocks.

**The lens kernel correction** (`INV-NZ-TRUEZ-01`). The map-making sample stays the
photometric catalog selected by `valid_for_cl`, but the theory lens kernel uses the
calibrated true-redshift n(z) from
`desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5`, group `zphot_std0p05_spec_ratio_corrected`.
The `Z_PHOT_MEDIAN` histogram must not be used. The code checks
`desi_lens_redshift_kind = spectroscopic_calibrated_true_redshift`.

**Fixed cosmology:** `H0 67.36`, `Om0 0.30`, `Ob0 0.0493`, `sigma8 0.80`, `ns 0.9649`,
`w0 -1.0`, flat.

**v2 sampler configuration** (`INV-MCMC-TREEDEPTH-01`):

```yaml
sampler:
  num_warmup: 800
  num_samples: 8000
  num_chains: 4
  chain_method: vectorized
  max_tree_depth: 4
  target_accept_prob: 0.85
```

`max_tree_depth: 4` is low for 31 correlated parameters. If saturation is high the posterior
is biased and must not be described as converged — and r_hat plus acceptance rate will not
reveal it.

**Key outputs.** Fiducial theory vector at
`notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz/theory_data_vector_fast1024.npz`;
v1 combined chains under
`outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/stage31_hmc_400x16_v1/combined/`
(`chain_stage31_multigpu.npz`, `bestfit_params_stage31_multigpu.yaml`,
`fit_summary_stage31_multigpu.json`, …). The v1 best fit is saved as a normal GODMAX params
file: `param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v1.yaml`.

The corrected pipeline-v2 measurement is at
`data/xDESI/processed/multiprobe_namaster_true_nz/fast1024/xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear_pipev2_gshot.h5`;
its galaxy autos contain clustering plus the saved weighted Poisson shot response exactly
once. The signal-only `_pipev2.h5` source is retained as an immutable historical product.
Its validation report is the adjacent
`measurement_validation_nside1024_lmax1024_nbin10_linear_pipev2_gshot.json`. These paths are
gitignored cluster artifacts. The complete command/job/hash evidence is in
`knowledge/.kb/ledgers/2026-08-05-fast1024-v2-production.md` and the convention migration is
in `knowledge/.kb/ledgers/2026-08-05-galaxy-auto-shot-noise-included.md`.

The corrected mid-resolution measurement is at
`data/xDESI/processed/multiprobe_namaster_true_nz/midres2048/xdesi_multiprobe_cls_cov_nside2048_ell128_lmax4096_lmask6143_nbin16_log_pipev2_gshot.h5`.
It contains 46 spectra × 16 bands, a `736 x 736` finite/symmetric covariance and rank
`736/736` at the fixed `1e-8` cut. Separate seven-page `C_ell` and `D_ell` PDFs plus 14 family
PNGs are under the adjacent `plots/` directory; the three `g x kappa` bands beginning at ell
3001 are explicitly transfer-null and are not physical likelihood bands. No midres theory
vector, chi2 or posterior has been produced. Complete hashes, job accounting and structural
checks are in
`knowledge/.kb/ledgers/2026-08-05-midres2048-lmax4096-nbin16-production.md`; the exact
mean-convention migration and unchanged-covariance audit are in
`knowledge/.kb/ledgers/2026-08-05-galaxy-auto-shot-noise-included.md`.

The high-resolution spectra intermediate is under
`data/xDESI/processed/multiprobe_namaster_highres4096_ell8192_dr9random8/highres4096/`.
The final HDF there contains 46 spectra x 20 common bands, with 920 archival slots and 892
active likelihood entries after excluding only the 28 ACT-kappa transfer-null slots above
ell 3000. It keeps total `C_ell^gg + SN` as the primary/HMC default and saves a deterministic
weighted-Poisson-subtracted alternate view with the same conditional covariance. The first
five-node-capped covariance run completed 257/259 groups before a scheduler node failure;
minimal recovery job `6886882` computed only groups 207 and 247, and finalizer `6886883`
then validated all 259 shards and 1,081 blocks. The final covariance is finite and symmetric,
with correlation-eigencut ranks 920/920 for the archive and 892/892 for the active selection.
Its HMC-readiness attestation binds measurement SHA256
`9462890c673f6b5b6628d638f386a57dfcc287ed2244939d98bc7f9a6394637a` and confirms the two
galaxy-auto views share the same covariance. The corrected covariance-backed `C_ell` and
`D_ell` plots use all 892 active points and explicit log limits `[128, 8192]`; they omit only
the unsupported ACT-kappa placeholders. Submission/recovery evidence is in
`knowledge/.kb/ledgers/2026-08-15-xdesi-highres8192-efficient-production.md`; final product
and regenerated-plot evidence is in
`knowledge/.kb/ledgers/2026-08-16-xdesi-highres-plot-regeneration.md`.

All `outputs/` paths are gitignored and exist only on the cluster.

Pipeline-v2 theory readers reject legacy measurement products by default and require the
measurement and map/n(z) HDF5 products to share a content-addressed `map_product_id`. The
generic NumPy theory wrapper alone supports
`theory_to_data_vector.allow_legacy_product: true` for a historical-only reproduction. The
Stage-31 HMC/likelihood path has no legacy opt-in and hard-requires `_pipev2_gshot`; its default
fast1024 comparison configuration now points at that product and its matching `_pipev2` map.
Fiducial and best-fit comparison-vector caches written under the former contract must be
regenerated before plotting: current readers require an exact measurement fingerprint,
theory-payload hash, materialized-comparison-config identity and exact saved-response content
identity. Best-fit products also require the current Stage-31 likelihood, ordered
parameter/prior and chain identities.

## How to verify

```bash
python tools/kb/kb.py invariants --check --layer inference
pytest tests/test_xdesi_multiprobe_namaster.py -q -k "inventory or theory_to_data_vector"
git log --oneline -15 -- notebooks/xDESI/survey_measure/
```

To re-derive the goodness-of-fit statement, read `fit_summary_stage31_multigpu.json` on the
cluster and report chi2 with retained rank and parameter count together.

## Failure modes

- **Quoting the improvement factor instead of the absolute chi2.** The single easiest way to
  publish a wrong result from this pipeline (`INV-CHI2-HONEST-01`, blocker).
- **Treating the v1 best-fit params file as a measurement of gas or HOD physics.** It is an
  operational point for pasting.
- **Comparing smooth theory at `ell_eff`.** Produces a smooth ell-dependent residual tilt in
  steep spectra that no parameter can absorb (`INV-WINDOW-CMP-01`).
- **Photo-z histogram as lens kernel.** All four DESI galaxy families biased in the same
  direction; HOD drifts to compensate; galaxy auto and galaxy-cross cannot be fit together.
- **Disjoint true-z pz bins.** Adjacent-bin HOD parameters become strongly and unphysically
  anti-correlated in the posterior.
- **Pooling 16 workers without a consistency check.** Pooled contours tighter than any
  individual worker's.

## Open questions

- **The dominant open physics question:** why is `desi_g_auto` chi2 = 6411 for 40 data
  points? Candidate causes, in the order `xdesi-lead` should eliminate them: the historical
  galaxy mean/theory shot-noise convention (`INV-SHOTNOISE-01`), the lens kernel, scale cuts / 1h–2h transition at high
  ell, and only then HOD flexibility. Owner: `xdesi-lead`. **Blocking** any physical
  interpretation of the fit.
- Whether v2 (`max_tree_depth: 4`, 8000×4) converged is unresolved; the saturation fraction
  has not been recorded here. Owner: `inference-statistician`. Blocking any quoted posterior.
- The legacy-v1 fit numbers remain derived from the 2026-06-04 handoff and must be re-derived
  before publication. The pipeline-v2 fast and midres measurement-structure claims have
  separate 2026-08-05 execution ledgers, but neither establishes a physical fit. Owner:
  `xdesi-lead`.
