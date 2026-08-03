---
id: kb.xdesi.analysis-state
title: xDESI analysis state — what is measured, what is fitted, what it means
layer: 60-projects
owner: xdesi-lead
status: draft
confidence: medium
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
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.measurement.multiprobe-product, kb.xdesi.ksz-conventions, kb.xdesi.abacus-paste]
scope_digest: sha256:ac3e08f3792ca916ccdefae021108e9f
---

## Claim

The Stage-31 fit varies 31 fixed-cosmology astrophysical and HOD parameters against the
460-element `fast1024` data vector. The v1 best fit reaches whitened chi2 = 7346.23 against
an expectation near 428 — **it is not a good fit**, and is usable only as an operational
starting point for Abacus map-pasting work.

## Why it is true

From `BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` (dated 2026-06-04):

**Likelihood.** `chi2 = || W (data − theory) ||^2`, with `W` built from the covariance
correlation-matrix eigendecomposition at eigenvalue threshold 1e-8. For `fast1024` this
retains rank 459 of 460 (`INV-WHITEN-RANK-01`).

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

All `outputs/` paths are gitignored and exist only on the cluster.

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
  points? Candidate causes, in the order `xdesi-lead` should eliminate them: shot-noise
  subtraction (`INV-SHOTNOISE-01`), the lens kernel, scale cuts / 1h–2h transition at high
  ell, and only then HOD flexibility. Owner: `xdesi-lead`. **Blocking** any physical
  interpretation of the fit.
- Whether v2 (`max_tree_depth: 4`, 8000×4) converged is unresolved; the saturation fraction
  has not been recorded here. Owner: `inference-statistician`. Blocking any quoted posterior.
- This document is derived from the handoff summary dated 2026-06-04 and has not been
  re-checked against the code at `43e07ca`. Every number above should be re-derived before
  being quoted. Owner: `xdesi-lead`.
