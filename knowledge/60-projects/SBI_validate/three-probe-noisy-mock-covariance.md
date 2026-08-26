---
id: kb.sbi.three-probe-noisy-mock-covariance
title: Three-probe noisy mock and covariance contract
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/three_probe_noise_contract.py
  - notebooks/SBI_validate/submit_three_probe_noise_contract.sbatch
  - notebooks/SBI_validate/submit_three_probe_noise_realizations.sbatch
  - notebooks/SBI_validate/combine_three_probe_noise_realizations.py
  - notebooks/SBI_validate/rerun_three_probe_tau_noise.py
  - tests/test_sbi_three_probe_noise_contract.py
  - tests/test_sbi_three_probe_tau_noise.py
invariants:
  - INV-WINDOW-CMP-01
  - INV-BEAM-01
  - INV-SHOTNOISE-01
  - INV-NMT-COUPLED-01
  - INV-NMT-BANDMAJOR-01
  - INV-PRODUCT-PROV-01
  - INV-JAX-X64-01
  - INV-JAX-SEED-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_noise_contract.py
verified_at_commit: 29c3a27
verified_on: 2026-08-20
see_also:
  - kb.sbi.three-probe-noiseless-cl-validation
  - kb.sbi.three-probe-fast-paste
scope_digest: sha256:b3e77f2f5cc75438feb9e7573ab43dd1
---

## Claim

The validated product freezes one noise basis for both noisy pasted mocks and the next
resolved-theory HMC covariance. It uses 14 complete native bands through ell 2009; the
partial 2010--2048 band is excluded from inference. Twelve deterministic noisy map products
were generated without repasting, and their ensemble mean is statistically consistent with
the fixed noiseless pasted signal under the injected-noise scatter.

## Design

- Fixed signal: completed nside-1024 pasted galaxy, y, tau and CMB-kappa maps.
- y noise: official SO LAT Deproj-2 component-separation noise plus an independent Gaussian
  missing-sky term `Cyy(all mass,z) - Cyy(pasted support)` computed at the same cosmology and
  astrophysical point.
- kappa noise: official SO iterative MV CMB-lensing reconstruction noise.
- tau noise: explicitly provisional white depth, frozen in the contract rather than inherited
  from a changing survey default.
- Covariance: exact-mask NaMaster Gaussian 42x42 covariance ordered
  `gy[14],gkappa[14],gtau[14]`, with the same dense signal and noise curves used by mocks.
- Realizations: twelve deterministic independent continuous-field noise draws. The fixed
  signal is not re-pasted. Their sample covariance is rank-deficient and diagnostic only.

The all-sky y completion is an effective Gaussian-noise model over
`1e10 <= M/(Msun/h) <= 1e16` and `0.01 <= z <= 3`; it is not a realization of correlated
large-scale structure outside the pasted slice. The accepted effective tau forecast depth is
`0.023266988679843306 tau arcmin`, determined before drawing new maps by matching the
14-band analytic amplitude forecast S/N of gtau to gkappa. The earlier `1e-5` and `1e-3`
contracts are retained only as provenance and must not be used by the next HMC.

## Failure modes

- Applying the profile Gaussian again to noise or already-smoothed signal theory.
- Treating the outside-slice tSZ sky as correlated with the fixed slice galaxy map.
- Using the 12-realization sample covariance in HMC.
- Admitting the partial ell=2010--2048 band into the inference vector.
- Using band-averaged noise to synthesize a map instead of the frozen integer-ell curves.

## Products

All paths are relative to the repository root beneath
`data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/`:

- `noise_contract.h5`: dense signal/noise curves, fixed masked alms, exact windows, and the
  42x42 analytic HMC covariance and Cholesky factor.
- `namaster_workspace.fits`: exact scalar NaMaster workspace.
- `realizations/noise_realization_000.h5` through `011.h5`: three noisy nside-1024 maps and
  measured cross spectra.
- `noisy_ensemble.h5`: all draws, means, standard deviations, diagnostic sample covariance,
  theory, and HMC covariance.
- `noisy_mock_mean_vs_theory.png`: requested mean-plus-1sigma comparison.

The current HMC input is the versioned `noise_contract_tau_snrmatch_gkappa.h5`; its twelve
tau-only realizations are in `tau_snrmatch_gkappa_realizations/`, and the combined product is
`noisy_ensemble_tau_snrmatch_gkappa.h5`. The comparison PNG at the original requested path
uses this depth and fixes every panel to `60 <= ell <= 2100`. The previous plots are archived
as `noisy_mock_mean_vs_theory_tau1e-5_original.png` and
`noisy_mock_mean_vs_theory_tau1e-3.png`.
