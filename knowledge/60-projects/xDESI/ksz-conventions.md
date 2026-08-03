---
id: kb.xdesi.ksz-conventions
title: kSZ estimator, sign convention and velocity calibration
layer: 60-projects
owner: measurement-namaster
status: draft
confidence: medium
scope:
  - notebooks/xDESI/survey_measure/measure_ksz_high_ell.py
  - notebooks/xDESI/survey_measure/diagnose_ksz_harmonic.py
  - notebooks/xDESI/survey_measure/ksz_lowres_diagnostics.ipynb
invariants:
  - INV-KSZ-SIGN-01
  - INV-KSZ-CALIB-01
  - INV-KSZ-CATALOG-01
checks:
  - pytest tests/test_xdesi_multiprobe_namaster.py -q -k ksz
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.measurement.multiprobe-product, kb.xdesi.analysis-state]
scope_digest: sha256:e02766f76e6fcab67c2aed8312f62fa0
---

## Claim

The kSZ data vector is the **raw** `C_ell^{pi,T_uK}` measured from catalog momentum. Theory
supplies `C_ell^{g,tau}` only and is converted with a **negative** factor:
`C_ell^{pi,T_uK} = -T_CMB_uK * A_v_bin * C_ell^{g,tau}`. Positive-convention plots therefore
show `-D_ell^{pi,T}`. Three quantities must be kept distinct: the raw estimator, the theory
mapping, and the plotting convention.

## Why it is true

From `notebooks/xDESI/survey_measure/README.md` and
`BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` ("kSZ Convention"):

**Sign** (`INV-KSZ-SIGN-01`). `pi` is built from the supplied **positive** `vr_over_c`
catalog column. With the paper convention, positive gas corresponds to
`C_ell^{pi,T} = -r sigma_true sigma_rec C_ell^{tau,g}`. So internally a positive
`C_ell^{g,tau}` yields a negative raw `C_ell^{pi,T}`. Paper-style positive plots use
`D_ell^kSZ = -ell(ell+1) C_ell^{pi,T} / (2 pi)`. The high-ell diagnostic HDF5 stores both
`dl_raw_piT` and `dl_paper_ksz` explicitly — that separation is the safeguard, and it should
be preserved in any new product.

**Estimator** (`INV-KSZ-CATALOG-01`). Spectra are measured with
`pymaster.NmtFieldCatalogMomentum` from the saved `catalog/{ra_deg,dec_deg,weight,field}`
arrays in each `pi{i}` field. The pixelised `pi` maps in the map product are **diagnostics
only**. Momentum autos add back the catalog zero-lag `Nf` term in the covariance input, and
kSZ covariance inputs use coupled pseudo-`C_ell` divided by the mask-overlap `fsky`. Cached
map products created before the NaMaster 2.7 update lack the catalog arrays and must be
regenerated.

**Velocity calibration** (`INV-KSZ-CALIB-01`). `A_v_bin` combines:

- the photometric DESI velocity-reconstruction correlation `r = 0.3`, from
  `papers/ksz/2407.07152v2.pdf`;
- the saved per-bin imaging-weighted reconstructed velocity RMS
  (`rms_rec_vr_over_c_weighted`);
- Abacus `sigma_true_gas/c = [0.00105580879, 0.00104915865, 0.00103582548, 0.00101760550]`,
  from `data/xDESI/survey_data/docs/DESI_ABACUS_SIGMA_TRUE_GAS.md`.

Overrides are explicit: `ksz_sigma_true_over_c`, or free `A_v_bin` amplitudes. The kSZ
amplitude is **linear** in both `r` and `sigma_true`, so an undocumented change to either
rescales the inferred gas amplitude without changing fit quality.

Same weights throughout: DESI `delta_g`, DESI × shear/y/kappa, and the kSZ velocity-momentum
template all use `catalog/weight_imaging_mean1`.

**Nulls and significance.** `ksz_lowres_diagnostics.ipynb` recomputes the four DESI
velocity-momentum × ACT temperature spectra plus **shuffled-velocity nulls**, and reports
full-covariance chi-square and PTE. It deliberately does **not** call `sqrt(d^T C^-1 d)` a
detection S/N — a kSZ amplitude S/N requires a theory or template vector. Do not restate it
as a significance.

**Resolution limits.** The harmonic-space kSZ reference analysis fits roughly
`1000 < ell < 7000`. The low-resolution product reaches `ell_max = 2048`, covering only the
low end. It is adequate for smoke tests and large-scale crosses; it cannot validate kSZ
amplitude or shape. `measure_ksz_high_ell.py` exists for that.

## How to verify

```bash
pytest tests/test_xdesi_multiprobe_namaster.py -q -k ksz
python tools/kb/kb.py invariants --check --id INV-KSZ-SIGN-01 --id INV-KSZ-CALIB-01 --id INV-KSZ-CATALOG-01
```

Relevant existing tests: `test_ksz_velocity_amplitudes_use_saved_sigma_rec_and_paper_r`,
`test_default_ksz_velocity_amplitudes_use_abacus_sigma_true`,
`test_ksz_velocity_amplitudes_prefer_weighted_sigma_rec`,
`test_ksz_catalog_momentum_roundtrips_through_map_product`,
`test_catalog_momentum_covariance_input_adds_back_zero_lag_noise`,
`test_ksz_covariance_block_forces_all_inputs_to_pseudo_over_fsky`.

## Failure modes

- **Sign confusion between raw and plotted convention.** chi2 is minimised at negative gas
  amplitude, or the fitted optical depth comes out negative while the plot looks correct.
- **Using the pixelised `pi` map as the estimator.** Introduces pixel-window and shot-noise
  structure that the catalog-momentum estimator avoids; the covariance then does not match.
- **Old cache.** `KeyError` on `catalog/` arrays, or a momentum auto missing its
  white-noise floor with an implausibly small covariance diagonal.
- **Undocumented `r` or `sigma_true` change.** Gas amplitude shifts by a constant factor
  across all four kSZ bins while every other probe family is unchanged. This is the
  signature to look for whenever a gas result moves without a code change.
- **Quoting `sqrt(d^T C^-1 d)` as a detection.** Overstates the result; there is no template
  in that quantity.
- **Claiming kSZ validation from `lmax = 2048`.** Only the low end of the fitted range is
  covered.

## Open questions

- Derived from prose and the test-function inventory, not from line-level reading of
  `measure_ksz_high_ell.py` (38 KB) or the estimator in `multiprobe_namaster.py`.
  `confidence: medium`. Owner: `measurement-namaster`.
- The kSZ block chi2 improved only 43.18 → 21.72 between fiducial and v1 best fit, for 4
  spectra. Whether that reflects a genuinely weak constraint at `lmax = 2048` or a
  calibration problem is unresolved. Owner: `measurement-namaster` with `xdesi-lead`. Not
  blocking the galaxy-sector investigation, which dominates the misfit.
