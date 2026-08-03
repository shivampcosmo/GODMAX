---
id: kb.measurement.multiprobe-product
title: The xDESI multi-probe NaMaster product — schema, conventions, noise policy
layer: 50-data-products
owner: measurement-namaster
status: draft
confidence: medium
scope:
  - notebooks/xDESI/survey_measure/multiprobe_namaster.py
  - notebooks/xDESI/survey_measure/prepare_multiprobe_maps.py
  - notebooks/xDESI/survey_measure/measure_multiprobe_namaster.py
  - notebooks/xDESI/survey_measure/README.md
  - tests/test_xdesi_multiprobe_namaster.py
invariants:
  - INV-NMT-COUPLED-01
  - INV-NMT-BANDMAJOR-01
  - INV-SHEAR-SIGN-01
  - INV-KSZ-CATALOG-01
  - INV-SHOTNOISE-01
  - INV-BEAM-01
  - INV-PRODUCT-PROV-01
  - INV-DV-SHAPE-01
checks:
  - pytest tests/test_xdesi_multiprobe_namaster.py -q
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.xdesi.ksz-conventions, kb.xdesi.analysis-state]
scope_digest: sha256:b76e06359a4faa4146750c1fe780e754
---

## Claim

The multi-probe product is a 46-spectrum, 460-element decoupled-bandpower data vector with a
matching `(460, 460)` Gaussian covariance, saved as HDF5 together with the masks, weights,
n(z), priors and per-field noise policy needed to reproduce it. Theory enters the comparison
only through `theory_to_data_vector`, which applies the saved bandpower windows.

## Why it is true

`notebooks/xDESI/survey_measure/README.md` is the authoritative convention record; the
statements below are quoted from it and each has a corresponding invariant.

**Spectrum inventory** (46 spectra / 460 elements): 10 DES shear EE, 4 y × shear-E, 4 DESI
galaxy autos, 4 galaxy × y, 16 galaxy × shear-E, 4 galaxy × ACT kappa, 4 DESI momentum × ACT
temperature kSZ. Covariance rank after the 1e-8 eigenvalue cut is 459.
Covered by `test_default_spectrum_inventory_is_46` and
`test_component_labels_match_namaster_ordering`.

**Stages and binning** — three distinct, non-interchangeable schemes:

- `fast1024`: nside 1024, lmax 1024, 10 **linear** bins, edges
  `[8, 110, 212, 314, 415, 517, 619, 720, 822, 924, 1025]`.
- `midres2048`: nside 2048, ell 128–3000, 13 **hybrid-log** bins, left edges
  `[128, 160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280]`, right-exclusive
  edges `[160, …, 3001]`, 1 deg C2 mask apodisation, pair-overlap mean subtraction.
- DES Y3 fiducial low-res diagnostic: nside 1024, 32 equal-weight bandpowers with edges
  uniformly spaced in **sqrt(ell)** over ell 8–2048 — not logarithmic, not linear-width.

Covered by `test_linear_bandpowers_match_cpu_production_edges`,
`test_des_y3_fiducial_bandpowers_match_transferred_edge_rule`,
`test_sqrt_bandpowers_cover_requested_ell_range`.

**Covariance construction.** Blocks computed in decoupled bandpower space with
`nmt.gaussian_covariance(..., coupled=False)` (`INV-NMT-COUPLED-01`). Flattened arrays are
**band-major**, so blocks are extracted with
`cov.reshape(n_band, n_comp_a, n_band, n_comp_b)[:, comp_a, :, comp_b]`
(`INV-NMT-BANDMAJOR-01`). Map-field inputs are full-ell total spectra built from decoupled
measured bandpowers with auto-noise added back, log-smoothed and clipped positive for auto
components, then expanded as constant-in-band full-ell spectra. kSZ catalog-momentum inputs
follow the NaMaster kSZ tutorial convention: coupled pseudo-`C_ell` divided by the
mask-overlap `fsky`, with the catalog zero-lag `Nf` term added back for momentum autos.

**Noise policy** (`INV-SHOTNOISE-01`). DESI galaxy autos subtract weighted Poisson shot
noise `N_ell = area_sr * sum(w^2) / sum(w)^2`, saved in each `g{i}` field metadata. DES shear
same-bin shape noise and DESI same-bin weighted shot noise are the explicit covariance noise
templates. **No ACT or kSZ noise is subtracted from the saved data vector.** Each
`input_cls_for_covariance/*` dataset records its spin labels and noise policy.

**Field conventions.** DES spin-2 fields from `gamma1` and `gamma2_namaster`, multiplied by
`shear_e_to_kappa_sign = -1` (`INV-SHEAR-SIGN-01`) — leaves EE unchanged, aligns scalar ×
shear-E with positive-convergence theory. Masks are normalised by default and HEALPix pixel
windows are **retained** in the measured spectra, so theory must be filtered externally
before the saved NaMaster windows are applied.

**Inputs recorded with the product** (`INV-PRODUCT-PROV-01`):

- DESI galaxies and kSZ: `data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5`,
  selection `catalog/valid_for_cl`, weight `catalog/weight_imaging_mean1`.
- DESI masks: DR9 quality-cut random-count HEALPix maps in
  `data/desi_dr9_imaging_randoms/desi_dr9_randoms_1_0_lrg_quality_count_maps_nside1024_4096.h5`,
  read as `nside1024/random_count` or `nside4096/random_count`. For `midres2048`, if no
  native `nside2048/random_count` exists, a **sum-preserving nside4096 → nside2048 downgrade**
  is used and recorded in metadata.
- DESI theory kernel: calibrated true-z n(z) at `nz/desi/nz_dndz_by_pz`
  (`INV-NZ-TRUEZ-01`); the photo-z histogram lives separately under
  `nz/desi_photoz_diagnostic` and must not be used as theory.
- DES Y3 source n(z): FITS HDU `nz_source` from
  `2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits`; raw bin values and
  normalised theory `dN/dz` both saved under `nz/des_shear`.
- DES Y3 Gaussian priors under `priors/des_y3_gaussian` (`INV-PRIOR-DESY3-01`).
- ACT y and T theory get a 1.6 arcmin Gaussian beam before the saved windows
  (`INV-BEAM-01`).

**The theory path** (`INV-WINDOW-CMP-01`), from
`BACKLIGHT_PASTE_HANDOFF_SUMMARY.md`:

```python
theory_to_data_vector(
    measurement_h5, theory_cls, ell=ell_theory,
    shear_m_bias=saved_m_means, ksz_velocity_correlation=0.3,
    include_default_pixel_windows=True, include_default_act_beams=True,
    theory_shear_e_is_positive_kappa=True,
)
```

Smooth theory at `ell_eff` is a diagnostic only.

## How to verify

```bash
pytest tests/test_xdesi_multiprobe_namaster.py -q
python tools/kb/kb.py invariants --check --layer measurement
```

The suite builds its own synthetic HDF5 inputs, so it runs without cluster data. Expected:
all tests pass; 46-spectrum inventory and band-major extraction assertions in particular.

## Failure modes

- **Component-major covariance extraction.** Nothing raises; the matrix stays symmetric and
  positive-definite; covariance is attributed to the wrong probe pair; chi2 is plausible and
  wrong. The quietest failure in the pipeline.
- **`coupled=True`.** Covariance leading dimension becomes `n_ell` instead of `n_band`;
  whitening rank is wrong.
- **Missing shear sign.** Pristine EE spectra alongside four inverted cross families
  (y × shear-E, g × shear-E, kappa × shear-E) whose chi2 improves if theory is hand-flipped.
- **Beam applied twice.** Monotonic high-ell deficit confined to ACT y and T families,
  growing with ell, with low-ell bands unaffected.
- **Old kSZ cache.** `KeyError` on `catalog/` arrays, or a momentum auto with no white-noise
  floor and an implausibly small covariance diagonal. Caches predating the NaMaster 2.7
  update must be regenerated.
- **Mixing binning schemes across stages.** Positional indexing means residuals get
  attributed to the wrong band; symptom is a discontinuity at a family boundary.
- **Unrecorded mask realization.** Two stages disagree at low ell with no code difference.

## Open questions

- This document is derived from `survey_measure/README.md` and the handoff summary, plus the
  test-function inventory — **not** yet from line-level reading of the 157 KB
  `multiprobe_namaster.py`. `confidence: medium` until the anchors are line-level. Owner:
  `measurement-namaster`. Not blocking, but it means every claim here should be treated as a
  hypothesis to check against code at validation-loop S1.
- The `midres2048` DESI high-ell mask still uses a single DR9 random realization and is
  recorded as provisional in output metadata. More realizations would give a less sparse raw
  nside 4096 mask. Owner: `measurement-namaster`. Blocks final production MCMC.
- `lmax = 2048` covers only the low end of the ~1000–7000 range the harmonic kSZ reference
  analysis fits. Low-resolution products cannot validate kSZ amplitude or shape.
