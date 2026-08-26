---
id: kb.sbi.three-probe-noiseless-cl-validation
title: Noiseless pasted-map versus resolved-theory Cl validation
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/compare_three_probe_noiseless_cls.py
  - notebooks/SBI_validate/submit_three_probe_noiseless_cls.sbatch
  - notebooks/SBI_validate/diagnose_three_probe_empirical_hmf.py
  - notebooks/SBI_validate/submit_three_probe_empirical_hmf_diagnostic.sbatch
  - notebooks/SBI_validate/diagnose_three_probe_fullsky_cls.py
  - notebooks/SBI_validate/submit_three_probe_fullsky_diagnostic.sbatch
  - notebooks/SBI_validate/extend_three_probe_ell1536_hmf_bias.py
  - notebooks/SBI_validate/compare_three_probe_nside1024_ell2048.py
  - notebooks/SBI_validate/diagnose_three_probe_small_scale_transfer.py
  - notebooks/SBI_validate/diagnose_three_probe_large_scale_factorization.py
  - notebooks/SBI_validate/compare_three_probe_nside_resolution.py
  - notebooks/SBI_validate/submit_three_probe_ell1536_hmf_bias.sbatch
  - notebooks/SBI_validate/submit_three_probe_nside1024_ell2048.sbatch
  - notebooks/SBI_validate/three_probe_noiseless_estimator.py
  - notebooks/SBI_validate/three_probe_noiseless_theory.py
  - tests/test_sbi_three_probe_noiseless_cls.py
  - tests/test_sbi_three_probe_noiseless_estimator.py
  - tests/test_sbi_three_probe_noiseless_theory.py
  - tests/test_sbi_three_probe_ell1536_hmf_bias.py
  - tests/test_sbi_three_probe_nside1024_ell2048.py
  - tests/test_sbi_three_probe_small_scale_transfer.py
  - tests/test_sbi_three_probe_large_scale_factorization.py
  - tests/test_sbi_three_probe_nside_resolution.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-PHYS-UNITS-01
  - INV-NZ-NORM-01
  - INV-WINDOW-CMP-01
  - INV-BEAM-01
  - INV-SHOTNOISE-01
  - INV-NMT-COUPLED-01
  - INV-PRODUCT-PROV-01
  - INV-JAX-X64-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_noiseless_cls.py
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_noiseless_estimator.py tests/test_sbi_three_probe_noiseless_theory.py
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_ell1536_hmf_bias.py
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_nside1024_ell2048.py
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_small_scale_transfer.py
verified_at_commit: 29c3a27
verified_on: 2026-08-20
see_also:
  - kb.sbi.three-probe-fast-paste
  - kb.sbi.three-probe-resolved-theory
  - kb.sbi.three-way-mock-comparison-plan
supersedes: []
scope_digest: sha256:b2ac2f000ed4b527a7ea53d138e85b00
---

## Claim

The fail-closed, saved-window comparison is implemented and reproducible for the current
noiseless nside-512 paste in `gg`, `gy`, `gtau`, and `gkappa`. Inputs are matched to the
realized galaxy n(z), catalog cosmology, consumed HOD nbar, saved CMB lensing efficiency,
profile smoothing, projected aperture, galaxy pixel transfer and decoupled shot noise.
The current paste **does not pass** the pre-registered 5/10% agreement gate. Catalog
proxy-HMF weighting improves the residuals but does not close them. A separate ell=1536
diagnostic finds an effective large-scale galaxy-bias ratio of 0.914 relative to the
catalog-HMF/Tinker-bias model. Applying that ratio, inferred from `gg` only, reduces the
original-12 median absolute residuals to 5.8--7.5%; however maxima remain 17.6--29.6%, and
large-scale `gy` remains at 13.4%. Thus abundance plus an effective bias mismatch explains
most, but not all, of the discrepancy. This is not a direct halo-bias measurement. Mass
semantics and painter/pixel/profile response remain live hypotheses for the scale-dependent
remainder. A source-bound zero/one/two-smoothing diagnostic now shows that the half-pixel
Gaussian is applied exactly once in both the pasted maps and the projected-table theory.
Applying a second Gaussian or an additional continuous-field HEALPix window worsens the
high-ell residual substantially. The remaining bands are one-halo dominated and respond at
the 4--8% level to proxy-HMF replacement, while an unweighted representative pixel-centre
test has a near-unity median but broad mass/redshift-dependent tails. Smoothing is therefore
not the dominant small-scale failure; provisional mass semantics and a population-weighted
pixel/aperture response remain the leading unresolved hypotheses.

The nside-1024 half-pixel control closes most of that small-scale numerical question. With
the same catalog/HOD/cosmology/kernels and FWHM equal to half of a 1024 pixel (1.71774
arcmin), the last-band residual changes from +18.7% to +10.1% (`gg`), +6.5% to -7.2%
(`gy`), +19.5% to +3.9% (`gtau`) and +22.4% to +4.5% (`gkappa`). The 512 positive
high-ell excess was therefore predominantly a finite-resolution painter/smoothing effect.
The small-scale nside-1024 `gtau`/`gkappa` residuals are within 5%; `gg` and `gy` are mostly
within 10% with one band each just above 10%. The full 12-band gate still fails because the
low/mid-ell mock lies 10--31% below theory, especially for `gy`. HMF and the effective
large-scale bias remain plausible explanations for that separate discrepancy; neither is
validated by the resolution control.

A literal nside-1024 extension to ell=2048 uses a new 15x2049 NaMaster window: the two
complete added native bands are `[1268,1597)` and `[1597,2010)`, followed by an explicitly
partial `[2010,2049)` diagnostic band. In the complete added bands, mock/theory residuals
are respectively `gg` +10.31%/+9.47%, `gy` -6.09%/-4.62%, `gtau` +5.20%/+6.53%, and
`gkappa` +6.37%/+8.95%. In the partial final band they are +10.21%, -1.28%, +10.73%, and
+14.16%. The extension leaves every original mock and theory band stable to at most 0.241%,
below the pre-registered 0.5% null. Thus the complete high-ell cross bands are consistent at
roughly the requested 5--10% level; `gg` is at the boundary, while the partial endpoint band
is not accepted as a converged native-band result. The earlier low/mid-ell mismatch and
strict full-range gate failure remain unchanged.

A saved-product large-scale factorization diagnostic identifies the leading simple
explanation. Shot noise is only 1.4--4.3% of the first-five-band gg prediction and is not
the primary reason gg looks better. Shot-subtracted gg implies a descriptive galaxy
amplitude `A_g=0.911` (range 0.905--0.931). Dividing the cross ratios by this galaxy factor
gives nearly identical second-leg amplitudes for tau and kappa, 0.872 and 0.871, differing
by at most 0.20 percentage points, while the pressure-weighted y leg is lower at 0.790.
The independent catalog-proxy-HMF reweighting changes gg theory by only -1.5%, but gy by
-10.0% and gtau/gkappa by about -7.6%, exactly the observed ordering. Applying both that
HMF change and the gg-derived amplitude leaves median residuals of -5.8% for tau/kappa but
-12.3% for y. A same-nside cap-versus-full-sky null changes median ratios by only 1.8--3.8
percentage points, so the mask is not the primary cause. The likely explanation is therefore
an abundance/effective-bias mismatch caused by provisionally identifying the particle-count
mass proxy with Tinker M200c; gg uses only the relatively well-matched galaxy/HOD weighting,
whereas each cross adds a field-specific halo-mass weighting. Pressure weighting makes gy
most sensitive to the high-mass tail. This is a descriptive factorization, not an independent
halo-bias measurement.

## Why it is true

The immutable first HDF5/PNG/JSON bind the map, kernels, cosmology, estimator mask/windows,
shot template, current projected-table theory and source hashes. Separate diagnostics test
catalog proxy-HMF weighting, the full-sky realization, and a 13-band ell=1536 extension.
The extension leaves the original 12 recomputed bands stable to 0.251% and reports the new
partial band separately because ell=1536 is one mode beyond the usual nside-512 formal
limit. Independent referees constrain the interpretation to the supported negative and
partial-explanation claims. See the evidence ledgers.

The nside-1024 product repeats the same exact-window comparison only through ell=1535. Its
galaxy and realized-n(z) hashes are identical to nside 512, while the combined map has
707,483,094 halo-pixel pairs (10.69 per halo). The saved 512-versus-1024 residual diagnostic
makes the resolution movement explicit without overwriting either comparison.

The separate ell=2048 diagnostic product is
`data/SBI_validate/three_probe_mock/validation/noiseless_cls_gate3_nside1024/ell2048/`
`nside1024_ell2048_paste_vs_projected_theory.{h5,json,png}`. It embeds the same map SHA,
cosmology, normalized realized HOD n(z), half-pixel 1.71774 arcmin Gaussian, exact window,
galaxy pixel window and decoupled shot template. The projected y/e/m tables already include
the Gaussian, so no second harmonic Gaussian is applied.

The builder additionally returns its already-computed `chi(z)`, `dchi/dz`, and tau
conversion constant to standalone downstream covariance code. This is an API-only exposure
of arrays used by the same Limber calculation; it does not change any noiseless theory,
profile, transfer, or saved comparison value. The adjacent noiseless-theory tests and the
new noisy-covariance construction exercise this return contract.

The large-scale hypothesis product is
`data/SBI_validate/three_probe_mock/validation/noiseless_cls_gate3_nside1024/`
`large_scale_diagnostic/large_scale_factorization.{npz,json,png}` and binds the nside-1024
comparison, the same-nside cap/full-sky null, and the independent catalog-HMF artifact.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  tests/test_sbi_three_probe_noiseless_cls.py
```

## Failure modes

- Theory uses selected-halo n(z) instead of realized fixed-HOD galaxy n(z).
- Gaussian, HEALPix pixel, aperture or shot-noise factors are applied twice or omitted.
- Smooth theory is sampled at effective ell instead of passed through saved NaMaster windows.
- A diagnostic hypothesis overwrites the immutable first matched comparison.

## Open questions

- Obtain an independent halo-matter cross to measure the selected proxy-halo bias.
- Calibrate `M_particle_proxy` to a physical SO mass before treating Tinker residuals as a
  mass-function validation.
- Measure the end-to-end pixel-centre/aperture transfer using representative single halos.
- Weight that transfer by the actual HMF, HOD and field-specific one-halo integrands; the
  current 25-node result is deliberately unweighted and is not a correction.
- Separate pixel-centre sampling from the half-pixel beam rule with a future same-physical-
  FWHM 512/1024 pair if that attribution is needed; the completed control changes both as
  requested and validates their combined numerical operator.
- Re-run a current-hash exact comparison-grid refinement; older refinement evidence is not
  sufficient to discharge that remaining numerical null.
