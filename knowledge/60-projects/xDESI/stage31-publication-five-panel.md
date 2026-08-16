---
id: kb.xdesi.stage31-publication-five-panel
title: Stage-31 pz3 publication comparison figures
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: medium
scope:
  - notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py
  - notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py
  - notebooks/xDESI/abacus_paste/measure_stage31_pz3_shear_auto.py
invariants:
  - INV-WINDOW-CMP-01
  - INV-SHEAR-SIGN-01
  - INV-HOD-PZBIN-01
  - INV-ABACUS-COSMO-01
  - INV-PROC-EVIDENCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py --check-only
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py --check-only
  - /mnt/home/spandey/ceph/env/godmax-jaxhealpy/bin/python notebooks/xDESI/abacus_paste/measure_stage31_pz3_shear_auto.py --check-only
verified_at_commit: cf72943
verified_on: 2026-08-10
see_also: [kb.xdesi.abacus-paste, kb.xdesi.analysis-state]
supersedes: []
scope_digest: sha256:a03c7af6b6527dcf0d1aed9a99f7a7b3
---

## Claim

The Stage-31 pz3 publication figures render saved, windowed data/theory spectra and Abacus
simulation measurements without changing their numerical series. The full figure contains
five panels, including the focused pasted DES source-bin-3 shear auto; the compact figure
contains the `gg`, `g-y`, and `g-shear` panels in one row with the shared legend inside the
`gg` panel. Both shade the band ranges excluded by the exact Stage-31 likelihood cuts.

## Why it is true

The five exact spectrum names and their theory-product assignments are fixed at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py:25`. The script
requires the pz3 Abacus-cosmology configuration at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py:79`, verifies the
full-vector data and covariance exactly against the HDF5 measurement at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py:185`, and slices every
spectrum by its saved name and joint-vector bounds at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py:207`. It uses the saved
full-survey, bandpower-windowed MAP vector as the data-matched theory for all five panels at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py:241`. The check report
records SHA-256 digests for every plotted numerical series at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py:267`.

The compact renderer fixes exactly the `gg`, `g-y`, and `g-shear` spectrum names at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py:28`. It reads only
those saved survey and simulation groups at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py:112` and
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py:138`, then records the
same numerical and survey/simulation-window digests at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py:223`. The renderer
constructs one row of three axes and places the sole shared legend inside the first axis at
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py:278` and
`notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py:386`.

The focused simulation measurement reads `maps/map_kappa_wl_tomo3`, subtracts its mean on
the configured cap, constructs the E-only spin-2 proxy, and passes the standard tomo3 EE
spectrum specification to the existing NaMaster measurement path at
`notebooks/xDESI/abacus_paste/measure_stage31_pz3_shear_auto.py:76` and
`notebooks/xDESI/abacus_paste/measure_stage31_pz3_shear_auto.py:235`. It records zero shape
noise, the binary-cap mask policy, the shear sign convention, and the finite-cap caveat in
the focused HDF5 provenance.

The likelihood overlay uses the configured center-based family cuts. It shades
`1730 <= ell < 3001` for galaxy auto, galaxy-y, and galaxy-shear; and
`1000 <= ell < 3001` for galaxy-CMB-lensing. All 13 shear-auto bands satisfy its default
`ell_max=3000` cut and therefore remain unshaded.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_5panel.py --check-only
# Expected: input_identity.data_vector_equal=true,
# input_identity.covariance_equal=true, and five entries under spectra.

/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/xDESI/abacus_paste/plot_stage31_pz3_publication_3panel.py --check-only
# Expected: the same two identity flags and exactly three entries under spectra.
```

## Failure modes

An incorrect spectrum name selects the wrong tomographic bin; mixing the full-survey and
simulation-matched theory products changes the meaning of the theory label; selecting a
spin-2 component other than EE changes the shear-auto observable; or a missing pasted
shear-auto spectrum is accidentally represented as a simulated null.

## Open questions

The pasted shear field is an E-only spin-2 proxy generated from a cap-limited convergence
map over halos selected at `0.63 < z < 0.98` (actual catalog extrema
`0.63000065` and `0.97999948`). Its finite-cap and incomplete line-of-sight construction
remain upstream simulation caveats; the shear-auto panel labels this directly and the
figure provenance records it. The raw cap-simulation bandpowers also retain their saved
cap-estimator windows, while the data and blue theory retain the survey windows. The
legend's “theory matched” wording refers to the simulation's selected MAP physical inputs
and map transfers, not equality of cap and survey bandpower windows.

The repository's only existing all-z cap2400 continuous-field paste uses the older,
explicitly non-converged `hmcfailed` point, `ell_max=4096`, and 10 linear bands. It is not
a compatible substitute for the current 64-parameter MAP product. A physically complete
tomo3 shear-auto simulation requires a new current-parameter continuous-field paste over
the source-bin lensing line of sight; the present pz3-slice curve must not be interpreted
as that result.
