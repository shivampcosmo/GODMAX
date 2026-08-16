---
id: kb.des-cluster.redmapper-y-cross
title: DES-cluster redMaPPer x Compton-y simulation/data comparison
layer: 60-projects
owner: measurement-namaster
status: verified
confidence: medium
scope:
  - notebooks/DES_cluster/params_redmapper_y_cross.yaml
  - notebooks/DES_cluster/redmapper_y_cross.py
  - notebooks/DES_cluster/redmapper_y_sim_data_comparison.ipynb
  - notebooks/DES_cluster/params_redmapper_y_cross_z0p4_0p6.yaml
  - notebooks/DES_cluster/redmapper_y_sim_data_comparison_z0p4_0p6.ipynb
  - notebooks/DES_cluster/params_redmapper_y_cross_z0p4_0p6_thetaejx1p5.yaml
  - notebooks/DES_cluster/redmapper_y_sim_data_comparison_z0p4_0p6_thetaejx1p5.ipynb
  - notebooks/DES_cluster/test_redmapper_y_cross.py
invariants:
  - INV-BEAM-01
  - INV-PHYS-UNITS-01
  - INV-PRODUCT-PROV-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q notebooks/DES_cluster/test_redmapper_y_cross.py"
verified_at_commit: cf72943
verified_on: 2026-08-10
see_also:
  - kb.des-cluster.tsz-paste
supersedes: []
scope_digest: sha256:ac7d573525d1b7abbcc371ec99227fc2
---

## Claim

The scoped executed notebook measures a compensated TreeCorr cluster-y angular profile
for the requested simulation and for the ACT x DES data inputs using the same
strict richness/redshift cuts as `measurements/ACTxDES/Cluster_SZ_measurements.ipynb`.
It is a conditional diagnostic comparison, not a likelihood, fit, calibrated
PTE, or survey-grade detection.  Its final independent validation disposition
is REFUTED because the simulation pixel-resolution and random-density evidence
are insufficient for an unconditional science claim.

## Why it is expected to be true

The executed notebook applies the source notebook's strict `lambda > 20` and
`0.5 < z < 0.8` cuts, uses the `z` column, and computes TreeCorr's compensated
`NKCorrelation` profile.  Its dated ledger records the preregistered
random-density decision, a random-position null, a failed coarse-resolution
lap, the passing factor-2-versus-native data-map control, current hashes, and an
independent read-only reproduction.  The simulation uses a common inclusive
first-octant footprint; data CAR pixels use exact solid-angle weights.  These
facts support reproducibility of the displayed diagnostic but do not override
the failed resolution refutation recorded below.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  notebooks/DES_cluster/test_redmapper_y_cross.py

/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -c \
  "import nbformat; nb=nbformat.read('notebooks/DES_cluster/redmapper_y_sim_data_comparison.ipynb',4); nbformat.validate(nb); assert [c.execution_count for c in nb.cells if c.cell_type=='code']==[1,2,3,4,5]; assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']; print('executed notebook valid')"
```

The exact commands, outputs, hashes, and conditional interpretation are in
`knowledge/.kb/ledgers/2026-08-09-des-cluster-redmapper-y-cross.md`.

## Theta-ejection sensitivity rerun

The strict `lambda > 20`, `0.4 < z < 0.6` comparison was also re-executed with
the separate NSIDE-2048 simulation y map in which only `theta_ej_0` changes
from 2.0 to 3.0.  The ACT-side inputs and estimator configuration remain fixed,
and the final notebook deliberately replaces the verbose helper figure with a
single simple data/simulation profile panel.  This sensitivity run inherits
the same `refuted_data_resolution` claim boundary; it does not rehabilitate the
original simulation-resolution or random-density limitations.  Commands,
hashes, exact last-digit data null, and the simulation-profile change are in
`knowledge/.kb/ledgers/2026-08-10-des-cluster-thetaej-x1p5.md`.

## Failure modes

- Treating the first-octant simulation as full sky gives the random correction
  the wrong window.
- Using `z_lambda` instead of the source notebook's `z` changes the simulation
  sample.
- Equal weights on CAR map pixels over-weight high-|Dec| rows; the new estimator
  must use solid-angle pixel weights.
- A factor-3 (1.5-arcmin) ACT curve failed the fixed maximum-shift control; the
  accepted science curve uses factor 2 (1 arcmin) and is checked against native
  0.5-arcmin pixels without loosening the threshold.
- The selected 20x simulation random sample was its own reference.  The 10x
  sample failed the fixed median-shift condition, and no denser external
  reference was successfully validated.  Thus 20x is a bounded operational
  choice, not a demonstrated convergence plateau.
- A final-lap NSIDE=2048 versus NSIDE=1024 simulation coarsening control for
  theta >= 5 arcmin returned median 0.0290 sigma but maximum 2.332 sigma,
  failing the preregistered 0.50-sigma maximum.  No angular cut or tolerance
  was changed to hide that failure.
- The randomA-randomB chi-square survival values use the same 100-patch
  jackknife covariance and have no finite-covariance calibration.  Treat them
  as uncalibrated diagnostics, not PTEs or selection-function validation.
- A simulation/data amplitude ratio is not physically interpretable without
  verified ACT map units/transfer and matched redshift/richness distributions.

## Open questions

- The ACT FITS header does not record `BUNIT`, beam, or filter transfer.  Any
  dimensionless-y and 1.6-arcmin beam interpretation remains conditional on the
  established ACT analysis convention rather than file-local metadata.
- The simulation random catalog has no cosmology/unit attributes; its `R` field
  is treated as c000 comoving Mpc/h only after a geometry/radial control.
- The requested simulation uses the `alpha_run` redMaPPer catalog, whereas the
  historical data notebook names a different `des_run` catalog.  The common
  cuts/estimator are reproduced, not historical row identity across catalogs.
- The historical source notebook hash was not stored in the lap-2 HDF5, so its
  exact source identity remains an external provenance gap.
- Non-unit catalog weights are intentionally ignored to reproduce the source
  notebook's unweighted convention; the split-random null cannot validate that
  shared modeling choice.
- Threaded TreeCorr reductions reproduce all decisions and plots but differ in
  null/resolution summary values at roughly the final 12-15 decimal places.
