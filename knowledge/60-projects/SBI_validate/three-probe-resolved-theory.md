---
id: kb.sbi.three-probe-resolved-theory
title: Three-probe map-matched resolved theory
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: medium
scope:
  - notebooks/SBI_validate/three_probe_resolved_theory.py
  - notebooks/SBI_validate/validate_three_probe_resolved_theory.py
  - tests/test_sbi_three_probe_resolved_theory.py
invariants:
  - INV-PHYS-UNITS-01
  - INV-PHYS-1H2H-01
  - INV-JAX-GRAD-FINITE-01
  - INV-JAX-TRACE-01
  - INV-NZ-NORM-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_resolved_theory.py
  - "[slow] [needs-data] MPLCONFIGDIR=/tmp/godmax-mpl-cache /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/SBI_validate/validate_three_probe_resolved_theory.py"
verified_at_commit: 29c3a27
verified_on: 2026-08-19
see_also: [kb.sbi.three-probe-catalog-theory-input-contract, kb.sbi.three-way-mock-comparison-plan]
supersedes: []
scope_digest: sha256:00e89cc9c75a1738cc49784506f9533f
---

## Claim

The SBI three-probe resolved-theory adapter structurally assembles every galaxy, pressure,
electron, and matter 1h/2h component by direct integration over one exact catalog-matched
mass/redshift grid, without low-mass or unresolved completion. Its current spherical-support
profile transform is an unconverged candidate; it is not yet accepted as equivalent to the
projected painter and cannot support a posterior.

## Why it is true

`three_probe_resolved_theory.py:159-196` computes all ten 1h/2h pairs and all four raw effective
bias factors through the same dlnM quadrature. `three_probe_resolved_theory.py:226-278` directly
evaluates signed spherical transforms at every target k, avoiding FFTLog endpoint clamping and
sign erasure. `validate_three_probe_resolved_theory.py:120-185` applies exact catalog cosmology,
M/z grids, catalog n(z), numerical HMF/PK and construction order before Profiles/get_Pkz.

The 2026-08-19 baseline and robust commands produced current-code-bound artifacts with grids
(nM,nz,nk,nr)=(96,64,256,23) and (192,128,512,45), exact seven-shell c0000 cosmology and lens
integral 1.0. The independent referee returned CONDITIONAL: low-k bm moves from
0.898539--0.915114 to 0.934382--0.952208 and signed high-k zero crossings yield maximum symmetric
differences of 200%. Exact commands and hashes are in
`knowledge/.kb/ledgers/2026-08-19-sbi-three-probe-resolved-theory.md`.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  tests/test_sbi_three_probe_resolved_theory.py
# Expected: 6 passed.

MPLCONFIGDIR=/tmp/godmax-mpl-cache \
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/SBI_validate/validate_three_probe_resolved_theory.py

MPLCONFIGDIR=/tmp/godmax-mpl-cache \
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/SBI_validate/validate_three_probe_resolved_theory.py --robust
```

## Failure modes

- Consuming `bm_dmb_kz_mat` or `bm_nfw_kz_mat` silently restores missing low-mass bias or replaces it by unity.
- Rescaling the catalog proxy by `h` a second time shifts the mass threshold by 0.850914.
- Masking only HOD terms leaves y/electron/matter integrals on broader support than the pasted halo catalog.
- Treating the current radial grid as converged would turn a roughly 4% low-k matter response
  movement into a hidden posterior modeling error.
- Calling the spherical 8R200c cut identical to the painter's projected aperture would claim an
  operator equivalence that has not been tested.

## Open questions

- Radial convergence and projected-painter operator equivalence block quantitative use.
- Absolute comparison to a Tinker M200c model remains conditional on the provisional identification
  `M_particle_proxy == M200c`; this blocks a physical mass-definition validation but not the
  common-support assembly.
