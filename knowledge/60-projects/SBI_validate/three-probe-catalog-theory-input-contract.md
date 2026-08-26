---
id: kb.sbi.three-probe-catalog-theory-input-contract
title: Three-probe catalog-to-theory input contract
layer: 60-projects
owner: xdesi-lead
status: verified
confidence: medium
scope:
  - notebooks/SBI_validate/three_probe_mock_contract.py
  - notebooks/SBI_validate/validate_three_probe_catalog_theory.py
  - notebooks/SBI_validate/three_probe_mock_experiment.yaml
  - tests/test_sbi_three_probe_catalog_theory.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-NZ-NORM-01
  - INV-PHYS-UNITS-01
  - INV-PROC-EVIDENCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_catalog_theory.py
  - "[slow] [needs-data] MPLCONFIGDIR=/tmp/godmax-mpl-cache /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/SBI_validate/validate_three_probe_catalog_theory.py"
verified_at_commit: 29c3a27
verified_on: 2026-08-19
see_also: [kb.sbi.three-way-mock-comparison-plan, kb.xdesi.abacus-paste]
supersedes: []
scope_digest: sha256:fc49f6191c2721c7f2b1b7ff6f63a0cf
---

## Claim

The three-probe validator produces a fail-closed, provenance-bound input bundle in which the
selected c0000 catalog, catalog lens kernel, and numerical GODMAX HMF/bias diagnostic share the
catalog cosmology and exact resolved mass/redshift support. Its pass status covers only this input
contract; it does not satisfy the common-field integral or threshold-posterior requirements of
full Gate 2.

The YAML now also carries the downstream fast-paste contract. That section consumes, but does
not alter, this frozen catalog/cosmology/kernel contract. Its separate validation is recorded in
`knowledge/.kb/ledgers/2026-08-19-sbi-three-probe-fast-paste.md`.

## Why it is true

`three_probe_mock_experiment.yaml:75-126` freezes the catalog identity, source header, exact
`0.3<z<0.5` and `5e11<=Mproxy<1e16 Msun/h` conventions, no-unresolved-completion mode, kernel
settings, and numerical theory grids. `three_probe_mock_contract.py:73-202` rejects incomplete or
different catalog/source cosmology, wrong catalog hashes, wrong units or mass semantics, failed
source-shell coverage, boundary-shell leakage, and altered selection conventions.

`validate_three_probe_catalog_theory.py:211-298` applies the catalog cosmology before constructing
GODMAX, uses numerical rather than symbolic HMF/power, and verifies the effective constructor
cosmology and grid endpoints. `validate_three_probe_catalog_theory.py:525-715` streams every row,
requires every frozen source shell to contribute, writes normalized kernels and plotted arrays,
and records a status that leaves threshold posteriors pending.

The 2026-08-18 evidence command exited 0 with 66,159,463 primary rows, 63,635,766 rows at
`N_interp>=125`, 53,466,475 at `N_interp>=150`, seven contributing shells, three kernel integrals
equal to 1.0, and exact catalog/source/effective theory cosmology equality. The exact command and
all numbers are recorded in
`knowledge/.kb/ledgers/2026-08-18-sbi-three-probe-gate2-theory.md`.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  tests/test_sbi_three_probe_catalog_theory.py
# Expected: 3 passed.

MPLCONFIGDIR=/tmp/godmax-mpl-cache \
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/SBI_validate/validate_three_probe_catalog_theory.py
# Expected: exit 0 and status
# GATE2_INPUT_CONTRACT_PASS_COMMON_FIELD_INTEGRALS_AND_THRESHOLD_POSTERIORS_PENDING.
```

## Failure modes

- Missing or changed cosmology/header keys stop before GODMAX construction; allowing a fallback
  would instead yield a plausible HMF at the wrong `H0`, `ns`, or density parameters.
- Multiplying `M200c_hMsun` by `h` a second time shifts all masses by 0.850914 and can leave a
  deceptively smooth HMF curve.
- Reusing the old DESI lens kernel breaks catalog n(z) consistency even when its normalization is
  numerically one.
- Treating the resolved mass-weighted bias integral as a normalization check forces an invalid
  low-mass completion into a map-matched resolved calculation.
- Reporting the input-contract status as full Gate 2 hides the still-unrun 125/150 posterior
  sensitivity requirement.

## Open questions

- The shared exact-support masks/integrals for galaxy, pressure-y, electron-tau, and matter fields
  are not yet implemented. Owner: xdesi-lead + halo-model-physicist. Blocks full Gate 2.
- Posterior mean/width changes at 125 and 150 particles are not yet measured. Owner:
  inference-statistician. Blocks acceptance of the provisional `5e11 Msun/h` floor.
- A true empirical halo-bias validation needs a matched particle-matter map. Owner:
  abacus-paste-validator. It does not block the narrow input-contract artifact.
