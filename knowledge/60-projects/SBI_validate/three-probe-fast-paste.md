---
id: kb.sbi.three-probe-fast-paste
title: Three-probe fast split-aware map pasting
layer: 60-projects
owner: abacus-paste-validator
status: verified
confidence: medium
scope:
  - src/get_sim_maps.py
  - notebooks/xDESI/abacus_pasting_helpers.py
  - notebooks/SBI_validate/three_probe_mock_experiment.yaml
  - notebooks/SBI_validate/three_probe_fast_paste.py
  - notebooks/SBI_validate/submit_three_probe_fast_paste.sbatch
  - notebooks/SBI_validate/submit_three_probe_fast_combine.sbatch
  - tests/test_sbi_three_probe_fast_paste.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-JAX-SEED-01
  - INV-PHYS-UNITS-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_fast_paste.py"
verified_at_commit: 29c3a27
verified_on: 2026-08-19
see_also:
  - kb.sbi.three-probe-catalog-theory-input-contract
  - kb.sbi.three-probe-projected-operator
  - kb.xdesi.abacus-paste
supersedes: []
scope_digest: sha256:3f0a1f2d33b5c22810f1dddec51b4b23
---

## Claim

The frozen nside=512 paste is complete and strictly combined. It uses the exact c0000 catalog cosmology and resolved support, `physical_table_cosh` with 32 LOS nodes, `(nr,nM,nz)=(48,24,48)`, and a positivity-preserving real-space Gaussian profile convolution with FWHM exactly half the nside=512 pixel size (3.4354864118 arcmin). The product contains finite y/tau/CMB-kappa maps, the fixed HOD galaxy catalog, normalized realized galaxy n(z), the selected-halo input n(z), the exact CMB lensing-efficiency array, and the intended Gaussian transfer through ell=1535. Full map/theory agreement remains intentionally untested until the next saved-window Cl gate.

## Why it is true

The catalog/hash/cosmology/kernel preflight passes. On identical baseline profiles, 32 versus 64 LOS nodes changes the projected transform by at most `4.01e-5` under the frozen zero-mode-normalized metric. One-axis grid refinements move the representative transforms by at most 3.42%, below the pre-registered 5% gate. The Gaussian table closure is 0.421%/0.822%/1.311% for y/tau/kappa through ell=1000, and 64-to-128 convolution quadrature changes are at most 0.314%. Array job 6904954 completed all 32 splits with peak task RSS 3.82 GiB; strict combine job 6904955 completed with exact coverage of all 66,159,463 rows. The final 1.13 GB HDF5 has SHA256 `8aa0b7003ad2e3073e3f4440721939184817daddb5aaf069a91ee3d0b95e2536`; both input and realized n(z) integrate to 1. See `knowledge/.kb/ledgers/2026-08-19-sbi-three-probe-paste-run.md` for commands and exact output.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  tests/test_sbi_three_probe_fast_paste.py
/usr/bin/env JAX_PLATFORMS=cpu JAX_ENABLE_X64=True MPLCONFIGDIR=/tmp/matplotlib \
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/SBI_validate/three_probe_fast_paste.py validate-paste-contract \
  --config notebooks/SBI_validate/three_probe_mock_experiment.yaml \
  --output-dir data/SBI_validate/fast_paste_validation
```

## Failure modes

- A split or chunk processes a halo more or less than once.
- Catalog cosmology or M/z support differs from the resolved-theory contract.
- Changing parallel layout silently changes an unrecorded galaxy realization.
- A low-resolution profile or HEALPix grid exceeds the frozen accuracy envelope.
- Theory applies the saved Gaussian transfer more than once, or conflates it with galaxy-count pixelization.
- A combined product omits or changes realized HOD n(z), CMB Wkappa, smoothing arrays, or their code/config/catalog hashes.

## Open questions

- Noiseless saved-window Cl agreement, the map-level aperture/pixel-centre transfer closure, and the nside=1024 fixed-transfer control remain for the next validation gate. The intended Gaussian alone is not claimed to capture those additional painter operators exactly.
