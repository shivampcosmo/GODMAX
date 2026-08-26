---
id: kb.sbi.three-probe-projected-operator
title: Three-probe projected painter operator validation
layer: 60-projects
owner: abacus-paste-validator
status: verified
confidence: medium
scope:
  - notebooks/SBI_validate/three_probe_projected_operator.py
  - notebooks/SBI_validate/validate_three_probe_projected_operator.py
  - tests/test_sbi_three_probe_projected_operator.py
invariants:
  - INV-PHYS-UNITS-01
  - INV-PHYS-MASSBUDGET-01
  - INV-ABACUS-COSMO-01
  - INV-JAX-X64-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q tests/test_sbi_three_probe_projected_operator.py"
verified_at_commit: 29c3a27
verified_on: 2026-08-19
see_also: []
supersedes: []
scope_digest: sha256:505702869dfb9862a964d3895fd832b8
---

## Claim

The current spherical `8 R200c` profile mask is not the same operator as the
three-probe map painter.  The unit-consistent cosh path projects through the
global physical radial table, whereas the legacy default uses a mixed-coordinate
`min(r_table,100 Rp)` bound; both are sampled through a transverse `8 R200c`
pixel aperture rather than a spherical cut.  The host-side validation is diagnostic evidence only:
the projected operator is not yet implemented in the differentiable theory
graph, and pasted-map or posterior validation has not started.

## Why it is true

`src/get_sim_maps.py:440-472` shows that the optional cosh path projects a
positive physical profile to the global physical table edge; the legacy path
uses the distinct mixed-coordinate bound at `src/get_sim_maps.py:474-487`.
Independently, the SBI painter constructs a
physical `R200c`, sets the transverse aperture to eight times that radius and
passes physical `DA*theta` to the map interpolator
(`notebooks/pasting/paste_backlight_utils.py:544-554,570-574`).  Material at
spherical radius greater than `8 R200c` but transverse radius below it is
therefore retained by the painter and removed by the spherical candidate.

The comparison keeps `r` and `k` comoving, projected radius physical, and uses
`q_phys=k_comoving*(1+z)` (`three_probe_projected_operator.py:127-150`).  The
electron and matter physical profiles include the `a^-3` density conversion;
the y plane transform receives `a^-3` because GODMAX's existing `y3d` table is
physical pressure sampled on a comoving radius (`validate_three_probe_projected_operator.py:56-77`).

The catalog-bound run used all seven source headers, the exact c0000 cosmology,
`5e11 <= M < 1e16 Msun/h`, and `0.3 < z < 0.5`.  At the doubled
`(nM,nz,nr,nLOS,nRp)=(192,128,45,256,2049)` grid, over `0 <= k <= 2 h/Mpc`:

- the continuous cylindrical operator has median absolute symmetric difference
  0.00740 from the spherical candidate and maximum absolute difference divided
  by the spherical zero mode 0.01020;
- the unit-consistent painter-table emulation has 0.00965 and 0.03994;
- the legacy production-default projector at 32 LOS points has 0.03448 and
  0.06216.

These are descriptive results, not acceptance thresholds.  Exact commands are
recorded in the evidence ledger; array hashes and numerical results are in
`data/SBI_validate/three_probe_mock/validation/projected_operator/projected_operator_summary.json`.

The downstream fast-paste gate now exercises production-source parity at both 32 and 64 LOS
nodes. On common `(nr,nM,nz)=(48,24,48)` profiles, its 32-to-64 maximum zero-mode-normalized
change is `4.01e-5`. One-axis 48-to-96 radial refinement moves the representative painted
transforms by at most 3.42%; mass and redshift refinements are smaller. These results validate
the chosen fast painter within the registered 5% envelope, but do not make the projected
operator differentiable or establish map-level Cl agreement. Commands and current hashes are in
`knowledge/.kb/ledgers/2026-08-19-sbi-three-probe-fast-paste.md`.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  tests/test_sbi_three_probe_projected_operator.py
# Expected: 7 passed.

/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/SBI_validate/validate_three_probe_projected_operator.py
# Expected status: SUPPORT_GEOMETRY_MISMATCH_CONFIRMED_NUMERICAL_REPLACEMENT_UNCONVERGED
```

## Failure modes

- Treating projected radius as comoving omits the `1+z` phase conversion and
  produces a redshift-dependent profile error.
- Omitting the y `a^-3` plane-to-volume factor produces a hidden `(1+z)^3`
  normalization error.
- The default `legacy_log_radius` projector mixes a physical interpolation grid
  with a comoving upper bound (`src/get_sim_maps.py:474-487`); at its production
  32-point LOS setting the doubled-grid diagnostic spans about -6.1% to +6.2%
  at zero mode, depending on field/M/z.
- The painter table starts at `r_array[2]`, stores float32 values and fills
  out-of-range log values with `-20` (`src/get_sim_maps.py:138-139,239-267`),
  so a continuous cylinder calculation alone does not reproduce the actual
  tabulated painter.

## Open questions

- The old 23-to-45 radial-grid comparison was non-negligible and shifted mass/redshift nodes.
  The fast-paste gate instead freezes 48 nodes after common-target 48-to-96 evidence, but a much
  denser run is still required before claiming sub-percent replacement-operator convergence.
- The current validation is unsmoothed.  The painter's optional profile
  smoothing and HEALPix pixel-centre aperture need a separate transfer test.
- The cylindrical operator still needs a JAX-native, gradient-tested theory
  implementation before map or posterior validation.
