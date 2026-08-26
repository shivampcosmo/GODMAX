---
id: kb.physics.halo-model-ingredients
title: Halo-model ingredients and the consistency relations they must satisfy
layer: 20-physics
owner: halo-model-physicist
status: verified
confidence: medium
scope:
  - src/get_radial_profiles.py
  - src/hmf_symbolic.py
  - src/matter_pk_symbolic.py
  - src/get_B12_profile.py
  - src/get_OWLS_profile.py
invariants:
  - INV-PHYS-MASSBUDGET-01
  - INV-PHYS-BIASNORM-01
  - INV-PHYS-1H2H-01
  - INV-PHYS-UNITS-01
  - INV-NZ-NORM-01
checks:
  - "TODO(halo-model-physicist): mass-budget and bias-normalisation tests on the production grid"
verified_at_commit: 29c3a27
verified_on: 2026-08-16
see_also: [kb.arch.class-chain, kb.numerics.jax-contract]
scope_digest: sha256:aacb4feac67b0857b454a2d5d8d5b84d
---

## Claim

The model is a 1-halo/2-halo halo model with a Schneider-style baryonic correction, a
SHMR-based HOD, and symbolic-regression emulators for the expensive cosmology functions.
Its correctness must be established by consistency relations — mass budget, bias
normalisation, transition limits, unit conventions, and kernel normalisation — not by
agreement with a previous implementation. This document distinguishes the relations that
have targeted evidence from the production-grid checks that remain open.

## Why it is true

Ingredients recorded in `src/context/codebase_summary.md` (section 1) and `README.md:183-192`:

| Ingredient | Implementation |
|---|---|
| HMF | Tinker 2008 / Tinker 2010 multiplicity, symbolic emulator for sigma(R) |
| Halo bias | Tinker 2010 linear b(M,z) |
| c(M,z) | Duffy08, Prada12, Diemer15 |
| Dark matter | NFW, optional truncation |
| Baryons | Schneider-style BCM: gas ejection + stellar condensation + collisionless relaxation |
| HOD | Leauthaud+11 SHMR: Bernoulli centrals + Poisson satellites, NFW radii |
| Pressure | BCM-derived thermal, plus Battaglia 2012/2016 and OWLS/LeBrun15 |
| P(k) | 1h + 2h, halofit for the 2-halo regime, `poweradd` or `response` transition |
| C(ell) | Limber with lensing, galaxy, tSZ, tau and IA windows |
| Transforms | FFTLog via the JAX mcfit port (`src/mcfitjax/`) |
| Emulators | sigma(R), P_lin, P_halofit, D(z), A_s(sigma_8) |

The consistency relations that constitute correctness:

**Mass budget** (`INV-PHYS-MASSBUDGET-01`, blocker). The registered check is the component
fraction identity on the halo `(z, M)` grid. The implementation sets
`fstar_tot = fstar_cen + fstar_sat` (`src/get_radial_profiles.py:239`),
`fgas = Ob0/Om0 - fstar_tot` (`:254`), and
`fclm = 1 - Ob0/Om0 + fstar_sat` (`:255`), so
`fgas + fclm + fstar_cen = 1`. The current default `22 x 24` grid verifies that identity to
float64 roundoff; the exact command and absolute residual are in the linked re-verification
ledger. This fraction check does not by itself establish closure of the separately
discretised radial density profiles.

For the galaxy-satellite collisionless window, the default source path now transforms the
supplied cumulative mass directly. It treats `M(r0)` as one unresolved CLM cell with the
NFW-cusp asymptotic `M(<r) proportional to r^2`, and each later `diff(M)` as a
constant-volume-density shell (`src/get_radial_profiles.py:845-856` and
`src/get_Pkzs.py:30-89`). This makes the represented monopole exact and gives the required
`u(k)=1-k^2<r^2>/6+...` limit without clipping signed high-k values. The p=2 continuation is
**only** for the unresolved CLM cell: central stars have a different inner mass law and gas
another, so it is not applied to the total DMB profile.

This does not certify the physical radial mass budget. On the freshly executed default
`nr=23`, `r=0.005--8 Mpc/h`, `M=10^11.5--10^15.5 Msun/h` grid, the represented endpoint
relative to `fclm*Mtot` reaches only `0.777810570` for the least-covered halo. The Fourier
normalization is therefore exact for what the grid represents, while outer coverage remains
an independent open failure. The source integration deliberately leaves `rho_clm_mat`,
`rho_dmb_mat`, and all matter-only spectra unchanged.

On the same actual default radial interval, an `nr=95` direct transform provides the current
high-resolution numerical reference. At `nr=23`, the direct result differs by at most
`8.43e-4` for `k<=0.1 h/Mpc` and `1.405e-2` over the full sampled range; the legacy
density/FFTLog result differs by `1.272e-1` and `1.290e-1`, respectively. Direct full-range
movement decreases from `1.144e-2` between 23 and 47 nodes to `2.612e-3` between 47 and 95
nodes. Both mass-grid endpoints are finite; the largest halo supplies the worst `k<=1`
coarse-grid error. This establishes convergence and a large accuracy improvement, not exact
nonlinear-scale convergence at the default resolution.

**Bias normalisation** (`INV-PHYS-BIASNORM-01`, high).
`∫ b(M,z) n(M,z) M dM / rho_m = 1` within tolerance on the production grid, given the adopted
low-mass completeness correction. This is the statement that all matter lives in halos, and
the number is meaningless without stating the grid limits alongside it.

**Transition limits** (`INV-PHYS-1H2H-01`, high). `analysis.gg_transition_model` is recorded
(`poweradd` for xDESI); P_tot → 2-halo at low k; 1-halo dominates at high k; no unphysical
bump or dip at the crossover. Prescriptions differ by tens of percent near k ~ 1 h/Mpc —
exactly where the gas and HOD parameters are constrained, which is why
`notebooks/xDESI/abacus_paste/compare_physical_transition_variants.py` and
`compare_matter_electron_poweradd_variants.py` exist.

**Units and h** (`INV-PHYS-UNITS-01`, blocker). Explicit at every module boundary: Msun/h vs
Msun, Mpc/h vs Mpc, y dimensionless, CMB temperature in microkelvin, C_ell in the units of
the product being compared. For the repaired HOD path, `get_Mstar_Mh` divides the stored halo
mass by `h` before the SHMR inversion and multiplies the returned stellar mass by `h`
(`src/get_radial_profiles.py:621-630`); the satellite thresholds retain the same conversion
and mass ratios (`:641-650`). Both occupations returned by that interface are dimensionless.
This is a change-local audit, not a claim that all older module boundaries have complete unit
annotations.

**Kernel normalisation** (`INV-NZ-NORM-01`, high). Every Limber kernel integrates to 1 within
1e-6 on the projection grid. DES source n(z) arrives as raw FITS bin values and must be
normalised.

Recorded xDESI grid: `halo_params.zmin 0.005`, `zmax 3.0`, `nz 96`; projection
`analysis.nz_for_Cls 192`. These are physics choices — results depend on them.

The HOD construction repair is deliberately narrow. `run_stars_calc` invokes separate
class-level `get_Ncen` and `get_Nsat` methods (`src/get_radial_profiles.py:223-239,
:632-650`). The regression fixture checks both analytic occupation formulas, finite nonzero
derivatives with respect to the threshold, the component-fraction identity, and the
unchanged non-galaxy power-law branch (`tests/test_get_radial_profiles.py:95-176`). It does
not validate the full-likelihood prior corners or the convergence of radial profile
integrals.

`src/arxiv/` holds 24 superseded modules. History only.

## How to verify

```bash
python tools/kb/kb.py invariants --check --layer physics   # most are MANUAL — argue them
git grep -n "gg_transition_model" -- param_files notebooks/xDESI

# targeted HOD formulas, fraction identity, gradients, and non-galaxy null
/usr/bin/env JAX_PLATFORMS=cpu /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  -m pytest tests/test_get_radial_profiles.py -q
```

Every `MANUAL` result requires a written `HOLDS because <file:line>` argument plus the
numbers, in the evidence ledger. Report the grid limits with every integral-based number.

## Failure modes

- **Mass-budget violation.** Matter P(k) amplitude drifts with baryon parameters at
  k < 0.1 h/Mpc, where baryons should be irrelevant — degenerate with sigma8.
- **Confusing a normalized satellite window with mass closure.** `u_clm(0)=1` can remain
  exact when `Mclm(rmax)/(fclm*Mtot)` is substantially below one. Always report endpoint
  coverage and negative shell increments separately.
- **Bias normalisation off.** 2-halo amplitude systematically wrong; the HOD's effective bias
  disagrees with the matter large-scale bias.
- **Transition artefact.** A localised bump or dip near k ~ 1 h/Mpc that propagates into the
  mid-ell bandpowers of every galaxy and gas spectrum.
- **h-factor error.** Scale-independent multiplicative offset: it survives every shape-based
  test and is absorbed by an amplitude parameter. Best-fit amplitudes off by ~0.67, ~0.45, or
  ~1.49 with acceptable shape are the signature.
- **Unnormalised kernel.** Constant multiplicative offset in one probe family, degenerate
  with a calibration or bias amplitude.
- **Emulator extrapolation.** `matter_pk_symbolic.py` and `hmf_symbolic.py` are fits; outside
  their training domain they extrapolate smoothly and wrongly, with no warning.
- **An unphysical best fit accepted because chi2 fell.** With 31 flexible astrophysical
  parameters, chi2 improvement is nearly free. Always check whether the gas fraction is below
  the cosmic baryon fraction, the SHMR is sane at both ends, the satellite fraction is
  plausible, and the non-thermal pressure fraction is bounded. An unphysical requirement
  means the error is elsewhere — usually in the comparison, not the model.

## Open questions

- The HMF selector, concentration selector, HOD occupations, and component-fraction
  definitions have source anchors above, while the remaining ingredient inventory is still
  inherited from `src/context/codebase_summary.md` and `README.md`. A complete line audit of
  every profile remains open. Owner: `halo-model-physicist`.
- The synthetic regression now automates the HOD formulas and the algebraic component-
  fraction identity. Production-grid bias normalisation, 1h/2h transition limits, kernel
  normalisation, and a true integrated radial-profile closure test remain unautomated.
  A stronger diagnostic in the 2026-08-16 ledger found a material, resolution-dependent
  mismatch between the integral of `rho_dmb` and `Mtot`; it is not caused by the narrow HOD
  method repair and must not be confused with the passing registered fraction check. Owner:
  `repro-runner` with `halo-model-physicist`.
- The training domains of the symbolic emulators are not documented anywhere in the tree.
  Until they are, any new cosmology or mass/redshift range is an unquantified risk. Owner:
  `halo-model-physicist`.
