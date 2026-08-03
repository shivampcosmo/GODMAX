---
id: kb.physics.halo-model-ingredients
title: Halo-model ingredients and the consistency relations they must satisfy
layer: 20-physics
owner: halo-model-physicist
status: draft
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
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.arch.class-chain, kb.numerics.jax-contract]
scope_digest: sha256:0a1f338b2f415c7fedd4d74ec588171c
---

## Claim

The model is a 1-halo/2-halo halo model with a Schneider-style baryonic correction, a
SHMR-based HOD, and symbolic-regression emulators for the expensive cosmology functions. Its
correctness is established by consistency relations — mass budget, bias normalisation,
transition limits, unit conventions, kernel normalisation — not by agreement with a previous
implementation.

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

**Mass budget** (`INV-PHYS-MASSBUDGET-01`, blocker). Gas + stars + collisionless matter sums
to the total halo mass within tolerance at every grid point. The BCM redistributes mass; it
does not create it.

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
the product being compared.

**Kernel normalisation** (`INV-NZ-NORM-01`, high). Every Limber kernel integrates to 1 within
1e-6 on the projection grid. DES source n(z) arrives as raw FITS bin values and must be
normalised.

Recorded xDESI grid: `halo_params.zmin 0.005`, `zmax 3.0`, `nz 96`; projection
`analysis.nz_for_Cls 192`. These are physics choices — results depend on them.

`src/arxiv/` holds 24 superseded modules. History only.

## How to verify

```bash
python tools/kb/kb.py invariants --check --layer physics   # most are MANUAL — argue them
grep -rn "gg_transition_model" param_files/ notebooks/xDESI/ | head
```

Every `MANUAL` result requires a written `HOLDS because <file:line>` argument plus the
numbers, in the evidence ledger. Report the grid limits with every integral-based number.

## Failure modes

- **Mass-budget violation.** Matter P(k) amplitude drifts with baryon parameters at
  k < 0.1 h/Mpc, where baryons should be irrelevant — degenerate with sigma8.
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

- Derived from `src/context/codebase_summary.md` and `README.md`, not from line-level reading
  of `get_radial_profiles.py` (960 lines). `confidence: medium`. Owner:
  `halo-model-physicist`.
- **No consistency relation is automated.** All five invariants above are `check.kind:
  manual`. Mass budget and kernel normalisation are the two most tractable to convert into
  pytest cases and would give the largest protection per line of test code. Owner:
  `repro-runner` with `halo-model-physicist`.
- The training domains of the symbolic emulators are not documented anywhere in the tree.
  Until they are, any new cosmology or mass/redshift range is an unquantified risk. Owner:
  `halo-model-physicist`.
