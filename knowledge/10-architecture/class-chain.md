---
id: kb.arch.class-chain
title: The GODMAX class chain and its construction contract
layer: 10-architecture
owner: godmax-core
status: draft
confidence: medium
scope:
  - src/base_class.py
  - src/get_radial_profiles.py
  - src/get_Pkzs.py
  - src/get_Cls.py
  - src/get_Xis.py
  - src/get_covs.py
  - tests/test_get_radial_profiles.py
invariants:
  - INV-JAX-TRACE-01
  - INV-PHYS-UNITS-01
checks:
  - /usr/bin/env JAX_PLATFORMS=cpu /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest tests/test_get_radial_profiles.py -q
  - "TODO(godmax-core): construction smoke test — build the chain from params_default.yaml and assert C(ell) shape"
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.arch.params-dicts, kb.numerics.jax-contract]
scope_digest: sha256:3fcbe3ee73528ee27f9290f573f0e7e2
---

## Claim

The computation is a linear chain of five classes, each constructed by **dependency
injection** of the previous instance, with all four parameter dicts threaded through every
layer. A layer may depend only on layers below it. The `get_Cl` constructor is not JIT-able
standalone but traces correctly inside a numpyro model or a jitted function.

## Why it is true

The construction pattern is documented in `README.md:115-118` and is the pattern used by
every consumer:

```python
base_test     = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
profiles_test = Profiles(..., base_class_obj=base_test)
Pkz_test      = get_Pkz(..., Profiles_obj=profiles_test)
Cl_test       = get_Cl(..., Pkz_obj=Pkz_test)
```

Within `Profiles`, galaxy-enabled construction resolves `get_Ncen` and `get_Nsat` as
separate class-level, JIT-decorated methods (`src/get_radial_profiles.py:632-650`) before
building the central and satellite stellar fractions (`src/get_radial_profiles.py:223-239`).
`tests/test_get_radial_profiles.py` regression-tests this contract through the live
`run_stars_calc` path and checks the non-galaxy branch as its null control.

The chain and its responsibilities are recorded in `src/context/codebase_summary.md`
(section 2.1):

```text
base_class            src/base_class.py            cosmology, grids, linear P(k,z), growth
  -> Profiles         src/get_radial_profiles.py   HMF, c(M,z), NFW/gas/stellar/CLM, pressure, HOD
    -> get_Pkz        src/get_Pkzs.py              FFTLog -> u(k), 1h+2h P(k,z) per probe pair
      -> get_Cl       src/get_Cls.py               Limber -> C(ell)
        -> get_xi     src/get_Xis.py               Hankel -> xi(theta)
        -> get_cov    src/get_covs.py              Gaussian + trispectrum covariance
```

Branches: `setup_sim_map` and `get_sim_map` (`src/get_sim_maps.py`, 1555 lines) descend from
`Profiles`; `Battaglia_12_16` (`src/get_B12_profile.py`) and the OWLS/LeBrun15 profile
(`src/get_OWLS_profile.py`) descend from `base_class`.

There is **no packaged public API**. Consumers use `sys.path.insert` and import modules
directly — see `README.md:87-95` and `tests/test_xdesi_multiprobe_namaster.py:12-15`.
Every function signature is therefore effectively public, and callers are hand-written
imports spread across `run_scripts/`, `notebooks/`, and `tests/`.

`src/arxiv/` holds 24 superseded modules (`setup_power_spectra.py`,
`get_BCMP_profile_*_jit.py`, `get_power_spectra_NO_CONC_*.py`, …). It is history, not
current behaviour.

## How to verify

```bash
# the construction pattern used by real consumers
grep -rn "base_class_obj=\|Profiles_obj=\|Pkz_obj=" --include=*.py --include=*.ipynb . | grep -v src/arxiv

# enumerate callers before changing any signature (both .py and .ipynb)
grep -rn "from get_Cls import\|import get_Cls\|get_Cl(" --include=*.py --include=*.ipynb . | grep -v src/arxiv

# targeted core HOD construction, formulas, gradients, and non-galaxy null
/usr/bin/env JAX_PLATFORMS=cpu /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  -m pytest tests/test_get_radial_profiles.py -q

# module sizes — the split pressure points
find src -maxdepth 1 -name "*.py" | xargs wc -l | sort -rn | head
```

Expected: `get_sim_maps.py` ~1555 lines, `get_radial_profiles.py` ~960, `get_covs.py` ~673.

## Failure modes

- **Upward dependency** (`Profiles` reaching into `get_Cl`) creates a construction cycle
  whose symptom is a confusing tracing error raised from inside a numpyro model, far from
  the actual cause.
- **Signature change without caller enumeration.** Because callers are direct imports in
  notebooks and cluster scripts, a break lands hours later in a SLURM log on another
  machine, not in a test.
- **Concretising a traced value in a constructor** to make it JIT-able. Fails silently as a
  zero gradient for every parameter used during setup (`INV-JAX-TRACE-01`); NUTS then never
  moves that parameter off its initial value.
- **Merging two HOD helpers into one class method.** `run_stars_calc` either cannot resolve
  `get_Nsat` or recursively calls `get_Ncen`; galaxy-disabled configurations do not expose
  the break.
- **Importing from `src/arxiv/`.** Produces plausible results from a superseded model with
  no error.
- **Adding a fifth concern to `get_sim_maps.py` or `get_radial_profiles.py`.** These are at
  the size where several failure modes share one file — the same condition that forced
  `multiprobe_namaster.py` to get a dedicated owner.

## Open questions

- This document is derived from `README.md` and `src/context/codebase_summary.md`, not yet
  from line-level reading of each module. `confidence: medium` until anchored to specific
  constructor lines. Owner: `godmax-core`. Not blocking.
- The targeted HOD construction path is covered by `tests/test_get_radial_profiles.py`, but
  no automated test builds the complete class chain from `params_default.yaml` and asserts
  the resulting `C(ell)` shape. That remains the cheapest durable end-to-end coverage.
  Owner: `godmax-core` with `repro-runner`.
