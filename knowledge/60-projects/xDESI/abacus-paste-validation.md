---
id: kb.xdesi.abacus-paste
title: Abacus Backlight paste validation — configuration and null-test logic
layer: 60-projects
owner: abacus-paste-validator
status: verified
confidence: medium
scope:
  - notebooks/xDESI/abacus_pasting_config.yaml
  - notebooks/xDESI/abacus_paste/
  - notebooks/xDESI/abacus_pasting_helpers.py
  - notebooks/xDESI/preprocess_abacus_lightcone_halos.py
  - notebooks/xDESI/paste_abacus_maps.py
  - notebooks/xDESI/combine_abacus_map_splits.py
  - src/get_sim_maps.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-JAX-SEED-01
  - INV-HOD-PZBIN-01
checks:
  - python tools/kb/kb.py invariants --check --id INV-ABACUS-COSMO-01
  - python tools/kb/kb.py invariants --check --id INV-JAX-SEED-01
verified_at_commit: a3b3f96
verified_on: 2026-08-16
see_also: [kb.xdesi.analysis-state, kb.measurement.multiprobe-product]
scope_digest: sha256:c22f447ec0e8d5ba79035eeb5af2369f
---

## Claim

Pasting Abacus Backlight lightcone halos and comparing the measured maps to analytic GODMAX
theory is a **null test of the pipeline**, not a cosmology fit. It is only interpretable when
the theory uses the simulation's own cosmology and the explicitly selected theory component
uses the same halo mass cut as the pasted catalog.

## Why it is true

`notebooks/xDESI/abacus_pasting_config.yaml` is the canonical configuration.

**The null-test conditions** (`INV-ABACUS-COSMO-01`):

```yaml
godmax:
  override_cosmology_from_catalog: true
  theory_mass_cut_applied_to_hod: true
  use_fit_lens_nz: true
```

The cosmology flag is consumed by both GODMAX setup paths in
`abacus_pasting_helpers.py:388-389,459-460`. The mass-cut flag is currently a declarative
configuration marker: no Python runtime reads `theory_mass_cut_applied_to_hod`. Stage-31
enforces the actual comparison contract elsewhere. `build_theory` creates both a full-HOD
curve and a resolved curve; the latter masks `Profiles.Ncen_mat` and `Profiles.Nsat_mat`
below `godmax.resolved_catalog_log10_m_min_hmsun`
(`stage31_pz1_backlight_validation.py:1867-1872,2013-2037`) and the plotting command defaults
to `--theory-component resolved` (`stage31_pz1_backlight_validation.py:5477`). A consumer
must therefore select and provenance the resolved component; merely seeing the YAML boolean
is not evidence that a mass cut was applied.

With mismatched cosmology or an unmatched mass cut, a discrepancy cannot be attributed to
the paste pipeline.

**Simulation inputs** — `AbacusBacklight_base_c9999_ph9999` lightcone halos under
`/mnt/ceph/users/backlight/`, with **`read_only: true`**. Two catalog selections are
defined: `zlt1p0_logMgt12p5` (z < 1.0, log10 M > 12.5) and `zlt0p5_logMgt14p0`.

**Pasting settings:**

```yaml
pasting:
  nside: 1024
  max_paint_R200c_factor: 5.0
  smooth_profiles: true
  random_seed: 42
  pixel_batch_size: 2000
  num_splits: 4
  chunk_halos_by_nside: {64: 1000000, ..., 1024: 50000, 2048: 10000}
```

Maps requested: galaxy, y, kSZ, tau, CMB kappa, WL kappa, baryonified.

`max_paint_R200c_factor: 5.0` is a **physics choice**, not a performance knob: truncating the
profile changes the 1-halo term and therefore the high-ell power.

`random_seed: 42` is consumed, but the provenance contract is narrower than the old wording
implied. Each paste chunk receives
`base_seed + 100000 * split_index + chunk_id`
(`abacus_pasting_helpers.py:2400`), and `get_sim_maps.py:975-979` constructs and splits that
key. Centrals use Bernoulli sampling and satellites use Poisson sampling
(`get_sim_maps.py:1320-1335`). This is reproducible only for fixed catalog order, split
layout, chunking and configuration. The partial/final map HDF5 attributes do not currently
embed the base or derived seed (`abacus_pasting_helpers.py:2514-2556`); the timing sidecar
records the configuration path, not a frozen copy of its contents. Therefore
the explicit-key part of `INV-JAX-SEED-01` holds, but its requirement to record the seed in
output metadata is not yet satisfied by the partial or combined map HDF5 product.

**Theory inputs** — `params_default.yaml` merged with
`param_files/xDESI/params_fit_abacus.yaml`; source n(z) from the LSST Y1 forecast FITS
(`nz_source`, `BIN5`, floor 1e-4).

**Photometric-bin structure for map work** (`INV-HOD-PZBIN-01`). The four DESI photometric
bins overlap in true z, so map-level galaxy tracers must reproduce four overlapping kernels.
Two defensible implementations are:

1. **Catalog-like** — assign simulated galaxies to photometric bins probabilistically so each
   bin recovers the calibrated true n(z) and the measured angular density.
2. **Map/paste** — build four projected galaxy tracer maps from the four true-n(z) kernels
   with their per-pz HOD parameters.

The current Stage-31 implementation uses (2): every selected configuration declares one
`pasting.pz_bin`; `prepare_stage31_godmax_config` calls
`config_for_single_desi_pz` before model construction and records
`analysis.single_photometric_pz_bin` (`abacus_pasting_helpers.py:452-484`). Separate pz
configurations therefore create separate HOD realizations/theory blocks. The measurement
path explicitly records that no simulated photo-z assignment or photo-z cut is applied
(`stage31_pz1_backlight_validation.py:358-361,1272`). The pz label selects that bin's
calibrated true-n(z)/nbar and HOD block; it is not a disjoint true-redshift catalog cut.

**The input parameter point.**
`param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v1.yaml` is an
operational v1 pasting input whose fit has already been documented as unacceptable in
`kb.xdesi.analysis-state`. This audit did not re-run that fit. It is **not** a measurement of
gas or HOD physics.

**Scaling infrastructure.** `notebooks/xDESI/abacus_paste/` holds a deliberate capacity
ladder — `cap600` → `cap2400` → `cap4800` → fullsky — plus
`benchmark_stage31_fullsky_pixel_sample.py`, `abacus_paste_scaling_summary.py`,
`collect_scaling_job_status.py`, and `estimate_stage31_fullsky_counts.py`. Configs are
generated by `make_stage31_*.py` into `*.selected.yaml`; the encoded filenames
(`mmin11p147538`, `nside2048_lmax4096`, `cap2400`, `hmcbestfit` vs `hmcfailed`) are the
provenance record. SLURM: `gpu` partition, `a100-80gb`, `ili-sbi` conda env, 256 GB, 12 h
paste wall time.

`hmcbestfit` and `hmcfailed` are input-provenance labels, not goodness certificates. Check
which one produced a result before quoting it, and do not infer physical acceptability from
the filename.

**New opt-in projection/provenance paths.** The standard xDESI paste remains on the
`legacy_log_radius` projector by default. `src/get_sim_maps.py:61-103,418-484` now also
supports the unit-consistent `physical_table_cosh` projector, and
`run_paste_split(..., profiles_class_path=...)` can inject a validated `Profiles` subclass
while recording its fully-qualified class name in the timing and HDF5 products
(`abacus_pasting_helpers.py:1937,2078-2097,2261,2534`). These are opt-in comparison paths;
their existence does not change the canonical xDESI configuration. Analytically generated
k/ell setup grids are also canonicalized to 13 significant digits before use and hashing
(`abacus_pasting_helpers.py:83-142,402-409`) to suppress cross-architecture last-bit drift;
catalog values and profile values are not canonicalized.

## How to verify

```bash
git grep -n -E "override_cosmology_from_catalog|theory_mass_cut_applied_to_hod" -- notebooks/xDESI param_files
git grep -n -E "random_seed|max_paint_R200c_factor|read_only" -- notebooks/xDESI/abacus_pasting_config.yaml notebooks/xDESI/abacus_pasting_helpers.py src/get_sim_maps.py
git grep -n -E "config_for_single_desi_pz|single_photometric_pz_bin" -- notebooks/xDESI/abacus_pasting_helpers.py notebooks/xDESI/abacus_paste
git ls-files 'notebooks/xDESI/abacus_paste/*.selected.yaml'
```

## Failure modes

- **Cosmology or mass-cut mismatch.** A constant offset between pasted and analytic spectra
  that scales with the mass cut and does not vanish at large scales — and cannot be
  attributed to any stage.
- **Treating a declarative flag as enforcement.** The current mass-cut YAML boolean has no
  runtime consumer. The saved/selected theory component must be the explicitly masked
  resolved component.
- **Profile truncation too aggressive.** High-ell power deficit that moves when
  `max_paint_R200c_factor` changes. If it moves, it is numerical, not physical.
- **Unrecorded seed.** A null test that cannot be reproduced run to run, then dismissed as
  noise. The current final map product does not carry a self-contained seed/config record.
- **Disjoint-bin galaxy maps.** Double-counts galaxies in the true-z overlap regions and
  starves the tails, so each tracer is a blend of neighbouring bins.
- **Writing into the Abacus tree.** Corrupts a shared read-only input for every user.
- **Full-sky submission without a cap-based estimate.** Expensive, hard to recall, and the
  scaling ladder exists specifically to avoid it.
- **Conflating `hmcbestfit` and `hmcfailed` products.**

## Open questions

- Make the mass-cut contract fail closed: either consume
  `theory_mass_cut_applied_to_hod` or replace it with an explicit required theory-component
  choice and assert that its numeric cut equals the catalog cut. Owner:
  `abacus-paste-validator`. **Blocking** any comparison whose chosen theory component is not
  independently identified.
- Embed the base seed, derived-seed rule, configuration digest, split layout and chunk size
  in partial and combined HDF5 metadata. Until then, reproducing the HOD realization depends
  on a mutable external config path. Owner: `abacus-paste-validator` with `jax-numerics`.
- `notebooks/xDESI/abacus_paste/galaxy_clustering_excess_diagnosis.ipynb` suggests a known
  galaxy clustering excess in the pasted maps. Its status and relation to the dominant
  `desi_g_auto` survey-fit misfit documented in `kb.xdesi.analysis-state` is unresolved and
  worth pursuing — the two may share a cause. Owner: `abacus-paste-validator` with
  `xdesi-lead`.
