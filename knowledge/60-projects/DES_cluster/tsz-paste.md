---
id: kb.des-cluster.tsz-paste
title: DES cluster tSZ-only halo pasting
layer: 60-projects
owner: abacus-paste-validator
status: verified
confidence: medium
scope:
  - notebooks/DES_cluster/tsz_halo_pasting.ipynb
  - notebooks/DES_cluster/tsz_pasting.py
  - notebooks/DES_cluster/params_tsz.yaml
  - notebooks/DES_cluster/params_tsz_zmax0p85.yaml
  - notebooks/DES_cluster/params_tsz_zmax0p85_thetaejx1p5.yaml
  - notebooks/DES_cluster/benchmark_tsz_zmax.py
  - notebooks/DES_cluster/run_tsz_job.py
  - notebooks/DES_cluster/submit_tsz_zmax0p85_nside2048.sbatch
  - notebooks/DES_cluster/submit_tsz_zmax0p85_nside2048_thetaejx1p5.sbatch
  - notebooks/DES_cluster/test_tsz_pasting.py
  - notebooks/DES_cluster/validate_tsz_pasting.py
  - notebooks/DES_cluster/plot_tsz_halo_correlations.py
  - notebooks/DES_cluster/plot_tsz_halo_correlations.ipynb
  - notebooks/DES_cluster/test_tsz_correlations.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-PHYS-UNITS-01
  - INV-PHYS-MASSBUDGET-01
  - INV-JAX-X64-01
  - INV-PRODUCT-PROV-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q notebooks/DES_cluster/test_tsz_pasting.py"
  - "/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q notebooks/DES_cluster/test_tsz_correlations.py"
  - "/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/DES_cluster/validate_tsz_pasting.py --stage preflight"
verified_at_commit: cf72943
verified_on: 2026-08-10
see_also: []
supersedes: []
scope_digest: sha256:c005e4e47e5b2f2a75e8b8753966e8d2
---

## Claim

`notebooks/DES_cluster/tsz_pasting.py` produces one bounded-memory HEALPix RING
Compton-y map from every configured halo passing the strict mass cut, using the
c000 observer/cosmology, the asymptotic GODMAX pressure normalization, and the
unit-consistent physical LOS projector.  The result is explicitly a halo-only
one-halo product conditional on treating `M_interp` as `M200c` in Msun/h; the
source HDF5 does not establish that SO mass definition, so the absolute tSZ
amplitude remains provisional.

An optional inclusive `catalog.selection.redshift_max` is applied at the same
streamed boundary.  The z<=0.85 override expects 1,299,336 rows.  Preflight and
painting independently hash the ordered source-row indices and a complete run
fails if those identities differ.

`plot_tsz_halo_correlations.ipynb` is an executed, self-checking diagnostic for
that z<=0.85 NSIDE-2048 map.  It displays the RING Compton-y map and saves raw
masked Healpy pseudo-spectra for yy and pasted-halo-overdensity x y.  These are
angular power spectra under a sharp first-octant mask, not configuration-space
correlations or survey-decoupled bandpowers.

## Why it is true

The YAML fixes the AbacusSummit c000 observer at `[-990,-990,-990]` comoving
Mpc/h and records the exact c000 total-matter cosmology
(`notebooks/DES_cluster/params_tsz.yaml:22-38`).  Configuration validation
rejects a different origin, cosmology, normalization class, legacy projector,
non-RING ordering, negative pressure amplitude, smoothing, beam, or noise
(`notebooks/DES_cluster/tsz_pasting.py:109-210`).

The adapter forms `d=XYZ-observer`, derives RA/Dec from `d`, uses
`DA=|d|/(1+z)`, and derives proper `R200c` from the provisional mass proxy and
critical density (`notebooks/DES_cluster/tsz_pasting.py:282-331`).  The streamed
preflight verifies the filtered product's completeness metadata, field names,
finite values, the strict cut,
observer-relative distance against redshift, and mass/redshift interpolation
coverage before profile construction (`notebooks/DES_cluster/tsz_pasting.py:346-456`).

The shared selection mask combines the strict mass predicate with the optional
inclusive redshift maximum.  The already-filtered fast path may bypass only
the proven mass comparison; it always applies redshift selection.  The merged
configuration and every YAML source/hash are stored with the map.

The production path selects `AsymptoticNormalizationProfiles`, enables only
tSZ, checks the pressure and projected-y tables before their log interpolator,
and builds one setup object (`notebooks/DES_cluster/tsz_pasting.py:586-668`).  A
single fixed-shape JIT evaluator is padded only beyond the returned pair count
and reused for every chunk (`notebooks/DES_cluster/tsz_pasting.py:671-742`).
The reference checker evaluates the identical pair package through this thin
path and the established `get_sim_map` path
(`notebooks/DES_cluster/tsz_pasting.py:762-889`).

Before constructing the y interpolator, the helper extends only the projected
radius grid from the native `0.003514862` down to `1e-5` Mpc/h and evaluates
those added nodes with GODMAX's same `physical_table_cosh` projector.  The
evaluator rejects lower or upper radial extrapolation.  The filtered product's
nearest NSIDE 2048 center is `1.26937e-4` Mpc/h, so no catalog query is outside
the extended grid (command and output in the evidence ledger).

Only `/maps/map_ymap` is written.  Its attributes record selection, units,
observer and coordinate formulae, provisional mass semantics, cosmology,
profile/projector variants, grid, runtime, hashes of the catalog/config/helper/
GODMAX sources, and git state (`notebooks/DES_cluster/tsz_pasting.py:940-1088`).
The production driver then scans the saved map and atomically writes a sibling
`.validated.json` success marker containing the HDF5 SHA256 only after the
schema, count, selected-row identity, redshift, finite/nonnegative, projected-
grid, JAX backend, and x64 checks pass.
The final pressure amplitude is applied exactly once after map assembly; zero
amplitude never enters the logarithmic interpolator
(`notebooks/DES_cluster/tsz_pasting.py:1126-1193`).

The correlation helper refuses an HDF5 without the expected validation-marker
schema/status, matching marker/map SHA256, single-map schema, RING ordering,
exact halo count, and exact ordered-row digest.  The compact spectra product
also verifies and records the current catalog SHA256 against the production
map, alongside the map and marker hashes.  The helper independently streams
the configured strict-mass/inclusive-redshift
selection and constructs an int32 halo count map.  The catalog occupies only
the first octant, so both scalar fields use a common inclusive spherical-
triangle footprint; each footprint mean is removed and both fields are exactly
zero outside.  The notebook then calls `healpy.map2alm` sequentially with ring
weights, `lmax=4096`, and `iter=0`, followed by `alm2cl`.

The saved spectra are deliberately labelled raw masked mode-coupled pseudo-Cl:
there is no mode decoupling, fsky rescaling, pixel-window deconvolution, beam,
noise, or halo shot-noise subtraction.  The final execution recovered all
1,299,336 selected rows and the production digest, left the input map hash
unchanged, put every halo inside the footprint, and passed the independently
pre-registered `lmax=2048` transform control.  Exact commands and literal
outputs are in the dated correlation evidence ledger.

## How to verify

```bash
# Unit/schema/null/determinism/radial-domain/selection/publication checks:
# expected "10 passed".
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  notebooks/DES_cluster/test_tsz_pasting.py

# Filtered-product preflight: expected selected_rows=3,001,721 and
# max_distance_redshift_relative_error < 0.01.
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -c \
  "import sys; sys.path.insert(0,'notebooks/DES_cluster'); import tsz_pasting as t; c=t.load_params('notebooks/DES_cluster/params_tsz.yaml'); print(t.preflight_catalog(c))"

# Notebook JSON and source syntax: expected "notebook syntax PASS".
jq empty notebooks/DES_cluster/tsz_halo_pasting.ipynb

# Requested production selection: expected selected_rows=1,299,336 and z_max<=0.85.
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -c \
  "import sys; sys.path.insert(0,'notebooks/DES_cluster'); import tsz_pasting as t; c=t.load_params('notebooks/DES_cluster/params_tsz_zmax0p85.yaml'); print(t.preflight_catalog(c))"

# Correlation marker/schema/mask/binning checks: expected "4 passed".
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest -q \
  notebooks/DES_cluster/test_tsz_correlations.py

# The committed notebook is already executed.  To reproduce the figures,
# select the ili-sbi kernel and run all cells from a clean kernel:
# notebooks/DES_cluster/plot_tsz_halo_correlations.ipynb
```

## Failure modes

- Using raw `(X,Y,Z)` from origin zero gives incorrect RA/Dec while still
  yielding finite maps; the radial redshift preflight rejects this frame.
- Reusing the historical c9999 cosmology or z=0.60--1.02 comparison grid gives
  plausible but extrapolated values; exact-cosmology and grid-domain guards
  reject both.
- Importing JAX first with x64 disabled can alter profile numerics; the helper
  requests a kernel restart instead of changing precision after array creation.
- Sending zero pressure through the log interpolator produces `exp(-20)` rather
  than zero; the explicit amplitude-zero branch writes an exact null.
- Leaving the native projected minimum unchanged sends 467 real central pixel
  samples to the same `exp(-20)` sentinel.  Added projected nodes cover all of
  them; both lower and upper extrapolation now raise.
- Generic multi-probe allocation retains several unused full-sky maps and
  rebuilds bound JIT evaluators per halo chunk; this driver owns only one map
  and one evaluator.
- Interpreting `M_interp` as a proven SO mass would turn a conditional result
  into an unsupported absolute-amplitude claim; provenance keeps the mass
  definition provisional.
- Passing the z-limited count as `max_halos` paints an incorrect row prefix;
  the z<=0.85 config instead filters the complete HDF5 stream and verifies the
  selected/painted row-index digests.
- Treating the first-octant lightcone as a full-sky halo sample assigns
  `delta_h=-1` to the unobserved seven-eighths and dominates low multipoles.
  The diagnostic instead uses an explicit common octant footprint and reports
  uncorrected masked pseudo-spectra.
- Calling the plotted curves survey bandpowers, mask-decoupled spectra, or pure
  one-halo terms is incorrect.  Individual pasted components are one-halo
  profiles, while the realized map also contains distinct-halo clustering.

## Open questions

- The catalog producer must confirm the exact SO definition of `M_interp` before
  absolute tSZ amplitudes can be called fully physical.  This does not block
  generating the explicitly provisional map.
- A full 3,001,721-halo NSIDE 2048 all-z run has not been launched.  Only bounded
  validation runs were authorized in this session; the notebook is the launch
  surface.
- The 1,299,336-halo z<=0.85 NSIDE 2048 batch is configured for the explicitly
  approved one node, 40 CPUs, 256 GB RAM, and three-hour request.  Job 6786031
  completed successfully and its independently checked HDF5/validation-marker
  SHA256 values match; the dated evidence ledger records the literal commands.
- A separate sensitivity map with only `theta_ej_0` changed from 2.0 to 3.0
  completed as job 6804773.  It preserves the exact selected-row digest and
  baseline products; its config/map/marker hashes and old-versus-new map
  comparison are recorded in
  `knowledge/.kb/ledgers/2026-08-10-des-cluster-thetaej-x1p5.md`.
- The underlying 3D pressure table still starts at `halo_params.rmin=0.003`
  comoving Mpc/h and clamps its still smaller-radius pressure to that endpoint.
  The projected 64-versus-96-node central grid is converged on all 467 affected
  halo centers, but no lower-`rmin` 3D pressure-grid convergence was run.  Treat
  sub-3-kpc core samples as a residual small-scale approximation.
