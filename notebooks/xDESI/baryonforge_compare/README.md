# BaryonForge--GODMAX Backlight comparison

This directory contains the reproducible comparison of BaryonForge and GODMAX
on the c9999 Abacus Backlight pz3 cap. The accepted comparison is
`backlight_compare_projection_matched.yaml`. It uses the strict halo selection
`M200c_hMsun > 1e13`, the same buffered catalog rows, halo-only Compton-y and
CMB-convergence maps, RING NSIDE 1024, five-R200c paint support, no pixel or
beam smoothing, and one common analysis mask.

The full selected catalog contains 136,651 rows: 119,068 centers in the inner
600 deg2 cap and 17,583 centers in its one-degree buffer. The completed test
run uses a deterministic, mass-stratified subset of 64 inner-cap halos. It is a
bounded implementation test, not a production measurement of the catalog mass
function or its bandpowers.

## Accepted files

- Comparison contract:
  `backlight_compare_projection_matched.yaml`
- GODMAX parameters:
  `param_files/Pge/params_baryonforge_backlight_godmax_projection_matched.yaml`
- BaryonForge parameters:
  `../BaryonForge/examples/params_baryonforge_backlight_godmax_projection_matched.yaml`
- Output root:
  `data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/`

The historical controls remain separate:

- `backlight_compare.yaml`: native GODMAX 8-R200c normalization and legacy
  projectors.
- `backlight_compare_asymptotic.yaml`: matched 128-R200c normalization but
  legacy projectors.

Native defaults in both repositories are unchanged. The asymptotic GODMAX
class and both matched projectors are selected only by the accepted comparison
files.

## Physical decision: integration limits

The Schneider redistributed-component masses are defined by their asymptotic
mass budget, and hydrostatic pressure uses the boundary condition
`P(r -> infinity)=0`. A finite upper limit is therefore a numerical proxy for
infinity, not a new model parameter. GODMAX's native `0.01--8 R200c` interval
can end inside the gas-ejection scale, especially for low-mass halos.
BaryonForge's `1e-6--100 comoving Mpc` interval is also finite, but is much
closer to the model definition.

The accepted GODMAX-only proxy retains the physical interval
`0.01--128 R200c`, but it no longer enlarges GODMAX's global integration
axis. `num_points_trapz_int` is restored to the native value of 64. The
comparison-only NFW/total-mass normalization, gas normalization, enclosed
mass, and HSE chain use a fixed 64-node Gauss--Legendre rule in log radius,
implemented by
`matched_godmax_profiles.AsymptoticNormalizationProfiles`. The rule has a
static JAX shape and is not fitted or adapted to BaryonForge/GODMAX ratios.

The limit was selected by convergence, not by tuning the code ratio:

| candidate versus 256 R200c / GL512 | gas norm | extended DMO mass | full-chain HSE |
| --- | ---: | ---: | ---: |
| native 8 R200c / trap64 | 2.704 | 1.068e-2 | 4.788e-1 |
| 128 R200c / trap64 | 7.531e-4 | 1.807e-4 | 1.109e-2 |
| rejected 128 R200c / trap256 | 7.582e-4 | 1.123e-5 | 6.780e-4 |
| accepted 128 R200c / GL64 | 7.585e-4 | 2.190e-7 | 4.429e-5 |

The fixed acceptance limit is `1e-3`. Simply reducing the uniform trapezoid
from 256 to 64 fails the HSE gate, while GL64 passes it. The schema-v3,
method-aware convergence artifact rebuilds NFW normalization, total mass,
component fractions and densities, the enclosed-mass table, and HSE pressure
independently in NumPy. It reproduces production GODMAX `Mdmb` and `Ptot` to
`2.66e-15` and `5.11e-15`. Against a separate same-128-R200c continuum
reference, GL64 errors are `2.89e-15` in Mtot, `4.22e-15` in gas
normalization, `6.49e-10` in Mdmb, and `3.94e-5` in HSE pressure. Older
method-blind artifacts fail closed and cannot certify this variant.

The integration-only memory benchmark uses the exact `128 x 48 x 48`, float64
profile grid in fresh CPU processes with JAX preallocation disabled:

| rule | mean peak RSS | compiled full-grid HSE temporary |
| --- | ---: | ---: |
| native trap64 | 1.549 GiB | 0.771 GiB |
| rejected trap256 | 3.885 GiB | 3.092 GiB |
| accepted GL64 | 1.341 GiB | 0.563 GiB |

GL64 adds exactly 1,024 persistent bytes for two 64-element float64
node/weight vectors (`0.00268%` of stored profile-array bytes), reduces mean
peak RSS by `65.47%` relative to trap256, and is `13.39%` below native trap64
in this CPU measurement. This certifies the normalization/HSE integration
change, not absolute CUDA memory for the whole map setup. The separate
`nr=128` radial table and 128-point LOS projector are unchanged and outside
this integration-only benchmark.

## Physical decision: projected profiles

The earlier five-R200c mismatch was numerical rather than a different gas or
matter model.

- The legacy GODMAX path mixed physical transverse radii with a comoving table
  maximum and endpoint-clamped queries beyond physical table support. Its
  maximum error reaches 92.5% for y and 88.0% for kappa on the frozen nodes.
- BaryonForge's native real-space projector omits the near-line-of-sight
  interval. Raising `n_per_decade_proj` does not repair it: the scanned maximum
  errors are 7.18%, 1.95%, and 1.99% for y and 5.66%, 6.43%, and 6.52% for
  kappa at 24, 64, and 128 points per decade.

Both accepted paths instead use the same finite physical operation:

1. integrate along a fixed 100-comoving-Mpc line of sight;
2. remove the origin singularity with `l = R sinh(t)`;
3. use 128-point Gauss--Legendre quadrature in `t`;
4. cover `sqrt(R_perp**2 + l_max**2)` with the 3D table; and
5. return zero outside the declared physical table instead of clamping an
   endpoint.

For GODMAX, the required table support is 67.997961 comoving Mpc/h. The
accepted table uses `rmax=70 Mpc/h`, `nr=128`. Relative to an 80-Mpc/h,
192-point reference, its maximum changes are 0.1174% for y and 0.0780% for
kappa. The 128-point projector differs from a 512-point reference by 0.00824%
for y and 0.00425% for kappa. Dense common-LOS GODMAX and BaryonForge results
agree within 0.834% in y and 0.246% in kappa.

The projection diagnostic intentionally has top-level `ok=false`: no scanned
setting of the native BaryonForge projector satisfies the joint 2% target.
That failure is the registered reason for the opt-in BaryonForge matched
wrapper; it is not a failure of the final matched configuration.

## Parameter crosswalk

All BaryonForge mass pivots are physical Msun, so a GODMAX pivot in Msun/h is
divided by `h=0.6711`. The validator enforces every identity before model
construction.

| GODMAX | BaryonForge Schneider19 | Matched rule |
| --- | --- | --- |
| `H0, Om0, Ob0, sigma8, ns, w0` | `h, Omega_m, Omega_b, sigma8, n_s, w0` | c9999 catalog cosmology |
| `mdef_Delta=200`, `Duffy08` | `200c`, `Duffy08` | same mass and concentration definitions |
| `theta_ej_0`, `nu_theta_ej_M`, `nu_theta_ej_z` | `theta_ej`, `mu_theta_ej`, `nu_theta_ej` | direct coefficients; pivot divided by h |
| `theta_co_0`, `nu_theta_co_M`, `nu_theta_co_z` | `theta_co`, `mu_theta_co`, `nu_theta_co` | direct coefficients; pivot divided by h |
| `log10_Mc0`, `mu_beta`, `nu_z` | `M_c`, `mu_beta`, `nu_M_c` | `M_c=10**log10_Mc0/h` |
| `gamma_rhogas`, `delta_rhogas`, `epsilon_rt` | `gamma`, `delta`, `epsilon` | direct mapping |
| `A_starcga`, `log10_M1_starcga`, `eta_star`, `eta_cga` | `A`, `M1`, `eta=tau`, `eta_delta=tau_delta` | simple stellar branch |
| `backreaction=false` | `a=0` | no-backreaction baseline |
| `alpha_nt`, `beta_nt`, `n_nt` | `alpha_nt`, `nu_nt`, `gamma_nt` | `0.05, 0, 0.3` |
| `0.01--128 R200c`, log-GL64; native core 64 | `1e-6--100 Mpc`, 512 | finite asymptotic mass/HSE proxies |
| `physical_table_cosh`, 128, 100 Mpc LOS | `nonsingular_gauss_legendre`, 128, 100 Mpc LOS | common projection contract |

Every otherwise hidden BaryonForge mass, redshift, and concentration evolution
coefficient is set to zero. The one-halo matter profile is gas + central stars
+ collisionless matter; the globally renormalized `DarkMatterBaryon` composite
and two-halo term are not used. The adapter uses GODMAX's electron-pressure
factor `1/1.932`.

## Unit boundaries

- Catalog `M200c_hMsun` is in Msun/h. It is derived from
  `Interpolated_N * ParticleMassHMsun` and is provisionally treated as M200c.
- Catalog `R200c_hMpc` and `DA_hMpc` are proper/physical Mpc/h; their ratio is
  the GODMAX angular support.
- GODMAX consumes mass in Msun/h, comoving 3D radius in Mpc/h, and physical
  projected transverse radius in Mpc/h.
- BaryonForge/PyCCL consumes physical Msun and comoving Mpc. Like-for-like
  boundaries use `M_BF=M_G/h`, `r_BF=r_G/h`, and `rho_G=rho_BF/h**2`.
- The projected matter profile is converted to physical surface density before
  CMB convergence; y carries the required single scale-factor power.

## Accepted profile evidence

The final nine-node profile artifact compares masses `1e13, 1e14, 1e15 Msun/h`
at redshifts `0.65, 0.80, 0.95` through five R200c.

| quantity | final RMS log ratio | improvement over native 8R + legacy LOS |
| --- | ---: | ---: |
| gas density | 1.649e-4 | 1167.6x |
| total matter density | 2.234e-4 | 9.12x |
| direct y | 1.463e-3 | 145.9x |
| painter-tabulated y | 1.039e-3 | 200.5x |
| painter-tabulated CMB kappa | 1.868e-3 | 24.16x |

The gas-density BaryonForge/GODMAX ratios span `1.0000044--1.0002648`.
All nonprojected BaryonForge arrays are bitwise unchanged from the
integration-only comparison; the projected array changes are explicitly
registered to `physical_table_cosh_100mpc_v1`. Component fractions close
exactly and the independent component-density quadrature passes its unchanged
1% setup gate.

Profile artifacts:

- `profiles/profile_comparison.h5`: SHA-256
  `9fb607e23118b6ec5477ae9f898602c633ef610aee3a6756d58ce5546056cb21`
- `profiles/integration_convergence.h5`: SHA-256
  `480cfc6ece62ae28576da324c54b48dd1b33999bcc1007d9e0d1a1284fd4b103`
- `profiles/integration_change_summary.json`: SHA-256
  `165419bf5ab0c9794090df7c246c847273552252a4c85b86fbe5d3485f4fbfa2`
- `profiles/integration_memory_benchmark.json`: SHA-256
  `a273303a64d88429986321846ff06295ff07a9fb70d0c5c292077bcefb5f5c51`

The projection diagnosis and its four PNG/PDF plots are retained under the
historical asymptotic root's `profiles/` directory. Its HDF5 SHA-256 is
`ab453526fbd5c675ac11a5aa495685d58d2d51850a753a56391d4d294678056d`.

## Completed capped validation

The deterministic 64-halo test repaints both y and CMB kappa at NSIDE 1024.
Both fields have 1,628 nonzero pixels and footprint Jaccard 1.0. Fifty-four
five-R200c apertures are isolated.

| field | Pearson r | relative L1 | BF/GM total sum |
| --- | ---: | ---: | ---: |
| Compton-y | 0.999999923 | 0.00085978 | 1.00081542 |
| CMB kappa | 0.999999963 | 0.00203404 | 0.99796596 |

The aperture points remain tightly clustered around the one-to-one line and
all 13 bounded common-window spectra are finite. Relative to the historical
trap256 repaint, GL64's y relative-L1 is larger by `1.73e-4` and kappa by
`6.23e-6`. This is recorded rather than treated as an improvement: the
independent continuum test shows GL64 is the more accurate integral, so the
slightly closer old y map was numerical cancellation against the remaining
backend difference.

The smoke root contains the catalog, both maps, diagnostics, logs, eight
standard PNG/PDF plot pairs, and checksum manifests:

`validation/smoke64/`

Key SHA-256 values are:

- GODMAX map: `798862c126b2d92fcd717c16db281d7a62212b02ca101c71f062dad962c9e6dd`
- BaryonForge map: `9b1f1af60331de50acf493c0b030bb0214bc31e41b22145309e911664f7eca18`
- diagnostics: `8de7c874fda75b412729b18df316088e8effe0dd94ed195b0dc74f1043691484`
- plot manifest: `650e5a503624afa7c3bb34e8179abb46e3ccd2be656ab497588cc45bfe2a426d`

Every plot says “64-halo bounded smoke; not production statistics.” A direct
negative test of `measure_statistics.py` rejects the maps for `max_halos=64`,
incomplete-catalog provenance, and mismatch with the production catalog. It
creates no statistics output.

## Reproduction order

Run from the GODMAX repository root with the environment providing JAX,
PyCCL, healpy, and NaMaster.

1. Validate the full parameter/unit/catalog crosswalk:

   ```bash
   python notebooks/xDESI/baryonforge_compare/validate_config.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml
   ```

2. Materialize the shared catalog if it is not already present, then audit
   native angular support:

   ```bash
   python notebooks/xDESI/baryonforge_compare/prepare_catalog.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml
   python notebooks/xDESI/baryonforge_compare/audit_native_support.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --nside 1024 --n-jobs 8
   ```

3. Regenerate the profile, full-chain convergence, and bound summary:

   ```bash
   python notebooks/xDESI/baryonforge_compare/compare_profiles.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --overwrite
   python notebooks/xDESI/baryonforge_compare/check_integration_convergence.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml
   python notebooks/xDESI/baryonforge_compare/summarize_integration_change.py \
     --old data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600/profiles/profile_comparison.h5 \
     --new data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/profiles/profile_comparison.h5 \
     --convergence data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/profiles/integration_convergence.h5 \
     --output-json data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/profiles/integration_change_summary.json \
     --figure-dir data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/profiles/figures \
     --allow-registered-projection-change --overwrite
   ```

4. Reproduce the fresh-process integration-memory benchmark and its plot:

   ```bash
   python notebooks/xDESI/baryonforge_compare/benchmark_integration_memory.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --repeats 2
   ```

5. Reproduce the saved bounded run and all plots:

   ```bash
   python notebooks/xDESI/baryonforge_compare/run_bounded_validation.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --overwrite
   ```

   Use `--skip-paint --overwrite` to re-read existing maps and refresh only
   diagnostics and standard plots.

6. Inspect the full GODMAX target without launching work:

   ```bash
   python notebooks/xDESI/baryonforge_compare/paint_godmax.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --dry-run
   ```

7. Only inside appropriate allocations, run the full painters:

   ```bash
   python notebooks/xDESI/baryonforge_compare/paint_godmax.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --pixel-workers 1
   python notebooks/xDESI/baryonforge_compare/paint_baryonforge.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --n-jobs 8
   ```

8. Once both complete 136,651-halo maps exist, run the production statistics:

   ```bash
   python notebooks/xDESI/baryonforge_compare/measure_statistics.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --godmax-maps "$GODMAX_MAP" \
     --baryonforge-maps data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/maps/baryonforge_native_nside1024.h5 \
     --output data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/measurements/common_mask_statistics.h5
   ```

The statistics artifact contains yy, kappa-kappa, and y-kappa spectra for both
backends; same-field cross-backend spectra; both mixed-backend y-kappa orders;
residual spectra; transfer, amplitude, coherence, gain, and closure checks. A
complete one-split GODMAX paint is currently required because no
provenance-preserving split combiner is part of this comparison.

9. Plot every production C-ell and both map fields with provenance checks:

   ```bash
   python notebooks/xDESI/baryonforge_compare/plot_results.py \
     --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml \
     --statistics data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/measurements/common_mask_statistics.h5 \
     --godmax-maps "$GODMAX_MAP" \
     --baryonforge-maps data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/maps/baryonforge_native_nside1024.h5 \
     --output-dir data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/plots
   ```

   The driver writes `01_all_cls`, `02_cl_diagnostics`, `03_tsz_maps`, and
   `04_cmb_lensing_maps` as PNG/PDF pairs plus `plot_manifest.json`. It plots
   all 13 decoupled C-ell arrays and does not attach error bars because the
   deterministic backend comparison has no covariance product.

## 2400-deg2 production extension

`backlight_compare_projection_matched_cap2400.yaml` keeps every accepted
profile, memory, paint, and statistics setting above while changing only the
cap/catalog/output identity. The exact strict selection contains 504,174
buffered halos: 470,298 centers in the 2400-deg2 core and 33,876 in the
one-degree shell. The core is centered at RA 37.96875 deg, Dec
-34.953865257188454 deg, with radius 27.91480167872345 deg.

The fail-closed Slurm chain is:

```text
6754167 strict catalog
  +-- afterok -> 6754168 GODMAX H100 map
  +-- afterok -> 6754169 BaryonForge CPU map
                         +-- both afterok -> 6754171 statistics and plots
```

At submission time the jobs were pending scheduler priority/dependencies.
Outputs will live under
`data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap2400_projection_matched/`.
The final job writes the same 13-spectrum HDF5 contract and all eight plot
files; no 2400-deg2 result is valid unless its dependency and hash gates pass.

## Current boundary

Profile, capped-map, and complete-production gates pass. Final jobs 6753705
(BaryonForge map), 6753706 (GODMAX map), and 6753707 (paired statistics) all
completed successfully. The map products each paint all 136,651 halos and the
statistics product contains 13 finite ten-bin spectra plus shared-mask,
transfer, coherence, gain, and residual-closure diagnostics.

Generated k/ell/delta-ell setup grids are canonicalized to 13 significant
digits after their final arithmetic operation. This removes the observed
Rome-versus-H100-host last-bit provenance drift without changing either
production y or kappa array: all four old-versus-final map comparisons are
bitwise equal. The scheme and affected paths are embedded in the effective
config manifest; exact config, parameter, source, runtime, and catalog hashes
remain fail-closed.

The statistics artifact remains spectra-only: radial stacks and covariance are
not included. Full-chain HSE convergence is sampled at nine mass-redshift nodes
and five radii rather than asserted continuously. Absolute GPU allocator peak
for the complete setup was not measured. Runtime provenance records imported
PyYAML 6.0.2 versus distribution metadata 6.0.1 and imported SciPy 1.14.1
versus metadata 1.12.0; reproduce from recorded imports and module hashes.

The completed 600-deg2 plot manifest has SHA-256
`0ee219387717380761b883fc90594d89a314a5a5171588f3caf5ed8773f6dc5a`.
The 2400-deg2 chain is submitted as jobs 6754167, 6754168, 6754169, and
6754171, but was not yet complete at the recorded boundary.
