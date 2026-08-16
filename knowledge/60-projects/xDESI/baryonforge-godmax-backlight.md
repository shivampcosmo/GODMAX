---
id: kb.xdesi.baryonforge-godmax-backlight
title: BaryonForge versus GODMAX Backlight profile and map comparison
layer: 60-projects
owner: abacus-paste-validator
status: verified
confidence: high
scope:
  - notebooks/xDESI/baryonforge_compare/
  - notebooks/xDESI/abacus_pasting_helpers.py
  - param_files/Pge/params_baryonforge_backlight_godmax_projection_matched.yaml
  - param_files/params_default.yaml
  - src/get_sim_maps.py
  - tests/test_baryonforge_backlight_compare.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-PHYS-MASSBUDGET-01
  - INV-PHYS-UNITS-01
  - INV-PRODUCT-PROV-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/baryonforge_compare/validate_config.py --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/baryonforge_compare/check_integration_convergence.py --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/baryonforge_compare/benchmark_integration_memory.py --config notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml --repeats 2
  - /usr/bin/env JAX_PLATFORMS=cpu /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest tests/test_baryonforge_backlight_compare.py tests/test_get_radial_profiles.py -q
verified_at_commit: cf72943
verified_on: 2026-08-04
see_also: [kb.xdesi.abacus-paste, kb.physics.halo-model-ingredients]
supersedes: []
scope_digest: sha256:e734ac23d13ef664e12bc7b30c95dab5
---

## Claim

The accepted BaryonForge--GODMAX Backlight comparison uses the same strict
`M200c_hMsun > 1e13` halo rows, catalog cosmology, M200c definition, Duffy08
concentration, analytic Schneider coefficients, five-R200c sky support, RING
NSIDE 1024 geometry, and smoothing policy. It also makes the previously hidden
normalization/HSE and line-of-sight numerical contracts explicit.

The accepted comparison is
`notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml`.
It selects:

- GODMAX `0.01--128 R200c` with a fixed 64-node log-radius
  Gauss--Legendre rule for the affected mass-normalization -> component ->
  enclosed-mass -> HSE chain, while the global/native integration width stays
  at 64;
- a common fixed 100-comoving-Mpc LOS;
- nonsingular `l=R sinh(t)` Gauss--Legendre projection with 128 points in each
  backend; and
- a GODMAX 3D table with `rmax=70 Mpc/h`, `nr=128`, sufficient to cover the
  full LOS radius without endpoint extrapolation.

These are opt-in comparison numerics. Native GODMAX and BaryonForge defaults
remain unchanged. The comparison is verified for profiles, a deterministic
64-halo capped run, complete 136,651-halo production maps, and paired
catalog-representative bandpowers.

## Why the integration change is physical

The Schneider redistributed-component budget is asymptotic and the HSE
solution uses `P(r -> infinity)=0`. Therefore a finite upper bound represents
infinity numerically; it is not an analytic baryonic parameter to fit. Native
GODMAX integrates over `0.01--8 R200c`, which can terminate inside the ejected
gas distribution. BaryonForge's `1e-6--100 comoving Mpc` domain is finite too,
but is closer to the asymptotic definition.

The limit and rule were chosen with the pre-existing `1e-3` convergence
threshold. Restoring a uniform 128-R200c trapezoid directly to 64 points fails:
its full-chain HSE error is `1.1085e-2`. The selected 128-R200c/GL64 rule,
compared with a 256-R200c/GL512 reference, changes:

- gas normalization by `7.5848e-4`;
- extended DMO mass by `2.1900e-7`; and
- independently rebuilt full-chain HSE pressure by `4.4286e-5`.

The schema-v3, method-aware artifact independently reconstructs NFW normalization,
Mtot, stellar/gas/collisionless fractions, component densities, the radial
Mdmb table, and HSE pressure in NumPy. It reproduces the production GODMAX
Mdmb and Ptot arrays at `2.66e-15` and `5.11e-15`. A separate same-boundary
continuum reference gives GL64 errors of `2.89e-15` in Mtot, `4.22e-15` in gas
normalization, `6.49e-10` in Mdmb, and `3.94e-5` in HSE. Older method-blind
diagnostics fail closed.

The fixed rule keeps JAX shapes static and is not tuned to backend ratios. On
the exact `128 x 48 x 48` float64 profile grid, fresh-process mean peak RSS is
1,662,996,480 bytes for native trap64, 4,171,300,864 bytes for the rejected
trap256 path, and 1,440,247,808 bytes for GL64. GL64 stores only 1,024 extra
bytes for two 64-element node/weight arrays and reduces peak RSS by 65.47%
relative to trap256. This is an integration-only CPU result: the separate
`nr=128` table, 128-point LOS projector, and absolute CUDA allocator peak are
outside its scope.

## Why the projection change is physical

The analytic profiles were not responsible for the former sharp five-R200c
projected discrepancy.

1. The legacy GODMAX path supplied physical transverse radii but interpreted
   the table maximum as comoving and clamped beyond physical support. Its
   maximum frozen-node errors reach 92.54% for y and 88.00% for CMB kappa.
2. The native BaryonForge real-space projector omits the near-LOS interval.
   Increasing `n_per_decade_proj` from 24 to 64 or 128 does not converge the
   kappa result; the maximum errors are 5.66%, 6.43%, and 6.52%.
3. The matched paths evaluate the same finite operator: 100 comoving Mpc,
   `l=R sinh(t)`, 128-point Gauss--Legendre quadrature, physical table support,
   and no endpoint extrapolation.

The largest required GODMAX support is 67.997961 comoving Mpc/h. The selected
70-Mpc/h/128-node table differs from 80 Mpc/h/192 nodes by at most 0.1174% in
y and 0.0780% in kappa. The 128-point quadrature differs from 512 points by at
most 0.00824% and 0.00425%. Dense common-LOS GODMAX/BaryonForge profiles agree
within 0.834% in y and 0.246% in kappa.

The standalone projection diagnostic records `ok=false` because it requires a
passing *native* BaryonForge sampling choice and none exists. This is the
falsifier that authorizes the explicitly registered matched wrapper, not a
failure of the accepted final profile artifact.

## Parameter and unit contract

GODMAX masses and pivots are in Msun/h; BaryonForge/PyCCL uses physical Msun.
Each BaryonForge mass pivot is the GODMAX value divided by the catalog
`h=0.6711`. GODMAX's 3D radius is comoving Mpc/h and projected transverse
radius is physical Mpc/h. BaryonForge consumes comoving Mpc. Like-for-like
boundaries use `M_BF=M_G/h`, `r_BF=r_G/h`, and `rho_G=rho_BF/h**2`.

The catalog's `R200c_hMpc` and `DA_hMpc` are both proper/physical Mpc/h and set
the angular support. The catalog mass is
`Interpolated_N * ParticleMassHMsun`, provisionally treated as M200c; this
proxy label is propagated to all products.

The simple stellar branch, no backreaction (`false` versus `a=0`), no two-halo
term, zero hidden mass/redshift/concentration evolution, and radial
nonthermal term `(alpha,beta/growth,gamma)=(0.05,0,0.3)` are matched. The
matter profile is gas + central stars + collisionless matter. The adapter uses
the GODMAX electron-pressure factor `1/1.932` and applies the required
scale-factor conventions separately for matter and y.

## Evidence

The selected catalog contains 136,651 unique buffered rows. The nine profile
nodes use masses `1e13, 1e14, 1e15 Msun/h`, redshifts `0.65, 0.80, 0.95`, and
radii through five R200c.

Final BF/GM gas ratios span `1.0000078--1.0002729`. RMS log-ratio improvements
relative to native 8R plus legacy projection are:

| quantity | final RMS log ratio | improvement |
| --- | ---: | ---: |
| gas | 1.6494e-4 | 1167.61x |
| matter | 2.2337e-4 | 9.12x |
| direct y | 1.4628e-3 | 145.90x |
| painter-tabulated y | 1.0393e-3 | 200.51x |
| painter-tabulated kappa | 1.8678e-3 | 24.16x |

All nonprojected BaryonForge arrays remain bitwise unchanged; projected
changes are bound to the registered `physical_table_cosh_100mpc_v1` variant.

The deterministic smoke run contains 64 mass-stratified inner-cap halos, 54
isolated five-R200c apertures, and 1,628 nonzero pixels in both fields with
footprint Jaccard 1.0:

| field | Pearson | relative L1 | BF/GM sum |
| --- | ---: | ---: | ---: |
| y | 0.999999923 | 0.00085978 | 1.00081542 |
| CMB kappa | 0.999999963 | 0.00203404 | 0.99796596 |

The aperture points remain close to one-to-one, all 13 bounded common-window
spectra are finite, and eight PNG/PDF plot pairs were visually audited. The
historical trap256 repaint had y/kappa relative-L1 values `0.00068698` and
`0.00202781`; GL64 is larger by `1.73e-4` and `6.23e-6`. The old quadrature
therefore benefited from slight numerical cancellation against the residual
backend difference; the independent continuum reference, rather than map
ratio tuning, selects GL64.

### Complete-catalog production

The final production jobs completed successfully on 2026-08-04:

| product | job | elapsed | batch MaxRSS |
| --- | ---: | ---: | ---: |
| BaryonForge map | 6753705 | 41 s | 3,718,288 KiB |
| GODMAX map | 6753706 | 63 s | 1,286,412 KiB |
| paired statistics | 6753707 | 13 s | 1,895,288 KiB |

Both maps record 136,651/136,651 painted halos, `complete_catalog_paint=true`,
NSIDE 1024, RING ordering, catalog SHA `d2763c3a...`, and the same config,
source, runtime, parameter, and effective-GODMAX-config contracts. The exact
unchanged statistics provenance gate passed all 44 shared fields.

The first production attempt exposed platform-dependent final-bit differences
in generated `k`, `ell`, and `delta-ell` grids. Fifty-nine numeric leaves
differed, with maximum relative difference `4.06e-15` and maximum absolute
difference `4.55e-13`; no nonnumeric field differed. The values actually
passed to GODMAX are now canonicalized to 13 significant decimal digits only
for those generated grids. The policy and affected paths are embedded in the
effective-config manifest. A `1e-10` relative grid change remains detectable,
nonfinite grids fail, and no tolerance in `measure_statistics.py` was changed.
Independent refutation approved this narrow fix. Final Rome and H100-host
manifests have the identical effective-config SHA `dbcabe5c...`.

As the null control, both final y and CMB-kappa arrays are bitwise identical to
their pre-canonicalization production controls for both backends: zero changed
pixels and zero maximum absolute residual. The production statistics artifact
contains 13 finite spectra in ten bins (effective ell `58.5--974`) on one
600.0425 deg2 C2-apodized cap. All diagnostic validity masks pass. Direct
residual spectra close against the component-spectrum identities with maximum
absolute differences `3.51e-31` (yy), `1.51e-19` (kk), and `6.31e-25` (yk).
It deliberately contains no covariance or radial stacks.

Across the ten production bins, the median BF/GM square-root auto-amplitude
ratio is `1.000672` for y and `1.013863` for kappa; median same-field coherence
is `0.9999984` and `0.987648`, respectively. The median residual-auto fraction
relative to GODMAX is `3.81e-6` for y and `0.02516` for kappa. The median
BF/GM y-kappa ratio is `0.998985` (range `0.994858--1.004408`).

Primary artifact hashes:

```text
profile HDF5       9fb607e23118b6ec5477ae9f898602c633ef610aee3a6756d58ce5546056cb21
convergence HDF5   480cfc6ece62ae28576da324c54b48dd1b33999bcc1007d9e0d1a1284fd4b103
bound summary JSON 165419bf5ab0c9794090df7c246c847273552252a4c85b86fbe5d3485f4fbfa2
memory JSON        a273303a64d88429986321846ff06295ff07a9fb70d0c5c292077bcefb5f5c51
GODMAX smoke map   798862c126b2d92fcd717c16db281d7a62212b02ca101c71f062dad962c9e6dd
BF smoke map       9b1f1af60331de50acf493c0b030bb0214bc31e41b22145309e911664f7eca18
smoke diagnostics  8de7c874fda75b412729b18df316088e8effe0dd94ed195b0dc74f1043691484
plot manifest      650e5a503624afa7c3bb34e8179abb46e3ccd2be656ab497588cc45bfe2a426d
GODMAX full map    e265d43334934ffdd432302d736dbb4ba748e30af0e5e5bb67462bd96c97f017
BF full map        29c41523e1020c6b5c41894c15ee4058ae0865e1b2464863809dca754bbe3280
full statistics    ec65e6258cdb224aef33297dc4b90f83f9f569356df6b9e57c9bf61c564b7d19
```

The production statistics executable was deliberately invoked on the capped
products. It rejected them before mask/spectrum construction because of
`max_halos=64`, incomplete-catalog provenance, and mismatch with the full
contract, and created no output.

### Production plots and 2400-deg2 extension

The completed 600-deg2 statistics and maps are now plotted by the fail-closed
`plot_results.py` driver. It verifies the config and both map hashes against
the statistics metadata before reading the products. `01_all_cls` contains all
13 decoupled C-ell bandpowers with their horizontal bin widths and no invented
vertical covariance; `02_cl_diagnostics` contains amplitude, coherence,
residual-power, and mixed-backend ratios. Separate tSZ and CMB-kappa figures
show raw inner-cap maps with common backend color limits and symmetric
BaryonForge-minus-GODMAX residual scales. The gnomonic width uses
`tan(radius)`, which prevents clipping the larger cap. Eight PNG/PDF files and
their input/output hashes are recorded under
`mgt13_pz3_cap600_projection_matched/plots/`; the plot-manifest SHA is
`0ee219387717380761b883fc90594d89a314a5a5171588f3caf5ed8773f6dc5a`.

The registered 2400-deg2 cap is centered at RA `37.96875 deg`, Dec
`-34.953865257188454 deg`, with radius `27.91480167872345 deg` and the same
one-degree catalog buffer. Streaming the canonical 52,122,273-row parent
(SHA `481c49b4...`) through the unchanged strict predicate gives 504,174
buffered halos: 470,298 centers in the inner cap and 33,876 in the buffer
shell. The maximum five-R200c support is `0.375308 deg`, so discarding those
outer centers would bias the cap boundary.

`backlight_compare_projection_matched_cap2400.yaml` changes only the
cap/catalog/output identity. The GODMAX integration remains comparison-only
GL64 on `0.01--128 R200c`; the 3D table remains 128 nodes to 70 comoving
Mpc/h; the common LOS projection remains 128-point nonsingular quadrature to
100 comoving Mpc; the native/global integration width remains 64; and map and
statistics settings remain NSIDE/lmax 1024 with no smoothing. Full validation
passes with exact counts, cosmology, unit crosswalk, query-disc safety, and
the unchanged 1e-3 convergence policy.

The dependency chain was submitted on 2026-08-04 and was pending scheduler
priority when this entry was recorded:

| stage | job | dependency | requested resource |
| --- | ---: | --- | --- |
| strict catalog | 6754167 | none | Rome CPU, 4 cores/16 GiB, 15 min |
| GODMAX map | 6754168 | afterok 6754167 | one H100, 8 cores/64 GiB, 30 min |
| BaryonForge map | 6754169 | afterok 6754167 | Rome CPU, 8 cores/64 GiB, 15 min |
| statistics + plots | 6754171 | afterok 6754168:6754169 | Rome CPU, 16 cores/32 GiB, 15 min |

The last stage will create all 13 common-mask spectra and eight PNG/PDF plot
products only if both complete map gates pass. No 2400-deg2 map or bandpower
result is claimed until those jobs and artifact checks complete.

## Failure modes prevented

- Treating M200c as the integral of an extended NFW profile creates a false
  mass-budget failure; M200c is enclosed at R200c.
- Comparing physical radii with a comoving table boundary and endpoint
  clamping creates a redshift-dependent projected tail.
- Raising native BaryonForge points per decade cannot restore an interval the
  quadrature omits.
- A 100-R200c bound looks wide but fails the frozen gas-normalization gate.
- Restoring the uniform 128-R200c trapezoid directly to 64 points fails HSE;
  allocating the same 64 nodes with log-radius Gauss--Legendre passes.
- A held-fixed or method-blind pressure scan does not validate the
  integration-dependent HSE chain; method-aware schema v3 is mandatory.
- Nominal five-R200c support is not bit-identical because PyCCL recomputes
  R200c and DA; exact native footprints remain explicit diagnostics.
- Hashing uncanonicalized generated transcendental grids makes identical
  configs differ across CPU math libraries; canonicalize only the final grids
  used by GODMAX and retain exact hashes for every input and source file.
- The 64-halo spectra are not production bandpowers. Complete-catalog status,
  config/parameter/source hashes, units, geometry, smoothing, and nonzero maps
  are hard requirements of `measure_statistics.py`.

## Verification boundary

The focused comparison test file passes (`30 passed`), bytecode compilation
and diff checks pass, and both parameter files pass the strict crosswalk. The
profile, capped-map, complete-map, and paired-bandpower gates are closed as
PASS. The final jobs are 6753705, 6753706, and 6753707; all exited 0.

The detailed commands and output layout are in
`notebooks/xDESI/baryonforge_compare/README.md`. Historical native-8R and
integration-only artifacts remain immutable controls under their original
output roots.

## Open questions

- Production radial stacks and a covariance are not included in the first
  paired statistics artifact and would require separately specified products.
- Native backreaction solvers are not expected to match through parameter
  files alone. Any future comparison must be a separately labeled experiment.
- A provenance-preserving GODMAX split combiner is required before using more
  than one production split.
- HSE convergence is sampled at nine mass-redshift nodes and five radii, not
  claimed as a continuous-domain bound.
- Imported PyYAML 6.0.2 and SciPy 1.14.1 disagree with their local distribution
  metadata (6.0.1 and 1.12.0). Exact imported module hashes are recorded;
  environment recreation must follow those imported versions rather than
  metadata alone.
