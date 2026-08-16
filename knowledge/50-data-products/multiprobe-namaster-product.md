---
id: kb.measurement.multiprobe-product
title: The xDESI multi-probe NaMaster product — schema, conventions, noise policy
layer: 50-data-products
owner: measurement-namaster
status: verified
confidence: high
scope:
  - notebooks/xDESI/survey_measure/multiprobe_namaster.py
  - notebooks/xDESI/survey_measure/run_multiprobe_production.py
  - notebooks/xDESI/survey_measure/submit_multiprobe_cpu.sh
  - notebooks/xDESI/survey_measure/submit_multiprobe_true_nz_cpu.sh
  - notebooks/xDESI/survey_measure/submit_multiprobe_midres_true_nz_cpu.sh
  - notebooks/xDESI/survey_measure/run_multiprobe_cpu_worker.slurm
  - notebooks/xDESI/survey_measure/run_multiprobe_cpu_worker.sbatch
  - notebooks/xDESI/survey_measure/run_multiprobe_cov_bundle_worker.sbatch
  - notebooks/xDESI/survey_measure/run_multiprobe_finalize_worker.sbatch
  - notebooks/xDESI/survey_measure/submit_multiprobe_highres4096_efficient.sh
  - notebooks/xDESI/survey_measure/build_desi_dr9_multi_random_mask.sbatch
  - notebooks/xDESI/survey_measure/prune_stage31_cov_for_shearfix.py
  - notebooks/xDESI/survey_measure/prepare_multiprobe_maps.py
  - notebooks/xDESI/survey_measure/measure_multiprobe_namaster.py
  - notebooks/xDESI/survey_measure/plot_multiprobe_measurement.py
  - notebooks/xDESI/survey_measure/plot_highres_pilot_dell.py
  - notebooks/xDESI/survey_measure/migrate_galaxy_auto_shot_noise.py
  - notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py
  - notebooks/xDESI/survey_measure/DES_SHEAR_TOMO44_FIG4_METHOD.md
  - notebooks/xDESI/survey_measure/README.md
  - tests/test_xdesi_multiprobe_namaster.py
invariants:
  - INV-NMT-COUPLED-01
  - INV-NMT-BANDMAJOR-01
  - INV-SHEAR-SIGN-01
  - INV-KSZ-CATALOG-01
  - INV-SHOTNOISE-01
  - INV-BEAM-01
  - INV-PRODUCT-PROV-01
  - INV-DV-SHAPE-01
checks:
  - pytest tests/test_xdesi_multiprobe_namaster.py -q
verified_at_commit: cf72943
verified_on: 2026-08-16
see_also: [kb.xdesi.ksz-conventions, kb.xdesi.analysis-state]
scope_digest: sha256:354e9d94a2b69487cc0535c5387f747a
---

## Claim

The multi-probe product is a 46-spectrum decoupled-bandpower data vector (460 elements for
`fast1024`, 736 for the 16-band `midres2048`, and a 920-slot archive/892-element active
selection for the 20-band `highres4096`) with a matching Gaussian/iNKA covariance, saved as HDF5 together
with the masks, weights, n(z), priors and per-field noise policy needed to reproduce it.
Theory enters the comparison only through `theory_to_data_vector`, which applies the saved
bandpower windows and field-specific transfers. Current signal-plus-shot spectra and final
products carry `_pipev2_gshot`; their reusable map and covariance artifacts retain `_pipev2`.
Convention metadata rejects old signal-only spectra rather than relabelling them. Final
products additionally expose a deterministic weighted-Poisson-subtracted galaxy-auto view;
the backward-compatible primary and HMC default remains total `C_ell^gg + SN`.

## Why it is true

`notebooks/xDESI/survey_measure/README.md` is the authoritative convention record; the
statements below are quoted from it and each has a corresponding invariant.

**Spectrum inventory** (46 spectra; 460 fast elements, 736 mid-resolution elements,
920 high-resolution archive slots): 10 DES shear EE, 4 y × shear-E, 4 DESI
galaxy autos, 4 galaxy × y, 16 galaxy × shear-E, 4 galaxy × ACT kappa, 4 DESI momentum × ACT
temperature kSZ. The regenerated pipeline-v2 fast covariance retains all 460 modes at the
fixed 1e-8 correlation eigenvalue cut (`corr_eig_min = 0.0657707658`); the legacy fast
covariance retained 459. Rank is product-specific and must not be transferred between them.
The completed 16-band pipeline-v2 `midres2048` product likewise retains `736/736` modes at
the same fixed cut, with correlation eigenvalues `[0.3511573968, 3.5575833833]`; all 1,081
upper-triangle covariance blocks exactly match their band-major joint-matrix positions.
Covered by `test_default_spectrum_inventory_is_46` and
`test_component_labels_match_namaster_ordering`. The high-resolution 920/28/892 packing
contract is covered by
`test_highres_kappa_packing_keeps_raw_covariance_basis_and_marks_exactly_28_placeholders`.

**Stages and binning** — four distinct, non-interchangeable schemes:

- `fast1024`: nside 1024, lmax 1024, 10 **linear** bins, edges
  `[8, 110, 212, 314, 415, 517, 619, 720, 822, 924, 1025]`.
- `midres2048`: nside 2048, science ell 128–4096, full HEALPix mask bandwidth
  `lmax_mask=6143`, and 16 **hybrid-log** bins. The first 13 bands preserve the established
  through-3000 table; complete left edges are
  `[128, 160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693]`
  and right-exclusive edges are
  `[160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693, 4097]`.
  V2 does not globally apodise the source/catalog masks and does not use pair-dependent mean
  subtraction. Exact edge arrays and mask bandwidth are part of manifest, shard, workspace,
  map and measurement compatibility identities.
- `highres4096`: nside 4096, science ell 128–8192, full HEALPix mask bandwidth
  `lmax_mask=12287`, and 20 hybrid-log bins. The first 13 bands are byte-identical to
  the established through-inclusive-3000 table. The seven added right-exclusive
  intervals have boundaries
  `[3001, 3464, 3998, 4615, 5327, 6149, 7098, 8193]` and cover through inclusive
  ell 8192. This stage requires at least eight recorded DR9 random realizations and
  confirmed ACT-temperature units `uK_CMB`.
- DES Y3 fiducial low-res diagnostic: nside 1024, 32 equal-weight bandpowers with edges
  uniformly spaced in **sqrt(ell)** over ell 8–2048 — not logarithmic, not linear-width.

Covered by `test_linear_bandpowers_match_cpu_production_edges`,
`test_des_y3_fiducial_bandpowers_match_transferred_edge_rule`,
`test_sqrt_bandpowers_cover_requested_ell_range`.

**Covariance construction.** Blocks computed in decoupled bandpower space with
`nmt.gaussian_covariance(..., coupled=False)` (`INV-NMT-COUPLED-01`). Flattened arrays are
**band-major**, so blocks are extracted with
`cov.reshape(n_band, n_comp_a, n_band, n_comp_b)[:, comp_a, :, comp_b]`
(`INV-NMT-BANDMAJOR-01`). Ordinary map-field inputs use NaMaster's data-derived improved
narrow-kernel approximation (`nmt.get_iNKA_cell`), retaining all coupled spin components.
kSZ catalog-momentum inputs follow the NaMaster tutorial convention: coupled pseudo-`C_ell`
divided by mask overlap, with catalog zero-lag `Nf` added back only for literal momentum
autos. The old decouple/smooth/unbin input is retained only as an explicit legacy mode.

**Noise policy** (`INV-SHOTNOISE-01`). DES shear autos subtract the catalog-derived constant
coupled shape-noise pseudo-spectrum matched to the un-apodised weighted-count mask. DESI
galaxy autos instead retain clustering signal plus the exact conditional weighted Poisson
shot noise **exactly once** in `spectra/<name>/cl` and `joint/data_vector`. The catalog
template is the constant coupled pseudo-noise
`Omega_pix sum(w^2) / [N_pix (alpha random_mean)^2]`; its decoupled bandpower response is
saved as `noise_decoupled_all_components[component]`. The default iNKA covariance reads the
raw map auto, which already contains the same term, and therefore performs no second
add-back. Theory windows the clustering signal normally and then adds
`A_shot,pz * noise_decoupled` in bandpower space; a flat term must not be passed through the
ordinary galaxy pixel transfer. `A_shot,pz=1` is the catalog Poisson prediction and can be
made a per-bin nuisance parameter. **No ACT or kSZ noise is subtracted from the saved data
vector.** Each `input_cls_for_covariance/*` dataset records its spin labels and noise policy.

**Galaxy-auto mean views.** The final product keeps `joint/data_vector`,
`joint/data_vector_raw`, and `spectra/desi_g_auto_pz*/cl` as the total primary view. Under
`joint/views/total`, those arrays and `joint/cov` are HDF5 hard links to the primary datasets.
The `joint/views/weighted_poisson_subtracted` vectors subtract the saved decoupled template
in exactly four galaxy-auto spectra (80 archive entries for `highres4096`) and change no
other spectrum; per-spectrum subtracted arrays are saved alongside each affected auto.
Both views hard-link the same full covariance. This is required, not an approximation:
conditional on the fixed catalog/mask template, subtracting a deterministic mean leaves the
estimator covariance and every cross-covariance unchanged. A covariance recomputed from a
signal-only auto would incorrectly remove the observed Poisson fluctuations. The common
validity mask still removes only the 28 unsupported high-ell `g x kappa` entries. The
central loader accepts `galaxy_auto_view="total"` or
`"weighted_poisson_subtracted"`; its default is total so a future HMC fit can model
clustering plus free per-pz shot-template amplitudes without changing the measured vector.

**Field and mask conventions.** DES spin-2 fields from `gamma1` and `gamma2_namaster`, multiplied by
`shear_e_to_kappa_sign = -1` (`INV-SHEAR-SIGN-01`) — leaves EE unchanged, aligns scalar ×
shear-E with positive-convergence theory. DES shear and DESI catalog masks remain
un-apodised. ACT masks use bounded spline reprojection; T/kappa source tapers are preserved,
  and the already-masked kappa map is passed with `masked_on_input=True`. HEALPix pixel windows
  apply to DES shear and DESI galaxy map fields, but not to catalog momentum or harmonically
  reprojected ACT fields. Production ACT maps are reprojected at native CAR resolution rather
  than block-averaged with an unmodelled anisotropic response. ACT beams and the kappa filter
  remain field-specific transfers.

**Inputs recorded with the product** (`INV-PRODUCT-PROV-01`):

- DESI galaxies and kSZ: `data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5`,
  selection `catalog/valid_for_cl`, weight `catalog/weight_imaging_mean1`.
- DESI masks: DR9 quality-cut random-count HEALPix maps in
  the identity-tagged eight-realization product
  `data/desi_dr9_imaging_randoms/desi_dr9_randoms_i0-1-10-11-12-13-14-15_0d91e56b0550_lrg_quality_count_maps_nside1024_2048_4096.h5`,
  using the native requested nside group. The companion provenance product records exact
  indices, per-pair cuts and cumulative support. Schema v3 binds and verifies the complete
  18-file source inventory against `SHA256SUMS.raw.txt`; the loader compares the saved
  ledger/inventory digests to the active survey manifest. The legacy one-random map remains
  only as a pair-0 bitwise null. High resolution fails closed unless the map records at least
  eight realizations with this provenance contract. Older
  `midres2048` products may instead record a **sum-preserving nside4096 → nside2048
  downgrade** from the historical map.
- DESI theory kernel: calibrated true-z n(z) at `nz/desi/nz_dndz_by_pz`
  (`INV-NZ-TRUEZ-01`); the photo-z histogram lives separately under
  `nz/desi_photoz_diagnostic` and must not be used as theory.
- DES Y3 source n(z): FITS HDU `nz_source` from
  `2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits`; raw bin values and
  normalised theory `dN/dz` both saved under `nz/des_shear`.
- DES Y3 Gaussian priors under `priors/des_y3_gaussian` (`INV-PRIOR-DESY3-01`).
- ACT y and T theory get a 1.6 arcmin Gaussian beam before the saved windows
  (`INV-BEAM-01`).
- Every v2 map records a content-addressed `map_product_id` over its input/config metadata
  and every saved mask, map and catalog array, the n(z) metadata, and each field's complete
  estimator contract (`name`, `kind`, `spin`, mask reference and metadata). Loading
  recomputes the relevant byte/metadata digests;
  spectra reuse requires the exact originating map ID and all estimator config keys.
  Covariance manifests, shards and workspaces record algorithm/config/group/map identities,
  and shards additionally record their actual representative-mask digest. Assembly rejects
  missing, duplicate, mixed-provenance or conflicting inputs rather than silently overwriting.
  The efficient high-resolution Slurm DAG is bound to a submission-time digest of its six
  runtime source/script files and a full-file digest of its immutable covariance work plan.
  The plan binds the map identity, spectra SHA256, manifest/config/group identities, and the
  exact hashes of reused shards; finalization re-attests those inputs before assembly. Every
  phase fails closed if the frozen bytes change before execution.

**High-resolution archive/analysis split.** All 46 spectra use the same 20-band grid, so
the archival layout has 920 slots. The ACT kappa response is nonzero through ell 3000 and
zero from 3001 onward. For only the four `desi_g_act_kappa` spectra, bins 13..19 are marked
false in `joint/data_vector_valid` and are exact zeros in `joint/data_vector`. Their raw
estimator values remain in `spectra/<name>/cl` and `joint/data_vector_raw`; the complete
920x920 raw-estimator Gaussian covariance and all covariance blocks remain unmodified.
Statistical consumers select 892 entries and `cov[np.ix_(valid,valid)]`, the ordinary
principal submatrix, never a Schur complement and never the inverse of the full matrix
with zero placeholders. The validity policy uses complete band support and requires the
right-exclusive edge 3001 exactly; a straddling bin is an error. This contract uses schema
`xdesi_multiprobe_measurement_v2` and tag `_gkell3000_dvvalidv1`, so older readers fail
closed. The completed high-resolution covariance retains all `920/920` archival modes and
all `892/892` active modes at the fixed `1e-8` correlation-eigenvalue cut; the 28 excluded
slots remain only the four ACT-kappa spectra's seven transfer-null bands.

**Efficient high-resolution production DAG.** Submission 2026-08-15 reuses the validated
nside-4096 map, complete 46-spectrum pilot, and compatible covariance groups 116, 237 and
257; it submits no preparation or spectra job. Of 259 manifest groups, 256 missing groups
are assigned exactly once to 24 balanced bundles (16 of size 11 and 8 of size 10). Job
`6884226` first runs a one-node worst-case stress bundle. Only after it succeeds does array
`6884227_1-23%5` run, enforcing a hard five-covariance-node cap. Every Rome node allocates
121 CPUs and 880 GiB; it launches up to eleven disjoint
`srun --exclusive --exact` steps of 11 CPUs/80 GiB, stages the map once to node-local
storage, and disables the non-reusable covariance-workspace cache. Shared-partition job
`6884228` then assembles, validates, exercises both HMC loaders, writes the readiness JSON,
and plots using 2 CPUs/4 GiB. Its `afterok` dependency covers the complete array. The stress
job and 22 of 23 array tasks completed; task `6884227_7` ended in scheduler state
`NODE_FAIL`, leaving only covariance groups 207 and 247 absent and preventing that
finalizer from starting. Minimal recovery job `6886882` recomputed only those two groups,
then finalizer `6886883` completed assembly and validation. The result contains all 259
compatible shards and all 1,081 finite covariance blocks. The canonical HDF is
`data/xDESI/processed/multiprobe_namaster_highres4096_ell8192_dr9random8/highres4096/xdesi_multiprobe_cls_cov_nside4096_ell128_lmax8192_lmask12287_nbin20_log_pipev2_gshot_gkell3000_dvvalidv1.h5`
with SHA256 `9462890c673f6b5b6628d638f386a57dfcc287ed2244939d98bc7f9a6394637a`.
This establishes a structurally HMC-ready saved input, not a Stage-31 theory/gradient run,
fit quality, S/N, or posterior result.

**Plot products.** The terminal production job runs only after HDF5 validation and writes
separate all-family `C_ell` and `D_ell` PDFs and PNGs. Both use one-sigma errors from the
diagonal of the saved joint covariance. The `ell(ell+1)/(2 pi)` conversion is applied to
the mean and error identically. Raw `C_ell^{pi,T}` is retained in the kSZ `C_ell` view;
only the `D_ell` view applies the paper-style minus sign.
Primary production plots apply no ell cut or fixed kSZ y-range, use a log x-axis for log
bandpowers, and fail if any saved family is unknown or omitted. Validity-aware production
plots omit the seven transfer-null `g x kappa` bands; archive diagnostics can inspect their
raw per-spectrum values without treating them as likelihood data. The corrected final
high-resolution plots set every visible log axis explicitly from the saved first left edge
through the requested display maximum, `[128, 8192]`, after selecting log scale. Galaxy-auto
panels explicitly label the primary total `C_ell^gg + N_ell^shot` view; kSZ `C_ell` labels
its `10^3` display scale, and `D_ell` retains the paper-style `-10^3` convention. The final
inventory is seven family PNGs and one seven-page PDF for each of `C_ell` and `D_ell`, plus
one summary JSON. The 2026-08-16 regeneration changed only plot geometry/labels: the HDF and
all pre-registered plotted-value/error hashes remained exact.

**The theory path** (`INV-WINDOW-CMP-01`), from
`BACKLIGHT_PASTE_HANDOFF_SUMMARY.md`:

```python
theory_to_data_vector(
    measurement_h5, theory_cls, ell=ell_theory,
    shear_m_bias=saved_m_means, ksz_velocity_correlation=0.3,
    desi_galaxy_shot_noise_amplitudes={1: A1, 2: A2, 3: A3, 4: A4},
    include_default_pixel_windows=True, include_default_act_beams=True,
    theory_shear_e_is_positive_kappa=True,
)
```

Smooth theory at `ell_eff` is a diagnostic only.
For galaxy autos, `theory_cls` contains clustering signal only; the wrapper applies the
ordinary signal response first and the saved shot-template response second. The JAX-native
Stage-31 path uses the same ordering. Optional sampled parameters named
`desi_galaxy_shot_noise_amplitude_pz{1..4}` override the fixed config/default amplitudes;
their saved values under `params.other_params` are reused by subsequent theory-vector
consumers. Declaring both sampled/saved values and an explicit fixed wrapper amplitude is a
hard ambiguity error. No prior widths are selected by the measurement code.
Theory consumers reject products without all pipeline-v2 algorithm versions by default. They
also require the measurement and the separate map/n(z) HDF5 to carry the same `map_product_id`,
and verify that the consumed source/lens n(z) arrays match the embedded content-addressed
metadata. Only the generic NumPy `theory_to_data_vector` wrapper exposes
`allow_legacy_product=True` as a historical-only opt-in. The Stage-31 HMC/likelihood path has
no legacy opt-in and hard-requires the current `_pipev2_gshot` measurement convention.
New cached theory-vector NPZs are also content-bound to the exact measurement arrays and the
materialized comparison configuration. A separate response fingerprint hashes the effective
saved field transfers, ordered spectrum metadata, bandpower windows and selected galaxy-shot
templates, so changing response content at an unchanged HDF5 path invalidates the cache.
Stage-31 best-fit vectors are additionally bound to the likelihood, exact parameter/prior
contract, saved sample and chi2. The comparison plot fails closed on older caches that lack
these identities; rerun the theory-vector producer instead of copying metadata onto an old
vector.
For validity-mask products, the generic wrapper and central measurement loader return the
active 892-element layout by default. The saved mask is included in response/cache identity;
an explicit archive-diagnostic option is required to return the full 920-slot layout. The
Stage-31 likelihood intersects this mask with any user scale cuts before constructing
windows and the whitener, and cannot re-enable an invalid band.

## How to verify

```bash
pytest tests/test_xdesi_multiprobe_namaster.py -q
python tools/kb/kb.py invariants --check --layer measurement
```

The suite builds its own synthetic HDF5 inputs, so it runs without cluster data. Expected:
all tests pass; 46-spectrum inventory, band-major extraction, exact 20-band edge table,
920/28/892 validity packing and active principal-submatrix assertions in particular.
The generated v2 fast product was additionally validated with:

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  notebooks/xDESI/survey_measure/run_multiprobe_production.py validate \
  --stage fast1024 \
  --output-dir data/xDESI/processed/multiprobe_namaster_true_nz
```

It reports shape `(460,460)`, finite/symmetric covariance, a strictly positive diagonal,
no negative covariance/correlation eigenmodes, and rank `460/460` at `1e-8`. The current
signal-plus-shot file is
`fast1024/xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear_pipev2_gshot.h5`
(SHA256 `d315f8512edaa58b3920b934526d9ba0a70f5b3fbf10a512840ba090813cd835`). It was
losslessly migrated from the retained signal-only `_pipev2.h5` source (SHA256
`0efb9db2e625bc11fd3cff6fcc1ac17ef5ff89b488722361a2ff5625ec25c5b8`): exactly the 40
galaxy-auto elements changed, while 1,339 covariance-related datasets were array-identical.
Four galaxy-auto covariance-input `noise_policy` attributes were intentionally updated to
say that both the raw iNKA input and saved mean contain shot noise; all other protected HDF5
attributes match the source.
The migration also materialized the source's implicit `lmax_mask=lmax=1024` metadata so the
new copy passes the current fail-closed identity check. Full command and null-control
evidence are in `knowledge/.kb/ledgers/2026-08-05-galaxy-auto-shot-noise-included.md`;
the original production evidence remains in
`knowledge/.kb/ledgers/2026-08-05-fast1024-v2-production.md`.

The completed mid-resolution product is validated with the same command using
`--stage midres2048`. It reports shape `(736,736)`, rank `736/736` at `1e-8`, no negative
covariance/correlation modes, and a strictly positive diagonal. The HDF5 SHA256 is
`32d04b92a5b33f6e3c5c8ae28f6ac36682a4f0377502183f90a9d050a74eab1b` for the current
`_pipev2_gshot.h5` product. It was losslessly migrated from the retained signal-only source
(SHA256 `bca1000bed30f89969dc7f1185e87967c3e230b2528bf087ff61594240fd67a9`): exactly the 64
galaxy-auto elements changed, while the full covariance and 1,339 covariance-related
datasets remained array-identical; only the same four convention-bearing `noise_policy`
attributes changed within the covariance-input group. Its two seven-page all-family plot PDFs and 14 family
PNGs use covariance-diagonal errors. Migration evidence is in
`knowledge/.kb/ledgers/2026-08-05-galaxy-auto-shot-noise-included.md`; original submission,
edge/block checks and scheduler accounting remain in
`knowledge/.kb/ledgers/2026-08-05-midres2048-lmax4096-nbin16-production.md`.

## Failure modes

- **Component-major covariance extraction.** Nothing raises; the matrix stays symmetric and
  positive-definite; covariance is attributed to the wrong probe pair; chi2 is plausible and
  wrong. The quietest failure in the pipeline.
- **`coupled=True`.** Covariance leading dimension becomes `n_ell` instead of `n_band`;
  whitening rank is wrong.
- **Missing shear sign.** Pristine EE spectra alongside four inverted cross families
  (y × shear-E, g × shear-E, kappa × shear-E) whose chi2 improves if theory is hand-flipped.
- **Beam applied twice.** Monotonic high-ell deficit confined to ACT y and T families,
  growing with ell, with low-ell bands unaffected.
- **Old kSZ cache.** `KeyError` on `catalog/` arrays, or a momentum auto with no white-noise
  floor and an implausibly small covariance diagonal. Caches predating the NaMaster 2.7
  update must be regenerated.
- **Legacy reuse.** Missing `_pipev2_gshot`, the mean-convention attribute, algorithm
  versions or digests is a hard error for current spectra/final products. `_pipev2` map and
  covariance artifacts remain reusable because the map realization and iNKA covariance did
  not change. The
  old four-shear-auto patch and selective covariance pruning are unsafe because corrected
  masks/windows affect every spectrum and Wick block with those endpoints.
- **Harmonic ACT mask reprojection.** Positive Gibbs support appears outside the footprint;
  mask means, coupling workspaces and all ACT covariance blocks change silently.
- **Pair-dependent mean subtraction.** The same field acquires a different realization for
  each partner, so Wick contractions no longer describe the measured estimators.
- **Mixing binning schemes across stages.** Positional indexing means residuals get
  attributed to the wrong band; symptom is a discontinuity at a family boundary.
- **Unrecorded mask realization.** Two stages disagree at low ell with no code difference.

## Open questions

- The eight-realization DR9 schema-v3 mask passed its full-source checksum gate, pair-0
  bitwise null, output content checksums, cumulative-support audit and reuse validation.
  Its eighth realization still adds 1.5147% of final native nside-4096 support, so it
  addresses the known one-random sparsity but does not by itself model
  finite-random or catalog-selection uncertainty; mock/resampling validation remains needed
  before final inference, and apodisation is not a substitute.
- The saved covariance is Gaussian/iNKA only; connected non-Gaussian, super-sample,
  foreground and catalog-selection terms need mock/simulation validation.
- ACT T units are user-confirmed as `uK_CMB` for `highres4096`. The transferred velocity
  catalog still does not reproduce the published cleaning/symmetrisation, so a published-
  sample kSZ amplitude remains a separate matched remeasurement.
- The `highres4096` Gaussian/iNKA measurement and covariance are complete and structurally
  HMC-ready, with the full and active ranks recorded above. No Stage-31 high-resolution
  theory vector, gradient/sampler smoke test, S/N, chi2, posterior, or physical conclusion
  has yet been produced; product completion must not be read as model validation.
