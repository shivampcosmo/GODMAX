---
id: kb.measurement.xdesi-input-provenance
title: xDESI measurement-input preparation and DESI catalog provenance
layer: 50-data-products
owner: measurement-namaster
status: verified
confidence: high
scope:
  - measurements/xDESI/scripts/compute_dr10_imaging_weights.py
  - measurements/xDESI/scripts/plot_des_y3_tomo4_paper_style_covariance.py
  - measurements/xDESI/scripts/prepare_act_desi_ksz_hdf5.py
  - measurements/xDESI/scripts/prepare_des_y3_shear_maps.py
  - measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py
  - measurements/xDESI/scripts/prepare_desi_dr9_lrg_nz.py
  - measurements/xDESI/notebooks/diagnose_des_y3_shear_tomo4_covariance.ipynb
  - measurements/xDESI/notebooks/prepare_act_desi_ksz_hdf5.ipynb
  - measurements/xDESI/notebooks/prepare_des_y3_shear_maps.ipynb
  - measurements/xDESI/notebooks/validate_desi_dr9_ksz_cleaned_lrg_catalog_and_nz.ipynb
  - notebooks/xDESI/survey_measure/build_desi_dr9_multi_random_mask.sbatch
  - notebooks/xDESI/survey_measure/multiprobe_namaster.py
invariants:
  - INV-PRODUCT-PROV-01
  - INV-NZ-TRUEZ-01
  - INV-KSZ-CALIB-01
checks:
  - python -m py_compile measurements/xDESI/scripts/compute_dr10_imaging_weights.py measurements/xDESI/scripts/plot_des_y3_tomo4_paper_style_covariance.py measurements/xDESI/scripts/prepare_act_desi_ksz_hdf5.py measurements/xDESI/scripts/prepare_des_y3_shear_maps.py measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py measurements/xDESI/scripts/prepare_desi_dr9_lrg_nz.py
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m pytest tests/test_xdesi_multiprobe_namaster.py -q -k "survey_bundle or build_desi_fields or calibrated_true_nz"
  - /usr/bin/env PATH=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin:/usr/bin:/bin python tools/kb/kb.py invariants --check --id INV-PRODUCT-PROV-01
  - /usr/bin/env PATH=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin:/usr/bin:/bin python tools/kb/kb.py invariants --check --id INV-KSZ-CALIB-01
  - rg -n "spectroscopic_calibrated_true_redshift" notebooks/xDESI/survey_measure/multiprobe_namaster.py notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py
  - "[needs-data] jq -e '.products.desi_dr9_extended_velocity_catalogs.combined == \"data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5\"' data/xDESI/survey_data/manifest.json"
  - bash -n notebooks/xDESI/survey_measure/build_desi_dr9_multi_random_mask.sbatch
  - "[needs-data] /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py --transfer-root data/xDESI/survey_data --random-source-root data/xDESI/survey_data/desi_dr9/legacy-survey-0.49.0 --random-indices 0 1 10 11 12 13 14 15 --skip-catalogs --validate-only"
verified_at_commit: cf72943
verified_on: 2026-08-05
see_also: [kb.measurement.multiprobe-product, kb.xdesi.ksz-conventions, kb.xdesi.analysis-state]
supersedes: []
scope_digest: sha256:96cff16d439ec64e928c27b02eb4b2a1
---

## Claim

The active xDESI NaMaster galaxy measurements use the **DESI Legacy Surveys DR9
Extended LRG** object catalog, with public DR9 imaging weights, DR9 randoms and a
spectroscopically calibrated DR9 true-redshift distribution. They do not use the DR10
Extended object catalog or the approximate DR10 imaging-weight workflow as the object,
weight, mask or lens-`n(z)` input. A separate kSZ velocity-amplitude calibration is an
important exception to the broader phrase "no DR10-derived input": its Abacus metadata
names DR10 Extended photometric bins, but it does not define the measured galaxy sample.
The retained `fast1024` and `midres2048` products used the legacy one-realization DR9
random mask. The new `highres4096` production uses a validated count map made from the
exact eight transferred DR9 random/LRG-mask pairs. It fails closed below eight realizations
or when the manifest identity/index set, full-source checksum ledger/inventory, HDF schema,
count sum, or native-map checksum disagrees. The nside-4096 support is much denser than the
retained one-random map but still shows measurable finite-random support growth, so the
measurement can proceed while its catalog-selection covariance remains explicitly
provisional.

## Why it is true

### Object lineage and the meaning of `Extended`

Here `DR9` and `DR10` identify the **Legacy Surveys imaging/target-catalog
release**, not a DESI spectroscopic survey data release. The transferred DR9 random
realizations come from `desi/public/ets/target/catalogs/dr9/0.49.0`, while the separate
DR10 random comes from the Legacy Surveys `dr10/randoms` tree
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:55`;
`measurements/xDESI/scripts/prepare_act_desi_ksz_hdf5.py:57`).

The production catalog builder names its input and output DR9 Extended explicitly. It
joins the four velocity files
`DESI_pz{1..4}/extended_catalog_allfoot_perbin_sigmaz0.0500.txt` to the public
`lrg_xcorr_2023/v1/catalogs/dr9_extended_lrg_pzbins.fits` catalog and its two public DR9
weight tables (`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:2`,
`:49`, `:81`). The `sigmaz0.0500` velocity sample imposes
`Z_PHOT_STD <= 0.05 * (1 + Z_PHOT_MEDIAN)`
(`measurements/xDESI/scripts/prepare_desi_dr9_lrg_nz.py:2`, `:42`, `:62`, `:224`).
This is the **Extended LRG** selection, not the standard/main-LRG catalog and not a
spectroscopic DESI clustering catalog.
The four `pz` labels and their boundaries are inherited from the public `pz_bin` column
and the four input files; these preparation scripts do not reapply numerical pz-bin edges
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:81`, `:234`).

For each photo-z bin, the builder matches the velocity rows back to the public DR9
Extended catalog by sky position and `Z_PHOT_MEDIAN`, attaches the public precomputed DR9
weights, and emits
`data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5`
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:234`, `:279`, `:303`,
`:401`). The match tolerances are `1e-4` arcsec and `1e-5` in redshift
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:70`, `:307`).

The recommended object selection is `catalog/valid_for_cl`. It requires a successful
public-DR9 match, the DR9 LRG quality footprint, and a finite positive imaging weight;
the measurement weight is `catalog/weight_imaging_mean1`, the public DR9 Extended weight
renormalized to mean one within each photo-z bin
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:313`, `:327`,
`:343`, `:373`). The quality footprint requires `NOBS_G/R/Z >= 2`, `lrg_mask == 0`,
`EBV < 0.15`, stellar density below 2500 deg^-2, and removal of the
`DEC < -10.5, 120 < RA < 260` islands
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:200`, `:343`).

The generated combined HDF5 contains 19,911,871 rows, of which 19,386,574 pass
`valid_for_cl`: 2,794,391, 4,743,086, 6,187,158 and 5,661,939 in pz1--pz4. These are
execution-derived product counts; the exact inspection command and output are in
`knowledge/.kb/ledgers/2026-08-05-xdesi-input-provenance.md`.

### What the active measurement consumes

`SurveyBundle.from_root` resolves only the manifest keys
`desi_dr9_extended_velocity_catalogs`, `desi_dr9_imaging_randoms` and
`desi_dr9_redshift_distributions`; it never resolves a DR10 DESI key
(`notebooks/xDESI/survey_measure/multiprobe_namaster.py:123`). The field builder records
`desi_release = DR9 Extended LRG`, filters on `valid_for_cl`, uses
`weight_imaging_mean1`, and saves those choices in every galaxy and momentum field
(`notebooks/xDESI/survey_measure/multiprobe_namaster.py:1677`, `:1711`, `:1808`,
`:1884`). Both the diagnostic galaxy maps and the catalog-momentum kSZ estimator therefore
come from the same selected DR9 rows; the latter uses `(ra_deg, dec_deg)`, the DR9 imaging
weight, and `vr_over_c` (`notebooks/xDESI/survey_measure/multiprobe_namaster.py:1894`).

The current submission wrappers default to
`data/xDESI/processed/multiprobe_namaster_true_nz`, and the current default map names have
the `pipev2` suffix (`notebooks/xDESI/survey_measure/submit_multiprobe_true_nz_cpu.sh:11`;
`notebooks/xDESI/survey_measure/submit_multiprobe_midres_true_nz_cpu.sh:14`;
`tests/test_xdesi_multiprobe_namaster.py:211`, `:228`). The live manifest, those current
`fast1024` and `midres2048` products, and two retained reference products in the same
`multiprobe_namaster_true_nz` tree all resolve the same combined DR9 catalog and record
the calibrated true-`n(z)` group. The current fast product retains 2,789,662, 4,734,999,
6,176,941 and 5,652,806 objects after applying its native nside-1024 random mask. The
current mid-resolution product retains 2,367,662, 4,018,807, 5,243,607 and 4,804,333
after applying the sum-preserving nside-4096 to nside-2048 mask downgrade. Those
stage-specific counts and the unchanged DR9 catalog identity were read directly from
saved HDF5 metadata with the command in the evidence ledger; they are not notebook
outputs. Similarly named products outside `multiprobe_namaster_true_nz` are not used as
evidence for the calibrated true-`n(z)` claim. These retained counts describe the legacy
one-realization masks and must not be relabelled as counts from the new eight-random mask.

The angular selection is also DR9. The legacy product used the quality-cut
`randoms-1-0.fits` realization, with initial `NOBS_G/R/Z >= 1` and MASKBITS 1, 12 and 13
vetoed before the LRG footprint cuts. The multi-pair builder applies those same cuts while
streaming every selected pair (`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:216`,
`:1150`, `:1178`).

The transferred local source tree is
`data/xDESI/survey_data/desi_dr9/legacy-survey-0.49.0`. It contains exact paired random and
Zhou et al. LRG-mask indices `[0, 1, 10, 11, 12, 13, 14, 15]`. Every random and its
corresponding sole-column `uint8` `lrg_mask` table reports the same 51,738,616 rows, for
413,908,928 random rows in total. The masks contain no coordinate or object-ID columns, so
their row-order correspondence is an upstream Zhou file-pair contract rather than something
this builder can independently rederive. The write-free preflight checks pair completeness,
required random columns, mask schema and row equality, the expected per-file row count,
33 evenly spaced paired samples, and the complete finite nside-64 RING stellar-density
table. Schema v3 additionally binds every used source to
`data/xDESI/survey_data/desi_dr9/legacy-survey-0.49.0/SHA256SUMS.raw.txt`, whose exact-byte
SHA-256 is `4381d3a08b94854a9f480501ef4a3562f5533bb0c4ef62caec67b9aa9dd19c77`.
The build invocation streamed and matched all 18 full-file digests before processing any
map rows. The resulting input identity is
`0d91e56b0550d0c6bd867b011f5a6d2d90eb58dc0e93ea45f6840ec7221d6a02`; the canonical
18-entry hash-inventory digest saved in the survey manifest is
`5353911f47c3c1e141c4099ebf385bd44ad9bc2279a3fcbbbaf2b2feadc06b7e`.
The full-file stellar-density-table SHA-256 is
`bf62f084f28b1e7e7766799c3ba1f049cee9fae839d9ce4ccb66f0843713b6e4`. All eight gzip
LRG-mask streams also passed CRC validation. The ignored checksum ledger is local cluster
evidence, not a routed knowledge scope, because files below `data/` do not travel between
clones.

For this identity, the builder wrote the atomic outputs
`data/xDESI/survey_data/data/desi_dr9_imaging_randoms/desi_dr9_randoms_i0-1-10-11-12-13-14-15_0d91e56b0550_lrg_quality_provenance.h5`
and
`data/xDESI/survey_data/data/desi_dr9_imaging_randoms/desi_dr9_randoms_i0-1-10-11-12-13-14-15_0d91e56b0550_lrg_quality_count_maps_nside1024_2048_4096.h5`.
Their complete-file SHA-256 values are respectively
`17133656866a732540c75d201b13888f6be56f4dac2c1033cc3a48dca2dc7ccd` and
`3b8be76aa45a89717998b953a33cd83cd8af5db543f4f1eafad5c0b03ac492c1`.
SLURM job `6882555`, run under the guarded wrapper
`notebooks/xDESI/survey_measure/build_desi_dr9_multi_random_mask.sbatch`, completed with
exit `0:0` in 7 minutes 42 seconds. The products contain 332,967,823 quality-cut randoms.
Their native-map support/count identities are:

- nside 1024: 5,667,488 occupied pixels, SHA-256
  `b4c9248114061b942c5d076a76e182a138e3d8c8f3b1266b74a00538569df4d8`;
- nside 2048: 22,367,589 occupied pixels, SHA-256
  `291add2290953adea0fdd6d5197e888f2b6e39296e154617f9ca4e0a01bf49c5`;
- nside 4096: 85,176,008 occupied pixels, mean 3.90917385 randoms per occupied pixel,
  SHA-256 `8de46e58377a530936d890d489c3cf9e2383fc6fbdf8ef87886682c9d7a8ccd7`.

The eighth realization added 1,043, 12,051 and 1,290,129 newly occupied pixels at nside
1024, 2048 and 4096 respectively. The last increment is 1.5147% of final nside-4096
support, so native high-resolution support is improved but not asymptotically converged.
Applied to the unchanged `valid_for_cl` catalog, the final nside-4096 support retains
2,724,485, 4,625,155, 6,034,266 and 5,523,687 galaxies in pz1--pz4, or
97.4983%, 97.5136%, 97.5289% and 97.5582% of each full selected bin.

The build has four internal controls. First, all selected raw randoms, Zhou masks and two
shared inputs must match the bound full-file checksum ledger before a new product may be
built. Second, when both transferred legacy pair-0 products
are present, pair 0 must reproduce the old cut counts and the nside-1024 and nside-4096
maps bit-for-bit; a partial legacy reference or any mismatch aborts
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py`). Third, after
each added pair, cumulative nonzero-pixel support and total counts must be non-decreasing,
and each nside-1024/2048/4096 map sum must equal the cumulative quality-cut count; support,
count and mean occupied-pixel count are retained for later convergence assessment
(`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py`). Fourth, the
wrapper runs the preflight, the streamed build, and then the same command again so the
existing-product identity/schema/content-checksum reuse validator must pass
(`notebooks/xDESI/survey_measure/build_desi_dr9_multi_random_mask.sbatch`). Job
`6882555` satisfied all four: all 18 full hashes matched, pair 0 matched both legacy maps
bit-for-bit, every cumulative
sum/support assertion passed, and the immediate second invocation reused the products only
after recomputing all stored map checksums. As a null control, all three v3 count arrays are
also bit-for-bit identical to the earlier sampled-identity build.

The `highres4096` measurement config requires at least eight random realizations, and the
loader interprets an old product without realization metadata as one realization before
raising if the requirement is unmet. It additionally compares realization count, exact
indices, full-source input identity and random-product schema against the survey manifest,
requires the exact 18 expected source paths and valid 64-hex digests, compares the ledger
and inventory digests to the manifest, then checks the native nside array against its byte
checksum and count sum (`notebooks/xDESI/survey_measure/multiprobe_namaster.py`). Thus neither a legacy
one-random count map nor a different eight-random set can silently enter the new
high-resolution run. The object rows, weights and true-`n(z)` remain the same DR9 Extended
products; only the random-derived angular-selection product changed.

### Redshift and kSZ sample details

`catalog/z` is the photometric `Z_PHOT_MEDIAN` used for bin assignment; it is not the
theory lens kernel. The active theory input is
`data/desi_dr9_redshift_distributions/desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5`,
group `zphot_std0p05_spec_ratio_corrected`, dataset `nz_unit_integral`
(`notebooks/xDESI/survey_measure/multiprobe_namaster.py:82`, `:937`, `:1667`). That group
anchors to the public spectroscopic calibration, applies the measured response of the
photo-z-uncertainty cut, and is renormalized to the exact full-`valid_for_cl` surface
density (`measurements/xDESI/scripts/prepare_desi_dr9_lrg_nz.py:201`, `:224`, `:269`).
The photo-z histogram is saved only as a diagnostic.

The velocity field is source ASCII column 15 divided by `3e5 km/s`, with no sign flip
at catalog preparation (`measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py:289`,
`:327`). The active maps use the full-footprint `valid_for_cl` object list. They do **not**
apply the final kSZ-paper ACT-overlap and velocity-outlier cleaning at object level. The
transfer n(z) file contains separate `ksz_paper_scaled_counts/*` groups that represent
that paper selection only by count rescaling
(`measurements/xDESI/scripts/prepare_desi_dr9_lrg_nz.py:185`, `:392`). Those groups must
not be described as the object list used by the current harmonic-space measurement.

Separately, the survey bundle loads
`data/desi_abacus_velocity_calibration/sigma_true_gas_abacus_extended_lrg_zerr0p0_ph201_photometric_bins.json`
for the kSZ velocity-amplitude normalization (`notebooks/xDESI/survey_measure/multiprobe_namaster.py:47`,
`:146`, `:1705`). That file describes its Abacus calibration as belonging to **DESI DR10
Extended photometric bins**. It supplies `sigma_true_gas`, not galaxy rows, imaging
weights, a random mask or a lens kernel. The catalog answer is therefore still DR9
Extended, but it is not correct to claim that every ancillary calibration in the current
kSZ path is DR9-derived. The scientific compatibility of this DR10-labelled calibration
with the DR9 measurement sample is not established by this provenance audit.

### Why the DR10 catalog-preparation files are present but inactive

`prepare_act_desi_ksz_hdf5.py` is the earlier transfer/bootstrap builder: it reads
`extended_catalog_dr10_allfoot_perbin_sigmaz0.0500.txt`, writes
`desi_dr10_extended_*`, and constructs an initial DR10-only manifest
(`measurements/xDESI/scripts/prepare_act_desi_ksz_hdf5.py:49`, `:102`, `:261`, `:952`).
`compute_dr10_imaging_weights.py` likewise targets the combined DR10 catalog. Its own
metadata calls the result approximate because it applies DR9-trained coefficients to
DR10 random-derived templates and says a DR10 refit is required for production
(`measurements/xDESI/scripts/compute_dr10_imaging_weights.py:204`, `:493`). The active
downstream measurement contains no DR10 object, imaging-weight, random-mask or lens-`n(z)`
product reference. The preparation notebooks and the DES Y3 notebook retain DR10 example
paths, so their source cells document the older transfer branch, not the active estimator
choice. This statement is deliberately narrower than "no DR10-derived input" because of
the separate Abacus `sigma_true_gas` calibration above.

The other four files in `measurements/xDESI` prepare or diagnose DES Y3 shear inputs and
do not choose a DESI release (`measurements/xDESI/scripts/prepare_des_y3_shear_maps.py:2`;
`measurements/xDESI/scripts/plot_des_y3_tomo4_paper_style_covariance.py:2`). They are in
this document's scope so the complete transferred measurement-code tree has an owner and
future edits are routed through the provenance review.

## How to verify

```bash
python -m py_compile measurements/xDESI/scripts/*.py
pytest tests/test_xdesi_multiprobe_namaster.py -q \
  -k "survey_bundle or build_desi_fields or calibrated_true_nz"
/usr/bin/env PATH=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin:/usr/bin:/bin \
  python tools/kb/kb.py invariants --check --id INV-PRODUCT-PROV-01
/usr/bin/env PATH=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin:/usr/bin:/bin \
  python tools/kb/kb.py invariants --check --id INV-KSZ-CALIB-01
rg -n "spectroscopic_calibrated_true_redshift" \
  notebooks/xDESI/survey_measure/multiprobe_namaster.py \
  notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py

# The catalog loader must have DR9 references and no DR10 product key.
rg -n "desi_dr9_extended_velocity_catalogs|DR9 Extended LRG" \
  notebooks/xDESI/survey_measure/multiprobe_namaster.py
if rg -n -i "dr10" notebooks/xDESI/survey_measure/multiprobe_namaster.py; then exit 1; fi

# This ancillary kSZ calibration is the explicit DR10-labelled exception.
jq -r '.description' \
  data/xDESI/survey_data/data/desi_abacus_velocity_calibration/\
sigma_true_gas_abacus_extended_lrg_zerr0p0_ph201_photometric_bins.json

# Needs the gitignored cluster survey bundle.
jq -e '.products.desi_dr9_extended_velocity_catalogs.combined ==
  "data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5"' \
  data/xDESI/survey_data/manifest.json

# Write-free eight-pair source preflight. It must report 8 pairs, 413908928 total rows,
# nsides 1024/2048/4096, an 18-file SHA inventory and identity
# 0d91e56b0550d0c6bd867b011f5a6d2d90eb58dc0e93ea45f6840ec7221d6a02.
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py \
  --transfer-root data/xDESI/survey_data \
  --random-source-root data/xDESI/survey_data/desi_dr9/legacy-survey-0.49.0 \
  --random-indices 0 1 10 11 12 13 14 15 \
  --skip-catalogs --validate-only

# The eight compressed Zhou-mask streams must pass gzip CRC validation.
gzip -t data/xDESI/survey_data/desi_dr9/legacy-survey-0.49.0/zhou-lrg-xcorr-2023-v1/\
catalogs/lrgmask_v1.1/randoms-1-{0,1,10,11,12,13,14,15}-lrgmask_v1.1.fits.gz

# Slow raw-source null, needs the gitignored cluster inputs. The production builder runs
# the equivalent check with --verify-full-sha256 before writing schema v3.
(cd data/xDESI/survey_data/desi_dr9/legacy-survey-0.49.0 && \
  sha256sum -c SHA256SUMS.raw.txt)

bash -n notebooks/xDESI/survey_measure/build_desi_dr9_multi_random_mask.sbatch
```

Expected: all commands exit zero; the pytest selection passes; the downstream DR10 search
prints nothing; the manifest assertion returns `true`; and the random preflight prints the
exact eight-pair identity and source-row total above. These checks validate the transferred
inputs and wrapper syntax. The builder's second identical invocation must report that it is
reusing the validated products; `sacct -j 6882555` reports `COMPLETED|0:0|00:07:42`.

## Failure modes

- **Calling the measurement DR10 because a preparation notebook says DR10.** The reported
  release no longer matches any saved field metadata; a reproduction built from the claimed
  inputs cannot match the product.
- **Using the DR10 compact HDF5 in the active loader.** It lacks the production
  `valid_for_cl` and `weight_imaging_mean1` schema, so validation fails; bypassing that check
  silently changes the sample, weight and mask together.
- **Rerunning the old bootstrap manifest writer.** `prepare_act_desi_ksz_hdf5.py` writes an
  initial DR10-only `manifest.json`; the active loader then fails on the missing DR9 product
  keys. Preserve or deliberately reconstruct the later DR9 manifest entries.
- **Calling `catalog/z` a true-redshift kernel.** All galaxy cross amplitudes shift together
  and HOD parameters absorb the error; use the calibrated true-z group instead.
- **Calling the full harmonic catalog “kSZ-paper cleaned.”** The paper count scaling does not
  identify or remove individual ACT-outside or velocity-outlier objects. An apparent matched
  kSZ comparison would therefore compare different object selections.
- **Interpreting DR9 catalog provenance as DR9 calibration provenance.** The active Abacus
  `sigma_true_gas` file is labelled for DR10 Extended photometric bins. This does not change
  galaxy membership, but it can rescale a calibrated kSZ amplitude if the samples are not
  transferable.
- **Relabelling a retained one-random product as the eight-random product.** Existing
  `fast1024`/`midres2048` counts then appear to have improved mask provenance without any
  map change. The `highres4096` loader rejects products whose realization metadata is below
  eight; bypassing that guard restores the sparse-mask failure silently.
- **Treating manifest output names as sufficient provenance.** A manifest path can exist
  before build evidence does. Production must require both atomic HDF5 outputs, their encoded
  identity/schema/content checks, the pair-0 null and the cumulative diagnostics; otherwise
  a missing or stale partial map can define the galaxy selection.
- **Bypassing the manifest-to-HDF identity check.** A different count-map file can advertise
  eight realizations while encoding another set. The active loader compares the exact index
  array, sampled input identity, schema, sum and native-map checksum; bypassing it restores
  the mixed-mask failure silently.

## Open questions

- Exact object-level matching to the final kSZ-paper ACT-overlap and velocity-outlier-cleaned
  sample is not available in the transferred rows. It blocks claiming an exactly matched
  paper-sample kSZ amplitude, but it does not affect the conclusion that the measured catalog
  is DR9 Extended. Owner: `measurement-namaster`.
- Eight DR9 random/LRG-mask pairs have been built and validated, so the random-input gate for
  `highres4096` is cleared. Native nside-4096 support still grew by 1.5147% on the eighth
  realization and the mean occupied-pixel count is only 3.91. The measurement requested by
  the user may proceed, but finite-random/catalog-selection uncertainty is not included in
  the Gaussian/iNKA covariance and must remain a limitation for physical inference. The
  retained one-realization `fast1024`/`midres2048` products keep their original provisional
  mask provenance rather than inheriting the new one. Owner: `measurement-namaster`.
- The included DR10 workflow is not production-ready: its weights use DR9 coefficients on
  DR10 templates, and it has no active matched n(z)/mask/selection contract. A future DR10
  migration must be treated as a new measurement product, not a path substitution. Owner:
  `measurement-namaster`.
- The active Abacus `sigma_true_gas` calibration is labelled for DR10 Extended photometric
  bins while the measured objects are DR9 Extended. Its transferability must be validated
  before using the fixed calibration for a physical kSZ amplitude; fitting free `A_v_bin`
  remains the explicit fallback. Owner: `measurement-namaster`.
