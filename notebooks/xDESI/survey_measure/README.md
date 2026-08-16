# xDESI Multi-Probe NaMaster Pipeline

Use the conda environment that has NaMaster and pixell:

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/survey_measure/prepare_multiprobe_maps.py --stage lowres --force
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/survey_measure/measure_multiprobe_namaster.py --stage lowres --force
```

The low-resolution products are written to:

```text
data/xDESI/processed/multiprobe_namaster/lowres/
```

Production products use the same schema:

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/survey_measure/prepare_multiprobe_maps.py --stage full --force
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/survey_measure/measure_multiprobe_namaster.py --stage full --force
```

CPU-submittable production stages:

```bash
# Submit both stages independently.
notebooks/xDESI/survey_measure/submit_multiprobe_cpu.sh --stages fast1024,midres2048

# Resume the validated nside-4096, ell<=8192 pilot with the efficient covariance DAG.
# This reuses its map/spectra and enforces at most five covariance nodes.
notebooks/xDESI/survey_measure/submit_multiprobe_highres4096_efficient.sh --max-nodes 5

# Run only the fast validation product.
notebooks/xDESI/survey_measure/submit_multiprobe_cpu.sh --stages fast1024

# Submit both, but start midres2048 only after fast1024 validates.
notebooks/xDESI/survey_measure/submit_multiprobe_cpu.sh --stages fast1024,midres2048 --gate-midres-on-fast
```

The CPU production stages are:

- `fast1024`: `nside=1024`, `lmax=1024`, 10 linear NaMaster bins with edges `[8, 110, 212, 314, 415, 517, 619, 720, 822, 924, 1025]`.
- `midres2048`: `nside=2048`, science `ell=128..4096`, mask bandwidth `lmax_mask=6143`, and 16 hybrid-log NaMaster bins. The first 13 bands preserve the established through-3000 table exactly; the complete left edges are `[128, 160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693]` and right-exclusive edges are `[160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693, 4097]`. Pipeline v2 preserves the source/catalog masks and uses fixed per-field means; it does not apply the old global 1 deg C2 apodization or pair-dependent mean subtraction.
- `highres4096`: `nside=4096`, science `ell=128..8192`, mask bandwidth `lmax_mask=12287`, and 20 common hybrid-log bins. It preserves the first 13 bands through inclusive `ell=3000`; its seven added right-exclusive intervals are `[3001,3464)`, `[3464,3998)`, `[3998,4615)`, `[4615,5327)`, `[5327,6149)`, `[6149,7098)`, and `[7098,8193)`. This stage requires a random-count product that records at least eight independent DR9 realizations and confirmed ACT-temperature units of `uK_CMB`.

These stages use native DES shear maps at the requested `nside`, including
`data/des_y3_shear_maps/des_y3_metacal_shear_maps_nside2048.h5` for
`midres2048` and the nside-4096 counterpart for `highres4096`. The split production driver writes spectra first, then computes
Gaussian covariance shards grouped by reduced mask/spin keys, and finally
assembles the full covariance: `(460, 460)` for `fast1024` and `(736, 736)` for
`midres2048`. The high-resolution HDF5 archives a full `(920,920)` raw-estimator
Gaussian/iNKA covariance. Exactly 28 packed data-vector cells -- seven bands in
each of the four DESI-galaxy x ACT-kappa spectra -- are zero placeholders, so
the active likelihood vector and ordinary principal covariance submatrix have
sizes 892 and `(892,892)`. After validation, the terminal
plot job writes separate multi-page `C_ell` and `D_ell` PDFs plus one family
PNG per probe family. Every error bar is the square root of the corresponding
diagonal element of the saved joint covariance; the same positive
`ell(ell+1)/(2 pi)` factor transforms both the mean and error from `C_ell` to
`D_ell`. The kSZ `C_ell` plot remains raw `C_ell^{pi,T}`, while its `D_ell`
plot uses the documented paper-style `-D_ell^{pi,T}` display convention.
The primary production plots apply no ell cut or fixed kSZ y-limit, use a log
x-axis for hybrid-log products, and refuse unknown/missing probe families. The
ACT kappa transfer is exactly zero from ell 3001 in the transferred input. Raw
high-ell galaxy x kappa estimator values remain archived under
`spectra/<name>/cl` and `joint/data_vector_raw`; `joint/data_vector` stores exact
zeros there and `joint/data_vector_valid` marks them false. Production plots and
likelihoods select the saved validity mask and therefore do not interpret those
transfer-null bands as measurements.

The speed-optimized true-n(z) midres submission is:

```bash
notebooks/xDESI/survey_measure/submit_multiprobe_midres_true_nz_cpu.sh
```

Its default covariance profile is one group per exclusive Rome node, at most
96 spin-2 plus 10 scalar array tasks concurrently. Science tasks request 128
CPUs and 128 GiB; prepare/spectra, covariance, and downstream hard limits are
2, 4, and 1 hours respectively. The submission is resumable by exact product,
manifest, map, group, mask, and source identities and does not use `--force`.

The resume-only `highres4096` production packs 10--11 independent covariance
groups onto one exclusive Rome node. Each group runs in a disjoint
`srun --exclusive --exact` step with 11 CPUs and 80 GiB; the node allocation is
121 CPUs/880 GiB/4 h. A worst-case one-node stress bundle must succeed before
the remaining 23 bundles are eligible, and the main array throttle is `%5`.
The 2.8-GiB map is staged once per node. Every group uses
`--no-cov-workspace-cache`; caching 259 unique workspaces would otherwise
project to roughly 1 TiB with no cross-group reuse. Assembly, validation, HMC
loader checks and plots share one `genx` job at 2 CPUs/4 GiB/30 min. The
2026-08-15 production job chain is `6884226 -> 6884227 -> 6884228`; it reuses
the validated map, spectra and three pilot shards and submits no preparation or
spectra work.

At submission, the efficient driver hashes all six runtime estimator/driver/
loader/worker/submission files plus the work-plan file. Workers recompute both
digests before doing work. The plan binds all 259 manifest groups, the map
identity, spectra SHA256 and reused-shard hashes; finalization re-attests those
frozen inputs before assembly. A stage lock and pre-write active-job check stop
a duplicate invocation from rewriting the plan used by a queued DAG.

Pipeline-v2 map products, covariance workspaces, manifests, and shards carry a
`_pipev2` suffix. Spectra and final measurement products use `_pipev2_gshot` to
make the galaxy-auto mean convention explicit: those bandpowers retain catalog
shot noise exactly once. Reusing the existing `_pipev2` map/covariance artifacts
is intentional because their raw-map iNKA inputs already include that same noise.
The final HDF keeps this total `C_ell^gg+SN` vector as the default/HMC view and
also saves `joint/views/weighted_poisson_subtracted`. That alternate subtracts
the saved decoupled template in only the four galaxy autos. Both view groups
hard-link the same `joint/cov`: conditional on the fixed catalog/mask template,
a deterministic mean translation leaves the full covariance and all cross
blocks unchanged. Do not recompute a signal-only covariance for the alternate.
Legacy or mismatched products are otherwise rejected: map construction,
covariance configuration, manifest, group, map-product, mask-byte, and
galaxy-auto-mean identities are checked during production and assembly.
Existing signal-only pipeline-v2 measurements can be preserved and converted without a map
or covariance rerun using `migrate_galaxy_auto_shot_noise.py SOURCE`; it creates a new
`_pipev2_gshot.h5` file, archives the exact old galaxy-auto arrays inside it, and audits that
every covariance-related dataset and all 42 non-auto spectra remain exactly unchanged. It
never overwrites the source or an existing destination.
The map-product ID is content-addressed over every saved mask, map, and catalog
array, each field's estimator metadata (`kind`, `spin`, mask reference, noise/masking
policy), the saved n(z) metadata, and the input/config metadata. Loading verifies those bytes, spectra
reuse requires the exact originating map ID and construction configuration, and
each covariance shard is independently bound to its representative mask bytes.
The generic NumPy `theory_to_data_vector` wrapper rejects legacy products by default and
requires the measurement HDF5 and separate map/n(z) HDF5 to share that exact map-product ID.
Its `allow_legacy_product: true` setting is an explicit historical-only opt-in. The Stage-31
HMC/likelihood path has no legacy opt-in: it hard-requires the current `_pipev2_gshot`
measurement and matching `_pipev2` map/n(z) product.
Validity-mask products use measurement schema `xdesi_multiprobe_measurement_v2`
and the additional `_gkell3000_dvvalidv1` tag. The generic theory wrapper returns
only active elements by default; archive-layout diagnostics must opt in explicitly.

The target data vector contains 46 spectra: 10 DES shear EE spectra, 4 y x shear-E spectra, 4 DESI galaxy autos, 4 DESI galaxy x y spectra, 16 DESI galaxy x shear-E spectra, 4 DESI galaxy x ACT kappa spectra, and 4 DESI momentum x ACT temperature kSZ spectra.

Open `lowres_multiprobe_diagnostics.ipynb` after the low-resolution products exist. It reads only the cached HDF5 products.

Important convention and scale notes:

- DES spin-2 maps are read from the transferred `gamma1` and `gamma2_namaster` datasets, then multiplied by `shear_e_to_kappa_sign=-1` by default before constructing the NaMaster field. This leaves all DES shear EE spectra unchanged, but makes scalar x shear-E spectra use the same sign as scalar x positive-convergence theory.
- The theory wrapper assumes DES shear theory spectra are in this positive-convergence E-mode convention and uses the saved field metadata to convert to the measurement convention.
- The low-resolution product uses the DES Y3 harmonic-space paper's fiducial NaMaster binning for all spectra: `nside=1024`, 32 equal-weight bandpowers with edges uniformly spaced in `sqrt(ell)` over `ell=8..2048`. This is not logarithmic binning and not linear-width binning; it matches the DES transfer product's stored bandpower edge rule. DES shear and DESI map fields retain their appropriate HEALPix pixel response. Catalog-momentum and harmonically reprojected ACT fields do not receive a spurious output HEALPix pixel window in the theory wrapper.
- DES shear uses the original weighted-count mask without apodization and subtracts the catalog-derived constant coupled shape-noise pseudo-`C_ell` matched to that mask. Production reads the normalized `mask_weight`; it is the same weighted estimator as literal `mask_weight_raw` up to a constant mask rescaling, with the saved noise level rescaled by the square of that constant. This follows the DES Y3 harmonic-space estimator: apodizing the mask's empty pixels destroys effective area, and estimating noise from a high-ell data plateau removes signal.
- Gaussian covariance blocks are computed in decoupled bandpower space with `nmt.gaussian_covariance(..., coupled=False)`. Ordinary map-field inputs use NaMaster's data-derived improved narrow-kernel approximation (`nmt.get_iNKA_cell`), retaining all coupled spin components. kSZ catalog-momentum inputs use the equivalent coupled pseudo-`C_ell` divided by mask overlap, with catalog zero-lag `Nf` added back only for literal momentum autos. Do not use `coupled=True` for saved bandpower covariance; in this NaMaster version it returns full coupled-ell pseudo-spectrum covariance. Flattened NaMaster covariance arrays are band-major, so component blocks are extracted with `cov.reshape(n_band, n_comp_a, n_band, n_comp_b)[:, comp_a, :, comp_b]`.
- DES shear same-bin shape noise is subtracted from the saved shear mean auto-spectra. DESI same-bin weighted galaxy shot noise is instead retained exactly once in the saved galaxy-auto means. The raw map autos contain both signal and noise, so the default iNKA covariance inputs are already total spectra and receive no subtraction or second add-back. The kSZ momentum auto covariance input explicitly restores NaMaster's catalog `Nf` zero-lag term. No ACT or kSZ noise is subtracted from the saved data vector. Each saved `input_cls_for_covariance/*` dataset records its spin labels and noise policy. For theory comparison, first apply the saved bandpower window to the signal-only galaxy clustering theory and then add `A_shot,pz * noise_decoupled_all_components[component]`; do not send a flat shot term through the signal transfer/window a second time.
- Run `diagnose_des_shear_harmonic.py` for a DES-only paper-style check. The diagnostic compares the old quick `ell<=1024` convention, the new `ell<=2048` convention, and a DES-paper-like raw-mask, pixel-window-deconvolved shear-only convention; the much cleaner upper-right panel in Fig. 4 is the non-tomographic all-source-bin combination, which is not part of the 46-spectrum tomographic data vector.
- Open `des_shear_tomo44_fig4_check.ipynb` for the quick source-bin 4x4 check against the Fig. 4 convention. It runs only one shear EE spectrum with the paper-like raw-mask, pixel-window-deconvolved setup and saves `data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.{json,png}`. The equivalent CLI is:
  ```bash
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/survey_measure/diagnose_des_shear_harmonic.py --single-pair --scenario paper_like_raw_mask_pixwin_ell8_2048 --output data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.json --plot-output data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.png
  ```
- Open `ksz_lowres_diagnostics.ipynb` for a focused kSZ estimator check. It reads the `nside=1024, lmax=2048` map product, recomputes only the four DESI velocity-momentum x ACT temperature spectra plus shuffled-velocity nulls, saves `ksz_lowres_diagnostic.{json,png}` and `ksz_lowres_jointmean_diagnostic.{json,png}`, and reports full-covariance chi-square/PTE null tests. The notebook deliberately does not call `sqrt(d^T C^-1 d)` a detection S/N; a kSZ amplitude S/N requires a theory/template vector.
- The low-resolution product is useful for smoke tests and most large-scale cross-correlations, but it is still not a full high-ell kSZ validation product. The harmonic-space kSZ reference analysis fits roughly `1000 < ell < 7000`; `ell_max=2048` covers only the low end of that range. Use the full/high-ell run to validate kSZ SNR and shape.

Inputs and nuisance metadata now saved with the products:

- DESI galaxy and kSZ fields use the DR9 Extended LRG weighted catalog `data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5`. The default selection is `catalog/valid_for_cl`, and the imaging systematic weight is `catalog/weight_imaging_mean1`.
- DESI angular masks for the new high-resolution run use the identity-tagged eight-realization DR9 product `data/desi_dr9_imaging_randoms/desi_dr9_randoms_i0-1-10-11-12-13-14-15_0d91e56b0550_lrg_quality_count_maps_nside1024_2048_4096.h5`. Its source indices are `[0,1,10,11,12,13,14,15]`; the companion provenance HDF5 records per-pair cuts and cumulative support. Schema v3 binds the exact 18-file source inventory to `desi_dr9/legacy-survey-0.49.0/SHA256SUMS.raw.txt` and attests that every full-file hash was verified before the map was built. The high-resolution loader cross-checks that ledger and inventory against the survey manifest and fails closed unless the product records at least eight realizations. Historical one-random products remain provenance references only. The pipeline reads the native nside group when available; older `midres2048` products may record a sum-preserving `nside4096 -> nside2048` downgrade.
- DESI `delta_g`, DESI x shear/y/kappa, and the kSZ velocity-momentum template all use the same imaging weights. The kSZ spectra are measured with `pymaster.NmtFieldCatalogMomentum` from the saved `catalog/{ra_deg,dec_deg,weight,field}` arrays in each `pi{i}` field; the pixelized `pi` maps in the map product are diagnostics only. Regenerate cached map products made before the NaMaster 2.7 update, because older caches do not contain these catalog arrays. The saved DESI `nz/desi/nz_dndz_by_pz` is the calibrated true-redshift theory `dN/dz` from `desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5`, group `zphot_std0p05_spec_ratio_corrected`, for the full `valid_for_cl` catalog. Catalog `Z_PHOT_MEDIAN` histograms are saved separately under `nz/desi_photoz_diagnostic` and must not be used as theory kernels.
- To regenerate the corrected `fast1024` product quickly for theory comparison, use `bash notebooks/xDESI/survey_measure/submit_multiprobe_true_nz_cpu.sh`. This wrapper writes to `data/xDESI/processed/multiprobe_namaster_true_nz`, runs scalar covariance first, then fans the expensive spin-2 covariance over 8 full CPU nodes.
- DESI galaxy auto spectra retain the exact conditional Poisson noise of the weighted masked field, `Omega_pix * sum(w^2) / [N_pix (alpha random_mean)^2]`, exactly once in the saved mean. The estimator decouples that constant coupled pseudo-`C_ell` with the same workspace and saves the resulting bandpower response as `noise_decoupled_all_components`; this is the template multiplied by a fixed or fitted `A_shot,pz` after windowing the signal-only theory. The former full-sky-equivalent `area_sr * sum(w^2) / sum(w)^2` is retained as provenance but is not coupled through the variable random-count mask a second time.
- DES Y3 source-bin n(z) curves are loaded from `/mnt/ceph/users/spandey/GODMAX/data/DESxACT/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits` HDU `nz_source`. Products save raw FITS bin values and normalized theory `dN/dz` under `nz/des_shear`.
- DES Y3 Gaussian priors are saved under `priors/des_y3_gaussian`: `Delta_z_bias_bin{1..4}` sigmas `[0.018, 0.015, 0.011, 0.017]`, and `mult_shear_bias_bin{1..4}` means/sigmas `[(-0.006, 0.009), (-0.020, 0.008), (-0.024, 0.008), (-0.037, 0.008)]`.
- ACT masks are reprojected with bounded local spline interpolation; harmonic mask reprojection is forbidden because ringing creates false footprint support. The available ACT kappa map is already masked, so its NaMaster field uses `masked_on_input=True` and its source mask is not applied twice. ACT y and ACT CMB temperature theory curves are multiplied by a 1.6 arcmin Gaussian beam by default before the saved NaMaster bandpower windows. The high-resolution product records the user-confirmed ACT temperature units as `uK_CMB`. Pass an extra `transfer_functions["y"]` or `transfer_functions["T"]` only if using additional map-specific filtering beyond this beam.
- Production stages reproject ACT sky maps from their native CAR resolution (`act_downgrade=1`). The former CAR block average introduced an anisotropic pixel response that was not represented by the scalar theory transfer.
- kSZ uses the photometric DESI velocity-reconstruction calibration `r=0.3` from `papers/ksz/2407.07152v2.pdf`, the saved per-bin imaging-weighted reconstructed velocity RMS, and Abacus `sigma_true_gas/c = [0.00105580879, 0.00104915865, 0.00103582548, 0.00101760550]` from `data/xDESI/survey_data/docs/DESI_ABACUS_SIGMA_TRUE_GAS.md`. You can still override with `ksz_sigma_true_over_c` or fit free `A_v_bin` amplitudes.
- kSZ sign convention: the measured harmonic data vector is the raw `C_ell^{pi,T_uK}` with `pi` built from the supplied positive `vr_over_c` catalog column. With the paper convention, positive gas corresponds to `C_ell^{pi,T} = -r sigma_true sigma_rec C_ell^{tau,g}`. Paper-style positive plots therefore show `D_ell^kSZ = -ell(ell+1) C_ell^{pi,T}/(2*pi)`. The high-ell diagnostic HDF5 stores both `dl_raw_piT` and `dl_paper_ksz` explicitly.

Known limitations before final production MCMC:

- The eight-realization DR9 count map passed its build-time full-source checksum gate, pair-0 bitwise null, count conservation, output checksums, cumulative-support audit, and a no-write reuse validation. Eight realizations reduce the known one-random sparsity, but finite-random/catalog-selection uncertainty still requires mock or resampling validation before final inference; mask apodization is not a substitute.
- The saved covariance is Gaussian/iNKA only. Connected non-Gaussian, super-sample, foreground, and catalog-selection covariance terms are not included and need simulation or mock validation before final physical inference.
- ACT temperature units are confirmed as `uK_CMB` for the high-resolution product. The published kSZ velocity cleaning/symmetrization is still not reproduced by the transferred catalog, so that remains an amplitude-systematics check.
