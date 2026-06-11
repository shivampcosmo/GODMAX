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

# Run only the fast validation product.
notebooks/xDESI/survey_measure/submit_multiprobe_cpu.sh --stages fast1024

# Submit both, but start midres2048 only after fast1024 validates.
notebooks/xDESI/survey_measure/submit_multiprobe_cpu.sh --stages fast1024,midres2048 --gate-midres-on-fast
```

The CPU production stages are:

- `fast1024`: `nside=1024`, `lmax=1024`, 10 linear NaMaster bins with edges `[8, 110, 212, 314, 415, 517, 619, 720, 822, 924, 1025]`.
- `midres2048`: `nside=2048`, `lmax=4096`, 10 linear NaMaster bins with edges `[8, 417, 826, 1235, 1644, 2053, 2462, 2871, 3280, 3689, 4097]`.

These stages use native DES shear maps at the requested `nside`, including
`data/des_y3_shear_maps/des_y3_metacal_shear_maps_nside2048.h5` for
`midres2048`. The split production driver writes spectra first, then computes
exact covariance shards grouped by reduced mask/spin keys, and finally
assembles the full `(460, 460)` covariance.

The target data vector contains 46 spectra: 10 DES shear EE spectra, 4 y x shear-E spectra, 4 DESI galaxy autos, 4 DESI galaxy x y spectra, 16 DESI galaxy x shear-E spectra, 4 DESI galaxy x ACT kappa spectra, and 4 DESI momentum x ACT temperature kSZ spectra.

Open `lowres_multiprobe_diagnostics.ipynb` after the low-resolution products exist. It reads only the cached HDF5 products.

Important convention and scale notes:

- DES spin-2 maps are read from the transferred `gamma1` and `gamma2_namaster` datasets, then multiplied by `shear_e_to_kappa_sign=-1` by default before constructing the NaMaster field. This leaves all DES shear EE spectra unchanged, but makes scalar x shear-E spectra use the same sign as scalar x positive-convergence theory.
- The theory wrapper assumes DES shear theory spectra are in this positive-convergence E-mode convention and uses the saved field metadata to convert to the measurement convention.
- The low-resolution product now uses the DES Y3 harmonic-space paper's fiducial NaMaster binning for all spectra: `nside=1024`, 32 equal-weight bandpowers with edges uniformly spaced in `sqrt(ell)` over `ell=8..2048`. This is not logarithmic binning and not linear-width binning; it matches the DES transfer product's stored bandpower edge rule. The multi-probe product still uses normalized masks by default and keeps HEALPix pixel windows in the measured spectra so theory can be filtered externally before applying the saved NaMaster windows.
- Gaussian covariance blocks are computed in decoupled bandpower space with `nmt.gaussian_covariance(..., coupled=False)`. Ordinary map-field input spectra are full-ell total spectra built from decoupled measured bandpowers with auto-noise added back, then log-smoothed/sanitized for positive auto components and expanded as constant-in-band full-ell spectra. kSZ catalog-momentum inputs follow the NaMaster kSZ tutorial convention: coupled pseudo-`C_ell` divided by the appropriate mask-overlap `fsky`, with the catalog zero-lag `Nf` term added back for momentum autos. Do not use `coupled=True` for saved bandpower covariance; in this NaMaster version it returns full coupled-ell pseudo-spectrum covariance. Flattened NaMaster covariance arrays are band-major, so component blocks are extracted with `cov.reshape(n_band, n_comp_a, n_band, n_comp_b)[:, comp_a, :, comp_b]`.
- The explicit covariance noise templates are DES shear same-bin shape noise and DESI same-bin weighted galaxy shot noise. The kSZ momentum auto covariance input also includes NaMaster's catalog `Nf` zero-lag term. ACT y, ACT temperature, ACT kappa, and all cross-pair covariance inputs use measured auto/cross spectra as data-derived total spectra; no ACT or kSZ noise is subtracted from the saved data vector. Each saved `input_cls_for_covariance/*` dataset records its spin labels and noise policy.
- Run `diagnose_des_shear_harmonic.py` for a DES-only paper-style check. The diagnostic compares the old quick `ell<=1024` convention, the new `ell<=2048` convention, and a DES-paper-like raw-mask, pixel-window-deconvolved shear-only convention; the much cleaner upper-right panel in Fig. 4 is the non-tomographic all-source-bin combination, which is not part of the 46-spectrum tomographic data vector.
- Open `des_shear_tomo44_fig4_check.ipynb` for the quick source-bin 4x4 check against the Fig. 4 convention. It runs only one shear EE spectrum with the paper-like raw-mask, pixel-window-deconvolved setup and saves `data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.{json,png}`. The equivalent CLI is:
  ```bash
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python notebooks/xDESI/survey_measure/diagnose_des_shear_harmonic.py --single-pair --scenario paper_like_raw_mask_pixwin_ell8_2048 --output data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.json --plot-output data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.png
  ```
- Open `ksz_lowres_diagnostics.ipynb` for a focused kSZ estimator check. It reads the `nside=1024, lmax=2048` map product, recomputes only the four DESI velocity-momentum x ACT temperature spectra plus shuffled-velocity nulls, saves `ksz_lowres_diagnostic.{json,png}` and `ksz_lowres_jointmean_diagnostic.{json,png}`, and reports full-covariance chi-square/PTE null tests. The notebook deliberately does not call `sqrt(d^T C^-1 d)` a detection S/N; a kSZ amplitude S/N requires a theory/template vector.
- The low-resolution product is useful for smoke tests and most large-scale cross-correlations, but it is still not a full high-ell kSZ validation product. The harmonic-space kSZ reference analysis fits roughly `1000 < ell < 7000`; `ell_max=2048` covers only the low end of that range. Use the full/high-ell run to validate kSZ SNR and shape.

Inputs and nuisance metadata now saved with the products:

- DESI galaxy and kSZ fields use the DR9 Extended LRG weighted catalog `data/desi_dr9_extended_velocity_catalogs/desi_dr9_extended_all_pz_compact_with_weights.h5`. The default selection is `catalog/valid_for_cl`, and the imaging systematic weight is `catalog/weight_imaging_mean1`.
- DESI angular masks use the DR9 quality-cut random-count HEALPix maps in `data/desi_dr9_imaging_randoms/desi_dr9_randoms_1_0_lrg_quality_count_maps_nside1024_4096.h5`. The pipeline reads `nside1024/random_count` or `nside4096/random_count` directly when available. For `midres2048`, if no native `nside2048/random_count` exists, it uses a sum-preserving `nside4096 -> nside2048` downgrade and records that derivation in output metadata.
- DESI `delta_g`, DESI x shear/y/kappa, and the kSZ velocity-momentum template all use the same imaging weights. The kSZ spectra are measured with `pymaster.NmtFieldCatalogMomentum` from the saved `catalog/{ra_deg,dec_deg,weight,field}` arrays in each `pi{i}` field; the pixelized `pi` maps in the map product are diagnostics only. Regenerate cached map products made before the NaMaster 2.7 update, because older caches do not contain these catalog arrays. The saved DESI `nz/desi/nz_dndz_by_pz` is the calibrated true-redshift theory `dN/dz` from `desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5`, group `zphot_std0p05_spec_ratio_corrected`, for the full `valid_for_cl` catalog. Catalog `Z_PHOT_MEDIAN` histograms are saved separately under `nz/desi_photoz_diagnostic` and must not be used as theory kernels.
- To regenerate the corrected `fast1024` product quickly for theory comparison, use `bash notebooks/xDESI/survey_measure/submit_multiprobe_true_nz_cpu.sh`. This wrapper writes to `data/xDESI/processed/multiprobe_namaster_true_nz`, runs scalar covariance first, then fans the expensive spin-2 covariance over 8 full CPU nodes.
- DESI galaxy auto spectra subtract weighted Poisson shot noise, `N_ell = area_sr * sum(w^2) / sum(w)^2`, saved in each `g{i}` field metadata.
- DES Y3 source-bin n(z) curves are loaded from `/mnt/ceph/users/spandey/GODMAX/data/DESxACT/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits` HDU `nz_source`. Products save raw FITS bin values and normalized theory `dN/dz` under `nz/des_shear`.
- DES Y3 Gaussian priors are saved under `priors/des_y3_gaussian`: `Delta_z_bias_bin{1..4}` sigmas `[0.018, 0.015, 0.011, 0.017]`, and `mult_shear_bias_bin{1..4}` means/sigmas `[(-0.006, 0.009), (-0.020, 0.008), (-0.024, 0.008), (-0.037, 0.008)]`.
- ACT y and ACT CMB temperature theory curves are multiplied by a 1.6 arcmin Gaussian beam by default before the saved NaMaster bandpower windows. Pass an extra `transfer_functions["y"]` or `transfer_functions["T"]` only if using additional map-specific filtering beyond this beam.
- kSZ uses the photometric DESI velocity-reconstruction calibration `r=0.3` from `papers/ksz/2407.07152v2.pdf`, the saved per-bin imaging-weighted reconstructed velocity RMS, and Abacus `sigma_true_gas/c = [0.00105580879, 0.00104915865, 0.00103582548, 0.00101760550]` from `data/xDESI/survey_data/docs/DESI_ABACUS_SIGMA_TRUE_GAS.md`. You can still override with `ksz_sigma_true_over_c` or fit free `A_v_bin` amplitudes.
- kSZ sign convention: the measured harmonic data vector is the raw `C_ell^{pi,T_uK}` with `pi` built from the supplied positive `vr_over_c` catalog column. With the paper convention, positive gas corresponds to `C_ell^{pi,T} = -r sigma_true sigma_rec C_ell^{tau,g}`. Paper-style positive plots therefore show `D_ell^kSZ = -ell(ell+1) C_ell^{pi,T}/(2*pi)`. The high-ell diagnostic HDF5 stores both `dl_raw_piT` and `dl_paper_ksz` explicitly.

Known staged inputs before final production MCMC:

- More DR9 random realizations, or an explicit smoothing/apodization choice, for a less sparse raw `nside=4096` DESI high-ell mask. The current transferred product uses one random realization and is recorded as provisional in output metadata.
