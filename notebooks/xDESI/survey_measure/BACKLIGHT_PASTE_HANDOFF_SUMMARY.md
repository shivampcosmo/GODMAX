# xDESI GODMAX Multi-Probe Handoff For Backlight Paste-Map Work

Date: 2026-06-04

This file summarizes the work done in the xDESI multi-probe GODMAX thread and is intended as context for the next step: using the current best-fit astrophysical/HOD parameters to paste or paint maps in the Abacus Backlight simulations, following the direction started in `notebooks/xDESI/abacus_quick_paste_validation.ipynb`.

## Objective

We built an end-to-end comparison between the xDESI multi-probe NaMaster measurements and analytic GODMAX 2pt theory, then fit a fixed-cosmology Stage-31 astrophysical/HOD model. The current best-fit parameter point has been saved as a normal GODMAX params YAML and should be the v1 input for Backlight paste-map validation.

Primary v1 best-fit params file:

```text
param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v1.yaml
```

This is a full GODMAX-style params file with top-level:

```yaml
sim_params:
other_params:
halo_params:
analysis:
```

It was copied from the combined 16-chain HMC v1 output and corresponds to:

```text
best whitened chi2 = 7346.232641444647
```

The fit is a large improvement over the fiducial point but is not yet statistically good. With 460 data-vector elements, covariance rank 459, and 31 varied parameters, a good fit would be expected around chi2 ~ 459 - 31 = 428, up to order sqrt(2 * 428) scatter.

## Measurement Products

The target measurement is the fast1024 true-n(z) multi-probe product:

```text
data/xDESI/processed/multiprobe_namaster_true_nz/fast1024/xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear.h5
data/xDESI/processed/multiprobe_namaster_true_nz/fast1024/xdesi_multiprobe_maps_nside1024_lmax1024_nbin10_linear.h5
```

Stage settings:

```text
nside = 1024
lmax = 1024
nbin = 10
binning = linear
data-vector length = 460
number of spectra = 46
covariance rank after eigencut = 459
```

The measured spectra are:

- DES Y3 shear x shear EE: 10 spectra
- ACT y x DES shear E: 4 spectra
- DESI galaxy auto, one per photometric bin: 4 spectra
- DESI galaxy x ACT y: 4 spectra
- DESI galaxy x DES shear E: 16 spectra
- DESI galaxy x ACT CMB kappa: 4 spectra
- DESI kSZ momentum x ACT temperature: 4 spectra

The measurement convention is raw bandpower `C_ell` in the HDF5 data vector. Plotting converts most spectra to `D_ell = ell(ell+1) C_ell / 2pi`; galaxy autos are usually left as signal-only `C_ell`; kSZ plots use the positive convention `-D_ell^{pi,T}` or `-10^3 D_ell^{pi,T}`.

## DESI Photometric Catalog And n(z)

The galaxy maps use the full DR9 Extended LRG photometric CL sample:

```text
catalog/valid_for_cl
catalog/weight_imaging_mean1
DR9 random-count mask
```

This sample is used consistently for:

- galaxy overdensity maps
- galaxy x shear/y/kappa spectra
- kSZ momentum spectra

Important correction made during this thread:

- The map-making sample remains the photometric catalog selected by `valid_for_cl`.
- The theory lens kernel must not use the `Z_PHOT_MEDIAN` histogram.
- The theory lens kernel now uses calibrated true-redshift n(z) for each photometric bin.

The calibrated true-redshift n(z) source is:

```text
data/xDESI/survey_data/data/desi_dr9_redshift_distributions/desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5
group: zphot_std0p05_spec_ratio_corrected
```

The new map product stores the theory kernel here:

```text
nz/desi/z_mid
nz/desi/z_edges
nz/desi/dz
nz/desi/nz_dndz_by_pz
```

The old photo-z diagnostic histogram is kept separately under a diagnostic path and should not be used for theory.

The HMC/theory code explicitly checks:

```text
desi_lens_redshift_kind = spectroscopic_calibrated_true_redshift
```

## Theory Configuration

Main comparison config:

```text
param_files/xDESI/params_multiprobe_fast1024_true_nz_theory.yaml
```

This is a path-backed config that merges:

```text
param_files/params_default.yaml
param_files/xDESI/params_fit_abacus.yaml
```

with fixed comparison overrides.

Fixed cosmology used for the analytic comparisons and HMC:

```yaml
flat: true
H0: 67.36
Om0: 0.30
Ob0: 0.0493
sigma8: 0.80
ns: 0.9649
w0: -1.0
```

Important theory settings:

```yaml
analysis.hod_params_model: perbin
analysis.gg_transition_model: poweradd
analysis.beam_fwhm_arcmin: 0.0
analysis.zmin_for_Cls: 0.005
analysis.zmax_for_Cls: 3.0
analysis.nz_for_Cls: 192
halo_params.zmin: 0.005
halo_params.zmax: 3.0
halo_params.nz: 96
```

The ACT beams, pixel windows, masks, transfer functions, shear sign, shear m-bias, and kSZ conversion are applied through the measurement/theory wrapper, not directly by smooth GODMAX curves.

The exact theory-to-data path is:

```python
theory_to_data_vector(
    measurement_h5,
    theory_cls,
    ell=ell_theory,
    shear_m_bias=saved_m_means,
    ksz_velocity_correlation=0.3,
    include_default_pixel_windows=True,
    include_default_act_beams=True,
    theory_shear_e_is_positive_kappa=True,
)
```

Do not compare smooth theory at `ell_eff` as the main result. The accepted comparison is windowed theory in the saved 460-element measurement convention.

## Key Theory/HMC Code Added

Main helper module:

```text
notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py
```

Responsibilities:

- load path-backed comparison configs
- materialize DES source and DESI lens true-n(z) inputs
- compute DESI comoving abundance targets
- build GODMAX models
- extract theory spectra into measurement-compatible keys
- call `theory_to_data_vector`
- plot per-family comparison panels

Main Stage-31 HMC module:

```text
notebooks/xDESI/survey_measure/godmax_multiprobe_hmc_stage31.py
```

Responsibilities:

- define the 31 varied parameters
- pack/unpack named samples
- build per-photometric-bin GODMAX galaxy theory blocks
- compute the JAX-native 460-element theory vector
- compute whitened full-covariance chi2
- run NumPyro NUTS
- save chains, best-fit params, best-fit theory vector, summaries, and plots

Important plotting script:

```text
notebooks/xDESI/survey_measure/plot_stage31_bestfit_vs_fiducial_cls.py
```

This reads saved windowed vectors and makes data/fiducial/best-fit overlays without rerunning GODMAX.

## Photometric-Bin HOD Fix

The true-redshift distributions for photometric pz bins overlap. Therefore, it is wrong to assign per-pz HOD parameters using disjoint true-z support intervals.

The fix implemented:

- Build one shared non-galaxy WL/CMB theory block for shear/y-only spectra.
- For each DESI photometric pz bin, build separate galaxy theory blocks using that pz bin's HOD parameters and that pz bin's true-n(z)/nbar(z).
- Assemble the 46-spectrum theory vector from these per-pz blocks.

This matters for Backlight map work. The simulated/pasted galaxy maps should reproduce four photometric-bin tracers whose true-z kernels overlap. Do not treat the pz bins as disjoint true-redshift slices.

For map-level work there are two reasonable implementations:

1. Catalog-like approach: assign simulated galaxies into photometric bins probabilistically so that each pz bin recovers the calibrated true-n(z) and measured angular density.
2. Map/paste approach: create four projected galaxy tracer maps using the four true-n(z) kernels and their corresponding per-pz HOD parameters.

The second approach is closer to the analytic GODMAX comparison and may be simpler for a paste-map validation.

## kSZ Convention

The measured kSZ spectra are raw:

```text
C_b^{pi,T_uK}
```

Theory supplies only:

```text
C_ell^{g,tau}
```

The wrapper applies:

```text
C_ell^{pi,T_uK} = -T_CMB_uK * A_v_bin * C_ell^{g,tau}
```

where `A_v_bin` uses:

- saved photometric velocity calibration
- `r = 0.3`
- saved `rms_rec_vr_over_c_weighted`
- saved Abacus `sigma_true_gas/c`

Internally, positive `C_ell^{g,tau}` gives negative raw `C_ell^{pi,T}`. Plots use the positive convention:

```text
-D_ell^{pi,T}
```

For Backlight kSZ paste-map work, keep this distinction clear:

- raw estimator/sign convention for measured vector
- positive plotted convention for diagnostics
- same velocity calibration if comparing to the current measurement product

## Stage-31 Fit

HMC config:

```text
param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml
```

Prior file:

```text
param_files/xDESI/priors_multiprobe_fast1024_hmc_stage31.yaml
```

The Stage-31 parameter vector varies:

Global baryonic scalars:

```text
log10_Mstar0_theta_ej
theta_ej_0
nu_theta_ej_M
nu_theta_ej_z
log10_Mc0
mu_beta
alpha_nt
```

Per-pz-bin HOD entries for pz bins 1-4, leaving HOD array entry 0 fixed:

```text
log10M1_fshmr_array[1:5]
log10M1_a_fshmr_array[1:5]
delta_fshmr_array[1:5]
gamma_fshmr_array[1:5]
siglogMstar_Ncen_array[1:5]
alphasat_Nsat_array[1:5]
```

Fixed:

- cosmology
- DES shear calibration and source redshift shifts
- IA defaults inherited from xDESI params
- omitted HOD arrays
- fixed-zero HOD evolution arrays
- analysis settings except explicit comparison overrides

The likelihood is full-covariance whitened chi2:

```text
chi2 = || W (data - theory) ||^2
```

where `W` is built from the covariance correlation-matrix eigendecomposition with eigenvalue threshold `1e-8`. For fast1024 this keeps rank 459 out of 460 modes.

## Fit Outputs And Diagnostics

Fiducial true-n(z) theory vector:

```text
notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz/theory_data_vector_fast1024.npz
```

V1 multigpu HMC combined output:

```text
notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/stage31_hmc_400x16_v1/combined/
```

Key v1 files:

```text
chain_stage31_multigpu.npz
bestfit_params_stage31_multigpu.yaml
bestfit_theory_data_vector_stage31_multigpu.npz
fit_summary_stage31_multigpu.json
posterior_predictive_comparison_stage31_multigpu.pdf
```

Data/fiducial/v1-best overlay:

```text
notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/stage31_hmc_400x16_v1/combined/bestfit_vs_fiducial_cls/stage31_multigpu_20260604_bestfit_vs_fiducial_cls.pdf
```

V1 fit stats:

```text
fiducial whitened chi2 = 1646728.163018249
v1 best-fit whitened chi2 = 7346.232641444647
```

Per-family v1 block chi2 from the overlay summary:

```text
des_shear_EE:          fiducial 132.52,    best 124.93
act_y_des_shear_E:     fiducial 159.23,    best 60.01
desi_g_auto:           fiducial 1623261.22, best 6411.27
desi_g_act_y:          fiducial 14287.50,  best 177.10
desi_g_des_shear_E:    fiducial 6521.06,   best 420.76
desi_g_act_kappa:      fiducial 2711.17,   best 141.51
desi_pi_act_T:         fiducial 43.18,     best 21.72
```

The v1 best-fit improves the galaxy-related probes substantially but still leaves a poor full joint chi2. It should be treated as an operational v1 map-pasting point, not a final physical fit.

## Current V2 HMC Setup

The HMC config has been updated for a longer run:

```yaml
sampler:
  num_warmup: 800
  num_samples: 8000
  num_chains: 4
  chain_method: vectorized
  max_tree_depth: 4
  target_accept_prob: 0.85
```

The 4-GPU submission script is:

```text
notebooks/xDESI/survey_measure/submit_godmax_hmc_stage31_multigpu.sh
```

It now:

- defaults to `RUN_VERSION=v2`
- reads sampler values from `params_multiprobe_fast1024_hmc_stage31.yaml`
- runs 4 workers on 4 H100 GPUs
- pins one visible GPU per worker
- runs 4 vectorized chains per worker, 16 total chains
- disables NumPyro progress bars with `--no-progress`
- writes heartbeat logs every 120 seconds
- initializes from v1 best-fit params by default

Suggested v2 command:

```bash
sbatch notebooks/xDESI/survey_measure/submit_godmax_hmc_stage31_multigpu.sh stage31_hmc_8000x16_v2
```

Expected v2 combined outputs:

```text
notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/stage31_hmc_8000x16_v2/combined/
chain_stage31_multigpu_v2.npz
bestfit_params_stage31_multigpu_v2.yaml
bestfit_theory_data_vector_stage31_multigpu_v2.npz
fit_summary_stage31_multigpu_v2.json
```

## Backlight Paste-Map Next Step

The next task is to take the best-fit parameters, currently v1, and paste/paint maps in Abacus Backlight simulations. Relevant starting notebook:

```text
notebooks/xDESI/abacus_quick_paste_validation.ipynb
```

Recommended input params for first Backlight paste validation:

```text
param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v1.yaml
```

Recommended validation sequence:

1. Load v1 params and confirm fixed cosmology/astrophysical/HOD values match the analytic v1 fit.
2. Identify the Backlight lightcone shell and halo products needed for the same redshift range used in theory, roughly `0.005 < z < 3.0`.
3. Generate or paste four DESI photometric-bin galaxy maps, preserving the overlapping true-n(z) kernels and the per-pz HOD parameters.
4. Generate/paste baryonic fields using the shared v1 baryonic parameters, especially quantities needed for y/tau/kSZ-related comparisons.
5. Use the same masks, binning, and NaMaster conventions as the fast1024 measurement where possible.
6. Measure the simulated pseudo-Cls in the same 46-spectrum data-vector order.
7. Compare Backlight pseudo-Cls to:
   - analytic GODMAX at the same v1 params
   - measured xDESI fast1024 true-n(z) data
8. Check sign/unit conventions before interpreting kSZ:
   - raw simulated/measurement vector should follow `C_ell^{pi,T_uK}`
   - diagnostic plots should use positive `-D_ell^{pi,T}`
9. Check galaxy auto spectra carefully:
   - measurement comparison uses shot-noise-subtracted signal-only autos
   - any simulated shot noise should be handled consistently
10. Check abundance:
   - each pz-bin projected density should recover the measured `nbar_per_sr`
   - comoving nbar(z) should be `nbar_per_sr_i * n_i_true(z) / [chi(z)^2 dchi/dz]`

Key conceptual point for Backlight:

The four DESI bins are photometric bins, not four disjoint true-redshift shells. Their true-redshift kernels overlap. A paste-map implementation should therefore treat them as four separate projected tracers with their own HOD parameters and calibrated true-n(z), or assign simulated objects into photo-z bins in a way that reproduces those calibrated true-n(z) distributions and angular densities.

## Files To Reuse

Use these in the next chat/workstream:

```text
param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v1.yaml
param_files/xDESI/params_multiprobe_fast1024_true_nz_theory.yaml
param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml
param_files/xDESI/priors_multiprobe_fast1024_hmc_stage31.yaml
notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py
notebooks/xDESI/survey_measure/godmax_multiprobe_hmc_stage31.py
notebooks/xDESI/survey_measure/plot_stage31_bestfit_vs_fiducial_cls.py
notebooks/xDESI/survey_measure/submit_godmax_hmc_stage31_multigpu.sh
```

Most useful diagnostic plot:

```text
notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/stage31_hmc_400x16_v1/combined/bestfit_vs_fiducial_cls/stage31_multigpu_20260604_bestfit_vs_fiducial_cls.pdf
```

Most useful fit summary:

```text
notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/stage31_hmc_400x16_v1/combined/fit_summary_stage31_multigpu.json
```

## Known Caveats

- The v1 best-fit is not statistically acceptable yet; it is a practical first map-pasting point.
- The fit is still dominated by residuals in galaxy auto spectra, despite large improvement.
- The current likelihood uses the full covariance with one near-singular mode dropped.
- The kSZ comparison is convention-sensitive; always track raw versus plotted sign.
- The analytic theory comparison assumes calibrated true-n(z) for the photometric catalog; Backlight maps must preserve this or the comparison will not be apples to apples.
- The Stage-31 fit does not vary cosmology, DES shear calibration, source redshift shifts, IA, or all HOD parameters.
