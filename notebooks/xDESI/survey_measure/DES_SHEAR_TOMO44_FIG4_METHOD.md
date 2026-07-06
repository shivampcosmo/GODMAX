# DES Shear Tomo 4x4 Fig. 4-Style NaMaster Check

This note documents exactly how the quick DES Y3 source-bin `4x4` shear
power-spectrum plot was made in
`notebooks/xDESI/survey_measure/des_shear_tomo44_fig4_check.ipynb` and
`notebooks/xDESI/survey_measure/diagnose_des_shear_harmonic.py`.

The goal is to reproduce the measurement convention of the DES Y3
harmonic-space shear paper, `notebooks/xDESI/papers/shear_harmonic/2203.07128v1.pdf`,
Fig. 4, for only the tomographic `4,4` EE panel. This is not the
non-tomographic all-source-bin panel in the upper right of Fig. 4.

## Environment

Use the conda environment that has NaMaster:

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
```

Do not use the default `python`; it does not reliably have `pymaster`.

## Input Product

The input survey bundle root is:

```text
data/xDESI/survey_data
```

The DES Y3 shear map file is resolved from `manifest.json` as:

```text
data/xDESI/survey_data/data/des_y3_shear_maps/des_y3_metacal_shear_maps_nside1024.h5
```

For DES source bin 4, use HDF5 group:

```text
maps/tomo3
```

The indexing is zero-based in the HDF5 file: source bin 4 corresponds to
`maps/tomo3`.

Use these datasets:

```text
maps/tomo3/gamma1
maps/tomo3/gamma2_namaster
maps/tomo3/mask_weight_raw
bandpowers/ell_left
bandpowers/ell_right
pixel_window/polarization
```

Use this shape-noise attribute from `maps/tomo3`:

```text
shape_noise_pseudo_cl_raw_weight_mask
```

For the current file, this attribute is:

```text
9.527137283442391e-05
```

The stored paper-style band edges begin and end as:

```text
ell_left first 5:  [8, 17, 30, 46, 66]
ell_right first 5: [17, 30, 46, 66, 89]

ell_left last 5:   [1492, 1596, 1704, 1815, 1930]
ell_right last 5:  [1596, 1704, 1815, 1930, 2049]
```

NaMaster treats the right edge as exclusive, so the final band goes up to
`ell=2048`.

## Paper-Like Settings

Use the `paper_like_raw_mask_pixwin_ell8_2048` scenario in
`diagnose_des_shear_harmonic.py`.

Exact settings:

```text
nside = 1024
lmax = 2048
ell_min = 8
n_bins = 32
mask_dataset = mask_weight_raw
noise_attr = shape_noise_pseudo_cl_raw_weight_mask
deconvolve_pixel_window = True
spin = 2
purify_e = False
purify_b = False
n_iter = 0
n_iter_mask = 0
lmax_mask = 2048
lite = True
```

These settings match the important measurement choices described in the
paper:

- 32 bandpowers.
- Band edges uniformly spaced in `sqrt(ell)`, not log-spaced.
- Equal weights inside each band, with the measurement additionally using
  the HEALPix polarization pixel-window correction.
- Raw Metacalibration weighted mask.
- No E/B purification.
- Noise-bias subtraction for auto-spectra.

## Field Construction

Clean maps and masks as follows:

```python
mask = np.nan_to_num(mask_weight_raw, nan=0.0, posinf=0.0, neginf=0.0)
mask[mask < 0.0] = 0.0

gamma1 = np.nan_to_num(gamma1, nan=0.0, posinf=0.0, neginf=0.0)
gamma2 = np.nan_to_num(gamma2_namaster, nan=0.0, posinf=0.0, neginf=0.0)
gamma1[mask <= 0.0] = 0.0
gamma2[mask <= 0.0] = 0.0
```

The code also multiplies both shear components by
`shear_e_to_kappa_sign = -1`. This sign is used elsewhere so scalar x
shear-E cross-spectra follow a positive-convergence convention. For the
shear auto-spectrum EE, the sign cancels and does not change the result.

Use the NaMaster field:

```python
field = nmt.NmtField(
    mask,
    [gamma1, gamma2_namaster],
    spin=2,
    purify_e=False,
    purify_b=False,
    n_iter=0,
    n_iter_mask=0,
    lmax=2048,
    lmax_mask=2048,
    lite=True,
)
```

Important: use `gamma2_namaster`, not a generic or raw `gamma2` dataset.
The transferred product already stores the second spin component in the
convention expected by NaMaster.

## Bandpowers And Pixel Window

For the paper-like scenario, read the stored edges from the HDF5 file:

```python
left = h5["bandpowers/ell_left"][:].astype(np.int32)
right = h5["bandpowers/ell_right"][:].astype(np.int32)
```

If those edges need to be recreated, use:

```python
edges = np.rint(np.linspace(np.sqrt(8), np.sqrt(2048), 33) ** 2).astype(np.int64)
edges[0] = 8
edges[-1] = 2049
left = edges[:-1].astype(np.int32)
right = edges[1:].astype(np.int32)
```

For the paper-like pixel-window correction, load:

```python
pixwin_full = h5["pixel_window/polarization"][:2049]
```

Then build the NaMaster bin object with `f_ell = 1 / pixwin**2` for every
integer multipole inside all bins:

```python
ells = np.concatenate([np.arange(li, ri, dtype=np.int64) for li, ri in zip(left, right)])
pixwin = pixwin_full[ells]
f_ell = np.ones_like(pixwin, dtype=np.float64)
good = pixwin > 0
f_ell[good] = 1.0 / np.square(pixwin[good])

bins = nmt.NmtBin.from_edges(left, right, f_ell=f_ell)
```

Use `bins.get_effective_ells()` as the plotted band-center multipoles.

For this run, the first and last effective multipoles are:

```text
first 5: [12.0, 23.0, 37.5, 55.5, 77.0]
last 5:  [1543.5, 1649.5, 1759.0, 1872.0, 1989.0]
```

## Spectrum Measurement

The tomo `4x4` spectrum spec is:

```text
name = des_shear_EE_tomo4x4
fields = (s4, s4)
component = 0
component labels = [EE, EB, BE, BB]
```

Measure with:

```python
workspace = nmt.NmtWorkspace.from_fields(field, field, bins)
pcl = nmt.compute_coupled_cell(field, field)
```

For the shear auto-spectrum, subtract flat shape noise in the EE and BB
components before or during NaMaster decoupling, following the local
pipeline convention:

```python
noise_level = h5["maps/tomo3"].attrs["shape_noise_pseudo_cl_raw_weight_mask"]
noise = np.zeros((4, lmax + 1), dtype=np.float64)
noise[0, :] = noise_level  # EE
noise[3, :] = noise_level  # BB

cl_all = workspace.decouple_cell(pcl, cl_noise=noise)
cl_EE = cl_all[0]
cl_EB = cl_all[1]
cl_BE = cl_all[2]
cl_BB = cl_all[3]
```

The saved diagnostic JSON stores all four components under
`cl_all_components`. The plot uses `cl_EE` for the points and can overlay
`EB`, `BE`, and `BB` as diagnostic lines.

## Error Bars

The quick check computes a one-spectrum NaMaster Gaussian covariance block
for the same field pair. In the code this is done through
`compute_covariance_block(spec, spec, ...)`.

The input covariance spectra are data-derived but are now converted to the
correct NaMaster space before calling the covariance routine. The pipeline does
not pass masked pseudo-`C_ell`s divided by a mask-overlap factor.

The corrected recipe is:

```python
pcl = nmt.compute_coupled_cell(field_a, field_b)
signal_bpw = workspace.decouple_cell(pcl, cl_noise=noise_pseudo_cl)
noise_bpw = workspace.decouple_cell(noise_pseudo_cl)
total_bpw = signal_bpw + noise_bpw
total_bpw[EB] = 0.0
total_bpw[BE] = 0.0
total_bpw = smooth_and_clip_positive_total_bandpowers(total_bpw)
input_cl = expand_each_bandpower_as_constant_full_ell(total_bpw, ell_left, ell_right, lmax)
```

The final expansion is deliberately a constant-in-band copy, matching the
standalone corrected-covariance script. It is not `bins.unbin_cell`; using the
NaMaster weighted unbinning convention here changes the covariance input model
and no longer reproduces the paper-style diagnostic error bars.

For DES shear auto-spectra, the saved data vector remains the noise-bias
subtracted EE bandpower. The covariance input is the total EE/BB power with
the same shape-noise template added back. This is the convention needed by
`nmt.gaussian_covariance(..., coupled=False)`.

Then `nmt.gaussian_covariance` is called with true spins:

```text
spin_a1 = 2
spin_a2 = 2
spin_b1 = 2
spin_b2 = 2
coupled = False
```

Only the EE x EE block is extracted. The plotted error bars are:

```python
err_EE = np.sqrt(np.diag(cov_EE_EE))
```

Important caveat: this is a local Gaussian diagnostic covariance, not the
full DES paper covariance used in the final DES cosmology analysis. The
DES paper covariance includes additional modeling beyond this quick check.
Use the error bars for debugging the measurement convention, not as an
exact reproduction of DES likelihood errors.

Do not use `coupled=True` for this plot. In this NaMaster installation that
returns the full coupled-ell pseudo-`C_ell` covariance, e.g. `(4*(lmax+1),
4*(lmax+1))` for a spin-2 auto-spectrum, not the `(4*n_band, 4*n_band)`
decoupled bandpower covariance.

## Plot Convention

The paper Fig. 4 caption says the points are scaled by the mean multipole of
each bandpower. Therefore plot:

```python
y = ell_eff * cl_EE * 1.0e7
yerr = ell_eff * err_EE * 1.0e7
```

The y-axis label is:

```text
Lbar * C_L^EE [1e-7]
```

The x-axis is drawn in a square-root multipole coordinate so that the
sqrt-spaced bands appear roughly evenly spaced, while tick labels are shown
as ell values:

```python
x = np.sqrt(ell_eff)
ticks_ell = np.asarray([0, 100, 400, 900, 1600], dtype=np.float64)
ax.set_xticks(np.sqrt(ticks_ell))
ax.set_xticklabels([str(int(v)) for v in ticks_ell])
ax.set_xlim(0.0, np.sqrt(2048.0) * 1.02)
ax.set_xlabel("Multipole ell")
```

This is why the plot visually matches the paper better than a standard log
x-axis.

## Command To Reproduce

Run the focused diagnostic from the repository root:

```bash
/usr/bin/env MPLCONFIGDIR=/tmp/matplotlib-codex \
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
notebooks/xDESI/survey_measure/diagnose_des_shear_harmonic.py \
--single-pair \
--scenario paper_like_raw_mask_pixwin_ell8_2048 \
--output data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.json \
--plot-output data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.png
```

Output files:

```text
data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.json
data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_tomo4x4_fig4_check.png
```

The notebook version is:

```text
notebooks/xDESI/survey_measure/des_shear_tomo44_fig4_check.ipynb
```

If executing the notebook with `nbconvert`, force the `ili-sbi` kernelspec:

```bash
/usr/bin/env \
JUPYTER_PATH=/mnt/home/spandey/miniconda3/envs/ili-sbi/share/jupyter \
JUPYTER_CONFIG_DIR=/tmp/jupyter-config-codex \
JUPYTER_DATA_DIR=/tmp/jupyter-data-codex \
IPYTHONDIR=/tmp/ipython-codex \
MPLCONFIGDIR=/tmp/matplotlib-codex \
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/jupyter nbconvert \
--to notebook --execute --inplace \
--ExecutePreprocessor.kernel_name=python3 \
notebooks/xDESI/survey_measure/des_shear_tomo44_fig4_check.ipynb
```

## Current Result

For the current data product and settings:

```text
scenario: paper_like_raw_mask_pixwin_ell8_2048
diag S/N: 23.898
ell range: 8..2048
number of bands: 32
negative EE bands: 0 / 32
Lbar*C_L^EE [1e-7] min / median / max: 1.497 / 3.623 / 6.866
```

First eight plotted EE points:

```text
ell=   12.0:   1.497 +/-   1.309
ell=   23.0:   6.056 +/-   1.498
ell=   37.5:   6.866 +/-   1.507
ell=   55.5:   5.958 +/-   1.227
ell=   77.0:   5.302 +/-   1.039
ell=  102.0:   6.026 +/-   0.914
ell=  131.0:   5.142 +/-   0.816
ell=  163.0:   3.846 +/-   0.764
```

## Common Failure Modes

If another implementation does not resemble the paper's tomo `4,4` panel,
check these first:

1. It must plot `ell_eff * C_ell * 1e7`, not raw `C_ell`.
2. It must use `mask_weight_raw` for this paper-like check, not
   `mask_weight`.
3. It must use `shape_noise_pseudo_cl_raw_weight_mask`, not the normalized
   mask noise attribute.
4. It must use `gamma2_namaster`.
5. It must use `spin=2`, no E/B purification, and no mask apodization.
6. It must use the stored `ell=8..2048`, 32-bin, sqrt-spaced band edges.
7. It must include the HEALPix polarization pixel-window correction through
   `f_ell = 1 / pixel_window[pixel_ell]^2`.
8. It should draw the x-axis in `sqrt(ell)` coordinates if the visual goal is
   to resemble the paper.
9. The covariance must be requested with `coupled=False` and supplied with
   full-ell total spectra in that same decoupled convention. Do not treat the
   `coupled=True` full pseudo-`C_ell` covariance as a 32-band covariance.
10. For flattened NaMaster covariance arrays, extract components in band-major
    order: `cov.reshape(n_band, n_comp_a, n_band, n_comp_b)[:, comp_a, :, comp_b]`.
    Do not use `comp * n_band + band` indexing.
11. Do not compare this single tomo `4,4` panel to the paper's upper-right
   non-tomographic all-source-bin panel; that panel has a much cleaner visual
   detection.
12. Do not interpret the quick Gaussian error bars as an exact reproduction
    of the DES paper likelihood covariance.
