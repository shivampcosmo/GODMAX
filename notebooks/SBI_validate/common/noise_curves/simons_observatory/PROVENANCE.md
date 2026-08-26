# Simons Observatory noise-curve provenance

Vendored from the official
[`simonsobs/so_noise_models`](https://github.com/simonsobs/so_noise_models)
repository at commit `fac881eb5ee012673d8994443caa3c6ad7fac2b6` on
2026-08-16.

## CMB-lensing convergence noise

- Source:
  `LAT_lensing_noise/lensing_v3_1_1/nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat`
- Upstream recommendation: deproj0, SENS1 baseline sensitivity, iterative
  reconstruction, `fsky=0.4`.
- Selected column: zero-based column 7, `N_lensing_MV (all)`.
- Units: convergence `N_L`, with no factors of `L` or `2*pi`.
- Tabulated support: `2 <= L <= 5000`; the implementation forbids
  extrapolation.
- SHA256:
  `21e95d82e47ad75c1665f4c5a317e67dcbd3ee1a110664555875fab2cb53c052`.

## Compton-y noise

- Source:
  `LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt`
- Selected column: zero-based column 3, `Deproj-2` (fiducial CIB SED
  deprojection), as requested for this analysis.
- Units: dimensionless Compton-y `N_ell`, with no factors of `ell`, `ell+1`,
  or `2*pi`.
- Tabulated support: `80 <= ell <= 7979`; the implementation forbids
  extrapolation.
- SHA256:
  `04cba7b20e06002b1a6af67e853aa643fbd3b5decea78f99e480cc7cc6b42f64`.

The upstream README files and license are copied alongside the numerical
tables. The covariance uses both curves in deconvolved sky-field space. The
native GODMAX y beam is therefore removed exactly once from every full-theory
y leg before combining signal and noise. Each saved band noise is computed by
direct lookup of every integer multipole in the bin and a `(2 ell + 1)`-weighted
average. Interpolation, extrapolation, and partially supported bins are
forbidden. The common complete-bin requirement is set by the convergence table,
which ends at `L=5000`.
