# GODMAX Codebase Summary — Pasting & Backlight Mapping Subsystem

> **Purpose**: Context anchor for LLM-assisted development sessions.
> **Scope**: `notebooks/pasting/` (notebook + utilities) and all `src/` modules.
> **Generated**: 2026-03-18

---

## 1. High-Level Overview

**GODMAX** is a JAX-accelerated halo-model framework for computing cross-correlation observables of the large-scale structure. It models the thermodynamic and gravitational signatures of dark-matter halos — thermal Sunyaev–Zel'dovich (tSZ / Compton-*y*), kinetic SZ (kSZ), optical depth (τ), CMB lensing convergence (κ), and galaxy clustering (g) — and provides two complementary output modes:

1. **Analytic power spectra** — 3D halo-model power spectra P(k,z) projected to angular C(ℓ) via the Limber approximation.
2. **Simulated HEALPix maps** — halo-by-halo "pasting" of projected 2D profiles onto a pixelised sky, producing full-sky or partial-sky signal and galaxy-catalog maps from N-body halo catalogs.

The **pasting / backlight-mapping subsystem** (the focus of this document) exercises both modes to validate map-level observables against their analytic predictions. The validation notebook (`paste_backlight_maps_analytic_test.ipynb`) orchestrates the full pipeline: configuration → theory prediction → map generation → measurement → comparison.

### Key Physics Ingredients

| Ingredient | Description |
|---|---|
| **Halo Mass Function (HMF)** | Tinker 2008 (T08) / Tinker 2010 (T10) multiplicity functions with symbolic-regression emulators for σ(R) |
| **Halo Bias** | Tinker 2010 linear halo bias b(M,z) |
| **Concentration–Mass** | Duffy08, Prada12, Diemer15 c(M,z) relations |
| **NFW Profile** | Standard Navarro–Frenk–White density for dark matter halos |
| **Baryonic Correction Model (BCM)** | Schneider et al. gas ejection + stellar condensation + collision-less matter relaxation (adiabatic contraction/expansion) |
| **HOD (Halo Occupation Distribution)** | Leauthaud+11 SHMR-based model: Bernoulli centrals + Poisson satellites with NFW-distributed radii |
| **Pressure Profiles** | BCM-derived thermal pressure, plus Battaglia 2012/2016 and OWLS/LeBrun15 alternatives |
| **Power Spectrum** | 1-halo + 2-halo decomposition; halofit nonlinear P(k) for the 2-halo regime; 'poweradd' or 'response' transition models |
| **Angular Spectra** | Limber-integrated C(ℓ) with lensing, galaxy, tSZ, τ, and intrinsic-alignment window functions |
| **FFTLog Transforms** | JAX-ported mcfit library for Hankel / spherical-Bessel transforms (real-space ↔ Fourier-space) |
| **Symbolic Regression Emulators** | Fast analytic approximations replacing expensive numerical integrals for σ(R), P_lin(k), P_halofit(k), D(z), A_s(σ_8) |

---

## 2. Architecture & Data Flow

### 2.1 Class Hierarchy

The core computation pipeline is a linear inheritance chain. Each layer adds one conceptual stage:

```
base_class                        (src/base_class.py)
  ↓  cosmology, grids, linear P(k,z), growth factors
Profiles                          (src/get_radial_profiles.py)
  ↓  HMF, concentrations, NFW/gas/stellar/CLM/pressure profiles, HOD
get_Pkz                           (src/get_Pkzs.py)
  ↓  FFTLog → Fourier profiles u(k), 1h+2h P(k,z) for all probe pairs
get_Cl                            (src/get_Cls.py)
  ↓  Limber integration → C(ℓ) for all probe pairs
get_xi    (src/get_Xis.py)        get_cov   (src/get_covs.py)
  ↓  Hankel → ξ(θ)                 ↓  Gaussian + trispectrum covariance
```

Two additional classes branch from `Profiles` for map-level work:

```
Profiles
  ├─→ setup_sim_map               (src/get_sim_maps.py)
  │     Precomputes 2D projected profiles, builds 3D interpolators
  │
  └─→ get_sim_map                 (src/get_sim_maps.py)
        Pixel-level map assembly, HOD galaxy sampling, HEALPix output
```

Alternative profile models inherit directly from `base_class`:

```
base_class
  ├─→ Battaglia_12_16             (src/get_B12_profile.py)
  └─→ LeBrun15                    (src/get_OWLS_profile.py)
```

### 2.2 End-to-End Data Flow (Notebook Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│  paste_backlight_maps_analytic_test.ipynb                           │
│                                                                     │
│  1. build_config()  ──→  sim_params, halo_params, analysis,        │
│     (YAML + nbar)        other_params, cosmo_jax, zarray, nz_lens  │
│                                                                     │
│  2. Initialize analytic pipeline:                                   │
│     base_class → Profiles → get_Pkz → get_Cl                      │
│     (with M_halo_cut = 10^14 for high-mass mask)                   │
│                                                                     │
│  3. Theory C(ℓ): gy, gτ, gg  ──→  reference curves                │
│                                                                     │
│  4. setup_sim_map()  ──→  2D profile interpolators (y, ne, ρ_m)    │
│                                                                     │
│  5. load_halo_catalog()  ──→  (ra, dec, z, M200c, v_los)          │
│                                                                     │
│  6. generate_maps()  ──→  HEALPix maps (y, τ, κ, kSZ)            │
│                            + galaxy mock catalog                    │
│                                                                     │
│  7. Measure C(ℓ) from maps  ──→  compare with theory               │
│     • C_ℓ^{gg} with shot noise, 1h/2h decomposition               │
│     • C_ℓ^{gy}, C_ℓ^{gτ}                                          │
│     • Band-averaged ratios                                          │
│                                                                     │
│  8. Diagnostics: n(z), HOD(M), HMF, Mthresh, consistency checks   │
│                                                                     │
│  9. Visualization: Mollweide + gnomonic projections                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Data Structures

| Structure | Description |
|---|---|
| `cosmo_jax` | `jax_cosmo.Cosmology` instance — background cosmology (H₀, Ωm, Ωb, σ₈, n_s, w₀, wₐ) |
| `sim_params_dict` | Simulation grid parameters (nside, z-bins, ℓ-range, mass range) |
| `halo_params_dict` | HOD, BCM, concentration, HMF model selections and parameter values |
| `analysis_dict` | Probe selections, bin definitions, covariance settings |
| `other_params_dict` | IA parameters, nuisance terms |
| `Prof_test` | Instance of `setup_sim_map` — holds 3D `interpax.Interpolator3D` objects for y2D, ne2D, rhom2D |
| `mock_gals` | Structured array — (ra, dec, z, is_central, host_M200c, host_z) per galaxy |

---

## 3. Directory & File Index

### 3.1 Notebook & Utilities (`notebooks/pasting/`)

#### `paste_backlight_maps_analytic_test.ipynb`

**Purpose**: End-to-end validation of map-pasted observables against analytic halo-model predictions.

**Workflow stages** (see §2.2 above): config → theory C(ℓ) → profile interpolators → halo catalog → map generation → power spectrum measurement → diagnostics → visualization.

**Key variables**:
- `Cls_test` — instance of `get_Cl`; holds all analytic C(ℓ) arrays
- `Prof_test` — instance of `setup_sim_map`; holds profile interpolators
- `M_halo_cut = 1e14` — hard mass ceiling applied to the halo catalog
- `nbar_comoving = 1e-4` — canonical comoving galaxy number density [h³ Mpc⁻³]

---

#### `paste_backlight_utils.py` (~1200 lines)

**Purpose**: Central orchestration utility bridging configuration, halo catalog I/O, map generation, and all diagnostic analysis for the notebook.

| Function | Lines (approx.) | Description |
|---|---|---|
| `get_project_paths()` | 20 | Resolves GODMAX project directory paths |
| `build_config(params_path, data_path, nbar_comoving)` | 80 | Loads YAML params, builds all config dicts, derives n(z) from nbar via `nbar_to_nz_lens()`, returns 8-tuple of config objects |
| `compute_dV_dz_per_sr(cosmo_jax, z_array)` | 15 | Comoving volume element dV/dz per steradian |
| `nbar_to_nz_lens(nbar_comoving, z_array, cosmo_jax, f_sky)` | 25 | Converts comoving number density to angular n(z) for the Limber integral |
| `check_nz_nbar_consistency(nbar, nz, z_array, cosmo_jax)` | 20 | Round-trip validation: integrate n(z) back to nbar and check agreement |
| `update_nz_from_mock_catalog(mock_gals, ...)` | 30 | Replaces analytic n(z) with histogram measured from the mock galaxy catalog |
| `load_halo_catalog(fname)` | 25 | Reads HDF5 halo catalog fields: ra, dec, z, M200c, vlos |
| `generate_maps(ra, dec, z, M200c, vlos, Prof_test, ...)` | 200 | **Main map generation pipeline**: chunks halos, finds pixels via `process_halo`, calls JAX profile evaluation, builds galaxy catalog, saves to pickle |
| `make_galaxy_map(mock_gals, nside, zmin, zmax)` | 30 | Bins galaxy catalog into HEALPix overdensity map δ_g |
| `compute_ngal_vs_z(...)` | 40 | Measured vs theory galaxy number density n(z) comparison |
| `measure_hod_from_catalog(mock_gals, M200c_all, z_all, Cls_test)` | 60 | HOD diagnostics: ⟨N_cen⟩, ⟨N_sat⟩ vs M in redshift bins |
| `compute_shot_noise_Cl(mock_gals, nside, zmin, zmax)` | 20 | Map-level shot noise estimate from galaxy surface density |
| `compute_hod_shot_noise_Cl(Cls_test)` | 20 | Analytic HOD-integral shot noise |
| `compute_Cl_gg_1h_2h(Cls_test)` | 25 | Extracts 1-halo and 2-halo decomposition of C_ℓ^{gg} from the model |
| `compute_Cl_ratio_in_bands(ell_meas, Cl_meas, ell_th, Cl_th)` | 30 | Band-averaged measured/theory ratio for quantitative comparison |
| `compare_sim_vs_theory_hmf(M200c_all, z_all, Cls_test)` | 50 | Compares halo mass function from the catalog against Tinker theory |
| `print_diagnostic_summary(...)` | 40 | Formatted table of all diagnostic metrics |
| `stack_snapshot_maps(...)` | 40 | Stacks multiple redshift-snapshot map outputs into a single combined map |

---

### 3.2 Core Source Modules (`src/`)

#### `base_class.py` (~380 lines)

**Purpose**: Foundation for all GODMAX computations. Reads parameter dictionaries, initializes `jax_cosmo.Cosmology`, computes linear P(k,z), growth factors, comoving distances.

| Entity | Type | Description |
|---|---|---|
| `base_class` | Class | Root of the inheritance hierarchy |
| `read_all_input()` | Method | Parses all input dicts into instance attributes; sets up r, k, M, z, ℓ grids |
| `get_power_spectra_cosmo()` | Method | Computes linear P(k,z) and optionally symbolic-regression emulated P(k) |
| `get_rho_m(z)` | Method | Mean matter density ρ_m(z) |
| `get_Ez(z)` | Method | Dimensionless Hubble parameter E(z) = H(z)/H₀ |
| `get_rho_c(z)` | Method | Critical density ρ_c(z) |
| `logspace_trapezoidal_integral()` | Method | Log-spaced trapezoidal quadrature utility |
| `get_vmapped_func(func, num_args)` | Function | Helper to vmap a function over `num_args` leading axes |
| `get_vmapped_func_warg(func, n1, n2)` | Function | vmap dispatch with split argument groups |

**Key stored state**: cosmological parameters, grid arrays (r_array, z_array, k_array, M_array, ell_array), n(z) distributions, IA parameters, BCM parameters, HOD parameters.

---

#### `get_radial_profiles.py` (~900 lines)

**Purpose**: Computes all radial halo profiles — NFW dark matter, BCM gas, stellar, collision-less matter (CLM), and thermal pressure — plus the HMF, concentration–mass relations, and HOD.

| Entity | Type | Description |
|---|---|---|
| `Profiles` | Class | Inherits `base_class` |
| `setup_hmf()` | Method | Computes σ(M,z) via TophatVar or symbolic emulator |
| `get_hmf()` | Method | Evaluates dn/dlnM using selected multiplicity function (T08/T10) |
| `get_fsigma_Mz_T08()` | Method | Tinker 2008 multiplicity function f(σ) |
| `get_fsigma_Mz_T10()` | Method | Tinker 2010 multiplicity function f(σ) |
| `get_conc_Mz()` | Method | Dispatches to selected c(M,z) model |
| `get_conc_Mz_Duffy08()` | Method | Duffy et al. 2008 concentration–mass relation |
| `get_conc_Mz_Prada12()` | Method | Prada et al. 2012 concentration–mass relation |
| `get_conc_Mz_Diemer15()` | Method | Diemer & Kravtsov 2015 concentration–mass relation |
| `setup_main_calc()` | Method | Orchestrates full profile computation pipeline |
| `get_DMO_profiles()` | Method | NFW profiles for dark-matter-only halos |
| `run_stars_calc()` | Method | Central + satellite stellar density profiles (BCM) |
| `run_gas_calc()` | Method | Gas density profile ρ_gas(r) with ejection + bound components (BCM) |
| `run_clm_calc()` | Method | Collision-less matter profile via adiabatic relaxation (BCM) |
| `run_cga_calc()` | Method | Central galaxy contribution |
| `run_dmb_calc()` | Method | Total baryonified dark matter profile ρ_dmb(r) |
| `run_pressure_calc()` | Method | Thermal + non-thermal pressure P_tot(r) from hydrostatic equilibrium |
| `get_rho_nfw_normed()` | Method | Normalized NFW density profile |
| `get_rho_gas_normed()` | Method | Normalized gas density profile |
| `get_Mthresh(jz)` | Method | Stellar mass threshold solver from nbar constraint |
| `get_Ncen(jz, jM)` | Method | Central occupation ⟨N_cen⟩(M,z) via error function |
| `get_Nsat(jz, jM)` | Method | Satellite occupation ⟨N_sat⟩(M,z) via power law |
| `get_Mstar_Mh()` | Method | Stellar-to-halo mass relation (Leauthaud+11 SHMR) |
| `get_Mh_Mstar()` | Method | Inverse SHMR: M_h(M_*) |
| `get_fstar_cen()` | Method | Central stellar mass fraction |
| `get_fstar_sat()` | Method | Satellite stellar mass fraction |
| `get_Ptot()` | Method | Total (thermal + non-thermal) pressure from hydrostatic equilibrium |
| `get_Pnt_fac()` | Method | Non-thermal pressure fraction P_nt/P_tot |

---

#### `get_Pkzs.py` (~296 lines)

**Purpose**: Computes 3D power spectra P(k,z) for all field cross-correlations (matter, gas, tSZ, τ, galaxy, κ).

| Entity | Type | Description |
|---|---|---|
| `get_Pkz` | Class | Inherits `Profiles` |
| FFTLog transform | Init | Converts real-space profiles → Fourier-space ũ(k) via `xi2P` |
| `uk_dmb` | Attribute | Fourier-space baryonified matter profile |
| `uk_nfw` | Attribute | Fourier-space NFW profile |
| `uk_y` | Attribute | Fourier-space Compton-y profile |
| `uk_clm` | Attribute | Fourier-space collision-less matter profile |
| `uk_ne` | Attribute | Fourier-space electron density profile |
| `ukg_cross` | Attribute | Galaxy–field cross Fourier kernel (HOD-weighted) |
| `ukg_auto_sqr` | Attribute | Galaxy auto Fourier kernel squared (HOD-weighted) |
| `get_bias_Mz()` | Method | Tinker 2010 linear halo bias b(M,z) |
| `get_P_1h(jk, jz, probe1, probe2)` | Method | 1-halo power: ∫ dM (dn/dM) ũ₁(k,M) ũ₂(k,M) |
| `get_P_2h(jk, jz, probe1, probe2)` | Method | 2-halo power: b₁ b₂ P_lin(k,z) (effective bias from HMF-weighted integrals) |
| Suppression factor | Logic | `Pmm_sup = P_halofit / P_nfw_tot` ensures correct total matter power |
| Transition models | Logic | `'poweradd'` (default) or `'response'` for combining 1h+2h terms |

---

#### `get_Cls.py` (~323 lines)

**Purpose**: Projects 3D P(k,z) → angular C(ℓ) via Limber integration for all probe pair combinations.

| Entity | Type | Description |
|---|---|---|
| `get_Cl` | Class | Inherits `get_Pkz` |
| `get_P_lz()` | Method | Maps P(k,z) → P(ℓ/χ, z) for Limber substitution |
| 2D interpolators | Init | `interpax` 2D interpolators for all spectra over (ℓ, z) |
| `get_weak_lensing_kernel()` | Method | Lensing efficiency q(z) for source bins |
| CMB lensing kernel | Method | W_κ(z) for CMB lensing convergence |
| NLA intrinsic alignment | Method | A_IA(z) with redshift and luminosity scaling |
| Galaxy n(z) lens | Method | Galaxy window W_g(z) = n(z) · H(z)/c |
| tSZ Compton-y weight | Method | W_y(z) = σ_T / (m_e c²) · a⁻¹ |
| τ weight | Method | W_τ(z) = σ_T · n_e(z) · dχ/dz |
| `get_Cl_tot(jb1, jb2, probe1, probe2)` | Method | Main Limber integral: C_ℓ = ∫ dz W₁ W₂ P(ℓ/χ, z) / χ² |
| `Cl_kappa_kappa_tot_mat` | Attribute | κκ angular power spectrum |
| `Cl_gal_gal_tot_mat` | Attribute | gg angular power spectrum (with 1h, 2h, shot noise components) |
| `Cl_gal_kappa_tot_mat` | Attribute | gκ angular cross-spectrum |
| `Cl_kappa_y_tot_mat` | Attribute | κy angular cross-spectrum |
| `Cl_gal_y_tot_mat` | Attribute | gy angular cross-spectrum |
| `Cl_gal_tau_tot_mat` | Attribute | gτ angular cross-spectrum |
| `Pge_tot_mat` | Attribute | Galaxy–electron 3D cross-power P_ge(k,z) |

---

#### `get_sim_maps.py` (~953 lines)

**Purpose**: Simulated sky-map generation via halo-by-halo profile pasting onto HEALPix grids, plus HOD-based galaxy catalog population.

| Entity | Type | Description |
|---|---|---|
| `setup_sim_map` | Class | Inherits `Profiles`; precomputes 2D projected profiles |
| `_generic_2D_projection()` | Method | Abel-transform-like line-of-sight integration of 3D profiles → 2D |
| 3D interpolators | Attribute | `interpax.Interpolator3D` for y2D(θ, M, z), ne2D(θ, M, z), rhom2D(θ, M, z) |
| Beam convolution | Logic | Smoothing via Hankel-space multiplication with Gaussian beam |
| `get_sim_map` | Class | Inherits `Profiles`; full map generation |
| `populate_one_halo()` | Method | HOD galaxy sampling: Bernoulli centrals + Poisson satellites |
| `sample_radii_from_ppf_3d()` | Method | Inverse-CDF sampling for satellite radii from NFW profile |
| `place_satellites()` | Method | Converts sampled 3D radii → angular (ra, dec) offsets from halo center |
| `_assemble_map()` | Method | Efficient HEALPix pixel accumulation from per-halo contributions |
| `process_halo()` | Method | Per-halo pipeline: find pixels within angular extent → evaluate profiles → accumulate into map |

---

#### `get_B12_profile.py` (~190 lines)

**Purpose**: Battaglia 2012/2016 parametric pressure and gas density profiles — an alternative to the BCM-derived profiles.

| Entity | Type | Description |
|---|---|---|
| `Battaglia_12_16` | Class | Inherits `base_class` |
| Pressure profile | Method | Generalized NFW (GNFW) form with power-law M,z-dependent parameters (P₀, x_c, β) |
| Density profile | Method | Battaglia 2016 gas density fitting function |

---

#### `get_OWLS_profile.py` (~244 lines)

**Purpose**: OWLS/cosmo-OWLS pressure profiles calibrated to hydrodynamical simulations (Le Brun et al. 2015).

| Entity | Type | Description |
|---|---|---|
| `LeBrun15` | Class | Inherits `base_class` |
| Mass conversion | Logic | Uses `colossus` for M200c → M500c conversion |
| AGN feedback variants | Config | `'ref'`, `'agn_8'`, `'agn_8p5'` — three feedback strengths |

---

#### `get_covs.py` (~674 lines)

**Purpose**: Gaussian and non-Gaussian covariance matrices for angular power spectra.

| Entity | Type | Description |
|---|---|---|
| `get_cov` | Class | Inherits `get_Cl` |
| `get_cov_G()` | Method | Gaussian covariance: diagonal in ℓ, includes noise terms (shape noise, shot noise) |
| `get_cov_NG()` | Method | Non-Gaussian covariance via connected trispectrum |
| `get_T_ABCD_NG()` | Method | 1-halo trispectrum T(ℓ₁, ℓ₂) for non-Gaussian covariance |
| `Cl_y_y_tot_mat` | Attribute | tSZ auto-spectrum (needed for tSZ covariance) |
| Real-space transform | Logic | Optional two-Bessel FFT for real-space covariance |

---

#### `get_Xis.py` (~89 lines)

**Purpose**: Real-space correlation functions ξ(θ) from angular power spectra via Hankel transforms.

| Entity | Type | Description |
|---|---|---|
| `get_xi` | Class | Inherits `get_Cl` |
| `gty_out_mat` | Attribute | Shear–tSZ tangential profile γ_t × y(θ) |
| `xip_out_mat` | Attribute | Shear ξ₊(θ) correlation function |
| `xim_out_mat` | Attribute | Shear ξ₋(θ) correlation function |

---

#### `hmf_symbolic.py` (~107 lines)

**Purpose**: Symbolic-regression emulators for HMF ingredients, replacing expensive numerical integrals.

| Function | Description |
|---|---|
| `symbolic_lnsigma_corr()` | Correction to ln σ(R) from symbolic regression fit |
| `symbolic_dlnsigmadR()` | d ln σ / dR emulator |
| `symbolic_sigma()` | Direct σ(R) emulator |

---

#### `matter_pk_symbolic.py` (~376 lines)

**Purpose**: Symbolic-regression emulators for matter power spectrum ingredients and background quantities.

| Function | Description |
|---|---|
| `symbolic_As()` | A_s(σ₈) mapping |
| `symbolic_D()` | Linear growth factor D(z) |
| `symbolic_pklin()` | Linear matter power spectrum P_lin(k) |
| `symbolic_ksigma()` | k_σ for halofit |
| `symbolic_neff()` | Effective spectral index n_eff for halofit |
| `symbolic_C()` | Spectral curvature C for halofit |
| `symbolic_pkhalofit()` | Nonlinear P(k) via halofit emulator |
| `get_eisensteinhu_nw()` | No-wiggle Eisenstein & Hu 1998 transfer function |

---

#### `gaussian_tension.py` (~290 lines)

**Purpose**: Gaussian tension metrics (from Marco Raveri's tensiometer) for comparing posterior distributions.

| Function | Description |
|---|---|
| `get_Neff()` | Effective number of constrained parameters |
| `gaussian_approximation()` | Gaussian approximation to posterior distributions |
| `get_localized_covariance()` | Localized covariance for tension computation |

---

#### `setup_baryonification.py`

**Purpose**: Placeholder (empty file) for future baryonification setup code.

---

### 3.3 Helper Modules (`src/helpers/`)

#### `constants.py` (~94 lines)

**Purpose**: Physical constants in CGS units and standard cosmological quantities.

| Constant | Description |
|---|---|
| `RHO_CRIT_0_KPC3` | Critical density at z = 0 in units of M☉ kpc⁻³ |
| `DELTA_COLLAPSE` | Linear collapse threshold δ_c ≈ 1.686 |
| `SIGMA_T` | Thomson cross-section |
| `M_E`, `M_P` | Electron, proton masses |
| `K_B` | Boltzmann constant |
| Unit conversions | Mpc ↔ cm, keV ↔ erg, etc. |

---

#### `jax_cosmo_power.py` (~293 lines)

**Purpose**: Modified `jax_cosmo` power spectrum module with halofit implementation (Takahashi 2012 recalibration).

| Function | Description |
|---|---|
| `linear_matter_power()` | Linear P(k,z) from transfer function |
| `halofit_parameters()` | Computes halofit fitting parameters (k_σ, n_eff, C) |
| `halofit()` | Full halofit nonlinear mapping |
| `nonlinear_matter_power()` | Top-level P_nl(k,z) = halofit(P_lin) |

---

#### `twobessel.py` (~275 lines)

**Purpose**: 2D FFTLog for real-space covariance transforms (Xiao Fang algorithm).

| Class | Description |
|---|---|
| `two_sph_bessel` | Double spherical-Bessel transform |
| `two_Bessel` | Double cylindrical-Bessel transform |

---

### 3.4 mcfitjax Subpackage (`src/mcfitjax/`)

A JAX port of the Python [mcfit](https://github.com/eelregit/mcfit) library for multiplicative convolution integral transforms based on FFTLog.

#### `mcfit_jax.py` (~565 lines)

**Purpose**: Core FFTLog engine.

| Entity | Type | Description |
|---|---|---|
| `mcfit` | Class | Main FFTLog class: input grid → Mellin-space → output grid |
| Padding | Logic | Configurable zero-padding (extrap='const', 'loglin', etc.) |
| Low-ringing | Logic | Optimal bias parameter selection to minimize edge artifacts |

---

#### `transforms.py` (~155 lines)

**Purpose**: Named transform subclasses wrapping `mcfit`.

| Class | Transform |
|---|---|
| `Hankel` | Hankel transform (order ν) |
| `SphericalBessel` | Spherical Bessel transform (order ℓ) |
| `FourierSine` | Fourier sine transform |
| `FourierCosine` | Fourier cosine transform |
| `DoubleBessel` | Double-Bessel transform |
| `DoubleSphericalBessel` | Double spherical-Bessel transform |
| `GaussSmooth` | Gaussian smoothing via Fourier convolution |

---

#### `cosmology_jax.py` (~138 lines)

**Purpose**: Cosmology-specific transforms.

| Class / Function | Description |
|---|---|
| `P2xi` | 3D power spectrum → correlation function ξ(r) |
| `xi2P` | Correlation / real-space profile → Fourier power P(k) |
| `C2w` | Angular C(ℓ) → angular correlation w(θ) |
| `w2C` | Angular correlation → C(ℓ) |
| `TophatVar` | Variance σ²(R) in a top-hat window |
| `GaussVar` | Variance σ²(R) in a Gaussian window |

---

#### `kernels.py` (~82 lines)

**Purpose**: Mellin-space kernels for the FFTLog decomposition.

| Function | Description |
|---|---|
| `Mellin_BesselJ(nu, z)` | Mellin transform of cylindrical Bessel J_ν |
| `Mellin_SphericalBesselJ(ell, z)` | Mellin transform of spherical Bessel j_ℓ |
| `Mellin_TophatSq(d, z)` | Mellin transform of top-hat window squared |
| `Mellin_GaussSq(z)` | Mellin transform of Gaussian window squared |

---

#### `loggamma_jax.py` (~237 lines)

**Purpose**: JAX-compatible complex log-Gamma function (from Adam Coogan).

Uses series expansion and Stirling approximation with reflection formula for robustness across the complex plane.

---

## 4. Dependencies & Relationships

### 4.1 External Dependencies

| Package | Role |
|---|---|
| **JAX** (`jax`, `jax.numpy`, `jax.random`) | Core autodiff and GPU acceleration framework |
| **jax_cosmo** | Background cosmology: distances, growth factor, H(z), P_lin(k) |
| **interpax** | JAX-compatible N-dimensional interpolation (Interpolator1D, 2D, 3D) |
| **HEALPix** (`healpy`) | Pixelised sky maps, `ang2pix`, `query_disc`, `anafast` |
| **NumPy / SciPy** | Array utilities, special functions (where JAX equivalents are unavailable) |
| **h5py** | HDF5 halo catalog I/O |
| **colossus** | Mass definition conversions (M200c ↔ M500c) in `get_OWLS_profile.py` |
| **PyYAML** | Configuration file parsing |
| **matplotlib** | Plotting in the validation notebook |

### 4.2 Internal Coupling Map

```
paste_backlight_maps_analytic_test.ipynb
    ├── paste_backlight_utils.py  (orchestration)
    │       ├── build_config()  →  reads YAML, creates cosmo_jax
    │       ├── generate_maps() →  calls get_sim_map methods
    │       └── diagnostics     →  calls get_Cl attributes
    │
    ├── base_class      ←── all modules depend on this
    ├── Profiles         ←── get_Pkz, get_sim_maps depend on this
    ├── get_Pkz          ←── get_Cls depends on this
    ├── get_Cl           ←── get_covs, get_Xis, paste_backlight_utils depend on this
    ├── setup_sim_map    ←── generate_maps() in utils
    └── get_sim_map      ←── generate_maps() in utils

mcfitjax/
    ├── mcfit_jax.py     ←── transforms.py, cosmology_jax.py depend on this
    ├── kernels.py       ←── mcfit_jax.py imports kernel functions
    ├── loggamma_jax.py  ←── kernels.py imports complex log-Gamma
    ├── transforms.py    ←── cosmology_jax.py inherits transform classes
    └── cosmology_jax.py ←── get_Pkzs.py uses xi2P; base_class uses TophatVar

helpers/
    ├── constants.py       ←── base_class, Profiles, get_sim_maps import constants
    ├── jax_cosmo_power.py ←── base_class uses for halofit P_nl(k)
    └── twobessel.py       ←── get_covs.py uses for real-space covariance
```

### 4.3 Tight Couplings & Design Notes

1. **Linear inheritance chain** (`base_class` → `Profiles` → `get_Pkz` → `get_Cl`): Each layer assumes all parent state is fully initialized. Instantiating `get_Cl` triggers the entire pipeline from cosmology through to angular spectra.

2. **`paste_backlight_utils.py` ↔ `get_Cl` / `get_sim_map`**: The utility module directly accesses internal attributes of `get_Cl` (e.g., `Cl_gal_y_tot_mat`, `Cl_gal_gal_1h_mat`) and `Profiles` (e.g., `dndlnM_mat`, `Ncen_mat`). Changes to internal attribute names in the source modules will break the utilities.

3. **mcfitjax ↔ profile pipeline**: `xi2P` (in `cosmology_jax.py`) is the critical transform used by `get_Pkzs.py` to convert real-space profiles ρ(r) into Fourier-space ũ(k). The `TophatVar` transform is used by `base_class` for σ(R) computation.

4. **`interpax` interpolators in `setup_sim_map`**: The 3D interpolators built by `setup_sim_map` over (θ, M, z) are the bridge between the analytic profile computation and the per-halo pixel-level map pasting in `get_sim_map`.

5. **Symbolic emulators as drop-in replacements**: `hmf_symbolic.py` and `matter_pk_symbolic.py` provide fast alternatives to the numerical integrals in `base_class` and `Profiles`. They are toggled by configuration flags and must remain numerically consistent with their numerical counterparts.

6. **Configuration flow**: YAML → `build_config()` → 4 dictionaries → `base_class.read_all_input()` → instance attributes. All downstream modules read from these attributes rather than re-parsing config.

---

*End of codebase summary.*
