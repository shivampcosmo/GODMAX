---
id: kb.sbi.three-way-mock-comparison-plan
title: Plan for the theory-HMC, theory-SBI, and pasted-mock-SBI comparison
layer: 60-projects
owner: xdesi-lead
status: deprecated
confidence: medium
scope:
  - notebooks/SBI_validate/01_validate_fiducial_theory_datavector.ipynb
  - notebooks/SBI_validate/backlight_metadata.py
  - notebooks/SBI_validate/fiducial_theory_datavector.py
  - notebooks/SBI_validate/gaussian_covariance.py
  - notebooks/SBI_validate/survey_defaults.py
  - notebooks/SBI_validate/theory_sbi_utils.py
  - notebooks/SBI_validate/run_hmc_five_parameter_probe_scan.py
  - notebooks/SBI_validate/run_hmc_five_parameter_probe_checkpointed.py
  - notebooks/SBI_validate/run_sbi_theory_cls.py
  - notebooks/SBI_validate/run_sbi_five_parameter_probe_sequential.py
  - notebooks/SBI_validate/run_simulator_native_active_sbi.py
  - notebooks/SBI_validate/pasted_map_cls_validation.py
  - notebooks/SBI_validate/map_sbi_pasted_utils.py
  - notebooks/SBI_validate/run_map_sbi_pasted_worker.py
  - notebooks/xDESI/abacus_lightcone_catalog.py
  - notebooks/xDESI/survey_measure/multiprobe_namaster.py
  - notebooks/pasting/paste_backlight_utils.py
  - src/get_sim_maps.py
invariants:
  - INV-ABACUS-COSMO-01
  - INV-PHYS-UNITS-01
  - INV-NZ-NORM-01
  - INV-WINDOW-CMP-01
  - INV-BEAM-01
  - INV-SHOTNOISE-01
  - INV-NMT-BANDMAJOR-01
  - INV-NMT-COUPLED-01
  - INV-PRODUCT-PROV-01
  - INV-JAX-X64-01
  - INV-JAX-GRAD-FINITE-01
  - INV-JAX-SEED-01
  - INV-MCMC-CONVERGENCE-01
  - INV-MCMC-TREEDEPTH-01
  - INV-WHITEN-RANK-01
  - INV-CHI2-HONEST-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - python -m pytest -q tests/test_sbi_so_noise_covariance.py
  - python tools/kb/kb.py invariants --check --id INV-ABACUS-COSMO-01 --id INV-JAX-SEED-01 --id INV-PROC-NOTOLERANCE-01
verified_at_commit: UNSTAMPED
verified_on: 2026-08-18
see_also: [kb.sbi.three-probe-posterior-comparison-execution, kb.sbi.analytical-hmc-sbi, kb.sbi.simulator-native-efficient, kb.physics.halo-model-ingredients, kb.inference.likelihood-and-convergence, kb.xdesi.abacus-paste]
supersedes: []
---

## Claim

> **Deprecated on 2026-08-20.** This document preserves the original pre-paste planning
> history. Its nside-512, ell-1535, 36-vector, and provisional tau-noise decisions are no
> longer the active experiment. Use
> `kb.sbi.three-probe-posterior-comparison-execution` for the nside-1024, dense-ell-2048,
> 14-band/42-vector, S/N-matched-tau execution contract. Do not launch a job from this file.

The requested three-contour comparison is feasible, but no new scientific contour should be
generated yet. It is valid only if theory HMC, theory SBI, and pasted-mock SBI use the same
five-parameter prior, fixed parameters, observed vector, probe/bin order, covariance, mass and
redshift contract, cosmology, and observable conventions.

The theory-SBI posterior may be used to place expensive mock simulations. It must remain a
proposal distribution: the final mock posterior must still target the original five-parameter
box prior. A pure posterior cloud relabelled as a new prior is not correct.

The user has now fixed the parent simulation, response-based target, angular resolution,
resolved forward model, provisional tau-noise interpretation, mass semantics and all-in
simulation budget. For this exercise only, `InterpolatedN * ParticleMassHMsun` is treated as
`M200c`. This is a declared provisional modeling assumption, not a claim that the catalog
contains a measured spherical-overdensity mass. Inference starts at the user-selected operational
mass floor frozen before inference, not at the lowest row merely present in the catalog.

The accepted affordable target is a theory-anchored, common-random-numbers estimate of the
pasted **parameter response conditional on the c0000/ph000 phase and frozen galaxy/HOD
realization**. Pairing strongly reduces the fixed-phase contribution, but does not make one
phase an ensemble mean: parameter-dependent phase residuals remain. This is a conditional
response comparison, not a measurement of the absolute theory-versus-paste offset at the
reference point or an ensemble-calibrated simulation forecast.

After the P0a provenance audit showed that the public 100-particle reliability calibration is
not Backlight-specific, the user replaced that documentation gate with an explicit operational
mass floor: `M_particle_proxy_hMsun >= 5e11 Msun/h`. With the audited c0000 particle mass this
lies between the pre-registered 100- and 125-particle thresholds. It remains a provisional
working selection and does not promote the proxy to a measured spherical-overdensity mass.

### Decisions frozen after the planning discussion

| item | frozen choice |
|---|---|
| simulation | `AbacusBacklight_base_c0000_ph000` |
| redshift | strict `0.3 < z < 0.5` |
| mass | `InterpolatedN * ParticleMassHMsun`, provisionally treated as `M200c` in `Msun/h`; user-selected operational floor `M_particle_proxy_hMsun >= 5e11 Msun/h`; upper bound `< 1e16 Msun/h`; retain 125/150-particle sensitivity branches |
| tau noise | current effective tau noise allowed for the first result, with every result labelled provisional |
| mock target | paired fixed-phase conditional response anchored to resolved theory |
| production maps | `nside=512`, harmonic support `0 <= ell <= 1535` |
| inference bands | the 12 existing bands with complete support below the right-exclusive `ell=1536` boundary; 36 elements in `gy,gkappa,gtau` order |
| forward model | one fully consistent resolved-halo model for HMC, theory SBI and mock SBI |
| observation noise | synthesize Gaussian field-noise maps from the same unbinned noise spectra and observable basis used by the covariance, add them before the estimator, and save the measured noisy 36-vector; no bandpower-space noise injection |
| mask and sky fraction | one deterministic common scalar mask for g, y, CMB-kappa and tau; final `mean(mask**2)=0.4`; exact NaMaster decoupling, saved bandpower windows and mask covariance in all three methods |
| map retention | retain the full map bundle for every successful expensive paste execution, including validation and diagnostics, up to the all-in cap; content-address bitwise-identical fields rather than duplicating them |
| design provenance | save the original prior, every normalized sequential proposal, per-map `log p0`, per-map `log q`, round/role/seeds and importance-eligibility flags for later summary-statistic inference |
| budget | adaptive three-round design; hard cap 600 successful expensive paste executions in total, including validation, training, holdout and every alternate nside/profile/HOD/cap diagnostic; stop earlier only after oracle, holdout and posterior-stability gates pass |

## Why it is true

### What can be reused

- The current five varied parameters and original priors are defined at
  `notebooks/SBI_validate/run_hmc_five_parameter_probe_scan.py:41-49`:

  | parameter | fiducial | scientific prior |
  |---|---:|---:|
  | `theta_ej_0` | 2.0 | Uniform `[0.5, 8.0]` |
  | `alpha_nt` | 0.05 | Uniform `[0.0, 0.5]` |
  | `mu_beta` | 0.5 | Uniform `[0.005, 1.5]` |
  | `theta_co_0` | 0.05 | Uniform `[0.001, 0.5]` |
  | `nu_theta_ej_M` | -0.1 | Uniform `[-1.0, 1.0]` |

- The fixed complement is `log10_Mstar0_theta_ej=15`, `nu_theta_ej_z=0`,
  `log10_Mc0=13.75`, `delta_rhogas=7`, and `gamma_rhogas=2`
  (`run_hmc_five_parameter_probe_scan.py:50-63`).
- The current joint inference order is `gy`, `gkappa`, `gtau`, with 17 bands per probe and
  51 elements in total (`run_hmc_five_parameter_probe_scan.py:36-40` and
  `run_simulator_native_active_sbi.py:51-52`). The requested prose order `gy, gtau, gkappa`
  must not silently change the stored order.
- The current theory SNPE implementation handles later proposal rounds correctly by passing
  the previous posterior through `proposal=` when simulations are appended
  (`run_sbi_theory_cls.py:462-512`). Its componentwise probit transform preserves the original
  bounded Uniform prior (`run_sbi_theory_cls.py:261-283`).
- The current covariance is fixed at the fiducial point, Gaussian, diagonal between ell bands,
  and includes all same-band cross-probe blocks (`gaussian_covariance.py:117-185`). It uses
  DESI-like galaxy shot noise, official SO y Deproj-2 noise, and official SO iterative MV CMB
  convergence noise (`survey_defaults.py:283-329`).
- The existing simulator-native runner already separates conditional-mean and stochastic
  simulators and records unique parameter points and seeds
  (`run_simulator_native_active_sbi.py:121-208`). Its present simulation sufficiency is still
  unverified; it is an implementation starting point, not evidence that 600 pastes are enough.

### What cannot be reused unchanged

1. The old validation notebook uses `0.4 < z < 0.6`, a raw `10^13` mass cut, and
   `nside=512` (`01_validate_fiducial_theory_datavector.ipynb`, cells 2 and 8). Its cached
   products are not the requested experiment.
2. The exact HDF5 file used there is
   `data/backlight/halo_catalog_Mlim_1e13_zlim_0.4_0.6.h5`
   (`backlight_metadata.py:13-17`). Direct inspection on 2026-08-18 found 4,813,871 rows,
   `0.40000018 <= z <= 0.59999924`, and raw proxy masses above
   `1.0001099e13`. It cannot recover either the requested lower-mass halos or the missing
   `0.3 < z < 0.4` slice.
3. The HDF5 column called `M200c` is a particle-count proxy. The current helper describes it
   as `InterpolatedN * ParticleMassMsun` and converts it to `Msun/h`
   (`backlight_metadata.py:87-101,113-138`). The user has approved that proxy as the working
   `M200c` for this exercise, but its provisional semantics must remain in every product label,
   metadata record and final figure.
4. Direct schema inspection of the c0000/ph000 `halo_lightcone` tree found no independent
   `M200c` or `R200c` field. `halo_timeslice/SO_radius` is an evolving CompaSO mean-density
   radius at a snapshot, so it will not be used to create a second, apparently more physical
   mass definition. `R200c` needed by the painter is derived consistently from the accepted
   proxy and the c0000 critical density, with that derivation recorded as provisional.
5. The user has resolved the simulation identity as c0000/ph000. This is different from the
   current xDESI c9999/ph9999 default, so no xDESI catalog/config may be reused without an
   explicit c0000 override. The old c0000 HDF5 still cannot be reused because it is pre-cut,
   proxy-mass based and lacks provenance attributes and halo IDs.
6. `hod_mass_cut` is only a lower, galaxy-occupation mask on `Ncen` and `Nsat`
   (`fiducial_theory_datavector.py:295-299`). It is not a two-sided cut on every halo-model
   contribution. The requested selection needs explicit theory integration bounds and an
   identical catalog predicate.
7. The present paste validation sends every angularly valid halo to the y/tau/kappa painter;
   the redshift selection is applied only later when the galaxy overdensity map is made
   (`pasted_map_cls_validation.py:157-183,219-247`). This does not enforce one common halo
   slice.
8. `map_sbi_pasted_utils.py:538-585` neither applies the c0000 catalog cosmology/mass
   conversion nor the requested mass/redshift predicate. It also hard-codes the map profile
   grid to `12 <= log10(M/[Msun/h]) <= 15.75`.
9. The old map helpers copy and modify `halo_params_dict` after constructing `Profiles`, then
   inject the old `Profiles` object into `setup_sim_map` (`pasted_map_cls_validation.py:114-155`).
   Lowering the copied map-grid bound therefore does not prove that the underlying profile grid
   contains the new low masses.
10. The current 51-vector reaches a final band upper edge near ell 4000. The accepted
    `nside=512` map has `lmax=1535`. Complete-support selection retains native bands 0--11,
    whose last upper edge is 1267.91456463, giving a 36-vector. The next existing band extends
    to 1596.20986651 and must be dropped. Calling the current center-based selector with
    `ell_max=1536` would incorrectly keep that partial band.
11. Map spectra are mode-count-weighted band averages (`pasted_map_cls_validation.py:250-264`),
   while the current inference product stores native theory values at bin centers. One saved
   bandpower operator must replace this mismatch.
12. `full` theory and `map_matched_resolved` theory are different physical targets. The user
    selected the resolved target for all three inference methods. The current five-parameter
    HMC/SBI defaults to `full`, so those products and entry-point defaults must not be reused.
    The current resolved implementation's spherical support cut is only an approximation to the
    map painter's projected aperture and must pass the operator-matching gate or be replaced.
13. Official SO y and CMB-kappa noise is currently selected only when `theory_mode="full"`;
    map-matched theory falls back to legacy y noise and LSST-like shape noise
    (`fiducial_theory_datavector.py:911-955`, `survey_defaults.py:283-329`). Noise choice must
    be independent of theory branch.
14. The tau auto-noise is an effective white depth, explicitly labelled legacy/provisional
    (`survey_defaults.py:276-279`). The user approved it for the first algorithm-validation
    result only. Every artifact and figure containing `gtau` must say `PROVISIONAL TAU NOISE`;
    it is not a survey forecast.
15. Current map cache checks do not hash the five varied parameters, mass/redshift selection,
    or full theory contract (`pasted_map_cls_validation.py:49-80`). A scan could silently reuse
    one map at multiple parameter points.
16. The old wrapper relies on the implicit HOD seed 42 in `src/get_sim_maps.py`; it does not
    record an explicit seed in the saved map contract.
17. `notebooks/pasting/paste_backlight_utils.py:519-538` repaints the full halo array in each
    loop when more than one chunk is requested. A much larger low-mass catalog would expose
    this bug. It must be fixed and regression-tested before any production scan.
18. The HMC products behind the referenced final-round figures failed their existing
    convergence gates. Those figures are diagnostics, not accepted posterior constraints
    (`knowledge/60-projects/SBI_validate/analytical-hmc-sbi.md`). Their 17-band/full-theory
    posterior also targets the wrong forward model, so it cannot be the round-1 proposal. The
    proposal must come from the newly accepted 12-band resolved-theory SBI.
19. The user-selected catalog builder,
    `notebooks/xDESI/abacus_lightcone_catalog.py`, already computes the accepted working mass,
    derives a compatible radius and records the raw provenance
    (`abacus_lightcone_catalog.py:257-278,420-476`). It still needs an experiment-specific
    c0000 configuration, strict `z > z_min`, an upper mass bound, source-row indices, all-shard
    discovery, an explicit `N_interp` completeness cut and unambiguous provisional mass
    metadata. Extend this helper rather than create a second catalog pipeline.
20. `default_noise_dict()` returns band-averaged noise in the current SO path. A 36-by-36
    bandpower covariance or twelve band averages cannot be passed to `synalm` to make a map.
    Map noise requires the underlying unbinned integer-ell field-noise curves in the same
    observable and beam basis used by the covariance.
21. The current SBI covariance represents sky coverage only through scalar values
    `fsky_g=0.34` and `fsky_so=fsky_k=0.40`; it has no map masks and is diagonal between ell
    bands. The user has replaced this with one common `fsky=0.4` mask for g, y, CMB-kappa and
    tau. Every scalar fallback default must also become 0.4, but the accepted covariance must be
    generated from the exact common mask and saved NaMaster workspaces rather than dividing a
    full-sky covariance by 0.4.
22. The earlier storage policy discarded most maps after measuring Cls, and the earlier design
    rejected nearby theta values. Neither is compatible with later higher-order-statistic use:
    the maps must be retained, and randomized bank points must remain draws from an explicitly
    normalized proposal. Forced reference/diagnostic points and any deterministic acquisition
    have no sampling density and must be flagged rather than assigned fabricated importance
    weights.

## How to verify

### 0. Materialize the accepted physical contract before editing analysis code

Write the following decisions into one immutable YAML file. Every product embeds the YAML
contents and SHA256, not only its path.

1. **Simulation:** `AbacusBacklight_base_c0000_ph000`, read-only. Build a new selected catalog;
   never reuse the old pre-cut HDF5 or the c9999 xDESI default.
2. **Mass:** use `M_working = InterpolatedN * ParticleMassHMsun`, in `Msun/h`, and provisionally
   treat it as `M200c`. Save both `N_interp` and `M_particle_proxy_hMsun`; compatible legacy
   columns such as `M200c_hMsun` may be written for the painter, but the required metadata is
   `mass_semantics=interpolated_particle_count_proxy_treated_as_M200c` and
   `mass_definition_status=provisional_assumption`. Never replace it with `N`, `SO_radius`, or
   an undocumented conversion.
3. **Lower mass bound:** use the user-selected operational floor
   `M_particle_proxy_hMsun >= 5e11 Msun/h` over the whole selected redshift range. Read and
   record the actual c0000 `ParticleMassHMsun` at build time and record the corresponding
   (possibly non-integer) `N_interp` value; the mass threshold itself is the authoritative
   predicate and is not rounded to a particle count. Retain the pre-registered 125- and
   150-particle sensitivity branches. The minimum row present is diagnostic only, and this
   operational floor is not advertised as a demonstrated completeness limit.
4. **Selection:** use
   `(z > 0.3) & (z < 0.5) & (M_working >= 5e11 Msun/h) & (M_working < 1e16 Msun/h)` for every
   pasted field, galaxy occupation and resolved theory integral, with the exactly equivalent
   mass lower bound recorded in the analytic theory configuration. Freeze this mass floor before
   the fiducial validation; never lower it after seeing agreement or contours.
5. **Kappa:** `gkappa` means galaxy x CMB convergence. Require `kappa_source="cmb"`; reject
   `map_rhom` and LSST shape-noise branches.
6. **Tau:** use the current effective tau noise only for this first algorithm-validation
   exercise. Stamp every product and figure `PROVISIONAL TAU NOISE`.
7. **Forward model and response:** use one fixed-phase conditional resolved-halo target
   throughout,

   ```text
   response(theta) = Cl_paste_resolved(theta, c0000/ph000, hod_seed0)
                   - Cl_paste_resolved(theta_ref, c0000/ph000, hod_seed0)

   mu_mock(theta)  = mu_theory_resolved(theta_ref) + response(theta) .
   ```

   The same halo rows, galaxy map, HOD seed, estimator and reference paste are used in both
   response terms. Common random numbers reduce, but do not generally eliminate,
   `delta(theta,phase)-delta(theta_ref,phase)`. The response anchor cannot test an absolute
   theory-versus-paste offset at `theta_ref`; Step 3 tests that offset separately. Caps and HOD
   seeds are sensitivity checks, not substitutes for independent simulation phases.
8. **Common observation:** choose `theta_truth=theta_ref` for the headline comparison and freeze
   one held-out map-noise seed before fitting. Build the reference signal maps, synthesize y,
   CMB-kappa and tau Gaussian harmonic-space noise fields from the same unbinned `N_ell` inputs
   and observable/beam basis used to construct the covariance, add those noise fields to the
   corresponding maps, and run the exact estimator:

   ```text
   m_g_obs     = m_g_paste(theta_ref, phase, hod_seed0)
   m_y_obs     = m_y_paste(theta_ref, phase)     + n_y(map_noise_seed)
   m_kappa_obs = m_kappa_paste(theta_ref, phase) + n_kappa(map_noise_seed)
   m_tau_obs   = m_tau_paste(theta_ref, phase)   + n_tau(map_noise_seed)
   d_obs       = estimator(m_g_obs, m_y_obs, m_kappa_obs, m_tau_obs)
   ```

   Under the current covariance, field cross-noise spectra are zero, so use independent,
   named random streams for y, kappa and tau. If nonzero cross-noise is later introduced, draw
   the fields jointly with a per-ell positive-semidefinite spectral matrix. The sampled HOD
   galaxy map already realizes galaxy shot noise, so add no Gaussian galaxy-noise map. HMC,
   theory SBI and mock SBI all condition on this exact saved, map-measured vector. The mock
   model mean remains the theory-anchored paired response; the fixed reference phase and map
   noise occur once in the observation, while the frozen full covariance describes uncertainty
   about the ensemble mean.
9. **Common mask:** use one deterministic axisymmetric scalar cap for g, y, CMB-kappa and tau.
   Apply a fixed 1-degree C2 taper to control sharp-edge mode coupling, and solve the untapered cap
   radius once so the final pixelized mask obeys
   `mean(mask**2)=0.4` at `nside=512`. This equation defines the quoted `fsky=0.4`; also record
   `mean(mask)`, `mean(mask**4)`, nonzero support fraction, cap center/radius, taper type/width,
   coordinate frame, ordering and SHA256. Do not change the taper after inspecting spectra. The same saved mask array is passed to every
   `NmtField`. Set `fsky_g=fsky_y=fsky_kappa=fsky_tau=0.4` in scalar diagnostic defaults, but
   never divide the exact mask covariance by another 0.4 factor. The selected lightcone must
   cover the entire nonzero mask support; missing coverage is a hard failure.
10. **Angular support:** production maps use `nside=512` and `lmax=1535`, corresponding to a
   right-exclusive limit of 1536. Retain only the 12 existing bins with complete support:
   native indices 0--11, last real upper edge 1267.91456463. The selected vector has 36
   elements ordered `gy[0:12],gkappa[0:12],gtau[0:12]`. Do not call the current center-based
   `ell_max=1536` selector, because it admits a thirteenth partial band.

   | bin | center | real edge `[low, high)` | integer ell support |
   |---:|---:|---:|---:|
   | 0 | 90.3570 | `[80.0000, 100.7140)` | 80--100 |
   | 1 | 113.7527 | `[100.7140, 126.7915)` | 101--126 |
   | 2 | 143.2062 | `[126.7915, 159.6210)` | 127--159 |
   | 3 | 180.2860 | `[159.6210, 200.9509)` | 160--200 |
   | 4 | 226.9666 | `[200.9509, 252.9822)` | 201--252 |
   | 5 | 285.7340 | `[252.9822, 318.4857)` | 253--318 |
   | 6 | 359.7178 | `[318.4857, 400.9498)` | 319--400 |
   | 7 | 452.8578 | `[400.9498, 504.7659)` | 401--504 |
   | 8 | 570.1142 | `[504.7659, 635.4626)` | 505--635 |
   | 9 | 717.7313 | `[635.4626, 800.0000)` | 636--800 |
   | 10 | 903.5702 | `[800.0000, 1007.1403)` | 801--1007 |
   | 11 | 1137.5275 | `[1007.1403, 1267.9146)` | 1008--1267 |

   Native bin 12 is rejected because its real edge is
   `[1267.9146,1596.2099)`, even though its center is below 1536.
11. **Persistent map bank:** retain every successful expensive execution, whether it belongs to
    training, validation, sealed holdout, resolution/profile/HOD/cap diagnostics or a failed
    scientific gate. Save full-sky, unmasked signal maps and the field-noise components; the
    noisy map is reconstructed as their declared sum before applying the common mask. A
    bitwise-identical fixed galaxy/HOD map may be stored once by content hash and referenced by
    every compatible execution. Never discard an expensive map merely because its Cl vector has
    been extracted.
12. **Interpretation:** the result is resolved-model, fixed-phase conditional-response and
    provisional in tau. It is not a full-halo-model or phase-ensemble survey forecast.

**Remaining stop gates:** before any expensive paste, read and freeze the actual c0000 particle
mass and the documentation supporting the complete-particle threshold; audit the common-cap
coverage; and freeze the mask taper, signal/noise beam, pixel-window and observable basis. These are
read-only implementation audits, not unanswered scientific choices. No production run follows
until they pass and the user separately authorizes execution.

### 1. Create one fail-closed experiment contract

Add a small loader/validator, not another large notebook. The contract should contain:

- parent simulation name and phase, source paths and hashes;
- raw `InterpolatedN` field and particle-mass header provenance; the working-mass equation,
  provisional proxy semantics, exact `M_min=5e11 Msun/h`, equivalent particle count, 125/150-particle
  sensitivity cuts, `M_min_present`, rejected counts, row count and exact selection predicate;
- cosmological parameters read from the source catalog;
- `M_min`, `M_max`, `z_min`, and `z_max` in one declared convention;
- five varied parameters, their order, fiducials, original box priors, normalized physical- and
  probit-space log densities and transform Jacobian convention;
- every fixed parameter;
- `gy,gkappa,gtau` order, native band indices 0--11, exact real/integer ell edges, the common
  scalar NaMaster workspace/windows, pair-specific external transfer products, field beams,
  pixel windows, `nside=512` and `lmax=1535`;
- common mask generator and saved mask hash: cap center/frame, solved pre-taper radius,
  1-degree C2 taper, HEALPix ordering, `mean(w)`, `mean(w**2)=0.4`, `mean(w**4)`, support
  fraction, and proof that every selected lightcone row lies in a catalog covering its support;
- covariance, correlation-scaled Cholesky, retained rank and SHA256 values;
- unbinned integer-ell y, kappa and tau noise spectra, cross-noise policy, observable/beam
  basis, map-noise algorithm and seeds; galaxy shot-noise/HOD policy; band-averaged noise,
  covariance and `fsky` provenance;
- map projector, profile support, HOD policy and every random seed;
- map-bank root, schema/version, dtype, compression, checksum and atomic-write policy; map roles,
  content-addressed shared-field references, storage forecast and free-space gate;
- each round's normalized proposal family, component parameters/weights, sample and log-density
  implementation version, serialization hash and fit diagnostics; per-map theta in both
  coordinate systems, `log_p0`, `log_q_generating`, all-round proposal log densities, round,
  sampling role, RNG state and importance-eligibility flag;
- every boundary unit: angles in degrees, redshift dimensionless, velocity in km/s if used,
  the declared mass proxy/definition in `Msun/h`, radii and distances in `Mpc/h` or `Mpc`,
  dimensionless `delta_g`, y, tau and kappa maps, dimensionless cross-`Cl`, covariance in
  `Cl^2`, and dimensionless beams, pixel windows and bandpower matrices;
- source-code and parameter-file hashes.
- literal flags `forward_model=map_matched_resolved`,
  `mock_target=paired_fixed_phase_conditional_response`, and
  `tau_noise_status=provisional`,
  `mass_definition_status=provisional_assumption`, and
  `observation_noise_domain=harmonic_map`,
  `common_mask_fsky2=0.4`, and
  `map_retention=all_successful_expensive_executions`.

Every reader must compare the complete contract and fail on a mismatch. Do not use filename
agreement or `np.allclose` as provenance.

Suggested implementation files are deliberately few:

```text
notebooks/SBI_validate/three_probe_mock_experiment.yaml
notebooks/SBI_validate/submit_three_probe_catalog.sbatch
notebooks/SBI_validate/validate_three_probe_catalog_preflight.py
notebooks/SBI_validate/three_probe_mock_contract.py
notebooks/SBI_validate/validate_three_probe_pasting.py
notebooks/SBI_validate/pasted_three_probe_simulator.py
notebooks/SBI_validate/three_probe_map_bank.py
notebooks/SBI_validate/run_pasted_three_probe_sbi.py
notebooks/SBI_validate/plot_three_way_five_parameter_getdist.py
```

During implementation add only two focused regression files:

```text
tests/test_sbi_three_probe_namaster.py
tests/test_sbi_three_probe_map_bank.py
```

The first uses synthetic scalar maps to test the common-mask hash/fsky moments, weighted-mean
subtraction, coupled-to-decoupled recovery, saved-window theory projection, transfer applied
once, `coupled=False` covariance and band-major extraction. The second tests atomic map
round-trip/checksums, content-addressed references, prior/probit Jacobians, normalized proposal
round-trip, IID/forced role flags and the 15% prior-support bound. Keep these as small synthetic
fixtures; they do not read the Abacus tree or launch jobs.

Extend `notebooks/xDESI/abacus_lightcone_catalog.py` for the catalog product. Reuse the current
covariance, HMC, theory-SBI and simulator-native modules. Do not copy their likelihood or
parameter definitions into new files.

Keep the implementation deliberately plain:

- one experiment YAML is the source of truth;
- one small frozen contract object validates and hashes it;
- the existing catalog helper only builds/validates HDF5 rows;
- one simulator function maps `theta -> saved field maps + 36-vector` and returns provenance;
- one small map-bank module performs atomic HDF5 writes, hashes and manifest validation;
- one orchestration script owns proposals, checkpoints and the adaptive ledger;
- one plotting script reads only accepted manifests;
- notebooks contain plots and commentary, not reusable logic;
- no framework, plugin registry, inheritance hierarchy or duplicate parameter table is added.

Every command supports a cheap `--validate-only` or one-theta mode before any batch submission.
Atomically finalize one manifest row only after its map bundle and checksums are complete, so an
interruption cannot create a valid-looking partial map or lose an expensive completed paste.

**Gate 1:** one command prints exact equality of the observation labels, 12-band edges,
pair-specific bandpower operators, 36-by-36 covariance, Cholesky, priors, fixed values, catalog
selection, common-mask hash/fsky moments, resolved target, mass-proxy semantics, noise observable
basis and cosmology hashes for all three methods. It also proves that no selected window has
weight above ell 1535, every field receives the exact same mask array, and no consumer silently
substitutes a different mass definition, old galaxy `fsky=0.34`, or a bandpower-space noise draw.

### 2. Build the selected catalog and matching theory configuration

1. Use `notebooks/xDESI/abacus_lightcone_catalog.py` as the sole catalog builder. Add one
   experiment-specific c0000/ph000 YAML; do not change the canonical c9999 production config.
   Preserve the helper's streaming, controlled output roots, compression and atomic rename.
2. Add an explicit working-mass contract to that YAML: source root/file template,
   `InterpolatedN` dataset/aliases, `ParticleMassHMsun` header field, raw/final units, accepted
   multiplication, provisional interpretation, validity/sentinel policy, complete-particle
   threshold and source checksums. The only accepted mode is named explicitly, for example
   `interpolated_particle_proxy_as_m200c`; there is no generic `auto` or silent fallback mode.
3. Run a tiny read-only header/schema audit first. Confirm that every selected shard exposes
   the same particle mass and valid interpolated count semantics. The inspected c0000 shard has
   `ParticleMassHMsun=4.200431928473044e9` and `MinL1HaloNP=35`. The public AbacusSummit
   recommendation of at least 100 particles for reliable interpolated halo properties
   ([Hadzhiyska et al. 2022](https://academic.oup.com/mnras/article/509/2/2194/6408495)) is
   context, not Backlight-specific validation. The authoritative primary selection is the
   later user decision `M_particle_proxy_hMsun >= 5e11 Msun/h`; derive and record its exact
   particle-count equivalent from every selected source header without rounding it.
4. Stream every matching `lightcone_halo_info_*.asdf` shard in c0000/ph000 without changing
   source order. Compute `N_interp` and `M_particle_proxy_hMsun` once and save
   `(source_file_index, source_row_index, HaloIndex)` for every retained row. Do not load the
   full selected catalog into memory.
5. Populate painter-compatible `M200c_hMsun` from `M_particle_proxy_hMsun` only under the
   explicit user-approved mode. Required attributes include the literal equation,
   `mass_semantics=interpolated_particle_count_proxy_treated_as_M200c`,
   `mass_definition_status=provisional_assumption`, particle mass, units and user-decision
   provenance. Do not write `overdensity=200` or `measured_SO_mass=true`, because neither is a
   fact about the source field.
6. Compute redshift from `InterpolatedComovingDist` with the complete c0000 cosmology and test
   the distance-to-redshift interpolation at shell anchors. Apply literal strict bounds
   `z > 0.3` and `z < 0.5`; the helper's current lower-inclusive comparison must change.
7. Record `M_min_present`, but apply the exact mass predicate
   `M_particle_proxy_hMsun >= 5e11 Msun/h` while building the production sample. Store its exact
   particle-count equivalent from the source header without rounding. Also build or stream-count
   the pre-registered 125- and 150-particle sensitivity samples without changing any other field.
8. Derive painter-compatible `R200c` from the accepted working mass and c0000 critical density
   exactly once, with explicit physical/comoving units and the same provisional status. Do not
   use native `SO_radius` as `R200c`.
9. Apply
   `(M_particle_proxy_hMsun >= 5e11) & (M_particle_proxy_hMsun < 1e16) & (z > 0.3) & (z < 0.5)` before
   painting any of `g`, `y`, `tau`, or CMB-kappa. The current galaxy-only redshift filter is
   forbidden.
10. Save counts rejected by each validity/selection reason, retained extrema, source manifest,
   selection hash, row-order hash and source-header cosmology. Finalize discovered extrema and
   counts at atomic close; do not guess them in the initial HDF5 attributes.
11. Reconstruct the GODMAX objects only after the c0000 cosmology and new mass/redshift grids are
   set. Apply the identical numerical working-mass bounds to galaxy occupation and every
   resolved y/electron/matter 1-halo and 2-halo integral; disable unresolved/missing-field
   completion. The theory label must state that its integration variable is being identified
   with the particle-count proxy for this comparison.
12. Build the normalized lens kernel from the selected catalog n(z), not a stale 0.4--0.6
   top-hat. Verify normalization and preserve the same kernel in all consumers.
13. Add small synthetic fixtures for source order, strict redshift boundaries, count-to-mass
   conversion, invalid counts, particle-threshold inclusion, upper mass exclusion, all-shard
   discovery and required provisional metadata. Keep the complete Abacus input tree read-only.

**Gate 2:** 100% of saved rows satisfy the frozen working-mass and redshift predicate;
catalog and theory use the same numerical bounds, units and c0000 cosmology; the actual header
particle mass and mass-floor decision are recorded; the lens kernel passes its existing
normalization invariant; and HMF/bias residuals are reported over the accepted range. Repeating
the complete pipeline at 125 and 150 particles must move every theory-posterior mean by less
than 0.2 marginalized sigma and every 68% width by less than 10%; otherwise the `5e11 Msun/h`
result is not accepted and the most conservative stable cut becomes the common selection in a
new pre-registered run. A paste/theory PTE cannot override this gate.

**Null control:** rows outside the new window are absent, while all non-derived retained source
values and their order are bitwise unchanged. Changing only the mass threshold changes only the
rows admitted by that threshold and derived catalog hashes. Removing the explicit proxy mode
from a synthetic fixture must fail rather than silently select `N`, `SO_radius`, or another mass.

### 3. Re-do the noise-free theory-versus-paste validation

This step replaces the computational part of
`01_validate_fiducial_theory_datavector.ipynb`. Keep the notebook only as a thin report.

1. Fix the fiducial five-parameter point and an explicit HOD seed.
2. Generate `g`, `y`, `tau`, and CMB-kappa signal maps from the selected halos.
3. Build the common cap mask once, save it, and use it bitwise for `g`, `y`, `kappa` and `tau`.
   All four are spin-0 fields. Subtract each field's weighted mean using that same mask before
   constructing `nmt.NmtField`; for the galaxy field, construct overdensity and its weighted
   mean from the masked selected catalog rather than normalizing it on a different sky area.
4. Build one scalar `NmtBin` for native bands 0--11 and one scalar `NmtWorkspace` from the common
   mask, `lmax=1535` and the frozen mask-harmonic settings. For every cross-spectrum, compute
   `nmt.compute_coupled_cell(field_g, field_a)` and then
   `workspace.decouple_cell(coupled_cell)`. Cross-field map noise is independent, so subtract no
   additive cross-noise bias. Save coupled spectra, decoupled spectra, effective ells and the
   complete `workspace.get_bandpower_windows()` output.
5. Use one forward-convolved observable convention. `NmtField` receives no hidden beam
   deconvolution. Record a transfer for every field: the HEALPix scalar pixel window for each
   pasted HEALPix map, multiplied by the physical painter/instrument smoothing already present
   for that field and by no second beam. Generate noise maps with the corresponding observed-map
   noise `N_ell_obs = transfer_field**2 * N_ell_deconvolved`. Build the analytic pair operator
   from `window @ (transfer_g * transfer_a * C_ell_theory)`. This follows the existing saved-
   window convention and avoids unstable high-ell map deconvolution.
6. Apply each saved pair operator to smooth resolved theory for HMC, theory SBI, simulator
   anchoring and covariance inputs. Comparing theory only at `ell_eff`, averaging with a
   full-sky mode-count operator, or applying the transfer after the NaMaster window is forbidden.
   Assert that every retained response row has zero input support above ell 1535.
7. Keep a full-sky `anafast` measurement only as a null control with its own full-sky operator
   and `fsky=1` covariance. It never enters the headline likelihood and cannot replace the
   masked result if the latter fails.
8. First compare pasted maps to **map-matched resolved theory**. This is the pipeline null.
9. Optionally compare to full theory as a labelled diagnostic of missing low-mass, diffuse and
   unresolved components. It is not an inference target and cannot veto an otherwise valid
   resolved-model experiment.
10. Repeat a small set of parameter anchors: fiducial, posterior center, and at least one
   low/high displacement along each weakly constrained direction. Fiducial-only agreement
   does not validate a 5-D posterior volume.
11. Repeat the fiducial at `nside=1024` using the same physical smoothing beam, halos, HOD seed,
   profile support and 12 physical band windows. Do not let the painter's nside-dependent default
   smoothing change this control. Compare forward-pixel-windowed and explicitly deconvolved
   conventions. Also repeat one finer theory ell grid and two profile-support radii chosen
   before looking at the result.
12. Repeat with one denser mass grid, one denser redshift grid and the supported alternate
   interpolator. Require the result to pass the same componentwise stability gate; this is
   essential because both requested integration bounds differ from the old product.
13. If the galaxy catalog remains stochastic, run enough HOD seeds that its standard error is
   small compared with the frozen survey error. Because the five varied parameters do not
   change the HOD, build the galaxy map once and reuse it bitwise at every theta. Prefer a
   validated expected-occupation field; otherwise use one frozen realization and common HOD
   seed in every response difference.
14. Fix scalar-field signs independently of the shared profile/theory code: central stacks of
    y, CMB-kappa and tau at retained galaxy/halo positions must have the pre-registered positive
    sign. Randomized-position and shuffled-catalog stacks must be consistent with zero.

Use per-probe and joint whitened residuals, chi-square/PTE, and a residual-versus-ell slope.
Median map/theory ratios are diagnostics only; they hide covariance and sign errors.

**Provisional pre-registered Gate 3, to approve before execution:**

- joint and per-probe PTE are not below 0.01;
- no individual retained band has absolute pull above 4;
- no significant coherent residual slope remains at two-sided p below 0.01;
- resolution/profile-support changes move every retained component by less than 0.25 of its
  frozen survey sigma;
- the nside512-versus-1024 response has joint `Delta chi2 < 1` and induces less than 0.1 sigma
  shift in every marginalized parameter;
- mass-grid, redshift-grid and interpolator changes satisfy that same componentwise threshold;
- y, CMB-kappa and tau central stacks have the physical positive sign, while randomized and
  shuffled-position stack nulls pass their pre-registered zero test;
- the parameter-independent galaxy-map/`gg` null is unchanged when only the five baryonic
  parameters move;
- g, y, kappa and tau record the identical common-mask SHA; the saved mask has
  `abs(mean(mask**2)-0.4) < 1e-6`, and no configuration or covariance metadata contains the old
  galaxy value 0.34;
- Gaussian-map injection through the mask, common workspace and saved transfer-window operators
  recovers the input bandpowers within the pre-registered Monte Carlo interval; adding signal
  only above ell 1535 at nside1024 changes every retained band by less than 0.25 of its frozen
  survey sigma;
- `map_rhom` never enters the inference vector.

These thresholds must not be weakened after seeing the maps. A failure localizes a problem;
it is not permission to trim ell bands or enlarge errors.

**Stop condition:** any failure of the resolved theory/paste, operator, sign, grid, support or
resolution gates stops inference. A failure confined to the optional full-theory diagnostic is
recorded but does not block this explicitly resolved-model exercise.

### 4. Freeze covariance, observation and noise policy

Build one fresh 36-by-36 covariance for `gy,gkappa,gtau` on native bands 0--11. Use resolved
signal auto/cross spectra with the accepted survey assumptions, but decouple the survey-noise
choice from `theory_mode` so resolved theory still receives official SO y and CMB-kappa noise.
For the selected masked headline branch, use the same frozen common mask and NaMaster
bandpower windows in measurement and `gaussian_covariance(..., coupled=False)`. Extract blocks
in band-major order before converting once to the stored probe-major 36-vector; retain all
cross-probe and cross-band terms produced by the masks. Do not use three independent diagonal
errors. Set every scalar survey-area fallback to 0.4, including galaxy and tau, but retain the
scalar-fsky, ell-diagonal covariance only as a diagnostic; the exact common-mask covariance is
the accepted likelihood. Tau noise remains explicitly provisional.

#### 4.1 One field-level noise source of truth

Refactor the existing survey-noise loader minimally so it returns the unbinned integer-ell
field spectra used by both consumers:

```text
N_ell_fields, noise_contract
    -> bandpower averaging and Gaussian covariance
    -> harmonic Gaussian map-noise generator
```

The generator must not factorize the 36-by-36 bandpower covariance into pixels or alms. That
matrix describes fluctuations of estimated cross-bandpowers, including signal/sample-variance
terms; it is not a map angular-power spectrum. Instead, synthesize the map fields from the same
unbinned `N_ell` primitives that enter that covariance.

- y uses the selected official SO Deproj-2 curve.
- CMB kappa uses the selected official SO iterative MV convergence curve.
- tau uses the current effective white-depth curve and remains provisional.
- Under the frozen covariance model, y, kappa and tau cross-noise spectra are zero, so draw
  independent named Gaussian-alm streams. If the covariance later includes nonzero field
  cross-noise, replace this with a joint per-ell spectral-matrix draw and rerun every noise gate.
- The sampled HOD galaxy catalog already produces the galaxy shot-noise realization. Set the
  covariance `N_ell^gg` from the realized surface density and do not add a Gaussian galaxy-noise
  map. If expected occupations replace the sampled catalog, explicitly choose a Poisson galaxy
  realization instead; never use both policies.
- Load enough integer-ell support for every mode used by the estimator/window coupling. No
  extrapolation beyond a noise table is allowed.

Before generation, freeze one observable-basis table for every field: sky/deconvolved versus
beam-convolved map, beam transfer, HEALPix pixel window, painter smoothing, estimator
deconvolution and covariance convention. Use the forward-convolved map basis frozen in Step 3:
NaMaster decouples the mask but does not secretly remove a beam/pixel transfer. Current SO noise
curves are documented in a beam-deconvolved sky-field basis, so multiply them by the square of
the appropriate field transfer before generating observed noise maps. Apply the same transfer
product to analytic signal and covariance inputs before the saved NaMaster windows. Every beam
and pixel window appears exactly once. A mismatch is a stop, not a correction chosen after
viewing spectra.

#### 4.2 Make the common noisy observation at map level

For the fixed-phase conditional-response branch, the model and observed-vector construction are

```text
response(theta) = Cl_paste_resolved(theta, phase, hod_seed0)
                - Cl_paste_resolved(theta_ref, phase, hod_seed0)
mu_mock(theta)  = mu_theory_resolved(theta_ref) + response(theta)

m_g_obs         = m_g_paste(theta_ref, phase, hod_seed0)
m_y_obs         = m_y_paste(theta_ref, phase)     + n_y(seed_obs)
m_kappa_obs     = m_kappa_paste(theta_ref, phase) + n_kappa(seed_obs)
m_tau_obs       = m_tau_paste(theta_ref, phase)   + n_tau(seed_obs)
d_obs           = exact_estimator(m_g_obs, m_y_obs, m_kappa_obs, m_tau_obs)
```

Use one pre-registered held-out `seed_obs`. Save signal-map hashes, noise-map/alms hashes,
observed-map hashes and the measured vector. The full likelihood covariance remains the same
saved 36-by-36 `C`; it is not separately drawn and added to `d_obs`. The reference N-body phase,
HOD realization and map noise are actual fluctuations in the observed maps, while `C` describes
their ensemble uncertainty in the likelihood. The observed vector therefore contains the
absolute fixed-phase reference residual. Because all three methods condition on this same
vector, that residual cannot masquerade as a difference caused merely by using different data;
the Step-3 absolute validation must show it is acceptable before inference.

The paired model response still uses identical noiseless signal maps, phase, HOD seed and
estimator at `theta` and `theta_ref`. It reduces common fixed-phase fluctuations but retains
`delta(theta,phase)-delta(theta_ref,phase)`, so it remains a conditional response rather than a
phase-ensemble response.

#### 4.3 Validate map noise and covariance separately

Run these cheap, pre-registered tests before freezing `d_obs`:

1. Noise-off and same-seed replay: zero noise recovers the signal-only estimator output; the
   same seed reproduces alms, maps and bandpowers; changing the noise seed leaves every signal
   map and catalog hash unchanged.
2. Noise-only maps: repeated realizations reproduce each input unbinned `N_ell` and its exact
   band/window average within Monte Carlo uncertainty. Empirical y-kappa, y-tau and kappa-tau
   noise cross-spectra are consistent with the frozen zero-cross-noise model.
3. Fixed signal plus repeated noise: remeasure the exact estimator. The empirical mean must
   recover the signal cross-bandpowers, and the empirical covariance must match the
   **conditional map-noise contribution** calculated for the fixed signal maps. Do not compare
   this conditional scatter to the full survey covariance and call a difference a failure.
4. Full Gaussian estimator suite: draw cheap joint Gaussian signal maps for g, y, kappa and tau
   from the frozen theoretical field auto/cross spectra, add the field-noise maps, apply the
   actual mask/beam/pixel windows and estimator, and verify the empirical 36-by-36 covariance
   against `C`, including off-diagonal probe blocks. The per-ell signal spectral matrix must be
   positive semidefinite without a physics-changing repair. These are Gaussian map tests, not
   expensive N-body pastes and not a substitute for Step 3.
5. Galaxy-noise control: the sampled HOD map plus its realized surface density supplies exactly
   one shot-noise realization and one `N_ell^gg` covariance term. No additional galaxy map noise
   exists. Changing only the HOD seed changes the galaxy realization and its recorded density,
   not y/kappa/tau noise streams.

Freeze the number of cheap realizations from a Monte Carlo precision calculation before running
them; report confidence intervals, not an arbitrary percent-difference threshold. All map-noise
draws and remeasurements are cheap operations on cached signal products and are not expensive
paste executions, but their count, runtime and storage still appear in the evidence ledger.

**Gate 4:** covariance is finite and symmetric, has positive diagonal, is full-rank in the
declared basis, needs zero correlation-space jitter, and reconstructs from its Cholesky to
machine precision. All five validation groups above pass their pre-registered Monte Carlo
tests; every band-noise average uses the exact signal-window support; the observable/beam basis
and common-mask hash match bitwise across the map generator, estimator, theory wrapper and
covariance metadata; every field and scalar fallback reports fsky 0.4; and no galaxy,
field noise, beam, pixel window or bandpower-space covariance draw is applied twice. Re-measure
five diagnostic theta values on two predeclared disjoint caps and with two alternate HOD seeds
as a falsifier for severe conditional-response sensitivity. This is not evidence of a phase
ensemble. If cap/HOD changes exceed the componentwise or posterior-stability gates, use
deterministic expected occupations or model the stochastic nuisance explicitly; do not hide it
inside an inflated covariance.

### 5. Run theory HMC and theory SBI on the common observation

Run both after `d_obs` is frozen. Use the exact same resolved theory, 36-element observation,
36-by-36 covariance, Cholesky, mass/redshift selection, priors and fixed values. Entry points
must assert `theory_mode="map_matched_resolved"`; a full-theory default is a hard error.

#### HMC

- Enable JAX and NumPyro float64 before arrays are created.
- Check finite, nonzero gradients at the fiducial, posterior center and prior corners.
- Use four or more chains with dispersed initial states, not four copies of the fiducial.
- Adapt a dense mass matrix and set tree depth based on a pre-run, never by accepting
  saturation after the fact.
- Preserve chain structure and all sampler diagnostics.

**HMC gate:** maximum rank-normalized R-hat at most 1.01; minimum bulk and tail ESS at least
400 for every parameter; zero divergences; zero tree-depth saturation; finite positive final
step sizes; mutually consistent chains. Report absolute best-fit chi-square, retained rank,
five varied parameters, expected `rank - 5`, and per-probe posterior-predictive residuals.

#### Theory SBI

- Keep the original box prior and exact probit transform.
- Train with the same observation and frozen covariance.
- Run at least three independent network seeds.
- Use held-out theory simulations plus SBC/TARP or empirical coverage on cheap synthetic
  observations. A train/validation loss is not a posterior calibration test.

**Theory HMC versus SBI gate:** for every marginal, mean displacement is at most 0.2 pooled
sigma and the 68% width ratio is within 10%; maximum absolute correlation-matrix difference is
at most 0.1. Nominal 68% and 95% coverage must lie inside their binomial confidence intervals.
Posterior-predictive residuals must agree in the full 36-vector, not only in compressed space.

**Absolute goodness gate shared by all three final posteriors:** freeze this rule before opening
the observation or mock holdouts.

- At the global maximum-likelihood point, compute
  `T_joint = (d_obs-mu)^T C^-1 (d_obs-mu)` and
  `T_p=(d_p-mu_p)^T C_pp^-1(d_p-mu_p)` for `gy`, `gkappa`, and `gtau`. The per-probe blocks use
  the exact marginal covariance blocks, not three diagonal error arrays.
- Calibrate all four statistics with one pre-registered parametric bootstrap of at least 2,000
  cheap observations generated at `theta_ref` from the frozen covariance. Apply the identical
  deterministic multi-start bounded fit to every replicate and to `d_obs`; do this once for the
  resolved-theory forward model shared by HMC/SBI and once for the accepted pasted-response
  emulator. Define each PTE as the fraction of fitted replicates with `T_rep >= T_obs`.
- Require every joint and per-probe PTE to lie in `[0.01,0.99]` and no retained component to have
  absolute pull above 4. Also report `chi2_min`, retained rank, five fitted parameters and the
  nominal `retained_rank - 5` expectation as an independent sanity check; the empirical
  bootstrap, not a post-hoc choice between references, determines the PTE gate.
- Continue to report posterior-predictive joint/per-probe residuals and PTEs. They must also lie
  in `[0.01,0.99]`, but they do not replace the fitted parametric-bootstrap test above.
- A failure by any method or probe blocks the final constraint figure. It may be saved only as
  `DIAGNOSTIC -- ABSOLUTE GOODNESS FAILED`; posterior agreement cannot override this gate.

Do not use the current final-round PDFs as the accepted baseline; their HMC source chains failed
the existing gate and their mass/redshift contract differs.

### 6. Prove the adaptive low-budget method before spending on maps

Hide the analytic theory callable behind the exact mock-simulator interface. Cap it at the same
three-round schedule and the same number of unique parameter points planned for pasted maps.
Run the strong-stop, normal and contingency branches below. This is the oracle budget test.

The preferred low-budget statistic is a fixed five-dimensional score

```text
t(d) = F^(-1/2) J^T C^(-1) [d - mu_ref]
F    = J^T C^(-1) J
```

where `J`, `C`, `mu_ref`, and the orientation of `F^(-1/2)` are frozen before mock generation.
Use this score for proposal diagnostics and for freezing the next round's normalized mixture,
not for deterministic candidate ranking. Always retain and reconstruct the raw 36-vector for
likelihood and posterior-predictive checks.

Primary inference choice: learn a multi-output Gaussian-process emulator for the paired pasted
response. Start with the complete whitened 36-vector; use a fixed PCA/score basis only if the
analytic oracle shows it improves heldout accuracy without losing posterior information.
Reconstruct the full vector and propagate emulator uncertainty into a Gaussian synthetic
likelihood. The final density is
`p0(theta) * Normal(d_obs; mu_emulated(theta), C + C_emulator(theta))`; it never contains the
simulation-design density. This is much more credible with at most 480 expensive training
points than retraining the existing raw-51D MDN SNPE, which currently uses 65,536 theory
simulations. A neural SNLE is the second choice only if the analytic oracle shows that the
GP/synthetic likelihood fails.

Raw SNPE-C remains an option only if the oracle test passes. In that case every external
proposal must have evaluable normalized `sample` and `log_prob` methods and must be supplied as
the proposal when simulations are appended.

**Gate 5:** the oracle must pass the same HMC-agreement thresholds as theory SBI at the exact
budget branch proposed for real pastes; full-vector posterior predictives agree; the final
heldout likelihood-error gates pass; and at least three cheap GP initialization/training seeds
are stable. If the oracle cannot pass by the 480-training-point contingency cap, change the
inference architecture or request a larger budget before generating any pasted response. Do not
use real mocks to tune the method.

### 7. Run three defensive mock-SBI rounds

Use an adaptive cumulative training schedule rather than spending 200 points automatically in
each round:

The confirmed accounting treats 600 as an **all-in successful expensive-paste cap**, not merely
a count of unique production theta values. Reserve 36 executions for non-reusable nside/profile/
HOD/cap diagnostics. If the final predeclared diagnostic matrix needs more than 36, reduce the
contingency training bank by the excess; never borrow from the sealed holdout. Track and publish
both unique contracted theta values and total paste executions, plus failed/retried jobs,
cheap map-noise remeasurements and node/GPU hours separately. Reusing a cached signal paste with
new noise maps does not consume another paste slot, but repainting any signal field does.

| stage | cumulative training theta | internal validation | sealed final holdout | diagnostic reserve | all-in maximum |
|---|---:|---:|---:|---:|---:|
| round 1 | 96 | 12 | 48 reserved, unopened | up to 36 | 192 |
| round 2 | 224 | 24 cumulative | 48 reserved, unopened | up to 36 | 332 |
| round 3, strong path | 320 | 36 cumulative | 48 reserved, unopened | up to 36 | 440 |
| round 3, normal path | 400 | 36 cumulative | 48 reserved, unopened | up to 36 | 520 |
| contingency cap | 480 | 36 cumulative | 48 reserved, unopened | 36 | 600 |

The 12 round-specific validation points are never trained on, but they may inform the next
round's frozen normalized mixture
and the choice between the pre-registered strong, normal and contingency branches. Draw the 48
final holdouts from a fixed defensive theory/prior mixture before round 1 and do not inspect them
until the proposal schedule, emulator, hyperparameters and stopping branch are frozen. If this
final audit fails, reject the result; do not tune on it. Any further simulations then need a new
budget decision.

Force `theta_ref` into the first 96 training points. Reuse its validated, cached map products for
every response difference, so it counts once rather than becoming an extra paste. Whenever a
diagnostic nside512 theta satisfies the final contract, reuse it in the training bank and release
its diagnostic-reserve slot. Different nside, cap, profile support or HOD seed is a distinct
execution even at the same theta and is not silently counted as a training point.

Use a defensive proposal in prior-probit coordinates:

```text
round 1: 0.60 * frozen theory posterior
       + 0.25 * broadened theory posterior
       + 0.15 * original prior

round 2: 0.60 * mock posterior round 1
       + 0.25 * frozen theory posterior
       + 0.15 * original prior

round 3: 0.60 * mock posterior round 2
       + 0.25 * frozen theory posterior
       + 0.15 * original prior
```

Represent every posterior-derived proposal by a simple normalized density fitted and validated
in five-dimensional probit space, initially a Gaussian mixture with its component count frozen
by the analytic oracle. Store its parameters, normalized `sample`/`log_prob` implementation,
environment and hash. Transform its density back to physical theta with the exact probit
Jacobian. Fit the broadened component once and freeze its inflation/temperature in the oracle.
The prior component protects against theory/mock shifts and missing modes.

Draw randomized bank points directly from the frozen round mixture without candidate ranking,
minimum-separation rejection, rounding or post-draw selection. Those operations change the
sampling density and invalidate the saved `log q`. An exact floating-point duplicate may reuse a
content-addressed paste, but the manifest retains both proposal-draw records and their
multiplicity. `theta_ref`, resolution/profile/HOD/cap controls and any deliberately chosen
anchor are marked `sampling_role=forced_or_diagnostic` and
`importance_eligible=false`; never fabricate a proposal density for them. Adaptation changes
only the next round's normalized mixture before its IID draws are made.

For GP/synthetic-likelihood or SNLE SBI, these distributions place simulations while the final
posterior is always proportional to `p0(theta) * p_mock(d_obs | theta)`. Their proposal density
does not enter the fitted conditional mean/likelihood, but defensive support and heldout tests
remain mandatory. For SNPE-C, pass the exact mixture proposal to the sequential correction.
Never set `prior = theory_posterior`.

Use one deterministic paired response per unique theta. The GP emulates this conditional mean,
and the known survey covariance enters analytically in the synthetic likelihood, so noisy
training replicas are unnecessary. Nevertheless generate and archive one independently seeded
map-noise set and its noisy maps for every standard-contract execution so the same bank can be
used for later map statistics; this archived noisy realization does not replace the noiseless
response in GP training. Only an oracle-approved neural-likelihood fallback may use
cheap stochastic replicas; if used, every replica must be made by adding new harmonic
field-noise maps to the cached signal maps and rerunning the exact estimator. Direct
`L @ epsilon` augmentation of a measured Cl vector is forbidden for this experiment. Split
strictly by unique expensive theta so noise variants of one signal paste cannot cross training,
validation or test partitions. These replicas capture instrument/reconstruction noise
conditional on the cached signal phase; they do not empirically calibrate the full cosmic/sample
variance, which remains in the frozen analytic covariance.

Save every successful expensive execution permanently for this project. Each standard-contract
bundle contains the full-sky, unmasked native-dtype signal maps for g, y, CMB-kappa and tau; the
y/kappa/tau harmonic-noise or map components; the reconstructed noisy maps used by the estimator;
coupled and decoupled bandpowers; and all hashes. Content-address a bitwise-identical fixed
galaxy map, common mask or noise realization and reference it rather than writing duplicate
arrays. Diagnostic variants are also retained but carry their different nside/profile/HOD/cap
contract and are excluded from the standard inference bank. No lossy quantization or deletion is
allowed. Before production, use the one-theta output size to forecast the strong/normal/600-cap
storage and require adequate quota plus recovery margin.

**Adaptive stop and Mock-SBI gates:** use the internal validation set before opening final
holdouts.

- Take the 320-point strong round-3 path only if the round-2 oracle and real internal validation
  both have `p95 |Delta logL| <= 0.05`, `max |Delta logL| <= 0.25`, posterior mean drift at most
  0.05 sigma, width change at most 5%, and correlation change at most 0.05.
- Otherwise use the normal 400-point round-3 path.
- Stop at the normal path only if `p95 |Delta logL| <= 0.10`, `max |Delta logL| <= 0.50`,
  posterior mean drift is at most 0.10 sigma, width change is at most 10%, correlation change is
  at most 0.10, and three GP training seeds agree.
- Use the contingency mixture draw to 480 training points only when the emulator is broadly
  correct but one normal gate is borderline. A gross round-1/2 failure triggers a method or
  compression review, not automatic spending.
- After selecting the branch, open the 48 final holdouts and require the normal likelihood-error
  gates with no unexplained probe-specific or boundary failure. The cap result is rejected if
  this final audit fails.

Also require full-vector posterior predictives to cover the observation and show the posterior
change between the last two cumulative training banks. Separately for theory HMC, theory SBI and
mock SBI, report absolute joint and per-probe best-fit chi-square/PTE, retained rank, number of
varied parameters, `rank - n_varied`, and full-vector posterior-predictive residuals. Apply the
shared `[0.01,0.99]` joint and per-probe goodness rule from Step 5. A stable emulator posterior
with a failed absolute-fit gate does not pass.

**Proposal nulls:** run a cheap theory toy with correct versus deliberately missing proposal
correction; only the corrected run may recover the original-prior posterior. Run a shifted-
observation stress test whose posterior lies partly outside the narrow theory proposal; the
defensive mixture must recover it.

### 8. Preserve a reusable map bank and exact prior/proposal provenance

Use a plain, append-safe layout rather than a database or custom framework:

```text
outputs/.../three_probe_map_bank/
  experiment_contract.yaml
  common/mask_nside512.fits
  common/noise_curves_and_transfers.h5
  common/namaster_workspace.fits
  common/original_prior.json
  proposals/round_00.json + round_00_arrays.npz
  proposals/round_01.json + round_01_arrays.npz
  proposals/round_02.json + round_02_arrays.npz
  maps/<execution_id>.h5
  records/<execution_id>.json
  manifest.jsonl
```

Workers write one temporary HDF5 bundle and one temporary JSON record, validate and hash them,
then rename each atomically. A single serial collector validates all records and writes the
canonical manifest; workers never concurrently append to one JSONL file. Store the common mask,
workspace and truly identical maps once by content hash, but every execution record resolves all
references and can be reconstructed without consulting a notebook.

Each map bundle/record contains or references:

- execution ID, standard/diagnostic contract hash, git commit and dirty-tree digest;
- physical theta and probit theta in the canonical five-parameter order; every fixed parameter;
- full-sky unmasked signal maps `g`, `y`, `kappa`, `tau` in native dtype and RING ordering;
- y/kappa/tau noise alms or maps and the noisy maps actually passed to NaMaster;
- common-mask/workspace/transfer hashes; HOD, field-noise and proposal RNG seeds;
- coupled Cls, decoupled 36-vector, response vector and covariance/observation hashes;
- role: `iid_training`, `iid_internal_validation`, `iid_sealed_holdout`, `forced_reference`,
  `diagnostic`, or `retry`; retry ancestry and whether it consumed a successful-paste slot;
- original-prior and proposal fields described below; and
- dataset shapes, dtypes, units, byte counts and SHA256 hashes.

Do not downcast or quantize maps for storage. Use lossless HDF5 compression and shuffle only.
The preflight one-theta benchmark reports compressed and uncompressed bytes per execution and
projects storage for the 440-, 520- and 600-execution branches. Production does not start unless
the target location has at least the projected cap usage plus 50% recovery/checkpoint margin.
Every expensive execution that completed successfully remains stored even if its scientific
gate later fails. If the adaptive inference stops before 600, retain every map actually made;
do not generate unnecessary pastes merely to fill the archive.

#### Prior and proposal record

The canonical `original_prior.json` stores parameter names/order, bounds, fiducials, units and
the normalized five-dimensional density

```text
log p0(theta) = -sum_j log(high_j - low_j)  inside the box,
                -infinity                  outside the box.
```

It also stores the exact componentwise prior-CDF/probit transform and forward/inverse Jacobians.
Every posterior-derived proposal is exported as a portable normalized Gaussian mixture in
probit space with JSON/NPZ parameters, not only as a Python pickle or a cloud of samples. Its
wrapper exposes deterministic `sample(seed)` and normalized `log_prob(theta)` in physical
coordinates. Validate normalization and posterior approximation in the cheap oracle before the
proposal is allowed to generate expensive maps.

For every randomized map record, save:

```text
round_id
proposal_hash
mixture_component_weights
log_p0(theta)
log_q_generating(theta)
log_q_component_k(theta) for every component
proposal_seed and draw_index
importance_eligible = true
```

After the final round, evaluate every realized round proposal at every randomized bank theta and
add those values to the collected manifest. For a prior-predictive calculation using the pooled
randomized standard-contract bank, use the deterministic-mixture density

```text
q_bank(theta) = sum_r (n_r / N_random) q_r(theta)
w(theta)      = p0(theta) / q_bank(theta) .
```

Every round proposal contains the original prior with weight 0.15, so
`q_bank(theta) >= 0.15*p0(theta)` over the box and the raw importance ratio is bounded above by
`1/0.15`. Verify this algebraically and numerically. Forced, diagnostic and changed-contract
maps have `importance_eligible=false` and are excluded from this calculation.

For a future higher-order-statistic likelihood or SNLE, the proposal density does not multiply
the final posterior: train only on compatible standard-contract maps and evaluate
`p0(theta) * p(summary_obs | theta)`. For a future direct SNPE-C posterior, append each round
with its exact saved proposal object so the sequential correction targets `p0`. Never pool the
adaptive maps and label them original-prior simulations. Make a new map-ID-level train/validation/
test split for each new statistic so one signal map or its noise variants cannot cross splits.
Before quoting a higher-order-statistic posterior, rerun a support/coverage test: a bank designed
around the Cl posterior may still miss a mode favored by a different statistic despite the 15%
prior defense.

The bank remains conditional on the c0000/ph000 phase, the resolved-halo prescription and the
frozen HOD policy. Correct prior weights do not turn it into a phase ensemble or add unresolved
matter. Every later statistic and figure must retain those labels.

**Map-bank gate:** every successful expensive execution has exactly one complete record and a
resolvable map bundle; all array hashes and content-addressed references pass; remeasuring the
saved noisy maps reproduces the stored coupled and decoupled Cls; all standard bundles share the
contract/mask/workspace hashes; proposal JSON/NPZ round-trips preserve `sample` and `log_prob`;
every IID record has finite consistent `log_p0`/`log_q`, every non-IID record is ineligible, the
15% prior-support bound holds, the ledger count is at most 600, and projected/actual storage is
reported. A missing map or proposal record blocks final SBI and later reuse.

### 9. Make the final GetDist comparison

Plot only final accepted products, all conditioned on the same `d_obs`:

1. direct resolved-theory HMC;
2. sequential resolved-theory SBI;
3. fixed-phase conditional resolved pasted-response SBI.

Use the same parameter order, labels, prior ranges, weights and truth markers. The plotter must
assert exact contract hashes before reading samples. Write a manifest containing all source
sample hashes, observation/covariance/catalog hashes, convergence/calibration verdicts, and the
GetDist settings. The manifest must also contain the absolute joint/per-probe goodness-of-fit
numbers and gate verdict for each contour; all three must pass before the figure is called a
constraint comparison.

The expected result is HMC/theory-SBI agreement within Gate 5. If the paste validation and
response-emulator gates pass, the mock-SBI contour should be statistically compatible but can
differ in orientation or width because it uses the pasted response rather than analytic theory
derivatives. All three can move away from the truth because they share one noisy observation.
Any coherent mock-only shift must be localized in the 36-vector before being interpreted as a
response difference. It cannot be interpreted as an absolute theory/paste offset at the anchor.

Place `FIXED-PHASE CONDITIONAL RESOLVED RESPONSE -- COMMON fsky2=0.4 MASK -- PARTICLE-COUNT
MASS PROXY TREATED AS M200c -- PROVISIONAL TAU NOISE` on the figure and in its manifest.
If the nside resolution gate is not passed, the figure is diagnostic regardless of inference
calibration.

If any upstream gate fails, the figure may still be saved with a large `DIAGNOSTIC -- GATE
FAILED` label and a machine-readable failure reason. It must not be described as a constraint.

### 10. Execution order and cost control

```text
P0 audit the c0000 particle-mass/header consistency, contributing-shell coverage and noise/beam basis
  -> P1 extend abacus_lightcone_catalog.py and build the selected c0000/ph000 proxy-mass catalog
  -> P2 low-resolution/noise-free fiducial paste
  -> P3 resolution and parameter-anchor paste validation
  -> P4 frozen covariance and held-out observation
  -> P5 converged common-observation HMC + calibrated theory SBI
  -> P6 adaptive analytic oracle (strong, normal, contingency budgets)
  -> P7 one-theta timing/memory/storage/map-roundtrip benchmark
  -> P8 mock rounds 1, 2, 3 with atomic persistent map bundles
  -> P9 collect and validate prior/proposal/map-bank manifest
  -> P10 independent refutation and three-way plot
```

Do not submit cluster work until a one-theta benchmark reports compile time, steady-state time,
peak memory and output size. Then state node/GPU count, wall time and expected evidence and ask
for approval. Use the smallest sky cap and `nside` that can test bookkeeping first, then a larger
cap/resolution before production. Never submit the full contingency budget before the earlier
rounds pass. By default the 600 cap counts all successful expensive paste executions, including
validation, final holdouts and nside/profile/HOD/cap diagnostics. The cost request must show the
exact predeclared execution matrix, reusable points, job/chunk count, retries and total node/GPU
hours and projected/available map storage; it must not report only unique theta values. Every
successfully completed paste is retained even if a later gate rejects it.

At every phase, create an evidence ledger with exact commands, environment, git/dirty-tree
state, catalog/config/source hashes, seed, and real output. Stored notebook output is not
evidence.

## Failure modes

- **Pure theory-posterior “prior”.** Produces a narrow answer with no support for a real
  theory/mock shift. The result targets the wrong distribution.
- **Different observations.** HMC/theory SBI and mock SBI contours move for ordinary noise, so
  an overlay looks discrepant even when all methods are correct.
- **Minimum present called complete.** The lowest valid catalog row is an order statistic, not a
  completeness proof. Use the frozen complete-particle threshold across the full redshift
  interval and pass the 125/150-particle posterior-stability test.
- **Proxy mass advertised as true SO mass.** The user-approved working identification is
  provisional. Hiding it can turn a mass-definition amplitude/shape error into an apparently
  physical shift of the five baryonic parameters.
- **Native `SO_radius` called `R200c`.** Its evolving mean-density definition and snapshot epoch
  do not match a 200-critical mass at the lightcone crossing.
- **One-sided HOD mask mistaken for a halo cut.** Galaxy occupation is truncated while y, tau,
  kappa and two-halo integrals are not.
- **Galaxy-only redshift filter.** The tracer map is selected but the pasted non-galaxy fields
  are not; “same slice” is false.
- **Full versus resolved theory mixed.** The map pipeline can pass its null while the contour
  comparison targets a different model.
- **Cl-center versus band-average comparison.** Creates a smooth ell-dependent residual that
  parameters can fit.
- **Center cut used as a window cut.** `ell_max=1536` passed to the current selector retains a
  thirteenth band extending to ell 1596; the map, noise and covariance then average different
  multipoles.
- **Formal `3*nside-1` treated as guaranteed accuracy.** The retained band above about
  `2*nside` is provisional until the fixed-beam nside1024 comparison passes.
- **Common fsky implemented only as a covariance scalar.** Maps are effectively full-sky while
  errors are divided by 0.4. Use the same saved mask for every field, exact NaMaster windows and
  exact mask covariance.
- **Different field masks with the same scalar fsky.** Equal areas do not imply equal mode
  coupling or overlap. Require one bitwise-identical mask hash for g, y, kappa and tau.
- **Coupled pseudo-Cl treated as the data vector.** The mask suppresses/redistributes power and
  the inferred parameters absorb it. Save coupled spectra for diagnostics but infer only from
  `workspace.decouple_cell` output.
- **NaMaster band-major covariance read as probe-major.** Shapes and positive definiteness can
  survive while covariance blocks are assigned to the wrong bands. Reshape band-major first.
- **Beam or pixel window applied twice.** Produces a monotonic high-ell deficit.
- **CMB kappa with LSST shape noise.** Gives plausible but physically unrelated gkappa errors.
- **Provisional tau noise called a forecast.** The apparent information gain from gtau has no
  defensible survey meaning.
- **Fixed-phase residual inserted into both mean and observation.** Treating the absolute
  reference paste as a deterministic model correction while also using an ensemble covariance
  can remove or duplicate its fluctuation. Keep only the paired response in the model mean and
  let the common map-measured observation carry the reference realization once.
- **Conditional response called an ensemble response.** Pairing one phase leaves
  `delta(theta,phase)-delta(theta_ref,phase)`; caps and HOD seeds do not establish a phase
  ensemble.
- **HOD/shot noise realized twice.** A sampled galaxy catalog already carries its Poisson
  realization. Adding a separate Gaussian galaxy map-noise field creates a second realization;
  keep the one `N_ell^gg` term in the likelihood covariance without adding that second map.
- **Bandpower covariance used as a map spectrum.** A 36-dimensional draw has no unique map
  realization and omits the required harmonic/beam structure. Generate maps from field-level
  unbinned `N_ell` curves.
- **Map-level plus Cl-level noise.** Adding a covariance-vector draw after measuring noisy maps
  artificially broadens mock SBI.
- **Noise replicas leaked across splits.** Validation looks excellent because the same expensive
  theta/map appears in training.
- **Adaptive bank called a prior sample.** Sequential maps are drawn from `q_r`, not `p0`.
  Prior-predictive estimates require saved normalized proposals and multiple-importance weights;
  SNPE requires the proposal object; likelihood-based inference applies `p0` only at inference.
- **Minimum-separation filtering with uncorrected log q.** Rejecting candidate draws changes the
  proposal distribution. Draw reusable bank points directly from the normalized mixture and
  flag forced points as non-IID.
- **Cl-only archive.** Discarding maps prevents higher-order-statistic reuse and hides future
  estimator changes. Every successful expensive paste must have a validated persistent bundle.
- **Unsafe cache reuse.** Different parameter points receive one old map.
- **Chunk loop repaints the full catalog.** Low-mass maps have amplitudes proportional to the
  number of chunks.
- **Adaptive budget stopped because the triangle looks stable.** Fewer than 600 executions are
  allowed only through the pre-registered oracle, internal-validation, posterior-stability and
  sealed-final-holdout gates.
- **HMC convergence inferred from R-hat alone.** Tree-depth saturation or low tail ESS makes
  contours too narrow.
- **Threshold changed after failure.** Converts evidence of a problem into a false pass.

## Remaining prerequisites

1. **Mass/header audit — resolve before catalog production.** Read every contributing c0000
   source header, record `ParticleMassHMsun`, verify one cosmology/schema, and apply the
   user-selected operational floor `M_particle_proxy_hMsun >= 5e11 Msun/h` for
   `0.3<z<0.5`. Record its exact particle-count equivalent without rounding and keep the mass
   semantics provisional. No additional mass-definition choice is needed.
2. **Noise and observable-basis audit — resolve before maps.** Expose the unbinned y, CMB-kappa
   and tau `N_ell` curves and freeze whether every signal/noise map and estimator output is beam
   convolved or deconvolved. Confirm curve support over all coupled modes. This is required to
   turn the covariance assumptions into physical Gaussian map noise without double smoothing.
3. **Common-cap implementation audit — resolve before maps.** Construct the fixed 1-degree-C2
   axisymmetric cap with final `mean(mask**2)=0.4`, verify c0000 coverage, and freeze its mask,
   workspace and transfer hashes. Confirm every scalar survey default is 0.4. This user choice
   is resolved; the audit tests its implementation rather than choosing a footprint.
4. **Storage/quota audit — resolve before production.** The one-theta benchmark must measure
   losslessly compressed sizes for signal, noise and noisy map datasets, project the 440/520/600
   branches, and verify available quota with 50% recovery margin. No completed expensive map may
   be deleted to rescue an underestimated budget without a new user decision.
5. **Gate sign-off and run authorization.** The numerical thresholds in Gates 2--5 and the
   adaptive stop rules are pre-registered planning choices. Confirm them before execution; no
   catalog build, paste, sampler or cluster job starts until the user explicitly authorizes it.

All scientific choices from the discussion are now resolved: c0000/ph000, strict 0.3--0.5,
particle-count proxy treated provisionally as `M200c`, complete-particle mass floor, provisional
tau noise, paired fixed-phase conditional response, common map-noisy observation, one common
`fsky2=0.4` mask and exact NaMaster estimator/covariance, `nside=512`, 12 complete bands/36
elements, persistent maps and normalized prior/proposal provenance, and an adaptive all-in
budget with 440/520/600 maximum expensive paste executions on the strong/normal/contingency
paths.
