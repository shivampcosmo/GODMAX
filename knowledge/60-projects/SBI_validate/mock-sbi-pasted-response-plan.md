---
id: kb.sbi.mock-sbi-pasted-response-plan
title: Plan for mock SBI on pasted Abacus maps with a 200-300 paste budget
layer: 60-projects
owner: xdesi-lead
status: draft
confidence: medium
scope:
  - notebooks/SBI_validate/three_probe_mock_experiment.yaml
  - notebooks/SBI_validate/three_probe_fast_paste.py
  - notebooks/SBI_validate/three_probe_noise_contract.py
  - notebooks/SBI_validate/three_probe_noiseless_estimator.py
  - notebooks/SBI_validate/three_probe_inference_contract.py
  - notebooks/SBI_validate/three_probe_jax_forward_model.py
  - notebooks/SBI_validate/build_three_probe_inference_manifest.py
  - notebooks/SBI_validate/run_simulator_native_active_sbi.py
  - notebooks/xDESI/paste_abacus_maps.py
  - notebooks/xDESI/combine_abacus_map_splits.py
  - notebooks/xDESI/abacus_pasting_helpers.py
  - src/get_sim_maps.py
invariants:
  - INV-NMT-COUPLED-01
  - INV-NMT-BANDMAJOR-01
  - INV-WINDOW-CMP-01
  - INV-SHOTNOISE-01
  - INV-PRODUCT-PROV-01
  - INV-JAX-SEED-01
  - INV-JAX-X64-01
  - INV-WHITEN-RANK-01
  - INV-CHI2-HONEST-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "STAGE0 oracle budget test (no GPU): notebooks/SBI_validate/oracle_paste_budget_test.py"
  - "STAGE1 null: paste one split with the frozen theta and require bitwise equality with the archived split"
  - "STAGE1 null: paste one split with get_galmap=false and require bitwise-equal y/tau/kappa"
verified_at_commit: UNSTAMPED
verified_on: 2026-08-23
see_also:
  - kb.sbi.selfconsistent-observation-decides-agreement
  - kb.sbi.three-probe-noisy-mock-covariance
  - kb.sbi.simulator-native-efficient
  - kb.sbi.three-way-mock-comparison-plan
supersedes: []
---

# Mock SBI on pasted maps with a 200-300 paste budget

## 0. What this is for

The theory-side campaign is finished: theory HMC and theory SBI agree on a
self-consistent observation (`kb.sbi.selfconsistent-observation-decides-agreement`).
That comparison deliberately removed model misspecification by making the observation
the analytic model's own prediction. It therefore says nothing about whether the
analytic model recovers the parameters of a **pasted** universe.

This plan answers that question. The simulator is the Abacus Backlight paste pipeline,
the observation is the pasted mock at the fiducial point plus its frozen noise
realization, and the covariance is the one the theory runs already use. The expensive
resource is the paste: the budget is ~200-300 pasted parameter points, and almost every
design decision below exists to spend that budget well.

The deliverable is a three-way posterior comparison — theory HMC, theory SBI, mock SBI —
on one common observation, with an absolute goodness-of-fit statement for each.

## 1. Audit findings

All numbers in this section were produced this session on `workergpu159` against the
frozen contract. They are the evidence the plan is built on; anything marked ASSUMED is
a hypothesis with a gate attached.

### 1.1 The observation and covariance already exist and are exactly what is wanted

The **production** contract's `data_vector` is already the pasted mock: Abacus pasted
maps at the fiducial gas parameters plus noise realization index 0, measured through the
saved NaMaster workspace on 14 complete bands per probe, `gy`/`gkappa`/`gtau`,
`ell` in [80, 2010), nside 1024, lmax 2048.

- observation: `data/SBI_validate/three_probe_inference/inference_contract.yaml`
- covariance / cholesky / window / pixel_window_g / profile_smoothing_bell: the same
  arrays, hash-pinned, that the self-consistent contract reuses byte-for-byte.

So "use the mock Cls at the fiducial point as the data vector, with the same covariance
as theory" requires **no new observation and no new covariance**. It requires the
production contract, which is where this campaign started before the self-consistent
detour. This is worth stating plainly because it removes a whole class of potential
inconsistency: mock SBI and theory HMC will read the same file.

### 1.2 The y noise is instrument + missing-sky yy, and it is already correct

`three_probe_noise_contract.py:265`:

```python
y_noise = y_inst + missing_yy      # missing_yy = all_yy - slice_cls["yy"]
```

- `y_inst` = SO tSZ deproj2 noise curve, dense to lmax 2048, zeroed below the first
  tabulated multipole and below ell 80.
- `missing_yy` = numerical Tinker all-sky yy over z in [0.01, 3], log10 M in [10, 16]
  (nM 64, nz 96, nk 128), smoothed by the **same** half-pixel flat-sky `Bell**2` as the
  painted maps, minus the yy actually contained in the resolved slice.

This is the "yy from the uncorrelated structure" that must be added to the y noise. It
is the variance from pressure outside the pasted slice (z outside 0.3-0.5, M below
5e11), which contributes to the yy leg of the `gy` covariance but not to the `gy`
signal, because `unresolved_completion: false`. It enters only through
`total["yy"] += noise["y"]` and then `nmt.gaussian_covariance(..., coupled=False)`.

Two consequences the plan must respect:

1. **Do not rebuild the covariance.** It is already built this way and hash-pinned.
   Rebuilding invites a silent divergence from the theory runs.
2. `missing_yy` is formally θ-dependent (it scales with the pressure profile), so a
   fully self-consistent mock simulator would have a θ-dependent covariance. Theory HMC
   holds it fixed; mock SBI must hold it fixed at the same value or the comparison is
   confounded. This is a documented approximation, not an oversight, and Stage 2 measures
   its size (§7.4).

### 1.3 Noise adds **exactly** linearly in bandpower space — verified

This is the most important result for the budget. Because the galaxy leg is frozen
(`freeze_galaxy_catalog: true`, and none of the five sampled parameters touches the HOD),
every measured vector decomposes as

```text
x(theta, seed) = mu_paste(theta) + nu(seed)
```

with `nu(seed)` **independent of theta**. The estimator is bilinear in the two alms and
`decouple_cell` is linear, and `g` is the same fixed alm in all three spectra, so

`decouple[ alm2cl(g, s(theta) + n(seed)) ] = decouple[ alm2cl(g, s(theta)) ] + decouple[ alm2cl(g, n(seed)) ]`

Verified against the archived realization 0 (stored total vs fixed signal + independently
recomputed noise-only cross):

| spectrum | max relative difference |
|---|---|
| `gy` | 8.41e-15 |
| `gkappa` | 7.56e-15 |
| `gtau` | 7.01e-15 |

Machine precision. Therefore:

- A **noise bank** of pre-measured `nu(seed)` vectors can be built once and reused for
  every pasted point, at zero marginal cost. Measured: **2.28 s per draw** (3 fields,
  synalm -> alm2map -> mask -> map2alm -> alm2cl -> decouple, OMP=8). A bank of 2000
  costs 76 min single-task, ~2.4 min across 32 tasks.
- This is **not** the forbidden `L @ epsilon` augmentation. Every `nu` comes from a
  field-level harmonic draw pushed through the exact estimator, which is precisely what
  the prior plan's prohibition demands. The linearity is a proven identity, not an
  approximation to it.
- Noise realizations are therefore free and unlimited. **The entire budget question is
  how many `mu_paste(theta)` evaluations we need**, and nothing else.

### 1.4 The 12-realization ensemble reproduces the analysis covariance

Needed because the noise draws vary only y/tau/kappa; the galaxy field is fixed, so
they do not resample galaxy shot noise or the g-side cosmic variance that the analytic
covariance includes. Measured:

| test | result | expected |
|---|---|---|
| per-band sd ratio (12 draws vs analytic), median | **0.987** | 1.0 +/- 21% per band at 11 dof |
| whitened chi2 about the fixed signal, mean of 12 | **44.33** | 42.0 |
| whitened chi2 about the ensemble mean, mean of 12 | **40.43** | 38.50 |
| chi2(ensemble mean - fixed signal) | **3.90** | 3.5 |

The per-probe medians are `gy` 1.016, `gkappa` 0.949, `gtau` 1.006; the full range
0.579-1.394 is what 11 dof produces.

This settles a conceptual worry that would otherwise block the whole design. In a
paired fixed-phase analysis the same realized g field appears in both the model mean and
the observation, so its fluctuation cancels in the residual; using the full covariance
would then double-count it. The measurement says there is nothing material to
double-count: for these cross-spectra the variance is dominated by the y/tau/kappa noise
legs beating against the fixed g field, and the noise-only scatter already accounts for
the full covariance to ~5%, which is the resolution of 12 draws. **Use the frozen
covariance unchanged.** Stage 2 replaces the 12-draw bound with an exact analytic
decomposition (§7.4).

### 1.5 Paste versus theory at the same parameters

The pasted signal and the analytic theory of the *same* resolved slice differ by far
more than the noise:

| | median fractional difference | worst |
|---|---|---|
| `gy` | **-15.3%** | -30.8% |
| `gkappa` | -7.9% | -22.6% |
| `gtau` | -7.7% | -22.6% |

`chi2(paste - theory) = 190.3` on a 42-vector. The paste is systematically *low*. This
is the inconsistency that makes mock SBI a different inference from theory SBI, and it
is the thing the pasted response has to supply.

### 1.6 The theory posterior on the pasted observation excludes the truth

This is the finding that most changes the plan, and it directly contradicts the naive
version of "use the theory-SBI posterior as the proposal".

| theory posterior | truth Mahalanobis^2 (5 dof, probit space) | truth percentile |
|---|---|---|
| **production** (obs = pasted mock) | **24.48** | **99.982** |
| self-consistent (obs = theory at fiducial) | 1.64 | 10.4 |

Marginals of the production posterior against the mock truth: `alpha_nt` +1.65 sigma,
`theta_co_0` +0.80, `theta_ej_0` +0.66, `nu_theta_ej_M` +0.62, `mu_beta` -0.19. The
misfit (chi2 ~161-165 for 37 dof) is absorbed by moving the parameters, and the joint
displacement is a ~4-sigma-equivalent exclusion of the point the maps were painted at.

Implication: a proposal drawn from the production theory posterior would place almost all
of a 200-300 point budget where the paste model does **not** fit. Restoring coverage by
inflation alone needs a factor 1.49 in sd just to put the truth at the 95% boundary, so
a safe inflation is ~2-2.5x — which throws away most of the benefit of using a posterior
at all. §5 solves this differently.

### 1.7 Paste cost, measured

Per split, nside 1024, 32 splits, 1x A100-80GB, from
`abacus_pasted_maps_..._split000of032.h5.timing.json` and the array logs:

| component | seconds | theta-dependent? |
|---|---|---|
| `total_time_s` | **472.7** | |
| jax module import | 9.1 | no |
| catalog read (`hdf5_read_time_s`) | 19.7 | no |
| `godmax_prepare_config` | 3.2 | no |
| `base_class` | 8.3 | no (cosmology only) |
| `Profiles` | 11.8 | **yes** |
| `setup_sim_map_main` | 7.2 | **yes** |
| chunk loop, 22 chunks | ~399 | **yes** |
| -- of which `galaxy_population` | ~12.0 / chunk | **no** (frozen HOD) |
| -- of which y + tau + kappa | ~5.5 / chunk | **yes** |
| -- of which cpu pixel-neighbour build | ~0.3 / chunk | no |
| hdf5 write | 5.2 | |

- **Current cost per parameter point: 32 x 472.7 s = 4.20 GPU-hours.**
- `galaxy_population` is ~66% of the chunk loop and is pure waste for a gas-parameter
  scan: the galaxy catalog is frozen, none of the five parameters enters the HOD, and the
  galaxy map already exists.
- With `get_galmap: false` the chunk loop falls to ~22 x 6.1 s; per split ~208 s;
  **~1.85 GPU-hours per point** (2.3x).
- The irreducible painting work for y+tau+kappa is 32 x 22 x ~6.1 s = **~1.19
  GPU-hours per point**. Everything else is amortizable overhead: 32 x (9.1 + 19.7 +
  3.2 + 8.3) = 0.36 GPU-h of it is theta-independent and can be amortized across many
  theta inside one process.
- Storage: **1.246 GB** per combined nside-1024 map; 125 MB x 32 transient splits.

`freeze_galaxy_catalog: true` combined with `get_galmap: true` forces
`num_splits == 32` (`abacus_pasting_helpers.py:2141`). Turning the galaxy map off lifts
that constraint, so the split count becomes a free tuning parameter.

### 1.8 The paste is deterministic in theta

Nothing in the y/tau/kappa path consumes random numbers. The only RNG is HOD galaxy
sampling, which is frozen and which we are switching off. Therefore `mu_paste(theta)` is
a **smooth deterministic function** with no simulation scatter — one paste per theta is
sufficient, and the emulator target is noiseless. (`random_seed: 20260819`,
`freeze_galaxy_catalog: true`.)

### 1.9 The blocking code gap

**There is no mechanism to paste at any theta other than the one in
`param_files/params_default.yaml`.** `prepare_fast_paste_godmax_config` deep-copies
`params["sim_params"]` and never applies a per-run override
(`three_probe_fast_paste.py:145-152`). And the default values *are* the mock truth:

```yaml
sim_params:
  theta_ej_0: 2.0        # truth
  nu_theta_ej_M: 0.0     # truth
  theta_co_0: 0.05       # truth
  mu_beta: 0.6           # truth
  alpha_nt: 0.18         # truth
```

So a naive campaign launch would silently paste 300 copies of the truth. This is the
first thing to fix, fail-closed (§6.1).

### 1.10 Prior art that is reusable, and prior art that is not

Reusable as-is:

- `three_probe_noise_contract.py` — mask, workspace, fixed g alm, noise curves,
  covariance, and the `realize` path that defines the noise bank.
- The saved NaMaster workspace, hash-checked on load. Reuse it; never rebuild it.
- `run_simulator_native_active_sbi.py` — an existing checkpointed active-design runner
  behind a replaceable `simulate(theta, seeds)` interface, with seed/role/row bookkeeping
  and resume. Its acquisition never touches an exact likelihood.
- `notebooks/xDESI/paste_abacus_maps.py` + `combine_abacus_map_splits.py` — the paste and
  combine drivers, unchanged apart from the theta override.

Not reusable unchanged:

- `run_simulator_native_active_sbi.py` hardcodes a **51**-component vector and
  `gy[0:17]/gkappa[17:34]/gtau[34:51]`. The contract is **42** with 14 bands per probe.
  This must be re-pinned, not coerced.
- `map_npe_utils.py` / `map_sbi_pasted_utils.py` are the earlier nside-512, two-parameter
  generation and target a different band definition. Read for intent; do not import.
- `three_probe_noiseless_estimator.py:make_scalar_namaster_measurement` is hard-locked to
  the **12** native diagnostic bands. The inference path is `inference_bins` in
  `three_probe_noise_contract.py` (14 bands). Do not cross them.
- `kb.sbi.three-way-mock-comparison-plan` is `status: deprecated` and written against a
  36/51-vector. Its failure-mode list is still the best in the repo and is carried
  forward in §8; its budget ladder is superseded by §6.

Prohibitions recorded there that this plan keeps:

1. No `L @ epsilon` augmentation of a measured Cl vector. (Satisfied by §1.3.)
2. No map-level *and* Cl-level noise on the same sample.
3. No second Gaussian galaxy noise map — the sampled catalog already carries its Poisson
   realization; `N_ell^gg` stays in the covariance only.
4. Split train/validation/test by **unique expensive theta**, never by noise replica.
5. Do not select design or validation points using an exact likelihood or HMC value.

## 2. What the simulator is, precisely

One mock evaluation at parameter point `theta`:

```text
1. paste     y, tau, kappa_cmb maps at theta, nside 1024, frozen catalog/HOD/mask
2. combine   32 splits -> one map file
3. measure   centre each map on the mask, map2alm(lmax=2048, iter=0),
             alm2cl against the FROZEN g alm, workspace.decouple_cell
             -> mu_paste(theta), a 42-vector on the 14 inference bands
4. augment   x = mu_paste(theta) + nu(seed) for any seed in the noise bank
```

Steps 3 and 4 cost ~1.1 s and ~0 s. Step 1 costs 1.2-4.2 GPU-hours. Step 2 is minutes.

The observation is `mu_paste(theta_fid) + nu(0)` — already built, already the production
contract's `data_vector`, with `nu(0)` the frozen realization-0 noise. **The observation
is a member of the same family the simulator generates**, which is what makes the
mock-SBI posterior interpretable and its coverage testable.

## 3. The inference target

```text
p(theta | d_obs)  proportional to  p0(theta) * N( d_obs ; mu_model(theta), C + C_emu(theta) )
```

- `p0` = the contract's uniform priors, in probit coordinates.
- `C` = the frozen 42x42 contract covariance, unchanged.
- `C_emu` = emulator predictive covariance, propagated, not ignored.
- The design/proposal density **never** appears in the posterior.

Two estimators are run and required to agree:

- **A (reference):** NUTS on the analytic Gaussian likelihood above, with `mu_model`
  from the emulator. Cheap, gradient-friendly, directly comparable to theory HMC.
- **B (the mock SBI the user asked for):** NPE trained on `(theta_i, mu_model(theta_i) +
  nu(seed_ij))` pairs, with the same 5-dim score compression the theory campaign used,
  and the proposal passed to the sequential correction. The free noise bank means B is
  trained on as many samples as it needs; its only expensive input is the same set of
  pastes.

A and B disagreeing is a diagnostic of the inference machinery, exactly as in the theory
campaign. A and B agreeing but disagreeing with theory HMC is the physics result.

## 4. How the paste budget is minimized

Four independent levers, in order of size.

### 4.1 Emulate the transfer, not the response (largest lever)

Do not emulate `mu_paste(theta)` directly. Emulate

```text
r_b(theta) = mu_paste,b(theta) / mu_theory,b(theta)          b = 1..42
```

where `mu_theory` is the existing differentiable JAX forward model, free to evaluate.
The model mean is then `mu_model(theta) = mu_theory(theta) * r(theta)`.

Why this is the right factorization:

- `mu_theory(theta)` already carries all the *fast* parameter dependence — the same
  profiles, the same Limber projection, the same window. Across the prior the 42-vector
  varies by orders of magnitude; the emulator would have to learn all of that.
- `r(theta)` is near-constant: measured at the fiducial point it is ~0.85 (`gy`), ~0.92
  (`gkappa`), ~0.92 (`gtau`) (§1.5). Its non-unity comes mostly from things that do not
  depend on the gas parameters at all: the Abacus halo mass function and bias versus
  Tinker, the provisional particle-count mass proxy, discreteness, and pixel/resolution
  effects. Only the 1-halo/2-halo transition and the profile-shape response are genuinely
  theta-dependent.
- A function that is flat to a few percent over the posterior needs far fewer design
  points than one that spans decades. This is the standard ratio-to-reference trick used
  by cosmological emulators, and it is what makes 200-300 plausible where a direct
  emulator would want thousands.

This is a **hypothesis with a gate**. Stage 2 measures how flat `r` actually is over the
region of interest and how many points its emulator needs; if `r` turns out to be as
structured as `mu`, the plan falls back to emulating `mu_paste` directly and the budget
must be renegotiated before any production pastes. Both variants are also compared as
emulator targets against the same held-out pastes, so the choice is made on evidence,
not on this argument.

Also fit and compare, on the same holdout, the paired difference
`Delta_b(theta) = mu_paste,b(theta) - mu_paste,b(theta_ref)` in units of
`sqrt(C_bb)` — the `mock_target: paired_fixed_phase_conditional_response` form the
experiment contract already declares. Whichever target wins the holdout is the one used.

### 4.2 Free, unlimited noise (§1.3)

One paste yields as many training samples as wanted. This decouples "how much training
data does NPE need" (a lot) from "how many pastes can we afford" (few). It is the reason
the user's instinct — many noise realizations per pasted map — is exactly right, and it
is exact rather than approximate.

### 4.3 Make each paste 2.3-3.5x cheaper

- `get_galmap: false` — removes 66% of the chunk loop. **1.85 GPU-h.** Requires the null
  test in §6.2.
- A persistent multi-theta worker: load JAX, read the catalog slice, build pixel
  neighbours, construct `base_class`, and JIT-compile **once**, then loop over theta
  re-running only `Profiles` + `setup_sim_map` + the paint. Amortizes ~0.36 GPU-h of
  theta-independent overhead per point, and lets the split count be chosen for wall-clock
  rather than for the frozen-galaxy constraint. Projected **~1.2-1.3 GPU-h per point**.
- Write only the three maps that changed. The galaxy map, kernels and mask are identical
  across theta — content-address them once and reference. Cuts per-point storage well
  below 1.246 GB.

### 4.4 Spend the points where the posterior is, without letting the theory posterior mislead the design (§5)

## 5. The proposal

Given §1.6, using the production theory posterior as the round-1 proposal would be a
design error: it puts the point the maps were painted at in its 0.018% tail.

The round-1 proposal is instead built from a **one-point transfer-corrected theory
model**, which is oracle-free in the only sense that matters and is far better centred:

```text
r_hat_b = mu_paste,b(theta_ref) / mu_theory,b(theta_ref)      # 42 numbers, from the ONE existing paste
mu_tilde(theta) = mu_theory(theta) * r_hat                    # zeroth-order paste emulator
q_0 = posterior of NUTS on N(d_obs; mu_tilde(theta), C)       # the round-1 guide
```

This uses only a simulation output and its own parameter label — which is what every
sequential SBI round does — and it costs nothing, because that paste already exists. It
removes the dominant part of the misfit (a per-band multiplicative offset) before the
design is chosen, so `q_0` should sit near where the paste model actually fits.

Round-1 proposal, in probit coordinates, as a normalized mixture with stored
`sample`/`log_prob`:

```text
round 1: 0.50 * q_0
       + 0.30 * q_0 broadened (covariance x 4, i.e. sd x 2)
       + 0.20 * prior
round r: 0.55 * mock posterior from round r-1
       + 0.25 * q_0 broadened
       + 0.20 * prior
```

The prior and broadened weights are deliberately heavier than the deprecated plan's
0.15/0.25, because §1.6 is direct evidence that a theory-derived guide can be badly
displaced on this problem. Cost of the insurance: ~20% of the budget. It buys the ability
to *detect* displacement rather than assume it away.

Rules:

- Draw bank points IID from the frozen normalized mixture. No candidate ranking, no
  minimum-separation rejection, no rounding, no post-draw selection — all of these change
  the density and invalidate the stored `log q`.
- `theta_ref` and any deliberate anchor or resolution diagnostic is flagged
  `sampling_role=forced_or_diagnostic`, `importance_eligible=false`. Never fabricate a
  proposal density for it.
- For NPE (estimator B) pass the exact mixture as the proposal to the sequential
  correction. **Never** set `prior = theory posterior`.
- For estimator A the proposal does not enter the posterior at all; it only places
  simulations.
- Adaptation changes only the *next* round's mixture, before its draws are made.

An honest caveat to record with the result: `q_0` is derived from a paste at the truth,
so the design is better centred than a blind design would be. That affects *efficiency*,
not bias — the posterior is proportional to `p0 * likelihood` and the emulator is
validated on a sealed holdout drawn from a fixed defensive mixture before round 1. It
does mean the coverage claim is conditional, and the shifted-observation stress test in
§7.7 is what makes it credible.

## 6. Stage-by-stage plan

### Stage 0 — Oracle budget test. No GPU pastes. Do this first.

Answer "how many pastes?" before spending any, by using the free theory model as a
stand-in for the expensive simulator, exactly as the prior plan's Gate 5 requires.

1. Draw 400 theta from the round-1 mixture of §5.
2. Evaluate `mu_theory(theta)` for all of them (free) and treat it as the "expensive"
   simulator output. Build a synthetic transfer `r_synth(theta)` with a controlled amount
   of theta-dependence, bracketing "flat" and "as structured as mu".
3. Fit the emulator (GP on the whitened 42-vector, and on the score-compressed 5-vector)
   with N = 24, 48, 96, 144, 200, 288 training points; hold out the rest.
4. Report, per N: p95 and max `|Delta log L|` over the holdout, posterior mean drift in
   sigma, width change, correlation change, and the stability of three emulator seeds.
5. Run estimator A and estimator B end to end at each N and check they agree with each
   other and with the exact-likelihood posterior of the same synthetic problem.

**Gate 0:** the smallest N meeting `p95 |Delta log L| <= 0.10`, `max <= 0.50`, mean drift
<= 0.10 sigma, width change <= 10%, correlation change <= 0.10, with three stable seeds,
is the pre-registered round-3 target. If no N <= 288 passes even in the *flat* transfer
case, the method is wrong and the architecture must change before any paste is submitted.

This stage also produces the two proposal nulls (§7.7) and costs no cluster time.

### Stage 1 — Make the simulator parameterizable, and prove it changed nothing else

1. **Theta override, fail-closed.** Add an explicit gas-parameter override to
   `prepare_fast_paste_godmax_config`, e.g. `config["pasting"]["gas_parameter_overrides"]`
   plus a `--theta-json` CLI path, which:
   - accepts **only** the five declared keys, rejecting any other key by name;
   - rejects a value outside the contract's prior bounds;
   - **requires** the override to be present when a campaign flag is set, so the
     `params_default.yaml` truth can never be pasted by omission (§1.9);
   - writes the override, its canonical-JSON sha256, and the resolved `sim_params` hash
     into the output HDF5 attrs and the run manifest;
   - leaves `validate_fast_paste_config` passing unchanged — grid, projector, LOS points,
     smoothing, catalog, mask and HOD are untouched.
2. **Null test A (reproduction).** Paste split 000 with the override set to the frozen
   values. Require **bitwise** equality of `map_ymap`, `map_tau`, `map_kappa_cmb` with the
   archived `split000of032`. Any difference means the override path perturbed something
   it should not have.
3. **Null test B (galaxy skip).** Paste split 000 with `get_galmap: false`. Require
   bitwise-equal y/tau/kappa. This is what licenses the 2.3x saving in §4.3.
4. **Null test C (split invariance).** With the galaxy map off, paste the same theta with
   a different `num_splits` and require the combined y/tau/kappa to agree to float
   round-off. Guards the "chunk loop repaints the full catalog" failure mode.
5. **One-theta benchmark.** Report compile time, steady-state time per split, peak GPU
   and host memory, output size, and end-to-end wall time for one complete point through
   paste -> combine -> measure. Required before any array submission.
6. **Measurement path.** A single function that takes a combined map file and returns the
   42-vector by reusing the *saved* workspace, the *saved* mask and the *saved* fixed g
   alm. Assert the workspace sha, the mask sha, and that the g alm is bitwise identical to
   the contract's. Never rebuild the workspace.
7. **Noise bank.** Build 2000 `nu(seed)` vectors on seeds disjoint from the observation
   seed and from the `mock_sbi_training` / `mock_sbi_holdout` namespaces already
   pre-registered in the manifest. Store with per-vector seed and sha.

**Gate 1:** nulls A, B and C pass bitwise / to round-off; the benchmark is recorded; the
measurement path reproduces the contract's `data_vector` from the archived map plus
`nu(0)` to machine precision.

That last check is the single most valuable one in the plan: it proves the whole chain
map -> measurement -> noise -> observation is the same chain that produced the theory
runs' observation.

### Stage 2 — Transfer-flatness scan. 12 pastes. The decision gate.

Paste 12 points spanning the round-1 mixture: `theta_ref`, plus a 2-level fractional
design over the five parameters chosen to bracket the posterior and reach into the
prior. Then:

1. Measure `r(theta)` for all 12. Report `max_theta |r_b(theta) - r_b(theta_ref)|` per
   band and per probe.
2. Compare the variation of `r` with the variation of `mu_theory` over the same 12
   points, in units of `sqrt(C_bb)`. The ratio of those two is the emulator-effort saving
   the factorization actually delivers.
3. Fit the three candidate targets (`r`, `Delta`, raw `mu_paste`) on 8 points, predict
   the other 4, and report `|Delta log L|`.
4. Recompute `missing_yy` at the extreme design points and report the induced change in
   the y-noise curve and in `diag(C)` — the size of the fixed-covariance approximation of
   §1.2.

**Gate 2:** `r` varies by materially less than `mu_theory` over the design, at least one
target reaches the Stage-0 accuracy trend at 8 points, and the covariance
approximation is small compared with the noise. If `r` is not flat, stop and renegotiate
the budget — do not proceed to 200+ pastes on a failed factorization.

Cost: 12 x ~1.3 GPU-h = **~16 GPU-hours**.

### Stage 3 — Round 1. 96 pastes.

Draw 96 IID from the round-1 mixture, including `theta_ref` (already pasted; it counts
once and is reused). Paste, combine, measure, archive. Fit the emulator on the winning
target. Run estimators A and B. Report the round-1 mock posterior, its absolute chi2 and
PTE, and the emulator's accuracy on the 12 round-specific validation points (never
trained on).

Cost: ~96 x 1.3 = **~125 GPU-hours**.

### Stage 4 — Round 2. 96 pastes.

Draw from the round-2 mixture centred on the round-1 mock posterior. Same products.
Report posterior drift between rounds 1 and 2 — the primary convergence statistic.

Cost: **~125 GPU-hours**.

### Stage 5 — Round 3, holdout audit, and the three-way comparison.

Top up to the Stage-0 gated target (expected 200-288 unique training theta). Then, and
only then, open the **sealed 24-point holdout** drawn before round 1 from a fixed
defensive mixture. Require the Stage-0 gate thresholds. If the holdout fails, the result
is rejected — it is not tuned on.

Final products:

- getdist triangle overlaying theory HMC, theory SBI, mock SBI on the common observation,
  with the truth marked.
- For each of the three, separately: absolute best-fit chi2, dof as `retained rank -
  n_varied` = `42 - 5 = 37`, PTE, and per-probe breakdown. Report the absolute number,
  never only an improvement (`INV-CHI2-HONEST-01`).
- Full-vector posterior predictive against the observation.
- Pulls of each posterior against the known truth, and the coverage statement with its
  §5 caveat.
- Estimator A versus estimator B agreement.

## 7. What to be careful about

### 7.1 Inherited conventions that must not be re-derived

These are already fixed by the contract. Mock SBI must inherit them identically, because
they cancel in the comparison only if both sides use the same choice.

- **Pixel window asymmetry.** `total_observed_cls` applies `pixwin` with exponent equal
  to the number of `g` legs — so `gy`/`gkappa`/`gtau` get `pixwin**1`, not `**2`. The
  painted y/tau/kappa maps carry the half-pixel-FWHM Gaussian `profile_smoothing_Bell`
  instead of the HEALPix pixel window; the galaxy count map carries the true pixel
  window. Do not "fix" this asymmetry.
- **Bandpower windows, not `ell_eff`.** Theory is compared through the saved 14x2049
  window (`INV-WINDOW-CMP-01`). A centre-versus-band-average slip produces a smooth
  ell-dependent tilt that the five parameters will partly absorb.
- **14 complete bands, ell in [80, 2010).** The 15-edge array is frozen. The partial
  final band is diagnostic-only. An `ell_max` passed to a centre-based selector silently
  retains a band that extends past it.
- **Decoupled, never coupled.** Infer only from `workspace.decouple_cell` output; save
  coupled spectra as diagnostics.
- **Band-major covariance.** `INV-NMT-BANDMAJOR-01`; the frozen matrix is already built
  block-by-block correctly, which is another reason not to rebuild it.
- **`coupled=False`** in `gaussian_covariance` (`INV-NMT-COUPLED-01`).
- **One bitwise-identical mask** for g, y, kappa, tau. Assert the sha.
- **`jax_enable_x64` before any array is created** (`INV-JAX-X64-01`).

### 7.2 The mass and slice caveats stay caveats

The catalog mass is an interpolated particle-count proxy treated as M200c
(`mass_semantics: interpolated_particle_count_proxy_treated_as_M200c`,
`mass_definition_status: provisional_assumption`). The tau noise is
`tau_noise_status: provisional` — an SNR-matched depth of 0.0233 arcmin chosen to match
`gkappa`, not a survey forecast. `unresolved_completion: false`: the pasted signal
contains only z in (0.3, 0.5) and M >= 5e11, and the mass below that floor which really
does correlate with the galaxies is omitted from *both* the mock signal and the theory
model. None of these invalidate the comparison — both sides share them — but none of them
may be quoted as a physical result. Any statement about `gtau` information content is
about a provisional noise level.

### 7.3 The traps specific to this design

- **Pasting the truth 300 times.** §1.9. Fail-closed override, and assert per-point that
  the recorded `sim_params` hash differs from the reference for every non-anchor point.
- **Unsafe cache reuse.** Two theta receiving one map. Content-address every paste by the
  canonical hash of its resolved parameters plus the config sha, and assert the mapping is
  injective before training.
- **Emulating the wrong thing well.** A very accurate emulator of `r` still gives a wrong
  posterior if `mu_theory` is evaluated with a different grid, aperture or window than the
  one behind `r_hat`. Pin the forward-model construction (grid, `dense_radius_nodes`,
  contract path) once and hash it into every emulator artifact.
- **Emulator uncertainty dropped.** `C_emu(theta)` must enter the likelihood. Omitting it
  makes the mock posterior too narrow in exactly the regions where the design is sparse —
  which is where the tails are.
- **Holdout leakage.** Split by unique theta. A noise replica of a training paste
  appearing in validation makes the emulator look perfect.
- **Design points chosen with the likelihood.** Acquisition may use only previously
  simulated discrepancies, never an exact likelihood or an HMC value. Recorded as a
  failure mode in `kb.sbi.simulator-native-efficient`.
- **Double-counted noise.** Either `nu(seed)` or nothing. Never `nu(seed)` plus a
  Cholesky draw; never a Gaussian galaxy noise map on top of the Poisson catalog.
- **Reporting the improvement.** The absolute chi2 against `42 - 5 = 37` is the result.

### 7.4 Validations to run beyond the stage gates

1. **Exact covariance decomposition.** Rebuild the analytic covariance twice — once with
   the full `total_cls`, once with the y/tau/kappa signal legs zeroed — and report the
   fraction of each band's variance carried by the noise legs. This replaces the 12-draw
   bound of §1.4 with an exact statement, and is the rigorous answer to "is the fixed g
   field a problem".
2. **Observation reconstruction.** Reproduce the contract `data_vector` from the archived
   map + `nu(0)` to machine precision (Gate 1).
3. **Noise-bank calibration.** With 2000 draws, the empirical covariance of `nu` should
   match the frozen `C` far better than 12 draws allowed. Report the whitened eigenvalue
   spectrum of `C^{-1/2} Cov(nu) C^{-1/2}` — it should be 1 to ~sqrt(2/2000) ~ 3%.
   A systematic departure means the noise model and the covariance disagree, which would
   invalidate both this campaign and the theory runs.
4. **Theta-dependence of `missing_yy`** (Stage 2, item 4).
5. **Resolution control.** One design point pasted at nside 512 and at nside 1024,
   compared through the same bands, to confirm the retained bands above ~2*nside are not
   resolution-limited. Flagged as a diagnostic execution, not a training point.
6. **Emulator seed stability.** Three emulator initializations at the final N.
7. **Posterior predictive.** Full 42-vector, all three methods.

### 7.5 Cluster hygiene

Nothing is submitted without stating node count, wall time and the evidence the job will
produce, and without the one-theta benchmark of Stage 1. `sbatch`/`srun`/`salloc` are
run by the user, never by an agent. Prefer the cheapest sufficient evidence: Stage 0 has
no GPU cost at all, Stage 2 is 12 points, and no production array is submitted before
Gate 2 passes.

### 7.6 Storage

At ~1.25 GB per combined map, 300 points is ~375 GB, on top of the 823 GB `data/`
already holds. Writing only the three changed maps and content-addressing the shared
galaxy map, kernels and mask should bring this well down. **Check the ceph quota before
Stage 3**, and record the projected total in the cost request. Every successful paste is
retained even if a later gate rejects the result.

### 7.7 Two nulls that make the result credible

- **Proposal-correction null.** A cheap theory-only toy run twice, once with the
  sequential proposal correction and once with it deliberately omitted. Only the
  corrected run may recover the original-prior posterior. Catches the single most common
  silent error in sequential NPE.
- **Shifted-observation stress test.** An observation constructed to sit partly outside
  the narrow round-1 guide. The defensive mixture must recover it. This is the direct
  test of the §1.6 risk and the honest answer to the §5 caveat.

## 8. Budget

Per-point cost 1.3 GPU-h assumes Gate 1 confirms the galaxy skip and the persistent
worker; the conservative fallback is 1.85 GPU-h, and the unoptimized cost is 4.20.

| stage | unique theta | GPU-hours at 1.3 | at 1.85 (fallback) |
|---|---:|---:|---:|
| Stage 0 oracle | 0 | 0 | 0 |
| Stage 1 nulls + benchmark | ~3 splits + 1 point | ~2 | ~3 |
| Stage 2 flatness scan | 12 | 16 | 22 |
| Stage 3 round 1 | 96 | 125 | 178 |
| Stage 4 round 2 | 96 | 125 | 178 |
| Stage 5 round 3 top-up | up to 96 | up to 125 | up to 178 |
| sealed holdout | 24 | 31 | 44 |
| diagnostics reserve (nside/resolution) | 8 | 10 | 15 |
| **total, all-in cap** | **~336 executions** | **~434** | **~618** |

Unique *training* theta lands at 200-288, inside the requested 200-300; the all-in
execution count is larger because holdout, validation and diagnostics are counted
honestly rather than hidden. At 8 concurrent A100s that is ~55 hours of wall clock at the
optimized rate; at 16, ~27 hours. Report unique theta and total executions separately,
plus failed/retried jobs, always.

## 9. Decisions needed before Stage 1

1. **Budget accounting.** Is the ~200-300 a cap on unique *training* theta (this plan's
   reading) or on *all* paste executions? If the latter, the ladder compresses to roughly
   64 / 64 / 64 with a 16-point holdout, and Stage 0 must gate on the smaller N.
2. **Storage.** Confirm ceph headroom for ~375 GB, or approve keeping full maps only for
   the anchors, holdout and diagnostics and 42-vectors plus hashes for the rest.
3. **Estimator priority.** Estimator A (analytic Gaussian likelihood on the emulator) is
   the reference and is cheap; estimator B (NPE) is the mock SBI as asked for. Both are
   planned. Confirm that A being the quoted reference is acceptable, with B as the
   agreement test.

## 10. Status

Nothing in §6 has been executed. §1 is measured; §4.1, §4.3 and the budget are
projections with gates attached. `verified_at_commit: UNSTAMPED` until Gate 1 passes.

---

# Execution record

Everything below was produced on 2026-08-23/24 on `workergpu159` (H100 PCIe) and
`workergpu040` (A100-SXM4-80GB). Commands are in
`notebooks/SBI_validate/mock_sbi_sbatch/README.md`; artefacts in
`data/SBI_validate/mock_sbi/`.

## Gate 1a — the measurement chain is the frozen one. PASS

`validate_mock_sbi_foundations.py` -> `foundations.json`

| check | measured |
|---|---|
| archived paste measured here vs the contract's `fixed_bandpowers` | **8.64e-15** relative, whitened chi2 4.3e-25 |
| `mu_paste(theta_ref) + nu(observation seeds)` vs the contract `data_vector` | **1.88e-15** relative, whitened chi2 **1.63e-26** |
| float64-throughout vs the frozen observation's mixed-precision mask | 3.19e-10 relative, chi2 2.5e-17 |

The second row is the one that matters: the simulator this campaign will run **is**
the process that produced the observation the theory runs already used.

Two things were found on the way and are now handled rather than tolerated:

* The contract stores its mask as **float32** while `build_contract` computed every
  bandpower, window and covariance with the float64 mask. Measuring with the stored
  copy costs 7.6e-11 relative on the 42-vector. `mock_sbi_common.canonical_mask`
  regenerates the float64 mask analytically and *requires* it to round bitwise to
  the stored array, so the recovery is a verified identity. The frozen observation
  itself mixes the two precisions (signal alms float64, noise float32), so
  reproducing it bitwise needs the same mixture; production uses float64 throughout
  and the difference is recorded above.
* The `(20260821, 201)` noise namespace contains a genuine 32-bit **seed collision**
  (value 2823752942 at flat positions 1592 and 3128, both `kappa`) among 3 x 2048
  draws. Two draws sharing a seed produce perfectly correlated deviates.
  `noise_bank_seeds` now over-generates and rejects, deterministically and
  prefix-stably.

## Noise linearity and the bank. Verified

`x(theta, seed) = mu_paste(theta) + nu(seed)` exactly, with `nu` independent of
`theta`: max relative difference **8.41e-15** (`gy`), 7.56e-15 (`gkappa`),
7.01e-15 (`gtau`) against the archived realization. One noise vector costs
**2.28 s**. The bank is therefore free and unlimited, and the entire budget question
reduces to the number of `mu_paste` evaluations.

## The transfer and the round-1 guide. PASS

`build_mock_sbi_transfer.py` -> `transfer_and_guide.json`

| | `gy` | `gkappa` | `gtau` |
|---|---|---|---|
| median `r_hat` | **0.8465** | 0.9213 | 0.9230 |
| range | 0.692-0.954 | 0.774-1.090 | 0.774-1.065 |

| quantity | value |
|---|---|
| chi2(obs, raw theory at the reference point) | **218.67** |
| chi2(obs, transfer-corrected theory at the reference point) | **38.42** |
| chi2 at the corrected MAP | **33.48** for 42 - 5 = 37 (PTE ~ 0.63) |
| generating point under the **raw** theory Laplace guide | Mahalanobis^2 30.82 -> **99.999th percentile** |
| generating point under the **transfer-corrected** guide | Mahalanobis^2 4.00 -> **45.07th percentile** |

So the section-5 design is validated: a 42-number correction measured from the one
paste that already exists takes the misfit from 218.67 to 38.42 and moves the
generating point from the extreme tail of the design guide to its middle. Drawing the
paste budget from the raw theory posterior would have spent it where the paste model
cannot fit.

## Null control on the theta-override patch. PASS

`check_three_probe_backend_parity.py` after the patch, A100 versus the H100
reference: `vectors_sha256` **identical**, max relative forward difference **0.0**,
chi2 at the MAP 161.094134 both sides. `sources_identical: false` — the source hash
moved, the numbers did not. This is the "show what did not change" evidence for
`resolve_gas_parameter_override`, which is additionally unit-tested against
required-but-missing, incomplete, unknown-key and out-of-prior inputs.

The generated paste configs differ from the frozen experiment YAML in exactly 9 of
153 keys: the five overrides, `require_gas_parameter_overrides`, `get_galmap`,
`nside`, `run_name`. 144 keys are identical.

## Stage 0 — the budget test. Full-response arm: FAIL, and that is the result

`oracle_paste_budget_test.py --target full --recipe concentrated`, 512 design points,
holdout gate on the posterior-relevant subset, 3 emulator seeds.

| N | p95 \|dchi2\| | max \|dchi2\| | median relative error |
|---:|---:|---:|---:|
| 24 | 40.9 | 156.6 | 5.2e-3 |
| 48 | 6.83 | 38.3 | 9.2e-4 |
| 96 | 3.98 | 26.8 | 5.9e-4 |
| 144 | 3.81 | 18.3 | 4.0e-4 |
| 200 | 2.38 | 8.37 | 3.5e-4 |
| 288 | 1.40 | 10.7 | 2.7e-4 |
| 384 | **1.11** | 7.28 | 1.8e-4 |

At N=384: posterior mean drift 1.46 sigma, width **+700%**, correlation change 1.37.
The gate is p95 <= 0.10, drift <= 0.10 sigma, width <= 10%.

Eight times the points buys six times the accuracy, so the pre-registered gate would
need thousands of pastes. **Emulating the full response is not viable at any
affordable budget.** Per the plan's Gate 0 this is not a reason to buy more pastes;
it means the transfer factorization is load-bearing rather than an optimization, and
Stage 2's flatness gate is the decisive experiment of the whole campaign.

The null control localises the width inflation: with `C_emu` dropped the width change
falls from +700% to **+48%**. The emulator's own uncertainty is doing it — which is
the honest behaviour of an emulator that does not know the response, not a bug.

Three defects were found and fixed while getting here, each worth recording:

1. **Linear-space emulation does not work.** Over this design the per-band raw
   response spans up to a factor 105 (median 3.8), and a stationary GP cannot fit
   both the guide core and the prior tail: at N=48, p95 \|dchi2\| ~ 8.2e3 with every
   length scale pinned at its bound. In log space the same target has per-band sd
   0.16 (max 0.71). Log space is also the physically natural choice, since these
   amplitudes respond multiplicatively to the gas parameters.
2. **The emulator covariance is rank-k, not diagonal.** Propagating only the diagonal
   of `Cov(log mu) = B^T diag(sigma^2) B` through the whitening is a different object,
   and it inflated the width by up to 5.4x on its own. It is now carried exactly via
   Woodbury at rank k.
3. **The likelihood-error gate must be evaluated where the posterior lives.** The
   mixture deliberately puts draws in the prior, which reach chi2 ~ 7.4e5 against a
   floor of 159.5; a p95 over all of them measures behaviour no chain ever visits.
   The gate is applied to points within `dchi2 <= 30` of the design minimum (the
   99.99% point of a chi2_5 is 25.7) and the all-holdout numbers are reported beside
   it. The *threshold* is unchanged.

Also measured: the analytic forward model costs **21 ms** per evaluation (6 ms
batched at 16), which is what makes emcee on the exact model affordable and the
Stage-0 test free.

## Gate 1b — paste nulls. PASS, both bitwise

`validate_mock_sbi_paste_nulls.py`, split 000 of 032, nside 1024, against the archived
split.

| null | `map_ymap` | `map_tau` | `map_kappa_cmb` | `n_galaxies` | split wall | GPU-h per theta |
|---|---|---|---|---|---|---|
| A: override set to the frozen values, galaxy map on | bitwise | bitwise | bitwise | 1,997,508 | **508.6 s** | **4.52** |
| B: `get_galmap: false` | bitwise | bitwise | bitwise | **0** | **196.6 s** | **1.75** |

`max|abs difference| = 0.000e+00` in every case. Null A proves the theta-override path
is inert when set to the frozen point, through the real paste pipeline rather than by
inspection. Null B proves the galaxy skip changes nothing: the galaxy map is genuinely
not painted (`n_galaxies` 0) and the three theta-dependent maps are bit-for-bit
identical.

**The galaxy skip is therefore licensed, at a measured 2.6x** (508.6 -> 196.6 s), close
to the projected 2.3x. Per-split output also falls from 125.4 MB to 90.6 MB. Use
**1.75 GPU-h per theta** in any cost request, not the plan's projected 1.3: the
persistent multi-theta worker of section 4.3 is not yet built, and 34 s of the remaining
196.6 s per split is theta-independent setup that it would amortize.

## The noise bank does NOT reproduce the frozen covariance. 92.7%, and it matters

`build_mock_sbi_noise_bank.py --count 2048` -> `noise_bank_training.report.json`

| quantity | measured | expected |
|---|---|---|
| mean whitened chi2 / dim | **0.9266 +/- 0.0045** | 1.0 |
| whitened eigenvalues below the Marchenko-Pastur floor (0.734 at q=42/2048) | **8 of 42** | 0 |
| lowest whitened eigenvalue | 0.474 | >= 0.734 |
| largest | 1.305 | <= 1.307 |

This **revokes the section 1.4 claim** that noise draws reproduce the covariance to ~5%.
That claim rested on 12 realizations giving 1.055 +/- 0.063, which cannot distinguish
1.0 from 0.927. With 2048 draws the deficit is 16 sigma and structured.

The cause is identified, not merely suspected. The per-band whitened standard deviation
tracks the **signal fraction of the second leg's auto power** band by band:

| band | 0 | 1 | 2 | 3 | ... | 13 |
|---|---|---|---|---|---|---|
| `gkappa` whitened sd | **0.760** | 0.773 | 0.802 | 0.834 | | 1.033 |
| kappa-kappa signal fraction | **0.216** | 0.167 | 0.124 | 0.085 | | 0.001 |
| `gtau` whitened sd | 0.893 | 0.882 | 0.896 | 0.848 | | 1.070 |
| tau-tau signal fraction | 0.107 | 0.086 | 0.069 | 0.054 | | 0.002 |
| `gy` whitened sd | 0.940 | 0.895 | 0.929 | 0.902 | | 1.006 |
| y-y signal fraction | 0.006 | 0.007 | 0.008 | 0.010 | | 0.038 |

The frozen covariance uses `total["kappakappa"] = signal + noise`, and at band 0 the
CMB-lensing *signal* is 21.6% of the total kappa auto power. The noise bank draws only
`noise_cls/kappa`, so it is missing the signal contribution to the `gkappa` variance --
largest exactly where the signal fraction is largest, and absent by band 13 where it is
0.001. `gy` is barely affected because the effective y noise (instrument plus
missing-sky yy) dominates its own auto power everywhere.

**This is the fixed-phase versus ensemble distinction, and it is not a bug in either
object.** The frozen `C` is the correct covariance for an analysis that averages over
signal realizations. The frozen observation is a *single* realization with a fixed
signal field -- the experiment contract calls its target
`paired_fixed_phase_conditional_response` -- and conditional on that fixed field the
correct covariance is the noise-only one, i.e. ~0.927 `C`.

Consequences, stated plainly:

* The mock simulator `x = mu_paste(theta) + nu(seed)` has covariance ~0.927 `C`, not
  `C`. Using `C` in the likelihood is **conservative**: posterior widths come out about
  3.6% too large, more in the lowest `gkappa` bands.
* The same inconsistency applies to the **existing theory HMC and theory SBI results**,
  which used `C` against the same fixed-phase observation. It is small, but it is not
  zero and it was not previously quantified.
* Per the instruction that mock SBI use the same covariance as theory, the campaign
  keeps `C`. That preserves the three-way comparison, since all three methods then carry
  the identical 7.3% conservatism. Switching mock SBI to the noise-only covariance would
  make it the only method with a different likelihood and would manufacture an apparent
  disagreement.
* What must **not** happen is quoting a mock-SBI width as exact to better than a few
  percent, or attributing a ~4% width difference between methods to the inference
  machinery.

The exact analytic decomposition of section 7.4 item 1 is now the way to turn this from
an empirical 0.9266 into a per-band statement, and it is the highest-value outstanding
diagnostic.

### Open item this creates for the A-versus-B agreement test

The 7.3% is not neutral between the two estimators, and this must be settled before the
three-way plot is made:

* **Estimator A** (analytic Gaussian likelihood on the emulator) uses `C` explicitly.
* **Estimator B** (NPE) learns whatever noise its *training samples* carry, and those are
  `mu_paste(theta) + nu(seed)`, i.e. ~0.927 `C`.

So A and B encode different noise models and should be expected to differ by roughly
**3.6% in posterior width**, with B the narrower — from the noise model, not from the
inference machinery. Reporting that difference as an SBI-versus-likelihood disagreement
would be wrong, and it is exactly the size of gap this campaign is trying to resolve.

Three ways to close it, in order of preference:

1. **Add the missing variance at the map level.** Extend the bank to draw a Gaussian
   realization of the y/tau/kappa *signal* from `signal_cls` in addition to the noise.
   This is the principled fix and stays inside the map-level rule. It does not close the
   gap completely, because the galaxy leg stays fixed and so the `C_gg` and `C_gy^2`
   contributions are still conditional, so its residual must be measured the same way
   (2048 draws, whitened chi2 per dim) before it is trusted.
2. **Report both.** Run A with `C` and with the bank's empirical covariance, and quote
   the pair. Honest, cheap, and makes the size of the effect explicit rather than
   folding it into a width comparison.
3. **Accept and label.** Keep `C` for A, keep the bank for B, and state the expected
   3.6% offset up front. Acceptable only if the offset is small compared with whatever
   difference is actually observed.

Do **not** close it by drawing bandpower noise as `L @ epsilon` from `C`. That would make
B's noise match A's likelihood exactly and would look like a clean fix, but it bypasses
the mask coupling and the estimator, it is forbidden for this experiment, and it would
hide precisely the structure measured above.

## Direct sequential NPE on mock simulations, no emulator. Tested, FAIL at 288 points

`oracle_direct_npe_test.py`.  Architecture: no transfer function, no GP.  A few hundred
distinct expensive points, each augmented with 64 free noise draws, trained directly with
NPE.  Proposal: the theory posterior inflated x2 in sd, as requested.  Observation and
exact reference as everywhere else: the production contract's data vector, and the
archived 10,000-sample NUTS chain of the same stand-in model on it.

Why the inflated proposal is necessary, measured.  In probit coordinates the prior is
exactly N(0, I).  The theory posterior's covariance has eigen-sds
[0.044, 0.197, 0.952, 1.111, 1.205]: only two of five directions are constrained, and the
tightest is 23x narrower than the prior.  About 1.8% of prior draws land inside the
posterior's 2-sigma region, so a 288-point prior design would hold roughly 5 useful points
against roughly 288 from the proposal -- a ~55x difference in yield.  The proposal is not
what fails below.

**Naive x2 inflation is the wrong way to apply it.** It exceeds the prior in three of five
directions, which both wastes draws outside the prior and collapses the p0/q importance
weights: measured 5.1% effective sample size, and a posterior that was too *wide* before
reweighting (+1.35) came out too *narrow* after (-0.48 on `theta_ej_0`).  Inflating in the
covariance eigenbasis and clipping each eigen-sd at the prior's 1.0 keeps the full x2
margin where the posterior is informative and gives a design 10x tighter in volume
(0.0348 vs 0.3554 of the prior).  All results below use the capped form.

### SNPE-C, the native sequential path: dies at round 2

Both compressions, 96 distinct points x 64 noise draws in round 1:

```
round 2 TRAINING FAILED: AssertionError('NaN/Inf present in the evaluation of the
MoG proposal posterior...')
```

This is the atomic loss, and it is the same failure the theory campaign hit at round 3
with **65,536** simulations per round on differently-initialised networks.  It is
therefore not a simulation-count problem and more simulations will not fix it.

### Pooled training with analytic reweighting: runs, but does not reach the gates

Avoids the atomic loss entirely -- train plain NPE on all rounds pooled, then reweight from
the analytic pooled proposal mixture to the true prior.

| arm | round | distinct pts | ESS | Pareto k | max drift | max width | max corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| score (5-D) | 1 | 96 | 32.4% | +0.35 | 1.011 s | +0.323 | 0.515 |
| score | 2 | 192 | **70.0%** | +0.51 | **0.416 s** | **+0.454** | 0.602 |
| score | 3 | 288 | **1.3%** | **+1.09** | 1.343 s | +2.130 | 0.995 |
| raw (42-D) | 1 | 96 | 0.0% | +1.28 | 7.269 s | +1.760 | 0.987 |
| raw | 2 | 192 | 35.7% | +0.25 | 0.868 s | +0.742 | 0.784 |
| raw | 3 | 288 | 55.5% | +0.17 | 0.848 s | +0.643 | 0.518 |

Gates: drift <= 0.10 sigma, width <= 10%, correlation <= 0.10.

Three findings:

1. **Nothing reaches the gates.** The best point anywhere is the score arm at 192 distinct
   points: drift 0.416 sigma and width +45%, i.e. about 4x outside both. That is a visibly
   different contour, not a marginal miss.
2. **Sequential sharpening broke the correction that made the earlier rounds valid.** The
   score arm's importance diagnostics went ESS 70% -> **1.3%** and Pareto k +0.51 ->
   **+1.09** from round 2 to round 3. k > 0.7 means the weight distribution has no finite
   variance, so the round-3 posterior (`theta_co_0` sd 3.374 against an exact 1.078) is an
   artifact of a failed importance correction, not a posterior. The mechanism the design
   relies on to sharpen is the mechanism that destroys its validity. The one consolation is
   that k is a *detector*: this failure announces itself rather than producing a
   plausible-looking wrong contour.
3. **Compression matters enormously and is not sufficient.** Raw 42-D at 96 distinct points
   drifts by 7.3 sigma; the 5-D score at the same budget drifts by 1.0. But raw is also the
   more stable across rounds (k improving 1.28 -> 0.25 -> 0.17) while remaining
   persistently over-confident, 49-64% too narrow at every round. Over-confidence is the
   dangerous direction.

### What this does and does not establish

It establishes that *this* estimator family -- MDN SNPE-C, these hyperparameters, this
reweighting scheme -- does not deliver a quotable posterior at 288 pasted points. It does
not establish that no simulation-based estimator can.

The most promising untested alternative is **SNRE or SNLE**: learning the likelihood or the
likelihood ratio means the proposal never needs correcting at all, because the prior is
applied analytically at sampling time. That removes precisely the failure in finding 2, and
it is cheap to test in this same harness with no pastes.

### A measurement-precision problem that affects both routes

The emulator route's posterior metrics turned out to be unstable between emcee seeds by
hundreds of percent: at N=384 the factorized arm gave width **+4.356** in one run and
**+0.300** in another with identical code. The cause is heavy tails wherever `C_emu`
becomes O(1), which a plain standard deviation is not robust to. Any earlier statement in
this document of the form "the posterior is Nx too wide" from a single run is therefore
unreliable and is superseded by this note.

**Consequence: neither route can currently be declared better, and no paste should be
submitted on the strength of these comparisons.** Both harnesses need a tail-robust width
statistic and enough sampling to make the comparison reproducible before the architecture
choice is made.

## SNLE and SNRE: partial, interrupted by the session ending. Most promising so far

`oracle_snle_snre_test.py`, score compression, 96 distinct points per round, 64 free
noise draws each, capped x2 theory-posterior design.  Both arms were killed when the
interactive allocation expired, so these are round-1 and round-2 results only and no arm
reached its round-3 verdict.  **Nothing below is a verdict; it is a partial ladder.**

| arm | round | pts | moment drift | moment width | robust drift | robust width | MCMC |
|---|---:|---:|---:|---:|---:|---:|---:|
| SNRE | 1 | 96 | 2.177 s | +0.618 | 2.012 s | +0.720 | 25 s |
| SNRE | 2 | 192 | 0.769 s | +0.325 | 0.973 s | +0.721 | 25 s |
| SNLE | 1 | 96 | **0.484 s** | **+0.293** | 0.458 s | +0.361 | 248 s |

Gates: drift <= 0.10 sigma, width <= 10%.

Three observations, all provisional:

1. **SNLE's round 1 is the best single result any arm has produced**: 0.484 sigma and
   +29% at 96 distinct points, against direct NPE's best of 0.416 sigma and +45% which
   needed 192 points and then degraded. It is also the only arm whose moment and robust
   statistics agree closely at round 1 (0.484 vs 0.458, +0.293 vs +0.361), meaning the
   number is not a tail artifact.
2. **Both arms improve monotonically with rounds**, which is the property the pooled-NPE
   arm lacked -- there is no proposal correction to collapse, exactly as the structure
   predicts.
3. **Still 4-5x outside the gates.** Nothing here yet justifies a paste campaign.

SNLE's MCMC is 10x slower than SNRE's (248 s versus 25 s per round), which matters for
the sequential loop but not for the final posterior.

The complete four-arm comparison (SNLE and SNRE x score and raw, three rounds each) is
`notebooks/SBI_validate/mock_sbi_sbatch/07_estimator_bakeoff.sbatch`, 12 h on one node
and no pastes.  **The architecture decision, and therefore any paste submission, is
blocked on it.**
