# xDESI — directory-scoped agent instructions

Codex reads `AGENTS.md` hierarchically, so this file applies on top of the repo-root
`AGENTS.md` whenever you are working under `notebooks/xDESI/`. Adopt the
`godmax-xdesi-lead` skill for anything spanning more than one stage.

Run first, every time (Codex does not run the Claude lifecycle hooks):

```bash
python tools/kb/kb.py which notebooks/xDESI/survey_measure/ notebooks/xDESI/abacus_paste/
python tools/kb/kb.py stale
python tools/kb/kb.py invariants --layer measurement
```

## What this directory is

Two coupled programmes:

**1. Survey measurement and fit** — `survey_measure/`
A 46-spectrum multi-probe data vector from DESI DR9 Extended LRGs, DES Y3 shear,
and ACT (y, CMB temperature, CMB kappa), measured with NaMaster, compared against GODMAX
analytic theory, and fitted with a fixed-cosmology 31-parameter Stage-31 astrophysical + HOD
model via NumPyro NUTS.

- `multiprobe_namaster.py` (~157 KB) — the estimator: fields, masks, bandpowers, covariance,
  n(z), kSZ, priors, `theory_to_data_vector`. **Four distinct failure modes live in this one
  file** — use `godmax-measurement-namaster` for its internals.
- `godmax_multiprobe_theory_utils.py` — configs, n(z) materialisation, theory extraction.
- `godmax_multiprobe_hmc_stage31.py` — the 31-parameter likelihood and NUTS driver.
- `combine_godmax_hmc_stage31_workers.py` — pools multi-GPU workers; highest-risk file for
  statistical validity.
- `README.md` and `BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` — **the convention record.** Read both
  before your first substantive change; most of the invariant registry was extracted from
  them. They are prose, so treat them as hypotheses to check against code, not as proof.

Stages: `fast1024` (nside 1024, lmax 1024, 10 linear bins) for validation; `midres2048`
(nside 2048, ell 128–4096, 16 hybrid-log bins, lmax_mask 6143, no added apodisation) for production.

**2. Abacus Backlight paste validation** — `abacus_paste/` (77 files) plus the `abacus_*`
helpers. Paste the best-fit point onto Abacus lightcone halos, measure with the same
estimator, check map-level against analytic. This is a **null test of the pipeline**, not a
cosmology fit. Use `godmax-abacus-paste-validator`.

## State you must not misrepresent

- v1 best-fit whitened chi2 = **7346.23** against an expectation of about
  **459 − 31 = 428 ± ~29**. **Not a good fit.** An operational point for map-pasting, nothing
  more. `desi_g_auto` contributes 6411 of the total — the misfit is localised, not diffuse.
- The 1.65e6 → 7.3e3 improvement is not evidence of a good model. Quoting it without the
  absolute comparison violates blocker `INV-CHI2-HONEST-01`.
- The `midres2048` DESI high-ell mask uses one DR9 random realization and is recorded as
  provisional.
- `ell_max = 2048` covers only the low end of the ~1000–7000 range the harmonic kSZ reference
  analysis fits. Low-resolution products cannot validate kSZ amplitude or shape.

When asked "how is the fit doing", lead with the absolute goodness of fit and the dominant
family — never the improvement factor.

## The six conventions that break this analysis silently

Full statements: `python tools/kb/kb.py invariants --layer measurement`.

1. **Windowed theory only** (`INV-WINDOW-CMP-01`) — `theory_to_data_vector(...)` in the saved
   product-specific convention. Smooth theory at `ell_eff` is a diagnostic. Bandpowers are wide and
   partly logarithmic; the band average differs from the value at `ell_eff` in an
   ell-dependent way that mimics real physics.
2. **Calibrated true-z lens kernels** (`INV-NZ-TRUEZ-01`) — never the `Z_PHOT_MEDIAN`
   histogram. Guard key: `desi_lens_redshift_kind = spectroscopic_calibrated_true_redshift`.
3. **Overlapping photometric bins** (`INV-HOD-PZBIN-01`) — per-pz HOD needs one galaxy theory
   block per photometric bin, each with its own true n(z) and nbar(z), on top of one shared
   non-galaxy block. Never disjoint true-z slices. Applies to paste-map work too.
4. **Shear sign** (`INV-SHEAR-SIGN-01`) — `shear_e_to_kappa_sign = -1` leaves EE unchanged and
   flips every scalar × shear-E.
5. **kSZ sign and calibration** (`INV-KSZ-SIGN-01`, `INV-KSZ-CALIB-01`) — measured vector is
   raw `C_ell^{pi,T}`; theory maps through `-T_CMB_uK * A_v_bin * C_ell^{g,tau}`; plots use
   `-D_ell`. Amplitude uses r = 0.3 and the Abacus `sigma_true_gas/c` values.
6. **Data-vector layout** (`INV-DV-SHAPE-01`) — 46 spectra; 460 elements for fast1024 and
   736 for the 16-band midres2048 product. Covariance rank is measured per exact product;
   pipeline-v2 fast1024 is 460/460 while rank 459 belongs to legacy v1. Everything downstream
   indexes positionally.

## How to localise a discrepancy

In this order. **Never skip to step 3** — a model change that absorbs a step-2 error fits
better and is wrong, which is the worst available outcome.

1. Is the **measurement** self-consistent? Nulls, shuffled-velocity tests, B-modes,
   mask/apodisation variations. → `godmax-measurement-namaster`
2. Is the **comparison** right? Windows, beams, pixel windows, signs, m-bias, units. This is
   where most discrepancies actually live. → `godmax-xdesi-lead`
3. Is the **model** right? Profiles, HOD, transition model, kernels. →
   `godmax-halo-model-physicist`
4. Is the **inference** right? Whitening rank, tree depth, convergence, worker pooling. →
   `godmax-inference-statistician`

## Notebooks and cluster jobs

Very large notebooks live here: `abacus_particle_shell_residual_tests.ipynb` (3.9 MB),
`test_fit_abacus.ipynb` (2.6 MB), `abacus_gg_profile_consistency_tests.ipynb` (1.8 MB),
`abacus_satellite_clustering_diagnostics.ipynb` (1.5 MB). Their stored outputs are from
unknown code versions and are **never evidence** (`INV-PROC-EVIDENCE-01`). Read source cells
for intent; re-execute for numbers, or mark the claim UNVERIFIED. Prefer promoting reusable
logic into a `.py` module so `tests/` can reach it.

`kb` hashes only notebook *source* cells, so re-running a notebook does not falsely mark
knowledge stale — you do not need to avoid executing them.

Paths are cluster-absolute (`/mnt/ceph/users/spandey/...`,
`/mnt/home/spandey/miniconda3/envs/ili-sbi/`). **Never submit `sbatch`/`srun` without
asking**; state node count, wall time, and the evidence the job produces. Use the capacity
ladder — `cap600` → `cap2400` → `cap4800` → fullsky — and never submit a full-sky paste
without a cap-based scaling estimate. `hmcbestfit` and `hmcfailed` config variants exist to
contrast a good and a bad parameter point; check which produced a result before quoting it.
