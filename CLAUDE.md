# GODMAX — working agreement for agents

GODMAX is a JAX halo-model framework for cosmological cross-correlations (tSZ, kSZ, tau,
CMB lensing, galaxy clustering, weak lensing), used for real analyses whose outputs get
published. The expensive failure mode here is **not a crash — it is a plausible wrong
number**. Everything below exists to prevent that.

## Before you change anything

```bash
python tools/kb/kb.py which <files you will touch>   # who owns this, which rules apply
python tools/kb/kb.py stale                          # what you must not trust
```

`knowledge/70-validation/VALIDATION_LOOP.md` governs every change: S0 charter → S1 locate →
S2 **pre-register a falsifiable prediction** → S3 invariant self-check → S4 execute →
S5 evidence → S6 refute → S7 gate → S8 record. Maximum three laps, then escalate.

New machine or fresh clone: `bash tools/kb/install.sh` (`.git/hooks` is not tracked, so the
push gate must be installed per clone).

## The five rules that override convenience

1. **No number without the command that produced it.** Stored notebook outputs are never
   evidence — this repository holds megabytes of output from unknown code versions.
   Re-execute, or mark the claim UNVERIFIED. (`INV-PROC-EVIDENCE-01`)
2. **Never loosen a tolerance, eigenvalue cut, prior width, or ell range to make a check
   pass.** That converts a detected error into an undetected one. It is a physics change:
   own document, invariant review, explicit user sign-off. (`INV-PROC-NOTOLERANCE-01`)
3. **Report the absolute result, not the improvement.** Goodness of fit is judged against
   `retained rank − n_varied`. Stage-31 `fast1024`: `459 − 31 = 428 ± ~29`. The v1 best fit
   at whitened chi2 = 7346 is **not a good fit** — it is an operational point for
   map-pasting. Quoting the 224× improvement without the absolute number is a blocker
   violation. (`INV-CHI2-HONEST-01`)
4. **Show what did *not* change.** A fix reported without a null control has demonstrated
   nothing. This is the most frequently skipped and most valuable line of evidence.
5. **Escalate rather than lower the bar.** If a result will not come right, say so with the
   evidence and the remaining hypotheses. A confident wrong answer costs far more than an
   honest unresolved one.

## Conventions that fail silently

These have each produced a wrong number with no error message. Full statements:
`python tools/kb/kb.py invariants`.

| Rule | Getting it wrong looks like |
|---|---|
| NaMaster covariance is **band-major**: `cov.reshape(n_band, n_comp_a, n_band, n_comp_b)[:, a, :, b]` | Nothing raises. Matrix stays positive-definite. Covariance attributed to the wrong probe pair. |
| Covariance must be `gaussian_covariance(..., coupled=False)` | Leading dimension `n_ell` instead of `n_band`; wrong whitening rank. |
| `shear_e_to_kappa_sign = -1` on DES spin-2 fields | Pristine shear EE alongside four inverted cross families. |
| kSZ vector is **raw** `C_ell^{pi,T}`; theory maps via `-T_CMB_uK * A_v_bin * C_ell^{g,tau}`; plots show `-D_ell` | Fit prefers negative gas amplitude while the plot looks right. |
| DESI lens kernel uses **calibrated true-z** n(z), never `Z_PHOT_MEDIAN` | All four galaxy families biased the same way; HOD drifts to compensate. |
| Photometric pz bins **overlap** in true z — per-pz HOD needs separate theory blocks | Adjacent-bin HOD parameters unphysically anti-correlated. |
| Theory compared through **saved bandpower windows**, not at `ell_eff` | Smooth ell-dependent residual tilt no parameter can absorb. |
| ACT y and T get the 1.6 arcmin beam **once** | Monotonic high-ell deficit confined to ACT families. |
| `jax_enable_x64` **before any array is created** | Whitening rank drops below 459; chi2 varies run to run. |
| Never concretise a traced value in a constructor | Exactly zero gradient; the parameter never moves; reads as "unconstrained by data". |
| Guards on the **inputs** of a division/log — `jnp.where` evaluates all arms | Divergences with healthy acceptance; posterior truncated inside the prior. |
| Units and h conventions explicit at every boundary | Amplitudes off by ~0.67 / 0.45 / 1.49 with acceptable shape. |

## Who to ask

Ownership is by **failure mode**, not directory — `kb which` gives the answer mechanically.

| Symptom | Agent |
|---|---|
| wrong estimator, covariance, mask, sign, or noise policy | `measurement-namaster` |
| physically wrong model; unphysical fitted parameter | `halo-model-physicist` |
| zero/NaN gradient, precision, tracing, speed | `jax-numerics` |
| wrong statistical conclusion; convergence; chi2 | `inference-statistician` |
| two xDESI stages disagree; measurement vs theory | `xdesi-lead` |
| pasted maps vs analytic theory | `abacus-paste-validator` |
| broken API contract across the `src/` chain | `godmax-core` |
| knowledge stale, code unowned, gate blocking | `kb-curator` |
| **is this actually right?** | `physics-referee` (refutes; never fixes) |
| needs numbers, reproducibly | `repro-runner` |

Commands: `/kb-status` · `/kb-sync` · `/validate` · `/xdesi-status` · `/invariant-check` ·
`/kb-new`

## Repository map

```text
src/               core library: base_class -> Profiles -> get_Pkz -> get_Cl -> {get_xi, get_cov}
                   dependency injection, all four params dicts threaded through every layer
src/arxiv/         24 SUPERSEDED modules — history only, never import, never cite
src/mcfitjax/      JAX port of mcfit (FFTLog); precision-critical
param_files/       YAML configs, deep-merged: params_default.yaml + project override
notebooks/xDESI/   the active analysis — survey_measure/ (measurement + HMC),
                   abacus_paste/ (map pasting, 77 files)
run_scripts/       samplers per project: pge/, dtai/, delta/
tests/             ONE file (812 lines), covers the xDESI measurement. src/ is untested.
knowledge/         the knowledge tree — read before acting, update after
tools/kb/kb.py     staleness, routing, invariant checks, the push gate
```

Config paths are cluster-absolute (`/mnt/ceph/users/spandey/...`,
`/mnt/home/spandey/miniconda3/envs/ili-sbi/`). `data/`, `outputs/`, `results/`, `logs/` are
gitignored and never travel between machines — a cluster result reaches the laptop only
through the tracked knowledge tree and journal.

## Practical notes

- **Never submit `sbatch` / `srun` / `salloc` without asking.** State node count, wall time,
  and the evidence the job will produce. Over ~1 node-hour is an escalation.
- **Prefer the cheapest sufficient evidence:** `fast1024` before `midres2048`; `cap600`
  before `cap2400` before fullsky. The scaling ladder in `abacus_paste/` exists for this.
- **`tests/test_xdesi_multiprobe_namaster.py` builds its own synthetic HDF5 inputs**, so it
  runs without cluster data. Use it first, and extend it with every fix — it is why the
  measurement conventions are still intact.
- **Adding a params key?** Add it to `params_default.yaml` too, or every other config breaks
  with a `KeyError` at a random depth.
- **Notebooks:** read source cells for intent; re-execute for numbers. Prefer promoting
  reusable logic into a `.py` module so tests can reach it.
- **`hmcbestfit` vs `hmcfailed`** paste configs exist to contrast a good and a bad parameter
  point. Check which produced a result before quoting it.

## Known open threads

1. **`desi_g_auto` chi2 = 6411** of 7346 total, for 40 data points — the dominant misfit.
   Blocks any physical interpretation of the Stage-31 fit. Eliminate in order: shot-noise
   subtraction → lens kernel → scale cuts / 1h–2h transition → HOD flexibility.
2. **v2 chains not diagnosed.** `max_tree_depth: 4` for 31 correlated parameters is low; the
   saturation fraction, r_hat and ESS are unrecorded, so no v2 posterior is quotable.
3. **`midres2048` DESI mask is provisional** — one DR9 random realization.
4. **kSZ at `lmax = 2048`** covers only the low end of the ~1000–7000 reference range; it
   cannot validate kSZ amplitude or shape.
5. **`src/` has no test coverage.** A construction smoke test and a gradient-flow test are
   the cheapest durable improvements available.

Most seed knowledge documents are `status: draft`, `confidence: medium` **by design**: they
were extracted from the prose in `survey_measure/README.md`,
`BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` and `src/context/codebase_summary.md`, not from
line-level reading. Treat them as good hypotheses, verify at S1, and promote them with an
evidence ledger.
