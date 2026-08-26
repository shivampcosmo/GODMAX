---
id: kb.sbi.npe-probit-plateau-runaway
title: The NPE walks onto the probit plateau because only HMC carries the prior in its potential
layer: 60-projects
owner: inference-statistician
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/run_sbi_three_probe_v2.py
  - notebooks/SBI_validate/run_hmc_three_probe_v2.py
  - notebooks/SBI_validate/three_probe_agreement_common.py
  - tests/three_probe_v2/test_sbi_region_and_resume.py
invariants:
  - INV-JAX-SEED-01
  - INV-PRODUCT-PROV-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "E2E_ROOT=... python tests/three_probe_v2/test_sbi_region_and_resume.py (26 checks, no GPU)"
  - "E2E_ROOT=... python tests/three_probe_v2/test_pipeline_end_to_end.py (33 checks, no GPU)"
verified_at_commit: 29c3a27
verified_on: 2026-08-22
see_also: [kb.sbi.analytical-hmc-sbi, kb.sbi.three-probe-posterior-comparison-plan]
---

# The NPE walks onto the probit plateau because only HMC carries the prior in its potential

## The failure

Job 6928795 (three-probe SBI v2, 4 rounds x 32,768) died after 48 minutes, in
round 4's training:

```
AssertionError: NaN/Inf present in posterior eval.
  sbi/inference/snpe/snpe_c.py:331 in _log_prob_proposal_posterior_atomic
  sbi/utils/torchutils.py:372 in assert_all_finite
```

Rounds 1-3 completed and were saved. There was no `diagnostics.json`, so nothing
about Pareto k or the exact-versus-NPE agreement was adjudicated.

## Mechanism

Both samplers work in standard-normal probit coordinates,
`theta = low + (high - low) * Phi(u)`, which makes a `N(0,1)` prior on `u`
*exactly* a uniform prior on `theta`. Measured, not assumed: `Phi(u) == 1.0`
exactly in float64 for `u >= 8.30`. Beyond that radius `theta` pins to the prior
edge and **the forward model is exactly constant** — an infinite likelihood
plateau carrying no information at all.

The `N(0,1)` prior is what suppresses it, and the two samplers hold that prior in
completely different places:

| | where the prior lives | can it reach the plateau? |
|---|---|---|
| HMC | inside the potential — `run_hmc_three_probe_v2.py:226` samples `dist.Normal(0,1)` and line 229 subtracts `0.5*u.u` back out of the factor | no: log-prior at `\|u\|=8` is `-32`, at `\|u\|=10^4` is `-1.9e6` |
| NPE | nowhere inside the density; only in SNPE-C's atomic-loss correction | yes: the fitted density extrapolates freely |

So the NPE's tails ran away across rounds, each round's proposal drawn from the
previous round's runaway density:

| round | max\|u\| of the 40,000 posterior draws | plateau-pinned draws | max\|u\| of the training proposal |
|---|---|---|---|
| 1 | 6.6 | 0 / 40,000 | 4.9 |
| 2 | 44.2 | 1 / 40,000 | 4.8 |
| 3 | **18,201** | 3 / 40,000 | **1,948** |
| 4 | — crashed in training — | | |

The body of the posterior was never sick: round 3's median `max\|u\|` was 1.42 and
only 4 of its 32,768 training rows were pinned. A ~1e-4 tail fraction destroyed a
48-minute job.

## Why this also corrupted the diagnostics, silently

This is the more expensive half. `np.cov(posterior_samples)` and
`credible_interval_summary` are computed over the raw draws, so three outliers at
`\|u\| ~ 1e4` inflated round 3's per-parameter standard deviations from ~1 to
32.9 and 105.9. Round stability is measured in units of the *pooled* sigma, so
the inflation made the gate look **better** than reality:

| | shift as run | shift with the plateau removed |
|---|---|---|
| `u_theta_ej_0` | 0.167 | **0.380** (outside the 0.30 gate) |
| `u_alpha_nt` | 0.000 | 0.090 |
| `u_mu_beta` | 0.001 | 0.053 |
| `u_theta_co_0` | 0.015 | 0.144 |
| `u_nu_theta_ej_M` | 0.364 | 0.365 |

Had round 4 trained successfully, the campaign would have reported an
artificially small round-to-round drift. The same inflation would have made the
Student-t importance proposal — the job's *only* independent check — useless,
since its second mixture component is built from `np.cov(posterior_samples)`.

## The fix

Draws are restricted to `\|u\|_inf <= IDENTIFIABLE_U_RADIUS = 8.0` by **re-drawing**,
never clipping (clipping piles mass onto the boundary and distorts the density
silently). The two restrictions are not the same kind of statement:

- **Proposal draws: statistically free.** sbi's
  `_log_prob_proposal_posterior_atomic` never evaluates the proposal density —
  only `prior.log_prob(atoms)` and the net at batch atoms; the proposal enters
  solely through the round index. Restricting it changes efficiency, not the
  estimated target. Retroactively dropping already-saved plateau rows is the same
  redefinition applied to a saved proposal, and is equally free.
- **Posterior draws: a genuine support statement**, recorded as
  `identifiable_region` in `diagnostics.json` with
  `excluded_prior_mass_upper_bound = exp(-8^2/2) = 1.27e-14`. This is *not* a
  loosened tolerance under `INV-PROC-NOTOLERANCE-01`: it tightens every gate it
  touches (see the table above) and the excluded region is provably
  uninformative, since the forward model there is bit-identical to its value at
  the boundary.

`sample_in_region` refuses outright rather than degrading if acceptance inside
the region falls below 2%: a density placing 98% of its mass on the plateau is
unusable, not merely inefficient.

Two supporting changes:

- **Per-round training seed pinned** (`seed_from_entropy(..., round, 4)` before
  every `inference.train`). Without it the RNG state entering training depended
  on how many draws earlier rounds happened to consume, so a `--resume` produced
  a different network than the original run and any round it regenerated was
  unreproducible. With it, a round's network is a deterministic function of its
  data and round index alone — test [4] asserts a regenerated round is
  bit-identical.
- **A late training failure no longer loses the whole campaign.** The assertion
  is caught, recorded in `training_failure`, and the run proceeds to the
  exact-likelihood reference with the last good posterior. `all_rounds_trained`
  is a gate item, so a truncated campaign can never report `PASS`.

## Provenance-checked resume

`--resume` replays saved `round_*_simulations.npz` instead of re-simulating, and
refuses across a changed contract, reference point, numerical-source aggregate,
grid, or per-round seed. The runner's own sha is deliberately **not** part of that
check: the saved `(u, summary)` pairs depend on the forward model, the reference
point and the seeds, not on the training code. Because the sbatch calls the runner
twice (`--preflight-only`, then the real run), the original `preflight.json` is
snapshotted to `resume_provenance.json` on first use — otherwise the second
invocation would compare against provenance the first invocation had just written,
and the check would be vacuous.

## Evidence

- `tests/three_probe_v2/test_sbi_region_and_resume.py` — 26/26, including the
  literal `\|u\| = 1.82e4` pathology, `theta(u=8.31) == prior upper bound exactly`,
  resume calling the simulator **zero** times, and bit-identical regeneration.
- `tests/three_probe_v2/test_pipeline_end_to_end.py` — 33/33 with the change
  (Pareto k 0.555, ESS 3830/6000, max NPE width error 0.047 on the stub).
- Verified against job 6928795's own saved artifacts: 32 numerical source files
  unchanged, reference point sha matches, and all three stored round seeds
  reproduce exactly, so rounds 1-3's 98,304 GPU simulations are reusable.

## What this does not explain

Two of five round-3 shifts are outside the 0.30 gate once the plateau is removed.
That is a real convergence question about the NPE at 98,304 simulations, not a
numerical artifact, and round 4 is what tests it. A `COMPLETED_REJECTED` on round
stability remains a live possible outcome.
