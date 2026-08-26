---
id: kb.sbi.relaxed-agreement-gate
title: The mock-SBI agreement gate was relaxed from 0.10 sigma / 10% to 0.20 sigma / 20% by explicit user decision on 2026-08-25
layer: 60-projects
owner: inference-statistician
status: draft
confidence: medium
scope:
  - notebooks/SBI_validate/oracle_mock_selfconsistent_test.py
  - notebooks/SBI_validate/analyze_mock_sbi_seed_ensemble.py
  - notebooks/SBI_validate/mock_sbi_sbatch/11_sc_reference_60k.sbatch
  - notebooks/SBI_validate/mock_sbi_sbatch/12_sc_seeds_snre.sbatch
  - notebooks/SBI_validate/mock_sbi_sbatch/13_sc_seeds_snle.sbatch
invariants:
  - INV-PROC-NOTOLERANCE-01
  - INV-CHI2-HONEST-01
  - INV-PROC-EVIDENCE-01
  - INV-JAX-SEED-01
checks:
  - "python notebooks/SBI_validate/analyze_mock_sbi_seed_ensemble.py --self-check (summary-only rescoring must reproduce compare_posteriors exactly)"
  - "every arm JSON must contain BOTH profiles under rounds[].profiles"
verified_at_commit: UNSTAMPED
verified_on: 2026-08-25
see_also:
  - kb.sbi.selfconsistent-observation-decides-agreement
  - kb.sbi.mock-sbi-pasted-response-plan
---

# The agreement gate was relaxed, on the record

## What changed

`GATE_PROFILES` in `oracle_mock_selfconsistent_test.py` now holds two profiles:

| profile | mean drift | width | correlation |
|---|---|---|---|
| `preregistered` | 0.10 sigma | 10% | 0.10 |
| `relaxed_20260825` | **0.20 sigma** | **20%** | **0.20** |

`INV-PROC-NOTOLERANCE-01` forbids loosening a tolerance to make a check pass, and names
the required process: own document, invariant review, explicit user sign-off. This is
that document. The sign-off is the user's instruction of 2026-08-25, given after the
first ladder was reported as failing and after the concern was raised and restated.

The user specified drift and width. The correlation threshold was scaled by the same
factor of 2 rather than left at 0.10, because leaving one leg of a three-leg gate
untouched would make the profile fail on a criterion the relaxation was never argued
about. That is a judgement, not an instruction, and it is recorded here as such.

## Why it is not a silent relaxation

Every round is scored against **both** profiles and both verdicts are stored in
`rounds[].profiles` and printed on every progress line as
`[strict PASS|fail | relaxed PASS|fail]`. `--gate-profile` selects only which one sets
the process exit status. A result can therefore never be reported as passing without the
pre-registered verdict beside it. The original thresholds are unchanged and still
present in the code.

## The argument that was made for it, and its limit

The theory-side agreement already on record in `agreement_sc_run01_final.pdf` --
theory-SBI against theory-HMC on the self-consistent contract -- has max mean shift
**0.114 sigma** and 90% width ratios **1.546 / 1.414 / 1.223**, i.e. 22-55% too wide.
So the 0.10 sigma / 10% bar is stricter than the agreement the published figure
demonstrates, and an arm judged against it would be held to a standard the theory
comparison itself does not meet.

The limit of that argument, which must travel with it: the theory-side width ratios were
diagnosed as **the density estimator alone** (the exact-likelihood reference agrees with
HMC to within 3.3% on every width). Relaxing the mock gate to 20% therefore admits an
error of the same kind and roughly the same size as one already identified as an
estimator defect. It does not make that defect acceptable physics; it makes the mock arm
no worse than the theory arm. Any statement of the form "mock SBI agrees with theory"
under this profile means "agrees to within a 20% width inflation that both methods
share", and must be written that way.

## What the relaxation does NOT license

* It does not apply to chi2, PTE, retained rank, eigenvalue cuts, prior widths, ell
  ranges, or any paste-versus-theory comparison. It is scoped to the oracle's
  posterior-agreement statistic and nothing else.
* It does not license quoting a single-seed number. See below.
* It does not change the reference's own noise, which is a separate and independent
  limit on what any gate can resolve.

## Two measurement limits that bound any verdict, relaxed or not

**Run-to-run variability exceeds the gate.** Two SNLE runs with identical designs and the
same seed offset gave **0.549** and **0.194** sigma drift at 128 points; seed 0 and seed
1 gave **0.522** and **0.078** at 512 points. So the scatter is not only the seed choice
but genuine nondeterminism in training and MCMC, and it is 2-5x the relaxed gate. Four
replicates per configuration are run for this reason, and the reported statistic is the
mean over replicates with its standard error. A single-seed pass is not a pass.

**The reference has its own noise.** At ESS 451 of 10,000 the 10k reference carries 0.047
sigma on means and 3.3% on widths -- half the strict gate. The 60k reference
(`oracle_sc_exact_reference_60k.npz`, job 11) targets ESS ~2700, i.e. 0.019 sigma and
1.4%, a tenth of the relaxed gate. Until that exists, a drift of 0.08-0.20 sigma is not
resolved and no verdict at either threshold is quotable.

Rescoring an existing arm against a new reference needs no re-running: drift, width and
correlation change are pure functions of each posterior's stored first and second moments
and the reference's. `analyze_mock_sbi_seed_ensemble.py --self-check` proves this by
reproducing `compare_posteriors` from summaries alone, and it is asserted to 1e-12.

## Averaging the replicates, two ways

Both are reported because they answer different questions.

* **statistic mean** -- mean and standard error, over replicates, of each replicate's own
  max-over-parameters drift and width. Estimates how well *one run* of a configuration
  does, with an honest uncertainty. This is the number for "is this architecture good
  enough".
* **pooled ensemble** -- concatenate the replicates' posterior draws and score the
  mixture. A deep-ensemble posterior that marginalises over the network initialisation.
  This is the number for "what would actually be plotted", and it is *expected to be
  wider* than any single replicate whenever the replicates disagree in the mean, because
  the between-replicate scatter is folded in. That inflation is information, not an
  artefact.

## Outcome, 2026-08-25: the relaxation turned out not to be needed

Jobs 11, 12 and 13 ran. **The four-seed pooled ensemble of SNLE + score compression at
M=64 clears the PRE-REGISTERED gate at 512 pasted points**, so the relaxed profile is not
load-bearing for the headline result. It is load-bearing only for a single run.

| SNLE score M=64, 512 pts | drift (sigma) | width |
|---|---|---|
| single run, mean of 4 seeds | 0.133 +- 0.043 | +0.127 +- 0.050 |
| four-seed pooled ensemble | **0.050** | **-0.066** |

Pooled per-parameter drift `[0.027, 0.043, 0.050, 0.039, 0.050]`, width
`[-0.066, -0.038, -0.004, +0.003, -0.013]`, correlation change 0.045: `preregistered`
PASS and `relaxed_20260825` PASS. At 384 points the pooled ensemble is drift 0.102 /
width 0.062, i.e. relaxed PASS and strict fail by 0.002 -- so 512 is where the strict
gate is actually met, which is the budget that was proposed anyway.

The seeds scatter around the truth in *different directions*, so pooling averages the
offsets out; that is why the ensemble beats every single run. **It is nearly free**: four
network trainings (minutes each) against ONE set of pastes, so it does not change the
paste budget at all. Production must therefore train four networks per round and quote
the ensemble, not one network.

Drift improves monotonically with pasted points for this arm alone: pooled 0.146 ->
0.137 -> 0.102 -> 0.050 at 128 / 256 / 384 / 512.

## A reference error this campaign made, and the null control that caught it

The 10k reference was **biased narrow**, not merely noisy:

| | Pareto k | ESS | sd |
|---|---|---|---|
| 10k | +0.398 | 451 | 0.3673 0.2325 0.9468 0.9824 0.0936 |
| 60k seed 0 | +0.160 | 4168.8 | 0.4335 0.2657 0.9719 1.0018 0.1047 |
| 60k seed 7 (independent) | +0.181 | 4182.7 | 0.4377 0.2679 0.9733 0.9980 0.1052 |

Two *independent* 60k references agree on widths to **0.98%** and on means to 0.059
sigma. The 10k reference disagrees with them by **15.3% / 12.5% / 10.6%** on the three
constrained parameters -- 15x the reproducibility, so it is a bias. Self-normalised
importance sampling at ESS 451 concentrates on high-weight draws and under-covers the
tails, which systematically shrinks the variance.

Consequence, recorded because it inverted a conclusion: against the 10k reference the
arms looked **too wide**, i.e. conservative. Against the correct reference **SNRE-raw is
over-confident by 25-30%** (`-30.3% / -25.1% / -24.3%` at 512 points), which is the
dangerous direction. A width verdict is only as good as the reference's own width, and
that number needs its own replicate before any of it is quoted.

Precision floor to carry with the headline: the reference reproduces its mean to ~0.06
sigma, so the pooled drift of 0.050 sigma is **at the floor and not distinguishable from
zero**. The honest statement is "drift <= 0.06 sigma, limited by the reference", not
"drift = 0.050".

## Arm-by-arm outcome, four seeds each, scored against the 60k reference

| arm | compression | M | pooled drift @512 | signed width @512 | verdict |
|---|---|---|---|---|---|
| SNLE | score | 64 | **0.050** | **-6.6%** | strict PASS |
| SNRE | raw | 64 | 0.064 | **-30.3%** | over-confident |
| SNRE | score | 64 | 0.642 | +21.8% .. +38.2% | broken below 384 pts (7.0 sigma @128) |
| SNRE | score | 5 | 0.282 | +76% .. +82% | fails |
| NPE | score | 64 | n/a | n/a | proposal correction collapses, k 1.18 / 0.78 |

Score compression is essential for NPE and for SNLE, and actively harmful for SNRE.
Free augmentations matter: SNRE-score at M=5 is +76-82% wide against +44% at M=64.

## Status

`analyze_mock_sbi_seed_ensemble.py --self-check` passes on all 64 rounds, so every number
above was rescored from stored moments without re-running an arm. Still UNSTAMPED: the
result is an oracle on a stand-in simulator (`r_hat * mu_theory`), not on real pastes, and
the paste campaign has not been run.
