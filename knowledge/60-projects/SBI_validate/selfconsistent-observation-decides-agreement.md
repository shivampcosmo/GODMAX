---
id: kb.sbi.selfconsistent-observation-decides-agreement
title: HMC and SBI could not agree because the observation was a pasted-map measurement; on a self-consistent one the sampler agrees and only the NPE width does not
layer: 60-projects
owner: inference-statistician
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/build_three_probe_selfconsistent_observation.py
  - notebooks/SBI_validate/verify_three_probe_selfconsistent_contract.py
  - notebooks/SBI_validate/three_probe_inference_contract.py
  - notebooks/SBI_validate/run_hmc_three_probe_v2.py
  - notebooks/SBI_validate/run_sbi_three_probe_v2.py
  - notebooks/SBI_validate/compare_three_probe_v2_hmc_sbi.py
  - notebooks/SBI_validate/plot_three_probe_sc_agreement.py
invariants:
  - INV-CHI2-HONEST-01
  - INV-PRODUCT-PROV-01
  - INV-MCMC-TREEDEPTH-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "python notebooks/SBI_validate/compare_three_probe_v2_hmc_sbi.py --hmc-dir .../hmc_sc/run01 --sbi-dir .../sbi_sc/run01 (NPE vs HMC DISAGREE on 3 widths; exact vs HMC AGREE)"
  - "python verify_three_probe_selfconsistent_contract.py --contract .../inference_contract_selfconsistent.yaml"
  - "E2E_ROOT=... python tests/three_probe_v2/test_pipeline_end_to_end.py (41 checks)"
  - "E2E_ROOT=... python tests/three_probe_v2/test_sbi_region_and_resume.py (31 checks)"
  - "python tests/three_probe_v2/test_contract_registry.py (13 checks)"
  - "python tests/three_probe_v2/test_defensive_proposal.py (9 checks)"
verified_at_commit: 29c3a27
verified_on: 2026-08-24
see_also: [kb.sbi.npe-probit-plateau-runaway, kb.sbi.analytical-hmc-sbi,
           kb.sbi.mock-sbi-pasted-response-plan]
scope_digest: sha256:0e66d9c399efa4c429f35b6d0c8955ed
---

# HMC and SBI could not agree because the observation was a pasted-map measurement

## The question that should have been asked first

The v2 campaign asked "do HMC and score-compressed SBI agree?" while the contract's
observation was **Abacus pasted maps plus one noise realization**
(`observation.h5`, `observation_realization_index = 0`, source hashes
`signal_map` / `noisy_ensemble` / `parent_realization`). The samplers' forward
model is the analytic JAX halo model. Those are different functions, so the
chi-square floor is the paste-versus-theory mismatch:

| | chi2 |
|---|---|
| production MAP | 161.09 |
| **at the parameters the maps were painted at** | **218.67** |
| nominal `42 - 5` | 37 +- 8.6 |

The model fits *worse* at the generating parameters than at its own MAP. It cannot
reach the paste anywhere. The theory and pasted data vectors differ by up to
**48%** band by band (first `gy` band: ratio 1.4517).

Every difficulty in the campaign followed from that, and none of it was a sampler
defect:

* the observation is a ~13 sigma outlier for the simulator, which is why
  conditional NPE on the raw 42-vector failed and the score compression was needed;
* the posterior is dragged onto a stiff non-Gaussian profile as the model strains
  to absorb a misfit it cannot absorb -- `delta chi2 = 10` at one Laplace sigma,
  flattening to 14 at two, so a step size of 0.02 with ~205 leapfrog steps per
  trajectory against a depth-7 ceiling of 127: **72.24% saturation**;
* the same stiffness broke the exact-likelihood importance reference
  (Pareto k 0.935, ESS 25 of 20,000), so the one independent check could not
  adjudicate;
* NPE-versus-HMC mean shifts reached 0.935 sigma with a width ratio of 0.693.

A wrong diagnosis to retire: the ridge is **not curved**. A profile-likelihood scan
shows curvature explains only 1.0% (production) and 4.8% (self-consistent) of the
ridge scatter. The apparent bend in the HMC marginals was the probit map plus
marginalisation. The mass matrix was also never at fault -- warm-up adaptation took
it from condition 3044 against the sampled covariance down to 1.66-2.54.

## The fix: a self-consistent theory observation

`build_three_probe_selfconsistent_observation.py` writes an observation that IS the
forward model's own prediction at a fixed parameter point, reusing the production
covariance, Cholesky, window, galaxy pixel window and profile Bell **byte for
byte** (asserted in `test_contract_registry.py`). Then chi2 at the generating
point is **0.0 exactly** at build time and `2.39e-29` when replayed through
`build_problem`, so a disagreement can only be the inference machinery.

Registered as a second contract, not a replacement. Identity is pinned by one hash
-- the contract file's own -- and the contract declares the hash of every array it
admits, closing the chain `pinned sha -> contract -> array hashes -> arrays`. An
unregistered path and a tampered contract are both refused. The generating point
lives in a sibling file no sampler reads, so `FORBIDDEN_ORACLE_TOKENS` still holds
over the contract itself.

## Measured effect, converged grid, one A100 trial each

| | production `run01` | self-consistent trial |
|---|---|---|
| tree-depth saturation | 72.24% | **0.000%** |
| mean leapfrog steps | 107.4 | **19.5** (max 63 of 127) |
| sampling rate | 8.12 s/it (H100) | 3.74 s/it (A100) |
| HMC replayed median chi2 | 164.61 | **3.01** |
| SBI Pareto k | 0.935 | **0.349** |
| SBI max importance weight | 0.181 | **0.0166** |
| SBI posterior-weighted chi2 | 165.50 | **3.83** |
| NPE-vs-HMC mean shifts (sigma) | 0.935 0.774 0.157 0.246 0.908 | **0.029 0.019 0.025 0.038 0.018** |
| exact-vs-HMC verdict | DISAGREE | **AGREE** |

Pulls of each posterior mean against the known truth agree between methods to
within 0.1 sigma everywhere; the largest is `theta_co_0` at +1.14 (HMC) / +1.24
(NPE), which is nearly prior-dominated. Residual gap: NPE 90% widths are 1.24 and
1.20 times HMC's on `theta_ej_0` and `nu_theta_ej_M`, outside the [0.85, 1.18]
gate. This document previously attributed that to the trial's 49,152-simulation
budget and expected the production 262,144 to remove it. **That is refuted** --
see the next section.

## Two supporting changes

**The importance proposal is now shape-following.** A two-component elliptical
mixture cannot cover a sharp-core/broad-shoulder target. It is now a Student-t(4)
mixture of the MAP Laplace, one deliberately over-dispersed copy of it, and up to
8 k-means components of the NPE draws. On a banana target: ESS 7724 versus 2693
(2.87x), max weight 0.0006 versus 0.0044. With no usable cluster the weight goes
to the **broad** component, never the narrow one.

**Both runners take `--max-wall-seconds`.** A wall budget enforced by SLURM killing
the process loses the final artifact and the gate verdict, leaving only
checkpoints. HMC declines to start a chunk that would overrun; SBI declines to
start a *round* (a round cannot be stopped part-way) and goes to the validation.
Either way the artifact records `stopped_early` / `budget_stop` and
`reached_requested_draws` / `completed_requested_rounds` become gate items, so a
short campaign can never report PASS. This makes a deadline a guarantee rather than
a rate extrapolation.

## 2026-08-24: the production budget refutes the simulation-budget explanation

The trial was superseded by the full pair on the same contract
(`3d229e15dee45c0c039397cf67751cdc2c90d85b6695827b82fcafb93cba2dc6`, verified
through the loader for both legs): HMC `hmc_sc/run01` complete at 1800 draws x 4
chains, theory SBI `sbi_sc/run01` at 4 rounds / 262,144 simulations. The earlier
figure `agreement_sc_run01_partial.pdf` was made mid-flight from
`checkpoint_001100.npz` against 2 rounds, and had no companion comparison JSON --
so the plot showed agreement that had never been measured.

Artifacts, both new:

* `data/SBI_validate/three_probe_inference/agreement_sc_run01_final.pdf`
* `data/SBI_validate/three_probe_inference/comparison_sc_run01_final.json`

Widths moved the **wrong way** with 5.3x the simulations:

| NPE-vs-HMC 90% width ratio | trial (49,152) | production (262,144) |
|---|---|---|
| `u_theta_ej_0` | 1.238 | **1.546** |
| `u_alpha_nt` | 1.098 | **1.414** |
| `u_nu_theta_ej_M` | 1.195 | **1.223** |

So more simulations is not the explanation, and the gate verdict is
**NPE vs HMC: DISAGREE** on three widths. Means are fine everywhere (<= 0.114
sigma), so this is a width-only failure.

The three-way decomposition localises it, and this is the load-bearing result:

| comparison | max mean shift | width ratio range | isolates |
|---|---|---|---|
| exact reference vs HMC | 0.076 | 0.967 - 1.009 | the sampler |
| NPE vs exact reference | 0.107 | 0.973 - **1.541** | the network |
| NPE vs HMC | 0.114 | 0.977 - **1.546** | both together |

The exact-likelihood reference agrees with HMC to within 3.3% on every width.
The same inflation appears against the exact reference as against HMC. **The
inflation is the density estimator alone** -- not the sampler, not the score
compression, not the contract. Any further work on this belongs in the NPE
architecture / calibration, not in the HMC configuration.

Importance diagnostics for the reference are healthy, so it can carry that
conclusion: Pareto k 0.329, ESS 993.4 of 20,000, max weight 0.0067. getdist
reports a 0.005 outlier fraction on the weighted KDE, which is cosmetic smoothing
at that Pareto k, not a broken estimate.

Coverage against the known truth is consistent between methods -- pulls
+0.12/-0.36/+0.20/+1.22/+0.05 (HMC) against +0.11/-0.16/+0.31/+1.28/-0.06 (NPE).
Read the `theta_co_0` pull of ~+1.2 with care: the rendered marginals show
`mu_beta` and `theta_co_0` are **prior-dominated**, near-flat out to the box
edges, so all three methods agreeing there carries little information and the
pull is not a tension.

### There is still no converged HMC chain

All four HMC runs on record are gate-rejected. The oldest wrong assumption to
avoid repeating is treating any of them as ground truth:

| run | date | r_hat | min ESS | divergences | depth saturation | rejected on |
|---|---|---|---|---|---|---|
| `hmc_sc/run01` | 08-23 | 1.0037 | 1063 | **12** | 0.014% | divergences |
| `hmc_selfconsistent_trial/ta080` | 08-23 | 1.0318 | 68 | 3 | 0% | r_hat, ESS, div |
| `hmc_v2/run01` | 08-23 | 1.0127 | 712 | 0 | **72.2%** | r_hat, saturation |
| `hmc_depth6/job_6917288` | 08-20 | 1.0061 | 953 | 15 | 22.9% | divergences |

`hmc_sc/run01` is much the healthiest -- the self-consistent contract took
saturation from 72.2% to 0.014% -- but 12 divergences against a threshold of 0
still reject it. Nothing here is quotable as a certified posterior, and
`agreement_sc_run01_final.pdf` therefore carries no title claiming otherwise.
See [[posterior-gates-need-valid-reference]] for what scoring against a rejected
chain cost the mock-SBI campaign.

### Reproducing the figure and the numbers

Both read saved artifacts only -- no forward model, no sampler, seconds to run:

```bash
D=data/SBI_validate/three_probe_inference
python notebooks/SBI_validate/plot_three_probe_sc_agreement.py \
  --hmc-dir $D/hmc_sc/run01 --sbi-dir $D/sbi_sc/run01 \
  --generating-point $D/observation_selfconsistent_generating_point.json \
  --no-title --legend-fontsize 15 --output $D/agreement_sc_run01_final.pdf
python notebooks/SBI_validate/compare_three_probe_v2_hmc_sbi.py \
  --hmc-dir $D/hmc_sc/run01 --sbi-dir $D/sbi_sc/run01 \
  --output $D/comparison_sc_run01_final.json
```

`--no-title` and `--legend-fontsize` were added on 2026-08-24; defaults are
unchanged, so the earlier figures still reproduce.

### How the exact-likelihood reference is built

Worth stating because it is neither a chain nor a network, and it is what makes
the decomposition above possible. It is self-normalised importance sampling
against the exact analytic likelihood
(`run_sbi_three_probe_v2.py`, the block after the rounds loop):

1. build a 10-component Student-t(4) proposal -- 0.30 Laplace at the pinned MAP
   (scale x1.5), 0.10 the same centre inflated x9, and 0.60 spread over 8 k-means
   clusters of the NPE draws;
2. draw 20,000 points, restricted to `|u|_inf <= 8`;
3. evaluate the **real** forward model at every one,
   `batch_chi2 = jax.jit(jax.vmap(problem.chi2_u))`;
4. `log_weights = (-0.5*chi2 - 0.5*|u|^2) - student_t_mixture_logpdf(...)`;
5. the weighted set is the posterior; getdist plots it with those weights.

The NPE enters **only** through the proposal, which divides out exactly in step 4,
so the network can cost efficiency but never correctness -- that is precisely why
it can adjudicate the network. The Laplace and over-dispersed components exist to
bound the weight of any point the NPE clusters misplace: an earlier run at 768
simulations had an NPE posterior 18-53x too wide and gave Pareto k 15.0 with ESS
1.66 of 384, i.e. a reference that failed exactly when it was needed.

## What this contract does NOT settle

The 48% paste-versus-theory discrepancy is untouched and remains the real physics
thread -- the low-ell `gkappa`/`gtau` deficit, owned by `halo-model-physicist`.
This contract answers "is the inference machinery correct", which had to be
answered first; it does not license any statement about the pasted maps.
Correspondingly, a chi2 of ~3 against a nominal 37 is **expected** here and would
be suspicious on a noisy observation: `compare_three_probe_v2_hmc_sbi.py` used to
print "both posteriors describe a model that does not fit this data vector"
unconditionally, which was true on the production contract and false here. It is
now computed from the numbers.
