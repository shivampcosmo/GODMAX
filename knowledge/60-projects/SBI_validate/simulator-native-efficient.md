---
id: kb.sbi.simulator-native-efficient
title: Simulator-native efficient inference
layer: 60-projects
owner: inference-statistician
status: verified
confidence: medium
scope:
  - notebooks/SBI_validate/run_simulator_native_active_sbi.py
  - notebooks/SBI_validate/simulator_full_theory_development.py
  - notebooks/SBI_validate/simulator_full_theory_development.json
  - notebooks/SBI_validate/submit_simulator_native_active_sbi.sbatch
  - notebooks/SBI_validate/submit_simulator_native_active_sbi.sh
invariants:
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
  - INV-JAX-SEED-01
checks:
  - /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m py_compile notebooks/SBI_validate/run_simulator_native_active_sbi.py
  - python tools/kb/kb.py invariants --check --id INV-PROC-NOTOLERANCE-01 --id INV-JAX-SEED-01
verified_at_commit: 29c3a27
verified_on: 2026-08-17
see_also: [kb.sbi.analytical-hmc-sbi]
supersedes: []
scope_digest: sha256:df7311cb2523d87346718637044aa867
---

## Claim

The efficient-SBI experiment consumes either conditional-mean or stochastic mock outputs
through a parameter-and-seed interface, uses the complete covariance-whitened data vector,
and does not call or validate against an analytic theory evaluator while choosing simulations
or accepting a posterior.

## Why it is true

The inference runner dynamically loads only a `simulate(theta, seeds)` backend and rejects
anything that does not declare the full covariance-whitened 51-vector and exact nested order
`gy[0:17]`, `gkappa[17:34]`, `gtau[34:51]`
(`run_simulator_native_active_sbi.py:120-145`).  Simulation rows, seeds, roles, and both
unique-row and returned-execution counts are checkpointed (`:146-230`).  Acquisition uses
only previously simulated discrepancies (`:242-315`), while HMC is first touched inside the
plotting path (`:550-598`) after acceptance has been computed.

The development adapter pins the same observation and covariance hashes as HMC, evaluates a
full nonlinear vector, and whitens it (`simulator_full_theory_development.py:47-130`).  It is
an interface test backend: replacing it with a mock simulator requires no inference-code
change.  Current smoke, HMC-null, stochastic-noise, and interruption/resume commands are
recorded in `knowledge/.kb/ledgers/2026-08-17-simulator-native-efficient-sbi.md`.  These prove
workflow readiness only; production posterior acceptance is deliberately still open.

## How to verify

```bash
/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m py_compile \
  notebooks/SBI_validate/run_simulator_native_active_sbi.py
python tools/kb/kb.py invariants --check \
  --id INV-PROC-NOTOLERANCE-01 --id INV-JAX-SEED-01
```

## Failure modes

- Hidden theory calls make the reported realization count inapplicable to mock simulations.
- Selecting validation points using HMC or exact likelihood values makes posterior agreement
  circular even when the displayed contour looks good.
- Reusing training simulations as holdouts hides emulator bias near the posterior.
- Omitting simulation seeds prevents exact replay of a stochastic simulator.
- Importing the development theory adapter from the inference runner would turn a replaceable
  test backend into a hidden production dependency.

## Open questions

- The first production benchmark uses conditional means plus the already-fixed covariance;
  stochastic mock observations follow the separate noise-aware branch and require their own
  held-out calibration evidence before use.
- Whether 368 shared unique realizations (372 executions with the replay null) suffice is
  exploratory and remains unverified until the held-out simulator and coverage gates pass for
  all three probe configurations.
