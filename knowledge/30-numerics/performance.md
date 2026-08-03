---
id: kb.numerics.performance
title: Likelihood-evaluation performance profile and safe optimisation targets
layer: 30-numerics
owner: jax-numerics
status: draft
confidence: low
scope:
  - src/get_radial_profiles.py
  - src/get_Pkzs.py
  - src/get_Cls.py
  - src/matter_pk_symbolic.py
  - src/hmf_symbolic.py
invariants:
  - INV-JAX-TRACE-01
  - INV-JAX-X64-01
checks:
  - "TODO(jax-numerics): benchmark harness reporting compile vs steady-state time per likelihood eval"
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.numerics.jax-contract, kb.xdesi.analysis-state]
scope_digest: sha256:3122aa7f940bef55aa9bf05b44812867
---

## Claim

The dominant cost per likelihood evaluation is rebuilding the full class hierarchy
(order 5–15 s). Optimisation is worth pursuing because it gates feasible chain lengths, but
**no** optimisation may alter forward values or gradients, and the constructor must keep
tracing correctly inside the numpyro model.

## Why it is true

This document currently records an **unverified** analysis carried over from earlier
profiling notes (`performance_bottlenecks.md` referenced in project memory, not present in
the tree at `43e07ca`). It is marked `confidence: low` for that reason and must be
re-measured before being acted on.

The recorded claims, to be re-tested:

1. **Full class hierarchy rebuilt per likelihood evaluation** (~5–15 s) — the top cost. Each
   `base_class → Profiles → get_Pkz → get_Cl` construction redoes grid setup and builds
   `interpax` interpolators.
2. **Multi-way `jnp.where` branches evaluate every arm.** This is required for correctness
   under tracing, so it cannot simply be removed.
3. **Quick wins:** the symbolic P(k) and HMF emulators in place of numerical integrals;
   precomputing log-arrays; removing timing decorators from traced code.

Why it matters concretely: the Stage-31 v2 configuration requests 8000 samples × 4 chains.
At 5 s per evaluation with the tree depths NUTS actually uses, that is not feasible; the
existing workaround is fanning across up to 16 GPU workers, which then creates the chain
pooling risk described in `INV-MCMC-CONVERGENCE-01`. Making the likelihood faster reduces a
statistical risk, not just a wall-clock one.

## How to verify

Re-measure before acting. Compile and steady-state time must be reported **separately**, in a
fresh process, with a warm JIT cache for the steady-state number:

```bash
python - <<'EOF'
# import order matters: enable x64 BEFORE any array is created (INV-JAX-X64-01)
import time, jax
jax.config.update("jax_enable_x64", True)
# ... build the likelihood closure ...
# t0 = time.perf_counter(); f(params).block_until_ready()   # first call: compile + run
# t1 = time.perf_counter(); f(params).block_until_ready()   # second: steady state
# print("compile+run", t1-t0, "steady", time.perf_counter()-t1)
EOF
```

`block_until_ready()` is mandatory — JAX is asynchronous, and timing without it measures
dispatch, not computation. A timing decorator placed inside a traced function measures trace
time and is meaningless.

## Failure modes

- **Optimising with a forward-only comparison.** Passes for changes that destroy
  differentiability. Always compare `jax.grad` output as well
  (`INV-JAX-TRACE-01`).
- **Hoisting construction out of the traced region to "cache" it.** If any sampled parameter
  is consumed during construction, this silently freezes it — zero gradient, parameter never
  moves. This is the most likely way an optimisation here corrupts a result.
- **Resolving a `jnp.where` selector that depends on a sampled parameter.** Only *static*
  selectors — config choices — may be resolved at construction time.
- **Reaching for the symbolic emulators outside their training domain.** They extrapolate
  smoothly and wrongly; a new cosmology, mass, or redshift range needs a domain check first.
- **Reporting a speedup from a warm cache against a cold baseline.**

## Open questions

- **Everything in this document needs re-measurement.** `performance_bottlenecks.md` is not
  present in the tree at `43e07ca`, so the 5–15 s figure and the ranking are unverified.
  Owner: `jax-numerics`. Not blocking correctness, but blocking any optimisation work — do
  not act on these numbers until they are reproduced.
- Whether the construction cost can be reduced while preserving `INV-JAX-TRACE-01` is an
  open design question. Any proposal needs the gradient-agreement test in
  `kb.numerics.jax-contract` and `physics-referee` sign-off, because the failure mode is a
  silently frozen parameter rather than a visible error.
