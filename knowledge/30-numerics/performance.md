---
id: kb.numerics.performance
title: Likelihood-evaluation performance evidence and safe optimisation targets
layer: 30-numerics
owner: jax-numerics
status: verified
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
verified_at_commit: 29c3a27
verified_on: 2026-08-16
see_also: [kb.numerics.jax-contract, kb.xdesi.analysis-state]
scope_digest: sha256:c4d8cb071743052e35a4f722564f6eed
---

## Claim

No reproducible compile-versus-steady-state benchmark for one full likelihood evaluation is
currently recorded in this document's scope. The historical estimate that rebuilding the
class hierarchy costs roughly 5–15 seconds is therefore a hypothesis, not an evidence-backed
performance result, and must not be used for resource planning. Any optimisation must leave
forward values and gradients unchanged and keep constructor-time parameter flow trace-safe.

## Why it is true

The historical analysis came from `performance_bottlenecks.md`, referenced in project
memory but absent from the tracked tree. A tracked-source search finds no single-evaluation
benchmark that separates compilation from a warmed execution for this class chain. The MAP
driver has a related population `value_and_grad` benchmark
(`notebooks/xDESI/survey_measure/godmax_multiprobe_map_stage31.py:618-716`), but that is a
different workload and does not verify the historical estimate here.

The recorded claims, to be re-tested:

1. **Full class hierarchy rebuilt per likelihood evaluation** (~5–15 s) — the top cost. Each
   `base_class → Profiles → get_Pkz → get_Cl` construction redoes grid setup and builds
   `interpax` interpolators.
2. **Multi-way `jnp.where` branches evaluate every arm.** This is required for correctness
   under tracing, so it cannot simply be removed.
3. **Quick wins:** the symbolic P(k) and HMF emulators in place of numerical integrals;
   precomputing log-arrays; removing timing decorators from traced code.

The changed HOD construction path does now have targeted numerical evidence: the restored
`get_Ncen` and `get_Nsat` methods remain `jit(static_argnums=(0,))`
(`src/get_radial_profiles.py:632-650`), and the regression test obtains finite, nonzero
`jax.jacrev` derivatives with respect to the HOD threshold
(`tests/test_get_radial_profiles.py:132-141`). That is trace-safety evidence for this narrow
path, not a full-likelihood gradient proof and not a timing result.

Performance still matters because it gates feasible chain lengths, but a resource estimate
must start from a synchronized benchmark of the actual likelihood and sampler configuration,
not from the historical numbers above.

A narrower CLM benchmark is now recorded. For the reduced CPU/x64 full-constructor `Pgg`
gradient graph, `direct_shell` used 5,262,760 compiled temporary bytes versus 5,263,144 for
`legacy_fftlog`; first compile+run was 11.72 versus 12.30 seconds and warm evaluations were
about 0.009 seconds for both. The difference is intentionally reported as negligible for the
full graph, whose memory is dominated elsewhere. The direct path nevertheless removes the
CLM FFTLog and log interpolation and forms its target/raw outputs in one contraction. No GPU
memory, full likelihood throughput, or NUTS samples-per-second claim follows from this CPU
test.

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

For the current HOD correctness/null control:

```bash
/usr/bin/env JAX_PLATFORMS=cpu /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python \
  -m pytest tests/test_get_radial_profiles.py -q
```

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

- **The historical performance ranking needs re-measurement.** The absent profiling note
  cannot support the 5–15 second estimate or identify the dominant cost. Owner:
  `jax-numerics`. This does not block the already-tested HOD correctness repair, but it blocks
  optimisation or resource-allocation claims based on those numbers.
- Whether the construction cost can be reduced while preserving `INV-JAX-TRACE-01` is an
  open design question. Any proposal needs the gradient-agreement test in
  `kb.numerics.jax-contract` and `physics-referee` sign-off, because the failure mode is a
  silently frozen parameter rather than a visible error.
