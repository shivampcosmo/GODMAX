---
id: kb.numerics.jax-contract
title: The JAX contract — x64, tracing boundaries, gradient flow, determinism
layer: 30-numerics
owner: jax-numerics
status: draft
confidence: medium
scope:
  - src/get_Cls.py
  - src/get_Pkzs.py
  - src/mcfitjax/mcfit_jax.py
  - src/helpers/jax_cosmo_power.py
invariants:
  - INV-JAX-X64-01
  - INV-JAX-GRAD-FINITE-01
  - INV-JAX-TRACE-01
  - INV-JAX-SEED-01
checks:
  - "TODO(jax-numerics): gradient-finiteness test at fiducial, best fit and prior corners"
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.arch.class-chain, kb.numerics.performance]
scope_digest: sha256:3feaf673879475d4319ecb9b5fae3cf3
---

## Claim

Four rules make the differentiable pipeline trustworthy: float64 is enabled before any array
exists; the gradient is finite everywhere in the prior volume; constructors trace correctly
without concretising traced values; and every stochastic step records its key. All four fail
**silently** when violated — none raises an exception.

## Why it is true

**x64** (`INV-JAX-X64-01`). `README.md:150-159` documents the required preamble:

```python
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)
numpyro.set_platform("gpu"); numpyro.enable_x64()
```

Order matters: JAX fixes an array's dtype at creation, so enabling x64 after any array exists
leaves that array in float32 and produces a mixed-precision graph. Two things in this
pipeline are not float32-safe: the FFTLog and Limber integrals, and the covariance
correlation-matrix eigendecomposition at threshold 1e-8 (`INV-WHITEN-RANK-01`) — where
float32 noise sits near the cut and silently changes the retained rank.

**Tracing** (`INV-JAX-TRACE-01`). Per `src/context/codebase_summary.md` and the project
memory: the `get_Cl` constructor creates `interpax` interpolators and is **not JIT-able
standalone**, but it **traces correctly** when constructed inside a numpyro model or a jitted
function; individual methods are JIT-compatible. This is intended behaviour. The trap is
"fixing" it: applying `float()`, `int()`, `.item()`, `bool()`, or Python control flow to a
traced value inside the constructor severs gradient flow to every parameter used during
setup. The symptom is an **exactly zero gradient** for a parameter that demonstrably changes
the likelihood — so NUTS never moves it, and the marginal comes out identical to the prior.

**Gradient finiteness** (`INV-JAX-GRAD-FINITE-01`). `jnp.where` evaluates **all** branches,
so a NaN in an unused arm still poisons the reverse-mode gradient. Guards must go on the
*inputs* of a division or log — the safe-denominator pattern — not only on the output. A
non-finite gradient in one prior corner does not crash NUTS; it produces divergences and a
posterior that stops short of the prior boundary.

**Determinism** (`INV-JAX-SEED-01`). Stochastic steps take an explicit key or seed recorded
in metadata — e.g. `random_seed: 42` in `notebooks/xDESI/abacus_pasting_config.yaml`.

## How to verify

```bash
# x64 present in every entry point that runs physics
grep -rln "jax_enable_x64" run_scripts/ notebooks/xDESI/survey_measure/

# at runtime
python -c "import jax; print('x64:', jax.config.jax_enable_x64, jax.devices())"

# the two tests that actually catch trace breakage (write these as a pytest case)
python - <<'EOF'
# forward agreement is NOT sufficient; the gradient test is the one that matters
# g = jax.grad(loglike)(params)
# assert jnp.all(jnp.isfinite(g))
# assert jnp.abs(g[i_param_used_in_constructor]) > 0   # zero => a traced value was concretised
EOF
```

Expected: `x64: True`; gradient finite at fiducial, best fit, and every prior corner; nonzero
gradient for at least one parameter consumed during construction.

## Failure modes

- **Late x64.** Whitening rank drops below 459; chi2 varies between runs at identical
  parameters; gradients noisy at the 1e-3 level; the eigenvalue cut removes far more modes
  than expected.
- **Concretised traced value.** Exactly zero gradient; the parameter never moves; its
  posterior marginal is indistinguishable from its prior. Easy to misread as "unconstrained
  by the data".
- **Guard on the output of a division instead of its inputs.** Divergences with a healthy
  acceptance rate; posterior edges that stop before the prior boundary.
- **Interpolator or emulator used outside its domain.** `interpax` and the symbolic
  regressors (`matter_pk_symbolic.py`, `hmf_symbolic.py`) do not fail at the boundary — they
  extrapolate smoothly and wrongly.
- **Timing measured inside a traced function.** Measures trace time, not execution time, and
  has produced imaginary speedups here.

## Open questions

- The x64 preamble has not been confirmed present in **every** entry point; the `grep` above
  is the check and it has not been run against a full inventory of runnable scripts. Owner:
  `jax-numerics`. Potentially blocking: a single missing preamble in a production script
  would silently degrade a chain.
- No automated gradient-finiteness test exists. `INV-JAX-GRAD-FINITE-01` and
  `INV-JAX-TRACE-01` are both `check.kind: manual`, i.e. enforced only by an agent
  remembering to argue them. Converting them into pytest cases is the highest-value
  numerics work available. Owner: `repro-runner` with `jax-numerics`.
