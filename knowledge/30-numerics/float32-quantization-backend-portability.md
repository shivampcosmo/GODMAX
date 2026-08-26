---
id: kb.numerics.float32-quantization-backend-portability
title: The float32 painter quantisation is not backend-portable as a bare cast
layer: 30-numerics
owner: jax-numerics
status: verified
confidence: high
scope:
  - notebooks/SBI_validate/three_probe_jax_forward_model.py
  - notebooks/SBI_validate/three_probe_agreement_common.py
  - notebooks/SBI_validate/check_three_probe_backend_parity.py
  - notebooks/SBI_validate/diagnose_three_probe_backend_divergence.py
invariants:
  - INV-JAX-X64-01
  - INV-PHYS-UNITS-01
  - INV-PRODUCT-PROV-01
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
checks:
  - "notebooks/SBI_validate/three_probe_jax_forward_model.py::assert_float32_quantization_effective (runs on every forward-model construction)"
  - "sbatch notebooks/SBI_validate/submit_three_probe_backend_bisect.sbatch"
verified_at_commit: 29c3a27
verified_on: 2026-08-23
see_also: [kb.numerics.jax-contract, kb.sbi.three-probe-hmc-sbi-verification-summary]
---

# The float32 painter quantisation is not backend-portable as a bare cast

## The failure

`three_probe_jax_forward_model.py` reproduces the Abacus painter's operator, and
the painter stored its projected-profile tables as **float32**. The forward model
therefore narrows deliberately, in `_project_smooth_transform_field`:

```python
projected = projected.astype(jnp.float32).astype(jnp.float64)      # was line 272
smoothed  = smooth_radial_gaussian_jax(...).astype(jnp.float32).astype(jnp.float64)
```

**XLA GPU eliminates that convert pair. XLA CPU performs it.** The GPU was
therefore evaluating a different operator from the CPU, and from the non-JAX host
implementation the forward model is validated against.

Evidence (bisect job 6928490, converged grid `(256,48,48,2049)`, byte-identical
sources across 31 hashed files, both backends on one node):

| probe input | CPU result | GPU result |
|---|---|---|
| `1e-38` | `0.0` | `1.000000e-38` |
| `5.443e-42` | `0.0` | `5.443134e-42` |
| `1e-46` | `0.0` | `1.000000e-46` |

Two facts identify it as elision rather than a denormal-handling difference.
`1e-46` cannot be represented in float32 at any exponent, denormal included, so
the GPU cannot have narrowed at all. And where both are non-zero they differ by
`4.955e-08`, which is float32 epsilon (`2**-24`) — the CPU losing precision the
GPU kept. 236 of 341 probe values that CPU flushed to zero came back non-zero on
GPU; zero went the other way.

## Why it was catastrophic rather than cosmetic

`painter_log_interpolate_jax` branches on whether a value is zero:

```python
safe_log = jnp.where(values > 0.0, jnp.log(jnp.maximum(values, tiny)), -20.0)
```

CPU's zeros take the `-20.0` log-space extrapolation; GPU's tiny non-zeros take
`log(value) ~ -100`. An ~84-unit jump in log space, then `jnp.exp`:

| probe point | CPU | GPU | ratio |
|---|---|---|---|
| 199 | 1.218e-27 | 1.133e-26 | 9.3x |
| 200 | 3.761e-31 | 8.385e-29 | 223x |
| 201 | 1.590e-33 | 4.097e-31 | **258x** |

Propagation was clean and diagnostic: every leaf primitive agreed to ~1e-15
(`j0_safe` 2.4e-15, `i0e` 5.9e-16, `ndtr` 1.4e-15, `interpax` cubic 9.6e-16,
`solve_triangular` 5.3e-15), `Profiles` and `get_Pkz` arrays to ~1e-12, then
`_project_smooth_transform_field` outputs jumped to max relative 1.7-1.9 and
reached the 42-vector at 8.276e-05 — a whitened chi-square gap of **334.159**
against an expected goodness-of-fit scatter of 8.6.

CPU is the correct side: it is the one that agrees with the host GODMAX
implementation to a median 1.2e-05 in the frozen non-regression check.

## What does NOT fix it

Each was tested and falsified, in this order:

1. **ptxas version** (job 6928453). Every JAX log carries
   `CUDA <=12.6.2 miscompile certain edge cases around clamping`, and `CUDA_HOME`
   pointed at a module CUDA 12.5.1. Substituting the conda `ptxas 12.8.61`
   produced **bit-identical** output — `|GPU_new - CPU|` and `|GPU_old - CPU|`
   both exactly 334.159077. That warning is a red herring for this code.
2. **`jax.lax.optimization_barrier` between the converts** (job 6928544).
   Bit-identical to the unguarded version on both backends. This is not a
   scheduling or fusion problem.
3. **`--xla_gpu_enable_fast_min_max=false`**, **`--xla_gpu_autotune_level=0`**,
   **`--xla_allow_excess_precision=false`** (jobs 6928490, 6928551). All five GPU
   arms identical in every run.

## The fix, and two wrong turns on the way to it

`quantize_to_float32` rounds the mantissa by integer bit manipulation in float64
throughout, so **no float32 convert exists for any pass to remove**:
round-to-nearest-ties-to-even on the low 29 mantissa bits above the smallest
normal, rounding onto the `2**-149` denormal grid below it, and overflow to
infinity. The derivative is supplied by `custom_jvp` as the identity.

**Wrong turn 1 -- zero gradient.** The first bitcast revision returned the
rounded value directly. `bitcast_convert_type` has no meaningful JVP, so the
derivative was **exactly 0.0** where the original cast gave 1.0. Every column of
`dmu/dtheta` vanished, `J^T C^-1 J` went singular, and jobs 6928581 and 6928582
died with `RuntimeError: Score Gram matrix is not positive definite` after
multistart L-BFGS stalled wherever it started (potentials 84.1, 83.6, 86.7,
5880.9 where the previous forward gave 251.847 from all four starts). This is
`INV-JAX-GRAD-FINITE-01` exactly. A `stop_gradient` straight-through also works
but is an *algebraic identity*, and XLA has already shown on this code that it
exploits those, so `custom_jvp` is used instead.

**Wrong turn 2 -- flush-to-zero.** The next revision flushed below `2**-126`,
because a jitted `astype(float32).astype(float64)` appeared to do so. It does
not: XLA elides that cast **on CPU as well as GPU**, shape-dependently, so the
"reference" was returning its own input. Measured against numpy's *eager* cast --
the correct target, since the painter wrote its tables with numpy and the host
validator casts eagerly -- flush-to-zero was wrong in **11,323 of 102,012** probe
values. IEEE rounds denormals onto the `2**-149` grid.

Lesson: **a jitted float32 round-trip is not a valid reference for itself.** Only
numpy eager is.

Verified bit-exact against numpy eager float32 across **104,011 values**:
`1e-46` to `1e30` both signs, the denormal grid and its exact half-way points,
4,000 points across the min-normal boundary, round-to-even ties,
exactly-representable values, `+-0`, `+-inf`, `NaN`, overflow. Zero mismatches.
Derivative exactly 1.0 everywhere, finite, and stable under
`jit(grad(vmap(...)))`.

`assert_float32_quantization_effective()` runs on every forward-model
construction and raises on all three failure modes -- inactive, inexact against
numpy, or gradient-dead.

**Independent validation.** The frozen non-regression check against the non-JAX
host implementation, `validate_three_probe_jax_hmc_forward.py`, is **PASS** on the
fixed forward: jax-vs-host median 8.5e-06 (gy), 1.7e-05 (gkappa), 1.1e-05 (gtau)
against a 0.5% gate, and `absolute_chi2_at_audit_point` 218.6719 versus the
historical 218.666. It is now step 1 of stage 1 so this cannot silently regress.

**Backend parity, measured directly** on one node, GPU (A100-80GB) versus CPU,
converged grid, five probit points:

| quantity | before | after |
|---|---|---|
| 42-vector max relative | 1.506e+01 | **6.219e-08** |
| chi-square at the MAP | 334.159 gap | **4.26e-07** |
| gradient max relative | dead | 1.687e-06 |

Production runs on H100 rather than A100; the mechanism is compiler-level, not
device-level, and the stage-1 gate re-verifies on H100 and fails closed.

## The residual, and the tolerance decision

31 of 82 arrays still exceed 1e-9, four of them the deliberate bare-cast positive
controls. The real residual begins at `l2_transform_u_y` (1.002e-01) in **97 of
55,296 entries (0.175%)**, whose absolute values are `3.745e-16` to `4.663e-04`
against an array maximum of `1.306e-02`.

Interpretation: the operator is **genuinely discontinuous** — the `values > 0.0`
branch above — so ordinary ~1e-16 Gauss-Legendre reduction-order differences flip
which side of a float32 rounding boundary a far-tail entry lands on, and the
branch converts that into an O(1) *relative* change in an entry of absolute value
~1e-16. This is amplification of legitimate float64 non-determinism through a
discontinuity the painter itself has, not a residual quantisation defect. It is
not expected to be removable without changing the operator.

The parity gate was therefore **re-expressed on the observable** rather than
relaxed, with explicit user sign-off on 2026-08-22:

| gate | threshold | measured | headroom |
|---|---|---|---|
| 42-vector max relative | `1e-6` | 4.054e-08 | 25x |
| \|delta whitened chi2\| | `1e-3` | 3.294e-05 | 30x |

`1e-3` in chi-square is `1.2e-04` of the goodness-of-fit scatter (8.6). The
property that makes this a re-expression and not a relaxation, per
`INV-PROC-NOTOLERANCE-01`: **the gate still rejects the bug it was written for by
2e+07x on the vector criterion and 3e+05x on the chi-square criterion**, and
rejects the coarse-grid 88-unit version equally. A configuration 100x worse than
the current fix also fails. `PARITY_INTERMEDIATE_TOLERANCE = 1e-9` is retained
for *reporting* intermediate divergence, never as a gate.

## Blast radius

Every GPU-produced number in this project that went through
`_project_smooth_transform_field` was affected, including the rejected depth-6
HMC chain and the whole superseded theory-SBI campaign. That reframes the
depth-6 rejection: it was rejected for divergences and tree-depth saturation,
but its likelihood was also wrong.

**Not audited:** `src/get_sim_maps.py` has 39 one-way float32 narrowings and no
eliminable round-trip pair, so it does not carry this bug — but it is the
map-painting path, it mixes float32 tables into float64 arithmetic, and the
implicit-promotion question there is open. The frozen pasted maps are
hash-pinned, so today's data vector is unaffected either way.
`validate_three_probe_projected_operator.py:139` has the only other round-trip in
the repository; it is eager `np.float32`, never traced, and so is safe — which is
why the host reference stayed correct.
