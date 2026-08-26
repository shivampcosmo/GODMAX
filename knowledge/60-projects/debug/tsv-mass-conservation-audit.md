---
id: kb.debug.tsv-mass-conservation-audit
title: TSV branch collisionless-mass audit
layer: 60-projects
owner: halo-model-physicist
status: verified
confidence: medium
scope:
  - src/base_class.py
  - src/get_radial_profiles.py
  - src/get_Pkzs.py
  - param_files/params_default.yaml
  - tests/test_get_radial_profiles.py
  - notebooks/debug_tsv/tsv_vs_original_mass_conservation.ipynb
  - notebooks/debug_tsv/compare_clm_shell_fourier.ipynb
  - notebooks/debug_tsv/compare_clm_source_power_ratios.ipynb
invariants:
  - INV-PHYS-MASSBUDGET-01
  - INV-PHYS-BIASNORM-01
  - INV-PHYS-UNITS-01
  - INV-JAX-X64-01
  - INV-JAX-GRAD-FINITE-01
  - INV-JAX-TRACE-01
  - INV-PROC-EVIDENCE-01
checks:
  - jq -e '(.nbformat == 4) and (.cells[29].execution_count != null)' notebooks/debug_tsv/tsv_vs_original_mass_conservation.ipynb
  - jq -e '([.cells[] | select(.cell_type=="code") | .execution_count] | all(. != null)) and ([.cells[].outputs[]? | select(.output_type=="error")] | length == 0)' notebooks/debug_tsv/compare_clm_shell_fourier.ipynb
  - jq -e '([.cells[] | select(.cell_type=="code") | .execution_count] | all(. != null)) and ([.cells[].outputs[]? | select(.output_type=="error")] | length == 0)' notebooks/debug_tsv/compare_clm_source_power_ratios.ipynb
verified_at_commit: 29c3a27
verified_on: 2026-08-16
see_also: [kb.physics.halo-model-ingredients, kb.numerics.jax-contract, kb.validation.loop]
supersedes: []
scope_digest: sha256:8d8d3dad3bbb6d9d936dc9cb5cd398a6
---

## Claim

The TSV central finite-difference reconstruction of collisionless density is a material,
convergent improvement over differentiating `jnp.interp` at its knots. The final matched
large-scale galaxy-power ratio returns to unity, but raw central `u_clm` overshoots one and
the existing upper clip makes that pipeline result unsuitable as independent mass-closure
evidence. The method is not exactly mass-conservative on the default radial grid, and the
TSV branch is not ready for HMC because its constructor concretises sampled tracers and its
Stage-31 callers retain imports made obsolete by the package refactor. The DMB/halofit null
is only approximate on large scales and fails over the full k range.

The finite-volume candidate is now integrated in the main source as the selectable
`direct_shell` default for the galaxy-satellite Fourier window. It carries
`[M(r0), diff(M)]` as shell masses and applies an analytic spherical-shell Fourier kernel
directly. It reproduces every supplied cumulative node and raw `u_clm(k=0)=1`, restores raw
large-scale Pgg without the TSV upper clip, and compiles to one static-kernel contraction.
Its fixed inner continuation `M(<r) proportional to r^2` is the NFW-cusp limit and avoids
resolving the interval from zero to the first radial node. `legacy_fftlog` remains available
for historical reproduction. The integration is deliberately limited: `rho_clm`, DMB, HSE,
and maps still use their previous real-space representation; the represented endpoint can
miss the physical target; and nonlinear-k convergence remains incomplete.

## Why it is true

The original `get_rho_clm` differentiates a piecewise-linear `jnp.interp` at its own knots
(`src/get_radial_profiles.py:847-868`); TSV uses `jnp.gradient(ln_Mclm, ln_r)`
(`/mnt/ceph/users/spandey/ltu-godmax/tsv-godmax/GODMAX/src/godmax/get_radial_profiles.py:959-982`).
The executed notebook applies both operators to the same TSV default-grid `Mclm_mat`. The
original reconstruction gives recovered endpoint-mass ratios 0.888975--0.958364; TSV gives
1.007332--1.026940. A 23--256-point synthetic resolution sweep shows the TSV error converging
from 0.042414 to 0.000312 while the original error remains 0.009189 at 256 points.

The derivative-isolated TSV power test at fixed configuration changes only
`Profiles.get_rho_clm`. The low-k maximum error in the backreaction/no-backreaction
`Pgg_2h` ratio drops from 0.071108 to 0.00002175. However, the central raw first-bin
`u_clm` spans 1.010908--1.326173 and the interpolated/clipped values are all exactly one;
the final Pgg result is therefore confounded by clipping. The corresponding
central/original change in DMB/halofit is 0.005117 at `k <= 1e-2 h/Mpc` but 0.232510 over
the full k grid, so the unqualified null claim is false. Repeating the calculation with
backreaction disabled gives exact zero old/central differences in CLM density, Pgg2h, and
Pmm, which is the appropriate branch null.

The isolated central operator passes float64 CPU `jit`, nested `(4,5,128)` `vmap`, and
reverse-mode differentiation under JAX/jaxlib 0.5.0. No GPU was present. End-to-end tracing
fails because the TSV grid warning calls `float(jnp.max(self.r_ej_mat))` at
`get_radial_profiles.py:188-195`; Stage-31 samples `theta_ej_0` and related exponents. The
Stage-31 builder still imports flat modules at
`notebooks/xDESI/survey_measure/godmax_multiprobe_hmc_stage31.py:948-959`, while a clean
`PYTHONPATH=TSV/src` exposes only `godmax.*` modules.

The full commands and unrounded outputs are recorded in
`knowledge/.kb/ledgers/2026-08-16-tsv-mass-conservation-audit.md` and in the freshly executed
notebook cells. The notebook also documents independent TSV changes that must not be
attributed to the derivative, including the extra `ln(10)` at TSV radial-profile line 855,
new tSZ/HSE behavior, `split_cga_fftlog`, radial cutoffs, and removal of the original
unit-consistent map projector.

The follow-up shell notebook uses the same TSV `Mclm`, HOD, HMF, and grids for every
method. On the reduced `nr=23,nz=2,nM=6,nk=32,r=0.005--48 Mpc/h` grid, shell cumulative
node and explicit-zero-mode errors are both `2.22e-16`. At the first finite k, raw shell
`u_clm` spans `0.999999573--0.99999999999`, whereas raw original and TSV FFTLog span
`0.841--1.202` and `1.011--1.326`. The raw shell two-halo Pgg ratio differs from unity by
`6.14e-7` at `k<=1e-2 h/Mpc` and `5.95e-5` at `k<0.1 h/Mpc`; its total-Pgg differences are
`6.52e-7` and `7.34e-5`. Repeating at `nr=47` gives `5.52e-7` for the narrower low-k
two-halo comparison. The full-k `nr=23` versus `47` shell window still differs by `0.0140`,
so the coarse grid is established only for the large-scale question.

The same run exposes the physical limits: 14 outer increments are negative, the minimum
increment is `-1.16e-6` of endpoint mass, and endpoint/(fclm*Mtot) spans
`0.9767--1.0015` even with `rmax=48 Mpc/h`. The analytic inner solution conserves the
represented `M(r0)` but cannot recover mass omitted by the finite lower integration bound
in `get_Mnfw`, nor can endpoint normalization repair outer undercoverage. These failures
remain visible; no clip, sorting, monotonic projection, or renormalization is applied.

On JAX 0.5.0 CPU/x64, the isolated end-to-end target-grid shell transform reports 3,072
temporary bytes versus 12,608 and 13,440 for original+FFTLog and TSV+FFTLog. Its reverse
gradient agrees with a parameter finite difference to `2.55e-10` relative error, and its
lowered HLO has no host callback. This supports the graph design but is not a GPU or full
likelihood benchmark. Exact commands and fresh outputs are in
`knowledge/.kb/ledgers/2026-08-16-tsv-shell-clm-comparison.md`.

Fresh source-integration evidence is recorded in
`knowledge/.kb/ledgers/2026-08-16-clm-shell-source-integration.md` and its independent S5
reproduction. On the matched reduced source grid the explicit zero-mode error is `0.0`, and
the backreaction/no-backreaction low-k deviations are `1.20e-7` for `Pgg_2h` and `1.15e-7`
for total `Pgg`. Seventeen registered direct/legacy null arrays—including `rho_clm`,
`rho_dmb`, every DMB/NFW matter spectrum, halofit, HMF, bias, and HOD—are bitwise identical.
The full reduced CPU/x64 `Pgg` objective has finite, nonzero gradients in all three varied
ejection parameters at fiducial and both registered corners, with no host callback in HLO.
The actual default `rmax=8 Mpc/h` grid has no negative shell increments in this run, but its
minimum endpoint/(fclm*Mtot) is only `0.777810570`; exact `u(0)` therefore remains a
represented-profile statement, not physical outer-mass closure.

The second source-validation lap compares actual CLM profiles on that default radial range
against an `nr=95` direct reference. At `k<=0.1 h/Mpc`, coarse `nr=23` maximum absolute
errors are `8.43e-4` for direct shells and `1.272e-1` for legacy FFTLog. Full-range direct
movement decreases from `1.144e-2` (23 to 47 nodes) to `2.612e-3` (47 to 95). A CPU/x64
NumPyro smoke completes eight warmup plus eight NUTS samples with all three ejection
parameters moving, finite samples, and zero divergences. All current galaxy-dependent 3D
spectra and projected galaxy Cl families are finite; projected kappa-kappa and kappa-y are
bitwise direct/legacy nulls. These checks improve the source-default case while leaving GPU
runtime and physical outer coverage explicitly open.

The executed source-integration notebook
`notebooks/debug_tsv/compare_clm_source_power_ratios.ipynb` constructs the matched four-way
matrix `(direct_shell|legacy_fftlog) x (backreaction off|on)` and stores two diagnostic
figures: raw `u_clm` plus `Pmm_hydro/Pmm_DMO` and `Pgg_hydro/Pgg_no-backreaction`, followed
by direct/legacy residuals. On its disclosed coarse `r=0.005--48 Mpc/h`, `nz=4`, `nM=6`,
`nk=64` grid, all twelve registered matter/profile/HMF/HOD nulls are exact, the matter-ratio
method residual is exactly zero, and the maximum low-k total-Pgg deviation falls from
`0.135073` in legacy mode to `1.5914e-7` in direct mode. The notebook reports ten tiny
negative extended-grid increments and endpoint coverage separately rather than treating
the exact shell zero mode as physical closure. Its separate source-default `rmax=8` check
records zero negative increments but minimum endpoint coverage `0.77780998`.

## How to verify

```bash
# Execute all cells in the ili-sbi environment; expected exit code is zero.
/usr/bin/env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/godmax-mpl \
  IPYTHONDIR=/tmp/godmax-ipython JUPYTER_CONFIG_DIR=/tmp/godmax-jupyter \
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 \
  --ExecutePreprocessor.kernel_name=ili-sbi \
  notebooks/debug_tsv/tsv_vs_original_mass_conservation.ipynb

# Every code cell executed and no uncaught error output was stored: true.
jq -e '([.cells[] | select(.cell_type=="code") | .execution_count] | all(. != null)) and ([.cells[].outputs[]? | select(.output_type=="error")] | length == 0)' \
  notebooks/debug_tsv/tsv_vs_original_mass_conservation.ipynb

# Execute and validate the shell follow-up; expected exit code zero and jq output true.
/usr/bin/env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/godmax-mpl \
  IPYTHONDIR=/tmp/godmax-ipython JUPYTER_CONFIG_DIR=/tmp/godmax-jupyter \
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=ili-sbi \
  notebooks/debug_tsv/compare_clm_shell_fourier.ipynb

jq -e '([.cells[] | select(.cell_type=="code") | .execution_count] | all(. != null)) and ([.cells[].outputs[]? | select(.output_type=="error")] | length == 0)' \
  notebooks/debug_tsv/compare_clm_shell_fourier.ipynb

# Re-execute and validate the source-integrated power-ratio comparison.
/usr/bin/env JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/godmax-mpl \
  IPYTHONDIR=/tmp/godmax-ipython JUPYTER_CONFIG_DIR=/tmp/godmax-jupyter \
  /mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=ili-sbi \
  notebooks/debug_tsv/compare_clm_source_power_ratios.ipynb

jq -e '([.cells[] | select(.cell_type=="code") | .execution_count] | all(. != null)) and ([.cells[].outputs[]? | select(.output_type=="error")] | length == 0)' \
  notebooks/debug_tsv/compare_clm_source_power_ratios.ipynb
```

## Failure modes

- Comparing branch defaults attributes changes from `theta_ej_0`, halo-mass range,
  `split_cga_fftlog`, or tSZ/HSE to the derivative.
- Treating clipped `u_clm <= 1` as raw mass closure hides the observed central-difference
  overshoot and can make the final Pgg ratio look physically validated.
- Calling the low-k DMB/halofit null an all-k null misses a 23% nonlinear-scale response in
  the reduced matched test.
- Launching Stage-31 HMC before moving the warning out of traced construction raises
  `ConcretizationTypeError`; after that, stale imports raise `ModuleNotFoundError`.
- Running an allowed JAX 0.4.28 installation raises `NotImplementedError` for the
  coordinate-array `jnp.gradient`; use scalar log-grid spacing or raise the dependency floor.
- Treating exact shell `u_clm(0)=1` as proof of the physical mass budget can hide an
  incomplete outer profile; endpoint/(fclm*Mtot), negative increments, and boundary
  convergence are separate mandatory checks.
- Omitting the inner mass gives `u(0)=1-M(r0)/M(rmax)`; treating it as a point preserves
  only the monopole. The fixed-p=2 window encodes the NFW cusp, while the fitted-p variant
  is diagnostic and must never be clipped inside HMC.

## Open questions

- Actual GPU compilation, memory use, and NUTS throughput were not tested because this host
  exposed only a CPU device. This blocks the GPU-performance claim, not the operator-level
  JAX result.
- The finite-volume shell/direct-transform route is now source-integrated for galaxy
  `u_clm` and Pgg. Migrating DMB, HSE, inverse-CDF sampling, and maps remains open and must
  use component-specific inner behavior rather than applying the CLM p=2 cell wholesale.
- The tSZ/HSE, split-CGA FFTLog, Cl/Pyy relocation, and map-projection changes need separate
  physical and API validation; none is established by this CLM audit.
