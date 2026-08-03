---
id: kb.arch.params-dicts
title: The four parameter dicts and the deep-merge config convention
layer: 10-architecture
owner: godmax-core
status: draft
confidence: medium
scope:
  - param_files/params_default.yaml
  - param_files/xDESI/params_multiprobe_fast1024_true_nz_theory.yaml
  - param_files/xDESI/priors_multiprobe_fast1024_hmc_stage31.yaml
invariants:
  - INV-PHYS-UNITS-01
checks:
  - "TODO(godmax-core): assert every key used by src/ exists in params_default.yaml"
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.arch.class-chain, kb.xdesi.analysis-state]
scope_digest: sha256:75a98b9480d6510908ac79d83ac18d36
---

## Claim

Configuration is four dicts — `sim_params`, `halo_params`, `analysis`, `other_params` —
produced by **deep-merging** `param_files/params_default.yaml` with a project override via
`deepmerge.always_merger`. All four are threaded through every layer of the class chain.
Because the merge is deep, a partial override leaves unrelated defaults in place.

## Why it is true

The loading pattern is documented in `README.md:110-113`:

```python
default_data = read_yaml(abs_path_params + '/params_default.yaml')
new_data     = read_yaml(abs_path_params + '/Pge/params.yaml')
merged_data  = always_merger.merge(default_data, new_data)
```

`deepmerge` is a declared dependency (`pyproject.toml`). The four categories and their
contents are documented in `README.md:123-142`:

| dict | holds |
|---|---|
| `sim_params_dict` | cosmology (H0, Om0, sigma8, …), BCM gas, stellar/SHMR, HOD, non-thermal pressure |
| `halo_params_dict` | mass and redshift grid limits and node counts |
| `analysis_dict` | which observables, model choices, accuracy settings |
| `other_params_dict` | intrinsic alignment, photo-z bias, systematics |

`param_files/xDESI/` holds 15+ variants of the same base config, distinguished by encoded
filenames (`fast1024`, `midres2048`, `true_nz`, `abacus_cosmo`, `simple1h2h`, `lmax3000`,
`64param`, `warm100_2000`). Deep merge is what makes that tractable: each variant states
only its differences.

Some `xDESI` configs are **path-backed** — they merge `params_default.yaml` with
`param_files/xDESI/params_fit_abacus.yaml` and then apply fixed overrides
(`BACKLIGHT_PASTE_HANDOFF_SUMMARY.md`, "Theory Configuration").

Recorded xDESI settings that are physics choices rather than implementation details, per the
same source:

```yaml
analysis.hod_params_model: perbin
analysis.gg_transition_model: poweradd
analysis.beam_fwhm_arcmin: 0.0     # beams applied by the measurement wrapper, not here
analysis.zmin_for_Cls: 0.005
analysis.zmax_for_Cls: 3.0
analysis.nz_for_Cls: 192
halo_params.zmin: 0.005
halo_params.zmax: 3.0
halo_params.nz: 96
```

Fixed cosmology for the Stage-31 comparisons and HMC: `H0 67.36`, `Om0 0.30`, `Ob0 0.0493`,
`sigma8 0.80`, `ns 0.9649`, `w0 -1.0`, flat.

Note `analysis.beam_fwhm_arcmin: 0.0`: the ACT beams, pixel windows, masks, transfer
functions, shear sign, m-bias and kSZ conversion are applied by the measurement/theory
wrapper, **not** by smooth GODMAX curves (`INV-BEAM-01`, `INV-WINDOW-CMP-01`).

## How to verify

```bash
# the merge convention in real consumers
grep -rn "always_merger" --include=*.py --include=*.ipynb . | grep -v src/arxiv

# the config variant family
ls param_files/xDESI/

# what a given config actually resolves to
python - <<'EOF'
import yaml
from deepmerge import always_merger
d = yaml.safe_load(open('param_files/params_default.yaml'))
o = yaml.safe_load(open('param_files/xDESI/params_multiprobe_fast1024_true_nz_theory.yaml'))
m = always_merger.merge(d, o)
for k in ('sim_params','halo_params','analysis','other_params'):
    print(k, sorted((m.get(k) or {}).keys()))
EOF
```

## Failure modes

- **A new key added to a project config but not to `params_default.yaml`.** Every *other*
  config then raises `KeyError` at a random depth in the chain, far from the config.
- **A silently inherited default.** Deep merge's strength is also its hazard: an override
  that omits a key inherits the default, so two configs can differ in a parameter neither
  file mentions. Symptom: two runs disagree with no visible config difference.
- **Assuming `analysis.beam_fwhm_arcmin: 0.0` means "no beam".** The beam is applied by the
  measurement wrapper. Setting a nonzero value here applies it twice — a monotonic high-ell
  deficit confined to the ACT y and T families.
- **Treating `halo_params.nz` or `nz_for_Cls` as a performance knob.** They set the
  integration grid; changing them changes results. Coordinate with
  `halo-model-physicist`.

## Open questions

- The full key inventory of `params_default.yaml` has not been cross-checked against what
  `src/` actually reads. A test asserting that every key consumed by `src/` exists in the
  default file would catch the dominant failure mode above. Owner: `godmax-core`.
- Which `xDESI` config is currently canonical for production is recorded in
  `kb.xdesi.analysis-state`, not here; this document describes the mechanism only.
