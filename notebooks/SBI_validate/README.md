# SBI_validate — mock-SBI vs theory-SBI vs theory-HMC

Seven stages, one folder each, each with a `00_overview.ipynb` that says what the stage does
and shows its essential plots. `10_mock_sbi_validation_figures.ipynb` at the top level is the
summary across all of them.

| folder | what it produces |
|---|---|
| `01_catalog_and_theory_inputs/` | lens n(z), HMF and bias from the Abacus `c0000_ph000` catalog |
| `02_frozen_contract/` | mask, bandpower windows, per-probe noise, fixed galaxy alm, and the 42x42 NaMaster covariance every method whitens with |
| `03_theory_forward_model/` | the differentiable JAX halo model, its resolved-P(k) and projected-operator validations, the pinned reference point |
| `04_theory_hmc_and_sbi/` | theory HMC (NUTS) and theory SBI (NPE) on the noiseless self-consistent observation, and their comparison |
| `05_pasting/` | the 128-point design, the Backlight pastes, the measured 42-vectors, the noise-augmented training set |
| `06_paste_vs_theory/` | the null test: do pasted maps reproduce the theory Cls at nside 1024 through ell 2048? |
| `07_mock_sbi_snle/` | NLE on the pasted responses, and the three-way posterior comparison |
| `common/` | modules imported by more than one stage |
| `arxiv/` | superseded and diagnostic work, kept for history |

## Layout notes

`common/` holds every module that more than one stage imports. Scripts inside a stage folder
carry a five-line bootstrap that puts the repository's `SBI_validate/` and `common/` on
`sys.path`, so they run directly:

```bash
python 07_mock_sbi_snle/run_mock_sbi_snle.py --help
```

Two things to know if you move files again:

* Modules anchor paths with `Path(__file__).resolve().parents[N]`. Changing a file's depth
  changes what `N` must be, and nothing raises — it just silently points at the wrong
  directory. There are 31 such sites; a static check that every one resolves to the
  repository root is the cheapest way to catch a mistake.
* `plot_three_way_triangle.py` and `plot_mock_sbi_training_spread.py` call
  `matplotlib.use("Agg")` at import. Importing them from a notebook switches the backend and
  every figure created afterwards silently stops displaying. Run them as scripts; do not
  import them into a notebook.
