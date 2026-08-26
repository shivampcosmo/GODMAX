# Mock SBI on pasted Abacus maps — submission order

Plan: `knowledge/60-projects/SBI_validate/mock-sbi-pasted-response-plan.md`

Every job below is submitted by hand. Nothing in this directory submits anything itself,
and no stage should be launched before the preceding gate has passed.

`REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX`
`D=$REPO/notebooks/SBI_validate/mock_sbi_sbatch`

---

## Stage 1a — foundations, guide, noise bank

These are cheap and have no dependencies on each other.

```bash
sbatch $D/01_foundations_and_guide.sbatch     # 1 A100, ~15 min
sbatch $D/00_noise_bank.sbatch                # CPU only, ~25 min
```

**Gate 1a.** `data/SBI_validate/mock_sbi/foundations.json` must be `PASS`: the archived
paste measured through `mock_sbi_common` reproduces the noise contract's bandpowers, and
`mu_paste + nu(observation seeds)` reproduces the inference contract's `data_vector`.
`transfer_and_guide.json` must be `PASS`: the transfer-corrected guide brackets the frozen
paste's own parameter point. `backend_parity.json` must be `PORTABLE` with the parity
vector sha unchanged — that is the null control proving the theta-override patch moved no
number. The noise-bank report's `mean_whitened_chi2_over_dim` must sit at 1 within
`expected_relative_precision`.

## Stage 1b — paste nulls and the one-theta benchmark

```bash
sbatch $D/03_stage1_paste_nulls.sbatch        # 1 A100, ~25 min
```

**Gate 1b.** Both reports must be `PASS` with `bitwise_identical: true` on
`map_ymap`, `map_tau`, `map_kappa_cmb`.

* Null A failing means the override machinery perturbs the maps even when set to the
  frozen values — stop and fix it.
* Null B failing means `get_galmap: false` is not safe. The campaign still runs, but
  every plan must be regenerated with `--keep-galaxy-map` and the budget re-costed at
  4.20 GPU-h per point instead of ~1.85. Do not proceed on the cheaper number until
  this test has passed.

The benchmark block prints the projected GPU-hours per theta. Use that number, not the
plan's estimate, in any cost request.

## Stage 2 — transfer-flatness scan (12 pastes, the decision gate)

```bash
cd $REPO/notebooks/SBI_validate
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
$PY mock_sbi_design.py --round flatness --count 12 --recipe defensive \
    --include-reference-point
# 'defensive' here on purpose: the flatness scan must reach into the prior, and a
# 12-draw 'concentrated' design typically contains no prior component at all.

PLAN=$REPO/data/SBI_validate/mock_sbi/paste_plan/round_flatness/paste_plan.json
PLAN=$PLAN NUM_SPLITS=32 sbatch --array=0-383%8 $D/04_paste_array.sbatch
# then, only after the array completes:
PLAN=$PLAN sbatch --array=0-12 $D/05_combine_array.sbatch
PLAN=$PLAN sbatch $D/06_measure_responses.sbatch
```

The anchor (array index 12) declares a cached map and its paste tasks exit immediately,
so the real cost is 12 theta, not 13.

**Gate 2.** From `responses.json`: `r(theta) = mu_paste/mu_theory` must vary over the
design by materially less than `mu_theory` itself, in units of `sqrt(C_bb)`. If it does
not, stop — the factorization is the whole reason 200-300 pastes is plausible, and
Stage 0 has already shown that emulating the full response does not reach the gate at
any affordable N.

## Stages 3-5 — production rounds

Same three commands as Stage 2 with `--round 1`, then `--round 2`, then `--round 3`,
each drawn from the previous round's mock posterior. Sizes come from Stage 0's
`recommended_n_train`, not from a guess. Draw the sealed holdout **before** round 1
and do not measure it until the emulator, its hyperparameters and the stopping branch
are frozen:

```bash
$PY mock_sbi_design.py --round holdout --count 24 --recipe defensive
```

## Stage 0 — the budget test (run this first of all; no pastes)

```bash
sbatch $D/02_stage0_oracle.sbatch             # 1 A100, up to 8 h
```

It needs only `transfer_and_guide.json` from Stage 1a. Its `recommended_n_train` per arm
is what sizes Stages 3-5.
