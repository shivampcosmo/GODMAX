---
id: kb.validation.evidence-ledger
title: Evidence ledger template
layer: 70-validation
owner: repro-runner
status: verified
confidence: high
scope:
  - tools/kb/kb.py
invariants: [INV-PROC-EVIDENCE-01]
checks:
  - python tools/kb/kb.py invariants --check --id INV-PROC-EVIDENCE-01
verified_at_commit: a3b3f96
verified_on: 2026-08-16
see_also: [kb.validation.loop]
scope_digest: sha256:de534b2779d372d815fa2307df26c208
---

## Claim

Every validated change produces exactly one evidence ledger at
`knowledge/.kb/ledgers/YYYY-MM-DD-<slug>.md`, following the template below. `kb verify`
refuses to re-stamp a knowledge document without one.

## Why it is true

`tools/kb/kb.py verify` requires `--evidence <path>` or `--bootstrap`; the requirement is
enforced in the `cmd_verify` function. The directory `knowledge/.kb/` is gitignored
(`.gitignore`), so ledgers are local working evidence; their durable summary lives in
`knowledge/90-journal/`.

## How to verify

```bash
python tools/kb/kb.py verify --doc kb.validation.loop        # must fail: no --evidence
python tools/kb/kb.py ledger new probe-test                  # scaffolds a ledger
```

## Failure modes

Ledgers written after the fact from memory. The tell is paraphrased output, rounded
numbers, and absent commands. A ledger is a transcript, not a summary.

## Open questions

None.

---

# Template — copy below this line

```markdown
# Ledger: <one-line change description>

- **Date:** 2026-08-03
- **Agent:** measurement-namaster
- **Branch / commit at start:** ltuSP @ 43e07ca
- **Task (S0):** <observable outcome, not "fix the code">
- **Scope may change:** notebooks/xDESI/survey_measure/multiprobe_namaster.py
- **kb which output:**
  ```
  <paste the actual output of `python tools/kb/kb.py which <paths>`>
  ```

## S1 — Locate

| # | Claim | Anchor |
|---|-------|--------|
| 1 |       | `file.py:123` |

Contradictions found against knowledge documents: <none / describe>

## S2 — Pre-registered prediction

- **Direction:**
- **Magnitude:**
- **Affected families:**
- **Predicted UNCHANGED (the null control):**
- **Falsifier:** <what observation would prove this change wrong>

## S3 — Invariant self-check

```text
INV-XXX-01   HOLDS/AT-RISK/VIOLATED/N-A   <reason with file:line>
```

## S4 — Change made

```diff
<the actual diff, or `git diff --stat` plus the substantive hunks>
```

## S5 — Evidence

### Invariant checks
```bash
$ python tools/kb/kb.py invariants --check --id INV-XXX-01
<real output>
```

### Owning-document checks
```bash
$ python tools/kb/kb.py check --scope <path>
<real output>
```

### Prediction test
```bash
$ <command>
<real output>
```
**Predicted:** <from S2>  **Observed:** <actual>  **Verdict:** CONFIRMED / REFUTED

### Null control — what must NOT have changed
```bash
$ <command comparing the untouched families / spectra>
<real output>
```
**Verdict:** unchanged to <tolerance> / CHANGED (explain)

### Resolution or grid robustness (S6 q6)
```bash
$ <same test at a different nside / ell range / grid>
<real output>
```

## S6 — Refutation

Performed by: <physics-referee | self, with justification>

1. **Sign:**
2. **Units and h:**
3. **Double application:**
4. **Degeneracy:**
5. **Coincidence:**
6. **Interpolation and grid:**
7. **Goodness, not improvement:** absolute chi2 = <>, retained rank = <>, n_varied = <>,
   expected = <rank - k> ± <sqrt(2 dof)>. Acceptable: yes / no.

## S7 — Gate

```bash
$ python tools/kb/kb.py gate
<real output>
```

- [ ] all S3 invariants HOLDS/N-A with anchors
- [ ] all AT-RISK invariants have passing checks
- [ ] S2 prediction confirmed (or refuted and re-looped)
- [ ] null control clean
- [ ] `kb gate` exit 0
- [ ] refutation answered (7/7)
- [ ] every number has a command

**Verdict:** PASS / FAIL (lap N of 3) / ESCALATED

## S8 — Recorded

- Knowledge documents re-stamped: <ids>
- New documents created: <ids>
- Journal entry: <link>
- **Residual risk / what a future session should distrust:**
```
