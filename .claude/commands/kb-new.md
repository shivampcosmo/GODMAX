---
description: Create a knowledge document for code that has no owner yet
argument-hint: "<file or topic to document>"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task
---

Create a knowledge document for: $ARGUMENTS

## 1. Confirm it is actually unowned

```bash
python tools/kb/kb.py which $ARGUMENTS
```

If a document already owns it, **extend that one** instead — a second document with
overlapping `scope` means two sources of truth and two staleness signals for one claim.

## 2. Decide layer and owner

| layer | for |
|---|---|
| `10-architecture` | class chain, API contracts, parameter flow → `godmax-core` |
| `20-physics` | model correctness, profiles, HOD, projection → `halo-model-physicist` |
| `30-numerics` | JAX, precision, gradients, performance → `jax-numerics` |
| `40-inference` | likelihood, priors, sampler, chi2 → `inference-statistician` |
| `50-data-products` | on-disk products, HDF5 schemas, estimator → `measurement-namaster` |
| `60-projects/xDESI` | analysis state and cross-stage decisions → `xdesi-lead` |

Layer rule: a document may depend on its own layer or lower, never higher.

## 3. Scaffold

```bash
python tools/kb/kb.py new --id kb.<layer>.<slug> --title "<title>" \
  --layer <dir> --owner <agent> --scope <precise file paths>
```

`scope` must be **precise file paths**, not a directory tree. It is both the staleness
trigger and the routing key; an over-broad scope makes the document permanently stale,
therefore ignored.

## 4. Fill it in by reading the code, not by remembering

Follow `knowledge/_schema/FORMAT.md`. Fixed sections:

- **`## Claim`** — one ground truth, flatly stated. If it needs "also", split the file.
- **`## Why it is true`** — every sentence anchored to `file.py:123` or a command plus its
  real output. Hedged words ("should", "presumably", "appears to") belong in Open questions.
- **`## How to verify`** — copy-pasteable commands with the expected result stated
  numerically.
- **`## Failure modes`** — the *observed symptom* when this is violated: the wrong number,
  the shape mismatch, the sign flip. This is the section that saves the most time later.
- **`## Open questions`** — with owner and whether it blocks. "None." is valid.

## 5. Register any new rule, then verify

If the document reveals a convention that must never break, add it to
`knowledge/00-invariants/invariants.yaml` with a falsifiable `statement`, a
`violation_symptom` an engineer could recognise, and a `check`. Then:

```bash
python tools/kb/kb.py invariants --lint
python tools/kb/kb.py ledger new <slug>          # record the evidence you gathered
python tools/kb/kb.py verify --doc kb.<layer>.<slug> --evidence knowledge/.kb/ledgers/<f>.md
python tools/kb/kb.py index
```

Set `confidence` honestly: `medium` on a precise statement beats `high` on a vague one, and
`low` blocks the document from being cited as gate evidence.
