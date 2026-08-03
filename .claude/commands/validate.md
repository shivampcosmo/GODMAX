---
description: Run the full validation loop over the current working changes before pushing
argument-hint: "[what you changed, in one line]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task, NotebookEdit
---

Validate the current working changes end to end, following
`knowledge/70-validation/VALIDATION_LOOP.md`.

Context: $1

## 1. Scope and routing

```bash
git status --porcelain
git diff --stat
python tools/kb/kb.py which $(git diff --name-only HEAD; git diff --cached --name-only | sort -u)
```

## 2. Reconstruct the loop for changes already made

The change exists, so S2 (pre-registered prediction) cannot be done honestly after the
fact. Instead, state explicitly:

- what the change was **intended** to do — direction, magnitude, affected families;
- what it was intended to leave **unchanged** (this becomes the null control);
- and note in the ledger that the prediction was reconstructed, not pre-registered. Do not
  present a reconstructed prediction as a confirmed one.

## 3. Invariant self-check (S3)

For every invariant `kb which` returned, write one line: `HOLDS` with a `file:line` reason,
`AT-RISK`, `VIOLATED`, or `N/A`. Bare `HOLDS` is rejected. Every `blocker` invariant with
`check.kind: manual` needs an explicit `HOLDS because <file:line>` argument.

## 4. Evidence (S5) — delegate to **repro-runner**

```bash
python tools/kb/kb.py ledger new <slug>
python tools/kb/kb.py invariants --check
python tools/kb/kb.py check --scope <changed dirs>
pytest tests/ -q
```

Required in the ledger: the environment block, the checks, the intended-effect test, **the
null control**, and one resolution/grid variation.

## 5. Refutation (S6) — delegate to **physics-referee**

Mandatory for anything touching physics, conventions, statistics, or a number that will be
quoted. Answer all seven refutation questions in writing.

## 6. Gate (S7)

```bash
python tools/kb/kb.py gate --dry-run
```

## 7. Record (S8)

```bash
python tools/kb/kb.py verify --doc <id> --evidence knowledge/.kb/ledgers/<ledger>.md
python tools/kb/kb.py journal "<what changed and why>" --agent <agent> \
  --invariants INV-… --evidence knowledge/.kb/ledgers/<ledger>.md
```

## Report

State **PASS**, **FAIL (lap N)**, or **ESCALATED**, then: the invariant table, the evidence
with real commands and output, the refutation verdict, and what a future session should
still distrust. If anything failed, say so plainly with the output — do not summarise a
failure as a caveat.
