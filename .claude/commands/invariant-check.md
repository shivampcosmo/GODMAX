---
description: Run the invariant registry and argue every manual invariant in scope
argument-hint: "[layer | severity | INV-ID | file path]"
allowed-tools: Bash, Read, Grep, Glob, Task
---

Run the invariant registry and, critically, **argue the manual ones** — an automated pass
covers only part of the registry.

## 1. Automated

```bash
python tools/kb/kb.py invariants --lint
python tools/kb/kb.py invariants --check ${1:+--id $1}
```

If `$1` looks like a layer (`physics`, `numerics`, `measurement`, `inference`, `data`,
`process`) use `--layer $1`. If it is a severity, use `--severity $1`. If it is a file path,
first run `python tools/kb/kb.py which $1` and check the invariants it returns.

## 2. Manual — this is the part that matters

Every `MANUAL` result is a blocker or high-severity rule enforced only by an agent
reasoning about it. For each one in scope, produce:

```text
INV-XXX-NN   HOLDS because <specific reason>   anchor: path/file.py:123
             evidence: <command + the number it produced>
```

Route each to its owner rather than guessing:

- `INV-PHYS-*` → **halo-model-physicist** (mass budget, bias normalisation, 1h/2h
  transition, units and h)
- `INV-JAX-*` → **jax-numerics** (x64, gradient finiteness, tracing, seeds)
- `INV-NMT-*`, `INV-KSZ-*`, `INV-SHEAR-*`, `INV-BEAM-*`, `INV-SHOTNOISE-*` →
  **measurement-namaster**
- `INV-WHITEN-*`, `INV-CHI2-*`, `INV-MCMC-*`, `INV-PRIOR-*`, `INV-HOD-ARRAY0-*` →
  **inference-statistician**
- `INV-WINDOW-*`, `INV-NZ-*`, `INV-DV-*`, `INV-HOD-PZBIN-*` → **xdesi-lead**
- `INV-ABACUS-*` → **abacus-paste-validator**
- `INV-PROC-*` → **physics-referee**

A manual invariant that nobody can argue with an anchor is **not holding**. Report it as
such — do not record it as passing because no evidence contradicts it.

## 3. Report

A table: invariant, severity, PASS/FAIL/MANUAL-HOLDS/MANUAL-UNARGUED/SKIP, and the anchor
or command. Then, separately: which manual invariants are the best candidates for
conversion into pytest cases in `tests/`, and what numeric criterion each would need.
Converting a manual blocker into an automated test is the highest-leverage improvement
available to this system — name the top two.
