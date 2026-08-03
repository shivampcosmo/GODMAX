# Knowledge-tree format specification

This file is the contract between the knowledge tree, `tools/kb/kb.py`, and every agent
in `.claude/agents/`. If you change the schema here, update `kb.py` in the same commit.

---

## 1. Why this format

Three requirements drove it:

1. **Machine-checkable staleness.** Every document declares the *code it describes*
   (`scope`). `kb` hashes those files; if the hash moves, the document is automatically
   `stale`. Knowledge cannot silently rot.
2. **Claims must be falsifiable.** Every document declares `checks` — commands that
   prove it is still true — and every physics/convention rule is registered as an
   *invariant* with an executable check in `00-invariants/invariants.yaml`.
3. **Agent routing.** `kb which <file>` maps any source file to the documents,
   invariants, and owning agent that govern it. Agents load only their slice, not the
   whole tree.

The tree is **Markdown + YAML frontmatter**, not a database, for a blunt reason: it must
diff, merge, and review inside the same git workflow as the code it describes. A binary
or single-file store would conflict on every parallel branch.

---

## 2. Layout

```text
knowledge/
├── INDEX.md                 # generated map — do not hand-edit the table
├── _schema/                 # this spec + templates
├── 00-invariants/           # invariants.yaml: the rules that must never break
├── 10-architecture/         # class hierarchy, API contracts, data flow
├── 20-physics/              # halo model, profiles, HOD, projection, units
├── 30-numerics/             # JAX, differentiability, precision, performance
├── 40-inference/            # likelihood, priors, HMC, convergence, chi2
├── 50-data-products/        # on-disk products, HDF5 schemas, provenance
├── 60-projects/             # per-project state; xDESI/ is the deepest
│   └── xDESI/
├── 70-validation/           # the validation loop, evidence rules, agent roster
└── 90-journal/              # append-only decision log, one file per month
```

**Layer rule:** a document may cite invariants from any layer, but may only *depend* on
documents at its own layer or lower. `60-projects/xDESI` may depend on `20-physics`;
`20-physics` may never depend on `60-projects`. This keeps project churn from
invalidating core knowledge.

**One claim per file.** If a file needs the word "also" in its `## Claim`, split it.
Small files mean precise staleness: a change to `get_Cls.py` should not invalidate
everything anyone ever wrote about the halo model.

---

## 3. Frontmatter schema

Required fields are marked ✱. Unknown fields are preserved but ignored.

```yaml
---
id: kb.measurement.namaster-conventions   # ✱ stable, unique, dot-separated, never reused
title: NaMaster measurement conventions   # ✱ human-readable
layer: 50-data-products                   # ✱ must match the containing directory
owner: measurement-namaster               # ✱ agent name from .claude/agents/
status: verified                          # ✱ verified | draft | stale | deprecated
confidence: high                          # ✱ high | medium | low
scope:                                    # ✱ repo-relative paths or globs this doc describes
  - notebooks/xDESI/survey_measure/multiprobe_namaster.py
  - tests/test_xdesi_multiprobe_namaster.py
invariants:                               # invariant IDs this doc explains or relies on
  - INV-NMT-COUPLED-01
  - INV-SHEAR-SIGN-01
checks:                                   # commands that prove this doc still true
  - pytest tests/test_xdesi_multiprobe_namaster.py -q -k covariance
verified_at_commit: 43e07ca               # informational: commit at last verification
verified_on: 2026-08-03                   # ✱ ISO date
scope_digest: sha256:ab12…                # managed by kb — do not hand-edit
see_also: [kb.physics.ksz-estimator]
supersedes: []
---
```

### Field semantics that matter

- **`scope`** is the staleness trigger and the routing key. Be precise. Listing `src/`
  makes the document permanently stale and useless for routing. Listing nothing makes it
  unverifiable — `kb` reports `scope: empty` as an error.
- **`status`** is *asserted* by a human or agent; `stale` is *computed* by `kb`. If the
  digest moved, `kb` reports stale regardless of what the file says. You cannot lie your
  way to green.
- **`confidence`** is about the claim, not the prose. `medium` and `low` are respectable
  and preferred over a confident wrong statement. `low` confidence blocks a document from
  being cited as evidence in a validation gate.
- **`checks`** must be runnable from the repo root, non-interactive, and fast (< 60 s) or
  marked `[slow]`. A check that needs cluster data must be written as
  `[needs-data] <command>` so `kb` can skip rather than fail it.
- **`owner`** is a single agent. Shared ownership means nobody re-verifies it.
- **`scope` must not contain another knowledge `.md` document.** Staleness would become
  circular: re-stamping document B rewrites B's frontmatter, which changes B's hash, which
  stales document A that scoped it — forever, on every verification pass. Scope the *code*
  that both documents describe, and use `see_also` to express the relationship between
  documents. (`invariants.yaml` is safe to scope: `kb` never rewrites it.)

---

## 4. Body schema

Fixed headings, in this order. Agents parse them; do not rename them.

```markdown
## Claim
The single ground truth, stated flatly. No hedging, no history.

## Why it is true
Evidence. Every sentence anchored to `path/to/file.py:123` or a command + its output.
"I believe" / "should" / "presumably" are not evidence — move those to Open questions.

## How to verify
Exact, copy-pasteable commands. State the expected result numerically.

## Failure modes
What breaks if this is violated, and the *observed symptom* — the wrong number, the
shape mismatch, the sign flip. This is the section that saves the most time later.

## Open questions
Known unknowns, with owner and blocking status. Empty is fine; omitting it is not.
```

---

## 5. Invariant registry

`00-invariants/invariants.yaml` is the single machine-readable list of rules that must
never break. Schema per entry:

```yaml
- id: INV-NMT-COUPLED-01        # ✱ INV-<AREA>-<SLUG>-<NN>, immutable once published
  title: Bandpower covariance must be computed decoupled
  layer: measurement            # physics|numerics|measurement|inference|data|process
  severity: blocker             # blocker | high | medium
  statement: |                  # ✱ the rule, imperative, testable
    Saved bandpower covariance must come from
    nmt.gaussian_covariance(..., coupled=False).
  rationale: |                  # ✱ why — the physics or API reason
    With coupled=True this NaMaster version returns full coupled-ell pseudo-spectrum
    covariance, not bandpower covariance, so the saved matrix would not match the
    saved data vector.
  scope: [notebooks/xDESI/survey_measure/multiprobe_namaster.py]
  evidence: [notebooks/xDESI/survey_measure/README.md]
  check:
    kind: pytest                # pytest | command | grep | manual
    run: pytest tests/test_xdesi_multiprobe_namaster.py -q -k covariance
  violation_symptom: |
    Covariance dimension equals n_ell rather than n_band; chi2 is wildly wrong.
  owner: measurement-namaster
```

**Severity drives the gate:**

| severity  | `kb gate` behaviour on failure         |
|-----------|----------------------------------------|
| `blocker` | push is **blocked**                    |
| `high`    | push blocked unless `--allow-high`     |
| `medium`  | warning, recorded in the journal       |

`kind: manual` invariants cannot be auto-checked. They must be walked explicitly in the
validation loop's self-check step and the reasoning recorded in the evidence ledger.
An invariant with `kind: manual` and `severity: blocker` requires a written
`HOLDS because …` line citing `file:line` before any gate can pass.

---

## 6. Lifecycle

```text
draft ──(evidence + check passes)──> verified ──(scope digest moves)──> stale
                                        ▲                                │
                                        └────(re-verify with evidence)───┘
                                                                         │
                                                    (claim no longer true)│
                                                                         ▼
                                                                    deprecated
```

Rules:

- **Never delete a document.** Set `status: deprecated`, keep the `id`, and add
  `see_also` pointing at the replacement. Deleted knowledge gets re-derived wrongly.
- **Re-verification requires evidence.** `kb verify --doc <id>` refuses to re-stamp
  unless you pass `--evidence <ledger-path>` or `--bootstrap` (initial import only).
- **`id` is immutable.** Renaming a file is fine; changing its `id` orphans every
  reference to it.

---

## 7. Journal

`90-journal/YYYY-MM.md`, append-only, newest entry at the top of the month's file.
One entry per decision, per validated change, and per gate bypass.

```markdown
### 2026-08-03T14:22Z — stage31 v2 HMC: max_tree_depth raised 4 → 8
- **Agent:** inference-statistician
- **Commits:** 43e07ca
- **Scope:** notebooks/xDESI/survey_measure/godmax_multiprobe_hmc_stage31.py
- **Invariants:** INV-MCMC-TREEDEPTH-01 (was AT-RISK, now HOLDS)
- **Evidence:** knowledge/.kb/ledgers/2026-08-03-treedepth.md
- **Decision:** depth 4 saturated on 96% of transitions, biasing the posterior.
- **Result:** saturation 96% → 3%; chi2 unchanged within 0.4.
- **Follow-up:** rerun v2 chains; v1 posterior must not be quoted.
```

The journal is the only place where *why* survives. Code shows what, the tree shows
what-is-true-now, the journal shows what changed and on what grounds.

---

## 8. Hard prohibitions

These exist because each has silently corrupted a research result before:

1. **Never hand-edit `scope_digest`.** It is the only unfakeable signal in the system.
2. **Never widen a tolerance or eigenvalue cut to make a check pass.** That is a physics
   change: it needs its own document, its own invariant review, and user sign-off.
3. **Never cite a notebook's stored output as evidence.** Stored outputs are unversioned
   and routinely stale. Re-execute the cell, or mark the claim `UNVERIFIED`.
4. **Never record a number without the command that produced it.**
5. **Never mark `verified` on the basis of "code looks right".** Only a passing check,
   or an explicit `HOLDS because <file:line>` argument, counts.
