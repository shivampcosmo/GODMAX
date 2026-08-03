# The GODMAX knowledge tree

This directory is what agents read before they act, and what they must update after they act.
It exists because the expensive failure in this repository is not a crash — it is a
**plausible wrong number** produced by an agent acting on a convention it did not know about.

## Start here

| You want to | Read |
|---|---|
| understand the format | [`_schema/FORMAT.md`](_schema/FORMAT.md) |
| know the rules that must never break | [`00-invariants/invariants.yaml`](00-invariants/invariants.yaml) |
| know how any change gets validated | [`70-validation/VALIDATION_LOOP.md`](70-validation/VALIDATION_LOOP.md) |
| know who owns what | [`70-validation/AGENT_ROSTER.md`](70-validation/AGENT_ROSTER.md) |
| know how this stays fresh across push/pull | [`70-validation/GIT_SYNC.md`](70-validation/GIT_SYNC.md) |
| browse everything | [`INDEX.md`](INDEX.md) (generated) |
| know what changed and why | [`90-journal/`](90-journal/) |

## The four moving parts

**1. Documents** — Markdown with YAML frontmatter, one claim each. Each declares `scope`: the
code it describes. Fixed body sections: Claim / Why it is true / How to verify / Failure modes
/ Open questions.

**2. Invariants** — `00-invariants/invariants.yaml`. Machine-readable rules with a severity, a
`violation_symptom`, and an executable or manual `check`. `blocker` severity blocks a push.
This is the highest-leverage file in the repository.

**3. Staleness** — content-addressed. `kb` hashes each document's `scope` files; a digest
mismatch marks the document `stale` automatically. You cannot assert your way to green.
Notebook *outputs* are excluded from the hash, so re-running a notebook does not falsely
invalidate knowledge.

**4. The journal** — `90-journal/YYYY-MM.md`, append-only. Code shows *what*; the tree shows
*what is true now*; the journal shows *what changed and on what grounds*. It is the only place
`why` survives, and it is how a cluster result reaches a laptop — `data/`, `outputs/` and
`results/` are gitignored and never travel.

## Layers

```text
00-invariants/     rules that must never break
10-architecture/   class chain, API contracts, parameter flow
20-physics/        halo model, profiles, HOD, projection, units
30-numerics/       JAX, precision, gradients, performance
40-inference/      likelihood, priors, sampler, convergence, chi2
50-data-products/  on-disk products, HDF5 schemas, provenance
60-projects/       per-project state; xDESI/ is the deepest
70-validation/     the loop, evidence rules, roster, git sync
90-journal/        append-only decision log
```

**Layer rule:** a document may depend on its own layer or lower, never higher.
`60-projects/xDESI` may depend on `20-physics`; the reverse would let project churn
invalidate core knowledge.

## Everyday commands

```bash
python tools/kb/kb.py status                 # what is verified, what is stale
python tools/kb/kb.py which <file>           # who owns this code, which rules apply
python tools/kb/kb.py invariants --check     # run the executable rules
python tools/kb/kb.py gate --dry-run         # what a push would check
python tools/kb/kb.py doctor                 # is the system actually wired up
```

In Claude Code: `/kb-status`, `/kb-sync`, `/validate`, `/xdesi-status`, `/invariant-check`,
`/kb-new`.

First time on a machine: `bash tools/kb/install.sh` — `.git/hooks` is not tracked, so the
gate must be installed per clone.

## Reading a document honestly

`status` and `confidence` are load-bearing, not decoration:

- **`status: verified`** — the claim was checked against the code at the stamped digest.
- **`status: stale`** — the code moved. Treat the claim as a **hypothesis**, not a fact.
- **`status: draft`** — written but not evidence-verified.
- **`confidence: medium`** — usually means the claim came from prose (a README or handoff
  document) rather than from line-level reading of the code. Several of the seed documents are
  in this state and say so in Open questions.
- **`confidence: low`** — cannot be cited as evidence in a validation gate.

Most seed documents in this tree are `draft` / `medium` **by design**: they were extracted
from `notebooks/xDESI/survey_measure/README.md`,
`BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` and `src/context/codebase_summary.md`, which are good
sources but are prose, not checks. Promoting them to `verified` means reading the code and
producing an evidence ledger. That work is listed in each document's Open questions.

## The one thing to remember

A document that is confidently wrong is worse than no document, because it replaces a search
with a mistake. When in doubt: lower the confidence, write what you actually verified, and put
the rest in Open questions.
