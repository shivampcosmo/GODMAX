---
id: kb.validation.agent-roster
title: Agent roster, ownership boundaries and routing
layer: 70-validation
owner: kb-curator
status: verified
confidence: high
scope:
  - .claude/agents/
  - .claude/commands/
invariants: [INV-PROC-EVIDENCE-01, INV-PROC-KB-FRESH-01]
checks:
  - python tools/kb/kb.py doctor --check-agents
verified_at_commit: 43e07ca
verified_on: 2026-08-03
see_also: [kb.validation.loop]
scope_digest: sha256:35cba77dc0b654fd37b3faa1f84c7546
---

## Claim

Ten agents divide the codebase along **failure-mode boundaries**, not directory
boundaries. Each owns a disjoint set of knowledge documents and invariants, so exactly one
agent is accountable for any given wrong number. `physics-referee` owns no code and exists
only to refute.

## Why it is true

Ownership is declared in each document's `owner` field and each invariant's `owner` field;
`python tools/kb/kb.py doctor --check-agents` fails if any `owner` names an agent absent
from `.claude/agents/`, or if any agent owns nothing.

---

## The roster

| Agent | Owns the failure mode | Primary scope |
|---|---|---|
| `kb-curator` | knowledge rot, unowned code, broken sync | `knowledge/`, `tools/kb/`, `.claude/` |
| `godmax-core` | broken API contracts across the class chain | `src/base_class.py`, `src/get_*.py` interfaces |
| `halo-model-physicist` | physically wrong model | profiles, BCM, HMF, HOD, P(k), Limber |
| `jax-numerics` | silently wrong arithmetic | x64, tracing, gradients, JIT, precision |
| `measurement-namaster` | wrong estimator or covariance | `multiprobe_namaster.py`, masks, bandpowers |
| `inference-statistician` | wrong statistical conclusion | likelihood, priors, NUTS, chi2, convergence |
| `xdesi-lead` | inconsistency across the xDESI analysis | all of `notebooks/xDESI/` |
| `abacus-paste-validator` | map-level pipeline errors | `notebooks/xDESI/abacus_paste/`, pasting |
| `physics-referee` | **false confidence** | nothing — refutes everything |
| `repro-runner` | unreproducible evidence | executes checks, writes ledgers |

### Why these boundaries

The split is by *how a result goes wrong*, because that is how you know whom to ask:

- A **sign error** in shear × scalar is `measurement-namaster` — the convention lives in
  field construction, not in the physics model.
- A **wrong gas amplitude** from a good fit is `halo-model-physicist`.
- A **good fit that is not good enough** is `inference-statistician`
  (`INV-CHI2-HONEST-01`).
- A **zero gradient** is `jax-numerics`, even if the symptom appears in the posterior.
- **Two xDESI stages disagreeing** is `xdesi-lead`, because no single-file owner can see
  it.

Directory-based ownership would put `multiprobe_namaster.py` (157 KB, estimator +
covariance + n(z) + kSZ + priors) under one agent, which is precisely the file where four
different failure modes live.

---

## Routing

Mechanical, from the knowledge tree:

```bash
python tools/kb/kb.py which notebooks/xDESI/survey_measure/multiprobe_namaster.py
# -> documents, invariants, owner agent
```

Escalation ladder for cross-cutting work:

```text
xdesi-lead  ──delegates──>  measurement-namaster
                            inference-statistician     ──evidence──> repro-runner
                            halo-model-physicist
                            abacus-paste-validator
                                    │
                                    └── all results ──> physics-referee (S6) ──> gate
```

`xdesi-lead` coordinates but does not adjudicate physics: a disputed physical claim goes
to `halo-model-physicist` for the model and `physics-referee` for the refutation.

### Who may mark a document `verified`

Only the document's `owner`, and only with an evidence ledger. `physics-referee` may
*revoke* verification on any document (set `status: draft`, `confidence: low`) without
owning it — the asymmetry is deliberate: it should be easy to withdraw a claim and hard to
assert one.

## How to verify

```bash
python tools/kb/kb.py doctor --check-agents
ls .claude/agents/*.md | wc -l                        # 10
grep -c "VALIDATION_LOOP" .claude/agents/*.md          # every file >= 1
```

## Failure modes

- **Two agents editing one file in one session.** Each sees half the change; the S5 null
  control looks clean to both. Serialise, or give the work to `xdesi-lead`.
- **An agent self-refuting a physics change.** Defeats S6. The author's disposition cannot
  supply adversarial pressure.
- **An agent marking another's document verified.** Breaks accountability; blocked by
  `kb verify --doc` owner check.
- **Orphaned invariants.** An invariant whose `owner` no longer exists is never checked.
  `doctor --check-agents` catches it.

## Open questions

- `repro-runner` and `physics-referee` overlap when a refutation needs a long cluster run.
  Current convention: `physics-referee` specifies the refutation test, `repro-runner`
  executes it. Owner: `kb-curator`.
