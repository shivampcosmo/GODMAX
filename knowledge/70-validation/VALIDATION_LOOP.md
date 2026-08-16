---
id: kb.validation.loop
title: The GODMAX validation loop
layer: 70-validation
owner: physics-referee
status: verified
confidence: high
scope:
  - .claude/agents/
  - knowledge/00-invariants/invariants.yaml
  - tools/kb/kb.py
invariants:
  - INV-PROC-EVIDENCE-01
  - INV-PROC-NOTOLERANCE-01
  - INV-PROC-KB-FRESH-01
checks:
  - python tools/kb/kb.py invariants --lint
verified_at_commit: a3b3f96
verified_on: 2026-08-16
see_also: [kb.validation.evidence-ledger, kb.validation.agent-roster, kb.validation.git-sync]
scope_digest: sha256:d90049686b530be7027f1bc57b308b26
---

## Claim

Every agent in `.claude/agents/` executes the same eight-state loop, S0–S8, and may not
report success without passing S7. The loop is mandatory, ordered, and its output is an
evidence ledger. An agent that cannot reach S7 escalates to the user; it never lowers the
bar to pass.

## Why it is true

The loop is written into every agent's system prompt in `.claude/agents/*.md`, gated in
software by `python tools/kb/kb.py gate` (wired into `.git/hooks/pre-push` by
`tools/kb/install.sh`), and its process rules are registered as blocker invariants
`INV-PROC-EVIDENCE-01`, `INV-PROC-NOTOLERANCE-01` and `INV-PROC-KB-FRESH-01` in
`knowledge/00-invariants/invariants.yaml`.

---

## The loop

```text
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   ▼                                                                │
 S0 CHARTER ─ S1 LOCATE ─ S2 PREDICT ─ S3 SELF-CHECK ─ S4 EXECUTE ──┤
                                                                    │
              S5 EVIDENCE ─ S6 REFUTE ─ S7 GATE ─ S8 RECORD         │
                                          │                         │
                                          └── FAIL ─────────────────┘
                                              (max 3 laps, then escalate)
```

### S0 — CHARTER

State, in three lines or fewer:

- the task, restated in terms of an observable outcome ("y × shear-E residual tilt is
  removed", not "fix the beam code");
- the **scope**: the exact files that will be read and the exact files that may change;
- the **owning knowledge and invariants**, obtained mechanically:

```bash
python tools/kb/kb.py which <path> [<path> ...]
```

If `kb which` returns no owner for a file you are about to change, that is a gap: create
a draft knowledge document before proceeding. Silence is not permission.

### S1 — LOCATE

Read the code. Produce a claim table where **every row cites `file.py:line`**.

| # | Claim | Anchor |
|---|-------|--------|
| 1 | Beam is applied once, inside the theory wrapper | `multiprobe_namaster.py:NNNN` |

Rules:

- Memory is not a source. Neither is this document, the README, or a previous session's
  summary — those are *hypotheses to check against the code*, and they may be stale.
- A stored notebook output is never an anchor (`INV-PROC-EVIDENCE-01`). Re-execute the
  cell or mark the row `UNVERIFIED`.
- If S1 contradicts a knowledge document, **stop** and report the contradiction. Do not
  quietly follow the code and leave the document wrong; do not quietly follow the
  document and leave the code wrong. Which one is right is a decision, and it gets
  recorded in the journal.

### S2 — PREDICT (pre-registration)

Before changing anything, write down what the change will do — quantitatively and
falsifiably. This is the step that prevents post-hoc rationalisation, and it is where
physical reasoning actually happens.

```markdown
### Pre-registered prediction
- **Direction:** y × shear-E bandpowers decrease at ell > 800.
- **Magnitude:** 5–15% at the highest band; < 1% below ell = 300.
- **Affected families:** act_y_des_shear_E, desi_g_act_y. Others unchanged bit-for-bit.
- **Unaffected invariants:** INV-DV-SHAPE-01, INV-SHEAR-SIGN-01 (no field construction).
- **Falsifier:** if des_shear_EE changes at all, the change is wrong.
```

A prediction that cannot be falsified is not a prediction. "Things will improve" fails
this step. If you genuinely cannot predict the sign or the order of magnitude, say so
explicitly — that is itself a finding, and it means the change is exploratory and must be
labelled as such in S8.

### S3 — SELF-CHECK against invariants

For every invariant returned by `kb which` for the changed scope, write one line:

```text
INV-BEAM-01          HOLDS     beam applied once at multiprobe_namaster.py:NNNN; no second factor in theory_utils
INV-DV-SHAPE-01      HOLDS     no change to spectrum inventory or ordering
INV-WINDOW-CMP-01    AT-RISK   touches the windowing path; must re-run the windowed comparison in S5
INV-NMT-BANDMAJOR-01 N/A       covariance extraction untouched
```

- `HOLDS` requires a reason with an anchor. Bare `HOLDS` is rejected at S7.
- `AT-RISK` is the honest answer whenever the change touches an invariant's `scope`. It
  obliges a specific S5 check.
- `VIOLATED` stops the loop. Return to S2 with a different design, or escalate. You may
  not proceed with a known violation because the result "looks better".
- Every `blocker` invariant with `check.kind: manual` in scope requires an explicit
  `HOLDS because <file:line>` line — there is no automated substitute.

### S4 — EXECUTE

Make the smallest change that tests the prediction. One concern per lap.

- Do not opportunistically reformat, rename, or "clean up" adjacent code. It makes the
  S5 diff uninterpretable, and diff interpretability is the entire point.
- Never edit a test, tolerance, eigenvalue cut, ell range, or convergence threshold in
  the same lap as a physics change (`INV-PROC-NOTOLERANCE-01`).

### S5 — EVIDENCE

Run the checks. Record real commands and real output in the ledger — never a paraphrase,
never a remembered number (`INV-PROC-EVIDENCE-01`).

Minimum evidence set:

1. **Invariant checks** for everything marked `AT-RISK` in S3:
   ```bash
   python tools/kb/kb.py invariants --check --id INV-BEAM-01 --id INV-WINDOW-CMP-01
   ```
2. **The owning documents' `checks`**:
   ```bash
   python tools/kb/kb.py check --scope notebooks/xDESI/survey_measure/
   ```
3. **The prediction test** from S2 — the numbers that confirm or refute it.
4. **A null / unchanged control.** Show that what you predicted would *not* change did
   not change. An agent that only measures the thing it hoped to improve has measured
   nothing. This is the most frequently skipped and most valuable line of evidence.

If the prediction from S2 is refuted: that is a **result, not a failure**. Record it, and
go back to S2 with a corrected physical understanding. Do not adjust the prediction after
seeing the output and present it as confirmed — that is the specific dishonesty this loop
exists to prevent.

### S6 — REFUTE (adversarial)

The change must survive an attempt to break it, performed with the opposite disposition
from the one that built it.

- For anything touching physics, conventions, statistics, or a published number, hand off
  to the `physics-referee` agent, whose brief is to *refute*.
- For a mechanical change with a passing automated check, the author may self-refute, but
  must still answer the refutation checklist in writing.

Refutation checklist — answer all seven:

1. **Sign.** Could this be right in magnitude and wrong in sign? Which independent
   measurement fixes the sign? (`INV-SHEAR-SIGN-01`, `INV-KSZ-SIGN-01`)
2. **Units and h.** Which conventions cross this boundary, and where is the conversion?
   (`INV-PHYS-UNITS-01`)
3. **Double application.** Is any factor — beam, pixel window, mask, weight, shot noise,
   m-bias — now applied twice or zero times?
4. **Degeneracy.** Is the improvement absorbing a different error? What would the same
   improvement look like if the true cause were elsewhere?
5. **Coincidence.** Would this evidence also appear if the change did nothing? (Answering
   this is what the S5 null control is for.)
6. **Interpolation and grid.** Does the result depend on grid resolution, ell range, mass
   limits, or z range? Re-run at one different resolution.
7. **Goodness, not improvement.** Is the absolute fit acceptable against `dof - k`, or
   merely better than before? (`INV-CHI2-HONEST-01`)

### S7 — GATE

`PASS` requires **all** of:

- [ ] every S3 invariant is `HOLDS` with an anchored reason, or `N/A`;
- [ ] every `AT-RISK` invariant has a passing S5 check;
- [ ] the S2 prediction was confirmed, **or** was refuted and the loop re-run;
- [ ] a null control shows the predicted-unchanged quantities are unchanged;
- [ ] `python tools/kb/kb.py gate` exits 0;
- [ ] S6 refutation attempted and all seven questions answered;
- [ ] every number in the report has an adjacent command.

Otherwise `FAIL` → return to S2. **Maximum three laps**, then stop and escalate to the
user with: what was tried, what the evidence showed, and the two or three hypotheses that
remain. Three laps without convergence means the problem is mis-framed, and more laps
will produce a confident wrong answer rather than a correct one.

**Escalate immediately, without spending laps, when:**

- an invariant is `VIOLATED` and the only way forward is to weaken it;
- the fix requires changing a tolerance, eigenvalue cut, or prior width;
- the code and a knowledge document disagree about physics;
- the change would invalidate an already-quoted result or a running chain;
- a cluster job costing more than ~1 node-hour would be needed to get evidence.

### S8 — RECORD

Two writes, both mandatory:

1. **Knowledge** — update every document whose claims changed; re-stamp:
   ```bash
   python tools/kb/kb.py verify --doc <id> --evidence knowledge/.kb/ledgers/<ledger>.md
   ```
   New behaviour with no owning document means a new document (from
   `knowledge/_schema/doc-template.md`), not a paragraph bolted onto an unrelated file.

2. **Journal** — one entry, in the format of `_schema/FORMAT.md` section 7:
   ```bash
   python tools/kb/kb.py journal "stage31 beam applied once in theory wrapper" \
     --agent measurement-namaster --invariants INV-BEAM-01 \
     --evidence knowledge/.kb/ledgers/2026-08-03-beam.md
   ```

An exploratory change whose prediction was refuted still gets a journal entry. Negative
results are the cheapest knowledge in the tree and the most often lost.

---

## How to verify

```bash
# The process invariants lint and the gate runs:
python tools/kb/kb.py invariants --lint
python tools/kb/kb.py gate --dry-run

# Every agent must reference this loop:
grep -l "VALIDATION_LOOP" .claude/agents/*.md | wc -l   # must equal the agent count
ls .claude/agents/*.md | wc -l
```

The pre-push gate runs every registered blocker check with visible progress, then executes
the complete `pytest tests/ -q -x` suite. Pytest selectors are intentionally kept literal
and individually enforced; the full suite additionally catches regressions that are not yet
registered as invariants. Registry marker checks use `git grep` with an explicit tracked
scope and must never recurse through ignored runtime products.

## Failure modes

- **Skipping S2.** The dominant failure. Without a pre-registered prediction, any output
  can be narrated as success, and agents are extremely good at narration.
- **Skipping the S5 null control.** Produces "the fix worked" reports for changes that
  did nothing, or that improved one family by breaking another.
- **Self-refutation at S6 on physics changes.** The agent that built the change cannot
  supply the adversarial disposition. Route to `physics-referee`.
- **Laundering a violation into a tolerance change.** Caught by
  `INV-PROC-NOTOLERANCE-01` and `kb gate --check-tolerances`, but only if the gate runs.
- **Looping past three laps.** Produces a large diff, an unreadable ledger, and a wrong
  result defended by a long argument.

## Open questions

- The tolerance sentinel (`python tools/kb/kb.py tolerance-check`) flags numeric-literal-only
  diffs in `tests/` and `param_files/`. It will not catch a tolerance widened via a config
  indirection, nor one inside a `.py` module. Owner: `physics-referee`. Not blocking; S6
  question 7 covers it manually.
- 13 of the 32 invariants are `check.kind: manual`, so they are enforced only by an agent
  writing the `HOLDS because <file:line>` argument at S3. A further 13 need `pytest`, which
  SKIPs where it is unavailable. Run `python tools/kb/kb.py doctor` to see the
  machine-checkable fraction in the current environment before trusting a green gate.
  Converting manual blockers into pytest cases is tracked in `90-journal/2026-08.md`.
  Owner: `repro-runner`.
