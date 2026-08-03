---
description: Situation report on the xDESI multi-probe analysis — state, conventions, open threads
allowed-tools: Bash, Read, Grep, Glob, Task
---

Produce a situation report on the xDESI analysis. Act as **xdesi-lead**
(`.claude/agents/xdesi-lead.md`).

## 1. Knowledge state

```bash
python tools/kb/kb.py which notebooks/xDESI/survey_measure/ notebooks/xDESI/abacus_paste/
python tools/kb/kb.py stale
python tools/kb/kb.py invariants --layer measurement
python tools/kb/kb.py invariants --layer data
```

## 2. Code state

```bash
git log --oneline -15 -- notebooks/xDESI/
git status --porcelain -- notebooks/xDESI/ param_files/xDESI/
ls -t notebooks/xDESI/survey_measure/*.py | head -10
```

## 3. Verification state

```bash
pytest tests/test_xdesi_multiprobe_namaster.py -q 2>&1 | tail -20
python tools/kb/kb.py invariants --check --severity blocker
```

## 4. Report

Structure it as:

**Where the analysis stands.** Lead with the **absolute** goodness of fit — chi2, retained
rank, parameter count, expectation (`rank − k`) — and the dominant misfit family. Never lead
with a relative improvement (`INV-CHI2-HONEST-01`). Say plainly whether the current best fit
is acceptable; if it is an operational point rather than a physical result, say that.

**What is verified vs assumed.** Which knowledge documents are verified at the current
digest; which are stale; which claims currently rest on prose in `survey_measure/README.md`
or `BACKLIGHT_PASTE_HANDOFF_SUMMARY.md` rather than on an executable check.

**Convention risk.** Which blocker invariants are `manual` (enforced only by an agent
arguing them) versus automated. Flag any convention that a recent diff touched.

**Provisional inputs.** Anything recorded as provisional — e.g. the single DR9 random
realization behind the midres2048 mask, and the `ell_max = 2048` limitation for kSZ
validation.

**Open threads and the next decision.** What is blocked, on what, and the two or three
concrete options with what each would cost.

Anchor every claim to a `file:line` or a command you ran. Where the handoff documents and
the code disagree, report the contradiction rather than picking one silently.
