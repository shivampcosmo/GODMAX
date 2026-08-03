---
id: kb.validation.git-sync
title: How the knowledge base stays in sync with every push and pull
layer: 70-validation
owner: kb-curator
status: verified
confidence: high
scope:
  - tools/kb/kb.py
  - tools/kb/githooks/pre-push
  - tools/kb/githooks/post-merge
  - tools/kb/githooks/post-checkout
  - tools/kb/githooks/post-rewrite
  - tools/kb/install.sh
  - .claude/hooks/session-start.sh
  - .claude/hooks/on-file-edit.sh
  - .claude/hooks/session-stop.sh
  - .claude/settings.json
  - .gitignore
invariants: [INV-PROC-KB-FRESH-01, INV-PROC-NOTOLERANCE-01]
checks:
  - python tools/kb/kb.py doctor
verified_at_commit: 43e07ca
verified_on: 2026-08-04
see_also: [kb.validation.loop]
scope_digest: sha256:9c63612e982495b2b9973db74f756132
---

## Claim

Knowledge freshness is enforced by content digests, not by discipline. Each document
declares the code it describes (`scope`); `kb` hashes those files; a digest mismatch marks
the document `stale` automatically. `post-merge` and `post-checkout` recompute staleness on
every pull and branch switch; `pre-push` blocks a push whose commits touch the scope of a
non-verified document. `kb doctor` separately verifies that every on-disk framework source
is committed in `HEAD`, not merely allowed by `.gitignore`, so the system really travels to
new clones.

## Why it is true

The mechanism is content-addressed, so it cannot be bypassed by editing frontmatter:
`scope_digest` is recomputed from `git hash-object` over the resolved scope files and
compared to the stamped value (`tools/kb/kb.py`, `compute_scope_digest`). Working-tree
edits are caught as well as committed ones, because hashing reads the file on disk, not
the index. The gate is registered as blocker invariant `INV-PROC-KB-FRESH-01`.

---

## Installation (once per clone, on every machine)

```bash
python tools/kb/kb.py install-hooks
```

This copies `tools/kb/githooks/*` into `.git/hooks/`. It must be re-run per clone —
`.git/hooks` is not tracked by git, which is exactly why the hook *sources* live in
`tools/kb/githooks/` (tracked) and are installed from there.

Verify:

```bash
python tools/kb/kb.py doctor
```

The doctor distinguishes three states: eligible under `.gitignore`, present in the git
index, and committed in `HEAD`. Only the last state justifies saying that the framework
travels with clones. A newly created source that is untracked, or only staged, is a problem.

---

## What fires when

| Event | Hook | Action | Blocking |
|---|---|---|---|
| `git pull` / `git merge` | `post-merge` | recompute digests; write `knowledge/.kb/PENDING.md` | no |
| `git checkout <branch>` | `post-checkout` | same, if the branch changed | no |
| `git rebase` | `post-rewrite` | same | no |
| `git push` | `pre-push` | `kb gate` on the commits being pushed | **yes** |
| Claude session start | `SessionStart` | inject `PENDING.md` into context | no |
| Claude edits a file | `PostToolUse` | record the path; name the owning documents | no |
| Claude session end | `Stop` | warn about touched-but-unverified documents | no |

### On pull — awareness, never blocking

A pull brings in other people's (or the cluster's) commits. Blocking a pull is useless:
the change already exists. Instead, `post-merge` answers "what do I now not know?" and
writes `knowledge/.kb/PENDING.md`:

```text
KB PENDING — generated 2026-08-03T14:02Z from merge 43e07ca..8f19bc2

STALE (scope changed since last verification):
  kb.xdesi.multiprobe-measurement   notebooks/xDESI/survey_measure/multiprobe_namaster.py
      owner: measurement-namaster
      changed: multiprobe_namaster.py (+142 -18)
      invariants at risk: INV-NMT-BANDMAJOR-01, INV-KSZ-SIGN-01

UNOWNED (changed files with no knowledge document):
  notebooks/xDESI/abacus_paste/stage31_pz1_backlight_validation.py

Next: /kb-sync
```

`SessionStart` injects this file, so the first thing any agent knows in a fresh session is
which of its own documents it must distrust. That single behaviour prevents most
confidently-wrong agent output in a multi-machine workflow like this one (laptop + cluster).

### On push — the gate

`pre-push` runs `python tools/kb/kb.py gate --range <local_sha>..<remote_sha>`, which
checks, in order:

1. **Invariant lint** — `invariants.yaml` parses and every entry is well formed.
2. **Blocker invariant checks** — every automatable `severity: blocker` check runs.
3. **Fast test suite** — `pytest tests/ -q -x` (skipped with a warning if pytest or its
   data dependencies are unavailable, so a laptop without cluster data can still push).
4. **Knowledge freshness** — for every file changed in the pushed range, the owning
   documents must be `verified` at the current digest. Stale or missing → **block**.
5. **Journal** — the pushed range must be referenced by a `90-journal/` entry → **block**.
6. **Tolerance sentinel** — a diff that changes *only* numeric literals in `tests/` or
   `param_files/` is flagged as a suspected tolerance relaxation
   (`INV-PROC-NOTOLERANCE-01`) → **block** pending an explicit journal note. Also available
   standalone as `python tools/kb/kb.py tolerance-check`.

Exit codes: `0` pass, `1` blocked, `2` configuration error (missing hook, unparseable
registry — treated as blocked).

**A check that cannot run here is not enforced here.** `pytest`-backed invariants SKIP when
pytest is absent, and the xDESI measurement tests additionally need `pymaster`/NaMaster,
which lives in the cluster `ili-sbi` environment. `kb doctor` prints the machine-checkable
fraction for the current environment for exactly this reason: a green gate on a laptop
without the scientific stack is a weaker statement than a green gate on the cluster, and the
tool should say so rather than imply coverage it does not have.

**No invariant check may invoke `kb gate`.** The gate runs blocker invariant checks, so such
a check recurses until it is killed. `cmd_gate` guards re-entry with `GODMAX_KB_IN_GATE`, and
the two process invariants use non-recursive checks (`tolerance-check`, `stale --exit-code`).

### Bypassing the gate

Bypass exists because a hard-blocked push during a cluster deadline causes worse
behaviour than a recorded bypass:

```bash
GODMAX_KB_GATE=warn git push          # downgrade blocks to warnings, records the bypass
git push --no-verify                  # skips hooks entirely — NOT recorded, avoid
```

`GODMAX_KB_GATE=warn` appends a `GATE BYPASSED` entry to the current month's journal with
the reason from `GODMAX_KB_GATE_REASON` (required; the bypass refuses without it). This is
the honest escape hatch. `--no-verify` leaves no trace and should be treated as a defect
in the workflow, not a tool.

---

## Why digests and not commit ancestry

Commit-range staleness (`git log <verified_at_commit>..HEAD -- <scope>`) breaks in exactly
the situations this repository lives in:

- **uncommitted working-tree edits** — the most common state during an agent session;
- **rebases and squashes** — `verified_at_commit` stops being an ancestor of `HEAD`;
- **cherry-picks across the many branches here** (`main`, `ltuSP`, `map`, `pge_paper`,
  `DESxACT`, …) — the same content arrives under a different SHA;
- **reverts** — content returns to a previously verified state, and a digest correctly
  reports "not stale" while ancestry reports stale forever.

`verified_at_commit` is kept as informational context for humans reading a diff. The digest
is what the gate trusts.

---

## Multi-machine workflow (laptop ↔ cluster)

Products under `data/`, `outputs/`, `results/` are gitignored and never travel. Knowledge
about them does. So a cluster run's outcome reaches the laptop only through the tracked
tree:

```bash
# on the cluster, after a run completes
python tools/kb/kb.py journal "stage31 v2 chains: 8000x4, depth 4 saturated 96%" \
  --agent inference-statistician --invariants INV-MCMC-TREEDEPTH-01 \
  --evidence knowledge/.kb/ledgers/2026-08-03-v2-chains.md
python tools/kb/kb.py verify --doc kb.xdesi.stage31-inference --evidence <ledger>
git add knowledge && git commit -m "kb: stage31 v2 chain diagnostics" && git push

# on the laptop
git pull            # post-merge marks nothing stale; the journal carries the result
```

Record the absolute paths of large products in the knowledge document rather than copying
them. The tree is an index to the data, not a copy of it.

## How to verify

```bash
python tools/kb/kb.py doctor          # hooks installed, registry parses, tree consistent
python tools/kb/kb.py gate --dry-run  # gate logic without blocking
python tools/kb/kb.py stale           # current staleness
git check-ignore -v knowledge/.kb/x.json                       # cache ignored
git ls-files -- .claude/agents knowledge/70-validation tools/kb
git ls-tree -r --name-only HEAD -- .claude/agents knowledge/70-validation tools/kb
```

## Failure modes

- **Hooks not installed on a new clone.** `kb doctor` reports it; `SessionStart` warns.
  Until then the gate is silently absent — this is the main residual hole in the design.
- **`.claude/` re-ignored or never committed.** Ignore negations only make files eligible
  for tracking; they do not put files in a commit. `kb doctor` checks both eligibility and
  actual `HEAD` membership for the complete framework source set.
- **Over-broad `scope`.** A document scoping `src/` is permanently stale, gets ignored,
  and trains everyone to bypass the gate. `kb doctor` warns on scopes matching more than
  25 files.
- **Journal-only commits to satisfy the gate.** Detectable in review: a journal entry with
  no ledger and no re-stamped document is theatre.

## Open questions

- Hook installation cannot be forced by git itself. A `SessionStart` auto-install would be
  more robust but would mean a tracked file writing to `.git/hooks` without consent;
  currently it warns instead. Owner: `kb-curator`.
