---
id: kb.validation.multi-tool
title: Running the agent system under both Claude Code and Codex
layer: 70-validation
owner: kb-curator
status: verified
confidence: high
scope:
  - tools/kb/sync_codex.py
  - tools/kb/codex/godmax.rules
  - AGENTS.md
  - notebooks/xDESI/AGENTS.md
  - CLAUDE.md
invariants: [INV-PROC-KB-FRESH-01, INV-PROC-EVIDENCE-01]
checks:
  - python tools/kb/sync_codex.py --check
verified_at_commit: a6dc164
verified_on: 2026-08-04
see_also: [kb.validation.git-sync, kb.validation.agent-roster, kb.validation.loop]
scope_digest: sha256:a3b0b84e01a1e8651503ac9c8c23b9e8
---

## Claim

The knowledge base, invariant registry, `kb` tooling and git hooks are tool-agnostic and
behave identically under Claude Code and Codex. Only the *delivery layer* differs. Codex does
not run the repository's Claude Code lifecycle hooks, so their checks remain explicit
instructions. Modern Codex runtimes can dispatch bounded role-specific workers; when that
capability is unavailable, the same generated skills fall back to sequential role adoption.

## Why it is true

**Shared, identical under both tools.** `tools/kb/kb.py` is plain Python with no Claude
dependency; `tools/kb/githooks/*` are git hooks that fire on git's own events regardless of
which agent — or no agent — is driving. So staleness tracking, `kb which` routing, the
invariant registry, the journal, and above all **the blocking `pre-push` gate** are the same
under either tool.

**Generated, not duplicated.** Canonical role text lives in the tracked `.claude/agents/`
(10 files) and `.claude/commands/` (6 files). `tools/kb/sync_codex.py` generates the Codex
artifacts from it:

| Claude Code | Codex |
|---|---|
| `.claude/agents/<a>.md` | `$CODEX_HOME/skills/godmax-<a>/SKILL.md` |
| `.claude/commands/<c>.md` | `$CODEX_HOME/prompts/godmax-<c>.md` |
| `CLAUDE.md` | `AGENTS.md` (+ `notebooks/xDESI/AGENTS.md`, hierarchical) |
| `settings.json` `permissions` | `$CODEX_HOME/rules/godmax.rules` |
| `settings.json` `hooks` | **no equivalent** — instructed in `AGENTS.md` |
| subagent dispatch | collaboration workers when available; sequential fallback |

`.claude/agents/` stays canonical rather than moving to a neutral directory, so the working
Claude path is never a generated artifact — only one side is generated, and drift has one
direction. Drift is detected by a manifest digest over the sources plus `REWRITE_VERSION`
and an exact comparison of every installed skill/prompt against freshly generated content
(`sync_codex.py --check`, wired into `kb doctor`).

**Why `$CODEX_HOME` artifacts cannot travel.** Codex skills and prompts are user-level, not
repo-level, so unlike `.claude/agents/` they are outside git — the same constraint as
`.git/hooks`. Both are installed per machine by `bash tools/kb/install.sh`.

**Three transformations that are load-bearing**, applied by `sync_codex.py` rather than copying
text:

1. **A "Running under Codex" preamble** is prepended to every skill, telling the model to run
   `kb status` / `which` / `verify` / `journal` itself, because no hook will. It also defines
   bounded role dispatch, one writer per file, root-owned integration and a sequential
   fallback for runtimes without collaboration tools.
2. **Prose agent references are namespaced** — `` `measurement-namaster` `` becomes
   `` `godmax-measurement-namaster` `` — but **only outside fenced code blocks**. Inside a
   fence the plain name must survive, because `kb journal --agent measurement-namaster` and
   `kb verify --agent …` are matched against a knowledge document's `owner:` field, which uses
   the unprefixed name. Prefixing there would break the ownership check in `kb.py`.
3. **Claude frontmatter is stripped.** `tools:` and `model:` have no Codex equivalent;
   `description`/`argument-hint`/`allowed-tools` are folded into prose for prompts so no
   literal YAML is injected into the prompt text.

Names are prefixed `godmax-` because Codex skills and prompts are **global** — they are
offered in every Codex session across every project. Descriptions are scoped to this
repository for the same reason. `godmax-core` is not double-prefixed.

## Delivery differences under Codex

Record these honestly; a future session must not assume Codex sessions run Claude's lifecycle
hooks or that every Codex build exposes the same collaboration capacity.

| Behaviour | Claude Code | Codex |
|---|---|---|
| staleness surfaced at session start | `SessionStart` hook, automatic | instructed in `AGENTS.md` |
| edit routed to owning invariants | `PostToolUse` hook, automatic | instructed |
| unverified documents flagged at end | `Stop` hook, automatic | instructed |
| pre-push gate | **mechanical** | **mechanical** (identical) |
| pull/branch-switch sync | **mechanical** | **mechanical** (identical) |
| role routing | `Task` subagent | collaboration worker when available; sequential fallback |
| S6 adversarial refutation | independent subagent | fresh-context referee worker; separate-session fallback |

**S6 must remain independent.** `physics-referee` works because the refuter did not build the
thing. In a dispatch-capable Codex runtime, launch a fresh-context, read-only
`godmax-physics-referee` worker and provide only the claim, evidence ledger, changed paths and
routed invariants — never the author's reasoning. If dispatch is unavailable or capacity is
exhausted, use a separate Codex task/session. Same-session self-review is acceptable only for
mechanical changes and must be labelled non-independent.

## Codex dispatch protocol

The root coordinator runs `kb status`, `kb stale`, and `kb which <exact paths>` before
dispatch. Each worker receives a bounded contract containing the observable outcome, exact
read/write scope, routed documents and invariants, pre-registered prediction and falsifier,
required evidence and null control, and prohibited actions.

All workers share the same worktree. Parallel read-only investigations and disjoint write
scopes are safe; overlapping writes are not. Exactly one agent writes a given file, and the
root coordinator integrates the result. If two failure modes touch one file, serialise the
work or route integration through `xdesi-lead`. Keep child dispatch under the root coordinator
unless nested dispatch is explicitly useful and capacity is available.

`repro-runner` remains an execution-only S5 role: it records commands and output but does not
interpret or fix. `physics-referee` remains a read-only S6 role: it attempts refutation and
never repairs the claim. The root coordinator disposes findings, runs S7 and coordinates S8;
each document is verified by its declared owner, and the root records the integrated journal
entry.

## How to verify

```bash
python tools/kb/sync_codex.py --check     # artifacts current, no drift
python tools/kb/kb.py doctor              # includes the Codex drift check ('cdx' line)
python tools/kb/sync_codex.py --dry-run   # what would be written, changing nothing

# obsolete single-agent-only wording must be absent
grep -R "No subagent dispatc[h]\|cannot spaw[n] X\|single-agent sessio[n]" \
  AGENTS.md knowledge/70-validation/MULTI_TOOL.md tools/kb/sync_codex.py \
  ~/.codex/skills/godmax-*                # expected: no matches

# the code-fence protection that keeps kb's owner check working
grep -rn -- "--agent" ~/.codex/skills/godmax-physics-referee/SKILL.md
#   must show `--agent physics-referee`, NOT `--agent godmax-physics-referee`
```

Expected: 10 skills, 6 prompts, `--check` exit 0, and no occurrence of `godmax-godmax`
anywhere under `$CODEX_HOME`.

## Failure modes

- **Hand-editing `~/.codex/skills/godmax-*`.** Silently overwritten by the next sync. Edit
  the tracked `.claude/agents/` source and re-sync.
- **Changing `build_skill` / `rewrite_agent_refs` without bumping `REWRITE_VERSION`.**
  `--check` then reports artifacts as current when their generation logic has changed, so
  stale text persists indefinitely.
- **Trusting only the manifest after a generated file was hand-edited.** `--check` compares
  every installed skill and prompt byte-for-byte with freshly generated content, so source
  freshness and target integrity are both required.
- **Prefixing `--agent` values inside code fences.** `kb verify --agent godmax-xdesi-lead`
  fails the owner check against `owner: xdesi-lead`, and the fix looks like a `kb.py` bug.
- **Assuming a Codex session was gated like a Claude one.** The push gate is identical, but
  nothing forced `kb status` at session start, so a Codex session can get further into wrong
  work before the gate catches it.
- **Two dispatched workers editing one file.** Codex workers share the worktree; the later
  write can silently invalidate the earlier worker's evidence. Enforce one writer per file.
- **Giving S6 the author's reasoning history.** That converts an independent refutation into
  a disposition switch. Use a fresh-context referee with only the claim and ledger.
- **Adding an agent without re-syncing.** It exists for Claude Code and not for Codex. `kb
  doctor` reports the drift; nothing else will.
- **Writing to `default.rules` when `godmax.rules` was expected.** `sync_codex.py` writes a
  separate `godmax.rules` by default and never touches the user's existing file;
  `--append-default-rules` merges an idempotent delimited block instead, taking a `.bak`
  first, for Codex builds that read only `default.rules`.

## Open questions

- Whether this Codex build loads every `*.rules` file in `$CODEX_HOME/rules/` or only
  `default.rules` was not determined empirically. The default write to `godmax.rules` is
  non-destructive either way; if permission prompts persist for `kb` commands, re-run with
  `--append-default-rules`. Owner: `kb-curator`. Not blocking — permissions are convenience,
  not correctness.
- Whether Codex substitutes `$1` / `$ARGUMENTS` in custom prompts was not verified in this
  build; generated prompts carry a note to supply the argument inline if not. Owner:
  `kb-curator`. Not blocking.
- Collaboration tools and slot counts vary across Codex runtimes. The generated instructions
  are capability-aware: dispatch bounded roles when available and use the sequential/separate-
  session fallback otherwise. Owner: `kb-curator`. Not blocking.
