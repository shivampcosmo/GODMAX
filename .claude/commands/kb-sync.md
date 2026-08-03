---
description: Reconcile the knowledge base after a pull, merge, or branch switch
argument-hint: "[git range, e.g. ORIG_HEAD..HEAD]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task
---

Reconcile the knowledge tree with the current state of the code.

Run these first:

```bash
python tools/kb/kb.py sync --range ${1:-ORIG_HEAD..HEAD}
python tools/kb/kb.py stale
```

Then, acting as **kb-curator** (follow `.claude/agents/kb-curator.md` and
`knowledge/70-validation/VALIDATION_LOOP.md`):

1. **For each stale document**, read the actual diff over its `scope`:
   ```bash
   git diff ${1:-ORIG_HEAD..HEAD} -- <scope paths>
   ```
   Decide, with the diff in front of you, exactly one of:
   - claim still true → re-verify with an evidence ledger
     (`python tools/kb/kb.py verify --doc <id> --evidence <ledger>`);
   - claim now false → rewrite it, or set `status: deprecated` with `see_also`. Never delete;
   - not yours to judge → hand to the owning agent with the specific diff and question.

2. **For each UNOWNED changed file**, scaffold a draft and route it:
   ```bash
   python tools/kb/kb.py new --id kb.<layer>.<slug> --title "…" --layer <dir> \
     --owner <agent> --scope <file>
   ```

3. **Check whether any invariant is now wrong**, not just any document. A changed convention
   in the code with an unchanged `statement` in `invariants.yaml` is the most dangerous
   state the system can be in — the registry would then enforce the wrong rule.
   ```bash
   python tools/kb/kb.py invariants --check
   ```

4. **Refresh the index and report**:
   ```bash
   python tools/kb/kb.py index
   ```

Report: what changed, what you re-verified, what you deprecated, what you routed to whom,
and anything you could not resolve. Do not mark anything verified you did not actually check
against the diff.
