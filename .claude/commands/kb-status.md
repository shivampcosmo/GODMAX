---
description: Show what the knowledge base knows, what is stale, and whether the system is wired up
allowed-tools: Bash, Read
---

```bash
python tools/kb/kb.py status
python tools/kb/kb.py doctor
```

Then summarise for me, briefly:

1. **Coverage** — how much of the code has an owning document, and the biggest unowned area.
2. **Freshness** — what is stale and who owns it.
3. **Enforcement** — how many blocker invariants are automated versus `manual`. Manual ones
   are enforced only by an agent remembering to argue them, so this ratio is the real
   measure of how much the system protects.
4. **Wiring** — are the git hooks installed on this clone? If not, the pre-push gate is
   inactive and the fix is `bash tools/kb/install.sh`.
5. **The single highest-value next action**, with the command to do it.

Keep it short. If everything is healthy, say so in two lines.
