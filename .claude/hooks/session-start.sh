#!/usr/bin/env bash
# GODMAX_KB_HOOK — SessionStart: tell the session what it must not trust.
#
# Emits the pending-staleness report so that the first thing any agent knows is
# which knowledge documents are out of date. This is the single highest-value hook
# in the system: it is what stops an agent from acting confidently on a document
# that describes code someone changed on another machine.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT" || exit 0

PY="${GODMAX_KB_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -f tools/kb/kb.py ] || exit 0

"$PY" tools/kb/kb.py sync --quiet >/dev/null 2>&1 || true

PENDING="knowledge/.kb/PENDING.md"

echo "=== GODMAX knowledge base ==="

if [ -f "$PENDING" ]; then
  if grep -q "STALE: none" "$PENDING" && ! grep -q "^UNOWNED" "$PENDING"; then
    echo "All knowledge documents are verified at the current digest."
  else
    echo "Some knowledge is STALE. Treat the documents below as hypotheses to"
    echo "re-check against the code, not as facts (validation loop step S1)."
    echo
    sed -n '1,60p' "$PENDING"
  fi
else
  echo "No knowledge cache yet. Run: python tools/kb/kb.py sync"
fi

# Warn if the gate is not actually wired up on this clone.
if [ ! -f .git/hooks/pre-push ] || ! grep -q GODMAX_KB_HOOK .git/hooks/pre-push 2>/dev/null; then
  echo
  echo "WARNING: git hooks are not installed on this clone, so the pre-push"
  echo "validation gate is INACTIVE. Run: bash tools/kb/install.sh"
fi

echo
echo "Process: knowledge/70-validation/VALIDATION_LOOP.md governs every change."
echo "Routing: python tools/kb/kb.py which <file>"
echo "============================="
exit 0
