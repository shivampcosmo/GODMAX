#!/usr/bin/env bash
# GODMAX_KB_HOOK — Stop: close the validation loop before the session ends.
#
# Reports files edited this session whose owning knowledge documents were never
# re-verified (validation loop step S8). Advisory only — it never blocks — but it
# is the last chance to notice that a change shipped without its knowledge update.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT" || exit 0

PY="${GODMAX_KB_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -f tools/kb/kb.py ] || exit 0

"$PY" tools/kb/kb.py session-end 2>/dev/null || true
exit 0
