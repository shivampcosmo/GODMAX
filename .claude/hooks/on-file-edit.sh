#!/usr/bin/env bash
# GODMAX_KB_HOOK — PostToolUse(Edit|Write|NotebookEdit): route the edit.
#
# Reads the tool payload on stdin, records the edited path for the Stop hook, and
# prints the owning knowledge documents and invariants so the agent is reminded —
# at the moment of the edit — which rules it has just put at risk.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT" || exit 0

PY="${GODMAX_KB_PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -f tools/kb/kb.py ] || exit 0

PAYLOAD="$(cat 2>/dev/null || true)"
[ -z "$PAYLOAD" ] && exit 0

FILE_PATH="$(printf '%s' "$PAYLOAD" | "$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
ti = d.get("tool_input") or {}
p = ti.get("file_path") or ti.get("notebook_path") or ""
print(p)
' 2>/dev/null || true)"

[ -z "$FILE_PATH" ] && exit 0

# Normalise to a repo-relative path; ignore edits outside the repo.
case "$FILE_PATH" in
  "$REPO_ROOT"/*) REL="${FILE_PATH#"$REPO_ROOT"/}" ;;
  /*) exit 0 ;;
  *) REL="$FILE_PATH" ;;
esac

# Knowledge and tooling edits do not need routing back to themselves.
case "$REL" in
  knowledge/*|.claude/*|tools/kb/*) exit 0 ;;
esac

"$PY" tools/kb/kb.py touch "$REL" 2>/dev/null || true
exit 0
