#!/usr/bin/env bash
# One-time setup per clone (laptop, cluster, or any new machine).
#
#   bash tools/kb/install.sh
#
# .git/hooks is not tracked by git, so hook *sources* live in tools/kb/githooks/
# (tracked) and are copied into place from there. Re-run this after every fresh
# clone. Everything else in the agent system travels with the repository.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${GODMAX_KB_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

echo "GODMAX agent system — setup"
echo "repo: $REPO_ROOT"
echo

echo "[1/6] installing git hooks"
"$PY" tools/kb/kb.py install-hooks "$@"

echo "[2/6] creating local cache"
mkdir -p knowledge/.kb/ledgers
echo "  knowledge/.kb/ (gitignored)"

echo "[3/6] checking optional dependencies"
"$PY" - <<'EOF'
import importlib.util
import shutil
for mod, why in (("yaml", "invariant registry"),):
    print(f"  {'ok  ' if importlib.util.find_spec(mod) else 'MISSING'} {mod:8} ({why})")
print(f"  {'ok  ' if shutil.which('pytest') else 'MISSING'} pytest   (gate test step)")
EOF
echo "  install missing pieces with: pip install pyyaml pytest"

echo "[4/6] refreshing knowledge state"
"$PY" tools/kb/kb.py sync --quiet || true

echo "[5/6] installing Codex skills and prompts"
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
if [ -d "$CODEX_HOME_DIR" ]; then
  "$PY" tools/kb/sync_codex.py >/dev/null && \
    echo "  installed godmax-* skills and /godmax-* prompts into $CODEX_HOME_DIR"
else
  echo "  skipped: no $CODEX_HOME_DIR (Codex not installed here)"
  echo "  if you add Codex later: python tools/kb/sync_codex.py"
fi

echo "[6/6] health check"
"$PY" tools/kb/kb.py doctor || true

cat <<'EOF'

Done. Day-to-day entry points:

  python tools/kb/kb.py status              what is known and what is stale
  python tools/kb/kb.py which <file>        who owns this code, which rules apply
  python tools/kb/kb.py invariants --check  run the executable physics/convention rules
  python tools/kb/kb.py gate --dry-run      what a push would check
  python tools/kb/sync_codex.py --check     are the Codex artifacts current

In Claude Code:  /kb-status  /kb-sync  /validate  /xdesi-status  /invariant-check  /kb-new
In Codex:        /godmax-kb-status  /godmax-kb-sync  /godmax-validate
                 /godmax-xdesi-status  /godmax-invariant-check  /godmax-kb-new
                 plus the godmax-* skills, offered by description

Both tools share the same knowledge base, invariants, and git hooks. Codex does not run the
Claude lifecycle hooks, so AGENTS.md instructs the model to run those checks itself; the
git-level pre-push gate blocks identically under either tool.
EOF
