#!/usr/bin/env python3
"""GODMAX knowledge-base tool.

Single entry point for the knowledge tree in `knowledge/`: staleness, routing,
invariant checks, the pre-push gate, and the journal.

Design notes that matter:

* **Staleness is content-addressed.** Each document declares `scope` (the code it
  describes). We hash those files and compare to the stamped `scope_digest`. This
  catches uncommitted edits, survives rebases/cherry-picks across the many branches
  in this repo, and correctly un-stales a revert. See knowledge/70-validation/GIT_SYNC.md.

* **Notebook outputs are excluded from the digest.** `.ipynb` files in this repo carry
  megabytes of stored output that churns on every execution. We hash only the source
  cells, so re-running a notebook does not falsely invalidate knowledge. (Stored output
  is never evidence anyway -- INV-PROC-EVIDENCE-01.)

* **No hard third-party dependency.** Frontmatter is parsed by a small local parser.
  PyYAML is used for `invariants.yaml` when available and degrades with a clear message.

Usage:  python tools/kb/kb.py <command> [options]
        python tools/kb/kb.py --help
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
KB = REPO / "knowledge"
CACHE = KB / ".kb"
LEDGERS = CACHE / "ledgers"
JOURNAL_DIR = KB / "90-journal"
INVARIANTS_FILE = KB / "00-invariants" / "invariants.yaml"
AGENTS_DIR = REPO / ".claude" / "agents"
PENDING = CACHE / "PENDING.md"
STATE = CACHE / "state.json"
TOUCHED = CACHE / "session_touched.txt"

# Sources that constitute the portable agent/knowledge framework. `kb doctor` verifies
# both git-index membership and presence in HEAD; .gitignore eligibility alone does not
# make a file travel with clones.
FRAMEWORK_SOURCE_ROOTS = (
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/agents",
    ".claude/commands",
    ".claude/hooks",
    ".claude/settings.json",
    "knowledge",
    "notebooks/xDESI/AGENTS.md",
    "tools/kb",
)

SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", ".kb", "node_modules",
    ".pytest_cache", ".mypy_cache", "arxiv",
}
# Only these extensions participate in a scope digest / directory expansion.
CODE_EXTS = {
    ".py", ".ipynb", ".yaml", ".yml", ".sh", ".sbatch", ".json",
    ".toml", ".cfg", ".md", ".slurm",
}
MAX_SCOPE_FILES_WARN = 25

OK, WARN, BAD, DIM = "\033[32m", "\033[33m", "\033[31m", "\033[2m"
END = "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    OK = WARN = BAD = DIM = END = ""


def _c(color: str, text: str) -> str:
    return f"{color}{text}{END}"


def today() -> str:
    return _dt.date.today().isoformat()


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


# --------------------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------------------

def git(*args: str, check: bool = False) -> str:
    try:
        res = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=check
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def short_head() -> str:
    return git("rev-parse", "--short", "HEAD") or "UNKNOWN"


def changed_files_in_range(rng: str) -> list[str]:
    out = git("diff", "--name-only", rng)
    return [ln for ln in out.splitlines() if ln.strip()]


def commits_in_range(rng: str) -> list[str]:
    out = git("log", "--format=%h", rng)
    return [ln for ln in out.splitlines() if ln.strip()]


def framework_source_files() -> set[str]:
    """Return on-disk framework sources that must be committed.

    Local caches, machine-specific settings and Finder/Python metadata are deliberately
    excluded. Any new real source created under a framework root is included automatically,
    so doctor catches an untracked addition instead of relying on a hand-maintained count.
    """
    files: set[str] = set()
    for root in FRAMEWORK_SOURCE_ROOTS:
        path = REPO / root
        candidates = [path] if path.is_file() else path.rglob("*") if path.is_dir() else []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(REPO)
            if ".kb" in rel.parts or "__pycache__" in rel.parts:
                continue
            if candidate.name == ".DS_Store" or str(rel) == ".claude/settings.local.json":
                continue
            files.add(str(rel))
    return files


# --------------------------------------------------------------------------------------
# Minimal frontmatter parser
# --------------------------------------------------------------------------------------

def _coerce(v: str) -> Any:
    v = v.strip()
    if not v:
        return ""
    if v[0] in "'\"" and v[-1:] == v[0] and len(v) > 1:
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_coerce(p) for p in inner.split(",")]
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    return v


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse leading `---` YAML-ish frontmatter. Supports scalars, inline lists,
    and `- item` block lists. Nested mappings are not used by the schema."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    meta: dict[str, Any] = {}
    key: str | None = None
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if stripped.startswith("- ") and key is not None:
            meta.setdefault(key, [])
            if isinstance(meta[key], list):
                meta[key].append(_coerce(stripped[2:]))
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", raw)
        if m:
            key, val = m.group(1), m.group(2)
            meta[key] = _coerce(val) if val.strip() else []
    body = "\n".join(lines[end + 1:])
    return meta, body


def dump_frontmatter_value(v: Any) -> str:
    if isinstance(v, list):
        return "[]" if not v else "\n" + "\n".join(f"  - {x}" for x in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def rewrite_frontmatter(path: Path, updates: dict[str, Any]) -> None:
    """Update keys in place, preserving order and body. Appends unknown keys."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise SystemExit(f"{path}: no frontmatter to update")
    lines = text.splitlines()
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    head, body = lines[1:end], lines[end:]

    out: list[str] = []
    seen: set[str] = set()
    skipping = False
    for raw in head:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:", raw)
        if m:
            k = m.group(1)
            skipping = False
            if k in updates:
                out.append(f"{k}: {dump_frontmatter_value(updates[k])}")
                seen.add(k)
                skipping = True  # drop following block-list items of the old value
                continue
            out.append(raw)
            continue
        if skipping and raw.strip().startswith("- "):
            continue
        skipping = False
        out.append(raw)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}: {dump_frontmatter_value(v)}")
    path.write_text("\n".join(["---", *out, *body]) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------------------

def file_digest(path: Path) -> str:
    """sha256 of a file's *semantic* content.

    For .ipynb we hash only cell sources and cell types, so re-executing a notebook
    (which rewrites outputs and execution counts) does not mark knowledge stale.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return "MISSING"
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(raw.decode("utf-8", "replace"))
            parts = []
            for cell in nb.get("cells", []):
                src = cell.get("source", "")
                if isinstance(src, list):
                    src = "".join(src)
                parts.append(f"{cell.get('cell_type','?')}\n{src}")
            raw = "\n\x00\n".join(parts).encode("utf-8")
        except Exception:
            pass  # malformed notebook: fall back to raw bytes
    return hashlib.sha256(raw).hexdigest()


def expand_scope(patterns: list[str]) -> list[Path]:
    """Resolve scope entries (files, directories, globs) to concrete code-like files."""
    found: set[Path] = set()
    for pat in patterns:
        pat = str(pat).strip()
        if not pat:
            continue
        target = REPO / pat
        if target.is_dir():
            for p in sorted(target.rglob("*")):
                if any(part in SKIP_DIRS for part in p.parts):
                    continue
                if p.is_file() and p.suffix in CODE_EXTS:
                    found.add(p)
            continue
        if target.is_file():
            found.add(target)
            continue
        for p in sorted(REPO.glob(pat)):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.is_file():
                found.add(p)
    return sorted(found)


def compute_scope_digest(patterns: list[str]) -> tuple[str, list[Path], list[str]]:
    files = expand_scope(patterns)
    missing = [p for p in patterns if not expand_scope([p])]
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(REPO)).encode())
        h.update(b"\0")
        h.update(file_digest(f).encode())
        h.update(b"\n")
    return "sha256:" + h.hexdigest()[:32], files, missing


# --------------------------------------------------------------------------------------
# Document model
# --------------------------------------------------------------------------------------

class Doc:
    def __init__(self, path: Path):
        self.path = path
        self.rel = str(path.relative_to(REPO))
        text = path.read_text(encoding="utf-8", errors="replace")
        self.meta, self.body = parse_frontmatter(text)

    # -- frontmatter accessors ---------------------------------------------------
    @property
    def id(self) -> str:
        return str(self.meta.get("id", "")) or f"<no-id:{self.rel}>"

    @property
    def title(self) -> str:
        return str(self.meta.get("title", "(untitled)"))

    @property
    def owner(self) -> str:
        return str(self.meta.get("owner", ""))

    @property
    def declared_status(self) -> str:
        return str(self.meta.get("status", "draft"))

    @property
    def confidence(self) -> str:
        return str(self.meta.get("confidence", "low"))

    @property
    def layer(self) -> str:
        return str(self.meta.get("layer", ""))

    def _list(self, key: str) -> list[str]:
        v = self.meta.get(key, [])
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return [str(v)] if str(v).strip() else []

    @property
    def scope(self) -> list[str]:
        return self._list("scope")

    @property
    def invariants(self) -> list[str]:
        return self._list("invariants")

    @property
    def checks(self) -> list[str]:
        return self._list("checks")

    # -- derived ----------------------------------------------------------------
    @property
    def stamped_digest(self) -> str:
        return str(self.meta.get("scope_digest", ""))

    def evaluate(self) -> dict[str, Any]:
        digest, files, missing = compute_scope_digest(self.scope)
        declared = self.declared_status
        if declared == "deprecated":
            status = "deprecated"
        elif not self.scope:
            status = "error:no-scope"
        elif missing:
            status = "error:missing-scope"
        elif not self.stamped_digest or self.stamped_digest == "UNSTAMPED":
            status = "unstamped"
        elif self.stamped_digest != digest:
            status = "stale"
        elif declared == "draft":
            status = "draft"
        else:
            status = "verified"
        return {
            "status": status,
            "digest": digest,
            "files": [str(f.relative_to(REPO)) for f in files],
            "missing": missing,
        }


def load_docs() -> list[Doc]:
    if not KB.exists():
        return []
    docs = []
    for p in sorted(KB.rglob("*.md")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(KB)
        if rel.parts and rel.parts[0] in ("_schema", "90-journal"):
            continue
        if p.name in ("INDEX.md", "README.md"):
            continue
        doc = Doc(p)
        if doc.meta.get("id"):
            docs.append(doc)
    return docs


# --------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------

def load_invariants() -> tuple[list[dict], str | None]:
    if not INVARIANTS_FILE.exists():
        return [], f"missing {INVARIANTS_FILE.relative_to(REPO)}"
    try:
        import yaml  # type: ignore
    except ImportError:
        return [], ("PyYAML not installed; invariant commands need it "
                    "(pip install pyyaml)")
    try:
        data = yaml.safe_load(INVARIANTS_FILE.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return [], f"cannot parse invariants.yaml: {exc}"
    invs = data.get("invariants") or []
    if not isinstance(invs, list):
        return [], "invariants.yaml: 'invariants' must be a list"
    return invs, None


REQUIRED_INV_FIELDS = ("id", "title", "layer", "severity", "statement",
                       "rationale", "scope", "check", "owner")
VALID_SEVERITY = {"blocker", "high", "medium"}
VALID_CHECK_KIND = {"pytest", "command", "grep", "manual"}


def lint_invariants() -> list[str]:
    invs, err = load_invariants()
    if err:
        return [err]
    problems: list[str] = []
    seen: set[str] = set()
    for i, inv in enumerate(invs):
        tag = inv.get("id", f"<entry {i}>")
        for f in REQUIRED_INV_FIELDS:
            if not inv.get(f):
                problems.append(f"{tag}: missing required field '{f}'")
        if tag in seen:
            problems.append(f"{tag}: duplicate id")
        seen.add(tag)
        if inv.get("severity") not in VALID_SEVERITY:
            problems.append(f"{tag}: severity must be one of {sorted(VALID_SEVERITY)}")
        chk = inv.get("check") or {}
        if not isinstance(chk, dict):
            problems.append(f"{tag}: check must be a mapping")
        else:
            if chk.get("kind") not in VALID_CHECK_KIND:
                problems.append(
                    f"{tag}: check.kind must be one of {sorted(VALID_CHECK_KIND)}")
            if not chk.get("run"):
                problems.append(f"{tag}: check.run is empty")
        if not re.match(r"^INV-[A-Z0-9]+(-[A-Z0-9]+)*-\d{2}$", str(tag)):
            problems.append(f"{tag}: id must match INV-<AREA>-<SLUG>-<NN>")
    return problems


def run_shell(cmd: str, timeout: int = 900) -> tuple[int, str]:
    try:
        res = subprocess.run(
            cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=timeout
        )
        return res.returncode, (res.stdout + res.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except OSError as exc:
        return 127, str(exc)


def check_invariant(inv: dict, verbose: bool = False) -> tuple[str, str]:
    """Return (result, detail). result in PASS/FAIL/MANUAL/SKIP."""
    chk = inv.get("check") or {}
    kind, run = chk.get("kind"), str(chk.get("run", "")).strip()
    if kind == "manual":
        return "MANUAL", "requires written HOLDS-because argument in the ledger"
    if not run:
        return "SKIP", "no check.run"
    first = run.splitlines()[0]
    if first.startswith("[needs-data]"):
        return "SKIP", "needs cluster data"
    if kind == "pytest" and shutil.which("pytest") is None:
        return "SKIP", "pytest not installed"
    code, out = run_shell(first)
    if kind == "grep":
        # grep exit 1 == no match == the invariant's marker is absent
        return ("PASS", "marker present") if code == 0 else ("FAIL", "marker absent")
    if code == 0:
        return "PASS", out.strip().splitlines()[-1] if out.strip() and verbose else ""
    tail = "\n".join(out.strip().splitlines()[-8:])
    lowered = out.lower()
    if code in (4, 5) or "no tests ran" in lowered or "error: file or directory not found" in lowered:
        return "SKIP", "no matching tests"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "SKIP", f"missing dependency\n{tail}"
    return "FAIL", tail


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------

def cmd_status(args) -> int:
    docs = load_docs()
    if not docs:
        print("No knowledge documents found under knowledge/.")
        return 0
    counts: dict[str, int] = {}
    rows = []
    for d in docs:
        ev = d.evaluate()
        counts[ev["status"]] = counts.get(ev["status"], 0) + 1
        rows.append((d, ev))

    print(f"\n{_c(DIM, 'GODMAX knowledge base')}  @ {short_head()}  "
          f"branch {git('rev-parse','--abbrev-ref','HEAD') or '?'}")
    print(f"{len(docs)} documents\n")

    def mark(s: str) -> str:
        if s == "verified":
            return _c(OK, "verified ")
        if s in ("stale", "unstamped"):
            return _c(WARN, f"{s:9}")
        if s.startswith("error"):
            return _c(BAD, f"{s:9}")
        return f"{s:9}"

    for d, ev in sorted(rows, key=lambda r: (r[0].layer, r[0].id)):
        print(f"  {mark(ev['status'])} {d.id:<44} {_c(DIM, d.owner)}")
        if ev["missing"]:
            print(f"      {_c(BAD, 'missing scope: ' + ', '.join(ev['missing']))}")

    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    invs, err = load_invariants()
    if err:
        print(f"\n  invariants: {_c(WARN, err)}")
    else:
        sev: dict[str, int] = {}
        for i in invs:
            sev[i.get("severity", "?")] = sev.get(i.get("severity", "?"), 0) + 1
        print(f"  invariants: {len(invs)} (" +
              ", ".join(f"{k}={v}" for k, v in sorted(sev.items())) + ")")
    if PENDING.exists():
        print(f"\n  {_c(WARN, 'pending sync:')} {PENDING.relative_to(REPO)}")
    print()
    return 0


def cmd_stale(args) -> int:
    docs = load_docs()
    out = []
    for d in docs:
        ev = d.evaluate()
        if ev["status"] in ("stale", "unstamped") or ev["status"].startswith("error"):
            out.append({
                "id": d.id, "path": d.rel, "owner": d.owner,
                "status": ev["status"], "invariants": d.invariants,
                "scope": d.scope, "missing": ev["missing"],
            })
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    if not out:
        print(_c(OK, "All knowledge documents are verified at the current digest."))
        return 0
    print(f"{len(out)} document(s) need attention:\n")
    for o in out:
        print(f"  {_c(WARN, o['status']):<20} {o['id']}")
        print(f"      owner:      {o['owner'] or _c(BAD,'UNOWNED')}")
        print(f"      path:       {o['path']}")
        if o["invariants"]:
            print(f"      invariants: {', '.join(o['invariants'])}")
        if o["missing"]:
            print(f"      {_c(BAD, 'missing scope entries: ' + ', '.join(o['missing']))}")
        print()
    return 1 if args.exit_code else 0


def cmd_which(args) -> int:
    docs = load_docs()
    invs, _ = load_invariants()
    targets = [str(Path(t)) for t in args.paths]

    def matches(patterns: list[str], target: str) -> bool:
        for pat in patterns:
            pat = str(pat).rstrip("/")
            if target == pat or target.startswith(pat + "/"):
                return True
            if fnmatch.fnmatch(target, pat):
                return True
        return False

    any_hit = False
    for t in targets:
        print(f"\n{_c(DIM, '── ')}{t}")
        dhits = [d for d in docs if matches(d.scope, t)]
        ihits = [i for i in invs if matches(i.get("scope") or [], t)]
        if dhits:
            any_hit = True
            print("  documents:")
            for d in dhits:
                ev = d.evaluate()
                flag = _c(OK, "verified") if ev["status"] == "verified" else _c(WARN, ev["status"])
                print(f"    {flag:<20} {d.id}")
                print(f"      owner: {d.owner}   path: {d.rel}")
        else:
            print(f"  documents: {_c(WARN, 'NONE — unowned code, create a draft first')}")
        if ihits:
            print("  invariants:")
            for i in ihits:
                s = i.get("severity", "?")
                col = BAD if s == "blocker" else (WARN if s == "high" else DIM)
                print(f"    {_c(col, s):<20} {i['id']}  {i.get('title','')}")
                print(f"      check: {i.get('check',{}).get('kind')}  owner: {i.get('owner')}")
        else:
            print("  invariants: none in scope")
        owners = sorted({d.owner for d in dhits if d.owner} |
                        {i.get("owner") for i in ihits if i.get("owner")})
        if owners:
            print(f"  {_c(OK,'route to:')} {', '.join(str(o) for o in owners)}")
    print()
    return 0 if any_hit else 0


def cmd_check(args) -> int:
    docs = load_docs()
    if args.doc:
        docs = [d for d in docs if d.id in args.doc]
    if args.scope:
        sel = []
        for d in docs:
            for s in d.scope:
                if any(str(s).startswith(str(x).rstrip("/")) or str(x).startswith(str(s).rstrip("/"))
                       for x in args.scope):
                    sel.append(d)
                    break
        docs = sel
    if not docs:
        print("No matching documents.")
        return 0
    failures = 0
    for d in docs:
        print(f"\n{_c(DIM,'──')} {d.id}")
        if not d.checks:
            print(f"  {_c(WARN,'no checks declared')}")
            continue
        for chk in d.checks:
            if chk.startswith("[needs-data]") or chk.startswith("TODO"):
                print(f"  {_c(DIM,'SKIP')}  {chk}")
                continue
            code, out = run_shell(chk)
            if code == 0:
                print(f"  {_c(OK,'PASS')}  {chk}")
            else:
                lowered = out.lower()
                if ("no tests ran" in lowered or "modulenotfounderror" in lowered
                        or code in (4, 5) or "not found" in lowered):
                    print(f"  {_c(DIM,'SKIP')}  {chk}  ({code})")
                    continue
                failures += 1
                print(f"  {_c(BAD,'FAIL')}  {chk}")
                for ln in out.strip().splitlines()[-8:]:
                    print(f"        {ln}")
    print()
    return 1 if failures else 0


def cmd_invariants(args) -> int:
    if args.lint:
        problems = lint_invariants()
        if problems:
            print(_c(BAD, f"{len(problems)} problem(s) in invariants.yaml:"))
            for p in problems:
                print(f"  - {p}")
            return 1
        invs, _ = load_invariants()
        print(_c(OK, f"invariants.yaml OK — {len(invs)} entries, schema valid"))
        return 0

    invs, err = load_invariants()
    if err:
        print(_c(BAD, err))
        return 2
    if args.id:
        invs = [i for i in invs if i.get("id") in args.id]
        if not invs:
            print(_c(BAD, "no invariant matched --id"))
            return 2
    if args.layer:
        invs = [i for i in invs if i.get("layer") == args.layer]
    if args.severity:
        invs = [i for i in invs if i.get("severity") == args.severity]

    if not args.check:
        for i in invs:
            sev = i.get("severity", "?")
            col = BAD if sev == "blocker" else (WARN if sev == "high" else DIM)
            print(f"\n{_c(col, sev.upper()):<20} {i['id']}  {i.get('title','')}")
            print(f"  layer: {i.get('layer')}   owner: {i.get('owner')}   "
                  f"check: {i.get('check',{}).get('kind')}")
            stmt = str(i.get("statement", "")).strip().splitlines()
            for ln in stmt[:4]:
                print(f"  {ln.strip()}")
        print()
        return 0

    results: dict[str, int] = {}
    hard_fail = 0
    for i in invs:
        res, detail = check_invariant(i, verbose=args.verbose)
        results[res] = results.get(res, 0) + 1
        col = {"PASS": OK, "FAIL": BAD, "MANUAL": WARN, "SKIP": DIM}[res]
        print(f"  {_c(col, res):<18} {i['id']:<28} {_c(DIM, i.get('severity',''))}")
        if detail and (res != "PASS" or args.verbose):
            for ln in str(detail).splitlines()[:8]:
                print(f"        {ln}")
        if res == "FAIL":
            sev = i.get("severity")
            if sev == "blocker" or (sev == "high" and not args.allow_high):
                hard_fail += 1
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(results.items())))
    if results.get("MANUAL"):
        print(_c(WARN, "\n  MANUAL invariants need a written HOLDS-because argument "
                       "with file:line in the evidence ledger (validation loop S3)."))
    print()
    return 1 if hard_fail else 0


def cmd_verify(args) -> int:
    docs = {d.id: d for d in load_docs()}
    if args.all:
        if not args.bootstrap:
            print(_c(BAD, "--all requires --bootstrap (initial import only)."))
            return 2
        targets = list(docs.values())
    else:
        if not args.doc:
            print(_c(BAD, "specify --doc <id> (repeatable) or --all --bootstrap"))
            return 2
        missing = [i for i in args.doc if i not in docs]
        if missing:
            print(_c(BAD, f"unknown doc id(s): {', '.join(missing)}"))
            return 2
        targets = [docs[i] for i in args.doc]

    if not args.bootstrap and not args.evidence:
        print(_c(BAD, "refusing to re-stamp without --evidence <ledger path>.\n"
                      "  Knowledge is verified by evidence, not by assertion "
                      "(INV-PROC-EVIDENCE-01).\n"
                      "  Scaffold one:  python tools/kb/kb.py ledger new <slug>"))
        return 2
    if args.evidence and not (REPO / args.evidence).exists() and not Path(args.evidence).exists():
        print(_c(BAD, f"evidence ledger not found: {args.evidence}"))
        return 2

    for d in targets:
        ev = d.evaluate()
        if ev["missing"]:
            print(_c(BAD, f"{d.id}: scope entries do not exist: {', '.join(ev['missing'])}"))
            return 1
        if not d.scope:
            print(_c(BAD, f"{d.id}: empty scope — cannot verify"))
            return 1
        if args.agent and d.owner and args.agent != d.owner:
            print(_c(BAD, f"{d.id}: owned by '{d.owner}', not '{args.agent}'. "
                          "Only the owner may verify (see AGENT_ROSTER.md)."))
            return 1
        updates = {
            "scope_digest": ev["digest"],
            "verified_at_commit": short_head(),
            "verified_on": today(),
        }
        # --bootstrap records the current digest as a baseline so staleness tracking
        # starts working. It must NOT upgrade a draft to verified: nobody checked it.
        # Only an explicit evidence-backed verify asserts verification.
        if not args.bootstrap:
            updates["status"] = "verified"
        if args.confidence:
            updates["confidence"] = args.confidence
        rewrite_frontmatter(d.path, updates)
        print(f"  {_c(OK,'stamped')} {d.id}  digest={ev['digest'][:22]}…  "
              f"files={len(ev['files'])}")
    _write_state()
    return 0


def cmd_journal(args) -> int:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    month = _dt.date.today().strftime("%Y-%m")
    jf = JOURNAL_DIR / f"{month}.md"
    commits = args.commits or [short_head()]
    entry = [
        f"### {now_utc()} — {args.message}",
        f"- **Agent:** {args.agent or 'unspecified'}",
        f"- **Commits:** {', '.join(commits)}",
        f"- **Branch:** {git('rev-parse','--abbrev-ref','HEAD') or '?'}",
    ]
    if args.scope:
        entry.append(f"- **Scope:** {', '.join(args.scope)}")
    if args.invariants:
        entry.append(f"- **Invariants:** {', '.join(args.invariants)}")
    if args.docs:
        entry.append(f"- **Documents:** {', '.join(args.docs)}")
    entry.append(f"- **Evidence:** {args.evidence or _c('', 'NONE — exploratory')}")
    if args.decision:
        entry.append(f"- **Decision:** {args.decision}")
    if args.result:
        entry.append(f"- **Result:** {args.result}")
    if args.followup:
        entry.append(f"- **Follow-up:** {args.followup}")
    entry.append("")
    block = "\n".join(entry)

    if jf.exists():
        text = jf.read_text(encoding="utf-8")
        lines = text.splitlines()
        idx = next((i for i, ln in enumerate(lines) if ln.startswith("### ")), len(lines))
        new = lines[:idx] + block.splitlines() + [""] + lines[idx:]
        jf.write_text("\n".join(new).rstrip() + "\n", encoding="utf-8")
    else:
        header = (f"# Journal — {month}\n\n"
                  "Append-only. Newest entry first. Format: "
                  "`knowledge/_schema/FORMAT.md` section 7.\n\n")
        jf.write_text(header + block, encoding="utf-8")
    print(f"  {_c(OK,'journal')} {jf.relative_to(REPO)}")
    return 0


def _write_state() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    state = {
        "generated": now_utc(),
        "head": short_head(),
        "docs": {d.id: {"status": d.evaluate()["status"], "owner": d.owner,
                        "path": d.rel} for d in docs},
    }
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cmd_sync(args) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    LEDGERS.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    invs, _ = load_invariants()

    changed: list[str] = []
    if args.range:
        changed = changed_files_in_range(args.range)

    problem_docs = []
    for d in docs:
        ev = d.evaluate()
        if ev["status"] in ("stale", "unstamped") or ev["status"].startswith("error"):
            problem_docs.append((d, ev))

    owned_patterns: list[str] = []
    for d in docs:
        owned_patterns += [str(s).rstrip("/") for s in d.scope]
    for i in invs:
        owned_patterns += [str(s).rstrip("/") for s in (i.get("scope") or [])]

    def is_owned(f: str) -> bool:
        return any(f == p or f.startswith(p + "/") or fnmatch.fnmatch(f, p)
                   for p in owned_patterns)

    unowned = [f for f in changed
               if not is_owned(f)
               and not f.startswith(("knowledge/", ".claude/", "tools/kb/"))
               and Path(f).suffix in CODE_EXTS]

    lines = [f"KB PENDING — generated {now_utc()}"]
    if args.range:
        lines[0] += f" from {args.range}"
    lines += [f"HEAD {short_head()} on {git('rev-parse','--abbrev-ref','HEAD') or '?'}", ""]

    if problem_docs:
        lines.append("STALE / UNVERIFIED (scope changed since last verification):")
        for d, ev in problem_docs:
            lines.append(f"  {d.id}   [{ev['status']}]")
            lines.append(f"      owner: {d.owner or 'UNOWNED'}")
            lines.append(f"      doc:   {d.rel}")
            if d.invariants:
                lines.append(f"      invariants at risk: {', '.join(d.invariants)}")
            touched = [f for f in changed if is_owned(f) and
                       any(f.startswith(str(s).rstrip('/')) for s in d.scope)]
            if touched:
                lines.append(f"      changed: {', '.join(touched[:6])}")
        lines.append("")
    else:
        lines += ["STALE: none — every document verified at its current digest.", ""]

    if unowned:
        lines.append("UNOWNED (changed code with no knowledge document):")
        for f in unowned[:40]:
            lines.append(f"  {f}")
        if len(unowned) > 40:
            lines.append(f"  … and {len(unowned)-40} more")
        lines.append("")

    if problem_docs or unowned:
        lines += ["Next: run /kb-sync in Claude Code, or:",
                  "  python tools/kb/kb.py stale",
                  "  python tools/kb/kb.py which <changed file>", ""]

    PENDING.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_state()
    if args.quiet:
        n = len(problem_docs)
        if n or unowned:
            print(f"[kb] {n} stale doc(s), {len(unowned)} unowned file(s) — "
                  f"see knowledge/.kb/PENDING.md")
        return 0
    print("\n".join(lines))
    return 0


def cmd_tolerance_check(args) -> int:
    """Standalone tolerance sentinel. Exists so INV-PROC-NOTOLERANCE-01 can have an
    executable check without calling `kb gate` (which runs invariant checks, which would
    re-enter the gate — an infinite recursion)."""
    rng = args.range
    if not rng:
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        rng = f"{upstream}..HEAD" if upstream else "HEAD~1..HEAD"
    suspicious = _tolerance_sentinel(rng)
    if not suspicious:
        print(_c(OK, f"no suspected threshold relaxation in {rng}"))
        return 0
    print(_c(WARN, "numeric-literal-only changes (possible tolerance relaxation):"))
    for f in suspicious:
        print(f"  {f}")
    print("If this loosens a threshold it is a physics change: it needs its own knowledge\n"
          "document, invariant review and explicit user sign-off "
          "(INV-PROC-NOTOLERANCE-01).")
    return 1


def cmd_gate(args) -> int:
    # Guard against re-entry: an invariant whose check invokes `kb gate` would recurse
    # forever, since the gate runs blocker invariant checks.
    if os.environ.get("GODMAX_KB_IN_GATE") == "1":
        print(_c(WARN, "kb gate: already inside a gate run — refusing to recurse"))
        return 0
    os.environ["GODMAX_KB_IN_GATE"] = "1"
    rng = args.range
    if not rng:
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        rng = f"{upstream}..HEAD" if upstream else "HEAD~1..HEAD"
    warn_only = os.environ.get("GODMAX_KB_GATE", "").lower() == "warn" or args.dry_run

    print(f"\n{_c(DIM,'kb gate')}  range={rng}  "
          f"mode={'WARN' if warn_only else 'BLOCK'}")
    blockers: list[str] = []
    warnings: list[str] = []

    # 1. invariant lint
    problems = lint_invariants()
    if problems:
        blockers.append(f"invariants.yaml has {len(problems)} schema problem(s)")
        for p in problems[:5]:
            print(f"  {_c(BAD,'LINT')} {p}")
    else:
        print(f"  {_c(OK,'LINT')} invariants.yaml schema valid")

    # 2. blocker invariant checks
    invs, err = load_invariants()
    if err:
        warnings.append(err)
    else:
        fails = []
        manual = []
        for i in invs:
            if i.get("severity") != "blocker":
                continue
            res, detail = check_invariant(i)
            if res == "FAIL":
                fails.append(i["id"])
                print(f"  {_c(BAD,'INV ')} {i['id']} FAILED")
                for ln in str(detail).splitlines()[:4]:
                    print(f"        {ln}")
            elif res == "MANUAL":
                manual.append(i["id"])
        if fails:
            blockers.append(f"blocker invariants failing: {', '.join(fails)}")
        else:
            print(f"  {_c(OK,'INV ')} no automated blocker invariant is failing")
        if manual:
            print(f"  {_c(WARN,'INV ')} {len(manual)} manual blocker invariant(s) — "
                  "must be argued in the ledger")

    # 3. fast tests
    if shutil.which("pytest") is None:
        print(f"  {_c(DIM,'TEST')} pytest not installed — skipped")
    else:
        code, out = run_shell("pytest tests/ -q -x", timeout=1200)
        lowered = out.lower()
        if code == 0:
            print(f"  {_c(OK,'TEST')} pytest tests/ passed")
        elif "modulenotfounderror" in lowered or "importerror" in lowered or code == 5:
            print(f"  {_c(DIM,'TEST')} skipped (missing dependency or no tests)")
            warnings.append("test suite could not run (dependencies unavailable)")
        else:
            blockers.append("pytest tests/ failed")
            print(f"  {_c(BAD,'TEST')} pytest failed")
            for ln in out.strip().splitlines()[-12:]:
                print(f"        {ln}")

    # 4. knowledge freshness for the pushed range
    changed = changed_files_in_range(rng)
    docs = load_docs()
    if changed:
        impacted = []
        for d in docs:
            pats = [str(s).rstrip("/") for s in d.scope]
            hit = [f for f in changed
                   if any(f == p or f.startswith(p + "/") or fnmatch.fnmatch(f, p)
                          for p in pats)]
            if not hit:
                continue
            ev = d.evaluate()
            if ev["status"] != "verified":
                impacted.append((d, ev, hit))
        if impacted:
            blockers.append(
                f"{len(impacted)} knowledge document(s) not verified at current digest")
            for d, ev, hit in impacted:
                print(f"  {_c(BAD,'KB  ')} {d.id} [{ev['status']}] owner={d.owner}")
                print(f"        changed: {', '.join(hit[:4])}")
        else:
            print(f"  {_c(OK,'KB  ')} owning documents verified for {len(changed)} "
                  "changed file(s)")
    else:
        print(f"  {_c(DIM,'KB  ')} no changed files in range")

    # 5. journal reference
    shas = commits_in_range(rng)
    code_changed = [f for f in changed
                    if Path(f).suffix in CODE_EXTS
                    and not f.startswith(("knowledge/", ".claude/", "tools/kb/"))]
    if shas and code_changed:
        jtext = ""
        if JOURNAL_DIR.exists():
            for jf in JOURNAL_DIR.glob("*.md"):
                jtext += jf.read_text(encoding="utf-8", errors="replace")
        if any(s in jtext for s in shas):
            print(f"  {_c(OK,'JRNL')} journal references the pushed commits")
        else:
            blockers.append("no journal entry references the pushed commits")
            print(f"  {_c(BAD,'JRNL')} no journal entry for {', '.join(shas[:4])}")
            print(f"        fix: python tools/kb/kb.py journal \"<what changed and why>\" "
                  f"--agent <agent> --commits {shas[0]}")
    else:
        print(f"  {_c(DIM,'JRNL')} no code commits requiring a journal entry")

    # 6. tolerance sentinel
    if args.check_tolerances or not args.dry_run:
        suspicious = _tolerance_sentinel(rng)
        if suspicious:
            warnings.append("possible tolerance relaxation (INV-PROC-NOTOLERANCE-01)")
            print(f"  {_c(WARN,'TOL ')} numeric-literal-only changes in:")
            for f in suspicious:
                print(f"        {f}")
            print("        If this loosens a threshold, it is a physics change: it needs "
                  "its own\n        document, invariant review and user sign-off.")
        else:
            print(f"  {_c(OK,'TOL ')} no suspected threshold relaxation")

    print()
    for w in warnings:
        print(f"  {_c(WARN,'warning:')} {w}")
    if not blockers:
        print(_c(OK, "  GATE PASS\n"))
        return 0
    for b in blockers:
        print(f"  {_c(BAD,'blocker:')} {b}")
    if warn_only:
        reason = os.environ.get("GODMAX_KB_GATE_REASON", "")
        if os.environ.get("GODMAX_KB_GATE", "").lower() == "warn":
            if not reason:
                print(_c(BAD, "\n  GODMAX_KB_GATE=warn requires GODMAX_KB_GATE_REASON=\"…\"\n"))
                return 1
            _record_bypass(rng, blockers, reason)
            print(_c(WARN, f"\n  GATE BYPASSED (recorded in journal): {reason}\n"))
        else:
            print(_c(WARN, "\n  GATE would BLOCK (dry-run)\n"))
        return 0
    print(_c(BAD, "\n  GATE BLOCKED\n"))
    print("  Options:")
    print("    - fix the blockers above (preferred)")
    print("    - GODMAX_KB_GATE=warn GODMAX_KB_GATE_REASON=\"…\" git push   "
          "(recorded bypass)\n")
    return 1


def _record_bypass(rng: str, blockers: list[str], reason: str) -> None:
    ns = argparse.Namespace(
        message=f"GATE BYPASSED — {reason}", agent="kb-curator",
        commits=commits_in_range(rng)[:5] or [short_head()], scope=None,
        invariants=["INV-PROC-KB-FRESH-01"], docs=None, evidence=None,
        decision=f"Bypassed with blockers: {'; '.join(blockers)}",
        result=None, followup="Clear these blockers before the next push.",
    )
    cmd_journal(ns)


def _tolerance_sentinel(rng: str) -> list[str]:
    """Flag files whose diff changes only numeric literals, in threshold-bearing dirs."""
    out = []
    for f in changed_files_in_range(rng):
        if not (f.startswith("tests/") or f.startswith("param_files/")):
            continue
        diff = git("diff", "-U0", rng, "--", f)
        adds = [l[1:] for l in diff.splitlines()
                if l.startswith("+") and not l.startswith("+++")]
        dels = [l[1:] for l in diff.splitlines()
                if l.startswith("-") and not l.startswith("---")]
        if not adds or len(adds) != len(dels):
            continue
        num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        only_numbers = True
        for a, d in zip(adds, dels):
            if num.sub("#", a).strip() != num.sub("#", d).strip():
                only_numbers = False
                break
        if only_numbers:
            out.append(f)
    return out


def cmd_install_hooks(args) -> int:
    src = REPO / "tools" / "kb" / "githooks"
    dst = REPO / ".git" / "hooks"
    if not src.is_dir():
        print(_c(BAD, f"missing {src.relative_to(REPO)}"))
        return 2
    if not dst.is_dir():
        print(_c(BAD, "no .git/hooks — not a git repository?"))
        return 2
    for hook in sorted(src.iterdir()):
        if not hook.is_file():
            continue
        target = dst / hook.name
        if target.exists() and not args.force:
            existing = target.read_text(encoding="utf-8", errors="replace")
            if "GODMAX_KB_HOOK" not in existing:
                print(f"  {_c(WARN,'skip')} {hook.name} exists and is not ours "
                      "(use --force to overwrite)")
                continue
        shutil.copyfile(hook, target)
        target.chmod(0o755)
        print(f"  {_c(OK,'installed')} .git/hooks/{hook.name}")
    CACHE.mkdir(parents=True, exist_ok=True)
    LEDGERS.mkdir(parents=True, exist_ok=True)
    print(f"\n  Verify with: python tools/kb/kb.py doctor\n")
    return 0


def cmd_doctor(args) -> int:
    problems, warns = [], []
    print(f"\n{_c(DIM,'kb doctor')}\n")

    # hooks
    hooks_dir = REPO / ".git" / "hooks"
    for h in ("pre-push", "post-merge", "post-checkout", "post-rewrite"):
        p = hooks_dir / h
        if p.exists() and "GODMAX_KB_HOOK" in p.read_text(encoding="utf-8", errors="replace"):
            print(f"  {_c(OK,'hook')} {h} installed")
        else:
            problems.append(f"hook {h} not installed — run: python tools/kb/kb.py install-hooks")
            print(f"  {_c(BAD,'hook')} {h} MISSING")

    # Git portability. Ignore negations make the sources eligible for tracking; they do
    # not prove that the files are indexed or committed.
    gi = (REPO / ".gitignore").read_text(encoding="utf-8", errors="replace")
    if ".claude/*" in gi and "!.claude/agents/" in gi:
        print(f"  {_c(OK,'git ')} .claude framework is eligible for tracking")
    else:
        problems.append(".gitignore does not un-ignore .claude/agents — the agent "
                        "system will not travel with clones")
        print(f"  {_c(BAD,'git ')} .claude/agents appears ignored")

    expected_sources = framework_source_files()
    indexed_sources = set(git("ls-files", "--", *FRAMEWORK_SOURCE_ROOTS).splitlines())
    committed_sources = set(git("ls-tree", "-r", "--name-only", "HEAD", "--",
                                *FRAMEWORK_SOURCE_ROOTS).splitlines())
    not_indexed = sorted(expected_sources - indexed_sources)
    not_committed = sorted(expected_sources - committed_sources)
    if not_indexed:
        problems.append(f"{len(not_indexed)} framework source file(s) are not in the git "
                        "index: " + ", ".join(not_indexed[:4]))
        print(f"  {_c(BAD,'git ')} {len(not_indexed)} framework source file(s) untracked")
    elif not_committed:
        problems.append(f"{len(not_committed)} framework source file(s) are staged but "
                        "not committed in HEAD: " + ", ".join(not_committed[:4]))
        print(f"  {_c(BAD,'git ')} {len(not_committed)} framework source file(s) not in HEAD")
    else:
        print(f"  {_c(OK,'git ')} {len(expected_sources)} framework source files committed")

    code, _ = run_shell("git check-ignore -q knowledge/.kb/x.json")
    print(f"  {_c(OK,'git ') if code == 0 else _c(WARN,'git ')} "
          f"knowledge/.kb cache {'ignored' if code == 0 else 'NOT ignored'}")

    # registry
    problems += [f"invariants.yaml: {p}" for p in lint_invariants()]
    invs, err = load_invariants()
    if err:
        warns.append(err)
    print(f"  {_c(OK,'inv ') if not err else _c(WARN,'inv ')} "
          f"{len(invs)} invariants loaded")

    # Enforcement level in THIS environment. An invariant whose check cannot run here is
    # not enforced here, however green the gate looks — say so rather than imply coverage.
    if invs:
        kinds: dict[str, int] = {}
        for i in invs:
            kinds[(i.get("check") or {}).get("kind", "?")] = \
                kinds.get((i.get("check") or {}).get("kind", "?"), 0) + 1
        have_pytest = shutil.which("pytest") is not None
        runnable = kinds.get("command", 0) + kinds.get("grep", 0) + (
            kinds.get("pytest", 0) if have_pytest else 0)
        print(f"  {_c(OK,'enf ') if runnable else _c(WARN,'enf ')} "
              f"{runnable}/{len(invs)} invariants are machine-checkable here "
              f"({kinds.get('manual',0)} manual, "
              f"{kinds.get('pytest',0)} need pytest)")
        if not have_pytest:
            warns.append(
                f"pytest is not on PATH: {kinds.get('pytest',0)} invariant checks and the "
                "gate's test step will SKIP, not PASS. Enforcement here is weaker than on "
                "a machine with the full environment. Fix: pip install pytest (and note "
                "that the xDESI measurement tests also need pymaster/NaMaster).")

    # documents
    docs = load_docs()
    print(f"  {_c(OK,'docs')} {len(docs)} documents")
    for d in docs:
        ev = d.evaluate()
        if not d.scope:
            problems.append(f"{d.id}: empty scope")
        if ev["missing"]:
            problems.append(f"{d.id}: scope entries do not exist: {', '.join(ev['missing'])}")
        if len(ev["files"]) > MAX_SCOPE_FILES_WARN:
            warns.append(f"{d.id}: scope matches {len(ev['files'])} files — too broad "
                         "to be useful for routing or staleness")
        if not d.owner:
            problems.append(f"{d.id}: no owner")
        # A scope containing another knowledge .md makes staleness circular: re-stamping
        # that document rewrites its frontmatter, changing its hash, which re-stales this
        # one on every pass. Scope the shared *code*; relate documents via see_also.
        selfref = [f for f in ev["files"]
                   if f.startswith("knowledge/") and f.endswith(".md")]
        if selfref:
            problems.append(f"{d.id}: scope includes knowledge document(s) "
                            f"{', '.join(selfref)} — circular staleness; scope the code "
                            "instead and use see_also")

    # agents
    if args.check_agents or True:
        agent_names = {p.stem for p in AGENTS_DIR.glob("*.md")} if AGENTS_DIR.is_dir() else set()
        print(f"  {_c(OK,'agnt') if agent_names else _c(BAD,'agnt')} "
              f"{len(agent_names)} agents in .claude/agents/")
        owners = {d.owner for d in docs if d.owner} | {
            str(i.get("owner")) for i in invs if i.get("owner")}
        for o in sorted(owners):
            if o and o not in agent_names:
                problems.append(f"owner '{o}' has no agent definition in .claude/agents/")
        for a in sorted(agent_names):
            if a not in owners:
                warns.append(f"agent '{a}' owns no document or invariant")
            body = (AGENTS_DIR / f"{a}.md").read_text(encoding="utf-8", errors="replace")
            if "VALIDATION_LOOP" not in body:
                problems.append(f"agent '{a}' does not reference VALIDATION_LOOP.md")

    # Codex side: generated artifacts under $CODEX_HOME must not drift from .claude/ sources.
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    if codex_home.is_dir():
        code, out = run_shell(f"{sys.executable} tools/kb/sync_codex.py --check", timeout=60)
        first = (out.strip().splitlines() or [""])[0]
        if code == 0:
            print(f"  {_c(OK,'cdx ')} {first}")
        else:
            warns.append(f"Codex artifacts out of date — {first} "
                         "(fix: python tools/kb/sync_codex.py)")
            print(f"  {_c(WARN,'cdx ')} {first}")
    else:
        print(f"  {_c(DIM,'cdx ')} no {codex_home} — Codex not in use on this machine")

    print()
    for w in warns:
        print(f"  {_c(WARN,'warning:')} {w}")
    if problems:
        for p in problems:
            print(f"  {_c(BAD,'problem:')} {p}")
        print(_c(BAD, f"\n  {len(problems)} problem(s)\n"))
        return 1
    print(_c(OK, "  healthy\n"))
    return 0


def cmd_index(args) -> int:
    docs = load_docs()
    invs, _ = load_invariants()
    by_layer: dict[str, list[Doc]] = {}
    for d in docs:
        by_layer.setdefault(d.layer or "(unlayered)", []).append(d)

    lines = [
        "# GODMAX knowledge tree — index",
        "",
        f"<!-- GENERATED by `python tools/kb/kb.py index` at {now_utc()} — "
        "do not hand-edit below this line -->",
        "",
        f"{len(docs)} documents · {len(invs)} invariants · HEAD `{short_head()}`",
        "",
        "Start here: [`_schema/FORMAT.md`](_schema/FORMAT.md) for the format,",
        "[`70-validation/VALIDATION_LOOP.md`](70-validation/VALIDATION_LOOP.md) for the",
        "process every agent follows,",
        "[`00-invariants/invariants.yaml`](00-invariants/invariants.yaml) for the rules",
        "that must never break.",
        "",
        "```bash",
        "python tools/kb/kb.py status                 # what is verified / stale",
        "python tools/kb/kb.py which <file>           # who owns this code",
        "python tools/kb/kb.py invariants --check     # run the executable rules",
        "python tools/kb/kb.py gate --dry-run         # what a push would check",
        "```",
        "",
    ]
    for layer in sorted(by_layer):
        lines += [f"## {layer}", "", "| status | id | title | owner |", "|---|---|---|---|"]
        for d in sorted(by_layer[layer], key=lambda x: x.id):
            st = d.evaluate()["status"]
            icon = {"verified": "✅", "stale": "⚠️", "unstamped": "⚠️",
                    "draft": "📝", "deprecated": "🗄️"}.get(st, "❌")
            rel = os.path.relpath(d.path, KB)
            lines.append(f"| {icon} {st} | `{d.id}` | [{d.title}]({rel}) | {d.owner} |")
        lines.append("")

    if invs:
        lines += ["## Invariants by severity", "",
                  "| severity | id | title | check | owner |", "|---|---|---|---|---|"]
        order = {"blocker": 0, "high": 1, "medium": 2}
        for i in sorted(invs, key=lambda x: (order.get(x.get("severity"), 9), x["id"])):
            lines.append(f"| {i.get('severity')} | `{i['id']}` | {i.get('title','')} "
                         f"| {i.get('check',{}).get('kind')} | {i.get('owner')} |")
        lines.append("")

    (KB / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  {_c(OK,'wrote')} knowledge/INDEX.md ({len(docs)} docs, {len(invs)} invariants)")
    return 0


def cmd_ledger(args) -> int:
    LEDGERS.mkdir(parents=True, exist_ok=True)
    if args.ledger_cmd != "new":
        print("usage: kb ledger new <slug>")
        return 2
    slug = re.sub(r"[^a-z0-9-]+", "-", args.slug.lower()).strip("-")
    path = LEDGERS / f"{today()}-{slug}.md"
    if path.exists():
        print(f"  {_c(WARN,'exists')} {path.relative_to(REPO)}")
        return 0
    tmpl = KB / "70-validation" / "EVIDENCE_LEDGER_TEMPLATE.md"
    body = ""
    if tmpl.exists():
        text = tmpl.read_text(encoding="utf-8")
        marker = "# Template — copy below this line"
        if marker in text:
            body = text.split(marker, 1)[1].strip()
            body = re.sub(r"^```markdown\n", "", body)
            body = re.sub(r"\n```$", "", body)
    if body:
        # Substitute into the template's own header rather than prepending a second one.
        branch = git("rev-parse", "--abbrev-ref", "HEAD") or "?"
        body = (body
                .replace("# Ledger: <one-line change description>", f"# Ledger: {args.slug}")
                .replace("- **Date:** 2026-08-03", f"- **Date:** {today()}")
                .replace("- **Branch / commit at start:** ltuSP @ 43e07ca",
                         f"- **Branch / commit at start:** {branch} @ {short_head()}"))
    else:
        body = (f"# Ledger: {args.slug}\n\n- **Date:** {today()}\n"
                "(template missing: knowledge/70-validation/EVIDENCE_LEDGER_TEMPLATE.md)\n")
    path.write_text(body, encoding="utf-8")
    print(f"  {_c(OK,'created')} {path.relative_to(REPO)}")
    return 0


def cmd_new(args) -> int:
    tmpl = KB / "_schema" / "doc-template.md"
    if not tmpl.exists():
        print(_c(BAD, "missing knowledge/_schema/doc-template.md"))
        return 2
    layer_dir = KB / args.layer
    if not layer_dir.is_dir():
        print(_c(BAD, f"unknown layer directory: {args.layer}"))
        return 2
    slug = args.id.split(".")[-1]
    sub = layer_dir / args.subdir if args.subdir else layer_dir
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{slug}.md"
    if path.exists():
        print(_c(BAD, f"already exists: {path.relative_to(REPO)}"))
        return 1
    text = tmpl.read_text(encoding="utf-8")
    text = (text.replace("kb.LAYER.SLUG", args.id)
                .replace("title: TITLE", f"title: {args.title}")
                .replace("layer: 20-physics", f"layer: {args.layer}")
                .replace("owner: halo-model-physicist", f"owner: {args.owner}")
                .replace("verified_on: 1970-01-01", f"verified_on: {today()}"))
    if args.scope:
        text = text.replace("  - src/PATH.py",
                            "\n".join(f"  - {s}" for s in args.scope))
    path.write_text(text, encoding="utf-8")
    print(f"  {_c(OK,'created')} {path.relative_to(REPO)}\n"
          f"  Fill in Claim / Why / How to verify, then:\n"
          f"    python tools/kb/kb.py verify --doc {args.id} --evidence <ledger>")
    return 0


def cmd_touch(args) -> int:
    """Record a file edited during a session (used by the PostToolUse hook)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    with TOUCHED.open("a", encoding="utf-8") as fh:
        for p in args.paths:
            fh.write(p.strip() + "\n")
    if args.quiet:
        return 0
    ns = argparse.Namespace(paths=args.paths)
    return cmd_which(ns)


def cmd_session_end(args) -> int:
    """Warn about files touched this session whose owning documents are unverified."""
    if not TOUCHED.exists():
        return 0
    paths = sorted({ln.strip() for ln in TOUCHED.read_text(encoding="utf-8").splitlines()
                    if ln.strip()})
    docs = load_docs()
    unverified = []
    for d in docs:
        pats = [str(s).rstrip("/") for s in d.scope]
        if not any(p == q or p.startswith(q + "/") for p in paths for q in pats):
            continue
        if d.evaluate()["status"] != "verified":
            unverified.append(d)
    if unverified:
        print("[kb] Files were edited whose knowledge documents are not re-verified:")
        for d in unverified:
            print(f"  - {d.id} (owner {d.owner})")
        print("  Finish the validation loop: S5 evidence -> S7 gate -> S8 record.")
        print("  python tools/kb/kb.py verify --doc <id> --evidence <ledger>")
    TOUCHED.unlink(missing_ok=True)
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kb", description="GODMAX knowledge-base tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="overview of documents and invariants").set_defaults(
        func=cmd_status)

    s = sub.add_parser("stale", help="documents whose scope changed since verification")
    s.add_argument("--json", action="store_true")
    s.add_argument("--exit-code", action="store_true",
                   help="exit 1 if anything is stale")
    s.set_defaults(func=cmd_stale)

    s = sub.add_parser("which", help="which documents/invariants/agent own these paths")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=cmd_which)

    s = sub.add_parser("check", help="run the checks declared by documents")
    s.add_argument("--doc", action="append")
    s.add_argument("--scope", action="append")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("invariants", help="list, lint or run invariants")
    s.add_argument("--lint", action="store_true")
    s.add_argument("--check", action="store_true")
    s.add_argument("--id", action="append")
    s.add_argument("--layer")
    s.add_argument("--severity", choices=sorted(VALID_SEVERITY))
    s.add_argument("--allow-high", action="store_true")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_invariants)

    s = sub.add_parser("verify", help="re-stamp a document after producing evidence")
    s.add_argument("--doc", action="append")
    s.add_argument("--all", action="store_true")
    s.add_argument("--bootstrap", action="store_true",
                   help="initial import only: stamp without an evidence ledger")
    s.add_argument("--evidence")
    s.add_argument("--agent", help="verifying agent; must match the document owner")
    s.add_argument("--confidence", choices=("high", "medium", "low"))
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("journal", help="append a journal entry")
    s.add_argument("message")
    s.add_argument("--agent")
    s.add_argument("--commits", action="append")
    s.add_argument("--scope", action="append")
    s.add_argument("--invariants", action="append")
    s.add_argument("--docs", action="append")
    s.add_argument("--evidence")
    s.add_argument("--decision")
    s.add_argument("--result")
    s.add_argument("--followup")
    s.set_defaults(func=cmd_journal)

    s = sub.add_parser("sync", help="recompute staleness and write PENDING.md")
    s.add_argument("--range", help="git range, e.g. ORIG_HEAD..HEAD")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("tolerance-check",
                       help="flag numeric-literal-only diffs in tests/ and param_files/")
    s.add_argument("--range")
    s.set_defaults(func=cmd_tolerance_check)

    s = sub.add_parser("gate", help="the pre-push gate")
    s.add_argument("--range")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--allow-high", action="store_true")
    s.add_argument("--check-tolerances", action="store_true")
    s.set_defaults(func=cmd_gate)

    s = sub.add_parser("install-hooks", help="install git hooks into .git/hooks")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_install_hooks)

    s = sub.add_parser("doctor", help="check the whole system is wired correctly")
    s.add_argument("--check-agents", action="store_true")
    s.set_defaults(func=cmd_doctor)

    sub.add_parser("index", help="regenerate knowledge/INDEX.md").set_defaults(
        func=cmd_index)

    s = sub.add_parser("ledger", help="scaffold an evidence ledger")
    s.add_argument("ledger_cmd", choices=["new"])
    s.add_argument("slug")
    s.set_defaults(func=cmd_ledger)

    s = sub.add_parser("new", help="scaffold a knowledge document")
    s.add_argument("--id", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--layer", required=True)
    s.add_argument("--owner", required=True)
    s.add_argument("--scope", action="append")
    s.add_argument("--subdir")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("touch", help="record edited paths (PostToolUse hook)")
    s.add_argument("paths", nargs="+")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_touch)

    sub.add_parser("session-end", help="Stop-hook warning about unverified docs"
                   ).set_defaults(func=cmd_session_end)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
