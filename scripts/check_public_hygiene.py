"""Check tracked files against the public-repository rules.

This repository is public. Three classes of content are easy to add without
noticing and hard to remove once pushed, because git history is append-only:

1. **Absolute paths.** They disclose the account name, the cluster
   allocation, and — most importantly — where data held under a use
   agreement lives. Paths belong in the gitignored ``.env`` and are read
   from the environment.
2. **Agent-directed prose.** Documentation on a public research repository
   addresses someone reproducing the work. Text written to or about an
   assistant tells that reader the docs were not written for them.
3. **Unpublished documents under ``docs/``.** MkDocs renders every file in
   the docs directory, whether or not it appears in the navigation, so a
   file dropped there is published even when nothing links to it.

Run it directly, or as the ``publication hygiene`` step in CI::

    python scripts/check_public_hygiene.py

Exits 0 when clean, 1 when any rule is violated, and prints one line per
violation as ``path:line: rule: text``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Filesystem roots that identify this deployment. Extend per site.
PATH_PATTERNS = (
    re.compile(r"/orcd/"),
    re.compile(r"/nese/"),
    re.compile(r"/home/[a-z][a-z0-9_-]*/"),
)

# Assistant names and register tells. Kept deliberately short: a long list
# produces false positives on ordinary technical prose, and the rule is about
# who the documentation addresses, not about banning vocabulary.
PROSE_PATTERNS = (
    re.compile(r"\bclaude\b", re.IGNORECASE),
    re.compile(r"\banthropic\b", re.IGNORECASE),
    re.compile(r"\bcopilot\b", re.IGNORECASE),
    re.compile(r"\bchatgpt\b", re.IGNORECASE),
    re.compile(r"\bload[- ]bearing\b", re.IGNORECASE),
)

# This file necessarily contains the patterns it searches for. Lockfiles are
# generated and may legitimately carry resolved paths.
EXEMPT = frozenset({"scripts/check_public_hygiene.py", "uv.lock"})

# .gitignore has to name the assistant files in order to keep them untracked,
# which is the rule working rather than failing.
PROSE_EXEMPT = frozenset({".gitignore"})


class Violation(NamedTuple):
    """One rule breach, located precisely enough to fix without searching."""

    path: str
    line: int
    rule: str
    text: str


def tracked_text_files() -> list[str]:
    """Return tracked, non-exempt paths that are plausibly text."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    return [p for p in out if p and p not in EXEMPT]


def scan_content(paths: list[str]) -> list[Violation]:
    """Apply the path and prose patterns line by line."""
    found: list[Violation] = []
    for rel in paths:
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary or vanished; nothing to read
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat in PATH_PATTERNS:
                if pat.search(line):
                    found.append(Violation(rel, lineno, "absolute-path", line.strip()))
                    break
            if rel in PROSE_EXEMPT:
                continue
            for pat in PROSE_PATTERNS:
                if pat.search(line):
                    found.append(Violation(rel, lineno, "agent-prose", line.strip()))
                    break
    return found


def scan_orphan_docs() -> list[Violation]:
    """Find docs that MkDocs would publish without anyone linking them."""
    cfg_path = REPO_ROOT / "mkdocs.yml"
    if not cfg_path.exists():
        return []
    cfg = yaml.safe_load(cfg_path.read_text())
    nav: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            nav.add(node)

    walk(cfg.get("nav"))
    excluded = {
        line.strip() for line in (cfg.get("exclude_docs") or "").splitlines() if line.strip()
    }
    docs_dir = REPO_ROOT / (cfg.get("docs_dir") or "docs")
    found: list[Violation] = []
    for path in sorted(docs_dir.rglob("*.md")):
        rel = str(path.relative_to(docs_dir))
        if rel in nav:
            continue
        if any(rel == e or rel.startswith(e.rstrip("/") + "/") for e in excluded):
            continue
        found.append(
            Violation(
                str(path.relative_to(REPO_ROOT)),
                1,
                "orphan-doc",
                "in docs/ but absent from nav and exclude_docs, so it publishes unlinked",
            )
        )
    return found


def main() -> int:
    """Report every violation, then exit non-zero if there were any."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    violations = scan_content(tracked_text_files()) + scan_orphan_docs()
    for v in sorted(violations):
        print(f"{v.path}:{v.line}: {v.rule}: {v.text}")
    if violations:
        by_rule = {r: sum(1 for v in violations if v.rule == r) for _, _, r, _ in violations}
        summary = ", ".join(f"{n} {r}" for r, n in sorted(by_rule.items()))
        sys.stdout.flush()  # keep the listing above the summary under redirection
        print(f"\nFAIL: {len(violations)} violations ({summary})", file=sys.stderr)
        return 1
    print("OK: no public-repository rule violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
