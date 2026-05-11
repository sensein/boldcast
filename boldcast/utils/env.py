"""Minimal ``.env`` loader for Yibei-runs scripts.

Pure-Python, zero-dependency: reads ``KEY=VALUE`` lines from a ``.env``
file at the repo root, drops blank lines and ``#`` comments, sets the
parsed keys in ``os.environ``.

Used by ``scripts/day*_*.py`` so that ``python scripts/foo.py`` works on a
fresh shell / fresh compute node without the user having to remember
``set -a && source .env && set +a`` each time.

Defaults to ``override=True`` so a stale shell var (e.g., a left-over
``SCRATCH_DIR='$SCRATCH'`` from an earlier session) cannot silently win
over the project ``.env``. See schist memory id 22 for the incident
this guard prevents.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_repo_dotenv"]


def load_repo_dotenv(
    repo_root: Path | str, override: bool = True, max_parents: int = 4
) -> None:
    """Load the project ``.env`` into ``os.environ`` if it can be found.

    Looks for ``.env`` at ``repo_root`` first, then walks up to ``max_parents``
    parent directories. This handles the boldcast worktree layout where
    ``.env`` lives at the main repo root and is *not* duplicated into each
    ``.worktrees/<branch>/`` checkout (worktrees don't share untracked
    files, and ``.env`` is gitignored).

    Parameters
    ----------
    repo_root : Path | str
        Starting directory (typically ``Path(__file__).parent.parent`` from
        a script under ``scripts/``).
    override : bool
        If ``True`` (default), values in ``.env`` overwrite any existing
        entries in ``os.environ``. If ``False``, existing env vars win.
    max_parents : int
        Maximum number of parent directories to walk before giving up.
        Default 4 covers ``<main>/.worktrees/<branch>/`` (2 parents up to
        main) with slack to spare.
    """
    start = Path(repo_root).resolve()
    env_path: Path | None = None
    for depth in range(max_parents + 1):
        candidate = (start if depth == 0 else start.parents[depth - 1]) / ".env"
        if candidate.exists():
            env_path = candidate
            break
    if env_path is None:
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip optional matching quotes around the value.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
