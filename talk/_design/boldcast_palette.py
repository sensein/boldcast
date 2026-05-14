"""BOLDcast-specific palette wrapper: loads 'mit' from the slide-design skill.

Per-deck wrapper pattern: each deck that uses the shared slide-design skill
creates a thin wrapper under its own `_design/` package. This keeps the skill
itself un-modified while allowing the deck to set a fixed palette name and
expose a pre-loaded COLORS dict for import convenience.

Usage
-----
    from _design.boldcast_palette import COLORS, apply_rcparams
    apply_rcparams("mit")        # mutates rcParams, returns color dict
    c = COLORS["primary"]        # "#002896"
"""
from __future__ import annotations
import sys
from pathlib import Path

_SKILL_BUILD = Path("/home/yibei/.claude/skills/slide-design/_build")
if str(_SKILL_BUILD) not in sys.path:
    sys.path.insert(0, str(_SKILL_BUILD))

from palette_mpl import load, apply_rcparams  # noqa: E402

PALETTE_NAME = "mit"
COLORS = load(PALETTE_NAME)
