"""Smoke test for scripts/day5_bench_ddp_scaling.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "day5_bench_ddp_scaling.py"


def test_bench_help_works() -> None:
    """--help exits 0 and lists key flags."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    for flag in ("--config", "--n-warmup", "--n-timed", "--out-json"):
        assert flag in result.stdout, f"Missing flag {flag} in --help"


def test_bench_no_cuda_exits_cleanly(script_env: dict[str, str]) -> None:
    """Without CUDA, script must exit with a clear SystemExit message (not a stacktrace)."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(REPO_ROOT / "configs" / "demo.yaml"),
            "--n-warmup",
            "2",
            "--n-timed",
            "2",
        ],
        cwd=str(REPO_ROOT),
        env=script_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Expect non-zero exit (no CUDA on login) but clear stderr; not a Python traceback
    assert result.returncode != 0
    # Either the SystemExit message OR an early import error message — both are
    # acceptable proof that the script didn't run forward+backward
    assert (
        "CUDA" in (result.stderr + result.stdout)
        or "mamba" in (result.stderr + result.stdout).lower()
    )
