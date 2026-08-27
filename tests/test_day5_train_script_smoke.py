"""Smoke test for scripts/day5_train_boldcast.py.

Doesn't need CUDA or HCP data — verifies argparse + import path + that
the script exits cleanly in --dry-run mode.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "day5_train_boldcast.py"


def test_day5_script_dry_run_completes(tmp_path: Path) -> None:
    """--dry-run should construct argparse + verify imports without loading data."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(REPO_ROOT / "configs" / "demo.yaml"),
            "--out-dir",
            str(tmp_path / "smoke_out"),
            "--max-steps",
            "10",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"day5 script dry-run failed (returncode={result.returncode}).\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # Sanity-check stdout content
    assert "dry-run mode" in result.stdout, (
        f"Expected dry-run banner in stdout. Got:\n{result.stdout}"
    )


def test_day5_script_help_works() -> None:
    """--help should exit 0 and mention key flags."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    for flag in ("--config", "--max-steps", "--out-dir", "--val-every", "--dry-run"):
        assert flag in result.stdout, f"Expected {flag} in --help output. Got:\n{result.stdout}"


def test_day5_sh_script_is_valid_bash() -> None:
    """Day-5 SLURM .sh script parses cleanly and contains key directives."""
    sh = REPO_ROOT / "scripts" / "day5_train_boldcast.sh"
    # bash -n: parse-only check
    result = subprocess.run(
        ["bash", "-n", str(sh)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"bash -n failed: stderr={result.stderr}"
    content = sh.read_text()
    # Required SBATCH directives
    for directive in [
        "--partition=mit_normal_gpu",
        "--gres=gpu:h200:2",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=06:00:00",
    ]:
        assert directive in content, f"Missing SBATCH directive: {directive}"
    # Required body
    assert "torchrun" in content
    assert "--nproc-per-node=2" in content
    assert "scripts/day5_train_boldcast.py" in content
    assert "micromamba activate" in content
    assert "set +u" in content and "set -u" in content
    assert "PYTHONUNBUFFERED" in content
