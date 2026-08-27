"""Smoke test for scripts/day5_train_boldcast.py.

Doesn't need CUDA or HCP data — verifies argparse + import path + that
the script exits cleanly in --dry-run mode.

One exception: --dry-run still resolves the config far enough to read the
subject-list files it names, and those are gitignored (real HCP subject
IDs, DUA), so the dry-run case is skipped where they are absent — notably
in CI. See the skipif below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "day5_train_boldcast.py"

# Named by configs/demo.yaml data.subjects_train_file. Gitignored, so present
# on ORCD and on a dev checkout that copied it, absent on a GitHub runner.
SUBJECT_LIST = REPO_ROOT / "configs" / "subjects_train_familydisjoint.txt"


@pytest.mark.skipif(
    not SUBJECT_LIST.exists(),
    reason=(
        f"needs {SUBJECT_LIST.name}, which is gitignored (real HCP subject IDs, "
        "DUA) and so is absent in CI. Regenerate with "
        "scripts/build_family_disjoint_splits.py or copy it between clusters."
    ),
)
def test_day5_script_dry_run_completes(tmp_path: Path, script_env: dict[str, str]) -> None:
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
        env=script_env,
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
