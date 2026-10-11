"""scripts/submit.sh passes partition and account from the env file to sbatch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMIT = REPO_ROOT / "scripts" / "submit.sh"


def _submit(tmp_path: Path, script: str, env_lines: str) -> str:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        '#!/usr/bin/env bash\necho "P=${SBATCH_PARTITION:-} A=${SBATCH_ACCOUNT:-} ARGS=$*"\n'
    )
    sbatch.chmod(0o755)
    env_file = tmp_path / "site.env"
    env_file.write_text(env_lines)
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "BOLDCAST_ENV_FILE": str(env_file),
    }
    out = subprocess.run(
        ["bash", str(SUBMIT), script, "--extra"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


SITE = "BOLDCAST_GPU_PARTITION=gpuq\nBOLDCAST_CPU_PARTITION=cpuq\nBOLDCAST_SLURM_ACCOUNT=acct\n"


def test_gpu_wrapper_gets_gpu_partition(tmp_path: Path) -> None:
    out = _submit(tmp_path, "scripts/day5_train_boldcast.sh", SITE)
    assert out == "P=gpuq A=acct ARGS=scripts/day5_train_boldcast.sh --extra"


def test_cpu_wrapper_gets_cpu_partition(tmp_path: Path) -> None:
    out = _submit(tmp_path, "scripts/day2_build_cache.sh", SITE)
    assert out.startswith("P=cpuq A=acct ")


def test_unset_settings_leave_sbatch_defaults(tmp_path: Path) -> None:
    out = _submit(tmp_path, "scripts/day2_build_cache.sh", "")
    assert out.startswith("P= A= ")
