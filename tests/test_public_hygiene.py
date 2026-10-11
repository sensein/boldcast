"""The public-repository check flags cluster settings and site identifiers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_public_hygiene", REPO_ROOT / "scripts" / "check_public_hygiene.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hygiene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    mod = _load()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "LOCAL_PATTERNS_FILE", tmp_path / ".hygiene-local")
    return mod


def _rules(mod: ModuleType, tmp_path: Path, text: str) -> list[str]:
    (tmp_path / "job.sh").write_text(text)
    return [v.rule for v in mod.scan_content(["job.sh"])]


@pytest.mark.parametrize(
    "line",
    [
        "#SBATCH --partition=gpu_main",
        "#SBATCH --account=lab_acct",
        "#SBATCH -p gpu_main",
        "  srun -p gpu_main --gres=gpu:1 python x.py",
        "srun --partition=gpu_main python x.py",
    ],
)
def test_cluster_settings_flagged(hygiene: ModuleType, tmp_path: Path, line: str) -> None:
    assert _rules(hygiene, tmp_path, line + "\n") == ["cluster-setting"]


def test_resource_requests_pass(hygiene: ModuleType, tmp_path: Path) -> None:
    text = "#SBATCH --gres=gpu:2\n#SBATCH --mem=64G\nsrun --gres=gpu:1 python x.py\n"
    assert _rules(hygiene, tmp_path, text) == []


def test_local_patterns_apply_only_when_present(hygiene: ModuleType, tmp_path: Path) -> None:
    assert _rules(hygiene, tmp_path, "runs on sitecluster nodes\n") == []
    (tmp_path / ".hygiene-local").write_text("# comment\n\\bsitecluster\\b\n")
    assert _rules(hygiene, tmp_path, "runs on sitecluster nodes\n") == ["site-identifier"]
