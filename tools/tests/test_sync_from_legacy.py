"""Keep checkout guidance and publication edits safe during legacy imports."""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sync(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "sync_from_legacy.py"
    spec = importlib.util.spec_from_file_location("legacy_sync_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("package", ["hipct_seg_debug", "skeleton_analysis", "coronary_sdf"])
def test_import_excludes_checkout_guidance_but_keeps_package_guidance(sync, tmp_path, monkeypatch, package):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for relative in ("CLAUDE.md", "AGENTS.md", "README.md", "docs/AGENTS.md"):
        source = legacy / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("checkout or package instructions", encoding="utf-8")
    monkeypatch.setattr(sync, "REPO", tmp_path / "monorepo")
    monkeypatch.setitem(sync.LEGACY, package, legacy)
    plan = sync.build_plan(package)
    assert set(plan.added) == {"README.md", "docs/AGENTS.md"}
    assert "CLAUDE.md" not in plan.sources
    assert "AGENTS.md" not in plan.sources


def test_apply_preserves_protected_files_and_imports_source(sync, tmp_path, monkeypatch):
    legacy = tmp_path / "legacy"
    repo = tmp_path / "monorepo"
    destination = repo / "packages" / "hipct_seg_debug"
    legacy.mkdir()
    destination.mkdir(parents=True)
    (legacy / "README.md").write_text("retired package documentation", encoding="utf-8")
    (destination / "README.md").write_text("published documentation", encoding="utf-8")
    (legacy / "CLAUDE.md").write_text("This checkout is retired", encoding="utf-8")
    source = legacy / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("VALUE = 42\n", encoding="utf-8")
    monkeypatch.setattr(sync, "REPO", repo)
    monkeypatch.setitem(sync.LEGACY, "hipct_seg_debug", legacy)
    assert sync.main(["--apply", "--only", "hipct_seg_debug"]) == 0
    assert (destination / "README.md").read_text(encoding="utf-8") == "published documentation"
    assert (destination / "src" / "example.py").read_bytes() == source.read_bytes()
    assert not (destination / "CLAUDE.md").exists()
