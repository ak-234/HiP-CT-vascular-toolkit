"""The environment checker's pure parts, without touching the real environment."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def chk(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "check_environment.py"
    spec = importlib.util.spec_from_file_location("check_environment_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- interpreter


@pytest.mark.parametrize("version, ok", [
    ((3, 12, 0), True),
    ((3, 12, 14, "final", 0), True),
    ((3, 11, 9), False),
    ((3, 13, 0), False),
])
def test_only_python_312_is_accepted(chk, version, ok):
    assert chk.python_version_ok(version) is ok


# --------------------------------------------------------------------------- user site


def test_user_site_leak_detected_when_directory_is_on_path(chk, tmp_path):
    user_site = tmp_path / "Roaming" / "Python" / "Python312" / "site-packages"
    assert chk.user_site_leak([str(tmp_path / "env"), str(user_site)], str(user_site), True)


def test_user_site_leak_ignores_disabled_or_absent(chk, tmp_path):
    user_site = tmp_path / "site-packages"
    assert not chk.user_site_leak([str(user_site)], str(user_site), False)
    assert not chk.user_site_leak([str(user_site)], None, True)
    assert not chk.user_site_leak([str(tmp_path / "elsewhere")], str(user_site), True)


def test_check_user_site_reports_warn_with_fix(chk, monkeypatch, tmp_path):
    user_site = tmp_path / "site-packages"
    monkeypatch.setattr(chk.site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(chk.site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(chk.sys, "path", [str(user_site)])
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "hipct")
    (result,) = chk.check_user_site()
    assert result.status == chk.WARN
    assert "PYTHONNOUSERSITE=1 -n hipct" in result.fix


# --------------------------------------------------------------------------- editable source


def test_is_within_same_tree(chk, tmp_path):
    root = tmp_path / "repo"
    (root / "packages" / "x").mkdir(parents=True)
    assert chk.is_within(root / "packages" / "x", root)
    assert chk.is_within(root, root)
    assert not chk.is_within(tmp_path / "other", root)


def test_is_within_rejects_other_drive(chk):
    # On Windows these are different drives; elsewhere they are two distinct
    # relative directories. Either way the second is not inside the first.
    assert not chk.is_within(Path("F:/HiP-CT-vascular-toolkit/packages/x"),
                             Path("C:/Users/me/HiP-CT-vascular-toolkit"))


def test_parse_direct_url_editable(chk):
    text = json.dumps({
        "url": "file:///C:/Users/me/HiP-CT-vascular-toolkit/packages/hipct_seg_debug",
        "dir_info": {"editable": True},
    })
    path = chk.parse_direct_url(text)
    assert path is not None
    assert path.parts[-2:] == ("packages", "hipct_seg_debug")


def test_parse_direct_url_posix(chk):
    text = json.dumps({"url": "file:///home/me/repo/packages/coronary_sdf",
                       "dir_info": {"editable": True}})
    path = chk.parse_direct_url(text)
    assert path is not None
    assert path.parts[-3:] == ("repo", "packages", "coronary_sdf")


@pytest.mark.parametrize("text", [
    None,
    "",
    "not json",
    "[]",
    json.dumps({"url": "file:///x", "dir_info": {}}),
    json.dumps({"url": "file:///x", "dir_info": {"editable": False}}),
    json.dumps({"url": "https://pypi.org/x.whl", "archive_info": {}}),
    json.dumps({"url": "https://example.com/x", "dir_info": {"editable": True}}),
])
def test_parse_direct_url_non_editable_or_malformed(chk, text):
    assert chk.parse_direct_url(text) is None


def test_check_packages_flags_editable_from_another_checkout(chk, monkeypatch, tmp_path):
    class Spec:
        origin = str(tmp_path / "other" / "src" / "hipct_seg_debug" / "__init__.py")

    monkeypatch.setattr(chk.importlib.util, "find_spec", lambda name: Spec())
    monkeypatch.setattr(chk.metadata, "version", lambda name: "1.0.0")
    monkeypatch.setattr(chk, "editable_source",
                        lambda dist: tmp_path / "other" / "packages" / dist.replace("-", "_"))
    results = chk.check_packages()
    by_name = {r.name: r for r in results}
    assert by_name["hipct_seg_debug"].status == chk.FAIL
    assert "this checkout is" in by_name["hipct_seg_debug"].detail
    assert "pip uninstall" in by_name["hipct_seg_debug"].fix
    assert "siblings" not in by_name


def test_check_packages_names_missing_sibling(chk, monkeypatch):
    class Spec:
        origin = None

    monkeypatch.setattr(chk.importlib.util, "find_spec",
                        lambda name: Spec() if name == "hipct_seg_debug" else None)
    monkeypatch.setattr(chk.metadata, "version", lambda name: "1.0.0")
    monkeypatch.setattr(chk, "editable_source",
                        lambda dist: chk.REPO_ROOT / "packages" / "hipct_seg_debug")
    results = chk.check_packages()
    siblings = [r for r in results if r.name == "siblings"]
    assert {r.status for r in siblings} == {chk.FAIL}
    assert {("coronary_sdf" in r.detail, "skeleton_analysis" in r.detail) for r in siblings} == {
        (True, False), (False, True)}


# --------------------------------------------------------------------------- pins


def _lookup(installed):
    return lambda name: installed.get(name)


def test_check_pins_exact_match_and_mismatch(chk):
    requires = ["numpy==1.26.4", "scipy==1.13.1", "napari[pyqt5]==0.5.6"]
    results = chk.check_pins(requires, _lookup({"numpy": "1.26.4", "scipy": "1.17.1", "napari": "0.5.6"}))
    by_name = {r.name: r.status for r in results}
    assert by_name == {"pin numpy": chk.PASS, "pin scipy": chk.FAIL, "pin napari": chk.PASS}


def test_check_pins_missing_package_fails(chk):
    (result,) = chk.check_pins(["numba==0.60.0"], _lookup({}))
    assert result.status == chk.FAIL
    assert "not installed" in result.detail


def test_check_pins_skips_extras(chk):
    results = chk.check_pins(['pytest==9.1.1; extra == "dev"', "numpy==1.26.4"], _lookup({"numpy": "1.26.4"}))
    assert [r.name for r in results] == ["pin numpy"]


def test_numpy_two_is_rejected(chk):
    assert chk.numpy_major_ok("1.26.4")
    assert not chk.numpy_major_ok("2.0.0")
    assert not chk.numpy_major_ok("2.3.1")
    assert not chk.numpy_major_ok("garbage")


def test_check_versions_fails_on_numpy_two(chk, monkeypatch):
    monkeypatch.setattr(chk, "_installed_version", lambda name: "2.1.0" if name == "numpy" else None)
    monkeypatch.setattr(chk.metadata, "requires", lambda name: None)
    (result,) = chk.check_versions()
    assert result.name == "numpy"
    assert result.status == chk.FAIL


# --------------------------------------------------------------------------- CLI


def test_main_exit_status_follows_failures(chk, monkeypatch, capsys):
    monkeypatch.setattr(chk, "run_checks", lambda gui: [chk.Result("a", chk.PASS, "fine"),
                                                        chk.Result("b", chk.WARN, "meh", "do x")])
    assert chk.main([]) == 0
    out = capsys.readouterr().out
    assert "0 failed, 1 warnings" in out
    assert "fix: do x" in out

    monkeypatch.setattr(chk, "run_checks", lambda gui: [chk.Result("a", chk.FAIL, "broken", "do y")])
    assert chk.main(["--no-gui"]) == 1


def test_main_json_output(chk, monkeypatch, capsys):
    monkeypatch.setattr(chk, "run_checks", lambda gui: [chk.Result("a", chk.PASS, "fine")])
    assert chk.main(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data == [{"name": "a", "status": "PASS", "detail": "fine", "fix": ""}]


def test_no_gui_flag_reaches_run_checks(chk, monkeypatch):
    seen = {}

    def fake(gui):
        seen["gui"] = gui
        return []

    monkeypatch.setattr(chk, "run_checks", fake)
    chk.main(["--no-gui"])
    assert seen["gui"] is False
    chk.main([])
    assert seen["gui"] is True
