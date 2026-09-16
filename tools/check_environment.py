"""Report whether the current Python can run the toolkit, and what to fix if not.

Run it with the interpreter you intend to use::

    python tools/check_environment.py
    python tools/check_environment.py --no-gui     # headless machines / CI
    python tools/check_environment.py --json

One line per check, ``PASS`` / ``WARN`` / ``FAIL`` / ``INFO``, each failure
followed by the command that fixes it. The exit status is 1 if anything FAILs.

The checks exist because each has gone wrong at least once:

* the wrong interpreter (base conda, system Python, or 3.13) -- hipct_seg_debug
  requires 3.12 exactly;
* the per-user site-packages (``%APPDATA%\\Python``, ``~/.local``) leaking into
  a fresh environment, so a stale editable install elsewhere shadows this
  checkout -- the script reads each package's ``direct_url.json`` and reports
  when the source tree it points at is not this repository;
* ``pip`` belonging to a different interpreter than ``python``;
* numpy drifting to 2.x, which breaks numba and the RLE decoder;
* the Qt / napari stack importing on paper but aborting on load.

Only the standard library is imported at module level, so this runs in an
environment where nothing else does. ``packaging`` is used for version
specifiers when it happens to be importable; otherwise only exact ``==`` pins
are checked and the rest are reported as not checked.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import site
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

REPO_ROOT = Path(__file__).resolve().parents[1]

# (import name, distribution name, checkout directory), in install order.
PACKAGES = (
    ("skeleton_analysis", "skeleton_analysis", "packages/skeleton_analysis"),
    ("coronary_sdf", "coronary-sdf", "packages/coronary_sdf"),
    ("hipct_seg_debug", "hipct-seg-debug", "packages/hipct_seg_debug"),
)
CONSOLE_SCRIPTS = ("hipct-seg-debug", "hipct-edit", "coronary-sdf", "skeleton-analysis")
# Imported in a subprocess: a Qt platform-plugin failure aborts the interpreter
# rather than raising, and that must not take the checker down with it.
GUI_MODULES = ("numba", "PyQt5", "napari", "pyvistaqt", "vtk", "meshlib")
# The variables the README documents for pointing tools at data. Reported, never
# judged: they are per run, not per environment.
DATA_VARS = (
    "HIPCT_GRAPH",
    "HIPCT_SEG",
    "HIPCT_OUT",
    "CORONARY_SDF_INPUT",
    "CORONARY_SDF_OUTPUT_DIR",
    "CORONARY_SDF_STL",
    "CORONARY_SDF_MSH",
    "HIPCT_CORONARY_SDF",
)
REQUIRED_PYTHON = (3, 12)

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


@dataclass
class Result:
    name: str
    status: str
    detail: str
    fix: str = ""


# --------------------------------------------------------------------------- pure helpers


def python_version_ok(version_info: tuple[int, ...]) -> bool:
    """True for 3.12.x only; hipct_seg_debug declares ``>=3.12,<3.13``."""
    return tuple(version_info[:2]) == REQUIRED_PYTHON


def user_site_leak(sys_path: list[str], user_site: str | None, enabled: bool | None) -> bool:
    """True when the per-user site-packages directory is importable from."""
    if not enabled or not user_site:
        return False
    wanted = os.path.normcase(os.path.normpath(user_site))
    return any(os.path.normcase(os.path.normpath(p)) == wanted for p in sys_path if p)


def is_within(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` or below it, after resolving both.

    A different drive letter, or a resolved path on another tree, is False.
    """
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def url_to_path(url: str) -> Path:
    """``file:///C:/x/y`` -> ``C:\\x\\y``; ``file:///home/x`` -> ``/home/x``."""
    parsed = urlparse(url)
    return Path(url2pathname(unquote(parsed.path)))


def parse_direct_url(text: str | None) -> Path | None:
    """The source directory of an editable install, from its ``direct_url.json``.

    None for a regular (non-editable) install, or anything unparseable.
    """
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    dir_info = data.get("dir_info")
    if not isinstance(dir_info, dict) or not dir_info.get("editable"):
        return None
    url = data.get("url")
    if not isinstance(url, str) or not url.startswith("file:"):
        return None
    return url_to_path(url)


def editable_source(dist_name: str) -> Path | None:
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return None
    return parse_direct_url(dist.read_text("direct_url.json"))


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;]*?)\s*(;.*)?$")


def check_pins(requires: list[str], installed_version) -> list[Result]:
    """Compare a distribution's ``Requires-Dist`` against what is installed.

    ``installed_version(name) -> str | None`` is injected so the tests can run
    without touching the real environment. Requirements guarded by an
    ``extra`` marker are skipped; extras are optional by definition.
    """
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:  # pragma: no cover - depends on the environment
        SpecifierSet = None

    out: list[Result] = []
    for req in requires:
        m = _REQ.match(req)
        if not m:
            continue
        name, _extras, spec, marker = m.groups()
        if marker and "extra" in marker:
            continue
        have = installed_version(name)
        label = f"pin {name}"
        if have is None:
            out.append(Result(label, FAIL, f"{req} -- not installed",
                              "python -m pip install -r requirements.txt"))
            continue
        if not spec:
            out.append(Result(label, PASS, f"{have}"))
            continue
        if SpecifierSet is not None:
            ok = SpecifierSet(spec).contains(have, prereleases=True)
        elif spec.startswith("==") and "*" not in spec:
            ok = normalise_version(have) == normalise_version(spec[2:])
        else:
            out.append(Result(label, INFO, f"{have} installed; '{spec}' not checked (no 'packaging')"))
            continue
        if ok:
            out.append(Result(label, PASS, f"{have} satisfies {spec}"))
        else:
            out.append(Result(label, FAIL, f"{have} installed, {spec} required",
                              "python -m pip install -r requirements.txt"))
    return out


def normalise_version(v: str) -> str:
    return re.sub(r"(\.0)+$", "", v.strip())


def numpy_major_ok(version: str) -> bool:
    """numpy 2.x breaks numba (the Amira RLE decoder) and contourpy."""
    try:
        return int(version.split(".")[0]) < 2
    except ValueError:
        return False


# --------------------------------------------------------------------------- checks


def check_interpreter() -> list[Result]:
    out = []
    version = ".".join(str(n) for n in sys.version_info[:3])
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if conda_env and conda_env != "base":
        where = f'conda env "{conda_env}"'
    elif in_venv:
        where = "venv"
    elif conda_env == "base":
        where = "conda base"
    else:
        where = "no virtual environment"
    detail = f"{version} at {sys.executable} ({where})"

    if python_version_ok(sys.version_info):
        out.append(Result("python", PASS, detail))
    else:
        out.append(Result("python", FAIL, detail,
                          "conda env create -f environment.yml && conda activate hipct"))
    if where in ("conda base", "no virtual environment"):
        out.append(Result("environment", WARN, f"running in {where}; the toolkit expects its own env",
                          "conda env create -f environment.yml && conda activate hipct"))
    return out


def check_user_site() -> list[Result]:
    try:
        user_site = site.getusersitepackages()
    except Exception:  # pragma: no cover - only on exotic layouts
        user_site = None
    enabled = getattr(site, "ENABLE_USER_SITE", None)
    if not user_site_leak(sys.path, user_site, enabled):
        return [Result("user-site", PASS, "per-user site-packages not on sys.path")]
    env = os.environ.get("CONDA_DEFAULT_ENV") or "<env>"
    return [Result(
        "user-site", WARN,
        f"{user_site} is on sys.path; packages installed there shadow this environment",
        f"conda env config vars set PYTHONNOUSERSITE=1 -n {env}  (then deactivate and re-activate)",
    )]


def check_pip() -> list[Result]:
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "--version"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [Result("pip", FAIL, f"could not run python -m pip: {exc}",
                       f"{sys.executable} -m ensurepip --upgrade")]
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines() or ["pip missing"]
        return [Result("pip", FAIL, tail[-1], f"{sys.executable} -m ensurepip --upgrade")]
    m = re.search(r"from (.+?) \(python", proc.stdout)
    pip_path = Path(m.group(1)) if m else None
    if pip_path and is_within(pip_path, Path(sys.prefix)):
        return [Result("pip", PASS, proc.stdout.strip())]
    return [Result("pip", FAIL, f"{proc.stdout.strip()} -- outside {sys.prefix}",
                   "always install with `python -m pip ...`, not bare `pip`")]


def check_packages() -> list[Result]:
    out = []
    present = {}
    for import_name, dist_name, rel_dir in PACKAGES:
        spec = importlib.util.find_spec(import_name)
        present[import_name] = spec is not None
        if spec is None:
            out.append(Result(import_name, FAIL, "not installed",
                              "python -m pip install -r requirements.txt"))
            continue
        try:
            version = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            version = "?"
        source = editable_source(dist_name)
        expected = REPO_ROOT / rel_dir
        if source is None:
            origin = Path(spec.origin).parent if spec.origin else None
            if origin and is_within(origin, expected):
                out.append(Result(import_name, PASS, f"{version} from {origin}"))
            else:
                out.append(Result(import_name, WARN,
                                  f"{version} installed non-editable from {origin}; "
                                  f"edits in {expected} will not be picked up",
                                  "python -m pip install -r requirements.txt"))
        elif is_within(source, expected):
            out.append(Result(import_name, PASS, f"{version} editable from {source}"))
        else:
            out.append(Result(import_name, FAIL,
                              f"{version} editable from {source}, but this checkout is {expected}",
                              f"python -m pip uninstall -y {dist_name} && "
                              "python -m pip install -r requirements.txt"))
    if present.get("hipct_seg_debug"):
        for sibling in ("coronary_sdf", "skeleton_analysis"):
            if not present.get(sibling):
                out.append(Result("siblings", FAIL,
                                  f"hipct_seg_debug's edit/SDF commands import {sibling}, which is missing",
                                  "python -m pip install -r requirements.txt"))
    return out


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_versions() -> list[Result]:
    out = []
    numpy = _installed_version("numpy")
    if numpy is None:
        out.append(Result("numpy", FAIL, "not installed", "python -m pip install -r requirements.txt"))
    elif numpy_major_ok(numpy):
        out.append(Result("numpy", PASS, f"{numpy} (<2, as numba needs)"))
    else:
        out.append(Result("numpy", FAIL, f"{numpy}; numpy 2.x breaks numba and the RLE decoder",
                          "python -m pip install -r requirements.txt   # re-applies numpy==1.26.4"))
    requires = None
    try:
        requires = metadata.requires("hipct-seg-debug")
    except metadata.PackageNotFoundError:
        pass
    if requires:
        pins = check_pins(requires, _installed_version)
        # Collapse the passes: one line for "all pins satisfied" reads better
        # than thirty, but every mismatch stays visible.
        bad = [r for r in pins if r.status != PASS]
        if bad:
            out.extend(bad)
        else:
            out.append(Result("pins", PASS, f"all {len(pins)} hipct_seg_debug pins satisfied"))
    return out


def check_gui(timeout: float = 240.0) -> list[Result]:
    code = (
        "import importlib, sys\n"
        f"for m in {GUI_MODULES!r}:\n"
        "    importlib.import_module(m)\n"
        "    print('ok', m)\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return [Result("gui-imports", FAIL, f"importing {', '.join(GUI_MODULES)} hung for {timeout:.0f}s")]
    except OSError as exc:  # pragma: no cover
        return [Result("gui-imports", FAIL, f"could not start subprocess: {exc}")]
    if proc.returncode == 0:
        return [Result("gui-imports", PASS, ", ".join(GUI_MODULES))]
    imported = [line.split(" ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("ok ")]
    failed = next((m for m in GUI_MODULES if m not in imported), "?")
    tail = " | ".join(proc.stderr.strip().splitlines()[-3:]) or f"exit status {proc.returncode}"
    return [Result("gui-imports", FAIL, f"import {failed} failed: {tail}",
                   "python -m pip install -r requirements.txt")]


def check_console_scripts() -> list[Result]:
    out = []
    prefix = Path(sys.prefix)
    for name in CONSOLE_SCRIPTS:
        found = shutil.which(name)
        if found is None:
            out.append(Result(f"script {name}", WARN, "not on PATH",
                              "activate the environment, or use `python -m <package>`"))
        elif not is_within(Path(found), prefix):
            out.append(Result(f"script {name}", WARN, f"{found} belongs to another environment",
                              "activate the environment before running it"))
        else:
            out.append(Result(f"script {name}", PASS, found))
    return out


def check_data_vars() -> list[Result]:
    set_ = [v for v in DATA_VARS if os.environ.get(v)]
    unset = [v for v in DATA_VARS if not os.environ.get(v)]
    detail = (f"set: {', '.join(set_)}" if set_ else "none set") + \
             (f"; unset: {', '.join(unset)}" if unset else "")
    return [Result("data-vars", INFO, detail + " (per run; see README 'Pointing the tools at data')")]


def run_checks(gui: bool = True) -> list[Result]:
    results = []
    results += check_interpreter()
    results += check_user_site()
    results += check_pip()
    results += check_packages()
    results += check_versions()
    if gui:
        results += check_gui()
    results += check_console_scripts()
    results += check_data_vars()
    return results


# --------------------------------------------------------------------------- CLI


def format_results(results: list[Result]) -> str:
    width = max(len(r.name) for r in results) if results else 0
    lines = []
    for r in results:
        lines.append(f"{r.status:<5} {r.name:<{width}}  {r.detail}")
        if r.fix and r.status in (WARN, FAIL):
            lines.append(f"{'':<5} {'':<{width}}  fix: {r.fix}")
    fails = sum(r.status == FAIL for r in results)
    warns = sum(r.status == WARN for r in results)
    lines.append("")
    lines.append(f"{fails} failed, {warns} warnings  --  {sys.executable}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-gui", action="store_true",
                    help="skip importing the Qt/napari/vtk stack (headless machines, CI)")
    ap.add_argument("--json", action="store_true", help="emit the results as a JSON list")
    args = ap.parse_args(argv)

    results = run_checks(gui=not args.no_gui)
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print(format_results(results))
    return 1 if any(r.status == FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
