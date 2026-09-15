"""Read-only preflight for building/running the native CGAL extension."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def _visual_studio_cpp() -> str | None:
    compiler = shutil.which("cl")
    if compiler:
        return compiler
    roots = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    ]
    for root in roots:
        visual_studio = root / "Microsoft Visual Studio" / "2022"
        if not visual_studio.exists():
            continue
        matches = sorted(visual_studio.glob("*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if matches:
            return str(matches[-1])
    return None


def checks() -> list[tuple[str, bool, str]]:
    conda = shutil.which("conda")
    if conda is None:
        system_conda = Path(r"C:\ProgramData\anaconda3\Scripts\conda.exe")
        conda = str(system_conda) if system_conda.exists() else None
    compiler = _visual_studio_cpp() if os.name == "nt" else shutil.which("c++")
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    try:
        import CGAL  # type: ignore  # noqa: F401

        cgal = True
    except ImportError:
        # cgal-cpp is header-only and normally has no Python module. Its CMake
        # package is the authoritative build-time check.
        prefix = Path(sys.prefix)
        cgal = any(prefix.glob("Library/lib/cmake/CGAL*/CGALConfig.cmake")) or any(
            prefix.glob("lib/cmake/CGAL*/CGALConfig.cmake")
        )
    return [
        ("conda", conda is not None, conda or "not found"),
        (
            "C++ compiler",
            compiler is not None,
            compiler
            or (
                "Install Visual Studio 2022 Build Tools with 'Desktop development "
                "with C++'"
                if os.name == "nt"
                else "install a C++17 compiler"
            ),
        ),
        ("cmake", cmake is not None, cmake or "activate the CGAL Conda environment"),
        ("ninja", ninja is not None, ninja or "activate the CGAL Conda environment"),
        ("CGAL CMake package", cgal, str(sys.prefix)),
    ]


def main() -> int:
    failed = False
    for name, passed, detail in checks():
        print(f"[{'OK' if passed else 'MISSING'}] {name}: {detail}")
        failed |= not passed
    if failed:
        print(
            "\nCreate/activate the environment from environment-cgal-win64.yml, "
            "then install native/cgal with pip."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
