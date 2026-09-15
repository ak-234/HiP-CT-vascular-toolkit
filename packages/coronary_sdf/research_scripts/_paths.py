"""Dataset locations for the research scripts, supplied by the environment.

These scripts are one-off diagnostics kept under version control for
reproducibility, not part of the installable package. They were written
against one machine's drive layout; rather than bake those paths in, each
one now asks for what it needs by name and fails immediately, with
instructions, when it is missing.

Set the variables once per shell::

    export HIPCT_GRAPH=/path/to/graph.am
    export HIPCT_SEG=/path/to/segmentation.am
    export HIPCT_OUT=/path/to/output_dir      # optional, defaults to ./out

The reporting style deliberately matches
``hipct_seg_debug.edit._deps.ensure_coronary_sdf``: say what was looked for
and what to set, rather than raising a bare ``KeyError``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["graph", "segmentation", "out_dir", "require"]


def require(var: str, what: str) -> Path:
    """Return the path named by environment variable ``var``, or explain."""
    value = os.environ.get(var)
    if not value:
        raise SystemExit(
            f"{what} not set. Export {var} to point at it, e.g.\n"
            f"    set {var}=C:\\data\\my_case.am       (Windows)\n"
            f"    export {var}=/data/my_case.am        (POSIX)"
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"{what}: {var} points at a missing path: {path}")
    return path


def graph() -> Path:
    """The input Amira SpatialGraph (.am / .xml)."""
    return require("HIPCT_GRAPH", "Input spatial graph")


def segmentation() -> Path:
    """The matching Amira segmentation lattice (.am)."""
    return require("HIPCT_SEG", "Segmentation lattice")


def out_dir() -> Path:
    """Where to write results; defaults to ./out under the current directory."""
    path = Path(os.environ.get("HIPCT_OUT") or (Path.cwd() / "out")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
