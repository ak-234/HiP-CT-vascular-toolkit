"""Locate the sibling pipelines this package builds on.

``coronary_sdf`` and ``skeleton_analysis`` live beside this package in the
HiP-CT Vascular Toolkit monorepo and both ship packaging metadata, so the
supported setup is to pip-install all three (see the root README). When that
has been done these helpers are no-ops: the import simply resolves.

They still exist for the two cases where it has not. An uninstalled checkout
is handled by searching the monorepo layout below; anything else has to be
named by ``HIPCT_CORONARY_SDF``. Either way a missing package fails with an
explanation of what was looked for rather than a bare ``ModuleNotFoundError``,
and the search happens once per entry point instead of scattering ``sys.path``
surgery through every module.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Where to look for the directory *containing* ``coronary_sdf``, in order.
# The environment variable wins so a checkout elsewhere can be used without
# editing code.
_ENV_VAR = "HIPCT_CORONARY_SDF"
# ``parents[3]`` is the checkout root: this file is
# ``<root>/src/hipct_seg_debug/edit/_deps.py``. A sibling checkout of
# ``coronary_sdf`` therefore lives beside the root, and a vendored one inside it.
_REPO_ROOT = Path(__file__).resolve().parents[3]
# In the monorepo the checkout root is ``packages/hipct_seg_debug``, so its
# sibling package's importable root is ``../coronary_sdf/src``.
_MONOREPO_SIBLING = _REPO_ROOT.parent / "coronary_sdf" / "src"
# No absolute path here: a drive letter from the machine this was written on is
# either wrong or, worse, right about the wrong data. Only locations relative to
# the checkout are searched; anywhere else has to be named by the variable above.
_DEFAULT_PARENTS = (_MONOREPO_SIBLING, _REPO_ROOT.parent, _REPO_ROOT)


def _candidate_parents() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get(_ENV_VAR)
    if env:
        p = Path(env)
        # Accept either the package directory itself or its parent.
        out.append(p.parent if p.name == "coronary_sdf" else p)
    out.extend(_DEFAULT_PARENTS)
    return out


def ensure_coronary_sdf() -> None:
    """Make ``import coronary_sdf`` work, or raise with what to do about it.

    Idempotent, and a no-op when the package is already importable.
    """
    if "coronary_sdf" in sys.modules:
        return
    if importlib.util.find_spec("coronary_sdf") is not None:
        return

    tried = []
    for parent in _candidate_parents():
        tried.append(parent)
        if (parent / "coronary_sdf" / "__init__.py").is_file():
            sys.path.insert(0, str(parent))
            importlib.invalidate_caches()
            return

    raise ImportError(
        "cannot find the 'coronary_sdf' package. Looked in: "
        + ", ".join(str(p) for p in tried)
        + f". Set {_ENV_VAR} to the directory containing it."
    )


def ensure_skeleton_analysis() -> None:
    """Check ``skeleton_analysis`` is importable, or explain how to install it."""
    if importlib.util.find_spec("skeleton_analysis") is not None:
        return
    raise ImportError(
        "the 'skeleton_analysis' package is not installed. From its checkout "
        "run:  pip install -e '.[image,viz3d]'"
    )
