"""ConsoleSimpleware entry point for the coronary region automation.

The project path is supplied by ConsoleSimpleware's ``--input-value`` option.
This wrapper deliberately saves only after the imported automation returns
successfully, so a failed run cannot overwrite the source SIP with partial work.
"""

import sys
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from simpleware.scripting import App


# Defaults to this file's own package directory, which is correct for a normal
# checkout or install. Set CORONARY_SDF_REPOSITORY_DIR when running from
# Simpleware's Scripting tab, where __file__ does not identify this package.
REPOSITORY_DIR = Path(
    os.environ.get("CORONARY_SDF_REPOSITORY_DIR")
    or Path(__file__).resolve().parent
)
_STARTED_AT = time.perf_counter()


def _log(message):
    elapsed = time.perf_counter() - _STARTED_AT
    print(
        "[{} +{:8.1f}s] {}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), elapsed, message
        ),
        flush=True,
    )


def main():
    # GetInstance() is the required root call for instance methods such as
    # GetInputValue() and OpenDocument() in the X-2025.06 Python API.
    app = App.GetInstance()
    input_value = str(app.GetInputValue() or "").strip().strip('"')
    if not input_value:
        raise RuntimeError(
            "No project was supplied. Pass --input-value=<project.sip>."
        )

    project_path = Path(input_value).expanduser().resolve()
    if project_path.suffix.lower() != ".sip":
        raise RuntimeError("Expected a .sip project, got: {}".format(project_path))
    if not project_path.is_file():
        raise RuntimeError("Simpleware project not found: {}".format(project_path))

    repository_dir = REPOSITORY_DIR.expanduser().resolve()
    automation_path = repository_dir / "simpleware_coronary_regions.py"
    if not automation_path.is_file():
        raise RuntimeError("Automation script not found: {}".format(automation_path))
    repository_text = str(repository_dir)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)

    _log("Opening Simpleware project: {}".format(project_path))
    document = app.OpenDocument(str(project_path))
    _log("Project opened; importing coronary automation.")
    import simpleware_coronary_regions

    active_surface_stl = str(
        os.environ.get("CORONARY_ACTIVE_SURFACE_STL", "")
    ).strip().strip('"')
    if active_surface_stl:
        active_surface_path = Path(active_surface_stl).expanduser().resolve()
        if not active_surface_path.is_file():
            raise RuntimeError(
                "Extracted active-surface STL not found: {}".format(
                    active_surface_path
                )
            )
        simpleware_coronary_regions.STL_SURFACE_PATH = str(active_surface_path)
        _log(
            "Using SIP-embedded active surface for cropping and exact plane "
            "validation: {}".format(active_surface_path)
        )

    # Viewer/dataset highlighting calls are irrelevant in headless mode and can
    # dereference missing UI state in ConsoleSimpleware.
    simpleware_coronary_regions.UPDATE_GUI_VISIBILITY = False
    _log("Running simpleware_coronary_regions.main().")
    simpleware_coronary_regions.main()
    _log("Automation succeeded; saving project now.")
    document.Save()
    _log(
        "Project saved successfully: {} (total {:.1f} seconds).".format(
            project_path, time.perf_counter() - _STARTED_AT
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _log("ERROR: {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
