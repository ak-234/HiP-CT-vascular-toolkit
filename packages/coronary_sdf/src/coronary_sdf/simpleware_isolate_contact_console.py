"""Create a Simpleware diagnostic project containing one outlet contact.

The source SIP is copied by ``run_simpleware_contact_diagnostic.ps1`` before
this entry point is called.  This script therefore edits only that copy: it
removes every CFD contact and every clipping ROI except the requested plane,
then assigns that one plane as a pressure outlet.  Opening the resulting SIP
shows exactly the surface region selected by that plane.
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from simpleware.scripting import App, Doc, Model


# Defaults to this file's own package directory, which is correct for a normal
# checkout or install. Set CORONARY_SDF_REPOSITORY_DIR when running from
# Simpleware's Scripting tab, where __file__ does not identify this package.
REPOSITORY_DIR = Path(
    os.environ.get("CORONARY_SDF_REPOSITORY_DIR")
    or Path(__file__).resolve().parent
)
_STARTED_AT = time.perf_counter()


def _log(message):
    print(
        "[{} +{:7.1f}s] {}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            time.perf_counter() - _STARTED_AT,
            message,
        ),
        flush=True,
    )


def _resolve_plane(rois, requested):
    exact = [roi for roi in rois if roi.GetName().casefold() == requested.casefold()]
    if len(exact) == 1:
        return exact[0]
    partial = [roi for roi in rois if requested.casefold() in roi.GetName().casefold()]
    if len(partial) == 1:
        return partial[0]
    names = [roi.GetName() for roi in rois if roi.GetName().startswith("COR_OUTLET_")]
    if not partial:
        raise RuntimeError(
            "No outlet plane matches {!r}. Available scripted planes: {}".format(
                requested, ", ".join(names)
            )
        )
    raise RuntimeError(
        "Plane selector {!r} is ambiguous: {}".format(
            requested, ", ".join(roi.GetName() for roi in partial)
        )
    )


def main():
    app = App.GetInstance()
    project_text = str(app.GetInputValue() or "").strip().strip('"')
    requested = str(os.environ.get("CORONARY_DIAGNOSTIC_PLANE", "")).strip()
    if not project_text:
        raise RuntimeError("No diagnostic .sip path was supplied via --input-value.")
    if not requested:
        raise RuntimeError("CORONARY_DIAGNOSTIC_PLANE is not set.")

    project = Path(project_text).expanduser().resolve()
    if not project.is_file() or project.suffix.lower() != ".sip":
        raise RuntimeError("Diagnostic project not found: {}".format(project))

    repository_text = str(REPOSITORY_DIR.resolve())
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    import simpleware_coronary_regions as regions

    _log("Opening diagnostic project copy: {}".format(project))
    document = app.OpenDocument(str(project))
    model = document.GetActiveModel()
    if model.GetModelType() != Model.Cfd:
        raise RuntimeError("The active model is not a CFD model.")
    surface = regions._choose_surface(document)
    part = regions._choose_part(model, surface)

    rois = list(document.GetRegionOfInterestVolumes(Doc.Clipping))
    selected = _resolve_plane(rois, requested)
    selected_name = selected.GetName()
    _log("Isolating clipping ROI: {}".format(selected_name))

    model.RemoveAllCfdContacts()
    removed = 0
    for roi in rois:
        if roi.GetName() == selected_name:
            continue
        document.RemoveRegionOfInterestVolumeByName(Doc.Clipping, roi.GetName())
        removed += 1
    model.AddCfdSurfaceContact(part, selected, Model.PressureOutletContact)
    _log(
        "Removed {} other clipping ROI(s); assigned one pressure-outlet contact "
        "from part {!r}.".format(removed, part.GetName())
    )

    # These calls persist the useful diagnostic display state for the GUI.
    document.ActivateRegionOfInterestVolumes(Doc.Clipping)
    document.SetRegionOfInterestVolumesVisible(Doc.Clipping, True)
    document.SetRegionOfInterestSurfacesHighlighted(Doc.Clipping, True)
    document.AddLogComment(
        "Coronary single-contact diagnostic",
        "Only {} is assigned as a CFD contact.".format(selected_name),
    )
    document.Save()
    _log("Saved one-plane diagnostic project: {}".format(project))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _log("ERROR: {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
