"""Read-only ConsoleSimpleware verification for a generated coronary SIP."""

import sys
import traceback
from pathlib import Path

from simpleware.scripting import App, Doc, Model


EXPECTED_PLANES = 79
EXPECTED_REFINEMENTS = None
EXPECTED_INVERTED_PLANES = 0
# Adaptive finite rectangles can safely use a narrow 0.142 mm side where an
# adjacent outlet loop constrains one axis; the exact-loop validator, rather
# than this coarse persistence check, is the plane-coverage authority.
EXPECTED_MIN_PLANE_SIDE_MM = 0.10
EXPECTED_MIN_REFINEMENT_DIAMETER_MM = 0.16
EXPECTED_REFINEMENT_SHAPE = Doc.Ellipsoid


def _log(message):
    print("[VERIFY] {}".format(message), flush=True)


def main():
    app = App.GetInstance()
    project_path = Path(str(app.GetInputValue() or "").strip().strip('"')).resolve()
    if not project_path.is_file():
        raise RuntimeError("Project not found: {}".format(project_path))

    _log("Opening saved project: {}".format(project_path))
    document = app.OpenDocument(str(project_path))
    model = document.GetActiveModel()
    if model.GetModelType() != Model.Cfd:
        raise RuntimeError("The persisted active model is not a CFD model")

    all_clipping = list(document.GetRegionOfInterestVolumes(Doc.Clipping))
    planes = [
        roi for roi in all_clipping
        if roi.GetName().startswith(("COR_OUTLET_", "COR_INLET_"))
    ]
    other_clipping_names = [
        roi.GetName() for roi in all_clipping if roi not in planes
    ]
    refinements = [
        item for item in model.GetFeFreeMeshRefinementVolumes()
        if item.GetName().startswith("COR_SMALL_")
    ]
    inverted_planes = sum(roi.GetShape().GetInvert() for roi in planes)
    expected_shape_refinements = sum(
        item.GetShape().GetShapeType() == EXPECTED_REFINEMENT_SHAPE
        for item in refinements
    )
    plane_scales = [roi.GetShape().GetScale() for roi in planes]
    refinement_scales = [item.GetShape().GetScale() for item in refinements]
    plane_sides = [
        component
        for scale in plane_scales
        for component in (float(scale.GetX()), float(scale.GetY()))
    ]
    refinement_diameters = [
        component
        for scale in refinement_scales
        for component in (float(scale.GetX()), float(scale.GetY()))
    ]
    plane_scale_valid = bool(plane_sides) and (
        min(plane_sides) + 1.0e-9 >= EXPECTED_MIN_PLANE_SIDE_MM
    )
    refinement_scale_valid = bool(refinement_diameters) and (
        min(refinement_diameters) + 1.0e-9
        >= EXPECTED_MIN_REFINEMENT_DIAMETER_MM
    )
    _log(
        "Reopened successfully: {} scripted plane(s), {} other clipping ROI(s), "
        "{} refinement region(s); {} inverted plane(s), {} expected-shape "
        "primitive(s).".format(
            len(planes),
            len(other_clipping_names),
            len(refinements),
            inverted_planes,
            expected_shape_refinements,
        )
    )
    if other_clipping_names:
        _log("Other clipping ROI names: {}".format(other_clipping_names))
    if plane_sides:
        _log(
            "Plane full-side scale range: {:.3f} to {:.3f} mm.".format(
                min(plane_sides), max(plane_sides)
            )
        )
    if refinement_diameters:
        _log(
            "Refinement full-diameter scale range: {:.3f} to {:.3f} mm.".format(
                min(refinement_diameters), max(refinement_diameters)
            )
        )
    if (
        len(planes) != EXPECTED_PLANES
        or other_clipping_names
        or (EXPECTED_REFINEMENTS is not None and len(refinements) != EXPECTED_REFINEMENTS)
        or inverted_planes != EXPECTED_INVERTED_PLANES
        or expected_shape_refinements != len(refinements)
        or not refinements
        or not plane_scale_valid
        or not refinement_scale_valid
    ):
        raise RuntimeError(
            "Persisted ROI verification failed; expected {}/{} planes/refinements, "
            "{} inverted planes, the configured primitive shape, and no legacy "
            "clipping ROIs; found {}/{}, with "
            "{} inverted plane(s) and "
            "{} legacy ROI(s)."
            .format(
                EXPECTED_PLANES,
                EXPECTED_REFINEMENTS if EXPECTED_REFINEMENTS is not None else "nonzero",
                EXPECTED_INVERTED_PLANES,
                len(planes),
                expected_shape_refinements,
                inverted_planes,
                len(other_clipping_names),
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
