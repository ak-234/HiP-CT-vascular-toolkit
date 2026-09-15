"""Add visible P### labels to existing coronary outlet planes in the GUI.

Run this file from the Simpleware X-2025.06 Scripting tab after opening the SIP.
The headless ConsoleSimpleware process does not initialise the Annotations tool.
"""

import re

from simpleware.scripting import App, Doc, IntVector, RealPoint3D, RealSize


OUTLET_PREFIX = "COR_OUTLET_"
LABEL_PREFIX = "COR_PLANE_ID_"


def _xyz(vector):
    return float(vector.GetX()), float(vector.GetY()), float(vector.GetZ())


def main():
    document = App.GetDocument()
    try:
        annotations = document.GetAnnotations()
    except Exception as exc:
        raise RuntimeError(
            "The Annotations tool is not available in this Simpleware GUI document"
        ) from exc

    for annotation in list(annotations.GetAll()):
        if annotation.GetName().startswith(LABEL_PREFIX):
            annotations.RemoveAnnotation(annotation)

    planes = sorted(
        (
            roi for roi in document.GetRegionOfInterestVolumes(Doc.Clipping)
            if roi.GetName().startswith(OUTLET_PREFIX)
        ),
        key=lambda roi: roi.GetName(),
    )
    created = 0
    for fallback_ordinal, roi in enumerate(planes, start=1):
        name = roi.GetName()
        match = re.match(r"COR_OUTLET_(\d+)_", name)
        ordinal = int(match.group(1)) if match else fallback_ordinal
        centre = _xyz(roi.GetShape().GetCentre())
        scale = _xyz(roi.GetShape().GetScale())
        offset = max(0.20, 1.25 * max(abs(scale[0]), abs(scale[1])))
        corner = (
            centre[0] + offset,
            centre[1] + offset,
            centre[2] + offset,
        )
        annotation = annotations.AddTextBox(
            LABEL_PREFIX + name,
            Doc.OrientationXY,
            IntVector(),
            RealPoint3D(*centre),
            RealPoint3D(*corner),
            RealSize(48.0, 18.0),
            "P{:03d}".format(ordinal),
            1.0,
            False,
            True,
        )
        annotation.SetDrawAnchor(True)
        annotation.SetDrawBackground(True)
        annotation.SetDrawBorder(True)
        annotation.SetScaleWithZoom(False)
        created += 1

    document.AddLogComment(
        "Coronary plane labels",
        "Created {} P### labels for COR_OUTLET clipping planes.".format(created),
    )
    return created


if __name__ == "__main__":
    main()
