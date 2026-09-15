"""Collapsed-vessel radius outlier detection and correction.

Port of the ``Outlier correction/`` folder. Detection (``detect``) is pure
numpy and always available; the oblique cross-section resampling (``oblique``)
uses the optional ``[image]`` extra (scikit-image / tifffile) and is imported
lazily so importing this package never requires those dependencies.
"""

from skeleton_analysis.outlier.detect import (
    along_segment_outliers,
    correct_along_segment_thickness,
    detect_collapsed_segments,
    filloutliers_nearest,
    isoutlier_percentiles,
    matlab_prctile,
)
from skeleton_analysis.outlier.correct import (
    replace_thickness_values,
    write_corrected_graph,
    apply_manual_plane_selection,
)
from skeleton_analysis.outlier.viz3d import segment_bbox, show_segment_volume

__all__ = [
    "matlab_prctile",
    "isoutlier_percentiles",
    "filloutliers_nearest",
    "along_segment_outliers",
    "correct_along_segment_thickness",
    "detect_collapsed_segments",
    "replace_thickness_values",
    "write_corrected_graph",
    "apply_manual_plane_selection",
    "segment_bbox",
    "show_segment_volume",
]
