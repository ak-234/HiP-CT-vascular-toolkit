"""Interactive correction of HiP-CT segmentations and skeletons.

The viewer in the parent package is read-only: it tells you *where* a
segmentation or skeleton is wrong. This subpackage closes the loop -- edit the
skeleton graph or paint the segmentation, and see the SDF lumen surface
regenerate around the edit.

Layout:

``adapter``       ``SpatialGraph`` <-> the ``coronary_sdf`` ``(nodes, points, segments)`` triple
``sdfconfig``     scoped, restorable overrides for ``coronary_sdf.config``
``history``       ``Patch`` (what an edit touched, and where) plus undo/redo
``graphmodel``    ``EditableGraph`` -- the reversible edit operations
``sdfpatch``      ``SdfSession`` -- local SDF rebuild and mesh splicing
``worker``        off-GUI-thread execution with coalescing

and, for the voxel half:

``maskedit``      ``MaskEdits`` (a sparse correction store) and ``MaskSource``
``paint``         the writable napari layer and its controls
``reskeletonise`` a painted region back into centreline, spliced into the graph
``skeletonise``   the whole mask into a fresh centreline graph
``lattice``       decode the RLE lattice into an array analysis tools accept

The one number that governs the design: a full ``generate_sdf_surface`` run is
5-10 minutes, while the same code over a 10 mm box is sub-second. Every edit
therefore reports the box it touched, and only that box is rebuilt.
"""

from __future__ import annotations

__all__ = [
    "Patch",
    "History",
    "Triple",
    "from_spatial_graph",
    "to_spatial_graph",
    "read_triple",
    "sdf_config",
    "HEADLESS",
    "PREVIEW",
]

from .adapter import Triple, from_spatial_graph, read_triple, to_spatial_graph
from .history import History, Patch
from .sdfconfig import HEADLESS, PREVIEW, sdf_config
