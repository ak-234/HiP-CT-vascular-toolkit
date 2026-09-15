"""Reconnecting what the skeletoniser left apart.

A HiP-CT coronary skeleton arrives fragmented: thin distal vessels drop below the
segmentation threshold, a branch is cut where contrast fades, and Avizo's
centreline tree comes out as several disconnected components. The surface
pipeline then meshes each fragment separately, which is fatal for CFD -- flow
cannot cross a gap.

Four kinds of break, four modules:

``gaps``          a jump *inside* one edge's point list, with no intermediate
                  points. Delegates to ``coronary_sdf.bridge_centerline_gaps``,
                  which already does this well.
``endpoints``     two free ends that should be one vessel. Geometric gates ported
                  from the upstream editing toolkit, with the search rebuilt on
                  a KD-tree.
``tjunction``     a free end that belongs on the *side* of another vessel. Needs
                  a new node in the middle of that vessel, so it goes through
                  ``EditableGraph.split_segment``.
``dpc``           the DPC walk from Med. Image Anal. 2025 (arXiv:2504.01597):
                  instead of interpolating between two endpoints, walk through
                  the image from one to the other, one voxel at a time, scoring
                  each step by distance, centreline probability and direction.

``segmentation`` repairs the voxel mask rather than the graph.

The geometric modules are cheap and propose; ``dpc`` is expensive and decides.
Everything produces :class:`~.candidates.Bridge` objects and nothing applies
itself -- :func:`~.candidates.apply_bridges` is a separate, explicit step, so a
proposal can be drawn in the viewer and accepted or rejected before it changes
anything.
"""

from __future__ import annotations

from .candidates import Bridge, apply_bridges, summarise

__all__ = ["Bridge", "apply_bridges", "summarise"]
