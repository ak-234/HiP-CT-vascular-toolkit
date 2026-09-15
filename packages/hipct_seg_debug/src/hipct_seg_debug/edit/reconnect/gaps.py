"""Fill large jumps *inside* a single edge's point list.

Distinct from every other module here: nothing is disconnected in the graph sense.
One edge simply stores two clusters of points with a big empty step between them.
Avizo draws that as a straight line and it looks fine; the SDF pipeline samples
capsules only at stored points, so it produces a fragmented tube with a hole.

``coronary_sdf.centreline_reconnection.bridge_centerline_gaps`` already solves
this well and has its own tests (``_test_bridge_gaps.py``): a C1 cubic Hermite
fill that leaves each side along its own local tangent, gated by both an absolute
floor and a relative "this step is more than five vessel-widths" test so normal
wide sampling on a thin vessel is not mistaken for a gap.

This module exists to surface it on an :class:`~..graphmodel.EditableGraph` --
undoably, and reporting where it acted so the surface can be rebuilt there. The
one hazard is that the underlying function rewrites ``seg["point_ids"]`` in
place, so it is never handed the live graph.
"""

from __future__ import annotations

import numpy as np

from ..history import Patch
from ..sdfconfig import HEADLESS, sdf_config


def find(graph, *, big_jump_ratio: float | None = None, min_gap_um: float | None = None
         ) -> list[dict]:
    """Report the gaps without touching anything.

    Each record is ``{seg_id, i, gap_mm, n_inserted}`` -- ``i`` being the index in
    the segment's point list that the gap follows.
    """
    from ..adapter import Triple

    with sdf_config(HEADLESS) as config:
        from coronary_sdf.centreline_reconnection import bridge_centerline_gaps

        working = Triple(
            nodes=dict(graph.nodes),
            points=dict(graph.points),
            segments=[{**s, "point_ids": list(s["point_ids"])} for s in graph.segments],
        )
        _points, _n, records = bridge_centerline_gaps(
            working.points, working.segments,
            target_spacing_mm=config.DENSIFY_TARGET_SPACING_MM,
            big_jump_ratio=(config.GAP_BIG_JUMP_RATIO if big_jump_ratio is None
                            else big_jump_ratio),
            min_gap_um=(config.CENTERLINE_MAX_GAP_UM if min_gap_um is None else min_gap_um),
            radius_scale=config.RADIUS_SCALE,
            verbose=False,
        )
    return _drop_already_invented(graph, records)


def _drop_already_invented(graph, records) -> list[dict]:
    """Refuse to fill a step that already has one of Avizo's fills on one side.

    Two interpolators in a row is worse than either alone. Where a span is already
    flagged, the honest treatments are the two this package offers -- leave it, or let
    `connect` cut it out so the image can rebuild the join -- and quietly laying a
    Hermite curve over it produces a bridge that is twice invented and no longer looks
    like either.
    """
    from ..interpolation import mask_for_segment

    out = []
    for record in records:
        sid = int(record["seg_id"])
        if not graph.has_segment(sid):
            continue
        invented = mask_for_segment(graph, sid)
        i = int(record["i"])
        if invented.any() and invented[max(i, 0):min(i + 2, invented.size)].any():
            continue
        out.append(record)
    return out


def apply(graph, *, big_jump_ratio: float | None = None, min_gap_um: float | None = None
          ) -> tuple[int, Patch]:
    """Fill every detected gap, as one undo step. Returns ``(n_filled, patch)``.

    The interpolated points are inserted through :meth:`EditableGraph.insert_point`
    rather than by swapping in the rewritten dict, so each one is individually
    reversible and the patch covers exactly where they landed.
    """
    records = find(graph, big_jump_ratio=big_jump_ratio, min_gap_um=min_gap_um)
    if not records:
        return 0, Patch.empty()

    with sdf_config(HEADLESS) as config:
        spacing_um = config.DENSIFY_TARGET_SPACING_MM * 1000.0

    filled = 0
    with graph.batch("bridge centreline gaps"):
        # Descending index order: inserting shifts every later index in the same
        # segment, and going backwards means the recorded indices stay valid.
        for record in sorted(records, key=lambda r: (-r["seg_id"], -r["i"])):
            sid = int(record["seg_id"])
            if not graph.has_segment(sid):
                continue
            i = int(record["i"])
            ids = graph.segment(sid)["point_ids"]
            if not 0 <= i < len(ids) - 1:
                continue
            path, radii = _hermite_fill(graph, sid, i, spacing_um)
            if path is None:
                continue
            for k in range(len(path)):
                graph.insert_point(sid, i + 1 + k, path[k], radii[k])
            filled += 1

    return filled, graph.last_patch


def _hermite_fill(graph, sid: int, i: int, spacing_um: float):
    """Interior points for the gap after index `i`, matching the upstream method.

    One-sided local tangents scaled by the gap length, evaluated as a cubic
    Hermite -- so the fill leaves each side along the direction that side was
    already travelling instead of cutting the corner as a chord would.
    """
    from scipy.interpolate import CubicHermiteSpline

    coords = graph.coords(sid)
    radii = graph.radii(sid)
    if not 0 <= i < len(coords) - 1:
        return None, None

    pa, pb = coords[i], coords[i + 1]
    step = float(np.linalg.norm(pb - pa))
    if step < 1e-9:
        return None, None

    ta = pa - coords[i - 1] if i > 0 else pb - pa
    tb = coords[i + 2] - pb if i + 2 < len(coords) else pb - pa
    ta = ta / max(np.linalg.norm(ta), 1e-9)
    tb = tb / max(np.linalg.norm(tb), 1e-9)

    n_interior = max(int(np.ceil(step / max(spacing_um, 1e-6))) - 1, 1)
    t = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    path = np.column_stack([
        CubicHermiteSpline([0.0, 1.0], [pa[k], pb[k]], [ta[k] * step, tb[k] * step])(t)
        for k in range(3)
    ])
    return path, np.linspace(radii[i], radii[i + 1], n_interior + 2)[1:-1]
