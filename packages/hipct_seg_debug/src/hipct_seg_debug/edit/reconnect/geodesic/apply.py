"""Commit a decided repair: mask, graph and provenance, as one undoable step.

The ordering is not cosmetic. A repair makes two claims -- *these voxels are
lumen* and *a vessel runs here* -- and they have to agree, so the mask is edited
first and the centreline is then **re-derived from the edited mask** rather than
taken from the route that motivated it. The route is a path through a cost field;
the centreline is the medial axis of the lumen that now exists. They are close but
they are not the same object, and writing the route onto the graph while writing a
transported cross-section into the mask would leave a graph that is not the
skeleton of its own segmentation.

Everything lands inside one ``graph.batch``, so a repair is one press of undo. The
mask edit is reversible through the same transaction: :class:`~..maskedit.MaskEdits`
records the voxels this repair set, and :func:`revert` clears exactly those.

**The paired-output rule lives here in spirit and in the CLI in fact.** A run that
writes a repaired graph without its repaired mask produces two files that disagree
about where the vessels are, and nothing downstream can tell which is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....rle_write import write_lattice
from ..candidates import resample_by_arclength

#: Per-edge provenance. 0 is the file's own edges, so an unedited graph reads as
#: "original" without anything having to write a column for it.
ORIGINAL, GEOMETRY, DPC, GEODESIC, RESKELETONISED = 0, 1, 2, 3, 4
ORIGIN_NAMES = {
    ORIGINAL: "original", GEOMETRY: "geometry", DPC: "dpc",
    GEODESIC: "geodesic", RESKELETONISED: "reskeletonised",
}
#: Amira field names for the provenance columns. Capitalised to match the
#: convention of the fields Avizo itself writes.
ORIGIN_FIELD = "ReconnectionOrigin"
SCORE_FIELD = "RouteScore"
REVIEWED_FIELD = "RouteReviewed"


@dataclass
class Applied:
    """What one commit actually did."""

    candidate: object
    segments: list = field(default_factory=list)
    voxels_added: int = 0
    planes_touched: list = field(default_factory=list)
    reskeletonised: object = None
    origin: int = GEODESIC
    ok: bool = True
    reason: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"not applied: {self.reason}"
        bits = [f"{ORIGIN_NAMES[self.origin]}"]
        if self.voxels_added:
            bits.append(f"+{self.voxels_added:,} mask voxel(s)")
        if self.segments:
            bits.append(f"+{len(self.segments)} segment(s)")
        if self.reskeletonised is not None:
            bits.append(self.reskeletonised.describe())
        return ", ".join(bits)


def apply_plan(graph, plan, source, frame, *, reviewed=None,
               label: str = "geodesic reconnect", reskeletonise: bool = True,
               progress=None) -> list:
    """Commit every accepted candidate, as one undo step.

    `source` is a :class:`~..maskedit.MaskSource` -- the mask edits go into its
    store, which is what every other reader in the toolkit already composites, so
    a repair is visible to the viewer, to a later re-skeletonisation and to the
    writer without anything being notified.

    `reviewed` is the set of candidates an operator has approved; they are applied
    alongside the automatically accepted ones and are marked as reviewed in the
    provenance, which is the distinction an audit needs to make later.
    """
    approved = list(plan.accepted())
    reviewed_set = {id(c) for c in (reviewed or ())}
    approved.extend(c for c in (reviewed or ()) if id(c) not in {id(a) for a in approved})

    out: list[Applied] = []
    with graph.batch(label):
        for n, candidate in enumerate(approved, 1):
            result = apply_one(
                graph, candidate, source, frame,
                reviewed=id(candidate) in reviewed_set,
                reskeletonise=reskeletonise,
            )
            out.append(result)
            if progress is not None:
                progress(n, len(approved), result)
    return out


def apply_one(graph, candidate, source, frame, *, reviewed: bool = False,
              reskeletonise: bool = True) -> Applied:
    """Commit one candidate. Must be called inside a ``graph.batch``."""
    if candidate.kind == "reskeletonise":
        return _apply_reskeletonise(graph, candidate, source, frame, reviewed)
    if candidate.route is None or not len(candidate.route.path_zyx):
        return Applied(candidate, ok=False, reason="the candidate carries no route")

    voxels = _completion_voxels(candidate)
    added, planes = paint(source, voxels)

    if reskeletonise:
        result = _reskeletonise_route(graph, candidate, source, frame)
        if result is not None and result.applied:
            _stamp_patch(graph, result, GEODESIC, candidate, reviewed)
            return Applied(candidate, voxels_added=added, planes_touched=planes,
                           reskeletonised=result, origin=GEODESIC)

    # The re-derivation either was not asked for or found nothing to trace -- which
    # happens when the completion is a bare connectivity core one voxel wide. The
    # route itself is then the honest centreline, and it is welded directly.
    sid = weld(graph, candidate, frame, reviewed=reviewed)
    return Applied(candidate, segments=[] if sid is None else [sid],
                   voxels_added=added, planes_touched=planes, origin=GEODESIC,
                   ok=sid is not None,
                   reason="" if sid is not None else "the route could not be welded")


def _apply_reskeletonise(graph, candidate, source, frame, reviewed: bool) -> Applied:
    """A break inside one mask component: re-derive, add no voxels.

    The whole repair is the re-derivation, and it is the one case where the mask
    is guaranteed untouched -- worth asserting rather than assuming, because a
    completion accidentally reaching this path would inflate a lumen that was
    never broken.
    """
    result = _reskeletonise_route(graph, candidate, source, frame)
    if result is None or not result.applied:
        return Applied(candidate, ok=False, origin=RESKELETONISED,
                       reason=(result.reason if result is not None
                               else "no box could be built"))
    _stamp_patch(graph, result, RESKELETONISED, candidate, reviewed)
    return Applied(candidate, reskeletonised=result, origin=RESKELETONISED)


# ----------------------------------------------------------------------- the mask


def _completion_voxels(candidate) -> np.ndarray:
    """The voxels this candidate would add: its transported shape, or its core."""
    if candidate.completion is not None and len(candidate.completion.voxels_zyx):
        return candidate.completion.voxels_zyx
    return np.asarray(candidate.route.path_zyx, dtype=np.int64)


def paint(source, voxels_zyx) -> tuple[int, list[int]]:
    """Set `voxels_zyx` in the mask edit store. Returns ``(n_set, planes)``.

    Only voxels that are not already foreground are written, so the store records
    the repair rather than a restatement of the mask -- which is what makes
    :func:`revert` able to remove exactly what was added.
    """
    voxels = np.asarray(voxels_zyx, dtype=np.int64).reshape(-1, 3)
    if not len(voxels):
        return 0, []
    edits = source.edits
    added = 0
    planes: list[int] = []
    for z in np.unique(voxels[:, 0]):
        rows = voxels[voxels[:, 0] == z]
        plane = np.asarray(source.slice_z(int(z)))
        fresh = plane[rows[:, 1], rows[:, 2]] == 0
        if not fresh.any():
            continue
        target = rows[fresh]
        added += edits.set_plane(int(z), target[:, 1], target[:, 2],
                                 np.ones(len(target), dtype=np.uint8))
        planes.append(int(z))
    return added, planes


def revert(source, voxels_zyx) -> int:
    """Undo a :func:`paint`, by clearing those voxels from the edit store."""
    voxels = np.asarray(voxels_zyx, dtype=np.int64).reshape(-1, 3)
    if not len(voxels):
        return 0
    cleared = 0
    for z in np.unique(voxels[:, 0]):
        rows = voxels[voxels[:, 0] == z]
        cleared += source.edits.set_plane(int(z), rows[:, 1], rows[:, 2],
                                          np.zeros(len(rows), dtype=np.uint8))
    return cleared


def write_segmentation(path, source, frame, *, progress=None) -> dict:
    """Write the repaired mask as a fresh ``HxByteRLE`` lattice.

    Decodes the whole volume, which is the honest cost of the format: the codec is
    a single sequential stream, so a byte written near the start shifts everything
    after it and there is no such thing as patching one plane in place.

    The bounding box is taken from the frame rather than copied from the input
    header, so a mask written here describes its own geometry and cannot silently
    inherit a stale one.
    """
    from ...lattice import decode_volume

    volume = decode_volume(source, stride=1,
                           progress=(lambda d, t, e: progress(d, t))
                           if progress is not None else None)
    origin = np.asarray(frame.seg_origin, dtype=np.float64)
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    dims = np.asarray(frame.seg_dims, dtype=np.int64)  # (nx, ny, nz)
    hi = origin + (dims - 1) * spacing
    bbox = np.empty(6, dtype=np.float64)
    bbox[0::2], bbox[1::2] = origin, hi
    return write_lattice(path, volume, bbox)


# ---------------------------------------------------------------------- the graph


def _reskeletonise_route(graph, candidate, source, frame):
    """Re-derive the centreline in a box around the repair and splice it in."""
    from ...reskeletonise import reskeletonise_box

    box = _box_for(candidate, frame)
    if box is None:
        return None
    mode = "replace" if candidate.kind == "reskeletonise" else "add"
    try:
        return reskeletonise_box(graph, source, frame, box, mode=mode,
                                 grow_um=_grow_for(candidate))
    except Exception as exc:  # noqa: BLE001 - fall back to welding the route
        from ...reskeletonise import ReskeletoniseReport

        return ReskeletoniseReport(mode=mode, box_um=box, applied=False,
                                   reason=f"re-skeletonisation failed: {exc}")


def _grow_for(candidate):
    """How far past the repair the local trace should follow the skeleton.

    ``None`` -- the default, three local radii -- for everything except a mask end.
    There the whole point of the repair is that a run of lumen carries no
    centreline, and a growth of three radii would describe the first fraction of it
    and stop, leaving most of the lobe exactly as undescribed as it started. The
    lobe's own traced length is what the growth has to cover.
    """
    mask_end = getattr(candidate.classified, "target_mask_end", None)
    if mask_end is None or not len(mask_end.skeleton_um):
        return None
    axis = np.asarray(mask_end.skeleton_um, dtype=np.float64).reshape(-1, 3)
    along = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum()) \
        if len(axis) > 1 else 0.0
    return max(along, 3.0 * float(mask_end.radius_um)) + float(mask_end.radius_um)


def _box_for(candidate, frame):
    """The world box a repair touched, padded by a vessel width.

    A mask end contributes its whole traced axis, not just its tip. The lobe is the
    reason this repair exists and it carries no centreline: leaving it outside the
    box would re-derive up to its tip and stop, welding onto a point with nothing
    behind it and leaving the lumen that motivated the repair as undescribed as it
    was before.
    """
    points = []
    if candidate.route is not None and len(candidate.route.path_zyx):
        points.append(candidate.route.path_um(frame))
    classified = candidate.classified
    points.append(np.asarray(classified.source.point_um, dtype=np.float64)[None, :])
    if classified.target is not None:
        points.append(np.asarray(classified.target.point_um, dtype=np.float64)[None, :])
    mask_end = getattr(classified, "target_mask_end", None)
    if mask_end is not None and len(mask_end.skeleton_um):
        points.append(np.asarray(mask_end.skeleton_um, dtype=np.float64).reshape(-1, 3))
    stacked = np.vstack([p for p in points if len(p)])
    if not len(stacked):
        return None
    pad = 2.0 * max(float(classified.source.radius_um), 1.0)
    return np.array([stacked.min(axis=0) - pad, stacked.max(axis=0) + pad])


def weld(graph, candidate, frame, *, reviewed: bool = False):
    """Add the route to the graph as a segment, with its provenance.

    Used when re-skeletonisation found nothing to trace, which is the case where
    the completion is a bare connectivity core. Splits the parent first for a
    T-junction, exactly as :func:`~..candidates.apply_bridges` does and for the
    same reason: the point id survives a split, the segment id does not.
    """
    from ..candidates import _split_for_tjunction

    classified = candidate.classified
    source_node = classified.source.node
    if source_node not in graph.nodes:
        return None

    if classified.target_segment is not None:
        proposal = candidate.proposal
        if proposal is None:
            return None
        target_node = _split_for_tjunction(graph, proposal)
        if target_node is None:
            return None
    elif getattr(classified, "target_mask_end", None) is not None:
        # There is no node here yet -- that is the whole point of a mask end. The
        # route did reach real lumen, so the segment gets a node at the tip and the
        # lobe behind it stays undescribed until something re-derives it. That is
        # weaker than the re-skeletonised path and it is the fallback, not the
        # default: `apply_one` only reaches here when the re-derivation found
        # nothing to trace.
        target_node = graph.add_node(classified.target.point_um)
    else:
        target_node = classified.target.node
        if target_node not in graph.nodes:
            return None

    path = candidate.route.path_um(frame)
    spacing = max(0.9 * float(classified.source.radius_um), 1.0)
    coords = resample_by_arclength(path, spacing)
    if len(coords) < 2:
        return None
    # The route's own start and end are voxel centres; the graph's nodes are where
    # the vessel actually ends. Snapping avoids a sub-voxel stub at each join.
    coords[0] = classified.source.point_um
    coords[-1] = graph.nodes[target_node][:3]

    radii = _radii_for(candidate, classified, len(coords))
    return graph.add_segment(
        source_node, target_node, coords, radii,
        attrs=provenance_attrs(GEODESIC, candidate, reviewed),
    )


def _radii_for(candidate, classified, n: int) -> np.ndarray:
    """Perimeter-equivalent radii along the welded route.

    From the transported cross-sections when they measured, so the graph carries
    the *anatomical* calibre of a collapsed vessel rather than the half-width of
    the slit the mask records. See :func:`..shape.graph_radii` for why those are
    two different numbers and both are wanted.
    """
    from . import shape as shape_mod

    source_r = float(classified.source.radius_um)
    target_r = float(classified.target.radius_um) if classified.target else source_r
    if candidate.completion is not None:
        return shape_mod.graph_radii(candidate.completion, source_r, target_r, n)
    return np.linspace(source_r, target_r, n)


# ------------------------------------------------------------------- provenance


def provenance_attrs(origin: int, candidate, reviewed: bool) -> dict:
    """The per-edge columns that say where an edge came from.

    Written on every edge this package creates. An edge with no such column reads
    as ``ORIGINAL``, which is exactly right for a file that has never been through
    here -- so the columns cost nothing on an unedited graph and are unambiguous
    on an edited one.
    """
    return {
        ORIGIN_FIELD: int(origin),
        SCORE_FIELD: float(getattr(candidate, "confidence", 0.0)),
        REVIEWED_FIELD: int(bool(reviewed)),
        "reconnected": 1.0,  # kept for compatibility with the geometry-only path
        "strahler": 1,
    }


def _stamp_patch(graph, result, origin: int, candidate, reviewed: bool) -> None:
    """Mark the segments a re-skeletonisation created with this repair's provenance.

    ``reskeletonise_box`` splices through ``EditableGraph`` and does not know about
    this package's columns, so they are applied afterwards to whatever it added.
    The patch records which segments those were.
    """
    patch = getattr(result, "patch", None)
    if patch is None:
        return
    attrs = provenance_attrs(origin, candidate, reviewed)
    for sid in patch.seg_ids:
        if not graph.has_segment(sid):
            continue
        if graph.segment(sid).get(ORIGIN_FIELD, ORIGINAL) != ORIGINAL:
            continue  # an earlier repair already claimed it
        graph.set_segment_attrs(sid, attrs)


def origin_counts(graph) -> dict[str, int]:
    """How many edges of each origin the graph now holds -- an audit round-trip check."""
    counts: dict[str, int] = {}
    for segment in graph.segments:
        name = ORIGIN_NAMES.get(int(segment.get(ORIGIN_FIELD, ORIGINAL)), "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts
