"""Decide *what kind of break this is* before deciding where the path goes.

Four situations look identical in the graph -- two free ends a few hundred
micrometres apart -- and want four different repairs. Running the expensive one
on all of them is not merely wasteful; three of the four answers it would give
are wrong:

``reskeletonise``   the two ends are already in **one mask component**. The lumen
                    is continuous and only the centreline is broken, so the honest
                    repair re-derives the centreline there. Painting a route
                    through voxels that are already foreground invents nothing but
                    also fixes nothing, and it would overwrite a real lumen shape
                    with a synthetic capsule.
``geodesic``        different graph components *and* different mask components.
                    This is the only case that needs a path search, and the only
                    one that may add voxels.
``fragment``        a mask component with no graph on it at all sits between or
                    beyond the ends -- a vessel the skeletoniser dropped. It is
                    skeletonised locally and then joins the chain as an
                    intermediate waypoint or as the target itself.
``unassociated``    an endpoint that lands on no component within reach, or whose
                    two nearest components disagree. Never repaired automatically:
                    the association is the premise of every later step, and a
                    guessed premise produces a confident wrong answer.

The distinction between the first two is the one that matters most, because it is
the one a purely geometric proposer cannot see and the one whose failure mode is
silent -- a mask-connected break repaired as if it were a mask gap looks like a
success in every count anyone checks afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: How far from an endpoint to look for its own component, in local radii. Small:
#: this absorbs frame rounding and a one-voxel-thin collapsed tube, not a genuine
#: separation.
ASSOCIATION_REACH_RADII = 1.5
#: ...with a floor in voxels, because a distal twig's radius can be sub-voxel and
#: 1.5 x nothing is nothing.
ASSOCIATION_REACH_VOXELS = 2.5
#: A mask component smaller than this many voxels is debris unless it is long and
#: thin. Judged against the *local* vessel volume in `fragment_candidates`.
DEBRIS_VOXELS = 12
#: A fragment must be at least this elongated to be a vessel rather than a blob.
MIN_FRAGMENT_ELONGATION = 2.0


@dataclass
class Association:
    """Where one graph endpoint sits in the mask."""

    node: int
    point_um: np.ndarray
    tangent: np.ndarray  # outward unit direction
    radius_um: float
    index_zyx: np.ndarray  # nearest integer segmentation voxel
    component: int  # 0 when nothing was found within reach
    distance_vox: float  # from the endpoint to that component
    ambiguous: bool = False
    note: str = ""
    #: Points on the intact vessel behind this end, for calibration. Supplied only
    #: for a mask end, whose "vessel behind it" is a locally traced axis rather than
    #: a run of graph the calibrator could look up by node id.
    tail_points_um: np.ndarray | None = None

    @property
    def associated(self) -> bool:
        return self.component > 0 and not self.ambiguous


@dataclass
class Fragment:
    """A mask component carrying no centreline: a vessel the skeletoniser dropped."""

    component: int
    voxels: int
    box_zyx: np.ndarray  # (6,) half-open
    centroid_um: np.ndarray
    extent_um: np.ndarray  # principal-axis lengths, longest first
    elongation: float
    plausible: bool
    reason: str = ""


@dataclass
class Classified:
    """One candidate pair, with the repair it turns out to want."""

    kind: str  # "reskeletonise" | "geodesic" | "fragment" | "unassociated"
    source: Association
    target: Association | None = None
    target_segment: int | None = None
    target_index: int | None = None
    #: Set when the far side is a mask free end rather than anything in the graph.
    target_mask_end: Any = None
    fragments: list[Fragment] = field(default_factory=list)
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return self.kind == "unassociated"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        target = self.target.node if self.target is not None else self.target_segment
        return (f"<Classified {self.kind} {self.source.node} -> {target} "
                f"comp {self.source.component}->"
                f"{self.target.component if self.target else '?'}>")


def associate(index, frame, graph, nodes=None, *,
              reach_radii: float = ASSOCIATION_REACH_RADII,
              reach_voxels: float = ASSOCIATION_REACH_VOXELS) -> dict[int, Association]:
    """Map every free end onto the mask component that contains -- or nearly
    contains -- it.

    "Nearly" is deliberate and bounded. The centreline is a derived object sampled
    on a different grid from the mask it came from, so a vertex landing one voxel
    outside its own lumen is routine and means nothing. A vertex landing *four*
    voxels out means the graph and the mask disagree about where this vessel is,
    and that is a finding rather than a rounding error -- so it is reported as
    unassociated rather than snapped to whatever happens to be nearest.
    """
    from ..candidates import endpoint_tangent

    ends = list(nodes) if nodes is not None else graph.endpoints()
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    out: dict[int, Association] = {}

    for node in ends:
        tangent = endpoint_tangent(graph, node)
        if tangent is None:
            continue
        direction, radius = tangent
        point = np.asarray(graph.nodes[node][:3], dtype=np.float64)
        ijk = np.asarray(frame.um_to_seg(point[None, :]), dtype=np.float64)[0]
        zyx = np.round(ijk[::-1]).astype(np.int64)

        reach = max(reach_radii * radius / float(spacing.min()), reach_voxels)
        label, distance = index.nearest_label(int(zyx[0]), int(zyx[1]), int(zyx[2]),
                                              int(np.ceil(reach)))
        association = Association(
            node=node, point_um=point, tangent=direction, radius_um=float(radius),
            index_zyx=zyx, component=int(label), distance_vox=float(distance),
        )
        if label == 0:
            association.note = (
                f"no mask within {reach:.1f} voxels of the endpoint"
            )
        elif distance > reach:
            association.component = 0
            association.note = f"nearest mask is {distance:.1f} voxels away"
        else:
            # A vertex sitting between two components is the conflicting case: it
            # is not obvious which vessel the graph meant, and picking the closer
            # one silently decides a question the operator should see.
            rival = _rival_component(index, zyx, label, int(np.ceil(reach)))
            if rival:
                association.ambiguous = True
                association.note = (
                    f"endpoint lies between mask components {label} and {rival}"
                )
        out[node] = association
    return out


def _rival_component(index, zyx, mine: int, radius: int) -> int:
    """A *different* component also touching the endpoint's neighbourhood.

    Only the immediate shell is searched: two vessels that both pass within a
    couple of voxels of a free end genuinely make its association ambiguous,
    whereas one four voxels away is simply a neighbour.
    """
    r = max(1, min(int(radius), 2))
    z, y, x = (int(v) for v in zyx)
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                label = index.label_at(z + dz, y + dy, x + dx)
                if label and label != mine:
                    return label
    return 0


def classify(pair, associations, index, frame, *, fragments=None) -> Classified:
    """Which of the four repairs does this endpoint pair want?

    `pair` is anything with ``source_node`` and either ``target_node`` or
    ``target_segment``/``target_index`` -- i.e. a :class:`~..candidates.Bridge`
    from the existing geometric proposers, which is where candidates come from.
    """
    source = associations.get(pair.source_node)
    if source is None:
        return Classified(
            kind="unassociated",
            source=Association(pair.source_node, np.zeros(3), np.zeros(3), 0.0,
                               np.zeros(3, np.int64), 0, float("inf")),
            reason="the free end is too short to give a direction",
        )

    target = associations.get(getattr(pair, "target_node", None))
    if not source.associated:
        return Classified(kind="unassociated", source=source, target=target,
                          reason=source.note or "source endpoint is not on the mask")

    mask_end = getattr(pair, "target_mask_end", None)
    if target is None and mask_end is not None:
        target = _associate_mask_end(index, mask_end)
        if not target.associated:
            return Classified(
                kind="unassociated", source=source, target=target,
                target_mask_end=mask_end,
                reason=target.note or "the mask end is not on the mask",
            )

    if target is None and getattr(pair, "target_segment", None) is not None:
        target_assoc = _associate_point(index, frame, pair.coords[-1],
                                        source.radius_um)
        if not target_assoc.associated:
            return Classified(
                kind="unassociated", source=source,
                target_segment=pair.target_segment, target_index=pair.target_index,
                reason=target_assoc.note or "attachment point is not on the mask",
            )
        target = target_assoc

    if target is None:
        return Classified(kind="unassociated", source=source,
                          reason="the target endpoint is too short to give a direction")
    if not target.associated:
        return Classified(kind="unassociated", source=source, target=target,
                          target_segment=getattr(pair, "target_segment", None),
                          target_index=getattr(pair, "target_index", None),
                          reason=target.note or "target endpoint is not on the mask")

    common = dict(
        source=source, target=target,
        target_segment=getattr(pair, "target_segment", None),
        target_index=getattr(pair, "target_index", None),
        target_mask_end=mask_end,
    )
    if source.component == target.component:
        return Classified(
            kind="reskeletonise", reason=(
                f"both ends are inside mask component {source.component}; "
                "the lumen is continuous and only the centreline is broken"
            ),
            metrics={"component": source.component}, **common,
        )

    between = _fragments_between(fragments or [], source, target, index, frame)
    return Classified(
        kind="fragment" if between else "geodesic",
        fragments=between,
        reason=(
            f"mask components {source.component} and {target.component} are distinct"
            + (f", with {len(between)} unskeletonised fragment(s) in between"
               if between else "")
        ),
        metrics={"source_component": source.component,
                 "target_component": target.component},
        **common,
    )


def _associate_mask_end(index, mask_end) -> Association:
    """A mask free end as an :class:`Association`, so the rest of the pipeline
    cannot tell it from a graph endpoint.

    No search is needed and none is done. The tip *is* a foreground voxel -- that
    is how it was found -- so its component is read off directly, and a disagreement
    with the index would mean the mask changed under us rather than that the tip is
    near something. The sentinel node id is ``-1``, the same one a T-junction's
    attachment point uses: neither is a node, and code that asks for one must get
    the same answer in both cases.
    """
    zyx = np.asarray(mask_end.key, dtype=np.int64)
    label = index.label_at(int(zyx[0]), int(zyx[1]), int(zyx[2]))
    out = Association(
        node=-1, point_um=np.asarray(mask_end.point_um, dtype=np.float64),
        tangent=np.asarray(mask_end.tangent, dtype=np.float64),
        radius_um=float(mask_end.radius_um), index_zyx=zyx,
        component=int(label), distance_vox=0.0,
        tail_points_um=np.asarray(mask_end.skeleton_um, dtype=np.float64),
    )
    if not label:
        out.note = "the mask end no longer lands on foreground"
    return out


def _associate_point(index, frame, point_um, radius_um: float) -> Association:
    """Associate a bare world point -- a T-junction's attachment, not a node."""
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    point = np.asarray(point_um, dtype=np.float64).reshape(3)
    ijk = np.asarray(frame.um_to_seg(point[None, :]), dtype=np.float64)[0]
    zyx = np.round(ijk[::-1]).astype(np.int64)
    reach = max(ASSOCIATION_REACH_RADII * radius_um / float(spacing.min()),
                ASSOCIATION_REACH_VOXELS)
    label, distance = index.nearest_label(int(zyx[0]), int(zyx[1]), int(zyx[2]),
                                          int(np.ceil(reach)))
    out = Association(node=-1, point_um=point, tangent=np.zeros(3),
                      radius_um=float(radius_um), index_zyx=zyx,
                      component=int(label), distance_vox=float(distance))
    if not label:
        out.note = f"no mask within {reach:.1f} voxels of the attachment point"
    return out


# ------------------------------------------------------------------- fragments


def graph_components_on_mask(index, frame, graph) -> set[int]:
    """Every mask component that some centreline point falls in.

    The complement is what :func:`fragment_candidates` searches: a component the
    graph has never described. Sampling *all* points rather than endpoints is the
    point -- a fragment is defined by having no centreline anywhere on it, and a
    long vessel whose only graph is in its middle is not a fragment.
    """
    points = np.asarray([p[:3] for p in graph.points.values()], dtype=np.float64)
    if not len(points):
        return set()
    ijk = np.asarray(frame.um_to_seg(points), dtype=np.float64)
    zyx = np.round(ijk[:, ::-1]).astype(np.int64)
    return {int(v) for v in index.labels_at(zyx) if v}


def fragment_candidates(index, frame, graph, *, reach_um: float,
                        near_points_um=None, described=None,
                        min_voxels: int = DEBRIS_VOXELS,
                        min_elongation: float = MIN_FRAGMENT_ELONGATION
                        ) -> list[Fragment]:
    """Mask components with no centreline, near enough to matter and shaped like a vessel.

    Two filters, and they refuse different things. **Size** removes speckle. **Shape**
    removes the compact blobs that survive it -- a calcification, a chunk of
    thresholded myocardium -- because a fragment earns its place in a repair by
    being a piece of *tube*, and a sphere of twenty voxels is not one however close
    to the gap it sits.
    """
    described = graph_components_on_mask(index, frame, graph) if described is None \
        else set(described)
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)  # (x, y, z)
    anchors = None
    if near_points_um is not None:
        anchors = np.asarray(near_points_um, dtype=np.float64).reshape(-1, 3)

    out: list[Fragment] = []
    for label in range(1, index.n + 1):
        if label in described:
            continue
        size = int(index.sizes[label])
        box = index.boxes[label]
        centre_zyx = np.array([(box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0,
                               (box[4] + box[5]) / 2.0])
        centroid = np.asarray(frame.seg_to_um([centre_zyx[::-1]]), dtype=np.float64)[0]
        if anchors is not None:
            if float(np.min(np.linalg.norm(anchors - centroid, axis=1))) > reach_um:
                continue

        extent, elongation = _principal_extent(index, label, spacing, size)
        plausible, reason = True, ""
        if size < min_voxels:
            plausible, reason = False, f"only {size} voxels"
        elif elongation < min_elongation:
            plausible, reason = False, (
                f"not elongated ({elongation:.1f}:1); a blob, not a tube"
            )
        out.append(Fragment(
            component=label, voxels=size, box_zyx=np.asarray(box, dtype=np.int64),
            centroid_um=centroid, extent_um=extent, elongation=elongation,
            plausible=plausible, reason=reason,
        ))
    return out


def _principal_extent(index, label: int, spacing_xyz, size: int,
                      limit: int = 20_000) -> tuple[np.ndarray, float]:
    """Principal-axis lengths in um, longest first, and the elongation ratio.

    From the voxel covariance rather than the bounding box: a diagonal vessel has
    a near-cubic bounding box and would be thrown out as a blob by an
    axis-aligned test, which is exactly the fragment most worth keeping.
    """
    voxels = index.voxels(label, limit=limit)
    if len(voxels) < 2:
        return np.zeros(3), 0.0
    spacing_zyx = np.asarray(spacing_xyz, dtype=np.float64)[::-1]
    points = voxels.astype(np.float64) * spacing_zyx
    centred = points - points.mean(axis=0)
    # Eigenvalues of the covariance are variances; 2*sqrt gives a length-like
    # number comparable between the axes, which is all the ratio needs.
    values = np.linalg.eigvalsh(np.cov(centred.T) if len(points) > 3
                                else np.eye(3) * 1e-12)
    extent = 2.0 * np.sqrt(np.maximum(values, 0.0))[::-1]
    elongation = float(extent[0] / max(extent[1], 1e-9))
    return extent, elongation


def _fragments_between(fragments, source: Association, target: Association,
                       index, frame) -> list[Fragment]:
    """Plausible fragments lying inside the corridor between two ends.

    A generous cylinder rather than the straight segment: the vessel that dropped
    out is the reason the ends are apart in the first place, so it is not obliged
    to sit on the chord between them.
    """
    a, b = source.point_um, target.point_um
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return []
    axis = axis / length
    radius = max(source.radius_um, target.radius_um)
    out = []
    for fragment in fragments:
        if not fragment.plausible:
            continue
        offset = fragment.centroid_um - a
        t = float(np.dot(offset, axis))
        if not -radius <= t <= length + radius:
            continue
        perpendicular = float(np.linalg.norm(offset - t * axis))
        if perpendicular <= max(3.0 * radius, 0.5 * length):
            out.append(fragment)
    return sorted(out, key=lambda f: float(np.dot(f.centroid_um - a, axis)))


def summarise(classified) -> str:
    """A tally by repair kind, with the unassociated ones named."""
    counts: dict[str, int] = {}
    for item in classified:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    if not counts:
        return "no candidate pairs to classify"
    lines = [f"{sum(counts.values())} candidate pair(s) classified"]
    for kind in ("reskeletonise", "geodesic", "fragment", "unassociated"):
        if kind in counts:
            lines.append(f"  {kind:<14} {counts[kind]}")
    for item in classified:
        if item.kind == "unassociated":
            lines.append(f"    node {item.source.node}: {item.reason}")
    return "\n".join(lines)
