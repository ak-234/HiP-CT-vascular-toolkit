"""Attach a free end onto the *side* of another vessel.

The case the existing toolkit cannot express: it joins endpoint to endpoint only,
so a branch whose parent was skeletonised as one unbroken run has nowhere to
attach. This is also the paper's "Type 3 / branch occurrence" case, where the
predicted continuous centreline does not contain the branch the disconnected one
belongs to.

The target is a *point along* a segment, so applying the bridge has to split that
segment into two and create a node -- which is why these candidates carry a point
id rather than a node id, and why :func:`~.candidates.apply_bridges` re-resolves
through it: an earlier split on the same vessel invalidates the segment id but
never the point id.
"""

from __future__ import annotations

import numpy as np

from .candidates import (
    CONE_ANGLE_DEG,
    CONE_LENGTH_FACTOR,
    RADIUS_RATIO_MAX,
    TORTUOSITY_MAX,
    Bridge,
    _new_stats,
    endpoint_tangent,
    gate_geometry,
    hermite_path,
    resample_by_arclength,
)

# A daughter joining a parent should not be thicker than it. Slightly above 1 to
# absorb radius noise at the tip, where the estimate is worst.
MAX_DAUGHTER_RATIO = 1.25


def propose(
    graph,
    *,
    cone_angle_deg: float = CONE_ANGLE_DEG,
    cone_length_factor: float = CONE_LENGTH_FACTOR,
    radius_ratio_max: float = RADIUS_RATIO_MAX,
    tortuosity_max: float = TORTUOSITY_MAX,
    max_daughter_ratio: float = MAX_DAUGHTER_RATIO,
    same_component: bool = False,
    min_edge_distance: int = 3,
    spacing_um: float | None = None,
    keep_rejected: bool = False,
    stats: dict | None = None,
    backbone_nodes: set[int] | None = None,
    target_backbone_only: bool = False,
) -> list[Bridge]:
    """Propose bridges from free ends onto the interior of other segments.

    `min_edge_distance` keeps the attachment away from a segment's own ends,
    where the right answer is an end-to-end join and where splitting would make a
    degenerate stub.

    `stats` is filled as in :func:`~.endpoints.propose`, and for the same reason:
    the two prunes here -- the free end's own vessel, and the same-component test --
    both fire before a :class:`Bridge` exists.
    """
    from scipy.spatial import cKDTree

    counts = _new_stats(stats, "tjunction")
    ends = graph.endpoints()
    counts["ends_considered"] = len(ends)
    if not ends:
        return []

    # One cloud of every interior centreline point, so the search is a ball query
    # rather than a scan over segments.
    pids: list[int] = []
    owner: list[int] = []
    index_in_seg: list[int] = []
    for seg in graph.segments:
        ids = seg["point_ids"]
        for k in range(min_edge_distance, max(len(ids) - min_edge_distance, min_edge_distance)):
            pids.append(ids[k])
            owner.append(seg["id"])
            index_in_seg.append(k)
    if not pids:
        return []

    cloud = np.array([graph.points[p][:3] for p in pids], dtype=np.float64)
    cloud_r = np.array([graph.points[p][3] for p in pids], dtype=np.float64)
    tree = cKDTree(cloud)
    node_of_component = _component_of_node(graph)
    backbone_nodes = set(backbone_nodes or ())

    out: list[Bridge] = []
    reaches: list[float] = []
    for source in sorted(ends, key=lambda n: -_radius_at(graph, n)):
        if target_backbone_only and source in backbone_nodes:
            continue
        tangent = endpoint_tangent(graph, source)
        if tangent is None:
            counts["ends_without_tangent"] += 1
            continue
        counts["ends_with_tangent"] += 1
        t0, r0 = tangent
        p0 = np.asarray(graph.nodes[source][:3], dtype=np.float64)
        own = graph.node_segments(source)
        reach = cone_length_factor * r0
        reaches.append(reach)
        best: Bridge | None = None

        for j in tree.query_ball_point(p0, reach):
            sid = owner[j]
            counts["pairs_in_reach"] += 1
            if sid in own:
                counts["pruned_own_segment"] += 1
                continue  # never attach a vessel to itself
            if target_backbone_only:
                target_seg = graph.segment(sid)
                if not ({target_seg["node1"], target_seg["node2"]} & backbone_nodes):
                    continue
            if not same_component:
                seg = graph.segment(sid)
                if node_of_component.get(seg["node1"]) == node_of_component.get(source):
                    counts["pruned_same_component"] += 1
                    continue

            p1 = cloud[j]
            r1 = float(max(cloud_r[j], 1e-6))
            if r0 > max_daughter_ratio * r1:
                if keep_rejected:
                    out.append(
                        Bridge("tjunction", source, np.array([p0, p1]), np.array([r0, r1]),
                               target_segment=sid, target_index=index_in_seg[j])
                        .reject("the branch is thicker than the vessel it would join")
                    )
                continue

            # Arrive perpendicular to the trunk rather than along it: a side
            # branch leaves its parent at an angle, and aiming the Hermite down
            # the parent's own axis would make the join run parallel and fuse.
            t1 = _approach_direction(graph, sid, index_in_seg[j], p0, p1)
            path = hermite_path(p0, t0, p1, t1, 32)
            bridge = Bridge(
                kind="tjunction", source_node=source, coords=path,
                radii=np.linspace(r0, min(r0, r1), len(path)),
                target_segment=sid, target_index=index_in_seg[j],
                reconnection_type=3,
            )
            bridge.metrics["target_point_id"] = pids[j]
            gate_geometry(
                bridge, r0, r1,
                cone_angle_deg=cone_angle_deg,
                cone_length_factor=cone_length_factor,
                radius_ratio_max=radius_ratio_max,
                tortuosity_max=tortuosity_max,
                source_tangent=t0,
            )
            bridge.score = _score(bridge, r0, r1)
            if not bridge.accepted:
                if keep_rejected:
                    out.append(bridge)
                continue
            if best is None or bridge.score > best.score:
                best = bridge

        if best is not None:
            spacing = spacing_um if spacing_um is not None else max(0.9 * r0, 1.0)
            coords = resample_by_arclength(best.coords, spacing)
            best.coords = coords
            r_end = min(best.metrics["r_source"], best.metrics["r_target"])
            best.radii = np.linspace(best.metrics["r_source"], r_end, len(coords))
            out.append(best)

    if reaches:
        counts["reach_um_median"] = float(np.median(reaches))
    # `nearest_cross_component_um` is deliberately left unset: `endpoints.propose`
    # already reports the distance between the pieces, and repeating it here against
    # a different point set would give a second, slightly different number for the
    # same fact.
    return out


def _approach_direction(graph, sid: int, index: int, p0: np.ndarray, p1: np.ndarray
                        ) -> np.ndarray:
    """A unit direction arriving at the trunk from the branch's side.

    Takes the component of the incoming chord perpendicular to the trunk's local
    tangent, so the bridge meets the parent across its axis rather than sliding
    along it.
    """
    coords = graph.coords(sid)
    lo = max(index - 1, 0)
    hi = min(index + 1, len(coords) - 1)
    axis = coords[hi] - coords[lo]
    norm = float(np.linalg.norm(axis))
    chord = p1 - p0
    if norm < 1e-9:
        n = float(np.linalg.norm(chord))
        return chord / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    axis = axis / norm
    perpendicular = chord - np.dot(chord, axis) * axis
    n = float(np.linalg.norm(perpendicular))
    if n < 1e-9:
        n2 = float(np.linalg.norm(chord))
        return chord / n2 if n2 > 1e-9 else axis
    return perpendicular / n


def _radius_at(graph, node: int) -> float:
    info = endpoint_tangent(graph, node)
    return info[1] if info is not None else 0.0


def _score(bridge: Bridge, r0: float, r1: float) -> float:
    span = max(bridge.span_um, 1e-6)
    reach = max(bridge.metrics.get("reach_um", span), 1e-6)
    closeness = 1.0 - min(span / reach, 1.0)
    straightness = 1.0 / max(bridge.tortuosity, 1.0)
    # A plausible daughter is thinner than its parent, so reward that rather than
    # rewarding a radius match the way an end-to-end join does.
    plausible = float(np.clip(r0 / max(r1, 1e-9), 0.0, 1.0))
    cone = bridge.metrics.get("cone_deg")
    aim = 1.0 - min(cone / 180.0, 1.0) if cone is not None else 0.5
    return float(0.4 * closeness + 0.2 * straightness + 0.2 * plausible + 0.2 * aim)


def _component_of_node(graph) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, seg_ids in enumerate(graph.components()):
        for sid in seg_ids:
            seg = graph.segment(sid)
            out[seg["node1"]] = i
            out[seg["node2"]] = i
    return out
