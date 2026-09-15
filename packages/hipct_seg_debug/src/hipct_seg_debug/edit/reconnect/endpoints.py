"""End-to-end reconnection: two free ends that should be one vessel.

The gates are ported from the earlier skeleton-graph-editing-toolkit script
``reconnect_disconnected_segments_in_spatial_graph.py``,
whose thresholds were tuned on HiP-CT coronary data and are worth keeping. Three
things about that implementation are not worth keeping:

* ``reconnect_end_points`` is **defined twice in that file** (at line 301 and
  line 805). Python keeps the second, so the newer Hermite-interpolating version
  is dead code and the Catmull-Rom one is what actually ran. Here the Hermite
  version is the one implemented, because leaving each end along its own tangent
  is what stops the SDF creasing at the join.
* the search is a nested Python loop over all endpoint pairs with no spatial
  index. On a graph with thousands of free ends that is the whole runtime; a
  ``cKDTree`` ball query makes it linear in the number of *nearby* pairs.
* an endpoint may be used at most once, in a single greedy pass. That is retained
  here -- a free end really does have only one continuation -- but the ordering is
  made explicit and the rejected pairs are reported rather than discarded, so a
  near-miss can be inspected instead of silently vanishing.
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
    nearest_cross_component,
    resample_by_arclength,
)


def propose(
    graph,
    *,
    cone_angle_deg: float = CONE_ANGLE_DEG,
    cone_length_factor: float = CONE_LENGTH_FACTOR,
    radius_ratio_max: float = RADIUS_RATIO_MAX,
    tortuosity_max: float = TORTUOSITY_MAX,
    same_component: bool = False,
    spacing_um: float | None = None,
    keep_rejected: bool = False,
    stats: dict | None = None,
    reconnection_type: int | None = None,
    backbone_nodes: set[int] | None = None,
) -> list[Bridge]:
    """Propose end-to-end bridges between degree-1 nodes.

    `same_component` allows joins inside one component (which create a loop);
    off by default because a coronary tree is a tree, and a loop is nearly always
    a skeletonisation artefact rather than a real anastomosis.

    `stats`, if given, is filled with what the search *considered* -- not just what
    it proposed. Two of the ways a pair dies happen before a :class:`Bridge` exists
    (nothing within reach, and the same-component prune), so without this an empty
    result is indistinguishable from a graph with no free ends at all.
    """
    from scipy.spatial import cKDTree

    counts = _new_stats(stats, "endpoint")
    ends = graph.endpoints()
    counts["ends_considered"] = len(ends)
    if len(ends) < 2:
        return []

    info: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for node in ends:
        tangent = endpoint_tangent(graph, node)
        if tangent is None:
            continue
        direction, radius = tangent
        info[node] = (np.asarray(graph.nodes[node][:3], dtype=np.float64), direction, radius)
    counts["ends_with_tangent"] = len(info)
    counts["ends_without_tangent"] = len(ends) - len(info)
    if len(info) < 2:
        return []

    node_of_component = _component_of_node(graph)
    backbone_nodes = set(backbone_nodes or ())
    ids = list(info)
    positions = np.array([info[n][0] for n in ids])
    tree = cKDTree(positions)
    counts["reach_um_median"] = float(
        np.median([cone_length_factor * info[n][2] for n in ids])
    )

    # Thickest first: a trunk should claim its continuation before a twig gets to
    # propose a competing one for the same free end.
    order = sorted(ids, key=lambda n: -info[n][2])
    used: set[int] = set()
    out: list[Bridge] = []

    for source in order:
        if source in used:
            continue
        p0, t0, r0 = info[source]
        reach = cone_length_factor * r0
        best: Bridge | None = None

        for j in tree.query_ball_point(p0, reach):
            target = ids[j]
            if target == source or target in used:
                continue
            counts["pairs_in_reach"] += 1
            if not same_component and node_of_component.get(source) == node_of_component.get(target):
                counts["pruned_same_component"] += 1
                continue
            pair_type = 2 if ((source in backbone_nodes) ^ (target in backbone_nodes)) else 1
            if reconnection_type is not None and pair_type != reconnection_type:
                continue
            p1, t1, r1 = info[target]

            # Both ends must point at each other: -t1 is the target's inward
            # direction, which is where the bridge should arrive.
            path = hermite_path(p0, t0, p1, -t1, 32)
            bridge = Bridge(
                kind="endpoint", source_node=source, target_node=target,
                coords=path, radii=np.linspace(r0, r1, len(path)),
                reconnection_type=pair_type,
            )
            gate_geometry(
                bridge, r0, r1,
                cone_angle_deg=cone_angle_deg,
                cone_length_factor=cone_length_factor,
                radius_ratio_max=radius_ratio_max,
                tortuosity_max=tortuosity_max,
                source_tangent=t0,
            )
            # Also require the *target* to be facing back, so two ends that merely
            # happen to be close but run parallel are not joined across.
            if bridge.accepted:
                span = bridge.span_um
                if span > 1e-9:
                    approach = (p0 - p1) / span
                    cos_back = float(np.clip(np.dot(t1, approach), -1.0, 1.0))
                    bridge.metrics["target_cone_deg"] = float(
                        np.degrees(np.arccos(cos_back))
                    )
                    if bridge.metrics["target_cone_deg"] > cone_angle_deg:
                        bridge.reject("the target end faces away")

            bridge.score = _score(bridge, r0, r1)
            if not bridge.accepted:
                if keep_rejected:
                    out.append(bridge)
                continue
            if best is None or bridge.score > best.score:
                best = bridge

        if best is not None:
            spacing = spacing_um if spacing_um is not None else max(
                0.9 * min(best.metrics["r_source"], best.metrics["r_target"]), 1.0
            )
            coords = resample_by_arclength(best.coords, spacing)
            best.coords = coords
            best.radii = np.linspace(
                best.metrics["r_source"], best.metrics["r_target"], len(coords)
            )
            used.add(best.source_node)
            used.add(best.target_node)
            out.append(best)

    # Only worth the KD-tree work when the answer is "nothing": it is the number
    # that says whether the reach or the geometry was the binding constraint.
    if not any(b.accepted for b in out):
        counts["nearest_cross_component_um"] = nearest_cross_component(
            positions, [node_of_component.get(n, -1) for n in ids]
        )
    return out


def _score(bridge: Bridge, r0: float, r1: float) -> float:
    """Higher is better. Prefers short, straight, radius-matched joins."""
    span = max(bridge.span_um, 1e-6)
    reach = max(bridge.metrics.get("reach_um", span), 1e-6)
    closeness = 1.0 - min(span / reach, 1.0)
    straightness = 1.0 / max(bridge.tortuosity, 1.0)
    match = min(r0, r1) / max(r0, r1, 1e-9)
    cone = bridge.metrics.get("cone_deg")
    aim = 1.0 - min(cone / 180.0, 1.0) if cone is not None else 0.5
    return float(0.35 * closeness + 0.25 * straightness + 0.25 * match + 0.15 * aim)


def _component_of_node(graph) -> dict[int, int]:
    """``{node id: component index}``, so cross-component joins can be preferred."""
    out: dict[int, int] = {}
    for i, seg_ids in enumerate(graph.components()):
        for sid in seg_ids:
            seg = graph.segment(sid)
            out[seg["node1"]] = i
            out[seg["node2"]] = i
    return out
