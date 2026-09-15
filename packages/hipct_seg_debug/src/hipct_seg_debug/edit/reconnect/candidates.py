"""What a proposed reconnection looks like, and the geometry every proposer shares.

A :class:`Bridge` is a *proposal*. Nothing here mutates a graph; applying is a
separate explicit step so a candidate can be drawn, scored and rejected first.
That matters because every gate in this package is a heuristic tuned on one
dataset, and a reconnection that is wrong is worse than one that is missing: a
spurious bridge silently reroutes flow in the CFD run downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# Defaults ported from the production values in
# the earlier skeleton-graph-editing-toolkit script
# (reconnect_disconnected_segments_in_spatial_graph.py)
# (its main() at :1266-1271), which were tuned on HiP-CT coronary data.
CONE_ANGLE_DEG = 50.0
CONE_LENGTH_FACTOR = 15.0  # max reach, in multiples of the endpoint radius
RADIUS_RATIO_MAX = 5.0
TORTUOSITY_MAX = 1.8
MIN_RADIUS_UM = 0.5


@dataclass
class Bridge:
    """One proposed reconnection, with the evidence for and against it."""

    kind: str  # "endpoint" | "tjunction" | "gap"
    source_node: int
    coords: np.ndarray  # (N, 3) um, source -> target
    radii: np.ndarray  # (N,) um

    target_node: int | None = None
    target_segment: int | None = None  # for a T-junction: the vessel to split
    target_index: int | None = None  # ...and where along its point list
    #: A free end of the *mask* that carries no centreline, from
    #: :mod:`.geodesic.lobes`. The third kind of target, and the only one that is
    #: not already in the graph: nothing downstream may assume a target exists as
    #: a node or a segment.
    target_mask_end: Any = None
    reconnection_type: int | None = None  # paper Type 1, 2, or 3

    score: float = 0.0
    accepted: bool = True
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def length_um(self) -> float:
        return float(np.linalg.norm(np.diff(self.coords, axis=0), axis=1).sum())

    @property
    def span_um(self) -> float:
        return float(np.linalg.norm(self.coords[-1] - self.coords[0]))

    @property
    def tortuosity(self) -> float:
        span = self.span_um
        return self.length_um / span if span > 1e-9 else float("inf")

    def reject(self, reason: str) -> "Bridge":
        self.accepted = False
        self.reason = reason
        return self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.target_node is not None:
            target = f"node {self.target_node}"
        elif self.target_mask_end is not None:
            target = f"mask end {self.target_mask_end.key}"
        else:
            target = f"seg {self.target_segment}[{self.target_index}]"
        state = "ok" if self.accepted else f"rejected: {self.reason}"
        return (
            f"<Bridge {self.kind} {self.source_node} -> {target} "
            f"{self.span_um:.0f}um score={self.score:.3f} "
            f"type={self.reconnection_type or '?'} {state}>"
        )


def endpoint_tangent(graph, node: int, max_reach_um: float = 10.0 * 40.0
                     ) -> tuple[np.ndarray, float] | None:
    """Outward unit direction at a degree-1 node, and the local radius.

    Walks back along the segment until a point at least a few samples away is
    found, because consecutive centreline points are often near-coincident and
    differencing two of those gives a direction made of noise. The toolkit does
    the same thing with a fixed ladder of offsets (k = 5,4,3,2,1) and gives up if
    nothing lies within 10 voxels; here the ladder is replaced by "walk until far
    enough", which behaves the same on dense runs and better on sparse ones.
    """
    segs = graph.node_segments(node)
    if len(segs) != 1:
        return None
    sid = next(iter(segs))
    coords = graph.coords(sid)
    radii = graph.radii(sid)
    if len(coords) < 2:
        return None

    at_start = graph.segment(sid)["node1"] == node
    if not at_start:
        coords = coords[::-1]
        radii = radii[::-1]

    tip = coords[0]
    for i in range(1, len(coords)):
        delta = tip - coords[i]
        dist = float(np.linalg.norm(delta))
        if dist > 1e-6 and (dist >= 1.0 or i == len(coords) - 1):
            radius = float(np.median(radii[: i + 1]))
            return delta / dist, max(radius, MIN_RADIUS_UM)
    return None


def hermite_path(
    p0: np.ndarray, t0: np.ndarray, p1: np.ndarray, t1: np.ndarray, n: int
) -> np.ndarray:
    """A C1 cubic Hermite from `p0` to `p1` leaving along `t0` and arriving along `t1`.

    Leaving along the existing tangent is the point: a straight chord between two
    endpoints puts a visible kink at both ends, and the SDF turns that kink into a
    crease or a self-intersection where the swept tube folds over itself.
    """
    span = float(np.linalg.norm(p1 - p0))
    s = np.linspace(0.0, 1.0, max(int(n), 2))[:, None]
    m0 = np.asarray(t0, dtype=np.float64) * span
    m1 = np.asarray(t1, dtype=np.float64) * span
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def resample_by_arclength(coords: np.ndarray, spacing_um: float) -> np.ndarray:
    """Re-space a polyline at roughly `spacing_um`, keeping both endpoints exact."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if len(coords) < 2:
        return coords
    step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = float(arc[-1])
    if total <= 1e-9:
        return coords[[0, -1]]
    n = max(int(np.ceil(total / max(spacing_um, 1e-6))) + 1, 2)
    want = np.linspace(0.0, total, n)
    out = np.column_stack([np.interp(want, arc, coords[:, k]) for k in range(3)])
    out[0], out[-1] = coords[0], coords[-1]
    return out


def gate_geometry(
    bridge: Bridge,
    r_source: float,
    r_target: float,
    *,
    cone_angle_deg: float = CONE_ANGLE_DEG,
    cone_length_factor: float = CONE_LENGTH_FACTOR,
    radius_ratio_max: float = RADIUS_RATIO_MAX,
    tortuosity_max: float = TORTUOSITY_MAX,
    source_tangent: np.ndarray | None = None,
) -> Bridge:
    """Apply the four standard gates, recording which one failed.

    Each is cheap and each rules out a different way of being wrong:

    * **cone angle** -- the target must lie roughly ahead of where the vessel was
      already going, not off to the side;
    * **reach** -- proportional to radius, because a 2 mm trunk may plausibly
      bridge much further than a 40 um twig;
    * **radius ratio** -- a trunk does not continue as a capillary;
    * **tortuosity** -- if the path has to wander to get there, it is not the
      same vessel.
    """
    span = bridge.span_um
    r_min, r_max = min(r_source, r_target), max(r_source, r_target)
    bridge.metrics.update(
        span_um=span, r_source=r_source, r_target=r_target,
        radius_ratio=r_max / max(r_min, 1e-9), tortuosity=bridge.tortuosity,
    )

    if source_tangent is not None and span > 1e-9:
        direction = (bridge.coords[-1] - bridge.coords[0]) / span
        cos = float(np.clip(np.dot(source_tangent, direction), -1.0, 1.0))
        bridge.metrics["cone_deg"] = float(np.degrees(np.arccos(cos)))
        if bridge.metrics["cone_deg"] > cone_angle_deg:
            return bridge.reject(f"outside the {cone_angle_deg:g}deg search cone")

    reach = cone_length_factor * r_source
    bridge.metrics["reach_um"] = reach
    if span > reach:
        return bridge.reject(f"further than {cone_length_factor:g} x radius")
    if r_max / max(r_min, 1e-9) > radius_ratio_max:
        return bridge.reject(f"radius ratio above {radius_ratio_max:g}")
    if bridge.tortuosity > tortuosity_max:
        return bridge.reject(f"tortuosity above {tortuosity_max:g}")
    return bridge


def apply_bridges(graph, bridges: Iterable[Bridge], label: str = "reconnect") -> list[int]:
    """Add the accepted bridges to `graph` as one undo step.

    Returns the new segment ids. A T-junction bridge splits its target segment
    first, which is why this cannot be a simple loop of ``add_segment``: the split
    invalidates the segment id every later candidate on that vessel was holding,
    so those are re-resolved through the point id, which survives the split.
    """
    accepted = [b for b in bridges if b.accepted]
    if not accepted:
        return []

    created: list[int] = []
    with graph.batch(label):
        for bridge in accepted:
            source = bridge.source_node
            if source not in graph.nodes:
                continue

            if bridge.target_node is not None:
                target = bridge.target_node
                if target not in graph.nodes:
                    continue
            else:
                target = _split_for_tjunction(graph, bridge)
                if target is None:
                    continue

            coords = np.asarray(bridge.coords, dtype=np.float64).reshape(-1, 3)
            radii = np.asarray(bridge.radii, dtype=np.float64).ravel()
            if len(coords) < 2 or len(radii) != len(coords):
                continue
            created.append(
                graph.add_segment(source, target, coords, radii,
                                  attrs={"strahler": 1, "reconnected": 1.0})
            )
    return created


def _split_for_tjunction(graph, bridge: Bridge) -> int | None:
    """Turn the T-junction target into a node, tolerating an earlier split.

    ``target_segment`` may already have been split by a previous bridge, so the
    stored id can be stale. The *point* id is not: resolve through it.
    """
    pid = bridge.metrics.get("target_point_id")
    if pid is None:
        return None
    sid = graph.segment_of_point().get(pid)
    if sid is None:
        return None
    ids = graph.segment(sid)["point_ids"]
    try:
        index = ids.index(pid)
    except ValueError:
        return None
    if not 0 < index < len(ids) - 1:
        # The attachment point landed on an existing node; use it directly.
        seg = graph.segment(sid)
        return seg["node1"] if index == 0 else seg["node2"]
    node, _a, _b = graph.split_segment(sid, index)
    return node


# ------------------------------------------------------------------- reporting

#: Counters every proposer fills, so an empty result can still be explained. The
#: two that matter are `pairs_in_reach` and `pruned_same_component`: both of those
#: ways of losing a pair happen *before* a `Bridge` exists, so neither is visible
#: in the candidate list however hard you look at it.
_STAT_KEYS = (
    "ends_considered",
    "ends_with_tangent",
    "ends_without_tangent",
    "pairs_in_reach",
    "pruned_same_component",
    "pruned_own_segment",
)


def _new_stats(stats: dict | None, kind: str) -> dict:
    """Zero the counters in place, so a caller's dict is filled even on early return."""
    out = stats if stats is not None else {}
    out.clear()
    out["kind"] = kind
    for key in _STAT_KEYS:
        out[key] = 0
    return out


def nearest_cross_component(positions: np.ndarray, labels: Sequence[int]) -> float | None:
    """Closest distance between two points carrying different labels, in um.

    Answers "how far apart are the pieces?", which is the question behind an empty
    result: a reach of 2.6 mm cannot bridge a 5.5 mm separation however the cone is
    tuned, and no amount of staring at a candidate list reveals that.

    Exact, via one KD-tree per label over the complement. That is
    ``O(n_labels * n log n)``, so it gives up rather than crawl on a graph that has
    been shattered into thousands of pieces -- and returns ``None``, which the
    report prints as "not computed" instead of quietly showing a wrong number.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    tags = np.asarray(labels)
    if len(points) < 2:
        return None
    unique = np.unique(tags)
    if len(unique) < 2 or len(unique) * len(points) > 200_000:
        return None

    best = np.inf
    for label in unique:
        mine = tags == label
        others = points[~mine]
        if not len(others):
            continue
        distance, _ = cKDTree(others).query(points[mine], k=1)
        best = min(best, float(np.min(distance)))
    return None if not np.isfinite(best) else best


def summarise(bridges: Sequence[Bridge], stats: dict | None = None) -> str:
    """A one-screen report: what was proposed, what survived, and why not.

    `stats` is a proposer's counter dict. It is what makes a zero informative --
    "161 free ends, 0 pairs within reach" is a finding, and the bare "no candidates"
    it replaces reads as a broken tool.
    """
    if not bridges:
        return _explain_empty(stats)
    ok = [b for b in bridges if b.accepted]
    lines = [f"{len(ok)} accepted of {len(bridges)} candidates"]
    by_kind: dict[str, int] = {}
    for b in ok:
        by_kind[b.kind] = by_kind.get(b.kind, 0) + 1
    for kind, n in sorted(by_kind.items()):
        lines.append(f"  {kind:<10} {n}")
    reasons: dict[str, int] = {}
    for b in bridges:
        if not b.accepted:
            reasons[b.reason] = reasons.get(b.reason, 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"  rejected: {reason} ({n})")
    lines.extend(_pruned_lines(stats))
    return "\n".join(lines)


def _explain_empty(stats: dict | None) -> str:
    """Say which stage consumed everything, rather than just "none"."""
    if not stats:
        return "no reconnection candidates"

    ends = stats.get("ends_considered", 0)
    lines = [f"no reconnection candidates ({ends} free end(s) examined)"]
    without = stats.get("ends_without_tangent", 0)
    if without:
        lines.append(f"  {without} free end(s) too short to give a direction")

    in_reach = stats.get("pairs_in_reach", 0)
    if not in_reach:
        reach = stats.get("reach_um_median")
        detail = f" (median reach {reach / 1000.0:.2f} mm)" if reach else ""
        lines.append(f"  nothing within reach of any free end{detail}")
        nearest = stats.get("nearest_cross_component_um")
        if nearest is not None:
            lines.append(
                f"  the nearest two pieces are {nearest / 1000.0:.2f} mm apart -- "
                "raise --reach-factor if that gap is real"
            )
    else:
        lines.append(f"  {in_reach} pair(s) within reach")
    lines.extend(_pruned_lines(stats))
    return "\n".join(lines)


def _pruned_lines(stats: dict | None) -> list[str]:
    """The prunes that never became candidates, so they never became visible."""
    if not stats:
        return []
    lines = []
    same = stats.get("pruned_same_component", 0)
    if same:
        lines.append(
            f"  pruned: {same} pair(s) inside one component "
            "(pass --same-component to allow those)"
        )
    own = stats.get("pruned_own_segment", 0)
    if own:
        lines.append(f"  pruned: {own} point(s) on the free end's own vessel")
    return lines
