"""Clean the graph first, then measure each free end properly.

Every proposal cone, every seed direction and every calibration tail the rest of
this package uses starts from the free ends of the graph. The stock estimate of an
end's direction, :func:`~..candidates.endpoint_tangent`, differences the tip against
the first centreline point at least a micrometre behind it -- on a skeletonised,
voxel-staircased centreline that is a direction made largely of noise, and the
same noise then sets which pairs are even considered.

So two things happen before anything is proposed:

1. **Segmentation-constrained refinement** (:mod:`...centreline_refine`), on the
   whole graph. It never changes topology or radii, it holds every terminal node
   fixed -- the tips stay exactly where they were -- and it fits the geometry
   behind each tip to the lumen it actually sits in. Radii are then remeasured on
   the refined geometry, exactly as ``refine-centreline`` does.
2. **End profiling.** For each degree-1 node, the tangent from a radius-scaled
   quadratic fit (:func:`~....crosssection.robust_edge_tangents`) and the
   cross-section of *this* component cut perpendicular to it. The section's
   principal axes give the major axis of the collapsed ellipse and its flatness,
   which is what sizes the look-ahead and orients the metric near the seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Centreline points either side of a tip that count as its intact tail.
TAIL_POINTS = 12
#: Arclength window, in local radii, for the end tangent fit.
TANGENT_WINDOW_RADII = 4.0


@dataclass
class EndProfile:
    """What is known about one free end after refinement."""

    node: int
    point_um: np.ndarray
    tangent: np.ndarray  # outward unit direction, world (x, y, z)
    radius_um: float  # median graph radius over the tail
    u: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in-plane axes
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    major_axis: np.ndarray | None = None  # unit, world; None when unmeasurable
    major_um: float = 0.0  # half-lengths of the fitted ellipse
    minor_um: float = 0.0
    flatness: float = 1.0
    r_perimeter_um: float = 0.0
    r_area_um: float = 0.0
    component: int = 0
    tail_points_um: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    valid: bool = True
    reason: str = ""

    @property
    def normal(self) -> np.ndarray | None:
        """The collapse direction: in-plane, perpendicular to the major axis."""
        if self.major_axis is None:
            return None
        n = np.cross(self.tangent, self.major_axis)
        norm = float(np.linalg.norm(n))
        return n / norm if norm > 1e-9 else None

    def describe(self) -> str:
        bits = [f"node {self.node}", f"r={self.radius_um:.0f}um"]
        if self.major_axis is not None:
            bits.append(f"ellipse {self.major_um:.0f}x{self.minor_um:.0f}um "
                        f"flat={self.flatness:.1f}")
        else:
            bits.append("section unmeasured" + (f" ({self.reason})" if self.reason else ""))
        if self.component:
            bits.append(f"component {self.component}")
        return ", ".join(bits)


# --------------------------------------------------------------------- cleaning


def clean(graph, frame, labels, *, method: str = "centroid-coherent",
          strength: float = 0.1, max_iterations: int = 25, workers: int = 1,
          max_half: int = 256, progress=None) -> dict:
    """Refine the whole graph against the segmentation and remeasure its radii.

    Mirrors ``refine-centreline`` without the file handling. ``method="none"``
    does nothing and says so, which is what a caller that has already refined
    its input passes. Returns the refinement report as a plain dict with the
    radius-measurement summary attached.
    """
    from ... import radius_perimeter as rp
    from ...centreline_refine import METHODS, refine

    if method not in METHODS:
        raise ValueError(f"unknown refinement method: {method}")
    if method == "none":
        return {"method": "none", "moved_points": 0, "skipped": True}

    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    report = refine(
        graph, frame, labels, method=method, strength=strength,
        max_iterations=max_iterations, max_half=max_half, workers=workers,
        progress=progress,
    ).to_dict()
    if report["moved_points"]:
        result = rp.measure_radii(graph, frame, labels, workers=workers,
                                  max_half=max_half)
        rp.apply_radii(graph, result)
        report["radii_require_remeasurement"] = False
        report["radius_measurement"] = result.describe()
    # The same displacement column ``refine-centreline`` writes, so a graph that
    # came through here reads the same in the viewer.
    moved = graph.triple.point_attrs.setdefault("centreline_displacement_um", {})
    graph.triple.point_attr_dtypes["centreline_displacement_um"] = "float"
    for sid in graph.segment_ids():
        d = np.linalg.norm(graph.coords(sid) - before[sid], axis=1)
        moved.update(dict(zip(graph.segment(sid)["point_ids"], map(float, d))))
    return report


# -------------------------------------------------------------------- profiling


def _tail(graph, node: int, n: int = TAIL_POINTS):
    """Coordinates and radii of the segment at `node`, ordered tip first."""
    segs = graph.node_segments(node)
    if len(segs) != 1:
        return None
    sid = next(iter(segs))
    coords = np.asarray(graph.coords(sid), dtype=np.float64)
    radii = np.asarray(graph.radii(sid), dtype=np.float64)
    if len(coords) < 2:
        return None
    if graph.segment(sid)["node1"] != node:
        coords, radii = coords[::-1], radii[::-1]
    return sid, coords[:n], radii[:n]


def end_tangent(graph, node: int, spacing_um: float, *,
                window_radii: float = TANGENT_WINDOW_RADII
                ) -> tuple[np.ndarray, float] | None:
    """Outward unit tangent at a free end from a radius-scaled quadratic fit."""
    from ....crosssection import robust_edge_tangents

    tail = _tail(graph, node)
    if tail is None:
        return None
    _sid, coords, radii = tail
    finite = radii[np.isfinite(radii) & (radii > 0)]
    radius = float(np.median(finite)) if len(finite) else float(spacing_um)
    tangents = robust_edge_tangents(coords, np.full(len(coords), radius),
                                    spacing_um=spacing_um, window_radii=window_radii)
    direction = tangents[0]
    # The fit's sign is arbitrary; outward means "away from the rest of the segment".
    inward = coords[min(len(coords) - 1, 3)] - coords[0]
    if float(np.dot(direction, inward)) > 0:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9 or not np.isfinite(norm):
        return None
    return direction / norm, radius


def _ellipse_axes(section):
    """Principal half-lengths and in-plane major axis of a section's lumen."""
    inside = section.sdf < 0
    if int(inside.sum()) < 2:
        return None
    ii, jj = np.nonzero(inside)
    pts = np.column_stack([ii, jj]).astype(np.float64) * section.step_um
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts / len(pts)
    values, vectors = np.linalg.eigh(cov)
    # Second moments of a uniform ellipse are a^2/4, b^2/4.
    minor = 2.0 * float(np.sqrt(max(values[0], 0.0)))
    major = 2.0 * float(np.sqrt(max(values[1], 0.0)))
    # One voxel of extent still means half a step of physical width.
    minor = max(minor, 0.5 * section.step_um)
    major = max(major, minor)
    axis = vectors[:, 1]
    world = axis[0] * section.u + axis[1] * section.v
    norm = float(np.linalg.norm(world))
    if norm < 1e-9:
        return None
    return major, minor, world / norm


def profile_end(graph, frame, index, node: int, *, tangents=None,
                window_radii: float = TANGENT_WINDOW_RADII) -> EndProfile | None:
    """Everything the connector wants to know about one free end."""
    from ..geodesic import shape

    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    sp = float(spacing.min())
    if tangents is not None and node in tangents:
        direction, radius = tangents[node]
        direction = np.asarray(direction, dtype=np.float64)
        estimate = (direction / max(np.linalg.norm(direction), 1e-9), float(radius))
    else:
        estimate = end_tangent(graph, node, sp, window_radii=window_radii)
    if estimate is None:
        return None
    direction, radius = estimate
    tail = _tail(graph, node)
    coords = tail[1] if tail is not None else np.empty((0, 3))
    point = np.asarray(graph.nodes[node][:3], dtype=np.float64)
    ijk = np.asarray(frame.um_to_seg(point[None, :]), dtype=np.float64)[0]
    zyx = np.round(ijk[::-1]).astype(np.int64)
    reach = int(np.ceil(max(2.0 * radius / sp, 2.0)))
    component, _distance = index.nearest_label(int(zyx[0]), int(zyx[1]), int(zyx[2]),
                                               reach)
    profile = EndProfile(
        node=node, point_um=point, tangent=direction, radius_um=float(radius),
        component=int(component), tail_points_um=coords[::-1].copy(),
    )
    if not component:
        profile.reason = "the tip is not on the mask"
        return profile

    # Cut a little behind the tip: the terminal voxel of a thinning is often a
    # single voxel and says nothing about the section the vessel actually has.
    centre = point - direction * min(radius, 2.0 * sp)
    try:
        section = shape.extract_section(index, frame, centre, direction,
                                        int(component), float(radius))
    except Exception as exc:  # noqa: BLE001 - a failed cut must not lose the end
        profile.reason = f"section failed: {exc}"
        return profile
    profile.u, profile.v = section.u, section.v
    if not section.valid:
        profile.reason = section.reason or "section is empty"
        return profile
    profile.r_perimeter_um = section.r_perimeter
    profile.r_area_um = section.r_area
    profile.flatness = section.flatness
    axes = _ellipse_axes(section)
    if axes is None:
        profile.reason = "section too small to orient"
        return profile
    profile.major_um, profile.minor_um, profile.major_axis = axes
    return profile


def profile_ends(graph, frame, index, *, nodes=None, tangents=None,
                 window_radii: float = TANGENT_WINDOW_RADII) -> dict[int, EndProfile]:
    """:func:`profile_end` for every degree-1 node (or the `nodes` given)."""
    ends = list(nodes) if nodes is not None else graph.endpoints()
    out: dict[int, EndProfile] = {}
    for node in ends:
        profile = profile_end(graph, frame, index, node, tangents=tangents,
                              window_radii=window_radii)
        if profile is not None:
            out[node] = profile
    return out


def as_tangents(profiles) -> dict[int, tuple[np.ndarray, float]]:
    """The ``{node: (direction, radius_um)}`` the proposers accept."""
    return {node: (p.tangent, p.radius_um) for node, p in profiles.items()}


def summarise(profiles, report=None) -> str:
    lines = []
    if report:
        if report.get("skipped"):
            lines.append("refinement skipped (method none)")
        else:
            lines.append(
                f"refined with {report.get('method')}: {report.get('moved_points', 0)} "
                f"point(s) moved in {report.get('iterations', 0)} iteration(s)"
                + (", converged" if report.get("converged") else "")
                + f"; outside edges {report.get('outside_edges_before', 0)} -> "
                f"{report.get('outside_edges_after', 0)}"
            )
            if report.get("radius_measurement"):
                lines.append(f"  radii: {report['radius_measurement']}")
    measured = [p for p in profiles.values() if p.major_axis is not None]
    lines.append(f"{len(profiles)} free end(s) profiled, {len(measured)} with a "
                 f"measured section")
    if measured:
        flat = np.array([p.flatness for p in measured])
        lines.append(f"  flatness median {np.median(flat):.2f}, max {flat.max():.2f}")
    return "\n".join(lines)
