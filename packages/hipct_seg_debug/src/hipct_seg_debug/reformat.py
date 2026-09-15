"""Look at a vessel along its own axis, instead of through the acquisition grid.

Every other view in this package reads the data the way it was acquired: the slice
browser (:mod:`~.viewer2d`) shows raw ``(slice, row, col)``, and the only oblique
sampling that exists is mask-side and nearest-neighbour
(:class:`~.crosssection._PlaneSampler`). A coronary crosses slices at an arbitrary
angle, so an axial slice cuts it obliquely and every cross-section it shows is a
smear whose apparent calibre depends on the angle rather than on the vessel.

This module builds the other view: a stack of square images sampled on the planes
**perpendicular to the centreline**, one per step along it, so scrolling the stack
walks down the vessel and each image is a true cross-section. The raw stack is
sampled by cubic B-spline interpolation, the segmentation nearest-neighbour, and both
on exactly the same planes.

Four things decide whether such a stack means anything.

**The grid must not ask for more than the data has.** A plane's pixel pitch is a free
choice, and choosing it finer than the voxel does not reveal anything -- it enlarges
the interpolation kernel. The ``native`` mode exists for that reason and is the
default: it pins the pitch to one raw voxel and lets the *half-width* follow from the
frame size, so one output pixel is one voxel whatever the vessel is doing. The other
three modes derive the pitch from a half-width instead, which is what lets a small
radius -- or the curvature clamp below -- turn into magnification. Whichever is in
use, :attr:`PlaneGeometry.oversampling` states the factor rather than leaving a soft
picture unexplained.

**The frame must not twist.** The two in-plane axes have to be transported along the
centreline with no rotation about the tangent, or the image spins as you scroll and
a feature that stays put looks like it is moving. :func:`frames` uses the
rotation-minimizing frame in :mod:`~.edit.reconnect.geodesic.shape`, which is the
only twist-free frame in the tree; ``crosssection._plane_axes`` and
``viewer3d.radius_circle_polydata`` both pick an arbitrary seed per point, which is
fine for drawing one circle and wrong for a stack.

**The planes must not intersect each other.** This is the constraint that is easy to
miss and impossible to see once it has happened. Every plane perpendicular to a
locally circular arc of radius ``R`` passes through the arc's centre of curvature, at
distance ``R``. So a stack of half-width ``h`` is free of self-intersection between
neighbouring planes exactly when ``h < R``: past that the planes fold through each
other, the same tissue appears in two images, and the tissue that should have been
between them appears in none. Discretely, with turn ``theta_i`` over arclength step
``ds_i``, ``R_i = ds_i / theta_i`` and the condition is ``h * theta_i < ds_i``.

A voxel-derived centreline turns a median 19.5 degrees between consecutive points
(measured on LADAF-2024-28, see ``skeleton_optimise.plane_normals``), which puts
``R`` at a fraction of a voxel and makes *every* usable half-width illegal. So the
centreline is resampled to uniform arclength and smoothed until the bound is met,
and if it still is not, the half-width is clamped. Both remedies are reported with
numbers -- a reformat that quietly halved its own field of view is a reformat that
misreports calibre.

**The criterion is local.** It certifies neighbouring planes, and cannot see a
hairpin whose two limbs come back within ``2h`` of each other. :func:`planes_disjoint`
is the direct global check, and it is advisory: no amount of smoothing fixes a
hairpin, and the only honest remedy is a narrower ``h``.

Units are micrometres and world ``(x, y, z)`` throughout, as everywhere else in this
package; :class:`~.frame.WorldFrame` is the only thing that knows about the two grids
underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Default half-width, in multiples of the local centreline radius. Four radii puts
#: the vessel wall about a quarter of the way in from the edge, which leaves room to
#: see what the lumen is sitting next to without wasting the frame on it.
HALF_WIDTH_RADII = 4.0

#: Odd, so there is an exact centre pixel and the centreline lands on it rather than
#: between two.
DEFAULT_SIZE_PX = 129

#: Keep the half-width this far inside the curvature bound. ``R`` is estimated from a
#: discrete polyline, so meeting the bound exactly is meeting it to within the
#: estimate's own error.
DEFAULT_SAFETY = 0.8

#: How far the smoothing window grows per attempt, and how many attempts.
SMOOTH_GROWTH = 1.6
MAX_SMOOTH_ITERS = 8

#: Two limbs closer than this many half-widths, but far apart along the vessel, are
#: reported by :func:`planes_disjoint`.
APPROACH_FACTOR = 1.0

#: Near-parallel planes do not cross inside the field of view even when they are
#: close, so they are not worth reporting.
APPROACH_PARALLEL_COS = 0.985  # ~10 degrees

#: Spline order for the raw greyscale. Cubic rather than linear, measured on a real
#: slice by coarsening 2x and restoring: RMSE 173 against linear's 235 (**-26%**), and
#: 73% of the edge energy recovered against 55%. Quintic gains a further 2% for three
#: times the cost, which does not earn its place.
#:
#: It matters here more than it would elsewhere because the default grid is
#: *magnifying*: at the median graph radius a radius-mode plane samples at ~15 um/px
#: against a 33 um voxel, so what the eye reads is largely the interpolation kernel --
#: and linear's is a triangle.
DEFAULT_ORDER = 3

#: Block padding, in voxels, per spline order. **Not cosmetic above order 1.**
#:
#: ``map_coordinates`` prefilters for ``order >= 2``, and the spline prefilter is an IIR
#: filter -- it is *not local*, so a per-block prefilter differs from a whole-volume one
#: near the block edge and the difference lands as a seam. Measured against a
#: whole-volume prefilter at order 3: pad 1 gives 4.3e-3, pad 4 gives 8.4e-5, pad 8
#: gives 4.5e-7, and pad 16 is exact. ``test_chunking_does_not_change_the_answer``
#: asserts exact equality across memory budgets and is the guard for this.
#:
#: One genuine exception, documented rather than papered over: a block clipped at the
#: true volume boundary cannot carry its pad, so its prefilter differs there. That is at
#: the edge of the data, where there is nothing to recover anyway.
SPLINE_PAD = {0: 1, 1: 1, 2: 12, 3: 16, 4: 24, 5: 32}


class ReformatError(RuntimeError):
    """The requested reformat cannot be built as asked."""


# --------------------------------------------------------------------------- #
# Assembling a path out of a segment selection
# --------------------------------------------------------------------------- #


@dataclass
class Chain:
    """One ordered, oriented run of segments through the graph."""

    #: ``(segment id, reversed)`` in walk order. ``reversed`` means the segment's own
    #: point run, which is always node1 -> node2, is traversed backwards.
    steps: tuple
    start_node: int
    end_node: int
    length_um: float
    notes: tuple = ()

    @property
    def segment_ids(self) -> tuple:
        return tuple(sid for sid, _rev in self.steps)


def chain_segments(graph, segment_ids) -> tuple[list, list]:
    """Order a set of segment ids into oriented runs. ``(chains, notes)``.

    ``graph`` is an :class:`~.edit.graphmodel.EditableGraph`. What decides the
    topology is each node's degree **within the selection**, not its degree in the
    tree: picking two segments either side of a bifurcation is a chain, even though
    the node between them has degree 3 in the graph.

    Rules, in order:

    * a node with selection-degree greater than 2 is a branch *inside* the selection.
      There is no total order through it, so the run is split there and the node is
      named in the notes rather than guessed at.
    * selection-degree-1 nodes are the ends of an open chain.
    * a component with no degree-1 node is a cycle; it is broken at its lowest
      segment id and said so.

    Orientation follows from ``EditableGraph.coords``, which always runs node1 ->
    node2 (``graphmodel.py:157``), so a segment entered at its ``node2`` is reversed.
    Chains come back longest first, which is the order the caller wants when it is
    going to use one and mention the rest.
    """
    selected = [sid for sid in dict.fromkeys(segment_ids) if graph.has_segment(sid)]
    notes: list[str] = []
    missing = len(list(segment_ids)) - len(selected)
    if missing > 0:
        notes.append(f"{missing} selected segment(s) are no longer in the graph")
    if not selected:
        return [], notes

    # Incidence restricted to the selection. `node_segments` is the graph's own index
    # (graphmodel.py:151), intersected rather than rebuilt.
    at_node: dict[int, list[int]] = {}
    for sid in selected:
        seg = graph.segment(sid)
        for key in ("node1", "node2"):
            at_node.setdefault(seg[key], []).append(sid)

    branch_nodes = {nid for nid, sids in at_node.items() if len(sids) > 2}
    for nid in sorted(branch_nodes):
        notes.append(
            f"node {nid} joins {len(at_node[nid])} selected segments; the chain is "
            f"split there rather than guessing which way to go through it"
        )

    remaining = set(selected)
    chains: list[Chain] = []

    def walk(node: int, sid: int) -> Chain:
        steps: list[tuple[int, bool]] = []
        here, cur = node, sid
        while True:
            seg = graph.segment(cur)
            n1, n2 = seg["node1"], seg["node2"]
            # Entered at node1 -> traversed forwards; entered at node2 -> reversed.
            steps.append((cur, here != n1))
            remaining.discard(cur)
            other = n2 if here == n1 else n1
            if other == here:  # a self-loop has no far end to continue from
                break
            nxt = [s for s in at_node.get(other, ()) if s != cur and s in remaining]
            # Stop at a branch or a free end; carry on only through a plain join.
            if other in branch_nodes or len(nxt) != 1:
                here = other
                break
            here, cur = other, nxt[0]
        end = here if here != node else (graph.segment(steps[-1][0])["node2"])
        coords, _radii, _sids = chain_arrays(graph, steps)
        length = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
        return Chain(tuple(steps), node, end, length)

    # Seed from the ends first, so an open chain is walked from one of its ends and
    # comes out whole rather than as two halves meeting in the middle.
    seeds = [
        (nid, sids[0])
        for nid, sids in sorted(at_node.items())
        if len(sids) == 1 or nid in branch_nodes
    ]
    for nid, _ in seeds:
        for sid in sorted(at_node.get(nid, ())):
            if sid in remaining:
                chains.append(walk(nid, sid))

    while remaining:  # whatever is left is a cycle
        sid = min(remaining)
        seg = graph.segment(sid)
        notes.append(f"segments {sorted(remaining)} form a loop; broken at segment {sid}")
        chains.append(walk(seg["node1"], sid))

    chains.sort(key=lambda c: c.length_um, reverse=True)
    if len(chains) > 1:
        notes.append(
            f"the selection is {len(chains)} separate runs "
            f"({', '.join(f'{len(c.steps)} seg / {c.length_um / 1000:.1f} mm' for c in chains)})"
        )
    return chains, notes


def chain_arrays(graph, steps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate a chain's points. ``(coords_um, radii_um, segment id per point)``.

    Amira gives each edge its own copy of a shared node's position
    (``graphmodel.py:15-23``), so the first point of every segment after the first
    duplicates the previous segment's last. Dropped -- a zero-length step would put a
    zero-length chord into the arclength resample and an undefined tangent into the
    frame. Only dropped when the two really do coincide, within the package's own
    junction tolerance; a wider gap means the segments do not actually meet, and
    welding over it silently would invent centreline that is not there.
    """
    from .edit.graphmodel import NODE_EPS_UM

    coords: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    sids: list[np.ndarray] = []
    for k, (sid, reverse) in enumerate(steps):
        c = graph.coords(sid)
        r = graph.radii(sid)
        if reverse:
            c, r = c[::-1], r[::-1]
        if k and len(coords) and len(c):
            gap = float(np.linalg.norm(c[0] - coords[-1][-1]))
            if gap <= NODE_EPS_UM:
                c, r = c[1:], r[1:]
        if not len(c):
            continue
        coords.append(c)
        radii.append(r)
        sids.append(np.full(len(c), int(sid), dtype=np.int64))
    if not coords:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)
    return (
        np.concatenate(coords).astype(np.float64),
        np.concatenate(radii).astype(np.float64),
        np.concatenate(sids),
    )


# --------------------------------------------------------------------------- #
# Frames and curvature
# --------------------------------------------------------------------------- #


def _unit(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def frames(coords_um, *, seed_normal=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Twist-free ``(tangents, normals, binormals)`` at every point.

    Wraps :func:`~.edit.reconnect.geodesic.shape.rotation_minimizing_frame`, which
    derives its tangents from ``np.gradient`` -- **one-sided at the first and last
    index**. A one-sided difference is the chord, whose direction on a circular arc is
    the tangent at the *midpoint* of the step rather than at its end, so the two end
    tangents are each half a turn out of true. :func:`curvature` is where that is dealt
    with, because it is the only thing the error matters to; padding the polyline here
    does not help, since a linearly extrapolated point reproduces exactly the same
    one-sided difference.

    ``seed_normal`` fixes the rotation of the whole stack: the frame transports it
    with zero twist, so one seed decides the in-plane orientation of every image. Left
    at ``None`` it reproduces the frame's own choice (world +z, or +x where the
    tangent is within ~25 degrees of z), which is deterministic, so a rebuild of the
    same path gives the same pictures.
    """
    from .edit.reconnect.geodesic.shape import rotation_minimizing_frame

    pts = np.asarray(coords_um, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        t = np.array([[0.0, 0.0, 1.0]])
        return t, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])

    tangents, normals, binormals = rotation_minimizing_frame(pts, seed_normal)
    return _unit(tangents), _unit(normals), _unit(binormals)


def turn_angles(tangents) -> np.ndarray:
    """``(N-1,)`` angle between consecutive unit tangents, in radians.

    Computed as ``2 * arcsin(|T_{i+1} - T_i| / 2)``, which is an exact identity for
    unit vectors, rather than ``arccos(T_i . T_{i+1})``.

    **This is defensive, not a fix for an observed error.** ``arccos`` is
    ill-conditioned near 1 -- its relative error grows like ``eps / theta^2``, so it
    loses roughly twice the digits ``theta`` is small by -- but measured in float64 at
    the turn angles this data actually produces (a 33 um step on a 2 mm radius is
    0.0165 rad) the two agree to 4e-14, and ``arccos`` only costs more than 1% of ``R``
    below ``theta ~ 3e-8``, which is a radius of curvature of a kilometre. So the
    choice buys nothing here today; it removes a trap that would bite if the tangents
    were ever float32, where ``arccos`` is already 2.3% out at a milliradian.

    ``skeleton_optimise._turn_deg`` is the existing ``arccos`` form, and is entirely
    fine for its own 120-degree threshold.
    """
    t = _unit(tangents)
    chord = np.linalg.norm(np.diff(t, axis=0), axis=1)
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


@dataclass
class Curvature:
    """Where the perpendicular planes would envelope, and how close this path is."""

    theta_rad: np.ndarray  # (N-1,) turn between consecutive tangents
    ds_um: np.ndarray  # (N-1,) arclength step
    r_step_um: np.ndarray  # (N-1,) ds / theta; inf on a straight run
    r_point_um: np.ndarray  # (N,) the tighter of the two steps meeting at a point
    r_min_um: float
    at_index: int

    def describe(self) -> str:
        if not np.isfinite(self.r_min_um):
            return "curvature: straight (no bound on the half-width)"
        return (
            f"curvature: tightest radius {self.r_min_um:,.0f} um at plane "
            f"{self.at_index} (median turn {np.degrees(np.median(self.theta_rad)):.2f} deg)"
        )


def curvature(coords_um, tangents) -> Curvature:
    """Turn angle, step length and local radius of curvature along a path.

    ``tangents`` **must be the frame's own** -- the ones :func:`frames` returned for
    this same path. Certifying a stack against tangents other than the ones that
    actually defined its planes certifies nothing.

    **The two end steps are discarded and replaced by their neighbours.** The frame's
    end tangents are one-sided differences (see :func:`frames`), i.e. chords, whose
    direction is the tangent half a step in from the end. The turn measured across the
    first step is therefore only half the real one, and ``R = ds / theta`` comes out at
    **twice** the true radius of curvature -- an optimistic bound, at exactly the two
    places a path is most likely to have been cut mid-bend. Measured on a 5 mm circular
    arc: 9,999 um reported at each end against a true 5,000 um. Copying the neighbour
    is conservative and, on any path smooth enough to reformat, correct to within the
    discretisation.
    """
    pts = np.asarray(coords_um, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        empty = np.zeros(0)
        return Curvature(empty, empty, empty, np.full(n, np.inf), np.inf, 0)

    theta = turn_angles(tangents)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_step = np.where(theta > 1e-12, ds / np.maximum(theta, 1e-12), np.inf)
    if len(r_step) >= 3:
        r_step = r_step.copy()
        r_step[0], r_step[-1] = r_step[1], r_step[-2]

    # A point is bounded by whichever of the two steps meeting at it turns harder.
    r_point = np.full(n, np.inf)
    r_point[:-1] = np.minimum(r_point[:-1], r_step)
    r_point[1:] = np.minimum(r_point[1:], r_step)

    at = int(np.argmin(r_point))
    return Curvature(theta, ds, r_step, r_point, float(r_point[at]), at)


def collision_free(curv: Curvature, half_um, *, safety=DEFAULT_SAFETY):
    """``(ok, worst ratio, index)`` for a half-width against a curvature bound.

    ``half_um`` is a scalar or one value per plane. The stack is free of
    self-intersection between neighbouring planes where ``half < safety * R``; the
    worst ratio is ``max(half / (safety * R))`` and 1.0 is the boundary.
    """
    h = np.broadcast_to(np.asarray(half_um, dtype=np.float64), curv.r_point_um.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = h / np.maximum(safety * curv.r_point_um, 1e-12)
    ratio = np.where(np.isfinite(curv.r_point_um), ratio, 0.0)
    at = int(np.argmax(ratio))
    worst = float(ratio[at])
    return worst <= 1.0, worst, at


def planes_disjoint(coords_um, tangents, normals, binormals, half_um):
    """Direct check that no two images in the stack show the same tissue.

    ``(ok, offenders)``, where each offender is ``(i, j, centre distance um)``.

    Two planes collide in one of two ways, and they need different tests:

    * **they cross.** Non-parallel squares that each have corners on both sides of the
      other's plane genuinely intersect, and the tissue along that line of
      intersection appears in both images. This is the failure the curvature bound
      predicts, and here it is measured instead of predicted.
    * **they coincide.** Parallel planes never cross, however close they are -- but two
      *anti*-parallel planes lying on top of each other show exactly the same tissue
      twice, which is the same defect arrived at from the other direction. That is what
      a hairpin does, and no crossing test will ever see it.

    Planes that are close **along the vessel** are exempt from the second test: they
    are supposed to overlap, that is what sampling a vessel at less than its own width
    means. Only pairs whose arclength separation exceeds their combined reach are
    judged, which is precisely the statement "this vessel has come back on itself".

    Advisory, and deliberately so. Smoothing cannot fix a hairpin -- the two limbs are
    genuinely there -- and the only remedy is a narrower half-width, which is a
    decision about what the operator wants to see rather than one this function should
    take.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(coords_um, dtype=np.float64).reshape(-1, 3)
    t = _unit(tangents)
    u = _unit(normals)
    v = _unit(binormals)
    if len(pts) < 2:
        return True, []
    h = np.broadcast_to(np.asarray(half_um, dtype=np.float64), (len(pts),)).astype(np.float64)

    diag = np.sqrt(2.0) * h
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])

    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=float(2.0 * diag.max()), output_type="ndarray")
    if not len(pairs):
        return True, []

    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(pts[i] - pts[j], axis=1)
    within = dist <= diag[i] + diag[j]
    i, j, dist = i[within], j[within], dist[within]

    parallel = np.abs(np.einsum("ij,ij->i", t[i], t[j])) > APPROACH_PARALLEL_COS
    # Adjacent along the vessel: overlapping is the design, not a defect.
    adjacent = np.abs(arclen[i] - arclen[j]) <= diag[i] + diag[j]

    offenders = []
    for a, b, d, par, adj in zip(
        i.tolist(), j.tolist(), dist.tolist(), parallel.tolist(), adjacent.tolist()
    ):
        if par:
            if not adj and d <= APPROACH_FACTOR * (h[a] + h[b]):
                offenders.append((a, b, float(d)))
        elif _squares_cross(pts, t, u, v, h, a, b):
            offenders.append((a, b, float(d)))
    return not offenders, offenders


def _squares_cross(pts, t, u, v, h, a: int, b: int) -> bool:
    """Do square ``a`` and square ``b`` each straddle the other's plane?"""
    for p, q in ((a, b), (b, a)):
        corners = (
            pts[p]
            + h[p] * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
            @ np.vstack([u[p], v[p]])
        )
        side = (corners - pts[q]) @ t[q]
        if not (side.min() < 0.0 < side.max()):
            return False
    return True


# --------------------------------------------------------------------------- #
# The centreline the stack is built on
# --------------------------------------------------------------------------- #


@dataclass
class Centreline:
    """A path resampled, smoothed and framed, ready to cut planes on."""

    coords_um: np.ndarray  # (N,3) uniform arclength spacing
    radii_um: np.ndarray  # (N,)
    arclen_um: np.ndarray  # (N,) from 0
    tangents: np.ndarray  # (N,3) unit
    normals: np.ndarray  # (N,3) unit -- the plane's +u (image columns)
    binormals: np.ndarray  # (N,3) unit -- the plane's +v (image rows)
    seg_ids: np.ndarray  # (N,) which segment each plane came from
    step_um: float  # nominal arclength between planes
    curvature: Curvature
    smooth_window_um: float = 0.0
    smooth_iters: int = 0
    max_move_um: float = 0.0
    median_move_um: float = 0.0
    seed_normal: np.ndarray = None
    notes: tuple = ()

    @property
    def length_um(self) -> float:
        return float(self.arclen_um[-1]) if len(self.arclen_um) else 0.0

    def describe(self) -> str:
        return (
            f"{len(self.coords_um)} planes over {self.length_um / 1000:.2f} mm at "
            f"{self.step_um:.1f} um; {self.curvature.describe()}"
        )


def _resample(coords, radii, seg_ids, step_um):
    """Uniform-arclength resample, carrying radii and segment ids across.

    Coordinates go through ``candidates.resample_by_arclength`` so there is one
    arclength resampler in the tree. The attributes are then interpolated against the
    *original* arclength at the positions that resampler used -- which are a
    ``linspace`` over the total, read back off the output length rather than
    recomputed, so the two cannot drift apart if its point-count rule ever changes.
    """
    from .edit.reconnect.candidates import resample_by_arclength

    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if len(coords) < 2:
        return coords, np.asarray(radii, dtype=np.float64), np.asarray(seg_ids)

    out = resample_by_arclength(coords, step_um)
    step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    want = np.linspace(0.0, float(arc[-1]), len(out))

    new_radii = np.interp(want, arc, np.asarray(radii, dtype=np.float64))
    # A segment id is a label, so nearest rather than interpolated.
    nearest = np.clip(np.searchsorted(arc, want), 0, len(seg_ids) - 1)
    return out, new_radii, np.asarray(seg_ids)[nearest]


def build_centreline(
    coords_um,
    radii_um,
    seg_ids=None,
    *,
    step_um,
    half_of=None,
    safety=DEFAULT_SAFETY,
    smooth_window_um=None,
    max_smooth_iters=MAX_SMOOTH_ITERS,
    seed_normal=None,
    max_move_frac=0.5,
) -> Centreline:
    """Resample, smooth until the plane stack is collision-free, and frame it.

    ``half_of(radii_um) -> (N,)`` says how wide the planes would be for a given set of
    radii. It is a callable rather than an array because the half-width depends on the
    radii, which the resampling changes -- so the bound has to be re-evaluated on each
    iteration against the width that iteration would actually produce.

    The loop, at most ``max_smooth_iters`` times: frame, measure, stop if the bound is
    met, otherwise grow the smoothing window and re-smooth. Smoothing is
    ``skeleton_optimise._gaussian_smooth`` -- arclength-parameterised, so the window is
    a physical distance rather than a point count -- applied to the **whole**
    concatenated path rather than per segment, because the kink at a junction seam is
    exactly where the bound bites and per-segment smoothing pins both sides of it.

    Two things are reported whatever happens, because both are ways of meeting the
    target that are not really meeting it: how far smoothing moved the centreline, and
    whether the window ever got wide enough to touch a neighbouring point at all
    (``_gaussian_smooth`` returns that; a window narrower than the point spacing is a
    silent no-op).

    ``_gaussian_smooth`` pins the two end points, so a kink in the first or last pair
    survives every iteration. When that is what is left, the loop stops early and says
    so rather than spending its remaining iterations achieving nothing; the caller's
    half-width clamp is the only remedy there.
    """
    from .edit.skeleton_optimise import _gaussian_smooth

    coords = np.asarray(coords_um, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii_um, dtype=np.float64).reshape(-1)
    if seg_ids is None:
        seg_ids = np.zeros(len(coords), dtype=np.int64)
    seg_ids = np.asarray(seg_ids).reshape(-1)
    if len(coords) < 2:
        raise ReformatError("a reformat needs at least two centreline points")

    original = coords.copy()
    if half_of is None:
        half_of = lambda r: HALF_WIDTH_RADII * r  # noqa: E731

    notes: list[str] = []
    coords, radii, seg_ids = _resample(coords, radii, seg_ids, step_um)

    window = float(smooth_window_um) if smooth_window_um else 0.0
    iters = 0
    tangents = normals = binormals = None
    curv = None
    for attempt in range(int(max_smooth_iters) + 1):
        tangents, normals, binormals = frames(coords, seed_normal=seed_normal)
        curv = curvature(coords, tangents)
        ok, worst, at = collision_free(curv, half_of(radii), safety=safety)
        if ok:
            break
        if attempt == max_smooth_iters:
            notes.append(
                f"still {worst:.2f}x over the curvature bound after {iters} smoothing "
                f"pass(es); the half-width will be clamped"
            )
            break

        window = max(window * SMOOTH_GROWTH, 3.0 * step_um) if window else 3.0 * step_um
        smoothed, reached = _gaussian_smooth(coords, window)
        if not reached:
            # The window did not span a neighbour, so the pass was a no-op. Widen it
            # rather than counting a wasted attempt against the budget.
            window *= 2.0
            smoothed, reached = _gaussian_smooth(coords, window)
        if max_move_frac:
            from .edit.skeleton_optimise import _clamp_moves

            smoothed, n_clamped = _clamp_moves(coords, smoothed, radii, max_move_frac)
            if n_clamped:
                notes.append(
                    f"{n_clamped} point(s) hit the {max_move_frac:g}-radius move limit "
                    f"at window {window:,.0f} um"
                )
        if np.allclose(smoothed, coords, atol=1e-9):
            notes.append(
                "smoothing has stopped changing the path (the remaining bend is at a "
                "pinned end point); only clamping the half-width can help here"
            )
            break
        coords, radii, seg_ids = _resample(smoothed, radii, seg_ids, step_um)
        iters += 1

    tangents, normals, binormals = frames(coords, seed_normal=seed_normal)
    curv = curvature(coords, tangents)

    move = _displacement(original, coords)
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))])
    if iters:
        notes.append(
            f"smoothed {iters} pass(es) at a {window:,.0f} um window; the centreline "
            f"moved a median {np.median(move):,.0f} um, at most {move.max():,.0f} um"
        )
    return Centreline(
        coords_um=coords,
        radii_um=radii,
        arclen_um=arclen,
        tangents=tangents,
        normals=normals,
        binormals=binormals,
        seg_ids=seg_ids,
        step_um=float(step_um),
        curvature=curv,
        smooth_window_um=window,
        smooth_iters=iters,
        max_move_um=float(move.max()) if len(move) else 0.0,
        median_move_um=float(np.median(move)) if len(move) else 0.0,
        seed_normal=None if seed_normal is None else np.asarray(seed_normal, dtype=np.float64),
        notes=tuple(notes),
    )


def _displacement(original, moved) -> np.ndarray:
    """How far each smoothed point ended up from the path it was smoothed from.

    Nearest-point rather than by index: resampling changes the point count, so index
    ``i`` before and after are not the same place on the vessel and comparing them
    would report the resampling as if it were a displacement.
    """
    from scipy.spatial import cKDTree

    if not len(original) or not len(moved):
        return np.zeros(0)
    d, _ = cKDTree(original).query(moved)
    return np.asarray(d, dtype=np.float64)


# --------------------------------------------------------------------------- #
# How big each plane is
# --------------------------------------------------------------------------- #


@dataclass
class PlaneGeometry:
    """The physical size of each image in the stack, and what it cost to get there."""

    half_um: np.ndarray  # (N,) half-extent of the square, in um
    px_um: np.ndarray  # (N,) sample pitch within the plane
    size_px: int
    mode: str
    requested_um: np.ndarray  # (N,) what the mode asked for before clamping
    n_clamped: int
    r_min_um: float
    #: The raw voxel size, so the grid can say whether it is resolving or magnifying.
    voxel_um: float = 0.0
    #: ``native`` only: planes whose fixed frame reaches past the curvature bound, and
    #: the frame size that would stay inside it. Reported rather than applied -- see
    #: :func:`plane_geometry` for why clamping is the wrong remedy in that mode.
    n_over_bound: int = 0
    suggested_size_px: int = 0

    @property
    def clamped(self) -> bool:
        return self.n_clamped > 0

    @property
    def oversampling(self) -> np.ndarray:
        """(N,) how many output pixels the grid puts across one raw voxel.

        Above 1 the plane is being *magnified* rather than resolved: the sampling asks
        for detail finer than the acquisition, so what comes back is largely the
        interpolation kernel. This is the number that explains a soft-looking section,
        and nothing showed it before.
        """
        if self.voxel_um <= 0:
            return np.ones_like(self.px_um)
        return self.voxel_um / np.maximum(self.px_um, 1e-9)

    def describe(self) -> str:
        out = (
            f"planes: {self.size_px}x{self.size_px} px, half-width "
            f"{self.half_um.min():,.0f}-{self.half_um.max():,.0f} um "
            f"({self.px_um.min():.1f}-{self.px_um.max():.1f} um/px), mode '{self.mode}'"
        )
        if self.voxel_um > 0:
            over = self.oversampling
            out += (
                f"\n  sampling {over.min():.2f}-{over.max():.2f}x the {self.voxel_um:.1f} "
                f"um voxel"
            )
            if over.max() > 1.5:
                out += (
                    " -- above 1.5x the section is magnified rather than resolved; "
                    "'match voxel' sizes the grid to the data"
                )
        if self.clamped:
            out += (
                f"\n  half-width clamped at {self.n_clamped} of {len(self.half_um)} "
                f"planes: asked {self.requested_um.max():,.0f} um, tightest curvature "
                f"radius {self.r_min_um:,.0f} um"
            )
        if self.n_over_bound:
            out += (
                f"\n  the fixed frame reaches past the curvature bound at "
                f"{self.n_over_bound} of {len(self.half_um)} planes (tightest radius "
                f"{self.r_min_um:,.0f} um); {self.suggested_size_px}x"
                f"{self.suggested_size_px} px would stay inside it. Not clamped: "
                f"narrowing the frame here would shrink the pixel, not the view."
            )
        return out


def plane_geometry(
    centreline: Centreline,
    *,
    mode="radius",
    radii_k=HALF_WIDTH_RADII,
    size_px=DEFAULT_SIZE_PX,
    half_um=None,
    px_um=None,
    safety=DEFAULT_SAFETY,
    min_half_um=0.0,
    voxel_um=0.0,
    native_scale=1.0,
) -> PlaneGeometry:
    """Decide each plane's physical half-width and sample pitch.

    Three modes:

    ``radius``
        Half-width is ``radii_k`` times the local radius, on a fixed pixel grid, so a
        capillary and an artery both fill the frame. The scale then varies down the
        stack, which is a real cost -- see :mod:`~.viewer_reformat` for why the viewer
        cannot show a physical ruler in this mode.

    ``fixed``
        One half-width for the whole stack, ``radii_k`` times the largest radius on
        the path. Physically comparable end to end, at the price of a mostly-empty
        frame wherever the vessel is small.

    ``manual``
        ``half_um`` and ``px_um`` exactly as given; ``size_px`` follows from them.

    ``native``
        **A fixed frame at the acquisition's own resolution, and the only mode that
        cannot magnify.** The pitch is pinned to ``native_scale`` raw voxels -- one, by
        default -- and the half-width *follows* from the frame size rather than the
        other way round. Every other mode derives the pitch from a half-width, so any
        half-width smaller than ``size_px / 2`` voxels magnifies, whether it came from
        a small radius or from the curvature clamp cutting it. Here that cannot happen:
        one output pixel is one voxel by construction, so the frame is a window on the
        data rather than an enlargement of it.

    The three width-driven modes clamp the half-width, per plane, to ``safety`` times
    the local radius of curvature, because past that the planes fold through each
    other. The clamp is counted and reported -- it changes what the images mean, so it
    does not get to happen quietly.

    **``native`` does not clamp, and that is deliberate.** With the pitch pinned,
    shrinking the half-width cannot shrink the field of view; it would shrink the pitch
    instead, which is exactly the magnification the mode exists to prevent. So the
    bound becomes a *report* -- how many planes exceed it, and what ``size_px`` would
    stay inside -- and reducing the frame is the caller's decision.
    """
    r = centreline.radii_um
    n = len(r)
    size_px = max(int(size_px) | 1, 3)  # odd, so there is a centre pixel
    bound = safety * centreline.curvature.r_point_um

    if mode == "native":
        if voxel_um <= 0:
            raise ReformatError("native plane geometry needs the raw voxel size")
        pitch_value = float(voxel_um) * float(native_scale)
        half_value = (size_px // 2) * pitch_value
        half = np.full(n, half_value)
        want = half.copy()
        over = int((half > np.where(np.isfinite(bound), bound, np.inf) + 1e-9).sum())
        inside = np.min(bound[np.isfinite(bound)]) if np.isfinite(bound).any() else np.inf
        suggested = (
            2 * int(np.floor(inside / max(pitch_value, 1e-9))) + 1
            if np.isfinite(inside) else size_px
        )
        return PlaneGeometry(
            half_um=half, px_um=np.full(n, pitch_value), size_px=size_px, mode=mode,
            requested_um=want, n_clamped=0,
            r_min_um=centreline.curvature.r_min_um, voxel_um=float(voxel_um),
            n_over_bound=over, suggested_size_px=max(min(suggested, size_px), 3),
        )

    if mode == "manual":
        if half_um is None or px_um is None:
            raise ReformatError("manual plane geometry needs both half_um and px_um")
        size_px = 2 * int(round(float(half_um) / float(px_um))) + 1
        want = np.full(n, float(half_um))
    elif mode == "fixed":
        want = np.full(n, radii_k * float(np.max(r)) if len(r) else 0.0)
    elif mode == "radius":
        want = radii_k * r
    else:
        raise ReformatError(
            f"unknown plane mode {mode!r}; expected radius/fixed/manual/native"
        )

    want = np.maximum(want, max(float(min_half_um), 1e-6))

    used = np.minimum(want, np.where(np.isfinite(bound), bound, want))
    n_clamped = int((used < want - 1e-9).sum())
    if mode == "fixed":
        # One width for the whole stack means the tightest bend decides it for
        # everyone; keeping a per-plane width here would quietly turn this into
        # `radius` mode.
        used = np.full(n, float(used.min()))

    half = used
    pitch = half / float(size_px // 2)
    return PlaneGeometry(
        half_um=half,
        px_um=pitch,
        size_px=size_px,
        mode=mode,
        requested_um=want,
        n_clamped=n_clamped,
        r_min_um=centreline.curvature.r_min_um,
        voxel_um=float(voxel_um),
    )


def plane_points(centreline: Centreline, geom: PlaneGeometry, lo: int, hi: int) -> np.ndarray:
    """``(hi-lo, size, size, 3)`` world positions of one run of planes' samples.

    Built a run at a time rather than all at once: the full array for a long vessel is
    hundreds of megabytes of float64 and there is never a reason to hold it, since the
    samplers consume it in chunks anyway.

    Axis 1 is ``+binormal`` (image rows), axis 2 is ``+normal`` (image columns), and
    the centre pixel is the centreline point exactly.
    """
    g = np.arange(geom.size_px, dtype=np.float64) - geom.size_px // 2
    off = g[None, :] * geom.px_um[lo:hi, None]  # (M, size) in um
    return (
        centreline.coords_um[lo:hi, None, None, :]
        + off[:, None, :, None] * centreline.normals[lo:hi, None, None, :]
        + off[:, :, None, None] * centreline.binormals[lo:hi, None, None, :]
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


@dataclass
class SampleStats:
    """What a build actually cost, so a slow one can be explained."""

    chunks: int = 0
    slices_read: int = 0
    n_samples: int = 0
    n_outside: int = 0
    #: Every raw slice index any block covered, so the re-read at block seams is
    #: visible. A run that reads twice what it needs is invisible otherwise, except
    #: as an unexplained doubling of the wall clock.
    slices: set = field(default_factory=set)
    #: Wall clock per phase, following the house pattern (`reskeletonise.report.seconds`).
    #: Decode and interpolation are worth separating because they respond to completely
    #: different things -- the first to how the TIFFs are stored, the second to the
    #: spline order -- and without the split a slow build has no diagnosis.
    seconds: dict = field(default_factory=dict)
    order: int = DEFAULT_ORDER
    #: How the windows were actually read. A directory that fell back to whole-page
    #: decodes is an order of magnitude slower and otherwise looks identical.
    strip_reads: int = 0
    whole_page_reads: int = 0

    @property
    def distinct_slices(self) -> int:
        return len(self.slices)

    @property
    def outside(self) -> float:
        return self.n_outside / self.n_samples if self.n_samples else 0.0

    def describe(self) -> str:
        waste = self.slices_read - self.distinct_slices
        out = (
            f"sampling: {self.chunks} block(s), {self.slices_read:,} slice reads "
            f"over {self.distinct_slices:,} distinct slices"
            + (f" ({waste:,} re-read at block seams)" if waste > 0 else "")
            + (f"; {100 * self.outside:.1f}% of samples outside the volume"
               if self.outside > 0 else "")
        )
        if self.whole_page_reads:
            out += (
                f"\n  {self.whole_page_reads:,} of {self.strip_reads + self.whole_page_reads:,} "
                f"windows needed a whole-page decode (~19x the cost of a strip read)"
            )
        if self.seconds:
            out += ("\n  order " + str(self.order) + "; "
                    + ", ".join(f"{k} {v:.1f}s" for k, v in self.seconds.items()))
        return out


class ObliqueSampler:
    """Trilinear samples of the raw stack at arbitrary world positions.

    There is no such sampler anywhere else in this package -- everything oblique is
    mask-side and nearest-neighbour -- and the reason is that the raw stack is ~92 GB
    read one TIFF at a time. So the work is organised around *which slices a group of
    planes needs*, not around the planes.

    Consecutive planes are spatially adjacent, so a contiguous run of them shares a
    small ``(slice, row, col)`` box. The run is grown until that box would exceed the
    memory budget, read once with ``TiffStack.read_stack_window``, and every sample in
    the run taken from it with a single ``map_coordinates`` call.

    Two details in that are load-bearing rather than incidental:

    * **the block is padded by one voxel on every side.** ``map_coordinates`` with
      ``mode="constant"`` returns ``cval`` for any coordinate outside the block, so a
      sample sitting a fraction of a voxel inside the last row would come back 0
      instead of interpolated -- a black stripe down every block seam. The pad costs
      one extra decode per seam.
    * **the block is cast to float32 first.** ``map_coordinates`` keeps its input's
      dtype, so an ``order=1`` blend of ``uint16`` truncates back to integers and the
      image picks up a visible quantisation crawl along the vessel.

    Because ``WorldFrame.um_to_raw`` is a pure per-axis scale with an axis reversal
    (``frame.py:107``), the axis-aligned raw box of a sampled plane is exactly the box
    of its four corners -- so the planner works off ``(N, 4, 3)`` rather than off every
    sample, and is exact rather than conservative.
    """

    def __init__(self, stack, frame, *, budget_mb=512.0, order=DEFAULT_ORDER):
        self.stack = stack
        self.frame = frame
        self.budget = float(budget_mb) * 1e6
        self.order = int(order)
        if self.order not in SPLINE_PAD:
            raise ReformatError(
                f"interpolation order {self.order} is not supported; "
                f"expected one of {sorted(SPLINE_PAD)}"
            )
        self.pad = SPLINE_PAD[self.order]

    def sample(self, points_um, *, stats=None) -> np.ndarray:
        """``(...)`` float32 samples for ``(..., 3)`` world positions, 0 outside.

        The primitive: reads **one** block covering everything it was given. Chunking
        is :meth:`sample_planes`' job, which feeds this a run at a time.
        """
        from scipy import ndimage

        pts = np.asarray(points_um, dtype=np.float64)
        lead = pts.shape[:-1]
        raw = self.frame.um_to_raw(pts.reshape(-1, 3)).reshape(lead + (3,))
        out = np.zeros(lead, dtype=np.float32)
        if not out.size:
            return out

        nz, n_rows, n_cols = (int(v) for v in self.stack.shape)
        flat = raw.reshape(-1, 3)
        n_out = int((~self._inside(flat, nz, n_rows, n_cols)).sum())

        lo, hi = self._box(flat)
        pad = self.pad
        z0 = max(int(np.floor(lo[0])) - pad, 0)
        z1 = min(int(np.ceil(hi[0])) + pad + 1, nz)
        r0 = int(np.floor(lo[1])) - pad
        r1 = int(np.ceil(hi[1])) + pad + 1
        c0 = int(np.floor(lo[2])) - pad
        c1 = int(np.ceil(hi[2])) + pad + 1

        if stats is not None:
            stats.chunks += 1
            stats.n_samples += flat.shape[0]
            stats.n_outside += n_out

        if z1 <= z0:  # the whole request is off the end of the stack
            return out

        import time

        t0 = time.perf_counter()
        block = self.stack.read_stack_window(z0, z1, r0, r1, c0, c1).astype(np.float32)
        t1 = time.perf_counter()
        local = flat - np.array([z0, r0, c0], dtype=np.float64)
        out.reshape(-1)[:] = ndimage.map_coordinates(
            block, local.T, order=self.order, mode="constant", cval=0.0
        )
        if stats is not None:
            stats.slices_read += z1 - z0
            stats.slices.update(range(z0, z1))
            stats.seconds["decode"] = stats.seconds.get("decode", 0.0) + (t1 - t0)
            stats.seconds["interpolate"] = (
                stats.seconds.get("interpolate", 0.0) + time.perf_counter() - t1
            )
        return out

    def sample_planes(self, centreline, geom, *, progress=None, stats=None) -> np.ndarray:
        """``(N, size, size)`` samples for a whole stack, read block by block."""
        n = len(centreline.coords_um)
        size = geom.size_px
        out = np.zeros((n, size, size), dtype=np.float32)
        for lo, hi in self.plan(centreline, geom):
            out[lo:hi] = self.sample(plane_points(centreline, geom, lo, hi), stats=stats)
            if progress is not None:
                progress(hi, n)
        return out

    def plan(self, centreline, geom) -> list:
        """Contiguous runs of planes whose shared raw block fits the budget.

        Grown greedily from the four corners of each plane, which -- see the class
        docstring -- give the exact raw box. A single plane that already exceeds the
        budget is still emitted on its own: splitting within a plane would help, and
        every real case is orders of magnitude under.
        """
        corners = self._corners(centreline, geom)
        lo_i = np.minimum.reduce(corners, axis=1)
        hi_i = np.maximum.reduce(corners, axis=1)

        runs: list[tuple[int, int]] = []
        n = len(lo_i)
        if not n:
            return runs
        start = 0
        lo = lo_i[0].copy()
        hi = hi_i[0].copy()
        for i in range(1, n):
            nlo = np.minimum(lo, lo_i[i])
            nhi = np.maximum(hi, hi_i[i])
            if self._cost(nlo, nhi, self.pad) > self.budget and i > start:
                runs.append((start, i))
                start, lo, hi = i, lo_i[i].copy(), hi_i[i].copy()
            else:
                lo, hi = nlo, nhi
        runs.append((start, n))
        return runs

    def _corners(self, centreline, geom) -> np.ndarray:
        """``(N, 4, 3)`` raw positions of each plane's four corners."""
        signs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
        h = geom.half_um[:, None, None]
        world = (
            centreline.coords_um[:, None, :]
            + h * signs[None, :, 0, None] * centreline.normals[:, None, :]
            + h * signs[None, :, 1, None] * centreline.binormals[:, None, :]
        )
        return self.frame.um_to_raw(world.reshape(-1, 3)).reshape(len(world), 4, 3)

    @staticmethod
    def _box(points):
        return points.min(axis=0), points.max(axis=0)

    @staticmethod
    def _cost(lo, hi, pad: int) -> float:
        """Bytes the block spanning ``[lo, hi]`` would take, as float32.

        ``pad`` on each side, so a higher spline order -- which needs a much wider pad
        to keep its prefilter seam-free -- correctly buys fewer planes per block rather
        than quietly overrunning the budget.
        """
        span = np.ceil(hi) - np.floor(lo) + 2.0 * pad + 1.0
        return float(np.prod(np.maximum(span, 1.0)) * 4.0)  # float32

    @staticmethod
    def _inside(raw, nz, n_rows, n_cols) -> np.ndarray:
        hi = np.array([nz - 1, n_rows - 1, n_cols - 1], dtype=np.float64)
        return np.all((raw >= 0.0) & (raw <= hi), axis=-1)


class LabelSampler:
    """Nearest-neighbour samples of the segmentation lattice at world positions.

    Nearest, never linear: a label is not a quantity to average, and an interpolated
    0.5 in the overlay would draw a boundary that is not in the mask.

    Delegates the actual gather to ``crosssection._PlaneSampler.at``, which keeps a
    decoded-slice cache and groups the lookup by z so each slice is touched once per
    call -- the same machinery the collapse detector and the radius measurer already
    share, rather than a fourth copy of it.
    """

    def __init__(self, labels, frame, *, cache_slices=160, budget_mb=64.0):
        from .crosssection import _PlaneSampler

        # `cache_slices` is this class's historical knob, kept because callers pass
        # it; the sampler budgets in bytes now that it decodes row bands rather than
        # whole planes, so the count is converted at its own plane size.
        plane_bytes = max(int(frame.seg_dims[0]) * int(frame.seg_dims[1]), 1)
        self.sampler = _PlaneSampler(
            labels, frame, cache_bytes=int(cache_slices) * plane_bytes
        )
        self.frame = frame
        self.budget = float(budget_mb) * 1e6

    def sample(self, points_um) -> np.ndarray:
        pts = np.asarray(points_um, dtype=np.float64)
        lead = pts.shape[:-1]
        ijk = self.frame.um_to_seg(pts.reshape(-1, 3))
        return self.sampler.at(ijk).reshape(lead)

    def sample_planes(self, centreline, geom, *, progress=None) -> np.ndarray:
        n = len(centreline.coords_um)
        size = geom.size_px
        out = np.zeros((n, size, size), dtype=np.uint8)
        # Chunked only to bound the point array; the sampler's own slice cache makes
        # the z grouping cheap either way.
        per = max(int(self.budget // max(size * size * 3 * 8, 1)), 1)
        for lo in range(0, n, per):
            hi = min(lo + per, n)
            out[lo:hi] = self.sample(plane_points(centreline, geom, lo, hi))
            if progress is not None:
                progress(hi, n)
        return out


# --------------------------------------------------------------------------- #
# The whole thing
# --------------------------------------------------------------------------- #


@dataclass
class Reformat:
    """A built stack: the images, the geometry that made them, and the audit."""

    raw: np.ndarray  # (N, S, S)
    mask: np.ndarray | None  # (N, S, S) uint8
    centreline: Centreline
    geometry: PlaneGeometry
    graph_points: np.ndarray  # (M,3) as (plane index, row px, col px)
    stats: SampleStats
    chains: tuple = ()
    notes: tuple = ()

    @property
    def n_planes(self) -> int:
        return len(self.raw)

    def describe(self) -> str:
        lines = [
            self.centreline.describe(),
            self.geometry.describe(),
            self.stats.describe(),
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def build(
    graph,
    frame,
    stack,
    segment_ids,
    *,
    labels=None,
    mode="radius",
    radii_k=HALF_WIDTH_RADII,
    size_px=DEFAULT_SIZE_PX,
    half_um=None,
    px_um=None,
    step_um=None,
    safety=DEFAULT_SAFETY,
    smooth_window_um=None,
    max_smooth_iters=MAX_SMOOTH_ITERS,
    seed_normal=None,
    native_scale=1.0,
    order=DEFAULT_ORDER,
    budget_mb=512.0,
    max_planes=8000,
    with_graph_points=True,
    progress=None,
) -> Reformat:
    """Build a reformatted stack for a selection of segments.

    ``graph`` is an :class:`~.edit.graphmodel.EditableGraph`, ``frame`` a
    :class:`~.frame.WorldFrame`, ``stack`` a :class:`~.tiffstack.TiffStack`, and
    ``labels`` anything with ``slice_z`` (or a decoded array), as everywhere else.

    The selection is chained into one continuous path; a selection that is not a
    single chain uses its longest run and says what it left out.

    ``step_um`` defaults to one raw voxel, which is where the information runs out --
    sampling finer than the acquisition invents detail, and the collision bound gets
    tighter as the step shrinks for no gain.
    """
    notes: list[str] = []
    chains, chain_notes = chain_segments(graph, segment_ids)
    notes.extend(chain_notes)
    if not chains:
        raise ReformatError("nothing to reformat: no usable segments in the selection")
    chain = chains[0]
    if len(chains) > 1:
        # `chain_segments` reports *that* the selection splits; this says what the
        # build then did about it. A stack is one continuous path by definition, so
        # the rest cannot be in it -- and a note saying only "3 separate runs" leaves
        # the operator to work out which segments they are actually looking at.
        left = sorted(s for c in chains[1:] for s in c.segment_ids)
        notes.append(
            f"built the longest run only: segments {list(chain.segment_ids)}, "
            f"{chain.length_um / 1000:.2f} mm. Segments {left} are not connected to it "
            f"and were not sampled -- reformat them separately."
        )

    coords, radii, sids = chain_arrays(graph, chain.steps)
    if len(coords) < 2:
        raise ReformatError("the selected run is a single point")

    if step_um is None:
        step_um = float(np.min(frame.raw_voxel))
    step_um = max(float(step_um), 1e-3)

    n_expect = int(np.ceil(chain.length_um / step_um)) + 1
    if n_expect > int(max_planes):
        raise ReformatError(
            f"that run is {chain.length_um / 1000:.1f} mm, which is {n_expect:,} planes "
            f"at {step_um:.1f} um (limit {int(max_planes):,}). Raise the step, or the "
            f"limit if you mean it."
        )

    voxel_um = float(np.min(frame.raw_voxel))
    if mode == "native":
        # The frame size decides the width here, not the radii -- so the smoothing loop
        # is chasing one constant rather than something that moves as it resamples.
        native_half = (max(int(size_px) | 1, 3) // 2) * voxel_um * float(native_scale)
        half_of = lambda r: np.full(len(r), native_half)  # noqa: E731
    elif mode == "manual":
        half_of = (lambda _r: np.full(len(_r), float(half_um))) if half_um else None
    elif mode == "fixed":
        half_of = lambda r: np.full(len(r), radii_k * float(np.max(r)))  # noqa: E731
    else:
        half_of = lambda r: radii_k * r  # noqa: E731

    centreline = build_centreline(
        coords, radii, sids,
        step_um=step_um,
        half_of=half_of,
        safety=safety,
        smooth_window_um=smooth_window_um,
        max_smooth_iters=max_smooth_iters,
        seed_normal=seed_normal,
    )
    notes.extend(centreline.notes)

    geom = plane_geometry(
        centreline, mode=mode, radii_k=radii_k, size_px=size_px,
        half_um=half_um, px_um=px_um, safety=safety,
        min_half_um=voxel_um, voxel_um=voxel_um, native_scale=native_scale,
    )

    ok, offenders = planes_disjoint(
        centreline.coords_um, centreline.tangents, centreline.normals,
        centreline.binormals, geom.half_um,
    )
    if not ok:
        worst = min(offenders, key=lambda o: o[2])
        notes.append(
            f"{len(offenders)} pair(s) of planes still cross, the closest being planes "
            f"{worst[0]} and {worst[1]} at {worst[2]:,.0f} um apart. That is the vessel "
            f"coming back on itself, which smoothing cannot fix -- narrow the "
            f"half-width if those images matter."
        )

    stats = SampleStats(order=int(order))
    sampler = ObliqueSampler(stack, frame, budget_mb=budget_mb, order=order)
    before = (getattr(stack, "strip_reads", 0), getattr(stack, "whole_page_reads", 0))
    raw = sampler.sample_planes(centreline, geom, progress=progress, stats=stats)
    raw = _to_stack_dtype(raw, stack.dtype)
    stats.strip_reads = getattr(stack, "strip_reads", 0) - before[0]
    stats.whole_page_reads = getattr(stack, "whole_page_reads", 0) - before[1]

    mask = None
    if labels is not None:
        mask = LabelSampler(labels, frame).sample_planes(centreline, geom, progress=progress)

    pts = _graph_points(graph, centreline, geom) if with_graph_points else np.zeros((0, 3))

    return Reformat(
        raw=raw, mask=mask, centreline=centreline, geometry=geom,
        graph_points=pts, stats=stats, chains=tuple(chains), notes=tuple(notes),
    )


def plane_corners(reformat: "Reformat", indices=None) -> np.ndarray:
    """``(M, 4, 3)`` world-um corners of each plane, in texture order.

    The order is the one ``viewer3d.slice_plane_quad`` establishes for the axial image
    plane -- ``(0,0), (1,0), (1,1), (0,1)`` in texture coordinates -- so the same
    ``pv.Texture`` convention applies and the same row flip in
    ``viewer3d.slice_texture_array`` comes out right.

    Working that through, because getting it wrong mirrors the anatomy plausibly: the
    image's column axis is ``+normal`` and its row axis is ``+binormal``. The flip in
    ``slice_texture_array`` puts image row 0 at texture ``v = 0``, which is the pair of
    corners at ``-half * binormal``. So row 0 sits at ``-half`` along the binormal,
    which is exactly where :func:`plane_points` sampled it.
    """
    line = reformat.centreline
    idx = np.arange(len(line.coords_um)) if indices is None else np.asarray(indices)
    h = reformat.geometry.half_um[idx][:, None]
    c = line.coords_um[idx]
    u = line.normals[idx]
    v = line.binormals[idx]
    return np.stack([
        c - h * u - h * v,
        c + h * u - h * v,
        c + h * u + h * v,
        c - h * u + h * v,
    ], axis=1)


def texture_indices(n: int, limit: int) -> np.ndarray:
    """Up to ``limit`` evenly spaced plane indices, always including both ends.

    Each textured plane is its own actor and its own texture upload, so a 2,700-plane
    run drawn in full would cost more than the tree it is drawn over. Showing a fixed
    number of them evenly spaced conveys the same thing -- where the sampling went and
    how it is oriented -- at a cost that does not depend on the run's length.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if limit <= 0 or n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, int(limit)).round().astype(np.int64))


def _to_stack_dtype(values: np.ndarray, dtype) -> np.ndarray:
    """Round back into the raw stack's own dtype, clipped to its range."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return values.astype(dtype)
    info = np.iinfo(dtype)
    return np.clip(np.rint(values), info.min, info.max).astype(dtype)


def _graph_points(graph, centreline: Centreline, geom: PlaneGeometry) -> np.ndarray:
    """``(M, 3)`` graph points lying in each plane, as ``(plane, row px, col px)``.

    A point counts as "in" a plane when it is within half a step of it along the
    tangent and inside the square. Which means a branch leaving the vessel shows up as
    a mark walking out of the frame over a few planes -- the thing you actually want
    to see, and the reason this is the graph's points rather than just this chain's.
    """
    from scipy.spatial import cKDTree

    all_pts = np.array([p[:3] for p in graph.points.values()], dtype=np.float64)
    if not len(all_pts):
        return np.zeros((0, 3))
    tree = cKDTree(all_pts)
    half_step = 0.5 * centreline.step_um
    centre = geom.size_px // 2

    out = []
    for i in range(len(centreline.coords_um)):
        reach = float(np.sqrt(2.0) * geom.half_um[i])
        idx = tree.query_ball_point(centreline.coords_um[i], reach)
        if not idx:
            continue
        d = all_pts[idx] - centreline.coords_um[i]
        along = d @ centreline.tangents[i]
        near = np.abs(along) <= half_step
        if not near.any():
            continue
        d = d[near]
        u = d @ centreline.normals[i]
        v = d @ centreline.binormals[i]
        keep = (np.abs(u) <= geom.half_um[i]) & (np.abs(v) <= geom.half_um[i])
        if not keep.any():
            continue
        col = u[keep] / geom.px_um[i] + centre
        row = v[keep] / geom.px_um[i] + centre
        out.append(np.column_stack([np.full(keep.sum(), i, dtype=np.float64), row, col]))
    return np.concatenate(out) if out else np.zeros((0, 3))
