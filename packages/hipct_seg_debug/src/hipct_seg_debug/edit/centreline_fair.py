"""Junction-aware centreline fairing: smooth each vessel *through* its bifurcations.

A minimum-cost-path skeleton (:mod:`~.jin_mgf`) has the right topology and stays on
the medial axis, but its points are voxel centres: consecutive steps turn a median
45 degrees on LADAF-2021-17. Smoothing each segment on its own -- what
:func:`~.skeleton_optimise.smooth_centreline` and the DFS refinement both do, with the
junction node held still -- removes the staircase and leaves a kink at every
bifurcation (a median 20-22 degrees between a branch's first radius and the next two,
on both). The kink is not noise. A daughter's first radius of centreline runs inside
its *parent's* lumen, where it has no centre of its own; it heads wherever the path
search happened to enter the parent, and the vessel turns onto its real course only
once it is clear.

So this fits vessels, not segments:

1. **Degree-2 nodes are rejoined** (:func:`~.skeleton_optimise._contract_degree2`),
   so nothing splits a vessel except a real branch.
2. **Segments are chained into vessels** (:func:`vessel_chains`). Walking out from the
   root, a vessel continues at each junction into the outgoing segment that deflects
   least from its arriving direction, if any deflects less than
   :data:`MAX_DEFLECTION_DEG`; every other outgoing segment starts a vessel of its own.
   Each segment is labelled with its vessel's ``vessel_id``.
3. **Each vessel is faired as one curve, parents before daughters**
   (:func:`_fair_polyline`). A junction on the parent is an ordinary interior point of
   the parent's curve, so the main vessel passes through it without bending. A daughter
   then starts from wherever its parent's fit put the node, and its points within
   :data:`EMERGENCE_RADII` parent radii of it carry no data weight: the bending penalty
   alone joins the node to the daughter's real course, which is what removes the kink.

The smoother is a penalised least-squares fit on the points themselves -- data weight
in arclength quadrature against a second-derivative penalty of length ``l`` -- with
``l`` :data:`SMOOTH_RADII` local radii and never under :data:`MIN_SMOOTH_VOXELS`. A
spline *displacement* (``centreline_refine.spline_fit``'s default mode) cannot do this
job: a correction smooth over two voxels cannot cancel a staircase whose period is one.

**Containment** is the guard, as in ``refine-centreline``: an edge of the faired line
that leaves the segmentation -- smoothing cutting the inside of a tight bend -- pins its
two points back to where they were and the vessel is refitted, up to
:data:`PIN_ROUNDS` times; then the step is backtracked; and a vessel that still exits
is left as it was and reported. Edges already outside before fairing are pinned from
the start and never counted against it.

Radii and topology (apart from step 1) are never changed.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .skeleton_optimise import _contract_degree2, _radius_away, roughness

# A daughter's points within this many parent radii of its junction lie inside the
# parent's lumen and carry no data weight. The same margin `recentre` keeps clear of a
# junction (`RECENTRE_JUNCTION_MARGIN`), for the same reason: there is no centre there.
EMERGENCE_RADII = 1.5
# A vessel's own points within this many of its radii of a junction it passes through
# carry no data weight either: the medial path through a junction's confluence bends
# into the carina. Swept 0 / 1 / 1.5 on LADAF-2021-17 (both trees, jin-mgf): the median
# kink on segments continuing through a junction went 11.3 / 11.0 / 10.6 degrees.
THROUGH_RADII = 1.5
# Smoothing length, in local radii -- Jin radii, i.e. FDT, which understate a flattened
# vessel's width. Swept on the same tree with THROUGH_RADII 1.5:
#   radii  turn median/p95  junction kink median/p95  pinned for containment
#   1.0    1.8 / 8.4 deg    8.7 / 30.5 deg              380
#   1.5    1.3 / 6.9        6.5 / 26.8                  434
#   2.0    1.0 / 6.1        5.1 / 24.1                  831
#   3.0    0.7 / 13.8       3.5 / 25.8                 3907
# At 3 the fit cuts so many bends that the points pinned back into the lumen put kinks
# of their own into the line (p95 turn 13.8). The DFS refinement of the left tree is
# 1.3 / 6.2 turn and 21.7 / 50.1 kink on the same measures.
SMOOTH_RADII = 2.0
# ...and never shorter than this many voxels, whatever the radius says: Jin radii are
# FDT thicknesses, which on a flattened vessel are a voxel or two.
MIN_SMOOTH_VOXELS = 2.0
# A vessel continues through a junction only into a branch deflecting less than this.
MAX_DEFLECTION_DEG = 60.0
# No point moves further than this many local radii (or two voxels, if more).
MAX_MOVE_RADII = 1.0
# Refits pinning newly outside edges before falling back to a backtracked step.
PIN_ROUNDS = 4
# Samples along each edge for the containment test. Faired edges are under two voxels
# long, so nine samples are under a quarter-voxel apart.
EDGE_SAMPLES = 9
# A point of no data weight still needs a trace of one, or a run of them with no
# bending to constrain it (a straight stretch) would leave the system singular.
_NO_WEIGHT = 1e-6


@dataclass
class Chain:
    """One vessel: segments in order, each with the direction it is walked in."""

    segments: list = field(default_factory=list)  # (sid, reversed) node1 -> node2 if not reversed
    parent: int | None = None  # index of the vessel this one branches from
    start_node: int | None = None


@dataclass
class FairReport:
    vessels: int = 0
    branch_vessels: int = 0
    contracted_degree2: int = 0
    moved_points: int = 0
    median_move_um: float = 0.
    max_move_um: float = 0.
    clamped_points: int = 0
    pinned_points: int = 0
    backtracked_vessels: int = 0
    blocked_vessels: list = field(default_factory=list)
    outside_edges_before: int | None = None
    outside_edges_after: int | None = None
    parameters: dict = field(default_factory=dict)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    seconds: float = 0.

    def to_dict(self):
        return asdict(self)

    def describe(self) -> str:
        b, a = self.before, self.after
        lines = [
            f"faired {self.vessels} vessel(s) ({self.branch_vessels} branching from another), "
            f"{self.contracted_degree2} degree-2 node(s) rejoined",
            f"  moved {self.moved_points:,} points, median {self.median_move_um:.1f} um "
            f"max {self.max_move_um:.1f} um; {self.clamped_points} clamped, "
            f"{self.pinned_points} pinned for containment, {self.backtracked_vessels} "
            f"vessel(s) backtracked, {len(self.blocked_vessels)} left unchanged",
        ]
        if b and a:
            lines.append(
                f"  turn median {b['turn_median_deg']:.1f} -> {a['turn_median_deg']:.1f} deg, "
                f"p95 {b['turn_p95_deg']:.1f} -> {a['turn_p95_deg']:.1f} deg")
            lines.append(
                f"  junction kink median {b['kink_median_deg']:.1f} -> "
                f"{a['kink_median_deg']:.1f} deg, p95 {b['kink_p95_deg']:.1f} -> "
                f"{a['kink_p95_deg']:.1f} deg; through-angle median "
                f"{b['through_median_deg']:.1f} -> {a['through_median_deg']:.1f} deg")
        if self.outside_edges_before is not None:
            lines.append(f"  edges outside the segmentation: {self.outside_edges_before} -> "
                         f"{self.outside_edges_after}")
        lines.append(f"  ({self.seconds:.1f}s)")
        return "\n".join(lines)


# ------------------------------------------------------------------- measures


def _nanpct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def turn_stats(graph) -> dict:
    """Median and p95 turn between consecutive centreline steps, in degrees."""
    turns = []
    for sid in graph.segment_ids():
        d = np.diff(graph.coords(sid), axis=0)
        length = np.linalg.norm(d, axis=1)
        unit = d[length > 1e-9] / length[length > 1e-9, None]
        if len(unit) > 1:
            turns.append(np.degrees(np.arccos(np.clip(
                np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1., 1.))))
    turns = np.concatenate(turns) if turns else np.zeros(0)
    return dict(turn_median_deg=_nanpct(turns, 50), turn_p95_deg=_nanpct(turns, 95))


def _end_chords(graph, sid, nid, near_radii=1.0, far_radii=3.0):
    """Chords node -> `near_radii` and `near_radii` -> `far_radii` local radii along `sid`.

    None if the segment is shorter than ``far_radii + 1`` radii, so a stub is not
    judged on chords that reach its other end.
    """
    x, r = graph.coords(sid), graph.radii(sid)
    if graph.segment(sid)["node1"] != nid:
        x, r = x[::-1], r[::-1]
    if len(x) < 3:
        return None
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
    radius = max(float(np.median(r)), 1e-6)
    if s[-1] < (far_radii + 1) * radius:
        return None
    i1 = min(int(np.searchsorted(s, near_radii * radius)), len(x) - 1)
    i3 = min(int(np.searchsorted(s, far_radii * radius)), len(x) - 1)
    near, far = x[i1] - x[0], x[i3] - x[i1]
    if np.linalg.norm(near) < 1e-9 or np.linalg.norm(far) < 1e-9:
        return None
    return near / np.linalg.norm(near), far / np.linalg.norm(far)


def junction_kinks(graph) -> dict:
    """How sharply segments turn just after leaving a junction, and how straight the
    straightest pair passes through it.

    `kink` is the angle between a segment's first local radius and its next two, at
    every junction end (degree >= 3); `through` is, per junction, the smallest
    deflection between any two of its segments' outward courses. A vessel faired
    through its junction has a small `through`; a branch leaving without a kink has a
    small `kink`.
    """
    kinks, through = [], []
    for nid in graph.nodes:
        if graph.degree(nid) < 3:
            continue
        courses = []
        for sid in graph.node_segments(nid):
            chords = _end_chords(graph, sid, nid)
            if chords is None:
                continue
            near, far = chords
            kinks.append(float(np.degrees(np.arccos(np.clip(near @ far, -1., 1.)))))
            courses.append(far)
        best = [float(np.degrees(np.arccos(np.clip(-a @ b, -1., 1.))))
                for i, a in enumerate(courses) for b in courses[i + 1:]]
        if best:
            through.append(min(best))
    return dict(kink_median_deg=_nanpct(kinks, 50), kink_p95_deg=_nanpct(kinks, 95),
                through_median_deg=_nanpct(through, 50), junction_ends=len(kinks))


def measures(graph) -> dict:
    return {**turn_stats(graph), **junction_kinks(graph),
            "reversing": roughness(graph).reversing}


# ------------------------------------------------------------- vessel chains


def _oriented(graph, sid, nid):
    """`sid`'s coordinates walked away from `nid`, and whether that is node2 -> node1."""
    seg = graph.segment(sid)
    rev = seg["node1"] != nid
    x = graph.coords(sid)
    return (x[::-1] if rev else x), rev


def _leaving(graph, sid, nid, radii=2.0):
    """Unit direction leaving `nid` along `sid`, over about `radii` local radii.

    The estimate :func:`~.section_validation._through_pair` uses: a chord, because the
    first step of a voxel path points wherever the staircase turned.
    """
    x, _ = _oriented(graph, sid, nid)
    if len(x) < 2:
        return None
    r = _radius_away(graph, sid, nid)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
    k = int(np.clip(np.searchsorted(s, radii * max(r, 1e-6) if np.isfinite(r) else 0.),
                    1, len(x) - 1))
    d = x[k] - x[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def _far_node(graph, sid, nid):
    seg = graph.segment(sid)
    return seg["node2"] if seg["node1"] == nid else seg["node1"]


def vessel_chains(graph, roots=(), *, max_deflection_deg=MAX_DEFLECTION_DEG) -> list[Chain]:
    """Partition the segments into vessels, parents before the vessels branching off them.

    Breadth-first from each root (the given ones, then :func:`~.roots.auto` for any
    component without one). A segment arriving at a node already reached closes a loop:
    its vessel ends there, at a node an earlier vessel has placed, so both of its ends
    are held when it is faired.
    """
    explicit = [int(n) for n in roots if n in graph.nodes]
    try:
        from .roots import auto
        automatic = auto(graph)
    except Exception:  # noqa: BLE001 -- any component still gets a fallback root below
        automatic = []
    leaves = sorted(n for n in graph.nodes if graph.degree(n) == 1)
    starts = explicit + automatic + leaves + sorted(graph.nodes)
    cos_limit = np.cos(np.radians(max_deflection_deg))
    reached, used = set(), set()
    chains: list[Chain] = []

    for root in starts:
        if root in reached or graph.degree(root) == 0:
            continue
        reached.add(root)
        pending = deque((root, sid, None) for sid in sorted(graph.node_segments(root)))
        while pending:
            node, sid, parent = pending.popleft()
            if sid in used:
                continue
            chain = Chain(parent=parent, start_node=node)
            index = len(chains)
            chains.append(chain)
            while True:
                used.add(sid)
                chain.segments.append((sid, graph.segment(sid)["node1"] != node))
                far = _far_node(graph, sid, node)
                if far in reached:
                    break  # a terminal reached twice is impossible; this closes a loop
                reached.add(far)
                out = [s for s in sorted(graph.node_segments(far)) if s not in used]
                if not out:
                    break
                arriving = _leaving(graph, sid, far)
                best, best_key = None, None
                for cand in out:
                    leaving = _leaving(graph, cand, far)
                    if arriving is None or leaving is None:
                        continue
                    straight = float(-(arriving @ leaving))
                    if straight < cos_limit:
                        continue
                    key = (round(straight, 6), _radius_away(graph, cand, far))
                    if best_key is None or key > best_key:
                        best, best_key = cand, key
                pending.extend((far, s, index) for s in out if s != best)
                if best is None:
                    break
                node, sid = far, best
    return chains


# ----------------------------------------------------------------- the fit


def _fair_polyline(x, weights, length, fixed):
    """Penalised least squares on the points: ``sum q w |y-x|^2 + sum q l^4 |y''|^2``.

    `x` (n, 3), `weights` and `length` (n,), `fixed` indices kept exactly. The second
    derivative is the non-uniform three-point one in `x`'s own arclength, and `q` is
    arclength quadrature, so the fit does not depend on how densely the line is
    sampled; ``l`` is where it stops smoothing, as a physical length.
    """
    n = len(x)
    h = np.maximum(np.linalg.norm(np.diff(x, axis=0), axis=1), 1e-9)
    q = np.r_[h[0] / 2, (h[:-1] + h[1:]) / 2, h[-1] / 2]
    m = n - 2
    rows = np.repeat(np.arange(m), 3)
    cols = (np.arange(m)[:, None] + np.arange(3)).ravel()
    a, b = h[:-1], h[1:]
    coef = 2. / (a + b)
    vals = np.column_stack([coef / a, -coef * (1 / a + 1 / b), coef / b]).ravel()
    d2 = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))
    bend = d2.T @ sparse.diags(q[1:-1] * length[1:-1] ** 4) @ d2
    data = q * np.maximum(weights, _NO_WEIGHT)
    system = (sparse.diags(data) + bend).tocsr()
    rhs = data[:, None] * x
    fixed = np.unique(np.asarray(sorted(fixed), dtype=np.int64))
    free = np.setdiff1d(np.arange(n), fixed)
    y = x.copy()
    if len(free):
        a_ff = system[free][:, free].tocsc()
        b_f = rhs[free] - system[free][:, fixed] @ x[fixed]
        y[free] = np.asarray(spsolve(a_ff, b_f)).reshape(len(free), 3)
    return y


def _outside_edges(x, inside):
    """(n-1,) bool: edges of polyline `x` with any sample outside the segmentation."""
    if len(x) < 2:
        return np.zeros(0, dtype=bool)
    t = np.linspace(0., 1., EDGE_SAMPLES)
    pts = x[:-1, None, :] + t[None, :, None] * np.diff(x, axis=0)[:, None, :]
    return ~np.asarray(inside(pts.reshape(-1, 3)), dtype=bool).reshape(len(x) - 1, -1).all(axis=1)


def _count_outside(graph, inside):
    return sum(int(_outside_edges(graph.coords(sid), inside).sum())
               for sid in graph.segment_ids())


def _assemble(graph, chain):
    """A vessel's points end to end, its radii, and each segment's slice of them."""
    xs, rs, pieces, at = [], [], [], 0
    for k, (sid, rev) in enumerate(chain.segments):
        x, r = graph.coords(sid), graph.radii(sid)
        if rev:
            x, r = x[::-1], r[::-1]
        start = at
        if k and len(x) and np.linalg.norm(x[0] - xs[-1][-1]) <= 1e-6:
            x, r = x[1:], r[1:]
            start = at - 1  # this segment begins on the previous one's last point
        xs.append(x)
        rs.append(r)
        at += len(x)
        pieces.append((sid, rev, start, at))
    return np.vstack(xs), np.concatenate(rs), pieces


def _parent_radius(graph, chains, chain):
    """The radius of the vessel `chain` branches from, at the node it leaves from."""
    if chain.parent is None:
        return None
    sids = [s for s, _ in chains[chain.parent].segments
            if s in graph.node_segments(chain.start_node)]
    radii = [_radius_away(graph, s, chain.start_node) for s in sids]
    radii = [r for r in radii if np.isfinite(r)]
    return max(radii) if radii else None


def fair(graph, *, spacing_um, roots=(), inside=None, smooth_radii=SMOOTH_RADII,
         min_smooth_voxels=MIN_SMOOTH_VOXELS, emergence_radii=EMERGENCE_RADII,
         through_radii=THROUGH_RADII, max_deflection_deg=MAX_DEFLECTION_DEG,
         max_move_radii=MAX_MOVE_RADII, contract=True, progress=None) -> FairReport:
    """Fair every vessel of an ``EditableGraph`` in place; see the module docstring.

    `inside` maps ``(N, 3)`` micrometre points to booleans (in the segmentation); None
    skips the containment guard. `spacing_um` is the segmentation voxel size, which
    sets the smoothing and move floors.
    """
    t0 = time.monotonic()
    sp = float(spacing_um)
    report = FairReport(parameters=dict(
        smooth_radii=smooth_radii, min_smooth_voxels=min_smooth_voxels,
        emergence_radii=emergence_radii, through_radii=through_radii,
        max_deflection_deg=max_deflection_deg,
        max_move_radii=max_move_radii, spacing_um=sp, contained=inside is not None))
    report.before = measures(graph)
    if inside is not None:
        report.outside_edges_before = _count_outside(graph, inside)
    moves = []
    with graph.batch("fair centreline"):
        if contract:
            report.contracted_degree2 = _contract_degree2(graph)
        chains = vessel_chains(graph, roots, max_deflection_deg=max_deflection_deg)
        report.vessels = len(chains)
        report.branch_vessels = sum(c.parent is not None for c in chains)
        for vid, chain in enumerate(chains):
            for sid, _ in chain.segments:
                graph.set_segment_attrs(sid, {"vessel_id": vid})
        for vid, chain in enumerate(chains):
            x, r, pieces = _assemble(graph, chain)
            n = len(x)
            pre_bad = _outside_edges(x, inside) if inside is not None and n > 1 else None
            if n < 3:
                continue
            s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
            r = np.where(np.isfinite(r) & (r > 0), r, sp)
            length = np.maximum(smooth_radii * r, min_smooth_voxels * sp)
            limit = np.maximum(max_move_radii * r, 2 * sp)
            weights = np.ones(n)
            # The parent's own path is pulled towards each branch it passes, too: the
            # medial axis of a junction's confluence bends into the carina. Bridge it.
            for k in range(1, len(pieces)):
                sid_in, sid_out = pieces[k - 1][0], pieces[k][0]
                node = graph.segment(sid_in)["node2" if not pieces[k - 1][1] else "node1"]
                if graph.degree(node) < 3 or through_radii <= 0:
                    continue
                own = [_radius_away(graph, sid, node) for sid in (sid_in, sid_out)]
                own = [v for v in own if np.isfinite(v)]
                half = through_radii * max(min(own) if own else sp, sp)
                at = pieces[k][2]
                zone = np.abs(s - s[at]) < half
                weights[zone] = 0.
                limit[zone] = np.maximum(limit[zone], half)
            parent_r = _parent_radius(graph, chains, chain)
            if parent_r is not None:
                zone = s < emergence_radii * max(parent_r, sp)
                zone[0] = False
                weights[zone] = 0.
                limit[zone] = np.maximum(limit[zone], emergence_radii * max(parent_r, sp))
            fixed = {0, n - 1}
            if pre_bad is not None and pre_bad.any():
                bad = np.flatnonzero(pre_bad)
                fixed |= set(bad.tolist()) | set((bad + 1).tolist())
            pinned = 0
            y = x
            for _ in range(PIN_ROUNDS + 1):
                y = _fair_polyline(x, weights, length, fixed)
                step = np.linalg.norm(y - x, axis=1)
                over = step > limit
                if over.any():
                    y = x + (y - x) * np.where(over, limit / np.maximum(step, 1e-12), 1.)[:, None]
                if inside is None:
                    break
                new_bad = _outside_edges(y, inside) & ~pre_bad
                if not new_bad.any():
                    break
                bad = np.flatnonzero(new_bad)
                grow = (set(bad.tolist()) | set((bad + 1).tolist())) - fixed
                if not grow:
                    break
                pinned += len(grow)
                fixed |= grow
            report.clamped_points += int(over.sum())
            report.pinned_points += pinned
            if inside is not None:
                alpha = 1.
                while alpha >= 1 / 64 and (_outside_edges(x + alpha * (y - x), inside)
                                           & ~pre_bad).any():
                    alpha /= 2
                if alpha < 1 / 64:
                    report.blocked_vessels.append(vid)
                    continue
                if alpha < 1:
                    report.backtracked_vessels += 1
                    y = x + alpha * (y - x)
            step = np.linalg.norm(y - x, axis=1)
            moves.extend(step[step > 1e-9].tolist())
            for sid, rev, lo, hi in pieces:
                piece = y[lo:hi]
                graph.set_segment_coords(sid, piece[::-1] if rev else piece)
            if progress is not None:
                progress(vid + 1, len(chains))
    report.moved_points = len(moves)
    if moves:
        report.median_move_um = float(np.median(moves))
        report.max_move_um = float(np.max(moves))
    if inside is not None:
        report.outside_edges_after = _count_outside(graph, inside)
    report.after = measures(graph)
    report.seconds = time.monotonic() - t0
    return report


def frame_inside(labels, frame):
    """An `inside` test for :func:`fair` over a label lattice or decoded volume."""
    from ..crosssection import _PlaneSampler

    sampler = _PlaneSampler(labels, frame)
    return lambda xyz: sampler.at(frame.um_to_seg(xyz)) > 0


def volume_inside(volume, origin_um, spacing_um):
    """An `inside` test over a ``(nz, ny, nx)`` mask whose voxel 0 sits at `origin_um`."""
    mask = np.asarray(volume) > 0
    origin = np.asarray(origin_um, dtype=np.float64)
    spacing = np.broadcast_to(np.asarray(spacing_um, dtype=np.float64), (3,))
    upper = np.array(mask.shape[::-1])

    def inside(xyz):
        ijk = np.rint((np.asarray(xyz, dtype=np.float64) - origin) / spacing).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < upper), axis=1)
        out = np.zeros(len(ijk), dtype=bool)
        out[ok] = mask[ijk[ok, 2], ijk[ok, 1], ijk[ok, 0]]
        return out

    return inside
