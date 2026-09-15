"""Prune, smooth and re-centre a centreline, so the geometry is worth measuring.

A skeleton derived from a mask is wrong in two ways that matter downstream, and both
of them corrupt the *radius* as much as the geometry:

**Spurious side branches.** Lee thinning turns every bulge on the segmentation surface
into a short stub off the main line. Measured against the Avizo skeleton on
LADAF-2024-28, the generated one scores a bifurcation Dice of 0.425 with tp 144,
fn 6 and **fp 383** -- almost every branch point it invents is one of these. They
inflate branch counts, break Strahler order, and put fictitious carinas into any mesh
swept along the graph.

**A centreline that is not central.** HiP-CT is imaged ex vivo, so lumens are collapsed
and their cross-sections are elliptical or slit-like. Thinning keeps a line that is
topologically correct but visits the medial *ridge*, which for a slit runs along its
major axis rather than down its middle. Every cross-section measured perpendicular to
such a line is cut at a tilt, comes out elongated, and reports too large a perimeter --
so a bad centreline shows up as a bad radius, not as a visibly bad line.

Three operations, in the order they have to run:

1. :func:`prune_spurs` -- topology. A leaf branch shorter than a couple of parent radii
   is contained inside the parent's own lumen and cannot be a vessel that goes anywhere.
2. :func:`smooth_centreline` -- geometry, and mostly for the *tangent*: the plane the
   radius is measured in is perpendicular to it, so tangent noise is radius error.
3. :func:`recentre` -- moves each point onto the centroid of its own lumen cross-section.

**Why the centroid and not the distance transform maximum.** The maximum inscribed
circle is the intuitive notion of "centre", and it is the wrong one here. For a
collapsed slit the distance transform has a *flat ridge* along the major axis -- every
point on it is equally maximal -- so an EDT-max rule is free to put the centreline
anywhere along the slit, which is exactly the failure being fixed. The area centroid is
uniquely defined for any shape, degenerate or not, and for a circular section it agrees
with the EDT maximum anyway. It costs nothing extra because the cross-section is
already being cut to measure the perimeter (:mod:`~.radius_perimeter`).

**Re-centring is the dangerous step, and four bounds hold it.** Moving a point to a
centroid measured in a plane whose orientation comes from the line being corrected is a
feedback loop, and the first version of this ran open: on LADAF-2024-28 at stride 1 it
scattered 4.41% of points into out-and-back spikes -- against 0.00% in the Lee skeleton
it was given -- lengthened the centreline 11%, and threw 3% of points a full radius or
more, onto the vessel wall. Each bound answers a different failure, and all four were
added because the failure was measured:

* :func:`plane_normals` takes the cut plane's direction over a chord a couple of radii
  long. A central difference on a voxel staircase rotates 19.5 degrees between
  *neighbouring* points, so neighbours cut differently tilted planes and find unrelated
  centroids.
* :data:`RECENTRE_GROW_RADII` stops the cut window chasing a section that never closes,
  and `touches_border` makes that a refusal rather than a guess.
* :data:`RECENTRE_MAX_MOVE_SPACING` keeps a point between the two it sits between, which
  a bound expressed in radii cannot see on a finely sampled line.
* :func:`_revert_new_reversals` says the rest directly: re-centring may not leave the
  centreline doubling back worse than it found it.

**Each alone leaves between 0.03% and 2.3% of points doubling back; together, 0.026%.**
Leave any one out and the other three cover for it, which is why all four are cheap
insurance rather than one fix and three ornaments. Ablated on LADAF-2024-28 at stride 1
and ranked by the measures that are *not* the one being optimised -- longest step and
total path -- the spacing bound is the strongest single fix (513 um / 3079 mm alone) and
the reversal guard the weakest (873 um / 3183 mm), even though the reversal guard scores
best on the doubling-back count itself. It would: it is defined in terms of it. Damping
and the radius clamp turn out not to matter once the four are in place -- ``damping=1``
with a full-radius clamp gives the same 0.026% -- and are kept for the shape of the
iteration, not for the result.

:func:`roughness` reports what all four are for, on every run. The super metric does not
catch this on its own -- cl-sensitivity is a containment measure, and 4.41% of points
moved it by about 0.005, while the volume term actively *rewarded* the zig-zag, because
`V` is proportional to path length at fixed radii.

Nothing here imports ``skeleton_analysis``: that package scores skeletons, it has no
optimiser (see :mod:`~.optimise`). Scoring the result is :func:`~.optimise.compare`'s
job, and the sweep in ``optimise-skeleton --sweep`` is what turns these parameters from
assertions into measurements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# A leaf shorter than this many *parent* radii is inside the parent lumen and cannot
# be a vessel that goes anywhere. Two radii is one full parent diameter -- a daughter
# vessel has to at least clear its parent before it counts as having emerged.
PRUNE_LENGTH_FACTOR = 2.0
# ...unless it is nearly as thick as its parent, in which case it is a real vessel
# truncated by the field of view or by the segmentation, not a thinning artefact.
# This is a *protection*, so the threshold is deliberately high: it fires rarely.
PRUNE_RADIUS_RATIO = 0.8
# Free ends within this far of the lattice edge are vessels the scan cut off, and are
# never pruned however short they are. `candidates.py` guards its proposals the same way.
BOUNDARY_MARGIN_UM = 200.0
# Smoothing window, derived from the data rather than fixed, because a fixed number of
# micrometres is right for exactly one voxel size: at stride 8 the points are 528 um
# apart, so the 150 um that suits stride 1 selects each point alone and smooths nothing.
#
# Two scales bound it, and the smaller wins. Three point spacings is the *floor* of
# usefulness -- below that the window contains no neighbour. One median vessel radius is
# the *ceiling* of safety: the staircase to be removed is sub-voxel, while the curvature
# to be kept turns over a vessel width, so a window wider than the vessel starts cutting
# real corners and carrying the centreline out of its own lumen. Measured on
# LADAF-2024-28 at stride 4, an unbounded 3-spacing window (792 um) took cl-sensitivity
# from 0.951 to 0.926 and raised M_S; the swept optimum there was no smoothing at all.
SMOOTH_WINDOW_SPACINGS = 3.0
SMOOTH_WINDOW_RADII = 1.0
# No point may be smoothed further than this fraction of its own radius, so smoothing
# can never carry a centreline out of its own lumen.
SMOOTH_MAX_MOVE_FRAC = 0.5
# Nor may re-centring, and for the same reason `SMOOTH_MAX_MOVE_FRAC` gives: half a radius
# still leaves the point inside its own lumen, a whole one puts it on the wall. This was 1.0,
# and measurably too loose -- on LADAF-2024-28 at stride 1 it produced a visible pile-up of
# points sitting at exactly 0.9-1.1 radii from the input skeleton, i.e. the clamp binding
# rather than protecting. It is also applied against the coordinates re-centring *started*
# from rather than the previous pass's, so N passes can no longer walk N radii.
RECENTRE_MAX_MOVE_FRAC = 0.5
# The cut plane's normal is estimated over a chord this many local radii long, not from the
# +-1-point central difference this used to use. A Lee skeleton is a voxel staircase turning
# 45 degrees at the median point, and a central difference on it gives a normal that rotates
# a median 19.5 degrees (p95 39) between *neighbouring* points. Neighbours then cut differently
# tilted sections of the same vessel, land on unrelated centroids, and scatter -- which is the
# zig-zag this constant exists to remove. A chord over a vessel width averages the staircase
# out while still following real curvature, which turns over several widths.
RECENTRE_TANGENT_RADII = 2.0
# Move only this fraction of the way to the centroid each pass. Full steps make the iteration
# oscillate between passes rather than converge, because each pass re-estimates the plane from
# the line the last one moved; a damped step is the standard remedy and costs one more pass.
RECENTRE_DAMPING = 0.5
# How far the cut window may grow, in expected radii, before the section is called untrustworthy.
# `crosssection.cut` doubles its window while the blob touches the border, to `max_half` -- 64
# voxels, a 4.2 mm half-width at stride 1. That is right for *measuring* a big vessel and wrong
# for *locating* one: an oblique plane cuts a streak along the vessel that touches the border at
# any width, so the growth runs away and the centroid ends up somewhere down the vessel.
RECENTRE_GROW_RADII = 4.0
# Which connected component the centroid is taken over. 8-connectivity is the honest notion of
# one lumen (4- splits a diagonally pinched section and the centroid then sits in whichever half
# holds the centre pixel), but it is also the labelling that bridges to a vessel touching at a
# single corner -- the ex-vivo collapse case. Neither is obviously right, so this is a swept
# parameter rather than an assertion. See `optimise-skeleton --sweep "recentre-blob=blob4,blob8"`.
RECENTRE_BLOB = "blob8"
# ...and no point may move further than this fraction of the gap to its nearer neighbour.
# A radius bound keeps a point inside its own lumen; it says nothing about whether the point
# is still *between* the two points it used to sit between. Where the centreline is sampled
# more finely than it is wide -- the Avizo graph's points sit 0.40 radii apart -- a move of
# half a radius is a move past the neighbour, and the segment folds. Below 0.5 the chain
# provably cannot cross: two points each moving less than half the gap between them can meet
# but not swap. This is the guard the zig-zag needed on an already-smooth input, where the
# tangent was never the problem.
RECENTRE_MAX_MOVE_SPACING = 0.5
# Points within this many local radii of a node are not re-centred. Approaching a
# bifurcation the perpendicular plane starts to cut *both* daughters, which are one
# connected component, so its centroid slides into the crotch between them -- outside
# either lumen. Measured on LADAF-2024-28 at stride 4, re-centring without this margin
# dropped cl-sensitivity from 0.949 to 0.871. `radius_repair.JUNCTION_MARGIN` exists
# for the same reason, and `candidates._radius_along` refuses to read a radius at a
# junction for the same underlying one.
RECENTRE_JUNCTION_MARGIN = 1.5


@dataclass
class PruneReport:
    n_removed: int = 0
    n_rounds: int = 0
    removed_length_um: float = 0.0
    kept_thick: int = 0
    kept_boundary: int = 0
    n_contracted: int = 0

    def describe(self) -> str:
        return (
            f"{self.n_removed} spur(s) removed in {self.n_rounds} round(s), "
            f"{self.removed_length_um / 1000.0:.1f} mm of centreline; "
            f"{self.n_contracted} degree-2 node(s) contracted; "
            f"kept {self.kept_thick} thick and {self.kept_boundary} at the boundary"
        )


@dataclass
class MoveReport:
    label: str
    n_moved: int = 0
    n_clamped: int = 0
    n_skipped: int = 0
    n_near_junction: int = 0
    n_truncated: int = 0
    n_would_fold: int = 0
    n_reverted: int = 0
    n_window_too_small: int = 0
    median_move_um: float = 0.0
    max_move_um: float = 0.0

    def describe(self) -> str:
        line = (
            f"{self.label}: {self.n_moved} point(s) moved, median "
            f"{self.median_move_um:.1f} um, max {self.max_move_um:.1f} um; "
            f"{self.n_clamped} clamped, {self.n_skipped} skipped"
        )
        if self.n_near_junction:
            line += f", {self.n_near_junction} held near a junction"
        if self.n_truncated:
            line += (
                f", {self.n_truncated} refused for a section that never closed inside "
                f"its window"
            )
        if self.n_would_fold:
            line += f", {self.n_would_fold} shortened to stay between their neighbours"
        if self.n_reverted:
            line += f", {self.n_reverted} put back rather than double back"
        if self.n_window_too_small:
            line += (
                f"\n    [warning] {self.n_window_too_small} segment(s) had no "
                f"neighbouring point inside the smoothing window, so smoothing did "
                f"nothing there - raise --smooth-um above the point spacing"
            )
        return line


@dataclass
class OptimiseReport:
    loops: LoopReport = field(default_factory=lambda: LoopReport())
    prune: PruneReport = field(default_factory=PruneReport)
    moves: list = field(default_factory=list)
    smooth: object | None = None  # a `smoothers.SmoothResult`, when one ran
    roughness: object | None = None  # a `Roughness`, measured on the finished graph
    seconds: dict = field(default_factory=dict)

    def describe(self) -> str:
        lines = ["  " + self.loops.describe(), "  " + self.prune.describe()]
        lines += ["  " + m.describe() for m in self.moves]
        if self.smooth is not None:
            lines.append("  " + self.smooth.describe())
        if self.roughness is not None:
            lines.append("  " + self.roughness.describe())
        lines.append("  " + ", ".join(f"{k} {v:.1f}s" for k, v in self.seconds.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------- loops


@dataclass
class LoopBreak:
    """One cycle edge that was deleted, kept so the decision can be reviewed."""

    seg_id: int
    length_um: float
    radius_um: float
    xyz: tuple


@dataclass
class LoopReport:
    breaks: list = field(default_factory=list)
    n_cycles_before: int = 0
    n_cycles_after: int = 0

    def describe(self) -> str:
        if not self.breaks:
            return f"{self.n_cycles_before} cycle(s); none broken"
        r = [b.radius_um for b in self.breaks]
        return (
            f"{self.n_cycles_before} cycle(s) -> {self.n_cycles_after}; "
            f"{len(self.breaks)} edge(s) removed, radius {min(r):.0f}-{max(r):.0f} um, "
            f"total {sum(b.length_um for b in self.breaks) / 1000.0:.1f} mm"
        )


def _cycle_basis(graph):
    """``(independent cycles as node rings, segment ids the basis cannot see)``.

    ``cycle_basis`` needs a simple graph, so it misses two things a voxel skeleton
    readily produces: a self-loop, and a pair of parallel segments between the same
    two nodes -- which is precisely the shape a vessel collapsed in its middle and
    segmented as two tubes takes. Both are cycles; both are returned separately.
    """
    import networkx as nx

    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes)
    seen: set[tuple[int, int]] = set()
    extra: list[int] = []
    for seg in graph.segments:
        a, b = seg["node1"], seg["node2"]
        if a == b:
            extra.append(seg["id"])
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            extra.append(seg["id"])  # a parallel edge is a two-segment loop
            continue
        seen.add(key)
        simple.add_edge(a, b)
    return nx.cycle_basis(simple), extra


def _weakness(graph, sid: int):
    """Sort key picking the edge of a cycle least likely to be a real vessel.

    Smallest mean radius first -- a false bridge welded between two vessels, or the
    thin second lumen of a vessel segmented as two, is narrower than either real
    vessel -- then greatest length, because of two equally thin candidates the longer
    one is the less likely to be a genuine short connector.
    """
    radii = graph.radii(sid)
    return (
        float(np.mean(radii)) if len(radii) else float("inf"),
        -_seg_length_um(graph, sid),
        int(sid),
    )


def count_cycles(graph) -> int:
    """Independent cycles: ``E - N + components``, the graph's first Betti number."""
    n_edges = len(graph.segments)
    nodes = {n for seg in graph.segments for n in (seg["node1"], seg["node2"])}
    return int(n_edges - len(nodes) + len(graph.components()))


def remove_loops(graph, *, max_rounds: int | None = None) -> LoopReport:
    """Break every cycle, leaving a forest. Returns what was removed and from where.

    **This is a prior about the anatomy, not a fact about the image.** HiP-CT images
    coronary arteries ex vivo, and at this calibre the arterial tree has no
    anastomoses -- so a loop in the skeleton is one of the two failures this whole
    repository exists to find: a vessel that collapsed in the middle and was segmented
    as two parallel tubes that rejoin, or a segmentation that has run into a
    neighbouring vessel and welded them together. Either way the loop is an artefact.

    The edge broken in each cycle is the one with the **smallest mean radius**, tied
    on the greatest length. A false bridge between two vessels is thinner than either
    of them, so the smallest radius is where the segmentation is least sure; and of two
    equally thin candidates the longer one is the less likely to be a real short
    connector. Every break is recorded with its position and radius, because unlike
    pruning this operation destroys topology that cannot be recovered by re-running.

    ``networkx.cycle_basis`` gives an independent cycle per iteration rather than every
    cycle, so the loop re-derives the basis after each break: removing one edge can
    dissolve several members of the basis at once, and re-deriving is what stops a
    figure-of-eight losing three edges where two suffice.

    Note this deliberately moves the graph *away* from the segmentation's Euler
    characteristic. :class:`~.supermetric.ImageTerms` carries ``tree_chi`` so the super
    metric scores against the tree ideal instead and does not fight the prior; see
    ``SKELETONISATION.md``.
    """
    report = LoopReport()
    report.n_cycles_before = count_cycles(graph)

    # One round breaks one edge and so kills at least one cycle, but contraction can
    # expose a parallel pair that was hidden behind a degree-2 node, so allow slack.
    # A fixed cap would silently stop early: stride 4 alone needs 30 rounds, and the
    # full-resolution graph has far more cycles than any round number chosen by hand.
    if max_rounds is None:
        max_rounds = 2 * report.n_cycles_before + 10

    for _ in range(max(int(max_rounds), 1)):
        cycles, unseen = _cycle_basis(graph)
        if not cycles and not unseen:
            break

        if unseen:
            # A self-loop or a parallel pair: the thinnest of them goes first.
            candidates = [_weakness(graph, s) for s in unseen]
        else:
            ring = cycles[0]
            candidates = []
            for a, b in zip(ring, ring[1:] + ring[:1]):
                for s in graph.node_segments(a) & graph.node_segments(b):
                    candidates.append(_weakness(graph, s))
        if not candidates:
            break
        candidates.sort()
        sid = candidates[0][2]

        radii = graph.radii(sid)
        coords = graph.coords(sid)
        mid = coords[len(coords) // 2] if len(coords) else np.zeros(3)
        report.breaks.append(LoopBreak(
            seg_id=int(sid),
            length_um=_seg_length_um(graph, sid),
            radius_um=float(np.mean(radii)) if len(radii) else float("nan"),
            xyz=tuple(float(v) for v in mid),
        ))
        with graph.batch("break loop"):
            graph.delete_segment(sid)
        _contract_degree2(graph)

    report.n_cycles_after = count_cycles(graph)
    return report


# --------------------------------------------------------------------- pruning


def _seg_length_um(graph, sid: int) -> float:
    coords = graph.coords(sid)
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())


def _radius_away(graph, sid: int, nid: int, factor: float = 2.0) -> float:
    """Radius on `sid`, walked `factor` local radii away from node `nid`.

    Never read the radius *at* a junction. The distance transform there is the
    distance to the outside of the whole carina, which is larger than either
    vessel's own radius -- `candidates._radius_along` exists for the same reason,
    and this is its ``EditableGraph`` counterpart (that one indexes a
    ``SpatialGraph`` by edge number, which this module does not have).
    """
    coords = graph.coords(sid)
    rad = graph.radii(sid)
    if len(coords) == 0:
        return float("nan")
    if len(coords) == 1:
        return float(rad[0])
    node = np.asarray(graph.nodes[nid][:3], dtype=np.float64)
    if np.linalg.norm(coords[0] - node) > np.linalg.norm(coords[-1] - node):
        coords, rad = coords[::-1], rad[::-1]
    s = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
    )
    k = int(np.searchsorted(s, min(factor * float(rad[0]), float(s[-1]))))
    return float(rad[min(k, len(rad) - 1)])


def _near_boundary(xyz, bbox_um, margin_um: float) -> bool:
    lo = np.asarray(bbox_um[0::2], dtype=np.float64) + margin_um
    hi = np.asarray(bbox_um[1::2], dtype=np.float64) - margin_um
    p = np.asarray(xyz, dtype=np.float64)
    return bool(np.any(p < lo) or np.any(p > hi))


def prune_spurs(
    graph,
    *,
    length_factor: float = PRUNE_LENGTH_FACTOR,
    radius_ratio: float = PRUNE_RADIUS_RATIO,
    min_length_um: float | None = None,
    bbox_um=None,
    boundary_margin_um: float = BOUNDARY_MARGIN_UM,
    max_rounds: int = 8,
) -> PruneReport:
    """Delete leaf branches too short to be vessels, to a fixed point.

    A *spur* is a segment with one endpoint of degree 1 and the other of degree 3 or
    more. It goes when its arclength is below ``length_factor`` times the radius of
    the thickest other branch at its junction -- measured away from that junction,
    never at it.

    Three things are never pruned:

    * a spur whose own radius is at least ``radius_ratio`` of its parent's. That is a
      real vessel the scan or the segmentation cut short, not a thinning artefact;
    * a free end within ``boundary_margin_um`` of the lattice edge, for the same reason;
    * the last segment of a connected component, which would delete the component.

    Iterating matters: removing a spur drops its junction to degree 2, which can expose
    the next spur along and, on a chain of them, a whole false twig. Between rounds the
    surviving pair at each emptied junction is contracted back into one segment --
    otherwise a single vessel arrives downstream as two segments joined by a node that
    means nothing, and every per-segment measure counts it twice.

    *Why not ``skeleton_to_graph(min_branch_voxels=...)``:* that prunes chains during the
    voxel trace, and :mod:`~.reskeletonise` already records why it is not enough -- "a
    spur one voxel off a 26-connected line is itself adjacent to three line voxels, so
    it reads as a junction, and a voxel-level prune stops one short of removing it."
    """
    report = PruneReport()

    for _ in range(max(int(max_rounds), 1)):
        protected = {sid for comp in graph.components() if len(comp) == 1 for sid in comp}
        doomed: list[tuple[int, float]] = []

        for seg in list(graph.segments):
            sid = seg["id"]
            if sid in protected:
                continue
            n1, n2 = seg["node1"], seg["node2"]
            d1, d2 = graph.degree(n1), graph.degree(n2)
            if d1 == 1 and d2 >= 3:
                tip, junction = n1, n2
            elif d2 == 1 and d1 >= 3:
                tip, junction = n2, n1
            else:
                continue

            siblings = [s for s in graph.node_segments(junction) if s != sid]
            if not siblings:
                continue
            r_parent = max(
                (_radius_away(graph, s, junction) for s in siblings), default=float("nan")
            )
            if not np.isfinite(r_parent) or r_parent <= 0:
                continue

            limit = length_factor * r_parent
            if min_length_um is not None:
                limit = max(limit, float(min_length_um))
            if _seg_length_um(graph, sid) >= limit:
                continue

            if bbox_um is not None and _near_boundary(
                graph.nodes[tip][:3], bbox_um, boundary_margin_um
            ):
                report.kept_boundary += 1
                continue

            if _radius_away(graph, sid, junction) >= radius_ratio * r_parent:
                report.kept_thick += 1
                continue

            doomed.append((sid, _seg_length_um(graph, sid)))

        if not doomed:
            break

        # One spur per junction per round. Taking every child of a bifurcation at
        # once can delete a whole real fork whose two halves are each individually
        # short; leaving the rest to the next round re-measures them against the
        # junction they actually have now.
        seen: set[int] = set()
        removed = 0
        with graph.batch("prune spurs"):
            for sid, length in doomed:
                if not graph.has_segment(sid):
                    continue
                seg = graph.segment(sid)
                junction = (
                    seg["node2"] if graph.degree(seg["node1"]) == 1 else seg["node1"]
                )
                if junction in seen:
                    continue
                seen.add(junction)
                graph.delete_segment(sid)
                report.removed_length_um += length
                removed += 1

        report.n_removed += removed
        report.n_rounds += 1
        report.n_contracted += _contract_degree2(graph)
        if removed == 0:
            break

    return report


def _oriented(graph, sid: int, nid: int, *, tail: bool):
    """A segment's coords and radii ordered so `nid`'s end is last (`tail`) or first."""
    coords, rad = graph.coords(sid), graph.radii(sid)
    seg = graph.segment(sid)
    at_start = seg["node1"] == nid
    if seg["node1"] == seg["node2"]:  # a self-loop has no far end to orient by
        at_start = True
    far = seg["node2"] if at_start else seg["node1"]
    # `nid` sits at index 0 when it is node1. Put it last for `tail`, first otherwise.
    if at_start == tail:
        coords, rad = coords[::-1], rad[::-1]
    return coords, rad, far


def _contract_degree2(graph) -> int:
    """Rejoin the two segments meeting at a degree-2 node. Returns how many went.

    Dropping a spur leaves its junction behind carrying only two surviving chains,
    so one vessel arrives downstream as two segments joined by a node that means
    nothing -- and every per-segment measure, Strahler order included, counts it
    twice. :func:`~.reskeletonise.contract_degree2` does this for the same reason
    after a local re-skeletonisation, but it rewrites a whole ``Triple``; doing it
    here through ``add_segment``/``delete_segment`` keeps every step on the history,
    so pruning stays undoable in the GUI like every other edit.
    """
    joined = 0
    changed = True
    while changed:
        changed = False
        for nid in [n for n in list(graph.nodes) if graph.degree(n) == 2]:
            sids = list(graph.node_segments(nid))
            if len(sids) != 2 or sids[0] == sids[1]:
                continue
            ca, ra, far_a = _oriented(graph, sids[0], nid, tail=True)
            cb, rb, far_b = _oriented(graph, sids[1], nid, tail=False)
            if far_a == far_b:
                continue  # merging would make a self-loop; leave the node in place
            # Both chains carry their own copy of the shared junction point.
            if np.linalg.norm(ca[-1] - cb[0]) <= 1e-6:
                cb, rb = cb[1:], rb[1:]
            if len(ca) + len(cb) < 2:
                continue
            attrs = {
                k: v for k, v in graph.segment(sids[0]).items()
                if k not in ("id", "node1", "node2", "point_ids")
            }
            with graph.batch("contract degree-2 node"):
                graph.add_segment(far_a, far_b, np.vstack([ca, cb]),
                                  np.concatenate([ra, rb]), attrs=attrs)
                graph.delete_segment(sids[0])
                graph.delete_segment(sids[1])
            joined += 1
            changed = True
    return joined


# ------------------------------------------------------------------- smoothing


def _gaussian_smooth(coords: np.ndarray, window_um: float) -> tuple[np.ndarray, bool]:
    """Arclength-parameterised Gaussian smoothing of one polyline.

    Parameterised by arclength rather than by index because the points are not
    evenly spaced -- a voxel skeleton steps sqrt(3) times further on a diagonal than
    on an axis, so an index window is a physically variable window.

    Returns ``(smoothed, reached_neighbours)``. A window narrower than the point
    spacing selects each point alone, so the weighted average returns it unchanged
    and smoothing silently does nothing -- which is exactly the sort of quiet no-op
    that gets mistaken for "smoothing did not help". The flag is what
    :func:`smooth_centreline` reports instead.
    """
    n = len(coords)
    if n < 3 or window_um <= 0:
        return coords.copy(), False
    step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    sigma = window_um / 2.0
    out = coords.copy()
    reached = False
    # Only the interior moves; the ends belong to nodes and anchor the topology.
    for i in range(1, n - 1):
        d = s - s[i]
        near = np.abs(d) <= window_um
        if int(near.sum()) > 1:
            reached = True
        w = np.exp(-0.5 * (d[near] / sigma) ** 2)
        out[i] = (coords[near] * w[:, None]).sum(axis=0) / w.sum()
    out[0], out[-1] = coords[0], coords[-1]
    return out, reached


def _clamp_moves(old: np.ndarray, new: np.ndarray, radii: np.ndarray,
                 max_move_frac: float) -> tuple[np.ndarray, int]:
    """Limit each point's displacement to `max_move_frac` of its own radius."""
    delta = new - old
    dist = np.linalg.norm(delta, axis=1)
    limit = np.maximum(max_move_frac * np.asarray(radii, dtype=np.float64), 1e-9)
    over = dist > limit
    if over.any():
        scale = np.ones_like(dist)
        scale[over] = limit[over] / dist[over]
        new = old + delta * scale[:, None]
    return new, int(over.sum())


def _arclength(coords: np.ndarray) -> np.ndarray:
    """Cumulative distance along `coords`, starting at 0."""
    if len(coords) < 2:
        return np.zeros(len(coords))
    step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def plane_normals(coords: np.ndarray, radii: np.ndarray, arclen: np.ndarray,
                  tangent_radii: float = RECENTRE_TANGENT_RADII) -> np.ndarray:
    """Local direction at every point, as a chord spanning `tangent_radii` radii each way.

    The alternative -- ``np.gradient(coords, axis=0)`` -- is a chord between the two
    *adjacent* points, which on a thinned voxel skeleton is a chord between two staircase
    steps and points wherever the staircase happened to turn. Measured on LADAF-2024-28 at
    stride 1 it rotates a median 19.5 degrees from one point to the next, so consecutive
    cut planes disagree about which way the vessel runs, and the centroids they find scatter.

    Spanning a physical distance instead makes the estimate independent of point spacing,
    which is the same reason :data:`SMOOTH_WINDOW_SPACINGS` is expressed in spacings and
    :data:`SMOOTH_WINDOW_RADII` in radii. The span is measured in each point's *own* radius
    so a capillary is not averaged over the length that suits an artery.

    Always spans at least the immediate neighbours, so this degrades to the central
    difference on a segment too short to hold the window rather than returning nothing.
    """
    n = len(coords)
    if n < 3:
        return np.gradient(coords, axis=0) if n > 1 else np.zeros_like(coords)

    want = np.maximum(np.asarray(radii, dtype=np.float64), 0.0) * float(tangent_radii)
    here = np.arange(n)
    lo = np.searchsorted(arclen, arclen - want, side="left")
    hi = np.searchsorted(arclen, arclen + want, side="right") - 1
    # `here -/+ 1` keeps a real chord when the window is narrower than the point spacing,
    # and the clip keeps it on the segment at the two ends.
    lo = np.clip(np.minimum(lo, here - 1), 0, n - 1)
    hi = np.clip(np.maximum(hi, here + 1), 0, n - 1)

    tangents = coords[hi] - coords[lo]
    # A closed or doubled-back window leaves a zero chord with no direction in it; the
    # central difference is the honest fallback rather than an arbitrary axis.
    dead = np.linalg.norm(tangents, axis=1) < 1e-9
    if dead.any():
        tangents[dead] = np.gradient(coords, axis=0)[dead]
    return tangents


@dataclass
class Roughness:
    """How much a centreline doubles back on itself. The zig-zag, as a number.

    `reversing` is the statistic that matters: the share of interior points whose two
    neighbours lie on the *same side*, i.e. where the line turns through more than
    `TURN_DEG` and heads back the way it came. A real vessel does not do this, so any
    non-zero value is artefact. It is reported because the super metric does not catch it --
    cl-sensitivity is a containment measure and a defect affecting 3% of points barely
    moves it, which is exactly how this went unnoticed.
    """

    TURN_DEG = 120.0

    n_points: int = 0
    reversing: float = 0.0  # fraction of interior points turning more than TURN_DEG
    median_step_um: float = 0.0
    max_step_um: float = 0.0
    length_mm: float = 0.0

    def describe(self) -> str:
        return (
            f"roughness: {100 * self.reversing:.2f}% of points turn >{self.TURN_DEG:.0f} deg, "
            f"step median {self.median_step_um:.1f} um max {self.max_step_um:.1f} um, "
            f"{self.length_mm:.1f} mm total"
        )


def roughness(graph) -> Roughness:
    """Measure `graph`'s centreline for doubling-back. Cheap enough to run every time."""
    steps: list = []
    turns: list = []
    for sid in graph.segment_ids():
        coords = graph.coords(sid)
        if len(coords) < 2:
            continue
        d = np.diff(coords, axis=0)
        length = np.linalg.norm(d, axis=1)
        steps.append(length)
        real = length > 1e-9
        unit = d[real] / length[real, None]
        if len(unit) > 1:
            cos = np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)
            turns.append(np.degrees(np.arccos(cos)))
    if not steps:
        return Roughness()
    steps = np.concatenate(steps)
    turns = np.concatenate(turns) if turns else np.zeros(0)
    return Roughness(
        n_points=int(len(steps)),
        reversing=float((turns > Roughness.TURN_DEG).mean()) if len(turns) else 0.0,
        median_step_um=float(np.median(steps)),
        max_step_um=float(np.max(steps)),
        length_mm=float(steps.sum()) / 1000.0,
    )


def _clamp_to_spacing(old: np.ndarray, new: np.ndarray,
                      frac: float) -> tuple[np.ndarray, int]:
    """Limit each point's displacement to `frac` of the gap to its nearer neighbour.

    Ordering along a segment is a property of the *chain*, not of any one point, and the
    radius clamp cannot see it: half a radius is a small move for a point in a wide vessel
    and a leap past its neighbour when the line is sampled four times per radius. With
    `frac <= 0.5` no two points can cross, because each moves less than half the distance
    between them.
    """
    if len(old) < 2 or frac <= 0:
        return new, 0
    gap = np.linalg.norm(np.diff(old, axis=0), axis=1)
    # Each interior point is bounded by the smaller of the two gaps it sits between; the
    # ends have only one, and are not moved by `recentre` anyway.
    nearer = np.empty(len(old))
    nearer[0], nearer[-1] = gap[0], gap[-1]
    if len(old) > 2:
        nearer[1:-1] = np.minimum(gap[:-1], gap[1:])

    delta = new - old
    dist = np.linalg.norm(delta, axis=1)
    limit = frac * nearer
    over = dist > limit
    if over.any():
        scale = np.ones_like(dist)
        scale[over] = limit[over] / np.maximum(dist[over], 1e-30)
        new = old + delta * scale[:, None]
    return new, int(over.sum())


def _turn_deg(coords: np.ndarray) -> np.ndarray:
    """Turn angle at every point, in degrees; 0 at the two ends."""
    out = np.zeros(len(coords))
    if len(coords) < 3:
        return out
    d = np.diff(coords, axis=0)
    length = np.linalg.norm(d, axis=1)
    unit = d / np.maximum(length, 1e-12)[:, None]
    cos = np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)
    out[1:-1] = np.degrees(np.arccos(cos))
    return out


def _revert_new_reversals(old: np.ndarray, new: np.ndarray,
                          threshold_deg: float, rounds: int = 12) -> tuple[np.ndarray, int]:
    """Undo any move that leaves a point doubling back worse than it already did.

    The bounds above are geometric and local; this one is stated directly in the quantity
    being protected, and it is the only guard that holds on an *oversampled* graph. The
    Avizo export puts its points 0.16 local radii apart, so 9.69% of them already turn
    past 120 degrees before anything touches them; at that density a 15 um nudge takes a
    110-degree corner to 137. No radius bound can see that, and no spacing bound stops it,
    because 15 um really is a small fraction of the 60 um gap.

    So the rule is simply that re-centring may not make the doubling-back worse.

    A point's turn angle is a property of it *and its two neighbours*, so putting one point
    back changes two more angles -- and reverting only the offender leaves those two
    unexamined. Each round therefore takes the offender's neighbours with it, and the set
    of reverted points only ever grows, so the loop terminates: at the fixpoint any point
    still worse than it started would have to have its whole neighbourhood already back at
    `old`, which would make its angle exactly what it started as.
    """
    if len(old) < 3:
        return new, 0
    before = _turn_deg(old)
    reverted = np.zeros(len(old), bool)
    for _ in range(rounds):
        now = _turn_deg(new)
        worse = (now > threshold_deg) & (now > before)
        if not worse.any():
            break
        grow = worse.copy()
        grow[:-1] |= worse[1:]
        grow[1:] |= worse[:-1]
        if not (grow & ~reverted).any():
            break
        reverted |= grow
        new = np.where(reverted[:, None], old, new)
    return new, int(reverted.sum())


def median_point_spacing(graph) -> float:
    """Median distance between consecutive centreline points, in um."""
    steps = []
    for sid in graph.segment_ids():
        coords = graph.coords(sid)
        if len(coords) > 1:
            steps.append(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    return float(np.median(np.concatenate(steps))) if steps else 0.0


def default_smooth_window(graph) -> float:
    """The smoothing window this graph implies -- see :data:`SMOOTH_WINDOW_SPACINGS`."""
    spacing = SMOOTH_WINDOW_SPACINGS * median_point_spacing(graph)
    radii = [graph.radii(sid) for sid in graph.segment_ids()]
    radii = [r for r in radii if len(r)]
    if not radii:
        return spacing
    ceiling = SMOOTH_WINDOW_RADII * float(np.median(np.concatenate(radii)))
    return min(spacing, ceiling) if ceiling > 0 else spacing


def _invented(graph, sid: int, n: int) -> np.ndarray:
    """(n,) bool -- points on this segment that Avizo interpolated."""
    from .interpolation import mask_for_segment

    flat = mask_for_segment(graph, sid)
    return flat if flat.size == n else np.zeros(n, dtype=bool)


def _smooth_real_runs(coords, invented, window_um: float):
    """Smooth each run of real points on its own, treating a fill as a break.

    Holding the invented points still would not be enough. A Gaussian window spanning
    a fill averages the real points *with* it, so a straight invented line pulls the
    genuine centreline either side of it towards itself -- the artefact would end up
    reshaping the data it was supposed to be excluded from. Smoothing run by run means
    the fill is not in anybody's window.
    """
    out = np.asarray(coords, dtype=np.float64).copy()
    reached = True
    real = ~np.asarray(invented, dtype=bool)
    i = 0
    while i < len(real):
        if not real[i]:
            i += 1
            continue
        j = i
        while j < len(real) and real[j]:
            j += 1
        if j - i >= 3:
            piece, ok = _gaussian_smooth(out[i:j], window_um)
            out[i:j] = piece
            reached = reached and ok
        i = j
    return out, reached


def smooth_centreline(
    graph,
    *,
    window_um: float | None = None,
    max_move_frac: float = SMOOTH_MAX_MOVE_FRAC,
    label: str = "smooth",
) -> MoveReport:
    """Smooth every segment's interior, clamped so a point cannot leave its lumen.

    Runs before re-centring and again after. The point is less the look of the line
    than the *tangent*: the cross-section a radius is measured in is perpendicular to
    it, and on a raw voxel skeleton consecutive steps can differ by 45 degrees, which
    tilts the cut and inflates the perimeter it reports.

    `window_um` defaults to **three times the median point spacing**, not to a fixed
    number of micrometres. A fixed default is right for exactly one voxel size: at
    stride 8 the points are 528 um apart, so the 150 um that suits stride 1 selects
    each point alone and smooths nothing at all. Scaling to the data means the window
    always spans a few points, whatever the stride and whether the graph came from a
    voxel skeleton or from Avizo. Pass a number to override it.
    """
    report = MoveReport(label)
    moves: list[float] = []
    if window_um is None:
        window_um = default_smooth_window(graph)
    with graph.batch(label):
        for sid in graph.segment_ids():
            coords = graph.coords(sid)
            if len(coords) < 3:
                report.n_skipped += len(coords)
                continue
            invented = _invented(graph, sid, len(coords))
            if invented.any():
                new, reached = _smooth_real_runs(coords, invented, window_um)
                report.n_skipped += int(invented.sum())
            else:
                new, reached = _gaussian_smooth(coords, window_um)
            if not reached:
                report.n_window_too_small += 1
            new, clamped = _clamp_moves(coords, new, graph.radii(sid), max_move_frac)
            dist = np.linalg.norm(new - coords, axis=1)
            if not (dist > 1e-9).any():
                continue
            graph.set_segment_coords(sid, new)
            report.n_moved += int((dist > 1e-9).sum())
            report.n_clamped += clamped
            moves.extend(dist[dist > 1e-9].tolist())
    if moves:
        report.median_move_um = float(np.median(moves))
        report.max_move_um = float(np.max(moves))
    return report


# ----------------------------------------------------------------- re-centring


def recentre(
    graph,
    frame,
    labels,
    *,
    max_move_frac: float = RECENTRE_MAX_MOVE_FRAC,
    max_move_spacing: float = RECENTRE_MAX_MOVE_SPACING,
    junction_margin: float = RECENTRE_JUNCTION_MARGIN,
    tangent_radii: float = RECENTRE_TANGENT_RADII,
    damping: float = RECENTRE_DAMPING,
    grow_radii: float = RECENTRE_GROW_RADII,
    blob: str = RECENTRE_BLOB,
    origin: dict | None = None,
    max_half: int = 64,
    label: str = "recentre",
    progress=None,
) -> MoveReport:
    """Move every interior point onto the centroid of its own lumen cross-section.

    The cross-section is cut perpendicular to the local tangent by
    :func:`~..crosssection.cut`, which is the same cut :mod:`~.radius_perimeter`
    measures the perimeter in -- one notion of "the lumen here", used by both.

    Junction nodes are **not** moved, and neither is any point within
    `junction_margin` local radii of one. A node is shared by every segment meeting
    there, so moving it is a topological act rather than a geometric one -- but the
    deeper reason applies to its neighbours too: approaching a bifurcation the
    perpendicular plane begins to cut *both* daughter vessels, which are one connected
    component, so the centroid slides into the crotch between them and is outside
    either lumen. Measured on LADAF-2024-28 at stride 4, dropping this margin cost
    cl-sensitivity 0.949 -> 0.871.

    Points whose position is outside the mask, or whose section is too small or too
    truncated to locate, are left where they are; the following smoothing pass pulls them
    along with their neighbours.

    Four things keep a pass from doing harm, and every one was added after measuring what
    happens without it -- see the module docstring for the numbers:

    * The plane's normal comes from :func:`plane_normals`, a chord over `tangent_radii`
      local radii, not from the adjacent points. This is the one that matters most.
    * A cut that still reaches the edge of its own window (`touches_border`) is refused
      rather than used. It is not a cross-section, and its centroid can be anywhere.
    * :func:`_clamp_to_spacing` keeps each point between the two it sits between, which no
      bound expressed in radii can do on a finely sampled line.
    * :func:`_revert_new_reversals` says the rest directly: whatever the bounds allowed,
      a pass may not leave the centreline doubling back worse than it found it.

    The step is also damped, and clamped against `origin` -- where re-centring *began* --
    rather than against the previous pass, so repeated passes converge on the lumen centre
    instead of walking `max_move_frac` radii per pass away from the input.
    """
    from ..crosssection import _PlaneSampler, cut

    if blob not in ("blob4", "blob8"):
        raise ValueError(f"recentre blob must be 'blob4' or 'blob8', not {blob!r}")

    report = MoveReport(label)
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    dims = np.asarray(frame.seg_dims, dtype=np.float64)
    moves: list[float] = []

    sids = graph.segment_ids()
    with graph.batch(label):
        for n_done, sid in enumerate(sids, 1):
            coords = graph.coords(sid)
            radii = graph.radii(sid)
            if len(coords) < 3:
                report.n_skipped += len(coords)
                continue

            # Arclength from each end, so the junction margin is a physical distance
            # rather than a point count -- the points are not evenly spaced. The tangent
            # window is measured along the same arclength, so it is computed once.
            arclen = _arclength(coords)
            tangents = plane_normals(coords, radii, arclen, tangent_radii)
            ijk = frame.um_to_seg(coords)
            new = coords.copy()

            from_start = arclen
            from_end = arclen[-1] - arclen
            # ...and only at an end that is actually a junction. A free end is one
            # vessel stopping, not two meeting: its perpendicular section is a single
            # lumen whose centroid is exactly what re-centring wants, so holding the
            # margin there would abandon every vessel tip for no reason.
            seg = graph.segment(sid)
            if graph.degree(seg["node1"]) < 3:
                from_start = np.full(len(coords), np.inf)
            if graph.degree(seg["node2"]) < 3:
                from_end = np.full(len(coords), np.inf)

            invented = _invented(graph, sid, len(coords))
            for i in range(1, len(coords) - 1):
                if invented[i]:
                    # Re-centring asks "where is the middle of the lumen at this point?".
                    # For a point Avizo invented there need not be a lumen there at all,
                    # and the centroid of whatever the cut plane happens to catch is not
                    # an answer to any question worth asking.
                    report.n_skipped += 1
                    continue
                if np.any(ijk[i] < 0) or np.any(ijk[i] >= dims):
                    report.n_skipped += 1
                    continue
                margin = junction_margin * float(radii[i])
                if from_start[i] < margin or from_end[i] < margin:
                    report.n_near_junction += 1
                    continue
                rp = max(float(radii[i]) / sp, 1.0)
                c = cut(sampler, ijk[i], tangents[i],
                        min(int(rp * 2.5) + 2, max_half), max_half=max_half,
                        grow_to=int(np.ceil(rp * grow_radii)) + 2)
                if c is None:
                    report.n_skipped += 1
                    continue
                if not c.trustworthy:
                    # Still touching the window edge at the cap: a streak, not a section.
                    report.n_truncated += 1
                    continue
                yy, xx = np.nonzero(getattr(c, blob))
                du = float(yy.mean() - c.half)
                dv = float(xx.mean() - c.half)
                new[i] = coords[i] + damping * (du * c.u + dv * c.v) * sp

            # Two clamps, answering two different questions. The spacing one goes first
            # and is measured against *this* pass's line, because that is the chain whose
            # ordering must survive; the radius one is measured against where re-centring
            # began, so repeated passes converge rather than compound.
            new, folded = _clamp_to_spacing(coords, new, max_move_spacing)
            was = coords if origin is None else origin.get(sid, coords)
            if len(was) != len(coords):  # a segment changed shape under us; be safe
                was = coords
            new, clamped = _clamp_moves(was, new, radii, max_move_frac)
            # Last, and stated in the quantity that matters: whatever the bounds allowed,
            # re-centring does not leave a point doubling back worse than it found it.
            new, reverted = _revert_new_reversals(coords, new, Roughness.TURN_DEG)
            report.n_would_fold += folded
            report.n_reverted += reverted
            dist = np.linalg.norm(new - coords, axis=1)
            if (dist > 1e-9).any():
                graph.set_segment_coords(sid, new)
                report.n_moved += int((dist > 1e-9).sum())
                report.n_clamped += clamped
                moves.extend(dist[dist > 1e-9].tolist())
            if progress is not None:
                progress(n_done, len(sids))

    if moves:
        report.median_move_um = float(np.median(moves))
        report.max_move_um = float(np.max(moves))
    return report


# -------------------------------------------------------------- the whole thing


def optimise_skeleton(
    graph,
    frame=None,
    labels=None,
    *,
    deloop: bool = True,
    prune: bool = True,
    length_factor: float = PRUNE_LENGTH_FACTOR,
    radius_ratio: float = PRUNE_RADIUS_RATIO,
    min_length_um: float | None = None,
    boundary_margin_um: float = BOUNDARY_MARGIN_UM,
    smooth_um: float | None = None,
    smoother: str = "gaussian",
    drift_radius_factor: float | None = None,
    recentre_passes: int = 2,
    max_move_frac: float = RECENTRE_MAX_MOVE_FRAC,
    tangent_radii: float = RECENTRE_TANGENT_RADII,
    damping: float = RECENTRE_DAMPING,
    grow_radii: float = RECENTRE_GROW_RADII,
    blob: str = RECENTRE_BLOB,
    max_half: int = 64,
    verbose: bool = False,
    progress=None,
) -> OptimiseReport:
    """de-loop -> prune -> recentre x N -> smooth, on an ``EditableGraph``.

    De-looping comes first because it changes which segments exist, and a spur that
    looks prunable while a false loop still holds its far end may not be one once the
    loop is broken -- and vice versa. Doing topology before geometry also means the
    smoother is never asked to smooth a segment that is about to be deleted.

    Re-centring runs more than once on purpose: it changes the line, the line
    determines the tangent, and the tangent determines the plane the next pass cuts.
    The second pass is measuring a different, better section than the first. Every pass
    is clamped against the coordinates the *first* one started from, so this is an
    iteration converging on the lumen centre and not a random walk away from the input;
    with `damping` below 1 it is the standard damped fixed-point iteration.

    **Smoothing is last, and runs once.** :mod:`~.smoothers` can dispatch to
    ``coronary_sdf``'s constrained multiscale optimiser, which *certifies* its output
    as it returns -- curvature within ``kappa*r <= 0.95``, no new branch contact, drift
    within a quarter radius. Re-centring afterwards would move points out from under
    all three guarantees without saying so, so nothing follows the smoother. The cost is
    that re-centring pass 1 works from an unsmoothed line; :func:`plane_normals` takes
    its direction over a chord a couple of vessel radii long rather than from the
    adjacent points, which is what makes that survivable.

    `frame` and `labels` are only needed for re-centring; without them this de-loops,
    prunes and smooths, which is still worth doing and needs no image.
    """
    report = OptimiseReport()

    if deloop:
        t = time.time()
        report.loops = remove_loops(graph)
        report.seconds["deloop"] = time.time() - t
        if verbose:
            print("   ", report.loops.describe())

    if prune:
        t = time.time()
        bbox = frame.seg_bbox_um if frame is not None else None
        report.prune = prune_spurs(
            graph, length_factor=length_factor, radius_ratio=radius_ratio,
            min_length_um=min_length_um, bbox_um=bbox,
            boundary_margin_um=boundary_margin_um,
        )
        report.seconds["prune"] = time.time() - t
        if verbose:
            print("   ", report.prune.describe())

    if labels is not None and frame is not None and recentre_passes > 0:
        # Snapshotted once, after the topology edits and before any point moves, so every
        # pass measures its displacement from the same place.
        started_from = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
        for k in range(int(recentre_passes)):
            t = time.time()
            m = recentre(graph, frame, labels, max_move_frac=max_move_frac,
                         tangent_radii=tangent_radii, damping=damping,
                         grow_radii=grow_radii, blob=blob, origin=started_from,
                         max_half=max_half, label=f"recentre {k + 1}",
                         progress=progress)
            report.moves.append(m)
            report.seconds[f"recentre{k + 1}"] = time.time() - t
            if verbose:
                print("   ", m.describe())

    if smoother and smoother != "none":
        from . import smoothers

        t = time.time()
        report.smooth = smoothers.smooth(
            smoother, graph, window_um=smooth_um,
            drift_radius_factor=drift_radius_factor, verbose=verbose,
        )
        report.seconds["smooth"] = time.time() - t

    report.roughness = roughness(graph)
    if verbose:
        print("   ", report.roughness.describe())

    report.seconds["total"] = sum(
        v for k, v in report.seconds.items() if k != "total"
    )
    return report
