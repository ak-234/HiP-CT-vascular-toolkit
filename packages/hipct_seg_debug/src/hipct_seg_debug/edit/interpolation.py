"""Find the points Avizo invented, and give the rest of the toolkit a way to skip them.

Where the segmentation has a hole, Avizo does not leave the spatial graph broken. It
writes points across the hole so the edge stays continuous, and the result is
indistinguishable from real centreline unless you look at it closely. It is not
skeletonisation output: nothing measured it against the image, its radius is generated
rather than observed, and -- the part that actually costs something downstream -- it makes
a tree that is genuinely in two pieces look like one, so the reconnection stages never get
asked to rebuild the join from the greyscale.

Three independent signatures, each carrying its own reason code so a report can say *why*
a point was flagged rather than only that it was. Every number below was measured on
``ASCII_smooth_thick_adj_LADAF_2024_28.Spatial-Graph.attributegraph.am`` (311 vertices,
309 edges, 29,122 points) and on the LADAF-2024-56 graph beside it.

**STRAIGHT_BRIDGE.** A run of points with no curvature at all, whose radius is an exact
linear ramp, meeting the real vessel with a large radius step at *both* ends. Only the
last clause makes this specific rather than merely descriptive -- straight vessel is
common, and so is straight vessel with a linear taper:

=================================================  ========  ========
gate                                               LADAF-28  LADAF-56
=================================================  ========  ========
collinear run of >= 3 points                            941      1033
...and the radius is an exact linear ramp               941       673
...and the radius steps >= 5% at both anchors             2         1
...and that step dwarfs the run's own taper               2         1
=================================================  ========  ========

The survivors are edges 183 and 197 on LADAF-28 -- 26 points, 0.09% of the graph -- and
edge 216 on LADAF-56. Edge 183 is 13 points whose chord equals their arclength to seven
digits, 11 of them invented: the interior radius steps by a constant 0.2418 um, and the
two points that anchor it to the real vessel jump 24% and 32%. The separation is not
marginal. Those two ends move 0.24 and 0.19 against an interior that moves 0.001 per step,
a ratio near 250, while the *next* best run in either graph does not clear 5% at both
ends at all. Note also that the ramp gate on its own removes nothing from LADAF-28: the
work is done by the anchors, which is the point.

**FLOOR_RADIUS.** ``adjust_thickness.py`` calibrates Avizo's thickness with a global
linear fit, ``r = slope * t + intercept``, floored at ``voxel / 2``. A point where Avizo
wrote no thickness therefore lands exactly on the intercept, and shows up as a spike at
the very bottom of the radius distribution *detached from the continuum*: LADAF-28 puts
10 points at 81.269 um with the next distinct radius at 124.63 (1.53x), LADAF-56 puts 8
at 78.424 with the next at 120.75 (1.54x). On a raw pre-calibration export this is simply
a radius of zero.

**OFF_MASK**, optional, needs the lattice. A real centreline is inside the vessel it
describes. Sampling Labels at all 29,122 points of the LADAF-28 graph, 29,110 -- 99.96% --
land on foreground, so being outside is rare enough to mean something. It is scored over
*runs* rather than points, because all twelve of the LADAF-28 exceptions are isolated: a
smoothed centreline clipping the outside of a bend, which is an ordinary thing for a
smoother to do. A fill across a hole is several points long by construction. With the run
gate applied, LADAF-28 yields no OFF_MASK flags at all.

Worth stating plainly, because it contradicts the obvious story: on LADAF-28 the two
straight bridges lie **entirely inside** the segmentation, and none of the off-mask points
belong to them. So these signatures are independent evidence, and neither implies the
other. Whatever hole edge 183 was drawn across is not a hole in the mask this graph is
paired with today -- the mask may have been repaired since, or the fill made against an
earlier one. The run is still not skeletonisation output, which is what the flag records.

A note on what a flag does and does not license. A *span* -- several consecutive points --
is an invented piece of vessel, and that is what gets split for reconnection and drawn as
a bridge in the viewer. A *lone* flagged point, which is what FLOOR_RADIUS almost always
produces, is an invented radius on a real point: keep it out of the measurements, but do
not tear the graph apart over it. :func:`split_flagged` enforces that distinction.

Two ways to consume the result, and the difference matters:

* :func:`mask_for_segment` for anything that *measures* -- radius, scoring, smoothing. The
  graph keeps its shape; those stages simply do not read the invented points.
* :func:`split_flagged` for anything that needs the *topology* to be honest --
  reconnection, and the SDF surface. A hole left inside an edge is not enough there,
  because ``coronary_sdf.centreline_reconnection.bridge_centerline_gaps`` runs inside
  ``sdfpatch.preprocess_graph`` and would immediately fill it back in. Splitting removes
  the hole and the edge together.

Detection is never implicit. :func:`flags` reads the stored field and returns "nothing
flagged" when it is absent; only the ``flag-interpolation`` command and the 3D viewer's
own display layer call :func:`detect`. A perfectly straight, linearly interpolated segment
is what half the synthetic fixtures in ``tests/conftest_geometry.py`` are made of, and a
detector that ran itself inside every pipeline would quietly change what those tests mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

#: Per-point field name, written beside the radius in the ``.am``.
FIELD = "avizo_interpolated"
#: Where `UNSAMPLED_JUMP` is recorded. A **separate** field, not another bit in `FIELD`,
#: because the two make different claims and `FIELD` has one consumer-visible meaning:
#: *this point is invented, keep it out of every measurement*. A jump's two anchors are
#: ordinary measured centreline -- it is the empty step between them that is fabricated --
#: so folding it into `FIELD` would silently drop two real points from every radius and
#: length statistic in the toolkit. Value 1 on point i means "the step from i to i+1 is
#: an unsampled connector".
JUMP_FIELD = "unsampled_jump"

# Reason codes, combined as a bitmask. Same idiom as `radius_perimeter`'s SOURCE_NAMES /
# REJECT_NAMES / RESOLUTION_NAMES: an int per point plus a table to render it.
REAL = 0
STRAIGHT_BRIDGE = 1
FLOOR_RADIUS = 2
OFF_MASK = 4
UNSAMPLED_JUMP = 8

REASON_NAMES = {
    STRAIGHT_BRIDGE: "straight bridge (zero curvature, linear radius ramp)",
    FLOOR_RADIUS: "degenerate radius (at the graph's floor)",
    OFF_MASK: "outside the segmentation",
    UNSAMPLED_JUMP: "unsampled jump across a mask break",
}

#: The three signatures that mark a *point*. `UNSAMPLED_JUMP` is deliberately absent:
#: it marks a *step*, and the two points either side of it are real measured centreline
#: that must stay in every measurement.
POINT_REASONS = (STRAIGHT_BRIDGE, FLOOR_RADIUS, OFF_MASK)

# --- STRAIGHT_BRIDGE thresholds -------------------------------------------------------
#: Two consecutive unit tangents count as one direction above this dot product.
#: 1 - 1e-6 is a turn of 0.081 degrees; the real spans dot to 1.0 in float64.
COLLINEAR_COS = 1.0 - 1e-6
#: Radius curvature, as ``|r[i-1] - 2 r[i] + r[i+1]| / r[i]``, below which a run counts as
#: an exact linear ramp. The artefact's is zero to rounding; a smoothed real centreline
#: on this data sits near 1e-3.
RAMP_TOL = 1e-4
#: Relative radius step at each end of a run, below which nothing is called a bridge
#: however sharp the run's own ramp is. An absolute floor, so a near-constant run cannot
#: qualify on a proportionally-large but physically tiny step.
ANCHOR_JUMP = 0.05
#: ...and the step must also be this many times the run's own per-step radius change.
#: The absolute gate alone is not enough: a vessel tapering steeply enough (5%+ per point,
#: which a thin one does) clears it at both ends purely by continuing to taper. What marks
#: a generated run is that its ends do not *match* its middle -- on the real artefact the
#: interior moves 0.1% per step against a 24% anchor jump, a ratio around 250.
ANCHOR_JUMP_RATIO = 10.0
#: Shortest run worth calling a bridge, in points.
MIN_SPAN_POINTS = 3

# --- FLOOR_RADIUS thresholds ----------------------------------------------------------
#: The bottom cluster of the radius distribution must be at least this much smaller than
#: the next distinct value to count as a detached floor rather than the low tail of a
#: continuum. Measured ratio on both graphs is ~1.53.
FLOOR_GAP_RATIO = 1.25
#: Values within this relative distance of each other are one cluster; the floor is
#: written by a float fit, so it lands on a couple of neighbouring representations.
FLOOR_CLUSTER_TOL = 1e-3

# --- UNSAMPLED_JUMP thresholds --------------------------------------------------------
#: Absolute floor on what counts as a jump. Measured on LADAF-28: consecutive centreline
#: points sit 93 um apart (median; p99 = 114), and of the 38 steps above 500 um, *every
#: one* above 600 um crosses into a different mask component -- 24 of them -- while the
#: eleven that do not are all 510-590 um and confined to three coarsely-sampled segments.
#: The separation is clean, so the threshold sits in the empty band.
JUMP_MIN_UM = 600.0
#: ...and the step must also be this many times the segment's *own* median spacing, so a
#: genuinely coarse trunk is judged against itself rather than against a global number.
JUMP_STEP_RATIO = 5.0


@dataclass
class Span:
    """One run of invented points inside a single segment.

    ``n_points == 0`` is the :data:`UNSAMPLED_JUMP` case and means the invented thing is
    the **edge** between ``ids[start - 1]`` and ``ids[start]`` rather than any point.
    Avizo writes those where it joined two traced runs without sampling between them, so
    there is literally nothing to flag -- which is exactly why the three point-based
    signatures cannot see them. The index convention is chosen so that the removal in
    :func:`_split_one` needs no special case: ``ids[start - 1:start + 1]`` is the pair
    either side, the same slice a one-point span would anchor against.
    """

    seg_id: int
    start: int  # index into the segment's point list
    n_points: int
    point_ids: list[int]
    reason: int
    length_um: float = 0.0
    radius_um: float = 0.0
    anchor_jump: float = 0.0  # smaller of the two end discontinuities
    off_mask_fraction: float | None = None
    #: For a jump only: the point ids either side of the step. Segment ids do not
    #: survive a split and point ids do, so this is what lets a *second* jump in the
    #: same segment still be found after the first one has cut it in half.
    anchors: tuple[int, int] | None = None

    @property
    def stop(self) -> int:
        """One past the last flagged index, so ``ids[start:stop]`` is the span."""
        return self.start + self.n_points

    @property
    def is_jump(self) -> bool:
        """No points of its own: the fabricated part is the step, not a run."""
        return self.n_points == 0

    def describe(self) -> str:
        bits = [n for c, n in REASON_NAMES.items() if self.reason & c]
        if self.is_jump:
            return (
                f"segment {self.seg_id:>5} step {self.start - 1}->{self.start}: "
                f"{self.length_um / 1000.0:6.2f} mm across no points, "
                f"radius {self.radius_um:7.1f} um -- {', '.join(bits)}"
            )
        out = (
            f"segment {self.seg_id:>5} points {self.start}-{self.stop - 1} "
            f"({self.n_points:>3}): {self.length_um / 1000.0:6.2f} mm, "
            f"radius {self.radius_um:7.1f} um"
        )
        if self.anchor_jump:
            out += f", anchor jump {100 * self.anchor_jump:.0f}%"
        if self.off_mask_fraction is not None:
            out += f", {100 * self.off_mask_fraction:.0f}% outside the mask"
        return out + f" -- {', '.join(bits)}"


@dataclass
class Detection:
    """Every flagged point, and the spans they group into."""

    flags: dict[int, int] = field(default_factory=dict)  # {point id: reason bitmask}
    spans: list[Span] = field(default_factory=list)
    n_points: int = 0
    floor_um: float | None = None
    checked_mask: bool = False
    checked_jumps: bool = False

    @property
    def n_flagged(self) -> int:
        return sum(1 for v in self.flags.values() if v)

    @property
    def jumps(self) -> list["Span"]:
        return [s for s in self.spans if s.is_jump]

    def counts(self) -> dict[int, int]:
        """Flagged *points* per signature. Jumps have none by construction."""
        return {
            code: sum(1 for v in self.flags.values() if v & code)
            for code in POINT_REASONS
        }

    def describe(self, *, show_spans: bool = False, limit: int = 20) -> str:
        if not self.n_flagged and not self.jumps:
            detail = ""
            if self.floor_um is None:
                detail = " (no detached radius floor; nothing straight enough either)"
            if not self.checked_jumps:
                detail += " (pass --seg to also look for unsampled jumps)"
            return f"no interpolated points found in {self.n_points:,} points{detail}"

        if not self.n_flagged:
            lines = [f"no interpolated points in {self.n_points:,} points"]
        else:
            lines = []

        runs = [s for s in self.spans if not s.is_jump]
        if self.n_flagged:
            lines.append(
                f"{self.n_flagged:,} interpolated point(s) of {self.n_points:,} "
                f"({100.0 * self.n_flagged / max(self.n_points, 1):.2f}%) "
                f"in {len(runs)} span(s)"
            )
        for code, n in self.counts().items():
            if n:
                lines.append(f"  {n:>6}  {REASON_NAMES[code]}")
        if self.jumps:
            total = sum(s.length_um for s in self.jumps)
            lines.append(
                f"{len(self.jumps)} unsampled jump(s) across a mask break, "
                f"{total / 1000.0:.1f} mm in total"
            )
            lines.append("  these carry no points at all, which is why the three "
                         "point signatures cannot see them")
        if self.floor_um is not None:
            lines.append(f"  radius floor detected at {self.floor_um:.3f} um")
        if not self.checked_mask:
            lines.append("  (pass --seg to also check these against the segmentation)")
        if show_spans:
            ordered = sorted(self.spans, key=lambda s: (-s.n_points, -s.length_um))
            for span in ordered[:limit]:
                lines.append("    " + span.describe())
            if len(ordered) > limit:
                lines.append(f"    ... and {len(ordered) - limit} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# reading what is already stored
# --------------------------------------------------------------------------- #
def _triple(obj) -> Any:
    """Accept a ``Triple`` or an ``EditableGraph`` and return the ``Triple``."""
    return getattr(obj, "triple", obj)


def flags(obj) -> dict[int, int]:
    """``{point id: reason bitmask}`` from the stored field, or empty when absent.

    Deliberately does not fall back to :func:`detect`. See the module docstring.
    """
    triple = _triple(obj)
    stored = triple.point_attrs.get(FIELD)
    if not stored:
        return {}
    # Filtered to points that still exist. Deleting a point does not reach into the
    # per-point attribute dicts -- nothing in `graphmodel` does -- so after a split the
    # store still holds entries for the points that were removed, and a caller asking
    # "what is flagged?" would be handed ids that are no longer in the graph.
    live = triple.points
    return {pid: int(v) for pid, v in stored.items() if int(v) and pid in live}


def jump_flags(obj) -> dict[int, int]:
    """``{point id: 1}`` for points whose *following* step is an unsampled connector."""
    raw = _triple(obj).point_attrs.get(JUMP_FIELD) or {}
    return {int(k): int(v) for k, v in raw.items() if int(v)}


def has_flags(obj) -> bool:
    """Whether this graph has been through ``flag-interpolation``.

    True if *either* field is present: a graph whose only artefact was an unsampled jump
    has been checked just as thoroughly as one with a straight bridge in it, and
    reporting it as unchecked would send `connect` down the "run flag-interpolation
    first" path it has already been down.
    """
    attrs = _triple(obj).point_attrs
    return FIELD in attrs or JUMP_FIELD in attrs


def flagged_points(obj) -> set[int]:
    return set(flags(obj))


def mask_for_segment(obj, sid: int) -> np.ndarray:
    """(N,) bool over one segment's points -- True where the point was invented."""
    triple = _triple(obj)
    marked = flags(triple)
    seg = next((s for s in triple.segments if s["id"] == sid), None)
    if seg is None:
        return np.zeros(0, dtype=bool)
    if not marked:
        return np.zeros(len(seg["point_ids"]), dtype=bool)
    return np.array([pid in marked for pid in seg["point_ids"]], dtype=bool)


def mask(obj) -> np.ndarray:
    """(P,) bool in ``point_order()`` -- the flat concatenation over segments."""
    triple = _triple(obj)
    marked = flags(triple)
    order = [pid for seg in triple.segments for pid in seg["point_ids"]]
    if not marked:
        return np.zeros(len(order), dtype=bool)
    return np.array([pid in marked for pid in order], dtype=bool)


def flags_array(graph, *, detect_if_absent: bool = False) -> np.ndarray:
    """(P,) reason codes for an ``amira.SpatialGraph``, in its own point order.

    The array form the 3D viewer wants, since it indexes points positionally and its
    pick ids *are* those indices. ``detect_if_absent`` is for the viewer alone: it opens
    whatever file it is handed, the layer it feeds is display-only, and being shown a
    grey bridge that turns out not to be recorded in the file is a smaller surprise than
    not being shown it at all. No pipeline passes it.
    """
    n = int(graph.n_point)
    stored = (getattr(graph, "point_attrs", None) or {}).get(FIELD)
    if stored is not None:
        arr = np.asarray(stored).ravel()
        if arr.size == n:
            return arr.astype(np.int64)
    if not detect_if_absent:
        return np.zeros(n, dtype=np.int64)

    from .adapter import from_spatial_graph

    # `from_spatial_graph` numbers points by array index, so the ids come back as the
    # positions this array is indexed by.
    found = detect(from_spatial_graph(graph))
    out = np.zeros(n, dtype=np.int64)
    for pid, value in found.flags.items():
        if 0 <= pid < n:
            out[pid] = value
    return out


def jump_breaks_array(graph) -> np.ndarray:
    """(P,) bool in an ``amira.SpatialGraph``'s own point order.

    True at point *i* when the step from *i* to *i + 1* is an unsampled jump. The
    array form the 3D viewer wants, and the counterpart of :func:`flags_array` --
    which cannot express this, because it marks points and a jump is a *step*.

    Deliberately never detects on demand. Unlike the three point signatures, this one
    needs the mask labelled, and a viewer that silently spent thirteen seconds decoding
    2.34 GB to colour a line would be a surprising thing for opening a file to do.
    """
    n = int(graph.n_point)
    stored = (getattr(graph, "point_attrs", None) or {}).get(JUMP_FIELD)
    out = np.zeros(n, dtype=bool)
    if stored is None:
        return out
    arr = np.asarray(stored).ravel()
    if arr.size == n:
        out[:] = arr > 0
    return out


def runs_in_edges(flagged: np.ndarray, edge_offsets, *, pad: int = 0) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` runs of flagged points, never crossing an edge boundary.

    ``pad`` extends each run by that many points on each side without leaving its edge --
    one is enough to pick up the real anchors, so a drawn bridge meets the vessel it was
    invented to join instead of floating a point short at each end.
    """
    out: list[tuple[int, int]] = []
    offsets = np.asarray(edge_offsets, dtype=np.int64)
    for a, b in zip(offsets[:-1], offsets[1:]):
        a, b = int(a), int(b)
        for start, stop in _runs(list(flagged[a:b].astype(bool)), 1):
            out.append((max(a + start - pad, a), min(a + stop + pad, b)))
    return out


def spans(obj) -> list[Span]:
    """Rebuild the span records from the stored per-point fields.

    Both of them: runs of invented points out of `FIELD`, and zero-point jump spans out
    of `JUMP_FIELD`. `split_flagged` works from this, so a jump that is stored but not
    rebuilt here would be silently un-cuttable.
    """
    triple = _triple(obj)
    marked = flags(triple)
    jumped = jump_flags(triple)
    if not marked and not jumped:
        return []
    points = triple.points
    out: list[Span] = []
    for seg in triple.segments:
        ids = seg["point_ids"]
        i = 0
        while i < len(ids):
            if ids[i] in marked:
                j = i
                while j < len(ids) and ids[j] in marked:
                    j += 1
                run = ids[i:j]
                reason = 0
                for pid in run:
                    reason |= marked[pid]
                out.append(
                    Span(
                        seg_id=seg["id"], start=i, n_points=j - i, point_ids=list(run),
                        reason=reason,
                        length_um=_polyline_length([points[p][:3] for p in run]),
                        radius_um=float(np.median([points[p][3] for p in run])),
                    )
                )
                i = j
            else:
                i += 1

        for i, pid in enumerate(ids[:-1]):
            if pid not in jumped:
                continue
            pair = [points[ids[i]][:3], points[ids[i + 1]][:3]]
            out.append(
                Span(
                    seg_id=seg["id"], start=i + 1, n_points=0, point_ids=[],
                    reason=UNSAMPLED_JUMP,
                    length_um=_polyline_length(pair),
                    radius_um=float(np.median(
                        [points[ids[i]][3], points[ids[i + 1]][3]]
                    )),
                    anchors=(int(ids[i]), int(ids[i + 1])),
                )
            )
    return out


def _polyline_length(coords) -> float:
    arr = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if len(arr) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def detect(
    obj,
    *,
    labels=None,
    frame=None,
    collinear_cos: float = COLLINEAR_COS,
    ramp_tol: float = RAMP_TOL,
    anchor_jump: float = ANCHOR_JUMP,
    anchor_jump_ratio: float = ANCHOR_JUMP_RATIO,
    min_span_points: int = MIN_SPAN_POINTS,
    floor_gap_ratio: float = FLOOR_GAP_RATIO,
    components=None,
    min_step_um: float = JUMP_MIN_UM,
    step_ratio: float = JUMP_STEP_RATIO,
) -> Detection:
    """Run every signature over a graph.

    ``labels``/``frame`` enable the OFF_MASK check; ``components``/``frame`` enable
    :func:`detect_jumps`, which is the only signature that needs the segmentation
    labelled rather than merely sampled.
    """
    triple = _triple(obj)
    points = triple.points
    order = [pid for seg in triple.segments for pid in seg["point_ids"]]
    result = Detection(n_points=len(order))
    marks: dict[int, int] = {}

    # -- B: a radius sitting on the graph's degenerate floor --------------------
    all_radii = np.array([points[p][3] for p in order], dtype=np.float64)
    floor = _radius_floor(all_radii, floor_gap_ratio)
    result.floor_um = floor
    for pid in order:
        r = float(points[pid][3])
        if not np.isfinite(r) or r <= 0.0:
            marks[pid] = marks.get(pid, 0) | FLOOR_RADIUS
        elif floor is not None and r <= floor * (1.0 + 1e-9):
            marks[pid] = marks.get(pid, 0) | FLOOR_RADIUS

    # -- A: a straight, linearly-tapered run bounded by two radius jumps --------
    for seg in triple.segments:
        ids = seg["point_ids"]
        if len(ids) < min_span_points + 2:
            continue
        coords = np.array([points[p][:3] for p in ids], dtype=np.float64)
        radii = np.array([points[p][3] for p in ids], dtype=np.float64)
        for start, stop in _straight_bridges(
            coords, radii, collinear_cos, ramp_tol, anchor_jump,
            anchor_jump_ratio, min_span_points
        ):
            for pid in ids[start:stop]:
                marks[pid] = marks.get(pid, 0) | STRAIGHT_BRIDGE

    # -- C: a *run* outside the segmentation, when we were given one ------------
    # Runs, not points. Measured on LADAF-28, twelve points fall outside the mask and
    # every one of them is on its own -- a smoothed centreline clipping the wall of a
    # curve, which is an ordinary thing for a smoother to do and not an invented vessel.
    # A bridge across a hole is by construction several points long.
    if labels is not None and frame is not None:
        result.checked_mask = True
        inside = {}
        for seg in triple.segments:
            ids = seg["point_ids"]
            xyz = np.array([points[p][:3] for p in ids], dtype=np.float64)
            for pid, hit in zip(ids, sample_mask(xyz, frame, labels)):
                inside[pid] = bool(hit)
            for start, stop in _runs(
                [not inside[pid] for pid in ids], min_span_points
            ):
                for pid in ids[start:stop]:
                    marks[pid] = marks.get(pid, 0) | OFF_MASK

    result.flags = {pid: v for pid, v in marks.items() if v}
    result.spans = _group_spans(triple, result.flags, labels, frame)

    # -- D: a step that leaps across a break in the mask ------------------------
    if components is not None and frame is not None:
        result.checked_jumps = True
        result.spans.extend(detect_jumps(
            triple, components, frame,
            min_step_um=min_step_um, step_ratio=step_ratio,
        ))
    return result


def candidate_jumps(
    obj,
    *,
    min_step_um: float = JUMP_MIN_UM,
    step_ratio: float = JUMP_STEP_RATIO,
) -> list[Span]:
    """Steps that look like an unsampled jump, judged from the graph alone.

    :func:`detect_jumps` has two gates and only the second -- "the two ends land in
    different mask components" -- needs the segmentation. The first is pure geometry:
    a step far larger than the segment's own median spacing. Separating them matters
    because a caller without the mask would otherwise learn *nothing* about the one
    artefact the three point-based signatures cannot see, and would be told the graph
    is clean. On LADAF-28 that reported "3 points in 1 span" for a graph carrying 24
    invented bridges and 72.4 mm of centreline through empty space.

    A candidate is not a finding. A long step that stays inside one component is
    under-sampling, not a break, and only :func:`detect_jumps` can tell the two apart.
    """
    triple = _triple(obj)
    points = triple.points
    out: list[Span] = []

    for seg in triple.segments:
        ids = seg["point_ids"]
        if len(ids) < 2:
            continue
        coords = np.array([points[p][:3] for p in ids], dtype=np.float64)
        radii = np.array([points[p][3] for p in ids], dtype=np.float64)
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        if not len(steps):
            continue
        threshold = max(float(min_step_um), step_ratio * float(np.median(steps)))
        for i in np.flatnonzero(steps > threshold):
            i = int(i)
            out.append(
                Span(
                    seg_id=seg["id"], start=i + 1, n_points=0, point_ids=[],
                    reason=UNSAMPLED_JUMP,
                    length_um=float(steps[i]),
                    radius_um=float(np.median(radii[i:i + 2])),
                    anchors=(int(ids[i]), int(ids[i + 1])),
                )
            )
    return out


def detect_jumps(
    obj,
    components,
    frame,
    *,
    min_step_um: float = JUMP_MIN_UM,
    step_ratio: float = JUMP_STEP_RATIO,
) -> list[Span]:
    """Steps inside one edge that leap across a genuine break in the segmentation.

    The fourth signature, and the only one that cannot be seen from the graph alone.
    Avizo joins two traced runs by writing a single enormous step with **no points
    between them**, so there is nothing for :func:`_straight_bridges` to find a run in
    and nothing for the radius floor to land on -- the three point-based signatures are
    blind to it by construction, and on LADAF-28 they miss all 27 of these.

    Two gates, and the second is what makes it a finding rather than a heuristic:

    * the step is an outlier **against its own segment** -- `step_ratio` times that
      segment's median spacing, and at least `min_step_um`. Judging each segment against
      itself is what stops a legitimately coarse trunk being torn apart;
    * the two ends land in **different mask components**. That is a direct measurement,
      not a proxy: it says the segmentation does not connect what the graph claims is
      one vessel. A long step that stays inside one component is under-sampling, which
      is not a break and is left alone.

    `components` is a :class:`~.reconnect.geodesic.components.ComponentIndex` -- only
    ``nearest_label`` is used, so any equivalent lookup works.
    """
    triple = _triple(obj)
    points = triple.points
    out: list[Span] = []

    # One definition of the geometric gate, shared with `candidate_jumps`, so the
    # screen a caller sees without the mask cannot drift from the one applied with it.
    for span in candidate_jumps(triple, min_step_um=min_step_um,
                                step_ratio=step_ratio):
        lo, hi = span.anchors
        a = _component_at(components, frame, np.asarray(points[lo][:3], dtype=float))
        b = _component_at(components, frame, np.asarray(points[hi][:3], dtype=float))
        if not a or not b or a == b:
            continue
        out.append(span)
    return out


def _component_at(components, frame, point_um, reach: int = 3) -> int:
    """Mask component containing -- or within `reach` voxels of -- a world point.

    The tolerance matters: a centreline vertex is derived on a different grid from the
    mask it came from, so landing a voxel outside its own lumen is routine and must not
    read as "different component from its neighbour", which would manufacture a break at
    every bend.
    """
    ijk = np.asarray(frame.um_to_seg(np.asarray(point_um, dtype=np.float64)[None, :]),
                     dtype=np.float64)[0]
    z, y, x = (int(v) for v in np.round(ijk[::-1]))
    label, _distance = components.nearest_label(z, y, x, reach)
    return int(label)


def _radius_floor(radii: np.ndarray, gap_ratio: float) -> float | None:
    """The top of a detached cluster at the bottom of the radius distribution.

    ``None`` when the smallest radii shade continuously into the rest, which is what an
    honest measured graph looks like. A calibration intercept, by contrast, is a spike
    with visible empty space above it.
    """
    valid = radii[np.isfinite(radii) & (radii > 0.0)]
    if valid.size < 3:
        return None
    unique = np.unique(valid)
    if unique.size < 3:
        return None
    # Grow the bottom cluster over values that are the same number to within rounding.
    k = 1
    while k < unique.size and unique[k] <= unique[0] * (1.0 + FLOOR_CLUSTER_TOL):
        k += 1
    if k >= unique.size:
        return None
    if unique[k] < unique[k - 1] * gap_ratio:
        return None
    return float(unique[k - 1])


def _straight_bridges(
    coords: np.ndarray,
    radii: np.ndarray,
    collinear_cos: float,
    ramp_tol: float,
    anchor_jump: float,
    anchor_jump_ratio: float,
    min_span_points: int,
) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` point index ranges of the invented runs in one segment.

    Both tests are indexed by *interior point*: ``collinear[p - 1]`` is the turn at point
    ``p`` and ``flat[p - 1]`` its radius curvature, for ``p`` in ``1 .. n - 2``.

    The one non-obvious step is the extension at the end. A fill's own first and last
    points fail the flatness test -- their radius curvature is enormous, because that is
    exactly where the invented ramp meets the real vessel -- so the flat run lands one
    point short at each end. Extending across a point that is still perfectly straight
    recovers them, and leaves the anchors, which are real, outside the span. Getting this
    wrong is not cosmetic: it measures the anchor jump between two invented points
    instead of across the discontinuity, and the span is then thrown away.
    """
    n = len(coords)
    if n < min_span_points + 2:
        return []

    step = np.diff(coords, axis=0)
    length = np.linalg.norm(step, axis=1)
    if not np.all(np.isfinite(length)) or np.any(length <= 1e-12):
        return []
    direction = step / length[:, None]
    collinear = np.einsum("ij,ij->i", direction[:-1], direction[1:]) >= collinear_cos

    with np.errstate(invalid="ignore", divide="ignore"):
        curvature = np.abs(radii[:-2] - 2.0 * radii[1:-1] + radii[2:]) / np.maximum(
            np.abs(radii[1:-1]), 1e-12
        )
    flat = (curvature <= ramp_tol) & np.isfinite(curvature)
    smooth = collinear & flat

    out: list[tuple[int, int]] = []
    i = 0
    while i < len(smooth):
        if not smooth[i]:
            i += 1
            continue
        j = i
        while j < len(smooth) and smooth[j]:
            j += 1
        # smooth[i:j] -> point indices a .. b inclusive.
        a, b = i + 1, j
        i = j

        # Reclaim the fill's own end points: still straight, but their radius curvature
        # breaks because the ramp meets the real vessel there.
        if a - 1 >= 1 and collinear[a - 2]:
            a -= 1
        if b + 1 <= n - 2 and collinear[b]:
            b += 1

        start, stop = a, b + 1
        if stop - start < min_span_points:
            continue
        pre = _relative_step(radii, start - 1)
        post = _relative_step(radii, stop - 1)
        if pre is None or post is None:
            continue
        if min(pre, post) < anchor_jump:
            continue
        # ...and both ends must disagree with the run's own taper, not merely continue it.
        interior = [
            _relative_step(radii, k) for k in range(start, stop - 1)
        ]
        interior = [v for v in interior if v is not None]
        own = float(np.median(interior)) if interior else 0.0
        if min(pre, post) < anchor_jump_ratio * own:
            continue
        out.append((start, stop))
    return out


def _runs(hits, min_length: int) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` maximal runs of True at least `min_length` long."""
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(hits):
        if not hits[i]:
            i += 1
            continue
        j = i
        while j < len(hits) and hits[j]:
            j += 1
        if j - i >= min_length:
            out.append((i, j))
        i = j
    return out


def _relative_step(radii: np.ndarray, i: int) -> float | None:
    """``|r[i+1] - r[i]| / r[i]`` for the step leaving point ``i``, or None off the end."""
    if i < 0 or i + 1 >= len(radii):
        return None
    base = abs(float(radii[i]))
    if base <= 1e-12:
        return None
    return abs(float(radii[i + 1]) - float(radii[i])) / base


def _group_spans(triple, marked: dict[int, int], labels, frame) -> list[Span]:
    """Consecutive flagged points inside one segment, with their evidence attached."""
    if not marked:
        return []
    points = triple.points
    out: list[Span] = []
    for seg in triple.segments:
        ids = seg["point_ids"]
        radii = None
        i = 0
        while i < len(ids):
            if ids[i] not in marked:
                i += 1
                continue
            j = i
            while j < len(ids) and ids[j] in marked:
                j += 1
            run = ids[i:j]
            reason = 0
            for pid in run:
                reason |= marked[pid]
            if radii is None:
                radii = np.array([points[p][3] for p in ids], dtype=np.float64)
            pre = _relative_step(radii, i - 1)
            post = _relative_step(radii, j - 1)
            jump = min(pre, post) if (pre is not None and post is not None) else 0.0
            frac = None
            if labels is not None and frame is not None:
                inside = sample_mask(
                    np.array([points[p][:3] for p in run], dtype=np.float64), frame, labels
                )
                frac = float(1.0 - inside.mean()) if len(inside) else None
            out.append(
                Span(
                    seg_id=seg["id"], start=i, n_points=j - i, point_ids=list(run),
                    reason=reason,
                    length_um=_polyline_length([points[p][:3] for p in run]),
                    radius_um=float(np.median([points[p][3] for p in run])),
                    anchor_jump=float(jump),
                    off_mask_fraction=frac,
                )
            )
            i = j
    return out


def sample_mask(xyz: np.ndarray, frame, labels) -> np.ndarray:
    """(N,) bool -- whether each world-um point lands on segmentation foreground.

    Decoded one z-plane at a time in sorted order, because ``ByteRLELattice`` decodes a
    whole slice per call and a graph's points arrive in edge order, not depth order.
    Points outside the lattice count as outside the mask.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), dtype=bool)
    if not len(pts):
        return out

    idx = np.asarray(frame.um_to_seg_index(pts)).astype(np.int64).reshape(-1, 3)
    nx, ny, nz = (int(v) for v in frame.seg_dims)
    ok = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx)
        & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
        & (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )
    plane = None
    current = -1
    for p in np.argsort(idx[:, 2], kind="stable"):
        if not ok[p]:
            continue
        z = int(idx[p, 2])
        if z != current:
            plane = labels.slice_z(z)
            current = z
        out[p] = bool(plane[int(idx[p, 1]), int(idx[p, 0])])
    return out


# --------------------------------------------------------------------------- #
# writing the field
# --------------------------------------------------------------------------- #
def annotate(obj, detection: Detection) -> None:
    """Attach the reason codes to the graph as a per-point field.

    Every point gets an entry, not only the flagged ones: a field that covered a subset
    would be filled with the writer's default for the rest, which happens to be the right
    answer here but only by luck. Being explicit means a later reader cannot tell the
    difference between "checked, clean" and "never checked" only from the point count --
    the presence of the field itself is that signal.
    """
    triple = _triple(obj)
    order = [pid for seg in triple.segments for pid in seg["point_ids"]]
    triple.point_attrs[FIELD] = {pid: int(detection.flags.get(pid, REAL)) for pid in order}
    triple.point_attr_dtypes[FIELD] = np.dtype(np.int64)

    if not detection.checked_jumps:
        return
    # Written on the point *before* the step, so "the gap after me is fabricated". Every
    # point gets an entry for the same reason `FIELD` does: the presence of the field is
    # what distinguishes "checked, clean" from "never checked".
    by_segment = {seg["id"]: seg["point_ids"] for seg in triple.segments}
    marked: set[int] = set()
    for span in detection.jumps:
        ids = by_segment.get(span.seg_id)
        if ids and 0 < span.start <= len(ids) - 1:
            marked.add(ids[span.start - 1])
    triple.point_attrs[JUMP_FIELD] = {pid: int(pid in marked) for pid in order}
    triple.point_attr_dtypes[JUMP_FIELD] = np.dtype(np.int64)


def set_flags(obj, point_ids: Iterable[int], reason: int) -> None:
    """Mark specific points, creating the field if it is not there yet.

    Used when points are recreated -- ``restore_unbridged`` puts a fill back through
    ``add_segment``, which mints new point ids that no earlier detection can know about.
    """
    triple = _triple(obj)
    store = triple.point_attrs.setdefault(FIELD, {})
    triple.point_attr_dtypes.setdefault(FIELD, np.dtype(np.int64))
    for pid in point_ids:
        store[pid] = int(reason)


# --------------------------------------------------------------------------- #
# the virtual split
# --------------------------------------------------------------------------- #
@dataclass
class SplitRecord:
    """One removed fill, and everything needed to put it back exactly."""

    seg_id: int  # the segment it came out of, before the split
    node1: int  # free end on the upstream side
    node2: int  # free end on the downstream side
    coords: np.ndarray  # (N, 3) um, node1 -> node2, including both anchor points
    radii: np.ndarray  # (N,) um
    reason: int
    attrs: dict[str, Any] = field(default_factory=dict)
    whole_edge: bool = False  # the fill *was* the entire edge; no free ends were made

    @property
    def length_um(self) -> float:
        return _polyline_length(self.coords)


def split_flagged(graph, *, label: str = "split Avizo interpolation") -> list[SplitRecord]:
    """Remove every flagged span and split its edge, so the break becomes real.

    What the reconnection stages need that a mask cannot give them: after this,
    ``graph.endpoints()`` reports the anchors of each removed fill as genuine free ends,
    and ``graph.components()`` reports the pieces the tree was actually in. Everything
    downstream -- the cone gates, the Type 1/2/3 staging, the DPC walk -- then works
    unchanged and on the truth.

    One case does not produce free ends. When the fill is an entire edge between two
    junctions, deleting it leaves those junctions at degree two, not one, because their
    other vessels are real. The record is still kept (so the fill can be restored) and
    ``whole_edge`` says so, but no endpoint-to-endpoint proposal can pick it up -- that
    join is vessel-to-vessel, which no proposer here handles.

    Only spans of at least :data:`MIN_SPAN_POINTS` are split. A lone point at the radius
    floor is an invented *radius*, not an invented vessel -- it is right to keep it out of
    every measurement, and wrong to tear the graph in half over it.

    Runs as one batch, so it is a single undo step.
    """
    records: list[SplitRecord] = []
    # A jump has no points, so the minimum-length rule -- which exists to stop a single
    # invented *radius* tearing the graph in half -- must not be applied to it.
    todo = [s for s in spans(graph)
            if s.is_jump or s.n_points >= MIN_SPAN_POINTS]
    if not todo:
        return records

    with graph.batch(label):
        # Deepest index first inside each segment, so an earlier split cannot invalidate
        # the index a later span is holding.
        for span in sorted(todo, key=lambda s: (-s.seg_id, -s.start)):
            record = _split_one(graph, span)
            if record is not None:
                records.append(record)
    return records


def _split_one(graph, span: Span) -> SplitRecord | None:
    """Cut one fill out, leaving a node on each surviving side.

    Index bookkeeping, since it is easy to get wrong by one. The fill occupies
    ``ids[start:stop]``; its anchors -- real points, shared with the vessel either side --
    are ``ids[start - 1]`` and ``ids[stop]``. ``split_segment(sid, i)`` puts a node *at*
    ``ids[i]`` and hands back the piece ending there and the piece starting there, so the
    two splits go at the anchors, never at a flagged point.
    """
    sid, start = span.seg_id, span.start
    if span.is_jump and span.anchors is not None:
        # A segment id does not survive `split_segment`, and three LADAF-28 segments
        # carry two jumps each -- so cutting the first retires the id the second is
        # holding and it would be dropped without a word. The point ids do survive,
        # exactly as `candidates._split_for_tjunction` relies on.
        resolved = _resolve_jump(graph, span)
        if resolved is None:
            return None
        sid, start = resolved
    if not graph.has_segment(sid):
        return None
    seg = graph.segment(sid)
    ids = list(seg["point_ids"])
    stop = start + span.n_points
    if start < 0 or stop > len(ids) or stop < start:
        return None
    if stop == start and not (0 < start <= len(ids) - 1):
        return None  # a jump needs a real point on each side of it
    if ids[start:stop] != span.point_ids:
        return None  # the segment moved under us; leave it alone

    attrs = {
        k: v for k, v in seg.items() if k not in ("id", "node1", "node2", "point_ids")
    }
    # The fill plus one anchor either side, captured before anything is deleted:
    # `add_segment` snaps a restored centreline onto its two nodes, and the nodes end up
    # exactly on those anchors.
    lo, hi = max(start - 1, 0), min(stop, len(ids) - 1)
    keep = ids[lo:hi + 1]
    coords = np.array([graph.points[p][:3] for p in keep], dtype=np.float64)
    radii = np.array([graph.points[p][3] for p in keep], dtype=np.float64)

    # An anchor that *is* one of the segment's own nodes needs no split on that side.
    left_is_node = start <= 1
    right_is_node = stop >= len(ids) - 1

    if left_is_node and right_is_node:
        node1, node2 = seg["node1"], seg["node2"]
        graph.delete_segment(sid)
        # `_delete_segment` drops a node it just orphaned. Only a fill whose anchor had
        # no other vessel can do that, and then there is nothing to reconnect it to.
        if node1 not in graph.nodes or node2 not in graph.nodes:
            return None
        return SplitRecord(sid, node1, node2, coords, radii, span.reason, attrs, True)

    if left_is_node:
        node1 = seg["node1"]
        node_hi, fill_side, _rest = graph.split_segment(sid, stop)
        graph.delete_segment(fill_side)
        if node1 not in graph.nodes or node_hi not in graph.nodes:
            return None
        return SplitRecord(sid, node1, node_hi, coords, radii, span.reason, attrs)

    if right_is_node:
        node2 = seg["node2"]
        node_lo, _rest, fill_side = graph.split_segment(sid, start - 1)
        graph.delete_segment(fill_side)
        if node_lo not in graph.nodes or node2 not in graph.nodes:
            return None
        return SplitRecord(sid, node_lo, node2, coords, radii, span.reason, attrs)

    # Interior on both sides: cut the far anchor off first, so the near split's index is
    # still measured from the same start of the point list.
    node_hi, low_piece, _rest = graph.split_segment(sid, stop)
    node_lo, _keep_low, fill_side = graph.split_segment(low_piece, start - 1)
    graph.delete_segment(fill_side)
    if node_lo not in graph.nodes or node_hi not in graph.nodes:
        return None
    return SplitRecord(sid, node_lo, node_hi, coords, radii, span.reason, attrs, False)


def _resolve_jump(graph, span: Span) -> tuple[int, int] | None:
    """``(segment id, start)`` for a jump, re-resolved through its two anchor points.

    Returns ``None`` when the two anchors are no longer adjacent in one segment, which
    means something else has already cut this step -- the right answer being to leave it
    alone rather than to cut it twice.
    """
    lo, hi = span.anchors
    owner = graph.segment_of_point()
    sid = owner.get(lo)
    if sid is None or owner.get(hi) != sid or not graph.has_segment(sid):
        return None
    ids = graph.segment(sid)["point_ids"]
    try:
        i = ids.index(lo)
    except ValueError:
        return None
    if i + 1 >= len(ids) or ids[i + 1] != hi:
        return None
    return sid, i + 1


def restore_unbridged(
    graph, records: Iterable[SplitRecord], *, label: str = "restore Avizo interpolation"
) -> list[SplitRecord]:
    """Put back every fill whose two sides are still in different pieces.

    The point of removing them was to let the image decide. Where it did -- where a
    DPC-validated bridge now joins the two sides -- the fill has been replaced by
    something measured, and it should stay gone. Where it did not, throwing away Avizo's
    line as well would be a silent loss of connectivity that nobody asked for, so it goes
    back, still flagged, and still excluded from every measurement.

    Returns the records that were actually restored.
    """
    records = [r for r in records if r is not None]
    if not records:
        return []

    restored: list[SplitRecord] = []
    with graph.batch(label):
        for record in records:
            if record.node1 not in graph.nodes or record.node2 not in graph.nodes:
                continue
            if _same_component(graph, record.node1, record.node2):
                continue
            attrs = {**record.attrs, FIELD: 1}
            attrs.pop("id", None)
            sid = graph.add_segment(
                record.node1, record.node2, record.coords, record.radii, attrs=attrs
            )
            # Interior points only: the two anchors are shared with the real vessels
            # either side and were never invented.
            pids = graph.segment(sid)["point_ids"]
            set_flags(graph, pids[1:-1], record.reason or STRAIGHT_BRIDGE)
            restored.append(record)
    return restored


def _same_component(graph, a: int, b: int) -> bool:
    """Whether two nodes are already joined by any path."""
    if a == b:
        return True
    seen = {a}
    stack = [a]
    while stack:
        nid = stack.pop()
        for sid in graph.node_segments(nid):
            seg = graph.segment(sid)
            other = seg["node2"] if seg["node1"] == nid else seg["node1"]
            if other == b:
                return True
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return False
