"""The planes ``radius-perimeter`` actually cuts, turned into geometry you can look at.

:func:`~.radius_perimeter.measure_radii` reports what it *decided* -- a radius per
point and a reason code -- and :mod:`~.diagnostics` prints those decisions as tables.
Neither shows the thing the decision was made from: a square of segmentation sampled
somewhere in space, at some orientation, of some size. When the measured radii come
back wrong the question is almost always about that square, and there are only three
ways for it to be wrong:

* **it is in the wrong place.** The centreline point is taken as the section's centre
  and never moved, so if the skeleton runs off the lumen's axis -- which is exactly
  what a thinned skeleton does on a bend, and what a collapsed lumen's flat distance
  ridge does everywhere -- the section is cut off-centre. The perimeter of an
  off-centre cut through a tube is *not* the perimeter of its cross-section.
  :func:`~.skeleton_optimise` exists to re-centre, and ``radius-perimeter`` does not
  run it. :attr:`SectionFrame.offset_um` is what that costs here: the distance from
  the point to its own section's area centroid, in the plane.
* **it is the wrong size.** The window starts at ``2.5 * stored radius`` and doubles
  while the blob touches its border, up to ``max_half``. A section that still reaches
  the border is refused as ``TRUNCATED`` and its radius is interpolated from
  neighbours instead of measured. :attr:`SectionFrame.extent_ratio` says how much of
  the window the section fills and :attr:`SectionFrame.grew` says whether doubling
  was needed at all.
* **it is at the wrong angle.** An oblique cut of a tube of radius ``r`` at ``theta``
  off its axis is an ellipse of semi-axes ``r`` and ``r / cos theta``, and its
  perimeter -- which is what the radius is taken from -- is too long by about that
  factor. The stability slab cannot see this, because it is stepped along the
  candidate normal and a straight vessel gives it three identical ellipses either
  way; see ``crosssection.TRANSVERSE_AXIS_RATIO``. :attr:`SectionFrame.axis_ratio`
  says how flat the section is and :attr:`SectionFrame.obliquity` says whether
  rotating the plane shortened its boundary -- which is what separates a collapsed
  lumen from a tilted cut, since both are ellipses.

So this module re-cuts, at *sampled* points rather than all of them, through the same
:func:`~..crosssection.stable_transverse_cut` the pass uses, with the same tangents
from :func:`~..crosssection.robust_edge_tangents`, and returns the plane's corners,
the measured lumen's boundary and its centroid in world um -- ready for
``Picker3D.show_cross_sections``.

**Two things it deliberately does not reproduce.** The junction mask
(``_adaptive_junction_mask``) and the tapers run *after* every section is known, over
whole segments; a sampled subset cannot reproduce them, so a frame marked
``accepted`` here may still be discarded as ``junction`` by the real pass. And the
second measurement pass (``n_passes``) feeds measured radii back as the geometry
scale; this always uses the stored radius, which is what pass 1 does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Verdicts, in the order a summary lists them. The first three are the pass's own
#: reject reasons under their own names (`radius_perimeter.REJECT_NAMES`); the last
#: two are states that never reach a reject code because the point is skipped first.
VERDICTS = ("accepted", "truncated", "unstable", "unmeasurable", "interpolated", "outside")

#: `half_radii` above which a window that *grew* is called a runaway in the summary.
#: It starts at 2.5 radii and doubles, so past 6 it has doubled at least once beyond
#: its own vessel's scale and is chasing something that is not this lumen.
RUNAWAY_HALF_RADII = 6.0


def _merge_cell(f) -> str:
    """The `merge` column: ``foreign/total`` rivals inside this blob, or ``-``.

    Two numbers rather than one, because the same observation means opposite things
    either side of a junction. ``0/2`` is two neighbours' centrelines in a section at
    a branch node, which is what a junction *is*. ``1/1`` is a vessel sharing no node
    with this one, inside the blob this radius was measured from.
    """
    if f.merged_rivals < 0:
        return "-"
    return f"{f.merged_foreign}/{f.merged_rivals}"


@dataclass
class SectionFrame:
    """One cut plane, and everything about it that can be drawn or blamed."""

    sid: int
    index: int  # position within the segment's own point order
    verdict: str
    point_um: np.ndarray  # (3,) the centreline point the plane is centred on
    tangent: np.ndarray  # (3,) the normal actually used, after any tangent search
    corners_um: np.ndarray  # (4, 3) the sampled window, in draw order
    contour_um: np.ndarray | None  # (M, 3) closed ring: the measured lumen boundary
    centroid_um: np.ndarray | None  # (3,) area centroid of the measured section
    half: int  # window half-width in segmentation voxels, after any growth
    half_um: float
    stored_radius_um: float  # what the graph carries here, and the geometry scale used
    radius_um: float  # measured at this section; NaN where nothing was measured
    offset_um: float  # |centroid - point| in the plane: the re-centring that was not done
    extent_ratio: float  # furthest section voxel / half -- 1.0 means it fills the window
    grew: bool  # the window had to be doubled at least once
    searched: bool  # a rotated tangent beat the fitted one
    #: ``major / minor`` semi-axis of the measured section. 1.0 is round; above it
    #: the lumen is collapsed, the plane is oblique, or both -- which is what
    #: `obliquity` is for.
    axis_ratio: float = float("nan")
    #: How much longer the boundary was on the *fitted* tangent than on the one
    #: finally used. 1.0 means the fitted tangent was already the best cut in the
    #: search cone, so an elongated section here is a genuinely collapsed lumen.
    #: Above 1.0 the fitted tangent was oblique and would have over-read the radius
    #: by this factor. NaN where there was nothing to compare against -- see
    #: `crosssection.TRANSVERSE_AXIS_RATIO`.
    obliquity: float = float("nan")
    #: How many *other* segments' centreline samples lie inside this section's own
    #: 8-connected blob -- the pass's own ownership test
    #: (`radius_perimeter._rival_lies_in_blob`), reported rather than acted on. This
    #: is "my window has swallowed somebody else's vessel", and it is the one thing
    #: `axis_ratio` and `obliquity` cannot say: two vessels measured as one and a
    #: single collapsed lumen are both elongated at obliquity 1.00, because rotating
    #: the plane shortens neither. ``-1`` means the question was not asked.
    merged_rivals: int = -1
    #: Of those, how many share a node with this segment. At a junction the lumens
    #: really are continuous, so a neighbour's centreline inside this section is what
    #: a junction *looks like* and is not evidence of anything. The difference is.
    merged_adjacent: int = -1

    @property
    def measured(self) -> bool:
        return bool(np.isfinite(self.radius_um))

    @property
    def merged_foreign(self) -> int:
        """Rivals in the blob that share no node with this segment, or ``-1``.

        The number that means something is wrong. `measure_radii` would send this
        section to its local 3-D watershed or refuse it as ``BRANCH_OVERLAP``; the
        survey does neither and reports the merged reading, so this is the only
        warning that the radius beside it is of two vessels.
        """
        if self.merged_rivals < 0:
            return -1
        return int(self.merged_rivals - self.merged_adjacent)

    @property
    def half_radii(self) -> float:
        """Window half-width in multiples of this point's *own* stored radius.

        `grew` says whether the window had to double; this says how far it ran. It
        starts at 2.5 radii, so 5 is one doubling. Measured on real data: a 260 um
        vessel reported at half=88 voxels was at 11 -- a 2.8 mm window around a
        vessel 0.26 mm wide, which had long since stopped being about this vessel.
        `grow_radii` is the ceiling, and it is off by default because the pass's is.
        """
        r = self.stored_radius_um
        return float(self.half_um / r) if r > 0 else float("nan")

    @property
    def offset_radii(self) -> float:
        """The centroid offset in units of the *measured* radius, or NaN."""
        r = self.radius_um
        return float(self.offset_um / r) if np.isfinite(r) and r > 0 else float("nan")

    @property
    def radius_ratio(self) -> float:
        """``measured / stored``. What ``radius-perimeter`` would change here."""
        if not np.isfinite(self.radius_um) or self.stored_radius_um <= 0:
            return float("nan")
        return float(self.radius_um / self.stored_radius_um)


@dataclass
class SectionSurvey:
    """Every sampled frame, plus the counts a summary line needs."""

    frames: list = field(default_factory=list)
    n_points: int = 0  # points in the segments visited, sampled or not
    stride: int = 1
    seconds: float = 0.0
    truncated_by_cap: bool = False  # `max_frames` stopped the walk early
    grow_radii: float | None = None  # the growth ceiling this run used; None is none
    rivals_tested: bool = False  # the merge check ran
    rival_error: str = ""  # ...or why it could not

    def counts(self) -> dict:
        out = {name: 0 for name in VERDICTS}
        for f in self.frames:
            out[f.verdict] = out.get(f.verdict, 0) + 1
        return out

    def segments(self) -> list:
        seen = []
        for f in self.frames:
            if f.sid not in seen:
                seen.append(f.sid)
        return seen

    def describe(self) -> str:
        if not self.frames:
            return "no sections sampled"
        counts = self.counts()
        parts = ", ".join(f"{counts[k]} {k}" for k in VERDICTS if counts.get(k))
        line = (
            f"{len(self.frames)} sections over {len(self.segments())} segment(s) "
            f"(every {self.stride} point(s) of {self.n_points:,}): {parts}"
        )
        offsets = np.array([f.offset_um for f in self.frames if f.measured])
        radii = np.array([f.offset_radii for f in self.frames if f.measured])
        radii = radii[np.isfinite(radii)]
        if offsets.size:
            line += (
                f"\n  centroid offset: median {np.median(offsets):,.0f} um, "
                f"p95 {np.percentile(offsets, 95):,.0f} um"
            )
            if radii.size:
                line += (
                    f" ({np.median(radii):.2f} r, p95 {np.percentile(radii, 95):.2f} r)"
                    " -- how far re-centring would move each point"
                )
        extent = np.array([f.extent_ratio for f in self.frames if f.measured])
        grew = sum(1 for f in self.frames if f.grew)
        # Over frames that actually *grew*, and only those: a window forced wide with
        # `initial_half` is the operator's own decision, and advising against it would
        # be wrong.
        reach = np.array([f.half_radii for f in self.frames if f.grew])
        reach = reach[np.isfinite(reach)]
        if extent.size:
            line += (
                f"\n  window fill: median {np.median(extent):.2f}, "
                f"p95 {np.percentile(extent, 95):.2f} of the half-width; "
                f"grown on {grew} of {len(self.frames)}"
            )
            if reach.size:
                line += (
                    f"; the widest grew to {reach.max():.1f}x its own stored radius"
                )
        if reach.size and self.grow_radii is None and reach.max() >= RUNAWAY_HALF_RADII:
            line += (
                f"\n  a window ran to {reach.max():.0f}x this point's own radius with "
                "no growth ceiling -- which is what `radius-perimeter` itself does by "
                "default. A plane that is not transverse cuts a streak *along* the "
                "vessel, and a streak touches the border at any width, so the window "
                "doubles until it stops: by then it holds the vessel next door too, "
                "and the radius is of both. Set 'grow ceiling' to 4, cut again, and "
                "read the 'merge' column."
            )
        ratio = np.array([f.radius_ratio for f in self.frames])
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            line += (
                f"\n  measured/stored radius: median {np.median(ratio):.2f}, "
                f"p5 {np.percentile(ratio, 5):.2f}, p95 {np.percentile(ratio, 95):.2f}"
            )
        if not any(f.measured for f in self.frames):
            # The measured-lumen and centroid-offset layers are fed from measured
            # frames only -- see `drawables` -- so they come back empty here, and an
            # empty layer looks identical to a broken one. Say which it is, point at
            # the row that does have geometry, and name the two things that produce a
            # whole run of refusals.
            line += (
                "\n  nothing was measured, so the measured-lumen and centroid-offset "
                "layers are empty; the windows are still drawn in orange, and each "
                "refused blob's own boundary with them."
            )
            if counts.get("unmeasurable"):
                line += (
                    f"\n  {counts['unmeasurable']} section(s) found no lumen at the "
                    "centreline point at all. That is usually the voxel size: if the "
                    "session was loaded at a size the graph does not agree with, every "
                    "point is displaced by that ratio and lands outside its own vessel. "
                    "Check the Data tab against the size the graph records."
                )
            if counts.get("truncated"):
                line += (
                    f"\n  {counts['truncated']} section(s) reached the window border: "
                    "raise 'max half' until they close -- a vessel needs 2.5x its "
                    "radius in voxels."
                )
        shape = self.describe_shape()
        if shape:
            line += "\n  " + shape
        merges = self.describe_merges()
        if merges:
            line += "\n  " + merges
        if self.truncated_by_cap:
            line += "\n  stopped at the frame cap -- raise it or sample more coarsely"
        return line

    def describe_shape(self) -> str:
        """Whether the elongated sections are collapsed lumens or oblique cuts.

        The distinction the radius turns on, and the one a shape measure alone
        cannot make, because both give an ellipse. :attr:`SectionFrame.obliquity`
        can, because it is the answer to "would rotating the plane have shortened
        this boundary?" -- yes for a tilted cut, no for a genuinely flat lumen.
        """
        measured = [f for f in self.frames if f.measured]
        if not measured:
            return ""
        axis = np.array([f.axis_ratio for f in measured])
        axis = axis[np.isfinite(axis)]
        obl = np.array([f.obliquity for f in measured])
        obl = obl[np.isfinite(obl)]
        parts = []
        if axis.size:
            parts.append(
                f"section shape: median {np.median(axis):.2f}:1, "
                f"p95 {np.percentile(axis, 95):.2f}:1 "
                f"({int((axis > 1.5).sum())} of {axis.size} flatter than 1.5:1)"
            )
        if obl.size:
            rotated = sum(1 for f in measured if f.searched)
            parts.append(
                f"obliquity: median {np.median(obl):.2f}, "
                f"p95 {np.percentile(obl, 95):.2f}, rotated on {rotated} of "
                f"{len(measured)} -- above 1.0 the fitted tangent was tilted and "
                f"would have over-read the radius by that factor"
            )
        return "\n  ".join(parts)

    def describe_merges(self) -> str:
        """Whether an elongated section is one collapsed lumen or two merged vessels.

        The distinction :meth:`describe_shape` *cannot* make: a genuinely flat lumen
        and a section that has merged with the vessel beside it are both elongated at
        obliquity 1.00, because rotating the plane shortens neither. What separates
        them is whether somebody else's centreline is inside this blob -- the same
        test `radius_perimeter` uses to decide ownership, except that the pass then
        resolves or refuses the cut and this reports it and measures it anyway.
        """
        if not self.rivals_tested:
            if self.rival_error:
                return f"merge check failed: {self.rival_error}"
            # An absent column reads as "no merges", so say which it is.
            return ("merge check off -- an elongated section here could be one "
                    "collapsed lumen or two vessels measured as one, and nothing "
                    "else in this table can tell you which")
        tested = [f for f in self.frames if f.merged_rivals >= 0]
        if not tested:
            return ""
        foreign = [f for f in tested if f.merged_foreign > 0]
        neighbourly = [f for f in tested
                       if f.merged_rivals > 0 and not f.merged_foreign]
        out = (
            f"merged sections: {len(foreign)} of {len(tested)} hold a non-adjacent "
            f"segment's centreline -- the radius there is of both vessels; "
            f"{len(neighbourly)} more hold only a neighbour's, which is what a "
            f"junction looks like and is expected"
        )
        if foreign:
            worst = max(foreign, key=lambda f: (f.half_radii
                                                if np.isfinite(f.half_radii) else 0.0))
            out += (
                f"\n  worst: segment {worst.sid} point {worst.index} -- "
                f"{worst.merged_foreign} foreign centreline(s) in a window "
                f"{worst.half_radii:.1f}x its own radius, r={worst.radius_um:,.0f} um "
                f"against a stored {worst.stored_radius_um:,.0f} um"
            )
        return out

    def table(self, limit: int = 40) -> list:
        """One line per frame, worst centroid offset first. For the log pane."""
        rows = sorted(
            self.frames,
            key=lambda f: (-f.offset_radii if np.isfinite(f.offset_radii) else 0.0),
        )
        out = [f"{'seg':>5} {'pt':>5} {'verdict':<13} {'r_um':>8} {'stored':>8} "
               f"{'off_um':>8} {'off/r':>6} {'half':>5} {'h/r':>5} {'fill':>5} "
               f"{'shape':>6} {'obliq':>6} {'merge':>6}"]
        for f in rows[:limit]:
            out.append(
                f"{f.sid:>5} {f.index:>5} {f.verdict:<13} "
                f"{f.radius_um:>8.1f} {f.stored_radius_um:>8.1f} "
                f"{f.offset_um:>8.1f} {f.offset_radii:>6.2f} "
                f"{f.half:>5} {f.half_radii:>5.1f} {f.extent_ratio:>5.2f} "
                f"{f.axis_ratio:>6.2f} {f.obliquity:>6.2f} {_merge_cell(f):>6}"
            )
        if len(rows) > limit:
            out.append(f"  ... {len(rows) - limit} more")
        return out


def sample_indices(n: int, stride: int, max_per_segment: int | None = None) -> np.ndarray:
    """Which points of an ``n``-point segment to cut.

    Both ends are always included. They are where the junction mask bites and where a
    tangent is one-sided, so a sampling that quietly dropped them would hide the two
    places a section is most likely to be wrong.
    """
    if n <= 0:
        return np.zeros(0, dtype=int)
    stride = max(int(stride), 1)
    idx = np.unique(np.concatenate([np.arange(0, n, stride), [n - 1]]))
    if max_per_segment is not None and len(idx) > max_per_segment > 0:
        keep = np.linspace(0, len(idx) - 1, int(max_per_segment)).round().astype(int)
        idx = idx[np.unique(keep)]
    return idx.astype(int)


def _contour_offsets(blob: np.ndarray, half: int) -> np.ndarray | None:
    """The blob's boundary as ``(M, 2)`` offsets from the window centre, closed.

    ``cv2`` reports contour points as ``(x, y)`` -- column then row -- and the plane is
    indexed ``[u, v]``, so the pair is swapped back here. The same contour the
    perimeter is measured from, by the same call, so what is drawn is what was
    measured rather than a second opinion about the same blob.
    """
    import cv2

    contours, _ = cv2.findContours(
        (blob.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    best = max(contours, key=lambda c: cv2.arcLength(c, True))
    pts = np.asarray(best, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return None
    uv = np.column_stack([pts[:, 1], pts[:, 0]]) - float(half)
    return np.vstack([uv, uv[:1]])  # closed ring


def _to_um(frame, centre_ijk, u, v, uv) -> np.ndarray:
    """``(M, 2)`` in-plane offsets in voxels -> ``(M, 3)`` world um.

    Through the plane's own axes and `frame.seg_to_um`, so an anisotropic lattice --
    where a step of one voxel along `u` is not the same distance as along `v` -- comes
    out as the parallelogram it actually is rather than as an assumed square.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    ijk = (
        np.asarray(centre_ijk, dtype=np.float64)[None, :]
        + uv[:, :1] * np.asarray(u, dtype=np.float64)[None, :]
        + uv[:, 1:2] * np.asarray(v, dtype=np.float64)[None, :]
    )
    return np.asarray(frame.seg_to_um(ijk), dtype=np.float64)


def survey(
    graph,
    frame,
    labels,
    sids=None,
    *,
    stride: int = 8,
    max_per_segment: int | None = None,
    max_frames: int = 600,
    branch_aware: bool = True,
    max_half: int = 128,
    initial_half: int | None = None,
    tangent_search_degrees: float = 20.0,
    transverse_axis_ratio: float | None = None,
    min_blob_voxels: int | None = None,
    perimeter_correction: bool = True,
    # Stated here rather than left to `stable_transverse_cut`'s own defaults, because
    # this function's contract is to reproduce `measure_radii` and a default that
    # agrees by coincidence is one that stops agreeing without anything failing. That
    # is exactly what happened: the pass gained `grow_radii` and the two stability
    # thresholds, this did not, and the only symptom was that a 260 um vessel came
    # back at 1,062 um. `test_the_survey_defaults_match_the_pass_it_reproduces`
    # compares the two signatures so the next one has to be a decision.
    grow_radii: float | None = None,
    stability_variation: float = 1.5,
    stability_centroid_radii: float = 0.5,
    stability_centroid_mode: str = "drift",
    rival_check: bool = True,
    progress=None,
) -> SectionSurvey:
    """Re-cut sampled points of `sids` and return the planes as drawable geometry.

    `sids` defaults to every segment, which on a full tree is why `stride` and
    `max_frames` exist: a section costs up to three plane samples per candidate
    tangent, so cutting all 37k points here would cost what the measurement pass
    costs, and the answer this is for is visible in a few hundred.

    `initial_half` forces the starting window (in voxels) instead of deriving it from
    the stored radius -- the direct test of "the planes are not big enough".

    `transverse_axis_ratio` is the pass's own gate on re-cutting an elongated but
    stable section over the tangent cone (`crosssection.TRANSVERSE_AXIS_RATIO`).
    Setting it to `inf` here reproduces the pre-fix behaviour -- trust the fitted
    tangent whenever its section was stable -- which is the direct test of "the
    planes are cut at the wrong axis": the radii that move are the oblique ones.

    `grow_radii` is the pass's ceiling on how far one window may double, in multiples
    of that point's own radius, and it is off by default *because the pass's is*. A
    panel whose default differed would hand back a clean number the real run will not
    reproduce. What this does instead is make the runaway legible:
    :attr:`SectionFrame.half_radii` reports how far each window actually ran and
    :meth:`SectionSurvey.describe` names the worst.

    `rival_check` counts, per section, how many *other* segments' centrelines lie
    inside its own blob -- the pass's own ownership test, reported rather than acted
    on. It costs one KD-tree over every centreline point of the whole graph, however
    few segments were asked for, because a rival is by definition not in the
    selection.
    """
    import time

    from ..crosssection import (
        MIN_BLOB_VOXELS,
        TRANSVERSE_AXIS_RATIO,
        _PlaneSampler,
        _blob_axis_ratio,
        _perimeter_um,
        cut,
        robust_edge_tangents,
        stable_transverse_cut,
    )
    from .interpolation import mask_for_segment
    from .radius_perimeter import (
        _BranchContext,
        _grow_to,
        _rival_lies_in_blob,
        correct_perimeter_radius,
    )

    t0 = time.time()
    if min_blob_voxels is None:
        min_blob_voxels = MIN_BLOB_VOXELS
    if transverse_axis_ratio is None:
        transverse_axis_ratio = TRANSVERSE_AXIS_RATIO
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    dims = np.asarray(frame.seg_dims, dtype=np.float64)
    sids = list(graph.segment_ids()) if sids is None else [int(s) for s in sids]

    out = SectionSurvey(
        stride=max(int(stride), 1),
        grow_radii=None if grow_radii is None else float(grow_radii),
    )

    # Once per call, and over the *whole* graph however few segments were asked for: a
    # rival is by definition not in the selection. Wrapped because a diagnostic must
    # not take the cut down with it -- a graph too thin to carry `node_segments` should
    # cost the merge column, not the sections.
    ctx = None
    if rival_check:
        try:
            ctx = _BranchContext.build(graph)
            out.rivals_tested = True
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            out.rival_error = f"{type(exc).__name__}: {exc}"

    for n_done, sid in enumerate(sids, 1):
        if not graph.has_segment(sid):
            continue
        coords = np.asarray(graph.coords(sid), dtype=np.float64)
        stored = np.asarray(graph.radii(sid), dtype=np.float64)
        n = len(coords)
        out.n_points += n
        if n == 0:
            continue
        # The same tangents the pass fits, from the same stored radii: a debug view
        # cut on a different normal would be answering a different question.
        tangents = robust_edge_tangents(coords, stored, spacing_um=sp)
        ijk = frame.um_to_seg(coords)
        invented = mask_for_segment(graph, sid)
        if invented.size != n:
            invented = np.zeros(n, dtype=bool)

        for i in sample_indices(n, out.stride, max_per_segment):
            if len(out.frames) >= max_frames > 0:
                out.truncated_by_cap = True
                break
            centre = ijk[i]
            tangent = np.asarray(tangents[i], dtype=float)
            rp_vox = max(float(stored[i]) / sp, 1.0)
            nominal_half = (
                min(int(rp_vox * 2.5) + 2, max_half) if initial_half is None
                else min(max(int(initial_half), 2), max_half)
            )

            if np.any(centre < 0) or np.any(centre >= dims):
                out.frames.append(_frame_from(
                    frame, sid, i, "outside", coords[i], tangent, centre,
                    None, nominal_half, sp, float(stored[i]),
                ))
                continue

            chosen = None
            if branch_aware:
                chosen = stable_transverse_cut(
                    sampler, centre, tangent, rp_vox, spacing_um=sp,
                    max_half=max_half, search_degrees=tangent_search_degrees,
                    min_blob_voxels=min_blob_voxels, initial_half=initial_half,
                    slab_offsets=(0.0, 0.25, 0.5) if i == 0 else (
                        (-0.5, -0.25, 0.0) if i == n - 1 else (-0.5, 0.0, 0.5)
                    ),
                    transverse_axis_ratio=transverse_axis_ratio,
                    grow_radii=grow_radii,
                    centroid_mode=stability_centroid_mode,
                    max_variation=stability_variation,
                    max_centroid_radii=stability_centroid_radii,
                )
            c = chosen.cut if chosen is not None else None
            searched = bool(chosen.searched) if chosen is not None else False
            obliquity = float(chosen.obliquity) if chosen is not None else float("nan")
            if chosen is not None:
                tangent = np.asarray(chosen.tangent, dtype=float)

            if c is not None:
                verdict = "truncated" if c.touches_border else "accepted"
            else:
                # The same plain cut on the fitted tangent that `measure_radii` probes
                # with, and with branch-aware off it is the whole selector. After a
                # stable cut has already refused, it is what separates "no section
                # here at all" from "a section that would not close in its window";
                # on its own it is simply the legacy answer.
                c = cut(sampler, centre, tangent, nominal_half, max_half=max_half,
                        min_blob_voxels=min_blob_voxels,
                        grow_to=_grow_to(grow_radii, rp_vox))
                verdict = (
                    "unmeasurable" if c is None
                    else "truncated" if c.touches_border
                    else "unstable" if branch_aware
                    else "accepted"
                )
            if invented[i]:
                # Reported over any other verdict: the pass refuses these before it
                # cuts, so whatever the section looks like it is not measured.
                verdict = "interpolated"

            # Counted for refused frames too: "a blob merged with its neighbour" is
            # one of the three things the amber refused-ring layer is drawn to show,
            # and this is the number behind it. On the *chosen* tangent, which is what
            # `measure_radii` queries with. Both counts are kept rather than one gate
            # because the pass's own gate collapses them, and what that costs is the
            # open question.
            n_rivals = n_adjacent = -1
            if ctx is not None and c is not None:
                n_rivals = n_adjacent = 0
                for item in ctx.rivals(sid, coords[i], tangent, float(stored[i])):
                    if _rival_lies_in_blob(c, coords[i], item[2], sp):
                        n_rivals += 1
                        n_adjacent += bool(item[4])

            out.frames.append(_frame_from(
                frame, sid, i, verdict, coords[i], tangent, centre, c,
                nominal_half, sp, float(stored[i]),
                perimeter_correction=perimeter_correction,
                perimeter_um=_perimeter_um, correct=correct_perimeter_radius,
                searched=searched, obliquity=obliquity,
                axis_ratio=_blob_axis_ratio,
                merged_rivals=n_rivals, merged_adjacent=n_adjacent,
            ))
        if progress is not None:
            progress(n_done, len(sids))
        if out.truncated_by_cap:
            break

    out.seconds = time.time() - t0
    return out


def _frame_from(frame, sid, i, verdict, point_um, tangent, centre_ijk, c,
                nominal_half, sp, stored_um, *, perimeter_correction=True,
                perimeter_um=None, correct=None, searched=False,
                obliquity=float("nan"), axis_ratio=None,
                merged_rivals=-1, merged_adjacent=-1) -> SectionFrame:
    """Assemble one :class:`SectionFrame`, with or without a cut behind it.

    A refused point still gets its window drawn -- from the fitted tangent and the
    nominal half-width -- because "there is no section here" and "the section did not
    fit in this square" look identical in a table and completely different on screen.
    """
    half = int(c.half) if c is not None else int(nominal_half)
    if c is not None:
        u, v = np.asarray(c.u, dtype=float), np.asarray(c.v, dtype=float)
    else:
        from ..crosssection import _plane_axes

        axes = _plane_axes(np.asarray(tangent, dtype=float))
        u, v = axes if axes is not None else (
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        )
    corners = _to_um(
        frame, centre_ijk, u, v,
        [(-half, -half), (half, -half), (half, half), (-half, half)],
    )

    contour = centroid = None
    radius = float("nan")
    offset = float("nan")
    extent = float("nan")
    shape = float("nan")
    if c is not None:
        blob = c.blob4
        if axis_ratio is not None:
            shape = float(axis_ratio(blob))
        uv = np.argwhere(blob).astype(np.float64) - float(half)
        if len(uv):
            centroid = _to_um(frame, centre_ijk, u, v, uv.mean(axis=0)[None, :])[0]
            offset = float(np.linalg.norm(centroid - np.asarray(point_um, dtype=float)))
            extent = float(np.abs(uv).max() / max(half, 1))
        ring = _contour_offsets(blob, half)
        if ring is not None:
            contour = _to_um(frame, centre_ijk, u, v, ring)
        # Only a section that closed inside its own window gets a radius. A truncated
        # blob still has a perimeter -- part lumen, part window edge -- and reporting
        # it would put a number in the table that the pass itself refuses. Its
        # centroid is drawn anyway, because seeing *where* the crop sat is the point,
        # but `measured` stays False so it is kept out of every statistic.
        if verdict == "accepted" and perimeter_um is not None:
            r_perim = perimeter_um(blob, sp) / (2.0 * np.pi)
            radius = float(
                correct(r_perim, sp) if perimeter_correction and correct else r_perim
            )

    return SectionFrame(
        sid=int(sid), index=int(i), verdict=verdict,
        point_um=np.asarray(point_um, dtype=float),
        tangent=np.asarray(tangent, dtype=float),
        corners_um=corners, contour_um=contour, centroid_um=centroid,
        half=half, half_um=float(half) * sp,
        stored_radius_um=float(stored_um), radius_um=radius,
        offset_um=offset, extent_ratio=extent,
        grew=bool(getattr(c, "grew", False)), searched=bool(searched),
        axis_ratio=shape, obliquity=float(obliquity),
        merged_rivals=int(merged_rivals), merged_adjacent=int(merged_adjacent),
    )


def drawables(survey_or_frames, *, ok_verdicts=("accepted",)):
    """Split a survey into the five arrays ``Picker3D.show_cross_sections`` wants.

    Kept here rather than in the viewer so the split -- which frames read as a
    measurement and which as a failure -- is stated once, next to the verdicts it is
    made from, and can be tested without a plotter.

    **The boundary follows the verdict onto a different layer; the offset does not
    follow it at all.** A cut that reaches its window border grows until it stops, so
    a plane that is not actually perpendicular becomes a slab *along* the vessel and
    comes back as a serrated ribbon hundreds of voxels long. On a layer labelled
    "measured lumen" that ribbon reads as a cross-section of impossible size, which is
    the opposite of what the verdict says about it. But it is also *the* diagnostic: a
    streak means the plane was cut on the wrong axis, a blob filling its window means
    the window was too small, a blob merged with its neighbour means the cut caught a
    second vessel. So it is drawn, on its own row, in the refused colour.

    Its **centroid** is not. `PlaneCut.trustworthy` refuses the same number for the
    same reason: the area centroid of a slab that grew along the vessel can sit
    hundreds of um downstream of the point, and the offset layer reports a *distance*
    -- how far re-centring would move this point. A slab's centroid is not a centre.
    Note `offset_um` is finite for a refused frame, so the `ok and` below is
    load-bearing rather than decorative.

    `contours` and `refused` therefore **partition** the frames that have a ring:
    every ring that exists is drawn exactly once, on the layer its verdict names.
    """
    frames = getattr(survey_or_frames, "frames", survey_or_frames)
    good, bad, contours, offsets, refused = [], [], [], [], []
    for f in frames:
        ok = f.verdict in ok_verdicts
        (good if ok else bad).append(f.corners_um)
        if f.contour_um is not None:
            (contours if ok else refused).append(f.contour_um)
        if ok and f.centroid_um is not None and np.isfinite(f.offset_um):
            offsets.append(np.vstack([f.point_um, f.centroid_um]))
    return good, bad, contours, offsets, refused
