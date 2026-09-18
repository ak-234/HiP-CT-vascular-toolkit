"""Replace every point's radius with one measured from its own cross-section.

``adjust_thickness.py`` already established what the radius *should* be. It cuts
the lumen perpendicular to the centreline, keeps the component containing the plane
centre, and assigns

    r = cv2.arcLength(contour) / (2 * pi)

on the assumption that a collapsed lumen's *perimeter* survives fixation even though
its shape does not -- so a slit is deliberately reconstructed as a rounder, larger
circle. That assumption is the design and is not in question here.

What it then does is the problem: it **throws the measurements away**. The per-point
perimeters are reduced to a single global linear fit
``r ~ slope * (Amira distance-transform thickness) + intercept`` and that fit is
applied everywhere. Measured over 11,094 cross-sections the fit is right on average --
``r_stored / r_perimeter`` has median 0.96 -- but correlation is only ~0.66 with a
p5-p95 spread of 0.49-2.98, so at any one location the stored radius can be out by a
factor of two in either direction. :func:`~..crosssection.find_sites` reports exactly
this as ``perimeter_mismatch``.

This module keeps the measurements. Each point gets the radius of the section cut at
*that* point, so ``perimeter_mismatch`` becomes true by construction rather than
something to audit for afterwards.

**The estimator is hybrid, and says which one it used.** Below :data:`GATE_VOXELS`
the area estimator ``sqrt(area/pi)`` is used instead of ``perimeter/(2*pi)``, being
insensitive to boundary roughness. Which one produced each value is written out as a
per-point ``radius_source`` field: a graph whose radii come from two estimators is
fine, a graph that *hides* which is not.

**The perimeter estimator under-reads, and is corrected for it.** This repository
and ``skeleton_analysis``'s ``PORTING.md`` both used to say the opposite --
that the staircase boundary *inflates* a section a voxel or two across, citing a
radius-5 disc reading 5.25 against area's 5.08. That figure is real but was taken on
a disc of integer radius centred on a voxel centre, which is the single most
lattice-favourable configuration there is; swept over sub-voxel offsets the same
radius reads 4.80, and a two-voxel section reads 0.82x true. See
:data:`CHAIN_CODE_FACTOR` for the mechanism and :func:`correct_perimeter_radius` for
the inverse, which is applied by default and marked ``PERIMETER_CORRECTED``.

**MeanRadius is re-derived everywhere.** ``Python_port_test.py`` Stage 5 re-derives it
only on the edges it corrected, deliberately -- its input file's ``thickness`` and
``MeanRadius`` sit on scales differing by a non-constant ~2.46x (``PORTING.md`` gotcha
#17), so a global re-derivation would rescale untouched edges. That caveat is what
makes its output mixed-scale unless the input already used perimeter radii. Here every
point is re-measured, so there is no untouched edge and no second scale to preserve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..crosssection import TRANSVERSE_AXIS_RATIO
from .interpolation import mask_for_segment

# Below this many voxels of radius, `perimeter/(2*pi)` is measuring the staircase and
# not the lumen. Three voxels is where a digitised disc's perimeter error falls under
# roughly 5% -- see `tests/test_radius_perimeter.py`, which asserts the crossover.
#
# **Zero, so every point is measured by perimeter.** The staircase argument is about a
# section being *small*; the reason this pass exists at all is that perimeter survives
# *collapse*. A section that is both -- which is most of a thin ex-vivo vessel -- meets
# the size gate and gets area, and area under-reads a flattened ellipse badly (10:1
# gives r_area/r_perim ~ 0.57). Measured on LADAF-2024-28 at GATE_VOXELS=3.0: 5,568
# points (19.1%) took the area branch and read a median 0.73x the perimeter points in
# their *own* segment, as little as 0.36x -- a stretch rendered half the width of the
# vessel either side of it, widening again where the estimator switched back. That is
# a bigger error than the staircase inflation it was avoiding, and it lands on exactly
# the vessels the perimeter argument was made for. Set to 3.0 to restore the hybrid.
GATE_VOXELS = 0.0

# Confidence guards.  These are expressed relative to the input radius because the
# input remains a useful local scale even when its absolute calibration is imperfect.
JUNCTION_RADII = 2.0
#: Most a junction may consume of one segment, as a fraction of that segment's own
#: arclength, per end. Both ends together therefore leave at least
#: ``1 - 2 * JUNCTION_MASK_MAX_FRACTION`` of every segment free to be measured.
#:
#: **Without this the mask can eat a whole segment, and on LADAF-2024-28 it ate 66
#: of 309 (21%) -- 1,679 points over 166.9 mm, every one of them rejected as
#: `junction` with nothing measured to fall back on.** `_adaptive_junction_mask`
#: walks inward until it meets two consecutive *exclusive* sections; a section is
#: exclusive when it is stable and no topology-adjacent branch is near it. Near a
#: junction an adjacent branch is always near it, so on a short segment between two
#: junctions the walk never terminates and the `else` branch commits everything.
#: Measured on segments 265 and 268: every point stable (19/19 and 9/9), every point
#: adjacent (19/19 and 9/9), nothing ever exclusive, both ends consuming the full
#: 1994 um and 797 um.
#:
#: The fill that follows is then anchored on whichever neighbours survived, and the
#: result is not a bias but a scramble: re-measuring all 66 against the segmentation
#: put four fifths of them more than 25% off, as many too wide as too narrow. The
#: median ratio is 0.93, which is exactly why this went unnoticed -- the errors
#: cancel in the aggregate statistics.
#:
#: 0.4 leaves a fifth of every segment measurable. Set to ``None`` to restore the
#: unbounded walk. This does not weaken the exclusivity test itself, which is doing
#: real work where a section genuinely does straddle two lumens; it only refuses to
#: let that test consume an entire branch.
JUNCTION_MASK_MAX_FRACTION = 0.4
RUNAWAY_HALF_RADII = 4.0
RUNAWAY_FACTOR = 3.0
# A junction daughter at or above this fraction of its parent's calibre is the
# vessel continuing, not a branch emerging through the parent's wall, so it must
# not receive a carina taper. Measured on LADAF_2024_28, half of all tapered
# daughters sat at >= 0.70 and a quarter at >= 0.90 of their parent.
CONTINUATION_RATIO = 0.70

#: What to do with a segment that yielded no trustworthy cross-section at all.
#: `retain` keeps its input radii -- which are the uncorrected `adjust_thickness`
#: values this pass exists to replace, and on LADAF_2024_28 they reached 2356 um
#: on a segment whose own median is 129 um, an 18x spike that renders as a
#: sphere. `rescale` keeps the segment's SHAPE but puts it on the measured scale,
#: using the median measured/input ratio from the rest of the tree. `drop`
#: refuses to invent a calibre and interpolates across the segment from the
#: measured radii at its junction neighbours.
JUNCTION_FLARE_MODES = ("none", "parent", "all")
FALLBACK_POLICIES = ("retain", "rescale", "drop")
FALLBACK_POLICY = "rescale"
LOCAL_RADIUS_FACTOR = 2.0

#: ``cv2.arcLength`` does not measure the boundary of the blob, it measures a
#: staircase through the *centres* of the boundary pixels. Both of those are errors
#: and they have opposite signs:
#:
#: * tracing centres rather than edges follows a circle of radius ``r - 0.5``
#:   instead of ``r`` -- at a two-voxel section that alone is -25%;
#: * the traced path is a chain code, longer than the smooth path it approximates
#:   by ``mean(cos t + (sqrt(2) - 1) sin t)`` for ``t`` uniform on ``[0, 45)``
#:   degrees, which integrates to 1.0548.
#:
#: So ``r_est = 1.0548 * (r - 0.5)``, which fits every radius from 0.75 to 8 voxels
#: to within 0.08 voxels and **changes sign at 9.6 voxels**: above that the
#: estimator over-reads, below it under-reads, and nearly every coronary section is
#: below it. Inverting the model recovers the true radius to within 3% down to one
#: voxel. Both constants are derived rather than fitted, and the inverse was
#: checked on radii held out from the ones that stated the model; see
#: ``coronary_sdf``'s ``research_scripts/subvoxel_bias.py``, which measures this against
#: analytic discs, ellipses and cylinders, and reproduces it through this function.
#:
#: The residual is a consistent +2%. It is left alone deliberately: absorbing it
#: into the constants would be fitting them to the digitisation of one test.
CHAIN_CODE_FACTOR = 1.0548
CENTRE_TRACE_INSET_VOXELS = 0.5


def correct_perimeter_radius(r_um: float, spacing_um: float) -> float:
    """Undo the estimator's digitisation bias. See :data:`CHAIN_CODE_FACTOR`.

    A section that traced no contour at all -- one or two voxels, whose closed
    contour has zero length -- has nothing to correct, and is left at zero rather
    than lifted to half a voxel out of nowhere.
    """
    if not (r_um > 0.0):
        return r_um
    return r_um / CHAIN_CODE_FACTOR + CENTRE_TRACE_INSET_VOXELS * spacing_um


PERIMETER, AREA, FILLED, PERIMETER_CORRECTED = 0, 1, 2, 3
SOURCE_NAMES = {
    PERIMETER: "perimeter",
    AREA: "area",
    FILLED: "filled",
    PERIMETER_CORRECTED: "perimeter (bias-corrected)",
}

(
    ACCEPTED,
    UNMEASURABLE,
    TRUNCATED,
    JUNCTION,
    RUNAWAY,
    CEILING,
    UNSTABLE,
    BRANCH_OVERLAP,
    INTERPOLATED_INPUT,
) = range(9)
REJECT_NAMES = {
    ACCEPTED: "accepted",
    UNMEASURABLE: "unmeasurable",
    TRUNCATED: "truncated",
    JUNCTION: "junction",
    RUNAWAY: "runaway/local outlier",
    CEILING: "local factor",
    UNSTABLE: "unstable tangent/cross-section",
    BRANCH_OVERLAP: "unresolved branch overlap",
    INTERPOLATED_INPUT: "interpolated by Avizo (not measured)",
}

(
    DIRECT_PLANE,
    OWNED_PLANE,
    INTERPOLATED,
    BIF_PARENT,
    BIF_DAUGHTER,
    INPUT_FALLBACK,
    BIF_CONTINUATION,
) = range(7)
RESOLUTION_NAMES = {
    DIRECT_PLANE: "direct plane",
    OWNED_PLANE: "3D branch-owned plane",
    INTERPOLATED: "ordinary interpolation",
    BIF_PARENT: "parent-through taper",
    BIF_DAUGHTER: "daughter-emergence taper",
    INPUT_FALLBACK: "retained input fallback",
    BIF_CONTINUATION: "through-junction continuation",
}


@dataclass
class RadiusResult:
    """New radii for every point, in point order per segment, and their provenance."""

    radii: dict = field(default_factory=dict)  # {sid: (N,) um}
    source: dict = field(default_factory=dict)  # {sid: (N,) int8, see SOURCE_NAMES}
    reject_reason: dict = field(default_factory=dict)  # {sid: (N,) int8, see REJECT_NAMES}
    resolution_mode: dict = field(default_factory=dict)  # see RESOLUTION_NAMES
    section_rejection_counts: dict = field(default_factory=dict)
    section_target_obliquity_degrees: dict = field(default_factory=dict)
    n_measured: int = 0
    n_filled: int = 0
    n_truncated: int = 0
    n_ownership_failed: int = 0
    junction_lengths_um: list[float] = field(default_factory=list)
    fallback_segments: list[int] = field(default_factory=list)
    ratio: np.ndarray = field(default_factory=lambda: np.zeros(0))
    before_radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    after_radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seconds: float = 0.0
    # Median |r - r_prev| / r_prev for each measurement pass after the first.
    # Near zero means the geometry scales taken from the stored radii were good
    # enough and one pass would have done; a large value means the stored radii
    # were distorting the measurement geometry and the feedback mattered.
    pass_movement: list[float] = field(default_factory=list)

    def counts(self) -> dict:
        out = {k: 0 for k in SOURCE_NAMES}
        for arr in self.source.values():
            for k in SOURCE_NAMES:
                out[k] += int((arr == k).sum())
        return out

    def rejection_counts(self) -> dict:
        out = {k: 0 for k in REJECT_NAMES}
        for arr in self.reject_reason.values():
            for k in REJECT_NAMES:
                out[k] += int((arr == k).sum())
        return out

    def resolution_counts(self) -> dict:
        out = {k: 0 for k in RESOLUTION_NAMES}
        for arr in self.resolution_mode.values():
            for k in RESOLUTION_NAMES:
                out[k] += int((arr == k).sum())
        return out

    def describe(self) -> str:
        c = self.counts()
        total = sum(c.values()) or 1
        parts = ", ".join(
            f"{SOURCE_NAMES[k]} {c[k]:,} ({100.0 * c[k] / total:.1f}%)" for k in SOURCE_NAMES
        )
        line = f"{total:,} point radii: {parts}"
        modes = self.resolution_counts()
        if sum(modes.values()):
            line += "\n  resolution: " + ", ".join(
                f"{RESOLUTION_NAMES[k]} {modes[k]:,}" for k in RESOLUTION_NAMES
            )
        rejected = self.rejection_counts()
        line += "\n  confidence: " + ", ".join(
            f"{REJECT_NAMES[k]} {rejected[k]:,}" for k in REJECT_NAMES
        )
        if len(self.ratio):
            q = np.percentile(self.ratio, [5, 50, 95])
            line += (
                f"\n  r_new / r_old  p5 {q[0]:.2f}  median {q[1]:.2f}  p95 {q[2]:.2f}"
            )
        def distribution(name, values):
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values) & (values > 0)]
            if not len(values):
                return ""
            q = np.percentile(values, [50, 95, 99])
            return (f"\n  {name} radius (um) median {q[0]:.0f}  p95 {q[1]:.0f}  "
                    f"p99 {q[2]:.0f}  max {values.max():.0f}")

        line += distribution("before", self.before_radii)
        line += distribution("after", self.after_radii)
        if self.n_truncated:
            line += (
                f"\n  encountered {self.n_truncated} section(s) that never closed "
                f"inside the sampling window; they were rejected or subsequently "
                f"classified as junction points"
            )
        if self.junction_lengths_um:
            q = np.percentile(self.junction_lengths_um, [50, 95])
            line += (
                f"\n  adaptive junction runs {len(self.junction_lengths_um):,}: "
                f"median {q[0]:.0f} um, p95 {q[1]:.0f} um"
            )
        if self.n_ownership_failed:
            line += (
                f"\n  encountered {self.n_ownership_failed:,} unresolved 3D ownership "
                "sections before final junction classification"
            )
        if self.fallback_segments:
            line += (
                f"\n  retained input radii on {len(set(self.fallback_segments)):,} "
                "segment(s) with no trustworthy measurement"
            )
        return line + f"\n  ({self.seconds:.1f}s)"


def _fill_gaps(values: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate NaNs along one segment in log space; returns (filled, was_filled).

    Log space rather than linear for the reason :mod:`~.radius_repair` gives for its
    own taper fit: it is scale-free, so a trunk and a twig are treated alike, and an
    interpolated radius can never come out negative, which a linear fit across a long
    gap readily does.

    Runs at the very start or end of a segment are held constant at the nearest
    measured value rather than extrapolated. Extrapolating a taper off the end of the
    evidence is what ``radius_repair.MAX_TAPER_PER_MM`` exists to prevent, and there
    is no reason to invent it here where a real measurement is one point away.
    """
    values = np.asarray(values, dtype=np.float64).copy()
    bad = ~np.isfinite(values) | (values <= 0)
    if not bad.any():
        return values, bad
    good = ~bad
    if not good.any():
        # Nothing measurable on this whole segment: keep what the graph already had.
        return np.asarray(fallback, dtype=np.float64).copy(), bad
    idx = np.arange(len(values))
    # np.interp holds the end values constant outside the measured range, which is
    # exactly the "nearest, do not extrapolate" rule wanted at the segment ends.
    values[bad] = np.exp(np.interp(idx[bad], idx[good], np.log(values[good])))
    return values, bad


def _robust_high_correction_mask(
    arc: np.ndarray,
    measured: np.ndarray,
    old: np.ndarray,
    grew_too_far: np.ndarray,
    *,
    factor: float = RUNAWAY_FACTOR,
) -> np.ndarray:
    """Flag implausibly large correction ratios without letting highs fit themselves.

    The first log-linear fit estimates the segment trend.  The second fit uses only
    the lower residual half, so a cluster of inflated sections cannot become its own
    baseline.  A high ratio alone is not enough: the sampler must also have grown
    beyond four input radii, evidence that it was chasing a complex component.
    """
    arc = np.asarray(arc, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    ratio = np.full(len(old), np.nan)
    ok = np.isfinite(measured) & (measured > 0) & np.isfinite(old) & (old > 0)
    ratio[ok] = measured[ok] / old[ok]
    if ok.sum() < 4:
        return np.zeros(len(old), dtype=bool)

    from .radius_repair import fit_log_taper

    slope, intercept, _ = fit_log_taper(arc[ok], ratio[ok])
    residual = np.full(len(old), np.nan)
    residual[ok] = np.log(ratio[ok]) - (slope * arc[ok] + intercept)
    keep = ok & (residual <= np.median(residual[ok]))
    if keep.sum() >= 4:
        slope, intercept, _ = fit_log_taper(arc[keep], ratio[keep])
    trend = np.exp(slope * arc + intercept)
    return ok & grew_too_far & (ratio > float(factor) * trend)


def _robust_local_high_mask(
    arc: np.ndarray,
    measured: np.ndarray,
    old: np.ndarray,
    *,
    factor: float = LOCAL_RADIUS_FACTOR,
    input_calibration: float = 1.0,
) -> np.ndarray:
    """Flag radii above a robust, spatially local surrounding-radius estimate.

    The neighbourhood width is eight local input radii (and at least six point
    spacings). A median is used rather than a mean so one large contour cannot pull
    its own ceiling upward. With at least three sections, use the measurements as
    the baseline: capping them by the input radius rejects real sustained corrections
    on collapsed vessels. Sparse measurements retain the conservative input check.
    The estimate is repeated without first-pass highs.
    """
    arc = np.asarray(arc, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    ok = np.isfinite(measured) & (measured > 0)
    if ok.sum() < 3:
        valid_old = old[np.isfinite(old) & (old > 0)]
        if not len(valid_old):
            return np.zeros(len(measured), dtype=bool)
        baseline = float(np.median(valid_old)) * float(input_calibration)
        return ok & (measured > float(factor) * baseline)
    step = float(np.median(np.diff(arc)[np.diff(arc) > 0])) if np.any(np.diff(arc) > 0) else 1.0

    def baseline(trusted: np.ndarray) -> np.ndarray:
        out = np.full(len(measured), np.nan)
        for i in range(len(measured)):
            lo, hi = max(0, i - 8), min(len(old), i + 9)
            local_old = old[lo:hi]
            local_old = local_old[np.isfinite(local_old) & (local_old > 0)]
            scale = float(np.median(local_old)) if len(local_old) else step
            width = max(8.0 * scale, 6.0 * step)
            use = trusted & (np.abs(arc - arc[i]) <= width)
            if use.sum() >= 3:
                out[i] = float(np.median(measured[use]))
        # At a sparse endpoint, use the segment's robust lower-half trend rather
        # than silently disabling the gate.
        missing = ~np.isfinite(out)
        if missing.any():
            from .radius_repair import fit_log_taper

            slope, intercept, _ = fit_log_taper(arc[trusted], measured[trusted])
            residual = np.log(measured[trusted]) - (slope * arc[trusted] + intercept)
            ids = np.flatnonzero(trusted)
            lower = ids[residual <= np.median(residual)]
            if len(lower) >= 3:
                slope, intercept, _ = fit_log_taper(arc[lower], measured[lower])
            out[missing] = np.exp(slope * arc[missing] + intercept)
        return out

    first = baseline(ok)
    first_high = ok & (measured > float(factor) * first)
    trusted = ok & ~first_high
    second = baseline(trusted) if trusted.sum() >= 3 else first
    return ok & (measured > float(factor) * second)


def _grow_to(grow_radii: float | None, radius_vox: float) -> int | None:
    """Window-growth ceiling for one sample, in voxels. `None` means `max_half`."""
    if grow_radii is None:
        return None
    return int(float(grow_radii) * float(radius_vox)) + 2


def _ownership_seeds(in_blob, near_junctions: bool) -> list[int]:
    """Which rivals get a watershed marker, given the ones found inside the blob.

    **Every branch in the blob, not only the foreign ones.** `_resolve_owned_cut`
    seeds from ``[sid] + rival_sids``, and the watershed assigns every unseeded voxel
    to whichever marker is nearest -- so an adjacent branch that is fused here but
    left unseeded has its territory handed to *this* vessel, and the cut that was
    supposed to remove a merge adds one. Seeding it gives it somewhere of its own to
    go. It is still not a *trigger*: a neighbour sharing a node is what a junction is,
    and only a foreign centreline is a reason to re-cut at all.

    `near_junctions=False` restores the behaviour this replaced, where the whole
    ownership path was skipped whenever any adjacent branch was near; the seed list
    is then the foreign ones alone, because no adjacent branch can be present.
    """
    if near_junctions:
        return [int(item[0]) for item in in_blob]
    return [int(item[0]) for item in in_blob if not item[4]]


def _junction_scales(graph, spacing_um: float) -> dict[int, float]:
    """Input-radius scale of each true branch node (degree three or greater)."""
    scales: dict[int, float] = {}
    for seg in graph.segments:
        sid = seg["id"]
        rad = graph.radii(sid)
        if not len(rad):
            continue
        for nid, end in ((seg["node1"], rad[:3]), (seg["node2"], rad[-3:])):
            if graph.degree(nid) < 3:
                continue
            valid = end[np.isfinite(end) & (end > 0)]
            value = float(np.median(valid)) if len(valid) else float(spacing_um)
            scales[nid] = max(scales.get(nid, float(spacing_um)), value)
    return scales


@dataclass
class _BranchContext:
    """Spatial index over edge-owned centreline samples."""

    coords: np.ndarray
    radii: np.ndarray
    sids: np.ndarray
    local_ids: np.ndarray
    tree: object
    adjacency: dict[int, set[int]]
    max_radius: float

    @classmethod
    def build(cls, graph) -> "_BranchContext":
        from scipy.spatial import cKDTree

        parts_c, parts_r, parts_s, parts_i = [], [], [], []
        adjacency = {sid: set() for sid in graph.segment_ids()}
        for nid in graph.nodes:
            incident = list(graph.node_segments(nid))
            for sid in incident:
                adjacency.setdefault(sid, set()).update(x for x in incident if x != sid)
        for sid in graph.segment_ids():
            c = graph.coords(sid)
            r = graph.radii(sid)
            parts_c.append(c)
            parts_r.append(r)
            parts_s.append(np.full(len(c), sid, dtype=np.int64))
            parts_i.append(np.arange(len(c), dtype=np.int64))
        coords = np.vstack(parts_c) if parts_c else np.empty((0, 3))
        radii = np.concatenate(parts_r) if parts_r else np.empty(0)
        valid_r = radii[np.isfinite(radii) & (radii > 0)]
        max_radius = float(np.percentile(valid_r, 99.5)) if len(valid_r) else 1.0
        return cls(
            coords,
            radii,
            np.concatenate(parts_s) if parts_s else np.empty(0, dtype=np.int64),
            np.concatenate(parts_i) if parts_i else np.empty(0, dtype=np.int64),
            cKDTree(coords) if len(coords) else None,
            adjacency,
            max_radius,
        )

    def rivals(self, sid: int, point, tangent, radius: float) -> list[tuple]:
        """Closest cross-section-intersecting sample from every rival edge."""
        if self.tree is None:
            return []
        p = np.asarray(point, dtype=np.float64)
        t = np.asarray(tangent, dtype=np.float64)
        nt = float(np.linalg.norm(t))
        if nt < 1e-9:
            return []
        t /= nt
        search = max(float(radius), 1.0) + max(self.max_radius, 1.0)
        ids = self.tree.query_ball_point(p, search)
        best: dict[int, tuple] = {}
        for flat in ids:
            other = int(self.sids[flat])
            if other == sid:
                continue
            rr = float(self.radii[flat])
            if not np.isfinite(rr) or rr <= 0:
                continue
            diff = self.coords[flat] - p
            axial = abs(float(np.dot(diff, t)))
            transverse = float(np.linalg.norm(diff - np.dot(diff, t) * t))
            if axial > rr or transverse > float(radius) + rr:
                continue
            score = transverse + axial
            old = best.get(other)
            item = (
                other,
                int(self.local_ids[flat]),
                self.coords[flat],
                rr,
                other in self.adjacency.get(sid, ()),
                score,
            )
            if old is None or score < old[-1]:
                best[other] = item
        return list(best.values())


def _rival_lies_in_blob(cut, point_um, rival_um, spacing_um: float) -> bool:
    diff_vox = (np.asarray(rival_um) - np.asarray(point_um)) / float(spacing_um)
    row = int(np.rint(np.dot(diff_vox, cut.u))) + cut.half
    col = int(np.rint(np.dot(diff_vox, cut.v))) + cut.half
    return (
        0 <= row < cut.blob8.shape[0]
        and 0 <= col < cut.blob8.shape[1]
        and bool(cut.blob8[row, col])
    )


def _adaptive_junction_mask(
    graph,
    sid: int,
    arc: np.ndarray,
    stable: np.ndarray,
    adjacent_overlap: np.ndarray,
    max_fraction: float = JUNCTION_MASK_MAX_FRACTION,
    runs_by_node: dict | None = None,
    node_limits: dict | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Contiguous endpoint runs ending at two stable, exclusive sections.

    Bounded two ways per end, whichever is tighter: `max_fraction` of the segment's
    own arclength -- see :data:`JUNCTION_MASK_MAX_FRACTION` for why that exists and
    what it costs -- and `node_limits`, an absolute arclength per node.

    **The fraction alone is the wrong shape of bound, and on a short segment it is
    the only one that binds.** A junction's influence is a property of the *junction*
    -- it reaches a couple of parent radii into each branch, which is what
    :data:`JUNCTION_RADII` says and what the non-branch-aware path has always used.
    A fraction of the segment instead measures how long the segment happens to be, so
    the same node consumes 40% of a 100-radius branch and 40% of a 6-radius one. Since
    the exclusivity test is unsatisfiable near a junction the walk always runs to its
    bound, and a short segment between two branch nodes therefore loses 80% of itself
    to a region that is only a few radii wide. Measured on LADAF-2021-17: 1,169 of
    6,106 segments were at least 75% masked, and 2,168 ended with no measurement
    anywhere, their radii falling back to the uncorrected input.
    """
    n = len(arc)
    mask = np.zeros(n, dtype=bool)
    lengths: list[float] = []
    if n == 0:
        return mask, lengths
    total = float(arc[-1]) if n > 1 else 0.0
    limit = (
        max_fraction * total
        if (max_fraction is not None and np.isfinite(max_fraction) and total > 0)
        else np.inf
    )
    seg = graph.segment(sid)
    for nid, reverse in ((seg["node1"], False), (seg["node2"], True)):
        if graph.degree(nid) < 3:
            continue
        node_limit = float((node_limits or {}).get(int(nid), np.inf))
        limit_here = min(limit, node_limit) if np.isfinite(node_limit) else limit
        order = np.arange(n - 1, -1, -1) if reverse else np.arange(n)
        endpoint_arc = arc[-1] if reverse else arc[0]
        pending: list[int] = []
        committed: list[int] = [int(order[0])]
        consecutive = 0
        for raw_idx in order[1:]:
            idx = int(raw_idx)
            if abs(float(arc[idx]) - float(endpoint_arc)) > limit_here:
                # The cap, and it deliberately drops `pending` rather than
                # committing it: those points were exclusive but unconfirmed, and
                # past the cap the benefit of the doubt goes to measuring them.
                break
            exclusive = bool(stable[idx] and not adjacent_overlap[idx])
            if exclusive:
                pending.append(idx)
                consecutive += 1
                if consecutive >= 2:
                    break
            else:
                committed.extend(pending)
                pending.clear()
                committed.append(idx)
                consecutive = 0
        else:
            committed.extend(pending)
        mask[np.asarray(committed, dtype=int)] = True
        if runs_by_node is not None:
            runs_by_node.setdefault(int(nid), []).extend(int(i) for i in committed)
        if committed:
            lengths.append(float(max(abs(arc[i] - endpoint_arc) for i in committed)))
    return mask, lengths


def _raster_markers(mask: np.ndarray, origin: np.ndarray, branches: list[np.ndarray]):
    """Rasterize branch centreline samples as watershed seeds; None if unusable.

    **A seed has to be unambiguous, and a contested voxel is not a failure.** Two
    branches that share a node share that node's coordinates exactly, so their
    centrelines land on the same voxel there and any single-pass rasteriser sees a
    collision. Treating that as fatal is what confined this whole mechanism to
    branches that merely *touch*, and excluded every bifurcation -- the one case
    where lumens are guaranteed to be fused and ownership actually needs deciding.

    So the pass is split in two: count which branches claim each voxel, then seed
    only the voxels claimed by exactly one. A contested voxel is left unseeded and
    the watershed assigns it like any other, which is the right treatment -- it is
    contested territory, not a contradiction. A branch left with no seed at all is
    still fatal, because it would silently forfeit its lumen to a sibling.
    """
    nz, ny, nx = mask.shape
    claims: dict[tuple[int, int, int], int] = {}
    contested: set[tuple[int, int, int]] = set()
    per_branch: list[list[tuple[int, int, int]]] = []
    for label, coords in enumerate(branches, 1):
        local = np.rint(np.asarray(coords) - origin[None, :]).astype(int)
        mine: list[tuple[int, int, int]] = []
        for i, j, k in local:
            if not (0 <= k < nz and 0 <= j < ny and 0 <= i < nx):
                continue
            if not mask[k, j, i]:
                continue
            key = (int(k), int(j), int(i))
            mine.append(key)
            if claims.setdefault(key, label) != label:
                contested.add(key)
        per_branch.append(mine)

    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[~mask] = -1
    for label, mine in enumerate(per_branch, 1):
        placed = 0
        for key in mine:
            if key in contested:
                continue
            markers[key] = label
            placed += 1
        if placed == 0:
            return None
    return markers


def _owned_plane_cut(labels_plane: np.ndarray, half: int, u, v):
    from scipy import ndimage
    from ..crosssection import PlaneCut

    if labels_plane[half, half] != 1:
        return None
    target = labels_plane == 1
    lab4, _ = ndimage.label(target)
    blob4 = lab4 == lab4[half, half]
    lab8, n8 = ndimage.label(target, structure=np.ones((3, 3), dtype=int))
    blob8 = lab8 == lab8[half, half]
    touches = bool(
        blob8[0].any() or blob8[-1].any() or blob8[:, 0].any() or blob8[:, -1].any()
    )
    return PlaneCut(
        (labels_plane > 0).astype(np.uint8), blob4, blob8, int(n8),
        np.asarray(u), np.asarray(v), int(half), touches, False,
    )


def _ownership_volume(sampler, coords_ijk, sid, rival_sids, centre_ijk, extent, stable_ownership):
    from scipy import ndimage
    lo = np.asarray(centre_ijk) - extent
    hi = np.asarray(centre_ijk) + extent + 1
    roi, origin = sampler.box(lo, hi)
    if roi.size == 0 or not roi.any():
        return None
    branch_ids = [sid] + sorted(set(int(x) for x in rival_sids))
    branch_coords = [_roi_polyline_samples(coords_ijk[x], origin, roi.shape)
                     for x in branch_ids if x in coords_ijk]
    if len(branch_coords) != len(branch_ids):
        return None
    markers = _raster_markers(roi.astype(bool), origin, branch_coords)
    if markers is None:
        return None
    positive = sorted(set(int(x) for x in np.unique(markers) if x > 0))
    if positive != list(range(1, len(branch_ids) + 1)):
        return None

    distance = ndimage.distance_transform_edt(roi > 0)
    if stable_ownership:
        from skimage.segmentation import watershed
        # Float distance preserves narrow saddles. FIFO plateau handling prevents
        # a seed from claiming the far vessel's outer shell on quantized ties.
        owned = watershed(-distance, np.maximum(markers, 0), mask=roi > 0, connectivity=1)
    else:
        inv = np.zeros_like(distance)
        inv[roi > 0] = 1.0 / (distance[roi > 0] + 0.5)
        vmax = float(inv.max()) or 1.0
        elevation = np.rint(254.0 * inv / vmax).astype(np.uint8)
        elevation[roi == 0] = 255
        owned = ndimage.watershed_ift(
            elevation, markers, structure=ndimage.generate_binary_structure(3, 1)
        )
    if np.any(owned[markers > 0] != markers[markers > 0]):
        return None
    owned[roi == 0] = 0
    return owned, origin


def _resolve_owned_slab(
    sampler,
    graph,
    coords_ijk: dict[int, np.ndarray],
    sid: int,
    rival_sids: list[int],
    centre_ijk: np.ndarray,
    tangent: np.ndarray,
    radius_vox: float,
    half: int,
    max_half: int,
    spacing_um: float,
    offsets=(-0.5, 0.0, 0.5),
    stable_ownership=True,
    volume_cache=None,
):
    """Separate non-adjacent touching branches in a local 3-D watershed ROI."""
    from ..crosssection import _plane_axes, sample_label_plane

    extent = min(int(max_half), max(int(4.0 * radius_vox) + 2, int(half) + 2))
    key = (tuple(np.asarray(centre_ijk)), extent, sid, tuple(sorted(rival_sids)), stable_ownership)
    volume = volume_cache.get(key) if volume_cache is not None else None
    if volume is None:
        volume = _ownership_volume(sampler, coords_ijk, sid, rival_sids, centre_ijk, extent, stable_ownership)
        if volume is None:
            return None
        if volume_cache is not None:
            # Bound memory to one ROI per station, shared across alternative
            # orientations with the same finite-branch ownership seeds.
            volume_cache.clear()
            volume_cache[key] = volume
    owned, origin = volume
    axes = _plane_axes(np.asarray(tangent, dtype=float))
    if axes is None:
        return None
    u, v = axes
    cuts = []
    for offset in offsets:
        centre = np.asarray(centre_ijk) + offset * float(radius_vox) * tangent
        plane = sample_label_plane(owned, origin, centre, u, v, half)
        c = _owned_plane_cut(plane, half, u, v)
        if c is None or c.touches_border:
            return None
        cuts.append(c)
    return cuts


def _resolve_owned_cut(sampler, graph, coords_ijk, sid, rival_sids, centre_ijk,
                       tangent, radius_vox, half, max_half, spacing_um):
    from ..crosssection import _perimeter_um
    cuts = _resolve_owned_slab(sampler, graph, coords_ijk, sid, rival_sids, centre_ijk,
                               tangent, radius_vox, half, max_half, spacing_um, stable_ownership=False)
    if cuts is None:
        return None
    areas = np.array([float(c.blob4.sum()) for c in cuts])
    perimeters = np.array([_perimeter_um(c.blob4, spacing_um) for c in cuts])
    if (np.any(areas <= 0) or np.any(perimeters <= 0)
            or areas.max()/areas.min() > 1.5 or perimeters.max()/perimeters.min() > 1.5):
        return None
    return cuts[1]


def _roi_polyline_samples(coords, origin, shape):
    """Seed finite edges, including edges with both stored points outside the ROI.

    Clip before voxel traversal so uneven sampling neither loses a branch nor
    rasterizes an entire long vessel for a small ownership volume.
    """
    from .centreline_refine import line_samples_ijk
    lower, upper = np.asarray(origin), np.asarray(origin)+np.asarray(shape[::-1])-1
    samples = []
    for a, b in zip(coords[:-1], coords[1:]):
        d = b-a
        lo, hi = 0., 1.
        for axis in range(3):
            if abs(d[axis]) < 1e-12:
                if a[axis] < lower[axis] or a[axis] > upper[axis]:
                    hi = -1.
                    break
            else:
                t = sorted(((lower[axis]-a[axis])/d[axis], (upper[axis]-a[axis])/d[axis]))
                lo, hi = max(lo, t[0]), min(hi, t[1])
        if lo <= hi:
            samples.append(line_samples_ijk(a+lo*d, a+hi*d))
    return np.vstack(samples) if samples else np.empty((0, 3))


def _directed_topology(graph, root_edges=()) -> tuple[dict[int, int], dict[int, int | None]]:
    """Root each component using explicit roots, Strahler, free ends and calibre."""
    roots = {int(x) for x in (root_edges or ())}
    known = set(graph.segment_ids())
    missing = sorted(roots - known)
    if missing:
        raise ValueError(f"unknown root edge(s): {', '.join(map(str, missing))}")
    adjacency = {sid: set() for sid in known}
    for nid in graph.nodes:
        incident = list(graph.node_segments(nid))
        for sid in incident:
            adjacency[sid].update(x for x in incident if x != sid)
    depth: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    for component in graph.components():
        forced = sorted(component & roots)
        if len(forced) > 1:
            raise ValueError(
                "multiple --root-edge values select the same component: "
                + ", ".join(map(str, forced))
            )

        def root_key(sid: int):
            seg = graph.segment(sid)
            strahler = int(seg.get("strahler", 0))
            has_free = any(graph.degree(seg[k]) == 1 for k in ("node1", "node2"))
            mean = seg.get("MeanRadius")
            if mean is None or not np.isfinite(mean) or float(mean) <= 0:
                vals = graph.radii(sid)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                mean = float(np.median(vals)) if len(vals) else 0.0
            return strahler, bool(has_free), float(mean), -int(sid)

        root = forced[0] if forced else max(component, key=root_key)
        depth[root], parent[root] = 0, None
        queue = [root]
        while queue:
            current = queue.pop(0)
            for other in sorted(adjacency[current]):
                if other not in component or other in depth:
                    continue
                depth[other] = depth[current] + 1
                parent[other] = current
                queue.append(other)
    return depth, parent


def _junction_parents(graph, root_edges=()) -> dict[int, int]:
    depth, _parents = _directed_topology(graph, root_edges)
    out: dict[int, int] = {}
    for nid in graph.nodes:
        incident = sorted(graph.node_segments(nid))
        if len(incident) < 3:
            continue
        levels = np.array([depth.get(sid, np.iinfo(np.int32).max) for sid in incident])
        best = int(levels.min())
        candidates = [sid for sid, level in zip(incident, levels) if int(level) == best]
        if len(candidates) == 1:
            out[nid] = candidates[0]
    return out


def _endpoint_order(graph, sid: int, nid: int, n: int) -> np.ndarray:
    seg = graph.segment(sid)
    if seg["node1"] == nid:
        return np.arange(n)
    if seg["node2"] == nid:
        return np.arange(n - 1, -1, -1)
    return np.empty(0, dtype=int)


def _junction_anchor_radii(graph, sid: int, measured: dict,
                           exclude: set[int] | None = None) -> list[float]:
    """Measured radii of the neighbouring segments at each end of ``sid``.

    Used when a segment has no trustworthy cross-section of its own: rather than
    keeping an uncorrected input calibre, the span is interpolated between what
    its neighbours were actually measured at. Returns ``[]`` when no neighbour
    has a usable radius, so the caller can fall back again.

    ``exclude`` names the segments that are themselves un-measured. Without it a
    neighbour holding nothing but retained input would anchor the interpolation,
    which is the calibre this policy exists to refuse -- the spike would simply
    move one segment along.
    """
    out: list[float] = []
    seg = graph.segment(sid)
    skip = exclude or ()
    for nid in (seg["node1"], seg["node2"]):
        vals: list[float] = []
        for other in graph.node_segments(nid):
            if other == sid or other in skip:
                continue
            arr = measured.get(other)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if len(arr):
                vals.append(float(np.median(arr)))
        if vals:
            out.append(float(np.median(vals)))
    if len(out) == 1:
        out = [out[0], out[0]]
    return out


def _apply_bifurcation_tapers(
    graph,
    measured: dict[int, np.ndarray],
    source: dict[int, np.ndarray],
    reject: dict[int, np.ndarray],
    modes: dict[int, np.ndarray],
    arcs: dict[int, np.ndarray],
    *,
    spacing_um: float,
    root_edges=(),
    carina_tip_factor: float = 0.1,
    continuation_ratio: float = CONTINUATION_RATIO,
    daughter_carina: bool = False,
) -> set[int]:
    """Author parent-through and, optionally, daughter-emergence junction profiles.

    The two halves are separable and are separated here because only one of them is
    an extrapolation of something measured. Carrying a parent's own log-linear trend
    across the span where its sections were refused says nothing the vessel did not
    already say on both sides of the node. Narrowing a daughter to a carina tip is a
    model of junction anatomy, and one whose sign is wrong for an ostium.
    """
    if not np.isfinite(carina_tip_factor) or not (0 < carina_tip_factor < 1):
        raise ValueError("carina_tip_factor must be between zero and one")
    if not np.isfinite(continuation_ratio) or not (0 < continuation_ratio <= 1):
        raise ValueError("continuation_ratio must be in (0, 1]")
    parents = _junction_parents(graph, root_edges)
    fallback: set[int] = set()
    for nid in graph.nodes:
        if graph.degree(nid) < 3:
            continue
        if any(sid not in measured for sid in graph.node_segments(nid)):
            # Regional measurement retains the full graph as spatial context.
            # A profile requires measurements from every incident branch.
            continue
        parent_sid = parents.get(nid)
        if parent_sid is None:
            for sid in graph.node_segments(nid):
                values = measured[sid]
                order = _endpoint_order(graph, sid, nid, len(values))
                run = []
                for idx in order:
                    if reject[sid][idx] != JUNCTION:
                        break
                    run.append(int(idx))
                if run:
                    values[run] = graph.radii(sid)[run]
                    source[sid][run] = FILLED
                    modes[sid][run] = INPUT_FALLBACK
                fallback.add(sid)
            continue
        # Calibre of the parent at this node, used below to tell an ostium from a
        # continuation. Taken from the parent's own trusted radii rather than its
        # junction run, which is exactly the region being authored.
        parent_calibre = np.nan
        if parent_sid is not None:
            pv_ = measured[parent_sid]
            pr_ = reject[parent_sid]
            ok_ = np.isfinite(pv_) & (pv_ > 0) & (pr_ == ACCEPTED)
            if ok_.any():
                parent_calibre = float(np.median(pv_[ok_]))
            else:
                raw_ = graph.radii(parent_sid)
                raw_ = raw_[np.isfinite(raw_) & (raw_ > 0)]
                if len(raw_):
                    parent_calibre = float(np.median(raw_))

        for sid in graph.node_segments(nid):
            values = measured[sid]
            n = len(values)
            order = _endpoint_order(graph, sid, nid, n)
            if not len(order):
                continue
            run = []
            for idx in order:
                if reject[sid][idx] != JUNCTION:
                    break
                run.append(int(idx))
            if not run:
                continue
            after = order[len(run) :]
            trusted = [
                int(i) for i in after
                if reject[sid][i] == ACCEPTED
                and np.isfinite(values[i]) and values[i] > 0
            ]
            if not trusted:
                values[run] = graph.radii(sid)[run]
                source[sid][run] = FILLED
                modes[sid][run] = INPUT_FALLBACK
                fallback.add(sid)
                continue
            boundary = trusted[0]
            arc = arcs[sid]
            node_arc = float(arc[order[0]])
            boundary_distance = abs(float(arc[boundary]) - node_arc)
            if boundary_distance <= 1e-9:
                values[run] = graph.radii(sid)[run]
                source[sid][run] = FILLED
                modes[sid][run] = INPUT_FALLBACK
                fallback.add(sid)
                continue
            # A carina taper models an OSTIUM: a branch emerging through the wall
            # of a distinctly larger parent, whose lumen genuinely narrows to
            # nothing at the carina. Topological depth alone does not identify
            # one. At a bifurcation where the main vessel continues and sheds a
            # side branch, the continuation is also a "daughter" by depth, and
            # tapering it to `carina_tip_factor` collapses a vessel that never
            # narrowed -- the visible artefact of a main vessel pinching at every
            # point its segments meet.
            #
            # Measured on LADAF_2024_28: of 307 tapered daughters, only 9.1% are
            # below 0.30 of their parent's calibre, while 26.1% sit at 0.70-0.90
            # and a further 23.1% at >= 0.90 -- the same vessel. Half of all
            # carina tapers were firing on continuations.
            #
            # So a daughter of comparable calibre is treated as the vessel
            # continuing: it gets the same log-linear trend extrapolation the
            # parent gets, which carries its own radius through the junction
            # instead of pinching it.
            is_continuation = (
                sid != parent_sid
                and np.isfinite(parent_calibre) and parent_calibre > 0
                and float(values[boundary]) >= float(continuation_ratio) * parent_calibre
            )
            if sid == parent_sid or is_continuation:
                healthy = np.asarray(trusted[: min(8, len(trusted))], dtype=int)
                x = np.abs(arc[healthy] - node_arc)
                y = np.log(values[healthy])
                if len(healthy) >= 2 and np.ptp(x) > 1e-9:
                    slope, intercept = np.polyfit(x, y, 1)
                    pred = np.exp(slope * np.abs(arc[run] - node_arc) + intercept)
                else:
                    pred = np.full(len(run), float(values[boundary]))
                lo = float(values[healthy].min()) / 1.05
                hi = float(values[healthy].max()) * 1.05
                values[run] = np.clip(pred, lo, hi)
                modes[sid][run] = BIF_PARENT if sid == parent_sid else BIF_CONTINUATION
            elif daughter_carina:
                anchor = float(values[boundary])
                tip = max(float(carina_tip_factor) * anchor, 0.5 * float(spacing_um))
                distance = np.abs(arc[run] - node_arc)
                t = np.clip(distance / boundary_distance, 0.0, 1.0)
                smooth = t * t * (3.0 - 2.0 * t)
                values[run] = np.exp((1.0 - smooth) * np.log(tip) + smooth * np.log(anchor))
                modes[sid][run] = BIF_DAUGHTER
            else:
                # An ostial daughter, with the carina model switched off. Leave the
                # run untouched so `_fill_gaps` interpolates it and labels it
                # `INTERPOLATED`. The taper this would otherwise apply narrows the
                # daughter to `carina_tip_factor` **at** the node and widens it
                # distally -- the opposite of an ostium, which is widest where it
                # meets the parent. Authoring the parent's trend is an
                # extrapolation of that vessel's own measurements; authoring a
                # daughter pinch is a claim about carina geometry that nothing here
                # has observed.
                continue
            source[sid][run] = FILLED
    return fallback


def measure_radii(
    graph,
    frame,
    labels,
    *,
    gate_voxels: float = GATE_VOXELS,
    max_half: int = 128,
    max_radius_factor: float = LOCAL_RADIUS_FACTOR,
    min_blob_voxels: int | None = None,
    branch_aware: bool = True,
    root_edges=(),
    tangent_search_degrees: float = 20.0,
    #: How elliptical a section may be before the tangent search runs at all --
    #: `crosssection.TRANSVERSE_AXIS_RATIO`, and the reason it is not simply the
    #: stability test. Raising it towards infinity restores the pre-fix behaviour
    #: of trusting the fitted tangent whenever its section was stable, which is
    #: cheaper and over-reads every oblique cut by up to a quarter.
    transverse_axis_ratio: float = TRANSVERSE_AXIS_RATIO,
    carina_tip_factor: float = 0.1,
    continuation_ratio: float = CONTINUATION_RATIO,
    bifurcation_tapers: bool = False,
    # Carry a parent's -- and a same-calibre continuation's -- measured trend across
    # its junction run, instead of holding the last measured value flat. The
    # exclusivity test `_adaptive_junction_mask` uses is unsatisfiable near a
    # junction (measured: adjacent rivals of comparable calibre are present at
    # essentially every junction point, so no section is ever exclusive and the walk
    # only ever stops at its own bound), so a through-vessel cannot be *measured*
    # across a node however the gates are tuned. Extrapolating its own trend is the
    # weakest available substitute: it is a statement about one vessel, made from
    # that vessel's own accepted sections either side. It authors nothing for
    # ostial daughters -- that stays with `bifurcation_tapers`, still off.
    junction_parent_profile: bool = True,
    #: What to do with the measured section inside a junction run. Approaching a
    #: branched node the section genuinely widens -- measured over 261 segments,
    #: each normalised by its own interior, `perimeter/2pi` runs 1.32x at under half
    #: a radius from the node, 1.21x, 1.11x, 1.07x, 1.05x by three radii, with the
    #: aspect ratio up only 1.13x, so most of the flare is round and a radius can
    #: carry it. That is an ostium, and `none` discards it.
    #:
    #: It is discarded because the section is wider *for a reason*: at the node the
    #: lumens are continuous, so each branch measures the shared region. Keep it on
    #: every branch and an N-way junction contributes that region N times to the
    #: union, which is the junction bulge `coronary_sdf`'s flat-cap machinery exists
    #: to suppress. Keeping it on the parent alone attributes it once.
    #:
    #: `none` -- discard, interpolate the run (previous behaviour).
    #: `parent` -- the parent at that node keeps its measurements; daughters do not.
    #: `all` -- every branch keeps its own, and the shared lumen is counted N times.
    junction_flare: str = "none",
    # On, unlike `bifurcation_tapers` and `fallback_taper`, and for the reason those
    # are off: this is not a model of an unobserved vessel, it is the inverse of a
    # measured, closed-form property of the estimator, checked against known
    # geometry. Leaving it off would mean knowingly writing out radii that are 10-25%
    # small at the thin end. Off restores the raw reading.
    perimeter_correction: bool = True,
    junction_mask_max_fraction: float | None = JUNCTION_MASK_MAX_FRACTION,
    #: Second, absolute ceiling on a junction run, in multiples of the node's own
    #: input radius -- the bound the non-branch-aware path has always used, applied
    #: alongside the fraction so the tighter of the two wins. See
    #: :func:`_adaptive_junction_mask`.
    #:
    #: **Off by default, because it was measured and it did not help.** On
    #: LADAF-2021-17 it cuts the mask's reach from a median 50% of a segment to
    #: 35% and frees a third of all points, which looks like a large win until the
    #: freed points are actually cut: over segments 3612, 3655 and eight of their
    #: neighbours -- 1,371 points, none of which the pass could measure -- every
    #: point released from the mask came back `truncated` or `unstable` instead,
    #: and the accepted count stayed at zero at 2.0 radii and at 1.0. The mask was
    #: covering points that were already unmeasurable, so shrinking it only
    #: relabels them. It is kept because the bound is the right *shape* -- a
    #: junction reaches a few parent radii into a branch wherever that branch
    #: ends, which a fraction of the segment does not express -- and somewhere
    #: with measurable vessels either side of a node it may pay. It is off
    #: because nothing here has shown that it does.
    junction_mask_radii: float | None = None,
    #: Ceiling on how far one sample's window may double, in multiples of that
    #: point's own radius. `None` lets it run to `max_half`, which is what it has
    #: always done. See :func:`~..crosssection.stable_transverse_cut`: the growth
    #: is unbounded relative to the vessel, so a plane that is not transverse
    #: escalates to `max_half` and is rejected anyway. This is what makes a large
    #: `max_half` affordable, and `max_half` is what the widest vessels need.
    grow_radii: float | None = None,
    #: The two gates `stable_transverse_cut` applies to its three-plane slab:
    #: Area/perimeter variation and centroid motion across the slab. A fixed
    #: off-centre centreline does not invalidate a perimeter measurement. The old
    #: absolute-offset check is available explicitly for comparison.
    #: Attempt branch ownership where lumens actually fuse.
    #:
    #: `adjacent_overlap[i]` is `any(item[4] for item in rivals)` -- a whole-point
    #: flag. It used to sit *inside* both `connected_nonadjacent` comprehensions,
    #: where it does not depend on the loop variable, so it emptied the list whenever
    #: any topologically-adjacent branch was merely nearby. Near a junction one always
    #: is (this module says so at `JUNCTION_MASK_MAX_FRACTION`), so the 3-D watershed
    #: never ran and `BRANCH_OVERLAP` was never raised at exactly the places lumens
    #: merge: the merged blob was measured and marked `ACCEPTED`. On LADAF-2021-17
    #: `OWNED_PLANE` resolved ~100 of 336,865 points.
    #:
    #: `False` restores that. See :func:`_ownership_seeds` for why turning it on also
    #: has to widen the marker set rather than only the trigger.
    ownership_near_junctions: bool = True,
    stability_variation: float = 1.5,
    stability_centroid_radii: float = 0.5,
    stability_centroid_mode: str = "drift",
    fallback_policy: str = FALLBACK_POLICY,
    # Off for the same reason `bifurcation_tapers` is off: a taper across an
    # un-measured span is a model, not a measurement. See the second pass below.
    fallback_taper: bool = False,
    n_passes: int = 1,
    workers: int = 1,
    progress=None,
    section_filter: bool = False,
    _segment_ids=None,
    _raw_only: bool = False,
) -> RadiusResult:
    """Measure the radius of every centreline point's own cross-section.

    In branch-aware mode an edge-local fitted tangent is first validated on a short
    three-plane slab. The 4-connected component containing the plane centre is measured
    when exclusive; connected non-adjacent rivals are separated in a local 3-D marker
    watershed. Topology-adjacent shared lumen is excluded and receives an authored
    parent-through/daughter-emergence profile after all trusted sections are known.

    Points whose section cannot be measured (the centreline is outside the mask, or
    the section is smaller than the blob floor) are interpolated from their measured
    neighbours along the same segment and marked ``FILLED``.

    `labels` should be the **full-resolution** lattice even when the skeleton came from
    a strided run: the radius is the quantity being corrected, and measuring it on a
    decimated mask would give back the error being removed. That is affordable because
    the sections are cut by streaming z-slices out of the RLE lattice rather than by
    decoding the volume.

    ``workers > 1`` distributes provisional measurements across processes. Each
    worker sees the full branch context; calibration, rejection, junction profiles
    and gap filling still run once on the combined measurements. This mode supports
    one measurement pass and array or Amira-lattice inputs, not live mask edits.
    """
    from ..crosssection import (
        MIN_BLOB_VOXELS,
        _PlaneSampler,
        _perimeter_um,
        cut,
        robust_edge_tangents,
        stable_transverse_cut,
    )
    from . import skeleton_optimise

    t0 = time.time()
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if section_filter and not branch_aware:
        raise ValueError('section_filter requires full branch-aware validation')
    if workers > 1 and int(n_passes) != 1:
        raise ValueError("parallel measurement currently requires n_passes=1")
    if min_blob_voxels is None:
        min_blob_voxels = MIN_BLOB_VOXELS

    if not np.isfinite(max_radius_factor) or max_radius_factor <= 1.0:
        raise ValueError("max_radius_factor must be finite and greater than one")
    if fallback_policy not in FALLBACK_POLICIES:
        raise ValueError(f"fallback_policy must be one of {FALLBACK_POLICIES}")
    if stability_centroid_mode not in ("offset", "drift"):
        raise ValueError("stability_centroid_mode must be 'offset' or 'drift'")
    if junction_flare not in JUNCTION_FLARE_MODES:
        raise ValueError(f"junction_flare must be one of {JUNCTION_FLARE_MODES}")
    junction_parents = (
        _junction_parents(graph, root_edges) if junction_flare == "parent" else {}
    )
    if not np.isfinite(tangent_search_degrees) or not (0 <= tangent_search_degrees <= 45):
        raise ValueError("tangent_search_degrees must be between zero and 45")

    result = RadiusResult()
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    dims = np.asarray(frame.seg_dims, dtype=np.float64)
    gate_um = float(gate_voxels) * sp
    ratios: list[float] = []
    before_all: list[np.ndarray] = []
    # Segments that yielded no trustworthy cross-section anywhere, resolved in a
    # second pass once every other segment's measurement exists.
    no_trusted: list[int] = []
    raw_measured: dict[int, np.ndarray] = {}
    raw_source: dict[int, np.ndarray] = {}
    raw_reject: dict[int, np.ndarray] = {}
    raw_grew: dict[int, np.ndarray] = {}
    raw_modes: dict[int, np.ndarray] = {}
    old_by_sid: dict[int, np.ndarray] = {}
    arc_by_sid: dict[int, np.ndarray] = {}
    invented_by_sid: dict[int, np.ndarray] = {}
    branch_context = _BranchContext.build(graph) if branch_aware else None
    coords_ijk = {sid: frame.um_to_seg(graph.coords(sid)) for sid in graph.segment_ids()}
    junction_scales = _junction_scales(graph, sp)

    sids = graph.segment_ids() if _segment_ids is None else list(_segment_ids)
    # Every geometric scale in the loop below -- the tangent-fit window, the cut
    # half-width, the stability slab offsets, the rival search and the runaway
    # guard -- was taken from the stored radius, which is the quantity this pass
    # exists to replace. As this module's docstring records, `r_stored/r_perimeter`
    # has a p5-p95 spread of 0.49-2.98, so wherever the stored radius is wrong the
    # measurement geometry is wrong in the same place: the window is too short to
    # fit a stable tangent, or too small to contain the section, and the
    # three-plane stability test compares planes too close together to
    # discriminate. Feeding pass 1's accepted measurements back as the scale
    # breaks that circularity. `old` stays the true original everywhere it is
    # used as a reporting baseline.
    #
    # Off by default (`n_passes=1`), because feeding a *larger* measured radius
    # back has a cost: it grows the cut window, and where two non-adjacent
    # vessels touch, the larger window swallows the neighbour and defeats the
    # marker-watershed ownership resolution -- exactly what
    # `test_nonadjacent_touching_vessels_use_local_3d_ownership` catches, which
    # drops from OWNED_PLANE to INPUT_FALLBACK at n_passes=2. Enable it
    # deliberately and read `RadiusResult.pass_movement` to see whether the
    # stored radii were distorting the measurement geometry on your data.
    n_passes = max(1, int(n_passes))
    scale_by_sid: dict[int, np.ndarray] = {}
    from .section_validation import SectionContext
    section_context = SectionContext(graph) if branch_aware and section_filter else None
    pass_medians: list[float] = []
    if workers > 1:
        from .radius_parallel import measure_provisional

        raw = measure_provisional(
            graph, frame, labels, sids, workers=workers, progress=progress,
            options=dict(
                gate_voxels=gate_voxels, max_half=max_half,
                section_filter=section_filter,
                min_blob_voxels=min_blob_voxels, branch_aware=branch_aware,
                root_edges=root_edges, tangent_search_degrees=tangent_search_degrees,
                transverse_axis_ratio=transverse_axis_ratio,
                junction_flare=junction_flare, perimeter_correction=perimeter_correction,
                junction_mask_max_fraction=junction_mask_max_fraction,
                junction_mask_radii=junction_mask_radii,
                grow_radii=grow_radii,
                stability_centroid_mode=stability_centroid_mode,
                ownership_near_junctions=ownership_near_junctions,
                stability_variation=stability_variation,
                stability_centroid_radii=stability_centroid_radii,
            ),
        )
        raw_measured, raw_source, raw_reject = raw["measured"], raw["source"], raw["reject"]
        raw_grew, raw_modes = raw["grew"], raw["modes"]
        old_by_sid, arc_by_sid = raw["old"], raw["arc"]
        invented_by_sid, before_all = raw["invented"], raw["before"]
        result = raw["result"]
    for _pass in range(n_passes if workers == 1 else 0):
        final_pass = _pass == n_passes - 1
        result.n_truncated = 0
        result.n_ownership_failed = 0
        result.junction_lengths_um.clear()
        prev_measured = {sid: arr.copy() for sid, arr in raw_measured.items()}
        for n_done, sid in enumerate(sids, 1):
            coords = graph.coords(sid)
            old = graph.radii(sid)
            # Geometry scale: pass 1 has only the stored radius; later passes use
            # what was actually measured, falling back to the stored value at points
            # that were rejected so a failure cannot propagate its scale.
            scale = scale_by_sid.get(sid)
            if scale is None or len(scale) != len(old):
                scale = old
            n = len(coords)
            measured = np.full(n, np.nan)
            source = np.full(n, FILLED, dtype=np.int8)
            reject = np.full(n, UNMEASURABLE, dtype=np.int8)
            modes = np.full(n, INTERPOLATED, dtype=np.int8)
            grew_too_far = np.zeros(n, dtype=bool)
            if n == 0:
                raw_measured[sid], raw_source[sid], raw_reject[sid] = measured, source, reject
                raw_grew[sid], old_by_sid[sid], arc_by_sid[sid] = grew_too_far, old, np.zeros(0)
                raw_modes[sid] = modes
                continue

            if final_pass:
                before_all.append(old.copy())

            # The same chord estimator re-centring uses, and for a sharper reason here: an
            # oblique plane cuts an *ellipse*, whose perimeter exceeds the true section's by
            # roughly 1/cos of the tilt. A tangent that swings a median 19.5 degrees between
            # neighbours -- what `np.gradient` gives on a thinned skeleton -- therefore
            # over-states `r = perimeter / 2pi` by a few percent, systematically, on the one
            # number this whole pass exists to get right.
            tangents = (
                robust_edge_tangents(coords, scale, spacing_um=sp)
                if branch_aware
                else (
                    skeleton_optimise.plane_normals(
                        coords, np.maximum(scale, 0.0), skeleton_optimise._arclength(coords)
                    )
                    if n >= 2
                    else np.array([[0.0, 0.0, 1.0]])
                )
            )
            ijk = coords_ijk[sid]

            arc = skeleton_optimise._arclength(coords)
            seg = graph.segment(sid)
            legacy_junction = np.zeros(n, dtype=bool)
            if not branch_aware:
                if graph.degree(seg["node1"]) >= 3:
                    legacy_junction |= arc <= JUNCTION_RADII * junction_scales[seg["node1"]]
                if graph.degree(seg["node2"]) >= 3:
                    legacy_junction |= (
                        (arc[-1] - arc) <= JUNCTION_RADII * junction_scales[seg["node2"]]
                    )
            stable = np.zeros(n, dtype=bool)
            adjacent_overlap = np.zeros(n, dtype=bool)
            invented = mask_for_segment(graph, sid)
            if invented.size != n:
                invented = np.zeros(n, dtype=bool)
            invented_by_sid[sid] = invented
            if section_filter:
                from .section_validation import REJECTION_REASONS
                result.section_rejection_counts[sid] = np.zeros((n, len(REJECTION_REASONS)), dtype=int)
                result.section_target_obliquity_degrees[sid] = np.zeros(n)

            for i in range(n):
                if invented[i]:
                    # There is no cross-section to measure here: the centreline itself was
                    # interpolated. Measuring anyway would produce a number that looks like
                    # every other measured radius and describes a vessel nobody observed.
                    reject[i] = INTERPOLATED_INPUT
                    continue
                if legacy_junction[i]:
                    reject[i] = JUNCTION
                    continue
                if np.any(ijk[i] < 0) or np.any(ijk[i] >= dims):
                    continue
                rp = max(float(scale[i]) / sp, 1.0)
                chosen = None
                section_diagnostics = {}
                if branch_aware:
                    chosen = stable_transverse_cut(
                        sampler,
                        ijk[i],
                        tangents[i],
                        rp,
                        spacing_um=sp,
                        max_half=max_half,
                        search_degrees=tangent_search_degrees,
                        min_blob_voxels=min_blob_voxels,
                        slab_offsets=(0.0, 0.25, 0.5) if i == 0 else (
                            (-0.5, -0.25, 0.0) if i == n - 1 else (-0.5, 0.0, 0.5)
                        ),
                        transverse_axis_ratio=(np.inf if section_filter else transverse_axis_ratio),
                        validator=(section_context.validator(sid, tangents[i], sampler, frame, section_diagnostics)
                                   if section_context is not None else None),
                        diagnostics=section_diagnostics,
                        grow_radii=grow_radii,
                        centroid_mode=stability_centroid_mode,
                        max_variation=stability_variation,
                        max_centroid_radii=stability_centroid_radii,
                    )
                    c = chosen.cut if chosen is not None else None
                    if section_filter:
                        result.section_rejection_counts[sid][i] = [
                            section_diagnostics.get(reason, 0) for reason in REJECTION_REASONS]
                        result.section_target_obliquity_degrees[sid][i] = section_diagnostics.get(
                            'max_target_obliquity_degrees', 0.)
                    if chosen is not None and chosen.owned:
                        modes[i] = OWNED_PLANE
                    tangent = chosen.tangent if chosen is not None else tangents[i]
                    if section_filter and chosen is None:
                        if "neighbouring_lumen_contamination" in section_diagnostics:
                            reject[i] = BRANCH_OVERLAP
                        elif "truncation" in section_diagnostics:
                            reject[i] = TRUNCATED
                            result.n_truncated += 1
                        elif "unstable_section" in section_diagnostics:
                            reject[i] = UNSTABLE
                        continue
                else:
                    c = cut(
                        sampler, ijk[i], tangents[i], min(int(rp * 2.5) + 2, max_half),
                        max_half=max_half, min_blob_voxels=min_blob_voxels,
                        grow_to=_grow_to(grow_radii, rp),
                    )
                    tangent = tangents[i]

                rivals = branch_context.rivals(sid, coords[i], tangent, float(scale[i])) if branch_context else []
                adjacent_overlap[i] = any(item[4] for item in rivals)
                if section_filter:
                    # Every accepted candidate and slab companion is exclusive
                    # under the shared finite-volume test. Do not reintroduce
                    # projection-only junction masking or ownership below.
                    rivals = []
                    adjacent_overlap[i] = False

                if c is None:
                    if branch_aware:
                        probe = cut(
                            sampler, ijk[i], tangent, min(int(rp * 2.5) + 2, max_half),
                            max_half=max_half, min_blob_voxels=min_blob_voxels,
                            grow_to=_grow_to(grow_radii, rp),
                        )
                        in_blob = [
                            item for item in rivals
                            if probe is not None
                            and (ownership_near_junctions or not adjacent_overlap[i])
                            and _rival_lies_in_blob(probe, coords[i], item[2], sp)
                        ]
                        connected_nonadjacent = [
                            int(item[0]) for item in in_blob if not item[4]
                        ]
                        if probe is not None and not probe.touches_border and connected_nonadjacent:
                            owned = _resolve_owned_cut(
                                sampler, graph, coords_ijk, sid,
                                _ownership_seeds(in_blob, ownership_near_junctions),
                                ijk[i], tangent, rp, probe.half, max_half, sp,
                            )
                            if owned is not None:
                                c = owned
                                modes[i] = OWNED_PLANE
                                stable[i] = True
                            else:
                                reject[i] = BRANCH_OVERLAP
                                result.n_ownership_failed += 1
                        elif probe is not None and probe.touches_border:
                            result.n_truncated += 1
                            reject[i] = TRUNCATED
                        elif probe is not None:
                            reject[i] = UNSTABLE
                    if c is None:
                        continue
                if c.touches_border:
                    result.n_truncated += 1
                    reject[i] = TRUNCATED
                    continue
                stable[i] = True

                if branch_aware:
                    in_blob = [
                        item for item in rivals
                        if (ownership_near_junctions or not adjacent_overlap[i])
                        and _rival_lies_in_blob(c, coords[i], item[2], sp)
                    ]
                    connected_nonadjacent = [
                        int(item[0]) for item in in_blob if not item[4]
                    ]
                    if connected_nonadjacent:
                        seeds = _ownership_seeds(in_blob, ownership_near_junctions)
                        owned = _resolve_owned_cut(
                            sampler,
                            graph,
                            coords_ijk,
                            sid,
                            seeds,
                            ijk[i],
                            tangent,
                            rp,
                            c.half,
                            max_half,
                            sp,
                        )
                        if owned is None and not np.allclose(tangent, tangents[i]):
                            # The cone search saw the combined touching lumens, not
                            # this branch's owned section. Its winning orientation
                            # can lose valid seeds at an endpoint. Retry ownership
                            # on the fitted branch tangent before refusing the cut.
                            owned = _resolve_owned_cut(
                                sampler, graph, coords_ijk, sid, seeds,
                                ijk[i], tangents[i], rp, c.half, max_half, sp,
                            )
                        if owned is None:
                            reject[i] = BRANCH_OVERLAP
                            result.n_ownership_failed += 1
                            continue
                        c = owned
                        modes[i] = OWNED_PLANE
                    elif modes[i] != OWNED_PLANE:
                        modes[i] = DIRECT_PLANE
                else:
                    modes[i] = DIRECT_PLANE
                area = float(c.blob4.sum()) * sp * sp
                r_area = float(np.sqrt(area / np.pi))
                if r_area >= gate_um:
                    r_perim = _perimeter_um(c.blob4, sp) / (2.0 * np.pi)
                    if perimeter_correction:
                        measured[i] = correct_perimeter_radius(r_perim, sp)
                        source[i] = PERIMETER_CORRECTED
                    else:
                        measured[i] = r_perim
                        source[i] = PERIMETER
                else:
                    measured[i] = r_area
                    source[i] = AREA

                reject[i] = ACCEPTED
                grew_too_far[i] = (
                    float(c.half) * sp > RUNAWAY_HALF_RADII * max(float(scale[i]), sp)
                )

            if branch_aware and not section_filter:
                node_runs: dict[int, list[int]] = {}
                junction, lengths = _adaptive_junction_mask(
                    graph, sid, arc, stable, adjacent_overlap,
                    max_fraction=junction_mask_max_fraction,
                    runs_by_node=node_runs,
                    node_limits=(
                        None if junction_mask_radii is None else
                        {nid: float(junction_mask_radii) * scale
                         for nid, scale in junction_scales.items()}
                    ),
                )
                if junction_flare != "none" and junction.any():
                    keep = np.zeros(n, dtype=bool)
                    for nid, idxs in node_runs.items():
                        if junction_flare == "all" or junction_parents.get(nid) == sid:
                            keep[np.asarray(idxs, dtype=int)] = True
                    # Only a point that actually produced a section can keep one; the
                    # rest of the run is un-measured for the usual reasons.
                    keep &= np.isfinite(measured) & (source != FILLED)
                    junction &= ~keep
                if junction.any():
                    measured[junction] = np.nan
                    source[junction] = FILLED
                    reject[junction] = JUNCTION
                    modes[junction] = INTERPOLATED
                    result.junction_lengths_um.extend(lengths)
            raw_measured[sid], raw_source[sid], raw_reject[sid] = measured, source, reject
            raw_grew[sid], old_by_sid[sid], arc_by_sid[sid] = grew_too_far, old, arc
            raw_modes[sid] = modes

            if progress is not None:
                progress(n_done, len(sids))

        # How far this pass moved the accepted radii relative to the previous
        # one. Measured on every pass including the last -- that is the number
        # that says whether the circular dependency mattered here.
        moved = []
        for sid in sids:
            m, src = raw_measured.get(sid), raw_source.get(sid)
            prev = prev_measured.get(sid)
            if m is None or src is None or prev is None or len(prev) != len(m):
                continue
            both = (np.isfinite(m) & (src != FILLED) & (m > 0)
                    & np.isfinite(prev) & (prev > 0))
            if both.any():
                moved.append(np.abs(m[both] - prev[both]) / prev[both])
        if moved:
            pass_medians.append(float(np.median(np.concatenate(moved))))

        # Feed this pass's accepted measurements back as the geometry scale for
        # the next one. Only points that were actually measured contribute;
        # rejected points keep whatever scale they had, so a failed section
        # cannot hand a wrong length scale to its own retry.
        if not final_pass:
            for sid in sids:
                base = scale_by_sid.get(sid)
                fallback = old_by_sid.get(sid)
                if fallback is None:
                    continue
                if base is None or len(base) != len(fallback):
                    base = fallback
                m, src = raw_measured.get(sid), raw_source.get(sid)
                if m is None or src is None or len(m) != len(base):
                    continue
                accepted = np.isfinite(m) & (src != FILLED) & (m > 0)
                scale_by_sid[sid] = np.where(accepted, m, base)

    if _raw_only:
        # Workers measure against the whole graph's branch context. Calibration,
        # confidence filtering, junction profiles and fallbacks are applied once by
        # the parent after all batches have returned, in original segment order.
        return dict(measured=raw_measured, source=raw_source, reject=raw_reject,
                    grew=raw_grew, modes=raw_modes, old=old_by_sid, arc=arc_by_sid,
                    invented=invented_by_sid, before=before_all, result=result)

    # Confidence filtering is deliberately a second pass: every local baseline and
    # junction target is made from the complete set of provisional measurements.
    calibration_ratios = []
    for sid in sids:
        measured, old = raw_measured[sid], old_by_sid[sid]
        ok = np.isfinite(measured) & (measured > 0) & np.isfinite(old) & (old > 0)
        calibration_ratios.extend((measured[ok] / old[ok]).tolist())
    input_calibration = (
        float(np.median(calibration_ratios)) if calibration_ratios else 1.0
    )

    for sid in sids:
        measured, source, reject = raw_measured[sid], raw_source[sid], raw_reject[sid]
        old, arc = old_by_sid[sid], arc_by_sid[sid]
        runaway = _robust_high_correction_mask(arc, measured, old, raw_grew[sid])
        measured[runaway] = np.nan
        source[runaway] = FILLED
        reject[runaway] = RUNAWAY
        raw_modes[sid][runaway] = INTERPOLATED
        local_high = _robust_local_high_mask(
            arc, measured, old, factor=max_radius_factor,
            input_calibration=input_calibration,
        )
        measured[local_high] = np.nan
        source[local_high] = FILLED
        reject[local_high] = CEILING
        raw_modes[sid][local_high] = INTERPOLATED

    # OFF BY DEFAULT. The tapers AUTHOR a radius profile through each junction
    # rather than measuring one, and that model has not been validated against the
    # segmentation it is standing in for -- on this graph it covered ~7,000 points,
    # of which roughly half were vessel continuations being pinched to a tenth of
    # their radius. Until the carina model is tested on its own terms, an honest
    # interpolation labelled as such is the safer input to a CFD surface.
    #
    # The tapers AUTHOR a radius profile through each junction rather than
    # measuring one: a parent-through trend, a daughter ostium narrowing to
    # `carina_tip_factor`, and now a through-junction continuation. That is a
    # model of carina geometry, not an observation of it, and on this graph it
    # covers ~7,000 points. With `bifurcation_tapers=False` the junction runs are
    # left as ordinary rejected spans, so `_fill_gaps` interpolates them from the
    # surrounding trusted radii and labels them `INTERPOLATED` -- no invented
    # carina, and provenance that says plainly the value was not measured.
    taper_fallback = _apply_bifurcation_tapers(
        graph,
        raw_measured,
        raw_source,
        raw_reject,
        raw_modes,
        arc_by_sid,
        spacing_um=sp,
        root_edges=root_edges,
        carina_tip_factor=carina_tip_factor,
        continuation_ratio=continuation_ratio,
        daughter_carina=bifurcation_tapers,
    ) if (not section_filter and (bifurcation_tapers or junction_parent_profile)) else set()
    result.fallback_segments.extend(sorted(taper_fallback))
    for sid in sids:
        measured, source, reject = raw_measured[sid], raw_source[sid], raw_reject[sid]
        modes = raw_modes[sid]
        old, arc = old_by_sid[sid], arc_by_sid[sid]
        accepted = reject == ACCEPTED
        ratios.extend((measured[accepted & (old > 0)] / old[accepted & (old > 0)]).tolist())
        result.n_measured += int(accepted.sum())
        had_trusted = bool(np.any(accepted))
        filled, was_filled = _fill_gaps(measured, old)
        authored = ((modes == BIF_PARENT) | (modes == BIF_DAUGHTER)
                    | (modes == BIF_CONTINUATION) | (modes == INPUT_FALLBACK))
        modes[was_filled & ~authored] = INTERPOLATED if had_trusted else INPUT_FALLBACK
        # Last word on the invented points, after the tapers and the gap fill: they keep
        # exactly the radius they came in with. Interpolating them from their measured
        # neighbours would be a defensible number, but it would also be a *new* number
        # attached to a point that has no cross-section -- and it would arrive labelled
        # like every ordinary interpolation, which is the confusion worth avoiding.
        invented = invented_by_sid.get(sid)
        if invented is not None and invented.size == len(filled) and invented.any():
            filled[invented] = old[invented]
            source[invented] = FILLED
            modes[invented] = INPUT_FALLBACK
        if not had_trusted:
            # No cross-section on this segment could be trusted, so `filled` is
            # still the uncorrected input. Those are the values this pass exists
            # to replace, and left alone they enter the CFD surface as spheres.
            # Resolving them is deferred to a second pass below: the policies read
            # the *neighbours'* measurements, and half of any segment's neighbours
            # are still unprocessed at this point in the loop.
            no_trusted.append(sid)
            result.fallback_segments.append(sid)
        result.n_filled += int((source == FILLED).sum())
        result.radii[sid], result.source[sid] = filled, source
        result.reject_reason[sid] = reject
        result.resolution_mode[sid] = modes

    # ---- second pass: segments that measured nothing at all -------------------
    # Every measurement now exists, so a segment's junction neighbours can be read
    # whatever order they came in. Doing this inside the loop above anchored each
    # span on the subset of its neighbours that happened to be processed earlier,
    # which on a leaf is usually none -- and a leaf whose one junction sits later in
    # the ordering then kept its uncorrected input, spike and all.
    calibrated = np.isfinite(input_calibration) and float(input_calibration) > 0
    untrusted = set(no_trusted)
    if fallback_policy == "rescale" and calibrated:
        # Keep each segment's shape, move it onto the measured scale using the
        # ratio the rest of the tree actually showed.
        for sid in no_trusted:
            result.radii[sid] = result.radii[sid] * float(input_calibration)
    elif fallback_policy == "drop":
        # Refuse to invent a calibre: hand each span to its junction neighbours,
        # which were measured. Un-measured neighbours are excluded, so a segment
        # in a cluster of them has no anchor on the first round -- but a cluster
        # usually touches measured tissue *somewhere* on its far side, and once
        # that member is resolved it can anchor the rest. Rounds proceed outward
        # from the measured tree rather than in segment-id order, and stop when a
        # round resolves nothing; whatever is left is enclosed entirely by
        # un-measured segments and gets the scale fallback below.
        #
        # A value resolved on round 2 is inferred from one that was itself
        # inferred. That is weaker than a measurement and is still reported as
        # `INPUT_FALLBACK`, but it beats leaving a segment at an input calibre
        # known to be wrong by up to a factor of two.
        while True:
            progressed = False
            for sid in sorted(untrusted):
                anchors = _junction_anchor_radii(graph, sid, result.radii, untrusted)
                if not anchors:
                    continue
                filled = result.radii[sid]
                lo, hi = anchors[0], anchors[-1]
                if fallback_taper:
                    t = np.linspace(0.0, 1.0, len(filled))
                    result.radii[sid] = np.exp((1.0 - t) * np.log(lo)
                                               + t * np.log(hi))
                else:
                    # A ramp between the two ends asserts the vessel narrows
                    # steadily across a span where nothing was measured. That is
                    # a model of taper, not an observation of one, and it is the
                    # same unverified claim the carina taper makes. One constant
                    # calibre says only what the anchors say: the vessel here is
                    # about this wide.
                    result.radii[sid] = np.full(len(filled),
                                                float(np.sqrt(lo * hi)))
                untrusted.discard(sid)
                progressed = True
            if not progressed:
                break
        if calibrated:
            for sid in sorted(untrusted):
                result.radii[sid] = result.radii[sid] * float(input_calibration)

    after_all = [result.radii[sid].copy() for sid in sids]

    result.ratio = np.asarray(ratios, dtype=np.float64)
    result.before_radii = np.concatenate(before_all) if before_all else np.zeros(0)
    result.after_radii = np.concatenate(after_all) if after_all else np.zeros(0)
    result.pass_movement = pass_medians
    result.seconds = time.time() - t0
    return result


def apply_radii(graph, result: RadiusResult, *, mean_radius: bool = True) -> None:
    """Write the measured radii onto the graph, with their provenance.

    ``radius_source`` goes on as a per-point field so the estimator mix is visible in
    the written ``.am`` -- :func:`~.amira_write.write_spatial_graph` emits any extra
    scalar of the right length -- and ``MeanRadius`` is re-derived on every edge from
    the new thickness, which is only correct because every point was re-measured.
    """
    with graph.batch("perimeter radii"):
        for sid, radii in result.radii.items():
            if graph.has_segment(sid) and len(radii):
                graph.set_segment_radii(sid, radii)

    # Keyed by point id, not by position: that is the ``Triple`` contract, and it is
    # what lets the field survive a later edit that inserts or deletes points.
    triple = graph.triple if hasattr(graph, "triple") else graph
    by_id: dict[int, int] = {}
    reject_by_id: dict[int, int] = {}
    resolution_by_id: dict[int, int] = {}
    for sid, source in result.source.items():
        if not graph.has_segment(sid):
            continue
        pids = graph.segment(sid)["point_ids"]
        for pid, value in zip(pids, np.asarray(source, dtype=np.int64)):
            by_id[int(pid)] = int(value)
        for pid, value in zip(pids, np.asarray(result.reject_reason[sid], dtype=np.int64)):
            reject_by_id[int(pid)] = int(value)
        for pid, value in zip(pids, np.asarray(result.resolution_mode[sid], dtype=np.int64)):
            resolution_by_id[int(pid)] = int(value)
    triple.point_attrs.setdefault("radius_source", {}).update(by_id)
    triple.point_attr_dtypes["radius_source"] = np.dtype(np.int64)
    triple.point_attrs.setdefault("radius_reject_reason", {}).update(reject_by_id)
    triple.point_attr_dtypes["radius_reject_reason"] = np.dtype(np.int64)
    triple.point_attrs.setdefault("radius_resolution_mode", {}).update(resolution_by_id)
    triple.point_attr_dtypes["radius_resolution_mode"] = np.dtype(np.int64)
    from .section_validation import REJECTION_REASONS
    for sid, counts in result.section_rejection_counts.items():
        pids = graph.segment(sid)['point_ids']
        for col, reason in enumerate(REJECTION_REASONS):
            name = 'radius_section_rejected_'+reason
            triple.point_attrs.setdefault(name, {}).update(dict(zip(pids, counts[:, col].tolist())))
            triple.point_attr_dtypes[name] = np.dtype(np.int64)
        name = 'radius_section_max_obliquity_degrees'
        triple.point_attrs.setdefault(name, {}).update(dict(zip(
            pids, result.section_target_obliquity_degrees[sid].tolist())))
        triple.point_attr_dtypes[name] = np.dtype(np.float64)
    if 'radius_measured_um' in triple.point_attrs:
        # A deliberate remeasurement supersedes cached reconstruction provenance
        # only on the measured segments. The next profile starts from these values.
        for sid, radii in result.radii.items():
            pids = graph.segment(sid)['point_ids']
            for name in ('radius_measured_um', 'radius_reconstruction_um'):
                triple.point_attrs.setdefault(name, {}).update(dict(zip(pids, radii.tolist())))
            for name in ('radius_adjustment_um', 'radius_adjustment_reason'):
                triple.point_attrs.setdefault(name, {}).update(dict.fromkeys(pids, 0))

    if mean_radius:
        from .optimise import set_edge_field

        means = np.array(
            [float(np.mean(graph.radii(sid)))
             if len(graph.radii(sid)) else np.nan
             for sid in graph.segment_ids()],
            dtype=np.float64,
        )
        set_edge_field(graph, "MeanRadius", means, np.float64)
