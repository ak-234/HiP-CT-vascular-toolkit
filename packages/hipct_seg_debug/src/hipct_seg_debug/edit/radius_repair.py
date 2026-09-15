"""Restore radii across a collapsed vessel by extrapolating its taper.

A collapsed vessel does not lose its centreline -- the skeleton runs straight
through it -- it loses its *calibre*. The pipeline assigns radius from the
cross-section perimeter, and where the lumen has flattened the perimeter
under-states it badly, so the surface pinches to a thread over a span that should
be a smoothly tapering tube.

Nothing existing can fix that:

* ``skeleton_analysis.outlier.filloutliers_nearest`` copies the nearest good
  *value*, which flattens the region to a constant rather than continuing the
  taper, and its threshold is a percentile of the whole segment -- a long collapse
  moves the percentile and hides itself.
* ``adjust_thickness.fit_line`` fits one global ``r_eff ~ slope*t + intercept`` for
  the entire tree, so it cannot express local geometry at all (and its clamped
  intercept is where the ~156 um terminal radius floor comes from).
* ``coronary_sdf.smoothing``'s four radius passes are all junction- or
  endpoint-anchored: ``smooth_segment_radii`` deliberately preserves the level, and
  the other three only reach a few points in from a node. A mid-segment collapse
  is out of range of every one of them.

So the model here is new, and deliberately the simplest one that respects the
physiology: **radius falls at a roughly constant fraction per unit length along an
unbranched vessel**, i.e. ``log r`` is linear in arclength. Fitting in log space
rather than in ``r`` matters for two reasons -- it is scale-free, so a trunk and a
twig get the same treatment, and an extrapolated radius can never come out
negative, which a linear fit across a long gap readily does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .history import Patch
from .interpolation import mask_for_segment

# Refuse to fit from fewer than this many healthy points on a side.
MIN_HEALTHY = 5
# Largest fractional radius change per millimetre the fit may extrapolate with.
# ln(r) slope of 0.5/mm is a 65% change over 1 mm, already far steeper than any
# real coronary taper; beyond this a short noisy window is driving the fit.
MAX_TAPER_PER_MM = 0.5
# Points at each end of a segment that are never treated as collapsed.
#
# This is not a tuning knob, it is a correctness guard. A lumen at a bifurcation
# is *legitimately* non-circular -- that is what a carina looks like -- so the
# image detector fires there on healthy anatomy. Measured on LADAF-2024-28, 15 of
# 83 `collapse_severity` runs began at point 0 of their segment, i.e. exactly on a
# junction. Filling those would inflate a carina on no evidence, and the junction
# neighbourhood already belongs to `coronary_sdf.smoothing`'s
# `prune_bifurcation_shrink` and `smooth_radius_transitions`.
JUNCTION_MARGIN = 6
# How far above the larger of the two boundary radii a blended fill may go.
#
# Also a correctness guard rather than a taste setting. When both ends of a gap
# are anchored on healthy tissue, the vessel between them is a length of
# unbranched tube: it may taper from one to the other, but it has no reason to
# swell above both. A short fitting window gives a badly-determined slope -- six
# points spanning 0.3 mm, extrapolated across a 2 mm gap -- and r-squared does not
# catch it, because the problem is the *lever arm*, not the scatter. Measured on
# LADAF-2024-28 without this, `seg 237[6:39]` went from 375 um to 572 um at the
# centre of a span whose two ends were 353 and 375.
MAX_BULGE = 1.05


@dataclass
class CollapsedSpan:
    """A contiguous run of points on one segment whose radius is not to be trusted."""

    seg_id: int
    i0: int  # first index of the span, within the segment's point list
    i1: int  # last index, inclusive
    source: str = "manual"  # "image" | "outlier" | "high_outlier" | "manual"
    metric: float = 0.0
    length_um: float = 0.0

    @property
    def n_points(self) -> int:
        return self.i1 - self.i0 + 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<CollapsedSpan seg {self.seg_id}[{self.i0}:{self.i1}] "
                f"{self.n_points} pts {self.length_um:.0f}um {self.source}>")


@dataclass
class FillReport:
    """What a fill did, or why it declined to."""

    span: CollapsedSpan
    applied: bool
    reason: str = ""
    n_changed: int = 0
    r_before: tuple = (0.0, 0.0)  # (min, max) over the span
    r_after: tuple = (0.0, 0.0)
    slope_per_mm: float = 0.0
    r2: float = 0.0
    sides: str = ""  # "both" | "proximal" | "distal"

    def __repr__(self) -> str:  # pragma: no cover
        if not self.applied:
            return f"<FillReport {self.span} declined: {self.reason}>"
        return (f"<FillReport {self.span} {self.r_before[0]:.0f}-{self.r_before[1]:.0f} -> "
                f"{self.r_after[0]:.0f}-{self.r_after[1]:.0f} um, {self.sides}, "
                f"slope {self.slope_per_mm:+.3f}/mm r2={self.r2:.2f}>")


def arclength(coords: np.ndarray) -> np.ndarray:
    """(N,) cumulative distance along a polyline, starting at 0. Same units as `coords`."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if len(coords) < 2:
        return np.zeros(len(coords))
    step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def fit_log_taper(arc_um: np.ndarray, radii_um: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of ``log(r) = intercept + slope * s``.

    Returns ``(slope_per_um, intercept, r2)``. Points with a non-positive or
    non-finite radius are dropped -- a collapsed point often reads as zero, and
    ``log(0)`` would poison the fit rather than merely bias it.
    """
    arc = np.asarray(arc_um, dtype=np.float64).ravel()
    rad = np.asarray(radii_um, dtype=np.float64).ravel()
    ok = np.isfinite(arc) & np.isfinite(rad) & (rad > 0)
    if ok.sum() < 2:
        return 0.0, float(np.log(np.median(rad[rad > 0]))) if (rad > 0).any() else 0.0, 0.0

    x, y = arc[ok], np.log(rad[ok])
    if np.ptp(x) < 1e-9:
        return 0.0, float(y.mean()), 0.0

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return float(slope), float(intercept), float(r2)


def _clamped(slope_per_um: float, max_per_mm: float) -> float:
    """Bound the taper so a short noisy window cannot explode over a long gap."""
    limit = max_per_mm / 1000.0  # per um
    return float(np.clip(slope_per_um, -limit, limit))


def taper_fill(
    arc_um: np.ndarray,
    radii_um: np.ndarray,
    i0: int,
    i1: int,
    *,
    min_healthy: int = MIN_HEALTHY,
    window: int = 40,
    max_taper_per_mm: float = MAX_TAPER_PER_MM,
    max_bulge: float = MAX_BULGE,
    only_increase: bool = True,
    blend: bool = True,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Replace ``radii_um[i0:i1+1]`` with values continuing the taper either side.

    Both sides are fitted separately over up to `window` healthy points, then
    evaluated across the gap and blended by distance -- so the result meets the
    healthy radius exactly at *both* boundaries instead of stepping at the far one,
    which a proximal-only extrapolation always does. With too few points distally
    (a collapse that runs into a terminal) it falls back to proximal-only, and vice
    versa.

    `exclude` is a per-point boolean of radii that must not be used as evidence -- the
    ones Avizo interpolated. Extrapolating a real vessel's taper from an invented radius
    is how a made-up number gets laundered into a measurement.

    Returns ``(new_radii, info)``; `new_radii` is a copy.
    """
    arc = np.asarray(arc_um, dtype=np.float64).ravel()
    rad = np.asarray(radii_um, dtype=np.float64).ravel().copy()
    n = len(rad)
    i0, i1 = int(max(i0, 0)), int(min(i1, n - 1))
    info: dict = {"sides": "", "slope_per_mm": 0.0, "r2": 0.0, "reason": ""}
    if i0 > i1:
        info["reason"] = "empty span"
        return rad, info

    prox_lo = max(0, i0 - window)
    dist_hi = min(n, i1 + 1 + window)
    prox = np.arange(prox_lo, i0)
    dist = np.arange(i1 + 1, dist_hi)

    bad = (
        np.zeros(n, dtype=bool) if exclude is None
        else np.asarray(exclude, dtype=bool).ravel()
    )
    if len(bad) != n:
        bad = np.zeros(n, dtype=bool)

    def usable(idx):
        if not len(idx):
            return idx
        return idx[np.isfinite(rad[idx]) & (rad[idx] > 0) & ~bad[idx]]

    prox, dist = usable(prox), usable(dist)
    have_prox, have_dist = len(prox) >= min_healthy, len(dist) >= min_healthy
    if not have_prox and not have_dist:
        info["reason"] = (
            f"only {len(prox)} proximal and {len(dist)} distal healthy points, "
            f"need {min_healthy} on a side"
        )
        return rad, info

    span = np.arange(i0, i1 + 1)
    fits = {}
    if have_prox:
        s, b, r2 = fit_log_taper(arc[prox], rad[prox])
        fits["proximal"] = (_clamped(s, max_taper_per_mm), b, r2, arc[prox[-1]])
    if have_dist:
        s, b, r2 = fit_log_taper(arc[dist], rad[dist])
        fits["distal"] = (_clamped(s, max_taper_per_mm), b, r2, arc[dist[0]])

    def evaluate(key):
        slope, intercept, _r2, anchor = fits[key]
        # Re-anchor on the healthy radius at the boundary so the fill is continuous
        # there even when the fit's own intercept is slightly off.
        r_anchor = float(np.exp(slope * anchor + intercept))
        boundary = rad[prox[-1]] if key == "proximal" else rad[dist[0]]
        offset = np.log(boundary) - np.log(r_anchor)
        return np.exp(slope * arc[span] + intercept + offset)

    if have_prox and have_dist and blend:
        a, b = evaluate("proximal"), evaluate("distal")
        s0, s1 = arc[prox[-1]], arc[dist[0]]
        t = np.clip((arc[span] - s0) / max(s1 - s0, 1e-9), 0.0, 1.0)
        # Smoothstep rather than linear: zero gradient at both ends, so the join
        # has no visible crease once the surface is swept along it.
        w = t * t * (3.0 - 2.0 * t)
        filled = np.exp((1.0 - w) * np.log(a) + w * np.log(b))
        # Both ends are anchored on healthy tissue, so this is a length of
        # unbranched tube between two known calibres. Cap it there -- see MAX_BULGE.
        lo_r, hi_r = float(rad[prox[-1]]), float(rad[dist[0]])
        ceiling = max(lo_r, hi_r) * float(max_bulge)
        floor = min(lo_r, hi_r) / float(max_bulge)
        clipped = np.clip(filled, floor, ceiling)
        info["bulge_clipped"] = int(np.count_nonzero(~np.isclose(clipped, filled)))
        filled = clipped
        info["sides"] = "both"
        info["slope_per_mm"] = float(
            0.5 * (fits["proximal"][0] + fits["distal"][0]) * 1000.0
        )
        info["r2"] = float(min(fits["proximal"][2], fits["distal"][2]))
    else:
        key = "proximal" if have_prox else "distal"
        filled = evaluate(key)
        # One-sided extrapolation has no second boundary to restrain it.  It may
        # continue a falling taper, but it must not invent a new maximum beyond the
        # healthy window that supports the fit.
        trusted = rad[prox] if have_prox else rad[dist]
        ceiling = float(np.max(trusted)) * float(max_bulge)
        clipped = np.minimum(filled, ceiling)
        info["bulge_clipped"] = int(np.count_nonzero(~np.isclose(clipped, filled)))
        filled = clipped
        info["sides"] = key
        info["slope_per_mm"] = float(fits[key][0] * 1000.0)
        info["r2"] = float(fits[key][2])

    if only_increase:
        # A collapse under-states the radius, so the fit is a floor, not a
        # replacement -- anywhere the stored value is already larger, keep it.
        filled = np.maximum(filled, rad[span])

    rad[span] = filled
    return rad, info


# --------------------------------------------------------------------- detectors


def find_outlier_spans(
    graph,
    *,
    factor: float = 0.6,
    min_points: int = 3,
    window: int = 40,
    min_segment_points: int = 20,
    margin: int = JUNCTION_MARGIN,
    mode: str = "low",
) -> list[CollapsedSpan]:
    """Spans whose radius sits far below or above the segment's own trend.

    Needs no image: fits ``log r`` against arclength robustly over the whole
    segment, then flags contiguous runs below ``factor`` times that trend. The fit
    For ``mode='low'`` the second fit uses the upper residual half (healthy rather
    than collapsed); for ``mode='high'`` it uses the lower residual half.  Thus the
    outliers being sought cannot establish their own baseline.  The same ``factor``
    is reciprocal in high mode: with 0.6, values above ``trend / 0.6`` are high.
    """
    if mode not in ("low", "high"):
        raise ValueError("mode must be 'low' or 'high'")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be positive")
    out: list[CollapsedSpan] = []
    for seg in graph.segments:
        pids = seg["point_ids"]
        if len(pids) < min_segment_points:
            continue
        coords = graph.coords(seg["id"])
        rad = graph.radii(seg["id"])
        arc = arclength(coords)
        # Invented radii are neither evidence nor a target: they cannot establish the
        # trend, and "this made-up number disagrees with the trend" is not a collapse.
        ok = np.isfinite(rad) & (rad > 0) & ~mask_for_segment(graph, seg["id"])
        if ok.sum() < max(min_points * 2, 8):
            continue

        slope, intercept, _ = fit_log_taper(arc[ok], rad[ok])
        resid = np.log(np.where(ok, rad, 1.0)) - (slope * arc + intercept)
        # Re-fit on the half opposite the outliers: the healthy tissue.
        median = np.median(resid[ok])
        keep = ok & ((resid >= median) if mode == "low" else (resid <= median))
        if keep.sum() >= 4:
            slope, intercept, _ = fit_log_taper(arc[keep], rad[keep])

        trend = np.exp(slope * arc + intercept)
        bad = (
            ok & (rad < factor * trend)
            if mode == "low"
            else ok & (rad > trend / float(factor))
        )
        if not bad.any():
            continue
        for i0, i1 in _runs_of(bad, min_points):
            clipped = clip_to_interior(i0, i1, len(rad), margin=margin,
                                       min_points=min_points)
            if clipped is None:
                continue
            i0, i1 = clipped
            out.append(CollapsedSpan(
                seg_id=seg["id"], i0=i0, i1=i1,
                source="outlier" if mode == "low" else "high_outlier",
                metric=float(np.median(rad[i0:i1 + 1] / np.maximum(trend[i0:i1 + 1], 1e-9))),
                length_um=float(arc[i1] - arc[i0]),
            ))
    return out


def has_perimeter_radii(graph) -> bool:
    """Whether graph provenance says its radii were already perimeter-measured."""
    triple = graph.triple if hasattr(graph, "triple") else graph
    return "radius_source" in getattr(triple, "point_attrs", {})


def clip_to_interior(i0: int, i1: int, n_points: int, *, margin: int = JUNCTION_MARGIN,
                     min_points: int = 1) -> tuple[int, int] | None:
    """Trim a span away from the segment's ends, or reject it if nothing survives.

    See :data:`JUNCTION_MARGIN` for why this exists. A span that merely *reaches*
    the margin is trimmed and kept; one that lies entirely inside it is dropped,
    because there is no evidence it is anything but a junction.
    """
    lo = max(int(i0), int(margin))
    hi = min(int(i1), int(n_points) - 1 - int(margin))
    if hi < lo or (hi - lo + 1) < min_points:
        return None
    return lo, hi


def _runs_of(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Inclusive ``(start, stop)`` index pairs of True runs at least `min_len` long."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return [
        (int(s), int(t - 1)) for s, t in zip(edges[0::2], edges[1::2]) if t - s >= min_len
    ]


def spans_from_candidates(graph, candidates: Iterable, kinds=("collapse_severity",),
                          *, margin: int = JUNCTION_MARGIN, min_points: int = 3
                          ) -> list[CollapsedSpan]:
    """Convert image-detected candidates into spans.

    ``crosssection.find_sites`` reports ``point_a``/``point_b`` -- the global point
    indices bounding each collapsed run -- which have to be translated into the
    editable graph's own segment-local indices. The bridge is the point *order*:
    both representations lay points out edge by edge, so a global index is a
    position in ``graph.point_order()``.
    """
    order = graph.point_order()
    owner = graph.segment_of_point()
    out: list[CollapsedSpan] = []
    for cand in candidates:
        kind = getattr(cand, "kind", None)
        if kinds and kind not in kinds:
            continue
        a = int(getattr(cand, "point_a", -1))
        b = int(getattr(cand, "point_b", -1))
        if a < 0 or b < a or b >= len(order):
            continue
        pid_a, pid_b = order[a], order[b]
        sid = owner.get(pid_a)
        if sid is None or owner.get(pid_b) != sid:
            continue
        pids = graph.segment(sid)["point_ids"]
        try:
            i0, i1 = pids.index(pid_a), pids.index(pid_b)
        except ValueError:
            continue
        if i1 < i0:
            i0, i1 = i1, i0
        clipped = clip_to_interior(i0, i1, len(pids), margin=margin, min_points=min_points)
        if clipped is None:
            continue
        i0, i1 = clipped
        out.append(CollapsedSpan(
            seg_id=sid, i0=i0, i1=i1, source="image",
            metric=float(getattr(cand, "score", 0.0)),
            length_um=float(getattr(cand, "contact_um", 0.0)),
        ))
    return out


def span_around(graph, sid: int, index: int, *, factor: float = 0.6,
                min_points: int = 1, window: int = 40,
                margin: int = JUNCTION_MARGIN) -> CollapsedSpan | None:
    """The collapsed run containing `index` on segment `sid`, for a viewer pick.

    Same trend test as :func:`find_outlier_spans` but anchored on one point, so a
    single click can say "fix the dip I am looking at" without the user marking
    both ends of it.
    """
    coords = graph.coords(sid)
    rad = graph.radii(sid)
    if len(rad) < 4:
        return None
    arc = arclength(coords)
    ok = np.isfinite(rad) & (rad > 0)
    if ok.sum() < 4:
        return None
    slope, intercept, _ = fit_log_taper(arc[ok], rad[ok])
    resid = np.log(np.where(ok, rad, 1.0)) - (slope * arc + intercept)
    keep = ok & (resid >= np.median(resid[ok]))
    if keep.sum() >= 4:
        slope, intercept, _ = fit_log_taper(arc[keep], rad[keep])
    trend = np.exp(slope * arc + intercept)
    bad = ok & (rad < factor * trend)
    if not bad[index]:
        return None
    i0 = i1 = int(index)
    while i0 > 0 and bad[i0 - 1]:
        i0 -= 1
    while i1 < len(bad) - 1 and bad[i1 + 1]:
        i1 += 1
    clipped = clip_to_interior(i0, i1, len(rad), margin=margin, min_points=min_points)
    if clipped is None:
        return None
    i0, i1 = clipped
    return CollapsedSpan(
        seg_id=sid, i0=i0, i1=i1, source="manual",
        metric=float(np.median(rad[i0:i1 + 1] / np.maximum(trend[i0:i1 + 1], 1e-9))),
        length_um=float(arc[i1] - arc[i0]),
    )


# ------------------------------------------------------------------- application


def fill_span(graph, span: CollapsedSpan, **kw) -> FillReport:
    """Apply :func:`taper_fill` to one span, as a single undoable edit."""
    if not graph.has_segment(span.seg_id):
        return FillReport(span, False, "segment no longer exists")
    rad = graph.radii(span.seg_id)
    arc = arclength(graph.coords(span.seg_id))
    before = rad[span.i0:span.i1 + 1]

    kw.setdefault("exclude", mask_for_segment(graph, span.seg_id))
    new, info = taper_fill(arc, rad, span.i0, span.i1, **kw)
    if info["reason"]:
        return FillReport(span, False, info["reason"])

    after = new[span.i0:span.i1 + 1]
    changed = int(np.count_nonzero(~np.isclose(before, after)))
    if changed == 0:
        return FillReport(span, False, "the fit did not move any radius")

    graph.set_segment_radii(span.seg_id, new)
    return FillReport(
        span=span, applied=True, n_changed=changed,
        r_before=(float(before.min()), float(before.max())),
        r_after=(float(after.min()), float(after.max())),
        slope_per_mm=info["slope_per_mm"], r2=info["r2"], sides=info["sides"],
    )


def merge_spans(graph, spans: Iterable[CollapsedSpan]) -> list[CollapsedSpan]:
    """Union overlapping or directly adjacent bad spans on each segment.

    A high point at the boundary of a low/image span is not healthy evidence.  The
    union makes the taper fit look outside every detector's bad region, preventing
    that high from surviving as the neighbouring repair's anchor.
    """
    ordered = sorted(spans, key=lambda s: (s.seg_id, s.i0, s.i1))
    merged: list[CollapsedSpan] = []
    for span in ordered:
        if not graph.has_segment(span.seg_id):
            merged.append(span)
            continue
        if not merged or merged[-1].seg_id != span.seg_id or span.i0 > merged[-1].i1 + 1:
            merged.append(CollapsedSpan(
                span.seg_id, span.i0, span.i1, span.source, span.metric, span.length_um
            ))
            continue
        prev = merged[-1]
        prev.i1 = max(prev.i1, span.i1)
        names = set(prev.source.split("+")) | set(span.source.split("+"))
        prev.source = "+".join(sorted(names))
        prev.metric = max(abs(prev.metric), abs(span.metric))
        arc = arclength(graph.coords(prev.seg_id))
        if len(arc):
            prev.length_um = float(arc[min(prev.i1, len(arc) - 1)] - arc[max(prev.i0, 0)])
    return merged


def fill_spans(graph, spans: Iterable[CollapsedSpan], label: str = "repair radii",
               **kw) -> tuple[list[FillReport], Patch]:
    """Fill many spans as one undo step. Returns the reports and the covering patch."""
    spans = merge_spans(graph, spans)
    reports: list[FillReport] = []
    if not spans:
        return reports, Patch.empty()
    # Descending index order within a segment, so a fill cannot disturb the indices
    # a later span on the same segment is holding.
    spans.sort(key=lambda s: (s.seg_id, -s.i0))
    with graph.batch(label):
        for span in spans:
            reports.append(fill_span(graph, span, **kw))
    return reports, graph.last_patch


def summarise(reports: Iterable[FillReport]) -> str:
    reports = list(reports)
    if not reports:
        return "no collapsed spans"
    done = [r for r in reports if r.applied]
    lines = [f"{len(done)} of {len(reports)} spans filled"]
    for r in done:
        lines.append(
            f"  seg {r.span.seg_id}[{r.span.i0}:{r.span.i1}] "
            f"{r.r_before[0]:.0f}-{r.r_before[1]:.0f} -> "
            f"{r.r_after[0]:.0f}-{r.r_after[1]:.0f} um  "
            f"({r.sides}, slope {r.slope_per_mm:+.3f}/mm, r2 {r.r2:.2f})"
        )
    declined: dict[str, int] = {}
    for r in reports:
        if not r.applied:
            declined[r.reason] = declined.get(r.reason, 0) + 1
    for reason, n in sorted(declined.items(), key=lambda kv: -kv[1]):
        lines.append(f"  declined: {reason} ({n})")
    return "\n".join(lines)
