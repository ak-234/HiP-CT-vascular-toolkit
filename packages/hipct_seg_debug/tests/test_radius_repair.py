"""The taper fill must recover a known taper, and refuse when it cannot.

The failure that matters is a silent one: a fill that produces plausible numbers
from too little evidence. So most of these tests are about the guards -- refusing
on too few healthy points, clamping an implausible slope, and never lowering a
radius unless asked to.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.radius_repair import (
    CollapsedSpan,
    arclength,
    fill_span,
    fill_spans,
    find_outlier_spans,
    fit_log_taper,
    has_perimeter_radii,
    merge_spans,
    span_around,
    summarise,
    taper_fill,
)

# A vessel that tapers 20% per mm, sampled every 50 um over 10 mm.
SPACING_UM = 50.0
N = 200
R0 = 400.0
TAPER_PER_MM = -0.2


def truth(n=N, r0=R0, taper_per_mm=TAPER_PER_MM):
    arc = np.arange(n, dtype=float) * SPACING_UM
    rad = r0 * np.exp(taper_per_mm / 1000.0 * arc)
    return arc, rad


def collapsed(i0, i1, fraction=0.2, **kw):
    arc, rad = truth(**kw)
    broken = rad.copy()
    broken[i0:i1 + 1] *= fraction
    return arc, rad, broken


# ------------------------------------------------------------------ the fit

def test_fit_recovers_a_known_taper():
    arc, rad = truth()
    slope, intercept, r2 = fit_log_taper(arc, rad)
    assert slope * 1000.0 == pytest.approx(TAPER_PER_MM, rel=1e-6)
    assert np.exp(intercept) == pytest.approx(R0, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_ignores_zero_and_negative_radii():
    """A collapsed point often reads as zero; log(0) would poison the fit."""
    arc, rad = truth()
    rad[10:20] = 0.0
    slope, _, _ = fit_log_taper(arc, rad)
    assert slope * 1000.0 == pytest.approx(TAPER_PER_MM, rel=1e-6)


def test_fit_of_a_constant_profile_is_flat():
    arc = np.arange(50, dtype=float) * SPACING_UM
    slope, intercept, _ = fit_log_taper(arc, np.full(50, 123.0))
    assert slope == pytest.approx(0.0, abs=1e-12)
    assert np.exp(intercept) == pytest.approx(123.0, rel=1e-9)


# ----------------------------------------------------------------- the fill

def test_fill_recovers_the_true_radii_across_a_gap():
    i0, i1 = 80, 120
    arc, true_r, broken = collapsed(i0, i1)
    filled, info = taper_fill(arc, broken, i0, i1)

    assert info["sides"] == "both"
    got = filled[i0:i1 + 1]
    want = true_r[i0:i1 + 1]
    assert np.allclose(got, want, rtol=0.02), f"max error {np.abs(got/want - 1).max():.3%}"
    # Untouched outside the span.
    assert np.array_equal(filled[:i0], broken[:i0])
    assert np.array_equal(filled[i1 + 1:], broken[i1 + 1:])


def test_fill_is_continuous_at_both_boundaries():
    """The point of blending: a proximal-only extrapolation steps at the far end."""
    i0, i1 = 80, 120
    arc, _true, broken = collapsed(i0, i1)
    filled, _ = taper_fill(arc, broken, i0, i1)

    step_lo = abs(filled[i0] - filled[i0 - 1]) / filled[i0 - 1]
    step_hi = abs(filled[i1 + 1] - filled[i1]) / filled[i1]
    assert step_lo < 0.02, f"step of {step_lo:.2%} at the proximal boundary"
    assert step_hi < 0.02, f"step of {step_hi:.2%} at the distal boundary"


def test_fill_is_monotone_over_a_monotone_taper():
    i0, i1 = 60, 140
    arc, _true, broken = collapsed(i0, i1)
    filled, _ = taper_fill(arc, broken, i0, i1)
    span = filled[i0:i1 + 1]
    assert np.all(np.diff(span) < 0), "a falling taper should stay falling across the fill"


def test_fill_falls_back_to_proximal_when_the_distal_side_is_gone():
    """A collapse running into a terminal has healthy tissue on one side only."""
    i0, i1 = 150, N - 1
    arc, true_r, broken = collapsed(i0, i1)
    filled, info = taper_fill(arc, broken, i0, i1)

    assert info["sides"] == "proximal"
    got, want = filled[i0:i1 + 1], true_r[i0:i1 + 1]
    assert np.allclose(got, want, rtol=0.05)


def test_fill_falls_back_to_distal_when_the_proximal_side_is_gone():
    i0, i1 = 0, 50
    arc, true_r, broken = collapsed(i0, i1)
    filled, info = taper_fill(arc, broken, i0, i1)
    assert info["sides"] == "distal"
    # With evidence only distally, extrapolation is capped at 1.05 times the
    # trusted window maximum instead of inventing a new global maximum.
    trusted_max = broken[i1 + 1:i1 + 1 + 40].max()
    assert filled[i0:i1 + 1].max() <= trusted_max * 1.05 + 1e-6
    assert info["bulge_clipped"] > 0


def test_fill_refuses_with_too_few_healthy_points():
    """Must decline and say so, not guess from two points."""
    arc, _true, broken = collapsed(2, N - 3)
    filled, info = taper_fill(arc, broken, 2, N - 3, min_healthy=5)
    assert info["reason"], "should have declined"
    assert "healthy" in info["reason"]
    assert np.array_equal(filled, broken), "declining must leave the radii untouched"


def test_only_increase_never_lowers_a_radius():
    arc, _true, broken = collapsed(80, 120)
    # Inflate part of the span well above the trend.
    broken[100:110] = 5000.0
    filled, _ = taper_fill(arc, broken, 80, 120, only_increase=True)
    assert np.all(filled[80:121] >= broken[80:121] - 1e-9)
    assert filled[105] == pytest.approx(5000.0)


def test_only_increase_off_lets_the_fit_pull_an_inflated_span_down():
    arc, true_r, broken = collapsed(80, 120)
    broken[100:110] = 5000.0
    filled, _ = taper_fill(arc, broken, 80, 120, only_increase=False)
    assert filled[105] < 5000.0
    assert filled[105] == pytest.approx(true_r[105], rel=0.05)


def test_the_slope_clamp_binds_on_an_implausible_taper():
    """A steep short window must not extrapolate to nonsense over a long gap."""
    arc = np.arange(N, dtype=float) * SPACING_UM
    rad = np.full(N, 300.0)
    # A violently falling proximal window: 5x over 40 points (2 mm) = -0.8/mm.
    rad[:40] = np.linspace(1500.0, 300.0, 40)
    rad[60:180] = 1.0  # the "collapse"

    loose, info_loose = taper_fill(arc, rad, 60, 179, max_taper_per_mm=5.0,
                                   only_increase=False)
    tight, info_tight = taper_fill(arc, rad, 60, 179, max_taper_per_mm=0.05,
                                   only_increase=False)
    assert abs(info_tight["slope_per_mm"]) < abs(info_loose["slope_per_mm"])
    assert abs(info_tight["slope_per_mm"]) <= 0.05 + 1e-9
    assert np.all(np.isfinite(tight)) and np.all(tight > 0)


def test_a_short_fitting_window_cannot_bulge_the_middle_of_a_gap():
    """The lever-arm failure: little evidence, extrapolated a long way.

    Six healthy points spanning 0.3 mm give a badly-determined slope; carried
    across a 2 mm gap it swells the middle far above both ends. r-squared does not
    catch this, because the problem is the lever arm and not the scatter.
    """
    from hipct_seg_debug.edit.radius_repair import MAX_BULGE

    arc = np.arange(N, dtype=float) * SPACING_UM
    rad = np.full(N, 400.0)
    rad[:6] = np.linspace(360.0, 400.0, 6)   # a steep, short proximal window
    rad[6:46] = 80.0                          # the collapse
    filled, info = taper_fill(arc, rad, 6, 45, min_healthy=5)

    ends = max(rad[5], rad[46])
    assert filled[6:46].max() <= ends * MAX_BULGE + 1e-6, (
        f"the fill bulged to {filled[6:46].max():.0f} um between ends of "
        f"{rad[5]:.0f} and {rad[46]:.0f}"
    )
    assert info["bulge_clipped"] > 0, "the guard should have reported clipping"


def test_the_bulge_cap_does_not_touch_a_well_determined_taper():
    i0, i1 = 80, 120
    arc, true_r, broken = collapsed(i0, i1)
    filled, info = taper_fill(arc, broken, i0, i1)
    assert info.get("bulge_clipped", 0) == 0
    assert np.allclose(filled[i0:i1 + 1], true_r[i0:i1 + 1], rtol=0.02)


def test_fill_never_produces_a_negative_or_zero_radius():
    """The reason the fit is in log space at all."""
    arc = np.arange(N, dtype=float) * SPACING_UM
    rad = np.concatenate([np.linspace(800.0, 50.0, 40), np.full(N - 40, 1.0)])
    filled, _ = taper_fill(arc, rad, 40, N - 1, max_taper_per_mm=5.0, only_increase=False)
    assert np.all(filled > 0), "a linear fit would have gone negative here"


# ---------------------------------------------------------------- detection

def make_graph(radii, spacing=SPACING_UM) -> EditableGraph:
    n = len(radii)
    nodes = {0: (0.0, 0.0, 0.0, 1), 1: (0.0, 0.0, float((n - 1) * spacing), 1)}
    points = {
        i: (0.0, 0.0, float(i * spacing), float(radii[i])) for i in range(n)
    }
    segments = [{"id": 0, "node1": 0, "node2": 1, "point_ids": list(range(n))}]
    g = EditableGraph(Triple(nodes=nodes, points=points, segments=segments))
    g._flush_degrees()
    return g


def test_outlier_detector_finds_one_span_with_the_right_bounds():
    _arc, _true, broken = collapsed(80, 120)
    spans = find_outlier_spans(make_graph(broken))
    assert len(spans) == 1
    span = spans[0]
    assert span.source == "outlier"
    # Bounds within a couple of points of the truth: the trend test is a threshold,
    # so the very edges of the dip may sit either side of it.
    assert abs(span.i0 - 80) <= 3 and abs(span.i1 - 120) <= 3
    assert span.length_um == pytest.approx(40 * SPACING_UM, rel=0.15)


def test_outlier_detector_finds_nothing_on_a_clean_taper():
    _arc, rad = truth()
    assert find_outlier_spans(make_graph(rad)) == []


def test_outlier_detector_is_not_fooled_by_a_half_collapsed_segment():
    """An ordinary least-squares trend would conclude the collapse is normal."""
    _arc, _true, broken = collapsed(100, N - 1)
    spans = find_outlier_spans(make_graph(broken))
    assert spans, "the robust re-fit should still see the second half as collapsed"
    assert spans[0].i0 < 115


def test_high_outlier_detector_uses_the_lower_healthy_half():
    _arc, rad = truth()
    broken = rad.copy()
    broken[80:121] *= 5.0
    spans = find_outlier_spans(make_graph(broken), mode="high")
    assert len(spans) == 1
    assert spans[0].source == "high_outlier"
    assert abs(spans[0].i0 - 80) <= 3 and abs(spans[0].i1 - 120) <= 3


def test_high_span_is_corrected_only_by_a_decreasing_fill():
    _arc, true_r = truth()
    high = true_r.copy()
    high[80:121] *= 5.0
    span = find_outlier_spans(make_graph(high), mode="high")[0]

    unchanged = make_graph(high)
    report = fill_span(unchanged, span, only_increase=True)
    assert not report.applied
    assert np.allclose(unchanged.radii(0), high)

    repaired = make_graph(high)
    report = fill_span(repaired, span, only_increase=False)
    assert report.applied
    assert repaired.radii(0)[100] == pytest.approx(true_r[100], rel=0.05)


def test_span_around_grows_from_a_single_pick():
    _arc, _true, broken = collapsed(80, 120)
    g = make_graph(broken)
    span = span_around(g, 0, 100)
    assert span is not None and span.source == "manual"
    assert abs(span.i0 - 80) <= 3 and abs(span.i1 - 120) <= 3


def test_span_around_returns_none_on_healthy_tissue():
    _arc, _true, broken = collapsed(80, 120)
    assert span_around(make_graph(broken), 0, 10) is None


# ------------------------------------------------------- the junction margin

def test_a_span_entirely_at_a_segment_end_is_dropped():
    """A lumen at a bifurcation is legitimately non-circular, not collapsed.

    On the real graph 15 of 83 image-detected collapse runs began at point 0 of
    their segment -- i.e. exactly on a junction. Filling those would inflate a
    carina on no evidence.
    """
    from hipct_seg_debug.edit.radius_repair import JUNCTION_MARGIN

    _arc, _true, broken = collapsed(0, JUNCTION_MARGIN - 2)
    assert find_outlier_spans(make_graph(broken)) == []


def test_a_span_reaching_the_margin_is_trimmed_not_dropped():
    from hipct_seg_debug.edit.radius_repair import JUNCTION_MARGIN, clip_to_interior

    _arc, _true, broken = collapsed(0, 60)
    spans = find_outlier_spans(make_graph(broken))
    assert len(spans) == 1
    assert spans[0].i0 >= JUNCTION_MARGIN, "the junction end should have been trimmed off"
    assert spans[0].i1 >= 55, "the rest of the run should survive"

    assert clip_to_interior(0, 60, N) == (JUNCTION_MARGIN, 60)
    assert clip_to_interior(0, 2, N) is None
    assert clip_to_interior(N - 3, N - 1, N) is None


def test_the_margin_can_be_disabled():
    _arc, _true, broken = collapsed(0, 4)
    spans = find_outlier_spans(make_graph(broken), margin=0, min_points=3)
    assert len(spans) == 1 and spans[0].i0 == 0


# -------------------------------------------------------------- application

def test_fill_span_applies_to_the_graph_and_is_undoable():
    _arc, true_r, broken = collapsed(80, 120)
    g = make_graph(broken)
    before = g.radii(0).copy()

    report = fill_span(g, CollapsedSpan(seg_id=0, i0=80, i1=120))
    assert report.applied, report.reason
    assert report.n_changed > 30
    assert report.sides == "both"

    after = g.radii(0)
    assert np.allclose(after[80:121], true_r[80:121], rtol=0.02)
    assert np.array_equal(after[:80], before[:80])

    g.undo()
    assert np.array_equal(g.radii(0), before)


def test_fill_span_reports_a_patch_naming_the_segment():
    _arc, _true, broken = collapsed(80, 120)
    g = make_graph(broken)
    fill_span(g, CollapsedSpan(seg_id=0, i0=80, i1=120))
    assert g.last_patch.seg_ids == {0}
    assert g.last_patch.aabb is not None


def test_fill_spans_is_one_undo_step():
    _arc, _true, broken = collapsed(40, 60)
    broken[120:150] *= 0.2
    g = make_graph(broken)
    spans = find_outlier_spans(g)
    assert len(spans) == 2

    reports, patch = fill_spans(g, spans)
    assert all(r.applied for r in reports), [r.reason for r in reports]
    assert len(g.history.labels()) == 1
    assert patch.seg_ids == {0}

    g.undo()
    assert np.allclose(g.radii(0), broken)


def test_adjacent_low_high_and_image_spans_merge_before_anchor_selection():
    _arc, _true, broken = collapsed(70, 90)
    broken[91:101] = 5000.0
    g = make_graph(broken)
    spans = [
        CollapsedSpan(0, 70, 90, "outlier"),
        CollapsedSpan(0, 91, 100, "high_outlier"),
        CollapsedSpan(0, 85, 95, "image"),
    ]
    merged = merge_spans(g, spans)
    assert len(merged) == 1
    assert (merged[0].i0, merged[0].i1) == (70, 100)
    assert set(merged[0].source.split("+")) == {"outlier", "high_outlier", "image"}

    reports, _ = fill_spans(g, spans, only_increase=False)
    assert len(reports) == 1 and reports[0].applied
    assert g.radii(0)[95] < 1000.0


def test_perimeter_provenance_disables_redundant_image_collapse_detection():
    g = make_graph(truth()[1])
    assert not has_perimeter_radii(g)
    g.triple.point_attrs["radius_source"] = {pid: 0 for pid in g.point_order()}
    assert has_perimeter_radii(g)


def test_fill_span_declines_on_a_missing_segment():
    g = make_graph(truth()[1])
    report = fill_span(g, CollapsedSpan(seg_id=99, i0=1, i1=2))
    assert not report.applied and "no longer exists" in report.reason


def test_summarise_reports_both_outcomes():
    _arc, _true, broken = collapsed(80, 120)
    g = make_graph(broken)
    reports, _ = fill_spans(g, [
        CollapsedSpan(seg_id=0, i0=80, i1=120),
        CollapsedSpan(seg_id=0, i0=2, i1=N - 3),  # too few healthy points
    ])
    text = summarise(reports)
    assert "spans filled" in text
    assert "declined" in text
    assert summarise([]) == "no collapsed spans"


def test_arclength_matches_spacing():
    coords = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10) * 50.0])
    arc = arclength(coords)
    assert arc[0] == 0.0
    assert np.allclose(np.diff(arc), 50.0)
