"""Closed transverse lumen measurements must not inherit an incorrect input radius."""
import numpy as np

from hipct_seg_debug import crosssection as cs
from hipct_seg_debug.edit import radius_perimeter as rp

from .conftest_geometry import cylinder, graph_from, make_frame


def test_offset_centreline_can_measure_the_full_lumen_without_moving_it():
    shape = (60, 60, 80)
    frame = make_frame(shape)
    mask = np.maximum(cylinder(shape, 4, 5, 75, cy=12, cz=30),
                      cylinder(shape, 8, 5, 75, cy=40, cz=30))
    xyz = frame.seg_to_um([[8, 12, 30], [72, 12, 30],
                          [32, 46, 30], [48, 46, 30]])
    graph = graph_from(xyz, [(0, 1, 64, 40.), (2, 3, 16, 20.)])
    before = graph.coords(1).copy()
    result = rp.measure_radii(graph, frame, mask)
    np.testing.assert_array_equal(graph.coords(1), before)
    assert (result.reject_reason[1] == rp.ACCEPTED).all()
    np.testing.assert_allclose(result.radii[1], 80., rtol=.12)


def test_sustained_measurements_can_correct_an_underestimated_input_by_more_than_two():
    arc = np.arange(30.) * 32.04
    measured = np.full(30, 1100.)
    old = np.full(30, 250.)
    assert not rp._robust_local_high_mask(arc, measured, old).any()
    measured[15] = 3500.
    assert np.flatnonzero(rp._robust_local_high_mask(arc, measured, old)).tolist() == [15]


def test_drift_check_rejects_a_stable_but_oblique_cylinder_cut():
    shape = (100, 100, 160)
    frame = make_frame(shape)
    sampler = cs._PlaneSampler(cylinder(shape, 8, 5, 155), frame)
    tangent = np.array([1., 1., 0.]) / np.sqrt(2.)
    result = cs.stable_transverse_cut(
        sampler, [80., 50., 50.], tangent, 8., spacing_um=10.,
        search_degrees=0., centroid_mode="drift",
    )
    assert result is None


def test_wide_cylinder_does_not_fallback_at_the_old_window_limit():
    shape = (160, 160, 200)
    frame = make_frame(shape)
    mask = cylinder(shape, 70, 5, 195)
    xyz = frame.seg_to_um([[80, 80, 80], [120, 80, 80]])
    graph = graph_from(xyz, [(0, 1, 4, 700.)])
    result = rp.measure_radii(graph, frame, mask)
    assert (result.reject_reason[0] == rp.ACCEPTED).all()
    np.testing.assert_allclose(result.radii[0], 700., rtol=.05)
