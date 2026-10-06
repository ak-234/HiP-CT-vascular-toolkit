import numpy as np
import pytest

from hipct_seg_debug.crosssection import _PlaneSampler
from hipct_seg_debug.edit.junction_refine import refine_neighbourhoods
from .conftest_geometry import make_frame, graph_from


def test_degree_two_combined_fit_removes_kink_with_outer_anchors_fixed():
    frame = make_frame((20, 40, 80))
    graph = graph_from([(50, 150, 100), (400, 180, 100), (750, 150, 100)],
                       [(0, 1, 21, 50.), (1, 2, 21, 50.)])
    old = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    observations, scales = {}, {}
    for sid in graph.segment_ids():
        target = old[sid].copy()
        target[:, 1] = 150.
        observations[sid] = (target, np.ones(len(target)), list(range(len(target))), [])
        scales[sid] = np.full(len(target), 50.)
    report = refine_neighbourhoods(graph, {1: np.array([400., 150., 100.])}, observations,
                                  scales, _PlaneSampler(np.ones((20, 40, 80), dtype='uint8'), frame),
                                  frame, .01)
    assert report[1]['objective_after'] < report[1]['objective_before']
    a, b = graph.coords(0), graph.coords(1)
    np.testing.assert_array_equal(a[-1], b[0])
    np.testing.assert_array_equal(a[0], old[0][0])
    np.testing.assert_array_equal(b[-1], old[1][-1])
    ua, ub = a[-1]-a[-2], b[1]-b[0]
    cosine = ua@ub/(np.linalg.norm(ua)*np.linalg.norm(ub))
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 1.


@pytest.mark.parametrize('degree', [3, 4, 5, 6])
def test_joint_branch_fit_preserves_individual_directions_and_outer_anchors(degree):
    centre = np.array([500., 500., 300.])
    theta = np.arange(degree)*2*np.pi/degree
    directions = np.c_[np.cos(theta), np.sin(theta), np.zeros(degree)]
    nodes = np.vstack([centre, centre+350*directions])
    edges = [(0, i+1, 25, 35.+i*5.) for i in range(degree)]
    truth = graph_from(nodes, edges)
    nodes[0, 1] += 35.
    graph = graph_from(nodes, edges)
    old = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    radii = {sid: graph.radii(sid).copy() for sid in graph.segment_ids()}
    observations, scales = {}, {}
    for sid in graph.segment_ids():
        # Uneven sampling is represented physically in both observed curves.
        t = np.linspace(0, 1, 25)**1.4
        x = old[sid][0]+t[:, None]*(old[sid][-1]-old[sid][0])
        graph.set_segment_coords(sid, x)
        old[sid] = x.copy()
        target = truth.coords(sid)[0]+t[:, None]*(truth.coords(sid)[-1]-truth.coords(sid)[0])
        w = np.ones(25)
        w[:5] = 0
        observations[sid] = (target, w, list(range(5, 25)), [])
        scales[sid] = radii[sid]
    frame = make_frame((70, 110, 110))
    report = refine_neighbourhoods(graph, {0: centre}, observations, scales,
        _PlaneSampler(np.ones((70, 110, 110), dtype='uint8'), frame), frame, .01)
    assert report[0]['objective_after'] < report[0]['objective_before']
    assert np.linalg.norm(graph.coords(0)[0]-centre) < 35.
    for sid in graph.segment_ids():
        np.testing.assert_array_equal(graph.coords(sid)[-2:], old[sid][-2:])
        np.testing.assert_array_equal(graph.coords(sid)[0], graph.coords(0)[0])
        np.testing.assert_array_equal(graph.radii(sid), radii[sid])
        direction = graph.coords(sid)[1]-graph.coords(sid)[0]
        assert direction@directions[sid]/np.linalg.norm(direction) > .9


def test_overlapping_junctions_jointly_support_an_unmeasurable_internal_link():
    nodes = np.array([[350., 400., 200.], [450., 400., 200.],
                      [100., 250., 200.], [100., 550., 200.],
                      [700., 250., 200.], [700., 550., 200.]])
    edges = [(0, 1, 7, 40.), (0, 2, 25, 40.), (0, 3, 25, 35.),
             (1, 4, 25, 40.), (1, 5, 25, 30.)]
    truth = graph_from(nodes, edges)
    nodes[:2, 1] += 30.
    graph = graph_from(nodes, edges)
    observations, scales = {}, {}
    for sid in graph.segment_ids():
        w = np.ones(len(graph.coords(sid)))
        w[:5] = 0
        if sid == 0:
            w[:] = 0
        observations[sid] = (truth.coords(sid), w, np.flatnonzero(w).tolist(), [])
        scales[sid] = graph.radii(sid)
    frame = make_frame((50, 90, 90))
    old = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    report = refine_neighbourhoods(graph, {0: nodes[0], 1: nodes[1]}, observations, scales,
        _PlaneSampler(np.ones((50, 90, 90), dtype='uint8'), frame), frame, .01)
    assert report[0] == report[1]
    assert report[0]['nodes'] == [0, 1]
    assert not report[0]['unsupported_approaches']
    assert report[0]['objective_after'] < report[0]['objective_before']
    assert np.max(np.abs(graph.coords(0)[:, 1]-400)) < 30.
    for sid in (1, 2, 3, 4):
        np.testing.assert_array_equal(graph.coords(sid)[-2:], old[sid][-2:])
