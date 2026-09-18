import numpy as np

from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.radius_profile import prepare_profile
from hipct_seg_debug.edit import radius_perimeter as rp


def continuation(r1, r2, t1, t2):
    points = {i: (float(i*10), 0., 0., r) for i, r in enumerate(r1)}
    offset = len(points)
    points.update({offset+i: (float((offset-1+i)*10), 0., 0., r) for i, r in enumerate(r2)})
    ids = [list(range(offset)), list(range(offset, offset+len(r2)))]
    nodes = {0: (*points[0][:3], 0), 1: (*points[offset-1][:3], 0),
             2: (*points[offset+len(r2)-1][:3], 0)}
    segments = [dict(id=i, node1=i, node2=i+1, point_ids=p) for i, p in enumerate(ids)]
    graph = EditableGraph(Triple(nodes, points, segments))
    trust = dict(zip(list(points), list(t1)+list(t2)))
    graph.triple.point_attrs['radius_source'] = {p: rp.PERIMETER if t else rp.FILLED for p, t in trust.items()}
    graph.triple.point_attrs['radius_reject_reason'] = {p: rp.ACCEPTED if t else rp.JUNCTION for p, t in trust.items()}
    return graph


def test_unsupported_step_is_joined_without_changing_trusted_anchors():
    graph = continuation([10., 10., 99., 99.], [1., 1., 20., 20.],
                         [1, 1, 0, 0], [0, 0, 1, 1])
    before = dict(graph.points)
    report = prepare_profile(graph, policy='confidence', spacing_um=1)
    assert report.changed_points == 4
    assert graph.radii(0)[-1] == graph.radii(1)[0]
    assert all(10 <= p[3] <= 20 for p in graph.points.values())
    assert graph.radii(0)[:2].tolist() == [10., 10.]
    assert graph.radii(1)[-2:].tolist() == [20., 20.]
    assert graph.triple.point_attrs['radius_measured_um'] == {p: v[3] for p, v in before.items()}
    assert all(graph.points[p][:3] == before[p][:3] for p in before)
    assert not report.conflicts


def test_trusted_step_is_flagged_and_not_averaged():
    graph = continuation([10.]*4, [12.]*4, [1]*4, [1]*4)
    raw = dict(graph.points)
    report = prepare_profile(graph, policy='confidence')
    assert report.status == 'review_required'
    assert report.conflicts[0]['reason'] == 'conflicting_trusted_anchors'
    assert graph.points == raw


def test_supported_narrowing_is_preserved_and_no_extrapolation_occurs():
    graph = continuation([50., 20., 10., 20.], [20., 10., 20., 90.],
                         [0, 1, 1, 1], [1, 1, 1, 0])
    raw = dict(graph.points)
    prepare_profile(graph, policy='confidence')
    assert graph.points == raw


def test_preserve_policy_keeps_all_radii_exact():
    graph = continuation([10., 10., 99.], [1., 1., 20.], [1, 0, 0], [0, 0, 1])
    raw = dict(graph.points)
    report = prepare_profile(graph)
    assert report.changed_points == 0 and graph.points == raw


def test_partial_remeasurement_keeps_other_segment_provenance_and_mean_radius():
    graph = continuation([10.]*4, [12.]*4, [1]*4, [1]*4)
    result = rp.RadiusResult(radii={0: np.full(4, 11.)}, source={0: np.full(4, rp.AREA)},
                             reject_reason={0: np.full(4, rp.ACCEPTED)},
                             resolution_mode={0: np.full(4, rp.DIRECT_PLANE)})
    rp.apply_radii(graph, result)
    assert graph.radii(1).tolist() == [12.]*4
    assert all(graph.triple.point_attrs['radius_source'][p] == rp.PERIMETER
               for p in graph.segment(1)['point_ids'])


def test_junction_extension_uses_each_branch_own_calibre():
    from .conftest_geometry import graph_from
    graph = graph_from([(0, 0, 0), (1000, 0, 0), (0, 1000, 0), (0, 0, 1000)],
                       [(0, 1, 12, 100.), (0, 2, 12, 150.), (0, 3, 12, 200.)])
    for sid in graph.segment_ids():
        ids = graph.segment(sid)['point_ids']
        for i, pid in enumerate(ids):
            graph.triple.point_attrs.setdefault('radius_source', {})[pid] = rp.FILLED if i < 2 else rp.PERIMETER
            graph.triple.point_attrs.setdefault('radius_reject_reason', {})[pid] = rp.JUNCTION if i < 2 else rp.ACCEPTED
            if i < 2:
                graph.set_radius(pid, 999.)
    report = prepare_profile(graph, policy='confidence', spacing_um=10.)
    assert report.status == 'prepared'
    for sid, r in enumerate([100., 150., 200.]):
        np.testing.assert_array_equal(graph.radii(sid), r)


def test_already_correct_unsupported_span_is_recognised_and_preparation_is_repeatable():
    graph = continuation([10.]*4, [10.]*4, [1, 1, 0, 0], [0, 0, 1, 1])
    first = prepare_profile(graph, policy='confidence').to_dict()
    attrs = {name: values.copy() for name, values in graph.triple.point_attrs.items()}
    second = prepare_profile(graph, policy='confidence').to_dict()
    assert first['status'] == 'prepared' and not first['unsupported_segments']
    assert second == first
    assert graph.triple.point_attrs == attrs
