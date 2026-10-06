import copy

import numpy as np
import pytest

from hipct_seg_debug.edit import centreline_refine as cr
from hipct_seg_debug.edit import radius_perimeter as rp
from hipct_seg_debug.edit.dfs_paths import root_paths, path_samples
from hipct_seg_debug.edit.radius_profile import prepare_profile, DFS_INTERPOLATION
from .conftest_geometry import graph_from, make_frame


def tree():
    return graph_from([(100, 500, 300), (500, 530, 300), (1000, 500, 300), (500, 800, 300)],
                      [(0, 1, 25, 50.), (1, 2, 31, 50.), (1, 3, 21, 25.)])


def test_paths_rank_physical_length_and_ignore_edge_orientation():
    g = tree()
    first = root_paths(g, [0])
    assert [p['segments'] for p in first] == [[0, 1], [0, 2]]
    coords, _ = path_samples(g, first[0])
    seg = g.segment(1)
    seg['node1'], seg['node2'] = seg['node2'], seg['node1']
    seg['point_ids'] = seg['point_ids'][::-1]
    other = root_paths(g, [0])
    assert other == first
    np.testing.assert_array_equal(path_samples(g, other[0])[0], coords)


def test_cycle_and_conflicting_roots_are_explicit_errors():
    with pytest.raises(ValueError, match='one root'):
        root_paths(tree(), [0, 2])
    g = graph_from([(0, 0, 0), (100, 0, 0), (50, 100, 0)],
                   [(0, 1, 5, 10.), (1, 2, 5, 10.), (2, 0, 5, 10.)])
    with pytest.raises(ValueError, match='cycle'):
        root_paths(g, [0])


def test_supported_path_moves_shared_node_despite_unsupported_daughter(monkeypatch):
    g = tree()
    before = {sid: g.coords(sid).copy() for sid in g.segment_ids()}
    def sections(graph, sid, *args, **kwargs):
        x = graph.coords(sid).copy()
        if sid == 2:
            return x, np.zeros(len(x)), [], []
        x[:, 1] = 500.
        good = list(range(3, len(x)-3))
        weights = np.zeros(len(x))
        weights[good] = 1.
        return x, weights, good, [50.]*len(good)
    monkeypatch.setattr(cr, '_targets', sections)
    report = cr.refine(g, make_frame((70, 110, 120)), np.ones((70, 110, 120), dtype='uint8'),
                       method='dfs-centroid', root_nodes=[0], max_iterations=2)
    assert abs(g.coords(0)[-1, 1]-500.) < 25.
    assert report.moved_nodes == 1
    assert not report.converged
    assert not report.segments[2]['curve_supported']
    assert report.outside_edges_after == 0
    np.testing.assert_array_equal(g.coords(0)[-1], g.coords(1)[0])
    np.testing.assert_array_equal(g.coords(0)[-1], g.coords(2)[0])
    for sid, endpoint in [(0, 0), (1, -1), (2, -1)]:
        np.testing.assert_array_equal(g.coords(sid)[endpoint], before[sid][endpoint])
        np.testing.assert_array_equal(g.radii(sid), 25. if sid == 2 else 50.)


def stamp(g, sid, trusted_indices):
    for i, pid in enumerate(g.segment(sid)['point_ids']):
        accepted = i in trusted_indices
        g.triple.point_attrs.setdefault('radius_source', {})[pid] = rp.PERIMETER if accepted else rp.FILLED
        g.triple.point_attrs.setdefault('radius_reject_reason', {})[pid] = rp.ACCEPTED if accepted else rp.JUNCTION


def test_path_interpolates_merged_junction_without_inflating_daughter():
    g = tree()
    for sid in g.segment_ids():
        ids = list(range(len(g.coords(sid))))
        good = ids[:-4] if sid == 0 else ids[4:]
        stamp(g, sid, good)
        radius = g.radii(sid).copy()
        radius[[i for i in ids if i not in good]] = 999.
        g.set_segment_radii(sid, radius)
    raw = copy.deepcopy(g.points)
    report = prepare_profile(g, policy='dfs-confidence', spacing_um=10., root_nodes=[0])
    assert report.status == 'prepared'
    for sid in (0, 1):
        np.testing.assert_allclose(g.radii(sid), 50.)
    np.testing.assert_array_equal(g.radii(2), 25.)
    assert report.paths[0]['interpolated_points'] == 8
    assert g.triple.point_attrs['radius_adjustment_reason'][g.segment(0)['point_ids'][-1]] == DFS_INTERPOLATION
    assert g.triple.point_attrs['radius_measured_um'] == {p: v[3] for p, v in raw.items()}


def test_dfs_profile_preserves_supported_narrowing_and_flags_long_missing_link():
    g = graph_from([(0, 0, 0), (100, 0, 0), (1000, 0, 0), (1100, 0, 0)],
                   [(0, 1, 11, 20.), (1, 2, 21, 999.), (2, 3, 11, 30.)])
    stamp(g, 0, range(11))
    stamp(g, 1, [])
    stamp(g, 2, range(11))
    r = g.radii(0).copy()
    r[5] = 10.
    g.set_segment_radii(0, r)
    report = prepare_profile(g, policy='dfs-confidence', root_nodes=[0])
    assert g.radii(0)[5] == 10.
    assert 1 in report.unsupported_segments
    assert report.paths[0]['rejected_spans'][0]['reason'] == 'anchors_too_far_apart'


def test_dfs_cli_options_are_opt_in():
    from hipct_seg_debug.edit.__main__ import build_parser
    parser = build_parser()
    args = parser.parse_args(['refine-centreline', 'input.am', '--method', 'dfs-centroid', '--root-node', '7'])
    assert args.root_node == [7]
    args = parser.parse_args(['prepare-reconstruction', 'input.am', '--radius-profile', 'dfs-confidence'])
    assert args.radius_profile == 'dfs-confidence'


def test_submicron_joint_move_keeps_all_endpoint_records_identical():
    g = tree()
    x = g.coords(0).copy()
    x[-1, 1] += .001  # Default relative allclose used to drop this node update.
    g.set_segment_coords(0, x)
    np.testing.assert_array_equal(g.coords(1)[0], x[-1])
    np.testing.assert_array_equal(g.coords(2)[0], x[-1])
    np.testing.assert_array_equal(g.nodes[1][:3], x[-1])


def test_later_daughter_fit_keeps_previous_path_fixed_outside_junction(monkeypatch):
    from hipct_seg_debug.edit.dfs_refine import fit_paths
    from hipct_seg_debug.crosssection import _PlaneSampler
    from hipct_seg_debug.edit import dfs_refine
    g = tree()
    frame = make_frame((70, 110, 120))
    sampler = _PlaneSampler(np.ones((70, 110, 120), dtype='uint8'), frame)
    observations, scales = {}, {}
    for sid in g.segment_ids():
        x = g.coords(sid).copy()
        x[:, 1] -= 5.
        observations[sid] = x, np.ones(len(x)), list(range(len(x))), [25.]*len(x)
        scales[sid] = np.full(len(x), 25.)
    real_fit, snapshots = dfs_refine.fit_cluster, []
    def capture(*args, **kwargs):
        result = real_fit(*args, **kwargs)
        snapshots.append(({sid: g.coords(sid).copy() for sid in g.segment_ids()}, result))
        return result
    monkeypatch.setattr(dfs_refine, 'fit_cluster', capture)
    fit_paths(g, root_paths(g, [0]), g.segment_ids(), observations, scales,
              sampler, frame, .1, {0, 2, 3})
    assert len(snapshots) == 2
    assert all(r['status'] in ('moving', 'stationary') for _, r in snapshots)
    # The junction the longest path placed is fixed for the daughter, so the earlier
    # path is not released at all -- not even inside a bounded neighbourhood.
    assert set(snapshots[1][1]['spans']) == {2}
    for sid in (0, 1):
        np.testing.assert_array_equal(snapshots[0][0][sid], g.coords(sid))


def test_two_point_link_is_supported_by_sections_on_both_sides(monkeypatch):
    g = graph_from([(100, 500, 300), (450, 530, 300), (500, 530, 300), (950, 500, 300)],
                   [(0, 1, 25, 50.), (1, 2, 2, 50.), (2, 3, 31, 50.)])
    def sections(graph, sid, *args, **kwargs):
        x = graph.coords(sid).copy()
        x[:, 1] = 500.
        good = list(range(3, len(x)-3))
        weights = np.zeros(len(x))
        weights[good] = 1.
        return x, weights, good, [50.]*len(good)
    monkeypatch.setattr(cr, '_targets', sections)
    report = cr.refine(g, make_frame((70, 110, 120)), np.ones((70, 110, 120), dtype='uint8'),
                       method='dfs-centroid', root_nodes=[0], max_iterations=2)
    assert report.moved_nodes == 2
    assert report.segments[1]['curve_supported']
    assert g.coords(1)[:, 1].max() < 510.
    for node, left, right in [(1, 0, 1), (2, 1, 2)]:
        np.testing.assert_array_equal(g.coords(left)[-1], g.coords(right)[0])
        assert np.linalg.norm(np.asarray(g.nodes[node][:3])-g.coords(right)[0]) == 0.


def test_dfs_uses_actual_segmentation_sections_and_preserves_endpoints():
    from .test_centreline_refine import offset_slit
    g, frame, mask = offset_slit(step=2)
    original = g.coords(0).copy()
    report = cr.refine(g, frame, mask, method='dfs-centroid',
                       max_iterations=6, max_samples=12, root_nodes=[0])
    assert np.median(abs(g.coords(0)[10:-10, 1]-250.)) < 12.
    assert report.outside_edges_after == report.outside_edges_before == 0
    np.testing.assert_array_equal(g.coords(0)[[0, -1]], original[[0, -1]])


def test_dfs_does_not_average_conflicting_trusted_join_records():
    from .test_radius_profile import continuation
    g = continuation([10.]*4, [12.]*4, [1]*4, [1]*4)
    before = copy.deepcopy(g.points)
    report = prepare_profile(g, policy='dfs-confidence', root_nodes=[0])
    assert report.status == 'review_required'
    assert report.conflicts[0]['reason'] == 'conflicting_trusted_anchors'
    assert g.points == before


def test_saved_path_order_survives_length_change_and_refuses_stale_topology():
    from hipct_seg_debug.edit.dfs_paths import validate_path_plan
    g = tree()
    plan = root_paths(g, [0])
    x = g.coords(2).copy()
    x[1:-1, 2] += 1000.
    g.set_segment_coords(2, x)
    assert root_paths(g, [0])[0]['terminal'] != plan[0]['terminal']
    assert validate_path_plan(g, plan) == plan
    with pytest.raises(ValueError, match='topology'):
        validate_path_plan(g, plan[:-1])
    with pytest.raises(ValueError, match='roots'):
        validate_path_plan(g, plan, [2])


def test_entire_unmeasurable_short_segment_can_use_path_anchors():
    g = graph_from([(0, 0, 0), (100, 0, 0), (125, 0, 0), (225, 0, 0)],
                   [(0, 1, 11, 20.), (1, 2, 4, 999.), (2, 3, 11, 30.)])
    stamp(g, 0, range(9))
    stamp(g, 1, [])
    stamp(g, 2, range(2, 11))
    report = prepare_profile(g, policy='dfs-confidence', root_nodes=[0])
    assert report.status == 'prepared'
    assert np.all((g.radii(1) >= 20.) & (g.radii(1) <= 30.))
    assert not report.unsupported_segments
    assert g.radii(0)[-1] == g.radii(1)[0]
    assert g.radii(1)[-1] == g.radii(2)[0]


def test_duplicate_joint_observations_are_not_two_independent_sections(monkeypatch):
    g = graph_from([(100, 500, 300), (450, 530, 300), (950, 500, 300)],
                   [(0, 1, 25, 50.), (1, 2, 31, 50.)])
    before = copy.deepcopy(g.points)
    def sections(graph, sid, *args, **kwargs):
        x = graph.coords(sid).copy()
        good = [len(x)-1] if sid == 0 else [0]
        weights = np.zeros(len(x))
        weights[good] = 1.
        return x, weights, good, [50.]
    monkeypatch.setattr(cr, '_targets', sections)
    report = cr.refine(g, make_frame((70, 110, 120)), np.ones((70, 110, 120), dtype='uint8'),
                       method='dfs-centroid', root_nodes=[0], max_iterations=1)
    assert not report.converged
    assert report.neighbourhoods[0]['reason'] == 'path_has_fewer_than_two_distinct_sections'
    assert g.points == before


def test_repeated_failed_regional_prefix_is_attempted_once(monkeypatch):
    from hipct_seg_debug.edit import dfs_refine
    from hipct_seg_debug.crosssection import _PlaneSampler
    g = tree()
    calls, progress = [], []
    def blocked(*args, **kwargs):
        calls.append(1)
        return dict(status='blocked', nodes=[], segments=[0], spans={0: [0, 24]})
    monkeypatch.setattr(dfs_refine, 'fit_cluster', blocked)
    x = g.coords(0)
    frame = make_frame((70, 110, 120))
    dfs_refine.fit_paths(g, root_paths(g, [0]), [0],
        {0: (x, np.ones(len(x)), list(range(len(x))), [50.]*len(x))},
        {0: np.full(len(x), 50.)}, _PlaneSampler(np.ones((70, 110, 120), dtype='uint8'), frame),
        frame, .1, {0, 2, 3}, progress=progress.append)
    assert len(calls) == len(progress) == 1
    assert progress[0]['status'] == 'blocked'


def test_failed_run_is_not_retried_after_an_unrelated_accepted_fit(monkeypatch):
    # Paths by length: 0-1-2-3 (segments 0,1 selected; blocked), 0-5 (segment 4;
    # accepted), then 0-1-2-4, whose selected run is 0,1 again with unchanged inputs.
    # A global revision counter retried it because segment 4 succeeded in between.
    from hipct_seg_debug.edit import dfs_refine
    from hipct_seg_debug.crosssection import _PlaneSampler
    g = graph_from([(100, 500, 300), (300, 500, 300), (500, 500, 300), (1500, 500, 300),
                    (700, 600, 300), (100, 1400, 300)],
                   [(0, 1, 9, 50.), (1, 2, 9, 50.), (2, 3, 9, 50.), (2, 4, 9, 50.), (0, 5, 9, 50.)])
    calls = []
    def fake(graph, movable, spans, *args, **kwargs):
        calls.append(sorted(spans))
        status = 'blocked' if 1 in spans else 'moving'
        return dict(status=status, nodes=sorted(movable), segments=sorted(spans), spans={})
    monkeypatch.setattr(dfs_refine, 'fit_cluster', fake)
    selected = [0, 1, 4]
    obs = {sid: (g.coords(sid), np.ones(9), list(range(9)), [50.]*9) for sid in selected}
    frame = make_frame((70, 150, 160))
    paths = root_paths(g, [0])
    assert [p['terminal'] for p in paths] == [3, 5, 4]
    rows = dfs_refine.fit_paths(g, paths, selected, obs, {s: np.full(9, 50.) for s in selected},
                                _PlaneSampler(np.ones((70, 150, 160), dtype='uint8'), frame),
                                frame, .1, {0, 3, 4, 5})
    assert [r['status'] for r in rows.values()] == ['blocked', 'moving']
    assert len(calls) == 2


@pytest.mark.parametrize('rounds', [0, 4])
# Near the node the daughter is dragged +x with the node; 3-6 samples out, the
# bending term pulls it slightly -x. Each case puts background on the dragged side.
@pytest.mark.parametrize('x, rows, cols, at_node', [
    (495.001, (58, 64), (0, 50), False), (504.999, (55, 60), (51, 120), True)])
def test_grazing_side_approach_is_pinned_instead_of_blocking_the_path(monkeypatch, rounds,
                                                                       x, rows, cols, at_node):
    # The daughter runs 0.001 um short of a voxel boundary with background beyond,
    # the 3041 situation: moving it any distance that way leaves the segmentation.
    from hipct_seg_debug.crosssection import _PlaneSampler
    from hipct_seg_debug.edit import dfs_refine, junction_cluster
    monkeypatch.setattr(junction_cluster, 'PIN_ROUNDS', rounds)
    g = graph_from([(100, 500, 300), (x, 530, 300), (1000, 500, 300), (x, 800, 300)],
                   [(0, 1, 25, 50.), (1, 2, 31, 50.), (1, 3, 21, 25.)])
    frame = make_frame((70, 110, 120))
    mask = np.ones((70, 110, 120), dtype='uint8')
    mask[:, rows[0]:rows[1], cols[0]:cols[1]] = 0
    sampler = _PlaneSampler(mask, frame)
    before = {sid: g.coords(sid).copy() for sid in g.segment_ids()}
    observations, scales = {}, {}
    for sid in g.segment_ids():
        target = g.coords(sid).copy()
        if sid != 2:
            target[:, 0] += 20.  # the through path wants its shared node further in +x
        observations[sid] = target, np.ones(len(target)), list(range(len(target))), [25.]*len(target)
        scales[sid] = np.full(len(target), 25.)
    rows = dfs_refine.fit_paths(g, root_paths(g, [0])[:1], g.segment_ids(), observations,
                                scales, sampler, frame, .1, {0, 2, 3})
    row = rows[0]
    if rounds == 0:
        assert row['status'] == 'blocked' and set(row['blocked_reasons']) == {2}
        assert 'pinned_points' not in row
        detail = row['blocked_detail'][2]
        assert detail['contained_alpha'] is None
        assert detail['exit_near_moved_node'] == at_node
        for sid, x in before.items():
            np.testing.assert_array_equal(g.coords(sid), x)
        return
    assert row['status'] == 'moving'
    pinned = row['pinned_points'][2]
    np.testing.assert_array_equal(g.coords(2)[pinned], before[2][pinned])
    if at_node:
        # Pinning reached the shared node: it is reported, and the node stays put.
        assert row['pinned_nodes'] == [1]
        np.testing.assert_array_equal(g.coords(0)[-1], before[0][-1])
    else:
        assert set(row['pinned_points']) == {2} and row['pinned_nodes'] == []
        assert g.coords(0)[-1, 0] > before[0][-1, 0] + 1.
    for sid in g.segment_ids():
        assert not cr.bad_edges(g.coords(sid), sampler, frame).any()


def test_region_with_an_unsupported_segment_can_still_report_settled_geometry(monkeypatch):
    # `converged` needs every segment supported, which segment 2 (no sections) never
    # is; `supported_stable` says the supported rest has stopped moving.
    g = tree()
    def sections(graph, sid, *args, **kwargs):
        x = graph.coords(sid).copy()
        if sid == 2:
            return x, np.zeros(len(x)), [], []
        x[:, 1] = 500.
        good = list(range(3, len(x)-3))
        weights = np.zeros(len(x))
        weights[good] = 1.
        return x, weights, good, [50.]*len(good)
    monkeypatch.setattr(cr, '_targets', sections)
    report = cr.refine(g, make_frame((70, 110, 120)), np.ones((70, 110, 120), dtype='uint8'),
                       method='dfs-centroid', root_nodes=[0], max_iterations=12)
    assert not report.converged
    assert report.unsupported_segments == [2]
    assert report.supported_stable
    last = report.history[-1]
    assert last['supported_max_move_um'] < 1. and last['unsupported_segments'] == 1
    assert {'median_move_um', 'peak_segment', 'pinned_points'} <= set(last)


def test_later_path_attaches_to_a_junction_placed_by_an_earlier_fit(monkeypatch):
    # Region 318: two fits both treated node 636 as movable and pulled it to different
    # places every iteration. The daughter path must see node 1 as fixed once the
    # longest path has placed it.
    from hipct_seg_debug.edit import dfs_refine
    from hipct_seg_debug.crosssection import _PlaneSampler
    g = tree()
    calls = []
    def fake(graph, movable, spans, *args, **kwargs):
        calls.append((sorted(movable), sorted(spans)))
        return dict(status='moving', nodes=sorted(movable), segments=sorted(spans), spans={})
    monkeypatch.setattr(dfs_refine, 'fit_cluster', fake)
    obs = {sid: (g.coords(sid), np.ones(len(g.coords(sid))), list(range(len(g.coords(sid)))),
                 [50.]*len(g.coords(sid))) for sid in g.segment_ids()}
    frame = make_frame((70, 110, 120))
    dfs_refine.fit_paths(g, root_paths(g, [0]), g.segment_ids(), obs,
                         {s: np.full(len(g.coords(s)), 50.) for s in g.segment_ids()},
                         _PlaneSampler(np.ones((70, 110, 120), dtype='uint8'), frame),
                         frame, .1, {0, 2, 3})
    assert calls[0][0] == [1]           # the longest path moves the junction
    assert calls[1] == ([], [2])        # the daughter fits only itself, node 1 fixed


@pytest.mark.parametrize('rounds', [0, 4])
def test_partial_step_pins_what_exits_at_the_full_step(monkeypatch, rounds):
    # Region 318, segment 2699: the contained optimum was ~80 um away, the line search
    # accepted 1/16..1/128 of it each iteration, and pinning never ran because the
    # search never failed outright. Here daughter vertex 4 may not move more than
    # 1 um: half a step passes, the full step does not.
    from hipct_seg_debug.crosssection import _PlaneSampler
    from hipct_seg_debug.edit import dfs_refine, junction_cluster
    monkeypatch.setattr(junction_cluster, 'PIN_ROUNDS', rounds)
    g = tree()
    frame = make_frame((70, 110, 120))
    fixed_point = g.coords(2)[4].copy()
    def bad_edges(x, sampler, frame):
        out = np.zeros(len(x)-1, dtype=bool)
        if len(x) == len(g.coords(2)) and np.allclose(x[-1], g.coords(2)[-1]):
            if np.linalg.norm(x[4]-fixed_point) > 1.:
                out[3:5] = True
        return out
    def rejection(old, new, sampler, frame, *, original_bad=None):
        return 'new_segmentation_exit' if bad_edges(new, sampler, frame).any() else None
    monkeypatch.setattr(cr, 'bad_edges', bad_edges)
    monkeypatch.setattr(cr, 'movement_rejection', rejection)
    observations, scales = {}, {}
    for sid in g.segment_ids():
        target = g.coords(sid).copy()
        if sid != 2:
            target[:, 0] += 60.
        observations[sid] = target, np.ones(len(target)), list(range(len(target))), [25.]*len(target)
        scales[sid] = np.full(len(target), 25.)
    node_before = g.coords(0)[-1].copy()
    [row] = dfs_refine.fit_paths(g, root_paths(g, [0])[:1], g.segment_ids(), observations, scales,
                                 _PlaneSampler(np.ones((70, 110, 120), dtype='uint8'), frame),
                                 frame, .1, {0, 2, 3}).values()
    assert row['status'] == 'moving'
    if rounds == 0:
        assert row['step'] < 1. and 'pinned_points' not in row
    else:
        assert row['step'] == 1. and set(row['pinned_points']) == {2}
        assert 4 in row['pinned_points'][2] and row['pinned_nodes'] == []
        np.testing.assert_array_equal(g.coords(2)[4], fixed_point)
    assert np.linalg.norm(g.coords(0)[-1]-node_before) > 1.


def test_data_less_segment_borrows_neighbour_calibre_for_the_fit_only():
    # 905 on region 3612/3655: a spur into a bulge with stored radius 1.67 mm and no
    # sections dominated the joint fit through calibre**4.
    g = tree()
    scale = {0: np.full(25, 50.), 1: np.full(31, 60.), 2: np.full(21, 500.)}
    fit = cr._fit_calibre(g, [0, 1, 2], scale, measured={0, 1})
    np.testing.assert_array_equal(fit[2], np.full(21, 55.))
    np.testing.assert_array_equal(fit[0], scale[0])
    assert scale[2][0] == 500.  # section sampling keeps the stored scale
    lonely = cr._fit_calibre(g, [0, 1, 2], scale, measured=set())
    np.testing.assert_array_equal(lonely[2], scale[2])
