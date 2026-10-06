import numpy as np
import pytest

from hipct_seg_debug.crosssection import _PlaneSampler
from hipct_seg_debug.edit import centreline_refine as cr
from hipct_seg_debug.edit import centreline_clearance as cc
from .conftest_geometry import make_frame, slit, axis_graph, graph_from


def offset_slit(step=1):
    shape = (20, 50, 100)
    frame = make_frame(shape)
    mask = slit(shape, 12, 2, 2, 98, cy=25, cz=10)
    g = axis_graph(frame, 5, 95, 15., cy=34, cz=10, step=step)
    return g, frame, mask


def test_line_containment_checks_between_points():
    frame = make_frame((5, 5, 8))
    mask = np.ones((5, 5, 8), dtype=np.uint8)
    mask[:, :, 3] = 0
    sampler = _PlaneSampler(mask, frame)
    points = frame.seg_to_um([[1, 2, 2], [6, 2, 2]])
    assert (sampler.at(frame.um_to_seg(points)) > 0).all()
    assert cr.bad_edges(points, sampler, frame).tolist() == [True]


def test_spline_displacement_preserves_gap_anchors_and_descends():
    from hipct_seg_debug.edit.junction_refine import fit_objective
    s = np.linspace(0, 500, 81)
    x = np.c_[s, 8*np.sin(s/17)+3*np.cos(s/3), np.zeros(len(s))]
    target = x.copy()
    target[:, 1] = 0
    weights = np.ones(len(x))
    pinned = [23, 24, 25]
    result = cr.spline_fit(x, target, weights, 35., 2., fixed_points=pinned)
    np.testing.assert_array_equal(result[[0, *pinned, 80]], x[[0, *pinned, 80]])
    assert np.linalg.norm(result-x) > 1
    assert fit_objective(result, target, weights, 35., x, 2., .1) < fit_objective(
        x, target, weights, 35., x, 2., .1)


def test_preexisting_gap_movement_reports_constraint_not_convergence():
    frame = make_frame((5, 5, 12))
    mask = np.ones((5, 5, 12), dtype=np.uint8)
    mask[:, :, 5] = 0
    x = frame.seg_to_um([[2, 2, 2], [4, 2, 2], [6, 2, 2], [9, 2, 2]])
    changed = x.copy()
    changed[1, 1] += .01
    assert cr.movement_rejection(x, changed, _PlaneSampler(mask, frame), frame) == 'preexisting_gap_anchor_moved'


@pytest.mark.parametrize('unsupported', [True, False])
def test_junction_and_approaches_move_together_or_remain_unresolved(monkeypatch, unsupported):
    frame = make_frame((70, 110, 110))
    graph = graph_from([(500, 530, 300), (100, 500, 300), (900, 500, 300), (500, 900, 300)],
                       [(0, 1, 25, 50.), (0, 2, 25, 50.), (0, 3, 25, 50.)])
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    def observations(g, sid, *args, **kwargs):
        x = g.coords(sid).copy()
        if sid == 2 and unsupported:
            return x, np.zeros(len(x)), [], []
        x[:, 1] = before[sid][:, 1]-30*np.linspace(1, 0, len(x))
        return x, np.ones(len(x)), list(range(len(x))), [50.]*len(x)
    monkeypatch.setattr(cr, '_targets', observations)
    report = cr.refine(graph, frame, np.ones((70, 110, 110), dtype='uint8'),
                       method='centroid-coherent', max_iterations=2)
    if not unsupported:
        assert np.linalg.norm(graph.coords(0)[0]-before[0][0]) > 1.
        for sid in before:
            np.testing.assert_array_equal(graph.coords(sid)[0], graph.coords(0)[0])
            np.testing.assert_array_equal(graph.coords(sid)[-1], before[sid][-1])
            np.testing.assert_array_equal(graph.radii(sid), np.full(25, 50.))
        return
    assert not report.converged
    assert report.neighbourhoods[0]['status'] == 'insufficient_support'
    for sid, old in before.items():
        np.testing.assert_array_equal(graph.coords(sid), old)
        assert report.segments[sid]['blocked_reason'] == 'junction_fit_unresolved'


def test_compressed_section_processes_match_serial_fitting(tmp_path):
    import copy
    from hipct_seg_debug import rle_write, amira, rle
    shape = (20, 80, 100)
    frame = make_frame(shape)
    mask = np.maximum(slit(shape, 7, 2, 2, 98, cy=20, cz=10),
                      slit(shape, 7, 2, 2, 98, cy=60, cz=10))
    xyz = frame.seg_to_um([[5, 24, 10], [95, 24, 10], [5, 64, 10], [95, 64, 10]])
    graph = graph_from(xyz, [(0, 1, 31, 30.), (2, 3, 31, 30.)])
    other = copy.deepcopy(graph)
    path = tmp_path/'slits.am'
    rle_write.write_lattice(path, mask, frame.seg_bbox_um)
    header = amira.read_lattice_header(path)
    labels = rle.open_lattice(path, header.fields['Labels'], header.dims, cache_dir=tmp_path/'cache')
    serial = cr.refine(graph, frame, labels, method='centroid-coherent',
                       max_iterations=2, max_samples=6, workers=1)
    progress, checkpoints = [], []
    parallel = cr.refine(other, frame, labels, method='centroid-coherent',
                         max_iterations=2, max_samples=6, workers=2,
                         section_progress=progress.append, checkpoint=checkpoints.append)
    for sid in graph.segment_ids():
        np.testing.assert_allclose(other.coords(sid), graph.coords(sid), atol=1e-9)
    assert serial.history == parallel.history
    assert len(progress) == 2*parallel.iterations
    assert len(checkpoints) == parallel.iterations


def test_movement_cannot_jump_into_another_lumen():
    frame = make_frame((5, 8, 10))
    mask = np.ones((5, 8, 10), dtype=np.uint8)
    mask[:, 4, :] = 0
    old = frame.seg_to_um([[1, 2, 2], [3, 2, 2], [6, 2, 2]])
    new = old + [0, 40, 0]
    assert not cr.feasible_move(old, new, _PlaneSampler(mask, frame), frame)


def test_offset_slit_corrects_more_than_input_radius_without_altering_it():
    g, frame, mask = offset_slit()
    old = g.coords(0).copy()
    radii = g.radii(0).copy()
    result = cr.refine(g, frame, mask, max_iterations=12, max_samples=20)
    x = g.coords(0)
    assert np.median(abs(x[20:-20, 1]-250)) < 10
    assert result.moved_points > 0
    assert result.outside_edges_after == 0
    np.testing.assert_array_equal(x[[0, -1]], old[[0, -1]])
    np.testing.assert_array_equal(g.radii(0), radii)


def test_spacing_and_coordinate_scale_do_not_change_laplacian_physics():
    s = np.linspace(0, 1000, 51)
    x = np.c_[s, 10*np.sin(s/30), np.zeros(len(s))]
    y = cr.laplacian_fit(x, 40, 10)
    scaled = cr.laplacian_fit(x*3, 120, 30)
    np.testing.assert_allclose(scaled/3, y, atol=1e-9)
    assert np.std(y[:, 1]) < np.std(x[:, 1])
    uneven = np.sort(np.r_[s, s[1:-1]+3])
    xx = np.c_[uneven, 10*np.sin(uneven/30), np.zeros(len(uneven))]
    yy = cr.laplacian_fit(xx, 40, 10)
    np.testing.assert_allclose(np.interp(s, uneven, yy[:, 1]), y[:, 1], atol=.6)


def test_compensation_reduces_shrinkage_on_a_bend():
    t = np.linspace(0, np.pi, 101)
    x = np.c_[100*np.cos(t), 100*np.sin(t), np.zeros(len(t))]
    plain = cr.laplacian_fit(x, 30, 2, strength=.2)
    compensated = cr.laplacian_fit(x, 30, 2, strength=.2, taubin=True)
    assert np.mean(np.linalg.norm(compensated-x, axis=1)) < np.mean(np.linalg.norm(plain-x, axis=1))


def test_worker_count_does_not_change_output():
    a, frame, mask = offset_slit(step=3)
    b, _, _ = offset_slit(step=3)
    cr.refine(a, frame, mask, max_iterations=3, max_samples=10, workers=1)
    cr.refine(b, frame, mask, max_iterations=3, max_samples=10, workers=2)
    np.testing.assert_array_equal(a.coords(0), b.coords(0))


def test_joint_node_moves_once_and_all_incident_endpoints_follow():
    shape = (30, 80, 100)
    frame = make_frame(shape)
    truth = np.array([[50, 40, 15], [8, 40, 15], [90, 15, 15], [90, 65, 15.]])
    zz, yy, xx = np.indices(shape)
    voxels = np.stack([xx, yy, zz], axis=-1)
    mask = np.zeros(shape, dtype=np.uint8)
    for end in truth[1:]:
        d = end-truth[0]
        t = np.clip(np.sum((voxels-truth[0])*d, axis=-1)/(d@d), 0, 1)
        mask |= (np.linalg.norm(voxels-truth[0]-t[..., None]*d, axis=-1) <= 4)
    displaced = truth.copy()
    displaced[0, 1] += 2
    g = graph_from(frame.seg_to_um(displaced), [(0, 1, 40, 40.),
                                               (0, 2, 40, 40.), (0, 3, 40, 40.)])
    before = np.asarray(g.nodes[0][:3])
    result = cr.refine(g, frame, mask, max_iterations=8, max_samples=16)
    node = np.asarray(g.nodes[0][:3])
    assert np.linalg.norm(node-truth[0]*10) < np.linalg.norm(before-truth[0]*10)
    assert result.moved_nodes == 1
    assert g.segment_ids() == [0, 1, 2]
    for sid in g.segment_ids():
        np.testing.assert_allclose(g.coords(sid)[0], node, atol=1e-8)


def test_variable_scale_smooths_large_vessel_noise_more_than_small():
    s = np.linspace(0, 2000, 201)
    x = np.c_[s, np.sin(s/20)*5, np.zeros(len(s))]
    scale = np.where(s < 1000, 20., 100.)
    out = cr.laplacian_fit(x, scale, 5)
    assert np.std(out[120:-10, 1]) < np.std(out[10:80, 1])/2


def collision_fixture(separation=50., mask_half=15.):
    frame = make_frame((50, 50, 100))
    shape = (50, 50, 100)
    mask = np.zeros(shape, dtype=np.uint8)
    g = graph_from([(50, 200, 250), (950, 200, 250),
                    (50, 200+separation, 250), (950, 200+separation, 250)],
                   [(0, 1, 61, 30.), (2, 3, 61, 30.)])
    for sid in g.segment_ids():
        x = g.coords(sid)
        # Only middle spans collide; terminals have sufficient clearance.
        x[:, 1] += (-1 if sid == 0 else 1)*25*(np.cos(np.linspace(0, 2*np.pi, len(x)))+1)
        with g.batch('fixture'):
            g.set_segment_coords(sid, x)
    for sid in g.segment_ids():
        x = g.coords(sid)
        for col in range(5, 96):
            cy = np.interp(col*10, x[:, 0], x[:, 1])/10
            ys = np.arange(50)
            mask[24:27, np.abs(ys-cy) <= mask_half/10, col] = 1
    return g, frame, mask


def test_clearance_preserves_radii_and_reduces_existing_contact_smoothly():
    g, frame, mask = collision_fixture()
    before = {sid: g.coords(sid).copy() for sid in g.segment_ids()}
    radii = {sid: g.radii(sid).copy() for sid in g.segment_ids()}
    first = cc.contacts(g)
    assert first
    report = cc.prepare(g, frame, mask, max_iterations=12)
    after = cc.contacts(g)
    assert not after or min(c.clearance_um for c in after) > min(c.clearance_um for c in first)
    assert report.moved_points > 0
    for sid in g.segment_ids():
        displacement = g.coords(sid)-before[sid]
        np.testing.assert_array_equal(displacement[[0, -1]], 0.)
        np.testing.assert_array_equal(g.radii(sid), radii[sid])
        assert np.max(np.linalg.norm(np.diff(displacement, n=2, axis=0), axis=1)) < 2.
    assert report.surface_validation_required


def test_impossible_clearance_is_not_reported_as_success():
    g, frame, mask = collision_fixture(separation=20., mask_half=5.)
    report = cc.prepare(g, frame, mask, max_displacement_radii=.01, max_iterations=3)
    assert report.status.startswith('unresolved')
    assert report.contacts_after > 0
    assert report.radii_preserved


def test_clearance_separates_actual_circular_tube_meshes():
    import pyvista as pv
    g, frame, mask = collision_fixture()

    def meshes():
        return [pv.lines_from_points(g.coords(sid)).tube(radius=30., n_sides=32,
                                                         capping=True).triangulate()
                for sid in g.segment_ids()]

    a, b = meshes()
    _, before = a.collision(b, contact_mode=0)
    assert before > 0
    report = cc.prepare(g, frame, mask)
    a, b = meshes()
    _, after = a.collision(b, contact_mode=0)
    assert report.contacts_after == 0
    assert after == 0


def test_tight_local_bend_is_reported_even_without_nonlocal_contact():
    t = np.linspace(0, np.pi, 41)
    x = np.c_[10*np.cos(t), 10*np.sin(t), np.zeros(len(t))]
    assert cc.tight_bends(x, np.full(len(t), 20.)).all()


def test_parallel_continuation_does_not_count_as_a_crossing_section():
    from hipct_seg_debug.crosssection import cut
    shape = (20, 40, 100)
    frame = make_frame(shape)
    mask = slit(shape, 5, 2, 2, 98, cy=20, cz=10)
    g = graph_from(frame.seg_to_um([[5, 20, 10], [50, 20, 10], [95, 20, 10]]),
                   [(0, 1, 30, 100.), (1, 2, 30, 100.)])
    origin = frame.seg_to_um([45, 20, 10])[0]
    c = cut(_PlaneSampler(mask, frame), [45, 20, 10], [1, 0, 0], 12)
    assert not cr._crosses_section(g, 1, origin, np.array([1., 0, 0]), c, 10.)


def test_bump_has_smooth_compact_support():
    s = np.linspace(-2, 2, 4001)
    f = cc._bump(s, 0, 1)
    assert (f[np.abs(s) >= 1] == 0).all()
    assert abs(np.gradient(f, s)[1000]) < 1e-4
    assert abs(np.gradient(np.gradient(f, s), s)[1000]) < .04


def test_cli_exposes_separate_opt_in_stages():
    from hipct_seg_debug.edit.__main__ import build_parser
    p = build_parser()
    args = p.parse_args(['refine-centreline', 'a.am', '--method', 'laplacian'])
    assert not args.geometry_only
    args = p.parse_args(['prepare-reconstruction', 'a.am'])
    assert args.max_displacement_radii == .5


@pytest.mark.parametrize('kw', [dict(workers=0), dict(strength=-1), dict(max_samples=2)])
def test_invalid_settings_are_refused(kw):
    g, frame, mask = offset_slit()
    with pytest.raises(ValueError):
        cr.refine(g, frame, mask, **kw)


def test_support_is_compared_by_site_not_point_index():
    # The same two places accepted, landing on different point indices after a move.
    assert cr._support_signature([10, 20], [0, 10, 20, 30]) == cr._support_signature(
        [11, 21], [0, 11, 21, 30])
    assert cr._support_signature([10], [0, 10, 20, 30]) != cr._support_signature(
        [20], [0, 10, 20, 30])


def test_held_site_keeps_its_result_until_it_moves_a_voxel():
    x = np.c_[np.arange(10.)*10, np.zeros(10), np.zeros(10)]
    sites, memory = np.array([0, 5, 9]), {}
    target = x.copy()
    target[5, 1] = 3.
    weights = np.zeros(10)
    weights[5] = .8
    first, held, fresh = cr._hold_sections(7, x, (target, weights, [5], [40.]), sites, memory, 10.)
    assert (held, fresh) == (0, 3) and first[2] == [5]
    # Next iteration: site 5 moved 4 um (< 10) and is now rejected; the old result stands.
    moved = x + [0., 4., 0.]
    obs, held, fresh = cr._hold_sections(7, moved, (moved.copy(), np.zeros(10), [], []),
                                         sites, memory, 10.)
    assert (held, fresh) == (3, 0) and obs[2] == [5] and obs[3] == [40.]
    np.testing.assert_array_equal(obs[0][5], target[5])
    assert obs[1][5] == .8
    # The curve slid 6 um along itself: the site's index now lands on point 4's
    # neighbour side, but the held target stays on the point nearest its anchor.
    slid = x + [6., 0., 0.]
    obs, held, fresh = cr._hold_sections(7, slid, (slid.copy(), np.zeros(10), [], []),
                                         sites, memory, 10.)
    assert obs[2] == [4] and obs[1][4] == .8 and obs[1][5] == 0.
    np.testing.assert_array_equal(obs[0][4], target[5])
    assert cr._support_signature(obs[2], sites) == cr._support_signature([5], sites)
    # Moved 12 um from where it was measured: re-measured, and the rejection counts.
    far = x + [0., 12., 0.]
    obs, held, fresh = cr._hold_sections(7, far, (far.copy(), np.zeros(10), [], []),
                                         sites, memory, 10.)
    assert fresh == 3 and obs[2] == []


@pytest.mark.parametrize('hold', [0., 1.])
def test_flip_flopping_section_on_a_still_curve_changes_support_only_without_hold(
        monkeypatch, hold):
    g, frame, mask = offset_slit()
    g.set_segment_coords(0, g.coords(0) - [0., 90., 0.])  # already centred: nothing to fit
    calls = []
    def sections(graph, sid, frame, sampler, ctx, scale, max_half, max_samples, *a, **k):
        x = graph.coords(sid)
        sites = cr._section_sites(x, max_samples)
        good = [int(i) for i in sites[1:-1]]
        if len(calls) % 2:
            good = good[1:]  # the second site is rejected every other iteration
        calls.append(1)
        weights = np.zeros(len(x))
        weights[good] = 1.
        return x.copy(), weights, good, [15.]*len(good)
    monkeypatch.setattr(cr, '_targets', sections)
    report = cr.refine(g, frame, mask, method='centroid-spline', max_iterations=5,
                       max_samples=8, section_hold_voxels=hold)
    changes = [h['changed_support_segments'] for h in report.history]
    if hold:
        assert changes == [0]*len(changes)
        assert report.history[-1]['held_sections'] > 0
    else:
        assert sum(changes) > 0


def test_shape_change_ignores_sliding_but_sees_real_movement():
    line = np.c_[np.linspace(0., 100., 11), np.zeros(11), np.zeros(11)]
    slid = line.copy()
    slid[1:-1, 0] += 4.  # interior points re-spaced along the same straight curve
    assert cr._shape_change(line, slid) < 1e-9
    assert np.linalg.norm(slid-line, axis=1).max() == pytest.approx(4.)
    assert cr._shape_change(line, line+[0., 3., 0.]) == pytest.approx(3.)
    shorter = line.copy()
    shorter[-1, 0] = 80.  # an end pulled back along the curve; point 9 still reaches 90
    assert cr._shape_change(line, shorter) == pytest.approx(10.)
