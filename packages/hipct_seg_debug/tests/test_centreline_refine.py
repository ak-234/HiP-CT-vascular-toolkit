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
