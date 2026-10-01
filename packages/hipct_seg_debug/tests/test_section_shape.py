from types import SimpleNamespace

import numpy as np

from hipct_seg_debug.edit.section_shape import section_shape, centring_summary, join_diagnostics
from hipct_seg_debug.edit import centreline_refine as cr
from .conftest_geometry import make_frame, slit, axis_graph, graph_from


def test_flat_ellipse_retains_shape_and_centres():
    u, v = np.indices((101, 101))
    mask = ((u-55)/30)**2 + ((v-47)/5)**2 <= 1
    row = section_shape(SimpleNamespace(blob8=mask, half=50), 2.)
    np.testing.assert_allclose(row['centroid_uv_um'], [10, -6], atol=.2)
    assert row['axis_ratio'] > 5
    assert row['ellipse_centroid_disagreement_um'] < .5
    assert row['ellipse_radial_rms'] < .1
    row.update(accepted=True, point=8)
    assert centring_summary([row], 2.)['status'] == 'off_centre'


def test_concave_centroid_outside_is_identified():
    mask = np.zeros((41, 41), dtype=bool)
    mask[5:36, 5:10] = True
    mask[5:10, 5:36] = True
    mask[31:36, 5:36] = True
    row = section_shape(SimpleNamespace(blob8=mask, half=20), 1.)
    assert not row['centroid_inside']
    assert centring_summary([], 1.)['status'] == 'insufficient_support'


def test_refinement_recentres_flattened_lumen_without_moving_endpoints():
    shape = (20, 50, 100)
    frame = make_frame(shape)
    mask = slit(shape, 12, 2, 2, 98, cy=25, cz=10)
    g = axis_graph(frame, 5, 95, 15., cy=25, cz=10)
    original = g.coords(0).copy()
    x = original.copy()
    x[:, 1] += 60*np.sin(np.linspace(0, np.pi, len(x)))**2
    g.set_segment_coords(0, x)
    r = cr.refine(g, frame, mask, method='dfs-centroid-shape', strength=.01,
                  max_iterations=5, max_samples=16, max_half=64)
    assert np.median(np.abs(g.coords(0)[:, 1]-250)) < 10
    assert r.outside_edges_after == 0
    np.testing.assert_array_equal(g.coords(0)[[0, -1]], original[[0, -1]])
    np.testing.assert_array_equal(g.radii(0), np.full(len(x), 15.))
    assert 'centring_final' in r.segments[0]


def test_fixed_offcentre_endpoints_are_not_certified_as_centred():
    shape = (20, 50, 100)
    frame = make_frame(shape)
    mask = slit(shape, 12, 2, 2, 98, cy=25, cz=10)
    g = axis_graph(frame, 5, 95, 15., cy=34, cz=10)
    r = cr.refine(g, frame, mask, method='dfs-centroid-shape', strength=.01,
                  max_iterations=3, max_samples=12, max_half=64)
    assert not r.converged
    assert r.segments[0]['centring_final']['status'] == 'off_centre'
    assert r.segments[0]['centring_final']['outside_tolerance_points']


def test_degree_two_join_moves_jointly_in_flat_lumen():
    shape = (20, 50, 110)
    frame = make_frame(shape)
    mask = slit(shape, 12, 2, 2, 108, cy=25, cz=10)
    g = graph_from([(50, 250, 100), (530, 300, 100), (1050, 250, 100)],
                   [(0, 1, 45, 15.), (1, 2, 37, 15.)])
    r = cr.refine(g, frame, mask, method='dfs-centroid-shape', strength=.01,
                  max_iterations=4, max_samples=12, max_half=64, root_nodes=[0])
    assert abs(g.nodes[1][1]-250) < 12
    np.testing.assert_array_equal(g.coords(0)[-1], g.coords(1)[0])
    a, b = np.diff(g.coords(0), axis=0)[-1], np.diff(g.coords(1), axis=0)[0]
    assert np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) > .999
    assert r.outside_edges_after == 0


def test_join_audit_distinguishes_a_kink_from_a_branch():
    g = graph_from([(0, 0, 0), (100, 0, 0), (150, 50, 0)],
                   [(0, 1, 5, 20.), (1, 2, 5, 20.)])
    row = join_diagnostics(g, [0, 1])[1]
    assert row['status'] == 'review_required'
    np.testing.assert_allclose(row['mismatch_degrees'], 45.)
    assert not join_diagnostics(g, [0])


def test_centring_fit_is_stable_under_uneven_point_sampling():
    shape = (20, 50, 110)
    frame = make_frame(shape)
    mask = slit(shape, 12, 2, 2, 108, cy=25, cz=10)
    curves = []
    for exponent in (1., 1.7):
        g = graph_from([(50, 250, 100), (1050, 250, 100)], [(0, 1, 101, 15.)])
        t = np.linspace(0, 1, 101)**exponent
        x = np.c_[50+1000*t, 250+50*np.sin(np.pi*t)**2, np.full(101, 100)]
        g.set_segment_coords(0, x)
        report = cr.refine(g, frame, mask, method='dfs-centroid-shape', strength=.01,
                           max_iterations=4, max_samples=16, max_half=64)
        assert report.outside_edges_after == 0
        curve = g.coords(0)
        curves.append(np.interp(np.linspace(200, 900, 25), curve[:, 0], curve[:, 1]))
    np.testing.assert_allclose(curves[0], curves[1], atol=5.)


def test_parallel_sections_and_final_audit_match_serial(tmp_path):
    import copy
    from hipct_seg_debug import amira, rle, rle_write
    shape = (12, 80, 80)
    frame = make_frame(shape)
    g = graph_from([(50, 80, 50), (200, 80, 50), (550, 650, 50), (750, 650, 50)],
                   [(0, 1, 20, 20.), (2, 3, 20, 20.)])
    other = copy.deepcopy(g)
    mask = np.zeros(shape, dtype='uint8')
    mask[3:8, 6:11, 2:25] = 1
    mask[3:8, 63:68, 52:78] = 1
    path = tmp_path/'labels.am'
    rle_write.write_lattice(path, mask, frame.seg_bbox_um)
    header = amira.read_lattice_header(path)
    labels = rle.open_lattice(path, header.fields['Labels'], header.dims, cache_dir=tmp_path/'cache')
    serial = cr.refine(g, frame, labels, method='dfs-centroid-shape', max_iterations=1, max_samples=4)
    parallel = cr.refine(other, frame, labels, method='dfs-centroid-shape', max_iterations=1,
                         max_samples=4, workers=2)
    for sid in g.segment_ids():
        np.testing.assert_allclose(g.coords(sid), other.coords(sid), atol=1e-8)
        assert serial.segments[sid]['centring_final'] == parallel.segments[sid]['centring_final']
