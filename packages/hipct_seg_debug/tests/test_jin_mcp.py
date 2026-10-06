"""Analytic fields, independent geometry and export checks for Jin MCP."""
import networkx as nx
import numpy as np
import pytest

from hipct_seg_debug.edit.jin_mcp import extract, fields
from hipct_seg_debug.edit.skeletonisers import skeletonise
from .conftest_geometry import make_frame


def bar():
    m = np.zeros((11, 11, 61))
    m[3:8, 3:8, 2:59] = 1
    return m


def test_analytic_binary_and_fuzzy_slab():
    m = bar()
    f, lsf = fields(m)
    np.testing.assert_allclose(f[3:8, 5, 30], [.5, 1.5, 2.5, 1.5, .5])
    assert lsf[5, 5, 30] == 1
    assert lsf[4, 5, 30] == 0
    ff, ll = fields(m * .4)
    np.testing.assert_allclose(ff, f * .4, atol=1e-12)
    np.testing.assert_allclose(ll, lsf, atol=1e-12)


def test_straight_tube_medial_contained_tree():
    m = bar()
    result = extract(m, root_zyx=(5, 5, 4), max_iterations=1)
    assert np.all(result.coordinates_zyx[:, :2] == 5)
    assert result.coordinates_zyx[:, 2].max() >= 55
    assert np.all(m[tuple(result.coordinates_zyx.T)])
    g = nx.Graph(result.edges.tolist())
    assert nx.is_tree(g)
    assert sum(d == 1 for _, d in g.degree) == 2


def test_three_arm_tree_and_round_batching():
    z, y, x = np.indices((13, 71, 71))
    m = ((z - 6)**2 + (y - 35)**2 <= 4) & (x >= 4) & (x <= 66)
    m |= ((z - 6)**2 + (x - 35)**2 <= 4) & (y >= 35) & (y <= 66)
    r = extract(m, root_zyx=(6, 35, 6))
    g = nx.Graph(r.edges.tolist())
    assert nx.is_tree(g)
    assert sum(d == 1 for _, d in g.degree) == 3
    assert np.all(m[tuple(r.coordinates_zyx.T)])
    ends = r.coordinates_zyx[[i for i, d in g.degree if d == 1]]
    assert any(p[1] > 60 for p in ends)
    assert any(p[2] > 60 for p in ends)
    assert any(p[2] < 10 for p in ends)


def test_flattened_tube_exposes_medial_sheet_limitation():
    m = np.zeros((9, 19, 61))
    m[3:6, 3:16, 2:59] = 1
    r = extract(m, root_zyx=(4, 9, 4))
    assert np.all(r.coordinates_zyx[:, 0] == 4)
    # This algorithm selects paths on a medial sheet; it does NOT guarantee a
    # unique cross-sectional centreline for a very flattened lumen. Keep this
    # counterexample visible instead of tuning away its false branches.
    assert np.all(m[tuple(r.coordinates_zyx.T)])
    assert sum(d == 1 for _, d in nx.Graph(r.edges.tolist()).degree) > 2
    assert r.report['review_required']
    again = extract(m, root_zyx=(4, 9, 4))
    np.testing.assert_array_equal(r.edges, again.edges)
    np.testing.assert_array_equal(r.coordinates_zyx, again.coordinates_zyx)


def test_export_world_units_and_report(tmp_path):
    m = bar()
    c = skeletonise('jin-mcp', m, make_frame(m.shape), root_zyx=(5, 5, 4), report_dir=tmp_path)
    points = np.asarray(list(c.triple.points.values()))
    np.testing.assert_allclose(points[:, 1:3], 50)
    np.testing.assert_allclose(points[:, 3], 25)
    assert len(list(tmp_path.glob('component_*.json'))) == 1
    assert 'placeholders' in c.detail


@pytest.mark.parametrize('kind', ['disconnected', 'invalid', 'large', 'root', 'anisotropic'])
def test_refuses_unsupported_inputs(kind):
    m = bar()
    with pytest.raises(ValueError):
        if kind == 'disconnected':
            m[0, 0, 0] = 1
            extract(m)
        elif kind == 'invalid':
            extract(m * 2)
        elif kind == 'large':
            extract(m, max_voxels=10)
        elif kind == 'root':
            extract(m, root_zyx=(0, 0, 0))
        else:
            from hipct_seg_debug.edit.skeletonisers import _jin_mcp
            _jin_mcp(m, make_frame(m.shape), np.array([1., 2., 1.]))


def test_isolated_root_is_preserved():
    m = np.ones((1, 1, 1))
    c = skeletonise('jin-mcp', m, make_frame((3, 3, 3)))
    assert len(c.triple.nodes) == 1
    assert not c.triple.segments


def test_curved_tube_preserves_corner():
    z, y, x = np.indices((13, 61, 61))
    m = ((z-6)**2 + (y-12)**2 <= 4) & (x >= 5) & (x <= 45)
    m |= ((z-6)**2 + (x-45)**2 <= 4) & (y >= 12) & (y <= 55)
    r = extract(m, root_zyx=(6, 12, 7))
    assert nx.is_tree(nx.Graph(r.edges.tolist()))
    assert np.all(m[tuple(r.coordinates_zyx.T)])
    assert np.min(np.linalg.norm(r.coordinates_zyx - (6, 12, 45), axis=1)) <= 1
    assert r.coordinates_zyx[:, 1].max() > 50


def test_cavity_is_refused():
    m = np.ones((9, 9, 9))
    m[4, 4, 4] = 0
    with pytest.raises(ValueError, match='Euler'):
        extract(m)


def test_six_arm_multifurcation_batches_independent_subtrees():
    z, y, x = np.indices((61, 61, 61))
    m = ((z-30)**2 + (y-30)**2 <= 4) & (x >= 4) & (x <= 56)
    m |= ((z-30)**2 + (x-30)**2 <= 4) & (y >= 4) & (y <= 56)
    m |= ((y-30)**2 + (x-30)**2 <= 4) & (z >= 4) & (z <= 56)
    r = extract(m, root_zyx=(30, 30, 6))
    g = nx.Graph(r.edges.tolist())
    assert nx.is_tree(g)
    assert sum(d == 1 for _, d in g.degree) == 6
    assert any(sum(b['accepted'] for b in row['branches']) > 1 for row in r.report['rounds'])


def test_shallow_protrusion_does_not_add_branch():
    m = bar()
    m[5, 8, 30] = 1
    r = extract(m, root_zyx=(5, 5, 4))
    assert sum(d == 1 for _, d in nx.Graph(r.edges.tolist()).degree) == 2


def test_roi_decoder_only_reads_window_and_adjusts_frame(monkeypatch):
    from hipct_seg_debug.edit import __main__ as cli
    class Labels:
        nz, ny, nx = 1000, 1000, 1000
        calls = []

        def slice_window(self, z, y0, y1, x0, x1):
            self.calls.append((z, y0, y1, x0, x1))
            return np.ones((y1-y0, x1-x0), dtype=np.uint8)

    labels = Labels()
    monkeypatch.setattr(cli, '_open_lattice', lambda _: (labels, make_frame((1000,)*3), 'fake.am'))
    args = cli.build_parser().parse_args(['skeletonise-all', '--algorithms', 'jin-mcp',
                                         '--roi-zyx', '10', '13', '20', '24', '30', '35'])
    volume, frame, _, _ = cli._decoded(args)
    assert volume.shape == (3, 4, 5)
    assert labels.calls == [(z, 20, 24, 30, 35) for z in range(10, 13)]
    np.testing.assert_allclose(frame.seg_origin, [300, 200, 100])
    np.testing.assert_array_equal(frame.seg_dims, [5, 4, 3])
    args.roi_zyx = None
    with pytest.raises(ValueError, match='before decoding'):
        cli._decoded(args)
    assert len(labels.calls) == 3


def test_command_writes_graph_and_acceptance_report(monkeypatch, tmp_path):
    import json
    from hipct_seg_debug.edit import __main__ as cli
    from hipct_seg_debug.amira import read_spatial_graph
    m = bar()
    monkeypatch.setattr(cli, '_decoded', lambda _: (m, make_frame(m.shape), None, 'fake.am'))
    monkeypatch.setattr(cli, '_seg_materials', lambda _: None)
    args = cli.build_parser().parse_args(['skeletonise-all', '--algorithms', 'jin-mcp',
                                         '--jin-root-zyx', '5', '5', '4', '--no-score',
                                         '--out-dir', str(tmp_path)])
    assert cli.cmd_skeletonise_all(args) == 0
    assert read_spatial_graph(tmp_path / 'jin-mcp.am') is not None
    report = json.loads(next((tmp_path / 'jin-mcp-reports').glob('*.json')).read_text())
    assert report['root_zyx'] == [5, 5, 4]
    assert report['review_required']


def test_centroid_hybrid_writes_separate_refinement_report(tmp_path):
    import json
    m = bar()
    raw = skeletonise('jin-mcp', m, make_frame(m.shape), root_zyx=(5, 5, 4))
    hybrid = skeletonise('jin-mcp-centroid', m, make_frame(m.shape), root_zyx=(5, 5, 4),
                         refine_iterations=2, report_dir=tmp_path)
    assert len(raw.triple.segments) == len(hybrid.triple.segments)
    np.testing.assert_array_equal([p[3] for p in raw.triple.points.values()],
                                  [p[3] for p in hybrid.triple.points.values()])
    report = json.loads(next(tmp_path.glob('*_refinement.json')).read_text())
    assert report['method'] == 'dfs-centroid-shape'
    assert all('centring_final' in r for r in report['segments'].values())
