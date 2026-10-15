"""Wave-front scooping: centroids of geodesic shells, loops kept."""
import networkx as nx
import numpy as np
import pytest

from hipct_seg_debug.edit import skeleton_phantoms as sp
from hipct_seg_debug.edit.geodesic_scoop import extract
from hipct_seg_debug.edit.skeletonisers import skeletonise

from .conftest_geometry import cylinder, make_frame, slit

SPACING = (10.0, 10.0, 10.0)


def _degrees(result):
    g = nx.Graph(result.edges.tolist())
    g.add_nodes_from(range(len(result.coordinates_zyx)))
    return g, np.array([d for _, d in sorted(g.degree)])


def test_straight_tube_is_one_on_axis_chain():
    m = cylinder((21, 21, 60), 5, 4, 56)
    r = extract(m, SPACING, root_zyx=(10, 10, 5))
    g, deg = _degrees(r)
    assert nx.is_tree(g) and (deg == 1).sum() == 2 and (deg >= 3).sum() == 0
    interior = r.coordinates_zyx[(r.coordinates_zyx[:, 2] > 12) & (r.coordinates_zyx[:, 2] < 48)]
    np.testing.assert_allclose(interior[:, :2], 10, atol=0.3)


def test_slit_centreline_sits_at_mid_width():
    """Where a distance ridge is flat, a shell's centroid is still the middle."""
    m = slit((15, 31, 60), 12, 1, 4, 56)
    r = extract(m, SPACING, root_zyx=(7, 15, 5))
    _, deg = _degrees(r)
    assert (deg == 1).sum() == 2
    interior = r.coordinates_zyx[(r.coordinates_zyx[:, 2] > 12) & (r.coordinates_zyx[:, 2] < 48)]
    np.testing.assert_allclose(interior[:, 1], 15, atol=0.3)
    np.testing.assert_allclose(interior[:, 0], 7, atol=0.3)


def test_y_gives_three_terminals_and_one_junction():
    p = sp.build("y-bifurcation")
    _, deg = _degrees(extract(p.mask, SPACING, root_zyx=p.root_zyx))
    assert (deg == 1).sum() == 3 and (deg >= 3).sum() == 1


def test_a_loop_is_kept_not_cut():
    p = sp.build("loop")
    r = extract(p.mask, SPACING, root_zyx=p.root_zyx)
    g, deg = _degrees(r)
    assert r.report["loops"] == 1
    assert g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g) == 1
    assert (deg == 1).sum() == 2


def test_bumps_do_not_become_branches():
    p = sp.build("protrusions")
    _, deg = _degrees(extract(p.mask, SPACING, root_zyx=p.root_zyx))
    assert (deg == 1).sum() == 2


def test_anisotropic_spacing_is_physical():
    """Doubling z spacing on a z-aligned tube doubles its physical length, not its shape."""
    m = np.zeros((40, 15, 15), dtype=np.uint8)
    zz, yy, xx = np.indices(m.shape)
    m[((yy - 7) ** 2 + (xx - 7) ** 2 <= 16) & (zz >= 3) & (zz < 37)] = 1
    iso = extract(m, (10.0, 10.0, 10.0), root_zyx=(4, 7, 7))
    aniso = extract(m, (10.0, 10.0, 20.0), root_zyx=(4, 7, 7))
    for r in (iso, aniso):
        _, deg = _degrees(r)
        assert (deg == 1).sum() == 2
    # Same physical shell width in both: twice the physical length, twice the shells.
    iso = extract(m, (10.0, 10.0, 10.0), root_zyx=(4, 7, 7), shell_um=20.0)
    assert aniso.report["shells"] > 1.5 * iso.report["shells"]
    with pytest.raises(ValueError, match="thinner than the largest voxel side"):
        extract(m, (10.0, 10.0, 20.0), root_zyx=(4, 7, 7), shell_um=10.0)


def test_backend_exports_world_coordinates():
    m = cylinder((21, 21, 60), 5, 4, 56)
    c = skeletonise("geodesic-scoop", m, make_frame(m.shape), root_zyx=(10, 10, 5))
    pts = np.array(list(c.triple.points.values()))
    assert len(c.triple.segments) == 1
    np.testing.assert_allclose(np.median(pts[:, 1:3], axis=0), 100, atol=3)
    assert "placeholders" in c.detail


def test_root_outside_mask_is_refused():
    m = cylinder((21, 21, 60), 5, 4, 56)
    with pytest.raises(ValueError, match="inside the segmentation"):
        extract(m, SPACING, root_zyx=(0, 0, 0))
