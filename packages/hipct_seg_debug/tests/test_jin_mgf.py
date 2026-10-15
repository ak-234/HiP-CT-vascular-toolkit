"""Collapse-aware Jin MCP: hooks, medial dimension, MGF ridge, section marking."""
import networkx as nx
import numpy as np
import pytest

from hipct_seg_debug.edit import jin_mcp, jin_mgf
from hipct_seg_debug.edit import skeleton_phantoms as sp


def flattened():
    """The counterexample `test_jin_mcp` keeps for the published method."""
    m = np.zeros((9, 19, 61))
    m[3:6, 3:16, 2:59] = 1
    return m


def leaves(result):
    return sum(d == 1 for _, d in nx.Graph(result.edges.tolist()).degree)


def test_default_hooks_reproduce_published_jin_exactly():
    m = flattened()
    a = jin_mcp.extract(m, root_zyx=(4, 9, 4))
    b = jin_mgf.extract(m, root_zyx=(4, 9, 4), medialness="lsf", marking="dilate")
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.coordinates_zyx, b.coordinates_zyx)


def test_published_jin_ladders_and_both_fixes_remove_them():
    m = flattened()
    assert leaves(jin_mcp.extract(m, root_zyx=(4, 9, 4))) > 2
    for kw in (dict(medialness="mgf", marking="dilate"),
               dict(medialness="lsf", marking="section"),
               dict()):
        r = jin_mgf.extract(m, root_zyx=(4, 9, 4), **kw)
        assert leaves(r) == 2, kw
        assert np.all(m[tuple(r.coordinates_zyx.T)])


def test_hessian_dimension_tells_a_tube_from_a_slit():
    for name, expect in (("round", 1), ("curved-collapsed", 2)):
        p = sp.build(name)
        fdt, lsf = jin_mcp.fields(p.mask.astype(float))
        dim, normal = jin_mgf.hessian_dimension(p.mask, lsf)
        labelled = dim[dim > 0]
        assert np.mean(labelled == expect) > 0.8, name


def test_exact_and_rim_mgf_put_the_ridge_mid_slit():
    m = flattened()
    fdt, lsf = jin_mcp.fields(m)
    dim, normal = jin_mgf.hessian_dimension(m > 0, lsf)
    sheet = jin_mgf.sheet_support(dim)
    assert sheet[4, 9, 30] and sheet[4, 4, 30], "the rim rows belong to the sheet"
    for f in (jin_mgf._mgf_exact(m > 0, sheet),
              jin_mgf._mgf_rim(m > 0, sheet, lsf > .5, normal)):
        row = f[4, :, 30]
        assert np.nanargmax(row) == 9
        np.testing.assert_allclose(row[9 - 3], row[9 + 3], atol=0.5)
        w = jin_mgf.ridge_weight(f, sheet)
        assert w[4, 9, 30] > 0.8 and w[4, 5, 30] < 0.2


def test_mgf_centreline_is_mid_slit():
    r = jin_mgf.extract(flattened(), root_zyx=(4, 9, 4))
    body = r.coordinates_zyx[(r.coordinates_zyx[:, 2] > 10) & (r.coordinates_zyx[:, 2] < 50)]
    assert np.all(body[:, 1] == 9) and np.all(body[:, 0] == 4)


def test_an_unclosed_section_is_not_marked():
    """A slab still open at its window edge is a vessel running in-plane, not a section."""
    m = np.zeros((9, 41, 41), dtype=bool)
    m[3:6, :, :] = True  # a sheet filling the window: never closes
    out = np.zeros_like(m)
    ok = jin_mgf._fill_section(m, np.array([4, 20, 20]), np.array([0.0, 0.0, 1.0]), 6,
                               0.87, out, jin_mcp.OFFSETS)
    assert not ok and not out.any()
    m2 = np.zeros_like(m)
    m2[3:6, 15:26, 15:26] = True
    ok = jin_mgf._fill_section(m2, np.array([4, 20, 20]), np.array([0.0, 0.0, 1.0]), 8,
                               0.87, out, jin_mcp.OFFSETS)
    assert ok and out[4, 15:26, 20].all() and not out[:, :, 18].any()


def test_backend_and_report():
    from hipct_seg_debug.edit.skeletonisers import skeletonise

    from .conftest_geometry import make_frame
    m = flattened()
    c = skeletonise("jin-mgf", m, make_frame(m.shape), root_zyx=(4, 9, 4))
    assert len(c.triple.segments) == 1
    assert "Jin MGF [mgf/section]" in c.detail


@pytest.mark.parametrize("kw", [dict(medialness="x"), dict(marking="x"), dict(mgf_mode="x"),
                                dict(classifier="x")])
def test_bad_options_are_refused(kw):
    with pytest.raises(ValueError):
        jin_mgf.extract(flattened(), root_zyx=(4, 9, 4), **kw)


@pytest.mark.parametrize("tile, halo", [(4, 3), (4, 6), (8, 2)])
def test_tiles_partition_points_and_gather_every_neighbour_in_the_halo(tile, halo):
    """Tiled filters are exact only if `near` is everything in the haloed box."""
    from hipct_seg_debug.edit import jin_mgf

    rng = np.random.default_rng(0)
    shape = (13, 17, 11)
    zyx = np.unique(rng.integers(0, shape, size=(300, 3)), axis=0)
    owned = []
    for lo, hi, own, near in jin_mgf._tiles(zyx, shape, halo=halo, tile=tile):
        cell = zyx[own] // tile
        assert (cell == cell[0]).all(), "own is one tile"
        assert (lo == np.maximum(cell[0] * tile - halo, 0)).all()
        assert (hi == np.minimum(cell[0] * tile + tile + halo, shape)).all()
        inside = np.flatnonzero(np.all((zyx >= lo) & (zyx < hi), axis=1))
        assert sorted(near.tolist()) == inside.tolist()
        owned.extend(own.tolist())
    assert sorted(owned) == list(range(len(zyx)))
