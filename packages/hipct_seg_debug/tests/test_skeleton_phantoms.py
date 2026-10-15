"""The phantoms the extraction benchmark is scored against.

Everything the benchmark reports is relative to these, so their own geometry is
checked against closed forms first: perimeter conservation, topology, and the
analytic radius table the note's argument rests on.
"""
import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import euler_number

from hipct_seg_debug.edit import skeleton_phantoms as sp


@pytest.mark.parametrize("lam", [1.0, 0.5, 0.3, 0.2, 0.1])
def test_semi_axes_conserve_the_inflated_perimeter(lam):
    a, b = sp.semi_axes(7.0, lam)
    assert b == pytest.approx(lam * a)
    assert sp.ellipse_perimeter(a, b) == pytest.approx(2 * np.pi * 7.0, rel=1e-9)


def test_analytic_table_matches_the_note():
    """The note's section 2.2 columns: inscribed 0.65/0.43/0.30/0.16, area 0.92/0.79/0.67/0.49."""
    rows = {r["lam"]: r for r in sp.analytic_radius_table()}
    for lam, inscribed, area in ((0.5, 0.65, 0.92), (0.3, 0.43, 0.79), (0.2, 0.30, 0.67),
                                 (0.1, 0.16, 0.49)):
        assert rows[lam]["inscribed"] == pytest.approx(inscribed, abs=0.01)
        assert rows[lam]["area"] == pytest.approx(area, abs=0.01)
    assert rows[1.0]["inscribed"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", sp.PHANTOMS)
def test_mask_topology_matches_the_ground_truth(name):
    p = sp.build(name)
    _, cc = ndimage.label(p.mask, structure=np.ones((3, 3, 3)))
    assert cc == p.components == 1
    # Euler characteristic of a solid with one component and no cavities: 1 - beta1.
    assert euler_number(p.mask, connectivity=3) == 1 - p.beta1
    assert p.mask[p.root_zyx]


def test_ground_truth_counts():
    assert sp.build("y-bifurcation").junctions.shape[0] == 1
    assert sp.build("y-bifurcation").terminals.shape[0] == 3
    loop = sp.build("loop")
    assert loop.beta1 == 1 and len(loop.terminals) == 2 and len(loop.junctions) == 2


def test_collapse_bands_section_area_tracks_the_ellipse():
    """Voxel area within 15% of pi*a*b, at R0 = 8, wherever a band is uniform."""
    p = sp.build("collapse-bands")
    edges = p.extra["band_edges"]
    pts = p.branches[0].points
    x0, x1 = pts[0, 2], pts[-1, 2]
    for i, lam in enumerate(sp.COLLAPSE_BANDS):
        x = int(round(x0 + (x1 - x0) * (edges[i] + edges[i + 1]) / 2))
        a, b = sp.semi_axes(8.0, lam)
        area = float(p.mask[:, :, x].sum())
        assert area == pytest.approx(np.pi * a * b, rel=0.15 if lam > 0.1 else 0.35)


def test_ends_are_rounded_not_flat():
    """A flat cap forks a slit's medial sheet into its corners; the caps are domes."""
    p = sp.build("round")
    end = int(round(p.branches[0].points[-1, 2]))
    beyond = p.mask[:, :, end + 3].sum()
    assert 0 < beyond < p.mask[:, :, end - 3].sum()


def test_unknown_phantom_is_refused():
    with pytest.raises(ValueError, match="unknown phantom"):
        sp.build("torus")
