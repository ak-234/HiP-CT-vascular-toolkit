"""Label lattices that name their trees, and the split that follows from it.

An Avizo ``.Regions.am`` carries a ``Materials`` block -- on this data ``Exterior``,
``Left_Tree``, ``Right_Tree`` -- and the voxels hold the material's **index**, 0/1/2.
Everything here exists because connectivity alone cannot recover that: at 32 um the
two coronaries touch, so ``volume > 0`` labels them as one component, and the file has
said which is which all along.

What is pinned:

* the ``Id`` field in the header is *not* the voxel value, and is not read;
* two materials that touch come back as two trees, where connectivity gives one;
* the main body of each tree is numbered before any fragment of either;
* a mask with no Materials block behaves exactly as it did before.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug import amira
from hipct_seg_debug.amira import Material
from hipct_seg_debug.edit import components as comp

from .conftest_geometry import make_frame

HEADER = """# AmiraMesh BINARY-LITTLE-ENDIAN 3.0

define Lattice 4 4 4

Parameters {
    Materials {
        Exterior {
            Color 0 0 0,
            &Color "AAAA==",
            Id "0"
        }
        Left_Tree {
            Color 0.0615396350622177 0 1,
            Id "9"
        }
        Right_Tree {
            Color 0.5692378282547 0 1,
            Id "10"
        }
    }
    Content "4x4x4 byte, uniform coordinates",
    BoundingBox 0 3 0 3 0 3,
    CoordType "uniform"
}

Lattice { byte Labels } @1(HxByteRLE,17)
"""


# ------------------------------------------------------------------ the header


def test_the_materials_block_is_read_in_file_order():
    materials = amira.parse_materials(HEADER)

    assert [m.name for m in materials] == ["Exterior", "Left_Tree", "Right_Tree"]


def test_the_voxel_value_is_the_index_not_the_id_field():
    """`Left_Tree` says `Id "9"` and the voxels hold 1. Reading Id selects nothing."""
    materials = amira.parse_materials(HEADER)

    assert [m.value for m in materials] == [0, 1, 2]


def test_avizos_own_colour_comes_through():
    left = amira.parse_materials(HEADER)[1]

    assert left.color == pytest.approx((0.0615396350622177, 0.0, 1.0))


def test_a_lattice_with_no_materials_block_reports_none():
    assert amira.parse_materials("define Lattice 4 4 4\nLattice { byte Labels } @1") == ()


def test_regions_drops_the_exterior_by_value_not_by_name():
    info = amira.LatticeInfo(path=None, dims=np.asarray([4, 4, 4]),
                             bbox=np.zeros(6), fields={},
                             materials=(Material("Background", 0),
                                        Material("Left_Tree", 1)))

    assert [m.name for m in info.regions] == ["Left_Tree"]
    assert info.material("left_tree").value == 1
    assert info.material("nope") is None


# ------------------------------------------------------------------- the split


MATERIALS = (Material("Exterior", 0), Material("Left_Tree", 1),
             Material("Right_Tree", 2))


def touching_trees():
    """Left and right sharing a face, plus one detached fragment of the left."""
    volume = np.zeros((20, 20, 40), np.uint8)
    volume[5:15, 5:15, 2:14] = 1        # Left main body
    volume[5:15, 5:15, 14:26] = 2       # Right, against the left's face
    volume[1:4, 1:4, 30:34] = 1         # a detached fragment of the left
    return volume, make_frame(volume.shape)


def test_connectivity_alone_fuses_two_touching_trees():
    """The reason materials are read at all -- not a defect in `split_components`."""
    volume, frame = touching_trees()

    parts, _stats = comp.split_components(volume, frame)

    assert parts[0].voxels == 2400, "both coronaries in one component"
    assert parts[0].material == ""


def test_materials_separate_what_connectivity_fuses():
    volume, frame = touching_trees()

    parts, _stats = comp.split_components(volume, frame, materials=MATERIALS)

    assert [(p.material, p.voxels) for p in parts[:2]] == [
        ("Left_Tree", 1200), ("Right_Tree", 1200)
    ]


def test_every_main_body_is_numbered_before_any_fragment():
    """Otherwise the right coronary lands behind 960 specks of the left one."""
    volume, frame = touching_trees()

    parts, _stats = comp.split_components(volume, frame, materials=MATERIALS)

    assert [p.index for p in parts if p.is_main_body] == [0, 1]
    assert parts[2].material == "Left_Tree" and parts[2].rank == 1


def test_max_trees_keeps_the_main_body_of_each_tree():
    """A global cap could drop a whole coronary; per material it cannot."""
    volume, frame = touching_trees()

    parts, _stats = comp.split_components(volume, frame, materials=MATERIALS,
                                          max_trees=1)

    assert [p.material for p in parts] == ["Left_Tree", "Right_Tree"]
    assert all(p.is_main_body for p in parts)


def test_the_labelling_gives_every_component_a_globally_unique_label():
    """`assign_trees` rasterises one array, so two materials must not share a label."""
    volume, frame = touching_trees()

    parts, stats = comp.split_components(volume, frame, materials=MATERIALS)

    labels = [p.label for p in parts]
    assert len(set(labels)) == len(labels)
    for part in parts:
        assert int((stats.labels == part.label).sum()) == part.voxels


def test_a_lattice_info_can_be_passed_straight_in():
    volume, frame = touching_trees()
    info = amira.LatticeInfo(path=None, dims=np.asarray([40, 20, 20]),
                             bbox=np.zeros(6), fields={}, materials=MATERIALS)

    parts, _stats = comp.split_components(volume, frame, materials=info)

    assert [p.material for p in parts[:2]] == ["Left_Tree", "Right_Tree"]


def test_without_materials_the_split_is_exactly_what_it_was():
    volume, frame = touching_trees()

    plain, _ = comp.split_components(volume, frame)
    ignored, _ = comp.split_components(volume, frame, materials=())

    assert [(p.index, p.label, p.voxels) for p in plain] == \
           [(p.index, p.label, p.voxels) for p in ignored]
    assert all(p.material == "" and p.rank == p.index for p in plain)


# ------------------------------------------------------ the 3-D view of them


def test_material_colour_is_avizos_own():
    from hipct_seg_debug.viewer3d import material_color

    assert material_color(Material("Left_Tree", 1, (0.0, 0.5, 1.0))) == "#0080ff"


def test_a_material_with_no_colour_falls_back_to_the_viewers_blue():
    from hipct_seg_debug.viewer3d import SEG_COLOR, material_color

    # Avizo writes black for a material nobody ever coloured, and a black vessel on a
    # black background is invisible rather than merely wrong.
    assert material_color(Material("Exterior", 0, (0.0, 0.0, 0.0))) == SEG_COLOR
    assert material_color(Material("Left_Tree", 1, None)) == SEG_COLOR
    assert material_color(None) == SEG_COLOR


def test_one_isosurface_per_material_rather_than_one_fused_surface():
    """`segmentation_volume` contours `mask > 0`, which welds the two coronaries."""
    pytest.importorskip("pyvista")
    from hipct_seg_debug.viewer3d import material_surfaces, segmentation_volume

    volume, frame = touching_trees()

    _grid, fused = segmentation_volume(frame, None, 1, volume=volume)
    surfaces = material_surfaces(frame, None, MATERIALS[1:], 1, volume=volume)

    assert fused is not None and fused.n_cells
    assert [m.name for m, _s in surfaces] == ["Left_Tree", "Right_Tree"]
    assert all(s.n_cells for _m, s in surfaces)


def test_a_material_with_no_voxels_is_skipped_rather_than_drawn_empty():
    pytest.importorskip("pyvista")
    from hipct_seg_debug.viewer3d import material_surfaces

    volume, frame = touching_trees()
    absent = (*MATERIALS[1:], Material("Aorta", 7))

    surfaces = material_surfaces(frame, None, absent, 1, volume=volume)

    assert [m.name for m, _s in surfaces] == ["Left_Tree", "Right_Tree"]


# ------------------------------------------------------ one labelling, not N


def test_the_foreground_is_labelled_once_however_many_materials(monkeypatch):
    """The volume is walked once, not once per material.

    A 26-connected pass over the full lattice is 47.8 G voxels and the expensive part
    of the whole command; doing it per material doubled a run that already took the
    best part of an hour.
    """
    from hipct_seg_debug.edit.reconnect import segmentation as seg

    calls = []
    real = seg.components

    def counted(mask, connectivity=3):
        calls.append(np.asarray(mask).size)
        return real(mask, connectivity=connectivity)

    monkeypatch.setattr(seg, "components", counted)
    volume, frame = touching_trees()

    comp.split_components(volume, frame, materials=MATERIALS)

    assert len(calls) == 1, "one whole-volume labelling"
    assert calls[0] == volume.size


def test_only_a_component_that_really_straddles_is_re_labelled(monkeypatch):
    """And it is re-labelled inside its own box, not over the volume."""
    from scipy import ndimage

    windows = []
    real = ndimage.label

    def counted(mask, structure=None, output=None):
        windows.append(np.asarray(mask).size)
        return real(mask, structure=structure, output=output)

    monkeypatch.setattr(ndimage, "label", counted)
    volume, frame = touching_trees()

    comp.split_components(volume, frame, materials=MATERIALS)

    # One whole-volume labelling, then the straddling component's box only.
    assert windows[0] == volume.size
    assert all(w < volume.size for w in windows[1:]), "the rest are boxes"


def test_nothing_is_re_labelled_when_no_component_straddles(monkeypatch):
    from scipy import ndimage

    calls = []
    real = ndimage.label
    monkeypatch.setattr(ndimage, "label",
                        lambda m, structure=None, output=None: (
                            calls.append(1) or real(m, structure=structure,
                                                    output=output)))
    # Left and right kept apart, so connectivity already separates them.
    volume = np.zeros((20, 20, 40), np.uint8)
    volume[5:15, 5:15, 2:12] = 1
    volume[5:15, 5:15, 20:30] = 2
    frame = make_frame(volume.shape)

    parts, _stats = comp.split_components(volume, frame, materials=MATERIALS)

    assert len(calls) == 1, "the single labelling, and no re-splitting"
    assert [p.material for p in parts] == ["Left_Tree", "Right_Tree"]


def test_a_straddling_component_keeps_correct_sizes_and_labels():
    """The re-split pieces must own exactly the voxels the labelling says they do."""
    volume, frame = touching_trees()

    parts, stats = comp.split_components(volume, frame, materials=MATERIALS)

    for part in parts:
        assert int((stats.labels == part.label).sum()) == part.voxels
    assert int((stats.labels > 0).sum()) == int((volume > 0).sum()), "no voxel lost"
