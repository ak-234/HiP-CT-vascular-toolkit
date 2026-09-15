"""The voxel size is stated, not inferred, and stating it rescales the world.

A bounding box written from a rounded voxel size is wrong in the one way nothing
downstream can notice: it is internally consistent. Every conversion agrees with every
other, every validation check passes, and every radius, length and volume in the
session is off by the same factor. LADAF-2024-28 records 32.99 um for an acquisition
at 32.04 -- 2.96%, which is larger than most of the corrections `radius-perimeter`
exists to make.

So the tests here are about two things: that the stated value wins over the file's,
and that applying it is *idempotent* -- because the factor comes from the segmentation,
which does not change when a graph is written, a correction that did not mark its own
output would be applied again on the next load and would still pass every check.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug import amira, frame as frame_mod
from hipct_seg_debug.amira import LatticeInfo

FILE_VOXEL = 32.99  # what the bounding box says
TRUE_VOXEL = 32.04  # what the scan was actually taken at
DIMS = (40, 30, 20)


def _lattice(voxel=FILE_VOXEL, origin=0.0):
    dims = np.array(DIMS, dtype=np.int64)
    bbox = np.empty(6)
    bbox[0::2] = origin
    bbox[1::2] = origin + (dims - 1) * voxel
    return LatticeInfo(path=None, dims=dims, bbox=bbox, fields={})


def _frame(voxel=TRUE_VOXEL, *, truth=True, lattice=None):
    lattice = lattice if lattice is not None else _lattice()
    raw_shape = (DIMS[2], DIMS[1], DIMS[0])
    return frame_mod.WorldFrame.from_inputs(
        raw_shape, voxel, lattice, voxel_is_truth=truth
    )


# ------------------------------------------------------------------- the frame


def test_without_the_flag_the_bounding_box_still_wins():
    """The old reading, kept: it is right whenever the bounding box is right."""
    f = _frame(TRUE_VOXEL, truth=False)
    assert f.raw_voxel[0] == pytest.approx(FILE_VOXEL)
    assert not f.corrected
    assert f.world_scale == pytest.approx(np.ones(3))


def test_the_stated_voxel_size_wins_when_it_is_the_truth():
    f = _frame()
    assert f.raw_voxel[0] == pytest.approx(TRUE_VOXEL)
    assert f.seg_spacing[0] == pytest.approx(TRUE_VOXEL)
    assert f.corrected
    assert float(np.mean(f.world_scale)) == pytest.approx(TRUE_VOXEL / FILE_VOXEL)
    assert f.file_voxel_um == pytest.approx(FILE_VOXEL)


def test_voxel_indices_do_not_move_when_lengths_do():
    """The invariant the whole correction rests on: images and masks stay aligned."""
    plain = _frame(truth=False)
    fixed = _frame()
    ijk = np.array([[0, 0, 0], [10.0, 7.0, 3.0], [39.0, 29.0, 19.0]])

    for point in ijk:
        um_plain = plain.seg_to_um(point)
        um_fixed = fixed.seg_to_um(point)
        # The same voxel, a different length from the origin...
        assert not np.allclose(um_plain, um_fixed) or np.allclose(point, 0)
        # ...and it still round-trips to the voxel it came from.
        assert fixed.um_to_seg(um_fixed)[0] == pytest.approx(point, abs=1e-9)


def test_an_offset_lattice_keeps_its_place_in_voxels():
    """The origin is a length too. Left unscaled it would shift the whole lattice."""
    fixed = _frame(lattice=_lattice(origin=1000.0))
    assert fixed.um_to_seg(fixed.seg_origin)[0] == pytest.approx([0, 0, 0], abs=1e-9)


def test_the_binning_is_still_derived_from_the_stated_size():
    lattice = _lattice(voxel=2 * FILE_VOXEL)
    f = _frame(lattice=lattice)
    assert list(f.bin_factor) == [2, 2, 2]
    assert f.raw_voxel[0] == pytest.approx(TRUE_VOXEL)
    assert f.seg_spacing[0] == pytest.approx(2 * TRUE_VOXEL)


def test_the_correction_is_reported_not_buried():
    f = _frame()
    note = f.correction_note()
    assert "32.9900" in note and "32.0400" in note and "-2.88%" in note
    assert f.correction_note() in f.describe()
    assert _frame(truth=False).correction_note() == ""


def test_validation_says_which_reading_was_taken():
    class _Graph:
        points = np.zeros((3, 3))

    rows = frame_mod.validate(_frame(), _Graph())
    row = next(r for r in rows if "voxel" in r.name)
    assert row.passed and "corrected" in row.name
    assert "32.0400" in row.detail and "32.9900" in row.detail


# ------------------------------------------------------- applying it to a graph


class _FakeGraph:
    def __init__(self):
        self.points = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
        self.vertices = np.array([[0.0, 0.0, 0.0], [400.0, 500.0, 600.0]])
        self.thickness = np.array([50.0, 60.0])
        self.point_attrs = {"radius_source": np.array([1, 1])}
        self.edge_attrs = {"MeanRadius": np.array([55.0])}
        self.vertex_attrs = {}


def test_a_graph_moves_with_the_lattice():
    f = _frame()
    graph = _FakeGraph()
    k = float(np.mean(f.world_scale))

    assert frame_mod.rescale_graph(graph, f)
    assert graph.points[1] == pytest.approx(np.array([400.0, 500.0, 600.0]) * k)
    assert graph.thickness == pytest.approx(np.array([50.0, 60.0]) * k)
    # A radius by any name is a length; a reason code is not.
    assert graph.edge_attrs["MeanRadius"] == pytest.approx(np.array([55.0]) * k)
    assert list(graph.point_attrs["radius_source"]) == [1, 1]


def test_an_uncorrected_frame_leaves_the_graph_alone():
    graph = _FakeGraph()
    before = graph.points.copy()
    assert not frame_mod.rescale_graph(graph, _frame(truth=False))
    assert graph.points == pytest.approx(before)


def test_the_editable_triple_scales_the_same_way():
    from hipct_seg_debug.edit.adapter import Triple

    f = _frame()
    k = float(np.mean(f.world_scale))
    triple = Triple(
        nodes={0: (0.0, 0.0, 0.0, 1), 1: (100.0, 200.0, 300.0, 1)},
        points={0: (100.0, 200.0, 300.0, 40.0)},
        segments=[{"id": 0, "node1": 0, "node2": 1, "point_ids": [0]}],
    )
    assert frame_mod.rescale_triple(triple, f)
    x, y, z, r = triple.points[0]
    assert (x, y, z) == pytest.approx((100.0 * k, 200.0 * k, 300.0 * k))
    assert r == pytest.approx(40.0 * k)
    assert triple.nodes[1][:3] == pytest.approx((100.0 * k, 200.0 * k, 300.0 * k))
    assert triple.nodes[1][3] == 1, "the degree is not a length"


# ------------------------------------------------------------------- the stamp


def test_a_written_graph_records_the_scale_it_is_in(tmp_path):
    """Without this the correction is applied again on the next load, undetectably."""
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    graph = amira.SpatialGraph(
        path=None, n_vertex=2, n_edge=1, n_point=2,
        vertices=np.zeros((2, 3)), connectivity=np.array([[0, 1]]),
        n_edge_points=np.array([2]), points=np.zeros((2, 3)),
        thickness=np.array([1.0, 1.0]),
    )
    out = write_spatial_graph(graph, tmp_path / "g.am", voxel_um=TRUE_VOXEL)
    assert amira.read_voxel_stamp(out) == pytest.approx(TRUE_VOXEL)

    # Written again from the first as its parameter source: one stamp, the new value.
    again = write_spatial_graph(graph, tmp_path / "h.am",
                                parameters_from=out, voxel_um=16.0)
    assert amira.read_voxel_stamp(again) == pytest.approx(16.0)
    assert again.read_text(encoding="latin-1").count(amira.VOXEL_STAMP) == 1


def test_an_unstamped_file_reports_nothing_rather_than_guessing(tmp_path):
    path = tmp_path / "plain.am"
    path.write_text('# AmiraMesh 3D ASCII 2.0\n\nParameters {\n'
                    '    ContentType "HxSpatialGraph"\n}\n', encoding="latin-1")
    assert amira.read_voxel_stamp(path) is None
    assert amira.read_voxel_stamp(tmp_path / "missing.am") is None
