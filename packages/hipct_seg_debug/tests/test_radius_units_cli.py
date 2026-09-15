"""A corrected graph must sample the same voxels on a subsequent CLI run."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipct_seg_debug.amira import read_spatial_graph
from hipct_seg_debug.edit.__main__ import _correct_units, _load, _open_lattice
from hipct_seg_debug.edit.amira_write import write_spatial_graph
from hipct_seg_debug.edit.radius_perimeter import FILLED, measure_radii
from hipct_seg_debug.frame import WorldFrame

from .conftest_geometry import axis_graph, cylinder


@pytest.fixture
def inputs(tmp_path):
    from hipct_seg_debug.amira import read_lattice_header

    volume = cylinder((40, 40, 80), 6, 5, 75)
    dims = np.array(volume.shape[::-1])
    origin = np.full(3, 32990.0)
    hi = origin + (dims - 1) * 32.99
    bbox = " ".join(str(x) for pair in zip(origin, hi) for x in pair)
    header = (
        "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n"
        "define Lattice 80 40 40\n"
        f'Parameters {{ BoundingBox {bbox}, CoordType "uniform" }}\n'
        "Lattice { byte Labels } @1\n\n@1\n"
    )
    seg = tmp_path / "mask.am"
    seg.write_bytes(header.encode("ascii") + volume.tobytes())
    info = read_lattice_header(seg)
    frame = WorldFrame.from_inputs(volume.shape, 32.04, info, voxel_is_truth=True)
    graph = axis_graph(frame, 8, 72, 6 * 32.04, cy=20, cz=20)
    path = write_spatial_graph(graph.to_spatial_graph(), tmp_path / "graph.am",
                               voxel_um=32.04)
    return SimpleNamespace(seg=str(seg), graph=[str(path)], labels_field="Labels",
                           voxel_um=None, edits=None)


def test_omitted_voxel_uses_graph_stamp_and_measures_the_lumen(inputs):
    before = read_spatial_graph(inputs.graph[0])
    labels, frame, _ = _open_lattice(inputs)
    graph = _load(inputs.graph)
    stamp = _correct_units(graph, frame, inputs)
    assert frame.seg_spacing == pytest.approx([32.04] * 3)
    assert stamp == pytest.approx(32.04)
    np.testing.assert_allclose(graph.coords(0), before.points)
    result = measure_radii(graph, frame, labels)
    assert (result.source[0] != FILLED).all()
    assert np.median(result.radii[0]) == pytest.approx(6 * 32.04, rel=0.1)


def test_explicit_matching_voxel_does_not_scale_graph_twice(inputs):
    inputs.voxel_um = 32.04
    before = read_spatial_graph(inputs.graph[0])
    _, frame, _ = _open_lattice(inputs)
    graph = _load(inputs.graph)
    _correct_units(graph, frame, inputs)
    np.testing.assert_allclose(graph.coords(0), before.points)
    np.testing.assert_allclose(graph.radii(0), before.thickness)


@pytest.mark.parametrize("voxel", [32.99, 30.0])
def test_conflicting_explicit_voxel_cannot_silently_misalign_stamped_graph(inputs, voxel):
    inputs.voxel_um = voxel
    _, frame, _ = _open_lattice(inputs)
    graph = _load(inputs.graph)
    with pytest.raises(ValueError, match="scale does not match"):
        _correct_units(graph, frame, inputs)


@pytest.mark.parametrize("second_stamp", [None, 30.0])
def test_mixed_graph_units_are_rejected_before_sampling(inputs, tmp_path, second_stamp):
    graph = read_spatial_graph(inputs.graph[0])
    graph.path = None
    other = write_spatial_graph(graph, tmp_path / "other.am",
                                voxel_um=second_stamp)
    inputs.graph.append(str(other))
    with pytest.raises(ValueError, match="same.*scale"):
        _open_lattice(inputs)


def test_unstamped_graph_keeps_lattice_units(inputs, tmp_path):
    graph = read_spatial_graph(inputs.graph[0])
    graph.path = None
    other = write_spatial_graph(graph, tmp_path / "plain.am")
    inputs.graph = [str(other)]
    _, frame, _ = _open_lattice(inputs)
    assert frame.seg_spacing == pytest.approx([32.99] * 3)
    assert not frame.corrected
