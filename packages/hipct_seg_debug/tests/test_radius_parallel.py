"""Parallel plane work must preserve global calibration, junctions and provenance."""
from dataclasses import fields

import numpy as np
import pytest

from hipct_seg_debug.edit import radius_perimeter as rp

from .conftest_geometry import cylinder, graph_from, make_frame


def assert_same(serial, parallel):
    for field in fields(serial):
        if field.name == "seconds":
            continue
        a, b = getattr(serial, field.name), getattr(parallel, field.name)
        if isinstance(a, dict):
            assert list(a) == list(b), field.name
            for sid in a:
                np.testing.assert_allclose(a[sid], b[sid], rtol=1e-12, atol=1e-12,
                                           err_msg=f"{field.name}[{sid}]")
        else:
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=field.name)


def fixture():
    shape = (40, 40, 80)
    frame = make_frame(shape)
    # Three arms meet at one node; the last arm leaves the mask. Stored radii
    # deliberately use different correction ratios to exercise global calibration.
    xyz = frame.seg_to_um([[40, 20, 20], [8, 20, 20], [72, 20, 20], [40, 35, 20]])
    graph = graph_from(xyz, [(0, 1, 14, 25.), (0, 2, 14, 70.), (0, 3, 9, 50.)])
    mask = cylinder(shape, 6, 5, 75)
    return graph, frame, mask


@pytest.mark.parametrize("options", [
    {},
    dict(section_filter=True),
    dict(branch_aware=False, gate_voxels=9., perimeter_correction=False,
         fallback_policy="drop", fallback_taper=True, max_radius_factor=1.5),
    dict(junction_flare="parent", root_edges=[0], bifurcation_tapers=True,
         carina_tip_factor=.2, continuation_ratio=.8),
])
def test_parallel_matches_serial_with_global_junction_context(options):
    graph, frame, mask = fixture()
    serial = rp.measure_radii(graph, frame, mask, **options)
    progress = []
    parallel = rp.measure_radii(graph, frame, mask, workers=2,
                                 progress=lambda n, total: progress.append((n, total)),
                                 **options)
    assert progress[-1] == (3, 3)
    assert_same(serial, parallel)


def test_workers_reopen_compressed_lattice(tmp_path):
    from hipct_seg_debug import amira, rle, rle_write

    graph, frame, mask = fixture()
    path = tmp_path / "mask.am"
    rle_write.write_lattice(path, mask, frame.seg_bbox_um)
    info = amira.read_lattice_header(path)
    labels = rle.open_lattice(path, info.fields["Labels"], info.dims)
    serial = rp.measure_radii(graph, frame, labels)
    parallel = rp.measure_radii(graph, frame, labels, workers=2)
    assert_same(serial, parallel)


def test_parallel_preserves_nonadjacent_branch_ownership():
    frame = make_frame((40, 40, 80))
    mask = np.maximum(cylinder((40, 40, 80), 4, 5, 75, cy=16, cz=20),
                      cylinder((40, 40, 80), 4, 5, 75, cy=24, cz=20))
    xyz = frame.seg_to_um([[8, 16, 20], [72, 16, 20], [8, 24, 20], [72, 24, 20]])
    graph = graph_from(xyz, [(0, 1, 8, 40.), (2, 3, 8, 40.)])
    serial = rp.measure_radii(graph, frame, mask)
    assert sum((v == rp.OWNED_PLANE).sum() for v in serial.resolution_mode.values()) > 0
    assert_same(serial, rp.measure_radii(graph, frame, mask, workers=2))


@pytest.mark.parametrize('shared_filter', [False, True])
def test_regional_measurement_keeps_global_junction_context_without_missing_segment_lookup(shared_filter):
    graph, frame, mask = fixture()
    before = dict(graph.points)
    result = rp.measure_radii(graph, frame, mask, _segment_ids=[0], section_filter=shared_filter)
    assert set(result.radii) == {0}
    assert graph.points == before
    if shared_filter:
        assert not np.isin(result.resolution_mode[0],
                          [rp.BIF_PARENT, rp.BIF_DAUGHTER, rp.BIF_CONTINUATION]).any()
        assert result.section_rejection_counts[0].shape == (len(graph.coords(0)), 5)


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_worker_count_is_rejected(workers):
    graph, frame, mask = fixture()
    with pytest.raises(ValueError, match="workers"):
        rp.measure_radii(graph, frame, mask, workers=workers)


def test_parallel_multipass_is_explicitly_unsupported():
    graph, frame, mask = fixture()
    with pytest.raises(ValueError, match="n_passes"):
        rp.measure_radii(graph, frame, mask, workers=2, n_passes=2)


def test_command_forwards_worker_count(monkeypatch):
    from hipct_seg_debug.edit import __main__ as cli
    from hipct_seg_debug.edit import roots

    graph, frame, mask = fixture()
    result = rp.measure_radii(graph, frame, mask)
    args = cli.build_parser().parse_args([
        "radius-perimeter", "input.am", "--seg", "mask.am", "--workers", "3",
    ])
    monkeypatch.setattr(cli, "_load", lambda *a: graph)
    monkeypatch.setattr(cli, "_open_lattice", lambda *a: (mask, frame, "mask.am"))
    monkeypatch.setattr(cli, "_correct_units", lambda *a: None)
    monkeypatch.setattr(cli, "_save", lambda *a, **kw: None)
    monkeypatch.setattr(roots, "root_edges_for", lambda *a, **kw: ())
    received = {}

    def measure(*a, **kw):
        received.update(kw)
        return result

    monkeypatch.setattr(rp, "measure_radii", measure)
    assert cli.cmd_radius_perimeter(args) == 0
    assert received["workers"] == 3
