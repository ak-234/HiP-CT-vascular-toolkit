"""Saving a reformat stack, and getting the same object back.

The claim being pinned is not "the pixels survive" -- that would be true of `np.save`.
It is that **the geometry survives with them**: a reloaded stack has to drive the same
overlay readout (arclength, radius, um/px, local curvature) and the same 3D plane frames
as the one the builder produced. So the round-trip tests compare `describe()` and
`plane_corners()`, not just the arrays.

Three containers, one document, and they must not drift apart -- hence the
parametrisation over every format rather than one test for the one anybody uses.

The refusal boundary is the other thing worth stating once: a **schema** mismatch raises,
because the file cannot be understood; a **frame** or **graph** mismatch only notes,
because the file is understood perfectly well and merely does not belong to what is
loaded right now. The images are worth looking at either way.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hipct_seg_debug import reformat as rf
from hipct_seg_debug import reformat_io as rio

from .conftest_geometry import graph_from
from .test_reformat import FakeStack, tilted_cylinder, unit_frame

SHAPE = (70, 70, 70)
AXIS = np.array([1.0, 0.6, 0.45])


def _built(*, with_mask=True, size_px=21, mode="native", n_points=40):
    """A small real stack, sampled from a tilted cylinder phantom."""
    direction = AXIS / np.linalg.norm(AXIS)
    labels = tilted_cylinder(SHAPE, AXIS, 7.0)
    volume = (labels * 500 + 100).astype(np.uint16)
    ends = np.array([35.0, 35.0, 35.0]) + np.array([-18.0, 18.0])[:, None] * direction
    graph = graph_from([tuple(ends[0]), tuple(ends[1])], [(0, 1, n_points, 7.0)])
    frame = unit_frame(SHAPE)
    stack = rf.build(
        graph, frame, FakeStack(volume), [0],
        labels=labels if with_mask else None,
        mode=mode, size_px=size_px, step_um=1.0, radii_k=3.0,
    )
    return stack, frame, graph


@pytest.fixture(scope="module")
def built():
    return _built()


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", rio.FORMATS)
def test_a_saved_stack_reloads_as_the_same_object(built, fmt, tmp_path):
    """Not just the pixels -- the geometry that makes them readable."""
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / f"run_{fmt}", fmt=fmt, frame=frame, graph=graph)
    back = rio.load(path).reformat

    np.testing.assert_array_equal(back.raw, stack.raw)
    assert back.raw.dtype == stack.raw.dtype
    np.testing.assert_array_equal(back.mask, stack.mask)

    for name in ("coords_um", "radii_um", "arclen_um", "tangents",
                 "normals", "binormals", "seg_ids"):
        np.testing.assert_array_equal(
            getattr(back.centreline, name), getattr(stack.centreline, name)
        )
    for name in ("half_um", "px_um", "requested_um"):
        np.testing.assert_array_equal(
            getattr(back.geometry, name), getattr(stack.geometry, name)
        )
    for name in ("theta_rad", "ds_um", "r_step_um", "r_point_um"):
        np.testing.assert_array_equal(
            getattr(back.centreline.curvature, name),
            getattr(stack.centreline.curvature, name),
        )
    np.testing.assert_array_equal(back.graph_points, stack.graph_points)


@pytest.mark.parametrize("fmt", rio.FORMATS)
def test_the_report_a_reloaded_stack_prints_is_identical(built, fmt, tmp_path):
    """One assertion over the whole reporting path -- what the operator actually reads."""
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / f"run_{fmt}", fmt=fmt, frame=frame, graph=graph)
    assert rio.load(path).reformat.describe() == stack.describe()


@pytest.mark.parametrize("fmt", rio.FORMATS)
def test_the_3d_plane_geometry_survives(built, fmt, tmp_path):
    """The corners the 3D window draws come from the centreline and the half-widths."""
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / f"run_{fmt}", fmt=fmt, frame=frame, graph=graph)
    back = rio.load(path).reformat
    np.testing.assert_allclose(rf.plane_corners(back), rf.plane_corners(stack), atol=0)


@pytest.mark.parametrize("fmt", rio.FORMATS)
def test_no_mask_reloads_as_none_not_as_zeros(fmt, tmp_path):
    """"No segmentation was sampled" and "the segmentation was empty" differ."""
    stack, frame, graph = _built(with_mask=False)
    assert stack.mask is None
    path = rio.save(stack, tmp_path / f"run_{fmt}", fmt=fmt, frame=frame, graph=graph)
    assert rio.load(path).reformat.mask is None


@pytest.mark.parametrize("mode", ["native", "radius", "fixed"])
def test_every_plane_mode_round_trips(mode, tmp_path):
    """`radius` mode has a per-plane pitch, which is the one that could be flattened."""
    stack, frame, graph = _built(mode=mode)
    back = rio.load(rio.save(stack, tmp_path / mode, fmt="npz", frame=frame)).reformat
    assert back.geometry.mode == mode
    np.testing.assert_array_equal(back.geometry.px_um, stack.geometry.px_um)
    np.testing.assert_allclose(back.geometry.oversampling, stack.geometry.oversampling)


def test_a_straight_run_keeps_its_infinite_curvature(tmp_path):
    """JSON has no infinity, and a straight vessel genuinely has no curvature bound."""
    pts = np.linspace([10.0, 35.0, 35.0], [60.0, 35.0, 35.0], 40)
    graph = graph_from([tuple(pts[0]), tuple(pts[-1])], [(0, 1, 40, 5.0)])
    frame = unit_frame(SHAPE)
    stack = rf.build(graph, frame, FakeStack(np.zeros(SHAPE, dtype=np.uint16)), [0],
                     mode="native", size_px=15, step_um=1.0)
    assert not np.isfinite(stack.centreline.curvature.r_min_um)

    back = rio.load(rio.save(stack, tmp_path / "straight", fmt="npz")).reformat
    assert not np.isfinite(back.centreline.curvature.r_min_um)
    assert not np.isfinite(back.geometry.r_min_um)


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


def test_the_name_carries_mode_frame_size_and_segments(built):
    stack, _frame, _graph = built
    assert rio.suggest_name(stack, "LAD") == "LAD__native_21px__seg0"


def test_a_long_selection_collapses_to_a_count(tmp_path):
    """A chain of a dozen segments must not become a filename of a dozen numbers."""
    nodes = [(10.0 + 4 * i, 35.0, 35.0) for i in range(13)]
    edges = [(i, i + 1, 8, 5.0) for i in range(12)]
    graph = graph_from(nodes, edges)
    frame = unit_frame(SHAPE)
    stack = rf.build(graph, frame, FakeStack(np.zeros(SHAPE, dtype=np.uint16)),
                     list(range(12)), mode="native", size_px=15, step_um=1.0)
    name = rio.suggest_name(stack, "RCA")
    assert name == "RCA__native_15px__seg0+11more"
    assert len(name) < 60


def test_the_name_describes_the_run_built_not_the_selection_asked_for(tmp_path):
    """Those differ when the selection was not one connected chain."""
    nodes = [(10.0, 35.0, 35.0), (30.0, 35.0, 35.0), (50.0, 35.0, 35.0),
             (10.0, 60.0, 35.0), (30.0, 60.0, 35.0)]
    edges = [(0, 1, 10, 5.0), (1, 2, 10, 5.0), (3, 4, 10, 5.0)]
    graph = graph_from(nodes, edges)
    stack = rf.build(graph, unit_frame(SHAPE), FakeStack(np.zeros(SHAPE, np.uint16)),
                     [0, 1, 2], mode="native", size_px=15, step_um=1.0)
    # Segment 2 is disconnected and was not sampled, so it is not in the name.
    assert rio.suggest_name(stack, "LAD") == "LAD__native_15px__seg0-1"


@pytest.mark.parametrize("base,expect", [
    ("LAD/RCA: run 1", "LAD_RCA__run_1"),
    ("  trailing. ", "trailing"),
    ("", "reformat"),
    ("a<b>c|d?e*f", "a_b_c_d_e_f"),
])
def test_an_awkward_base_name_is_made_safe(built, base, expect):
    """Windows rejects some of these outright and silently eats a trailing dot."""
    stack, _frame, _graph = built
    assert rio.suggest_name(stack, base).startswith(expect + "__")


# --------------------------------------------------------------------------- #
# Provenance, and the refusal boundary
# --------------------------------------------------------------------------- #


def test_a_schema_mismatch_raises_and_names_both_schemas(built, tmp_path):
    """The one thing that is a refusal: a file that cannot be understood."""
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / "run", fmt="tiff", frame=frame, graph=graph)
    doc = json.loads((path / rio.DOCUMENT_NAME).read_text(encoding="utf-8"))
    doc["schema"] = "hipct.reformat/99"
    (path / rio.DOCUMENT_NAME).write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(rio.ReformatIOError, match="hipct.reformat/1.*hipct.reformat/99"):
        rio.load(path)


def test_a_different_frame_is_a_note_not_a_refusal(built, tmp_path):
    """The images are unaffected; only their world positions stop meaning anything."""
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / "run", fmt="npz", frame=frame, graph=graph)

    other = unit_frame((90, 90, 90))
    loaded = rio.check_against(rio.load(path), frame=other)
    assert loaded.frame_matches is False
    assert any("3D plane overlays are not drawn" in n for n in loaded.notes)
    # ...and it is still a usable stack.
    np.testing.assert_array_equal(loaded.reformat.raw, stack.raw)


def test_the_matching_frame_is_accepted_silently(built, tmp_path):
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / "run", fmt="npz", frame=frame, graph=graph)
    loaded = rio.check_against(rio.load(path), frame=frame, graph=graph)
    assert loaded.frame_matches is True
    assert loaded.notes == []


def test_a_changed_graph_is_a_note_only(built, tmp_path):
    """Reformatting a repaired graph is the intended workflow, as it is for crop."""
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / "run", fmt="npz", frame=frame, graph=graph)

    bigger = graph_from(
        [(10.0, 35.0, 35.0), (30.0, 35.0, 35.0), (50.0, 35.0, 35.0)],
        [(0, 1, 10, 5.0), (1, 2, 10, 5.0)],
    )
    loaded = rio.check_against(rio.load(path), frame=frame, graph=bigger)
    assert loaded.frame_matches is True
    assert any("graph has changed" in n for n in loaded.notes)


def test_provenance_records_the_segments_twice(built, tmp_path):
    """By id, and by a geometric key that survives the renumbering a deletion causes."""
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / "run", fmt="tiff", frame=frame, graph=graph)
    prov = rio.load(path).provenance

    assert prov["segments"] == [0]
    assert len(prov["segment_keys"]) == 1 and len(prov["segment_keys"][0]) == 16
    assert prov["source_counts"]["segments"] == len(graph.segments)
    assert prov["frame"]["raw_shape"] == list(SHAPE)


def test_saving_without_a_graph_or_frame_still_works(built, tmp_path):
    """Provenance is a courtesy; it must never be what stops a save."""
    stack, _frame, _graph = built
    loaded = rio.load(rio.save(stack, tmp_path / "bare", fmt="npz"))
    assert loaded.provenance.get("segments") == [0]
    assert "frame" not in loaded.provenance
    # With nothing recorded there is nothing to check against, so it says so.
    checked = rio.check_against(loaded, frame=unit_frame(SHAPE))
    assert checked.frame_matches is False
    assert any("no frame was recorded" in n for n in checked.notes)


# --------------------------------------------------------------------------- #
# The containers themselves
# --------------------------------------------------------------------------- #


def test_the_tiff_folder_opens_in_anything_that_reads_a_tiff(built, tmp_path):
    """The whole reason that format exists -- Fiji, without a script."""
    tifffile = pytest.importorskip("tifffile")
    stack, frame, graph = built
    path = rio.save(stack, tmp_path / "run", fmt="tiff", frame=frame, graph=graph)

    np.testing.assert_array_equal(tifffile.imread(path / "raw.tif"), stack.raw)
    np.testing.assert_array_equal(tifffile.imread(path / "mask.tif"), stack.mask)
    readme = (path / "README.txt").read_text(encoding="utf-8")
    assert "plane index" in readme and "centre pixel" in readme
    assert "um per plane" in readme and "um per pixel" in readme


def test_the_npz_is_one_file_and_the_folders_are_folders(built, tmp_path):
    stack, frame, _graph = built
    assert rio.save(stack, tmp_path / "a", fmt="npz", frame=frame).is_file()
    assert rio.save(stack, tmp_path / "b", fmt="tiff", frame=frame).is_dir()
    assert rio.save(stack, tmp_path / "c", fmt="npy", frame=frame).is_dir()


def test_a_missing_npz_suffix_is_supplied(built, tmp_path):
    stack, _frame, _graph = built
    assert rio.save(stack, tmp_path / "noext", fmt="npz").name == "noext.npz"


def test_a_folder_loads_from_its_document_as_well_as_from_itself(built, tmp_path):
    """One file dialog serves all three formats by accepting geometry.json."""
    stack, frame, _graph = built
    path = rio.save(stack, tmp_path / "run", fmt="npy", frame=frame)
    by_dir = rio.load(path).reformat
    by_doc = rio.load(path / rio.DOCUMENT_NAME).reformat
    np.testing.assert_array_equal(by_dir.raw, by_doc.raw)


def test_an_unknown_format_is_refused_by_name(built, tmp_path):
    stack, _frame, _graph = built
    with pytest.raises(rio.ReformatIOError, match="unknown format"):
        rio.save(stack, tmp_path / "x", fmt="hdf5")


def test_nonsense_files_report_the_file_rather_than_a_numpy_traceback(tmp_path):
    (tmp_path / "junk.npz").write_bytes(b"not an npz at all")
    with pytest.raises(rio.ReformatIOError, match="not a readable reformat stack"):
        rio.load(tmp_path / "junk.npz")

    (tmp_path / "empty").mkdir()
    with pytest.raises(rio.ReformatIOError, match="not a saved reformat stack"):
        rio.load(tmp_path / "empty")

    with pytest.raises(rio.ReformatIOError, match="does not exist"):
        rio.load(tmp_path / "absent.npz")


def test_loading_never_unpickles(built, tmp_path):
    """A stack is data. A file that can execute on open is not what was handed over.

    Checked against the parse tree rather than by counting substrings: the phrase also
    appears in prose in this module, so a text search passes for the wrong reason.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(rio))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "load"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "np"
    ]
    assert calls, "no np.load calls found -- has the reader been rewritten?"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "allow_pickle" in kw, f"np.load at line {call.lineno} does not pin it"
        assert kw["allow_pickle"].value is False, f"np.load at line {call.lineno}"

    stack, frame, _graph = built
    path = rio.save(stack, tmp_path / "run", fmt="npz", frame=frame)
    with np.load(path, allow_pickle=False) as z:   # must not need pickling to read
        assert "raw" in z.files and "meta" in z.files
