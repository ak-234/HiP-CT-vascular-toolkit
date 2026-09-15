"""The command line and the audit trail.

Two things are being pinned here and they are both about *files* rather than about
algorithms.

The first is the paired-output rule. ``connect --geodesic`` changes the graph and
the segmentation together, so writing one without the other produces two documents
that disagree about where the vessels are -- and nothing downstream can tell which
to believe. The CLI refuses, and this is where that refusal is held.

The second is that the audit trail round-trips. A decisions file that cannot be
read back, or that matches an operator's ruling onto the wrong candidate, is worse
than no audit trail at all: it looks like a record and is not one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hipct_seg_debug.edit.__main__ import build_parser, cmd_connect
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.amira_write import write_spatial_graph
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.reconnect.geodesic import apply as apply_mod
from hipct_seg_debug.edit.reconnect.geodesic import (
    GeodesicParams,
    audit,
    components,
    plan,
)
from hipct_seg_debug.rle_write import write_lattice

from .conftest_geodesic import (
    SHAPE,
    SPACING,
    broken_graph,
    make_frame,
    mask_source,
    slit,
)


@pytest.fixture
def dataset(tmp_path):
    """A broken slit vessel written out as a real ``.am`` pair."""
    frame = make_frame(SHAPE)
    volume = (slit(SHAPE, 6, 1, 5, 28) | slit(SHAPE, 6, 1, 30, 55)).astype(np.uint8)
    dims = np.array([SHAPE[2], SHAPE[1], SHAPE[0]])
    bbox = np.empty(6)
    bbox[0::2] = 0.0
    bbox[1::2] = (dims - 1) * SPACING
    seg = tmp_path / "seg.am"
    write_lattice(seg, volume, bbox)

    graph = broken_graph(frame, (5, 28), (30, 55))
    path = tmp_path / "graph.am"
    write_spatial_graph(graph.to_spatial_graph(), path)
    return {"graph": path, "seg": seg, "dir": tmp_path, "volume": volume,
            "frame": frame}


def _args(dataset, **overrides):
    argv = ["connect", str(dataset["graph"]), "--geodesic",
            "--seg", str(dataset["seg"]), "--voxel-um", str(SPACING)]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value not in (None, False):
            argv.extend([flag, str(value)])
    return build_parser().parse_args(argv)


# ------------------------------------------------------------------ the guards


def test_geodesic_and_dpc_are_mutually_exclusive(dataset):
    args = _args(dataset, dpc=True, raw="somewhere")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        cmd_connect(args)


def test_writing_the_graph_alone_is_refused(dataset):
    args = _args(dataset, out=str(dataset["dir"] / "out.am"))
    with pytest.raises(SystemExit, match="--out-seg"):
        cmd_connect(args)


def test_writing_the_mask_alone_is_refused(dataset):
    args = _args(dataset, out_seg=str(dataset["dir"] / "out-seg.am"))
    with pytest.raises(SystemExit, match="both"):
        cmd_connect(args)


def test_out_seg_without_geodesic_is_refused(dataset):
    argv = ["connect", str(dataset["graph"]), "--seg", str(dataset["seg"]),
            "--out-seg", str(dataset["dir"] / "x.am")]
    with pytest.raises(SystemExit, match="only meaningful with --geodesic"):
        cmd_connect(build_parser().parse_args(argv))


# ---------------------------------------------------------------------- runs


def test_dry_run_writes_no_graph_and_no_mask(dataset, capsys):
    assert cmd_connect(_args(dataset)) == 0
    printed = capsys.readouterr().out
    assert "dry run" in printed
    assert not list(dataset["dir"].glob("out*.am"))


def test_dry_run_still_writes_the_review_file(dataset):
    """Reviewing is not a write to the dataset, so a dry run may still produce it."""
    review = dataset["dir"] / "review.json"
    assert cmd_connect(_args(dataset, review_json=str(review))) == 0
    assert review.exists()
    document = json.loads(review.read_text(encoding="utf-8"))
    assert document["schema"] == audit.SCHEMA
    assert document["kind"] == "review"
    assert set(document["counts"]) == {"for_review", "accepted", "rejected"}


def test_a_full_run_writes_a_consistent_graph_and_mask(dataset):
    out_graph = dataset["dir"] / "out.am"
    out_seg = dataset["dir"] / "out-seg.am"
    decisions = dataset["dir"] / "decisions.json"

    assert cmd_connect(_args(dataset, out=str(out_graph), out_seg=str(out_seg),
                             decisions_json=str(decisions))) == 0
    assert out_graph.exists() and out_seg.exists()

    from hipct_seg_debug import amira, rle

    info = amira.read_lattice_header(out_seg)
    lattice = rle.ByteRLELattice(out_seg, info.fields["Labels"], info.dims)
    written = np.stack([lattice.slice_z(k) for k in range(lattice.nz)])

    # The two files have to tell the same story: one mask component and one graph
    # component, from two of each.
    assert components.from_array(written > 0).n == 1
    graph = EditableGraph(read_triple(out_graph))
    assert len(graph.components()) == 1
    # And the mask grew rather than being replaced wholesale.
    assert (written > 0).sum() > (dataset["volume"] > 0).sum()


def test_provenance_survives_the_amira_round_trip(dataset):
    """The columns must come back off disk, or the audit is only in memory."""
    out_graph = dataset["dir"] / "out.am"
    out_seg = dataset["dir"] / "out-seg.am"
    cmd_connect(_args(dataset, out=str(out_graph), out_seg=str(out_seg)))

    graph = EditableGraph(read_triple(out_graph))
    origins = [int(s.get(apply_mod.ORIGIN_FIELD, apply_mod.ORIGINAL))
               for s in graph.segments]
    assert apply_mod.ORIGINAL in origins, "the original edges lost their provenance"
    assert set(origins) - {apply_mod.ORIGINAL}, "no repaired edge was marked"
    for segment in graph.segments:
        assert apply_mod.SCORE_FIELD in segment
        assert apply_mod.REVIEWED_FIELD in segment

    counts = apply_mod.origin_counts(graph)
    assert counts.get("original") == 2


# --------------------------------------------------------------------- audit


def test_decisions_document_records_components_evidence_and_topology(dataset):
    frame = dataset["frame"]
    source = mask_source(dataset["volume"])
    index = components.build(source)
    graph = broken_graph(frame, (5, 28), (30, 55))
    result = plan(graph, index, frame, params=GeodesicParams(alternatives=1))
    applied = apply_mod.apply_plan(graph, result, source, frame)

    document = audit.decisions_document(result, frame, graph=graph, applied=applied)
    # It must be plain JSON, with no numpy and no non-finite tokens left in it.
    text = json.dumps(document)
    assert "Infinity" not in text and "NaN" not in text

    assert document["topology"]["graph_components_after"] == 1
    record = next(c for c in document["candidates"] if c["decision"]["accepted"])
    assert record["source"]["component"] and record["target"]["component"]
    assert record["source"]["component"] != record["target"]["component"]
    assert "mean_support" in record["evidence"]
    assert "alternatives" in record and "route" in record
    assert record["applied"]["ok"] is True


def test_a_decision_is_matched_by_endpoint_not_by_position(dataset):
    """A re-run on a changed graph must not apply a ruling to the wrong candidate."""
    frame = dataset["frame"]
    source = mask_source(dataset["volume"])
    index = components.build(source)
    graph = broken_graph(frame, (5, 28), (30, 55))
    result = plan(graph, index, frame, params=GeodesicParams(alternatives=1))

    document = audit.decisions_document(result, frame, graph=graph)
    # An operator accepts exactly one candidate, identified by its endpoints.
    target = document["candidates"][-1]
    target["decision"]["operator"] = {"accept": True}
    key = audit._key_of_record(target)

    fresh = plan(graph, index, frame, params=GeodesicParams(alternatives=1))
    # Reverse the order, so a positional match would pick the wrong one.
    fresh.candidates.reverse()
    approved, unmatched = audit.apply_decisions(fresh, document)

    assert not unmatched
    assert len(approved) == 1
    assert audit._key_of_candidate(approved[0]) == key
    # The ruling came from the document's *last* record. After the reversal a
    # different candidate sits at that index, so a positional match would have
    # accepted the wrong reconnection -- and did not.
    assert audit._key_of_candidate(fresh.candidates[-1]) != key


def test_operator_waypoints_are_carried_onto_the_candidate(dataset):
    frame = dataset["frame"]
    source = mask_source(dataset["volume"])
    index = components.build(source)
    graph = broken_graph(frame, (5, 28), (30, 55))
    result = plan(graph, index, frame, params=GeodesicParams(alternatives=1))

    document = audit.decisions_document(result, frame, graph=graph)
    document["candidates"][0]["decision"]["operator"] = {
        "accept": True, "waypoints_um": [[290.0, 200.0, 200.0]],
    }
    approved, _unmatched = audit.apply_decisions(result, document)
    assert approved
    np.testing.assert_allclose(approved[0].waypoints[0], [290.0, 200.0, 200.0])


def test_a_rejection_by_the_operator_overrides_an_accept(dataset):
    frame = dataset["frame"]
    source = mask_source(dataset["volume"])
    index = components.build(source)
    graph = broken_graph(frame, (5, 28), (30, 55))
    result = plan(graph, index, frame, params=GeodesicParams(alternatives=1))

    document = audit.decisions_document(result, frame, graph=graph)
    document["candidates"][0]["decision"]["operator"] = {
        "accept": False, "reason": "not a real vessel",
    }
    approved, _unmatched = audit.apply_decisions(result, document)
    assert not approved
    assert result.candidates[0].status == "reject"
    assert result.candidates[0].reason == "not a real vessel"


def test_a_foreign_schema_is_refused(tmp_path):
    """A document from another tool must not be silently interpreted as ours."""
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "something-else", "candidates": []}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        audit.load_decisions(path)


def test_a_non_finite_measurement_is_written_as_null(tmp_path):
    """`NaN` and `Infinity` are not JSON, and a document nothing can load is not a record.

    The check has to catch **plain** floats, not only ``np.floating``: a route is an
    array, ``_plain`` recurses through ``tolist()``, and that hands back Python floats.
    A single non-finite coordinate would otherwise make the whole audit trail
    unreadable by any strict parser -- which is every parser but Python's own.
    """
    document = audit._plain({
        "route": np.array([[1.0, np.nan, np.inf]]),
        "scalar": float("nan"),
        "numpy": np.float64("-inf"),
        "kept": [1.5, 2, True],
    })
    text = json.dumps(document)
    assert "NaN" not in text and "Infinity" not in text

    def refuse(token):  # what a strict parser does with a bare NaN token
        raise AssertionError(f"unloadable JSON constant {token!r}")

    back = json.loads(text, parse_constant=refuse)
    assert back["route"] == [[1.0, None, None]]
    assert back["scalar"] is None and back["numpy"] is None
    assert back["kept"] == [1.5, 2, True]


def test_a_written_document_round_trips_through_a_strict_parser(dataset):
    """The whole file, not just a hand-built dict."""
    decisions = dataset["dir"] / "decisions.json"
    assert cmd_connect(_args(dataset, decisions_json=str(decisions))) == 0

    def refuse(token):
        raise AssertionError(f"unloadable JSON constant {token!r}")

    document = json.loads(decisions.read_text(encoding="utf-8"), parse_constant=refuse)
    assert document["schema"] == audit.SCHEMA


def test_reading_back_a_decisions_file_applies_it_on_the_next_run(dataset, capsys):
    """The review loop, end to end: dry run, rule, re-run, and the ruling lands."""
    decisions = dataset["dir"] / "decisions.json"
    assert cmd_connect(_args(dataset, decisions_json=str(decisions))) == 0

    document = json.loads(decisions.read_text(encoding="utf-8"))
    for record in document["candidates"]:
        if record["status"] == "review":
            record["decision"]["operator"] = {"accept": True}
    decisions.write_text(json.dumps(document), encoding="utf-8")

    capsys.readouterr()
    assert cmd_connect(_args(dataset, decisions_json=str(decisions))) == 0
    printed = capsys.readouterr().out
    assert "approved from" in printed


# ---------------------------------------------------- endpoints from the mask


def test_the_mask_end_sweep_is_on_by_default_and_can_be_turned_off(dataset, capsys):
    """A break whose far side was pruned away is invisible without it."""
    from hipct_seg_debug.edit.__main__ import cmd_connect as run

    assert run(_args(dataset)) == 0
    on = capsys.readouterr().out
    assert "lumen no centreline describes" in on

    assert run(_args(dataset, no_mask_endpoints=True)) == 0
    off = capsys.readouterr().out
    assert "lumen no centreline describes" not in off
    assert "mask free end" not in off


def test_the_describedness_knob_reaches_the_sweep(dataset):
    args = _args(dataset, describe_radii=3.5, min_lobe_voxels=44)
    assert args.describe_radii == 3.5
    assert args.min_lobe_voxels == 44
    from hipct_seg_debug.edit.__main__ import cmd_connect as run

    assert run(args) == 0


def test_a_mask_end_ruling_round_trips_through_the_decisions_file(dataset, tmp_path):
    """The tip voxel is the identity, and it has to survive being written out."""
    from hipct_seg_debug.edit.reconnect.geodesic import lobes

    end = lobes.MaskEnd(key=(9, 20, 30), point_um=np.zeros(3),
                        tangent=np.array([1.0, 0.0, 0.0]), radius_um=10.0,
                        component=2, lobe_voxels=80, elongation=4.0,
                        skeleton_um=np.zeros((2, 3)))
    from hipct_seg_debug.edit.reconnect.geodesic import classify, route

    def association(node, component):
        return classify.Association(node=node, point_um=np.zeros(3),
                                    tangent=np.zeros(3), radius_um=10.0,
                                    index_zyx=np.zeros(3, np.int64),
                                    component=component, distance_vox=0.0)

    classified = classify.Classified(kind="geodesic", source=association(3, 1),
                                     target=association(-1, 2),
                                     target_mask_end=end)
    from hipct_seg_debug.edit.reconnect.geodesic import select

    candidate = route.Candidate(classified=classified, status="review")
    decision = select.Decision(candidate, False, "review", "needs a person")
    plan_ = route.Plan(candidates=[candidate], decisions=[decision],
                       associations={}, fragments=[])

    document = audit.decisions_document(plan_)
    document["candidates"][0]["decision"]["operator"] = {"accept": True}
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    approved, unmatched = audit.apply_decisions(plan_, audit.load_decisions(path))
    assert approved == [candidate]
    assert unmatched == []
