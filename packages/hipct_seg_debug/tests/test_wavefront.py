"""The whole wavefront pipeline, on the gaps it was built for.

The headline case is the one the sibling connector cannot even propose: a break
thirty radii long whose lumen the image still shows. Everything else here pins
the contract with the shared machinery -- the same ``Plan`` and ``Candidate``,
the same selector, the same apply and audit -- and the two things this package
adds to a decision: the refinement report and the tensor evidence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect import geodesic, wavefront
from hipct_seg_debug.edit.reconnect.geodesic import apply as apply_mod
from hipct_seg_debug.edit.reconnect.geodesic import audit, components
from hipct_seg_debug.edit.reconnect.wavefront import WavefrontParams

from .conftest_geodesic import (
    CZ,
    WALL,
    FakeStack,
    broken_graph,
    cylinder,
    decode,
    make_frame,
    mask_source,
    ribbon,
    ribbon_gap,
    slit,
)

SHAPE = (40, 40, 60)
LONG = (40, 40, 120)
FAST = WavefrontParams(alternatives=2, refine_method="none")


def _plan(volume, graph, *, shape=SHAPE, image=None, params=FAST, labels=None):
    frame = make_frame(shape)
    source = mask_source(np.asarray(volume, dtype=np.uint8))
    index = components.build(source)
    stack = None if image is None else FakeStack(image)
    result = wavefront.plan(graph, index, frame, labels, stack=stack, params=params)
    return result, index, frame, source


# ------------------------------------------------------------- short gaps


@pytest.mark.parametrize("name", ["tube", "ribbon", "slit"])
def test_a_two_voxel_gap_is_repaired_in_every_cross_section(name):
    frame = make_frame(SHAPE)
    builder = {
        "tube": lambda a, b: cylinder(SHAPE, 3, a, b),
        "ribbon": lambda a, b: ribbon(SHAPE, 6, 2, a, b),
        "slit": lambda a, b: slit(SHAPE, 6, 0, a, b),
    }[name]
    volume = builder(5, 28) | builder(30, 55)
    graph = broken_graph(frame, (5, 28), (30, 55))
    result, index, _frame, _source = _plan(volume, graph)
    assert index.n == 2
    accepted = result.accepted()
    assert len(accepted) == 1, result.summarise()
    candidate = accepted[0]
    assert candidate.kind == "geodesic"
    assert candidate.evidence["strategy"] == "dual_front"
    assert candidate.evidence["mask_gap_voxels"] <= 2
    assert "saddle_cost" in candidate.route.metrics
    assert result.stats["refinement"]["skipped"]
    assert result.stats["end_profiles"]["profiled"] == 4
    assert result.stats["engine"] == "lattice"


def test_plan_is_the_shared_plan_type_and_applies_with_wavefront_provenance():
    frame = make_frame(SHAPE)
    volume = ribbon(SHAPE, 6, 2, 5, 28) | ribbon(SHAPE, 6, 2, 30, 55)
    graph = broken_graph(frame, (5, 28), (30, 55))
    result, index, frame, source = _plan(volume, graph)
    assert isinstance(result, geodesic.Plan)
    before = decode(source).sum()
    applied = apply_mod.apply_plan(graph, result, source, frame,
                                   origin=apply_mod.WAVEFRONT)
    assert [a.ok for a in applied] == [True]
    assert applied[0].origin == apply_mod.WAVEFRONT
    assert len(graph.components()) == 1
    assert decode(source).sum() > before
    counts = apply_mod.origin_counts(graph)
    assert counts.get("wavefront", 0) >= 1 or counts.get("reskeletonised", 0) >= 1
    for segment in graph.segments:
        origin = int(segment.get(apply_mod.ORIGIN_FIELD, apply_mod.ORIGINAL))
        assert origin in (apply_mod.ORIGINAL, apply_mod.WAVEFRONT,
                          apply_mod.RESKELETONISED)


# -------------------------------------------------------------- long gaps


def test_a_thirty_radius_pinched_slit_is_repaired_where_the_sibling_cannot_propose():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    frame = make_frame(LONG)
    graph = broken_graph(frame, (5, 30), (90, 115))
    index = components.build(mask_source(mask))
    stack = FakeStack(image)

    sibling = geodesic.plan(graph, index, frame, stack=stack,
                            params=geodesic.GeodesicParams(alternatives=2))
    assert sibling.stats["proposals"] == 0, "the reach gate should refuse this pair"

    result = wavefront.plan(graph, index, frame, None, stack=stack, params=FAST)
    accepted = result.accepted()
    assert len(accepted) == 1, result.summarise()
    candidate = accepted[0]
    assert candidate.evidence["span_um"] > 580.0
    assert candidate.evidence["alignment"] > 0.8
    assert candidate.evidence["normal_crossing"] < 0.3
    assert candidate.evidence["mask_gap_voxels"] >= 55
    assert candidate.evidence["route_tortuosity"] < 1.1
    assert candidate.completion is not None and len(candidate.completion.voxels_zyx) > 50
    path = candidate.route.path_zyx
    assert np.all(np.abs(path[:, 0] - CZ) <= 1)  # in the slit plane throughout


def test_a_dropout_longer_than_the_allowance_is_refused_with_the_chain_reported():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    image[:, :, 50:70] = WALL + np.random.default_rng(1).normal(0.0, 4.0, (40, 40, 20))
    frame = make_frame(LONG)
    graph = broken_graph(frame, (5, 30), (90, 115))
    result, *_ = _plan(mask, graph, shape=LONG, image=image)
    assert not result.accepted()
    candidate = next(c for c in result.candidates if c.source_node == 1)
    assert "unsupported" in candidate.reason
    assert candidate.evidence["strategy"] == "chain"
    assert candidate.evidence["chain"]["stop_reason"] == "found"


def test_the_chain_fallback_can_be_switched_off():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    image[:, :, 50:70] = WALL + np.random.default_rng(1).normal(0.0, 4.0, (40, 40, 20))
    frame = make_frame(LONG)
    graph = broken_graph(frame, (5, 30), (90, 115))
    params = WavefrontParams(alternatives=2, refine_method="none", chain_fallback=False)
    result, *_ = _plan(mask, graph, shape=LONG, image=image, params=params)
    candidate = next(c for c in result.candidates if c.source_node == 1)
    assert candidate.evidence["strategy"] == "dual_front"
    assert "chain" not in candidate.evidence


def test_a_bend_into_a_parallel_vessel_is_blocked_not_routed():
    mask = ribbon(SHAPE, 6, 2, 5, 28) | ribbon(SHAPE, 6, 2, 30, 55)
    rival = ribbon(SHAPE, 6, 2, 5, 55, cz=CZ + 7)
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 28), (30, 55))
    result, index, *_ = _plan(mask | rival, graph)
    assert index.n == 3
    accepted = result.accepted()
    assert len(accepted) == 1
    assert accepted[0].evidence["competing_components"]
    path = accepted[0].route.path_zyx
    assert np.all(np.abs(path[:, 0] - CZ) <= 1)


# ---------------------------------------------------------- open ends


def test_open_end_exploration_finds_a_target_the_proposer_had_none_for():
    """The far side is there, but faces the wrong way for the geometric cone."""
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    frame = make_frame(LONG)
    graph = broken_graph(frame, (5, 30), (90, 115))
    index = components.build(mask_source(mask))
    stack = FakeStack(image)
    # A proposal cone of one degree leaves both pairs unproposed...
    params = WavefrontParams(alternatives=2, refine_method="none", explore_open_ends=True,
                             mask_endpoints=False)
    result = wavefront.plan(graph, index, frame, None, stack=stack, params=params,
                            gate_kwargs={"cone_angle_deg": 1.0, "cone_length_factor": 2.0})
    assert result.stats["proposals"] == 0
    # ...and exploration supplies them, through the same evidence as any other.
    assert result.stats["exploration"]["reached"] >= 1
    accepted = result.accepted()
    assert len(accepted) == 1, result.summarise()
    assert accepted[0].proposal.metrics.get("explored") is True
    assert accepted[0].evidence["strategy"] in ("dual_front", "chain")


# -------------------------------------------------------------- refinement


def test_refinement_runs_inside_plan_and_reports():
    from hipct_seg_debug.edit.centreline_synthetic_benchmark import fixture

    graph, frame, labels, _amp = fixture("flat", 3)
    index = components.build(mask_source(labels))
    params = WavefrontParams(alternatives=1, refine_method="laplacian",
                             refine_max_iterations=3, mask_endpoints=False)
    result = wavefront.plan(graph, index, frame, labels, params=params)
    assert result.stats["refinement"]["method"] == "laplacian"
    assert result.stats["refinement"]["moved_points"] > 0
    assert "centreline_displacement_um" in graph.triple.point_attrs
    assert result.stats["proposals"] == 0  # one intact vessel, nothing to join


# ------------------------------------------------------------------ audit


def test_decisions_document_carries_the_tensor_evidence():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    frame = make_frame(LONG)
    graph = broken_graph(frame, (5, 30), (90, 115))
    result, index, frame, source = _plan(mask, graph, shape=LONG, image=image)
    applied = apply_mod.apply_plan(graph, result, source, frame,
                                   origin=apply_mod.WAVEFRONT)
    document = audit.decisions_document(result, frame, graph=graph, applied=applied)
    text = json.dumps(document)
    assert "Infinity" not in text and "NaN" not in text
    record = next(c for c in document["candidates"] if c["decision"]["accepted"])
    assert record["evidence"]["strategy"] == "dual_front"
    assert "alignment" in record["evidence"]
    assert "end_profiles" in record["evidence"]
