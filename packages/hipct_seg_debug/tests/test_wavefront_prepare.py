"""The cleaning stage: refinement first, then a measured profile per free end."""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.centreline_synthetic_benchmark import fixture
from hipct_seg_debug.edit.reconnect.candidates import endpoint_tangent
from hipct_seg_debug.edit.reconnect.geodesic import components
from hipct_seg_debug.edit.reconnect.wavefront import prepare

from .conftest_geodesic import SHAPE, broken_graph, make_frame, mask_source, ribbon, slit


def _degrees_off_x(direction) -> float:
    return float(np.degrees(np.arccos(abs(float(np.dot(direction, [1.0, 0.0, 0.0]))))))


@pytest.fixture(scope="module")
def jittered():
    """A flat lumen whose centreline zig-zags: the benchmark's 'flat' phantom."""
    graph, frame, labels, _amplitude = fixture("flat", 3)
    return graph, frame, labels, components.build(mask_source(labels))


def test_two_point_tangent_is_noise_on_a_jittered_skeleton(jittered):
    graph, frame, _labels, _index = jittered
    worst = max(_degrees_off_x(endpoint_tangent(graph, n)[0]) for n in graph.endpoints())
    assert worst > 15.0


def test_clean_refines_and_remeasures_without_touching_topology(jittered):
    graph, frame, labels, index = jittered
    segments = {sid: dict(graph.segment(sid)) for sid in graph.segment_ids()}
    tips = {n: np.asarray(graph.nodes[n][:3]) for n in graph.endpoints()}
    report = prepare.clean(graph, frame, labels, workers=1)
    assert report["method"] == "centroid-coherent"
    assert report["moved_points"] > 0
    assert "radius_measurement" in report
    assert report["radii_require_remeasurement"] is False
    assert graph.segment_ids() == list(segments)
    for sid, before in segments.items():
        after = graph.segment(sid)
        assert (after["node1"], after["node2"], after["point_ids"]) == \
            (before["node1"], before["node2"], before["point_ids"])
    for n, tip in tips.items():
        assert np.allclose(graph.nodes[n][:3], tip)  # terminals are held fixed
    assert "centreline_displacement_um" in graph.triple.point_attrs

    profiles = prepare.profile_ends(graph, frame, index)
    assert set(profiles) == set(graph.endpoints())
    for profile in profiles.values():
        assert _degrees_off_x(profile.tangent) < 5.0
        assert profile.major_axis is not None
        assert abs(float(profile.major_axis[1])) > 0.95  # the ellipse is wide in y
        assert profile.major_um > profile.minor_um > 0
        assert profile.flatness > 1.1
        assert profile.component == 1
        assert len(profile.tail_points_um) > 3


def test_clean_with_method_none_is_a_no_op():
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 28), (30, 55))
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    report = prepare.clean(graph, frame, ribbon(SHAPE, 6, 2, 5, 55), method="none")
    assert report["skipped"] and report["moved_points"] == 0
    for sid, coords in before.items():
        assert np.array_equal(graph.coords(sid), coords)


def test_clean_rejects_an_unknown_method():
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 28), (30, 55))
    with pytest.raises(ValueError, match="unknown refinement method"):
        prepare.clean(graph, frame, ribbon(SHAPE, 6, 2, 5, 55), method="magic")


def test_profiles_point_outward_and_measure_the_slit():
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 28), (30, 55))
    index = components.build(mask_source(slit(SHAPE, 6, 0, 5, 28) | slit(SHAPE, 6, 0, 30, 55)))
    profiles = prepare.profile_ends(graph, frame, index)
    assert profiles[1].tangent[0] > 0.99  # the left run's right-hand end faces +x
    assert profiles[2].tangent[0] < -0.99  # the right run's left-hand end faces -x
    assert profiles[1].component != profiles[2].component
    assert profiles[1].flatness > 1.5
    assert abs(float(profiles[1].normal[2])) > 0.95  # collapsed along z
    tangents = prepare.as_tangents(profiles)
    assert set(tangents) == set(profiles)
    direction, radius = tangents[1]
    assert np.isclose(np.linalg.norm(direction), 1.0) and radius == pytest.approx(20.0)


def test_a_tip_off_the_mask_is_profiled_without_a_section():
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 28), (30, 55))
    index = components.build(mask_source(ribbon(SHAPE, 6, 2, 5, 10)))  # mask far away
    profile = prepare.profile_end(graph, frame, index, 2)
    assert profile is not None and profile.major_axis is None
    assert "not on the mask" in profile.reason
    assert "section unmeasured" in profile.describe()


def test_summarise_reports_refinement_and_sections():
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 28), (30, 55))
    index = components.build(mask_source(ribbon(SHAPE, 6, 2, 5, 28) | ribbon(SHAPE, 6, 2, 30, 55)))
    profiles = prepare.profile_ends(graph, frame, index)
    text = prepare.summarise(profiles, {"method": "laplacian", "moved_points": 3,
                                        "iterations": 2, "converged": True,
                                        "outside_edges_before": 0, "outside_edges_after": 0})
    assert "refined with laplacian" in text and "4 free end(s) profiled" in text
    assert "skipped" in prepare.summarise(profiles, {"skipped": True})
