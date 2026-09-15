"""Reconnection must join what is broken and refuse what merely looks close.

Every gate here is a heuristic, so the tests are built around the two failure
modes that matter: a break that should be mended and is not, and two vessels that
run near each other and get welded together. The second is the dangerous one -- a
spurious bridge silently reroutes flow in the CFD run downstream -- so most of
these tests are about refusing.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.reconnect import apply_bridges, summarise
from hipct_seg_debug.edit.reconnect import endpoints, tjunction
from hipct_seg_debug.edit.reconnect.candidates import (
    Bridge,
    endpoint_tangent,
    hermite_path,
    resample_by_arclength,
)


class GraphBuilder:
    """Assemble a graph out of straight runs, in micrometres."""

    def __init__(self):
        self.nodes: dict = {}
        self.points: dict = {}
        self.segments: list = []
        self._n = 0
        self._p = 0
        self._s = 0

    def node(self, xyz):
        self.nodes[self._n] = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 0)
        self._n += 1
        return self._n - 1

    def run(self, start, end, radius, n=12):
        a, b = self.node(start), self.node(end)
        ids = []
        for t in np.linspace(0.0, 1.0, n):
            xyz = np.asarray(start) * (1 - t) + np.asarray(end) * t
            self.points[self._p] = (*(float(v) for v in xyz), float(radius))
            ids.append(self._p)
            self._p += 1
        self.segments.append(
            {"id": self._s, "node1": a, "node2": b, "point_ids": ids, "strahler": 1}
        )
        self._s += 1
        return a, b, self._s - 1

    def build(self) -> EditableGraph:
        g = EditableGraph(
            Triple(nodes=self.nodes, points=self.points, segments=self.segments)
        )
        g._flush_degrees()
        return g


def broken_vessel(gap_um=300.0, radius=50.0):
    """One straight vessel cut in two, collinear, with a gap along z."""
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), radius)
    b.run((0, 0, 1000 + gap_um), (0, 0, 2000 + gap_um), radius)
    return b.build()


# ------------------------------------------------------------------ geometry

def test_endpoint_tangent_points_outward():
    g = broken_vessel()
    # Node 1 is the far end of the first run, so its outward direction is +z.
    direction, radius = endpoint_tangent(g, 1)
    assert np.allclose(direction, [0, 0, 1], atol=1e-6)
    assert radius == pytest.approx(50.0)
    # Node 0 is the near end, pointing the other way.
    direction, _ = endpoint_tangent(g, 0)
    assert np.allclose(direction, [0, 0, -1], atol=1e-6)


def test_endpoint_tangent_is_none_at_a_junction():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 100), 10.0)
    g = b.build()
    g.split_segment(0, 5)
    interior = next(n for n in g.nodes if g.degree(n) == 2)
    assert endpoint_tangent(g, interior) is None


def test_hermite_path_leaves_and_arrives_along_its_tangents():
    p0, p1 = np.array([0.0, 0, 0]), np.array([0.0, 0, 100])
    t0 = np.array([1.0, 0, 0])   # leaves along +x
    t1 = np.array([0.0, 0, 1])   # arrives along +z
    path = hermite_path(p0, t0, p1, t1, 40)
    assert np.allclose(path[0], p0) and np.allclose(path[-1], p1)
    out = path[1] - path[0]
    assert out[0] > abs(out[2]), "did not leave along its start tangent"
    incoming = path[-1] - path[-2]
    assert incoming[2] > abs(incoming[0]), "did not arrive along its end tangent"
    # A chord would be straight; this must bow off it.
    chord = np.linalg.norm(p1 - p0)
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    assert length > chord * 1.05


def test_resample_keeps_endpoints_and_spacing():
    coords = np.column_stack([np.zeros(5), np.zeros(5), np.linspace(0, 400, 5)])
    out = resample_by_arclength(coords, 50.0)
    assert np.allclose(out[0], coords[0]) and np.allclose(out[-1], coords[-1])
    step = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.all(step <= 50.0 + 1e-6)


# ------------------------------------------------------- endpoint reconnection

def test_a_collinear_break_is_reconnected():
    g = broken_vessel(gap_um=300.0)
    assert len(g.components()) == 2

    proposals = endpoints.propose(g)
    assert len(proposals) == 1
    bridge = proposals[0]
    assert bridge.accepted, bridge.reason
    assert bridge.kind == "endpoint"
    assert {bridge.source_node, bridge.target_node} == {1, 2}

    created = apply_bridges(g, proposals)
    assert len(created) == 1
    assert len(g.components()) == 1, "the two halves should now be one vessel"


def test_a_gap_beyond_the_reach_is_refused():
    # Reach is 15 x radius = 750 um for r = 50, and it bounds the KD ball query
    # itself -- so a pair this far apart is never even proposed, rather than
    # being proposed and then rejected.
    g = broken_vessel(gap_um=2000.0)
    assert endpoints.propose(g, keep_rejected=True) == []
    assert len(g.components()) == 2


def test_reach_scales_with_radius():
    """The same gap is bridgeable for a trunk and out of reach for a twig."""
    thick = endpoints.propose(broken_vessel(gap_um=600.0, radius=60.0))
    thin = endpoints.propose(broken_vessel(gap_um=600.0, radius=20.0))
    assert [b for b in thick if b.accepted], "a 60 um vessel should reach 600 um"
    assert not [b for b in thin if b.accepted], "a 20 um vessel should not"


def test_two_parallel_vessels_are_not_welded_across():
    """The dangerous false positive: close, but facing the same way."""
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), 50.0)
    b.run((200, 0, 0), (200, 0, 1000), 50.0)
    g = b.build()

    proposals = endpoints.propose(g, keep_rejected=True)
    accepted = [x for x in proposals if x.accepted]
    assert not accepted, f"welded two parallel vessels: {accepted}"
    assert any("cone" in x.reason or "faces away" in x.reason for x in proposals)


def test_a_thick_trunk_is_not_joined_to_a_capillary():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), 400.0)
    b.run((0, 0, 1300), (0, 0, 2300), 20.0)
    g = b.build()
    proposals = endpoints.propose(g, keep_rejected=True)
    assert not [x for x in proposals if x.accepted]
    assert any("radius ratio" in x.reason for x in proposals)


def test_an_end_pointing_sideways_is_refused():
    """Right in front of the source, but running across it rather than onward.

    The source's own cone is satisfied -- the target sits straight ahead -- so
    this is caught only by the second test, that the *target* faces back. It is a
    T-junction, not a continuation, and belongs to the other proposer.
    """
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), 50.0)          # tip at z=1000, pointing +z
    b.run((0, 0, 1300), (600, 0, 1300), 50.0)     # crosses in +x just above it
    g = b.build()

    proposals = endpoints.propose(g, keep_rejected=True)
    joined = [x for x in proposals if x.accepted and {x.source_node, x.target_node} == {1, 2}]
    assert not joined, "joined an end that runs across rather than onward"
    assert any("faces away" in x.reason for x in proposals)


def test_each_endpoint_is_used_at_most_once():
    """Two candidates competing for one free end: only the better one wins."""
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), 50.0)
    b.run((0, 0, 1300), (0, 0, 2300), 50.0)     # straight ahead
    b.run((250, 0, 1350), (900, 0, 2300), 50.0)  # off to the side
    g = b.build()

    proposals = [x for x in endpoints.propose(g) if x.accepted]
    used = [n for x in proposals for n in (x.source_node, x.target_node)]
    assert len(used) == len(set(used)), "an endpoint was consumed twice"


def test_joins_within_one_component_are_refused_by_default():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), 50.0)
    g = b.build()
    # Both free ends belong to the same component, so joining them makes a loop.
    assert not [x for x in endpoints.propose(g) if x.accepted]


# ------------------------------------------------------------- T-junctions

def test_a_side_branch_attaches_to_the_middle_of_a_trunk():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 2000), 200.0)          # trunk along z
    b.run((300, 0, 1000), (900, 0, 1000), 60.0)    # branch pointing away in +x
    g = b.build()
    n_segments = len(g.segments)

    proposals = tjunction.propose(g)
    accepted = [x for x in proposals if x.accepted]
    assert accepted, f"no T-junction proposed: {[x.reason for x in proposals]}"
    bridge = accepted[0]
    assert bridge.kind == "tjunction"
    assert bridge.target_segment == 0
    assert "target_point_id" in bridge.metrics

    apply_bridges(g, accepted)
    # The trunk was split in two and a bridge added: three segments where there
    # was one, plus the untouched branch.
    assert len(g.segments) == n_segments + 2
    assert len(g.components()) == 1
    assert any(g.degree(n) == 3 for n in g.nodes), "no bifurcation was created"


def test_a_branch_thicker_than_its_trunk_is_refused():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 2000), 40.0)
    b.run((200, 0, 1000), (900, 0, 1000), 300.0)
    g = b.build()
    accepted = [x for x in tjunction.propose(g, keep_rejected=True) if x.accepted]
    assert not accepted, "attached a trunk to a capillary as if it were a daughter"


def test_a_vessel_is_never_attached_to_itself():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 2000), 100.0)
    g = b.build()
    for bridge in tjunction.propose(g, same_component=True, keep_rejected=True):
        assert bridge.target_segment not in g.node_segments(bridge.source_node)


# ------------------------------------------------------------------ applying

def test_apply_ignores_rejected_bridges():
    g = broken_vessel(gap_um=300.0)
    proposals = endpoints.propose(g)
    for bridge in proposals:
        bridge.reject("test")
    assert apply_bridges(g, proposals) == []
    assert len(g.components()) == 2


def test_apply_is_a_single_undo_step():
    g = broken_vessel(gap_um=300.0)
    apply_bridges(g, endpoints.propose(g))
    assert len(g.history.labels()) == 1
    assert len(g.components()) == 1
    g.undo()
    assert len(g.components()) == 2, "undo did not remove the bridge"


def test_new_segments_are_marked_as_reconnected():
    g = broken_vessel(gap_um=300.0)
    created = apply_bridges(g, endpoints.propose(g))
    assert created
    assert g.segment(created[0])["reconnected"] == 1.0


def test_two_tjunctions_on_one_trunk_both_apply():
    """The stale-segment-id case: the first split invalidates the second's target."""
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 3000), 200.0)
    b.run((300, 0, 800), (900, 0, 800), 60.0)
    b.run((300, 0, 2200), (900, 0, 2200), 60.0)
    g = b.build()

    accepted = [x for x in tjunction.propose(g) if x.accepted]
    assert len(accepted) == 2, "both branches should have proposed a junction"
    created = apply_bridges(g, accepted)
    assert len(created) == 2, "the second bridge was lost to a stale segment id"
    assert len(g.components()) == 1


def test_summarise_reports_accepted_and_reasons():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), 50.0)
    b.run((200, 0, 0), (200, 0, 1000), 50.0)  # parallel: in reach, but refused
    g = b.build()
    text = summarise(endpoints.propose(g, keep_rejected=True))
    assert "candidates" in text
    assert "rejected" in text
    assert summarise([]) == "no reconnection candidates"


# -------------------------------------------------- explaining an empty result
#
# The bug these pin: on a real graph both proposers printed the bare "no
# reconnection candidates" while 161 free ends sat there, because the two ways a
# pair dies before a `Bridge` exists -- nothing within reach, and the
# same-component prune -- were counted nowhere and so could not be reported.


def forked_vessel(radius=100.0):
    """A trunk that forks, so the two tips are near each other in ONE component."""
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), radius)
    b.run((0, 0, 1000), (400, 0, 1600), radius)
    b.run((0, 0, 1000), (-400, 0, 1600), radius)
    g = b.build()
    # `run` mints a fresh node per call, so the three coincident ends at the fork
    # are three separate nodes -- and three components -- until they are welded.
    g.weld_coincident_nodes()
    return g


def test_same_component_prune_is_counted_not_silent():
    """The exact shape that printed a bare "no candidates" on the real graph."""
    g = forked_vessel()
    assert len(g.components()) == 1, "the fork should be one component"

    stats: dict = {}
    proposals = endpoints.propose(g, keep_rejected=True, stats=stats)

    assert stats["ends_considered"] == len(g.endpoints())
    assert stats["pairs_in_reach"] > 0, "the two tips are 800 um apart, well in reach"
    assert stats["pruned_same_component"] == stats["pairs_in_reach"], (
        "every pair here is inside one component, so every one should be pruned"
    )
    assert not proposals, "a pruned pair never becomes a candidate, by design"

    # The point of the whole change: that must not print as "nothing found".
    text = summarise(proposals, stats)
    assert "--same-component" in text
    assert str(stats["pruned_same_component"]) in text

    # ...and with the policy relaxed, the same pairs do become candidates.
    relaxed: dict = {}
    assert endpoints.propose(g, same_component=True, keep_rejected=True, stats=relaxed)
    assert relaxed["pruned_same_component"] == 0


def test_summarise_names_the_stage_that_consumed_everything():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 1000), 50.0)
    b.run((0, 0, 40_000), (0, 0, 41_000), 50.0)  # far out of any reach
    g = b.build()

    stats: dict = {}
    proposals = endpoints.propose(g, keep_rejected=True, stats=stats)
    assert not proposals
    assert stats["pairs_in_reach"] == 0
    text = summarise(proposals, stats)
    assert "4 free end(s) examined" in text
    assert "nothing within reach" in text
    # The number that says the reach, not the cone, was the binding constraint.
    assert stats["nearest_cross_component_um"] == pytest.approx(39_000.0)
    assert "39.00 mm apart" in text


def test_tjunction_counts_its_own_two_prunes():
    b = GraphBuilder()
    b.run((0, 0, 0), (0, 0, 3000), 200.0)
    b.run((300, 0, 800), (900, 0, 800), 60.0)
    g = b.build()

    stats: dict = {}
    tjunction.propose(g, keep_rejected=True, stats=stats)
    assert stats["kind"] == "tjunction"
    # A free end always sees its own vessel's interior points before anything else.
    assert stats["pruned_own_segment"] > 0


def test_summarise_without_stats_is_unchanged():
    """Every existing caller passes no stats and must get the old string."""
    assert summarise([]) == "no reconnection candidates"


def test_nearest_cross_component_gives_up_rather_than_crawl():
    from hipct_seg_debug.edit.reconnect.candidates import nearest_cross_component

    points = np.random.default_rng(0).normal(size=(1000, 3)) * 100.0
    # One label per point: n_labels * n = 1e6, past the guard.
    assert nearest_cross_component(points, np.arange(len(points))) is None
    # A sane number of labels is answered exactly.
    two = np.array([[0.0, 0, 0], [0.0, 0, 7.0], [0.0, 0, 100.0]])
    assert nearest_cross_component(two, [0, 1, 1]) == pytest.approx(7.0)
    assert nearest_cross_component(two, [0, 0, 0]) is None


def test_gaps_still_writes_when_it_finds_nothing(tmp_path):
    """A clean pass must not look like a missing file to the next step in a chain.

    `cmd_gaps` used to return before `_save`, so workflow 8 declared `step1.am` as
    step 2's output and then aborted with "gaps returned 0 but did not write it" --
    on a graph where there was simply nothing to fill.
    """
    pytest.importorskip("coronary_sdf")
    from hipct_seg_debug.edit.__main__ import main
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    source, out = tmp_path / "in.am", tmp_path / "out.am"
    write_spatial_graph(broken_vessel().to_spatial_graph(), source)

    assert main(["gaps", str(source), "--out", str(out)]) == 0
    assert out.exists(), "a graph with no gaps still has to be handed forward"


def test_gaps_names_what_it_did_not_look_at(capsys):
    """The reported bug: "0 gaps" read as "nothing to reconnect"."""
    pytest.importorskip("coronary_sdf")
    from hipct_seg_debug.edit.__main__ import _nothing_to_fill

    text = _nothing_to_fill(broken_vessel())
    assert "2 component(s)" in text and "4 free end(s)" in text
    assert "connect" in text


def test_bridge_geometry_helpers():
    coords = np.column_stack([np.zeros(3), np.zeros(3), [0.0, 50.0, 100.0]])
    bridge = Bridge("endpoint", 0, coords, np.full(3, 10.0), target_node=1)
    assert bridge.span_um == pytest.approx(100.0)
    assert bridge.length_um == pytest.approx(100.0)
    assert bridge.tortuosity == pytest.approx(1.0)
    assert "endpoint" in repr(bridge)
