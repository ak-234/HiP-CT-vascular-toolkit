"""The multi-algorithm front end.

The paper's finding is that algorithm choice alone can double a network's node count,
so the value of this module is that every backend hands back the *same* structure and
can therefore be scored on the same terms. These tests pin that contract; the TEASAR
backend needs ``kimimaro``, which is optional, so it is skipped when absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import skeletonisers as sk
from hipct_seg_debug.edit import supermetric as sm

from .conftest_geometry import cylinder, make_frame

SHAPE = (30, 30, 60)


@pytest.fixture
def frame():
    return make_frame(SHAPE)


@pytest.fixture
def tube():
    return cylinder(SHAPE, 4, 5, 55)


# --------------------------------------------------------------- edges_to_triple


def test_a_y_becomes_three_segments_and_one_junction():
    """Nodes are the vertices of degree != 2; the chains between them are segments."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
         [4, 1, 0], [5, 2, 0], [4, -1, 0], [5, -2, 0]], dtype=float
    ) * 100.0
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [3, 6], [6, 7]])

    triple = sk.edges_to_triple(verts, edges, np.full(len(verts), 50.0))

    assert len(triple.segments) == 3
    assert sorted(n[3] for n in triple.nodes.values()) == [1, 1, 1, 3]
    assert sum(len(s["point_ids"]) for s in triple.segments) == 10


def test_a_ring_with_no_junction_survives_as_a_self_loop():
    """Dropping it would lose a whole cycle silently."""
    angles = np.linspace(0.0, 2 * np.pi, 9)[:-1]
    verts = np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=1) * 100.0
    edges = np.array([[i, (i + 1) % 8] for i in range(8)])

    triple = sk.edges_to_triple(verts, edges, np.full(8, 20.0))

    assert len(triple.segments) == 1
    seg = triple.segments[0]
    assert seg["node1"] == seg["node2"], "closed on itself"


def test_the_origin_offset_is_applied():
    verts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    triple = sk.edges_to_triple(verts, edges, np.full(3, 10.0), origin_um=(1000, 20, 3))
    assert triple.nodes[0][:3] == (1000.0, 20.0, 3.0)


# ------------------------------------------------------------------- backends


def test_lee_produces_a_scoreable_skeleton(frame, tube):
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    cand = sk.skeletonise("lee", tube, frame)

    assert cand.name == "lee"
    assert len(cand.triple.segments) >= 1
    graph = EditableGraph(cand.triple)
    assert sm.cl_sensitivity(graph, frame, tube) > 0.9, "a thinned axis is inside"


def test_the_frame_describes_the_volume_with_no_stride_applied_twice(frame, tube):
    """A decimated volume's stride belongs in its frame, and nowhere else.

    Applying it here as well put every coordinate eight times too far out at stride 8:
    the skeleton derived *from* the mask scored a cl-sensitivity of exactly zero
    against it, and a network volume 600 times too large. Both are the shape of error
    that looks like a bad algorithm rather than a bad conversion.
    """
    import inspect

    from hipct_seg_debug.edit.graphmodel import EditableGraph

    assert "stride" not in inspect.signature(sk.skeletonise).parameters

    cand = sk.skeletonise("lee", tube, frame)
    coords = np.vstack([
        EditableGraph(cand.triple).coords(sid)
        for sid in EditableGraph(cand.triple).segment_ids()
    ])
    lo, hi = frame.seg_bbox_um[0::2], frame.seg_bbox_um[1::2]
    assert np.all(coords >= lo - 1.0) and np.all(coords <= hi + 1.0), \
        "every centreline point lies inside the lattice it came from"


def test_an_unknown_algorithm_is_refused(frame, tube):
    with pytest.raises(ValueError, match="unknown skeletoniser"):
        sk.skeletonise("autoskeleton", tube, frame)


def test_amira_without_a_path_says_what_is_missing(frame, tube):
    with pytest.raises(ValueError, match="--amira-graph"):
        sk.skeletonise("amira", tube, frame)


def test_teasar_traces_a_tube_or_says_why_it_cannot(frame, tube):
    """`kimimaro` is installed as of 2026-08-17, so this now exercises the real backend.

    The ImportError branch is kept rather than deleted: it is the path a fresh checkout
    takes, and it is the only thing asserting the message tells you what to install.
    """
    try:
        import kimimaro  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install kimimaro"):
            sk.skeletonise("teasar", tube, frame)
        return
    cand = sk.skeletonise("teasar", tube, frame)
    assert len(cand.triple.segments) >= 1
