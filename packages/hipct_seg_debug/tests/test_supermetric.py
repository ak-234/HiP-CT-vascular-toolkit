"""The five super-metric terms, against geometry whose answers are known.

Eq. 10 of Walsh & Berg (2024); see ``SKELETONISATION.md``. Each test pins one term to
a closed-form value, because the point of the metric is to be trusted as an objective:
a term that is merely self-consistent would rank algorithms confidently and wrongly.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import supermetric as sm

from .conftest_geometry import SPACING, axis_graph, cylinder, make_frame

pytest.importorskip("skimage")

SHAPE = (40, 40, 80)  # (nz, ny, nx)
RADIUS_VOX = 5
X0, X1 = 5, 75


@pytest.fixture
def frame():
    return make_frame(SHAPE)


@pytest.fixture
def tube():
    return cylinder(SHAPE, RADIUS_VOX, X0, X1)


@pytest.fixture
def centreline(frame):
    return axis_graph(frame, X0 + 1, X1 - 1, RADIUS_VOX * SPACING,
                      cy=SHAPE[1] // 2, cz=SHAPE[0] // 2)


# ------------------------------------------------------------------- the terms


def test_local_euler_is_positive_for_a_tree():
    """`chi = 2 - chi_classical`; the supplementary's `-chi - 2` would give -3."""
    assert sm.local_euler(1) == 1.0
    assert sm.local_euler(0) == 2.0, "the special case the reformulation exists for"
    assert sm.local_euler(-381) == 383.0


def test_graph_volume_matches_pi_r2_l(centreline):
    """V from the graph is the sum of subsegment cylinders (Table S13)."""
    length = float(np.ptp(centreline.coords(0)[:, 0]))
    expected = np.pi * (RADIUS_VOX * SPACING) ** 2 * length
    assert sm.graph_volume(centreline) == pytest.approx(expected, rel=1e-9)


def test_graph_components_and_euler_of_a_tree(centreline):
    assert sm.graph_components(centreline) == 1
    assert sm.graph_euler_classical(centreline) == 1, "2 nodes - 1 segment"


def test_a_loop_lowers_the_euler_number_by_one(centreline):
    before = sm.graph_euler_classical(centreline)
    mid = len(centreline.coords(0)) // 2
    centreline.split_segment(centreline.segment_ids()[0], mid)
    assert sm.graph_euler_classical(centreline) == before, "a split adds a node and an edge"

    ends = [n for n in centreline.nodes if centreline.degree(n) == 1]
    bow = np.linspace(centreline.nodes[ends[0]][:3], centreline.nodes[ends[1]][:3], 12)
    bow[1:-1, 1] += 40.0
    centreline.add_segment(ends[0], ends[1], bow, np.full(len(bow), 50.0))
    assert sm.graph_euler_classical(centreline) == before - 1
    assert sm.graph_components(centreline) == 1


def test_image_terms_of_one_tube(frame, tube):
    terms = sm.image_terms(tube, frame.seg_spacing)
    assert terms.components == 1
    assert terms.euler_classical == 1, "a solid rod has no tunnels"
    assert terms.voxel_count == int(tube.sum())
    # Voxelisation makes the digitised disc a little larger than pi r^2.
    analytic = np.pi * (RADIUS_VOX * SPACING) ** 2 * (X1 - X0) * SPACING
    assert terms.volume_um3 == pytest.approx(analytic, rel=0.15)


def test_tree_chi_reference_ignores_the_image(frame, tube):
    """`tree_chi` scores against the tree ideal, per the de-looping prior."""
    honest = sm.image_terms(tube, frame.seg_spacing, tree_chi=False)
    tree = sm.image_terms(tube, frame.seg_spacing, tree_chi=True)
    assert honest.chi == sm.local_euler(honest.euler_classical)
    assert tree.chi == sm.local_euler(1) == 1.0


# --------------------------------------------------------------- cl-sensitivity


def test_bresenham_fills_the_gap_between_sparse_points(frame, tube):
    """The divergence from `skeleton_analysis`: rasterise the lines, not the points.

    A graph carrying one point every ten voxels still covers every voxel between
    them. Sampling the mask at its points alone would inspect a tenth as much.
    """
    dense = axis_graph(frame, X0 + 1, X1 - 1, 50.0, cy=20, cz=20, step=1)
    sparse = axis_graph(frame, X0 + 1, X1 - 1, 50.0, cy=20, cz=20, step=10)

    assert len(sparse.points) < len(dense.points) / 5
    raster = sm.rasterise_centreline(sparse, frame)
    assert len(raster) > 5 * len(sparse.points), "the lines between points are filled"
    # Consecutive voxels along x, i.e. an unbroken 26-connected run.
    xs = np.sort(raster[:, 2])
    assert np.all(np.diff(xs) <= 1)


def test_cl_sensitivity_is_one_inside_and_zero_outside(frame, tube, centreline):
    assert sm.cl_sensitivity(centreline, frame, tube) == pytest.approx(1.0)

    with centreline.batch("push off axis"):
        for sid in centreline.segment_ids():
            coords = centreline.coords(sid)
            coords[:, 1] += 15 * SPACING  # well clear of a 5-voxel tube
            centreline.set_segment_coords(sid, coords)
    assert sm.cl_sensitivity(centreline, frame, tube) == pytest.approx(0.0)


# ------------------------------------------------------------------ the metric


def test_a_faithful_skeleton_scores_near_zero(frame, tube, centreline):
    image = sm.image_terms(tube, frame.seg_spacing)
    refs = sm.reference_bifurcations(tube, frame)
    metric = sm.super_metric(centreline, frame, tube, image, refs)

    assert metric.cl_sensitivity == pytest.approx(1.0)
    assert metric.cl == pytest.approx(0.0)
    assert metric.components == pytest.approx(0.0)
    assert metric.euler == pytest.approx(0.0)
    assert metric.volume < 0.2, "voxelisation only; no structural disagreement"


def test_an_unbranched_tube_drops_the_bifurcation_term(frame, tube, centreline):
    """No bifurcation anywhere is "not applicable", not "infinitely wrong"."""
    image = sm.image_terms(tube, frame.seg_spacing)
    refs = sm.reference_bifurcations(tube, frame)
    metric = sm.super_metric(centreline, frame, tube, image, refs)

    assert metric.dice_detail["n_candidate"] == 0
    assert np.isnan(metric.bifurcation)
    assert np.isfinite(metric.total), "a NaN term is excluded, not propagated"


def test_the_cl_weight_rejects_a_skeleton_leaving_the_mask(frame, tube, centreline):
    """`w_cl = 1/cl**3` is the paper's design: a drop in cl must dominate M_S."""
    image = sm.image_terms(tube, frame.seg_spacing)
    refs = sm.reference_bifurcations(tube, frame)
    good = sm.super_metric(centreline, frame, tube, image, refs)

    # Ramped, not uniform: a constant offset either stays wholly inside the tube or
    # leaves it entirely, and the interesting case is the partial one.
    with centreline.batch("tilt out of the lumen"):
        for sid in centreline.segment_ids():
            coords = centreline.coords(sid)
            ramp = np.linspace(0.0, 12.0 * SPACING, len(coords))
            coords[:, 1] += ramp
            centreline.set_segment_coords(sid, coords)
    worse = sm.super_metric(centreline, frame, tube, image, refs)

    assert 0.0 < worse.cl_sensitivity < 1.0, "partly in, partly out"
    assert worse.cl > good.cl
    assert worse.total > good.total
    # The whole point of the cubic weight: cl dominates every other term.
    assert worse.cl > worse.volume + worse.components + worse.euler


# --------------------------------------------------------------- per tree


def test_restricting_the_graph_euler_to_one_component_matches_the_default():
    """On a single-component graph the nominated and the automatic answers agree."""
    from .conftest_geometry import graph_from

    graph = graph_from(
        [(0, 0, 0), (400, 0, 0), (800, 300, 0), (800, -300, 0)],
        [(0, 1, 5, 90.0), (1, 2, 5, 50.0), (1, 3, 5, 45.0)],
    )
    comps = graph.components()

    assert sm.graph_euler_classical(graph, segs=comps[0]) == sm.graph_euler_classical(graph)


def test_the_default_graph_euler_ignores_the_second_tree():
    """The blind spot this whole per-tree path exists to close.

    A loop added to the *smaller* component does not move the default term at all,
    because it is measured on the largest component alone.
    """
    from .conftest_geometry import graph_from

    graph = graph_from(
        # A big tree, then a small one carrying a loop between nodes 4 and 5.
        [(0, 0, 0), (400, 0, 0), (800, 300, 0), (800, -300, 0),
         (0, 2000, 0), (400, 2000, 0)],
        [(0, 1, 5, 90.0), (1, 2, 5, 50.0), (1, 3, 5, 45.0),
         (4, 5, 5, 60.0), (4, 5, 5, 55.0)],
    )
    comps = graph.components()
    small = next(c for c in comps if len(c) == 2)

    assert sm.graph_euler_classical(graph) == 1, "the big tree is a tree"
    assert sm.graph_euler_classical(graph, segs=small) == 0, "the small one has a loop"


def test_image_terms_on_a_nominated_component_matches_scoring_it_alone(frame):
    from hipct_seg_debug.edit.reconnect.segmentation import components as label_components

    from .conftest_geometry import two_cylinders

    volume = two_cylinders(SHAPE)
    stats = label_components(volume > 0)
    label = int(stats.order_by_size()[0])

    nominated = sm.image_terms(volume, frame.seg_spacing, labels=stats.labels,
                               component=label)
    alone = sm.image_terms(stats.labels == label, frame.seg_spacing)

    assert nominated.voxel_count == alone.voxel_count
    assert nominated.euler_classical == alone.euler_classical
    assert nominated.components == 1


def test_each_tree_is_scored_against_its_own_component(frame):
    """Both trees get a score -- where the whole-graph chi sees only the larger."""
    from hipct_seg_debug.edit import components as comp
    from hipct_seg_debug.edit import skeletonisers as sk
    from hipct_seg_debug.edit.graphmodel import EditableGraph
    from hipct_seg_debug.edit.reconnect.segmentation import components as label_components

    from .conftest_geometry import two_cylinders

    volume = two_cylinders(SHAPE)
    cand = sk.skeletonise_per_component("lee", volume, frame)
    graph = EditableGraph(cand.triple)
    parts, _ = comp.split_components(volume, frame)
    stats = label_components(volume > 0)

    per_tree = sm.super_metric_per_tree(graph, frame, stats.labels, parts)

    assert sorted(per_tree) == [0, 1], "both trees scored, not just the largest"
    for index, metric in per_tree.items():
        assert metric.graph_components == 1, "each tree is one piece"
        assert metric.graph_euler_classical == 1, "and a tree, so chi_classical is 1"
        assert metric.cl_sensitivity > 0.9
    # The aggregate weights by voxel count, so it sits between the two.
    lo, hi = sorted(m.total for m in per_tree.values())
    assert lo <= sm.aggregate(per_tree, parts) <= hi


def test_the_aggregate_weights_the_big_tree_more_than_an_unweighted_mean():
    """A small fragment must not outvote the left main."""
    class _Part:
        def __init__(self, index, voxels):
            self.index = index
            self.voxels = voxels

    per_tree = {0: sm.SuperMetric(volume=1.0), 1: sm.SuperMetric(volume=9.0)}
    parts = [_Part(0, 9000), _Part(1, 1000)]

    weighted = sm.aggregate(per_tree, parts, objective="weighted")
    mean = sm.aggregate(per_tree, parts, objective="mean")

    assert weighted == pytest.approx(1.8)
    assert mean == pytest.approx(5.0)
    assert weighted < mean, "the big tree dominates, as it should"


def test_the_aggregate_is_nan_when_no_tree_scored():
    assert np.isnan(sm.aggregate({}, []))
