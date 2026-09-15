"""Focused invariance and correctness tests for the adaptive implicit backend."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.spatial import KDTree

from coronary_sdf.adaptive_octree import RadiusAdaptiveOctree
from coronary_sdf.adaptive_mesher import mesh_adaptive_implicit
from coronary_sdf.capsules import CapsuleArrays
from coronary_sdf.implicit_field import (
    GraphImplicitField,
    build_graph_implicit_field,
    tapered_capsule_values,
)
from coronary_sdf.geometry_constraints import find_capsule_conflicts
from coronary_sdf.sdf_field import adaptive_smin_k, compute_grid


def _capsules(
    starts: np.ndarray,
    ends: np.ndarray,
    radii_start: np.ndarray,
    radii_end: np.ndarray,
    segment_ids: np.ndarray,
    cap_bif_at_start: np.ndarray | None = None,
    cap_bif_at_end: np.ndarray | None = None,
) -> CapsuleArrays:
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    radii_start = np.asarray(radii_start, dtype=np.float64)
    radii_end = np.asarray(radii_end, dtype=np.float64)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    midpoints = 0.5 * (starts + ends)
    raw_tangent = ends - starts
    lengths = np.linalg.norm(raw_tangent, axis=1)
    tangents = raw_tangent / np.maximum(lengths[:, None], 1e-30)
    n_segments = int(segment_ids.max()) + 1
    arc_start = np.zeros(len(starts), dtype=np.float64)
    arc_end = np.zeros(len(starts), dtype=np.float64)
    segment_lengths = np.zeros(n_segments, dtype=np.float64)
    for segment_id in range(n_segments):
        cumulative = 0.0
        for primitive in np.flatnonzero(segment_ids == segment_id):
            arc_start[primitive] = cumulative
            cumulative += lengths[primitive]
            arc_end[primitive] = cumulative
        segment_lengths[segment_id] = cumulative
    if cap_bif_at_start is None:
        cap_bif_at_start = np.zeros(len(starts), dtype=bool)
    if cap_bif_at_end is None:
        cap_bif_at_end = np.zeros(len(starts), dtype=bool)
    return CapsuleArrays(
        starts=starts,
        ends=ends,
        radii_start=radii_start,
        radii_end=radii_end,
        seg_idx=segment_ids,
        midpoints=midpoints,
        tangents=tangents,
        max_radii=np.maximum(radii_start, radii_end),
        tree=KDTree(midpoints),
        arc_start=arc_start,
        arc_end=arc_end,
        seg_L=segment_lengths,
        cap_bif_at_start=np.asarray(cap_bif_at_start, dtype=bool),
        cap_bif_at_end=np.asarray(cap_bif_at_end, dtype=bool),
    )


def _star(degree: int = 5, scale: float = 1.0) -> tuple[CapsuleArrays, dict, dict]:
    directions = []
    for i in range(degree):
        angle = 2.0 * math.pi * i / degree
        directions.append((math.cos(angle), math.sin(angle), 0.35 if i % 2 else -0.2))
    directions = np.asarray(directions, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    starts = np.zeros((degree, 3), dtype=np.float64)
    ends = 5.0 * scale * directions
    radius = np.full(degree, 0.5 * scale)
    capsules = _capsules(starts, ends, radius, radius, np.arange(degree))
    nodes = {7: (0.0, 0.0, 0.0, degree)}
    node_to_segments = {7: set(range(degree))}
    return capsules, nodes, node_to_segments


class ImplicitFieldTests(unittest.TestCase):
    def test_legacy_grid_coordinates_match_declared_spacing(self) -> None:
        capsules = _capsules(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([0.25]),
            np.asarray([0.25]),
            np.asarray([0]),
        )
        grid = compute_grid(capsules)
        for coordinates in (grid.x, grid.y, grid.z):
            np.testing.assert_allclose(np.diff(coordinates), grid.voxel_size, atol=1e-14)
        np.testing.assert_allclose(
            grid.bbox_max,
            grid.bbox_min + (grid.dims - 1) * grid.voxel_size,
            atol=1e-14,
        )

    def test_legacy_adaptive_k_has_inverse_length_scaling(self) -> None:
        k = adaptive_smin_k(0.5)
        scaled_k = adaptive_smin_k(1.0)
        self.assertAlmostEqual(scaled_k, k / 2.0)

    def test_bvh_matches_exhaustive_minimum(self) -> None:
        rng = np.random.default_rng(11)
        starts = rng.normal(size=(80, 3))
        ends = starts + rng.normal(scale=0.4, size=(80, 3))
        r0 = rng.uniform(0.03, 0.3, size=80)
        r1 = rng.uniform(0.03, 0.3, size=80)
        capsules = _capsules(starts, ends, r0, r1, np.arange(80) // 4)
        field = GraphImplicitField(capsules, bvh_leaf_size=5)

        for point in rng.normal(scale=2.0, size=(100, 3)):
            values, _radii, _t = tapered_capsule_values(point, starts, ends, r0, r1)
            self.assertAlmostEqual(field.sample(point).value, float(values.min()), places=12)

    def test_multifurcation_blend_has_degree_independent_bound(self) -> None:
        capsules, nodes, incidence = _star(degree=5)
        beta = 0.18
        field = build_graph_implicit_field(
            capsules,
            nodes,
            incidence,
            blend_fraction=beta,
            support_factor=4.0,
        )
        hard = GraphImplicitField(capsules).sample(np.zeros(3)).value
        blended = field.sample(np.zeros(3)).value
        self.assertAlmostEqual(hard - blended, beta * 0.5, places=12)

    def test_asymmetric_bifurcation_clips_large_parent_cap(self) -> None:
        directions = np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, -0.5, 0.0]],
            dtype=np.float64,
        )
        directions[1:] /= np.linalg.norm(directions[1:], axis=1)[:, None]
        starts = np.asarray([[-4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ends = np.vstack((np.zeros(3), 4.0 * directions[1], 4.0 * directions[2]))
        r0 = np.asarray([1.2, 0.4, 0.4])
        r1 = r0.copy()
        capsules = _capsules(
            starts,
            ends,
            r0,
            r1,
            np.arange(3),
            cap_bif_at_start=np.asarray([False, True, True]),
            cap_bif_at_end=np.asarray([True, False, False]),
        )
        nodes = {9: (0.0, 0.0, 0.0, 3)}
        incidence = {9: {0, 1, 2}}
        clipped = build_graph_implicit_field(
            capsules, nodes, incidence, primitive_method="round_cone"
        )
        rounded = build_graph_implicit_field(
            capsules,
            nodes,
            incidence,
            primitive_method="round_cone",
            clip_bifurcation_caps=False,
        )

        # This point lies between the diverging daughters. The large parent's
        # terminal sphere incorrectly fills it unless the parent is one-sided.
        carina_probe = np.asarray([1.0, 0.0, 0.0])
        self.assertLess(rounded.sample(carina_probe).value, 0.0)
        self.assertGreater(clipped.sample(carina_probe).value, 0.0)

    def test_clipped_junction_is_globally_scale_equivariant(self) -> None:
        normalized_values = []
        for scale in (0.01, 1.0, 100.0):
            starts = scale * np.asarray(
                [[-4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
            ends = scale * np.asarray(
                [[0.0, 0.0, 0.0], [4.0, 2.0, 0.0], [4.0, -2.0, 0.0]]
            )
            radii = scale * np.asarray([1.2, 0.4, 0.4])
            capsules = _capsules(
                starts,
                ends,
                radii,
                radii,
                np.arange(3),
                cap_bif_at_start=np.asarray([False, True, True]),
                cap_bif_at_end=np.asarray([True, False, False]),
            )
            field = build_graph_implicit_field(
                capsules,
                {9: (0.0, 0.0, 0.0, 3)},
                {9: {0, 1, 2}},
                primitive_method="round_cone",
            )
            probes = scale * np.asarray(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
            )
            normalized_values.append(field.evaluate(probes)[0] / scale)
        np.testing.assert_allclose(normalized_values[0], normalized_values[1], atol=2e-12)
        np.testing.assert_allclose(normalized_values[1], normalized_values[2], atol=2e-12)

    def test_clipped_batch_evaluator_matches_scalar_oracle(self) -> None:
        starts = np.asarray([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ends = np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        radii = np.asarray([0.8, 0.3])
        capsules = _capsules(
            starts,
            ends,
            radii,
            radii,
            np.arange(2),
            cap_bif_at_start=np.asarray([False, True]),
            cap_bif_at_end=np.asarray([True, False]),
        )
        field = GraphImplicitField(capsules, primitive_method="round_cone")
        rng = np.random.default_rng(37)
        probes = rng.normal(size=(300, 3))
        batch = field.evaluate(probes)[0]
        scalar = np.asarray([field.sample(point).value for point in probes])
        np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=2e-12)

    def test_junction_plane_clips_dense_caps_in_local_intrinsic_halo(self) -> None:
        starts = np.asarray(
            [
                [-3.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -0.5, 0.0],
            ]
        )
        ends = np.asarray(
            [
                [-2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, -0.5, 0.0],
                [2.0, -1.0, 0.0],
            ]
        )
        segment_ids = np.asarray([0, 0, 0, 1, 1, 2, 2])
        radii = np.asarray([1.2, 1.2, 1.2, 0.4, 0.4, 0.4, 0.4])
        capsules = _capsules(
            starts,
            ends,
            radii,
            radii,
            segment_ids,
            cap_bif_at_start=np.asarray([False, False, False, True, False, True, False]),
            cap_bif_at_end=np.asarray([False, False, True, False, False, False, False]),
        )
        field = build_graph_implicit_field(
            capsules,
            {9: (0.0, 0.0, 0.0, 3)},
            {9: {0, 1, 2}},
            blend_fraction=0.05,
            support_factor=2.0,
            primitive_method="round_cone",
        )
        rounded = build_graph_implicit_field(
            capsules,
            {9: (0.0, 0.0, 0.0, 3)},
            {9: {0, 1, 2}},
            blend_fraction=0.05,
            support_factor=2.0,
            primitive_method="round_cone",
            clip_bifurcation_caps=False,
        )
        carina_probe = np.asarray([1.0, 0.0, 0.0])
        self.assertLess(rounded.sample(carina_probe).value, 0.0)
        self.assertGreater(field.sample(carina_probe).value, 0.0)
        self.assertAlmostEqual(field.sample(np.zeros(3)).value, -0.2, places=12)
        self.assertTrue(np.all(field.clip_plane_count[:3] == 1))

    def test_junction_plane_does_not_clip_tortuous_branch_returning_downstream(self) -> None:
        starts = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        ends = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        segment_ids = np.asarray([0, 0, 0, 0, 0, 1, 2])
        radii = np.full(len(starts), 0.2)
        capsules = _capsules(
            starts,
            ends,
            radii,
            radii,
            segment_ids,
            cap_bif_at_start=np.asarray([True, False, False, False, False, True, True]),
        )
        field = build_graph_implicit_field(
            capsules,
            {9: (0.0, 0.0, 0.0, 3)},
            {9: {0, 1, 2}},
            blend_fraction=0.05,
            support_factor=2.0,
            primitive_method="round_cone",
        )
        self.assertEqual(field.clip_plane_count[0], 1)
        self.assertTrue(np.all(field.clip_plane_count[1:5] == 0))
        self.assertLess(field.sample(np.asarray([-1.0, 2.0, 0.0])).value, 0.0)

    def test_global_scale_equivariance(self) -> None:
        base_capsules, nodes, incidence = _star(degree=5, scale=1.0)
        base = build_graph_implicit_field(base_capsules, nodes, incidence)
        probes = np.asarray(
            [[0.0, 0.0, 0.0], [0.4, 0.1, 0.2], [2.0, -0.3, 0.4], [8.0, 1.0, 0.0]]
        )
        reference, _owner, reference_radii, _gradient = base.evaluate(probes)

        for scale in (0.01, 100.0):
            scaled_capsules, scaled_nodes, scaled_incidence = _star(degree=5, scale=scale)
            scaled = build_graph_implicit_field(
                scaled_capsules, scaled_nodes, scaled_incidence
            )
            values, _owner, radii, _gradient = scaled.evaluate(probes * scale)
            np.testing.assert_allclose(values / scale, reference, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(
                radii / scale, reference_radii, rtol=2e-12, atol=2e-12
            )

    def test_nonadjacent_parallel_branches_are_not_smoothed_together(self) -> None:
        starts = np.asarray([[0.0, -0.7, 0.0], [0.0, 0.7, 0.0]])
        ends = np.asarray([[5.0, -0.7, 0.0], [5.0, 0.7, 0.0]])
        radius = np.asarray([0.5, 0.5])
        field = GraphImplicitField(_capsules(starts, ends, radius, radius, np.arange(2)))
        # The midpoint has 0.2 mm clearance to both surfaces. A global
        # smooth-min would depress this value and can create a bridge.
        self.assertAlmostEqual(field.sample(np.asarray([2.5, 0.0, 0.0])).value, 0.2)

    def test_fixed_radius_overlap_is_reported_scale_invariantly(self) -> None:
        normalized = []
        for scale in (0.01, 1.0, 100.0):
            starts = scale * np.asarray([[0.0, -0.4, 0.0], [0.0, 0.4, 0.0]])
            ends = scale * np.asarray([[5.0, -0.4, 0.0], [5.0, 0.4, 0.0]])
            radius = scale * np.asarray([0.5, 0.5])
            capsules = _capsules(starts, ends, radius, radius, np.arange(2))
            conflicts = find_capsule_conflicts(capsules)
            self.assertEqual(len(conflicts), 1)
            normalized.append(conflicts[0].normalized_clearance)
        np.testing.assert_allclose(normalized, [-0.4, -0.4, -0.4], atol=1e-12)

    def test_adjacent_segments_are_exempt_only_near_shared_node(self) -> None:
        starts = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0], [1.0, 1.5, 0.0],
        ])
        ends = np.asarray([
            [1.0, 1.0, 0.0], [5.0, 1.0, 0.0],
            [1.0, 1.5, 0.0], [5.0, 1.5, 0.0],
        ])
        radius = np.full(4, 0.3)
        capsules = _capsules(starts, ends, radius, radius, np.asarray([0,0,1,1]))
        lengths = np.linalg.norm(ends-starts, axis=1)
        capsules.arc_start[:] = [0.0, lengths[0], 0.0, lengths[2]]
        capsules.arc_end[:] = [lengths[0], lengths[0]+lengths[1], lengths[2], lengths[2]+lengths[3]]
        capsules.seg_L[:] = [lengths[0]+lengths[1], lengths[2]+lengths[3]]
        adjacency = np.ones((2,2), dtype=bool)
        conflicts = find_capsule_conflicts(
            capsules,
            adjacency,
            shared_node_positions={(0,1):np.zeros(3),(1,0):np.zeros(3)},
            shared_node_radii={(0,1):0.3,(1,0):0.3},
        )
        self.assertTrue(any(c.segment_a != c.segment_b for c in conflicts))

    def test_adaptive_hierarchy_is_scale_invariant(self) -> None:
        signatures = []
        for scale in (0.1, 1.0, 10.0):
            starts = np.asarray([[0.0, 0.0, 0.0]]) * scale
            ends = np.asarray([[2.0, 0.0, 0.0]]) * scale
            radius = np.asarray([0.25 * scale])
            field = GraphImplicitField(_capsules(starts, ends, radius, radius, np.asarray([0])))
            octree = RadiusAdaptiveOctree(
                field,
                cells_across_diameter=4.0,
                padding_radius_factor=1.0,
                maximum_depth=12,
            )
            stats = octree.build()
            sizes = np.sort(octree.leaf_arrays()["sizes"] / scale)
            signatures.append((stats.active_leaves, stats.maximum_depth, sizes))
        self.assertEqual(signatures[0][0:2], signatures[1][0:2])
        self.assertEqual(signatures[1][0:2], signatures[2][0:2])
        np.testing.assert_allclose(signatures[0][2], signatures[1][2], atol=1e-13)
        np.testing.assert_allclose(signatures[1][2], signatures[2][2], atol=1e-13)

    def test_reference_adaptive_mesh_is_watertight_and_manifold(self) -> None:
        starts = np.asarray([[0.0, 0.0, 0.0]])
        ends = np.asarray([[2.0, 0.0, 0.0]])
        radius = np.asarray([0.25])
        field = GraphImplicitField(_capsules(starts, ends, radius, radius, np.asarray([0])))
        octree = RadiusAdaptiveOctree(
            field,
            cells_across_diameter=4.0,
            padding_radius_factor=1.0,
            maximum_depth=10,
        )
        octree.build()
        mesh = mesh_adaptive_implicit(field, octree, maximum_points=100_000)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_manifold)
        self.assertGreater(len(mesh.faces), 0)
        surface_error = np.abs(field.evaluate(mesh.vertices)[0])
        self.assertLess(float(surface_error.max()), 2e-6)


if __name__ == "__main__":
    unittest.main()
