import numpy as np
import pytest

from coronary_sdf import simpleware_coronary_regions as regions


class _Vector:
    def __init__(self, x, y, z):
        self.values = (x, y, z)

    def GetX(self):
        return self.values[0]

    def GetY(self):
        return self.values[1]

    def GetZ(self):
        return self.values[2]


class _Node:
    def __init__(self, name):
        self.name = name
        self.spline = None

    def GetName(self):
        return self.name

    def GetSplines(self):
        return [self.spline]

    def GetPosition(self, _image_space):
        return _Vector(0.0 if self.name == "start" else 10.0, 0.0, 0.0)


class _StraightSpline:
    def __init__(self):
        self.start = _Node("start")
        self.end = _Node("end")
        self.start.spline = self
        self.end.spline = self

    def IsClosed(self):
        return False

    def GetLength(self, _image_space):
        return 10.0

    def GetStartNode(self):
        return self.start

    def GetEndNode(self):
        return self.end

    def GetName(self):
        return "straight"

    def GetRawDataPoints(self, _image_space):
        return [_Vector(float(x), 0.0, 0.0) for x in range(11)]

    def GetParameterAtDistance(self, distance, _image_space):
        return distance / 10.0

    def GetPosition(self, parameter, _image_space):
        return _Vector(10.0 * parameter, 0.0, 0.0)


def _unit_cube_triangles():
    p = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    ]
    return np.asarray([[p[a], p[b], p[c]] for a, b, c in faces])

def _rotate_z(axis, angle_degrees):
    axis = np.asarray(axis, dtype=float)
    z = np.array([0.0, 0.0, 1.0])
    angle = np.radians(angle_degrees)
    return (
        z * np.cos(angle)
        + np.cross(axis, z) * np.sin(angle)
        + axis * np.dot(axis, z) * (1.0 - np.cos(angle))
    )


def _rotation_matrix(axis, angle_degrees):
    axis = np.asarray(axis, dtype=float)
    angle = np.radians(angle_degrees)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * skew
    )


def test_rotation_from_positive_z_maps_to_requested_direction():
    for direction in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [1, 2, 3]):
        axis, angle = regions.rotation_from_positive_z(direction)
        expected = np.asarray(direction, dtype=float)
        expected /= np.linalg.norm(expected)
        assert np.allclose(_rotate_z(axis, angle), expected, atol=1.0e-10)


def test_rotation_from_basis_preserves_in_plane_rectangle_axes():
    first = np.array([0.0, 1.0, 0.0])
    second = np.array([0.0, 0.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    axis, angle = regions.rotation_from_basis(first, second, normal)
    matrix = _rotation_matrix(axis, angle)
    assert np.allclose(matrix[:, 0], first, atol=1.0e-9)
    assert np.allclose(matrix[:, 1], second, atol=1.0e-9)
    assert np.allclose(matrix[:, 2], normal, atol=1.0e-9)


def test_stl_clip_bridges_enclosed_parity_excursion(monkeypatch):
    class Solid:
        @staticmethod
        def contains(values):
            values = np.asarray(values, dtype=float)
            x = np.atleast_2d(values)[:, 0]
            result = ((x >= 0.0) & (x <= 1.0)) | ((x >= 2.0) & (x <= 3.0))
            return bool(result[0]) if values.ndim == 1 else result

    monkeypatch.setattr(regions, "GRAPH_CROP_SAMPLE_SPACING_MM", 0.25)
    monkeypatch.setattr(regions, "GRAPH_CROP_MIN_FRAGMENT_LENGTH_MM", 0.1)
    points = np.array([[-1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    fragments = regions._clip_polyline_to_stl(points, np.ones(2), Solid())
    assert len(fragments) == 1
    clipped, _radii = fragments[0]
    assert clipped[0, 0] == pytest.approx(0.0, abs=1.0e-6)
    assert clipped[-1, 0] == pytest.approx(3.0, abs=1.0e-6)
    assert np.any((clipped[:, 0] > 1.0) & (clipped[:, 0] < 2.0))


def test_rectangle_collision_checks_segments_with_outside_endpoints():
    crossing = np.array([[-2.0, 0.0], [2.0, 0.0]])
    separate = np.array([[-2.0, 2.0], [2.0, 2.0]])
    assert regions._polyline_intersects_rectangle(crossing, 1.0, 1.0)
    assert not regions._polyline_intersects_rectangle(separate, 1.0, 1.0)


def test_terminal_normals_point_out_of_the_tree():
    spline = _StraightSpline()
    start = regions._terminal_record(spline.start)
    end = regions._terminal_record(spline.end)

    assert start["inward_node_count"] >= 2
    assert end["inward_node_count"] >= 2
    assert np.allclose(start["centre"], [2.0, 0.0, 0.0])
    assert np.allclose(start["normal"], [-1.0, 0.0, 0.0])
    assert np.allclose(end["centre"], [8.0, 0.0, 0.0])
    assert np.allclose(end["normal"], [1.0, 0.0, 0.0])


def test_boundary_terminal_merge_prefers_amira_without_merging_same_source(monkeypatch):
    monkeypatch.setattr(regions, "BOUNDARY_TERMINAL_MERGE_DISTANCE_MM", 1.0)
    amira_a = {
        "centreline_source": "amira",
        "terminal_position": np.array([0.0, 0.0, 0.0]),
    }
    amira_b = {
        "centreline_source": "amira",
        "terminal_position": np.array([0.2, 0.0, 0.0]),
    }
    simpleware_duplicate = {
        "centreline_source": "simpleware",
        "terminal_position": np.array([0.1, 0.0, 0.0]),
    }
    simpleware_unique = {
        "centreline_source": "simpleware",
        "terminal_position": np.array([3.0, 0.0, 0.0]),
    }

    merged = regions._merge_boundary_terminals([
        ("amira", [amira_a, amira_b]),
        ("simpleware", [simpleware_duplicate, simpleware_unique]),
    ])

    assert merged == [amira_a, amira_b, simpleware_unique]


def test_surface_axis_fit_corrects_an_oblique_terminal_normal(monkeypatch):
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    axial = np.linspace(-2.0, 2.0, 31)
    surface_points = np.asarray([
        [np.cos(angle), np.sin(angle), z]
        for z in axial
        for angle in angles
    ])
    initial = np.array([0.30, 0.0, 1.0])
    initial /= np.linalg.norm(initial)
    record = {
        "radius": 1.0,
        "centre": np.zeros(3),
        "normal": initial.copy(),
    }
    monkeypatch.setattr(regions, "REFINE_PLANE_NORMAL_FROM_SURFACE", True)
    monkeypatch.setattr(regions, "SURFACE_AXIS_MIN_POINTS", 30)
    monkeypatch.setattr(regions, "SURFACE_AXIS_MAX_CORRECTION_DEGREES", 50.0)

    regions._refine_plane_normal_from_surface(
        record, surface_points, regions.cKDTree(surface_points)
    )

    assert record["surface_axis_used"] is True
    assert np.sum(record["normal"] * np.array([0.0, 0.0, 1.0])) > 0.999
    assert record["surface_axis_correction_degrees"] > 10.0


def test_tangent_average_suppresses_alternating_node_noise(monkeypatch):
    monkeypatch.setattr(regions, "TANGENT_AVERAGING_NODE_COUNT", 5)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],
        [2.0, -0.2, 0.0],
        [3.0, 0.2, 0.0],
        [4.0, -0.2, 0.0],
        [5.0, 0.0, 0.0],
    ])

    tangent, first, last = regions._averaged_inward_tangent(points, 2)

    assert last - first + 1 == 5
    assert tangent[0] > 0.98
    assert abs(tangent[1]) < 0.1


def test_amira_spline_interpolates_position_and_radius():
    start = regions._AmiraNode(0, [0.0, 0.0, 0.0])
    end = regions._AmiraNode(1, [4.0, 0.0, 0.0])
    spline = regions._AmiraSpline(
        7,
        start,
        end,
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        [0.2, 0.4, 0.6],
    )

    position = regions._xyz(spline.GetPosition(0.25, False))
    assert np.allclose(position, [1.0, 0.0, 0.0])
    assert spline.RadiusAtDistance(1.0) == pytest.approx(0.3)
    assert start.GetSplines() == [spline]
    assert end.GetSplines() == [spline]


def test_amira_alignment_rejects_coordinate_mismatch():
    start = regions._AmiraNode(0, [0.0, 0.0, 0.0])
    end = regions._AmiraNode(1, [4.0, 0.0, 0.0])
    spline = regions._AmiraSpline(
        7,
        start,
        end,
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        [0.2, 0.3, 0.4],
    )
    network = regions._AmiraNetwork([start, end], [spline], "graph.am")
    surface_points = np.array([
        [100.0, 100.0, 100.0],
        [101.0, 101.0, 101.0],
    ])
    surface_tree = regions.cKDTree(surface_points)

    with pytest.raises(RuntimeError, match="Check AMIRA_OUTPUT_UM_TO_MM"):
        regions._validate_amira_alignment(network, surface_points, surface_tree)


def test_refinement_can_be_selected_by_strahler_order(monkeypatch):
    start = regions._AmiraNode(0, [0.0, 0.0, 0.0])
    end = regions._AmiraNode(1, [4.0, 0.0, 0.0])
    spline = regions._AmiraSpline(
        7,
        start,
        end,
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        [1.0, 1.0],
        strahler_order=1,
    )
    network = regions._AmiraNetwork([start, end], [spline], "graph.am")
    surface_tree = regions.cKDTree(np.array([[0.0, 0.0, 0.0]]))
    monkeypatch.setattr(regions, "REFINEMENT_SAMPLE_SPACING_MM", 10.0)
    monkeypatch.setattr(regions, "SMALL_VESSEL_DIAMETER_MM", 1.0)
    monkeypatch.setattr(regions, "REFINEMENT_STRAHLER_ORDERS", {1, 2})

    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "strahler")
    selected = regions._small_vessel_segments(network, surface_tree)
    assert len(selected) == 1
    assert selected[0]["strahler_order"] == 1

    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "diameter")
    assert regions._small_vessel_segments(network, surface_tree) == []

    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "either")
    assert len(regions._small_vessel_segments(network, surface_tree)) == 1

    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "both")
    assert regions._small_vessel_segments(network, surface_tree) == []


def test_refinement_can_select_by_exact_surface_cross_section_radius(monkeypatch):
    start = regions._AmiraNode(0, [0.0, 0.0, 0.0])
    end = regions._AmiraNode(1, [1.0, 0.0, 0.0])
    spline = regions._AmiraSpline(
        7,
        start,
        end,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [0.8, 0.8],
        strahler_order=3,
    )
    network = regions._AmiraNetwork([start, end], [spline], "graph.am")
    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "radius")
    monkeypatch.setattr(regions, "REFINEMENT_RADIUS_SOURCE", "surface_cross_section")
    monkeypatch.setattr(regions, "SMALL_VESSEL_CROSS_SECTION_RADIUS_MM", 0.5)
    monkeypatch.setattr(regions, "REFINEMENT_SAMPLE_SPACING_MM", 2.0)
    monkeypatch.setattr(
        regions,
        "_validate_stl_plane_intersection",
        lambda *_args, **_kwargs: ({
            "surface_loop_equivalent_radius": 0.40,
            "surface_loop_radius": 0.45,
        }, None),
    )

    selected = regions._small_vessel_segments(
        network, None, stl_solid=object()
    )

    assert len(selected) == 1
    assert selected[0]["selection_radius"] == pytest.approx(0.40)
    assert selected[0]["radius"] == pytest.approx(0.40)
    assert selected[0]["surface_radius_measured"] is True


def test_refinement_rejects_implausibly_large_surface_radius(monkeypatch):
    start = regions._AmiraNode(0, [0.0, 0.0, 0.0])
    end = regions._AmiraNode(1, [1.0, 0.0, 0.0])
    spline = regions._AmiraSpline(
        7,
        start,
        end,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [0.8, 0.8],
        strahler_order=3,
    )
    network = regions._AmiraNetwork([start, end], [spline], "graph.am")
    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "radius")
    monkeypatch.setattr(regions, "REFINEMENT_RADIUS_SOURCE", "surface_cross_section")
    monkeypatch.setattr(regions, "SMALL_VESSEL_CROSS_SECTION_RADIUS_MM", 0.5)
    monkeypatch.setattr(regions, "REFINEMENT_SAMPLE_SPACING_MM", 2.0)
    monkeypatch.setattr(regions, "REFINEMENT_SURFACE_RADIUS_MIN_GRAPH_FACTOR", 0.4)
    monkeypatch.setattr(regions, "REFINEMENT_SURFACE_RADIUS_MAX_GRAPH_FACTOR", 1.6)
    monkeypatch.setattr(
        regions,
        "_validate_stl_plane_intersection",
        lambda *_args, **_kwargs: ({
            "surface_loop_equivalent_radius": 3.0,
            "surface_loop_radius": 3.5,
        }, None),
    )

    assert regions._small_vessel_segments(
        network, None, stl_solid=object()
    ) == []


def test_refinement_clearance_reduces_padding_around_unselected_branch(monkeypatch):
    monkeypatch.setattr(regions, "REFINEMENT_PRIMITIVE", "ellipsoid")
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_MM", 0.35)
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_RADIUS_FACTOR", 0.0)
    monkeypatch.setattr(regions, "REFINEMENT_MIN_PADDING_MM", 0.05)
    monkeypatch.setattr(regions, "REFINEMENT_CLEARANCE_SAFETY_FACTOR", 0.9)
    monkeypatch.setattr(regions, "REFINEMENT_CLEARANCE_LOCAL_EXCLUSION_MM", 0.1)
    segment = {
        "spline_name": "target",
        "segment_index": 0,
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([1.0, 0.0, 0.0]),
        "length": 1.0,
        "start_distance": 0.0,
        "end_distance": 1.0,
        "radius": 0.2,
    }
    clearance_samples = [
        {
            "spline_name": "target",
            "distances": np.array([0.0, 0.5, 1.0]),
            "points": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "radii": np.full(3, 0.2),
        },
        {
            "spline_name": "neighbour",
            "distances": np.array([0.0, 0.5, 1.0]),
            "points": np.array([[0.0, 0.7, 0.0], [0.5, 0.7, 0.0], [1.0, 0.7, 0.0]]),
            "radii": np.full(3, 0.2),
        },
    ]
    intervals = regions._selected_refinement_intervals([segment])

    assert regions._set_refinement_clearance(
        segment, clearance_samples, intervals
    )
    assert segment["clearance_reduced"] is True
    assert segment["region_radius"] == pytest.approx(0.45)
    assert "refinement_skip_reason" not in segment


def test_refinement_clearance_rejects_unavoidable_neighbour_overlap(monkeypatch):
    monkeypatch.setattr(regions, "REFINEMENT_PRIMITIVE", "ellipsoid")
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_MM", 0.35)
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_RADIUS_FACTOR", 0.0)
    monkeypatch.setattr(regions, "REFINEMENT_MIN_PADDING_MM", 0.05)
    monkeypatch.setattr(regions, "REFINEMENT_CLEARANCE_SAFETY_FACTOR", 0.9)
    monkeypatch.setattr(regions, "REFINEMENT_CLEARANCE_LOCAL_EXCLUSION_MM", 0.1)
    segment = {
        "spline_name": "target",
        "segment_index": 0,
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([1.0, 0.0, 0.0]),
        "length": 1.0,
        "start_distance": 0.0,
        "end_distance": 1.0,
        "radius": 0.2,
    }
    clearance_samples = [{
        "spline_name": "neighbour",
        "distances": np.array([0.5]),
        "points": np.array([[0.5, 0.4, 0.0]]),
        "radii": np.array([0.2]),
    }]
    intervals = regions._selected_refinement_intervals([segment])

    assert not regions._set_refinement_clearance(
        segment, clearance_samples, intervals
    )
    assert "target coverage requires" in segment["refinement_skip_reason"]


def test_refinement_clearance_ignores_other_selected_interval(monkeypatch):
    monkeypatch.setattr(regions, "REFINEMENT_PRIMITIVE", "sphere")
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_MM", 0.05)
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_RADIUS_FACTOR", 0.25)
    monkeypatch.setattr(regions, "REFINEMENT_MIN_PADDING_MM", 0.03)
    segment = {
        "spline_name": "target",
        "segment_index": 0,
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([0.3, 0.0, 0.0]),
        "length": 0.3,
        "start_distance": 0.0,
        "end_distance": 0.3,
        "radius": 0.2,
    }
    selected_neighbour = {
        "spline_name": "neighbour",
        "segment_index": 0,
        "start": np.array([0.0, 0.35, 0.0]),
        "end": np.array([0.3, 0.35, 0.0]),
        "length": 0.3,
        "start_distance": 0.0,
        "end_distance": 0.3,
        "radius": 0.2,
    }
    clearance_samples = [{
        "spline_name": "neighbour",
        "distances": np.array([0.0, 0.15, 0.3]),
        "points": np.array([
            [0.0, 0.35, 0.0], [0.15, 0.35, 0.0], [0.3, 0.35, 0.0]
        ]),
        "radii": np.full(3, 0.2),
    }]
    intervals = regions._selected_refinement_intervals(
        [segment, selected_neighbour]
    )

    assert regions._set_refinement_clearance(
        segment, clearance_samples, intervals
    )
    assert "refinement_skip_reason" not in segment


def test_refinement_clearance_ignores_same_and_connected_splines(monkeypatch):
    monkeypatch.setattr(regions, "REFINEMENT_PRIMITIVE", "sphere")
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_MM", 0.05)
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_RADIUS_FACTOR", 0.25)
    monkeypatch.setattr(regions, "REFINEMENT_MIN_PADDING_MM", 0.03)
    segment = {
        "spline_name": "target",
        "spline_start_node": "node_a",
        "spline_end_node": "node_b",
        "segment_index": 0,
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([0.3, 0.0, 0.0]),
        "length": 0.3,
        "start_distance": 0.0,
        "end_distance": 0.3,
        "radius": 0.2,
    }
    clearance_samples = [
        {
            "spline_name": "target",
            "start_node_name": "node_a",
            "end_node_name": "node_b",
            "distances": np.array([0.4, 0.6]),
            "points": np.array([[0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]),
            "radii": np.full(2, 0.2),
        },
        {
            "spline_name": "daughter",
            "start_node_name": "node_b",
            "end_node_name": "node_c",
            "distances": np.array([0.0, 0.2]),
            "points": np.array([[0.3, 0.0, 0.0], [0.3, 0.2, 0.0]]),
            "radii": np.full(2, 0.15),
        },
    ]
    radial, axial, obstacle_radius, names = regions._refinement_obstacle_coordinates(
        segment, clearance_samples
    )
    assert len(radial) == len(axial) == len(obstacle_radius) == len(names) == 0


def test_adaptive_sphere_spacing_merges_only_while_size_stays_local(monkeypatch):
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_RADIUS_FACTOR", 0.25)
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_MM", 0.05)
    monkeypatch.setattr(regions, "REFINEMENT_MIN_PADDING_MM", 0.03)
    monkeypatch.setattr(
        regions, "REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR", 1.35
    )
    monkeypatch.setattr(regions, "REFINEMENT_SPHERE_MAX_ARC_LENGTH_MM", 0.9)
    segments = []
    for index in range(4):
        start = 0.3 * index
        end = 0.3 * (index + 1)
        segments.append({
            "spline_name": "edge",
            "segment_index": index,
            "start": np.array([start, 0.0, 0.0]),
            "end": np.array([end, 0.0, 0.0]),
            "length": 0.3,
            "start_distance": start,
            "end_distance": end,
            "radius": 0.3,
            "start_radius": 0.3,
            "end_radius": 0.3,
            "selection_radius": 0.3,
            "start_selection_radius": 0.3,
            "end_selection_radius": 0.3,
            "graph_radius": 0.3,
            "surface_radius_measured": True,
            "surface_radius_outlier": False,
            "strahler_order": 1,
        })

    optimised = regions._optimise_refinement_spheres(segments)

    assert len(optimised) == 2
    assert [item["merged_segment_count"] for item in optimised] == [2, 2]
    assert [item["segment_end_index"] for item in optimised] == [1, 3]
    assert all(
        item["sphere_desired_radius"]
        <= 1.35 * item["sphere_wall_radius"] + 1.0e-12
        for item in optimised
    )


def test_adaptive_sphere_spacing_does_not_bridge_unselected_gap(monkeypatch):
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_RADIUS_FACTOR", 0.25)
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_MM", 0.05)
    monkeypatch.setattr(regions, "REFINEMENT_MIN_PADDING_MM", 0.03)
    monkeypatch.setattr(
        regions, "REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR", 2.0
    )
    monkeypatch.setattr(regions, "REFINEMENT_SPHERE_MAX_ARC_LENGTH_MM", 2.0)
    base = {
        "spline_name": "edge",
        "length": 0.3,
        "radius": 0.3,
        "start_radius": 0.3,
        "end_radius": 0.3,
        "selection_radius": 0.3,
        "start_selection_radius": 0.3,
        "end_selection_radius": 0.3,
        "graph_radius": 0.3,
        "surface_radius_measured": True,
        "surface_radius_outlier": False,
        "strahler_order": 1,
    }
    first = dict(base, segment_index=0, start=np.array([0.0, 0.0, 0.0]),
                 end=np.array([0.3, 0.0, 0.0]), start_distance=0.0,
                 end_distance=0.3)
    # Segment index 1 was not selected; index 2 must start a new sphere run.
    third = dict(base, segment_index=2, start=np.array([0.6, 0.0, 0.0]),
                 end=np.array([0.9, 0.0, 0.0]), start_distance=0.6,
                 end_distance=0.9)

    optimised = regions._optimise_refinement_spheres([first, third])

    assert len(optimised) == 2
    assert all(item["merged_segment_count"] == 1 for item in optimised)


def test_boundary_planes_do_not_filter_strahler_by_default(monkeypatch):
    monkeypatch.setattr(regions, "BOUNDARY_OUTLET_STRAHLER_ORDERS", None)
    order_one = {
        "centreline_source": "amira",
        "spline": type("Spline", (), {"strahler_order": 1})(),
    }
    order_two = {
        "centreline_source": "amira",
        "spline": type("Spline", (), {"strahler_order": 2})(),
    }
    simpleware = {
        "centreline_source": "simpleware",
        "spline": type("Spline", (), {})(),
    }

    retained, rejected = regions._filter_boundary_outlets_by_strahler([
        order_one, order_two, simpleware
    ])

    assert retained == [order_one, order_two, simpleware]
    assert rejected == []


def test_force_outward_plane_normal_flips_inward_normal():
    record = {
        "node_name": "outlet",
        "centre": np.array([0.0, 0.0, 0.0]),
        "terminal_position": np.array([1.0, 0.0, 0.0]),
        "normal": np.array([-1.0, 0.0, 0.0]),
    }

    regions._force_outward_plane_normal(record)

    assert np.allclose(record["normal"], [1.0, 0.0, 0.0])
    assert record["normal_was_flipped"] is True
    assert record["normal_outward_cosine"] == pytest.approx(1.0)


def test_surface_cross_section_can_enlarge_required_plane(monkeypatch):
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    surface_points = np.asarray([
        [1.4 * np.cos(angle), 1.4 * np.sin(angle), z]
        for z in (-0.03, 0.0, 0.03)
        for angle in angles
    ])
    record = {
        "node_name": "outlet",
        "centre": np.zeros(3),
        "normal": np.array([0.0, 0.0, 1.0]),
        "radius": 1.0,
        "safe_half_width": 2.0,
        "neighbour_surface_clearance": 3.0,
    }
    monkeypatch.setattr(regions, "SURFACE_CROSS_SECTION_MIN_POINTS", 12)

    regions._measure_surface_cross_section(
        record, surface_points, regions.cKDTree(surface_points)
    )

    assert record["surface_cross_section_used"] is True
    assert record["surface_cross_section_radius"] == pytest.approx(1.4)
    assert regions._plane_half_width(record) >= 1.4 * 1.1 - 1.0e-12

def test_strahler_selection_keeps_edge_with_tiny_endpoint_radius(monkeypatch):
    start = regions._AmiraNode(0, [0.0, 0.0, 0.0])
    end = regions._AmiraNode(1, [1.0, 0.0, 0.0])
    spline = regions._AmiraSpline(
        7,
        start,
        end,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [0.0, 0.0],
        strahler_order=1,
    )
    network = regions._AmiraNetwork([start, end], [spline], "graph.am")
    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "strahler")
    monkeypatch.setattr(regions, "REFINEMENT_STRAHLER_ORDERS", {1, 2})
    monkeypatch.setattr(regions, "REFINEMENT_SAMPLE_SPACING_MM", 2.0)
    monkeypatch.setattr(regions, "MIN_PLAUSIBLE_RADIUS_MM", 0.03)

    selected = regions._small_vessel_segments(network, None)

    assert len(selected) == 1
    assert selected[0]["radius"] == pytest.approx(0.03)


def test_strahler_selection_requires_edge_metadata(monkeypatch):
    spline = _StraightSpline()
    network = type("Network", (), {"GetSplines": lambda self: [spline]})()
    surface_tree = regions.cKDTree(np.array([[0.0, 1.0, 0.0]]))
    monkeypatch.setattr(regions, "REFINEMENT_SELECTION_MODE", "strahler")
    monkeypatch.setattr(regions, "REFINEMENT_STRAHLER_ORDERS", {1})

    with pytest.raises(RuntimeError, match="has no Strahler order"):
        regions._small_vessel_segments(network, surface_tree)


def test_stl_solid_classifies_inside_and_outside_points():
    solid = regions._StlSolid(_unit_cube_triangles(), "cube.stl")

    assert solid.contains([0.5, 0.5, 0.5])
    assert not solid.contains([-0.5, 0.5, 0.5])
    assert not solid.contains([1.5, 0.5, 0.5])
    assert solid.contains([[0.25, 0.25, 0.25], [2.0, 2.0, 2.0]]).tolist() == [
        True, False
    ]


def test_exact_stl_plane_validation_finds_one_complete_cube_loop(monkeypatch):
    monkeypatch.setattr(
        regions, "STL_PLANE_INTERSECTION_STITCH_TOLERANCE_MM", 1.0e-6
    )
    solid = regions._StlSolid(_unit_cube_triangles(), "cube.stl")

    result, reason = regions._validate_stl_plane_intersection(
        solid,
        centre=np.array([0.5, 0.5, 0.5]),
        normal=np.array([0.0, 0.0, 1.0]),
        graph_radius=0.5,
    )

    assert reason is None
    assert result["surface_loop_area"] == pytest.approx(1.0)
    assert result["surface_loop_radius"] == pytest.approx(np.sqrt(0.5))


def test_exact_stl_plane_validation_rejects_plane_outside_lumen(monkeypatch):
    monkeypatch.setattr(
        regions, "STL_PLANE_INTERSECTION_STITCH_TOLERANCE_MM", 1.0e-6
    )
    solid = regions._StlSolid(_unit_cube_triangles(), "cube.stl")

    result, reason = regions._validate_stl_plane_intersection(
        solid,
        centre=np.array([0.5, 0.5, 1.5]),
        normal=np.array([0.0, 0.0, 1.0]),
        graph_radius=0.5,
    )

    assert result is None
    assert "expected one closed loop" in reason


def test_clustered_stl_cap_fit_recovers_oblique_outlet_normal():
    angle = np.radians(40.0)
    expected_normal = np.array([np.sin(angle), 0.0, np.cos(angle)])
    first = np.array([0.0, 1.0, 0.0])
    second = np.cross(expected_normal, first)
    ring = [
        0.5 * (
            np.cos(theta) * first + np.sin(theta) * second
        )
        for theta in np.linspace(0.0, 2.0 * np.pi, 33)[:-1]
    ]
    triangles = np.asarray([
        [np.zeros(3), ring[index], ring[(index + 1) % len(ring)]]
        for index in range(len(ring))
    ])

    class Solid:
        def nearby_triangles(self, _centre, _radius):
            return triangles

    result, reason = regions._fit_stl_terminal_cap_clustered(
        Solid(),
        anchor=np.zeros(3),
        outward_seed=np.array([0.0, 0.0, 1.0]),
        graph_radius=0.5,
    )

    assert reason is None
    assert result["centre"] == pytest.approx(np.zeros(3), abs=1.0e-10)
    assert result["normal"] == pytest.approx(expected_normal, abs=1.0e-10)
    assert result["graph_deviation_degrees"] == pytest.approx(40.0)
    assert result["triangle_count"] == 32


def test_adaptive_inset_moves_inward_but_preserves_short_branches(monkeypatch):
    monkeypatch.setattr(regions, "STL_PLANE_DESIRED_INSET_MM", 0.20)
    monkeypatch.setattr(regions, "STL_PLANE_DESIRED_INSET_RADIUS_FACTOR", 0.75)
    monkeypatch.setattr(regions, "STL_PLANE_MAX_TERMINAL_BRANCH_FRACTION", 0.20)

    long_schedule, long_policy = regions._terminal_plane_inset_schedule(5.0, 0.2)
    assert long_schedule[0] == pytest.approx(0.20)
    assert not long_policy["short_branch_limited"]

    large_schedule, _large_policy = regions._terminal_plane_inset_schedule(5.0, 0.5)
    assert large_schedule[0] == pytest.approx(0.375)

    short_schedule, short_policy = regions._terminal_plane_inset_schedule(0.30, 0.2)
    assert short_schedule[0] == pytest.approx(0.06)
    assert float(np.max(short_schedule)) <= 0.060000001
    assert short_policy["short_branch_limited"]


def test_terminal_plane_search_keeps_averaged_amira_tangent(monkeypatch):
    record = {
        "node_name": "tip",
        "radius": 0.1,
        "normal": np.array([0.0, 0.0, 1.0]),
        "terminal_raw_points": np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.1],
            [0.2, 0.0, -0.3],
            [0.4, 0.0, -0.5],
        ]),
    }
    monkeypatch.setattr(
        regions,
        "_terminal_surface_anchor",
        lambda _record, _solid: (
            # Deliberately offset the surface anchor: it must affect placement,
            # never the anatomical tangent taken from the Amira polyline.
            np.array([0.3, 0.0, 0.05]),
            np.array([0.0, 0.0, 1.0]),
            0.0,
        ),
    )
    normals = []

    def validate(_solid, _centre, normal, _radius):
        normals.append(np.asarray(normal, dtype=float).copy())
        if len(normals) == 1:
            return None, "try farther inward"
        return {
            "surface_loop_radius": 0.1,
            "surface_loop_area": 0.02,
            "surface_loop_points": 8,
            "surface_loop_centre_offset": np.zeros(3),
            "plane_half_width": 0.12,
            "plane_half_extent_x": 0.12,
            "plane_half_extent_y": 0.12,
            "plane_first_axis": np.array([1.0, 0.0, 0.0]),
            "plane_second_axis": np.array([0.0, 1.0, 0.0]),
            "surface_intersection_loop_count": 1,
        }, None

    monkeypatch.setattr(regions, "_validate_stl_plane_intersection", validate)
    monkeypatch.setattr(regions, "_force_outward_plane_normal", lambda _record: None)

    regions._search_stl_validated_terminal_plane(record, object())

    # First candidate is rejected; the accepted candidate is checked again at
    # its final recentered position before it can be persisted.
    assert len(normals) == 3
    assert normals[0] == pytest.approx(normals[1])
    assert normals[1] == pytest.approx(normals[2])
    raw = record["terminal_raw_points"]
    cumulative = regions._cumulative_distances(raw)
    expected = regions._local_averaged_outward_tangent(
        raw, cumulative, 0.0, regions.STL_PLANE_TANGENT_AVERAGING_LENGTH_MM
    )
    assert abs(float(np.dot(normals[0], expected))) == pytest.approx(1.0)
    assert record["terminal_tangent_cosine"] == pytest.approx(1.0)


def test_amira_network_is_cropped_and_reterminated_at_stl(monkeypatch):
    monkeypatch.setattr(regions, "GRAPH_CROP_SAMPLE_SPACING_MM", 0.1)
    monkeypatch.setattr(regions, "GRAPH_CROP_MIN_FRAGMENT_LENGTH_MM", 0.01)
    start = regions._AmiraNode(0, [-1.0, 0.5, 0.5])
    end = regions._AmiraNode(1, [2.0, 0.5, 0.5])
    spline = regions._AmiraSpline(
        7,
        start,
        end,
        [[-1.0, 0.5, 0.5], [2.0, 0.5, 0.5]],
        [0.2, 0.2],
        strahler_order=2,
    )
    network = regions._AmiraNetwork([start, end], [spline], "graph.am")
    solid = regions._StlSolid(_unit_cube_triangles(), "cube.stl")

    cropped = regions._crop_amira_network_to_stl(network, solid)

    assert len(cropped.GetSplines()) == 1
    retained = cropped.GetSplines()[0]
    assert retained.strahler_order == 2
    assert retained.points[0, 0] == pytest.approx(0.0, abs=1.0e-5)
    assert retained.points[-1, 0] == pytest.approx(1.0, abs=1.0e-5)
    assert len([node for node in cropped.GetNodes() if len(node.GetSplines()) == 1]) == 2


def test_largest_terminal_is_automatic_inlet(monkeypatch):
    monkeypatch.setattr(regions, "INLET_NODE_NAMES", set())
    monkeypatch.setattr(regions, "INLET_COUNT", 1)
    terminals = [
        {"node_name": "distal_a", "radius": 0.3},
        {"node_name": "root", "radius": 2.0},
        {"node_name": "distal_b", "radius": 0.4},
    ]

    inlets, outlets = regions._classify_terminals(terminals)

    assert [item["node_name"] for item in inlets] == ["root"]
    assert {item["node_name"] for item in outlets} == {"distal_a", "distal_b"}


def test_inlet_plane_is_enabled_and_selected_by_default():
    outlet = {"node_name": "distal"}
    inlet = {"node_name": "root"}

    assert regions.CREATE_INLET_PLANES is True
    assert regions._boundary_plane_records([outlet], [inlet]) == [outlet, inlet]


def test_plane_diameter_is_scaled_and_clamped(monkeypatch):
    monkeypatch.setattr(regions, "OUTLET_PLANE_DIAMETER_FACTOR", 1.5)
    monkeypatch.setattr(regions, "OUTLET_PLANE_MIN_DIAMETER_MM", 0.4)
    monkeypatch.setattr(regions, "OUTLET_PLANE_MAX_DIAMETER_MM", 10.0)

    assert regions._plane_diameter(0.01) == 0.4
    assert regions._plane_diameter(1.0) == 3.0
    assert regions._plane_diameter(10.0) == 10.0


def test_plane_size_is_capped_by_neighbour_clearance(monkeypatch):
    monkeypatch.setattr(regions, "OUTLET_PLANE_DIAMETER_FACTOR", 1.5)
    monkeypatch.setattr(regions, "OUTLET_PLANE_MIN_RADIUS_FACTOR", 1.1)
    monkeypatch.setattr(regions, "OUTLET_PLANE_MIN_DIAMETER_MM", 0.4)
    monkeypatch.setattr(regions, "OUTLET_PLANE_MAX_DIAMETER_MM", 10.0)
    record = {
        "node_name": "outlet",
        "radius": 1.0,
        "safe_half_width": 1.2,
        "neighbour_surface_clearance": 2.0,
    }
    assert regions._plane_half_width(record) == 1.2

    record["safe_half_width"] = 1.0
    with pytest.raises(RuntimeError, match="No safe clipping plane"):
        regions._plane_half_width(record)


def test_clipping_plane_passes_half_extent_as_simpleware_scale(monkeypatch):
    class Vec:
        def __init__(self, x, y, z):
            self.values = (x, y, z)

    class Roi:
        def SetName(self, *_args):
            pass

    class Document:
        def CreateRegionOfInterestVolume(self, *args):
            self.args = args
            return Roi()

    monkeypatch.setattr(
        regions,
        "Doc",
        type("Doc", (), {"Clipping": "clipping", "FinitePlane": "plane"}),
    )
    monkeypatch.setattr(regions, "Vector3D", Vec)
    monkeypatch.setattr(regions, "INVERT_CLIPPING_PLANES", False)
    document = Document()
    record = {
        "normal": np.array([0.0, 0.0, 1.0]),
        "centre": np.array([1.0, 2.0, 3.0]),
        "plane_half_width": 0.75,
    }

    regions._create_clipping_plane(document, record, "outlet")

    assert document.args[3].values == pytest.approx((0.75, 0.75, 1.0))
    assert document.args[6] is False


def test_remove_existing_can_remove_legacy_manual_planes(monkeypatch):
    class Roi:
        def __init__(self, name):
            self.name = name

        def GetName(self):
            return self.name

    class Document:
        def __init__(self):
            self.rois = [Roi("Finite plane1"), Roi("COR_OUTLET_001_tip")]
            self.removed = []

        def GetRegionOfInterestVolumes(self, _kind):
            return self.rois

        def RemoveRegionOfInterestVolumeByName(self, _kind, name):
            self.removed.append(name)

    class Model:
        def __init__(self):
            self.contacts_removed = []

        def RemoveCfdSurfaceContact(self, part, roi):
            self.contacts_removed.append((part, roi.GetName()))

        def GetFeFreeMeshRefinementVolumes(self):
            return []

    monkeypatch.setattr(regions, "Doc", type("Doc", (), {"Clipping": object()}))
    monkeypatch.setattr(regions, "REPLACE_EXISTING", True)
    monkeypatch.setattr(regions, "REMOVE_ALL_EXISTING_CLIPPING_PLANES", True)
    document = Document()
    model = Model()

    regions._remove_existing(document, model, "part")

    assert document.removed == ["Finite plane1", "COR_OUTLET_001_tip"]
    assert [name for _part, name in model.contacts_removed] == document.removed


def test_refinement_cylinder_overlaps_chord_ends(monkeypatch):
    class Vec:
        def __init__(self, x, y, z):
            self.values = (x, y, z)

    class Refinement:
        def SetName(self, *_args):
            pass

        def SetValueType(self, *_args):
            pass

        def SetMeshSize(self, *_args):
            pass

        def SetRefinementType(self, *_args):
            pass

        def SetParts(self, *_args):
            pass

    class Model:
        def CreateFeFreeMeshRefinementVolume(self, *args):
            self.args = args
            return Refinement()

    monkeypatch.setattr(
        regions,
        "Doc",
        type("Doc", (), {"Cylinder": "cylinder", "Ellipsoid": "ellipsoid"}),
    )
    monkeypatch.setattr(regions, "Vector3D", Vec)
    monkeypatch.setattr(
        regions,
        "FeFreeMeshRefinementVolume",
        type("RefinementEnum", (), {"MM": 1, "Volume": 2, "Surface": 3}),
    )
    monkeypatch.setattr(regions, "REFINEMENT_PRIMITIVE", "cylinder")
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_MM", 0.35)
    monkeypatch.setattr(regions, "REFINEMENT_PADDING_RADIUS_FACTOR", 0.0)
    monkeypatch.setattr(regions, "REFINEMENT_AXIAL_OVERLAP_RADIUS_FACTOR", 1.0)
    model = Model()
    segment = {
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([0.0, 0.0, 1.0]),
        "length": 1.0,
        "radius": 0.2,
        "spline_name": "edge",
        "segment_index": 0,
        "strahler_order": 1,
    }

    regions._create_refinement(model, ["part"], segment, 1)

    assert model.args[0] == "cylinder"
    # Simpleware scales are full extents: diameter, diameter, total length.
    assert model.args[2].values == pytest.approx((1.10, 1.10, 2.10))

    # A sphere is represented by Simpleware's ellipsoid primitive with three
    # identical full extents; any chord-axis orientation is therefore harmless.
    monkeypatch.setattr(regions, "REFINEMENT_PRIMITIVE", "sphere")
    segment["region_radius"] = 0.30
    segment["axial_half_length"] = 0.80
    regions._create_refinement(model, ["part"], segment, 2)
    assert model.args[0] == "ellipsoid"
    assert model.args[2].values == pytest.approx((0.60, 0.60, 0.60))
