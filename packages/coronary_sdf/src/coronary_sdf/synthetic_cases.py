"""Parameterized synthetic vascular graphs used by the benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    nodes: dict[int, tuple]
    points: dict[int, tuple]
    segments: list[dict]
    expected_components: int
    scale: float
    category: str
    expected_valid: bool


def _case(
    name: str,
    branches: list[np.ndarray],
    radii: list[np.ndarray | float],
    *,
    expected_components: int = 1,
    scale: float = 1.0,
    category: str,
    expected_valid: bool = True,
) -> SyntheticCase:
    endpoint_nodes: dict[tuple[float, float, float], int] = {}
    node_positions: dict[int, np.ndarray] = {}
    points: dict[int, tuple] = {}
    segments: list[dict] = []
    next_node = 0
    next_point = 0
    for segment_id, (coords_raw, radius_raw) in enumerate(zip(branches, radii)):
        coords = np.asarray(coords_raw, dtype=np.float64) * scale
        radius = np.asarray(radius_raw, dtype=np.float64)
        if radius.ndim == 0:
            radius = np.full(len(coords), float(radius))
        radius = radius * scale
        node_ids = []
        for position in (coords[0], coords[-1]):
            key = tuple(np.round(position, 12))
            if key not in endpoint_nodes:
                endpoint_nodes[key] = next_node
                node_positions[next_node] = position
                next_node += 1
            node_ids.append(endpoint_nodes[key])
        ids = []
        for position, local_radius in zip(coords, radius):
            pid = next_point
            next_point += 1
            points[pid] = (
                float(position[0] * 1000.0),
                float(position[1] * 1000.0),
                float(position[2] * 1000.0),
                float(local_radius * 1000.0),
            )
            ids.append(pid)
        segments.append(
            {
                "id": segment_id,
                "node1": node_ids[0],
                "node2": node_ids[1],
                "point_ids": ids,
                "strahler": 1,
            }
        )
    degree = {node: 0 for node in node_positions}
    for segment in segments:
        degree[segment["node1"]] += 1
        degree[segment["node2"]] += 1
    nodes = {
        node: (*tuple(position * 1000.0), degree[node])
        for node, position in node_positions.items()
    }
    return SyntheticCase(
        name=name,
        nodes=nodes,
        points=points,
        segments=segments,
        expected_components=expected_components,
        scale=scale,
        category=category,
        expected_valid=expected_valid,
    )


def _line(a, b, count=25):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), count)


def synthetic_suite() -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []
    line = _line((0, 0, 0), (6, 0, 0), 31)
    cases.append(_case("straight", [line], [0.5], category="primitive"))
    cases.append(
        _case(
            "tapered",
            [line],
            [np.linspace(0.8, 0.2, len(line))],
            category="primitive",
        )
    )
    for kappa_r in (0.8, 1.0, 1.2):
        radius = 0.5
        bend_radius = radius / kappa_r
        # A 216-degree arc exercises the tight inner bend while keeping the
        # two capped ends separated for kappa*r < 1.  The former 270-degree
        # fixture made the end caps overlap at kappa*r=0.8 and introduced an
        # unrelated handle into a case declared genus zero.
        angle = np.linspace(-0.6 * math.pi, 0.6 * math.pi, 61)
        bend = np.column_stack(
            (bend_radius * np.cos(angle), bend_radius * np.sin(angle), np.zeros_like(angle))
        )
        cases.append(
            _case(
                f"torus_kappa_r_{kappa_r:.1f}",
                [bend],
                [radius],
                category="curvature",
                expected_valid=kappa_r < 1.0,
            )
        )
    hairpin = np.vstack(
        (
            _line((0, 0, 0), (4, 0, 0), 25),
            _line((4, 0.08, 0), (4, 0.8, 0), 7),
            _line((3.92, 0.8, 0), (0, 0.8, 0), 25),
        )
    )
    cases.append(
        _case(
            "hairpin_overlap",
            [hairpin],
            [0.5],
            category="clearance",
            expected_valid=False,
        )
    )
    for gap in (0.2, 0.0, -0.2):
        separation = 1.0 + gap
        branches = [
            _line((0, -separation / 2, 0), (6, -separation / 2, 0), 31),
            _line((0, separation / 2, 0), (6, separation / 2, 0), 31),
        ]
        cases.append(
            _case(
                f"parallel_clearance_{gap:+.1f}",
                branches,
                [0.5, 0.5],
                expected_components=2 if gap > 0 else 1,
                category="clearance",
                # At zero clearance the exact union has a point singularity,
                # so it is not a valid closed two-component CFD surface.
                expected_valid=gap > 0,
            )
        )

    def star(degree: int, angle_z: float = 0.15):
        branches = []
        for branch in range(degree):
            angle = 2 * math.pi * branch / degree
            direction = np.asarray(
                [math.cos(angle), math.sin(angle), angle_z * (-1) ** branch]
            )
            direction /= np.linalg.norm(direction)
            branches.append(_line((0, 0, 0), 5 * direction, 26))
        return branches

    for degree in (3, 4, 5):
        cases.append(
            _case(
                f"junction_degree_{degree}",
                star(degree),
                [0.5] * degree,
                category="junction",
            )
        )
    consecutive = [
        _line((-4, 0, 0), (0, 0, 0), 21),
        _line((0, 0, 0), (2, 1, 0), 15),
        _line((0, 0, 0), (2, -1, 0), 15),
        _line((2, 1, 0), (4, 2, 0), 15),
        _line((2, 1, 0), (4, 0.5, 0), 15),
    ]
    cases.append(
        _case(
            "consecutive_bifurcations",
            consecutive,
            [0.65, 0.45, 0.4, 0.3, 0.25],
            category="junction",
        )
    )
    for count in (9, 33, 129):
        sampled = _line((0, 0, 0), (6, 0, 0), count)
        cases.append(
            _case(
                f"sampling_density_{count}",
                [sampled],
                [0.5],
                category="invariance",
            )
        )
    rng = np.random.default_rng(20260812)
    noisy = line.copy()
    noisy[1:-1, 1:] += rng.normal(scale=0.04, size=(len(noisy) - 2, 2))
    cases.append(_case("spatial_noise", [noisy], [0.5], category="noise"))
    noisy_radius = 0.5 + rng.normal(scale=0.04, size=len(line))
    noisy_radius[[0, -1]] = 0.5
    cases.append(_case("radius_noise", [line], [noisy_radius], category="noise"))
    for scale in (0.01, 1.0, 100.0):
        cases.append(
            _case(
                f"global_scale_{scale:g}",
                star(3),
                [0.5] * 3,
                scale=scale,
                category="invariance",
            )
        )
    return cases


__all__ = ["SyntheticCase", "synthetic_suite"]
