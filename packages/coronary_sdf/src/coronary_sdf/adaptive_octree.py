"""Sparse, radius-adaptive sampling of :mod:`coronary_sdf.implicit_field`.

The hierarchy stores only cells which cannot be safely rejected using the
field's analytic Lipschitz bound.  Its resolution criterion is dimensionless:
``cell_size <= 2 * local_radius / cells_across_diameter``.

This module intentionally stops at a sampled hierarchy.  Polygonizing an
adaptive octree with ordinary marching cubes creates cracks; consumers must
use a conforming/manifold extractor (the VTK HyperTreeGrid backend or an edge-tree
polygonizer), not silently rebuild a global finest-resolution volume.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .implicit_field import FieldSample, GraphImplicitField


@dataclass(frozen=True)
class OctreeLeaf:
    center: np.ndarray
    size: float
    depth: int
    center_value: float
    local_radius: float
    corner_values: np.ndarray

    @property
    def has_corner_sign_change(self) -> bool:
        return bool(np.min(self.corner_values) <= 0.0 <= np.max(self.corner_values))


@dataclass(frozen=True)
class OctreeStatistics:
    visited_cells: int
    pruned_cells: int
    active_leaves: int
    sign_change_leaves: int
    maximum_depth: int
    field_evaluations: int


class RadiusAdaptiveOctree:
    """Build a sparse sampling hierarchy around an analytic zero level-set."""

    _CORNER_SIGNS = np.asarray(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (-1, 1, -1),
            (1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, 1, 1),
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        field: GraphImplicitField,
        *,
        cells_across_diameter: float = 12.0,
        padding_radius_factor: float = 2.0,
        maximum_depth: int = 30,
        maximum_active_leaves: int = 5_000_000,
    ) -> None:
        if cells_across_diameter < math.sqrt(3.0):
            raise ValueError(
                "cells_across_diameter must be at least sqrt(3) for reliable sampling"
            )
        if padding_radius_factor <= 0.0:
            raise ValueError("padding_radius_factor must be positive")
        if maximum_depth < 1:
            raise ValueError("maximum_depth must be positive")
        if maximum_active_leaves < 1:
            raise ValueError("maximum_active_leaves must be positive")
        self.field = field
        self.cells_across_diameter = float(cells_across_diameter)
        self.padding_radius_factor = float(padding_radius_factor)
        self.maximum_depth = int(maximum_depth)
        self.maximum_active_leaves = int(maximum_active_leaves)
        self.leaves: list[OctreeLeaf] = []
        self.statistics: OctreeStatistics | None = None

        pad = self.padding_radius_factor * float(np.max(field.max_radii))
        lower = field.bounds_min - pad
        upper = field.bounds_max + pad
        self.root_center = 0.5 * (lower + upper)
        self.root_size = float(np.max(upper - lower))
        if not math.isfinite(self.root_size) or self.root_size <= 0.0:
            raise ValueError("implicit-field bounds are degenerate")

    @staticmethod
    def _children(center: np.ndarray, size: float) -> np.ndarray:
        return center[None, :] + RadiusAdaptiveOctree._CORNER_SIGNS * (0.25 * size)

    @staticmethod
    def _corners(center: np.ndarray, size: float) -> np.ndarray:
        return center[None, :] + RadiusAdaptiveOctree._CORNER_SIGNS * (0.5 * size)

    def _target_size(self, center: np.ndarray, half_diagonal: float) -> float:
        radius = self.field.minimum_relevant_radius(center, half_diagonal)
        return 2.0 * radius / self.cells_across_diameter

    def build(self) -> OctreeStatistics:
        """Construct the hierarchy and return immutable build statistics."""

        self.leaves.clear()
        visited = 0
        pruned = 0
        field_evaluations = 0
        deepest = 0
        frontier_centers = self.root_center.reshape(1, 3)
        frontier_size = self.root_size
        depth = 0

        while len(frontier_centers):
            deepest = max(deepest, depth)
            visited += len(frontier_centers)
            center_values, _owners, center_radii, _gradients = self.field.evaluate(
                frontier_centers
            )
            field_evaluations += len(frontier_centers)
            half_diagonal = 0.5 * math.sqrt(3.0) * frontier_size
            uncertain = (
                np.abs(center_values)
                <= self.field.lipschitz_bound * half_diagonal
            )
            pruned += int(np.count_nonzero(~uncertain))

            next_centers: list[np.ndarray] = []
            leaf_centers: list[np.ndarray] = []
            leaf_values: list[float] = []
            leaf_radii: list[float] = []
            for index in np.flatnonzero(uncertain):
                center = frontier_centers[index]
                target_size = self._target_size(center, half_diagonal)
                if frontier_size > target_size and depth < self.maximum_depth:
                    next_centers.extend(self._children(center, frontier_size))
                else:
                    leaf_centers.append(center)
                    leaf_values.append(float(center_values[index]))
                    leaf_radii.append(float(center_radii[index]))

            if leaf_centers:
                leaf_center_array = np.asarray(leaf_centers, dtype=np.float64)
                corner_points = (
                    leaf_center_array[:, None, :]
                    + self._CORNER_SIGNS[None, :, :] * (0.5 * frontier_size)
                )
                flat_corner_values, _owners, _radii, _gradients = self.field.evaluate(
                    corner_points.reshape(-1, 3)
                )
                field_evaluations += len(flat_corner_values)
                corner_values = flat_corner_values.reshape(-1, 8)
                for index, center in enumerate(leaf_center_array):
                    self.leaves.append(
                        OctreeLeaf(
                            center=center.copy(),
                            size=float(frontier_size),
                            depth=int(depth),
                            center_value=leaf_values[index],
                            local_radius=leaf_radii[index],
                            corner_values=corner_values[index].copy(),
                        )
                    )
                if len(self.leaves) > self.maximum_active_leaves:
                    raise MemoryError(
                        "adaptive implicit sampler exceeded maximum_active_leaves; "
                        "reduce cells_across_diameter or split the graph by component"
                    )

            if not next_centers:
                break
            frontier_centers = np.asarray(next_centers, dtype=np.float64)
            frontier_size *= 0.5
            depth += 1

        sign_changes = sum(leaf.has_corner_sign_change for leaf in self.leaves)
        self.statistics = OctreeStatistics(
            visited_cells=visited,
            pruned_cells=pruned,
            active_leaves=len(self.leaves),
            sign_change_leaves=int(sign_changes),
            maximum_depth=deepest,
            field_evaluations=field_evaluations,
        )
        return self.statistics

    def leaf_arrays(self) -> dict[str, np.ndarray]:
        """Return a compact array representation suitable for diagnostics/I/O."""

        if not self.leaves:
            return {
                "centers": np.empty((0, 3), dtype=np.float64),
                "sizes": np.empty(0, dtype=np.float64),
                "depths": np.empty(0, dtype=np.int16),
                "center_values": np.empty(0, dtype=np.float64),
                "local_radii": np.empty(0, dtype=np.float64),
                "corner_values": np.empty((0, 8), dtype=np.float64),
            }
        return {
            "centers": np.asarray([leaf.center for leaf in self.leaves]),
            "sizes": np.asarray([leaf.size for leaf in self.leaves]),
            "depths": np.asarray([leaf.depth for leaf in self.leaves], dtype=np.int16),
            "center_values": np.asarray([leaf.center_value for leaf in self.leaves]),
            "local_radii": np.asarray([leaf.local_radius for leaf in self.leaves]),
            "corner_values": np.asarray([leaf.corner_values for leaf in self.leaves]),
        }


__all__ = ["OctreeLeaf", "OctreeStatistics", "RadiusAdaptiveOctree"]
