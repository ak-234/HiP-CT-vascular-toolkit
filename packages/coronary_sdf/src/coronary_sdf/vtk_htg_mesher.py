"""Compiler-free adaptive contouring with VTK HyperTreeGrid.

The Python layer builds a complete radius-adaptive octree and samples the
shared :class:`~coronary_sdf.implicit_field.GraphImplicitField` oracle.  The
installed VTK wheel performs only the final dual-grid contour operation; no
project-specific native extension or local compiler is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkVersion
from vtkmodules.vtkCommonDataModel import (
    vtkHyperTreeGrid,
    vtkHyperTreeGridNonOrientedCursor,
)
from vtkmodules.vtkFiltersHyperTree import vtkHyperTreeGridContour

from .adaptive_octree import RadiusAdaptiveOctree
from .implicit_field import GraphImplicitField


@dataclass(frozen=True)
class VtkHtgStatistics:
    visited_cells: int
    pruned_cells: int
    leaf_cells: int
    refined_cells: int
    maximum_depth: int
    field_evaluations: int
    hierarchy_seconds: float
    contour_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _coordinates(values: np.ndarray) -> vtkDoubleArray:
    result = vtkDoubleArray()
    result.SetNumberOfValues(len(values))
    for index, value in enumerate(values):
        result.SetValue(index, float(value))
    return result


def _build_sampled_tree(
    field: GraphImplicitField,
    *,
    cells_across_diameter: float,
    padding_radius_factor: float,
    maximum_depth: int,
    maximum_cells: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
    int,
    int,
]:
    """Return BFS node values/children and root geometry.

    ``child_start[i]`` is the first of eight consecutive logical children, or
    ``-1`` for a leaf. Keeping this as compact NumPy arrays avoids a Python
    object per octree cell while still allowing deterministic cursor replay.
    """

    if cells_across_diameter < math.sqrt(3.0):
        raise ValueError("cells_across_diameter must be at least sqrt(3)")
    if padding_radius_factor <= 0.0:
        raise ValueError("padding_radius_factor must be positive")
    if maximum_depth < 1 or maximum_cells < 1:
        raise ValueError("VTK HTG depth/cell limits must be positive")

    padding = padding_radius_factor * float(np.max(field.max_radii))
    lower = field.bounds_min - padding
    upper = field.bounds_max + padding
    root_center = 0.5 * (lower + upper)
    root_size = float(np.max(upper - lower))
    if not math.isfinite(root_size) or root_size <= 0.0:
        raise ValueError("implicit-field bounds are degenerate")
    lower = root_center - 0.5 * root_size

    value_levels: list[np.ndarray] = []
    child_levels: list[np.ndarray] = []
    frontier_centers = root_center.reshape(1, 3)
    frontier_size = root_size
    next_global_offset = 1
    visited = pruned = refined = 0
    deepest = 0

    for depth in range(maximum_depth + 1):
        if not len(frontier_centers):
            break
        deepest = depth
        visited += len(frontier_centers)
        if visited > maximum_cells:
            raise MemoryError(
                f"VTK HTG hierarchy requires more than {maximum_cells:,} cells; "
                "reduce cells_across_diameter or split the graph"
            )
        values, _owners, _radii, _gradients = field.evaluate(frontier_centers)
        values = np.asarray(values, dtype=np.float64)
        half_diagonal = 0.5 * math.sqrt(3.0) * frontier_size
        # A field may supply its own exclusion certificate. Fields that do not
        # (including GraphImplicitField) keep the historical Lipschitz test
        # exactly, so this hook cannot change existing candidates.
        certainly_outside = getattr(field, "certainly_outside_band", None)
        if certainly_outside is None:
            uncertain = np.abs(values) <= field.lipschitz_bound * half_diagonal
        else:
            uncertain = ~np.asarray(
                certainly_outside(frontier_centers, half_diagonal), dtype=bool
            )
        subdivide = np.zeros(len(frontier_centers), dtype=bool)
        if depth < maximum_depth:
            for index in np.flatnonzero(uncertain):
                local_radius = field.minimum_relevant_radius(
                    frontier_centers[index], half_diagonal
                )
                target = 2.0 * local_radius / cells_across_diameter
                subdivide[index] = frontier_size > target

        child_start = np.full(len(frontier_centers), -1, dtype=np.int64)
        parent_indices = np.flatnonzero(subdivide)
        if len(parent_indices):
            prospective_cells = visited + 8 * len(parent_indices)
            if prospective_cells > maximum_cells:
                raise MemoryError(
                    "VTK HTG hierarchy would require at least "
                    f"{prospective_cells:,} cells, exceeding "
                    f"maximum_cells={maximum_cells:,}"
                )
            child_start[parent_indices] = next_global_offset + 8 * np.arange(
                len(parent_indices), dtype=np.int64
            )
            child_centers = (
                frontier_centers[parent_indices, None, :]
                + RadiusAdaptiveOctree._CORNER_SIGNS[None, :, :]
                * (0.25 * frontier_size)
            ).reshape(-1, 3)
        else:
            child_centers = np.empty((0, 3), dtype=np.float64)

        value_levels.append(values)
        child_levels.append(child_start)
        refined += len(parent_indices)
        pruned += int(np.count_nonzero(~uncertain))
        next_global_offset += len(child_centers)
        frontier_centers = child_centers
        frontier_size *= 0.5

    node_values = np.concatenate(value_levels)
    child_starts = np.concatenate(child_levels)
    leaf_count = int(np.count_nonzero(child_starts < 0))
    return (
        node_values,
        child_starts,
        lower,
        root_size,
        visited,
        pruned,
        deepest,
    )


def _to_vtk_hyper_tree_grid(
    node_values: np.ndarray,
    child_starts: np.ndarray,
    lower: np.ndarray,
    root_size: float,
) -> vtkHyperTreeGrid:
    grid = vtkHyperTreeGrid()
    grid.SetBranchFactor(2)
    grid.SetDimensions(2, 2, 2)
    grid.SetXCoordinates(_coordinates(np.asarray([lower[0], lower[0] + root_size])))
    grid.SetYCoordinates(_coordinates(np.asarray([lower[1], lower[1] + root_size])))
    grid.SetZCoordinates(_coordinates(np.asarray([lower[2], lower[2] + root_size])))

    cursor = vtkHyperTreeGridNonOrientedCursor()
    grid.InitializeNonOrientedCursor(cursor, 0, True)
    cursor.SetGlobalIndexStart(0)
    vtk_values = vtkDoubleArray()
    vtk_values.SetName("implicit_value")
    vtk_values.SetNumberOfTuples(len(node_values))

    def replay(logical_index: int) -> None:
        vtk_index = int(cursor.GetGlobalNodeIndex())
        if vtk_index < 0 or vtk_index >= len(node_values):
            raise RuntimeError("VTK assigned an invalid HyperTreeGrid node index")
        vtk_values.SetValue(vtk_index, float(node_values[logical_index]))
        child_start = int(child_starts[logical_index])
        if child_start < 0:
            return
        cursor.SubdivideLeaf()
        for child in range(8):
            cursor.ToChild(child)
            replay(child_start + child)
            cursor.ToParent()

    replay(0)
    if grid.GetNumberOfCells() != len(node_values):
        raise RuntimeError(
            "VTK HyperTreeGrid cell count does not match sampled hierarchy "
            f"({grid.GetNumberOfCells()} != {len(node_values)})"
        )
    grid.GetCellData().SetScalars(vtk_values)
    return grid


def mesh_vtk_hyper_tree_grid(
    field: GraphImplicitField,
    *,
    cells_across_diameter: float = 12.0,
    padding_radius_factor: float = 2.0,
    maximum_depth: int = 30,
    maximum_cells: int = 5_000_000,
    decomposed_polyhedra: bool = False,
) -> tuple[pv.PolyData, VtkHtgStatistics]:
    """Contour ``field == 0`` on an adaptive VTK HyperTreeGrid."""

    hierarchy_started = time.perf_counter()
    (
        values,
        children,
        lower,
        root_size,
        visited,
        pruned,
        deepest,
    ) = _build_sampled_tree(
        field,
        cells_across_diameter=cells_across_diameter,
        padding_radius_factor=padding_radius_factor,
        maximum_depth=maximum_depth,
        maximum_cells=maximum_cells,
    )
    grid = _to_vtk_hyper_tree_grid(values, children, lower, root_size)
    hierarchy_seconds = time.perf_counter() - hierarchy_started

    contour_started = time.perf_counter()
    contour = vtkHyperTreeGridContour()
    contour.SetInputData(grid)
    contour.SetValue(0, 0.0)
    contour.SetStrategy3D(
        contour.USE_DECOMPOSED_POLYHEDRA
        if decomposed_polyhedra
        else contour.USE_VOXELS
    )
    contour.Update()
    output = contour.GetOutput()
    if output is None or output.GetNumberOfPoints() == 0:
        raise RuntimeError("VTK HyperTreeGrid contour returned an empty surface")
    surface = pv.wrap(output).extract_surface().triangulate().clean(tolerance=0.0)
    if not surface.n_points or not surface.n_cells:
        raise RuntimeError("VTK HyperTreeGrid contour returned no triangles")
    surface.compute_normals(
        cell_normals=True,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=True,
    )
    contour_seconds = time.perf_counter() - contour_started
    statistics = VtkHtgStatistics(
        visited_cells=visited,
        pruned_cells=pruned,
        leaf_cells=int(np.count_nonzero(children < 0)),
        refined_cells=int(np.count_nonzero(children >= 0)),
        maximum_depth=deepest,
        field_evaluations=visited,
        hierarchy_seconds=hierarchy_seconds,
        contour_seconds=contour_seconds,
    )
    for name, value in statistics.to_dict().items():
        surface.field_data[f"vtk_htg_{name}"] = np.asarray([value])
    surface.field_data["vtk_htg_vtk_version"] = np.asarray(
        [vtkVersion.GetVTKVersion()]
    )
    return surface, statistics


__all__ = ["VtkHtgStatistics", "mesh_vtk_hyper_tree_grid"]
