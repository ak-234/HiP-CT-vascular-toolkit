"""Convert VesselVio CSV output into an Amira spatial graph.

Port of ``VVToAmira_v3.m``. VesselVio (via the modified ``feature_extraction.py``)
exports a ``vertices.csv`` (a coordinate string ``[z, y, x]`` + radius per row)
and an ``edges.csv`` (0-based endpoint connectivity). This builds a
:class:`~skeleton_analysis.io.amira.SpatialGraph` with one straight segment
(2 points) per edge, its endpoint radii as ``thickness``, applying the same
z<->x axis swap and resolution scaling as the MATLAB script.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from skeleton_analysis.io.amira import (
    F_EDGE_CONNECTIVITY,
    F_NUM_EDGE_POINTS,
    F_POINT_COORDS,
    F_THICKNESS,
    F_VERTEX_COORDS,
    SpatialGraph,
    write_amira,
)

PathLike = Union[str, Path]
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_coord(cell) -> np.ndarray:
    """Parse a VesselVio coordinate cell (``"[z, y, x]"`` or 3 numbers) -> [z, y, x]."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=float)
    nums = _NUM_RE.findall(str(cell))
    if len(nums) < 3:
        raise ValueError(f"Could not parse 3 coordinates from {cell!r}")
    return np.array([float(n) for n in nums[:3]], dtype=float)


def vesselvio_to_spatial_graph(
    vertices_csv: PathLike,
    edges_csv: PathLike,
    resolution: Tuple[float, float] = (50.0, 50.0),
    swap_zx: bool = True,
    coord_column: int = 0,
    radius_column: int = 1,
    edge_columns: Sequence[int] = (0, 1),
) -> SpatialGraph:
    """Build a :class:`SpatialGraph` from VesselVio ``vertices``/``edges`` CSVs.

    Parameters
    ----------
    resolution : (xy, z)
        Voxel size in x/y and z (micrometres). x, y are scaled by ``resolution[0]``,
        z by ``resolution[1]``.
    swap_zx : bool
        Swap the z and x axes so the graph displays correctly in Amira (as the
        MATLAB script does).
    coord_column, radius_column : int
        Column positions of the coordinate string and radius in ``vertices_csv``.
    edge_columns : (int, int)
        Column positions of the two endpoint IDs in ``edges_csv`` (0-based IDs).
    """
    vdf = pd.read_csv(vertices_csv)
    coords = np.vstack([_parse_coord(c) for c in vdf.iloc[:, coord_column]])  # [z, y, x]
    radii = vdf.iloc[:, radius_column].to_numpy(dtype=float)

    if swap_zx:
        coords = coords[:, [2, 1, 0]]  # -> [x, y, z]
    coords[:, 0:2] *= resolution[0]
    coords[:, 2] *= resolution[1]

    edf = pd.read_csv(edges_csv)
    conn = edf.iloc[:, list(edge_columns)].to_numpy(dtype=np.int64)  # 0-based

    n_edges = len(conn)
    # Two points per edge: its two endpoint coordinates / radii.
    point_coords = np.empty((n_edges * 2, 3), dtype=float)
    thickness = np.empty(n_edges * 2, dtype=float)
    point_coords[0::2] = coords[conn[:, 0]]
    point_coords[1::2] = coords[conn[:, 1]]
    thickness[0::2] = radii[conn[:, 0]]
    thickness[1::2] = radii[conn[:, 1]]

    g = SpatialGraph()
    g.set_vertex_field(F_VERTEX_COORDS, coords)
    g.set_edge_field(F_EDGE_CONNECTIVITY, conn)
    g.set_edge_field(F_NUM_EDGE_POINTS, np.full(n_edges, 2, dtype=np.int64))
    g.set_point_field(F_POINT_COORDS, point_coords)
    g.set_point_field(F_THICKNESS, thickness)
    return g


def vesselvio_to_amira(
    vertices_csv: PathLike,
    edges_csv: PathLike,
    output_am: PathLike,
    **kwargs,
) -> SpatialGraph:
    """Convert VesselVio CSVs straight to an Amira ``.am`` file. Returns the graph."""
    g = vesselvio_to_spatial_graph(vertices_csv, edges_csv, **kwargs)
    write_amira(g, output_am)
    return g
