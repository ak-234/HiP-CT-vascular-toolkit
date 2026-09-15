"""Apply corrected thickness values and write the corrected spatial graph.

Ports ``replace_thickness_vals.m``, ``write_corrected_data.m`` and the manual
plane-selection step ``within_range.m``. Because a
:class:`~skeleton_analysis.io.amira.SpatialGraph` already carries the full graph,
correcting radii is just updating the ``thickness`` point field and writing the
graph back with :func:`skeleton_analysis.io.amira.write_amira`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np

from skeleton_analysis.io.amira import SpatialGraph, write_amira

PathLike = Union[str, Path]


def replace_thickness_values(
    graph: SpatialGraph,
    point_indices: Sequence[int],
    values: Sequence[float],
    thickness_field: str = "thickness",
    in_place: bool = True,
) -> np.ndarray:
    """Overwrite thickness at the given (0-based) point indices.

    Port of ``replace_thickness_vals.m`` (which keyed on Amira ``PointID``; here
    the point index is the position in the flat point array). Returns the updated
    thickness array. With ``in_place=True`` the graph's field is updated too.
    """
    point_indices = np.asarray(point_indices, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if point_indices.shape != values.shape:
        raise ValueError("point_indices and values must have the same length")

    thickness = np.asarray(graph.point_fields[thickness_field], dtype=float).copy()
    thickness[point_indices] = values
    if in_place:
        graph.set_point_field(thickness_field, thickness)
    return thickness


def write_corrected_graph(graph: SpatialGraph, output_path: PathLike) -> None:
    """Write a spatial graph whose thickness has been corrected.

    Thin wrapper over :func:`write_amira` (kept for parity with the MATLAB API).
    """
    write_amira(graph, output_path)


def apply_manual_plane_selection(
    segment_radii: Dict[int, np.ndarray],
    plane_table,
    segment_col: str = "genx_segment_no",
) -> Dict[int, np.ndarray]:
    """Snap auto-computed oblique radii to a manually-verified plane table.

    Faithful port of ``within_range.m`` (the human-in-the-loop QC step). Each row
    of ``plane_table`` names an outlier segment (``segment_col``) plus a set of
    approved "plane" values in the remaining columns:

    * a first value of NaN  -> that segment is **not** corrected (set to NaN);
    * a value that is a clean plane index (``round(v*100) % 10 == 0``) -> use the
      auto radius at that 1-based plane index;
    * any other value -> a literal approved radius.

    Each point's auto radius is then snapped to the nearest approved value. A
    single approved value is expanded to ``[v, 1.05v, 0.95v]`` (as in the MATLAB).
    Takes the table as an argument (fixing the ``data_path``/``datapath`` bug in
    the original). Returns a new ``{segment_id: radii}`` dict.

    Note: this replicates a fragile heuristic — it assumes literal approved radii
    are never clean integers/one-decimals, which holds for micrometre radii but
    not in general.
    """
    out: Dict[int, np.ndarray] = {k: np.asarray(v, dtype=float).copy()
                                  for k, v in segment_radii.items()}
    plane_cols = [c for c in plane_table.columns if c != segment_col]

    for _, row in plane_table.iterrows():
        seg_id = int(row[segment_col])
        planes = np.asarray(row[plane_cols].to_numpy(), dtype=float)

        if planes.size == 0 or np.isnan(planes[0]):
            out[seg_id] = np.array([np.nan])  # do not correct
            continue
        planes = planes[~np.isnan(planes)]
        if seg_id not in out:
            continue
        auto = np.asarray(out[seg_id], dtype=float)

        vals = []
        for v in planes:
            if round(v * 100) % 10 == 0:  # clean -> treat as a 1-based plane index
                idx = int(round(v)) - 1
                vals.append(auto[idx] if 0 <= idx < len(auto) else v)
            else:  # literal approved radius
                vals.append(v)
        vals = np.asarray(vals, dtype=float)
        if vals.size == 1:
            vals = np.array([vals[0], vals[0] * 1.05, vals[0] * 0.95])

        out[seg_id] = np.array([vals[np.argmin(np.abs(vals - r))] for r in auto])

    return out
