"""Murray's law analysis at branch points.

Port of the ``Metrics/murray_law.m`` variant. At each branch point we gather the
parent vessel radius and the sum of child radii (and their cubes, for the
classic ``r_parent^3 == sum(r_child^3)`` test), tagged by Strahler order, and
solve for the effective radius-scaling exponent gamma (``findEffectiveGamma.m``).

Fixes vs MATLAB
---------------
* ``findEffectiveGamma`` is called with the **scalar** parent radius for the
  current branch (the MATLAB code accidentally passed the growing ``parent_rad``
  vector).
* Interactive root selection is replaced by the ``root_id`` argument.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from skeleton_analysis.graph.build import reorient_edges, resolve_root
from skeleton_analysis.graph.neighbors import coordination_number
from skeleton_analysis.io.amira import SpatialGraph


def find_effective_gamma(r_parent: float, r_daughters, guess: float = 2.5) -> float:
    """Solve ``r_parent**gamma == sum(r_daughters**gamma)`` for gamma.

    Port of ``findEffectiveGamma.m`` (``fsolve`` -> :func:`scipy.optimize.fsolve`,
    initial guess 2.5).
    """
    r_daughters = np.asarray(r_daughters, dtype=float)

    def equation(gamma):
        return r_parent ** gamma - np.sum(r_daughters ** gamma)

    sol, _info, ier, _msg = fsolve(equation, guess, full_output=True)
    gamma = float(sol[0])
    return gamma if ier == 1 else float("nan")


def murray_law(
    graph: SpatialGraph,
    root_id: Optional[int] = None,
    radius_field: str = "MeanRadius",
    strahler_field: str = "strahler",
    compute_gamma: bool = True,
) -> pd.DataFrame:
    """Murray's-law quantities at every branch point.

    Requires per-edge ``radius_field`` (e.g. ``MeanRadius``); ``strahler_field``
    is optional. Returns a DataFrame with one row per branch point and columns:
    ``node, parent_rad, parent_rad_cubed, sumchild, sumchild_cubed, strahler,
    gamma_eff``.
    """
    edges0 = np.asarray(graph.edge_connectivity, dtype=np.int64)
    root = resolve_root(edges0, root_id)
    edges, _flipped = reorient_edges(edges0, root)

    if radius_field not in graph.edge_fields:
        raise KeyError(
            f"Edge field {radius_field!r} not present. Compute it first "
            f"(see metrics.mean_radius_per_edge) or pass radius_field=..."
        )
    radius = np.asarray(graph.edge_fields[radius_field], dtype=float)
    strahler = graph.edge_fields.get(strahler_field)
    coord = coordination_number(edges)

    rows = []
    for node in np.unique(edges):
        if coord[node] < 3:
            continue
        child_idx = np.flatnonzero(edges[:, 1] == node)  # edges into node
        parent_idx = np.flatnonzero(edges[:, 0] == node)  # edge leaving node
        if parent_idx.size == 0:
            continue  # root branch point has no parent vessel
        p = int(parent_idx[0])
        parent_rad = float(radius[p])
        children = radius[child_idx]

        gamma = (
            find_effective_gamma(parent_rad, children) if compute_gamma else float("nan")
        )
        rows.append(
            {
                "node": int(node),
                "parent_rad": parent_rad,
                "parent_rad_cubed": parent_rad ** 3,
                "sumchild": float(np.sum(children)),
                "sumchild_cubed": float(np.sum(children ** 3)),
                "strahler": int(strahler[p]) if strahler is not None else np.nan,
                "gamma_eff": gamma,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "node",
            "parent_rad",
            "parent_rad_cubed",
            "sumchild",
            "sumchild_cubed",
            "strahler",
            "gamma_eff",
        ],
    )
