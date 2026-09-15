"""Vascular metrics: branching angles, Murray's law, intervessel distance,
model-II regression, and the radius-scaling exponent.

Ports of the ``Metrics/`` folder (branching_angles_with_strahler.m,
murray_law.m, intervessel_distance.m, Exponent_calculation.m) plus the
third-party ``gmregress.m`` / ``gmregresspi.m`` reduced-major-axis regressions.
"""

from skeleton_analysis.metrics.regression import (
    GMRegressResult,
    gmregress,
    gmregresspi,
)
from skeleton_analysis.metrics.branching_angles import branching_ang, branching_angles
from skeleton_analysis.metrics.murray import find_effective_gamma, murray_law
from skeleton_analysis.metrics.intervessel import edge_midpoints, intervessel_distance
from skeleton_analysis.metrics.exponent import exponent_calculation
from skeleton_analysis.metrics.radius import mean_radius_per_edge
from skeleton_analysis.metrics.aggregate import (
    aggregate_by_strahler,
    branching_ratio,
    violin_by_strahler,
)
from skeleton_analysis.metrics import geometry
from skeleton_analysis.metrics.report import (
    edge_metrics_table,
    murray_table,
    assign_kmeans,
    plot_report,
    write_metric_graph,
    compare_states,
)

__all__ = [
    "GMRegressResult",
    "gmregress",
    "gmregresspi",
    "branching_ang",
    "branching_angles",
    "find_effective_gamma",
    "murray_law",
    "edge_midpoints",
    "intervessel_distance",
    "exponent_calculation",
    "mean_radius_per_edge",
    "aggregate_by_strahler",
    "branching_ratio",
    "violin_by_strahler",
    "geometry",
    "edge_metrics_table",
    "murray_table",
    "assign_kmeans",
    "plot_report",
    "write_metric_graph",
    "compare_states",
]
