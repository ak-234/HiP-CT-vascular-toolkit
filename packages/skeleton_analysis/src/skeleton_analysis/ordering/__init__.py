"""Strahler ordering and topological-generation numbering."""

from skeleton_analysis.ordering.strahler import strahler_order
from skeleton_analysis.ordering.topological import topological_generations
from skeleton_analysis.ordering.pipeline import (
    run_ordering,
    order_forest,
    auto_roots,
    OrderingResult,
)
from skeleton_analysis.ordering.root_picker import (
    pick_roots,
    tree_components,
    root_from_edge,
    strahler_edge_colors,
)

__all__ = [
    "strahler_order",
    "topological_generations",
    "run_ordering",
    "order_forest",
    "auto_roots",
    "OrderingResult",
    "pick_roots",
    "tree_components",
    "root_from_edge",
    "strahler_edge_colors",
]
