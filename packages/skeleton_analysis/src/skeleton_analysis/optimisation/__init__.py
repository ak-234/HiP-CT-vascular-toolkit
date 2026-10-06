"""Skeletonisation quality / optimisation metrics.

* :mod:`supermetric` - **the** skeleton super metric of Walsh et al. (2024),
  Eq. 10, with option presets (``PAPER``, ``CORONARY``). Use this one.
* :mod:`meta_metric` - bifurcation-matching Dice between a candidate skeleton
  and a binary ground-truth graph, plus the legacy combined "meta metric"
  (deprecated; a port of the original MATLAB, not the paper's metric).
* :mod:`cl_dice` - centreline Dice / sensitivity between image volumes (a
  cleaned, parameterised version of the original ``cl_dice.py``; requires the
  ``[image]`` extra).
"""

from skeleton_analysis.optimisation.meta_metric import (
    BifurcationDice,
    bifurcation_dice,
    bifurcation_dice_points,
    bifurcation_points,
    meta_metric,
)
from skeleton_analysis.optimisation.volume_metrics import (
    centreline_sensitivity,
    skeleton_junction_points,
    region_props_table,
    region_morphometrics,
    super_metric,
)

from skeleton_analysis.optimisation.supermetric import (
    CORONARY,
    PAPER,
    SkeletonGraph,
    SuperMetricOptions,
    reference_metrics,
)

__all__ = [
    "CORONARY",
    "PAPER",
    "SkeletonGraph",
    "SuperMetricOptions",
    "reference_metrics",
    "BifurcationDice",
    "bifurcation_dice",
    "bifurcation_dice_points",
    "bifurcation_points",
    "meta_metric",
    "centreline_sensitivity",
    "skeleton_junction_points",
    "region_props_table",
    "region_morphometrics",
    "super_metric",
]
