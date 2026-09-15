"""Collapse-aware geodesic reconnection: repair the graph and the mask together.

The rest of :mod:`..` repairs one or the other. The geometric proposers add a
centreline segment and leave the segmentation with a hole in it; the DPC walk
reads the image but still only writes a graph; ``segmentation.paint_bridges``
writes voxels but only along a centreline something else already decided. A run
that uses them produces a graph and a mask that disagree, and nothing downstream
can tell which one is right.

This package treats the two as one repair, and it is built around three facts
about this data that the existing tools do not encode:

**The specimen is collapsed.** Ex-vivo HiP-CT coronaries are slits and ribbons,
not tubes. A vesselness filter tuned for tubes scores them near zero and a
capsule-painting repair re-inflates them into an anatomy that is not there. The
evidence terms in :mod:`.cost` score tube, ribbon and sheet alike, and
:mod:`.shape` transports the observed cross-section across the gap instead of
assuming a circular one.

**Most "disconnections" are not mask gaps.** Two free ends are very often already
inside one mask component -- the lumen is continuous and only the centreline broke.
:mod:`.classify` separates those, and they are repaired by re-deriving the
centreline, inventing no voxels at all. Running a path search on them would produce
a plausible-looking route through material that never needed one.

**The graph does not know where all the vessels are.** Pruning and thinning leave
lumen in the mask that no centreline reaches, and a break whose far side is one of
those has no endpoint to pair with -- so the only pair the proposer can see is the
wrong one, and refusing it merely leaves the break unrepaired. :mod:`.lobes`
manufactures the missing ends from the mask itself, and they are put through the
same gates and the same evidence as any other candidate.

**Greedy is not good enough, and neither is confident.** :mod:`.astar` searches
globally with direction in the state, so a genuine dropout mid-gap costs what it
costs instead of ending the walk; and it returns alternatives, which is the only
honest basis for saying a route is ambiguous. Ambiguous routes go to an operator
rather than to a coin toss.

Nothing here applies itself. :func:`~.route.plan` decides, :mod:`.audit` records,
and :func:`~.apply.apply_plan` is a separate explicit step -- the same division the
rest of the package uses, for the same reason: a reconnection that is wrong is
worse than one that is missing.

    from hipct_seg_debug.edit.reconnect import geodesic

    index = geodesic.components.build(labels)
    plan = geodesic.plan(graph, index, frame, stack=stack)
    print(plan.summarise())
    geodesic.apply_plan(graph, plan, source, frame)
"""

from __future__ import annotations

from . import (
    apply,
    astar,
    audit,
    classify,
    components,
    corridor,
    cost,
    lobes,
    select,
    shape,
)
from .apply import (
    DPC,
    GEODESIC,
    GEOMETRY,
    ORIGINAL,
    RESKELETONISED,
    Applied,
    apply_one,
    apply_plan,
    origin_counts,
    write_segmentation,
)
from .route import Candidate, GeodesicParams, Plan, evaluate, plan, propose

__all__ = [
    # modules
    "apply", "astar", "audit", "classify", "components", "corridor", "cost",
    "lobes", "select", "shape",
    # the pipeline
    "GeodesicParams", "Candidate", "Plan", "plan", "propose", "evaluate",
    # committing
    "Applied", "apply_plan", "apply_one", "write_segmentation", "origin_counts",
    # provenance codes
    "ORIGINAL", "GEOMETRY", "DPC", "GEODESIC", "RESKELETONISED",
]
