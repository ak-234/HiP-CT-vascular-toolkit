"""Tensor-guided wavefront reconnection: large gaps, collapsed vessels.

The sibling :mod:`..geodesic` package repairs a break by pricing a corridor with
one number per voxel and running an orientation-aware A\\* through it. That is the
right design for a gap a few radii long. It is not enough for the long ones --
a vessel that vanishes for a millimetre and reappears, a pinched slit whose
mask has nothing across it, a continuation the skeletoniser never reached -- and
those are the gaps that leave a coronary tree in pieces. Three things change here:

**The graph is cleaned first.** Every seed direction and every proposal cone
starts at a free end, and a free end's direction on a raw skeleton is noise.
:mod:`.prepare` runs the segmentation-constrained centreline refinement over the
whole graph, remeasures the radii, and then profiles each free end: a fitted
tangent and the principal axes of its collapsed cross-section.

**The cost has a direction.** :mod:`.tensor` builds an image intensity tensor --
structure tensor for orientation, Hessian for planarity -- and a Riemannian metric
on top of the calibrated scalar cost, so a step along the vessel is cheap, across
the ribbon's width dearer, and through the collapsed wall dearest.

**The search is a wave, not a path.** :mod:`.propagate` sweeps a front over the
corridor and returns a map; :mod:`.bridge` asks it where two fronts meet, or
marches it forward keypoint by keypoint with a look-ahead cone along the vessel
axis sized by the ellipse's major axis, so a mid-gap dropout costs what it costs
rather than ending the search.

Everything downstream -- classification, the evidence gates, global selection,
the transactional apply with cross-section transport, the audit files -- is the
sibling package's, reused as is. Enable with ``edit connect --wavefront``.

    from hipct_seg_debug.edit.reconnect import wavefront

    index = geodesic.components.build(labels)
    plan = wavefront.plan(graph, index, frame, labels, stack=stack)
    print(plan.summarise())
    geodesic.apply_plan(graph, plan, source, frame, origin=geodesic.WAVEFRONT)
"""

from __future__ import annotations

from . import bridge, prepare, propagate, route, tensor
from .prepare import EndProfile, clean, profile_ends
from .propagate import Front, agd_available
from .route import WavefrontParams, evaluate, plan
from .tensor import AnisotropyParams, TensorField

__all__ = [
    "bridge", "prepare", "propagate", "route", "tensor",
    "WavefrontParams", "AnisotropyParams", "TensorField", "EndProfile", "Front",
    "plan", "evaluate", "clean", "profile_ends", "agd_available",
]
