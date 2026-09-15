"""Turn candidate pairs into decided routes.

The pipeline, and why it is in this order:

1. **Propose geometrically.** The existing cone / reach / radius / tortuosity gates
   in :mod:`..endpoints` and :mod:`..tjunction` are cheap, tuned on this data, and
   already generate the right pairs. Nothing here replaces them; this module only
   replaces what happens *after* a pair is proposed.
2. **Associate and classify.** Which mask component is each end on? That single
   question splits the work four ways (:mod:`.classify`) and removes most of it --
   a break whose two ends share a mask component never needs a path search.
3. **Build a corridor and price it** (:mod:`.corridor`, :mod:`.cost`), calibrated
   on this candidate's own intact ends.
4. **Search** (:mod:`.astar`), for the best route and up to two alternatives.
5. **Gate**, on evidence rather than on geometry -- the geometry has already had
   its say in step 1.
6. **Decide globally** (:mod:`.select`), because one candidate cannot see the
   endpoint it is competing for.

Steps 1-5 are per candidate and embarrassingly parallel; step 6 is not, and is why
the module returns a whole :class:`Plan` rather than yielding decisions as it goes.

Three ways a route dies here, and they are deliberately different outcomes:

``reject``   the evidence says no -- it crosses unsupported material, or the
             existing geometry gates refuse it. Nothing is written.
``review``   the evidence is *equivocal*: a second route is nearly as good, or the
             association was ambiguous, or the corridor had no raw greyscale to
             justify a gap this long. Written to the review file with its
             alternatives, and applied only if an operator says so.
``accept``   the evidence is clear and nothing else contests the endpoint.

The middle one is the reason this package exists. A heuristic that must answer yes
or no on every case will be confidently wrong on the hard ones, and the hard ones
are where a false connection reroutes flow in the CFD run downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..candidates import TORTUOSITY_MAX
from . import (
    astar,
    classify,
    corridor as corridor_mod,
    cost as cost_mod,
    lobes,
    select,
    shape,
)

#: A second route within this fraction of the best cost makes the choice ambiguous.
AMBIGUITY_MARGIN = 0.15
#: Contiguous unsupported path is capped at this many local radii.
MAX_UNSUPPORTED_FACTOR = 4.0
#: Without raw greyscale, only gaps this many segmentation voxels wide may be
#: closed automatically. Anything longer is a claim the mask alone cannot support.
MASK_ONLY_GAP_VOXELS = 2
#: How many alternatives to look for, by default.
ALTERNATIVES = 3


@dataclass
class GeodesicParams:
    """Everything tunable, in one place, so the CLI and the GUI configure the same thing."""

    alternatives: int = ALTERNATIVES
    ambiguity_margin: float = AMBIGUITY_MARGIN
    max_unsupported_factor: float = MAX_UNSUPPORTED_FACTOR
    mask_only_gap_voxels: int = MASK_ONLY_GAP_VOXELS
    unsupported_fraction: float = cost_mod.UNSUPPORTED_FRACTION
    turn_weight: float = astar.TURN_WEIGHT
    #: Re-applied to the **route**, not just to the proposal. The geometric gates run
    #: on a synthetic Hermite curve; A* then returns a different path entirely and
    #: nothing re-checked it. Measured on LADAF-2024-28: three of eleven accepted
    #: routes exceeded this, at 1.86, 2.30 and 2.92.
    tortuosity_max: float = TORTUOSITY_MAX
    redundancy_weight: float = cost_mod.REDUNDANCY_WEIGHT
    pad_factor: float = 6.0
    dark_lumen: bool = True
    min_confidence: float = 0.45
    review_confidence: float = 0.25
    allow_cycles: bool = False
    fragment_reach_radii: float = 8.0
    parent_target_um: float = 0.0  # 0 => 4 x source radius
    #: Manufacture candidate endpoints from mask lumen the graph never described
    #: (:mod:`.lobes`). Without it a break whose far side was pruned away has no
    #: target to propose, and the only pair visible is the wrong one.
    mask_endpoints: bool = True
    describe_radii: float = lobes.DESCRIBE_RADII
    lobe_search_radii: float = lobes.SEARCH_RADII
    min_lobe_voxels: int = lobes.MIN_LOBE_VOXELS
    #: A mask-end break inside one mask component needs no route -- but re-deriving
    #: the centreline is done in *replace* mode, which deletes the segments inside
    #: its box first. Past this span, in local radii, that box is large enough that
    #: the deletion deserves a human rather than a default.
    mask_end_reskeletonise_radii: float = 8.0


@dataclass
class Candidate:
    """One repair, from proposal through to decision."""

    classified: classify.Classified
    proposal: object = None  # the originating Bridge, for provenance
    route: astar.Route | None = None
    alternatives: list = field(default_factory=list)
    completion: shape.Completion | None = None
    waypoints: list = field(default_factory=list)
    confidence: float = 0.0
    status: str = "reject"  # "accept" | "review" | "reject"
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.classified.kind

    @property
    def source_node(self) -> int:
        return self.classified.source.node

    @property
    def target_node(self):
        target = self.classified.target
        return None if target is None or target.node < 0 else target.node

    def path_um(self, frame) -> np.ndarray:
        if self.route is None or not len(self.route.path_zyx):
            return np.empty((0, 3))
        return self.route.path_um(frame)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        mask_end = self.classified.target_mask_end
        if self.target_node is not None:
            target = self.target_node
        elif mask_end is not None:
            # Not "seg None": a mask end has neither a node nor a segment, and
            # printing the absence of one tells a reader nothing about which end
            # this is.
            target = "mask end " + "/".join(str(v) for v in mask_end.key)
        else:
            target = f"seg {self.classified.target_segment}"
        return (f"<Candidate {self.kind} {self.source_node} -> {target} "
                f"{self.status} conf={self.confidence:.3f}: {self.reason}>")


@dataclass
class Plan:
    """Every candidate, decided, plus what the run as a whole found."""

    candidates: list
    decisions: list
    associations: dict
    fragments: list
    stats: dict = field(default_factory=dict)
    mask_ends: list = field(default_factory=list)
    lobe_report: lobes.LobeReport | None = None

    def accepted(self) -> list:
        return [d.candidate for d in self.decisions if d.accepted]

    def for_review(self) -> list:
        return [d.candidate for d in self.decisions if d.status == "review"]

    def rejected(self) -> list:
        return [d.candidate for d in self.decisions if d.status == "reject"]

    def summarise(self) -> str:
        lines = [select.summarise(self.decisions)]
        by_kind: dict[str, int] = {}
        for decision in self.decisions:
            if decision.accepted:
                kind = decision.candidate.kind
                by_kind[kind] = by_kind.get(kind, 0) + 1
        for kind, n in sorted(by_kind.items()):
            lines.append(f"  accepted {kind:<14} {n}")
        for candidate in self.for_review():
            lines.append(f"  review: {candidate!r}")
        if self.stats.get("fragments_plausible"):
            lines.append(
                f"  {self.stats['fragments_plausible']} unskeletonised mask "
                f"fragment(s) near a break "
                f"({self.stats.get('fragments_debris', 0)} excluded as debris)"
            )
        if self.lobe_report is not None and self.lobe_report.anchors:
            lines.append(
                f"  {self.stats.get('mask_ends', 0)} mask free end(s) with no "
                f"centreline, giving {self.stats.get('mask_end_proposals', 0)} "
                f"extra proposal(s)"
            )
            # Most mask ends are lumen running *on* past a free end rather than
            # back at one, so a small number of proposals is the expected answer
            # and not a gate misfiring. Saying which gate refused what is the only
            # way to tell those two apart from the outside.
            refusals = {k[len("rejected_"):]: v
                        for k, v in (self.stats.get("mask_end_gates") or {}).items()
                        if k.startswith("rejected_") and v}
            if refusals:
                lines.append("    refused: " + ", ".join(
                    f"{v} {k.replace('_', ' ')}"
                    for k, v in sorted(refusals.items(), key=lambda kv: -kv[1])
                ))
        return "\n".join(lines)


# --------------------------------------------------------------------- proposal


def propose(graph, *, same_component: bool = False, tjunction: bool = True,
            gate_kwargs=None) -> list:
    """Candidate pairs from the existing geometry gates, both kinds.

    Deliberately reuses :mod:`..endpoints` and :mod:`..tjunction` rather than
    reimplementing the search. Their thresholds were tuned on this data and their
    KD-tree pass is what keeps candidate generation linear in the number of nearby
    pairs; what this package changes is the *decision*, not the shortlist.
    """
    from .. import endpoints, tjunction as tj

    kwargs = dict(gate_kwargs or {})
    out = list(endpoints.propose(graph, same_component=same_component,
                                 keep_rejected=False, **kwargs))
    if tjunction:
        out.extend(tj.propose(graph, same_component=same_component,
                              keep_rejected=False, **kwargs))
    return [b for b in out if b.accepted]


# ------------------------------------------------------------------ one candidate


def evaluate(candidate: Candidate, index, frame, *, stack=None,
             params: GeodesicParams | None = None, graph=None) -> Candidate:
    """Search and gate one classified candidate. Mutates and returns it."""
    p = params or GeodesicParams()
    kind = candidate.kind

    if kind == "unassociated":
        candidate.status = "review"
        candidate.reason = candidate.classified.reason
        return candidate

    if kind == "reskeletonise":
        return _evaluate_reskeletonise(candidate, index, frame, p)

    source = candidate.classified.source
    target = candidate.classified.target
    radius = max(float(source.radius_um), float(target.radius_um))
    allowed = {source.component, target.component}
    allowed.update(f.component for f in candidate.classified.fragments if f.plausible)

    anchor_points = [source.point_um, target.point_um]
    anchor_points.extend(f.centroid_um for f in candidate.classified.fragments
                         if f.plausible)
    if candidate.waypoints:
        anchor_points.extend(np.asarray(w, dtype=np.float64) for w in candidate.waypoints)

    try:
        box = corridor_mod.for_candidate(
            frame, index, np.asarray(anchor_points, dtype=np.float64),
            radius_um=radius, stack=stack, pad_factor=p.pad_factor,
        )
    except corridor_mod.CorridorTooLarge as exc:
        candidate.status = "reject"
        candidate.reason = f"the corridor would be too large to build ({exc})"
        return candidate

    tails = _calibration_tails(graph, candidate, box)
    field_ = cost_mod.build(
        box, index, allowed, radius_um=radius, calibration_points=tails,
        centreline_points=_centreline_in(graph, box),
        redundancy_weight=p.redundancy_weight, dark_lumen=p.dark_lumen,
    )
    candidate.evidence["corridor"] = box.describe()
    candidate.evidence["cost_field"] = field_.describe()
    candidate.evidence["has_raw"] = bool(box.has_raw)
    candidate.evidence["competing_components"] = dict(field_.competing)

    start = box.to_global(source.point_um[None, :])[0]
    goals = _goal_set(candidate, box, graph, radius)
    if not len(goals):
        candidate.status = "reject"
        candidate.reason = "the target does not fall inside the corridor"
        return candidate

    start = _nudge_into_field(field_, start)
    waypoints = [box.to_global(np.asarray(w, dtype=np.float64)[None, :])[0]
                 for w in candidate.waypoints]
    found = astar.routes(
        field_, start, goals, start_direction=_step_direction(source.tangent, box),
        alternatives=p.alternatives, turn_weight=p.turn_weight,
        waypoints=waypoints or None,
    )
    if not found or not len(found[0].path_zyx):
        candidate.status = "reject"
        candidate.reason = (found[0].reason if found else "no route was found")
        return candidate

    candidate.route = found[0]
    candidate.alternatives = found[1:]
    _gate(candidate, field_, box, p, radius)
    if candidate.status != "reject":
        _complete_shape(candidate, index, frame, box, p)
    return candidate


def _evaluate_reskeletonise(candidate, index, frame, p) -> Candidate:
    """A break whose lumen is already continuous: no path, no new voxels.

    The confidence is high and it is *not* a claim about the image -- it is the
    observation that the mask already connects these two ends, so the repair adds
    nothing and can only redistribute centreline inside foreground that was
    already there. That is about as safe as an automatic edit gets, which is why
    this is the one kind that does not need raw greyscale.
    """
    source = candidate.classified.source
    target = candidate.classified.target
    span = float(np.linalg.norm(target.point_um - source.point_um))
    candidate.confidence = 0.9
    candidate.status = "accept"
    candidate.reason = candidate.classified.reason
    candidate.evidence.update(
        component=int(source.component), span_um=span,
        adds_voxels=False,
        note="re-skeletonised inside existing foreground; no voxel is invented",
    )

    # The safety argument above is about *voxels*, and it still holds for a mask
    # end. What does not carry over is the scale: re-deriving in replace mode
    # deletes the segments inside its box first, and a mask end can be a long way
    # off, making that box large enough to take out centreline nobody was asking
    # about. The claim stays the same; the blast radius does not, so past a few
    # radii it goes to a person.
    if candidate.classified.target_mask_end is not None:
        allowance = p.mask_end_reskeletonise_radii * max(float(source.radius_um), 1.0)
        candidate.confidence = 0.75
        candidate.evidence["mask_end"] = list(candidate.classified.target_mask_end.key)
        if span > allowance:
            candidate.status = "review"
            candidate.confidence = 0.4
            candidate.reason = (
                f"the lumen is continuous, but re-deriving it would clear a "
                f"{span:.0f} um box (above {allowance:.0f} um) of existing centreline"
            )
    return candidate


def _goal_set(candidate, box, graph, radius_um) -> np.ndarray:
    """Where the search is allowed to finish.

    A single voxel for an end-to-end join. For a T-junction it is a *stretch* of
    the parent vessel around the attachment point -- the paper's type 3 behaviour
    -- because a branch does not arrive at one predetermined voxel of its parent
    and forcing it to would bend the last part of the route to hit a target the
    evidence never chose.

    A mask end gets the same treatment for the same reason, over the tip end of its
    traced axis. Its tip is one voxel of a thinning, not a landmark, and demanding
    that exact voxel would let a one-voxel error in the medial axis decide the shape
    of the last few hundred micrometres of the repair.
    """
    classified = candidate.classified
    if classified.target_mask_end is not None:
        end = classified.target_mask_end
        axis = np.asarray(end.skeleton_um, dtype=np.float64).reshape(-1, 3)
        tip = classified.target.point_um
        near = axis[np.linalg.norm(axis - tip, axis=1) <= 2.0 * radius_um] \
            if len(axis) else axis
        keep = np.vstack([tip[None, :], near]) if len(near) else tip[None, :]
        goals = np.unique(box.to_global(keep), axis=0)
        return goals[box.contains(goals)]
    if classified.target_segment is None or graph is None:
        point = classified.target.point_um[None, :]
        goals = box.to_global(point)
    else:
        coords = np.asarray(graph.coords(classified.target_segment), dtype=np.float64)
        anchor = classified.target.point_um
        keep = coords[np.linalg.norm(coords - anchor, axis=1) <= max(4.0 * radius_um,
                                                                    2.0 * radius_um)]
        if not len(keep):
            keep = anchor[None, :]
        goals = np.unique(box.to_global(keep), axis=0)
    inside = box.contains(goals)
    return goals[inside]


def _step_direction(tangent, box) -> np.ndarray:
    """A tangent in world um expressed as a direction on the voxel grid."""
    spacing_zyx = np.asarray(box.spacing_um, dtype=np.float64)[::-1]
    return np.asarray(tangent, dtype=np.float64)[::-1] / np.maximum(spacing_zyx, 1e-9)


def _nudge_into_field(field_, start) -> np.ndarray:
    """Move a start voxel off blocked material, if association put it there.

    The endpoint was associated to a component within a voxel or two, so the voxel
    it rounds to can belong to a *neighbour*. Refusing outright would turn a
    rounding artefact into a rejection, and snapping silently would hide a real
    conflict -- so the search starts at the nearest allowed voxel and the offset is
    recorded in the route's metrics.
    """
    local = field_.to_local(start)
    if np.all(local >= 0) and np.all(local < np.asarray(field_.cost.shape)) \
            and np.isfinite(field_.cost[tuple(local)]):
        return start
    free = np.argwhere(np.isfinite(field_.cost))
    if not len(free):
        return start
    nearest = free[int(np.argmin(np.linalg.norm(
        (free - local) * field_.spacing_zyx, axis=1)))]
    return field_.to_global(nearest)


# ----------------------------------------------------------------------- gating


def _gate(candidate: Candidate, field_, box, p: GeodesicParams, radius_um: float) -> None:
    """Accept, send to review, or reject -- on the image evidence.

    The order matters. Hard failures first, so a route that crosses an unrelated
    vessel is *rejected* rather than sent to an operator as an ambiguous choice;
    then the equivocal cases, which are the ones a person can actually adjudicate.
    """
    route = candidate.route
    spacing = field_.spacing_zyx
    unsupported = route.unsupported_um(spacing, p.unsupported_fraction)
    allowance = p.max_unsupported_factor * radius_um
    margin = _margin(route, candidate.alternatives)

    span_now = float(np.linalg.norm(
        candidate.classified.target.point_um - candidate.classified.source.point_um
    ))
    route_tortuosity = (route.length_um / span_now) if span_now > 1e-9 else float("inf")
    in_described = _described_fraction(route, field_)

    candidate.evidence.update(
        route_cost=float(route.cost),
        route_tortuosity=float(route_tortuosity),
        described_fraction=float(in_described),
        mean_support=route.mean_support,
        min_support=route.min_support,
        length_um=route.length_um,
        unsupported_um=float(unsupported),
        unsupported_allowance_um=float(allowance),
        alternative_margin=None if margin is None else float(margin),
        expanded=int(route.expanded),
        support_reference=float(field_.calibration.support_reference),
        contrast=bool(field_.calibration.separated),
    )

    span = float(np.linalg.norm(
        candidate.classified.target.point_um - candidate.classified.source.point_um
    ))
    candidate.evidence["span_um"] = span

    # -- hard rejections -------------------------------------------------------
    # The geometric gates ran on the Hermite proposal; this is the same question asked
    # of the path that will actually be written. A route may be three times the length
    # of the gap it spans and still be perfectly supported, because it spent that
    # length inside a real vessel -- which is exactly the failure the tortuosity gate
    # exists to catch and exactly what it was missing by being applied too early.
    if route_tortuosity > p.tortuosity_max:
        candidate.status = "reject"
        candidate.reason = (
            f"the route wanders: {route.length_um:.0f} um across a {span_now:.0f} um "
            f"gap (tortuosity {route_tortuosity:.2f}, above {p.tortuosity_max:g})"
        )
        return
    if unsupported > allowance:
        candidate.status = "reject"
        candidate.reason = (
            f"{unsupported:.0f} um of contiguous unsupported path, above the "
            f"{p.max_unsupported_factor:g} x radius allowance ({allowance:.0f} um)"
        )
        return
    crossed = _crosses_unrelated(route, field_)
    if crossed:
        candidate.status = "reject"
        candidate.reason = f"the route passes through unrelated component {crossed}"
        return

    # -- the confidence, and the two ways of being unsure -----------------------
    candidate.confidence = _confidence(route, margin, field_, p)
    gap = _mask_gap_voxels(route, field_)
    candidate.evidence["mask_gap_voxels"] = int(gap)

    if not box.has_raw and gap > p.mask_only_gap_voxels:
        candidate.status = "review"
        candidate.reason = (
            f"a {gap}-voxel mask gap needs raw image support; without --raw this "
            f"route is geometry alone"
        )
        return
    if margin is not None and margin < p.ambiguity_margin:
        candidate.status = "review"
        candidate.reason = (
            f"a second route is within {100 * margin:.0f}% of the best cost"
        )
        return
    if candidate.confidence < p.review_confidence:
        candidate.status = "reject"
        candidate.reason = f"confidence {candidate.confidence:.2f} is too low to propose"
        return
    if candidate.confidence < p.min_confidence:
        candidate.status = "review"
        candidate.reason = f"confidence {candidate.confidence:.2f} is below the accept bar"
        return

    candidate.status = "accept"
    candidate.reason = (
        f"supported route, {route.length_um:.0f} um, mean support "
        f"{route.mean_support:.2f} of the intact vessel"
    )


def _described_fraction(route, field_) -> float:
    """How much of the route runs through lumen the graph already describes.

    Reported whether or not it triggers anything: a repair that is mostly a second
    centreline along an existing vessel is a duplicate, and the number says so even
    when the tortuosity gate lets it through.
    """
    local = route.path_zyx - field_.lo_zyx
    shape_ = np.asarray(field_.cost.shape)
    keep = np.all((local >= 0) & (local < shape_), axis=1)
    if not keep.any():
        return 0.0
    described = field_.described[local[keep, 0], local[keep, 1], local[keep, 2]]
    return float(np.mean(described > 0.5))


def _margin(route, alternatives) -> float | None:
    """How much worse the next distinct route is, as a fraction. ``None`` if unique."""
    if not alternatives:
        suppressed = route.metrics.get("suppressed_cost")
        if suppressed is None or not np.isfinite(suppressed):
            return None
        # The only corridor through: its cost under suppression is not a rival.
        return float("inf")
    best = max(float(route.cost), 1e-9)
    return float((float(alternatives[0].cost) - best) / best)


def _confidence(route, margin, field_, p: GeodesicParams) -> float:
    """A [0, 1] score combining support, uniqueness and the calibration's quality.

    Not a probability and not presented as one. It orders candidates for the
    global forest pass and sets the accept / review bar; the numbers that justify
    a decision are the ones in ``evidence``, which is what the review file carries.

    **A term that was never measured is dropped, not scored as mediocre.** With
    ``--alternatives 1`` the search is explicitly told not to look for a rival, so
    folding in a neutral 0.5 for uniqueness would quietly push every candidate
    below the accept bar -- turning a performance flag into "send everything to
    review", which is a coupling nobody asked for and nothing would report.
    Renormalising over the terms actually available keeps the flag meaning what it
    says.
    """
    # Support is already in units of the intact vessel, so 1.0 is full marks and
    # scoring better than the vessel being completed is not extra evidence -- it
    # usually means the calibration tail clipped something. Clipping there rather
    # than rescaling keeps the term's meaning the same as its definition.
    support = float(np.clip(route.mean_support, 0.0, 1.0))
    trough = float(np.clip(route.min_support, 0.0, 1.0))
    # A corridor with no measurable lumen/wall contrast has had to lean entirely on
    # geometry, and should not produce a confident automatic accept.
    contrast = 1.0 if field_.calibration.separated else 0.6

    if p.alternatives <= 1:
        return float(np.clip(
            (0.4 * support + 0.2 * trough) / 0.6 * contrast, 0.0, 1.0
        ))

    if margin is None:
        uniqueness = 0.5  # asked for, and the answer did not come back
    elif not np.isfinite(margin):
        uniqueness = 1.0  # the only corridor through
    else:
        uniqueness = float(np.clip(margin / max(p.ambiguity_margin * 3.0, 1e-9), 0.0, 1.0))
    return float(np.clip(
        (0.4 * support + 0.2 * trough + 0.4 * uniqueness) * contrast, 0.0, 1.0
    ))


def _crosses_unrelated(route, field_) -> int:
    """Which unrelated component the route entered, if any. 0 when it entered none.

    Blocked voxels are infinite cost so the search cannot enter one, and this is
    the assertion that says so. It exists because that guarantee is the difference
    between a repair and a false connection, and it should fail loudly if a future
    change to the cost field ever softens the block into a penalty.
    """
    local = route.path_zyx - field_.lo_zyx
    shape_ = np.asarray(field_.cost.shape)
    keep = np.all((local >= 0) & (local < shape_), axis=1)
    if not keep.any():
        return 0
    hit = field_.blocked[local[keep, 0], local[keep, 1], local[keep, 2]]
    if not hit.any():
        return 0
    first = local[keep][hit][0]
    return int(field_.labels[first[0], first[1], first[2]])


def _mask_gap_voxels(route, field_) -> int:
    """The longest run of route voxels that are not already foreground.

    This is the size of the claim the repair is making. Zero means the mask was
    always continuous and the route only re-describes it; two means a dropout of
    the kind a morphological closing would mend; fifty means a vessel is being
    invented, and had better have raw greyscale behind it.
    """
    local = route.path_zyx - field_.lo_zyx
    shape_ = np.asarray(field_.cost.shape)
    keep = np.all((local >= 0) & (local < shape_), axis=1)
    if not keep.any():
        return 0
    on = field_.mine[local[keep, 0], local[keep, 1], local[keep, 2]]
    best = run = 0
    for value in on:
        run = 0 if value else run + 1
        best = max(best, run)
    return int(best)


def _centreline_in(graph, box) -> np.ndarray:
    """Every existing centreline point inside the corridor.

    What the graph already describes, which the cost field charges for travelling
    along. All of it, not just this candidate's own vessels: a route riding the trunk
    is redundant whether or not the trunk is one of its endpoints' components.
    """
    if graph is None or not graph.points:
        return np.empty((0, 3))
    points = np.asarray([p[:3] for p in graph.points.values()], dtype=np.float64)
    return points[box.contains(box.to_global(points))]


def _calibration_tails(graph, candidate, box) -> np.ndarray:
    """Points on the intact vessel either side of the break, for calibration.

    Both tails, not one. Calibrating on the source alone biases every measurement
    toward whichever end happened to be listed first, and the two ends of a break
    routinely differ -- that is often *why* they were skeletonised apart.

    A mask end has no node to walk back from, so it carries its own tail: the axis
    :mod:`.lobes` traced through the lumen behind it. That is the same object a
    graph tail is -- centreline inside intact vessel -- and leaving it out would
    quietly restore the one-sided calibration this function exists to avoid.
    """
    from ..dpc import _near_node_points

    blocks = []
    for association in (candidate.classified.source, candidate.classified.target):
        if association is None:
            continue
        if association.node < 0:
            tail = association.tail_points_um
            if tail is not None and len(tail):
                blocks.append(np.asarray(tail, dtype=np.float64))
            continue
        if graph is None:
            continue
        points = _near_node_points(graph, association.node, 12)
        if len(points):
            blocks.append(np.asarray(points, dtype=np.float64))
    if not blocks:
        return np.empty((0, 3))
    points = np.vstack(blocks)
    inside = box.contains(box.to_global(points))
    return points[inside]


def _complete_shape(candidate, index, frame, box, p) -> None:
    """Measure both ends and transport their cross-sections along the accepted route."""
    classified = candidate.classified
    path = candidate.route.path_um(frame)
    if len(path) < 2:
        return
    try:
        section_a, section_b = shape.measure_ends(
            index, frame, classified.source, classified.target
        )
        candidate.completion = shape.transport(
            path, section_a, section_b, frame,
            min_radius_um=float(classified.source.radius_um),
        )
    except Exception as exc:  # noqa: BLE001 - a failed cut must not lose the route
        candidate.completion = None
        candidate.evidence["completion_error"] = str(exc)
        return
    candidate.evidence["completion"] = candidate.completion.describe()
    candidate.evidence["flatness"] = [
        section_a.flatness if section_a.valid else None,
        section_b.flatness if section_b.valid else None,
    ]


# -------------------------------------------------------------------- the driver


def plan(graph, index, frame, *, stack=None, params: GeodesicParams | None = None,
         same_component: bool = False, tjunction: bool = True, gate_kwargs=None,
         proposals=None, progress=None, lobe_progress=None) -> Plan:
    """Propose, classify, search, gate and globally select, in one call.

    Passing `proposals` explicitly means "these pairs and no others", so the mask
    end sweep is skipped there: a caller that hand-picked its candidates does not
    want the planner adding more.
    """
    p = params or GeodesicParams()
    stats: dict = {}

    bridges = list(proposals) if proposals is not None else propose(
        graph, same_component=same_component, tjunction=tjunction,
        gate_kwargs=gate_kwargs,
    )
    stats["proposals"] = len(bridges)

    associations = classify.associate(index, frame, graph)
    stats["endpoints_associated"] = sum(1 for a in associations.values() if a.associated)
    stats["endpoints_unassociated"] = len(associations) - stats["endpoints_associated"]

    reach = p.fragment_reach_radii * float(np.median(
        [a.radius_um for a in associations.values()] or [50.0]
    ))
    anchors = np.array([a.point_um for a in associations.values()]) \
        if associations else None
    fragments = classify.fragment_candidates(index, frame, graph, reach_um=reach,
                                             near_points_um=anchors)
    stats["fragments_plausible"] = sum(1 for f in fragments if f.plausible)
    stats["fragments_debris"] = sum(1 for f in fragments if not f.plausible)

    # Manufacture the ends the skeletoniser did not leave behind, and propose to
    # them. This runs *after* the proposals above rather than instead of them: an
    # end-to-end pair and a mask-end pair for the same free end are both put to the
    # evidence, and `select` decides between them on confidence. Suppressing one in
    # favour of the other here would be deciding it on which proposer ran first.
    mask_ends: list = []
    lobe_report = None
    if p.mask_endpoints and proposals is None:
        mask_ends, lobe_report = lobes.find(
            index, frame, graph, associations,
            search_radii=p.lobe_search_radii, describe_radii=p.describe_radii,
            min_voxels=p.min_lobe_voxels, progress=lobe_progress,
        )
        lobe_stats: dict = {}
        extra = lobes.propose(graph, mask_ends, associations, stats=lobe_stats,
                              **_lobe_gate_kwargs(gate_kwargs))
        stats["mask_ends"] = len(mask_ends)
        stats["mask_end_proposals"] = len(extra)
        stats["mask_end_gates"] = lobe_stats
        bridges = list(bridges) + extra

    candidates: list[Candidate] = []
    for n, bridge in enumerate(bridges, 1):
        classified = classify.classify(bridge, associations, index, frame,
                                       fragments=fragments)
        candidate = Candidate(classified=classified, proposal=bridge)
        evaluate(candidate, index, frame, stack=stack, params=p, graph=graph)
        candidates.append(candidate)
        if progress is not None:
            progress(n, len(bridges), candidate)

    scored = [(c, c.confidence, c.status, c.reason) for c in candidates]
    decisions = select.select(graph, scored, allow_cycles=p.allow_cycles)
    return Plan(candidates=candidates, decisions=decisions,
                associations=associations, fragments=fragments, stats=stats,
                mask_ends=mask_ends, lobe_report=lobe_report)


def _lobe_gate_kwargs(gate_kwargs) -> dict:
    """The geometry gates the mask-end proposer shares with the others.

    Only the four it understands. ``gate_kwargs`` also carries flags meant for
    :mod:`..endpoints` alone, and forwarding those would be a ``TypeError`` the
    first time anyone passed one.
    """
    keys = ("cone_angle_deg", "cone_length_factor", "radius_ratio_max",
            "tortuosity_max")
    return {k: v for k, v in dict(gate_kwargs or {}).items() if k in keys}
