"""Clean, profile, propose, price, propagate, gate, select.

The same shape as :mod:`..geodesic.route`, and it returns the same
:class:`~..geodesic.route.Plan` of the same :class:`~..geodesic.route.Candidate`
objects, so the selector, the apply step, the audit files and the review panel
are all shared. What differs is in front of and inside the search:

1. **Clean** the graph (:func:`~.prepare.clean`) and **profile** its free ends
   (:func:`~.prepare.profile_ends`) so the tangents the proposers cone on are
   measured against the segmentation rather than differenced from two voxels.
2. **Propose** with the existing geometric gates but a far longer reach -- forty
   radii rather than fifteen -- because the search behind them can now afford it.
   Optionally **explore** from free ends nothing was proposed for, which is how a
   continuation the skeletoniser never reached becomes a candidate at all.
3. **Price** the corridor as before, then give it a direction (:mod:`.tensor`).
4. **Propagate**: dual fronts for a known far side, a keypoint chain when they
   fail or when the far side is unknown (:mod:`.bridge`).
5. **Gate** on the sibling's evidence gates *and* on the tensor: a route that
   runs across the local vessel orientation, or through collapsed walls, is
   refused whatever its scalar support said.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geodesic import classify, corridor as corridor_mod, cost as cost_mod, lobes, select
from ..geodesic import route as geo
from . import bridge as bridge_mod
from . import prepare, propagate as prop
from . import tensor as tensor_mod
from .tensor import AnisotropyParams

#: Proposal reach, in endpoint radii. The sibling uses 15; a wave can go further.
REACH_RADII = 40.0
#: Contiguous unsupported path allowed, in radii. Doubled from the sibling's 4:
#: the anisotropic metric and the chain both refuse wall crossings on their own,
#: so a longer dropout *along* the vessel is the case this package exists for.
MAX_UNSUPPORTED_FACTOR = 8.0
#: Keypoint step, in units of the source end's ellipse major half-axis.
KEYPOINT_STEP_MAJOR = 4.0
#: Mean |t . nu_1| over coherent voxels below which a route is refused.
MIN_ALIGNMENT = 0.5
#: Fraction of planar steps crossing the collapse normal above which it is refused.
MAX_NORMAL_CROSSING = 0.3
#: A corridor forty radii long is big; the tensor field costs ~44 bytes a voxel.
CORRIDOR_MAX_VOXELS = 12_000_000


@dataclass
class WavefrontParams(geo.GeodesicParams):
    """Everything tunable. Inherits the sibling's fields and their meaning."""

    reach_radii: float = REACH_RADII
    max_unsupported_factor: float = MAX_UNSUPPORTED_FACTOR
    pad_factor: float = 4.0  # the corridor is long, not wide
    keypoint_step_major: float = KEYPOINT_STEP_MAJOR
    lookahead_cone_deg: float = bridge_mod.LOOKAHEAD_CONE_DEG
    lookahead_cone_weight: float = bridge_mod.LOOKAHEAD_CONE_WEIGHT
    min_keypoint_support: float = bridge_mod.MIN_KEYPOINT_SUPPORT
    anisotropy: AnisotropyParams = field(default_factory=AnisotropyParams)
    stencil: int = 26
    engine: str = "auto"
    refine_method: str = "centroid-coherent"
    refine_strength: float = 0.1
    refine_max_iterations: int = 25
    refine_workers: int = 1
    chain_fallback: bool = True
    explore_open_ends: bool = False
    min_alignment: float = MIN_ALIGNMENT
    max_normal_crossing: float = MAX_NORMAL_CROSSING
    corridor_max_voxels: int = CORRIDOR_MAX_VOXELS


# ------------------------------------------------------------------ helpers


def _local_direction(tangent_xyz, box) -> np.ndarray | None:
    """A world tangent as a unit direction on the local ``(z, y, x)`` grid."""
    if tangent_xyz is None:
        return None
    d = geo._step_direction(tangent_xyz, box)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 and np.isfinite(n) else None


def _profiles_for(candidate, profiles) -> list:
    out = []
    for association in (candidate.classified.source, candidate.classified.target):
        if association is None or association.node < 0:
            continue
        p = (profiles or {}).get(association.node)
        if p is not None:
            out.append(p)
    return out


def _major_um(candidate, profiles, radius_um: float) -> float:
    p = (profiles or {}).get(candidate.classified.source.node)
    if p is not None and p.major_axis is not None and p.major_um > 0:
        return float(p.major_um)
    return float(radius_um)


def _tensor_gate(candidate, field_, p: WavefrontParams) -> None:
    """Refuse a route that disagrees with the tensor it was searched on.

    Runs after the sibling's evidence gates and only ever tightens the outcome.
    *Alignment* is the mean cosine between the route's steps and the local axis
    where the axis is well determined; *normal crossing* the fraction of steps in
    planar voxels that go through the collapse normal. A forced bridge in a chain
    is a claim the image supported only after being asked leadingly, so it always
    goes to review rather than straight to accept.
    """
    route = candidate.route
    local = route.path_zyx - field_.lo_zyx
    shape_ = np.asarray(field_.cost.shape)
    keep = np.all((local >= 0) & (local < shape_), axis=1)
    local = local[keep]
    if len(local) < 2:
        return
    steps = np.diff(local, axis=0).astype(np.float64) * field_.spacing_zyx
    norms = np.linalg.norm(steps, axis=1)
    ok = norms > 1e-9
    t = steps[ok] / norms[ok][:, None]
    at = local[1:][ok]
    axis = field_.axis[at[:, 0], at[:, 1], at[:, 2]].astype(np.float64)
    normal = field_.normal[at[:, 0], at[:, 1], at[:, 2]].astype(np.float64)
    planar = field_.planarity[at[:, 0], at[:, 1], at[:, 2]]
    axis_conf = field_.axis_confidence[at[:, 0], at[:, 1], at[:, 2]]
    ratio = tensor_mod.anisotropy_ratio(field_.aniso[at[:, 0], at[:, 1], at[:, 2]])

    # Only where the metric is actually anisotropic, which is where the tensor has
    # committed to a direction. The cut faces either side of a mask gap carry a
    # strong, coherent gradient *across* the gap and no lumen signature, and asking
    # them which way the vessel runs gets the wrong answer with great confidence;
    # the metric there is close to isotropic, so they are left out. Alignment
    # additionally needs the in-plane axis to be *determined*: on a flat sheet the
    # metric is planar-isotropic on purpose and its "axis" is whichever in-plane
    # direction won a coin toss, so only the normal crossing is judged there.
    shaped = ratio > 2.0
    oriented = shaped & (axis_conf > 0.3)
    alignment = float(np.mean(np.abs(np.sum(t[oriented] * axis[oriented], axis=1)))) \
        if oriented.any() else None
    sheet = shaped & (planar > 0.1)
    crossing = float(np.mean(np.abs(np.sum(t[sheet] * normal[sheet], axis=1)) > 0.7)) \
        if sheet.any() else None
    candidate.evidence.update(
        alignment=alignment, normal_crossing=crossing,
        anisotropic_fraction=float(np.mean(shaped)),
        oriented_fraction=float(np.mean(oriented)), planar_fraction=float(np.mean(sheet)),
    )
    if candidate.status == "reject":
        return
    if alignment is not None and alignment < p.min_alignment:
        candidate.status = "reject"
        candidate.reason = (
            f"the route cuts across the local vessel orientation (alignment "
            f"{alignment:.2f}, below {p.min_alignment:g})"
        )
        return
    if crossing is not None and crossing > p.max_normal_crossing:
        candidate.status = "reject"
        candidate.reason = (
            f"{100 * crossing:.0f}% of the route's planar steps pass through collapsed "
            f"walls (above {100 * p.max_normal_crossing:.0f}%)"
        )
        return
    forced = int(route.metrics.get("forced_bridges", 0))
    if forced and candidate.status == "accept":
        candidate.status = "review"
        candidate.reason = (
            f"the chain forced {forced} bridge(s) across unsupported gap; "
            f"{candidate.reason}"
        )


# ---------------------------------------------------------------- one candidate


def evaluate(candidate, index, frame, *, stack=None, params=None, graph=None,
             profiles=None):
    """Search and gate one classified candidate. Mutates and returns it."""
    p = params or WavefrontParams()
    kind = candidate.kind
    if kind in ("unassociated", "reskeletonise"):
        return geo.evaluate(candidate, index, frame, stack=stack, params=p, graph=graph)

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
            max_voxels=p.corridor_max_voxels,
        )
    except corridor_mod.CorridorTooLarge as exc:
        candidate.status = "reject"
        candidate.reason = f"the corridor would be too large to build ({exc})"
        return candidate

    tails = geo._calibration_tails(graph, candidate, box)
    scalar = cost_mod.build(
        box, index, allowed, radius_um=radius, calibration_points=tails,
        centreline_points=geo._centreline_in(graph, box),
        redundancy_weight=p.redundancy_weight, dark_lumen=p.dark_lumen,
    )
    ends = _profiles_for(candidate, profiles)
    field_ = tensor_mod.build(scalar, box, radius_um=radius, profiles=ends,
                              params=p.anisotropy, dark_lumen=p.dark_lumen)
    candidate.evidence["corridor"] = box.describe()
    candidate.evidence["cost_field"] = field_.describe()
    candidate.evidence["has_raw"] = bool(box.has_raw)
    candidate.evidence["competing_components"] = dict(scalar.competing)
    candidate.evidence["end_profiles"] = [e.describe() for e in ends]

    start = box.to_global(source.point_um[None, :])[0]
    goals = geo._goal_set(candidate, box, graph, radius)
    if not len(goals):
        candidate.status = "reject"
        candidate.reason = "the target does not fall inside the corridor"
        return candidate
    start = geo._nudge_into_field(field_, start)
    t_source = _local_direction(source.tangent, box)
    t_target = _local_direction(target.tangent, box) if target is not None else None

    found = bridge_mod.dual_front(
        field_, start, goals, source_tangent=t_source, target_tangent=t_target,
        alternatives=p.alternatives, engine=p.engine, stencil=p.stencil,
    )
    strategy = "dual_front"
    route = found[0] if found else None
    weak = (route is None or not len(route.path_zyx)
            or route.unsupported_um(field_.spacing_zyx, p.unsupported_fraction)
            > p.max_unsupported_factor * radius)
    if weak and p.chain_fallback and t_source is not None:
        chained = bridge_mod.chain(
            field_, start, t_source, goals,
            step_um=p.keypoint_step_major * _major_um(candidate, profiles, radius),
            max_length_um=p.reach_radii * radius, cone_deg=p.lookahead_cone_deg,
            cone_weight=p.lookahead_cone_weight,
            min_keypoint_support=p.min_keypoint_support,
            engine=p.engine, stencil=p.stencil,
        )
        candidate.evidence["chain"] = dict(
            stop_reason=chained.metrics.get("stop_reason"),
            keypoints=len(chained.metrics.get("keypoints_zyx", [])),
            forced_bridges=chained.metrics.get("forced_bridges", 0),
            length_um=chained.metrics.get("chain_length_um"),
        )
        if chained.reason == "found" and len(chained.path_zyx):
            found = [chained]
            route = chained
            strategy = "chain"
    candidate.evidence["strategy"] = strategy

    if route is None or not len(route.path_zyx):
        candidate.status = "reject"
        candidate.reason = route.reason if route is not None else "no route was found"
        return candidate

    candidate.route = route
    candidate.alternatives = found[1:]
    geo._gate(candidate, field_, box, p, radius)
    _tensor_gate(candidate, field_, p)
    if candidate.status != "reject":
        geo._complete_shape(candidate, index, frame, box, p)
    return candidate


# -------------------------------------------------------------- exploration


def explore(graph, index, frame, associations, profiles, skip_nodes, *, stack=None,
            params=None, progress=None) -> tuple[list, dict]:
    """Bridges from free ends nothing was proposed for, found by chaining.

    A free end with no pair within reach is not necessarily the end of a vessel;
    it is more often the end of what the skeletoniser reached. The chain is run
    from the tip with its own component allowed and every other component *as a
    goal* -- its foreground stays blocked, but the finite voxels touching it stop
    the front -- and whatever it reaches is turned into an ordinary proposal to
    the nearest graph point there, which then goes through the same evidence as
    any other. Nothing is applied from here directly.
    """
    from scipy.spatial import cKDTree

    from ..candidates import Bridge

    p = params or WavefrontParams()
    stats = {"explored": 0, "reached": 0, "no_graph_at_target": 0, "stopped": {}}
    out: list = []

    points, owner, index_in_seg = [], [], []
    for seg in graph.segments:
        for k, pid in enumerate(seg["point_ids"]):
            points.append(graph.points[pid][:3])
            owner.append(seg["id"])
            index_in_seg.append(k)
    if not points:
        return out, stats
    tree = cKDTree(np.asarray(points, dtype=np.float64))
    node_at = {}
    for nid, node in graph.nodes.items():
        node_at[nid] = np.asarray(node[:3], dtype=np.float64)

    ends = [n for n, a in associations.items()
            if a.associated and n not in skip_nodes and n in profiles]
    for n, node in enumerate(ends, 1):
        association = associations[node]
        profile = profiles[node]
        radius = float(association.radius_um)
        reach = p.reach_radii * radius
        tip = association.point_um
        ahead = tip + association.tangent * reach
        stats["explored"] += 1
        try:
            box = corridor_mod.for_candidate(
                frame, index, np.vstack([tip, ahead]), radius_um=radius, stack=stack,
                pad_factor=p.pad_factor, max_voxels=p.corridor_max_voxels,
            )
        except corridor_mod.CorridorTooLarge:
            stats["stopped"]["corridor too large"] = \
                stats["stopped"].get("corridor too large", 0) + 1
            continue
        tails = profile.tail_points_um
        scalar = cost_mod.build(
            box, index, {association.component}, radius_um=radius,
            calibration_points=tails if len(tails) else None,
            centreline_points=geo._centreline_in(graph, box),
            redundancy_weight=p.redundancy_weight, dark_lumen=p.dark_lumen,
        )
        field_ = tensor_mod.build(scalar, box, radius_um=radius, profiles=[profile],
                                  params=p.anisotropy, dark_lumen=p.dark_lumen)
        # Goals: finite voxels touching another component's foreground.
        from scipy import ndimage

        touching = ndimage.binary_dilation(scalar.blocked, np.ones((3, 3, 3), bool))
        goal_mask = touching & np.isfinite(scalar.cost)
        goals = np.argwhere(goal_mask) + scalar.lo_zyx
        start = geo._nudge_into_field(field_, box.to_global(tip[None, :])[0])
        t_source = _local_direction(association.tangent, box)
        if t_source is None:
            continue
        route = bridge_mod.chain(
            field_, start, t_source, goals if len(goals) else None,
            step_um=p.keypoint_step_major * max(profile.major_um, radius),
            max_length_um=reach, cone_deg=p.lookahead_cone_deg,
            cone_weight=p.lookahead_cone_weight,
            min_keypoint_support=p.min_keypoint_support,
            engine=p.engine, stencil=p.stencil,
        )
        if progress is not None:
            progress(n, len(ends), node, route.reason)
        if route.reason != "found":
            key = route.metrics.get("stop_reason", route.reason)
            stats["stopped"][key] = stats["stopped"].get(key, 0) + 1
            continue
        stats["reached"] += 1
        reached = np.asarray(route.metrics["reached_zyx"], dtype=np.int64)
        local = reached - scalar.lo_zyx
        lo = np.maximum(local - 1, 0)
        hi = np.minimum(local + 2, np.asarray(scalar.cost.shape))
        block = scalar.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        rivals = block[scalar.blocked[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]]
        if not len(rivals):
            continue
        hit = int(np.bincount(rivals).argmax())
        world = box.to_world(reached[None, :])[0]
        d, j = tree.query(world, k=1)
        if not np.isfinite(d) or d > 2.0 * max(radius, float(np.max(frame.seg_spacing))):
            stats["no_graph_at_target"] += 1
            continue
        path = route.path_um(frame)
        target_segment, target_index = int(owner[j]), int(index_in_seg[j])
        seg = graph.segment(target_segment)
        n_pts = len(seg["point_ids"])
        target_node = None
        if target_index == 0 and graph.degree(seg["node1"]) == 1:
            target_node = seg["node1"]
        elif target_index == n_pts - 1 and graph.degree(seg["node2"]) == 1:
            target_node = seg["node2"]
        if target_node is not None:
            coords = np.vstack([path, node_at[target_node][None, :]])
            bridge = Bridge(kind="endpoint", source_node=node, coords=coords,
                            radii=np.full(len(coords), radius), target_node=target_node,
                            reconnection_type=1)
        else:
            k = int(np.clip(target_index, 1, max(n_pts - 2, 1)))
            coords = np.vstack([path, np.asarray(points[j], dtype=np.float64)[None, :]])
            bridge = Bridge(kind="tjunction", source_node=node, coords=coords,
                            radii=np.full(len(coords), radius),
                            target_segment=target_segment, target_index=k,
                            reconnection_type=3)
            bridge.metrics["target_point_id"] = seg["point_ids"][k]
        bridge.metrics.update(explored=True, reached_component=hit,
                              chain_length_um=route.metrics.get("chain_length_um"),
                              keypoints=len(route.metrics.get("keypoints_zyx", [])))
        bridge.score = 0.5
        out.append(bridge)
    return out, stats


# --------------------------------------------------------------------- driver


def plan(graph, index, frame, labels=None, *, stack=None, params=None,
         same_component: bool = False, tjunction: bool = True, gate_kwargs=None,
         proposals=None, progress=None, lobe_progress=None, refine_progress=None,
         explore_progress=None) -> geo.Plan:
    """Clean, profile, propose, search, gate and globally select, in one call.

    `labels` is the segmentation lattice the refinement runs against; without
    it (or with ``refine_method="none"``) the graph is used as given. Passing
    `proposals` means "these pairs and no others", as in the sibling.
    """
    from .. import endpoints, tjunction as tj

    p = params or WavefrontParams()
    stats: dict = {}

    if labels is not None and p.refine_method != "none":
        stats["refinement"] = prepare.clean(
            graph, frame, labels, method=p.refine_method, strength=p.refine_strength,
            max_iterations=p.refine_max_iterations, workers=p.refine_workers,
            progress=refine_progress,
        )
    else:
        stats["refinement"] = {"method": "none", "moved_points": 0, "skipped": True}

    profiles = prepare.profile_ends(graph, frame, index)
    tangents = prepare.as_tangents(profiles)
    stats["end_profiles"] = {
        "profiled": len(profiles),
        "with_section": sum(1 for q in profiles.values() if q.major_axis is not None),
    }

    kw = dict(gate_kwargs or {})
    kw.setdefault("cone_length_factor", p.reach_radii)
    if proposals is not None:
        bridges = list(proposals)
    else:
        bridges = [b for b in endpoints.propose(graph, same_component=same_component,
                                                keep_rejected=False, tangents=tangents,
                                                **kw) if b.accepted]
        if tjunction:
            bridges.extend(b for b in tj.propose(graph, same_component=same_component,
                                                 keep_rejected=False, tangents=tangents,
                                                 **kw) if b.accepted)
    stats["proposals"] = len(bridges)

    associations = classify.associate(index, frame, graph, tangents=tangents)
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
                              tangents=tangents, **geo._lobe_gate_kwargs(kw))
        stats["mask_ends"] = len(mask_ends)
        stats["mask_end_proposals"] = len(extra)
        stats["mask_end_gates"] = lobe_stats
        bridges = list(bridges) + extra

    if p.explore_open_ends and proposals is None:
        proposed = {b.source_node for b in bridges}
        proposed |= {b.target_node for b in bridges if b.target_node is not None}
        explored, explore_stats = explore(
            graph, index, frame, associations, profiles, proposed, stack=stack,
            params=p, progress=explore_progress,
        )
        stats["exploration"] = explore_stats
        bridges = list(bridges) + explored

    candidates: list = []
    for n, bridge in enumerate(bridges, 1):
        classified = classify.classify(bridge, associations, index, frame,
                                       fragments=fragments)
        candidate = geo.Candidate(classified=classified, proposal=bridge)
        evaluate(candidate, index, frame, stack=stack, params=p, graph=graph,
                 profiles=profiles)
        candidates.append(candidate)
        if progress is not None:
            progress(n, len(bridges), candidate)

    scored = [(c, c.confidence, c.status, c.reason) for c in candidates]
    decisions = select.select(graph, scored, allow_cycles=p.allow_cycles)
    result = geo.Plan(candidates=candidates, decisions=decisions,
                      associations=associations, fragments=fragments, stats=stats,
                      mask_ends=mask_ends, lobe_report=lobe_report)
    result.stats["engine"] = prop.select_engine(p.engine)
    return result
