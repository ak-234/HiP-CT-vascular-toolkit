"""Collapse short inner branches that split one junction into several.

Phase two of the skeleton simplification in PMC10182136 (Fig. 4b).
Phase one there -- removing short leaves -- is :func:`~.skeleton_optimise.prune_spurs`;
phase three -- smoothing -- is centreline refinement. This module is the middle step.

A skeleton often resolves one branch point into two or more degree-3 nodes joined by
links shorter than the vessel they sit in (3695, 3698 and 3395 on the left tree, each
inside the lumen of a vessel about 1.6 mm in radius). Those links have no cross-section
of their own, they route the longest path through a stub a third of the parent's
calibre, and a joint junction fit has to move all of those nodes together. The paper
removes the links B0..Bk, welds their ends into one node at the endpoints' centre of
mass p_CA, then moves that node to the point of greatest distance to the segmentation
boundary near p_CA.

Two deviations, both recorded in every report:

* **The threshold is local by default.** The paper's automatic threshold is the
  diameter of the thinnest vessel in the tree -- 54 um here, which catches 8 of 3045
  inner links and none of the three above. A link qualifies here when it is shorter
  than `factor` times the larger radius of the other vessels at its ends, i.e. when
  it lies inside the parent lumen. The paper's global thresholds remain available
  through `min_length_um` / :func:`thinnest_diameter_um`.
* **A collapse that would take a centreline out of the segmentation is refused**, not
  applied. Moving a junction drags the first sample of every incident segment, and a
  new outside edge there would block every refinement run that follows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .skeleton_optimise import _radius_away, _seg_length_um

LINK_FACTOR = 1.0
#: Shortest distance, in voxels, over which a moved junction's shift fades out.
BLEND_MIN_VOXELS = 4
#: Cap on the ROI read to reposition one junction; ~64 MB of uint8.
MAX_ROI_VOXELS = 64_000_000


class RoiTooLarge(ValueError):
    """The repositioning ROI for one cluster exceeds :data:`MAX_ROI_VOXELS`."""


@dataclass
class LinkReport:
    rule: str
    factor: float | None
    min_length_um: float | None
    clusters: list[dict] = field(default_factory=list)

    @property
    def collapsed(self) -> list[dict]:
        """Collapsed clusters, or those that would be in a dry run."""
        return [c for c in self.clusters if c["status"] in ("collapsed", "accepted")]

    @property
    def refused(self) -> list[dict]:
        return [c for c in self.clusters if c["status"] == "refused"]

    def describe(self) -> str:
        links = sum(len(c["links"]) for c in self.clusters)
        dry = any(c["status"] == "accepted" for c in self.clusters)
        return (f"{links} short inner link(s) in {len(self.clusters)} junction cluster(s) "
                f"[{self.rule}]; {len(self.collapsed)} {'would collapse' if dry else 'collapsed'}, "
                f"{len(self.refused)} refused")

    def to_dict(self) -> dict:
        return dict(rule=self.rule, factor=self.factor, min_length_um=self.min_length_um,
                    n_clusters=len(self.clusters), n_collapsed=len(self.collapsed),
                    n_refused=len(self.refused), clusters=self.clusters)


def thinnest_diameter_um(graph) -> float:
    """The paper's automatic inner-edge threshold: the thinnest vessel's diameter."""
    radii = [float(np.median(graph.radii(sid))) for sid in graph.segment_ids()
             if len(graph.radii(sid))]
    radii = [r for r in radii if np.isfinite(r) and r > 0]
    if not radii:
        raise ValueError("no finite positive radii to derive a thinnest diameter from")
    return 2.0 * min(radii)


def _is_link(graph, sid) -> bool:
    seg = graph.segment(sid)
    a, b = seg["node1"], seg["node2"]
    return a != b and graph.degree(a) >= 3 and graph.degree(b) >= 3


def _parent_radius(graph, sid) -> float:
    """Larger junction-free radius among the other vessels at either end of `sid`."""
    seg = graph.segment(sid)
    radii = [_radius_away(graph, other, nid)
             for nid in (seg["node1"], seg["node2"])
             for other in graph.node_segments(nid) if other != sid]
    radii = [r for r in radii if np.isfinite(r) and r > 0]
    return max(radii, default=float("nan"))


def find_short_links(graph, *, factor: float | None = LINK_FACTOR,
                     min_length_um: float | None = None, sids=None) -> list[dict]:
    """Group qualifying inner links into clusters sharing nodes (union-find).

    A link has degree >= 3 at both ends. It qualifies by the ``local`` rule when its
    arclength is below `factor` x :func:`_parent_radius`, and by the ``global`` rule
    when below `min_length_um`. Either rule may be disabled with ``None``/0. `sids`
    limits the search to those segments.
    """
    if not factor and not min_length_um:
        raise ValueError("enable the local factor, a global min_length_um, or both")
    candidates = graph.segment_ids() if sids is None else sorted(set(sids))
    links = {}
    for sid in candidates:
        if not graph.has_segment(sid) or not _is_link(graph, sid):
            continue
        length = _seg_length_um(graph, sid)
        parent = _parent_radius(graph, sid)
        rules = []
        if factor and np.isfinite(parent) and length < factor * parent:
            rules.append("local")
        if min_length_um and length < min_length_um:
            rules.append("global")
        if rules:
            links[sid] = dict(length_um=length, parent_radius_um=parent, rules=rules)

    parent_of = {}
    def root(n):
        while parent_of.setdefault(n, n) != n:
            parent_of[n] = parent_of[parent_of[n]]
            n = parent_of[n]
        return n
    for sid in sorted(links):
        seg = graph.segment(sid)
        ra, rb = root(seg["node1"]), root(seg["node2"])
        parent_of[max(ra, rb)] = min(ra, rb)
    groups = {}
    for sid in sorted(links):
        groups.setdefault(root(graph.segment(sid)["node1"]), []).append(sid)
    clusters = []
    for members in groups.values():
        nodes = sorted({graph.segment(s)[k] for s in members for k in ("node1", "node2")})
        clusters.append(dict(
            links=members, nodes=nodes,
            link_lengths_um=[links[s]["length_um"] for s in members],
            link_radii_um=[float(np.median(graph.radii(s))) for s in members],
            parent_radius_um=[links[s]["parent_radius_um"] for s in members],
            rules=sorted({r for s in members for r in links[s]["rules"]})))
    return clusters


def _read_roi(labels, lo, hi) -> np.ndarray:
    """Foreground mask over voxel box ``[lo, hi)`` in (x, y, z), returned as (z, y, x)."""
    if isinstance(labels, np.ndarray):
        return labels[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] > 0
    return np.stack([labels.slice_rows(int(z), int(lo[1]), int(hi[1]))[:, lo[0]:hi[0]]
                     for z in range(lo[2], hi[2])]) > 0


def reposition(p_ca, ball_um, margin_um, anchors_um, labels, frame):
    """The point of greatest boundary distance within `ball_um` of `p_ca`.

    Only the foreground component holding most of `anchors_um` (the cluster's own
    nodes) is eligible, so a junction cannot hop into a neighbouring lumen. The ROI
    extends `margin_um` beyond the ball so the distance there is not truncated by the
    ROI edge. Ties go to the candidate nearest p_CA. Returns ``(p_new, edt_new,
    edt_ca)`` in um, or ``None`` when the cluster lies outside the segmentation.
    """
    spacing = np.asarray(frame.seg_spacing, dtype=float)
    centre = frame.um_to_seg(np.asarray(p_ca, float)[None])[0]
    half = np.ceil((ball_um + margin_um) / spacing).astype(int) + 2
    lo = np.maximum(0, np.floor(centre).astype(int) - half)
    hi = np.minimum(np.asarray(frame.seg_dims, int), np.ceil(centre).astype(int) + half + 1)
    if np.prod(hi - lo) > MAX_ROI_VOXELS:
        raise RoiTooLarge(f"{np.prod(hi - lo)} voxels")
    mask = _read_roi(labels, lo, hi)
    distance = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing[::-1])[1:-1, 1:-1, 1:-1]
    components, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    anchor_ijk = np.rint(frame.um_to_seg(np.asarray(anchors_um, float))).astype(int) - lo
    inside = np.all((anchor_ijk >= 0) & (anchor_ijk < hi - lo), axis=1)
    owners = components[anchor_ijk[inside, 2], anchor_ijk[inside, 1], anchor_ijk[inside, 0]]
    owners = owners[owners > 0]
    if not len(owners):
        return None
    component = np.bincount(owners).argmax()
    zz, yy, xx = np.nonzero(components == component)
    ijk = np.stack([xx, yy, zz], axis=1) + lo
    xyz = frame.seg_to_um(ijk)
    offset = np.linalg.norm(xyz - np.asarray(p_ca, float), axis=1)
    near = offset <= ball_um
    if not near.any():
        return None
    d = distance[zz, yy, xx]
    order = np.lexsort((offset[near], -d[near]))
    best = np.flatnonzero(near)[order[0]]
    ca = np.rint(centre).astype(int) - lo
    edt_ca = (float(distance[ca[2], ca[1], ca[0]])
              if np.all((ca >= 0) & (ca < hi - lo)) else 0.0)
    return xyz[best], float(d[best]), edt_ca


def _positions(p_edt, p_ca, kept_xyz):
    """Candidate junction positions, most preferred first."""
    return [("p_edt", np.asarray(p_edt, float)), ("p_ca", np.asarray(p_ca, float)),
            ("kept_node", np.asarray(kept_xyz, float))]


def _keep_node(graph, cluster) -> int:
    """The cluster node on the thickest vessel that is not itself a removed link."""
    links = set(cluster["links"])
    def thickness(nid):
        radii = [_radius_away(graph, s, nid) for s in graph.node_segments(nid) if s not in links]
        radii = [r for r in radii if np.isfinite(r)]
        return max(radii, default=-1.0)
    return max(cluster["nodes"], key=lambda n: (thickness(n), -n))


def collapse_links(graph, clusters, labels, frame, *, sampler=None,
                   apply: bool = True) -> list[dict]:
    """Weld each cluster into one repositioned node, or refuse it with a reason.

    Every check runs on proposed coordinates before the graph is touched, and all
    accepted collapses go into one ``graph.batch`` so a single undo reverts them.
    With `apply` false nothing is edited and accepted clusters report ``accepted``.
    """
    from ..crosssection import _PlaneSampler
    from .centreline_refine import bad_edges
    sampler = sampler if sampler is not None else _PlaneSampler(labels, frame)
    accepted, results = [], []
    claimed = set()
    for cluster in clusters:
        row = dict(cluster, status="refused")
        results.append(row)
        nodes, links = set(cluster["nodes"]), set(cluster["links"])
        if nodes & claimed:
            row["reason"] = "overlaps_another_cluster"
            continue
        incident = sorted({s for n in nodes for s in graph.node_segments(n)} - links)
        looped = [s for s in incident if {graph.segment(s)["node1"], graph.segment(s)["node2"]} <= nodes]
        if looped:
            # Both ends in the cluster: welding would delete a real vessel as a self-loop.
            row.update(reason="would_create_self_loop", segments=looped)
            continue
        ends = np.array([x for s in sorted(links) for x in graph.coords(s)[[0, -1]]])
        p_ca = ends.mean(axis=0)
        # "The closest neighbourhood of p_CA": no further than the farthest welded
        # node. A ball the length of the longest link let the maximum slide along
        # the thickest incident vessel to the ball's edge.
        ball = max(float(np.linalg.norm(ends - p_ca, axis=1).max()),
                   2.0 * float(np.max(frame.seg_spacing)))
        # The boundary distance inside the ball is at most about the parent radius,
        # so that much ROI beyond the ball keeps it from being truncated.
        margin = 1.25 * max((r for r in cluster["parent_radius_um"] if np.isfinite(r)), default=ball)
        anchors = np.array([graph.nodes[n][:3] for n in sorted(nodes)])
        try:
            found = reposition(p_ca, ball, margin, anchors, labels, frame)
        except RoiTooLarge as exc:
            row["reason"] = f"roi_too_large ({exc})"
            continue
        if found is None:
            row["reason"] = "junction_outside_segmentation"
            continue
        p_edt, edt_new, edt_ca = found
        kept = _keep_node(graph, cluster)

        min_blend = BLEND_MIN_VOXELS * float(np.max(frame.seg_spacing))

        def propose(point):
            # Per segment, the paper's snap of the end point first; if its new first
            # edge leaves the segmentation, a shift faded along the curve. Neither
            # wins everywhere: fading moves a run of samples, which exits a thin
            # daughter that a snapped chord stays inside, and vice versa at a fork.
            shaped, out, blended = {}, [], []
            for sid in incident:
                seg, x = graph.segment(sid), graph.coords(sid)
                before = bad_edges(x, sampler, frame)
                end = 0 if seg["node1"] in nodes else -1
                snapped = x.copy()
                snapped[end] = point
                for shape, y in (("snap", snapped), ("blend", blend_end(x, end, point, min_blend))):
                    if not np.any(bad_edges(y, sampler, frame) & ~before):
                        shaped[sid] = y
                        if shape == "blend":
                            blended.append(sid)
                        break
                else:
                    out.append(sid)
            return shaped, out, blended

        # The distance maximum can pull an incident segment's first edge across
        # background. Fall back to the paper's p_CA, then to where the kept node
        # already is, before refusing; the report says which was used.
        tried = {}
        for position, point in _positions(p_edt, p_ca, np.asarray(graph.nodes[kept][:3], float)):
            shaped, tried[position], blended = propose(point)
            if not tried[position]:
                break
        p_new = point
        row.update(kept_node=kept, blended_segments=blended, dropped_nodes=sorted(nodes - {kept}),
                   p_ca=p_ca.tolist(), p_edt=np.asarray(p_edt).tolist(),
                   p_new=np.asarray(p_new).tolist(), position=position,
                   edt_at_p_ca_um=edt_ca, edt_at_p_edt_um=edt_new,
                   move_from_p_ca_um=float(np.linalg.norm(p_new - p_ca)), ball_um=ball,
                   # The maximum lies on the search boundary, so the true one is
                   # further out; the collapse is still contained but worth a look.
                   at_ball_edge=bool(position == "p_edt" and np.linalg.norm(p_edt - p_ca)
                                     >= ball - float(np.max(frame.seg_spacing))),
                   new_degree=len(incident), incident_segments=incident)
        if tried[position]:
            row.update(reason="new_segmentation_exit", segments=tried[position],
                       exits_by_position=tried)
            continue
        row["status"] = "collapsed" if apply else "accepted"
        claimed |= nodes
        accepted.append((cluster, kept, p_new, shaped))
    if accepted and apply:
        with graph.batch("collapse short inner links"):
            for cluster, kept, p_new, shaped in accepted:
                for nid in cluster["nodes"]:
                    if nid != kept:
                        graph.merge_nodes(kept, nid)
                for sid in cluster["links"]:
                    # merge_nodes deletes a link once both its ends are welded.
                    if graph.has_segment(sid):
                        graph.delete_segment(sid)
                graph.move_node(kept, p_new)
                for sid, coords in shaped.items():
                    graph.set_segment_coords(sid, coords)
    return results


def blend_end(x, end, point, min_length_um):
    """Move polyline end `end` (0 or -1) to `point`, fading the shift along the curve.

    Snapping only the end point turns the first edge into a straight chord from the
    new junction to the old second sample, which cuts across background in a Y- or
    T-shaped confluence once a welded node was several hundred um away. The shift
    instead decays with a raised cosine over twice its own length (at least
    `min_length_um`), capped at half the segment so the far end never moves.
    """
    x = np.asarray(x, float)
    order = np.arange(len(x)) if end == 0 else np.arange(len(x))[::-1]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(x[order], axis=0), axis=1))]
    shift = np.asarray(point, float) - x[order[0]]
    length = min(max(2.0 * float(np.linalg.norm(shift)), min_length_um), 0.5 * s[-1])
    weight = np.where(s < length, 0.5 * (1 + np.cos(np.pi * s / max(length, 1e-9))), 0.)
    weight[0] = 1.0
    y = x.copy()
    y[order] = x[order] + weight[:, None] * shift
    return y


def saved_ids(graph) -> dict:
    """Old -> new ids after writing `graph` to Amira, listing only ids that change.

    :func:`~.adapter.to_spatial_graph` renumbers segments by list position and nodes
    by :func:`~.adapter.vertex_node_ids`, so removing links shifts every later id: a
    ``--segment`` or root quoted from the input names a different object afterwards.
    """
    from .adapter import vertex_node_ids
    segments = {int(seg["id"]): i for i, seg in enumerate(graph.triple.segments)}
    nodes = {int(nid): i for i, nid in enumerate(vertex_node_ids(graph.triple))}
    return dict(segments={k: v for k, v in segments.items() if k != v},
                nodes={k: v for k, v in nodes.items() if k != v})


def simplify_links(graph, labels, frame, *, factor: float | None = LINK_FACTOR,
                   min_length_um: float | None = None, auto_thinnest: bool = False,
                   sids=None, apply: bool = True) -> LinkReport:
    """Find and (unless `apply` is false) collapse short inner links."""
    if auto_thinnest:
        min_length_um = thinnest_diameter_um(graph)
    rules = [name for name, on in (("local", factor), ("global", min_length_um)) if on]
    report = LinkReport(rule="+".join(rules), factor=factor or None, min_length_um=min_length_um)
    clusters = find_short_links(graph, factor=factor, min_length_um=min_length_um, sids=sids)
    report.clusters = collapse_links(graph, clusters, labels, frame, apply=apply)
    return report
