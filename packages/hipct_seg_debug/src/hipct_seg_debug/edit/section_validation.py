"""Shared finite-branch section ownership checks, in world micrometres.

Angles are diagnostics, never standalone rejection gates. Foreign tube support
must intersect the selected component and connect to its centreline through
foreground. Flat segment-end support avoids extending a continuation backwards
into the preceding vessel as a spherical capsule would.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

REJECTION_REASONS = ('target_obliquity', 'neighbouring_lumen_contamination',
                     'unstable_section', 'truncation', 'insufficient_support')

# Treat a junction's through-vessel -- the pair of branches that continue nearly
# straight with similar calibre -- as one vessel for ownership. 3655 lies wholly
# within a flattened junction confluence; its calibre-matched continuation 3641
# (ratio 0.98) otherwise contaminates every 3655 section. On by default; set
# HIPCT_THROUGH_JUNCTIONS=0 to disable. Environment-driven so spawned section
# workers inherit it.
THROUGH_JUNCTIONS = os.environ.get("HIPCT_THROUGH_JUNCTIONS", "1") != "0"
THROUGH_MIN_CALIBRE_RATIO = 0.75
THROUGH_MIN_STRAIGHTNESS = np.cos(np.radians(45.))


def _through_pair(graph, nid, incident):
    """The incident pair continuing nearly straight with matching calibre, if any."""
    from .skeleton_optimise import _radius_away
    out = {}
    for sid in incident:
        x = graph.coords(sid)
        if len(x) < 2:
            continue
        if graph.segment(sid)["node2"] == nid:
            x = x[::-1]
        r = _radius_away(graph, sid, nid)
        # Direction leaving the node, over about two local radii.
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
        k = int(np.clip(np.searchsorted(s, 2 * max(r, 1e-6)), 1, len(x) - 1))
        d = x[k] - x[0]
        if np.linalg.norm(d) > 0 and np.isfinite(r) and r > 0:
            out[sid] = (d / np.linalg.norm(d), r)
    best = None
    ids = sorted(out)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            (da, ra), (db, rb) = out[a], out[b]
            straight = float(-(da @ db))
            ratio = min(ra, rb) / max(ra, rb)
            if straight >= THROUGH_MIN_STRAIGHTNESS and ratio >= THROUGH_MIN_CALIBRE_RATIO:
                if best is None or straight > best[0]:
                    best = (straight, a, b)
    return None if best is None else (best[1], best[2])


@dataclass
class SectionVerdict:
    accepted: bool = True
    reason: str = "accepted"
    target_angle_degrees: float = 0.
    contaminants: list = field(default_factory=list)
    cuts: list | None = None


def _endpoint_support(x, radii, arc):
    """Finite end planes from physical branch spans, not a noisy first edge.

    Only the adjacent two-radius span (at most half the branch) uses each cap.
    Distant bends may return across the plane and must retain their own support.
    Interpolation in arclength makes the model independent of point density.
    These planes constrain spatial tube extent; they are not angular gates.
    """
    caps = []
    for end in (0, -1):
        radius = float(radii[end])
        width = min(2*max(radius, 0.), arc[-1]/2) if np.isfinite(radius) else 0.
        if width <= 1e-8:
            continue
        position = width if end == 0 else arc[-1]-width
        anchor = np.array([np.interp(position, arc, x[:, k]) for k in range(3)])
        inward = anchor-x[end]
        norm = float(np.linalg.norm(inward))
        if norm > 1e-8:
            caps.append((end, x[end], inward/norm, width))
    return caps


class SectionContext:
    def __init__(self, graph):
        self.graph = graph
        parent = {sid: sid for sid in graph.segment_ids()}
        def root(sid):
            while parent[sid] != sid:
                parent[sid] = parent[parent[sid]]
                sid = parent[sid]
            return sid
        for nid in graph.nodes:
            incident = sorted(graph.node_segments(nid))
            if len(incident) == 2:
                parent[root(incident[1])] = root(incident[0])
            elif len(incident) >= 3 and THROUGH_JUNCTIONS:
                pair = _through_pair(graph, nid, incident)
                if pair is not None:
                    parent[root(pair[1])] = root(pair[0])
        self.continuation = {sid: root(sid) for sid in parent}
        records, arcs = [], []
        self.end_support, self.lengths = {}, {}
        for sid in sorted(graph.segment_ids()):
            x, r = graph.coords(sid), graph.radii(sid)
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
            if len(x) >= 2:
                self.end_support[sid] = _endpoint_support(x, r, arc)
                self.lengths[sid] = float(arc[-1])
            for i, (a, b) in enumerate(zip(x[:-1], x[1:])):
                length = float(np.linalg.norm(b-a))
                if length > 1e-8 and np.isfinite(r[i:i+2]).all():
                    records.append((sid, a, b, max(float(r[i]), 0.),
                                    max(float(r[i+1]), 0.), length))
                    arcs.append(float(arc[i]))
        self.records = records
        self.record_arcs = np.asarray(arcs)
        self.buckets = []
        if records:
            centres = np.array([(v[1]+v[2])/2 for v in records])
            bounds = np.array([v[5]/2+max(v[3:5]) for v in records])
            levels = np.floor(np.log2(np.maximum(bounds, 1e-6))).astype(int)
            for level in np.unique(levels):
                indices = np.flatnonzero(levels == level)
                self.buckets.append((cKDTree(centres[indices]), indices,
                                     float(bounds[indices].max())))

    def validate(self, sid, cut, origin_um, normal, target_tangent, sampler, frame):
        normal = np.asarray(normal, dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)
        target = np.asarray(target_tangent, dtype=float)
        target /= max(np.linalg.norm(target), 1e-12)
        verdict = SectionVerdict(target_angle_degrees=float(np.degrees(
            np.arccos(np.clip(abs(normal @ target), 0, 1)))))
        pixels = np.argwhere(cut.blob8)
        if not len(pixels):
            return SectionVerdict(False, "insufficient_support")
        spacing = float(frame.seg_spacing[0])
        uv = (pixels-cut.half)*spacing
        points = origin_um+uv[:, :1]*cut.u+uv[:, 1:]*cut.v
        reach = float(np.linalg.norm(points-origin_um, axis=1).max())
        candidates = []
        for tree, indices, bound in self.buckets:
            candidates.extend(indices[tree.query_ball_point(origin_um, reach+bound)])
        contaminated = {}
        for index in sorted(candidates):
            other, a, b, ra, rb, length = self.records[index]
            if self.continuation[other] == self.continuation[sid] or other in contaminated:
                continue
            direction = (b-a)/length
            signed = np.array([(a-origin_um) @ normal, (b-origin_um) @ normal])
            radial_reach = max(ra, rb)*np.sqrt(max(0., 1-(normal @ direction)**2))
            if signed.min() > radial_reach or signed.max() < -radial_reach:
                continue
            along = (points-a) @ direction
            inside = (along >= -1e-8) & (along <= length+1e-8)
            branch_arc = self.record_arcs[index]+along
            for end, endpoint, inward, width in self.end_support[other]:
                distance_from_end = branch_arc if end == 0 else self.lengths[other]-branch_arc
                inside &= ((distance_from_end > width) |
                           ((points-endpoint) @ inward >= -1e-8))
            if not inside.any():
                continue
            ids = np.flatnonzero(inside)
            centre = a+along[ids, None]*direction
            radius = ra+(rb-ra)*along[ids]/length
            distance = np.linalg.norm(points[ids]-centre, axis=1)
            ids_near = np.flatnonzero(distance < radius)
            if not len(ids_near):
                continue
            # Every tested line includes its foreign-axis endpoint. Exclude
            # endpoints already known to be background in one batch, rather
            # than building thousands of rays that must fail at their last voxel.
            # This is exactly the same acceptance predicate, not a relaxation.
            axis_foreground = sampler.at(frame.um_to_seg(centre[ids_near])) > 0
            ids_near = ids_near[axis_foreground]
            if not len(ids_near):
                continue
            # Start at the most central candidate, but inspect every possible
            # foreground connection until one proves shared lumen support.
            for j in ids_near[np.argsort(distance[ids_near])]:
                p, q = frame.um_to_seg(np.array([points[ids[j]], centre[j]]))
                # Shared exact voxel traversal is imported lazily to avoid a
                # crosssection/radius/refinement module import cycle.
                from .centreline_refine import line_samples_ijk
                if np.all(sampler.at(line_samples_ijk(p, q)) > 0):
                    contaminated[other] = dict(
                        segment=int(other), normal_tangent_alignment=float(abs(normal @ direction)),
                        edge_start_um=a.tolist(), edge_end_um=b.tolist(),
                        overlap_point_um=points[ids[j]].tolist(),
                        branch_axis_point_um=centre[j].tolist(),
                        branch_axis_arclength_um=float(branch_arc[ids[j]]),
                        branch_radius_um=float(radius[j]), axis_distance_um=float(distance[j]),
                        tangent_plane_angle_degrees=float(np.degrees(np.arcsin(
                            np.clip(abs(normal @ direction), 0, 1)))))
                    break
        if contaminated:
            verdict.accepted = False
            verdict.reason = "neighbouring_lumen_contamination"
            verdict.contaminants = [contaminated[k] for k in sorted(contaminated)]
        return verdict

    def validator(self, sid, target_tangent, sampler, frame, diagnostics=None, trace=None):
        volume_cache = {}
        segment = self.graph.segment(sid)
        incident = set(self.graph.node_segments(segment['node1'])) | set(
            self.graph.node_segments(segment['node2']))
        prechecked = [None, None]
        def check(cut, ijk, normal):
            verdict = self.validate(sid, cut, frame.seg_to_um(np.asarray(ijk))[0], normal,
                                    target_tangent, sampler, frame)
            if diagnostics is not None:
                diagnostics['max_target_obliquity_degrees'] = max(
                    diagnostics.get('max_target_obliquity_degrees', 0.), verdict.target_angle_degrees)
            return verdict
        def prevalidate(cut, ijk, normal):
            # Validation normalises its argument; do not alter the candidate
            # before its remaining image planes have been sampled.
            verdict = check(cut, ijk, np.asarray(normal).copy())
            prechecked[:] = [cut, verdict]
            if any(r['segment'] in incident for r in verdict.contaminants):
                return verdict
            # Non-incident ownership still requires the entire slab.
            return SectionVerdict()
        def validate_slab(cuts, centre, normal, radius_vox, offsets, max_half):
            verdicts = [None]*len(cuts)
            order = sorted(range(len(cuts)), key=lambda i: abs(offsets[i]))
            for i in order:
                verdicts[i] = (prechecked[1] if cuts[i] is prechecked[0] else
                               check(cuts[i], centre+offsets[i]*radius_vox*normal, normal))
                # An incident overlap makes this entire orientation ineligible.
                # Further plane checks cannot rescue it. Tracing deliberately
                # retains all planes for the diagnostic report.
                if trace is None and any(r['segment'] in incident for r in verdicts[i].contaminants):
                    return SectionVerdict(False, 'neighbouring_lumen_contamination',
                                          contaminants=verdicts[i].contaminants)
            rivals = sorted({row['segment'] for verdict in verdicts
                             for row in verdict.contaminants})
            # An incident branch in the merged junction has no exclusive boundary.
            # Never manufacture its ownership using a watershed through the node.
            if trace is not None:
                from ..crosssection import _perimeter_um
                trace(dict(segment=int(sid), centre_ijk=np.asarray(centre).tolist(),
                           normal=np.asarray(normal).tolist(), radius_vox=float(radius_vox),
                           offsets=list(offsets), incident_rivals=sorted(incident.intersection(rivals)),
                           sections=[dict(offset=float(offset), area_vox=int(c.blob8.sum()),
                                          perimeter_um=float(_perimeter_um(c.blob4, float(frame.seg_spacing[0]))),
                                          contaminants=v.contaminants, reason=v.reason)
                                     for c, v, offset in zip(cuts, verdicts, offsets)]))
            if not rivals:
                return next((v for v in verdicts if not v.accepted), SectionVerdict())
            if incident.intersection(rivals):
                return SectionVerdict(False, 'neighbouring_lumen_contamination',
                                      contaminants=[r for v in verdicts for r in v.contaminants])
            from .radius_perimeter import _resolve_owned_slab
            coords = {other: frame.um_to_seg(self.graph.coords(other)) for other in [sid, *rivals]}
            resolved = _resolve_owned_slab(
                sampler, self.graph, coords, sid, rivals, centre, normal, radius_vox,
                max(c.half for c in cuts), max_half, float(frame.seg_spacing[0]), offsets,
                volume_cache=volume_cache)
            if resolved is None:
                return SectionVerdict(False, 'neighbouring_lumen_contamination')
            # All three parallel sections use one 3-D ownership volume. The shared
            # caller then checks area, perimeter AND centroid stability on these
            # owned sections, using exactly the same criteria as exclusive cuts.
            if diagnostics is not None:
                diagnostics['owned_slabs'] = diagnostics.get('owned_slabs', 0)+1
            return SectionVerdict(cuts=resolved)
        check.validate_slab = validate_slab
        if trace is None:
            check.prevalidate = prevalidate
        return check
