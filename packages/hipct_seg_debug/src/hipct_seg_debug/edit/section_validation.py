"""Shared finite-branch section ownership checks, in world micrometres.

Angles are diagnostics, never standalone rejection gates. Foreign tube support
must intersect the selected component and connect to its centreline through
foreground. Flat segment-end support avoids extending a continuation backwards
into the preceding vessel as a spherical capsule would.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

REJECTION_REASONS = ('target_obliquity', 'neighbouring_lumen_contamination',
                     'unstable_section', 'truncation', 'insufficient_support')


@dataclass
class SectionVerdict:
    accepted: bool = True
    reason: str = "accepted"
    target_angle_degrees: float = 0.
    contaminants: list = field(default_factory=list)
    cuts: list | None = None


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
        self.continuation = {sid: root(sid) for sid in parent}
        records = []
        for sid in sorted(graph.segment_ids()):
            x, r = graph.coords(sid), graph.radii(sid)
            for i, (a, b) in enumerate(zip(x[:-1], x[1:])):
                length = float(np.linalg.norm(b-a))
                if length > 1e-8 and np.isfinite(r[i:i+2]).all():
                    records.append((sid, a, b, max(float(r[i]), 0.),
                                    max(float(r[i+1]), 0.), length))
        self.records = records
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
            if not inside.any():
                continue
            ids = np.flatnonzero(inside)
            centre = a+along[ids, None]*direction
            radius = ra+(rb-ra)*along[ids]/length
            distance = np.linalg.norm(points[ids]-centre, axis=1)
            ids_near = np.flatnonzero(distance < radius)
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
                        tangent_plane_angle_degrees=float(np.degrees(np.arcsin(
                            np.clip(abs(normal @ direction), 0, 1)))))
                    break
        if contaminated:
            verdict.accepted = False
            verdict.reason = "neighbouring_lumen_contamination"
            verdict.contaminants = [contaminated[k] for k in sorted(contaminated)]
        return verdict

    def validator(self, sid, target_tangent, sampler, frame, diagnostics=None):
        volume_cache = {}
        def check(cut, ijk, normal):
            verdict = self.validate(sid, cut, frame.seg_to_um(np.asarray(ijk))[0], normal,
                                    target_tangent, sampler, frame)
            if diagnostics is not None:
                diagnostics['max_target_obliquity_degrees'] = max(
                    diagnostics.get('max_target_obliquity_degrees', 0.), verdict.target_angle_degrees)
            return verdict
        def validate_slab(cuts, centre, normal, radius_vox, offsets, max_half):
            verdicts = [check(c, centre+offset*radius_vox*normal, normal)
                        for c, offset in zip(cuts, offsets)]
            rivals = sorted({row['segment'] for verdict in verdicts
                             for row in verdict.contaminants})
            if not rivals:
                return next((v for v in verdicts if not v.accepted), SectionVerdict())
            # An incident branch in the merged junction has no exclusive boundary.
            # Never manufacture its ownership using a watershed through the node.
            segment = self.graph.segment(sid)
            incident = set(self.graph.node_segments(segment['node1'])) | set(
                self.graph.node_segments(segment['node2']))
            if incident.intersection(rivals):
                return SectionVerdict(False, 'neighbouring_lumen_contamination')
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
        return check
