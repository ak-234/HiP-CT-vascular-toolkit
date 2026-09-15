"""Optional smooth displacement fields for circular reconstruction clearance.

The segmentation centreline is the measurement result. This module creates a
separate reconstruction layout; radii are transported unchanged and must not be
remeasured at those displaced positions. Capsule clearance is a conservative
preflight, not a certificate for a subsequently blended or smoothed surface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

import numpy as np
from scipy.spatial import cKDTree

from ..crosssection import _PlaneSampler
from .centreline_refine import arclength, bad_edges, feasible_move


@dataclass
class Contact:
    segment_a: int
    segment_b: int
    edge_a: int
    edge_b: int
    fraction_a: float
    fraction_b: float
    clearance_um: float
    position_a: list
    position_b: list


@dataclass
class ClearanceReport:
    contacts_before: int = 0
    contacts_after: int = 0
    iterations: int = 0
    moved_points: int = 0
    seconds: float = 0.
    collision_model: str = "conservative maximum-radius capsules"
    radii_preserved: bool = True
    surface_validation_required: bool = True
    status: str = "unresolved"
    outside_edges: int = 0
    tight_bends_before: int = 0
    tight_bends_after: int = 0
    history: list = field(default_factory=list)
    remaining: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _closest(p, p1, q, q1):
    """Exact closest parameters of two finite line segments, including parallel."""
    u, v, w = p1-p, q1-q, p-q
    a, b, c = u@u, u@v, v@v
    d, e = u@w, v@w
    candidates = []
    for s in (0., 1.):
        t = np.clip((b*s+e)/c, 0, 1) if c > 1e-20 else 0.
        candidates.append((s, t))
    for t in (0., 1.):
        s = np.clip((b*t-d)/a, 0, 1) if a > 1e-20 else 0.
        candidates.append((s, t))
    den = a*c-b*b
    if den > 1e-12*max(a*c, 1e-20):
        s, t = (b*e-c*d)/den, (a*e-b*d)/den
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    s, t = min(candidates, key=lambda st: np.linalg.norm(w+st[0]*u-st[1]*v))
    return float(s), float(t), p+s*u, q+t*v


def contacts(graph, *, coords=None, gap_um=0., pair_limit=2_000_000):
    """Capsule broad phase with bounded, local junction exemptions.

    Radii are bounded by each edge's maximum endpoint radius. This can flag a
    tapered edge unnecessarily, but cannot miss it by testing only the radius at
    the closest centreline pair. A pair limit raises; an incomplete check is never
    reported as clear.
    """
    if gap_um < 0 or not np.isfinite(gap_um):
        raise ValueError("gap_um must be finite and nonnegative")
    coords = coords or {sid: graph.coords(sid) for sid in graph.segment_ids()}
    edges = []
    arcs = {}
    for sid, x in coords.items():
        s = arcs[sid] = arclength(x)
        r = graph.radii(sid)
        for i in range(len(x)-1):
            edges.append((sid, i, x[i], x[i+1], float(max(r[i:i+2])), s[i], s[i+1]))
    if not edges:
        return []
    p = np.asarray([e[2] for e in edges])
    q = np.asarray([e[3] for e in edges])
    radius = np.asarray([e[4] for e in edges])
    mid = (p+q)/2
    bound = np.linalg.norm(q-p, axis=1)/2+radius+gap_um/2
    # Bucket by geometric size. Querying every small edge against the largest
    # artery's radius turns a multiscale tree into almost all-pairs work.
    bucket = np.floor(np.log2(np.maximum(bound, 1e-12))).astype(int)
    trees = []
    for key in np.unique(bucket):
        indices = np.flatnonzero(bucket == key)
        trees.append((indices, cKDTree(mid[indices]), float(bound[indices].max())))
    segment_info = {sid: graph.segment(sid) for sid in coords}
    segment_radii = {sid: graph.radii(sid) for sid in coords}
    out = []
    examined = 0
    for a, edge in enumerate(edges):
        neighbours = (int(indices[j]) for indices, tree, largest in trees
                      for j in tree.query_ball_point(mid[a], bound[a]+largest))
        for b in neighbours:
            if b <= a or np.linalg.norm(mid[a]-mid[b]) > bound[a]+bound[b]:
                continue
            other = edges[b]
            sid, i, _, _, ra, sa, ea = edge
            tid, j, _, _, rb, sb, eb = other
            if sid == tid and abs(i-j) <= 1:
                continue
            if sid == tid and max(ea, eb)-min(sa, sb) <= ra+rb:
                continue
            examined += 1
            if examined > pair_limit:
                raise RuntimeError("clearance pair limit reached; use a smaller region")
            u, v, pa, pb = _closest(p[a], q[a], p[b], q[b])
            distance = np.linalg.norm(pa-pb)
            if distance >= ra+rb+gap_um:
                continue
            aa, bb = sa+u*(ea-sa), sb+v*(eb-sb)
            if sid == tid:
                # Nearby samples of the same tube naturally overlap. Nonlocal
                # folds remain active; local tight bending is reported separately.
                if abs(aa-bb) <= ra+rb:
                    continue
            else:
                s1, s2 = segment_info[sid], segment_info[tid]
                common = {s1['node1'], s1['node2']} & {s2['node1'], s2['node2']}
                junction = False
                for nid in common:
                    d1 = aa if s1['node1'] == nid else arcs[sid][-1]-aa
                    d2 = bb if s2['node1'] == nid else arcs[tid][-1]-bb
                    r1 = segment_radii[sid][0 if s1['node1'] == nid else -1]
                    r2 = segment_radii[tid][0 if s2['node1'] == nid else -1]
                    # Never exempt entire branches just because they share a node.
                    if d1 <= 2*max(r1, r2) and d2 <= 2*max(r1, r2):
                        junction = True
                if junction:
                    continue
            out.append(Contact(sid, tid, i, j, u, v, float(distance-ra-rb),
                               pa.tolist(), pb.tolist()))
    return sorted(out, key=lambda c: (c.clearance_um, c.segment_a, c.edge_a,
                                      c.segment_b, c.edge_b))


def _bump(s, centre, width):
    """C2 displacement support with zero value/slope/curvature at its boundary."""
    t = (s-centre)/max(width, 1e-12)
    return np.maximum(1-t*t, 0.)**3


def tight_bends(x, radius):
    """Local circumcircle curvature flags tube folds missed by nonlocal pairs."""
    a, b = np.diff(x, axis=0)[:-1], np.diff(x, axis=0)[1:]
    denominator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)*np.linalg.norm(a+b, axis=1)
    curvature = 2*np.linalg.norm(np.cross(a, b), axis=1)/np.maximum(denominator, 1e-20)
    return curvature*np.asarray(radius)[1:-1] >= .95


def _field(x, arc_at, width):
    s = arclength(x)
    # Endpoint taper also pins root and shared junction coordinates. Clearance
    # changes branch interiors only, after joint junction refinement has finished.
    support = _bump(s, arc_at, width)
    taper = np.minimum(s/max(width, 1e-12), 1.)
    taper = np.minimum(taper, np.minimum((s[-1]-s)/max(width, 1e-12), 1.))
    taper = taper**3*(10-15*taper+6*taper*taper)
    return support*taper


def prepare(graph, frame, labels, *, max_iterations=30, gap_um=0.,
            max_displacement_radii=.5, progress=None):
    """Reduce existing collisions with whole-span smooth fields, never point pushes.

    Contact constraints drive a minimum-norm coefficient solve. Every accepted
    step reduces maximum penetration and passes segmentation and reversal checks.
    Fixed radii, finite displacement, and containment can make clearance infeasible.
    In that case the report remains unresolved.
    """
    if (max_iterations < 1 or max_displacement_radii <= 0
            or not np.isfinite(max_displacement_radii)):
        raise ValueError("invalid clearance parameters")
    t0 = time.monotonic()
    sampler = _PlaneSampler(labels, frame)
    original = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    radii = {sid: graph.radii(sid).copy() for sid in graph.segment_ids()}
    current = {sid: x.copy() for sid, x in original.items()}
    active = contacts(graph, coords=current, gap_um=gap_um)
    report = ClearanceReport(contacts_before=len(active))
    report.tight_bends_before = sum(int(tight_bends(x, radii[sid]).sum())
                                    for sid, x in current.items())
    for iteration in range(max_iterations):
        if not active:
            break
        # Deduplicate dense edge-pair contacts into one constraint per physical
        # neighbourhood. Fields overlap smoothly, so no isolated point is moved.
        selected = []
        for c in active:
            if any(c.segment_a == d.segment_a and c.segment_b == d.segment_b
                   and np.linalg.norm(np.array(c.position_a)-d.position_a)
                   < max(float(radii[c.segment_a][c.edge_a]), frame.seg_spacing[0])
                   for d in selected):
                continue
            selected.append(c)
            if len(selected) == 64:
                break
        fields, rows, needs = [], [], []
        for c in selected:
            pa, pb = np.array(c.position_a), np.array(c.position_b)
            direction = pa-pb
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                xa, xb = current[c.segment_a], current[c.segment_b]
                direction = np.cross(xa[c.edge_a+1]-xa[c.edge_a],
                                     xb[c.edge_b+1]-xb[c.edge_b])
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    tangent = xa[c.edge_a+1]-xa[c.edge_a]
                    axis = np.eye(3)[np.argmin(np.abs(tangent))]
                    direction = np.cross(tangent, axis)
                    norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction /= norm
            for sid, i, f in ((c.segment_a, c.edge_a, c.fraction_a),
                              (c.segment_b, c.edge_b, c.fraction_b)):
                arc = arclength(current[sid])
                centre = arc[i]+f*(arc[i+1]-arc[i])
                width = max(4*float(radii[sid][i]), 6*float(frame.seg_spacing[0]))
                fields.append((sid, _field(current[sid], centre, width)))
            rows.append((c, direction))
            needs.append(gap_um-c.clearance_um + .01*float(frame.seg_spacing[0]))
        if not fields:
            break
        a = np.zeros((len(rows), len(fields)*3))
        for row, (c, direction) in enumerate(rows):
            for col, (sid, values) in enumerate(fields):
                coefficient = 0.
                if sid == c.segment_a:
                    coefficient += (1-c.fraction_a)*values[c.edge_a]+c.fraction_a*values[c.edge_a+1]
                if sid == c.segment_b:
                    coefficient -= (1-c.fraction_b)*values[c.edge_b]+c.fraction_b*values[c.edge_b+1]
                a[row, 3*col:3*col+3] = coefficient*direction
        coeff = a.T @ np.linalg.solve(a @ a.T + .01*np.eye(len(rows)), needs)
        delta = {sid: np.zeros_like(x) for sid, x in current.items()}
        for j, (sid, values) in enumerate(fields):
            delta[sid] += values[:, None]*coeff[3*j:3*j+3]
        involved = {sid for sid, _ in fields}
        # Whole-field backtracking per branch prevents one immovable collision
        # freezing unrelated, feasible repairs elsewhere in the graph.
        for sid in involved:
            beta = 1.
            while beta >= 1/128:
                proposal = current[sid]+beta*delta[sid]
                if (np.all(np.linalg.norm(proposal-original[sid], axis=1)
                           <= max_displacement_radii*radii[sid]+1e-7)
                        and feasible_move(current[sid], proposal, sampler, frame)
                        and not np.any(tight_bends(proposal, radii[sid])
                                       & ~tight_bends(current[sid], radii[sid]))):
                    break
                beta /= 2
            delta[sid] *= beta if beta >= 1/128 else 0.
        accepted = False
        old_peak = max(gap_um-c.clearance_um for c in active)
        old_energy = sum((gap_um-c.clearance_um)**2 for c in active)
        old_pairs = {(c.segment_a, c.segment_b) for c in active}
        for alpha in (.75, .375, .1875, .09375, .046875, .0234375, .01171875):
            trial = {sid: x+alpha*delta[sid] for sid, x in current.items()}
            if any(np.any(np.linalg.norm(trial[sid]-original[sid], axis=1)
                          > max_displacement_radii*radii[sid]+1e-7) for sid in involved):
                continue
            if not all(feasible_move(current[sid], trial[sid], sampler, frame)
                       for sid in involved):
                continue
            next_contacts = contacts(graph, coords=trial, gap_um=gap_um)
            peak = max((gap_um-c.clearance_um for c in next_contacts), default=0.)
            energy = sum((gap_um-c.clearance_um)**2 for c in next_contacts)
            if peak > old_peak+1e-6 or energy >= old_energy-1e-6 or any((c.segment_a, c.segment_b) not in old_pairs
                                          for c in next_contacts):
                continue
            current, active = trial, next_contacts
            accepted = True
            break
        report.iterations = iteration+1
        report.history.append(dict(iteration=iteration+1, contacts=len(active), accepted=accepted))
        if progress:
            progress(report.history[-1])
        if not accepted:
            break
    with graph.batch("smooth reconstruction clearance"):
        for sid, x in current.items():
            graph.set_segment_coords(sid, x)
            report.moved_points += int((np.linalg.norm(x-original[sid], axis=1) > 1e-7).sum())
            np.testing.assert_array_equal(graph.radii(sid), radii[sid])
            report.outside_edges += int(bad_edges(x, sampler, frame).sum())
            report.tight_bends_after += int(tight_bends(x, radii[sid]).sum())
    report.contacts_after = len(active)
    report.remaining = [asdict(c) for c in active]
    report.status = "capsule-clear-surface-unvalidated" if not active else "unresolved"
    if report.outside_edges:
        report.status = "unresolved-containment"
    elif report.tight_bends_after:
        report.status = "unresolved-tight-bends"
    report.seconds = time.monotonic()-t0
    return report
