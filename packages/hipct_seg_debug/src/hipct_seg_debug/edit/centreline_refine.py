"""Opt-in, segmentation-constrained centreline fitting.

Geometry refinement and reconstruction clearance are separate operations. This
module never changes radii or topology. All distances are in WorldFrame micrometres.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import threading
import time

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve

from ..crosssection import _PlaneSampler, robust_edge_tangents, stable_transverse_cut
from .interpolation import mask_for_segment
from .radius_perimeter import _BranchContext, _rival_lies_in_blob

METHODS = ("none", "centroid-spline", "centroid-coherent", "laplacian", "taubin")


def arclength(x):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]


def line_samples_ijk(a, b):
    """One point in every nearest-neighbour voxel crossed by a straight edge.

    Fixed-distance sampling can miss a corner clipped between two samples. Split
    at every half-integer plane instead. Boundary points use the sampler's same
    rounding convention; endpoints are included too.
    """
    a, b = np.asarray(a), np.asarray(b)
    d = b-a
    ts = [np.array([0., 1.])]
    for k in range(3):
        if abs(d[k]) < 1e-12:
            continue
        lo, hi = sorted((a[k], b[k]))
        planes = np.arange(np.ceil(lo-.5), np.floor(hi-.5)+1) + .5
        t = (planes-a[k])/d[k]
        ts.append(t[(t > 0) & (t < 1)])
    t = np.unique(np.concatenate(ts))
    t = np.unique(np.r_[t, (t[1:]+t[:-1])/2])
    return a + t[:, None]*d


def bad_edges(x, sampler, frame):
    ijk = frame.um_to_seg(x)
    return np.array([not np.all(sampler.at(line_samples_ijk(a, b)) > 0)
                     for a, b in zip(ijk[:-1], ijk[1:])], dtype=bool)


def _reversals(x):
    d = np.diff(x, axis=0)
    length = np.linalg.norm(d, axis=1)
    unit = d / np.maximum(length[:, None], 1e-12)
    return np.sum(unit[:-1]*unit[1:], axis=1) < -.5


def feasible_move(old, new, sampler, frame):
    """No new bad edges, no motion of pre-existing bad edges, no new reversals.

    Each point's movement must also remain in foreground, preventing a proposal
    from jumping across background to a different vessel. Existing gaps remain
    explicitly unresolved rather than being silently repaired.
    """
    if not np.isfinite(new).all():
        return False
    original_bad = bad_edges(old, sampler, frame)
    if np.any(bad_edges(new, sampler, frame) & ~original_bad):
        return False
    if original_bad.any():
        fixed = np.r_[original_bad, False] | np.r_[False, original_bad]
        if np.any(np.linalg.norm(new[fixed]-old[fixed], axis=1) > 1e-7):
            return False
    if np.any(_reversals(new) & ~_reversals(old)):
        return False
    if np.any(np.linalg.norm(np.diff(new, axis=0), axis=1) < 1e-8):
        return False
    before, after = frame.um_to_seg(old), frame.um_to_seg(new)
    moved = np.linalg.norm(after-before, axis=1) > 1e-8
    return all(np.all(sampler.at(line_samples_ijk(a, b)) > 0)
               for a, b in zip(before[moved], after[moved]))


def spline_fit(x, target, weights, scale, spacing, strength=.1, ends=None):
    """Cubic smoothing fit in physical arclength, with exact shared endpoints.

    Centre observations use arclength quadrature so adding sample points does not
    increase their influence. Knot spacing and bending weights follow a measured
    section scale, never the original radius-based displacement cap.
    """
    n = len(x)
    if n < 4:
        return x.copy()
    s = arclength(x)
    if s[-1] <= 1e-9 or np.any(np.diff(s) <= 1e-9):
        return x.copy()
    local_scale = np.maximum(np.broadcast_to(scale, (n,)), 2*spacing)
    xi = np.r_[0., np.cumsum(np.diff(s)/np.maximum(
        (local_scale[:-1]+local_scale[1:])/2, 4*spacing))]
    internal = np.interp(np.arange(.5, xi[-1], .5), xi, s)
    if len(internal) > n-4:
        internal = internal[np.linspace(0, len(internal)-1, n-4, dtype=int)]
    count = len(internal)+4
    knots = np.r_[[0.]*4, internal, [s[-1]]*4]
    basis = BSpline.design_matrix(s, knots, 3).tocsr()
    # Integrate the squared second derivative with physical quadrature.
    spl = BSpline(knots, np.eye(count), 3)
    grid = np.linspace(0, s[-1], max(16, 2*count))
    bending = sparse.csr_matrix(spl.derivative(2)(grid))
    q = np.r_[np.diff(s)[0]/2, (s[2:]-s[:-2])/2, np.diff(s)[-1]/2]
    w = np.asarray(weights)*q
    # Weak support where the segmentation could not supply a measurement.
    w = np.maximum(w, .005*q)
    h = basis.T @ sparse.diags(w) @ basis
    local_grid = np.interp(grid, s, local_scale)
    h += strength*(grid[1]-grid[0])*(bending.T @ sparse.diags(local_grid**4) @ bending)
    rhs = basis.T @ (w[:, None]*target)
    fixed = np.array([0, count-1])
    end = x[[0, -1]] if ends is None else np.asarray(ends)
    free = np.arange(1, count-1)
    coeff = np.empty((count, 3))
    coeff[fixed] = end
    coeff[free] = spsolve(h[free][:, free].tocsc(), rhs[free]-h[free][:, fixed] @ end)
    out = basis @ coeff
    out[[0, -1]] = end
    return out


def laplacian_fit(x, scale, spacing, *, taubin=False, strength=.1):
    """Implicit arclength diffusion; optional two-pass shrinkage compensation.

    A finite-volume Laplacian handles unequal edge lengths. The local scale is
    measured from segmentation sections and gives diffusion time physical units.
    Both variants are candidates, not presumed geometry-preserving solutions.
    """
    if len(x) < 3:
        return x.copy()
    step = np.maximum(np.diff(arclength(x)), 1e-8)
    mass = np.r_[step[0]/2, (step[:-1]+step[1:])/2, step[-1]/2]
    local_scale = np.maximum(np.broadcast_to(scale, (len(x),)), 2*spacing)
    conductance = (local_scale[:-1]**2+local_scale[1:]**2)/(2*step)
    stiffness = sparse.diags(np.r_[conductance, 0]+np.r_[0, conductance])
    stiffness += sparse.diags(-conductance, 1)+sparse.diags(-conductance, -1)
    op = sparse.diags(1/mass) @ stiffness
    tau = strength
    h = (sparse.eye(len(x)) + tau*op).tolil()
    h[0, :] = 0
    h[0, 0] = 1
    h[-1, :] = 0
    h[-1, -1] = 1
    smooth = spsolve(h.tocsc(), x)
    if taubin:
        # Two stable implicit low-pass passes form an unsharp compensation,
        # cancelling the leading diffusion shrinkage term (2H-H^2).
        smooth = 2*smooth-spsolve(h.tocsc(), smooth)
    smooth[[0, -1]] = x[[0, -1]]
    return smooth


@dataclass
class RefinementReport:
    method: str
    iterations: int = 0
    converged: bool = False
    moved_points: int = 0
    moved_nodes: int = 0
    seconds: float = 0.
    outside_edges_before: int = 0
    outside_edges_after: int = 0
    radii_require_remeasurement: bool = False
    history: list = field(default_factory=list)
    segments: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _open_worker(labels, frame, local):
    from ..rle import ByteRLELattice, RawLattice, open_lattice
    if isinstance(labels, (ByteRLELattice, RawLattice)):
        labels = open_lattice(labels.path, labels.field, labels.dims,
                              cache_dir=getattr(labels, "_cache_dir", None))
    local.sampler = _PlaneSampler(labels, frame)


def _crosses_section(graph, sid, origin, tangent, cut, spacing):
    """A rival must actually intersect this plane, not merely project into it.

    A neighbouring continuation can be a radius away axially and still project
    into the same blob. Treating that projection as a merge masks entire short
    segments of otherwise unbranched lumen.
    """
    x = graph.coords(sid)
    axial = (x-origin) @ tangent
    for i in np.flatnonzero(axial[:-1]*axial[1:] <= 0):
        den = axial[i]-axial[i+1]
        f = axial[i]/den if abs(den) > 1e-12 else .5
        point = x[i]+f*(x[i+1]-x[i])
        if _rival_lies_in_blob(cut, origin, point, spacing):
            return True
    return False


def _targets(graph, sid, frame, sampler, ctx, scale, max_half, max_samples,
             coherent=False):
    x = graph.coords(sid)
    n = len(x)
    target, weights = x.copy(), np.zeros(n)
    sp = float(frame.seg_spacing[0])
    s = arclength(x)
    scale = np.broadcast_to(scale, (n,))
    tangents = robust_edge_tangents(x, scale, spacing_um=sp)
    ids = np.unique(np.searchsorted(s, np.linspace(0., s[-1], min(n, max_samples))))
    invented = mask_for_segment(graph, sid)
    accepted, measured_scale = [], []
    for i in ids:
        if len(invented) == n and invented[i]:
            continue
        chosen = stable_transverse_cut(
            sampler, frame.um_to_seg(x[i])[0], tangents[i], max(scale[i]/sp, 2.),
            spacing_um=sp, max_half=max_half, centroid_mode="drift",
            **({"transverse_axis_ratio": np.inf} if coherent else {}),
            slab_offsets=(0., .25, .5) if i == 0 else (
                (-.5, -.25, 0.) if i == n-1 else (-.5, 0., .5)),
        )
        if chosen is None:
            continue
        c = chosen.cut
        # Do not recenter a merged branch onto the combined lumen's centroid.
        # Unresolved regions receive curve support from exclusive sections.
        rivals = ctx.rivals(sid, x[i], chosen.tangent, max(scale[i], sp))
        if any(_crosses_section(graph, int(item[0]), x[i], chosen.tangent, c, sp)
               for item in rivals):
            continue
        centre = np.argwhere(c.blob8).mean(axis=0)-c.half
        candidate = x[i] + sp*(centre[0]*c.u + centre[1]*c.v)
        if not np.all(sampler.at(line_samples_ijk(frame.um_to_seg(x[i])[0],
                                                 frame.um_to_seg(candidate)[0])) > 0):
            continue
        target[i] = candidate
        weights[i] = 1/(1+chosen.centroid_ratio**2)
        accepted.append(int(i))
        measured_scale.append(float(np.sqrt(c.blob8.sum()/np.pi)*sp))
    return target, weights, accepted, measured_scale


def refine(graph, frame, labels, *, method="centroid-spline", sids=None,
           fixed_nodes=(), move_junctions=True, strength=.1, max_iterations=25,
           max_half=256, max_samples=32, workers=1, progress=None):
    """Refine selected segments with full-graph branch context, preserving radii.

    Junctions move only if every incident segment participates and provides two
    trusted sections. Roots, terminals and unsupported junctions remain anchored.
    """
    if method not in METHODS:
        raise ValueError(f"unknown refinement method: {method}")
    if (workers < 1 or max_iterations < 1 or max_samples < 4 or max_half < 4
            or not np.isfinite(strength) or strength <= 0):
        raise ValueError("invalid refinement parameters")
    if not np.allclose(frame.seg_spacing, frame.seg_spacing[0], rtol=1e-6):
        raise ValueError("cross-section refinement currently requires isotropic voxels")
    t0 = time.monotonic()
    sids = list(graph.segment_ids()) if sids is None else sorted(set(map(int, sids)))
    if any(not graph.has_segment(sid) for sid in sids):
        raise ValueError("unknown segment id")
    sp = float(frame.seg_spacing[0])
    original = {sid: graph.coords(sid).copy() for sid in sids}
    radius_snapshot = {sid: graph.radii(sid).copy() for sid in graph.segment_ids()}
    sampler = _PlaneSampler(labels, frame)
    report = RefinementReport(method)
    report.outside_edges_before = sum(int(bad_edges(x, sampler, frame).sum())
                                      for x in original.values())
    scale = {sid: np.maximum(2*sp, graph.radii(sid).copy()) for sid in sids}
    held = set(fixed_nodes) | {nid for nid in graph.nodes if graph.degree(nid) == 1}
    local = threading.local()
    stable_rounds = 0
    with ThreadPoolExecutor(max_workers=workers, initializer=_open_worker,
                            initargs=(labels, frame, local)) as pool:
        for iteration in range(max_iterations if method != "none" else 0):
            ctx = _BranchContext.build(graph)
            before = {sid: graph.coords(sid).copy() for sid in sids}

            def work(sid):
                if len(before[sid]) < 4:
                    return sid, (before[sid].copy(), np.zeros(len(before[sid])), [], [])
                return sid, _targets(graph, sid, frame, local.sampler, ctx, scale[sid],
                                     max_half, max_samples, method == "centroid-coherent")

            observations = dict(pool.map(work, sids))
            # One least-squares intersection of the supported incident centre rays.
            node_targets = {}
            if move_junctions and method.startswith("centroid-"):
                for nid in graph.nodes:
                    incident = list(graph.node_segments(nid))
                    if nid in held or len(incident) < 2 or not set(incident) <= set(sids):
                        continue
                    matrices, rhs = [], []
                    for sid in incident:
                        target, weights, good, _ = observations[sid]
                        if len(good) < 2:
                            break
                        near = good[:2] if graph.segment(sid)["node1"] == nid else good[-2:]
                        p, q = target[near]
                        tangent = q-p
                        norm = np.linalg.norm(tangent)
                        if norm < sp*.1:
                            break
                        tangent /= norm
                        projection = np.eye(3)-np.outer(tangent, tangent)
                        matrices.append(projection)
                        rhs.append(projection @ p)
                    if len(matrices) != len(incident):
                        continue
                    old_node = np.asarray(graph.nodes[nid][:3])
                    mat = sum(matrices)+.05*np.eye(3)
                    candidate = np.linalg.solve(mat, sum(rhs)+.05*old_node)
                    if np.all(sampler.at(line_samples_ijk(frame.um_to_seg(old_node)[0],
                                                         frame.um_to_seg(candidate)[0])) > 0):
                        node_targets[nid] = candidate

            proposals = {}
            accepted_total = 0
            for sid in sids:
                x = before[sid]
                target, weights, good, measured = observations[sid]
                accepted_total += len(good)
                if measured:
                    scale[sid] = np.maximum(2*sp, np.interp(
                        arclength(x), arclength(x)[good], measured))
                report.segments[sid] = dict(accepted_sections=len(good),
                                            sampled_points=min(len(x), max_samples),
                                            section_scale_um=float(np.median(scale[sid])))
                if len(good) < 2 or len(x) < 4:
                    proposals[sid] = x.copy()
                    continue
                if method.startswith("centroid-"):
                    seg = graph.segment(sid)
                    ends = [node_targets.get(seg[k], x[i])
                            for k, i in (("node1", 0), ("node2", -1))]
                    # Huber reweighting in voxel units avoids one centroid dominating.
                    residual = np.linalg.norm(target-x, axis=1)
                    # The first pass must be able to correct a sustained large offset.
                    typical = max(sp, float(np.median(residual[weights > 0])))
                    weights *= np.minimum(1., 2*typical/np.maximum(residual, sp))
                    proposals[sid] = spline_fit(x, target, weights, scale[sid], sp,
                                                strength, ends)
                else:
                    proposals[sid] = laplacian_fit(x, scale[sid], sp,
                                                   taubin=method == "taubin", strength=strength)

            # Connected groups share a single line-search step so junction endpoints
            # cannot split. With fixed nodes each segment can backtrack independently.
            groups = [{sid} for sid in sids]
            for nid in node_targets:
                incident = set(graph.node_segments(nid))
                touched = [g for g in groups if g & incident]
                groups = [g for g in groups if not g & incident]
                groups.append(set().union(*touched))
            moves = []
            blocked = 0
            with graph.batch("segmentation-constrained centreline refinement"):
                for group in groups:
                    alpha = .75
                    while alpha >= 1/256:
                        trial = {sid: before[sid]+alpha*(proposals[sid]-before[sid])
                                 for sid in group}
                        if all(feasible_move(before[sid], trial[sid], sampler, frame)
                               for sid in group):
                            break
                        alpha /= 2
                    if alpha < 1/256:
                        blocked += len(group)
                        continue
                    for sid in sorted(group):
                        moves.extend(np.linalg.norm(trial[sid]-before[sid], axis=1))
                        graph.set_segment_coords(sid, trial[sid])
            peak = float(max(moves, default=0.))
            report.iterations = iteration+1
            report.history.append(dict(iteration=iteration+1, max_move_um=peak,
                                       accepted_sections=accepted_total, blocked_segments=blocked))
            if progress:
                progress(report.history[-1])
            stable_rounds = stable_rounds+1 if peak < .1*sp and blocked == 0 else 0
            if stable_rounds >= 2:
                report.converged = True
                break
            if peak < 1e-9:
                break
    for sid in sids:
        x = graph.coords(sid)
        dist = np.linalg.norm(x-original[sid], axis=1)
        report.moved_points += int((dist > 1e-7).sum())
        report.outside_edges_after += int(bad_edges(x, sampler, frame).sum())
        report.segments.setdefault(sid, {}).update(
            median_move_um=float(np.median(dist)) if len(dist) else 0.,
            max_move_um=float(max(dist, default=0.)))
    report.moved_nodes = len({graph.segment(sid)[key] for sid in sids
                             for key, i in (("node1", 0), ("node2", -1))
                             if len(original[sid]) and np.linalg.norm(
                                 graph.coords(sid)[i]-original[sid][i]) > 1e-7})
    for sid, radius in radius_snapshot.items():
        np.testing.assert_array_equal(graph.radii(sid), radius)
    report.radii_require_remeasurement = report.moved_points > 0
    report.seconds = time.monotonic()-t0
    return report
