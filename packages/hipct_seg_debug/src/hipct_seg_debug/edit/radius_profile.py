"""Derived reconstruction radii: trusted anchors are immutable measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

from .centreline_refine import arclength
from .radius_perimeter import ACCEPTED, FILLED

UNCHANGED, GAP, CONTINUATION, CONFLICT, UNSUPPORTED, JUNCTION_EXTENSION = range(6)


@dataclass
class ProfileReport:
    policy: str
    changed_points: int = 0
    conflicts: list = field(default_factory=list)
    unsupported_segments: list = field(default_factory=list)
    junctions: list = field(default_factory=list)
    discontinuities: list = field(default_factory=list)
    status: str = "prepared"

    def to_dict(self):
        return asdict(self)


def trusted_radii(graph, sid):
    ids = graph.segment(sid)['point_ids']
    attrs = graph.triple.point_attrs
    reject = attrs.get('radius_reject_reason', {})
    source = attrs.get('radius_source', {})
    radius = graph.radii(sid)
    return np.array([reject.get(pid) == ACCEPTED and source.get(pid, FILLED) != FILLED
                     for pid in ids]) & np.isfinite(radius) & (radius > 0)


def interpolate_supported(s, radius, trusted):
    """No extrapolation or overshoot; accepted samples remain bitwise unchanged."""
    out = np.asarray(radius).copy()
    changed = np.zeros(len(out), dtype=bool)
    good = np.flatnonzero(trusted)
    if len(good) < 2:
        return out, changed
    unique, indices = np.unique(s[good], return_index=True)
    if len(unique) < 2:
        return out, changed
    good = good[indices]
    use = ~trusted & (s >= unique[0]) & (s <= unique[-1])
    out[use] = np.exp(PchipInterpolator(unique, np.log(radius[good]))(s[use]))
    # Explicit interval bounds guard rounding and future interpolator changes.
    right = np.clip(np.searchsorted(unique, s[use], side='right'), 1, len(unique)-1)
    lo = np.minimum(radius[good[right-1]], radius[good[right]])
    hi = np.maximum(radius[good[right-1]], radius[good[right]])
    out[use] = np.clip(out[use], lo, hi)
    # Support is still valid when interpolation reproduces the measured file's
    # previous fill exactly. It must not be reported as an unsupported span.
    changed[use] = True
    return out, changed


def prepare_profile(graph, *, policy="preserve", spacing_um=1., transition_radii=4.):
    if policy not in ('preserve', 'confidence'):
        raise ValueError('radius profile must be preserve or confidence')
    if spacing_um <= 0 or transition_radii <= 0:
        raise ValueError('profile lengths must be positive')
    report = ProfileReport(policy)
    measured = graph.triple.point_attrs.get('radius_measured_um', {})
    if policy == 'preserve' and measured:
        return report
    raw = {sid: np.array([measured.get(pid, graph.points[pid][3])
                         for pid in graph.segment(sid)['point_ids']], dtype=float)
           for sid in graph.segment_ids()}
    derived = {sid: r.copy() for sid, r in raw.items()}
    trusted = {sid: trusted_radii(graph, sid) for sid in raw}
    reasons = {sid: np.zeros(len(r), dtype=int) for sid, r in raw.items()}
    if policy == 'confidence':
        for sid in raw:
            s = arclength(graph.coords(sid))
            relative = np.abs(np.diff(np.log(np.maximum(raw[sid], 1e-12))))
            local = np.maximum(np.minimum(raw[sid][:-1], raw[sid][1:]), spacing_um)
            rate = relative*local/np.maximum(np.diff(s), spacing_um)
            for i in np.flatnonzero((relative > np.log(1.03)) & (rate > .1)):
                report.discontinuities.append(dict(segment=sid, points=[int(i), int(i+1)],
                    classification='supported_calibre_change' if trusted[sid][i:i+2].all()
                                   else 'within_vessel_noise_candidate',
                    measured_radii_um=raw[sid][i:i+2].tolist()))
            derived[sid], changed = interpolate_supported(s, raw[sid], trusted[sid])
            reasons[sid][changed] = GAP
            if not trusted[sid].all() and trusted[sid].sum() < 2:
                report.unsupported_segments.append(sid)
                reasons[sid][~trusted[sid]] = UNSUPPORTED
        for nid in sorted(graph.nodes):
            incident = sorted(graph.node_segments(nid))
            if len(incident) >= 3:
                report.junctions.append(dict(node=nid, segments=incident,
                                            classification='branch_calibre_difference'))
                # There is no exclusive measured circle at the shared node.
                # Transport each branch's own nearest trusted calibre only over
                # a bounded unsupported junction run; never average daughters.
                for sid in incident:
                    order = np.arange(len(raw[sid]))
                    if graph.segment(sid)['node2'] == nid:
                        order = order[::-1]
                    good = np.flatnonzero(trusted[sid][order])
                    if not len(good):
                        continue
                    first = int(good[0])
                    anchor = raw[sid][order[first]]
                    s = arclength(graph.coords(sid)[order])
                    if s[first] <= max(8*spacing_um, transition_radii*anchor):
                        derived[sid][order[:first]] = anchor
                        reasons[sid][order[:first]] = JUNCTION_EXTENSION
                continue
            if len(incident) != 2:
                continue
            a, b = incident
            ia, ib = np.arange(len(raw[a])), np.arange(len(raw[b]))
            if graph.segment(a)['node1'] == nid:
                ia = ia[::-1]
            if graph.segment(b)['node2'] == nid:
                ib = ib[::-1]
            if not len(ia) or not len(ib):
                continue
            r1, r2 = raw[a][ia[-1]], raw[b][ib[0]]
            if not np.isclose(r1, r2, rtol=1e-8):
                report.discontinuities.append(dict(node=nid, segments=incident,
                    classification='conflicting_trusted_degree_two_boundary'
                    if trusted[a][ia[-1]] and trusted[b][ib[0]] else 'artificial_degree_two_boundary_step'))
            if trusted[a][ia[-1]] and trusted[b][ib[0]] and not np.isclose(r1, r2, rtol=1e-8):
                report.conflicts.append(dict(node=nid, segments=incident,
                                             radii_um=[float(r1), float(r2)],
                                             reason='conflicting_trusted_anchors'))
                reasons[a][ia[-1]], reasons[b][ib[0]] = CONFLICT, CONFLICT
                continue
            sa = arclength(graph.coords(a)[ia])
            sb = arclength(graph.coords(b)[ib])
            ga, gb = ia[trusted[a][ia]], ib[trusted[b][ib]]
            anchor_scales = ([raw[a][ga[-1]]] if len(ga) else []) + (
                [raw[b][gb[0]]] if len(gb) else [])
            local = max(spacing_um*2, min(anchor_scales)) if anchor_scales else spacing_um*2
            length = max(8*spacing_um, transition_radii*local)
            ia = ia[sa >= sa[-1]-length]
            ib = ib[sb <= length]
            # Keep both node records for provenance, but consolidate their
            # interpolation anchor when one is trusted and the other is not.
            coords = np.vstack([graph.coords(a)[ia], graph.coords(b)[ib]])
            s = arclength(coords)
            values = np.r_[raw[a][ia], raw[b][ib]]
            good = np.r_[trusted[a][ia], trusted[b][ib]]
            joined, changed = interpolate_supported(s, values, good)
            for sid, ids, part in ((a, ia, slice(0, len(ia))), (b, ib, slice(len(ia), None))):
                use = changed[part]
                derived[sid][ids[use]] = joined[part][use]
                reasons[sid][ids[use]] = CONTINUATION
            if not np.isclose(derived[a][ia[-1]], derived[b][ib[0]], rtol=1e-8):
                report.conflicts.append(dict(node=nid, segments=incident,
                                             reason='unsupported_continuation'))
    if policy == 'confidence':
        report.unsupported_segments = [sid for sid in raw if np.any(
            ~trusted[sid] & np.isin(reasons[sid], [UNCHANGED, UNSUPPORTED]))]
    attrs = graph.triple.point_attrs
    with graph.batch('derived reconstruction radius profile'):
        for sid, radii in derived.items():
            np.testing.assert_array_equal(radii[trusted[sid]], raw[sid][trusted[sid]])
            graph.set_segment_radii(sid, radii)
            report.changed_points += int(np.count_nonzero(radii != raw[sid]))
            ids = graph.segment(sid)['point_ids']
            for name, values, dtype in (
                ('radius_measured_um', raw[sid], 'float'),
                ('radius_reconstruction_um', radii, 'float'),
                ('radius_adjustment_um', radii-raw[sid], 'float'),
                ('radius_adjustment_reason', reasons[sid], 'int'),
                ('radius_anchor_trusted', trusted[sid].astype(int), 'int')):
                attrs.setdefault(name, {}).update(dict(zip(ids, values.tolist())))
                graph.triple.point_attr_dtypes[name] = dtype
    if report.conflicts or report.unsupported_segments:
        report.status = 'review_required'
    from .optimise import set_edge_field
    set_edge_field(graph, 'MeanRadius', np.array([
        float(np.mean(graph.radii(sid))) if len(graph.radii(sid)) else np.nan
        for sid in graph.segment_ids()]), np.float64)
    return report
