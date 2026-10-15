"""Joint physical-arclength fitting of overlapping junction neighbourhoods."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

#: Re-solves after pinning vertices whose edges would newly leave the segmentation.
PIN_ROUNDS = 4


def neighbourhood_clusters(graph, nodes, observations, scales, spacing):
    """Merge overlapping incident spans before fitting; schedule by node ID."""
    from .centreline_refine import arclength
    spans = {}
    parent = {nid: nid for nid in nodes}
    def root(nid):
        while parent[nid] != nid:
            parent[nid] = parent[parent[nid]]
            nid = parent[nid]
        return nid
    for nid in sorted(nodes):
        spans[nid] = {}
        for sid in sorted(graph.node_segments(nid)):
            x = graph.coords(sid)
            order = np.arange(len(x))
            if graph.segment(sid)['node2'] == nid:
                order = order[::-1]
            s = arclength(x[order])
            limit = max(6*spacing, 4*scales[sid][order[0]])
            count = min(len(x), max(4, int(np.searchsorted(s, limit))+1))
            good = np.flatnonzero(np.isin(order, observations[sid][2]))
            # Place the outer anchor beyond two exclusive support sections.
            if len(good) >= 2:
                count = min(len(x), max(count, int(good[1])+3))
            spans[nid][sid] = (int(order[:count].min()), int(order[:count].max()))
    for sid in sorted(scales):
        seg = graph.segment(sid)
        a, b = seg['node1'], seg['node2']
        if a not in spans or b not in spans:
            continue
        left, right = spans[a][sid], spans[b][sid]
        if max(left[0], right[0]) <= min(left[1], right[1])+1:
            ra, rb = root(a), root(b)
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for nid in sorted(nodes):
        groups.setdefault(root(nid), []).append(nid)
    return [(group, {sid: (min(spans[n][sid][0] for n in group if sid in spans[n]),
                           max(spans[n][sid][1] for n in group if sid in spans[n]))
                     for sid in sorted({s for n in group for s in spans[n]})})
            for group in groups.values()]


def fit_cluster(graph, nodes, spans, observations, scales, sampler, frame, strength,
                *, require_external_support=True, section_weighting=False):
    """One quadratic solve with shared node variables and fixed outer tangents.

    Unsupported internal links receive support through both jointly fitted nodes.
    By default every external approach must supply two exclusive sections. DFS
    callers may relax this after checking support along the through path; missing
    side-approach support is still reported. Branch directions
    have separate soft derivative observations; degree-two derivatives are shared
    exactly by a linear equality constraint. A single containment line search
    accepts or refuses the entire neighbourhood.
    """
    from .centreline_refine import arclength, movement_rejection, bad_edges
    sp = float(frame.seg_spacing[0])
    nodes = set(nodes)
    old = {sid: graph.coords(sid).copy() for sid in spans}
    unsupported = [sid for sid in spans if not {
        graph.segment(sid)['node1'], graph.segment(sid)['node2']} <= nodes
        and len(observations[sid][2]) < 2]
    report = dict(nodes=sorted(nodes), segments=sorted(spans), unsupported_approaches=unsupported,
                  spans={sid: [int(lo), int(hi)] for sid, (lo, hi) in spans.items()})
    if unsupported and require_external_support:
        return dict(report, status='insufficient_support')
    original_bad = {sid: bad_edges(x, sampler, frame) for sid, x in old.items()}
    keys, coordinates, lookup, indices, fixed = {}, [], {}, {}, set()
    for sid, (lo, hi) in spans.items():
        seg = graph.segment(sid)
        ids = np.arange(lo, hi+1)
        local = []
        for i in ids:
            nid = seg['node1'] if i == 0 else seg['node2'] if i == len(old[sid])-1 else None
            key = ('node', nid) if nid is not None else ('point', sid, int(i))
            if key not in keys:
                keys[key] = len(coordinates)
                coordinates.append(old[sid][i])
            local.append(keys[key])
            lookup[sid, int(i)] = keys[key]
        indices[sid] = np.array(local)
        bad = original_bad[sid]
        immobile = np.r_[bad, False] | np.r_[False, bad]
        fixed.update(local[j] for j, i in enumerate(ids) if immobile[i])
        if lo > 0 or seg['node1'] not in nodes:
            fixed.update(local[:2])
        if hi < len(old[sid])-1 or seg['node2'] not in nodes:
            fixed.update(local[-2:])
    initial = np.asarray(coordinates)
    rows, cols, values, targets = [], [], [], []
    def add(coefficients, target, weight):
        row = len(targets)
        factor = np.sqrt(weight)
        for col, value in coefficients.items():
            rows.append(row)
            cols.append(col)
            values.append(value*factor)
        targets.append(np.asarray(target)*factor)
    for sid, (lo, hi) in spans.items():
        ids = np.arange(lo, hi+1)
        s = arclength(old[sid][ids])
        ds = np.diff(s)
        # A two-point link has no interior curvature term, but is a valid
        # connection between jointly fitted nodes. Reject only zero-length spans.
        if len(ds) < 1 or np.any(ds <= 1e-9):
            return dict(report, status='insufficient_support', reason='degenerate_span')
        q = np.r_[ds[0]/2, (ds[:-1]+ds[1:])/2, ds[-1]/2]
        support_weight = q.copy()
        if section_weighting:
            # A section represents its physical support interval, not the local
            # spacing of arbitrary exported centreline points. Bound its reach
            # by calibre so a long unsupported gap cannot create a huge anchor.
            good = np.flatnonzero(observations[sid][1][ids] > 0)
            if len(good):
                boundaries = np.r_[s[0], (s[good][:-1]+s[good][1:])/2, s[-1]]
                support_weight[good] = np.minimum(np.diff(boundaries),
                    4*np.maximum(2*sp, np.asarray(scales[sid])[ids[good]]))
        v = indices[sid]
        for j, i in enumerate(ids):
            add({v[j]: 1.}, observations[sid][0][i],
                max(.005, observations[sid][1][i])*support_weight[j])
        for j in range(1, len(ids)-1):
            calibre = max(2*sp, scales[sid][ids[j]])
            add({v[j-1]: 1/ds[j-1], v[j]: -1/ds[j-1]-1/ds[j], v[j+1]: 1/ds[j]},
                np.zeros(3), strength*calibre**4/q[j])
    constraints = []
    for nid in sorted(nodes):
        incident = sorted(graph.node_segments(nid))
        if len(incident) == 2:
            equation = {}
            for sid in incident:
                i, j = (0, 1) if graph.segment(sid)['node1'] == nid else (len(old[sid])-1, len(old[sid])-2)
                ds = max(np.linalg.norm(old[sid][j]-old[sid][i]), 1e-9)
                for col, value in ((lookup[sid, j], 1/ds), (lookup[sid, i], -1/ds)):
                    equation[col] = equation.get(col, 0.)+value
            constraints.append(equation)
        else:
            for sid in incident:
                order = np.arange(len(old[sid]))
                if graph.segment(sid)['node2'] == nid:
                    order = order[::-1]
                good = [i for i in order if i in observations[sid][2]]
                if len(good) < 2:
                    continue
                direction = observations[sid][0][good[1]]-observations[sid][0][good[0]]
                direction /= max(np.linalg.norm(direction), 1e-12)
                i, j = order[:2]
                ds = max(np.linalg.norm(old[sid][j]-old[sid][i]), 1e-9)
                calibre = max(2*sp, scales[sid][i])
                add({lookup[sid, j]: 1/ds, lookup[sid, i]: -1/ds}, direction,
                    strength*calibre**3)
    design = sparse.coo_matrix((values, (rows, cols)), shape=(len(targets), len(initial))).tocsr()
    target = np.asarray(targets)
    c = sparse.lil_matrix((len(constraints), len(initial)))
    for row, equation in enumerate(constraints):
        for col, value in equation.items():
            c[row, col] = value
    c = c.tocsr()
    def objective(x):
        # Include the degree-two kink in the acceptance objective when the input
        # violates its new equality constraint. Its physical weight is local.
        return float(np.sum((design@x-target)**2)+sp**3*np.sum((c@x)**2))
    before = objective(initial)
    owner = {col: key for key, col in keys.items()}
    base, pinned = set(fixed), set()
    best = None
    # Active set: a polyline that already grazes the boundary leaves it under any
    # step, however small. Pin the vertices of each new exit where they are (their
    # edges are contained there) and solve again, instead of letting one grazing
    # edge veto the whole neighbourhood. Every pinned vertex is reported.
    #
    # Pin also when the line search succeeds only partway. Otherwise a fit whose
    # contained optimum is ~80 um away creeps by 1/16..1/128 of it per iteration and
    # looks stationary, then jumps once a full failure finally triggers pinning
    # (segment 2699 on region 318). Every contained attempt is kept and the lowest
    # objective wins, so pinning can only improve on the partial step.
    for attempt in range(PIN_ROUNDS+1):
        fixed = np.array(sorted(base | pinned), dtype=int)
        free = np.setdiff1d(np.arange(len(initial)), fixed)
        if not len(free):
            return dict(report, status='stationary', max_move_um=0., **_pins(pinned, owner, lookup))
        matrix = design[:, free]
        rhs = target-design[:, fixed]@initial[fixed]
        h, b = matrix.T@matrix, matrix.T@rhs
        proposed = initial.copy()
        if len(constraints):
            cf = c[:, free]
            system = sparse.bmat([[h, cf.T], [cf, None]], format='csc')
            solution = spsolve(system, np.vstack([b, -c[:, fixed]@initial[fixed]]))
            proposed[free] = solution[:len(free)]
        else:
            proposed[free] = spsolve(h.tocsc(), b)
        alpha = 1.
        tried = []
        full = None
        while alpha >= 1/256:
            trial = initial+alpha*(proposed-initial)
            candidate = {sid: x.copy() for sid, x in old.items()}
            for sid, (lo, hi) in spans.items():
                candidate[sid][lo:hi+1] = trial[indices[sid]]
            after = objective(trial)
            reasons = {sid: reason for sid in spans
                       if (reason := movement_rejection(old[sid], candidate[sid], sampler, frame,
                                                        original_bad=original_bad[sid]))}
            tried.append((alpha, set(reasons)))
            uphill = not np.isfinite(after) or after > before+1e-9*max(1., before)
            if full is None:
                full = candidate, reasons, uphill
            if not uphill and not reasons:
                break
            alpha /= 2
        if alpha >= 1/256:
            if best is None or after < best['after']:
                best = dict(after=after, trial=trial, candidate=candidate, alpha=alpha,
                            pinned=set(pinned))
            if alpha == 1.:
                break
            # A partial step: pin what exits at the full step and try for more.
            candidate, reasons, uphill = full
        if (attempt == PIN_ROUNDS or uphill
                or set(reasons.values()) != {'new_segmentation_exit'}):
            break
        extra = set()
        for sid in reasons:
            lo, hi = spans[sid]
            for e in np.flatnonzero(bad_edges(candidate[sid], sampler, frame) & ~original_bad[sid]):
                extra.update(lookup[sid, i] for i in range(e-1, e+3) if lo <= i <= hi)
        extra -= base | pinned
        if not extra:
            break
        pinned |= extra
    if best is not None:
        trial, candidate, alpha, after = best['trial'], best['candidate'], best['alpha'], best['after']
        pinned = best['pinned']
    if best is None:
        return dict(report, status='blocked', objective_before=before,
                    objective_after=after, objective_increase=bool(uphill),
                    blocked_reasons=reasons,
                    blocked_detail={sid: _block_detail(graph, sid, nodes, spans[sid], old[sid],
                                                       candidate[sid], original_bad[sid],
                                                       observations, tried, sampler, frame)
                                    for sid in sorted(reasons)},
                    **_pins(pinned, owner, lookup))
    peak = float(np.linalg.norm(trial-initial, axis=1).max())
    for sid in sorted(spans):
        graph.set_segment_coords(sid, candidate[sid])
    return dict(report, status='stationary' if peak < .1*sp else 'moving',
                max_move_um=peak, step=alpha, objective_before=before, objective_after=after,
                **_pins(pinned, owner, lookup))


def _pins(pinned, owner, lookup):
    """Report pinned vertices per segment, and any pinned junction node, explicitly."""
    if not pinned:
        return {}
    points = {}
    for (sid, i), col in lookup.items():
        if col in pinned:
            points.setdefault(int(sid), []).append(int(i))
    return dict(pinned_points={sid: sorted(v) for sid, v in sorted(points.items())},
                pinned_nodes=sorted(int(owner[col][1]) for col in pinned if owner[col][0] == 'node'))


def _block_detail(graph, sid, nodes, span, old, new, original_bad, observations, tried,
                  sampler, frame):
    """Where and how early one segment vetoes a neighbourhood step.

    ``contained_alpha`` is the largest tried step at which this segment alone had no
    objection; ``None`` means even 1/256 of the step fails, i.e. the current polyline
    already touches the boundary where it would move. Exit edges within two edges of a
    jointly moved node point at the node shift rather than the segment's own fit.
    """
    from .centreline_refine import bad_edges
    exits = np.flatnonzero(bad_edges(new, sampler, frame) & ~original_bad)
    seg = graph.segment(sid)
    moved_ends = [i for i, key in ((0, 'node1'), (len(old)-2, 'node2')) if seg[key] in nodes]
    return dict(
        exit_edges=exits.tolist(),
        exit_near_moved_node=bool(any(abs(int(e)-end) <= 2 for e in exits for end in moved_ends)),
        contained_alpha=max((a for a, blocked in tried if sid not in blocked), default=None),
        released_span=[int(span[0]), int(span[1])], n_points=len(old),
        accepted_sections=len(observations[sid][2]),
        max_move_um=float(np.linalg.norm(new-old, axis=1).max()))
