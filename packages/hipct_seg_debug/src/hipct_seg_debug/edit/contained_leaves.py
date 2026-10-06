"""Find and remove leaves drawn across another vessel's lumen (medial-sheet spurs).

A flattened lumen skeletonises into a sheet, and the skeleton keeps branches across
it: on the left tree, segment 3655 runs along one edge of a slit about 2.8 x 0.8 mm,
and leaves 794 and 1046 lie inside the same slit. They have no lumen of their own,
but the shared section filter sees another centreline inside 3655's cross-section
and rejects every 3655 section as contaminated by a neighbouring lumen. With no
sections, 3655 stayed where bending put it, about 0.6 mm off the slit's centre.

Length rules cannot catch these: 794 is 7.8 radii long, because its radius was
measured across the narrow side of the slit. So containment is measured from the
segmentation. Each sampled leaf point is taken to the nearest point of a non-leaf
segment, that segment's cross-section is cut there, and the leaf point must fall in
the lumen component holding the host's own centreline. A leaf is contained when at
least `fraction` of its decidable samples are inside, and every sample in its outer
`tip_fraction` is (thresholds as in coronary_sdf's capsule-based
``prune_contained_leaves``, which uses graph radii and so cannot see a slit).
A sample is undecidable when it does not lie in the host's cut plane -- near the
junction its nearest host point is the shared node -- or the host cut fails.
"""
from __future__ import annotations

import numpy as np

CONTAINED_FRACTION = 0.8
TIP_FRACTION = 0.5
SAMPLES = 12
#: A sample further than this from the host's cut plane, in voxels, is undecidable.
PLANE_TOLERANCE_VOXELS = 1.5
#: Host stations considered for a sample: within this distance of it.
SEARCH_RADIUS_UM = 3000.


def _leaf_ids(graph, sids=None):
    """Segments with one free end and a junction (degree >= 3) at the other."""
    out = []
    for sid in (graph.segment_ids() if sids is None else sids):
        if not graph.has_segment(sid):
            continue
        seg = graph.segment(sid)
        d1, d2 = graph.degree(seg["node1"]), graph.degree(seg["node2"])
        if (d1 == 1 and d2 >= 3) or (d2 == 1 and d1 >= 3):
            out.append(sid)
    return out


def _interior_station(graph, host, j, sp):
    """Station `j` moved inward when it is within two host radii of a junction end."""
    y = graph.coords(host)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(y, axis=0), axis=1))]
    reach = max(2.0 * float(np.median(graph.radii(host))), 4.0 * sp)
    seg = graph.segment(host)
    if s[j] < reach and graph.degree(seg["node1"]) >= 3:
        return int(np.argmin(np.abs(s - min(reach, s[-1] / 2))))
    if s[-1] - s[j] < reach and graph.degree(seg["node2"]) >= 3:
        return int(np.argmin(np.abs(s - max(s[-1] - reach, s[-1] / 2))))
    return j


def _tangent(x, j):
    t = x[min(j + 1, len(x) - 1)] - x[max(j - 1, 0)]
    n = np.linalg.norm(t)
    return t / n if n > 0 else None


def find_contained_leaves(graph, labels, frame, *, sids=None, fraction=CONTAINED_FRACTION,
                          tip_fraction=TIP_FRACTION, samples=SAMPLES, sampler=None,
                          protected_nodes=(), any_host=True) -> list[dict]:
    """One row per leaf: whether it lies inside another segment's lumen, and why.

    Never flagged: a leaf whose free end is in `protected_nodes` (roots), and a leaf
    with a recorded Strahler order above 1. A true leaf has order 1; a degree-1
    segment of higher order is the trunk's root end -- 3721 on the left tree, order 8,
    radius ~2 mm -- which can read as "inside" a small branch whose cross-section
    near the trunk takes in the trunk's lumen.
    """
    protected_nodes = set(protected_nodes)
    from scipy.spatial import cKDTree

    from ..crosssection import _PlaneSampler, cut

    sampler = sampler if sampler is not None else _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    leaves = _leaf_ids(graph, sids)
    leaf_set = set(_leaf_ids(graph))
    hosts = [sid for sid in graph.segment_ids() if sid not in leaf_set and len(graph.coords(sid)) >= 2]
    if not leaves or not hosts:
        return []
    pts = np.vstack([graph.coords(s) for s in hosts])
    owner = np.concatenate([np.full(len(graph.coords(s)), s) for s in hosts])
    index = np.concatenate([np.arange(len(graph.coords(s))) for s in hosts])
    tree = cKDTree(pts)
    rows = []
    for leaf in leaves:
        seg = graph.segment(leaf)
        x = graph.coords(leaf)
        free = seg["node1"] if graph.degree(seg["node1"]) == 1 else seg["node2"]
        order = seg.get("strahler")
        if free in protected_nodes or (order is not None and int(order) > 1):
            rows.append(dict(segment=int(leaf), contained=False,
                             protected="root" if free in protected_nodes else f"strahler {int(order)}",
                             length_um=float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum()),
                             radius_um=float(np.median(graph.radii(leaf)))))
            continue
        if graph.degree(seg["node1"]) == 1:
            x = x[::-1]  # junction first, free tip last
        picks = np.unique(np.rint(np.linspace(1, len(x) - 1, min(samples, len(x) - 1))).astype(int))
        status, used = [], set()
        for i in picks:
            # The nearest host *station whose cut plane contains this point*. The single
            # nearest host point is often a host's end node, whose plane misses a leaf
            # running sideways from it -- yet 794 lies wholly in the plane of 3655's last
            # station.
            per_host = {}
            for k in tree.query_ball_point(x[i], SEARCH_RADIUS_UM):
                host, j = int(owner[k]), int(index[k])
                y = graph.coords(host)
                t = _tangent(y, j)
                if t is None:
                    continue
                offset = (x[i] - y[j]) / sp
                if abs(offset @ t) <= PLANE_TOLERANCE_VOXELS:
                    d = float(np.linalg.norm(offset))
                    if host not in per_host or d < per_host[host][0]:
                        per_host[host] = (d, j, t, offset)
            if not per_host:
                status.append(None)
                continue
            # With `any_host` (the default) every host whose plane holds the point is
            # asked, and one containing it suffices (1046 lies in 3655's sections but
            # nearer stations of other hosts); otherwise only the nearest such host.
            ranked = sorted(per_host.items(), key=lambda kv: kv[1][0])
            verdicts = []
            for host, (_, j, t, offset) in (ranked if any_host else ranked[:1]):
                y = graph.coords(host)
                # Judge against the host's cross-section away from its junctions. At a
                # junction every branch is joined to the host's lumen, so a genuine
                # branch leaving square to the host lies in that station's lumen blob
                # over its whole length -- as does 794. What tells them apart is the
                # host's normal footprint: 3655 is the same slit along its length and
                # 794 stays inside it, while a real branch leaves it. The section is
                # taken to be locally constant along the host (a prism).
                j = _interior_station(graph, host, j, sp)
                t = _tangent(y, j) if _tangent(y, j) is not None else t
                half = int(np.ceil(2 * max(2.0, float(graph.radii(host)[j]) / sp)))
                c = cut(sampler, frame.um_to_seg(y[j][None])[0], t, half, max_half=256)
                if c is None:
                    continue
                r = int(round(float(offset @ c.u))) + c.half
                q = int(round(float(offset @ c.v))) + c.half
                verdicts.append(0 <= r < c.blob8.shape[0] and 0 <= q < c.blob8.shape[1]
                                and bool(c.blob8[r, q]))
                used.add(host)
                if verdicts[-1]:
                    break
            status.append(any(verdicts) if verdicts else None)
        decided = [s for s in status if s is not None]
        tip = [s for s in status[-max(1, int(round(len(status) * tip_fraction))):] if s is not None]
        frac = float(np.mean(decided)) if decided else 0.0
        # The free end node itself often sits on the lumen wall, so "every tip sample"
        # rejected 794 (11 of 12 inside); the tip needs the same fraction as the whole.
        tip_ok = bool(tip) and float(np.mean(tip)) >= fraction
        enough = len(decided) >= max(1, len(status) // 2)
        rows.append(dict(segment=int(leaf), contained=enough and frac >= fraction and tip_ok,
                         fraction_inside=frac, decided=len(decided), sampled=len(status),
                         tip_inside=tip_ok, hosts=sorted(used),
                         length_um=float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum()),
                         radius_um=float(np.median(graph.radii(leaf)))))
    return rows


def prune_contained_leaves(graph, rows) -> dict:
    """Delete the contained leaves in one undoable batch.

    Unlike ``prune_spurs``, the degree-2 node a removed leaf leaves behind is kept
    rather than rejoined: rejoining builds a new segment with a new id, and removing
    794 would then have merged 3655 into an unrecognisable segment. Refinement
    handles a degree-2 node (shared position and derivative).
    """
    doomed = [r["segment"] for r in rows if r["contained"] and graph.has_segment(r["segment"])]
    with graph.batch("prune contained leaves"):
        for sid in doomed:
            graph.delete_segment(sid)
    return dict(removed=doomed)
