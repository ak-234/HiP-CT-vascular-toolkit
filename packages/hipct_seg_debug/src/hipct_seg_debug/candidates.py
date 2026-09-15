"""Topological heuristics that flag skeleton sites worth checking for collapse.

``premature_end``
    A branch of Strahler order >= 2 simply stops. Vessels taper through the orders;
    a trunk that terminates is the strongest sign the segmentation lost it.

``endpoint_gap``
    A dead-end *pointing at* another vessel that is far away **in the tree** but close
    in space. Consistent with the segmentation losing a collapsed stretch.

``parallel_pair``
    Two branches that are topologically distant yet run alongside each other, closer
    than their own radii and near-parallel, over a sustained length. Consistent with
    one vessel that collapsed in the middle being segmented as two, then re-inflated
    into round tubes by the circular cross-section assumption.

**Tree-awareness is essential.** Branches meeting at a bifurcation are naturally close
and near-parallel just past the junction. Filtering only on "shares a vertex" (hop 1)
leaves every uncle/grandparent pair at hop 2, and on the LADAF-2024-28 tree that
produced 33 candidates of which *all 33* were hop 2-4 in one component: pure false
positives. Both pair detectors therefore require a minimum hop distance in the
edge-adjacency graph and compatible Strahler orders.

``murray_deficit``
    A bifurcation where the daughters cannot carry the parent: ``sum(r_child**3)``
    far below ``r_parent**3``. Consistent with the segmentation having lost a
    collapsed daughter branch.

    **Sample the radii away from the junction.** All edges meeting at a vertex share
    that point, so reading each radius *at* the shared vertex returns the same value
    three times and the ratio degenerates to exactly 2.00 everywhere -- which is what
    an earlier version of this module did, leading to the false conclusion that
    Murray's law was unusable on this data. Measured two local radii along each
    branch, the tree gives median 0.81 with a p5-p95 of 0.14-1.90.

Radius-based *endpoint* tests really are weak here: 145 of 161 terminals sit at exactly
156.640 um (six distinct values in all), a floor left by ``adjust_thickness.py``. That
is why ``premature_end`` keys off Strahler order instead of size.

These are candidates, not verdicts. Their only job is to steer the operator to places
worth checking against the greyscale image, which remains the sole ground truth. The
graph says little about collapse on a cleaned tree like this one -- see
``crosssection.py`` for the detector that measures it directly from the image.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Candidate:
    id: int
    kind: str
    x_um: float
    y_um: float
    z_um: float
    raw_slice: int
    raw_row: int
    raw_col: int
    gap_um: float  # clear space between the two lumen walls (negative = overlapping)
    dist_um: float  # centre-to-centre distance
    radius_um: float
    partner_radius_um: float
    edge_a: int
    edge_b: int
    hops: float  # edges between edge_a and edge_b in the tree (inf = different trees)
    strahler_a: int
    strahler_b: int
    contact_um: float  # sustained length of the parallel run (0 for endpoints)
    cos_angle: float
    same_component: bool
    score: float  # lower = more suspicious
    detail: str
    # Global point indices bounding the run this was found in, inclusive, or -1
    # where the kind is a single point rather than a stretch. A collapse is a
    # region and a repair has to know where it starts and ends, not only where it
    # is worst -- `crosssection._runs` has always computed these and thrown them
    # away. Defaulted so the graph detectors, which are point-wise, need no change.
    point_a: int = -1
    point_b: int = -1

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x_um, self.y_um, self.z_um])

    @property
    def has_span(self) -> bool:
        return self.point_a >= 0 and self.point_b >= self.point_a

    def label(self) -> str:
        return f"#{self.id} {self.kind} @slice {self.raw_slice}"


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #
def _edge_tangents(graph) -> np.ndarray:
    """(P, 3) unit tangent at every centreline point, differenced within its own edge."""
    pts = graph.points
    tan = np.zeros_like(pts)
    off = graph.edge_offsets
    for e in range(graph.n_edge):
        a, b = off[e], off[e + 1]
        if b - a < 2:
            tan[a:b] = [0.0, 0.0, 1.0]
            continue
        d = np.gradient(pts[a:b], axis=0)
        tan[a:b] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return tan


def _arclength(graph) -> np.ndarray:
    """(P,) cumulative distance along each point's own edge, restarting per edge."""
    pts = graph.points
    s = np.zeros(len(pts))
    off = graph.edge_offsets
    for e in range(graph.n_edge):
        a, b = off[e], off[e + 1]
        if b - a < 2:
            continue
        step = np.linalg.norm(np.diff(pts[a:b], axis=0), axis=1)
        s[a + 1 : b] = np.cumsum(step)
    return s


def _edge_neighbourhood(graph):
    """(edges_at_vertex, related) where `related[e]` is every edge sharing a vertex with e."""
    edges_at_vertex: list[list[int]] = [[] for _ in range(graph.n_vertex)]
    for e, (a, b) in enumerate(graph.connectivity):
        edges_at_vertex[a].append(e)
        edges_at_vertex[b].append(e)
    related = [set() for _ in range(graph.n_edge)]
    for verts in edges_at_vertex:
        for e in verts:
            related[e].update(verts)
    return edges_at_vertex, related


def _components(graph) -> np.ndarray:
    """(V,) connected-component label per vertex."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = graph.n_vertex
    a, b = graph.connectivity[:, 0], graph.connectivity[:, 1]
    m = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, lab = connected_components(m, directed=False)
    return lab


def edge_hops(graph) -> np.ndarray:
    """(E, E) shortest path length between edges, counted in edges. inf across trees.

    Two edges are adjacent when they share a vertex, so hop 1 is a bifurcation partner,
    hop 2 an uncle or grandparent, and so on. This is the measure that separates
    "anatomically next to each other" from "should not be near each other at all".
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    edges_at_vertex, _ = _edge_neighbourhood(graph)
    rows, cols = [], []
    for verts in edges_at_vertex:
        for i in verts:
            for j in verts:
                if i != j:
                    rows.append(i)
                    cols.append(j)
    if not rows:
        return np.full((graph.n_edge, graph.n_edge), np.inf)
    adj = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(graph.n_edge, graph.n_edge)
    )
    return shortest_path(adj, directed=False, unweighted=True)


def _strahler(graph) -> np.ndarray:
    """(E,) Strahler order per edge, or ones if the attribute is absent."""
    s = graph.edge_attrs.get("strahler")
    if s is None:
        return np.ones(graph.n_edge, dtype=int)
    return np.asarray(s, dtype=int).ravel()


def _radius_along(graph, edge: int, vertex: int, factor: float = 2.0) -> float:
    """Radius on ``edge``, walked ``factor`` local radii away from ``vertex``.

    Never read the radius at the junction itself -- see the module docstring.
    """
    off = graph.edge_offsets
    a, b = int(off[edge]), int(off[edge + 1])
    pts = graph.points[a:b]
    rad = graph.thickness[a:b]
    if len(pts) < 2:
        return float(rad[0]) if len(rad) else float("nan")
    if np.linalg.norm(pts[0] - graph.vertices[vertex]) > np.linalg.norm(
        pts[-1] - graph.vertices[vertex]
    ):
        pts, rad = pts[::-1], rad[::-1]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    k = int(np.searchsorted(s, min(factor * rad[0], s[-1])))
    return float(rad[min(k, len(rad) - 1)])


def _endpoint_tips(graph):
    """For every degree-1 vertex: point index, outward unit direction, owning edge."""
    edges_at_vertex, _ = _edge_neighbourhood(graph)
    off = graph.edge_offsets
    pts = graph.points
    out = []
    for v in np.flatnonzero(graph.degree() == 1):
        e = edges_at_vertex[v][0]
        a, b = off[e], off[e + 1]
        if b - a < 2:
            continue
        head_first = np.linalg.norm(pts[a] - graph.vertices[v]) <= np.linalg.norm(
            pts[b - 1] - graph.vertices[v]
        )
        if head_first:
            tip = a
            inward = pts[min(a + 5, b - 1)] - pts[a]
        else:
            tip = b - 1
            inward = pts[max(b - 6, a)] - pts[b - 1]
        n = np.linalg.norm(inward)
        if n < 1e-9:
            continue
        out.append((int(tip), -inward / n, int(e), int(v)))
    return out


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def find_candidates(
    graph,
    frame,
    endpoint_max_gap_um: float = 3000.0,
    endpoint_cos: float = 0.7,
    parallel_dist_factor: float = 1.25,
    parallel_cos: float = 0.9,
    min_contact_um: float = 0.0,
    boundary_margin_um: float = 200.0,
    min_hops: int = 4,
    max_strahler_delta: int = 1,
    premature_min_strahler: int = 2,
    murray_percentile: float = 0.10,
    max_per_kind: int = 400,
) -> list[Candidate]:
    """Scan the spatial graph for the three signatures, best first.

    endpoint_max_gap_um
        Largest wall-to-wall gap a dead-end may have to its target and still be
        reported. The tip must also *point at* the target (``endpoint_cos``).
    parallel_dist_factor
        Report a pair when centre-to-centre distance is below
        ``factor x (r_i + r_j)``.
    min_contact_um
        Discard parallel runs shorter than this; raise it to suppress incidental
        crossings and keep only sustained side-by-side contact.
    min_hops
        Minimum separation in the edge-adjacency graph for a *pair* to be reported.
        Anything below this is an ordinary bifurcation neighbourhood. Pairs in
        different trees have infinite separation and always qualify.
    max_strahler_delta
        A collapse-split joins like with like, so require comparable Strahler orders.
    premature_min_strahler
        Lowest Strahler order whose termination counts as suspicious. Order-1 twigs
        end all the time; a higher order stopping does not.
    murray_percentile
        Report bifurcations in this lowest fraction of the tree's own
        ``sum(r_child^3)/r_parent^3`` distribution. Judging against the tree itself
        rather than a textbook 1.0 keeps the test meaningful on a real, pruned tree.
    """
    pts = graph.points
    rad = graph.thickness
    owner = graph.edge_of_point()
    tree = cKDTree(pts)
    _, related = _edge_neighbourhood(graph)
    comp = _components(graph)
    edge_comp = comp[graph.connectivity[:, 0]]
    tan = _edge_tangents(graph)
    arc = _arclength(graph)
    hops = edge_hops(graph)
    stra = _strahler(graph)

    seg_bbox = frame.seg_bbox_um
    lo = seg_bbox[0::2] + boundary_margin_um
    hi = seg_bbox[1::2] - boundary_margin_um

    def compatible(ea: int, eb: int) -> bool:
        """Topologically distant and of comparable calibre."""
        return hops[ea, eb] >= min_hops and abs(stra[ea] - stra[eb]) <= max_strahler_delta

    out: list[Candidate] = []
    cid = 0

    # ----------------------------------------------------------- premature_end
    # Radii are clamped at the terminals on this graph, so order -- not size -- is the
    # usable signal that a branch has stopped before it should have.
    ends = []
    for tip_idx, _direction, e, _v in _endpoint_tips(graph):
        if stra[e] < premature_min_strahler:
            continue
        tip = pts[tip_idx]
        if np.any(tip < lo) or np.any(tip > hi):
            continue  # leaves the imaged volume; not a segmentation failure
        ends.append((-stra[e], -rad[tip_idx], tip_idx, e))
    ends.sort()
    for _ns, _nr, tip_idx, e in ends[:max_per_kind]:
        tip = pts[tip_idx]
        zyx = frame.um_to_raw_index(tip)[0]
        cid += 1
        out.append(
            Candidate(
                id=cid,
                kind="premature_end",
                x_um=float(tip[0]),
                y_um=float(tip[1]),
                z_um=float(tip[2]),
                raw_slice=int(zyx[0]),
                raw_row=int(zyx[1]),
                raw_col=int(zyx[2]),
                gap_um=0.0,
                dist_um=0.0,
                radius_um=float(rad[tip_idx]),
                partner_radius_um=0.0,
                edge_a=int(e),
                edge_b=-1,
                hops=float("inf"),
                strahler_a=int(stra[e]),
                strahler_b=-1,
                contact_um=0.0,
                cos_angle=0.0,
                same_component=True,
                score=float(-stra[e]),
                detail=(
                    f"edge {e} terminates at Strahler order {stra[e]} "
                    f"(r = {rad[tip_idx]:.0f} um) - a vessel this order should keep branching"
                ),
            )
        )

    # ----------------------------------------------------------- murray_deficit
    edges_at_vertex, _ = _edge_neighbourhood(graph)
    ratios = []
    for v in np.flatnonzero(graph.degree() >= 3):
        pos = graph.vertices[v]
        if np.any(pos < lo) or np.any(pos > hi):
            continue
        es = edges_at_vertex[v]
        rs = np.array([_radius_along(graph, e, v) for e in es])
        if not np.all(np.isfinite(rs)) or rs.max() <= 0:
            continue
        parent = int(np.argmax(rs))
        children = np.delete(rs, parent)
        ratios.append(((children ** 3).sum() / rs[parent] ** 3, v, rs[parent], children, es, parent))

    if ratios:
        vals = np.array([r[0] for r in ratios])
        cut = float(np.percentile(vals, 100.0 * murray_percentile))
        for ratio, v, r_par, children, es, parent in sorted(ratios)[:max_per_kind]:
            if ratio > cut:
                break
            pos = graph.vertices[v]
            zyx = frame.um_to_raw_index(pos)[0]
            e_par = int(es[parent])
            cid += 1
            out.append(
                Candidate(
                    id=cid,
                    kind="murray_deficit",
                    x_um=float(pos[0]),
                    y_um=float(pos[1]),
                    z_um=float(pos[2]),
                    raw_slice=int(zyx[0]),
                    raw_row=int(zyx[1]),
                    raw_col=int(zyx[2]),
                    gap_um=0.0,
                    dist_um=0.0,
                    radius_um=float(r_par),
                    partner_radius_um=float(children.max()),
                    edge_a=e_par,
                    edge_b=-1,
                    hops=float("inf"),
                    strahler_a=int(stra[e_par]),
                    strahler_b=-1,
                    contact_um=0.0,
                    cos_angle=0.0,
                    same_component=True,
                    score=float(ratio),
                    detail=(
                        f"bifurcation at vertex {v}: daughters carry only "
                        f"{100 * ratio:.0f}% of the parent's r^3 "
                        f"(parent {r_par:.0f} um vs children "
                        f"{', '.join(f'{c:.0f}' for c in children)} um) - "
                        f"a branch may have been lost here"
                    ),
                )
            )

    # ------------------------------------------------------------- endpoint_gap
    hits = []
    for tip_idx, direction, e, _v in _endpoint_tips(graph):
        tip = pts[tip_idx]
        if np.any(tip < lo) or np.any(tip > hi):
            continue  # leaves the imaged volume; not a segmentation failure
        r_tip = rad[tip_idx]
        reach = endpoint_max_gap_um + r_tip + float(rad.max())
        best = None
        for j in tree.query_ball_point(tip, reach):
            if owner[j] == e or owner[j] in related[e]:
                continue
            delta = pts[j] - tip
            d = float(np.linalg.norm(delta))
            if d < 1e-9:
                continue
            cos = float(np.dot(direction, delta / d))
            if cos < endpoint_cos:
                continue  # the branch is not heading towards this vessel
            if not compatible(e, owner[j]):
                continue  # a neighbour in the tree, or a mismatched calibre
            gap = d - (r_tip + rad[j])
            if best is None or gap < best[0]:
                best = (gap, d, cos, int(j))
        if best is None:
            continue
        gap, d, cos, j = best
        if gap > endpoint_max_gap_um:
            continue
        hits.append((gap, d, cos, j, tip_idx, e, r_tip))

    hits.sort(key=lambda t: t[0])
    for gap, d, cos, j, tip_idx, e, r_tip in hits[:max_per_kind]:
        mid = 0.5 * (pts[tip_idx] + pts[j])
        zyx = frame.um_to_raw_index(mid)[0]
        cid += 1
        out.append(
            Candidate(
                id=cid,
                kind="endpoint_gap",
                x_um=float(mid[0]),
                y_um=float(mid[1]),
                z_um=float(mid[2]),
                raw_slice=int(zyx[0]),
                raw_row=int(zyx[1]),
                raw_col=int(zyx[2]),
                gap_um=float(gap),
                dist_um=float(d),
                radius_um=float(r_tip),
                partner_radius_um=float(rad[j]),
                edge_a=int(e),
                edge_b=int(owner[j]),
                hops=float(hops[e, owner[j]]),
                strahler_a=int(stra[e]),
                strahler_b=int(stra[owner[j]]),
                contact_um=0.0,
                cos_angle=float(cos),
                same_component=bool(edge_comp[e] == edge_comp[owner[j]]),
                score=float(gap),
                detail=(
                    f"edge {e} dead-ends pointing at edge {owner[j]} "
                    f"{d:.0f} um away, clear gap {gap:.0f} um; "
                    f"{hops[e, owner[j]]:.0f} hops apart, Strahler {stra[e]}/{stra[owner[j]]}"
                ),
            )
        )

    # ----------------------------------------------------------- parallel_pair
    # A pair qualifies only when d < f*(r_i + r_j) <= 2*f*max(r_i, r_j), so a search
    # radius of 2*f*r_i around each point still finds every pair -- from the side of
    # the larger radius. This keeps the search local; a single global radius would
    # return millions of pairs on a tree this size.
    neigh = tree.query_ball_point(pts, 2.0 * parallel_dist_factor * rad)
    src, dst = [], []
    for a, lst in enumerate(neigh):
        for b in lst:
            if b > a and owner[a] != owner[b]:
                src.append(a)
                dst.append(b)
    i = np.asarray(src, dtype=np.int64)
    j = np.asarray(dst, dtype=np.int64)

    if len(i):
        d = np.linalg.norm(pts[i] - pts[j], axis=1)
        rsum = rad[i] + rad[j]
        keep = d < parallel_dist_factor * rsum
        i, j, d, rsum = i[keep], j[keep], d[keep], rsum[keep]
    if len(i):
        cos = np.abs(np.sum(tan[i] * tan[j], axis=1))
        keep = cos > parallel_cos
        i, j, d, rsum, cos = i[keep], j[keep], d[keep], rsum[keep], cos[keep]
    if len(i):
        # Tree gate. Branches around a bifurcation are close and parallel by nature;
        # only a pair that is *distant in the tree* yet adjacent in space is suspicious.
        # Cheap to apply now the geometric filters have cut the list down.
        keep = np.array([compatible(owner[a], owner[b]) for a, b in zip(i, j)])
        i, j, d, rsum, cos = i[keep], j[keep], d[keep], rsum[keep], cos[keep]

    if len(i):
        mids = 0.5 * (pts[i] + pts[j])
        ratio = d / np.maximum(rsum, 1e-9)
        order = np.argsort(ratio)
        unassigned = np.ones(len(i), dtype=bool)
        clusters = []
        for k in order:
            if not unassigned[k]:
                continue
            # Cluster radius scales with vessel size: a fixed few-voxel radius would
            # split one large-vessel contact into dozens of duplicate reports.
            reach = max(4.0 * float(frame.seg_spacing.max()), 0.75 * float(rsum[k]))
            same = unassigned & (np.linalg.norm(mids - mids[k], axis=1) < reach)
            same &= (owner[i] == owner[i[k]]) & (owner[j] == owner[j[k]])
            unassigned[same] = False
            members = np.flatnonzero(same)
            span_a = float(arc[i[members]].ptp()) if len(members) > 1 else 0.0
            span_b = float(arc[j[members]].ptp()) if len(members) > 1 else 0.0
            clusters.append((k, members, max(span_a, span_b)))
            if len(clusters) >= max_per_kind * 4:
                break

        # Rank by closeness first, then reward sustained contact.
        clusters.sort(key=lambda c: (ratio[c[0]] / (1.0 + c[2] / 500.0)))
        for k, members, contact in clusters:
            if contact < min_contact_um:
                continue
            m = mids[k]
            zyx = frame.um_to_raw_index(m)[0]
            ea, eb = int(owner[i[k]]), int(owner[j[k]])
            cid += 1
            out.append(
                Candidate(
                    id=cid,
                    kind="parallel_pair",
                    x_um=float(m[0]),
                    y_um=float(m[1]),
                    z_um=float(m[2]),
                    raw_slice=int(zyx[0]),
                    raw_row=int(zyx[1]),
                    raw_col=int(zyx[2]),
                    gap_um=float(d[k] - rsum[k]),
                    dist_um=float(d[k]),
                    radius_um=float(rad[i[k]]),
                    partner_radius_um=float(rad[j[k]]),
                    edge_a=ea,
                    edge_b=eb,
                    hops=float(hops[ea, eb]),
                    strahler_a=int(stra[ea]),
                    strahler_b=int(stra[eb]),
                    contact_um=float(contact),
                    cos_angle=float(cos[k]),
                    same_component=bool(edge_comp[ea] == edge_comp[eb]),
                    score=float(ratio[k]),
                    detail=(
                        f"edges {ea} & {eb} run {d[k]:.0f} um apart "
                        f"(radii sum {rsum[k]:.0f} um) for {contact:.0f} um, |cos| {cos[k]:.2f}; "
                        f"{hops[ea, eb]:.0f} hops apart, Strahler {stra[ea]}/{stra[eb]}"
                    ),
                )
            )
            if sum(1 for c in out if c.kind == "parallel_pair") >= max_per_kind:
                break
    return out


def write_csv(cands: list[Candidate], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(Candidate)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for c in cands:
            w.writerow(asdict(c))
    return path


def summarise(cands: list[Candidate]) -> str:
    if not cands:
        return "no candidates found"
    lines = []
    for kind in ("premature_end", "murray_deficit", "endpoint_gap", "parallel_pair",
                 "perimeter_mismatch", "companion_lumen", "collapse_severity"):
        sub = [c for c in cands if c.kind == kind]
        if not sub:
            lines.append(f"{kind}: none")
            continue
        lines.append(f"{kind}: {len(sub)}")
        for c in sub[:5]:
            lines.append(f"    #{c.id:<4d} slice {c.raw_slice:<5d} {c.detail}")
        if len(sub) > 5:
            lines.append(f"    ... {len(sub) - 5} more")
    return "\n".join(lines)
