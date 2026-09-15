"""Cut a tree down to the vessels of interest, and record why in a sidecar.

Three rules, composable in one run, each of which drops a branch **and everything
downstream of it** rather than the branch alone -- a kept twig hanging off a dropped
parent is not a smaller tree, it is a broken one:

* **Strahler** -- drop every branch below a given order.
* **absolute take-off radius** -- drop every branch thinner than a given radius.
* **ostium ratio** -- drop a side branch whose take-off radius is below a fraction of
  the *ostial* radius of the main vessel it descends from. This is
  ``coronary_sdf.epicardial_annotation``'s rule, ported: the main vessels are named by
  hand, and a branch five generations off the LAD is still judged against the LAD's own
  proximal radius, not against its immediate parent's.

Plus whatever the operator marked by hand in the 3D window.

Naming a main vessel is a *selection*, not a rule, and it does not have to be made
one segment at a time: :func:`trace_path` fills a whole vessel in from two picks --
an ostium and a far end -- by taking the one simple path that runs between them. On
a tree that path is unique; on a graph carrying a fused cycle the weighting says
which route wins, and neither route may double back or loop.

**The sidecar is the artefact; the ``.am`` is derived from it.** It carries the named
main vessels, the rule and the resulting selection, so a crop can be reviewed, argued
with, and re-run against a repaired graph months later. That is also why segments are
identified in it by a *geometric* key rather than by ``seg["id"]``: an id is the source
edge index, and :func:`adapter.to_spatial_graph` renumbers edges to array indices every
time a graph is written, so a sidecar keyed by id silently names different vessels the
moment its graph has been through `gaps`, `connect` or a previous `crop`. A geometric
key survives renumbering, and when it does break it breaks loudly.

Two deliberate departures from `coronary_sdf`, both because this package measures
differently rather than because the rule differs:

* the take-off radius is read by :func:`skeleton_optimise._radius_away` -- walked a
  couple of local radii along the branch from the junction -- where `coronary_sdf`
  skips a fixed *number of contours*. Both exist to avoid reading the radius **at** a
  junction, where the distance transform measures the whole carina rather than either
  vessel; the numbers here will not match ``pruned_branches.csv``.
* one pass, and no degree-2 contraction. Contraction is what forced `coronary_sdf` to
  iterate: merging a parent *through* a de-branched node moves a grandchild's shared
  node and forces re-measurement. Without it, a second pass provably finds nothing --
  dropping a subtree at node *N* cannot change a surviving sibling's take-off (oriented
  at *N* and walked away from it), its distal degree, its descent, or any ostium, since
  a main vessel can never be dropped. Contraction would also destroy the very identity
  the sidecar is written in, so `crop` reports the degree-2 nodes it created and leaves
  them for ``optimise-skeleton``.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# One JSON convention for the whole package. `_plain` is private only because nothing
# outside the audit trail needed it before; its non-finite-float -> null and Path -> str
# handling is exactly the bug surface not to keep two copies of. If a third consumer
# appears, promote both into an `edit/sidecar.py` rather than copying them again.
# `write` is re-exported deliberately: callers say `crop.write(path, document)` and
# never need to know the audit trail got there first.
from .reconnect.geodesic.audit import _plain, write  # noqa: F401

SCHEMA = "hipct.crop/1"

#: Vessel names and colours from ``coronary_sdf.epicardial_annotation``. Neither carries
#: any semantics -- the names drive the picker's label cycle, the colours the overlay --
#: but keeping them identical means an operator reads the same tree the same way in both
#: tools.
PRESET_VESSEL_NAMES = ("LAD", "LCx", "IM", "RCA", "Diag", "OM", "PDA", "PLB", "Ramus")
VESSEL_COLORS = (
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#bcbd22", "#e377c2", "#8c564b",
)

#: Local radii to walk from a junction before reading a branch's take-off radius.
TAKEOFF_FACTOR = 2.0


class CropError(RuntimeError):
    """A crop that cannot be carried out as asked, rather than one that drops nothing."""


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def _point_key(graph, pid: int) -> str:
    x, y, z = graph.points[pid][:3]
    return f"{int(round(x))},{int(round(y))},{int(round(z))}"


def segment_key(graph, sid: int) -> str:
    """A geometric fingerprint of one segment, stable across a write and a re-read.

    The two endpoint coordinates plus the middle of the run, at 1 um resolution, each
    sorted so the key does not depend on which way the segment happens to run. Amira
    guarantees a segment's first and last points sit *on* its nodes, so this needs no
    node table -- and node ids are array indices too, so it could not use one anyway.

    ``coronary_sdf``'s ``build_segment_keys`` takes a single mid-point at ``len // 2``,
    which is **not** reversal-invariant on an even-length run: reversing maps that
    index to ``(len - 1) // 2``, a different point. Both are taken here and sorted, so
    the pair is invariant either way. The two tools' keys are therefore not
    interchangeable, and are not meant to be.
    """
    ids = graph.segment(sid)["point_ids"]
    if not ids:
        return ""
    ends = sorted((_point_key(graph, ids[0]), _point_key(graph, ids[-1])))
    mids = sorted((_point_key(graph, ids[len(ids) // 2]),
                   _point_key(graph, ids[(len(ids) - 1) // 2])))
    return hashlib.sha1("|".join(ends + mids).encode()).hexdigest()[:16]


def segment_keys(graph) -> dict[int, str]:
    """``{segment id: key}`` for every segment in the graph."""
    return {sid: segment_key(graph, sid) for sid in graph.segment_ids()}


def resolve_keys(graph, keys) -> tuple[dict[str, int], list[str]]:
    """``({key: segment id}, unresolved)`` for the keys this graph can place.

    A key matching **more than one** segment counts as unresolved. A two-point segment
    has ``mid == lo``, so two of them in the same place produce the same key, and
    picking either would be a guess presented as a fact.
    """
    by_key: dict[str, list[int]] = {}
    for sid, key in segment_keys(graph).items():
        by_key.setdefault(key, []).append(sid)
    found: dict[str, int] = {}
    unresolved: list[str] = []
    for key in keys:
        hits = by_key.get(str(key), ())
        if len(hits) == 1:
            found[str(key)] = int(hits[0])
        else:
            unresolved.append(str(key))
    return found, unresolved


def node_at(graph, sid: int, point_um) -> int:
    """Whichever endpoint node of ``sid`` is nearer the given world position.

    The sidecar records a coordinate rather than a node id for the same reason it
    records a segment key rather than a segment id: node ids are array indices.
    """
    seg = graph.segment(sid)
    target = np.asarray(point_um, dtype=np.float64)[:3]
    best, best_d = seg["node1"], np.inf
    for nid in (seg["node1"], seg["node2"]):
        node = graph.nodes.get(nid)
        if node is None:
            continue
        d = float(np.linalg.norm(np.asarray(node[:3], dtype=np.float64) - target))
        if d < best_d:
            best, best_d = nid, d
    return int(best)


def source_fingerprint(path, graph=None) -> dict:
    """``{sha1, segments, nodes, points}`` of the graph a sidecar was written against.

    A mismatch on load is a *warning*, never a refusal: re-cropping a repaired graph is
    the intended workflow, and the geometric keys are what actually has to resolve.
    """
    out: dict = {"sha1": None}
    if path is not None:
        try:
            digest = hashlib.sha1()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            out["sha1"] = digest.hexdigest()
        except OSError:
            pass  # a graph held only in memory, or a source that has since moved
    if graph is not None:
        out["segments"] = len(graph.segments)
        out["nodes"] = len(graph.nodes)
        out["points"] = len(graph.points)
    return out


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Topology:
    """Each component rooted, as parent/child over segment ids."""

    depth: dict[int, int]
    parent: dict[int, int | None]
    children: dict[int, list[int]]
    roots: list[int]

    def shared_node(self, graph, sid: int) -> int | None:
        """The node ``sid`` meets its parent at -- its ostium. None at a root."""
        par = self.parent.get(sid)
        if par is None or not graph.has_segment(par) or not graph.has_segment(sid):
            return None
        seg, parent_seg = graph.segment(sid), graph.segment(par)
        shared = {seg["node1"], seg["node2"]} & {parent_seg["node1"], parent_seg["node2"]}
        return min(shared) if shared else None

    def prox_node(self, graph, sid: int) -> int:
        """The proximal (inlet) node of ``sid``: its ostium, or its free end."""
        node = self.shared_node(graph, sid)
        if node is not None:
            return node
        seg = graph.segment(sid)
        n1, n2 = seg["node1"], seg["node2"]
        if graph.degree(n1) == 1:
            return n1
        return n2 if graph.degree(n2) == 1 else n1

    def descendants(self, sid: int) -> set[int]:
        """``sid`` and every segment below it, walking the rooted tree only.

        Not ``EditableGraph.subtree``, which is an undirected flood: on a graph with a
        cycle that walks back around through the loop and takes the trunk -- and the
        trunk is exactly what a crop must never remove by accident.
        """
        out: set[int] = set()
        stack = [sid]
        while stack:
            current = stack.pop()
            if current in out:
                continue
            out.add(current)
            stack.extend(self.children.get(current, ()))
        return out


def topology(graph, root_edges=()) -> Topology:
    """Root every component and return its parent/child structure.

    Wraps ``radius_perimeter._directed_topology`` rather than rooting again: its key
    -- highest Strahler, then a free end, then the thickest, then the lowest id -- is
    this package's answer to "which end is proximal", and a second answer that
    disagreed with it would be worse than either.
    """
    from .radius_perimeter import _directed_topology

    depth, parent = _directed_topology(graph, tuple(int(x) for x in (root_edges or ())))
    children: dict[int, list[int]] = {sid: [] for sid in depth}
    roots: list[int] = []
    for sid, par in parent.items():
        if par is None:
            roots.append(int(sid))
        else:
            children.setdefault(int(par), []).append(int(sid))
    for kids in children.values():
        kids.sort()
    return Topology(dict(depth), dict(parent), children, sorted(roots))


# --------------------------------------------------------------------------- #
# tracing a vessel between two picks
# --------------------------------------------------------------------------- #
#: What an unmeasured segment's radius is charged as when tracing by thickness, in um.
#: Not zero -- a segment carrying no radius must not become a free shortcut -- and not
#: a large number either, which would make it the cheapest thing in the graph. At 1 um
#: an unmeasured segment costs exactly its own length, so a graph with no radii at all
#: traces identically under either weight.
TRACE_UNMEASURED_UM = 1.0


@dataclass(frozen=True)
class TracedPath:
    """One vessel traced end to end: the segments it runs through, in order.

    ``nodes`` are the nodes *between* consecutive segments, so there is always one
    fewer of them than there are segments. Both lists are simple -- no segment and no
    node appears twice -- which is the whole point of :func:`trace_path`.
    """

    segments: list[int]
    nodes: list[int]
    length_um: float
    weight: str = "length"

    def describe(self) -> str:
        return (f"{len(self.segments)} segment(s), {self.length_um / 1000.0:,.1f} mm, "
                f"through {len(self.nodes)} node(s), by {self.weight}")


def segment_length_um(graph, sid: int) -> float:
    """Arc length along a segment's centreline, in um -- not the node-to-node distance.

    A coronary segment is rarely straight, and a chord under-reads a tortuous one by
    enough to change which route a trace prefers.
    """
    xyz = graph.coords(sid)
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _trace_cost(graph, sid: int, *, prefer_thick: bool) -> float:
    """What one segment costs to walk through.

    By length, or -- with ``prefer_thick`` -- by length in units of the segment's own
    radius, which is what makes a thin bridge expensive and a fat detour cheap. The
    median radius rather than the mean, because a single junction point reading the
    whole carina should not decide the route.
    """
    length = segment_length_um(graph, sid)
    if not prefer_thick:
        return length
    radii = graph.radii(sid)
    finite = radii[np.isfinite(radii) & (radii > 0.0)]
    radius = float(np.median(finite)) if finite.size else 0.0
    return length / (radius if radius > 0.0 else TRACE_UNMEASURED_UM)


def trace_path(graph, start_sid: int, end_sid: int, *,
               prefer_thick: bool = False) -> TracedPath:
    """The path from ``start_sid`` to ``end_sid``, as the segments it runs through.

    This is the automatic half of naming a main vessel: pick the ostium, pick the far
    end, and take everything the vessel passes through on the way rather than clicking
    forty segments. On a tree there is exactly **one** simple path between two segments,
    so on a clean graph the answer is not a choice at all and no weighting can change it.

    Where it becomes a choice is a graph carrying a cycle -- two vessels the
    segmentation fused where they merely cross, which is common enough on epicardial
    trees that it cannot be assumed away. Two guarantees hold there:

    * **no backtracking and no loop.** The search is a Dijkstra over *nodes*, so the
      route it reconstructs is a simple node path: it cannot revisit a node, therefore
      cannot reuse a segment, and cannot double back through the one it just left. The
      start and end segments are excluded from the middle of the walk for the same
      reason -- a path must not re-enter the segment it began in.
    * **which route wins is stated, not implied.** By default the shortest by arc
      length. With ``prefer_thick``, the shortest in units of each segment's own radius,
      which routes the trace around a thin false bridge and down the vessel instead --
      the right default when the two picks are epicardial vessels and the shortcut
      between them is an artefact.

    Raises :class:`CropError` when the two are in different components, which is a
    genuine answer -- there is no path -- rather than a failure to find one.
    """
    for sid in (start_sid, end_sid):
        if not graph.has_segment(int(sid)):
            raise CropError(f"segment {sid} is not in this graph")
    start_sid, end_sid = int(start_sid), int(end_sid)
    weight = "thickness" if prefer_thick else "length"
    if start_sid == end_sid:
        return TracedPath([start_sid], [], segment_length_um(graph, start_sid), weight)

    start_seg, end_seg = graph.segment(start_sid), graph.segment(end_sid)
    goals = {int(end_seg["node1"]), int(end_seg["node2"])}
    banned = {start_sid, end_sid}

    dist: dict[int, float] = {}
    came: dict[int, tuple[int, int]] = {}  # node -> (the node before it, segment walked)
    queue: list[tuple[float, int]] = []
    # Both ends of the start segment cost nothing: the whole segment is taken either
    # way, so which end the vessel leaves by is the search's business, not the pick's.
    for nid in (int(start_seg["node1"]), int(start_seg["node2"])):
        if nid in graph.nodes and nid not in dist:
            dist[nid] = 0.0
            heapq.heappush(queue, (0.0, nid))

    reached: int | None = None
    while queue:
        cost, nid = heapq.heappop(queue)
        if cost > dist.get(nid, float("inf")):
            continue  # a stale copy, left behind by a later relaxation
        if nid in goals:
            reached = nid
            break
        for sid in sorted(graph.node_segments(nid)):
            if sid in banned or not graph.has_segment(sid):
                continue
            seg = graph.segment(sid)
            other = int(seg["node2"] if int(seg["node1"]) == nid else seg["node1"])
            if other == nid:
                continue  # a self-loop arrives back where it left
            step = cost + _trace_cost(graph, sid, prefer_thick=prefer_thick)
            if step < dist.get(other, float("inf")):
                dist[other] = step
                came[other] = (nid, int(sid))
                heapq.heappush(queue, (step, other))

    if reached is None:
        raise CropError(
            f"segment {start_sid} and segment {end_sid} are not connected: nothing "
            "runs between them. Trace between two segments of the same tree."
        )

    middle: list[int] = []
    nodes: list[int] = [reached]
    nid = reached
    while nid in came:
        previous, sid = came[nid]
        middle.append(sid)
        nodes.append(previous)
        nid = previous
    middle.reverse()
    nodes.reverse()

    segments = [start_sid, *middle, end_sid]
    length = float(sum(segment_length_um(graph, sid) for sid in segments))
    return TracedPath(segments, nodes, length, weight)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def takeoff_radius_um(graph, sid: int, from_node: int,
                      *, factor: float = TAKEOFF_FACTOR) -> float:
    """A branch's radius just past its take-off, in um.

    Delegates to ``skeleton_optimise._radius_away``: never read the radius *at* a
    junction, where the distance transform is the distance to the outside of the whole
    carina and is larger than either vessel's own radius. Returns ``nan`` when the
    segment carries nothing measurable, and callers must treat that as "not measured"
    rather than as "thin" -- an unmeasured branch is not evidence for dropping it.
    """
    from .skeleton_optimise import _radius_away

    if not graph.has_segment(sid) or from_node not in graph.nodes:
        return float("nan")
    return float(_radius_away(graph, sid, from_node, factor))


def vessel_ostia(graph, vessels: dict[str, set[int]], topo: Topology,
                 *, factor: float = TAKEOFF_FACTOR) -> dict[str, dict]:
    """Per named vessel, its most proximal segment and the radius measured there.

    That radius is the denominator of the vessel's side-branch threshold, which is why
    it is taken at the vessel's own ostium and not, say, as a mean over its length: the
    rule is "a fraction of what came in", and what came in is measured once.
    """
    out: dict[str, dict] = {}
    for name, ids in vessels.items():
        live = [sid for sid in ids if graph.has_segment(sid)]
        if not live:
            continue
        ostial = min(live, key=lambda sid: (topo.depth.get(sid, 1 << 30), sid))
        node = topo.prox_node(graph, ostial)
        out[name] = {
            "seg_id": int(ostial),
            "seg_key": segment_key(graph, ostial),
            "node": int(node),
            "radius_um": takeoff_radius_um(graph, ostial, node, factor=factor),
            "depth": int(topo.depth.get(ostial, -1)),
            "n_segments": len(live),
        }
    return out


def descent_map(graph, topo: Topology, seg_vessel: dict[int, str]) -> dict[int, str | None]:
    """Every segment -> the main vessel it descends from, walking parents upwards.

    The walk is what makes the rule anatomical rather than local: a twig five
    generations off the LAD is judged against the LAD's ostium, because that is the
    vessel whose flow it is taking a share of.
    """
    descends: dict[int, str | None] = {}
    for start in list(topo.depth):
        path: list[int] = []
        current: int | None = start
        result: str | None = None
        while current is not None:
            if current in descends:
                result = descends[current]
                break
            if current in seg_vessel:
                result = seg_vessel[current]
                break
            path.append(current)
            current = topo.parent.get(current)
        for sid in path:
            descends[sid] = result
    return descends


def ancestors_of_main(topo: Topology, main: set[int]) -> set[int]:
    """Every segment on the path from a root down to an annotated main vessel.

    An unannotated left main proximal to the LAD is not a side branch of anything, and
    a rule that removed it would take the LAD with it.
    """
    out: set[int] = set()
    for sid in main:
        current = topo.parent.get(sid)
        while current is not None and current not in out:
            out.add(current)
            current = topo.parent.get(current)
    return out


# --------------------------------------------------------------------------- #
# the rules
# --------------------------------------------------------------------------- #
def _walk(topo: Topology, protected: set[int], decide) -> dict[int, dict]:
    """Traverse each rooted component, recording the take-offs ``decide`` rejects.

    A protected segment is traversed *through* -- a main vessel is never a take-off,
    but its side branches still have to be reached. A rejected one is recorded and not
    descended into, because its whole subtree is going with it.
    """
    out: dict[int, dict] = {}
    stack = list(topo.roots)
    while stack:
        sid = stack.pop()
        if sid in protected:
            stack.extend(topo.children.get(sid, ()))
            continue
        verdict = decide(sid)
        if verdict is None:
            stack.extend(topo.children.get(sid, ()))
            continue
        out[sid] = verdict
    return out


def by_strahler(graph, topo: Topology, *, minimum: int,
                protected: set[int]) -> dict[int, dict]:
    """Take-offs whose Strahler order is below ``minimum``.

    Take-off-and-subtree rather than a flat filter over all edges. On a well-ordered
    tree the two agree, because Strahler rises monotonically towards the root; on a
    looped or stale one they do not, and it is the flat filter that leaves distal
    fragments floating with nothing to hang from.
    """
    def decide(sid: int):
        order = graph.segment(sid).get("strahler")
        if order is None:
            return None
        order = int(order)
        if order >= minimum:
            return None
        return {"rule": "strahler", "vessel": None,
                "radius_um": None, "threshold_um": None, "strahler": order}

    return _walk(topo, protected, decide)


def by_ostium_radius(graph, topo: Topology, *, min_um: float, protected: set[int],
                     factor: float = TAKEOFF_FACTOR) -> dict[int, dict]:
    """Take-offs measuring below ``min_um``. Needs no main vessels."""
    def decide(sid: int):
        radius = takeoff_radius_um(graph, sid, topo.prox_node(graph, sid), factor=factor)
        if not np.isfinite(radius) or radius >= min_um:
            return None
        return {"rule": "radius", "vessel": None,
                "radius_um": radius, "threshold_um": float(min_um)}

    return _walk(topo, protected, decide)


def by_vessel_ratio(graph, topo: Topology, ostia: dict[str, dict],
                    descends: dict[int, str | None], *, ratio: float,
                    prune_unattributed: bool, protected: set[int],
                    factor: float = TAKEOFF_FACTOR) -> dict[int, dict]:
    """Take-offs below ``ratio`` x the ostial radius of the vessel they descend from.

    The comparison is strict, matching ``epicardial_annotation``: a branch measuring
    exactly the threshold is kept. A branch that descends from no named vessel is kept
    unless ``prune_unattributed`` -- an unattributed subtree is usually one nobody has
    annotated yet, not one nobody wants.
    """
    def decide(sid: int):
        vessel = descends.get(sid)
        if vessel is None:
            if not prune_unattributed:
                return None
            return {"rule": "unattributed", "vessel": None,
                    "radius_um": None, "threshold_um": None}
        ostium = ostia.get(vessel)
        if ostium is None or not np.isfinite(ostium["radius_um"]):
            return None
        threshold = float(ratio) * float(ostium["radius_um"])
        radius = takeoff_radius_um(graph, sid, topo.prox_node(graph, sid), factor=factor)
        if not np.isfinite(radius) or radius >= threshold:
            return None
        return {"rule": "ratio", "vessel": vessel,
                "radius_um": radius, "threshold_um": threshold}

    return _walk(topo, protected, decide)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """What to crop by. Every threshold is optional; they compose."""

    min_strahler: int | None = None
    min_ostium_um: float | None = None
    ratio: float | None = None
    prune_unattributed: bool = False
    takeoff_factor: float = TAKEOFF_FACTOR
    root_edges: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        return (self.min_strahler is None and self.min_ostium_um is None
                and self.ratio is None and not self.prune_unattributed)

    def as_dict(self) -> dict:
        return {
            "min_strahler": self.min_strahler,
            "min_ostium_um": self.min_ostium_um,
            "ratio": self.ratio,
            "ratio_denominator": (1.0 / self.ratio) if self.ratio else None,
            "prune_unattributed": bool(self.prune_unattributed),
            "takeoff_factor": float(self.takeoff_factor),
            "root_edges": list(self.root_edges),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Rule":
        data = data or {}
        ratio = data.get("ratio")
        if ratio is None and data.get("ratio_denominator"):
            ratio = 1.0 / float(data["ratio_denominator"])
        return cls(
            min_strahler=data.get("min_strahler"),
            min_ostium_um=data.get("min_ostium_um"),
            ratio=ratio,
            prune_unattributed=bool(data.get("prune_unattributed", False)),
            takeoff_factor=float(data.get("takeoff_factor", TAKEOFF_FACTOR)),
            root_edges=tuple(int(x) for x in data.get("root_edges", ())),
        )


@dataclass
class CropPlan:
    """What a crop would remove, and on what evidence. Nothing is applied yet."""

    takeoffs: dict[int, dict] = field(default_factory=dict)
    drop: set[int] = field(default_factory=set)
    protected: set[int] = field(default_factory=set)
    ostia: dict[str, dict] = field(default_factory=dict)
    vessels: dict[str, set[int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    n_segments: int = 0
    segments_after: int = 0
    components_after: int = 0
    degree2_after: int = 0

    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.takeoffs.values():
            counts[record["rule"]] = counts.get(record["rule"], 0) + 1
        return counts

    def summary(self) -> str:
        if not self.n_segments:
            return "nothing loaded"
        head = (f"would drop {len(self.drop):,} of {self.n_segments:,} segment(s) "
                f"at {len(self.takeoffs):,} take-off(s)")
        rules = "; ".join(f"{name} {count:,}" for name, count in sorted(self.by_rule().items()))
        tail = (f"{self.segments_after:,} segment(s) remain in "
                f"{self.components_after:,} component(s), "
                f"{self.degree2_after:,} now degree-2")
        return head + (f" [{rules}]" if rules else "") + f"\n{tail}"


def _after_counts(graph, drop: set[int]) -> tuple[int, int, int]:
    """(segments, components, degree-2 nodes) once ``drop`` is gone, without editing."""
    keep = [seg for seg in graph.segments if seg["id"] not in drop]
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    degree: dict[int, int] = {}
    for seg in keep:
        for key in ("node1", "node2"):
            degree[seg[key]] = degree.get(seg[key], 0) + 1
        ra, rb = find(seg["node1"]), find(seg["node2"])
        if ra != rb:
            parent[ra] = rb
    roots = {find(seg["node1"]) for seg in keep}
    return len(keep), len(roots), sum(1 for d in degree.values() if d == 2)


def plan(graph, rule: Rule, *, vessels=None, drop_segments=(), prune_at=()) -> CropPlan:
    """Decide what to remove. Pure: the graph is read, never touched.

    ``drop_segments`` are hand-marked segments (that one branch, on its own);
    ``prune_at`` are ``(segment id, from node)`` pairs meaning "everything past here",
    which is the one place an undirected walk is what was actually asked for.
    """
    plan_out = CropPlan(n_segments=len(graph.segments))
    if not graph.segments:
        plan_out.notes.append("the graph has no segments")
        return plan_out

    topo = topology(graph, rule.root_edges)

    vessels = {
        name: {int(sid) for sid in ids if graph.has_segment(int(sid))}
        for name, ids in (vessels or {}).items()
    }
    vessels = {name: ids for name, ids in vessels.items() if ids}
    plan_out.vessels = vessels

    seg_vessel = {sid: name for name, ids in vessels.items() for sid in ids}
    main = set(seg_vessel)
    plan_out.ostia = vessel_ostia(graph, vessels, topo, factor=rule.takeoff_factor)
    descends = descent_map(graph, topo, seg_vessel)

    # Roots are protected because a root has no take-off to measure, and dropping one
    # takes its whole component with it.
    protected = main | ancestors_of_main(topo, main) | set(topo.roots)
    plan_out.protected = protected

    found: dict[int, dict] = {}
    if rule.min_strahler is not None:
        if not any("strahler" in seg for seg in graph.segments):
            raise CropError(
                "this graph carries no Strahler order, so --min-strahler would drop "
                "every branch; run `optimise --order` first"
            )
        found.update(by_strahler(graph, topo, minimum=int(rule.min_strahler),
                                 protected=protected))
    if rule.min_ostium_um is not None:
        for sid, record in by_ostium_radius(
            graph, topo, min_um=float(rule.min_ostium_um), protected=protected,
            factor=rule.takeoff_factor,
        ).items():
            found.setdefault(sid, record)
    if rule.ratio is not None or rule.prune_unattributed:
        if rule.ratio is not None and not plan_out.ostia:
            raise CropError(
                "the ostium ratio needs at least one named main vessel; pick one in "
                "the Crop tab, or crop by --min-ostium-um instead"
            )
        for sid, record in by_vessel_ratio(
            graph, topo, plan_out.ostia, descends,
            ratio=float(rule.ratio) if rule.ratio is not None else float("inf"),
            prune_unattributed=rule.prune_unattributed, protected=protected,
            factor=rule.takeoff_factor,
        ).items():
            found.setdefault(sid, record)

    # Shallowest first, so a take-off that is already inside another's subtree is
    # dropped from the record rather than counted -- and reported -- twice.
    drop: set[int] = set()
    for sid in sorted(found, key=lambda s: (topo.depth.get(s, 1 << 30), s)):
        if sid in drop:
            continue
        subtree = topo.descendants(sid)
        record = dict(found[sid])
        record["n_subtree_removed"] = len(subtree)
        plan_out.takeoffs[sid] = record
        drop |= subtree

    for sid in drop_segments:
        sid = int(sid)
        if not graph.has_segment(sid):
            plan_out.notes.append(f"hand-marked segment {sid} is not in this graph")
            continue
        if sid in protected:
            plan_out.notes.append(
                f"refused to drop hand-marked segment {sid}: it is a main vessel or "
                "on the path to one"
            )
            continue
        if sid not in drop:
            plan_out.takeoffs.setdefault(sid, {
                "rule": "hand", "vessel": None, "radius_um": None,
                "threshold_um": None, "n_subtree_removed": 1,
            })
        drop.add(sid)

    for sid, from_node in prune_at:
        sid, from_node = int(sid), int(from_node)
        if not graph.has_segment(sid):
            plan_out.notes.append(f"hand-marked prune at segment {sid} is not in this graph")
            continue
        try:
            subtree = graph.subtree(sid, from_node)
        except ValueError as exc:
            plan_out.notes.append(f"hand-marked prune at segment {sid}: {exc}")
            continue
        clash = subtree & protected
        if clash:
            plan_out.notes.append(
                f"refused to prune from segment {sid}: it reaches {len(clash)} "
                "protected segment(s) -- a main vessel or the path to one"
            )
            continue
        if sid not in drop:
            plan_out.takeoffs.setdefault(sid, {
                "rule": "hand-subtree", "vessel": None, "radius_um": None,
                "threshold_um": None, "n_subtree_removed": len(subtree),
            })
        drop |= subtree

    plan_out.drop = drop
    after = _after_counts(graph, drop)
    plan_out.segments_after, plan_out.components_after, plan_out.degree2_after = after
    if plan_out.degree2_after:
        plan_out.notes.append(
            f"{plan_out.degree2_after} node(s) are now degree-2; run `optimise-skeleton` "
            "to contract them"
        )
    return plan_out


def apply(graph, plan: CropPlan) -> int:
    """Remove everything the plan selected. Returns how many segments went.

    One batch, so a crop is one undo step, and ``delete_segment`` rather than
    ``delete_subtree``: the plan has already decided the exact set, and re-deriving it
    while the graph is changing underneath would re-flood a different graph.
    """
    doomed = [sid for sid in sorted(plan.drop) if graph.has_segment(sid)]
    if not doomed:
        return 0
    with graph.batch("crop"):
        for sid in doomed:
            if graph.has_segment(sid):
                graph.delete_segment(sid)
    return len(doomed)


def reorder(graph, roots=None) -> str:
    """Recompute Strahler order and generation on the cropped tree.

    The stored orders describe the tree as it *was*: every branch removed changes the
    order of everything proximal to it, so writing them through unchanged would ship a
    file whose ``strahler`` column is quietly about a different graph.

    `roots` are the chosen root **node ids**, resolved onto the cropped graph. Passing
    them matters more here than anywhere else: a crop is defined against the root, and
    re-ordering the result from ``auto_roots`` would answer with a different root than
    the one the crop was planned against -- the two would disagree about which end of
    the tree is proximal, in the same file.

    Needs the external ``skeleton_analysis`` package, which is the one dependency here
    that is not vendored; the caller is expected to report the failure and offer
    ``--no-reorder`` rather than writing a file with a stale ordering in it.
    """
    from .optimise import order

    return order(graph, roots=list(roots) if roots else None).describe()


# --------------------------------------------------------------------------- #
# the sidecar
# --------------------------------------------------------------------------- #
def document(graph, plan: CropPlan, rule: Rule, *, source=None, out=None,
             colors=None, reordered: bool = False, keys=None, fingerprint=None) -> dict:
    """The crop as a plain-JSON record: the rule, the vessels, and what it selected.

    ``keys`` and ``fingerprint`` must both be taken **before** :func:`apply` whenever
    the document is written after it. By then the selection has been deleted, so the
    keys could not name what was dropped, and the counts would describe the *result*
    while ``source_sha1`` still described the input -- a fingerprint that disagrees
    with itself is worse than none.
    """
    colors = dict(colors or {})
    keys = dict(keys) if keys is not None else segment_keys(graph)
    vessels = {}
    for i, (name, ids) in enumerate(sorted(plan.vessels.items())):
        ostium = plan.ostia.get(name, {})
        vessels[name] = {
            "color": colors.get(name, VESSEL_COLORS[i % len(VESSEL_COLORS)]),
            "seg_keys": [keys[sid] for sid in sorted(ids) if sid in keys],
            "ostium": {
                "seg_key": ostium.get("seg_key"),
                "radius_um": ostium.get("radius_um"),
            },
        }

    takeoffs = []
    for sid in sorted(plan.takeoffs):
        record = plan.takeoffs[sid]
        takeoffs.append({
            "seg_key": keys.get(sid), "seg_id": int(sid),
            "rule": record["rule"], "vessel": record.get("vessel"),
            "radius_um": record.get("radius_um"),
            "threshold_um": record.get("threshold_um"),
            "n_subtree_removed": int(record.get("n_subtree_removed", 1)),
        })

    if fingerprint is None:
        fingerprint = source_fingerprint(source, graph) if source is not None else {}
    return _plain({
        "schema": SCHEMA,
        "kind": "crop",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source) if source is not None else None,
        "source_sha1": fingerprint.get("sha1"),
        "source_counts": {k: v for k, v in fingerprint.items() if k != "sha1"},
        "rule": {**rule.as_dict(), "reorder": bool(reordered)},
        "vessels": vessels,
        "manual": {"drop_segments": [], "prune_at": []},
        "selection": {
            "n_dropped": len(plan.drop),
            "takeoffs": takeoffs,
            "dropped_seg_keys": [keys[sid] for sid in sorted(plan.drop) if sid in keys],
        },
        "result": {
            "out": str(out) if out is not None else None,
            "segments_after": plan.segments_after,
            "components_after": plan.components_after,
            "degree2_nodes": plan.degree2_after,
            "reordered": bool(reordered),
        },
        "notes": list(plan.notes),
    })


def carry_manual(document: dict, graph, drop_segments=(), prune_at=()) -> dict:
    """Record hand-marked drops into a document, keyed geometrically like the rest."""
    keys = segment_keys(graph)
    document.setdefault("manual", {})
    document["manual"]["drop_segments"] = _plain([
        {"seg_key": keys.get(int(sid)), "seg_id": int(sid)}
        for sid in drop_segments if graph.has_segment(int(sid))
    ])
    document["manual"]["prune_at"] = _plain([
        {
            "seg_key": keys.get(int(sid)), "seg_id": int(sid),
            "from_node_um": list(graph.nodes[int(node)][:3]),
        }
        for sid, node in prune_at
        if graph.has_segment(int(sid)) and int(node) in graph.nodes
    ])
    return document


def load(path) -> dict:
    """Read a crop sidecar, and check it is one of ours.

    Strict equality on the schema, like ``audit.load_decisions``: a document written by
    a different version would be interpreted on the strength of fields that mean
    something else, and a crop that removes the wrong branches quietly is the worst
    outcome available here.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, found {schema!r}")
    return document


@dataclass
class Resolved:
    """A sidecar placed onto a specific graph."""

    rule: Rule = field(default_factory=Rule)
    vessels: dict[str, set[int]] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)
    drop_segments: list[int] = field(default_factory=list)
    prune_at: list[tuple[int, int]] = field(default_factory=list)
    recorded_drop: set[int] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def resolve(graph, document: dict, *, strict_drop: bool = False) -> Resolved:
    """Place a sidecar's keys onto ``graph``.

    What each failure costs decides how it is treated:

    * a **vessel** key that will not resolve is fatal. A main vessel silently missing
      its proximal segment moves the ostium, and the ostium is the denominator of the
      threshold for every branch below it;
    * a **hand-marked** key that will not resolve is a note. It is one local decision,
      and refusing the whole crop over a stub that has moved is disproportionate;
    * a **recorded drop** that will not resolve is informational -- unless ``strict_drop``
      (a replay), where dropping fewer segments than recorded is not a replay at all.
    """
    out = Resolved(rule=Rule.from_dict(document.get("rule")))

    # Hash the file *this* graph came from, not the one the sidecar names -- comparing
    # a recorded hash against a re-hash of the same recorded path can only ever agree,
    # which would make the check decorative.
    here = source_fingerprint(getattr(graph.triple, "source", None), graph)
    recorded = document.get("source_sha1")
    counts = document.get("source_counts") or {}
    if recorded and here.get("sha1") and recorded != here["sha1"]:
        out.notes.append(
            f"this sidecar was written against a different graph "
            f"(sha1 {recorded[:8]} vs {here['sha1'][:8]}); "
            "keys are being resolved geometrically"
        )
    elif counts.get("segments") not in (None, here.get("segments")):
        # No hash to compare -- a graph held in memory, or one whose source has moved.
        out.notes.append(
            f"this sidecar was written against a graph of "
            f"{counts['segments']} segment(s); this one has {here.get('segments')}"
        )

    for name, entry in (document.get("vessels") or {}).items():
        keys = list(entry.get("seg_keys") or ())
        found, unresolved = resolve_keys(graph, keys)
        if unresolved:
            raise CropError(
                f"main vessel {name!r}: {len(unresolved)} of {len(keys)} segment(s) "
                f"could not be placed on this graph ({', '.join(unresolved[:4])}"
                f"{'...' if len(unresolved) > 4 else ''}). Re-pick it in the Crop tab."
            )
        out.vessels[name] = {found[k] for k in keys if k in found}
        if entry.get("color"):
            out.colors[name] = str(entry["color"])

    manual = document.get("manual") or {}
    for record in manual.get("drop_segments") or ():
        found, unresolved = resolve_keys(graph, [record.get("seg_key")])
        if unresolved:
            out.notes.append(
                f"hand-marked drop {record.get('seg_key')} no longer resolves; skipped"
            )
            continue
        out.drop_segments.append(next(iter(found.values())))
    for record in manual.get("prune_at") or ():
        found, unresolved = resolve_keys(graph, [record.get("seg_key")])
        if unresolved:
            out.notes.append(
                f"hand-marked prune {record.get('seg_key')} no longer resolves; skipped"
            )
            continue
        sid = next(iter(found.values()))
        point = record.get("from_node_um")
        if point is None:
            out.notes.append(f"hand-marked prune at segment {sid} has no node; skipped")
            continue
        out.prune_at.append((sid, node_at(graph, sid, point)))

    recorded_keys = list((document.get("selection") or {}).get("dropped_seg_keys") or ())
    found, unresolved = resolve_keys(graph, recorded_keys)
    out.recorded_drop = set(found.values())
    if recorded_keys:
        if unresolved and strict_drop:
            raise CropError(
                f"--replay: {len(unresolved)} of {len(recorded_keys)} recorded drop(s) "
                "no longer resolve, so this would not be a replay"
            )
        out.notes.append(
            f"{len(found)} of {len(recorded_keys)} recorded drop(s) still resolve"
        )
    return out


def replay_plan(graph, document: dict) -> CropPlan:
    """A plan that drops exactly what the sidecar recorded, re-evaluating nothing."""
    resolved = resolve(graph, document, strict_drop=True)
    out = CropPlan(n_segments=len(graph.segments), vessels=resolved.vessels)
    out.drop = set(resolved.recorded_drop)
    out.notes.extend(resolved.notes)
    out.notes.append("replayed from the sidecar; the rule was not re-evaluated")
    after = _after_counts(graph, out.drop)
    out.segments_after, out.components_after, out.degree2_after = after
    return out


CSV_COLUMNS = ("seg_id", "seg_key", "rule", "vessel",
               "radius_um", "threshold_um", "n_subtree_removed")


def write_csv(path, graph, plan: CropPlan, *, keys=None) -> Path:
    """One row per dropped take-off, for reading beside `coronary_sdf`'s own table.

    As with :func:`document`, ``keys`` must be taken before :func:`apply` if this is
    written afterwards.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = dict(keys) if keys is not None else segment_keys(graph)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for sid in sorted(plan.takeoffs):
            record = plan.takeoffs[sid]
            writer.writerow([
                sid, keys.get(sid, ""), record["rule"], record.get("vessel") or "",
                _round(record.get("radius_um")), _round(record.get("threshold_um")),
                record.get("n_subtree_removed", 1),
            ])
    return destination


def _round(value, places: int = 3):
    if value is None or not np.isfinite(value):
        return ""
    return round(float(value), places)


