"""The editable skeleton graph.

Wraps a :class:`~.adapter.Triple` with reversible operations and the incidence
index the operations need. Every mutation goes through
:meth:`EditableGraph._put_node` / ``_put_point`` / ``_put_segment``, which is
what makes undo automatic: while an operation runs, a recorder stashes the first
value it sees for each touched id, so the inverse falls out of the edit rather
than having to be written by hand for each operation.

The operation set follows ``VascularMD``'s ``ArterialTree`` -- the one design on
this machine that thought about keeping a point-level and a topology-level view
of a vessel tree consistent under editing -- with its ``apply=True/False``
batching flag becoming :meth:`EditableGraph.batch`.

Two Amira conventions are load-bearing throughout:

* an edge's point run is ordered from ``node1`` to ``node2``, and its first and
  last points sit *on* those nodes;
* points are owned by exactly one edge. Two edges meeting at a node each carry
  their own copy of the joint position.

Break either and Avizo still opens the file, but ``coronary_sdf`` produces
fragmented tubes at the joins.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .adapter import NodeT, PointT, Triple, to_spatial_graph
from .history import Command, Composite, History, Patch

# Two node positions closer than this (um) are the same junction as far as
# editing is concerned. Matches coronary_sdf's NODE_COINCIDENCE_EPS_MM (0.01 mm),
# which `build_adjacency` uses to weld Avizo's near-duplicate nodes.
NODE_EPS_UM = 10.0


class _Recorder:
    """Collects the before/after state of everything an operation touches."""

    __slots__ = ("before", "after", "coords", "seg_ids")

    def __init__(self) -> None:
        # {"nodes"|"points"|"segments": {id: value-or-None}}
        self.before: dict[str, dict[int, Any]] = {"nodes": {}, "points": {}, "segments": {}}
        self.after: dict[str, dict[int, Any]] = {"nodes": {}, "points": {}, "segments": {}}
        self.coords: list[tuple[float, float, float]] = []
        # Segments the edit *affects*, which is a superset of the segments whose
        # own record changed: moving a point rewrites no segment but still
        # invalidates the surface around its owner.
        self.seg_ids: set[int] = set()

    def touch(self, kind: str, key: int, old: Any, new: Any) -> None:
        # Only the *first* `old` matters: it is the state to restore.
        self.before[kind].setdefault(key, old)
        self.after[kind][key] = new
        if kind in ("nodes", "points"):
            for value in (old, new):
                if value is not None:
                    self.coords.append(tuple(value[:3]))
        else:
            self.seg_ids.add(key)


class DeltaCommand(Command):
    """A reversible edit stored as before/after values for the ids it touched.

    Uniform across every operation, so there is exactly one place where undo can
    be wrong. ``None`` means "absent" in either direction, which covers creation
    and deletion without a special case.
    """

    def __init__(self, label: str, recorder: _Recorder):
        self.label = label
        self._before = {k: dict(v) for k, v in recorder.before.items()}
        self._after = {k: dict(v) for k, v in recorder.after.items()}
        self._seg_ids = frozenset(recorder.seg_ids)
        pts = np.asarray(recorder.coords, dtype=np.float64).reshape(-1, 3)
        self._aabb = np.array([pts.min(axis=0), pts.max(axis=0)]) if len(pts) else None

    @property
    def patch(self) -> Patch:
        return Patch(self._seg_ids, self._aabb)

    def do(self, graph: "EditableGraph") -> Patch:
        graph._apply(self._after)
        return self.patch

    def undo(self, graph: "EditableGraph") -> Patch:
        graph._apply(self._before)
        return self.patch


class EditableGraph:
    """A skeleton graph you can edit, with undo and per-edit patch reporting."""

    def __init__(self, triple: Triple):
        self.triple = triple
        self.history = History(self)
        self.root_pref: set[int] = set()
        self.last_patch: Patch = Patch.empty()

        self._rec: _Recorder | None = None
        self._batch: list[Command] | None = None
        # Nodes whose coordination number may be stale. Degree is *derived* from
        # the incidence index and recomputed after every write, never restored
        # from a recorded value -- an undo that put back a stale count would
        # leave the graph disagreeing with itself.
        self._dirty_nodes: set[int] = set()

        self._seg_by_id: dict[int, dict] = {s["id"]: s for s in triple.segments}
        self._node_segs: dict[int, set[int]] = {nid: set() for nid in triple.nodes}
        self._point_seg: dict[int, int] = {}
        for seg in triple.segments:
            for key in ("node1", "node2"):
                self._node_segs.setdefault(seg[key], set()).add(seg["id"])
            for pid in seg["point_ids"]:
                self._point_seg[pid] = seg["id"]

        self._next_node = (max(triple.nodes) + 1) if triple.nodes else 0
        self._next_point = (max(triple.points) + 1) if triple.points else 0
        self._next_seg = (max(self._seg_by_id) + 1) if self._seg_by_id else 0

    # ---------------------------------------------------------------- reading

    @property
    def nodes(self) -> dict[int, NodeT]:
        return self.triple.nodes

    @property
    def points(self) -> dict[int, PointT]:
        return self.triple.points

    @property
    def segments(self) -> list[dict]:
        return self.triple.segments

    def segment(self, sid: int) -> dict:
        return self._seg_by_id[sid]

    def has_segment(self, sid: int) -> bool:
        return sid in self._seg_by_id

    def segment_ids(self) -> list[int]:
        return [s["id"] for s in self.triple.segments]

    def node_segments(self, nid: int) -> set[int]:
        return self._node_segs.get(nid, set())

    def degree(self, nid: int) -> int:
        return len(self._node_segs.get(nid, ()))

    def coords(self, sid: int) -> np.ndarray:
        """(N, 3) centreline positions of a segment, in um, node1 -> node2."""
        pts = self.points
        return np.array(
            [pts[p][:3] for p in self._seg_by_id[sid]["point_ids"]], dtype=np.float64
        ).reshape(-1, 3)

    def radii(self, sid: int) -> np.ndarray:
        """(N,) radii of a segment's points, in um."""
        pts = self.points
        return np.array(
            [pts[p][3] for p in self._seg_by_id[sid]["point_ids"]], dtype=np.float64
        )

    def point_order(self) -> list[int]:
        """Point ids in render order -- the flat concatenation over segments.

        ``viewer3d.centreline_polydata`` builds its points in exactly this order,
        and its pick ids index into it. This is the bridge from "the user clicked
        vertex 12345" to "that is point id P of segment S".
        """
        return [pid for seg in self.triple.segments for pid in seg["point_ids"]]

    def segment_of_point(self) -> dict[int, int]:
        """``{point id: owning segment id}``, maintained incrementally."""
        return self._point_seg

    def endpoints(self) -> list[int]:
        """Degree-1 node ids -- the candidates every reconnector starts from."""
        return [nid for nid in self.nodes if self.degree(nid) == 1]

    def components(self) -> list[set[int]]:
        """Connected components as sets of segment ids, largest first."""
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for seg in self.triple.segments:
            ra, rb = find(seg["node1"]), find(seg["node2"])
            if ra != rb:
                parent[ra] = rb

        groups: dict[int, set[int]] = {}
        for seg in self.triple.segments:
            groups.setdefault(find(seg["node1"]), set()).add(seg["id"])
        return sorted(groups.values(), key=len, reverse=True)

    def bounds(self, seg_ids: Iterable[int] | None = None) -> np.ndarray:
        """(2, 3) world AABB over the given segments, or the whole graph."""
        ids = list(seg_ids) if seg_ids is not None else self.segment_ids()
        pts = self.points
        coords = [
            pts[p][:3]
            for sid in ids
            if sid in self._seg_by_id
            for p in self._seg_by_id[sid]["point_ids"]
        ]
        arr = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        if not len(arr):
            return np.zeros((2, 3))
        return np.array([arr.min(axis=0), arr.max(axis=0)])

    # ------------------------------------------------------------- conversion

    def to_triple(self) -> Triple:
        return self.triple

    def to_spatial_graph(self, path=None):
        return to_spatial_graph(self.triple, path)

    def snapshot(self) -> Triple:
        """An independent copy, for handing to a worker thread."""
        return self.triple.copy()

    # ---------------------------------------------------------- edit plumbing

    def _put_node(self, nid: int, value: NodeT | None) -> None:
        old = self.nodes.get(nid)
        if self._rec is not None:
            self._rec.touch("nodes", nid, old, value)
            self._rec.seg_ids.update(self._node_segs.get(nid, ()))
        self._dirty_nodes.add(nid)
        if value is None:
            self.nodes.pop(nid, None)
            self._node_segs.pop(nid, None)
        else:
            self.nodes[nid] = value
            self._node_segs.setdefault(nid, set())

    def _put_point(self, pid: int, value: PointT | None) -> None:
        old = self.points.get(pid)
        if self._rec is not None:
            self._rec.touch("points", pid, old, value)
            # Attribute the point to its owner so a pure geometry edit still
            # names the segment whose surface has to be rebuilt.
            owner = self._point_seg.get(pid)
            if owner is not None:
                self._rec.seg_ids.add(owner)
        if value is None:
            self.points.pop(pid, None)
        else:
            self.points[pid] = value

    def _put_segment(self, sid: int, value: dict | None) -> None:
        old = self._seg_by_id.get(sid)
        if self._rec is not None:
            self._rec.touch(
                "segments",
                sid,
                {**old, "point_ids": list(old["point_ids"])} if old else None,
                {**value, "point_ids": list(value["point_ids"])} if value else None,
            )
        self._write_segment(sid, value)

    def _write_segment(self, sid: int, value: dict | None) -> None:
        """Install a segment and repair both derived indices. No recording."""
        old = self._seg_by_id.get(sid)
        if old is not None:
            for key in ("node1", "node2"):
                self._node_segs.get(old[key], set()).discard(sid)
                self._dirty_nodes.add(old[key])
            for pid in old["point_ids"]:
                if self._point_seg.get(pid) == sid:
                    del self._point_seg[pid]
        if value is None:
            if old is not None:
                self._seg_by_id.pop(sid, None)
                self.triple.segments = [s for s in self.triple.segments if s["id"] != sid]
            return
        value = {**value, "id": sid}
        if old is None:
            self.triple.segments.append(value)
        else:
            idx = next(i for i, s in enumerate(self.triple.segments) if s["id"] == sid)
            self.triple.segments[idx] = value
        self._seg_by_id[sid] = value
        for key in ("node1", "node2"):
            self._node_segs.setdefault(value[key], set()).add(sid)
            self._dirty_nodes.add(value[key])
        for pid in value["point_ids"]:
            self._point_seg[pid] = sid

    def _apply(self, state: dict[str, dict[int, Any]]) -> None:
        """Restore/install a recorded state, without recording it in turn."""
        rec, self._rec = self._rec, None
        try:
            for pid, value in state["points"].items():
                if value is None:
                    self.points.pop(pid, None)
                else:
                    self.points[pid] = value
            for sid, value in state["segments"].items():
                self._write_segment(sid, value)
            for nid, value in state["nodes"].items():
                self._dirty_nodes.add(nid)
                if value is None:
                    self.nodes.pop(nid, None)
                    self._node_segs.pop(nid, None)
                else:
                    self.nodes[nid] = value
                    self._node_segs.setdefault(nid, set())
            self._flush_degrees()
        finally:
            self._rec = rec

    def _flush_degrees(self) -> None:
        """Recompute the coordination number of every node marked dirty.

        Degree is derived from the incidence index rather than recorded, which
        is what makes it correct in both directions: an operation that only
        changes *which* segments touch a node -- adding an edge, say -- records
        no node value at all, so there is nothing for an undo to restore.
        """
        for nid in self._dirty_nodes:
            cur = self.nodes.get(nid)
            if cur is not None:
                self.nodes[nid] = (cur[0], cur[1], cur[2], self.degree(nid))
        self._dirty_nodes.clear()

    @contextmanager
    def _operation(self, label: str) -> Iterator[_Recorder]:
        """Record one operation and push it onto the history (or the open batch).

        On exit ``self.last_patch`` holds what the operation touched; operations
        return that rather than computing a patch of their own, so the history
        entry and the rebuild request can never disagree.
        """
        if self._rec is not None:
            raise RuntimeError("nested edit operations are not supported; use batch()")
        rec = _Recorder()
        self._rec = rec
        try:
            yield rec
        finally:
            self._rec = None
        self._flush_degrees()
        command = DeltaCommand(label, rec)
        self.last_patch = command.patch
        if self._batch is not None:
            self._batch.append(command)
        else:
            self.history.push_done(command)

    @contextmanager
    def batch(self, label: str = "batch") -> Iterator[None]:
        """Collapse every edit made inside into one undo step.

        Edits still apply as they are issued -- a reconnection needs to see the
        node it just created -- they simply share a history entry, and
        ``last_patch`` ends up covering all of them.
        """
        if self._batch is not None:
            yield  # already batching; the outermost context owns the entry
            return
        self._batch = []
        merged = Patch.empty()
        try:
            yield
        finally:
            commands, self._batch = self._batch, None
        for command in commands:
            merged = merged.merged(command.patch)
        if commands:
            self.history.push_done(Composite(commands, label))
            self.last_patch = merged

    def _new_node_id(self) -> int:
        self._next_node += 1
        return self._next_node - 1

    def _new_point_id(self) -> int:
        self._next_point += 1
        return self._next_point - 1

    def _new_segment_id(self) -> int:
        self._next_seg += 1
        return self._next_seg - 1

    # -------------------------------------------------------------- undo/redo

    def undo(self) -> Patch | None:
        patch = self.history.undo()
        if patch is not None:
            self.last_patch = patch
        return patch

    def redo(self) -> Patch | None:
        patch = self.history.redo()
        if patch is not None:
            self.last_patch = patch
        return patch

    # ------------------------------------------------------------- operations

    def move_point(self, pid: int, xyz: Sequence[float]) -> Patch:
        """Move one centreline point. Drags a coincident node with it."""
        x, y, z = (float(v) for v in xyz)
        with self._operation("move point"):
            old = self.points[pid]
            self._put_point(pid, (x, y, z, old[3]))
            # A point sitting on a node *is* that node's position; letting them
            # drift apart is how tubes come adrift from their junction.
            for seg in list(self.triple.segments):
                ids = seg["point_ids"]
                if not ids:
                    continue
                if pid == ids[0]:
                    self._move_node(seg["node1"], (x, y, z))
                if pid == ids[-1]:
                    self._move_node(seg["node2"], (x, y, z))
        return self.last_patch

    def _move_node(self, nid: int, xyz: Sequence[float]) -> None:
        node = self.nodes.get(nid)
        if node is None:
            return
        self._put_node(nid, (float(xyz[0]), float(xyz[1]), float(xyz[2]), node[3]))

    def move_node(self, nid: int, xyz: Sequence[float]) -> Patch:
        """Move a junction, taking every incident segment's boundary point with it."""
        x, y, z = (float(v) for v in xyz)
        with self._operation("move node"):
            self._move_node(nid, (x, y, z))
            for sid in list(self.node_segments(nid)):
                seg = self._seg_by_id[sid]
                ids = seg["point_ids"]
                if not ids:
                    continue
                for end, which in ((ids[0], "node1"), (ids[-1], "node2")):
                    if seg[which] == nid:
                        r = self.points[end][3]
                        self._put_point(end, (x, y, z, r))
        return self.last_patch

    def set_radius(self, pid: int, radius: float) -> Patch:
        """Set one point's radius (um)."""
        with self._operation("set radius"):
            x, y, z, _ = self.points[pid]
            self._put_point(pid, (x, y, z, float(radius)))
        return self.last_patch

    def scale_radii(self, sid: int, factor: float) -> Patch:
        """Multiply every radius along a segment -- the usual collapse fix."""
        with self._operation("scale radii"):
            for pid in list(self._seg_by_id[sid]["point_ids"]):
                x, y, z, r = self.points[pid]
                self._put_point(pid, (x, y, z, float(r * factor)))
        return self.last_patch

    def set_segment_radii(self, sid: int, radii: Sequence[float]) -> Patch:
        """Replace a segment's radii wholesale (what the oblique corrector emits)."""
        ids = list(self._seg_by_id[sid]["point_ids"])
        values = np.asarray(radii, dtype=np.float64).ravel()
        if len(values) != len(ids):
            raise ValueError(f"segment {sid} has {len(ids)} points, got {len(values)} radii")
        with self._operation("set segment radii"):
            for pid, r in zip(ids, values):
                x, y, z, _ = self.points[pid]
                self._put_point(pid, (x, y, z, float(r)))
        return self.last_patch

    def set_segment_attrs(self, sid: int, attrs: dict[str, Any]) -> Patch:
        """Set per-edge scalar fields on one segment, undoably.

        The topology keys are refused rather than merged: rewriting ``node1`` or
        ``point_ids`` through an attribute setter would leave the incidence
        indices describing a graph that no longer exists, and the operations that
        do change those maintain them deliberately.
        """
        reserved = {"id", "node1", "node2", "point_ids"} & set(attrs)
        if reserved:
            raise ValueError(f"{sorted(reserved)} are topology, not attributes")
        with self._operation("set segment attributes"):
            self._put_segment(sid, {**self._seg_by_id[sid], **attrs})
        return self.last_patch

    def set_segment_coords(self, sid: int, coords) -> Patch:
        """Replace a segment's point positions wholesale (what the smoother emits).

        The bulk counterpart of :meth:`move_point`, and the reason it exists: that
        one scans *every* segment looking for a coincident node, which is O(E) a
        call, so smoothing 37k points through it is 45 million dict lookups. Here
        the only points that can sit on a node are the two ends, and which node
        each belongs to is already recorded, so the same invariant costs O(degree).
        """
        seg = self._seg_by_id[sid]
        ids = list(seg["point_ids"])
        xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        if len(xyz) != len(ids):
            raise ValueError(
                f"segment {sid} has {len(ids)} points, got {len(xyz)} coordinates"
            )
        if not ids:
            return Patch.empty()
        with self._operation("set segment coords"):
            for pid, p in zip(ids, xyz):
                r = self.points[pid][3]
                self._put_point(pid, (float(p[0]), float(p[1]), float(p[2]), r))
            # An end that actually moved takes its node -- and every other
            # segment's boundary point on that node -- with it, or tubes come
            # adrift from their junction exactly as `move_point` warns.
            for end, which in ((0, "node1"), (-1, "node2")):
                nid = seg[which]
                node = self.nodes.get(nid)
                if node is None or np.allclose(xyz[end], node[:3]):
                    continue
                self._move_node(nid, xyz[end])
                for other in list(self.node_segments(nid)):
                    if other == sid:
                        continue
                    o = self._seg_by_id[other]
                    oids = o["point_ids"]
                    if not oids:
                        continue
                    for oend, owhich in ((oids[0], "node1"), (oids[-1], "node2")):
                        if o[owhich] == nid:
                            r = self.points[oend][3]
                            self._put_point(
                                oend,
                                (float(xyz[end][0]), float(xyz[end][1]),
                                 float(xyz[end][2]), r),
                            )
        return self.last_patch

    def insert_point(self, sid: int, index: int, xyz: Sequence[float], radius: float) -> Patch:
        """Insert a centreline point into a segment before position `index`."""
        seg = self._seg_by_id[sid]
        if not 0 < index <= len(seg["point_ids"]) - 1:
            raise ValueError("cannot insert before the first or after the last point")
        with self._operation("insert point"):
            pid = self._new_point_id()
            self._put_point(pid, (float(xyz[0]), float(xyz[1]), float(xyz[2]), float(radius)))
            ids = list(seg["point_ids"])
            ids.insert(index, pid)
            self._put_segment(sid, {**seg, "point_ids": ids})
        return self.last_patch

    def delete_point(self, sid: int, index: int) -> Patch:
        """Delete an interior centreline point. Endpoints belong to nodes and stay."""
        seg = self._seg_by_id[sid]
        ids = list(seg["point_ids"])
        if not 0 < index < len(ids) - 1:
            raise ValueError("cannot delete a segment's first or last point")
        with self._operation("delete point"):
            pid = ids.pop(index)
            self._put_segment(sid, {**seg, "point_ids": ids})
            self._put_point(pid, None)
        return self.last_patch

    def delete_segment(self, sid: int) -> Patch:
        """Remove a segment, its points, and any node left with no edges."""
        with self._operation("delete segment"):
            self._delete_segment(sid)
        return self.last_patch

    def _delete_segment(self, sid: int) -> None:
        seg = self._seg_by_id[sid]
        nodes = (seg["node1"], seg["node2"])
        for pid in list(seg["point_ids"]):
            self._put_point(pid, None)
        self._put_segment(sid, None)
        for nid in nodes:
            if not self.node_segments(nid):
                self._put_node(nid, None)

    def subtree(self, sid: int, from_node: int) -> set[int]:
        """Segment ids reachable through `sid` when walking away from `from_node`."""
        seg = self._seg_by_id[sid]
        if from_node not in (seg["node1"], seg["node2"]):
            raise ValueError(f"node {from_node} is not an endpoint of segment {sid}")
        far = seg["node2"] if seg["node1"] == from_node else seg["node1"]
        seen = {sid}
        visited = {from_node}
        stack = [far]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for other in self.node_segments(nid):
                if other in seen:
                    continue
                seen.add(other)
                edge = self._seg_by_id[other]
                stack.append(edge["node2"] if edge["node1"] == nid else edge["node1"])
        return seen

    def delete_subtree(self, sid: int, from_node: int) -> Patch:
        """Prune a branch and everything downstream of it.

        Naturally a batch: each removal has to see the incidence the previous one
        left behind before it can tell whether a node is now orphaned.
        """
        doomed = sorted(self.subtree(sid, from_node))
        with self.batch("delete subtree"):
            for other in doomed:
                if other in self._seg_by_id:
                    with self._operation("delete segment"):
                        self._delete_segment(other)
        return self.last_patch

    def split_segment(self, sid: int, index: int) -> tuple[int, int, int]:
        """Split a segment at an interior point, creating a node there.

        Returns ``(new_node_id, segment_a, segment_b)``. This is the primitive a
        T-junction reconnection needs: an endpoint cannot attach to the middle of
        a vessel until that middle is a node.
        """
        seg = self._seg_by_id[sid]
        ids = list(seg["point_ids"])
        if not 0 < index < len(ids) - 1:
            raise ValueError("split index must be strictly inside the segment")

        with self._operation("split segment"):
            x, y, z, r = self.points[ids[index]]
            nid = self._new_node_id()
            self._put_node(nid, (x, y, z, 0))

            # The joint point is duplicated, not shared: each edge owns its run.
            twin = self._new_point_id()
            self._put_point(twin, (x, y, z, r))

            attrs = {
                k: v for k, v in seg.items()
                if k not in ("id", "node1", "node2", "point_ids")
            }
            sid_a, sid_b = self._new_segment_id(), self._new_segment_id()
            self._put_segment(
                sid_a,
                {"id": sid_a, "node1": seg["node1"], "node2": nid,
                 "point_ids": ids[: index + 1], **attrs},
            )
            self._put_segment(
                sid_b,
                {"id": sid_b, "node1": nid, "node2": seg["node2"],
                 "point_ids": [twin] + ids[index + 1:], **attrs},
            )
            self._put_segment(sid, None)
        return nid, sid_a, sid_b

    def add_node(self, xyz: Sequence[float]) -> int:
        """Create a free-standing node. Only useful mid-batch."""
        with self._operation("add node"):
            nid = self._new_node_id()
            self._put_node(nid, (float(xyz[0]), float(xyz[1]), float(xyz[2]), 0))
        return nid

    def add_segment(
        self,
        node1: int,
        node2: int,
        coords: np.ndarray,
        radii: Sequence[float],
        attrs: dict[str, Any] | None = None,
    ) -> int:
        """Connect two nodes with a new centreline. What every reconnector emits.

        `coords` must run from `node1` to `node2`; its endpoints are snapped onto
        the two node positions so the join is exact rather than nearly exact.
        """
        pts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        rad = np.asarray(radii, dtype=np.float64).ravel()
        if len(pts) < 2:
            raise ValueError("a segment needs at least two points")
        if len(rad) != len(pts):
            raise ValueError(f"{len(pts)} coords but {len(rad)} radii")
        for nid in (node1, node2):
            if nid not in self.nodes:
                raise KeyError(f"no such node: {nid}")

        pts = pts.copy()
        pts[0] = self.nodes[node1][:3]
        pts[-1] = self.nodes[node2][:3]

        with self._operation("add segment"):
            ids = []
            for (x, y, z), r in zip(pts, rad):
                pid = self._new_point_id()
                self._put_point(pid, (float(x), float(y), float(z), float(r)))
                ids.append(pid)
            sid = self._new_segment_id()
            self._put_segment(
                sid,
                {"id": sid, "node1": node1, "node2": node2, "point_ids": ids,
                 **(attrs or {})},
            )
        return sid

    def merge_nodes(self, keep: int, drop: int) -> Patch:
        """Weld two junctions into one, moving `drop`'s segments onto `keep`."""
        if keep == drop or drop not in self.nodes:
            return Patch.empty()
        pos = self.nodes[keep][:3]
        with self._operation("merge nodes"):
            for sid in list(self.node_segments(drop)):
                seg = dict(self._seg_by_id[sid])
                ids = list(seg["point_ids"])
                if seg["node1"] == drop:
                    seg["node1"] = keep
                    if ids:
                        self._put_point(ids[0], (*pos, self.points[ids[0]][3]))
                if seg["node2"] == drop:
                    seg["node2"] = keep
                    if ids:
                        self._put_point(ids[-1], (*pos, self.points[ids[-1]][3]))
                if seg["node1"] == seg["node2"]:
                    # The weld turned this edge into a self-loop; it has no
                    # geometry left to contribute.
                    self._delete_segment(sid)
                else:
                    self._put_segment(sid, seg)
            self._put_node(drop, None)
        return self.last_patch

    def weld_coincident_nodes(self, eps: float = NODE_EPS_UM) -> Patch:
        """Merge every pair of nodes closer than `eps` um.

        Avizo emits duplicate vertices at junctions; unwelded they read as two
        degree-1 endpoints sitting on top of each other, which makes every
        endpoint-based reconnector propose a zero-length bridge.
        """
        from scipy.spatial import cKDTree

        ids = list(self.nodes)
        if len(ids) < 2:
            return Patch.empty()
        pos = np.array([self.nodes[n][:3] for n in ids], dtype=np.float64)
        pairs = sorted(cKDTree(pos).query_pairs(eps))
        if not pairs:
            return Patch.empty()
        with self.batch("weld coincident nodes"):
            for i, j in pairs:
                a, b = ids[i], ids[j]
                if a in self.nodes and b in self.nodes:
                    self.merge_nodes(a, b)
        return self.last_patch

    def reroot(self, nid: int) -> Patch:
        """Mark a node as an inlet.

        Nothing about the geometry changes; ``build_directed_topology`` takes a
        ``root_pref`` and this is how the viewer supplies it, so parent/child
        relationships -- and therefore the anti-bridge carve -- come out right.
        """
        self.root_pref = {int(nid)}
        self.last_patch = Patch(frozenset(self.node_segments(nid)), None)
        return self.last_patch
