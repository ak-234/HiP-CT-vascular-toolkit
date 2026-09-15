"""Convert between the viewer's ``SpatialGraph`` and the pipeline's dict triple.

Two representations of the same Amira ``HxSpatialGraph`` exist side by side:

* :class:`hipct_seg_debug.amira.SpatialGraph` -- struct of arrays, what the
  viewer reads and what ``viewer3d.centreline_polydata`` renders;
* the ``(nodes, points, segments)`` triple -- dicts keyed by id, what every
  ``coronary_sdf`` function takes.

They agree on the two things that matter: **coordinates and radii are both in
micrometres**, and both treat the sole POINT float field as the radius. So the
conversion is pure bookkeeping. The triple is the editable form, because ids
survive insertion and deletion whereas array indices do not.

Round-tripping is exact for an unedited graph, including ``Parameters{}`` and
any vertex attributes the pipeline itself discards -- :class:`Triple` carries
them through so writing back never silently drops a field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..amira import SpatialGraph

# Aliases Avizo uses for the Strahler order. ``coronary_sdf`` looks up
# ``seg["strahler"]`` and nothing else, so whatever the file called it becomes
# "strahler" in the triple and is restored on the way back out.
STRAHLER_ALIASES = ("strahler", "StrahlerOrder", "Strahler", "StrahlerNumber")

# Node tuples are (x, y, z, coordination_number). The coordination number is
# derived from the edge connectivity, never authoritative -- see `_degrees`.
NodeT = tuple[float, float, float, int]
PointT = tuple[float, float, float, float]


@dataclass
class Triple:
    """The pipeline's ``(nodes, points, segments)`` plus what it would discard.

    ``nodes[nid] -> (x, y, z, degree)``, ``points[pid] -> (x, y, z, radius)``,
    and each segment is ``{id, node1, node2, point_ids, **edge_attrs}``. All
    ids are arbitrary integers; nothing downstream assumes they are contiguous
    or sorted, which is what makes the triple editable.
    """

    nodes: dict[int, NodeT]
    points: dict[int, PointT]
    segments: list[dict[str, Any]]

    # Carried through the round trip untouched.
    vertex_attrs: dict[str, np.ndarray] = field(default_factory=dict)
    edge_attr_dtypes: dict[str, np.dtype] = field(default_factory=dict)
    strahler_field: str | None = None  # what the source file called it
    source: Path | None = None

    # Per-point scalars beside the radius, ``{name: {point id: value}}`` -- keyed by
    # id, not by position, for the reason the class docstring gives: ids survive
    # insertion and deletion, array indices do not. A point with no entry is written
    # out as ``point_attr_fill``, so a field added before an edit stays consistent
    # with the points that edit created.
    point_attrs: dict[str, dict[int, Any]] = field(default_factory=dict)
    point_attr_dtypes: dict[str, np.dtype] = field(default_factory=dict)

    def as_args(self) -> tuple[dict, dict, list]:
        """The three positional arguments every ``coronary_sdf`` entry point wants."""
        return self.nodes, self.points, self.segments

    def copy(self) -> "Triple":
        """A deep-enough copy that edits cannot reach back into the original.

        The tuples inside ``nodes``/``points`` are immutable, so shallow-copying
        those dicts is safe; ``segments`` needs its dicts and its ``point_ids``
        lists copied, because ``bridge_centerline_gaps`` and friends mutate
        ``seg["point_ids"]`` in place.
        """
        return Triple(
            nodes=dict(self.nodes),
            points=dict(self.points),
            segments=[{**s, "point_ids": list(s["point_ids"])} for s in self.segments],
            vertex_attrs={k: v.copy() for k, v in self.vertex_attrs.items()},
            edge_attr_dtypes=dict(self.edge_attr_dtypes),
            strahler_field=self.strahler_field,
            source=self.source,
            point_attrs={k: dict(v) for k, v in self.point_attrs.items()},
            point_attr_dtypes=dict(self.point_attr_dtypes),
        )


def _degrees(segments, nodes) -> dict[int, int]:
    """Vertex degree over the edge connectivity, for every node in `nodes`."""
    deg = {nid: 0 for nid in nodes}
    for seg in segments:
        for key in ("node1", "node2"):
            nid = seg[key]
            if nid in deg:
                deg[nid] += 1
    return deg


def from_spatial_graph(graph: SpatialGraph) -> Triple:
    """``SpatialGraph`` -> :class:`Triple`.

    Vertex and point ids are the array indices, so an unedited graph converts to
    contiguous 0-based ids and :func:`to_spatial_graph` reproduces the input.
    """
    offsets = graph.edge_offsets
    degree = graph.degree()

    nodes: dict[int, NodeT] = {
        vid: (float(x), float(y), float(z), int(degree[vid]))
        for vid, (x, y, z) in enumerate(graph.vertices)
    }
    points: dict[int, PointT] = {
        pid: (float(x), float(y), float(z), float(graph.thickness[pid]))
        for pid, (x, y, z) in enumerate(graph.points)
    }

    # Per-edge scalars, renaming whatever Strahler alias the file used.
    strahler_field = next((n for n in STRAHLER_ALIASES if n in graph.edge_attrs), None)
    dtypes: dict[str, np.dtype] = {}
    scalars: dict[str, np.ndarray] = {}
    for name, arr in graph.edge_attrs.items():
        arr = np.asarray(arr)
        if arr.ndim > 1 and arr.shape[1] != 1:
            continue  # vector edge attributes have no place in a segment dict
        arr = arr.ravel()
        if arr.size != graph.n_edge:
            continue
        out_name = "strahler" if name == strahler_field else name
        scalars[out_name] = arr
        dtypes[out_name] = arr.dtype

    segments: list[dict[str, Any]] = []
    for eid in range(graph.n_edge):
        a, b = int(offsets[eid]), int(offsets[eid + 1])
        seg: dict[str, Any] = {
            "id": eid,
            "node1": int(graph.connectivity[eid, 0]),
            "node2": int(graph.connectivity[eid, 1]),
            "point_ids": list(range(a, b)),
        }
        for name, arr in scalars.items():
            seg[name] = int(arr[eid]) if arr.dtype.kind in "iu" else float(arr[eid])
        segments.append(seg)

    # Point ids are the array indices here, so a flat per-point column maps onto
    # them directly; after an edit that is no longer true, which is why the triple
    # stores them keyed by id from this moment on.
    point_attrs: dict[str, dict[int, Any]] = {}
    point_dtypes: dict[str, np.dtype] = {}
    for name, arr in graph.point_attrs.items():
        arr = np.asarray(arr)
        if arr.ndim > 1 and arr.shape[1] != 1:
            continue
        arr = arr.ravel()
        if arr.size != graph.n_point:
            continue
        point_attrs[name] = {pid: v.item() for pid, v in enumerate(arr)}
        point_dtypes[name] = arr.dtype

    return Triple(
        nodes=nodes,
        points=points,
        segments=segments,
        vertex_attrs={k: np.asarray(v).copy() for k, v in graph.vertex_attrs.items()},
        edge_attr_dtypes=dtypes,
        strahler_field=strahler_field,
        source=graph.path,
        point_attrs=point_attrs,
        point_attr_dtypes=point_dtypes,
    )


def vertex_node_ids(triple: Triple) -> list[int]:
    """Node ids in ``to_spatial_graph`` vertex order -- the inverse of its remap.

    Vertex ``i`` of the written graph is node ``vertex_node_ids(triple)[i]``. Anything
    that gets a *vertex index* back from a tool working on the converted graph -- the
    interactive root picker returns one -- needs this to name the node again, and
    re-deriving the rule by copy-paste is how the two drift apart.
    """
    used_nodes = {seg[k] for seg in triple.segments for k in ("node1", "node2")}
    return sorted(nid for nid in triple.nodes if nid in used_nodes)


def to_spatial_graph(triple: Triple, path: str | Path | None = None) -> SpatialGraph:
    """:class:`Triple` -> ``SpatialGraph``, renumbering ids to array indices.

    Vertices keep their relative order (sorted by id) and points are laid out in
    segment order, which is the layout Amira requires: ``EdgePointCoordinates``
    is one flat array sliced by the cumulative sum of ``NumEdgePoints``.

    A node with no incident segment is dropped -- Amira has no way to express an
    isolated vertex in a spatial graph, and leaving it in would shift every
    subsequent index. :func:`vertex_node_ids` inverts the renumbering.
    """
    nodes, points, segments = triple.nodes, triple.points, triple.segments

    keep = vertex_node_ids(triple)
    remap = {nid: i for i, nid in enumerate(keep)}

    vertices = np.array([nodes[nid][:3] for nid in keep], dtype=np.float64).reshape(-1, 3)
    connectivity = np.array(
        [[remap[s["node1"]], remap[s["node2"]]] for s in segments], dtype=np.int64
    ).reshape(-1, 2)
    n_edge_points = np.array([len(s["point_ids"]) for s in segments], dtype=np.int64)

    flat = [pid for seg in segments for pid in seg["point_ids"]]
    coords = np.array([points[pid][:3] for pid in flat], dtype=np.float64).reshape(-1, 3)
    thickness = np.array([points[pid][3] for pid in flat], dtype=np.float64)

    # Rebuild per-edge arrays, restoring the source file's Strahler field name.
    reserved = {"id", "node1", "node2", "point_ids"}
    names = {k for seg in segments for k in seg if k not in reserved}
    edge_attrs: dict[str, np.ndarray] = {}
    for name in sorted(names):
        dtype = triple.edge_attr_dtypes.get(name, np.float64)
        fill = 0 if np.dtype(dtype).kind in "iu" else 0.0
        col = np.array([seg.get(name, fill) for seg in segments], dtype=dtype)
        out_name = triple.strahler_field if name == "strahler" and triple.strahler_field else name
        edge_attrs[out_name] = col

    # Per-point scalars, laid out in the same flat segment order as `coords`. A point
    # created by an edit after the field was attached has no entry, so it is filled
    # rather than dropped: a short column would be silently discarded by the writer,
    # which is the one outcome worse than a filled one.
    point_attrs: dict[str, np.ndarray] = {}
    for name, by_id in triple.point_attrs.items():
        dtype = np.dtype(triple.point_attr_dtypes.get(name, np.float64))
        fill = 0 if dtype.kind in "iub" else np.nan
        point_attrs[name] = np.array(
            [by_id.get(pid, fill) for pid in flat], dtype=dtype
        )

    # Vertex attributes are per-vertex, so they must follow the same drop+remap.
    vertex_attrs = {}
    for name, arr in triple.vertex_attrs.items():
        arr = np.asarray(arr)
        if len(arr) >= len(keep):
            idx = [nid for nid in keep if nid < len(arr)]
            if len(idx) == len(keep):
                vertex_attrs[name] = arr[idx]

    return SpatialGraph(
        path=Path(path) if path is not None else (triple.source or Path("<edited>")),
        n_vertex=len(keep),
        n_edge=len(segments),
        n_point=len(flat),
        vertices=vertices,
        connectivity=connectivity,
        n_edge_points=n_edge_points,
        points=coords,
        thickness=thickness,
        edge_attrs=edge_attrs,
        vertex_attrs=vertex_attrs,
        point_attrs=point_attrs,
    )


def read_triple(path: str | Path) -> Triple:
    """Parse an ASCII ``.am`` spatial graph straight into a :class:`Triple`."""
    from ..amira import read_spatial_graph

    return from_spatial_graph(read_spatial_graph(path))
