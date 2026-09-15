"""Which node roots each tree, recorded so it only has to be chosen once.

Every stage that turns a skeleton into anatomy needs to know where the blood comes
in: :func:`~.radius_perimeter._directed_topology` roots each component to infer
parent and child at a bifurcation, ``order_forest`` counts Strahler up from the root,
and :mod:`~.crop` protects roots from being cropped away. All of them currently guess
-- ``auto_roots`` takes the largest-radius edge and then its lower-coordination
endpoint -- and the only manual override is ``--root-edge``, a segment id read out of
a viewer by hand.

This module records the answer instead. :func:`~.pick` opens one 3-D window per tree
(``skeleton_analysis.ordering.root_picker``, which already does exactly this), and
:func:`document` writes what was chosen to a sidecar every downstream command can
read.

Coordinates are authoritative
-----------------------------
The sidecar records each root's **world position** first, its segment key second, and
its ids only as a note. :mod:`~.crop`'s geometric key survives a *renumbering*, which
is all a crop needs; a roots sidecar has to survive a **re-skeletonisation**, where
the segments themselves are new objects in roughly the same place. Only a coordinate
survives that, so :func:`resolve` falls back to the nearest node when the key misses.

A miss is never fatal. Re-picking roots for a repaired graph is the intended
workflow, so an unresolvable tree earns a note and falls through to the automatic
root, exactly as :func:`~.crop.resolve` treats a segment it cannot place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .components import root_edge_for_node, tree_of_edge
from .crop import node_at, resolve_keys, segment_key, source_fingerprint
from .reconnect.geodesic.audit import _plain, write  # noqa: F401

SCHEMA = "hipct.roots/1"

# How far a recorded root may be from the nearest node before the match is refused,
# in multiples of the coarsest voxel. Four voxels is wide enough to absorb a
# re-skeletonisation moving a free end along the vessel, and far short of the distance
# to the next branch.
SNAP_VOXELS = 4.0


# --------------------------------------------------------------------------- #
# choosing
# --------------------------------------------------------------------------- #
def pick(graph, *, color_by: str = "strahler", style: str = "contour",
         n_sides: int = 16, ring_stride: int = 1, hover: bool = True,
         off_screen: bool = False, screenshot=None, preselect=None) -> list[int]:
    """Open the interactive picker; return one root **node id** per tree picked.

    ``pick_roots`` works on a ``skeleton_analysis`` ``SpatialGraph`` and returns
    indices into *its* vertex array, which are not this graph's node ids -- the
    conversion drops nodes with no incident segment and renumbers the rest.
    :func:`~.adapter.vertex_node_ids` is the exact inverse, and every translation is
    cross-checked against the coordinate, because a root silently attached to the
    wrong node would misdirect every parent/child inference downstream.

    Raises ``ImportError`` when PyVista or ``skeleton_analysis`` is missing; the
    caller is expected to fall back to ``auto_roots`` rather than fail.
    """
    from skeleton_analysis.ordering.root_picker import pick_roots

    from .adapter import vertex_node_ids
    from .optimise import as_sa_graph

    sa = as_sa_graph(graph)
    triple = graph.triple if hasattr(graph, "triple") else graph
    node_of_vertex = vertex_node_ids(triple)

    vertices = pick_roots(
        sa, color_by=color_by, style=style, n_sides=n_sides, ring_stride=ring_stride,
        hover=hover, off_screen=off_screen, screenshot=screenshot, preselect=preselect,
    )

    coords = np.asarray(sa.vertex_coords, dtype=np.float64)
    out = []
    for vi in vertices:
        vi = int(vi)
        if not 0 <= vi < len(node_of_vertex):
            raise ValueError(f"the picker returned vertex {vi}, which this graph has no node for")
        nid = int(node_of_vertex[vi])
        here = np.asarray(graph.nodes[nid][:3], dtype=np.float64)
        if not np.allclose(here, coords[vi], atol=1e-6):
            raise ValueError(
                f"vertex {vi} is at {coords[vi]} but node {nid} is at {here}; "
                "the vertex-to-node mapping is wrong, refusing to guess a root"
            )
        out.append(nid)
    return out


def auto(graph) -> list[int]:
    """One automatic root node per component -- the fallback when nothing was picked."""
    from skeleton_analysis.ordering.pipeline import auto_roots

    from .optimise import as_sa_graph
    from .adapter import vertex_node_ids

    triple = graph.triple if hasattr(graph, "triple") else graph
    node_of_vertex = vertex_node_ids(triple)
    return [int(node_of_vertex[int(v)]) for v in auto_roots(as_sa_graph(graph))
            if 0 <= int(v) < len(node_of_vertex)]


# --------------------------------------------------------------------------- #
# the sidecar
# --------------------------------------------------------------------------- #
def describe_roots(graph, nodes, *, method: str = "manual", parts=None) -> list[dict]:
    """One record per root: where it is, what it roots, and how it was chosen.

    `parts` are the :class:`~.components.TreePart` objects the mask was split into,
    when there were any; they supply the voxel count and bounding box that make a
    later renumbering *visible* rather than silent.
    """
    of_edge = tree_of_edge(graph)
    comps = graph.components()
    component_of: dict[int, int] = {}
    for i, comp in enumerate(comps):
        for sid in comp:
            component_of[sid] = i

    by_index = {int(p.index): p for p in (parts or ())}
    records, seen = [], set()
    for nid in nodes:
        nid = int(nid)
        try:
            sid, ambiguous = root_edge_for_node(graph, nid)
        except ValueError:
            continue
        comp = component_of.get(sid)
        if comp is None or comp in seen:
            continue
        seen.add(comp)

        # The tree this root belongs to: the modal `tree` value over its component,
        # so an edge left unplaced by `assign_trees` cannot decide the answer.
        labels = [of_edge[s] for s in comps[comp] if s in of_edge]
        if labels:
            values, counts = np.unique(np.asarray(labels), return_counts=True)
            index = int(values[int(np.argmax(counts))])
            source = "mask"
        else:
            index = comp
            source = "graph"

        node = graph.nodes[nid]
        record = {
            "index": index,
            "tree_source": source,
            "root": {
                "node_um": [float(node[0]), float(node[1]), float(node[2])],
                "seg_key": segment_key(graph, sid),
                "node_id": nid,
                "seg_id": int(sid),
                "degree": int(graph.degree(nid)),
                "ambiguous": bool(ambiguous),
                "method": method,
            },
        }
        part = by_index.get(index)
        if part is not None:
            record["label"] = int(part.label)
            record["voxels"] = int(part.voxels)
            # The mask's own name for this tree, when it had one. "Left_Tree" survives
            # a renumbering that "tree 0" does not, and it is the only field in here a
            # person can check against Avizo by eye.
            if getattr(part, "material", ""):
                record["material"] = str(part.material)
                record["material_value"] = int(part.material_value)
                record["rank"] = int(part.rank)
        records.append(record)
    return sorted(records, key=lambda r: r["index"])


def describe_source(path, graph) -> dict:
    """One ``sources`` entry: which graph was rooted, and what it looked like."""
    fingerprint = source_fingerprint(path, graph)
    return {"path": str(path) if path is not None else None,
            "sha1": fingerprint.get("sha1"),
            "counts": {k: v for k, v in fingerprint.items() if k != "sha1"}}


def document(graph, trees, *, source=None, mask=None, picker=None, keys=None,
             fingerprint=None, sources=None) -> dict:
    """The picked roots as a plain-JSON record.

    `sources` lists every graph the roots were picked on, for the case where one
    session rooted a left-tree skeleton and a right-tree skeleton in turn. The
    top-level ``source`` still names the first, so a reader written against a
    single-graph sidecar sees exactly what it did before -- and :func:`resolve` never
    consults either, because a root is placed by *coordinate*, not by provenance.
    """
    if fingerprint is None:
        fingerprint = source_fingerprint(source, graph)
    return _plain({
        "schema": SCHEMA,
        "kind": "roots",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source) if source is not None else None,
        "source_sha1": fingerprint.get("sha1"),
        "source_counts": {k: v for k, v in fingerprint.items() if k != "sha1"},
        "sources": [dict(s) for s in (sources or ())],
        "mask": dict(mask or {}),
        "picker": dict(picker or {}),
        "trees": list(trees),
    })


def load(path) -> dict:
    """Read a roots sidecar, and check it is one of ours.

    Strict equality on the schema, like :func:`~.crop.load`: a document written by a
    different version would be read on the strength of fields that mean something
    else, and a root in the wrong place reverses parent and child through a whole
    subtree without ever looking wrong.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, found {schema!r}")
    return document


# --------------------------------------------------------------------------- #
# placing it back onto a graph
# --------------------------------------------------------------------------- #
@dataclass
class ResolvedRoots:
    """A roots sidecar placed onto a specific graph."""

    root_edges: tuple = ()
    root_nodes: tuple = ()
    by_tree: dict = field(default_factory=dict)
    snapped_um: tuple = ()
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        out = f"{len(self.root_edges)} root(s) placed"
        if self.snapped_um:
            out += f", furthest snap {max(self.snapped_um):.1f} um"
        return out + "".join(f"\n  ! {n}" for n in self.notes)


def resolve(graph, document: dict, *, max_snap_um: float | None = None,
            frame=None) -> ResolvedRoots:
    """Place a sidecar's roots onto `graph`, by key first and coordinate second.

    Per tree, in order: the recorded segment key (exact, on the same graph); then the
    nearest node to the recorded position, accepted within `max_snap_um`. A tree that
    resolves to neither gets a note and no root, so the automatic pick applies to it.
    """
    if max_snap_um is None and frame is not None:
        max_snap_um = SNAP_VOXELS * float(np.max(frame.seg_spacing))

    records = list(document.get("trees", ()))
    wanted = [str(r["root"].get("seg_key") or "") for r in records]
    found, _ = resolve_keys(graph, [k for k in wanted if k])

    node_ids = list(graph.nodes)
    coords = np.array([graph.nodes[n][:3] for n in node_ids],
                      dtype=np.float64).reshape(-1, 3)
    tree = None
    if len(coords):
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)

    component_of: dict[int, int] = {}
    for i, comp in enumerate(graph.components()):
        for sid in comp:
            component_of[sid] = i

    edges, nodes, by_tree, snapped, notes = [], [], {}, [], []
    taken: set[int] = set()
    for record in records:
        index = int(record.get("index", -1))
        root = record.get("root", {})
        point = np.asarray(root.get("node_um", (0.0, 0.0, 0.0)), dtype=np.float64)
        key = str(root.get("seg_key") or "")

        nid = None
        if key and key in found:
            nid = node_at(graph, found[key], point)
        elif tree is not None:
            distance, i = tree.query(point)
            if max_snap_um is None or float(distance) <= float(max_snap_um):
                nid = int(node_ids[int(i)])
                snapped.append(float(distance))
            else:
                notes.append(
                    f"tree {index}: nearest node is {float(distance):.1f} um from the "
                    f"recorded root, beyond the {float(max_snap_um):.1f} um limit; "
                    "using the automatic root instead"
                )
        if nid is None:
            if not notes or f"tree {index}:" not in notes[-1]:
                notes.append(f"tree {index}: the recorded root could not be placed; "
                             "using the automatic root instead")
            continue

        try:
            sid, _ambiguous = root_edge_for_node(graph, nid)
        except ValueError:
            notes.append(f"tree {index}: node {nid} has no incident segment any more")
            continue

        comp = component_of.get(sid)
        if comp is not None and comp in taken:
            notes.append(f"tree {index}: its component already has a root; ignored")
            continue
        if comp is not None:
            taken.add(comp)
        edges.append(int(sid))
        nodes.append(int(nid))
        by_tree[index] = int(sid)

    return ResolvedRoots(tuple(edges), tuple(nodes), by_tree, tuple(snapped), notes)


def root_edges_for(graph, args, *, frame=None) -> tuple:
    """The root edges for this run: ``--root-edge`` wins, the sidecar fills the rest.

    Deduplicated **by component before** the ids reach
    :func:`~.radius_perimeter._directed_topology`, which raises when two roots select
    one component. An explicit ``--root-edge`` and a sidecar naming the same tree is a
    normal thing to do -- pinning one tree by hand and leaving the others -- and it
    must not look like a conflict.
    """
    explicit = [int(x) for x in (getattr(args, "root_edge", None) or ())]
    path = getattr(args, "roots_json", None)
    if not path or not Path(path).exists():
        return tuple(explicit)

    component_of: dict[int, int] = {}
    for i, comp in enumerate(graph.components()):
        for sid in comp:
            component_of[sid] = i

    taken = {component_of.get(sid) for sid in explicit}
    resolved = resolve(graph, load(path), frame=frame)
    for note in resolved.notes:
        print(f"  ! {note}")
    out = list(explicit)
    for sid in resolved.root_edges:
        comp = component_of.get(sid)
        if comp in taken:
            continue  # already pinned by hand
        taken.add(comp)
        out.append(int(sid))
    return tuple(out)


def root_nodes_for(graph, args, *, frame=None) -> tuple:
    """The root **nodes** for ordering, from the sidecar; ``()`` when there is none."""
    path = getattr(args, "roots_json", None)
    if not path or not Path(path).exists():
        return ()
    return resolve(graph, load(path), frame=frame).root_nodes
