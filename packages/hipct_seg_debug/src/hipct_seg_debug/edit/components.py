"""Split a mask into its connected components, and put the pieces back together.

Two components in this data are the anatomically distinct **left and right coronary
trees**, not a defect. Everything downstream of skeletonisation already knows that --
:func:`~.radius_perimeter._directed_topology` roots each component separately,
``order_forest`` is a forest algorithm, :mod:`~.reconnect.geodesic.select` builds a
spanning *forest* on purpose -- but skeletonisation itself ran over the whole volume
and produced one graph with no record of which tree an edge belonged to. This module
supplies that record.

The tree index is a property of the **mask labelling**, not of the graph: components
are numbered largest-first from the labelled volume, so ``lee.am``, ``teasar.am`` and
``amira.am`` derived from one mask all agree that ``tree == 0`` is the same anatomy.
That is the whole point of the field -- a per-graph numbering would make the three
incomparable, which is precisely the comparison :mod:`~.supermetric` exists to do.

**Indices are not portable across masks.** They depend on the labelling, so a
different stride, connectivity or ``min_voxels`` can renumber everything. The roots
sidecar records each tree's voxel count, bounding box and root *coordinate* so a
renumbering is visible rather than silent.

An origin, not a sub-frame
--------------------------
Each component is skeletonised inside its own bounding box, so its voxel ``(0, 0, 0)``
is not the volume's. The backends need exactly one thing to place the result back in
world micrometres -- ``frame.seg_origin`` -- so :class:`TreePart` carries an
``origin_um`` override and the frame itself is passed through untouched.

Rebuilding a ``WorldFrame`` per component would be actively worse. ``frame.seg_bbox_um``
is read by :func:`~.skeleton_optimise.prune_spurs` as *the lattice edge*, so a
per-component bbox face would protect free ends nowhere near the real edge;
``frame.seg_dims`` is read by ``supermetric._sample_mask``; and
``LatticeView.from_frame`` multiplies ``seg_spacing`` by the stride a second time.
All three fail silently. :func:`~.reskeletonise.reskeletonise_box` already takes the
origin route for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# The per-edge column name, in one place so the writer, the reader and the CLI agree.
TREE_FIELD = "tree"

#: How many components `split_components` lists individually before summarising. The
#: main bodies come first, so this always shows every tree that matters.
LIST_LIMIT = 12


@dataclass(frozen=True)
class TreePart:
    """One connected component of the mask, cropped to its own bounding box."""

    index: int  # 0-based, largest first -- this IS the tree id
    label: int  # its label in the connected-component labelling
    voxels: int  # foreground voxels, this component only
    origin_zyx: tuple[int, int, int]  # bbox corner on the parent decoded grid
    origin_um: np.ndarray  # (3,) um, what the backends need instead of frame.seg_origin
    volume: np.ndarray  # bool (nz, ny, nx), THIS component only
    #: The ``Materials`` entry this component came out of, when the mask declared one
    #: -- ``"Left_Tree"`` rather than "the second largest thing in the volume". Empty
    #: for a plain binary mask, which is the case connectivity alone has to serve.
    material: str = ""
    material_value: int = 0
    #: This component's rank *within its material*, largest first. 0 is the tree's
    #: main body; everything above it is a detached fragment of the same artery.
    rank: int = 0

    @property
    def is_main_body(self) -> bool:
        return self.rank == 0

    def describe(self) -> str:
        z, y, x = self.volume.shape
        who = f"tree {self.index}"
        if self.material:
            who += f" [{self.material}"
            who += "]" if self.is_main_body else f" fragment {self.rank}]"
        return (f"{who}: {self.voxels:,} voxels in a {x}x{y}x{z} box "
                f"at {tuple(int(v) for v in self.origin_zyx)}")


def regions_of(materials) -> tuple:
    """The foreground materials of `materials`, which may be a ``LatticeInfo``.

    ``Exterior`` is dropped by value, not by name: it is the material a label lattice
    stores as 0, whatever it happens to be called.
    """
    if materials is None:
        return ()
    items = getattr(materials, "materials", materials)
    return tuple(m for m in items if int(getattr(m, "value", 0)) != 0)


def _split_straddling(volume, stats, labels_of_interest, boxes, connectivity: int):
    """Re-label the listed components per material, in place, inside their own boxes.

    Only components that actually span two materials are touched. Each is re-labelled
    over ``volume == value`` restricted to that component and its bounding box, so the
    work is proportional to the straddle rather than to the volume.

    Returns ``{label: (voxels, material value)}`` for every label it touched -- the
    material is recorded here, where it is already known, rather than looked up
    afterwards: finding one voxel of a label by scanning the array costs a pass over
    the whole volume, and there is one such label per straddle.

    The component keeps its original label for the first material found in it, so
    nothing that already referred to it has to be rewritten.
    """
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, connectivity)
    touched: dict[int, tuple] = {}
    next_label = int(stats.n)
    for label in labels_of_interest:
        box = boxes[label - 1]
        if box is None:  # pragma: no cover - a label with no voxels cannot straddle
            continue
        window = stats.labels[box]  # a view: writing to it writes to `stats.labels`
        mine = window == label
        first = True
        for value in (int(v) for v in np.unique(volume[box][mine])):
            piece, n = ndimage.label((volume[box] == value) & mine, structure=structure)
            for k in range(1, n + 1):
                where = piece == k
                if first:
                    new, first = int(label), False
                else:
                    next_label += 1
                    new = next_label
                    window[where] = new
                touched[new] = (int(where.sum()), value)
    stats.n = next_label
    return touched


def _grouped_labels(volume, regions, connectivity: int):
    """``(stats, groups)`` -- components, split by material first and size second.

    `groups` is one ``(material, [(label, voxels), ...])`` entry per material in file
    order, each list largest first. `stats.labels` holds **every** component of every
    material under a globally unique label, because that single array is what
    :func:`assign_trees` rasterises a graph against.

    **One labelling, not one per material.** The obvious implementation -- label
    ``volume == value`` once per material -- walks the whole volume N times, and at
    stride 1 on a 47.8-billion-voxel lattice one 26-connected pass is already the
    expensive part of the command. So the foreground is labelled once, and each
    component is asked which material values it contains: almost every component lies
    in exactly one, and only the few that genuinely span two are re-labelled, inside
    their own bounding boxes.

    That is the case materials exist for -- two coronaries touching are a single
    26-connected component in ``volume > 0``, which connectivity alone cannot separate
    -- and it is rare enough to be worth paying for only where it happens.
    """
    from scipy import ndimage

    from .reconnect.segmentation import components as label_components

    volume = np.asarray(volume)
    stats = label_components(volume > 0, connectivity=connectivity)
    if not regions or stats.n == 0:
        order = [int(v) for v in stats.order_by_size()]
        return stats, [(None, [(lab, int(stats.sizes[lab])) for lab in order])]

    index = np.arange(1, stats.n + 1)
    lo = np.asarray(ndimage.minimum(volume, stats.labels, index=index))
    hi = np.asarray(ndimage.maximum(volume, stats.labels, index=index))
    material_of = {int(lab): int(v) for lab, v in zip(index, lo)}
    sizes = {int(lab): int(stats.sizes[lab]) for lab in index}

    straddling = [int(lab) for lab, a, b in zip(index, lo, hi) if a != b]
    if straddling:
        touched = _split_straddling(volume, stats, straddling,
                                    ndimage.find_objects(stats.labels), connectivity)
        for label, (n, value) in touched.items():
            sizes[label] = n
            material_of[label] = value

    stats.sizes = np.zeros(int(stats.n) + 1, dtype=np.int64)
    for label, n in sizes.items():
        stats.sizes[label] = n

    groups = []
    for material in regions:
        mine = [(lab, n) for lab, n in sizes.items()
                if material_of.get(lab) == int(material.value)]
        mine.sort(key=lambda e: (-e[1], e[0]))
        groups.append((material, mine))
    return stats, groups


def split_components(volume, frame, *, min_voxels: int = 0, max_trees: int | None = None,
                     connectivity: int = 3, pad: int = 1, verbose: bool = False,
                     materials=None):
    """``(parts, stats)`` -- one :class:`TreePart` per component worth skeletonising.

    `min_voxels` drops debris; `max_trees` keeps only the N largest. What is dropped is
    always reported, because a silently discarded component is a missing coronary tree.

    `pad` grows each bounding box by that many voxels, clipped to the volume, so the
    component is surrounded by background. Lee thinning curls toward a cut face, so a
    box drawn tight against the vessel would bend the centreline at both ends -- the
    same artefact :func:`~.reskeletonise.reskeletonise_box` pads against.

    Materials first, connectivity second
    ------------------------------------
    `materials` is a mask's ``Materials`` block -- pass the ``LatticeInfo`` itself, or
    ``()`` to ignore it. When it names more than the Exterior, the split runs **per
    material and then by connectivity inside each**, so ``Left_Tree``'s main body and
    its detached fragments stay distinguishable and neither can be confused with the
    right coronary.

    That ordering matters because connectivity alone answers a different question. On
    ``32.04um_artery_left_right.labels.Regions.am`` the binarised mask breaks into
    hundreds of pieces, so "tree 0" and "tree 1" are the two largest *fragments* --
    frequently both from the same artery -- while the file has said which is which all
    along. With materials, `max_trees` caps the parts kept **per material**, since its
    job there is "the main body of each tree", and a global cap could drop a whole
    coronary.
    """
    import time

    from scipy import ndimage

    regions = regions_of(materials)
    if verbose:
        # Said *before* the pass, not after. A 26-connected labelling of the full
        # LADAF-2021-17 lattice is 47.8 G voxels in one single-threaded union-find,
        # and a command that prints nothing for that long looks hung rather than busy.
        voxels = int(np.asarray(volume).size)
        print(f"  labelling {voxels / 1e9:.2f} G voxels, 26-connected"
              + (f", against {len(regions)} material(s)" if regions else "")
              + "..." + ("  (minutes at stride 1 -- use --stride for a trial run)"
                         if voxels > 4e9 else ""))
    started = time.time()
    stats, groups = _grouped_labels(volume, regions, connectivity)
    if verbose:
        print(f"    labelled in {time.time() - started:.1f}s")
    if stats.n == 0:
        if verbose:
            print("  no foreground: nothing to skeletonise")
        return [], stats

    # (material, rank within it, global label, voxels)
    kept: list[tuple] = []
    small: list[tuple] = []
    capped: list[tuple] = []
    for order, (material, entries) in enumerate(groups):
        big_enough = [e for e in entries if e[1] >= int(min_voxels)]
        small.extend((material, *e) for e in entries if e[1] < int(min_voxels))
        take = big_enough[: int(max_trees)] if max_trees is not None else big_enough
        capped.extend((material, *e) for e in big_enough[len(take):])
        kept.extend((order, material, rank, lab, n)
                    for rank, (lab, n) in enumerate(take))

    # Main bodies first, in material order, and only then the fragments. So on a mask
    # naming two coronaries, tree 0 is Left_Tree and tree 1 is Right_Tree -- the
    # numbering every `--max-trees 2` and `--scope largest` already assumes. Ordering
    # by material *then* rank would bury the right coronary behind however many
    # detached specks the left one has, and at stride 8 that is 960 of them.
    kept.sort(key=lambda k: (k[2], k[0]))
    kept = [(material, rank, lab, n) for _order, material, rank, lab, n in kept]

    if verbose:
        if regions:
            named = ", ".join(f"{m.name}={m.value}" for m in regions)
            print(f"  materials: {named}")
            for material, entries in groups:
                mine = [k for k in kept if k[0] is material]
                print(f"    {material.name}: {len(entries)} component(s), "
                      f"keeping {len(mine)}")
        print(f"  {stats.n} component(s); keeping {len(kept)}")
        # The two reasons are reported apart. A component excluded by `max_trees` can
        # be far larger than the threshold -- with `--scope largest` the whole right
        # coronary tree is -- and calling that "below N voxels" would misread as the
        # threshold having eaten an artery.
        if small:
            lost = sum(int(n) for _m, _lab, n in small)
            big = ", ".join(f"{int(n):,}" for _m, _lab, n in small[:5])
            print(f"    dropped {len(small)} below {min_voxels:,} voxels "
                  f"({lost:,} voxels total; largest: {big})")
        if capped:
            sizes = ", ".join(f"{int(n):,}" for _m, _lab, n in capped[:5])
            per = " per material" if regions else ""
            print(f"    set aside {len(capped)} component(s) beyond the "
                  f"{max_trees} kept{per}, of {sizes} voxels")

    # One pass for every bounding box, rather than an argwhere per label.
    boxes = ndimage.find_objects(stats.labels)
    shape = stats.labels.shape
    parts: list[TreePart] = []
    for index, (material, rank, label, _voxels) in enumerate(kept):
        box = boxes[label - 1]
        if box is None:  # pragma: no cover - a label with no voxels cannot be kept
            continue
        lo, hi = [], []
        for axis, sl in enumerate(box):
            lo.append(max(int(sl.start) - int(pad), 0))
            hi.append(min(int(sl.stop) + int(pad), int(shape[axis])))

        # Crop by LABEL, never by threshold. `volume[box] > 0` would admit any other
        # component that happens to pass through this bounding box, and that component
        # would then be skeletonised twice -- once in its own box and once here --
        # duplicating its centreline in the merged graph.
        sub = stats.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] == label
        got = int(sub.sum())
        want = int(stats.sizes[label])
        if got != want:  # pragma: no cover - a labelling/bbox disagreement
            raise AssertionError(
                f"component {label} has {want} voxels but its bounding box holds {got}"
            )

        origin_zyx = (lo[0], lo[1], lo[2])
        # (k, j, i) -> (i, j, k). No `* stride`: `lo` indexes the already-decoded grid,
        # whose pitch is frame.seg_spacing, which already carries the stride. Applying
        # it again is the bug that put every coordinate eight times too far out.
        origin_um = frame.seg_to_um([[lo[2], lo[1], lo[0]]])[0]
        part = TreePart(index=index, label=label, voxels=want, origin_zyx=origin_zyx,
                        origin_um=np.asarray(origin_um, dtype=np.float64), volume=sub,
                        material="" if material is None else str(material.name),
                        material_value=0 if material is None else int(material.value),
                        rank=int(rank))
        # Listed, but not all 1,614 of them: at `min_voxels` 0 a real coronary mask
        # splits into hundreds of single-voxel specks, and printing one line each
        # buries the two trees the operator is actually looking for.
        if verbose and index < LIST_LIMIT:
            print(f"    {part.describe()}")
        parts.append(part)
    if verbose and len(parts) > LIST_LIMIT:
        rest = parts[LIST_LIMIT:]
        print(f"    ... and {len(rest)} more, {sum(p.voxels for p in rest):,} voxels "
              f"in total (raise --min-component-voxels to drop them)")
    return parts, stats


def merge_triples(parts: Sequence[tuple[int, object]], *, tree_field: str = TREE_FIELD,
                  keep_subtrees: bool = False):
    """Concatenate per-component triples, renumbering ids and tagging every edge.

    Ids are renumbered contiguously from 0 in emission order rather than offset past
    the previous part's maximum. ``EditableGraph`` keys its segment index on
    ``seg["id"]``, so two parts sharing an id would make one of them silently
    disappear; renumbering from scratch also matches what ``to_spatial_graph``
    produces anyway.

    With `keep_subtrees` the int in each pair is a **base offset** rather than the tree
    index outright, and a part that already carries a tree field keeps its own
    numbering shifted past it. That is what stops :func:`merge_graphs` flattening a
    per-tree skeletonisation to one tree when it is loaded beside another graph.
    """
    from .adapter import Triple

    nodes: dict[int, tuple] = {}
    points: dict[int, tuple] = {}
    segments: list[dict] = []
    dtypes: dict[str, np.dtype] = {}
    strahler_field = None
    next_node = next_point = next_seg = 0

    for tree_index, triple in parts:
        if triple is None or not triple.segments:
            continue
        node_map: dict[int, int] = {}
        point_map: dict[int, int] = {}

        for nid, value in triple.nodes.items():
            node_map[nid] = next_node
            nodes[next_node] = value
            next_node += 1
        for pid, value in triple.points.items():
            point_map[pid] = next_point
            points[next_point] = value
            next_point += 1

        for seg in triple.segments:
            out = dict(seg)
            out["id"] = next_seg
            out["node1"] = node_map[seg["node1"]]
            out["node2"] = node_map[seg["node2"]]
            out["point_ids"] = [point_map[p] for p in seg["point_ids"]]
            out[tree_field] = int(tree_index) + (
                int(seg.get(tree_field, 0)) if keep_subtrees else 0
            )
            segments.append(out)
            next_seg += 1

        for name, dtype in triple.edge_attr_dtypes.items():
            dtypes.setdefault(name, dtype)
        if strahler_field is None:
            strahler_field = triple.strahler_field

    # Nodes left with no incident segment would be dropped by `to_spatial_graph`
    # anyway; keeping them here costs nothing and keeps the maps honest.
    dtypes[tree_field] = np.dtype(np.int64)
    return Triple(nodes=nodes, points=points, segments=segments,
                  edge_attr_dtypes=dtypes, strahler_field=strahler_field)


def graph_paths(value) -> list:
    """A skeleton argument as a list of paths, however it arrived.

    Variadic on both command lines, ``;``-separated in the GUI's Data tab, and a bare
    string in every saved session and script written before either. All three mean the
    same thing, and normalising in one place is what keeps the rest from caring.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = str(value).split(";")
    return [p.strip() for p in items if p.strip()]


def merge_graphs(triples, *, tree_field: str = TREE_FIELD):
    """Several skeletons as **one** graph, each source kept as its own tree.

    ``(triple, trees)``, where `trees` is the list of tree indices each source ended up
    holding. One graph rather than several is what makes the extra skeletons *editable*
    rather than mere backdrop: picking, the edit tools, `crop`, `reformat` and the
    writer are all defined against a single graph, and a second graph beside it would
    have to be read-only to stay honest. Merged, every one of those works on all of
    them, and the ``tree`` field is what keeps them apart -- which is the same field
    `--per-tree` skeletonisation, per-tree scoring and the roots sidecar already use.

    A source that already carries tree indices keeps them, shifted past everything
    before it, so loading a per-tree skeleton beside a single-tree one gives three
    trees rather than two.
    """
    parts, trees, offset = [], [], 0
    for triple in triples:
        existing = tree_indices(triple, tree_field=tree_field)
        mine = [offset + int(t) for t in existing] if existing else [offset]
        parts.append((offset, triple))
        trees.append(mine)
        offset = max(mine) + 1
    merged = merge_triples(parts, tree_field=tree_field, keep_subtrees=True)
    return merged, trees


def tree_of_edge(graph, *, tree_field: str = TREE_FIELD) -> dict[int, int]:
    """``{segment id: tree index}``, or ``{}`` when the graph carries no tree field."""
    out = {}
    for seg in graph.segments:
        if tree_field in seg:
            out[int(seg["id"])] = int(seg[tree_field])
    return out


def tree_indices(graph, *, tree_field: str = TREE_FIELD) -> list[int]:
    """Sorted tree indices present on the graph; ``[]`` when it carries none."""
    return sorted({v for v in tree_of_edge(graph, tree_field=tree_field).values()})


def subgraph_by_tree(graph, index: int, *, tree_field: str = TREE_FIELD):
    """A new :class:`~.graphmodel.EditableGraph` holding just one tree.

    Segment, node and point ids are preserved, so anything resolved against the whole
    graph -- a root edge, a segment key -- still names the same thing here.
    """
    from .adapter import Triple
    from .graphmodel import EditableGraph

    keep = [s for s in graph.segments if int(s.get(tree_field, -1)) == int(index)]
    kept_nodes = {s[k] for s in keep for k in ("node1", "node2")}
    kept_points = {p for s in keep for p in s["point_ids"]}
    triple = Triple(
        nodes={n: v for n, v in graph.nodes.items() if n in kept_nodes},
        points={p: v for p, v in graph.points.items() if p in kept_points},
        segments=[dict(s) for s in keep],
        edge_attr_dtypes=dict(graph.triple.edge_attr_dtypes),
        strahler_field=graph.triple.strahler_field,
    )
    return EditableGraph(triple)


def root_edge_for_node(graph, nid: int) -> tuple[int, bool]:
    """``(segment id, ambiguous)`` -- the edge that roots `nid`'s component.

    ``--root-edge`` names a *segment*, because that is what ``_directed_topology``
    intersects with each component, but the picker returns a *node*. For the normal
    case the translation is exact: ``root_from_edge`` deliberately returns the
    lower-coordination endpoint of the clicked edge, so the root is a free end with
    exactly one incident segment.

    A root on a junction is genuinely ambiguous -- several edges could carry the
    depth-0 slot -- so the thickest wins and the caller is told, because the sidecar's
    recorded coordinate is then the only thing that pins the direction.
    """
    incident = sorted(graph.node_segments(int(nid)))
    if not incident:
        raise ValueError(f"node {nid} has no incident segment, so it cannot root a tree")
    if len(incident) == 1:
        return int(incident[0]), False

    def calibre(sid: int) -> float:
        seg = graph.segment(sid)
        if "MeanRadius" in seg:
            return float(seg["MeanRadius"])
        r = graph.radii(sid)
        return float(np.median(r)) if len(r) else 0.0

    return int(max(incident, key=calibre)), True


def assign_trees(graph, labels_zyx, frame, *, order: Sequence[int],
                 verbose: bool = False) -> np.ndarray:
    """Per-edge tree index for a graph produced without a mask split.

    The ``amira`` candidate is an existing export -- there is no per-component run to
    tag its edges during -- so its edges are assigned by rasterising each one and
    asking the mask labelling which component it lies in. An edge that lands on no
    label at all (a centreline drifting outside the lumen) inherits its graph
    component's majority vote, and a graph component with no vote anywhere is left at
    ``-1`` rather than being guessed into a tree it may not belong to.

    Returns one index per segment, in ``graph.segments`` order.
    """
    from .supermetric import rasterise_segment

    tree_of_label = {int(lab): i for i, lab in enumerate(order)}
    shape = np.asarray(labels_zyx.shape, dtype=np.int64)
    ids = [int(s["id"]) for s in graph.segments]
    votes: dict[int, int] = {}

    for sid in ids:
        zyx = rasterise_segment(graph, sid, frame)
        if len(zyx) == 0:
            continue
        inb = np.all((zyx >= 0) & (zyx < shape[None, :]), axis=1)
        if not inb.any():
            continue
        hit = labels_zyx[zyx[inb, 0], zyx[inb, 1], zyx[inb, 2]]
        hit = hit[hit > 0]
        if len(hit) == 0:
            continue
        labs, counts = np.unique(hit, return_counts=True)
        winner = tree_of_label.get(int(labs[int(np.argmax(counts))]))
        if winner is not None:
            votes[sid] = winner

    # A component votes as a block: an edge that missed the mask still belongs to
    # whatever tree the rest of its component landed in.
    out = {sid: votes.get(sid, -1) for sid in ids}
    unplaced = 0
    for comp in graph.components():
        seen = [votes[sid] for sid in comp if sid in votes]
        if not seen:
            unplaced += len(comp)
            continue
        labs, counts = np.unique(np.asarray(seen), return_counts=True)
        majority = int(labs[int(np.argmax(counts))])
        for sid in comp:
            out[sid] = majority

    if unplaced and verbose:
        print(f"    {unplaced} segment(s) in components that touch no kept mask "
              f"component; left at tree -1")
    return np.array([out[sid] for sid in ids], dtype=np.int64)
