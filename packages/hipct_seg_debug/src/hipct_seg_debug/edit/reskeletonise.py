"""Turn a corrected patch of mask back into centreline, and stitch it in.

This is the step that makes painting worth doing. ``coronary_sdf`` never reads voxel
data -- the lumen surface is a function of the skeleton graph alone -- so a mask edit
reaches the surface only by becoming graph. Locally re-deriving the centreline is
that conversion.

Two modes, and the default is the cautious one for a measured reason. Scoring the
whole-volume skeletonisation against Avizo's graph gave 96% recall on bifurcations
and **383 false ones**; re-deriving a box wholesale would import that noise into a
graph there is good reason to trust. So:

``add``      keep only the chain running through voxels the user actually painted,
             grown far enough at each end to reach existing centreline, and weld it
             on. Nothing already in the graph is touched.
``replace``  delete every segment lying wholly inside the box and insert the whole
             locally-derived skeleton. Truest to "the mask is ground truth", and
             available when that is what you want.

Three details decide whether the result is usable:

* **Pad the box, then throw the pad away.** The medial axis of a *cropped* tube
  curls toward the cut face, so a skeleton traced right up to the box boundary
  bends in a way the anatomy does not. Skeletonising a padded box and keeping only
  the core leaves those artefacts in the discarded rind.
* **Insert only what the graph does not already cover.** The fragment is
  deliberately traced *past* the painted region, so that its ends run into
  territory the graph already describes and have something to weld to. Those
  overlapping points are then trimmed off again, leaving one anchor point at each
  end. Without the trim the splice lays a second centreline alongside the first.
* **Weld by proximity, not by inference.** The reconnectors in :mod:`.reconnect`
  reason about whether two ends *should* join across a gap. Here they already
  coincide, so this is a snap -- preferring an existing node over splitting a
  segment, and reporting an end with nothing in reach rather than guessing.

Short false branches are dropped by ``skeleton_to_graph``'s own
``min_branch_voxels`` rather than by pruning voxels here. A spur one voxel off a
26-connected line is itself adjacent to three line voxels, so it reads as a
junction, and a voxel-level prune stops one short of removing it. Dropping a branch
does leave its junction behind as a meaningless degree-2 node, so
:func:`contract_degree2` joins the two survivors back into one segment --
``coronary_sdf.merge_degree2`` would do the same before meshing, but not before the
graph is exported or reported on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .history import Patch

# Rind discarded after skeletonising, as a multiple of the local vessel radius.
PAD_FACTOR = 3.0
MIN_PAD_VOX = 10
# A degree-1 branch shorter than this many local radii is a thinning artefact.
PRUNE_FACTOR = 2.0
# How far past the painted region to follow the skeleton, so each end has existing
# centreline to weld to. Trimmed off again before the splice.
GROW_FACTOR = 3.0
# A fragment point this close to an existing one is already described by the graph.
# Much tighter than the weld tolerance on purpose: welding asks "can these be
# joined", this asks "is this the same centreline", and confusing the two trims a
# genuinely new run of vessel away.
COVER_VOXELS = 0.75


@dataclass
class ReskeletoniseReport:
    """What the re-derivation did, in enough detail to undo it knowingly."""

    mode: str
    box_um: np.ndarray
    applied: bool = False
    reason: str = ""
    n_skeleton_voxels: int = 0
    n_kept_voxels: int = 0
    n_trimmed_points: int = 0
    segments_added: int = 0
    segments_deleted: int = 0
    ends_welded: int = 0
    ends_free: int = 0
    length_mm: float = 0.0
    patch: Patch | None = None
    seconds: dict = field(default_factory=dict)

    def describe(self) -> str:
        if not self.applied:
            return f"re-skeletonise ({self.mode}): {self.reason or 'nothing to do'}"
        bits = [
            f"re-skeletonise ({self.mode}): +{self.segments_added} segment(s)",
            f"{self.length_mm:.2f} mm",
            f"{self.ends_welded} end(s) welded",
        ]
        if self.segments_deleted:
            bits.insert(1, f"-{self.segments_deleted} deleted")
        if self.ends_free:
            bits.append(f"{self.ends_free} left free")
        total = self.seconds.get("total")
        if total:
            bits.append(f"{total:.2f}s")
        return ", ".join(bits)


def reskeletonise_box(
    graph,
    source,
    frame,
    box_um,
    *,
    mode: str = "add",
    weld_um: float | None = None,
    prune_factor: float = PRUNE_FACTOR,
    grow_um: float | None = None,
    pad_factor: float = PAD_FACTOR,
    min_pad_vox: int = MIN_PAD_VOX,
    verbose: bool = False,
) -> ReskeletoniseReport:
    """Re-derive the centreline inside `box_um` from the (edited) mask and splice it.

    `graph` is an :class:`~.graphmodel.EditableGraph`, `source` a
    :class:`~.maskedit.MaskSource`, `box_um` a ``(2, 3)`` world box in micrometres.
    """
    from skimage.morphology import skeletonize

    from .skeletonise import edt_radii, skeleton_to_graph

    if mode not in ("add", "replace"):
        raise ValueError(f"mode must be 'add' or 'replace', not {mode!r}")
    t0 = time.time()
    box_um = np.asarray(box_um, dtype=np.float64).reshape(2, 3)
    report = ReskeletoniseReport(mode=mode, box_um=box_um)

    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)  # (x, y, z)
    spacing_zyx = spacing[::-1]
    r_local = _local_radius(graph, box_um, fallback=float(spacing.max()) * 2.0)
    pad = int(max(np.ceil(pad_factor * r_local / spacing.min()), min_pad_vox))
    reach = GROW_FACTOR * r_local if grow_um is None else float(grow_um)

    # Add mode keeps skeleton for `reach` beyond the painted box, so the fragment
    # runs into territory the graph already covers and its ends have something to
    # weld onto. Replace mode must not: what it keeps is exactly what it deleted.
    reach_vox = 0 if mode == "replace" else int(np.ceil(reach / spacing.min()))
    box_lo, box_hi = _box_to_seg(frame, box_um)
    core_lo = box_lo - reach_vox
    core_hi = box_hi + reach_vox
    lo = core_lo - pad
    hi = core_hi + pad + 1  # exclusive
    origin_kji = tuple(int(v) for v in lo)

    t = time.time()
    base = source.window(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], edited=False)
    window = base.copy()
    source.edits.apply(window, origin_kji)
    mask = window > 0
    report.seconds["decode"] = time.time() - t

    if not mask.any():
        report.reason = "the mask is empty in this box"
        report.seconds["total"] = time.time() - t0
        return report

    if mode == "add":
        added = (window > 0) & (base == 0)
        if not added.any():
            report.reason = "nothing was painted in this box (use replace to re-derive it)"
            report.seconds["total"] = time.time() - t0
            return report
    else:
        added = None

    t = time.time()
    skel = np.asarray(skeletonize(mask, method="lee")) > 0
    report.n_skeleton_voxels = int(skel.sum())
    # Discard the rind: everything the cut faces distorted lives in it.
    core = np.zeros_like(skel)
    core[pad:skel.shape[0] - pad, pad:skel.shape[1] - pad, pad:skel.shape[2] - pad] = True
    skel &= core
    report.seconds["skeletonise"] = time.time() - t

    if not skel.any():
        report.reason = "no skeleton survived the box (try a larger region)"
        report.seconds["total"] = time.time() - t0
        return report

    t = time.time()
    if added is not None:
        skel = select_painted(skel, added, spacing_zyx, reach)
    report.n_kept_voxels = int(skel.sum())
    report.seconds["select"] = time.time() - t

    if not skel.any():
        report.reason = "the painted voxels carry no centreline of their own"
        report.seconds["total"] = time.time() - t0
        return report

    t = time.time()
    coords = np.argwhere(skel)
    radii = edt_radii(mask, coords, spacing)
    origin_um = frame.seg_to_um([[lo[2], lo[1], lo[0]]])[0]
    min_branch = int(max(prune_factor * r_local / spacing.min(), 0))
    local = skeleton_to_graph(skel, radii, origin_um, spacing,
                              min_branch_voxels=min_branch).triple
    local = contract_degree2(local, eps_um=float(spacing.max()))
    report.seconds["trace"] = time.time() - t

    if not local.segments:
        report.reason = "the traced fragment had no segment with two points"
        report.seconds["total"] = time.time() - t0
        return report

    if weld_um is None:
        weld_um = max(2.0 * float(spacing.max()), r_local)
    cover_um = COVER_VOXELS * float(spacing.max())

    t = time.time()
    if mode == "add":
        local, trimmed = trim_to_new(local, graph, cover_um)
        report.n_trimmed_points = trimmed
        if not local.segments:
            report.reason = ("the painted region is already described by the graph "
                             "(nothing new to add)")
            report.seconds["total"] = time.time() - t0
            return report

    _splice(graph, local, box_um, mode, float(weld_um), report)
    report.seconds["splice"] = time.time() - t
    report.seconds["total"] = time.time() - t0
    if not report.segments_added and not report.segments_deleted:
        # `batch` only updates `last_patch` when it actually ran a command, so a
        # no-op splice would otherwise report the *previous* edit's box.
        report.patch = None
        report.reason = "every traced chain was too short to become a segment"
        return report
    report.applied = True
    if verbose:
        print(" ", report.describe())
    return report


# ---------------------------------------------------------------- voxel stages


def select_painted(skel: np.ndarray, added: np.ndarray, spacing_zyx,
                   grow_um: float) -> np.ndarray:
    """Keep the skeleton through the painted voxels, grown out to reach the graph.

    Growth is iterated 26-connected dilation restricted to the skeleton, one ring
    per step. That measures reach in chessboard steps rather than Euclidean
    distance, so a diagonal run overshoots by up to sqrt(3) -- deliberately
    tolerated, because reaching further only gives the weld more existing centreline
    to land on, whereas reaching too little leaves an end dangling.
    """
    from scipy import ndimage

    ball = np.ones((3, 3, 3), dtype=bool)
    keep = skel & ndimage.binary_dilation(np.asarray(added, dtype=bool), structure=ball)
    if not keep.any():
        return np.zeros_like(skel)
    steps = int(np.ceil(float(grow_um) / float(np.min(spacing_zyx))))
    for _ in range(max(steps, 0)):
        grown = ndimage.binary_dilation(keep, structure=ball) & skel
        if grown.sum() == keep.sum():
            break
        keep = grown
    return keep


def contract_degree2(local, eps_um: float):
    """Join the two segments meeting at a degree-2 node back into one.

    Dropping a short false branch leaves its junction behind with only two
    surviving chains, so a single vessel arrives as two segments joined by a node
    that means nothing. ``coronary_sdf.merge_degree2`` would fix that before
    meshing, but not before the graph is exported or reported on, so it is done
    here instead.

    The two chains each carry their own copy of the junction voxel, so the
    duplicate is dropped when the two ends coincide to within `eps_um`.
    """
    from .adapter import Triple

    segments = [{**s, "point_ids": list(s["point_ids"])} for s in local.segments]
    nodes = dict(local.nodes)

    merged_any = True
    while merged_any:
        merged_any = False
        incident: dict[int, list[int]] = {}
        for seg in segments:
            incident.setdefault(seg["node1"], []).append(seg["id"])
            incident.setdefault(seg["node2"], []).append(seg["id"])

        for nid, sids in incident.items():
            if len(sids) != 2 or sids[0] == sids[1]:
                continue
            by_id = {s["id"]: s for s in segments}
            a, b = by_id[sids[0]], by_id[sids[1]]
            ids_a, far_a = _oriented(a, nid, tail=True)
            ids_b, far_b = _oriented(b, nid, tail=False)
            if far_a == far_b:
                continue  # merging would make a self-loop; leave the node in place
            join = np.linalg.norm(
                np.asarray(local.points[ids_a[-1]][:3])
                - np.asarray(local.points[ids_b[0]][:3])
            )
            a["node1"], a["node2"] = far_a, far_b
            a["point_ids"] = ids_a + (ids_b[1:] if join <= eps_um else ids_b)
            segments = [s for s in segments if s["id"] != b["id"]]
            nodes.pop(nid, None)
            merged_any = True
            break

    used = {p for s in segments for p in s["point_ids"]}
    return Triple(nodes=nodes,
                  points={p: v for p, v in local.points.items() if p in used},
                  segments=segments)


def _oriented(seg, nid: int, *, tail: bool) -> tuple[list[int], int]:
    """`seg`'s points ordered so that `nid` is at the end (`tail`) or start."""
    ids = list(seg["point_ids"])
    at_end = seg["node2"] == nid
    if tail != at_end:
        ids.reverse()
    return ids, (seg["node1"] if at_end else seg["node2"])


def trim_to_new(local, graph, cover_um: float):
    """Cut each traced segment back to the part the graph does not already cover.

    The fragment was grown past the painted region on purpose, so its ends overlap
    existing centreline. Splicing that in unmodified would lay a second, parallel
    centreline over a stretch of vessel that is already described -- which then
    meshes as a lumpy double tube and, worse, welds by *splitting* the existing
    segment somewhere in its middle rather than joining at its tip.

    So each segment keeps the run of points that are farther than `cover_um` from
    any existing centreline, plus **one point either side** as the weld anchor.
    Returns ``(trimmed_triple, n_points_removed)``; a segment with nothing new left
    is dropped entirely.
    """
    from scipy.spatial import cKDTree

    existing = [graph.points[pid][:3] for pid in graph.points]
    if not existing:
        return local, 0

    tree = cKDTree(np.asarray(existing, dtype=np.float64))
    kept: list[dict] = []
    points: dict[int, tuple] = {}
    nodes: dict[int, tuple] = {}
    removed = 0

    for seg in local.segments:
        pids = seg["point_ids"]
        xyz = np.array([local.points[p][:3] for p in pids], dtype=np.float64)
        dist, _ = tree.query(xyz)
        new = dist > cover_um
        if not new.any():
            removed += len(pids)
            continue
        a = max(int(np.argmax(new)) - 1, 0)
        b = min(len(pids) - 1 - int(np.argmax(new[::-1])) + 1, len(pids) - 1)
        if b - a < 1:
            removed += len(pids)
            continue
        removed += len(pids) - (b - a + 1)
        span = pids[a:b + 1]
        points.update({p: local.points[p] for p in span})
        # The nodes move to wherever the trimmed ends now are; an untrimmed end
        # keeps its original node so a genuine junction is preserved.
        node1 = seg["node1"] if a == 0 else _fresh_node(nodes, local, points[span[0]])
        node2 = seg["node2"] if b == len(pids) - 1 else _fresh_node(nodes, local,
                                                                    points[span[-1]])
        if a == 0:
            nodes[node1] = local.nodes[node1]
        if b == len(pids) - 1:
            nodes[node2] = local.nodes[node2]
        kept.append({**seg, "node1": node1, "node2": node2, "point_ids": list(span)})

    from .adapter import Triple

    return Triple(nodes=nodes, points=points, segments=kept), removed


def _fresh_node(nodes: dict, local, point) -> int:
    nid = max([*nodes, *local.nodes], default=-1) + 1
    nodes[nid] = (float(point[0]), float(point[1]), float(point[2]), 0)
    return nid


# --------------------------------------------------------------------- splice


def _splice(graph, local, box_um, mode: str, weld_um: float,
            report: ReskeletoniseReport) -> None:
    """Insert `local` into `graph` as one undo step, welding its free ends on."""
    with graph.batch("re-skeletonise painted region"):
        if mode == "replace":
            report.segments_deleted = clear_box(graph, box_um)

        node_map: dict[int, int] = {}
        new_sids: list[int] = []
        for lseg in local.segments:
            pids = lseg["point_ids"]
            if len(pids) < 2:
                continue
            coords = np.array([local.points[p][:3] for p in pids], dtype=np.float64)
            radii = np.array([local.points[p][3] for p in pids], dtype=np.float64)
            for end in (lseg["node1"], lseg["node2"]):
                if end not in node_map:
                    node_map[end] = graph.add_node(local.nodes[end][:3])
            new_sids.append(
                graph.add_segment(node_map[lseg["node1"]], node_map[lseg["node2"]],
                                  coords, radii)
            )
            report.length_mm += float(
                np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()
            ) / 1000.0
        report.segments_added = len(new_sids)

        welded, free = weld_free_ends(graph, set(node_map.values()), new_sids, weld_um)
        report.ends_welded = welded
        report.ends_free = free

    report.patch = graph.last_patch


def clear_box(graph, box_um) -> int:
    """Remove every centreline inside `box_um`, splitting segments that straddle it.

    Deleting only the segments lying *wholly* inside would be simpler and wrong: a
    vessel that enters the box and leaves again would keep its old centreline
    there, and the freshly-derived one would be laid on top of it. So a straddling
    segment is cut at each boundary crossing and the inside pieces are deleted,
    which also leaves a node exactly at the face for the new fragment to weld to.

    A segment whose only inside point is one of its own endpoints is left alone --
    there is nothing to cut off but a single point, and its node is what the weld
    wants anyway.
    """
    deleted = 0
    queue = [seg["id"] for seg in graph.segments]
    while queue:
        sid = queue.pop()
        if not graph.has_segment(sid):
            continue
        pts = graph.coords(sid)
        if not len(pts):
            continue
        inside = _inside(pts, box_um)
        if not inside.any():
            continue
        if inside.all():
            graph.delete_segment(sid)
            deleted += 1
            continue
        crossings = np.flatnonzero(inside[:-1] != inside[1:]) + 1
        cuts = [int(c) for c in crossings if 0 < c < len(pts) - 1]
        if not cuts:
            continue
        _nid, first, second = graph.split_segment(sid, cuts[0])
        queue.extend([first, second])
    return deleted


def weld_free_ends(graph, new_nodes: set[int], new_sids: list[int],
                   weld_um: float) -> tuple[int, int]:
    """Snap each free end of the new fragment onto the nearest existing centreline.

    Returns ``(welded, left_free)``. An end with nothing inside `weld_um` is left
    alone: a fragment that genuinely reaches nothing is a real finding about the
    mask, and inventing an attachment would hide it.
    """
    from scipy.spatial import cKDTree

    mine = {pid for sid in new_sids if graph.has_segment(sid)
            for pid in graph.segment(sid)["point_ids"]}
    others = [pid for pid in graph.points if pid not in mine]
    if not others:
        return 0, sum(1 for n in new_nodes if graph.degree(n) == 1)

    pos = np.array([graph.points[pid][:3] for pid in others], dtype=np.float64)
    tree = cKDTree(pos)

    welded = free = 0
    for nid in sorted(new_nodes):
        if nid not in graph.nodes or graph.degree(nid) != 1:
            continue
        here = np.asarray(graph.nodes[nid][:3], dtype=np.float64)
        reachable = tree.query_ball_point(here, weld_um)
        if not reachable:
            free += 1
            continue
        anchor = _best_anchor(graph, [others[i] for i in reachable], here)
        if anchor is None or anchor == nid:
            free += 1
            continue
        graph.merge_nodes(anchor, nid)
        welded += 1
    return welded, free


def _best_anchor(graph, pids, here) -> int | None:
    """The node to weld onto, preferring an existing one over splitting a segment.

    Landing on the *tip* of a vessel and landing on its *side* look almost the same
    to a nearest-point query, and the right response differs: the tip already has a
    node, while the side needs one cut into it. Taking the plain nearest point would
    split a segment one point short of its own end and leave a one-point stub, so
    every existing node inside the tolerance wins over every interior point.
    """
    nodes, interior = [], []
    for pid in pids:
        sid = graph.segment_of_point().get(pid)
        if sid is None or not graph.has_segment(sid):
            continue
        seg = graph.segment(sid)
        ids = seg["point_ids"]
        index = ids.index(pid)
        if index == 0:
            nodes.append(seg["node1"])
        elif index == len(ids) - 1:
            nodes.append(seg["node2"])
        else:
            interior.append((sid, index, pid))

    if nodes:
        return min(nodes, key=lambda n: float(
            np.linalg.norm(np.asarray(graph.nodes[n][:3]) - here)))
    if not interior:
        return None
    sid, index, _pid = min(interior, key=lambda t: float(
        np.linalg.norm(np.asarray(graph.points[t[2]][:3]) - here)))
    nid, _a, _b = graph.split_segment(sid, index)
    return nid


def _inside(points: np.ndarray, box_um: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.all((pts >= box_um[0]) & (pts <= box_um[1]), axis=1)


def _box_to_seg(frame, box_um) -> tuple[np.ndarray, np.ndarray]:
    """World box (um) -> inclusive segmentation index bounds as ``(k, row, col)``."""
    corners = np.array(box_um, dtype=np.float64).reshape(2, 3)
    ijk = frame.um_to_seg_index(corners)  # (2, 3) as (i, j, k)
    lo_ijk = np.minimum(ijk[0], ijk[1])
    hi_ijk = np.maximum(ijk[0], ijk[1])
    return lo_ijk[::-1].astype(np.int64), hi_ijk[::-1].astype(np.int64)


def _local_radius(graph, box_um, fallback: float) -> float:
    """Median radius of the graph's own points in the box -- the scale everything
    else here is expressed in, so that one number sets the pad, the prune length and
    the weld tolerance together."""
    radii = []
    for seg in graph.segments:
        pts = graph.coords(seg["id"])
        hit = _inside(pts, box_um)
        if hit.any():
            radii.extend(graph.radii(seg["id"])[hit].tolist())
    if not radii:
        return float(fallback)
    return float(np.median(radii))
