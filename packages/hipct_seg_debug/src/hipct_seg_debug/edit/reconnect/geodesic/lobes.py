"""Free ends the mask has and the centreline does not.

Every proposer in this package starts from a **graph** endpoint, and that is a
premise rather than a fact. Pruning, a thinning artefact, a branch too short to
survive ``min_branch_voxels`` -- each leaves lumen in the segmentation that no
centreline reaches, and a break whose far side is one of those is invisible: no
endpoint exists there to pair with, so no candidate is ever generated and the
repair is not so much rejected as never considered.

The failure this causes is not a missed repair. It is a *wrong* one. Where the
mask has four free ends and the skeleton has two, the two that exist are the only
pair the proposer can see, so it joins them -- across the junction, through the
wall, into whichever vessel happened to be nearest -- and every gate downstream is
asked to adjudicate a pair that was mis-chosen before it arrived. Refusing that
pair (which the tortuosity and redundancy gates now do) leaves the break
unrepaired. Only supplying the missing end can repair it.

So this module manufactures the missing ends. Around each free end it takes the
local mask, skeletonises it, and keeps the tips of *that* skeleton which lie in
lumen the graph does not describe:

* **described** means "within :data:`DESCRIBE_RADII` local radii of an existing
  centreline point". The scale is the vessel's own radius because "the graph
  already covers this" means "inside the tube that centreline stands for", and
  that tube is 120 um across on a twig and 1.5 mm on the trunk.
* the medial axis is taken of the **whole local mask**, not of the undescribed
  part alone. Skeletonising a cropped tube curls the axis toward the cut face, and
  the cut face is exactly where the tangent would then be measured -- the one
  number the whole search is steered by.
* a tip lying on a window face is discarded. That is where the box ended, not
  where the vessel did, and it would propose a continuation into material nobody
  looked at.

Each surviving tip carries what a graph endpoint carries -- a position, an outward
tangent and a radius -- plus two things it does not: which mask component it sits
on, and which graph component its lobe hangs off. The second is what keeps the
global forest pass honest. A lobe attached to vessel V is not free-floating;
connecting to it joins V's component just as firmly as connecting to V would, and
a cycle test that could not see this would close the loops it exists to prevent.

**These are candidates, not centreline.** Nothing here writes to the graph. A tip
becomes real only if a route reaches it and the commit re-derives that region from
the mask -- at which point the lobe is skeletonised properly, in place, by
:mod:`...reskeletonise`, and welded like any other locally-traced fragment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A voxel is "described" when a centreline point lies within this many local radii
#: of it. Measured on LADAF-2024-28: at 1.5 the mask carries 354 undescribed lobes
#: against 161 skeleton free ends.
DESCRIBE_RADII = 1.5
#: How far around a free end to look, in local radii. Matches the reach the fragment
#: search already uses, so the two see the same neighbourhood.
SEARCH_RADII = 8.0
#: An undescribed region smaller than this is thinning noise, not a vessel.
MIN_LOBE_VOXELS = 20
#: ...and one less *anisotropic* than this is a blob -- a calcification, a speck of
#: thresholded myocardium -- however close to the gap it sits. Longest principal
#: axis over shortest, not over next-longest: a collapsed vessel is a ribbon, and a
#: short piece of ribbon is barely longer than it is wide while being many times
#: wider than it is thick. Judging it by the classical elongation ratio would throw
#: away precisely the morphology this package exists for.
MIN_LOBE_ELONGATION = 1.6
#: How far back along the traced skeleton the outward tangent is measured, in local
#: radii. Two: one is inside the tip's own noise, four reaches round a bend.
TANGENT_RADII = 2.0
#: How far the walk inward may go looking for described lumen to attach to.
ATTACH_RADII = 12.0
#: Two tips this close together, in radii, are one end found twice.
DUPLICATE_RADII = 1.5
#: Hard ceiling on one search window. A free end on the aorta would otherwise ask
#: for a box the size of the specimen.
MAX_WINDOW_VOXELS = 8_000_000
#: Minimum half-width of a window, in voxels, so a sub-voxel twig still gets a look.
MIN_WINDOW_VOXELS = 12

_NEIGHBOURHOOD = np.ones((3, 3, 3), dtype=np.uint8)


@dataclass
class MaskEnd:
    """A free end of the segmentation that carries no centreline.

    `key` is the tip's global voxel index and is the identity: it is stable across
    runs on the same mask, which is what lets an operator's ruling in a decisions
    file still mean something after the graph has moved on.
    """

    key: tuple
    point_um: np.ndarray
    tangent: np.ndarray  # outward unit direction, world um
    radius_um: float  # perimeter-equivalent, the convention the graph uses
    component: int  # mask component the tip sits on
    lobe_voxels: int
    elongation: float
    skeleton_um: np.ndarray  # (M, 3) the undescribed axis, tip first
    attach_node: int | None = None  # a graph node in the lobe's own component
    attach_um: np.ndarray | None = None  # where the lobe meets described lumen
    note: str = ""

    @property
    def index_zyx(self) -> np.ndarray:
        return np.asarray(self.key, dtype=np.int64)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<MaskEnd {self.key} r={self.radius_um:.0f}um comp "
                f"{self.component} lobe {self.lobe_voxels}vox "
                f"attach={self.attach_node}>")


@dataclass
class LobeReport:
    """What the sweep looked at, so an empty result is readable."""

    anchors: int = 0
    windows: int = 0
    windows_skipped_covered: int = 0
    windows_too_large: int = 0
    tips_found: int = 0
    tips_on_window_face: int = 0
    tips_small_lobe: int = 0
    tips_blob_lobe: int = 0
    tips_duplicate: int = 0
    ends: int = 0
    seconds: float = 0.0

    def describe(self) -> str:
        if not self.anchors:
            return "no associated free end to search around"
        bits = [(f"{self.ends} mask free end(s) with no centreline, from "
                 f"{self.windows} window(s) around {self.anchors} graph end(s)")]
        dropped = []
        if self.tips_on_window_face:
            dropped.append(f"{self.tips_on_window_face} on a window face")
        if self.tips_small_lobe:
            dropped.append(f"{self.tips_small_lobe} on a lobe too small")
        if self.tips_blob_lobe:
            dropped.append(f"{self.tips_blob_lobe} on a blob")
        if self.tips_duplicate:
            dropped.append(f"{self.tips_duplicate} already found")
        if dropped:
            bits.append("  discarded: " + ", ".join(dropped))
        if self.windows_too_large:
            bits.append(f"  {self.windows_too_large} window(s) too large to build")
        return "\n".join(bits)


# -------------------------------------------------------------------- the sweep


def find(index, frame, graph, associations=None, *,
         search_radii: float = SEARCH_RADII,
         describe_radii: float = DESCRIBE_RADII,
         min_voxels: int = MIN_LOBE_VOXELS,
         min_elongation: float = MIN_LOBE_ELONGATION,
         tangent_radii: float = TANGENT_RADII,
         duplicate_radii: float = DUPLICATE_RADII,
         max_window_voxels: int = MAX_WINDOW_VOXELS,
         progress=None):
    """Mask free ends with no centreline, in the neighbourhood of each graph end.

    `associations` is what :func:`.classify.associate` returned; only the ends that
    actually landed on the mask are used as anchors, because an unassociated end is
    already a finding and searching around it would compound one guess with another.

    Returns ``(ends, report)``. The report is not decoration: this step can
    legitimately find nothing, and "nothing" has half a dozen distinct causes that
    the count of ends alone cannot tell apart.
    """
    import time

    from scipy.spatial import cKDTree

    t0 = time.time()
    report = LobeReport()
    anchors = _anchors(associations)
    report.anchors = len(anchors)
    if not anchors:
        report.seconds = time.time() - t0
        return [], report

    points_um, owner_node = _graph_points(graph)
    tree = cKDTree(points_um) if len(points_um) else None
    spacing_zyx = np.asarray(frame.seg_spacing, dtype=np.float64)[::-1]
    shape = np.asarray(index.shape, dtype=np.int64)

    ends: list = []
    seen: set = set()
    done: list = []  # boxes already searched

    # Fattest first: a trunk's window swallows the twigs' windows, so processing it
    # first means the twigs are skipped rather than the other way round.
    for n, (point_um, radius_um) in enumerate(
            sorted(anchors, key=lambda a: -a[1]), 1):
        half = np.maximum(
            np.ceil(search_radii * radius_um / spacing_zyx), MIN_WINDOW_VOXELS
        ).astype(np.int64)
        centre = np.round(np.asarray(
            frame.um_to_seg(point_um[None, :]), dtype=np.float64)[0][::-1]
        ).astype(np.int64)
        lo = np.clip(centre - half, 0, shape)
        hi = np.clip(centre + half + 1, lo, shape)

        if _covered(centre, done):
            report.windows_skipped_covered += 1
            continue
        if int(np.prod(hi - lo)) > max_window_voxels:
            report.windows_too_large += 1
            continue

        done.append((lo, hi))
        report.windows += 1
        found = _window_ends(index, frame, lo, hi, radius_um, spacing_zyx,
                             tree, points_um, owner_node,
                             describe_radii=describe_radii, min_voxels=min_voxels,
                             min_elongation=min_elongation,
                             tangent_radii=tangent_radii, report=report)
        for end in found:
            if end.key in seen or _too_close(end, ends, duplicate_radii):
                report.tips_duplicate += 1
                continue
            seen.add(end.key)
            ends.append(end)
        if progress is not None:
            progress(n, len(anchors), len(ends))

    report.ends = len(ends)
    report.seconds = time.time() - t0
    return ends, report


def _anchors(associations) -> list:
    """Where to look: every free end that did land on the mask."""
    if associations is None:
        return []
    out = []
    for association in associations.values():
        if not association.associated:
            continue
        out.append((np.asarray(association.point_um, dtype=np.float64),
                    max(float(association.radius_um), 1e-6)))
    return out


def _graph_points(graph):
    """Every centreline point, and a node of the segment each one belongs to.

    The node is what the cycle test needs. A lobe hanging off the middle of a vessel
    has no *node* anywhere near it, so mapping the attachment to a nearby node
    directly would find nothing; mapping it to the nearest centreline point and then
    to that point's segment gives a node in the right component, which is the only
    thing the test ever asks.
    """
    if graph is None or not graph.points:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    owner_of_point = graph.segment_of_point()
    coords, owners = [], []
    for pid, value in graph.points.items():
        sid = owner_of_point.get(pid)
        if sid is None or not graph.has_segment(sid):
            continue
        coords.append(value[:3])
        owners.append(graph.segment(sid)["node1"])
    if not coords:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return (np.asarray(coords, dtype=np.float64),
            np.asarray(owners, dtype=np.int64))


def _covered(centre, done) -> bool:
    """Is this anchor already inside a window that has been searched?

    The middle half of the box, not the whole one: an anchor sitting in a window's
    outer rind can still see undescribed lumen that fell outside it, and skipping it
    there would lose exactly the ends nearest the edge of what has been looked at.
    """
    for lo, hi in done:
        mid = (lo + hi) / 2.0
        quarter = (hi - lo) / 4.0
        if np.all(np.abs(centre - mid) <= quarter):
            return True
    return False


def _too_close(end, kept, duplicate_radii: float) -> bool:
    for other in kept:
        scale = duplicate_radii * max(end.radius_um, other.radius_um)
        if float(np.linalg.norm(end.point_um - other.point_um)) <= scale:
            return True
    return False


# ------------------------------------------------------------------- one window


def _window_ends(index, frame, lo, hi, radius_um, spacing_zyx, tree, points_um,
                 owner_node, *, describe_radii, min_voxels, min_elongation,
                 tangent_radii, report) -> list:
    """Skeletonise one box of mask and keep the tips the graph does not describe."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    labels_window = index.window(lo, hi)
    if not labels_window.any():
        return []

    scale = max(describe_radii * float(radius_um), float(np.min(spacing_zyx)))
    # Everything below is a whole-array operation -- two distance transforms, a
    # thinning and a labelling -- and most of a window around a distal twig is
    # background. Cropping to what is actually foreground is worth several minutes
    # over a run and changes no answer, provided the pad is wide enough that every
    # centreline point which could describe a voxel inside the crop is still in it.
    pad = int(np.ceil(scale / float(np.min(spacing_zyx)))) + 1
    crop_lo, crop_hi = _foreground_box(labels_window, pad)
    if crop_lo is None:
        return []
    window_lo = np.asarray(lo, dtype=np.int64)  # the faces that mean "we stopped here"
    window_hi = np.asarray(hi, dtype=np.int64)
    lo = window_lo + crop_lo
    hi = window_lo + crop_hi
    labels_window = labels_window[crop_lo[0]:crop_hi[0], crop_lo[1]:crop_hi[1],
                                  crop_lo[2]:crop_hi[2]]
    mask = labels_window > 0

    described_distance = _centreline_distance(
        frame, lo, hi, mask.shape, spacing_zyx, tree, points_um, radius_um,
        describe_radii,
    )
    undescribed = mask & (described_distance > scale)
    if not undescribed.any():
        return []

    skel = np.asarray(skeletonize(mask, method="lee")) > 0
    if not skel.any():
        return []
    solid = skel.astype(np.uint8)
    degree = ndimage.convolve(solid, _NEIGHBOURHOOD, mode="constant", cval=0) - solid
    tips = skel & (degree <= 1) & undescribed
    if not tips.any():
        return []

    lobe_labels, _ = ndimage.label(undescribed, structure=_NEIGHBOURHOOD)
    radii = ndimage.distance_transform_edt(mask, sampling=spacing_zyx)
    face = _face_mask(mask.shape, lo, window_lo, window_hi)
    judged: dict = {}
    out: list = []

    for tip in np.argwhere(tips):
        report.tips_found += 1
        cell = tuple(int(v) for v in tip)
        if face[cell]:
            report.tips_on_window_face += 1
            continue
        lobe = int(lobe_labels[cell])
        if lobe not in judged:
            judged[lobe] = _judge_lobe(lobe_labels, lobe, spacing_zyx,
                                       min_voxels, min_elongation)
        ok, voxels, elongation, why = judged[lobe]
        if not ok:
            if "voxels" in why:
                report.tips_small_lobe += 1
            else:
                report.tips_blob_lobe += 1
            continue

        walk = _walk(skel, cell, spacing_zyx, described_distance, scale,
                     tangent_radii * float(radius_um),
                     ATTACH_RADII * float(radius_um))
        if walk is None:
            continue
        run, base, axis = walk
        tangent = _tangent(run, spacing_zyx)
        if tangent is None:
            continue

        tip_global = tuple(int(v) for v in (np.asarray(cell) + lo))
        point_um = np.asarray(frame.seg_to_um([tip_global[::-1]]),
                              dtype=np.float64)[0]
        # The radius convention has to match the graph's or the radius-ratio gate
        # compares two different measurements. The graph carries the
        # **perimeter-equivalent** radius, and on a collapsed vessel that is a very
        # different number from distance-to-background: a slit has the perimeter of
        # the vessel it was and the half-width of nothing at all. Taking the cheap
        # one here would make every mask end read as a twig beside its own trunk.
        along = np.asarray(run[: max(2, len(run) // 2)], dtype=np.int64)
        seed = max(float(np.median(radii[along[:, 0], along[:, 1], along[:, 2]])),
                   0.5 * float(np.min(spacing_zyx)))
        radius = _tip_radius(index, frame, run, lo, tangent,
                             int(labels_window[cell]), seed)

        attach_node, attach_um = None, None
        if base is not None:
            base_global = np.asarray(base, dtype=np.int64) + lo
            attach_um = np.asarray(frame.seg_to_um([base_global[::-1]]),
                                   dtype=np.float64)[0]
            attach_node = _owner_near(tree, owner_node, attach_um, 2.0 * scale)

        skeleton_um = np.asarray(frame.seg_to_um(
            (np.asarray(axis, dtype=np.int64) + lo)[:, ::-1]), dtype=np.float64)
        out.append(MaskEnd(
            key=tip_global, point_um=point_um, tangent=tangent, radius_um=radius,
            component=int(labels_window[cell]), lobe_voxels=voxels,
            elongation=elongation, skeleton_um=skeleton_um,
            attach_node=attach_node, attach_um=attach_um,
            note=("" if base is not None else
                  "no described lumen within reach along the mask"),
        ))
    return out


def _tip_radius(index, frame, run, lo, tangent, component: int, seed_um: float
                ) -> float:
    """The tip's calibre in the graph's own convention: perimeter-equivalent.

    Cut a little way back from the tip rather than at it. The very last voxel of a
    thinning sits where the lumen is closing, so a section there measures the taper
    instead of the vessel, and the taper is not what an arriving route has to match.

    Falls back to `seed_um` -- distance-to-background -- when the cut finds nothing
    usable, which happens on a lobe only a voxel or two thick. That is the wrong
    convention rather than a wrong number, and it is wrong in the direction that
    *under*-states the calibre, so it can only make the radius-ratio gate stricter.
    """
    from . import shape as shape_mod

    middle = np.asarray(run[len(run) // 2], dtype=np.int64) + np.asarray(lo)
    centre_um = np.asarray(frame.seg_to_um([middle[::-1]]), dtype=np.float64)[0]
    try:
        section = shape_mod.extract_section(index, frame, centre_um, tangent,
                                           component, seed_um)
    except Exception:  # noqa: BLE001 - a failed cut must not lose the end
        return seed_um
    if not section.valid or section.perimeter_um <= 0.0:
        return seed_um
    return max(float(section.r_perimeter), seed_um)


def _centreline_distance(frame, lo, hi, shape, spacing_zyx, tree, points_um,
                         radius_um, describe_radii) -> np.ndarray:
    """Distance from every voxel of the box to the nearest centreline point.

    Infinite where the graph has nothing in the box at all, which reads correctly as
    "nothing here is described" rather than as a distance of zero.
    """
    from scipy import ndimage

    empty = np.full(tuple(shape), np.inf, dtype=np.float64)
    if tree is None or not len(points_um):
        return empty

    lo_um = np.asarray(frame.seg_to_um([np.asarray(lo)[::-1]]), dtype=np.float64)[0]
    hi_um = np.asarray(frame.seg_to_um([(np.asarray(hi) - 1)[::-1]]),
                       dtype=np.float64)[0]
    centre = 0.5 * (lo_um + hi_um)
    # A ball query rather than a box one, padded so a centreline point just outside
    # the box still shortens the distance of the voxels it describes.
    reach = (float(np.linalg.norm(hi_um - lo_um)) / 2.0
             + describe_radii * float(radius_um) + 1.0)
    near = tree.query_ball_point(centre, reach)
    if not near:
        return empty

    idx = np.round(np.asarray(
        frame.um_to_seg(points_um[np.asarray(near, dtype=np.int64)]),
        dtype=np.float64)[:, ::-1]).astype(np.int64) - np.asarray(lo)
    keep = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    if not keep.any():
        return empty
    seed = np.zeros(tuple(shape), dtype=bool)
    seed[idx[keep, 0], idx[keep, 1], idx[keep, 2]] = True
    return ndimage.distance_transform_edt(~seed, sampling=spacing_zyx)


def _foreground_box(labels_window, pad: int):
    """The bounding box of foreground in the window, grown by `pad` and clipped.

    ``(None, None)`` when the window is empty.
    """
    where = np.argwhere(labels_window)
    if not len(where):
        return None, None
    shape = np.asarray(labels_window.shape)
    lo = np.maximum(where.min(axis=0) - pad, 0)
    hi = np.minimum(where.max(axis=0) + pad + 1, shape)
    return lo, hi


def _face_mask(shape, lo, window_lo, window_hi) -> np.ndarray:
    """True where this array's boundary is the *search window's* boundary.

    Not simply the edge of the array. After the foreground crop, most faces of the
    array sit in background a pad away from any vessel, and a tip there is a real
    vessel end rather than a truncation. Only the faces that coincide with the
    window's own faces mean "we stopped looking here".
    """
    out = np.zeros(tuple(shape), dtype=bool)
    lo = np.asarray(lo, dtype=np.int64)
    hi = lo + np.asarray(shape, dtype=np.int64)
    window_lo = np.asarray(window_lo, dtype=np.int64)
    window_hi = np.asarray(window_hi, dtype=np.int64)
    for axis in range(3):
        if lo[axis] <= window_lo[axis]:
            out[(slice(None),) * axis + (0,)] = True
        if hi[axis] >= window_hi[axis]:
            out[(slice(None),) * axis + (-1,)] = True
    return out


def _judge_lobe(lobe_labels, lobe: int, spacing_zyx, min_voxels: int,
                min_elongation: float):
    """Is this undescribed region a piece of vessel, or noise?"""
    coords = np.argwhere(lobe_labels == lobe)
    size = len(coords)
    if size < min_voxels:
        return False, size, 0.0, f"only {size} voxels"
    elongation = _elongation(coords, spacing_zyx)
    if elongation < min_elongation:
        return False, size, elongation, f"isotropic ({elongation:.1f}:1); a blob"
    return True, size, elongation, ""


def _elongation(coords_zyx, spacing_zyx) -> float:
    """Longest principal axis over the *shortest*: how far from a blob this is.

    Longest-over-shortest rather than longest-over-next-longest, which is what
    :func:`.classify._principal_extent` uses on whole components. The two agree on a
    round tube, where both cross-axes are the same, and disagree on the case that
    matters here: a short piece of collapsed ribbon is 130 um wide, 110 long and 30
    thick, so the classical ratio reads 1.2 and calls it a blob while this reads 4.3
    and calls it what it is. A sphere reads 1.0 either way.

    From the covariance rather than the bounding box, for the same reason
    :func:`.classify._principal_extent` does it: a diagonal vessel has a near-cubic
    bounding box and an axis-aligned test would call it a blob.
    """
    points = np.asarray(coords_zyx, dtype=np.float64) * np.asarray(spacing_zyx)
    if len(points) < 4:
        return 0.0
    centred = points - points.mean(axis=0)
    values = np.linalg.eigvalsh(np.cov(centred.T))
    extent = 2.0 * np.sqrt(np.maximum(values, 0.0))[::-1]
    return float(extent[0] / max(extent[2], 1e-9))


def _walk(skel, tip, spacing_zyx, described_distance, scale, tangent_reach,
          attach_reach):
    """Follow the skeleton inward from `tip`.

    Returns ``(run, base, axis)``, all in this array's coordinates:

    ``run``    the voxels out to `tangent_reach`, tip first. Short on purpose --
               it is what the outward tangent and the tip radius are measured over,
               and measuring those across a bend would misdirect the whole search.
    ``base``   the first voxel found inside described lumen: where this lobe joins
               the part of the vessel the graph already knows about, or ``None`` if
               the walk never got there within `attach_reach`.
    ``axis``   tip to base, tip first: the whole undescribed run. This is what the
               repair has to *describe*, so it sizes the box and the growth as well
               as supplying the calibration tail.

    Breadth-first rather than "follow the single neighbour", because a lobe is
    routinely a Y and the branch the tip is on ends at a junction a few voxels in.
    Stopping there would measure the tangent over three voxels of thinning noise.
    """
    shape = np.asarray(skel.shape)
    start = tuple(int(v) for v in tip)
    dist = {start: 0.0}
    parent: dict = {start: None}
    frontier = [start]
    run_end, run_best = start, 0.0
    far_end, far_best = start, 0.0
    base = None
    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)]
    step_um = {o: float(np.linalg.norm(np.asarray(o) * spacing_zyx))
               for o in offsets}

    while frontier:
        nxt = []
        for here in frontier:
            d_here = dist[here]
            if base is None and described_distance[here] <= scale:
                base = here
            if run_best <= d_here <= tangent_reach:
                run_end, run_best = here, d_here
            if d_here >= far_best:
                far_end, far_best = here, d_here
            if d_here > attach_reach:
                continue
            for offset in offsets:
                there = (here[0] + offset[0], here[1] + offset[1],
                         here[2] + offset[2])
                if not (0 <= there[0] < shape[0] and 0 <= there[1] < shape[1]
                        and 0 <= there[2] < shape[2]):
                    continue
                if not skel[there] or there in dist:
                    continue
                dist[there] = d_here + step_um[offset]
                parent[there] = here
                nxt.append(there)
        if base is not None and run_best >= tangent_reach:
            break
        frontier = nxt

    def chain(end):
        out = []
        node = end
        while node is not None:
            out.append(node)
            node = parent[node]
        out.reverse()  # the parent chain came back inward-first; hand it back tip-first
        return out

    run = chain(run_end)
    if len(run) < 2:
        return None
    # The axis stops where the lobe does -- at described lumen if the walk reached
    # any, and otherwise at the furthest the walk got. Running it on into described
    # territory would make a repair claim to describe centreline that already exists.
    return run, base, chain(base if base is not None else far_end)


def _tangent(run, spacing_zyx):
    """Outward unit direction in world um, from the run's inner end to its tip."""
    tip = np.asarray(run[0], dtype=np.float64) * spacing_zyx
    back = np.asarray(run[-1], dtype=np.float64) * spacing_zyx
    delta = (tip - back)[::-1]  # zyx -> xyz
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return None
    return delta / length


def _owner_near(tree, owner_node, point_um, reach_um):
    """A graph node in the component of the centreline nearest to `point_um`."""
    if tree is None or not len(owner_node):
        return None
    distance, which = tree.query(np.asarray(point_um, dtype=np.float64))
    if not np.isfinite(distance) or distance > reach_um:
        return None
    return int(owner_node[int(which)])


# ----------------------------------------------------------------- the proposer


def propose(graph, mask_ends, associations=None, *,
            cone_angle_deg: float | None = None,
            cone_length_factor: float | None = None,
            radius_ratio_max: float | None = None,
            tortuosity_max: float | None = None,
            keep_rejected: bool = False, stats: dict | None = None) -> list:
    """Bridges from graph free ends to mask free ends, through the same gates.

    Deliberately the same gates as :mod:`..endpoints`, applied to the same kind of
    Hermite proposal. A mask end is being offered as a peer of a graph end, and the
    honest way to make that claim is to put it through the identical test rather
    than a laxer one written for it.

    One bridge per source at most -- its best -- but a mask end may be proposed by
    several sources. Which of those wins is a global question, and :mod:`.select` is
    where global questions are answered.

    `stats`, if given, is filled with what the sweep found and each gate refused.
    Most mask free ends are *not* continuations of anything -- the commonest by far
    is lumen running on past a free end, which wants re-skeletonising rather than a
    route -- so a handful of proposals out of a hundred ends is the expected shape
    of the answer, and the counts are what make that distinguishable from a gate
    quietly throwing everything away.
    """
    from scipy.spatial import cKDTree

    from ..candidates import (
        CONE_ANGLE_DEG,
        CONE_LENGTH_FACTOR,
        RADIUS_RATIO_MAX,
        TORTUOSITY_MAX,
        Bridge,
        endpoint_tangent,
        gate_geometry,
        hermite_path,
        resample_by_arclength,
    )

    cone_angle_deg = CONE_ANGLE_DEG if cone_angle_deg is None else cone_angle_deg
    cone_length_factor = (CONE_LENGTH_FACTOR if cone_length_factor is None
                          else cone_length_factor)
    radius_ratio_max = (RADIUS_RATIO_MAX if radius_ratio_max is None
                        else radius_ratio_max)
    tortuosity_max = TORTUOSITY_MAX if tortuosity_max is None else tortuosity_max

    counts = {} if stats is None else stats
    for key in ("ends", "pairs_in_reach", "is_the_endpoint_itself", "proposed",
                "rejected_cone", "rejected_facing", "rejected_reach",
                "rejected_radius_ratio", "rejected_tortuosity"):
        counts.setdefault(key, 0)

    ends = list(mask_ends)
    counts["ends"] = len(ends)
    if not ends:
        return []
    tree = cKDTree(np.asarray([e.point_um for e in ends], dtype=np.float64))

    out: list = []
    for node in graph.endpoints():
        if associations is not None:
            association = associations.get(node)
            if association is not None and not association.associated:
                continue
        tangent = endpoint_tangent(graph, node)
        if tangent is None:
            continue
        t0, r0 = tangent
        p0 = np.asarray(graph.nodes[node][:3], dtype=np.float64)
        best = None

        for j in tree.query_ball_point(p0, cone_length_factor * r0):
            end = ends[j]
            # A tip that is this endpoint's own, found again through the mask, is
            # not a target. The describedness filter removes nearly all of these;
            # this removes the rest.
            counts["pairs_in_reach"] += 1
            if float(np.linalg.norm(end.point_um - p0)) <= max(r0, 1e-6):
                counts["is_the_endpoint_itself"] += 1
                continue
            path = hermite_path(p0, t0, end.point_um, -np.asarray(end.tangent), 32)
            bridge = Bridge(
                kind="mask-end", source_node=node, coords=path,
                radii=np.linspace(r0, end.radius_um, len(path)),
                reconnection_type=1,
            )
            bridge.target_mask_end = end
            gate_geometry(bridge, r0, end.radius_um,
                          cone_angle_deg=cone_angle_deg,
                          cone_length_factor=cone_length_factor,
                          radius_ratio_max=radius_ratio_max,
                          tortuosity_max=tortuosity_max, source_tangent=t0)
            if bridge.accepted:
                span = bridge.span_um
                if span > 1e-9:
                    approach = (p0 - end.point_um) / span
                    cos_back = float(np.clip(np.dot(end.tangent, approach), -1.0, 1.0))
                    bridge.metrics["target_cone_deg"] = float(
                        np.degrees(np.arccos(cos_back))
                    )
                    if bridge.metrics["target_cone_deg"] > cone_angle_deg:
                        bridge.reject("the mask end faces away")
            bridge.metrics["lobe_voxels"] = int(end.lobe_voxels)
            bridge.metrics["lobe_elongation"] = float(end.elongation)
            bridge.score = _score(bridge, r0, end.radius_um)
            if not bridge.accepted:
                counts[_reason_key(bridge.reason)] =                     counts.get(_reason_key(bridge.reason), 0) + 1
                if keep_rejected:
                    out.append(bridge)
                continue
            if best is None or bridge.score > best.score:
                best = bridge

        if best is not None:
            spacing = max(0.9 * min(best.metrics["r_source"],
                                    best.metrics["r_target"]), 1.0)
            coords = resample_by_arclength(best.coords, spacing)
            best.coords = coords
            best.radii = np.linspace(best.metrics["r_source"],
                                     best.metrics["r_target"], len(coords))
            counts["proposed"] += 1
            out.append(best)
    return out


def _reason_key(reason: str) -> str:
    """Which gate a refusal came from, as a stats key."""
    if "cone" in reason:
        return "rejected_cone"
    if "faces away" in reason:
        return "rejected_facing"
    if "radius" in reason:
        return "rejected_radius_ratio"
    if "tortuosity" in reason:
        return "rejected_tortuosity"
    if "further than" in reason:
        return "rejected_reach"
    return "rejected_other"


def _score(bridge, r0: float, r1: float) -> float:
    """Higher is better -- the same shape as :func:`..endpoints._score`.

    Damped by a constant, because a mask end is a weaker premise than a graph end:
    its position comes from one local thinning rather than from the skeletonisation
    the whole graph was built by. Where a mask-end proposal and an end-to-end
    proposal compete for the same free end and score alike, the one standing on the
    stronger premise should win.
    """
    span = max(bridge.span_um, 1e-6)
    reach = max(bridge.metrics.get("reach_um", span), 1e-6)
    closeness = 1.0 - min(span / reach, 1.0)
    straightness = 1.0 / max(bridge.tortuosity, 1.0)
    match = min(r0, r1) / max(r0, r1, 1e-9)
    cone = bridge.metrics.get("cone_deg")
    aim = 1.0 - min(cone / 180.0, 1.0) if cone is not None else 0.5
    return float(0.9 * (0.35 * closeness + 0.25 * straightness
                        + 0.25 * match + 0.15 * aim))


def summarise(ends, report=None) -> str:
    """A tally, with the largest few named."""
    if report is not None and not ends:
        return report.describe()
    lines = [f"{len(ends)} mask free end(s) with no centreline"]
    for end in sorted(ends, key=lambda e: -e.lobe_voxels)[:5]:
        lines.append(
            f"  {end.key} r={end.radius_um:.0f}um, lobe {end.lobe_voxels} voxels "
            f"({end.elongation:.1f}:1) on mask component {end.component}"
            + (f", hanging off node {end.attach_node}"
               if end.attach_node is not None
               else ", not attached to any described lumen")
        )
    return "\n".join(lines)
