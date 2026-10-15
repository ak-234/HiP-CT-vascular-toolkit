"""Seeded mask repairs: fill a collapsed lumen, split two fused vessels.

Two segmentation defects that no global pass in this package fixes, both measured on
LADAF-2021-17:

* **A collapsed segment with a missing chunk.** The lumen has flattened to a dark slit
  between two bright walls and the segmentation stopped following it, so the mask has
  a notch -- usually open on one side, which is why ``repair-mask --guarded`` (it fills
  only enclosed holes) never proposes it. Every skeletoniser then routes around the
  notch and leaves a false side branch or loop.
* **Two vessels fused** into one blob across a thin neck, which every skeletoniser
  reads as a junction.

Both are fixed here from seeds the user places, inside a bounded crop, and both
**decide locally**. That is not a convenience. ``mask_repair``'s module docstring
records that a global brightness gate failed on this dataset: lumen and enclosed tissue
overlap in intensity, and near edge 7981 the tissue is darker than the vessel's own
lumen. Calibrated on one vessel's own lumen and wall, inside a capsule around the seeds,
the same contrast is usable -- and where it is not, the tool says so and does nothing.

Nothing here touches a viewer or a file. Each tool takes crops on the segmentation grid
(``labels`` uint8, ``raw`` float32 with NaN outside the stack) and returns a
:class:`ToolResult` saying what to write, what it refused, and why. The paint panel
writes the result into napari's own undo history; the follow-up automated pass can
write the same result into an edit store.

Like ``mask_repair``, every gate errs towards *not* changing the mask: a refused region
leaves the defect the skeleton already had, while a wrongly accepted one invents
anatomy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

#: 26-connectivity, the reading every skeletoniser and ``mask_repair`` use.
CONN26 = np.ones((3, 3, 3), dtype=bool)
CONN6 = ndimage.generate_binary_structure(3, 1)

ROI_RADIUS_RANGE = (4.0, 30.0)  # voxels, for the automatic capsule radius
MIN_CALIBRATION = 50  # lumen voxels below which the eroded sample is abandoned
MIN_WALL = 20
LEAK_VOXELS = 8
NEAR_OWN = 2  # voxels: dark rim this close to the vessel's own mask is not a leak
MAX_SPLIT_VOXELS = 4_000_000


@dataclass
class ToolResult:
    """What a tool wants written into the crop it was given, and what it refused."""

    kind: str  # "fill" | "split"
    write: np.ndarray  # bool, same shape as the crop
    value: int  # label to write: the tree's label for a fill, 0 for a cut
    refused: np.ndarray  # bool: proposed and refused, for the preview
    reasons: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.write.any())

    def describe(self) -> str:
        n = int(self.write.sum())
        verb = {"fill": "filled", "split": "cut"}.get(self.kind, self.kind)
        head = (f"{self.kind}: {verb} {n:,} voxel(s)" if n
                else f"{self.kind}: nothing written")
        if self.kind == "fill" and n:
            head += f" as label {self.value}"
        refused = int(self.refused.sum())
        if refused:
            head += f", refused {refused:,}"
        lines = [head] + [f"  {r}" for r in self.reasons]
        cal = self.stats.get("calibration")
        if cal:
            lines.append(f"  calibration: {cal}")
        return "\n".join(lines)


def _refuse(kind, shape, reason, *, value=0, refused=None, stats=None) -> ToolResult:
    empty = np.zeros(shape, dtype=bool)
    return ToolResult(kind, empty, int(value),
                      refused if refused is not None else empty.copy(),
                      [reason], stats or {})


# ----------------------------------------------------------------------- helpers


def majority_label(labels, where) -> int:
    """The commonest non-zero label inside `where`, or 0 when there is none."""
    vals = np.asarray(labels)[np.asarray(where, dtype=bool)]
    vals = vals[vals > 0]
    if not vals.size:
        return 0
    return int(np.bincount(vals.astype(np.int64)).argmax())


def _seeds(seeds_zyx) -> np.ndarray:
    s = np.asarray(seeds_zyx, dtype=np.float64)
    return s.reshape(-1, 3) if s.size else np.empty((0, 3))


def seeds_to_roi(shape, seeds_zyx, radius: float) -> np.ndarray:
    """Union of capsules of `radius` voxels along the seed polyline, in order.

    One seed is a ball. The polyline is what lets a user trace a collapsed run with a
    handful of clicks and get a tube along it rather than a string of beads.
    """
    shape = tuple(int(v) for v in shape)
    out = np.zeros(shape, dtype=bool)
    pts = _seeds(seeds_zyx)
    if not len(pts):
        return out
    r = float(radius)
    segs = list(zip(pts[:-1], pts[1:])) if len(pts) > 1 else [(pts[0], pts[0])]
    for a, b in segs:
        lo = np.maximum(np.floor(np.minimum(a, b) - r).astype(int), 0)
        hi = np.minimum(np.ceil(np.maximum(a, b) + r).astype(int) + 1, shape)
        if np.any(hi <= lo):
            continue
        g = np.stack(np.meshgrid(*(np.arange(lo[d], hi[d]) for d in range(3)),
                                 indexing="ij"), axis=-1).astype(np.float64)
        ab = b - a
        L2 = float(ab @ ab)
        if L2 > 0:
            t = np.clip(((g - a) @ ab) / L2, 0.0, 1.0)
            near = a + t[..., None] * ab
        else:
            near = a
        d2 = np.sum((g - near) ** 2, axis=-1)
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= d2 <= r * r
    return out


def _polyline_voxels(shape, pts) -> np.ndarray:
    """Voxels on the straight segments between consecutive points (and the points)."""
    out = np.zeros(shape, dtype=bool)
    pts = _seeds(pts)
    for i in range(len(pts)):
        a = pts[i]
        b = pts[i + 1] if i + 1 < len(pts) else pts[i]
        n = int(np.ceil(np.max(np.abs(b - a)))) + 1
        line = np.rint(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)).astype(int)
        keep = np.all((line >= 0) & (line < np.asarray(shape)), axis=1)
        line = line[keep]
        out[line[:, 0], line[:, 1], line[:, 2]] = True
    return out


def auto_radius(fg, seeds_zyx, lo: float = ROI_RADIUS_RANGE[0],
                hi: float = ROI_RADIUS_RANGE[1]) -> float:
    """1.5 x the 95th-percentile mask radius near the seeds, clamped to [lo, hi].

    Measured on the mask *beside* the gap, because the gap itself has no radius to
    measure -- that is the defect.
    """
    near = seeds_to_roi(fg.shape, seeds_zyx, hi) & fg
    if not near.any():
        return float(lo)
    sl = ndimage.find_objects(near.astype(np.uint8))[0]
    pad = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
    edt = ndimage.distance_transform_edt(fg[pad])
    vals = edt[near[pad]]
    return float(np.clip(1.5 * np.percentile(vals, 95.0), lo, hi))


def _bbox(mask, pad: int, shape) -> tuple[slice, slice, slice] | None:
    found = ndimage.find_objects(mask.astype(np.uint8))
    if not found or found[0] is None:
        return None
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n))
                 for s, n in zip(found[0], shape))


def _ball(r: int) -> np.ndarray:
    g = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return g[0] ** 2 + g[1] ** 2 + g[2] ** 2 <= r * r


def _calibrate(raw, lumen, wall):
    """Lumen and wall statistics from two explicit samples.

    ``cost.calibrate`` takes the wall as a shell grown from the lumen. Here that
    shell would include the gap being filled -- the dark slit itself -- and other
    vessels' foreground, so the wall sample is passed in already cleaned.
    """
    from .reconnect.geodesic.cost import Calibration, _robust_sigma

    lv = raw[lumen]
    wv = raw[wall]
    lv = lv[np.isfinite(lv)]
    wv = wv[np.isfinite(wv)]
    if not lv.size or not wv.size:
        return None
    return Calibration(
        lumen_mu=float(np.median(lv)), lumen_sigma=_robust_sigma(lv),
        wall_mu=float(np.median(wv)), wall_sigma=_robust_sigma(wv),
        n_lumen=int(lv.size), n_wall=int(wv.size), dark_lumen=True,
    )


def _name(value: int, names) -> str:
    if names and int(value) in names:
        return f"{names[int(value)]} ({int(value)})"
    return f"label {int(value)}"


# ------------------------------------------------------------------------- fill


def fill_collapsed(labels, raw, seeds_zyx, *, value: int | None = None,
                   roi_radius: float | None = None, p_lumen: float = 0.5,
                   p_floor: float = 0.2, slit_min: float = 0.3,
                   slit_sigma: float = 1.0, max_vox: int = 50_000,
                   max_fraction: float = 0.6, thicken: int = 1,
                   leak_shell: int = 4, allow_join: bool = False,
                   names: dict | None = None) -> ToolResult:
    """Fill the dark lumen the mask missed, inside a capsule along the seeds.

    `seeds_zyx` are crop-local voxel coordinates placed along the collapsed run.
    The fill grows from the vessel's own mask through voxels whose intensity reads as
    *this vessel's* lumen, or as a thin slit between walls, and is refused piece by
    piece if it runs out of the capsule (an unrelated dark region, not a slit), would
    join the other tree, or would make a new tunnel. See the module docstring.

    `names` maps label values to material names for the messages.
    """
    from .mask_repair import TOPOLOGY_MARGIN, topology_change
    from .reconnect.geodesic.cost import flux_medialness

    labels = np.asarray(labels)
    shape = labels.shape
    seeds = _seeds(seeds_zyx)
    if not len(seeds):
        return _refuse("fill", shape, "no seeds: place points along the collapsed run")
    if raw is None:
        return _refuse("fill", shape, "the fill needs the raw image (--raw)")
    raw = np.asarray(raw, dtype=np.float32)
    fg = labels > 0

    R = float(roi_radius) if roi_radius else auto_radius(fg, seeds)
    capsule = seeds_to_roi(shape, seeds, R)
    roi = capsule & ~fg
    if not roi.any():
        return _refuse("fill", shape, "the seeds are already inside the mask",
                       stats={"roi_radius": R})

    # Work in a box around everything the fill could reach. The paint box is 192^3
    # and the Hessian-free flux still costs a few gradients; no reason to pay it on
    # tissue the capsule never visits.
    pad = leak_shell + max(thicken, 0) + TOPOLOGY_MARGIN + 3
    box = _bbox(capsule, pad, shape)
    L, F, Rw = labels[box], fg[box], raw[box]
    roi_b, cap_b = roi[box], capsule[box]
    seeds_b = seeds - np.array([s.start for s in box], dtype=np.float64)

    touch = ndimage.binary_dilation(roi_b, CONN26, iterations=2) & F
    if value is None:
        value = majority_label(L, touch)
    value = int(value)
    if value <= 0:
        return _refuse("fill", shape, "no mask touches the capsule: seed closer to "
                       "the vessel, or raise the ROI radius", stats={"roi_radius": R})
    comps, _ = ndimage.label(L == value, CONN26)
    ids = np.unique(comps[touch & (comps > 0)])
    own = np.isin(comps, ids[ids > 0])
    if not own.any():
        return _refuse("fill", shape, f"no {_name(value, names)} mask touches the "
                       "capsule", value=value, stats={"roi_radius": R})

    # ---- calibrate on this vessel, here
    near = seeds_to_roi(L.shape, seeds_b, 2.0 * R) & own
    lumen = ndimage.binary_erosion(near, CONN6)
    if lumen.sum() < MIN_CALIBRATION:
        lumen = near
    wall = ndimage.binary_dilation(near, CONN26, iterations=2) & ~F & ~roi_b
    if wall.sum() < MIN_WALL:
        wall = ndimage.binary_dilation(near, CONN26, iterations=2) & ~F
    cal = _calibrate(Rw, lumen, wall)
    stats = {"roi_radius": R, "value": value}
    if cal is None:
        return _refuse("fill", shape, "no raw values beside the vessel (outside the "
                       "stack?)", value=value, stats=stats)
    stats["calibration"] = cal.describe()
    if not cal.separated:
        return _refuse("fill", shape, "no lumen/wall contrast on this vessel: the raw "
                       "image cannot say what is lumen here - paint it by hand",
                       value=value, stats=stats)

    # ---- score
    valid = np.isfinite(Rw)
    filled = np.where(valid, Rw, np.float32(cal.wall_mu))
    p = np.where(valid, cal.lumen_likelihood(filled), 0.0)
    flux = flux_medialness(filled, slit_sigma, dark_lumen=True)
    ref_region = own | roi_b
    scale = float(np.percentile(flux[ref_region], 95.0)) if ref_region.any() else 0.0
    slit = np.clip(flux / scale, 0.0, 1.0) if scale > 0 else np.zeros_like(flux)
    cand = ((p >= p_lumen) | ((slit >= slit_min) & (p >= p_floor))) & valid & ~F
    roi_ext = ndimage.binary_dilation(cap_b, CONN26, iterations=leak_shell) & ~F

    # ---- grow from the vessel (and from seeds that are themselves lumen-like)
    seed_vox = _polyline_voxels(L.shape, seeds_b) & cand
    start = own | seed_vox
    grown = ndimage.binary_dilation(start, CONN6, iterations=0,
                                    mask=start | (cand & roi_ext)) & ~own

    # ---- leak: a slit is closed in by walls; an unrelated dark region keeps going
    shell = roi_ext & ~roi_b
    excused = ndimage.binary_dilation(own, CONN26, iterations=NEAR_OWN)
    pieces, n = ndimage.label(grown, CONN26)
    refused = np.zeros(L.shape, dtype=bool)
    keep = np.zeros(L.shape, dtype=bool)
    reasons: list[str] = []
    leaks = 0
    for i, sl in enumerate(ndimage.find_objects(pieces), start=1):
        if sl is None:
            continue
        piece = pieces[sl] == i
        if np.count_nonzero(piece & shell[sl] & ~excused[sl]) >= LEAK_VOXELS:
            refused[sl] |= piece & roi_b[sl]
            leaks += 1
        else:
            keep[sl] |= piece & roi_b[sl]
    if leaks:
        reasons.append(f"{leaks} dark region(s) continue past the capsule (a leak, "
                       "not a slit) - extend the seeds along the run, or paint by hand")

    # ---- thicken: a skeletoniser should see a tube, not a one-voxel sheet
    if thicken > 0 and keep.any():
        closed = ndimage.binary_closing(own | keep, _ball(int(thicken)))
        keep |= closed & roi_b & ndimage.binary_dilation(keep, CONN26, iterations=thicken)
        keep &= ~F

    total = int(keep.sum())
    stats["proposed"] = total
    cap = min(int(max_vox), int(max_fraction * roi_b.sum()))
    if total > cap:
        refused |= keep
        reasons.append(f"would add {total:,} voxels, over the cap of {cap:,} "
                       f"({max_fraction:.0%} of the capsule or {max_vox:,}): too much "
                       "of the capsule reads as lumen - seed tighter or lower the radius")
        return _expand(ToolResult("fill", np.zeros(L.shape, bool), value, refused,
                                  reasons, stats), box, shape)

    # ---- topology, one piece at a time against the mask plus what is accepted
    fg_now = F.copy()
    out = np.zeros(L.shape, dtype=bool)
    joined_loops = 0
    pieces, n = ndimage.label(keep, CONN26)
    sizes = ndimage.sum_labels(keep, pieces, index=np.arange(1, n + 1)) if n else []
    margin = max(TOPOLOGY_MARGIN, 3)
    for i in (np.argsort(sizes)[::-1] + 1 if n else []):
        sl = _bbox(pieces == i, margin, L.shape)
        piece = pieces[sl] == i
        db0, db1, _db2 = topology_change(fg_now[sl], piece)
        why = None
        if db0 > 0:
            why = "touches no mask (would start a separate piece)"
        elif db0 < 0:
            labs, _ = ndimage.label(fg_now[sl], CONN26)
            hit = np.unique(labs[ndimage.binary_dilation(piece, CONN26) & fg_now[sl]])
            vals = np.unique(np.where(out[sl], value, L[sl])[np.isin(labs, hit[hit > 0])])
            other = [int(v) for v in vals if v > 0 and v != value]
            if other:
                why = f"would merge with {_name(other[0], names)}"
            elif not allow_join:
                why = (f"would join {1 - db0} separate {_name(value, names)} pieces "
                       "(tick 'allow join' if that break is real)")
        if why is None and db1 > 0:
            why = "would make a new tunnel through the mask"
        if why is not None:
            refused[sl] |= piece
            reasons.append(f"refused {int(piece.sum()):,} voxel(s): {why}")
            continue
        if db1 < 0:
            joined_loops += 1
        out[sl] |= piece
        fg_now[sl] |= piece
    if joined_loops:
        reasons.append(f"closes {joined_loops} loop(s) - usually the false loop the "
                       "skeleton made around the notch")
    if not out.any() and not reasons:
        reasons.append("nothing in the capsule reads as this vessel's lumen")
    return _expand(ToolResult("fill", out, value, refused, reasons, stats), box, shape)


def _expand(result: ToolResult, box, shape) -> ToolResult:
    """Put a box-local result back into the caller's crop."""
    w = np.zeros(shape, dtype=bool)
    r = np.zeros(shape, dtype=bool)
    w[box] = result.write
    r[box] = result.refused
    result.write, result.refused = w, r
    return result


# ------------------------------------------------------------------------ split


def _snap(fg, seed, reach: int = 3):
    """The foreground voxel nearest `seed` within `reach`, or ``None``."""
    p = np.rint(seed).astype(int)
    shape = np.asarray(fg.shape)
    if np.all((p >= 0) & (p < shape)) and fg[tuple(p)]:
        return p
    lo = np.maximum(p - reach, 0)
    hi = np.minimum(p + reach + 1, shape)
    if np.any(hi <= lo):
        return None
    sub = fg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    hits = np.argwhere(sub)
    if not len(hits):
        return None
    hits = hits + lo
    return hits[np.argmin(np.sum((hits - p) ** 2, axis=1))]


def cut_between(basins, elevation, width: int = 1) -> np.ndarray:
    """A separating layer between adjacent basins that is safe under 26-connectivity.

    For each touching pair it removes the *whole* one-voxel boundary layer on one
    side -- every voxel of that basin with a 26-neighbour in the other -- so nothing
    left of one basin can touch the other. skimage's ``watershed_line`` leaves a
    6-connected line, through which two 26-connected vessels still touch at corners.
    The side removed is the one with the higher mean elevation, so the cut sits on
    the wall rather than in either lumen.
    """
    b = np.asarray(basins).copy()
    e = np.asarray(elevation, dtype=np.float64)
    cut = np.zeros(b.shape, dtype=bool)
    ids = [int(v) for v in np.unique(b) if v > 0]
    for ai, a in enumerate(ids):
        for c in ids[ai + 1:]:
            A, C = b == a, b == c
            sa = A & ndimage.binary_dilation(C, CONN26)
            if not sa.any():
                continue
            sc = C & ndimage.binary_dilation(A, CONN26)
            side, owner = (sa, A) if e[sa].mean() >= e[sc].mean() else (sc, C)
            if width > 1:
                side = ndimage.binary_dilation(side, CONN6, iterations=width - 1,
                                               mask=owner)
            cut |= side
            b[side] = 0
    return cut


def split_fused(labels, raw, seed_groups, *, margin: int | None = None,
                intensity_weight: float = 0.5, cut_width: int = 1,
                neck_max: float = 0.8, wall_min: float = 0.5, force: bool = False,
                names: dict | None = None) -> ToolResult:
    """Cut a fused blob into one piece per seed group, along its thinnest wall.

    `seed_groups` is a list of ``(N, 3)`` crop-local voxel coordinates, one group per
    vessel; a group's points are joined in order, so a few clicks along each vessel
    are enough. The cut is a seeded watershed on a mix of the distance transform
    (necks are low) and the raw image (the wall between two lumens is bright), and is
    refused when it would cross a healthy lumen rather than a neck.
    """
    from skimage.segmentation import watershed

    labels = np.asarray(labels)
    shape = labels.shape
    fg = labels > 0
    groups = [_seeds(g) for g in seed_groups]
    groups = [g for g in groups if len(g)]
    if len(groups) < 2:
        return _refuse("split", shape, "needs two seed groups, one on each vessel")

    snapped = []
    for gi, g in enumerate(groups, start=1):
        pts = []
        for s in g:
            q = _snap(fg, s)
            if q is None:
                return _refuse("split", shape, f"a group {gi} seed is not on the mask")
            pts.append(q)
        snapped.append(np.asarray(pts, dtype=np.float64))

    every = np.vstack(snapped)
    if margin is None:
        lo = np.maximum(every.min(axis=0).astype(int) - 40, 0)
        hi = np.minimum(every.max(axis=0).astype(int) + 41, shape)
        edt = ndimage.distance_transform_edt(fg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        local = (every - lo).astype(int)
        r = float(edt[local[:, 0], local[:, 1], local[:, 2]].max())
        margin = int(np.clip(2.0 * r, 8, 40))
    lo = np.maximum(every.min(axis=0).astype(int) - margin, 0)
    hi = np.minimum(every.max(axis=0).astype(int) + margin + 1, shape)
    if int(np.prod(hi - lo)) > MAX_SPLIT_VOXELS:
        return _refuse("split", shape, "the seeds span too large a region: split one "
                       "fusion at a time")
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    F = fg[box]
    local_groups = [g - lo for g in snapped]

    comps, _ = ndimage.label(F, CONN26)
    seed_comp = {int(c) for g in local_groups for c in comps[tuple(g.astype(int).T)]}
    if len(seed_comp) > 1:
        return _refuse("split", shape, "the seed groups are already in separate pieces "
                       "of the mask here - nothing to split")
    C = comps == seed_comp.pop()

    # ---- elevation: low in lumens, high on necks and walls
    d = ndimage.distance_transform_edt(C)
    geo = 1.0 - d / max(float(d.max()), 1e-6)
    w = 0.0
    stats: dict = {"margin": margin}
    reasons: list[str] = []
    norm = None
    if raw is not None:
        Rw = np.asarray(raw, dtype=np.float32)[box]
        inside = ndimage.binary_erosion(C, CONN6)
        if inside.sum() < MIN_CALIBRATION:
            inside = C
        wall = ndimage.binary_dilation(C, CONN26, iterations=2) & ~F
        cal = _calibrate(Rw, inside, wall)
        if cal is not None:
            stats["calibration"] = cal.describe()
        if cal is not None and cal.separated and cal.wall_mu != cal.lumen_mu:
            sm = ndimage.gaussian_filter(np.where(np.isfinite(Rw), Rw,
                                                  np.float32(cal.wall_mu)), 0.7)
            norm = np.clip((sm - cal.lumen_mu) / (cal.wall_mu - cal.lumen_mu), 0.0, 1.0)
            w = float(intensity_weight)
        else:
            reasons.append("no lumen/wall contrast here: cutting on shape alone")
    else:
        reasons.append("no raw image: cutting on shape alone")
    elevation = (1.0 - w) * geo + (w * norm if norm is not None else 0.0)

    # ---- markers along each group's polyline
    markers = np.zeros(F.shape, dtype=np.int32)
    for gi, g in enumerate(local_groups, start=1):
        line = _polyline_voxels(F.shape, g) & C
        if np.any(markers[line] > 0):
            return _refuse("split", shape, "the seed groups overlap: put each group on "
                           "its own vessel")
        markers[line] = gi
    basins = watershed(elevation, markers, mask=C, connectivity=1)
    stray = C & (basins == 0)
    if stray.any():
        _, idx = ndimage.distance_transform_edt(basins == 0, return_indices=True)
        basins[stray] = basins[tuple(i[stray] for i in idx)]

    cut = cut_between(basins, elevation, width=max(int(cut_width), 1))
    if not cut.any():
        return _refuse("split", shape, "the watershed found no boundary between the "
                       "groups", stats=stats)

    # ---- is this a neck, or a healthy lumen?
    seed_r = [float(np.median(d[tuple(g.astype(int).T)])) for g in local_groups]
    # The cut's thickest point against the vessels': a disc across a healthy tube
    # reaches the axis and scores ~1; a neck between two lumens never does.
    ratio = float(d[cut].max()) / max(min(seed_r), 1e-6)
    wall_score = float(norm[cut].mean()) if norm is not None else float("nan")
    stats.update(neck_ratio=round(ratio, 3), wall=round(wall_score, 3),
                 cut_voxels=int(cut.sum()))
    neck = ratio <= neck_max or (norm is not None and wall_score >= wall_min)
    full_cut = np.zeros(shape, dtype=bool)
    full_cut[box] = cut
    if not (neck or force):
        reasons.append(f"refused: the cut crosses a lumen, not a neck (neck ratio "
                       f"{ratio:.2f} > {neck_max}, wall {wall_score:.2f} < {wall_min}); "
                       "tick 'force' if you are sure")
        return ToolResult("split", np.zeros(shape, bool), 0, full_cut, reasons, stats)
    if force and not neck:
        reasons.append(f"forced: neck ratio {ratio:.2f}, wall {wall_score:.2f}")

    # ---- verify: each vessel whole, the vessels apart
    after, _ = ndimage.label(C & ~cut, CONN26)
    owners = []
    for gi, g in enumerate(local_groups, start=1):
        line = _polyline_voxels(F.shape, g) & C & ~cut
        got = {int(v) for v in np.unique(after[line]) if v > 0}
        if len(got) != 1:
            return ToolResult("split", np.zeros(shape, bool), 0, full_cut,
                              reasons + [f"refused: the cut breaks group {gi}'s vessel "
                                         "apart - add seeds along it"], stats)
        owners.append(got.pop())
    if len(set(owners)) < len(owners):
        return ToolResult("split", np.zeros(shape, bool), 0, full_cut,
                          reasons + ["refused: the groups are still joined around the "
                                     "cut"], stats)

    # Separate here; possibly still joined further away, which is a note, not a
    # failure -- a shared parent artery joins every pair of branches somewhere.
    before_n = ndimage.label(fg, CONN26)[1]
    whole, after_n = ndimage.label(fg & ~full_cut, CONN26)
    stats.update(components_before=int(before_n), components_after=int(after_n))
    ids = {int(whole[tuple((g + lo).astype(int).T)][0]) for g in local_groups}
    if len(ids) < len(local_groups):
        reasons.append("separated here, but still joined elsewhere in the paint box "
                       "(another fusion, or a genuine shared parent)")
    return ToolResult("split", full_cut, 0, np.zeros(shape, bool), reasons, stats)


# ---------------------------------------------------------------------- relabel


def relabel_added(source, *, reach: int = 8, tile: int = 32) -> dict:
    """Give every painted voxel the label of the tree it was painted onto.

    Repairs stores painted before the label picker existed, when every stroke wrote
    1: on a mask with Left_Tree=1 and Right_Tree=2, a correction to the right
    coronary silently moved it into the left tree.

    * a voxel painted over background takes the nearest original label within
      `reach` voxels (left as it is when there is none: ``isolated``);
    * a voxel painted over a *different* non-zero label is dropped, so the source
      shows through again (``restored``).

    Erasures are untouched. Returns the three counts.
    """
    edits = source.edits
    k, row, col, val = edits.entries()
    counts = {"relabelled": 0, "restored": 0, "isolated": 0}
    painted = val > 0
    if not painted.any():
        return counts
    k, row, col, val = k[painted], row[painted], col[painted], val[painted]
    key = np.stack([k // tile, row // tile, col // tile], axis=1)
    _, which = np.unique(key, axis=0, return_inverse=True)
    which = np.asarray(which).ravel()
    for t in range(int(which.max()) + 1):
        m = which == t
        kk, rr, cc, vv = k[m], row[m], col[m], val[m]
        k0, j0, i0 = int(kk.min()) - reach, int(rr.min()) - reach, int(cc.min()) - reach
        k1, j1, i1 = int(kk.max()) + reach + 1, int(rr.max()) + reach + 1, int(cc.max()) + reach + 1
        base = source.window(k0, k1, j0, j1, i0, i1, edited=False)
        lk, lr, lc = kk - k0, rr - j0, cc - i0
        under = base[lk, lr, lc]

        clash = (under > 0) & (under != vv)
        for z in np.unique(kk[clash]):
            sel = clash & (kk == z)
            counts["restored"] += edits.drop(int(z), rr[sel], cc[sel])

        bare = under == 0
        if not bare.any():
            continue
        dist, idx = ndimage.distance_transform_edt(base == 0, return_indices=True)
        nk, nr, nc = (i[lk, lr, lc] for i in idx)
        nearest = base[nk, nr, nc]
        found = bare & (dist[lk, lr, lc] <= reach) & (nearest > 0)
        counts["isolated"] += int(np.count_nonzero(bare & ~found))
        change = found & (nearest != vv)
        for z in np.unique(kk[change]):
            sel = change & (kk == z)
            counts["relabelled"] += edits.set_plane(int(z), rr[sel], cc[sel], nearest[sel])
    return counts
