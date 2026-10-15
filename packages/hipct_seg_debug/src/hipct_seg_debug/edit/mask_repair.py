"""Guarded repair of a vessel label mask before skeletonisation.

Two segmentation defects put wrong centrelines into every skeletoniser downstream.
Both were measured on LADAF-2021-17 around edges a full-tree ``jin-mgf`` got wrong:

* **A hollow lumen** (edge 9321). Over a run of slices only the vessel *wall* is
  labelled, as a ring, and the dark lumen inside is not. A hollow tube's medial set is
  a curved sheet, and the crop gave 126 tips and 121 junctions. A 3D fill-holes does
  nothing because the core is open where the ring breaks; filling the ring's interior
  slice by slice, along all three axes, gave 8 tips and 6 junctions.
* **Small enclosed voids** (edge 7981). A radius-2 closing took that crop from 7 tips
  and 5 junctions to 4 and 2.

Both repairs are dangerous used plainly. A closing welds two vessels that pass within
a few voxels, which makes a false loop or merges two trees; a slice fill fills the
interior of a vessel loop that lies in the slice plane, which turns tissue into lumen.
So every candidate region is tested before it is added, and is added only if it
passes:

**Guard 1 -- local topology, for every region.** In the region's bounding box plus
`TOPOLOGY_MARGIN` voxels, over the foreground of every label: ``db0 == 0`` (the region
touches exactly one local piece, so it merges nothing and starts nothing) and
``db1 <= 0`` (it makes no new tunnel). A bridge between two vessels does one or the
other. Closing patches and pinholes must also leave b1 *unchanged*: removing a loop is
what filling a loop's interior or zipping a hairpin shut does, and a radius-2 closing
allowed to do it closed 15 of 126 tight-loop and hairpin phantoms. Only a fill region
that passed guard 2 may remove one. Regions are applied one at a time against the
mask *plus everything accepted so far*, so two regions cannot build a bridge between
them.

**Guard 2 -- for slice-fill regions,** which are the only regions that may remove a
loop, so are the ones that could fill a loop's interior. It is stopped by geometry:

* a cap on each 2D hole's area (`max_hole_um` as an equivalent radius);
* a region enclosed in 3D is a void and may be filled: a loop's interior never is,
  being open above and below the loop's plane;
* any other region must be a **hollow core**: running along the normals of the
  slices it is a hole in (`ALIGN_MIN`), inside a ring no thicker than itself
  (`BAND_MAX`). A core is a hole in slices *across* its vessel and its wall is thin
  for its lumen; a loop's or a hairpin's interior is a hole in slices *in* its own
  plane, and a tight loop of a thick vessel -- the one shape a short core can be
  confused with -- is ringed by a whole vessel. On 246 phantom regions no loop
  passes both, with margins of 0.31 in alignment and 0.28 in band ratio.

The raw image is not used to decide. It was meant to be -- missing lumen should be
darker than tissue enclosed by a loop -- but on LADAF-2021-17 the two overlap, and
around edge 7981 the tissue is darker than the vessel's own lumen, so a brightness
gate refused the very cores it was there to let through. Given a sampler, the raw
medians of each fill region and its surrounding tissue are recorded for review.

Every gate errs towards *not* filling: a refused region leaves a defect in place,
which is what the skeleton had before, while a wrongly accepted one invents anatomy.
Everything refused is reported with its reason, and the CLI can draw a gallery of
both kinds for inspection.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from numba import njit
from scipy import ndimage

CLOSE_RADIUS = 0  # fill only; see SKELETONISATION.md 6.3 for why the closing is opt-in
MAX_HOLE_UM = 1500.0
PINHOLE_VOXELS = 16
TOPOLOGY_MARGIN = 3
#: A closing patch may join voxels only where the mask already joins them by a path at
#: most DETOUR_STRETCH times as long, plus DETOUR_SLACK hops. A void's shell or a dent
#: is about 1.6x; two parallel branches zipped together from their junction are not.
DETOUR_STRETCH = 3.0
DETOUR_SLACK = 6
DETOUR_MARGIN = 6
TISSUE_RINGS = (6, 10, 15, 20)
MIN_TISSUE_SAMPLES = 100
SAMPLE_CAP = 5000
#: An open (not 3D-enclosed) fill region is accepted only as a hollow core: running
#: along its slices' normals, alignment >= ALIGN_MIN, inside a ring no thicker than
#: itself, band ratio <= BAND_MAX. Chosen on 246 phantom regions -- 88 loops and
#: hairpins of every size and tilt, 158 straight, oblique, short and curved cores --
#: as the pair that fills no loop and the most cores; see SKELETONISATION.md 6.3.
ALIGN_MIN = 0.35
BAND_MAX = 1.0
#: Or: walled in on both sides over this fraction of its footprint (see `_capped`).
CAP_MIN = 0.8
CAP_REACH = 3
#: Or: a sleeve around the vessel's own core over this fraction of it (see `_sleeve`).
SLEEVE_MIN = 0.5
ENCLOSED_MARGIN = 12
FILL_PASSES = 3
TILE = 64
THREADS = 8


# ------------------------------------------------------------------ topology


@njit(cache=True)
def _betti(f):
    """``(b0, b2, chi)`` of the 26-connected foreground `f` within this crop.

    chi is that of the union of closed unit cubes -- the 26-connected reading, with a
    6-connected background. b2 counts background components that do not touch the
    crop's faces: what is enclosed *here*. b1 is then ``b0 + b2 - chi``.
    """
    nz, ny, nx = f.shape
    chi = 0
    for i in range(nz + 1):
        for j in range(ny + 1):
            for k in range(nx + 1):
                hit = False
                for a in range(i - 1, i + 1):
                    for b in range(j - 1, j + 1):
                        for c in range(k - 1, k + 1):
                            if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx and f[a, b, c]:
                                hit = True
                if hit:
                    chi += 1
    for i in range(nz):  # edges along z
        for j in range(ny + 1):
            for k in range(nx + 1):
                hit = False
                for b in range(j - 1, j + 1):
                    for c in range(k - 1, k + 1):
                        if 0 <= b < ny and 0 <= c < nx and f[i, b, c]:
                            hit = True
                if hit:
                    chi -= 1
    for i in range(nz + 1):  # edges along y
        for j in range(ny):
            for k in range(nx + 1):
                hit = False
                for a in range(i - 1, i + 1):
                    for c in range(k - 1, k + 1):
                        if 0 <= a < nz and 0 <= c < nx and f[a, j, c]:
                            hit = True
                if hit:
                    chi -= 1
    for i in range(nz + 1):  # edges along x
        for j in range(ny + 1):
            for k in range(nx):
                hit = False
                for a in range(i - 1, i + 1):
                    for b in range(j - 1, j + 1):
                        if 0 <= a < nz and 0 <= b < ny and f[a, b, k]:
                            hit = True
                if hit:
                    chi -= 1
    for i in range(nz + 1):  # faces normal to z
        for j in range(ny):
            for k in range(nx):
                if (i > 0 and f[i - 1, j, k]) or (i < nz and f[i, j, k]):
                    chi += 1
    for i in range(nz):  # faces normal to y
        for j in range(ny + 1):
            for k in range(nx):
                if (j > 0 and f[i, j - 1, k]) or (j < ny and f[i, j, k]):
                    chi += 1
    for i in range(nz):  # faces normal to x
        for j in range(ny):
            for k in range(nx + 1):
                if (k > 0 and f[i, j, k - 1]) or (k < nx and f[i, j, k]):
                    chi += 1
    n = nz * ny * nx
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                if f[i, j, k]:
                    chi -= 1

    seen = np.zeros(n, dtype=np.bool_)
    stack = np.empty(n, dtype=np.int64)
    b0 = 0
    for s in range(n):
        si, sj, sk = s // (ny * nx), (s // nx) % ny, s % nx
        if seen[s] or not f[si, sj, sk]:
            continue
        b0 += 1
        seen[s] = True
        top = 0
        stack[0] = s
        top = 1
        while top:
            top -= 1
            v = stack[top]
            vi, vj, vk = v // (ny * nx), (v // nx) % ny, v % nx
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        a, b, c = vi + di, vj + dj, vk + dk
                        if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx and f[a, b, c]:
                            w = (a * ny + b) * nx + c
                            if not seen[w]:
                                seen[w] = True
                                stack[top] = w
                                top += 1
    b2 = 0
    for s in range(n):
        si, sj, sk = s // (ny * nx), (s // nx) % ny, s % nx
        if seen[s] or f[si, sj, sk]:
            continue
        seen[s] = True
        stack[0] = s
        top = 1
        border = False
        while top:
            top -= 1
            v = stack[top]
            vi, vj, vk = v // (ny * nx), (v // nx) % ny, v % nx
            if vi == 0 or vj == 0 or vk == 0 or vi == nz - 1 or vj == ny - 1 or vk == nx - 1:
                border = True
            for d in range(6):
                a, b, c = vi, vj, vk
                if d == 0:
                    a -= 1
                elif d == 1:
                    a += 1
                elif d == 2:
                    b -= 1
                elif d == 3:
                    b += 1
                elif d == 4:
                    c -= 1
                else:
                    c += 1
                if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx and not f[a, b, c]:
                    w = (a * ny + b) * nx + c
                    if not seen[w]:
                        seen[w] = True
                        stack[top] = w
                        top += 1
        if not border:
            b2 += 1
    return b0, b2, chi


@njit(cache=True)
def _bfs(fg, si, sj, sk, dist):
    nz, ny, nx = fg.shape
    n = nz * ny * nx
    for v in range(n):
        dist.ravel()[v] = -1
    queue = np.empty(n, dtype=np.int64)
    s = (si * ny + sj) * nx + sk
    dist[si, sj, sk] = 0
    queue[0] = s
    head, tail = 0, 1
    while head < tail:
        v = queue[head]
        head += 1
        vi, vj, vk = v // (ny * nx), (v // nx) % ny, v % nx
        d = dist[vi, vj, vk] + 1
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    a, b, c = vi + di, vj + dj, vk + dk
                    if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx and fg[a, b, c] \
                            and dist[a, b, c] < 0:
                        dist[a, b, c] = d
                        queue[tail] = (a * ny + b) * nx + c
                        tail += 1


@njit(cache=True)
def _detour(fg, region, stretch):
    """Largest ``path - stretch * straight`` between voxels `region` touches.

    Paths run through `fg` (26-connected hops); "straight" is the Chebyshev distance.
    A double sweep: from any contact voxel to the farthest one by path, then from
    there to all of them. Returns a large number if some contact is unreachable.
    """
    nz, ny, nx = fg.shape
    contacts = []
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                if not fg[i, j, k]:
                    continue
                touch = False
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            a, b, c = i + di, j + dj, k + dk
                            if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx and region[a, b, c]:
                                touch = True
                if touch:
                    contacts.append((i, j, k))
    if len(contacts) < 2:
        return 0.0
    dist = np.empty(fg.shape, dtype=np.int64)
    i0, j0, k0 = contacts[0]
    _bfs(fg, i0, j0, k0, dist)
    far, best = 0, -1
    for t in range(len(contacts)):
        i, j, k = contacts[t]
        if dist[i, j, k] < 0:
            return 1e9
        if dist[i, j, k] > best:
            best, far = dist[i, j, k], t
    i1, j1, k1 = contacts[far]
    _bfs(fg, i1, j1, k1, dist)
    worst = 0.0
    for t in range(len(contacts)):
        i, j, k = contacts[t]
        straight = max(abs(i - i1), abs(j - j1), abs(k - k1))
        excess = dist[i, j, k] - stretch * straight
        if excess > worst:
            worst = excess
    return worst


@njit(cache=True, nogil=True)
def _chi_owned(c):
    """chi contribution of the cells a tile owns; `c` is its voxels with a low halo.

    `c` covers voxels ``[lo - 1, hi)`` of the global grid (zero outside it) and the tile
    owns the vertices ``[lo, hi)``. Every vertex, edge, face and cube is named by its
    lowest corner, so each is counted by exactly one tile, and an element with no
    foreground voxel around it contributes nothing -- which is why empty tiles can be
    skipped. The closed-cube (26-connected) reading, as in :func:`_betti`.
    """
    tz, ty, tx = c.shape[0] - 1, c.shape[1] - 1, c.shape[2] - 1
    chi = 0
    for a in range(tz):
        for b in range(ty):
            for d in range(tx):
                v = (c[a, b, d] or c[a, b, d + 1] or c[a, b + 1, d] or c[a, b + 1, d + 1]
                     or c[a + 1, b, d] or c[a + 1, b, d + 1] or c[a + 1, b + 1, d]
                     or c[a + 1, b + 1, d + 1])
                if not v:
                    continue
                chi += 1
                if c[a + 1, b, d] or c[a + 1, b, d + 1] or c[a + 1, b + 1, d] \
                        or c[a + 1, b + 1, d + 1]:
                    chi -= 1  # edge along z
                if c[a, b + 1, d] or c[a, b + 1, d + 1] or c[a + 1, b + 1, d] \
                        or c[a + 1, b + 1, d + 1]:
                    chi -= 1  # edge along y
                if c[a, b, d + 1] or c[a, b + 1, d + 1] or c[a + 1, b, d + 1] \
                        or c[a + 1, b + 1, d + 1]:
                    chi -= 1  # edge along x
                if c[a, b + 1, d + 1] or c[a + 1, b + 1, d + 1]:
                    chi += 1  # face normal to z
                if c[a + 1, b, d + 1] or c[a + 1, b + 1, d + 1]:
                    chi += 1  # face normal to y
                if c[a + 1, b + 1, d] or c[a + 1, b + 1, d + 1]:
                    chi += 1  # face normal to x
                if c[a + 1, b + 1, d + 1]:
                    chi -= 1  # the cube
    return chi


def euler_volume(mask, *, tile=TILE, threads=THREADS):
    """chi of the 26-connected foreground of `mask`, however large, tile by tile.

    Exact: each cell is owned by one tile and an empty neighbourhood contributes
    nothing. Equal to ``skimage.measure.euler_number(mask, connectivity=3)``, which
    needs several box-sized temporaries a whole tree box cannot afford.
    """
    from concurrent.futures import ThreadPoolExecutor

    mask = np.asarray(mask, dtype=bool)
    n = np.array(mask.shape)
    occupied = set(_occupied_tiles(mask, tile))
    # A tile owning vertices [lo, hi) reads voxels [lo - 1, hi): its own tile, and the
    # tiles one step lower along any axis.
    todo = set()
    for z, y, x in occupied:
        for dz in (0, tile):
            for dy in (0, tile):
                for dx in (0, tile):
                    todo.add((z + dz, y + dy, x + dx))
    todo = [t for t in todo if all(t[i] <= n[i] for i in range(3))]

    def work(lo):
        lo = np.array(lo)
        hi = np.minimum(lo + tile, n + 1)
        src_lo, src_hi = np.maximum(lo - 1, 0), np.minimum(hi, n)
        crop = np.zeros(tuple(hi - lo + 1), dtype=np.bool_)
        if np.all(src_hi > src_lo):
            dst = src_lo - (lo - 1)
            crop[dst[0]:dst[0] + src_hi[0] - src_lo[0], dst[1]:dst[1] + src_hi[1] - src_lo[1],
                 dst[2]:dst[2] + src_hi[2] - src_lo[2]] = mask[
                src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
        return _chi_owned(crop)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        return int(sum(pool.map(work, todo)))


def betti_volume(mask, *, tile=TILE, threads=THREADS):
    """``(b0, b1, b2)`` of the 26-connected foreground of a whole tree, exactly.

    b0 from a 26-connected labelling, b2 from the background's 6-connected components
    inside a one-voxel pad (all but the outside are cavities), and b1 = b0 + b2 - chi.
    Crop `mask` to its foreground first: the labellings are box-sized.
    """
    import cc3d

    from .reconnect.segmentation import components

    mask = np.asarray(mask, dtype=bool)
    chi = euler_volume(mask, tile=tile, threads=threads)
    b0 = components(mask, connectivity=3).n
    bg = np.pad(~mask, 1, constant_values=True)
    _, nbg = cc3d.connected_components(bg, connectivity=6, return_N=True,
                                       max_labels=int(bg.sum()) + 1)
    del bg
    b2 = int(nbg) - 1
    return int(b0), int(b0 + b2 - chi), int(b2)


def topology_change(fg, region):
    """``(db0, db1, db2)`` from adding `region` to `fg`; both are crops of one box."""
    b0, b2, chi = _betti(np.ascontiguousarray(fg))
    a0, a2, achi = _betti(np.ascontiguousarray(fg | region))
    db0, db2 = a0 - b0, a2 - b2
    return int(db0), int(db0 + db2 - (achi - chi)), int(db2)


# ------------------------------------------------------------------ candidates


def _ball(r):
    g = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return g[0] ** 2 + g[1] ** 2 + g[2] ** 2 <= r * r


def _occupied_tiles(mask, tile=TILE):
    """Corners of the `tile`-cubes of `mask` holding any foreground."""
    nz, ny, nx = mask.shape
    out = []
    for z0 in range(0, nz, tile):
        plane = mask[z0:z0 + tile].any(axis=0)
        if not plane.any():
            continue
        gy, gx = -(-ny // tile), -(-nx // tile)
        padded = np.zeros((gy * tile, gx * tile), dtype=bool)
        padded[:ny, :nx] = plane
        cells = padded.reshape(gy, tile, gx, tile).any(axis=(1, 3))
        out.extend((z0, int(y) * tile, int(x) * tile) for y, x in np.argwhere(cells))
    return out


def closing_additions(own, blocked, radius, *, tile=TILE, threads=THREADS):
    """Voxels a radius-`radius` ball closing of `own` adds, outside `blocked`.

    Closing reaches ``2 * radius``, so it runs on the occupied tiles with a
    ``2 * radius + 1`` halo and is exact: outside `own`'s array is background, and a
    crop clipped there is zero-padded to match.
    """
    from concurrent.futures import ThreadPoolExecutor

    out = np.zeros(own.shape, dtype=bool)
    if radius < 1:
        return out
    ball = _ball(radius)
    h = 2 * radius + 1
    shape = np.array(own.shape)

    def work(corner):
        lo = np.array(corner)
        hi = np.minimum(lo + tile, shape)
        clo, chi = np.maximum(lo - h, 0), np.minimum(hi + h, shape)
        crop = own[clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]]
        pad = [(int(h - (lo[i] - clo[i])), int(h - (chi[i] - hi[i]))) for i in range(3)]
        crop = np.pad(crop, pad)
        closed = ndimage.binary_closing(crop, structure=ball)
        inner = closed[h:h + hi[0] - lo[0], h:h + hi[1] - lo[1], h:h + hi[2] - lo[2]]
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        out[sl] = inner & ~own[sl] & ~blocked[sl]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(work, _occupied_tiles(own, tile)))
    return out


def slice_holes(own, blocked, max_px, *, threads=THREADS):
    """Voxels enclosed by `own` in a slice along any axis, in holes of ``<= max_px``.

    Returns a uint8 bit mask: bit ``a`` is set where the voxel is a hole in a slice
    normal to axis ``a`` (0 = z, 1 = y, 2 = x), which is what tells a vessel's hollow
    core from a loop's interior (see :func:`_alignment`).

    Each slice is cropped to its own foreground's box first: a hole enclosed by `own`
    never reaches that box's edge. Hole components are 4-connected, matching the
    cross-shaped background `binary_fill_holes` uses.
    """
    from concurrent.futures import ThreadPoolExecutor

    out = np.zeros(own.shape, dtype=np.uint8)
    for axis in range(3):
        mv = np.moveaxis(own, axis, 0)
        bv = np.moveaxis(blocked, axis, 0)
        ov = np.moveaxis(out, axis, 0)
        present = np.flatnonzero(mv.any(axis=(1, 2)))

        def work(k, mv=mv, bv=bv, ov=ov, bit=axis):
            m = mv[k]
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            mc = np.ascontiguousarray(m[r0:r1, c0:c1])
            h = ndimage.binary_fill_holes(mc) & ~mc & ~bv[k][r0:r1, c0:c1]
            if not h.any():
                return
            lab, n = ndimage.label(h)
            keep = np.bincount(lab.ravel()) <= max_px
            keep[0] = False
            ov[k][r0:r1, c0:c1] |= keep[lab].astype(np.uint8) << bit

        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(work, present))
    return out


def _label(mask):
    """26-connected labels, voxel counts and bounding boxes per label, label order.

    Through `segmentation.components`, which works around cc3d refusing uint32 on a
    whole tree box: it sizes the dtype from its own provisional-label estimate, 6.1 G
    on the LADAF-2021-17 left tree, not from the voxel count.
    """
    import cc3d

    from .reconnect.segmentation import components

    labels = components(mask, connectivity=3).labels.view(np.uint32)
    stats = cc3d.statistics(labels)
    counts = np.asarray(stats["voxel_counts"])
    boxes = stats["bounding_boxes"]
    return labels, counts, boxes


# ------------------------------------------------------------------ raw sampling


class StackSampler:
    """Raw values at segmentation voxels, through the frame's seg -> raw mapping.

    Each needed raw slice is decoded once per call, across threads. The voxel's raw
    index is the centre of its bin.
    """

    def __init__(self, stack, frame, threads=THREADS):
        self.stack, self.frame, self.threads = stack, frame, threads

    def raw_index(self, zyx):
        b = np.asarray(self.frame.bin_factor, dtype=np.int64)[::-1]
        s = np.asarray(self.frame.raw_start, dtype=np.int64)[::-1]
        return s + np.asarray(zyx, dtype=np.int64) * b + (b - 1) // 2

    def _plane(self, z):
        return self.stack._read_image(self.stack.files[int(z)])

    def __call__(self, zyx):
        from concurrent.futures import ThreadPoolExecutor

        r = self.raw_index(zyx)
        n = (self.stack.n_slices, self.stack.n_rows, self.stack.n_cols)
        bad = np.any((r < 0) | (r >= n), axis=1)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} segmentation voxel(s) map outside the raw stack "
                f"{n[::-1]} (cols, rows, slices): check --raw against --seg and --voxel-um")
        out = np.empty(len(r), dtype=np.float64)
        order = np.argsort(r[:, 0], kind="stable")
        zs = r[order, 0]
        cuts = np.flatnonzero(np.diff(zs)) + 1
        groups = np.split(order, cuts) if len(order) else []

        def work(idx):
            plane = self._plane(r[idx[0], 0])
            out[idx] = plane[r[idx, 1], r[idx, 2]]

        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            list(pool.map(work, groups))
        return out

    def window(self, z, rows, cols):
        """A raw window at segmentation slice `z` over segmentation rows/cols."""
        lo = self.raw_index([[z, rows[0], cols[0]]])[0]
        hi = self.raw_index([[z, rows[1] - 1, cols[1] - 1]])[0] + 1
        return self.stack.read_window(int(lo[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2]))

    def box(self, origin_kji, shape_kji) -> np.ndarray:
        """float32 raw values on a block of the *segmentation* grid, NaN off the stack.

        One value per segmentation voxel, taken at the centre of its bin -- the same
        pixel :meth:`raw_index` picks. NaN rather than ``read_window``'s zero pad:
        zero is darker than any lumen, so a padded edge would read as a vessel.
        """
        from concurrent.futures import ThreadPoolExecutor

        k0, j0, i0 = (int(v) for v in origin_kji)
        nk, nj, ni = (int(v) for v in shape_kji)
        out = np.full((nk, nj, ni), np.nan, dtype=np.float32)
        b = np.asarray(self.frame.bin_factor, dtype=np.int64)[::-1]  # (z, y, x)
        lo = self.raw_index([[k0, j0, i0]])[0]
        rz = lo[0] + np.arange(nk) * b[0]
        rr = lo[1] + np.arange(nj) * b[1]
        rc = lo[2] + np.arange(ni) * b[2]
        row_ok = (rr >= 0) & (rr < self.stack.n_rows)
        col_ok = (rc >= 0) & (rc < self.stack.n_cols)
        if not row_ok.any() or not col_ok.any():
            return out
        r0, r1 = int(rr[row_ok].min()), int(rr[row_ok].max()) + 1
        c0, c1 = int(rc[col_ok].min()), int(rc[col_ok].max()) + 1

        def work(p):
            z = int(rz[p])
            if not 0 <= z < self.stack.n_slices:
                return
            win = self.stack.read_window(z, r0, r1, c0, c1)
            out[p][np.ix_(row_ok, col_ok)] = win[rr[row_ok] - r0][:, rc[col_ok] - c0]

        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            list(pool.map(work, range(nk)))
        return out


class ArraySampler:
    """`StackSampler`'s interface over an in-memory raw volume on the segmentation grid."""

    def __init__(self, raw):
        self.raw = np.asarray(raw)

    def __call__(self, zyx):
        zyx = np.asarray(zyx, dtype=np.int64)
        return self.raw[zyx[:, 0], zyx[:, 1], zyx[:, 2]].astype(np.float64)

    def window(self, z, rows, cols):
        return self.raw[z, rows[0]:rows[1], cols[0]:cols[1]]

    def box(self, origin_kji, shape_kji) -> np.ndarray:
        """:meth:`StackSampler.box` over the in-memory volume, NaN outside it."""
        out = np.full(tuple(int(v) for v in shape_kji), np.nan, dtype=np.float32)
        src = tuple(slice(max(int(o), 0), min(int(o) + int(n), int(s)))
                    for o, n, s in zip(origin_kji, shape_kji, self.raw.shape))
        dst = tuple(slice(s.start - int(o), s.stop - int(o)) for s, o in zip(src, origin_kji))
        if all(s.stop > s.start for s in src):
            out[dst] = self.raw[src]
        return out


# ------------------------------------------------------------------ report


@dataclass
class Region:
    kind: str  # "close" or "fill"
    tree: int
    voxels: int
    lo: tuple  # global (z, y, x), inclusive
    hi: tuple  # global (z, y, x), exclusive
    accepted: bool = False
    reason: str = ""
    db0: int | None = None
    db1: int | None = None
    db2: int | None = None
    detour: float | None = None  # closing patches: see DETOUR_STRETCH
    flat: float | None = None  # sqrt(l3 / l2)
    round: float | None = None  # sqrt(l2 / l1)
    region_median: float | None = None
    enclosed: bool | None = None
    alignment: float | None = None
    band: float | None = None
    capped: float | None = None
    sleeve: float | None = None
    tissue_median: float | None = None
    tissue_ring: int | None = None


@dataclass
class RepairReport:
    regions: list = field(default_factory=list)
    trees: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    seconds: float = 0.0

    def count(self, kind=None, accepted=None, reason=None):
        return sum(1 for r in self.regions
                   if (kind is None or r.kind == kind)
                   and (accepted is None or r.accepted == accepted)
                   and (reason is None or r.reason.startswith(reason)))

    def added(self, kind=None):
        return sum(r.voxels for r in self.regions
                   if r.accepted and (kind is None or r.kind == kind))

    def reasons(self):
        out: dict = {}
        for r in self.regions:
            key = (r.kind, "accepted" if r.accepted else r.reason.split(":")[0])
            n, v = out.get(key, (0, 0))
            out[key] = (n + 1, v + r.voxels)
        return out

    def describe(self) -> str:
        lines = [f"guarded mask repair ({self.seconds:.0f}s): "
                 f"{self.added():,} voxels added -- closing {self.added('close'):,}, "
                 f"slice fill {self.added('fill'):,}"]
        for t in self.trees:
            lines.append(f"  tree {t['tree']} [{t['material']}]: +{t['added']:,} voxels "
                         f"({100 * t['added'] / max(t['voxels'], 1):.2f}% of {t['voxels']:,})")
        for (kind, why), (n, v) in sorted(self.reasons().items()):
            lines.append(f"  {kind:5s} {why:20s} {n:>9,} region(s) {v:>12,} voxels")
        net = sum(r.db1 for r in self.regions if r.accepted and r.db1 is not None)
        lines.append(f"  accepted regions: net db1 {net:+d}, any db0 != 0: "
                     f"{any(r.db0 for r in self.regions if r.accepted)}")
        return "\n".join(lines)

    def as_dict(self):
        return dict(settings=self.settings, seconds=self.seconds, trees=self.trees,
                    summary={f"{k}/{w}": dict(regions=n, voxels=v)
                             for (k, w), (n, v) in self.reasons().items()},
                    regions=[asdict(r) for r in self.regions])


# ------------------------------------------------------------------ the repair


def _shape(coords):
    """``(flat, round)``: sqrt(l3/l2) and sqrt(l2/l1) of the coordinate covariance."""
    if len(coords) < 4:
        return 1.0, 1.0
    ev = np.sort(np.linalg.eigvalsh(np.cov(coords.T.astype(np.float64))))[::-1]
    ev = np.maximum(ev, 1e-12)
    return float(np.sqrt(ev[2] / ev[1])), float(np.sqrt(ev[1] / ev[0]))


def _tissue(fg, region, lo, hi, shape, rng):
    """Non-mask voxels near `region` (global coords), the nearest ring with enough."""
    reach = TISSUE_RINGS[-1]
    clo, chi = np.maximum(np.asarray(lo) - reach, 0), np.minimum(np.asarray(hi) + reach, shape)
    sl = tuple(slice(a, b) for a, b in zip(clo, chi))
    r = region[sl]
    dist = ndimage.distance_transform_edt(~r)
    free = ~fg[sl] & ~r
    for d in TISSUE_RINGS:
        pick = np.argwhere(free & (dist <= d))
        if len(pick) >= MIN_TISSUE_SAMPLES:
            if len(pick) > SAMPLE_CAP:
                pick = pick[rng.choice(len(pick), SAMPLE_CAP, replace=False)]
            return pick + clo, d
    return None, None


def _alignment(coords, bits):
    """How far a region runs along the normals of the slices it is a hole in.

    A vessel's hollow core is a hole in slices *across* the vessel and runs along the
    vessel, so its long axis is near those slices' normal: 0.58-1.0 on phantoms and on
    LADAF-2021-17. A loop's or a hairpin's interior is a hole only in slices *in* its
    plane and its long axis lies in that plane: 0.00-0.30 over loops of radius 6-20
    voxels tilted up to 40 degrees and hairpins rotated up to 45.
    """
    if len(coords) < 4:
        return 0.0
    ev, vec = np.linalg.eigh(np.cov(coords.T.astype(np.float64)))
    e1 = vec[:, np.argmax(ev)]
    w = np.array([np.count_nonzero(bits & (1 << a)) for a in range(3)], dtype=np.float64)
    return float((w * np.abs(e1)).sum() / max(w.sum(), 1.0))


def _band_ratio(fg, region, bits):
    """How thick the ring around each 2D hole is, relative to the hole: voxel-weighted.

    In each slice a voxel is a hole in, the ring is the mask touching the hole; its
    thickness is the median distance, across mask-plus-hole, from those ring voxels to
    the outside; the hole's size is its largest inscribed radius. A hollow vessel's
    wall is thin for its lumen (0.2-0.7 on phantoms and on edge 9321); a loop's ring is
    a whole vessel, and a tight loop of a thick vessel -- whose hole is a short, fat
    pipe, indistinguishable from a core by shape -- scores 1.28-4.3.
    `fg`, `region` and `bits` are crops of one box with room around the region.
    """
    num = den = 0.0
    for axis in range(3):
        m = np.moveaxis(fg, axis, 0)
        r = np.moveaxis(region & ((bits & (1 << axis)) > 0), axis, 0)
        for k in np.flatnonzero(r.any(axis=(1, 2))):
            h = r[k]
            ring = ndimage.binary_dilation(h, np.ones((3, 3), dtype=bool)) & m[k]
            if not ring.any():
                continue
            band = float(np.median(ndimage.distance_transform_edt(m[k] | h)[ring]))
            size = float(ndimage.distance_transform_edt(h).max())
            n = int(h.sum())
            num += band / max(size, 1.0) * n
            den += n
    return num / den if den else float("inf")


def fill_gate(fg, region, bits, rlo, rhi):
    """Gate 2 for one slice-fill region: a dict of measures, with ``reason`` if refused.

    `region` and `bits` are crops at ``[rlo, rhi)`` of `fg`'s grid. A region enclosed
    in 3D is a void -- a loop's interior never is, it is open above and below the
    loop's plane -- and needs no shape test. Any other region must be a hollow core:
    running along its slices' normals (:func:`_alignment`) and inside a ring thinner
    than itself (:func:`_band_ratio`). Each test alone admits some loop; together none
    of 88 loop and hairpin phantoms passes, with a margin of 0.31 in alignment and 0.28
    in band ratio.
    """
    coords = np.argwhere(region)
    flat, rnd = _shape(coords)
    g = dict(flat=flat, round=rnd)
    m = ENCLOSED_MARGIN
    clo = np.maximum(rlo - m, 0)
    chi = np.minimum(rhi + m, np.array(fg.shape))
    crop = fg[tuple(slice(a, b) for a, b in zip(clo, chi))]
    inner = tuple(slice(int(a - c), int(b - c)) for a, b, c in zip(rlo, rhi, clo))
    enclosed = ndimage.binary_fill_holes(crop)[inner][region].all()
    g["enclosed"] = bool(enclosed)
    if enclosed:
        return g
    full_region = np.zeros(crop.shape, dtype=bool)
    full_region[inner] = region
    full_bits = np.zeros(crop.shape, dtype=np.uint8)
    full_bits[inner] = np.where(region, bits, 0)
    g["capped"] = _capped(crop, full_region, full_bits)
    if g["capped"] >= CAP_MIN:
        # Inside one vessel's walls: its lumen or wall lies just beyond it on both
        # sides. A loop's or a hairpin's interior is a hole straight through.
        return g
    g["sleeve"] = _sleeve(crop, full_region, full_bits)
    if g["sleeve"] >= SLEEVE_MIN:
        # An unlabelled vessel wall between a labelled outer band and the labelled
        # lumen: the hole has the vessel's own core inside it. A loop's interior is
        # empty tissue; a separate vessel threading one is refused by guard 1.
        return g
    g["alignment"] = _alignment(coords, bits[region])
    if g["alignment"] < ALIGN_MIN:
        g["reason"] = (f"shape: lies in its slices' plane, as a loop's interior does "
                       f"(alignment {g['alignment']:.2f} < {ALIGN_MIN}, "
                       f"capped {g['capped']:.2f} < {CAP_MIN})")
        return g
    g["band"] = _band_ratio(crop, full_region, full_bits)
    if g["band"] > BAND_MAX:
        g["reason"] = (f"ring: as thick as the hole, as a loop's vessel is "
                       f"(band {g['band']:.2f} > {BAND_MAX}, capped {g['capped']:.2f})")
    return g


def _sleeve(fg, region, bits):
    """Area-weighted fraction of a region's 2D holes that have mask inside them.

    In each slice a voxel is a hole in, the hole's own holes are filled; if that
    covers mask, the hole is a sleeve around a core, as an unlabelled vessel wall is
    around its labelled lumen (LADAF-2021-17's densest loop block: 362 loops, from
    bridges across such a wall). A loop's or a hairpin's interior holds no mask.
    """
    num = den = 0
    for axis in range(3):
        m = np.moveaxis(fg, axis, 0)
        r = np.moveaxis(region & ((bits & (1 << axis)) > 0), axis, 0)
        for k in np.flatnonzero(r.any(axis=(1, 2))):
            lab, n = ndimage.label(r[k])
            for i in range(1, n + 1):
                h = lab == i
                inner = ndimage.binary_fill_holes(h) & ~h
                a = int(h.sum())
                den += a
                if (inner & m[k]).any():
                    num += a
    return num / den if den else 0.0


def _capped(fg, region, bits, reach=CAP_REACH):
    """Fraction of a region's footprint walled in by mask on *both* sides.

    Along the axis the region is most often a hole in, each footprint column is
    checked for mask within `reach` voxels beyond the region at either end. A
    collapsed hollow vessel lying in the slice plane is walled above and below by its
    own wall (0.95-1.0 on phantoms), and a short hollow section across a vessel by its
    lumen either side; the interior of a loop or a hairpin is open straight through
    (0.00 on all 88 loop and hairpin phantoms). `fg`, `region`, `bits` share one crop.
    """
    w = [np.count_nonzero(bits[region] & (1 << a)) for a in range(3)]
    a = int(np.argmax(w))
    m = np.moveaxis(fg, a, 0)
    r = np.moveaxis(region, a, 0)
    foot = r.any(axis=0)
    if not foot.any():
        return 0.0
    n = r.shape[0]
    first = np.argmax(r, axis=0)
    last = n - 1 - np.argmax(r[::-1], axis=0)
    yy, xx = np.nonzero(foot)
    f, lst = first[yy, xx], last[yy, xx]
    below = np.zeros(len(yy), dtype=bool)
    above = np.zeros(len(yy), dtype=bool)
    for d in range(1, reach + 1):
        lo, hi = f - d, lst + d
        ok = lo >= 0
        below[ok] |= m[lo[ok], yy[ok], xx[ok]]
        ok = hi < n
        above[ok] |= m[hi[ok], yy[ok], xx[ok]]
    return float((below & above).mean())


def _intensity_notes(sampler, fg, labels, items, gated, lo, rng):
    """Raw medians of each accepted fill region and of the tissue around it.

    Recorded for review, not used to decide: on LADAF-2021-17 the two overlap, and
    around edge 7981 the tissue is darker than the vessel's own lumen.
    """
    batches = []
    for lab, rlo, rhi, coords in items:
        reach = TISSUE_RINGS[-1]
        clo = np.maximum(rlo - reach, 0)
        chi = np.minimum(rhi + reach, np.array(fg.shape))
        csl = tuple(slice(a, b) for a, b in zip(clo, chi))
        tissue, ring = _tissue(fg[csl], labels[csl] == lab, rlo - clo, rhi - clo,
                               np.array(chi - clo), rng)
        if tissue is None:
            continue
        if len(coords) > SAMPLE_CAP:
            coords = coords[rng.choice(len(coords), SAMPLE_CAP, replace=False)]
        batches.append((lab, coords + rlo + lo, tissue + clo + lo, ring))
    if not batches:
        return
    values = sampler(np.concatenate([np.concatenate([a, b]) for _, a, b, _ in batches]))
    at = 0
    for lab, reg_pts, tis_pts, ring in batches:
        rv = values[at:at + len(reg_pts)]
        at += len(reg_pts)
        tv = values[at:at + len(tis_pts)]
        at += len(tis_pts)
        gated[lab].update(region_median=float(np.median(rv)),
                          tissue_median=float(np.median(tv)), tissue_ring=int(ring))


def repair_tree(volume, part, *, sampler=None, close_radius=CLOSE_RADIUS, max_hole_px,
                tree=0, verbose=False, threads=THREADS):
    """Repair one component of `volume` in place; returns its `Region` list.

    `volume` is the whole label lattice (z, y, x); `part` a ``components.TreePart``.
    Accepted voxels take the label the component already has.
    """
    rng = np.random.default_rng(0)
    pad = 2 * close_radius + max(TOPOLOGY_MARGIN, DETOUR_MARGIN) + 2
    shape = np.array(volume.shape)
    o = np.array(part.origin_zyx)
    lo, hi = np.maximum(o - pad, 0), np.minimum(o + np.array(part.volume.shape) + pad, shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    view = volume[sl]
    own = np.zeros(view.shape, dtype=bool)
    inner = tuple(slice(int(o[i] - lo[i]), int(o[i] - lo[i] + part.volume.shape[i])) for i in range(3))
    own[inner] = part.volume
    value = int(view[own][0])
    fg = view > 0  # every label: another tree is never claimed and never bridged to
    regions: list[Region] = []
    t0 = time.time()

    def accept_in_order(cands, labels, kind, gated):
        for lab, cnt, box in cands:
            zs, ys, xs = box
            rlo = np.array([zs.start, ys.start, xs.start])
            rhi = np.array([zs.stop, ys.stop, xs.stop])
            reg = Region(kind, tree, int(cnt), tuple(int(v) for v in rlo + lo),
                         tuple(int(v) for v in rhi + lo))
            if gated is not None and lab in gated:
                reg.__dict__.update(gated[lab])
                if reg.reason:
                    regions.append(reg)
                    continue
            clo = np.maximum(rlo - TOPOLOGY_MARGIN, 0)
            chi = np.minimum(rhi + TOPOLOGY_MARGIN, np.array(fg.shape))
            csl = tuple(slice(a, b) for a, b in zip(clo, chi))
            r = labels[csl] == lab
            reg.db0, reg.db1, reg.db2 = topology_change(fg[csl], r)
            if kind == "close" and reg.db0 == 0 and reg.db1 <= 0:
                dlo = np.maximum(rlo - DETOUR_MARGIN, 0)
                dhi = np.minimum(rhi + DETOUR_MARGIN, np.array(fg.shape))
                dsl = tuple(slice(a, b) for a, b in zip(dlo, dhi))
                reg.detour = float(_detour(np.ascontiguousarray(fg[dsl]),
                                           np.ascontiguousarray(labels[dsl] == lab),
                                           DETOUR_STRETCH))
            if reg.db0 != 0:
                reg.reason = f"topology: db0 {reg.db0:+d}"
            elif reg.db1 > 0:
                reg.reason = f"topology: db1 {reg.db1:+d}"
            elif reg.db1 < 0 and (kind == "close" or cnt <= PINHOLE_VOXELS):
                # Removing a loop is what filling a loop's interior or welding two
                # parallel vessels does. Only a fill region that passed the core gate
                # (`fill_gate`) has shown it is not that: on phantoms a radius-2
                # closing allowed db1 < 0 closed 15 of 126 tight loops and hairpins.
                reg.reason = f"topology: removes a loop (db1 {reg.db1:+d}) ungated"
            elif reg.detour is not None and reg.detour > DETOUR_SLACK:
                reg.reason = (f"detour: joins voxels {reg.detour:.0f} hops further apart "
                              f"through the mask than {DETOUR_STRETCH:g}x straight")
            else:
                reg.accepted = True
                fg[csl] |= r
                own[csl] |= r
                view[csl][r] = value
            regions.append(reg)

    # 1. closing patches
    if close_radius > 0:
        add = closing_additions(own, fg & ~own, close_radius, threads=threads)
        if add.any():
            labels, counts, boxes = _label(add)
            del add
            cands = [(i, int(counts[i]), boxes[i]) for i in range(1, len(counts))]
            accept_in_order(cands, labels, "close", None)
            del labels
        if verbose:
            n = sum(1 for r in regions if r.kind == "close")
            print(f"    closing: {n:,} patch(es), {sum(r.accepted for r in regions):,} accepted "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # 2. slice-fill regions, from the mask as the closing left it. Filling a hollow core
    # can seal small pockets between the fill and the wall, so the fill is repeated --
    # for enclosed voids only, which a loop's interior never is, so nothing refused
    # before is proposed again -- until a pass adds nothing.
    t1 = time.time()
    for fill_pass in range(FILL_PASSES):
        later = fill_pass > 0
        holes = slice_holes(own, fg & ~own, max_hole_px, threads=threads)
        if not holes.any():
            break
        labels, counts, boxes = _label(holes > 0)
        cands = [(i, int(counts[i]), boxes[i]) for i in range(1, len(counts))]
        gated: dict = {}
        diagnose = []
        keep = []
        for lab, cnt, box in cands:
            zs, ys, xs = box
            rlo = np.array([zs.start, ys.start, xs.start])
            rhi = np.array([zs.stop, ys.stop, xs.stop])
            sub = labels[box] == lab
            if cnt <= PINHOLE_VOXELS and not later:
                gated[lab] = {}
                keep.append((lab, cnt, box))
                continue
            g = fill_gate(fg, sub, holes[box], rlo, rhi)
            if later and not g.get("enclosed"):
                continue
            gated[lab] = g
            keep.append((lab, cnt, box))
            if sampler is not None and not g.get("reason"):
                diagnose.append((lab, rlo, rhi, np.argwhere(sub)))
        if diagnose:
            _intensity_notes(sampler, fg, labels, diagnose, gated, lo, rng)
        before = sum(r.accepted for r in regions)
        accept_in_order(keep, labels, "fill", gated)
        del labels
        if sum(r.accepted for r in regions) == before:
            break
    if verbose:
        fills = [r for r in regions if r.kind == "fill"]
        print(f"    slice fill: {len(fills):,} region(s), {sum(r.accepted for r in fills):,} "
              f"accepted ({time.time() - t1:.0f}s)", flush=True)
    return regions


def repair_volume(volume, frame, *, sampler=None, close_radius=CLOSE_RADIUS,
                  max_hole_um=MAX_HOLE_UM, materials=None,
                  min_component_voxels=0, verbose=False, threads=THREADS) -> RepairReport:
    """Guarded repair of every component of the label lattice `volume`, in place.

    `sampler`, optional, gives raw values at segmentation voxels
    (:class:`StackSampler`); with it each accepted fill region's raw median and its
    surrounding tissue's are recorded for review. Nothing is decided by them.
    """
    from . import components as comp

    t0 = time.time()
    voxel = float(np.mean(frame.seg_spacing))
    max_hole_px = int(np.pi * (max_hole_um / voxel) ** 2)
    report = RepairReport(settings=dict(close_radius=close_radius, max_hole_um=max_hole_um,
                                        max_hole_px=max_hole_px,
                                        pinhole_voxels=PINHOLE_VOXELS,
                                        topology_margin=TOPOLOGY_MARGIN,
                                        detour_stretch=DETOUR_STRETCH,
                                        detour_slack=DETOUR_SLACK,
                                        align_min=ALIGN_MIN, band_max=BAND_MAX,
                                        cap_min=CAP_MIN, cap_reach=CAP_REACH,
                                        sleeve_min=SLEEVE_MIN,
                                        enclosed_margin=ENCLOSED_MARGIN))
    parts, _stats = comp._split_components(volume, frame, min_voxels=min_component_voxels,
                                           materials=materials, verbose=verbose)
    for part in parts:
        if verbose:
            print(f"  repairing {part.describe()}", flush=True)
        regions = repair_tree(volume, part, sampler=sampler, close_radius=close_radius,
                              max_hole_px=max_hole_px,
                              tree=part.index, verbose=verbose, threads=threads)
        report.regions.extend(regions)
        report.trees.append(dict(tree=part.index, material=part.material or "-",
                                 voxels=int(part.voxels),
                                 added=int(sum(r.voxels for r in regions if r.accepted))))
    report.seconds = time.time() - t0
    return report


def audit_gallery(report, before, after, sampler, out_dir, *, per_group=24):
    """PNGs of the largest regions per (kind, verdict): raw with both mask outlines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups: dict = {}
    for r in report.regions:
        key = f"{r.kind}_{'accepted' if r.accepted else r.reason.split(':')[0]}"
        groups.setdefault(key, []).append(r)
    written = []
    for key, regs in groups.items():
        regs = sorted(regs, key=lambda r: -r.voxels)[:per_group]
        if not regs:
            continue
        cols = 4
        rows = -(-len(regs) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, r in zip(axes.ravel(), regs):
            z = (r.lo[0] + r.hi[0] - 1) // 2
            m = 25
            y0, y1 = max(r.lo[1] - m, 0), min(r.hi[1] + m, before.shape[1])
            x0, x1 = max(r.lo[2] - m, 0), min(r.hi[2] + m, before.shape[2])
            raw = np.asarray(sampler.window(z, (y0, y1), (x0, x1)), dtype=float)
            v = np.percentile(raw, [1, 99]) if raw.size else (0, 1)
            ax.imshow(raw, cmap="gray", vmin=v[0], vmax=max(v[1], v[0] + 1))
            b = before[z, y0:y1, x0:x1] > 0
            a = after[z, y0:y1, x0:x1] > 0
            if b.any():
                ax.contour(b, [0.5], colors="lime", linewidths=0.7)
            new = a & ~b
            box = np.zeros_like(new)  # this region's own box: neighbours' additions excluded
            box[max(r.lo[1] - y0, 0):r.hi[1] - y0, max(r.lo[2] - x0, 0):r.hi[2] - x0] = True
            new &= box
            if r.accepted and new.any():
                ax.contour(new, [0.5], colors="yellow", linewidths=0.9)
            elif not r.accepted:
                # A refused region was never written, so mark where it would have been.
                ax.add_patch(plt.Rectangle((r.lo[2] - x0 - 0.5, r.lo[1] - y0 - 0.5),
                                           r.hi[2] - r.lo[2], r.hi[1] - r.lo[1],
                                           fill=False, edgecolor="red", linewidth=0.9))
            ax.set_title(f"{r.kind} {r.voxels} vox @ z{z} y{r.lo[1]} x{r.lo[2]}\n"
                         + ("accepted" if r.accepted else r.reason)[:60], fontsize=7)
        fig.suptitle(f"{key}: green = mask before, yellow = added in this region's box, "
                     "red box = refused region",
                     fontsize=10)
        fig.tight_layout()
        path = out / f"{key}.png"
        fig.savefig(path, dpi=70)
        plt.close(fig)
        written.append(path)
    return written
