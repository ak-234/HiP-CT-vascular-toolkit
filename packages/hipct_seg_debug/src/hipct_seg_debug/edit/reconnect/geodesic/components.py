"""26-connected components of the segmentation, without decoding the segmentation.

``lattice.decode_volume`` materialises 2.34 GB and ``scipy.ndimage.label`` wants
another 9.4 GB of int32 on top of it. Both are affordable on this machine and
neither is affordable inside a GUI that is already holding the mask, the raw
stack and a render window -- and a reconnection needs the component *identity* of
a few hundred endpoints, not a labelled volume.

So the labelling is done on the run-length structure the mask is already stored
in. Each z plane decodes to a handful of thousand horizontal runs; a run is four
integers, and the whole tree is a few megabytes of run table rather than gigabytes
of voxels. Two planes are resident at a time.

Connectivity is 26, matching :func:`~..segmentation.components` and for the same
reason: a 6- or 18-connected labelling beads a diagonal capillary into fragments
and would invent gaps for this package to then go and repair.

The one thing this cannot do is answer "which component is at (z, y, x)" by array
indexing, so :meth:`ComponentIndex.label_at` binary-searches the run table for
that row. That is a few microseconds and it is asked once per endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Runs shorter than this are still indexed -- debris is *identified* here and
#: judged later, in `classify`, where the vessel radius is known.
_RUN_COLUMNS = ("y", "x0", "x1", "label")


class _UnionFind:
    """Union-find over run ids, with path halving.

    A list rather than an array: the run count is not known until the last plane
    is read, and appending is what the streaming pass does.
    """

    def __init__(self) -> None:
        self.parent: list[int] = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Smaller id wins, so labels come out in first-appearance order and a
            # rebuild on the same mask is reproducible.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _plane_runs(plane: np.ndarray) -> np.ndarray:
    """Horizontal runs of a ``(ny, nx)`` boolean plane as ``(n, 3)`` ``[y, x0, x1)``.

    A zero column is appended before flattening so a run can never straddle two
    rows -- the alternative is a per-row Python loop, and there are 1250 planes.
    """
    ny, nx = plane.shape
    padded = np.zeros((ny, nx + 1), dtype=np.int8)
    padded[:, :nx] = plane.astype(np.int8, copy=False)
    flat = padded.ravel()
    edges = np.diff(np.concatenate([[np.int8(0)], flat]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return np.empty((0, 3), dtype=np.int64)
    stride = nx + 1
    y = starts // stride
    return np.column_stack([y, starts - y * stride, ends - y * stride]).astype(np.int64)


def _row_buckets(runs: np.ndarray, ny: int) -> np.ndarray:
    """Start index into `runs` for each row, given `runs` sorted by ``(y, x0)``.

    Returns ``(ny + 1,)`` offsets, so row ``y`` occupies ``runs[off[y]:off[y+1]]``.
    """
    counts = np.bincount(runs[:, 0], minlength=ny) if len(runs) else np.zeros(ny, np.int64)
    offsets = np.zeros(ny + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _link_rows(a: np.ndarray, b: np.ndarray, ids_a, ids_b, uf: _UnionFind) -> None:
    """Union every pair of runs from two row-slices whose x-spans touch or overlap.

    Both are sorted by ``x0``, so this is a merge scan rather than a product. The
    ``+1`` slack is what makes the connectivity 8 in plane and 26 in space: runs
    that merely touch corners are one vessel.
    """
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i, 2] + 1 <= b[j, 1]:  # a ends before b starts (with slack)
            i += 1
        elif b[j, 2] + 1 <= a[i, 1]:
            j += 1
        else:
            uf.union(int(ids_a[i]), int(ids_b[j]))
            # Advance whichever finishes first; the other may still meet the next.
            if a[i, 2] <= b[j, 2]:
                i += 1
            else:
                j += 1


@dataclass
class ComponentIndex:
    """Every 26-connected component of the mask, as a run table.

    ``runs`` is ``(N, 4)`` -- ``[y, x0, x1, label]``, x1 exclusive -- and
    ``plane_offsets`` gives the slice of it belonging to each z, so a lookup never
    scans past its own plane. Labels are 1-based; 0 means background.
    """

    runs: np.ndarray
    plane_offsets: np.ndarray  # (nz + 1,)
    row_offsets: list[np.ndarray]  # per plane, (ny + 1,) into that plane's runs
    sizes: np.ndarray  # (n + 1,) voxel counts, index 0 unused
    boxes: np.ndarray  # (n + 1, 6) z0 z1 y0 y1 x0 x1, half-open
    shape: tuple[int, int, int]  # (nz, ny, nx)
    n: int

    def label_at(self, z: int, y: int, x: int) -> int:
        """Component label containing voxel ``(z, y, x)``; 0 if it is background."""
        nz, ny, nx = self.shape
        if not (0 <= z < nz and 0 <= y < ny and 0 <= x < nx):
            return 0
        base = int(self.plane_offsets[z])
        rows = self.row_offsets[z]
        lo, hi = base + int(rows[y]), base + int(rows[y + 1])
        if hi <= lo:
            return 0
        block = self.runs[lo:hi]
        k = int(np.searchsorted(block[:, 1], x, side="right")) - 1
        if k < 0 or x >= block[k, 2]:
            return 0
        return int(block[k, 3])

    def labels_at(self, zyx) -> np.ndarray:
        """Vectorised :meth:`label_at` over ``(N, 3)`` integer indices."""
        idx = np.asarray(zyx, dtype=np.int64).reshape(-1, 3)
        return np.array([self.label_at(*row) for row in idx], dtype=np.int64)

    def nearest_label(self, z: int, y: int, x: int, radius: int) -> tuple[int, float]:
        """The nearest non-background label within `radius` voxels, and its distance.

        An endpoint is a centreline vertex, and a centreline vertex is *usually*
        inside its own lumen -- but not always: a re-sampled graph, a half-voxel
        offset in the frame, or a genuinely one-voxel-thin collapsed tube all put
        it just outside. Falling straight through to "no component" there would
        classify a repairable break as unassociated and send it to review, which
        is the most expensive possible answer to a rounding error.
        """
        here = self.label_at(z, y, x)
        if here:
            return here, 0.0
        best, best_d = 0, float("inf")
        for r in range(1, int(radius) + 1):
            for dz in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if max(abs(dz), abs(dy), abs(dx)) != r:
                            continue  # only the new shell
                        label = self.label_at(z + dz, y + dy, x + dx)
                        if not label:
                            continue
                        d = float(np.sqrt(dz * dz + dy * dy + dx * dx))
                        if d < best_d:
                            best, best_d = label, d
            if best:
                return best, best_d
        return 0, float("inf")

    def window(self, lo_zyx, hi_zyx) -> np.ndarray:
        """Decode component labels into a box, as ``(dz, dy, dx)`` int32.

        This is how the pathfinder learns which foreground it is allowed to route
        through and which foreground is a *competing target*, and it is the only
        place a labelled array is ever materialised -- for one corridor, not for
        the volume.
        """
        lo = np.asarray(lo_zyx, dtype=np.int64)
        hi = np.asarray(hi_zyx, dtype=np.int64)
        nz, ny, nx = self.shape
        lo = np.clip(lo, 0, [nz, ny, nx])
        hi = np.clip(hi, lo, [nz, ny, nx])
        out = np.zeros(tuple(hi - lo), dtype=np.int32)
        if out.size == 0:
            return out

        for z in range(int(lo[0]), int(hi[0])):
            base = int(self.plane_offsets[z])
            rows = self.row_offsets[z]
            start, stop = base + int(rows[lo[1]]), base + int(rows[hi[1]])
            for y, x0, x1, label in self.runs[start:stop]:
                a, b = max(int(x0), int(lo[2])), min(int(x1), int(hi[2]))
                if b > a:
                    out[z - lo[0], y - lo[1], a - lo[2]:b - lo[2]] = label
        return out

    def voxels(self, label: int, limit: int | None = None) -> np.ndarray:
        """Every voxel of one component as ``(M, 3)`` ``(z, y, x)``.

        Bounded by `limit` because a caller asking for the aorta wants to know
        that it is the aorta, not to receive forty million rows.
        """
        box = self.boxes[label]
        out: list[np.ndarray] = []
        total = 0
        for z in range(int(box[0]), int(box[1])):
            base = int(self.plane_offsets[z])
            block = self.runs[base:int(self.plane_offsets[z + 1])]
            for y, x0, x1, lab in block[block[:, 3] == label]:
                xs = np.arange(x0, x1)
                out.append(np.column_stack([np.full(len(xs), z), np.full(len(xs), y), xs]))
                total += len(xs)
                if limit is not None and total >= limit:
                    return np.vstack(out)[:limit]
        return np.vstack(out) if out else np.empty((0, 3), dtype=np.int64)

    def order_by_size(self) -> np.ndarray:
        return np.argsort(self.sizes[1:])[::-1] + 1

    def describe(self) -> str:
        if not self.n:
            return "mask has no foreground"
        big = self.order_by_size()[:3]
        head = ", ".join(f"#{int(k)}={int(self.sizes[k]):,}" for k in big)
        return (f"{self.n:,} mask component(s), {int(self.sizes.sum()):,} voxels; "
                f"largest {head}")


def build(labels, *, progress=None, z_range: tuple[int, int] | None = None
          ) -> ComponentIndex:
    """Label the mask by streaming it one plane at a time.

    `labels` is anything with ``slice_z(k)`` and ``nz``/``ny``/``nx`` -- a
    :class:`~...rle.ByteRLELattice`, a :class:`~..maskedit.MaskSource` (so a
    painting session's corrections are already composited in), or a
    :class:`~..lattice.MaskVolume`.

    `z_range` restricts the pass to a slab. Components are then labelled *within
    that slab*, which is right for a bounded repair and wrong for a global report,
    so it is opt-in and recorded in the returned shape.
    """
    nz = int(getattr(labels, "nz", 0) or labels.dims[2])
    ny = int(getattr(labels, "ny", 0) or labels.dims[1])
    nx = int(getattr(labels, "nx", 0) or labels.dims[0])
    z0, z1 = (0, nz) if z_range is None else (max(0, z_range[0]), min(nz, z_range[1]))

    uf = _UnionFind()
    per_plane: list[np.ndarray] = []  # (n, 3) runs
    per_plane_ids: list[np.ndarray] = []
    prev_runs = np.empty((0, 3), dtype=np.int64)
    prev_ids = np.empty(0, dtype=np.int64)
    prev_rows = np.zeros(ny + 1, dtype=np.int64)

    for z in range(z0, z1):
        plane = np.asarray(labels.slice_z(z)) > 0
        runs = _plane_runs(plane)
        ids = np.array([uf.add() for _ in range(len(runs))], dtype=np.int64)
        rows = _row_buckets(runs, ny)

        # In plane: row y against row y + 1 (8-connected).
        for y in range(ny - 1):
            a0, a1 = int(rows[y]), int(rows[y + 1])
            b0, b1 = int(rows[y + 1]), int(rows[y + 2])
            if a1 > a0 and b1 > b0:
                _link_rows(runs[a0:a1], runs[b0:b1], ids[a0:a1], ids[b0:b1], uf)

        # Against the previous plane: rows y-1, y, y+1 (26-connected).
        if len(prev_runs):
            for y in range(ny):
                a0, a1 = int(rows[y]), int(rows[y + 1])
                if a1 <= a0:
                    continue
                for dy in (-1, 0, 1):
                    yy = y + dy
                    if not 0 <= yy < ny:
                        continue
                    b0, b1 = int(prev_rows[yy]), int(prev_rows[yy + 1])
                    if b1 > b0:
                        _link_rows(runs[a0:a1], prev_runs[b0:b1],
                                   ids[a0:a1], prev_ids[b0:b1], uf)

        per_plane.append(runs)
        per_plane_ids.append(ids)
        prev_runs, prev_ids, prev_rows = runs, ids, rows
        if progress is not None and (z - z0) % 50 == 0:
            progress(z - z0, z1 - z0)
    if progress is not None:
        progress(z1 - z0, z1 - z0)

    return _finalise(per_plane, per_plane_ids, uf, (nz, ny, nx), z0, z1)


def _finalise(per_plane, per_plane_ids, uf, shape, z0, z1) -> ComponentIndex:
    """Resolve the union-find, compact the labels, and measure every component."""
    nz, ny, nx = shape
    roots = np.array([uf.find(i) for i in range(len(uf.parent))], dtype=np.int64)
    unique, compact = np.unique(roots, return_inverse=True) if len(roots) else (
        np.empty(0, np.int64), np.empty(0, np.int64)
    )
    compact = compact + 1  # 0 stays background
    n = len(unique)

    plane_offsets = np.zeros(nz + 1, dtype=np.int64)
    row_offsets: list[np.ndarray] = [np.zeros(ny + 1, dtype=np.int64) for _ in range(nz)]
    blocks: list[np.ndarray] = []
    sizes = np.zeros(n + 1, dtype=np.int64)
    boxes = np.zeros((n + 1, 6), dtype=np.int64)
    boxes[:, 0::2] = np.iinfo(np.int64).max
    boxes[:, 1::2] = np.iinfo(np.int64).min

    total = 0
    for k, (runs, ids) in enumerate(zip(per_plane, per_plane_ids)):
        z = z0 + k
        plane_offsets[z] = total
        if len(runs):
            label = compact[ids]
            block = np.column_stack([runs, label])
            # `label_at` binary-searches within a row, so the table must be sorted
            # by (y, x0) -- which `_plane_runs` already produces, but a sort here
            # costs nothing and makes the invariant local to where it is relied on.
            block = block[np.lexsort((block[:, 1], block[:, 0]))]
            blocks.append(block)
            row_offsets[z] = _row_buckets(block, ny)
            widths = block[:, 2] - block[:, 1]
            np.add.at(sizes, block[:, 3], widths)
            for y, x_lo, x_hi, lab in block:
                box = boxes[lab]
                box[0], box[1] = min(box[0], z), max(box[1], z + 1)
                box[2], box[3] = min(box[2], y), max(box[3], y + 1)
                box[4], box[5] = min(box[4], x_lo), max(box[5], x_hi)
            total += len(block)
        plane_offsets[z + 1] = total
    # Planes outside the scanned range hold no runs but must still terminate the
    # offsets monotonically, or a lookup there would read another plane's table.
    for z in range(z1, nz + 1):
        plane_offsets[z] = total
    for z in range(0, z0):
        plane_offsets[z] = 0

    runs = np.vstack(blocks) if blocks else np.empty((0, 4), dtype=np.int64)
    boxes[0] = 0
    return ComponentIndex(
        runs=runs, plane_offsets=plane_offsets, row_offsets=row_offsets,
        sizes=sizes, boxes=boxes, shape=(nz, ny, nx), n=n,
    )


def from_array(mask: np.ndarray) -> ComponentIndex:
    """Build an index from an in-memory ``(nz, ny, nx)`` array.

    For tests and for a caller that already holds the volume; the streaming
    :func:`build` is what runs on the real lattice.
    """
    array = np.asarray(mask)

    class _Adapter:
        nz, ny, nx = array.shape

        @staticmethod
        def slice_z(k):
            return array[k]

    return build(_Adapter())
