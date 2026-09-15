"""Turn the RLE lattice into a real array, and into something ``skeleton_analysis`` accepts.

Two obstacles sit between this package's segmentation and the analysis tools:

* ``rle.ByteRLELattice`` only ever hands out one slice at a time. That is exactly
  right for a pick, and exactly wrong for skeletonisation, which is inherently
  global. (It is no longer true that 2.34 GB never has to be resident: the viewer
  holds it on request through :class:`MaskVolume` at the bottom of this module.)
* everything in ``skeleton_analysis.optimisation`` expects an ``AmiraLattice``: a
  materialised ``(nz, ny, nx)`` array with ``world_to_index_zyx``. It has its own
  independent HxByteRLE decoder, so pointing it at the same file would decode the
  volume a second time and, worse, describe it with a slightly different geometry.

So: decode once, here, and wrap the result in the shape those functions want.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def decode_volume(labels, stride: int = 1, dtype=np.uint8, progress=None) -> np.ndarray:
    """Materialise the whole lattice as ``(nz, ny, nx)``.

    ``stride`` decimates every axis: stride 2 is 293 MB, stride 4 is 37 MB, against
    2.34 GB undecimated. At stride 1 on LADAF-2024-28 this is 1250 decodes and takes
    2.0 s end to end -- ~1.6 ms a plane, most of which is the copy rather than the
    decompression.

    Note the asymmetry: ``z`` genuinely skips decodes, but ``slice_z`` has no
    windowed form, so the in-plane ``[::s, ::s]`` throws away part of a plane that
    was decoded in full. Time therefore scales as 1/s and memory as 1/s**3.

    Preallocated and filled slice by slice rather than ``np.stack``-ed, because a
    stack would hold the list *and* the result at once -- 4.7 GB peak for a 2.34 GB
    array.
    """
    s = max(int(stride), 1)
    zs = range(0, labels.nz, s)
    out = np.empty((len(zs), (labels.ny + s - 1) // s, (labels.nx + s - 1) // s), dtype=dtype)
    t0 = time.time()
    for n, z in enumerate(zs):
        out[n] = labels.slice_z(z)[::s, ::s]
        if progress is not None and n % 100 == 0:
            progress(n, len(zs), time.time() - t0)
    if progress is not None:
        progress(len(zs), len(zs), time.time() - t0)
    return out


def decode_foreground_crop(
    labels, stride: int = 1, padding: int = 1, dtype=np.uint8, progress=None
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Decode only the foreground bounding box and return its sampled-grid origin.

    The first pass finds the non-zero extent one z-plane at a time; the second reads
    only that extent into a dense array.  This is intentionally a two-pass operation:
    a full-resolution HiP-CT lattice can be tens of GB, while its coronary foreground
    occupies a small fraction of the scan.  ``padding`` is in sampled voxels and keeps
    a zero-valued border around foreground wherever the source has one.

    Returns ``(crop, (k0, j0, i0))``, where the origin is in the volume produced by
    ``decode_volume(labels, stride)``.  Therefore source voxel zero for the crop is
    ``(k0, j0, i0) * stride``.
    """
    s = max(int(stride), 1)
    sampled_shape = np.array(
        [
            (labels.nz + s - 1) // s,
            (labels.ny + s - 1) // s,
            (labels.nx + s - 1) // s,
        ],
        dtype=np.int64,
    )
    lo = sampled_shape.copy()
    hi = np.full(3, -1, dtype=np.int64)
    zs = range(0, labels.nz, s)
    total = len(zs)
    t0 = time.time()

    for k, z in enumerate(zs):
        reader = getattr(labels, "slice_window", None)
        if reader is None:
            plane = labels.slice_z(z)[::s, ::s]
        else:
            plane = reader(z, 0, labels.ny, 0, labels.nx, step=s)
        rows, cols = np.nonzero(plane)
        if rows.size:
            lo = np.minimum(lo, (k, int(rows.min()), int(cols.min())))
            hi = np.maximum(hi, (k, int(rows.max()), int(cols.max())))
        if progress is not None and k % 100 == 0:
            progress("scan", k, total, time.time() - t0)

    if hi[0] < 0:
        if progress is not None:
            progress("scan", total, total, time.time() - t0)
        return np.zeros((1, 1, 1), dtype=dtype), (0, 0, 0)

    pad = max(int(padding), 0)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad + 1, sampled_shape)  # exclusive
    shape = tuple(int(v) for v in hi - lo)
    out = np.empty(shape, dtype=dtype)

    for out_k, sample_k in enumerate(range(int(lo[0]), int(hi[0]))):
        z = sample_k * s
        reader = getattr(labels, "slice_window", None)
        if reader is None:
            plane = labels.slice_z(z)
            window = plane[
                int(lo[1]) * s : int(hi[1]) * s : s,
                int(lo[2]) * s : int(hi[2]) * s : s,
            ]
        else:
            window = reader(
                z,
                int(lo[1]) * s,
                int(hi[1]) * s,
                int(lo[2]) * s,
                int(hi[2]) * s,
                step=s,
            )
        out[out_k] = window
        if progress is not None and out_k % 100 == 0:
            progress("decode", out_k, shape[0], time.time() - t0)

    if progress is not None:
        progress("decode", shape[0], shape[0], time.time() - t0)
    return out, tuple(int(v) for v in lo)


@dataclass
class LatticeView:
    """An ``AmiraLattice``-shaped view of a decoded volume.

    Only the members ``skeleton_analysis.optimisation`` actually touches are
    provided: ``volume``, ``dims``, ``bbox``, ``spacing``, ``origin`` and
    ``world_to_index_zyx``.

    **The two packages disagree about what a bounding box means**, and the
    disagreement is silent. ``AmiraLattice.spacing`` is derived node-centred as
    ``(hi - lo) / (dims - 1)``, whereas ``WorldFrame.seg_origin`` is documented as
    the *centre* of segmentation voxel (0, 0, 0). :meth:`from_frame` builds the
    bbox so the derived spacing comes back exactly equal to ``frame.seg_spacing``;
    ``tests/test_lattice.py`` pins that, because a half-voxel error here would
    quietly displace every metric that samples the volume at a graph point.
    """

    volume: np.ndarray  # (nz, ny, nx)
    dims: tuple  # (nx, ny, nz), matching Amira's x-fastest convention
    bbox: np.ndarray  # (6,) xmin xmax ymin ymax zmin zmax, voxel centres
    block: str = "Labels"

    @property
    def spacing(self) -> np.ndarray:
        """(3,) um, derived the way ``AmiraLattice`` derives it."""
        n = np.maximum(np.asarray(self.dims) - 1, 1)
        return (self.bbox[1::2] - self.bbox[0::2]) / n

    @property
    def origin(self) -> np.ndarray:
        return self.bbox[0::2].copy()

    def world_to_voxel(self, coords) -> np.ndarray:
        """(N,3) world um (x, y, z) -> fractional voxel (ix, iy, iz)."""
        xyz = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        return (xyz - self.origin) / self.spacing

    def world_to_index_zyx(self, coords) -> np.ndarray:
        """(N,3) world um (x, y, z) -> integer ``(iz, iy, ix)``, the array's own order."""
        return np.round(self.world_to_voxel(coords)).astype(np.int64)[:, ::-1]

    @classmethod
    def from_frame(cls, volume: np.ndarray, frame, stride: int = 1,
                   block: str = "Labels") -> "LatticeView":
        """Wrap `volume` using the geometry in a :class:`~..frame.WorldFrame`."""
        s = max(int(stride), 1)
        nz, ny, nx = volume.shape
        dims = (nx, ny, nz)
        spacing = np.asarray(frame.seg_spacing, dtype=np.float64) * s
        origin = np.asarray(frame.seg_origin, dtype=np.float64)
        # bbox spans voxel *centres*, so the far face is (n - 1) steps out. This is
        # what makes `spacing` round-trip exactly.
        hi = origin + (np.asarray(dims) - 1) * spacing
        bbox = np.empty(6, dtype=np.float64)
        bbox[0::2] = origin
        bbox[1::2] = hi
        return cls(volume=volume, dims=dims, bbox=bbox, block=block)


def load_lattice_view(labels, frame, stride: int = 1, verbose: bool = False) -> LatticeView:
    """Decode the lattice and wrap it, in one call."""

    def report(done, total, elapsed):
        if verbose:
            print(f"    decoding {done}/{total} slices ({elapsed:.1f}s)", end="\r")

    volume = decode_volume(labels, stride=stride, progress=report if verbose else None)
    if verbose:
        gb = volume.nbytes / 1e9
        print(f"    decoded {volume.shape} = {gb:.2f} GB" + " " * 20)
    return LatticeView.from_frame(volume, frame, stride=stride)


class MaskVolume:
    """The whole mask, decoded once at full resolution and kept.

    The viewer reads the lattice a plane at a time, which is right for a pick and
    wrong for anything global: the whole-tree isosurface re-decodes on every rebuild,
    and a local box costs sixty decodes per pick. On a machine with room for it,
    holding the 2.34 GB array is simply better, and this is the one place to ask for
    it.

    **Full resolution only, never keyed by stride.** A plane painted at k=5 has no
    representation at all in a stride-4 array, so a decimated cache could not be
    repaired after an edit without silently dropping corrections. Callers that want a
    stride slice ``array`` in memory, which is a view and costs nothing.

    Staleness is resolved by *pulling*, not by being told. `MaskEdits` carries a
    version counter and `maskedit`'s whole design is that a correction reaches every
    reader through one `slice_z` rather than by threading notifications through the
    call sites. `PaintSession.commit()` also runs on every pick, so a push would do
    work whether or not anything changed; a pull is an integer compare.
    """

    def __init__(self, labels):
        self.labels = labels
        self.array: np.ndarray | None = None
        self.decode_seconds = 0.0
        self._edits = None
        self._version = -1

    # -- state ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.array is not None

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes) if self.array is not None else 0

    @property
    def edits(self):
        """The edit store behind the source, if it is a `MaskSource`."""
        return getattr(self.labels, "edits", None)

    def describe(self) -> str:
        if self.array is None:
            return "mask not resident"
        return (f"mask resident: {self.array.shape} = {self.nbytes / 1e9:.2f} GB "
                f"in {self.decode_seconds:.1f}s")

    # -- access -----------------------------------------------------------

    def get(self, progress=None) -> np.ndarray:
        """The full-resolution array, decoding it if this is the first ask."""
        if self.array is None:
            t0 = time.time()
            self.array = decode_volume(self.labels, stride=1, progress=progress)
            self.decode_seconds = time.time() - t0
            edits = self.edits
            self._edits = edits
            self._version = getattr(edits, "version", 0) if edits is not None else 0
        else:
            self.refresh()
        return self.array

    def peek(self) -> np.ndarray | None:
        """The array only if it is already resident. Never triggers a decode.

        `g` is pressed far more often than `a`; a local box must not be the thing
        that suddenly costs two seconds and 2.34 GB.
        """
        if self.array is None:
            return None
        self.refresh()
        return self.array

    def slice_z(self, k: int) -> np.ndarray:
        """Plane `k`, read-only. Duck-types the lattice's own accessor.

        A view rather than a copy, marked read-only because `ByteRLELattice.slice_z`
        hands out a fresh array every call and a caller may reasonably believe it
        owns the result. Better an exception than silent corruption of the shared
        volume.
        """
        array = self.get()
        plane = array[int(k)]
        view = plane.view()
        view.flags.writeable = False
        return view

    # -- staleness --------------------------------------------------------

    def refresh(self) -> int:
        """Re-read the planes whose edits changed. Returns how many.

        Each stale plane is *replaced* with whatever the source now says it is,
        rather than having edits applied to it. That is what makes added, changed and
        removed corrections one code path: a `diff` that deleted every entry on a
        plane leaves `slice_z` returning the pristine base, and the resident copy
        follows.
        """
        edits = self.edits
        if self.array is None or edits is None:
            return 0
        if edits is not self._edits:
            # A different store entirely -- `MaskEdits.load` builds a new object.
            # Never happens inside one Session; refuse to be quietly wrong if it does.
            self.array = None
            self._edits = None
            self._version = -1
            return 0
        if edits.version == self._version:
            return 0

        stale = [k for k, v in edits.touched_at.items() if v > self._version]
        for k in stale:
            if 0 <= k < self.array.shape[0]:
                self.array[k] = self.labels.slice_z(k)
        self._version = edits.version
        return len(stale)

    def release(self) -> None:
        self.array = None
        self._edits = None
        self._version = -1
        self.decode_seconds = 0.0
