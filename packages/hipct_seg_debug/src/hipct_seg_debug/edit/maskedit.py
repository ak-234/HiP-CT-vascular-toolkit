"""Hand corrections to the segmentation, held apart from the segmentation.

The source mask is a 2.34 GB ``HxByteRLE`` lattice that this package can decode and
-- until :mod:`..rle_write` -- could not write. Even with an encoder, rewriting the
whole file to record a brush stroke would be absurd. So an edit session keeps a
**sparse sidecar**: the segmentation voxels whose value differs from the file, and
nothing else. A morning's careful painting is tens of thousands of voxels, which is
a few hundred kilobytes.

Two classes, and the second is the reason the first is shaped the way it is:

:class:`MaskEdits`   the store. Keyed by segmentation z plane, then by flat ``(row,
                     col)`` index within that plane.
:class:`MaskSource`  ``labels`` and a :class:`MaskEdits` presented as one lattice,
                     with the same ``slice_z`` / ``nx`` / ``ny`` / ``nz`` surface
                     :class:`~..rle.ByteRLELattice` has.

Everything in this package that reads the mask goes through ``slice_z`` -- the slab
gather in ``viewer2d``, the collapse detector in ``crosssection``, the lazy dask
layers in ``volume``, ``lattice.decode_volume`` and therefore ``skeletonise``. Making
:class:`MaskSource` answer to that one method is what puts a correction in front of
all of them at once, instead of threading an ``edits=`` argument through six call
sites.

**The invariant.** The store holds *exactly* the voxels that differ from the source.
:meth:`MaskEdits.diff` does not merely add changes, it also **removes entries that
have returned to their original value**. That is what makes napari's own Ctrl+Z
correct without this module knowing that undo exists: napari restores the array, the
next diff sees agreement with the baseline, and the entry disappears. Accumulating
paint events instead would be wrong -- ``Labels.undo()`` emits no paint event, which
was measured rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MaskEdits:
    """Segmentation voxels that differ from the source lattice.

    Coordinates are segmentation indices throughout: ``k`` is the z plane, ``row``
    is y, ``col`` is x. That is the lattice's own grid, deliberately not the raw
    pixel grid -- the two differ by ``frame.bin_factor`` (2 on LADAF-2024-28), and
    an edit expressed on the finer grid would need a lossy many-to-one reduction to
    land back on the mask.
    """

    def __init__(self, ny: int, nx: int, nz: int | None = None, source: str = ""):
        self.ny = int(ny)
        self.nx = int(nx)
        self.nz = int(nz) if nz is not None else None
        self.source = str(source)
        self._planes: dict[int, dict[int, int]] = {}
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # A monotonic counter, plus the version at which each plane last changed.
        # Anything holding a decoded copy of the mask compares these to find out
        # what it has to re-read -- a *pull*, deliberately, because `commit()` runs
        # on every pick and a push would make every reader the writer's business.
        # See `edit.lattice.MaskVolume.refresh`.
        self.version = 0
        self._touched_at: dict[int, int] = {}

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        return self.n_voxels

    @property
    def n_voxels(self) -> int:
        return sum(len(p) for p in self._planes.values())

    @property
    def is_empty(self) -> bool:
        return not any(self._planes.values())

    @property
    def touched_planes(self) -> list[int]:
        return sorted(k for k, p in self._planes.items() if p)

    @property
    def touched_at(self) -> dict[int, int]:
        """Plane -> the version at which it last changed.

        Keeps planes that have since been emptied, which `touched_planes` drops.
        That is the whole point: a reader holding a decoded copy has to re-read a
        plane whose edits were *removed* just as much as one that gained them.
        """
        return dict(self._touched_at)

    def _bump(self, planes) -> None:
        self.version += 1
        for k in planes:
            self._touched_at[int(k)] = self.version

    def stats(self) -> dict:
        """How many voxels were added to the mask, and how many taken away."""
        added = removed = 0
        for plane in self._planes.values():
            for value in plane.values():
                if value:
                    added += 1
                else:
                    removed += 1
        return {"added": added, "removed": removed, "total": added + removed,
                "planes": len(self.touched_planes)}

    def describe(self) -> str:
        s = self.stats()
        if not s["total"]:
            return "no mask edits"
        return (f"{s['total']:,} edited voxels on {s['planes']} plane(s): "
                f"+{s['added']:,} painted, -{s['removed']:,} erased")

    # ------------------------------------------------------------------ writing

    def _flat(self, rows, cols) -> np.ndarray:
        return np.asarray(rows, dtype=np.int64) * self.nx + np.asarray(cols, dtype=np.int64)

    def set_plane(self, k: int, rows, cols, values) -> int:
        """Record ``values`` at ``(rows, cols)`` on plane `k`. Returns how many."""
        rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        cols = np.atleast_1d(np.asarray(cols, dtype=np.int64))
        values = np.atleast_1d(np.asarray(values, dtype=np.uint8))
        if not len(rows):
            return 0
        if len(values) == 1 and len(rows) > 1:
            values = np.repeat(values, len(rows))
        plane = self._planes.setdefault(int(k), {})
        for flat, value in zip(self._flat(rows, cols).tolist(), values.tolist()):
            plane[int(flat)] = int(value)
        self._cache.pop(int(k), None)
        self._bump([k])
        return len(rows)

    def clear(self) -> None:
        touched = list(self._planes)
        self._planes.clear()
        self._cache.clear()
        if touched:
            self._bump(touched)

    def clear_window(self, origin_kji, shape_kji) -> int:
        """Forget every edit inside a window. The 'revert this box' operation."""
        k0, j0, i0 = (int(v) for v in origin_kji)
        nk, nj, ni = (int(v) for v in shape_kji)
        dropped = 0
        emptied = []
        for k in range(k0, k0 + nk):
            plane = self._planes.get(k)
            if not plane:
                continue
            hit = [f for f in plane if _in_window(f, self.nx, j0, j0 + nj, i0, i0 + ni)]
            for flat in hit:
                del plane[flat]
                dropped += 1
            if hit:
                emptied.append(k)
            self._cache.pop(k, None)
        # Only when something actually went. `diff` calls this on every commit, and
        # a commit happens on every pick -- an unconditional bump would put every
        # reader into a repair pass forever, for a window that had no edits in it.
        if emptied:
            self._bump(emptied)
        return dropped

    def diff(self, edited: np.ndarray, base: np.ndarray, origin_kji) -> int:
        """Replace this window's contribution to the store with ``edited != base``.

        Wholesale replacement rather than accumulation, which is what enforces the
        module invariant: a voxel the user painted and then undid agrees with `base`
        again, is therefore not in the new diff, and so is dropped. Entries outside
        the window are untouched.

        Returns the number of voxels the window now contributes.
        """
        edited = np.asarray(edited)
        base = np.asarray(base)
        if edited.shape != base.shape:
            raise ValueError(f"edited {edited.shape} and base {base.shape} differ")
        k0, j0, i0 = (int(v) for v in origin_kji)
        nk, nj, ni = edited.shape

        self.clear_window((k0, j0, i0), (nk, nj, ni))
        total = 0
        for p in range(nk):
            jj, ii = np.nonzero(edited[p] != base[p])
            if not len(jj):
                continue
            total += self.set_plane(k0 + p, jj + j0, ii + i0, edited[p][jj, ii])
        return total

    # ------------------------------------------------------------------ reading

    def _arrays(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """``(flat_index, value)`` for one plane, cached until the plane is written."""
        k = int(k)
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        plane = self._planes.get(k) or {}
        if plane:
            idx = np.fromiter(plane.keys(), dtype=np.int64, count=len(plane))
            val = np.fromiter(plane.values(), dtype=np.uint8, count=len(plane))
        else:
            idx = np.empty(0, dtype=np.int64)
            val = np.empty(0, dtype=np.uint8)
        self._cache[k] = (idx, val)
        return idx, val

    def apply_plane(self, plane: np.ndarray, k: int, j0: int = 0, i0: int = 0) -> int:
        """Composite plane `k`'s edits onto `plane`, in place. Returns how many landed."""
        idx, val = self._arrays(k)
        if not len(idx):
            return 0
        rows = idx // self.nx - j0
        cols = idx % self.nx - i0
        keep = (rows >= 0) & (rows < plane.shape[0]) & (cols >= 0) & (cols < plane.shape[1])
        if not keep.any():
            return 0
        plane[rows[keep], cols[keep]] = val[keep]
        return int(keep.sum())

    def apply(self, window: np.ndarray, origin_kji) -> int:
        """Composite every edit inside `window` onto it, in place."""
        k0, j0, i0 = (int(v) for v in origin_kji)
        return sum(
            self.apply_plane(window[p], k0 + p, j0, i0) for p in range(window.shape[0])
        )

    def added_mask(self, base: np.ndarray, origin_kji) -> np.ndarray:
        """Where the user *added* lumen: edited to non-zero over a zero source.

        This is what ``reskeletonise`` in add mode uses to decide which part of a
        freshly-derived local skeleton is new and which was already in the graph.
        """
        edited = np.asarray(base).copy()
        self.apply(edited, origin_kji)
        return (edited > 0) & (np.asarray(base) == 0)

    def bbox_seg(self) -> np.ndarray | None:
        """``(2, 3)`` inclusive ``(k, row, col)`` bounds of every edit, or ``None``."""
        planes = self.touched_planes
        if not planes:
            return None
        lo = np.array([planes[0], self.ny, self.nx], dtype=np.int64)
        hi = np.array([planes[-1], -1, -1], dtype=np.int64)
        for k in planes:
            idx, _ = self._arrays(k)
            rows, cols = idx // self.nx, idx % self.nx
            lo[1] = min(lo[1], int(rows.min()))
            lo[2] = min(lo[2], int(cols.min()))
            hi[1] = max(hi[1], int(rows.max()))
            hi[2] = max(hi[2], int(cols.max()))
        return np.array([lo, hi])

    # ---------------------------------------------------------------- on disk

    def save(self, path) -> Path:
        """Write the store as a small ``.npz``.

        Flat ``(k, row, col, value)`` columns rather than one array per plane: it
        stays one compact record however scattered the edits are, and it is
        readable by anything that can open an npz, which matters for a file that
        is the only record of manual work.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ks, rows, cols, vals = [], [], [], []
        for k in self.touched_planes:
            idx, val = self._arrays(k)
            ks.append(np.full(len(idx), k, dtype=np.int32))
            rows.append((idx // self.nx).astype(np.int32))
            cols.append((idx % self.nx).astype(np.int32))
            vals.append(val)
        empty32 = np.empty(0, dtype=np.int32)
        np.savez_compressed(
            path,
            k=np.concatenate(ks) if ks else empty32,
            row=np.concatenate(rows) if rows else empty32,
            col=np.concatenate(cols) if cols else empty32,
            value=np.concatenate(vals) if vals else np.empty(0, dtype=np.uint8),
            ny=self.ny, nx=self.nx,
            nz=-1 if self.nz is None else self.nz,
            source=self.source,
        )
        return path

    @classmethod
    def load(cls, path, *, expect_dims=None) -> "MaskEdits":
        """Read a store back.

        `expect_dims` is ``(nx, ny, nz)`` as ``WorldFrame`` and Amira order them. It
        is checked rather than trusted: a store applied to the wrong lattice would
        scatter corrections into unrelated tissue and look entirely plausible.
        """
        with np.load(Path(path), allow_pickle=False) as z:
            ny, nx = int(z["ny"]), int(z["nx"])
            nz = int(z["nz"])
            edits = cls(ny=ny, nx=nx, nz=None if nz < 0 else nz, source=str(z["source"]))
            ks, rows = z["k"], z["row"]
            cols, vals = z["col"], z["value"]
        if expect_dims is not None:
            want_nx, want_ny, want_nz = (int(v) for v in expect_dims)
            if (want_nx, want_ny) != (nx, ny) or (nz >= 0 and nz != want_nz):
                raise ValueError(
                    f"{Path(path).name}: edits are for a "
                    f"{nx}x{ny}x{nz if nz >= 0 else '?'} lattice, "
                    f"this one is {want_nx}x{want_ny}x{want_nz}"
                )
        for k in np.unique(ks):
            m = ks == k
            edits.set_plane(int(k), rows[m], cols[m], vals[m])
        return edits


def _in_window(flat: int, nx: int, j0: int, j1: int, i0: int, i1: int) -> bool:
    j, i = divmod(flat, nx)
    return j0 <= j < j1 and i0 <= i < i1


class MaskSource:
    """A lattice and its pending edits, presented as one read-only lattice.

    Deliberately the same duck type as :class:`~..rle.ByteRLELattice` -- ``nx``,
    ``ny``, ``nz``, ``dims``, ``slice_z`` -- so it can be dropped into
    ``Session.labels`` and every consumer downstream reads the corrected mask with
    no further plumbing. Unknown attributes fall through to the wrapped lattice, so
    the cache paths and the self-test's ``decode_sequential`` keep working.
    """

    def __init__(self, labels, edits: MaskEdits | None = None):
        self.labels = labels
        self.nx, self.ny, self.nz = int(labels.nx), int(labels.ny), int(labels.nz)
        self.edits = edits if edits is not None else MaskEdits(
            ny=self.ny, nx=self.nx, nz=self.nz,
            source=str(getattr(labels, "path", "")),
        )
        if (self.edits.ny, self.edits.nx) != (self.ny, self.nx):
            raise ValueError(
                f"edits are for {self.edits.nx}x{self.edits.ny} planes, "
                f"the lattice has {self.nx}x{self.ny}"
            )

    def __getattr__(self, name):  # pragma: no cover - delegation, exercised indirectly
        return getattr(self.labels, name)

    @property
    def dims(self) -> np.ndarray:
        return np.array([self.nx, self.ny, self.nz], dtype=np.int64)

    def slice_z(self, k: int) -> np.ndarray:
        """Plane `k` with the edits composited on. Same contract as the lattice's."""
        plane = self.labels.slice_z(k)
        if self.edits.is_empty:
            return plane
        plane = plane.copy()
        self.edits.apply_plane(plane, int(k))
        return plane

    def slice_rows(self, k: int, row0: int, row1: int) -> np.ndarray:
        """Rows ``[row0, row1)`` of plane `k`, with the edits composited on.

        Defined rather than delegated, and that is the whole point: ``__getattr__``
        would hand a caller the *unedited* lattice's band and nothing would say so.
        With no edits loaded the fast path is the wrapped reader's; with edits it
        falls back to compositing the full plane, which is correct and is the rarer
        case by far.
        """
        rows = getattr(self.labels, "slice_rows", None)
        if self.edits.is_empty and rows is not None:
            return rows(int(k), int(row0), int(row1))
        return self.slice_z(int(k))[int(row0):int(row1)]

    def base_slice_z(self, k: int) -> np.ndarray:
        """Plane `k` exactly as the file has it -- the diff baseline."""
        return self.labels.slice_z(k)

    def decode_sequential(self, n_slices: int) -> np.ndarray:
        """Composited counterpart of the lattice's reference decode.

        Overridden rather than delegated so ``selftest.test_rle_index`` keeps
        comparing like with like once a session has edits loaded.
        """
        out = self.labels.decode_sequential(n_slices).copy()
        for p in range(out.shape[0]):
            self.edits.apply_plane(out[p], p)
        return out

    def window(self, k0, k1, j0, j1, i0, i1, *, edited: bool = True) -> np.ndarray:
        """A ``(k1-k0, j1-j0, i1-i0)`` block, clipped to the lattice and zero-padded.

        Zero padding rather than a clipped shape: callers size a paint box or a
        re-skeletonisation box from world coordinates and should not have to
        re-derive it when the box runs off the edge of the volume.
        """
        k0, k1 = int(k0), int(k1)
        j0, j1 = int(j0), int(j1)
        i0, i1 = int(i0), int(i1)
        out = np.zeros((k1 - k0, j1 - j0, i1 - i0), dtype=np.uint8)
        read = self.slice_z if edited else self.base_slice_z
        for p, k in enumerate(range(k0, k1)):
            if not 0 <= k < self.nz:
                continue
            plane = read(k)
            sj0, sj1 = max(j0, 0), min(j1, self.ny)
            si0, si1 = max(i0, 0), min(i1, self.nx)
            if sj0 >= sj1 or si0 >= si1:
                continue
            out[p, sj0 - j0:sj1 - j0, si0 - i0:si1 - i0] = plane[sj0:sj1, si0:si1]
        return out

    def describe(self) -> str:
        return (f"segmentation {self.nx}x{self.ny}x{self.nz} "
                f"+ {self.edits.describe()}")
