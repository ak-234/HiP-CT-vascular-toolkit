"""Random-access readers for compressed and raw Amira scalar lattices.

The codec is a byte-oriented run-length encoding: read a control byte ``n``; if
``n > 127`` the next ``n - 128`` bytes are literal, otherwise the single following
byte repeats ``n`` times.

The stream is inherently sequential, so slice-level random access is provided by a
one-off index pass that records, for every z slice, the compressed offset to resume
from plus how many already-decoded output bytes to discard (runs straddle slice
boundaries). The index is tiny -- two int64 per slice -- and is cached to disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def _build_index(buf: np.ndarray, slice_nbytes: np.int64, n_slices: np.int64):
    """Walk the whole stream, recording an entry point per slice.

    Returns ``(offsets, discards, total_out)``. Entry ``s`` means: start decoding at
    ``buf[offsets[s]]`` and throw away the first ``discards[s]`` output bytes.
    """
    offsets = np.zeros(n_slices, dtype=np.int64)
    discards = np.zeros(n_slices, dtype=np.int64)
    n = buf.shape[0]

    k = np.int64(0)  # cursor into the compressed stream
    out = np.int64(0)  # decoded bytes produced so far
    nxt = np.int64(1)  # next slice boundary we are looking for

    # Walk the entire stream, not just up to the last boundary, so `out` reports the
    # true decoded size and the caller can verify it against nx*ny*nz.
    while k < n:
        ctrl = np.int64(buf[k])
        if ctrl > 127:
            run = ctrl - 128
            step = np.int64(1) + run
        else:
            run = ctrl
            step = np.int64(2)
        if run == 0:
            # Defensive: a zero-length packet would not advance `out` and could spin.
            k += step
            continue

        end = out + run
        # A single packet can span several slice boundaries when it is a long repeat.
        while nxt < n_slices and end >= nxt * slice_nbytes:
            boundary = nxt * slice_nbytes
            offsets[nxt] = k
            discards[nxt] = boundary - out
            nxt += 1

        out = end
        k += step

    return offsets, discards, out


@njit(cache=True)
def _decode(buf: np.ndarray, start: np.int64, discard: np.int64, want: np.int64):
    """Decode ``want`` output bytes starting ``discard`` bytes into the packet at ``start``."""
    res = np.empty(want, dtype=np.uint8)
    n = buf.shape[0]
    k = start
    produced = np.int64(0)  # bytes emitted, including the discarded prefix
    filled = np.int64(0)

    while filled < want and k < n:
        ctrl = np.int64(buf[k])
        if ctrl > 127:
            run = ctrl - 128
            literal = True
            k += 1
        else:
            run = ctrl
            literal = False
            k += 1
        if run == 0:
            if not literal:
                k += 1
            continue

        # How much of this packet lies past the discarded prefix.
        skip = np.int64(0)
        if produced < discard:
            skip = discard - produced
            if skip >= run:
                produced += run
                k += run if literal else 1
                continue

        take = run - skip
        if take > want - filled:
            take = want - filled

        if literal:
            for i in range(take):
                res[filled + i] = buf[k + skip + i]
            k += run
        else:
            val = buf[k]
            for i in range(take):
                res[filled + i] = val
            k += 1

        filled += take
        produced += run

    if filled < want:
        # Truncated stream: pad rather than fail, the caller validates coverage.
        for i in range(filled, want):
            res[i] = 0
    return res


class ByteRLELattice:
    """Slice-wise random access into one ``HxByteRLE`` field of an Amira lattice."""

    def __init__(self, path, field, dims, cache_dir=None):
        """
        path   : the .am file
        field  : `amira.LatticeField` describing this lattice
        dims   : (nx, ny, nz) with x varying fastest
        """
        self.path = Path(path)
        self.field = field
        self.dims = np.asarray(dims, dtype=np.int64)
        self.nx, self.ny, self.nz = (int(v) for v in self.dims)
        self.slice_nbytes = self.nx * self.ny

        # Read the compressed stream once into a plain contiguous array: numba needs a
        # real ndarray, and re-copying a memmap view on every slice access would dominate
        # the decode cost. These streams are only tens of MB.
        size = self.path.stat().st_size
        end = field.data_offset + field.n_bytes
        if end > size:
            raise ValueError(
                f"{self.path.name}: field '{field.name}' claims {field.n_bytes} bytes "
                f"at offset {field.data_offset}, past end of file ({size})"
            )
        with open(self.path, "rb") as fh:
            fh.seek(field.data_offset)
            self._buf = np.frombuffer(fh.read(field.n_bytes), dtype=np.uint8)

        self._cache_dir = Path(cache_dir) if cache_dir else self.path.parent / ".hipct_cache"
        self._offsets, self._discards = self._load_or_build_index()

    # -- index ------------------------------------------------------------- #
    @property
    def _index_path(self) -> Path:
        stem = self.path.stem.replace(" ", "_")
        return self._cache_dir / f"{stem}.{self.field.name}.sliceidx.npz"

    def _load_or_build_index(self):
        p = self._index_path
        if p.exists():
            try:
                z = np.load(p)
                if (
                    int(z["slice_nbytes"]) == self.slice_nbytes
                    and int(z["n_slices"]) == self.nz
                    and int(z["n_bytes"]) == self.field.n_bytes
                ):
                    return z["offsets"], z["discards"]
            except Exception:
                pass  # stale or corrupt cache -- rebuild

        offsets, discards, total = _build_index(
            np.ascontiguousarray(self._buf), np.int64(self.slice_nbytes), np.int64(self.nz)
        )
        expected = self.slice_nbytes * self.nz
        if total < expected:
            raise ValueError(
                f"{self.path.name}: field '{self.field.name}' decoded to {total} bytes, "
                f"expected {expected} ({self.nx}x{self.ny}x{self.nz})"
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            offsets=offsets,
            discards=discards,
            slice_nbytes=self.slice_nbytes,
            n_slices=self.nz,
            n_bytes=self.field.n_bytes,
        )
        return offsets, discards

    # -- access ------------------------------------------------------------ #
    def slice_z(self, z: int) -> np.ndarray:
        """Decode z slice ``z`` as a ``(ny, nx)`` uint8 array."""
        if not 0 <= z < self.nz:
            raise IndexError(f"z={z} out of range [0, {self.nz})")
        flat = _decode(
            np.ascontiguousarray(self._buf),
            np.int64(self._offsets[z]),
            np.int64(self._discards[z]),
            np.int64(self.slice_nbytes),
        )
        return flat.reshape(self.ny, self.nx)

    def slice_rows(self, z: int, row0: int, row1: int) -> np.ndarray:
        """Decode rows ``[row0, row1)`` of z slice ``z`` as a ``(row1-row0, nx)`` array.

        A row band is contiguous in the decoded stream, so reaching one costs a walk
        over the packets in front of it -- which skips whole literal runs without
        copying them -- and then a copy of the band alone. Measured on a
        3400x2964x4748 mask: 0.13 ms for 64 rows out of the middle of a slice against
        5.6 ms for the whole slice, a factor of 43. A cross-section is a few tens of
        rows wide, so every consumer that samples a plane rather than a volume should
        be asking for rows, not slices.
        """
        if not 0 <= z < self.nz:
            raise IndexError(f"z={z} out of range [0, {self.nz})")
        row0 = max(int(row0), 0)
        row1 = min(int(row1), self.ny)
        if row1 <= row0:
            return np.zeros((0, self.nx), dtype=np.uint8)
        flat = _decode(
            np.ascontiguousarray(self._buf),
            np.int64(self._offsets[z]),
            np.int64(self._discards[z] + row0 * self.nx),
            np.int64((row1 - row0) * self.nx),
        )
        return flat.reshape(row1 - row0, self.nx)

    def decode_sequential(self, n_slices: int) -> np.ndarray:
        """Reference decode of the first ``n_slices`` slices straight from the start.

        Used by the self-test to prove the cached index agrees with a plain pass.
        """
        want = self.slice_nbytes * int(n_slices)
        flat = _decode(np.ascontiguousarray(self._buf), np.int64(0), np.int64(0), np.int64(want))
        return flat.reshape(int(n_slices), self.ny, self.nx)


class RawLattice:
    """Slice-wise access to an uncompressed Amira lattice via a read-only memmap.

    A full-resolution HiP-CT mask can be tens of GB.  Mapping it keeps construction
    constant-memory, and :meth:`slice_z` copies only the requested plane so callers
    receive the same independent-array semantics as :class:`ByteRLELattice`.
    """

    def __init__(self, path, field, dims, cache_dir=None):
        del cache_dir  # Kept in the signature so both lattice readers are interchangeable.
        self.path = Path(path)
        self.field = field
        self.dims = np.asarray(dims, dtype=np.int64)
        self.nx, self.ny, self.nz = (int(v) for v in self.dims)
        self.slice_nbytes = self.nx * self.ny * np.dtype(field.dtype).itemsize

        if field.encoding is not None:
            raise ValueError(
                f"{self.path.name}: RawLattice cannot decode encoding '{field.encoding}'"
            )
        if field.n_components != 1:
            raise ValueError(
                f"{self.path.name}: raw field '{field.name}' has "
                f"{field.n_components} components; expected one"
            )

        expected = self.slice_nbytes * self.nz
        end = field.data_offset + expected
        size = self.path.stat().st_size
        if end > size:
            raise ValueError(
                f"{self.path.name}: raw field '{field.name}' needs {expected} bytes "
                f"at offset {field.data_offset}, past end of file ({size})"
            )
        self._volume = np.memmap(
            self.path,
            dtype=np.dtype(field.dtype),
            mode="r",
            offset=field.data_offset,
            shape=(self.nz, self.ny, self.nx),
            order="C",
        )

    def slice_z(self, z: int) -> np.ndarray:
        """Read z slice ``z`` as an independent ``(ny, nx)`` array."""
        if not 0 <= z < self.nz:
            raise IndexError(f"z={z} out of range [0, {self.nz})")
        return np.array(self._volume[z], copy=True)

    def slice_rows(self, z: int, row0: int, row1: int) -> np.ndarray:
        """Rows ``[row0, row1)`` of plane ``z``, matching the compressed reader."""
        if not 0 <= z < self.nz:
            raise IndexError(f"z={z} out of range [0, {self.nz})")
        row0 = max(int(row0), 0)
        row1 = min(int(row1), self.ny)
        if row1 <= row0:
            return np.zeros((0, self.nx), dtype=self._volume.dtype)
        return np.array(self._volume[z, row0:row1], copy=True)

    def slice_window(
        self, z: int, row0: int, row1: int, col0: int, col1: int, step: int = 1
    ) -> np.ndarray:
        """Read a strided window without first copying the complete raw plane."""
        if not 0 <= z < self.nz:
            raise IndexError(f"z={z} out of range [0, {self.nz})")
        s = max(int(step), 1)
        return np.array(self._volume[z, row0:row1:s, col0:col1:s], copy=True)

    def decode_sequential(self, n_slices: int) -> np.ndarray:
        """Read the first ``n_slices`` planes, matching the compressed reader API."""
        count = int(n_slices)
        if not 0 <= count <= self.nz:
            raise IndexError(f"n_slices={count} out of range [0, {self.nz}]")
        return np.array(self._volume[:count], copy=True)


def open_lattice(path, field, dims, cache_dir=None):
    """Open a supported scalar lattice without materialising its full volume."""
    if field.encoding is None:
        return RawLattice(path, field, dims, cache_dir=cache_dir)
    if field.encoding == "HxByteRLE":
        return ByteRLELattice(path, field, dims, cache_dir=cache_dir)
    raise ValueError(
        f"{Path(path).name}: unsupported lattice encoding '{field.encoding}' "
        f"for field '{field.name}'"
    )
