"""Streaming reader for binary uniform Amira/Avizo lattice volumes.

Only the byte-oriented formats needed by the paired coronary segmentations are
implemented. HxByteRLE streams are indexed once and decoded one z-slice at a
time, so a multi-gigabyte label lattice never needs to reside in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = lambda *args, **kwargs: (lambda function: function)


DATA_MARKER = b"# Data section follows"
_DECLARATION = re.compile(
    r"^\s*Lattice\s*\{\s*([A-Za-z0-9_]+)(?:\[(\d+)\])?\s+([A-Za-z0-9_]+)\s*\}\s*@([0-9]+)"
    r"(?:\(\s*([A-Za-z0-9_]+)\s*,\s*([0-9]+)\s*\))?\s*$"
)
_DTYPES = {"byte": np.uint8, "ubyte": np.uint8}


@dataclass
class LatticeField:
    name: str
    dtype: type
    components: int
    block_id: int
    encoding: str | None
    encoded_bytes: int | None
    data_offset: int = -1


@dataclass(frozen=True)
class LatticeHeader:
    path: Path
    dims: np.ndarray
    bbox_um: np.ndarray
    fields: dict[str, LatticeField]

    @property
    def spacing_um(self) -> np.ndarray:
        return (self.bbox_um[1::2] - self.bbox_um[0::2]) / np.maximum(
            self.dims - 1, 1
        )

    @property
    def spacing_mm(self) -> np.ndarray:
        return self.spacing_um / 1000.0

    @property
    def origin_mm(self) -> np.ndarray:
        return self.bbox_um[0::2] / 1000.0


def read_lattice_header(path: str | Path) -> LatticeHeader:
    path = Path(path)
    with path.open("rb") as stream:
        probe = stream.read(4 << 20)
    marker = probe.find(DATA_MARKER)
    if marker < 0:
        raise ValueError(f"{path.name}: data marker not found in first 4 MiB")
    first_line = probe.splitlines()[0]
    if b"BINARY-LITTLE-ENDIAN" not in first_line:
        raise ValueError(f"{path.name}: expected BINARY-LITTLE-ENDIAN Amira lattice")
    header_text = probe[:marker].decode("latin-1")
    dims_match = re.search(
        r"^\s*define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)",
        header_text,
        re.MULTILINE,
    )
    if not dims_match:
        raise ValueError(f"{path.name}: missing Lattice dimensions")
    dims = np.asarray([int(value) for value in dims_match.groups()], dtype=np.int64)
    bbox_match = re.search(r"BoundingBox\s+([-\d.eE+\s]+?),", header_text)
    if not bbox_match:
        raise ValueError(f"{path.name}: missing BoundingBox")
    bbox = np.asarray(bbox_match.group(1).split(), dtype=np.float64)
    if bbox.shape != (6,):
        raise ValueError(f"{path.name}: BoundingBox must contain six values")
    coord = re.search(r'CoordType\s+"(\w+)"', header_text)
    if coord and coord.group(1) != "uniform":
        raise ValueError(f"{path.name}: only uniform coordinates are supported")

    fields: dict[str, LatticeField] = {}
    for line in header_text.splitlines():
        match = _DECLARATION.match(line)
        if not match:
            continue
        base, components, name, block, encoding, encoded_bytes = match.groups()
        if base not in _DTYPES:
            raise ValueError(f"{path.name}: unsupported lattice dtype {base!r}")
        fields[name] = LatticeField(
            name=name,
            dtype=_DTYPES[base],
            components=int(components or 1),
            block_id=int(block),
            encoding=encoding,
            encoded_bytes=int(encoded_bytes) if encoded_bytes else None,
        )
    if not fields:
        raise ValueError(f"{path.name}: no lattice field declarations found")

    position = marker
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        for field in sorted(fields.values(), key=lambda item: item.block_id):
            stream.seek(position)
            window = stream.read(4096)
            token = f"\n@{field.block_id}".encode("ascii")
            relative = window.find(token)
            if relative < 0:
                raise ValueError(f"{path.name}: block @{field.block_id} marker not found")
            line_start = position + relative + 1
            stream.seek(line_start)
            stream.readline()
            field.data_offset = stream.tell()
            if field.encoded_bytes is None:
                size = (
                    int(np.prod(dims))
                    * field.components
                    * np.dtype(field.dtype).itemsize
                )
            else:
                size = field.encoded_bytes
            if field.data_offset + size > file_size:
                raise ValueError(f"{path.name}: block @{field.block_id} exceeds file size")
            position = field.data_offset + size
    return LatticeHeader(path=path, dims=dims, bbox_um=bbox, fields=fields)


@njit(cache=True)
def _build_slice_index(buffer, slice_bytes, slices):
    offsets = np.zeros(slices, dtype=np.int64)
    discards = np.zeros(slices, dtype=np.int64)
    cursor = 0
    produced = 0
    next_slice = 1
    while cursor < len(buffer):
        control = int(buffer[cursor])
        if control > 127:
            count = control - 128
            step = 1 + count
        else:
            count = control
            step = 2
        end = produced + count
        while next_slice < slices and end >= next_slice * slice_bytes:
            boundary = next_slice * slice_bytes
            offsets[next_slice] = cursor
            discards[next_slice] = boundary - produced
            next_slice += 1
        produced = end
        cursor += step
    return offsets, discards, produced


@njit(cache=True)
def _decode_bytes(buffer, start, discard, wanted):
    result = np.empty(wanted, dtype=np.uint8)
    cursor = start
    produced = 0
    filled = 0
    while filled < wanted and cursor < len(buffer):
        control = int(buffer[cursor])
        cursor += 1
        literal = control > 127
        count = control - 128 if literal else control
        if count == 0:
            if not literal:
                cursor += 1
            continue
        skip = max(discard - produced, 0)
        if skip >= count:
            produced += count
            cursor += count if literal else 1
            continue
        take = min(count - skip, wanted - filled)
        if literal:
            result[filled : filled + take] = buffer[
                cursor + skip : cursor + skip + take
            ]
            cursor += count
        else:
            result[filled : filled + take] = buffer[cursor]
            cursor += 1
        filled += take
        produced += count
    return result, filled


class AmiraByteLattice:
    """Slice-wise access to one byte field from an Amira lattice."""

    def __init__(
        self,
        header: LatticeHeader,
        field: LatticeField,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        if field.components != 1:
            raise ValueError("only scalar lattice fields are supported")
        self.header = header
        self.field = field
        self.nx, self.ny, self.nz = (int(value) for value in header.dims)
        self.slice_bytes = self.nx * self.ny
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if field.encoding is None:
            self._buffer = None
            self._offsets = self._discards = None
        elif field.encoding == "HxByteRLE":
            with header.path.open("rb") as stream:
                stream.seek(field.data_offset)
                self._buffer = np.frombuffer(
                    stream.read(field.encoded_bytes), dtype=np.uint8
                ).copy()
            self._offsets, self._discards = self._load_or_build_index()
        else:
            raise ValueError(f"unsupported encoding {field.encoding!r}")

    def _load_or_build_index(self):
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / (
                f"{self.header.path.stem}.{self.field.name}.slice-index.npz"
            )
            if cache_path.exists():
                saved = np.load(cache_path)
                if (
                    int(saved["encoded_bytes"]) == self.field.encoded_bytes
                    and int(saved["slice_bytes"]) == self.slice_bytes
                    and int(saved["slices"]) == self.nz
                ):
                    return saved["offsets"], saved["discards"]
        offsets, discards, decoded = _build_slice_index(
            self._buffer, self.slice_bytes, self.nz
        )
        expected = self.slice_bytes * self.nz
        if int(decoded) != expected:
            raise ValueError(
                f"{self.header.path.name}: {self.field.name} decoded to "
                f"{decoded:,} bytes, expected {expected:,}"
            )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                offsets=offsets,
                discards=discards,
                encoded_bytes=self.field.encoded_bytes,
                slice_bytes=self.slice_bytes,
                slices=self.nz,
            )
        return offsets, discards

    def slice_z(self, z: int) -> np.ndarray:
        if not 0 <= z < self.nz:
            raise IndexError(f"z={z} outside [0, {self.nz})")
        if self.field.encoding is None:
            offset = self.field.data_offset + z * self.slice_bytes
            with self.header.path.open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(self.slice_bytes)
            if len(raw) != self.slice_bytes:
                raise ValueError("raw lattice field is truncated")
            flat = np.frombuffer(raw, dtype=np.uint8).copy()
        else:
            flat, filled = _decode_bytes(
                self._buffer,
                int(self._offsets[z]),
                int(self._discards[z]),
                self.slice_bytes,
            )
            if int(filled) != self.slice_bytes:
                raise ValueError("HxByteRLE stream ended before a full slice was decoded")
        return flat.reshape(self.ny, self.nx)

    def decode_to_memmap(
        self,
        path: str | Path,
        *,
        threshold: int | None = None,
    ) -> np.memmap:
        """Decode to a ``(z,y,x)`` uint8 memory map using bounded RAM."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        volume = np.memmap(path, mode="w+", dtype=np.uint8, shape=(self.nz, self.ny, self.nx))
        for z in range(self.nz):
            image = self.slice_z(z)
            volume[z] = image if threshold is None else (image > threshold)
        volume.flush()
        return volume


def read_amira_lattice(
    path: str | Path,
    channel: str = "Labels",
    *,
    cache_dir: str | Path | None = None,
) -> AmiraByteLattice:
    header = read_lattice_header(path)
    if channel not in header.fields:
        raise KeyError(f"{channel!r} not found; fields={sorted(header.fields)}")
    return AmiraByteLattice(header, header.fields[channel], cache_dir=cache_dir)


__all__ = [
    "AmiraByteLattice",
    "LatticeField",
    "LatticeHeader",
    "read_amira_lattice",
    "read_lattice_header",
]
