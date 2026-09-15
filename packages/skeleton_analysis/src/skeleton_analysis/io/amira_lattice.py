"""Read Amira/Avizo binary ``Lattice`` (image volume) ``.am`` files.

The spatial-graph reader in :mod:`skeleton_analysis.io.amira` handles ASCII
``HxSpatialGraph`` files; this module handles the other Amira flavour: a
**uniform image lattice** (a 3-D voxel volume), which is what an Avizo-exported
segmentation is. These are typically ``BINARY-LITTLE-ENDIAN`` with the voxel
data ``HxByteRLE``-compressed, e.g.::

    # Avizo BINARY-LITTLE-ENDIAN 3.0
    define Lattice 1500 1250 1250
    Parameters { ... BoundingBox <x0 x1 y0 y1 z0 z1> ... CoordType "uniform" }
    Lattice { byte Labels } @1(HxByteRLE,37709212)
    Lattice { byte Probability } @2(HxByteRLE,38913759)
    # Data section follows
    @1
    <binary bytes>

Only ``byte`` (uint8) uniform lattices are supported (sufficient for
segmentations). The reader decodes a single named/indexed block on demand — the
full volume is ``NX*NY*NZ`` bytes (gigabytes for a large scan), so decoding just
``Labels`` avoids also materialising ``Probability``.

Coordinates: Avizo stores the flat data **X-fastest**, so the decoded volume is
reshaped to ``vol[z, y, x]``. World<->voxel mapping uses the ``BoundingBox``
(uniform node-centred spacing ``(hi-lo)/(dim-1)``).
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

_DEFINE_LATTICE_RE = re.compile(r"define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)")
_BBOX_RE = re.compile(r"BoundingBox\s+([-\d.eE+\s]+)")
# e.g.  Lattice { byte Labels } @1(HxByteRLE,37709212)
_DECL_RE = re.compile(
    r"Lattice\s*\{\s*(\w+)\s+(\w+)\s*\}\s*@(\d+)(?:\(\s*(\w+)\s*,\s*(\d+)\s*\))?"
)

_DTYPE_MAP = {
    "byte": np.uint8,
    "ubyte": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
}


@dataclass
class _Block:
    name: str
    dtype: str
    marker: int
    encoding: Optional[str]  # "HxByteRLE" | "HxZip" | None (raw)
    nbytes: Optional[int]  # on-disk (compressed) size when given


@dataclass
class AmiraLattice:
    """A decoded uniform image lattice."""

    volume: np.ndarray  # (nz, ny, nx)
    dims: Tuple[int, int, int]  # (nx, ny, nz)
    bbox: np.ndarray  # [x0, x1, y0, y1, z0, z1]
    block: str

    @property
    def spacing(self) -> np.ndarray:
        """Uniform voxel spacing (dx, dy, dz), node-centred: ``(hi-lo)/(dim-1)``."""
        lo = self.bbox[0::2]
        hi = self.bbox[1::2]
        dims = np.array(self.dims, dtype=float)
        denom = np.maximum(dims - 1.0, 1.0)
        return (hi - lo) / denom

    @property
    def origin(self) -> np.ndarray:
        """World coordinate of voxel (0,0,0) = lower bbox corner (x0, y0, z0)."""
        return self.bbox[0::2].astype(float)

    def world_to_voxel(self, coords: np.ndarray) -> np.ndarray:
        """Map world (x, y, z) coordinates to fractional voxel indices (ix, iy, iz)."""
        coords = np.asarray(coords, dtype=float)
        return (coords - self.origin) / self.spacing

    def world_to_index_zyx(self, coords: np.ndarray) -> np.ndarray:
        """Map world coords to fractional array indices in ``vol[z, y, x]`` order."""
        v = self.world_to_voxel(coords)  # (..., 3) as (ix, iy, iz)
        return v[..., ::-1]  # -> (iz, iy, ix)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------
def _read_header(path: PathLike) -> Tuple[Dict, List[_Block], int]:
    """Stream the ASCII header up to ``# Data section follows``.

    Returns ``(meta, blocks, data_start_byte)`` where ``data_start_byte`` is the
    byte offset of the first byte after the ``# Data section follows`` line.
    """
    dims: Optional[Tuple[int, int, int]] = None
    bbox: Optional[np.ndarray] = None
    blocks: List[_Block] = []
    data_start = 0
    with open(path, "rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"No '# Data section follows' found in {path}")
            line = raw.decode("latin-1")
            if dims is None:
                m = _DEFINE_LATTICE_RE.search(line)
                if m:
                    dims = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if bbox is None:
                m = _BBOX_RE.search(line)
                if m:
                    nums = [float(x) for x in m.group(1).split()][:6]
                    if len(nums) == 6:
                        bbox = np.array(nums, dtype=float)
            m = _DECL_RE.search(line)
            if m:
                blocks.append(
                    _Block(
                        dtype=m.group(1),
                        name=m.group(2),
                        marker=int(m.group(3)),
                        encoding=m.group(4),
                        nbytes=int(m.group(5)) if m.group(5) else None,
                    )
                )
            if line.strip().startswith("# Data section follows"):
                data_start = fh.tell()
                break

    if dims is None:
        raise ValueError(f"Missing 'define Lattice' in {path}")
    if bbox is None:
        # Fall back to a unit box spanning the voxel grid.
        nx, ny, nz = dims
        bbox = np.array([0, nx - 1, 0, ny - 1, 0, nz - 1], dtype=float)
    meta = {"dims": dims, "bbox": bbox}
    return meta, blocks, data_start


def _find_block(blocks: List[_Block], block: Union[str, int]) -> _Block:
    if isinstance(block, int):
        for b in blocks:
            if b.marker == block:
                return b
        raise KeyError(f"No lattice block with marker @{block}")
    for b in blocks:
        if b.name == block:
            return b
    raise KeyError(
        f"No lattice block named {block!r}; available: {[b.name for b in blocks]}"
    )


# ---------------------------------------------------------------------------
# Data-section navigation + decoders
# ---------------------------------------------------------------------------
def _seek_marker(fh, target_marker: int) -> None:
    """Position ``fh`` at the first byte after the ``@<marker>`` line."""
    fh.seek(0)
    # Skip to the data section first.
    while True:
        pos = fh.tell()
        raw = fh.readline()
        if not raw:
            raise ValueError("Reached EOF before data section")
        if raw.decode("latin-1").strip().startswith("# Data section follows"):
            break
    # Now find the @marker line.
    while True:
        raw = fh.readline()
        if not raw:
            raise ValueError(f"Marker @{target_marker} not found in data section")
        s = raw.decode("latin-1").strip()
        m = re.fullmatch(r"@(\d+)", s)
        if m and int(m.group(1)) == target_marker:
            return


def _decode_hxbyterle(data: bytes, out_size: int) -> np.ndarray:
    """Decode Amira ``HxByteRLE`` run-length-encoded bytes.

    Control byte ``c`` (the Amira/Avizo / ``ahds`` convention):

    * ``c & 0x80`` (high bit set) -> *literal*: copy the next ``c & 0x7f`` bytes
      verbatim;
    * otherwise (``c <= 127``) -> *run*: the next single byte repeated ``c`` times.

    Indexes the input as a ``bytes`` object (fast integer indexing) and writes
    into a preallocated numpy buffer via slice assignment, so it stays fast even
    for the ~37 MB -> ~2.3 GB expansion of a real segmentation.
    """
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    out = np.empty(out_size, dtype=np.uint8)
    src = data  # bytes: src[i] returns an int
    n = len(src)
    i = 0  # input index
    j = 0  # output index
    while j < out_size and i < n:
        c = src[i]; i += 1
        if c & 0x80:  # literal: copy the next (c & 0x7f) bytes verbatim
            count = c & 0x7F
            out[j : j + count] = np.frombuffer(src, dtype=np.uint8, count=count, offset=i)
            i += count
        else:  # run: repeat the next byte `c` times
            count = c
            out[j : j + count] = src[i]; i += 1
        j += count
    if j != out_size:
        raise ValueError(f"HxByteRLE decoded {j} bytes, expected {out_size}")
    return out


def read_amira_lattice(
    path: PathLike, block: Union[str, int] = "Labels"
) -> AmiraLattice:
    """Read one data block of an Amira binary uniform lattice.

    Parameters
    ----------
    block : str | int
        The lattice field to decode, by name (e.g. ``"Labels"``) or ``@`` marker
        (e.g. ``1``). Only one block is decoded, so other channels (e.g.
        ``Probability``) do not cost memory.

    Returns
    -------
    AmiraLattice with ``volume`` shaped ``(nz, ny, nx)`` (uint8 for ``byte``).
    """
    meta, blocks, _ = _read_header(path)
    if not blocks:
        raise ValueError(f"No 'Lattice {{ ... }} @N' declarations found in {path}")
    b = _find_block(blocks, block)

    np_dtype = _DTYPE_MAP.get(b.dtype)
    if np_dtype is None:
        raise NotImplementedError(f"Unsupported lattice dtype {b.dtype!r}")
    itemsize = np.dtype(np_dtype).itemsize

    nx, ny, nz = meta["dims"]
    n_vox = nx * ny * nz
    out_bytes = n_vox * itemsize

    with open(path, "rb") as fh:
        _seek_marker(fh, b.marker)
        if b.encoding == "HxByteRLE":
            comp = fh.read(b.nbytes) if b.nbytes else fh.read()
            flat = _decode_hxbyterle(comp, out_bytes)
        elif b.encoding in ("HxZip", "HxZ", "zip"):
            comp = fh.read(b.nbytes) if b.nbytes else fh.read()
            flat = np.frombuffer(zlib.decompress(comp), dtype=np.uint8)
        elif b.encoding is None:  # raw
            flat = np.frombuffer(fh.read(out_bytes), dtype=np.uint8)
        else:
            raise NotImplementedError(f"Unsupported lattice encoding {b.encoding!r}")

    if flat.size != out_bytes:
        raise ValueError(f"Decoded {flat.size} bytes, expected {out_bytes}")

    volume = flat.view(np_dtype).reshape(nz, ny, nx)  # X fastest -> vol[z, y, x]
    return AmiraLattice(volume=volume, dims=(nx, ny, nz), bbox=meta["bbox"], block=b.name)


def lattice_info(path: PathLike) -> Dict:
    """Return header metadata (dims, bbox, available blocks) without decoding data."""
    meta, blocks, _ = _read_header(path)
    return {
        "dims": meta["dims"],
        "bbox": meta["bbox"],
        "blocks": [
            {"name": b.name, "dtype": b.dtype, "marker": b.marker,
             "encoding": b.encoding, "nbytes": b.nbytes}
            for b in blocks
        ],
    }
