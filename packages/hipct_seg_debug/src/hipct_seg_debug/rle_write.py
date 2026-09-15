"""Write an ``HxByteRLE``-compressed Amira byte lattice.

:mod:`.rle` reads this format and could not write it, which was fine while the
segmentation was strictly an input. Once a session can *correct* the mask, the
correction has to leave in a form the rest of the world accepts -- and for this
project that means an Amira label lattice, so a repaired segmentation is a drop-in
replacement input for Avizo and for the existing surface pipeline.

The codec, inverted from :mod:`.rle`: a control byte ``n``; if ``n > 127`` the next
``n - 128`` bytes are literal, otherwise the single following byte repeats ``n``
times. Both run lengths therefore cap at 127, and neither may be zero.

:func:`encode` is greedy -- a run of three or more identical bytes becomes a repeat
packet, anything else accumulates into a literal packet. Three is the break-even
point once the cost of interrupting a literal run is counted, and a binary vessel
mask is overwhelmingly long runs of zero.

Volumes are encoded **one z plane at a time and concatenated**. That bounds peak
memory to one plane's worth of scratch instead of a second copy of the volume, and
costs a few bytes per plane, because a run that would have straddled a plane
boundary is emitted as two. The decoder is a plain byte stream and does not care
where packets begin.

Measured against the source file on LADAF-2024-28: decoding all 2.34 GB and
re-encoding it gives **37,725,961 bytes where Avizo wrote 37,709,212** -- 0.04%
larger, and byte-identical when decoded back. So a mask corrected here and written
out is a drop-in replacement for the original, not an approximation of it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit

MAGIC = "# Avizo BINARY-LITTLE-ENDIAN 3.0"
MAX_RUN = 127


@njit(cache=True)
def _encode_into(buf: np.ndarray, out: np.ndarray) -> np.int64:
    """Greedy byte RLE. Returns how many bytes of `out` were used.

    `out` must hold at least ``2 * len(buf) + 8``: every output byte is either a
    packet's control byte or one of its data bytes, data bytes never exceed the
    input, and no packet consumes fewer than one input byte.
    """
    n = buf.shape[0]
    k = np.int64(0)
    w = np.int64(0)

    while k < n:
        # How far the byte at k repeats, capped at one packet's worth.
        run = np.int64(1)
        while k + run < n and buf[k + run] == buf[k] and run < MAX_RUN:
            run += 1

        if run >= 3:
            out[w] = run
            out[w + 1] = buf[k]
            w += 2
            k += run
            continue

        # Otherwise gather literals until a repeat worth switching to shows up.
        start = k
        lit = np.int64(0)
        while k < n and lit < MAX_RUN:
            ahead = np.int64(1)
            while k + ahead < n and buf[k + ahead] == buf[k] and ahead < 3:
                ahead += 1
            if ahead >= 3:
                break
            k += 1
            lit += 1

        out[w] = 128 + lit
        w += 1
        for i in range(lit):
            out[w + i] = buf[start + i]
        w += lit

    return w


def encode(data: np.ndarray) -> bytes:
    """Compress a flat ``uint8`` array."""
    buf = np.ascontiguousarray(np.asarray(data, dtype=np.uint8).ravel())
    if buf.size == 0:
        return b""
    out = np.empty(2 * buf.size + 8, dtype=np.uint8)
    used = _encode_into(buf, out)
    return out[:used].tobytes()


def encode_volume(volume_zyx: np.ndarray, progress=None) -> bytes:
    """Compress a ``(nz, ny, nx)`` volume plane by plane, x varying fastest.

    That axis order is the lattice's own: :meth:`~.rle.ByteRLELattice.slice_z`
    reshapes its decoded bytes to ``(ny, nx)``, so a C-order ravel of each plane
    reproduces the stream exactly.
    """
    volume = np.asarray(volume_zyx, dtype=np.uint8)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D (nz, ny, nx) volume, got {volume.shape}")
    chunks = []
    t0 = time.time()
    for k in range(volume.shape[0]):
        chunks.append(encode(volume[k]))
        if progress is not None and k % 50 == 0:
            progress(k, volume.shape[0], time.time() - t0)
    if progress is not None:
        progress(volume.shape[0], volume.shape[0], time.time() - t0)
    return b"".join(chunks)


def header(dims, bbox, *, field: str = "Labels", n_bytes: int,
           colormap: str = "labels.am", extra: str = "") -> str:
    """The AmiraMesh header for one ``HxByteRLE`` byte field.

    `dims` is ``(nx, ny, nz)`` and `bbox` is the six voxel-*centre* bounds
    ``xmin xmax ymin ymax zmin zmax``, matching :class:`~.amira.LatticeInfo`. The
    bounds are written at full ``repr`` precision so a decode/encode round trip
    reproduces ``frame.seg_spacing`` bit for bit rather than to six decimals.

    Avizo's history log is deliberately not reproduced. It is a provenance record
    of the operations *that file* went through, and copying it onto a different
    volume would assert a lineage that is not true; the ``Colormap`` entry, which
    is what makes Avizo treat the field as a label field, is kept.
    """
    nx, ny, nz = (int(v) for v in dims)
    bounds = " ".join(repr(float(v)) for v in np.asarray(bbox, dtype=np.float64).ravel())
    body = extra.rstrip()
    if body:
        body = "\n" + "\n".join("    " + ln if ln.strip() else ln
                                for ln in body.splitlines())
    return (
        f"{MAGIC}\n\n\n"
        f"define Lattice {nx} {ny} {nz}\n\n"
        f"Parameters {{{body}\n"
        f'    Colormap "{colormap}",\n'
        f'    Content "{nx}x{ny}x{nz} byte, uniform coordinates",\n'
        f"    BoundingBox {bounds},\n"
        f'    CoordType "uniform"\n'
        f"}}\n\n"
        f"Lattice {{ byte {field} }} @1(HxByteRLE,{int(n_bytes)})\n\n"
        f"# Data section follows\n@1\n"
    )


def write_lattice(path, volume_zyx: np.ndarray, bbox, *, field: str = "Labels",
                  colormap: str = "labels.am", extra: str = "",
                  progress=None) -> dict:
    """Encode `volume_zyx` and write a complete ``.am`` lattice.

    Returns a small report -- raw bytes, compressed bytes, ratio and seconds --
    because the compression ratio is the quickest check that the volume being
    written is the one intended: a mask that suddenly compresses 3:1 instead of
    60:1 is noise, not anatomy.
    """
    volume = np.asarray(volume_zyx, dtype=np.uint8)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D (nz, ny, nx) volume, got {volume.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = volume.shape

    t0 = time.time()
    payload = encode_volume(volume, progress=progress)
    text = header((nx, ny, nz), bbox, field=field, n_bytes=len(payload),
                  colormap=colormap, extra=extra)
    with open(path, "wb") as fh:
        fh.write(text.encode("latin-1"))
        fh.write(payload)

    raw = int(volume.size)
    return {
        "path": path,
        "raw_bytes": raw,
        "encoded_bytes": len(payload),
        "ratio": raw / max(len(payload), 1),
        "seconds": time.time() - t0,
    }


def describe(report: dict) -> str:
    return (f"{report['path'].name}: {report['raw_bytes'] / 1e9:.2f} GB -> "
            f"{report['encoded_bytes'] / 1e6:.1f} MB "
            f"({report['ratio']:.0f}:1) in {report['seconds']:.1f}s")
