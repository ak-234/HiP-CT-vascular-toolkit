"""Tests for the Amira binary-lattice reader."""

import numpy as np
import pytest

from skeleton_analysis.io.amira_lattice import (
    _decode_hxbyterle,
    lattice_info,
    read_amira_lattice,
)


def _encode_hxbyterle(flat: np.ndarray) -> bytes:
    """Minimal HxByteRLE encoder (literal packets) for building fixtures.

    Uses the real Amira convention: a literal packet is control byte
    ``0x80 | count`` (high bit **set**) followed by ``count`` verbatim bytes.
    This produces a valid HxByteRLE stream the decoder must round-trip.
    """
    out = bytearray()
    i = 0
    n = len(flat)
    while i < n:
        count = min(127, n - i)
        out.append(0x80 | count)  # high bit set -> literal
        out.extend(int(v) for v in flat[i : i + count])
        i += count
    return bytes(out)


def test_hxbyterle_roundtrip_literals():
    data = np.arange(300, dtype=np.uint8) % 7
    enc = _encode_hxbyterle(data)
    dec = _decode_hxbyterle(enc, data.size)
    np.testing.assert_array_equal(dec, data)


def test_hxbyterle_runs():
    # Two runs: 5x value 9, then 3x value 2.  Run = control byte n (high bit
    # clear) followed by the repeated value.
    enc = bytes([5, 9, 3, 2])
    dec = _decode_hxbyterle(enc, 8)
    np.testing.assert_array_equal(dec, [9, 9, 9, 9, 9, 2, 2, 2])


def test_hxbyterle_mixed_literal_and_run():
    # Literal [1,2,3] (0x80|3) then run of 4x 7.
    enc = bytes([0x80 | 3, 1, 2, 3, 4, 7])
    dec = _decode_hxbyterle(enc, 7)
    np.testing.assert_array_equal(dec, [1, 2, 3, 7, 7, 7, 7])


def _write_lattice_am(path, volume_zyx, bbox, encoding="rle", block="Labels"):
    """Write a minimal Avizo binary byte-lattice .am with one block."""
    nz, ny, nx = volume_zyx.shape
    flat = volume_zyx.reshape(-1).astype(np.uint8)  # X fastest via C-order (z,y,x)
    if encoding == "rle":
        payload = _encode_hxbyterle(flat)
        decl = f"Lattice {{ byte {block} }} @1(HxByteRLE,{len(payload)})"
    else:
        payload = flat.tobytes()
        decl = f"Lattice {{ byte {block} }} @1"

    header = (
        "# Avizo BINARY-LITTLE-ENDIAN 3.0\n\n"
        f"define Lattice {nx} {ny} {nz}\n\n"
        "Parameters {\n"
        f'    Content "{nx}x{ny}x{nz} byte, uniform coordinates",\n'
        f"    BoundingBox {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]},\n"
        '    CoordType "uniform"\n'
        "}\n\n"
        f"{decl}\n\n"
        "# Data section follows\n"
        "@1\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("latin-1"))
        fh.write(payload)
        fh.write(b"\n")


def test_read_lattice_rle(tmp_path):
    # A distinctive 3x4x5 (z,y,x) volume.
    vol = (np.arange(3 * 4 * 5, dtype=np.uint8) % 5).reshape(3, 4, 5)
    bbox = [0.0, 8.0, 0.0, 6.0, 0.0, 4.0]  # spacing = span/(dim-1): x=8/4=2, y=6/3=2, z=4/2=2
    p = tmp_path / "lat.am"
    _write_lattice_am(p, vol, bbox, encoding="rle")

    lat = read_amira_lattice(p, block="Labels")
    assert lat.dims == (5, 4, 3)  # (nx, ny, nz)
    assert lat.volume.shape == (3, 4, 5)  # (nz, ny, nx)
    np.testing.assert_array_equal(lat.volume, vol)
    np.testing.assert_allclose(lat.bbox, bbox)
    np.testing.assert_allclose(lat.spacing, [2.0, 2.0, 2.0])


def test_read_lattice_raw(tmp_path):
    vol = (np.arange(2 * 2 * 2, dtype=np.uint8)).reshape(2, 2, 2)
    bbox = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    p = tmp_path / "lat_raw.am"
    _write_lattice_am(p, vol, bbox, encoding="raw")
    lat = read_amira_lattice(p)
    np.testing.assert_array_equal(lat.volume, vol)


def test_world_to_voxel(tmp_path):
    vol = np.zeros((3, 3, 3), dtype=np.uint8)
    bbox = [10.0, 30.0, 100.0, 120.0, 0.0, 20.0]  # spacing 10 each (span 20 / (3-1))
    p = tmp_path / "lat_map.am"
    _write_lattice_am(p, vol, bbox)
    lat = read_amira_lattice(p)
    # Lower corner -> voxel 0; upper corner -> voxel (dim-1).
    np.testing.assert_allclose(lat.world_to_voxel([10.0, 100.0, 0.0]), [0, 0, 0])
    np.testing.assert_allclose(lat.world_to_voxel([30.0, 120.0, 20.0]), [2, 2, 2])
    np.testing.assert_allclose(lat.world_to_voxel([20.0, 110.0, 10.0]), [1, 1, 1])
    # zyx ordering flips the axes.
    np.testing.assert_allclose(lat.world_to_index_zyx([30.0, 100.0, 0.0]), [0, 0, 2])


def test_lattice_info_lists_blocks(tmp_path):
    vol = np.zeros((2, 2, 2), dtype=np.uint8)
    p = tmp_path / "lat_info.am"
    _write_lattice_am(p, vol, [0, 1, 0, 1, 0, 1])
    info = lattice_info(p)
    assert info["dims"] == (2, 2, 2)
    assert info["blocks"][0]["name"] == "Labels"
    assert info["blocks"][0]["encoding"] == "HxByteRLE"


def test_unknown_block_raises(tmp_path):
    vol = np.zeros((2, 2, 2), dtype=np.uint8)
    p = tmp_path / "lat_x.am"
    _write_lattice_am(p, vol, [0, 1, 0, 1, 0, 1])
    with pytest.raises(KeyError):
        read_amira_lattice(p, block="Probability")
