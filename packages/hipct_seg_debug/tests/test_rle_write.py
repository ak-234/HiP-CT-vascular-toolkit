"""The HxByteRLE encoder, checked against the decoder that was written without it.

``rle.py`` existed for months before anything could write the format, so it is a
genuinely independent implementation of the same spec. Every test here therefore
round-trips through :class:`~..rle.ByteRLELattice` rather than through a decoder
written alongside the encoder, which would only prove the two agree with each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug import amira, rle, rle_write


def _write_and_read(tmp_path, volume, name="lattice.am", spacing=2.0, field="Labels"):
    nz, ny, nx = volume.shape
    origin = np.array([11.0, 22.0, 33.0])
    bbox = np.empty(6)
    bbox[0::2] = origin
    bbox[1::2] = origin + (np.array([nx, ny, nz]) - 1) * spacing
    path = tmp_path / name
    report = rle_write.write_lattice(path, volume, bbox, field=field)
    info = amira.read_lattice_header(path)
    lattice = rle.ByteRLELattice(path, info.fields[field], info.dims,
                                 cache_dir=tmp_path / "cache")
    return report, info, lattice


# ----------------------------------------------------------------- the codec


@pytest.mark.parametrize("case", ["blobs", "zeros", "ones", "alternating",
                                  "long_runs", "single_voxel", "full_range"])
def test_round_trip(tmp_path, case):
    rng = np.random.default_rng(11)
    volumes = {
        "blobs": (rng.random((6, 20, 24)) < 0.15).astype(np.uint8),
        "zeros": np.zeros((4, 10, 12), np.uint8),
        "ones": np.ones((3, 8, 9), np.uint8),
        # The pathological case for RLE: every packet is a one-byte literal.
        "alternating": (np.indices((3, 8, 9)).sum(0) % 2).astype(np.uint8),
        # Runs far longer than the 127-byte packet limit, so packets must split.
        "long_runs": np.concatenate(
            [np.zeros(500, np.uint8), np.ones(400, np.uint8)]
        ).reshape(1, 30, 30),
        "single_voxel": np.eye(1, 60, 17, dtype=np.uint8).reshape(1, 6, 10),
        # Not a label field at all: the codec is byte-oriented, not binary.
        "full_range": rng.integers(0, 256, (3, 7, 11), dtype=np.uint8),
    }
    volume = volumes[case]
    _report, _info, lattice = _write_and_read(tmp_path, volume)

    for k in range(volume.shape[0]):
        assert np.array_equal(lattice.slice_z(k), volume[k]), f"slice {k}"
    assert np.array_equal(lattice.decode_sequential(volume.shape[0]), volume)


# ------------------------------------------------------------------ row bands


@pytest.mark.parametrize("case", ["blobs", "long_runs", "alternating"])
def test_a_row_band_decodes_to_what_the_whole_slice_holds(tmp_path, case):
    """The band reader is the hot path in `radius-perimeter`; it must agree exactly.

    A band is reached by walking the packets in front of it and discarding their
    output, which is a second decoding path through the same stream -- the one place
    a run straddling the band boundary could go wrong without any caller noticing.
    """
    rng = np.random.default_rng(3)
    volumes = {
        "blobs": (rng.random((5, 23, 17)) < 0.2).astype(np.uint8),
        # Runs far longer than one row, so packets straddle every band boundary.
        "long_runs": np.repeat(
            (rng.random((5, 23, 1)) < 0.5).astype(np.uint8), 17, axis=2
        ),
        "alternating": (np.indices((5, 23, 17)).sum(0) % 2).astype(np.uint8),
    }
    volume = volumes[case]
    _report, _info, lattice = _write_and_read(tmp_path, volume, name=f"{case}.am")

    for k in range(volume.shape[0]):
        whole = lattice.slice_z(k)
        for row0 in range(volume.shape[1]):
            for rows in (1, 4, 9):
                band = lattice.slice_rows(k, row0, row0 + rows)
                assert np.array_equal(band, whole[row0:row0 + rows]), (k, row0, rows)


def test_a_row_band_is_clipped_to_the_lattice_rather_than_padded(tmp_path):
    volume = (np.random.default_rng(4).random((3, 12, 9)) < 0.3).astype(np.uint8)
    _report, _info, lattice = _write_and_read(tmp_path, volume)

    assert lattice.slice_rows(1, 8, 40).shape == (4, 9)
    assert np.array_equal(lattice.slice_rows(1, 8, 40), volume[1, 8:])
    assert lattice.slice_rows(1, -5, 3).shape == (3, 9)
    assert np.array_equal(lattice.slice_rows(1, -5, 3), volume[1, 0:3])
    assert lattice.slice_rows(1, 7, 7).shape == (0, 9)


def test_a_row_band_outside_the_stack_is_an_error(tmp_path):
    volume = np.ones((2, 6, 6), np.uint8)
    _report, _info, lattice = _write_and_read(tmp_path, volume)
    with pytest.raises(IndexError):
        lattice.slice_rows(2, 0, 3)


def test_the_raw_reader_gives_the_same_bands(tmp_path):
    """`RawLattice` and `ByteRLELattice` are interchangeable, this included."""
    volume = np.arange(3 * 15 * 11, dtype=np.uint8).reshape(3, 15, 11)
    header = (
        "# AmiraMesh BINARY-LITTLE-ENDIAN 3.0\n\n"
        "define Lattice 11 15 3\n\n"
        "Parameters {\n"
        "    BoundingBox 0 10 0 14 0 2,\n"
        '    CoordType "uniform"\n'
        "}\n\n"
        "Lattice { byte data } @1\n\n"
        "@1\n"
    ).encode("ascii")
    path = tmp_path / "raw.am"
    path.write_bytes(header + volume.tobytes())
    info = amira.read_lattice_header(path)
    raw = rle.open_lattice(path, info.fields["data"], info.dims)

    assert isinstance(raw, rle.RawLattice)
    for k in range(volume.shape[0]):
        assert np.array_equal(raw.slice_rows(k, 3, 9), volume[k, 3:9])
    assert raw.slice_rows(0, 12, 40).shape == (3, 11)


def test_encode_output_decodes_with_the_existing_decoder():
    """No file, no header: just the byte stream through ``rle._decode``."""
    rng = np.random.default_rng(5)
    data = (rng.random(4096) < 0.1).astype(np.uint8)
    payload = np.frombuffer(rle_write.encode(data), dtype=np.uint8)
    back = rle._decode(np.ascontiguousarray(payload), np.int64(0), np.int64(0),
                       np.int64(len(data)))
    assert np.array_equal(back, data)


def test_empty_input_encodes_to_nothing():
    assert rle_write.encode(np.empty(0, dtype=np.uint8)) == b""


def test_a_vessel_like_mask_actually_compresses(tmp_path):
    """A sparse mask must come out far smaller, or the codec is doing nothing."""
    volume = np.zeros((8, 64, 64), np.uint8)
    volume[:, 30:34, 20:44] = 1
    report, _info, _lat = _write_and_read(tmp_path, volume)
    assert report["ratio"] > 20


def test_alternating_bytes_are_allowed_to_grow(tmp_path):
    """The honest worst case: RLE cannot win, and must not lose much either."""
    volume = (np.indices((2, 16, 16)).sum(0) % 2).astype(np.uint8)
    report, _info, _lat = _write_and_read(tmp_path, volume)
    assert report["encoded_bytes"] <= report["raw_bytes"] * 1.02


# ---------------------------------------------------------------- the header


def test_geometry_round_trips_bit_for_bit(tmp_path):
    """The bounding box is the geometry; it must come back with every bit intact.

    Written at ``repr`` precision rather than a fixed number of decimals: the real
    dataset's spacing is 65.98000587940216 um, and six decimals would displace the
    far corner of a 1500-voxel axis by most of a voxel.

    The *derived* spacing, ``(hi - lo) / (n - 1)``, cannot be exact and is not asked
    to be: the origin is 69 mm and the extent 0.26 mm, so the subtraction loses the
    bottom few bits to cancellation. That is a property of the AmiraMesh format --
    ``LatticeInfo.spacing`` does the same arithmetic on the real file -- so the
    bounding box is what is pinned exactly, and the spacing only to well under a
    picometre.
    """
    volume = np.zeros((5, 7, 9), np.uint8)
    spacing = 65.98000587940216
    nz, ny, nx = volume.shape
    origin = np.array([5096.95703125, 16544.48828125, 69328.484375])
    bbox = np.empty(6)
    bbox[0::2] = origin
    bbox[1::2] = origin + (np.array([nx, ny, nz]) - 1) * spacing

    path = tmp_path / "geom.am"
    rle_write.write_lattice(path, volume, bbox)
    info = amira.read_lattice_header(path)

    assert np.array_equal(info.dims, [nx, ny, nz])
    assert np.array_equal(info.bbox, bbox)  # exact, not approximate
    assert np.array_equal(info.origin, origin)
    assert np.allclose(info.spacing, spacing, rtol=0, atol=1e-9)


def test_the_declared_size_matches_the_payload(tmp_path):
    volume = (np.random.default_rng(1).random((4, 12, 10)) < 0.3).astype(np.uint8)
    report, info, _lat = _write_and_read(tmp_path, volume)
    field = info.fields["Labels"]
    assert field.encoding == "HxByteRLE"
    assert field.n_bytes == report["encoded_bytes"]
    assert field.data_offset + field.n_bytes == (tmp_path / "lattice.am").stat().st_size


def test_field_name_is_honoured(tmp_path):
    volume = np.zeros((2, 3, 4), np.uint8)
    _report, info, _lat = _write_and_read(tmp_path, volume, field="Probability")
    assert set(info.fields) == {"Probability"}


def test_written_lattice_survives_the_frame_builder(tmp_path):
    """What matters downstream: ``WorldFrame`` must accept what we wrote."""
    from hipct_seg_debug.frame import WorldFrame

    volume = np.zeros((5, 7, 9), np.uint8)
    _report, info, _lat = _write_and_read(tmp_path, volume, spacing=66.0)
    frame = WorldFrame.from_inputs((10, 14, 18), 33.0, info)
    assert tuple(frame.bin_factor) == (2, 2, 2)
    assert np.allclose(frame.seg_spacing, 66.0)


def test_uncompressed_lattice_without_data_section_comment_is_read_lazily(tmp_path):
    """Avizo can write raw lattices with ``@1`` immediately after the header."""
    volume = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    header = (
        "# AmiraMesh BINARY-LITTLE-ENDIAN 3.0\n\n"
        "define Lattice 4 3 2\n\n"
        "Parameters {\n"
        "    BoundingBox 0 3 0 2 0 1,\n"
        '    CoordType "uniform"\n'
        "}\n\n"
        "Lattice { byte data } @1\n\n"
        "@1\n"
    ).encode("ascii")
    path = tmp_path / "raw.am"
    path.write_bytes(header + volume.tobytes())

    info = amira.read_lattice_header(path)
    assert np.array_equal(info.dims, [4, 3, 2])
    assert info.fields["data"].data_offset == len(header)
    assert info.fields["data"].encoding is None

    lattice = rle.open_lattice(path, info.fields["data"], info.dims)
    assert isinstance(lattice, rle.RawLattice)
    assert np.array_equal(lattice.slice_z(0), volume[0])
    assert np.array_equal(lattice.slice_z(1), volume[1])
    assert np.array_equal(lattice.decode_sequential(2), volume)


def test_command_loader_uses_a_sole_data_field_for_default_labels(tmp_path, capsys):
    from types import SimpleNamespace

    from hipct_seg_debug.edit.__main__ import _open_lattice

    volume = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    header = (
        "# AmiraMesh BINARY-LITTLE-ENDIAN 3.0\n\n"
        "define Lattice 4 3 2\n\n"
        "Parameters {\n"
        "    BoundingBox 0 3 0 2 0 1,\n"
        '    CoordType "uniform"\n'
        "}\n\n"
        "Lattice { byte data } @1\n\n"
        "@1\n"
    ).encode("ascii")
    path = tmp_path / "raw.am"
    path.write_bytes(header + volume.tobytes())
    args = SimpleNamespace(
        seg=str(path), labels_field="Labels", voxel_um=1.0, edits=None
    )

    labels, _frame, opened = _open_lattice(args)

    assert opened == str(path)
    assert labels.field.name == "data"
    assert args.labels_field == "Labels"
    assert "using sole field 'data'" in capsys.readouterr().out


def test_a_written_lattice_can_be_re_encoded(tmp_path):
    """Decode -> encode -> decode must be a fixed point, not merely close."""
    volume = (np.random.default_rng(9).random((6, 14, 16)) < 0.2).astype(np.uint8)
    _report, _info, first = _write_and_read(tmp_path, volume, name="a.am")
    decoded = np.stack([first.slice_z(k) for k in range(volume.shape[0])])

    _report2, _info2, second = _write_and_read(tmp_path, decoded, name="b.am")
    again = np.stack([second.slice_z(k) for k in range(volume.shape[0])])
    assert np.array_equal(decoded, again)
    assert np.array_equal(again, volume)


def test_non_3d_input_is_refused(tmp_path):
    with pytest.raises(ValueError, match="3-D"):
        rle_write.write_lattice(tmp_path / "x.am", np.zeros((4, 4), np.uint8),
                                np.zeros(6))


@pytest.mark.slow
def test_the_real_lattice_survives_a_full_round_trip(tmp_path):
    """Decode all 2.34 GB, re-encode, and read it back plane for plane.

    The claim being checked is that a corrected mask written out here is a *drop-in
    replacement* for the source, not an approximation of it -- so this compares
    against the real Avizo file rather than a synthetic one.
    """

    from hipct_seg_debug.edit.lattice import decode_volume

    from .realdata import REAL_SEG, SEG_REASON

    seg = REAL_SEG
    if not seg.is_file():
        pytest.skip(SEG_REASON)

    info = amira.read_lattice_header(seg)
    lattice = rle.ByteRLELattice(seg, info.fields["Labels"], info.dims)
    volume = decode_volume(lattice)

    nz, ny, nx = volume.shape
    bbox = np.empty(6)
    bbox[0::2] = info.origin
    bbox[1::2] = info.origin + (np.array([nx, ny, nz]) - 1) * info.spacing
    out = tmp_path / "reencoded.am"
    report = rle_write.write_lattice(out, volume, bbox)

    # Within a fraction of a percent of what Avizo itself produced.
    assert report["encoded_bytes"] < info.fields["Labels"].n_bytes * 1.01

    back = amira.read_lattice_header(out)
    reread = rle.ByteRLELattice(out, back.fields["Labels"], back.dims,
                                cache_dir=tmp_path / "cache")
    for k in range(0, nz, 53):
        assert np.array_equal(reread.slice_z(k), volume[k]), f"plane {k}"
    assert np.array_equal(back.bbox, bbox)


def test_extra_parameters_are_indented_into_the_block(tmp_path):
    volume = np.zeros((2, 2, 2), np.uint8)
    nz, ny, nx = volume.shape
    path = tmp_path / "extra.am"
    rle_write.write_lattice(path, volume, np.array([0, 1, 0, 1, 0, 1.0]),
                            extra='Note "corrected by hand",')
    info = amira.read_lattice_header(path)
    assert "Labels" in info.fields
    assert b'Note "corrected by hand"' in path.read_bytes()[:2000]
