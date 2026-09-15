"""Readers for the two AmiraMesh/Avizo flavours this pipeline produces.

* ASCII ``HxSpatialGraph`` -- the skeletonisation output (vertices, edges, points,
  per-point ``thickness``).
* Binary uniform ``Lattice`` -- the segmentation, stored either as raw scalar data
  or ``HxByteRLE``-compressed bytes.

The ASCII block parsing follows the same declaration/``@n`` scheme already used by
``adjust_thickness.py`` (which writes these files back out), so the two agree on
how blocks are located.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Matches e.g.  ``POINT { float thickness } @23``  or  ``EDGE { int[2] EdgeConnectivity } @3``
DECL_RE = re.compile(
    r"^\s*(VERTEX|EDGE|POINT)\s*\{\s*([A-Za-z]+)(?:\[(\d+)\])?\s+([A-Za-z0-9_]+)\s*\}\s*@(\d+)\s*$"
)
# Matches e.g. ``Lattice { byte Labels } @1(HxByteRLE,37709212)``
LATTICE_DECL_RE = re.compile(
    r"^\s*Lattice\s*\{\s*([A-Za-z0-9_]+)(?:\[(\d+)\])?\s+([A-Za-z0-9_]+)\s*\}\s*@(\d+)"
    r"(?:\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*\))?\s*$"
)
AT_RE = re.compile(r"^\s*@(\d+)\s*$")

DATA_SECTION = "# Data section follows"

#: Parameter this package writes into a graph it has corrected the units of, naming
#: the voxel size its coordinates are now in.
#:
#: Without it the correction is not idempotent, and silently so. The scale factor is
#: derived from the *segmentation's* bounding box, which does not change when a graph
#: is written; so re-loading a corrected graph beside the same uncorrected mask would
#: apply the same factor a second time, and the result would still sit inside the mask
#: and still pass every check. `read_voxel_stamp` is how the loader tells a graph that
#: has already been corrected from one that has not.
VOXEL_STAMP = "HiPCTVoxelSizeUm"
_STAMP_RE = re.compile(rf"^\s*{VOXEL_STAMP}\s+([0-9.eE+-]+)", re.MULTILINE)


def read_voxel_stamp(path) -> float | None:
    """The voxel size a graph declares its coordinates are in, or ``None``.

    Reads the header only -- these files run to hundreds of megabytes and the stamp
    is in the ``Parameters`` block, which is always near the top.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 16).decode("latin-1", errors="replace")
    except OSError:
        return None
    match = _STAMP_RE.search(head)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

_AM_DTYPES = {
    "byte": np.uint8,
    "ubyte": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
}


# --------------------------------------------------------------------------- #
# ASCII spatial graph
# --------------------------------------------------------------------------- #
@dataclass
class SpatialGraph:
    """An Amira ``HxSpatialGraph``. All coordinates and radii are in file units (um)."""

    path: Path
    n_vertex: int
    n_edge: int
    n_point: int
    vertices: np.ndarray  # (V, 3) float64
    connectivity: np.ndarray  # (E, 2) int64, indices into `vertices`
    n_edge_points: np.ndarray  # (E,) int64
    points: np.ndarray  # (P, 3) float64, concatenated per edge
    thickness: np.ndarray  # (P,) float64 -- RADIUS in um (see module docs)
    edge_attrs: dict = field(default_factory=dict)
    vertex_attrs: dict = field(default_factory=dict)
    # Per-POINT scalars other than the radius. Avizo writes none, but a graph whose
    # radii were measured rather than assigned needs to record *how* each one was
    # measured, and the only honest place for that is beside the value it describes.
    point_attrs: dict = field(default_factory=dict)

    @property
    def edge_offsets(self) -> np.ndarray:
        """(E+1,) start/stop offsets of each edge's run inside `points`."""
        return np.concatenate([[0], np.cumsum(self.n_edge_points)]).astype(np.int64)

    def edge_of_point(self) -> np.ndarray:
        """(P,) edge index owning each point."""
        return np.repeat(np.arange(self.n_edge), self.n_edge_points)

    def degree(self) -> np.ndarray:
        """(V,) vertex degree from the edge connectivity."""
        return np.bincount(self.connectivity.ravel(), minlength=self.n_vertex)


def _blocks(text: str) -> dict[int, str]:
    """Split the data section into ``{block_id: raw_text}``."""
    start = text.find(DATA_SECTION)
    if start < 0:
        raise ValueError("no '# Data section follows' marker")
    body = text[start + len(DATA_SECTION) :]

    marks = [
        (int(m.group(1)), m.start(), m.end())
        for m in re.finditer(r"^[ \t]*@(\d+)[ \t]*$", body, re.MULTILINE)
    ]
    out = {}
    for i, (bid, _, end) in enumerate(marks):
        stop = marks[i + 1][1] if i + 1 < len(marks) else len(body)
        out[bid] = body[end:stop]
    return out


def _to_array(raw: str, count: int, ncols: int, dtype) -> np.ndarray:
    vals = np.array(raw.split(), dtype=dtype)
    expect = count * ncols
    if vals.size != expect:
        raise ValueError(f"block has {vals.size} values, expected {expect}")
    return vals.reshape(count, ncols) if ncols > 1 else vals


def read_spatial_graph(path: str | Path) -> SpatialGraph:
    """Parse an ASCII AmiraMesh spatial graph."""
    path = Path(path)
    text = path.read_text(encoding="latin-1", errors="replace")
    header = text[: text.find(DATA_SECTION)]
    if "ASCII" not in header.split("\n", 1)[0]:
        raise ValueError(
            f"{path.name} is not an ASCII AmiraMesh file; export it as ASCII from Avizo"
        )

    counts = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"^\s*define\s+(\w+)\s+(\d+)", header, re.MULTILINE)
    }
    n_v, n_e, n_p = counts["VERTEX"], counts["EDGE"], counts["POINT"]
    sizes = {"VERTEX": n_v, "EDGE": n_e, "POINT": n_p}

    blocks = _blocks(text)
    decls = [DECL_RE.match(ln) for ln in header.split("\n")]
    decls = [m for m in decls if m]

    data: dict[tuple[str, str], np.ndarray] = {}
    for m in decls:
        kind, base, ncols, name, bid = m.groups()
        ncols = int(ncols) if ncols else 1
        bid = int(bid)
        if bid not in blocks:
            raise ValueError(f"declared block @{bid} for {kind} {name} has no data")
        dtype = np.int64 if base == "int" else np.float64
        data[(kind, name)] = _to_array(blocks[bid], sizes[kind], ncols, dtype)

    def take(kind, name, required=True):
        if (kind, name) in data:
            return data.pop((kind, name))
        if required:
            raise ValueError(f"{path.name} is missing {kind} {{ {name} }}")
        return None

    points = take("POINT", "EdgePointCoordinates")

    # The pipeline treats the sole POINT float field as radius; name it explicitly if
    # present, otherwise fall back to whatever single float field exists.
    if ("POINT", "thickness") in data:
        thickness = take("POINT", "thickness")
    else:
        floats = [k for k in data if k[0] == "POINT" and data[k].dtype.kind == "f"]
        if len(floats) != 1:
            raise ValueError(
                f"cannot identify the radius field; POINT float fields = {[k[1] for k in floats]}"
            )
        thickness = data.pop(floats[0])

    graph = SpatialGraph(
        path=path,
        n_vertex=n_v,
        n_edge=n_e,
        n_point=n_p,
        vertices=take("VERTEX", "VertexCoordinates"),
        connectivity=take("EDGE", "EdgeConnectivity").astype(np.int64),
        n_edge_points=take("EDGE", "NumEdgePoints").astype(np.int64).ravel(),
        points=points,
        thickness=np.asarray(thickness, dtype=np.float64).ravel(),
        edge_attrs={k[1]: v for k, v in data.items() if k[0] == "EDGE"},
        vertex_attrs={k[1]: v for k, v in data.items() if k[0] == "VERTEX"},
        point_attrs={k[1]: v for k, v in data.items() if k[0] == "POINT"},
    )

    total = int(graph.n_edge_points.sum())
    if total != n_p:
        raise ValueError(f"sum(NumEdgePoints)={total} but define POINT={n_p}")
    return graph


# --------------------------------------------------------------------------- #
# Binary uniform lattice
# --------------------------------------------------------------------------- #
@dataclass
class LatticeField:
    name: str
    dtype: type
    n_components: int
    block_id: int
    encoding: str | None  # 'HxByteRLE' or None for raw
    n_bytes: int | None  # compressed size when encoded
    data_offset: int = -1  # absolute byte offset of this field's payload


@dataclass(frozen=True)
class Material:
    """One entry of a label lattice's ``Materials`` block.

    ``value`` is the material's **index in the block**, which is what the lattice
    actually stores -- *not* the ``Id`` field Avizo also writes there. On
    ``32.04um_artery_left_right.labels.Regions.am`` the two disagree outright:
    ``Left_Tree`` and ``Right_Tree`` carry ``Id "9"`` and ``Id "10"``, and the voxels
    hold 1 and 2. Reading ``Id`` would look plausible and select nothing.
    """

    name: str
    value: int
    color: tuple[float, float, float] | None = None

    @property
    def is_exterior(self) -> bool:
        return self.value == 0


@dataclass
class LatticeInfo:
    path: Path
    dims: np.ndarray  # (3,) int, (nx, ny, nz) -- x varies fastest
    bbox: np.ndarray  # (6,) float, xmin xmax ymin ymax zmin zmax, VOXEL CENTRES
    fields: dict[str, LatticeField]
    #: The ``Materials`` block, in file order, when the lattice declares one. A plain
    #: binary mask has none, which is why this defaults to empty rather than to a
    #: synthesised Exterior/Interior pair: "no materials" and "one unnamed foreground"
    #: have to stay distinguishable, or every binary mask would grow a fake tree name.
    materials: tuple[Material, ...] = ()

    @property
    def regions(self) -> tuple[Material, ...]:
        """The foreground materials -- everything but Exterior.

        Two of them is the case this pipeline cares about: a mask whose left and right
        coronary trees were separated in Avizo rather than left for connectivity to
        guess at.
        """
        return tuple(m for m in self.materials if not m.is_exterior)

    def material(self, name: str) -> Material | None:
        """The material called `name`, case-insensitively, or ``None``."""
        want = name.strip().lower()
        for m in self.materials:
            if m.name.lower() == want:
                return m
        return None

    @property
    def spacing(self) -> np.ndarray:
        """(3,) voxel size derived from the bounding box (Amira 'uniform' = centre-to-centre)."""
        n = np.maximum(self.dims - 1, 1)
        return (self.bbox[1::2] - self.bbox[0::2]) / n

    @property
    def origin(self) -> np.ndarray:
        """(3,) physical position of voxel (0, 0, 0)."""
        return self.bbox[0::2].copy()

    @property
    def slice_nbytes(self) -> int:
        return int(self.dims[0]) * int(self.dims[1])


MATERIAL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\{")
MATERIAL_COLOR_RE = re.compile(r"^\s*Color\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")


def _matching_brace(text: str, open_at: int) -> int:
    """Index of the ``}`` closing the ``{`` at `open_at`, or -1."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_materials(header: str) -> tuple[Material, ...]:
    """The ``Materials`` block of a lattice header, in file order.

    Brace-counted rather than regexed as a whole, because ``Parameters`` nests several
    levels deep and an Avizo history log sits between ``Materials`` and the field
    declarations. Each material's ``value`` is its position in the block: Amira stores
    the *index*, and the sibling ``Id`` field is something else entirely (see
    :class:`Material`).
    """
    m = re.search(r"\bMaterials\s*\{", header)
    if m is None:
        return ()
    close = _matching_brace(header, m.end() - 1)
    if close < 0:
        return ()

    out: list[Material] = []
    depth, pending = 0, None
    for line in header[m.end():close].split("\n"):
        if depth == 0:
            hit = MATERIAL_RE.match(line)
            if hit:
                pending = Material(name=hit.group(1), value=len(out))
                out.append(pending)
        elif depth == 1 and pending is not None:
            color = MATERIAL_COLOR_RE.match(line)
            if color is not None and pending.color is None:
                # `Color r g b` is Avizo's own display colour; `&Color` on the next
                # line is the same thing base64-encoded, and is ignored.
                rgb = tuple(float(v) for v in color.groups())
                out[-1] = pending = Material(pending.name, pending.value, rgb)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            pending = None
    return tuple(out)


def read_lattice_header(path: str | Path) -> LatticeInfo:
    """Parse the header of a binary uniform-coordinate AmiraMesh lattice."""
    path = Path(path)
    with open(path, "rb") as fh:
        # The Avizo history log can be long.  Files written by some Avizo versions
        # include the comment below; raw lattices can instead go directly from the
        # declarations to the first ``@n`` marker.
        probe = fh.read(4 << 20)
    cut = probe.find(DATA_SECTION.encode())
    if cut < 0:
        first_marker = re.search(br"(?m)^@\d+\r?$", probe)
        if first_marker is None:
            raise ValueError(
                f"{path.name}: no '# Data section follows' or @n data marker "
                "found in first 4 MB"
            )
        cut = first_marker.start()
    header = probe[:cut].decode("latin-1")

    m = re.search(r"^\s*define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)", header, re.MULTILINE)
    if not m:
        raise ValueError(f"{path.name}: no 'define Lattice nx ny nz'")
    dims = np.array([int(g) for g in m.groups()], dtype=np.int64)

    m = re.search(r"BoundingBox\s+([-\d.eE+\s]+?),", header)
    if not m:
        raise ValueError(f"{path.name}: no BoundingBox")
    bbox = np.array(m.group(1).split(), dtype=np.float64)
    if bbox.size != 6:
        raise ValueError(f"{path.name}: BoundingBox has {bbox.size} values, expected 6")

    coord = re.search(r'CoordType\s+"(\w+)"', header)
    if coord and coord.group(1) != "uniform":
        raise ValueError(f"{path.name}: CoordType '{coord.group(1)}' is not supported (need uniform)")

    fields: dict[str, LatticeField] = {}
    for ln in header.split("\n"):
        m = LATTICE_DECL_RE.match(ln)
        if not m:
            continue
        base, ncomp, name, bid, enc, nbytes = m.groups()
        if base not in _AM_DTYPES:
            raise ValueError(f"{path.name}: unsupported lattice type '{base}'")
        fields[name] = LatticeField(
            name=name,
            dtype=_AM_DTYPES[base],
            n_components=int(ncomp) if ncomp else 1,
            block_id=int(bid),
            encoding=enc,
            n_bytes=int(nbytes) if nbytes else None,
        )
    if not fields:
        raise ValueError(f"{path.name}: no 'Lattice {{ ... }} @n' declarations")

    # Walk the data section, resolving each block's payload offset. Encoded blocks
    # declare their compressed size, so we can hop straight to the next @n marker.
    info = LatticeInfo(path=path, dims=dims, bbox=bbox, fields=fields,
                       materials=parse_materials(header))
    with open(path, "rb") as fh:
        pos = cut
        for fld in sorted(fields.values(), key=lambda f: f.block_id):
            marker = f"@{fld.block_id}".encode()
            fh.seek(pos)
            window = fh.read(4096)
            match = re.search(br"(?m)^" + re.escape(marker) + br"\r?$", window)
            if match is None:
                raise ValueError(f"{path.name}: data marker @{fld.block_id} not found near {pos}")
            # The regex stops immediately before the line-feed (and consumes a
            # carriage return when present), so the payload begins one byte later.
            start = pos + match.end()
            fh.seek(start)
            if fh.read(1) != b"\n":
                raise ValueError(
                    f"{path.name}: malformed data marker line @{fld.block_id} near {pos}"
                )
            start += 1
            fld.data_offset = start

            if fld.n_bytes is None:
                n = int(np.prod(dims)) * fld.n_components * np.dtype(fld.dtype).itemsize
            else:
                n = fld.n_bytes
            pos = start + n
    return info
