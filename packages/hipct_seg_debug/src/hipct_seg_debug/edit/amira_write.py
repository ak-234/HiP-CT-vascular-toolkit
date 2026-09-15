"""Write an ASCII Amira ``HxSpatialGraph``.

The parent package reads these and never writes them, which is why this lives
here rather than in ``amira.py``.

One detail matters more than the rest: **the ``Parameters { ... }`` block is
carried over from the source file verbatim**. It holds the
``TransformationMatrix`` that puts the graph in the scan's coordinate frame,
along with units and channel colours. The upstream editing toolkit drops it
(``write_to_conventional_am`` emits a bare ``ContentType "HxSpatialGraph"``), and
a graph that has silently lost its transform still opens in Avizo and still looks
right -- it is simply in the wrong place, which is a bad thing to discover after
meshing it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..amira import VOXEL_STAMP, SpatialGraph

HEADER = "# AmiraMesh 3D ASCII 2.0"
DEFAULT_PARAMETERS = 'Parameters {\n    ContentType "HxSpatialGraph"\n}'

_PARAM_RE = re.compile(r"^Parameters\s*\{", re.MULTILINE)
_STAMP_LINE_RE = re.compile(rf"^[ \t]*{VOXEL_STAMP}\s+[^\n]*\n?", re.MULTILINE)


def extract_parameters(path: str | Path) -> str | None:
    """Pull the ``Parameters { ... }`` block out of an existing ``.am``.

    Brace-counted rather than regex-matched to the end: the block nests
    (``Parameters { Materials { Exterior { ... } } }``) and a non-greedy match to
    the first ``}`` would truncate it.
    """
    try:
        text = Path(path).read_text(encoding="latin-1", errors="replace")
    except OSError:
        return None
    match = _PARAM_RE.search(text)
    if match is None:
        return None
    depth = 0
    for i in range(match.start(), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start(): i + 1]
    return None


def stamp_voxel_um(params: str, voxel_um: float | None) -> str:
    """Return `params` with the voxel-size stamp set to `voxel_um`.

    The stamp says what units this file's coordinates are in, so that a unit
    correction applied on load is not applied to its own output a second time -- see
    :data:`~..amira.VOXEL_STAMP`. An existing stamp is replaced rather than repeated:
    the value belongs to the file being written, not to whichever file the
    ``Parameters`` block was lifted from.
    """
    if voxel_um is None:
        return params
    body = _STAMP_LINE_RE.sub("", params).rstrip()
    if not body.endswith("}"):
        return params
    return f"{body[:-1].rstrip()}\n    {VOXEL_STAMP} {float(voxel_um):.6g}\n}}"


def _fmt_rows(arr: np.ndarray, integer: bool) -> str:
    arr = np.atleast_2d(arr.T).T if arr.ndim == 1 else arr
    if integer:
        return "\n".join(" ".join(str(int(v)) for v in row) for row in arr)
    # repr-style float formatting: enough digits to round-trip a float64 without
    # writing 17 digits for values that do not need them.
    return "\n".join(" ".join(f"{float(v):.9g}" for v in row) for row in arr)


def write_spatial_graph(
    graph: SpatialGraph,
    path: str | Path,
    *,
    parameters_from: str | Path | None = None,
    radius_field: str = "thickness",
    voxel_um: float | None = None,
) -> Path:
    """Write `graph` as an ASCII ``HxSpatialGraph``.

    `parameters_from` names a file to lift the ``Parameters`` block from; it
    defaults to the graph's own ``path``, which is the file it was read from.

    `voxel_um` records, in that block, the voxel size these coordinates are in. Pass
    it whenever the session corrected the units of what it loaded -- without it the
    correction is not idempotent and the next load repeats it, undetectably. Omitted,
    any stamp inherited from `parameters_from` is left exactly as it was.
    """
    path = Path(path)
    source = parameters_from if parameters_from is not None else graph.path
    params = extract_parameters(source) if source else None
    if params is None:
        params = DEFAULT_PARAMETERS
    params = stamp_voxel_um(params, voxel_um)

    n_v, n_e, n_p = graph.n_vertex, graph.n_edge, graph.n_point
    if int(graph.n_edge_points.sum()) != n_p:
        raise ValueError(
            f"sum(NumEdgePoints)={int(graph.n_edge_points.sum())} but POINT={n_p}"
        )
    if len(graph.thickness) != n_p:
        raise ValueError(f"{len(graph.thickness)} radii for {n_p} points")

    # Fixed blocks first, then whatever extra per-edge / per-vertex fields the
    # graph carries, so a file written here reads back with the same attributes.
    blocks: list[tuple[str, str, np.ndarray, bool]] = [
        ("VERTEX", "float[3] VertexCoordinates", graph.vertices, False),
        ("EDGE", "int[2] EdgeConnectivity", graph.connectivity, True),
        ("EDGE", "int NumEdgePoints", graph.n_edge_points, True),
        ("POINT", "float[3] EdgePointCoordinates", graph.points, False),
        ("POINT", f"float {radius_field}", graph.thickness, False),
    ]
    for name, arr in sorted(graph.edge_attrs.items()):
        arr = np.asarray(arr)
        if arr.ndim > 1 and arr.shape[1] != 1:
            continue
        arr = arr.ravel()
        if len(arr) != n_e:
            continue
        integer = arr.dtype.kind in "iub"
        blocks.append(("EDGE", f"{'int' if integer else 'float'} {name}", arr, integer))
    for name, arr in sorted(graph.vertex_attrs.items()):
        arr = np.asarray(arr)
        if arr.ndim > 1 and arr.shape[1] != 1:
            continue
        arr = arr.ravel()
        if len(arr) != n_v:
            continue
        integer = arr.dtype.kind in "iub"
        blocks.append(("VERTEX", f"{'int' if integer else 'float'} {name}", arr, integer))
    # Per-point scalars beside the radius. `radius_field` is already written above,
    # so it is skipped here rather than emitted twice under two block ids.
    for name, arr in sorted(graph.point_attrs.items()):
        if name in (radius_field, "EdgePointCoordinates"):
            continue
        arr = np.asarray(arr)
        if arr.ndim > 1 and arr.shape[1] != 1:
            continue
        arr = arr.ravel()
        if len(arr) != n_p:
            continue
        integer = arr.dtype.kind in "iub"
        blocks.append(("POINT", f"{'int' if integer else 'float'} {name}", arr, integer))

    lines = [
        HEADER,
        "",
        f"define VERTEX {n_v}",
        f"define EDGE {n_e}",
        f"define POINT {n_p}",
        "",
        params,
        "",
    ]
    for i, (kind, decl, _arr, _int) in enumerate(blocks, start=1):
        lines.append(f"{kind} {{ {decl} }} @{i}")
    lines += ["", "# Data section follows"]
    for i, (_kind, _decl, arr, integer) in enumerate(blocks, start=1):
        lines.append(f"@{i}")
        lines.append(_fmt_rows(np.asarray(arr), integer))
        lines.append("")

    path.write_text("\n".join(lines), encoding="latin-1")
    return path
