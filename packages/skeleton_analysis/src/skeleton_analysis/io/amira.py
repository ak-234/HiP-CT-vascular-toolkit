"""Read and write Amira/Avizo ASCII ``HxSpatialGraph`` (``.am``) files.

This module is the backbone of the package. It replaces the MATLAB
``ultimate_amira_read.m`` / ``make_dict.m`` generic reader **and** all the
folder-specific hard-coded readers/writers (``run_ordering.m`` read_data /
write_back_*, ``write_corrected_data.m``, the inline I/O in
``add_spatial_graphs.m``, ``write_amira_file.m`` and the ``VVToAmira_v3.m``
writer).

Design notes
------------
* The reader is *generic*: fields are discovered from their declaration lines
  (``DOMAIN { dtype[dim] Name } @N``) exactly like ``ultimate_amira_read`` did,
  so the ``@N`` numbering is never assumed to be fixed.
* The full ``Parameters { ... }`` block is captured **verbatim** so that written
  files round-trip cleanly back into Amira/Avizo (preserving units,
  ``TransformationMatrix``, colours, history log). The MATLAB writers discarded
  this, which lost the spatial transform — a real-world fragility we fix here.
* ``.am`` files are read/written as ``latin-1`` so the raw parameter bytes
  survive a round-trip losslessly regardless of their true text encoding.
* Node IDs in ``EdgeConnectivity`` are kept **0-based** (as stored in the file);
  the MATLAB ``+1`` offset used only for building ``digraph`` objects lives in
  :mod:`skeleton_analysis.graph.build`, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# Domains in an HxSpatialGraph, in canonical order.
_DOMAINS = ("VERTEX", "EDGE", "POINT")

# Field declaration line, e.g. ``EDGE { int[2] EdgeConnectivity } @3``.
_DECL_RE = re.compile(
    r"(VERTEX|EDGE|POINT)\s*\{\s*([A-Za-z_]\w*)\s*(?:\[(\d+)\])?\s+([A-Za-z_]\w*)\s*\}\s*@(\d+)"
)
_MARKER_RE = re.compile(r"@(\d+)\s*$")
_DEFINE_RE = {d: re.compile(rf"define\s+{d}\s+(\d+)") for d in _DOMAINS}

_DEFAULT_HEADER = "# AmiraMesh 3D ASCII 2.0"
_DEFAULT_PARAMETERS = 'Parameters {\n    ContentType "HxSpatialGraph"\n}'

# Standard field names used throughout the package.
F_VERTEX_COORDS = "VertexCoordinates"
F_EDGE_CONNECTIVITY = "EdgeConnectivity"
F_NUM_EDGE_POINTS = "NumEdgePoints"
F_POINT_COORDS = "EdgePointCoordinates"
F_THICKNESS = "thickness"


def _is_float_dtype(dtype_token: str) -> bool:
    """Amira dtype token -> is it a floating point type?"""
    return dtype_token.lower().startswith(("float", "double"))


@dataclass
class SpatialGraph:
    """An Amira/Avizo spatial graph held as plain numpy arrays.

    Fields are stored per domain in three insertion-ordered dicts. Each value is
    a ``(count,)`` array for scalar fields or a ``(count, dim)`` array for vector
    fields (e.g. ``VertexCoordinates`` is ``(n_vertices, 3)``). Integer vs float
    dtype is preserved and drives how the field is re-declared and written.
    """

    vertex_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    edge_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    point_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    # (domain, name) in declaration order; drives the write order + @N numbering.
    field_order: List[Tuple[str, str]] = field(default_factory=list)
    header: str = _DEFAULT_HEADER
    raw_parameters: Optional[str] = None

    # -- domain helpers ---------------------------------------------------
    def _domain_dict(self, domain: str) -> Dict[str, np.ndarray]:
        return {
            "VERTEX": self.vertex_fields,
            "EDGE": self.edge_fields,
            "POINT": self.point_fields,
        }[domain]

    def set_field(self, domain: str, name: str, values: np.ndarray) -> None:
        """Add or replace a field in ``domain`` ('VERTEX'|'EDGE'|'POINT')."""
        domain = domain.upper()
        if domain not in _DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}")
        self._domain_dict(domain)[name] = np.asarray(values)
        if (domain, name) not in self.field_order:
            self.field_order.append((domain, name))

    def set_edge_field(self, name: str, values: np.ndarray) -> None:
        self.set_field("EDGE", name, values)

    def set_vertex_field(self, name: str, values: np.ndarray) -> None:
        self.set_field("VERTEX", name, values)

    def set_point_field(self, name: str, values: np.ndarray) -> None:
        self.set_field("POINT", name, values)

    def copy(self) -> "SpatialGraph":
        """Deep copy (independent field arrays) — for snapshotting pipeline stages."""
        return SpatialGraph(
            vertex_fields={k: np.array(v, copy=True) for k, v in self.vertex_fields.items()},
            edge_fields={k: np.array(v, copy=True) for k, v in self.edge_fields.items()},
            point_fields={k: np.array(v, copy=True) for k, v in self.point_fields.items()},
            field_order=list(self.field_order),
            header=self.header,
            raw_parameters=self.raw_parameters,
        )

    # -- counts -----------------------------------------------------------
    @staticmethod
    def _count(d: Dict[str, np.ndarray]) -> int:
        for arr in d.values():
            return int(arr.shape[0])
        return 0

    @property
    def n_vertices(self) -> int:
        return self._count(self.vertex_fields)

    @property
    def n_edges(self) -> int:
        return self._count(self.edge_fields)

    @property
    def n_points(self) -> int:
        return self._count(self.point_fields)

    # -- standard-field convenience accessors -----------------------------
    @property
    def vertex_coords(self) -> np.ndarray:
        return self.vertex_fields[F_VERTEX_COORDS]

    @property
    def edge_connectivity(self) -> np.ndarray:
        return self.edge_fields[F_EDGE_CONNECTIVITY]

    @property
    def num_edge_points(self) -> np.ndarray:
        return self.edge_fields[F_NUM_EDGE_POINTS]

    @property
    def point_coords(self) -> np.ndarray:
        return self.point_fields[F_POINT_COORDS]

    @property
    def thickness(self) -> np.ndarray:
        return self.point_fields[F_THICKNESS]

    @thickness.setter
    def thickness(self, values: np.ndarray) -> None:
        self.set_point_field(F_THICKNESS, values)

    # -- ordered iteration for writing ------------------------------------
    def _ordered_fields(self) -> List[Tuple[str, str, np.ndarray]]:
        out: List[Tuple[str, str, np.ndarray]] = []
        seen = set()
        for domain, name in self.field_order:
            arr = self._domain_dict(domain).get(name)
            if arr is not None:
                out.append((domain, name, arr))
                seen.add((domain, name))
        # Append any fields added without touching field_order (defensive).
        for domain in _DOMAINS:
            for name, arr in self._domain_dict(domain).items():
                if (domain, name) not in seen:
                    out.append((domain, name, arr))
        return out

    # -- consistency check (ported from the MATLAB reader warnings) -------
    def check_consistency(self) -> List[str]:
        """Return a list of human-readable consistency warnings (empty if OK)."""
        warnings: List[str] = []
        if F_NUM_EDGE_POINTS in self.edge_fields:
            total = int(np.sum(self.num_edge_points))
            if total != self.n_points:
                warnings.append(
                    f"sum(NumEdgePoints)={total} != n_points={self.n_points}"
                )
        if F_EDGE_CONNECTIVITY in self.edge_fields:
            if self.edge_connectivity.shape[0] != self.n_edges:
                warnings.append("EdgeConnectivity length != n_edges")
        return warnings


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _capture_parameters(text: str) -> Tuple[Optional[str], int]:
    """Capture the verbatim ``Parameters { ... }`` block.

    Returns ``(block_text, end_index)`` where ``end_index`` is the character
    offset just past the closing brace. Braces inside double-quoted strings are
    ignored so base64 blobs / quoted paths cannot fool the matcher.
    """
    m = re.search(r"^Parameters\b", text, flags=re.MULTILINE)
    if m is None:
        return None, 0
    start = m.start()
    brace = text.find("{", start)
    if brace == -1:
        return None, 0
    depth = 0
    in_string = False
    i = brace
    while i < len(text):
        c = text[i]
        if in_string:
            if c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                return text[start:end], end
        i += 1
    return None, 0  # unbalanced; treat as absent


def read_amira(path: PathLike) -> SpatialGraph:
    """Parse an Amira/Avizo ASCII spatial-graph ``.am`` file.

    Generic: every field is discovered from its declaration line, so any number
    of VERTEX/EDGE/POINT attributes in any ``@N`` order are handled.
    """
    text = Path(path).read_text(encoding="latin-1")

    # Header line (the first non-empty line, e.g. "# Avizo 3D ASCII 3.0").
    header = _DEFAULT_HEADER
    for line in text.splitlines():
        if line.strip():
            header = line.strip()
            break

    # define VERTEX/EDGE/POINT counts.
    counts: Dict[str, int] = {}
    for domain, rx in _DEFINE_RE.items():
        m = rx.search(text)
        if m is None:
            raise ValueError(f"Missing 'define {domain}' in {path}")
        counts[domain] = int(m.group(1))

    # Verbatim Parameters block; everything after it holds declarations + data.
    raw_parameters, param_end = _capture_parameters(text)
    rest = text[param_end:] if param_end else text

    # Split declarations from the data section.
    marker = "# Data section follows"
    idx = rest.find(marker)
    if idx == -1:
        raise ValueError(f"Missing '# Data section follows' in {path}")
    decl_text = rest[:idx]
    data_text = rest[idx + len(marker):]

    # Parse field declarations (in order).
    declarations: List[Tuple[str, str, str, int, int]] = []  # domain,name,dtype,dim,marker
    for m in _DECL_RE.finditer(decl_text):
        domain, dtype_tok, dim_tok, name, marker_num = m.groups()
        dim = int(dim_tok) if dim_tok else 1
        declarations.append((domain, name, dtype_tok, dim, int(marker_num)))
    if not declarations:
        raise ValueError(f"No field declarations found in {path}")

    # Group the data section into marker -> flat token list.
    blocks: Dict[int, List[str]] = {}
    current: Optional[int] = None
    for line in data_text.splitlines():
        s = line.strip()
        if not s:
            continue
        mm = re.fullmatch(r"@(\d+)", s)
        if mm:
            current = int(mm.group(1))
            blocks.setdefault(current, [])
        elif current is not None:
            blocks[current].extend(s.split())

    graph = SpatialGraph(header=header, raw_parameters=raw_parameters)
    for domain, name, dtype_tok, dim, marker_num in declarations:
        count = counts[domain]
        tokens = blocks.get(marker_num)
        if tokens is None:
            raise ValueError(f"Data block @{marker_num} ({name}) missing in {path}")
        expected = count * dim
        if len(tokens) != expected:
            raise ValueError(
                f"Field {name} (@{marker_num}) expected {expected} values "
                f"({count}x{dim}) but found {len(tokens)} in {path}"
            )
        np_dtype = np.float64 if _is_float_dtype(dtype_tok) else np.int64
        arr = np.array(tokens, dtype=np_dtype)
        arr = arr.reshape(count) if dim == 1 else arr.reshape(count, dim)
        graph.set_field(domain, name, arr)

    return graph


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _format_value(v, is_int: bool) -> str:
    if is_int:
        return str(int(v))
    return format(float(v), ".15e")


def write_amira(graph: SpatialGraph, path: PathLike) -> None:
    """Write ``graph`` to an Amira/Avizo ASCII ``.am`` file.

    Emits header + preserved ``Parameters`` block + regenerated field
    declarations (renumbered ``@1..@k`` contiguously) + data. This is a
    complete, self-contained writer: newly computed fields (e.g. ``strahler`` /
    ``topo``) are declared and written as normal blocks, replacing the fragile
    MATLAB "hand-edit the header then append ``@22``" workflow.
    """
    specs = graph._ordered_fields()
    if not specs:
        raise ValueError("SpatialGraph has no fields to write")

    lines: List[str] = []
    lines.append(graph.header or _DEFAULT_HEADER)
    lines.append("")
    lines.append("")
    lines.append(f"define VERTEX {graph.n_vertices}")
    lines.append(f"define EDGE {graph.n_edges}")
    lines.append(f"define POINT {graph.n_points}")
    lines.append("")

    params = graph.raw_parameters if graph.raw_parameters else _DEFAULT_PARAMETERS
    lines.append(params.rstrip("\n"))
    lines.append("")

    # Declarations, renumbered contiguously.
    for i, (domain, name, arr) in enumerate(specs, start=1):
        dim = 1 if arr.ndim == 1 else arr.shape[1]
        dtype_str = "int" if np.issubdtype(arr.dtype, np.integer) else "float"
        dim_str = f"[{dim}]" if dim > 1 else ""
        lines.append(f"{domain} {{ {dtype_str}{dim_str} {name} }} @{i}")
    lines.append("")
    lines.append("# Data section follows")

    # Data blocks.
    for i, (domain, name, arr) in enumerate(specs, start=1):
        lines.append(f"@{i}")
        is_int = bool(np.issubdtype(arr.dtype, np.integer))
        a2 = arr.reshape(arr.shape[0], -1)
        for row in a2:
            lines.append(" ".join(_format_value(v, is_int) for v in row))
        lines.append("")

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="latin-1", newline="\n") as fh:
        fh.write(text)
