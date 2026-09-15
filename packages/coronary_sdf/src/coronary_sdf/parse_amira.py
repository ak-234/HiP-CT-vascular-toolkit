"""Amira SpatialGraph XML parser.

Reads an Amira/Avizo XML SpatialGraph export (Excel-XML format with
``Nodes``, ``Points``, and ``Segments`` worksheets) and returns three
plain Python containers:

    nodes    : dict[node_id]  -> (x, y, z, coordination_number)
    points   : dict[point_id] -> (x, y, z, thickness)
    segments : list[dict]      with keys: id, node1, node2,
               point_ids[, strahler]

Coordinates are in **micrometers** as stored in the file. Downstream
modules convert to mm where needed.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .config import runtime_config as config

NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}

# Field-name aliases: different Avizo/Amira exports name the per-point radius and
# the per-edge Strahler order differently (coronary uses "thickness"/"strahler";
# HiP-CT kidney uses "Radius"/"StrahlerOrder"). The first present alias wins; the
# canonical downstream keys stay "thickness"-as-radius (points[·][3]) and "strahler".
_RADIUS_FIELD_ALIASES = ("thickness", "Thickness", "Radius", "radius")
_STRAHLER_FIELD_ALIASES = ("strahler", "StrahlerOrder", "Strahler", "StrahlerNumber")


def _get_worksheet(root: ET.Element, name: str) -> ET.Element:
    """Return the ``<Table>`` element for a named worksheet."""
    for ws in root.findall(".//ss:Worksheet", NS):
        if ws.get("{urn:schemas-microsoft-com:office:spreadsheet}Name") == name:
            tbl = ws.find(".//ss:Table", NS)
            if tbl is None:
                raise ValueError(f"Worksheet '{name}' has no <Table>")
            return tbl
    raise ValueError(f"Worksheet '{name}' not found")


def _cell_text(cell: ET.Element) -> str | None:
    data = cell.find("ss:Data", NS)
    return data.text if data is not None else None


def _header_map(table: ET.Element) -> dict[str | None, int]:
    row0 = table.findall("ss:Row", NS)[0]
    cells = row0.findall("ss:Cell", NS)
    return {_cell_text(c): i for i, c in enumerate(cells)}


def parse_nodes(root: ET.Element) -> dict[int, tuple[float, float, float, int]]:
    """node_id -> (x, y, z, coordination_number)."""
    table = _get_worksheet(root, "Nodes")
    hdr = _header_map(table)
    ci = hdr.get("Node ID", 0)
    cx = hdr.get("X Coord", hdr.get("X coord", 1))
    cy = hdr.get("Y Coord", hdr.get("Y coord", 2))
    cz = hdr.get("Z Coord", hdr.get("Z coord", 3))
    cc = hdr.get("Coordination Number", len(hdr) - 1)
    nodes: dict[int, tuple[float, float, float, int]] = {}
    for ri, row in enumerate(table.findall("ss:Row", NS)):
        if ri == 0:
            continue
        cells = row.findall("ss:Cell", NS)
        try:
            nid = int(float(_cell_text(cells[ci])))
            x = float(_cell_text(cells[cx]))
            y = float(_cell_text(cells[cy]))
            z = float(_cell_text(cells[cz]))
            coord = int(float(_cell_text(cells[cc])))
            nodes[nid] = (x, y, z, coord)
        except (TypeError, ValueError, IndexError):
            pass
    return nodes


def parse_points(root: ET.Element) -> dict[int, tuple[float, float, float, float]]:
    """point_id -> (x, y, z, thickness)."""
    table = _get_worksheet(root, "Points")
    hdr = _header_map(table)
    ci = hdr.get("Point ID", 0)
    ct = next((hdr[n] for n in _RADIUS_FIELD_ALIASES if n in hdr), 1)
    cx = hdr.get("X Coord", hdr.get("X coord", 2))
    cy = hdr.get("Y Coord", hdr.get("Y coord", 3))
    cz = hdr.get("Z Coord", hdr.get("Z coord", 4))
    points: dict[int, tuple[float, float, float, float]] = {}
    for ri, row in enumerate(table.findall("ss:Row", NS)):
        if ri == 0:
            continue
        cells = row.findall("ss:Cell", NS)
        try:
            pid = int(float(_cell_text(cells[ci])))
            t = float(_cell_text(cells[ct]))
            x = float(_cell_text(cells[cx]))
            y = float(_cell_text(cells[cy]))
            z = float(_cell_text(cells[cz]))
            points[pid] = (x, y, z, t)
        except (TypeError, ValueError, IndexError):
            pass
    return points


def parse_segments(root: ET.Element) -> list[dict[str, Any]]:
    """List of dicts: ``{id, node1, node2, point_ids[, strahler], **extras}``.

    Every additional column under the Segments worksheet header (Generation,
    Length, BranchAngle, etc.) is surfaced into each segment dict using the
    raw column name as key; values are numeric-coerced when possible.
    """
    table = _get_worksheet(root, "Segments")
    hdr = _header_map(table)
    ci = hdr.get("Segment ID", 0)
    cn1 = hdr.get("Node ID #1", 1)
    cn2 = hdr.get("Node ID #2", 2)
    cp = hdr.get("Point IDs", len(hdr) - 1)
    cs = hdr.get("strahler", None)
    typed_indices = {ci, cn1, cn2, cp}
    if cs is not None:
        typed_indices.add(cs)
    extra_cols = {
        name: i for name, i in hdr.items()
        if name and i not in typed_indices
    }
    segments: list[dict[str, Any]] = []
    for ri, row in enumerate(table.findall("ss:Row", NS)):
        if ri == 0:
            continue
        cells = row.findall("ss:Cell", NS)
        try:
            sid = int(float(_cell_text(cells[ci])))
            node1 = int(float(_cell_text(cells[cn1])))
            node2 = int(float(_cell_text(cells[cn2])))
            pids = [int(p) for p in _cell_text(cells[cp]).split(",")]
            seg: dict[str, Any] = dict(id=sid, node1=node1, node2=node2, point_ids=pids)
            if cs is not None:
                seg["strahler"] = int(float(_cell_text(cells[cs])))
            for col_name, col_idx in extra_cols.items():
                if col_idx >= len(cells):
                    continue
                raw = _cell_text(cells[col_idx])
                if raw is None:
                    continue
                try:
                    seg[col_name] = int(float(raw))
                except (TypeError, ValueError):
                    try:
                        seg[col_name] = float(raw)
                    except (TypeError, ValueError):
                        seg[col_name] = raw
            segments.append(seg)
        except (TypeError, ValueError, IndexError):
            pass
    return segments


def _parse_excel_xml(xml_path: str | Path) -> tuple[
    dict[int, tuple[float, float, float, int]],
    dict[int, tuple[float, float, float, float]],
    list[dict[str, Any]],
]:
    """Parse an Excel-XML Amira spatial-graph and return (nodes, points, segments)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    nodes = parse_nodes(root)
    points = parse_points(root)
    segments = parse_segments(root)
    return nodes, points, segments


# ── Native Avizo/Amira ASCII SpatialGraph (.am) ──────────────────────────────
#
# Helpers adapted from run_ordering_multitree.py. The native format declares
# VERTEX/EDGE/POINT field arrays in a text header ("FIELD { type Name } @N")
# followed by "@N" data blocks. We read the blocks needed to rebuild the same
# (nodes, points, segments) structures the Excel-XML path produces.
# Radius / Strahler field-name aliases are defined at module top.


def _split_header_body(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split at the first ``@N`` marker into (header, body)."""
    for i, ln in enumerate(lines):
        if re.match(r"^\s*@\d+\s*$", ln):
            return lines[:i], lines[i:]
    raise ValueError("No '@N' data blocks found; not an ASCII .am SpatialGraph?")


def _parse_field_decls(header: list[str]) -> dict[str, dict[str, int]]:
    """Map kind -> {field_name -> block_id} for VERTEX/EDGE/POINT declarations."""
    out: dict[str, dict[str, int]] = {"VERTEX": {}, "EDGE": {}, "POINT": {}}
    for ln in header:
        m = re.match(r"^\s*(VERTEX|EDGE|POINT)\s*\{.*\b(\w+)\s*\}\s*@(\d+)\s*$", ln.strip())
        if m:
            out[m.group(1)][m.group(2)] = int(m.group(3))
    return out


def _read_block_rows(body: list[str], block_id: int) -> list[str]:
    """Return the non-blank, non-comment data rows under ``@block_id``."""
    in_block = False
    rows: list[str] = []
    for ln in body:
        m = re.match(r"^\s*@(\d+)\s*$", ln)
        if m:
            in_block = int(m.group(1)) == block_id
            continue
        if not in_block:
            continue
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(s)
    return rows


def _read_block_table(body: list[str], block_id: int, ncols: int, dtype) -> np.ndarray:
    """Read a ``@block_id`` data block as an (N, ncols) array (ncols=1 -> (N,))."""
    rows = _read_block_rows(body, block_id)
    vals = []
    for r in rows:
        parts = r.split()
        if len(parts) != ncols:
            raise ValueError(f"@{block_id}: expected {ncols} cols, got {len(parts)} in '{r}'")
        vals.append(parts)
    arr = np.array(vals, dtype=dtype) if vals else np.empty((0, ncols), dtype=dtype)
    return arr.reshape(-1) if ncols == 1 else arr


def parse_am(am_path: str | Path) -> tuple[
    dict[int, tuple[float, float, float, int]],
    dict[int, tuple[float, float, float, float]],
    list[dict[str, Any]],
]:
    """Parse a native Avizo/Amira ASCII SpatialGraph (.am) -> (nodes, points, segments).

    Output structures match :func:`_parse_excel_xml`:
        nodes[vid]  -> (x, y, z, coordination_number)
        points[pid] -> (x, y, z, thickness)
        segments    -> [{id, node1, node2, point_ids[, strahler], **edge_attrs}]
    Vertices/edges use their native 0-based indices as ids; edge points are
    assigned global sequential ids in edge order (sliced by NumEdgePoints).
    """
    with open(am_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()
    header, body = _split_header_body(lines)

    decls = _parse_field_decls(header)
    vd, ed, pd_ = decls["VERTEX"], decls["EDGE"], decls["POINT"]
    for kind, name, table in (
        ("VERTEX", "VertexCoordinates", vd),
        ("EDGE", "EdgeConnectivity", ed),
        ("EDGE", "NumEdgePoints", ed),
        ("POINT", "EdgePointCoordinates", pd_),
    ):
        if name not in table:
            raise ValueError(f"Native .am missing required {kind} field '{name}'")

    # Per-point radius: accept any known alias (thickness / Radius / ...).
    radius_field = next((n for n in _RADIUS_FIELD_ALIASES if n in pd_), None)
    if radius_field is None:
        raise ValueError(
            "Native .am missing a per-point radius field "
            f"(looked for {_RADIUS_FIELD_ALIASES}; POINT fields present: {sorted(pd_)})"
        )

    verts = _read_block_table(body, vd["VertexCoordinates"], 3, float)
    edge_conn = _read_block_table(body, ed["EdgeConnectivity"], 2, int)
    num_edge_pts = _read_block_table(body, ed["NumEdgePoints"], 1, int)
    ep_coords = _read_block_table(body, pd_["EdgePointCoordinates"], 3, float)
    thickness = _read_block_table(body, pd_[radius_field], 1, float)

    n_pts = len(ep_coords)
    offsets = np.concatenate([[0], np.cumsum(num_edge_pts)]).astype(int)
    if int(offsets[-1]) != n_pts:
        raise ValueError(
            f"sum(NumEdgePoints)={int(offsets[-1])} != EdgePointCoordinates rows={n_pts}"
        )
    if len(thickness) != n_pts:
        raise ValueError(
            f"thickness rows={len(thickness)} != EdgePointCoordinates rows={n_pts}"
        )

    # Coordination number = vertex degree over EdgeConnectivity (== the value
    # Avizo stores in the Excel-XML "Coordination Number" column).
    degree = np.zeros(len(verts), dtype=int)
    for u, v in edge_conn:
        degree[u] += 1
        degree[v] += 1

    nodes = {
        vid: (float(x), float(y), float(z), int(degree[vid]))
        for vid, (x, y, z) in enumerate(verts)
    }
    points = {
        pid: (float(x), float(y), float(z), float(thickness[pid]))
        for pid, (x, y, z) in enumerate(ep_coords)
    }

    # Optional per-edge scalar attributes (strahler, MeanRadius, CurvedLength, …)
    # surfaced into each segment dict by name, mirroring parse_segments' extras.
    skip = {"EdgeConnectivity", "NumEdgePoints"}
    extra_edge_fields = {
        name: blk for name, blk in ed.items() if name not in skip
    }
    extra_arrays: dict[str, np.ndarray] = {}
    for name, blk in extra_edge_fields.items():
        col = _read_block_rows(body, blk)
        if not col or len(col) != len(edge_conn):
            continue
        ncols = len(col[0].split())
        if ncols != 1:
            continue  # vector edge attrs (none expected) are skipped
        # Canonicalise any Strahler alias (e.g. "StrahlerOrder") to "strahler"
        # so the integer dtype + downstream s.get("strahler") lookups fire.
        out_name = "strahler" if name in _STRAHLER_FIELD_ALIASES else name
        dtype = int if out_name == "strahler" else float
        try:
            extra_arrays[out_name] = _read_block_table(body, blk, 1, dtype)
        except ValueError:
            extra_arrays[out_name] = _read_block_table(body, blk, 1, float)

    segments: list[dict[str, Any]] = []
    for eid, (u, v) in enumerate(edge_conn):
        a, b = int(offsets[eid]), int(offsets[eid + 1])
        seg: dict[str, Any] = dict(
            id=eid, node1=int(u), node2=int(v), point_ids=list(range(a, b))
        )
        for name, arr in extra_arrays.items():
            seg[name] = int(arr[eid]) if name == "strahler" else float(arr[eid])
        segments.append(seg)

    return nodes, points, segments


# ── Format autodetection + public dispatcher ─────────────────────────────────


def _detect_format(path: str | Path) -> str:
    """Return 'am' or 'xml' by sniffing content, falling back to the extension."""
    first = ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                if ln.strip():
                    first = ln.strip()
                    break
    except OSError:
        first = ""

    if first.startswith(("# Avizo", "# AmiraMesh", "# HyperSurface")):
        if "BINARY" in first.upper():
            raise ValueError(
                f"Binary Amira/Avizo .am is unsupported: {path}\n"
                "Re-export from Avizo/Amira as ASCII ('Avizo 3D ASCII')."
            )
        return "am"
    if first.startswith(("<?xml", "<?mso-application", "<")):
        return "xml"

    ext = Path(path).suffix.lower()
    if ext == ".am":
        return "am"
    if ext == ".xml":
        return "xml"
    raise ValueError(
        f"Cannot determine spatial-graph format for {path} "
        f"(first line: {first!r}, extension: {ext!r})"
    )


def parse_xml(path: str | Path) -> tuple[
    dict[int, tuple[float, float, float, int]],
    dict[int, tuple[float, float, float, float]],
    list[dict[str, Any]],
]:
    """Parse an Amira spatial-graph and return (nodes, points, segments).

    Autodetects native Avizo/Amira ASCII (.am) vs Excel-XML (.xml) and
    dispatches to the matching parser. Name kept for back-compat;
    :func:`parse_graph` is the format-neutral alias.
    """
    print(f"[PARSE] Reading {path}")
    fmt = _detect_format(path)
    nodes, points, segments = (
        parse_am(path) if fmt == "am" else _parse_excel_xml(path)
    )
    # Convert source coordinate units to micrometres so the downstream /1000
    # µm→mm convention yields correct physical mm. When the spatial-graph stores
    # voxel indices (HiP-CT .am), INPUT_VOXEL_SIZE_UM is the scan's voxel size;
    # 1.0 (data already in µm) is a no-op. Scale coords and per-point radius
    # together so vessel proportions are preserved — only absolute size changes.
    s = float(getattr(config, "INPUT_VOXEL_SIZE_UM", 1.0))
    if s != 1.0:
        nodes = {nid: (x * s, y * s, z * s, c) for nid, (x, y, z, c) in nodes.items()}
        points = {pid: (x * s, y * s, z * s, t * s) for pid, (x, y, z, t) in points.items()}
        print(f"  [UNITS] scaled coords+radii by {s} µm per input unit")
    print(f"  {len(nodes)} nodes, {len(points)} points, {len(segments)} segments")
    return nodes, points, segments


# Format-neutral alias for the dispatcher (preferred name going forward).
parse_graph = parse_xml


# ── Centerline geometry validation ───────────────────────────────────────────


def find_degenerate_segments(
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    *,
    max_gap_um: float = 500.0,
    gap_ratio: float = 0.5,
    min_points: int = 4,
) -> list[dict[str, Any]]:
    """Flag segments whose centerline has a single dominant gap.

    Some ``.am`` exports store degenerate polylines where the interior points
    collapse into a cluster at one endpoint, leaving a large jump between two
    consecutive points (e.g. ``node1 -> [pts near node2] -> node2``). Such
    edges fragment the downstream SDF/spline surfaces. This detector reports
    them without modifying any geometry.

    A segment (whose ``point_ids`` already include the two node endpoints) is
    flagged when its largest consecutive-point step exceeds ``max_gap_um`` and
    also accounts for more than ``gap_ratio`` of the polyline arc length.
    Coordinates are assumed to be in micrometres (as parsed).

    Pure; no I/O. Returns one dict per flagged segment,
    ``{id, node1, node2, n_points, max_gap_um, gap_ratio}``, worst first.
    """
    flagged: list[dict[str, Any]] = []
    for seg in segments:
        pids = seg["point_ids"]
        if len(pids) < min_points:
            continue
        coords = np.array(
            [(points[p][0], points[p][1], points[p][2]) for p in pids if p in points],
            dtype=float,
        )
        if len(coords) < min_points:
            continue
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        arc_len = float(steps.sum())
        if arc_len <= 0.0:
            continue
        max_gap = float(steps.max())
        ratio = max_gap / arc_len
        if max_gap > max_gap_um and ratio > gap_ratio:
            flagged.append(
                dict(
                    id=seg.get("id"),
                    node1=seg["node1"],
                    node2=seg["node2"],
                    n_points=len(pids),
                    max_gap_um=max_gap,
                    gap_ratio=ratio,
                )
            )
    flagged.sort(key=lambda d: d["gap_ratio"], reverse=True)
    return flagged


def report_ring_gap_attribution(
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    *,
    radius_scale: float = 1.0,
    gap_alpha: float = 1.0,
    big_jump_ratio: float = 5.0,
    max_gap_um: float = 500.0,
    gap_ratio: float = 0.5,
    label: str = "",
    top_n: int = 15,
) -> None:
    """Attribute the visible "disconnections" seen in the raw-contour debug viz.

    ``viz.debug_show_raw_data_contours`` draws one wireframe hoop per centerline
    sample and *no* connecting line, so a vessel reads as broken wherever two
    consecutive hoops do not visually overlap. This report mirrors that viz's
    exact extraction (same coords, same ``radii = thickness / 1000 *
    radius_scale``, same per-sample skip rule) and classifies every visible gap
    so we can tell how many breaks are a pure rendering artifact vs. a genuine
    source-data gap that Avizo hides behind a straight edge.

    A gap between two consecutive drawn hoops is "visible" when the
    centre-to-centre step exceeds ``gap_alpha * (r_i + r_j)``. Let
    ``ratio = step / (r_i + r_j)`` (how many vessel-widths the jump spans).
    Each visible gap is labelled:

      DATA-GAP : the segment is flagged by :func:`find_degenerate_segments` and
                 this is its single dominant step -- a real collapsed-polyline
                 gap that Avizo bridges with a straight edge.
      BIG-JUMP : ``ratio >= big_jump_ratio`` -- a jump many vessel-widths wide
                 that the flag missed (typically a mid-segment discontinuity in
                 a long/merged segment, where the jump is not arc-length
                 dominant). Also a real gap Avizo hides; will distort the SDF.
      JOINT    : a small gap (``ratio < big_jump_ratio``) touching a segment
                 endpoint -- a benign bifurcation joint between two
                 independently-drawn segments.
      THIN     : anything else -- an interior gap only a few vessel-widths wide,
                 i.e. the vessel is simply thinner than the local sample spacing
                 (a pure rendering artifact of the hoops-without-a-centerline viz).

    DATA-GAP + BIG-JUMP are the *real* discontinuities; JOINT + THIN are benign /
    rendering-only. Pure; prints only and mutates nothing.
    """
    prefix = f"{label}: " if label else ""
    if not segments or not points:
        print(f"  [GAP-DIAG] {prefix}no raw data")
        return

    flagged = {
        d["id"]
        for d in find_degenerate_segments(
            points, segments, max_gap_um=max_gap_um, gap_ratio=gap_ratio
        )
    }

    strict_alpha = 2.0 * gap_alpha
    n_drawn_total = 0
    n_skipped = 0
    n_strict = 0
    gaps: list[dict[str, Any]] = []  # one entry per visible gap

    for seg in segments:
        sid = seg.get("id")
        pids = [p for p in seg.get("point_ids", []) if p in points]
        if len(pids) < 2:
            continue
        coords = np.asarray([points[p][:3] for p in pids], dtype=np.float64) / 1000.0
        radii = (
            np.asarray([points[p][3] for p in pids], dtype=np.float64)
            / 1000.0
            * radius_scale
        )
        n_p = len(coords)

        # Per-sample tangent + "is a hoop actually drawn?" mask, matching
        # debug_show_raw_data_contours (skip near-zero tangent or radius<=0).
        drawn = np.zeros(n_p, dtype=bool)
        for i in range(n_p):
            if i == 0:
                tang = coords[1] - coords[0]
            elif i == n_p - 1:
                tang = coords[-1] - coords[-2]
            else:
                tang = coords[i + 1] - coords[i - 1]
            drawn[i] = np.linalg.norm(tang) >= 1e-12 and radii[i] > 0.0
        n_drawn = int(drawn.sum())
        n_drawn_total += n_drawn
        n_skipped += n_p - n_drawn

        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        dominant = int(np.argmax(steps)) if len(steps) else -1

        for i in range(n_p - 1):
            if not (drawn[i] and drawn[i + 1]):
                continue  # a skipped hoop is already tallied as a hole above
            sum_r = float(radii[i] + radii[i + 1])
            step = float(steps[i])
            if step <= gap_alpha * sum_r:
                continue  # hoops overlap -> reads as connected
            if step > strict_alpha * sum_r:
                n_strict += 1
            ratio = step / max(sum_r, 1e-9)
            if sid in flagged and i == dominant:
                cause = "DATA-GAP"
            elif ratio >= big_jump_ratio:
                cause = "BIG-JUMP"
            elif i == 0 or i + 1 == n_p - 1:
                cause = "JOINT"
            else:
                cause = "THIN"
            gaps.append(
                dict(
                    sid=sid,
                    i=i,
                    step=step,
                    ri=float(radii[i]),
                    rj=float(radii[i + 1]),
                    ratio=ratio,
                    cause=cause,
                )
            )

    counts = {"DATA-GAP": 0, "BIG-JUMP": 0, "JOINT": 0, "THIN": 0}
    for g in gaps:
        counts[g["cause"]] += 1
    n_gaps = len(gaps)
    n_real = counts["DATA-GAP"] + counts["BIG-JUMP"]
    data_ids = sorted({g["sid"] for g in gaps if g["cause"] == "DATA-GAP"})
    jump_ids = sorted({g["sid"] for g in gaps if g["cause"] == "BIG-JUMP"})

    print(f"  [GAP-DIAG] {prefix}{len(segments)} segments, {n_drawn_total} drawn rings")
    print(
        f"    Visible ring gaps (step > {gap_alpha:g}*(ri+rj)): {n_gaps}"
        f"   (stricter step > {strict_alpha:g}*(ri+rj): {n_strict})"
    )
    data_note = f"   {data_ids}" if data_ids else ""
    jump_note = f"   {jump_ids}" if jump_ids else ""
    print(f"    DATA-GAP (flagged degenerate segs):    {counts['DATA-GAP']}{data_note}")
    print(f"    BIG-JUMP (>= {big_jump_ratio:g}x vessel width, unflagged): {counts['BIG-JUMP']}{jump_note}")
    print(f"    JOINT    (small gap at endpoints):     {counts['JOINT']}")
    print(f"    THIN     (thin/coarse -> viz only):    {counts['THIN']}")
    print(f"    Skipped rings (radius<=0 / zero tangent): {n_skipped}")

    if not gaps:
        print(
            "    VERDICT: no visible ring gaps at this threshold -- the on-screen "
            "breaks are finer than gap_alpha*(ri+rj); lower GAP_VIS_ALPHA to surface them."
        )
        return

    gaps.sort(key=lambda g: g["ratio"], reverse=True)
    print("    Top offenders:")
    for g in gaps[:top_n]:
        print(
            f"      seg {g['sid']} i={g['i']} step={g['step']:.2f}mm "
            f"r=({g['ri']:.2f},{g['rj']:.2f}) step/(ri+rj)={g['ratio']:.1f} {g['cause']}"
        )
    pct_render = 100.0 * (counts["THIN"] + counts["JOINT"]) / n_gaps
    verdict = (
        f"{n_real} of {n_gaps} visible gaps are REAL discontinuities Avizo hides "
        f"({counts['DATA-GAP']} DATA-GAP, {counts['BIG-JUMP']} BIG-JUMP); the other "
        f"{pct_render:.0f}% are benign joints / thin-vessel rendering"
    )
    real_ids = sorted(set(data_ids) | set(jump_ids))
    if real_ids:
        verdict += f". Real-gap segs: {', '.join(str(s) for s in real_ids)}"
    print(f"    VERDICT: {verdict}.")
