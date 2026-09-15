"""Van der Giessen diameter-law outlet flow splits for the Avizo workflow.

Adapts the standalone ``giessen_flow_fractions.py`` (which reconstructs the
vessel tree geometrically from a Simpleware ``.vtk`` + contour/inlet-outlet
text files) to the Avizo skeletonisation workflow that the rest of this
package already processes.

Here the vessel tree is given explicitly by the Avizo spatial-graph
``.am.xml``: :func:`coronary_sdf.parse_amira.parse_xml` yields nodes / points /
segments with radii, and :func:`coronary_sdf.topology.build_directed_topology`
gives a Strahler-rooted parent -> child tree. So no geometric reconstruction is
needed — we read the tree directly, compute the Giessen split at every
bifurcation, propagate to the terminal outlets, and map each terminal to the
matching boundary domain in an ANSYS ``.msh`` file (the mesh produced from the
SDF surface). Outputs match the original script: a per-outlet flow-fraction CSV
and a CFX ``.ccl`` boundary-condition file.

Run::

    python -m coronary_sdf.flow_fractions [<input.am.xml> <input.msh> <output_dir>]

With no args it falls back to ``config.INPUT_XML`` plus the module-level
defaults below.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .parse_amira import parse_xml
from .pruning import prune_short_terminal_nubs, report_segment_radius_range
from .topology import (
    merge_degree2_segments,
    merge_split_multifurcations,
    split_by_graph,
    build_directed_topology,
)
from .smoothing import (
    densify_sparse_segments,
    smooth_segment_centerlines,
    smooth_segment_radii,
    smooth_radius_transitions,
    prune_terminal_shrink,
    prune_bifurcation_shrink,
)

# ============================================================
#                 MODULE PARAMETERS
# ============================================================
# Defaults for the no-argument run. Override on the command line with
# <input.am.xml> <input.msh> <output_dir>.
# No dataset path is hardcoded here; set CORONARY_SDF_MSH / CORONARY_SDF_OUTPUT_DIR
# or pass the three positional arguments.
DEFAULT_MSH_FILE = os.environ.get("CORONARY_SDF_MSH", "")
DEFAULT_OUTPUT_DIR = os.environ.get(
    "CORONARY_SDF_OUTPUT_DIR", str(Path.cwd() / "coronary_sdf_out")
)

# Flow-split parameters
GIESSEN_EXPONENT = 2.27   # Van der Giessen empirical exponent for coronaries
FRAMES_DOWNSTREAM = 10    # centerline points downstream of a bifurcation used
                          # to measure each daughter's representative diameter
BIF_SKIP_POINTS = 10       # points skipped at each bifurcation-connected segment
                          # end so the diameter window avoids the inflated junction
                          # region (not skipped at true-leaf or crop ends)
TRUNCATE_AT_CROP = True   # treat the segment a .msh crop lies in as the leaf and
                          # ignore spatial-graph points/branches beyond the crop

# Crop handling: when True, match each .msh outlet zone to its location on the
# FULL centreline tree and aggregate the Giessen flow of every removed downstream
# leaf into that crop-plane outlet (mass-conservative). When False, fall back to
# the legacy leaf-tip matching + survivor renormalisation.
AGGREGATE_CROPPED_FLOW = True

# Orphan stubs: a .msh outlet zone for a branch that is NOT in the (pruned) spatial
# graph. When True, measure the orphan's diameter from its .msh cross-section and
# inject it as an extra Giessen daughter at its attachment junction (mass-conserving).
# When False, orphans are reported but carry no flow.
RECONSTRUCT_ORPHAN_STUBS = True

# The spatial graph may hold several disconnected trees while the .msh contains
# only one of them. When True, pick the tree(s) the mesh actually sits on -- by
# coordinate proximity of the .msh outlet/inlet centroids to each tree's
# centreline/outlet-node coordinates, NOT by outlet count -- and ignore the rest.
RESTRICT_TO_MESH_TREES = True

# .msh boundary matching
MATCH_TOL_MM = 1.0                       # max nearest-centroid distance to accept
MSH_SCALE = 1.0                          # multiply .msh coords (e.g. 0.001 if metres)
MSH_OFFSET = (0.0, 0.0, 0.0)             # add to .msh coords (mm) after scaling
AUTO_ALIGN_SCALE = True                 # apply bbox-ratio scale auto-detected below
OUTLET_ZONE_KEYWORDS = ("outflow", "outlet", "opening", "plane")
INLET_ZONE_KEYWORDS = ("inflow", "inlet")

# Geometric inlet detection: override the name-based inlet/outlet labels using
# the tree topology. For each meshed tree, the .msh zone centroid nearest the
# trunk root node is the inlet; all other zones are outlets. Fixes an inlet face
# auto-named with an outlet keyword (e.g. "...-plane"/"...-outflow") that would
# otherwise be matched as a spurious crop outlet, leaving the tree with no inlet.
AUTO_DETECT_INLET = True
INLET_DETECT_TOL_MM = 1.0   # max trunk-root-node -> zone-centroid distance (mm)
                            # to accept a zone as that tree's inlet

# Manual inlet/root selection + tie handling (see _resolve_root_pref). The tree
# root segment (= inlet end) is normally chosen automatically (highest Strahler,
# preferring a free degree-1 end, then the thickest). When several segments tie
# at the max Strahler order the choice can be ambiguous; these knobs let you take
# control. INLET_SEG_OVERRIDE always wins if set; otherwise, on a tie and when
# stdin is a TTY, you are prompted (Enter = the automatic pick).
INLET_SEG_OVERRIDE: set[int] = set()   # original Amira seg id(s) to force as a tree root/inlet
INLET_PROMPT_ON_TIE = True             # prompt on a max-Strahler tie when interactive
INLET_PICK_3D = False                  # on a tie, go straight to the interactive 3D
                                       # picker (epicardial_annotation.run_root_picker)
                                       # instead of the console prompt. The console
                                       # prompt also accepts "p" to open the picker.

# Mesh zone renaming (writes a new .msh and updates CCL/CSV names)
RENAME_MSH_ZONES = True
RENAME_MSH_NUM_DIGITS = 3
RENAME_MSH_OUTLET_PREFIX = "Outlet"
RENAME_MSH_INLET_PREFIX = "Inlet"
RENAME_MSH_OUTPUT_SUFFIX = "_renamed"


# 3D viewer theme: background, and the colour of all text (titles, point labels,
# scalar-bar/slider text, the layer legend) and the plain line/point overlays.
VIZ_BG_COLOR = "black"
VIZ_TEXT_COLOR = "white"

# CFX / CCL
FLUID_DENSITY = 1060.0
CFX_DOMAIN = "Default Domain"

VISUALIZE = True          # pop up a PyVista view of the mesh, centrelines,
                          # labelled CFX regions, flow ratios and diameter contours
VIZ_MESH_MAX_FACES = 0  # decimate the .msh surface above this for display
                                 # (set 0/None to always render the full mesh)
VISUALIZE_TREE_GRAPH = True      # write a networkx/Graphviz flow-tree diagram
                                 # (edge labels = avg diameters, nodes = flow split)

# 3D lumen-surface view: every non-bifurcation branch coloured by its Giessen
# flow fraction (fraction of the inlet flow carried by that branch), with the
# bifurcation regions drawn translucent grey. Faces are bound to a branch by
# nearest smoothed centreline point, so the surface must be in the same mm frame
# as the spatial graph.
VISUALIZE_SURFACE_FLOW = True
SURFACE_VTK = "ask"                 # lumen-surface file to colour. "" = auto-discover
                                 # in output_dir (prefers *_regions.vtk, which
                                 # carries the per-face is_junction tag); "ask" =
                                 # choose interactively; anything else = that path
SURFACE_VTK_PROMPT = False       # same as SURFACE_VTK = "ask": always pick the
                                 # surface by hand (Tk file dialog, console list
                                 # if Tk is unavailable or cancelled)
SURFACE_FLOW_CMAP = "turbo"       # single-hue sequential: pale at low flow
                                 # fractions, saturated at high. Any matplotlib
                                 # sequential map works ("Blues", "Greens",
                                 # "Oranges", "Purples", "Greys"); append "_r" to
                                 # flip which end is pale.
SURFACE_FLOW_CMAP_RANGE = (0.25, 1.0)  # slice of the map actually used. The bottom
                                 # of a sequential map is near-white and would be
                                 # confusable with the white bifurcation collars,
                                 # so the palest bands are skipped.
SURFACE_FLOW_LOG_SCALE = True    # flow fractions span decades; log colour scale
# Fixed colour range, in % of the inlet flow, used by EVERY case so the same
# colour always means the same flow fraction across datasets and runs. Values
# outside the range clamp to the end bands (they are counted and reported).
SURFACE_FLOW_CLIM_PCT = (0.01, 100.0)
# Discrete colour bands over that range. On the log scale the default 8 bands
# over 4 decades gives one band per half-decade: 0.01-0.03-0.1-0.3-1-3-10-30-100.
SURFACE_FLOW_N_COLORS = 8
SURFACE_BIF_COLOR = "white"      # bifurcation collars (white reads on the dark bg)
SURFACE_BIF_OPACITY = 0.35
SURFACE_JUNCTION_FACTOR = 0.5    # how far the bifurcation zone reaches ALONG each
                                 # branch from a junction node, in multiples of that
                                 # node radius (arc length, not a ball -- see
                                 # _segment_junction_points). Trade-off measured on
                                 # LADAF_2024_28 (grey share of the surface -> share
                                 # of flow-colour steps that land inside a collar):
                                 # 0.4 -> 12%/81%, 0.5 -> 16%/88%, 0.6 -> 20%/91%,
                                 # 0.75 -> 25%/93%
SURFACE_JUNCTION_SOURCE = "auto"  # which bifurcation tag to trust:
                                 #   "surface" = the VTK own is_junction cell array
                                 #   "graph"   = balls around the graph deg>=3 nodes
                                 #   "both"    = union of the two
                                 #   "auto"    = "surface" when it covers the graph
                                 #               junctions, else "both" + a warning
                                 #               (the usual sign that the VTK surface
                                 #               and the .msh/graph are different
                                 #               geometries)
SURFACE_JUNCTION_COVERAGE_MIN = 0.5   # min share of graph-junction faces the surface
                                      # tag must also flag for "auto" to trust it
# Only treat a graph node as a bifurcation if the meshed tree still branches
# there. When the split runs on the FULL spatial graph but the mesh is a cropped
# subset, nodes whose side branches were cropped away are a plain pass-through on
# the mesh -- collaring them leaves colourless bands mid-vessel. Their flow is
# continuous too (a pruned branch's share is redistributed to its surviving
# sibling), so the band is pure artefact. False restores the old behaviour.
SURFACE_JUNCTION_REQUIRE_MESHED = True
# Cropping a side branch leaves its ostium -- a short stump of the branch mouth --
# on the surviving vessel. Those faces are nearest to the CROPPED branch's
# centreline, so they carry no split flow and would be drawn as "no flow
# assigned" holes mid-vessel. Fill that stump with the flow of the surviving
# vessel it opens off, out to SURFACE_OSTIUM_FACTOR x the take-off node radius
# (arc length along the stump). Anything deeper stays unassigned, so a whole
# branch that is simply missing from the .msh is still reported honestly.
SURFACE_FILL_CROPPED_OSTIA = True
SURFACE_OSTIUM_FACTOR = 2.0
# With no lumen .vtk to colour, build the surface from the .msh instead: its wall
# zones ARE the lumen wall, so they coloured exactly the same way. Slower (the .msh
# surface is usually far denser than the .vtk) and it carries no is_junction tag,
# so the bifurcation collars come from the graph zones.
SURFACE_FALLBACK_TO_MSH = True
SURFACE_FROM_MSH_MAX_FACES = 4_000_000   # decimate the .msh wall surface above this
                                         # before colouring (0/None = never)
SURFACE_UNASSIGNED_COLOR = "gray"     # faces over branches the split never reached
                                      # (pruned as absent from the .msh, or beyond a
                                      # crop): drawn flat, never given a flow colour
SURFACE_FLOW_SHOW_CENTERLINES = False
SURFACE_FLOW_WRITE_VTK = True    # also write <surface>_flow.vtk with the per-face
                                 # flow_fraction / flow_percent / is_junction arrays
SURFACE_ALIGN_WARN_MM = 2.0      # warn above this median face->centreline distance

# Combined viewer: one window holding the flow-coloured lumen surface, the CFX
# mesh, centrelines, diameter contours and the labelled outlet flow fractions,
# with a show/hide checkbox (or number key 1-9) and an opacity slider per layer.
# When True it replaces the two separate VISUALIZE / VISUALIZE_SURFACE_FLOW windows.
VISUALIZE_COMBINED = True
# Load and draw the CFX .msh surface at all. Reading a multi-million-face .msh is
# by far the slowest part of opening the viewer (and it mostly hides the lumen
# surface anyway), so set False -- or pass "nomesh" on the command line -- for a
# fast view of just the flow-coloured surface, centrelines and outlets.
VIZ_SHOW_CFX_MESH = True
COMBINED_WINDOW_SIZE = (1600, 1000)
LAYER_OPACITY = {                # starting opacity of each layer
    "branch": 1.00,              # lumen surface, coloured by branch flow fraction
    "bifurcations": 0.35,        # translucent grey junction regions
    "cfx_mesh": 0.00,            # cropped CFX .msh surface
    "centrelines": 0.00,
    "contours": 0.00,            # diameter rings + sample points
    "outlets": 0.00,             # CFX outlet / orphan-stub markers
    "inlets": 0.00,
    "dropped": 0.00,             # cropped-away leaves
    "unassigned": 0.00,          # faces with no flow (absent from the .msh split)
    "labels": 0.00,              # outlet name + flow % text
}

# Inflow is assumed to be assigned in CFX-Pre. This script outputs fractional flows.
# ============================================================


# ── .msh boundary extraction (adapted from giessen_flow_fractions.py) ─────────


def _tri_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Area of triangle (a, b, c) in the .msh coordinate units."""
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def extract_msh_boundaries(msh_path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Extract inlet/outlet boundaries and the 3D domain region from a ``.msh``.

    Returns ``(centroids, domain_location)``:
      * ``centroids`` — one dict per inlet/outlet zone with ``region_name``,
        ``ccl_name``, ``x``/``y``/``z`` (centroid), ``kind`` ('inlet' | 'outlet'),
        ``area_mm2`` and ``diam_mm`` (cross-section equivalent diameter).
      * ``domain_location`` — the CFX 3D-region name for the DOMAIN ``Location``,
        built from the ``(45 (id fluid/solid NAME 1))`` zone records with dashes
        replaced by spaces (CFX's convention), comma-joined if several. Empty
        string if no fluid/solid zone is found.
    Coordinates are taken straight from the file (no scale/offset applied here).
    """
    msh_path = str(msh_path)
    print(f"\n[MSH] Extracting boundaries from {msh_path}")
    max_node_id = 0
    with open(msh_path, "r") as f:
        for line in f:
            if line.startswith("(10 (0 "):
                match = re.search(r"\(10 \(0 [a-fA-F0-9]+ ([a-fA-F0-9]+) ", line)
                if match:
                    max_node_id = int(match.group(1), 16)
                    break
    if max_node_id <= 0:
        max_node_id = 5000000

    coords = np.zeros((max_node_id + 1, 3), dtype=np.float32)
    with open(msh_path, "r") as f:
        in_nodes = False
        current_node = 1
        for line in f:
            if not in_nodes:
                if line.startswith("(10 (1"):
                    in_nodes = True
            else:
                line = line.strip()
                if not line or line == ")":
                    in_nodes = False
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    coords[current_node, 0] = float(parts[0])
                    coords[current_node, 1] = float(parts[1])
                    coords[current_node, 2] = float(parts[2])
                    current_node += 1

    inlet_kw = tuple(k.lower() for k in INLET_ZONE_KEYWORDS)
    outlet_kw = tuple(k.lower() for k in OUTLET_ZONE_KEYWORDS)

    centroids: list[dict[str, Any]] = []
    domain_names: list[str] = []
    with open(msh_path, "r") as f:
        zone_name = None
        zone_kind = None
        in_faces = False
        sum_coords = np.zeros(3, dtype=np.float64)
        total_nodes = 0
        sum_area = 0.0
        for line in f:
            if '(0 " zone-name:' in line:
                start = line.find("zone-name:") + 10
                end = line.rfind('"')
                zone_name = line[start:end].strip()
            elif line.startswith("(45 ("):
                # Zone table: (45 (id type name domain) ()). The fluid/solid 3D
                # cell zone gives the DOMAIN Location for CFX.
                m = re.match(r"\(45 \(\d+\s+(\S+)\s+(\S+)", line)
                if m and m.group(1).lower() in ("fluid", "solid"):
                    domain_names.append(m.group(2))
            elif line.startswith("(13 ("):
                kind = None
                if zone_name is not None:
                    zl = zone_name.lower()
                    if any(kw in zl for kw in inlet_kw):
                        kind = "inlet"
                    elif any(kw in zl for kw in outlet_kw):
                        kind = "outlet"
                if kind is not None:
                    in_faces = True
                    zone_kind = kind
                    sum_coords[:] = 0.0
                    total_nodes = 0
                    sum_area = 0.0
                else:
                    zone_name = None
            else:
                if in_faces:
                    line = line.strip()
                    if not line or line == ")":
                        if total_nodes > 0:
                            area_mm2 = sum_area * (MSH_SCALE ** 2)
                            diam_mm = (2.0 * float(np.sqrt(area_mm2 / np.pi))
                                       if area_mm2 > 0 else 0.0)
                            centroids.append({
                                "region_name": zone_name.replace("-", " "),
                                "ccl_name": zone_name,
                                "kind": zone_kind,
                                "x": sum_coords[0] / total_nodes,
                                "y": sum_coords[1] / total_nodes,
                                "z": sum_coords[2] / total_nodes,
                                "area_mm2": float(area_mm2),
                                "diam_mm": float(diam_mm),
                            })
                        in_faces = False
                        zone_name = None
                        zone_kind = None
                        continue
                    parts = line.split()
                    if parts[0] == "3" and len(parts) >= 4:
                        n1, n2, n3 = (int(parts[1], 16), int(parts[2], 16), int(parts[3], 16))
                        sum_coords += coords[n1] + coords[n2] + coords[n3]
                        total_nodes += 3
                        sum_area += _tri_area(coords[n1], coords[n2], coords[n3])
                    elif parts[0] == "4" and len(parts) >= 5:
                        n1, n2, n3, n4 = (
                            int(parts[1], 16), int(parts[2], 16),
                            int(parts[3], 16), int(parts[4], 16),
                        )
                        sum_coords += coords[n1] + coords[n2] + coords[n3] + coords[n4]
                        total_nodes += 4
                        sum_area += (_tri_area(coords[n1], coords[n2], coords[n3])
                                     + _tri_area(coords[n1], coords[n3], coords[n4]))
    n_in = sum(1 for c in centroids if c["kind"] == "inlet")
    n_out = sum(1 for c in centroids if c["kind"] == "outlet")
    domain_location = ", ".join(n.replace("-", " ") for n in domain_names)
    print(f"[MSH] Extracted {len(centroids)} boundary regions "
          f"({n_in} inlet, {n_out} outlet).")
    if domain_location:
        print(f"[MSH] Domain 3D region(s): {domain_location}")
    else:
        print("[MSH][WARN] No fluid/solid (45) zone found — CCL DOMAIN Location "
              "will be left blank (set it manually in CFX).")
    return centroids, domain_location


_KIND_ID = {"wall": 0, "inlet": 1, "outlet": 2}


def extract_msh_surface(msh_path: str | Path):
    """Build a renderable boundary surface from an ANSYS ``.msh`` for the viz.

    Reads the node block then *all* ``(13 (`` face zones (walls, inlets, outlets)
    and returns ``(surface, zone_lookup)`` where ``surface`` is a ``pv.PolyData``
    carrying ``cell_data['zone_id']`` and ``cell_data['kind_id']`` (0 wall, 1
    inlet, 2 outlet), and ``zone_lookup`` maps ``zone_id -> {'name', 'kind'}``.
    Coordinates are raw ``.msh`` units (callers apply ``MSH_SCALE``/``MSH_OFFSET``).
    Returns ``(None, {})`` on any failure so the viz can fall back to centroids.
    """
    try:
        import pyvista as pv  # noqa: PLC0415 — optional viz dependency
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[MSH][VIS][WARN] pyvista unavailable, skipping surface: {exc}")
        return None, {}

    try:
        msh_path = str(msh_path)
        # --- node coordinates (1-based, mirrors extract_msh_boundaries) ---
        max_node_id = 0
        with open(msh_path, "r") as f:
            for line in f:
                if line.startswith("(10 (0 "):
                    m = re.search(r"\(10 \(0 [a-fA-F0-9]+ ([a-fA-F0-9]+) ", line)
                    if m:
                        max_node_id = int(m.group(1), 16)
                        break
        if max_node_id <= 0:
            max_node_id = 5_000_000
        coords = np.zeros((max_node_id + 1, 3), dtype=np.float64)
        n_nodes = 0
        with open(msh_path, "r") as f:
            in_nodes = False
            current = 1
            for line in f:
                if not in_nodes:
                    if line.startswith("(10 (1"):
                        in_nodes = True
                else:
                    s = line.strip()
                    if not s or s == ")":
                        in_nodes = False
                        continue
                    parts = s.split()
                    if len(parts) >= 3:
                        coords[current, 0] = float(parts[0])
                        coords[current, 1] = float(parts[1])
                        coords[current, 2] = float(parts[2])
                        current += 1
            n_nodes = current - 1
        if n_nodes <= 0:
            print("[MSH][VIS][WARN] no nodes parsed; skipping surface.")
            return None, {}

        inlet_kw = tuple(k.lower() for k in INLET_ZONE_KEYWORDS)
        outlet_kw = tuple(k.lower() for k in OUTLET_ZONE_KEYWORDS)

        faces_flat: list[int] = []
        cell_zone: list[int] = []
        cell_kind: list[int] = []
        zone_lookup: dict[int, dict[str, str]] = {}

        with open(msh_path, "r") as f:
            zone_name = None
            in_faces = False
            cur_zid = -1
            cur_kindid = 0
            for line in f:
                if '(0 " zone-name:' in line:
                    start = line.find("zone-name:") + 10
                    end = line.rfind('"')
                    zone_name = line[start:end].strip()
                elif line.startswith("(13 ("):
                    m = re.search(r"\(13 \(([a-fA-F0-9]+)\s", line)
                    zid = int(m.group(1), 16) if m else 0
                    if zid == 0:
                        continue  # declaration header, no face data
                    name = zone_name if zone_name else f"zone-{zid:x}"
                    zl = name.lower()
                    if any(kw in zl for kw in inlet_kw):
                        kind = "inlet"
                    elif any(kw in zl for kw in outlet_kw):
                        kind = "outlet"
                    else:
                        kind = "wall"
                    zone_lookup[zid] = {"name": name.replace("-", " "), "kind": kind}
                    cur_zid = zid
                    cur_kindid = _KIND_ID[kind]
                    in_faces = True
                    zone_name = None
                elif in_faces:
                    s = line.strip()
                    if not s or s[0] == ")":
                        in_faces = False
                        zone_name = None
                        continue
                    parts = s.split()
                    try:
                        if parts[0] == "3" and len(parts) >= 4:
                            ids = [int(parts[1], 16), int(parts[2], 16), int(parts[3], 16)]
                        elif parts[0] == "4" and len(parts) >= 5:
                            ids = [int(parts[1], 16), int(parts[2], 16),
                                   int(parts[3], 16), int(parts[4], 16)]
                        else:
                            continue
                    except ValueError:
                        continue
                    if any(i < 1 or i > n_nodes for i in ids):
                        continue
                    faces_flat.append(len(ids))
                    faces_flat.extend(i - 1 for i in ids)   # 1-based -> 0-based
                    cell_zone.append(cur_zid)
                    cell_kind.append(cur_kindid)

        if not cell_zone:
            print("[MSH][VIS][WARN] no boundary faces parsed; skipping surface.")
            return None, {}

        verts = coords[1:n_nodes + 1]
        surface = pv.PolyData(verts, faces=np.asarray(faces_flat, dtype=np.int64))
        surface.cell_data["zone_id"] = np.asarray(cell_zone, dtype=np.int32)
        surface.cell_data["kind_id"] = np.asarray(cell_kind, dtype=np.int32)
        n_wall = sum(1 for k in cell_kind if k == 0)
        print(f"[MSH] Surface: {surface.n_points:,} verts, {surface.n_cells:,} faces "
              f"({len(zone_lookup)} zones, {n_wall:,} wall faces).")
        return surface, zone_lookup
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[MSH][VIS][WARN] surface extraction failed: {exc}")
        return None, {}




# ── CFX CCL writer (adapted from giessen_flow_fractions.py) ───────────────────


def generate_cfx_ccl(tree_inlets, outlet_fractions, output_ccl: str | Path,
                     domain_location: str = "", opening=None,
                     inlet_mass_flow_kg_s=None, boundary_only: bool = False) -> None:
    """Write CFX-Pre boundary conditions; each outlet scales off its tree inlet.

    ``domain_location`` is the 3D mesh region name CFX assigns to the fluid
    cell zone (from :func:`extract_msh_boundaries`); it is written as the DOMAIN
    ``Location`` so CFX-Pre can bind the domain to the imported mesh.
    """
    lines: list[str] = []

    lines.append("# ============================================================")
    lines.append("# Giessen Flow Split Outlet Boundary Conditions for ANSYS CFX-Pre")
    lines.append(f"# Fluid density: {FLUID_DENSITY} kg/m^3")
    lines.append(f"# Giessen exponent: {GIESSEN_EXPONENT}")
    if inlet_mass_flow_kg_s is None:
        lines.append("# Each fixed Outlet scales as a fraction of its Tree INLET.")
    else:
        lines.append("# Numeric fixed-outlet mass flows derived from the audited inlet mass flow.")
    if opening is not None:
        lines.append("# The opening fraction is deliberately not renormalised into fixed outlets.")
    lines.append("# Boundary-only mode preserves the immutable seed physics and inlet setting.")
    lines.append("# Fluent cap zone types enforced (mass-flow inlet) for CFX field association.")
    lines.append("# ============================================================\n")

    lines.append("LIBRARY:")
    lines.append("  CEL:")
    lines.append("    EXPRESSIONS:")
    lines.append("      gr = Shear Strain Rate/grc")
    lines.append("      grc = 2.3 [s^-1]")
    lines.append("      k0 = 3.691")
    lines.append("      kinf = 1.778")
    lines.append("      muF = 1.32e-3 [Pa s]")
    lines.append("      muQmd = muF*(1-0.5*phi*(k0+kinf*sqrt(gr))/(1+sqrt(gr)))^(-2)")
    lines.append("      phi = 0.43")
    lines.append("")
    lines.append("      # Fractional flow splits")
    for out in outlet_fractions:
        lines.append(f"      QoutletNode{out['idx']} = {out['fraction']:.6f} # Tree {out['tree_id']}")
    lines.append("")
    for out in outlet_fractions:
        if inlet_mass_flow_kg_s is None:
            inlet = tree_inlets[out["tree_id"]]
            inlet_name = inlet.get("boundary_name", inlet["ccl_name"])
            value = f"massFlow()@{inlet_name} * QoutletNode{out['idx']}"
        else:
            # A massFlow() query on another boundary is evaluated while CFX is
            # constructing its boundary-condition database.  With a
            # non-Newtonian material this can request DENSITY before the solver
            # variable database exists (CAL_BCP_CDB / CRESLT= CVAR).  The
            # authoritative inlet flow is already audited, so materialise the
            # identical, unrenormalised Giessen split as a dimensional CEL
            # constant instead of creating that cross-boundary dependency.
            value = "{:.15g} [kg s^-1]".format(
                float(inlet_mass_flow_kg_s) * float(out["fraction"])
            )
        lines.append(f"      MoutletNode{out['idx']} = {value}")
    lines.append("    END")
    lines.append("  END")
    if not boundary_only:
        lines.append("  MATERIAL: Quemada")
        lines.append("    Material Group = User")
        lines.append("    Option = Pure Substance")
        lines.append("    PROPERTIES:")
        lines.append("      Option = General Material")
        lines.append("      EQUATION OF STATE:")
        lines.append("        Density = 1050 [kg m^-3]")
        lines.append("        Molar Mass = 1.0 [kg kmol^-1]")
        lines.append("        Option = Value")
        lines.append("      END")
        lines.append("      DYNAMIC VISCOSITY:")
        lines.append("        Dynamic Viscosity = muQmd")
        lines.append("        Option = Value")
        lines.append("      END")
        lines.append("    END")
        lines.append("  END")
    lines.append("END")
    lines.append("")

    lines.append("FLOW: Flow Analysis 1")
    lines.append(f"  DOMAIN: {CFX_DOMAIN}")
    if not boundary_only:
        lines.append("    Coord Frame = Coord 0")
        lines.append("    Domain Type = Fluid")
    if domain_location and not boundary_only:
        # Bind the domain to the imported mesh's 3D fluid region.
        lines.append(f"    Location = {domain_location}")
    elif not boundary_only:
        lines.append("    # Location = <SET 3D MESH REGION HERE> "
                     "# no fluid zone found in .msh")
    if not boundary_only:
        lines.append("    DOMAIN MODELS:")
        lines.append("      BUOYANCY MODEL:")
        lines.append("        Option = Non Buoyant")
        lines.append("      END")
        lines.append("      DOMAIN MOTION:")
        lines.append("        Option = Stationary")
        lines.append("      END")
        lines.append("      MESH DEFORMATION:")
        lines.append("        Option = None")
        lines.append("      END")
        lines.append("      REFERENCE PRESSURE:")
        lines.append("        Reference Pressure = 0 [atm]")
        lines.append("      END")
        lines.append("    END")
        lines.append("    FLUID DEFINITION: Fluid 1")
        lines.append("      Material = Quemada")
        lines.append("      Option = Material Library")
        lines.append("      MORPHOLOGY:")
        lines.append("        Option = Continuous Fluid")
        lines.append("      END")
        lines.append("    END")
        lines.append("    FLUID MODELS:")
        lines.append("      COMBUSTION MODEL:")
        lines.append("        Option = None")
        lines.append("      END")
        lines.append("      HEAT TRANSFER MODEL:")
        lines.append("        Option = None")
        lines.append("      END")
        lines.append("      THERMAL RADIATION MODEL:")
        lines.append("        Option = None")
        lines.append("      END")
        lines.append("      TURBULENCE MODEL:")
        lines.append("        Option = Laminar")
        lines.append("      END")
        lines.append("    END")

    for tid, inlet_info in tree_inlets.items():
        boundary_name = inlet_info.get("boundary_name", inlet_info["ccl_name"])
        lines.append(f"    &replace BOUNDARY: {boundary_name}")
        lines.append("      Boundary Type = INLET")
        lines.append(f"      Location = {inlet_info['region_name']}")
        lines.append("      BOUNDARY CONDITIONS:")
        lines.append("        FLOW DIRECTION:")
        lines.append("          Option = Normal to Boundary Condition")
        lines.append("        END")
        lines.append("        FLOW REGIME:")
        lines.append("          Option = Subsonic")
        lines.append("        END")
        lines.append("        MASS AND MOMENTUM:")
        # Match the audited TMM reference DEF exactly.  ``Bulk Mass Flow
        # Rate`` is a different CFX boundary formulation and can request the
        # generic DENSITY variable while the boundary database is being
        # constructed for an imported Fluent patch.
        lines.append("          Option = Mass Flow Rate")
        if inlet_mass_flow_kg_s is None:
            lines.append("          Mass Flow Rate = 0.001 [kg s^-1] # REPLACE THIS VALUE IN CFX-PRE")
        else:
            lines.append("          Mass Flow Rate = {:.12g} [kg s^-1]".format(float(inlet_mass_flow_kg_s)))
        lines.append("          Mass Flow Rate Area = As Specified")
        lines.append("        END")
        lines.append("      END")
        lines.append("    END")

    for out in outlet_fractions:
        boundary_name = out.get("boundary_name", out["ccl_name"])
        lines.append(f"    &replace BOUNDARY: {boundary_name}")
        lines.append("      Boundary Type = OUTLET")
        lines.append(f"      Location = {out['region_name']}")
        lines.append("      BOUNDARY CONDITIONS:")
        lines.append("        FLOW REGIME:")
        lines.append("          Option = Subsonic")
        lines.append("        END")
        lines.append("        MASS AND MOMENTUM:")
        lines.append("          Option = Mass Flow Rate")
        lines.append(f"          Mass Flow Rate = MoutletNode{out['idx']}")
        lines.append("          Mass Flow Rate Area = As Specified")
        lines.append("        END")
        lines.append("      END")
        lines.append("    END")

    if opening is not None:
        lines.append("    &replace BOUNDARY: {}".format(opening.get("opening_ccl_name", "Pressure_Opening")))
        lines.append("      Boundary Type = OPENING")
        lines.append("      Location = {}".format(opening["region_name"]))
        lines.append("      BOUNDARY CONDITIONS:")
        lines.append("        FLOW DIRECTION:")
        lines.append("          Option = Normal to Boundary Condition")
        lines.append("        END")
        lines.append("        FLOW REGIME:")
        lines.append("          Option = Subsonic")
        lines.append("        END")
        lines.append("        MASS AND MOMENTUM:")
        # Use the internal CCL enum, not the abbreviated GUI label
        # ("Opening Pres. and Dirn").
        lines.append("          Option = Opening Pressure and Direction")
        lines.append("          Relative Pressure = 0 [Pa]")
        lines.append("        END")
        lines.append("      END")
        lines.append("    END")

    lines.append("  END")
    if not boundary_only:
        lines.append("  INITIALISATION:")
        lines.append("    Option = Automatic")
        lines.append("    INITIAL CONDITIONS:")
        lines.append("      Velocity Type = Cartesian")
        lines.append("      CARTESIAN VELOCITY COMPONENTS:")
        lines.append("        Option = Automatic with Value")
        lines.append("        U = 0 [m s^-1]")
        lines.append("        V = 0 [m s^-1]")
        lines.append("        W = 0 [m s^-1]")
        lines.append("      END")
        lines.append("      STATIC PRESSURE:")
        lines.append("        Option = Automatic with Value")
        lines.append("        Relative Pressure = 0 [Pa]")
        lines.append("      END")
        lines.append("    END")
        lines.append("  END")
    lines.append("END")

    with open(output_ccl, "w") as f:
        f.write("\n".join(lines))
    print(f"[CCL] Exported boundary conditions to {output_ccl}")


# ── Avizo graph preprocessing (mirrors pipeline.run_pipeline) ─────────────────


def preprocess_topology(nodes, points, segments):
    """Top-level topology preprocessing identical to pipeline.run_pipeline.

    Strahler filter -> degree-2 contraction -> split-multifurcation merge ->
    short-terminal-nub prune -> densify. Returns ``(nodes, points, segments)``.
    Keeps the terminal set in step with the meshed SDF surface.
    """
    report_segment_radius_range("parse", segments, points)

    if config.MIN_STRAHLER_ORDER > 0:
        before = len(segments)
        segments = [s for s in segments if s.get("strahler", 0) >= config.MIN_STRAHLER_ORDER]
        if len(segments) < before:
            print(f"  Strahler filter (>= {config.MIN_STRAHLER_ORDER}): "
                  f"{before} -> {len(segments)} segments")

    if config.MERGE_DEGREE2_SEGMENTS:
        nodes, segments, n_merged = merge_degree2_segments(nodes, points, segments)
        if n_merged > 0:
            print(f"  Degree-2 contraction: merged {n_merged} segments")

    if config.MERGE_SPLIT_MULTIFURCATIONS:
        nodes, segments, n_collapsed, _records = merge_split_multifurcations(
            nodes, points, segments,
            max_len_factor=config.SPLIT_MULTIFURC_MAX_LEN_FACTOR,
            require_strahler=config.SPLIT_MULTIFURC_REQUIRE_STRAHLER,
            tangent_cos_min=config.SPLIT_MULTIFURC_TANGENT_COS_MIN,
        )
        if n_collapsed > 0:
            print(f"  Split-multifurcation contraction: collapsed {n_collapsed} stubs")

    if config.PRUNE_SHORT_TERMINAL_NUBS and config.MIN_TERMINAL_LENGTH_MM > 0:
        nodes, segments, n_pruned = prune_short_terminal_nubs(
            nodes, points, segments,
            min_length_mm=config.MIN_TERMINAL_LENGTH_MM,
            max_iters=config.PRUNE_ITER_MAX,
        )
        if n_pruned > 0:
            print(f"  Pruned {n_pruned} short terminal segments")

    if config.DENSIFY_SPARSE_SEGMENTS:
        points, n_densified = densify_sparse_segments(
            points, segments,
            target_spacing_mm=config.DENSIFY_TARGET_SPACING_MM,
            min_points=config.DENSIFY_MIN_POINTS,
            verbose=config.DENSIFY_VERBOSE,
        )
        if n_densified > 0:
            print(f"  Densified {n_densified} sparse segments")

    return nodes, points, segments


def smooth_graph(nodes, points, segments, smooth_radii=True):
    """Per-graph centerline + radius smoothing (mirrors generate_sdf_surface).

    The split reads radii from ``points`` afterwards. With ``smooth_radii=True``
    the radii are smoothed to match the surface; with ``smooth_radii=False`` the
    four radius passes are skipped so the split uses the raw spatial-graph radii
    (centerline geometry is smoothed either way, so mesh-outlet matching is
    identical). ``flow_fractions.run`` drives this from
    ``config.FLOW_SPLIT_USE_RAW_RADII``. Returns the ``points`` dict.
    """
    points, _ = smooth_segment_centerlines(nodes, points, segments)
    if smooth_radii:
        points, _ = smooth_segment_radii(nodes, points, segments)
        points, _ = prune_bifurcation_shrink(nodes, points, segments)
        points, _ = prune_terminal_shrink(nodes, points, segments)
        points, _ = smooth_radius_transitions(nodes, points, segments)
    return points


# ── Giessen split on the directed tree ────────────────────────────────────────


def _radius_mm(points, pid):
    return points[pid][3] / 1000.0 * config.RADIUS_SCALE


def _branch_diameter(seg, shared_node, points, nodes, crop_limit=None):
    """Representative daughter diameter (mm) measured on the *clean* tube, away
    from bifurcation regions, and never past a crop. Returns ``(diam, used_pids)``.

    The window starts ``BIF_SKIP_POINTS`` points in from ``shared_node`` (the
    proximal bifurcation) and ends ``BIF_SKIP_POINTS`` before the distal end when
    that end is also a bifurcation (deg>=3); a true-leaf or crop end is not
    skipped. ``crop_limit`` (index into ``seg['point_ids']``) caps the distal
    extent. Diameter = 2 * mean radius over the first FRAMES_DOWNSTREAM clean
    points. Short branch fallback: if the full skips leave no clean point, use
    2 * the minimum radius over the (crop-capped) segment (slimmest = least
    bifurcation-inflated).
    """
    pids = list(seg["point_ids"])
    if not pids:
        return 0.0, []

    # Orient so index 0 is the shared (proximal bifurcation) end.
    if seg["node1"] == shared_node:
        opids = pids
        distal_node = seg["node2"]
        climit = crop_limit
    else:
        opids = list(reversed(pids))
        distal_node = seg["node1"]
        climit = (len(pids) - 1 - crop_limit) if crop_limit is not None else None

    n = len(opids)
    end = n
    cropped = False
    if climit is not None and 0 <= climit < n:
        end = climit + 1
        cropped = end < n or True  # a matched crop end is a flat face, not a bif

    def _deg(nid):
        return nodes[nid][3] if nid in nodes else 0

    prox_skip = BIF_SKIP_POINTS
    dist_is_bif = (not cropped) and (_deg(distal_node) >= 3)
    dist_skip = BIF_SKIP_POINTS if dist_is_bif else 0

    lo, hi = prox_skip, end - dist_skip
    if lo < hi:
        window = [p for p in opids[lo:hi][:FRAMES_DOWNSTREAM] if p in points]
        rs = [_radius_mm(points, p) for p in window]
        if rs:
            return 2.0 * float(np.mean(rs)), window

    # Short branch: too tight to exclude the junction(s) -> slimmest cross-section.
    seg_pids = [p for p in opids[:end] if p in points]
    rs_all = [_radius_mm(points, p) for p in seg_pids]
    if rs_all:
        j = int(np.argmin(rs_all))
        return 2.0 * float(rs_all[j]), [seg_pids[j]]
    return 0.0, []


def branch_radius_mm(
    seg, shared_node, points, nodes,
    skip_points=None, n_average=None, crop_limit=None,
):
    """Representative radius (mm) of a branch just downstream of its ostium.

    Like :func:`_branch_diameter` but returns a *radius*, exposes the skip /
    average counts, and skips the first ``skip_points`` contours of the branch —
    where it intersects the parent and the segmented radius is an outlier — then
    averages the radius over the next ``n_average`` clean contours. Index 0 is
    oriented at ``shared_node`` (the ostium / proximal bifurcation); the distal
    end is also skipped when it is a bifurcation (deg>=3). ``crop_limit`` (index
    into ``seg['point_ids']``) caps the distal extent.

    Short-branch fallback: when the full skips leave no clean window, still drop
    the first 1-2 ostium contours, then take the **median** radius over the
    remaining contours (resists both the ostium and the distal-junction
    inflation without needing a clean window). Returns ``(radius_mm, used_pids)``.
    """
    skip = BIF_SKIP_POINTS if skip_points is None else int(skip_points)
    n_avg = FRAMES_DOWNSTREAM if n_average is None else int(n_average)
    pids = list(seg["point_ids"])
    if not pids:
        return 0.0, []

    if seg["node1"] == shared_node:
        opids = pids
        distal_node = seg["node2"]
        climit = crop_limit
    else:
        opids = list(reversed(pids))
        distal_node = seg["node1"]
        climit = (len(pids) - 1 - crop_limit) if crop_limit is not None else None

    end = len(opids)
    cropped = False
    if climit is not None and 0 <= climit < end:
        end = climit + 1
        cropped = True

    def _deg(nid):
        return nodes[nid][3] if nid in nodes else 0

    dist_skip = skip if ((not cropped) and _deg(distal_node) >= 3) else 0
    lo, hi = skip, end - dist_skip
    if lo < hi:
        window = [p for p in opids[lo:hi][:n_avg] if p in points]
        rs = [_radius_mm(points, p) for p in window]
        if rs:
            return float(np.mean(rs)), window

    # Short-branch fallback: drop the ostium contours we can, median of the rest.
    avail = [p for p in opids[:end] if p in points]
    if not avail:
        return 0.0, []
    start = min(skip, len(avail) - 1)            # never consume the whole branch
    start = max(start, min(2, len(avail) - 1))   # always drop >= the first 1-2
    tail = avail[start:] or [avail[-1]]
    rs = [_radius_mm(points, p) for p in tail]
    return float(np.median(rs)), tail


def _seg_mean_diameter(seg, points):
    """Mean diameter (mm) over a whole segment — trunk diameter at an orphan's
    attachment junction. Returns 2 * mean radius, or 0.0 if no radii."""
    rs = [_radius_mm(points, p) for p in seg["point_ids"] if p in points]
    return 2.0 * float(np.mean(rs)) if rs else 0.0


def _seg_centerline_mm(seg, points):
    """All centerline points of a segment as an (N, 3) array in mm."""
    coords = [points[p][:3] for p in seg["point_ids"] if p in points]
    if not coords:
        return np.empty((0, 3))
    return np.array(coords, dtype=np.float64) / 1000.0


def _resolve_root_pref(gid, nodes, segments, points=None):
    """Choose the root (= inlet) segment for one graph; return ``{local_idx}``.

    The root is normally auto-picked from the segments tying at the max Strahler
    order, preferring a free (degree-1) end then the thickest (``MeanRadius``).
    Manual control (see ``INLET_SEG_OVERRIDE`` / ``INLET_PROMPT_ON_TIE``):

    1. ``INLET_SEG_OVERRIDE`` (original Amira seg ids) always wins if it names a
       segment in this graph.
    2. On a max-Strahler tie (2+ tied segments), the candidates are listed; if
       ``INLET_PROMPT_ON_TIE`` and stdin is a TTY the user is prompted (Enter =
       auto pick, ``p`` = pick in 3D), otherwise the auto pick is used with an
       ``[INLET][TIE]`` note. ``INLET_PICK_3D`` skips the console prompt and opens
       the 3D picker directly. Picking in 3D needs ``points`` (the graph's point
       table) and PyVista; without either the console prompt is used.

    The returned local index is forced into ``build_directed_topology`` so the
    pre-split basics and the split agree on the same inlet.
    """
    n = len(segments)
    if n == 0:
        return set()

    node_to_segs: dict[int, set[int]] = {}
    for idx, seg in enumerate(segments):
        for nid in (seg["node1"], seg["node2"]):
            node_to_segs.setdefault(nid, set()).add(idx)

    def has_free(i):
        s = segments[i]
        return any(len(node_to_segs.get(nid, ())) == 1 for nid in (s["node1"], s["node2"]))

    def mean_radius(i):
        return int(segments[i].get("MeanRadius", 0))

    def inlet_node_pos(i):
        s = segments[i]
        n1, n2 = s["node1"], s["node2"]
        nid = n1 if len(node_to_segs.get(n1, ())) == 1 else (
            n2 if len(node_to_segs.get(n2, ())) == 1 else n1)
        if nid not in nodes:
            return None
        return np.array(nodes[nid][:3], dtype=np.float64) / 1000.0

    strah = [int(s.get("strahler", 0)) for s in segments]
    smax = max(strah)
    tied = [i for i in range(n) if strah[i] == smax]
    # Auto pick: best of the tied set — free-ended first, then thickest, then
    # lowest index (mirrors build_directed_topology's tie-break).
    auto = max(tied, key=lambda i: (has_free(i), mean_radius(i), -i))

    # 1) Manual override by original seg id (always wins).
    forced = [i for i in range(n) if int(segments[i]["id"]) in INLET_SEG_OVERRIDE]
    if forced:
        chosen = forced[0]
        print(f"[INLET] graph {gid}: root/inlet forced to seg id "
              f"{int(segments[chosen]['id'])} via INLET_SEG_OVERRIDE")
        return {chosen}

    # 2) Max-Strahler tie -> list candidates, prompt when interactive.
    if len(tied) > 1:
        order = sorted(tied, key=lambda i: (not has_free(i), -mean_radius(i), i))
        print(f"[INLET][TIE] graph {gid}: {len(order)} segments tie at Strahler "
              f"{smax} — choose the inlet/root segment:")
        for k, i in enumerate(order):
            pos = inlet_node_pos(i)
            pos_s = (f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
                     if pos is not None else "n/a")
            tags = ("free-end" if has_free(i) else "internal — no free end")
            star = " *" if i == auto else ""
            print(f"  [{k}] seg id {int(segments[i]['id']):<5d} {tags:<22s} "
                  f"r={mean_radius(i) / 1000.0:.3f}mm  inlet@{pos_s}{star}")
        default_k = order.index(auto)

        def pick_3d():
            """Open the 3D root picker; return the chosen idx, or None on failure."""
            if points is None:
                print("  [INLET][WARN] no point table available here — "
                      "cannot open the 3D picker; use the list instead.")
                return None
            try:
                from .epicardial_annotation import run_root_picker  # noqa: PLC0415
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"  [INLET][WARN] 3D picker unavailable ({exc}); "
                      f"use the list instead.")
                return None
            try:
                return run_root_picker(
                    nodes, points, segments, candidates=set(tied), auto=auto,
                    label=f"graph {gid}: click the inlet segment "
                          f"(orange = the {len(tied)} tied candidates)")
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"  [INLET][WARN] 3D picker failed ({exc}); "
                      f"falling back to the list.")
                return None

        if INLET_PROMPT_ON_TIE and INLET_PICK_3D:
            picked = pick_3d()
            if picked is not None:
                print(f"  [INLET] graph {gid}: using seg id "
                      f"{int(segments[picked]['id'])} (add it to "
                      f"INLET_SEG_OVERRIDE to make permanent).")
                return {picked}
            print(f"  [INLET] graph {gid}: no 3D selection — using auto pick.")
            return {auto}

        if INLET_PROMPT_ON_TIE and sys.stdin.isatty():
            try:
                raw = input(f"Select inlet segment [0-{len(order) - 1}] "
                            f"(Enter=auto/*={default_k}, p=pick in 3D): ").strip()
            except EOFError:
                raw = ""
            chosen = auto
            if raw.lower() == "p":
                picked = pick_3d()
                if picked is not None:
                    chosen = picked
                else:
                    print("  [INLET] nothing picked — using auto pick.")
            elif raw:
                try:
                    k = int(raw)
                    if 0 <= k < len(order):
                        chosen = order[k]
                    else:
                        print(f"  [INLET] '{raw}' out of range — using auto pick.")
                except ValueError:
                    print(f"  [INLET] '{raw}' not a number — using auto pick.")
            print(f"  [INLET] graph {gid}: using seg id {int(segments[chosen]['id'])} "
                  f"(add it to INLET_SEG_OVERRIDE to make permanent).")
            return {chosen}
        # Non-interactive or prompting disabled.
        print(f"  [INLET] graph {gid}: auto-selected seg id "
              f"{int(segments[auto]['id'])} (set INLET_SEG_OVERRIDE to override).")
        return {auto}

    # 3) No tie, no override.
    return {auto}


def _directed_basics(nodes, points, segments, root_pref=None):
    """Lightweight directed-tree info for pre-split matching/restriction/crop
    classification (no flow split): a ``tree_ctx``-shaped dict with
    ``parent_idx``, ``seg_root``, ``leaves``, ``seg_centerline``, ``seg_id``.

    ``root_pref`` (optional) forces the component root (see
    :func:`coronary_sdf.topology.build_directed_topology`)."""
    node_to_segs: dict[int, set[int]] = {}
    for idx, seg in enumerate(segments):
        for nid in (seg["node1"], seg["node2"]):
            node_to_segs.setdefault(nid, set()).add(idx)
    parent_idx = build_directed_topology(
        segments, node_to_segs, root_pref=root_pref)["parent_seg_idx"]
    n = len(segments)
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for c in range(n):
        p = int(parent_idx[c])
        if p >= 0:
            children[p].append(c)
    roots = [i for i in range(n) if int(parent_idx[i]) < 0]
    seg_root = np.full(n, -1, dtype=np.int64)
    for r in roots:
        seg_root[r] = r
        q = [r]
        while q:
            cur = q.pop(0)
            for ch in children[cur]:
                seg_root[ch] = r
                q.append(ch)
    leaves = [i for i in range(n) if not children[i] and int(seg_root[i]) >= 0]
    return {
        "parent_idx": parent_idx,
        "seg_root": seg_root,
        "leaves": leaves,
        "seg_centerline": [_seg_centerline_mm(s, points) for s in segments],
        "seg_id": [int(s["id"]) for s in segments],
    }


def compute_flow_fractions(nodes, points, segments, crop_pt=None, outlet_segs=None,
                           root_pref=None):
    """Return ``(outlets, tree_inlets, tree_ctx)`` for one smoothed graph.

    ``outlets`` is a list of dicts (one per terminal/leaf segment):
        idx, tree_id, region_id (seg id), seg_idx (local), fraction
        (per-tree-normalised), diameter_mm, node_id, pos (mm), centerline.
    ``tree_inlets`` maps tree_id -> dict(node_id, pos, centerline).
    ``crop_pt`` (optional) maps ``seg_idx -> distal point index`` for segments the
    mesh crop lies in: those segments are truncated to a leaf at the crop (their
    children and beyond-crop points are dropped from the split), so spatial-graph
    geometry past the crop is ignored.

    ``outlet_segs`` (optional) is the set of seg idxs that own a matched .msh
    outlet. When given, any branch with no outlet anywhere in its subtree is
    *pruned* from the split: at the bifurcation where it diverges its diameter-law
    share is conserved into the surviving sibling branches (local redistribution),
    rather than being renormalised globally across all outlets later.

    ``tree_ctx`` exposes the (possibly truncated/pruned) directed tree:
        parent_idx, seg_root (−1 = beyond crop / pruned / unreachable), leaves,
        flow (per-seg, root=1.0), seg_diam, seg_centerline (crop-capped (N,3) mm),
        seg_id, is_crop (set of crop-leaf seg idxs), diam_samples.
    Flow fractions are normalised per tree (root) to sum to exactly 1.0.
    """
    crop_pt = dict(crop_pt) if crop_pt else {}

    node_to_segs: dict[int, set[int]] = {}
    for idx, seg in enumerate(segments):
        for nid in (seg["node1"], seg["node2"]):
            node_to_segs.setdefault(nid, set()).add(idx)

    dtopo = build_directed_topology(segments, node_to_segs, root_pref=root_pref)
    parent_idx = dtopo["parent_seg_idx"]
    n = len(segments)

    # children lists + per-segment root label.
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for c in range(n):
        p = int(parent_idx[c])
        if p >= 0:
            children[p].append(c)
    roots = [i for i in range(n) if int(parent_idx[i]) < 0]

    # Crop truncation: a cropped segment becomes a leaf — drop its children so the
    # BFS stops there and beyond-crop segments stay unreachable (seg_root = −1).
    crop_segs = set(int(s) for s in crop_pt) if TRUNCATE_AT_CROP else set()
    for cs in crop_segs:
        if 0 <= cs < n:
            children[cs] = []

    # Subtree-outlet pruning: keep only segments that own a matched .msh outlet or
    # are an ancestor of one. Branches absent from the mesh are pruned from the
    # split so their diameter-law share is conserved into the surviving siblings at
    # their bifurcation (local), not renormalised globally across all outlets.
    if outlet_segs is None:
        keep = [True] * n
        n_pruned_branches = 0
    else:
        keep = [False] * n
        for s in outlet_segs:
            cur = int(s)
            while 0 <= cur < n and not keep[cur]:
                keep[cur] = True
                cur = int(parent_idx[cur])
        # Count genuinely mesh-absent branches = roots of maximal pruned subtrees
        # whose kept parent is NOT a crop segment (beyond-crop children of a crop
        # are intentionally truncated, not "absent", so they are not counted).
        n_pruned_branches = sum(
            1 for c in range(n)
            if not keep[c] and int(parent_idx[c]) >= 0
            and keep[int(parent_idx[c])] and int(parent_idx[c]) not in crop_segs
        )

    def degree(nid):
        return nodes[nid][3] if nid in nodes else 0

    def shared_node_with_parent(c):
        p = int(parent_idx[c])
        cn = {segments[c]["node1"], segments[c]["node2"]}
        pn = {segments[p]["node1"], segments[p]["node2"]}
        common = cn & pn
        return next(iter(common)) if common else None

    def prox_node(c):
        """The proximal (parent / inlet) node of segment c."""
        if int(parent_idx[c]) >= 0:
            return shared_node_with_parent(c)
        n1, n2 = segments[c]["node1"], segments[c]["node2"]
        return n1 if degree(n1) == 1 else (n2 if degree(n2) == 1 else n1)

    def capped_pids(c):
        """Segment c's point ids, truncated proximal->crop if c is a crop seg."""
        pids = segments[c]["point_ids"]
        ci = crop_pt.get(int(c)) if c in crop_segs else None
        if ci is None or not (0 <= ci < len(pids)):
            return list(pids)
        return list(pids[:ci + 1]) if segments[c]["node1"] == prox_node(c) \
            else list(pids[ci:])

    def capped_centerline_mm(c):
        coords = [points[p][:3] for p in capped_pids(c) if p in points]
        return np.array(coords, dtype=np.float64) / 1000.0 if coords else np.empty((0, 3))

    def crop_point_pos(c):
        """World pos (mm) of the crop point on segment c, or None."""
        ci = crop_pt.get(int(c))
        pids = segments[c]["point_ids"]
        if ci is None or not (0 <= ci < len(pids)) or pids[ci] not in points:
            return None
        return np.array(points[pids[ci]][:3], dtype=np.float64) / 1000.0

    print("\n[FLOW] Computing Giessen splits...")
    # Per-junction split fractions: frac[c] = d_c^E / sum(d_k^E) over siblings.
    flow = np.zeros(n, dtype=np.float64)
    seg_diam = np.zeros(n, dtype=np.float64)
    seg_root = np.full(n, -1, dtype=np.int64)
    # Centreline points + radii each daughter diameter was measured over (for viz).
    diam_samples: list[dict[str, Any]] = []

    for root in roots:
        if not keep[root]:
            continue
        flow[root] = 1.0
        seg_root[root] = root
        queue = [root]
        while queue:
            cur = queue.pop(0)
            kids = [c for c in children[cur] if keep[c]]   # prune mesh-absent branches
            if not kids:
                continue
            diams = {}
            for c in kids:
                sh = shared_node_with_parent(c)
                if sh is None:
                    diams[c] = 0.0
                    seg_diam[c] = 0.0
                    continue
                d, used = _branch_diameter(
                    segments[c], sh, points, nodes, crop_limit=crop_pt.get(int(c)))
                diams[c] = d
                seg_diam[c] = d
                coords = np.array([points[p][:3] for p in used if p in points],
                                  dtype=np.float64) / 1000.0
                radii = np.array([_radius_mm(points, p) for p in used if p in points],
                                 dtype=np.float64)
                if len(coords):
                    diam_samples.append({
                        "seg_idx": int(c),
                        "coords": coords,
                        "radii": radii,
                        "diameter_mm": float(d),
                    })
            d_sum = sum(d ** GIESSEN_EXPONENT for d in diams.values())
            for c in kids:
                frac = (diams[c] ** GIESSEN_EXPONENT / d_sum) if d_sum > 0 else (1.0 / len(kids))
                flow[c] = flow[cur] * frac
                seg_root[c] = root
                queue.append(c)

    if n_pruned_branches:
        print(f"  [PRUNE] {n_pruned_branches} branch(es) absent from the mesh pruned "
              f"from the split; flow conserved into surviving siblings at each "
              f"junction (no global renormalisation).")

    # Inlet node per tree (root): the root segment's degree-1 (proximal) node.
    tree_inlets: dict[int, dict[str, Any]] = {}
    for root in roots:
        n1, n2 = segments[root]["node1"], segments[root]["node2"]
        if degree(n1) == 1 and degree(n2) != 1:
            inlet_node = n1
        elif degree(n2) == 1 and degree(n1) != 1:
            inlet_node = n2
        else:
            inlet_node = n1
            if degree(n1) == 1 and degree(n2) == 1:
                print(f"  [WARN] root segment {root} is an isolated single branch; "
                      f"using node {n1} as inlet.")
            else:
                print(f"  [WARN] root segment {root} has no degree-1 end; "
                      f"using node {n1} as inlet (check topology).")
        tree_inlets[root] = {
            "node_id": int(inlet_node),
            "pos": np.array(nodes[inlet_node][:3], dtype=np.float64) / 1000.0,
            "centerline": _seg_centerline_mm(segments[root], points),
        }

    # Leaves = reachable segments with no children (crop segs are leaves; beyond-
    # crop segments are unreachable, seg_root = −1, and excluded).
    leaves = [i for i in range(n) if not children[i] and int(seg_root[i]) >= 0]
    by_root: dict[int, list[int]] = {}
    for leaf in leaves:
        by_root.setdefault(int(seg_root[leaf]), []).append(leaf)

    outlets: list[dict[str, Any]] = []
    out_idx = 0
    for root, leaf_list in by_root.items():
        total = sum(flow[leaf] for leaf in leaf_list)
        running = 0.0
        for i, leaf in enumerate(leaf_list):
            is_crop = leaf in crop_segs
            # Outlet position: the crop point for a cropped leaf, else the distal node.
            if is_crop and crop_point_pos(leaf) is not None:
                pos = crop_point_pos(leaf)
                distal = -1
            else:
                n1, n2 = segments[leaf]["node1"], segments[leaf]["node2"]
                if int(parent_idx[leaf]) >= 0:
                    sh = shared_node_with_parent(leaf)
                    distal = n2 if sh == n1 else n1
                else:
                    inlet_node = tree_inlets[root]["node_id"]
                    distal = n2 if inlet_node == n1 else n1
                pos = np.array(nodes[distal][:3], dtype=np.float64) / 1000.0
            raw = flow[leaf] / total if total > 0 else 0.0
            # Enforce exact 1.0 per tree on the last leaf to kill FP drift.
            if i == len(leaf_list) - 1:
                frac = 1.0 - running
            else:
                frac = raw
                running += frac
            outlets.append({
                "idx": out_idx,
                "tree_id": int(root),
                "region_id": int(segments[leaf]["id"]),
                "seg_idx": int(leaf),
                "is_crop": bool(is_crop),
                "fraction": float(frac),
                "diameter_mm": float(seg_diam[leaf]),
                "node_id": int(distal),
                "pos": pos,
                "centerline": capped_centerline_mm(leaf),
            })
            out_idx += 1

    # Root segments have no parent junction, so the split loop never measured a
    # daughter diameter for them — use their whole-segment (crop-capped) mean.
    for root in roots:
        seg_diam[root] = _seg_mean_diameter(
            {"point_ids": capped_pids(root)}, points)

    # Directed tree (truncated at crops). Beyond-crop segments keep seg_root = −1
    # and an empty centreline so callers naturally skip them.
    seg_centerline = [
        (capped_centerline_mm(i) if int(seg_root[i]) >= 0 else np.empty((0, 3)))
        for i in range(n)
    ]
    tree_ctx = {
        "parent_idx": parent_idx,
        "seg_root": seg_root,
        "leaves": leaves,
        "flow": flow,
        "seg_diam": seg_diam,
        "seg_id": [int(s["id"]) for s in segments],
        "seg_centerline": seg_centerline,
        "is_crop": crop_segs,
        "diam_samples": diam_samples,
    }

    print(f"  {len(roots)} tree(s), {len(outlets)} terminal outlet(s)")
    return outlets, tree_inlets, tree_ctx


# ── Coordinate alignment + .msh matching ──────────────────────────────────────


def _bbox_diag(pts):
    if len(pts) == 0:
        return 0.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def diagnose_alignment(msh_pts, cl_points):
    """Report .msh-centroid → skeleton-centerline distances; flag scale errors.

    Because the Simpleware outlets are cropped, each ``.msh`` outlet plane sits
    somewhere *along* a terminal branch, so its centroid lies near that branch's
    centerline (not at the skeleton tip). We therefore measure how far each
    ``.msh`` centroid is from the nearest branch centerline point.

    ``msh_pts`` is already scaled/offset. The scale warning is gated on the
    *best-quartile* distance so a ``.msh`` that covers only part of the tree
    (many branches with no nearby centroid) does not trigger a false alarm.
    Returns an estimated uniform scale (skeleton/.msh), or 1.0 if aligned.
    """
    if len(msh_pts) == 0 or len(cl_points) == 0:
        print("[ALIGN] No points to compare.")
        return 1.0
    kd = cKDTree(cl_points)
    d, _ = kd.query(msh_pts)
    med = float(np.median(d))
    dq = float(np.percentile(d, 25))
    within = float(np.mean(d <= MATCH_TOL_MM)) * 100.0
    print(f"[ALIGN] {len(msh_pts)} .msh centroids vs {len(cl_points)} centerline pts")
    print(f"[ALIGN] centroid->centerline distance: median {med:.3f} mm, "
          f"best-quartile {dq:.3f} mm, within {MATCH_TOL_MM} mm: {within:.0f}%")
    if dq > MATCH_TOL_MM:
        m_diag = _bbox_diag(msh_pts)
        c_diag = _bbox_diag(cl_points)
        ratio = (c_diag / m_diag) if m_diag > 0 else 1.0
        print(f"[ALIGN][WARN] even the closest .msh centroids are > {MATCH_TOL_MM} mm "
              f"from any branch — likely a unit/registration mismatch.")
        if abs(ratio - 0.001) / 0.001 < 0.2:
            print("[ALIGN][WARN] .msh looks like it is in METRES — set MSH_SCALE = 0.001.")
        else:
            print(f"[ALIGN][WARN] estimated scale (skeleton/.msh) = {ratio:.4g} — set MSH_SCALE.")
        return ratio
    return 1.0


def _msh_centroid_array(msh_bounds, kind):
    rows = [b for b in msh_bounds if b["kind"] == kind]
    if not rows:
        return rows, np.empty((0, 3))
    pts = np.array([[b["x"], b["y"], b["z"]] for b in rows], dtype=np.float64)
    pts = pts * MSH_SCALE + np.asarray(MSH_OFFSET, dtype=np.float64)
    return rows, pts


def _assign_zones_to_branches(zone_rows, zone_pts, endpoints, prefix):
    """Assign each .msh zone to the nearest skeleton branch centerline.

    Crop-robust: matches a zone centroid to the closest point of any branch's
    *centerline* (not its terminal tip), so a cropped outlet plane anywhere
    along the branch still resolves to the right branch. Greedy 1:1 by ascending
    distance — each branch claims at most one zone. Mutates ``endpoints`` (adds
    region_name / ccl_name / match_dist). Returns the number of zones assigned.
    """
    for ep in endpoints:
        tag = ep.get("idx", ep.get("tree_id", "?"))
        ep["region_name"] = f"placeholder_region_{prefix}_{tag}"
        ep["ccl_name"] = f"PLACEHOLDER_{prefix}_{tag}"
        ep["match_dist"] = float("inf")
    if len(zone_pts) == 0 or not endpoints:
        return 0

    # Pool every branch centerline point with a parallel owner-endpoint index.
    pools, owners = [], []
    for ei, ep in enumerate(endpoints):
        cl = ep.get("centerline")
        if cl is None or len(cl) == 0:
            cl = ep["pos"][None, :]
        pools.append(cl)
        owners.append(np.full(len(cl), ei, dtype=np.int64))
    pool = np.vstack(pools)
    owner = np.concatenate(owners)
    kd = cKDTree(pool)

    k = int(min(len(pool), 128))
    dists, idxs = kd.query(zone_pts, k=k)
    dists = np.atleast_2d(dists)
    idxs = np.atleast_2d(idxs)

    # Closest zones claim their branch first.
    order = np.argsort(dists[:, 0])
    claimed: set[int] = set()
    n_assigned = 0
    for zj in order:
        for dd, pi in zip(dists[zj], idxs[zj]):
            ei = int(owner[pi])
            if ei in claimed:
                continue
            # First unclaimed owner is the nearest available branch; accept iff
            # within tolerance (neighbours are distance-sorted, so a miss here
            # means every farther branch misses too).
            if dd <= MATCH_TOL_MM:
                ep = endpoints[ei]
                ep["region_name"] = zone_rows[zj]["region_name"]
                ep["ccl_name"] = zone_rows[zj]["ccl_name"]
                ep["match_dist"] = float(dd)
                claimed.add(ei)
                n_assigned += 1
            break
    return n_assigned


def _tree_inlet_node_positions(graph_data):
    """``(gid, root) -> trunk inlet node world pos (mm)``.

    Inlet node = the root segment's proximal degree-1 node (the same rule
    :func:`compute_flow_fractions` uses), but read off the lightweight directed
    basics built in Loop A so it is available *before* the split / matching runs.
    ``nodes[nid] = (x, y, z, coordination_number)``; the coordination number is the
    node degree and coordinates are micrometres (-> /1000 for mm).
    """
    positions: dict[tuple, np.ndarray] = {}
    for gid, d in graph_data.items():
        nodes, segments = d["nodes"], d["segments"]
        seg_root = d["tree_ctx"]["seg_root"]
        roots = sorted({int(r) for r in seg_root if int(r) >= 0})

        def degree(nid):
            return nodes[nid][3] if nid in nodes else 0

        for root in roots:
            n1, n2 = segments[root]["node1"], segments[root]["node2"]
            if degree(n1) == 1 and degree(n2) != 1:
                inlet = n1
            elif degree(n2) == 1 and degree(n1) != 1:
                inlet = n2
            else:
                inlet = n1
            positions[(int(gid), root)] = (
                np.array(nodes[inlet][:3], dtype=np.float64) / 1000.0)
    return positions


def reclassify_zone_kinds_by_geometry(msh_bounds, inlet_positions):
    """Promote a mislabelled inlet zone using tree geometry WITHOUT overriding a
    properly-named inlet.

    A zone whose NAME matches ``INLET_ZONE_KEYWORDS`` is trusted and kept as
    ``inlet`` (never demoted). For each tree with NO properly-named inlet within
    ``INLET_DETECT_TOL_MM`` of its trunk root node, the nearest zone to that root
    node is promoted from ``outlet`` to ``inlet``. No zone is ever demoted. Greedy
    1:1 by ascending distance so two trees cannot claim the same zone. Mutates
    ``msh_bounds[*]['kind']`` and returns the resulting inlet count.
    """
    if not msh_bounds or not inlet_positions:
        return sum(1 for b in msh_bounds if b.get("kind") == "inlet")

    centroids = (np.array([[b["x"], b["y"], b["z"]] for b in msh_bounds],
                          dtype=np.float64) * MSH_SCALE
                 + np.asarray(MSH_OFFSET, dtype=np.float64))

    # Properly-named inlets (name matches an inlet keyword) are trusted and kept.
    inlet_kw = tuple(k.lower() for k in INLET_ZONE_KEYWORDS)
    named_inlet: set[int] = set()
    for zj, b in enumerate(msh_bounds):
        if any(kw in b["ccl_name"].lower() for kw in inlet_kw):
            named_inlet.add(zj)
            b["kind"] = "inlet"

    # A tree is "satisfied" (skip geometric promotion) if a properly-named inlet
    # already lies within tolerance of its root node.
    satisfied: set[tuple] = set()
    for tid, pos in inlet_positions.items():
        for zj in named_inlet:
            if float(np.linalg.norm(centroids[zj] - pos)) <= INLET_DETECT_TOL_MM:
                satisfied.add(tid)
                break

    # Candidate (dist, tree_id, zone_idx) for unsatisfied trees over non-named
    # zones, within tolerance, closest first.
    candidates: list[tuple[float, tuple, int]] = []
    for tid, pos in inlet_positions.items():
        if tid in satisfied:
            continue
        d = np.linalg.norm(centroids - pos[None, :], axis=1)
        for zj in range(len(msh_bounds)):
            if zj in named_inlet:
                continue
            if d[zj] <= INLET_DETECT_TOL_MM:
                candidates.append((float(d[zj]), tid, zj))
    candidates.sort(key=lambda c: c[0])

    # Greedy 1:1 promotion — each tree claims its nearest still-free zone.
    promoted: dict[int, tuple[tuple, float]] = {}   # zone_idx -> (tid, dist)
    claimed_trees: set[tuple] = set()
    for dist, tid, zj in candidates:
        if tid in claimed_trees or zj in promoted:
            continue
        promoted[zj] = (tid, dist)
        claimed_trees.add(tid)

    for zj, (tid, dist) in promoted.items():
        b = msh_bounds[zj]
        old_kind = b.get("kind")
        b["kind"] = "inlet"
        if old_kind != "inlet":
            print(f"[INLET] zone '{b['ccl_name']}' {old_kind}->inlet for tree "
                  f"{tid} ({dist:.3f} mm from root node)")

    if not promoted and not named_inlet:
        print(f"[INLET][WARN] no .msh zone within {INLET_DETECT_TOL_MM} mm of any "
              f"tree root node and none named as an inlet — keeping name-based labels.")

    return sum(1 for b in msh_bounds if b.get("kind") == "inlet")


def identify_mesh_trees(graph_trees, msh_bounds):
    """Return the set of tree_ids the ``.msh`` actually sits on, by coordinate
    proximity (NOT by outlet count).

    Each ``.msh`` outlet/inlet centroid is assigned to the tree whose centreline
    coordinates (every segment's points — these include the leaf/outlet-node tips)
    are closest to it. A tree is "meshed" if it is the closest tree for at least
    one centroid within ``MATCH_TOL_MM``. Returns ``set[tree_id]`` (``(gid, root)``);
    falls back to *all* trees if nothing lands within tolerance.
    """
    # Per-tree KD-tree over its centreline coordinates, keyed by (gid, root).
    tree_ids: list[tuple] = []
    kds = []
    for gid, gt in graph_trees.items():
        tc = gt["tree_ctx"]
        seg_root = tc["seg_root"]
        cls = tc["seg_centerline"]
        roots = sorted({int(r) for r in seg_root})
        for root in roots:
            pool = [cls[s] for s in range(len(cls))
                    if int(seg_root[s]) == root and len(cls[s])]
            if not pool:
                continue
            tree_ids.append((int(gid), root))
            kds.append(cKDTree(np.vstack(pool)))
    if not tree_ids:
        return {tid for tid in graph_trees}  # nothing to decide on

    _o_rows, out_pts = _msh_centroid_array(msh_bounds, "outlet")
    _i_rows, in_pts = _msh_centroid_array(msh_bounds, "inlet")
    pts = [p for p in (out_pts, in_pts) if len(p)]
    centroids = np.vstack(pts) if pts else np.empty((0, 3))

    votes = {tid: 0 for tid in tree_ids}
    dists_by_tree: dict[tuple, list[float]] = {tid: [] for tid in tree_ids}
    for c in centroids:
        ds = [float(kd.query(c[None, :])[0][0]) for kd in kds]
        j = int(np.argmin(ds))
        dists_by_tree[tree_ids[j]].append(ds[j])
        if ds[j] <= MATCH_TOL_MM:
            votes[tree_ids[j]] += 1

    print("[TREE] mesh-vs-tree proximity (by centroid coordinates):")
    for tid in tree_ids:
        dl = dists_by_tree[tid]
        med = f"{float(np.median(dl)):.3f} mm" if dl else "n/a"
        print(f"  tree {tid}: closest for {len(dl)} centroid(s) "
              f"({votes[tid]} within {MATCH_TOL_MM} mm), median nearest {med}")

    meshed = {tid for tid, v in votes.items() if v > 0}
    if not meshed:
        print("[TREE][WARN] no tree within tolerance of any .msh centroid — "
              "keeping all trees.")
        return {tid for tid in votes}
    return meshed


def _filter_to_trees(graph_trees, all_outlets, all_tree_inlets, kept_tree_ids):
    """Drop trees not in ``kept_tree_ids`` from the per-graph data + flat lists.

    Returns ``(graph_trees, all_outlets, all_tree_inlets)`` restricted to the kept
    trees. A graph is kept whole only if it contributes a kept tree; per-segment
    filtering by ``seg_root`` would be needed for multi-root graphs (none here).
    """
    kept_gids = {gid for gid, _root in kept_tree_ids}
    graph_trees = {g: gt for g, gt in graph_trees.items() if g in kept_gids}
    all_outlets = [o for o in all_outlets if o["tree_id"] in kept_tree_ids]
    all_tree_inlets = {tid: v for tid, v in all_tree_inlets.items()
                       if tid in kept_tree_ids}
    return graph_trees, all_outlets, all_tree_inlets


def _match_zones_to_segments(zone_rows, zone_pts, graph_trees):
    """Match each .msh outlet zone to the nearest point of *any* segment's
    centreline across every graph (not just leaf tips).

    A Simpleware crop plane sits part-way along a branch, so its centroid lands
    near that branch's centreline mid-tree; pooling every segment lets the crop
    plane resolve to the correct trunk rather than a random nearby leaf. Greedy
    1:1 by ascending distance — each segment claims at most one zone.

    Returns ``(zone_owner, owner_ptidx, n_assigned)`` where
    ``zone_owner[zone_idx] = (gid, seg_idx, match_dist)`` and
    ``owner_ptidx[zone_idx] = nearest point index within that segment`` (the crop
    location along the branch).
    """
    zone_owner: dict[int, tuple[int, int, float]] = {}
    owner_ptidx: dict[int, int] = {}
    if len(zone_pts) == 0 or not graph_trees:
        return zone_owner, owner_ptidx, 0

    owners_meta: list[tuple[int, int]] = []   # (gid, seg_idx) per pool block
    pools, owners, locals_ = [], [], []
    for gid, gt in graph_trees.items():
        for seg_idx, cl in enumerate(gt["tree_ctx"]["seg_centerline"]):
            if cl is None or len(cl) == 0:
                continue
            mi = len(owners_meta)
            owners_meta.append((int(gid), int(seg_idx)))
            pools.append(cl)
            owners.append(np.full(len(cl), mi, dtype=np.int64))
            locals_.append(np.arange(len(cl), dtype=np.int64))
    if not pools:
        return zone_owner, owner_ptidx, 0
    pool = np.vstack(pools)
    owner = np.concatenate(owners)
    local = np.concatenate(locals_)
    kd = cKDTree(pool)

    k = int(min(len(pool), 128))
    dists, idxs = kd.query(zone_pts, k=k)
    dists = np.atleast_2d(dists)
    idxs = np.atleast_2d(idxs)

    order = np.argsort(dists[:, 0])      # closest zones claim their segment first
    claimed: set[int] = set()
    n_assigned = 0
    for zj in order:
        for dd, pi in zip(dists[zj], idxs[zj]):
            mi = int(owner[pi])
            if mi in claimed:
                continue
            # First unclaimed segment is the nearest available branch; accept iff
            # within tolerance (neighbours are distance-sorted).
            if dd <= MATCH_TOL_MM:
                gid, seg_idx = owners_meta[mi]
                zone_owner[int(zj)] = (gid, seg_idx, float(dd))
                owner_ptidx[int(zj)] = int(local[pi])
                claimed.add(mi)
                n_assigned += 1
            break
    return zone_owner, owner_ptidx, n_assigned


def _classify_zones(graph_trees, zones_by_gid):
    """Label each matched zone terminal / crop / orphan from tree topology.

    terminal = owner is a graph leaf. crop = owner is internal with no other
    matched outlet in its subtree (the trunk was cut here). orphan = owner is
    internal *and* its subtree holds another matched outlet, so this zone cannot
    be the trunk's terminal cut — it is a stub branch absent from the graph that
    matched onto the trunk within tolerance. Returns ``zone_kind[zone_idx]``.
    """
    zone_kind: dict[int, str] = {}
    for gid, owner_segs in zones_by_gid.items():
        tc = graph_trees[gid]["tree_ctx"]
        parent_idx = tc["parent_idx"]
        leaf_set = {int(l) for l in tc["leaves"]}
        crop_set = {int(s) for s in tc.get("is_crop", set())}
        matched = set(owner_segs.keys())
        # A segment "has a deeper matched outlet" if it lies on the root-path of
        # some other matched owner (i.e. it is that owner's ancestor).
        has_deeper: set[int] = set()
        for x in matched:
            cur = int(parent_idx[x])
            while cur >= 0:
                if cur in matched:
                    has_deeper.add(cur)
                cur = int(parent_idx[cur])
        for s, zj in owner_segs.items():
            if s in crop_set:
                # Truncated crop segment (now a leaf) — keep it labelled "crop".
                zone_kind[zj] = "crop"
            elif s in leaf_set:
                zone_kind[zj] = "terminal"
            elif s in has_deeper:
                zone_kind[zj] = "orphan"
            else:
                zone_kind[zj] = "crop"
    return zone_kind


def aggregate_outlet_fractions(graph_trees, zone_owner, out_rows):
    """Attribute the full-tree Giessen split to the matched .msh outlet zones.

    Cut zones (terminal + crop) each receive the summed flow of every leaf whose
    deepest cut on its root->leaf path is that zone — so a crop plane carries all
    its removed downstream branches. Orphan zones (stub branches absent from the
    graph) are injected as an extra Giessen daughter at their attachment junction
    using a diameter measured from the .msh cross-section. Leaves served by no cut
    zone are reported as cropped/lost. Each tree is renormalised to exactly 1.0.

    Mutates each per-leaf outlet (adds ``in_mesh`` / ``served_zone_idx``). Returns
    ``(cfx_outlets, zone_to_cfx)`` — one dict per matched zone (CCL-compatible
    keys idx/fraction/tree_id/ccl_name/region_name plus kind/n_served/diam_mm/...),
    and a ``zone_idx -> cfx_outlet`` lookup.
    """
    # Group matched zones by their owning graph, then by owner segment.
    zones_by_gid: dict[int, dict[int, int]] = {}
    for zj, (gid, seg_idx, _d) in zone_owner.items():
        zones_by_gid.setdefault(int(gid), {})[int(seg_idx)] = int(zj)

    zone_kind = _classify_zones(graph_trees, zones_by_gid)

    # Cut set per graph = terminal + crop owners (orphans excluded so they never
    # capture the trunk subtree).
    cut_by_gid: dict[int, dict[int, int]] = {
        gid: {s: zj for s, zj in owner_segs.items() if zone_kind[zj] != "orphan"}
        for gid, owner_segs in zones_by_gid.items()
    }

    # Roots (trees) that own at least one matched zone — only these are "in the
    # mesh". A tree with no matched zone is simply absent from this mesh (a
    # different vessel), not "cropped", so its leaves are skipped below.
    meshed_roots: dict[int, set[int]] = {}
    for gid, owner_segs in zones_by_gid.items():
        sr = graph_trees[gid]["tree_ctx"]["seg_root"]
        meshed_roots[int(gid)] = {int(sr[s]) for s in owner_segs}

    zone_served: dict[int, float] = {int(zj): 0.0 for zj in zone_owner}
    zone_leafids: dict[int, list[int]] = {int(zj): [] for zj in zone_owner}
    lost_by_tree: dict[tuple, float] = {}
    dropped_by_tree: dict[tuple, int] = {}

    for gid, gt in graph_trees.items():
        tc = gt["tree_ctx"]
        parent_idx = tc["parent_idx"]
        seg_root = tc["seg_root"]
        flow = tc["flow"]
        cut_segs = cut_by_gid.get(int(gid), {})
        leaf_outlet = {o["seg_idx"]: o for o in gt["outlets"]}

        gid_meshed_roots = meshed_roots.get(int(gid), set())
        for leaf in tc["leaves"]:
            root = int(seg_root[leaf])
            if root not in gid_meshed_roots:
                continue  # tree absent from this mesh — not a crop, just ignore
            cur = int(leaf)
            serving = None
            while cur >= 0:
                if cur in cut_segs:
                    serving = cut_segs[cur]
                    break
                cur = int(parent_idx[cur])
            o = leaf_outlet.get(int(leaf))
            tid = (int(gid), root)
            if serving is None:
                lost_by_tree[tid] = lost_by_tree.get(tid, 0.0) + float(flow[leaf])
                dropped_by_tree[tid] = dropped_by_tree.get(tid, 0) + 1
                if o is not None:
                    o["in_mesh"] = False
                    o["served_zone_idx"] = None
            else:
                zone_served[serving] += float(flow[leaf])
                zone_leafids[serving].append(int(tc["seg_id"][int(leaf)]))
                if o is not None:
                    o["in_mesh"] = True
                    o["served_zone_idx"] = serving

    # Orphan injection: stub takes a Giessen d^E share of the trunk it attaches to.
    for gid, owner_segs in zones_by_gid.items():
        tc = graph_trees[gid]["tree_ctx"]
        flow = tc["flow"]
        pts = graph_trees[gid]["points"]
        segs = graph_trees[gid]["segments"]
        for s, zj in owner_segs.items():
            if zone_kind[zj] != "orphan":
                continue
            d_orphan = float(out_rows[zj].get("diam_mm", 0.0))
            d_trunk = _seg_mean_diameter(segs[s], pts)
            f_attach = float(flow[s])
            denom = d_orphan ** GIESSEN_EXPONENT + d_trunk ** GIESSEN_EXPONENT
            share = (d_orphan ** GIESSEN_EXPONENT / denom) if (denom > 0 and d_orphan > 0) else 0.0
            inj = f_attach * share if RECONSTRUCT_ORPHAN_STUBS else 0.0
            zone_served[zj] = inj
            print(f"[ORPHAN] {out_rows[zj]['ccl_name']}: stub d={d_orphan:.3f} mm on trunk "
                  f"d={d_trunk:.3f} mm (f_attach={f_attach:.4f}) -> share {share * 100:.2f}% "
                  f"= {inj:.5f} of inlet" + ("" if RECONSTRUCT_ORPHAN_STUBS else " [disabled]"))

    # Build one outlet per matched zone.
    cfx_outlets: list[dict[str, Any]] = []
    zone_to_cfx: dict[int, dict[str, Any]] = {}
    out_idx = 0
    for zj in sorted(zone_owner):
        gid, seg_idx, dist = zone_owner[zj]
        tc = graph_trees[gid]["tree_ctx"]
        root = int(tc["seg_root"][seg_idx])
        row = out_rows[zj]
        pos = (np.array([row["x"], row["y"], row["z"]], dtype=np.float64) * MSH_SCALE
               + np.asarray(MSH_OFFSET, dtype=np.float64))
        cfx = {
            "idx": out_idx,
            "tree_id": (int(gid), root),
            "kind": zone_kind[zj],
            "fraction": zone_served[zj],          # renormalised per tree below
            "ccl_name": row["ccl_name"],
            "region_name": row["region_name"],
            "n_served": len(zone_leafids[zj]),
            "served_region_ids": zone_leafids[zj],
            "diam_mm": float(row.get("diam_mm", 0.0)),
            "match_dist": float(dist),
            "pos": pos,
            "owner_seg_idx": int(seg_idx),
            "zone_idx": int(zj),
        }
        cfx_outlets.append(cfx)
        zone_to_cfx[int(zj)] = cfx
        out_idx += 1

    # Renormalise each tree (cut + injected orphan) to exactly 1.0.
    by_tree: dict[tuple, list[dict[str, Any]]] = {}
    for cfx in cfx_outlets:
        by_tree.setdefault(cfx["tree_id"], []).append(cfx)
    for tid, outs in by_tree.items():
        tsum = sum(c["fraction"] for c in outs)
        lost = lost_by_tree.get(tid, 0.0)
        if lost > 1e-9 or dropped_by_tree.get(tid):
            print(f"[CROP] Tree {tid}: {dropped_by_tree.get(tid, 0)} skeleton leaf(s) "
                  f"cropped with no .msh outlet — {lost * 100:.3f}% of inlet flow "
                  f"renormalised across the {len(outs)} surviving outlet(s).")
        n_orphan = sum(1 for c in outs if c["kind"] == "orphan")
        if n_orphan:
            print(f"[ORPHAN] Tree {tid}: {n_orphan} injected stub outlet(s); "
                  f"tree renormalised to 1.0.")
        if tsum <= 0:
            continue
        running = 0.0
        for i, c in enumerate(outs):
            if i == len(outs) - 1:
                c["fraction"] = 1.0 - running     # exact 1.0 per tree
            else:
                c["fraction"] = c["fraction"] / tsum
                running += c["fraction"]

    return cfx_outlets, zone_to_cfx


def match_all(graph_trees, all_tree_inlets, msh_bounds, zone_owner=None):
    """Match .msh inlet zones to tree inlets and aggregate the Giessen split into
    crop-plane outlets. ``zone_owner`` (outlet zone -> (gid, seg, dist)) may be
    precomputed (e.g. the pre-split match that drove crop truncation); if None it
    is computed here against the current centrelines.

    Returns ``(cfx_outlets, zone_to_cfx)``. Mutates per-leaf outlets (in_mesh /
    served_zone_idx) and ``all_tree_inlets`` values (region_name/ccl_name/match_dist).
    """
    out_rows, out_pts = _msh_centroid_array(msh_bounds, "outlet")
    in_rows, in_pts = _msh_centroid_array(msh_bounds, "inlet")

    # Alignment diagnostic against every segment centreline (full tree).
    cl_pool = [cl for gt in graph_trees.values()
               for cl in gt["tree_ctx"]["seg_centerline"] if len(cl)]
    cl_all = np.vstack(cl_pool) if cl_pool else np.empty((0, 3))
    diagnose_alignment(out_pts, cl_all)

    if zone_owner is None:
        zone_owner, _ptidx, n_out = _match_zones_to_segments(out_rows, out_pts, graph_trees)
    else:
        n_out = len(zone_owner)
    inlet_list = list(all_tree_inlets.values())
    n_in = _assign_zones_to_branches(in_rows, in_pts, inlet_list, "INLET")

    cfx_outlets, zone_to_cfx = aggregate_outlet_fractions(graph_trees, zone_owner, out_rows)

    print(f"[MATCH] .msh outlet zones assigned: {n_out}/{len(out_rows)} "
          f"(matched to {len(cfx_outlets)} crop-plane outlet(s))")
    print(f"[MATCH] .msh inlet zones assigned:  {n_in}/{len(in_rows)}")
    n_unassigned_zones = len(out_rows) - n_out
    if n_unassigned_zones > 0:
        print(f"[MATCH][WARN] {n_unassigned_zones} .msh outlet zone(s) had no branch "
              f"within {MATCH_TOL_MM} mm — check MSH_SCALE / MATCH_TOL_MM.")
    return cfx_outlets, zone_to_cfx


def _format_zone_name(prefix: str, idx: int) -> str:
    if idx < 0 or idx >= 10 ** RENAME_MSH_NUM_DIGITS:
        raise ValueError(
            f"Cannot format {prefix}{idx}: exceeds {RENAME_MSH_NUM_DIGITS} digits. "
            "Increase RENAME_MSH_NUM_DIGITS or reduce the outlet/inlet count."
        )
    return f"{prefix}{idx:0{RENAME_MSH_NUM_DIGITS}d}"


def _next_available_index(used: set[int], start: int) -> int:
    limit = 10 ** RENAME_MSH_NUM_DIGITS
    idx = start
    while idx < limit and idx in used:
        idx += 1
    if idx >= limit:
        raise ValueError(
            "Ran out of Outlet_XXX/Inlet_XXX names. "
            "Increase RENAME_MSH_NUM_DIGITS or reduce the outlet/inlet count."
        )
    return idx


def _ensure_raw_names(outlets, tree_inlets) -> None:
    for o in outlets:
        o.setdefault("region_name_raw", o.get("region_name"))
        o.setdefault("ccl_name_raw", o.get("ccl_name"))
    for v in tree_inlets.values():
        v.setdefault("region_name_raw", v.get("region_name"))
        v.setdefault("ccl_name_raw", v.get("ccl_name"))


_ZONE45_RE = re.compile(
    r"^(\(45 \()(?P<id>\d+) (?P<type>\S+) (?P<name>\S+)"
    r"(?P<tail> \d+\) \(\))"
)
_FACE_ZONE_HEADER_RE = re.compile(
    r"^(?P<head>\(13 \()(?P<id>[0-9a-fA-F]+)"
    r"(?P<range>\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+)"
    r"(?P<bc>[0-9a-fA-F]+)(?P<tail>\s+\d+\)\s*\()"
)


def _required_fluent_zone_type(new_name: str) -> tuple[str, str] | None:
    """Return Fluent metadata type and face-header code for a cap zone."""
    if new_name.startswith(RENAME_MSH_OUTLET_PREFIX):
        return "pressure-outlet", "5"
    if new_name.startswith(RENAME_MSH_INLET_PREFIX):
        # CFX preserves enough of the imported Fluent patch class that a
        # Bulk Mass Flow Rate condition on a velocity-inlet locale cannot
        # retrieve DENSITY during CAL_BCP_CDB.  Fluent boundary code 20 is
        # hexadecimal ``14`` in the ASCII face-zone header.
        return "mass-flow-inlet", "14"
    return None


def _rewrite_msh_zone_names(src_path: str | Path, dst_path: Path, rename_map: dict[str, str]) -> int:
    """Rewrite zone names in both the ``(0 " zone-name: ")`` comments *and* the
    ``(45 (id type NAME 1) ())`` zone-table records. CFX reads the names from the
    ``(45)`` records, so renaming only the comments (the old behaviour) left the
    regions un-renamed in CFX-Pre."""
    # Face-zone headers precede the (45) zone table in Fluent ASCII files.  A
    # first pass is therefore needed to map the decimal table ID onto the raw
    # zone name before we can update the header's hexadecimal boundary code.
    zone_id_targets: dict[int, tuple[str, str, str]] = {}
    with open(src_path, "r") as src:
        for line in src:
            if not line.startswith("(45 ("):
                continue
            m = _ZONE45_RE.match(line)
            if not m:
                continue
            raw = m.group("name")
            new = rename_map.get(raw)
            required = _required_fluent_zone_type(new or "")
            if new and required:
                zone_id_targets[int(m.group("id"))] = (new, required[0], required[1])

    replaced = 0
    with open(src_path, "r") as src, open(dst_path, "w") as dst:
        for line in src:
            if '(0 " zone-name:' in line:
                start = line.find("zone-name:") + 10
                end = line.rfind('"')
                if end > start:
                    old = line[start:end].strip()
                    new = rename_map.get(old)
                    if new and new != old:
                        line = f"{line[:start]}{new}{line[end:]}"
                        replaced += 1
            elif line.startswith("(45 ("):
                m = _ZONE45_RE.match(line)
                if m:
                    old_name = m.group("name")
                    new = rename_map.get(old_name)
                    required = _required_fluent_zone_type(new or "")
                    new_type = required[0] if required else m.group("type")
                    if new and (new != old_name or new_type != m.group("type")):
                        line = (
                            f"{m.group(1)}{m.group('id')} {new_type} {new}"
                            f"{m.group('tail')}{line[m.end():]}"
                        )
                        replaced += 1
            elif line.startswith("(13 ("):
                m = _FACE_ZONE_HEADER_RE.match(line)
                if m:
                    target = zone_id_targets.get(int(m.group("id"), 16))
                    if target and m.group("bc").lower() != target[2]:
                        line = (
                            f"{m.group('head')}{m.group('id')}{m.group('range')}"
                            f"{target[2]}{m.group('tail')}{line[m.end():]}"
                        )
                        replaced += 1
            dst.write(line)
    return replaced


def _renamed_msh_zone_types_current(path: Path) -> bool:
    """Confirm every stable cap name has a solver-compatible Fluent type."""
    seen = 0
    with path.open("r") as stream:
        for line in stream:
            if not line.startswith("(45 ("):
                continue
            match = _ZONE45_RE.match(line)
            if not match:
                continue
            required = _required_fluent_zone_type(match.group("name"))
            if not required:
                continue
            seen += 1
            if match.group("type") != required[0]:
                return False
    return seen > 0


def rename_msh_zones(msh_path: str | Path, output_dir: Path, msh_bounds, cfx_outlets, tree_inlets) -> Path | None:
    """Rename each matched .msh outlet *zone* to Outlet_XXX (one per crop-plane
    outlet in ``cfx_outlets``) and each matched inlet to Inlet_XXX. Mutates the
    cfx_outlet / inlet dicts so downstream CSV/CCL use the renamed labels."""
    if not msh_bounds:
        return None

    _ensure_raw_names(cfx_outlets, tree_inlets)

    rename_map: dict[str, str] = {}
    outlet_used: set[int] = set()
    inlet_used: set[int] = set()

    for o in cfx_outlets:
        raw = o.get("ccl_name_raw") or o.get("ccl_name")
        if not raw:
            continue
        new_name = _format_zone_name(RENAME_MSH_OUTLET_PREFIX, int(o["idx"]))
        existing = rename_map.get(raw)
        if existing and existing != new_name:
            print(f"[RENAME][WARN] Zone '{raw}' already mapped to '{existing}', "
                  f"ignoring '{new_name}'.")
            new_name = existing
        rename_map[raw] = new_name
        outlet_used.add(int(o["idx"]))
        o["region_name"] = new_name
        o["ccl_name"] = new_name

    inlet_idx = 0
    for _tid, inlet in tree_inlets.items():
        if not np.isfinite(inlet.get("match_dist", float("inf"))):
            continue
        raw = inlet.get("ccl_name_raw") or inlet.get("ccl_name")
        if not raw:
            continue
        new_name = _format_zone_name(RENAME_MSH_INLET_PREFIX, inlet_idx)
        existing = rename_map.get(raw)
        if existing and existing != new_name:
            print(f"[RENAME][WARN] Zone '{raw}' already mapped to '{existing}', "
                  f"ignoring '{new_name}'.")
            new_name = existing
        rename_map[raw] = new_name
        inlet_used.add(inlet_idx)
        inlet["region_name"] = new_name
        inlet["ccl_name"] = new_name
        inlet_idx += 1

    out_rows = [b for b in msh_bounds if b.get("kind") == "outlet"]
    next_out_idx = 0
    for row in out_rows:
        raw = row.get("ccl_name")
        if not raw or raw in rename_map:
            continue
        next_out_idx = _next_available_index(outlet_used, next_out_idx)
        new_name = _format_zone_name(RENAME_MSH_OUTLET_PREFIX, next_out_idx)
        rename_map[raw] = new_name
        outlet_used.add(next_out_idx)
        next_out_idx += 1

    in_rows = [b for b in msh_bounds if b.get("kind") == "inlet"]
    next_in_idx = 0
    for row in in_rows:
        raw = row.get("ccl_name")
        if not raw or raw in rename_map:
            continue
        next_in_idx = _next_available_index(inlet_used, next_in_idx)
        new_name = _format_zone_name(RENAME_MSH_INLET_PREFIX, next_in_idx)
        rename_map[raw] = new_name
        inlet_used.add(next_in_idx)
        next_in_idx += 1

    if not rename_map:
        return None

    src = Path(msh_path)
    dst = output_dir / f"{src.stem}{RENAME_MSH_OUTPUT_SUFFIX}{src.suffix}"
    if (dst.is_file() and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns
            and _renamed_msh_zone_types_current(dst)):
        print(f"[RENAME] Reusing current renamed mesh {dst}")
        return dst
    replaced = _rewrite_msh_zone_names(src, dst, rename_map)
    print(f"[RENAME] Updated {replaced} zone-name line(s) in {dst}")
    return dst


# ── Orchestration + output ────────────────────────────────────────────────────


def _select_stable_cfx_opening(cfx_outlets, graph_trees, terminal_id=None):
    """Map a stable ``edgeN:nodeM`` Amira terminal onto one remeshed cap."""
    if not cfx_outlets:
        raise RuntimeError("No CFX outlet zones are available for opening selection")
    requested_edge = None
    if terminal_id:
        match = re.search(r"edge(\d+):node(\d+)", str(terminal_id))
        if not match:
            raise ValueError("Invalid opening terminal ID: {}".format(terminal_id))
        requested_edge = int(match.group(1))
    candidates = []
    for outlet in cfx_outlets:
        gid = int(outlet["tree_id"][0])
        owner = int(outlet["owner_seg_idx"])
        original_edge = int(graph_trees[gid]["tree_ctx"]["seg_id"][owner])
        outlet["owner_edge_id"] = original_edge
        if requested_edge is None or (
            original_edge == requested_edge
            or requested_edge in {int(v) for v in outlet.get("served_region_ids", [])}
        ):
            candidates.append(outlet)
    if requested_edge is not None:
        if len(candidates) != 1:
            raise RuntimeError(
                "Stable opening {} mapped to {} cap zones; exactly one is required"
                .format(terminal_id, len(candidates))
            )
        selected = candidates[0]
    else:
        # Fallback is deterministic but explicit stable IDs are preferred.
        selected = max(candidates, key=lambda row: (len(row.get("served_region_ids", [])), row["owner_edge_id"]))
    selected = dict(selected)
    selected["opening_terminal_id"] = terminal_id or "edge{}:node_unknown".format(selected["owner_edge_id"])
    selected["opening_ccl_name"] = "Pressure_Opening"
    selected["surface_area_mm2"] = math.pi * (0.5 * float(selected.get("diam_mm", 0.0))) ** 2
    return selected


def run(xml_path, msh_path, output_dir, surface_vtk=None,
        opening_terminal_id=None, boundary_only=False,
        inlet_mass_flow_kg_s=None):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "giessen_outlet_flow_fractions.csv"
    cfx_csv = out_dir / "giessen_cfx_outlet_flow_fractions.csv"
    output_ccl = out_dir / "giessen_boundary_conditions.ccl"

    msh_bounds = []
    domain_location = ""
    if os.path.exists(msh_path):
        msh_bounds, domain_location = extract_msh_boundaries(msh_path)
    else:
        print(f"[WARN] .msh file {msh_path} not found — boundary names will be placeholders.")

    nodes, points, segments = parse_xml(xml_path)
    nodes, points, segments = preprocess_topology(nodes, points, segments)

    graphs = split_by_graph(nodes, points, segments)
    print(f"  {len(graphs)} graph(s) detected")

    # Loop A: smooth each graph + lightweight directed basics (centrelines +
    # topology), used to restrict to the meshed tree and locate crops *before* the
    # split runs.
    graph_data: dict[int, dict[str, Any]] = {}
    root_pref_by_gid: dict[int, set[int]] = {}
    for gid, (g_nodes, g_points, g_segments) in sorted(graphs.items()):
        print("\n" + "=" * 60)
        print(f"  GRAPH {gid}: {len(g_nodes)} nodes, {len(g_segments)} segments")
        print("=" * 60)
        g_points = smooth_graph(g_nodes, g_points, g_segments,
                                smooth_radii=not config.FLOW_SPLIT_USE_RAW_RADII)
        rp = _resolve_root_pref(int(gid), g_nodes, g_segments, points=g_points)
        root_pref_by_gid[int(gid)] = rp
        graph_data[int(gid)] = {
            "nodes": g_nodes, "points": g_points, "segments": g_segments,
            "tree_ctx": _directed_basics(g_nodes, g_points, g_segments, root_pref=rp),
        }

    if INLET_SEG_OVERRIDE:
        seen = {int(s["id"]) for d in graph_data.values() for s in d["segments"]}
        missing = sorted(set(INLET_SEG_OVERRIDE) - seen)
        if missing:
            print(f"[INLET][WARN] INLET_SEG_OVERRIDE seg id(s) {missing} match no "
                  f"segment in any graph — ignored.")

    # Override name-based inlet/outlet labels using geometry, so an inlet face
    # auto-named with an outlet keyword (e.g. "...-plane"/"...-outflow") is not
    # matched as a spurious crop outlet. Run over all trees before restriction:
    # non-meshed trees' root nodes fall outside tolerance of every zone.
    if AUTO_DETECT_INLET and msh_bounds:
        inlet_positions = _tree_inlet_node_positions(graph_data)
        reclassify_zone_kinds_by_geometry(msh_bounds, inlet_positions)

    # The .msh may contain only one of several spatial-graph trees; keep the
    # tree(s) it sits on by coordinate proximity and ignore the rest (Part 5).
    if RESTRICT_TO_MESH_TREES and msh_bounds and len(graph_data) > 1:
        meshed_gids = {gid for gid, _root in identify_mesh_trees(graph_data, msh_bounds)}
        n_before = len(graph_data)
        graph_data = {g: d for g, d in graph_data.items() if g in meshed_gids}
        if n_before - len(graph_data):
            print(f"[TREE] Restricting to {len(graph_data)} mesh tree(s); "
                  f"ignoring {n_before - len(graph_data)} other tree(s) by proximity.")

    # Match outlet zones to (gid, seg, point) and pick genuine crop locations
    # (terminal/crop owners, NOT orphan stubs) so the split can truncate there.
    zone_owner: dict[int, tuple] = {}
    crop_pt_by_gid: dict[int, dict[int, int]] = {}
    outlet_segs_by_gid: dict[int, set[int]] = {}
    if msh_bounds:
        out_rows0, out_pts0 = _msh_centroid_array(msh_bounds, "outlet")
        zone_owner, owner_ptidx, _n = _match_zones_to_segments(out_rows0, out_pts0, graph_data)
        zones_by_gid0: dict[int, dict[int, int]] = {}
        for zj, (gid, seg, _d) in zone_owner.items():
            zones_by_gid0.setdefault(int(gid), {})[int(seg)] = int(zj)
            outlet_segs_by_gid.setdefault(int(gid), set()).add(int(seg))
        zone_kind0 = _classify_zones(graph_data, zones_by_gid0)
        if TRUNCATE_AT_CROP:
            # Only genuine crops (an internal segment cut mid-branch) are truncated
            # to a leaf. True leaf tips ("terminal") need no truncation; orphan
            # stubs must NOT truncate the trunk they matched onto.
            for zj, (gid, seg, _d) in zone_owner.items():
                if zone_kind0.get(zj) == "crop":
                    crop_pt_by_gid.setdefault(int(gid), {})[int(seg)] = owner_ptidx[zj]

    # Loop B: Giessen split on the (crop-truncated) tree.
    all_outlets: list[dict[str, Any]] = []
    all_tree_inlets: dict[Any, dict[str, Any]] = {}
    graph_trees: dict[int, dict[str, Any]] = {}
    out_idx_base = 0
    for gid in sorted(graph_data):
        d = graph_data[gid]
        outlets, tree_inlets, tree_ctx = compute_flow_fractions(
            d["nodes"], d["points"], d["segments"], crop_pt=crop_pt_by_gid.get(gid),
            outlet_segs=outlet_segs_by_gid.get(gid),
            root_pref=root_pref_by_gid.get(gid))
        for o in outlets:
            o["idx"] += out_idx_base
            o["tree_id"] = (gid, o["tree_id"])
        out_idx_base += len(outlets)
        remapped = {}
        for k, v in tree_inlets.items():
            v["tree_id"] = (gid, k)
            remapped[(gid, k)] = v
        all_outlets.extend(outlets)
        all_tree_inlets.update(remapped)
        graph_trees[int(gid)] = {
            "nodes": d["nodes"], "points": d["points"], "segments": d["segments"],
            "tree_ctx": tree_ctx, "outlets": outlets, "tree_inlets": tree_inlets,
        }

    # Inlet match + aggregate crops/orphans, reusing the pre-split outlet matches.
    if msh_bounds:
        cfx_outlets, zone_to_cfx = match_all(
            graph_trees, all_tree_inlets, msh_bounds, zone_owner=zone_owner)
    else:
        cfx_outlets, zone_to_cfx = [], {}

    if RENAME_MSH_ZONES:
        if os.path.exists(msh_path) and cfx_outlets:
            rename_msh_zones(msh_path, out_dir, msh_bounds, cfx_outlets, all_tree_inlets)
        elif not os.path.exists(msh_path):
            print("[MSH][WARN] Mesh file not found; skipping zone renaming.")

    if boundary_only:
        # The immutable CFX seed names its physics boundary objects with an
        # underscore (Inlet_000 / Outlet_000), while the reload mesh regions
        # use the compact Inlet000 / Outlet000 names written above. Keeping
        # these concepts separate makes an Append CCL import replace all old
        # seed boundary objects instead of leaving invalid duplicates behind.
        for c in cfx_outlets:
            c["boundary_name"] = "Outlet_{:03d}".format(int(c["idx"]))
        matched_inlet_index = 0
        for _tid, inlet in all_tree_inlets.items():
            if np.isfinite(inlet.get("match_dist", float("inf"))):
                inlet["boundary_name"] = "Inlet_{:03d}".format(matched_inlet_index)
                matched_inlet_index += 1

    # Report per tree (aggregated CFX outlets).
    print("\nCFX Outlet Flow Fractions By Tree:")
    cfx_by_tree: dict[tuple, list[dict[str, Any]]] = {}
    for c in cfx_outlets:
        cfx_by_tree.setdefault(c["tree_id"], []).append(c)
    for tid in sorted(cfx_by_tree, key=str):
        outs = cfx_by_tree[tid]
        tsum = sum(c["fraction"] for c in outs)
        kinds = {}
        for c in outs:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        kind_str = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        inlet_name = all_tree_inlets.get(tid, {}).get("region_name")
        print(f"  Tree {tid}: {len(outs)} outlet(s) [{kind_str}], sum={tsum:.6f} "
              f"| inlet -> {inlet_name}")

    def _served_name(o):
        if o.get("in_mesh") is False:
            return "DROPPED"
        zi = o.get("served_zone_idx")
        if zi is None:
            return ""
        c = zone_to_cfx.get(zi)
        return c["ccl_name"] if c else ""

    # Per-leaf CSV: full-tree Giessen split + crop accounting.
    with open(output_csv, "w", newline="") as f:
        fields = ["idx", "tree_id", "region_id", "fraction", "percentage",
                  "diameter_mm", "in_mesh", "served_ccl_name"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for o in all_outlets:
            writer.writerow({
                "idx": o["idx"],
                "tree_id": o["tree_id"],
                "region_id": o["region_id"],
                "fraction": o["fraction"],
                "percentage": f"{o['fraction'] * 100:.6f}%",
                "diameter_mm": f"{o['diameter_mm']:.4f}",
                "in_mesh": o.get("in_mesh", ""),
                "served_ccl_name": _served_name(o),
            })
    print(f"\nSaved per-leaf flow fractions to {output_csv}")

    # Per-CFX-outlet CSV: one row per .msh outlet zone (what CFX consumes).
    opening = None
    if opening_terminal_id is not None:
        opening = _select_stable_cfx_opening(
            cfx_outlets, graph_trees, opening_terminal_id
        )
        opening_path = out_dir / "opening_boundary.json"
        serializable = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in opening.items()
            if key not in {"served_region_ids"}
        }
        serializable["served_region_ids"] = [
            int(v) for v in opening.get("served_region_ids", [])
        ]
        opening_path.write_text(json.dumps(serializable, indent=2) + "\n")
        print(
            "[OPENING] {} -> {} at ({:.4f}, {:.4f}, {:.4f}) mm, "
            "area~{:.6f} mm^2".format(
                opening_terminal_id, opening["region_name"],
                *opening["pos"], opening["surface_area_mm2"]
            )
        )

    with open(cfx_csv, "w", newline="") as f:
        fields = ["idx", "ccl_name", "kind", "fraction", "percentage", "n_served",
                  "served_region_ids", "diam_mm", "tree_id", "x", "y", "z",
                  "match_dist_mm", "boundary_role", "stable_terminal_id"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in cfx_outlets:
            writer.writerow({
                "idx": c["idx"],
                "ccl_name": c.get("boundary_name", c["ccl_name"]),
                "kind": c["kind"],
                "fraction": c["fraction"],
                "percentage": f"{c['fraction'] * 100:.6f}%",
                "n_served": c["n_served"],
                "served_region_ids": ";".join(str(r) for r in c["served_region_ids"]),
                "diam_mm": f"{c['diam_mm']:.4f}",
                "tree_id": c["tree_id"],
                "x": f"{c['pos'][0]:.4f}",
                "y": f"{c['pos'][1]:.4f}",
                "z": f"{c['pos'][2]:.4f}",
                "match_dist_mm": f"{c['match_dist']:.3f}",
                "boundary_role": "opening" if opening is not None and c["zone_idx"] == opening["zone_idx"] else "fixed_outlet",
                "stable_terminal_id": opening_terminal_id if opening is not None and c["zone_idx"] == opening["zone_idx"] else "",
            })
    print(f"Saved per-CFX-outlet flow fractions to {cfx_csv}")

    # CCL: aggregated outlets for trees that also have a matched inlet zone.
    matched_inlets = {
        tid: v for tid, v in all_tree_inlets.items()
        if np.isfinite(v.get("match_dist", float("inf")))
    }
    ccl_outlets = [
        c for c in cfx_outlets
        if c["tree_id"] in matched_inlets
        and not (opening is not None and c["zone_idx"] == opening["zone_idx"])
    ]
    skipped = len(cfx_outlets) - len(ccl_outlets)
    if skipped:
        print(f"[CCL] Skipping {skipped} outlet(s) whose tree has no matched .msh inlet zone.")
    if not ccl_outlets:
        print("[CCL][WARN] No outlets matched to .msh zones with a matched inlet — "
              "skipping CCL export. Check the .msh zone names / alignment.")
    else:
        # Fractions already represent inlet fractions.  Once one cap becomes the
        # pressure opening, deliberately retain sum=1-f_opening so conservation
        # supplies the residual; never renormalise the fixed outlets.
        ccl_rows = [dict(c) for c in ccl_outlets]
        generate_cfx_ccl(
            matched_inlets, ccl_rows, output_ccl, domain_location,
            opening=opening, inlet_mass_flow_kg_s=inlet_mass_flow_kg_s,
            boundary_only=boundary_only,
        )

    # CFX remeshing only needs the renamed mesh, mapping tables, opening audit,
    # and boundary CCL.  The plotting code below can launch Graphviz/PyVista and
    # leave a console run waiting after those deliverables have already been
    # written, so keep boundary-only preparation strictly non-interactive.
    if boundary_only:
        print("[CFX] Boundary-only package complete; skipping visualisation.",
              flush=True)
        return

    # Flow-tree diagram (written before the blocking 3D window so it always saves).
    if VISUALIZE_TREE_GRAPH:
        tree_graph = build_flow_tree_graph(graph_trees, cfx_outlets)
        visualize_flow_tree(tree_graph, out_dir)

    # Resolve the lumen surface once; both viewers use it.
    surf_vtk = None
    if VISUALIZE_SURFACE_FLOW or VISUALIZE_COMBINED:
        if surface_vtk is not None and str(surface_vtk).strip().lower() == "ask":
            surf_vtk = _prompt_surface_vtk(out_dir)
        else:
            surf_vtk = surface_vtk or _discover_surface_vtk(out_dir)
        if surf_vtk is None:
            print("[SURF] No lumen-surface .vtk found in the output dir "
                  "(set SURFACE_VTK to point at one); "
                  + ("colouring the .msh wall zones instead."
                     if SURFACE_FALLBACK_TO_MSH else
                     "the flow-coloured surface layer will be missing."))

    def _viz_msh_surface(needed_for_colour=False):
        """The .msh surface for the viewer, or None when it is switched off.

        ``needed_for_colour`` overrides ``VIZ_SHOW_CFX_MESH``: with no lumen .vtk
        the .msh wall zones are the only surface left to colour, so it is read
        even though the mesh layer itself is off.
        """
        if not (VIZ_SHOW_CFX_MESH or needed_for_colour):
            print("[VIS] CFX mesh layer off (VIZ_SHOW_CFX_MESH=False) -- skipping "
                  "the .msh read for a faster viewer.")
            return None
        if not os.path.exists(msh_path):
            return None
        if needed_for_colour and not VIZ_SHOW_CFX_MESH:
            print("[VIS] No lumen .vtk to colour -- reading the .msh so its wall "
                  "zones can be coloured instead.")
        return extract_msh_surface(msh_path)[0]

    if VISUALIZE_COMBINED:
        # One window: surface + mesh + centrelines + labelled outlets, each layer
        # toggleable with its own opacity slider.
        need_msh_colour = surf_vtk is None and SURFACE_FALLBACK_TO_MSH
        visualize_combined(graph_trees, cfx_outlets, all_outlets, all_tree_inlets,
                           _viz_msh_surface(need_msh_colour),
                           surface_path=surf_vtk, out_dir=out_dir)
    else:
        # Lumen-surface flow map (before the blocking flow-split window).
        if VISUALIZE_SURFACE_FLOW and surf_vtk is not None:
            visualize_surface_flow(graph_trees, surf_vtk, out_dir=out_dir)

        if VISUALIZE:
            visualize_flow_split(graph_trees, cfx_outlets, all_outlets,
                                 all_tree_inlets, _viz_msh_surface())

    return all_outlets, cfx_outlets


def _build_centerline_polydata(graph_trees, pv):
    """One PolyData of every smoothed segment centreline, cell-coloured by a
    running segment index (turbo). Returns the PolyData or None."""
    line_pts: list[np.ndarray] = []
    line_conn: list[int] = []
    seg_scalar: list[int] = []
    offset = 0
    sidx = 0
    for gid in sorted(graph_trees):
        for cl in graph_trees[gid]["tree_ctx"]["seg_centerline"]:
            if cl is None or len(cl) < 2:
                sidx += 1
                continue
            n_p = len(cl)
            line_pts.append(np.asarray(cl, dtype=np.float64))
            line_conn.append(n_p)
            line_conn.extend(range(offset, offset + n_p))
            seg_scalar.append(sidx)
            offset += n_p
            sidx += 1
    if not line_pts:
        return None
    poly = pv.PolyData(np.vstack(line_pts), lines=np.asarray(line_conn, dtype=np.int64))
    poly.cell_data["seg_idx"] = np.asarray(seg_scalar, dtype=np.int32)
    return poly


def _build_diameter_contours(graph_trees, pv, compute_frenet_frame, n_circle_pts=16):
    """Ring polylines (one per diameter-sample point, radius = measured radius,
    perpendicular to the local tangent) + the centreline points used. Returns
    ``(rings_polydata_or_None, dots_polydata_or_None)``."""
    ring_pts: list[np.ndarray] = []
    ring_lines: list[int] = []
    dot_pts: list[np.ndarray] = []
    offset = 0
    for gid in sorted(graph_trees):
        for samp in graph_trees[gid]["tree_ctx"]["diam_samples"]:
            coords = np.asarray(samp["coords"], dtype=np.float64)
            radii = np.asarray(samp["radii"], dtype=np.float64)
            n_p = len(coords)
            if n_p == 0:
                continue
            dot_pts.append(coords)
            prev_normal = None
            for i in range(n_p):
                if n_p == 1:
                    continue
                if i == 0:
                    tang = coords[1] - coords[0]
                elif i == n_p - 1:
                    tang = coords[-1] - coords[-2]
                else:
                    tang = coords[i + 1] - coords[i - 1]
                if np.linalg.norm(tang) < 1e-12 or radii[i] <= 0:
                    continue
                _t, n_hat, b_hat = compute_frenet_frame(tang, prev_normal)
                prev_normal = n_hat
                theta = np.linspace(0.0, 2.0 * np.pi, n_circle_pts, endpoint=False)
                ring = coords[i] + radii[i] * (
                    np.cos(theta)[:, None] * n_hat + np.sin(theta)[:, None] * b_hat)
                ring_pts.append(ring)
                ring_lines.append(n_circle_pts + 1)
                ring_lines.extend(range(offset, offset + n_circle_pts))
                ring_lines.append(offset)
                offset += n_circle_pts
    rings = (pv.PolyData(np.vstack(ring_pts), lines=np.asarray(ring_lines, dtype=np.int64))
             if ring_pts else None)
    dots = pv.PolyData(np.vstack(dot_pts)) if dot_pts else None
    return rings, dots


def visualize_flow_split(graph_trees, cfx_outlets, all_outlets, tree_inlets, msh_surface):
    """Render the cropped CFX mesh, modified centrelines, labelled CFX boundary
    regions + flow-split ratios, the diameter contour rings/points, and the
    cropped-away (DROPPED) leaves and reconstructed (STUB) orphan outlets."""
    try:
        import pyvista as pv  # noqa: PLC0415 — optional viz dependency
        from .viz import _show_plotter
        from .splines import compute_frenet_frame
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[VIS][WARN] pyvista unavailable, skipping flow-split viz: {exc}")
        return

    title = "Giessen Flow Split"
    pl = pv.Plotter(title=title)
    pl.set_background(VIZ_BG_COLOR)

    # 1. Cropped CFX mesh surface (decimated for display if very large).
    if msh_surface is not None and getattr(msh_surface, "n_points", 0) > 0:
        surf = msh_surface.copy()
        surf.points = surf.points * MSH_SCALE + np.asarray(MSH_OFFSET, dtype=np.float64)
        if VIZ_MESH_MAX_FACES and surf.n_cells > VIZ_MESH_MAX_FACES:
            red = 1.0 - VIZ_MESH_MAX_FACES / surf.n_cells
            print(f"[VIS] CFX mesh has {surf.n_cells:,} faces; decimating ~{red * 100:.0f}% "
                  f"for display.")
            try:
                surf = surf.triangulate().decimate_pro(red)
            except Exception as exc:
                print(f"[VIS][WARN] decimation failed ({exc}); rendering full mesh.")
        pl.add_mesh(surf, color="lightgray", opacity=0.25, smooth_shading=True,
                    label="CFX mesh (cropped)")

    # 2. Modified (smoothed) centrelines.
    cl_poly = _build_centerline_polydata(graph_trees, pv)
    if cl_poly is not None:
        pl.add_mesh(cl_poly, scalars="seg_idx", cmap="turbo", line_width=3,
                    show_scalar_bar=False, label="Modified centrelines")

    # 3. Diameter contour rings + centreline points used for diameters.
    rings, dots = _build_diameter_contours(graph_trees, pv, compute_frenet_frame)
    if rings is not None:
        pl.add_mesh(rings, color=VIZ_TEXT_COLOR, line_width=1, opacity=0.6,
                    label="Diameter contours")
    if dots is not None:
        pl.add_mesh(dots, color=VIZ_TEXT_COLOR, point_size=5,
                    render_points_as_spheres=True,
                    label="Diameter sample points")

    # 4. CFX outlet zones coloured by flow fraction (orphans drawn separately).
    label_pts: list[np.ndarray] = []
    labels: list[str] = []
    cut = [c for c in cfx_outlets if c["kind"] != "orphan"]
    orphans = [c for c in cfx_outlets if c["kind"] == "orphan"]
    if cut:
        pts = np.array([c["pos"] for c in cut])
        cloud = pv.PolyData(pts)
        cloud["flow_percent"] = np.array([c["fraction"] for c in cut]) * 100.0
        pl.add_mesh(cloud, scalars="flow_percent", cmap=_flow_cmap(),
                    clim=(float(SURFACE_FLOW_CLIM_PCT[0]),
                          float(SURFACE_FLOW_CLIM_PCT[1])),
                    log_scale=SURFACE_FLOW_LOG_SCALE,
                    n_colors=SURFACE_FLOW_N_COLORS,
                    point_size=16, render_points_as_spheres=True,
                    scalar_bar_args={"title": FLOW_BAR_TITLE,
                                     "color": VIZ_TEXT_COLOR, "fmt": "%.3g"},
                    label="CFX outlets (terminal/crop)")
        for c in cut:
            label_pts.append(c["pos"])
            labels.append(f"{c['ccl_name']} {c['fraction'] * 100:.1f}%")
    if orphans:
        opts = np.array([c["pos"] for c in orphans])
        pl.add_mesh(pv.PolyData(opts), color="purple", point_size=18,
                    render_points_as_spheres=True, label="Orphan stub outlets")
        for c in orphans:
            label_pts.append(c["pos"])
            labels.append(f"STUB {c['ccl_name']} {c['fraction'] * 100:.1f}% "
                          f"(d={c['diam_mm']:.2f})")

    # 5. Inlets.
    if tree_inlets:
        ipts = np.array([v["pos"] for v in tree_inlets.values()])
        pl.add_mesh(pv.PolyData(ipts), color="red", point_size=22,
                    render_points_as_spheres=True, label="Inlets")
        for v in tree_inlets.values():
            label_pts.append(np.asarray(v["pos"]))
            labels.append(f"{v.get('ccl_name', 'INLET')}")

    # 6. Cropped-away (DROPPED) leaves.
    dropped = [o for o in all_outlets if o.get("in_mesh") is False]
    if dropped:
        dpts = np.array([o["pos"] for o in dropped])
        pl.add_mesh(pv.PolyData(dpts), color="orange", point_size=12,
                    render_points_as_spheres=True,
                    label=f"Cropped-away leaves (n={len(dropped)})")

    if label_pts:
        try:
            pl.add_point_labels(np.array(label_pts), labels, font_size=11,
                                text_color=VIZ_TEXT_COLOR, shape=None,
                                always_visible=True,
                                show_points=False)
        except Exception as exc:
            print(f"[VIS][WARN] point labels failed: {exc}")

    pl.add_legend(bcolor=VIZ_BG_COLOR)
    pl.add_title(title, color=VIZ_TEXT_COLOR)
    _show_plotter(pl, title)


# ── Lumen-surface flow-fraction viz ───────────────────────────────────────────


def _surface_candidates(out_dir: Path) -> list[Path]:
    """Readable surface files in ``out_dir``, best candidate first.

    ``*_regions.vtk`` leads (it carries the per-face ``is_junction`` tag written
    by :mod:`region_vtk`), then other ``.vtk``, then the other mesh formats
    PyVista can read; newest first within each group. ``*_flow.vtk`` (this
    module's own output) is excluded.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []

    def _by_mtime(paths):
        return sorted(paths, key=lambda q: q.stat().st_mtime, reverse=True)

    regions = _by_mtime(out_dir.glob("*_regions.vtk"))
    plain = _by_mtime(q for q in out_dir.glob("*.vtk")
                      if not q.name.endswith(("_regions.vtk", "_flow.vtk")))
    other = _by_mtime(q for ext in ("*.vtp", "*.stl", "*.ply", "*.obj")
                      for q in out_dir.glob(ext))
    return [*regions, *plain, *other]


def _prompt_surface_vtk(out_dir: Path) -> Path | None:
    """Ask the user which surface to colour.

    Tries a Tk file dialog first (opening in ``out_dir``); if Tk is unavailable
    or cancelled, falls back to a numbered console list of the ``out_dir``
    candidates, where Enter takes the top (auto-discovered) one.
    """
    out_dir = Path(out_dir)
    try:
        import tkinter as tk  # noqa: PLC0415 -- optional, GUI-only path
        from tkinter import filedialog  # noqa: PLC0415

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askopenfilename(
            title="Select the lumen surface to colour by flow fraction",
            initialdir=str(out_dir if out_dir.is_dir() else Path.cwd()),
            filetypes=[("Surface meshes", "*.vtk *.vtp *.stl *.ply *.obj"),
                       ("All files", "*.*")])
        root.destroy()
        if picked:
            return Path(picked)
        print("[SURF] File dialog cancelled.")
    except Exception as exc:
        print(f"[SURF] No file dialog ({exc}); falling back to the console list.")

    cands = _surface_candidates(out_dir)
    if not cands:
        print(f"[SURF] No surface files found in {out_dir}.")
        return None
    print(f"\n[SURF] Surfaces in {out_dir}:")
    for i, c in enumerate(cands):
        tag = "  <- auto" if i == 0 else ""
        print(f"  [{i}] {c.name}{tag}")
    if not sys.stdin.isatty():
        print(f"[SURF] Not interactive; using {cands[0].name}.")
        return cands[0]
    try:
        raw = input(f"Select surface [0-{len(cands) - 1}] "
                    f"(Enter={cands[0].name}, or paste a path): ").strip()
    except (EOFError, KeyboardInterrupt):
        return cands[0]
    if not raw:
        return cands[0]
    if raw.isdigit() and int(raw) < len(cands):
        return cands[int(raw)]
    p = Path(raw.strip('"'))
    if p.exists():
        return p
    print(f"[SURF][WARN] '{raw}' is not a listed index or an existing path; "
          f"using {cands[0].name}.")
    return cands[0]


def _discover_surface_vtk(out_dir: Path) -> Path | None:
    """Resolve which lumen surface to colour.

    ``SURFACE_VTK = "ask"`` (or ``SURFACE_VTK_PROMPT``) opens the picker of
    :func:`_prompt_surface_vtk`; any other non-empty ``SURFACE_VTK`` is used as a
    literal path; otherwise the best candidate in ``out_dir`` is auto-selected
    (see :func:`_surface_candidates`).
    """
    if SURFACE_VTK.strip().lower() == "ask" or SURFACE_VTK_PROMPT:
        return _prompt_surface_vtk(out_dir)
    if SURFACE_VTK:
        p = Path(SURFACE_VTK)
        if not p.exists():
            print(f"[SURF][WARN] SURFACE_VTK does not exist: {p}")
            return None
        return p
    cands = _surface_candidates(out_dir)
    return cands[0] if cands else None


def _junction_node_radii(graph_entry) -> dict[int, float]:
    """``{node_id: radius_mm}`` for the nodes that are bifurcations *on the mesh*.

    Radius = the largest incident-segment end radius, i.e. the size of the merged
    tube where the branches meet; it sets how far along each branch the collar
    reaches (see :func:`_segment_junction_points`).

    A node qualifies when it has degree>=3 in the graph AND, with
    ``SURFACE_JUNCTION_REQUIRE_MESHED``, at least three of its incident segments
    survived the split (``seg_root >= 0``). The second test is what keeps a
    cropped mesh clean: where the full graph branches but the mesh does not, the
    vessel runs straight through, so it gets no collar.
    """
    nodes, points, segments = (graph_entry["nodes"], graph_entry["points"],
                               graph_entry["segments"])
    node_r = _node_end_radii(graph_entry)
    kept_by_node = _kept_segments_per_node(graph_entry)

    out: dict[int, float] = {}
    n_dropped = 0
    for nid, nd in nodes.items():
        if len(nd) < 4 or nd[3] < 3:
            continue
        if SURFACE_JUNCTION_REQUIRE_MESHED and kept_by_node.get(nid, 0) < 3:
            n_dropped += 1          # branches here were cropped away / not meshed
            continue
        r = node_r.get(int(nid))
        if r is not None:
            out[int(nid)] = r
    if n_dropped:
        print(f"  [SURF] {n_dropped} graph bifurcation(s) are not bifurcations on "
              f"this mesh (their side branches are absent/cropped) -- no collar "
              f"drawn there, so the vessel stays coloured through them.")
    return out


def _node_end_radii(graph_entry) -> dict[int, float]:
    """``{node_id: radius_mm}`` for every node: the largest incident end radius."""
    points, segments = graph_entry["points"], graph_entry["segments"]
    out: dict[int, float] = {}
    for seg in segments:
        pids = list(seg["point_ids"])
        if not pids:
            continue
        for nid, pid in ((seg["node1"], pids[0]), (seg["node2"], pids[-1])):
            if pid in points:
                r = _radius_mm(points, pid)
                if r > out.get(int(nid), 0.0):
                    out[int(nid)] = float(r)
    return out


def _kept_segments_per_node(graph_entry) -> dict[int, int]:
    """``{node_id: how many incident segments survived the split}``."""
    segments = graph_entry["segments"]
    seg_root = np.asarray(graph_entry.get("tree_ctx", {}).get("seg_root", []))
    out: dict[int, int] = {}
    for si, seg in enumerate(segments):
        if not list(seg["point_ids"]):
            continue
        if len(seg_root) and not (si < len(seg_root) and int(seg_root[si]) >= 0):
            continue
        for nid in (seg["node1"], seg["node2"]):
            out[int(nid)] = out.get(int(nid), 0) + 1
    return out


def _kept_flow_per_node(graph_entry) -> dict[int, float]:
    """``{node_id: flow of the strongest surviving segment meeting there}``.

    That is the vessel a cropped side branch opens off, so it is the flow its
    leftover ostium should take.
    """
    ctx = graph_entry.get("tree_ctx", {})
    flow = np.asarray(ctx.get("flow", []), dtype=np.float64)
    seg_root = np.asarray(ctx.get("seg_root", []))
    out: dict[int, float] = {}
    for si, seg in enumerate(graph_entry["segments"]):
        if si >= len(flow) or not list(seg["point_ids"]):
            continue
        if len(seg_root) and not (si < len(seg_root) and int(seg_root[si]) >= 0):
            continue
        for nid in (seg["node1"], seg["node2"]):
            if float(flow[si]) > out.get(int(nid), -1.0):
                out[int(nid)] = float(flow[si])
    return out


def _ostium_flow(seg, cl, node_r, kept_flow):
    """Flow to paint on a cropped branch's leftover stump, NaN beyond it.

    Only the first ``SURFACE_OSTIUM_FACTOR`` x take-off-node radius of arc length
    is filled, and only from an end where a surviving vessel still meets it -- so
    a genuinely missing branch keeps its honest "no flow assigned" grey past the
    mouth.
    """
    vals = np.full(len(cl), np.nan, dtype=np.float64)
    if len(cl) == 0 or not SURFACE_FILL_CROPPED_OSTIA:
        return vals
    step = np.linalg.norm(np.diff(cl, axis=0), axis=1) if len(cl) > 1 else np.empty(0)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = float(arc[-1])
    for end, nid in ((0, seg["node1"]), (1, seg["node2"])):
        fr = kept_flow.get(int(nid))
        r = node_r.get(int(nid))
        if fr is None or r is None:
            continue
        along = arc if end == 0 else (total - arc)
        vals[along <= SURFACE_OSTIUM_FACTOR * r] = fr
    return vals


def _segment_junction_points(seg, cl, node_radii):
    """Bool mask over one segment's centreline points: inside a bifurcation zone.

    The zone is measured as **arc length along this branch** from a degree>=3 end
    node, out to ``SURFACE_JUNCTION_FACTOR`` x that node's radius. Arc length --
    rather than a ball around the node -- keeps the zone on the branches that
    actually meet there: a ball also swallows any unrelated vessel that happens
    to pass near the junction, which shows up as grey bands mid-branch.
    """
    n = len(cl)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    step = np.linalg.norm(np.diff(cl, axis=0), axis=1) if n > 1 else np.empty(0)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = float(arc[-1])
    for end, nid in ((0, seg["node1"]), (1, seg["node2"])):
        r = node_radii.get(int(nid))
        if r is None:
            continue
        along = arc if end == 0 else (total - arc)
        mask |= along <= SURFACE_JUNCTION_FACTOR * r
    return mask


def _surface_junction_mask(surf, graph_mask):
    """Combine the surface's own bifurcation tag with ``graph_mask``, per
    ``SURFACE_JUNCTION_SOURCE`` (``graph_mask`` may be ``None``).

    The surface's own ``is_junction`` cell array (written by :mod:`region_vtk`)
    is only meaningful when the surface was generated from *this* centreline
    tree. When it was not -- e.g. a hand-edited or differently-pruned mesh -- its
    collars miss most of the graph's bifurcations, and the flow colour then steps
    mid-tube instead of inside a grey collar. ``"auto"`` detects that by checking
    how much of the graph-derived junction area the surface tag also flags, and
    falls back to the union with a warning.
    """
    tag = surf.cell_data.get("is_junction")
    surf_mask = None
    if tag is not None and len(tag) == surf.n_cells:
        surf_mask = np.asarray(tag).astype(bool)
    if graph_mask is not None and not graph_mask.any():
        graph_mask = None

    src = str(SURFACE_JUNCTION_SOURCE).strip().lower()
    n = surf.n_cells

    def _pct(m):
        return f"{int(m.sum()):,}/{n:,} ({100.0 * m.mean():.1f}%)"

    if surf_mask is None and graph_mask is None:
        print("  [SURF][WARN] no is_junction array and no degree>=3 nodes; "
              "nothing will be greyed out.")
        return np.zeros(n, dtype=bool)
    if src == "surface" and surf_mask is not None:
        print(f"  [SURF] bifurcations from the surface tag: {_pct(surf_mask)}")
        return surf_mask
    if src == "graph" and graph_mask is not None:
        print(f"  [SURF] bifurcations from the graph junction zones "
              f"({SURFACE_JUNCTION_FACTOR:.2f} x radius along each branch): "
              f"{_pct(graph_mask)}")
        return graph_mask
    if surf_mask is None:
        print(f"  [SURF] no is_junction array; bifurcations from the graph "
              f"junction zones: {_pct(graph_mask)}")
        return graph_mask
    if graph_mask is None:
        print(f"  [SURF] bifurcations from the surface tag: {_pct(surf_mask)}")
        return surf_mask

    both = surf_mask | graph_mask
    coverage = float((surf_mask & graph_mask).sum()) / max(int(graph_mask.sum()), 1)
    if src == "both":
        print(f"  [SURF] bifurcations = surface tag | graph zones: {_pct(both)}")
        return both
    # "auto"
    if coverage >= SURFACE_JUNCTION_COVERAGE_MIN:
        print(f"  [SURF] using the surface's own is_junction tag: {_pct(surf_mask)} "
              f"(covers {100.0 * coverage:.0f}% of the graph junction area)")
        return surf_mask
    print(f"  [SURF][WARN] the surface is_junction tag covers only "
          f"{100.0 * coverage:.0f}% of the graph junction area -- this surface was "
          f"almost certainly built from a different tree than the one the split ran "
          f"on (a different or edited mesh). Falling back to the union so the colour "
          f"does not step mid-tube; set SURFACE_JUNCTION_SOURCE to 'surface' to keep "
          f"the tag as-is.")
    print(f"  [SURF] surface tag {_pct(surf_mask)} | graph zones {_pct(graph_mask)} "
          f"| union {_pct(both)}")
    return both


def _surface_flow_scalars(surf, graph_trees):
    """Per-face ``(flow, is_junction, nn_dist)`` for the lumen surface.

    ``flow[i]`` is the Giessen flow fraction (root = 1.0) of the branch whose
    centreline is nearest face *i*'s centroid, or NaN when that branch carries no
    split flow -- pruned as absent from the .msh, or lying beyond a crop plane.
    Every segment of the graph is in the search tree (not just the ones that
    survived the split), so a face over a pruned branch is reported as
    *unassigned* instead of silently stealing a neighbouring branch's colour.
    ``is_junction`` comes from :func:`_surface_junction_mask`.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 -- heavy optional import

    cl_pts: list[np.ndarray] = []
    cl_flow: list[np.ndarray] = []
    cl_junc: list[np.ndarray] = []
    n_ost = [0]
    for gid in sorted(graph_trees):
        d = graph_trees[gid]
        ctx = d["tree_ctx"]
        flow = np.asarray(ctx["flow"], dtype=np.float64)
        seg_root = np.asarray(ctx["seg_root"])
        node_radii = _junction_node_radii(d)
        node_r_all = _node_end_radii(d)
        kept_flow = _kept_flow_per_node(d)
        for si, seg in enumerate(d["segments"]):
            cl = _seg_centerline_mm(seg, d["points"])
            if len(cl) == 0:
                continue
            reached = si < len(seg_root) and int(seg_root[si]) >= 0
            cl_pts.append(cl)
            if reached:
                cl_flow.append(np.full(len(cl), float(flow[si])))
            else:
                # Cropped/absent branch: paint only its ostium with the flow of
                # the vessel it opens off; the rest stays unassigned.
                vals = _ostium_flow(seg, cl, node_r_all, kept_flow)
                n_ost[0] += int(np.isfinite(vals).sum())
                cl_flow.append(vals)
            cl_junc.append(_segment_junction_points(seg, cl, node_radii))
    if not cl_pts:
        raise ValueError("no centreline points available for surface colouring")
    cl_xyz = np.vstack(cl_pts)
    cl_val = np.concatenate(cl_flow)
    cl_isj = np.concatenate(cl_junc)

    if n_ost[0]:
        print(f"  [SURF] {n_ost[0]:,} centreline point(s) on cropped side branches "
              f"filled as ostia with the surviving vessel's flow "
              f"(SURFACE_OSTIUM_FACTOR={SURFACE_OSTIUM_FACTOR:g}).")

    cents = surf.cell_centers().points
    dist, nn = cKDTree(cl_xyz).query(cents)
    flow = cl_val[nn]

    # A face is in a bifurcation zone when the centreline point it belongs to is
    # -- so the zone follows the branch instead of a ball around the node.
    is_junc = _surface_junction_mask(surf, cl_isj[nn] if cl_isj.any() else None)
    return flow, is_junc, dist


def _flow_cmap(n_colors=None):
    """The flow colour map: ``SURFACE_FLOW_CMAP`` sliced to ``SURFACE_FLOW_CMAP_RANGE``.

    Returns a discrete ``ListedColormap`` of ``n_colors`` bands (default
    ``SURFACE_FLOW_N_COLORS``), or the plain map name if matplotlib is not
    importable -- PyVista accepts either.
    """
    n = int(n_colors or SURFACE_FLOW_N_COLORS)
    try:
        import matplotlib as mpl  # noqa: PLC0415 -- optional, PyVista pulls it in
        from matplotlib.colors import ListedColormap  # noqa: PLC0415
        try:
            base = mpl.colormaps[SURFACE_FLOW_CMAP]
        except (AttributeError, KeyError):
            base = mpl.cm.get_cmap(SURFACE_FLOW_CMAP)
        lo, hi = SURFACE_FLOW_CMAP_RANGE
        return ListedColormap(base(np.linspace(float(lo), float(hi), n)),
                              name=f"{SURFACE_FLOW_CMAP}_flow")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[VIS][WARN] could not slice '{SURFACE_FLOW_CMAP}' ({exc}); "
              f"using it unsliced.")
        return SURFACE_FLOW_CMAP


FLOW_BAR_TITLE = "Flow fraction (% of inlet)"


def _flow_bar_args(x=0.30, y=0.10, width=0.32, height=0.035):
    """``scalar_bar_args`` for a bare colour strip -- no title, no tick labels.

    VTK lays a horizontal bar's labels out on the title's side and re-flows them
    on every render (and again when the bar is wrapped in a draggable widget), so
    they cannot be reliably parked below the strip. The strip is drawn bare here
    and :func:`_add_flow_bar_ticks` writes the ticks and the title underneath it
    at positions we control.
    """
    return {"title": " ", "n_labels": 0, "color": VIZ_TEXT_COLOR,
            "vertical": False, "interactive": False,
            "position_x": x, "position_y": y, "width": width, "height": height}


def _add_flow_bar_ticks(pl, x=0.30, y=0.10, width=0.32, n_ticks=9,
                        title=FLOW_BAR_TITLE):
    """Tick numbers directly under the colour strip, then the title under those.

    Ticks are spaced across ``SURFACE_FLOW_CLIM_PCT`` -- geometrically when
    ``SURFACE_FLOW_LOG_SCALE``, else linearly -- matching how the strip maps
    values to colour. Positions are viewport fractions, so they hold on resize.
    """
    lo, hi = float(SURFACE_FLOW_CLIM_PCT[0]), float(SURFACE_FLOW_CLIM_PCT[1])
    if SURFACE_FLOW_LOG_SCALE and lo > 0:
        vals = np.geomspace(lo, hi, n_ticks)
    else:
        vals = np.linspace(lo, hi, n_ticks)
    labels = [f"{v:.3g}" for v in vals]

    # Label geometry in viewport fractions. add_text places text by its LEFT edge,
    # so each label is shifted half its width to sit centred on its tick.
    font_size = 9
    try:
        win_w = float(pl.window_size[0])
    except Exception:
        win_w = float(COMBINED_WINDOW_SIZE[0])
    # ~1.05 px per point per glyph, measured off rendered VTK text at font_size 9.
    char_w = 1.05 * font_size / max(win_w, 1.0)
    gap = 0.45 * char_w                           # min clear space between labels

    spans = []
    for i, txt in enumerate(labels):
        w = char_w * len(txt)
        centre = x + width * (i / max(n_ticks - 1, 1))
        left = centre - w / 2.0
        # keep the end labels inside the strip rather than hanging off it
        left = min(max(left, x - w / 4.0), x + width - w * 0.75)
        spans.append((left, left + w))

    # Drop labels that would collide. The two ends always stay; middles are kept
    # greedily left to right, and only if they also clear the final label.
    keep = [False] * n_ticks
    keep[0] = keep[-1] = True
    right_edge = spans[0][1]
    last_left = spans[-1][0]
    n_dropped = 0
    for i in range(1, n_ticks - 1):
        left, right = spans[i]
        if left >= right_edge + gap and right + gap <= last_left:
            keep[i] = True
            right_edge = right
        else:
            n_dropped += 1
    if n_dropped:
        print(f"  [VIS] scalar bar: {n_dropped} of {n_ticks} tick labels dropped "
              f"to avoid overlap (widen the window or lower n_ticks for more).")

    for i, txt in enumerate(labels):
        if not keep[i]:
            continue
        try:
            pl.add_text(txt, position=(spans[i][0], y - 0.035), viewport=True,
                        font_size=font_size, color=VIZ_TEXT_COLOR,
                        name=f"__flowtick{i}__")
        except Exception as exc:  # pragma: no cover - PyVista version dependent
            print(f"[VIS][WARN] scalar-bar tick label failed: {exc}")
            return
    try:
        pl.add_text(title, position=(x + width / 2 - 0.075, y - 0.075),
                    viewport=True, font_size=11, color=VIZ_TEXT_COLOR,
                    name="__flowbartitle__")
    except Exception as exc:  # pragma: no cover - PyVista version dependent
        print(f"[VIS][WARN] scalar-bar title failed: {exc}")


def _surface_from_msh(msh_surface, pv):
    """The .msh wall zones as a triangulated surface in graph (mm) coordinates.

    Inlet/outlet cap faces (``kind_id`` 1/2) are dropped -- they are flat lids,
    not vessel wall -- and ``MSH_SCALE``/``MSH_OFFSET`` are applied so the result
    lands in the same frame as the centrelines. Returns ``None`` if there is
    nothing usable.
    """
    if msh_surface is None or getattr(msh_surface, "n_cells", 0) == 0:
        return None
    surf = msh_surface.copy()
    surf.points = surf.points * MSH_SCALE + np.asarray(MSH_OFFSET, dtype=np.float64)
    kind = surf.cell_data.get("kind_id")
    if kind is not None and len(kind) == surf.n_cells:
        wall = np.where(np.asarray(kind) == 0)[0]
        if len(wall) and len(wall) < surf.n_cells:
            print(f"  [SURF] .msh: keeping {len(wall):,} wall faces, dropping "
                  f"{surf.n_cells - len(wall):,} inlet/outlet cap faces.")
            surf = surf.extract_cells(wall).extract_surface()
    surf = surf.triangulate()
    cap = SURFACE_FROM_MSH_MAX_FACES
    if cap and surf.n_cells > cap:
        red = 1.0 - cap / surf.n_cells
        print(f"  [SURF] .msh wall surface has {surf.n_cells:,} faces; decimating "
              f"~{red * 100:.0f}% to keep the colouring responsive "
              f"(raise SURFACE_FROM_MSH_MAX_FACES to keep them all).")
        try:
            surf = surf.decimate_pro(red)
        except Exception as exc:
            print(f"  [SURF][WARN] decimation failed ({exc}); colouring all faces.")
    return surf


def _prepare_surface_flow(graph_trees, surface_path, pv, out_dir=None,
                          surface=None, stem=None):
    """Read the lumen surface and tag it with the per-branch flow fraction.

    Returns ``(branch_mesh, bif_mesh, unassigned_mesh, clim, surf)`` -- the
    coloured branch faces, the bifurcation faces, the faces over branches the
    split never reached, the ``flow_percent`` colour limits and the full tagged
    surface -- or ``None`` if the surface is unusable. Also writes
    ``<surface>_flow.vtk`` when ``out_dir`` is given and ``SURFACE_FLOW_WRITE_VTK``.
    """
    if surface is not None:
        surf = surface
        stem = stem or "msh_wall_surface"
        print(f"\n[SURF] Colouring the .msh wall surface by branch flow fraction "
              f"({surf.n_cells:,} faces).")
    else:
        surface_path = Path(surface_path)
        if not surface_path.exists():
            print(f"[VIS][WARN] lumen surface VTK not found: {surface_path}")
            return None
        print(f"\n[SURF] Colouring lumen surface by branch flow fraction:\n"
              f"       {surface_path}")
        surf = pv.read(str(surface_path)).extract_surface().triangulate()
        stem = surface_path.stem
    if surf.n_cells == 0:
        print("[VIS][WARN] surface has no faces; skipping.")
        return None

    try:
        flow, is_junc, dist = _surface_flow_scalars(surf, graph_trees)
    except ValueError as exc:
        print(f"[VIS][WARN] {exc}; skipping surface flow viz.")
        return None

    med = float(np.median(dist))
    print(f"  [SURF] {surf.n_cells:,} faces; face->centreline distance "
          f"median {med:.3f} mm, max {float(dist.max()):.3f} mm")
    if med > SURFACE_ALIGN_WARN_MM:
        print(f"  [SURF][WARN] median distance > {SURFACE_ALIGN_WARN_MM} mm -- the "
              f"surface and the spatial graph may not be in the same frame; the "
              f"branch colours are probably wrong.")

    # Three disjoint face sets: bifurcation collars, branches with a flow, and
    # branches the split never reached (pruned/beyond crop -> no flow to show).
    unassigned = (~is_junc) & ~np.isfinite(flow)
    coloured = (~is_junc) & np.isfinite(flow)
    if unassigned.any():
        print(f"  [SURF] {int(unassigned.sum()):,}/{surf.n_cells:,} faces "
              f"({100.0 * unassigned.mean():.1f}%) sit over branches carrying no "
              f"split flow (pruned as absent from the .msh, or beyond a crop) -- "
              f"drawn flat {SURFACE_UNASSIGNED_COLOR}, not coloured.")

    branch_flow = np.where(coloured, flow, np.nan)
    surf.cell_data["flow_fraction"] = branch_flow
    surf.cell_data["flow_percent"] = branch_flow * 100.0
    surf.cell_data["is_junction"] = is_junc.astype(np.int32)
    surf.cell_data["unassigned"] = unassigned.astype(np.int32)

    # Fixed, case-independent colour range (see SURFACE_FLOW_CLIM_PCT).
    clim = (float(SURFACE_FLOW_CLIM_PCT[0]), float(SURFACE_FLOW_CLIM_PCT[1]))
    pct = flow[coloured] * 100.0
    if pct.size:
        under = int((pct < clim[0]).sum())
        over = int((pct > clim[1]).sum())
        print(f"  [SURF] colour range fixed at {clim[0]:g}-{clim[1]:g}% of inlet "
              f"flow in {SURFACE_FLOW_N_COLORS} bands; branch faces span "
              f"{pct.min():.3g}-{pct.max():.3g}%")
        if under or over:
            print(f"  [SURF] {under:,} face(s) below and {over:,} above the range "
                  f"are clamped to the end bands.")

    branch = surf.extract_cells(np.where(coloured)[0]) if coloured.any() else None
    bifs = surf.extract_cells(np.where(is_junc)[0]) if is_junc.any() else None
    unass = surf.extract_cells(np.where(unassigned)[0]) if unassigned.any() else None

    if out_dir is not None and SURFACE_FLOW_WRITE_VTK:
        dst = Path(out_dir) / f"{stem}_flow.vtk"
        try:
            surf.save(str(dst))
            print(f"  [SURF] Saved flow-coloured surface to {dst}")
        except Exception as exc:
            print(f"  [SURF][WARN] could not save {dst}: {exc}")

    return branch, bifs, unass, clim, surf


def visualize_surface_flow(graph_trees, surface_path, out_dir=None, show=True):
    """Render the lumen surface with each non-bifurcation branch coloured by its
    Giessen flow fraction; bifurcation regions are drawn translucent grey.

    Faces are assigned to a branch by nearest smoothed centreline point, so the
    surface and the spatial graph must share the same (mm) frame. Also writes
    ``<surface>_flow.vtk`` carrying the per-face ``flow_fraction`` (NaN in the
    bifurcation regions), ``flow_percent`` and ``is_junction`` arrays for ParaView.
    See :func:`visualize_combined` for the everything-in-one-window version.
    """
    try:
        import pyvista as pv  # noqa: PLC0415 -- optional viz dependency
        from .viz import _show_plotter
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[VIS][WARN] pyvista unavailable, skipping surface flow viz: {exc}")
        return None

    prep = _prepare_surface_flow(graph_trees, surface_path, pv, out_dir=out_dir)
    if prep is None:
        return None
    branch, bifs, unass, clim, surf = prep

    title = "Lumen Surface - Branch Flow Fractions"
    pl = pv.Plotter(title=title)
    pl.set_background(VIZ_BG_COLOR)

    if unass is not None:
        pl.add_mesh(unass, color=SURFACE_UNASSIGNED_COLOR, opacity=0.6,
                    smooth_shading=True, show_scalar_bar=False,
                    label="No flow assigned")
    if bifs is not None:
        pl.add_mesh(bifs, color=SURFACE_BIF_COLOR, opacity=SURFACE_BIF_OPACITY,
                    smooth_shading=True, show_scalar_bar=False,
                    label="Bifurcation regions")
    if branch is not None:
        pl.add_mesh(branch, scalars="flow_percent", cmap=_flow_cmap(),
                    clim=clim, log_scale=SURFACE_FLOW_LOG_SCALE,
                    n_colors=SURFACE_FLOW_N_COLORS,
                    smooth_shading=True, label="Branch flow fraction",
                    scalar_bar_args=_flow_bar_args())
        _add_flow_bar_ticks(pl)

    if SURFACE_FLOW_SHOW_CENTERLINES:
        cl_poly = _build_centerline_polydata(graph_trees, pv)
        if cl_poly is not None:
            pl.add_mesh(cl_poly, color=VIZ_TEXT_COLOR, line_width=2, opacity=0.5,
                        show_scalar_bar=False, label="Centrelines")

    try:
        pl.add_legend(bcolor=VIZ_BG_COLOR)
    except Exception as exc:
        print(f"  [SURF][WARN] legend failed: {exc}")
    pl.add_title(title, color=VIZ_TEXT_COLOR)

    if show:
        _show_plotter(pl, title)
    else:
        pl.close()
    return surf


# ── Combined viewer with per-layer toggles + opacity sliders ──────────────────


def _set_actor_opacity(actor, value: float) -> bool:
    """Set one actor's opacity; ``False`` if the actor has no usable opacity.

    ``vtkActor2D`` label actors (``add_point_labels``) report a property that
    accepts ``SetOpacity`` but ignore it -- the text is drawn by a
    ``vtkLabelPlacementMapper``, which has no opacity at all -- so they are
    rejected here and faded by visibility instead (see :func:`_add_layer_controls`).
    """
    try:
        actor.prop.opacity = float(value)
        return True
    except Exception:
        pass
    return False


def _set_actor_visible(actor, flag: bool) -> None:
    try:
        actor.SetVisibility(bool(flag))
    except Exception:
        try:
            actor.visibility = bool(flag)
        except Exception:
            pass


def _add_layer_controls(pl, layers, pv):
    """Checkbox (show/hide) + opacity slider for every non-empty layer.

    Checkboxes stack up the left edge with a coloured swatch and a number key
    shortcut (1-9); the matching opacity sliders stack up the right edge. Layers
    are ``{"name", "actors", "opacity", "color"}`` dicts; empty ones are skipped.
    """
    live = [ly for ly in layers if ly["actors"]]
    if not live:
        return
    n = len(live)

    def make_toggle(layer):
        def _toggle(flag):
            layer["visible"] = bool(flag)
            for a in layer["actors"]:
                _set_actor_visible(a, flag and layer["opacity"] >= 0.05)
            pl.render()
        return _toggle

    def make_opacity(layer):
        def _opacity(value):
            value = float(value)
            layer["opacity"] = value
            for a in layer["actors"]:
                if not _set_actor_opacity(a, value):
                    # No real opacity (text labels): fade == show/hide.
                    _set_actor_visible(a, layer["visible"] and value >= 0.05)
            pl.render()
        return _opacity

    # Checkboxes + labels, bottom-up on the left.
    y0, dy, size = 12, 34, 22
    for k, layer in enumerate(live):
        cb = make_toggle(layer)
        y = y0 + k * dy
        try:
            pl.add_checkbox_button_widget(
                cb, value=True, position=(10, y), size=size, border_size=2,
                color_on=layer.get("color", VIZ_TEXT_COLOR),
                color_off=VIZ_BG_COLOR, background_color="grey")
        except Exception as exc:
            print(f"[VIS][WARN] checkbox for '{layer['name']}' failed: {exc}")
        try:
            pl.add_text(f"{k + 1}. {layer['name']}", position=(10 + size + 8, y + 3),
                        font_size=8, color=VIZ_TEXT_COLOR)
        except Exception:
            pass
        if k < 9:  # number-key shortcut mirroring the checkbox
            def _key(layer=layer):
                layer["visible"] = not layer["visible"]
                for a in layer["actors"]:
                    _set_actor_visible(
                        a, layer["visible"] and layer["opacity"] >= 0.05)
                pl.render()
            pl.add_key_event(str(k + 1), _key)

    # Opacity sliders, bottom-up on the right (normalised viewport coords).
    height = min(0.10, 0.86 / max(n, 1))
    for k, layer in enumerate(live):
        yn = 0.09 + k * height
        kw = dict(value=layer["opacity"], title=layer["name"],
                  pointa=(0.70, yn), pointb=(0.97, yn), style="modern",
                  title_height=0.014, tube_width=0.0025, slider_width=0.018,
                  fmt="%.2f", color=VIZ_TEXT_COLOR)
        try:
            # interaction_event="always" -> the layer fades while dragging, not
            # only when the handle is released (PyVista defaults to "end").
            pl.add_slider_widget(make_opacity(layer), [0.0, 1.0],
                                 interaction_event="always", **kw)
        except TypeError:
            pl.add_slider_widget(make_opacity(layer), [0.0, 1.0], **kw)
        except Exception as exc:
            print(f"[VIS][WARN] opacity slider for '{layer['name']}' failed: {exc}")


def _warn_surface_msh_mismatch(surface_path, msh_surface, pv, tol_mm=1.0):
    """Warn when the lumen VTK and the CFX .msh are not the same geometry.

    They routinely are not -- the .msh is often a hand-edited / re-cropped copy --
    and that mismatch is what makes the flow colouring look patchy: the split runs
    on the branches the *.msh* has, while the colours are painted on the *VTK*
    surface, so any branch only one of them contains has no honest colour.
    Compares bounding boxes; a mismatch is reported, never corrected.
    """
    if surface_path is None or msh_surface is None:
        return
    if getattr(msh_surface, "n_points", 0) == 0:
        return
    try:
        surf = pv.read(str(surface_path))
        a = np.asarray(surf.bounds, dtype=np.float64)
        mp = np.asarray(msh_surface.points, dtype=np.float64) * MSH_SCALE + \
            np.asarray(MSH_OFFSET, dtype=np.float64)
        b = np.array([mp[:, 0].min(), mp[:, 0].max(), mp[:, 1].min(),
                      mp[:, 1].max(), mp[:, 2].min(), mp[:, 2].max()])
    except Exception as exc:
        print(f"[VIS][WARN] could not compare the surface and .msh extents: {exc}")
        return
    delta = np.abs(a - b)
    if delta.max() <= tol_mm:
        return
    print(f"[SURF][WARN] the lumen VTK and the CFX .msh are not the same geometry "
          f"(bounding boxes differ by up to {delta.max():.2f} mm).")
    print(f"             VTK  bbox x[{a[0]:.1f},{a[1]:.1f}] y[{a[2]:.1f},{a[3]:.1f}] "
          f"z[{a[4]:.1f},{a[5]:.1f}]")
    print(f"             .msh bbox x[{b[0]:.1f},{b[1]:.1f}] y[{b[2]:.1f},{b[3]:.1f}] "
          f"z[{b[4]:.1f},{b[5]:.1f}]")
    print(f"             The split follows the .msh; the colours are painted on the "
          f"VTK. Branches present in only one of them show as 'No flow assigned' "
          f"or take a neighbour's colour. Colour a VTK built from the same tree as "
          f"the .msh to remove the patches.")


def visualize_combined(graph_trees, cfx_outlets, all_outlets, tree_inlets,
                       msh_surface, surface_path=None, out_dir=None, show=True):
    """One window with every layer of the flow-split picture, each layer
    toggleable and with its own opacity slider.

    Layers: the lumen surface coloured by branch flow fraction, its translucent
    grey bifurcation regions, the cropped CFX mesh, the smoothed centrelines, the
    diameter contour rings/points, the labelled CFX outlets (name + flow %),
    the inlets, and the cropped-away leaves. Checkboxes down the left toggle a
    layer (number keys 1-9 do the same); sliders up the right set its opacity.
    ``surface_path=None`` simply drops the two surface layers.
    """
    try:
        import pyvista as pv  # noqa: PLC0415 -- optional viz dependency
        from .viz import _show_plotter
        from .splines import compute_frenet_frame
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[VIS][WARN] pyvista unavailable, skipping combined viz: {exc}")
        return None

    title = "Coronary Flow Split - Giessen-based Flow Fraction Percentages"
    pl = pv.Plotter(title=title, window_size=COMBINED_WINDOW_SIZE)
    pl.set_background(VIZ_BG_COLOR)
    layers: list[dict[str, Any]] = []

    def layer(name, actors, opacity, color):
        acts = [a for a in actors if a is not None]
        if acts:
            for a in acts:
                _set_actor_opacity(a, opacity)
            layers.append({"name": name, "actors": acts, "opacity": float(opacity),
                           "color": color, "visible": True})

    # 1. Lumen surface: branch flow fractions + translucent grey bifurcations.
    # With no .vtk, colour the .msh wall zones instead (same lumen wall, denser).
    from_msh = surface_path is None and SURFACE_FALLBACK_TO_MSH
    msh_surf = _surface_from_msh(msh_surface, pv) if from_msh else None
    if from_msh and msh_surf is None:
        print("[SURF] No lumen .vtk and no usable .msh wall surface -- the flow "
              "colouring layer will be missing.")
    if surface_path is not None or msh_surf is not None:
        prep = _prepare_surface_flow(graph_trees, surface_path, pv, out_dir=out_dir,
                                     surface=msh_surf)
        if prep is not None:
            branch, bifs, unass, clim, _surf = prep
            src_tag = " (from .msh)" if msh_surf is not None else ""
            if branch is not None:
                layer("Branch flow fraction" + src_tag, [pl.add_mesh(
                    branch, scalars="flow_percent", cmap=_flow_cmap(),
                    clim=clim, log_scale=SURFACE_FLOW_LOG_SCALE,
                    n_colors=SURFACE_FLOW_N_COLORS, smooth_shading=True,
                    scalar_bar_args=_flow_bar_args())],
                    LAYER_OPACITY["branch"], "red")
            if bifs is not None:
                layer("Bifurcation regions", [pl.add_mesh(
                    bifs, color=SURFACE_BIF_COLOR, smooth_shading=True,
                    show_scalar_bar=False)],
                    LAYER_OPACITY["bifurcations"], SURFACE_BIF_COLOR)
            if unass is not None:
                layer("No flow assigned", [pl.add_mesh(
                    unass, color=SURFACE_UNASSIGNED_COLOR, smooth_shading=True,
                    show_scalar_bar=False)],
                    LAYER_OPACITY["unassigned"], SURFACE_UNASSIGNED_COLOR)

    # 2. Cropped CFX mesh surface (decimated for display if very large). Skipped
    # when that same mesh is already on screen as the flow-coloured surface.
    if (msh_surf is None and msh_surface is not None
            and getattr(msh_surface, "n_points", 0) > 0):
        surf = msh_surface.copy()
        surf.points = surf.points * MSH_SCALE + np.asarray(MSH_OFFSET, dtype=np.float64)
        if VIZ_MESH_MAX_FACES and surf.n_cells > VIZ_MESH_MAX_FACES:
            red = 1.0 - VIZ_MESH_MAX_FACES / surf.n_cells
            print(f"[VIS] CFX mesh has {surf.n_cells:,} faces; decimating ~{red * 100:.0f}% "
                  f"for display.")
            try:
                surf = surf.triangulate().decimate_pro(red)
            except Exception as exc:
                print(f"[VIS][WARN] decimation failed ({exc}); rendering full mesh.")
        layer("CFX mesh (cropped)", [pl.add_mesh(
            surf, color="lightblue", smooth_shading=True, show_scalar_bar=False)],
            LAYER_OPACITY["cfx_mesh"], "lightblue")

    _add_flow_bar_ticks(pl)
    _warn_surface_msh_mismatch(surface_path, msh_surface, pv)

    # 3. Modified (smoothed) centrelines.
    cl_poly = _build_centerline_polydata(graph_trees, pv)
    if cl_poly is not None:
        layer("Centrelines", [pl.add_mesh(
            cl_poly, scalars="seg_idx", cmap="turbo", line_width=3,
            show_scalar_bar=False)], LAYER_OPACITY["centrelines"], VIZ_TEXT_COLOR)

    # 4. Diameter contour rings + the centreline points they were measured on.
    rings, dots = _build_diameter_contours(graph_trees, pv, compute_frenet_frame)
    ring_actors = []
    if rings is not None:
        ring_actors.append(pl.add_mesh(rings, color=VIZ_TEXT_COLOR, line_width=1))
    if dots is not None:
        ring_actors.append(pl.add_mesh(dots, color=VIZ_TEXT_COLOR, point_size=5,
                                       render_points_as_spheres=True))
    layer("Diameter contours", ring_actors, LAYER_OPACITY["contours"],
          VIZ_TEXT_COLOR)

    # 5. CFX outlet zones coloured by flow fraction (orphan stubs drawn apart).
    label_pts: list[np.ndarray] = []
    labels: list[str] = []
    cut = [c for c in cfx_outlets if c["kind"] != "orphan"]
    orphans = [c for c in cfx_outlets if c["kind"] == "orphan"]
    outlet_actors = []
    if cut:
        cloud = pv.PolyData(np.array([c["pos"] for c in cut]))
        cloud["flow_percent"] = np.array([c["fraction"] for c in cut]) * 100.0
        # Same cmap/range/bands as the surface, so an outlet dot and the branch
        # feeding it read as the same colour.
        outlet_actors.append(pl.add_mesh(
            cloud, scalars="flow_percent", cmap=_flow_cmap(),
            clim=(float(SURFACE_FLOW_CLIM_PCT[0]), float(SURFACE_FLOW_CLIM_PCT[1])),
            log_scale=SURFACE_FLOW_LOG_SCALE, n_colors=SURFACE_FLOW_N_COLORS,
            point_size=16, render_points_as_spheres=True, show_scalar_bar=False))
        for c in cut:
            label_pts.append(c["pos"])
            labels.append(f"{c['ccl_name']} {c['fraction'] * 100:.1f}%")
    if orphans:
        outlet_actors.append(pl.add_mesh(
            pv.PolyData(np.array([c["pos"] for c in orphans])), color="purple",
            point_size=18, render_points_as_spheres=True))
        for c in orphans:
            label_pts.append(c["pos"])
            labels.append(f"STUB {c['ccl_name']} {c['fraction'] * 100:.1f}% "
                          f"(d={c['diam_mm']:.2f})")
    layer("CFX outlets", outlet_actors, LAYER_OPACITY["outlets"], "green")

    # 6. Inlets.
    inlet_actors = []
    if tree_inlets:
        inlet_actors.append(pl.add_mesh(
            pv.PolyData(np.array([v["pos"] for v in tree_inlets.values()])),
            color="red", point_size=22, render_points_as_spheres=True))
        for v in tree_inlets.values():
            label_pts.append(np.asarray(v["pos"]))
            labels.append(f"{v.get('ccl_name', 'INLET')}")
    layer("Inlets", inlet_actors, LAYER_OPACITY["inlets"], "red")

    # 7. Cropped-away (DROPPED) leaves.
    dropped = [o for o in all_outlets if o.get("in_mesh") is False]
    if dropped:
        layer(f"Cropped-away leaves (n={len(dropped)})", [pl.add_mesh(
            pv.PolyData(np.array([o["pos"] for o in dropped])), color="orange",
            point_size=12, render_points_as_spheres=True)],
            LAYER_OPACITY["dropped"], "orange")

    # 8. Outlet / inlet text labels (their own layer so they can be hidden).
    if label_pts:
        try:
            layer("Flow-fraction labels", [pl.add_point_labels(
                np.array(label_pts), labels, font_size=11,
                text_color=VIZ_TEXT_COLOR,
                shape=None, always_visible=True, show_points=False)],
                LAYER_OPACITY["labels"], VIZ_TEXT_COLOR)
        except Exception as exc:
            print(f"[VIS][WARN] point labels failed: {exc}")

    _add_layer_controls(pl, layers, pv)
    pl.add_title(title, font_size=12, color=VIZ_TEXT_COLOR)
    print(f"[VIS] Combined view: {len(layers)} layer(s). Checkboxes (or keys 1-9) "
          f"toggle a layer; the sliders on the right set its opacity.")

    if show:
        _show_plotter(pl, title)
    else:
        pl.close()
    return pl


# ── NetworkX flow-tree diagram ────────────────────────────────────────────────


def build_flow_tree_graph(graph_trees, cfx_outlets=None):
    """Build a ``networkx.DiGraph`` of the directed Giessen tree.

    One node per segment (key ``(gid, seg_idx)``); a ``parent -> child`` edge for
    every junction (so an N-furcation is one node with N children). Node attrs:
    ``flow`` (cumulative fraction, root=1.0), ``local_split`` (flow/parent_flow,
    root=1.0), ``diameter`` (the average diameter used in the split), ``region_id``,
    ``is_root``, ``is_leaf``, plus leaf status (``in_mesh`` / ``served_ccl`` /
    ``kind``). Edge attr ``diameter`` = the child's average diameter.

    Node ``flow`` is sourced from the **final** ``cfx_outlets`` fractions (the same
    values the 3D viz and CCL use): each outlet's fraction is added to its owner
    segment and every ancestor, so a leaf's cum% equals its 3D/CSV outlet % exactly
    even after orphan injection / per-tree renormalisation. Falls back to the raw
    ``tree_ctx["flow"]`` only when no ``cfx_outlets`` are supplied.
    """
    import networkx as nx  # noqa: PLC0415 — already a project dependency

    # leaf seg -> (in_mesh, served zone idx)  from the per-leaf outlets.
    leaf_status: dict[tuple, dict[str, Any]] = {}
    for gid, gt in graph_trees.items():
        for o in gt["outlets"]:
            leaf_status[(int(gid), int(o["seg_idx"]))] = o
    zone_kind = {c["zone_idx"]: c["kind"] for c in (cfx_outlets or [])}
    zone_name = {c["zone_idx"]: c["ccl_name"] for c in (cfx_outlets or [])}

    # Cumulative final flow per node = sum of the cfx outlet fractions in its
    # subtree (outlet fraction added to its owner seg + every ancestor).
    node_flow: dict[tuple, float] = {}
    for c in (cfx_outlets or []):
        gid = int(c["tree_id"][0])
        if gid not in graph_trees:
            continue
        pidx = graph_trees[gid]["tree_ctx"]["parent_idx"]
        f = float(c["fraction"])
        cur = int(c["owner_seg_idx"])
        while cur >= 0:
            node_flow[(gid, cur)] = node_flow.get((gid, cur), 0.0) + f
            cur = int(pidx[cur])
    use_cfx_flow = bool(cfx_outlets)

    g = nx.DiGraph()
    for gid, gt in graph_trees.items():
        tc = gt["tree_ctx"]
        parent_idx = tc["parent_idx"]
        seg_root = tc["seg_root"]
        flow = tc["flow"]
        seg_diam = tc["seg_diam"]
        seg_id = tc["seg_id"]
        leaf_set = {int(l) for l in tc["leaves"]}
        n = len(seg_id)
        for s in range(n):
            if int(seg_root[s]) < 0:
                continue  # beyond a crop / unreachable — truncated away
            key = (int(gid), s)
            p = int(parent_idx[s])
            if use_cfx_flow:
                f_s = node_flow.get((int(gid), s), 0.0)
                f_p = node_flow.get((int(gid), p), 0.0) if p >= 0 else 1.0
            else:
                f_s = float(flow[s])
                f_p = float(flow[p]) if p >= 0 else 1.0
            local = (f_s / f_p) if f_p > 0 else 0.0
            attrs = {
                "flow": f_s,
                "local_split": local,
                "diameter": float(seg_diam[s]),
                "region_id": int(seg_id[s]),
                "is_root": p < 0,
                "is_leaf": s in leaf_set,
                "in_mesh": None,
                "served_ccl": None,
                "kind": None,
            }
            if s in leaf_set:
                o = leaf_status.get(key)
                if o is not None:
                    attrs["in_mesh"] = o.get("in_mesh")
                    zi = o.get("served_zone_idx")
                    if zi is not None:
                        attrs["served_ccl"] = zone_name.get(zi)
                        attrs["kind"] = zone_kind.get(zi, "terminal")
                    elif o.get("in_mesh") is False:
                        attrs["kind"] = "dropped"
            g.add_node(key, **attrs)
        for s in range(n):
            if int(seg_root[s]) < 0:
                continue
            p = int(parent_idx[s])
            if p >= 0:
                g.add_edge((int(gid), p), (int(gid), s), diameter=float(seg_diam[s]))
    return g


def _flow_hex(value, vmax):
    """viridis hex for a cumulative flow fraction, gamma-stretched so deep,
    tiny-fraction branches stay distinguishable."""
    try:
        import matplotlib as mpl  # noqa: PLC0415
        try:
            cmap = mpl.colormaps["viridis"]          # matplotlib >= 3.6
        except Exception:
            from matplotlib import cm  # noqa: PLC0415
            cmap = cm.get_cmap("viridis")
        norm = mpl.colors.PowerNorm(gamma=0.4, vmin=0.0, vmax=max(vmax, 1e-9))
        return mpl.colors.to_hex(cmap(norm(value)))
    except Exception:
        return "#cccccc"


def visualize_flow_tree(graph, out_dir, show=False):
    """Render the flow tree: edge labels = average diameters (mm), node labels =
    ``cum% (local%)``. Writes ``giessen_flow_tree.svg`` + ``.png`` to ``out_dir``;
    falls back to a matplotlib drawing if Graphviz/pydot is unavailable."""
    out_dir = Path(out_dir)
    if graph.number_of_nodes() == 0:
        print("[TREE] empty graph — skipping flow-tree diagram.")
        return None

    leaf_outline = {"terminal": "#1a9850", "crop": "#3b6fb6",
                    "orphan": "#7b3fa0", "dropped": "#d73027"}
    vmax = max((d["flow"] for _, d in graph.nodes(data=True)), default=1.0)

    # Primary path: Graphviz via pydot (clean hierarchical layout + edge labels).
    try:
        import networkx as nx  # noqa: PLC0415
        pd = nx.nx_pydot.to_pydot(graph)
        pd.set_rankdir("TB")
        pd.set_splines("true")
        pd.set_nodesep("0.25")
        pd.set_ranksep("0.6")
        for node in pd.get_nodes():
            name = node.get_name().strip('"')
            if name not in (str(n) for n in graph.nodes):
                continue
        # Map pydot nodes back to graph nodes by their stringified key.
        key_by_str = {str(n): n for n in graph.nodes}
        for node in pd.get_nodes():
            nm = node.get_name().strip('"')
            key = key_by_str.get(nm)
            if key is None:
                continue
            d = graph.nodes[key]
            cum = d["flow"] * 100.0
            loc = d["local_split"] * 100.0
            label = f"{cum:.2f}% ({loc:.0f}%)\\n#{d['region_id']}"
            if d["is_leaf"]:
                if d.get("served_ccl"):
                    label += f"\\n{d['served_ccl']}"
                elif d.get("kind") == "dropped":
                    label += "\\nDROPPED"
            node.set_label(label)
            node.set_style("filled")
            node.set_fillcolor(_flow_hex(d["flow"], vmax))
            node.set_fontsize("9")
            if d["is_root"]:
                node.set_shape("doubleoctagon")
                node.set_color("black")
                node.set_penwidth("2")
            elif d["is_leaf"]:
                node.set_shape("box")
                node.set_color(leaf_outline.get(d.get("kind"), "#777777"))
                node.set_penwidth("2")
            else:
                node.set_shape("ellipse")
                node.set_color("#777777")
        for edge in pd.get_edges():
            try:
                s = key_by_str.get(edge.get_source().strip('"'))
                t = key_by_str.get(edge.get_destination().strip('"'))
                dia = graph.edges[s, t]["diameter"]
            except Exception:
                continue
            edge.set_label(f"{dia:.2f}")
            edge.set_fontsize("8")
            edge.set_fontcolor("#333333")
        svg_path = out_dir / "giessen_flow_tree.svg"
        png_path = out_dir / "giessen_flow_tree.png"
        pd.write_svg(str(svg_path), prog="dot")
        pd.write_png(str(png_path), prog="dot")
        print(f"[TREE] Wrote flow-tree diagram: {svg_path} and {png_path}")
        return svg_path
    except Exception as exc:
        print(f"[TREE][WARN] Graphviz render failed ({exc}); trying matplotlib fallback.")

    # Fallback: matplotlib.
    try:
        import networkx as nx  # noqa: PLC0415
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        try:
            pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")
        except Exception:
            roots = [n for n, d in graph.nodes(data=True) if d["is_root"]]
            pos = nx.bfs_layout(graph, roots[0]) if roots else nx.spring_layout(graph)
        n_nodes = graph.number_of_nodes()
        side = max(8.0, n_nodes ** 0.5 * 1.5)
        fig, ax = plt.subplots(figsize=(side, side))
        node_colors = [_flow_hex(d["flow"], vmax) for _, d in graph.nodes(data=True)]
        nx.draw_networkx_edges(graph, pos, ax=ax, arrows=False, edge_color="#999999")
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=320)
        nlabels = {n: f"{d['flow']*100:.1f}%\n({d['local_split']*100:.0f}%)"
                   for n, d in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=nlabels, font_size=6, ax=ax)
        elabels = {(u, v): f"{d['diameter']:.2f}" for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=elabels, font_size=5, ax=ax)
        ax.set_axis_off()
        png_path = out_dir / "giessen_flow_tree.png"
        fig.tight_layout()
        fig.savefig(str(png_path), dpi=200)
        if show:
            plt.show()
        plt.close(fig)
        print(f"[TREE] Wrote flow-tree diagram (matplotlib): {png_path}")
        return png_path
    except Exception as exc:
        print(f"[TREE][WARN] flow-tree diagram failed: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    global VIZ_SHOW_CFX_MESH
    if args and args[-1].strip().lower() in ("nomesh", "--nomesh", "--no-mesh"):
        args.pop()
        VIZ_SHOW_CFX_MESH = False
    # Study-mode options are intentionally accepted after the legacy positional
    # arguments so existing commands remain valid.
    opening_terminal_id = None
    inlet_mass_flow_kg_s = None
    boundary_only = False
    retained = []
    for arg in args:
        if arg.startswith("--opening-terminal-id="):
            opening_terminal_id = arg.split("=", 1)[1]
        elif arg.startswith("--inlet-mass-flow-kg-s="):
            inlet_mass_flow_kg_s = float(arg.split("=", 1)[1])
        elif arg == "--boundary-only":
            boundary_only = True
        else:
            retained.append(arg)
    args = retained
    surf = None
    if len(args) == 0:
        xml = config.INPUT_XML
        msh = DEFAULT_MSH_FILE
        out = DEFAULT_OUTPUT_DIR
    elif len(args) in (3, 4):
        xml, msh, out = args[:3]
        surf = args[3] if len(args) == 4 else None
    else:
        print("usage: python -m coronary_sdf.flow_fractions "
              "[<input.am.xml> <input.msh> <output_dir> "
              "[<lumen_surface.vtk>|ask]] [nomesh] "
              "[--opening-terminal-id=edgeN:nodeM] [--boundary-only] "
              "[--inlet-mass-flow-kg-s=VALUE]")
        return 2

    print("=" * 60)
    print("CORONARY LUMEN -- Giessen outlet flow fractions")
    print(f"  exponent = {GIESSEN_EXPONENT}, frames downstream = {FRAMES_DOWNSTREAM}")
    print(f"  Strahler filter >= {config.MIN_STRAHLER_ORDER}")
    print("=" * 60)
    run(
        xml, msh, out, surface_vtk=surf,
        opening_terminal_id=opening_terminal_id,
        boundary_only=boundary_only,
        inlet_mass_flow_kg_s=inlet_mass_flow_kg_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
