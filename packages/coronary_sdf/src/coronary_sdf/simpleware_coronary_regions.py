"""Create coronary outlet boundaries and small-vessel mesh refinements.

Target API: Simpleware X-2025.06 (Python 3).

Run this file from Simpleware's Scripting tab with the coronary surface and CFD
model already present.  The script:

1. crops an Amira SpatialGraph to the exact meshing STL, then finds degree-one
   nodes in that cropped graph (including cuts through any Strahler order);
2. treats the largest terminal(s) as inlet(s) and creates finite clipping-plane
   ROIs at both inlet and outlet terminals;
3. adds velocity-inlet and pressure-outlet CFD contacts so the ROIs become
   exported boundaries;
4. uses graph radii (Amira) or surface-derived radii (Simpleware) and places
   overlapping ellipsoid +FE Free refinement volumes on small vessels.

The original recorded macro is retained separately in
``simpleware_boundary_region_mesh_refinement_API.py``.
"""

import math
import heapq
import os
import re
import struct
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from simpleware.scripting import (
        App,
        Doc,
        FeFreeMeshRefinementVolume,
        IntVector,
        Model,
        PartVector,
        RealPoint3D,
        RealSize,
        Vector3D,
    )
    _SIMPLEWARE_IMPORT_ERROR = None
except ImportError as exc:  # Allows geometry helpers to be tested outside Simpleware.
    App = Doc = FeFreeMeshRefinementVolume = IntVector = Model = None
    PartVector = RealPoint3D = RealSize = Vector3D = None
    _SIMPLEWARE_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# User configuration (all distances are millimetres)
# ---------------------------------------------------------------------------

# Boundary planes and refinements use the cropped Amira graph. The persisted
# Simpleware network is not used for boundary discovery because its generated
# topology omits some surface branches and can stop on non-terminal vessels.
# Strahler order is per-edge Amira graph data.
# ``CENTRELINE_SOURCE`` and ``BOUNDARY_CENTRELINE_SOURCE`` are retained as
# compatibility defaults for helper calls and older project configurations.
CENTRELINE_SOURCE = "amira"  # compatibility default: simpleware or amira
BOUNDARY_CENTRELINE_SOURCE = "amira"
BOUNDARY_CENTRELINE_SOURCES = ("amira",)
REFINEMENT_CENTRELINE_SOURCE = "amira"

# Terminal endpoints from the two extraction methods can land at slightly
# different axial positions on the same outlet. Amira is retained and the
# matching Simpleware terminal is suppressed within this distance.
BOUNDARY_TERMINAL_MERGE_DISTANCE_MM = 1.0
# Boundary eligibility is deliberately independent of Strahler order. A surface
# crop can terminate midway along an order-2/3 trunk, and flow_fractions.py
# likewise matches an outlet to any segment. Strahler is refinement-only.
BOUNDARY_OUTLET_STRAHLER_ORDERS = None

# Simpleware executes Scripting-tab contents as an internal object named
# "Script", so __file__ may not identify this repository. Set its directory
# explicitly so parse_amira.py can import its package-relative dependencies.
# Empty means "derive it from __file__", which is correct for a normal checkout
# or install; set CORONARY_SDF_REPOSITORY_DIR only when running from Simpleware's
# Scripting tab, where __file__ does not identify this package.
CORONARY_SDF_REPOSITORY_DIR = os.environ.get("CORONARY_SDF_REPOSITORY_DIR", "")

# The repository parser normalises its output to micrometres; 0.001 converts it
# to the millimetres used by Simpleware. The graph and surface must already share
# the same physical coordinate system. An optional affine handles registration.
AMIRA_SPATIAL_GRAPH_PATH = os.environ.get("CORONARY_SDF_INPUT", "")
AMIRA_OUTPUT_UM_TO_MM = 0.001
# Maximum allowance when checking that graph points fall within the surface's
# axis-aligned bounds. This catches unit/origin mistakes before model mutation.
AMIRA_ALIGNMENT_TOLERANCE_MM = 1.0
AMIRA_TO_SIMPLEWARE_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

# The exact STL used to construct the Simpleware mesh is the crop authority for
# an Amira graph. The STL must be a capped/watertight solid and share the
# Simpleware global frame after the optional scale/affine below.
CROP_AMIRA_TO_STL = True
STL_SURFACE_PATH = os.environ.get("CORONARY_SDF_STL", "")
STL_SCALE_TO_MM = 1.0
STL_TO_SIMPLEWARE_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
GRAPH_CROP_SAMPLE_SPACING_MM = 0.20
GRAPH_CROP_BISECTION_ITERATIONS = 24
GRAPH_CROP_MIN_FRAGMENT_LENGTH_MM = 0.20
# "relative" retains every substantial disconnected coronary tree while
# rejecting short cropped-branch ostia; "largest" keeps one tree; "all" keeps
# every in-STL fragment. Relative size is measured by retained edge count.
STL_GRAPH_COMPONENT_MODE = "relative"  # relative, largest, or all
STL_GRAPH_MIN_COMPONENT_EDGE_FRACTION = 0.25
STL_RAY_GRID_TARGET_TRIANGLES = 64
STL_RAY_GRID_MAX_CELLS_PER_AXIS = 256
# The console runner supplies an STL extracted from the SIP's own persisted
# surface.  A loose tolerance previously allowed a geometrically different
# pre-import STL to pass while its outlet cuts differed substantially.
STL_SELECTED_SURFACE_BOUNDS_TOLERANCE_MM = 0.01

# Leave blank to use the active surface/model. If the Simpleware source is used
# and more than one visible network exists, CENTRELINE_NETWORK_NAME is required.
SURFACE_NAME = ""
PART_NAME = ""
CENTRELINE_NETWORK_NAME = ""

# Generate centrelines only when the document currently contains none.
GENERATE_CENTRELINES_IF_NONE = True

# Terminal classification. Explicit names take precedence. If the set is empty,
# the INLET_COUNT terminals with the largest estimated radii are treated as
# inlets. They receive their own clipping planes and velocity-inlet contacts by
# default; disable CREATE_INLET_PLANES only for plane-placement diagnostics.
INLET_NODE_NAMES = set()
INLET_COUNT = 1
CREATE_INLET_PLANES = True

# Legacy non-STL fallback placement. When the configured STL is available, the
# exact loop search below supersedes these node-count settings.
OUTLET_INWARD_NODE_COUNT = 2
OUTLET_MIN_INSET_MM = 0.0
TANGENT_AVERAGING_NODE_COUNT = 5
# The final position is no longer taken blindly from the densified graph node
# count.  Starting at the actual STL exit/cap, search inward for the first plane
# that cuts one (and only one) complete closed surface loop.  This keeps planes
# close to cropped ends while avoiding hairpins, adjacent branches, and partial
# cuts.  The graph still supplies branch identity and the initial tangent.
STL_PLANE_SEARCH_MIN_INSET_MM = 0.10
STL_PLANE_SEARCH_STEP_MM = 0.05
STL_PLANE_SEARCH_MAX_INSET_RADIUS_FACTOR = 3.0
STL_PLANE_SEARCH_MIN_MAX_INSET_MM = 0.80
# Aim farther inside than the historical 0.10 mm placement. The retained
# terminal branch length is the hard safeguard: no plane may remove more than
# this fraction of the cap-to-junction branch. On a very short cropped fragment
# the target and minimum inset are therefore reduced automatically.
STL_PLANE_DESIRED_INSET_MM = 0.20
STL_PLANE_DESIRED_INSET_RADIUS_FACTOR = 0.75
STL_PLANE_MAX_TERMINAL_BRANCH_FRACTION = 0.20
STL_PLANE_TANGENT_WINDOWS_MM = (0.10, 0.20, 0.35)
# Average the Amira tangent over this arc length. Surface-anchor extension uses
# the terminal chord; each candidate plane uses a chord centred on its own inset
# location so curvature between the outlet and plane cannot make it oblique.
STL_PLANE_TANGENT_AVERAGING_LENGTH_MM = 0.25
# Prefer the actual triangulated outlet cap when one is present.  This is the
# only local surface feature that directly encodes the intended outlet normal;
# fitting the surrounding cylindrical wall instead can be biased by curvature
# or by a nearby vessel.  The Amira tangent remains a directional constraint
# and the fallback when no coherent planar cap is found.
STL_CAP_NORMAL_MAX_GRAPH_DEVIATION_DEGREES = 60.0
STL_CAP_NORMAL_COHERENCE_DEGREES = 8.0
STL_CAP_SEARCH_RADIUS_FACTOR = 2.5
STL_CAP_SEARCH_MIN_RADIUS_MM = 0.30
STL_CAP_MAX_PLANAR_ERROR_MM = 0.025
STL_CAP_MAX_PLANAR_ERROR_RADIUS_FACTOR = 0.08
STL_CAP_MIN_AREA_RADIUS_FACTOR = 0.20
# Coarsely rotate each graph-tangent seed and minimise the exact STL loop area.
# For a tube this is the perpendicular section; the closed-loop/collision tests
# keep the search from snapping to a nearby vessel.
# The plane normal must remain the locally averaged Amira tangent. Earlier
# versions rotated it by as much as 52.5 degrees to minimise intersection area;
# that could find a smaller but anatomically unrelated/oblique loop. Retain the
# names below for configuration compatibility, but the exact-STL search no
# longer performs that free angular optimisation.
STL_PLANE_NORMAL_SEARCH_DEGREES = ()
STL_PLANE_NORMAL_SEARCH_AZIMUTHS = 8
STL_PLANE_NORMAL_REFINE_DEGREES = 0.0
STL_PLANE_NORMAL_MAX_GRAPH_DEVIATION_DEGREES = 30.0
# Allow a meaningful border beyond every intersected STL facet. The old 1.08
# margin was visually too tight after Simpleware converted the finite ROI.
STL_PLANE_LOOP_MARGIN_FACTOR = 1.10
STL_PLANE_EDGE_PADDING_MM = 0.01
STL_PLANE_NEIGHBOUR_SEARCH_MIN_RADIUS_MM = 2.0
STL_PLANE_INTERSECTION_STITCH_TOLERANCE_MM = 0.002
STL_PLANE_MAX_TERMINAL_EXTENSION_MM = 2.0
STL_PLANE_TERMINAL_EXTENSION_RADIUS_FACTOR = 5.0
# Refine the averaged centreline tangent by fitting the principal axis of the
# nearby lumen-wall vertices. This corrects oblique planes on curved/noisy ends.
REFINE_PLANE_NORMAL_FROM_SURFACE = False
SURFACE_AXIS_AXIAL_RADIUS_FACTOR = 2.50
SURFACE_AXIS_RADIAL_RADIUS_FACTOR = 1.75
SURFACE_AXIS_MIN_SEARCH_RADIUS_MM = 0.40
SURFACE_AXIS_MIN_POINTS = 30
SURFACE_AXIS_MIN_EIGENVALUE_RATIO = 1.15
SURFACE_AXIS_MAX_CORRECTION_DEGREES = 50.0
# Validate plane coverage against the actual Simpleware surface at the inset
# location. This catches graph radii that underestimate a remeshed/offset lumen.
SURFACE_CROSS_SECTION_AXIAL_RADIUS_FACTOR = 0.20
SURFACE_CROSS_SECTION_RADIAL_RADIUS_FACTOR = 2.50
SURFACE_CROSS_SECTION_MIN_SLAB_MM = 0.04
SURFACE_CROSS_SECTION_MIN_SEARCH_RADIUS_MM = 0.40
SURFACE_CROSS_SECTION_MIN_POINTS = 12
SURFACE_CROSS_SECTION_PERCENTILE = 95.0
SURFACE_CROSS_SECTION_MARGIN_FACTOR = 1.10
# If a retained terminal branch is too short to satisfy both inset rules without
# reaching its junction, skip it and print a warning. Set False for fail-fast.
SKIP_UNSAFE_SHORT_TERMINALS = True
OUTLET_PLANE_DIAMETER_FACTOR = 1.35
OUTLET_PLANE_MIN_RADIUS_FACTOR = 1.10
OUTLET_PLANE_MIN_DIAMETER_MM = 0.15
OUTLET_PLANE_MAX_DIAMETER_MM = 20.0

# A plane is also capped using clearance to non-local centreline segments. The
# square's corners must remain inside this clearance. If the safe size cannot
# cover the local vessel radius, the script stops before changing the model.
PLANE_CLEARANCE_SAMPLE_SPACING_MM = 0.25
PLANE_LOCAL_EXCLUSION_MM = 2.0
PLANE_CLEARANCE_SAFETY_FACTOR = 0.85
PLANE_COLLISION_SLAB_MM = 0.10
# A centreline can cross an unrelated lumen, producing a synthetic STL fragment
# end that is not a cap; close true outlets can also be impossible to isolate at
# the required inset. Reject these explicitly instead of creating a contact that
# flood-selects the connected lumen. Every rejection is named in the run log.
SKIP_UNSAFE_CLEARANCE_TERMINALS = True
# X-2025.06 selects the short distal side for these outward-pointing plane
# normals with ``inverted=False`` (matching the recorded manual macro).
INVERT_CLIPPING_PLANES = False

# Creating the CFD contact is what makes a clipping ROI cut the mesh and become
# an exported boundary. Turn this off to inspect the planes without clipping.
ADD_CFD_BOUNDARY_CONDITIONS = True
OUTLET_CONTACT_TYPE = "pressure_outlet"  # pressure_outlet or generic_outlet
INLET_CONTACT_TYPE = "velocity_inlet"    # velocity_inlet or generic_inlet

# Small-vessel definition and refinement volume settings.
CREATE_SMALL_VESSEL_REFINEMENTS = True
# Selection modes: "diameter"/"radius", "strahler", "either"/
# "radius_or_strahler", or "both"/"radius_and_strahler". Strahler values are
# per-edge metadata and therefore require an Amira graph.
REFINEMENT_SELECTION_MODE = "either"
# Use the equivalent radius of the exact STL cross-section for both local
# selection and primitive coverage. "graph" retains the legacy Amira-radius
# behaviour. The graph radius remains the fallback where a unique STL loop
# cannot be measured (normally at a junction).
REFINEMENT_RADIUS_SOURCE = "surface_cross_section"  # graph or surface_cross_section
SMALL_VESSEL_DIAMETER_MM = 1.00
SMALL_VESSEL_CROSS_SECTION_RADIUS_MM = 0.50
REFINEMENT_STRAHLER_ORDERS = {1, 2}
MIN_PLAUSIBLE_RADIUS_MM = 0.03
# A 0.30 mm chord keeps neighbouring sphere centres close enough to form a
# smooth swept envelope without the large lateral reach of long primitives.
REFINEMENT_SAMPLE_SPACING_MM = 0.30
# Preferred sphere radius is local vessel radius * (1 + factor) plus the small
# absolute pad. This replaces the former fixed 0.35 mm pad, which overwhelmed
# distal vessels whose physical radius was only 0.05-0.20 mm.
REFINEMENT_PADDING_RADIUS_FACTOR = 0.25
REFINEMENT_PADDING_MM = 0.05
REFINEMENT_MIN_PADDING_MM = 0.03
# Reject oblique/merged STL section loops before their radius can create a
# giant primitive. The Amira radius is the fallback and remains the topology
# authority. A valid equivalent surface radius must stay within this range.
REFINEMENT_SURFACE_RADIUS_MIN_GRAPH_FACTOR = 0.40
REFINEMENT_SURFACE_RADIUS_MAX_GRAPH_FACTOR = 1.60
REFINEMENT_MESH_SIZE_MM = 0.20
# Sensitivity-study mode assigns each sphere its own characteristic length from
# the vessel cross-section: h = 2*r/n_D, clamped to practical Simpleware limits.
# The legacy fixed size above remains the default for ordinary region creation.
REFINEMENT_USE_RADIUS_MESH_SIZE = False
REFINEMENT_ELEMENTS_ACROSS_DIAMETER = 6.0
REFINEMENT_MIN_MESH_SIZE_MM = 0.02
REFINEMENT_MAX_MESH_SIZE_MM = 0.40
REFINEMENT_TYPE = "volume"  # volume or surface
REFINEMENT_PRIMITIVE = "sphere"  # cylinder, ellipsoid, or sphere
# Sphere optimisation operates on the 0.30 mm measurement/selection segments
# and greedily merges consecutive intervals. A merge is accepted only when one
# sphere covers every retained sample section without growing more than 35%
# beyond the largest local wall envelope. This naturally keeps dense spheres on
# small/tortuous vessels and uses wider spacing on broad/straight segments.
REFINEMENT_OPTIMIZE_SPHERE_COUNT = True
REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR = 1.35
REFINEMENT_SPHERE_MAX_ARC_LENGTH_MM = 0.90
# Extend each primitive beyond both chord endpoints. This closes gaps at curved
# joints and makes the union behave like overlapping centreline capsules.
REFINEMENT_AXIAL_OVERLAP_RADIUS_FACTOR = 1.0
# Adapt padding and axial overlap so an unselected nearby branch cannot enter a
# refinement primitive. Selected neighbouring segments may overlap because
# they belong to the same requested refinement region. If even the target wall
# plus minimum padding cannot be isolated, skip that primitive and report it.
REFINEMENT_NEIGHBOUR_AWARE = True
REFINEMENT_CLEARANCE_SAMPLE_SPACING_MM = 0.20
REFINEMENT_CLEARANCE_SAFETY_FACTOR = 0.90
REFINEMENT_CLEARANCE_LOCAL_EXCLUSION_MM = 0.10
SKIP_UNSAFE_REFINEMENT_SEGMENTS = True
MAX_REFINEMENT_VOLUMES = 5000

# Objects created by this script receive these prefixes. On a rerun, only
# objects with these prefixes are replaced; unrelated user objects are retained.
OUTLET_PREFIX = "COR_OUTLET_"
INLET_PREFIX = "COR_INLET_"
OPENING_PREFIX = "COR_OPENING_"
REFINEMENT_PREFIX = "COR_SMALL_"
REPLACE_EXISTING = True
# This project contains 79 manually recorded ``Finite planeN`` trials in
# addition to the scripted objects. Remove every old clipping ROI on a rerun so
# exactly one generated plane can remain at each accepted terminal.
REMOVE_ALL_EXISTING_CLIPPING_PLANES = True
# Add short persistent 3-D labels (P001, P002, ...) at the clipping planes.
# Their full annotation names match the ROI names, allowing a visually bad
# plane to be found immediately in the Document tree and diagnostic CSV.
CREATE_PLANE_ID_ANNOTATIONS = True
PLANE_ID_ANNOTATION_PREFIX = "COR_PLANE_ID_"
# Headless ConsoleSimpleware has no Dataset/3D viewer to update. The console
# entry point disables these GUI-only calls; interactive Scripting-tab runs keep
# their existing visibility/highlight behaviour.
UPDATE_GUI_VISIBILITY = True

# A sensitivity study needs one pressure reference.  When enabled, the outlet
# with the greatest graph-geodesic distance from the classified inlet is named
# as an opening.  Its plane geometry is otherwise identical to an outlet plane.
CREATE_DISTAL_OPENING = False
OPENING_CONTACT_TYPE = "generic_outlet"


EPS = 1.0e-12
_RUN_STARTED_AT = time.perf_counter()
_PLANE_ANNOTATIONS_AVAILABLE = None


def _log(message):
    """Write a timestamped progress line that appears immediately in console mode."""
    elapsed = time.perf_counter() - _RUN_STARTED_AT
    print(
        "[{} +{:8.1f}s] {}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            elapsed,
            message,
        ),
        flush=True,
    )


def _xyz(value):
    """Return a Simpleware Vector3D-like object as an ndarray."""
    return np.array(
        [float(value.GetX()), float(value.GetY()), float(value.GetZ())],
        dtype=float,
    )


def _vector3d(values):
    values = np.asarray(values, dtype=float)
    return Vector3D(float(values[0]), float(values[1]), float(values[2]))


def _unit(values, label="vector"):
    values = np.asarray(values, dtype=float)
    length = float(np.linalg.norm(values))
    if length <= EPS:
        raise ValueError("Cannot normalise zero-length {}".format(label))
    return values / length


def rotation_from_positive_z(direction):
    """Return (unit rotation axis, angle degrees) mapping +Z to direction."""
    normal = _unit(direction, "plane normal")
    dot = float(np.clip(normal[2], -1.0, 1.0))
    if dot >= 1.0 - 1.0e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if dot <= -1.0 + 1.0e-10:
        return np.array([1.0, 0.0, 0.0]), 180.0
    axis = _unit(np.cross(np.array([0.0, 0.0, 1.0]), normal), "rotation axis")
    return axis, math.degrees(math.acos(dot))


def rotation_from_basis(first, second, normal):
    """Return axis/angle rotating global XYZ onto a right-handed local basis."""
    first = _unit(first, "plane first axis")
    normal = _unit(normal, "plane normal")
    second = _unit(second, "plane second axis")
    matrix = np.column_stack((first, second, normal))
    # Remove harmless accumulated numerical skew before extracting axis/angle.
    u, _singular, vh = np.linalg.svd(matrix)
    matrix = u.dot(vh)
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u.dot(vh)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-10:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle <= 1.0e-7:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] <= EPS:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for index in range(3):
                if index != largest:
                    axis[index] = (
                        matrix[index, largest] + matrix[largest, index]
                    ) / (4.0 * axis[largest])
        return _unit(axis, "180-degree rotation axis"), 180.0
    axis = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ]) / (2.0 * math.sin(angle))
    return _unit(axis, "rotation axis"), math.degrees(angle)


def _safe_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")
    return cleaned or "unnamed"


class _ArrayVector3D:
    """Small Vector3D-compatible wrapper for Amira coordinates."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def GetX(self):
        return float(self.values[0])

    def GetY(self):
        return float(self.values[1])

    def GetZ(self):
        return float(self.values[2])


class _AmiraNode:
    def __init__(self, node_id, position, name=None):
        self.node_id = int(node_id)
        self.position = np.asarray(position, dtype=float)
        self.splines = []
        self.name = str(name) if name else "AmiraNode_{}".format(self.node_id)

    def GetName(self):
        return self.name

    def GetPosition(self, _image_space):
        return _ArrayVector3D(self.position)

    def GetSplines(self):
        return self.splines


class _AmiraSpline:
    """Polyline adapter exposing the Simpleware spline methods used below."""

    def __init__(
        self,
        edge_id,
        start_node,
        end_node,
        points,
        radii,
        strahler_order=None,
        name=None,
        use_surface_radius=False,
    ):
        self.edge_id = int(edge_id)
        self.start_node = start_node
        self.end_node = end_node
        self.strahler_order = (
            None if strahler_order is None else int(strahler_order)
        )
        self.name = str(name) if name else "AmiraEdge_{}".format(self.edge_id)
        self.use_surface_radius = bool(use_surface_radius)
        points = np.asarray(points, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if len(points) < 2 or len(points) != len(radii):
            raise ValueError("Amira edge {} has invalid point/radius data".format(edge_id))

        # The file normally stores edge points node1 -> node2, but orient by
        # geometry so terminal offsets and radius interpolation stay correct.
        if np.linalg.norm(points[-1] - start_node.position) < np.linalg.norm(
            points[0] - start_node.position
        ):
            points = points[::-1].copy()
            radii = radii[::-1].copy()
        self.points = points
        self.radii = radii
        self.cumulative = _cumulative_distances(points)
        if self.cumulative[-1] <= EPS:
            raise ValueError("Amira edge {} has zero length".format(edge_id))
        start_node.splines.append(self)
        end_node.splines.append(self)

    def GetName(self):
        return self.name

    def IsClosed(self):
        return self.start_node is self.end_node

    def GetLength(self, _image_space):
        return float(self.cumulative[-1])

    def GetStartNode(self):
        return self.start_node

    def GetEndNode(self):
        return self.end_node

    def GetRawDataPoints(self, _image_space):
        return [_ArrayVector3D(point) for point in self.points]

    def GetParameterAtDistance(self, distance, _image_space):
        return float(np.clip(distance / self.cumulative[-1], 0.0, 1.0))

    def _interpolate(self, values, distance):
        distance = float(np.clip(distance, 0.0, self.cumulative[-1]))
        upper = int(np.searchsorted(self.cumulative, distance, side="right"))
        upper = min(max(1, upper), len(self.cumulative) - 1)
        lower = upper - 1
        span = self.cumulative[upper] - self.cumulative[lower]
        fraction = 0.0 if span <= EPS else (distance - self.cumulative[lower]) / span
        return values[lower] + fraction * (values[upper] - values[lower])

    def GetPosition(self, parameter, _image_space):
        point = self._interpolate(self.points, float(parameter) * self.cumulative[-1])
        return _ArrayVector3D(point)

    def RadiusAtDistance(self, distance):
        return float(self._interpolate(self.radii, distance))

    def MedianRadiusNearDistance(self, distance, node_count):
        centre = int(np.argmin(np.abs(self.cumulative - float(distance))))
        requested = max(1, int(node_count))
        half = requested // 2
        first = max(0, centre - half)
        last = min(len(self.radii), first + requested)
        first = max(0, last - requested)
        return float(np.median(self.radii[first:last]))


class _AmiraNetwork:
    def __init__(self, nodes, splines, source_path):
        self.nodes = nodes
        self.splines = splines
        self.source_path = source_path

    def GetName(self):
        return "Amira_{}".format(Path(self.source_path).stem)

    def GetVisible(self):
        return True

    def GetNodes(self):
        return self.nodes

    def GetSplines(self):
        return self.splines


class _SimplewareSipNetwork:
    """Persisted Simpleware spline network decoded from SIP PathsData.bin."""

    def __init__(self, name, nodes, splines, source_path):
        self.name = str(name)
        self.nodes = nodes
        self.splines = splines
        self.source_path = str(source_path)

    def GetName(self):
        return self.name

    def GetVisible(self):
        return True

    def GetNodes(self):
        return self.nodes

    def GetSplines(self):
        return self.splines


class _StlSolid:
    """Watertight triangle solid with an accelerated +X parity test."""

    def __init__(self, triangles, source_path=""):
        triangles = np.asarray(triangles, dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("STL triangles must have shape (N, 3, 3)")
        if len(triangles) == 0:
            raise ValueError("STL contains no triangles")
        self.triangles = triangles
        self.source_path = str(source_path)
        flat = triangles.reshape(-1, 3)
        self.lower = np.min(flat, axis=0)
        self.upper = np.max(flat, axis=0)
        self.diagonal = float(np.linalg.norm(self.upper - self.lower))
        if self.diagonal <= EPS:
            raise ValueError("STL has zero spatial extent")
        self._ray_epsilon = max(1.0e-10, self.diagonal * 1.0e-10)
        self._build_ray_grid()
        self._build_triangle_index()

    def _build_triangle_index(self):
        """Index triangle centres for local, exact plane/surface intersections."""
        self._triangle_centres = np.mean(self.triangles, axis=1)
        self._triangle_radii = np.max(
            np.linalg.norm(
                self.triangles - self._triangle_centres[:, None, :], axis=2
            ),
            axis=1,
        )
        self._max_triangle_radius = float(np.max(self._triangle_radii))
        _log("Building STL triangle spatial index for exact plane validation...")
        self._triangle_tree = cKDTree(self._triangle_centres)
        _log(
            "STL triangle spatial index complete ({:,} triangles; maximum "
            "triangle radius {:.4f} mm).".format(
                len(self.triangles), self._max_triangle_radius
            )
        )

    def nearby_triangles(self, centre, radius):
        """Return triangles whose bounding spheres can touch a local ball."""
        centre = np.asarray(centre, dtype=float)
        radius = float(radius)
        indices = np.asarray(
            self._triangle_tree.query_ball_point(
                centre, radius + self._max_triangle_radius
            ),
            dtype=np.int64,
        )
        if len(indices) == 0:
            return self.triangles[:0]
        delta = self._triangle_centres[indices] - centre
        keep = np.linalg.norm(delta, axis=1) <= radius + self._triangle_radii[indices]
        return self.triangles[indices[keep]]

    def _build_ray_grid(self):
        # Rays travel along +X, so only triangle bounds in YZ are indexed.
        spans = np.maximum(self.upper[1:] - self.lower[1:], self._ray_epsilon)
        target_per_cell = max(1.0, float(STL_RAY_GRID_TARGET_TRIANGLES))
        target_cells = max(1.0, len(self.triangles) / target_per_cell)
        ny = int(round(math.sqrt(target_cells * spans[0] / spans[1])))
        nz = int(round(target_cells / max(1, ny)))
        limit = max(4, int(STL_RAY_GRID_MAX_CELLS_PER_AXIS))
        self._grid_shape = (
            min(limit, max(4, ny)),
            min(limit, max(4, nz)),
        )
        self._grid_step = spans / np.asarray(self._grid_shape, dtype=float)
        projected = self.triangles[:, :, 1:]
        lo = np.floor(
            (np.min(projected, axis=1) - self.lower[1:]) / self._grid_step
        ).astype(int)
        hi = np.floor(
            (np.max(projected, axis=1) - self.lower[1:]) / self._grid_step
        ).astype(int)
        lo = np.clip(lo, 0, np.asarray(self._grid_shape) - 1)
        hi = np.clip(hi, 0, np.asarray(self._grid_shape) - 1)
        grid = {}
        total_triangles = len(self.triangles)
        progress_interval = max(1, total_triangles // 20)
        _log(
            "Building STL ray index for {:,} triangles on a {} x {} grid..."
            .format(total_triangles, *self._grid_shape)
        )
        for triangle_index in range(len(self.triangles)):
            for iy in range(lo[triangle_index, 0], hi[triangle_index, 0] + 1):
                for iz in range(lo[triangle_index, 1], hi[triangle_index, 1] + 1):
                    grid.setdefault((iy, iz), []).append(triangle_index)
            completed = triangle_index + 1
            if completed % progress_interval == 0 or completed == total_triangles:
                _log(
                    "STL ray index: {:5.1f}% ({:,}/{:,} triangles).".format(
                        100.0 * completed / total_triangles,
                        completed,
                        total_triangles,
                    )
                )
        self._ray_grid = {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in grid.items()
        }
        _log("STL ray index complete: {:,} occupied grid cells.".format(len(grid)))

    def _contains_one(self, point):
        point = np.asarray(point, dtype=float).copy()
        if np.any(point < self.lower - self._ray_epsilon) or np.any(
            point > self.upper + self._ray_epsilon
        ):
            return False

        # Move the ray off coincident mesh edges/vertices without materially
        # moving the test point.
        point[1] += 0.754877666 * self._ray_epsilon
        point[2] += 0.569840291 * self._ray_epsilon
        cell = np.floor(
            (point[1:] - self.lower[1:]) / self._grid_step
        ).astype(int)
        cell = np.clip(cell, 0, np.asarray(self._grid_shape) - 1)
        indices = self._ray_grid.get((int(cell[0]), int(cell[1])))
        if indices is None:
            return False

        triangles = self.triangles[indices]
        y0, z0 = triangles[:, 0, 1], triangles[:, 0, 2]
        y1, z1 = triangles[:, 1, 1], triangles[:, 1, 2]
        y2, z2 = triangles[:, 2, 1], triangles[:, 2, 2]
        denominator = (z1 - z2) * (y0 - y2) + (y2 - y1) * (z0 - z2)
        valid = np.abs(denominator) > self._ray_epsilon
        if not np.any(valid):
            return False
        triangles = triangles[valid]
        denominator = denominator[valid]
        y0, z0 = y0[valid], z0[valid]
        y1, z1 = y1[valid], z1[valid]
        y2, z2 = y2[valid], z2[valid]
        alpha = (
            (z1 - z2) * (point[1] - y2)
            + (y2 - y1) * (point[2] - z2)
        ) / denominator
        beta = (
            (z2 - z0) * (point[1] - y2)
            + (y0 - y2) * (point[2] - z2)
        ) / denominator
        gamma = 1.0 - alpha - beta
        barycentric_tolerance = 1.0e-10
        hit = (
            (alpha >= -barycentric_tolerance)
            & (beta >= -barycentric_tolerance)
            & (gamma >= -barycentric_tolerance)
        )
        if not np.any(hit):
            return False
        triangles = triangles[hit]
        alpha, beta, gamma = alpha[hit], beta[hit], gamma[hit]
        intersection_x = (
            alpha * triangles[:, 0, 0]
            + beta * triangles[:, 1, 0]
            + gamma * triangles[:, 2, 0]
        )
        intersection_x = np.sort(
            intersection_x[intersection_x > point[0] + self._ray_epsilon]
        )
        if len(intersection_x) == 0:
            return False
        distinct = np.concatenate((
            [True], np.diff(intersection_x) > 10.0 * self._ray_epsilon
        ))
        return bool(np.count_nonzero(distinct) % 2)

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        result = np.asarray([self._contains_one(point) for point in points])
        return bool(result[0]) if scalar else result


def _transform_amira_positions(values):
    values = np.asarray(values, dtype=float) * float(AMIRA_OUTPUT_UM_TO_MM)
    affine = np.asarray(AMIRA_TO_SIMPLEWARE_AFFINE, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError("AMIRA_TO_SIMPLEWARE_AFFINE must be a 4x4 matrix")
    homogeneous = np.ones((len(values), 4), dtype=float)
    homogeneous[:, :3] = values
    transformed = homogeneous.dot(affine.T)
    w = transformed[:, 3]
    if np.any(np.abs(w) <= EPS):
        raise ValueError("AMIRA_TO_SIMPLEWARE_AFFINE produced a zero homogeneous w")
    return transformed[:, :3] / w[:, None]


def _transform_stl_triangles(triangles):
    triangles = np.asarray(triangles, dtype=float)
    flat = triangles.reshape(-1, 3) * float(STL_SCALE_TO_MM)
    affine = np.asarray(STL_TO_SIMPLEWARE_AFFINE, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError("STL_TO_SIMPLEWARE_AFFINE must be a 4x4 matrix")
    homogeneous = np.ones((len(flat), 4), dtype=float)
    homogeneous[:, :3] = flat
    transformed = homogeneous.dot(affine.T)
    w = transformed[:, 3]
    if np.any(np.abs(w) <= EPS):
        raise ValueError("STL_TO_SIMPLEWARE_AFFINE produced a zero homogeneous w")
    return (transformed[:, :3] / w[:, None]).reshape(-1, 3, 3)


def _read_stl_triangles(path):
    """Read binary or ASCII STL triangles without optional mesh packages."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(84)
    binary_count = struct.unpack("<I", header[80:84])[0] if len(header) == 84 else -1
    expected_binary_size = 84 + 50 * binary_count
    if binary_count >= 0 and expected_binary_size == file_size:
        record_type = np.dtype([
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ])
        records = np.fromfile(str(path), dtype=record_type, offset=84)
        return np.asarray(records["vertices"], dtype=float)

    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append([float(value) for value in fields[1:]])
    if len(vertices) == 0 or len(vertices) % 3:
        raise RuntimeError("Invalid or unsupported STL file: {}".format(path))
    return np.asarray(vertices, dtype=float).reshape(-1, 3, 3)


def _stl_open_boundary_loops(triangles):
    """Return ordered open-edge loops from an STL triangle soup."""
    triangles = np.asarray(triangles, dtype=float)
    triangle_count = len(triangles)
    flat = np.asarray(triangles, dtype="<f4").reshape(-1, 3)
    bits = flat.view(np.uint32).reshape(-1, 3).astype(np.uint64)
    vertex_hashes = (
        bits[:, 0] * np.uint64(11400714819323198485)
        ^ bits[:, 1] * np.uint64(14029467366897019727)
        ^ bits[:, 2] * np.uint64(1609587929392839161)
    ).reshape(triangle_count, 3)

    edge_start = np.empty(3 * triangle_count, dtype=np.uint64)
    edge_end = np.empty(3 * triangle_count, dtype=np.uint64)
    edge_start[:triangle_count] = vertex_hashes[:, 0]
    edge_end[:triangle_count] = vertex_hashes[:, 1]
    edge_start[triangle_count:2 * triangle_count] = vertex_hashes[:, 1]
    edge_end[triangle_count:2 * triangle_count] = vertex_hashes[:, 2]
    edge_start[2 * triangle_count:] = vertex_hashes[:, 2]
    edge_end[2 * triangle_count:] = vertex_hashes[:, 0]
    edge_low = np.minimum(edge_start, edge_end)
    edge_high = np.maximum(edge_start, edge_end)
    order = np.lexsort((edge_high, edge_low))
    sorted_low = edge_low[order]
    sorted_high = edge_high[order]
    repeated = (
        (sorted_low[1:] == sorted_low[:-1])
        & (sorted_high[1:] == sorted_high[:-1])
    )
    is_boundary = np.ones(len(order), dtype=bool)
    is_boundary[1:] &= ~repeated
    is_boundary[:-1] &= ~repeated
    boundary_edge_indices = order[is_boundary]

    boundary_start = edge_start[boundary_edge_indices]
    boundary_end = edge_end[boundary_edge_indices]
    adjacency = defaultdict(list)
    coordinates = {}
    local_start = (0, 1, 2)
    local_end = (1, 2, 0)
    for edge_index, start_hash, end_hash in zip(
        boundary_edge_indices, boundary_start, boundary_end
    ):
        triangle_index = int(edge_index % triangle_count)
        block = int(edge_index // triangle_count)
        start_key, end_key = int(start_hash), int(end_hash)
        adjacency[start_key].append(end_key)
        adjacency[end_key].append(start_key)
        coordinates.setdefault(
            start_key, triangles[triangle_index, local_start[block]].copy()
        )
        coordinates.setdefault(
            end_key, triangles[triangle_index, local_end[block]].copy()
        )

    loops = []
    visited_edges = set()
    for start, neighbours in adjacency.items():
        for first_next in neighbours:
            edge_key = (min(start, first_next), max(start, first_next))
            if edge_key in visited_edges:
                continue
            loop = [start]
            previous, current = start, first_next
            visited_edges.add(edge_key)
            while current != start:
                loop.append(current)
                candidates = [item for item in adjacency[current] if item != previous]
                if len(candidates) != 1:
                    loop = []
                    break
                following = candidates[0]
                edge_key = (min(current, following), max(current, following))
                if edge_key in visited_edges and following != start:
                    loop = []
                    break
                visited_edges.add(edge_key)
                previous, current = current, following
                if len(loop) > len(adjacency) + 1:
                    loop = []
                    break
            if len(loop) >= 3:
                loops.append(np.asarray([coordinates[key] for key in loop]))
    return loops, int(len(boundary_edge_indices))


def _describe_stl_boundary_loop(points):
    """Return centroid, plane normal, area, diameter, and planarity metrics."""
    points = np.asarray(points, dtype=float)
    centre = np.mean(points, axis=0)
    relative = points - centre
    next_relative = np.roll(relative, -1, axis=0)
    area_vector = 0.5 * np.sum(np.cross(relative, next_relative), axis=0)
    area = float(np.linalg.norm(area_vector))
    if area <= EPS:
        raise RuntimeError("STL boundary loop has zero projected area")
    normal = area_vector / area
    perimeter = float(np.sum(np.linalg.norm(
        np.roll(points, -1, axis=0) - points, axis=1
    )))
    equivalent_diameter = 2.0 * math.sqrt(area / math.pi)
    planarity_error = float(np.max(np.abs(np.sum(relative * normal, axis=1))))
    compactness = 4.0 * math.pi * area / max(perimeter * perimeter, EPS)
    return {
        "points": points,
        "centre": centre,
        "normal": normal,
        "area": area,
        "diameter": equivalent_diameter,
        "perimeter": perimeter,
        "planarity_error": planarity_error,
        "compactness": compactness,
    }


def _load_stl_solid():
    if not STL_SURFACE_PATH:
        raise RuntimeError(
            "Set STL_SURFACE_PATH to the exact capped STL used for meshing when "
            "CROP_AMIRA_TO_STL=True"
        )
    stl_path = Path(STL_SURFACE_PATH).expanduser().resolve()
    if not stl_path.is_file():
        raise RuntimeError("Meshing STL not found: {}".format(stl_path))
    _log("Reading meshing STL: {!s}".format(stl_path))
    triangles = _transform_stl_triangles(_read_stl_triangles(stl_path))
    _log("Loaded meshing STL {!s}: {:,} triangles.".format(
        stl_path, len(triangles)
    ))
    return _StlSolid(triangles, stl_path)


def _load_amira_network():
    if not AMIRA_SPATIAL_GRAPH_PATH:
        raise RuntimeError(
            "Set AMIRA_SPATIAL_GRAPH_PATH when CENTRELINE_SOURCE='amira'"
        )
    graph_path = Path(AMIRA_SPATIAL_GRAPH_PATH).expanduser().resolve()
    if not graph_path.is_file():
        raise RuntimeError("Amira SpatialGraph not found: {}".format(graph_path))

    # Add the package parent rather than the package directory so imports such
    # as coronary_sdf.parse_amira can resolve parse_amira's relative imports.
    if CORONARY_SDF_REPOSITORY_DIR:
        package_dir = Path(CORONARY_SDF_REPOSITORY_DIR).expanduser().resolve()
    else:
        script_file = globals().get("__file__")
        package_dir = (
            Path(script_file).resolve().parent if script_file else Path.cwd()
        )
    if not (package_dir / "__init__.py").is_file() or not (
        package_dir / "parse_amira.py"
    ).is_file():
        raise RuntimeError(
            "CORONARY_SDF_REPOSITORY_DIR must point to the coronary_sdf package "
            "directory containing __init__.py and parse_amira.py; got {}"
            .format(package_dir)
        )
    package_parent = str(package_dir.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    try:
        from coronary_sdf.parse_amira import parse_graph
    except ImportError as exc:
        raise RuntimeError(
            "Could not import coronary_sdf.parse_amira from {}. Verify "
            "CORONARY_SDF_REPOSITORY_DIR and its package dependencies."
            .format(package_dir)
        ) from exc

    _log("Reading Amira SpatialGraph: {!s}".format(graph_path))
    raw_nodes, raw_points, raw_segments = parse_graph(graph_path)
    node_ids = sorted(raw_nodes)
    node_positions = _transform_amira_positions(
        [raw_nodes[node_id][:3] for node_id in node_ids]
    )
    nodes_by_id = {
        node_id: _AmiraNode(node_id, position)
        for node_id, position in zip(node_ids, node_positions)
    }

    splines = []
    segment_progress_interval = max(1, len(raw_segments) // 10)
    for segment_ordinal, segment in enumerate(raw_segments, start=1):
        point_ids = segment["point_ids"]
        coordinates = _transform_amira_positions(
            [raw_points[point_id][:3] for point_id in point_ids]
        )
        radii = np.asarray(
            [raw_points[point_id][3] for point_id in point_ids], dtype=float
        ) * float(AMIRA_OUTPUT_UM_TO_MM)
        strahler_order = None
        for field_name in (
            "strahler", "StrahlerOrder", "Strahler", "StrahlerNumber"
        ):
            if field_name in segment and segment[field_name] not in (None, ""):
                strahler_order = int(float(segment[field_name]))
                break
        splines.append(_AmiraSpline(
            segment["id"],
            nodes_by_id[segment["node1"]],
            nodes_by_id[segment["node2"]],
            coordinates,
            radii,
            strahler_order,
        ))
        if (
            segment_ordinal % segment_progress_interval == 0
            or segment_ordinal == len(raw_segments)
        ):
            _log(
                "Amira edge adaptation: {:,}/{:,}.".format(
                    segment_ordinal, len(raw_segments)
                )
            )

    network = _AmiraNetwork(list(nodes_by_id.values()), splines, str(graph_path))
    if not network.nodes or not network.splines:
        raise RuntimeError(
            "Amira SpatialGraph contains no usable nodes/edges: {}".format(graph_path)
        )
    _log(
        "Loaded Amira SpatialGraph {!s}: {} nodes, {} edges.".format(
            graph_path, len(nodes_by_id), len(splines)
        )
    )
    return network


def _validate_amira_alignment(network, surface_points, surface_tree):
    """Fail early when an imported SpatialGraph is not registered to the surface."""
    if not isinstance(network, _AmiraNetwork):
        return

    graph_points = np.vstack([spline.points for spline in network.splines])
    graph_radii = np.concatenate([spline.radii for spline in network.splines])
    tolerance = float(AMIRA_ALIGNMENT_TOLERANCE_MM)
    lower = np.min(surface_points, axis=0) - tolerance
    upper = np.max(surface_points, axis=0) + tolerance
    inside = np.all((graph_points >= lower) & (graph_points <= upper), axis=1)
    inside_fraction = float(np.mean(inside))
    if inside_fraction < 0.90:
        raise RuntimeError(
            "Only {:.1%} of Amira graph points overlap the coronary surface "
            "bounds (90% required). Check AMIRA_OUTPUT_UM_TO_MM and "
            "AMIRA_TO_SIMPLEWARE_AFFINE.".format(inside_fraction)
        )

    # A centreline-to-wall distance should be similar to its graph radius. This
    # is diagnostic rather than fatal because stenoses and non-circular sections
    # can make nearest-surface distance differ substantially from Amira radius.
    if len(graph_points) > 2000:
        indices = np.linspace(0, len(graph_points) - 1, 2000, dtype=int)
        check_points = graph_points[indices]
        check_radii = graph_radii[indices]
    else:
        check_points = graph_points
        check_radii = graph_radii
    valid = check_radii > EPS
    if np.any(valid):
        wall_distances = np.asarray(
            surface_tree.query(check_points[valid], k=1)[0], dtype=float
        )
        median_ratio = float(np.median(wall_distances / check_radii[valid]))
        _log(
            "Amira/surface alignment: {:.1%} in bounds; median wall-distance/"
            "graph-radius = {:.3f}.".format(inside_fraction, median_ratio)
        )
        if not 0.5 <= median_ratio <= 2.0:
            _log(
                "WARNING: Amira radii differ markedly from nearest surface "
                "distances; verify graph units and registration."
            )


def _surface_points_global(surface):
    """Return all surface vertices transformed into global coordinates."""
    sw_points = surface.GetPoints()
    count = len(sw_points)
    if count == 0:
        raise RuntimeError("Surface {!r} contains no vertices".format(surface.GetName()))

    points = np.empty((count, 3), dtype=float)
    progress_interval = max(1, count // 10)
    for index, point in enumerate(sw_points):
        points[index] = _xyz(point)
        completed = index + 1
        if completed % progress_interval == 0 or completed == count:
            _log(
                "Surface vertex extraction: {:5.1f}% ({:,}/{:,}).".format(
                    100.0 * completed / count, completed, count
                )
            )

    matrix = surface.GetMatrix()
    transform = np.array(
        [[float(matrix.GetItem(row, column)) for column in range(4)]
         for row in range(4)],
        dtype=float,
    )
    homogeneous = np.ones((count, 4), dtype=float)
    homogeneous[:, :3] = points
    transformed = homogeneous.dot(transform.T)
    w = transformed[:, 3]
    non_unit_w = np.abs(w) > EPS
    transformed[non_unit_w, :3] /= w[non_unit_w, None]
    return transformed[:, :3]


def _choose_surface(document):
    if SURFACE_NAME:
        return document.GetSurfaceByName(SURFACE_NAME)

    try:
        return document.GetActiveSurface()
    except Exception:
        surfaces = list(document.GetSurfaces())
        if len(surfaces) == 1:
            return surfaces[0]
        names = [surface.GetName() for surface in surfaces]
        raise RuntimeError(
            "Select the coronary surface or set SURFACE_NAME. Available: {}".format(names)
        )


def _protobuf_read_varint(data, offset):
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise RuntimeError("Truncated PathsData protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise RuntimeError("Invalid PathsData protobuf varint")


def _protobuf_fields(data):
    """Return raw ``(number, wire_type, value)`` protobuf fields."""
    result = []
    offset = 0
    while offset < len(data):
        key, offset = _protobuf_read_varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if number <= 0:
            raise RuntimeError("Invalid zero-number PathsData protobuf field")
        if wire_type == 0:
            value, offset = _protobuf_read_varint(data, offset)
        elif wire_type == 1:
            value = data[offset:offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = _protobuf_read_varint(data, offset)
            value = data[offset:offset + length]
            offset += length
        elif wire_type == 5:
            value = data[offset:offset + 4]
            offset += 4
        else:
            raise RuntimeError(
                "Unsupported PathsData protobuf wire type {}".format(wire_type)
            )
        if offset > len(data):
            raise RuntimeError("Truncated PathsData protobuf field")
        result.append((number, wire_type, value))
    return result


def _protobuf_values(message, number, wire_type=None):
    return [
        value
        for field_number, field_wire, value in _protobuf_fields(message)
        if field_number == number
        and (wire_type is None or field_wire == wire_type)
    ]


def _protobuf_text(message, number):
    values = _protobuf_values(message, number, 2)
    if not values:
        raise RuntimeError("PathsData field {} is missing".format(number))
    return values[0].decode("utf-8")


def _protobuf_wrapped_text(message, number):
    values = _protobuf_values(message, number, 2)
    if not values:
        raise RuntimeError("PathsData wrapped field {} is missing".format(number))
    return _protobuf_text(values[0], 1)


def _protobuf_vector3(message):
    components = []
    parsed = _protobuf_fields(message)
    for number in (1, 2, 3):
        values = [
            value for field_number, wire_type, value in parsed
            if field_number == number and wire_type == 1
        ]
        if len(values) != 1 or len(values[0]) != 8:
            raise RuntimeError("Invalid PathsData Vector3D field")
        components.append(struct.unpack("<d", values[0])[0])
    return np.asarray(components, dtype=float)


def _load_simpleware_network_from_sip(document):
    """Decode the persisted surface-generated network without GUI-only APIs."""
    project_path = Path(document.GetFilePath()).resolve()
    if not project_path.is_file():
        raise RuntimeError(
            "The Simpleware centreline source requires a saved SIP project"
        )
    _log("Reading persisted Simpleware centrelines from {!s}...".format(
        project_path
    ))
    with zipfile.ZipFile(str(project_path), "r") as archive:
        try:
            paths_data = archive.read("PathsData.bin")
        except KeyError as exc:
            raise RuntimeError(
                "The SIP contains no PathsData.bin centreline data"
            ) from exc

    root_values = _protobuf_values(paths_data, 1, 2)
    if len(root_values) != 1:
        raise RuntimeError("Unexpected PathsData root structure")
    container = root_values[0]
    node_messages = _protobuf_values(container, 1, 2)
    spline_messages = _protobuf_values(container, 2, 2)
    network_messages = _protobuf_values(container, 5, 2)
    if not network_messages:
        raise RuntimeError("PathsData contains no spline network metadata")

    networks = []
    for message in network_messages:
        networks.append({
            "uuid": _protobuf_wrapped_text(message, 1),
            "name": _protobuf_text(message, 2),
        })
    if CENTRELINE_NETWORK_NAME:
        matches = [
            item for item in networks if item["name"] == CENTRELINE_NETWORK_NAME
        ]
    else:
        matches = networks
    if len(matches) != 1:
        raise RuntimeError(
            "Set CENTRELINE_NETWORK_NAME because SIP network candidates are: {}"
            .format([item["name"] for item in networks])
        )
    selected = matches[0]

    node_records = []
    for message in node_messages:
        if _protobuf_wrapped_text(message, 2) != selected["uuid"]:
            continue
        positions = _protobuf_values(message, 6, 2)
        if len(positions) != 1:
            raise RuntimeError("PathsData node position is missing or duplicated")
        node_records.append({
            "uuid": _protobuf_wrapped_text(message, 1),
            "name": _protobuf_text(message, 3),
            "position": _protobuf_vector3(positions[0]),
        })
    nodes_by_uuid = {}
    for ordinal, record in enumerate(node_records):
        nodes_by_uuid[record["uuid"]] = _AmiraNode(
            ordinal, record["position"], name=record["name"]
        )

    splines = []
    for message in spline_messages:
        start_uuid = _protobuf_wrapped_text(message, 4)
        end_uuid = _protobuf_wrapped_text(message, 5)
        if start_uuid not in nodes_by_uuid or end_uuid not in nodes_by_uuid:
            continue
        raw_point_messages = _protobuf_values(message, 8, 2)
        if len(raw_point_messages) < 2:
            continue
        representation_values = _protobuf_values(message, 16, 2)
        spline_name = (
            _protobuf_text(representation_values[0], 3)
            if representation_values else "SipLine_{:03d}".format(len(splines))
        )
        points = np.asarray([
            _protobuf_vector3(point) for point in raw_point_messages
        ])
        splines.append(_AmiraSpline(
            len(splines),
            nodes_by_uuid[start_uuid],
            nodes_by_uuid[end_uuid],
            points,
            np.zeros(len(points), dtype=float),
            name=spline_name,
            use_surface_radius=True,
        ))

    nodes = list(nodes_by_uuid.values())
    if not nodes or not splines:
        raise RuntimeError(
            "Selected SIP centreline network contains no usable topology"
        )
    degree_counts = {}
    for node in nodes:
        degree = len(node.GetSplines())
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    _log(
        "Decoded SIP network {!r}: {} nodes, {} splines, degree counts {}."
        .format(selected["name"], len(nodes), len(splines), degree_counts)
    )
    return _SimplewareSipNetwork(
        selected["name"], nodes, splines, project_path
    )


def _choose_network(document):
    centrelines = document.GetCentrelines()
    networks = list(centrelines.GetNetworks(False))
    if not networks and GENERATE_CENTRELINES_IF_NONE:
        _log("No centreline network found; generating centrelines...")
        centrelines.Generate()
        networks = list(centrelines.GetNetworks(False))

    if CENTRELINE_NETWORK_NAME:
        return centrelines.GetNetwork(CENTRELINE_NETWORK_NAME)

    visible = [network for network in networks if network.GetVisible()]
    candidates = visible if visible else networks
    if len(candidates) != 1:
        names = [network.GetName() for network in candidates]
        raise RuntimeError(
            "Set CENTRELINE_NETWORK_NAME because {} candidate networks exist: {}".format(
                len(candidates), names
            )
        )
    return candidates[0]


def _choose_centreline_source(document, source=None):
    source = CENTRELINE_SOURCE if source is None else str(source).lower()
    if source == "amira":
        return _load_amira_network()
    if source == "simpleware":
        return _load_simpleware_network_from_sip(document)
    raise ValueError("Centreline source must be 'simpleware' or 'amira'")


def _choose_part(model, surface):
    container = model.GetPartsContainer()
    if PART_NAME:
        return container.GetPartByName(PART_NAME)

    parts = list(container.GetParts())
    matching = [part for part in parts if part.GetName() == surface.GetName()]
    if len(matching) == 1:
        return matching[0]
    if len(parts) == 1:
        return parts[0]
    raise RuntimeError(
        "Set PART_NAME. Model parts are: {}".format([part.GetName() for part in parts])
    )


def _ordered_raw_points(spline, terminal_position):
    """Return raw spline points ordered from the terminal toward the tree."""
    points = np.asarray([_xyz(point) for point in spline.GetRawDataPoints(False)])
    if len(points) < 2:
        raise RuntimeError("Spline {!r} has fewer than two raw data points".format(
            spline.GetName()
        ))
    if np.linalg.norm(points[-1] - terminal_position) < np.linalg.norm(
        points[0] - terminal_position
    ):
        points = points[::-1].copy()
    return points


def _cumulative_distances(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def _densify_polyline(points, radii, spacing):
    dense_points = [np.asarray(points[0], dtype=float)]
    dense_radii = [float(radii[0])]
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        length = float(np.linalg.norm(end - start))
        subdivisions = max(1, int(math.ceil(length / float(spacing))))
        for step in range(1, subdivisions + 1):
            fraction = step / float(subdivisions)
            dense_points.append(start + fraction * (end - start))
            dense_radii.append(
                float(radii[index] + fraction * (radii[index + 1] - radii[index]))
            )
    return np.asarray(dense_points), np.asarray(dense_radii)


def _stl_transition_point(solid, start, end, start_radius, end_radius):
    """Bisect an STL inside/outside transition and interpolate its radius."""
    low_point = np.asarray(start, dtype=float)
    high_point = np.asarray(end, dtype=float)
    low_radius = float(start_radius)
    high_radius = float(end_radius)
    low_inside = solid.contains(low_point)
    if low_inside == solid.contains(high_point):
        raise ValueError("STL transition endpoints must lie on opposite sides")
    for _ in range(max(1, int(GRAPH_CROP_BISECTION_ITERATIONS))):
        mid_point = 0.5 * (low_point + high_point)
        mid_radius = 0.5 * (low_radius + high_radius)
        if solid.contains(mid_point) == low_inside:
            low_point, low_radius = mid_point, mid_radius
        else:
            high_point, high_radius = mid_point, mid_radius
    return 0.5 * (low_point + high_point), 0.5 * (low_radius + high_radius)


def _clip_polyline_to_stl(points, radii, solid):
    """Return the single anatomically continuous in-solid edge fragment.

    A centreline can briefly leave and re-enter a faceted/booleaned lumen even
    though its Amira edge is one continuous vessel. Treating every parity
    transition as a real crop split manufactured two degree-one nodes at that
    excursion and disconnected the true downstream outlet. Only outside runs
    at the two *ends* of an Amira edge are crop authority; enclosed outside runs
    are bridged because the SpatialGraph is the vessel-topology ground truth.
    """
    points, radii = _densify_polyline(
        np.asarray(points, dtype=float),
        np.asarray(radii, dtype=float),
        GRAPH_CROP_SAMPLE_SPACING_MM,
    )
    inside = np.asarray(solid.contains(points), dtype=bool)
    retained = np.flatnonzero(inside)
    if len(retained) == 0:
        return []

    first = int(retained[0])
    last = int(retained[-1])
    fragment_points = [point.copy() for point in points[first:last + 1]]
    fragment_radii = [float(value) for value in radii[first:last + 1]]

    # Replace the first/last sampled interior points with bisection-accurate STL
    # crossings when an outside prefix/suffix was actually removed.
    if first > 0:
        boundary_point, boundary_radius = _stl_transition_point(
            solid,
            points[first - 1],
            points[first],
            radii[first - 1],
            radii[first],
        )
        fragment_points.insert(0, boundary_point)
        fragment_radii.insert(0, boundary_radius)
    if last < len(points) - 1:
        boundary_point, boundary_radius = _stl_transition_point(
            solid,
            points[last],
            points[last + 1],
            radii[last],
            radii[last + 1],
        )
        fragment_points.append(boundary_point)
        fragment_radii.append(boundary_radius)

    fragment_points = np.asarray(fragment_points, dtype=float)
    fragment_radii = np.asarray(fragment_radii, dtype=float)
    if _cumulative_distances(fragment_points)[-1] < float(
        GRAPH_CROP_MIN_FRAGMENT_LENGTH_MM
    ):
        return []
    return [(fragment_points, fragment_radii)]


def _crop_amira_network_to_stl(network, solid):
    """Clip Amira edges to the meshing STL and rebuild degree/topology data."""
    if not isinstance(network, _AmiraNetwork):
        return network
    if float(GRAPH_CROP_SAMPLE_SPACING_MM) <= 0.0:
        raise ValueError("GRAPH_CROP_SAMPLE_SPACING_MM must be positive")

    original_nodes = list(network.nodes)
    original_splines = list(network.splines)
    fragment_records = []
    removed_edges = 0
    cut_edges = 0
    total_edges = len(original_splines)
    progress_interval = max(1, total_edges // 20)
    _log("Cropping {:,} Amira graph edges to the meshing STL...".format(total_edges))
    for edge_ordinal, spline in enumerate(original_splines, start=1):
        fragments = _clip_polyline_to_stl(
            spline.points, spline.radii, solid
        )
        if not fragments:
            removed_edges += 1
            if edge_ordinal % progress_interval == 0 or edge_ordinal == total_edges:
                _log(
                    "Graph-edge crop: {:5.1f}% ({:,}/{:,}); {:,} fragment(s) "
                    "retained so far.".format(
                        100.0 * edge_ordinal / total_edges,
                        edge_ordinal,
                        total_edges,
                        len(fragment_records),
                    )
                )
            continue
        full_length = float(spline.cumulative[-1])
        retained_length = sum(
            float(_cumulative_distances(points)[-1]) for points, _radii in fragments
        )
        if len(fragments) != 1 or retained_length < full_length - 1.0e-5:
            cut_edges += 1
        for fragment_index, (points, radii) in enumerate(fragments):
            fragment_records.append({
                "source": spline,
                "fragment_index": fragment_index,
                "fragment_count": len(fragments),
                "points": points,
                "radii": radii,
            })
        if edge_ordinal % progress_interval == 0 or edge_ordinal == total_edges:
            _log(
                "Graph-edge crop: {:5.1f}% ({:,}/{:,}); {:,} fragment(s) "
                "retained so far.".format(
                    100.0 * edge_ordinal / total_edges,
                    edge_ordinal,
                    total_edges,
                    len(fragment_records),
                )
            )

    if not fragment_records:
        raise RuntimeError(
            "No Amira centreline fragments lie inside the meshing STL. Check "
            "STL/Amira units, affine transforms, and that the STL is watertight."
        )

    # Old adjacency must not leak into the clipped degree calculation.
    for node in original_nodes:
        node.splines = []
    next_node_id = max((node.node_id for node in original_nodes), default=-1) + 1
    next_edge_id = max((spline.edge_id for spline in original_splines), default=-1) + 1
    used_nodes = set()
    clipped_splines = []
    endpoint_tolerance = max(1.0e-7, float(GRAPH_CROP_SAMPLE_SPACING_MM) * 1.0e-4)

    def cropped_node(position):
        nonlocal next_node_id
        node = _AmiraNode(next_node_id, position)
        next_node_id += 1
        return node

    for record in fragment_records:
        source = record["source"]
        points = record["points"]
        start_is_original = (
            np.linalg.norm(points[0] - source.points[0]) <= endpoint_tolerance
            and solid.contains(source.points[0])
        )
        end_is_original = (
            np.linalg.norm(points[-1] - source.points[-1]) <= endpoint_tolerance
            and solid.contains(source.points[-1])
        )
        start_node = source.start_node if start_is_original else cropped_node(points[0])
        end_node = source.end_node if end_is_original else cropped_node(points[-1])
        used_nodes.update((start_node, end_node))
        if record["fragment_count"] == 1:
            edge_id = source.edge_id
        else:
            edge_id = next_edge_id
            next_edge_id += 1
        clipped_splines.append(_AmiraSpline(
            edge_id,
            start_node,
            end_node,
            points,
            record["radii"],
            source.strahler_order,
        ))

    # Cropped-away side branches can leave short ostial fragments geometrically
    # inside the lumen STL but disconnected from its actual centreline tree. This
    # mirrors flow_fractions.py's restrict-to-meshed-tree step by retaining only
    # substantial connected components.
    components = []
    unseen = set(used_nodes)
    while unseen:
        seed = unseen.pop()
        component_nodes = {seed}
        component_splines = set()
        stack = [seed]
        while stack:
            node = stack.pop()
            for spline in node.GetSplines():
                component_splines.add(spline)
                for adjacent in (spline.GetStartNode(), spline.GetEndNode()):
                    if adjacent not in component_nodes:
                        component_nodes.add(adjacent)
                        unseen.discard(adjacent)
                        stack.append(adjacent)
        components.append((component_nodes, component_splines))

    component_mode = str(STL_GRAPH_COMPONENT_MODE).lower()
    if component_mode not in {"relative", "largest", "all"}:
        raise ValueError(
            "STL_GRAPH_COMPONENT_MODE must be 'relative', 'largest', or 'all'"
        )
    largest_edge_count = max(len(edges) for _nodes, edges in components)
    if component_mode == "all":
        retained_components = components
    elif component_mode == "largest":
        retained_components = [max(components, key=lambda item: len(item[1]))]
    else:
        fraction = float(STL_GRAPH_MIN_COMPONENT_EDGE_FRACTION)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "STL_GRAPH_MIN_COMPONENT_EDGE_FRACTION must lie between 0 and 1"
            )
        minimum_edges = max(1, int(math.ceil(fraction * largest_edge_count)))
        retained_components = [
            item for item in components if len(item[1]) >= minimum_edges
        ]
    retained_nodes = set().union(*(nodes for nodes, _edges in retained_components))
    retained_splines = set().union(*(edges for _nodes, edges in retained_components))
    clipped = _AmiraNetwork(
        sorted(retained_nodes, key=lambda node: node.node_id),
        sorted(retained_splines, key=lambda spline: spline.edge_id),
        network.source_path,
    )
    _log(
        "STL crop: {} Amira edges -> {} retained edge fragment(s) in {} "
        "component(s); {} cut, {} fully outside, {} discarded disconnected "
        "fragment component(s), {} terminal node(s).".format(
            len(original_splines),
            len(clipped.splines),
            len(retained_components),
            cut_edges,
            removed_edges,
            len(components) - len(retained_components),
            sum(len(node.GetSplines()) == 1 for node in clipped.nodes),
        )
    )
    return clipped


def _validate_stl_matches_selected_surface(solid, surface_points):
    surface_lower = np.min(surface_points, axis=0)
    surface_upper = np.max(surface_points, axis=0)
    separated = np.any(solid.upper < surface_lower) or np.any(
        surface_upper < solid.lower
    )
    if separated:
        raise RuntimeError(
            "The meshing STL and selected Simpleware surface bounds do not overlap. "
            "Check STL_SCALE_TO_MM and STL_TO_SIMPLEWARE_AFFINE."
        )
    bound_difference = float(np.max(np.abs(np.concatenate((
        solid.lower - surface_lower,
        solid.upper - surface_upper,
    )))))
    _log(
        "STL/selected-surface maximum bound difference: {:.3f} mm.".format(
            bound_difference
        )
    )
    if bound_difference > float(STL_SELECTED_SURFACE_BOUNDS_TOLERANCE_MM):
        raise RuntimeError(
            "The selected Simpleware surface differs from the configured meshing "
            "STL by {:.3f} mm at its bounds (tolerance {:.3f} mm). Select the "
            "surface created from this STL, or verify the STL scale/affine."
            .format(
                bound_difference,
                float(STL_SELECTED_SURFACE_BOUNDS_TOLERANCE_MM),
            )
        )


def _averaged_inward_tangent(points, centre_index):
    """Average consistently oriented unit segments around a raw-data node."""
    requested_nodes = max(3, int(TANGENT_AVERAGING_NODE_COUNT))
    half_window = requested_nodes // 2
    first = max(0, centre_index - half_window)
    last = min(len(points) - 1, centre_index + half_window)

    # Expand a clipped window at the terminal side toward the interior.
    while last - first + 1 < requested_nodes and last < len(points) - 1:
        last += 1
    while last - first + 1 < requested_nodes and first > 0:
        first -= 1

    directions = []
    for delta in np.diff(points[first:last + 1], axis=0):
        length = float(np.linalg.norm(delta))
        if length > EPS:
            directions.append(delta / length)
    if not directions:
        raise RuntimeError("Cannot estimate a tangent from coincident centreline nodes")

    reference = points[last] - points[first]
    if np.linalg.norm(reference) <= EPS:
        reference = directions[0]
    reference = _unit(reference, "tangent reference")
    aligned = [direction if np.dot(direction, reference) >= 0.0 else -direction
               for direction in directions]
    return (
        _unit(np.mean(aligned, axis=0), "averaged centreline tangent"),
        first,
        last,
    )


def _terminal_record(node):
    """Describe a terminal using an inward raw-node offset and mean tangent."""
    attached = list(node.GetSplines())
    if len(attached) != 1:
        return None

    spline = attached[0]
    if spline.IsClosed():
        return None
    length = float(spline.GetLength(False))
    if length <= EPS:
        return None

    node_name = node.GetName()
    starts_here = spline.GetStartNode().GetName() == node_name
    terminal_position = _xyz(node.GetPosition(False))
    raw_points = _ordered_raw_points(spline, terminal_position)
    cumulative = _cumulative_distances(raw_points)
    if cumulative[-1] <= EPS:
        raise RuntimeError("Spline {!r} contains only coincident raw points".format(
            spline.GetName()
        ))

    inward_index = max(1, int(OUTLET_INWARD_NODE_COUNT))
    minimum_index = int(np.searchsorted(
        cumulative, float(OUTLET_MIN_INSET_MM), side="left"
    ))
    inward_index = max(inward_index, minimum_index)
    if inward_index >= len(raw_points) - 1:
        raise RuntimeError(
            "Terminal {!r} on spline {!r} has only {} usable inward raw nodes; "
            "need at least {} plus one interior node".format(
                node_name, spline.GetName(), len(raw_points) - 1, inward_index
            )
        )

    distance_from_terminal = length * cumulative[inward_index] / cumulative[-1]
    distance = distance_from_terminal if starts_here else length - distance_from_terminal
    parameter = float(spline.GetParameterAtDistance(float(distance), False))
    centre = _xyz(spline.GetPosition(parameter, False))
    inward_tangent, tangent_first, tangent_last = _averaged_inward_tangent(
        raw_points, inward_index
    )
    outward = -inward_tangent

    return {
        "node": node,
        "node_name": node_name,
        "spline": spline,
        "terminal_position": terminal_position,
        "centre": centre,
        "normal": _unit(outward, "terminal tangent"),
        "length": length,
        "distance_from_start": float(distance),
        "inward_node_count": int(inward_index),
        "tangent_node_count": int(tangent_last - tangent_first + 1),
        "radius_probe_points": raw_points[tangent_first:tangent_last + 1],
        "terminal_raw_points": raw_points,
        "terminal_raw_cumulative": cumulative,
    }


def _find_terminals(network, surface_tree, source_name=None):
    terminals = []
    degree_counts = {}
    skipped = []
    for node in network.GetNodes():
        degree = len(list(node.GetSplines()))
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
        try:
            record = _terminal_record(node)
        except RuntimeError as exc:
            if degree != 1 or not SKIP_UNSAFE_SHORT_TERMINALS:
                raise
            skipped.append((node.GetName(), str(exc)))
            continue
        if record is None:
            continue
        if (
            hasattr(record["spline"], "MedianRadiusNearDistance")
            and not getattr(record["spline"], "use_surface_radius", False)
        ):
            record["radius"] = record["spline"].MedianRadiusNearDistance(
                record["distance_from_start"], record["tangent_node_count"]
            )
            record["radius_source"] = "amira"
        else:
            radii = surface_tree.query(record["radius_probe_points"], k=1)[0]
            record["radius"] = float(np.median(radii))
            record["radius_source"] = "surface"
        record["centreline_source"] = (
            str(source_name).lower() if source_name else "unknown"
        )
        terminals.append(record)
    for node_name, reason in skipped:
        _log(
            "WARNING: skipping unsafe terminal {!r}; no boundary plane will be "
            "created there. {}".format(node_name, reason)
        )
    if skipped:
        _log(
            "Skipped {} of {} degree-one node(s) because their terminal branch "
            "cannot satisfy the configured inward placement.".format(
                len(skipped), degree_counts.get(1, 0)
            )
        )
    if not terminals:
        raise RuntimeError(
            "The selected centreline network has no usable degree-one nodes. "
            "Node-degree counts: {}. Verify the configured boundary centreline "
            "network and its terminal-node topology."
            .format(dict(sorted(degree_counts.items())))
        )
    return terminals


def _merge_boundary_terminals(terminals_by_source):
    """Merge terminal sets in source-preference order without losing branches."""
    tolerance = float(BOUNDARY_TERMINAL_MERGE_DISTANCE_MM)
    if tolerance < 0.0:
        raise ValueError("BOUNDARY_TERMINAL_MERGE_DISTANCE_MM cannot be negative")

    merged = []
    duplicate_counts = {}
    for source_name, records in terminals_by_source:
        duplicate_count = 0
        for record in records:
            # Do not merge two terminals found by the same source: closely
            # adjacent outlet branches are still distinct. Only suppress a
            # lower-priority source when a preferred source already represents
            # the same endpoint.
            preferred = [
                item for item in merged
                if item["centreline_source"] != record["centreline_source"]
            ]
            if preferred:
                distances = [
                    float(np.linalg.norm(
                        item["terminal_position"] - record["terminal_position"]
                    ))
                    for item in preferred
                ]
                if min(distances) <= tolerance:
                    duplicate_count += 1
                    continue
            merged.append(record)
        duplicate_counts[source_name] = duplicate_count

    _log(
        "Merged boundary terminal sources: {} unique terminal(s); suppressed "
        "cross-source duplicates {} at <= {:.3f} mm.".format(
            len(merged), duplicate_counts, tolerance
        )
    )
    return merged


def _principal_axis_and_ratio(covariance, seed):
    """Return the dominant 3-D covariance axis without a LAPACK dependency."""
    covariance = np.asarray(covariance, dtype=float)
    vector = _unit(seed, "principal-axis seed")
    for _ in range(32):
        updated = np.sum(covariance * vector[None, :], axis=1)
        if np.linalg.norm(updated) <= EPS:
            raise RuntimeError("Local surface covariance has zero extent")
        vector = _unit(updated, "principal-axis iteration")
    covariance_vector = np.sum(covariance * vector[None, :], axis=1)
    largest = float(np.sum(vector * covariance_vector))

    basis = np.eye(3)[int(np.argmin(np.abs(vector)))]
    second_vector = _unit(
        basis - float(np.sum(basis * vector)) * vector,
        "secondary-axis seed",
    )
    deflated = covariance - largest * np.outer(vector, vector)
    for _ in range(32):
        updated = np.sum(deflated * second_vector[None, :], axis=1)
        updated -= float(np.sum(updated * vector)) * vector
        if np.linalg.norm(updated) <= EPS:
            break
        second_vector = _unit(updated, "secondary-axis iteration")
    covariance_second = np.sum(covariance * second_vector[None, :], axis=1)
    second = abs(float(np.sum(second_vector * covariance_second)))
    return vector, largest / max(second, EPS)


def _refine_plane_normal_from_surface(record, surface_points, surface_tree):
    """Fit the local lumen-wall principal axis and use it as plane normal."""
    record["surface_axis_used"] = False
    if not REFINE_PLANE_NORMAL_FROM_SURFACE:
        return

    radius = max(float(record["radius"]), float(MIN_PLAUSIBLE_RADIUS_MM))
    axial_half_length = float(SURFACE_AXIS_AXIAL_RADIUS_FACTOR) * radius
    radial_limit = float(SURFACE_AXIS_RADIAL_RADIUS_FACTOR) * radius
    search_radius = max(
        float(SURFACE_AXIS_MIN_SEARCH_RADIUS_MM),
        math.hypot(axial_half_length, radial_limit),
    )
    indices = surface_tree.query_ball_point(record["centre"], search_radius)
    if len(indices) < int(SURFACE_AXIS_MIN_POINTS):
        record["surface_axis_reason"] = "too_few_nearby_vertices"
        return

    nearby = np.asarray(surface_points[np.asarray(indices, dtype=int)], dtype=float)
    delta = nearby - record["centre"]
    initial = _unit(record["normal"], "initial terminal normal")
    axial = np.sum(delta * initial[None, :], axis=1)
    radial = np.linalg.norm(delta - axial[:, None] * initial, axis=1)
    selected = nearby[
        (np.abs(axial) <= axial_half_length) & (radial <= radial_limit)
    ]
    if len(selected) < int(SURFACE_AXIS_MIN_POINTS):
        record["surface_axis_reason"] = "too_few_cylindrical_vertices"
        return

    centred = selected - np.mean(selected, axis=0)
    covariance = np.sum(
        centred[:, :, None] * centred[:, None, :], axis=0
    ) / max(1, len(selected) - 1)
    fitted, eigenvalue_ratio = _principal_axis_and_ratio(covariance, initial)
    if eigenvalue_ratio < float(SURFACE_AXIS_MIN_EIGENVALUE_RATIO):
        record["surface_axis_reason"] = "ambiguous_local_axis"
        record["surface_axis_eigenvalue_ratio"] = eigenvalue_ratio
        return

    if np.sum(fitted * initial) < 0.0:
        fitted = -fitted
    cosine = float(np.clip(np.sum(fitted * initial), -1.0, 1.0))
    correction_degrees = math.degrees(math.acos(cosine))
    if correction_degrees > float(SURFACE_AXIS_MAX_CORRECTION_DEGREES):
        record["surface_axis_reason"] = "correction_exceeds_limit"
        record["surface_axis_correction_degrees"] = correction_degrees
        return

    record["normal"] = fitted
    record["surface_axis_used"] = True
    record["surface_axis_point_count"] = int(len(selected))
    record["surface_axis_eigenvalue_ratio"] = eigenvalue_ratio
    record["surface_axis_correction_degrees"] = correction_degrees


def _force_outward_plane_normal(record):
    """Orient the plane normal from its inset centre toward its own terminal."""
    # Exact STL validation may recenter the finite square tangentially on its
    # measured loop.  Use the original on-centreline point for the direction
    # check because tangential recentering does not change the plane itself.
    direction_centre = record.get("centreline_plane_centre", record["centre"])
    toward_terminal = np.asarray(record["terminal_position"], dtype=float) - np.asarray(
        direction_centre, dtype=float
    )
    separation = float(np.linalg.norm(toward_terminal))
    if separation <= EPS:
        raise RuntimeError(
            "Plane centre for terminal {!r} coincides with the terminal".format(
                record["node_name"]
            )
        )
    normal = _unit(record["normal"], "terminal plane normal")
    projection = float(np.dot(normal, toward_terminal))
    record["normal_was_flipped"] = projection < 0.0
    if projection < 0.0:
        normal = -normal
        projection = -projection
        # Preserve a right-handed (X, Y, normal) rectangle basis when the
        # outward-direction guard reverses the normal.
        if "plane_second_axis" in record:
            record["plane_second_axis"] = -np.asarray(
                record["plane_second_axis"], dtype=float
            )
    record["normal"] = normal
    record["normal_outward_cosine"] = projection / separation
    if record["normal_outward_cosine"] <= EPS:
        raise RuntimeError(
            "Plane normal for terminal {!r} is transverse to the terminal direction"
            .format(record["node_name"])
        )


def _plane_basis(normal):
    """Return a stable right-handed orthonormal basis in a plane."""
    normal = _unit(normal, "plane normal")
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = _unit(seed - np.dot(seed, normal) * normal, "plane basis")
    second = _unit(np.cross(normal, first), "plane basis")
    return first, second


def _polyline_position(points, cumulative, distance):
    """Interpolate a polyline at arc distance, clamped to its ends."""
    points = np.asarray(points, dtype=float)
    cumulative = np.asarray(cumulative, dtype=float)
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    upper = int(np.searchsorted(cumulative, distance, side="right"))
    upper = min(max(1, upper), len(cumulative) - 1)
    lower = upper - 1
    span = float(cumulative[upper] - cumulative[lower])
    fraction = 0.0 if span <= EPS else (distance - cumulative[lower]) / span
    return points[lower] + fraction * (points[upper] - points[lower])


def _local_averaged_outward_tangent(points, cumulative, distance, window):
    """Average the outward tangent around an arc position on an outlet path.

    Terminal paths are ordered from the surface outlet inward, so the vector
    from the larger arc position to the smaller one points outward.
    """
    total = float(cumulative[-1])
    half_window = 0.5 * min(float(window), total)
    outward_distance = max(0.0, float(distance) - half_window)
    inward_distance = min(total, float(distance) + half_window)
    if inward_distance - outward_distance <= EPS:
        raise RuntimeError("terminal path is too short to average its local tangent")
    outward_point = _polyline_position(points, cumulative, outward_distance)
    inward_point = _polyline_position(points, cumulative, inward_distance)
    return _unit(outward_point - inward_point, "local averaged Amira tangent")


def _terminal_surface_anchor(record, solid):
    """Locate the actual STL cap/exit along the terminal's outward tangent."""
    raw = np.asarray(record["terminal_raw_points"], dtype=float)
    if len(raw) < 2:
        raise RuntimeError("Terminal branch has fewer than two points")
    terminal = raw[0]
    # Use the same local arc-length chord that defines the final plane normal.
    # This remains tied to the Amira centreline even when the graph endpoint and
    # the active Simpleware surface terminate at slightly different positions.
    raw_cumulative = _cumulative_distances(raw)
    tangent_window = min(
        raw_cumulative[-1], float(STL_PLANE_TANGENT_AVERAGING_LENGTH_MM)
    )
    tangent_inward = _polyline_position(raw, raw_cumulative, tangent_window)
    outward = _unit(
        raw[0] - tangent_inward, "averaged terminal Amira tangent"
    )

    # Cropped graph fragments already terminate on the STL.  An original Amira
    # leaf can stop a little inside its generated cap, so extend it until the
    # first inside/outside transition and use that surface point as the anchor.
    probe_inside = bool(solid.contains(terminal - 0.01 * outward))
    terminal_inside = bool(solid.contains(terminal))
    if not terminal_inside and probe_inside:
        return terminal.copy(), outward, 0.0

    max_extension = max(
        float(STL_PLANE_MAX_TERMINAL_EXTENSION_MM),
        float(STL_PLANE_TERMINAL_EXTENSION_RADIUS_FACTOR)
        * max(float(record["radius"]), float(MIN_PLAUSIBLE_RADIUS_MM)),
    )
    step = min(0.05, max_extension / 20.0)
    previous = terminal.copy()
    previous_inside = terminal_inside or probe_inside
    if not previous_inside:
        # The crop bisection can classify its final floating-point midpoint as
        # outside.  The immediately inward raw point is the reliable bracket.
        previous = raw[1].copy()
        previous_inside = bool(solid.contains(previous))
    if not previous_inside:
        return terminal.copy(), outward, 0.0

    distance = step
    while distance <= max_extension + EPS:
        current = terminal + distance * outward
        if not solid.contains(current):
            anchor, _radius = _stl_transition_point(
                solid, previous, current, record["radius"], record["radius"]
            )
            return anchor, outward, float(np.linalg.norm(anchor - terminal))
        previous = current
        distance += step
    # A strongly curved cap may not lie on the extrapolated ray.  In that case
    # retain the graph tip; the exact loop search below remains the authority.
    return terminal.copy(), outward, 0.0


def _triangle_plane_segments(triangles, centre, normal):
    """Intersect triangles with an infinite plane and return line segments."""
    triangles = np.asarray(triangles, dtype=float)
    if len(triangles) == 0:
        return np.empty((0, 2, 3), dtype=float)
    centre = np.asarray(centre, dtype=float)
    normal = _unit(normal, "plane normal")
    signed = np.sum((triangles - centre[None, None, :]) * normal, axis=2)
    tolerance = 1.0e-9
    crossing = (np.min(signed, axis=1) <= tolerance) & (
        np.max(signed, axis=1) >= -tolerance
    )
    triangles = triangles[crossing]
    signed = signed[crossing]
    segments = []
    for triangle, distances in zip(triangles, signed):
        hits = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            d0, d1 = float(distances[first]), float(distances[second])
            p0, p1 = triangle[first], triangle[second]
            if abs(d0) <= tolerance:
                hits.append(p0)
            if d0 * d1 < -tolerance * tolerance:
                fraction = d0 / (d0 - d1)
                hits.append(p0 + fraction * (p1 - p0))
        if len(hits) < 2:
            continue
        unique = []
        for point in hits:
            if not any(np.linalg.norm(point - item) <= tolerance for item in unique):
                unique.append(point)
        if len(unique) < 2:
            continue
        if len(unique) == 2:
            segments.append(unique)
            continue
        # Degenerate vertex/coplanar cases: retain the farthest pair.
        best = max(
            ((np.linalg.norm(a - b), a, b) for i, a in enumerate(unique)
             for b in unique[i + 1:]),
            key=lambda item: item[0],
        )
        segments.append((best[1], best[2]))
    return np.asarray(segments, dtype=float).reshape(-1, 2, 3)


def _ordered_plane_intersection_loops(segments, centre, normal):
    """Stitch plane/triangle segments and return closed ordered 2-D loops."""
    if len(segments) == 0:
        return [], []
    first_axis, second_axis = _plane_basis(normal)
    flat = np.asarray(segments, dtype=float).reshape(-1, 3)
    relative = flat - np.asarray(centre, dtype=float)
    uv = np.column_stack((relative.dot(first_axis), relative.dot(second_axis)))
    tolerance = max(float(STL_PLANE_INTERSECTION_STITCH_TOLERANCE_MM), EPS)
    keys = np.round(uv / tolerance).astype(np.int64)
    node_for_key = {}
    node_uv = []
    segment_nodes = []
    for segment_index in range(len(segments)):
        pair = []
        for endpoint in (2 * segment_index, 2 * segment_index + 1):
            key = tuple(int(value) for value in keys[endpoint])
            if key not in node_for_key:
                node_for_key[key] = len(node_uv)
                node_uv.append(uv[endpoint])
            pair.append(node_for_key[key])
        if pair[0] != pair[1]:
            segment_nodes.append(tuple(pair))

    adjacency = defaultdict(set)
    for first, second in segment_nodes:
        adjacency[first].add(second)
        adjacency[second].add(first)

    loops = []
    open_components = []
    unseen = set(adjacency)
    while unseen:
        seed = unseen.pop()
        component = {seed}
        stack = [seed]
        while stack:
            node = stack.pop()
            for adjacent in adjacency[node]:
                if adjacent not in component:
                    component.add(adjacent)
                    unseen.discard(adjacent)
                    stack.append(adjacent)
        if not all(len(adjacency[node]) == 2 for node in component):
            open_components.append(np.asarray([node_uv[node] for node in component]))
            continue
        ordered = [seed]
        previous = None
        current = seed
        while True:
            choices = [node for node in adjacency[current] if node != previous]
            following = choices[0]
            if following == seed:
                break
            if following in ordered:
                ordered = []
                break
            ordered.append(following)
            previous, current = current, following
        if len(ordered) >= 3:
            loops.append(np.asarray([node_uv[node] for node in ordered]))
    return loops, open_components


def _point_in_polygon(point, polygon):
    """Even/odd 2-D point-in-polygon test."""
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        if ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / (y1 - y0) + x0
        ):
            inside = not inside
        previous = current
    return inside


def _polyline_intersects_rectangle(points, half_x, half_y):
    """Return whether a closed/open 2-D polyline touches an axis-aligned box."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return False
    if np.any(
        (np.abs(points[:, 0]) <= float(half_x) + EPS)
        & (np.abs(points[:, 1]) <= float(half_y) + EPS)
    ):
        return True
    # Liang-Barsky segment/rectangle clipping also catches a segment whose two
    # endpoints lie outside opposite sides of the finite ROI.
    lower = np.array([-float(half_x), -float(half_y)])
    upper = -lower
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        t0, t1 = 0.0, 1.0
        for axis in range(2):
            if abs(float(delta[axis])) <= EPS:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    t0, t1 = 1.0, 0.0
                    break
                continue
            enter = (lower[axis] - start[axis]) / delta[axis]
            leave = (upper[axis] - start[axis]) / delta[axis]
            if enter > leave:
                enter, leave = leave, enter
            t0 = max(t0, float(enter))
            t1 = min(t1, float(leave))
            if t0 > t1:
                break
        if t0 <= t1:
            return True
    return False


def _validate_stl_plane_intersection(solid, centre, normal, graph_radius):
    """Require exactly one complete STL loop around the plane centre."""
    search_radius = max(
        float(STL_PLANE_NEIGHBOUR_SEARCH_MIN_RADIUS_MM),
        8.0 * float(graph_radius),
    )
    triangles = solid.nearby_triangles(centre, search_radius)
    segments = _triangle_plane_segments(triangles, centre, normal)
    loops, open_components = _ordered_plane_intersection_loops(
        segments, centre, normal
    )
    containing = [loop for loop in loops if _point_in_polygon((0.0, 0.0), loop)]
    if len(containing) != 1:
        return None, "expected one closed loop around centre; found {}".format(
            len(containing)
        )
    target_index = next(
        index for index, loop in enumerate(loops) if loop is containing[0]
    )
    target = loops[target_index]
    shifted_target = np.roll(target, -1, axis=0)
    cross = (
        target[:, 0] * shifted_target[:, 1]
        - shifted_target[:, 0] * target[:, 1]
    )
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) <= EPS:
        return None, "surface intersection loop has zero area"
    polygon_centre = np.array([
        np.sum((target[:, 0] + shifted_target[:, 0]) * cross),
        np.sum((target[:, 1] + shifted_target[:, 1]) * cross),
    ]) / (6.0 * signed_area)
    target = target - polygon_centre[None, :]
    loops = [loop - polygon_centre[None, :] for loop in loops]
    target = loops[target_index]
    open_components = [
        component - polygon_centre[None, :] for component in open_components
    ]
    radial = np.linalg.norm(target, axis=1)
    loop_radius = float(np.max(radial))

    # Fit an in-plane oriented rectangle rather than a square. Closely adjacent
    # outlet loops can be isolated by a narrow rectangle even when the corner of
    # a covering square would cut the neighbour. Every candidate includes the
    # configured facet margin and is tested against every other exact surface
    # intersection.
    others = [
        loop for index, loop in enumerate(loops) if index != target_index
    ] + open_components
    rectangle_candidates = []
    minimum_half_extent = 0.5 * float(OUTLET_PLANE_MIN_DIAMETER_MM)
    for degrees in np.arange(0.0, 180.0, 2.5):
        angle = math.radians(float(degrees))
        rotation = np.array([
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ])
        rotated_target = target.dot(rotation.T)
        half_x = max(
            minimum_half_extent,
            float(np.max(np.abs(rotated_target[:, 0])))
            * float(STL_PLANE_LOOP_MARGIN_FACTOR)
            + float(STL_PLANE_EDGE_PADDING_MM),
        )
        half_y = max(
            minimum_half_extent,
            float(np.max(np.abs(rotated_target[:, 1])))
            * float(STL_PLANE_LOOP_MARGIN_FACTOR)
            + float(STL_PLANE_EDGE_PADDING_MM),
        )
        collides = False
        for component in others:
            rotated_component = component.dot(rotation.T)
            if _polyline_intersects_rectangle(
                rotated_component, half_x, half_y
            ):
                collides = True
                break
        if not collides:
            rectangle_candidates.append((half_x * half_y, angle, half_x, half_y))
    if not rectangle_candidates:
        return None, "no covering finite rectangle avoids additional surface intersections"
    _rectangle_area, rectangle_angle, half_x, half_y = min(
        rectangle_candidates, key=lambda item: item[0]
    )
    shifted = np.roll(target, -1, axis=0)
    area = 0.5 * abs(float(np.sum(
        target[:, 0] * shifted[:, 1] - shifted[:, 0] * target[:, 1]
    )))
    if area <= EPS:
        return None, "surface intersection loop has zero area"
    first_axis, second_axis = _plane_basis(normal)
    rectangle_first = (
        math.cos(rectangle_angle) * first_axis
        + math.sin(rectangle_angle) * second_axis
    )
    rectangle_second = (
        -math.sin(rectangle_angle) * first_axis
        + math.cos(rectangle_angle) * second_axis
    )
    return {
        "surface_loop_radius": loop_radius,
        "surface_loop_area": area,
        "surface_loop_equivalent_radius": math.sqrt(area / math.pi),
        "surface_loop_points": int(len(target)),
        "plane_half_width": max(half_x, half_y),
        "plane_half_extent_x": half_x,
        "plane_half_extent_y": half_y,
        "plane_first_axis": rectangle_first,
        "plane_second_axis": rectangle_second,
        "surface_intersection_loop_count": len(loops),
        "surface_loop_centre_offset": (
            polygon_centre[0] * first_axis + polygon_centre[1] * second_axis
        ),
    }, None


def _fit_stl_terminal_cap(solid, anchor, outward_seed, graph_radius):
    """Fit a coherent planar STL outlet cap close to a terminal anchor.

    Only triangles whose normals already agree with the Amira terminal tangent
    are considered.  A second, much tighter normal-coherence and coplanarity
    test prevents a curved lumen wall or an adjacent branch from becoming the
    plane authority.
    """
    anchor = np.asarray(anchor, dtype=float)
    outward_seed = _unit(outward_seed, "terminal outward tangent")
    radius = max(float(graph_radius), float(MIN_PLAUSIBLE_RADIUS_MM))
    search_radius = max(
        float(STL_CAP_SEARCH_MIN_RADIUS_MM),
        float(STL_CAP_SEARCH_RADIUS_FACTOR) * radius,
    )
    triangles = solid.nearby_triangles(anchor, search_radius)
    if len(triangles) == 0:
        return None, "no nearby STL triangles"

    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > EPS
    if not np.any(valid):
        return None, "nearby STL triangles have zero area"
    triangles = triangles[valid]
    cross = cross[valid]
    double_area = double_area[valid]
    normals = cross / double_area[:, None]
    alignment = normals.dot(outward_seed)
    normals[alignment < 0.0] *= -1.0
    alignment = np.abs(alignment)
    centres = np.mean(triangles, axis=1)
    delta = centres - anchor
    axial = np.abs(delta.dot(outward_seed))
    radial = np.linalg.norm(
        delta - delta.dot(outward_seed)[:, None] * outward_seed[None, :],
        axis=1,
    )
    graph_cosine = math.cos(math.radians(
        float(STL_CAP_NORMAL_MAX_GRAPH_DEVIATION_DEGREES)
    ))
    candidates = (
        (alignment >= graph_cosine)
        & (axial <= max(0.10, 0.75 * radius))
        & (radial <= max(0.20, 1.75 * radius))
    )
    if np.count_nonzero(candidates) < 2:
        return None, "no cap-normal triangle cluster near the terminal anchor"

    candidate_normals = normals[candidates]
    candidate_weights = double_area[candidates]
    mean_normal = _unit(
        np.sum(candidate_normals * candidate_weights[:, None], axis=0),
        "area-weighted cap normal",
    )
    if np.dot(mean_normal, outward_seed) < 0.0:
        mean_normal = -mean_normal
    coherence_cosine = math.cos(math.radians(
        float(STL_CAP_NORMAL_COHERENCE_DEGREES)
    ))
    coherent = candidates & (normals.dot(mean_normal) >= coherence_cosine)
    if np.count_nonzero(coherent) < 2:
        return None, "candidate cap triangles are not normal-coherent"

    # A true generated/cropped end cap is coplanar with the surface anchor.
    # Test every vertex, not merely triangle centroids, so a curved wall cannot
    # pass by averaging opposite residuals.
    planar_limit = max(
        float(STL_CAP_MAX_PLANAR_ERROR_MM),
        float(STL_CAP_MAX_PLANAR_ERROR_RADIUS_FACTOR) * radius,
    )
    vertex_residual = np.max(np.abs(
        np.sum(
            (triangles - anchor[None, None, :])
            * mean_normal[None, None, :],
            axis=2,
        )
    ), axis=1)
    anchored_coherent = coherent & (vertex_residual <= planar_limit)
    fitted_plane_offset = 0.0
    if np.count_nonzero(anchored_coherent) >= 2:
        coherent = anchored_coherent
    else:
        # The graph-ray/surface parity intersection can lie just off the flat
        # cap when the Amira endpoint is eccentric or the cap is oblique. Find
        # a nearby parallel plane cluster, but retain the same tangent, radial,
        # normal-coherence, area, and centre-offset constraints.
        minimum_area = (
            float(STL_CAP_MIN_AREA_RADIUS_FACTOR) * math.pi * radius * radius
        )
        coherent_indices = np.flatnonzero(coherent)
        centre_offsets = (centres - anchor).dot(mean_normal)
        clusters = []
        for seed_index in coherent_indices:
            seed_offset = float(centre_offsets[seed_index])
            cluster = coherent & (
                np.abs(centre_offsets - seed_offset) <= planar_limit
            )
            cluster_weights = double_area[cluster]
            if len(cluster_weights) < 2:
                continue
            plane_offset = float(np.average(
                centre_offsets[cluster], weights=cluster_weights
            ))
            residual = np.max(np.abs(
                np.sum(
                    (triangles - anchor[None, None, :])
                    * mean_normal[None, None, :],
                    axis=2,
                ) - plane_offset
            ), axis=1)
            cluster &= residual <= planar_limit
            cluster_weights = double_area[cluster]
            area = 0.5 * float(np.sum(cluster_weights))
            if np.count_nonzero(cluster) < 2 or area < minimum_area:
                continue
            cluster_centre = np.average(
                centres[cluster], axis=0, weights=cluster_weights
            )
            radial_offset = np.linalg.norm(
                (cluster_centre - anchor)
                - np.dot(cluster_centre - anchor, mean_normal) * mean_normal
            )
            scale = max(0.05, radius)
            score = abs(plane_offset) / scale + radial_offset / scale
            clusters.append((score, cluster, plane_offset, residual))
        if not clusters:
            return None, "normal-coherent triangles are not planar at the outlet"
        _score, coherent, fitted_plane_offset, vertex_residual = min(
            clusters, key=lambda item: item[0]
        )

    weights = double_area[coherent]
    fitted_normals = normals[coherent]
    mean_normal = _unit(
        np.sum(fitted_normals * weights[:, None], axis=0),
        "fitted STL cap normal",
    )
    if np.dot(mean_normal, outward_seed) < 0.0:
        mean_normal = -mean_normal
    fitted_centres = centres[coherent]
    cap_centre = np.sum(fitted_centres * weights[:, None], axis=0) / np.sum(weights)
    # Keep the fitted centre on the plane through the surface anchor.  The
    # in-plane component recentres an off-axis Amira endpoint on its STL cap.
    cap_centre -= (
        np.dot(cap_centre - anchor, mean_normal) - fitted_plane_offset
    ) * mean_normal
    total_area = 0.5 * float(np.sum(weights))
    minimum_area = (
        float(STL_CAP_MIN_AREA_RADIUS_FACTOR) * math.pi * radius * radius
    )
    if total_area < minimum_area:
        return None, "planar cap triangle area is too small"
    centre_offset = float(np.linalg.norm(cap_centre - anchor))
    if centre_offset > max(0.20, 1.25 * radius):
        return None, "fitted cap centre is too far from the Amira surface anchor"
    deviation = math.degrees(math.acos(float(np.clip(
        np.dot(mean_normal, outward_seed), -1.0, 1.0
    ))))
    return {
        "centre": cap_centre,
        "normal": mean_normal,
        "area": total_area,
        "triangle_count": int(np.count_nonzero(coherent)),
        "planarity_error": float(np.max(vertex_residual[coherent])),
        "graph_deviation_degrees": deviation,
        "anchor_offset_mm": centre_offset,
        "anchor_plane_offset_mm": float(fitted_plane_offset),
    }, None


def _fit_stl_terminal_cap_clustered(solid, anchor, outward_seed, graph_radius):
    """Find the dominant coherent planar cap without averaging wall normals."""
    if not hasattr(solid, "nearby_triangles"):
        return None, "STL triangle neighbourhoods are unavailable"
    anchor = np.asarray(anchor, dtype=float)
    outward_seed = _unit(outward_seed, "terminal outward tangent")
    radius = max(float(graph_radius), float(MIN_PLAUSIBLE_RADIUS_MM))
    triangles = solid.nearby_triangles(
        anchor,
        max(
            0.50,
            float(STL_CAP_SEARCH_RADIUS_FACTOR) * radius,
        ),
    )
    if len(triangles) == 0:
        return None, "no nearby STL triangles"
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > EPS
    triangles = triangles[valid]
    cross = cross[valid]
    double_area = double_area[valid]
    if len(triangles) == 0:
        return None, "nearby STL triangles have zero area"
    normals = cross / double_area[:, None]
    signed_alignment = normals.dot(outward_seed)
    normals[signed_alignment < 0.0] *= -1.0
    alignment = np.abs(signed_alignment)
    centres = np.mean(triangles, axis=1)
    delta = centres - anchor
    axial_component = delta.dot(outward_seed)
    axial = np.abs(axial_component)
    radial = np.linalg.norm(
        delta - axial_component[:, None] * outward_seed[None, :], axis=1
    )
    graph_cosine = math.cos(math.radians(
        float(STL_CAP_NORMAL_MAX_GRAPH_DEVIATION_DEGREES)
    ))
    candidates = (
        (alignment >= graph_cosine)
        & (axial <= max(0.20, 1.50 * radius))
        & (radial <= max(0.30, 2.50 * radius))
    )
    candidate_indices = np.flatnonzero(candidates)
    if len(candidate_indices) < 2:
        return None, "no cap-normal triangle cluster near the terminal anchor"

    coherence_cosine = math.cos(math.radians(
        float(STL_CAP_NORMAL_COHERENCE_DEGREES)
    ))
    planar_limit = max(
        float(STL_CAP_MAX_PLANAR_ERROR_MM),
        float(STL_CAP_MAX_PLANAR_ERROR_RADIUS_FACTOR) * radius,
    )
    minimum_area = (
        float(STL_CAP_MIN_AREA_RADIUS_FACTOR) * math.pi * radius * radius
    )
    # Test one representative of each small normal bin. A flat cap contributes
    # many identical normals, whereas iterating every facet adds no information.
    seed_indices = {}
    for index in candidate_indices:
        key = tuple(np.round(normals[index] / 0.02).astype(int))
        seed_indices.setdefault(key, int(index))

    solutions = []
    for seed_index in seed_indices.values():
        coherent = candidates & (
            normals.dot(normals[seed_index]) >= coherence_cosine
        )
        weights = double_area[coherent]
        if len(weights) < 2:
            continue
        mean_normal = _unit(
            np.sum(normals[coherent] * weights[:, None], axis=0),
            "clustered cap normal",
        )
        if np.dot(mean_normal, outward_seed) < 0.0:
            mean_normal = -mean_normal
        deviation = math.degrees(math.acos(float(np.clip(
            np.dot(mean_normal, outward_seed), -1.0, 1.0
        ))))
        if deviation > float(STL_CAP_NORMAL_MAX_GRAPH_DEVIATION_DEGREES) + EPS:
            continue
        coherent = candidates & (normals.dot(mean_normal) >= coherence_cosine)
        centre_offsets = (centres - anchor).dot(mean_normal)
        offset_seeds = [0.0]
        quantised_offsets = {}
        for index in np.flatnonzero(coherent):
            key = int(round(float(centre_offsets[index]) / planar_limit))
            quantised_offsets.setdefault(key, float(centre_offsets[index]))
        offset_seeds.extend(quantised_offsets.values())
        for seed_offset in offset_seeds:
            cluster = coherent & (
                np.abs(centre_offsets - float(seed_offset)) <= planar_limit
            )
            cluster_weights = double_area[cluster]
            if len(cluster_weights) < 2:
                continue
            plane_offset = float(np.average(
                centre_offsets[cluster], weights=cluster_weights
            ))
            residual = np.max(np.abs(
                np.sum(
                    (triangles - anchor[None, None, :])
                    * mean_normal[None, None, :],
                    axis=2,
                ) - plane_offset
            ), axis=1)
            cluster &= residual <= planar_limit
            cluster_weights = double_area[cluster]
            total_area = 0.5 * float(np.sum(cluster_weights))
            if np.count_nonzero(cluster) < 2 or total_area < minimum_area:
                continue
            # Refit once using only the accepted planar cluster.
            fitted_normal = _unit(
                np.sum(
                    normals[cluster] * cluster_weights[:, None], axis=0
                ),
                "fitted clustered STL cap normal",
            )
            if np.dot(fitted_normal, outward_seed) < 0.0:
                fitted_normal = -fitted_normal
            fitted_centres = centres[cluster]
            cap_centre = np.average(
                fitted_centres, axis=0, weights=cluster_weights
            )
            cap_centre -= (
                np.dot(cap_centre - anchor, fitted_normal) - plane_offset
            ) * fitted_normal
            centre_offset = float(np.linalg.norm(cap_centre - anchor))
            if centre_offset > max(0.20, 1.25 * radius):
                continue
            scale = max(0.05, radius)
            score = total_area / (
                1.0
                + abs(plane_offset) / scale
                + centre_offset / scale
            )
            fitted_deviation = math.degrees(math.acos(float(np.clip(
                np.dot(fitted_normal, outward_seed), -1.0, 1.0
            ))))
            solutions.append((score, {
                "centre": cap_centre,
                "normal": fitted_normal,
                "area": total_area,
                "triangle_count": int(np.count_nonzero(cluster)),
                "planarity_error": float(np.max(residual[cluster])),
                "graph_deviation_degrees": fitted_deviation,
                "anchor_offset_mm": centre_offset,
                "anchor_plane_offset_mm": plane_offset,
            }))
    if not solutions:
        return None, "no coherent planar STL cap cluster passed validation"
    return max(solutions, key=lambda item: item[0])[1], None


def _terminal_plane_inset_schedule(branch_length, radius):
    """Return preferred inset candidates without consuming a short branch."""
    branch_length = float(branch_length)
    radius = max(float(radius), float(MIN_PLAUSIBLE_RADIUS_MM))
    if branch_length <= EPS:
        raise RuntimeError("terminal branch has zero retained length")
    fraction = float(STL_PLANE_MAX_TERMINAL_BRANCH_FRACTION)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "STL_PLANE_MAX_TERMINAL_BRANCH_FRACTION must be between 0 and 1"
        )
    requested = max(
        float(STL_PLANE_DESIRED_INSET_MM),
        float(STL_PLANE_DESIRED_INSET_RADIUS_FACTOR) * radius,
    )
    branch_fraction_limit = fraction * branch_length
    legacy_limit = min(
        branch_length - 1.0e-6,
        max(
            float(STL_PLANE_SEARCH_MIN_MAX_INSET_MM),
            float(STL_PLANE_SEARCH_MAX_INSET_RADIUS_FACTOR) * radius,
        ),
    )
    maximum = min(branch_fraction_limit, legacy_limit)
    if maximum <= EPS:
        raise RuntimeError("terminal branch is too short for an inward plane")
    target = min(requested, maximum)
    # The usual 0.10 mm lower bound is relaxed only when it would itself remove
    # more than the configured fraction of a very short retained branch.
    minimum = min(float(STL_PLANE_SEARCH_MIN_INSET_MM), target)
    step = float(STL_PLANE_SEARCH_STEP_MM)
    if step <= EPS:
        raise ValueError("STL_PLANE_SEARCH_STEP_MM must be positive")

    closer = np.arange(target, minimum - 0.5 * step, -step)
    farther = np.arange(target + step, maximum + 0.5 * step, step)
    values = [target]
    values.extend(float(value) for value in closer[1:])
    if minimum < target - EPS and not any(
        abs(value - minimum) <= EPS for value in values
    ):
        values.append(minimum)
    values.extend(float(value) for value in farther)
    # Clamp round-off and retain stable order while removing duplicates.
    schedule = []
    for value in values:
        value = float(np.clip(value, minimum, maximum))
        if not any(abs(value - existing) <= 1.0e-9 for existing in schedule):
            schedule.append(value)
    return np.asarray(schedule, dtype=float), {
        "requested_inset_mm": requested,
        "target_inset_mm": target,
        "minimum_inset_mm": minimum,
        "maximum_inset_mm": maximum,
        "branch_fraction_limit_mm": branch_fraction_limit,
        "short_branch_limited": target < requested - EPS,
    }


def _search_stl_validated_terminal_plane(record, solid):
    """Move/orient a terminal plane to the first exact safe STL cross-section."""
    anchor, outward_seed, extension = _terminal_surface_anchor(record, solid)
    raw = np.asarray(record["terminal_raw_points"], dtype=float)
    path = np.vstack((anchor, raw))
    # Remove coincident anchor/terminal pairs produced by graph cropping.
    keep = np.concatenate(([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > EPS))
    path = path[keep]
    cumulative = _cumulative_distances(path)
    if len(path) < 3 or cumulative[-1] <= EPS:
        raise RuntimeError("terminal path is too short for STL plane search")

    radius = max(float(record["radius"]), float(MIN_PLAUSIBLE_RADIUS_MM))
    distances, inset_policy = _terminal_plane_inset_schedule(
        cumulative[-1], radius
    )
    minimum = float(np.min(distances))
    maximum = float(np.max(distances))
    raw_cumulative = _cumulative_distances(raw)
    tangent_window = min(
        raw_cumulative[-1], float(STL_PLANE_TANGENT_AVERAGING_LENGTH_MM)
    )
    if tangent_window <= EPS:
        raise RuntimeError("terminal path is too short to average its tangent")
    last_reason = "no candidate evaluated"
    anchor_to_graph_tip = float(np.linalg.norm(anchor - raw[0]))
    cap, cap_reason = _fit_stl_terminal_cap_clustered(
        solid, anchor, outward_seed, radius
    )
    if cap is not None:
        anchor = cap["centre"]
        record["stl_cap_used"] = True
        record["stl_cap_triangle_count"] = cap["triangle_count"]
        record["stl_cap_area"] = cap["area"]
        record["stl_cap_planarity_error_mm"] = cap["planarity_error"]
        record["stl_cap_graph_deviation_degrees"] = cap[
            "graph_deviation_degrees"
        ]
        record["stl_cap_anchor_offset_mm"] = cap["anchor_offset_mm"]
        record["stl_cap_anchor_plane_offset_mm"] = cap[
            "anchor_plane_offset_mm"
        ]
    else:
        record["stl_cap_used"] = False
        record["stl_cap_reason"] = cap_reason
    for inset in distances:
        centre = (
            anchor - float(inset) * cap["normal"]
            if cap is not None
            else _polyline_position(path, cumulative, inset)
        )
        graph_distance = max(0.0, float(inset) - anchor_to_graph_tip)
        local_outward = _local_averaged_outward_tangent(
            raw, raw_cumulative, graph_distance, tangent_window
        )
        if np.dot(local_outward, outward_seed) < 0.0:
            local_outward = -local_outward
        normal = cap["normal"].copy() if cap is not None else local_outward.copy()
        if np.dot(normal, anchor - centre) < 0.0:
            normal = -normal
        result, reason = _validate_stl_plane_intersection(
            solid, centre, normal, radius
        )
        if result is None:
            last_reason = reason
            continue
        final_centre = centre + result["surface_loop_centre_offset"]
        persisted_result, persisted_reason = _validate_stl_plane_intersection(
            solid, final_centre, normal, radius
        )
        if persisted_result is None:
            last_reason = (
                "recentered final plane is not a complete STL cross-section: {}"
                .format(persisted_reason)
            )
            continue
        record["terminal_position"] = anchor
        record["terminal_surface_anchor"] = anchor
        record["terminal_surface_extension_mm"] = extension
        record["centreline_plane_centre"] = centre
        record["centre"] = final_centre
        record["normal"] = normal
        record["surface_plane_inset_mm"] = float(inset)
        record["terminal_branch_length_mm"] = float(cumulative[-1])
        record["surface_plane_requested_inset_mm"] = inset_policy[
            "requested_inset_mm"
        ]
        record["surface_plane_target_inset_mm"] = inset_policy["target_inset_mm"]
        record["surface_plane_short_branch_limited"] = inset_policy[
            "short_branch_limited"
        ]
        record["surface_plane_removed_branch_fraction"] = float(
            inset / cumulative[-1]
        )
        record["terminal_tangent_window_mm"] = float(tangent_window)
        # Plane normals are sign-equivalent geometrically; outward-side
        # orientation is enforced separately below.
        record["terminal_tangent_cosine"] = abs(float(np.dot(
            normal, local_outward
        )))
        record["plane_normal_source"] = (
            "stl_terminal_cap" if cap is not None else "amira_local_tangent"
        )
        record["surface_cross_section_radius"] = result["surface_loop_radius"]
        record["surface_cross_section_used"] = True
        record["surface_cross_section_reason"] = "exact_stl_loop"
        record["surface_cross_section_point_count"] = result["surface_loop_points"]
        record.update(result)
        _force_outward_plane_normal(record)
        return
    raise RuntimeError(
        "no single complete, isolated STL cross-section in the adaptive {:.3f} "
        "to {:.3f} mm range ({})".format(minimum, maximum, last_reason)
    )


def _measure_surface_cross_section(record, surface_points, surface_tree):
    """Measure the wall radius in a thin slab through the proposed plane."""
    graph_radius = max(float(record["radius"]), float(MIN_PLAUSIBLE_RADIUS_MM))
    search_radius = max(
        float(SURFACE_CROSS_SECTION_MIN_SEARCH_RADIUS_MM),
        float(SURFACE_CROSS_SECTION_RADIAL_RADIUS_FACTOR) * graph_radius,
    )
    indices = surface_tree.query_ball_point(record["centre"], search_radius)
    record["surface_cross_section_used"] = False
    if len(indices) < int(SURFACE_CROSS_SECTION_MIN_POINTS):
        record["surface_cross_section_reason"] = "too_few_nearby_vertices"
        record["surface_cross_section_radius"] = graph_radius
        return

    delta = np.asarray(surface_points[np.asarray(indices, dtype=int)], dtype=float)
    delta -= np.asarray(record["centre"], dtype=float)
    normal = _unit(record["normal"], "terminal plane normal")
    axial = np.sum(delta * normal[None, :], axis=1)
    projected = delta - axial[:, None] * normal
    radial = np.linalg.norm(projected, axis=1)
    slab = max(
        float(SURFACE_CROSS_SECTION_MIN_SLAB_MM),
        float(SURFACE_CROSS_SECTION_AXIAL_RADIUS_FACTOR) * graph_radius,
    )
    selected = radial[
        (np.abs(axial) <= slab)
        & (radial <= float(SURFACE_CROSS_SECTION_RADIAL_RADIUS_FACTOR) * graph_radius)
    ]
    if len(selected) < int(SURFACE_CROSS_SECTION_MIN_POINTS):
        record["surface_cross_section_reason"] = "too_few_slab_vertices"
        record["surface_cross_section_radius"] = graph_radius
        return

    measured = float(np.percentile(
        selected, float(SURFACE_CROSS_SECTION_PERCENTILE)
    ))
    record["surface_cross_section_radius"] = max(graph_radius, measured)
    record["surface_cross_section_point_count"] = int(len(selected))
    record["surface_cross_section_used"] = True


def _classify_terminals(terminals):
    explicit = set(INLET_NODE_NAMES)
    if explicit:
        missing = explicit.difference(item["node_name"] for item in terminals)
        if missing:
            raise RuntimeError("INLET_NODE_NAMES were not found: {}".format(sorted(missing)))
        inlet_names = explicit
    else:
        count = max(0, min(int(INLET_COUNT), len(terminals)))
        ranked = sorted(terminals, key=lambda item: item["radius"], reverse=True)
        inlet_names = set(item["node_name"] for item in ranked[:count])

    inlets = [item for item in terminals if item["node_name"] in inlet_names]
    outlets = [item for item in terminals if item["node_name"] not in inlet_names]
    return inlets, outlets


def _select_distal_opening(network, inlets, outlets):
    """Return the reachable outlet farthest from the inlet along the graph."""
    if not isinstance(network, _AmiraNetwork):
        raise RuntimeError("Distal opening selection requires the Amira network")
    inlet_names = {
        item["node_name"] for item in inlets
        if item.get("centreline_source") == "amira"
    }
    if not inlet_names:
        raise RuntimeError("No Amira inlet is available for distal opening selection")
    adjacency = defaultdict(list)
    for spline in network.GetSplines():
        start = spline.GetStartNode().GetName()
        end = spline.GetEndNode().GetName()
        length = float(spline.GetLength(False))
        adjacency[start].append((end, length))
        adjacency[end].append((start, length))
    distances = {name: 0.0 for name in inlet_names}
    queue = [(0.0, name) for name in sorted(inlet_names)]
    heapq.heapify(queue)
    while queue:
        distance, name = heapq.heappop(queue)
        if distance > distances.get(name, math.inf) + EPS:
            continue
        for neighbour, length in adjacency.get(name, ()):
            candidate = distance + length
            if candidate + EPS < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    candidates = [
        item for item in outlets
        if item.get("centreline_source") == "amira"
        and item["node_name"] in distances
    ]
    if not candidates:
        raise RuntimeError("No reachable Amira outlet is available for the opening")
    selected = max(
        candidates,
        key=lambda item: (distances[item["node_name"]], item["node_name"]),
    )
    spline = selected["spline"]
    selected["opening_geodesic_distance_mm"] = float(
        distances[selected["node_name"]]
    )
    selected["opening_terminal_id"] = "edge{}:node{}".format(
        int(getattr(spline, "edge_id", -1)),
        int(getattr(selected["node"], "node_id", -1)),
    )
    return selected


def _boundary_plane_records(outlets, inlets):
    """Return every terminal that should receive a clipping-plane ROI."""
    records = list(outlets)
    if CREATE_INLET_PLANES:
        records.extend(inlets)
    return records


def _filter_boundary_outlets_by_strahler(outlets):
    """Keep only configured Amira terminal orders; leave non-Amira sources alone."""
    if BOUNDARY_OUTLET_STRAHLER_ORDERS is None:
        return list(outlets), []
    allowed = {int(value) for value in BOUNDARY_OUTLET_STRAHLER_ORDERS}
    retained = []
    rejected = []
    for record in outlets:
        spline_order = getattr(record["spline"], "strahler_order", None)
        if record.get("centreline_source") != "amira" or spline_order in allowed:
            retained.append(record)
        else:
            record["plane_skip_reason"] = (
                "terminal spline Strahler order {} is not in {}".format(
                    spline_order, sorted(allowed)
                )
            )
            rejected.append(record)
    if rejected:
        order_counts = {}
        for record in rejected:
            order = getattr(record["spline"], "strahler_order", None)
            order_counts[order] = order_counts.get(order, 0) + 1
        _log(
            "Rejected {} degree-one Amira node(s) on non-terminal Strahler "
            "orders {} before plane validation.".format(
                len(rejected), dict(sorted(order_counts.items()))
            )
        )
    return retained, rejected


def _plane_diameter(radius):
    diameter = 2.0 * float(radius) * float(OUTLET_PLANE_DIAMETER_FACTOR)
    return min(
        float(OUTLET_PLANE_MAX_DIAMETER_MM),
        max(float(OUTLET_PLANE_MIN_DIAMETER_MM), diameter),
    )


def _centreline_clearance_samples(
    network, surface_tree, spacing=None, purpose="plane"
):
    """Sample spline positions/radii for plane or refinement collision checks."""
    spacing = (
        float(PLANE_CLEARANCE_SAMPLE_SPACING_MM)
        if spacing is None else float(spacing)
    )
    records = []
    splines = list(network.GetSplines())
    total_splines = len(splines)
    progress_interval = max(1, total_splines // 10)
    _log(
        "Sampling {:,} centreline edges for {}-clearance validation...".format(
            total_splines, purpose
        )
    )
    for spline_ordinal, spline in enumerate(splines, start=1):
        if spline.IsClosed():
            continue
        length = float(spline.GetLength(False))
        segment_count = max(
            1, int(math.ceil(length / spacing))
        )
        distances = np.linspace(0.0, length, segment_count + 1)
        points = []
        for distance in distances:
            parameter = float(spline.GetParameterAtDistance(float(distance), False))
            points.append(_xyz(spline.GetPosition(parameter, False)))
        points = np.asarray(points)
        if (
            hasattr(spline, "RadiusAtDistance")
            and not getattr(spline, "use_surface_radius", False)
        ):
            radii = np.asarray(
                [spline.RadiusAtDistance(float(distance)) for distance in distances],
                dtype=float,
            )
        else:
            radii = np.asarray(surface_tree.query(points, k=1)[0], dtype=float)
        records.append({
            "spline_name": spline.GetName(),
            "start_node_name": spline.GetStartNode().GetName(),
            "end_node_name": spline.GetEndNode().GetName(),
            "length": length,
            "distances": distances,
            "points": points,
            "radii": radii,
        })
        if (
            spline_ordinal % progress_interval == 0
            or spline_ordinal == total_splines
        ):
            _log(
                "{}-clearance sampling: {:,}/{:,} edges.".format(
                    purpose.capitalize(), spline_ordinal, total_splines
                )
            )
    return records


def _set_plane_clearance(record, clearance_samples):
    """Set safe plane half-width using nearby non-local vessel centrelines."""
    best_surface_clearance = math.inf
    local_exclusion = max(
        float(PLANE_LOCAL_EXCLUSION_MM),
        3.0 * float(record["radius"]),
    )
    for samples in clearance_samples:
        delta = samples["points"] - record["centre"]
        is_local = np.zeros(len(delta), dtype=bool)
        if samples["spline_name"] == record["spline"].GetName():
            is_local = (
                np.abs(samples["distances"] - record["distance_from_start"])
                <= local_exclusion
            )

        axial = np.abs(delta.dot(record["normal"]))
        intersects_plane = axial <= (
            samples["radii"] + float(PLANE_COLLISION_SLAB_MM)
        )
        eligible = (~is_local) & intersects_plane
        if not np.any(eligible):
            continue

        eligible_delta = delta[eligible]
        eligible_axial = eligible_delta.dot(record["normal"])
        projected = eligible_delta - eligible_axial[:, None] * record["normal"]
        centreline_radial_distance = np.linalg.norm(projected, axis=1)
        surface_clearance = centreline_radial_distance - samples["radii"][eligible]
        best_surface_clearance = min(
            best_surface_clearance, float(np.min(surface_clearance))
        )

    safe_half_width = (
        math.inf
        if math.isinf(best_surface_clearance)
        else max(0.0, best_surface_clearance)
        * float(PLANE_CLEARANCE_SAFETY_FACTOR) / math.sqrt(2.0)
    )
    record["neighbour_surface_clearance"] = best_surface_clearance
    record["safe_half_width"] = safe_half_width


def _plane_half_width(record):
    """Choose a radius-relative half-width that cannot reach another vessel."""
    if record.get("surface_cross_section_reason") == "exact_stl_loop":
        half_width = float(record.get(
            "plane_half_width",
            float(record["surface_loop_radius"]) * float(
                STL_PLANE_LOOP_MARGIN_FACTOR
            ),
        ))
        maximum = 0.5 * float(OUTLET_PLANE_MAX_DIAMETER_MM)
        if half_width > maximum + EPS:
            raise RuntimeError(
                "Exact STL loop for terminal {!r} requires half-width {:.3f} mm, "
                "above configured maximum {:.3f} mm".format(
                    record["node_name"], half_width, maximum
                )
            )
        return half_width
    radius = float(record["radius"])
    desired = 0.5 * _plane_diameter(radius)
    required = max(
        0.5 * float(OUTLET_PLANE_MIN_DIAMETER_MM),
        radius * float(OUTLET_PLANE_MIN_RADIUS_FACTOR),
        float(record.get("surface_cross_section_radius", radius))
        * float(SURFACE_CROSS_SECTION_MARGIN_FACTOR),
    )
    desired = max(desired, required)
    safe = float(record.get("safe_half_width", math.inf))
    if safe + EPS < required:
        clearance = record.get("neighbour_surface_clearance", math.nan)
        raise RuntimeError(
            "No safe clipping plane for terminal {!r}: local radius {:.3f} mm "
            "requires half-width {:.3f} mm, but neighbouring-vessel clearance "
            "allows {:.3f} mm (estimated surface gap {:.3f} mm).".format(
                record["node_name"], radius, required, safe, clearance
            )
        )
    return min(max(desired, required), safe)


def _create_clipping_plane(document, record, name):
    if "plane_first_axis" in record and "plane_second_axis" in record:
        axis, angle = rotation_from_basis(
            record["plane_first_axis"],
            record["plane_second_axis"],
            record["normal"],
        )
    else:
        axis, angle = rotation_from_positive_z(record["normal"])
    half_width = float(record["plane_half_width"])
    # X-2025.06 uses ROI ``scale`` as the local half extent (as for its other
    # ROI primitives). Passing twice this value can reach an adjacent vessel
    # even though the exact finite-rectangle validator approved the footprint.
    side_x = float(record.get("plane_half_extent_x", half_width))
    side_y = float(record.get("plane_half_extent_y", half_width))
    roi = document.CreateRegionOfInterestVolume(
        Doc.Clipping,
        Doc.FinitePlane,
        _vector3d(record["centre"]),
        Vector3D(side_x, side_y, 1.0),
        _vector3d(axis),
        float(angle),
        bool(INVERT_CLIPPING_PLANES),
        False,
        False,
    )
    roi.SetName(name, False)
    return roi


def _create_plane_id_annotation(document, record, roi_name, ordinal):
    """Add a short persistent 3-D label whose name maps back to one ROI."""
    global _PLANE_ANNOTATIONS_AVAILABLE
    if not CREATE_PLANE_ID_ANNOTATIONS:
        return None
    if _PLANE_ANNOTATIONS_AVAILABLE is False:
        return None
    try:
        annotations = document.GetAnnotations()
    except Exception as exc:
        _PLANE_ANNOTATIONS_AVAILABLE = False
        _log(
            "Plane-ID annotations are unavailable in this Simpleware instance; "
            "use the GUI label helper after opening the saved SIP ({})".format(exc)
        )
        return None
    _PLANE_ANNOTATIONS_AVAILABLE = True
    centre = np.asarray(record["centre"], dtype=float)
    first = np.asarray(
        record.get("plane_first_axis", _plane_basis(record["normal"])[0]),
        dtype=float,
    )
    second = np.asarray(
        record.get("plane_second_axis", _plane_basis(record["normal"])[1]),
        dtype=float,
    )
    label_offset = max(
        0.20,
        1.25 * float(record.get("plane_half_width", record["radius"])),
    )
    corner = centre + label_offset * (first + second)
    annotation_name = "{}{}".format(PLANE_ID_ANNOTATION_PREFIX, roi_name)
    annotation = annotations.AddTextBox(
        annotation_name,
        Doc.OrientationXY,
        IntVector(),
        RealPoint3D(*[float(value) for value in centre]),
        RealPoint3D(*[float(value) for value in corner]),
        RealSize(48.0, 18.0),
        "P{:03d}".format(int(ordinal)),
        1.0,
        False,
        True,
    )
    annotation.SetDrawAnchor(True)
    annotation.SetDrawBackground(True)
    annotation.SetDrawBorder(True)
    annotation.SetScaleWithZoom(False)
    return annotation


def _contact_enum(kind):
    choices = {
        "pressure_outlet": Model.PressureOutletContact,
        "generic_outlet": Model.GenericOutletContact,
        "velocity_inlet": Model.VelocityInletContact,
        "generic_inlet": Model.GenericInletContact,
    }
    try:
        return choices[kind]
    except KeyError:
        raise ValueError("Unsupported CFD contact type: {!r}".format(kind))


def _remove_existing(document, model, part):
    global _PLANE_ANNOTATIONS_AVAILABLE
    if not REPLACE_EXISTING:
        _log("Existing scripted regions will be retained (REPLACE_EXISTING=False).")
        return

    removed_planes = 0
    for roi in list(document.GetRegionOfInterestVolumes(Doc.Clipping)):
        name = roi.GetName()
        scripted_name = (
            name.startswith(OUTLET_PREFIX)
            or name.startswith(INLET_PREFIX)
            or name.startswith(OPENING_PREFIX)
        )
        if REMOVE_ALL_EXISTING_CLIPPING_PLANES or scripted_name:
            try:
                model.RemoveCfdSurfaceContact(part, roi)
            except Exception:
                pass
            document.RemoveRegionOfInterestVolumeByName(Doc.Clipping, name)
            removed_planes += 1

    removed_refinements = 0
    for refinement in list(model.GetFeFreeMeshRefinementVolumes()):
        if refinement.GetName().startswith(REFINEMENT_PREFIX):
            model.RemoveFeFreeMeshRefinement(refinement)
            removed_refinements += 1
    removed_annotations = 0
    if hasattr(document, "GetAnnotations"):
        try:
            annotations = document.GetAnnotations()
        except Exception as exc:
            _PLANE_ANNOTATIONS_AVAILABLE = False
            if CREATE_PLANE_ID_ANNOTATIONS:
                _log(
                    "Plane-ID annotations are unavailable in this Simpleware "
                    "instance; run simpleware_add_plane_labels.py in the GUI "
                    "after opening the saved SIP ({})".format(exc)
                )
        else:
            _PLANE_ANNOTATIONS_AVAILABLE = True
            for annotation in list(annotations.GetAll()):
                if annotation.GetName().startswith(PLANE_ID_ANNOTATION_PREFIX):
                    annotations.RemoveAnnotation(annotation)
                    removed_annotations += 1
    _log(
        "Removed {} existing clipping plane(s), {} scripted refinement "
        "region(s), and {} plane-ID annotation(s){}.".format(
            removed_planes,
            removed_refinements,
            removed_annotations,
            " (including legacy/manual planes)"
            if REMOVE_ALL_EXISTING_CLIPPING_PLANES else "",
        )
    )


def _sample_spline(spline, spacing):
    length = float(spline.GetLength(False))
    segment_count = max(1, int(math.ceil(length / float(spacing))))
    distances = np.linspace(0.0, length, segment_count + 1)
    samples = []
    for distance in distances:
        parameter = float(spline.GetParameterAtDistance(float(distance), False))
        samples.append(_xyz(spline.GetPosition(parameter, False)))
    return np.asarray(samples), distances


def _refinement_sample_tangents(samples):
    """Return locally averaged unit tangents at sampled spline positions."""
    samples = np.asarray(samples, dtype=float)
    tangents = []
    for index in range(len(samples)):
        lower = max(0, index - 1)
        upper = min(len(samples) - 1, index + 1)
        if lower == upper:
            raise RuntimeError("A refinement spline has fewer than two samples")
        tangents.append(_unit(
            samples[upper] - samples[lower], "refinement sample tangent"
        ))
    return np.asarray(tangents)


def _refinement_surface_radii(
    spline, samples, graph_radii, stl_solid
):
    """Measure plausible equivalent radii from exact STL section loops.

    Equivalent radius drives both selection and the local sphere envelope. The
    former maximum-distance radius was very sensitive to one elongated/oblique
    loop and produced several giant primitives. Surface measurements are now
    accepted only when plausible relative to the Amira radius; otherwise that
    local graph radius is the deterministic fallback.
    """
    graph_radii = np.maximum(
        np.asarray(graph_radii, dtype=float), float(MIN_PLAUSIBLE_RADIUS_MM)
    )
    equivalent = graph_radii.copy()
    covering = graph_radii.copy()
    measured = np.zeros(len(samples), dtype=bool)
    outlier = np.zeros(len(samples), dtype=bool)
    if (
        str(REFINEMENT_RADIUS_SOURCE).lower() != "surface_cross_section"
        or stl_solid is None
    ):
        return equivalent, covering, measured, outlier

    tangents = _refinement_sample_tangents(samples)
    for index, (point, tangent, graph_radius) in enumerate(zip(
        samples, tangents, graph_radii
    )):
        result, _reason = _validate_stl_plane_intersection(
            stl_solid, point, tangent, graph_radius
        )
        if result is None:
            continue
        candidate = max(
            float(result["surface_loop_equivalent_radius"]),
            float(MIN_PLAUSIBLE_RADIUS_MM),
        )
        ratio = candidate / graph_radius
        if not (
            float(REFINEMENT_SURFACE_RADIUS_MIN_GRAPH_FACTOR)
            <= ratio
            <= float(REFINEMENT_SURFACE_RADIUS_MAX_GRAPH_FACTOR)
        ):
            outlier[index] = True
            continue
        equivalent[index] = candidate
        covering[index] = candidate
        measured[index] = True
    return equivalent, covering, measured, outlier


def _small_vessel_segments(network, surface_tree, stl_solid=None):
    selection_mode = str(REFINEMENT_SELECTION_MODE).lower()
    mode_aliases = {
        "radius": "diameter",
        "radius_or_strahler": "either",
        "radius_and_strahler": "both",
    }
    selection_mode = mode_aliases.get(selection_mode, selection_mode)
    valid_modes = {"all", "diameter", "strahler", "either", "both"}
    if selection_mode not in valid_modes:
        raise ValueError(
            "REFINEMENT_SELECTION_MODE must be one of {} (or a radius alias)"
            .format(
                sorted(valid_modes)
            )
        )
    radius_source = str(REFINEMENT_RADIUS_SOURCE).lower()
    if radius_source not in {"graph", "surface_cross_section"}:
        raise ValueError(
            "REFINEMENT_RADIUS_SOURCE must be 'graph' or "
            "'surface_cross_section'"
        )
    use_strahler = selection_mode in {"strahler", "either", "both"}
    selected_orders = {int(order) for order in REFINEMENT_STRAHLER_ORDERS}
    if use_strahler and not selected_orders:
        raise ValueError(
            "REFINEMENT_STRAHLER_ORDERS cannot be empty when Strahler selection "
            "is enabled"
        )

    threshold_radius = (
        float(SMALL_VESSEL_CROSS_SECTION_RADIUS_MM)
        if radius_source == "surface_cross_section"
        else 0.5 * float(SMALL_VESSEL_DIAMETER_MM)
    )
    candidates = []
    splines = list(network.GetSplines())
    total_splines = len(splines)
    progress_interval = max(1, total_splines // 10)
    _log(
        "Selecting small-vessel regions from {:,} centreline edges "
        "(mode={!r})...".format(total_splines, selection_mode)
    )
    for spline_ordinal, spline in enumerate(splines, start=1):
        if spline.IsClosed():
            continue
        strahler_order = getattr(spline, "strahler_order", None)
        if use_strahler and strahler_order is None:
            raise RuntimeError(
                "Spline {!r} has no Strahler order. Use an Amira graph containing "
                "per-edge strahler/StrahlerOrder data, or select diameter mode."
                .format(spline.GetName())
            )
        strahler_match = strahler_order in selected_orders
        samples, distances = _sample_spline(spline, REFINEMENT_SAMPLE_SPACING_MM)
        if (
            hasattr(spline, "RadiusAtDistance")
            and not getattr(spline, "use_surface_radius", False)
        ):
            graph_radii = np.asarray([
                spline.RadiusAtDistance(float(distance)) for distance in distances
            ], dtype=float)
        else:
            if surface_tree is None:
                graph_radii = np.full(
                    len(samples), float(MIN_PLAUSIBLE_RADIUS_MM), dtype=float
                )
            else:
                graph_radii = np.asarray(
                    surface_tree.query(samples, k=1)[0], dtype=float
                )
        (
            selection_radii,
            covering_radii,
            measured,
            radius_outlier,
        ) = _refinement_surface_radii(spline, samples, graph_radii, stl_solid)
        for index, (start, end) in enumerate(zip(samples[:-1], samples[1:])):
            # Requiring the maximum endpoint section to pass the radius
            # threshold avoids allowing a primitive on a taper to spill into a
            # larger parent vessel. Adjacent primitives cover the shared end.
            selection_radius = float(max(
                selection_radii[index], selection_radii[index + 1]
            ))
            segment_radius = float(max(
                covering_radii[index], covering_radii[index + 1]
            ))
            if (
                not (use_strahler and strahler_match)
                and selection_radius < float(MIN_PLAUSIBLE_RADIUS_MM)
            ):
                continue
            diameter_match = selection_radius <= threshold_radius
            selected = {
                "all": True,
                "diameter": diameter_match,
                "strahler": strahler_match,
                "either": diameter_match or strahler_match,
                "both": diameter_match and strahler_match,
            }[selection_mode]
            if not selected:
                continue
            length = float(np.linalg.norm(end - start))
            if length <= EPS:
                continue
            candidates.append(
                {
                    "spline": spline,
                    "spline_name": spline.GetName(),
                    "spline_start_node": spline.GetStartNode().GetName(),
                    "spline_end_node": spline.GetEndNode().GetName(),
                    "spline_length": float(spline.GetLength(False)),
                    "segment_index": index,
                    "start": start,
                    "end": end,
                    "length": length,
                    "start_distance": float(distances[index]),
                    "end_distance": float(distances[index + 1]),
                    # Never punch a hole in a Strahler-selected edge because of
                    # a zero/noisy endpoint radius; clamp only its primitive size.
                    "radius": max(
                        segment_radius, float(MIN_PLAUSIBLE_RADIUS_MM)
                    ),
                    "start_radius": float(covering_radii[index]),
                    "end_radius": float(covering_radii[index + 1]),
                    "selection_radius": selection_radius,
                    "start_selection_radius": float(selection_radii[index]),
                    "end_selection_radius": float(selection_radii[index + 1]),
                    "graph_radius": float(max(
                        graph_radii[index], graph_radii[index + 1],
                        float(MIN_PLAUSIBLE_RADIUS_MM),
                    )),
                    "surface_radius_measured": bool(
                        measured[index] and measured[index + 1]
                    ),
                    "surface_radius_outlier": bool(
                        radius_outlier[index] or radius_outlier[index + 1]
                    ),
                    "strahler_order": strahler_order,
                }
            )
        if (
            spline_ordinal % progress_interval == 0
            or spline_ordinal == total_splines
        ):
            _log(
                "Refinement selection: {:,}/{:,} edges; {:,} region(s) so far."
                .format(spline_ordinal, total_splines, len(candidates))
            )
    if (
        str(REFINEMENT_PRIMITIVE).lower() == "sphere"
        and REFINEMENT_OPTIMIZE_SPHERE_COUNT
    ):
        unoptimised_count = len(candidates)
        candidates = _optimise_refinement_spheres(candidates)
        _log(
            "Adaptive sphere spacing: {:,} selected measurement segment(s) "
            "-> {:,} covering sphere(s) ({:.1%} reduction).".format(
                unoptimised_count,
                len(candidates),
                (
                    0.0 if unoptimised_count == 0 else
                    1.0 - len(candidates) / float(unoptimised_count)
                ),
            )
        )
    if len(candidates) > int(MAX_REFINEMENT_VOLUMES):
        raise RuntimeError(
            "{} refinement volumes are required, exceeding MAX_REFINEMENT_VOLUMES={}. "
            "Increase REFINEMENT_SAMPLE_SPACING_MM or the limit.".format(
                len(candidates), MAX_REFINEMENT_VOLUMES
            )
        )
    return candidates


def _polyline_midpoint(points):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= EPS:
        return points[0].copy(), 0.0
    target = 0.5 * total
    cumulative = 0.0
    for start, end, length in zip(points[:-1], points[1:], lengths):
        if cumulative + length >= target:
            fraction = (target - cumulative) / float(length)
            return start + fraction * (end - start), total
        cumulative += float(length)
    return points[-1].copy(), total


def _sphere_radius_covering_sections(points, wall_radii, centre):
    """Sphere radius covering circular sections at sampled curve positions."""
    points = np.asarray(points, dtype=float)
    wall_radii = np.asarray(wall_radii, dtype=float)
    centre = np.asarray(centre, dtype=float)
    tangents = _refinement_sample_tangents(points)
    delta = points - centre[None, :]
    axial = np.sum(delta * tangents, axis=1)
    perpendicular = np.linalg.norm(
        delta - axial[:, None] * tangents, axis=1
    )
    # The farthest point on a circular section lies in the direction of the
    # centre offset projected into that section's plane.
    required_squared = (
        np.sum(delta * delta, axis=1)
        + wall_radii * wall_radii
        + 2.0 * wall_radii * perpendicular
    )
    return float(np.sqrt(np.max(required_squared)))


def _merged_refinement_sphere(segments):
    points = [np.asarray(segments[0]["start"], dtype=float)]
    coverage_radii = [float(segments[0]["start_radius"])]
    selection_radii = [float(segments[0]["start_selection_radius"])]
    for segment in segments:
        points.append(np.asarray(segment["end"], dtype=float))
        coverage_radii.append(float(segment["end_radius"]))
        selection_radii.append(float(segment["end_selection_radius"]))
    points = np.asarray(points)
    coverage_radii = np.asarray(coverage_radii)
    centre, arc_length = _polyline_midpoint(points)
    preferred_wall = np.asarray([
        _desired_refinement_wall_radius(radius) for radius in coverage_radii
    ])
    minimum_wall = coverage_radii + float(REFINEMENT_MIN_PADDING_MM)
    desired_sphere = _sphere_radius_covering_sections(
        points, preferred_wall, centre
    )
    minimum_sphere = _sphere_radius_covering_sections(
        points, minimum_wall, centre
    )

    merged = dict(segments[0])
    merged.update({
        "end": np.asarray(segments[-1]["end"], dtype=float),
        "end_distance": float(segments[-1]["end_distance"]),
        "end_radius": float(segments[-1]["end_radius"]),
        "end_selection_radius": float(
            segments[-1]["end_selection_radius"]
        ),
        "length": arc_length,
        "centre": centre,
        "radius": float(np.max(coverage_radii)),
        "selection_radius": float(np.max(selection_radii)),
        "graph_radius": float(max(item["graph_radius"] for item in segments)),
        "surface_radius_measured": all(
            item.get("surface_radius_measured", False) for item in segments
        ),
        "surface_radius_outlier": any(
            item.get("surface_radius_outlier", False) for item in segments
        ),
        "segment_end_index": int(segments[-1]["segment_index"]),
        "merged_segment_count": len(segments),
        "sphere_desired_radius": desired_sphere,
        "sphere_minimum_radius": minimum_sphere,
        "sphere_wall_radius": float(np.max(preferred_wall)),
    })
    return merged


def _optimise_refinement_spheres(segments):
    """Greedily merge selected intervals using radius/curvature-aware spacing."""
    if not segments:
        return []
    optimised = []
    index = 0
    maximum_arc = float(REFINEMENT_SPHERE_MAX_ARC_LENGTH_MM)
    maximum_expansion = float(
        REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR
    )
    while index < len(segments):
        run = [segments[index]]
        best = _merged_refinement_sphere(run)
        next_index = index + 1
        while next_index < len(segments):
            previous = segments[next_index - 1]
            candidate = segments[next_index]
            if (
                candidate["spline_name"] != previous["spline_name"]
                or int(candidate["segment_index"])
                != int(previous["segment_index"]) + 1
                or abs(
                    float(candidate["start_distance"])
                    - float(previous["end_distance"])
                ) > 1.0e-8
            ):
                break
            trial = _merged_refinement_sphere(run + [candidate])
            if trial["length"] > maximum_arc + EPS:
                break
            expansion = (
                trial["sphere_desired_radius"]
                / max(trial["sphere_wall_radius"], EPS)
            )
            if expansion > maximum_expansion + EPS:
                break
            run.append(candidate)
            best = trial
            next_index += 1
        optimised.append(best)
        index += len(run)
    return optimised


def _selected_refinement_intervals(segments):
    intervals = defaultdict(list)
    for segment in segments:
        intervals[segment["spline_name"]].append((
            float(segment["start_distance"]),
            float(segment["end_distance"]),
        ))
    return intervals


def _refinement_selected_sample_mask(samples, selected_intervals):
    mask = np.zeros(len(samples["distances"]), dtype=bool)
    guard = float(REFINEMENT_CLEARANCE_LOCAL_EXCLUSION_MM)
    for lower, upper in selected_intervals.get(samples["spline_name"], ()):
        mask |= (
            (samples["distances"] >= lower - guard)
            & (samples["distances"] <= upper + guard)
        )
    return mask


def _refinement_obstacle_coordinates(segment, clearance_samples, intervals=None):
    """Obstacle samples for one primitive.

    A primitive's own spline and directly connected splines are not obstacles:
    overlap along one vessel or through a genuine bifurcation is required for
    continuous refinement coverage.  Only non-topologically-adjacent branches
    constrain the primitive, matching the adaptive-study clearance policy.
    """
    direction = _unit(segment["end"] - segment["start"], "refinement segment")
    centre = np.asarray(
        segment.get("centre", 0.5 * (segment["start"] + segment["end"])),
        dtype=float,
    )
    radial_parts = []
    axial_parts = []
    radius_parts = []
    name_parts = []
    own_name = segment["spline_name"]
    own_nodes = {
        segment.get("spline_start_node"), segment.get("spline_end_node")
    }
    own_nodes.discard(None)
    for samples in clearance_samples:
        if samples["spline_name"] == own_name:
            continue
        shared = own_nodes.intersection({
            samples.get("start_node_name"), samples.get("end_node_name")
        })
        if shared:
            continue
        # A refinement volume may overlap another interval that is itself
        # selected at the same target size.  Such overlap does not leak the
        # refinement outside the region of interest and is required where
        # selected vessels run close together.  Previously ``intervals`` was
        # passed into this function but never applied, so every nearby vessel
        # was treated as an unselected obstacle.  In an all-edge adaptive
        # study that falsely rejected otherwise valid covering spheres.
        obstacle_mask = np.ones(len(samples["points"]), dtype=bool)
        if intervals is not None:
            obstacle_mask &= ~_refinement_selected_sample_mask(
                samples, intervals
            )
        if not np.any(obstacle_mask):
            continue
        sample_points = samples["points"][obstacle_mask]
        sample_radii = samples["radii"][obstacle_mask]
        delta = sample_points - centre
        signed_axial = delta.dot(direction)
        radial = np.linalg.norm(
            delta - signed_axial[:, None] * direction[None, :], axis=1
        )
        radial_parts.append(radial)
        axial_parts.append(np.abs(signed_axial))
        radius_parts.append(sample_radii)
        name_parts.extend([samples["spline_name"]] * len(radial))
    if not radial_parts:
        return (
            np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=object)
        )
    return (
        np.concatenate(radial_parts),
        np.concatenate(axial_parts),
        np.concatenate(radius_parts),
        np.asarray(name_parts, dtype=object),
    )


def _primitive_hits_obstacle(primitive, radial, axial, obstacle_radius, r, a):
    if primitive == "sphere":
        centre_distance = np.sqrt(radial * radial + axial * axial)
        return centre_distance - obstacle_radius <= r
    effective_radial = np.maximum(radial - obstacle_radius, 0.0)
    effective_axial = np.maximum(axial - obstacle_radius, 0.0)
    if primitive == "cylinder":
        return (effective_radial <= r) & (effective_axial <= a)
    return (effective_radial / r) ** 2 + (effective_axial / a) ** 2 <= 1.0


def _desired_refinement_wall_radius(coverage_radius):
    """Return the locally scaled radial envelope before chord accommodation."""
    coverage_radius = float(coverage_radius)
    return (
        coverage_radius
        * (1.0 + float(REFINEMENT_PADDING_RADIUS_FACTOR))
        + float(REFINEMENT_PADDING_MM)
    )


def _set_refinement_clearance(segment, clearance_samples, intervals):
    """Adapt one primitive so only selected centreline ownership enters it."""
    coverage_radius = float(segment["radius"])
    minimum_radius = coverage_radius + float(REFINEMENT_MIN_PADDING_MM)
    desired_radius = _desired_refinement_wall_radius(coverage_radius)
    half_chord = 0.5 * float(segment["length"])
    minimum_axial = half_chord
    desired_axial = half_chord + (
        desired_radius * float(REFINEMENT_AXIAL_OVERLAP_RADIUS_FACTOR)
    )
    primitive = str(REFINEMENT_PRIMITIVE).lower()
    if primitive == "sphere":
        minimum_radius = float(segment.get(
            "sphere_minimum_radius",
            math.sqrt(minimum_radius ** 2 + half_chord ** 2),
        ))
        desired_radius = float(segment.get(
            "sphere_desired_radius",
            math.sqrt(desired_radius ** 2 + half_chord ** 2),
        ))
        minimum_axial = minimum_radius
        desired_axial = desired_radius

    radial, axial, obstacle_radius, obstacle_names = (
        _refinement_obstacle_coordinates(
            segment, clearance_samples, intervals
        )
    )
    region_radius = desired_radius
    axial_half_length = desired_axial
    if len(radial) and primitive == "sphere":
        safety = float(REFINEMENT_CLEARANCE_SAFETY_FACTOR)
        centre_distance = np.sqrt(radial * radial + axial * axial)
        sphere_cap = float(np.min(np.maximum(
            centre_distance - obstacle_radius, 0.0
        ))) * safety
        region_radius = min(region_radius, sphere_cap)
        axial_half_length = region_radius
    elif len(radial):
        safety = float(REFINEMENT_CLEARANCE_SAFETY_FACTOR)
        # Cap the radial extent only for obstacles that cannot be escaped by
        # shortening to the required chord; cap the axial extent only for
        # obstacles that cannot be escaped by shrinking to required wall
        # coverage. This avoids unnecessarily collapsing both dimensions for a
        # simple parallel neighbour.
        axial_relevant = axial <= minimum_axial + obstacle_radius
        if np.any(axial_relevant):
            radial_cap = np.min(
                np.maximum(radial[axial_relevant] - obstacle_radius[axial_relevant], 0.0)
            ) * safety
            region_radius = min(region_radius, float(radial_cap))
        radial_relevant = radial <= minimum_radius + obstacle_radius
        if np.any(radial_relevant):
            axial_cap = np.min(
                np.maximum(axial[radial_relevant] - obstacle_radius[radial_relevant], 0.0)
            ) * safety
            axial_half_length = min(axial_half_length, float(axial_cap))

    unsafe_reason = None
    if region_radius + EPS < minimum_radius:
        unsafe_reason = (
            "neighbour clearance permits radial extent {:.3f} mm but target "
            "coverage requires {:.3f} mm".format(region_radius, minimum_radius)
        )
    elif axial_half_length + EPS < minimum_axial:
        unsafe_reason = (
            "neighbour clearance permits axial half-length {:.3f} mm but the "
            "target chord requires {:.3f} mm".format(
                axial_half_length, minimum_axial
            )
        )
    else:
        region_radius = max(region_radius, minimum_radius)
        axial_half_length = max(axial_half_length, minimum_axial)
        if len(radial):
            collision = _primitive_hits_obstacle(
                primitive, radial, axial, obstacle_radius,
                region_radius, axial_half_length,
            )
            if np.any(collision):
                first = int(np.flatnonzero(collision)[0])
                unsafe_reason = (
                    "minimum covering primitive still intersects unselected "
                    "edge {!r}".format(obstacle_names[first])
                )

    segment["desired_region_radius"] = desired_radius
    segment["desired_axial_half_length"] = desired_axial
    segment["region_radius"] = region_radius
    segment["axial_half_length"] = axial_half_length
    segment["clearance_reduced"] = bool(
        region_radius < desired_radius - EPS
        or axial_half_length < desired_axial - EPS
    )
    if unsafe_reason is not None:
        segment["refinement_skip_reason"] = unsafe_reason
        return False
    return True


def _apply_refinement_clearance(segments, clearance_samples):
    if not REFINEMENT_NEIGHBOUR_AWARE:
        return list(segments), []
    intervals = _selected_refinement_intervals(segments)
    accepted = []
    rejected = []
    for segment in segments:
        if _set_refinement_clearance(segment, clearance_samples, intervals):
            accepted.append(segment)
        else:
            rejected.append(segment)
    if rejected and not SKIP_UNSAFE_REFINEMENT_SEGMENTS:
        first = rejected[0]
        raise RuntimeError(
            "Unsafe refinement segment {} s{}: {}".format(
                first["spline_name"], first["segment_index"],
                first["refinement_skip_reason"],
            )
        )
    return accepted, rejected


def _refinement_mesh_size(segment):
    """Characteristic length for one radius-aware refinement primitive."""
    if not REFINEMENT_USE_RADIUS_MESH_SIZE:
        return float(REFINEMENT_MESH_SIZE_MM)
    across = float(REFINEMENT_ELEMENTS_ACROSS_DIAMETER)
    if across <= 0.0:
        raise ValueError("REFINEMENT_ELEMENTS_ACROSS_DIAMETER must be positive")
    radius = max(
        float(segment.get("selection_radius", segment["radius"])),
        float(MIN_PLAUSIBLE_RADIUS_MM),
    )
    size = 2.0 * radius / across
    return min(
        float(REFINEMENT_MAX_MESH_SIZE_MM),
        max(float(REFINEMENT_MIN_MESH_SIZE_MM), size),
    )


def _create_refinement(model, part_vector, segment, ordinal):
    direction = _unit(segment["end"] - segment["start"], "refinement segment")
    axis, angle = rotation_from_positive_z(direction)
    centre = np.asarray(
        segment.get("centre", 0.5 * (segment["start"] + segment["end"])),
        dtype=float,
    )
    region_radius = float(segment.get(
        "region_radius",
        _desired_refinement_wall_radius(segment["radius"]),
    ))
    axial_half_length = float(segment.get(
        "axial_half_length",
        0.5 * float(segment["length"]) + (
            region_radius * float(REFINEMENT_AXIAL_OVERLAP_RADIUS_FACTOR)
        ),
    ))
    primitive = str(REFINEMENT_PRIMITIVE).lower()
    if primitive == "cylinder":
        shape = Doc.Cylinder
    elif primitive in {"ellipsoid", "sphere"}:
        shape = Doc.Ellipsoid
        if primitive == "sphere":
            axial_half_length = region_radius
    else:
        raise ValueError(
            "REFINEMENT_PRIMITIVE must be 'cylinder', 'ellipsoid', or 'sphere'"
        )

    refinement = model.CreateFeFreeMeshRefinementVolume(
        shape,
        _vector3d(centre),
        # Simpleware primitive scale uses full local-axis extents. Keep the
        # geometric calculations in radii/half-lengths, then convert here.
        Vector3D(
            2.0 * region_radius,
            2.0 * region_radius,
            2.0 * axial_half_length,
        ),
        _vector3d(axis),
        float(angle),
        False,
    )
    order_text = (
        "_o{:02d}".format(segment["strahler_order"])
        if segment["strahler_order"] is not None
        else ""
    )
    start_index = int(segment["segment_index"])
    end_index = int(segment.get("segment_end_index", start_index))
    segment_text = (
        "s{:03d}".format(start_index)
        if end_index == start_index else
        "s{:03d}-{:03d}".format(start_index, end_index)
    )
    name = "{}{:03d}_{}{}_{}".format(
        REFINEMENT_PREFIX,
        ordinal,
        _safe_name(segment["spline_name"]),
        order_text,
        segment_text,
    )
    refinement.SetName(name, False)
    refinement.SetValueType(FeFreeMeshRefinementVolume.MM)
    mesh_size = _refinement_mesh_size(segment)
    segment["mesh_size_mm"] = mesh_size
    refinement.SetMeshSize(mesh_size)
    refinement.SetRefinementType(
        FeFreeMeshRefinementVolume.Volume
        if REFINEMENT_TYPE == "volume"
        else FeFreeMeshRefinementVolume.Surface
    )
    refinement.SetParts(part_vector)
    return refinement


def main():
    global _RUN_STARTED_AT
    _RUN_STARTED_AT = time.perf_counter()
    _log("Starting coronary boundary/refinement automation.")
    if _SIMPLEWARE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Run this script inside Simpleware X-2025.06"
        ) from _SIMPLEWARE_IMPORT_ERROR
    if REFINEMENT_TYPE not in ("volume", "surface"):
        raise ValueError("REFINEMENT_TYPE must be 'volume' or 'surface'")
    if str(REFINEMENT_PRIMITIVE).lower() not in (
        "cylinder", "ellipsoid", "sphere"
    ):
        raise ValueError(
            "REFINEMENT_PRIMITIVE must be 'cylinder', 'ellipsoid', or 'sphere'"
        )
    if float(REFINEMENT_AXIAL_OVERLAP_RADIUS_FACTOR) < 0.0:
        raise ValueError("REFINEMENT_AXIAL_OVERLAP_RADIUS_FACTOR cannot be negative")
    if float(REFINEMENT_MIN_PADDING_MM) < 0.0:
        raise ValueError("REFINEMENT_MIN_PADDING_MM cannot be negative")
    if float(REFINEMENT_PADDING_MM) < 0.0:
        raise ValueError("REFINEMENT_PADDING_MM cannot be negative")
    if float(REFINEMENT_PADDING_RADIUS_FACTOR) < 0.0:
        raise ValueError("REFINEMENT_PADDING_RADIUS_FACTOR cannot be negative")
    if (
        float(REFINEMENT_PADDING_MM)
        + float(MIN_PLAUSIBLE_RADIUS_MM)
        * float(REFINEMENT_PADDING_RADIUS_FACTOR)
        < float(REFINEMENT_MIN_PADDING_MM)
    ):
        raise ValueError(
            "The configured absolute + relative refinement padding at "
            "MIN_PLAUSIBLE_RADIUS_MM cannot be smaller than "
            "REFINEMENT_MIN_PADDING_MM"
        )
    if not (
        0.0 < float(REFINEMENT_SURFACE_RADIUS_MIN_GRAPH_FACTOR)
        <= 1.0
        <= float(REFINEMENT_SURFACE_RADIUS_MAX_GRAPH_FACTOR)
    ):
        raise ValueError(
            "Surface-radius graph factors must bracket 1.0 and remain positive"
        )
    if float(REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR) < 1.0:
        raise ValueError(
            "REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR must be >= 1.0"
        )
    if float(REFINEMENT_SPHERE_MAX_ARC_LENGTH_MM) <= 0.0:
        raise ValueError("REFINEMENT_SPHERE_MAX_ARC_LENGTH_MM must be positive")
    if not 0.0 < float(REFINEMENT_CLEARANCE_SAFETY_FACTOR) <= 1.0:
        raise ValueError(
            "REFINEMENT_CLEARANCE_SAFETY_FACTOR must be in (0, 1]"
        )
    if float(REFINEMENT_ELEMENTS_ACROSS_DIAMETER) <= 0.0:
        raise ValueError("REFINEMENT_ELEMENTS_ACROSS_DIAMETER must be positive")
    if not (
        0.0 < float(REFINEMENT_MIN_MESH_SIZE_MM)
        <= float(REFINEMENT_MAX_MESH_SIZE_MM)
    ):
        raise ValueError(
            "Refinement mesh-size limits must be positive and ordered"
        )

    document = App.GetDocument()
    model = document.GetActiveModel()
    if model.GetModelType() != Model.Cfd:
        raise RuntimeError("The active model must be a CFD model")

    surface = _choose_surface(document)
    part = _choose_part(model, surface)
    boundary_source_names = tuple(
        str(source).lower() for source in BOUNDARY_CENTRELINE_SOURCES
    )
    if not boundary_source_names:
        boundary_source_names = (str(BOUNDARY_CENTRELINE_SOURCE).lower(),)
    refinement_source_name = str(REFINEMENT_CENTRELINE_SOURCE).lower()
    _log(
        "Using surface {!r}, part {!r}, boundary sources {!r}, and refinement "
        "source {!r}.".format(
            surface.GetName(),
            part.GetName(),
            boundary_source_names,
            refinement_source_name,
        )
    )

    _log("Reading surface vertices from {!r}...".format(surface.GetName()))
    surface_points = _surface_points_global(surface)
    _log("Building nearest-surface spatial index...")
    surface_tree = cKDTree(surface_points)
    _log("Built surface index from {:,} vertices.".format(len(surface_points)))

    source_names = set(boundary_source_names)
    source_names.add(refinement_source_name)
    invalid_sources = source_names.difference({"amira", "simpleware"})
    if invalid_sources:
        raise ValueError(
            "Boundary/refinement centreline sources must be 'simpleware' or "
            "'amira'; found {}".format(sorted(invalid_sources))
        )

    stl_solid = None
    if "amira" in source_names and CROP_AMIRA_TO_STL:
        stl_solid = _load_stl_solid()
        _log("Validating the STL against the selected Simpleware surface...")
        _validate_stl_matches_selected_surface(stl_solid, surface_points)

    source_networks = {}
    for source_name in sorted(source_names):
        _log("Loading {!r} centreline source...".format(source_name))
        source_network = _choose_centreline_source(document, source_name)
        if isinstance(source_network, _AmiraNetwork) and CROP_AMIRA_TO_STL:
            source_network = _crop_amira_network_to_stl(
                source_network, stl_solid
            )
        _log("Validating {!r} centreline/surface alignment...".format(
            source_name
        ))
        _validate_amira_alignment(
            source_network, surface_points, surface_tree
        )
        source_networks[source_name] = source_network

    boundary_networks = [
        (source_name, source_networks[source_name])
        for source_name in boundary_source_names
    ]
    refinement_network = source_networks[refinement_source_name]

    terminals_by_source = []
    for source_name, boundary_network in boundary_networks:
        _log(
            "Finding usable degree-one terminal nodes in the {!r} boundary "
            "network...".format(source_name)
        )
        source_terminals = _find_terminals(
            boundary_network, surface_tree, source_name=source_name
        )
        _log(
            "Boundary source {!r}: {} usable terminal(s).".format(
                source_name, len(source_terminals)
            )
        )
        terminals_by_source.append((source_name, source_terminals))
    terminals = _merge_boundary_terminals(terminals_by_source)
    _log("Locating and validating terminal planes against exact STL triangles...")
    for record in terminals:
        if stl_solid is not None:
            try:
                _search_stl_validated_terminal_plane(record, stl_solid)
            except RuntimeError as exc:
                record["plane_geometry_skip_reason"] = str(exc)
                _log(
                    "WARNING: terminal {!r} has no exact safe STL plane: {}"
                    .format(record["node_name"], exc)
                )
        else:
            _refine_plane_normal_from_surface(record, surface_points, surface_tree)
            _force_outward_plane_normal(record)
            _measure_surface_cross_section(record, surface_points, surface_tree)
    inlets, outlets = _classify_terminals(terminals)
    outlets, strahler_rejected_outlets = _filter_boundary_outlets_by_strahler(
        outlets
    )
    opening_record = None
    if CREATE_DISTAL_OPENING:
        if "amira" not in source_networks:
            raise RuntimeError(
                "CREATE_DISTAL_OPENING requires the Amira centreline source"
            )
        opening_record = _select_distal_opening(
            source_networks["amira"], inlets, outlets
        )
        _log(
            "Selected distal pressure opening {} at {:.3f} mm Amira "
            "geodesic distance (terminal {}).".format(
                opening_record["opening_terminal_id"],
                opening_record["opening_geodesic_distance_mm"],
                opening_record["node_name"],
            )
        )
    plane_records = _boundary_plane_records(outlets, inlets)
    clearance_samples_by_source = {
        source_name: _centreline_clearance_samples(network, surface_tree)
        for source_name, network in boundary_networks
    }
    total_plane_records = len(plane_records)
    plane_progress_interval = max(1, total_plane_records // 10)
    _log(
        "Validating size and neighbour clearance for {:,} candidate plane(s)..."
        .format(total_plane_records)
    )
    for plane_ordinal, record in enumerate(plane_records, start=1):
        if "plane_geometry_skip_reason" in record:
            record["plane_skip_reason"] = record["plane_geometry_skip_reason"]
            continue
        _set_plane_clearance(
            record,
            clearance_samples_by_source[record["centreline_source"]],
        )
        # Resolve all size conflicts before removing or creating any objects.
        try:
            record["plane_half_width"] = _plane_half_width(record)
        except RuntimeError as exc:
            if not SKIP_UNSAFE_CLEARANCE_TERMINALS:
                raise
            record["plane_skip_reason"] = str(exc)
            _log(
                "WARNING: skipping clearance-unsafe terminal {!r}; no boundary "
                "plane will be created there. {}".format(
                    record["node_name"], exc
                )
            )
        if (
            plane_ordinal % plane_progress_interval == 0
            or plane_ordinal == total_plane_records
        ):
            _log(
                "Terminal-plane validation: {:,}/{:,}.".format(
                    plane_ordinal, total_plane_records
                )
            )
    safe_outlets = [item for item in outlets if "plane_half_width" in item]
    safe_inlets = [item for item in inlets if "plane_half_width" in item]
    safe_opening = None
    if opening_record is not None:
        if opening_record not in safe_outlets:
            raise RuntimeError(
                "The selected distal opening terminal {!r} failed clipping-plane "
                "validation; a pressure-reference boundary is mandatory. {}"
                .format(
                    opening_record["node_name"],
                    opening_record.get("plane_skip_reason", "unknown reason"),
                )
            )
        safe_opening = opening_record
    rejected_refinement_segments = []
    if CREATE_SMALL_VESSEL_REFINEMENTS:
        small_segments = _small_vessel_segments(
            refinement_network, surface_tree, stl_solid=stl_solid
        )
        if REFINEMENT_NEIGHBOUR_AWARE:
            refinement_clearance_samples = _centreline_clearance_samples(
                refinement_network,
                surface_tree,
                spacing=REFINEMENT_CLEARANCE_SAMPLE_SPACING_MM,
                purpose="refinement",
            )
            small_segments, rejected_refinement_segments = (
                _apply_refinement_clearance(
                    small_segments, refinement_clearance_samples
                )
            )
    else:
        small_segments = []

    _log(
        "Terminal nodes: {} ({} inlet candidate(s), {} eligible outlet(s), "
        "{} outlet candidate(s) rejected by boundary filter)".format(
            len(terminals),
            len(inlets),
            len(outlets),
            len(strahler_rejected_outlets),
        )
    )
    for item in sorted(terminals, key=lambda value: value["node_name"]):
        role = (
            "inlet" if item in inlets
            else "reject" if item in strahler_rejected_outlets
            else "outlet"
        )
        plane_text = ""
        if "plane_half_width" in item:
            plane_text = ", plane diameter = {:.3f} mm".format(
                2.0 * item["plane_half_width"]
            )
        elif "plane_skip_reason" in item:
            plane_text = ", PLANE REJECTED ({})".format(
                item["plane_skip_reason"]
            )
        _log(
            "  {:6s} {:30s} [{}] local diameter = {:.3f} mm ({}){}; "
            "surface cross-section diameter = {:.3f} mm ({}); placement {}, "
            "{}-node tangent mean, terminal-tangent cosine {:.3f}, "
            "outward-side cosine {:.3f}{}, surface-axis {}".format(
                role,
                item["node_name"],
                item["centreline_source"],
                2.0 * item["radius"],
                item["radius_source"],
                plane_text,
                2.0 * item.get("surface_cross_section_radius", item["radius"]),
                "measured" if item.get("surface_cross_section_used") else
                "fallback: {}".format(item.get("surface_cross_section_reason", "unknown")),
                (
                    "{:.3f} mm exact-STL ({:.1%} terminal branch retained{})"
                    .format(
                        item["surface_plane_inset_mm"],
                        1.0 - item.get(
                            "surface_plane_removed_branch_fraction", 0.0
                        ),
                        ", short-branch limited" if item.get(
                            "surface_plane_short_branch_limited", False
                        ) else "",
                    )
                    if "surface_plane_inset_mm" in item
                    else "{} centreline nodes inward".format(
                        item["inward_node_count"]
                    )
                ),
                item["tangent_node_count"],
                item.get(
                    "terminal_tangent_cosine",
                    item["normal_outward_cosine"],
                ),
                item["normal_outward_cosine"],
                " (normal flipped)" if item.get("normal_was_flipped") else "",
                (
                    "corrected {:.1f} deg".format(
                        item["surface_axis_correction_degrees"]
                    )
                    if item.get("surface_axis_used")
                    else "unchanged ({})".format(
                        item.get("surface_axis_reason", "disabled")
                    )
                ),
            )
        )
    _log(
        "Small-vessel refinement segments: {} accepted, {} clearance-unsafe "
        "skipped; {} accepted primitive(s) had padding/overlap reduced; {} "
        "use complete exact-STL endpoint sections; {} use an Amira fallback "
        "because at least one STL section radius was implausible.".format(
            len(small_segments),
            len(rejected_refinement_segments),
            sum(item.get("clearance_reduced", False) for item in small_segments),
            sum(
                item.get("surface_radius_measured", False)
                for item in small_segments
            ),
            sum(
                item.get("surface_radius_outlier", False)
                for item in small_segments
            ),
        )
    )
    if small_segments:
        sphere_radii = np.asarray([
            float(item.get(
                "region_radius",
                _desired_refinement_wall_radius(item["radius"]),
            ))
            for item in small_segments
        ])
        _log(
            "Accepted refinement primitive radius range: {:.3f}-{:.3f} mm "
            "(median {:.3f}, 95th percentile {:.3f}).".format(
                float(np.min(sphere_radii)),
                float(np.max(sphere_radii)),
                float(np.median(sphere_radii)),
                float(np.percentile(sphere_radii, 95.0)),
            )
        )
        mesh_sizes = np.asarray([
            _refinement_mesh_size(item) for item in small_segments
        ])
        _log(
            "Accepted refinement mesh-size range: {:.4f}-{:.4f} mm "
            "(n_D={:.3f}, radius-aware={}).".format(
                float(np.min(mesh_sizes)),
                float(np.max(mesh_sizes)),
                float(REFINEMENT_ELEMENTS_ACROSS_DIAMETER),
                bool(REFINEMENT_USE_RADIUS_MESH_SIZE),
            )
        )
    for segment in rejected_refinement_segments:
        _log(
            "  REFINEMENT REJECTED {} s{:03d}: {}".format(
                segment["spline_name"], segment["segment_index"],
                segment["refinement_skip_reason"],
            )
        )
    _log(
        "Safe clipping planes: {} outlet(s), {} inlet(s); {} terminal(s) "
        "skipped for neighbour clearance.".format(
            len(safe_outlets),
            len(safe_inlets) if CREATE_INLET_PLANES else 0,
            sum("plane_skip_reason" in item for item in plane_records),
        )
    )

    # Candidate discovery completes before any existing scripted objects are
    # removed, so a geometry/configuration error leaves the model unchanged.
    _log("Geometry validation complete; updating scripted model objects.")
    _remove_existing(document, model, part)

    created_planes = []
    ordered_outlets = sorted(
        [item for item in safe_outlets if item is not safe_opening],
        key=lambda item: item["node_name"],
    )
    total_outlet_planes = len(ordered_outlets)
    _log("Creating {:,} outlet clipping plane(s)...".format(total_outlet_planes))
    for ordinal, record in enumerate(ordered_outlets, start=1):
        name = "{}{:03d}_{}_{}".format(
            OUTLET_PREFIX,
            ordinal,
            _safe_name(record["centreline_source"]),
            _safe_name(record["node_name"]),
        )
        roi = _create_clipping_plane(document, record, name)
        if ADD_CFD_BOUNDARY_CONDITIONS:
            model.AddCfdSurfaceContact(part, roi, _contact_enum(OUTLET_CONTACT_TYPE))
        try:
            _create_plane_id_annotation(document, record, name, ordinal)
        except Exception as exc:
            _log(
                "WARNING: could not create 3-D ID label P{:03d} for {}: {}"
                .format(ordinal, name, exc)
            )
        _log(
            "CONTACT_MAP {} label=P{:03d} node={} centre=({:.6f},{:.6f},{:.6f}) "
            "normal=({:.6f},{:.6f},{:.6f}) scale=({:.6f},{:.6f}) "
            "normal_source={} cap_deviation_deg={:.3f}".format(
                name,
                ordinal,
                record["node_name"],
                record["centre"][0], record["centre"][1], record["centre"][2],
                record["normal"][0], record["normal"][1], record["normal"][2],
                float(record.get("plane_half_extent_x", record["plane_half_width"])),
                float(record.get("plane_half_extent_y", record["plane_half_width"])),
                record.get("plane_normal_source", "unknown"),
                float(record.get("stl_cap_graph_deviation_degrees", 0.0)),
            )
        )
        created_planes.append(roi)
        if ordinal % 10 == 0 or ordinal == total_outlet_planes:
            _log(
                "Outlet plane creation: {:,}/{:,}.".format(
                    ordinal, total_outlet_planes
                )
            )

    if safe_opening is not None:
        name = "{}001_{}_{}".format(
            OPENING_PREFIX,
            _safe_name(safe_opening["centreline_source"]),
            _safe_name(safe_opening["node_name"]),
        )
        roi = _create_clipping_plane(document, safe_opening, name)
        if ADD_CFD_BOUNDARY_CONDITIONS:
            model.AddCfdSurfaceContact(
                part, roi, _contact_enum(OPENING_CONTACT_TYPE)
            )
        created_planes.append(roi)
        _log(
            "CONTACT_MAP {} role=opening stable_terminal_id={} "
            "geodesic_mm={:.6f} centre=({:.6f},{:.6f},{:.6f}) "
            "normal=({:.6f},{:.6f},{:.6f})".format(
                name,
                safe_opening["opening_terminal_id"],
                safe_opening["opening_geodesic_distance_mm"],
                safe_opening["centre"][0], safe_opening["centre"][1],
                safe_opening["centre"][2], safe_opening["normal"][0],
                safe_opening["normal"][1], safe_opening["normal"][2],
            )
        )

    if CREATE_INLET_PLANES:
        ordered_inlets = sorted(safe_inlets, key=lambda item: item["node_name"])
        total_inlet_planes = len(ordered_inlets)
        _log("Creating {:,} inlet clipping plane(s)...".format(total_inlet_planes))
        for ordinal, record in enumerate(ordered_inlets, start=1):
            name = "{}{:03d}_{}_{}".format(
                INLET_PREFIX,
                ordinal,
                _safe_name(record["centreline_source"]),
                _safe_name(record["node_name"]),
            )
            roi = _create_clipping_plane(document, record, name)
            if ADD_CFD_BOUNDARY_CONDITIONS:
                model.AddCfdSurfaceContact(part, roi, _contact_enum(INLET_CONTACT_TYPE))
            created_planes.append(roi)
            if ordinal % 10 == 0 or ordinal == total_inlet_planes:
                _log(
                    "Inlet plane creation: {:,}/{:,}.".format(
                        ordinal, total_inlet_planes
                    )
                )

    part_vector = PartVector()
    part_vector.append(part)
    refinements = []
    total_refinements = len(small_segments)
    _log(
        "Creating {:,} small-vessel refinement region(s)...".format(
            total_refinements
        )
    )
    for ordinal, segment in enumerate(small_segments, start=1):
        refinements.append(
            _create_refinement(model, part_vector, segment, ordinal)
        )
        if ordinal % 25 == 0 or ordinal == total_refinements:
            _log(
                "Refinement creation: {:,}/{:,}.".format(
                    ordinal, total_refinements
                )
            )

    if UPDATE_GUI_VISIBILITY:
        _log("Updating GUI visibility/highlighting for the created objects...")
        document.ActivateRegionOfInterestVolumes(Doc.Clipping)
        model.SetFeFreeMeshRefinementVolumesVisible(True)
        model.SetFeFreeMeshRefinementSurfacesHighlighted(True)
    else:
        _log("Skipping GUI-only visibility/highlighting in console mode.")

    summary = (
        "Created {} coronary clipping plane(s), {} CFD boundary assignment(s), "
        "and {} small-vessel refinement volume(s)."
    ).format(
        len(created_planes),
        len(created_planes) if ADD_CFD_BOUNDARY_CONDITIONS else 0,
        len(refinements),
    )
    _log(summary)
    document.AddLogComment("Coronary boundary/refinement automation", summary)
    _log(
        "Automation complete in {:.1f} seconds; the caller may now save the "
        "document.".format(time.perf_counter() - _RUN_STARTED_AT)
    )
    # Console study callers need the exact STL-cropped graph used to build the
    # planes/refinements.  Returning plain Python data avoids repeating (or,
    # worse, approximating) the authoritative crop in post-processing.
    cropped_amira = source_networks.get("amira")
    return {
        "created_counts": {
            "inlets": len(safe_inlets) if CREATE_INLET_PLANES else 0,
            "outlets": len(ordered_outlets),
            "openings": 1 if safe_opening is not None else 0,
            "clipping_planes": len(created_planes),
            "refinement_volumes": len(refinements),
            "refinement_candidates": (
                len(small_segments) + len(rejected_refinement_segments)
            ),
            "refinement_rejected": len(rejected_refinement_segments),
        },
        "cropped_amira_graph": None if cropped_amira is None else {
            "nodes": [
                {
                    "node_id": int(node.node_id),
                    "position_mm": [float(v) for v in node.position],
                }
                for node in cropped_amira.nodes
            ],
            "edges": [
                {
                    "edge_id": int(spline.edge_id),
                    "node1": int(spline.start_node.node_id),
                    "node2": int(spline.end_node.node_id),
                    "strahler": None if spline.strahler_order is None else int(spline.strahler_order),
                    "points_mm": [[float(v) for v in point] for point in spline.points],
                    "radii_mm": [float(value) for value in spline.radii],
                }
                for spline in cropped_amira.splines
            ],
        },
    }


if __name__ == "__main__":
    main()
