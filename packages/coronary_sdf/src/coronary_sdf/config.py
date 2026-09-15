"""Pipeline configuration.

Module-level constants are the source of truth and match the names used
in the legacy Coronary_lumen_octree.py so existing patches still apply
verbatim. ``SdfConfig`` is a ``frozen=True`` dataclass that snapshots the
module defaults at construction time. Pass an ``SdfConfig`` into the
pipeline to run with non-default values without mutating the module.

Only SDF-pipeline knobs are kept. HRBF, loft, hybrid, meshsdf hybrid,
hermite hybrid, octree, and dual-contouring knobs are removed.

Each module-level constant below carries a `# Consumed by:` annotation
naming the function(s) in the package that read it. `# UNUSED` means no
consumer outside this file â€” candidate for future cleanup.
"""

from __future__ import annotations
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Iterator

# â”€â”€ Defaults: input/output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Input spatial-graph path. Either a native Avizo/Amira ASCII SpatialGraph
# (.am) or an Excel-XML SpatialGraph export (.xml); parse_amira.parse_xml
# autodetects the format. INPUT_XML / INPUT_AM are kept as back-compat aliases.
# Consumed by: __main__.main
#
# No dataset path is hardcoded: a drive letter from the machine this was written
# on is either wrong elsewhere or, worse, right about the wrong data. Point
# CORONARY_SDF_INPUT at the graph to use, or pass the path on the command line.
INPUT_PATH: str = os.environ.get("CORONARY_SDF_INPUT", "")
# Back-compat aliases for the renamed INPUT_PATH constant.
# Consumed by: epicardial_annotation, flow_fractions, _probe_multifurc
INPUT_XML: str = INPUT_PATH
INPUT_AM: str = INPUT_PATH
# Consumed by: __main__.main, viz.debug_show_mesh
# Defaults to ./coronary_sdf_out under the current directory so a bare run
# writes somewhere obvious and local rather than failing late.
OUTPUT_DIR: str = os.environ.get(
    "CORONARY_SDF_OUTPUT_DIR", str(Path.cwd() / "coronary_sdf_out")
)


def require_input_path(path: str | None = None) -> Path:
    """Return the input graph path, or explain how to supply one.

    Mirrors how ``hipct_seg_debug.edit._deps`` reports what it looked for:
    an unset path should fail immediately with instructions, not halfway
    through a pipeline run.
    """
    chosen = path or INPUT_PATH
    if not chosen:
        raise SystemExit(
            "No input spatial graph given. Pass one on the command line, or set "
            "the CORONARY_SDF_INPUT environment variable to an Avizo/Amira "
            "ASCII SpatialGraph (.am) or Excel-XML (.xml) export."
        )
    p = Path(chosen)
    if not p.is_file():
        raise SystemExit(f"Input spatial graph not found: {p}")
    return p

# Fast mode preset (overrides applied at bottom of file).
# UNUSED â€” only self-referenced by the FAST_MODE override block at the end of this file
FAST_MODE: bool = False

# â”€â”€ General geometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Physical size of one input coordinate unit, in micrometres. The pipeline's
# hardcoded /1000 converts µm â†’ mm; set this to the source-image voxel size when
# the spatial-graph stores voxel indices instead of µm. 1.0 = data already in µm
# (coronary .xml/.am â€” unchanged behaviour).
# Consumed by: parse_amira.parse_xml
INPUT_VOXEL_SIZE_UM: float = 1.0
# Consumed by: sdf_field.compute_grid
PADDING: float = 0.01          # mm â€” empty space around bounding box
# Consumed by: sdf_field.collect_endpoint_info, sdf_field.find_bifurcations,
#              pruning.segment_mean_radius, splines.prepare_segment_spline,
#              topology.label_capsules_by_cross_section, viz.debug_show_raw_data_contours
RADIUS_SCALE: float = 1.0      # multiply all radii (1.0 if thickness=radius)

# Treat measured graph radii as immutable geometry. When enabled, denoising,
# junction targeting, carina tapering and terminal clamping are skipped.
# Consumed by: pipeline.generate_sdf_surface
PRESERVE_INPUT_RADII: bool = True

# â”€â”€ Centerline smoothing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Linearly interpolate extra centerline points + radii along under-sampled
# segments before smoothing / capsule generation. Fixes the case where a
# daughter has so few raw cross-sections that it under-renders and the bif
# appears split.
# Consumed by: pipeline.run_pipeline
DENSIFY_SPARSE_SEGMENTS: bool = True
# Maximum allowed inter-point spacing (mm). Segments coarser than this get
# interpolated.
# Consumed by: smoothing.densify_sparse_segments
DENSIFY_TARGET_SPACING_MM: float = 0.2
# Minimum point count per segment after densification â€” also lifts segments
# above BSPLINE_MIN_PTS so they receive bspline smoothing.
# Consumed by: smoothing.densify_sparse_segments
DENSIFY_MIN_POINTS: int = 40
# Print per-segment decision row from densify_sparse_segments.
# Consumed by: pipeline.run_pipeline, smoothing.densify_sparse_segments
DENSIFY_VERBOSE: bool = False
# Bridge point-less centerline spans (two stored points separated by a large jump
# with no intermediate points, as in some .am exports Avizo draws as a straight
# line) with a curvature-following cubic-Hermite fill, before densify. Gap test
# reuses GAP_BIG_JUMP_RATIO (width multiple) + CENTERLINE_MAX_GAP_UM (absolute
# floor); spacing reuses DENSIFY_TARGET_SPACING_MM.
# Consumed by: pipeline.run_pipeline (via centreline_reconnection.bridge_centerline_gaps)
#
# OFF, and the reason is a measurement rather than a preference. On LADAF-2024-28
# the segmentation is in 55 disconnected components, and `flag-interpolation --seg`
# confirms all 24 of the graph's point-less jumps cross between two of them --
# 72.4 mm in total, a single step reaching 9.8 mm against a median point spacing of
# 93 um. Those jumps are Avizo joining two traced runs, not vessel. Bridging them
# draws lumen through proven background, and the fill is smooth and plausible
# enough that nothing downstream can tell it from the real thing.
BRIDGE_CENTERLINE_GAPS: bool = False
# ...and instead cut the segment there, so the halves become the separate
# components the segmentation says they are. Uses the same two gates the bridge
# used, so exactly the spans that would have been filled are the spans that are cut.
# Consumed by: pipeline.run_pipeline (via centreline_reconnection.split_unsampled_jumps)
SPLIT_UNSAMPLED_JUMPS: bool = True
# Second gate for the cut, alongside GAP_BIG_JUMP_RATIO: a step this many times the
# segment's *own* median spacing is an outlier against how that segment is sampled.
# The width gate alone is radius-scaled and so misses a break in a thin vessel --
# measured here, 18 of the 24 jumps `flag-interpolation --seg` confirms against the
# mask, missing six of 0.6-1.2 mm in vessels 140-230 um across. Matches
# `hipct_seg_debug.edit.interpolation.JUMP_STEP_RATIO`. 0 disables it.
SPLIT_JUMP_STEP_RATIO: float = 5.0
# Then discard any connected component with less centreline than this. Cutting the
# jumps strands whatever was hanging off them; a fragment with no inlet is not a
# vessel the solver can use, and a free-floating tube in the surface is worse than
# an absent one. Set both to 0.0 to keep every fragment.
#
# The fraction is what does the work; the absolute floor is a backstop. Measured on
# LADAF-2024-28 after cutting the 24 jumps: 20 components, of which the two trees
# are 1248 mm and 973 mm (100% and 78% of the largest) and the largest stranded
# fragment is 127 mm (10%). 0.25 sits in the middle of that empty band, so the
# two trees survive and the 18 fragments (458 mm, 17% of the centreline) do not.
# Those fragments are not discarded evidence -- they are the vessels whose
# connection was never imaged, and reconnecting them is the intended future work.
# Consumed by: pipeline.run_pipeline (via centreline_reconnection.drop_small_components)
MIN_COMPONENT_LENGTH_MM: float = 5.0
MIN_COMPONENT_LENGTH_FRACTION: float = 0.25
# Per-segment smoothing-drift diagnostic. Prints adjacent-point drift and
# tangent rotation at each segment endpoint introduced by centerline
# smoothing, with a sparse-vs-dense summary. Used to localise whether
# sparse-after-densification daughters rotate at the bif tangent.
# Consumed by: smoothing.smooth_segment_centerlines
SMOOTH_DRIFT_VERBOSE: bool = True

# Consumed by: smoothing.smooth_centerline, pipeline.generate_sdf_surface
CENTERLINE_SMOOTHER: str = "savgol"  # 'bspline' | 'savgol' | 'none' — bspline scales smoothing with segment length, better for large vessels
# Behaviour when the constrained optimiser cannot satisfy its feasibility
# contract. ``report`` preserves the input and records diagnostics; ``error``
# aborts before an invalid surface can be labelled CFD-ready.
CENTERLINE_CONSTRAINT_FAILURE: str = "report"  # 'report' | 'error'
# Trust region for the constrained multiscale smoother: no point may move more
# than this fraction of its own local radius from the raw centreline. Scale-
# equivariant by construction (the optimiser works in radius-normalised arc
# length). Previously read via getattr with a hardcoded 0.25 fallback, so it
# could not be tuned or recorded in SdfConfig.
#
# 0.05 is chosen from measured MESH outcomes, not from the optimiser's own
# report: its clearance certificate exempts junctions, so it cannot see the
# damage a large step does there and reports success at every setting. Meshed
# self-intersecting pairs (left / right tree, LADAF_2024_28, h=0.12mm):
#     savgol baseline   20 /  52
#     drift 0.02         4 /  44
#     drift 0.05         0 /  35     <- only setting better than savgol on both
#     drift 0.10         0 / 157
#     drift 0.25       178 / 188
# The old 0.25 default was set when a global line-search scalar capped actual
# motion at 0.0044 r, so the value was never exercised; once the per-segment
# line search let the optimiser reach its trust region, 0.25 became harmful.
# Consumed by: centerline_optimizer.smooth_centerlines_constrained_multiscale
CENTERLINE_MAX_DRIFT_RADIUS_FACTOR: float = 0.05
# Consumed by: smoothing.smooth_centerline, smoothing.smooth_centerline_savgol
SAVGOL_WINDOW: int = 13                # odd integer (savgol path)
# Consumed by: smoothing.smooth_centerline, smoothing.smooth_centerline_savgol
SAVGOL_POLYORDER: int = 3
# Consumed by: smoothing.smooth_centerline_bspline
BSPLINE_DEGREE: int = 3                # cubic
# Consumed by: smoothing.smooth_centerline_bspline
BSPLINE_S_PER_POINT: float = 0.005    # mm^2 per point; s = s_per_pt * N — moderate smoothing that preserves vessel clearance
# Consumed by: smoothing.smooth_centerline_bspline
BSPLINE_MIN_PTS: int = 40

# Radius-adaptive B-spline smoothing. s = spp_eff * N, where spp_eff scales
# the per-point residual budget with vessel radius so allowed centreline drift
# stays a constant fraction of radius: thin vessels barely move (no contour/SDF
# overlap), major vessels smooth strongly (no neighbour-contour intersection).
# spp_eff = clip(BSPLINE_S_PER_POINT * (r_med / REF)^POWER, MIN, MAX).
# Mirrors sdf_field.adaptive_smin_k. False = legacy uniform BSPLINE_S_PER_POINT.
# Consumed by: smoothing.adaptive_bspline_s_per_point
BSPLINE_S_ADAPTIVE: bool = True
# Consumed by: smoothing.adaptive_bspline_s_per_point
BSPLINE_S_REF_RADIUS: float = 1.5          # mm â€” spp_eff == BSPLINE_S_PER_POINT at this median radius
# Consumed by: smoothing.adaptive_bspline_s_per_point
BSPLINE_S_RADIUS_POWER: float = 2.5        # s budget ~ radius^2.5 (steeper boost for large vessels)
# Consumed by: smoothing.adaptive_bspline_s_per_point
BSPLINE_S_PER_POINT_MIN: float = 0.0002    # floor â€” thin vessels (~0.014mm RMS drift cap)
# Consumed by: smoothing.adaptive_bspline_s_per_point
BSPLINE_S_PER_POINT_MAX: float = 0.15      # ceiling â€” major vessels (raised so big vessels smooth further)

# Per-point drift cap on the B-spline smoother: no smoothed point may move more
# than D = max(BSPLINE_MAX_DRIFT_MM, FACTOR * local_radius) from the original
# (densified) centreline. Sub-cap noise/waviness corrections pass through; the
# large low-frequency deviation that bows a large vessel's centreline inward is
# clamped, so strong de-noising is kept without the inward bend. FACTOR <= 0
# disables the cap.
# Consumed by: smoothing.smooth_segment_centerlines
BSPLINE_MAX_DRIFT_RADIUS_FACTOR: float = 0.10   # peak drift <= 10% of local radius
# Consumed by: smoothing.smooth_segment_centerlines
BSPLINE_MAX_DRIFT_MM: float = 0.0               # no absolute floor: scale-equivariant

# Kink-safe near-bifurcation protection. At every segment end that meets a
# bifurcation (degree >= 3) node, the heavy endpoint weight is tapered over this
# many points so the B-spline passes through the raw near-junction samples (start/
# end tangent ~ raw, no junction tangent swing), and the curvature pass holds its
# Hermite span this many points clear of the node. Keeps daughters on their raw
# approach so smoothing can't nudge them together at the carina. 0 disables.
# Consumed by: smoothing.smooth_segment_centerlines, smoothing.limit_centerline_curvature
CENTERLINE_BIF_PIN_POINTS: int = 3

# Curvature-limiting pass: where the smoothed centreline's radius of curvature
# R_c drops below the local vessel radius r (kappa*r > 1.0), the swept tube
# self-intersects on the inner side of the bend and the same-segment hard union
# (sdf_field hard_sdf) fuses the two arms, deleting the inner wall. The pass
# fires ONLY on such self-intersecting points and applies the minimum
# straightening (tangent-matched Hermite, blended by the smallest alpha) that
# lifts R_c just past the boundary, capped so the moved body never penetrates a
# non-adjacent branch more than the original did. Runs after
# smooth_segment_centerlines, before capsule build.
# Consumed by: pipeline.generate_sdf_surface, smoothing.limit_centerline_curvature
LIMIT_CENTERLINE_CURVATURE: bool = False
# Detection fires at R_c < r (self-intersection); the pass straightens just to
# R_c >= FACTOR * r, a small margin past the boundary to preserve geometry.
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_MIN_RADIUS_FACTOR: float = 1.05  # correction target margin past R_c = r
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_SMOOTH_WINDOW: int = 8           # half-window (points) smoothed around each violation
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_SMOOTH_MAX_ITERS: int = 24       # max window-widen escalations per segment
# Cap straightening so the moved span never penetrates a non-adjacent branch's
# tube more than the original did (preserves vessel-vessel separation).
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_COLLISION_AWARE: bool = True
# Allowed extra penetration into a non-adjacent tube beyond the original (mm);
# 0 = none. Small => closely preserve original geometry with minimal clearance.
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_COLLISION_MARGIN_MM: float = 0.0
# Per-point displacement cap on the curvature pass: after the minimal-straightening
# blend is chosen, no point may move more than
# D = max(CURVATURE_MAX_DRIFT_MM, CURVATURE_MAX_DRIFT_RADIUS_FACTOR * local_radius)
# from the (B-spline-smoothed) centreline. Bounds the inward bow when a tight bend
# would otherwise drag the whole segment toward its chord. Lower => less bow / more
# residual self-intersection; higher => fixes more bends / more bow. FACTOR <= 0
# disables the cap. Mirrors BSPLINE_MAX_DRIFT_*.
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_MAX_DRIFT_RADIUS_FACTOR: float = 0.25  # user constraint: move <= 25% of local r
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_MAX_DRIFT_MM: float = 0.0              # no absolute floor: scale-equivariant
# Alpha resolution for the minimal-straightening blend (orig -> full Hermite).
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_BLEND_STEPS: int = 40
# Consumed by: smoothing.limit_centerline_curvature
CURVATURE_VERBOSE: bool = True

# Ghost-point tangent extension for the B-spline smoother. Pads each segment
# with virtual points along the local raw tangent so splprep cannot curl the
# last 1-2 points to satisfy a position-only endpoint pin (the cause of
# visible kinks where segments meet at junctions).
# UNUSED â€” no consumer outside config.py
BSPLINE_TANGENT_AVERAGE_PTS: int = 4    # raw points spanned when estimating endpoint tangent
# UNUSED
BSPLINE_GHOST_PTS: int = 2              # virtual points appended beyond each endpoint
# UNUSED
BSPLINE_GHOST_WEIGHT: float = 1.0e3     # splprep weight on ghost points (~ endpoint pin)

# â”€â”€ Radius transition smoothing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: smoothing.smooth_radius_transitions
SMOOTH_RADIUS_TRANSITIONS: bool = True
# Print one line per fired junction: nid, target_r, parent (strahler, r),
# daughter endpoint radii. Helps verify which junctions are being smoothed
# and which are skipped by the threshold.
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_TRANSITION_VERBOSE: bool = False
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_JUMP_THRESHOLD: float = 0.01    # fractional change to trigger
# Consumed by: smoothing.smooth_radius_transitions, centreline_reconnection.merge_degree2_segments
RADIUS_BLEND_POINTS: int = 10          # number of points to blend on either side of a jump
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_BLEND_METHOD: str = "cubic"    # 'linear' | 'cubic' | 'cosine'
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_JUNCTION_TARGET: str = "parent"          # 'weighted_mean' | 'mean' | 'parent'
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_JUNCTION_WEIGHTING: str = "radius"     # 'strahler' | 'radius' | 'equal'
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_JUNCTION_WEIGHT_POWER: float = 1.0
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_ENDPOINT_OUTLIER_RATIO: float = 2.0      # clamp endpoint to +/- ratio of local interior median
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_ENDPOINT_LOOKAHEAD: int = 3             # points inward used for interior median

# When True, the per-segment radius blend at junctions is asymmetric:
# the parent (highest Strahler, tie-break by endpoint radius) retains
# most of its original profile, while daughters fully interpolate to
# target_r. When False, all incident segments use the existing
# symmetric blend.
# Consumed by: smoothing.smooth_radius_transitions
ASYMMETRIC_JUNCTION_BLEND: bool = True
# Mix factor in [0, 1] applied to the parent segment. 0.0 = parent
# completely unchanged; 1.0 = parent gets the full symmetric blend
# (legacy behaviour). 0.2 keeps the parent mostly intact while still
# allowing a small correction at outlier endpoints.
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_JUNCTION_PARENT_BLEND_WEIGHT: float = 0.2
# Mix factor in [0, 1] applied to daughter segments. 1.0 = full blend
# to target_r at the node (legacy behaviour for all segments).
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_JUNCTION_DAUGHTER_BLEND_WEIGHT: float = 1.0

# Anti-neck clamp. target_r is the max-Strahler "parent" radius, which at a
# multifurcation can be THINNER than a thick through-vessel that is merely
# lower-Strahler. Blending that vessel to target_r necks it at the node ->
# the dish indentation on the trunk. When True, each segment's node-side
# target is max(target_r, its own interior radius), so thick vessels keep
# full radius (no dish) while thin daughters still flare up to target_r.
# Consumed by: smoothing.smooth_radius_transitions
RADIUS_JUNCTION_NO_NECK: bool = True

# â”€â”€ In-segment radius smoothing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Savitzky-Golay denoise of r(s) along each segment's interior. Runs before
# the junction-side passes so they see a clean interior; segment endpoints
# are pinned to their original values so the existing junction-blend logic
# is unchanged.
# Consumed by: smoothing.smooth_segment_radii
SMOOTH_SEGMENT_RADII: bool = True
# Consumed by: smoothing.smooth_segment_radii
RADIUS_SAVGOL_WINDOW: int = 101          # odd integer; auto-clipped to segment length
# Consumed by: smoothing.smooth_segment_radii
RADIUS_SAVGOL_POLYORDER: int = 3

# Flow-fraction split radius source. When True, the Giessen outlet split uses the
# raw spatial-graph radii: flow_fractions.smooth_graph skips the four
# radius-smoothing passes (smooth_segment_radii, prune_bifurcation_shrink,
# prune_terminal_shrink, smooth_radius_transitions) while still smoothing
# centerlines and running topology preprocessing. This decouples the outlet
# boundary conditions from surface-generation smoothing parameters. Centerline
# geometry (and therefore mesh-outlet matching) is unaffected.
# Consumed by: flow_fractions.run
FLOW_SPLIT_USE_RAW_RADII: bool = True

# â”€â”€ Radius interpolation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# # UNUSED â€” the active SDF pipeline uses per-capsule linear interpolation
# # (rad_at_t = r0 + t*(r1-r0)) inside sdf_field.evaluate_sdf, and capsule
# # build reads sp["radii"] directly. The cs_rad cubic was a leftover from
# # the removed HRBF / loft / hermite surface generators and is no longer
# # constructed.
RADIUS_INTERPOLATOR: str = "linear"    # 'linear' | 'pchip'

# Terminal contour shrink-pruning (runs alongside radius smoothing).
# When a terminal point's radius is less than this fraction of the next
# more-interior point's radius, clamp it up to the interior value.
# Walks up to MAX_WALK points inward from the terminal.
# Consumed by: smoothing.prune_terminal_shrink
PRUNE_TERMINAL_SHRINK: bool = True
# Consumed by: smoothing.prune_terminal_shrink
TERMINAL_SHRINK_THRESHOLD: float = 0.75
# Consumed by: smoothing.prune_terminal_shrink
TERMINAL_SHRINK_MAX_WALK: int = 4
# Consumed by: smoothing.prune_terminal_shrink
TERMINAL_SHRINK_LOOKAHEAD: int = 3       # median of next-N interior radii as reference

# Capsule-layer terminal radius clamp (independent of the threshold-based
# prune_terminal_shrink above). For each spline whose endpoint is a
# degree-1 (terminal) node, force radii[0] or radii[-1] up to the median
# of the next CLAMP_LOOKAHEAD interior radii â€” unconditional, no
# threshold. Stops side-branch tips from rendering as a sharp point in
# the SDF iso-surface.
# Consumed by: capsules.clamp_terminal_capsule_radii,
#              pipeline.generate_sdf_surface
FORCE_TERMINAL_CAPSULE_NO_SHRINK: bool = True
# Consumed by: capsules.clamp_terminal_capsule_radii
TERMINAL_CAPSULE_CLAMP_LOOKAHEAD: int = 3
# Print per-endpoint detail (seg_id, node_id, before/after) for every
# terminal endpoint considered by clamp_terminal_capsule_radii. Helps
# identify which terminals still taper after the clamp.
# Consumed by: pipeline.generate_sdf_surface
TERMINAL_CLAMP_VERBOSE: bool = False

# Bifurcation-side contour shrink pruning. Symmetric to the terminal
# pruner but applied to segment endpoints at degree>=3 nodes. Each
# offending point is clamped to the median of the next LOOKAHEAD
# interior radii â€” robust to multi-point noisy stretches at the bif.
# Consumed by: smoothing.prune_bifurcation_shrink
PRUNE_BIFURCATION_SHRINK: bool = True
# Consumed by: smoothing.prune_bifurcation_shrink
BIFURCATION_SHRINK_THRESHOLD: float = 0.5
# Consumed by: smoothing.prune_bifurcation_shrink
BIFURCATION_SHRINK_MAX_WALK: int = 5
# Consumed by: smoothing.prune_bifurcation_shrink
BIFURCATION_SHRINK_LOOKAHEAD: int = 3

# Number of consecutive stable points required before the shrink walk halts.
# 1 = stop on first stable point (legacy). 2+ = tolerate single-point noise
# inside the shrink cluster (e.g. [0.10, 0.12, 0.10, 0.5] is fully repaired).
# Consumed by: smoothing.prune_terminal_shrink, smoothing.prune_bifurcation_shrink
SHRINK_STOP_AFTER_STABLE: int = 2

# â”€â”€ Bifurcation carina taper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Taper the last K radii of each spline endpoint that terminates at a deg>=3
# node DOWN to a small "carina tip" radius. Converts each hemispherical
# capsule end-cap into a conical tip; N cones converging at a bif node
# smooth-min into a clean Y/T/X carina instead of a ball-shaped union of N
# hemispheres. Terminal (deg==1) endpoints are untouched.
# Consumed by: bif_trim.taper_bifurcation_carina, pipeline.generate_sdf_surface
BIF_CARINA_ENABLE: bool = False
# Carina tip radius as a fraction of the interior reference radius. 0.1 = tip
# is 10% of local vessel radius â€” small enough that the residual hemisphere
# at the apex is sub-voxel for typical resolutions.
# Consumed by: bif_trim.taper_bifurcation_carina
BIF_CARINA_TIP_RADIUS_FACTOR: float = 0.1
# Absolute floor on the carina tip radius (mm). Keeps the capsule SDF
# numerically well-behaved (no zero-radius singularity) while staying small
# enough that the apex bulge is sub-voxel.
# Consumed by: bif_trim.taper_bifurcation_carina
BIF_CARINA_TIP_MIN_MM: float = 0.02
# Maximum points to taper per endpoint. Caps the taper region so a long
# is_junction run (XS_JUNC_NODE_PROXIMITY_FACTOR up to ~4*r) doesn't pull
# the cone too far back into the parent vessel. K = min(is_junction_run,
# BIF_CARINA_TAPER_MAX_PTS).
# Consumed by: bif_trim.taper_bifurcation_carina
BIF_CARINA_TAPER_MAX_PTS: int = 6
# Minimum is_junction run required to fire the taper at an endpoint. Below
# this, the cross-section overlap is too shallow to justify the radii edit.
# Consumed by: bif_trim.taper_bifurcation_carina
BIF_CARINA_TAPER_MIN_PTS: int = 2
# Per-endpoint taper records to stdout (mirrors TERMINAL_CLAMP_VERBOSE).
# Consumed by: pipeline.generate_sdf_surface
BIF_CARINA_VERBOSE: bool = False

# Apply a planar truncation at bif-incident capsule endpoints (degree>=3
# nodes), replacing the hemispherical end-cap with a flat plane through
# the capsule endpoint. Eliminates the spherical bulge that N hemispheres
# converging at a bif would otherwise produce, without narrowing the
# daughter's cross-section the way carina taper does. Applies per-capsule
# in the SDF evaluation, so adjacent capsules at the same bif don't punch
# holes into one another.
# Consumed by: sdf_field.evaluate_sdf
SDF_FLAT_CAP_BIF: bool = True

# Width of the flat-cap cylinder->plane transition band, as a fraction of the
# local capsule radius. The cap carves the daughter inward over this band; a
# hardcoded 2-voxel band makes that a sharp recessed ring at the ostium on
# thick vessels (the "indentation"). Radius-scaling spreads it into a smooth
# shoulder. Effective half-band h_soft = max(voxel_size, radius * factor);
# the transition spans 2*h_soft inside the cylinder up to the bif plane.
# 0.0 = legacy hardcoded voxel-sized band.
# Consumed by: sdf_field.evaluate_sdf
SDF_FLAT_CAP_BIF_SOFT_FACTOR: float = 0.0   # reset to baseline; band width had no effect on the ostium dish

# Optional shift (fraction of local radius) of the flat-cap plane PAST the bif
# node, so the daughter extends slightly into the parent before truncating and
# fills the ostium valley. 0.0 = truncate exactly at the node (legacy). Keep
# small (~0.2-0.4); a large shift re-grows the bif bulge flat-cap removes.
# Consumed by: sdf_field.evaluate_sdf
SDF_FLAT_CAP_BIF_SHIFT_FACTOR: float = 0.0

# â”€â”€ Tree pruning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: pipeline.generate_sdf_surface
MIN_STRAHLER_ORDER: int = 1            # keep segments with Strahler >= this
# Consumed by: pruning.prune_by_radius
MIN_SEGMENT_LENGTH: float = 0.0        # mm

# â”€â”€ Input geometry validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Detect (but do not repair) segments whose centerline has a single dominant
# gap â€” interior points collapsed at one endpoint / large internal jump. Such
# degenerate polylines (seen in some resampled .am exports) fragment the SDF
# surfaces; this gate flags them right after parse so bad inputs are caught
# before the expensive surface generation. Consumed by: pipeline.run_pipeline
# (via parse_amira.find_degenerate_segments).
VALIDATE_CENTERLINE_GEOMETRY: bool = True
CENTERLINE_MAX_GAP_UM: float = 500.0      # Âµm; ignore gaps below this absolute size
CENTERLINE_GAP_RATIO: float = 0.5         # flag if one step exceeds this fraction of arc length
CENTERLINE_VALIDATION_MODE: str = "warn"  # "off" | "warn" | "error"

# Print-only attribution of the visible "disconnections" in the raw-contour
# debug viz (debug_show_raw_data_contours). Classifies every visible ring gap as
# a genuine data gap (flagged degenerate segment), a bifurcation joint, or a
# thin/coarse-sampling rendering artifact -- so we can tell whether on-screen
# breaks are real geometry or just the hoops-without-a-centerline rendering.
# Consumed by: pipeline.generate_sdf_surface (via
# parse_amira.report_ring_gap_attribution). Read-only; mutates no geometry and
# runs independently of DEBUG_VIS.
REPORT_RING_GAP_ATTRIBUTION: bool = True
GAP_VIS_ALPHA: float = 1.0       # a gap is "visible" when step > alpha*(r_i + r_j)
GAP_BIG_JUMP_RATIO: float = 5.0  # gap >= this many vessel-widths -> real jump, not thin sampling

# UNUSED
MIN_RADIUS: float = 0.0                # mm
# UNUSED
PRUNE_BY_RADIUS: bool = False
# Consumed by: pruning.prune_by_radius
PRUNE_MIN_RADIUS: float = 0.15         # mm â€” used iff PRUNE_BY_RADIUS
# Consumed by: sdf_field.build_adjacency, smoothing.prune_terminal_shrink,
#              smoothing.prune_bifurcation_shrink
NODE_COINCIDENCE_EPS_MM: float = 0.01   # merge near-coincident nodes
# Consumed by: pipeline.generate_sdf_surface
MERGE_DEGREE2_SEGMENTS: bool = True    # contract degree-2 pass-through nodes
# Collapse a bif-bif segment when the Strahler signature matches a split
# multifurcation (connector at parent's Strahler order, all other daughters
# at lower Strahler) AND length/tangent guards pass. Catches the
# trifurcation-as-two-Y-bifs artefact without merging real adjacent bifs.
# Consumed by: pipeline.generate_sdf_surface
MERGE_SPLIT_MULTIFURCATIONS: bool = True
# Length threshold (multiple of MIN endpoint radius) â€” conservative.
# Consumed by: pipeline.generate_sdf_surface
SPLIT_MULTIFURC_MAX_LEN_FACTOR: float = 0.6
# Require the Strahler discriminator. False = length-only (debug aid).
# Consumed by: pipeline.generate_sdf_surface
SPLIT_MULTIFURC_REQUIRE_STRAHLER: bool = True
# Minimum |cos(angle)| between connector tangent and trunk continuation.
# Consumed by: pipeline.generate_sdf_surface
SPLIT_MULTIFURC_TANGENT_COS_MIN: float = 0.7
# When True, print every per-candidate decision record.
# Consumed by: pipeline.generate_sdf_surface
BIF_MERGE_VERBOSE: bool = False
# Consumed by: pipeline.generate_sdf_surface
PRUNE_SHORT_TERMINAL_NUBS: bool = False
# Consumed by: pipeline.generate_sdf_surface, pruning.prune_short_terminal_nubs
MIN_TERMINAL_LENGTH_MM: float = 1.5
# Consumed by: pipeline.generate_sdf_surface, pruning.prune_short_terminal_nubs
PRUNE_ITER_MAX: int = 1
# Consumed by: pruning.prune_short_terminal_nubs
STUB_SEGMENT_MAX_LENGTH_MM: float = 1.0   # report-only

# â”€â”€ Capsule sampling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# # UNUSED â€” capsule sampling is driven by smoothed centerline point count;
# # these knobs only fed the rolled-back build_capsules_with_trim
CAPSULE_SAMPLE_FACTOR: float = 0.5     # sample at local_radius * this
# # UNUSED
CAPSULE_MIN_SAMPLE: float = 0.01       # mm
# # UNUSED
CAPSULE_MAX_SAMPLE: float = 0.1        # mm
# Consumed by: splines.prepare_segment_spline
CAPSULE_SAMPLE_STRIDE_PCT: float = 0.01      # stride = round(N * pct / 100)
# Consumed by: splines.prepare_segment_spline
CAPSULE_SAMPLE_STRIDE_MAX_PCT: float = 5.0 # stride must stay < max_pct of N

# â”€â”€ Smooth-min blending â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# # Consumed by: sdf_field.smooth_min_exp
SMIN_K_DEFAULT: float = 6.0            # legacy (octree fast path)
# # UNUSED
SMIN_ADAPTIVE: bool = True             # legacy
# Consumed by: sdf_field.adaptive_smin_k, sdf_field.evaluate_sdf,
#              pipeline.generate_sdf_surface, __main__.main
BSPLINE_SMIN_K: float = 6.0           # active log-sum-exp k
# Consumed by: sdf_field.adaptive_smin_k, sdf_field.evaluate_sdf
SMIN_ADAPTIVE_BLEND: bool = True
# Consumed by: sdf_field.adaptive_smin_k, sdf_field.evaluate_sdf
SMIN_K_REF_RADIUS: float = 1.5    # May 21 working value â€” k=BSPLINE_SMIN_K at this radius
# Consumed by: sdf_field.adaptive_smin_k, sdf_field.evaluate_sdf
SMIN_K_MIN: float = 6.0                # lower clip on k â€” flat-cap removed bif balls so moderate sharpness OK
# Consumed by: sdf_field.adaptive_smin_k, sdf_field.evaluate_sdf
SMIN_K_MAX: float = 50.0               # upper clip on k â€” May 21 working value
# Consumed by: sdf_field.build_narrow_band, sdf_field.evaluate_sdf
SMIN_PROXIMITY_BLEND_FACTOR: float = 0.4  # wider blend zone fades the smin raise over a longer arc -> smoother smin<->hard transition (probe: crease ~4x lower vs 0.2)
# Consumed by: sdf_field.evaluate_sdf
BLEND_BULGE_CAP_MM: float = 0.03       # smaller smin raise -> less to fade at the smin<->hard boundary; flat-cap still closes geometry

# Soft-knee width (mm) for the bulge-cap saturation. The hard
# np.minimum(depression, cap) injects a C1 slope break exactly where the
# cap engages â€” the visible "crease ring" at bif ostia. With knee > 0 the
# cap corner is rounded over this width (reuses the polynomial smin), so
# the fillet saturates smoothly. 0.0 = legacy hard clamp.
# Consumed by: sdf_field.evaluate_sdf
BLEND_BULGE_CAP_SOFT_KNEE_MM: float = 0.05   # â‰ˆ cap â†’ gentle knee, fix on by default

# Optional radius-scaling of the cap so larger vessels get a fuller, smoother
# fillet. Effective cap = max(BLEND_BULGE_CAP_MM, owner_radius * factor).
# 0.0 = pure absolute mm (legacy). Raise (e.g. 0.05-0.1) if ostia still read
# as too sharp after the soft knee removes the crease.
# Consumed by: sdf_field.evaluate_sdf
BLEND_BULGE_CAP_RADIUS_FACTOR: float = 0.0

# Half-thickness (mm) of the band around the intersection curve where
# smin is allowed to fire. The blend gate requires BOTH |hard_sdf| and
# |rival_sdf| to be < this value â€” i.e. the voxel must lie within the
# band of both the owner-segment surface AND at least one adjacent
# rival's surface. 0.0 = off (legacy: rely on proximity-gap gate alone).
# Consumed by: sdf_field.evaluate_sdf
SMIN_INTERSECTION_BAND_MM: float = 1.0  # wide enough to bridge typical ostium gaps
# Consumed by: sdf_field.evaluate_sdf
SMOOTH_MIN_INTERIOR_ONLY: bool = False
# Consumed by: sdf_field.evaluate_sdf, pipeline.generate_sdf_surface
USE_CROSS_SECTION_BLEND_GATE: bool = False
# # UNUSED
SKIP_PROJECTION_IN_JUNCTIONS: bool = True

# â”€â”€ Smooth-min variant â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 'logsumexp' (default, N-ary, global support, controlled by BSPLINE_SMIN_K +
# BLEND_BULGE_CAP_MM cap) or 'polynomial' (textbook 2-ary iquilezles smin,
# compact support, controlled by SMIN_POLY_K_FACTOR). Polynomial gives a
# cleaner geometrically-explicit fillet at 2-way bifs but leaves a small
# seam at 3+-way bifurcations (rare in coronary anatomy: e.g. LAD/LCx/RI).
# Consumed by: sdf_field.evaluate_sdf
SMIN_VARIANT: str = "logsumexp"  # 'logsumexp' | 'polynomial' â€” N-ary handles non-planar multifurcations symmetrically (polynomial 2-ary drops 3rd+ rivals)

# Polynomial smin 'k' as a multiplier on min(owner_radius, rival_radius).
# k = factor * min(r_owner, r_rival). Fillet depth at the carina = k / 4.
# Factor 0.8 â‰ˆ 0.2 * r_smaller dip for a 2-way bif, matching the
# sdf_plan.md target. Only used when SMIN_VARIANT == "polynomial".
# Consumed by: sdf_field.evaluate_sdf
SMIN_POLY_K_FACTOR: float = 0.8

# Selects the gate that controls where smooth-min fires:
#   "multi_gate"   - legacy stack: proximity-gap smoothstep ANDed with the
#                    cross-section is_junction flag, the Euclidean bif-ball,
#                    the parallel-rival suppression, and the wedge gate.
#                    Empirically tuned, addresses parallel-sidebranch
#                    bridges and unrelated-vessel near-misses.
#   "t_projection" - sdf_plan.md target: a single fade
#                    (1 - clamp(t_relevant/2, 0, 1))^2 driven by the
#                    segment-level normalised t at the owner and rival
#                    closest points. t_relevant depends on the relation
#                    (parent: t_owner; child: t_rival; sibling: max).
#                    Requires the directed topology (parent/child/sibling
#                    masks) and the per-capsule arc annotations.
# Consumed by: sdf_field.evaluate_sdf
SMIN_GATE_VARIANT: str = "multi_gate"   # May 21 working recipe â€” proximity-gap smoothstep Ã— gates

# Diagnostic mode: force sdf_final = hard_sdf by zeroing blend_w at every
# voxel right before the smooth/hard mix. Smooth-min still runs (so diag
# counters / arrays stay consistent), but its contribution to the final
# SDF is zeroed. Use to A/B test whether smooth-min is the cause of an
# observed bridge / bulge / fusion.
# Consumed by: sdf_field.evaluate_sdf
FORCE_HARD_MIN_ONLY: bool = False

# Topology-aware variant of SDF_CARVE_PROTECT_JUNCTION_BALL. When True,
# the in_jball shield relaxes to fire ONLY when the offending rival
# (the wall-band rival for the wall-band carve, the non-adjacent rival
# for the deep-merger carve) is itself one of the segments incident on
# the nearest bif. Lets carves break accidental fusion between
# unrelated sub-trees that happen to pass close to a bif, while still
# preserving the carina of the bif's own siblings/parent.
# Requires bif_seg_incident to be plumbed into evaluate_sdf; falls back
# to the legacy geometric shield when bif_seg_incident is None.
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_PROTECT_JUNCTION_BALL_TOPO_AWARE: bool = True

# â”€â”€ Bifurcation-localised blending (legacy fast path) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# # UNUSED â€” legacy octree fast-path, superseded by adaptive smooth-min
BIFURCATION_BLEND_ONLY: bool = True
# # UNUSED
BIFURCATION_BLEND_RADIUS: float = 1  # x shared-node radius
# # UNUSED
BIF_PARALLEL_ANGLE_THRESH: float = 90.0
# # UNUSED
BIF_PARALLEL_BLEND_FACTOR: float = 0.5

# â”€â”€ Bif-ostium blending gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Hard outer boundary on where smin engages: a voxel must lie within
# BIF_BLEND_RADIUS_FACTOR * r_bif of a degree>=3 node for blend_w to be
# non-zero. ANDed with the existing cross-section gate. Set False to keep
# the cross-section gate as the sole blend region selector.
# Consumed by: sdf_field.evaluate_sdf
BIF_BLEND_ENABLE: bool = False   # binary 1.25*r_bif ball mask produces blend_w step â†’ surface roughness; smoothstep intersection-curve gate replaces it

# Multiplier on the bif node's representative radius. 2.5 matches the
# existing SDF_CARVE_JUNCTION_PROTECT_FACTOR for consistency; tighten
# (e.g. 1.5) to shrink the blend zone and eliminate opposite-side bulges
# on the parent wall across from a daughter ostium.
# Consumed by: sdf_field.evaluate_sdf
BIF_BLEND_RADIUS_FACTOR: float = 1.25

# â”€â”€ Parallel-adjacent-rival blend gate (anti-bridge for sidebranches) â”€
# Near-parallel adjacent rivals (e.g. a short sidebranch almost parallel
# to its parent main branch) cause smooth-min to fire along the entire
# shared run, creating a bridge between the two surfaces. This gate
# short-circuits the blend (blend_w := 0) for such rivals when the voxel
# is outside a tight ball around the shared bif node, so the carina
# fillet survives but the parallel run does not bridge.
# Consumed by: sdf_field.evaluate_sdf
BLEND_PARALLEL_RIVAL_GATE_ENABLE: bool = True
# Lowered from 0.90 -> 0.70: sibling daughters at a 20-40 degree bif
# angle have cos ~0.77-0.94, so 0.90 was filtering out most of them.
# 0.70 catches all sibling daughters that run roughly together past the
# bif; non-parallel adjacent rivals (different sub-tree direction) still
# fall through.
# Consumed by: sdf_field.evaluate_sdf
BLEND_PARALLEL_COS_THRESHOLD: float = 0.7
# Multiplier on r_bif. Lowered from 1.0 -> 0.3 so the carve fires inside
# the bif region (where sibling-daughter bridges live) while the carina
# itself within 0.3*r_bif of the actual bif node stays protected.
# Consumed by: sdf_field.evaluate_sdf
BLEND_PARALLEL_BIF_PROTECT_FACTOR: float = 0.3
# Consumed by: sdf_field.evaluate_sdf
BIF_WEDGE_GATE_ENABLE: bool = False    # May 21 working recipe: no wedge gate â€” bif-ball + cos threshold suffice

# Smoothstep half-width for the wedge half-space ramp, in mm. The
# binary "dot_owner >= 0 AND dot_rival >= 0" wedge is replaced by a
# product of two smoothsteps that ramp from 0 at dot = -WEDGE_RAMP_MM
# to 1 at dot = +WEDGE_RAMP_MM. Larger = softer / more diffuse wedge.
# Set to 0 to fall back to the binary mask.
# Consumed by: sdf_field.evaluate_sdf
BIF_WEDGE_RAMP_MM: float = 0.1     # absolute floor; effective ramp = max(floor, r * BIF_WEDGE_RAMP_RADIUS_FACTOR)

# Radius-scaling for BIF_WEDGE_RAMP_MM. Effective ramp =
# max(BIF_WEDGE_RAMP_MM, min(owner_r, rival_r) * BIF_WEDGE_RAMP_RADIUS_FACTOR).
# 0.0 = pure absolute mm (legacy). 0.5 = ramp grows linearly with the
# thinner of the two vessels, with the absolute floor still respected.
# Consumed by: sdf_field.evaluate_sdf
BIF_WEDGE_RAMP_RADIUS_FACTOR: float = 0.5

# Smoothstep ramp width for the in_tight_bif boundary, as a multiple of
# r_bif. The binary "d < FACTOR * r_bif" boundary at r = FACTOR * r_bif
# is replaced by a smoothstep from "fully inside" at d = FACTOR*r to
# "fully outside" at d = (FACTOR + RAMP_FACTOR)*r. Set to 0 to fall back
# to the binary mask.
# Consumed by: sdf_field.evaluate_sdf
BLEND_PARALLEL_BIF_PROTECT_RAMP_FACTOR: float = 0.4

# â”€â”€ Parallel-adjacent-rival SDF carve (anti-bridge, hard-min path) â”€â”€â”€â”€
# Symmetric to the patch-81..87 SDF_CARVE_NON_ADJACENT path, but for
# adjacent rivals (parent + sidebranch share a bif node). Fires only
# where the adjacent rival is near-parallel AND we are outside the tight
# bif-ball, so the carina fillet at real bifurcations is preserved.
# Reuses BLEND_PARALLEL_COS_THRESHOLD and BLEND_PARALLEL_BIF_PROTECT_FACTOR.
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_ADJACENT_PARALLEL: bool = False   # May 21 working recipe: unified wall-band carve handles siblings

# â”€â”€ SDF grid + narrow band â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: sdf_field.compute_grid, pipeline.generate_sdf_surface
BSPLINE_SDF_RESOLUTION: float | None = None       # mm â€” voxel size; None = auto
# Upper bound on the dense grid. compute_grid takes the COARSEST of
# min_radius/2.5, (bbox_volume / this)^(1/3) and DENSE_MIN_SPACING_MM. At the
# old 2e8 the bbox term won on a whole coronary tree (0.120 mm vs the 0.052 mm
# the radius rule asked for), leaving the thinnest vessels at R/h ~ 1 where the
# marching-cubes radius deficit is several percent. The bbox is a poor proxy
# for cost -- the narrow band is only ~3% of the grid -- so the cap is set from
# available memory and the backends' addressing limit instead. MeshLib's
# SimpleVolume and VTK's ImageData index voxels with a signed 32-bit type, so
# 2**31 is a hard ceiling no amount of RAM lifts; 2e9 sits just under it and is
# ~8 GB float32 plus a similar MeshLib copy. compute_grid coarsens and warns if
# a grid still lands past the ceiling.
# Consumed by: sdf_field.compute_grid
BSPLINE_SDF_MAX_VOXELS: int = 2_000_000_000

# Number of nearest capsules (by midpoint) considered per voxel in the SDF
# eval. Too small at a THICK junction (many ~0.2mm densified capsules + branch
# capsules crowd within ~R of a surface voxel) drops the trunk capsule that is
# actually nearest, so hard_sdf overestimates distance and the surface caves
# inward = a dish at thick bifurcations. 16 was too few; raise for correctness.
# Consumed by: sdf_field.evaluate_sdf
SDF_MAX_CAPSULE_QUERY: int = 64

# Pre-compensate the capsule radii for the marching-cubes discretisation
# deficit. Extracting a curved iso-surface on a finite grid places it slightly
# inside the true surface; the error is second order in the voxel size,
# delta_r ~ -COEFF * h^2 / r, so it matters only on the thinnest vessels (a few
# percent at R/h ~ 2.5, negligible above R/h ~ 10). Inflating each capsule by
# +COEFF * h^2 / r before evaluation cancels the leading term.
# COEFF must be measured for a CYLINDER through this pipeline -- a cylinder has
# half a sphere's curvature, so the sphere-fit value does not transfer.
# Consumed by: capsules.precompensate_capsule_radii, pipeline.generate_sdf_surface
SDF_RADIUS_PRECOMPENSATE: bool = False
# Consumed by: capsules.precompensate_capsule_radii
SDF_RADIUS_PRECOMPENSATE_COEFF: float = 0.0

# â”€â”€ Cross-section junction labelling (patch 84) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: topology.label_capsules_by_cross_section, _smoke_test
XS_JUNC_NODE_PROXIMITY_FACTOR: float = 3.0   # May 21 working value

# â”€â”€ Anti-bridge carve (patches 81-87) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: sdf_field.evaluate_sdf, sdf_field.report_non_adjacent_proximity
SDF_CARVE_NON_ADJACENT: bool = False
# For rivals in non-adjacent segments, carve any SDF voxel whose center is
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_WALL_BAND_FACTOR: float = 0.7    # May 21 working value â€” wide band catches sibling-parallel overlap
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_HONOR_JUNCTION_GATE: bool = False      # diag showed gate kills 99.93% of wall-band candidates on this dataset â€” disable to let carve actually fire
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_PROTECT_JUNCTION_BALL: bool = False     # patch 86
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_JUNCTION_PROTECT_FACTOR: float = 1.5  # May 21 working value â€” protect carina from the wider wall-band carve
# Consumed by: sdf_field.evaluate_sdf, _smoke_test
SDF_CARVE_PARALLEL_COS_THRESHOLD: float = 0.7   # match BLEND_PARALLEL_COS_THRESHOLD so ~45deg rivals still carve
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_TANGENT_BUFFER_VOXELS: float = 0.0     # extra tangent-touch carve

# Patch 88: non-adjacent deep-merger carve. Pushes SDF outward at any
# voxel that lies inside a non-adjacent rival's capsule, even if the
# rival surface is well past the wall band (i.e. deep overlap that the
# patch-81..87 wall-band carve rejects). Honors the junction gate and
# bif-ball protection so it never disturbs adjacent-segment bif blending.
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_NON_ADJACENT_DEEP: bool = True
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_DEEP_MIN_OVERLAP_MM: float = 0.0        # fire on any positive overlap
# Consumed by: sdf_field.evaluate_sdf
SDF_CARVE_DEEP_BUFFER_VOXELS: float = 1.0        # push 1 voxel past the midpoint

# Last-resort: unconditionally force SDF positive (by this margin in mm)
# at any voxel where non_adj_min < 0 â€” i.e. anywhere a non-adjacent rival
# is genuinely inside the voxel. Bypasses every gate including the
# protect ball. 0.0 = off. Use only when SDF_CARVE_NON_ADJACENT_DEEP +
# its buffer still leave a bridge. Risks puncturing legitimate bif
# geometry; raise gradually from 0.0.
# Consumed by: sdf_field.evaluate_sdf
FORCE_NON_ADJ_GAP_MM: float = 0.0

# â”€â”€ Mesh-level anti-bridge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: pipeline.generate_sdf_surface
MESH_CUT_NON_ADJACENT_BRIDGES: bool = False
# Consumed by: mesh_extract.cut_non_adjacent_bridges
MESH_BRIDGE_CUT_HOLE_FILL: bool = True

# â”€â”€ SDF post-processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: sdf_field.report_non_adjacent_proximity
SDF_FIELD_METHOD: str = "legacy"  # 'legacy' | 'graph_implicit'
# What a failing connected component does to the rest of the run.
# ``error`` (default) preserves the historical fail-fast behaviour for the
# compatibility and CFD profiles; ``continue`` reconstructs every remaining
# component and records the failure in the manifest instead. The manifest is
# written either way, so a partial output directory is never ambiguous.
# Consumed by: pipeline._run_pipeline
PIPELINE_COMPONENT_FAILURE: str = "error"  # 'error' | 'continue'
# How the legacy point oracle decides a cell cannot contain the zero set.
# ``geometric`` uses a hard-union distance certificate; ``none`` refines every
# cell to its target size, which is slower but cannot hide a hole and is the
# reference used to prove the certificate sound.
# Consumed by: pipeline._generate_sdf_surface, legacy_field_adapter.LegacyPointField
LEGACY_ORACLE_PRUNE_MODE: str = "geometric"  # 'geometric' | 'none'
# Absolute dense-grid floor retained by the compatibility profile. Set to
# ``None`` for scale-equivariant synthetic/reference runs. ``None`` is now the
# default: at 0.08 this term became the binding one once BSPLINE_SDF_MAX_VOXELS
# was raised, capping the resolution improvement short of min_radius/2.5.
DENSE_MIN_SPACING_MM: float | None = None
# Primitive used by GraphImplicitField. ``radial`` is the historical
# sign-correct tubular field; ``round_cone`` is the exact Euclidean distance to
# the linearly tapered union-of-balls primitive.
IMPLICIT_PRIMITIVE_METHOD: str = "radial"  # 'radial' | 'round_cone'
IMPLICIT_JUNCTION_BLEND_FRACTION: float = 0.15
IMPLICIT_JUNCTION_SUPPORT_FACTOR: float = 4.0
IMPLICIT_BVH_LEAF_SIZE: int = 8
IMPLICIT_CELLS_ACROSS_DIAMETER: float = 12.0
IMPLICIT_ADAPTIVE_AUDIT: bool = False
IMPLICIT_ADAPTIVE_MAX_DEPTH: int = 30
IMPLICIT_ADAPTIVE_MAX_LEAVES: int = 5_000_000
IMPLICIT_ADAPTIVE_MAX_POINTS: int = 500_000
IMPLICIT_GEOMETRY_CHECK: bool = True
IMPLICIT_GEOMETRY_CLEARANCE_FRACTION: float = 0.0
IMPLICIT_GEOMETRY_MAX_REPORT: int = 20
IMPLICIT_FAIL_ON_GEOMETRY_CONFLICT: bool = False
# VTK HyperTreeGrid adaptive contour settings. The voxel strategy is the only
# strategy that passed strict manifold validation; decomposed polyhedra remain
# available solely as an explicit diagnostic ablation.
VTK_HTG_PADDING_RADIUS_FACTOR: float = 2.0
VTK_HTG_MAX_CELLS: int = 5_000_000
VTK_HTG_DECOMPOSED_POLYHEDRA: bool = False
# Optional native CGAL criteria. These are inert unless the explicitly
# requested mesh method is ``cgal_mesh3``.
CGAL_FACET_ANGLE_DEG: float = 30.0
CGAL_FACET_DISTANCE_FRACTION: float = 0.25
CGAL_CELL_SIZE_FACTOR: float = 2.0
CGAL_CELL_RADIUS_EDGE_RATIO: float = 2.0
CGAL_SEQUENTIAL: bool = True
CGAL_REQUIRED_VERSION: str = "6.2"
CGAL_NATIVE_API_VERSION: int = 3
CGAL_EXTENSION_BUILD_VERSION: str = "0.3.0"
# Common validation policy used by every extraction backend. The historical
# default stays non-breaking; the benchmark/CFD profile uses ``error``.
OUTPUT_VALIDATION_MODE: str = "warn"  # 'off' | 'warn' | 'error'
OUTPUT_VALIDATE_SELF_INTERSECTIONS: bool = True
SDF_GAUSSIAN_SIGMA_VOXELS: float = 0.0          # 0 = off
# Consumed by: pipeline.generate_sdf_surface
SDF_KEEP_LARGEST_ONLY: bool = False
# Drop connected components below this fraction of total surface area. Marching
# cubes leaves sub-voxel dust that dominates the component count (measured: only
# 3 of 27-65 components exceed 0.01% of area) and makes beta0 meaningless as a
# topology check. 1e-4 removes the dust while keeping every substantive piece --
# including the genuine ~1.1%-of-area second component this tree reproducibly
# contains, which SDF_KEEP_LARGEST_ONLY would discard. 0.0 disables.
# Consumed by: pipeline.generate_sdf_surface
MESH_MIN_COMPONENT_AREA_FRACTION: float = 1.0e-4
# UNUSED
DISABLE_REFINEMENT_FOR_DEBUG: bool = True
# UNUSED
MULTIRES_REFINE: bool = False
# Consumed by: viz.debug_show_capsule_tubes
MULTIRES_VOXEL_FACTOR: float = 5.0
# UNUSED
MULTIRES_FINE_FACTOR: float = 10.0
# UNUSED
ADAPTIVE_REFINE_CIRCUMF_TARGET: int = 128
# UNUSED
ADAPTIVE_REFINE_MAX_ITERS: int = 1

# â”€â”€ Mesh extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 'vtk_htg' (adaptive VTK HyperTreeGrid), 'adaptive' (experimental SciPy),
# 'cgal_mesh3' (optional native CGAL), 'poisson' (Open3D), 'meshlib'
# (MeshLib), or 'mc' (VTK FlyingEdges).
# Consumed by: mesh_extract.extract_isosurface, pipeline.generate_sdf_surface,
#              __main__.main, _smoke_test
SDF_MESH_METHOD: str = "meshlib"
# Consumed by: mesh_extract.mesh_from_sdf_poisson, _smoke_test
SDF_POISSON_DEPTH: int = 12
# Consumed by: mesh_extract.mesh_from_sdf_poisson
SDF_POISSON_DENSITY_QUANTILE: float = 0.00
# Volume-preserving relaxation applied to the whole extracted surface. These
# were previously dead (mesh_from_sdf_meshlib hardcoded 30 iterations at force
# 0.5 and extract_isosurface never forwarded them). relaxKeepVolume diverges
# where the local volume has collapsed, emitting NaN vertices; the guarded
# chunked runner in mesh_extract reverts such a chunk, and a lower iteration
# count keeps it well clear of that regime.
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
SDF_MESHLIB_RELAX_ITERS: int = 5
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
SDF_MESHLIB_RELAX_FORCE: float = 0.4
# Isotropic remesh target edge as a multiple of the SDF voxel size. The field
# carries no detail below one voxel, so a factor < 1 adds no geometry and only
# manufactures the sliver triangles that make the relaxation diverge.
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
MESHLIB_TARGET_EDGE_FACTOR: float = 1.25
# Hard floor on the thin-vessel refine target edge, as a multiple of the voxel
# size. Stops the circumference-driven target from resolving far below the grid.
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
THIN_VESSEL_MIN_EDGE_FACTOR: float = 0.5
# Fraction of mesh vertices allowed to be non-finite before the component is
# treated as failed rather than repaired. drop_nonfinite_vertices removes every
# face touching a bad vertex, so a large fraction silently shatters the lumen
# into many components; that must be an error, not a warning.
# Consumed by: pipeline.generate_sdf_surface
MESH_MAX_NONFINITE_FRACTION: float = 0.001

# â”€â”€ Flat caps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: mesh_repair.create_flat_caps, pipeline.generate_sdf_surface
SDF_FLAT_TERMINAL_CAPS: bool = True
# Consumed by: sdf_field.evaluate_sdf, mesh_repair.create_flat_caps
SDF_FLAT_CAP_REACH_FACTOR: float = 1.5
# UNUSED
OPEN_OUTLETS: bool = False
# Consumed by: pipeline.generate_sdf_surface
FLAT_CAP_OUTLETS: bool = False
# Consumed by: pipeline.generate_sdf_surface
FLAT_CAP_REQUIRE_MANIFOLD: bool = True
# Consumed by: pipeline.generate_sdf_surface
FLAT_CAP_MAX_FACES: int = 20_000_000

# â”€â”€ Mesh repair â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: pipeline.generate_sdf_surface
MESH_REPAIR: bool = False              # patch 88: raw Poisson by default
# Consumed by: mesh_repair.repair_mesh
MESH_FILL_HOLES: bool = True
# Consumed by: mesh_repair.repair_mesh
MESH_REMOVE_DEGENERATE: bool = True
# Consumed by: mesh_repair.repair_mesh
MESH_FIX_NORMALS: bool = True
# Consumed by: mesh_repair.repair_mesh, mesh_repair.run_pymeshfix
USE_PYMESHFIX: bool = True
# Consumed by: mesh_repair.run_pymeshfix, mesh_repair.repair_mesh
PYMESHFIX_TIMEOUT_SEC: int = 30
# Consumed by: mesh_repair.run_pymeshfix
PYMESHFIX_MAX_FACES: int = 30_000_000

# â”€â”€ Mesh smoothing (post-extract) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Localized MeshLib remesh + volume-preserving relax of thin vessels, whose
# small circumference gets too few edges at the SDF voxel size -> faceted
# cross-sections. Region-restricted (conforming, no cracks); thick vessels and
# junctions untouched.
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
# Native MeshLib region pass inside mesh_from_sdf_meshlib (no lossy round-trip;
# verified to preserve face count and localize to the thin region).
THIN_VESSEL_REFINE: bool = True
# Refine vessels with local radius below this (mm).
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
THIN_VESSEL_REFINE_RADIUS_MM: float = 0.2
# Target number of edges around the circumference in refined thin vessels.
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
THIN_VESSEL_CIRCUMF_TARGET: int = 20
# Volume-preserving relax iterations/force over the refined thin region.
# Disabled (0) by default. relaxKeepVolume holds each neighbourhood's volume
# fixed by rescaling; restricted to a thin-vessel REGION, whose enclosed volume
# is near zero, that scale factor diverges and every vertex in the region comes
# back non-finite. Measured on 5 of 5 components across three configurations
# (30 iters/force 0.5 and 5 iters/force 0.4, thin edge floors h*0.2 and h*0.5):
# it failed within the first 5 iterations every time and the guard reverted it
# to zero, so it has never contributed geometry -- only cost. The unrestricted
# global relax is unaffected and stays on. Plain (non-volume-preserving) relax
# is not a substitute here: it shrinks thin vessels, which is the radius bias
# this pipeline is trying to remove.
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
THIN_VESSEL_REFINE_RELAX_ITERS: int = 0
# Consumed by: mesh_extract.mesh_from_sdf_meshlib
THIN_VESSEL_REFINE_RELAX_FORCE: float = 0.4

# Volume-preserving (VTK smooth_taubin) post-extract polish. With
# TAUBIN_JUNCTION_ONLY it is feathered to the bifurcation regions to round off
# the residual smin<->hard ridge without touching the rest of the vessel.
# Consumed by: mesh_repair.radius_constrained_taubin
TAUBIN_ITERS: int = 20                  # 0 = off
# Consumed by: mesh_repair.radius_constrained_taubin
TAUBIN_BAND: float = 0.1
# Per-vertex Taubin displacement clamp as a fraction of the local vessel radius.
# Consumed by: mesh_repair.radius_constrained_taubin
TAUBIN_MAX_DISP_FACTOR: float = 0.2
# Restrict the Taubin smoothing to a feathered ball around each bifurcation
# node (only polish the junction ridge; leave the rest of the surface exact).
# Consumed by: mesh_repair.radius_constrained_taubin
TAUBIN_JUNCTION_ONLY: bool = True
# Full smoothing within this multiple of the bif node radius.
# Consumed by: mesh_repair.radius_constrained_taubin
TAUBIN_JUNCTION_FACTOR: float = 2.0
# Ramp the smoothing weight to zero over this further multiple of r_bif.
# Consumed by: mesh_repair.radius_constrained_taubin
TAUBIN_JUNCTION_FEATHER: float = 1.0
# UNUSED
SUBDIVIDE_ITERS: int = 3

# â”€â”€ Diagnostics / viz â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consumed by: sdf_field.evaluate_sdf, pipeline.generate_sdf_surface
BLEND_DIAGNOSTIC: bool = False
# Per-narrow-band diagnostic arrays (gates / carves / smin / non-adjacent
# proximity / topology / saddle-protect). Independent of BLEND_DIAGNOSTIC.
# When True, evaluate_sdf populates SdfVolume.diag and viz.debug_show_blend_
# diagnostics overlays toggleable layers on the mesh for debugging
# bifurcation bulges and vessel-fusion bridges.
# Memory: 13 arrays at ~50 MB total for a typical narrow band.
# Consumed by: sdf_field.evaluate_sdf, pipeline.generate_sdf_surface,
#              viz.debug_show_blend_diagnostics
DETAILED_BLEND_DIAGNOSTIC: bool = False
# Consumed by: pipeline.generate_sdf_surface
PREVIEW_SDF_BEFORE_MC: bool = True
# Consumed by: pipeline.generate_sdf_surface
WRITE_REGION_VTK: bool = True
# Consumed by: pipeline.generate_sdf_surface
PROXIMITY_DIAGNOSTIC_TOP_N: int = 0  # 0 = off; rank non-adjacent rivals by proximity and write the top N closest to VTK for debugging
# Consumed by: pipeline.generate_sdf_surface, viz.debug_show_* (all)
DEBUG_VIS: bool = True
# Consumed by: viz._show_plotter
DEBUG_VIS_BLOCK: bool = True
# Consumed by: viz._show_plotter
DEBUG_VIS_SAVE_FALLBACK: bool = True
# Consumed by: viz.debug_show_capsule_tree, debug_show_raw_data_contours,
#              debug_show_smoothed_centerlines
DEBUG_VIS_COLOR_BY: str = "strahler"   # "segment" | "radius" | "strahler"

# â”€â”€ Fast mode overrides â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”— 
if FAST_MODE:
    BSPLINE_SDF_RESOLUTION = 0.15
    TAUBIN_ITERS = 12
    MESH_REPAIR = True


# â”€â”€ Dataclass wrapper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



@dataclass(frozen=True)
class SdfConfig:
    """Frozen snapshot of pipeline configuration.

    Defaults are seeded from the module-level constants above. Construct
    with keyword overrides for a non-default run:

        cfg = SdfConfig(SDF_CARVE_PARALLEL_COS_THRESHOLD=0.5,
                        SDF_POISSON_DEPTH=11)
    """

    # I/O
    INPUT_PATH: str = INPUT_PATH
    INPUT_XML: str = INPUT_XML    # back-compat alias of INPUT_PATH
    INPUT_AM: str = INPUT_AM      # back-compat alias of INPUT_PATH
    OUTPUT_DIR: str = OUTPUT_DIR

    # general
    INPUT_VOXEL_SIZE_UM: float = INPUT_VOXEL_SIZE_UM
    FAST_MODE: bool = FAST_MODE
    PADDING: float = PADDING
    RADIUS_SCALE: float = RADIUS_SCALE
    PRESERVE_INPUT_RADII: bool = PRESERVE_INPUT_RADII

    # centerline smoothing
    DENSIFY_SPARSE_SEGMENTS: bool = DENSIFY_SPARSE_SEGMENTS
    DENSIFY_TARGET_SPACING_MM: float = DENSIFY_TARGET_SPACING_MM
    DENSIFY_MIN_POINTS: int = DENSIFY_MIN_POINTS
    DENSIFY_VERBOSE: bool = DENSIFY_VERBOSE
    BRIDGE_CENTERLINE_GAPS: bool = BRIDGE_CENTERLINE_GAPS
    SPLIT_UNSAMPLED_JUMPS: bool = SPLIT_UNSAMPLED_JUMPS
    SPLIT_JUMP_STEP_RATIO: float = SPLIT_JUMP_STEP_RATIO
    MIN_COMPONENT_LENGTH_MM: float = MIN_COMPONENT_LENGTH_MM
    MIN_COMPONENT_LENGTH_FRACTION: float = MIN_COMPONENT_LENGTH_FRACTION
    SMOOTH_DRIFT_VERBOSE: bool = SMOOTH_DRIFT_VERBOSE
    CENTERLINE_SMOOTHER: str = CENTERLINE_SMOOTHER
    CENTERLINE_CONSTRAINT_FAILURE: str = CENTERLINE_CONSTRAINT_FAILURE
    CENTERLINE_MAX_DRIFT_RADIUS_FACTOR: float = CENTERLINE_MAX_DRIFT_RADIUS_FACTOR
    SAVGOL_WINDOW: int = SAVGOL_WINDOW
    SAVGOL_POLYORDER: int = SAVGOL_POLYORDER
    BSPLINE_DEGREE: int = BSPLINE_DEGREE
    BSPLINE_S_PER_POINT: float = BSPLINE_S_PER_POINT
    BSPLINE_MIN_PTS: int = BSPLINE_MIN_PTS
    BSPLINE_TANGENT_AVERAGE_PTS: int = BSPLINE_TANGENT_AVERAGE_PTS
    BSPLINE_GHOST_PTS: int = BSPLINE_GHOST_PTS
    BSPLINE_GHOST_WEIGHT: float = BSPLINE_GHOST_WEIGHT
    BSPLINE_S_ADAPTIVE: bool = BSPLINE_S_ADAPTIVE
    BSPLINE_S_REF_RADIUS: float = BSPLINE_S_REF_RADIUS
    BSPLINE_S_RADIUS_POWER: float = BSPLINE_S_RADIUS_POWER
    BSPLINE_S_PER_POINT_MIN: float = BSPLINE_S_PER_POINT_MIN
    BSPLINE_S_PER_POINT_MAX: float = BSPLINE_S_PER_POINT_MAX
    BSPLINE_MAX_DRIFT_RADIUS_FACTOR: float = BSPLINE_MAX_DRIFT_RADIUS_FACTOR
    BSPLINE_MAX_DRIFT_MM: float = BSPLINE_MAX_DRIFT_MM
    CENTERLINE_BIF_PIN_POINTS: int = CENTERLINE_BIF_PIN_POINTS
    LIMIT_CENTERLINE_CURVATURE: bool = LIMIT_CENTERLINE_CURVATURE
    CURVATURE_MIN_RADIUS_FACTOR: float = CURVATURE_MIN_RADIUS_FACTOR
    CURVATURE_SMOOTH_WINDOW: int = CURVATURE_SMOOTH_WINDOW
    CURVATURE_SMOOTH_MAX_ITERS: int = CURVATURE_SMOOTH_MAX_ITERS
    CURVATURE_COLLISION_AWARE: bool = CURVATURE_COLLISION_AWARE
    CURVATURE_COLLISION_MARGIN_MM: float = CURVATURE_COLLISION_MARGIN_MM
    CURVATURE_MAX_DRIFT_RADIUS_FACTOR: float = CURVATURE_MAX_DRIFT_RADIUS_FACTOR
    CURVATURE_MAX_DRIFT_MM: float = CURVATURE_MAX_DRIFT_MM
    CURVATURE_BLEND_STEPS: int = CURVATURE_BLEND_STEPS
    CURVATURE_VERBOSE: bool = CURVATURE_VERBOSE

    # radius
    SMOOTH_RADIUS_TRANSITIONS: bool = SMOOTH_RADIUS_TRANSITIONS
    RADIUS_JUMP_THRESHOLD: float = RADIUS_JUMP_THRESHOLD
    RADIUS_BLEND_POINTS: int = RADIUS_BLEND_POINTS
    RADIUS_BLEND_METHOD: str = RADIUS_BLEND_METHOD
    RADIUS_JUNCTION_TARGET: str = RADIUS_JUNCTION_TARGET
    RADIUS_JUNCTION_NO_NECK: bool = RADIUS_JUNCTION_NO_NECK
    RADIUS_JUNCTION_WEIGHTING: str = RADIUS_JUNCTION_WEIGHTING
    RADIUS_JUNCTION_WEIGHT_POWER: float = RADIUS_JUNCTION_WEIGHT_POWER
    ASYMMETRIC_JUNCTION_BLEND: bool = ASYMMETRIC_JUNCTION_BLEND
    RADIUS_ENDPOINT_OUTLIER_RATIO: float = RADIUS_ENDPOINT_OUTLIER_RATIO
    RADIUS_ENDPOINT_LOOKAHEAD: int = RADIUS_ENDPOINT_LOOKAHEAD
    RADIUS_INTERPOLATOR: str = RADIUS_INTERPOLATOR
    SMOOTH_SEGMENT_RADII: bool = SMOOTH_SEGMENT_RADII
    RADIUS_SAVGOL_WINDOW: int = RADIUS_SAVGOL_WINDOW
    RADIUS_SAVGOL_POLYORDER: int = RADIUS_SAVGOL_POLYORDER
    FLOW_SPLIT_USE_RAW_RADII: bool = FLOW_SPLIT_USE_RAW_RADII
    RADIUS_JUNCTION_PARENT_BLEND_WEIGHT: float = RADIUS_JUNCTION_PARENT_BLEND_WEIGHT
    RADIUS_JUNCTION_DAUGHTER_BLEND_WEIGHT: float = RADIUS_JUNCTION_DAUGHTER_BLEND_WEIGHT
    RADIUS_TRANSITION_VERBOSE: bool = RADIUS_TRANSITION_VERBOSE
    PRUNE_TERMINAL_SHRINK: bool = PRUNE_TERMINAL_SHRINK
    TERMINAL_SHRINK_THRESHOLD: float = TERMINAL_SHRINK_THRESHOLD
    TERMINAL_SHRINK_LOOKAHEAD: int = TERMINAL_SHRINK_LOOKAHEAD
    TERMINAL_SHRINK_MAX_WALK: int = TERMINAL_SHRINK_MAX_WALK
    PRUNE_BIFURCATION_SHRINK: bool = PRUNE_BIFURCATION_SHRINK
    BIFURCATION_SHRINK_THRESHOLD: float = BIFURCATION_SHRINK_THRESHOLD
    BIFURCATION_SHRINK_LOOKAHEAD: int = BIFURCATION_SHRINK_LOOKAHEAD
    BIFURCATION_SHRINK_MAX_WALK: int = BIFURCATION_SHRINK_MAX_WALK
    SHRINK_STOP_AFTER_STABLE: int = SHRINK_STOP_AFTER_STABLE

    # pruning / topology
    MIN_STRAHLER_ORDER: int = MIN_STRAHLER_ORDER
    MIN_SEGMENT_LENGTH: float = MIN_SEGMENT_LENGTH

    # input geometry validation
    VALIDATE_CENTERLINE_GEOMETRY: bool = VALIDATE_CENTERLINE_GEOMETRY
    CENTERLINE_MAX_GAP_UM: float = CENTERLINE_MAX_GAP_UM
    CENTERLINE_GAP_RATIO: float = CENTERLINE_GAP_RATIO
    CENTERLINE_VALIDATION_MODE: str = CENTERLINE_VALIDATION_MODE
    REPORT_RING_GAP_ATTRIBUTION: bool = REPORT_RING_GAP_ATTRIBUTION
    GAP_VIS_ALPHA: float = GAP_VIS_ALPHA
    GAP_BIG_JUMP_RATIO: float = GAP_BIG_JUMP_RATIO
    MIN_RADIUS: float = MIN_RADIUS
    PRUNE_BY_RADIUS: bool = PRUNE_BY_RADIUS
    PRUNE_MIN_RADIUS: float = PRUNE_MIN_RADIUS
    NODE_COINCIDENCE_EPS_MM: float = NODE_COINCIDENCE_EPS_MM
    MERGE_DEGREE2_SEGMENTS: bool = MERGE_DEGREE2_SEGMENTS
    MERGE_SPLIT_MULTIFURCATIONS: bool = MERGE_SPLIT_MULTIFURCATIONS
    SPLIT_MULTIFURC_MAX_LEN_FACTOR: float = SPLIT_MULTIFURC_MAX_LEN_FACTOR
    SPLIT_MULTIFURC_REQUIRE_STRAHLER: bool = SPLIT_MULTIFURC_REQUIRE_STRAHLER
    SPLIT_MULTIFURC_TANGENT_COS_MIN: float = SPLIT_MULTIFURC_TANGENT_COS_MIN
    BIF_MERGE_VERBOSE: bool = BIF_MERGE_VERBOSE
    PRUNE_SHORT_TERMINAL_NUBS: bool = PRUNE_SHORT_TERMINAL_NUBS
    MIN_TERMINAL_LENGTH_MM: float = MIN_TERMINAL_LENGTH_MM
    PRUNE_ITER_MAX: int = PRUNE_ITER_MAX
    STUB_SEGMENT_MAX_LENGTH_MM: float = STUB_SEGMENT_MAX_LENGTH_MM

    # bif carina taper
    BIF_CARINA_ENABLE: bool = BIF_CARINA_ENABLE
    BIF_CARINA_TIP_RADIUS_FACTOR: float = BIF_CARINA_TIP_RADIUS_FACTOR
    BIF_CARINA_TIP_MIN_MM: float = BIF_CARINA_TIP_MIN_MM
    BIF_CARINA_TAPER_MAX_PTS: int = BIF_CARINA_TAPER_MAX_PTS
    BIF_CARINA_TAPER_MIN_PTS: int = BIF_CARINA_TAPER_MIN_PTS
    BIF_CARINA_VERBOSE: bool = BIF_CARINA_VERBOSE

    # bif flat-cap (geometric alternative to carina taper)
    SDF_FLAT_CAP_BIF: bool = SDF_FLAT_CAP_BIF
    SDF_FLAT_CAP_BIF_SOFT_FACTOR: float = SDF_FLAT_CAP_BIF_SOFT_FACTOR
    SDF_FLAT_CAP_BIF_SHIFT_FACTOR: float = SDF_FLAT_CAP_BIF_SHIFT_FACTOR

    # capsule sampling
    CAPSULE_SAMPLE_FACTOR: float = CAPSULE_SAMPLE_FACTOR
    CAPSULE_MIN_SAMPLE: float = CAPSULE_MIN_SAMPLE
    CAPSULE_MAX_SAMPLE: float = CAPSULE_MAX_SAMPLE
    CAPSULE_SAMPLE_STRIDE_PCT: float = CAPSULE_SAMPLE_STRIDE_PCT
    CAPSULE_SAMPLE_STRIDE_MAX_PCT: float = CAPSULE_SAMPLE_STRIDE_MAX_PCT
    FORCE_TERMINAL_CAPSULE_NO_SHRINK: bool = FORCE_TERMINAL_CAPSULE_NO_SHRINK
    TERMINAL_CAPSULE_CLAMP_LOOKAHEAD: int = TERMINAL_CAPSULE_CLAMP_LOOKAHEAD
    TERMINAL_CLAMP_VERBOSE: bool = TERMINAL_CLAMP_VERBOSE

    # smooth-min
    SMIN_K_DEFAULT: float = SMIN_K_DEFAULT
    SMIN_ADAPTIVE: bool = SMIN_ADAPTIVE
    BSPLINE_SMIN_K: float = BSPLINE_SMIN_K
    SMIN_ADAPTIVE_BLEND: bool = SMIN_ADAPTIVE_BLEND
    SMIN_K_REF_RADIUS: float = SMIN_K_REF_RADIUS
    SMIN_K_MIN: float = SMIN_K_MIN
    SMIN_K_MAX: float = SMIN_K_MAX
    SMIN_PROXIMITY_BLEND_FACTOR: float = SMIN_PROXIMITY_BLEND_FACTOR
    BLEND_BULGE_CAP_MM: float = BLEND_BULGE_CAP_MM
    BLEND_BULGE_CAP_SOFT_KNEE_MM: float = BLEND_BULGE_CAP_SOFT_KNEE_MM
    BLEND_BULGE_CAP_RADIUS_FACTOR: float = BLEND_BULGE_CAP_RADIUS_FACTOR
    SMIN_INTERSECTION_BAND_MM: float = SMIN_INTERSECTION_BAND_MM
    SMOOTH_MIN_INTERIOR_ONLY: bool = SMOOTH_MIN_INTERIOR_ONLY
    USE_CROSS_SECTION_BLEND_GATE: bool = USE_CROSS_SECTION_BLEND_GATE
    SKIP_PROJECTION_IN_JUNCTIONS: bool = SKIP_PROJECTION_IN_JUNCTIONS

    # smin variant
    SMIN_VARIANT: str = SMIN_VARIANT
    SMIN_POLY_K_FACTOR: float = SMIN_POLY_K_FACTOR
    SMIN_GATE_VARIANT: str = SMIN_GATE_VARIANT
    FORCE_HARD_MIN_ONLY: bool = FORCE_HARD_MIN_ONLY
    SDF_CARVE_PROTECT_JUNCTION_BALL_TOPO_AWARE: bool = SDF_CARVE_PROTECT_JUNCTION_BALL_TOPO_AWARE

    # bif blending (legacy)
    BIFURCATION_BLEND_ONLY: bool = BIFURCATION_BLEND_ONLY
    BIFURCATION_BLEND_RADIUS: float = BIFURCATION_BLEND_RADIUS
    BIF_PARALLEL_ANGLE_THRESH: float = BIF_PARALLEL_ANGLE_THRESH
    BIF_PARALLEL_BLEND_FACTOR: float = BIF_PARALLEL_BLEND_FACTOR

    # bif-ostium blend gate
    BIF_BLEND_ENABLE: bool = BIF_BLEND_ENABLE
    BIF_BLEND_RADIUS_FACTOR: float = BIF_BLEND_RADIUS_FACTOR

    # parallel-adjacent-rival blend gate
    BLEND_PARALLEL_RIVAL_GATE_ENABLE: bool = BLEND_PARALLEL_RIVAL_GATE_ENABLE
    BLEND_PARALLEL_COS_THRESHOLD: float = BLEND_PARALLEL_COS_THRESHOLD
    BLEND_PARALLEL_BIF_PROTECT_FACTOR: float = BLEND_PARALLEL_BIF_PROTECT_FACTOR
    BIF_WEDGE_GATE_ENABLE: bool = BIF_WEDGE_GATE_ENABLE
    BIF_WEDGE_RAMP_MM: float = BIF_WEDGE_RAMP_MM
    BIF_WEDGE_RAMP_RADIUS_FACTOR: float = BIF_WEDGE_RAMP_RADIUS_FACTOR
    BLEND_PARALLEL_BIF_PROTECT_RAMP_FACTOR: float = BLEND_PARALLEL_BIF_PROTECT_RAMP_FACTOR

    # parallel-adjacent-rival SDF carve (hard-min path)
    SDF_CARVE_ADJACENT_PARALLEL: bool = SDF_CARVE_ADJACENT_PARALLEL

    # SDF grid
    BSPLINE_SDF_RESOLUTION: float | None = BSPLINE_SDF_RESOLUTION
    BSPLINE_SDF_MAX_VOXELS: int = BSPLINE_SDF_MAX_VOXELS
    SDF_MAX_CAPSULE_QUERY: int = SDF_MAX_CAPSULE_QUERY
    SDF_RADIUS_PRECOMPENSATE: bool = SDF_RADIUS_PRECOMPENSATE
    SDF_RADIUS_PRECOMPENSATE_COEFF: float = SDF_RADIUS_PRECOMPENSATE_COEFF

    # junction labelling
    XS_JUNC_NODE_PROXIMITY_FACTOR: float = XS_JUNC_NODE_PROXIMITY_FACTOR

    # anti-bridge carve
    SDF_CARVE_NON_ADJACENT: bool = SDF_CARVE_NON_ADJACENT
    SDF_CARVE_WALL_BAND_FACTOR: float = SDF_CARVE_WALL_BAND_FACTOR
    SDF_CARVE_HONOR_JUNCTION_GATE: bool = SDF_CARVE_HONOR_JUNCTION_GATE
    SDF_CARVE_PROTECT_JUNCTION_BALL: bool = SDF_CARVE_PROTECT_JUNCTION_BALL
    SDF_CARVE_JUNCTION_PROTECT_FACTOR: float = SDF_CARVE_JUNCTION_PROTECT_FACTOR
    SDF_CARVE_PARALLEL_COS_THRESHOLD: float = SDF_CARVE_PARALLEL_COS_THRESHOLD
    SDF_CARVE_TANGENT_BUFFER_VOXELS: float = SDF_CARVE_TANGENT_BUFFER_VOXELS
    SDF_CARVE_NON_ADJACENT_DEEP: bool = SDF_CARVE_NON_ADJACENT_DEEP
    SDF_CARVE_DEEP_MIN_OVERLAP_MM: float = SDF_CARVE_DEEP_MIN_OVERLAP_MM
    SDF_CARVE_DEEP_BUFFER_VOXELS: float = SDF_CARVE_DEEP_BUFFER_VOXELS
    FORCE_NON_ADJ_GAP_MM: float = FORCE_NON_ADJ_GAP_MM

    # mesh-level bridge cut
    MESH_CUT_NON_ADJACENT_BRIDGES: bool = MESH_CUT_NON_ADJACENT_BRIDGES
    MESH_BRIDGE_CUT_HOLE_FILL: bool = MESH_BRIDGE_CUT_HOLE_FILL

    # SDF post-processing
    SDF_FIELD_METHOD: str = SDF_FIELD_METHOD
    PIPELINE_COMPONENT_FAILURE: str = PIPELINE_COMPONENT_FAILURE
    LEGACY_ORACLE_PRUNE_MODE: str = LEGACY_ORACLE_PRUNE_MODE
    DENSE_MIN_SPACING_MM: float | None = DENSE_MIN_SPACING_MM
    IMPLICIT_PRIMITIVE_METHOD: str = IMPLICIT_PRIMITIVE_METHOD
    IMPLICIT_JUNCTION_BLEND_FRACTION: float = IMPLICIT_JUNCTION_BLEND_FRACTION
    IMPLICIT_JUNCTION_SUPPORT_FACTOR: float = IMPLICIT_JUNCTION_SUPPORT_FACTOR
    IMPLICIT_BVH_LEAF_SIZE: int = IMPLICIT_BVH_LEAF_SIZE
    IMPLICIT_CELLS_ACROSS_DIAMETER: float = IMPLICIT_CELLS_ACROSS_DIAMETER
    IMPLICIT_ADAPTIVE_AUDIT: bool = IMPLICIT_ADAPTIVE_AUDIT
    IMPLICIT_ADAPTIVE_MAX_DEPTH: int = IMPLICIT_ADAPTIVE_MAX_DEPTH
    IMPLICIT_ADAPTIVE_MAX_LEAVES: int = IMPLICIT_ADAPTIVE_MAX_LEAVES
    IMPLICIT_ADAPTIVE_MAX_POINTS: int = IMPLICIT_ADAPTIVE_MAX_POINTS
    IMPLICIT_GEOMETRY_CHECK: bool = IMPLICIT_GEOMETRY_CHECK
    IMPLICIT_GEOMETRY_CLEARANCE_FRACTION: float = IMPLICIT_GEOMETRY_CLEARANCE_FRACTION
    IMPLICIT_GEOMETRY_MAX_REPORT: int = IMPLICIT_GEOMETRY_MAX_REPORT
    IMPLICIT_FAIL_ON_GEOMETRY_CONFLICT: bool = IMPLICIT_FAIL_ON_GEOMETRY_CONFLICT
    VTK_HTG_PADDING_RADIUS_FACTOR: float = VTK_HTG_PADDING_RADIUS_FACTOR
    VTK_HTG_MAX_CELLS: int = VTK_HTG_MAX_CELLS
    VTK_HTG_DECOMPOSED_POLYHEDRA: bool = VTK_HTG_DECOMPOSED_POLYHEDRA
    CGAL_FACET_ANGLE_DEG: float = CGAL_FACET_ANGLE_DEG
    CGAL_FACET_DISTANCE_FRACTION: float = CGAL_FACET_DISTANCE_FRACTION
    CGAL_CELL_SIZE_FACTOR: float = CGAL_CELL_SIZE_FACTOR
    CGAL_CELL_RADIUS_EDGE_RATIO: float = CGAL_CELL_RADIUS_EDGE_RATIO
    CGAL_SEQUENTIAL: bool = CGAL_SEQUENTIAL
    CGAL_REQUIRED_VERSION: str = CGAL_REQUIRED_VERSION
    CGAL_NATIVE_API_VERSION: int = CGAL_NATIVE_API_VERSION
    CGAL_EXTENSION_BUILD_VERSION: str = CGAL_EXTENSION_BUILD_VERSION
    OUTPUT_VALIDATION_MODE: str = OUTPUT_VALIDATION_MODE
    OUTPUT_VALIDATE_SELF_INTERSECTIONS: bool = OUTPUT_VALIDATE_SELF_INTERSECTIONS
    SDF_GAUSSIAN_SIGMA_VOXELS: float = SDF_GAUSSIAN_SIGMA_VOXELS
    SDF_KEEP_LARGEST_ONLY: bool = SDF_KEEP_LARGEST_ONLY
    MESH_MIN_COMPONENT_AREA_FRACTION: float = MESH_MIN_COMPONENT_AREA_FRACTION
    DISABLE_REFINEMENT_FOR_DEBUG: bool = DISABLE_REFINEMENT_FOR_DEBUG
    MULTIRES_REFINE: bool = MULTIRES_REFINE
    MULTIRES_VOXEL_FACTOR: float = MULTIRES_VOXEL_FACTOR
    MULTIRES_FINE_FACTOR: float = MULTIRES_FINE_FACTOR
    ADAPTIVE_REFINE_CIRCUMF_TARGET: int = ADAPTIVE_REFINE_CIRCUMF_TARGET
    ADAPTIVE_REFINE_MAX_ITERS: int = ADAPTIVE_REFINE_MAX_ITERS

    # mesh extraction
    SDF_MESH_METHOD: str = SDF_MESH_METHOD
    SDF_POISSON_DEPTH: int = SDF_POISSON_DEPTH
    SDF_POISSON_DENSITY_QUANTILE: float = SDF_POISSON_DENSITY_QUANTILE
    SDF_MESHLIB_RELAX_ITERS: int = SDF_MESHLIB_RELAX_ITERS
    SDF_MESHLIB_RELAX_FORCE: float = SDF_MESHLIB_RELAX_FORCE
    MESHLIB_TARGET_EDGE_FACTOR: float = MESHLIB_TARGET_EDGE_FACTOR
    MESH_MAX_NONFINITE_FRACTION: float = MESH_MAX_NONFINITE_FRACTION

    # flat caps
    SDF_FLAT_TERMINAL_CAPS: bool = SDF_FLAT_TERMINAL_CAPS
    SDF_FLAT_CAP_REACH_FACTOR: float = SDF_FLAT_CAP_REACH_FACTOR
    OPEN_OUTLETS: bool = OPEN_OUTLETS
    FLAT_CAP_OUTLETS: bool = FLAT_CAP_OUTLETS
    FLAT_CAP_REQUIRE_MANIFOLD: bool = FLAT_CAP_REQUIRE_MANIFOLD
    FLAT_CAP_MAX_FACES: int = FLAT_CAP_MAX_FACES

    # repair
    MESH_REPAIR: bool = MESH_REPAIR
    MESH_FILL_HOLES: bool = MESH_FILL_HOLES
    MESH_REMOVE_DEGENERATE: bool = MESH_REMOVE_DEGENERATE
    MESH_FIX_NORMALS: bool = MESH_FIX_NORMALS
    USE_PYMESHFIX: bool = USE_PYMESHFIX
    PYMESHFIX_TIMEOUT_SEC: int = PYMESHFIX_TIMEOUT_SEC
    PYMESHFIX_MAX_FACES: int = PYMESHFIX_MAX_FACES

    # thin-vessel refinement
    THIN_VESSEL_REFINE: bool = THIN_VESSEL_REFINE
    THIN_VESSEL_REFINE_RADIUS_MM: float = THIN_VESSEL_REFINE_RADIUS_MM
    THIN_VESSEL_CIRCUMF_TARGET: int = THIN_VESSEL_CIRCUMF_TARGET
    THIN_VESSEL_REFINE_RELAX_ITERS: int = THIN_VESSEL_REFINE_RELAX_ITERS
    THIN_VESSEL_REFINE_RELAX_FORCE: float = THIN_VESSEL_REFINE_RELAX_FORCE
    THIN_VESSEL_MIN_EDGE_FACTOR: float = THIN_VESSEL_MIN_EDGE_FACTOR

    # smoothing
    TAUBIN_ITERS: int = TAUBIN_ITERS
    TAUBIN_BAND: float = TAUBIN_BAND
    TAUBIN_MAX_DISP_FACTOR: float = TAUBIN_MAX_DISP_FACTOR
    TAUBIN_JUNCTION_ONLY: bool = TAUBIN_JUNCTION_ONLY
    TAUBIN_JUNCTION_FACTOR: float = TAUBIN_JUNCTION_FACTOR
    TAUBIN_JUNCTION_FEATHER: float = TAUBIN_JUNCTION_FEATHER
    SUBDIVIDE_ITERS: int = SUBDIVIDE_ITERS

    # diagnostics / viz
    BLEND_DIAGNOSTIC: bool = BLEND_DIAGNOSTIC
    DETAILED_BLEND_DIAGNOSTIC: bool = DETAILED_BLEND_DIAGNOSTIC
    PREVIEW_SDF_BEFORE_MC: bool = PREVIEW_SDF_BEFORE_MC
    WRITE_REGION_VTK: bool = WRITE_REGION_VTK
    PROXIMITY_DIAGNOSTIC_TOP_N: int = PROXIMITY_DIAGNOSTIC_TOP_N
    DEBUG_VIS: bool = DEBUG_VIS
    DEBUG_VIS_BLOCK: bool = DEBUG_VIS_BLOCK
    DEBUG_VIS_SAVE_FALLBACK: bool = DEBUG_VIS_SAVE_FALLBACK
    DEBUG_VIS_COLOR_BY: str = DEBUG_VIS_COLOR_BY

    def __post_init__(self) -> None:
        if self.CENTERLINE_SMOOTHER not in {
            "none", "savgol", "bspline", "constrained_multiscale"
        }:
            raise ValueError(f"unknown CENTERLINE_SMOOTHER={self.CENTERLINE_SMOOTHER!r}")
        if self.CENTERLINE_CONSTRAINT_FAILURE not in {"report", "error"}:
            raise ValueError(
                "CENTERLINE_CONSTRAINT_FAILURE must be 'report' or 'error'"
            )
        if self.SDF_FIELD_METHOD not in {"legacy", "graph_implicit"}:
            raise ValueError(f"unknown SDF_FIELD_METHOD={self.SDF_FIELD_METHOD!r}")
        if self.PIPELINE_COMPONENT_FAILURE not in {"error", "continue"}:
            raise ValueError(
                "PIPELINE_COMPONENT_FAILURE must be 'error' or 'continue'"
            )
        if self.LEGACY_ORACLE_PRUNE_MODE not in {"geometric", "none"}:
            raise ValueError(
                "LEGACY_ORACLE_PRUNE_MODE must be 'geometric' or 'none'"
            )
        if self.IMPLICIT_PRIMITIVE_METHOD not in {"radial", "round_cone"}:
            raise ValueError(
                "IMPLICIT_PRIMITIVE_METHOD must be 'radial' or 'round_cone'"
            )
        if self.SDF_MESH_METHOD not in {
            "adaptive", "vtk_htg", "cgal_mesh3", "meshlib", "mc", "poisson"
        }:
            raise ValueError(f"unknown SDF_MESH_METHOD={self.SDF_MESH_METHOD!r}")
        if self.OUTPUT_VALIDATION_MODE not in {"off", "warn", "error"}:
            raise ValueError("OUTPUT_VALIDATION_MODE must be 'off', 'warn', or 'error'")
        if self.IMPLICIT_JUNCTION_BLEND_FRACTION < 0:
            raise ValueError("IMPLICIT_JUNCTION_BLEND_FRACTION must be non-negative")
        if self.IMPLICIT_JUNCTION_SUPPORT_FACTOR <= 1:
            raise ValueError("IMPLICIT_JUNCTION_SUPPORT_FACTOR must be greater than one")
        if self.IMPLICIT_CELLS_ACROSS_DIAMETER <= 0:
            raise ValueError("IMPLICIT_CELLS_ACROSS_DIAMETER must be positive")
        if self.BSPLINE_SDF_RESOLUTION is not None and self.BSPLINE_SDF_RESOLUTION <= 0:
            raise ValueError("BSPLINE_SDF_RESOLUTION must be positive or None")
        if self.DENSE_MIN_SPACING_MM is not None and self.DENSE_MIN_SPACING_MM <= 0:
            raise ValueError("DENSE_MIN_SPACING_MM must be positive or None")
        if self.VTK_HTG_PADDING_RADIUS_FACTOR <= 0:
            raise ValueError("VTK_HTG_PADDING_RADIUS_FACTOR must be positive")
        if self.VTK_HTG_MAX_CELLS <= 0:
            raise ValueError("VTK_HTG_MAX_CELLS must be positive")
        if not 0 < self.CGAL_FACET_ANGLE_DEG < 180:
            raise ValueError("CGAL_FACET_ANGLE_DEG must be between 0 and 180")
        if self.CGAL_FACET_DISTANCE_FRACTION <= 0:
            raise ValueError("CGAL_FACET_DISTANCE_FRACTION must be positive")
        if self.CGAL_CELL_SIZE_FACTOR <= 0:
            raise ValueError("CGAL_CELL_SIZE_FACTOR must be positive")
        if self.CGAL_CELL_RADIUS_EDGE_RATIO <= 0:
            raise ValueError("CGAL_CELL_RADIUS_EDGE_RATIO must be positive")
        if not self.CGAL_REQUIRED_VERSION:
            raise ValueError("CGAL_REQUIRED_VERSION must not be empty")
        if self.CGAL_NATIVE_API_VERSION <= 0:
            raise ValueError("CGAL_NATIVE_API_VERSION must be positive")
        if not self.CGAL_EXTENSION_BUILD_VERSION:
            raise ValueError("CGAL_EXTENSION_BUILD_VERSION must not be empty")

    def with_overrides(self, **overrides) -> "SdfConfig":
        """Return a validated copy with the supplied keyword overrides."""

        return replace(self, **overrides)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable configuration snapshot."""

        return asdict(self)


def default_config() -> SdfConfig:
    """Return an ``SdfConfig`` populated from the current module values."""
    return SdfConfig()


_ACTIVE_CONFIG: ContextVar[SdfConfig | None] = ContextVar(
    "coronary_sdf_active_config", default=None
)


class _RuntimeConfigProxy:
    """Read-only, context-local view used by pipeline implementation modules.

    Existing functions can continue to read ``config.NAME`` while a caller can
    safely run two configurations in separate threads/tasks. No module-level
    constant is rewritten and leaving :func:`use_config` restores the previous
    context automatically.
    """

    def __getattr__(self, name: str):
        active = _ACTIVE_CONFIG.get()
        if active is not None and hasattr(active, name):
            return getattr(active, name)
        try:
            return globals()[name]
        except KeyError as exc:  # pragma: no cover - normal AttributeError protocol
            raise AttributeError(name) from exc


runtime_config = _RuntimeConfigProxy()


@contextmanager
def use_config(cfg: SdfConfig | None = None) -> Iterator[SdfConfig]:
    """Bind an immutable runtime configuration without mutating module globals."""

    resolved = default_config() if cfg is None else cfg
    if not isinstance(resolved, SdfConfig):
        raise TypeError("cfg must be an SdfConfig or None")
    token = _ACTIVE_CONFIG.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_CONFIG.reset(token)


# Explicitly export public API symbols. Avoid dynamic computation of __all__ so
# static analysis tools can reliably determine exports.
__all__ = [
    "SdfConfig",
    "default_config",
    "runtime_config",
    "use_config",
]
