"""Rebuild the SDF lumen surface inside a box, fast enough to watch.

``coronary_sdf.pipeline.generate_sdf_surface`` is one 440-line function that
always processes a whole connected component, always writes files, and -- under
the shipped config -- always opens blocking debug windows. Re-running it after
every edit is the several-minute loop this package exists to replace.

The observation that makes a live preview possible is that ``Grid`` is a plain
dataclass with no invariants, and ``build_narrow_band`` / ``evaluate_sdf`` take
one as an ordinary argument. Nothing assumes it covers the whole tree.
``_probe_multifurc.py:429-446`` already exploits this to probe a 10 mm cube.

So :class:`SdfSession` does the graph-level preprocessing once (about a second
for LADAF-2024-28, and independent of grid size) and then evaluates the SDF only
inside the box an edit touched. On this dataset the full grid is 201 M voxels
with a 2.5 M-voxel narrow band; a 10 mm box is roughly 700 k voxels with a band
under 100 k.

**The local grid is a literal sub-block of the full grid.** ``evaluate_sdf``
samples world positions out of ``grid.x/y/z`` (``sdf_field.py:616``), so slicing
those arrays rather than rebuilding them means the patch is evaluated at exactly
the same world points as a full run would use. Any other construction leaves a
sub-voxel offset, and a sub-voxel offset is a visible seam.

That property is also why the box needs so little margin: since every capsule
reaches ``evaluate_sdf`` regardless of the grid, the field inside the box is
already correct on its first voxel. Only marching cubes and the Taubin pass need
room -- see :meth:`SdfSession.margin_mm`.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from .adapter import Triple
from .sdfconfig import PREVIEW, sdf_config

UM_PER_MM = 1000.0

# Voxels of margin evaluated outside the requested box before trimming back.
# Sized for the post-field mesh stencils, not for the SDF blend -- see
# :meth:`SdfSession.margin_mm` for why those are different things.
PATCH_MARGIN_VOXELS = 6.0

# Parent-through, daughter-emergence, and explicit retained-input fallback. A fallback
# is not a taper, but it is still an audited decision by radius-perimeter and must not
# be silently replaced by the generic downstream bifurcation repair.
AUTHORED_BIF_MODES = frozenset((3, 4, 5))  # radius_resolution_mode values
AUTHORED_RADIUS_PROFILE = {
    "SMOOTH_SEGMENT_RADII": False,
    "PRUNE_BIFURCATION_SHRINK": False,
    "SMOOTH_RADIUS_TRANSITIONS": False,
    "BIF_CARINA_ENABLE": False,
}


def has_complete_authored_bifurcation_taper(triple: Triple) -> bool:
    """True when every degree>=3 endpoint carries an authored parent/daughter role."""
    by_id = triple.point_attrs.get("radius_resolution_mode")
    if not by_id:
        return False
    incident: dict[int, list[dict]] = {}
    for seg in triple.segments:
        incident.setdefault(seg["node1"], []).append(seg)
        incident.setdefault(seg["node2"], []).append(seg)
    saw = False
    for nid, segments in incident.items():
        if len(segments) < 3:
            continue
        saw = True
        for seg in segments:
            pids = seg["point_ids"]
            if not pids:
                return False
            endpoint = pids[0] if seg["node1"] == nid else pids[-1]
            if int(by_id.get(endpoint, -1)) not in AUTHORED_BIF_MODES:
                return False
    return saw


@dataclass
class PatchResult:
    """One local rebuild: the surface, and enough context to splice it in."""

    surface: pv.PolyData  # micrometres, to match the viewer's world frame
    box_um: np.ndarray  # (2, 3) the inner box this patch is authoritative for
    voxel_size_mm: float
    n_band: int
    seconds: float
    log: str = ""

    @property
    def bounds_um(self) -> tuple[float, ...]:
        """The inner box as PyVista's ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
        b = self.box_um
        return (b[0, 0], b[1, 0], b[0, 1], b[1, 1], b[0, 2], b[1, 2])


@dataclass
class _Context:
    """Everything the SDF evaluation needs that does not depend on the grid."""

    capsules: Any
    cap_is_junction: np.ndarray
    adj_matrix: np.ndarray
    bif: Any
    term: Any
    shared_node_pos: np.ndarray
    shared_node_has: np.ndarray
    seg_end_pos: np.ndarray
    seg_end_tan: np.ndarray
    seg_end_tan_ok: np.ndarray
    is_parent: np.ndarray
    is_child: np.ndarray
    is_sibling: np.ndarray
    bif_seg_incident: np.ndarray
    full_grid: Any
    n_segments: int
    seconds: float = 0.0


def preprocess_graph(triple: Triple) -> Triple:
    """The graph-level clean-up ``run_pipeline`` does before meshing anything.

    This chain lives in ``run_pipeline`` (``pipeline.py:619-706``), *not* in
    ``generate_sdf_surface`` -- Strahler filtering, degree-2 contraction,
    split-multifurcation collapse, nub pruning, gap bridging and densification.
    Skipping it changes the point spacing, which changes every spline fit, which
    moves the surface by a third of a voxel. The preview and the export must
    therefore run the *same* chain, which is why it is factored out here and both
    paths are handed its result.

    Returns a new :class:`Triple`; the input is untouched. Several of these
    functions rewrite ``seg["point_ids"]`` in place, so re-running them on their
    own output is not idempotent.
    """
    from coronary_sdf import config
    from coronary_sdf.centreline_reconnection import (
        merge_degree2_segments,
        merge_split_multifurcations,
    )
    from coronary_sdf.pruning import prune_short_terminal_nubs
    from coronary_sdf.smoothing import bridge_centerline_gaps, densify_sparse_segments

    working = triple.copy()
    nodes, points, segments = working.as_args()

    if config.MIN_STRAHLER_ORDER > 0:
        segments = [s for s in segments if s.get("strahler", 0) >= config.MIN_STRAHLER_ORDER]
    if config.MERGE_DEGREE2_SEGMENTS:
        nodes, segments, _ = merge_degree2_segments(nodes, points, segments)
    if config.MERGE_SPLIT_MULTIFURCATIONS:
        nodes, segments, _, _ = merge_split_multifurcations(
            nodes, points, segments,
            max_len_factor=config.SPLIT_MULTIFURC_MAX_LEN_FACTOR,
            require_strahler=config.SPLIT_MULTIFURC_REQUIRE_STRAHLER,
            tangent_cos_min=config.SPLIT_MULTIFURC_TANGENT_COS_MIN,
        )
    if config.PRUNE_SHORT_TERMINAL_NUBS and config.MIN_TERMINAL_LENGTH_MM > 0:
        nodes, segments, _ = prune_short_terminal_nubs(
            nodes, points, segments,
            min_length_mm=config.MIN_TERMINAL_LENGTH_MM,
            max_iters=config.PRUNE_ITER_MAX,
        )
    if config.BRIDGE_CENTERLINE_GAPS:
        points, _, _ = bridge_centerline_gaps(
            points, segments,
            target_spacing_mm=config.DENSIFY_TARGET_SPACING_MM,
            big_jump_ratio=config.GAP_BIG_JUMP_RATIO,
            min_gap_um=config.CENTERLINE_MAX_GAP_UM,
            radius_scale=config.RADIUS_SCALE,
            verbose=config.DENSIFY_VERBOSE,
        )
    if config.DENSIFY_SPARSE_SEGMENTS:
        points, _ = densify_sparse_segments(
            points, segments,
            target_spacing_mm=config.DENSIFY_TARGET_SPACING_MM,
            min_points=config.DENSIFY_MIN_POINTS,
            verbose=config.DENSIFY_VERBOSE,
        )

    working.nodes = nodes
    working.points = points
    working.segments = segments
    return working


@contextlib.contextmanager
def _quiet(enabled: bool):
    """Swallow the pipeline's very chatty stdout, keeping it for error reports."""
    if not enabled:
        yield None
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


class SdfSession:
    """Holds the SDF context for one graph and rebuilds it box by box.

    Preprocessing is redone from the *raw* edited graph on every
    :meth:`set_graph`, exactly as a full pipeline run would. That is what makes
    the preview trustworthy: the smoothing, radius pruning and spline fitting a
    patch sees are the same ones the exported STL will see. Only the SDF
    evaluation -- the part whose cost scales with volume -- is localised.
    """

    def __init__(
        self,
        triple: Triple,
        *,
        profile: dict[str, Any] | None = None,
        voxel_size_mm: float | None = None,
        quiet: bool = True,
    ):
        self.profile = dict(PREVIEW if profile is None else profile)
        if voxel_size_mm is not None:
            # Pin the resolution rather than letting compute_grid derive it, so
            # the patch grid can never drift between rebuilds.
            self.profile["BSPLINE_SDF_RESOLUTION"] = float(voxel_size_mm)
        self.quiet = quiet
        self.triple: Triple | None = None
        self._ctx: _Context | None = None
        self.set_graph(triple)

    # ------------------------------------------------------------------ setup

    def set_graph(self, triple: Triple, root_pref: set[int] | None = None) -> None:
        """Re-run all preprocessing against an edited graph.

        Both the preview and :meth:`rebuild_full` consume ``self._clean``, so the
        two cannot drift: whatever the export will mesh is what the patch meshes.
        """
        t0 = time.time()
        self.triple = triple
        self.root_pref = set(root_pref) if root_pref else None
        self.authored_bifurcation_taper = has_complete_authored_bifurcation_taper(triple)
        self._surface_profile = dict(self.profile)
        if self.authored_bifurcation_taper:
            self._surface_profile.update(AUTHORED_RADIUS_PROFILE)
        with _quiet(self.quiet), sdf_config(self._surface_profile):
            self._clean = preprocess_graph(triple)
            self._ctx = self._prepare(self._clean)
        self._ctx.seconds = time.time() - t0

    @property
    def voxel_size_mm(self) -> float:
        return float(self._ctx.full_grid.voxel_size)

    @property
    def n_capsules(self) -> int:
        return int(self._ctx.capsules.n)

    @property
    def prepare_seconds(self) -> float:
        return self._ctx.seconds

    def _prepare(self, clean: Triple) -> _Context:
        """Mirror ``generate_sdf_surface``'s setup, stopping before the grid.

        Takes the output of :func:`preprocess_graph`, not the raw graph.

        Deliberately *not* split into connected components. ``run_pipeline``
        splits so each component gets its own STL; here a neighbouring component
        that happens to pass close by must stay visible to the anti-bridge carve,
        or the preview would fuse vessels the export keeps apart.
        """
        from coronary_sdf import config
        from coronary_sdf.bif_trim import taper_bifurcation_carina
        from coronary_sdf.capsules import build_capsules, clamp_terminal_capsule_radii
        from coronary_sdf.centreline_reconnection import node_id_canon_map
        from coronary_sdf.sdf_field import (
            build_adjacency,
            build_terminal_set,
            collect_endpoint_info,
            compute_grid,
            find_bifurcations,
        )
        from coronary_sdf.smoothing import (
            limit_centerline_curvature,
            prune_bifurcation_shrink,
            prune_terminal_shrink,
            smooth_radius_transitions,
            smooth_segment_centerlines,
            smooth_segment_radii,
        )
        from coronary_sdf.splines import branch_tangent_at_node, prepare_segment_spline
        from coronary_sdf.topology import build_directed_topology, label_capsules_by_cross_section

        # A copy again: the smoothing passes return new points dicts but the
        # spline/capsule stages read the segment dicts directly.
        working = clean.copy()
        nodes, points, segments = working.as_args()

        points, _ = smooth_segment_centerlines(nodes, points, segments)
        if config.LIMIT_CENTERLINE_CURVATURE:
            points, _ = limit_centerline_curvature(nodes, points, segments)
        points, _ = smooth_segment_radii(nodes, points, segments)
        points, _ = prune_bifurcation_shrink(nodes, points, segments)
        points, _ = prune_terminal_shrink(nodes, points, segments)
        points, _ = smooth_radius_transitions(nodes, points, segments)

        node_to_segs: dict[int, set[int]] = {}
        for idx, seg in enumerate(segments):
            for nid in (seg["node1"], seg["node2"]):
                node_to_segs.setdefault(nid, set()).add(idx)

        if config.SDF_FLAT_TERMINAL_CAPS:
            endpoint_info = collect_endpoint_info(nodes, points, segments, node_to_segs)
        else:
            endpoint_info = []
        term = build_terminal_set(endpoint_info)

        adj_matrix, shared_pos, _ = build_adjacency(nodes, points, segments, node_to_segs)
        bif = find_bifurcations(nodes, points, segments, node_to_segs)

        dtopo = build_directed_topology(segments, node_to_segs, root_pref=self.root_pref)

        n_segs = len(segments)
        n_bifs = int(len(bif.node_ids)) if bif.node_ids is not None else 0
        bif_seg_incident = np.zeros((n_bifs, n_segs), dtype=bool)
        for bi in range(n_bifs):
            for sj in node_to_segs.get(int(bif.node_ids[bi]), set()):
                if 0 <= sj < n_segs:
                    bif_seg_incident[bi, sj] = True

        seg_id_to_idx = {seg["id"]: i for i, seg in enumerate(segments)}
        segment_splines: list[dict | None] = []
        for seg in segments:
            sp = prepare_segment_spline(seg, points, nodes)
            if sp is not None:
                sp["seg_idx"] = seg_id_to_idx[seg["id"]]
            segment_splines.append(sp)
        valid_splines = [s for s in segment_splines if s is not None]
        if not valid_splines:
            raise ValueError("no segment produced a usable spline")

        canon = node_id_canon_map(nodes, config.NODE_COINCIDENCE_EPS_MM)
        canonical: dict[int, set[int]] = {}
        for nid, segs in node_to_segs.items():
            canonical.setdefault(canon.get(nid, nid), set()).update(segs)
        terminal_ids = {
            nid for nid in node_to_segs
            if len(canonical.get(canon.get(nid, nid), set())) == 1
        }
        clamp_terminal_capsule_radii(valid_splines, terminal_ids)
        if config.BIF_CARINA_ENABLE:
            taper_bifurcation_carina(valid_splines, nodes, points, segments)

        seg_end_pos = np.full((n_segs, 2, 3), np.nan)
        seg_end_tan = np.full((n_segs, 2, 3), np.nan)
        seg_end_tan_ok = np.zeros((n_segs, 2), dtype=bool)
        for seg, sp in zip(segments, segment_splines):
            if sp is None:
                continue
            si = int(sp["seg_idx"])
            for slot, key in ((0, "node1"), (1, "node2")):
                nid = seg[key]
                if nid in nodes:
                    seg_end_pos[si, slot] = np.asarray(nodes[nid][:3]) / UM_PER_MM
                tan = branch_tangent_at_node(sp, nid, seg)
                if tan is not None:
                    seg_end_tan[si, slot] = tan
                    seg_end_tan_ok[si, slot] = True

        capsules = build_capsules(valid_splines, node_to_segs=node_to_segs)
        if config.USE_CROSS_SECTION_BLEND_GATE:
            cap_is_junction = label_capsules_by_cross_section(
                capsules.midpoints, nodes, points, segments
            )
        else:
            cap_is_junction = np.zeros(capsules.n, dtype=bool)

        shared_node_pos = np.full((n_segs, n_segs, 3), np.nan)
        shared_node_has = np.zeros((n_segs, n_segs), dtype=bool)
        for (si, sj), pos in shared_pos.items():
            shared_node_pos[si, sj] = pos
            shared_node_has[si, sj] = True

        # The full grid is never evaluated -- it exists so every patch can be a
        # sub-block of one fixed lattice, and so the voxel size is decided once
        # from the whole tree rather than from whatever happens to be in a box.
        full_grid = compute_grid(capsules)

        return _Context(
            capsules=capsules,
            cap_is_junction=cap_is_junction,
            adj_matrix=adj_matrix,
            bif=bif,
            term=term,
            shared_node_pos=shared_node_pos,
            shared_node_has=shared_node_has,
            seg_end_pos=seg_end_pos,
            seg_end_tan=seg_end_tan,
            seg_end_tan_ok=seg_end_tan_ok,
            is_parent=dtopo["is_parent"],
            is_child=dtopo["is_child"],
            is_sibling=dtopo["is_sibling"],
            bif_seg_incident=bif_seg_incident,
            full_grid=full_grid,
            n_segments=n_segs,
        )

    # ------------------------------------------------------------- the rebuild

    def margin_mm(self) -> float:
        """How far outside the box to evaluate before trimming back.

        It is tempting to pad by the smooth-min blend reach (one vessel radius
        plus ``SMIN_PROXIMITY_BLEND_FACTOR``, as ``build_narrow_band`` does at
        ``sdf_field.py:444-447``). That is unnecessary: **every capsule is passed
        to** ``evaluate_sdf`` **whatever the grid is**, so the field at a voxel
        does not depend on the box the voxel happens to sit in. The blend and the
        anti-bridge carve are already correct on the first voxel inside the box.

        What *does* need margin is the work done after the field -- marching
        cubes needs a neighbouring voxel, and ``radius_constrained_taubin``
        smooths over a stencil that widens with its iteration count. Measured on
        LADAF-2024-28, deviation from a patch padded by the full blend reach is
        ~1% of a voxel (mean) at any margin from 2 to 30 voxels; the residual is
        meshlib's global relax pass, not the margin. Six voxels buys the Taubin
        stencil room while keeping the evaluated volume ~3x the box rather than
        the ~9x a blend-sized pad costs.
        """
        return float(PATCH_MARGIN_VOXELS * self._ctx.full_grid.voxel_size)

    def _subgrid(self, box_mm: np.ndarray):
        """A Grid that is a literal sub-block of the full grid's lattice.

        Returns ``(grid, mesh_origin)``. The two are not the same point, and the
        difference is the whole reason a naive patch lands ~40 um off a full run:

        ``compute_grid`` samples the field at ``linspace(bbox_min, bbox_max,
        dims)``, whose step is ``extent / (dims - 1)``, but hands the mesher
        ``voxel_size`` as the spacing. Those differ by up to ``1 / (dims - 1)``,
        about 0.15% here. A full run accumulates that discrepancy from index 0,
        so the mesh it produces sits at ``bbox_min + index * voxel_size``. A patch
        that starts its own indexing at ``i0`` would place the same sample at
        ``x[i0] + i * voxel_size`` and be off by ``i0 * (voxel_size - step)`` --
        a few hundred voxels in, that is tens of micrometres of pure translation.

        So the field keeps the true sample positions (``grid.x`` sliced verbatim)
        and the mesher is given the origin the full run would have used.
        """
        from coronary_sdf.sdf_field import Grid

        full = self._ctx.full_grid
        axes = (full.x, full.y, full.z)
        lo, hi = [], []
        for axis, a, b in zip(axes, box_mm[0], box_mm[1]):
            # searchsorted on the full lattice: no arithmetic that could drift.
            i0 = int(np.clip(np.searchsorted(axis, a, side="right") - 1, 0, len(axis) - 1))
            i1 = int(np.clip(np.searchsorted(axis, b, side="left") + 1, i0 + 2, len(axis)))
            lo.append(i0)
            hi.append(i1)

        x = full.x[lo[0]:hi[0]]
        y = full.y[lo[1]:hi[1]]
        z = full.z[lo[2]:hi[2]]
        bbox_min = np.array([x[0], y[0], z[0]])
        bbox_max = np.array([x[-1], y[-1], z[-1]])
        dims = np.array([len(x), len(y), len(z)], dtype=np.int64)
        grid = Grid(bbox_min, bbox_max, float(full.voxel_size), dims, x, y, z)
        mesh_origin = full.bbox_min + np.asarray(lo, dtype=np.float64) * full.voxel_size
        return grid, mesh_origin

    def rebuild(self, box_um, *, pad_um: float | None = None) -> PatchResult:
        """Rebuild the surface inside `box_um`, an AABB in micrometres.

        The evaluated region is `box_um` grown by :meth:`margin_mm` to give the
        mesh stencils room; the returned surface is trimmed back to `box_um`,
        which is the region the patch is authoritative for.
        """
        t0 = time.time()
        inner_um = np.asarray(box_um, dtype=np.float64).reshape(2, 3)
        pad_mm = self.margin_mm() if pad_um is None else float(pad_um) / UM_PER_MM
        inner_mm = inner_um / UM_PER_MM
        outer_mm = np.array([inner_mm[0] - pad_mm, inner_mm[1] + pad_mm])

        ctx = self._ctx
        with _quiet(self.quiet) as log, sdf_config(self.profile):
            from coronary_sdf.mesh_extract import drop_nonfinite_vertices, extract_isosurface
            from coronary_sdf.mesh_repair import radius_constrained_taubin
            from coronary_sdf.sdf_field import build_narrow_band, evaluate_sdf

            grid, mesh_origin = self._subgrid(outer_mm)
            nb_idx = build_narrow_band(ctx.capsules, grid)
            if len(nb_idx) == 0:
                return PatchResult(pv.PolyData(), inner_um, grid.voxel_size, 0,
                                   time.time() - t0, log.getvalue() if log else "")

            sdf = evaluate_sdf(
                capsules=ctx.capsules,
                cap_is_junction=ctx.cap_is_junction,
                adj_matrix=ctx.adj_matrix,
                shared_node_pos=ctx.shared_node_pos,
                shared_node_has=ctx.shared_node_has,
                seg_end_pos=ctx.seg_end_pos,
                seg_end_tan=ctx.seg_end_tan,
                seg_end_tan_ok=ctx.seg_end_tan_ok,
                is_parent=ctx.is_parent,
                is_child=ctx.is_child,
                is_sibling=ctx.is_sibling,
                bif_seg_incident=ctx.bif_seg_incident,
                bif=ctx.bif,
                term=ctx.term,
                grid=grid,
                nb_idx=nb_idx,
            )

            # mesh_origin, not grid.bbox_min -- see _subgrid.
            surface = extract_isosurface(
                sdf.sdf, mesh_origin, grid.voxel_size, grid.dims,
                capsule_tree=ctx.capsules.tree, cap_max_radii=ctx.capsules.max_radii,
            )
            if surface is not None and surface.n_points:
                surface, _ = drop_nonfinite_vertices(surface)
            if surface is not None and surface.n_cells:
                surface = surface.triangulate().clean(tolerance=0.0)
                surface = radius_constrained_taubin(
                    surface, ctx.capsules.tree, ctx.capsules.max_radii, grid.voxel_size,
                    term_pos=ctx.term.pos, term_nrm=ctx.term.nrm,
                    term_rad=ctx.term.rad, term_tree=ctx.term.tree,
                    bif_pos=ctx.bif.positions, bif_rad=ctx.bif.radii,
                )
            text = log.getvalue() if log else ""

        if surface is None:
            surface = pv.PolyData()
        if surface.n_points:
            surface = surface.scale(UM_PER_MM, inplace=False)
            surface = _clip_to_box(surface, inner_um)

        return PatchResult(
            surface=surface,
            box_um=inner_um,
            voxel_size_mm=float(grid.voxel_size),
            n_band=int(len(nb_idx)),
            seconds=time.time() - t0,
            log=text,
        )

    def rebuild_around(self, patch, *, min_extent_um: float = 4000.0) -> PatchResult:
        """Rebuild around a :class:`~.history.Patch`, with a sane minimum size.

        A radius edit on a single point reports a zero-volume box; expanding to
        at least `min_extent_um` keeps the patch big enough to contain the
        surface change, which spreads over roughly the local vessel diameter.
        """
        if patch is None or patch.aabb is None:
            raise ValueError("patch has no spatial extent to rebuild")
        box = np.array(patch.aabb, dtype=np.float64)
        centre = box.mean(axis=0)
        half = np.maximum((box[1] - box[0]) / 2.0, min_extent_um / 2.0)
        return self.rebuild(np.array([centre - half, centre + half]))

    # ------------------------------------------------------------------ export

    def rebuild_full(self, output_dir: str | Path, *, graph_id: int = 0) -> pv.PolyData | None:
        """The real thing: an unmodified ``generate_sdf_surface`` over the graph.

        Fed from the same :func:`preprocess_graph` output the patches use, so
        this is the geometry a preview is previewing. ``generate_sdf_surface``
        alone would *not* be equivalent -- it does none of that chain, and
        skipping it moves the surface by about a third of a voxel.

        Not split into connected components: ``run_pipeline`` writes one STL per
        component, but a session models the whole scene the viewer shows.
        """
        from coronary_sdf.pipeline import generate_sdf_surface

        working = self._clean.copy()
        with _quiet(self.quiet), sdf_config(self._surface_profile):
            return generate_sdf_surface(*working.as_args(), Path(output_dir), graph_id)


def _clip_to_box(mesh: pv.PolyData, box_um: np.ndarray) -> pv.PolyData:
    """Keep whole cells whose centroid is inside the box.

    Same centroid test as ``stl_slice.clip_to_roi``, which that module documents
    as ~40x faster than ``clip_box(crinkle=True)`` on a large mesh. Keeping cells
    whole means the splice boundary follows triangle edges instead of cutting
    them, so no sliver triangles are created at the seam.
    """
    if mesh.n_cells == 0:
        return mesh
    cc = np.asarray(mesh.cell_centers().points)
    inside = np.all((cc >= box_um[0]) & (cc <= box_um[1]), axis=1)
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return pv.PolyData()
    if idx.size == mesh.n_cells:
        return mesh
    return mesh.extract_cells(idx).extract_surface()


def splice(base: pv.PolyData, patch: PatchResult) -> pv.PolyData:
    """Replace the part of `base` inside the patch box with the patch.

    Both meshes are in micrometres. The result is not welded across the seam --
    the two sides meet but do not share vertices -- which is fine for display and
    is why the exported STL comes from :meth:`SdfSession.rebuild_full` rather
    than from an accumulation of patches.
    """
    if base is None or base.n_cells == 0:
        return patch.surface
    cc = np.asarray(base.cell_centers().points)
    box = patch.box_um
    outside = ~np.all((cc >= box[0]) & (cc <= box[1]), axis=1)
    idx = np.flatnonzero(outside)
    kept = base.extract_cells(idx).extract_surface() if idx.size else pv.PolyData()
    if patch.surface.n_cells == 0:
        return kept
    if kept.n_cells == 0:
        return patch.surface
    return kept.merge(patch.surface)
