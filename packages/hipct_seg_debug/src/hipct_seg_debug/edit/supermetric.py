"""The Walsh-Berg super metric, scored on this package's graphs.

The metric itself -- Eq. 10 of

    C.L. Walsh, M. Berg, H. West, N.A. Holroyd, S. Walker-Samuel, R.J. Shipley,
    "Reconstructing microvascular network skeletons from 3D images: What is the
    ground truth?", Computers in Biology and Medicine 171 (2024) 108140.

-- lives in one place, :mod:`skeleton_analysis.optimisation.supermetric`. This
module adapts an :class:`~.graphmodel.EditableGraph` and a
:class:`~..frame.WorldFrame` to it, and adds what is specific to the coronary
pipeline: per-tree scoring, voxel-weighted aggregation, and reading the mask from
an RLE lattice without decoding it.

    M_S = |V_I - V_S|/V_I + |cc_I - cc_S|/cc_I + |chi_I - chi_S|/chi_I
          + |1 - cl_S| / cl_S**3   +   |1 - B_S| / B_S**2

Lower is better; zero is identical.

**Options.** Scoring here defaults to the ``CORONARY`` preset of
:class:`~skeleton_analysis.optimisation.supermetric.SuperMetricOptions`: the
bifurcation reference is the mask's own skeleton junctions (there is no manual
annotation), there is no sub-volume, ``chi`` is scored against a tree (see
:attr:`ImageTerms.chi`), and points Avizo interpolated are left out of ``V`` and
``cl``. Pass ``options=PAPER`` (or ``--metric-preset paper``) for numbers that are
comparable with the paper. ``docs/SKELETONISATION.md`` records each departure.

**The non-linear weights are the design, not a detail.** ``cl`` enters as ``1/cl**3``,
so even a small drop in the fraction of centreline lying inside the mask dominates the
sum and rejects the skeleton. ``B`` is weighted more softly at ``1/B**2`` because its
reference is annotated (or, here, derived) rather than measured. :func:`super_metric`
therefore returns every term separately as well as the total -- the paper's own use of
it (Fig. 8B) is to read which term dominates, which says *how* an algorithm is failing.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np

from ._deps import ensure_skeleton_analysis

ensure_skeleton_analysis()
from skeleton_analysis.optimisation import supermetric as core  # noqa: E402
from skeleton_analysis.optimisation.supermetric import (  # noqa: E402,F401
    CORONARY,
    PAPER,
    SuperMetricOptions,
)


def local_euler(chi_classical: float) -> float:
    """``2 - chi_classical`` -- strictly positive for any graph or solid."""
    return float(core.local_euler(chi_classical))


def options_for(preset: str = "coronary", *, tree_chi: bool | None = None,
                bb_threshold: float | None = None) -> SuperMetricOptions:
    """The options a command line asks for.

    `preset` is ``"coronary"`` or ``"paper"``. `tree_chi` overrides the preset's chi
    reference. A `bb_threshold` (um) replaces the radius-based bifurcation tolerance
    with that fixed distance, as before this module used the shared core.
    """
    base = core.PRESETS[preset]
    changes = {}
    if tree_chi is not None:
        changes["chi_reference"] = "tree" if tree_chi else "image"
    if bb_threshold is not None:
        changes.update(tolerance="fixed", tolerance_value=float(bb_threshold))
    return SuperMetricOptions(**{**asdict(base), **changes}) if changes else base


# ------------------------------------------------------------------ adapters


def _frame_geometry(frame):
    """``(voxel_size, origin)`` in um, (x, y, z), of the segmentation lattice."""
    return (np.asarray(frame.seg_spacing, dtype=np.float64),
            np.asarray(frame.seg_origin, dtype=np.float64))


def _valid_points(graph, sid: int, n: int) -> np.ndarray:
    """(n,) bool -- False where Avizo invented the point rather than skeletonising it."""
    from .interpolation import mask_for_segment

    invented = mask_for_segment(graph, sid)
    if invented.size != n:
        return np.ones(n, dtype=bool)
    return ~invented


def to_skeleton_graph(graph, segs=None, *, include_isolated: bool = True):
    """The shared metric's :class:`SkeletonGraph` for an ``EditableGraph``.

    `segs` restricts it to those segment ids (and their nodes). Isolated nodes are
    included for the whole graph -- each is a component, per the paper -- but not
    when `segs` nominates a subgraph. A one-point segment is repeated so it still
    rasterises to its voxel and contributes no length.
    """
    seg_ids = list(graph.segment_ids()) if segs is None else [int(s) for s in segs]
    if segs is None and include_isolated:
        node_ids = list(graph.nodes)
    else:
        node_ids = sorted({graph.segment(s)[k] for s in seg_ids for k in ("node1", "node2")})
    index = {nid: i for i, nid in enumerate(node_ids)}
    nodes = np.array([graph.nodes[n][:3] for n in node_ids], dtype=np.float64).reshape(-1, 3)
    edges, points, radii, valid = [], [], [], []
    for sid in seg_ids:
        seg = graph.segment(sid)
        coords = graph.coords(sid)
        r = graph.radii(sid)
        ok = _valid_points(graph, sid, len(coords))
        if len(coords) == 0:
            coords = np.array([graph.nodes[seg["node1"]][:3]], dtype=np.float64)
            r, ok = np.zeros(1), np.ones(1, dtype=bool)
        if len(coords) == 1:
            coords, r, ok = np.repeat(coords, 2, axis=0), np.repeat(r, 2), np.repeat(ok, 2)
        edges.append((index[seg["node1"]], index[seg["node2"]]))
        points.append(coords)
        radii.append(r)
        valid.append(ok)
    with warnings.catch_warnings():
        # Pipeline segments need not carry their end nodes as points; the metric
        # does not rely on it.
        warnings.filterwarnings("ignore", message=".*do not start/end at their edge's nodes")
        return core.SkeletonGraph(nodes, np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                                  points, radii, valid)


# ------------------------------------------------------------------ graph terms


def graph_volume(graph, *, exclude_invented: bool = True) -> float:
    """Total network volume in um^3: the sum of subsegment cylinders, each of the
    mean of its two radii (Table S13). Subsegments Avizo interpolated are left out
    by default -- an invented length carrying an invented radius."""
    return to_skeleton_graph(graph).volume(exclude_invented)


def graph_components(graph) -> int:
    """Number of subnetworks -- ``cc`` from the graph. An isolated node counts."""
    return to_skeleton_graph(graph).n_components()


def graph_euler_classical(graph, segs=None) -> int:
    """``N - E`` of the **largest** subgraph (by volume), the graph form of the Euler
    number (Table S13). A tree gives 1 and each independent loop subtracts one.

    `segs` scores one nominated component's segment ids instead -- the per-tree path.
    """
    if segs is not None:
        segs = list(segs)
        if not segs:
            return 0
        return to_skeleton_graph(graph, segs).largest_component_euler_classical()
    if not graph.segment_ids():
        return 0
    return to_skeleton_graph(graph, include_isolated=False).largest_component_euler_classical()


def _raster_zyx(sg, frame, exclude_invented: bool) -> np.ndarray:
    voxel_size, origin = _frame_geometry(frame)
    return core.rasterise_voxels(sg, voxel_size, origin, exclude_invented)


def rasterise_centreline(graph, frame, *, exclude_invented: bool = True) -> np.ndarray:
    """Unique ``(iz, iy, ix)`` voxels covered by the centreline, Bresenham lines
    between consecutive points (Table S13), including voxels outside the lattice."""
    return _raster_zyx(to_skeleton_graph(graph), frame, exclude_invented)


def rasterise_segment(graph, sid, frame) -> np.ndarray:
    """Unique ``(iz, iy, ix)`` voxels covered by one segment's centreline."""
    return _raster_zyx(to_skeleton_graph(graph, [sid]), frame, True)


def _sample_mask(labels, zyx: np.ndarray, dims) -> np.ndarray:
    """Mask value at each ``(iz, iy, ix)``; accepts a decoded array or an RLE lattice."""
    nx, ny, nz = (int(v) for v in dims)
    inb = (
        (zyx[:, 0] >= 0) & (zyx[:, 0] < nz)
        & (zyx[:, 1] >= 0) & (zyx[:, 1] < ny)
        & (zyx[:, 2] >= 0) & (zyx[:, 2] < nx)
    )
    sel = zyx[inb]
    res = np.zeros(len(zyx), dtype=bool)
    if len(sel) == 0:
        return res
    if isinstance(labels, np.ndarray):
        vals = labels[sel[:, 0], sel[:, 1], sel[:, 2]]
    else:
        # One decode per z plane, not one per voxel: the centreline touches a few
        # thousand planes and re-decoding each would dominate the whole metric.
        vals = np.zeros(len(sel), dtype=np.uint8)
        for kz in np.unique(sel[:, 0]):
            m = sel[:, 0] == kz
            plane = labels.slice_z(int(kz))
            vals[m] = plane[sel[m, 1], sel[m, 2]]
    res[inb] = vals > 0
    return res


def _cl_from_skeleton(sg, frame, labels, exclude_invented: bool) -> float:
    zyx = _raster_zyx(sg, frame, exclude_invented)
    if len(zyx) == 0:
        return float("nan")
    return float(_sample_mask(labels, zyx, frame.seg_dims).sum() / len(zyx))


def cl_sensitivity(graph, frame, labels, *, exclude_invented: bool = True) -> float:
    """Fraction of the rasterised centreline lying inside the segmentation,
    ``sum(V_I . l_S) / sum(l_S)`` (Table S13). Voxels outside the lattice are misses."""
    return _cl_from_skeleton(to_skeleton_graph(graph), frame, labels, exclude_invented)


# ------------------------------------------------------------------ image terms


@dataclass
class ImageTerms:
    """``V``, ``cc`` and ``chi`` of the binary image -- the gold standard side."""

    volume_um3: float
    components: int
    euler_classical: int
    voxel_count: int
    tree_chi: bool = False
    seconds: float = 0.0

    @property
    def chi(self) -> float:
        """The reference local Euler characteristic.

        With `tree_chi` the reference is the tree ideal (``chi_classical = 1``) rather
        than the segmentation's own loop count. This pipeline images coronary arteries
        ex vivo, where the true anatomy is a tree: loops in a skeleton here come from a
        vessel that collapsed in the middle and was segmented as two, or from a
        segmentation touching its neighbour. :func:`~.skeleton_optimise.remove_loops`
        deletes them on purpose, so scoring against the image's loop count would make
        the metric fight that prior. **It also makes M_S incomparable with the paper's
        published values** -- pass ``tree_chi=False`` for a comparable number.
        """
        return local_euler(1 if self.tree_chi else self.euler_classical)

    def describe(self) -> str:
        return (
            f"image: V {self.volume_um3 / 1e9:.3f} mm^3 ({self.voxel_count:,} voxels), "
            f"cc {self.components}, chi_classical {self.euler_classical} "
            f"-> chi {self.chi:.0f}" + (" (tree reference)" if self.tree_chi else "")
        )


def image_terms(volume, spacing_um, *, tree_chi: bool = False,
                connectivity: int = 3, labels=None, component=None) -> ImageTerms:
    """``V``, ``cc`` and ``chi_classical`` of a decoded binary volume.

    26-connectivity, and the Euler number of the **largest component alone**, per
    Table S13 (computed by the shared core). `labels` and `component` score one
    already-labelled component instead -- the per-tree path.
    """
    from skimage.measure import euler_number

    t0 = time.time()
    sp = np.asarray(spacing_um, dtype=np.float64)
    voxel_volume = float(sp[0] * sp[1] * sp[2])

    if labels is not None and component is not None:
        binary = np.asarray(labels) == int(component)
        n_vox = int(np.count_nonzero(binary))
        if n_vox == 0:
            return ImageTerms(0.0, 0, 0, 0, tree_chi, time.time() - t0)
        return ImageTerms(
            volume_um3=n_vox * voxel_volume,
            components=1,  # a nominated component is one component, by construction
            euler_classical=int(euler_number(binary, connectivity=connectivity)),
            voxel_count=n_vox,
            tree_chi=tree_chi,
            seconds=time.time() - t0,
        )

    binary = np.asarray(volume) > 0
    n_vox = int(np.count_nonzero(binary))
    if n_vox == 0:
        return ImageTerms(0.0, 0, 0, 0, tree_chi, time.time() - t0)
    m = core.binary_measures(binary, sp)
    return ImageTerms(
        volume_um3=m["volume"],
        components=m["n_components"],
        euler_classical=m["euler_classical"],
        voxel_count=n_vox,
        tree_chi=tree_chi,
        seconds=time.time() - t0,
    )


class ReferenceBifurcations(np.ndarray):
    """``(n, 3)`` reference bifurcation coordinates in um, carrying ``radius_um``.

    A plain array everywhere it is used as one; the radius (the mask's distance map
    at each point) is what the radius-based match tolerance needs.
    """

    radius_um: np.ndarray | None = None

    def __array_finalize__(self, obj):
        if obj is not None:
            self.radius_um = getattr(obj, "radius_um", None)


def reference_bifurcations(volume, frame, *, stride: int = 1) -> ReferenceBifurcations:
    """Bifurcation points of the segmentation's own skeleton, in um ``(x, y, z)``.

    The paper annotates these by hand in a subvolume. Automating it from the mask keeps
    the gold standard the binary image and makes a parameter sweep runnable without a
    day of annotation first -- at the cost of inheriting Lee thinning's own errors,
    which is why ``B`` is the softly weighted term.
    """
    voxel_size, origin = _frame_geometry(frame)
    voxel_size = voxel_size * max(int(stride), 1)
    binary = np.asarray(volume) > 0
    pts = core.auto_reference_bifurcations(binary, voxel_size, origin)
    out = np.asarray(pts, dtype=np.float64).reshape(-1, 3).view(ReferenceBifurcations)
    radius = core.local_radius(binary, out, voxel_size, origin)
    out.radius_um = np.maximum(radius, voxel_size.min())
    return out


def super_metric_per_tree(graph, frame, labels_zyx, parts, *, bb_threshold: float | None = None,
                          tree_chi: bool = True, verbose: bool = False,
                          options: SuperMetricOptions | None = None) -> dict:
    """``{tree index: SuperMetric}`` -- each tree scored against its own component.

    The default :func:`super_metric` measures ``chi`` on the largest component alone,
    on both the graph side and the image side. With a left and a right coronary tree
    in one mask that means the right one contributes nothing to that term, so a sweep
    optimising ``M_S`` is partly blind to half the anatomy. Here each tree is the whole
    of its own gold standard.

    **The numbers are not comparable with the whole-graph ones**: ``cc`` becomes
    ``|1 - cc_s| / 1``, so a tree that skeletonised into three fragments scores 2.0.
    That is a harsher and arguably truer reading, which is why it is opt-in.
    """
    from .components import subgraph_by_tree

    out: dict = {}
    labels_zyx = np.asarray(labels_zyx)
    for part in parts:
        sub = subgraph_by_tree(graph, part.index)
        if not sub.segments:
            if verbose:
                print(f"    tree {part.index}: no segment carries this tree; skipped")
            continue
        # The full-size mask of this component alone: `frame` describes the whole
        # volume, so the cropped `part.volume` would index into the wrong place.
        mask = labels_zyx == part.label
        image = image_terms(mask, frame.seg_spacing, tree_chi=tree_chi)
        refs = reference_bifurcations(mask, frame, stride=1)
        out[part.index] = super_metric(sub, frame, mask, image, refs,
                                       bb_threshold=bb_threshold, options=options)
        del mask
    return out


def aggregate(per_tree: dict, parts=None, *, objective: str = "weighted") -> float:
    """Combine per-tree scores into one number for a sweep to minimise.

    ``weighted`` -- by voxel count -- is the default because the alternatives both
    mislead: an unweighted ``mean`` lets a 3000-voxel fragment outvote the left main,
    and a plain sum rescales with the number of trees. A tree whose score is
    infinite makes the aggregate infinite; a NaN (unscored) tree is skipped.
    """
    totals = {i: m.total for i, m in per_tree.items()}
    good = {i: t for i, t in totals.items() if not np.isnan(t)}
    if not good:
        return float("nan")
    if objective == "mean":
        return float(np.mean(list(good.values())))
    if objective == "sum":
        return float(np.sum(list(good.values())))
    weights = {int(p.index): float(p.voxels) for p in (parts or ())}
    w = np.array([weights.get(i, 1.0) for i in good], dtype=np.float64)
    v = np.array(list(good.values()), dtype=np.float64)
    return float(np.sum(w * v) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(v))


# ------------------------------------------------------------------ the metric


@dataclass
class SuperMetric:
    """Every term of Eq. 10, and the total. Lower is better."""

    volume: float = float("nan")
    components: float = float("nan")
    euler: float = float("nan")
    cl: float = float("nan")
    bifurcation: float = float("nan")

    graph_volume_um3: float = 0.0
    graph_components: int = 0
    graph_euler_classical: int = 0
    cl_sensitivity: float = float("nan")
    bifurcation_dice: float = float("nan")
    dice_detail: dict = field(default_factory=dict)
    image: ImageTerms | None = None
    options: SuperMetricOptions = CORONARY
    seconds: float = 0.0

    @property
    def total(self) -> float:
        """Sum of the terms. A NaN term (not applicable) is dropped; an infinite
        one -- cl or B of zero -- makes the total infinite, as the paper intends."""
        return core.combine_terms({"V": self.volume, "cc": self.components, "chi": self.euler,
                                   "cl": self.cl, "B": self.bifurcation})

    @property
    def paper_comparable(self) -> bool:
        return self.options.paper_comparable and not (self.image and self.image.tree_chi)

    def describe(self) -> str:
        d = self.dice_detail
        lines = [
            f"M_S = {self.total:.3f}"
            + ("" if self.paper_comparable else "   (options differ from the paper)"),
            f"  V   {self.volume:8.3f}   graph {self.graph_volume_um3 / 1e9:.3f} mm^3"
            + (f" vs image {self.image.volume_um3 / 1e9:.3f} mm^3" if self.image else ""),
            f"  cc  {self.components:8.3f}   graph {self.graph_components}"
            + (f" vs image {self.image.components}" if self.image else ""),
            f"  chi {self.euler:8.3f}   graph {local_euler(self.graph_euler_classical):.0f}"
            + (f" vs {self.image.chi:.0f}" if self.image else "")
            + (" (tree reference)" if self.image and self.image.tree_chi else ""),
            f"  cl  {self.cl:8.3f}   sensitivity {self.cl_sensitivity:.4f}",
            f"  B   {self.bifurcation:8.3f}   dice {self.bifurcation_dice:.3f}"
            + (f" (tp {d['tp']}, fp {d['fp']} [duplicate {d['fp_duplicate']}, isolated "
               f"{d['fp_isolated']}], fn {d['fn']})" if d else ""),
            f"  ({self.seconds:.1f}s)",
        ]
        return "\n".join(lines)

    def row(self, name: str) -> str:
        return (
            f"  {name:<14} {self.total:8.3f} {self.volume:8.3f} {self.components:8.3f} "
            f"{self.euler:8.3f} {self.cl:8.3f} {self.bifurcation:8.3f}  "
            f"cl={self.cl_sensitivity:.4f} B={self.bifurcation_dice:.3f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"  {'candidate':<14} {'M_S':>8} {'V':>8} {'cc':>8} {'chi':>8} "
            f"{'cl':>8} {'B':>8}"
        )


def _tolerances(refs, labels, frame, options: SuperMetricOptions) -> np.ndarray:
    """Match tolerance per reference bifurcation, in um."""
    if options.tolerance == "fixed":
        return np.full(len(refs), float(options.tolerance_value))
    radius = getattr(refs, "radius_um", None)
    if radius is None or len(radius) != len(refs):
        if not isinstance(labels, np.ndarray):
            raise ValueError("radius-based tolerance needs the reference radii: build "
                             "ref_bifurcations with reference_bifurcations(), or pass a "
                             "decoded mask, or use a fixed bb_threshold")
        voxel_size, origin = _frame_geometry(frame)
        radius = np.maximum(core.local_radius(labels > 0, refs, voxel_size, origin),
                            voxel_size.min())
    return core.bifurcation_tolerance(radius, options)


def super_metric(
    graph,
    frame,
    labels,
    image: ImageTerms,
    ref_bifurcations,
    *,
    bb_threshold: float | None = None,
    options: SuperMetricOptions | None = None,
) -> SuperMetric:
    """Score one skeleton against the binary image it came from.

    `image` and `ref_bifurcations` are computed once per segmentation
    (:func:`image_terms`, :func:`reference_bifurcations`) and reused across every
    candidate and every sweep sample. `options` defaults to ``CORONARY``; a
    `bb_threshold` (um) swaps the radius-based match tolerance for a fixed one.
    The chi reference is ``image.chi`` (set by ``image_terms(tree_chi=...)``).
    """
    t0 = time.time()
    options = options_for("coronary") if options is None else options
    if bb_threshold is not None:
        options = SuperMetricOptions(**{**asdict(options), "tolerance": "fixed",
                                        "tolerance_value": float(bb_threshold)})
    excl = options.exclude_invalid_points

    sg = to_skeleton_graph(graph)
    v_s = sg.volume(excl)
    cc_s = sg.n_components()
    chi_class_s = graph_euler_classical(graph)
    cl_s = _cl_from_skeleton(sg, frame, labels, excl)

    refs = np.asarray(ref_bifurcations, dtype=np.float64).reshape(-1, 3)
    tol = _tolerances(ref_bifurcations, labels, frame, options)
    dice = core.match_bifurcations(sg.bifurcations(), refs, tol, None, options.matching)

    return SuperMetric(
        volume=core.relative_term(image.volume_um3, v_s),
        components=core.relative_term(image.components, cc_s),
        euler=core.relative_term(image.chi, local_euler(chi_class_s)),
        cl=core.overlap_term(cl_s, 3) if np.isfinite(cl_s) else float("inf"),
        # NaN Dice (no bifurcation on either side, e.g. a lone tube) = not applicable.
        bifurcation=core.overlap_term(dice.dice, 2),
        graph_volume_um3=v_s,
        graph_components=cc_s,
        graph_euler_classical=chi_class_s,
        cl_sensitivity=cl_s,
        bifurcation_dice=dice.dice,
        dice_detail={
            "tp": dice.tp, "fp": dice.fp, "fn": dice.fn,
            "fp_duplicate": dice.fp_duplicate, "fp_isolated": dice.fp_isolated,
            "n_candidate": dice.tp + dice.fp, "n_reference": len(refs),
            "matching": options.matching,
        },
        image=image,
        options=options,
        seconds=time.time() - t0,
    )
