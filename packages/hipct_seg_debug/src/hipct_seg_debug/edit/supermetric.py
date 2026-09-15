"""The Walsh-Berg super metric: how much did this skeletonisation lose?

Faithful implementation of Eq. 10 of

    C.L. Walsh, M. Berg, H. West, N.A. Holroyd, S. Walker-Samuel, R.J. Shipley,
    "Reconstructing microvascular network skeletons from 3D images: What is the
    ground truth?", Computers in Biology and Medicine 171 (2024) 108140.

``docs/SKELETONISATION.md`` carries the full translation, the two
deliberate departures, and the table of where ``skeleton_analysis`` diverges from the
paper. The short version:

There is no ground-truth skeleton -- the only published manual consensus centrelines
took over 500 hours of expert time for three coronary branches -- so the *segmentation*
is the gold standard, and the question becomes how much of its information the
skeletonisation threw away. Five terms, each comparing the graph ``S`` to the binary
image ``I``::

    M_S = |V_I - V_S|/V_I + |cc_I - cc_S|/cc_I + |chi_I - chi_S|/chi_I
          + |1 - cl_S| / cl_S**3   +   |1 - B_S| / B_S**2

Lower is better; zero is identical.

**The non-linear weights are the design, not a detail.** ``cl`` enters as ``1/cl**3``,
so even a small drop in the fraction of centreline lying inside the mask dominates the
sum and rejects the skeleton: an algorithm may reasonably cut a corner, but long
stretches of centreline *outside* the segmentation never plausible. ``B`` is weighted
more softly at ``1/B**2`` because its reference is annotated rather than measured.
:func:`super_metric` therefore returns every term separately as well as the total --
the paper's own use of it (Fig. 8B) is to read which term dominates, which says *how*
an algorithm is failing. For Amira AutoSkeleton that was the volume term, pinning the
fault on its 1/5-Chamfer radius estimator rather than on its topology.

Why this is not ``skeleton_analysis.optimisation.super_metric``: that function computes
``V``, ``cc`` and ``chi`` from **two volumes**, so as :mod:`~.optimise` already records,
"its Volume/CC/Euler terms are identical for both and contribute exactly nothing" when
comparing two skeletons of one mask. Taking them from the **graph** instead, per the
paper's Table S13, is what makes the metric an objective a skeleton can be optimised
against. It also uses an unweighted RMS, a branch-point *count* where the paper uses
bifurcation *Dice*, and a point-sampled ``cl`` where the paper rasterises the lines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# The paper's local Euler characteristic, reformulated from the classical one so that a
# network with a single loop (chi_classical == 0) does not make the relative difference
# undefined. Supplementary section 7 prints ``chi = -chi_classical - 2``, which is
# negative for a tree and so contradicts its own "always positive" requirement; the main
# text's form below is the one used. The same transform is applied to image and graph,
# which is the only way their relative difference means anything.
def local_euler(chi_classical: float) -> float:
    """``2 - chi_classical`` -- strictly positive for any graph or solid."""
    return 2.0 - float(chi_classical)


# ------------------------------------------------------------------ graph terms


def graph_volume(graph) -> float:
    """Total network volume in um^3, as the sum of subsegment frusta.

    Table S13's "sum of volume of all subsegments". A subsegment is the span between
    two consecutive centreline points, so its volume is taken as a cylinder of the
    mean of the two radii -- which is what makes this term sensitive to the radius
    estimator, and is why it is the term that exposed AutoSkeleton's 1/5-Chamfer
    approximation in the paper.

    Subsegments Avizo interpolated are left out. Both of this term's inputs are
    fabricated there -- an invented length carrying an invented radius -- so counting
    them lets a graph score better for containing more of what it made up.
    """
    total = 0.0
    for sid in graph.segment_ids():
        coords = graph.coords(sid)
        radii = graph.radii(sid)
        if len(coords) < 2:
            continue
        lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        mean_r = 0.5 * (radii[:-1] + radii[1:])
        keep = _real_subsegments(graph, sid, len(coords))
        total += float(np.sum(np.pi * mean_r[keep] * mean_r[keep] * lengths[keep]))
    return total


def _real_subsegments(graph, sid: int, n: int) -> np.ndarray:
    """(n-1,) bool -- spans between two consecutive points that were actually skeletonised.

    A span counts as invented if *either* end is, which is the conservative reading:
    the half-step from a real anchor into a fill is still following the fill.
    """
    from .interpolation import mask_for_segment

    invented = mask_for_segment(graph, sid)
    if invented.size != n or not invented.any():
        return np.ones(max(n - 1, 0), dtype=bool)
    return ~(invented[:-1] | invented[1:])


def graph_components(graph) -> int:
    """Number of subnetworks -- ``cc`` computed from the graph."""
    return len(graph.components())


def graph_euler_classical(graph, segs=None) -> int:
    """``N - E`` of the **largest** subgraph, the graph form of the Euler number.

    Table S13: "For the largest subgraph no. of nodes - no. of segments". A tree gives
    1, and each independent loop subtracts one, so this counts the graph's loops the
    way tunnels count in the solid.

    Largest by segment count, matching ``components()``' own ordering, because the
    paper's concern is the component carrying the blood -- Supplementary Figure S2
    shows the largest connected component holds the great majority of the volume.

    `segs` scores one nominated component's segment ids instead. On a mask holding a
    left *and* a right coronary tree the default measures the left one alone, so a
    change confined to the right moves this term not at all -- which is why
    :func:`super_metric_per_tree` passes each tree explicitly.
    """
    if segs is not None:
        biggest = set(int(s) for s in segs)
    else:
        comps = graph.components()
        if not comps:
            return 0
        biggest = comps[0]
    nodes = set()
    for sid in biggest:
        seg = graph.segment(sid)
        nodes.add(seg["node1"])
        nodes.add(seg["node2"])
    return len(nodes) - len(biggest)


def rasterise_centreline(graph, frame) -> np.ndarray:
    """Unique ``(iz, iy, ix)`` voxels covered by the centreline, Bresenham-style.

    The paper transforms the spatial graph "to a binary image of lines via Bresenham
    algorithm" before measuring overlap, and the distinction matters. ``skeleton_
    analysis``'s ``centreline_sensitivity`` instead samples the mask *at the graph's
    points*; on a smoothed Avizo graph consecutive points are many voxels apart, so
    that scores a centreline as wholly inside the mask even when the straight segment
    between two of its points leaves the vessel entirely.

    Integer DDA rather than the textbook error-accumulating Bresenham: stepping the
    longest axis one voxel at a time and rounding the other two gives the same
    26-connected voxel chain in three dimensions, in vectorised form.
    """
    ijk_all: list[np.ndarray] = []
    for sid in graph.segment_ids():
        ijk_all.extend(_rasterise_segment_ijk(graph, sid, frame))
    if not ijk_all:
        return np.empty((0, 3), dtype=np.int64)
    flat = np.vstack(ijk_all)
    # Stored as (iz, iy, ix) to index a volume directly.
    zyx = flat[:, ::-1]
    return np.unique(zyx, axis=0)


def _rasterise_segment_ijk(graph, sid, frame) -> list[np.ndarray]:
    """The DDA chunks for one segment, as ``(i, j, k)`` -- see :func:`rasterise_centreline`."""
    out: list[np.ndarray] = []
    coords = graph.coords(sid)
    if len(coords) == 0:
        return out
    ijk = frame.um_to_seg(coords)  # (N, 3) as (i, j, k) = (x, y, z) indices
    if len(ijk) == 1:
        out.append(np.rint(ijk).astype(np.int64))
        return out
    a, b = ijk[:-1], ijk[1:]
    # An invented span is not centreline, so it neither earns credit for lying
    # inside the mask nor is penalised for leaving it.
    keep = _real_subsegments(graph, sid, len(coords))
    a, b = a[keep], b[keep]
    steps = np.ceil(np.abs(b - a).max(axis=1)).astype(np.int64)
    for p, q, n in zip(a, b, steps):
        if n <= 0:
            out.append(np.rint(p).astype(np.int64)[None, :])
            continue
        t = np.linspace(0.0, 1.0, int(n) + 1)[:, None]
        out.append(np.rint(p[None, :] * (1.0 - t) + q[None, :] * t).astype(np.int64))
    return out


def rasterise_segment(graph, sid, frame) -> np.ndarray:
    """Unique ``(iz, iy, ix)`` voxels covered by one segment's centreline.

    The per-segment half of :func:`rasterise_centreline`, so a caller that needs to
    know *which* segment covered a voxel -- assigning each edge to a mask component,
    for one -- does not have to re-implement the DDA.
    """
    chunks = _rasterise_segment_ijk(graph, sid, frame)
    if not chunks:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.vstack(chunks)[:, ::-1], axis=0)


def _sample_mask(labels, zyx: np.ndarray, dims) -> np.ndarray:
    """Mask value at each ``(iz, iy, ix)``; accepts a decoded array or an RLE lattice."""
    nx, ny, nz = (int(v) for v in dims)
    inb = (
        (zyx[:, 0] >= 0) & (zyx[:, 0] < nz)
        & (zyx[:, 1] >= 0) & (zyx[:, 1] < ny)
        & (zyx[:, 2] >= 0) & (zyx[:, 2] < nx)
    )
    out = np.zeros(len(zyx), dtype=bool)
    sel = zyx[inb]
    if len(sel) == 0:
        return out
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
    res = np.zeros(len(zyx), dtype=bool)
    res[inb] = vals > 0
    return res


def cl_sensitivity(graph, frame, labels) -> float:
    """Fraction of the rasterised centreline lying inside the segmentation.

    ``sum(V_I . l_S) / sum(l_S)`` of Table S13 -- the sensitivity half of clDice, and
    the term the ``1/cl**3`` weight makes dominant.
    """
    zyx = rasterise_centreline(graph, frame)
    if len(zyx) == 0:
        return float("nan")
    inside = _sample_mask(labels, zyx, frame.seg_dims)
    return float(inside.sum() / len(zyx))


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

    ``connectivity=3`` is 26-connectivity, which is what the paper specifies for ``cc``.
    The Euler number is taken on the **largest component alone**, per Table S13; on the
    whole volume it would be dominated by however many specks of debris the
    segmentation left behind.

    `labels` and `component` score one already-labelled component instead of
    relabelling and taking the biggest -- the per-tree path, where "largest" is the
    wrong question because each tree is scored on its own terms.
    """
    from skimage.measure import euler_number, label

    t0 = time.time()
    binary = np.asarray(volume) > 0
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

    n_vox = int(np.count_nonzero(binary))
    labelled = label(binary, connectivity=connectivity)
    n_cc = int(labelled.max())
    if n_cc == 0:
        return ImageTerms(0.0, 0, 0, 0, tree_chi, time.time() - t0)

    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    biggest = int(np.argmax(counts))
    chi = int(euler_number(labelled == biggest, connectivity=connectivity))

    return ImageTerms(
        volume_um3=n_vox * voxel_volume,
        components=n_cc,
        euler_classical=chi,
        voxel_count=n_vox,
        tree_chi=tree_chi,
        seconds=time.time() - t0,
    )


def super_metric_per_tree(graph, frame, labels_zyx, parts, *, bb_threshold: float = 900.0,
                          tree_chi: bool = True, verbose: bool = False) -> dict:
    """``{tree index: SuperMetric}`` -- each tree scored against its own component.

    The default :func:`super_metric` measures ``chi`` on the largest component alone,
    on both the graph side and the image side. With a left and a right coronary tree
    in one mask that means the right one contributes nothing to that term, so a sweep
    optimising ``M_S`` is partly blind to half the anatomy. Here each tree is the whole
    of its own gold standard.

    **The numbers are not comparable with the whole-graph ones**, and not only because
    they are per tree. ``cc`` in particular becomes ``|1 - cc_s| / 1``, so a tree that
    skeletonised into three fragments scores 2.0 where the same fragmentation against
    a global count of hundreds was a small ratio. That is a harsher and arguably
    truer reading, which is exactly why it is opt-in.

    `labels_zyx` is the connected-component labelling the parts came from -- the
    ``stats.labels`` of :func:`~.components.split_components` -- at the same
    resolution as `frame`.
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
                                       bb_threshold=bb_threshold)
        del mask
    return out


def aggregate(per_tree: dict, parts=None, *, objective: str = "weighted") -> float:
    """Combine per-tree scores into one number for a sweep to minimise.

    ``weighted`` -- by voxel count -- is the default because the alternatives both
    mislead: an unweighted ``mean`` lets a 3000-voxel fragment outvote the left main,
    and a plain sum rescales with the number of trees, so a sweep on one mask could
    not be compared with a sweep on another.
    """
    totals = {i: m.total for i, m in per_tree.items()}
    good = {i: t for i, t in totals.items() if np.isfinite(t)}
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


def reference_bifurcations(volume, frame, *, stride: int = 1) -> np.ndarray:
    """Bifurcation points of the segmentation's own skeleton, in um ``(x, y, z)``.

    The paper annotates these by hand in a subvolume. Automating it from the mask keeps
    the gold standard the binary image, as intended, and makes a parameter sweep
    runnable without a day of annotation first -- at the cost of inheriting Lee
    thinning's own errors, which is why ``B`` is the softly weighted term.

    Delegates to ``skeleton_analysis``'s tested implementation, which crops to the
    non-zero bounding box, finds voxels with >= 3 skeleton neighbours, and clusters
    them so one anatomical branch point yields one coordinate rather than a blob.
    """
    from ._deps import ensure_skeleton_analysis
    from .lattice import LatticeView

    ensure_skeleton_analysis()
    from skeleton_analysis.optimisation.volume_metrics import skeleton_junction_points

    view = LatticeView.from_frame(np.asarray(volume), frame, stride=stride)
    return np.asarray(skeleton_junction_points(view), dtype=np.float64).reshape(-1, 3)


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
    seconds: float = 0.0

    @property
    def total(self) -> float:
        terms = [self.volume, self.components, self.euler, self.cl, self.bifurcation]
        good = [t for t in terms if np.isfinite(t)]
        return float(np.sum(good)) if good else float("nan")

    def describe(self) -> str:
        d = self.dice_detail
        lines = [
            f"M_S = {self.total:.3f}",
            f"  V   {self.volume:8.3f}   graph {self.graph_volume_um3 / 1e9:.3f} mm^3"
            + (f" vs image {self.image.volume_um3 / 1e9:.3f} mm^3" if self.image else ""),
            f"  cc  {self.components:8.3f}   graph {self.graph_components}"
            + (f" vs image {self.image.components}" if self.image else ""),
            f"  chi {self.euler:8.3f}   graph {local_euler(self.graph_euler_classical):.0f}"
            + (f" vs {self.image.chi:.0f}" if self.image else "")
            + (" (tree reference)" if self.image and self.image.tree_chi else ""),
            f"  cl  {self.cl:8.3f}   sensitivity {self.cl_sensitivity:.4f}",
            f"  B   {self.bifurcation:8.3f}   dice {self.bifurcation_dice:.3f}"
            + (f" (tp {d['tp']}, fp {d['fp']}, fn {d['fn']})" if d else ""),
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


def _relative(image_value: float, graph_value: float) -> float:
    """``|f_I - f_S| / f_I``, or NaN when the reference is zero."""
    if not np.isfinite(image_value) or image_value == 0:
        return float("nan")
    return float(abs(image_value - graph_value) / abs(image_value))


def super_metric(
    graph,
    frame,
    labels,
    image: ImageTerms,
    ref_bifurcations,
    *,
    bb_threshold: float = 900.0,
) -> SuperMetric:
    """Score one skeleton against the binary image it came from.

    `image` and `ref_bifurcations` are computed once per segmentation
    (:func:`image_terms`, :func:`reference_bifurcations`) and reused across every
    candidate and every sweep sample -- they are the expensive half, and they do not
    depend on the skeleton.
    """
    from ._deps import ensure_skeleton_analysis

    t0 = time.time()
    ensure_skeleton_analysis()
    from skeleton_analysis.optimisation.meta_metric import bifurcation_dice_points

    v_s = graph_volume(graph)
    cc_s = graph_components(graph)
    chi_class_s = graph_euler_classical(graph)
    cl_s = cl_sensitivity(graph, frame, labels)

    cand = np.array(
        [graph.nodes[n][:3] for n in graph.nodes if graph.degree(n) >= 3],
        dtype=np.float64,
    ).reshape(-1, 3)
    dice = bifurcation_dice_points(
        cand, np.asarray(ref_bifurcations, dtype=np.float64).reshape(-1, 3),
        threshold=bb_threshold,
    )
    b_s = float(dice.dice)

    # w_cl = 1/cl^3 and w_B = 1/B^2 applied to |1 - x|, exactly as Eq. 10. Both blow up
    # as the measure falls away from 1, which is the intent: it is what rejects a
    # spatially wrong skeleton however good its bulk statistics look.
    cl_term = (
        abs(1.0 - cl_s) / cl_s ** 3 if np.isfinite(cl_s) and cl_s > 0 else float("inf")
    )

    # A Dice of zero means the skeleton put its branch points somewhere else entirely,
    # which is infinitely bad. A Dice that is *undefined* -- neither the skeleton nor
    # the mask has a single bifurcation, as in a lone unbranched tube -- means the term
    # does not apply, and NaN drops it from the total rather than condemning the
    # candidate. The paper takes the same line, using only the keys "present in
    # reference with a non-zero value".
    if dice.n_candidate == 0 and dice.n_reference == 0:
        b_term = float("nan")
    elif not np.isfinite(b_s) or b_s <= 0:
        b_term = float("inf")
    else:
        b_term = abs(1.0 - b_s) / b_s ** 2

    return SuperMetric(
        volume=_relative(image.volume_um3, v_s),
        components=_relative(image.components, cc_s),
        euler=_relative(image.chi, local_euler(chi_class_s)),
        cl=cl_term,
        bifurcation=b_term,
        graph_volume_um3=v_s,
        graph_components=cc_s,
        graph_euler_classical=chi_class_s,
        cl_sensitivity=cl_s,
        bifurcation_dice=b_s,
        dice_detail={
            "tp": int(dice.tp), "fp": int(dice.fp), "fn": int(dice.fn),
            "n_candidate": int(dice.n_candidate), "n_reference": int(dice.n_reference),
        },
        image=image,
        seconds=time.time() - t0,
    )
