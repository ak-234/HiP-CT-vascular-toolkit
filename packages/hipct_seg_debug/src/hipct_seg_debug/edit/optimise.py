"""Order a skeleton, correct its radii, and score it against a reference.

Wires up `skeleton_analysis`, which has all of this and is called by nothing here.
Two things about that package are worth stating before using it, because both are
easy to get wrong from the names alone:

* **its ``optimisation`` subpackage does not optimise a skeleton, it scores one.**
  There is no sweep, no objective minimiser -- ``meta_metric`` and ``super_metric``
  are objective *functions*, and the candidates they compare were always meant to
  be generated externally. The functions that actually change a skeleton live in
  ``outlier``.
* **``super_metric`` compares two segmentations, not two skeletons.** Given one
  mask and two centreline graphs, its Volume/CC/Euler terms are identical for both
  and contribute exactly nothing. What discriminates is the bifurcation Dice and
  the per-skeleton centreline sensitivity, so those are what :func:`compare`
  reports; ``super_metric`` is available but is not the headline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ._deps import ensure_skeleton_analysis


def _to_spatial(graph):
    """Accept a Triple, an EditableGraph or a SpatialGraph; return a SpatialGraph."""
    from ..amira import SpatialGraph
    from .adapter import Triple, to_spatial_graph

    if isinstance(graph, SpatialGraph):
        return graph
    if isinstance(graph, Triple):
        return to_spatial_graph(graph)
    if hasattr(graph, "to_spatial_graph"):
        return graph.to_spatial_graph()
    raise TypeError(f"cannot make a SpatialGraph from {type(graph).__name__}")


def as_sa_graph(graph):
    """Convert to the ``skeleton_analysis`` SpatialGraph, which is a different class.

    The two packages parse the same file format into different dataclasses. Rather
    than round-trip through disk, rebuild the fields ``skeleton_analysis`` reads.

    Its accessors -- ``vertex_coords``, ``edge_connectivity`` and so on -- are
    read-only properties backed by three field dicts keyed on the Amira field
    names, so everything goes in through ``set_field``.
    """
    ensure_skeleton_analysis()
    from skeleton_analysis.io.amira import SpatialGraph as SaGraph

    g = _to_spatial(graph)
    sa = SaGraph()
    sa.set_field("VERTEX", "VertexCoordinates", np.asarray(g.vertices, dtype=np.float64))
    sa.set_field("EDGE", "EdgeConnectivity", np.asarray(g.connectivity, dtype=np.int64))
    sa.set_field("EDGE", "NumEdgePoints", np.asarray(g.n_edge_points, dtype=np.int64))
    sa.set_field("POINT", "EdgePointCoordinates", np.asarray(g.points, dtype=np.float64))
    sa.set_field("POINT", "thickness", np.asarray(g.thickness, dtype=np.float64))
    for name, arr in g.edge_attrs.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) == g.n_edge:
            sa.set_field("EDGE", name, arr)
    for name, arr in g.vertex_attrs.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) == g.n_vertex:
            sa.set_field("VERTEX", name, arr)
    return sa


# Public since the root picker and the per-component skeletoniser both need it; the
# private name stays so nothing that already imported it breaks.
_as_sa_graph = as_sa_graph


def _fill_unrooted(sa, chosen, auto) -> list:
    """`chosen` roots, plus `auto`'s for every component `chosen` does not reach.

    One root per component is what ``order_forest`` expects, and a component with
    none keeps order 0 throughout -- indistinguishable from a genuinely unordered
    graph. So a chosen root *replaces* its component's automatic one and every other
    component keeps the automatic pick.
    """
    import networkx as nx

    edges = np.asarray(sa.edge_connectivity, dtype=np.int64)
    g = nx.Graph()
    g.add_nodes_from(range(int(sa.n_vertices)))
    g.add_edges_from((int(a), int(b)) for a, b in edges)

    where = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for nid in comp:
            where[int(nid)] = i

    out, taken = [], set()
    for nid in chosen:
        comp = where.get(int(nid))
        if comp is None or comp in taken:
            continue  # an unknown node, or a second pick in one component
        taken.add(comp)
        out.append(int(nid))
    for nid in auto:
        comp = where.get(int(nid))
        if comp is not None and comp not in taken:
            taken.add(comp)
            out.append(int(nid))
    return out


@dataclass
class OrderingReport:
    n_roots: int
    roots: list
    strahler: np.ndarray
    topo: np.ndarray
    n_flipped: int

    def describe(self) -> str:
        s = self.strahler[self.strahler > 0]
        return (
            f"{self.n_roots} root(s) {self.roots[:5]}"
            + ("..." if len(self.roots) > 5 else "")
            + f"; Strahler 1-{int(s.max()) if len(s) else 0}, "
            f"{self.n_flipped} edge(s) reoriented"
        )


def order(graph, roots=None) -> OrderingReport:
    """Assign Strahler order and topological generation per edge.

    Uses the array functions rather than ``run_ordering``, which reads its input
    from a *path* and would force a disk round-trip for an in-memory graph.

    ``auto_roots`` picks one root per connected component -- the largest-radius
    edge, then whichever of its endpoints has the lower coordination number, so a
    trunk's free end wins over its junction. It reads a per-edge ``MeanRadius``,
    which is synthesised here when the graph does not carry one.

    `roots` supplies chosen roots as **node ids** -- from the interactive picker, by
    way of :mod:`~.roots` -- and any component they do not cover is filled in from
    ``auto_roots``. The filling is not optional: ``order_forest`` leaves the edges of
    an unrooted component at order 0, so a partial pick would silently unorder every
    tree the operator skipped.
    """
    ensure_skeleton_analysis()
    from skeleton_analysis.metrics.radius import mean_radius_per_edge
    from skeleton_analysis.ordering.pipeline import auto_roots, order_forest

    sa = as_sa_graph(graph)
    if "MeanRadius" not in sa.edge_fields:
        sa.edge_fields["MeanRadius"] = mean_radius_per_edge(sa)

    auto = auto_roots(sa)
    roots = _fill_unrooted(sa, roots, auto) if roots else auto
    strahler, topo, flipped = order_forest(sa.edge_connectivity, roots)

    set_edge_field(graph, "strahler", strahler, np.int64)
    set_edge_field(graph, "topo", topo, np.int64)
    set_edge_field(graph, "MeanRadius", sa.edge_fields["MeanRadius"], np.float64,
                   overwrite=False)
    return OrderingReport(len(roots), list(map(int, roots)), strahler, topo, len(flipped))


def set_edge_field(graph, name: str, values, dtype=np.float64,
                   *, overwrite: bool = True) -> None:
    """Attach a per-edge array to whatever kind of graph object this is.

    **Not** ``_to_spatial(graph).edge_attrs[name] = ...``. For a ``Triple`` or an
    ``EditableGraph`` that helper *builds* a fresh ``SpatialGraph`` rather than
    returning a view, so the assignment lands on a temporary that is discarded the
    moment the function returns. That is what made ``skeletonise --order`` write a
    file with no ordering in it, and ``optimise --oblique`` fail looking up a
    ``strahler`` field that had been computed and thrown away.

    A ``Triple`` keeps edge attributes on its segment dicts, which is where
    ``to_spatial_graph`` reads them from on the way back out; ``edge_attr_dtypes``
    is what stops an integer order being written as a float.
    """
    from ..amira import SpatialGraph
    from .adapter import Triple

    values = np.asarray(values, dtype=dtype)

    if isinstance(graph, SpatialGraph):
        if overwrite or name not in graph.edge_attrs:
            graph.edge_attrs[name] = values
        return

    triple = graph.triple if hasattr(graph, "triple") else graph
    if not isinstance(triple, Triple):
        raise TypeError(f"cannot set an edge field on {type(graph).__name__}")
    if len(values) != len(triple.segments):
        raise ValueError(
            f"{name}: {len(values)} values for {len(triple.segments)} segments"
        )
    if not overwrite and all(name in seg for seg in triple.segments):
        return
    triple.edge_attr_dtypes[name] = np.dtype(dtype)
    for seg, value in zip(triple.segments, values):
        seg[name] = value.item()


def correct_radii(graph, lattice=None, *, flagged_only: bool = True,
                  half_size: int = 12, threshold: float = 0.5,
                  method: str = "perimeter", verbose: bool = False):
    """Clean a skeleton's radii, optionally re-measuring them against the image.

    Two passes, matching what ``Python_port_test.py`` does:

    1. ``correct_along_segment_thickness`` -- pure numpy, no image. Replaces
       per-segment radius outliers with the nearest good value. Note this is a
       nearest-*value* copy, not an interpolation, so it flattens a run rather than
       continuing its taper -- for a genuine collapse use
       :mod:`~.radius_repair` instead, which is what it is for.
    2. ``segment_radii_from_volume`` -- re-measures radius from oblique
       cross-sections of the mask. Needs a decoded volume, so it is optional.

    Returns ``(thickness, changed_indices)``; the graph is not modified.
    """
    ensure_skeleton_analysis()
    from skeleton_analysis.outlier.detect import (
        correct_along_segment_thickness,
        detect_collapsed_segments,
    )

    sa = as_sa_graph(graph)
    original = np.asarray(sa.point_fields["thickness"], dtype=np.float64).copy()
    invented = _invented_mask(graph, len(original))

    thickness, changed = correct_along_segment_thickness(sa)
    thickness, changed = _keep_invented(thickness, changed, original, invented)
    if verbose:
        print(f"    along-segment outliers: {len(changed)} point radii replaced")
        if invented.any():
            print(f"    {int(invented.sum())} interpolated radii left exactly as Avizo "
                  "wrote them")

    if lattice is None:
        return thickness, changed

    from skeleton_analysis.metrics.radius import mean_radius_per_edge
    from skeleton_analysis.outlier.oblique import segment_radii_from_volume

    sa.point_fields["thickness"] = thickness
    if "MeanRadius" not in sa.edge_fields:
        sa.edge_fields["MeanRadius"] = mean_radius_per_edge(sa)

    # `detect_collapsed_segments` selects by Strahler order, so an unordered graph
    # has nothing to select on. Re-measuring every segment is the honest fallback --
    # slower, but it answers the question that was asked instead of raising a
    # KeyError about a field the caller never knew existed.
    if flagged_only and "strahler" not in sa.edge_fields:
        if verbose:
            print("    no Strahler order on this graph, so there is nothing to select "
                  "on - call order() first to re-measure only the flagged segments")
        flagged_only = False
    edges = (
        detect_collapsed_segments(sa, flag_orders=(5, 4, 3, 2, 1))
        if flagged_only
        else np.arange(len(sa.edge_connectivity))
    )
    if verbose:
        which = "flagged" if flagged_only else "all"
        print(f"    re-measuring {which} {len(edges)} segment(s) against the image...")

    res = float(np.mean(lattice.spacing))
    starts = np.concatenate([[0], np.cumsum(sa.num_edge_points)[:-1]])
    touched = list(changed)
    for e in edges:
        s, n = int(starts[e]), int(sa.num_edge_points[e])
        centre_vox = lattice.world_to_index_zyx(sa.point_coords[s:s + n])
        rads = segment_radii_from_volume(
            lattice.volume, centre_vox, res=res, half_size=half_size,
            threshold=threshold, method=method,
        )
        ok = np.isfinite(rads)
        thickness[s:s + n][ok] = rads[ok]
        touched.extend((s + np.flatnonzero(ok)).tolist())
    thickness, touched = _keep_invented(
        thickness, np.asarray(touched, dtype=np.int64), original, invented
    )
    return thickness, np.unique(np.asarray(touched, dtype=np.int64))


def _invented_mask(graph, n: int) -> np.ndarray:
    """(n,) bool in point order -- the radii Avizo interpolated."""
    from .interpolation import mask

    try:
        flat = mask(graph)
    except Exception:  # noqa: BLE001 - a graph shape this cannot read is not an error
        return np.zeros(n, dtype=bool)
    if flat.size != n:
        return np.zeros(n, dtype=bool)
    return flat


def _keep_invented(thickness, changed, original, invented):
    """Put invented radii back, and drop them from the list of what was repaired.

    Both passes above go through ``skeleton_analysis``, which knows nothing about this
    field, so the exclusion has to be applied on the way out rather than on the way in.
    That is a real limitation and worth naming: an invented radius still participates in
    the per-segment percentile ``correct_along_segment_thickness`` computes, so it can
    still tilt the threshold applied to its neighbours. What it cannot do any more is
    come back out of here rewritten and counted as a measurement.
    """
    if not invented.any():
        return thickness, changed
    thickness = np.asarray(thickness, dtype=np.float64).copy()
    thickness[invented] = original[invented]
    changed = np.asarray(changed, dtype=np.int64).ravel()
    return thickness, changed[~invented[changed]] if changed.size else changed


@dataclass
class GraphStats:
    name: str
    n_segments: int
    n_nodes: int
    n_points: int
    n_components: int
    n_free_ends: int
    n_bifurcations: int
    total_length_mm: float
    radius_um: tuple  # (p5, median, p95)
    centreline_sensitivity: float = float("nan")

    def row(self) -> str:
        p5, med, p95 = self.radius_um
        return (
            f"  {self.name:<12} {self.n_segments:>6} seg  {self.n_nodes:>6} nodes  "
            f"{self.n_components:>3} comp  {self.n_free_ends:>5} ends  "
            f"{self.n_bifurcations:>5} bifs  {self.total_length_mm:>8.1f} mm  "
            f"r {p5:>5.0f}/{med:>5.0f}/{p95:>5.0f} um  "
            f"sens {self.centreline_sensitivity:.4f}"
        )


def graph_stats(graph, name: str, lattice=None) -> GraphStats:
    """Per-skeleton facts, and its sensitivity against the mask when one is given."""
    from .graphmodel import EditableGraph
    from .adapter import Triple, from_spatial_graph

    from ..amira import SpatialGraph

    if isinstance(graph, EditableGraph):
        g = graph
    elif isinstance(graph, Triple):
        g = EditableGraph(graph.copy())
    elif isinstance(graph, SpatialGraph):
        g = EditableGraph(from_spatial_graph(graph))
    else:
        raise TypeError(type(graph).__name__)

    length = 0.0
    radii = []
    for seg in g.segments:
        coords = g.coords(seg["id"])
        if len(coords) > 1:
            length += float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
        radii.append(g.radii(seg["id"]))
    rad = np.concatenate(radii) if radii else np.zeros(1)

    sensitivity = float("nan")
    if lattice is not None:
        ensure_skeleton_analysis()
        from skeleton_analysis.optimisation.volume_metrics import centreline_sensitivity

        try:
            sensitivity = float(centreline_sensitivity(as_sa_graph(g), lattice))
        except Exception:  # noqa: BLE001 - a metric failing must not lose the rest
            sensitivity = float("nan")

    return GraphStats(
        name=name,
        n_segments=len(g.segments),
        n_nodes=len(g.nodes),
        n_points=len(g.points),
        n_components=len(g.components()),
        n_free_ends=len(g.endpoints()),
        n_bifurcations=sum(1 for n in g.nodes if g.degree(n) >= 3),
        total_length_mm=length / 1000.0,
        radius_um=tuple(float(v) for v in np.percentile(rad, [5, 50, 95])),
        centreline_sensitivity=sensitivity,
    )


@dataclass
class Comparison:
    candidate: GraphStats
    reference: GraphStats
    bifurcation: dict = field(default_factory=dict)
    morphometrics: dict = field(default_factory=dict)
    seconds: float = 0.0

    def describe(self) -> str:
        lines = [
            "skeleton comparison",
            f"  {'':12} {'segments':>10} {'nodes':>12} {'comp':>9} {'ends':>10} "
            f"{'bifs':>10} {'length':>13} {'radius p5/50/95':>26} {'sens':>10}",
            self.reference.row(),
            self.candidate.row(),
        ]
        b = self.bifurcation
        if b:
            lines.append(
                f"\n  bifurcation Dice {b['dice']:.3f}  "
                f"(tp {b['tp']}, fp {b['fp']}, fn {b['fn']}; "
                f"{b['n_candidate']} candidate vs {b['n_reference']} reference, "
                f"threshold {b['threshold']:.0f} um)"
            )
        if self.morphometrics:
            m = self.morphometrics
            lines.append(
                f"  mask: {m.get('connected_components', 0):.0f} components, "
                f"Euler {m.get('euler_number', 0):.0f}, "
                f"volume {m.get('volume', 0)/1e9:.2f} mm^3"
            )
        lines.append(f"  ({self.seconds:.1f}s)")
        return "\n".join(lines)


def compare(candidate, reference, lattice=None, *, bb_threshold: float = 900.0
            ) -> Comparison:
    """Score a generated skeleton against a reference one.

    The headline number is the **bifurcation Dice**: greedy nearest-neighbour
    matching of the two skeletons' branch points within `bb_threshold` um. That is
    the metric that genuinely compares two centrelines of the same object; volume
    measures cannot, because both describe the same mask.
    """
    t0 = time.time()
    ensure_skeleton_analysis()
    from skeleton_analysis.optimisation.meta_metric import (
        bifurcation_dice_points,
        bifurcation_points,
    )

    cand_stats = graph_stats(candidate, "candidate", lattice)
    ref_stats = graph_stats(reference, "reference", lattice)

    cand_pts = bifurcation_points(as_sa_graph(candidate))
    ref_pts = bifurcation_points(as_sa_graph(reference))
    dice = bifurcation_dice_points(cand_pts, ref_pts, threshold=bb_threshold)

    morph = {}
    if lattice is not None:
        from skeleton_analysis.optimisation.volume_metrics import region_morphometrics

        try:
            morph = dict(region_morphometrics(
                lattice.volume, voxel_size=float(np.mean(lattice.spacing))
            ))
        except Exception:  # noqa: BLE001
            morph = {}

    return Comparison(
        candidate=cand_stats,
        reference=ref_stats,
        bifurcation={
            "dice": float(dice.dice), "tp": int(dice.tp), "fp": int(dice.fp),
            "fn": int(dice.fn), "n_candidate": int(dice.n_candidate),
            "n_reference": int(dice.n_reference), "threshold": float(bb_threshold),
        },
        morphometrics=morph,
        seconds=time.time() - t0,
    )
