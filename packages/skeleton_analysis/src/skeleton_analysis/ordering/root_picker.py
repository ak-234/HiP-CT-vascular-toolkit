"""Interactive per-tree **root** picker for Strahler ordering.

One PyVista window per tree (connected component); each vessel segment is drawn
as **cross-section contour rings** (on a parallel-transported Frenet frame) plus
its **skeletonisation centreline points** (dots), coloured by **Strahler order**
(with a legend). Left-click the inlet segment of the tree to choose it — the
picked segment's degree-1 (proximal) endpoint becomes that tree's **root** for
Strahler/topological reordering. This is the in-package replacement for the
external ``coronary_sdf`` picker the driver used to borrow; unlike that "prune"
picker, the click here **assigns a root** (highlighted green), it does not mark
anything for removal.

The pure helpers (:func:`tree_components`, :func:`root_from_edge`,
:func:`strahler_edge_colors`) are GUI-free and unit-tested. :func:`pick_roots`
needs the optional ``[viz3d]`` (PyVista) + ``[viz]`` (matplotlib) extras, both
imported **lazily** so ``import skeleton_analysis.ordering`` never requires them.
``off_screen=True`` renders the first tree headlessly to a PNG (tests / capture).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from skeleton_analysis.graph.neighbors import coordination_number
from skeleton_analysis.io.amira import SpatialGraph

_GREY = (0.72, 0.72, 0.72)
_GREEN = (0.15, 0.80, 0.20)


# ── pure, GUI-free helpers ────────────────────────────────────────────────────


def _edge_starts(nump: np.ndarray) -> np.ndarray:
    """Start index of each edge's point run in the flat point arrays."""
    nump = np.asarray(nump, dtype=np.int64)
    return np.concatenate([[0], np.cumsum(nump)[:-1]]).astype(np.int64)


def tree_components(graph: SpatialGraph) -> List[np.ndarray]:
    """Connected components as arrays of **global edge indices**, largest first.

    Mirrors the ``networkx`` component pattern in
    :func:`skeleton_analysis.ordering.pipeline.auto_roots`, but returns the edges
    (not the nodes) of each tree so the picker can render one window per tree.
    """
    import networkx as nx

    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)
    g = nx.Graph()
    g.add_nodes_from(range(int(graph.n_vertices)))
    g.add_edges_from((int(a), int(b)) for a, b in edges)

    out: List[np.ndarray] = []
    for comp in sorted(nx.connected_components(g), key=len, reverse=True):
        comp_set = set(comp)
        mask = np.array(
            [int(a) in comp_set and int(b) in comp_set for a, b in edges], dtype=bool
        )
        idx = np.flatnonzero(mask)
        if idx.size:
            out.append(idx)
    return out


def root_from_edge(edge_idx: int, edges: np.ndarray, coord: dict) -> int:
    """Root node of the tree whose inlet is edge ``edge_idx``.

    The root is the edge endpoint with the smaller coordination number (the
    degree-1 proximal inlet); ties resolve to the first endpoint. Identical rule
    to :func:`skeleton_analysis.ordering.pipeline.auto_roots`.
    """
    a, b = int(edges[edge_idx, 0]), int(edges[edge_idx, 1])
    return a if coord.get(a, 0) <= coord.get(b, 0) else b


def strahler_edge_colors(
    graph: SpatialGraph, strahler_field: str = "strahler"
) -> Tuple[Callable[[int], tuple], List[Tuple[str, tuple]]]:
    """``(color_fn, legend)`` colouring each edge by Strahler order.

    ``color_fn(edge_idx) -> (r, g, b)`` maps an edge to a discrete ``tab10``
    colour by its Strahler order; ``legend`` is ``[(f"order {k}", rgb), …]`` for
    ``pl.add_legend``. Falls back to flat grey + an empty legend when the Strahler
    field is absent. Colour convention matches
    ``coronary_sdf.manual_prune.build_segment_colors``.
    """
    if strahler_field not in graph.edge_fields:
        return (lambda i: _GREY), []

    import matplotlib.pyplot as plt

    arr = np.asarray(graph.edge_fields[strahler_field]).astype(np.int64)
    if arr.size == 0:
        return (lambda i: _GREY), []
    lo, hi = int(arr.min()), int(arr.max())
    ncol = max(hi - lo + 1, 1)
    cmap = plt.get_cmap("tab10", ncol)
    colors = {i: tuple(cmap(int(arr[i]) - lo)[:3]) for i in range(arr.size)}
    legend = [(f"order {k}", tuple(cmap(k - lo)[:3])) for k in range(lo, hi + 1)]

    def fn(i: int, _c=colors) -> tuple:
        return _c.get(i, _GREY)

    return fn, legend


def _scene_diag(pts: np.ndarray) -> float:
    if pts.size == 0:
        return 1.0
    span = pts.max(axis=0) - pts.min(axis=0)
    d = float(np.linalg.norm(span))
    return d if d > 0 else 1.0


def compute_frenet_frame(
    tangent: np.ndarray, prev_normal: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable parallel-transport frame ``(t, n, b)`` for a tangent.

    Port of ``coronary_sdf.splines.compute_frenet_frame``: the tangent is
    normalised; when a ``prev_normal`` is supplied it is re-orthogonalised
    against ``t`` (parallel transport, keeping the frame from spinning), and
    dropped only if it becomes degenerate. Otherwise the normal is seeded from
    the coordinate axis least aligned with ``t``. ``b = t x n``.
    """
    t = np.asarray(tangent, dtype=float)
    t = t / max(float(np.linalg.norm(t)), 1e-12)

    n = None
    if prev_normal is not None:
        n = np.asarray(prev_normal, dtype=float)
        n = n - np.dot(n, t) * t
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-6:
            n = n / n_norm
        else:
            n = None

    if n is None:
        candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        best = candidates[int(np.argmin(np.abs(candidates @ t)))]
        n = best - np.dot(best, t) * t
        n = n / max(float(np.linalg.norm(n)), 1e-12)

    b = np.cross(t, n)
    return t, n, b


def _segment_contour_mesh(pv, coords, radii, n_sides: int = 16, ring_stride: int = 1):
    """Cross-section contour rings + centreline points as one ``pv.PolyData``.

    Port of ``coronary_sdf.epicardial_annotation._segment_contour_mesh``. Per
    (subsampled) centreline point, a closed ``n_sides``-gon of radius ``r``
    perpendicular to the local tangent (parallel-transported Frenet frame). All
    centreline points are added as vertex cells so every skeletonisation point
    renders as a dot, regardless of ``ring_stride``. ``pv`` is passed in so the
    module never imports PyVista at top level.
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(coords)
    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    cos_t, sin_t = np.cos(theta)[:, None], np.sin(theta)[:, None]

    ring_pts: List[np.ndarray] = []
    ring_lines: List[int] = []
    offset = 0
    prev_normal = None
    sample = list(range(0, n, max(1, ring_stride)))
    if n >= 2 and (n - 1) not in sample:
        sample.append(n - 1)
    for i in sample:
        if n < 2:
            break
        if i == 0:
            tang = coords[1] - coords[0]
        elif i == n - 1:
            tang = coords[-1] - coords[-2]
        else:
            tang = coords[i + 1] - coords[i - 1]
        if np.linalg.norm(tang) < 1e-12 or radii[i] <= 0:
            continue
        _t, n_hat, b_hat = compute_frenet_frame(tang, prev_normal)
        prev_normal = n_hat
        ring = coords[i] + radii[i] * (cos_t * n_hat + sin_t * b_hat)
        ring_pts.append(ring)
        ring_lines.append(n_sides + 1)
        ring_lines.extend(range(offset, offset + n_sides))
        ring_lines.append(offset)  # close the loop
        offset += n_sides

    all_pts = np.vstack(ring_pts + [coords]) if ring_pts else coords
    poly = pv.PolyData(all_pts)
    if ring_lines:
        poly.lines = np.asarray(ring_lines, dtype=np.int64)
    # Vertex cells for the centreline points (they begin at ``offset``).
    verts = np.empty((n, 2), dtype=np.int64)
    verts[:, 0] = 1
    verts[:, 1] = np.arange(offset, offset + n)
    poly.verts = verts.ravel()
    return poly


def _segment_line_mesh(pv, coords):
    """The graph itself: one polyline along the segment, a vertex at each end.

    No radius is consulted, so nothing is swept, tessellated or triangulated -- the
    mesh is the ``n`` centreline points and ``n - 1`` line cells. On a whole coronary
    tree that is the difference between a scene that orbits at once and one that
    redraws in steps, which matters because the picker's whole job is to let someone
    look around before clicking.

    The vertex cells are the segment's two **nodes**: an edge's first and last
    centreline points are its graph vertices, so they need no separate lookup. They
    are what makes a bifurcation visible when every branch is a bare line.
    """
    n = len(coords)
    poly = pv.PolyData(np.asarray(coords, dtype=float))
    if n >= 2:
        poly.lines = np.hstack([[n], np.arange(n, dtype=np.int64)]).astype(np.int64)
        poly.verts = np.array([1, 0, 1, n - 1], dtype=np.int64)
    else:
        poly.verts = np.array([1, 0], dtype=np.int64)
    return poly


# ── interactive picker ────────────────────────────────────────────────────────


def pick_roots(
    graph: SpatialGraph,
    color_by: str = "strahler",
    style: str = "contour",
    n_sides: int = 16,
    ring_stride: int = 1,
    tube_radius: Optional[float] = None,
    hover: bool = True,
    off_screen: bool = False,
    screenshot: Optional[str] = None,
    preselect: Optional[int] = None,
) -> List[int]:
    """Interactively choose one Strahler **root** per tree.

    Opens one window per connected component (largest first). Each segment is
    drawn (by default) as **Frenet cross-section contour rings + its
    skeletonisation centreline points**, coloured by Strahler order (legend
    shown); **left-click the inlet (root) segment** to select it (turns green),
    ``q`` to confirm and go to the next tree, ``x`` to stop, ``c`` to clear. The
    root of each selected tree is the clicked segment's degree-1 endpoint
    (:func:`root_from_edge`).

    Parameters
    ----------
    style : {"contour", "tube", "lines"}
        ``"contour"`` (default): contour rings on a parallel-transported Frenet
        frame plus centreline dots. ``"tube"``: a single solid tube per segment
        (a lighter / easier-to-click fallback). ``"lines"``: the graph itself --
        one polyline per edge and a point at each node, nothing swept and no
        radius consulted. The lightest of the three by a wide margin, and the one
        to reach for on a whole coronary tree.
    n_sides, ring_stride : int
        Contour tessellation: ``n_sides`` per ring, one ring every
        ``ring_stride`` centreline points (``1`` = a ring at every point). Dots
        are drawn at every centreline point regardless of ``ring_stride``.
    tube_radius : float, optional
        Fixed radius (world units) for the tube style / ring floor; by default
        each ring uses its own per-point ``thickness``.
    off_screen, screenshot, preselect :
        Headless mode: render only the first tree to ``screenshot`` (PNG). If
        ``preselect`` (a global edge index) is given it is drawn green and its
        root returned; no interaction happens. Used by the tests.

    Returns one global root node id per tree the user selected, in tree order.
    Requires the ``[viz3d]`` + ``[viz]`` extras (PyVista + matplotlib).
    """
    import pyvista as pv  # lazy: [viz3d] extra

    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)
    nump = np.asarray(graph.num_edge_points, dtype=np.int64)
    starts = _edge_starts(nump)
    pcoords = np.asarray(graph.point_coords, dtype=float)
    coord = coordination_number(edges)
    try:
        thickness = np.asarray(graph.thickness, dtype=float)
    except Exception:
        thickness = np.zeros(len(pcoords), dtype=float)

    color_fn, legend = (
        strahler_edge_colors(graph) if color_by == "strahler" else ((lambda i: _GREY), [])
    )
    floor = 0.0015 * _scene_diag(pcoords)

    def edge_points(e: int) -> np.ndarray:
        s, n = int(starts[e]), int(nump[e])
        return pcoords[s : s + n]

    def edge_radii(e: int) -> np.ndarray:
        """Per-point ring radius for edge ``e`` (thickness, floored, nan->floor)."""
        if tube_radius is not None:
            n = int(nump[e])
            return np.full(n, float(tube_radius))
        s, n = int(starts[e]), int(nump[e])
        seg = thickness[s : s + n].astype(float)
        seg = np.where(np.isfinite(seg), seg, floor)
        return np.maximum(seg, floor)

    def edge_tube_radius(e: int) -> float:
        if tube_radius is not None:
            return float(tube_radius)
        seg = edge_radii(e)
        return float(np.mean(seg)) if seg.size else floor

    comps = tree_components(graph)
    roots: List[int] = []

    for ti, comp in enumerate(comps):
        picked = _pick_one_tree(
            pv, graph, comp, ti, len(comps), edges, coord, edge_points,
            edge_radii, edge_tube_radius, color_fn, legend,
            style=style, n_sides=n_sides, ring_stride=ring_stride, hover=hover,
            off_screen=off_screen, screenshot=screenshot, preselect=preselect,
        )
        if picked is not None:
            roots.append(int(root_from_edge(picked, edges, coord)))
        if off_screen:
            break  # headless mode renders only the first tree
    return roots


def _pick_one_tree(
    pv,
    graph: SpatialGraph,
    comp: np.ndarray,
    tree_i: int,
    n_trees: int,
    edges: np.ndarray,
    coord: dict,
    edge_points,
    edge_radii,
    edge_tube_radius,
    color_fn,
    legend,
    style: str = "contour",
    n_sides: int = 16,
    ring_stride: int = 1,
    hover: bool = True,
    off_screen: bool = False,
    screenshot: Optional[str] = None,
    preselect: Optional[int] = None,
) -> Optional[int]:
    """Render one tree; return the picked **global edge index** (or ``None``)."""
    from scipy.spatial import cKDTree

    pl = pv.Plotter(off_screen=off_screen,
                    title=f"Root selection - tree {tree_i + 1}/{n_trees}")
    pl.set_background("white")

    vis: dict = {}                 # edge idx -> actor
    tag_pts: List[np.ndarray] = []
    tag_idx: List[np.ndarray] = []
    for e in comp:
        e = int(e)
        pts = edge_points(e)
        if len(pts) >= 2 and style == "contour":
            mesh = _segment_contour_mesh(pv, pts, edge_radii(e),
                                         n_sides=n_sides, ring_stride=ring_stride)
        elif style == "lines" and len(pts) >= 1:
            # Before the tube branch, and taking the single-point case too: a lone
            # point is a vertex here, not a sphere, or "lines" would still tessellate.
            mesh = _segment_line_mesh(pv, pts)
        elif len(pts) >= 2:
            mesh = pv.lines_from_points(pts).tube(radius=edge_tube_radius(e))
        elif len(pts) == 1:
            mesh = pv.Sphere(radius=edge_tube_radius(e), center=pts[0])
        else:
            continue
        mesh.field_data["edge_idx"] = np.array([e], dtype=np.int64)
        # "lines" draws a node for every segment end -- 7,400 of them on one coronary
        # tree -- so they are flat and small there. At the 6 px spheres the other
        # styles use they cover the edges completely, and the point of this style is
        # to see the edges. The line is widened to stay the dominant mark.
        wire = style == "lines"
        vis[e] = pl.add_mesh(mesh, color=color_fn(e), pickable=True,
                             render_points_as_spheres=not wire,
                             point_size=3.0 if wire else 6.0,
                             line_width=3.0 if wire else 2.0)
        mp = np.asarray(mesh.points, dtype=float)
        if len(mp):
            tag_pts.append(mp)
            tag_idx.append(np.full(len(mp), e, dtype=np.int64))

    pick_pts = np.vstack(tag_pts) if tag_pts else np.empty((0, 3))
    pick_tag = np.concatenate(tag_idx) if tag_idx else np.empty(0, dtype=np.int64)
    pick_kd = cKDTree(pick_pts) if len(pick_pts) else None

    state = {"selected": None, "txt": None}

    def recolor(e: Optional[int]) -> None:
        a = vis.get(e)
        if a is not None:
            try:
                a.prop.color = _GREEN if e == state["selected"] else color_fn(e)
            except Exception:
                pass

    def update_text() -> None:
        sel = state["selected"]
        lines = [
            f"ROOT SELECTION  -  TREE {tree_i + 1}/{n_trees}  ({len(comp)} segments)",
            "CLICK the inlet (root) segment (green) | DRAG=rotate",
            "c=clear | q=confirm+next tree | x=stop",
            (f"selected root edge: {sel}  -> node {root_from_edge(sel, edges, coord)}"
             if sel is not None else "selected root edge: (none)"),
        ]
        if state["txt"] is not None:
            pl.remove_actor(state["txt"])
        state["txt"] = pl.add_text("\n".join(lines), font_size=9, color="black")

    def select(e: Optional[int]) -> None:
        if e is None:
            return
        prev = state["selected"]
        state["selected"] = None if e == prev else int(e)
        if prev is not None:
            recolor(prev)
        recolor(e)
        update_text()

    if legend:
        try:
            pl.add_legend(legend, bcolor="white")
        except Exception as exc:  # legend is a nicety; never block picking
            print(f"[ROOT-PICK][WARN] could not draw colour legend: {exc}")

    # Headless: optionally pre-select an edge, screenshot, and return it.
    if off_screen:
        if preselect is not None and int(preselect) in vis:
            state["selected"] = int(preselect)
            recolor(int(preselect))
        update_text()
        pl.show(screenshot=screenshot, auto_close=True)
        pl.close()
        return state["selected"]

    # ── interactive: distinguish a click from a drag-rotate, plus hover. ──
    from pyvista import _vtk

    cell_picker = _vtk.vtkCellPicker()
    cell_picker.SetTolerance(0.008)
    press = {"xy": None}
    button = {"down": False}
    hover_state = {"edge": None, "xy": None}
    CLICK_TOL2 = 49  # (<=7 px)^2 counts as a click, not a rotate

    def resolve_edge(picker) -> Optional[int]:
        ds = picker.GetDataSet()
        if ds is not None:
            try:
                fd = pv.wrap(ds).field_data
                if "edge_idx" in fd:
                    return int(fd["edge_idx"][0])
            except Exception:
                pass
        if pick_kd is not None:
            pos = np.asarray(picker.GetPickPosition(), dtype=float)
            return int(pick_tag[pick_kd.query(pos)[1]])
        return None

    def edge_at(xy) -> Optional[int]:
        try:
            cell_picker.Pick(float(xy[0]), float(xy[1]), 0, pl.renderer)
        except Exception:
            return None
        return resolve_edge(cell_picker)

    def on_press(*_a) -> None:
        button["down"] = True
        try:
            press["xy"] = pl.iren.get_event_position()
        except Exception:
            press["xy"] = None

    def on_release(*_a) -> None:
        button["down"] = False
        p = press["xy"]
        press["xy"] = None
        try:
            rel = pl.iren.get_event_position()
        except Exception:
            return
        if p is None:
            return
        dx, dy = rel[0] - p[0], rel[1] - p[1]
        if dx * dx + dy * dy <= CLICK_TOL2:
            select(edge_at(rel))

    def on_move(*_a) -> None:
        if button["down"]:
            return
        try:
            xy = pl.iren.get_event_position()
        except Exception:
            return
        lx = hover_state["xy"]
        if lx is not None and (xy[0] - lx[0]) ** 2 + (xy[1] - lx[1]) ** 2 < 16:
            return
        hover_state["xy"] = xy
        e = edge_at(xy)
        if e == hover_state["edge"]:
            return
        hover_state["edge"] = e
        try:
            if e is None:
                pl.remove_actor("__hover__", reset_camera=False)
            else:
                cc = edge_points(e)
                if len(cc) >= 2:
                    conn = np.concatenate([[len(cc)], np.arange(len(cc), dtype=np.int64)])
                    pl.add_mesh(pv.PolyData(cc, lines=conn), color="deepskyblue",
                                line_width=6.0, pickable=False, name="__hover__",
                                reset_camera=False)
            pl.render()
        except Exception:
            pass

    pl.iren.add_observer("LeftButtonPressEvent", on_press)
    pl.iren.add_observer("LeftButtonReleaseEvent", on_release)
    if hover:
        try:
            pl.iren.track_mouse_position(on_move)
        except Exception:
            pass

    pl.add_key_event("c", lambda: select(state["selected"]))  # clear = re-toggle
    pl.add_key_event("x", pl.close)
    update_text()
    print(f"[ROOT-PICK] tree {tree_i + 1}/{n_trees}: click the inlet (root) segment, "
          "'q' next tree, 'x' stop.")
    pl.show()
    return state["selected"]
