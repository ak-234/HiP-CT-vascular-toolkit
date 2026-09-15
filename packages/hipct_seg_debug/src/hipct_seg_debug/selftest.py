"""End-to-end consistency tests that go all the way back to the greyscale image.

The structural checks in `frame.validate` prove the grids line up. These prove the
*interpretation* is right: that ``thickness`` really is a radius in micrometres, that
raw slice indexing is off by nothing, and that the cached RLE index agrees with a
plain sequential decode.
"""

from __future__ import annotations

import numpy as np


def _radial_profile(img, cy, cx, r_px, n_bins=12, out_to=3.0):
    """Mean intensity in concentric annuli out to ``out_to`` x the radius."""
    half = int(np.ceil(r_px * out_to)) + 2
    r0, r1 = max(0, int(cy) - half), min(img.shape[0], int(cy) + half)
    c0, c1 = max(0, int(cx) - half), min(img.shape[1], int(cx) + half)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None, None
    patch = img[r0:r1, c0:c1].astype(np.float64)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(r_px, 1e-9)
    edges = np.linspace(0.0, out_to, n_bins + 1)
    prof = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = (d >= edges[k]) & (d < edges[k + 1])
        if m.any():
            prof[k] = patch[m].mean()
    return 0.5 * (edges[:-1] + edges[1:]), prof


def test_wall_at_radius(session, n_points=20, verbose=True):
    """At fat centreline points the greyscale must show a lumen ringed by a bright wall.

    Lumen interiors are darker than the surrounding myocardium and the wall appears as
    a bright ring just outside ``r`` -- slightly outside, because the wall has
    thickness and ``thickness`` records the lumen radius. If the radius were really a
    diameter, or the slice index were off by more than a slice, or the axes were
    transposed, this structure would vanish; so the same profile is also measured at a
    decoy centre well off the vessel, and the real one must be far more prominent.
    """
    graph, fr, stack = session.graph, session.frame, session.stack
    order = np.argsort(-graph.thickness)
    picked, seen = [], set()
    for i in order:
        z = int(round(graph.points[i, 2] / fr.raw_voxel[2]))
        if z in seen or not (0 <= z < stack.n_slices):
            continue
        seen.add(z)
        picked.append(i)
        if len(picked) >= n_points:
            break

    peaks, contrasts, prom_real, prom_decoy = [], [], [], []
    rng = np.random.default_rng(0)
    for i in picked:
        x, y, z = graph.points[i]
        r_px = graph.thickness[i] / fr.raw_voxel[0]
        zi = int(round(z / fr.raw_voxel[2]))
        cy, cx = y / fr.raw_voxel[1], x / fr.raw_voxel[0]
        img = stack.read_slice(zi)
        centres, prof = _radial_profile(img, cy, cx, r_px)
        if prof is None or np.isnan(prof).any():
            continue
        inside = prof[centres < 0.8].mean()
        peaks.append(centres[int(np.nanargmax(prof))])
        contrasts.append(inside - prof[(centres > 1.5) & (centres < 2.5)].mean())
        prom_real.append(prof.max() - inside)

        # Same measurement 6 radii away: nothing should ring there.
        ang = rng.uniform(0, 2 * np.pi)
        dy, dx = 6 * r_px * np.sin(ang), 6 * r_px * np.cos(ang)
        _, dprof = _radial_profile(img, cy + dy, cx + dx, r_px)
        if dprof is not None and not np.isnan(dprof).any():
            prom_decoy.append(dprof.max() - dprof[centres < 0.8].mean())

    if not peaks:
        print("  [FAIL] radial profile: no usable points")
        return False
    peaks = np.array(peaks)
    contrasts = np.array(contrasts)
    peak_med = float(np.median(peaks))
    dark_frac = float(np.mean(contrasts < 0))
    real = float(np.median(prom_real))
    decoy = float(np.median(prom_decoy)) if prom_decoy else 0.0

    ok_peak = 0.8 <= peak_med <= 1.6
    ok_dark = dark_frac >= 0.8
    ok_prom = real > 3.0 * max(decoy, 1e-9)
    if verbose:
        print(f"  [{'PASS' if ok_peak else 'FAIL'}] wall ring just outside r  "
              f"median peak at r/R = {peak_med:.2f} over {len(peaks)} points "
              f"(expect 1.0-1.5)")
        print(f"  [{'PASS' if ok_dark else 'FAIL'}] lumen darker than tissue  "
              f"{100 * dark_frac:.0f}% of points, mean contrast {contrasts.mean():+.0f} counts")
        print(f"  [{'PASS' if ok_prom else 'FAIL'}] ring is where the graph says "
              f"prominence {real:.0f} counts at the centreline vs {decoy:.0f} "
              f"at a decoy 6r away ({real / max(decoy, 1e-9):.1f}x)")
    return ok_peak and ok_dark and ok_prom


def test_rle_index(session, n_slices=64, verbose=True):
    """The cached per-slice index must reproduce a plain decode from the stream start."""
    lat = session.labels
    ref = lat.decode_sequential(n_slices)
    bad = [z for z in range(n_slices) if not np.array_equal(ref[z], lat.slice_z(z))]
    ok = not bad
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] RLE index == sequential   "
              f"{n_slices} slices byte-identical" if ok else
              f"  [FAIL] RLE index == sequential   mismatched slices {bad[:5]}")
    return ok


def test_roundtrip(session, verbose=True):
    """A centreline point must survive um -> raw -> um and um -> segmentation intact."""
    graph, fr = session.graph, session.frame
    i = int(np.argmax(graph.thickness))
    xyz = graph.points[i]
    zyx = fr.um_to_raw_index(xyz)[0]
    back = fr.raw_to_um([zyx])[0]
    err = float(np.linalg.norm(back - xyz))
    ijk = fr.um_to_seg_index(xyz)[0]
    inside = int(session.labels.slice_z(int(ijk[2]))[ijk[1], ijk[0]])

    # The mask blob through that point should be about 2r wide.
    sl = session.labels.slice_z(int(ijk[2]))
    row = sl[ijk[1]]
    lo = ijk[0]
    while lo > 0 and row[lo - 1]:
        lo -= 1
    hi = ijk[0]
    while hi < len(row) - 1 and row[hi + 1]:
        hi += 1
    width_um = (hi - lo + 1) * fr.seg_spacing[0]
    expect = 2 * graph.thickness[i]
    ratio = width_um / expect

    ok_r = err <= float(fr.raw_voxel.max())
    ok_in = inside == 1
    ok_w = 0.5 <= ratio <= 2.0
    if verbose:
        print(f"  [{'PASS' if ok_r else 'FAIL'}] um->raw->um round trip    "
              f"{err:.2f} um error (<= 1 voxel = {fr.raw_voxel.max():.2f})")
        print(f"  [{'PASS' if ok_in else 'FAIL'}] fattest point in mask     "
              f"raw slice {zyx[0]} row {zyx[1]} col {zyx[2]} -> "
              f"seg ({ijk[0]}, {ijk[1]}, {ijk[2]}) = {inside}")
        print(f"  [{'PASS' if ok_w else 'FAIL'}] mask width matches 2r     "
              f"{width_um:.0f} um across vs 2r = {expect:.0f} um (ratio {ratio:.2f})")
    return ok_r and ok_in and ok_w


def test_slab(session, verbose=True):
    """Building a slab must produce aligned, non-empty layers."""
    graph = session.graph
    i = int(np.argmax(graph.thickness))
    slab = session.builder().build(graph.points[i], half_slices=2, roi_px=200, with_tube=True)
    seg_px = int(slab.seg.sum()) if slab.seg is not None else 0
    tube_px = int(slab.tube.sum()) if slab.tube is not None else 0
    mid = slab.raw.shape[0] // 2
    iou = 0.0
    if slab.seg is not None and slab.tube is not None:
        a, b = slab.seg[mid] > 0, slab.tube[mid] > 0
        iou = float((a & b).sum() / max((a | b).sum(), 1))
    ok = slab.raw.size > 0 and seg_px > 0 and tube_px > 0 and iou > 0.3
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] slab layers agree         "
              f"raw {slab.raw.shape}, seg {seg_px} px, tube {tube_px} px, "
              f"mid-slice IoU {iou:.2f}")
    return ok


def test_full_slice_slab(session, verbose=True):
    """``roi_px=0`` must cover the whole slice and stay aligned with the crop.

    The crop and the full slice are two views of the same data, so wherever they
    overlap they must agree exactly -- if they do not, `translate` is wrong and
    every overlay is displaced.
    """
    graph = session.graph
    i = int(np.argmax(graph.thickness))
    builder = session.builder()
    full = builder.build(graph.points[i], half_slices=2, roi_px=0, with_tube=False)
    crop = builder.build(graph.points[i], half_slices=2, roi_px=200, with_tube=False)
    _, n_rows, n_cols = session.stack.shape

    ok_extent = (
        full.row0 == 0 and full.col0 == 0
        and full.row1 == n_rows and full.col1 == n_cols
        and full.raw.shape[1:] == (n_rows, n_cols)
        and full.translate == (full.z_lo, 0, 0)
        and full.full_slice and not crop.full_slice
    )

    # The crop's window, read out of the full slice, must be the crop.
    sub = full.raw[:, crop.row0:crop.row1, crop.col0:crop.col1]
    ok_raw = sub.shape == crop.raw.shape and np.array_equal(sub, crop.raw)
    ok_seg = True
    if full.seg is not None and crop.seg is not None:
        sseg = full.seg[:, crop.row0:crop.row1, crop.col0:crop.col1]
        ok_seg = sseg.shape == crop.seg.shape and np.array_equal(sseg, crop.seg)

    ok = ok_extent and ok_raw and ok_seg
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] full-slice slab           "
              f"{full.raw.shape} covers the slice, translate {full.translate}; "
              f"the 200 px crop read out of it is "
              f"{'identical' if ok_raw and ok_seg else 'DIFFERENT'}")
    return ok


def test_lazy_seg_alignment(session, verbose=True):
    """The lazy whole-volume mask must land exactly where the slab's gather puts it.

    The slab upsamples the lattice onto the raw grid by integer division
    (``frame.raw_axis_to_seg_axis``); the volume layer instead stays on the
    lattice grid and is placed by napari's ``scale``/``translate``. Those are two
    different routes to the same picture, and the half-voxel term in
    ``volume.seg_placement`` is what makes them agree. Get it wrong and the mask
    shifts by half a raw voxel -- entirely plausible-looking, and wrong.
    """
    from .volume import seg_placement

    fr = session.frame
    if session.labels is None:
        if verbose:
            print("  [PASS] lazy mask alignment      no segmentation loaded, skipped")
        return True

    scale, translate = seg_placement(fr)
    nx, ny, nz = (int(v) for v in fr.seg_dims)

    # Walk raw indices across the lattice and compare the two mappings.
    rng = np.random.default_rng(0)
    cols = rng.integers(int(fr.raw_start[0]), int(fr.raw_start[0]) + nx * int(fr.bin_factor[0]), 400)
    rows = rng.integers(int(fr.raw_start[1]), int(fr.raw_start[1]) + ny * int(fr.bin_factor[1]), 400)
    slices = rng.integers(int(fr.raw_start[2]), int(fr.raw_start[2]) + nz * int(fr.bin_factor[2]), 400)

    # Route A: the gather the slab does.
    gather = np.column_stack([
        fr.raw_axis_to_seg_axis(slices, 2),
        fr.raw_axis_to_seg_axis(rows, 1),
        fr.raw_axis_to_seg_axis(cols, 0),
    ])
    # Route B: invert napari's world = index * scale + translate, at pixel centres.
    world = np.column_stack([slices, rows, cols]).astype(np.float64)
    placed = np.round((world - np.asarray(translate)) / np.asarray(scale)).astype(np.int64)

    inside = np.all((gather >= 0) & (gather < [nz, ny, nx]), axis=1)
    n_bad = int((gather[inside] != placed[inside]).any(axis=1).sum())
    ok = n_bad == 0

    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] lazy mask alignment       "
              f"scale {tuple(int(v) for v in scale)}, translate "
              f"{tuple(round(float(v), 1) for v in translate)}; "
              f"{int(inside.sum())} probes, {n_bad} disagree with the slab's gather")
    return ok


def test_segmentation_volume(session, verbose=True):
    """The whole-tree mask must be built where the lattice actually is."""
    from .viewer3d import segmentation_volume

    fr = session.frame
    if session.labels is None:
        if verbose:
            print("  [PASS] whole-tree mask          no segmentation loaded, skipped")
        return True

    stride = 16  # coarse: this test is about placement, not fidelity
    grid, surf = segmentation_volume(fr, session.labels, stride=stride)
    if surf is None or surf.n_cells == 0:
        if verbose:
            print("  [FAIL] whole-tree mask          produced no geometry")
        return False

    bb = fr.seg_bbox_um
    pad = np.asarray(fr.seg_spacing) * stride
    b = np.asarray(surf.bounds, dtype=np.float64)
    lo = np.array([bb[0], bb[2], bb[4]]) - pad
    hi = np.array([bb[1], bb[3], bb[5]]) + pad
    ok_bounds = bool(
        np.all(b[0::2] >= lo - 1e-6) and np.all(b[1::2] <= hi + 1e-6)
    )

    # The fattest centreline point is inside the vessel tree, so it must be
    # inside the mesh's bounds -- a mesh built at the wrong origin would not.
    graph = session.graph
    p = graph.points[int(np.argmax(graph.thickness))]
    ok_point = bool(np.all(p >= b[0::2] - 1e-6) and np.all(p <= b[1::2] + 1e-6))

    ok = ok_bounds and ok_point
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] whole-tree mask           "
              f"stride {stride}: {surf.n_cells:,} triangles spanning "
              f"{b[1]-b[0]:,.0f} x {b[3]-b[2]:,.0f} x {b[5]-b[4]:,.0f} um; "
              f"inside the lattice bbox: {ok_bounds}, contains the fattest point: {ok_point}")
    return ok


def test_perimeter_assumption(session, verbose=True):
    """The design's own assumption must hold in the bulk, and the audit must find it.

    `adjust_thickness.py` assigns `r = perimeter / (2*pi)`, so re-measuring the lumen
    perimeter should reproduce the stored radius on average. If the median ratio drifted
    away from 1 the audit would be measuring something else -- a different plane, the
    wrong connected component, or a units error -- and every `perimeter_mismatch` it
    reported would be an artefact. It is precisely because the bulk agrees that the
    outliers mean anything.

    The hand-inspected site near raw slice 2727 (a ~750 um assigned radius over a
    ~130 um slit) must still surface, now as high collapse *severity* rather than as a
    defect: it is the design working, not failing.
    """
    from . import crosssection

    sites, profile = crosssection.find_sites(
        session.graph, session.frame, session.labels
    )
    ratio = profile.perimeter_ratio(session.graph.thickness)[profile.measured]
    ratio = ratio[np.isfinite(ratio)]
    med = float(np.median(ratio))
    ok_bulk = 0.85 <= med <= 1.10

    iso = profile.isoperimetric[profile.measured]
    ok_iso = float(np.nanmedian(iso)) >= 1.0  # isoperimetric ratio cannot be below 1

    severe = [c for c in sites if c.kind == "collapse_severity"]
    near = [c for c in severe if abs(c.raw_slice - 2727) <= 60]
    ok_known = bool(near)

    kinds = {}
    for c in sites:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1

    if verbose:
        print(f"  [{'PASS' if ok_bulk else 'FAIL'}] perimeter rule reproduced "
              f"median r_stored/r_perimeter = {med:.2f} over {len(ratio)} "
              f"cross-sections (expect ~1.0)")
        print(f"  [{'PASS' if ok_iso else 'FAIL'}] isoperimetric sane        "
              f"median {np.nanmedian(iso):.2f} (must be >= 1.0)")
        print(f"  [{'PASS' if ok_known else 'FAIL'}] known collapse ranked     "
              f"{kinds}; slice ~2727 present as collapse_severity: {ok_known}")
    return ok_bulk and ok_iso and ok_known


def test_murray(session, verbose=True):
    """Murray's law must be sampled away from junctions, or it degenerates to 2.00.

    Every edge at a vertex shares that point, so reading the three radii *at* the vertex
    returns the same number three times and the ratio is identically 2.00. An earlier
    version did exactly that and wrongly concluded the test was unusable here.
    """
    from . import candidates

    graph = session.graph
    edges_at_vertex, _ = candidates._edge_neighbourhood(graph)
    ratios = []
    for v in np.flatnonzero(graph.degree() >= 3):
        rs = np.array([candidates._radius_along(graph, e, v) for e in edges_at_vertex[v]])
        if not np.all(np.isfinite(rs)) or rs.max() <= 0:
            continue
        p = int(np.argmax(rs))
        ratios.append((np.delete(rs, p) ** 3).sum() / rs[p] ** 3)
    ratios = np.array(ratios)
    degenerate = float(np.mean(np.abs(ratios - 2.0) < 0.01))
    med = float(np.median(ratios))
    ok = degenerate < 0.05 and 0.4 <= med <= 1.5
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] Murray sampled off-junction "
              f"median {med:.2f} over {len(ratios)} bifurcations, "
              f"{100 * degenerate:.0f}% at exactly 2.00 (the degenerate value)")
    return ok


def test_candidates_tree_aware(session, verbose=True):
    """No reported pair may be a bifurcation neighbour."""
    from . import candidates

    cs = candidates.find_candidates(session.graph, session.frame)
    pairs = [c for c in cs if c.edge_b >= 0]
    bad = [c for c in pairs if c.hops < 4 or abs(c.strahler_a - c.strahler_b) > 1]
    ok = not bad
    if verbose:
        kinds = {}
        for c in cs:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        print(f"  [{'PASS' if ok else 'FAIL'}] graph candidates tree-aware "
              f"{len(cs)} total {kinds}; {len(bad)} violate hops>=4 / |dStrahler|<=1")
    return ok


def test_centreline_point_ids(session, verbose=True):
    """The 3D picker reads a clicked vertex id straight back into ``graph.points``.

    ``viewer3d`` picks with a ``vtkPointPicker`` restricted to the centreline actor and
    treats the reported point id as an index into the spatial graph. That holds only
    while ``centreline_polydata`` keeps every graph point, in order, and adds none of
    its own. If it ever stopped holding, every pick would silently report a *different*
    vessel -- indistinguishable by eye from the picker simply being inaccurate.
    """
    from .viewer3d import centreline_polydata

    graph = session.graph
    poly = centreline_polydata(graph)
    same_count = poly.n_points == graph.n_point
    # PolyData stores float32; compare at that precision, not float64.
    worst = (
        float(np.abs(np.asarray(poly.points) - graph.points.astype(np.float32)).max())
        if same_count
        else float("inf")
    )
    ok = same_count and worst == 0.0
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] centreline point ids index the graph "
              f"{poly.n_points} polydata points vs {graph.n_point} graph points, "
              f"max coordinate difference {worst:g}")
    return ok


def test_slice_plane_geometry(session, verbose=True):
    """The 3D image plane must sit where the frame says, the right way up.

    The row flip in ``slice_texture_array`` is the part worth guarding: ``pv.Texture``
    puts array row 0 at the high end of the quad, while this frame puts row 0 at y = 0.
    Getting it wrong mirrors the anatomy top-to-bottom, which looks entirely plausible
    on screen and would silently misplace every feature the operator reads off it.
    """
    from .viewer3d import slice_plane_quad, slice_texture_array

    fr, stack = session.frame, session.stack
    nz, n_rows, n_cols = stack.shape
    z = min(nz // 2, nz - 1)

    # A ramp that is darkest in row 0: after the flip, row 0 of the texture is the
    # brightest, because it is the one that will be drawn at high y.
    ramp = np.repeat(np.arange(8, dtype=np.uint16)[:, None] * 1000, 4, axis=1)
    tex = slice_texture_array(ramp)
    flipped = tex[0, :, 0].mean() > tex[-1, :, 0].mean()

    quad = np.asarray(slice_plane_quad(fr, stack.shape, z).points)
    want_lo = fr.raw_to_um([[z, 0, 0]])[0]
    want_hi = fr.raw_to_um([[z, n_rows - 1, n_cols - 1]])[0]
    err = max(
        float(np.abs(quad[0] - want_lo).max()),
        float(np.abs(quad[2] - want_hi).max()),
    )
    ok = flipped and err < 1e-6
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] image plane placed and upright "
              f"corners match raw_to_um to {err:.1e} um, texture row-flipped: {flipped}")
    return ok


def test_segmentation_box(session, verbose=True):
    """The 3D mask isosurface must be built around the point it was asked for.

    Placing the box tests origin, spacing and the ravel order all at once: the lattice is
    indexed (z, y, x) but ``pv.ImageData`` wants x fastest, so a transposed volume would
    put the mask somewhere else entirely -- and on a thin vessel, "somewhere else" reads
    as empty.
    """
    from .viewer3d import segmentation_box

    graph, fr = session.graph, session.frame
    i = int(np.argmax(graph.thickness))
    pt = graph.points[i]

    # The box has to reach past the wall or the whole thing is mask and marching cubes
    # has no surface to find -- the case `Picker3D._seg_half_um` scales the box to avoid.
    half = max(3.0 * float(graph.thickness[i]), 800.0)
    grid, surf = segmentation_box(fr, session.labels, pt, half_um=half)
    if grid is None:
        if verbose:
            print("  [FAIL] segmentation box placed  box was empty")
        return False

    idx = grid.find_closest_point(pt)
    hit = int(grid.point_data["mask"][idx])
    d = float(np.linalg.norm(np.asarray(grid.points[idx]) - pt))
    ok = hit > 0 and d <= float(fr.seg_spacing.max()) and surf is not None and surf.n_cells > 0
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] segmentation box placed "
              f"fattest centreline point (r = {graph.thickness[i]:,.0f} um) lands on a "
              f"mask voxel ({hit}) {d:.1f} um away; +/-{half:,.0f} um box gives "
              f"{0 if surf is None else surf.n_cells} triangles")
    return ok


def test_slice_shapes_3d(session, verbose=True):
    """The 3D shape overlays must land exactly where the slice browser draws them.

    ``Picker3D.set_slice_shapes`` re-expresses the slab's contours and ellipses in world
    micrometres. Both windows then claim to be showing the same geometry, so the
    conversion has to round-trip: back through ``um_to_raw`` it must reproduce the raw
    (slice, row, col) the browser used, and each ring must sit on its own slice.
    """
    from .viewer3d import SHAPE_KEYS, Picker3D

    graph, fr = session.graph, session.frame
    # The surface is a partial model -- it covers ~42% of this skeleton, and the fattest
    # point of all is 35 mm away from it -- so pick the fattest point the surface really
    # reaches, or the STL contour would be legitimately empty and untested.
    i = int(np.argmax(graph.thickness))
    if session.mesh is not None and session.mesh.n_points:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(np.asarray(session.mesh.points)).query(graph.points)
        covered = np.flatnonzero(d < graph.thickness * 1.5)
        if covered.size:
            i = int(covered[np.argmax(graph.thickness[covered])])
    slab = session.builder().build(graph.points[i], half_slices=2, roi_px=200)

    picker = Picker3D(graph, session.mesh, [], fr, stack=session.stack,
                      labels=session.labels)
    picker.set_slice_shapes(slab)

    counts = {k: sum(len(v) for v in picker._shapes[k].values()) for k in SHAPE_KEYS}
    worst_z, worst_rt = 0.0, 0.0
    for key in SHAPE_KEYS:
        for z, polys in picker._shapes[key].items():
            if not slab.z_lo <= z < slab.z_hi:
                worst_z = float("inf")
            for pl in polys:
                back = fr.um_to_raw(pl)
                worst_z = max(worst_z, float(np.abs(back[:, 0] - z).max()))
                # x = col * vx and y = row * vy, so the round trip must be exact.
                worst_rt = max(worst_rt, float(np.abs(fr.raw_to_um(back) - pl).max()))

    # Every kind the browser drew here must survive the trip; none may be silently lost.
    expected = {
        "stl_contour": len(slab.contours or []),
        "assumed": len(slab.circles or []),
        "perimeter": len(slab.perim_circles or []),
    }
    ok = counts == expected and sum(counts.values()) > 0 and worst_z < 0.5 and worst_rt < 1e-6
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] 3D slice shapes round-trip "
              f"{counts} vs slab {expected}, each on its own slice to {worst_z:.2e}, "
              f"um->raw->um error {worst_rt:.1e} um")
    return ok


def test_layer_registry(session, verbose=True):
    """Every layer the panel offers must actually resolve to something in ``Picker3D``.

    A key present in ``LAYERS`` but unknown to the accessors gives a panel row that
    silently controls nothing -- there is no error, the slider just does not move
    anything, which is the sort of thing that is only noticed months later. ``Picker3D``
    needs no plotter to be constructed, so this costs nothing.

    Checked twice: once with the dataset loaded, and once with none. The empty state
    is reachable now that the control panel can unload and swap datasets, and a row
    that raised there would take the window down at the moment it was supposed to be
    telling you why a load failed.
    """
    from .viewer3d import COLOR_MODES, LAYERS, Picker3D

    loaded = Picker3D(session.graph, session.mesh, [], session.frame,
                      stack=session.stack, labels=session.labels)
    empty = Picker3D(graph=None)

    bad = []
    for picker, phase in ((loaded, "loaded"), (empty, "empty")):
        for key, _label, default, _visible in LAYERS:
            try:
                picker.set_layer_opacity(key, 0.25)
                ok = (
                    isinstance(picker.layer_actors(key), list)
                    and abs(picker.layer_opacity(key) - 0.25) < 1e-9
                    and isinstance(picker.layer_visible(key), bool)
                    and isinstance(picker.layer_available(key), bool)
                )
            except Exception as exc:  # noqa: BLE001 - report the key, not a traceback
                bad.append(f"{key}/{phase} ({type(exc).__name__})")
                continue
            if not ok:
                bad.append(f"{key}/{phase}")
            picker.set_layer_opacity(key, default)

        # The colour combo is not a layer row, but it sits in the same panel and is
        # refreshed in the same empty state, where there is no graph to offer modes for.
        try:
            for key, _label in picker.color_modes():
                picker.set_color_by(key)
            if picker.color_by() not in {k for k, _l, _t in COLOR_MODES}:
                bad.append(f"colour mode/{phase}")
        except Exception as exc:  # noqa: BLE001 - report the phase, not a traceback
            bad.append(f"colour mode/{phase} ({type(exc).__name__})")

        # And the legend checkbox, for the same reason: not a layer row, driven from
        # the same panel, and toggled here with no plotter behind it -- which is the
        # state a panel refresh after a failed load would find it in.
        try:
            picker.set_legend_visible(False)
            picker.set_legend_visible(True)
            if not (isinstance(picker.legend_available(), bool)
                    and picker.legend_visible() is True):
                bad.append(f"legend/{phase}")
        except Exception as exc:  # noqa: BLE001 - report the phase, not a traceback
            bad.append(f"legend/{phase} ({type(exc).__name__})")

    ok = not bad
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] layer panel keys resolve "
              f"{len(LAYERS)} layers and every colour mode, loaded and empty: "
              f"{', '.join(k for k, _, _, _ in LAYERS)}"
              + (f"; broken: {bad}" if bad else ""))
    return ok


def run_selftest(session) -> bool:
    print("End-to-end self-test")
    results = [
        test_rle_index(session),
        test_roundtrip(session),
        test_wall_at_radius(session),
        test_slab(session),
        test_full_slice_slab(session),
        test_lazy_seg_alignment(session),
        test_segmentation_volume(session),
        test_centreline_point_ids(session),
        test_slice_plane_geometry(session),
        test_segmentation_box(session),
        test_slice_shapes_3d(session),
        test_layer_registry(session),
        test_candidates_tree_aware(session),
        test_murray(session),
        test_perimeter_assumption(session),
    ]
    ok = all(results)
    print(f"\n  {'all self-tests passed' if ok else 'SELF-TEST FAILURES - see above'}\n")
    return ok
