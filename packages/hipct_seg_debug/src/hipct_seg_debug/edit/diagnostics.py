"""Read a written graph back and ask why each radius is what it is.

The tree-wide counters `measure_radii` prints say how many points were refused and
for what reason. They cannot say whether *this* stretch of *this* branch is narrow
because the vessel is narrow or because the pass could not measure it, and that is
the question that comes up while looking at a vessel in the viewer. Each function
here answers one form of it and prints a table.

Nothing here re-runs the measurement. Provenance is read back out of the ``.am``
(``radius_source``, ``radius_reject_reason``, ``radius_resolution_mode``), so what is
reported is the pass's own verdict rather than a reconstruction of it; sections are
re-cut only to describe geometry the verdict does not record.

**Two traps, both of which produced wrong answers before being found.**

* Cut with a bounded ``grow_to``. Unbounded, a plane that is not truly perpendicular
  doubles its window to ``max_half`` and encloses a streak *along* the vessel --
  thousands of voxels, a major axis of 100+, and a "section" that is nothing of the
  kind.
* Ask :func:`junction_terms` about the graph the pass **consumed**, not the one it
  wrote. ``_BranchContext.rivals`` scales its search by the radius handed to it, so
  running it on the output asks a different question than the pass asked -- and if
  the output radius is wrong, which is the case under investigation, the answer is
  wrong with it.
"""

from __future__ import annotations

import numpy as np

from . import radius_perimeter as rp


def _axes(blob: np.ndarray) -> tuple[float, float]:
    """Major and minor axis lengths in voxels, from the blob's second moments.

    For a uniformly filled ellipse of semi-axes a and b the covariance eigenvalues
    are ``a^2/4`` and ``b^2/4``, so the full axis lengths are ``4*sqrt(lambda)``.
    """
    ij = np.argwhere(blob).astype(float)
    if len(ij) < 2:
        return float(len(ij)), float(len(ij))
    cov = np.cov((ij - ij.mean(axis=0)).T)
    ev = np.sort(np.linalg.eigvalsh(np.atleast_2d(cov)))
    return float(4.0 * np.sqrt(max(ev[-1], 0.0))), float(4.0 * np.sqrt(max(ev[0], 0.0)))


def _emit(out, line=""):
    print(line, file=out) if out is not None else print(line)


def segment_report(graph, frame, labels, sids, *, triple=None, out=None) -> None:
    """Per-point radius, provenance and re-cut section geometry for named segments."""
    from ..crosssection import _PlaneSampler, _perimeter_um, cut, robust_edge_tangents

    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    attrs = getattr(triple, "point_attrs", {}) if triple is not None else {}
    src, rej = attrs.get("radius_source", {}), attrs.get("radius_reject_reason", {})
    _emit(out, f"spacing {sp:.2f} um; MIN_BLOB_VOXELS gates on the `vox` column")
    for sid in sids:
        seg = graph.segment(sid)
        pids = list(seg["point_ids"])
        coords, radii = graph.coords(sid), graph.radii(sid)
        n = len(coords)
        ijk = frame.um_to_seg(coords)
        tangents = robust_edge_tangents(coords, radii, spacing_um=sp)
        arc = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()) if n > 1 else 0.0
        n_meas = sum(1 for p in pids if int(src.get(p, rp.FILLED)) != rp.FILLED)
        _emit(out)
        _emit(out, f"--- segment {sid}: {n} points, {n_meas} measured "
                   f"({n_meas / max(n, 1):.0%}), radius {radii.min():.0f}-"
                   f"{radii.max():.0f} um (median {np.median(radii):.0f})")
        _emit(out, f"    length {arc:.0f} um; end node degrees "
                   f"{graph.degree(seg['node1'])} and {graph.degree(seg['node2'])}; "
                   f"`open` = still touching the window edge at four radii")
        _emit(out, f"    {'i':>3}{'r um':>7}{'source':>26}{'reject':>26}{'vox':>6}"
                   f"{'thick':>7}{'major':>7}{'pinch':>7}{'r_per':>7}{'r_area':>7}"
                   f"{'open':>6}")
        for i, pid in enumerate(pids):
            r_vox = max(float(radii[i]) / sp, 1.0)
            c = cut(sampler, ijk[i], tangents[i], min(int(r_vox * 2.5) + 2, 64),
                    max_half=64, grow_to=int(4.0 * r_vox) + 2, min_blob_voxels=1)
            if c is None:
                geom = f"{'--':>6}{'--':>7}{'--':>7}{'--':>7}{'--':>7}{'--':>7}{'--':>6}"
            else:
                area = float(c.blob4.sum())
                major, minor = _axes(c.blob4)
                geom = (f"{area:>6.0f}{minor:>7.2f}{major:>7.2f}"
                        f"{float(c.blob8.sum()) / max(area, 1e-9):>7.2f}"
                        f"{_perimeter_um(c.blob4, sp) / (2 * np.pi):>7.0f}"
                        f"{np.sqrt(area / np.pi) * sp:>7.0f}"
                        f"{('YES' if c.touches_border else '-'):>6}")
            s = int(src.get(pid, rp.FILLED))
            r = int(rej.get(pid, rp.UNMEASURABLE))
            _emit(out, f"    {i:>3}{radii[i]:>7.0f}"
                       f"{rp.SOURCE_NAMES.get(s, s):>26}"
                       f"{rp.REJECT_NAMES.get(r, r):>26}{geom}")


def junction_terms(graph, frame, labels, sids, *, out=None) -> None:
    """Which term of the junction mask refuses each point: `stable`, or adjacency.

    `_adaptive_junction_mask` walks in from each degree-3 node until it meets two
    consecutive *exclusive* sections, where exclusive is
    ``stable and not adjacent_overlap``, and commits the whole segment if it never
    finds two in a row. This prints both terms so the answer is read rather than
    inferred. Give it the graph the pass **consumed**.
    """
    from ..crosssection import _PlaneSampler, robust_edge_tangents, stable_transverse_cut

    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    ctx = rp._BranchContext.build(graph)
    for sid in sids:
        seg = graph.segment(sid)
        coords, scale = graph.coords(sid), graph.radii(sid)
        n = len(coords)
        ijk = frame.um_to_seg(coords)
        tangents = robust_edge_tangents(coords, scale, spacing_um=sp)
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(coords, axis=0), axis=1))])
        _emit(out)
        _emit(out, f"--- segment {sid}: {n} points, input radii "
                   f"{np.nanmin(scale):.0f}-{np.nanmax(scale):.0f} um; end node "
                   f"degrees {graph.degree(seg['node1'])} and "
                   f"{graph.degree(seg['node2'])}")
        _emit(out, f"    {'i':>3}{'stable':>8}{'adj_ovl':>9}{'exclusive':>11}"
                   f"{'rivals':>8}   adjacent rival segments")
        stable = np.zeros(n, dtype=bool)
        adj = np.zeros(n, dtype=bool)
        for i in range(n):
            r_vox = max(float(scale[i]) / sp, 1.0)
            chosen = stable_transverse_cut(
                sampler, ijk[i], tangents[i], r_vox, spacing_um=sp, max_half=64,
                search_degrees=20.0, min_blob_voxels=12,
                slab_offsets=(0.0, 0.25, 0.5) if i == 0 else (
                    (-0.5, -0.25, 0.0) if i == n - 1 else (-0.5, 0.0, 0.5)))
            stable[i] = chosen is not None and not chosen.cut.touches_border
            tangent = chosen.tangent if chosen is not None else tangents[i]
            rivals = ctx.rivals(sid, coords[i], tangent, float(scale[i]))
            adj[i] = any(item[4] for item in rivals)
            names = sorted({int(item[0]) for item in rivals if item[4]})
            _emit(out, f"    {i:>3}{str(bool(stable[i])):>8}{str(bool(adj[i])):>9}"
                       f"{str(bool(stable[i] and not adj[i])):>11}{len(rivals):>8}"
                       f"   {names if names else '-'}")
        exclusive = stable & ~adj
        best = run = 0
        for e in exclusive:
            run = run + 1 if e else 0
            best = max(best, run)
        mask, lengths = rp._adaptive_junction_mask(graph, sid, arc, stable, adj)
        _emit(out, f"\n    stable {stable.sum()}/{n}   adjacent_overlap "
                   f"{adj.sum()}/{n}   exclusive {exclusive.sum()}/{n}")
        _emit(out, f"    longest run of consecutive exclusive sections: {best} "
                   f"(the mask needs 2 to stop walking)")
        _emit(out, f"    _adaptive_junction_mask masks {int(mask.sum())}/{n} points"
                   f"  (runs {[f'{v:.0f} um' for v in lengths]})")


def reformat_radius(graph, frame, labels, sids, *, size_px=81, out=None) -> None:
    """Measure each segment again on `reformat`'s planes -- a different construction.

    `crosssection.cut` shares its plane logic with the pass being audited: both take
    a per-point tangent and search a cone around it, so agreement between them is
    weaker evidence than it looks. `reformat` resamples to uniform arclength, smooths
    until the plane stack is provably collision-free, then carries a
    parallel-transport frame, so a plane's normal depends on the whole path rather
    than its immediate neighbours.

    ``mode="native"`` pins one output pixel to one segmentation voxel, so the
    estimator sees the same staircase as `cut` does. Any other mode resamples the
    mask onto a different grid and changes the estimator's bias with it.
    """
    from scipy import ndimage

    from .. import reformat as rf
    from ..crosssection import _perimeter_um

    sampler = rf.LabelSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    half_value = (size_px // 2) * sp
    for sid in sids:
        coords, radii = graph.coords(sid), graph.radii(sid)
        if len(coords) < 2:
            continue
        centreline = rf.build_centreline(
            coords, radii, np.full(len(coords), sid, dtype=np.int64), step_um=sp,
            half_of=lambda r, _h=half_value: np.full(len(r), _h))
        geom = rf.plane_geometry(centreline, mode="native", size_px=size_px,
                                 voxel_um=sp, native_scale=1.0)
        planes = sampler.sample_planes(centreline, geom)
        r_per, r_area, opened, lost = [], [], 0, 0
        for k in range(len(planes)):
            pl = planes[k] > 0
            h = pl.shape[0] // 2
            if not pl[h, h]:
                lost += 1
                continue
            lab4, _ = ndimage.label(pl)
            blob = lab4 == lab4[h, h]
            lab8, _ = ndimage.label(pl, structure=np.ones((3, 3), dtype=int))
            b8 = lab8 == lab8[h, h]
            if b8[0].any() or b8[-1].any() or b8[:, 0].any() or b8[:, -1].any():
                opened += 1
                continue
            pitch = float(geom.px_um[k])
            area = float(blob.sum())
            r_per.append(_perimeter_um(blob, pitch) / (2.0 * np.pi))
            r_area.append(np.sqrt(area / np.pi) * pitch)
        _emit(out)
        _emit(out, f"--- segment {sid}")
        _emit(out, f"    {centreline.describe()}")
        _emit(out, f"    smoothing moved the centreline: median "
                   f"{centreline.median_move_um:.1f} um, max "
                   f"{centreline.max_move_um:.1f} um")
        _emit(out, f"    planes {len(planes)}: {len(r_per)} usable, {opened} "
                   f"truncated by the frame, {lost} with the centre off the mask")
        if not r_per:
            _emit(out, "    nothing measurable")
            continue
        arr = np.asarray(r_per)
        cor = np.array([rp.correct_perimeter_radius(v, sp) for v in arr])
        def q(v):
            return (f"{np.percentile(v, 5):.0f} / {np.median(v):.0f} / "
                    f"{np.percentile(v, 95):.0f}")
        _emit(out, "    p5 / median / p95, um")
        _emit(out, f"      r_perimeter raw        {q(arr)}")
        _emit(out, f"      r_perimeter corrected  {q(cor)}")
        _emit(out, f"      r_area                 {q(np.asarray(r_area))}")
        _emit(out, f"      graph radius (median)  {np.median(radii):.0f}")


#: Bins of distance-to-node, in units of the segment's own local radius.
_FLARE_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0))


def ostium_flare(graph, frame, labels, *, out=None) -> None:
    """How much wider is the section approaching a branched node, and is it round?

    **Each segment is normalised by its own interior**, and it has to be. Binning raw
    distance-to-node across the tree compares near-node *proximal* sections against
    far-from-node *distal* ones and reports a flare twice the real size with the
    aspect ratio moving the wrong way, because the far bin is a different vessel
    population entirely.

    A flare a radius can carry grows both axes and leaves the aspect ratio flat. A
    rising aspect ratio is the section elongating toward its neighbour, which an
    isotropic radius renders as a bulge in every direction rather than an opening in
    one.
    """
    from ..crosssection import _PlaneSampler, _perimeter_um, cut, robust_edge_tangents

    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    acc = {b: {"r": [], "a": []} for b in _FLARE_BINS}
    n_seg = 0
    for seg in graph.segments:
        sid = seg["id"]
        coords, radii = graph.coords(sid), graph.radii(sid)
        n = len(coords)
        if n < 8:
            continue
        if graph.degree(seg["node1"]) < 3 and graph.degree(seg["node2"]) < 3:
            continue
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(coords, axis=0), axis=1))])
        d1 = arc if graph.degree(seg["node1"]) >= 3 else np.full(n, np.inf)
        d2 = (arc[-1] - arc) if graph.degree(seg["node2"]) >= 3 else np.full(n, np.inf)
        dnode = np.minimum(d1, d2)
        tangents = robust_edge_tangents(coords, radii, spacing_um=sp)
        ijk = frame.um_to_seg(coords)
        r_per = np.full(n, np.nan)
        aspect = np.full(n, np.nan)
        d_over_r = np.full(n, np.nan)
        for i in range(n):
            r_vox = max(float(radii[i]) / sp, 1.0)
            c = cut(sampler, ijk[i], tangents[i], min(int(r_vox * 2.5) + 2, 64),
                    max_half=64, grow_to=int(4.0 * r_vox) + 2, min_blob_voxels=12)
            if c is None or c.touches_border:
                continue
            major, minor = _axes(c.blob4)
            if minor <= 0:
                continue
            r_per[i] = _perimeter_um(c.blob4, sp) / (2.0 * np.pi)
            aspect[i] = major / minor
            d_over_r[i] = dnode[i] / max(float(radii[i]), 1e-9)
        ref = np.isfinite(r_per) & np.isfinite(d_over_r) & (d_over_r > 2.0)
        if ref.sum() < 3:
            continue
        r0, a0 = np.median(r_per[ref]), np.median(aspect[ref])
        if not (r0 > 0 and a0 > 0):
            continue
        n_seg += 1
        for lo, hi in _FLARE_BINS:
            sel = np.isfinite(r_per) & (d_over_r >= lo) & (d_over_r < hi)
            if sel.any():
                acc[(lo, hi)]["r"].append(float(np.median(r_per[sel])) / r0)
                acc[(lo, hi)]["a"].append(float(np.median(aspect[sel])) / a0)
    _emit(out, f"{n_seg} segments with a branched end and a usable interior reference")
    _emit(out, "each normalised by its OWN interior, so vessel calibre cancels")
    _emit(out)
    _emit(out, f"  {'distance to node':>18}{'segs':>6}{'r_perim / own':>15}"
               f"{'aspect / own':>14}")
    _emit(out, f"  {'(local radii)':>18}")
    for b in _FLARE_BINS:
        r = np.asarray(acc[b]["r"])
        a = np.asarray(acc[b]["a"])
        if len(r) < 5:
            continue
        _emit(out, f"  {f'{b[0]:.1f}-{b[1]:.1f}':>18}{len(r):>6}"
                   f"{np.median(r):>15.2f}{np.median(a):>14.2f}")
