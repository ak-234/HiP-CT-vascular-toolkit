"""Centerline and radius smoothing.

- ``smooth_centerline_savgol`` -- Savitzky-Golay along (x, y, z), endpoints
  preserved.
- ``smooth_centerline_bspline`` -- scipy ``splprep`` chord-length
  parametrised B-spline approximation, endpoints pinned.
- ``smooth_centerline`` -- top-level dispatch on
  ``config.CENTERLINE_SMOOTHER``.
- ``smooth_radius_transitions`` -- linear/cubic/cosine blend of radii
  near every junction node when the relative jump exceeds
  ``config.RADIUS_JUMP_THRESHOLD``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import runtime_config as config
from .centreline_reconnection import node_id_canon_map, bridge_centerline_gaps


def smooth_centerline_savgol(
    coords: np.ndarray,
    window_length: int | None = None,
    polyorder: int | None = None,
) -> np.ndarray:
    """Savitzky-Golay smoother that preserves the first and last points."""
    from scipy.signal import savgol_filter

    wl = config.SAVGOL_WINDOW if window_length is None else window_length
    po = config.SAVGOL_POLYORDER if polyorder is None else polyorder
    n = len(coords)
    if n < 3 or wl < 3:
        return coords.copy()
    wl = min(wl, n)
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, po + 2)
    if wl > n:
        return coords.copy()
    smoothed = savgol_filter(coords, wl, po, axis=0)
    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]
    return smoothed


def smooth_centerline_bspline(
    coords: np.ndarray,
    degree: int | None = None,
    s_per_point: float | None = None,
    min_pts: int | None = None,
    pin_start: int = 0,
    pin_end: int = 0,
) -> np.ndarray:
    """Chord-length-parametrised scipy ``splprep`` smoother with endpoint pinning.

    ``pin_start`` / ``pin_end`` taper the heavy endpoint weight over that many
    interior points at each end (graded, kink-safe), so the spline passes through
    the raw near-junction samples — keeping the approach tangent ~ raw at
    bifurcations without the weight discontinuity a hard pin would cause."""
    from scipy.interpolate import splprep, splev

    deg = config.BSPLINE_DEGREE if degree is None else degree
    spp = config.BSPLINE_S_PER_POINT if s_per_point is None else s_per_point
    mp = config.BSPLINE_MIN_PTS if min_pts is None else min_pts

    n = len(coords)
    if n < max(mp, deg + 1):
        return np.asarray(coords).copy()
    coords = np.asarray(coords, dtype=np.float64)

    diffs = np.diff(coords, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    if seglen.sum() < 1e-12:
        return coords.copy()
    u = np.zeros(n)
    u[1:] = np.cumsum(seglen)
    u /= u[-1]

    s = max(spp * n, 1e-12)
    # Strong-weight the endpoints so the fitted spline passes close to them.
    # Hard-overwriting smoothed[0]/[-1] back to coords[0]/[-1] at the end (as
    # we used to) jumps a free smoothed endpoint by up to a few tenths of a
    # mm with strong smoothing, which kinks the local tangent — visible as
    # wildly tilted cross-section circles right at segment joins. Weighting
    # the endpoints heavily makes splprep fit them directly without that
    # discontinuity; the final pin is then a sub-micron correction.
    W_HI = 1.0e4
    w = np.ones(n, dtype=np.float64)
    w[0] = w[-1] = W_HI
    # Graded taper of the heavy weight over the first/last K interior points at a
    # bifurcation end: high near the node, smoothstep down to 1 by point K. Forces
    # the fit through the raw near-junction samples (no tangent swing) without a
    # weight jump (no kink).
    if pin_start > 0 and n > 2:
        kk = min(int(pin_start), n - 2)
        for i in range(1, kk + 1):
            t = i / (kk + 1)
            s_step = t * t * (3.0 - 2.0 * t)
            w[i] = max(w[i], W_HI * (1.0 - s_step) + s_step)
    if pin_end > 0 and n > 2:
        kk = min(int(pin_end), n - 2)
        for i in range(1, kk + 1):
            t = i / (kk + 1)
            s_step = t * t * (3.0 - 2.0 * t)
            w[-1 - i] = max(w[-1 - i], W_HI * (1.0 - s_step) + s_step)
    try:
        tck, _ = splprep(
            [coords[:, 0], coords[:, 1], coords[:, 2]],
            u=u,
            w=w,
            k=min(deg, n - 1),
            s=s,
        )
        xs, ys, zs = splev(u, tck)
        smoothed = np.column_stack([xs, ys, zs])
    except Exception:
        return coords.copy()

    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]
    return smoothed


def adaptive_bspline_s_per_point(radius_mm: float, spp_base: float | None = None) -> float:
    """Radius-adaptive per-point B-spline residual budget (mm^2). Mirrors
    sdf_field.adaptive_smin_k: spp_base at BSPLINE_S_REF_RADIUS, scaled by
    (r / ref)^POWER, clipped to [MIN, MAX]. Keeps allowed centreline drift a
    constant fraction of vessel radius."""
    base = config.BSPLINE_S_PER_POINT if spp_base is None else spp_base
    if not config.BSPLINE_S_ADAPTIVE:
        return float(base)
    ref = max(config.BSPLINE_S_REF_RADIUS, 1e-9)
    spp = base * (max(float(radius_mm), 0.0) / ref) ** config.BSPLINE_S_RADIUS_POWER
    return float(np.clip(spp, config.BSPLINE_S_PER_POINT_MIN, config.BSPLINE_S_PER_POINT_MAX))


def densify_sparse_segments(
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    target_spacing_mm: float,
    min_points: int,
    verbose: bool = False,
) -> tuple[dict[int, tuple], int]:
    """Linearly interpolate extra centerline points + radii along under-sampled
    segments. Adds fresh point ids past max(points); mutates seg["point_ids"]
    in place. Returns (new_points_dict, n_segments_densified).

    Coordinates and thickness are stored in the same convention as
    parse_amira: micrometers for x, y, z and the thickness column. Each
    segment's first and last ``point_ids`` are preserved so node anchoring
    is unaffected.

    When ``verbose=True``, prints one line per processed segment with the
    decision (DENSIFY / SKIP-already-dense / SKIP-len / SKIP-pts) and the
    before/after counts.
    """
    if not segments:
        return dict(points), 0

    pts_out: dict[int, tuple] = dict(points)
    next_pid = (max(pts_out.keys()) + 1) if pts_out else 0
    n_densified = 0

    if verbose:
        print(
            f"[DENSIFY] action      seg_id  before  after  total_mm  required  reason"
        )

    for seg in segments:
        sid = seg.get("id", "?")
        pids = list(seg.get("point_ids", []))
        seg["raw_n_pts"] = len(pids)  # captured BEFORE densification for downstream diagnostics
        n = len(pids)
        if n < 2:
            if verbose:
                print(
                    f"[DENSIFY] SKIP-pts    {sid:>6}  {n:>6}  {n:>5}  {0.0:>8.3f}  {0:>8}  n<2"
                )
            continue
        coords = np.array(
            [
                (pts_out[p][0], pts_out[p][1], pts_out[p][2])
                for p in pids
            ],
            dtype=np.float64,
        )
        radii = np.array([float(pts_out[p][3]) for p in pids], dtype=np.float64)
        seg_lens = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_um = float(seg_lens.sum())
        if total_um <= 0.0:
            if verbose:
                print(
                    f"[DENSIFY] SKIP-len    {sid:>6}  {n:>6}  {n:>5}  {0.0:>8.3f}  {0:>8}  zero-length"
                )
            continue
        total_mm = total_um / 1000.0
        required = max(
            int(min_points),
            int(np.ceil(total_mm / max(target_spacing_mm, 1e-9))) + 1,
        )
        if n >= required:
            if verbose:
                print(
                    f"[DENSIFY] SKIP-dense  {sid:>6}  {n:>6}  {n:>5}  {total_mm:>8.3f}  {required:>8}  n>=required"
                )
            continue

        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])  # micrometers
        target_arc = np.linspace(0.0, total_um, required)

        new_xyz = np.empty((required, 3), dtype=np.float64)
        new_r = np.empty(required, dtype=np.float64)
        for axis in range(3):
            new_xyz[:, axis] = np.interp(target_arc, cum, coords[:, axis])
        new_r[:] = np.interp(target_arc, cum, radii)

        new_pids: list[int] = [pids[0]]
        for k in range(1, required - 1):
            pts_out[next_pid] = (
                float(new_xyz[k, 0]),
                float(new_xyz[k, 1]),
                float(new_xyz[k, 2]),
                float(new_r[k]),
            )
            new_pids.append(next_pid)
            next_pid += 1
        new_pids.append(pids[-1])
        seg["point_ids"] = new_pids
        n_densified += 1
        if verbose:
            print(
                f"[DENSIFY] DENSIFY     {sid:>6}  {n:>6}  {len(new_pids):>5}  {total_mm:>8.3f}  {required:>8}  n<required"
            )

    return pts_out, n_densified


def smooth_centerline(
    coords: np.ndarray,
    s_per_point: float | None = None,
    pin_start: int = 0,
    pin_end: int = 0,
) -> np.ndarray:
    """Dispatch on ``config.CENTERLINE_SMOOTHER``.

    ``s_per_point`` overrides the B-spline residual budget for this segment
    (used by the radius-adaptive callers); ``None`` keeps the global default.
    ``pin_start`` / ``pin_end`` taper the B-spline endpoint weight over that many
    near-junction points (bspline path only).
    """
    method = config.CENTERLINE_SMOOTHER
    if method == "bspline":
        return smooth_centerline_bspline(
            coords, s_per_point=s_per_point, pin_start=pin_start, pin_end=pin_end
        )
    if method == "savgol":
        if config.SAVGOL_WINDOW >= 3 and len(coords) >= config.SAVGOL_WINDOW:
            return smooth_centerline_savgol(coords, config.SAVGOL_WINDOW, config.SAVGOL_POLYORDER)
        return np.asarray(coords).copy()
    return np.asarray(coords).copy()


def smooth_segment_centerlines(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, tuple], int]:
    """Apply ``smooth_centerline`` to each segment's (x, y, z) coordinates and
    write the smoothed positions back into the ``points`` dict.

    Endpoints (first and last point of each segment) are pinned to their
    original positions by the smoother. Consecutive coincident samples are
    detected and excluded from the smoother input but their original
    positions are preserved in ``points``.

    Coordinates are stored in micrometers in the ``points`` dict; the
    smoother operates in mm and the result is rescaled.
    """
    if config.CENTERLINE_SMOOTHER == "none":
        return points, 0

    print(
        f"\n[CENTERLINE SMOOTH] Smoothing segment centerlines "
        f"via {config.CENTERLINE_SMOOTHER}..."
    )

    points = {pid: list(data) for pid, data in points.items()}
    n_segs_smoothed = 0
    n_points_modified = 0

    drift_verbose = bool(getattr(config, "SMOOTH_DRIFT_VERBOSE", False))
    drift_records: list[dict[str, Any]] = []

    # Bifurcation-degree map for kink-safe near-junction pin protection: taper the
    # B-spline endpoint weight over CENTERLINE_BIF_PIN_POINTS points at any end that
    # meets a deg>=3 node, so daughters keep their raw approach at the carina.
    node_to_segs: dict[int, set[int]] = {}
    for _si, _seg in enumerate(segments):
        for _nid in (_seg["node1"], _seg["node2"]):
            node_to_segs.setdefault(_nid, set()).add(_si)
    k_pin = int(getattr(config, "CENTERLINE_BIF_PIN_POINTS", 0))

    for seg in segments:
        pids = seg["point_ids"]
        if len(pids) < 2:
            continue
        coords = np.array(
            [
                [points[p][0] / 1000.0, points[p][1] / 1000.0, points[p][2] / 1000.0]
                for p in pids
            ],
            dtype=np.float64,
        )
        # Skip consecutive coincident samples — they crash chord-length splprep.
        d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        keep_mask = np.concatenate([[True], d > 1e-9])
        keep_pids = [pid for pid, k in zip(pids, keep_mask) if k]
        if len(keep_pids) < 2:
            continue

        coords_kept = coords[keep_mask]
        r_med = (
            float(np.median([points[p][3] for p in keep_pids]))
            / 1000.0
            * config.RADIUS_SCALE
        )
        spp_eff = adaptive_bspline_s_per_point(r_med)
        pin_start = k_pin if len(node_to_segs.get(seg["node1"], ())) >= 3 else 0
        pin_end = k_pin if len(node_to_segs.get(seg["node2"], ())) >= 3 else 0
        coords_smoothed = smooth_centerline(
            coords_kept, s_per_point=spp_eff, pin_start=pin_start, pin_end=pin_end
        )
        if coords_smoothed.shape != coords_kept.shape:
            continue

        # Per-point drift cap: bound how far any smoothed point may move from the
        # original centreline. Sub-cap noise/waviness corrections pass through;
        # the large low-frequency deviation that would bow a big vessel inward is
        # clamped (D = max(abs floor, FACTOR * local radius)). Endpoints carry
        # ~zero displacement (pinned), so node anchoring is unaffected.
        drift_factor = float(config.BSPLINE_MAX_DRIFT_RADIUS_FACTOR)
        seg_max_drift = 0.0
        seg_n_clamped = 0
        if drift_factor > 0.0:
            radii_kept = np.array(
                [float(points[p][3]) / 1000.0 * config.RADIUS_SCALE for p in keep_pids],
                dtype=np.float64,
            )
            disp = coords_smoothed - coords_kept
            mag = np.linalg.norm(disp, axis=1)
            seg_max_drift = float(mag.max()) if len(mag) else 0.0
            dcap = np.maximum(config.BSPLINE_MAX_DRIFT_MM, drift_factor * radii_kept)
            over = mag > dcap
            seg_n_clamped = int(over.sum())
            if over.any():
                scale = np.where(over, dcap / np.maximum(mag, 1e-12), 1.0)
                coords_smoothed = coords_kept + disp * scale[:, None]

        if drift_verbose and len(coords_kept) >= 3:
            sid = seg.get("id", "?")
            raw_n = int(seg.get("raw_n_pts", len(coords_kept)))
            was_densified = raw_n < int(config.DENSIFY_MIN_POINTS)

            def _rot_deg(t_before: np.ndarray, t_after: np.ndarray) -> float:
                nb = float(np.linalg.norm(t_before))
                na = float(np.linalg.norm(t_after))
                if nb < 1e-12 or na < 1e-12:
                    return 0.0
                cos = float(np.clip(np.dot(t_before, t_after) / (nb * na), -1.0, 1.0))
                return float(np.degrees(np.arccos(cos)))

            adj_drift_start = float(np.linalg.norm(coords_smoothed[1] - coords_kept[1]))
            adj_drift_end = float(np.linalg.norm(coords_smoothed[-2] - coords_kept[-2]))
            rot_start = _rot_deg(
                coords_kept[1] - coords_kept[0],
                coords_smoothed[1] - coords_smoothed[0],
            )
            rot_end = _rot_deg(
                coords_kept[-1] - coords_kept[-2],
                coords_smoothed[-1] - coords_smoothed[-2],
            )

            print(
                f"[SMOOTH-DRIFT] seg id={sid:>4}  end=start  "
                f"adj_drift={adj_drift_start:.3f}mm  tangent_rotation={rot_start:5.2f}deg  "
                f"r_med={r_med:.3f}mm  spp_eff={spp_eff:.5f}  "
                f"raw_n={raw_n}  densified={was_densified}"
            )
            print(
                f"[SMOOTH-DRIFT] seg id={sid:>4}  end=end    "
                f"adj_drift={adj_drift_end:.3f}mm  tangent_rotation={rot_end:5.2f}deg  "
                f"r_med={r_med:.3f}mm  spp_eff={spp_eff:.5f}  "
                f"max_drift={seg_max_drift:.3f}mm  clamped={seg_n_clamped}  "
                f"raw_n={raw_n}  densified={was_densified}"
            )
            drift_records.append({
                "seg_id": sid, "raw_n": raw_n, "was_densified": was_densified,
                "r_med_mm": r_med, "spp_eff": spp_eff,
                "max_drift_mm": seg_max_drift, "n_clamped": seg_n_clamped,
                "adj_drift_start_mm": adj_drift_start, "adj_drift_end_mm": adj_drift_end,
                "rot_start_deg": rot_start, "rot_end_deg": rot_end,
            })

        for i, pid in enumerate(keep_pids):
            new_xyz = coords_smoothed[i] * 1000.0
            if (
                abs(new_xyz[0] - points[pid][0]) > 1e-6
                or abs(new_xyz[1] - points[pid][1]) > 1e-6
                or abs(new_xyz[2] - points[pid][2]) > 1e-6
            ):
                points[pid][0] = float(new_xyz[0])
                points[pid][1] = float(new_xyz[1])
                points[pid][2] = float(new_xyz[2])
                n_points_modified += 1
        n_segs_smoothed += 1

    points = {pid: tuple(data) for pid, data in points.items()}

    print(
        f"  Smoothed {n_segs_smoothed} segments, "
        f"modified {n_points_modified} point positions"
    )

    if drift_verbose and drift_records:
        all_drifts = [
            d for rec in drift_records for d in (rec["adj_drift_start_mm"], rec["adj_drift_end_mm"])
        ]
        all_rots = [
            r for rec in drift_records for r in (rec["rot_start_deg"], rec["rot_end_deg"])
        ]
        dense_rots = [
            r for rec in drift_records if not rec["was_densified"]
            for r in (rec["rot_start_deg"], rec["rot_end_deg"])
        ]
        sparse_rots = [
            r for rec in drift_records if rec["was_densified"]
            for r in (rec["rot_start_deg"], rec["rot_end_deg"])
        ]
        max_drift = max(all_drifts)
        max_rot = max(all_rots)
        # Identify the seg id that hit max_rot.
        worst = max(
            drift_records,
            key=lambda r: max(r["rot_start_deg"], r["rot_end_deg"]),
        )
        print(f"[SMOOTH-DRIFT] summary over {len(drift_records)} segments:")
        print(
            f"  max adj_drift           = {max_drift:.3f} mm   "
            f"(seg id={worst['seg_id']}, raw_n={worst['raw_n']}, densified={worst['was_densified']})"
        )
        print(f"  max tangent_rotation    = {max_rot:.2f} deg")
        print(f"  median adj_drift        = {float(np.median(all_drifts)):.3f} mm")
        print(f"  median tangent_rotation = {float(np.median(all_rots)):.2f} deg")
        # Interior drift-cap stats (the inward-bend guard): pre-cap peak drift and
        # how many points were clamped back. A high max here means the cap is
        # actively preventing a large-vessel bow; raise/lower FACTOR to tune.
        if any("max_drift_mm" in rec for rec in drift_records):
            worst_drift = max(drift_records, key=lambda r: r.get("max_drift_mm", 0.0))
            total_clamped = sum(int(rec.get("n_clamped", 0)) for rec in drift_records)
            print(
                f"  max interior drift      = {worst_drift.get('max_drift_mm', 0.0):.3f} mm "
                f"(pre-cap; seg id={worst_drift['seg_id']}, r_med={worst_drift['r_med_mm']:.2f}mm)"
            )
            print(f"  total points clamped    = {total_clamped}")
        # Bucket adj_drift by radius (thin = below the adaptive reference) so the
        # radius-adaptive law can be tuned: thin should be near-zero, thick higher.
        r_ref = float(config.BSPLINE_S_REF_RADIUS)
        thin_drifts = [
            d for rec in drift_records if rec["r_med_mm"] < r_ref
            for d in (rec["adj_drift_start_mm"], rec["adj_drift_end_mm"])
        ]
        thick_drifts = [
            d for rec in drift_records if rec["r_med_mm"] >= r_ref
            for d in (rec["adj_drift_start_mm"], rec["adj_drift_end_mm"])
        ]
        if thin_drifts:
            print(
                f"  thin   (r_med< {r_ref:g}mm) : "
                f"median drift = {float(np.median(thin_drifts)):.3f} mm, "
                f"max drift = {max(thin_drifts):.3f} mm, "
                f"n_ends = {len(thin_drifts)}"
            )
        if thick_drifts:
            print(
                f"  thick  (r_med>={r_ref:g}mm) : "
                f"median drift = {float(np.median(thick_drifts)):.3f} mm, "
                f"max drift = {max(thick_drifts):.3f} mm, "
                f"n_ends = {len(thick_drifts)}"
            )
        if dense_rots:
            print(
                f"  dense  (raw_n>={config.DENSIFY_MIN_POINTS}) : "
                f"median rot = {float(np.median(dense_rots)):.2f} deg, "
                f"max rot = {max(dense_rots):.2f} deg, "
                f"n_ends = {len(dense_rots)}"
            )
        if sparse_rots:
            print(
                f"  sparse (raw_n< {config.DENSIFY_MIN_POINTS}) : "
                f"median rot = {float(np.median(sparse_rots)):.2f} deg, "
                f"max rot = {max(sparse_rots):.2f} deg, "
                f"n_ends = {len(sparse_rots)}"
            )

    return points, n_points_modified


def _menger_curvature(coords: np.ndarray) -> np.ndarray:
    """Discrete curvature kappa (1/mm) per point via the Menger circumradius of
    each interior triple ``(p[i-1], p[i], p[i+1])``. Endpoints return 0.

    ``kappa = 4 * Area / (a * b * c)`` where ``a, b, c`` are the triangle side
    lengths and ``Area`` is its area; ``kappa = 1 / circumradius``.
    """
    n = len(coords)
    kappa = np.zeros(n, dtype=np.float64)
    if n < 3:
        return kappa
    p_prev = coords[:-2]
    p_cur = coords[1:-1]
    p_next = coords[2:]
    a = np.linalg.norm(p_cur - p_prev, axis=1)
    b = np.linalg.norm(p_next - p_cur, axis=1)
    c = np.linalg.norm(p_next - p_prev, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(p_cur - p_prev, p_next - p_prev), axis=1)
    denom = a * b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(denom > 1e-12, 4.0 * area / denom, 0.0)
    kappa[1:-1] = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    return kappa


def _hermite_span_replace(coords: np.ndarray, a: int, b: int) -> np.ndarray:
    """Replace ``coords[a:b+1]`` with a cubic Hermite curve from ``coords[a]`` to
    ``coords[b]`` whose end tangents match the incoming/outgoing segment
    directions (``coords[a]-coords[a-1]`` and ``coords[b+1]-coords[b]``).

    Tangent-matching makes the replacement C1-continuous at both ends (no kink),
    and the Hermite distributes a moderate turn smoothly across the span, so the
    interior curvature drops. Requires ``1 <= a`` and ``b <= len(coords)-2``.
    Endpoints ``a`` and ``b`` are unchanged.
    """
    from scipy.interpolate import CubicHermiteSpline

    out = coords.copy()
    pa, pb = coords[a], coords[b]
    ta = coords[a] - coords[a - 1]
    tb = coords[b + 1] - coords[b]
    na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
    if na < 1e-12 or nb < 1e-12:
        return out
    ta /= na
    tb /= nb
    L = float(np.linalg.norm(pb - pa))
    if L < 1e-12:
        return out
    t = np.linspace(0.0, 1.0, b - a + 1)
    for ax in range(3):
        hs = CubicHermiteSpline([0.0, 1.0], [pa[ax], pb[ax]], [ta[ax] * L, tb[ax] * L])
        out[a:b + 1, ax] = hs(t)
    return out


def _build_segment_tubes(
    points: dict[int, tuple], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Global capsule tubes (mm) over all segments for non-adjacent clearance
    tests. Mirrors epicardial_annotation._segment_capsules but kept local to
    avoid an import cycle. Returns ``p0, p1, r0, r1, seg`` arrays, a KDTree of
    capsule midpoints, and the max ``r + 0.5*len`` search pad."""
    from scipy.spatial import cKDTree

    p0s, p1s, r0s, r1s, segs = [], [], [], [], []
    for si, seg in enumerate(segments):
        pids = [p for p in seg["point_ids"] if p in points]
        if len(pids) < 2:
            continue
        coords = np.array(
            [[points[p][0], points[p][1], points[p][2]] for p in pids]
        ) / 1000.0
        radii = np.array(
            [float(points[p][3]) / 1000.0 * config.RADIUS_SCALE for p in pids]
        )
        p0s.append(coords[:-1])
        p1s.append(coords[1:])
        r0s.append(radii[:-1])
        r1s.append(radii[1:])
        segs.append(np.full(len(pids) - 1, si, dtype=np.int64))
    if not p0s:
        return {"empty": True}
    P0 = np.vstack(p0s); P1 = np.vstack(p1s)
    R0 = np.concatenate(r0s); R1 = np.concatenate(r1s)
    SEG = np.concatenate(segs)
    mids = 0.5 * (P0 + P1)
    cap_len = np.linalg.norm(P1 - P0, axis=1)
    pad = float(np.max(np.maximum(R0, R1) + 0.5 * cap_len)) if len(cap_len) else 0.0
    return {"empty": False, "P0": P0, "P1": P1, "R0": R0, "R1": R1, "SEG": SEG,
            "kd": cKDTree(mids), "pad": pad}


def _nonadjacent_clearance(
    pts_mm: np.ndarray, radii_mm: np.ndarray, si: int,
    adj_si: set[int], tubes: dict[str, Any],
) -> float:
    """Minimum signed tube-to-tube clearance between the given points (of segment
    ``si``, with per-point radii) and every NON-adjacent segment's capsule.

    For each point: ``clearance = (dist_to_other_axis - r_other) - r_self``.
    Negative = the segment's tube overlaps a non-adjacent tube. Returns the min
    over all points (``+inf`` if nothing nearby)."""
    if tubes.get("empty", True):
        return float("inf")
    P0, P1, R0, R1, SEG = tubes["P0"], tubes["P1"], tubes["R0"], tubes["R1"], tubes["SEG"]
    kd, pad = tubes["kd"], tubes["pad"]
    worst = float("inf")
    for q, rq in zip(pts_mm, radii_mm):
        cand = kd.query_ball_point(q, float(rq) + pad)
        if not cand:
            continue
        cand = np.asarray(
            [c for c in cand if SEG[c] != si and SEG[c] not in adj_si], dtype=np.int64
        )
        if cand.size == 0:
            continue
        a = P0[cand]
        dvec = P1[cand] - a
        pa = q - a
        dd = np.einsum("ij,ij->i", dvec, dvec)
        t = np.clip(np.einsum("ij,ij->i", pa, dvec) / np.maximum(dd, 1e-12), 0.0, 1.0)
        closest = a + t[:, None] * dvec
        dist = np.linalg.norm(q - closest, axis=1)
        rad = R0[cand] + t * (R1[cand] - R0[cand])
        clear = float(np.min(dist - rad)) - float(rq)
        if clear < worst:
            worst = clear
    return worst


def limit_centerline_curvature(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, tuple], int]:
    """Locally straighten centreline bends that *self-intersect* — points where
    the radius of curvature ``R_c < r`` (``kappa*r > 1.0``), the condition under
    which a segment's swept tube self-overlaps on the inner side of the bend and
    the same-segment hard union (sdf_field) fuses the arms, deleting the inner
    wall.

    Only self-intersecting points are touched. For each such segment, the
    violating run (widened by ``CURVATURE_SMOOTH_WINDOW``) is replaced by a
    tangent-matched cubic Hermite (:func:`_hermite_span_replace`) blended in by
    the *smallest* alpha that lifts ``R_c`` just past ``FACTOR * r`` — minimal
    deviation from the original. When ``CURVATURE_COLLISION_AWARE``, the blend is
    capped so the moved body never penetrates a non-adjacent branch's tube more
    than the original did (preserving vessel-vessel separation); a fully blocked
    bend is left unchanged. Segment endpoints stay pinned to their nodes.

    Mutates and returns ``points`` (coordinates in µm), plus the count of moved
    points. Run after :func:`smooth_segment_centerlines` so curvature is measured
    on the smoothed centreline.
    """
    factor = float(config.CURVATURE_MIN_RADIUS_FACTOR)
    half_w = int(config.CURVATURE_SMOOTH_WINDOW)
    max_iters = int(config.CURVATURE_SMOOTH_MAX_ITERS)
    blend_steps = max(int(config.CURVATURE_BLEND_STEPS), 1)
    collision_aware = bool(config.CURVATURE_COLLISION_AWARE)
    margin = float(config.CURVATURE_COLLISION_MARGIN_MM)
    verbose = bool(getattr(config, "CURVATURE_VERBOSE", False))
    thresh = 1.0 / max(factor, 1e-9)        # correction target: kappa*r <= thresh
    SELF_INTERSECT = 1.0                     # detection: kappa*r > 1.0  (R_c < r)

    print("\n[CURVATURE] Straightening self-intersecting bends "
          f"(R_c < radius), minimal correction to R_c >= {factor:g} * radius...")

    node_to_segs: dict[int, set[int]] = {}      # node id -> incident segment indices
    for _si, _seg in enumerate(segments):
        for _nid in (_seg["node1"], _seg["node2"]):
            node_to_segs.setdefault(_nid, set()).add(_si)
    tubes = _build_segment_tubes(points, segments) if collision_aware else {"empty": True}

    points = {pid: list(data) for pid, data in points.items()}
    n_points_modified = 0
    n_segs_fixed = 0
    worst_before = 0.0
    worst_after = 0.0
    n_hit_cap = 0
    n_collision_capped = 0
    n_drift_capped = 0

    for si, seg in enumerate(segments):
        pids = seg["point_ids"]
        if len(pids) < 3:
            continue
        coords_all = np.array(
            [[points[p][0] / 1000.0, points[p][1] / 1000.0, points[p][2] / 1000.0]
             for p in pids],
            dtype=np.float64,
        )
        # Drop consecutive coincident samples (they break chord-length splprep
        # and skew the discrete curvature); operate on the kept subset.
        d = np.linalg.norm(np.diff(coords_all, axis=0), axis=1)
        keep_mask = np.concatenate([[True], d > 1e-9])
        keep_pids = [pid for pid, k in zip(pids, keep_mask) if k]
        if len(keep_pids) < 3:
            continue
        coords = coords_all[keep_mask]
        radii = np.array(
            [float(points[p][3]) / 1000.0 * config.RADIUS_SCALE for p in keep_pids],
            dtype=np.float64,
        )
        n = len(coords)

        # Near-bifurcation keep-out: hold the Hermite span CENTERLINE_BIF_PIN_POINTS
        # points clear of any bifurcation (deg>=3) end, so the pass can't drag a
        # daughter's near-carina points toward its sibling. 0 at terminal ends.
        k_pin = int(getattr(config, "CENTERLINE_BIF_PIN_POINTS", 0))
        k_start = k_pin if len(node_to_segs.get(seg["node1"], ())) >= 3 else 0
        k_end = k_pin if len(node_to_segs.get(seg["node2"], ())) >= 3 else 0

        ratio = _menger_curvature(coords) * radii      # kappa * r ; > 1.0 => self-intersecting
        viol = ratio > SELF_INTERSECT
        if not viol.any():
            continue
        seg_before = float(ratio.max())

        # Segments sharing a node with si are "adjacent" (legitimately touch at
        # the bifurcation); everything else is non-adjacent and must not be hit.
        adj_si: set[int] = set()
        for nid in (seg["node1"], seg["node2"]):
            adj_si |= node_to_segs.get(nid, set())
        adj_si.discard(si)

        idx = np.where(viol)[0]
        coords_orig = coords.copy()
        # Reference straightening = tangent-matched Hermite over the violating run
        # +/- WINDOW (widened only if the full blend can't reach the target — a
        # base window usually overshoots). Blend orig -> Hermite by the SMALLEST
        # alpha that lifts R_c just past FACTOR*r (minimal geometry change), and
        # stop increasing alpha once the moved body would penetrate a non-adjacent
        # tube more than the original did (clearance cap). Fully blocked -> no
        # change. best = (coords, kappa*r max, alpha, collision_capped).
        best = (coords_orig, seg_before, 0.0, False)
        iters_used = 0
        for grow in range(max_iters):
            a = max(1 + k_start, int(idx[0]) - half_w - 2 * grow)
            b = min(n - 2 - k_end, int(idx[-1]) + half_w + 2 * grow)
            if b - a + 1 < 3:
                break
            iters_used = grow + 1
            full = _hermite_span_replace(coords_orig, a, b)
            span = slice(a, b + 1)
            if collision_aware:
                orig_clear = _nonadjacent_clearance(
                    coords_orig[span], radii[span], si, adj_si, tubes)
                clear_floor = min(orig_clear, 0.0) - margin
            win_coords, win_max, win_alpha, capped = coords_orig, seg_before, 0.0, False
            for k in range(1, blend_steps + 1):
                alpha = k / blend_steps
                cand = coords_orig.copy()
                cand[span] = coords_orig[span] + alpha * (full[span] - coords_orig[span])
                if collision_aware:
                    if _nonadjacent_clearance(cand[span], radii[span], si, adj_si, tubes) < clear_floor:
                        capped = True
                        break                          # keep last clear win_*
                win_coords, win_alpha = cand, alpha
                win_max = float((_menger_curvature(cand) * radii).max())
                if win_max <= thresh:
                    break
            if win_alpha > 0.0 and win_max < best[1]:
                best = (win_coords, win_max, win_alpha, capped)
            if best[1] <= thresh:
                break
            if a <= 1 and b >= n - 2:
                break                                 # cannot widen the span further

        best_coords, best_max, best_alpha, best_capped = best
        if best_alpha <= 0.0:
            continue                                  # nothing safely applicable -> leave as-is

        # Per-point displacement cap: bound how far the curvature correction may move
        # the centreline from the (B-spline-smoothed) path, so a tight bend can't drag
        # the whole segment inward toward its chord. Mirrors the B-spline drift cap in
        # smooth_segment_centerlines. Points outside the span carry zero displacement.
        curv_drift_factor = float(getattr(config, "CURVATURE_MAX_DRIFT_RADIUS_FACTOR", 0.0))
        if curv_drift_factor > 0.0:
            disp = best_coords - coords_orig
            mag = np.linalg.norm(disp, axis=1)
            dcap = np.maximum(config.CURVATURE_MAX_DRIFT_MM, curv_drift_factor * radii)
            over = mag > dcap
            if over.any():
                scale = np.where(over, dcap / np.maximum(mag, 1e-12), 1.0)
                best_coords = coords_orig + disp * scale[:, None]
                best_max = float((_menger_curvature(best_coords) * radii).max())
                n_drift_capped += 1

        if best_max > thresh:
            if best_capped:
                n_collision_capped += 1               # held back by a non-adjacent branch
            else:
                n_hit_cap += 1                         # geometry-limited (e.g. hairpin)

        coords = best_coords
        seg_after = best_max
        worst_before = max(worst_before, seg_before)
        worst_after = max(worst_after, seg_after)
        n_segs_fixed += 1

        moved = 0
        for i, pid in enumerate(keep_pids):
            new = coords[i] * 1000.0
            if (abs(new[0] - points[pid][0]) > 1e-6
                    or abs(new[1] - points[pid][1]) > 1e-6
                    or abs(new[2] - points[pid][2]) > 1e-6):
                points[pid][0] = float(new[0])
                points[pid][1] = float(new[1])
                points[pid][2] = float(new[2])
                moved += 1
        n_points_modified += moved

        if verbose:
            print(
                f"[CURVATURE] seg id={seg.get('id', '?'):>4}  "
                f"R_c/r worst: {(1.0 / seg_before if seg_before > 0 else float('inf')):.2f}"
                f" -> {(1.0 / seg_after if seg_after > 0 else float('inf')):.2f} "
                f"(target >= {factor:g})  alpha={best_alpha:.2f}  "
                f"{'COLLISION-CAPPED  ' if best_capped and best_max > thresh else ''}moved={moved}"
            )

    points = {pid: tuple(data) for pid, data in points.items()}
    print(
        f"  Fixed {n_segs_fixed} self-intersecting segment(s), "
        f"modified {n_points_modified} points"
        + (f"; {n_collision_capped} held back by a non-adjacent branch"
           if n_collision_capped else "")
        + (f"; {n_hit_cap} geometry-limited" if n_hit_cap else "")
        + (f"; {n_drift_capped} drift-capped" if n_drift_capped else "")
    )
    return points, n_points_modified


def smooth_segment_radii(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, tuple], int]:
    """Savitzky-Golay denoise r(s) along each segment's interior.

    Endpoints (``point_ids[0]`` and ``point_ids[-1]``) are pinned to their
    original values so the downstream junction passes
    (:func:`prune_terminal_shrink`, :func:`prune_bifurcation_shrink`,
    :func:`smooth_radius_transitions`) see the same endpoint radii they
    would have without this pass.
    """
    if not config.SMOOTH_SEGMENT_RADII:
        return points, 0

    from scipy.signal import savgol_filter

    print("\n[RADIUS SMOOTH] Denoising in-segment radius profiles via Savitzky-Golay...")

    polyorder = int(config.RADIUS_SAVGOL_POLYORDER)
    window_cfg = int(config.RADIUS_SAVGOL_WINDOW)

    points = {pid: list(data) for pid, data in points.items()}
    n_segs_smoothed = 0
    n_pts_mod = 0
    n_segs_skipped = 0

    for seg in segments:
        pids = seg["point_ids"]
        n = len(pids)
        if n < polyorder + 2:
            n_segs_skipped += 1
            continue

        wl = min(window_cfg, n)
        if wl % 2 == 0:
            wl -= 1
        wl = max(wl, polyorder + 2)
        if wl > n or wl < 3:
            n_segs_skipped += 1
            continue

        radii = np.array([points[pid][3] for pid in pids], dtype=np.float64)
        smoothed = savgol_filter(radii, wl, polyorder)
        # Pin both endpoints.
        smoothed[0] = radii[0]
        smoothed[-1] = radii[-1]

        seg_modified = False
        for i, pid in enumerate(pids):
            new_r = float(smoothed[i])
            if abs(new_r - points[pid][3]) > 1e-9:
                points[pid][3] = new_r
                n_pts_mod += 1
                seg_modified = True
        if seg_modified:
            n_segs_smoothed += 1

    points = {pid: tuple(data) for pid, data in points.items()}

    print(
        f"  Smoothed {n_segs_smoothed} segments, modified {n_pts_mod} point radii "
        f"(window={window_cfg}, polyorder={polyorder}, "
        f"{n_segs_skipped} segments too short)"
    )
    return points, n_segs_smoothed


def smooth_radius_transitions(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, tuple], int]:
    """Blend radii near junctions to remove sharp jumps at segment endpoints.

    Uses a target radius (weighted mean or parent) and blends each connected
    segment from the target at the node toward its local interior reference.
    Endpoint outliers are clamped against a local interior median before
    computing the target. Only fires when the jump exceeds
    ``config.RADIUS_JUMP_THRESHOLD`` or an outlier clamp is applied.
    """
    if not config.SMOOTH_RADIUS_TRANSITIONS:
        return points, 0

    print("\n[RADIUS SMOOTH] Smoothing radius transitions at segment junctions...")

    canon = node_id_canon_map(nodes, config.NODE_COINCIDENCE_EPS_MM)

    def _canon(nid: int) -> int:
        return canon.get(nid, nid)

    node_to_segs: dict[int, list[dict[str, Any]]] = {}
    seen: dict[int, set[tuple[int, int]]] = {}

    def _add(cnid: int, seg: dict[str, Any], endpoint_idx: int, node_id: int) -> None:
        seg_key = seg.get("id", id(seg))
        seen.setdefault(cnid, set())
        if (seg_key, endpoint_idx) in seen[cnid]:
            return
        seen[cnid].add((seg_key, endpoint_idx))
        node_to_segs.setdefault(cnid, []).append(
            {"seg": seg, "endpoint_idx": endpoint_idx, "node_id": node_id}
        )

    for seg in segments:
        pids = seg["point_ids"]
        if len(pids) < 2:
            continue
        first_pt = np.array(points[pids[0]][:3], dtype=np.float64)
        last_pt = np.array(points[pids[-1]][:3], dtype=np.float64)

        nid1 = seg["node1"]
        nid2 = seg["node2"]
        n1_pos = np.array(nodes[nid1][:3], dtype=np.float64)
        n2_pos = np.array(nodes[nid2][:3], dtype=np.float64)
        n1_idx = 0 if np.linalg.norm(first_pt - n1_pos) < np.linalg.norm(last_pt - n1_pos) else -1
        n2_idx = 0 if np.linalg.norm(first_pt - n2_pos) < np.linalg.norm(last_pt - n2_pos) else -1

        c1 = _canon(nid1)
        c2 = _canon(nid2)
        if c1 == c2:
            if nid1 == c1:
                _add(c1, seg, n1_idx, nid1)
            elif nid2 == c2:
                _add(c2, seg, n2_idx, nid2)
            else:
                _add(c1, seg, n1_idx, nid1)
        else:
            _add(c1, seg, n1_idx, nid1)
            _add(c2, seg, n2_idx, nid2)

    potential = {nid: lst for nid, lst in node_to_segs.items() if len(lst) >= 2}
    print(f"  Found {len(potential)} potential junction nodes (2+ segments, canonicalized)")

    points = {pid: list(data) for pid, data in points.items()}

    n_smoothed = 0
    n_pts_mod = 0
    n_below = 0
    n_outlier_clamped = 0

    for nid, connected_segs in potential.items():
        endpoint_info: list[dict[str, Any]] = []
        for entry in connected_segs:
            seg = entry["seg"]
            pids = seg["point_ids"]
            if len(pids) < 2:
                continue
            endpoint_idx = entry["endpoint_idx"]
            if endpoint_idx == 0:
                ep_pids = pids[: config.RADIUS_BLEND_POINTS + 1]
            else:
                ep_pids = pids[-(config.RADIUS_BLEND_POINTS + 1):]
                ep_pids = list(reversed(ep_pids))

            radii = [float(points[pid][3]) for pid in ep_pids]
            if not radii:
                continue
            look = min(int(config.RADIUS_ENDPOINT_LOOKAHEAD), max(len(radii) - 1, 0))
            if look > 0:
                interior_rs = [r for r in radii[1: 1 + look] if r > 0]
            else:
                interior_rs = []
            if interior_rs:
                interior_ref = float(np.median(interior_rs))
            else:
                interior_ref = float(radii[-1]) if radii[-1] > 0 else float(radii[0])

            endpoint_r = float(radii[0])
            endpoint_eff = endpoint_r
            clamped = False
            ratio = float(config.RADIUS_ENDPOINT_OUTLIER_RATIO)
            if ratio > 1.0 and interior_ref > 0:
                low = interior_ref / ratio
                high = interior_ref * ratio
                if endpoint_eff < low:
                    endpoint_eff = low
                    clamped = True
                elif endpoint_eff > high:
                    endpoint_eff = high
                    clamped = True
            if clamped:
                n_outlier_clamped += 1

            endpoint_info.append(
                {
                    "pids": ep_pids,
                    "endpoint_r": endpoint_r,
                    "endpoint_eff": endpoint_eff,
                    "interior_ref": interior_ref,
                    "clamped": clamped,
                    "strahler": int(seg.get("strahler", 0)),
                }
            )
        if len(endpoint_info) < 2:
            continue

        target_mode = str(config.RADIUS_JUNCTION_TARGET).lower()
        if target_mode == "parent":
            # Sort and pick target by interior_ref, not endpoint_eff.
            # Amira often writes near-uniform thicknesses at junction
            # points themselves, so endpoint_eff doesn't discriminate
            # parent from daughters; interior_ref reflects each
            # segment's true local radius.
            endpoint_info.sort(
                key=lambda e: (e["strahler"], e["interior_ref"]), reverse=True
            )
            target_r = float(endpoint_info[0]["interior_ref"])
        else:
            weight_mode = str(config.RADIUS_JUNCTION_WEIGHTING).lower()
            power = float(config.RADIUS_JUNCTION_WEIGHT_POWER)
            weighted: list[tuple[float, float]] = []
            for info in endpoint_info:
                r_eff = float(info["endpoint_eff"])
                if r_eff <= 0:
                    continue
                if target_mode == "mean" or weight_mode == "equal":
                    w = 1.0
                elif weight_mode == "radius":
                    w = max(r_eff, 1.0e-6) ** power
                else:
                    w = max(info["strahler"], 1) ** power
                weighted.append((w, r_eff))
            sum_w = sum(w for w, _ in weighted)
            if sum_w <= 0:
                n_below += 1
                continue
            target_r = sum(w * r for w, r in weighted) / sum_w

        if target_r <= 0:
            n_below += 1
            continue

        clamped_here = any(info["clamped"] for info in endpoint_info)
        # Fire the junction blend if either an endpoint OR an interior
        # radius differs from target_r by more than the jump threshold.
        # Adding interior_ref to the gate is necessary because Amira
        # often gives uniform endpoint thicknesses at junctions while
        # the real radius mismatch lives in the segment interiors.
        needs = clamped_here or any(
            info["endpoint_eff"] > 0
            and abs(info["endpoint_eff"] - target_r) / target_r > config.RADIUS_JUMP_THRESHOLD
            for info in endpoint_info
        ) or any(
            info["interior_ref"] > 0
            and abs(info["interior_ref"] - target_r) / target_r > config.RADIUS_JUMP_THRESHOLD
            for info in endpoint_info
        )
        if not needs:
            n_below += 1
            continue

        # Identify the parent at this junction = argmax over endpoint_info
        # by (strahler desc, interior_ref desc). Reused by the asymmetric
        # blend below regardless of which target_mode produced target_r,
        # and by the verbose log immediately after. We use interior_ref
        # (not endpoint_eff) because Amira often writes uniform endpoint
        # thicknesses at junctions, so endpoint_eff can't tell parent
        # from daughter; interior_ref reflects the true local radius.
        parent_idx_sorted = max(
            range(len(endpoint_info)),
            key=lambda i: (
                endpoint_info[i]["strahler"],
                endpoint_info[i]["interior_ref"],
            ),
        )

        n_smoothed += 1

        if config.RADIUS_TRANSITION_VERBOSE:
            parent_info = endpoint_info[parent_idx_sorted]
            daughter_endpoints = ", ".join(
                f"{info['endpoint_eff']:.3f}"
                for i, info in enumerate(endpoint_info)
                if i != parent_idx_sorted
            )
            print(
                f"    junction nid={nid}  target_r={target_r:.3f}  "
                f"parent(strahler={parent_info['strahler']}, "
                f"r={parent_info['endpoint_eff']:.3f})  "
                f"daughters=[{daughter_endpoints}]"
            )

        asym = bool(config.ASYMMETRIC_JUNCTION_BLEND)
        parent_w = float(config.RADIUS_JUNCTION_PARENT_BLEND_WEIGHT)
        daughter_w = float(config.RADIUS_JUNCTION_DAUGHTER_BLEND_WEIGHT)

        for idx, info in enumerate(endpoint_info):
            pids = info["pids"]
            n_blend = min(len(pids), config.RADIUS_BLEND_POINTS + 1)
            if n_blend < 2:
                continue
            far_r = float(info["interior_ref"])
            if far_r <= 0:
                current = [points[pid][3] for pid in pids[:n_blend]]
                far_r = float(current[-1]) if current else 0.0
            # Anti-neck: never pull a segment's node radius BELOW its own local
            # interior radius. target_r is the (max-Strahler) parent's radius,
            # which at a multifurcation can be THINNER than a thick through-
            # vessel that happens to be lower Strahler; without this clamp that
            # vessel necks down to target_r at the node -> a dish indentation
            # on the trunk. max(target_r, far_r) keeps thick vessels at full
            # radius while thin daughters still flare up to target_r as before.
            if getattr(config, "RADIUS_JUNCTION_NO_NECK", True):
                node_target = max(target_r, far_r)
            else:
                node_target = target_r
            if asym:
                w_seg = parent_w if idx == parent_idx_sorted else daughter_w
            else:
                w_seg = 1.0
            for i in range(n_blend):
                t = i / max(n_blend - 1, 1)
                if config.RADIUS_BLEND_METHOD == "cosine":
                    bf = (1.0 - np.cos(t * np.pi)) / 2.0
                elif config.RADIUS_BLEND_METHOD == "cubic":
                    bf = t * t * (3 - 2 * t)
                else:
                    bf = t
                blended_r = node_target * (1.0 - bf) + far_r * bf
                pid = pids[i]
                original_r = float(points[pid][3])
                new_r = original_r * (1.0 - w_seg) + blended_r * w_seg
                if abs(new_r - original_r) > 1e-6:
                    points[pid][3] = new_r
                    n_pts_mod += 1

    points = {pid: tuple(data) for pid, data in points.items()}

    print(f"  Smoothed {n_smoothed} junctions, modified {n_pts_mod} point radii")
    print(
        f"  {n_below} junctions found but below threshold "
        f"({config.RADIUS_JUMP_THRESHOLD*100:.1f}% jump)"
    )
    print(f"  Clamped {n_outlier_clamped} endpoint outliers at junctions")
    print(f"  Blend points: {config.RADIUS_BLEND_POINTS}, method: {config.RADIUS_BLEND_METHOD}")
    if config.ASYMMETRIC_JUNCTION_BLEND:
        print(
            f"  Asymmetric junction blend: parent_w="
            f"{float(config.RADIUS_JUNCTION_PARENT_BLEND_WEIGHT):.2f}, "
            f"daughter_w="
            f"{float(config.RADIUS_JUNCTION_DAUGHTER_BLEND_WEIGHT):.2f}"
        )
    return points, n_smoothed


def prune_terminal_shrink(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, tuple], int]:
    """Clamp abrupt-shrink terminal contours up to their preceding interior radius.

    For each segment whose endpoint is a terminal (degree-1) node, walk inward
    from the terminal point. If the terminal-side radius is below
    ``config.TERMINAL_SHRINK_THRESHOLD * r_interior``, clamp it to the interior
    radius. Continue walking up to ``config.TERMINAL_SHRINK_MAX_WALK`` points
    until a stable contour is found. Fixes annotation artifacts where the last
    one or two centerline samples carry a much smaller thickness than the rest
    of the vessel.
    """
    if not config.PRUNE_TERMINAL_SHRINK:
        return points, 0

    print("\n[RADIUS SMOOTH] Pruning abrupt terminal contour shrinks...")

    # Canonicalize node IDs so near-coincident Amira nodes that actually
    # belong to a single junction are not mistakenly treated as terminals.
    canon = node_id_canon_map(nodes, config.NODE_COINCIDENCE_EPS_MM)

    def _canon(nid: int) -> int:
        return canon.get(nid, nid)

    canonical_segs: dict[int, list[dict[str, Any]]] = {}
    for seg in segments:
        for nid in (seg["node1"], seg["node2"]):
            canonical_segs.setdefault(_canon(nid), []).append(seg)

    terminals = [
        (nid, canonical_segs[_canon(nid)][0])
        for nid in nodes
        if len(canonical_segs.get(_canon(nid), [])) == 1
    ]

    points = {pid: list(data) for pid, data in points.items()}

    threshold = float(config.TERMINAL_SHRINK_THRESHOLD)
    max_walk = int(config.TERMINAL_SHRINK_MAX_WALK)
    look_ahead = int(config.TERMINAL_SHRINK_LOOKAHEAD)
    stop_after = int(config.SHRINK_STOP_AFTER_STABLE)

    n_clamped = 0
    n_fixed_segs = 0

    for nid, seg in terminals:
        pids = seg["point_ids"]
        if len(pids) < 2:
            continue
        node_pos = np.array(nodes[nid][:3], dtype=np.float64)
        first_pt = np.array(points[pids[0]][:3], dtype=np.float64)
        last_pt = np.array(points[pids[-1]][:3], dtype=np.float64)
        # Walk indices ordered terminal -> interior. walk[0] is the terminal
        # point; walk[k+1] is more interior than walk[k]. Horizon must hold
        # max_walk outer points plus look_ahead inner reference points.
        horizon = max_walk + look_ahead + 1
        if np.linalg.norm(first_pt - node_pos) < np.linalg.norm(last_pt - node_pos):
            walk = list(range(min(horizon, len(pids))))
        else:
            walk = [-i - 1 for i in range(min(horizon, len(pids)))]

        seg_fixed = False
        stable_run = 0
        for k in range(min(max_walk, len(walk))):
            pid_outer = pids[walk[k]]
            ahead = walk[k + 1: k + 1 + look_ahead]
            inner_rs = [
                points[pids[i]][3]
                for i in ahead
                if points[pids[i]][3] > 0
            ]
            if not inner_rs:
                break
            r_ref = float(np.median(inner_rs))
            r_out = points[pid_outer][3]
            if r_out < threshold * r_ref:
                points[pid_outer][3] = r_ref
                n_clamped += 1
                seg_fixed = True
                stable_run = 0
            else:
                stable_run += 1
                if stable_run >= stop_after:
                    break
        if seg_fixed:
            n_fixed_segs += 1

    points = {pid: tuple(data) for pid, data in points.items()}

    print(
        f"  Clamped {n_clamped} terminal points across {n_fixed_segs} segments "
        f"(threshold: r < {threshold:.2f} * median of next {look_ahead}, "
        f"max walk: {max_walk}, stop after {stop_after} stable)"
    )
    return points, n_clamped


def prune_bifurcation_shrink(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, tuple], int]:
    """Clamp abrupt-shrink bifurcation-side contours to the local interior median.

    Symmetric to :func:`prune_terminal_shrink` but operates on segment
    endpoints at degree>=3 (bifurcation) nodes. Walks inward from the bif
    endpoint and, while each outer point's radius is below ``threshold *``
    the median of the next ``LOOKAHEAD`` interior radii, clamps it to that
    median. The look-ahead median is more robust than a single-pair check
    when the noisy stretch spans several consecutive points.
    """
    if not config.PRUNE_BIFURCATION_SHRINK:
        return points, 0

    print("\n[RADIUS SMOOTH] Pruning abrupt bifurcation contour shrinks...")

    threshold = float(config.BIFURCATION_SHRINK_THRESHOLD)
    max_walk = int(config.BIFURCATION_SHRINK_MAX_WALK)
    look_ahead = int(config.BIFURCATION_SHRINK_LOOKAHEAD)
    stop_after = int(config.SHRINK_STOP_AFTER_STABLE)

    # Canonicalize node IDs so a single anatomical bifurcation encoded as
    # multiple near-coincident Amira nodes (each holding one branch endpoint)
    # is detected as one degree>=3 junction instead of three degree-1
    # terminals.
    canon = node_id_canon_map(nodes, config.NODE_COINCIDENCE_EPS_MM)

    def _canon(nid: int) -> int:
        return canon.get(nid, nid)

    canonical_degree: dict[int, int] = {}
    for seg in segments:
        for nid in (seg["node1"], seg["node2"]):
            cid = _canon(nid)
            canonical_degree[cid] = canonical_degree.get(cid, 0) + 1

    points = {pid: list(data) for pid, data in points.items()}
    n_clamped = 0
    n_fixed_segs = 0

    for seg in segments:
        pids = seg["point_ids"]
        if len(pids) < 2:
            continue
        for nid in (seg["node1"], seg["node2"]):
            if canonical_degree.get(_canon(nid), 0) < 3:
                continue  # only bifurcation nodes (canonical degree)
            node_pos = np.array(nodes[nid][:3], dtype=np.float64)
            first_pt = np.array(points[pids[0]][:3], dtype=np.float64)
            last_pt = np.array(points[pids[-1]][:3], dtype=np.float64)
            horizon = max_walk + look_ahead + 1
            if np.linalg.norm(first_pt - node_pos) < np.linalg.norm(last_pt - node_pos):
                walk = list(range(min(horizon, len(pids))))
            else:
                walk = [-i - 1 for i in range(min(horizon, len(pids)))]

            seg_fixed = False
            stable_run = 0
            for k in range(min(max_walk, len(walk))):
                pid_outer = pids[walk[k]]
                ahead = walk[k + 1: k + 1 + look_ahead]
                if not ahead:
                    break
                inner_rs = [
                    points[pids[i]][3]
                    for i in ahead
                    if points[pids[i]][3] > 0
                ]
                if not inner_rs:
                    break
                r_ref = float(np.median(inner_rs))
                r_out = points[pid_outer][3]
                if r_out < threshold * r_ref:
                    points[pid_outer][3] = r_ref
                    n_clamped += 1
                    seg_fixed = True
                    stable_run = 0
                else:
                    stable_run += 1
                    if stable_run >= stop_after:
                        break
            if seg_fixed:
                n_fixed_segs += 1

    points = {pid: tuple(data) for pid, data in points.items()}

    print(
        f"  Clamped {n_clamped} bif-side contours across {n_fixed_segs} segments "
        f"(threshold: r < {threshold:.2f} * median of next {look_ahead}, max walk: {max_walk})"
    )
    return points, n_clamped


__all__ = [
    "smooth_centerline_savgol",
    "smooth_centerline_bspline",
    "smooth_centerline",
    "adaptive_bspline_s_per_point",
    "densify_sparse_segments",
    "smooth_segment_centerlines",
    "limit_centerline_curvature",
    "smooth_segment_radii",
    "smooth_radius_transitions",
    "prune_terminal_shrink",
    "prune_bifurcation_shrink",
]
