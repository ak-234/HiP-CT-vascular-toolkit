"""Independent section-centre evidence and final geometry qualification.

An elongated section is not by itself an oblique section. Ellipse fits are
diagnostics, not replacement contours or a reason to force a flattened lumen
to be round. All offsets returned here are in physical micrometres.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import EllipseModel, find_contours


def section_shape(cut, spacing):
    mask = np.asarray(cut.blob8, dtype=bool)
    pixels = np.argwhere(mask)
    centre = pixels.mean(axis=0)
    offset = (centre-cut.half)*spacing
    covariance = np.cov(pixels.T, bias=True)*spacing**2
    axes = 2*np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), spacing**2/12))
    minor, major = map(float, axes)
    inside = bool(ndimage.map_coordinates(mask.astype(float), centre[:, None],
                                         order=0, mode='constant', cval=0)[0] > 0)
    row = dict(centroid_uv_um=offset.tolist(), minor_semiaxis_um=minor,
               major_semiaxis_um=major, axis_ratio=major/minor,
               offset_um=float(np.linalg.norm(offset)),
               offset_short_radius=float(np.linalg.norm(offset)/minor),
               centroid_inside=inside, ellipse_status='unavailable')
    contours = find_contours(mask.astype(float), .5)
    if contours:
        boundary = max(contours, key=len)
        # Bound fitting cost independently of cross-section pixel area.
        boundary = boundary[::max(1, len(boundary)//512)]
        model = EllipseModel()
        if len(boundary) >= 6 and model.estimate(boundary):
            cu, cv, a, b, angle = model.params
            fitted = np.array([cu, cv])
            delta = boundary-fitted
            rotated = delta @ np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle), np.cos(angle)]])
            residual = np.linalg.norm(rotated/np.array([a, b]), axis=1)-1
            if np.isfinite(residual).all() and min(a, b) > 0:
                rms = float(np.sqrt(np.mean(residual**2)))
                ellipse_inside = bool(ndimage.map_coordinates(mask.astype(float), fitted[:, None],
                    order=0, mode='constant', cval=0)[0] > 0)
                row.update(ellipse_status=('diagnostic_only' if rms <= .12 and ellipse_inside
                                           and min(a, b) >= 2 else 'poor_fit'),
                    ellipse_centroid_disagreement_um=float(np.linalg.norm(fitted-centre)*spacing),
                    ellipse_radial_rms=rms,
                    ellipse_axis_ratio=float(max(a, b)/min(a, b)),
                    ellipse_centre_uv_um=((fitted-cut.half)*spacing).tolist())
    return row


def centring_summary(records, spacing):
    """Provisional tolerance: max(0.75 voxel, 15% of section short semi-axis).

    Keep physical and normalised errors; the tolerance is not a biological
    accuracy claim. No reliable sections means unknown, never centred.
    """
    valid = [r for r in records if r.get('accepted')]
    if not valid:
        return dict(status='insufficient_support', accepted_sections=0)
    offsets = np.array([r['offset_um'] for r in valid])
    minor = np.array([r['minor_semiaxis_um'] for r in valid])
    tolerance = np.maximum(.75*spacing, .15*minor)
    failed = [r['point'] for r, bad in zip(valid, offsets > tolerance) if bad]
    return dict(status='off_centre' if failed else 'centred', accepted_sections=len(valid),
                max_offset_um=float(offsets.max()), p95_offset_um=float(np.percentile(offsets, 95)),
                max_offset_short_radius=float(np.max(offsets/minor)),
                outside_tolerance_points=failed,
                tolerance='max(0.75 voxel, 0.15 short semi-axis); provisional')


def curve_diagnostics(x):
    d = np.diff(x, axis=0)
    length = np.linalg.norm(d, axis=1)
    if len(length) < 2:
        return dict(max_turn_degrees=0., max_curvature_per_um=0.)
    unit = d/np.maximum(length[:, None], 1e-12)
    angle = np.arccos(np.clip(np.sum(unit[:-1]*unit[1:], axis=1), -1., 1.))
    curvature = 2*np.sin(angle/2)/np.maximum((length[:-1]+length[1:])/2, 1e-12)
    return dict(max_turn_degrees=float(np.rad2deg(angle.max())),
                max_curvature_per_um=float(curvature.max()))


def join_diagnostics(graph, selected):
    selected = set(selected)
    rows = {}
    for node in graph.nodes:
        incident = sorted(graph.node_segments(node))
        if len(incident) != 2 or not set(incident) <= selected:
            continue
        directions = []
        for sid in incident:
            x = graph.coords(sid)
            if graph.segment(sid)['node2'] == node:
                x = x[::-1]
            d = x[1]-x[0]
            directions.append(d/max(np.linalg.norm(d), 1e-12))
        residual = float(np.linalg.norm(directions[0]+directions[1]))
        rows[node] = dict(segments=incident, unit_tangent_residual=residual,
            mismatch_degrees=float(np.rad2deg(np.arccos(np.clip(
                -np.dot(directions[0], directions[1]), -1, 1)))),
            status='continuous' if residual <= 1e-3 else 'review_required')
    return rows
