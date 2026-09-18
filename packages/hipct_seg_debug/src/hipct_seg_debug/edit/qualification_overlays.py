"""Portable segmentation-plane and radius-profile review artifacts."""
from __future__ import annotations

import numpy as np


def write_overlays(measured, reconstruction, targets, frame, labels, directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from ..crosssection import _PlaneSampler, cut, robust_edge_tangents, _plane_axes
    from .centreline_refine import arclength
    from .radius_profile import trusted_radii
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    outputs = []
    for sid in sorted(targets):
        x, r = measured.coords(sid), measured.radii(sid)
        y, profile = reconstruction.coords(sid), reconstruction.radii(sid)
        tangent = robust_edge_tangents(x, r, spacing_um=sp)
        surface_tangent = robust_edge_tangents(y, profile, spacing_um=sp)
        figure = Figure(figsize=(13, 7), constrained_layout=True)
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(2, 3, height_ratios=(1, 2))
        ax = figure.add_subplot(grid[0, :])
        s = arclength(x)/1000.
        ax.plot(s, r, label='Measured graph (includes flagged fills)')
        ax.plot(s, profile, '--', label='Reconstruction profile')
        trusted = trusted_radii(measured, sid)
        ax.scatter(s[trusted], r[trusted], s=12, color='black', label='Trusted section')
        ax.set(xlabel='Arclength (mm)', ylabel='Radius (um)', title=f'Segment {sid}')
        ax.legend(fontsize=8)
        for slot, i in enumerate(np.rint(np.linspace(.15, .85, 3)*(len(x)-1)).astype(int)):
            ax = figure.add_subplot(grid[1, slot])
            section = cut(sampler, frame.um_to_seg(x[i])[0], tangent[i],
                          min(256, max(12, int(3*max(r[i], profile[i])/sp))), max_half=256)
            if section is None:
                ax.text(.5, .5, 'No segmentation section', ha='center')
                continue
            c = section
            half = c.half*sp
            ax.imshow(c.plane.T, origin='lower', cmap='gray',
                      extent=(-half, half, -half, half), interpolation='nearest')
            ax.plot(0, 0, '+', color='cyan', label='Measured centre')
            axes = _plane_axes(surface_tangent[i])
            if axes is not None:
                u, v = axes
                theta = np.linspace(0, 2*np.pi, 100)
                circle = y[i]+profile[i]*(np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v)-x[i]
                ax.plot(circle@c.u, circle@c.v, color='orange', linewidth=1,
                        label='Projected reconstruction circle')
            ax.set(title=f'Point {i}; trusted={bool(trusted[i])}', xlabel='Plane u (um)', ylabel='Plane v (um)')
            if c.touches_border:
                ax.set_title(f'Point {i}; truncated section')
        path = directory/f'segment_{sid}_overlay.png'
        figure.savefig(path, dpi=150)
        outputs.append(path.name)
    return outputs
