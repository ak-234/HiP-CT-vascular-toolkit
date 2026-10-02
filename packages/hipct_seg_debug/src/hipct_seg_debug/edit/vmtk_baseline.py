"""Optional, explicitly seeded VMTK baseline; no automatic surface smoothing."""
from __future__ import annotations

import numpy as np


def extract(volume, spacing, origin, source_points, target_points):
    try:
        from vmtk import vmtkscripts
    except ImportError as exc:
        raise ImportError('VMTK comparison unavailable: install VMTK in a compatible environment; '
                          'this backend is optional and has not run here') from exc
    from skimage.measure import marching_cubes
    import pyvista as pv
    from .skeletonisers import edges_to_triple

    sources, targets = np.asarray(source_points, dtype=float), np.asarray(target_points, dtype=float)
    if (sources.ndim != 2 or targets.ndim != 2 or sources.shape[1:] != (3,)
            or targets.shape[1:] != (3,) or not len(sources) or not len(targets)
            or not np.isfinite(sources).all() or not np.isfinite(targets).all()):
        raise ValueError('VMTK needs explicit finite source and target XYZ coordinates in world micrometres')
    mask = np.asarray(volume) > 0
    if not mask.any() or mask.size > 2_000_000:
        raise ValueError('VMTK baseline requires a nonempty ROI of at most 2000000 voxels')
    from scipy.ndimage import label
    if label(mask, np.ones((3, 3, 3)))[1] != 1:
        raise ValueError('VMTK baseline requires one connected component')
    spacing, origin = np.asarray(spacing), np.asarray(origin)
    for seed in np.vstack([sources, targets]):
        ijk = (seed-origin)/spacing
        if np.any(ijk < -.5) or np.any(ijk > np.asarray(mask.shape[::-1])-.5):
            raise ValueError('VMTK seed is outside the selected ROI')
    vertices, faces, _, _ = marching_cubes(np.pad(mask.astype(np.float32), 1), .5)
    vertices = (vertices[:, ::-1]-1)*spacing + origin
    surface = pv.PolyData(vertices, np.c_[np.full(len(faces), 3), faces].ravel())
    script = vmtkscripts.vmtkCenterlines()
    script.Surface = surface
    script.SeedSelectorName = 'pointlist'
    script.SourcePoints = sources.ravel().tolist()
    script.TargetPoints = targets.ravel().tolist()
    script.AppendEndPoints = 0
    script.Resampling = 0
    script.Execute()
    lines = pv.wrap(script.Centerlines).clean(tolerance=0, absolute=True)
    if 'MaximumInscribedSphereRadius' not in lines.point_data:
        raise ValueError('VMTK did not return maximal-inscribed-sphere radii')
    cells, edges, i = np.asarray(lines.lines), [], 0
    while i < len(cells):
        count = int(cells[i])
        ids = cells[i+1:i+1+count]
        edges.extend(zip(ids[:-1], ids[1:]))
        i += count+1
    if not edges:
        raise ValueError('VMTK did not return any centreline edges')
    triple = edges_to_triple(lines.points, edges, lines.point_data['MaximumInscribedSphereRadius'])
    return triple, ('VMTK comparison; explicit seeds snap to nearest surface vertices; '
                    'new IDs and inscribed-sphere radii; requires containment/centring review and remeasurement')
