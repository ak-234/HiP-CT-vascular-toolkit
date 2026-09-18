"""Local shared-node fitting with exact outer anchors and objective descent."""
from __future__ import annotations

import numpy as np


def fit_objective(x, target, weights, scale, reference, spacing, strength):
    from .centreline_refine import arclength
    s = arclength(reference)
    if len(s) < 3 or np.any(np.diff(s) <= 1e-9):
        return float(np.sum((x-target)**2))
    step = np.diff(s)
    q = np.r_[step[0]/2, (step[:-1]+step[1:])/2, step[-1]/2]
    data = np.sum(np.maximum(weights, .005)*q*np.sum((x-target)**2, axis=1))
    velocity = np.diff(x, axis=0)/step[:, None]
    curvature = np.diff(velocity, axis=0)/q[1:-1, None]
    local = np.maximum(np.broadcast_to(scale, (len(x),)), 2*spacing)
    bending = np.sum(q[1:-1]*local[1:-1]**4*np.sum(curvature**2, axis=1))
    return float(data+strength*bending)


def refine_neighbourhoods(graph, node_targets, observations, scales, sampler, frame, strength):
    """Fit overlapping junction spans in one solve with fixed outer anchors."""
    from .junction_cluster import neighbourhood_clusters, fit_cluster
    reports = {}
    for nodes, spans in neighbourhood_clusters(
            graph, node_targets, observations, scales, float(frame.seg_spacing[0])):
        result = fit_cluster(graph, nodes, spans, observations, scales, sampler, frame, strength)
        for nid in nodes:
            reports[nid] = result
    return reports
