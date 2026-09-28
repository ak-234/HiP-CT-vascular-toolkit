"""Exact observation reuse for immutable segmentation during one refinement run."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from .interpolation import mask_for_segment


class SectionObservationCache:
    """Invalidate for any changed curve whose tube could reach a sampled ROI.

    Bounds deliberately cover the largest allowed plane/ownership volume, all
    candidate orientations, slab offsets and voxel rounding. This is not a
    distance-based approximation of the section test: actual observations are
    reused only when every possible influencing curve and the target scale match.
    Topology, radii, fitting options and segmentation are fixed within a run.
    """

    def __init__(self, spacing, max_half):
        self.spacing, self.max_half = float(spacing), int(max_half)
        self.entries, self.pending = {}, {}

    def refresh(self, graph):
        self.graph = graph
        self.sids = sorted(graph.segment_ids())
        self.positions = {sid: i for i, sid in enumerate(self.sids)}
        self.lower, self.upper, self.signatures = [], [], []
        for sid in self.sids:
            x, r = graph.coords(sid), graph.radii(sid)
            h = hashlib.sha256(x.tobytes()+r.tobytes()).digest()
            self.signatures.append(h)
            finite_r = r[np.isfinite(r)]
            extent = max(0., float(np.max(finite_r))) if len(finite_r) else 0.
            finite_x = x[np.isfinite(x).all(axis=1)]
            self.lower.append(finite_x.min(axis=0)-extent if len(finite_x) else np.full(3, np.inf))
            self.upper.append(finite_x.max(axis=0)+extent if len(finite_x) else np.full(3, -np.inf))
        self.lower, self.upper = np.asarray(self.lower), np.asarray(self.upper)
        self.pending.clear()

    def get(self, sid, scale):
        x = self.graph.coords(sid)
        if not len(x) or not np.isfinite(x).all() or not np.isfinite(scale).all():
            return None
        margin = 2*self.max_half*self.spacing + float(np.max(scale))/2 + 2*self.spacing
        near = np.flatnonzero(np.all(self.upper >= x.min(axis=0)-margin, axis=1)
                              & np.all(self.lower <= x.max(axis=0)+margin, axis=1))
        h = hashlib.sha256(np.asarray(scale).tobytes())
        h.update(self.signatures[self.positions[sid]])
        h.update(np.asarray(mask_for_segment(self.graph, sid)).tobytes())
        h.update(near.tobytes())
        for i in near:
            h.update(self.signatures[i])
        key = h.digest()
        self.pending[sid] = key
        entry = self.entries.get(sid)
        if entry is not None and entry[0] == key:
            return copy.deepcopy(entry[1:])
        return None

    def put(self, sid, observations, diagnostics):
        if sid in self.pending:
            self.entries[sid] = (self.pending[sid], copy.deepcopy(observations), diagnostics.copy())
