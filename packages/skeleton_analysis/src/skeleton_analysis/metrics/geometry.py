"""Per-edge segment geometry computed from the graph (not read from Amira).

These recompute length / tortuosity / volume / surface-area / radius from the
``EdgePointCoordinates`` + ``NumEdgePoints`` + ``thickness`` arrays, so they
respond to the collapsed-vessel corrections (Amira's stored ``CurvedLength`` /
``Tortuosity`` / ``Volume`` / ``MeanRadius`` fields are static and cannot be
recomputed after correction).

The definitions are the standard ones, matching VesselVio's
``feature_extraction.py`` (length = centreline arc-length, tortuosity =
length/chord, volume = pi*r^2*L, lateral surface area = 2*pi*r*L, radius =
mean/max/min/SD of the per-point radius). Coordinates are already in micrometres,
so ``resolution`` is 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from skeleton_analysis.io.amira import SpatialGraph


def _edge_slices(graph: SpatialGraph) -> Tuple[np.ndarray, np.ndarray]:
    nump = np.asarray(graph.num_edge_points, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(nump)[:-1]])
    return starts, nump


def segment_lengths(graph: SpatialGraph) -> np.ndarray:
    """Polyline arc-length of every edge (sum of consecutive point distances)."""
    pts = np.asarray(graph.point_coords, dtype=float)
    starts, nump = _edge_slices(graph)
    out = np.zeros(len(nump), dtype=float)
    for i, (s, n) in enumerate(zip(starts, nump)):
        if n < 2:
            continue
        seg = pts[s : s + n]
        out[i] = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    return out


def chord_lengths(graph: SpatialGraph) -> np.ndarray:
    """Straight-line distance between each edge's first and last point."""
    pts = np.asarray(graph.point_coords, dtype=float)
    starts, nump = _edge_slices(graph)
    out = np.zeros(len(nump), dtype=float)
    for i, (s, n) in enumerate(zip(starts, nump)):
        if n < 2:
            continue
        out[i] = float(np.linalg.norm(pts[s + n - 1] - pts[s]))
    return out


def tortuosity(graph: SpatialGraph, min_chord: float = 1e-9) -> np.ndarray:
    """Tortuosity = arc-length / chord-length per edge (0 for loops, chord~0)."""
    length = segment_lengths(graph)
    chord = chord_lengths(graph)
    out = np.zeros_like(length)
    ok = chord > min_chord
    out[ok] = length[ok] / chord[ok]
    return out


@dataclass
class RadiusStats:
    avg: np.ndarray
    max: np.ndarray
    min: np.ndarray
    sd: np.ndarray


def radius_stats(graph: SpatialGraph, thickness_field: str = "thickness") -> RadiusStats:
    """Per-edge mean/max/min/SD of the per-point radius (``thickness``)."""
    thick = np.asarray(graph.point_fields[thickness_field], dtype=float)
    starts, nump = _edge_slices(graph)
    n = len(nump)
    avg = np.full(n, np.nan); mx = np.full(n, np.nan)
    mn = np.full(n, np.nan); sd = np.full(n, np.nan)
    for i, (s, k) in enumerate(zip(starts, nump)):
        if k <= 0:
            continue
        seg = thick[s : s + k]
        avg[i] = float(np.mean(seg)); mx[i] = float(np.max(seg))
        mn[i] = float(np.min(seg)); sd[i] = float(np.std(seg))
    return RadiusStats(avg=avg, max=mx, min=mn, sd=sd)


def volumes(graph: SpatialGraph) -> np.ndarray:
    """Vessel volume per edge: pi * r_avg^2 * length."""
    r = radius_stats(graph).avg
    return np.pi * r ** 2 * segment_lengths(graph)


def surface_areas(graph: SpatialGraph) -> np.ndarray:
    """Lateral surface area per edge: 2 * pi * r_avg * length (no end caps)."""
    r = radius_stats(graph).avg
    return 2 * np.pi * r * segment_lengths(graph)
