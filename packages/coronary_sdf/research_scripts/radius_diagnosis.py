"""Step 0 diagnosis: confirm the cut-plane-orientation mechanism before changing code.

Read-only. Compares the two tangent estimators actually used by the two radius
measurement paths, on the real spatial graph:

* one-step difference ``centreline[k] - centreline[k+1]``  -- what
  ``skeleton_analysis.outlier.oblique.segment_radii_from_volume`` uses;
* ``crosssection.robust_edge_tangents``                    -- what
  ``hipct_seg_debug.edit.radius_perimeter.measure_radii`` uses.

An oblique cut plane slices an ellipse whose perimeter exceeds the true
section's by ~1/cos(tilt), and ``r = perimeter / 2pi``, so tangent error
converts directly into radius error. This quantifies that conversion rather
than assuming it.
"""
from __future__ import annotations

from pathlib import Path

import _paths

import numpy as np

from coronary_sdf.parse_amira import parse_xml
from hipct_seg_debug.crosssection import robust_edge_tangents

GRAPH = _paths.graph()
UM_PER_MM = 1000.0


def _swing_deg(t: np.ndarray) -> np.ndarray:
    """Angle between consecutive unit tangents, degrees."""
    if len(t) < 2:
        return np.zeros(0)
    d = np.einsum("ij,ij->i", t[:-1], t[1:])
    return np.degrees(np.arccos(np.clip(np.abs(d), -1.0, 1.0)))


def _onestep(coords: np.ndarray) -> np.ndarray:
    """The oblique.py estimator: raw difference to the next point."""
    n = len(coords)
    t = np.zeros((n, 3))
    d = coords[1:] - coords[:-1]
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, nrm, out=np.zeros_like(d), where=nrm > 1e-12)
    t[:-1] = d
    t[-1] = d[-1] if len(d) else np.array([0.0, 0.0, 1.0])
    return t


def main() -> int:
    nodes, points, segments = parse_xml(GRAPH)
    print(f"graph: {len(nodes)} nodes, {len(points)} points, {len(segments)} segments\n")

    sw_one, sw_rob, radii_all, strahler_all = [], [], [], []
    for seg in segments:
        pids = [p for p in seg["point_ids"] if p in points]
        if len(pids) < 3:
            continue
        arr = np.asarray([points[p] for p in pids], dtype=np.float64)
        c_um = arr[:, :3]
        r_um = arr[:, 3]
        # robust_edge_tangents works in the graph's own units (um here).
        t_rob = robust_edge_tangents(c_um, r_um, spacing_um=float(np.median(
            np.linalg.norm(np.diff(c_um, axis=0), axis=1))) or 1.0)
        t_one = _onestep(c_um)
        sw_one.append(_swing_deg(t_one))
        sw_rob.append(_swing_deg(t_rob))
        radii_all.append(r_um[:-1])
        strahler_all.append(np.full(len(r_um) - 1, int(seg.get("strahler", 0))))

    one = np.concatenate(sw_one)
    rob = np.concatenate(sw_rob)
    rad = np.concatenate(radii_all) / UM_PER_MM
    sth = np.concatenate(strahler_all)
    ok = np.isfinite(one) & np.isfinite(rob)
    one, rob, rad, sth = one[ok], rob[ok], rad[ok], sth[ok]

    def q(x):
        return np.percentile(x, [50, 90, 99])

    print(f"tangent swing between neighbours, degrees   (n={len(one):,})")
    print(f"{'estimator':<26}{'median':>9}{'p90':>9}{'p99':>9}")
    for name, x in (("one-step difference", one), ("robust_edge_tangents", rob)):
        a, b, c = q(x)
        print(f"{name:<26}{a:>9.2f}{b:>9.2f}{c:>9.2f}")

    # Tilt -> radius inflation. A plane tilted by theta from transverse cuts an
    # ellipse with semi-axes (r, r/cos theta); Ramanujan's perimeter gives the
    # inflation factor. Half the neighbour swing approximates the per-point tilt.
    def inflation(theta_deg):
        th = np.radians(np.asarray(theta_deg) * 0.5)
        a = 1.0 / np.maximum(np.cos(th), 1e-9)
        b = np.ones_like(a)
        h = ((a - b) ** 2) / ((a + b) ** 2)
        per = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        return per / (2 * np.pi)

    print(f"\nimplied radius inflation from tilt (perimeter/2pi, ellipse)")
    print(f"{'estimator':<26}{'median':>9}{'p90':>9}{'p99':>9}")
    for name, x in (("one-step difference", one), ("robust_edge_tangents", rob)):
        infl = (inflation(x) - 1.0) * 100.0
        a, b, c = np.percentile(infl, [50, 90, 99])
        print(f"{name:<26}{a:>8.2f}%{b:>8.2f}%{c:>8.2f}%")

    print(f"\nby Strahler order (median swing, degrees)")
    print(f"{'order':>6}{'n':>8}{'r_mm':>8}{'one-step':>11}{'robust':>9}")
    for o in sorted(set(sth.tolist())):
        m = sth == o
        if m.sum() < 20:
            continue
        print(f"{o:>6}{int(m.sum()):>8}{np.median(rad[m]):>8.3f}"
              f"{np.median(one[m]):>11.2f}{np.median(rob[m]):>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
