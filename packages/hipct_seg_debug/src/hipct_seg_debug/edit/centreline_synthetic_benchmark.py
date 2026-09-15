"""Compare centreline methods on round, flattened and curved analytical lumens.

Run with ``python -m hipct_seg_debug.edit.centreline_synthetic_benchmark --out results.json``.
The reported position error is transverse to the phantom's x axis; it is not a
closest-curve distance or an independent validation of real segmented anatomy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from ..amira import LatticeInfo
from ..crosssection import _PlaneSampler
from ..frame import WorldFrame
from .adapter import Triple
from .centreline_benchmark import edt_candidate
from .centreline_refine import bad_edges, refine
from .graphmodel import EditableGraph
from .skeleton_optimise import recentre
from .smoothers import smooth

METHODS = ("none", "legacy-gaussian", "legacy-multiscale", "edt-geodesic",
           "centroid-spline", "centroid-coherent", "laplacian", "taubin")
SHAPES = ("round", "flat", "curved-flat")
SPACING = 10.0


def fixture(kind, step):
    """One offset/noisy branch with fixed true endpoints and underestimated radii."""
    if kind not in SHAPES or step not in (1, 3):
        raise ValueError("expected a supported phantom shape and point step 1 or 3")
    shape = (48, 100, 120)
    dims = np.array(shape[::-1])
    bbox = np.zeros(6)
    bbox[1::2] = (dims-1)*SPACING
    info = LatticeInfo(path=None, dims=dims, bbox=bbox, fields={})
    frame = WorldFrame.from_inputs(tuple(2*n for n in shape), SPACING/2, info)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    a, b = (5, 5) if kind == "round" else (10, 3)
    amplitude = 8 if kind == "curved-flat" else 0
    cy = 50+amplitude*np.sin((xx-5)*np.pi/110)
    mask = (((yy-cy)/a)**2+((zz-24)/b)**2 <= 1) & (xx >= 2) & (xx <= 117)
    x = np.arange(5, 116, step)
    y = 50+amplitude*np.sin((x-5)*np.pi/110)+3+.7*np.sin(x*1.7)
    y[[0, -1]] = 50+amplitude*np.sin((x[[0, -1]]-5)*np.pi/110)
    xyz = frame.seg_to_um(np.c_[x, y, np.full(len(x), 24.)])
    points = {i: (*map(float, p), 15.) for i, p in enumerate(xyz)}
    nodes = {0: (*xyz[0], 0), 1: (*xyz[-1], 0)}
    segments = [{"id": 0, "node1": 0, "node2": 1, "point_ids": list(points)}]
    graph = EditableGraph(Triple(nodes, points, segments))
    return graph, frame, mask.astype(np.uint8), amplitude


def score(x, amplitude):
    """Score away from fixed endpoints against the analytical centre and tangent."""
    use = (x[:, 0] > 25*SPACING) & (x[:, 0] < 95*SPACING)
    phase = (x[:, 0]/SPACING-5)*np.pi/110
    true_y = (50+amplitude*np.sin(phase))*SPACING
    error = np.hypot(x[:, 1]-true_y, x[:, 2]-24*SPACING)/SPACING
    tangent = np.gradient(x, axis=0)
    true_t = np.c_[np.ones(len(x)), amplitude*np.pi/110*np.cos(phase), np.zeros(len(x))]
    cosine = np.abs(np.sum(tangent*true_t, axis=1))/np.maximum(
        np.linalg.norm(tangent, axis=1)*np.linalg.norm(true_t, axis=1), 1e-12)
    angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
    return dict(median_error_vox=float(np.median(error[use])),
                p95_error_vox=float(np.percentile(error[use], 95)),
                median_tangent_deg=float(np.median(angle[use])))


def run(methods=METHODS):
    rows = []
    for kind in SHAPES:
        for step in (1, 3):
            original, frame, labels, amplitude = fixture(kind, step)
            for method in methods:
                graph = EditableGraph(copy.deepcopy(original.triple))
                start = time.monotonic()
                row = dict(shape=kind, point_step_vox=step, method=method)
                try:
                    detail = {}
                    if method.startswith("legacy-"):
                        origin = {0: graph.coords(0).copy()}
                        for _ in range(2):
                            recentre(graph, frame, labels, origin=origin)
                        smooth(method.removeprefix("legacy-"), graph)
                    elif method == "edt-geodesic":
                        graph.set_segment_coords(0, edt_candidate(graph, 0, frame, labels))
                    elif method != "none":
                        detail = refine(graph, frame, labels, method=method, strength=.01,
                                        max_iterations=25, max_samples=24).to_dict()
                    row.update(score(graph.coords(0), amplitude))
                    row.update(status="evaluated", converged=detail.get("converged"),
                               outside_edges=int(bad_edges(
                                   graph.coords(0), _PlaneSampler(labels, frame), frame).sum()))
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["seconds"] = time.monotonic()-start
                rows.append(row)
                print(kind, step, method, row.get("median_error_vox", row.get("error")), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    rows = run(args.methods)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return int(any(row["status"] == "failed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
