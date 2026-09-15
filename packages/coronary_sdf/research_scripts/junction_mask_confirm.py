"""Confirm which term of the junction mask is refusing these segments.

`_adaptive_junction_mask` masks points from each degree-3 end node inward until it
meets two consecutive *exclusive* sections, where

    exclusive = stable[i] and not adjacent_overlap[i]

and commits the whole segment if it never finds two in a row. `segment_diagnosis.py`
established that `stable` holds at 18 of 19 points on segment 265, which leaves
`adjacent_overlap` -- but that was read off the code rather than measured. This
measures it, by rebuilding the same `_BranchContext` the pass builds and asking it
the same question at the same points.

**The input graph, not the output.** The pass consumed `flagged_recentred.am`; the
radii in `radius_perim_corrected.am` are its result. `rivals()` scales its search by
the radius it is handed, so asking with the output radii would be asking a different
question than the pass asked.

Run: python research_scripts/junction_mask_confirm.py 265 268
"""
from __future__ import annotations

import sys

import numpy as np

import _paths

from hipct_seg_debug import amira, rle
from hipct_seg_debug.crosssection import (
    _PlaneSampler,
    robust_edge_tangents,
    stable_transverse_cut,
)
from hipct_seg_debug.edit import radius_perimeter as rp
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.frame import WorldFrame

INPUT_GRAPH = _paths.graph()
SEG = _paths.segmentation()


def open_lattice(path: str):
    info = amira.read_lattice_header(path)
    field = info.fields["Labels" if "Labels" in info.fields else next(iter(info.fields))]
    labels = rle.open_lattice(path, field, info.dims)
    raw_shape = tuple(int(v) for v in (info.dims[2], info.dims[1], info.dims[0]))
    return labels, WorldFrame.from_inputs(raw_shape, float(info.spacing[0]) / 2.0, info)


def main(argv: list[str]) -> int:
    sids = [int(a) for a in argv[1:]] or [265, 268]
    graph = EditableGraph(read_triple(INPUT_GRAPH))
    labels, frame = open_lattice(SEG)
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    ctx = rp._BranchContext.build(graph)

    for sid in sids:
        seg = graph.segment(sid)
        coords = graph.coords(sid)
        scale = graph.radii(sid)
        ijk = frame.um_to_seg(coords)
        tangents = robust_edge_tangents(coords, scale, spacing_um=sp)
        n = len(coords)
        print(f"--- segment {sid}: {n} points, input radii "
              f"{np.nanmin(scale):.0f}-{np.nanmax(scale):.0f} um; "
              f"end node degrees {graph.degree(seg['node1'])} and "
              f"{graph.degree(seg['node2'])}")
        print(f"    {'i':>3}{'stable':>8}{'adj_ovl':>9}{'exclusive':>11}"
              f"{'rivals':>8}   adjacent rival segments")
        stable = np.zeros(n, dtype=bool)
        adj = np.zeros(n, dtype=bool)
        for i in range(n):
            rp_vox = max(float(scale[i]) / sp, 1.0)
            chosen = stable_transverse_cut(
                sampler, ijk[i], tangents[i], rp_vox, spacing_um=sp, max_half=64,
                search_degrees=20.0, min_blob_voxels=12,
                slab_offsets=(0.0, 0.25, 0.5) if i == 0 else (
                    (-0.5, -0.25, 0.0) if i == n - 1 else (-0.5, 0.0, 0.5)),
            )
            c = chosen.cut if chosen is not None else None
            tangent = chosen.tangent if chosen is not None else tangents[i]
            stable[i] = c is not None and not c.touches_border
            rivals = ctx.rivals(sid, coords[i], tangent, float(scale[i]))
            adj[i] = any(item[4] for item in rivals)
            names = sorted({int(item[0]) for item in rivals if item[4]})
            excl = bool(stable[i] and not adj[i])
            print(f"    {i:>3}{str(bool(stable[i])):>8}{str(bool(adj[i])):>9}"
                  f"{str(excl):>11}{len(rivals):>8}   {names if names else '-'}")

        exclusive = stable & ~adj
        runs = 0
        best = 0
        for e in exclusive:
            runs = runs + 1 if e else 0
            best = max(best, runs)
        print(f"\n    stable {stable.sum()}/{n}   "
              f"adjacent_overlap {adj.sum()}/{n}   "
              f"exclusive {exclusive.sum()}/{n}")
        print(f"    longest run of consecutive exclusive sections: {best}")
        print(f"    the mask needs 2 to stop walking -> "
              f"{'STOPS, segment partly measured' if best >= 2 else 'NEVER STOPS, whole segment masked'}")

        mask, lengths = rp._adaptive_junction_mask(graph, sid, _arc(coords), stable, adj)
        print(f"    _adaptive_junction_mask masks {int(mask.sum())}/{n} points"
              f"  (runs {[f'{v:.0f} um' for v in lengths]})\n")
    return 0


def _arc(coords: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
