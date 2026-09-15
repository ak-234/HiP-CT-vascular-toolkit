"""Can the watershed separate branches that *share a node*, not just ones that touch?

`measure_radii` already owns machinery for measuring a section when another vessel is
fused to it: `_resolve_owned_cut` cuts a local ROI, marks each branch with its own
centreline, runs a marker watershed, and keeps the territory belonging to this
segment. It is only ever handed rivals that are **not** topology-adjacent --

    connected_nonadjacent = [... if not adjacent_overlap[i] and not item[4] ...]

where `item[4]` is "this rival shares a node with me". So the one case where lumens
are guaranteed to be fused -- a bifurcation -- is the one case never resolved. Those
points are masked instead, and their radius interpolated.

That is defensible only if the watershed cannot do the job here. Near a carina the
two lumens genuinely are continuous, so there is no correct answer in the sense of a
boundary that exists in the tissue. But there is a well-posed one: the part of the
lumen nearer this branch's axis than any sibling's, which is what an ostium is
normally taken to mean, and what the watershed computes.

This probe asks the question without changing the pipeline: at every point of the
named segments, hand `_resolve_owned_cut` *all* the rivals including the adjacent
ones, and report whether it returns a section and what that section measures. If it
succeeds, the ostial flare becomes something measured rather than modelled.

Run: python research_scripts/owned_junction_probe.py 265 268 270
"""
from __future__ import annotations

import sys

import numpy as np

import _paths

from hipct_seg_debug import amira, rle
from hipct_seg_debug.crosssection import (
    _PlaneSampler,
    _perimeter_um,
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
    sids = [int(a) for a in argv[1:]] or [265, 268, 270]
    graph = EditableGraph(read_triple(INPUT_GRAPH))
    labels, frame = open_lattice(SEG)
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    ctx = rp._BranchContext.build(graph)
    coords_ijk = {s: frame.um_to_seg(graph.coords(s)) for s in graph.segment_ids()}

    for sid in sids:
        coords = graph.coords(sid)
        scale = graph.radii(sid)
        ijk = coords_ijk[sid]
        tangents = robust_edge_tangents(coords, scale, spacing_um=sp)
        n = len(coords)
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(coords, axis=0), axis=1))])
        seg = graph.segment(sid)
        print(f"--- segment {sid}: {n} points, {arc[-1]:.0f} um, end node degrees "
              f"{graph.degree(seg['node1'])} and {graph.degree(seg['node2'])}")
        print(f"    {'i':>3}{'arc um':>8}{'rivals':>8}{'adjacent':>26}"
              f"{'whole r':>9}{'owned r':>9}{'owned/whole':>12}")
        n_owned = 0
        for i in range(n):
            rp_vox = max(float(scale[i]) / sp, 1.0)
            chosen = stable_transverse_cut(
                sampler, ijk[i], tangents[i], rp_vox, spacing_um=sp, max_half=64,
                search_degrees=20.0, min_blob_voxels=12,
                slab_offsets=(0.0, 0.25, 0.5) if i == 0 else (
                    (-0.5, -0.25, 0.0) if i == n - 1 else (-0.5, 0.0, 0.5)),
            )
            if chosen is None or chosen.cut.touches_border:
                print(f"    {i:>3}{arc[i]:>8.0f}   no stable section")
                continue
            c, tangent = chosen.cut, chosen.tangent
            whole = _perimeter_um(c.blob4, sp) / (2.0 * np.pi)
            rivals = ctx.rivals(sid, coords[i], tangent, float(scale[i]))
            adj = sorted({int(x[0]) for x in rivals if x[4]})
            # Every rival, adjacent ones included -- the change this probe exists to
            # evaluate. `_resolve_owned_cut` is N-ary already, so a trifurcation is
            # not a special case: it is just more markers.
            allr = sorted({int(x[0]) for x in rivals})
            owned = rp._resolve_owned_cut(
                sampler, graph, coords_ijk, sid, allr, ijk[i], tangent,
                rp_vox, c.half, 64, sp,
            ) if allr else None
            if owned is None:
                print(f"    {i:>3}{arc[i]:>8.0f}{len(rivals):>8}{str(adj):>26}"
                      f"{whole:>9.0f}{'--':>9}{'watershed declined':>12}")
                continue
            n_owned += 1
            r_owned = _perimeter_um(owned.blob4, sp) / (2.0 * np.pi)
            print(f"    {i:>3}{arc[i]:>8.0f}{len(rivals):>8}{str(adj):>26}"
                  f"{whole:>9.0f}{r_owned:>9.0f}{r_owned / whole:>12.2f}")
        print(f"    watershed resolved {n_owned}/{n} points\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
