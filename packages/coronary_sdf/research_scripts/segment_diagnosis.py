"""Why did these particular points get the radius they got?

The tree-wide counters say how many points were refused and for what reason. They
cannot say whether *this* stretch of *this* branch is narrow because the vessel is
narrow or because the pass could not measure it. This prints, per point:

* what was written out -- radius, `radius_source`, `radius_reject_reason`,
  `radius_resolution_mode`, all read back from the .am rather than recomputed, so
  it is the pass's own verdict and not a reconstruction of it;
* what the section actually looks like -- re-cut here with the floor lowered to one
  voxel so that a section the pass refused is still described rather than vanishing.

The geometry columns are the ones that decide between the two stories:

* `vox` is `blob4`'s area, which is what `MIN_BLOB_VOXELS = 12` gates on;
* `thick` is the minor axis in voxels, from the blob's second moments. Below about
  two, tracing pixel centres has almost no enclosed width left to trace and the
  perimeter estimator fails regardless of the floor;
* `pinch` is `blob8` area over `blob4` area. Above 1.0, 4-connectivity has split the
  lumen and the pass measured a fragment.

Run: python research_scripts/segment_diagnosis.py 265 268 270
     python research_scripts/segment_diagnosis.py --graph <other.am> 265 268
"""
from __future__ import annotations

import sys

import numpy as np

import _paths

from hipct_seg_debug import amira, rle
from hipct_seg_debug.crosssection import (
    _PlaneSampler,
    _perimeter_um,
    cut,
    robust_edge_tangents,
    stable_transverse_cut,
)
from hipct_seg_debug.edit import radius_perimeter as rp
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.frame import WorldFrame

GRAPH = _paths.graph()
SEG = _paths.segmentation()


def open_lattice(path: str):
    """The same frame `edit/__main__._open_lattice` builds, without its argparse."""
    info = amira.read_lattice_header(path)
    field_name = "Labels" if "Labels" in info.fields else next(iter(info.fields))
    labels = rle.open_lattice(path, info.fields[field_name], info.dims)
    raw_shape = tuple(int(v) for v in (info.dims[2], info.dims[1], info.dims[0]))
    frame = WorldFrame.from_inputs(raw_shape, float(info.spacing[0]) / 2.0, info)
    return labels, frame


def blob_axes(blob: np.ndarray) -> tuple[float, float]:
    """Major and minor axis lengths in voxels, from the blob's second moments.

    For a uniformly filled ellipse of semi-axes a and b the covariance eigenvalues
    are a^2/4 and b^2/4, so the full axis lengths are 4*sqrt(lambda).
    """
    ij = np.argwhere(blob).astype(float)
    if len(ij) < 2:
        return float(len(ij)), float(len(ij))
    cov = np.cov((ij - ij.mean(axis=0)).T)
    ev = np.sort(np.linalg.eigvalsh(np.atleast_2d(cov)))
    return float(4.0 * np.sqrt(max(ev[-1], 0.0))), float(4.0 * np.sqrt(max(ev[0], 0.0)))


def main(argv: list[str]) -> int:
    args = argv[1:]
    graph_path = GRAPH
    if "--graph" in args:
        i = args.index("--graph")
        graph_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    sids = [int(a) for a in args] or [265, 268, 270]

    triple = read_triple(graph_path)
    graph = EditableGraph(triple)
    labels, frame = open_lattice(SEG)
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])
    attrs = triple.point_attrs
    src = attrs.get("radius_source", {})
    rej = attrs.get("radius_reject_reason", {})
    mode = attrs.get("radius_resolution_mode", {})
    print(f"graph {graph_path}")
    print(f"  spacing {sp:.2f} um; MIN_BLOB_VOXELS = 12 gates on the `vox` column\n")

    for sid in sids:
        seg = graph.segment(sid)
        pids = list(seg["point_ids"])
        pts = np.array([graph.points[p][:3] for p in pids], dtype=float)
        radii = np.array([graph.points[p][3] for p in pids], dtype=float)
        ijk = frame.um_to_seg(pts)
        tangents = robust_edge_tangents(pts, radii, spacing_um=sp)

        n_meas = sum(1 for p in pids if src.get(p, 2) != rp.FILLED)
        arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        d1, d2 = graph.degree(seg["node1"]), graph.degree(seg["node2"])
        print(f"--- segment {sid}: {len(pids)} points, "
              f"{n_meas} measured ({n_meas / len(pids):.0%}), "
              f"radius {radii.min():.0f}-{radii.max():.0f} um "
              f"(median {np.median(radii):.0f})")
        print(f"    length {arc:.0f} um; end node degrees {d1} and {d2}; "
              f"`open` = the section still touched the window edge at 4 radii")
        print(f"    {'i':>3}{'r um':>7}{'source':>26}{'reject':>26}"
              f"{'vox':>6}{'thick':>7}{'major':>7}{'pinch':>7}{'r_per':>7}{'r_area':>7}"
              f"{'open':>6}")
        for i, pid in enumerate(pids):
            s = int(src.get(pid, rp.FILLED))
            r_reason = int(rej.get(pid, rp.UNMEASURABLE))
            # Floor of one voxel: describe the section even where the pass refused it.
            # `grow_to` is bounded at a few radii deliberately. Left unbounded the
            # window doubles to `max_half` whenever the plane is not truly
            # perpendicular, and what it then encloses is a streak *along* the
            # vessel rather than a section across it -- areas in the thousands and
            # major axes of 100+ voxels are that artefact, not lumen.
            rp_vox = max(float(radii[i]) / sp, 1.0)
            c = cut(sampler, ijk[i], tangents[i], min(int(rp_vox * 2.5) + 2, 64),
                    max_half=64, grow_to=int(4.0 * rp_vox) + 2, min_blob_voxels=1)
            if c is None:
                geom = f"{'--':>6}{'--':>7}{'--':>7}{'--':>7}{'--':>7}{'--':>7}{'--':>6}"
            else:
                area = float(c.blob4.sum())
                major, minor = blob_axes(c.blob4)
                pinch = float(c.blob8.sum()) / area if area else float("nan")
                r_per = _perimeter_um(c.blob4, sp) / (2.0 * np.pi)
                geom = (f"{area:>6.0f}{minor:>7.2f}{major:>7.2f}{pinch:>7.2f}"
                        f"{r_per:>7.0f}{np.sqrt(area / np.pi) * sp:>7.0f}"
                        f"{('YES' if c.touches_border else '-'):>6}")
            print(f"    {i:>3}{radii[i]:>7.0f}{rp.SOURCE_NAMES.get(s, s):>26}"
                  f"{rp.REJECT_NAMES.get(r_reason, r_reason):>26}{geom}")
        # Did the pass have a usable section here at all? `_adaptive_junction_mask`
        # only masks a run it could not bracket with two exclusive *stable* cuts, so
        # "why junction" reduces to "why was nothing stable".
        n_stable = 0
        for i in range(len(pids)):
            rp_vox = max(float(radii[i]) / sp, 1.0)
            st = stable_transverse_cut(sampler, ijk[i], tangents[i], rp_vox,
                                       spacing_um=sp, max_half=64,
                                       search_degrees=20.0, min_blob_voxels=12)
            if st is not None and not st.cut.touches_border:
                n_stable += 1
        print(f"    stable_transverse_cut succeeds at {n_stable}/{len(pids)} points")

        # What the constant fill was built from: the measured radii of the segments
        # meeting this one at its two end nodes.
        for node in (seg["node1"], seg["node2"]):
            for other in graph.segments:
                if other["id"] == sid or node not in (other["node1"], other["node2"]):
                    continue
                r_other = np.array([graph.points[q][3] for q in other["point_ids"]])
                meas = sum(1 for q in other["point_ids"] if src.get(q, 2) != rp.FILLED)
                print(f"      node {node}: neighbour segment {other['id']:>4} "
                      f"median {np.median(r_other):>6.0f} um, "
                      f"{meas}/{len(other['point_ids'])} measured")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
