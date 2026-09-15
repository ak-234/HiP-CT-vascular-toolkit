"""Is the perimeter better conserved than the area, from in vivo to ex vivo?

The design assumption behind `radius_perimeter` is that a lumen collapses when the
blood pressure goes: the wall folds, the enclosed *area* falls, but the wall's
*length* is preserved, so `perimeter/2pi` still reports the vessel's in-vivo calibre
while `sqrt(area/pi)` under-reports it. That assumption has been asserted throughout
and never tested.

**It cannot be tested by comparing the two estimators against collapse severity.**
With Q the isoperimetric ratio `P^2 / (4 pi A)`,

    r_area / r_perim = sqrt(A/pi) / (P/2pi) = 2 sqrt(pi A) / P = Q^(-1/2)

exactly, for every section, whatever its shape. Regressing one against the other on Q
recovers an algebraic identity and would "confirm" the hypothesis on random noise.
An external reference is required, and there are two that do not involve Q.

**Test A -- longitudinal smoothness.** A vessel's true calibre varies smoothly along
its own length: it tapers, and it steps down at branches. Collapse does not -- it is
set by local wall folding and varies section to section. So whichever estimator is
*smoother* along a segment is the one less contaminated by collapse. This is not the
identity above: it asks which quantity behaves like a vessel calibre, using
longitudinal structure that Q says nothing about.

**Test B -- Murray's law.** `r_parent^3 = sum(r_daughter^3)` is a statement about the
vessel *in vivo*, derived from flow and metabolic cost, and it is scale-invariant --
uniform shrinkage of the whole tree leaves it untouched. So it cannot detect
shrinkage, and does not claim to. What it detects is *heterogeneous* collapse: if one
branch at a junction is folded flat and another is round, an area-based radius breaks
the relation while a conserved perimeter keeps it. That is the discriminating case.

Sections come from `reformat` planes at `mode="native"`, one pixel per segmentation
voxel, so both estimators see the same digitisation.

Run: python research_scripts/collapse_conservation.py
"""
from __future__ import annotations

import numpy as np

import _paths
from scipy import ndimage

from hipct_seg_debug import amira, reformat, rle
from hipct_seg_debug.crosssection import _perimeter_um
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.frame import WorldFrame

GRAPH = _paths.graph()
SEG = _paths.segmentation()
SIZE_PX = 81
MIN_PLANES = 8


def open_lattice(path: str):
    info = amira.read_lattice_header(path)
    field = info.fields["Labels" if "Labels" in info.fields else next(iter(info.fields))]
    labels = rle.open_lattice(path, field, info.dims)
    raw_shape = tuple(int(v) for v in (info.dims[2], info.dims[1], info.dims[0]))
    return labels, WorldFrame.from_inputs(raw_shape, float(info.spacing[0]) / 2.0, info)


def sections(sampler, graph, sid, sp):
    """Per-plane (r_perimeter, r_area, isoperimetric Q) along one segment."""
    coords, radii = graph.coords(sid), graph.radii(sid)
    if len(coords) < 2:
        return None
    half_value = (SIZE_PX // 2) * sp
    try:
        cl = reformat.build_centreline(
            coords, radii, np.full(len(coords), sid, dtype=np.int64), step_um=sp,
            half_of=lambda r, _h=half_value: np.full(len(r), _h))
        geom = reformat.plane_geometry(cl, mode="native", size_px=SIZE_PX,
                                       voxel_um=sp, native_scale=1.0)
        planes = sampler.sample_planes(cl, geom)
    except Exception:
        return None
    rp_, ra_, q_ = [], [], []
    for k in range(len(planes)):
        pl = planes[k] > 0
        h = pl.shape[0] // 2
        if not pl[h, h]:
            continue
        lab4, _ = ndimage.label(pl)
        blob = lab4 == lab4[h, h]
        lab8, _ = ndimage.label(pl, structure=np.ones((3, 3), dtype=int))
        b8 = lab8 == lab8[h, h]
        if b8[0].any() or b8[-1].any() or b8[:, 0].any() or b8[:, -1].any():
            continue
        pitch = float(geom.px_um[k])
        area = float(blob.sum()) * pitch * pitch
        per = _perimeter_um(blob, pitch)
        if area <= 0 or per <= 0:
            continue
        rp_.append(per / (2.0 * np.pi))
        ra_.append(np.sqrt(area / np.pi))
        q_.append(per * per / (4.0 * np.pi * area))
    if len(rp_) < MIN_PLANES:
        return None
    return np.array(rp_), np.array(ra_), np.array(q_)


def roughness(values: np.ndarray) -> float:
    """Median absolute step in log radius between neighbouring planes.

    Log, so it is a fractional change and a 2 mm trunk and a 200 um branch are
    comparable. Median, so one genuine step at a branch does not set the score.
    """
    v = np.log(values[values > 0])
    return float(np.median(np.abs(np.diff(v)))) if len(v) > 2 else float("nan")


def main() -> int:
    g = EditableGraph(read_triple(GRAPH))
    labels, frame = open_lattice(SEG)
    sp = float(frame.seg_spacing[0])
    sampler = reformat.LabelSampler(labels, frame)

    per_seg = {}
    for seg in g.segments:
        got = sections(sampler, g, seg["id"], sp)
        if got is not None:
            per_seg[seg["id"]] = got
    print(f"{len(per_seg)} of {len(g.segments)} segments yielded >= {MIN_PLANES} "
          f"usable sections\n")

    q_all = np.concatenate([v[2] for v in per_seg.values()])
    print(f"isoperimetric ratio Q over {len(q_all):,} sections: "
          f"p50 {np.median(q_all):.2f}  p90 {np.percentile(q_all, 90):.2f}  "
          f"p99 {np.percentile(q_all, 99):.2f}   (1.00 = a circle)")
    print(f"  sections with Q > 1.5 (clearly non-circular): "
          f"{(q_all > 1.5).mean():.1%}\n")

    print("TEST A - which estimator varies more smoothly along its own vessel")
    rp_r = np.array([roughness(v[0]) for v in per_seg.values()])
    ra_r = np.array([roughness(v[1]) for v in per_seg.values()])
    ok = np.isfinite(rp_r) & np.isfinite(ra_r)
    rp_r, ra_r = rp_r[ok], ra_r[ok]
    print(f"  median |step| in log radius, per segment ({ok.sum()} segments):")
    print(f"    r_perimeter  {np.median(rp_r):.4f}")
    print(f"    r_area       {np.median(ra_r):.4f}")
    win = (rp_r < ra_r).mean()
    print(f"  perimeter smoother in {win:.0%} of segments"
          f"   (50% = no difference)")

    # Restricted to the segments that actually contain collapsed sections: if the
    # mechanism is collapse, the gap has to widen where collapse is present.
    collapsed = np.array([np.mean(v[2] > 1.5) for v in per_seg.values()])[ok]
    for lo, hi, tag in ((0.0, 0.05, "almost no collapsed sections"),
                        (0.05, 0.25, "some"),
                        (0.25, 1.01, "mostly collapsed")):
        sel = (collapsed >= lo) & (collapsed < hi)
        if sel.sum() < 5:
            continue
        print(f"    {tag:<28} n={sel.sum():>3}  perimeter smoother in "
              f"{(rp_r[sel] < ra_r[sel]).mean():>4.0%}   "
              f"({np.median(rp_r[sel]):.4f} vs {np.median(ra_r[sel]):.4f})")

    print("\nTEST B - Murray's law at every degree-3 node")
    rows = []
    for nid in g.nodes:
        inc = [s for s in g.node_segments(nid) if s in per_seg]
        if g.degree(nid) != 3 or len(inc) != 3:
            continue
        vals = {}
        for est in (0, 1):
            r = sorted((float(np.median(per_seg[s][est])) for s in inc), reverse=True)
            parent, d1, d2 = r[0], r[1], r[2]
            vals[est] = (d1 ** 3 + d2 ** 3) / parent ** 3
        rows.append((nid, vals[0], vals[1]))
    if rows:
        mp = np.array([r[1] for r in rows])
        ma = np.array([r[2] for r in rows])
        print(f"  {len(rows)} junctions; sum(daughter^3)/parent^3, 1.00 = Murray exactly")
        for tag, v in (("r_perimeter", mp), ("r_area", ma)):
            print(f"    {tag:<12} median {np.median(v):.3f}   "
                  f"median |log| {np.median(np.abs(np.log(v))):.3f}   "
                  f"within 25%: {(np.abs(np.log(v)) < np.log(1.25)).mean():.0%}")
        better = np.abs(np.log(mp)) < np.abs(np.log(ma))
        print(f"  perimeter closer to Murray at {better.mean():.0%} of junctions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
