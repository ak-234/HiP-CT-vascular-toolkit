"""How much calibre does the junction mask cost, over the whole tree?

`junction_mask_confirm.py` shows *why* segments 265 and 268 are wholly masked;
`reformat_radius.py` shows what they should have read. This asks the same question of
every segment the mask consumed entirely -- 66 of 309 on LADAF-2024-28 -- and
measures each one on `reformat`'s parallel-transport planes.

**The input graph is not the answer to compare against.** Its radii are what this
whole pass exists to distrust: p99 1740 um, max 4348 um. A large written/input ratio
can as easily be a wrong input as a wrong output. So each segment is re-measured from
the segmentation instead, and the input is carried only as a third column.

Run: python research_scripts/junction_damage.py
"""
from __future__ import annotations

import numpy as np

import _paths
from scipy import ndimage

from hipct_seg_debug import amira, reformat, rle
from hipct_seg_debug.crosssection import _perimeter_um
from hipct_seg_debug.edit import radius_perimeter as rp
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.frame import WorldFrame

INPUT_GRAPH = _paths.graph()
OUTPUT_GRAPH = _paths.out_dir() / "junction_damage.am"
SEG = _paths.segmentation()
SIZE_PX = 81


def open_lattice(path: str):
    info = amira.read_lattice_header(path)
    field = info.fields["Labels" if "Labels" in info.fields else next(iter(info.fields))]
    labels = rle.open_lattice(path, field, info.dims)
    raw_shape = tuple(int(v) for v in (info.dims[2], info.dims[1], info.dims[0]))
    return labels, WorldFrame.from_inputs(raw_shape, float(info.spacing[0]) / 2.0, info)


def measure(sampler, graph, sid, sp):
    """Corrected perimeter radius per plane, on reformat's frames. NaN if unusable."""
    coords, radii = graph.coords(sid), graph.radii(sid)
    if len(coords) < 2:
        return np.zeros(0)
    half_value = (SIZE_PX // 2) * sp
    try:
        centreline = reformat.build_centreline(
            coords, radii, np.full(len(coords), sid, dtype=np.int64), step_um=sp,
            half_of=lambda r, _h=half_value: np.full(len(r), _h),
        )
        geom = reformat.plane_geometry(centreline, mode="native", size_px=SIZE_PX,
                                       voxel_um=sp, native_scale=1.0)
        planes = sampler.sample_planes(centreline, geom)
    except Exception:
        return np.zeros(0)
    out = []
    for k in range(len(planes)):
        pl = planes[k] > 0
        h = pl.shape[0] // 2
        if not pl[h, h]:
            continue
        lab4, _ = ndimage.label(pl)
        blob4 = lab4 == lab4[h, h]
        lab8, _ = ndimage.label(pl, structure=np.ones((3, 3), dtype=int))
        b8 = lab8 == lab8[h, h]
        if b8[0].any() or b8[-1].any() or b8[:, 0].any() or b8[:, -1].any():
            continue  # truncated by the frame; its perimeter is partly the frame
        pitch = float(geom.px_um[k])
        out.append(rp.correct_perimeter_radius(
            _perimeter_um(blob4, pitch) / (2.0 * np.pi), sp))
    return np.asarray(out)


def summarise(vals: np.ndarray) -> tuple[float, int, int]:
    """Median over planes that traced a contour, and how many did not.

    A one- or two-voxel blob closes a contour of zero length, so it reports 0 rather
    than a radius. Those are counted, not averaged in: a segment that is mostly
    zeros is genuinely below what the mask can resolve, which is a different finding
    from one the mask resolves and the pass discarded.
    """
    good = vals[vals > 0]
    return (float(np.median(good)) if len(good) else float("nan"),
            len(good), int((vals <= 0).sum()))


def main() -> int:
    out_triple = read_triple(OUTPUT_GRAPH)
    g = EditableGraph(out_triple)
    gin = EditableGraph(read_triple(INPUT_GRAPH))
    labels, frame = open_lattice(SEG)
    sp = float(frame.seg_spacing[0])
    sampler = reformat.LabelSampler(labels, frame)
    rej = out_triple.point_attrs["radius_reject_reason"]
    src = out_triple.point_attrs["radius_source"]

    targets = []
    for seg in g.segments:
        pids = seg["point_ids"]
        if not pids:
            continue
        if (all(int(src.get(p, 2)) == rp.FILLED for p in pids)
                and all(int(rej.get(p, -1)) == rp.JUNCTION for p in pids)):
            targets.append(seg["id"])
    print(f"{len(targets)} segments wholly masked as `junction`; re-measuring each\n")
    print(f"  {'sid':>5}{'pts':>5}{'len um':>9}{'planes':>8}{'written':>9}"
          f"{'measured':>10}{'input':>8}{'w/meas':>8}")

    rows = []
    for sid in targets:
        vals = measure(sampler, gin, sid, sp)
        m, n_good, n_zero = summarise(vals)
        if n_good < 3:
            print(f"  {sid:>5}{len(g.coords(sid)):>5}{'':>9}{n_good:>8}"
                  f"   only {n_good} plane(s) traced a contour, {n_zero} degenerate"
                  f" -- below what the mask resolves, not a mask-discard")
            continue
        w = float(np.median(g.radii(sid)))
        r_in = float(np.median(gin.radii(sid)))
        c = g.coords(sid)
        arc = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        rows.append((sid, len(c), arc, n_good, w, m, r_in, w / m))

    rows.sort(key=lambda r: r[7])
    for r in rows:
        print(f"  {r[0]:>5}{r[1]:>5}{r[2]:>9.0f}{r[3]:>8}{r[4]:>9.0f}"
              f"{r[5]:>10.0f}{r[6]:>8.0f}{r[7]:>8.2f}")

    ratio = np.array([r[7] for r in rows])
    arcs = np.array([r[2] for r in rows])
    print(f"\n  {len(rows)} segments measured, {arcs.sum() / 1000:.1f} mm")
    print(f"  written / measured   p5 {np.percentile(ratio, 5):.2f}   "
          f"median {np.median(ratio):.2f}   p95 {np.percentile(ratio, 95):.2f}")
    for thr in (0.5, 0.75, 1.5):
        sel = ratio < thr if thr < 1 else ratio > thr
        word = "under" if thr < 1 else "over"
        print(f"  {word} by more than {abs(1 - thr):.0%}: {sel.sum()} segments, "
              f"{arcs[sel].sum() / 1000:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
