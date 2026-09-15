"""Measure the perimeter radius on planes cut by `reformat`, not by `crosssection`.

`segment_diagnosis.py` says segments 265 and 268 carry about a third of their true
calibre, using `crosssection.cut` to re-measure. That shares its plane construction
with the pass being audited -- both take a per-point tangent and search a cone around
it -- so agreement between them is weaker evidence than it looks.

`reformat` builds planes a different way: resample the path to uniform arclength,
smooth it until the plane stack is provably collision-free, then carry a
parallel-transport frame along it. The normal at a point depends on the whole path
rather than on its immediate neighbours, so a tangent that is wrong locally cannot
tilt one plane on its own. If both constructions land on the same radius, the number
is a property of the segmentation rather than of either estimator.

**``mode="native"`` matters.** It pins one output pixel to one segmentation voxel, so
the digitisation is the same as `cut`'s one-voxel-pitch plane and the perimeter
estimator has the same staircase to trace. Any other mode resamples the mask onto a
finer or coarser grid and changes the estimator's bias along with it, which would
make the comparison meaningless.

Run: python research_scripts/reformat_radius.py 265 268
"""
from __future__ import annotations

import sys

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
OUTPUT_GRAPH = _paths.out_dir() / "reformat_radius.am"
SEG = _paths.segmentation()
SIZE_PX = 81  # odd; +-40 voxels is 2.6 mm, well past any coronary section here


def open_lattice(path: str):
    info = amira.read_lattice_header(path)
    field = info.fields["Labels" if "Labels" in info.fields else next(iter(info.fields))]
    labels = rle.open_lattice(path, field, info.dims)
    raw_shape = tuple(int(v) for v in (info.dims[2], info.dims[1], info.dims[0]))
    return labels, WorldFrame.from_inputs(raw_shape, float(info.spacing[0]) / 2.0, info)


def centre_component(plane: np.ndarray):
    """`crosssection.cut`'s selection: the 4-connected blob holding the centre pixel."""
    h = plane.shape[0] // 2
    if not plane[h, h]:
        return None, None
    lab4, _ = ndimage.label(plane)
    lab8, _ = ndimage.label(plane, structure=np.ones((3, 3), dtype=int))
    return lab4 == lab4[h, h], lab8 == lab8[h, h]


def main(argv: list[str]) -> int:
    sids = [int(a) for a in argv[1:]] or [265, 268]
    graph = EditableGraph(read_triple(INPUT_GRAPH))
    written = EditableGraph(read_triple(OUTPUT_GRAPH))
    labels, frame = open_lattice(SEG)
    sp = float(frame.seg_spacing[0])
    sampler = reformat.LabelSampler(labels, frame)

    for sid in sids:
        coords = graph.coords(sid)
        radii = graph.radii(sid)
        seg_ids = np.full(len(coords), sid, dtype=np.int64)
        half_value = (SIZE_PX // 2) * sp

        centreline = reformat.build_centreline(
            coords, radii, seg_ids,
            step_um=sp,  # one plane per voxel of arclength
            half_of=lambda r, _h=half_value: np.full(len(r), _h),
        )
        geom = reformat.plane_geometry(
            centreline, mode="native", size_px=SIZE_PX, voxel_um=sp, native_scale=1.0,
        )
        planes = sampler.sample_planes(centreline, geom)

        r_per, r_area, opened, lost, pinched = [], [], 0, 0, 0
        for k in range(len(planes)):
            blob4, blob8 = centre_component(planes[k] > 0)
            if blob4 is None:
                lost += 1
                continue
            if blob8[0].any() or blob8[-1].any() or blob8[:, 0].any() or blob8[:, -1].any():
                opened += 1
                continue  # truncated by the frame: its perimeter is partly the frame
            if blob8.sum() > blob4.sum():
                pinched += 1
            pitch = float(geom.px_um[k])
            r_per.append(_perimeter_um(blob4, pitch) / (2.0 * np.pi))
            r_area.append(np.sqrt(float(blob4.sum()) / np.pi) * pitch)

        w = written.radii(sid)
        print(f"--- segment {sid}")
        print(f"    {centreline.describe()}")
        print(f"    smoothing moved the centreline: median "
              f"{centreline.median_move_um:.1f} um, max {centreline.max_move_um:.1f} um"
              f"{'  ' + '; '.join(centreline.notes) if centreline.notes else ''}")
        print(f"    planes {len(planes)}: {len(r_per)} usable, {opened} truncated by "
              f"the frame, {lost} with the centre off the mask, {pinched} pinched")
        if not r_per:
            print("    nothing measurable\n")
            continue
        r_per = np.asarray(r_per)
        r_area = np.asarray(r_area)
        corrected = np.asarray([rp.correct_perimeter_radius(v, sp) for v in r_per])
        q = lambda v: f"{np.percentile(v, 5):.0f} / {np.median(v):.0f} / {np.percentile(v, 95):.0f}"
        print(f"    reformat planes, p5 / median / p95:")
        print(f"      r_perimeter raw        {q(r_per)} um")
        print(f"      r_perimeter corrected  {q(corrected)} um")
        print(f"      r_area                 {q(r_area)} um")
        print(f"    for comparison:")
        print(f"      input graph  (flagged_recentred) median {np.median(radii):.0f} um")
        print(f"      written out  (radius_perim_corrected) median {np.median(w):.0f} um")
        print(f"      ratio written / reformat-corrected     "
              f"{np.median(w) / np.median(corrected):.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
