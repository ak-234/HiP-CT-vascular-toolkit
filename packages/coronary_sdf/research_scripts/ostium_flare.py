"""Does the measured section flare at an ostium, and is the flare round?

Two questions, and the second decides what the first is worth.

Approaching a junction the measured `perimeter/2pi` rises sharply -- on segment 265,
749 um mid-segment to 1374 um at the node. That is a real feature of the
segmentation, not an artefact: the lumen genuinely opens where the branches meet.

But a radius is isotropic and an ostium is not. If the section grows in both axes the
flare is round and a radius can carry it. If it grows in one axis only, the section is
*elongating* toward the sibling, and `perimeter/2pi` converts that elongated merge
into an equivalent circle -- which a capsule-or-sphere SDF then renders as a round
bulge, in every direction, including the ones the lumen does not open into.

So this measures both the flare and its shape, binned by distance to the node in
local radii, over every segment end that meets a degree-3-or-higher node.

Run: python research_scripts/ostium_flare.py
"""
from __future__ import annotations

import numpy as np

import _paths

from hipct_seg_debug import amira, rle
from hipct_seg_debug.crosssection import (
    _PlaneSampler,
    _perimeter_um,
    cut,
    robust_edge_tangents,
)
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.frame import WorldFrame

GRAPH = _paths.graph()
SEG = _paths.segmentation()
#: Bins of distance-to-node, in units of the segment's own local radius.
BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]


def open_lattice(path: str):
    info = amira.read_lattice_header(path)
    field = info.fields["Labels" if "Labels" in info.fields else next(iter(info.fields))]
    labels = rle.open_lattice(path, field, info.dims)
    raw_shape = tuple(int(v) for v in (info.dims[2], info.dims[1], info.dims[0]))
    return labels, WorldFrame.from_inputs(raw_shape, float(info.spacing[0]) / 2.0, info)


def axes(blob: np.ndarray) -> tuple[float, float]:
    ij = np.argwhere(blob).astype(float)
    if len(ij) < 2:
        return float(len(ij)), float(len(ij))
    cov = np.cov((ij - ij.mean(axis=0)).T)
    ev = np.sort(np.linalg.eigvalsh(np.atleast_2d(cov)))
    return float(4 * np.sqrt(max(ev[-1], 0.0))), float(4 * np.sqrt(max(ev[0], 0.0)))


def main() -> int:
    g = EditableGraph(read_triple(GRAPH))
    labels, frame = open_lattice(SEG)
    sampler = _PlaneSampler(labels, frame)
    sp = float(frame.seg_spacing[0])

    rows = []  # (d_over_r, r_per, r_area, major, minor)
    for seg in g.segments:
        sid = seg["id"]
        coords, radii = g.coords(sid), g.radii(sid)
        n = len(coords)
        if n < 4:
            continue
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(coords, axis=0), axis=1))])
        # Distance to the nearer *branched* end only. A degree-1 end is a terminus,
        # not an ostium, and including it would dilute the very effect under test.
        d1 = arc if g.degree(seg["node1"]) >= 3 else np.full(n, np.inf)
        d2 = (arc[-1] - arc) if g.degree(seg["node2"]) >= 3 else np.full(n, np.inf)
        dnode = np.minimum(d1, d2)
        if not np.isfinite(dnode).any():
            continue
        tangents = robust_edge_tangents(coords, radii, spacing_um=sp)
        ijk = frame.um_to_seg(coords)
        for i in range(n):
            if not np.isfinite(dnode[i]):
                continue
            r_vox = max(float(radii[i]) / sp, 1.0)
            c = cut(sampler, ijk[i], tangents[i], min(int(r_vox * 2.5) + 2, 64),
                    max_half=64, grow_to=int(4.0 * r_vox) + 2, min_blob_voxels=12)
            if c is None or c.touches_border:
                continue
            area = float(c.blob4.sum())
            major, minor = axes(c.blob4)
            if minor <= 0:
                continue
            rows.append((dnode[i] / max(float(radii[i]), 1e-9),
                         _perimeter_um(c.blob4, sp) / (2 * np.pi),
                         np.sqrt(area / np.pi) * sp, major, minor))
    a = np.asarray(rows)
    print(f"{len(a):,} sections on segment ends that meet a degree-3+ node\n")
    print(f"  {'distance to node':>18}{'n':>7}{'r_perim':>10}{'r_area':>9}"
          f"{'r_per/r_area':>14}{'major/minor':>13}")
    print(f"  {'(local radii)':>18}{'':>7}{'median um':>10}{'um':>9}"
          f"{'':>14}{'':>13}")
    far = None
    for lo, hi in BINS:
        sel = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if sel.sum() < 5:
            continue
        b = a[sel]
        aspect = np.median(b[:, 3] / b[:, 4])
        if far is None and lo >= 3.0:
            far = (np.median(b[:, 1]), aspect)
        label = f"{lo:.1f}-{hi:.1f}" if hi < 1e8 else f">{lo:.1f}"
        print(f"  {label:>18}{sel.sum():>7}{np.median(b[:, 1]):>10.0f}"
              f"{np.median(b[:, 2]):>9.0f}{np.median(b[:, 1] / b[:, 2]):>14.2f}"
              f"{aspect:>13.2f}")

    near = a[a[:, 0] < 0.5]
    if len(near) and far:
        print(f"\n  approaching the node, relative to sections >3 radii away:")
        print(f"    r_perimeter   x{np.median(near[:, 1]) / far[0]:.2f}")
        print(f"    major/minor   {far[1]:.2f} -> {np.median(near[:, 3] / near[:, 4]):.2f}")
        print("\n  A flare a radius can carry would grow both axes and leave the")
        print("  aspect ratio flat. A rising aspect ratio is the section elongating")
        print("  toward the neighbouring branch, which an isotropic radius renders")
        print("  as a bulge in every direction instead of an opening in one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
