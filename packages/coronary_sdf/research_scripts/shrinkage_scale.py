"""Can the perimeter/area gap stand in for global shrinkage?

Ex-vivo vessels are smaller than in-vivo ones: no luminal pressure, and HiP-CT
specimens are dehydrated on top of that. `r_perimeter` is always the larger of the
two estimators -- by the isoperimetric inequality `Q = P^2/(4 pi A) >= 1` and
`r_perim / r_area = sqrt(Q)` exactly -- so it is tempting to read that uplift as
recovering the in-vivo calibre.

**The two are different kinds of error and only one of them is a scale.** Shrinkage
is approximately an isotropic factor: every length shrinks by the same k, so a
correction for it must be a single multiplier applied everywhere. The perimeter
uplift is `sqrt(Q)`, which is a function of *section shape* -- it is large where a
lumen is folded and 1.00 where it is round. Substituting one for the other only works
if `sqrt(Q)` is close to constant across the tree.

That is the test: not whether perimeter gives bigger numbers (it must), but whether
the amount it adds is uniform. A shape-driven uplift used as a shrinkage correction
would inflate flattened vessels and leave round ones untouched, which is not what
shrinkage does to a specimen.

Run: python research_scripts/shrinkage_scale.py
"""
from __future__ import annotations

import numpy as np

from collapse_conservation import GRAPH, open_lattice, sections, SEG
from hipct_seg_debug import reformat
from hipct_seg_debug.edit import radius_perimeter as rp_mod
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph

#: Approximate in-vivo adult human calibres, angiographic/IVUS, as *diameters* in mm.
#: Quoted as ranges because that is how they are reported and because this specimen's
#: sex, age and heart size are not being matched to any particular series. They are
#: here to give the ex-vivo numbers a scale to be judged against, not to calibrate to.
IN_VIVO_DIAMETER_MM = {
    "left main": (4.0, 5.0),
    "proximal LAD": (3.0, 3.9),
    "proximal LCx": (2.9, 3.5),
    "proximal RCA": (3.0, 4.0),
}


def main() -> int:
    g = EditableGraph(read_triple(GRAPH))
    labels, frame = open_lattice(SEG)
    sp = float(frame.seg_spacing[0])
    sampler = reformat.LabelSampler(labels, frame)

    rows = []
    for seg in g.segments:
        got = sections(sampler, g, seg["id"], sp)
        if got is None:
            continue
        rp_, ra_, q_ = got
        # The *corrected* perimeter. Raw, the uplift is dominated by the estimator's
        # own centre-tracing bias rather than by section shape -- which shows up as a
        # measured Q below 1.0 on small vessels, a value the isoperimetric inequality
        # forbids for any real shape. Correcting it puts both estimators on the same
        # footing so what is left is geometry.
        rp_ = np.array([rp_mod.correct_perimeter_radius(v, sp) for v in rp_])
        ok = rp_ > 0
        if ok.sum() < 4:
            continue
        rp_, ra_ = rp_[ok], ra_[ok]
        q_ = (rp_ / ra_) ** 2
        rows.append((seg["id"], float(np.median(rp_)), float(np.median(ra_)),
                     float(np.median(q_)), len(rp_)))
    rp = np.array([r[1] for r in rows])
    ra = np.array([r[2] for r in rows])
    q = np.array([r[3] for r in rows])
    uplift = rp / ra
    print(f"{len(rows)} segments\n")

    print("Is the perimeter uplift a constant, the way a shrinkage factor must be?")
    print(f"  r_perimeter / r_area over segments: "
          f"p5 {np.percentile(uplift,5):.3f}  median {np.median(uplift):.3f}  "
          f"p95 {np.percentile(uplift,95):.3f}")
    print(f"  spread p95/p5 = {np.percentile(uplift,95)/np.percentile(uplift,5):.2f}x"
          f"   (1.00x would be a pure scale)\n")

    print("  by vessel size, largest first -- a scale factor must not trend here:")
    order = np.argsort(-ra)
    print(f"    {'decile':>8}{'n':>5}{'r_area um':>12}{'uplift':>9}{'Q':>7}")
    for d in range(10):
        sel = order[d * len(order) // 10:(d + 1) * len(order) // 10]
        if len(sel) == 0:
            continue
        print(f"    {d+1:>8}{len(sel):>5}{np.median(ra[sel]):>12.0f}"
              f"{np.median(uplift[sel]):>9.3f}{np.median(q[sel]):>7.2f}")

    print("\nThe largest vessels against in-vivo literature (diameters, mm):")
    print(f"    {'sid':>5}{'r_area':>9}{'r_perim':>9}{'d_area':>9}{'d_perim':>9}{'Q':>7}")
    for r in sorted(rows, key=lambda r: -r[2])[:8]:
        print(f"    {r[0]:>5}{r[2]:>9.0f}{r[1]:>9.0f}"
              f"{2*r[2]/1000:>9.2f}{2*r[1]/1000:>9.2f}{r[3]:>7.2f}")
    print()
    for name, (lo, hi) in IN_VIVO_DIAMETER_MM.items():
        print(f"    {name:<14} in vivo {lo:.1f}-{hi:.1f} mm diameter")

    big = sorted(rows, key=lambda r: -r[2])[:5]
    d_area = np.median([2 * r[2] / 1000 for r in big])
    d_per = np.median([2 * r[1] / 1000 for r in big])
    print(f"\n  the five largest segments measure {d_area:.2f} mm (area) and "
          f"{d_per:.2f} mm (perimeter) in diameter")
    for name, (lo, hi) in (("proximal LAD", IN_VIVO_DIAMETER_MM["proximal LAD"]),
                           ("left main", IN_VIVO_DIAMETER_MM["left main"])):
        mid = 0.5 * (lo + hi)
        print(f"    to reach the middle of {name} ({mid:.1f} mm) they would need "
              f"x{mid/d_area:.2f} (area) or x{mid/d_per:.2f} (perimeter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
