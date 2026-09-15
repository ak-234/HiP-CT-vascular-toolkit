"""Ground truth for the sub-floor radii: how does the pipeline's estimator read a
vessel one to three voxels across?

`crosssection.MIN_BLOB_VOXELS = 12` refuses any section smaller than 12 voxels --
at the LADAF-2024-28 spacing of 65.98 um an equivalent radius of 129 um, about two
voxels -- and 14.6% of the tree's points carry a radius below that. Lowering the
floor is only worth doing if what comes back from the smaller sections is a
measurement rather than a digitisation artefact, and that is decidable on a
cylinder of known radius even though the real tree has no answer.

The estimator under test is the pipeline's own: `cv2.arcLength` around the
**4-connected** blob holding the plane centre, `r = perimeter / 2pi`
(`crosssection._perimeter_um`, reached from `radius_perimeter` line 1162).
`oblique_synthetic.py` tests a *different* estimator -- `skeleton_analysis` uses
skimage's `regionprops` perimeter -- so it cannot answer this question.

Two errors run in opposite directions at this size and neither is negligible:

* `cv2` traces pixel **centres**, so the contour of a disc is inset by roughly half
  a voxel on each side and the perimeter is short by ~2pi*0.5 -- a whole voxel of
  radius, which at r = 2 is 50%;
* the traced path is a staircase, which for a smooth boundary is long.

Which one wins at 1-3 voxels is the measurement, not a thing to assume.

Part 1 isolates the estimator on rasterised discs over a grid of sub-voxel offsets
(the axis of a real vessel does not pass through voxel centres, and at this size the
offset changes the digitisation completely). Part 2 runs the real
`radius_perimeter.measure_radii` on synthetic tubes at each candidate floor, so the
number includes tangent estimation, plane resampling and window growth.

Run: python research_scripts/subvoxel_bias.py
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from hipct_seg_debug.crosssection import _perimeter_um
from hipct_seg_debug.edit import radius_perimeter as rp
from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.tests.conftest_geometry import SPACING, cylinder, make_frame

SPACING_UM = 65.98  # LADAF-2024-28 segmentation spacing
FLOORS = (4, 6, 8, 12)  # MIN_BLOB_VOXELS candidates from the handoff
OFFSETS = 24  # sub-voxel offset grid is OFFSETS x OFFSETS


def rasterise_disc(radius_vox: float, dy: float, dz: float, half: int) -> np.ndarray:
    """A digitised disc of the given radius, its centre offset from a voxel centre.

    Nearest-neighbour on the voxel lattice, which is what `_PlaneSampler.plane`
    does when it cuts a tube perpendicular to its own axis.
    """
    ax = np.arange(-half, half + 1, dtype=float)
    yy, zz = np.meshgrid(ax, ax, indexing="ij")
    return (((yy - dy) ** 2 + (zz - dz) ** 2) <= radius_vox * radius_vox).astype(np.uint8)


def centre_blob(plane: np.ndarray) -> np.ndarray | None:
    """The 4-connected component holding the plane centre; None if that voxel is
    background, which is exactly when `crosssection.cut` returns None."""
    half = plane.shape[0] // 2
    if not plane[half, half]:
        return None
    lab, _ = ndimage.label(plane)
    return lab == lab[half, half]


def estimator_bias(radius_vox: float) -> dict:
    """Both estimators over a grid of sub-voxel offsets, at one true radius."""
    half = int(np.ceil(radius_vox)) + 4
    grid = (np.arange(OFFSETS) + 0.5) / OFFSETS - 0.5  # (-0.5, 0.5), no centre bias
    n_vox, r_perim, r_area = [], [], []
    refused = 0
    for dy in grid:
        for dz in grid:
            blob = centre_blob(rasterise_disc(radius_vox, dy, dz, half))
            if blob is None:
                refused += 1
                continue
            area = float(blob.sum())
            n_vox.append(area)
            r_perim.append(_perimeter_um(blob, 1.0) / (2.0 * np.pi))
            r_area.append(np.sqrt(area / np.pi))
    return {
        "r_true": radius_vox,
        "n": np.asarray(n_vox),
        "perim": np.asarray(r_perim),
        "area": np.asarray(r_area),
        "refused": refused,
        "total": OFFSETS * OFFSETS,
    }


def med(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if x.size else float("nan")


def part1() -> list[dict]:
    print("PART 1 - the estimator alone, on rasterised discs")
    print(f"         {OFFSETS * OFFSETS} sub-voxel offsets per radius; "
          f"ratios are estimate / true radius\n")
    print(f"    {'r true':>7}{'r true':>8}  {'voxels':>16}   "
          f"{'perim/true':>21}   {'area/true':>10}")
    print(f"    {'(vox)':>7}{'(um)':>8}  {'median':>7}{'p5-p95':>9}   "
          f"{'median':>7}{'p5':>7}{'p95':>7}   {'median':>10}")
    rows = []
    for r in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        row = estimator_bias(r)
        rows.append(row)
        n, p, a = row["n"], row["perim"] / r, row["area"] / r
        if n.size == 0:
            print(f"    {r:>7.2f}{r * SPACING_UM:>8.0f}   every offset refused")
            continue
        n5, n95 = np.percentile(n, [5, 95])
        p5, p50, p95 = np.percentile(p, [5, 50, 95])
        print(f"    {r:>7.2f}{r * SPACING_UM:>8.0f}  {med(n):>7.0f}{f'{n5:.0f}-{n95:.0f}':>9}   "
              f"{p50:>7.3f}{p5:>7.3f}{p95:>7.3f}   {med(a):>10.3f}")
    return rows


def part1_floors(rows: list[dict]) -> None:
    """What each candidate floor admits, and what the admitted sections then read.

    Conditioning on the floor is itself a bias: it keeps the offsets whose
    digitisation came out large and drops the ones that came out small, so the
    accepted population reads high even where the full population does not.
    """
    print("\nPART 1b - per candidate MIN_BLOB_VOXELS: fraction of sections admitted,")
    print("          and the median perim/true of the ones that are\n")
    head = "".join(f"{f'floor {f}':>18}" for f in FLOORS)
    print(f"    {'r true (vox)':>13}{head}")
    print(f"    {'':>13}" + "".join(f"{'admit':>9}{'ratio':>9}" for _ in FLOORS))
    for row in rows:
        cells = ""
        for f in FLOORS:
            keep = row["n"] >= f
            admit = keep.sum() / row["total"]  # refusals count against the floor
            ratio = med(row["perim"][keep] / row["r_true"]) if keep.any() else float("nan")
            cells += f"{admit:>8.0%} " + (f"{ratio:>8.3f} " if keep.any() else f"{'--':>8} ")
        print(f"    {row['r_true']:>13.2f}{cells}")


# ------------------------------------------------------------------- the model

def model(r: float) -> float:
    """`r_perim` predicted from first principles, for a round section.

    `cv2` walks the *centres* of the boundary pixels, so it traces a circle of
    radius ``r - 0.5`` rather than ``r``; and it walks that circle as a chain code,
    whose length exceeds a smooth path's by ``mean(cos t + (sqrt(2) - 1) sin t)``
    over ``t`` in ``[0, 45)`` degrees, which is 1.0548. The two errors have opposite
    signs, so there is a radius at which they cancel and the estimator looks exact.
    """
    return 1.0548 * (r - 0.5)


CROSSOVER = 0.5 * 1.0548 / 0.0548  # voxels; below it the estimator under-reads


def part1_model(rows: list[dict]) -> None:
    print("\nPART 1c - measured against the closed-form model 1.0548 * (r - 0.5)\n")
    print(f"    {'r true (vox)':>13}{'measured':>10}{'model':>9}{'residual':>10}")
    for row in rows:
        if row["perim"].size == 0:
            continue
        m = med(row["perim"])
        print(f"    {row['r_true']:>13.2f}{m:>10.3f}{model(row['r_true']):>9.3f}"
              f"{m - model(row['r_true']):>10.3f}")
    print(f"\n    The model changes sign at r = {CROSSOVER:.1f} voxels = "
          f"{CROSSOVER * SPACING_UM:.0f} um.")
    print("    Below that the perimeter estimator under-reads; above it, over-reads.")


# --------------------------------------------------------- the collapsed section

def ellipse_axes(r_perim_vox: float, aspect: float) -> tuple[float, float]:
    """Semi-axes of the ellipse of the given aspect ratio whose perimeter is
    ``2 pi r_perim_vox`` -- whose true answer, under the design assumption, is
    exactly ``r_perim_vox``. Ramanujan's approximation; error under 1e-5 at 10:1."""
    a, b = aspect, 1.0
    p1 = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    s = 2 * np.pi * r_perim_vox / p1
    return a * s, b * s


def rasterise_ellipse(a, b, phi, dy, dz, half) -> np.ndarray:
    """A digitised ellipse, rotated by `phi` and offset from the voxel centre."""
    ax = np.arange(-half, half + 1, dtype=float)
    yy, zz = np.meshgrid(ax, ax, indexing="ij")
    y, z = yy - dy, zz - dz
    yr = y * np.cos(phi) + z * np.sin(phi)
    zr = -y * np.sin(phi) + z * np.cos(phi)
    return ((yr / a) ** 2 + (zr / b) ** 2 <= 1.0).astype(np.uint8)


def part2() -> list[dict]:
    """A collapsed lumen is why the perimeter estimator was chosen at all.

    The two estimators answer different questions here, so each is scored against
    *its own* target: perimeter against ``P/2pi``, which is what the pipeline wants
    written out, and area against ``sqrt(ab/pi)``. Their ratio is reported
    separately, because that gap is the model choice and not an estimator error.
    """
    print("\n\nPART 2 - collapsed sections: ellipses of a known perimeter.")
    print("         Orientation and sub-voxel offset are both swept. Each estimator")
    print("         is scored against its own target, so this is estimator accuracy;")
    print("         the last column is the model gap between them.\n")
    print(f"    {'target':>7}{'aspect':>8}{'voxels':>8}   {'perim / (P/2pi)':>22}"
          f"   {'area / sqrt(ab/pi)':>18}{'r_area/r_perim':>16}")
    print(f"    {'r (vox)':>7}{'a:b':>8}{'median':>8}   {'median':>8}{'p5':>7}{'p95':>7}"
          f"   {'median':>18}{'median':>16}")
    rows = []
    offs = (np.arange(8) + 0.5) / 8 - 0.5
    phis = np.linspace(0.0, np.pi / 2, 12, endpoint=False)
    for r_p in (1.5, 2.0, 3.0, 5.0):
        for aspect in (1.0, 2.0, 4.0, 8.0):
            a, b = ellipse_axes(r_p, aspect)
            r_a_true = np.sqrt(a * b)  # area is pi*a*b, so sqrt(area/pi) = sqrt(a*b)
            half = int(np.ceil(a)) + 4
            nv, pr, ar = [], [], []
            for phi in phis:
                for dy in offs:
                    for dz in offs:
                        blob = centre_blob(rasterise_ellipse(a, b, phi, dy, dz, half))
                        if blob is None:
                            continue
                        area = float(blob.sum())
                        nv.append(area)
                        pr.append(_perimeter_um(blob, 1.0) / (2.0 * np.pi))
                        ar.append(np.sqrt(area / np.pi))
            if not pr:
                continue
            pr, ar = np.asarray(pr), np.asarray(ar)
            rows.append({"r_true": r_p, "aspect": aspect, "perim": pr, "area": ar})
            q5, q50, q95 = np.percentile(pr / r_p, [5, 50, 95])
            # A blob of one or two voxels traces a degenerate contour of zero
            # length, so the model gap is undefined for it rather than infinite.
            gap = med(ar[pr > 0] / pr[pr > 0])
            print(f"    {r_p:>7.1f}{aspect:>7.0f}:1{med(nv):>8.0f}   "
                  f"{q50:>8.3f}{q5:>7.3f}{q95:>7.3f}   "
                  f"{med(ar / r_a_true):>18.3f}{gap:>16.3f}")
        print()
    return rows


# ---------------------------------------------------------------- the pipeline

SHAPE = (40, 40, 80)


def offset_axis_graph(frame, x0, x1, radius_um, cy, cz):
    """`conftest_geometry.axis_graph`, but the row and slice may be fractional --
    the point being tested is that a vessel axis does not lie on voxel centres."""
    xs = np.arange(x0, x1, dtype=float)
    ijk = np.stack([xs, np.full_like(xs, cy), np.full_like(xs, cz)], axis=1)
    xyz = frame.seg_to_um(ijk)
    points = {i: (float(p[0]), float(p[1]), float(p[2]), float(radius_um))
              for i, p in enumerate(xyz)}
    nodes = {0: (*xyz[0], 0), 1: (*xyz[-1], 0)}
    segments = [{"id": 0, "node1": 0, "node2": 1, "point_ids": list(range(len(xs)))}]
    return EditableGraph(Triple(nodes, points, segments))


def pipeline_case(frame, r_vox, off, floor, gate):
    """One `measure_radii` run on a straight tube of known radius."""
    cy = cz = 20 + off
    mask = cylinder(SHAPE, r_vox, 5, 75, cy=cy, cz=cz)
    graph = offset_axis_graph(frame, 8, 72, r_vox * SPACING, cy, cz)
    res = rp.measure_radii(graph, frame, mask, min_blob_voxels=floor, gate_voxels=gate)
    measured = res.source[0] != rp.FILLED
    vals = res.radii[0][measured] / SPACING  # voxels
    return measured.mean(), (med(vals) / r_vox if measured.any() else float("nan"))


PIPE_OFFSETS = (0.0, 0.13, 0.27, 0.41, 0.5, 0.68)


def part3() -> None:
    """The same question asked of the real pass rather than of the estimator alone.

    The offsets are swept for the same reason as in part 1, and it matters more here
    than it looks: at ``off = 0`` a tube of integer radius is the lattice-aligned
    special case, whose boundary runs straight along the axes at the four cardinal
    points and whose perimeter therefore comes out high. Reporting that one offset
    would say the estimator is unbiased at r >= 3, which the swept median denies.
    """
    print("\nPART 3 - the whole pass. `radius_perimeter.measure_radii` on the same")
    print("         tubes, so tangent estimation, plane resampling and window growth")
    print("         are included. The centreline is placed *on* the tube axis, so")
    print(f"         this is digitisation alone, not a centring error. {len(PIPE_OFFSETS)}")
    print("         sub-voxel axis positions per row; `ratio` is their median and")
    print("         `range` their spread. The last block is `gate_voxels=99`: the")
    print("         area estimator everywhere, which is what GATE_VOXELS used to do.\n")
    frame = make_frame(SHAPE)
    print(f"    {'r true':>7}   " +
          "".join(f"{f'floor {f}':>26}" for f in FLOORS) + f"{'area estimator':>26}")
    print(f"    {'(vox)':>7}   " +
          "".join(f"{'cover':>8}{'ratio':>8}{'range':>10}" for _ in FLOORS) +
          f"{'cover':>8}{'ratio':>8}{'range':>10}")
    for r_vox in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        cells = ""
        for floor, gate in [(f, 0.0) for f in FLOORS] + [(4, 99.0)]:
            runs = [pipeline_case(frame, r_vox, off, floor, gate) for off in PIPE_OFFSETS]
            cover = med([c for c, _ in runs])
            vals = [v for c, v in runs if c and np.isfinite(v)]
            if vals:
                cells += (f"{cover:>7.0%} {med(vals):>7.3f} "
                          f"{f'{min(vals):.2f}-{max(vals):.2f}':>9} ")
            else:
                cells += f"{cover:>7.0%} {'--':>7} {'--':>9} "
        print(f"    {r_vox:>7.2f}   {cells}")


# --------------------------------------------------------------- the correction

def correct(r_est_vox):
    """Invert :func:`model`: recover the true radius from what the estimator read.

    Both constants are derived rather than fitted -- 1.0548 is the chain-code
    integral and 0.5 is half a voxel of centre-tracing inset -- so applying this
    to the same measurements the model was stated from is not circular. It is
    nonetheless checked below on radii that appear nowhere in part 1.

    A section that traced no contour at all (one or two voxels) has nothing to
    correct, so it stays at zero rather than being lifted to half a voxel.
    """
    r = np.asarray(r_est_vox, dtype=float)
    return np.where(r > 0, r / 1.0548 + 0.5, 0.0)


HELD_OUT = (1.1, 1.4, 1.9, 2.3, 2.8, 3.6, 4.5, 7.0)


def part4(ellipse_rows: list[dict]) -> None:
    print("\n\nPART 4 - inverting the model. `r / 1.0548 + 0.5`, on radii that appear")
    print("         nowhere in part 1, so the check is held out from the statement.\n")
    print(f"    {'r true':>7}{'r true':>8}   {'raw perim/true':>22}   "
          f"{'corrected/true':>22}")
    print(f"    {'(vox)':>7}{'(um)':>8}   {'median':>8}{'p5':>7}{'p95':>7}   "
          f"{'median':>8}{'p5':>7}{'p95':>7}")
    for r in HELD_OUT:
        row = estimator_bias(r)
        if row["perim"].size == 0:
            continue
        raw = np.percentile(row["perim"] / r, [5, 50, 95])
        cor = np.percentile(correct(row["perim"]) / r, [5, 50, 95])
        print(f"    {r:>7.2f}{r * SPACING_UM:>8.0f}   "
              f"{raw[1]:>8.3f}{raw[0]:>7.3f}{raw[2]:>7.3f}   "
              f"{cor[1]:>8.3f}{cor[0]:>7.3f}{cor[2]:>7.3f}")

    print("\n    The model was derived for a circle. On the collapsed sections of")
    print("    part 2 it is being extrapolated, so it is checked there too:\n")
    print(f"    {'target':>7}{'aspect':>8}   {'raw':>8}{'corrected':>11}")
    for row in ellipse_rows:
        r_p, pr = row["r_true"], row["perim"]
        print(f"    {r_p:>7.1f}{row['aspect']:>7.0f}:1   "
              f"{med(pr / r_p):>8.3f}{med(correct(pr) / r_p):>11.3f}")


def part4_pipeline() -> None:
    """The correction where it would actually be applied: after `measure_radii`.

    Applied in micrometres, so the half-voxel inset is half a *spacing*.
    """
    print("\n    And through the whole pass, at the floor that admits the thin end:\n")
    frame = make_frame(SHAPE)
    print(f"    {'r true':>7}   {'raw':>8}{'corrected':>11}{'area est.':>11}")
    for r_vox in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        raw, cor, area = [], [], []
        for off in PIPE_OFFSETS:
            cov, ratio = pipeline_case(frame, r_vox, off, 4, 0.0)
            if cov and np.isfinite(ratio):
                raw.append(ratio)
                # `ratio` is r_est/r_true in voxels, so the correction applies to it
                # directly once it is scaled back up by the true radius.
                cor.append(float(correct(ratio * r_vox)) / r_vox)
            cov, ratio = pipeline_case(frame, r_vox, off, 4, 99.0)
            if cov and np.isfinite(ratio):
                area.append(ratio)
        print(f"    {r_vox:>7.2f}   {med(raw):>8.3f}{med(cor):>11.3f}{med(area):>11.3f}")


# ------------------------------------------------------------ severe collapse

def part5() -> None:
    """The case the perimeter estimator was chosen for, pushed until it breaks.

    The design assumption is that a collapsed lumen's *perimeter* survives fixation
    even though its shape does not, so perimeter is the honest measure of a slit.
    That is an argument about the vessel. It says nothing about whether a digitised
    slit still *has* a measurable perimeter, and at one voxel of thickness it does
    not: `cv2` traces pixel centres, and the centres of a one-voxel-thick blob form
    a line of zero enclosed width. The contour goes out and back, so the length it
    reports is the slit's *length*, not its circumference.

    Two separate failures are reported because they have different fixes:

    * `admitted` -- `MIN_BLOB_VOXELS` counts **voxels**, i.e. area, while the
      quantity being measured is perimeter. A collapsed lumen has small area and
      large perimeter by construction, so an area floor refuses exactly the
      sections perimeter exists to handle.
    * `pinched` -- 4-connectivity splits a diagonally-pinched slit into pieces, so
      `blob4` is a fragment of the lumen. This repository already tested that and
      found it did not matter, but the test was run over *accepted* sections; a
      fragment small enough to be refused never entered that population, so the
      regime below is the one the earlier test could not see.
    """
    print("\n\nPART 5 - severe collapse: how flat before the estimator stops working")
    print("         Ellipses of a known perimeter, orientation and offset swept, taken")
    print("         to thicknesses at and below one voxel. `admitted` is the fraction")
    print("         that clears MIN_BLOB_VOXELS = 12; the ratios are over the admitted")
    print("         ones only, which flatters them.\n")
    print(f"    {'target':>7}{'aspect':>8}{'thick':>8}{'voxels':>8}{'admit':>8}"
          f"{'pinched':>9}{'lost':>7}   {'raw':>7}{'corrected':>11}{'area':>8}")
    print(f"    {'r (vox)':>7}{'a:b':>8}{'(vox)':>8}{'median':>8}{'@12':>8}"
          f"{'blob4':>9}{'centre':>7}   {'ratio':>7}{'ratio':>11}{'ratio':>8}")
    offs = (np.arange(6) + 0.5) / 6 - 0.5
    phis = np.linspace(0.0, np.pi / 2, 12, endpoint=False)
    for r_p in (3.0, 5.0, 8.0):
        for aspect in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            a, b = ellipse_axes(r_p, aspect)
            half = int(np.ceil(a)) + 4
            nv, pr, ar, pinched, lost, total = [], [], [], 0, 0, 0
            for phi in phis:
                for dy in offs:
                    for dz in offs:
                        total += 1
                        plane = rasterise_ellipse(a, b, phi, dy, dz, half)
                        blob = centre_blob(plane)
                        if blob is None:
                            lost += 1  # the centre voxel is background: cut() returns None
                            continue
                        lab8, _ = ndimage.label(plane, structure=np.ones((3, 3), int))
                        blob8 = lab8 == lab8[half, half]
                        if blob8.sum() > blob.sum():
                            pinched += 1
                        area = float(blob.sum())
                        nv.append(area)
                        pr.append(_perimeter_um(blob, 1.0) / (2.0 * np.pi))
                        ar.append(np.sqrt(area / np.pi))
            if not nv:
                print(f"    {r_p:>7.1f}{aspect:>7.0f}:1{2 * b:>8.2f}"
                      f"{'--':>8}{0.0:>8.0%}{'--':>9}{lost / total:>7.0%}"
                      f"   {'--':>7}{'--':>11}{'--':>8}")
                continue
            nv, pr, ar = np.asarray(nv), np.asarray(pr), np.asarray(ar)
            keep = nv >= 12
            admit = keep.sum() / total
            cells = (f"{med(pr[keep]) / r_p:>7.3f}{med(correct(pr[keep])) / r_p:>11.3f}"
                     f"{med(ar[keep]) / r_p:>8.3f}") if keep.any() else \
                    f"{'--':>7}{'--':>11}{'--':>8}"
            print(f"    {r_p:>7.1f}{aspect:>7.0f}:1{2 * b:>8.2f}{med(nv):>8.0f}"
                  f"{admit:>8.0%}{pinched / total:>9.0%}{lost / total:>7.0%}   {cells}")
        print()


def main() -> int:
    rows = part1()
    part1_floors(rows)
    part1_model(rows)
    ellipse_rows = part2()
    part3()
    part4(ellipse_rows)
    part4_pipeline()
    part5()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


