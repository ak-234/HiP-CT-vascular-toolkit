"""Synthetic ground-truth check for the oblique cross-section radius fixes.

The real graph has no known answer, so the fixes are demonstrated on a cylinder
of known radius. Three cases, each isolating one defect:

1. straight tube          -- baseline accuracy, both estimators should agree;
2. jitter sweep           -- centreline noise at fixed true radius. The one-step
                             tangent tilts the cut plane and inflates
                             perimeter/2pi by ~1/cos(tilt); the robust tangent
                             should stay flat;
3. large radius           -- a tube wider than the historical fixed window,
                             which cannot be measured by a fixed window at all.

Run: python research_scripts/oblique_synthetic.py
"""
from __future__ import annotations

import numpy as np

from skeleton_analysis.outlier.oblique import (
    cross_section_radius,
    oblique_slice,
    segment_radii_from_volume,
)


def cylinder(radius_vox: float, length: int = 40, pad: float = 14.0):
    """Binary cylinder of the given radius along z; volume axes (z, y, x)."""
    half = int(np.ceil(radius_vox + pad))
    size = 2 * half + 1
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((yy - half) ** 2 + (xx - half) ** 2) <= radius_vox ** 2
    vol = np.repeat(disc[None, :, :].astype(float), length, axis=0)
    centre = np.array([[z, float(half), float(half)] for z in range(length)])
    return vol, centre, half


def legacy_profile(volume, centreline, half_size, res=1.0):
    """The pre-fix path: one-step-difference normal, fixed window, guess on miss."""
    p = len(centreline)
    out = np.full(p, np.nan)
    for k in range(p - 1):
        normal = centreline[k] - centreline[k + 1]
        if np.linalg.norm(normal) == 0:
            continue
        sl, _ = oblique_slice(volume, centreline[k], normal, half_size=half_size)
        out[k] = cross_section_radius(sl, res=res, threshold=0.5,
                                      method="perimeter", require_center=False)
    if p >= 2:
        out[-1] = out[-2]
    return out


def med(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def main() -> int:
    rng = np.random.default_rng(0)

    print("case 1 - straight tube, true radius 8.0 vox")
    vol, cen, half = cylinder(8.0)
    old = legacy_profile(vol, cen, half_size=20)
    new = segment_radii_from_volume(vol, cen, res=1.0, threshold=0.5,
                                    radii=np.full(len(cen), 8.0), max_half=64)
    print(f"    legacy {med(old):6.3f}   fixed {med(new):6.3f}   (truth 8.000)\n")

    print("case 2 - jitter sweep, true radius 8.0 vox (radius must not move)")
    print(f"    {'jitter':>8}{'legacy':>10}{'fixed':>10}{'legacy err':>12}{'fixed err':>11}")
    for amp in (0.0, 0.25, 0.5, 1.0, 1.5):
        vol, cen, half = cylinder(8.0)
        j = cen.copy()
        if amp > 0:
            j[:, 1:] += rng.normal(0.0, amp, size=(len(j), 2))
        old = legacy_profile(vol, j, half_size=20)
        new = segment_radii_from_volume(vol, j, res=1.0, threshold=0.5,
                                        radii=np.full(len(j), 8.0), max_half=64)
        o, n = med(old), med(new)
        print(f"    {amp:>8.2f}{o:>10.3f}{n:>10.3f}"
              f"{(o - 8.0) / 8.0 * 100:>11.1f}%{(n - 8.0) / 8.0 * 100:>10.1f}%")

    print("\ncase 3 - large radius vs the historical fixed window (half_size=20)")
    print(f"    {'true r':>8}{'legacy':>10}{'fixed':>10}{'legacy err':>12}{'fixed err':>11}")
    for r in (8.0, 16.0, 25.0, 40.0):
        vol, cen, half = cylinder(r)
        old = legacy_profile(vol, cen, half_size=20)
        rep = {}
        new = segment_radii_from_volume(vol, cen, res=1.0, threshold=0.5,
                                        radii=np.full(len(cen), r), max_half=128,
                                        report=rep)
        o, n = med(old), med(new)
        print(f"    {r:>8.1f}{o:>10.3f}{n:>10.3f}"
              f"{(o - r) / r * 100:>11.1f}%{(n - r) / r * 100:>10.1f}%"
              f"   clipped={rep.get('n_clipped', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
