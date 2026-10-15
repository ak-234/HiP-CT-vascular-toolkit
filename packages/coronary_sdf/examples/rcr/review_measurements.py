"""Plot a selected measurement audit without treating interpolation as measurement.

python review_measurements.py --before source.am --measured measured_subset.am
  --segments 67 270 307 308 --out review.png
Requires the package's optional matplotlib dependency and hipct_seg_debug.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coronary_sdf.rcr.prepare import hydraulic_profile, UnsupportedHydraulicSpan
from hipct_seg_debug.edit.__main__ import _load
from hipct_seg_debug.edit.radius_profile import trusted_radii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True)
    parser.add_argument('--measured', required=True)
    parser.add_argument('--segments', type=int, nargs='+', required=True)
    parser.add_argument('--spacing-um', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    before, measured = _load([args.before]), _load([args.measured])
    fig, axes = plt.subplots(len(args.segments), 1, figsize=(10, 2.8*len(args.segments)), squeeze=False)
    report = []
    for ax, sid in zip(axes[:, 0], args.segments):
        seg = measured.segment(sid)
        xyz = measured.coords(sid)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        r = measured.radii(sid)
        trusted = trusted_radii(measured, sid)
        modes = measured.triple.point_attrs['radius_resolution_mode']
        trusted &= np.array([modes.get(p) in (0, 1) for p in seg['point_ids']])
        ax.plot(s/1000, before.radii(sid), color='#666666', lw=1.5, label='Original radius')
        ax.plot(s/1000, r, color='#d89020', ls='--', lw=1, label='Estimator output (includes fills)')
        ax.scatter(s[trusted]/1000, r[trusted], color='#087e8b', s=18, zorder=3, label='Trusted perimeter measurement')
        row = dict(segment=sid, samples=len(r), trusted=int(trusted.sum()))
        try:
            hydraulic, _ = hydraulic_profile(s, r, trusted,
                (measured.degree(seg['node1']) >= 3, measured.degree(seg['node2']) >= 3), args.spacing_um)
            ax.plot(s/1000, hydraulic, color='#087e8b', lw=.8, alpha=.5, label='Supported hydraulic profile')
            row['hydraulic_status'] = 'supported'
        except UnsupportedHydraulicSpan as exc:
            row.update(hydraulic_status='unsupported', spans=exc.spans)
            for i, span in enumerate(exc.spans):
                ax.axvspan(span['start_um']/1000, span['end_um']/1000, color='#bc3446', alpha=.14,
                           label='Unresolved hydraulic span' if i == 0 else None)
        ax.set_title(f"Source segment {sid}: {row['trusted']}/{row['samples']} trusted; {row['hydraulic_status']}", loc='left')
        ax.set(xlabel='Arclength from stored segment start (mm)', ylabel='Radius (µm)')
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc='best')
        report.append(row)
    fig.suptitle('LADAF measurement audit — fills are not direct measurements', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
