"""Read-only section ownership diagnostics; never writes corrected graph data.

Run with ``python -m hipct_seg_debug.edit.section_debug --help``.
Alternate slab widths are sensitivity experiments, not validated measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..crosssection import _PlaneSampler, robust_edge_tangents, stable_transverse_cut
from .section_validation import SectionContext


def inspect_section(graph, sid, point, frame, sampler, context, half_span=.5):
    """Trace every completed candidate slab and the final ownership decision."""
    x, radius = graph.coords(sid), graph.radii(sid)
    sp = float(frame.seg_spacing[0])
    normal = robust_edge_tangents(x, radius, spacing_um=sp)[point]
    events, diagnostics = [], {}
    chosen = stable_transverse_cut(
        sampler, frame.um_to_seg(x[point])[0], normal, radius[point]/sp,
        spacing_um=sp, max_half=256, centroid_mode='drift', transverse_axis_ratio=np.inf,
        slab_offsets=(-half_span, 0., half_span),
        validator=context.validator(sid, normal, sampler, frame, diagnostics, trace=events.append),
        diagnostics=diagnostics)
    result = dict(segment=int(sid), point=int(point), radius_um=float(radius[point]),
                  slab_half_span_radii=half_span, accepted=chosen is not None,
                  diagnostics=diagnostics, candidate_slabs=events,
                  status='diagnostic_only')
    if chosen is not None:
        result['selected'] = dict(normal=chosen.tangent.tolist(), owned=bool(chosen.owned),
                                  perimeter_um=float(chosen.perimeter_um),
                                  area_vox=int(chosen.cut.blob8.sum()),
                                  centroid_ratio=float(chosen.centroid_ratio))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--seg', required=True)
    parser.add_argument('--segment', type=int, action='append', required=True)
    parser.add_argument('--point', type=int, action='append',
                        help='point index in each segment; default samples its three quartiles')
    parser.add_argument('--slab-half-span', type=float, default=.5,
                        help='parallel check-plane distance in input radii; diagnostic sensitivity only')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.slab_half_span) or args.slab_half_span <= 0:
        parser.error('--slab-half-span must be positive and finite')
    from .__main__ import _load, _open_lattice, _correct_units
    from .junction_qualification import fingerprint, write_json
    options = SimpleNamespace(graph=[args.graph], seg=args.seg, labels_field='Labels',
                              voxel_um=None, edits=None)
    graph = _load(options.graph)
    for sid in args.segment:
        if not graph.has_segment(sid):
            parser.error(f'unknown segment {sid}')
        if args.point and any(i < 0 or i >= len(graph.coords(sid)) for i in args.point):
            parser.error(f'point index outside segment {sid}')
    labels, frame, _ = _open_lattice(options)
    _correct_units(graph, frame, options)
    sampler, context = _PlaneSampler(labels, frame), SectionContext(graph)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(options=vars(args), graph=fingerprint(args.graph), segmentation=fingerprint(args.seg))
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('diagnostic inputs changed; use a new output directory')
    write_json(manifest_path, manifest)
    rows = []
    for sid in sorted(set(args.segment)):
        n = len(graph.coords(sid))
        points = args.point if args.point is not None else [n//4, n//2, 3*n//4]
        for point in sorted(set(points)):
            row = inspect_section(graph, sid, point, frame, sampler, context, args.slab_half_span)
            rows.append(row)
            write_json(out/'sections.json', rows)
            print(json.dumps(dict(segment=sid, point=point, accepted=row['accepted'],
                                  diagnostics=row['diagnostics'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
