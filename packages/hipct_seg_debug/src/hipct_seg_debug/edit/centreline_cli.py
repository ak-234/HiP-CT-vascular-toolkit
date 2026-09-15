"""CLI integration for experimental centreline and reconstruction refinement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def add_parsers(sub, common, seg_common):
    from .centreline_refine import METHODS
    p = sub.add_parser("refine-centreline", parents=[common, seg_common],
                       help="fit the centreline to segmentation without changing topology")
    p.add_argument("--method", choices=METHODS, default="centroid-spline")
    p.add_argument("--segment", dest="segments", type=int, action="append", default=None)
    p.add_argument("--roots-json", default=None)
    p.add_argument("--fixed-node", dest="fixed_nodes", type=int, action="append", default=[])
    p.add_argument("--fixed-junctions", action="store_true")
    p.add_argument("--strength", type=float, default=.1)
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--max-half", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--geometry-only", action="store_true",
                   help="retain radii as unmeasured placeholders instead of remeasuring")
    p.add_argument("--report-json", default=None)
    p = sub.add_parser("prepare-reconstruction", parents=[common, seg_common],
                       help="optional smooth clearance deformation; preserves measured radii")
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--gap-um", type=float, default=0.)
    p.add_argument("--max-displacement-radii", type=float, default=.5)
    p.add_argument("--report-json", default=None)


def run(args):
    from .__main__ import _correct_units, _load, _open_lattice, _save
    from . import radius_perimeter as rp
    from .roots import root_nodes_for

    if args.out and Path(args.out).resolve() in {Path(p).resolve() for p in args.graph}:
        raise ValueError("write a separate output; the measurement graph must be preserved")
    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    stamp = _correct_units(graph, frame, args)
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    if args.command == "refine-centreline":
        from .centreline_refine import refine
        report = refine(
            graph, frame, labels, method=args.method, sids=args.segments,
            fixed_nodes=set(args.fixed_nodes) | set(root_nodes_for(graph, args, frame=frame)),
            move_junctions=not args.fixed_junctions, strength=args.strength,
            max_iterations=args.max_iterations, max_half=args.max_half,
            max_samples=args.max_samples, workers=args.workers,
            progress=lambda row: print(json.dumps(row), flush=True),
        ).to_dict()
        if report['moved_points']:
            if not args.geometry_only:
                result = rp.measure_radii(graph, frame, labels, workers=args.workers,
                                           max_half=args.max_half)
                rp.apply_radii(graph, result)
                report['radii_require_remeasurement'] = False
                report['radius_measurement'] = result.describe()
            else:
                # Every section on a changed segment can acquire a new direction.
                # Do not leave old measurements stamped accepted at new positions.
                for sid in graph.segment_ids():
                    if np.array_equal(graph.coords(sid), before[sid]):
                        continue
                    for name, value in (("radius_source", rp.FILLED),
                                        ("radius_reject_reason", rp.UNMEASURABLE),
                                        ("radius_resolution_mode", rp.INPUT_FALLBACK)):
                        attr = graph.triple.point_attrs.setdefault(name, {})
                        attr.update({pid: value for pid in graph.segment(sid)['point_ids']})
                        graph.triple.point_attr_dtypes[name] = "int"
    else:
        from .centreline_clearance import prepare
        report = prepare(graph, frame, labels, max_iterations=args.max_iterations,
                         gap_um=args.gap_um, max_displacement_radii=args.max_displacement_radii,
                         progress=lambda row: print(json.dumps(row), flush=True)).to_dict()
        report['measurement_graph'] = list(args.graph)
        report['radius_policy'] = 'transported unchanged from measurement graph; do not remeasure'
    moved = graph.triple.point_attrs.setdefault('centreline_displacement_um', {})
    graph.triple.point_attr_dtypes['centreline_displacement_um'] = 'float'
    for sid in graph.segment_ids():
        d = np.linalg.norm(graph.coords(sid)-before[sid], axis=1)
        moved.update(dict(zip(graph.segment(sid)['point_ids'], map(float, d))))
    print(json.dumps(report, indent=2), flush=True)
    _save(graph, args.out, args.graph, voxel_um=stamp)
    report_path = args.report_json or (str(args.out)+'.report.json' if args.out else None)
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Unresolved preflights produce a review artifact with a non-success exit status.
    return 2 if args.command == 'prepare-reconstruction' and report['status'] != 'capsule-clear-surface-unvalidated' else 0
