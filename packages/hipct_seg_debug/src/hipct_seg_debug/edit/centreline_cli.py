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
    p.add_argument("--root-node", type=int, action="append", default=[])
    p.add_argument("--fixed-node", dest="fixed_nodes", type=int, action="append", default=[])
    p.add_argument("--fixed-junctions", action="store_true")
    p.add_argument("--strength", type=float, default=.1)
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--max-half", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument('--no-section-cache', action='store_true',
                   help='disable exact section reuse for benchmarking')
    p.add_argument("--geometry-only", action="store_true",
                   help="retain radii as unmeasured placeholders instead of remeasuring")
    p.add_argument("--report-json", default=None)
    p = sub.add_parser("prepare-reconstruction", parents=[common, seg_common],
                       help="optional smooth clearance deformation; preserves measured radii")
    p.add_argument("--max-iterations", type=int, default=30)
    p.add_argument("--gap-um", type=float, default=0.)
    p.add_argument("--max-displacement-radii", type=float, default=.5)
    p.add_argument("--radius-profile", choices=("preserve", "confidence", "dfs-confidence"), default="preserve")
    p.add_argument("--transition-radii", type=float, default=4.,
                   help="maximum cross-segment interpolation span in local radius units (minimum eight voxels)")
    p.add_argument("--path-plan-json", default=None,
                   help="refinement report whose original DFS ordering should be retained")
    p.add_argument("--roots-json", default=None)
    p.add_argument("--root-node", type=int, action="append", default=[])
    p.add_argument("--skip-clearance", action="store_true",
                   help="derive a radius profile without displacing centrelines")
    p.add_argument("--report-json", default=None)
    p = sub.add_parser("simplify-skeleton", parents=[common, seg_common],
                       help="remove short leaves, then collapse short inner links into "
                            "one junction (PMC10182136, Fig. 4); a dry run unless --apply")
    p.add_argument("--segment", dest="segments", type=int, action="append", default=None,
                   help="limit link collapse to this segment's junction neighbourhood; "
                        "leaf pruning is tree-wide and is skipped")
    p.add_argument("--no-leaf-prune", action="store_true", help="skip phase one")
    p.add_argument("--no-contained-prune", action="store_true",
                   help="keep leaves that lie inside another vessel's lumen "
                        "(medial-sheet spurs of flattened lumens)")
    p.add_argument("--contained-nearest-host", action="store_true",
                   help="a leaf point counts as contained only if inside the nearest "
                        "vessel's section (default: any nearby vessel's)")
    p.add_argument("--root-node", dest="root_nodes", type=int, action="append", default=[],
                   help="never remove a leaf ending at this node; repeat for several")
    p.add_argument("--leaf-factor", type=float, default=None,
                   help="prune_spurs length factor (parent radii)")
    p.add_argument("--leaf-min-um", type=float, default=None,
                   help="prune_spurs absolute leaf tolerance, the paper's user threshold")
    p.add_argument("--link-factor", type=float, default=1.0,
                   help="collapse inner links shorter than this many parent radii; 0 disables")
    p.add_argument("--min-length-um", type=float, default=None,
                   help="also collapse inner links shorter than this (paper, manual)")
    p.add_argument("--auto-thinnest", action="store_true",
                   help="also collapse inner links shorter than the thinnest vessel's "
                        "diameter (paper, automatic)")
    p.add_argument("--apply", action="store_true", help="edit the graph and write --out")
    p.add_argument("--report-json", default=None)


def run_simplify(args):
    from .__main__ import _correct_units, _load, _open_lattice, _save
    from .junction_links import simplify_links
    from .junction_qualification import neighbourhood
    from .skeleton_optimise import PRUNE_LENGTH_FACTOR, prune_spurs

    if args.apply and not args.out:
        raise ValueError("--apply needs --out")
    if args.out and Path(args.out).resolve() in {Path(p).resolve() for p in args.graph}:
        raise ValueError("write a separate output; the input graph must be preserved")
    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    stamp = _correct_units(graph, frame, args)
    report = dict(method="PMC10182136 Fig. 4 phases 1-2", applied=bool(args.apply))
    region = None
    if args.segments:
        region = sorted({s for sid in args.segments for s in neighbourhood(graph, sid)})
        report["region_segments"] = region
    if args.no_leaf_prune or region is not None:
        report["leaves"] = dict(skipped="region" if region is not None else "requested")
    else:
        # prune_spurs edits in place; a dry run measures it on a copy.
        target = graph if args.apply else _load(args.graph)
        if not args.apply:
            _correct_units(target, frame, args)
        leaves = prune_spurs(target, length_factor=args.leaf_factor or PRUNE_LENGTH_FACTOR,
                             min_length_um=args.leaf_min_um, bbox_um=frame.seg_bbox_um)
        print(leaves.describe())
        report["leaves"] = dict(vars(leaves), describe=leaves.describe())
    if not args.no_contained_prune:
        from .contained_leaves import find_contained_leaves, prune_contained_leaves
        rows = find_contained_leaves(graph, labels, frame, sids=region,
                                     protected_nodes=args.root_nodes,
                                     any_host=not args.contained_nearest_host)
        flagged = [r for r in rows if r["contained"]]
        print(f"{len(flagged)} of {len(rows)} leaves lie inside another vessel's lumen"
              f"{'' if args.apply else ' (would be removed)'}: {[r['segment'] for r in flagged]}")
        report["contained_leaves"] = dict(rows=rows)
        if args.apply and flagged:
            report["contained_leaves"].update(prune_contained_leaves(graph, rows))
            if region is not None:
                region = [s for s in region if graph.has_segment(s)]
    links = simplify_links(graph, labels, frame, factor=args.link_factor or None,
                           min_length_um=args.min_length_um, auto_thinnest=args.auto_thinnest,
                           sids=region, apply=args.apply)
    print(links.describe())
    for row in links.clusters:
        print(f"  links {row['links']} nodes {row['nodes']}: {row['status']}"
              + (f" ({row['reason']})" if row.get('reason') else "")
              + (f", moved {row['move_from_p_ca_um']:.0f} of {row['ball_um']:.0f} um from p_CA"
                 + (" [at search edge: review]" if row['at_ball_edge'] else "")
                 if 'p_new' in row else ""))
    report["links"] = links.to_dict()
    if args.apply:
        from .junction_links import saved_ids
        report["renumbered"] = renumbered = saved_ids(graph)
        for sid in args.segments or ():
            if not graph.has_segment(sid):
                print(f"segment {sid} was removed")
                continue
            new = renumbered["segments"].get(sid, sid)
            if new != sid:
                print(f"segment {sid} is saved as {new}; node and segment ids shift after "
                      "each removed link (map in the report's 'renumbered')")
        _save(graph, args.out, args.graph, voxel_um=stamp)
    report_path = args.report_json or (str(args.out)+'.report.json' if args.out else None)
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')
    return 0


def run(args):
    from .__main__ import _correct_units, _load, _open_lattice, _save
    from . import radius_perimeter as rp
    from .roots import root_nodes_for

    if args.out and Path(args.out).resolve() in {Path(p).resolve() for p in args.graph}:
        raise ValueError("write a separate output; the measurement graph must be preserved")
    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    stamp = _correct_units(graph, frame, args)
    roots = set(getattr(args, 'root_node', ())) | set(root_nodes_for(graph, args, frame=frame))
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    if args.command == "refine-centreline":
        from .centreline_refine import refine
        report = refine(
            graph, frame, labels, method=args.method, sids=args.segments,
            fixed_nodes=set(args.fixed_nodes) | roots, root_nodes=roots,
            move_junctions=not args.fixed_junctions, strength=args.strength,
            max_iterations=args.max_iterations, max_half=args.max_half,
            max_samples=args.max_samples, workers=args.workers,
            reuse_sections=not getattr(args, 'no_section_cache', False),
            progress=lambda row: print(json.dumps(row), flush=True),
            path_progress=lambda row: print(json.dumps(dict(stage='path_fit', **row)), flush=True),
        ).to_dict()
        if not args.geometry_only:
            result = rp.measure_radii(graph, frame, labels, workers=args.workers,
                                       max_half=args.max_half, section_filter=True, _segment_ids=args.segments)
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
        from .radius_profile import prepare_profile
        import copy
        from .graphmodel import EditableGraph
        measurements = copy.deepcopy(graph.triple)
        measurement_paths = list(args.graph)
        path_plan = None
        if getattr(args, 'path_plan_json', None):
            document = json.loads(Path(args.path_plan_json).read_text(encoding='utf-8'))
            path_plan = document.get('geometry', document).get('path_plan')
            if not path_plan or args.radius_profile != 'dfs-confidence':
                raise ValueError('--path-plan-json requires a DFS refinement report and dfs-confidence')
        profile = prepare_profile(graph, policy=args.radius_profile,
                                  spacing_um=float(frame.seg_spacing[0]), root_nodes=roots,
                                  transition_radii=getattr(args, 'transition_radii', 4.), path_plan=path_plan).to_dict()
        # Revisit conflicting trusted endpoints using the shared section filter.
        conflicts = sorted({sid for row in profile['conflicts'] for sid in row['segments']})
        if conflicts:
            graph = EditableGraph(measurements)
            result = rp.measure_radii(graph, frame, labels, _segment_ids=conflicts,
                                      section_filter=True)
            rp.apply_radii(graph, result)
            if not args.out:
                raise ValueError('an output path is required to retain conflicting-anchor remeasurements')
            measurement_path = str(args.out)+'.remeasured.am'
            if Path(measurement_path).resolve() in {Path(p).resolve() for p in args.graph}:
                raise ValueError('remeasurement output would overwrite an input graph')
            _save(graph, measurement_path, args.graph, voxel_um=stamp)
            measurement_paths = [measurement_path]
            profile = prepare_profile(graph, policy=args.radius_profile,
                                      spacing_um=float(frame.seg_spacing[0]), root_nodes=roots,
                                      transition_radii=getattr(args, 'transition_radii', 4.), path_plan=path_plan).to_dict()
            profile['remeasured_segments'] = conflicts
        report = (dict(status='clearance_not_checked', surface_validation_required=True)
                  if args.skip_clearance else prepare(graph, frame, labels, max_iterations=args.max_iterations,
                         gap_um=args.gap_um, max_displacement_radii=args.max_displacement_radii,
                         progress=lambda row: print(json.dumps(row), flush=True)).to_dict())
        report['radius_profile'] = profile
        report['measurement_graph'] = measurement_paths
        report['radius_policy'] = 'derived profile transported unchanged during clearance; do not remeasure displaced geometry'
        report['surface_contract'] = dict(smooth_centrelines=False, rewrite_radii=False,
                                           radius_field='Thickness', surface_validation_required=True)
    moved = graph.triple.point_attrs.setdefault('centreline_displacement_um', {})
    graph.triple.point_attr_dtypes['centreline_displacement_um'] = 'float'
    for sid in graph.segment_ids():
        d = np.linalg.norm(graph.coords(sid)-before[sid], axis=1)
        moved.update(dict(zip(graph.segment(sid)['point_ids'], map(float, d))))
    print(json.dumps(report, indent=2), flush=True)
    _save(graph, args.out, args.graph, voxel_um=stamp)
    if args.out:
        import hashlib
        with open(args.out, 'rb') as stream:
            report['output_graph_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report_path = args.report_json or (str(args.out)+'.report.json' if args.out else None)
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Unresolved preflights produce a review artifact with a non-success exit status.
    if args.command == 'prepare-reconstruction':
        return 2 if (report['status'] != 'capsule-clear-surface-unvalidated'
                     or report['radius_profile']['status'] != 'prepared') else 0
    return 0 if report.get('converged', False) else 2
