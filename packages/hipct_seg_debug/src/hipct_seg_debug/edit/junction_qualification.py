"""Checkpointed junction qualification and gated full-tree reconstruction.

Dataset paths are arguments. Region outputs retain the full graph's IDs; the
surface review uses the target segment and every branch incident to its nodes.
Outer endpoints of that review surface are artificial ROI boundaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def subset(graph, sids):
    from .adapter import Triple
    from .graphmodel import EditableGraph
    segments = [copy.deepcopy(graph.segment(sid)) for sid in sorted(sids)]
    pids = {pid for seg in segments for pid in seg['point_ids']}
    nids = {seg[key] for seg in segments for key in ('node1', 'node2')}
    return EditableGraph(Triple(
        nodes={nid: graph.nodes[nid] for nid in nids},
        points={pid: graph.points[pid] for pid in pids}, segments=segments,
        point_attrs={k: {p: v for p, v in values.items() if p in pids}
                     for k, values in graph.triple.point_attrs.items()},
        point_attr_dtypes=graph.triple.point_attr_dtypes.copy(),
        edge_attr_dtypes=graph.triple.edge_attr_dtypes.copy()))


def neighbourhood(graph, sid):
    from .centreline_refine import arclength
    nodes = {graph.segment(sid)[key] for key in ('node1', 'node2')}
    pending, selected = sorted(nodes), {sid}
    while pending:
        nid = pending.pop()
        for other in sorted(graph.node_segments(nid)):
            selected.add(other)
            seg = graph.segment(other)
            end = seg['node2'] if seg['node1'] == nid else seg['node1']
            radius = graph.radii(other)
            # A short internal link cannot provide fixed anchors between two
            # overlapping junction patches. Include the next junction together.
            if (end not in nodes and graph.degree(end) >= 2 and len(radius)
                    and arclength(graph.coords(other))[-1] <= 4*(radius[0]+radius[-1])):
                nodes.add(end)
                pending.append(end)
    return sorted(selected)


def evaluate(args, target=None):
    from .__main__ import _load, _open_lattice, _correct_units, _save
    from .centreline_refine import refine
    from .radius_perimeter import measure_radii, apply_radii, ACCEPTED
    from .radius_profile import prepare_profile
    from .centreline_clearance import prepare
    from .prepared_surface import reconstruct
    from .centreline_benchmark import reference_sections, section_errors
    graph = _load([args.graph])
    lattice_args = SimpleNamespace(graph=[args.graph], seg=args.seg, labels_field='Labels',
                                   voxel_um=None, edits=None)
    labels, frame, _ = _open_lattice(lattice_args)
    stamp = _correct_units(graph, frame, lattice_args)
    selected = graph.segment_ids() if target is None else neighbourhood(graph, target)
    directory = Path(args.out_dir)/('full_tree' if target is None else f'region_{target}')
    directory.mkdir(parents=True, exist_ok=True)
    refs, _ = reference_sections(graph, target, frame, labels) if target is not None else ([], 0)
    baseline = section_errors(graph.coords(target), refs)[0] if refs else []
    geometry_path = directory/'geometry.am'
    if geometry_path.exists() and (directory/'geometry.json').exists():
        graph = _load([str(geometry_path)])
        geometry = json.loads((directory/'geometry.json').read_text())
    else:
        def checkpoint(partial):
            _save(graph, str(directory/'geometry.iteration.am'), [args.graph], voxel_um=stamp)
            write_json(directory/'geometry.iteration.json', dict(partial, checkpoint_only=True,
                                                                radii_require_remeasurement=True))
        geometry = refine(graph, frame, labels, method='centroid-coherent', sids=selected,
                           fixed_nodes=args.fixed_node, strength=.01,
                           max_iterations=args.max_iterations, max_samples=args.max_samples,
                           workers=args.workers,
                           checkpoint=checkpoint,
                           section_progress=lambda row: print(json.dumps(dict(region=target,
                               stage='section_support', **row)), flush=True),
                           progress=lambda row: print(json.dumps(dict(region=target, **row)), flush=True)).to_dict()
        _save(graph, str(geometry_path), [args.graph], voxel_um=stamp)
        write_json(directory/'geometry.json', geometry)
    if args.geometry_only:
        report = dict(target_segment=target, target_segments=sorted(set(args.segment).intersection(selected)),
                      selected_segments=selected, geometry=geometry,
                      status='geometry_only', radii_require_remeasurement=True,
                      next_stage='remeasure after geometry qualification')
        write_json(directory/'report.json', report)
        return report
    measured_path = directory/'measured.am'
    if measured_path.exists() and (directory/'measurement.json').exists():
        graph = _load([str(measured_path)])
        measurement = json.loads((directory/'measurement.json').read_text())
    else:
        result = measure_radii(graph, frame, labels, _segment_ids=selected,
                                section_filter=True, workers=args.workers, max_half=256,
                                progress=lambda a, b: print(f'region {target}: radii {a}/{b}', flush=True))
        apply_radii(graph, result)
        measurement = {sid: dict(accepted=int(np.sum(result.reject_reason[sid] == ACCEPTED)),
                                  points=len(graph.coords(sid)),
                                  median_radius_um=float(np.median(graph.radii(sid)))) for sid in selected}
        _save(graph, str(measured_path), [args.graph], voxel_um=stamp)
        write_json(directory/'measurement.json', measurement)
    if getattr(args, 'measurement_only', False):
        report = dict(target_segment=target, selected_segments=selected, geometry=geometry,
                      measurement=measurement, measurement_graph=[str(measured_path)],
                      status='review_required', next_stage='derive reconstruction profiles and validate surfaces')
        write_json(directory/'report.json', report)
        return report
    review = subset(graph, selected)
    profile = prepare_profile(review, policy='confidence', spacing_um=float(frame.seg_spacing[0])).to_dict()
    conflicts = sorted({sid for row in profile['conflicts'] for sid in row['segments']})
    if conflicts:
        # Recheck measurements before authoring a profile, with full branch context.
        result = measure_radii(graph, frame, labels, _segment_ids=conflicts,
                                section_filter=True, workers=args.workers)
        apply_radii(graph, result)
        _save(graph, str(measured_path), [args.graph], voxel_um=stamp)
        review = subset(graph, selected)
        profile = prepare_profile(review, policy='confidence', spacing_um=float(frame.seg_spacing[0])).to_dict()
        profile['remeasured_segments'] = conflicts
    clearance = prepare(review, frame, labels, max_iterations=30).to_dict()
    mesh, surface = reconstruct(review, cells_across_diameter=args.cells_across_diameter,
                                maximum_cells=args.maximum_cells)
    mesh.save(directory/'surface.vtp')
    from .qualification_overlays import write_overlays
    target_segments = sorted(set(args.segment).intersection(selected))
    overlays = write_overlays(graph, review, target_segments, frame, labels, directory)
    # Copy only the prepared neighbourhood back into the full graph for an Amira
    # review artifact whose segment and point numbering still matches the input.
    for sid in selected:
        graph.set_segment_coords(sid, review.coords(sid))
        graph.set_segment_radii(sid, review.radii(sid))
    for name, values in review.triple.point_attrs.items():
        graph.triple.point_attrs.setdefault(name, {}).update(values)
        graph.triple.point_attr_dtypes[name] = review.triple.point_attr_dtypes[name]
    prepared_path = directory/'reconstruction.am'
    _save(graph, str(prepared_path), [args.graph], voxel_um=stamp)
    with prepared_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    valid_baseline = [v for v in baseline if v is not None]
    corrected = section_errors(_load([str(measured_path)]).coords(target), refs)[0] if refs else []
    valid_corrected = [v for v in corrected if v is not None]
    passed = (geometry['converged'] and geometry['outside_edges_after'] <= geometry['outside_edges_before']
              and profile['status'] == 'prepared'
              and clearance['status'] == 'capsule-clear-surface-unvalidated'
              and surface['status'] == 'validated_against_graph')
    report = dict(target_segment=target, target_segments=target_segments, overlays=overlays,
                  selected_segments=selected, geometry=geometry,
                  measurement=measurement, radius_profile=profile, clearance=clearance, surface=surface,
                  original_median_section_error_um=float(np.median(valid_baseline)) if valid_baseline else None,
                  corrected_median_section_error_um=float(np.median(valid_corrected)) if valid_corrected else None,
                  output_graph_sha256=digest, measurement_graph=[str(measured_path)],
                  surface_contract=dict(smooth_centrelines=False, rewrite_radii=False),
                  status='qualified' if passed else 'review_required')
    write_json(directory/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--seg', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--segment', type=int, action='append', default=[])
    parser.add_argument('--fixed-node', type=int, action='append', default=[])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-iterations', type=int, default=25)
    parser.add_argument('--max-samples', type=int, default=16)
    parser.add_argument('--cells-across-diameter', type=float, default=8.)
    parser.add_argument('--maximum-cells', type=int, default=5_000_000)
    parser.add_argument('--full-tree', action='store_true', help='run full tree only after every selected region qualifies')
    parser.add_argument('--experimental-full-tree', action='store_true',
                        help='run the full tree directly; bypass regional scheduling, not quality checks')
    parser.add_argument('--measurement-only', action='store_true',
                        help='stop after saving remeasured radii; reconstruction remains unvalidated')
    parser.add_argument('--geometry-only', action='store_true',
                        help='checkpoint a geometry experiment without starting radius or surface work')
    args = parser.parse_args(argv)
    if not args.segment and not args.experimental_full_tree:
        parser.error('provide the known failure regions and untouched controls with --segment')
    if args.experimental_full_tree and (args.full_tree or args.segment):
        parser.error('--experimental-full-tree is a standalone full-tree experiment')
    if args.geometry_only and args.measurement_only:
        parser.error('choose one stopping stage')
    if args.geometry_only and args.full_tree:
        parser.error('--geometry-only cannot qualify a full-tree reconstruction')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    code = hashlib.sha256()
    import hipct_seg_debug
    import coronary_sdf
    for package in (hipct_seg_debug, coronary_sdf):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob('*.py')):
            code.update(str(path.relative_to(root)).encode())
            code.update(path.read_bytes())
    manifest = dict(options=vars(args), graph=fingerprint(args.graph), segmentation=fingerprint(args.seg),
                    code_sha256=code.hexdigest())
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != manifest:
        raise ValueError('checkpoint inputs, code or options changed; use a new output directory')
    write_json(out/'manifest.json', manifest)
    if args.experimental_full_tree:
        report = evaluate(args)
        report['experimental_full_tree'] = True
        write_json(out/'summary.json', [report])
        return 0 if report['status'] == 'qualified' else 2
    reports = []
    for target in sorted(set(args.segment)):
        if any(target in row.get('target_segments', []) for row in reports):
            continue
        path = out/f'region_{target}'/'report.json'
        try:
            report = json.loads(path.read_text()) if path.exists() else evaluate(args, target)
        except Exception as exc:
            report = dict(target_segment=target, status='failed', error=f'{type(exc).__name__}: {exc}',
                          traceback=traceback.format_exc())
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path.parent/'failure.json', report)
        reports.append(report)
        write_json(out/'summary.json', reports)
    passed = all(row['status'] == 'qualified' for row in reports)
    if args.full_tree and passed:
        reports.append(evaluate(args))
        write_json(out/'summary.json', reports)
    elif args.full_tree:
        print('Full-tree gate not passed; review the regional reports.', flush=True)
    return 0 if all(row['status'] == 'qualified' for row in reports) else 2


if __name__ == '__main__':
    raise SystemExit(main())
