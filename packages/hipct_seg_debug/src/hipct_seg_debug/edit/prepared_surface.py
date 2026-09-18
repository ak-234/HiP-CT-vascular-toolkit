"""Direct prepared-graph surfacing without hidden centreline/radius rewrites."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def prepared_field(graph, *, blend_fraction=.15):
    from coronary_sdf.capsules import build_capsules
    from coronary_sdf.implicit_field import build_graph_implicit_field
    runs = []
    incident = {nid: set(graph.node_segments(nid)) for nid in graph.nodes}
    for sid in sorted(graph.segment_ids()):
        x, r = graph.coords(sid), graph.radii(sid)
        if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(r).all() or np.any(r <= 0):
            raise ValueError(f'segment {sid} has unsupported surface geometry/radii')
        if np.any(np.linalg.norm(np.diff(x, axis=0), axis=1) <= 1e-9):
            raise ValueError(f'segment {sid} contains a zero-length edge')
        segment = graph.segment(sid)
        runs.append(dict(coords=x/1000., radii=r/1000., seg_idx=sid,
                         node1_id=segment['node1'], node2_id=segment['node2']))
    if not runs:
        raise ValueError('prepared graph is empty')
    capsules = build_capsules(runs, node_to_segs=incident)
    return build_graph_implicit_field(capsules, graph.nodes, incident,
                                      blend_fraction=blend_fraction, support_factor=4.)


def expected_topology(graph):
    incident = {nid: set(graph.node_segments(nid)) for nid in graph.nodes if graph.degree(nid)}
    unseen = set(incident)
    components = 0
    while unseen:
        stack = [min(unseen)]
        components += 1
        while stack:
            nid = stack.pop()
            if nid not in unseen:
                continue
            unseen.remove(nid)
            for sid in incident[nid]:
                edge = graph.segment(sid)
                stack.extend([edge['node1'], edge['node2']])
    return components, len(graph.segment_ids())-len(incident)+components


def branch_radius_audit(graph, surface, field):
    """Actual mesh rays away from junction supports; misses stay explicit."""
    from ..crosssection import _plane_axes, robust_edge_tangents
    rows = []
    for sid in sorted(graph.segment_ids()):
        x, r = graph.coords(sid)/1000., graph.radii(sid)/1000.
        tangents = robust_edge_tangents(x, r, spacing_um=max(np.median(r)/4, 1e-6))
        for i in np.unique(np.rint(np.linspace(.2, .8, min(3, len(x)))*(len(x)-1)).astype(int)):
            if any(np.linalg.norm(x[i]-j.position) <= j.support_radius for j in field.junctions):
                continue
            axes = _plane_axes(tangents[i])
            if axes is None:
                continue
            u, v = axes
            distances = []
            for theta in np.arange(16)*2*np.pi/16:
                direction = np.cos(theta)*u+np.sin(theta)*v
                hits, _ = surface.ray_trace(x[i], x[i]+3*r[i]*direction, first_point=False)
                distance = np.linalg.norm(np.asarray(hits).reshape(-1, 3)-x[i], axis=1)
                distance = distance[distance > 1e-8]
                distances.append(float(distance.min()/r[i]-1) if len(distance) else None)
            good = [v for v in distances if v is not None]
            rows.append(dict(segment=sid, point=int(i), missed_rays=16-len(good),
                             max_absolute_relative_error=max(map(abs, good), default=None)))
    return rows


def reconstruct(graph, *, cells_across_diameter=12., maximum_cells=5_000_000):
    from coronary_sdf.vtk_htg_mesher import mesh_vtk_hyper_tree_grid
    from coronary_sdf.mesh_validation import validate_mesh
    field = prepared_field(graph)
    mesh, statistics = mesh_vtk_hyper_tree_grid(
        field, cells_across_diameter=cells_across_diameter, maximum_cells=maximum_cells)
    components, genus = expected_topology(graph)
    validation = validate_mesh(mesh, expected_components=components, expected_genus=genus)
    radius_checks = branch_radius_audit(graph, mesh, field)
    # This is a numerical reconstruction tolerance, not anatomical accuracy.
    tolerance = 2./cells_across_diameter
    bad = [row for row in radius_checks if row['missed_rays'] or
           row['max_absolute_relative_error'] is None or row['max_absolute_relative_error'] > tolerance]
    junction_checks = junction_mesh_audit(graph, mesh, field, tolerance)
    report = dict(surface_units='mm', geometry_rewritten=False, radii_rewritten=False,
                  mesh_validation=validation.to_dict(), statistics=statistics.to_dict(),
                  branch_radius_checks=radius_checks, relative_radius_tolerance=tolerance,
                  junction_count=len(field.junctions),
                  junction_mesh_checks=junction_checks,
                  status='validated_against_graph' if validation.valid and not bad and radius_checks
                         and all(row['valid'] for row in junction_checks)
                         else 'review_required', anatomical_validation_required=True)
    return mesh, report


def junction_mesh_audit(graph, mesh, field, tolerance):
    """Check every patch vertex against the intended blend and unblended union.

    Negative residuals flag artificial inward necks; positive residuals beyond
    the declared blend depth flag uncontrolled outward bulges. These numerical
    bounds complement topology/intersection checks; carina anatomy still needs
    segmentation overlays and known-shape junction fixtures.
    """
    if not field.junctions:
        return []
    raw = prepared_field(graph, blend_fraction=0.)
    rows = []
    points = np.asarray(mesh.points)
    for junction in field.junctions:
        inside = np.linalg.norm(points-junction.position, axis=1) <= junction.support_radius
        selected = points[inside]
        if not len(selected):
            rows.append(dict(node=junction.node_id, valid=False, reason='no_patch_vertices'))
            continue
        values, _, radius, _ = field.evaluate(selected)
        unblended, _, _, _ = raw.evaluate(selected)
        # Overlapping junctions have independently bounded, jointly assessed
        # smooth unions. Use the largest eligible depth at each mesh vertex.
        depth = np.zeros(len(selected))
        for neighbour in field.junctions:
            relevant = np.linalg.norm(selected-neighbour.position, axis=1) <= neighbour.support_radius
            depth[relevant] = np.maximum(depth[relevant], neighbour.blend_depth)
        error = tolerance*np.maximum(radius, 1e-12)
        rows.append(dict(node=junction.node_id, vertices=len(selected),
            max_absolute_relative_field_error=float(np.max(np.abs(values)/radius)),
            min_unblended_residual_mm=float(unblended.min()),
            max_unblended_residual_mm=float(unblended.max()),
            valid=bool(np.all(np.abs(values) <= error) and np.all(unblended >= -error)
                       and np.all(unblended <= depth+error))))
    return rows


def run(args):
    from .__main__ import _load
    if len(args.graph) != 1:
        raise ValueError('prepared surfacing requires exactly one graph')
    contract = json.loads(Path(args.prepared_report).read_text(encoding='utf-8'))
    with open(args.graph[0], 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if contract.get('output_graph_sha256') != digest or 'surface_contract' not in contract:
        raise ValueError('preparation report does not match this graph')
    graph = _load(args.graph)
    if ('selected_segments' in contract
            and set(contract['selected_segments']) != set(graph.segment_ids())):
        raise ValueError('regional preparation does not qualify the unprepared remainder of this graph')
    mesh, report = reconstruct(graph, cells_across_diameter=args.cells_across_diameter,
                               maximum_cells=args.maximum_cells)
    report['preparation_report'] = str(args.prepared_report)
    report['preparation_status'] = contract.get('status')
    clearance_status = contract.get('clearance', {}).get('status', contract.get('status'))
    if clearance_status != 'capsule-clear-surface-unvalidated' or \
            contract.get('radius_profile', {}).get('status') != 'prepared':
        report['status'] = 'review_required'
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh.save(out/'prepared_surface.vtp')
    mesh.save(out/'prepared_surface.stl')
    (out/'surface_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'validated_against_graph' else 2
