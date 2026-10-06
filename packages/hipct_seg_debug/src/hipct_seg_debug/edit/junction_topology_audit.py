"""Read-only candidates for complex junctions in an existing graph.

Short, unsupported links between branching nodes may describe either closely
spaced bifurcations or one multifurcation. This audit does not decide anatomical
connectivity, merge nodes, or use a radius-based distance as proof of topology.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(graph, geometry, sids=None):
    rows = geometry.get('segments', {})
    selected = sorted(map(int, rows)) if sids is None else sorted(set(map(int, sids)))
    if any(not graph.has_segment(sid) for sid in selected):
        raise ValueError('unknown audited segment')
    nodes = {graph.segment(sid)[k] for sid in selected for k in ('node1', 'node2')}
    branching = sorted(n for n in nodes if graph.degree(n) >= 3)
    parent = {n: n for n in branching}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    links = []
    for sid in selected:
        seg = graph.segment(sid)
        a, b = seg['node1'], seg['node2']
        if a not in parent or b not in parent:
            continue
        x, radius = graph.coords(sid), graph.radii(sid)
        length = float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
        if not len(radius) or not np.isfinite(radius[[0, -1]]).all() or min(radius[[0, -1]]) <= 0:
            continue
        extent = float(radius[0]+radius[-1])
        if length > extent:
            continue
        row = rows.get(str(sid), rows.get(sid, {}))
        count = row.get('centring_final', {}).get('accepted_sections')
        evidence = ('not_audited' if count is None else
                    'exclusive_sections_observed' if count > 0 else 'no_exclusive_sections_observed')
        links.append(dict(segment=sid, nodes=[a, b], length_um=length,
                          endpoint_radius_sum_um=extent, length_radius_ratio=length/extent,
                          accepted_sections=count, section_evidence=evidence))
        # Missing evidence must not masquerade as a failed section audit.
        if count == 0:
            ra, rb = root(a), root(b)
            parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for node in branching:
        groups.setdefault(root(node), []).append(node)
    complexes = []
    for members in groups.values():
        if len(members) == 1 and graph.degree(members[0]) < 4:
            continue
        member_set = set(members)
        incident = sorted({sid for n in members for sid in graph.node_segments(n)})
        ports, internal = [], []
        for sid in incident:
            seg = graph.segment(sid)
            endpoints = [seg['node1'], seg['node2']]
            if set(endpoints) <= member_set:
                internal.append(sid)
            else:
                node = next(n for n in endpoints if n in member_set)
                far = endpoints[1] if endpoints[0] == node else endpoints[0]
                ports.append(dict(segment=sid, node=node, remote_node=far))
        complexes.append(dict(nodes=members, internal_segments=internal, external_ports=ports,
                              incident_branch_count=len(ports), status='topology_review_required',
                              interpretation='A multifurcation and closely spaced bifurcations remain alternatives.'))

    node_rows = []
    for node in branching:
        incident = sorted(graph.node_segments(node))
        errors = []
        for sid in incident:
            seg = graph.segment(sid)
            i = 0 if seg['node1'] == node else -1
            errors.append(np.linalg.norm(graph.coords(sid)[i]-np.asarray(graph.nodes[node][:3])))
        node_rows.append(dict(node=node, degree=graph.degree(node), segments=incident,
                              position_um=list(map(float, graph.nodes[node][:3])),
                              max_endpoint_mismatch_um=float(max(errors, default=0.))))
    return dict(status='diagnostic_only', nodes=node_rows, short_links=links, complexes=complexes,
                connectivity_modified=False,
                limitations=[
                    'Stored radii define a provisional proximity scale and can be wrong in flattened vessels.',
                    'No accepted sections at sampled points does not prove absence of exclusive lumen.',
                    'Imported connectivity is not independently verified by the refinement.',
                    'Section evidence must come from a report matching this graph.',
                    'Segmentation overlays and all external approaches are required before a topology edit.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--geometry-report', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    from .__main__ import _load
    from .junction_qualification import fingerprint, write_json
    graph = _load([args.graph])
    report = audit(graph, json.loads(Path(args.geometry_report).read_text()))
    report['inputs'] = dict(graph=fingerprint(args.graph), geometry_report=fingerprint(args.geometry_report))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps(dict(complexes=len(report['complexes']), status=report['status'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
