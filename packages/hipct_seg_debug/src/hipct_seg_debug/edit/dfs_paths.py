"""Deterministic root-to-terminal paths, ordered by physical length."""
from __future__ import annotations

import numpy as np


def root_paths(graph, roots=()):
    from .roots import auto
    explicit = set(map(int, roots))
    if not explicit <= set(graph.nodes):
        raise ValueError('unknown root node')
    starts = sorted(explicit) + [n for n in auto(graph) if n not in explicit]
    seen, paths = set(), []
    for root in starts:
        if root in seen:
            if root in explicit:
                raise ValueError('provide only one root per connected tree')
            continue
        parent, distance = {root: (None, None)}, {root: 0.}
        stack, leaves = [root], []
        seen.add(root)
        while stack:
            node = stack.pop()
            children = []
            for sid in sorted(graph.node_segments(node)):
                if sid == parent[node][1]:
                    continue
                seg = graph.segment(sid)
                other = seg['node2'] if seg['node1'] == node else seg['node1']
                if other in seen:
                    raise ValueError('DFS fitting requires a forest; cycle edges must not be discarded')
                seen.add(other)
                parent[other] = (node, sid)
                distance[other] = distance[node]+float(np.linalg.norm(np.diff(graph.coords(sid), axis=0), axis=1).sum())
                children.append(other)
            if not children and node != root:
                leaves.append(node)
            stack.extend(reversed(children))
        for leaf in leaves:
            nodes, segments = [leaf], []
            while nodes[-1] != root:
                node, sid = parent[nodes[-1]]
                nodes.append(node)
                segments.append(sid)
            paths.append(dict(root=root, terminal=leaf, nodes=nodes[::-1],
                              segments=segments[::-1], length_um=distance[leaf]))
    return sorted(paths, key=lambda p: (-p['length_um'], p['root'], p['terminal']))


def path_samples(graph, path, segments=None):
    """Coordinates plus (segment, local point) ownership; retain joint records."""
    selected = set(path['segments'] if segments is None else segments)
    refs, coords = [], []
    for node, sid in zip(path['nodes'], path['segments']):
        if sid not in selected:
            continue
        x = graph.coords(sid)
        order = range(len(x)) if graph.segment(sid)['node1'] == node else range(len(x)-1, -1, -1)
        for i in order:
            refs.append((sid, i))
            coords.append(x[i])
    return np.asarray(coords), refs


def validate_path_plan(graph, paths, roots=()):
    """Retain the geometry fit's original order, checking topology before reuse."""
    plan_roots = {p['root'] for p in paths}
    if roots and not set(roots) <= plan_roots:
        raise ValueError('path plan disagrees with requested roots')
    actual = root_paths(graph, plan_roots)
    def signature(p):
        return p['root'], p['terminal'], tuple(p['nodes']), tuple(p['segments'])
    signatures = [signature(p) for p in paths]
    if len(set(signatures)) != len(signatures) or set(signatures) != {signature(p) for p in actual}:
        raise ValueError('path plan does not match graph topology')
    return paths
