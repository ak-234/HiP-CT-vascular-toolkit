"""Longest-path-first joint fitting, retaining earlier curves outside junctions."""
from __future__ import annotations

import hashlib

import numpy as np

from .centreline_refine import arclength
from .dfs_paths import path_samples
from .junction_cluster import fit_cluster


def _fit_inputs(graph, main, held):
    """Fingerprint of everything a fit of `main` reads that an earlier fit can change.

    Observations are fixed for the whole call, so geometry is what varies: the run's
    own segments and every segment incident to its nodes (their released spans),
    plus which of its nodes an earlier fit has since frozen.
    """
    nodes = {graph.segment(sid)[key] for sid in main for key in ('node1', 'node2')}
    touched = sorted(set(main) | {s for n in nodes for s in graph.node_segments(n)})
    digest = hashlib.sha1(repr((tuple(main), sorted(nodes & held))).encode())
    for sid in touched:
        digest.update(np.ascontiguousarray(graph.coords(sid)).tobytes())
    return digest.hexdigest()


def fit_paths(graph, paths, selected, observations, scales, sampler, frame, strength, held,
              progress=None, section_weighting=False):
    selected, held, processed = set(selected), set(held), set()
    reports = {}
    # A failed run is retried only when a later accepted fit changed its inputs, not
    # after any accepted fit anywhere (the 3717 run re-solved one cluster six times).
    failed = set()
    spacing = float(frame.seg_spacing[0])
    for path in paths:
        # A regional selection can intersect a path in several disjoint runs.
        runs, run = [], []
        for sid in path['segments']:
            if sid in selected and sid not in processed:
                run.append(sid)
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        for main in runs:
            key = _fit_inputs(graph, main, held)
            if key in failed:
                continue
            coords, refs = path_samples(graph, path, main)
            good = np.array([i in observations[sid][2] for sid, i in refs])
            path_s = arclength(coords)
            base = dict(path_root=path['root'], path_terminal=path['terminal'],
                        path_length_um=path['length_um'], new_segments=main)
            if good.sum() < 2 or np.ptp(path_s[good]) < 1e-9:
                reports[len(reports)] = dict(base, status='insufficient_support', nodes=[],
                    segments=main, supported_segments=[], reason='path_has_fewer_than_two_distinct_sections')
                failed.add(key)
                if progress:
                    progress(dict(path_terminal=path['terminal'], new_segments=main,
                                  status='insufficient_support'))
                continue
            nodes = {graph.segment(sid)[key] for sid in main for key in ('node1', 'node2')}
            movable = {n for n in nodes if n not in held and graph.degree(n) > 1
                       and set(graph.node_segments(n)) <= selected}
            spans = {sid: (0, len(graph.coords(sid))-1) for sid in main}
            # Every incident approach participates in a node movement. Previously
            # fitted branches are released only inside this bounded neighbourhood;
            # their outer two samples anchor both position and direction.
            for node in sorted(movable):
                for sid in sorted(graph.node_segments(node)):
                    if sid in main:
                        continue
                    x = graph.coords(sid)
                    order = np.arange(len(x))
                    if graph.segment(sid)['node2'] == node:
                        order = order[::-1]
                    s = arclength(x[order])
                    length = max(6*spacing, 4*float(scales[sid][order[0]]))
                    count = min(len(x), max(4, int(np.searchsorted(s, length))+1))
                    bounds = int(order[:count].min()), int(order[:count].max())
                    if sid in spans:
                        bounds = min(bounds[0], spans[sid][0]), max(bounds[1], spans[sid][1])
                    spans[sid] = bounds
            result = fit_cluster(graph, movable, spans, observations, scales, sampler, frame,
                                 strength, require_external_support=False,
                                 **({'section_weighting': True} if section_weighting else {}))
            # The path can bridge an unmeasurable short segment only when accepted
            # sections bracket it. Unsupported side approaches remain explicit.
            left, right = path_s[good][[0, -1]]
            supported = []
            for sid in main:
                indices = [i for i, (owner, _) in enumerate(refs) if owner == sid]
                if len(observations[sid][2]) >= 2 or (path_s[indices[0]] >= left and path_s[indices[-1]] <= right):
                    supported.append(sid)
            result.update(base, supported_segments=supported)
            reports[len(reports)] = result
            if progress:
                progress(dict(path_terminal=path['terminal'], new_segments=main,
                              status=result['status'], max_move_um=result.get('max_move_um', 0.),
                              **({'blocked_detail': result['blocked_detail']}
                                 if 'blocked_detail' in result else {})))
            if result['status'] in ('moving', 'stationary'):
                processed.update(main)
                # Longest path first: a later path attaches to a junction an earlier
                # fit has placed, instead of moving it again from its own subset of
                # sections. Two overlapping fits otherwise pull the same node to
                # different places every iteration (node 636 on region 318 diverged
                # from 14 to 103 um with frozen sections). It may move again in the
                # next iteration's first fit.
                held.update(movable)
            else:
                failed.add(key)
    return reports
