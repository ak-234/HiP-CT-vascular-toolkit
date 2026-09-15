"""Reproducible centreline comparison on fixed segmentation sections.

Run as a module with --graph, --seg and --out-dir. Real-section residuals are
diagnostics, not independent anatomical ground truth. Synthetic tests provide
known-centre validation; no method is promoted automatically from these scores.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

from ..crosssection import _PlaneSampler, robust_edge_tangents, stable_transverse_cut
from . import centreline_refine as cr
from .adapter import Triple
from .graphmodel import EditableGraph

_STATE = None
VARIANTS = ("none", "legacy-none", "legacy-gaussian", "legacy-savgol",
            "legacy-bspline", "legacy-multiscale", "edt-geodesic",
            "centroid-spline:0.01", "centroid-spline:0.1", "centroid-spline:1",
            "laplacian", "taubin", "centroid-coherent:0.01")


def region_graph(graph, sid, padding):
    """Whole segment context in a spatial ROI, plus all immediate neighbours."""
    x = graph.coords(sid)
    lo, hi = x.min(axis=0)-padding, x.max(axis=0)+padding
    include = {sid}
    for k in ('node1', 'node2'):
        include.update(graph.node_segments(graph.segment(sid)[k]))
    for other in graph.segment_ids():
        xx = graph.coords(other)
        if len(xx) and np.all(xx.max(axis=0) >= lo) and np.all(xx.min(axis=0) <= hi):
            include.add(other)
    segments = [s for s in graph.segments if s['id'] in include]
    pids = {p for s in segments for p in s['point_ids']}
    nids = {s[k] for s in segments for k in ('node1', 'node2')}
    return EditableGraph(Triple(
        nodes={i: graph.nodes[i] for i in nids},
        points={i: graph.points[i] for i in pids},
        segments=copy.deepcopy(segments),
        point_attrs={k: {i: values[i] for i in pids if i in values}
                     for k, values in graph.triple.point_attrs.items()},
        point_attr_dtypes=graph.triple.point_attr_dtypes.copy()))


def edt_candidate(graph, sid, frame, labels, max_voxels=8_000_000):
    """Fixed-endpoint 3-D geodesic with inverse squared EDT cost, for comparison."""
    x = graph.coords(sid)
    ijk = frame.um_to_seg(x)
    pad = max(6, int(np.ceil(np.median(graph.radii(sid))/frame.seg_spacing[0]*2)))
    lo = np.maximum(0, np.floor(ijk.min(axis=0)).astype(int)-pad)
    hi = np.minimum(frame.seg_dims, np.ceil(ijk.max(axis=0)).astype(int)+pad+1)
    if np.prod(hi-lo) > max_voxels:
        raise ValueError('EDT ROI exceeds benchmark memory cap')
    if isinstance(labels, np.ndarray):
        mask = labels[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] > 0
    else:
        mask = np.stack([labels.slice_rows(int(z), int(lo[1]), int(hi[1]))[:, lo[0]:hi[0]]
                         for z in range(lo[2], hi[2])]) > 0
    distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1]
    cost = np.where(mask, 1/(distance+.5)**2, np.inf)
    a, b = np.rint(ijk[[0, -1]]-lo).astype(int)[:, ::-1]
    path, cost_value = route_through_array(cost, a, b, fully_connected=True, geometric=True)
    if not np.isfinite(cost_value):
        raise ValueError('EDT endpoints not connected within foreground')
    path = frame.seg_to_um(np.asarray(path)[:, ::-1]+lo)
    s, old_s = cr.arclength(path), cr.arclength(x)
    new = np.column_stack([np.interp(old_s/old_s[-1]*s[-1], s, path[:, k]) for k in range(3)])
    new[[0, -1]] = x[[0, -1]]
    return new


def _init(graph_path, seg):
    from threadpoolctl import threadpool_limits
    from .__main__ import _load, _open_lattice, _correct_units
    threadpool_limits(1)
    args = SimpleNamespace(graph=[graph_path], seg=seg, voxel_um=None,
                           labels_field='Labels', edits=None)
    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    _correct_units(graph, frame, args)
    global _STATE
    _STATE = graph, labels, frame


def reference_sections(graph, sid, frame, labels):
    """Fixed pre-refinement stations and planes, retained for every candidate."""
    x, radius = graph.coords(sid), graph.radii(sid)
    sampler = _PlaneSampler(labels, frame)
    tangents = robust_edge_tangents(x, radius, spacing_um=frame.seg_spacing[0])
    ids = np.unique(np.rint(np.linspace(.1, .9, 13)*(len(x)-1)).astype(int))
    refs = []
    ctx = cr._BranchContext.build(graph)
    for i in ids:
        sp = float(frame.seg_spacing[0])
        chosen = stable_transverse_cut(sampler, frame.um_to_seg(x[i])[0], tangents[i],
                                       max(radius[i]/sp, 2.), spacing_um=sp,
                                       max_half=256, centroid_mode='drift')
        if chosen is None:
            continue
        c = chosen.cut
        if any(cr._crosses_section(graph, int(r[0]), x[i], chosen.tangent, c, sp)
               for r in ctx.rivals(sid, x[i], chosen.tangent, radius[i])):
            continue
        centre = np.argwhere(c.blob8).mean(axis=0)-c.half
        centroid = x[i]+sp*(centre[0]*c.u+centre[1]*c.v)
        refs.append(dict(index=int(i), fraction=float(i/(len(x)-1)),
                         origin=x[i].tolist(), tangent=chosen.tangent.tolist(),
                         centroid=centroid.tolist()))
    return refs, len(ids)


def section_errors(x, references):
    errors, angles = [], []
    for ref in references:
        t = np.asarray(ref['tangent'])
        axial = (x-ref['origin']) @ t
        indices = np.flatnonzero(axial[:-1]*axial[1:] <= 0)
        if not len(indices):
            errors.append(None)
            angles.append(None)
            continue
        i = min(indices, key=lambda i: abs((i+.5)/(len(x)-1)-ref['fraction']))
        den = axial[i]-axial[i+1]
        f = axial[i]/den if abs(den) > 1e-12 else .5
        point = x[i]+f*(x[i+1]-x[i])
        direction = x[i+1]-x[i]
        errors.append(float(np.linalg.norm(point-ref['centroid'])))
        angles.append(float(np.degrees(np.arccos(np.clip(abs(direction@t)/max(
            np.linalg.norm(direction), 1e-12), 0, 1)))))
    return errors, angles


def _run_region(task):
    sid, split, variants, max_iterations = task
    full, labels, frame = _STATE
    padding = max(float(np.max(full.radii(sid))), float(np.percentile(
        np.concatenate([full.radii(s) for s in full.segment_ids()]), 99.5))) + 256*frame.seg_spacing[0]
    base = region_graph(full, sid, padding)
    references, attempted = reference_sections(base, sid, frame, labels)
    sampler = _PlaneSampler(labels, frame)
    original = base.coords(sid).copy()
    before_bad = cr.bad_edges(original, sampler, frame)
    rows = []
    for variant in variants:
        graph = EditableGraph(copy.deepcopy(base.triple))
        start = time.monotonic()
        status, detail = 'evaluated', {}
        try:
            if variant.startswith('legacy-'):
                from .skeleton_optimise import recentre
                from .smoothers import smooth
                # Legacy baselines use an isolated copy of this segment so their
                # graph-wide smoother does not change the comparison's context.
                segment = copy.deepcopy(graph.segment(sid))
                pid = segment['point_ids']
                small = EditableGraph(Triple(
                    {nid: graph.nodes[nid] for nid in (segment['node1'], segment['node2'])},
                    {p: graph.points[p] for p in pid}, [segment]))
                origin = {sid: small.coords(sid).copy()}
                for _ in range(2):
                    recentre(small, frame, labels, origin=origin)
                smooth(variant.removeprefix('legacy-'), small)
                graph.set_segment_coords(sid, small.coords(sid))
                detail['context_limitation'] = 'legacy single-segment baseline; junctions not qualified'
            elif variant == 'edt-geodesic':
                graph.set_segment_coords(sid, edt_candidate(graph, sid, frame, labels))
            elif variant != 'none':
                parts = variant.split(':')
                detail = cr.refine(graph, frame, labels, method=parts[0], sids=[sid],
                                   move_junctions=False, strength=float(parts[1]) if len(parts)>1 else .1,
                                   max_samples=16, max_iterations=max_iterations).to_dict()
            x = graph.coords(sid)
            errors, angles = section_errors(x, references)
            valid = [v for v in errors if v is not None]
            row = dict(segment=sid, split=split, method=variant, status=status,
                       reference_sections=len(references), attempted_sections=attempted,
                       missed_sections=sum(v is None for v in errors),
                       median_error_um=float(np.median(valid)) if valid else None,
                       p95_error_um=float(np.percentile(valid, 95)) if valid else None,
                       median_tangent_deg=float(np.median([v for v in angles if v is not None])) if valid else None,
                       new_outside_edges=int((cr.bad_edges(x, sampler, frame) & ~before_bad).sum()),
                       new_reversals=int((cr._reversals(x) & ~cr._reversals(original)).sum()),
                       seconds=time.monotonic()-start, detail=detail)
            row['coords'] = x.tolist()
        except Exception as exc:
            row = dict(segment=sid, split=split, method=variant, status='failed',
                       error=f'{type(exc).__name__}: {exc}', seconds=time.monotonic()-start)
        rows.append(row)
    return dict(segment=sid, split=split, references=references, rows=rows)


def select_regions(graph):
    mandatory = [3717, 3655, 3612, 1698, 216, 946]
    mandatory = [s for s in mandatory if graph.has_segment(s)]
    eligible = [sid for sid in graph.segment_ids() if len(graph.coords(sid)) >= 8 and sid not in mandatory]
    eligible.sort(key=lambda sid: (float(np.median(graph.radii(sid))), sid))
    rng = np.random.default_rng(202117)
    development, heldout = mandatory.copy(), []
    for group in np.array_split(eligible, 6):
        ids = list(map(int, rng.permutation(group)))
        heldout.extend(ids[:4])
        development.extend(ids[4:7])
    return [(s, 'development') for s in development[:24]] + [(s, 'heldout') for s in heldout[:24]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', required=True)
    p.add_argument('--seg', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-iterations', type=int, default=25)
    p.add_argument('--methods', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = p.parse_args()
    from .__main__ import _load
    graph = _load([args.graph])
    regions = select_regions(graph)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(graph=args.graph, segmentation=args.seg, regions=regions,
                    methods=args.methods, max_iterations=args.max_iterations,
                    seed=202117, metric='fixed-section residual; not anatomical ground truth')
    path = out/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('output directory already contains a different benchmark manifest')
    path.write_text(json.dumps(manifest, indent=2))
    tasks = [(s, split, args.methods, args.max_iterations) for s, split in regions
             if not (out/f'region_{s}.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init, initargs=(args.graph, args.seg)) as pool:
        futures = {pool.submit(_run_region, task): task[0] for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            (out/f"region_{result['segment']}.json").write_text(json.dumps(result, indent=2))
            print(f'{i}/{len(tasks)} regions complete: {result["segment"]}', flush=True)
    rows = []
    for sid, _ in regions:
        rows.extend(json.loads((out/f'region_{sid}.json').read_text())['rows'])
    summary = []
    for variant in args.methods:
        for split in ('development', 'heldout'):
            group = [r for r in rows if r['method'] == variant and r['split'] == split]
            valid = [r for r in group if r.get('median_error_um') is not None]
            summary.append(dict(method=variant, split=split, regions=len(group),
                                evaluated=len(valid), failures=sum(r['status']=='failed' for r in group),
                                median_error_um=float(np.median([r['median_error_um'] for r in valid])) if valid else None,
                                new_outside_edges=sum(r.get('new_outside_edges', 0) for r in group),
                                new_reversals=sum(r.get('new_reversals', 0) for r in group),
                                seconds=sum(r['seconds'] for r in group)))
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
