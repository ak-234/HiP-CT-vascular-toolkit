"""Adapter to the existing image-based radius estimator; never edits source data."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import networkx as nx
import numpy as np

from .network import positive


class UnsupportedHydraulicSpan(ValueError):
    def __init__(self, s, supported):
        indices = np.flatnonzero(~supported)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
        self.spans = [dict(first_sample=int(g[0]), last_sample=int(g[-1]),
                          start_um=float(s[g[0]]), end_um=float(s[g[-1]]), samples=len(g))
                      for g in groups if len(g)]
        super().__init__(f"Unsupported hydraulic span: {len(indices)} samples")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def open_segmentation(path, voxel_um, cache_dir):
    """Existing lattice/frame readers with caches confined to the output directory."""
    from hipct_seg_debug import amira, rle
    from hipct_seg_debug.frame import WorldFrame
    info = amira.read_lattice_header(path)
    field = "Labels" if "Labels" in info.fields else next(iter(info.fields)) if len(info.fields) == 1 else None
    if field is None:
        raise ValueError("Segmentation must have a Labels field or exactly one lattice field")
    labels = rle.open_lattice(path, info.fields[field], info.dims, cache_dir=cache_dir)
    frame = WorldFrame.from_inputs(tuple(info.dims[::-1]), voxel_um, info, voxel_is_truth=True)
    return labels, frame, info


def hydraulic_profile(s, measured, trusted, junction_ends, spacing_um, override=None):
    """Only bounded gaps and short branch-specific junction extensions are supported."""
    from hipct_seg_debug.edit.radius_profile import interpolate_supported
    s, measured, trusted = np.asarray(s, float), np.asarray(measured, float), np.asarray(trusted, bool)
    if override is not None:
        r = np.asarray(override["radius_um"], float)
        if not override.get("reason") or r.shape != measured.shape or not np.all(np.isfinite(r)) or np.any(r <= 0):
            raise ValueError("Hydraulic overrides need a reason and one positive radius per point")
        return r, {"source": "explicit_override", "reason": override["reason"]}
    r, changed = interpolate_supported(s, measured, trusted)
    supported = trusted | changed
    anchors = np.flatnonzero(trusted)
    if len(anchors):
        for at_start, junction in zip((True, False), junction_ends):
            idx = anchors[0] if at_start else anchors[-1]
            span = s[idx]-s[0] if at_start else s[-1]-s[idx]
            if junction and span <= max(8*spacing_um, 4*measured[idx]):
                sl = slice(0, idx) if at_start else slice(idx+1, None)
                r[sl] = measured[idx]
                supported[sl] = True
    if not np.all(supported):
        raise UnsupportedHydraulicSpan(s, supported)
    return r, {"source": "trusted_and_supported", "trusted": int(trusted.sum()),
               "modelled": int((~trusted).sum())}


def truncate_tree(tree, cutoff):
    """First below-threshold sample on each rooted path, retaining an outlet stub.

    Cuts are on sampled centreline points, with their actual corrected radii recorded.
    The shortest retained stub has one original sample interval. No sealing or
    survivor-flow renormalisation is performed. Loops need explicit cut definitions.
    """
    positive(cutoff, "clinical cutoff")
    graph = nx.Graph()
    by_pair = {}
    for edge in tree["edges"]:
        u, v = str(edge["u"]), str(edge["v"])
        if graph.has_edge(u, v):
            raise ValueError("Automatic truncation requires a tree; supply explicit cuts for loops")
        graph.add_edge(u, v)
        by_pair[frozenset((u, v))] = edge
    if not nx.is_tree(graph):
        raise ValueError("Automatic truncation requires a tree; supply explicit retained_edge_ids/outlets")
    root = str(tree["root"])
    if graph.degree(root) != 1:
        raise ValueError("Clinical inlet must be a free endpoint")
    directed = nx.bfs_tree(graph, root)
    oriented = {}
    for u, v in directed.edges:
        edge = dict(by_pair[frozenset((u, v))])
        if str(edge["u"]) != u:
            edge["radius_m"] = edge["radius_m"][::-1]
            edge["xyz_m"] = edge["xyz_m"][::-1]
            if "surface_radius_m" in edge:
                edge["surface_radius_m"] = edge["surface_radius_m"][::-1]
            if "point_attributes" in edge:
                edge["point_attributes"] = {k: v[::-1] for k, v in edge["point_attributes"].items()}
            edge["s_m"] = (edge["s_m"][-1]-np.array(edge["s_m"])[::-1]).tolist()
        edge.update(u=u, v=v)
        oriented[(u, v)] = edge
    retained, outlets, replaced = [], [], {}
    queue = [root]
    while queue:
        u = queue.pop()
        for v in directed.successors(u):
            edge = oriented[(u, v)]
            radii = np.asarray(edge["radius_m"])
            indices = np.flatnonzero(radii[1:] < cutoff)+1
            cut = int(indices[0]) if len(indices) else None
            if cut is None:
                retained.append(str(edge["id"]))
                if directed.out_degree(v):
                    queue.append(v)
                else:
                    outlets.append(dict(id=f"terminal:{edge['id']}", node=v, radius_m=float(radii[-1])))
            elif cut == len(radii)-1:
                retained.append(str(edge["id"]))
                outlets.append(dict(id=f"cut:{edge['id']}", node=v, radius_m=float(radii[cut])))
            else:
                node = f"cut:{edge['id']}:{cut}"
                prox, distal = dict(edge), dict(edge)
                prox.update(id=f"{edge['id']}:prox", v=node)
                distal.update(id=f"{edge['id']}:dist", u=node)
                for key in ("radius_m", "xyz_m"):
                    prox[key], distal[key] = edge[key][:cut+1], edge[key][cut:]
                if "surface_radius_m" in edge:
                    prox["surface_radius_m"], distal["surface_radius_m"] = edge["surface_radius_m"][:cut+1], edge["surface_radius_m"][cut:]
                if "point_attributes" in edge:
                    prox["point_attributes"] = {k: v[:cut+1] for k, v in edge["point_attributes"].items()}
                    distal["point_attributes"] = {k: v[cut:] for k, v in edge["point_attributes"].items()}
                prox["s_m"] = edge["s_m"][:cut+1]
                distal["s_m"] = (np.array(edge["s_m"])[cut:]-edge["s_m"][cut]).tolist()
                replaced[str(edge["id"])] = (prox, distal)
                retained.append(prox["id"])
                outlets.append(dict(id=node, node=node, radius_m=float(radii[cut])))
    edges = []
    for edge in oriented.values():
        edges.extend(replaced.get(str(edge["id"]), [edge]))
    return {**tree, "edges": edges, "retained_edge_ids": retained, "outlets": outlets,
            "crop_policy": "first-below-threshold-sample-with-open-stub", "clinical_radius_m": cutoff}


def write_network_graph(tree, path, retained_only=False, surface=False):
    """Geometry export plus stable-source sidecar; coordinates/radii in micrometres."""
    from hipct_seg_debug.edit.adapter import Triple, to_spatial_graph
    from hipct_seg_debug.edit.amira_write import write_spatial_graph
    selected = [e for e in tree["edges"] if not retained_only or str(e["id"]) in tree["retained_edge_ids"]]
    names = sorted({str(e[k]) for e in selected for k in ("u", "v")})
    node_ids = {name: i for i, name in enumerate(names)}
    nodes, points, segments, mapping, point_attrs = {}, {}, [], [], {}
    for sid, e in enumerate(selected):
        pids = []
        radii = e.get("surface_radius_m", e["radius_m"]) if surface else e["radius_m"]
        for i, (xyz, radius) in enumerate(zip(e["xyz_m"], radii)):
            pid = len(points)
            points[pid] = (*[float(x)*1e6 for x in xyz], float(radius)*1e6)
            pids.append(pid)
            for name, values in e.get("point_attributes", {}).items():
                point_attrs.setdefault(name, {})[pid] = values[i] if values[i] is not None else np.nan
            point_attrs.setdefault("radius_hydraulic_um", {})[pid] = e["radius_m"][i]*1e6
        u, v = node_ids[str(e["u"])], node_ids[str(e["v"])]
        nodes[u] = (*points[pids[0]][:3], 0)
        nodes[v] = (*points[pids[-1]][:3], 0)
        attrs = {k: val for k, val in e.get("attributes", {}).items() if isinstance(val, (int, float))}
        length_um = (e["s_m"][-1]-e["s_m"][0])*1e6
        chord_um = float(np.linalg.norm(np.asarray(e["xyz_m"][-1])-e["xyz_m"][0]))*1e6
        for key, value in (("CurvedLength", length_um), ("ChordLength", chord_um)):
            if key in attrs:
                attrs[key] = value
        if "Tortuosity" in attrs and chord_um > 0:
            attrs["Tortuosity"] = length_um/chord_um
        if "Volume" in attrs:
            r = np.asarray(radii)*1e6
            attrs["Volume"] = float(np.sum(np.diff(e["s_m"])*1e6*np.pi*(r[:-1]**2+r[:-1]*r[1:]+r[1:]**2)/3))
        segments.append({**attrs, "id": sid, "node1": u, "node2": v, "point_ids": pids,
                         "MeanRadius": float(np.mean(radii))*1e6})
        mapping.append(dict(export_segment_id=sid, source_edge_id=e["id"],
                            source_segment_id=e.get("source_segment_id", e["id"])))
    for nid in nodes:
        nodes[nid] = (*nodes[nid][:3], sum(nid in (e["node1"], e["node2"]) for e in segments))
    triple = Triple(nodes, points, segments, point_attrs=point_attrs)
    write_spatial_graph(to_spatial_graph(triple), path, voxel_um=tree.get("voxel_um"),
                        parameters_from=tree.get("source_graph"))
    write_json(str(path)+".mapping.json", dict(nodes=node_ids, edges=mapping,
               profile="surface" if surface else "hydraulic"))


def prepare(config, output, *, audit_only=False, surfaces=False):
    from hipct_seg_debug import amira
    from hipct_seg_debug.edit import __main__ as edit
    from hipct_seg_debug.edit import radius_perimeter as rp
    from hipct_seg_debug.edit.radius_profile import trusted_radii
    geometry = config["geometry"]
    graph_path, segmentation = Path(geometry["graph"]), Path(geometry["segmentation"])
    for path in (graph_path, segmentation):
        if not path.is_file():
            raise ValueError(f"Input file not found: {path}")
    graph = edit._load([str(graph_path)])
    info = amira.read_lattice_header(segmentation)
    output = Path(output)
    roots = geometry.get("roots", {})
    subset_ids = geometry.get("measurement_segment_ids")
    if subset_ids is not None:
        subset_ids = list(map(int, subset_ids))
        if not subset_ids or len(set(subset_ids)) != len(subset_ids) or not set(subset_ids) <= set(graph.segment_ids()):
            raise ValueError("measurement_segment_ids must be unique existing source segment IDs")
    measured_path = output/("measured_subset.am" if subset_ids is not None else "corrected.am")
    audit = dict(graph=str(graph_path), graph_sha256=digest(graph_path),
                 segmentation=str(segmentation), segmentation_sha256=digest(segmentation),
                 segmentation_size_bytes=segmentation.stat().st_size,
                 segmentation_spacing_um=np.asarray(info.spacing).tolist(),
                 graph_voxel_stamp_um=amira.read_voxel_stamp(graph_path),
                 components=len(graph.components()), segments=len(graph.segments),
                 radius_quantiles_um=np.quantile([p[3] for p in graph.points.values()], [0,.05,.5,.95,1]).tolist(),
                 endpoints=[dict(node=int(n), xyz_um=list(graph.nodes[n][:3]),
                            segment_ids=sorted(graph.node_segments(n))) for n in graph.endpoints()],
                 configured_roots=roots, status="audit_only")
    write_json(output/"geometry_audit.json", audit)
    if audit_only:
        return audit
    voxel = positive(geometry.get("acquisition_voxel_um", 0), "verified acquisition voxel size")
    if not geometry.get("voxel_size_source") or geometry.get("centreline_audit_accepted") is not True:
        raise ValueError("Record voxel_size_source and centreline_audit_accepted after checking geometry_audit.json")
    if not roots:
        raise ValueError("Explicit left/right root-node mapping is required")
    component_sets = [{graph.segment(s)[key] for s in c for key in ("node1", "node2")}
                      for c in graph.components()]
    selected_components = []
    root_edges = []
    for name, data in roots.items():
        node = int(data["node"])
        if node not in graph.nodes or graph.degree(node) != 1:
            raise ValueError(f"Invalid inlet root {name}: {node}")
        root_edges.extend(graph.node_segments(node))
        # components() returns node sets.
        component = next((c for c in component_sets if node in c), None)
        if component is None or any(component == c for c in selected_components):
            raise ValueError("Each independent tree must have exactly one configured inlet")
        selected_components.append(component)
    if len(selected_components) != len(component_sets):
        raise ValueError("Every graph component needs an explicit tree/root mapping")
    argv = ["radius-perimeter", str(graph_path), "--seg", str(segmentation), "--voxel-um", str(voxel),
            "--out", str(measured_path), "--gate-voxels", "0", "--section-filter"]
    settings = geometry.get("radius_settings", {})
    allowed = {"max_half", "max_radius_factor", "workers", "min_blob_voxels", "grow_radii", "fallback_policy"}
    if set(settings)-allowed:
        raise ValueError(f"Unsupported radius settings: {set(settings)-allowed}")
    for name, value in settings.items():
        argv.extend(["--"+name.replace("_", "-"), str(value)])
    for edge in root_edges:
        argv.extend(["--root-edge", str(edge)])
    args = edit.build_parser().parse_args(argv)
    labels, frame, _ = open_segmentation(segmentation, voxel, output/"cache")
    binning = np.asarray(frame.seg_spacing)/voxel
    expected_bin = np.asarray(geometry["segmentation_binning"], dtype=float)
    if not np.allclose(binning, expected_bin, rtol=1e-5):
        raise ValueError(f"Segmentation binning mismatch: frame {binning}, configured {expected_bin}")
    stamp = edit._correct_units(graph, frame, args)
    # Bounds check before expensive remeasurement. The section estimator provides
    # the actual foreground/ownership checks, recorded per point afterwards.
    coords = np.array([p[:3] for p in graph.points.values()])
    ijk = frame.um_to_seg(coords)
    if np.any(ijk < -1) or np.any(ijk > np.asarray(info.dims)+1):
        raise ValueError("Graph points lie outside the matching segmentation frame")
    # Reuse its parser and estimator, preserving in-memory stable IDs.
    log = output/"radius_perimeter.log"
    signature = inspect.signature(rp.measure_radii)
    effective = {k: getattr(args, k) for k in signature.parameters
                 if k not in ("graph", "frame", "labels") and hasattr(args, k) and getattr(args, k) is not None}
    effective.update(root_edges=root_edges, tangent_search_degrees=args.tangent_search_deg)
    if subset_ids is not None:
        effective["_segment_ids"] = subset_ids
    before_scope = np.concatenate([graph.radii(s) for s in (subset_ids if subset_ids is not None else graph.segment_ids())])
    def progress(done, total):
        write_json(output/"radius_progress.json", dict(completed_segments=done, total_segments=total))
        if done == total or done % max(1, total//20) == 0:
            print(f"radius-perimeter: {done}/{total} segments", flush=True)
    result = rp.measure_radii(graph, frame, labels, progress=progress, **effective)
    log.write_text(result.describe(), encoding="utf-8")
    rp.apply_radii(graph, result)
    edit._save(graph, args.out, args.graph, voxel_um=stamp)
    corrected = graph
    failures, edges, coverage = [], {}, []
    overrides = geometry.get("hydraulic_overrides", {})
    for sid in (subset_ids if subset_ids is not None else corrected.segment_ids()):
        seg = corrected.segment(sid)
        xyz = corrected.coords(sid)
        arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        trusted = trusted_radii(corrected, sid)
        modes = corrected.triple.point_attrs.get("radius_resolution_mode", {})
        trusted &= np.array([modes.get(pid) in (rp.DIRECT_PLANE, rp.OWNED_PLANE)
                             for pid in seg["point_ids"]])
        point_coverage = dict(segment=sid, points=len(arclength), trusted_measurements=int(trusted.sum()),
                              fallback_or_modelled=int((~trusted).sum()), hydraulic_status="unsupported")
        coverage.append(point_coverage)
        if len(arclength) < 2 or np.any(np.diff(arclength) <= 0):
            failures.append(dict(segment=sid, reason="duplicate/insufficient centreline samples"))
            continue
        try:
            r, provenance = hydraulic_profile(arclength, corrected.radii(sid), trusted,
                (corrected.degree(seg["node1"]) >= 3, corrected.degree(seg["node2"]) >= 3),
                float(max(frame.seg_spacing)), overrides.get(str(sid)))
        except ValueError as exc:
            failures.append(dict(segment=sid, reason=str(exc), unsupported_spans=getattr(exc, "spans", [])))
            continue
        point_coverage.update(hydraulic_status="supported", **provenance)
        attrs = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v
                 for k, v in seg.items() if k not in ("id", "node1", "node2", "point_ids")
                 and isinstance(v, (str, int, float, np.integer, np.floating))}
        edges[sid] = dict(id=str(sid), source_segment_id=sid, u=str(seg["node1"]), v=str(seg["node2"]),
                         s_m=(arclength*1e-6).tolist(), xyz_m=(xyz*1e-6).tolist(), radius_m=(r*1e-6).tolist(),
                         surface_radius_m=(corrected.radii(sid)*1e-6).tolist(),
                         point_attributes={k: [float(values[pid]) if pid in values and np.isfinite(values[pid]) else None
                                               for pid in seg["point_ids"]]
                                           for k, values in corrected.triple.point_attrs.items()},
                         hydraulic_supported=True, attributes=attrs, radius_provenance=provenance)
    audit.update(status="unsupported_hydraulic_spans" if failures else "prepared",
                 failures=failures, coverage=coverage, effective_cli=argv, acquisition_voxel_um=voxel,
                 effective_estimator_settings={k: effective.get(k, v.default) for k, v in signature.parameters.items()
                    if v.default is not inspect.Parameter.empty and k != "progress"},
                 estimator_sha256=digest(inspect.getfile(rp)),
                 corrected_sha256=digest(measured_path), measurement_output=str(measured_path),
                 measurement_segment_ids=subset_ids, network_generated=False,
                 measured_scope_before_quantiles_um=np.quantile(before_scope, [0,.05,.5,.95,1]).tolist(),
                 measured_scope_after_quantiles_um=np.quantile(np.concatenate([corrected.radii(s)
                     for s in (subset_ids if subset_ids is not None else corrected.segment_ids())]), [0,.05,.5,.95,1]).tolist(),
                 after_radius_quantiles_um=np.quantile([p[3] for p in corrected.points.values()], [0,.05,.5,.95,1]).tolist())
    write_json(output/"geometry_audit.json", audit)
    if subset_ids is not None:
        # An explicitly selected audit is never exported as a fully measured network.
        audit["status"] = "subset_has_unsupported_spans" if failures else "subset_measurement_complete"
        write_json(output/"geometry_audit.json", audit)
        return audit
    if failures:
        raise ValueError(f"{len(failures)} unsupported hydraulic segments; inspect geometry_audit.json")
    trees = []
    for (name, data), component in zip(roots.items(), selected_components):
        subset = [e for sid, e in edges.items() if corrected.segment(sid)["node1"] in component]
        terminals = []
        for n in component:
            if corrected.degree(n) == 1 and n != int(data["node"]):
                sid = next(iter(corrected.node_segments(n)))
                edge = edges[sid]
                terminals.append(dict(node=str(n), radius_m=edge["radius_m"][0 if edge["u"] == str(n) else -1]))
        tree = dict(id=name, root=str(data["node"]), edges=subset, terminals=terminals, voxel_um=voxel,
                    source_graph=str(graph_path))
        if "retained_edge_ids" in data:
            tree.update(retained_edge_ids=list(map(str, data["retained_edge_ids"])), outlets=data["outlets"])
        else:
            tree = truncate_tree(tree, geometry.get("clinical_radius_m", 0.0005))
        trees.append(tree)
        write_network_graph(tree, output/f"{name}_hydraulic.am")
        write_network_graph(tree, output/f"{name}_clinical.am", retained_only=True)
        write_network_graph(tree, output/f"{name}_full_surface.am", surface=True)
        write_network_graph(tree, output/f"{name}_clinical_surface.am", retained_only=True, surface=True)
        if surfaces:
            for suffix in ("full_surface", "clinical_surface"):
                subprocess.run([sys.executable, "-m", "hipct_seg_debug.edit", "surface",
                    str(output/f"{name}_{suffix}.am"), "--out-dir", str(output/f"{name}_{suffix}_surface")], check=True)
    network = dict(schema="coronary-rcr-network/1", trees=trees,
                   provenance={k: audit[k] for k in ("graph_sha256", "segmentation_sha256", "corrected_sha256", "estimator_sha256", "acquisition_voxel_um")})
    write_json(output/"network.json", network)
    audit["network_generated"] = True
    write_json(output/"geometry_audit.json", audit)
    return audit
