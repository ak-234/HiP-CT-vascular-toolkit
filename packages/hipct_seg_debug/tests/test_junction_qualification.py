import hashlib
import json
from types import SimpleNamespace

from hipct_seg_debug import rle_write
from hipct_seg_debug.edit.__main__ import _save, _load
from hipct_seg_debug.edit.junction_qualification import evaluate
from .conftest_geometry import cylinder, make_frame, graph_from


def test_experimental_full_tree_runs_without_regional_qualification(tmp_path, monkeypatch):
    from hipct_seg_debug.edit import junction_qualification as jq
    source = tmp_path/'input.am'
    source.write_text('fixture')
    calls = []
    def evaluate(args, target=None):
        calls.append((target, args.measurement_only))
        return {'status': 'review_required'}
    monkeypatch.setattr(jq, 'evaluate', evaluate)
    out = tmp_path/'review'
    status = jq.main(['--graph', str(source), '--seg', str(source), '--out-dir', str(out),
                     '--experimental-full-tree', '--measurement-only'])
    assert calls == [(None, True)]
    assert status == 2
    report = json.loads((out/'summary.json').read_text())[0]
    assert report['status'] == 'review_required'
    assert report['experimental_full_tree'] is True


def test_regional_pipeline_writes_separate_measurements_profile_mesh_and_overlays(tmp_path):
    frame = make_frame((32, 32, 70))
    labels = cylinder((32, 32, 70), 5, 2, 68)
    graph = graph_from(frame.seg_to_um([[8, 16, 16], [62, 16, 16]]), [(0, 1, 20, 50.)])
    original, seg = tmp_path/'input.am', tmp_path/'labels.am'
    _save(graph, str(original), [], voxel_um=10.)
    rle_write.write_lattice(seg, labels, frame.seg_bbox_um)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    args = SimpleNamespace(graph=str(original), seg=str(seg), segment=[0],
        out_dir=str(tmp_path/'review'), fixed_node=[], workers=1, max_iterations=5,
        max_samples=8, cells_across_diameter=6., maximum_cells=200_000, geometry_only=False)
    report = evaluate(args, 0)
    directory = tmp_path/'review'/'region_0'
    assert report['status'] == 'qualified'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    assert (directory/'measured.am').exists()
    assert (directory/'reconstruction.am').exists()
    assert (directory/'surface.vtp').exists()
    assert (directory/'segment_0_overlay.png').exists()
    measured = _load([str(directory/'measured.am')])
    reconstruction = _load([str(directory/'reconstruction.am')])
    assert 'radius_reconstruction_um' not in measured.triple.point_attrs
    assert 'radius_reconstruction_um' in reconstruction.triple.point_attrs
    assert report['surface']['geometry_rewritten'] is False
    # Completed stages are usable on an identical rerun, without treating the
    # reconstruction output as the new measurement input.
    resumed = evaluate(args, 0)
    assert resumed['status'] == 'qualified'
    assert json.loads((directory/'report.json').read_text())['measurement_graph'] == [str(directory/'measured.am')]
    from hipct_seg_debug.edit.prepared_surface import run
    surface_args = SimpleNamespace(graph=[str(directory/'reconstruction.am')],
        prepared_report=str(directory/'report.json'), cells_across_diameter=6.,
        maximum_cells=200_000, out_dir=str(tmp_path/'direct_surface'))
    assert run(surface_args) == 0
