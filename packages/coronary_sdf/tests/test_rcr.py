import copy
import json

import numpy as np
import pytest

from coronary_sdf.rcr.calibration import calibrate_case, physiology
from coronary_sdf.rcr.dynamics import RCRState, impedance, parameters, step
from coronary_sdf.rcr.network import calibrate_scale, partition, segment_resistance, solve
from coronary_sdf.rcr.prepare import hydraulic_profile, truncate_tree
from coronary_sdf.rcr.cfx import checkpoint, export


def edge(key, u, v, radius=0.001, length=0.01):
    return dict(id=key, u=u, v=v, s_m=[0, length], radius_m=[radius]*2,
                xyz_m=[[0, 0, 0], [length, 0, 0]], hydraulic_supported=True)


def case():
    tree = dict(id="left", root="0", edges=[edge("0", "0", "1"),
        edge("1", "1", "2", .0008), edge("2", "1", "3", .0006),
        edge("3", "2", "4", .0004), edge("4", "3", "5", .0003)],
        retained_edge_ids=["0", "1", "2"],
        outlets=[dict(id="A", node="2", radius_m=.0008), dict(id="B", node="3", radius_m=.0006)],
        terminals=[dict(node="4", radius_m=.0004), dict(node="5", radius_m=.0003)])
    right = copy.deepcopy(tree)
    right["id"] = "right"
    cfg = dict(tree_fractions={"left": .7, "right": .3}, rcr=dict(mode="fixed", alpha=.1, tau_distal_s=1.),
        structured_tree=dict(daughter_radius_ratios=[.7, .65], length_radius_ratio=5,
            terminal_radius_m=.0002, max_generations=12, max_segments=10000))
    return {"trees": [tree, right]}, cfg


def test_baseline_and_profile_integral():
    p = physiology({})
    assert p["flow_m3_s"]*6e7 == pytest.approx(73.29920556)
    assert p["whole_resistance_Pa_s_m3"]/.7 == pytest.approx(1.371950477e10)
    e = edge("e", "0", "1")
    assert segment_resistance(e) == pytest.approx(8*.0035*.01/(np.pi*.001**4))
    e["radius_m"] = [.001, .002]
    assert segment_resistance(e) == pytest.approx(8*.0035/np.pi*.01*(.001**-3-.002**-3)/(3*.001))
    e["hydraulic_supported"] = False
    with pytest.raises(ValueError, match="Unsupported"):
        segment_resistance(e)


def test_loop_network_and_conservation():
    # A symmetric Wheatstone bridge has zero bridge flow and R=1+2=3.
    edges = [("a", "r", "a", 1), ("b", "r", "b", 1), ("c", "a", "t", 1),
             ("d", "b", "t", 1), ("bridge", "a", "b", 7)]
    result = solve(edges, "r", {"t": 2}, 11, 5)
    assert result.inlet_flow == pytest.approx(2)
    assert result.edge_flow["bridge"] == pytest.approx(0, abs=1e-12)
    assert result.relative_mass_error < 1e-12


def test_feasibility_and_all_methods():
    assert calibrate_scale(lambda x: 1/(2+x), .25) == pytest.approx(2)
    with pytest.raises(ValueError, match="Infeasible"):
        calibrate_scale(lambda x: 1/(2+x), 1)
    network, cfg = case()
    result = calibrate_case(network, cfg, ["diameter", "hipct", "structured", "legacy-giessen"])
    for tree in result["results"]:
        assert tree["achieved_flow_m3_s"] == pytest.approx(tree["target_flow_m3_s"], rel=1e-7)
        assert sum(o["flow_fraction"] for o in tree["outlets"]) == pytest.approx(1)
        for o in tree["outlets"]:
            assert o["Rp_Pa_s_per_m3"]+o["Rd_Pa_s_per_m3"] == pytest.approx(o["R_T_Pa_s_per_m3"])
            assert o["initial_outlet_pressure_pa"] == pytest.approx(o["initial_pc_pa"]+o["Rp_Pa_s_per_m3"]*o["flow_m3_s"])


def test_shared_distal_rejected():
    network, _ = case()
    t = network["trees"][0]
    t["edges"].append(edge("shared", "4", "5"))
    with pytest.raises(ValueError, match="Shared"):
        partition(t, .0035)


def test_hydraulic_gaps_and_junction_calibre():
    r, report = hydraulic_profile(np.arange(5.), [10., 1., .01, 1., 20.],
                                  [True, False, False, False, True], (False, False), 1.)
    assert np.all((r >= 10) & (r <= 20))
    assert report["modelled"] == 3
    with pytest.raises(ValueError, match="Unsupported"):
        hydraulic_profile(np.arange(3.), [1., 10., 10.], [False, True, True], (False, False), 1.)
    r, _ = hydraulic_profile(np.arange(3.), [.01, 10., 10.], [False, True, True], (True, False), 1.)
    np.testing.assert_array_equal(r, [10., 10., 10.])


def test_truncation_splits_a_sampled_edge():
    e = edge("a", "0", "1")
    e.update(s_m=[0, .005, .01], radius_m=[.001, .0004, .0003], xyz_m=[[0,0,0],[.005,0,0],[.01,0,0]])
    tree = truncate_tree(dict(root="0", edges=[e]), .0005)
    assert tree["retained_edge_ids"] == ["a:prox"]
    assert tree["edges"][1]["s_m"] == [0., .005]


def test_equilibrium_reverse_flow_outlet_independence_and_restart():
    rp, rd, c, pv, q = np.array([2.,3.]), np.array([8.,7.]), np.array([.1,.2]), 5., np.array([1.,-1.])
    pc = pv+rd*q
    updated, pressure = step(pc, q, .02, rp, rd, c, pv)
    np.testing.assert_allclose(updated, pc)
    np.testing.assert_allclose(pressure, pv+(rp+rd)*q)
    state = RCRState(pc, q)
    for i in range(10):
        state.begin(i, [2., -1.])
        first = state.trial(.02, rp, rd, c, pv)
        np.testing.assert_array_equal(first, state.trial(.02, rp, rd, c, pv))
        state.commit()
    restored = RCRState(**state.checkpoint())
    for s in (state, restored):
        s.begin(10, [2., -1.])
        s.trial(.02, rp, rd, c, pv)
        s.commit()
    np.testing.assert_array_equal(state.pc, restored.pc)
    assert state.pc[1] == pytest.approx(pc[1])
    with pytest.raises(ValueError):
        state.commit()


def test_transient_time_step_convergence_and_impedance_fit():
    errors = []
    for dt in (.1, .05, .025):
        pc = 5.
        for _ in range(round(1/dt)):
            pc, _ = step(pc, 1., dt, 2., 8., .1, 5.)
        errors.append(abs(pc-(5+8*(1-np.exp(-1/.8)))))
    assert errors[2] < .6*errors[1] < .36*errors[0]
    freq = np.arange(1, 11)
    z = impedance(2*np.pi*freq, 2e9, 8e9, 1e-10)
    fit = parameters(1e10, .001, np.pi*1e-6, 1050, dict(mode="impedance"),
                     np.column_stack((freq, z.real, z.imag)))
    assert fit["Rp_Pa_s_per_m3"] == pytest.approx(2e9, rel=1e-5)
    assert fit["C_m3_per_Pa"] == pytest.approx(1e-10, rel=1e-5)


def test_export_preserves_values_and_mapping(tmp_path):
    network, cfg = case()
    result = calibrate_case(network, cfg, ["hipct"])
    mesh = tmp_path/"mesh.msh"
    # Zone-table-only fixture validates text export, not a usable CFD mesh.
    mesh.write_text('(45 (1 fluid lumen)())\n(45 (2 wall wall)())\n(45 (3 velocity-inlet inlet)())\n(45 (4 pressure-outlet a)())\n(45 (5 pressure-outlet b)())\n')
    cfg["cfx"] = dict(mesh=str(mesh), mesh_units="mm", inlet_regions={"left":"inlet"},
                      outlet_regions={"left":{"A":"a", "B":"b"}})
    out = tmp_path/"export"
    out.mkdir()
    status = export(result, cfg, "left", "hipct", out, True)
    assert status["text_generated"] and status["resolved_text_generated"]
    assert not status["def_generated"] and not status["solver_validated"]
    text = (out/"case.ccl").read_text()
    assert 'BOUNDARY: Outlet0' in text and 'BOUNDARY: Outlet00' not in text
    assert 'DENSITY = 1050' in text
    assert 'area@Inlet' in text
    for row in result["results"][0]["outlets"]:
        assert f'{row["Rp_Pa_s_per_m3"]:.17e}' in text
        assert f'{row["Rd_Pa_s_per_m3"]:.17e}' in text
        assert f'{row["C_m3_per_Pa"]:.17e}' in text
    cfg["cfx"]["outlet_regions"]["left"]["B"] = "a"
    with pytest.raises(ValueError, match="unique"):
        export(result, cfg, "left", "hipct", tmp_path/"bad", True)


def test_checkpoint_requires_completed_matching_step(tmp_path):
    path = tmp_path/"state.dat"
    path.write_text('2 12 .001\n0 100 1e-7\n1 200 -1e-7\nCOMMIT 12\n')
    assert checkpoint(path, 2, 12, .001) == [(100,1e-7),(200,-1e-7)]
    with pytest.raises(ValueError):
        checkpoint(path, 2, 13, .001)
    path.write_text('2 12 .001\n0 100 1e-7\n')
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint(path, 2, 12, .001)


def test_prepare_cylinder_keeps_measurement_provenance(tmp_path):
    from coronary_sdf.rcr.prepare import prepare
    from hipct_seg_debug.edit.adapter import Triple, to_spatial_graph
    from hipct_seg_debug.edit.amira_write import write_spatial_graph
    from hipct_seg_debug.amira import read_spatial_graph
    z, y, x = np.indices((32, 32, 48))
    volume = (((z-16)**2+(y-16)**2) <= 36).astype('uint8')
    seg = tmp_path/'mask.am'
    seg.write_bytes(b'# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\ndefine Lattice 48 32 32\n'
        b'Parameters { BoundingBox 0 1504 0 992 0 992, CoordType "uniform" }\n'
        b'Lattice { byte Labels } @1\n\n@1\n'+volume.tobytes())
    points = {i: (float(x)*32, 512., 512., 192.) for i, x in enumerate(range(8,40,2))}
    triple = Triple({0: (*points[0][:3],1), 1: (*points[15][:3],1)}, points,
                    [dict(id=0,node1=0,node2=1,point_ids=list(points),MeanRadius=192.,AnatomyId=7)])
    path = tmp_path/'graph.am'
    write_spatial_graph(to_spatial_graph(triple),path,voxel_um=32.)
    cfg = dict(geometry=dict(graph=str(path),segmentation=str(seg),roots={"left":{"node":0}},
        acquisition_voxel_um=32.,voxel_size_source="synthetic exact",centreline_audit_accepted=True,
        segmentation_binning=[1,1,1],radius_settings={"workers":1}))
    output=tmp_path/'prepared'
    output.mkdir()
    before=path.read_bytes()
    report=prepare(cfg,output)
    assert path.read_bytes() == before
    assert not (tmp_path/'.hipct_cache').exists()
    assert report['status']=='prepared'
    data=json.loads((output/'network.json').read_text())
    e=data['trees'][0]['edges'][0]
    assert e['attributes']['AnatomyId']==7
    assert 'radius_source' in e['point_attributes']
    corrected=read_spatial_graph(output/'left_clinical_surface.am')
    assert corrected.thickness.min()>0
    subset=tmp_path/'subset'
    subset.mkdir()
    cfg['geometry']['measurement_segment_ids']=[0]
    partial=prepare(cfg,subset)
    assert partial['status']=='subset_measurement_complete'
    assert not partial['network_generated'] and not (subset/'network.json').exists()
    assert (subset/'measured_subset.am').is_file()


def test_sinusoidal_impedance_of_lagged_integrator_converges():
    errors = []
    for dt in (.01, .005, .0025):
        n = round(1/dt)
        times = np.arange(1, 20*n+1)*dt
        pc = 0.
        pressure = []
        for t in times:
            q = np.cos(2*np.pi*(t-dt))
            pc, value = step(pc, q, dt, 2., 8., .1, 0.)
            pressure.append(float(value))
        z = 2*np.mean(np.array(pressure[-n:])*np.exp(-2j*np.pi*times[-n:]))
        errors.append(abs(z-impedance(2*np.pi, 2., 8., .1)))
    assert errors[2] < .55*errors[1] < .303*errors[0]


def test_more_than_100_outlets_and_restart_export(tmp_path):
    network, cfg = case()
    results = calibrate_case(network, cfg, ["diameter"])
    row = results["results"][0]["outlets"][0]
    q = results["results"][0]["target_flow_m3_s"]/123
    rows = [{**row, "outlet_id": f"branch{k}", "flow_m3_s": q,
             "flow_fraction": 1/123, "initial_pc_pa": row["distal_pressure_pa"]+q*row["Rd_Pa_s_per_m3"],
             "initial_outlet_pressure_pa": row["distal_pressure_pa"]+q*row["R_T_Pa_s_per_m3"]}
            for k in range(123)]
    results["results"][0]["outlets"] = rows
    cfg["cfx"] = dict(inlet_regions={"left":"inlet"},
                      outlet_regions={"left":{r["outlet_id"]: f"region{k}" for k,r in enumerate(rows)}})
    out = tmp_path/'many'
    out.mkdir()
    status = export(results, cfg, 'left', 'diameter', out, True)
    assert 'BOUNDARY: Outlet122' in (out/'case.ccl').read_text()
    states=tmp_path/'checkpoint.dat'
    states.write_text('123 42 .001\n'+''.join(f'{k} {1000+k} 1e-8\n' for k in range(123))+'COMMIT 42\n')
    cfg['cfx'].update(restart=str(states),restart_step=42,restart_mapping_sha256=status['mapping_sha256'])
    restart=tmp_path/'restart'
    restart.mkdir()
    export(results,cfg,'left','diameter',restart,True)
    text=(restart/'case.ccl').read_text()
    assert 'RCR_PC_122 = 1122' in text
    assert 'RCR_STEP = 42' in text


def test_comparison_separates_unqualified_cfd():
    from coronary_sdf.rcr.compare import compare
    network, cfg = case()
    result = compare(calibrate_case(network,cfg,['diameter','hipct']),cfg)
    assert result['cfd_status']=='not_supplied'
    assert not result['cfd_comparisons']
    assert len(result['network_flow_comparisons'])==4


def test_legacy_giessen_uses_clean_branch_calibre_not_junction_tip():
    from coronary_sdf.rcr.calibration import legacy_fractions
    network, _ = case()
    tree = network['trees'][0]
    tree['edges'][1]['radius_m'] = [.002]*10+[.0005]*10
    tree['edges'][2]['radius_m'] = [.0005]*20
    fractions = legacy_fractions(tree, None)
    assert fractions == pytest.approx({'2': .5, '3': .5})
