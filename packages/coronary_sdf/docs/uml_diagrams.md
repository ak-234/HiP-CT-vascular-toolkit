# `coronary_sdf` — UML diagrams

Architecture reference for the `coronary_sdf` package: a coronary-lumen surface-reconstruction
pipeline (Amira spatial-graph → smooth-min capsule signed-distance field → iso-surface mesh) plus
downstream CFD boundary-condition and wall-shear-stress (WSS) post-processing tools.

> **Reading note.** The package is **function-oriented**, not object-oriented: no class
> inheritance, no enums, and only six dataclasses (all plain data holders). The load-bearing
> structure therefore lives in the **module dependency graph** (Section 1) and the **pipeline data
> flow** (Section 2). The class diagram (Section 3) covers the dataclasses; the API maps
> (Section 4) list each module's public free functions.
>
> These diagrams are Mermaid. They render in VS Code's Markdown preview (`Ctrl+Shift+V`) and on
> GitHub. Scope is production modules only — the `_smoke_test`, `_probe_multifurc`, and `_test_*`
> scripts are excluded.

---

## 1. Module dependency graph (UML package diagram)

Each node is a module. **Solid arrow = eager top-level import; dashed arrow = lazy in-function
import.** Every module also imports `config` — those edges are omitted to keep the graph readable,
and `config` is shown as a shared node. Note the `smoothing ↔ topology` cycle: `smoothing` imports
`topology` eagerly (`smoothing.py:21`), while `topology` imports `smoothing` lazily
(`topology.py:537`) to break the import cycle.

```mermaid
graph TD
    classDef cfg fill:#fde68a,stroke:#b45309,color:#000;
    classDef standalone fill:#e5e7eb,stroke:#6b7280,color:#000,stroke-dasharray:4 3;

    config["config<br/><i>(imported by all)</i>"]:::cfg

    subgraph Foundation
        parse_amira
        splines
    end

    subgraph GraphProcessing["Graph processing"]
        topology
        pruning
        smoothing
        bif_trim
    end

    subgraph Core["SDF → Mesh core"]
        capsules
        sdf_field
        mesh_extract
        mesh_repair
        region_vtk
        viz
    end

    subgraph Orchestration
        __main__
        pipeline
    end

    subgraph Tools["CFD / WSS tools"]
        flow_fractions
        resistance_based_outlet_BC["resistance_based_outlet_BC"]:::standalone
        epicardial_annotation
        manual_prune
        wss_postprocess
        wss_contour_compare
    end

    subgraph Alt["Standalone"]
        tube_union
    end

    %% ── Eager imports ──
    __main__ --> pipeline
    pruning --> topology
    smoothing --> topology
    bif_trim --> topology
    region_vtk --> topology
    sdf_field --> capsules
    sdf_field --> splines
    mesh_repair --> mesh_extract

    pipeline --> parse_amira
    pipeline --> pruning
    pipeline --> topology
    pipeline --> smoothing
    pipeline --> splines
    pipeline --> bif_trim
    pipeline --> capsules
    pipeline --> sdf_field
    pipeline --> mesh_extract
    pipeline --> mesh_repair
    pipeline --> viz
    pipeline --> region_vtk

    flow_fractions --> parse_amira
    flow_fractions --> pruning
    flow_fractions --> topology
    flow_fractions --> smoothing

    epicardial_annotation --> parse_amira
    epicardial_annotation --> splines
    epicardial_annotation --> topology
    epicardial_annotation --> flow_fractions
    epicardial_annotation --> pruning

    manual_prune --> parse_amira
    manual_prune --> flow_fractions
    manual_prune --> splines
    manual_prune --> epicardial_annotation

    wss_postprocess --> parse_amira
    wss_postprocess --> splines
    wss_postprocess --> flow_fractions
    wss_postprocess --> topology
    wss_postprocess --> epicardial_annotation

    wss_contour_compare --> flow_fractions
    wss_contour_compare --> splines
    wss_contour_compare --> wss_postprocess

    tube_union --> parse_amira
    tube_union --> splines
    tube_union --> smoothing

    %% ── Lazy (in-function) imports ──
    topology -.-> smoothing
    sdf_field -.-> topology
    viz -.-> splines
    flow_fractions -.-> viz
    flow_fractions -.-> splines
    epicardial_annotation -.-> pipeline
    manual_prune -.-> pipeline
    wss_contour_compare -.-> viz
```

`resistance_based_outlet_BC` has no internal imports (stdlib only) — it is coupled to the rest only
through a CSV file it reads (see Section 2). `parse_amira` depends on `config` alone.

---

## 2. Pipeline data-flow diagram

Files (parallelograms) and processing stages (rectangles), following the order documented in
`__init__.py`. Entry-point commands are rounded. The two boundary-condition scripts are linked
through the flow-fraction CSV; the WSS tools consume the annotation sidecar plus CFD output.

```mermaid
flowchart TD
    classDef file fill:#dbeafe,stroke:#1d4ed8,color:#000;
    classDef stage fill:#dcfce7,stroke:#15803d,color:#000;
    classDef entry fill:#fce7f3,stroke:#be185d,color:#000;
    classDef ext fill:#f3f4f6,stroke:#6b7280,color:#000,stroke-dasharray:4 3;

    amInput[/"Amira spatial graph<br/>*.am / *.am.xml"/]:::file

    cli(["python -m coronary_sdf"]):::entry
    parse["parse_amira.parse_xml"]:::stage
    topo["topology"]:::stage
    smooth["pruning / smoothing"]:::stage
    spl["splines"]:::stage
    bt["bif_trim"]:::stage
    caps["capsules.build_capsules"]:::stage
    sdf["sdf_field.evaluate_sdf"]:::stage
    mext["mesh_extract"]:::stage
    mrep["mesh_repair"]:::stage
    rvtk["region_vtk"]:::stage

    stl[/"lumen_bspline*.stl"/]:::file
    vtk[/"lumen_bspline*.vtk"/]:::file
    rvtkf[/"lumen_bspline*_regions.vtk"/]:::file

    meshing["external meshing"]:::ext
    msh[/"ANSYS .msh"/]:::file

    ff["flow_fractions.run"]:::stage
    ffcsv[/"giessen_cfx_outlet_<br/>flow_fractions.csv"/]:::file
    ffccl[/"CFX .ccl (flow split)"/]:::file
    renmsh[/"renamed .msh"/]:::file

    rbo["resistance_based_outlet_BC"]:::stage
    rccl[/"resistance pressure-outlet .ccl"/]:::file

    epi["epicardial_annotation /<br/>manual_prune (picker)"]:::stage
    sidecar[/"epicardial.json sidecar"/]:::file
    pervessel[/"per-vessel / pruned *.am.xml"/]:::file

    wssin[/"CFD wall-node WSS CSV"/]:::file
    wpp["wss_postprocess"]:::stage
    wssout[/"wss_all.csv,<br/>wss_&lt;vessel&gt;.csv/.png"/]:::file

    wcc["wss_contour_compare"]:::stage
    wccout[/"wss_seg_long.csv,<br/>wss_seg_{max,min}_wide.csv"/]:::file

    tube["tube_union (alt path,<br/>MeshLib boolean union)"]:::stage
    tstl[/"single .stl"/]:::file

    %% Main pipeline
    amInput --> cli --> parse --> topo --> smooth --> spl --> bt --> caps --> sdf --> mext --> mrep --> rvtk
    mrep --> stl
    mrep --> vtk
    rvtk --> rvtkf
    stl --> meshing --> msh

    %% CFD boundary conditions
    amInput --> ff
    msh --> ff
    ff --> ffcsv
    ff --> ffccl
    ff --> renmsh
    ffcsv --> rbo --> rccl

    %% Annotation + WSS
    amInput --> epi
    epi --> sidecar
    epi --> pervessel
    pervessel -.->|"can re-invoke run_pipeline"| cli

    amInput --> wpp
    sidecar --> wpp
    wssin --> wpp --> wssout

    amInput --> wcc
    sidecar --> wcc
    wssin --> wcc --> wccout

    %% Independent alternative
    amInput --> tube --> tstl
```

---

## 3. Dataclass class diagram

The complete class model: one frozen config dataclass plus five data holders. Fields and types are
taken directly from the source. Dashed dependencies show which factory function builds each holder
and how `evaluate_sdf` consumes and produces them.

```mermaid
classDiagram
    class SdfConfig {
        <<frozen dataclass — config>>
        +INPUT_PATH
        +INPUT_XML
        +INPUT_AM
        +OUTPUT_DIR
        +smoothing_params
        +radius_params
        +sdf_params
        +mesh_params
        +approx_100_scalar_fields
    }

    class CapsuleArrays {
        <<dataclass — capsules>>
        +starts : ndarray Nx3
        +ends : ndarray Nx3
        +radii_start : ndarray N
        +radii_end : ndarray N
        +seg_idx : ndarray N
        +midpoints : ndarray Nx3
        +tangents : ndarray Nx3
        +max_radii : ndarray N
        +tree : KDTree
        +arc_start : ndarray N
        +arc_end : ndarray N
        +seg_L : ndarray n_segs
        +cap_bif_at_start : ndarray bool
        +cap_bif_at_end : ndarray bool
        +n() int
    }

    class TerminalSet {
        <<dataclass — sdf_field>>
        +pos : ndarray | None
        +nrm : ndarray | None
        +rad : ndarray | None
        +tree : KDTree | None
    }

    class BifurcationSet {
        <<dataclass — sdf_field>>
        +positions : ndarray
        +radii : ndarray
        +tree : KDTree | None
        +node_ids : ndarray
    }

    class Grid {
        <<dataclass — sdf_field>>
        +bbox_min : ndarray
        +bbox_max : ndarray
        +voxel_size : float
        +dims : ndarray len3
        +x : ndarray
        +y : ndarray
        +z : ndarray
        +n_voxels() int
    }

    class SdfVolume {
        <<dataclass — sdf_field>>
        +sdf : ndarray float32
        +path_vol : ndarray | None
        +blend_weight_vol : ndarray | None
        +nb_idx : ndarray | None
        +diag : dict | None
    }

    class evaluate_sdf {
        <<function — sdf_field>>
    }

    CapsuleArrays ..> KDTree : indexes midpoints
    evaluate_sdf ..> CapsuleArrays : uses
    evaluate_sdf ..> BifurcationSet : uses
    evaluate_sdf ..> TerminalSet : uses
    evaluate_sdf ..> Grid : uses
    evaluate_sdf ..> SdfVolume : returns
    Grid ..> CapsuleArrays : compute_grid()
```

Factory functions: `default_config() → SdfConfig`, `build_capsules() → CapsuleArrays`,
`build_terminal_set() → TerminalSet`, `find_bifurcations() → BifurcationSet`,
`compute_grid() → Grid`, `evaluate_sdf() → SdfVolume`.

---

## 4. Per-module API maps

Because behaviour lives in free functions rather than methods, each module is drawn as a box whose
"methods" are its public functions. Split by layer for legibility.

### 4a. Parsing & graph processing

```mermaid
classDiagram
    class parse_amira {
        +parse_am()
        +parse_xml()
        +parse_nodes()
        +parse_points()
        +parse_segments()
        +find_degenerate_segments()
    }
    class centreline_reconnection {
        +find_connected_components()
        +split_by_graph()
        +merge_degree2_segments()
        +merge_split_multifurcations()
        +node_id_canon_map()
        +bridge_centerline_gaps()
    }
    class topology {
        +build_nx_tree()
        +label_centreline_topology_aware()
        +label_capsules_by_cross_section()
        +build_directed_topology()
    }
    class pruning {
        +segment_mean_radius()
        +segment_arc_length()
        +report_segment_radius_range()
        +prune_short_terminal_nubs()
        +prune_by_radius()
    }
    class smoothing {
        +smooth_centerline_savgol()
        +smooth_centerline_bspline()
        +densify_sparse_segments()
        +smooth_segment_centerlines()
        +limit_centerline_curvature()
        +smooth_segment_radii()
        +smooth_radius_transitions()
        +prune_terminal_shrink()
        +prune_bifurcation_shrink()
    }
```

### 4b. Geometry → SDF → mesh

```mermaid
classDiagram
    class splines {
        +compute_frenet_frame()
        +prepare_segment_spline()
        +branch_tangent_at_node()
    }
    class capsules {
        +clamp_terminal_capsule_radii()
        +build_capsules()
    }
    class bif_trim {
        +taper_bifurcation_carina()
    }
    class sdf_field {
        +smooth_min_exp()
        +smooth_min_poly_pair()
        +soft_cap()
        +adaptive_smin_k()
        +collect_endpoint_info()
        +build_terminal_set()
        +find_bifurcations()
        +build_adjacency()
        +report_non_adjacent_proximity()
        +compute_grid()
        +build_narrow_band()
        +evaluate_sdf()
    }
    class mesh_extract {
        +fast_contour_zero()
        +mesh_from_sdf_poisson()
        +mesh_from_sdf_meshlib()
        +drop_nonfinite_vertices()
        +extract_isosurface()
        +cut_non_adjacent_bridges()
        +create_flat_caps()
        +extract_triangle_faces()
    }
    class mesh_repair {
        +fast_clean_triangle_mesh()
        +run_pymeshfix()
        +repair_mesh()
        +radius_constrained_taubin()
    }
    class region_vtk {
        +compute_bifurcation_levels()
        +label_surface_topology()
        +generate_region_vtk()
        +emit_region_vtk_for_surface()
    }
    class viz {
        +debug_show_mesh()
        +debug_show_capsule_tree()
        +debug_show_sdf_preview()
        +debug_show_blend_paths()
        +debug_show_capsule_tubes()
        +debug_show_smoothed_centerlines()
        +debug_show_blend_diagnostics()
    }
```

### 4c. Orchestration

```mermaid
classDiagram
    class pipeline {
        +generate_sdf_surface()
        +run_pipeline()
    }
    class __main__ {
        +cli_entry()
    }
    __main__ ..> pipeline : run_pipeline()
```

`__main__` is the package CLI entry point (`python -m coronary_sdf <input> <output_dir>`); its
`cli_entry` delegates to `pipeline.run_pipeline`.

### 4d. CFD / WSS tools

```mermaid
classDiagram
    class flow_fractions {
        +extract_msh_boundaries()
        +extract_msh_surface()
        +generate_cfx_ccl()
        +preprocess_topology()
        +smooth_graph()
        +branch_radius_mm()
        +compute_flow_fractions()
        +identify_mesh_trees()
        +aggregate_outlet_fractions()
        +match_all()
        +rename_msh_zones()
        +run()
        +main()
    }
    class resistance_based_outlet_BC {
        +read_outlets()
        +compute_resistances()
        +generate_resistance_ccl()
        +main()
    }
    class epicardial_annotation {
        +assign_vessel()
        +compute_vessel_ostia()
        +prune_by_radius_ratio()
        +prune_contained_leaves()
        +save_annotation()
        +load_or_create_annotation()
        +run_picker()
        +apply_manual_prune()
        +write_amira_xml()
        +main()
    }
    class manual_prune {
        +build_segment_colors()
        +write_removal_logs()
        +main()
    }
    class wss_postprocess {
        +read_wss_csv()
        +build_vessel_centrelines()
        +sample_stations()
        +sweep_station()
        +evaluate_vessel()
        +plot_vessel_wss()
        +main()
    }
    class wss_contour_compare {
        +parse_ratio_arg()
        +filter_vessels()
        +build_segment_table()
        +assign_and_sample()
        +write_wide_csv()
        +main()
    }
    class tube_union {
        +build_tube_solid()
        +sphere_mesh()
        +union_meshes()
        +run()
        +main()
    }
```
