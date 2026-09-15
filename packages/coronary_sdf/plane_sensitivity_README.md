# PDF-style plane mesh-sensitivity analysis

This post-processing stage reuses the four completed global CFX results. It
does not generate a mesh or run the CFX solver.

Run it from `packages/coronary_sdf`:

```powershell
.\run_mesh_sensitivity_study.ps1 `
  -Config .\mesh_sensitivity_study_adaptive_surface_01d88a48e419.json `
  -Stage plane-sensitivity `
  -Resume
```

Outputs are under
`mesh_sensitivity_output_adaptive_surface_01d88a48e419\plane_sensitivity`:

- `geometry\strahler_planes.vtk/.vtm`: exact STL, cropped Amira graph, and five
  Strahler representative sections.
- `geometry\radius_bin_planes.vtk/.vtm`: the same anatomy and eight logarithmic
  radius-bin sections.
- `geometry\*_plane_definitions.csv` and `plane_definitions.json`: persistent
  section point, normal, area, radius, clearance, edge and validation metadata.
- `reports\per_plane_values.csv`: L1--L4 steady WSS, velocity and pressure.
- `reports\adjacent_differences_gci.csv`: adjacent changes, observed order,
  fine-grid GCI and metric pass/fail.
- `reports\strahler_*` and `radius_bin_*`: separate CSV, convergence plot and
  publication-style table outputs.
- `reports\plane_sensitivity_summary.json`: machine-readable study result.

Pressure changes are normalized by the matching finer mesh's inlet-to-opening
pressure drop. Steady WSS and velocity are intentionally not labelled TAWSS or
peak-systolic quantities.
