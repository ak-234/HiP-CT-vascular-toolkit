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

WSS is the **unweighted arithmetic mean of all finite WSS point values on the
associated segment wall**, rather than a plane intersection or short wall band.
The maximum uses the same point set. Wall points are assigned once to the nearest
cropped centreline edge, sampled at 0.1 mm; junctions are partitioned by nearest
edge. There is no midpoint-radius cutoff and no surface-area weighting. The
wall-only export excludes inlet/outlet caps. Reports include the edge ID, finite
point count, invalid-value count, and WSS scope. Fewer than eight finite points
is reported as insufficient support. Velocity and pressure retain their
cross-section definitions. Resumed runs recompute WSS and all derived reports.
Group WSS statistics describe the distribution of these segment means, with
equal weight per vessel; surface-area-weighted WSS group statistics are omitted.

`reports/mesh_quality_summary.csv` combines element counts and measured CFX
minimum, volume-average and maximum aspect ratio, mesh expansion factor and
orthogonality angle (degrees). Courant number is included as a solver diagnostic.
The accompanying `cfx_courant_mesh_quality.png/.pdf` shows these measurements
across mesh levels. Plane-hotspot overlap is reported for plane velocity only;
it does not describe the support of full-segment WSS.

Section validation rejects cuts whose maximum radius exceeds 1.5 times the
equivalent radius, or whose centroid is farther than one equivalent radius from
the centreline station. These cuts require review of the local tangent rather
than being accepted solely because their intersection loop is closed. Cached
CFD section values are reused only when the saved extraction plane ID, point,
normal and bounds match the current geometry.

In the local adaptive-surface study, `VES_E0287` was reviewed and corrected using
a three-local-diameter tangent window at its original arc-length midpoint. Its
short-window tangent followed a centreline kink and produced an elongated cut.
The corrected section uses its updated STL radius-bin assignment; WSS continues
to use the same full segment point cloud.
