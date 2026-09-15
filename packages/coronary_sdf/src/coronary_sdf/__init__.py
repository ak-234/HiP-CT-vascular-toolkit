"""coronary_sdf -- Coronary lumen surface reconstruction (SDF pipeline only).

A modular refactor of the monolithic Coronary_lumen_octree.py, keeping
the smooth-min capsule SDF + anti-bridge carve + Screened Poisson
(or MeshLib/VTK marching-cubes) pipeline, plus experimental graph-implicit
and adaptive extraction backends. All other surface methods
(HRBF, loft, hybrid, meshsdf hybrid, hermite hybrid, octree/DC) are
dropped.

Pipeline (top-down):

    parse_amira -> topology -> smoothing / pruning -> splines
        -> capsules -> sdf_field -> mesh_extract -> mesh_repair
        -> region_vtk -> save

Entry point (CLI):

    python -m coronary_sdf <input.am.xml> <output_dir>

Each module corresponds to one pipeline step and exposes a small,
testable API. Configuration lives in ``coronary_sdf.config``.
"""

__version__ = "0.2.0"
