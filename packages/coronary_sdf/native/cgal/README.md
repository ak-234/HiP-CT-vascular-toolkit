# Native CGAL Mesh_3 backend

This directory builds the optional top-level `coronary_sdf_cgal` extension.
It copies capsule, segment, junction, bounds, and sizing arrays into C++ once;
CGAL never calls back into Python while meshing.

## Windows prerequisites

1. Install Visual Studio 2022 Build Tools with the **Desktop development with
   C++** workload and a Windows SDK.
2. Use Conda 25.1 or newer. Older Conda releases can mis-detect recent Windows
   builds as `__win=0` and reject the `cgal-cpp` solve.
3. From the directory containing the `coronary_sdf` package, run:

```powershell
$env:CONDA_OVERRIDE_WIN = "10.0"
conda info | Select-String "__win"
conda env create --prefix ./coronary_sdf/.conda-cgal --file coronary_sdf/environment-cgal-win64.yml
conda activate ./coronary_sdf/.conda-cgal
python -m coronary_sdf.cgal_build_preflight
python -m pip install --no-build-isolation ./coronary_sdf/native/cgal
python -c "import coronary_sdf_cgal as c; print(c.CGAL_VERSION, c.COMPILER)"
```

The `conda info` check must report `__win=10.0=0` (or a newer 10.x build), not
`__win=0=0`. Conda 24.11 ignores this override; upgrade the Conda executable or
use a current project-local Miniforge installation before retrying.

The preflight intentionally fails with installation guidance when the MSVC
compiler, CMake, Ninja, or CGAL CMake package is missing. The reconstruction
pipeline treats absence/build failure as a hard backend failure and never
falls back to a dense or SciPy extractor.

## Qualification

Installing the extension does not enable the CFD CLI profile. Run the staged
acceptance benchmark at 6, 12, and 18 cells across local diameter. Only a
benchmark result that passes every required synthetic case and both LADAF
cases emits `cfd_qualified_profile.json`, which can then be supplied with
`python -m coronary_sdf --profile cfd --qualified-config <path>`.
