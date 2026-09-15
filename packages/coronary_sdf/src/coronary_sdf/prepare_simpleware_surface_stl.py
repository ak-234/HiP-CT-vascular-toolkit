"""Extract the active persisted Simpleware surface mesh from a SIP as STL.

This runs before ConsoleSimpleware.  It makes the exact surface stored in the
project, including its outlet cuts, authoritative for graph cropping and plane
validation instead of relying on the pre-import source STL.
"""

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import vtk


def _normalise(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _choose_entry(entries, surface_name):
    if not entries:
        raise RuntimeError("The SIP contains no SurfaceMesh*.vtu entry")
    if surface_name:
        wanted = _normalise(surface_name)
        matches = [entry for entry in entries if wanted in _normalise(entry.filename)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return max(matches, key=lambda entry: entry.file_size)
    if len(entries) == 1:
        return entries[0]
    return max(entries, key=lambda entry: entry.file_size)


def extract_surface(project, output, surface_name=None):
    project = Path(project).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not project.is_file():
        raise FileNotFoundError("Simpleware project not found: {}".format(project))
    if output.exists():
        raise FileExistsError("Refusing to overwrite surface cache: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with zipfile.ZipFile(project) as archive:
            entry = _choose_entry(
                [
                    item for item in archive.infolist()
                    if item.filename.startswith("SurfaceMesh")
                    and item.filename.lower().endswith(".vtu")
                ],
                surface_name,
            )
            with tempfile.NamedTemporaryFile(
                suffix=".vtu", delete=False, dir=str(output.parent)
            ) as temporary:
                temporary_path = Path(temporary.name)
                with archive.open(entry) as source:
                    while True:
                        block = source.read(8 * 1024 * 1024)
                        if not block:
                            break
                        temporary.write(block)

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(temporary_path))
        reader.Update()
        grid = reader.GetOutput()
        if grid is None or grid.GetNumberOfCells() == 0:
            raise RuntimeError("Embedded surface VTU is empty or unreadable")

        geometry = vtk.vtkGeometryFilter()
        geometry.SetInputData(grid)
        geometry.Update()
        polydata = geometry.GetOutput()
        non_triangles = sum(
            1 for index in range(polydata.GetNumberOfCells())
            if polydata.GetCellType(index) != vtk.VTK_TRIANGLE
        )
        if non_triangles:
            raise RuntimeError(
                "Embedded surface contains {} non-triangle cells".format(
                    non_triangles
                )
            )

        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(output))
        writer.SetFileTypeToBinary()
        writer.SetInputData(polydata)
        if writer.Write() != 1 or not output.is_file():
            raise RuntimeError("Failed to write extracted surface STL")
        print(
            "Embedded surface: {!r}; {:,} points, {:,} triangles -> {}".format(
                entry.filename,
                polydata.GetNumberOfPoints(),
                polydata.GetNumberOfCells(),
                output,
            ),
            flush=True,
        )
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    parser.add_argument("output")
    parser.add_argument("--surface-name", default=None)
    arguments = parser.parse_args()
    extract_surface(arguments.project, arguments.output, arguments.surface_name)


if __name__ == "__main__":
    main()
