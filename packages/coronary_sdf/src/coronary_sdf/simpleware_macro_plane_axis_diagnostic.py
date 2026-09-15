"""Infer Simpleware finite-plane local normal from the recorded manual macro."""

import math
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import simpleware_coronary_regions as regions


PATTERN = re.compile(
    r"FinitePlane, Vector3D\(([^)]*)\), Vector3D\([^)]*\), "
    r"Vector3D\(([^)]*)\), ([^,]+), False"
)


def _numbers(text):
    return np.asarray([float(item.strip()) for item in text.split(",")])


def _rotation(axis, degrees):
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(float(degrees))
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * skew
    )


def main():
    text = Path("simpleware_boundary_region_mesh_refinement_API.py").read_text()
    planes = [(_numbers(a), _numbers(b), float(c)) for a, b, c in PATTERN.findall(text)]
    network = regions._load_amira_network()
    points = []
    tangents = []
    for spline in network.splines:
        for first, second in zip(spline.points[:-1], spline.points[1:]):
            delta = second - first
            length = float(np.linalg.norm(delta))
            if length > regions.EPS:
                points.append(0.5 * (first + second))
                tangents.append(delta / length)
    tree = cKDTree(np.asarray(points))
    tangents = np.asarray(tangents)
    scores = {sign: [[], [], []] for sign in (1, -1)}
    distances = []
    for centre, axis, angle in planes:
        distance, index = tree.query(centre)
        distances.append(float(distance))
        tangent = tangents[int(index)]
        for sign in (1, -1):
            matrix = _rotation(axis, sign * angle)
            for local_axis in range(3):
                scores[sign][local_axis].append(
                    abs(float(np.dot(matrix[:, local_axis], tangent)))
                )
    print("PLANES", len(planes))
    print("NEAREST_GRAPH_MM", np.percentile(distances, [0, 25, 50, 75, 100]).tolist())
    for sign in (1, -1):
        for local_axis, values in enumerate(scores[sign]):
            print(
                "SIGN", sign, "LOCAL", "XYZ"[local_axis],
                "MEDIAN", float(np.median(values)),
                "P75", float(np.percentile(values, 75)),
                "P90", float(np.percentile(values, 90)),
            )


if __name__ == "__main__":
    main()
