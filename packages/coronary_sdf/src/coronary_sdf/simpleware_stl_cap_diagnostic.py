"""Find large coplanar compact triangle groups that may be STL outlet caps."""

import math
import sys

import numpy as np

import simpleware_coronary_regions as regions


def _basis(normal):
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def main(path):
    triangles = regions._transform_stl_triangles(
        regions._read_stl_triangles(path)
    )
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_a, edge_b)
    twice_area = np.linalg.norm(cross, axis=1)
    valid = twice_area > 1.0e-12
    triangles = triangles[valid]
    cross = cross[valid]
    twice_area = twice_area[valid]
    normals = cross / twice_area[:, None]
    centroids = np.mean(triangles, axis=1)
    offsets = np.sum(normals * centroids, axis=1)

    # Meshmixer cap triangles are coplanar. Fine rounding retains those groups
    # while preventing a curved vessel wall from becoming one large patch.
    keys = np.empty((len(normals), 4), dtype=np.int32)
    keys[:, :3] = np.rint(normals * 100.0).astype(np.int32)
    keys[:, 3] = np.rint(offsets * 100.0).astype(np.int32)
    unique_keys, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    candidate_groups = np.where(counts >= 20)[0]
    print(
        "TRIANGLES={} GROUPS={} GROUPS_GE_20={}".format(
            len(triangles), len(unique_keys), len(candidate_groups)
        ),
        flush=True,
    )

    results = []
    for ordinal, group in enumerate(candidate_groups, start=1):
        indices = np.where(inverse == group)[0]
        group_triangles = triangles[indices]
        area = 0.5 * float(np.sum(twice_area[indices]))
        weights = twice_area[indices]
        normal = np.sum(normals[indices] * weights[:, None], axis=0)
        normal /= np.linalg.norm(normal)
        points = group_triangles.reshape(-1, 3)
        centre = np.mean(points, axis=0)
        first, second = _basis(normal)
        relative = points - centre
        projected = np.column_stack((
            np.sum(relative * first[None, :], axis=1),
            np.sum(relative * second[None, :], axis=1),
        ))
        covariance = np.sum(
            projected[:, :, None] * projected[:, None, :], axis=0
        ) / max(1, len(projected) - 1)
        trace = float(covariance[0, 0] + covariance[1, 1])
        determinant = float(
            covariance[0, 0] * covariance[1, 1]
            - covariance[0, 1] * covariance[1, 0]
        )
        discriminant = math.sqrt(max(0.0, 0.25 * trace * trace - determinant))
        major = 0.5 * trace + discriminant
        minor = 0.5 * trace - discriminant
        aspect = math.sqrt(max(major, 0.0) / max(minor, 1.0e-12))
        radial = np.linalg.norm(projected, axis=1)
        max_radius = float(np.max(radial))
        fill = area / max(math.pi * max_radius * max_radius, 1.0e-12)
        plane_error = float(np.max(np.abs(np.sum(relative * normal, axis=1))))
        if area >= 0.005 and aspect <= 3.0 and fill >= 0.20 and plane_error <= 0.01:
            results.append((area, len(indices), aspect, fill, plane_error, centre, normal))
        if ordinal % 250 == 0:
            print("ANALYSED {}/{}".format(ordinal, len(candidate_groups)), flush=True)

    results.sort(reverse=True, key=lambda item: item[0])
    print("COMPACT_PLANAR_GROUPS={}".format(len(results)), flush=True)
    for index, item in enumerate(results, start=1):
        area, count, aspect, fill, error, centre, normal = item
        print(
            "CAP {:03d} triangles={} area={:.6f} eq_diameter={:.4f} "
            "aspect={:.3f} fill={:.3f} error={:.6f} centre={} normal={}".format(
                index,
                count,
                area,
                2.0 * math.sqrt(area / math.pi),
                aspect,
                fill,
                error,
                centre.tolist(),
                normal.tolist(),
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
