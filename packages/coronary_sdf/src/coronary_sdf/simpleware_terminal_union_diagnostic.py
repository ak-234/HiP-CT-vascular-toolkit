"""Compare persisted Simpleware and STL-cropped Amira terminal locations."""

import sys
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

import simpleware_coronary_regions as regions


class _Document:
    def __init__(self, path):
        self.path = path

    def GetFilePath(self):
        return self.path


def _records(network):
    records = []
    for node in network.GetNodes():
        if len(list(node.GetSplines())) != 1:
            continue
        try:
            records.append(regions._terminal_record(node))
        except RuntimeError:
            pass
    return records


def main(project_path, component_mode=None):
    if component_mode:
        regions.STL_GRAPH_COMPONENT_MODE = component_mode
    simpleware = regions._load_simpleware_network_from_sip(
        _Document(project_path)
    )
    amira = regions._crop_amira_network_to_stl(
        regions._load_amira_network(), regions._load_stl_solid()
    )
    simpleware_records = _records(simpleware)
    amira_records = _records(amira)
    simpleware_points = np.asarray([
        regions._xyz(item["node"].GetPosition(False))
        for item in simpleware_records
    ])
    amira_points = np.asarray([
        regions._xyz(item["node"].GetPosition(False))
        for item in amira_records
    ])
    amira_distance, nearest_simpleware = cKDTree(simpleware_points).query(
        amira_points
    )
    simpleware_distance, _ = cKDTree(amira_points).query(simpleware_points)
    print("COUNTS Simpleware={} Amira={}".format(
        len(simpleware_records), len(amira_records)
    ))
    print("AMIRA_TERMINAL_STRAHLER_COUNTS {}".format(dict(sorted(Counter(
        item["spline"].strahler_order for item in amira_records
    ).items()))))
    print("AMIRA_TERMINAL_SYNTHETIC_COUNTS {}".format(dict(sorted(Counter(
        (
            "synthetic" if int(item["node"].node_id) >= 311 else "original",
            item["spline"].strahler_order,
        )
        for item in amira_records
    ).items()))))
    for item in amira_records:
        if item["node_name"] in {"AmiraNode_377", "AmiraNode_376", "AmiraNode_370"}:
            print(
                "FOCUS {} synthetic={} strahler={} position={} normal={}".format(
                    item["node_name"],
                    int(item["node"].node_id) >= 311,
                    item["spline"].strahler_order,
                    regions._xyz(item["node"].GetPosition(False)).tolist(),
                    item["normal"].tolist(),
                )
            )
    print("AMIRA_TO_SIMPLEWARE_PERCENTILES {}".format(
        np.percentile(amira_distance, [0, 10, 25, 50, 75, 90, 100]).tolist()
    ))
    print("SIMPLEWARE_TO_AMIRA_PERCENTILES {}".format(
        np.percentile(simpleware_distance, [0, 10, 25, 50, 75, 90, 100]).tolist()
    ))
    for tolerance in (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00):
        matched_indices = np.where(amira_distance <= tolerance)[0]
        tangent_dots = [
            float(np.dot(
                amira_records[int(index)]["normal"],
                simpleware_records[int(nearest_simpleware[index])]["normal"],
            ))
            for index in matched_indices
        ]
        print(
            "TOLERANCE {:.2f}: Amira matched={}, Simpleware matched={}, union={}, "
            "tangent_dot={}..{}".
            format(
                tolerance,
                int(np.sum(amira_distance <= tolerance)),
                int(np.sum(simpleware_distance <= tolerance)),
                len(simpleware_records) + int(np.sum(amira_distance > tolerance)),
                min(tangent_dots) if tangent_dots else None,
                max(tangent_dots) if tangent_dots else None,
            )
        )
    print("AMIRA TERMINALS UNMATCHED AT 0.50 MM")
    for index in np.where(amira_distance > 0.50)[0]:
        amira_record = amira_records[int(index)]
        simpleware_record = simpleware_records[int(nearest_simpleware[index])]
        tangent_dot = float(np.dot(
            amira_record["normal"], simpleware_record["normal"]
        ))
        delta = amira_points[index] - simpleware_points[int(nearest_simpleware[index])]
        mean_axis = amira_record["normal"] + simpleware_record["normal"]
        mean_axis /= np.linalg.norm(mean_axis)
        axial = abs(float(np.dot(delta, mean_axis)))
        radial = float(np.linalg.norm(delta - np.dot(delta, mean_axis) * mean_axis))
        print(
            "  {} distance={:.4f} mm axial={:.4f} radial={:.4f} "
            "tangent_dot={:.4f} position={}".format(
                amira_record["node_name"],
                float(amira_distance[index]),
                axial,
                radial,
                tangent_dot,
                amira_points[index].tolist(),
            )
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
