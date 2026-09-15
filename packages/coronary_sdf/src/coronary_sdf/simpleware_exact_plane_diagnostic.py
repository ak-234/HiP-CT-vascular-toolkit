"""Read-only real-surface diagnostic for coronary outlet plane discovery."""

import sys

import simpleware_coronary_regions as regions


def main():
    if len(sys.argv) > 1:
        regions.STL_SURFACE_PATH = sys.argv[1]
    solid = regions._load_stl_solid()
    network = regions._crop_amira_network_to_stl(
        regions._load_amira_network(), solid
    )
    terminals = regions._find_terminals(network, None, "amira")
    accepted = []
    rejected = []
    for ordinal, record in enumerate(terminals, start=1):
        try:
            regions._search_stl_validated_terminal_plane(record, solid)
        except RuntimeError as exc:
            rejected.append((record["node_name"], str(exc)))
            print("REJECT", record["node_name"], exc, flush=True)
        else:
            accepted.append(record)
            print(
                "ACCEPT",
                record["node_name"],
                "inset={:.3f}".format(record["surface_plane_inset_mm"]),
                "diameter={:.3f}".format(2.0 * record["surface_loop_radius"]),
                "area={:.4f}".format(record["surface_loop_area"]),
                "tangent_cos={:.3f}".format(
                    record["terminal_tangent_cosine"]
                ),
                "side_cos={:.3f}".format(record["normal_outward_cosine"]),
                flush=True,
            )
        if ordinal % 10 == 0:
            print("PROGRESS", ordinal, len(terminals), flush=True)
    print("SUMMARY", len(terminals), len(accepted), len(rejected), flush=True)
    if accepted:
        print(
            "INSET_RANGE",
            min(item["surface_plane_inset_mm"] for item in accepted),
            max(item["surface_plane_inset_mm"] for item in accepted),
            flush=True,
        )
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
