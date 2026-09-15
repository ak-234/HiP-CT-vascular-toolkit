"""Command-line interface for skeleton_analysis.

Subcommands (batch replacements for the interactive MATLAB scripts):

* ``order``      - Strahler + topological ordering of an ``.am`` graph.
* ``merge``      - merge two ``.am`` graphs into one.
* ``vesselvio``  - convert VesselVio vertices/edges CSVs to an ``.am`` graph.
* ``roots``      - list candidate root nodes for a graph.
* ``info``       - print a summary of an ``.am`` graph.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np


def _cmd_info(args) -> int:
    from skeleton_analysis.io.amira import read_amira

    g = read_amira(args.input)
    print(f"header:    {g.header}")
    print(f"vertices:  {g.n_vertices}")
    print(f"edges:     {g.n_edges}")
    print(f"points:    {g.n_points}")
    print("vertex fields:", list(g.vertex_fields))
    print("edge fields:  ", list(g.edge_fields))
    print("point fields: ", list(g.point_fields))
    warnings = g.check_consistency()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  -", w)
    return 0


def _cmd_roots(args) -> int:
    from skeleton_analysis.ordering.pipeline import root_candidates

    cands = root_candidates(args.input)
    print("root candidates (out-degree-0 nodes):", cands)
    return 0


def _cmd_order(args) -> int:
    from skeleton_analysis.ordering.pipeline import run_ordering

    result = run_ordering(args.input, args.output, root_id=args.root)
    print(f"root: {result.root_id}")
    print(f"edges flipped to orient toward root: {result.flipped_edges.size}")
    s = result.strahler
    print(f"Strahler orders 1..{int(s.max())}; max generation {int(result.topo.max())}")
    if args.output:
        print(f"written: {args.output}")
    return 0


def _cmd_merge(args) -> int:
    from skeleton_analysis.io.amira import read_amira, write_amira
    from skeleton_analysis.utils.merge import add_spatial_graphs

    g1 = read_amira(args.graph1)
    g2 = read_amira(args.graph2)
    merged = add_spatial_graphs(g1, g2, match_tol=args.tol)
    write_amira(merged, args.output)
    print(f"merged {g1.n_vertices}+{g2.n_vertices} vertices -> {merged.n_vertices}")
    print(f"written: {args.output}")
    return 0


def _cmd_vesselvio(args) -> int:
    from skeleton_analysis.io.vesselvio import vesselvio_to_amira

    g = vesselvio_to_amira(
        args.vertices,
        args.edges,
        args.output,
        resolution=(args.res_xy, args.res_z),
        swap_zx=not args.no_swap,
    )
    print(f"converted {g.n_vertices} vertices, {g.n_edges} edges -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="skeleton-analysis", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("info", help="summarise an .am graph")
    pi.add_argument("input")
    pi.set_defaults(func=_cmd_info)

    pr = sub.add_parser("roots", help="list candidate root nodes")
    pr.add_argument("input")
    pr.set_defaults(func=_cmd_roots)

    po = sub.add_parser("order", help="Strahler + topological ordering")
    po.add_argument("input")
    po.add_argument("output", nargs="?", default=None, help="output .am (optional)")
    po.add_argument("--root", type=int, default=None, help="root node ID (0-based)")
    po.set_defaults(func=_cmd_order)

    pm = sub.add_parser("merge", help="merge two .am graphs")
    pm.add_argument("graph1")
    pm.add_argument("graph2", help="the smaller graph")
    pm.add_argument("output")
    pm.add_argument("--tol", type=float, default=0.0, help="vertex match tolerance")
    pm.set_defaults(func=_cmd_merge)

    pv = sub.add_parser("vesselvio", help="convert VesselVio CSVs to .am")
    pv.add_argument("vertices", help="vertices.csv")
    pv.add_argument("edges", help="edges.csv")
    pv.add_argument("output", help="output .am")
    pv.add_argument("--res-xy", type=float, default=50.0)
    pv.add_argument("--res-z", type=float, default=50.0)
    pv.add_argument("--no-swap", action="store_true", help="do not swap z<->x axes")
    pv.set_defaults(func=_cmd_vesselvio)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
