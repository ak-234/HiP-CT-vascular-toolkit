"""Command line for the reconnection and repair tools.

    python -m hipct_seg_debug.edit report   graph.am
    python -m hipct_seg_debug.edit gaps     graph.am --out fixed.am
    python -m hipct_seg_debug.edit connect  graph.am --out fixed.am --tjunction
    python -m hipct_seg_debug.edit surface  graph.am --out-dir surfaces/

Everything is a dry run unless ``--out`` is given: the proposers print what they
would do and why they refused the rest, and nothing is written. That is the right
default for a set of heuristics -- a bridge that should not exist reroutes flow in
whatever CFD run comes afterwards, and it is much cheaper to notice here.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np


def _add_roots_json(parser) -> None:
    """``--roots-json``, on every command that consumes a chosen root."""
    parser.add_argument(
        "--roots-json", default=None,
        help="roots sidecar written by `pick-roots`: one chosen root per tree. Read "
             "if it exists; an explicit --root-edge wins for its own component",
    )


def _add_pick_roots_args(parser, *, roots_json: bool = True) -> None:
    """``--pick-roots``, for a command that derives or refines a skeleton.

    The window always opens **after** the long work is finished, never in the middle
    of it: `skeletonise-all` can spend 25 minutes deriving candidates, and a 3-D
    window blocking half way through is how an afternoon gets lost. So this is an
    opt-in flag rather than the default, and the standalone `pick-roots` command
    stays the way to root a graph that already exists.
    """
    parser.add_argument(
        "--pick-roots", action="store_true",
        help="open the 3-D picker once the skeleton is finished and root each tree "
             "by hand, instead of guessing with `auto_roots`",
    )
    parser.add_argument("--style", default="tube",
                        choices=("contour", "tube", "lines"),
                        help="with --pick-roots: 'tube' is one solid tube per segment "
                             "(the default here), 'lines' the graph itself -- an edge "
                             "per polyline and a point per node, and much the fastest "
                             "on a whole tree -- 'contour' rings plus centreline dots")
    parser.add_argument("--color-by", default="strahler", choices=("strahler", "none"),
                        help="with --pick-roots: colour each segment by Strahler "
                             "order, so the trunk is obvious before anything is clicked")
    if roots_json:
        parser.add_argument(
            "--roots-json", default=None,
            help="with --pick-roots, write the chosen roots here; the sidecar is "
                 "what `optimise-skeleton`, `radius-perimeter` and `crop` read back",
        )


SCOPES = ("whole", "per-tree", "largest")


def _add_scope_arg(parser) -> None:
    """``--scope``, on its own, for a parser that already has the per-tree knobs."""
    parser.add_argument(
        "--scope", default=None, choices=SCOPES,
        help="what M_S is computed over. 'whole' (default) is the published "
             "definition: V/cc/cl/B over the whole graph, but chi over the largest "
             "component alone. 'per-tree' scores each mask component on its own "
             "terms. 'largest' scores only the largest component, applying that "
             "restriction to all five terms rather than just chi. The whole-graph "
             "table is always printed first, because the other two define cc and chi "
             "differently and are not comparable with it",
    )


def _add_per_tree_score_args(parser) -> None:
    """``--scope`` for the commands that *score* rather than skeletonise.

    A separate helper from :func:`_add_per_tree_args` because scope means a different
    thing here -- what a score is computed *over*, rather than how the skeleton is
    derived -- and because these commands take an aggregate as well.
    """
    _add_scope_arg(parser)
    parser.add_argument(
        "--min-component-voxels", type=int, default=0,
        help="with --scope per-tree/largest, ignore mask components smaller than this",
    )
    parser.add_argument("--max-trees", type=int, default=None,
                        help="with --scope per-tree, score only the N largest "
                             "components; --scope largest is the same as N=1")
    parser.add_argument(
        "--objective", default="weighted", choices=("weighted", "mean", "sum"),
        help="with --scope per-tree, how the per-tree scores combine into one "
             "number: weighted by voxel count (default), an unweighted mean, or a "
             "sum that rescales with tree count",
    )


def _resolve_scope(args) -> str:
    """The scoring scope, defaulting to ``per-tree`` when the skeleton was split.

    Deriving the skeleton per tree and then ranking it on a whole-graph ``M_S`` would
    judge both trees on the largest one's chi, which is the blind spot ``--per-tree``
    exists to close -- so it carries through unless ``--scope`` says otherwise.
    """
    scope = getattr(args, "scope", None)
    if scope:
        return scope
    return "per-tree" if getattr(args, "per_tree", False) else "whole"


def _add_per_tree_args(parser) -> None:
    """The per-component skeletonisation flags, on the two commands that derive one.

    Deliberately not on ``seg_common``: a dozen commands take a segmentation and only
    these two turn it into a skeleton, so putting it there would advertise a flag most
    of them ignore -- the mistake the ``--out`` split at the top of `build_parser`
    exists to undo.
    """
    parser.add_argument(
        "--per-tree", action="store_true",
        help="skeletonise each mask component separately and tag every edge with its "
             "tree index; two components here are the left and right coronary trees. "
             "The skeleton shifts slightly -- thinning inside a tight box sees a "
             "different neighbourhood than a whole-volume run",
    )
    parser.add_argument(
        "--min-component-voxels", type=int, default=0,
        help="with --per-tree, ignore mask components smaller than this (0 keeps all)",
    )
    parser.add_argument(
        "--max-trees", type=int, default=None,
        help="with --per-tree, keep only the N largest components",
    )


def _check_per_tree(args) -> str | None:
    """The message to refuse with when the per-tree knobs were set without --per-tree.

    Refusing rather than inferring ``--per-tree``: silently switching to a different
    skeletonisation because a threshold was set is exactly the sort of implicit change
    that makes two runs incomparable.
    """
    if getattr(args, "per_tree", False):
        return None
    given = [
        name for name, value in (("--min-component-voxels", getattr(args, "min_component_voxels", 0)),
                                 ("--max-trees", getattr(args, "max_trees", None)))
        if value
    ]
    if given:
        return f"{' and '.join(given)} only applies with --per-tree"
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hipct_seg_debug.edit",
        description="Reconnect and repair HiP-CT vessel skeletons.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Split in two so `--out` reaches only the commands that honour it: `report`
    # writes nothing at all, and `surface` writes a mesh to `--out-dir`. Both used
    # to advertise an `--out` they silently ignored.
    graph_arg = argparse.ArgumentParser(add_help=False)
    # Variadic everywhere, so every command that processes a skeleton takes the left
    # and right trees together. `_load` merges them into one graph carrying one `tree`
    # index per source, and `--out` receives that one graph -- the same arrangement
    # the GUI's `skeleton (.am)` field uses, for the same reason.
    graph_arg.add_argument("graph", nargs="+", metavar="GRAPH",
                           help="ASCII Amira .am spatial graph; pass several and they "
                                "are merged into one, each held as its own tree")

    common = argparse.ArgumentParser(add_help=False, parents=[graph_arg])
    common.add_argument("--out", default=None, help="write the edited graph here")

    sub.add_parser("report", parents=[graph_arg],
                   help="describe the graph: components, free ends, gaps")

    gaps = sub.add_parser("gaps", parents=[common],
                          help="fill large jumps inside a single edge")
    gaps.add_argument("--min-gap-um", type=float, default=None,
                      help="absolute floor for what counts as a gap")
    gaps.add_argument("--big-jump-ratio", type=float, default=None,
                      help="a gap must also exceed this many vessel widths")

    # ---- segmentation-driven commands (these need the lattice) --------------
    # Declared before `connect` because `connect --dpc` reads the image too, and a
    # parent parser has to exist before the subparser that inherits it.
    seg_common = argparse.ArgumentParser(add_help=False)
    seg_common.add_argument("--seg", default=None,
                            help="binary Amira label lattice (.am); defaults to the "
                                 "LADAF-2024-28 path")
    seg_common.add_argument("--labels-field", default="Labels")
    seg_common.add_argument("--voxel-um", type=float, default=None,
                            help="the acquisition's own raw voxel size. Given, it is "
                                 "the authority: the lattice bounding box is read as "
                                 "the same lattice in the wrong units and every length "
                                 "-- the graph's coordinates and radii included -- is "
                                 "rescaled onto it, and outputs are stamped with it. "
                                 "Omitted, use the graph's recorded voxel size; "
                                 "unstamped graphs use the bounding box")
    seg_common.add_argument("--stride", type=int, default=1,
                            help="decimate the lattice on every axis before decoding")
    seg_common.add_argument("--edits", default=None,
                            help="mask edit store (.npz) from a painting session; "
                                 "composited onto the lattice before anything reads it")
    seg_common.add_argument(
        "--ignore-materials", action="store_true",
        help="split the mask by connectivity alone, ignoring a Materials block that "
             "already names the trees (Left_Tree, Right_Tree). The old behaviour, kept "
             "for comparing against results produced before the split was aware of them",
    )

    connect = sub.add_parser("connect", parents=[common, seg_common],
                             help="bridge disconnected ends")
    connect.add_argument("--tjunction", action="store_true",
                         help="also attach free ends onto the side of other vessels")
    connect.add_argument("--same-component", action="store_true",
                         help="allow joins inside one component (creates loops)")
    connect.add_argument("--cone-deg", type=float, default=None)
    connect.add_argument("--reach-factor", type=float, default=None,
                         help="max span, in multiples of the endpoint radius")
    connect.add_argument("--radius-ratio", type=float, default=None)
    connect.add_argument("--tortuosity", type=float, default=None)
    connect.add_argument("--show-rejected", action="store_true",
                         help="list every refused candidate, not just a tally")
    connect.add_argument("--keep-interpolation", action="store_true",
                         help="leave Avizo's interpolated fills in place; by default "
                              "each one is removed first, so the break it was hiding "
                              "becomes two real free ends and the image gets to decide")
    # ---- the DPC walk: let the image decide, not just the geometry ----------
    connect.add_argument("--dpc", action="store_true",
                         help="re-walk each proposal through the image and keep only "
                              "the ones the greyscale supports; needs --raw and --seg")
    connect.add_argument("--raw", default=None,
                         help="directory of raw image slices, for the --dpc greyscale")
    # The DPC staging calls the biggest components "the backbone" -- on this data the
    # left and right coronary trees. Both knobs exist because that guess is only right
    # while the graph holds exactly the trees it should.
    connect.add_argument("--backbone-trees", type=int, action="append", default=[],
                         help="tree index to treat as backbone for the --dpc Type 1/2 "
                              "staging; repeat for several. Needs a graph carrying a "
                              "`tree` field (skeletonise --per-tree)")
    connect.add_argument("--backbone-components", type=int, default=2,
                         help="without --backbone-trees, how many of the largest "
                              "components count as backbone (default 2: left and right)")
    connect.add_argument("--dpc-learned", action="store_true",
                         help="train the centreline-probability model on this graph's "
                              "own skeleton instead of using the vesselness field; "
                              "better across a true gap, where the mask is absent")
    connect.add_argument("--cfc-model", default=None,
                         help="persisted DF21 artifact from train-cfc; uses raw patches")
    connect.add_argument("--dpc-omega", type=float, default=None,
                         help="weight on the probability term (the paper's 5.0)")
    connect.add_argument("--dpc-distance-weight", type=float, default=1.0)
    connect.add_argument("--dpc-cosine-weight", type=float, default=1.0)
    connect.add_argument("--dpc-neighbourhood", choices=("two-level", "full"),
                         default="two-level")
    connect.add_argument("--dpc-min-probability", type=float, default=None,
                         help="reject a walk whose mean probability falls below this")
    connect.add_argument("--dpc-pad-factor", type=float, default=4.0,
                         help="ROI padding around a bridge, in source radii")
    # HiP-CT coronary lumen is *darker* than the myocardium around it (selftest.py:36
    # measures exactly this), and the vesselness filter responds to bright tubes, so
    # the default inverts. A scan with contrast agent would want this flag.
    connect.add_argument("--dpc-bright-lumen", action="store_true",
                         help="the lumen is brighter than the surrounding tissue "
                              "(default assumes darker, as in native HiP-CT)")

    # ---- the geodesic connector: repair the graph and the mask together -----
    # Opt-in, and mutually exclusive with --dpc: they are two different answers to
    # the same question and running both would apply one on top of the other's
    # topology, so neither result would mean anything.
    connect.add_argument("--geodesic", action="store_true",
                         help="collapse-aware geodesic reconnection: classify each "
                              "break, search a direction-aware route through the "
                              "image, and repair the segmentation alongside the "
                              "graph; needs --seg, and --raw for anything but the "
                              "shortest mask gaps")
    connect.add_argument("--out-seg", default=None,
                         help="write the repaired segmentation here; required "
                              "whenever --geodesic is given --out, because a graph "
                              "written without its mask is a pair of files that "
                              "disagree about where the vessels are")
    connect.add_argument("--review-json", default=None,
                         help="write the candidates needing a human decision here")
    connect.add_argument("--decisions-json", default=None,
                         help="write the full record of every candidate here; if the "
                              "file already exists it is read first and any operator "
                              "rulings in it are applied")
    connect.add_argument("--max-unsupported-factor", type=float, default=4.0,
                         help="reject a route with more than this many local radii "
                              "of contiguous unsupported path (default 4)")
    connect.add_argument("--alternatives", type=int, default=3,
                         help="how many spatially distinct routes to look for; the "
                              "second is what decides whether a route is ambiguous "
                              "(default 3)")
    connect.add_argument("--no-tjunction-geodesic", action="store_true",
                         help="with --geodesic, propose end-to-end joins only")
    connect.add_argument("--no-mask-endpoints", action="store_true",
                         help="with --geodesic, propose only between endpoints the "
                              "graph already has; by default the mask is searched "
                              "for lumen no centreline describes and the free ends "
                              "of that lumen are proposed to as well, which is what "
                              "lets a break whose far side was pruned away be seen "
                              "at all")
    connect.add_argument("--describe-radii", type=float, default=None,
                         help="how close a centreline point must be, in local "
                              "radii, for lumen to count as already described "
                              "(default 1.5); larger finds fewer mask free ends")
    connect.add_argument("--min-lobe-voxels", type=int, default=None,
                         help="the smallest patch of undescribed lumen that may "
                              "carry a mask free end (default 20)")

    # ---- what Avizo invented rather than measured ---------------------------
    # Everything downstream reads the per-point field this writes, and nothing detects
    # on its own -- see `interpolation.py`'s docstring for why that is deliberate.
    flagint = sub.add_parser("flag-interpolation", parents=[common, seg_common],
                             help="mark the points Avizo interpolated across a gap")
    flagint.add_argument("--show-spans", action="store_true",
                         help="list every span with its evidence, not just a tally")
    flagint.add_argument("--anchor-jump", type=float, default=None,
                         help="minimum radius step at each end of a straight run, as a "
                              "fraction (default 0.05)")
    flagint.add_argument("--min-span-points", type=int, default=None,
                         help="shortest run that counts as an invented vessel")
    flagint.add_argument("--min-jump-um", type=float, default=None,
                         help="shortest intra-edge step that can count as an unsampled "
                              "jump (default 600); needs --seg")
    flagint.add_argument("--no-jumps", action="store_true",
                         help="skip the unsampled-jump signature, which needs the mask "
                              "labelled as well as sampled")

    train = sub.add_parser("train-cfc", parents=[graph_arg],
                           help="train and persist a scan-specific DF21 classifier")
    train.add_argument("--seg", default=None, help="binary Amira label lattice")
    train.add_argument("--labels-field", default="Labels")
    train.add_argument("--raw", required=True, help="directory of raw image slices")
    train.add_argument("--out-model", required=True, help="new CFC artifact directory")
    train.add_argument("--voxel-um", type=float, default=None)
    train.add_argument("--max-positive", type=int, default=None,
                       help="deterministically cap positive skeleton voxels")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--overwrite", action="store_true",
                       help="replace known files in an existing artifact directory")

    evaluate = sub.add_parser("evaluate-dpc", parents=[graph_arg, seg_common],
                              help="sweep omega over reviewed Type 1/2/3 regions")
    evaluate.add_argument("--raw", required=True)
    evaluate.add_argument("--cfc-model", required=True)
    evaluate.add_argument("--regions", default=str(
        Path(__file__).with_name("reconnect") / "dpc_eval_regions.json"))
    evaluate.add_argument("--omega", default="0:7",
                          help="inclusive range such as 0:7 or comma-separated values")
    evaluate.add_argument("--cases", default=None,
                          help="optional comma-separated region IDs (default: all)")
    evaluate.add_argument("--output", default="dpc-evaluation.json")
    evaluate.add_argument("--no-ablations", action="store_true")

    export_regions = sub.add_parser("export-dpc-regions", parents=[graph_arg, seg_common],
                                    help="export blinded review contact sheets")
    export_regions.add_argument("--raw", required=True)
    export_regions.add_argument("--regions", default=str(
        Path(__file__).with_name("reconnect") / "dpc_eval_regions.json"))
    export_regions.add_argument("--output", required=True)

    surface = sub.add_parser("surface", parents=[graph_arg],
                             help="generate the lumen surface for a graph")
    surface.add_argument("--out-dir", default="surface", help="where to write the STL")
    surface.add_argument("--voxel-mm", type=float, default=None)
    surface.add_argument("--keep-interpolation", action="store_true",
                         help="mesh Avizo's interpolated fills too; by default they are "
                              "cut out, because a capsule swept along an invented "
                              "centreline is an invented vessel in the STL")

    skel = sub.add_parser("skeletonise", parents=[seg_common],
                          help="derive a centreline graph from the segmentation")
    skel.add_argument("--out", default=None, help="write the generated graph here (.am)")
    skel.add_argument("--order", action="store_true",
                      help="also assign Strahler order and topological generation")
    _add_pick_roots_args(skel)
    _add_per_tree_args(skel)

    opt = sub.add_parser("optimise", parents=[common, seg_common],
                         help="order a graph, clean its radii, and score it")
    opt.add_argument("--reference", default=None,
                     help="graph to score against (.am); defaults to the Avizo skeleton")
    opt.add_argument("--oblique", action="store_true",
                     help="also re-measure flagged radii from oblique cross-sections")
    opt.add_argument("--sensitivity", action="store_true",
                     help="also report centreline sensitivity, which needs the lattice "
                          "decoded (2.34 GB at stride 1)")
    opt.add_argument("--bb-threshold", type=float, default=900.0,
                     help="bifurcation match distance in um")
    # `optimise --order` is where most graphs first get a Strahler column, so it is
    # the last place that should still be guessing at the root.
    _add_roots_json(opt)

    rad = sub.add_parser("repair-radius", parents=[common],
                         help="fill collapsed radii by extrapolating the local taper")
    rad.add_argument("--seg", default=None, help="lattice, for the image-based detector")
    rad.add_argument("--labels-field", default="Labels")
    rad.add_argument("--voxel-um", type=float, default=None)
    # Painting a collapsed lumen open changes what the image detector finds, so
    # this command needs the corrections as much as the ones on `seg_common` do.
    rad.add_argument("--edits", default=None,
                     help="mask edit store (.npz) to composite before measuring")
    rad.add_argument("--source", choices=("outlier", "image", "both"), default="both",
                     help="how to find collapsed spans")
    rad.add_argument("--factor", type=float, default=0.6,
                     help="a span is collapsed below this fraction of the local trend")
    rad.add_argument("--margin", type=int, default=None,
                     help="points at each segment end never treated as collapsed")
    rad.add_argument("--max-taper-per-mm", type=float, default=None,
                     help="clamp on the fitted taper")
    rad.add_argument("--allow-decrease", action="store_true",
                     help="let the fit lower an over-inflated span, not just raise a collapsed one")

    mask = sub.add_parser("repair-mask", parents=[seg_common],
                          help="cull debris and close small breaks in the segmentation")
    mask.add_argument("--out", default=None, help="write the repaired mask here (.tif)")
    mask.add_argument("--min-voxels", type=int, default=0,
                      help="drop connected components smaller than this")
    mask.add_argument("--keep-largest", type=int, default=2,
                      help="components this large are protected regardless of --min-voxels")
    mask.add_argument("--close", type=int, default=0,
                      help="radius in voxels for morphological closing; 0 to skip")

    # ---- the Walsh-Berg workflow: run every algorithm, score, refine, re-measure --
    # See SKELETONISATION.md. These four are one chain: `skeletonise-all` proposes
    # candidates, `score` ranks them, `optimise-skeleton` cleans the winner, and
    # `radius-perimeter` replaces its radii with measurements.
    skel_all = sub.add_parser("skeletonise-all", parents=[seg_common],
                              help="run several skeletonisation algorithms and score each")
    # Always writes, like `surface` and unlike the proposers. The dry-run-unless-`--out`
    # convention is for heuristics whose *printed plan* is the useful output and whose
    # writing is the risky act; this one spends up to 25 minutes deriving skeletons, and
    # discarding them because a flag was left blank is not a safe default, it is a
    # wasted afternoon.
    skel_all.add_argument("--out-dir", default="skeletons",
                          help="where to write every candidate .am")
    skel_all.add_argument("--algorithms", default="lee",
                          help="comma-separated: lee, teasar, amira")
    skel_all.add_argument("--amira-graph", default=None,
                          help="existing Avizo .am to score as the 'amira' candidate")
    skel_all.add_argument("--teasar-scale", type=float, default=None,
                          help="TEASAR penalty scale (Centerline Tree's 'slope')")
    skel_all.add_argument("--teasar-const-um", type=float, default=None,
                          help="TEASAR fixed penalty in um (its 'zeroVal')")
    skel_all.add_argument("--no-tree-chi", action="store_true",
                          help="score chi against the segmentation's loop count rather "
                               "than the tree ideal, for a number comparable to the paper")
    skel_all.add_argument(
        "--no-score",
        action="store_true",
        help="write skeletons without the dense image topology/super-metric pass; "
             "useful for full-resolution volumes too large to score in memory",
    )
    skel_all.add_argument("--bb-threshold", type=float, default=900.0,
                          help="bifurcation match distance in um")
    # `--per-tree` here means both halves at once: derive the skeleton per component
    # and score each tree on its own terms, which is the only combination that makes
    # the ranking mean anything -- a per-tree skeleton ranked by a whole-graph M_S
    # would still be judged on the largest component's chi alone.
    _add_per_tree_args(skel_all)
    # `--scope` defaults to `per-tree` here when `--per-tree` is given, so the usual
    # pairing needs one flag; pass `--scope whole` to split the skeleton but still
    # rank the candidates on the published metric.
    _add_scope_arg(skel_all)
    skel_all.add_argument(
        "--objective", default="weighted", choices=("weighted", "mean", "sum"),
        help="with --scope per-tree, how per-tree scores combine when ranking "
             "candidates",
    )
    _add_pick_roots_args(skel_all)

    opt_skel = sub.add_parser("optimise-skeleton", parents=[common, seg_common],
                              help="de-loop, prune spurs, smooth and re-centre a skeleton")
    opt_skel.add_argument("--no-deloop", action="store_true",
                          help="keep cycles (they are collapse artefacts by default)")
    opt_skel.add_argument("--no-prune", action="store_true")
    opt_skel.add_argument("--no-recentre", action="store_true")
    opt_skel.add_argument("--prune-factor", type=float, default=None,
                          help="drop a leaf shorter than this many local vessel radii")
    opt_skel.add_argument("--prune-radius-ratio", type=float, default=None,
                          help="never prune a leaf this thick relative to its parent")
    opt_skel.add_argument("--smoother", default="gaussian",
                          choices=("none", "gaussian", "savgol", "bspline", "multiscale"),
                          help="which centreline smoother; savgol/bspline/multiscale "
                               "come from coronary_sdf")
    opt_skel.add_argument("--smooth-um", type=float, default=None,
                          help="smoothing window for --smoother gaussian; 0 disables")
    opt_skel.add_argument("--drift-radius-factor", type=float, default=None,
                          help="trust region for --smoother multiscale, in local "
                               "radii (coronary_sdf's undeclared default is 0.25)")
    opt_skel.add_argument("--recentre-passes", type=int, default=None)
    opt_skel.add_argument("--recentre-tangent-radii", type=float, default=None,
                          help="length of the chord the cut plane's normal is taken "
                               "over, in local radii; small values re-introduce the "
                               "voxel staircase and scatter the centreline")
    opt_skel.add_argument("--recentre-damping", type=float, default=None,
                          help="fraction of the way to the cross-section centroid to "
                               "move per pass; 1 takes the full step and oscillates")
    opt_skel.add_argument("--recentre-max-move-frac", type=float, default=None,
                          help="furthest a point may end up from where re-centring "
                               "started, in its own radii (all passes together)")
    opt_skel.add_argument("--recentre-grow-radii", type=float, default=None,
                          help="how far the cut window may grow chasing a section that "
                               "reaches its edge, in expected radii")
    opt_skel.add_argument("--recentre-blob", default=None, choices=("blob4", "blob8"),
                          help="connectivity the centroid is taken over; 8 keeps a "
                               "diagonally pinched lumen whole, 4 refuses to bridge to "
                               "a vessel touching at one corner")
    opt_skel.add_argument("--sweep", default=None,
                          help="grid to score instead of writing, e.g. "
                               "\"prune-factor=1,2,3;smooth-um=0,100,200\"")
    opt_skel.add_argument("--lhs", type=int, default=0,
                          help="Latin-hypercube samples between each --sweep range "
                               "(the paper's design); 0 uses the listed values as a grid")
    opt_skel.add_argument("--bb-threshold", type=float, default=900.0)
    opt_skel.add_argument("--no-tree-chi", action="store_true")
    _add_roots_json(opt_skel)
    _add_pick_roots_args(opt_skel, roots_json=False)
    _add_per_tree_score_args(opt_skel)

    # ---- diagnostics: read a written graph back and ask why a radius is what it is.
    # No `--out`: these report, they do not edit, so they take `graph_arg` rather
    # than `common`. They appear in the GUI's Commands panel automatically, because
    # `cliform.describe_parser` generates it from this parser.
    diag = sub.add_parser("segment-diagnosis", parents=[graph_arg, seg_common],
                          help="per-point radius, provenance and re-cut section "
                               "geometry for named segments")
    diag.add_argument("--segment", type=int, action="append", default=[],
                      required=True,
                      help="segment id to report; repeat for several")

    jmask = sub.add_parser("junction-mask", parents=[graph_arg, seg_common],
                           help="which term of the junction mask refuses each "
                                "point: `stable`, or adjacency")
    jmask.add_argument("--segment", type=int, action="append", default=[],
                       required=True,
                       help="segment id to report; repeat for several")

    rref = sub.add_parser("reformat-radius", parents=[graph_arg, seg_common],
                          help="measure named segments again on reformat's "
                               "parallel-transport planes, an independent "
                               "plane construction")
    rref.add_argument("--segment", type=int, action="append", default=[],
                      required=True,
                      help="segment id to measure; repeat for several")
    rref.add_argument("--size-px", type=int, default=81,
                      help="plane size in pixels; one pixel is one segmentation "
                           "voxel, so this is also the field of view in voxels")

    sub.add_parser("ostium-flare", parents=[graph_arg, seg_common],
                   help="how much wider the section gets approaching a branched "
                        "node, and whether the flare is round")

    from .centreline_cli import add_parsers as add_centreline_parsers
    add_centreline_parsers(sub, common, seg_common)

    radp = sub.add_parser("radius-perimeter", parents=[common, seg_common],
                          help="replace every radius with its own cross-section perimeter")
    radp.add_argument("--gate-voxels", type=float, default=None,
                      help="below this radius in voxels use sqrt(area/pi) instead. "
                           "Note the original reason for this gate -- that a "
                           "digitised perimeter over-states a tiny section -- is "
                           "measurably wrong: it under-states one. See "
                           "--no-perimeter-correction")
    radp.add_argument("--junction-mask-max-fraction", type=float, default=None,
                      help="most of a segment's own length one junction may consume, "
                           "per end (default 0.4, so a fifth of every segment stays "
                           "measurable). Unbounded, the mask ate 66 of 309 segments "
                           "whole on LADAF-2024-28; pass a value >= 0.5 to restore "
                           "that behaviour")
    radp.add_argument("--no-ownership-near-junctions",
                      dest="ownership_near_junctions", action="store_false",
                      help="restore the old behaviour, where local 3-D branch "
                           "ownership was skipped at any point with a "
                           "topologically-adjacent branch nearby. Near a junction "
                           "one always is, so the watershed never ran and no "
                           "`unresolved branch overlap` was raised at exactly the "
                           "places two lumens fuse: the merged blob was measured "
                           "and accepted")
    radp.add_argument("--min-blob-voxels", type=int, default=None,
                      help="smallest section, in segmentation voxels, that may be "
                           "measured at all (default 12, i.e. a 2-voxel radius). "
                           "The estimator itself holds to ~12%% down to a 1-voxel "
                           "radius -- measured against analytic cylinders swept over "
                           "sub-voxel offsets -- and breaks below that, so 4 is the "
                           "resolution limit. The default is inherited from the "
                           "shape-fitting use in `crosssection.measure`, which needs "
                           "a minor eigenvalue; the radius only needs a perimeter")
    radp.add_argument("--grow-radii", type=float, default=None,
                      help="ceiling on how far one window may double, in multiples "
                           "of that point's own radius. Unbounded (the default) a "
                           "window grows to --max-half whenever the blob touches "
                           "the border, which for a plane that is not transverse "
                           "means every time -- so raising --max-half to reach the "
                           "widest vessels makes every *narrow* one expensive "
                           "before it is rejected anyway. Pair '--max-half 192 "
                           "--grow-radii 4' to get the wide sections without that "
                           "cost")
    radp.add_argument("--junction-mask-radii", type=float, default=None,
                      help="second ceiling on a junction run, in multiples of the "
                           "node's own input radius, applied alongside "
                           "--junction-mask-max-fraction so the tighter wins "
                           "(default 2.0). A junction's influence is a few parent "
                           "radii wide wherever it sits, so a fraction of the "
                           "segment over-masks a short branch and under-masks a "
                           "long one. Pass 0 to disable and leave the fraction "
                           "alone")
    radp.add_argument("--no-perimeter-correction", dest="perimeter_correction",
                      action="store_false",
                      help="write the raw perimeter/2pi reading instead of "
                           "correcting its digitisation bias. cv2 traces pixel "
                           "centres as a chain code, which under-reads every "
                           "section below 9.6 voxels and by 27% at 1.5; the "
                           "correction inverts that and is on by default")
    radp.add_argument("--max-half", type=int, default=128,
                      help="largest half-width of the sampling window, in voxels")
    radp.add_argument("--stability-centroid-mode", choices=("drift", "offset"),
                      default="drift", help="check centroid movement across the slab "
                      "(drift), or restore the old distance-from-centreline check (offset)")
    radp.add_argument("--workers", type=int, default=1,
                      help="parallel measurement processes; each keeps the full branch "
                           "context, with final filtering performed on the combined graph")
    radp.add_argument("--max-radius-factor", type=float, default=2.0,
                      help="reject a radius above this multiple of its robust local "
                           "surrounding radius, then interpolate it")
    radp.add_argument("--no-branch-aware", dest="branch_aware", action="store_false",
                      help="use the legacy single-plane selector without tangent "
                           "validation or local 3-D branch ownership")
    radp.add_argument("--root-edge", type=int, action="append", default=[],
                      help="edge id to root its connected component for bifurcation "
                           "parent/child inference; repeat for multiple components")
    radp.add_argument("--tangent-search-deg", type=float, default=20.0,
                      help="maximum angular correction searched for an unstable "
                           "cross-section (default: 20 degrees)")
    radp.add_argument("--transverse-axis-ratio", type=float, default=None,
                      help="how elliptical a section may be before the tangent "
                           "search runs even though the section was stable. The "
                           "three-plane stability test cannot see tilt -- on a "
                           "straight vessel an oblique plane scores a perfect "
                           "1.000 and over-reads the radius by up to a quarter -- "
                           "so an elongated section is re-cut over the search cone "
                           "and the shortest boundary wins. Raise it to disable "
                           "that (cheaper, and wrong on every oblique cut); "
                           "default 1.10")
    radp.add_argument("--fallback-policy", default=None,
                      choices=("retain", "rescale", "drop"),
                      help="what to do with a segment that yielded no trustworthy "
                           "cross-section: 'retain' keeps its uncorrected input radii, "
                           "'rescale' puts them on the measured scale, 'drop' "
                           "interpolates across it from its measured neighbours "
                           "(default: rescale)")
    radp.add_argument("--fallback-taper", action="store_true",
                      help="with --fallback-policy drop, ramp the calibre between "
                           "the two junction anchors instead of holding it constant; "
                           "off by default because a taper across a span where "
                           "nothing was measured is a model, not an observation")
    radp.add_argument("--junction-flare", default=None,
                      choices=("none", "parent", "all"),
                      help="keep the measured section inside a junction run instead "
                           "of interpolating it. The section really is ~32%% wider "
                           "at a branched node, but it is wider because the lumens "
                           "are continuous there, so 'all' counts the shared region "
                           "once per branch; 'parent' attributes it once "
                           "(default: none)")
    radp.add_argument("--no-junction-parent-profile",
                      dest="junction_parent_profile", action="store_false",
                      help="stop carrying a parent's measured trend across its "
                           "junction run and hold the last measured value flat "
                           "instead. The trend extrapolation is on by default; it "
                           "authors nothing for ostial daughters")
    radp.add_argument("--bif-taper", dest="bifurcation_tapers",
                      action="store_true",
                      help="author a parent-through / carina profile across junction "
                           "runs instead of interpolating them. Off by default: the "
                           "carina model is unvalidated against the segmentation")
    radp.add_argument("--continuation-ratio", type=float, default=None,
                      help="a junction daughter at or above this fraction of its "
                           "parent's calibre is the vessel continuing, not a branch "
                           "emerging from it, so it is carried through rather than "
                           "tapered to the carina tip (default: 0.70)")
    radp.add_argument("--carina-tip-factor", type=float, default=0.1,
                      help="daughter radius at the shared node as a fraction of its "
                           "first trusted post-ostial radius (default: 0.1)")
    _add_roots_json(radp)

    crop = sub.add_parser("crop", parents=[common],
                          help="crop the tree: by Strahler order, by absolute take-off "
                               "radius, or by side-branch radius relative to the ostium "
                               "of a named main vessel")
    crop.add_argument("--crop-json", default=None,
                      help="the crop sidecar: the named main vessels, the rule and the "
                           "resulting selection. Read first if it exists, and rewritten "
                           "with what this run selected. This file is the durable "
                           "artefact; the .am is derived from it")
    crop.add_argument("--min-strahler", type=int, default=None,
                      help="drop every branch below this Strahler order; needs a graph "
                           "that carries orders (run `optimise --order` first)")
    crop.add_argument("--min-ostium-um", type=float, default=None,
                      help="drop every branch whose take-off radius is below this, in "
                           "um; needs no main vessels")
    crop.add_argument("--ratio", type=float, default=None,
                      help="drop a side branch whose take-off radius is below this "
                           "fraction of the ostial radius of the main vessel it "
                           "descends from")
    crop.add_argument("--ratio-denominator", type=float, default=None,
                      help="the same threshold as 1/N -- coronary_sdf's --ratios, one "
                           "value per run")
    crop.add_argument("--prune-unattributed", action="store_true",
                      help="also drop subtrees that descend from no named main vessel; "
                           "off by default, because an unattributed subtree is usually "
                           "one nobody has annotated yet, not one nobody wants")
    crop.add_argument("--takeoff-factor", type=float, default=2.0,
                      help="local radii to walk along a branch from its junction before "
                           "reading its take-off radius; never read it *at* the junction, "
                           "where the measurement is of the whole carina. This is a "
                           "distance where coronary_sdf skips a fixed number of "
                           "contours, so the radii here will not match its own tables")
    crop.add_argument("--root-edge", type=int, action="append", default=[],
                      help="edge id to root its connected component, deciding which end "
                           "is proximal; repeat for multiple components")
    crop.add_argument("--no-reorder", dest="reorder", action="store_false",
                      help="write the pre-crop Strahler orders instead of recomputing "
                           "them; they describe the tree as it was, so this is only "
                           "right when `skeleton_analysis` is unavailable")
    crop.add_argument("--replay", action="store_true",
                      help="drop exactly the segments the sidecar recorded rather than "
                           "re-evaluating the rule")
    crop.add_argument("--report-csv", default=None,
                      help="one row per dropped take-off, beside coronary_sdf's "
                           "pruned_branches.csv")
    _add_roots_json(crop)

    # Still its own command, now that `--pick-roots` exists on the three commands that
    # produce a skeleton: this is how a graph that *already exists* gets rooted, and
    # how a previous pick is revised without re-deriving anything. The flag is opt-in
    # over there for the same reason -- `skeletonise-all` spends up to 25 minutes
    # deriving skeletons, and a 3-D window blocking half way through is not something
    # to walk into by accident, so its picker only opens once the work is done.
    # Its own positional rather than `graph_arg`: this is the one command that takes
    # **several** skeletons, so a left-tree graph and a right-tree graph can be rooted
    # in one session, against one mask labelling, into one sidecar.
    pick = sub.add_parser(
        "pick-roots", parents=[seg_common],
        help="click each tree's inlet in 3-D and record the roots",
        description=(
            "Click each tree's inlet in 3-D and record the roots. "
            "--seg is OPTIONAL and the picker never reads it: the window is built "
            "from the graph alone, and the root is recorded by world coordinate. It "
            "does one thing -- it anchors the tree numbering to the mask labelling "
            "(and to Left_Tree/Right_Tree, where the mask names them) instead of to "
            "the graph's own component order. Skip it and the roots are identical; "
            "the sidecar just says tree_source 'graph' rather than 'mask'. Passing "
            "it costs a full decode and a connected-component labelling, which at "
            "--stride 1 on a whole heart is tens of minutes."
        ),
    )
    pick.add_argument("graph", nargs="+",
                      help="ASCII Amira .am spatial graph; repeat to root several "
                           "skeletons (e.g. a left-tree and a right-tree graph) into "
                           "one sidecar")
    pick.add_argument("--roots-json", default=None,
                      help="write the chosen roots here; read first if it exists so a "
                           "previous pick can be revised rather than redone")
    pick.add_argument("--out", default=None,
                      help="also write the graph with Strahler orders taken from the "
                           "chosen roots (.am)")
    pick.add_argument("--auto", action="store_true",
                      help="skip the window and record the automatic roots -- the "
                           "largest-radius edge of each component, then its "
                           "lower-coordination endpoint")
    pick.add_argument("--color-by", default="strahler", choices=("strahler", "none"),
                      help="colour each segment by Strahler order, so the trunk is "
                           "obvious before anything is clicked")
    pick.add_argument("--style", default="contour",
                      choices=("contour", "tube", "lines"),
                      help="'contour' rings plus centreline dots; 'tube' one solid "
                           "tube per segment (lighter, and easier to click); 'lines' "
                           "the graph edges as bare polylines with a point at each "
                           "node -- nothing swept, and the only one that stays "
                           "responsive on a whole coronary tree")
    pick.add_argument("--n-sides", type=int, default=16,
                      help="ring tessellation for --style contour")
    pick.add_argument("--ring-stride", type=int, default=1,
                      help="draw a ring every N centreline points")
    # `seg_common`, not a hand-picked subset of it: `--seg` is optional here -- the
    # graph alone is enough to pick a root, and the mask only anchors the tree indices
    # to the labelling -- but the moment it is given, `_decoded` runs the same
    # `_open_lattice` as every other command and reads every flag that parser defines.
    # Re-declaring four of the five is what left `--voxel-um` missing and turned a
    # `pick-roots --seg` run into an AttributeError.
    pick.add_argument("--min-component-voxels", type=int, default=0,
                      help="with --seg, ignore mask components smaller than this")
    pick.add_argument("--screenshot", default=None,
                      help="render the first tree to a PNG instead of opening a window")
    pick.add_argument("--preselect", type=int, default=None,
                      help="with --screenshot, the edge index to draw as chosen")

    score = sub.add_parser("score", parents=[graph_arg, seg_common],
                           help="the five super-metric terms and M_S for one graph")
    score.add_argument("--bb-threshold", type=float, default=900.0)
    score.add_argument("--no-tree-chi", action="store_true")
    _add_per_tree_score_args(score)

    export = sub.add_parser("mask-export", parents=[seg_common],
                            help="write the corrected mask out as .am or .tif")
    export.add_argument("--out", required=True,
                        help="destination; .am writes an HxByteRLE Amira lattice, "
                             ".tif a uint8 stack")
    export.add_argument("--field", default="Labels",
                        help="field name to write into the .am")
    return p


def _load(paths):
    """One skeleton, or several merged into one graph holding a tree each.

    A single path is read as it always was. Several go through
    :func:`~.components.merge_graphs`, so every command downstream still has exactly
    one graph to work on -- and one ``--out`` to write -- while the sources stay
    distinguishable through the ``tree`` field that `crop`, per-tree scoring and the
    roots sidecar already read.

    **A merge renumbers ids.** ``--segment``, ``--root-edge`` and anything else quoting
    a number read off a viewer names a different object afterwards, so it is said out
    loud rather than left to be discovered. Nothing is renumbered for a single graph,
    which is what keeps the usual case exactly as it was.
    """
    from .adapter import read_triple
    from .components import graph_paths, merge_graphs
    from .graphmodel import EditableGraph

    paths = graph_paths(paths)
    if not paths:
        raise SystemExit("no skeleton given")
    if len(paths) == 1:
        graph = EditableGraph(read_triple(paths[0]))
        print(f"{Path(paths[0]).name}: {len(graph.segments)} segments, "
              f"{len(graph.nodes)} nodes")
        return graph

    triples = []
    for path in paths:
        triple = read_triple(path)
        print(f"{Path(path).name}: {len(triple.segments)} segments, "
              f"{len(triple.nodes)} nodes")
        triples.append(triple)
    merged, trees = merge_graphs(triples)
    graph = EditableGraph(merged)
    named = ", ".join(f"{Path(p).name} -> tree {'/'.join(str(t) for t in ts)}"
                      for p, ts in zip(paths, trees))
    print(f"  merged {len(paths)} skeletons into {len(graph.segments)} segments, "
          f"{max(max(t) for t in trees) + 1} tree(s): {named}")
    print("  ! ids are renumbered by the merge: --segment and --root-edge name "
          "positions in the merged graph, not in any one input")
    return graph


def _correct_units(graph, frame, args) -> float | None:
    """Put a loaded graph onto the frame's scale when ``--voxel-um`` corrected it.

    Returns the voxel size to stamp into anything written from this graph, or None
    when nothing was corrected and the file's own units still apply.

    A graph that already records this scale is left alone. The factor is derived from
    the *segmentation's* bounding box, which does not change when a graph is written,
    so a command run twice -- or run on its own output -- would otherwise apply the
    same correction again, and the result would still sit inside the mask and still
    pass every check. See `amira.VOXEL_STAMP`.
    """
    if frame is None:
        return None
    from ..amira import read_voxel_stamp
    from ..frame import rescale_triple
    from .components import graph_paths

    paths = graph_paths(getattr(args, "graph", None))
    stamps = [read_voxel_stamp(path) for path in paths]
    already = [
        v for v in stamps
        if v is not None and abs(v - frame.voxel_um) <= 1e-6 * max(v, 1.0)
    ]
    if stamps and len(already) == len(stamps):
        print(f"  the skeleton already records {frame.voxel_um:.4f} um; not rescaled again")
        return frame.voxel_um
    if any(v is not None for v in stamps):
        raise ValueError(
            f"skeleton voxel scale does not match the {frame.voxel_um:.4f} um frame: "
            + ", ".join(
                f"{Path(p).name} " + (f"is at {v:.4f} um" if v is not None else "is unmarked")
                for p, v in zip(paths, stamps)
            )
            + ". Use graphs recorded at the same scale and a matching --voxel-um, "
              "or omit --voxel-um to use their recorded scale."
        )
    if not frame.corrected:
        return None
    rescale_triple(graph.triple, frame)
    print(f"  skeleton rescaled onto {frame.voxel_um:.4f} um "
          f"(x{float(frame.world_scale.mean()):.6f})")
    return frame.voxel_um


def _save(graph, out, source, voxel_um=None):
    if out is None:
        print("\n(dry run: pass --out to write the result)")
        return
    from .amira_write import write_spatial_graph
    from .components import graph_paths

    # The `Parameters` block -- and with it the TransformationMatrix -- comes from the
    # first input. Several merged skeletons of one dataset share it; one that does not
    # was never going to overlay the others in the first place.
    #
    # `voxel_um` is stamped into that block when the units were corrected on load, so
    # the output says what scale it is in and is not corrected a second time. Left
    # None, any stamp the source carried is inherited untouched.
    paths = graph_paths(source)
    write_spatial_graph(graph.to_spatial_graph(), out,
                        parameters_from=paths[0] if paths else None,
                        voxel_um=voxel_um)
    print(f"\nwrote {out}")



def cmd_segment_diagnosis(args) -> int:
    from . import diagnostics

    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    _correct_units(graph, frame, args)
    diagnostics.segment_report(graph, frame, labels, args.segment,
                               triple=graph.triple)
    return 0


def cmd_junction_mask(args) -> int:
    from . import diagnostics

    # Deliberately loud: `_BranchContext.rivals` scales its search by the radius it
    # is handed, so running this on a graph the pass *wrote* asks a different
    # question than the pass asked -- and answers it with the radius under suspicion.
    print("  note: give this the graph the measurement CONSUMED, not the one it "
          "wrote; the adjacency search is scaled by the radii in the file")
    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    _correct_units(graph, frame, args)
    diagnostics.junction_terms(graph, frame, labels, args.segment)
    return 0


def cmd_reformat_radius(args) -> int:
    from . import diagnostics

    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    _correct_units(graph, frame, args)
    diagnostics.reformat_radius(graph, frame, labels, args.segment,
                                size_px=args.size_px)
    return 0


def cmd_ostium_flare(args) -> int:
    from . import diagnostics

    graph = _load(args.graph)
    labels, frame, _ = _open_lattice(args)
    _correct_units(graph, frame, args)
    diagnostics.ostium_flare(graph, frame, labels)
    return 0


def cmd_report(args) -> int:
    import numpy as np

    from .reconnect import gaps as gaps_mod

    graph = _load(args.graph)
    components = graph.components()
    print(f"  components : {len(components)}  sizes={[len(c) for c in components]}")
    print(f"  free ends  : {len(graph.endpoints())}")

    radii = [float(np.median(graph.radii(s['id']))) for s in graph.segments]
    if radii:
        print(f"  radii (um) : min={min(radii):.0f} "
              f"median={np.median(radii):.0f} max={max(radii):.0f}")

    records = gaps_mod.find(graph)
    print(f"  intra-segment gaps: {len(records)}")
    for record in sorted(records, key=lambda r: -r["gap_mm"])[:10]:
        print(f"    segment {record['seg_id']:>5} at point {record['i']:>5}: "
              f"{record['gap_mm']:.2f} mm")

    from . import interpolation as interp

    if interp.has_flags(graph):
        found = interp.spans(graph)
        n = sum(s.n_points for s in found)
        print(f"  interpolated: {n} point(s) in {len(found)} span(s)")
    else:
        print("  interpolated: not checked -- run `flag-interpolation`"
              f"{_interpolation_hint(graph)}")
    return 0


def _interpolation_hint(graph) -> str:
    """", which would flag N point(s)" -- or nothing, when there is nothing to say.

    Detection is cheap and read-only, so an unflagged graph can still be *told* what it
    is carrying. Acting on it silently is the thing that would be wrong: half the
    synthetic fixtures in the test suite are straight linear-ramp segments, and a
    detector that ran itself inside every pipeline would change what they mean.
    """
    from . import interpolation as interp

    try:
        found = interp.detect(graph)
        # The three point-based signatures cannot see a zero-point jump, so without
        # this the loudest artefact in the graph is the one the hint stays silent
        # about: LADAF-28 reported "3 points in 1 span" while carrying 24 invented
        # bridges. The geometric half of the jump gate needs no mask, so it is free
        # to run here; confirming them against the segmentation is not.
        jumps = interp.candidate_jumps(graph)
    except Exception:  # noqa: BLE001 - a hint must never break the command it decorates
        return ""
    parts = []
    if found.n_flagged:
        parts.append(f"{found.n_flagged} point(s) in {len(found.spans)} span(s)")
    if jumps:
        total = sum(s.length_um for s in jumps) / 1000.0
        parts.append(f"{len(jumps)} candidate unsampled jump(s), {total:.1f} mm, "
                     f"needing --seg to confirm")
    if not parts:
        return ""
    return " (it would flag " + "; ".join(parts) + ")"


def _warn_unflagged(graph) -> None:
    """Say so, once, when a measuring command is about to read invented points."""
    from . import interpolation as interp

    if interp.has_flags(graph):
        return
    hint = _interpolation_hint(graph)
    if hint:
        print(f"note: this graph has not been through `flag-interpolation`{hint}.")
        print("      Those points are being measured as if they were real.")


def cmd_gaps(args) -> int:
    from .reconnect import gaps as gaps_mod

    graph = _load(args.graph)
    records = gaps_mod.find(graph, big_jump_ratio=args.big_jump_ratio,
                            min_gap_um=args.min_gap_um)
    print(f"\n{len(records)} gap(s) to fill")
    for record in sorted(records, key=lambda r: -r["gap_mm"]):
        print(f"  segment {record['seg_id']:>5} at point {record['i']:>5}: "
              f"{record['gap_mm']:.2f} mm -> {record['n_inserted']} points")

    if not records:
        # Zero here means "no jump *inside* an edge", which on a skeletonised graph
        # is true by construction and says nothing about whether the graph is in one
        # piece. Reporting the bare 0 reads as "nothing to reconnect" and is how this
        # command gets blamed for a break it was never looking for.
        print(_nothing_to_fill(graph))
        # Still write, so a chain that declared this step's --out can carry on: the
        # graph is unchanged, but "no gaps" is a pass, not a missing file.
        _save(graph, args.out, args.graph)
        return 0

    filled, _patch = gaps_mod.apply(graph, big_jump_ratio=args.big_jump_ratio,
                                    min_gap_um=args.min_gap_um)
    print(f"\nfilled {filled} gap(s)")
    _save(graph, args.out, args.graph)
    return 0


def _nothing_to_fill(graph) -> str:
    """What `gaps` did *not* look at, for the case where it found nothing."""
    components = len(graph.components())
    ends = len(graph.endpoints())
    if components < 2 and ends == 0:
        return "  (and the graph is in one piece with no free ends)"
    return (
        "  (gaps only looks inside a single edge. This graph has "
        f"{components} component(s) and {ends} free end(s) -- run `connect` for those.)"
    )


def cmd_flag_interpolation(args) -> int:
    from . import interpolation as interp

    graph = _load(args.graph)

    labels = frame = None
    components = None
    if args.seg:
        # Only when asked for: the lattice check is ground truth but it costs a decode,
        # and the two geometric signatures need nothing but the graph.
        labels, frame, path = _open_lattice(args)
        print(f"  checking against {Path(path).name}")
        if not args.no_jumps:
            # The jump signature needs the mask *labelled*, not merely sampled. That is
            # a second pass over the lattice, but a streaming one -- seconds, against the
            # decode that has already happened.
            from .reconnect.geodesic import components as components_mod

            def indexing(done, total):
                print(f"    labelling plane {done}/{total}", end="\r")

            components = components_mod.build(labels, progress=indexing)
            print(" " * 40, end="\r")
            print(f"  {components.describe()}")

    kw = {}
    if args.anchor_jump is not None:
        kw["anchor_jump"] = args.anchor_jump
    if args.min_span_points is not None:
        kw["min_span_points"] = args.min_span_points
    if args.min_jump_um is not None:
        kw["min_step_um"] = args.min_jump_um

    found = interp.detect(graph, labels=labels, frame=frame,
                          components=components, **kw)
    print()
    print(found.describe(show_spans=args.show_spans))
    if components is None and not args.no_jumps:
        # Say the number rather than only inviting the flag. Without the mask this
        # command cannot confirm a jump, but staying quiet about the candidates let
        # a graph with 24 invented bridges read as one flagged span.
        screen = interp.candidate_jumps(
            graph, **({"min_step_um": args.min_jump_um}
                      if args.min_jump_um is not None else {}))
        if screen:
            total = sum(s.length_um for s in screen) / 1000.0
            print(f"\n{len(screen)} step(s) look like unsampled jumps "
                  f"({total:.1f} mm in total), but confirming that they cross a "
                  f"break in the segmentation needs --seg.")
            print("  They carry no points, so none of the signatures above can "
                  "see them.")

    interp.annotate(graph, found)
    splittable = [s for s in found.spans
                  if s.is_jump or s.n_points >= interp.MIN_SPAN_POINTS]
    if splittable:
        print(f"\n{len(splittable)} span(s) are long enough to be treated as a break by "
              "`connect` and `surface`;")
        print("  the rest are single invented radii, masked out of measurement only.")
    _save(graph, args.out, args.graph)
    return 0


def _split_interpolation(args, graph):
    """Cut out every flagged fill so the breaks they hid become real. Returns records.

    Nothing happens on a graph that has not been through `flag-interpolation`, which is
    the same rule every other consumer follows -- but it says so, because silently
    proposing reconnections across a fill that is already there is exactly the failure
    this is meant to remove.
    """
    from . import interpolation as interp

    if getattr(args, "keep_interpolation", False):
        return []
    if not interp.has_flags(graph):
        hint = _interpolation_hint(graph)
        if hint:
            print(f"\nnote: not flagged for interpolation{hint}.")
            print("      Run `flag-interpolation` first to let those breaks be seen.")
        return []

    records = interp.split_flagged(graph)
    if not records:
        return []
    print(f"\nremoved {len(records)} interpolated fill(s) so the break is visible:")
    for record in records:
        where = "whole edge" if record.whole_edge else "inside an edge"
        print(f"    segment {record.seg_id:>5} ({where}): "
              f"{record.length_um / 1000.0:.2f} mm between nodes "
              f"{record.node1} and {record.node2}")
    whole = sum(1 for r in records if r.whole_edge)
    if whole:
        # Worth saying rather than leaving to be discovered: those two junctions drop to
        # degree two, and every proposer here starts from a degree-one node.
        print(f"  {whole} of those joined two junctions, so no free end was created -- "
              "no end-to-end\n  proposal can reach them.")
    print(f"  components now {len(graph.components())}, "
          f"free ends {len(graph.endpoints())}")
    return records


def _restore_interpolation(graph, records) -> None:
    """Put back whatever the image did not replace, still flagged."""
    if not records:
        return
    from . import interpolation as interp

    restored = interp.restore_unbridged(graph, records)
    replaced = len(records) - len(restored)
    if replaced:
        # "their two sides are joined again", not "a bridge replaced them": the test is
        # connectivity, and a bridge elsewhere in the component would satisfy it too.
        # Putting Avizo's straight line back on top of a real join would add a loop.
        print(f"\n{replaced} fill(s) left out -- their two sides are joined again")
    if restored:
        print(f"{len(restored)} fill(s) restored -- nothing bridged them, so Avizo's "
              "line goes back")
        print("  (still flagged, so they stay out of every measurement)")


def cmd_connect(args) -> int:
    from .reconnect import apply_bridges, endpoints, summarise, tjunction

    # Checked before the proposal pass rather than after it: being told you forgot
    # a flag is much less annoying than being told it once the search has run.
    if args.dpc and not args.raw:
        raise SystemExit("connect --dpc needs --raw <directory of image slices>")
    if args.cfc_model and not args.dpc:
        raise SystemExit("connect --cfc-model also needs --dpc")
    if args.cfc_model and args.dpc_learned:
        raise SystemExit("--cfc-model and --dpc-learned are mutually exclusive")
    if args.geodesic and args.dpc:
        raise SystemExit(
            "--geodesic and --dpc are mutually exclusive: both decide the same "
            "candidates, and running one over the other's topology would leave a "
            "result neither of them chose"
        )
    if args.geodesic:
        return cmd_connect_geodesic(args)
    if args.out_seg:
        raise SystemExit("--out-seg is only meaningful with --geodesic")

    graph = _load(args.graph)
    print(f"  components before: {len(graph.components())}")

    # Before anything is proposed: a fill that is still in place makes its own break
    # invisible, so every gate downstream would be answering the wrong question.
    split_records = _split_interpolation(args, graph)

    kw = {}
    if args.cone_deg is not None:
        kw["cone_angle_deg"] = args.cone_deg
    if args.reach_factor is not None:
        kw["cone_length_factor"] = args.reach_factor
    if args.radius_ratio is not None:
        kw["radius_ratio_max"] = args.radius_ratio
    if args.tortuosity is not None:
        kw["tortuosity_max"] = args.tortuosity

    proposals = []
    applied = []
    if args.dpc:
        # The paper's stages are topology-dependent. Apply each accepted stage
        # before proposing the next so components/endpoints are genuinely fresh.
        backbone_nodes = _backbone_nodes(
            graph, args.backbone_components, trees=args.backbone_trees
        )
        context = None
        for reconnect_type in (1, 2):
            stage_stats: dict = {}
            stage = endpoints.propose(
                graph, same_component=args.same_component, keep_rejected=True,
                stats=stage_stats, reconnection_type=reconnect_type,
                backbone_nodes=backbone_nodes, **kw
            )
            if reconnect_type == 2:
                for bridge in stage:
                    _orient_type2(bridge, backbone_nodes)
            print(f"\nType {reconnect_type} endpoint reconnection:")
            print(summarise(stage, stage_stats))
            accepted = [b for b in stage if b.accepted]
            context = _refine_with_dpc(args, graph, accepted, context=context)
            accepted = [b for b in accepted if b.accepted]
            if accepted:
                apply_bridges(graph, accepted, label=f"DPC Type {reconnect_type}")
                applied.extend(accepted)
            proposals.extend(stage)

        if args.tjunction:
            stage_stats = {}
            stage = tjunction.propose(
                graph, same_component=args.same_component, keep_rejected=True,
                stats=stage_stats, backbone_nodes=backbone_nodes,
                target_backbone_only=True, **kw
            )
            print("\nType 3 endpoint-to-vessel reconnection:")
            print(summarise(stage, stage_stats))
            accepted = [b for b in stage if b.accepted]
            context = _refine_with_dpc(args, graph, accepted, context=context)
            accepted = [b for b in accepted if b.accepted]
            if accepted:
                apply_bridges(graph, accepted, label="DPC Type 3")
                applied.extend(accepted)
            proposals.extend(stage)
    else:
        end_stats: dict = {}
        proposals = endpoints.propose(
            graph, same_component=args.same_component, keep_rejected=True,
            stats=end_stats, **kw
        )
        print("\nend-to-end:")
        print(summarise(proposals, end_stats))
        if args.tjunction:
            tj_stats: dict = {}
            tj = tjunction.propose(
                graph, same_component=args.same_component, keep_rejected=True,
                stats=tj_stats, **kw
            )
            print("\nend-to-vessel (T-junction):")
            print(summarise(tj, tj_stats))
            proposals += tj
        applied = [b for b in proposals if b.accepted]
        if applied:
            apply_bridges(graph, applied)

    print(f"\n{len(applied)} bridge(s) applied:")
    for bridge in applied:
        print(f"    {bridge}")
    if args.show_rejected:
        print("\nrefused:")
        for bridge in proposals:
            if not bridge.accepted:
                print(f"    {bridge}")
    if applied:
        print(f"\ncomponents now {len(graph.components())}")
    _restore_interpolation(graph, split_records)
    _save(graph, args.out, args.graph)
    return 0


def cmd_connect_geodesic(args) -> int:
    """``connect --geodesic``: repair the graph and the segmentation together.

    The paired-output rule is enforced here rather than in the library, because it
    is a statement about *files*: a repaired graph written next to the original
    mask is two documents that disagree about where the vessels are, and nothing
    downstream can tell which one to believe. Writing either therefore requires
    both.

    Dry run is still the default and still means what it does everywhere else --
    with neither output path given, everything is proposed, searched and decided,
    and nothing is written.
    """
    from .maskedit import MaskEdits, MaskSource
    from .reconnect import geodesic
    from .reconnect.geodesic import audit

    if bool(args.out) != bool(args.out_seg):
        raise SystemExit(
            "connect --geodesic writes the graph and the mask together: pass both "
            "--out GRAPH.am and --out-seg SEG.am, or neither for a dry run"
        )

    graph = _load(args.graph)
    print(f"  components before: {len(graph.components())}")
    split_records = _split_interpolation(args, graph)

    labels, frame, seg_path = _open_lattice(args)
    source = labels if isinstance(labels, MaskSource) else MaskSource(
        labels, MaskEdits(ny=int(frame.seg_dims[1]), nx=int(frame.seg_dims[0]),
                          nz=int(frame.seg_dims[2]), source=str(seg_path))
    )

    stack = None
    if args.raw:
        stack, frame = _open_raw(args)

    print("\nindexing mask components (streaming, one plane at a time)...")

    def indexing(done, total):
        print(f"    plane {done}/{total}", end="\r")

    index = geodesic.components.build(source, progress=indexing)
    print(" " * 40, end="\r")
    print(f"  {index.describe()}")

    params = geodesic.GeodesicParams(
        alternatives=max(int(args.alternatives), 1),
        max_unsupported_factor=float(args.max_unsupported_factor),
        dark_lumen=not args.dpc_bright_lumen,
        mask_endpoints=not args.no_mask_endpoints,
    )
    if args.describe_radii is not None:
        params.describe_radii = float(args.describe_radii)
    if args.min_lobe_voxels is not None:
        params.min_lobe_voxels = int(args.min_lobe_voxels)
    if args.tortuosity is not None:
        # One flag, both places: the proposal gate and the re-check on the route the
        # search actually returns.
        params.tortuosity_max = float(args.tortuosity)
    kw = {}
    if args.cone_deg is not None:
        kw["cone_angle_deg"] = args.cone_deg
    if args.reach_factor is not None:
        kw["cone_length_factor"] = args.reach_factor
    if args.radius_ratio is not None:
        kw["radius_ratio_max"] = args.radius_ratio
    if args.tortuosity is not None:
        kw["tortuosity_max"] = args.tortuosity

    def report(done, total, candidate):
        print(f"    [{done}/{total}] {candidate!r}")

    def lobe_report(done, total, found):
        print(f"    free end {done}/{total}, {found} mask end(s) so far", end="\r")

    if params.mask_endpoints:
        print("\nsearching the mask for lumen no centreline describes...")
    print("\nplanning:")
    plan = geodesic.plan(
        graph, index, frame, stack=stack, params=params,
        same_component=args.same_component,
        tjunction=not args.no_tjunction_geodesic, gate_kwargs=kw, progress=report,
        lobe_progress=lobe_report,
    )
    print()
    if plan.lobe_report is not None:
        print(f"  {plan.lobe_report.describe()} "
              f"({plan.lobe_report.seconds:.1f}s)")
    print(plan.summarise())

    # An operator's earlier rulings, if this run was given a decisions file that
    # already exists. Matched by endpoint rather than by position, so a re-run on
    # a graph that has moved on cannot apply a decision to the wrong candidate.
    approved = []
    if args.decisions_json and Path(args.decisions_json).exists():
        document = audit.load_decisions(args.decisions_json)
        approved, unmatched = audit.apply_decisions(plan, document)
        print(f"\n{len(approved)} reviewed candidate(s) approved from "
              f"{args.decisions_json}")
        if unmatched:
            print(f"  {len(unmatched)} ruling(s) no longer match any candidate "
                  "(the graph has changed since they were made)")

    if args.show_rejected:
        print("\nrefused:")
        for candidate in plan.rejected():
            print(f"    {candidate!r}")

    if args.review_json:
        path = audit.write(args.review_json,
                           audit.review_document(plan, frame, graph=graph))
        print(f"\nwrote {path} ({len(plan.for_review())} candidate(s) to review)")

    if not args.out:
        applied = []
        print("\n(dry run: pass --out and --out-seg to write the result)")
    else:
        applied = geodesic.apply_plan(graph, plan, source, frame, reviewed=approved)
        print(f"\n{sum(1 for a in applied if a.ok)} repair(s) applied:")
        for result in applied:
            print(f"    {result.describe()}")
        print(f"\ncomponents now {len(graph.components())}")
        print(f"  edge provenance: {geodesic.origin_counts(graph)}")

        _restore_interpolation(graph, split_records)
        _save(graph, args.out, args.graph)

        print(f"\nwriting the repaired segmentation to {args.out_seg}...")

        def encoding(done, total):
            print(f"    plane {done}/{total}", end="\r")

        written = geodesic.write_segmentation(args.out_seg, source, frame,
                                              progress=encoding)
        print(" " * 40, end="\r")
        from ..rle_write import describe as describe_write

        print(f"  {describe_write(written)}")

    if args.decisions_json:
        path = audit.write(
            args.decisions_json,
            audit.decisions_document(plan, frame, graph=graph, applied=applied,
                                     arguments=vars(args)),
        )
        print(f"wrote {path}")
    return 0


def _backbone_nodes(graph, count: int = 2, *, trees=None) -> set[int]:
    """Nodes of the trees the blood actually flows through.

    With an explicit `trees` -- tree indices, from ``--backbone-trees`` -- the backbone
    is the union of those trees' segments, which is a statement about the anatomy.
    Without one it falls back to the `count` largest components by centreline point
    count: the same guess as before, now visible on the command line instead of
    hard-coded, so a graph fragmented into three real trees or one where a repair has
    already merged the left and right can be corrected rather than mis-staged.

    Deliberately **not** read from the ``tree`` field by default. Doing so would
    silently change what ``connect --dpc`` proposes for anyone who happened to run
    ``skeletonise-all --per-tree`` upstream, which is exactly the kind of implicit
    change that makes two runs incomparable.
    """
    if trees:
        from . import components as comp

        wanted = {int(t) for t in trees}
        of_edge = comp.tree_of_edge(graph)
        chosen = [{sid for sid in of_edge if of_edge[sid] in wanted}]
        if not chosen[0]:
            print(f"  no segment carries tree(s) {sorted(wanted)}; "
                  f"falling back to the {count} largest components")
            chosen = []
    else:
        chosen = []
    if not chosen:
        chosen = sorted(
            graph.components(),
            key=lambda segs: -sum(len(graph.coords(sid)) for sid in segs),
        )[:count]
    nodes = set()
    for segs in chosen:
        for sid in segs:
            segment = graph.segment(sid)
            nodes.update((segment["node1"], segment["node2"]))
    return nodes


def _orient_type2(bridge, backbone_nodes) -> None:
    """Walk from the disconnected fragment toward the backbone, never backwards."""
    if bridge.source_node not in backbone_nodes:
        return
    bridge.source_node, bridge.target_node = bridge.target_node, bridge.source_node
    bridge.coords = bridge.coords[::-1].copy()
    bridge.radii = bridge.radii[::-1].copy()
    r0, r1 = bridge.metrics.get("r_source"), bridge.metrics.get("r_target")
    if r0 is not None and r1 is not None:
        bridge.metrics["r_source"], bridge.metrics["r_target"] = r1, r0


def cmd_train_cfc(args) -> int:
    from .components import graph_paths
    from .reconnect.cfc import train_cfc

    segmentation = require_seg(args)
    # `train_cfc` reads the graph itself, from a path, so it gets the first. Training
    # a scan-specific model on the left tree and applying it to both is the intended
    # use; merging here would mean re-deriving a graph the trainer is about to read.
    paths = graph_paths(args.graph)
    if len(paths) > 1:
        print(f"  training on {Path(paths[0]).name} alone "
              f"({len(paths) - 1} other skeleton(s) ignored)")

    def progress(done, total):
        print(f"    extracting raw patch {done:,}/{total:,}", end="\r")

    result = train_cfc(
        paths[0], segmentation, args.raw, args.out_model,
        labels_field=args.labels_field, voxel_um=args.voxel_um,
        max_positive=args.max_positive, seed=args.seed,
        overwrite=args.overwrite, progress=progress,
    )
    print(" " * 60, end="\r")
    metrics = result["metrics"]
    print(f"CFC artifact: {result['output']}")
    print(
        "  validation: accuracy {accuracy:.3f}, sensitivity {sensitivity:.3f}, "
        "specificity {specificity:.3f}, ROC-AUC {roc_auc:.3f}, PR-AUC {pr_auc:.3f}"
        .format(**metrics)
    )
    return 0


def cmd_evaluate_dpc(args) -> int:
    import json

    from .reconnect.evaluation import evaluate_dpc

    graph = _load(args.graph)
    labels, _lattice_frame, _ = _open_lattice(args)
    stack, frame = _open_raw(args)
    report = evaluate_dpc(
        graph, stack, frame, labels, args.cfc_model, args.regions,
        omega_spec=args.omega, include_ablations=not args.no_ablations,
        case_ids=args.cases.split(",") if args.cases else None,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"selected omega: {report['selected_omega']} ({report['selection_status']})")
    print(f"wrote {destination}")
    return 0


def cmd_export_dpc_regions(args) -> int:
    from .reconnect.evaluation import export_regions

    graph = _load(args.graph)
    labels, _lattice_frame, _ = _open_lattice(args)
    stack, frame = _open_raw(args)
    written = export_regions(graph, stack, frame, labels, args.regions, args.output)
    print(f"wrote {len(written)} review files to {args.output}")
    return 0


def _open_raw(args):
    """The raw image stack and a frame that maps world um onto *its* grid.

    ``_open_lattice`` builds a frame from the lattice alone with a nominal voxel
    size, which is right for the commands that never read the greyscale. The DPC
    walk does read it, so the frame has to know the real stack shape -- otherwise
    every ROI is clipped against a fictitious volume.
    """
    from .. import amira
    from ..frame import WorldFrame
    from ..tiffstack import TiffStack

    stack = TiffStack(args.raw)
    info = amira.read_lattice_header(require_seg(args))
    nominal = args.voxel_um or stack.nominal_voxel_um or float(info.spacing[0]) / 2.0
    frame = WorldFrame.from_inputs(stack.shape, nominal, info)
    print(f"  raw stack: {stack.shape} slices/rows/cols, "
          f"voxel {frame.raw_voxel[0]:.3f} um, binned {frame.bin_factor[0]}x")
    if getattr(stack, "skipped", 0):
        # Said out loud because it used to be counted *as slices*, which shifted every
        # z index and quietly moved every raw sample onto the wrong plane.
        print(f"  ignored {stack.skipped} unsupported/sidecar file(s) in the raw directory")
    return stack, frame


def _refine_with_dpc(args, graph, bridges, *, context=None):
    """Walk every accepted proposal through the image; reject in place what fails.

    The division of labour the package was built around and never wired up: the
    geometric proposers are cheap and generate pairs, and the walk -- which has to
    read the greyscale -- only runs on pairs that already passed the cone, radius
    and tortuosity gates.
    """
    from .reconnect import dpc
    from .reconnect import roi as roi_mod
    from .reconnect.probability import FieldProbability, LearnedProbability

    if not bridges:
        print("\nDPC: no accepted proposals to walk")
        return context

    if context is None:
        labels, _lattice_frame, _seg_path = _open_lattice(args)
        stack, frame = _open_raw(args)
        params = dpc.DpcParams()
        if args.dpc_omega is not None:
            params.omega = args.dpc_omega
        params.w_distance = args.dpc_distance_weight
        params.w_cosine = args.dpc_cosine_weight
        params.neighbourhood_policy = args.dpc_neighbourhood
        if args.dpc_min_probability is not None:
            params.min_mean_probability = args.dpc_min_probability
        cfc_model = None
        if args.cfc_model:
            from .reconnect.cfc import load_model

            cfc_model = load_model(args.cfc_model)
        context = {
            "labels": labels, "stack": stack, "frame": frame, "params": params,
            "cfc_model": cfc_model, "learned_model": None,
        }
    labels, stack, frame, params = (
        context["labels"], context["stack"], context["frame"], context["params"]
    )

    spans = []
    for bridge in bridges:
        extra = None
        if bridge.kind == "tjunction":
            # Only the neighbourhood of the attachment point is a plausible
            # destination. Boxing in the whole parent vessel -- which can be
            # centimetres long -- would inflate the ROI for no gain.
            keep = max(2.0 * bridge.metrics.get("r_source", 0.0), 200.0)
            extra = roi_mod.near_target_segment(graph, bridge, keep_um=keep)
        spans.append(roi_mod.span_for(
            bridge, frame, pad_factor=args.dpc_pad_factor, extra_points_um=extra
        ))

    def report(done, total):
        print(f"    reading raw slice {done}/{total}", end="\r")

    print(f"\nDPC: walking {len(bridges)} proposal(s) through the image...")
    rois = roi_mod.build_many(stack, frame, spans, labels=labels, progress=report)
    print(" " * 40, end="\r")

    if args.dpc_learned and context["learned_model"] is None:
        context["learned_model"] = _train_probability(graph, rois)
    model = context["learned_model"]

    for n, (bridge, roi) in enumerate(zip(bridges, rois), 1):
        if roi is None:
            bridge.reject("DPC: the region of interest would be too large to read")
            continue

        if args.cfc_model:
            from .reconnect.cfc import CfcProbability

            probability = CfcProbability(
                roi, args.cfc_model, model=context["cfc_model"],
                expected_raw_shape=stack.shape
            )
        elif args.dpc_learned:
            if model is None:
                bridge.reject("DPC: no region held enough centreline to train on")
                continue
            probability = LearnedProbability(roi, model=model)
        else:
            probability = FieldProbability(roi, dark_vessels=not args.dpc_bright_lumen)

        dpc.refine(graph, roi, probability, [bridge], params=params)
        print(f"    [{n}/{len(bridges)}] {bridge}")
    return context


def _train_probability(graph, rois):
    """Fit one centreline-probability model, on whichever ROI holds the most vessel.

    The features are re-derived per ROI -- and normalised against that ROI's own
    intensity percentiles, which is what lets one model transfer between boxes with
    different exposure -- but the *classifier* is trained once. Two reasons not to
    refit per bridge: a small box supplies a few dozen labels where the point of
    using the skeleton as ground truth is that it supplies millions, and a model
    that changes between bridges makes their scores incomparable.

    Training on the *best* box rather than the first is free here, because
    `build_many` has already read them all.
    """
    from .reconnect.probability import LearnedProbability

    points = np.asarray([p[:3] for p in graph.points.values()], dtype=np.float64)
    radii = np.asarray([p[3] for p in graph.points.values()], dtype=np.float64)

    best_roi, best_inside, best_n = None, None, 0
    for roi in rois:
        if roi is None:
            continue
        inside = roi.inside(roi.to_index(points))
        n = int(inside.sum())
        if n > best_n:
            best_roi, best_inside, best_n = roi, inside, n

    if best_roi is None or best_n < 20:
        print("    no region holds enough centreline to train on; "
              "re-run without --dpc-learned to use the vesselness field")
        return None

    fitted = LearnedProbability(best_roi).fit(
        points[best_inside], radius_um=radii[best_inside], seed=0
    )
    print(f"    trained the probability model on {best_n} centreline points "
          f"(training score {fitted.training_score:.3f})")
    if best_n < 200:
        print(f"    note: {best_n} points is a thin training set -- the default "
              "vesselness field may well do better here")
    return fitted.model


def cmd_surface(args) -> int:
    from .sdfpatch import SdfSession

    graph = _load(args.graph)
    # Cut, not mask. A hole left inside an edge would be closed straight back up by
    # `bridge_centerline_gaps`, which `sdfpatch.preprocess_graph` runs before meshing.
    _split_interpolation(args, graph)
    session = SdfSession(graph.snapshot(), voxel_size_mm=args.voxel_mm, quiet=False)
    print(f"  {session.n_capsules:,} capsules, "
          f"voxel {session.voxel_size_mm * 1000:.0f} um")
    surface = session.rebuild_full(args.out_dir)
    if surface is None:
        print("no surface produced")
        return 1
    print(f"\n{surface.n_points:,} vertices / {surface.n_cells:,} triangles "
          f"-> {args.out_dir}")
    return 0


# The lattice and the reference graph fall back to the environment, never to a literal
# path. These two used to name one machine's files, so any command that omitted `--seg`
# silently measured against somebody else's scan instead of failing.
ENV_SEG = "HIPCT_SEG"
ENV_GRAPH = "HIPCT_GRAPH"


def default_seg() -> str | None:
    """The configured lattice, or None when nothing is set."""
    return os.environ.get(ENV_SEG) or None


def default_graph() -> str | None:
    """The configured reference graph, or None when nothing is set."""
    return os.environ.get(ENV_GRAPH) or None


def require_seg(args) -> str:
    """The lattice path, or a clean exit saying how to supply one."""
    path = args.seg or default_seg()
    if not path:
        raise SystemExit(
            f"no segmentation lattice: pass --seg PATH, or set {ENV_SEG} in the "
            "environment."
        )
    return path


def require_graph(path) -> str:
    """A reference graph path, or a clean exit saying how to supply one."""
    if not path:
        raise SystemExit(
            f"no reference graph: pass one explicitly, or set {ENV_GRAPH} in the "
            "environment."
        )
    return path


def _seg_materials(args):
    """The segmentation's ``Materials`` block, or ``()`` when there is none to use.

    Re-reads the header rather than threading a ``LatticeInfo`` out through
    ``_decoded``'s four-tuple, which is unpacked at ten call sites: the parse reads a
    4 MB probe and costs milliseconds, against the minutes ``_decoded`` itself spends.

    A mask with no ``Materials`` block, or one naming only the Exterior, returns empty
    and every caller falls back to splitting by connectivity -- which is all a plain
    binary mask ever supported.
    """
    if getattr(args, "ignore_materials", False):
        return ()
    from .. import amira

    try:
        return amira.read_lattice_header(require_seg(args)).materials
    except (OSError, ValueError):
        return ()  # `_open_lattice` reports it properly a moment later


def _report_materials(materials) -> tuple:
    """Announce a material split, and return the foreground materials."""
    from . import components as comp

    regions = comp.regions_of(materials)
    if len(regions) > 1:
        named = ", ".join(f"{m.name} ({m.value})" for m in regions)
        print(f"  the mask names its trees: {named}"
              f"\n  splitting by material first, then by connectivity inside each "
              f"(--ignore-materials for the old behaviour)")
    return regions


def _open_lattice(args):
    """Open the label lattice and build the frame it shares with the raw grid.

    A ``WorldFrame`` normally needs the raw image stack to establish the binning.
    None of these commands read the greyscale. A graph's recorded voxel size is
    used when ``--voxel-um`` is omitted, so an already corrected graph still samples
    the same voxels. Only unstamped inputs fall back to the lattice's own units.

    **``--voxel-um`` changes that.** Given, it is taken as the truth and the lattice
    bounding box is treated as the same lattice measured in the wrong units: every
    length is rescaled onto the stated scale (see `WorldFrame.from_inputs`). That is
    the only way a command here can write out radii in real micrometres when the
    bounding box was written from a rounded voxel size. Omitted, nothing is corrected
    and the file's own units are used, exactly as before.
    """
    from .. import amira, rle
    from ..frame import WorldFrame
    from .components import graph_paths

    path = require_seg(args)
    info = amira.read_lattice_header(path)
    stated = getattr(args, "voxel_um", None)
    if stated is None:
        paths = graph_paths(getattr(args, "graph", None))
        stamps = [amira.read_voxel_stamp(p) for p in paths]
        recorded = [v for v in stamps if v is not None]
        if recorded:
            if len(recorded) != len(stamps) or not np.allclose(
                recorded, recorded[0], rtol=1e-6, atol=0.0
            ):
                raise ValueError(
                    "input graphs must share the same voxel scale; "
                    "stamped and unstamped graphs cannot be sampled together"
                )
            stated = recorded[0]
            print(f"  voxel size: using graph's recorded {stated:.4f} um")
    field_name = args.labels_field
    if field_name not in info.fields:
        if field_name == "Labels" and len(info.fields) == 1:
            field_name = next(iter(info.fields))
            print(f"  label field: using sole field '{field_name}' (requested 'Labels')")
        else:
            raise ValueError(
                f"{Path(path).name}: label field '{field_name}' not found; "
                f"available fields: {sorted(info.fields)}"
            )
    field = info.fields[field_name]
    labels = rle.open_lattice(path, field, info.dims)
    nominal = stated or float(info.spacing[0]) / 2.0
    # Raw shape is only used for reporting here; derive it from the lattice.
    raw_shape = tuple(int(v) for v in (info.dims[2], info.dims[1], info.dims[0]))
    frame = WorldFrame.from_inputs(raw_shape, nominal, info,
                                   voxel_is_truth=stated is not None)
    if frame.corrected:
        print(f"  voxel size: {frame.correction_note()}")

    # A painting session's corrections are composited here, once, so that every
    # command below reads the mask the user actually meant.
    if getattr(args, "edits", None):
        from .maskedit import MaskEdits, MaskSource

        edits = MaskEdits.load(args.edits, expect_dims=info.dims)
        print(f"  mask edits: {edits.describe()} ({Path(args.edits).name})")
        labels = MaskSource(labels, edits)
    return labels, frame, path


def cmd_skeletonise(args) -> int:
    from .skeletonise import skeletonise_lattice

    refusal = _check_per_tree(args)
    if refusal:
        print(refusal)
        return 2

    if args.per_tree:
        from . import skeletonisers as sk

        volume, frame, _labels, path = _decoded(args)
        print(f"{Path(path).name}: lattice {tuple(int(v) for v in frame.seg_dims)}, "
              f"spacing {frame.seg_spacing[0]:.2f} um\n")
        candidate = sk.skeletonise_per_component(
            "lee", volume, frame,
            min_component_voxels=args.min_component_voxels,
            max_trees=args.max_trees, verbose=True,
            materials=_seg_materials(args),
        )
        print(" ", candidate.describe())
        triple = candidate.triple
    else:
        labels, frame, path = _open_lattice(args)
        print(f"{Path(path).name}: lattice {tuple(int(v) for v in frame.seg_dims)}, "
              f"spacing {frame.seg_spacing[0]:.2f} um\n")
        triple = skeletonise_lattice(labels, frame,
                                     stride=max(int(args.stride), 1)).triple

    # Roots after the skeleton exists, and before the ordering that consumes them.
    # Lee thinning has no root concept, so there is nothing to pick *for* until the
    # centrelines are out; and a graph with more than one tree needs more than one
    # root, which is exactly what the picker walks through.
    graph, records, method, roots = None, [], "auto", []
    if args.pick_roots:
        from .graphmodel import EditableGraph

        graph = EditableGraph(triple)
        records, method = _roots_for_graph(graph, args, interactive=True)
        roots = [r["root"]["node_id"] for r in records]
    elif args.roots_json and Path(args.roots_json).exists():
        # `--roots-json` is a write destination under `--pick-roots` and a *source*
        # without it. A sidecar that already exists is an answer, and re-deriving a
        # skeleton is precisely the case its world coordinates were recorded to
        # survive -- so ordering the new graph from `auto_roots` while it sits there
        # unread is the one outcome nobody wants.
        from . import roots as roots_mod
        from .graphmodel import EditableGraph

        graph = EditableGraph(triple)
        roots = list(roots_mod.root_nodes_for(graph, args, frame=frame))
        if roots:
            print(f"  {len(roots)} root(s) from {Path(args.roots_json).name}")

    if args.order or roots:
        from .optimise import order

        print(" ordering:", order(triple, roots=roots or None).describe())

    if args.out:
        from .amira_write import write_spatial_graph
        from .adapter import to_spatial_graph

        write_spatial_graph(to_spatial_graph(triple), args.out)
        print(f"\nwrote {args.out}")
    else:
        print("\n(dry run: pass --out to write the generated graph)")

    # After the graph, so `source_fingerprint` hashes a file that exists: the sidecar
    # names the graph its roots belong to, and that graph is only on disk now.
    if records:
        _write_roots(graph, records, args, method=method, source=args.out)
    return 0


def cmd_optimise(args) -> int:
    from . import roots as roots_mod
    from .lattice import LatticeView, decode_volume
    from .optimise import compare, correct_radii, order

    graph = _load(args.graph)
    _warn_unflagged(graph)

    # The lattice first when there is one, because the roots sidecar snaps a recorded
    # root to the nearest node within a few *voxels*, and only the frame knows how big
    # a voxel is. Without it the snap is unbounded, which is how a root lands on the
    # wrong tree. Ordering still runs before `--oblique`, which is what needs it.
    lattice = frame = None
    if args.seg or args.oblique or args.sensitivity:
        labels, frame, path = _open_lattice(args)
        print(f"\ndecoding {Path(path).name} (stride {args.stride})...")
        volume = decode_volume(labels, stride=max(int(args.stride), 1))
        lattice = LatticeView.from_frame(volume, frame, stride=max(int(args.stride), 1))
        print(f"  {volume.shape} = {volume.nbytes / 1e9:.2f} GB")

    print("\nordering...")
    chosen = roots_mod.root_nodes_for(graph, args, frame=frame)
    if chosen:
        print(f"  from {len(chosen)} chosen root(s) ({Path(args.roots_json).name})")
    print(" ", order(graph.triple, roots=chosen or None).describe())

    print("\ncorrecting radii...")
    thickness, changed = correct_radii(
        graph.triple, lattice if args.oblique else None, verbose=True
    )
    print(f"  {len(changed)} point radii changed")

    reference = _load(require_graph(args.reference or default_graph()))
    print()
    print(compare(graph.triple, reference.triple, lattice,
                  bb_threshold=args.bb_threshold).describe())

    if args.out:
        # Apply the corrected radii before writing. `correct_radii` returns one flat
        # array in point order, so each segment needs its own points located in it --
        # once, through a dict. Scanning `point_order()` with `.index()` per point
        # made this quadratic, which on 29k points is minutes of pure lookup.
        position = {pid: i for i, pid in enumerate(graph.point_order())}
        with graph.batch("corrected radii"):
            for sid in graph.segment_ids():
                pids = graph.segment(sid)["point_ids"]
                idx = [position[p] for p in pids]
                graph.set_segment_radii(sid, thickness[idx])
        _save(graph, args.out, args.graph)
    else:
        print("\n(dry run: pass --out to write the ordered, radius-corrected graph)")
    return 0


def cmd_repair_radius(args) -> int:
    from . import radius_repair as rr

    graph = _load(args.graph)
    _warn_unflagged(graph)
    spans = []

    # Highs are always audited.  They are deliberately not changed under the legacy
    # only-increase behaviour unless the user explicitly authorises decreases.
    high_kw = {"factor": args.factor, "mode": "high"}
    if args.margin is not None:
        high_kw["margin"] = args.margin
    highs = rr.find_outlier_spans(graph, **high_kw)
    suffix = "will repair" if args.allow_decrease else "report only; use --allow-decrease"
    print(f"high-radius spans: {len(highs)} ({suffix})")
    if args.allow_decrease:
        spans += highs

    if args.source in ("outlier", "both"):
        kw = {"factor": args.factor}
        if args.margin is not None:
            kw["margin"] = args.margin
        found = rr.find_outlier_spans(graph, **kw)
        print(f"radius-outlier spans: {len(found)}")
        spans += found

    if args.source in ("image", "both"):
        if rr.has_perimeter_radii(graph):
            print("image collapse detection skipped: radius_source is present, so "
                  "perimeter measurement has already compensated collapsed shape")
        elif not (args.seg or default_seg()):
            print("image detection needs --seg")
        else:
            from .. import crosssection

            labels, frame, path = _open_lattice(args)
            print(f"\nmeasuring cross-sections against {Path(path).name} (slow)...")
            sg = graph.to_spatial_graph()
            cands, _profile = crosssection.find_sites(sg, frame, labels)
            kw = {}
            if args.margin is not None:
                kw["margin"] = args.margin
            found = rr.spans_from_candidates(graph, cands, **kw)
            print(f"image collapse spans: {len(found)} "
                  f"(of {len(cands)} candidates)")
            spans += found

    if not spans:
        print("\nno repairable spans found")
        return 0

    kw = {"only_increase": not args.allow_decrease}
    if args.max_taper_per_mm is not None:
        kw["max_taper_per_mm"] = args.max_taper_per_mm
    reports, patch = rr.fill_spans(graph, spans, **kw)
    print()
    print(rr.summarise(reports))
    _save(graph, args.out, args.graph)
    return 0


def _decoded(args):
    """``(volume, frame, labels, path)`` -- the lattice decoded once, at ``--stride``.

    Every command below needs the same decoded mask, and at stride 1 it is 2.34 GB, so
    decoding it per algorithm or per sweep sample is minutes of pure repeat.
    """
    from dataclasses import replace

    from .lattice import decode_foreground_crop, decode_volume

    labels, frame, path = _open_lattice(args)
    stride = max(int(args.stride), 1)
    sampled_shape = tuple(
        (n + stride - 1) // stride for n in (labels.nz, labels.ny, labels.nx)
    )
    # `math.prod`, not `np.prod`: on Windows numpy's default integer is int32, so
    # `np.prod((4748, 2964, 3400))` wraps 47,848,444,800 to 603,804,544 and the guard
    # below reads 0.6 GB for a 47.8 GB volume -- disabling the foreground crop for
    # exactly the volumes it exists to protect, which is how a `--stride 1` run on the
    # full LADAF-2021-17 lattice came to decode the whole thing.
    decoded_bytes = math.prod(int(n) for n in sampled_shape)
    origin_zyx = (0, 0, 0)

    # Dense topology/scoring operations create several arrays per voxel.  Above this
    # size, decoding the all-zero exterior first is both dangerous and pointless.
    # Cropping changes neither foreground nor world coordinates.
    if decoded_bytes > 8 * (1 << 30):
        print(
            f"{Path(path).name}: foreground scan before decoding "
            f"{decoded_bytes / 1e9:.2f} GB (stride {stride})..."
        )

        def report(phase, done, total, elapsed):
            print(
                f"    {phase} {done}/{total} slices ({elapsed:.1f}s)",
                end="\r",
            )

        volume, origin_zyx = decode_foreground_crop(
            labels, stride=stride, padding=1, progress=report
        )
        print(" " * 72, end="\r")
        print(
            f"  foreground crop origin {origin_zyx} on sampled z/y/x grid, "
            f"shape {volume.shape}"
        )
    else:
        print(f"{Path(path).name}: decoding (stride {stride})...")
        volume = decode_volume(labels, stride=stride)

    print(f"  {volume.shape} = {volume.nbytes / 1e9:.2f} GB, "
          f"{int((volume > 0).sum()):,} foreground voxels")
    if stride > 1 or origin_zyx != (0, 0, 0) or volume.shape != sampled_shape:
        # `frame` must describe the returned crop exactly.  Its origin shifts by the
        # crop offset in source voxels and its spacing incorporates decimation.
        spacing = np.asarray(frame.seg_spacing, dtype=np.float64) * stride
        origin = np.asarray(frame.seg_origin, dtype=np.float64) + (
            np.asarray(origin_zyx[::-1], dtype=np.float64)
            * stride
            * np.asarray(frame.seg_spacing, dtype=np.float64)
        )
        frame = replace(
            frame,
            seg_dims=np.asarray(volume.shape[::-1], dtype=np.int64),
            seg_origin=origin,
            seg_spacing=spacing,
        )
    return volume, frame, labels, path


def _scoring_context(args, volume, frame):
    """``(ImageTerms, reference bifurcations)`` -- the half that does not vary."""
    from . import supermetric as sm

    print("\nscoring context (computed once, reused for every candidate)...")
    image = sm.image_terms(volume, frame.seg_spacing,
                           tree_chi=not args.no_tree_chi)
    print(" ", image.describe())
    refs = sm.reference_bifurcations(volume, frame)
    print(f"  {len(refs)} reference bifurcation(s) from the mask's own skeleton")
    return image, refs


def cmd_skeletonise_all(args) -> int:
    from . import skeletonisers as sk
    from . import supermetric as sm
    from .adapter import to_spatial_graph
    from .amira_write import write_spatial_graph
    from .graphmodel import EditableGraph

    names = [n.strip() for n in args.algorithms.split(",") if n.strip()]
    unknown = [n for n in names if n not in sk.ALGORITHMS]
    if unknown:
        print(f"unknown algorithm(s) {unknown}; choose from {list(sk.ALGORITHMS)}")
        return 2

    refusal = _check_per_tree(args)
    if refusal:
        print(refusal)
        return 2

    volume, frame, _labels, path = _decoded(args)
    # Read once here rather than per algorithm: every candidate is split the same way,
    # and the announcement should appear once rather than three times.
    materials = _seg_materials(args)
    _report_materials(materials)
    if args.no_score:
        image = refs = None
        print("\nscoring disabled (--no-score)")
    else:
        image, refs = _scoring_context(args, volume, frame)

    params: dict[str, dict] = {n: {} for n in names}
    if "teasar" in params:
        if args.teasar_scale is not None:
            params["teasar"]["scale"] = args.teasar_scale
        if args.teasar_const_um is not None:
            params["teasar"]["const_um"] = args.teasar_const_um
    if "amira" in params:
        params["amira"]["graph"] = require_graph(args.amira_graph or default_graph())

    scope = _resolve_scope(args)
    scored: list[tuple] = []
    # Every candidate that reached disk, scored or not: `--no-score` skips `scored`
    # entirely, and `--pick-roots` still needs something to open a window on.
    written: list[tuple] = []
    produced = 0
    for name in names:
        print(f"\n-- {name}")
        try:
            if args.per_tree:
                cand = sk.skeletonise_per_component(
                    name, volume, frame,
                    min_component_voxels=args.min_component_voxels,
                    max_trees=args.max_trees, verbose=True,
                    materials=materials, **params[name],
                )
            else:
                cand = sk.skeletonise(name, volume, frame, **params[name])
        except (ImportError, ValueError) as exc:
            print(f"  skipped: {exc}")
            continue
        print(" ", cand.describe())
        if not args.no_score:
            graph = EditableGraph(cand.triple)
            metric = sm.super_metric(
                graph, frame, volume, image, refs,
                bb_threshold=args.bb_threshold,
            )
            print(metric.describe())
            rank = metric.total
            if scope != "whole":
                _scores, total = _print_scoped_scores(args, graph, frame, volume, scope)
                if total is not None:
                    rank = total
            scored.append((rank, name, cand, metric))

        # Resolved, so the log says where the file actually is rather than a relative
        # stub -- this runs from a GUI as often as from a shell.
        out = Path(args.out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"{name}.am"
        write_spatial_graph(to_spatial_graph(cand.triple), dest,
                            parameters_from=params.get("amira", {}).get("graph"))
        print(f"  wrote {dest}")
        written.append((name, cand))
        produced += 1

    if not produced:
        print("\nno candidate produced a skeleton")
        return 1

    if args.no_score:
        print(f"\ncandidates written to {Path(args.out_dir).resolve()}")
        winner = written[0]
    else:
        print("\n" + sm.SuperMetric.header())
        for _total, name, _cand, metric in sorted(scored, key=lambda r: r[0]):
            print(metric.row(name))
        best = min(scored, key=lambda r: r[0])
        label = {"whole": "M_S", "largest": "largest-component M_S",
                 "per-tree": f"{args.objective} per-tree M_S"}[scope]
        print(f"\nlowest {label}: {best[1]} ({best[0]:.3f})")
        print(f"candidates written to {Path(args.out_dir).resolve()}")
        winner = (best[1], best[2])

    # Last, and once. Every candidate is derived from the same mask, so their trees
    # sit in the same place; the sidecar keys on the world coordinate, not on ids, so
    # roots picked on the winner resolve onto any of the others too. Picking per
    # candidate would mean clicking the same two inlets three times over.
    if args.pick_roots:
        from .graphmodel import EditableGraph

        name, cand = winner
        print(f"\nrooting the {name} candidate...")
        graph = EditableGraph(cand.triple)
        records, method = _roots_for_graph(graph, args, interactive=True)
        if records:
            _write_roots(graph, records, args, method=method,
                         source=Path(args.out_dir).resolve() / f"{name}.am",
                         mask={"path": str(path), "stride": max(int(args.stride), 1)})
    return 0


#: ``optimise-skeleton`` flag (argparse dest) -> ``optimise_skeleton`` parameter, wherever
#: the two differ. The flags are named for the *stage* they belong to, which reads better
#: on a command line -- ``--prune-factor``, ``--recentre-damping`` -- while the function
#: signature groups by role. One table, because ``--sweep`` sets the same parameters by
#: their flag names and silently drifted out of step with `_optimise_kwargs`: sweeping the
#: documented ``prune-factor`` raised ``TypeError: unexpected keyword argument``.
OPTIMISE_ALIASES = {
    "prune_factor": "length_factor",
    "prune_radius_ratio": "radius_ratio",
    "recentre_tangent_radii": "tangent_radii",
    "recentre_damping": "damping",
    "recentre_max_move_frac": "max_move_frac",
    "recentre_grow_radii": "grow_radii",
    "recentre_blob": "blob",
}

#: Flags passed straight through under their own name.
_OPTIMISE_DIRECT = ("smooth_um", "drift_radius_factor")


def optimise_parameter(name: str) -> str:
    """The `skeleton_optimise.optimise_skeleton` parameter a CLI flag name sets."""
    return OPTIMISE_ALIASES.get(name, name)


def _optimise_kwargs(args) -> dict:
    kw: dict = {
        "deloop": not args.no_deloop,
        "prune": not args.no_prune,
        "smoother": args.smoother,
        "recentre_passes": 0 if args.no_recentre else None,
    }
    if args.recentre_passes is not None and not args.no_recentre:
        kw["recentre_passes"] = args.recentre_passes
    # Anything left unset keeps `skeleton_optimise`'s own default, so that module stays
    # the single place those numbers are stated and justified.
    for flag in (*OPTIMISE_ALIASES, *_OPTIMISE_DIRECT):
        value = getattr(args, flag, None)
        if value is not None:
            kw[optimise_parameter(flag)] = value
    return {k: v for k, v in kw.items() if v is not None}


def _parse_sweep(spec: str) -> dict:
    """``"prune-factor=1,2,3;smoother=gaussian,multiscale"`` -> a grid of values.

    Numbers become floats and anything else stays a string, so a categorical choice
    like the smoother can be swept alongside a numeric threshold. Without that,
    ``--smoother`` would be the one parameter the metric cannot be asked to choose --
    which is the whole reason the backends are selectable.
    """
    def value(text: str):
        try:
            return float(text)
        except ValueError:
            return text

    grid: dict[str, list] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"malformed sweep term {part!r}; expected name=v1,v2")
        name, values = part.split("=", 1)
        grid[name.strip().replace("-", "_")] = [
            value(v.strip()) for v in values.split(",") if v.strip()
        ]
    return grid


def _check_sweep_names(sample: dict) -> None:
    """Refuse a sweep term no stage will read, before spending an hour finding out.

    A sweep decodes the volume and computes the scoring context up front, so a misspelled
    term used to surface as a ``TypeError`` several minutes in, after the expensive part.
    """
    import inspect

    from . import skeleton_optimise as so

    known = set(inspect.signature(so.optimise_skeleton).parameters)
    unknown = sorted(k for k in sample if optimise_parameter(k) not in known)
    if unknown:
        options = sorted(set(OPTIMISE_ALIASES) | (known - {"graph", "frame", "labels"}))
        raise ValueError(
            f"--sweep names nothing that optimise-skeleton sets: {', '.join(unknown)}. "
            f"Choose from: {', '.join(options)}"
        )


def _sweep_samples(grid: dict, lhs: int) -> list[dict]:
    """The grid's product, or `lhs` Latin-hypercube samples across each term's range.

    Latin hypercube is the paper's own design (10 samples per algorithm, Table S14),
    and it is what makes a four-parameter sweep affordable: the full product of four
    three-valued terms is 81 runs, each of which decodes nothing but does re-measure
    every cross-section.

    A hypercube can only span a *continuum*, so a categorical term -- a smoother name
    -- is crossed with the samples instead of being interpolated between. Sampling
    "half way between savgol and multiscale" is not a thing.
    """
    import itertools

    names = sorted(grid)
    numeric = [n for n in names if all(isinstance(v, float) for v in grid[n])]
    categorical = [n for n in names if n not in numeric]

    if lhs and lhs > 0 and numeric:
        from scipy.stats import qmc

        lo = np.array([min(grid[n]) for n in numeric], dtype=np.float64)
        hi = np.array([max(grid[n]) for n in numeric], dtype=np.float64)
        unit = qmc.LatinHypercube(d=len(numeric), seed=0).random(int(lhs))
        rows = [dict(zip(numeric, lo + row * (hi - lo))) for row in unit]
        if not categorical:
            return rows
        return [
            {**row, **dict(zip(categorical, combo))}
            for combo in itertools.product(*(grid[n] for n in categorical))
            for row in rows
        ]

    return [dict(zip(names, combo)) for combo in itertools.product(*(grid[n] for n in names))]


def cmd_optimise_skeleton(args) -> int:
    from . import skeleton_optimise as so
    from . import supermetric as sm
    from .graphmodel import EditableGraph

    if args.sweep and args.pick_roots:
        # A sweep builds a throwaway graph per sample and writes none of them, so
        # every root clicked would be discarded with the trial it was clicked on.
        print("--pick-roots and --sweep do not combine: a sweep writes nothing, so "
              "there is no graph for the roots to belong to. Sweep first, then re-run "
              "with the winning flags and --pick-roots.")
        return 2

    source = _load(args.graph)
    _warn_unflagged(source)
    volume, frame, _labels, _path = _decoded(args)

    if args.sweep:
        image, refs = _scoring_context(args, volume, frame)
        samples = _sweep_samples(_parse_sweep(args.sweep), args.lhs)
        print(f"\nsweeping {len(samples)} parameter set(s)...")
        rows = []
        _check_sweep_names(samples[0] if samples else {})

        # Split once, outside the loop: the labelling depends on the mask, which no
        # sweep sample touches, and at stride 1 relabelling per sample is minutes of
        # pure repeat -- the same argument that decodes the volume once.
        scope = _resolve_scope(args)
        parts = stats = None
        if scope != "whole":
            from . import components as comp

            parts, stats = comp.split_components(
                volume, frame, min_voxels=args.min_component_voxels,
                max_trees=1 if scope == "largest" else args.max_trees, verbose=True,
                materials=_seg_materials(args),
            )

        for i, sample in enumerate(samples, 1):
            kw = _optimise_kwargs(args)
            kw.update({optimise_parameter(k): v for k, v in sample.items()})
            trial = EditableGraph(source.triple.copy())
            report = so.optimise_skeleton(trial, frame, volume, **kw)
            metric = sm.super_metric(trial, frame, volume, image, refs,
                                     bb_threshold=args.bb_threshold)
            label = " ".join(
                f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(sample.items())
            )
            rank = metric.total
            if parts:
                # Each trial is a different graph, so the tree field has to be
                # reassigned from the mask -- it does not survive `copy()` unless the
                # source carried one, and pruning changes which segments exist.
                scores = sm.super_metric_per_tree(
                    _tagged(trial, stats, parts, frame), frame, stats.labels, parts,
                    bb_threshold=args.bb_threshold, tree_chi=not args.no_tree_chi,
                )
                if scores:
                    rank = (next(iter(scores.values())).total if scope == "largest"
                            else sm.aggregate(scores, parts, objective=args.objective))
            rows.append((rank, label, metric, report.roughness))
            shown = scope if parts else "M_S"
            print(f"  [{i}/{len(samples)}] {shown} {rank:8.3f}  "
                  f"reversing {100 * report.roughness.reversing:5.2f}%  {label}")

        print("\n" + sm.SuperMetric.header())
        for _t, label, metric, _r in sorted(rows, key=lambda r: r[0]):
            print(metric.row(label[:14]))
        # Reported alongside `M_S`, not folded into it. A centreline doubling back on
        # itself is an artefact by inspection, but it moves cl-sensitivity so little --
        # 4.41% of points cost about 0.005 -- that ranking on `M_S` alone would call the
        # broken result a tie. The paper's metric is not wrong; it is answering a
        # different question, and this column exists so both are visible at once.
        print("\n  reversing (>120 deg turns), lower is better")
        for _t, label, _m, rough in sorted(rows, key=lambda r: r[3].reversing):
            print(f"    {100 * rough.reversing:6.2f}%  {rough.length_mm:8.1f} mm  {label}")
        best = min(rows, key=lambda r: r[0])
        label = {"whole": "M_S", "largest": "largest-component M_S",
                 "per-tree": f"{args.objective} per-tree M_S"}[scope]
        print(f"\nlowest {label} {best[0]:.3f} at {best[1]}")
        print("(a sweep writes nothing; re-run with those flags and --out)")
        return 0

    print("\noptimising...")
    report = so.optimise_skeleton(source, frame, volume, verbose=True,
                                  **_optimise_kwargs(args))
    print(report.describe())
    _per_tree_breakdown(source)

    # `--scope` means the same thing here as it does on `score` and in the sweep
    # above. Without it the breakdown just printed is structural only, and says
    # nothing about how well either tree scored.
    scope = _resolve_scope(args)
    if scope != "whole":
        image, refs = _scoring_context(args, volume, frame)
        metric = sm.super_metric(source, frame, volume, image, refs,
                                 bb_threshold=args.bb_threshold)
        print()
        print(metric.describe())
        _print_scoped_scores(args, source, frame, volume, scope)

    # Ordering last: pruning and de-looping change what the tree *is*, so orders
    # computed before them would describe a graph that no longer exists. The picker
    # sits in the same place for the same reason -- a root clicked on the input graph
    # can be on a spur that pruning then deletes.
    records, method = [], "auto"
    if args.pick_roots or args.roots_json:
        from . import roots as roots_mod
        from .optimise import order

        if args.pick_roots:
            records, method = _roots_for_graph(source, args, interactive=True)
            chosen = [r["root"]["node_id"] for r in records]
        else:
            chosen = list(roots_mod.root_nodes_for(source, args, frame=frame))
        # Unconditional once either flag is given: `order` fills every component the
        # operator skipped from `auto_roots`, and a graph left at order 0 throughout
        # is indistinguishable from one that was never ordered.
        print("\nordering from the chosen roots...")
        print(" ", order(source.triple, roots=chosen or None).describe())

    _save(source, args.out, args.graph)
    # After `_save`, so the sidecar names -- and hashes -- the *refined* graph the
    # roots were clicked on, not the input that pruning has since changed.
    if records:
        _write_roots(source, records, args, method=method, source=args.out)
    return 0


def _per_tree_breakdown(graph) -> None:
    """What each tree came out as, when the graph carries tree identity.

    ``optimise_skeleton`` is already component-local in every stage -- de-looping
    works per cycle, ``prune_spurs`` protects the last segment of a component, and
    re-centring and smoothing are per segment -- so there is nothing to split. What
    was missing was only the ability to *see* the two trees separately.
    """
    from . import components as comp

    indices = comp.tree_indices(graph)
    if len(indices) < 2:
        return
    print("\n  per tree:")
    for index in indices:
        sub = comp.subgraph_by_tree(graph, index)
        length = sum(
            float(np.linalg.norm(np.diff(sub.coords(sid), axis=0), axis=1).sum())
            for sid in sub.segment_ids()
        )
        free = sum(1 for nid in sub.nodes if sub.degree(nid) == 1)
        print(f"    tree {index}: {len(sub.segments)} segments, {free} free ends, "
              f"{length / 1000:,.1f} mm")


def cmd_radius_perimeter(args) -> int:
    from . import radius_perimeter as rp
    from . import roots as roots_mod

    graph = _load(args.graph)
    _warn_unflagged(graph)
    # Deliberately the full-resolution lattice, whatever `--stride` says: the radius is
    # what this command exists to fix, so measuring it on a decimated mask would hand
    # back the error being removed. Streaming z-slices keeps that affordable.
    labels, frame, path = _open_lattice(args)
    stamp = _correct_units(graph, frame, args)
    print(f"\nmeasuring every cross-section against {Path(path).name} "
          f"at full resolution (slow)...")

    root_edges = roots_mod.root_edges_for(graph, args, frame=frame)
    if root_edges:
        print(f"  rooting {len(root_edges)} component(s) at edge(s) {list(root_edges)}")

    kw = {}
    if args.gate_voxels is not None:
        kw["gate_voxels"] = args.gate_voxels


    def report(done, total):
        print(f"    {done}/{total} segments", end="\r")

    result = rp.measure_radii(
        graph, frame, labels, max_half=args.max_half,
        max_radius_factor=args.max_radius_factor,
        branch_aware=args.branch_aware,
        root_edges=root_edges,
        tangent_search_degrees=args.tangent_search_deg,
        **({} if args.transverse_axis_ratio is None
           else {"transverse_axis_ratio": args.transverse_axis_ratio}),
        carina_tip_factor=args.carina_tip_factor,
        bifurcation_tapers=args.bifurcation_tapers,
        junction_parent_profile=args.junction_parent_profile,
        **({} if args.junction_flare is None
           else {"junction_flare": args.junction_flare}),
        perimeter_correction=args.perimeter_correction,
        **({} if args.junction_mask_max_fraction is None
           else {"junction_mask_max_fraction": args.junction_mask_max_fraction}),
        **({} if args.junction_mask_radii is None
           else {"junction_mask_radii": args.junction_mask_radii or None}),
        **({} if args.grow_radii is None
           else {"grow_radii": args.grow_radii or None}),
        **({} if args.min_blob_voxels is None
           else {"min_blob_voxels": args.min_blob_voxels}),
        ownership_near_junctions=args.ownership_near_junctions,
        fallback_taper=args.fallback_taper,
        **({} if args.fallback_policy is None
           else {"fallback_policy": args.fallback_policy}),
        **({} if args.continuation_ratio is None
           else {"continuation_ratio": args.continuation_ratio}),
        workers=args.workers,
        stability_centroid_mode=args.stability_centroid_mode,
        progress=report,
        **kw,
    )
    print(" " * 40, end="\r")
    print(result.describe())
    rp.apply_radii(graph, result)
    _save(graph, args.out, args.graph, voxel_um=stamp)
    return 0


def cmd_refine_centreline(args) -> int:
    from .centreline_cli import run
    return run(args)


def cmd_crop(args) -> int:
    from pathlib import Path

    from . import components as comp_mod
    from . import crop as crop_mod
    from . import roots as roots_mod

    if args.ratio is not None and args.ratio_denominator is not None:
        print("give --ratio or --ratio-denominator, not both")
        return 2
    ratio = args.ratio
    if ratio is None and args.ratio_denominator:
        ratio = 1.0 / float(args.ratio_denominator)

    graph = _load(args.graph)
    _warn_unflagged(graph)

    document = resolved = None
    manual_block = None
    if args.crop_json and Path(args.crop_json).exists():
        document = crop_mod.load(args.crop_json)
        # Kept verbatim: the hand-marked segments are about to be deleted, so they
        # cannot be re-derived from the cropped graph, and an operator's marks are
        # exactly the part of the record that must not evaporate on a re-run.
        manual_block = document.get("manual")
        resolved = crop_mod.resolve(graph, document, strict_drop=args.replay)
        print(f"\nread {args.crop_json}: "
              f"{len(resolved.vessels)} main vessel(s) "
              f"[{', '.join(sorted(resolved.vessels)) or 'none'}]")
        for note in resolved.notes:
            print(f"  note: {note}")
    elif args.replay:
        raise SystemExit("--replay needs an existing --crop-json to replay")

    def _pick(flag, saved):
        # A flag given on this run wins; anything omitted falls back to the sidecar,
        # which is what makes `crop graph.am --crop-json old.json` reproduce a crop.
        return flag if flag is not None else saved

    saved = resolved.rule if resolved is not None else crop_mod.Rule()
    rule = crop_mod.Rule(
        min_strahler=_pick(args.min_strahler, saved.min_strahler),
        min_ostium_um=_pick(args.min_ostium_um, saved.min_ostium_um),
        ratio=_pick(ratio, saved.ratio),
        prune_unattributed=args.prune_unattributed or saved.prune_unattributed,
        takeoff_factor=args.takeoff_factor,
        # `--root-edge` first, then the roots sidecar, then whatever the crop sidecar
        # recorded last time -- most explicit wins, and each only fills what is still
        # unrooted.
        root_edges=roots_mod.root_edges_for(graph, args) or saved.root_edges,
    )

    try:
        if args.replay:
            plan = crop_mod.replay_plan(graph, document)
        else:
            plan = crop_mod.plan(
                graph, rule,
                vessels=resolved.vessels if resolved else {},
                drop_segments=resolved.drop_segments if resolved else (),
                prune_at=resolved.prune_at if resolved else (),
            )
    except crop_mod.CropError as exc:
        raise SystemExit(str(exc))

    for name, ostium in sorted(plan.ostia.items()):
        print(f"  {name:<6} {ostium['n_segments']:>3} segment(s), "
              f"ostial radius {ostium['radius_um']:,.0f} um")
    print(f"\n{plan.summary()}")
    for note in plan.notes:
        print(f"  note: {note}")
    for sid in sorted(plan.takeoffs, key=lambda s: -plan.takeoffs[s]["n_subtree_removed"])[:20]:
        record = plan.takeoffs[sid]
        radius = record.get("radius_um")
        threshold = record.get("threshold_um")
        measured = (f"{radius:,.0f} um under {threshold:,.0f} um"
                    if radius is not None and threshold is not None else record["rule"])
        print(f"  segment {sid:>5}  {measured}  "
              f"takes {record['n_subtree_removed']} segment(s)"
              + (f"  [{record['vessel']}]" if record.get("vessel") else ""))
    if len(plan.takeoffs) > 20:
        print(f"  ... and {len(plan.takeoffs) - 20} more (see --report-csv)")

    if not plan.drop and rule.is_empty and not args.replay:
        print("\nno rule given and nothing marked by hand: pass --min-strahler, "
              "--min-ostium-um or --ratio, or mark branches in the Crop tab")

    # Both are taken *before* the deletion: afterwards there is nothing left to name
    # the dropped segments with, and the counts would describe the result rather than
    # the input the sidecar claims to have been written against.
    keys = crop_mod.segment_keys(graph)
    # The graph the crop was planned against: the first input names it, and its
    # counts come from `graph`, which is the merged whole when several were given.
    source_path = (comp_mod.graph_paths(args.graph) or [None])[0]
    fingerprint = crop_mod.source_fingerprint(source_path, graph)
    if args.report_csv:
        print(f"wrote {crop_mod.write_csv(args.report_csv, graph, plan, keys=keys)}")

    removed = crop_mod.apply(graph, plan)
    print(f"\nremoved {removed} segment(s); {len(graph.segments)} remain in "
          f"{len(graph.components())} component(s)")

    reordered = False
    if args.reorder and removed:
        try:
            # Re-resolved against the *cropped* graph: the sidecar keys on world
            # coordinates, so the root survives the segments it was recorded on being
            # deleted -- which is exactly what a crop does.
            print(f"re-ordered: "
                  f"{crop_mod.reorder(graph, roots_mod.root_nodes_for(graph, args))}")
            reordered = True
        except Exception as exc:  # noqa: BLE001 - the message is the point, not the type
            raise SystemExit(
                f"could not recompute the Strahler order: {exc}\n"
                "The stored orders describe the tree as it was, so this run has not "
                "written anything. Re-run with --no-reorder to keep them anyway."
            )

    if args.crop_json:
        document = crop_mod.document(graph, plan, rule, source=source_path,
                                     out=args.out, reordered=reordered, keys=keys,
                                     fingerprint=fingerprint,
                                     colors=resolved.colors if resolved else None)
        if manual_block:
            document["manual"] = manual_block
        print(f"wrote {crop_mod.write(args.crop_json, document)}")

    _save(graph, args.out, args.graph)
    return 0


def _roots_for_graph(graph, args, *, interactive: bool, parts=None):
    """This run's roots: the 3-D picker when asked for it, ``auto_roots`` otherwise.

    One code path for `pick-roots`, `skeletonise`, `skeletonise-all` and
    `optimise-skeleton`, because they must all agree on what a *skipped* tree means.
    ``pick_roots`` opens **one window per connected component, largest first**, and
    returns only the trees actually clicked -- so on a graph carrying debris the
    operator roots the left and right coronaries and presses ``x``, and
    :func:`~.optimise.order` fills every component they skipped from ``auto_roots``
    rather than leaving its edges at order 0.

    Returns ``(records, method)``: :func:`~.roots.describe_roots` records -- one per
    *component*, deduplicated, so two clicks in one tree cannot produce two roots --
    and "manual" or "auto" for the sidecar's provenance. An empty list means the
    picker opened and nothing was chosen, which is a decision, not a failure: the
    caller records nothing and ordering falls back to the automatic root.
    """
    from . import roots as roots_mod
    from .optimise import order

    n_components = len(graph.components())
    print(f"\n{n_components} tree(s) to root")
    if interactive and n_components > 2:
        # The left and right coronaries are components 1 and 2 by size on this data;
        # everything after them is debris that nothing downstream reads a root from.
        print("  one window per tree, largest first: root the trees you care about "
              "and press `x` to stop.\n  Whatever you skip keeps its automatic root.")

    method, nodes = "auto", []
    if interactive:
        if getattr(args, "color_by", "strahler") == "strahler" and not any(
            "strahler" in seg for seg in graph.segments
        ):
            print("  ordering first, so the picker can colour by Strahler...")
            print("   ", order(graph.triple).describe())
        try:
            nodes = roots_mod.pick(
                graph,
                color_by=getattr(args, "color_by", "strahler"),
                style=getattr(args, "style", "contour"),
                n_sides=getattr(args, "n_sides", 16),
                ring_stride=getattr(args, "ring_stride", 1),
                off_screen=bool(getattr(args, "screenshot", None)),
                screenshot=getattr(args, "screenshot", None),
                preselect=getattr(args, "preselect", None),
            )
            method = "manual"
            if getattr(args, "screenshot", None):
                print(f"  rendered the first tree to {args.screenshot}")
        except ImportError as exc:
            print(f"\n  the interactive picker needs PyVista and matplotlib ({exc});"
                  f"\n  falling back to the automatic roots. To install them:"
                  f"\n      pip install -e .[viz,viz3d]")
            interactive = False

    if not interactive and not nodes:
        nodes = roots_mod.auto(graph)
    if not nodes:
        return [], method
    if method == "manual" and len(nodes) < n_components:
        # `pick_roots` appends only for trees that were actually picked, so a skipped
        # tree is a real outcome rather than an error -- but it must be visible, since
        # ordering will fall back to the automatic root for it.
        print(f"  {len(nodes)} of {n_components} tree(s) picked; the rest keep their "
              f"automatic root")

    records = roots_mod.describe_roots(graph, nodes, method=method, parts=parts)
    for record in records:
        root = record["root"]
        print(f"  tree {record['index']}: node {root['node_id']} on segment "
              f"{root['seg_id']} at {tuple(round(v) for v in root['node_um'])}"
              + ("  (junction -- ambiguous)" if root["ambiguous"] else ""))
    return records, method


def _write_roots(graph, records, args, *, method: str, source=None, mask=None,
                 sources=None) -> bool:
    """Record the chosen roots to ``--roots-json``, when one was given.

    The sidecar is the deliverable, not the ordered graph: it records each root's
    world coordinate first, so it survives the re-skeletonisation that a `.am` full of
    ids does not, and it is what every downstream command reads.
    """
    from . import roots as roots_mod

    path = getattr(args, "roots_json", None)
    if not path:
        print("\n(pass --roots-json PATH to record the roots for the commands "
              "downstream)")
        return False
    document = roots_mod.document(
        graph, records, source=source, mask=dict(mask or {}), sources=sources,
        picker={"method": method, "color_by": getattr(args, "color_by", "strahler"),
                "style": getattr(args, "style", "contour")},
    )
    roots_mod.write(path, document)
    print(f"\nwrote {path}")
    return True


def _renumber_trees(records, taken: dict, *, source=None) -> list:
    """Give `records` tree indices that do not collide with those already `taken`.

    Only relevant when one session roots **several skeletons**. A left-tree-only graph
    and a right-tree-only graph both call their sole component tree 0 when there is no
    mask to anchor them, and two records numbered 0 in one sidecar would read as one
    tree picked twice. With ``--seg`` this almost never fires: `assign_trees` has
    already given each graph the index of the material it overlaps, so the left graph
    arrives as tree 0 and the right as tree 1 by themselves.

    Two records are the *same tree* when the mask named them the same material, and
    otherwise when they came from the same graph. So a second pick within one graph
    keeps its number, and two anonymous graphs that both said "tree 0" do not.
    """
    out = []
    for record in records:
        index = int(record.get("index", -1))
        who = record.get("material") or str(source)
        if index in taken and taken[index] != who:
            index = max(taken) + 1
            record = dict(record, index=index, renumbered_from=int(record["index"]))
        taken[index] = who
        out.append(record)
    return out


def cmd_pick_roots(args) -> int:
    from . import components as comp
    from . import roots as roots_mod
    from .optimise import order

    paths = list(args.graph)
    graphs = [(p, _load(p)) for p in paths]

    # Anchoring the tree indices to the mask is optional, because the graph alone is
    # enough to pick a root; without it the indices fall back to component size order
    # and the sidecar records that, so a later reader knows which it is looking at.
    # The mask is decoded and split **once** and shared by every graph, which is the
    # whole reason several skeletons can be rooted into one coherent sidecar: they are
    # all numbered against the same labelling.
    mask_info: dict = {}
    parts = None
    if args.seg:
        # Said before the decode, not after it. The picker does not read the mask --
        # the window comes from the graph and the root is recorded by coordinate --
        # so this whole branch buys tree *numbering* and nothing else, and it is the
        # only expensive thing the command does.
        print("  --seg given: decoding it to anchor the tree numbering to the mask "
              "labelling.\n  The roots themselves do not need it; drop --seg to skip "
              "this entirely.")
        volume, frame, _labels, path = _decoded(args)
        parts, stats = comp.split_components(
            volume, frame, min_voxels=args.min_component_voxels, verbose=True,
            materials=_seg_materials(args),
        )
        for graph_path, graph in graphs:
            if comp.tree_indices(graph) == [] and parts:
                trees = comp.assign_trees(graph, stats.labels, frame,
                                          order=[p.label for p in parts], verbose=True)
                from .optimise import set_edge_field

                set_edge_field(graph.triple, comp.TREE_FIELD, trees, np.int64)
                named = {int(t) for t in trees.tolist() if t >= 0}
                where = ", ".join(sorted({p.material for p in parts
                                          if p.index in named and p.material}))
                print(f"  {Path(graph_path).name}: assigned {len(named)} tree(s) "
                      f"from the mask" + (f" ({where})" if where else ""))
    else:
        frame = None

    all_trees: list = []
    sources: list = []
    taken: dict = {}
    method = "auto"
    for graph_path, graph in graphs:
        if len(graphs) > 1:
            print(f"\n== {Path(graph_path).name}")
        trees, how = _roots_for_graph(graph, args, interactive=not args.auto,
                                      parts=parts)
        sources.append(roots_mod.describe_source(graph_path, graph))
        if not trees:
            continue
        method = "manual" if how == "manual" else method
        all_trees.extend(_renumber_trees(trees, taken, source=graph_path))

        if args.out and len(graphs) == 1:
            print("\nordering from the chosen roots...")
            print(" ", order(graph.triple, roots=[r["root"]["node_id"] for r in trees]
                             ).describe())
            _save(graph, args.out, graph_path)

    if not all_trees:
        # The picker opened and nothing was clicked. Recording the automatic roots
        # here would be the one outcome the operator demonstrably did not want.
        print("\nno root chosen; nothing written")
        return 1

    if args.seg:
        mask_info = {
            "path": str(path), "stride": max(int(args.stride), 1), "connectivity": 3,
            "min_component_voxels": int(args.min_component_voxels),
            "n_components": int(stats.n), "n_kept": len(parts),
            "materials": [{"name": m.name, "value": int(m.value)}
                          for m in comp.regions_of(_seg_materials(args))],
        }
    if args.out and len(graphs) > 1:
        # One `--out` cannot receive several graphs, and picking which one it meant
        # would be a guess. The sidecar is the deliverable here anyway.
        print(f"\n! --out is ignored when several graphs are rooted at once; "
              f"re-run `optimise-skeleton --roots-json {args.roots_json or 'ROOTS'}` "
              f"per graph to write ordered copies")

    _write_roots(graphs[0][1], all_trees, args, method=method, source=paths[0],
                 mask=mask_info, sources=sources)
    return 0


def cmd_score(args) -> int:
    from . import supermetric as sm

    graph = _load(args.graph)
    _warn_unflagged(graph)
    volume, frame, _labels, _path = _decoded(args)
    image, refs = _scoring_context(args, volume, frame)
    metric = sm.super_metric(graph, frame, volume, image, refs,
                             bb_threshold=args.bb_threshold)
    print()
    print(metric.describe())

    scope = _resolve_scope(args)
    if scope != "whole":
        _print_scoped_scores(args, graph, frame, volume, scope)
    return 0


def _tagged(graph, stats, parts, frame):
    """`graph`, guaranteed to carry a ``tree`` field on **every** segment.

    A sweep trial starts as a copy, so it inherits whatever tags the source had -- but
    de-looping and spur pruning delete segments and contract degree-2 nodes, and a
    segment produced by a contraction carries no tag. An untagged segment is invisible
    to :func:`~.components.subgraph_by_tree`, so it would silently drop out of the
    per-tree score rather than failing.
    """
    from . import components as comp

    tagged = comp.tree_of_edge(graph)
    if len(tagged) == len(graph.segments) and tagged:
        return graph

    trees = comp.assign_trees(graph, stats.labels, frame,
                              order=[p.label for p in parts])
    from .optimise import set_edge_field

    set_edge_field(graph.triple, comp.TREE_FIELD, trees, np.int64)
    return graph


def _scoped_scores(args, graph, frame, volume, scope: str, *, verbose: bool = True):
    """``(per component score, parts)`` for ``--scope per-tree`` or ``largest``.

    Always printed **after** the whole-graph table, never instead of it: both
    restricted scopes redefine ``cc`` (each component is one component, so the term
    becomes ``|1 - cc_s| / 1``) and ``chi``, so a run reporting a different number has
    to show the comparable one beside it or the two get quoted against each other
    later as if they measured the same thing.

    ``largest`` is ``per-tree`` capped at one component. The difference from the
    default whole-graph scoring is that the restriction reaches **all five terms**
    rather than only chi -- so the second tree is excluded outright instead of
    contributing to V, cc, cl and B while being invisible to chi.
    """
    from . import components as comp
    from . import supermetric as sm

    max_trees = 1 if scope == "largest" else args.max_trees
    parts, stats = comp.split_components(
        volume, frame, min_voxels=args.min_component_voxels,
        max_trees=max_trees, verbose=verbose, materials=_seg_materials(args),
    )
    if not parts:
        if verbose:
            print(f"\nno mask component to score with --scope {scope}")
        return None, ()

    if verbose and len(comp.tree_of_edge(graph)) != len(graph.segments):
        print("  tagging the graph's segments from the mask labelling...")
    _tagged(graph, stats, parts, frame)

    if verbose:
        if scope == "largest":
            print("\nlargest component only (all five terms, not just chi --"
                  "\n  every other tree is excluded outright)")
        else:
            print("\nper tree (each tree against its own mask component --"
                  "\n  not comparable with the M_S above: cc is |1 - cc_s| / 1 here)")

    scores = sm.super_metric_per_tree(
        graph, frame, stats.labels, parts,
        bb_threshold=args.bb_threshold, tree_chi=not args.no_tree_chi, verbose=verbose,
    )
    if not scores:
        if verbose:
            print("  nothing could be scored at this scope")
        return None, parts
    return scores, parts


def _print_scoped_scores(args, graph, frame, volume, scope: str):
    """Print the restricted-scope table and return ``(scores, ranking number)``."""
    from . import supermetric as sm

    scores, parts = _scoped_scores(args, graph, frame, volume, scope)
    if not scores:
        return None, None

    print("\n" + sm.SuperMetric.header())
    for index in sorted(scores):
        label = "largest" if scope == "largest" else f"tree {index}"
        print(scores[index].row(label))

    if scope == "largest":
        total = next(iter(scores.values())).total
        print(f"\n  largest-component M_S: {total:.3f}")
    else:
        objective = getattr(args, "objective", "weighted")
        total = sm.aggregate(scores, parts, objective=objective)
        print(f"\n  {objective} M_S over {len(scores)} tree(s): {total:.3f}")
    return scores, total


def cmd_repair_mask(args) -> int:
    from .lattice import decode_volume
    from .reconnect import segmentation as seg

    labels, frame, path = _open_lattice(args)
    stride = max(int(args.stride), 1)
    print(f"{Path(path).name}: decoding (stride {stride})...")
    mask = decode_volume(labels, stride=stride) > 0
    voxel = float(frame.seg_spacing[0]) * stride
    before = seg.report(mask, voxel_size_um=voxel)
    print(f"  {mask.shape}, {int(mask.sum()):,} foreground voxels")

    if args.min_voxels > 0:
        mask, removed = seg.cull_small(mask, args.min_voxels,
                                       keep_largest=args.keep_largest)
        print(f"culled {removed} component(s) smaller than {args.min_voxels} voxels")
    if args.close > 0:
        mask = seg.close_gaps(mask, args.close)
        print(f"closed gaps with a radius-{args.close} ball")

    after = seg.report(mask, voxel_size_um=voxel)
    print("\nmorphometrics:")
    print(seg.compare(before, after))
    print("\n  read these as a pair: a closing that mends a real break lowers the")
    print("  component count and leaves Euler alone; one that welds two unrelated")
    print("  vessels lowers both.")

    if args.out:
        import tifffile

        tifffile.imwrite(args.out, mask.astype(np.uint8) * 255)
        print(f"\nwrote {args.out}")
    else:
        print("\n(dry run: pass --out to write the repaired mask)")
    return 0


def cmd_mask_export(args) -> int:
    """Write the mask -- source plus any painted corrections -- back out.

    Label values are preserved rather than scaled to 0/255, so an ``.am`` written
    here is a drop-in replacement for the source lattice. (``repair-mask --out``
    scales, because what it writes is a thresholded boolean.)
    """
    from .. import amira, rle_write
    from .lattice import decode_volume

    labels, frame, path = _open_lattice(args)
    stride = max(int(args.stride), 1)
    out = Path(args.out)
    print(f"{Path(path).name}: decoding (stride {stride})...")
    mask = decode_volume(labels, stride=stride)
    values, counts = np.unique(mask, return_counts=True)
    print(f"  {mask.shape}, values "
          + ", ".join(f"{int(v)}:{int(c):,}" for v, c in zip(values, counts)))

    if out.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(out, mask)
        print(f"\nwrote {out}")
        return 0

    info = amira.read_lattice_header(path)
    nz, ny, nx = mask.shape
    # The bbox spans voxel *centres*, and a strided export has a coarser spacing,
    # so the far face moves in. Getting this wrong shifts the whole volume.
    spacing = np.asarray(info.spacing, dtype=np.float64) * stride
    origin = np.asarray(info.origin, dtype=np.float64)
    bbox = np.empty(6, dtype=np.float64)
    bbox[0::2] = origin
    bbox[1::2] = origin + (np.array([nx, ny, nz]) - 1) * spacing

    def report_progress(done, total, elapsed):
        print(f"    encoding {done}/{total} planes ({elapsed:.1f}s)", end="\r")

    result = rle_write.write_lattice(out, mask, bbox, field=args.field,
                                     progress=report_progress)
    print(" " * 40, end="\r")
    print("\n" + rle_write.describe(result))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "report": cmd_report,
        "gaps": cmd_gaps,
        "connect": cmd_connect,
        "flag-interpolation": cmd_flag_interpolation,
        "train-cfc": cmd_train_cfc,
        "evaluate-dpc": cmd_evaluate_dpc,
        "export-dpc-regions": cmd_export_dpc_regions,
        "surface": cmd_surface,
        "skeletonise": cmd_skeletonise,
        "skeletonise-all": cmd_skeletonise_all,
        "optimise": cmd_optimise,
        "optimise-skeleton": cmd_optimise_skeleton,
        "radius-perimeter": cmd_radius_perimeter,
        "refine-centreline": cmd_refine_centreline,
        "prepare-reconstruction": cmd_refine_centreline,
        "segment-diagnosis": cmd_segment_diagnosis,
        "junction-mask": cmd_junction_mask,
        "reformat-radius": cmd_reformat_radius,
        "ostium-flare": cmd_ostium_flare,
        "crop": cmd_crop,
        "pick-roots": cmd_pick_roots,
        "score": cmd_score,
        "repair-radius": cmd_repair_radius,
        "repair-mask": cmd_repair_mask,
        "mask-export": cmd_mask_export,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
