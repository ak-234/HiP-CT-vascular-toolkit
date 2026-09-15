"""Command-line entry point.

    python -m hipct_seg_debug --validate-only
    python -m hipct_seg_debug                      # 3D pick -> slice browser -> repeat
    python -m hipct_seg_debug --edit --paint       # ...and correct what you find
    python -m hipct_seg_debug --goto-slice 3361 --goto-row 2638 --goto-col 969

See CLI.md for every command and flag.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from . import amira, candidates, crosssection, frame, rle, stl_slice, tiffstack, viewer2d

#: The four inputs, and the environment variable each falls back to.
#:
#: Deliberately *not* literal paths. This mapping used to hold one machine's absolute
#: locations, which meant a fresh install with no arguments went looking for somebody
#: else's scan -- and found it, on the machine it was authored on. A checkout cannot
#: know where anyone's data lives, so the only defaults are the ones a user sets.
ENV_VARS = {
    "raw": "HIPCT_RAW",
    "graph": "HIPCT_GRAPH",
    "seg": "HIPCT_SEG",
    "surface": "HIPCT_SURFACE",
}

#: The three needed for a session to load at all. `--surface` missing is a warning.
REQUIRED_INPUTS = ("raw", "graph", "seg")


def env_default(key: str) -> str | None:
    """The configured path for one input, or None. An empty variable counts as unset."""
    return os.environ.get(ENV_VARS[key]) or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hipct_seg_debug",
        description="Overlay a HiP-CT stack with its segmentation, skeleton and reconstructed surface "
        "to find collapsed-vessel failures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw", default=env_default("raw"), help="directory of raw image slices")
    # Several: a left-tree skeleton and a right-tree one are loaded as **one** graph
    # holding two trees, not as one graph plus read-only scenery. See
    # `components.merge_graphs` for why that is the editable arrangement.
    p.add_argument("--graph", nargs="+", default=env_default("graph"), metavar="PATH",
                   help="ASCII Amira spatial graph (.am); pass several and they are "
                        "held as separate trees in one editable graph")
    p.add_argument("--seg", default=env_default("seg"), help="binary Amira label lattice (.am)")
    p.add_argument("--surface", default=env_default("surface"), help="reconstructed lumen surface (.stl)")
    p.add_argument("--pattern", default="*", help="glob for raw slices (supported image files only)")
    p.add_argument("--cache", default=None,
                   help="cache directory (default: $HIPCT_CACHE, else ./cache)")
    p.add_argument("--cfc-model", default=None,
                   help="persisted DF21 artifact used by CFC workflow variants")
    p.add_argument("--cfc-python", default=None,
                   help="Python 3.9 executable for CFC workflow jobs; alternatively HIPCT_CFC_PYTHON")

    p.add_argument("--voxel-um", type=float, default=None,
                   help="raw voxel size in um. REQUIRED: it is never inferred, because "
                        "an inferred one is wrong silently. The load error names what "
                        "the file names and the segmentation bounding box suggest")
    p.add_argument("--stl-scale", type=float, default=1000.0, help="multiplier taking surface units to um")
    p.add_argument("--labels-field", default="Labels", help="lattice field holding the binary mask")
    p.add_argument("--probability-field", default="Probability", help="optional second lattice field")

    p.add_argument("--slab", type=int, default=5, help="slices shown above and below the pick")
    p.add_argument("--roi", type=int, default=400,
                   help="width/height of the ROI in raw pixels; 0 for the whole slice")
    p.add_argument("--volume", action="store_true",
                   help="add lazy whole-dataset layers so the z slider can leave the slab")
    p.add_argument("--seg-box-um", type=float, default=2000.0,
                   help="half-extent of the 3D segmentation isosurface box around a pick ('g')")
    p.add_argument("--seg-stride", type=int, default=1,
                   help="decimation of the whole-tree 3D mask ('a'); 1 is full "
                        "resolution (3.75M triangles, ~6s), 4 is ~0.5s and 213k. "
                        "The layer panel changes this without a restart")
    p.add_argument("--plane-opacity", type=float, default=0.55,
                   help="starting opacity of the raw image plane in the 3D view ('i')")
    p.add_argument("--surface-opacity", type=float, default=0.35,
                   help="starting opacity of the reconstructed surface in the 3D view")

    p.add_argument("--validate-only", action="store_true", help="run coordinate checks and exit")
    p.add_argument("--selftest", action="store_true", help="run the end-to-end consistency tests and exit")
    # Detection is on by default, so only the negative flag exists. (There used to be
    # a `--candidates` beside this one; nothing ever read it.)
    p.add_argument("--no-candidates", action="store_true", help="skip candidate detection entirely")
    p.add_argument("--no-crosssection", action="store_true",
                   help="skip the image-based collapse detector (graph heuristics only)")
    p.add_argument("--min-hops", type=int, default=4,
                   help="minimum tree separation for a reported pair; below this is a bifurcation")
    p.add_argument("--murray-percentile", type=float, default=0.10,
                   help="flag bifurcations in this lowest fraction of the tree's own "
                        "sum(r_child^3)/r_parent^3 distribution")
    p.add_argument("--severity-iso", type=float, default=1.6,
                   help="isoperimetric ratio above which a lumen is reported as strongly "
                        "collapsed (informational: re-inflation is by design)")
    p.add_argument("--mismatch-factor", type=float, default=1.5,
                   help="report where the assigned radius differs from the perimeter "
                        "measured at that cross-section by more than this factor")
    p.add_argument("--probability", action="store_true", help="load the probability field as an overlay")
    p.add_argument("--no-surface", action="store_true", help="skip the surface entirely")
    p.add_argument("--no-tube", action="store_true", help="skip the rasterised skeleton-radius layer")
    p.add_argument("--no-strict", action="store_true", help="warn instead of aborting on failed checks")

    # Editing. Off by default: without it this stays the read-only auditing tool
    # it has always been, and coronary_sdf is never imported.
    p.add_argument("--edit", action="store_true",
                   help="enable skeleton editing with live SDF surface regeneration")
    p.add_argument("--edit-box-um", type=float, default=4000.0,
                   help="side of the box rebuilt around an edit (um)")
    p.add_argument("--edit-voxel-mm", type=float, default=None,
                   help="pin the SDF voxel size (mm); default derives it from the graph")

    # Voxel painting. Independent of --edit: correcting the mask and exporting it
    # needs no coronary_sdf, only re-skeletonising into the graph does.
    p.add_argument("--paint", action="store_true",
                   help="add a writable segmentation layer to the slice browser")
    p.add_argument("--paint-box", type=int, default=192,
                   help="side of the paintable block, in segmentation voxels")
    p.add_argument("--edits", default=None,
                   help="mask edit store (.npz); loaded at start, saved on exit")

    p.add_argument("--goto-slice", type=int, default=None, help="skip the 3D step: raw slice index")
    p.add_argument("--goto-row", type=int, default=None, help="skip the 3D step: raw row")
    p.add_argument("--goto-col", type=int, default=None, help="skip the 3D step: raw column")
    p.add_argument("--goto-um", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="skip the 3D step: world um")
    p.add_argument("--goto-candidate", type=int, default=None, help="skip the 3D step: candidate id")
    return p


class InputError(Exception):
    """An input is missing, wrong, or not what it claims to be.

    Raised instead of `SystemExit` so the GUI can catch it and show the message
    rather than having the interpreter shut down underneath a live window. `main`
    converts it straight back, so the command line behaves exactly as before.
    """


class MissingInputs(InputError):
    """Nothing was named at all -- as opposed to something named that will not load.

    The distinction is the whole point: a bad path is a mistake to report and exit on,
    while *no* path is an ordinary way to start the viewer. The window has a Data tab
    that loads a dataset, so `main` opens it empty rather than refusing to start. Every
    other entry point still treats this as fatal, because none of them has anywhere to
    ask.
    """

    def __init__(self, message: str, missing=()):
        super().__init__(message)
        #: The `REQUIRED_INPUTS` keys that were not given, so a caller can name the
        #: flags and variables itself rather than parsing them back out of the text.
        self.missing = tuple(missing)


def _default_cache() -> Path:
    """Where decoded lattices and candidate CSVs land when `--cache` is not given.

    Deliberately *not* relative to this file: once the package is pip-installed the
    package directory is inside site-packages, which is the wrong place — and often
    an unwritable one — for tens of megabytes of decoded slices. `HIPCT_CACHE` wins
    so one cache can be shared between checkouts; otherwise it is `cache/` under the
    working directory, which is what running from a checkout has always produced.
    """
    env = os.environ.get("HIPCT_CACHE")
    return Path(env) if env else Path.cwd() / "cache"


#: Re-exported: it lives beside `merge_graphs`, which is the thing that needs it, and
#: `edit.__main__` must not import this module to get at it -- that would drag napari
#: into every command-line run.
from .edit.components import graph_paths  # noqa: E402,F401


def load_graphs(paths):
    """``(graph, trees)`` -- one skeleton, or several merged into one editable graph.

    A single path is read and returned untouched, so an unedited graph still round
    trips through the writer exactly as it did. Several are merged by
    :func:`~.edit.components.merge_graphs`, which tags each source's edges with their
    own ``tree`` index: one graph, so every tool that is defined against
    ``session.graph`` -- picking, the edit handles, `crop`, `reformat`, the writer --
    reaches all of them, and the trees stay distinguishable for everything that asks
    per tree.
    """
    from . import amira

    if not paths:
        raise InputError("no skeleton to load")
    if len(paths) == 1:
        return amira.read_spatial_graph(paths[0]), [[0]]

    from .edit.adapter import from_spatial_graph, to_spatial_graph
    from .edit.components import merge_graphs

    triples = []
    for path in paths:
        one = amira.read_spatial_graph(path)
        print(f"  skeleton      {one.n_vertex} vertices, {one.n_edge} edges "
              f"({Path(path).name})")
        triples.append(from_spatial_graph(one))
    merged, trees = merge_graphs(triples)
    graph = to_spatial_graph(merged, paths[0])
    named = ", ".join(f"{Path(p).name} -> tree {'/'.join(str(t) for t in ts)}"
                      for p, ts in zip(paths, trees))
    print(f"  merged        {len(paths)} skeletons as {max(max(t) for t in trees) + 1} "
          f"tree(s): {named}")
    return graph, trees


class Session:
    """Everything loaded and cross-checked, ready to serve slabs."""

    def __init__(self, args):
        self.args = args
        self._builder = None
        cache = Path(_default_cache() if args.cache is None else args.cache)
        cache.mkdir(parents=True, exist_ok=True)
        self.cache = cache

        missing = [k for k in REQUIRED_INPUTS if not getattr(args, k, None)]
        if missing:
            raise MissingInputs(
                "no "
                + ", ".join(k for k in missing)
                + " to load. Pass "
                + ", ".join(f"--{k} PATH" for k in missing)
                + " on the command line, or set "
                + ", ".join(ENV_VARS[k] for k in missing)
                + " in the environment to avoid retyping them.",
                missing,
            )

        print("Loading inputs")
        self.stack = tiffstack.TiffStack(args.raw, pattern=args.pattern)
        print(f"  raw stack     {self.stack.n_slices} slices, "
              f"{self.stack.n_rows} x {self.stack.n_cols}, {self.stack.dtype}")

        self.graph_paths = graph_paths(args.graph)
        self.graph, self.graph_trees = load_graphs(self.graph_paths)
        print(f"  spatial graph {self.graph.n_vertex} vertices, {self.graph.n_edge} edges, "
              f"{self.graph.n_point} points")

        self.lattice_info = amira.read_lattice_header(args.seg)
        print(f"  segmentation  {'x'.join(str(int(v)) for v in self.lattice_info.dims)}, "
              f"fields {sorted(self.lattice_info.fields)}")
        # A `.Regions.am` names its trees; a plain binary mask has no Materials block
        # at all. Everything that treats the two apart reads this one attribute.
        self.materials = self.lattice_info.materials
        if len(self.lattice_info.regions) > 1:
            named = ", ".join(f"{m.name}={m.value}"
                              for m in self.lattice_info.regions)
            print(f"  materials     {named}")

        # **Asked for, never inferred.** Both of the values this used to fall back on
        # -- the folder/TIFF-tag guess and the lattice bounding box -- are wrong in
        # exactly the way that cannot be noticed: they are internally consistent, so
        # the session validates cleanly and every radius, length and volume in it is
        # off by one quiet factor. LADAF-2024-28 records 32.99 um for an acquisition
        # at 32.04, and the only place the true number appears is in a file name that
        # nothing reads. So the two guesses are offered as suggestions and the
        # operator states the value; `voxel_is_truth` then makes it the authority.
        voxel = args.voxel_um
        if voxel is None:
            raise InputError(
                "the raw voxel size is required and is not inferred: pass --voxel-um "
                "(the Data tab has a field for it).\n"
                f"  slice names / TIFF tags suggest: "
                f"{self._suggest(self.stack.nominal_voxel_um)}\n"
                f"  the segmentation bounding box implies: "
                f"{self._suggest(self._bbox_voxel())} at bin 1\n"
                "  Give the acquisition's own value. A rounded one is not harmless: "
                "it scales every radius and length in the session by the same error."
            )
        self.frame = frame.WorldFrame.from_inputs(
            self.stack.shape, voxel, self.lattice_info, stl_scale=args.stl_scale,
            voxel_is_truth=True,
        )
        self._apply_unit_correction()

        labels_field = args.labels_field
        if labels_field not in self.lattice_info.fields:
            if labels_field == "Labels" and len(self.lattice_info.fields) == 1:
                actual = next(iter(self.lattice_info.fields))
                print(f"  label field   using sole field '{actual}' (requested 'Labels')")
                labels_field = actual
            else:
                raise InputError(
                    f"--labels-field '{labels_field}' not in "
                    f"{sorted(self.lattice_info.fields)}"
                )
        self.labels = rle.open_lattice(
            args.seg, self.lattice_info.fields[labels_field], self.lattice_info.dims, cache
        )

        # Painting wraps the lattice in a composited reader. Everything downstream
        # reaches the mask through `slice_z`, so the overlays, the collapse
        # detector, the isosurfaces and the skeletoniser all see the corrections
        # without any of them knowing an edit store exists.
        self.edits = None
        if args.paint:
            from .edit.maskedit import MaskEdits, MaskSource

            if args.edits and Path(args.edits).exists():
                self.edits = MaskEdits.load(args.edits, expect_dims=self.frame.seg_dims)
                print(f"  mask edits    {self.edits.describe()} ({args.edits})")
            self.labels = MaskSource(self.labels, self.edits)
            self.edits = self.labels.edits

        # The whole mask, held in RAM once someone asks for it. Constructed here so
        # it wraps the *final* labels -- the composited MaskSource when painting is
        # on -- but nothing is decoded until the first whole-tree isosurface.
        from .edit.lattice import MaskVolume

        self.mask = MaskVolume(self.labels)

        self.probability = None
        if args.probability and args.probability_field in self.lattice_info.fields:
            self.probability = rle.open_lattice(
                args.seg,
                self.lattice_info.fields[args.probability_field],
                self.lattice_info.dims,
                cache,
            )

        self.mesh = None
        self.slicer = None
        if not args.no_surface:
            # Three cases, not two. `--surface` is genuinely optional, so it can be
            # absent as well as wrong, and `Path(None)` raises rather than reporting.
            # It used to be unreachable only because the flag carried a built-in
            # default that always named a real file.
            if not args.surface:
                print("  surface       not given; continuing without it")
            else:
                surf = Path(args.surface)
                if surf.exists():
                    # The frame's scale, not the argument's: it carries the unit
                    # correction, and a surface left on the file scale would sit a
                    # few percent out from the graph it is drawn with.
                    self.mesh = stl_slice.load_surface(surf, scale=self.frame.stl_scale)
                    self.slicer = stl_slice.SurfaceSlicer(self.mesh)
                    print(f"  surface       {self.mesh.n_cells} triangles")
                else:
                    print(f"  surface       not found ({surf}); continuing without it")

        print()
        print(self.frame.describe())
        print()

    @staticmethod
    def _suggest(value) -> str:
        return "nothing" if value is None else f"--voxel-um {float(value):.4f}"

    def _bbox_voxel(self):
        """What the lattice bounding box implies, or None if it cannot say.

        Only ever a suggestion in an error message, so a header that does not carry a
        bounding box must not turn "you have to tell me the voxel size" into a
        traceback about the header.
        """
        try:
            return float(self.lattice_info.spacing[0])
        except Exception:  # noqa: BLE001 - a suggestion is allowed to be unavailable
            return None

    def _apply_unit_correction(self) -> None:
        """Put the graph and anything else read in file units onto the frame's scale.

        Nothing happens unless the stated voxel size disagreed with the segmentation
        bounding box. When it did, the graph moves with the lattice, because the two
        have to keep describing the same object.

        **A graph this package has already corrected is left alone.** The scale factor
        comes from the *segmentation*, which does not change when a graph is written,
        so re-loading an output beside the same mask would apply the same factor a
        second time -- and the result would still sit inside the mask and still pass
        every check. `amira.read_voxel_stamp` is what distinguishes them, and a
        selection that mixes stamped and unstamped files is refused rather than
        guessed at: they are in two different unit systems and merging them is not
        something a default can decide.
        """
        if not self.frame.corrected:
            return
        stamps = [amira.read_voxel_stamp(path) for path in self.graph_paths]
        matching = [
            v for v in stamps
            if v is not None and abs(v - self.frame.voxel_um) <= 1e-6 * max(v, 1.0)
        ]
        if len(matching) == len(stamps) and stamps:
            print(f"  voxel size    {self.frame.voxel_um:.4f} um; the skeleton already "
                  f"records this scale, so it is not rescaled again")
            return
        if matching:
            raise InputError(
                "these skeletons are not in the same units: "
                + ", ".join(
                    f"{Path(p).name} "
                    + (f"is at {v:.4f} um" if v is not None else "is unmarked")
                    for p, v in zip(self.graph_paths, stamps)
                )
                + ". Load files that share one scale, or drop --voxel-um to work in "
                  "the files' own units."
            )
        print(f"  voxel size    {self.frame.correction_note()}")
        frame.rescale_graph(self.graph, self.frame)

    def validate(self, strict=True):
        checks = frame.validate(self.frame, self.graph, labels=self.labels, mesh_um=self.mesh)
        frame.report(checks, strict=strict)
        self._coverage_notes()
        return checks

    def _coverage_notes(self):
        """Report how much of the tree each input actually covers -- not a pass/fail."""
        from scipy.spatial import cKDTree

        if self.mesh is not None and self.mesh.n_points:
            tree = cKDTree(np.asarray(self.mesh.points))
            d, _ = tree.query(self.graph.points)
            covered = float(np.mean(d < self.graph.thickness * 1.5))
            print(f"  note: the surface covers {100 * covered:.0f}% of the skeleton "
                  f"({'partial model' if covered < 0.9 else 'full model'})")
        if self.probability is not None:
            sample = np.concatenate([self.probability.slice_z(z).ravel()[::97] for z in (200, 600, 1000)])
            uniq = np.unique(sample)
            if len(uniq) <= 8 and (sample == sample.max()).mean() > 0.95:
                print(f"  note: the probability field is degenerate here "
                      f"(values {list(uniq)}, {100 * (sample == sample.max()).mean():.1f}% at max) - "
                      f"it is Avizo label confidence, not the network's pre-threshold output, "
                      f"so it carries no sub-threshold vessel signal")
        print()

    def builder(self):
        """The slab builder, cached so its decoded-slice cache stays warm.

        `interactive` kept one across picks and `inspect` made one per call; both
        still get what they had, and the rule for when the cache must be dropped now
        lives in `invalidate_builder` rather than at each call site.
        """
        if self._builder is None:
            self._builder = viewer2d.SlabBuilder(
                self.frame,
                self.stack,
                self.graph,
                labels=self.labels,
                probability=self.probability,
                slicer=self.slicer,
            )
        return self._builder

    def invalidate_builder(self):
        """Drop the cached builder. Its per-plane caches are keyed to one frame."""
        self._builder = None

    def reload_graph(self, path):
        """Re-read the skeleton alone, leaving the frame and the lattice in place.

        The graph is the only input that does not participate in the coordinate
        frame, which is what makes this safe -- and it is the input a repair chain
        changes at every step. Takes several, like the initial load: re-selecting a
        left and a right skeleton must go through the 0.4 s path too, not force a
        full reload for the one input that never needed one.
        """
        paths = graph_paths(path)
        self.graph, self.graph_trees = load_graphs(paths)
        self.graph_paths = paths
        self.args.graph = list(paths)
        # The step in a repair chain that loads what the last command wrote is exactly
        # where a corrected graph comes back, so the same rule applies here as on the
        # first load: rescale it, unless it says it is already on this scale.
        self._apply_unit_correction()
        self.invalidate_builder()
        print(f"  spatial graph {self.graph.n_vertex} vertices, {self.graph.n_edge} edges, "
              f"{self.graph.n_point} points  "
              f"({', '.join(Path(p).name for p in paths)})")
        return self.graph

    def reload_surface(self, path):
        """Re-read the STL alone. Only `stl_scale` ties it to the frame."""
        surf = Path(path)
        if not surf.exists():
            self.mesh = None
            self.slicer = None
        else:
            self.mesh = stl_slice.load_surface(surf, scale=self.frame.stl_scale)
            self.slicer = stl_slice.SurfaceSlicer(self.mesh)
            print(f"  surface       {self.mesh.n_cells} triangles  ({surf.name})")
        self.args.surface = str(path)
        self.invalidate_builder()
        return self.mesh

    def close(self):
        """Drop the big buffers, so "did we leak?" has an answer.

        The compressed lattice is tens of MB and the decoded slice caches rather
        more; a GUI that loads five datasets in a session needs them to go. The
        resident mask is 2.34 GB and is now the largest single object here, so it
        goes first.
        """
        self._builder = None
        if getattr(self, "mask", None) is not None:
            self.mask.release()
        self.mask = None
        for owner, attr in ((self.stack, "_cache"), (self.labels, "_buf"),
                            (self.probability, "_buf")):
            if owner is not None and hasattr(owner, attr):
                try:
                    getattr(owner, attr).clear()
                except AttributeError:
                    setattr(owner, attr, None)
        self.labels = None
        self.probability = None
        self.mesh = None
        self.slicer = None
        self.stack = None

    def find_candidates(self):
        """Graph heuristics plus, unless disabled, the image-based collapse detector."""
        c = candidates.find_candidates(
            self.graph,
            self.frame,
            min_hops=self.args.min_hops,
            murray_percentile=self.args.murray_percentile,
        )
        if not self.args.no_crosssection:
            sites, profile = crosssection.find_sites(
                self.graph,
                self.frame,
                self.labels,
                severity_iso=self.args.severity_iso,
                mismatch_factor=self.args.mismatch_factor,
                start_id=len(c),
            )
            print(f"  {profile.summary(self.graph.thickness)}")
            c = c + sites
        out = candidates.write_csv(c, self.cache / "candidates.csv")
        print(candidates.summarise(c))
        print(f"\n  -> {out}\n")
        return c


def resolve_target(args, session, cands) -> np.ndarray | None:
    """Turn any --goto-* flag into a world point in micrometres."""
    if args.goto_um is not None:
        return np.asarray(args.goto_um, dtype=np.float64)
    if args.goto_candidate is not None:
        match = [c for c in cands if c.id == args.goto_candidate]
        if not match:
            raise SystemExit(f"no candidate with id {args.goto_candidate}")
        return match[0].xyz
    if args.goto_slice is not None:
        row = args.goto_row if args.goto_row is not None else session.stack.n_rows // 2
        col = args.goto_col if args.goto_col is not None else session.stack.n_cols // 2
        return session.frame.raw_to_um([[args.goto_slice, row, col]])[0]
    return None


def build_slab(session, target_um, args):
    print(f"Building slab at {target_um[0]:,.0f}, {target_um[1]:,.0f}, {target_um[2]:,.0f} um")
    slab = session.builder().build(
        target_um,
        half_slices=args.slab,
        roi_px=args.roi,
        with_probability=args.probability,
        with_tube=not args.no_tube,
        with_surface=not args.no_surface,
    )
    print(f"  slices {slab.z_lo}-{slab.z_hi - 1}, rows {slab.row0}-{slab.row1}, "
          f"cols {slab.col0}-{slab.col1}")
    print(f"  {len(slab.contours or [])} surface contours, {len(slab.skel_pts)} centreline points")
    return slab


def inspect(session, target_um, args):
    """Headless path: build one slab and block on its own napari window."""
    paint = None
    if args.paint:
        from .edit.paint import PaintSession

        paint = PaintSession(session.labels, session.frame,
                             size_vox=args.paint_box, edits_path=args.edits)
    viewer2d.show(build_slab(session, target_um, args), session.frame, paint=paint)
    if paint is not None:
        paint.commit()
        saved = paint.save()
        if saved is not None:
            print(f"[paint] wrote {saved}  ({paint.edits.describe()})")
        elif not paint.edits.is_empty:
            print(f"[paint] WARNING: {paint.edits.describe()} were never saved "
                  f"(no --edits path was given)")


def interactive(session, cands, args) -> int:
    """3D window and slice browser side by side, sharing one Qt event loop.

    The state that used to live in this function's closures is now a `ViewerApp`,
    so the control panel can swap the dataset underneath it. What this function
    does -- and what the command line sees -- is unchanged.
    """
    import napari

    from .app import ViewerApp
    from .controlpanel import attach_control_dock
    from .viewer3d import Picker3D

    app = ViewerApp(args, session=session)
    app.cands = list(cands or [])

    # `session` is None when nothing was named on the command line. Every one of these
    # is already optional on `Picker3D`, and `_populate` skips the actors whose data is
    # missing -- it is the same state `app.unload()` leaves behind between datasets, so
    # an empty window is a state the picker already knows how to be in.
    picker = Picker3D(
        session.graph if session else None,
        session.mesh if session else None,
        app.cands,
        session.frame if session else None,
        on_open=app.open_slices,
        stack=session.stack if session else None,
        labels=session.labels if session else None,
        mask=session.mask if session else None,
        materials=session.materials if session else (),
        seg_box_um=args.seg_box_um, seg_stride=args.seg_stride,
        plane_opacity=args.plane_opacity,
        surface_opacity=args.surface_opacity,
    )
    app.picker = picker
    plotter = picker.build()

    app._apply_volume()
    app._apply_paint()
    if getattr(args, "edit", False):
        app.enable_edit()

    attach_control_dock(app, plotter)

    if session is None:
        print("3D window open with no dataset. Name the raw folder, the skeleton and "
              "the segmentation\nin the control dock's Data tab and press 'Load all'.")
    else:
        print("3D window open. Double-click a vessel, press 'v' for slices, 'n' to walk "
              "candidates, 'q' to finish.")
    print("The 'control' dock runs every command in docs/CLI.md and can load a "
          "different dataset without restarting.\n")
    if app.paint is not None:
        from .edit.paint import PAINT_LAYER

        print(f"Paint from the 'paint' dock in the slice window: select the "
              f"'{PAINT_LAYER}' layer, then 2 to paint, 4 to erase, "
              f"[ and ] for brush size.\n")

    try:
        napari.run()
    finally:
        app.finish()
    return 0


def needs_a_dataset(args) -> str:
    """The name of the requested mode that cannot run without inputs, or "".

    Everything here either reads the data immediately or navigates to a place inside
    it. Only the bare interactive viewer can start without any and wait to be told.
    """
    if args.validate_only:
        return "--validate-only"
    if args.selftest:
        return "--selftest"
    for flag in ("goto_um", "goto_candidate", "goto_slice"):
        if getattr(args, flag, None) is not None:
            return "--" + flag.replace("_", "-")
    return ""


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        session = Session(args)
    except MissingInputs as exc:
        # Nothing was named. That is fatal for every mode that has to read the data,
        # and merely an empty start for the viewer, which can be pointed at a dataset
        # from its Data tab once it is open.
        mode = needs_a_dataset(args)
        if mode:
            raise SystemExit(f"{mode} needs a dataset. {exc}") from exc
        session = None
        flags = " ".join(f"--{k}" for k in exc.missing)
        variables = " ".join(ENV_VARS[k] for k in exc.missing)
        print(f"No dataset named ({flags}; or {variables}).")
        print("Starting empty - load one from the control dock's Data tab.\n")
    except InputError as exc:
        # `SystemExit(message)` prints to stderr and exits 1, which is exactly what
        # the two `raise SystemExit` calls this replaced used to do.
        raise SystemExit(str(exc)) from exc

    strict = not args.no_strict
    if args.validate_only:
        session.validate(strict=strict)
        return 0

    if args.selftest:
        from .selftest import run_selftest

        return 0 if run_selftest(session) else 1

    cands = []
    if session is not None:
        session.validate(strict=strict)

        if not args.no_candidates:
            cands = session.find_candidates()

        target = resolve_target(args, session, cands)
        if target is not None:
            inspect(session, target, args)
            return 0

    return interactive(session, cands, args)


if __name__ == "__main__":
    sys.exit(main())
