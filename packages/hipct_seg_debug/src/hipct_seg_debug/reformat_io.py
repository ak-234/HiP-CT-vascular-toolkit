"""Write a reformat stack to disk, and open it again without rebuilding it.

Building a reformat is the expensive part: 707 planes of a 26 mm run take 16 s, 1441
planes of a 47 mm run take 37 s, and almost all of that is TIFF decode. Without this
module that work is thrown away when the napari window closes.

**A stack is not just images, and saving it as though it were would be the mistake
here.** The napari overlay reads arclength, radius, um/px and the local radius of
curvature for whichever plane is on screen; the 3D window draws the plane frames and
walks the current section along the vessel. Every one of those numbers lives on the
:class:`~.reformat.Centreline` and :class:`~.reformat.PlaneGeometry` beside the arrays.
Save the pixels alone and what comes back is a silent picture with no scale and no place
in the tree -- so the geometry travels with it, and a reloaded stack is the same object
the builder produced.

Three containers, one document. `npz` is a single compressed file; `tiff` and `npy` are
directories, the first so the images open directly in Fiji. They differ only in how the
same two things -- an array table and a JSON document -- are laid out, and both writers
and both readers go through :func:`_arrays` and :func:`_document` so the formats cannot
drift apart.

Two conventions are borrowed rather than reinvented:

* ``edit/maskedit.py`` writes with ``np.savez_compressed`` and reads with
  ``allow_pickle=False``, checking the lattice dimensions on load rather than trusting
  them -- because a store applied to the wrong lattice "would scatter corrections into
  unrelated tissue and look entirely plausible". A reformat opened against the wrong
  tree fails the same way, so the frame is checked too.
* ``edit/crop.py`` versions its sidecar with a schema string compared for strict
  equality, and records a source fingerprint whose mismatch is a *warning* rather than a
  refusal. Both apply here, and the difference between the two is deliberate -- see
  :func:`load`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import reformat as rf

#: Compared for strict equality on load. A document written by a different version would
#: be read on the strength of fields that mean something else.
SCHEMA = "hipct.reformat/1"

#: The three layouts the GUI offers.
FORMATS = ("npz", "tiff", "npy")

#: Sidecar name inside the two directory formats. Also what :func:`load` is pointed at to
#: open one, so a single file dialog can serve all three.
DOCUMENT_NAME = "geometry.json"

#: Beyond this many segments the name carries a count instead of the list.
NAME_MAX_SEGMENTS = 3

#: Arrays on the centreline, geometry and curvature, by the attribute they come from.
_CENTRELINE_ARRAYS = ("coords_um", "radii_um", "arclen_um", "tangents",
                      "normals", "binormals", "seg_ids")
_GEOMETRY_ARRAYS = ("half_um", "px_um", "requested_um")
_CURVATURE_ARRAYS = ("theta_rad", "ds_um", "r_step_um", "r_point_um")


class ReformatIOError(RuntimeError):
    """A stack on disk cannot be read as one."""


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #


def suggest_name(reformat, base: str) -> str:
    """``LAD`` -> ``LAD__native_61px__seg306``.

    Mode and frame size come first because they are what decide whether two stacks are
    comparable at all; the segment list comes last because it is the part that can run
    long, and gets collapsed to a count past :data:`NAME_MAX_SEGMENTS`.

    The segment ids are those of the chain **actually built**, not of the selection.
    Those differ whenever the selection was not one connected run, and a filename that
    describes the request rather than the file is worse than no filename at all.
    """
    geom = reformat.geometry
    stem = _sanitise(base) or "reformat"
    segs = _built_segments(reformat)
    if not segs:
        tail = "segnone"
    elif len(segs) <= NAME_MAX_SEGMENTS:
        tail = "seg" + "-".join(str(s) for s in segs)
    else:
        tail = f"seg{segs[0]}+{len(segs) - 1}more"
    return f"{stem}__{geom.mode}_{geom.size_px}px__{tail}"


def _built_segments(reformat) -> list[int]:
    """Segment ids of the run that was sampled, in the order it was walked."""
    chains = getattr(reformat, "chains", ())
    if chains:
        return list(chains[0].segment_ids)
    ids = np.asarray(reformat.centreline.seg_ids)
    # Order of first appearance, not sorted: it is a path, and the order is meaningful.
    _vals, first = np.unique(ids, return_index=True)
    return [int(v) for v in ids[np.sort(first)]]


def _sanitise(text: str) -> str:
    """A user's base name, reduced to something every filesystem will take.

    Windows rejects ``< > : " / \\ | ? *`` outright and silently drops a trailing dot or
    space, which turns a name the operator typed into a different file. Replaced rather
    than stripped so two names that differ only in punctuation do not collide.
    """
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(text or "")).strip(" .")
    return re.sub(r"\s+", "_", cleaned)[:80]


# --------------------------------------------------------------------------- #
# The one document, and the one array table
# --------------------------------------------------------------------------- #


def _arrays(reformat) -> dict:
    """Everything bulk and numeric, by the name it is stored under.

    ``mask`` is omitted rather than stored as zeros when absent, so the distinction
    between "no segmentation was sampled" and "the segmentation was empty here" survives
    the round trip. Those mean different things to anyone reading the stack later.
    """
    line, geom, curv = reformat.centreline, reformat.geometry, reformat.centreline.curvature
    out = {"raw": np.asarray(reformat.raw)}
    if reformat.mask is not None:
        out["mask"] = np.asarray(reformat.mask)
    for name in _CENTRELINE_ARRAYS:
        out[name] = np.asarray(getattr(line, name))
    for name in _GEOMETRY_ARRAYS:
        out[f"geom_{name}"] = np.asarray(getattr(geom, name))
    for name in _CURVATURE_ARRAYS:
        out[f"curv_{name}"] = np.asarray(getattr(curv, name))
    out["graph_points"] = np.asarray(reformat.graph_points).reshape(-1, 3)
    # A set of raw slice indices: bulk and numeric, so it belongs here rather than
    # bloating the JSON with a few thousand integers.
    out["stat_slices"] = np.asarray(sorted(reformat.stats.slices), dtype=np.int64)
    if line.seed_normal is not None:
        out["seed_normal"] = np.asarray(line.seed_normal, dtype=np.float64)
    return out


def _document(reformat, provenance: dict | None = None) -> dict:
    """Everything scalar or structural, as plain JSON."""
    line, geom = reformat.centreline, reformat.geometry
    curv, stats = line.curvature, reformat.stats
    return {
        "schema": SCHEMA,
        "kind": "reformat",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "centreline": {
            "step_um": float(line.step_um),
            "smooth_window_um": float(line.smooth_window_um),
            "smooth_iters": int(line.smooth_iters),
            "max_move_um": float(line.max_move_um),
            "median_move_um": float(line.median_move_um),
            "notes": list(line.notes),
        },
        "curvature": {"r_min_um": _num(curv.r_min_um), "at_index": int(curv.at_index)},
        "geometry": {
            "size_px": int(geom.size_px),
            "mode": str(geom.mode),
            "n_clamped": int(geom.n_clamped),
            "r_min_um": _num(geom.r_min_um),
            "voxel_um": float(geom.voxel_um),
            "n_over_bound": int(geom.n_over_bound),
            "suggested_size_px": int(geom.suggested_size_px),
        },
        "stats": {
            "chunks": int(stats.chunks),
            "slices_read": int(stats.slices_read),
            "n_samples": int(stats.n_samples),
            "n_outside": int(stats.n_outside),
            "seconds": {k: float(v) for k, v in stats.seconds.items()},
            "order": int(stats.order),
            "strip_reads": int(stats.strip_reads),
            "whole_page_reads": int(stats.whole_page_reads),
        },
        "chains": [
            {
                "steps": [[int(sid), bool(rev)] for sid, rev in c.steps],
                "start_node": int(c.start_node),
                "end_node": int(c.end_node),
                "length_um": float(c.length_um),
                "notes": list(c.notes),
            }
            for c in reformat.chains
        ],
        "notes": list(reformat.notes),
        "provenance": provenance or {},
    }


def _num(value) -> float | None:
    """JSON has no infinity. A straight run genuinely has no curvature bound."""
    v = float(value)
    return None if not np.isfinite(v) else v


def _unnum(value) -> float:
    return np.inf if value is None else float(value)


def provenance(reformat, *, frame=None, graph=None, source=None) -> dict:
    """Where this stack came from, in enough detail to notice it does not belong here.

    The frame is recorded because every world coordinate in the stack is meaningless
    against a different one -- the 3D plane frames would be drawn somewhere unrelated and
    look entirely plausible doing it.

    The segments are recorded twice, by id and by ``crop.segment_key``. Ids are array
    indices and are renumbered by any deletion; the geometric keys are reversal-invariant
    and survive that, so a stack can still say which vessel it is in a repaired graph.
    """
    out: dict = {"segments": _built_segments(reformat)}
    if source is not None:
        out["source"] = str(source)
    if graph is not None:
        try:
            from .edit.crop import segment_key, source_fingerprint

            fingerprint = source_fingerprint(source, graph)
            out["source_sha1"] = fingerprint.get("sha1")
            out["source_counts"] = {k: v for k, v in fingerprint.items() if k != "sha1"}
            out["segment_keys"] = [
                segment_key(graph, sid) for sid in out["segments"]
                if graph.has_segment(sid)
            ]
        except Exception:  # noqa: BLE001 - provenance must never cost you the save
            pass
    if frame is not None:
        out["frame"] = {
            "raw_shape": [int(v) for v in frame.raw_shape],
            "raw_voxel": [float(v) for v in np.asarray(frame.raw_voxel)],
            "seg_dims": [int(v) for v in np.asarray(frame.seg_dims)],
            "seg_origin": [float(v) for v in np.asarray(frame.seg_origin)],
            "seg_spacing": [float(v) for v in np.asarray(frame.seg_spacing)],
        }
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def save(reformat, path, *, fmt: str = "npz", frame=None, graph=None, source=None) -> Path:
    """Write a stack. Returns the path actually written.

    ``fmt`` is one of :data:`FORMATS`. ``npz`` writes one file; ``tiff`` and ``npy``
    write a directory. The suffix is supplied when it is missing rather than demanded, so
    a name typed without one still lands somewhere sensible.
    """
    if fmt not in FORMATS:
        raise ReformatIOError(f"unknown format {fmt!r}; expected one of {list(FORMATS)}")
    path = Path(path)
    arrays = _arrays(reformat)
    document = _document(reformat, provenance(reformat, frame=frame, graph=graph,
                                              source=source))

    if fmt == "npz":
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.array(json.dumps(document)), **arrays)
        return path

    path.mkdir(parents=True, exist_ok=True)
    (path / DOCUMENT_NAME).write_text(json.dumps(document, indent=2), encoding="utf-8")

    if fmt == "npy":
        for name, value in arrays.items():
            np.save(path / f"{name}.npy", value)
        return path

    import tifffile

    # Images as TIFF so they open in Fiji; everything else stays .npy beside them. A
    # centreline is not an image and writing it as one would only make it harder to read.
    for name in ("raw", "mask"):
        if name in arrays:
            tifffile.imwrite(path / f"{name}.tif", arrays[name])
    for name, value in arrays.items():
        if name not in ("raw", "mask"):
            np.save(path / f"{name}.npy", value)
    (path / "README.txt").write_text(_readme(reformat), encoding="utf-8")
    return path


def _readme(reformat) -> str:
    """What someone opening `raw.tif` in Fiji cannot otherwise know."""
    geom, line = reformat.geometry, reformat.centreline
    return (
        "A curved-planar reformat: cross-sections cut perpendicular to a vessel\n"
        "centreline and stacked along it.\n"
        "\n"
        f"raw.tif   {reformat.raw.shape}  {reformat.raw.dtype}\n"
        + (f"mask.tif  {reformat.mask.shape}  {reformat.mask.dtype}  (segmentation)\n"
           if reformat.mask is not None else "")
        + "\n"
        "Axis 0 is the plane index, walking ALONG the vessel. Axes 1 and 2 are within\n"
        "the plane: axis 1 is +binormal (rows), axis 2 is +normal (columns), and the\n"
        "centreline is the exact centre pixel of every image.\n"
        "\n"
        f"Along the vessel : {line.step_um:.3f} um per plane, "
        f"{line.length_um / 1000:.2f} mm total over {len(reformat.raw)} planes\n"
        f"In plane         : {geom.px_um.min():.3f}-{geom.px_um.max():.3f} um per pixel"
        + ("  (constant)\n" if geom.px_um.max() - geom.px_um.min() < 1e-9 else
           "  (VARIES per plane -- see geom_px_um.npy)\n")
        + f"Mode             : {geom.mode}\n"
        "\n"
        "Note the plane spacing and the pixel pitch are different numbers, so the stack\n"
        "is not isotropic unless they happen to match. geometry.json carries the full\n"
        "per-plane geometry, and the .npy files beside it hold the centreline itself.\n"
    )


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@dataclass
class LoadedReformat:
    """A stack read back from disk, with what is known about where it came from."""

    reformat: object
    document: dict
    path: Path
    #: True until :func:`check_against` is given a frame that disagrees. World
    #: coordinates are only meaningful against the frame the stack was cut in.
    frame_matches: bool = True
    notes: list = field(default_factory=list)

    @property
    def provenance(self) -> dict:
        return self.document.get("provenance", {})

    def describe(self) -> str:
        out = [self.reformat.describe()]
        written = self.document.get("written")
        if written:
            out.append(f"  loaded from {self.path.name}, written {written}")
        out += [f"  note: {n}" for n in self.notes]
        return "\n".join(out)


def load(path) -> LoadedReformat:
    """Read a stack back. Point at the ``.npz``, or at a folder or its ``geometry.json``.

    The schema is compared for strict equality and a mismatch **raises**, following
    ``crop.load``: a document written by a different version would be interpreted on the
    strength of fields that mean something else, and a stack silently misread is worse
    than one that will not open.

    Everything *else* wrong with a stack is a note rather than a refusal -- see
    :func:`check_against`. The distinction is that a schema mismatch means the file
    cannot be understood, whereas a frame or graph mismatch means it can be understood
    perfectly well and merely does not belong to what is currently loaded. The images are
    self-contained and worth looking at either way.
    """
    path = Path(path)
    if path.is_dir():
        arrays, document = _read_directory(path)
    elif path.name == DOCUMENT_NAME:
        path = path.parent
        arrays, document = _read_directory(path)
    else:
        arrays, document = _read_npz(path)

    schema = document.get("schema")
    if schema != SCHEMA:
        raise ReformatIOError(
            f"{path.name}: expected schema {SCHEMA!r}, found {schema!r}"
        )
    return LoadedReformat(reformat=_rebuild(arrays, document), document=document,
                          path=path)


def _read_npz(path: Path) -> tuple[dict, dict]:
    if not path.exists():
        raise ReformatIOError(f"{path} does not exist")
    try:
        # allow_pickle=False throughout: a stack is data, and a file that can execute on
        # open is not something to hand to an operator who was told it was an image.
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            document = json.loads(str(z["meta"]))
    except ReformatIOError:
        raise
    except Exception as exc:  # noqa: BLE001 - report the file, not a numpy traceback
        raise ReformatIOError(f"{path.name}: not a readable reformat stack ({exc})") from exc
    return arrays, document


def _read_directory(path: Path) -> tuple[dict, dict]:
    doc_path = path / DOCUMENT_NAME
    if not doc_path.is_file():
        raise ReformatIOError(f"{path}: no {DOCUMENT_NAME}; not a saved reformat stack")
    document = json.loads(doc_path.read_text(encoding="utf-8"))
    arrays = {p.stem: np.load(p, allow_pickle=False) for p in sorted(path.glob("*.npy"))}
    for name in ("raw", "mask"):
        tif = path / f"{name}.tif"
        if tif.is_file():
            import tifffile

            arrays[name] = tifffile.imread(tif)
    if "raw" not in arrays:
        raise ReformatIOError(f"{path}: no raw.tif or raw.npy")
    return arrays, document


def _rebuild(arrays: dict, document: dict):
    """Reassemble the dataclasses. The inverse of :func:`_arrays` + :func:`_document`."""
    curv_doc, geom_doc = document["curvature"], document["geometry"]
    line_doc, stat_doc = document["centreline"], document["stats"]

    curvature = rf.Curvature(
        theta_rad=arrays["curv_theta_rad"], ds_um=arrays["curv_ds_um"],
        r_step_um=arrays["curv_r_step_um"], r_point_um=arrays["curv_r_point_um"],
        r_min_um=_unnum(curv_doc["r_min_um"]), at_index=int(curv_doc["at_index"]),
    )
    centreline = rf.Centreline(
        **{name: arrays[name] for name in _CENTRELINE_ARRAYS},
        step_um=float(line_doc["step_um"]),
        curvature=curvature,
        smooth_window_um=float(line_doc["smooth_window_um"]),
        smooth_iters=int(line_doc["smooth_iters"]),
        max_move_um=float(line_doc["max_move_um"]),
        median_move_um=float(line_doc["median_move_um"]),
        seed_normal=arrays.get("seed_normal"),
        notes=tuple(line_doc["notes"]),
    )
    geometry = rf.PlaneGeometry(
        **{name: arrays[f"geom_{name}"] for name in _GEOMETRY_ARRAYS},
        size_px=int(geom_doc["size_px"]), mode=str(geom_doc["mode"]),
        n_clamped=int(geom_doc["n_clamped"]), r_min_um=_unnum(geom_doc["r_min_um"]),
        voxel_um=float(geom_doc["voxel_um"]),
        n_over_bound=int(geom_doc["n_over_bound"]),
        suggested_size_px=int(geom_doc["suggested_size_px"]),
    )
    stats = rf.SampleStats(
        chunks=int(stat_doc["chunks"]), slices_read=int(stat_doc["slices_read"]),
        n_samples=int(stat_doc["n_samples"]), n_outside=int(stat_doc["n_outside"]),
        slices=set(int(v) for v in arrays.get("stat_slices", ())),
        seconds={k: float(v) for k, v in stat_doc["seconds"].items()},
        order=int(stat_doc["order"]), strip_reads=int(stat_doc["strip_reads"]),
        whole_page_reads=int(stat_doc["whole_page_reads"]),
    )
    chains = tuple(
        rf.Chain(
            steps=tuple((int(sid), bool(rev)) for sid, rev in c["steps"]),
            start_node=int(c["start_node"]), end_node=int(c["end_node"]),
            length_um=float(c["length_um"]), notes=tuple(c["notes"]),
        )
        for c in document["chains"]
    )
    return rf.Reformat(
        raw=arrays["raw"], mask=arrays.get("mask"), centreline=centreline,
        geometry=geometry, graph_points=arrays["graph_points"], stats=stats,
        chains=chains, notes=tuple(document["notes"]),
    )


def check_against(loaded: LoadedReformat, frame=None, graph=None) -> LoadedReformat:
    """Compare a loaded stack with the session it is about to be shown beside.

    Sets ``frame_matches`` and appends notes. **Never raises**, because none of this
    stops the images being worth looking at -- it only decides whether the *world*
    coordinates mean anything here, and therefore whether the 3D overlays can be drawn.
    """
    saved = loaded.provenance.get("frame")
    if frame is not None and saved:
        same = (
            [int(v) for v in frame.raw_shape] == saved.get("raw_shape")
            and np.allclose(np.asarray(frame.raw_voxel, dtype=float),
                            saved.get("raw_voxel", []), rtol=0, atol=1e-6)
            and np.allclose(np.asarray(frame.seg_origin, dtype=float),
                            saved.get("seg_origin", []), rtol=0, atol=1e-6)
        )
        loaded.frame_matches = bool(same)
        if not same:
            loaded.notes.append(
                "this stack was cut in a different coordinate frame from the dataset "
                "now loaded. The images are unaffected, but its world positions are not "
                "meaningful here, so the 3D plane overlays are not drawn."
            )
    elif frame is not None and not saved:
        loaded.frame_matches = False
        loaded.notes.append("no frame was recorded with this stack; 3D overlays skipped")

    counts = loaded.provenance.get("source_counts")
    if graph is not None and counts:
        now = {"segments": len(graph.segments), "nodes": len(graph.nodes),
               "points": len(graph.points)}
        if any(counts.get(k) != v for k, v in now.items()):
            loaded.notes.append(
                f"the graph has changed since this stack was cut "
                f"(saved {counts}, now {now}); segment ids in the report may no longer "
                f"name the same vessels"
            )
    return loaded
