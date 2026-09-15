"""Process workers for provisional radius measurements with full branch context.

Only plane measurements are distributed. The caller combines them before any
graph-wide calibration or junction/fallback inference, preserving serial results.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np


_WORKER = None


def _init_worker(triple, frame, lattice, options):
    global _WORKER
    import cv2
    from threadpoolctl import threadpool_limits

    from ..rle import open_lattice
    from .graphmodel import EditableGraph

    # Each worker owns one CPU; nested BLAS/OpenCV pools would oversubscribe it.
    threadpool_limits(limits=1)
    cv2.setNumThreads(1)
    if isinstance(lattice, tuple):
        path, field, dims, cache = lattice
        labels = open_lattice(path, field, dims, cache_dir=cache)
    else:
        labels = lattice
    _WORKER = EditableGraph(triple), frame, labels, options


def _measure_batch(sids):
    from .radius_perimeter import measure_radii

    graph, frame, labels, options = _WORKER
    return measure_radii(graph, frame, labels, _segment_ids=sids, _raw_only=True,
                         **options)


def measure_provisional(graph, frame, labels, sids, *, workers, options, progress=None):
    from ..rle import ByteRLELattice, RawLattice
    from .radius_perimeter import RadiusResult

    if isinstance(labels, (ByteRLELattice, RawLattice)):
        # Reopen the stream once per worker, instead of pickling its compressed
        # bytes into every submitted task. Row-band caches remain worker-local.
        lattice = (labels.path, labels.field, labels.dims,
                   getattr(labels, "_cache_dir", None))
    elif isinstance(labels, np.ndarray):
        lattice = labels
    else:
        raise ValueError("workers > 1 requires an array or an Amira lattice; "
                         "use workers=1 for a mask with live edits")

    names = ("measured", "source", "reject", "grew", "modes", "old", "arc", "invented")
    combined = {name: {} for name in names}
    combined.update(before=[], result=RadiusResult())
    if not sids:
        return combined
    workers = min(int(workers), len(sids))
    # Several batches per worker balance long vessels against short branches.
    # Consecutive edges retain locality in each worker's decoded row cache.
    batch_size = max(1, min(32, (len(sids) + workers * 4 - 1) // (workers * 4)))
    batches = [sids[i:i + batch_size] for i in range(0, len(sids), batch_size)]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker, initargs=(graph.triple, frame, lattice, options),
    ) as pool:
        done = 0
        # map preserves input order even when workers finish out of order. This
        # also preserves calibration input and junction-length report ordering.
        for batch, raw in zip(batches, pool.map(_measure_batch, batches)):
            for name in names:
                combined[name].update(raw[name])
            combined["before"].extend(raw["before"])
            result, partial = combined["result"], raw["result"]
            result.n_truncated += partial.n_truncated
            result.n_ownership_failed += partial.n_ownership_failed
            result.junction_lengths_um.extend(partial.junction_lengths_um)
            done += len(batch)
            if progress is not None:
                progress(done, len(sids))
    return combined
