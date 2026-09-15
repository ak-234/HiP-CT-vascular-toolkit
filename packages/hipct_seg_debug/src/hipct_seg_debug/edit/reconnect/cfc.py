"""DF21 Cascade Forest training and raw-patch inference for the DPC walk.

The feature contract is deliberately small and rigid: a centred raw ``15^3``
patch max-pooled to ``7^3``, followed by the centred raw ``7^3`` patch.  Both are
flattened in C-order (z, y, x), giving exactly 686 values.  The manifest written
beside a model pins this contract so a differently prepared model fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

LARGE_SIDE = 15
SMALL_SIDE = 7
PATCH_RADIUS = LARGE_SIDE // 2
FEATURE_WIDTH = 2 * SMALL_SIDE**3
ARTIFACT_VERSION = 1
FEATURE_ORDER = "pooled-15-then-7;c-order;zyx;raw"


def patch_feature(patch15: np.ndarray) -> np.ndarray:
    """Return the paper's 686-value descriptor from one centred 15-cube."""
    patch = np.asarray(patch15)
    if patch.shape != (LARGE_SIDE, LARGE_SIDE, LARGE_SIDE):
        raise ValueError(f"CFC patch must be 15x15x15, got {patch.shape}")
    core = patch[:14, :14, :14]
    pooled = core.reshape(7, 2, 7, 2, 7, 2).max(axis=(1, 3, 5))
    small = patch[4:11, 4:11, 4:11]
    return np.concatenate([pooled.ravel(order="C"), small.ravel(order="C")])


def valid_centres(zyx: np.ndarray, shape: Iterable[int]) -> np.ndarray:
    centres = np.asarray(zyx, dtype=np.int64).reshape(-1, 3)
    limit = np.asarray(tuple(shape), dtype=np.int64)
    return np.all((centres >= PATCH_RADIUS) & (centres < limit - PATCH_RADIUS), axis=1)


def features_from_volume(volume: np.ndarray, centres_zyx: np.ndarray) -> np.ndarray:
    """Extract features from an in-memory z/y/x volume at rounded voxel centres."""
    centres = np.rint(np.asarray(centres_zyx)).astype(np.int64).reshape(-1, 3)
    if not np.all(valid_centres(centres, volume.shape)):
        raise ValueError("one or more CFC patch centres are within 7 voxels of a boundary")
    out = np.empty((len(centres), FEATURE_WIDTH), dtype=np.asarray(volume).dtype)
    for i, (z, y, x) in enumerate(centres):
        patch = volume[z - 7:z + 8, y - 7:y + 8, x - 7:x + 8]
        out[i] = patch_feature(patch)
    return out


def features_from_stack(stack, centres_zyx: np.ndarray, progress=None) -> np.ndarray:
    """Extract scattered features while visiting centres in z order.

    ``TiffStack`` caches decoded planes. Sorting means the 15 planes surrounding
    neighbouring centres are shared rather than repeatedly decoded.
    """
    centres = np.rint(np.asarray(centres_zyx)).astype(np.int64).reshape(-1, 3)
    if not np.all(valid_centres(centres, stack.shape)):
        raise ValueError("one or more CFC patch centres are within 7 voxels of a boundary")
    order = np.argsort(centres[:, 0], kind="stable")
    out = np.empty((len(centres), FEATURE_WIDTH), dtype=stack.dtype)
    for done, i in enumerate(order, 1):
        z, y, x = centres[i]
        patch = stack.read_stack_window(z - 7, z + 8, y - 7, y + 8, x - 7, x + 8)
        out[i] = patch_feature(patch)
        if progress is not None and (done == len(order) or done % 250 == 0):
            progress(done, len(order))
    return out


class CfcProbability:
    """A persisted DF21 model evaluated on raw voxels from one DPC ROI."""

    def __init__(self, roi, artifact: str | Path, *, model=None,
                 expected_raw_shape=None):
        self.roi = roi
        self.artifact = Path(artifact)
        self.manifest = json.loads((self.artifact / "manifest.json").read_text())
        self._validate_manifest(expected_raw_shape)
        self.model = model if model is not None else load_model(self.artifact)
        self._cache: dict[tuple[int, int, int], float] = {}

    def _validate_manifest(self, raw_shape) -> None:
        feature = self.manifest.get("feature", {})
        if self.manifest.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported CFC artifact version")
        if feature.get("width") != FEATURE_WIDTH or feature.get("order") != FEATURE_ORDER:
            raise ValueError("CFC artifact uses an incompatible patch feature contract")
        trained_spacing = np.asarray(self.manifest["geometry"]["raw_voxel_um"], float)
        if not np.allclose(trained_spacing, self.roi.spacing_um, rtol=0, atol=1e-5):
            raise ValueError(
                f"CFC raw spacing mismatch: model {trained_spacing.tolist()} vs "
                f"ROI {np.asarray(self.roi.spacing_um).tolist()}"
            )
        if raw_shape is not None and tuple(raw_shape) != tuple(
                self.manifest["geometry"]["raw_shape_zyx"]):
            raise ValueError("CFC artifact was trained on a different raw stack shape")

    def __call__(self, points_um: np.ndarray) -> np.ndarray:
        idx = np.rint(self.roi.to_index(points_um)).astype(np.int64)
        out = np.zeros(len(idx), dtype=np.float64)
        missing: list[tuple[int, tuple[int, int, int]]] = []
        for i, centre in enumerate(idx):
            key = tuple(int(v) for v in centre)
            if key in self._cache:
                out[i] = self._cache[key]
            elif valid_centres(centre.reshape(1, 3), self.roi.volume.shape)[0]:
                missing.append((i, key))
        if missing:
            centres = np.asarray([key for _, key in missing], dtype=np.int64)
            x = features_from_volume(self.roi.volume, centres)
            values = np.asarray(self.model.predict_proba(x), dtype=np.float64)[:, 1]
            for (i, key), value in zip(missing, values):
                out[i] = self._cache[key] = float(value)
        return out

    def precompute(self, batch_size: int = 4096) -> int:
        """Fill the ROI cache in batches, amortising DF21's per-call overhead.

        This is intended for an omega sweep, where the same small review ROI is
        walked dozens of times. A one-off connection remains lazily evaluated.
        """
        shape = np.asarray(self.roi.volume.shape, dtype=np.int64)
        axes = [np.arange(PATCH_RADIUS, int(n) - PATCH_RADIUS) for n in shape]
        total = int(np.prod([len(axis) for axis in axes]))
        if total <= 0:
            return 0
        centres = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        for start in range(0, len(centres), batch_size):
            batch = centres[start:start + batch_size]
            missing = np.asarray([
                c for c in batch if tuple(int(v) for v in c) not in self._cache
            ], dtype=np.int64).reshape(-1, 3)
            if not len(missing):
                continue
            values = np.asarray(
                self.model.predict_proba(features_from_volume(self.roi.volume, missing)),
                dtype=np.float64,
            )[:, 1]
            self._cache.update({tuple(int(v) for v in c): float(p)
                                for c, p in zip(missing, values)})
        return total


def load_model(artifact: str | Path):
    try:
        from deepforest import CascadeForestClassifier
    except ImportError as exc:  # pragma: no cover - exercised in the Python 3.9 env
        raise RuntimeError(
            "DF21 is unavailable; run this command in the hipct-cfc Python 3.9 environment"
        ) from exc
    model = CascadeForestClassifier(verbose=0)
    model.load(str(Path(artifact) / "model"))
    model.verbose = 0  # load restores the training-time verbosity from the artifact
    return model


@dataclass
class TrainingSamples:
    raw_zyx: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    groups: np.ndarray


class _Reservoir:
    """Uniform fixed-size sample using random priorities, updated in chunks."""

    def __init__(self, size: int, rng: np.random.Generator):
        self.size = int(size)
        self.rng = rng
        self.coords = np.empty((0, 3), dtype=np.int64)
        self.keys = np.empty(0, dtype=np.float64)

    def update(self, coords: np.ndarray) -> None:
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
        if not len(coords) or self.size <= 0:
            return
        keys = self.rng.random(len(coords))
        combined_coords = np.vstack([self.coords, coords])
        combined_keys = np.concatenate([self.keys, keys])
        if len(combined_keys) > self.size:
            keep = np.argpartition(combined_keys, -self.size)[-self.size:]
            combined_coords, combined_keys = combined_coords[keep], combined_keys[keep]
        self.coords, self.keys = combined_coords, combined_keys


def _positive_samples(spatial_graph, frame, max_positive: int | None,
                      rng: np.random.Generator):
    raw = frame.um_to_raw_index(spatial_graph.points)
    edge_groups = _edge_components(spatial_graph)
    point_groups = np.repeat(edge_groups, spatial_graph.n_edge_points)
    unique, first = np.unique(raw, axis=0, return_index=True)
    groups = point_groups[first]
    keep = valid_centres(unique, frame.raw_shape)
    unique, groups = unique[keep], groups[keep]
    if max_positive is not None and len(unique) > max_positive:
        pick = rng.choice(len(unique), int(max_positive), replace=False)
        unique, groups = unique[pick], groups[pick]
    return unique, groups


def _edge_components(graph) -> np.ndarray:
    parent = np.arange(graph.n_vertex)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in graph.connectivity:
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[rb] = ra
    roots = np.array([root(int(a)) for a in graph.connectivity[:, 0]])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def sample_training_voxels(spatial_graph, labels, frame, *, max_positive=None,
                           seed: int = 0, shell_voxels: float = 7.0,
                           chunk_slices: int = 24) -> TrainingSamples:
    """Create the N/2N/2N paper sampling split without materialising the lattice."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    positive, positive_groups = _positive_samples(spatial_graph, frame, max_positive, rng)
    if not len(positive):
        raise ValueError("no skeleton voxels support a complete 15x15x15 raw patch")
    wanted = 2 * len(positive)
    lumen = _Reservoir(wanted, rng)
    # Seeds are foreground voxels from which a bounded random offset can prove an
    # outside point is within seven voxels of the wall. This avoids a full-volume
    # float64 distance transform (hundreds of MB per slab on this segmentation).
    shell_seeds = _Reservoir(max(2 * wanted, 10_000), rng)
    shell = _Reservoir(wanted, rng)

    positive_seg = frame.um_to_seg_index(frame.raw_to_um(positive))[:, ::-1]
    excluded_by_z: dict[int, list[tuple[int, int]]] = {}
    for z, y, x in positive_seg:
        excluded_by_z.setdefault(int(z), []).append((int(y), int(x)))

    for z in range(labels.nz):
        plane = np.asarray(labels.slice_z(z)) > 0
        for y, x in excluded_by_z.get(z, ()):
            if 0 <= y < plane.shape[0] and 0 <= x < plane.shape[1]:
                plane[y, x] = False
        yx = np.argwhere(plane)
        if not len(yx):
            continue
        seg = np.column_stack([np.full(len(yx), z, dtype=np.int64), yx])
        shell_seeds.update(seg)
        raw = _seg_zyx_to_raw(frame, seg)
        lumen.update(raw[valid_centres(raw, frame.raw_shape)])

    if not len(shell_seeds.coords):
        raise ValueError("segmentation contains no foreground voxels for negative sampling")

    # Rejection-sample outside voxels around true foreground seeds. Every accepted
    # point is therefore outside the lumen and no more than seven segmentation
    # voxels from it, without approximating the wall distance.
    shell_seen: set[tuple[int, int, int]] = set()
    for _attempt in range(100):
        if len(shell.coords) >= wanted:
            break
        batch = max(3 * wanted, 2048)
        source = shell_seeds.coords[rng.integers(0, len(shell_seeds.coords), batch)]
        direction = rng.normal(size=(batch, 3))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        radius = shell_voxels * np.cbrt(rng.random(batch))
        candidate = source + np.rint(direction * radius[:, None]).astype(np.int64)
        shape = np.array([labels.nz, labels.ny, labels.nx], dtype=np.int64)
        keep = np.all((candidate >= 0) & (candidate < shape), axis=1)
        candidate = np.unique(candidate[keep], axis=0)
        outside = np.zeros(len(candidate), dtype=bool)
        for z in np.unique(candidate[:, 0]):
            at = np.flatnonzero(candidate[:, 0] == z)
            plane = np.asarray(labels.slice_z(int(z)))
            outside[at] = plane[candidate[at, 1], candidate[at, 2]] == 0
        candidate = candidate[outside]
        candidate = np.asarray([
            c for c in candidate if tuple(int(v) for v in c) not in shell_seen
        ], dtype=np.int64).reshape(-1, 3)
        shell_seen.update(tuple(int(v) for v in c) for c in candidate)
        raw = _seg_zyx_to_raw(frame, candidate)
        shell.update(raw[valid_centres(raw, frame.raw_shape)])

    if len(lumen.coords) < wanted or len(shell.coords) < wanted:
        raise ValueError(
            f"segmentation supplied only {len(lumen.coords)} lumen and "
            f"{len(shell.coords)} shell negatives; need {wanted} of each"
        )
    negative = np.vstack([lumen.coords, shell.coords])
    nearest = cKDTree(positive).query(negative, k=1)[1]
    negative_groups = positive_groups[nearest]
    return TrainingSamples(
        raw_zyx=np.vstack([positive, negative]),
        labels=np.concatenate([
            np.ones(len(positive), dtype=np.uint8),
            np.zeros(2 * wanted, dtype=np.uint8),
        ]),
        classes=np.concatenate([
            np.zeros(len(positive), dtype=np.uint8),
            np.ones(wanted, dtype=np.uint8),
            np.full(wanted, 2, dtype=np.uint8),
        ]),
        groups=np.concatenate([positive_groups, negative_groups]),
    )


def _seg_zyx_to_raw(frame, seg_zyx: np.ndarray) -> np.ndarray:
    world = frame.seg_to_um(np.asarray(seg_zyx)[:, ::-1])
    return frame.um_to_raw_index(world)


def _df21(seed: int):
    try:
        from deepforest import CascadeForestClassifier
    except ImportError as exc:
        raise RuntimeError(
            "train-cfc requires deep-forest in the hipct-cfc Python 3.9 environment"
        ) from exc
    return CascadeForestClassifier(
        n_bins=255,
        max_layers=20,
        n_estimators=2,
        n_trees=100,
        n_tolerant_rounds=2,
        random_state=seed,
        n_jobs=-1,
        partial_mode=True,
    )


def _metrics(y_true, probability) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_recall_curve,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=np.uint8)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = probability >= 0.5
    tp = int(np.sum((y_true == 1) & predicted))
    tn = int(np.sum((y_true == 0) & ~predicted))
    fp = int(np.sum((y_true == 0) & predicted))
    fn = int(np.sum((y_true == 1) & ~predicted))
    precision, recall, _ = precision_recall_curve(y_true, probability)
    pr_auc = float(abs(np.trapz(precision, recall)))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": pr_auc,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def train_cfc(graph_path, segmentation_path, raw_directory, output_directory, *,
              labels_field="Labels", voxel_um=None, max_positive=None, seed=0,
              overwrite=False, progress=None) -> dict:
    """Train, validate, refit, and persist one scan-specific DF21 artifact."""
    from sklearn.model_selection import GroupShuffleSplit

    from ... import amira, rle
    from ...frame import WorldFrame
    from ...tiffstack import TiffStack

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"CFC output is not empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        model_path = output / "model"
        if model_path.is_dir():
            shutil.rmtree(model_path)
        for name in ("manifest.json", "metrics.json", "samples.npz"):
            path = output / name
            if path.exists():
                path.unlink()

    graph = amira.read_spatial_graph(graph_path)
    info = amira.read_lattice_header(segmentation_path)
    labels = rle.open_lattice(segmentation_path, info.fields[labels_field], info.dims)
    stack = TiffStack(raw_directory)
    nominal = voxel_um or stack.nominal_voxel_um or float(info.spacing[0]) / 2.0
    frame = WorldFrame.from_inputs(stack.shape, nominal, info)
    _validate_geometry(graph, frame)

    samples = sample_training_voxels(
        graph, labels, frame, max_positive=max_positive, seed=seed
    )
    x = features_from_stack(stack, samples.raw_zyx, progress=progress)
    groups = samples.groups
    if len(np.unique(groups)) < 2:
        blocks = samples.raw_zyx // 64
        _, groups = np.unique(blocks, axis=0, return_inverse=True)
    split = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                   random_state=seed).split(x, samples.labels, groups))
    train_idx, validation_idx = split
    validation_model = _df21(seed)
    validation_model.fit(x[train_idx], samples.labels[train_idx])
    probability = validation_model.predict_proba(x[validation_idx])[:, 1]
    metrics = _metrics(samples.labels[validation_idx], probability)
    metrics.update({
        "n_train": int(len(train_idx)), "n_validation": int(len(validation_idx)),
        "validation_groups": sorted(int(v) for v in np.unique(groups[validation_idx])),
    })

    model = _df21(seed)
    model.fit(x, samples.labels)
    model_dir = output / "model"
    model.save(str(model_dir))
    np.savez_compressed(
        output / "samples.npz", raw_zyx=samples.raw_zyx,
        labels=samples.labels, classes=samples.classes, groups=samples.groups,
    )
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_unix": time.time(),
        "scan": "LADAF-2024-28",
        "feature": {
            "large_side": LARGE_SIDE, "small_side": SMALL_SIDE,
            "pool": "max;2x2x2;stride-2;first-14", "width": FEATURE_WIDTH,
            "order": FEATURE_ORDER, "normalisation": "none",
        },
        "sampling": {
            "positive": int(np.sum(samples.classes == 0)),
            "lumen_negative": int(np.sum(samples.classes == 1)),
            "shell_negative": int(np.sum(samples.classes == 2)),
            "shell_segmentation_voxels": 7,
            "seed": seed,
        },
        "df21": {
            "n_bins": 255, "max_layers": 20, "n_estimators": 2,
            "n_trees": 100, "n_tolerant_rounds": 2, "random_state": seed,
            "n_jobs": -1, "partial_mode": True,
        },
        "geometry": {
            "raw_shape_zyx": list(stack.shape),
            "raw_voxel_um": np.asarray(frame.raw_voxel).tolist(),
            "seg_dims_xyz": np.asarray(frame.seg_dims).tolist(),
            "seg_origin_um": np.asarray(frame.seg_origin).tolist(),
            "seg_spacing_um": np.asarray(frame.seg_spacing).tolist(),
            "bin_factor_xyz": np.asarray(frame.bin_factor).tolist(),
        },
        "sources": {
            "graph": _fingerprint_file(graph_path),
            "segmentation": _fingerprint_file(segmentation_path),
            "raw": _fingerprint_stack(stack),
        },
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics, "output": str(output)}


def _validate_geometry(graph, frame) -> None:
    if np.any(np.asarray(frame.raw_voxel) <= 0) or np.any(np.asarray(frame.seg_spacing) <= 0):
        raise ValueError("raw and segmentation spacing must be positive in x/y/z order")
    if not np.all(frame.bin_factor == 2):
        raise ValueError(f"expected a 2x2x2 segmentation binning, got {frame.bin_factor}")
    raw_shape_xyz = np.asarray(frame.raw_shape[::-1], dtype=np.int64)
    covered_hi = frame.raw_start + frame.bin_factor * frame.seg_dims - 1
    if np.any(frame.raw_start < 0) or np.any(covered_hi >= raw_shape_xyz):
        raise ValueError("segmentation crop does not fit inside the raw TIFF geometry")
    raw = frame.um_to_raw(graph.points)
    if not np.all((raw >= 0) & (raw <= np.asarray(frame.raw_shape) - 1)):
        raise ValueError("skeleton contains points outside the raw TIFF stack")
    if not np.any(valid_centres(np.rint(raw).astype(int), frame.raw_shape)):
        raise ValueError("no skeleton points support a full 15x15x15 patch")


def _fingerprint_file(path) -> dict:
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            stream.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
    return {
        "path": str(path.resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "edge_sha256": digest.hexdigest(),
    }


def _fingerprint_stack(stack) -> dict:
    files = stack.files
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {
        "path": str(stack.directory.resolve()), "files": len(files),
        "listing_sha256": digest.hexdigest(), "dtype": str(stack.dtype),
    }
