from __future__ import annotations

import json

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect.cfc import (
    FEATURE_ORDER,
    FEATURE_WIDTH,
    CfcProbability,
    features_from_volume,
    patch_feature,
    sample_training_voxels,
)
from hipct_seg_debug.edit.reconnect.probability import Roi


def test_patch_feature_is_exactly_pooled_large_then_centred_small():
    patch = np.arange(15**3, dtype=np.uint16).reshape(15, 15, 15)
    feature = patch_feature(patch)
    pooled = patch[:14, :14, :14].reshape(7, 2, 7, 2, 7, 2).max((1, 3, 5))
    assert feature.shape == (FEATURE_WIDTH,)
    assert np.array_equal(feature[:343], pooled.ravel(order="C"))
    assert np.array_equal(feature[343:], patch[4:11, 4:11, 4:11].ravel(order="C"))


def test_patch_extraction_rejects_boundary_centres():
    volume = np.zeros((20, 20, 20), dtype=np.uint16)
    with pytest.raises(ValueError, match="within 7 voxels"):
        features_from_volume(volume, [[6, 10, 10]])


class FakeModel:
    def __init__(self):
        self.calls = 0

    def predict_proba(self, x):
        self.calls += 1
        p = np.full(len(x), 0.75)
        return np.column_stack([1 - p, p])


def _artifact(tmp_path, spacing=(1.0, 1.0, 1.0)):
    manifest = {
        "artifact_version": 1,
        "feature": {"width": FEATURE_WIDTH, "order": FEATURE_ORDER},
        "geometry": {"raw_voxel_um": list(spacing), "raw_shape_zyx": [40, 40, 40]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_cfc_probability_caches_repeated_raw_voxels(tmp_path):
    roi = Roi(np.ones((25, 25, 25), dtype=np.uint16), np.zeros(3), np.ones(3))
    model = FakeModel()
    probability = CfcProbability(roi, _artifact(tmp_path), model=model)
    point = roi.to_world([[12, 12, 12]])
    assert probability(point)[0] == pytest.approx(0.75)
    assert probability(point)[0] == pytest.approx(0.75)
    assert model.calls == 1


def test_cfc_probability_can_precompute_a_review_roi(tmp_path):
    roi = Roi(np.ones((16, 16, 16), dtype=np.uint16), np.zeros(3), np.ones(3))
    model = FakeModel()
    probability = CfcProbability(roi, _artifact(tmp_path), model=model)
    assert probability.precompute(batch_size=2) == 8
    calls = model.calls
    assert probability(roi.to_world([[7, 7, 7]]))[0] == pytest.approx(0.75)
    assert model.calls == calls


def test_cfc_probability_rejects_scan_spacing_mismatch(tmp_path):
    roi = Roi(np.ones((25, 25, 25)), np.zeros(3), np.ones(3) * 2)
    with pytest.raises(ValueError, match="spacing mismatch"):
        CfcProbability(roi, _artifact(tmp_path), model=FakeModel())


def test_sampling_has_the_paper_n_2n_2n_class_ratio():
    from hipct_seg_debug.frame import WorldFrame

    class Graph:
        n_vertex = 4
        connectivity = np.array([[0, 1], [2, 3]])
        n_edge_points = np.array([2, 2])
        points = np.array([
            [15.0, 15.0, 15.0], [17.0, 15.0, 15.0],
            [15.0, 17.0, 17.0], [17.0, 17.0, 17.0],
        ])

        @property
        def n_edge(self):
            return len(self.connectivity)

    class Labels:
        def __init__(self):
            self.volume = np.zeros((25, 25, 25), dtype=np.uint8)
            self.volume[4:21, 4:21, 4:21] = 1
            self.nz, self.ny, self.nx = self.volume.shape

        def slice_z(self, z):
            return self.volume[z]

    frame = WorldFrame(
        raw_shape=(50, 50, 50), raw_voxel=np.ones(3),
        seg_dims=np.array([25, 25, 25]), seg_origin=np.full(3, 0.5),
        seg_spacing=np.full(3, 2.0), nominal_voxel=np.ones(3),
    )
    samples = sample_training_voxels(Graph(), Labels(), frame, seed=4, chunk_slices=8)
    counts = np.bincount(samples.classes, minlength=3)
    assert counts.tolist() == [4, 8, 8]
    assert samples.labels.sum() == 4
