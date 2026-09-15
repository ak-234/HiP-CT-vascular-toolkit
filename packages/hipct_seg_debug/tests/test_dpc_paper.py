from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect.dpc import (
    DpcParams,
    DpcResult,
    _adf_pvalue,
    _cosine_term,
    _offsets,
    walk,
    validate,
)
from hipct_seg_debug.edit.reconnect.evaluation import parse_omega, reconnection_metrics
from hipct_seg_debug.edit.reconnect.probability import Roi


def test_two_level_neighbourhood_sets_are_disjoint_and_paper_sized():
    fine = _offsets(2, "fine")
    coarse = _offsets(2, "coarse")
    assert len(fine) == 26
    assert np.all(np.linalg.norm(fine, axis=1) < 2)
    assert np.all((np.linalg.norm(coarse, axis=1) >= 2)
                  & (np.linalg.norm(coarse, axis=1) <= 3))
    assert not set(map(tuple, fine)) & set(map(tuple, coarse))


def test_cosine_term_uses_the_less_than_or_equal_half_branch():
    unit = np.eye(3)
    active = _cosine_term(unit, [np.array([1.0, 0, 0]), np.array([0.0, 1, 0])])
    inactive = _cosine_term(unit, [np.array([1.0, 0, 0]), np.array([1.0, 0, 0])])
    assert np.array_equal(active, [1.0, 1.0, 0.0])
    assert np.array_equal(inactive, np.zeros(3))


def test_distance_score_is_not_minmax_normalised():
    roi = Roi(np.zeros((30, 30, 30)), np.zeros(3), np.ones(3))

    def constant(points):
        return np.full(len(points), 0.5)

    result = walk(
        roi, constant, roi.to_world([[15, 15, 10]])[0], roi.to_world([[15, 15, 16]])[0],
        start_direction=np.array([1.0, 0, 0]),
        params=DpcParams(max_steps=1, neighbourhood_policy="two-level"),
    )
    assert result.distance_scores[0] < -1.0
    assert result.probability_scores[0] == 0.0


def test_paper_reconnection_metrics_are_exact():
    metrics = reconnection_metrics(["TP_b", "TP_s", "TN_b", "FP_b", "FP_s", "FN_b"])
    assert metrics["RecAcc"] == pytest.approx(3 / 6)
    assert metrics["RecSen"] == pytest.approx(2 / 3)
    assert metrics["RecSpe"] == pytest.approx(1 / 3)


def test_omega_parser_is_inclusive():
    assert parse_omega("0:7") == list(range(8))
    assert parse_omega("1,5,7") == [1, 5, 7]


def test_adf_distinguishes_stationary_noise_from_a_random_walk():
    rng = np.random.default_rng(12)
    stationary = rng.normal(size=200)
    random_walk = np.cumsum(rng.normal(size=200))
    assert _adf_pvalue(stationary) < 0.05
    assert _adf_pvalue(random_walk) > 0.05


def test_validation_records_paper_and_hipct_diagnostics_before_rejecting():
    rng = np.random.default_rng(13)
    probability = rng.normal(0.02, 0.002, 30)
    result = DpcResult(
        np.zeros((30, 3)), probability, True, 29,
        probability_sequence=probability,
        grayscale_sequence=rng.normal(size=30),
    )
    ok, _reason, stats = validate(result, np.full(30, 0.8))
    assert not ok
    assert "probability_adf_p" in stats and "grayscale_adf_p" in stats
    assert stats["paper_validation"]["accepted"] in (True, False)
    assert not stats["hipct_safeguards"]["accepted"]
