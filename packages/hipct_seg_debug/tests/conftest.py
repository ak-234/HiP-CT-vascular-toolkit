"""Shared pytest configuration for the edit package's tests."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: needs the real dataset and runs a full pipeline (minutes, not seconds)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run tests that drive a full coronary_sdf pipeline run",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow (drives a full pipeline run)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
