"""Where the optional real-dataset fixtures come from.

A handful of tests cross-check this package against a genuine Amira export rather
than a synthetic one. No checkout ships such a file -- they are gigabytes, and they
are somebody's unpublished scan -- so the paths are named by the environment and the
tests skip when nothing is set.

They used to be absolute paths written into the test modules. That made a fresh
install on the authoring machine silently measure against one particular scan, and a
fresh install anywhere else skip with a message naming a drive it could not explain.

Set whichever you have; each is a full path to the file:

    HIPCT_GRAPH       the spatial graph (.am)
    HIPCT_SEG         the binary label lattice (.am)
    HIPCT_GRAPH_ALT   a second spatial graph, for the cross-dataset comparisons

These are the same variables the CLI reads for `--graph` and `--seg`.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Stands in for an unset variable. Chosen so `.is_file()` is False rather than
#: raising, and so a skip message says which variable to set.
_UNSET = "<unset>"


def from_env(name: str) -> Path:
    """The path named by `name`, or one that cannot exist."""
    value = os.environ.get(name)
    return Path(value) if value else Path(f"{_UNSET}:{name}")


def reason(name: str, what: str) -> str:
    """Why a real-data test is being skipped, and what to do about it."""
    return f"no {what}: set {name} to one to run this test"


REAL_AM = from_env("HIPCT_GRAPH")
REAL_AM_ALT = from_env("HIPCT_GRAPH_ALT")
REAL_SEG = from_env("HIPCT_SEG")

GRAPH_REASON = reason("HIPCT_GRAPH", "spatial graph")
GRAPH_ALT_REASON = reason("HIPCT_GRAPH_ALT", "second spatial graph")
SEG_REASON = reason("HIPCT_SEG", "label lattice")
