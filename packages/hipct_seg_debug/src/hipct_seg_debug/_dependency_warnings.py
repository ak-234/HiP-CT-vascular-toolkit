"""Opt-in filtering of known, unrelated dependency import warnings."""
from __future__ import annotations

import os
import warnings


def quiet_dependency_warnings():
    # Spawned workers import the package afresh, before their initializers run.
    os.environ['HIPCT_QUIET_DEPENDENCY_WARNINGS'] = '1'
    warnings.filterwarnings(
        'ignore',
        message=r'^(TripleDES|Blowfish) has been moved to cryptography\.hazmat\.decrepit\.',
        category=Warning,
        module=r'^paramiko\.(pkey|transport)$',
    )
