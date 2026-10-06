"""Debug tool for HiP-CT vessel segmentation / skeletonisation / surface pipelines.

Overlays the raw image stack, the Amira segmentation, the skeleton spatial graph
and the reconstructed lumen surface in a single physical coordinate frame, so that
collapsed-vessel failures can be identified by eye against the greyscale ground
truth.
"""

__version__ = "1.0.0"

# Configure before edit's imports load optional dependencies. Worker processes
# inherit the environment switch, even when their argv no longer has CLI flags.
import os as _os
import sys as _sys

if ('--quiet-dependency-warnings' in _sys.argv
        or _os.environ.get('HIPCT_QUIET_DEPENDENCY_WARNINGS') == '1'):
    from ._dependency_warnings import quiet_dependency_warnings as _quiet_warnings
    _quiet_warnings()
