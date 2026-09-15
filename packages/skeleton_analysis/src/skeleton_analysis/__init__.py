"""skeleton_analysis: skeletonisation fixes and topological analysis of
vascular spatial graphs (Amira/Avizo ``HxSpatialGraph``).

Python port of the MATLAB ``Skeleton_analysis`` package.
"""

from skeleton_analysis.io.amira import SpatialGraph, read_amira, write_amira

__version__ = "0.1.0"

__all__ = ["SpatialGraph", "read_amira", "write_amira", "__version__"]
