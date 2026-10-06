"""Invalidate radius trust without changing numeric radius placeholders."""
import numpy as np


def invalidate_radii(graph, segments):
    from .radius_perimeter import FILLED, UNMEASURABLE, INPUT_FALLBACK
    pids = {pid for sid in segments for pid in graph.segment(sid)['point_ids']}
    for name, value in (('radius_source', FILLED), ('radius_reject_reason', UNMEASURABLE),
                        ('radius_resolution_mode', INPUT_FALLBACK)):
        graph.triple.point_attrs.setdefault(name, {}).update(dict.fromkeys(pids, value))
        graph.triple.point_attr_dtypes[name] = np.dtype(np.int64)
