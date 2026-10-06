"""Bounded interpolation of unsupported radii along primary DFS paths."""
from __future__ import annotations

import numpy as np

from .centreline_refine import arclength
from .dfs_paths import path_samples, root_paths, validate_path_plan


def apply_path_profiles(graph, raw, trusted, derived, reasons, report, *, roots,
                        spacing_um, transition_radii, reason_code, path_plan=None):
    from .radius_profile import interpolate_supported
    processed = set()
    paths = (root_paths(graph, roots) if path_plan is None
             else validate_path_plan(graph, path_plan, roots))
    for path in paths:
        # A later daughter starts at its own attachment record. It never borrows
        # a trusted parent radius as a daughter anchor.
        suffix = [sid for sid in path['segments'] if sid not in processed]
        processed.update(suffix)
        if not suffix:
            continue
        coords, refs = path_samples(graph, path, suffix)
        s = arclength(coords)
        values = np.array([raw[sid][i] for sid, i in refs])
        good = np.array([trusted[sid][i] for sid, i in refs])
        anchors = np.flatnonzero(good)
        row = dict(root=path['root'], terminal=path['terminal'], segments=suffix,
                   interpolated_points=0, rejected_spans=[])
        report.paths.append(row)
        # Duplicate joint records can contain incompatible trusted measurements.
        # Do not choose whichever happens to occur first as the path anchor.
        conflicting = [(int(a), int(b)) for a, b in zip(anchors[:-1], anchors[1:])
                       if s[a] == s[b] and not np.isclose(values[a], values[b], rtol=1e-8)]
        boundaries = [0]+[b for _, b in conflicting]+[len(s)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            part = slice(start, stop)
            fitted, use = interpolate_supported(s[part], values[part], good[part])
            local_anchors = np.flatnonzero(good[part])+start
            for left, right in zip(local_anchors[:-1], local_anchors[1:]):
                if right-left <= 1 or s[right] == s[left]:
                    continue
                limit = max(8*spacing_um, transition_radii*min(values[left], values[right]))
                if s[right]-s[left] > limit:
                    use[left-start+1:right-start] = False
                    row['rejected_spans'].append(dict(
                        from_point=list(refs[left]), to_point=list(refs[right]),
                        length_um=float(s[right]-s[left]), limit_um=float(limit),
                        reason='anchors_too_far_apart'))
            for k in np.flatnonzero(use):
                sid, i = refs[start+k]
                derived[sid][i] = fitted[k]
                reasons[sid][i] = reason_code
                row['interpolated_points'] += 1
