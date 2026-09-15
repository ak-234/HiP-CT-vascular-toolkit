"""Point-query adapter presenting the legacy field to grid-free extractors.

The adaptive backends consume a field oracle — ``evaluate``, ``bounds_*``,
``minimum_relevant_radius`` and a pruning rule — while the legacy field was only
ever defined as a narrow-band grid evaluator. Without this adapter the
"legacy field + adaptive extraction" cell of the field-versus-extractor ablation
cannot be built at all, and a regression seen on the graph field cannot be
attributed to the field rather than to the extractor.

Two design choices carry the scientific weight:

*Sizing is shared, not reimplemented.* Bounds and the local target-size law are
delegated to a :class:`~coronary_sdf.implicit_field.GraphImplicitField` built
from the same capsules, so the octree hierarchy is identical between the two
field ablations by construction and the only difference is the field value.

*Pruning uses a geometric certificate, not a Lipschitz guess.* The legacy field
is not Lipschitz with a small constant — the junction blend ramps over
``owner_radius * SMIN_PROXIMITY_BLEND_FACTOR``, so the bound diverges as the
radius shrinks. Feeding an under-estimate to the octree's ``|f| <= L * halfdiag``
test silently deletes surface. Setting the bound to infinity is *not* the safe
fallback either: it also multiplies the candidate-search radius inside
``minimum_relevant_radius``, which on real data (minimum radius 1e-6 mm) would
refine the entire root box to maximum depth. Instead this adapter exposes an
explicit ``certainly_outside_band`` predicate backed by a pure hard-union bound
field, and refuses to prune interior cells at all.
"""

from __future__ import annotations

import numpy as np

from .capsules import CapsuleArrays
from .config import runtime_config as config
from .implicit_field import GraphImplicitField
from .sdf_field import BifurcationSet, TerminalSet, evaluate_sdf_points

PRUNE_MODES = ("geometric", "none")


class LegacyPointField:
    """Expose :func:`sdf_field.evaluate_sdf_points` as a field oracle."""

    def __init__(
        self,
        capsules: CapsuleArrays,
        *,
        length_scale: float,
        sizing_field: GraphImplicitField,
        cap_is_junction: np.ndarray,
        adj_matrix: np.ndarray,
        bif: BifurcationSet,
        term: TerminalSet,
        shared_node_pos: np.ndarray | None = None,
        shared_node_has: np.ndarray | None = None,
        seg_end_pos: np.ndarray | None = None,
        seg_end_tan: np.ndarray | None = None,
        seg_end_tan_ok: np.ndarray | None = None,
        is_parent: np.ndarray | None = None,
        is_child: np.ndarray | None = None,
        is_sibling: np.ndarray | None = None,
        bif_seg_incident: np.ndarray | None = None,
        prune_mode: str = "geometric",
    ) -> None:
        if prune_mode not in PRUNE_MODES:
            raise ValueError(f"prune_mode must be one of {PRUNE_MODES}")
        if not np.isfinite(length_scale) or length_scale <= 0.0:
            raise ValueError("length_scale must be a positive, finite length in mm")
        if config.SDF_CARVE_NON_ADJACENT:
            # The carve subtracts an unbounded amount near a non-adjacent rival,
            # which the outside certificate below does not model.
            raise ValueError(
                "the legacy point oracle does not support SDF_CARVE_NON_ADJACENT; "
                "the geometric prune certificate would be unsound"
            )

        self.length_scale = float(length_scale)
        self.prune_mode = prune_mode
        self.sizing_field = sizing_field
        self._kernel_kwargs = dict(
            capsules=capsules,
            cap_is_junction=cap_is_junction,
            adj_matrix=adj_matrix,
            bif=bif,
            term=term,
            shared_node_pos=shared_node_pos,
            shared_node_has=shared_node_has,
            seg_end_pos=seg_end_pos,
            seg_end_tan=seg_end_tan,
            seg_end_tan_ok=seg_end_tan_ok,
            is_parent=is_parent,
            is_child=is_child,
            is_sibling=is_sibling,
            bif_seg_incident=bif_seg_incident,
        )

        # Geometry mirrored from the shared sizing oracle so the two field
        # ablations build byte-identical hierarchies.
        self.bounds_min = sizing_field.bounds_min
        self.bounds_max = sizing_field.bounds_max
        self.max_radii = sizing_field.max_radii
        self.minimum_positive_radius = sizing_field.minimum_positive_radius
        self.lipschitz_bound = sizing_field.lipschitz_bound

        # Pure hard union of radial capsules: exactly the legacy field before
        # any blending, capping or clamping, and therefore a lower bound on it
        # up to the bounded blend depression.
        self._bound_field = GraphImplicitField(
            capsules, (), primitive_method="radial", clip_bifurcation_caps=False
        )
        self._bound_lipschitz = float(np.max(self._bound_field.primitive_lipschitz))
        self._blend_depth = max(
            float(config.BLEND_BULGE_CAP_MM),
            float(np.max(self.max_radii)) * float(config.BLEND_BULGE_CAP_RADIUS_FACTOR),
        )

    def minimum_relevant_radius(self, point: np.ndarray, cell_radius: float) -> float:
        """Delegate the target-size law to the shared sizing oracle.

        Note this uses the *graph* field's Lipschitz bound for its internal
        candidate search. That is intentional: it is what makes the octree
        identical between the legacy and graph cells.
        """

        return self.sizing_field.minimum_relevant_radius(point, cell_radius)

    def evaluate(
        self, points: np.ndarray, *, with_gradient: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(value, owner, radius, gradient)`` for an ``(N,3)`` array."""

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        values = evaluate_sdf_points(
            points, length_scale=self.length_scale, **self._kernel_kwargs
        )
        # Owner and local radius are geometric quantities used only for sizing
        # and telemetry, so they come from the shared oracle rather than being
        # re-derived; the *value* is what differs between the two fields.
        _v, owners, radii, _g = self.sizing_field.evaluate(points)
        gradients = None
        if with_gradient:
            eps = 1e-4 * self.length_scale
            gradients = np.empty((len(points), 3), dtype=np.float64)
            for axis in range(3):
                offset = np.zeros(3)
                offset[axis] = eps
                ahead = evaluate_sdf_points(
                    points + offset,
                    length_scale=self.length_scale,
                    **self._kernel_kwargs,
                )
                behind = evaluate_sdf_points(
                    points - offset,
                    length_scale=self.length_scale,
                    **self._kernel_kwargs,
                )
                gradients[:, axis] = (ahead - behind) / (2.0 * eps)
        return values.astype(np.float64), owners, radii, gradients

    def certainly_outside_band(
        self, centers: np.ndarray, half_diagonal: float
    ) -> np.ndarray:
        """Cells provably containing no part of the zero set.

        Sound because the legacy value is at least the hard-union distance minus
        the bounded blend depression, and the flat-cap and terminal clamps only
        raise it. Interior cells are never certified: those clamps can push an
        interior point exterior anywhere along a bifurcation-incident capsule,
        so there is no cheap inside certificate.
        """

        centers = np.asarray(centers, dtype=np.float64)
        if self.prune_mode == "none" or not len(centers):
            return np.zeros(len(centers), dtype=bool)
        hard, _owners, _radii, _g = self._bound_field.evaluate(centers)
        margin = self._bound_lipschitz * float(half_diagonal) + self._blend_depth
        return np.asarray(hard) > margin


def build_legacy_point_field(
    capsules: CapsuleArrays,
    sizing_field: GraphImplicitField,
    *,
    length_scale: float,
    prune_mode: str = "geometric",
    **topology: object,
) -> LegacyPointField:
    """Convenience constructor mirroring ``build_graph_implicit_field``."""

    return LegacyPointField(
        capsules,
        length_scale=length_scale,
        sizing_field=sizing_field,
        prune_mode=prune_mode,
        **topology,  # type: ignore[arg-type]
    )


__all__ = ["PRUNE_MODES", "LegacyPointField", "build_legacy_point_field"]
