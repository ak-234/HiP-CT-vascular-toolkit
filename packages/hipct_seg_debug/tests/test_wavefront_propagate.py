"""The wave-propagation solver on fields whose answer is known in closed form."""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect.geodesic import astar
from hipct_seg_debug.edit.reconnect.wavefront import propagate as prop
from hipct_seg_debug.edit.reconnect.wavefront import tensor

SPACING = 10.0


class _Scalar:
    """The minimum a :class:`TensorField` needs from its scalar half."""

    def __init__(self, cost, spacing=SPACING):
        self.cost = np.asarray(cost, dtype=np.float64)
        self.spacing_zyx = np.array([spacing] * 3)
        self.support = np.ones_like(self.cost, dtype=np.float32)
        self.lo_zyx = np.zeros(3, dtype=np.int64)

    def to_local(self, zyx):
        return np.asarray(zyx, dtype=np.int64) - self.lo_zyx


def _unit_field(shape=(30, 30, 30)):
    return tensor.isotropic(_Scalar(np.ones(shape)))


def _euclid(shape, centre):
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    return np.linalg.norm((grid - np.asarray(centre)) * SPACING, axis=-1)


def test_offsets_are_primitive_and_symmetric():
    o26, o98 = prop.offsets(26), prop.offsets(98)
    assert len(o26) == 26 and len(o98) == 98
    for o in (o26, o98):
        assert set(map(tuple, o)) == set(map(tuple, -o))
    assert (2, 2, 0) not in set(map(tuple, o98))  # a multiple of (1, 1, 0)


@pytest.mark.parametrize("stencil, tolerance", [(26, 0.14), (98, 0.06)])
def test_isotropic_arrival_approximates_euclidean_distance(stencil, tolerance):
    field = _unit_field()
    front = prop.propagate(field, [(15, 15, 15)], engine="lattice", stencil=stencil)
    truth = _euclid(field.cost.shape, (15, 15, 15))
    far = truth > 8 * SPACING
    ratio = front.arrival[far] / truth[far]
    assert np.all(ratio >= 1.0 - 1e-9)  # a lattice path is never shorter
    assert float(np.max(ratio) - 1.0) < tolerance
    assert front.expanded == field.cost.size
    assert np.all(front.origin[np.isfinite(front.arrival)] == 0)


def test_length_map_is_physical_path_length():
    field = _unit_field()
    front = prop.propagate(field, [(15, 15, 15)], engine="lattice")
    path = front.backtrack((3, 20, 27))
    assert tuple(path[0]) == (15, 15, 15) and tuple(path[-1]) == (3, 20, 27)
    steps = np.linalg.norm(np.diff(path, axis=0) * SPACING, axis=1)
    assert np.isclose(steps.sum(), front.length[3, 20, 27])
    assert np.isclose(front.arrival[3, 20, 27], steps.sum())  # unit cost


def test_anisotropic_metric_prefers_the_cheap_axis():
    field = _unit_field()
    field.aniso[..., 0] = 25.0  # z is five times dearer than x
    field.aniso[..., 3] = 25.0  # and so is y
    front = prop.propagate(field, [(15, 15, 15)], engine="lattice")
    assert np.isclose(front.arrival[15, 15, 25], 100.0)
    assert np.isclose(front.arrival[25, 15, 15], 500.0)
    # A diagonal between the two costs what the metric says it costs.
    e = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    assert np.isclose(field.step_cost((15, 15, 15), e), np.sqrt(0.5 * 25 + 0.5))


def test_cone_makes_backward_propagation_dear_but_not_impossible():
    field = _unit_field()
    front = prop.propagate(field, [(15, 15, 15)], engine="lattice",
                           cone=((0, 0, 1), 15.0, 4.0))
    assert np.isclose(front.arrival[15, 15, 25], 100.0)
    assert front.arrival[15, 15, 5] > 5.0 * 100.0
    assert np.isfinite(front.arrival[15, 15, 5])


def test_stop_at_first_goal_settles_only_what_it_needs():
    field = _unit_field()
    front = prop.propagate(field, [(15, 15, 15)], goals_zyx=[(15, 15, 25)],
                           stop="first_goal", engine="lattice")
    assert len(front.reached) == 1
    assert tuple(front.reached[0][0]) == (15, 15, 25)
    assert front.expanded < field.cost.size / 3


def test_max_length_bounds_the_sweep():
    field = _unit_field()
    front = prop.propagate(field, [(15, 15, 15)], max_length_um=50.0, engine="lattice")
    settled = np.isfinite(front.arrival)
    assert settled.sum() < 2000
    assert float(np.max(front.length[settled])) < 50.0 + SPACING * np.sqrt(3.0) + 1e-6


def test_blocked_voxels_are_never_entered_and_seeds_carry_labels():
    cost = np.ones((20, 20, 20))
    cost[:, :, 10] = np.inf  # a wall across x
    cost[10, 10, 10] = 1.0  # with one hole in it
    field = tensor.isotropic(_Scalar(cost))
    front = prop.propagate(field, [(10, 10, 2), (10, 10, 17)], seed_labels=[1, 2],
                           engine="lattice")
    assert np.all(np.isinf(front.arrival[:, :, 10][cost[:, :, 10] == np.inf]))
    assert np.isfinite(front.arrival[10, 10, 10])
    assert front.origin[10, 10, 2] == 1 and front.origin[10, 10, 17] == 2
    assert front.origin[10, 10, 5] == 1 and front.origin[10, 10, 14] == 2


def test_allowed_mask_restricts_the_domain():
    field = _unit_field((12, 12, 12))
    allowed = np.zeros(field.cost.shape, dtype=bool)
    allowed[6, 6, :] = True
    front = prop.propagate(field, [(6, 6, 0)], allowed=allowed, engine="lattice")
    assert np.isfinite(front.arrival).sum() == 12


def test_matches_astar_on_a_scalar_field():
    rng = np.random.default_rng(3)
    cost = 0.2 + rng.random((14, 14, 22))
    field = tensor.isotropic(_Scalar(cost))
    goal = (7, 7, 20)
    front = prop.propagate(field, [(7, 7, 1)], goals_zyx=[goal], stop="first_goal",
                           engine="lattice")
    _path, expected, _n, reason = astar.search(
        cost, (7, 7, 1), np.array([goal]), spacing_zyx=field.spacing_zyx,
        orientation=False, base_cost=0.0,
    )
    assert reason == "found"
    assert np.isclose(front.arrival[goal], expected, rtol=1e-9)


def test_coarse_corridor_is_a_tube_around_the_coarse_route():
    field = _unit_field((24, 24, 60))
    allowed = prop.coarse_corridor(field, [(12, 12, 2)], [(12, 12, 57)], threshold=1000)
    assert allowed is not None and allowed.any()
    assert allowed[12, 12, 30]
    assert not allowed[2, 2, 30]
    assert allowed.sum() < field.cost.size / 2


def test_small_fields_skip_the_coarse_pass():
    field = _unit_field((8, 8, 8))
    assert prop.coarse_corridor(field, [(4, 4, 0)], [(4, 4, 7)]) is None


def test_engine_selection():
    assert prop.select_engine("lattice") == "lattice"
    with pytest.raises(ValueError):
        prop.select_engine("fmm")
    if not prop.agd_available():
        with pytest.raises(RuntimeError, match="agd"):
            prop.select_engine("agd")
        assert prop.select_engine("auto") == "lattice"


def test_front_from_values_reconstructs_parents_and_lengths():
    """The agd adapter's post-processing, on a value map the lattice produced."""
    field = _unit_field((10, 10, 10))
    reference = prop.propagate(field, [(5, 5, 5)], engine="lattice")
    front = prop._front_from_values(field, reference.arrival, [(5, 5, 5)], None,
                                    [(5, 5, 9)], prop.STOP_FIRST_GOAL, np.inf,
                                    engine="agd")
    path = front.backtrack((1, 8, 9))
    assert tuple(path[0]) == (5, 5, 5) and tuple(path[-1]) == (1, 8, 9)
    assert np.isclose(front.length[1, 8, 9], reference.length[1, 8, 9])
    assert len(front.reached) == 1


@pytest.mark.skipif(not prop.agd_available(), reason="agd is not installed")
def test_agd_engine_agrees_with_the_lattice():  # pragma: no cover - optional dep
    field = _unit_field((16, 16, 16))
    a = prop.propagate(field, [(8, 8, 8)], engine="agd")
    b = prop.propagate(field, [(8, 8, 8)], engine="lattice", stencil=98)
    far = _euclid(field.cost.shape, (8, 8, 8)) > 4 * SPACING
    assert np.allclose(a.arrival[far], b.arrival[far], rtol=0.08)
