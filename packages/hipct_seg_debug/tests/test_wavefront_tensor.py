"""The image intensity tensor on phantoms whose orientation is known."""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect.geodesic import components, corridor, cost
from hipct_seg_debug.edit.reconnect.wavefront import prepare, tensor

from .conftest_geodesic import (
    CY,
    CZ,
    LUMEN,
    WALL,
    FakeStack,
    broken_graph,
    cylinder,
    greyscale,
    make_frame,
    mask_source,
    ribbon,
)

SHAPE = (36, 36, 60)


def _flat_ribbon_image():
    mask = ribbon(SHAPE, 7, 2, 4, 56)
    return mask, greyscale(mask, noise=2.0, blur=0.8)


def test_orientation_recovers_axis_and_collapse_normal():
    mask, image = _flat_ribbon_image()
    axis, normal, coherence, axis_conf = tensor.orientation(image, sigma_vox=1.5,
                                                            rho_vox=5.0)
    inside = mask.astype(bool)
    inside[:, :, :12] = inside[:, :, 48:] = False  # away from the cut ends
    assert float(np.median(np.abs(axis[inside][:, 2]))) > 0.95  # nu_1 along x
    assert float(np.median(np.abs(normal[inside][:, 0]))) > 0.95  # nu_3 along z
    assert float(np.median(coherence[inside])) > 0.5
    assert float(np.median(axis_conf[inside])) > 0.3


def test_planarity_is_high_in_a_dark_slit_and_low_in_background():
    mask, image = _flat_ribbon_image()
    plan, normal = tensor.planarity(image, sigmas=[1.4, 2.0, 2.8], dark_lumen=True)
    # The response peaks on the mid-plane of the slit and falls off toward its
    # faces, so the test reads the mid-plane rather than the whole interior.
    midplane = np.zeros(SHAPE, dtype=bool)
    midplane[CZ, CY - 5:CY + 6, 12:48] = True
    outside = np.zeros(SHAPE, dtype=bool)
    outside[4:8, 4:8, 20:40] = True
    assert float(np.median(plan[midplane])) > 0.45
    assert float(np.median(plan[outside])) < 0.1
    assert float(np.median(np.abs(normal[midplane][:, 0]))) > 0.9


def test_round_tube_is_not_planar():
    mask = cylinder(SHAPE, 4, 4, 56)
    image = greyscale(mask, noise=2.0)
    plan, _normal = tensor.planarity(image, sigmas=[2.0, 2.8, 4.0], dark_lumen=True)
    core = np.zeros(SHAPE, dtype=bool)
    core[CZ - 1:CZ + 2, CY - 1:CY + 2, 16:44] = True
    assert float(np.median(plan[core])) < 0.3


def _field_for(mask, image, radius_um=20.0, graph_gap=((4, 28), (32, 56))):
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, *graph_gap)
    index = components.build(mask_source(mask))
    profiles = prepare.profile_ends(graph, frame, index)
    box = corridor.for_candidate(frame, index, np.array([[200.0, 200.0, 200.0],
                                                         [400.0, 200.0, 200.0]]),
                                 radius_um=radius_um, stack=FakeStack(image),
                                 pad_factor=4.0)
    scalar = cost.build(box, index, set(range(1, index.n + 1)), radius_um=radius_um)
    return tensor.build(scalar, box, radius_um=radius_um,
                        profiles=list(profiles.values())), box


def test_metric_is_dear_through_the_collapse_normal_and_cheap_along_the_axis():
    mask = ribbon(SHAPE, 7, 2, 4, 56)
    mask[:, :, 28:32] = 0
    image = greyscale(ribbon(SHAPE, 7, 2, 4, 56), noise=2.0)
    field, box = _field_for(mask, image)
    local = box.to_global(np.array([[300.0, 200.0, 200.0]]))[0] - box.lo_zyx
    along = field.step_cost(local, (0, 0, 1))
    across = field.step_cost(local, (0, 1, 0))
    through = field.step_cost(local, (1, 0, 0))
    assert np.isclose(along, 1.0, atol=0.15)
    assert through > across > along
    assert through / along <= np.sqrt(field.params.max_ratio) + 1e-6
    ratio = tensor.anisotropy_ratio(field.aniso[tuple(local)])
    assert 2.0 < ratio <= field.params.max_ratio + 1e-6


def test_metric_is_identity_where_there_is_no_lumen():
    mask, image = _flat_ribbon_image()
    field, box = _field_for(mask, image)
    corner = np.array([1, 1, box.shape[2] // 2])
    assert np.allclose(field.matrices()[tuple(corner)], np.eye(3), atol=0.05)


def test_a_bright_planar_artefact_gets_no_anisotropy():
    """A ring artefact is planar too, but it is not lumen, and must stay isotropic."""
    mask, image = _flat_ribbon_image()
    image[6, :, :] = WALL + 40.0  # a bright plane across the whole field
    field, box = _field_for(mask, image)
    plane = field.matrices()[6 - box.lo_zyx[0], 4:8, 10:50]
    assert np.allclose(plane, np.eye(3), atol=0.1)


def test_scales_follow_the_measured_ellipse():
    mask, image = _flat_ribbon_image()
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (4, 28), (32, 56))
    index = components.build(mask_source(mask))
    profiles = list(prepare.profile_ends(graph, frame, index).values())
    sigma, rho = tensor.scales_for(profiles, 20.0, 10.0)
    assert sigma < rho
    assert 0.7 <= sigma <= 4.0 and 1.5 <= rho <= 8.0
    assert tensor.scales_for([], 20.0, 10.0) == (2.0, 2.0)


def test_isotropic_field_is_the_scalar_problem():
    class Scalar:
        cost = np.ones((4, 5, 6))
        spacing_zyx = np.array([10.0] * 3)

    field = tensor.isotropic(Scalar())
    assert np.allclose(field.matrices(), np.eye(3))
    assert field.step_cost((1, 1, 1), (1, 1, 1)) == pytest.approx(1.0)
    assert np.all(field.lumen == 0.5)


def test_dark_and_bright_lumen_conventions_agree_on_planarity():
    mask, image = _flat_ribbon_image()
    inside = mask.astype(bool)
    inside[:, :, :12] = inside[:, :, 48:] = False
    dark, _ = tensor.planarity(image, sigmas=[2.0], dark_lumen=True)
    bright, _ = tensor.planarity(WALL + LUMEN - image, sigmas=[2.0], dark_lumen=False)
    # Not bit-identical: the Gaussian derivatives pad with zero, and a constant
    # offset changes what that padding means at the boundary.
    assert np.allclose(dark[inside], bright[inside], rtol=0.03, atol=1e-3)
