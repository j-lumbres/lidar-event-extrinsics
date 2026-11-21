import numpy as np
import pytest

from lidar_event_extrinsics.geometry import (
    apply,
    check_rigid,
    invert,
    make_transform,
    rotation_angle_deg,
    rotation_from_axis_angle,
    rotation_from_quaternion,
    rotation_vector_from_matrix,
    transform_difference,
)


def random_transform(seed):
    generator = np.random.default_rng(seed)
    rotation = rotation_from_axis_angle(generator.normal(size=3))
    translation = generator.normal(size=3)
    return make_transform(rotation, translation)


def test_invert_undoes_transform():
    T = random_transform(0)
    points = np.random.default_rng(1).normal(size=(20, 3))
    np.testing.assert_allclose(apply(invert(T), apply(T, points)), points, atol=1e-12)


def test_chain_matches_applying_one_after_another():
    T_a_from_b = random_transform(2)
    T_b_from_c = random_transform(3)
    points_c = np.random.default_rng(4).normal(size=(20, 3))

    one_after_another = apply(T_a_from_b, apply(T_b_from_c, points_c))
    chained = apply(T_a_from_b @ T_b_from_c, points_c)
    np.testing.assert_allclose(chained, one_after_another, atol=1e-12)


def test_check_rigid_rejects_a_reflection():
    T = np.eye(4)
    T[0, 0] = -1.0
    with pytest.raises(ValueError, match="det"):
        check_rigid(T)


def test_quaternion_of_90_degrees_about_z():
    half = np.sqrt(0.5)
    rotation = rotation_from_quaternion(0.0, 0.0, half, half)
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_axis_angle_and_angle_round_trip():
    rotation = rotation_from_axis_angle([0.0, np.radians(2.5), 0.0])
    assert rotation_angle_deg(rotation) == pytest.approx(2.5)


def test_transform_difference():
    T_a = np.eye(4)
    T_b = make_transform(rotation_from_axis_angle([np.radians(1.0), 0.0, 0.0]), [0.03, 0.04, 0.0])
    angle_deg, distance_m = transform_difference(T_a, T_b)
    assert angle_deg == pytest.approx(1.0)
    assert distance_m == pytest.approx(0.05)


def test_rotation_vector_round_trip():
    rotation_vector = np.array([0.1, -0.4, 0.25])
    rotation = rotation_from_axis_angle(rotation_vector)
    np.testing.assert_allclose(rotation_vector_from_matrix(rotation), rotation_vector, atol=1e-12)
