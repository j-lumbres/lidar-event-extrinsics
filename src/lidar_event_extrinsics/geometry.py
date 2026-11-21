"""Rigid transforms as 4x4 matrices.

A transform named T_camera_from_lidar takes a point written in the LiDAR frame and
returns the same point written in the camera frame:

    p_camera = R @ p_lidar + t

With names written this way, a chain is correct when the inner frames match, as in
T_event_from_lidar = T_event_from_regular @ T_regular_from_lidar.
"""

import numpy as np

ROTATION_TOLERANCE = 1e-5


def check_rigid(T, name="transform"):
    """Raise ValueError unless T is a 4x4 rigid transform.

    Calibration files store rotations with a few printed digits, so the check allows
    a small tolerance. A flipped axis (det = -1) is a real error and is always caught.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} must have last row [0, 0, 0, 1]")

    rotation = T[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=ROTATION_TOLERANCE):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=ROTATION_TOLERANCE):
        raise ValueError(f"{name} rotation has det != +1")
    return T


def make_transform(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def invert(T):
    # From p_b = R p_a + t it follows that p_a = R^T p_b - R^T t.
    rotation = T[:3, :3]
    translation = T[:3, 3]
    rotation_inverse = rotation.T
    translation_inverse = -rotation_inverse @ translation
    return make_transform(rotation_inverse, translation_inverse)


def apply(T, points):
    """Transform an (..., 3) array of points."""
    points = np.asarray(points, dtype=np.float64)
    rotation = T[:3, :3]
    translation = T[:3, 3]
    return points @ rotation.T + translation


def rotation_from_quaternion(qx, qy, qz, qw):
    """Rotation matrix from a unit quaternion in (x, y, z, w) order."""
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def rotation_from_axis_angle(rotation_vector):
    """Rodrigues' formula: rotate by |v| radians about the axis v / |v|."""
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        return np.eye(3)

    axis = rotation_vector / angle
    axis_cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * axis_cross + (1 - np.cos(angle)) * axis_cross @ axis_cross


def rotation_vector_from_matrix(rotation):
    """Inverse of Rodrigues' formula, for rotations below 180 degrees.

    The angle comes from the trace, and the axis from the antisymmetric part
    R - R^T, which equals 2 sin(angle) times the cross-product matrix of the axis.
    """
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-12:
        return np.zeros(3)

    axis_times_sin = 0.5 * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    axis = axis_times_sin / np.sin(angle)
    return angle * axis


def rotation_angle_deg(rotation):
    # The angle of a rotation follows from its trace: trace(R) = 1 + 2 cos(angle).
    cos_angle = (np.trace(rotation) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def transform_difference(T_a, T_b):
    """Rotation angle in degrees and translation distance in meters between two transforms."""
    rotation_between = T_a[:3, :3].T @ T_b[:3, :3]
    angle_deg = rotation_angle_deg(rotation_between)
    distance_m = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    return angle_deg, distance_m
