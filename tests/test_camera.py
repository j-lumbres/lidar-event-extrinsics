import numpy as np
import pytest

from lidar_event_extrinsics.camera import Camera

cv2 = pytest.importorskip("cv2")

# The VECtor left event camera.
CAMERA = Camera(
    fx=327.32749,
    fy=327.46184,
    cx=304.97749,
    cy=235.37621,
    width=640,
    height=480,
    distortion=np.array([-0.031982, 0.041966, -0.000507, -0.001031, 0.0]),
)


def test_projection_matches_opencv():
    generator = np.random.default_rng(0)
    points = np.column_stack(
        [
            generator.uniform(-3, 3, 200),
            generator.uniform(-2, 2, 200),
            generator.uniform(2, 20, 200),
        ]
    )
    uv, _, _ = CAMERA.project(points)

    camera_matrix = np.array([[CAMERA.fx, 0, CAMERA.cx], [0, CAMERA.fy, CAMERA.cy], [0, 0, 1]])
    uv_opencv, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), camera_matrix, CAMERA.distortion
    )
    np.testing.assert_allclose(uv, uv_opencv[:, 0], atol=1e-8)


def test_points_behind_or_outside_are_not_visible():
    points = np.array(
        [
            [0.0, 0.0, 5.0],  # straight ahead
            [0.0, 0.0, -5.0],  # behind
            [50.0, 0.0, 5.0],  # far to the side
            [0.0, 0.0, 0.1],  # closer than the minimum depth
        ]
    )
    _, depth, visible = CAMERA.project(points)
    assert visible.tolist() == [True, False, False, False]
    assert depth[0] == 5.0
