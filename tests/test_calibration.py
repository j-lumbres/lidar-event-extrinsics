import numpy as np

from lidar_event_extrinsics.calibration import (
    SweepEdges,
    calibrate,
    search_extrinsic,
    total_score,
)
from lidar_event_extrinsics.camera import Camera
from lidar_event_extrinsics.edges import TimedEventMaps
from lidar_event_extrinsics.geometry import (
    apply,
    invert,
    make_transform,
    rotation_angle_deg,
    rotation_from_axis_angle,
)
from lidar_event_extrinsics.vector import Events

CAMERA = Camera(fx=100.0, fy=100.0, cx=80.0, cy=60.0, width=160, height=120, distortion=np.zeros(5))
# LiDAR x forward, y left, z up, turned into camera z forward, x right, y down.
LIDAR_TO_CAMERA_AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
T_TRUE = make_transform(LIDAR_TO_CAMERA_AXES, [0.05, -0.1, -0.08])


def make_scene_points():
    # Vertical poles and horizontal bars at several depths, so that every rotation
    # axis moves some edge across the image.
    points = []
    for forward, left in [(3.0, 1.0), (4.0, -0.8), (5.0, 0.2), (3.5, -1.5)]:
        for up in np.linspace(-0.8, 0.8, 40):
            points.append([forward, left, up])
    for forward, up in [(4.0, 0.6), (3.0, -0.5)]:
        for left in np.linspace(-1.2, 1.2, 60):
            points.append([forward, left, up])
    return np.array(points)


def make_sweep_edges(scene_points):
    # Events fire where the true extrinsic puts each scene point, all within 10 ms.
    uv, _, visible = CAMERA.project(apply(T_TRUE, scene_points))
    uv = uv[visible]
    n_repeats = 5
    events = Events(
        x=np.tile(np.round(uv[:, 0]).astype(int), n_repeats),
        y=np.tile(np.round(uv[:, 1]).astype(int), n_repeats),
        t_us=np.sort(np.random.default_rng(0).integers(0, 10_000, n_repeats * len(uv))),
        polarity=np.ones(n_repeats * len(uv), dtype=np.int8),
    )
    maps = TimedEventMaps(events, CAMERA.width, CAMERA.height, kind="idt", gamma=0.9)
    return SweepEdges(
        points=scene_points,
        times_us=np.full(len(scene_points), 5000, dtype=np.int64),
        weights=np.ones(len(scene_points)),
        maps=maps,
    )


def test_true_extrinsic_scores_highest():
    sweeps = [make_sweep_edges(make_scene_points())]
    rotated = make_transform(rotation_from_axis_angle(np.radians([2.0, 0.0, 0.0])), np.zeros(3))
    assert total_score(sweeps, CAMERA, T_TRUE) > total_score(sweeps, CAMERA, rotated @ T_TRUE)


def test_search_recovers_a_rotated_start():
    sweeps = [make_sweep_edges(make_scene_points())]
    error = make_transform(rotation_from_axis_angle(np.radians([1.5, -2.0, 1.0])), np.zeros(3))
    T_start = error @ T_TRUE

    T_found, history = search_extrinsic(sweeps, CAMERA, T_start)

    remaining = T_found @ invert(T_TRUE)
    assert rotation_angle_deg(T_start[:3, :3] @ T_TRUE[:3, :3].T) > 2.5
    assert rotation_angle_deg(remaining[:3, :3]) < 0.3
    assert np.linalg.norm(remaining[:3, 3]) < 0.03
    assert history[-1][1] > history[0][1]


def test_calibrate_recovers_a_larger_rotation():
    sweeps = [make_sweep_edges(make_scene_points())]
    error = make_transform(rotation_from_axis_angle(np.radians([3.0, -3.0, 2.0])), np.zeros(3))

    T_found, _ = calibrate(sweeps, CAMERA, error @ T_TRUE)

    remaining = T_found @ invert(T_TRUE)
    assert rotation_angle_deg(remaining[:3, :3]) < 0.3


def mean_pixel_difference(scene_points, T_a, T_b):
    uv_a, _, visible_a = CAMERA.project(apply(T_a, scene_points))
    uv_b, _, visible_b = CAMERA.project(apply(T_b, scene_points))
    both = visible_a & visible_b
    return np.mean(np.linalg.norm(uv_a[both] - uv_b[both], axis=1))


def test_calibrate_recovers_rotation_and_translation_to_a_pixel():
    # The score reads whole pixels, and a small rotation and a small shift can move
    # the edges almost the same way. The fair check is therefore how far the scene
    # lands from where the true extrinsic puts it, not the six numbers one by one.
    scene_points = make_scene_points()
    sweeps = [make_sweep_edges(scene_points)]
    error = make_transform(
        rotation_from_axis_angle(np.radians([1.0, 0.5, -1.0])), [0.03, -0.02, 0.04]
    )
    T_start = error @ T_TRUE

    T_found, _ = calibrate(sweeps, CAMERA, T_start)

    assert mean_pixel_difference(scene_points, T_start, T_TRUE) > 2.5
    assert mean_pixel_difference(scene_points, T_found, T_TRUE) < 1.0


def test_rotation_only_keeps_the_start_translation():
    sweeps = [make_sweep_edges(make_scene_points())]
    error = make_transform(rotation_from_axis_angle(np.radians([1.0, -1.0, 0.5])), np.zeros(3))
    T_start = error @ T_TRUE

    T_found, _ = calibrate(sweeps, CAMERA, T_start, rotation_only=True)

    correction = T_found @ invert(T_start)
    np.testing.assert_allclose(correction[:3, 3], 0.0, atol=1e-12)
    assert rotation_angle_deg((T_found @ invert(T_TRUE))[:3, :3]) < 0.3
