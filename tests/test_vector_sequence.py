"""Checks against the real VECtor sequence. Skipped when the files are not downloaded.

Set VECTOR_DIR to the folder with the files, default data/raw/vector in the repository.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from lidar_event_extrinsics.edges import (
    alignment_score,
    depth_edge_mask,
    event_count_image,
    project_edges,
)
from lidar_event_extrinsics.geometry import (
    check_rigid,
    make_transform,
    rotation_from_axis_angle,
    transform_difference,
)
from lidar_event_extrinsics.vector import (
    event_time_range_us,
    load_event_camera,
    load_events,
    load_extrinsics,
    load_ground_truth,
    load_imu,
    load_lidar_sweep,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
FILES = sequence_files(DATA_DIR)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in FILES.values()),
    reason=f"VECtor sequence not found in {DATA_DIR}",
)


def test_extrinsic_chain_is_rigid():
    extrinsics = load_extrinsics(FILES["calib"])
    for name, T in extrinsics.items():
        check_rigid(T, name)


def test_cad_and_faro_extrinsics_differ_by_under_a_degree_and_a_few_cm():
    extrinsics = load_extrinsics(FILES["calib"])
    angle_deg, distance_m = transform_difference(
        extrinsics["T_event_from_lidar_faro"], extrinsics["T_event_from_lidar_cad"]
    )
    assert angle_deg == pytest.approx(0.78, abs=0.01)
    assert distance_m == pytest.approx(0.0597, abs=0.001)


def test_lidar_x_axis_looks_where_the_event_camera_looks():
    # The camera looks along its z axis. The Ouster x axis should point the same way.
    T_event_from_lidar = load_extrinsics(FILES["calib"])["T_event_from_lidar_faro"]
    lidar_x_in_event = T_event_from_lidar[:3, :3] @ [1.0, 0.0, 0.0]
    assert lidar_x_in_event[2] > 0.99


def test_sweeps_match_ground_truth_times():
    ground_truth = load_ground_truth(FILES["ground_truth"])
    assert len(ground_truth) == 443

    sweep = load_lidar_sweep(FILES["lidar"], ground_truth.t_us[100])
    assert sweep.xyz.shape == (128, 2048, 3)
    assert abs(sweep.stamp_us - ground_truth.t_us[100]) <= 1

    # One revolution takes 100 ms, and points are stamped from the first column on.
    time_in_sweep_us = sweep.point_time_us - sweep.stamp_us
    assert time_in_sweep_us.min() == 0
    assert 99_000 < time_in_sweep_us.max() < 100_000
    assert sweep.valid.mean() > 0.5


def test_ground_truth_moves_along_lidar_x():
    # If the rows really are T_world_from_lidar, the velocity seen in the LiDAR frame
    # points forward along +x while the scooter drives.
    ground_truth = load_ground_truth(FILES["ground_truth"])
    positions = ground_truth.positions()
    for index in (50, 150, 250, 350):
        dt_s = (ground_truth.t_us[index + 1] - ground_truth.t_us[index]) * 1e-6
        velocity_world = (positions[index + 1] - positions[index]) / dt_s
        R_world_from_lidar = ground_truth.T_world_from_lidar[index, :3, :3]
        velocity_lidar = R_world_from_lidar.T @ velocity_world
        assert velocity_lidar[0] > 0.9 * np.linalg.norm(velocity_lidar)


def test_all_streams_share_one_clock():
    ground_truth = load_ground_truth(FILES["ground_truth"])
    imu = load_imu(FILES["imu"])
    first_event_us, last_event_us = event_time_range_us(FILES["events"])

    # All streams start within 10 ms of each other and run for about 44 s.
    starts = [ground_truth.t_us[0], imu.t_us[0], first_event_us]
    assert max(starts) - min(starts) < 10_000
    assert 44.0 < (last_event_us - first_event_us) * 1e-6 < 44.5
    assert np.all(np.diff(imu.t_us) > 0)


def test_events_are_sorted_in_bounds_with_signed_polarity():
    camera = load_event_camera(FILES["calib"])
    first_event_us, _ = event_time_range_us(FILES["events"])
    events = load_events(FILES["events"], first_event_us + 10_000_000, first_event_us + 10_010_000)

    assert len(events) > 1000
    assert np.all(np.diff(events.t_us) >= 0)
    assert events.x.max() < camera.width and events.y.max() < camera.height
    assert set(np.unique(events.polarity)) == {-1, 1}


def test_extrinsic_fits_points_in_the_published_frame():
    # The bag stores points in the Ouster os_sensor frame. The other Ouster frame,
    # os_lidar, is turned 180 degrees about z. Projected depth edges should land on
    # events more often without that turn, if the extrinsic is meant for os_sensor.
    camera = load_event_camera(FILES["calib"])
    T_published = load_extrinsics(FILES["calib"])["T_event_from_lidar_faro"]
    turn_about_z = make_transform(rotation_from_axis_angle([0.0, 0.0, np.pi]), np.zeros(3))
    T_turned = T_published @ turn_about_z
    ground_truth = load_ground_truth(FILES["ground_truth"])

    published_scores = []
    turned_scores = []
    for sweep_index in range(0, len(ground_truth), 20):
        sweep = load_lidar_sweep(FILES["lidar"], ground_truth.t_us[sweep_index])
        edges = depth_edge_mask(sweep)

        for T_event_from_lidar, scores in (
            (T_published, published_scores),
            (T_turned, turned_scores),
        ):
            uv, edge_times_us = project_edges(sweep, edges, camera, T_event_from_lidar)
            if len(uv) < 50:
                scores.append(np.nan)
                continue
            events = load_events(
                FILES["events"], edge_times_us.min() - 5000, edge_times_us.max() + 5000
            )
            event_image = event_count_image(events, camera.width, camera.height)
            scores.append(alignment_score(event_image, uv))

    published_scores = np.array(published_scores)
    turned_scores = np.array(turned_scores)
    both = np.isfinite(published_scores) & np.isfinite(turned_scores)
    wins = np.count_nonzero(published_scores[both] > turned_scores[both])

    assert wins > 0.6 * np.count_nonzero(both)
    assert np.median(published_scores[both]) > np.median(turned_scores[both]) + 0.3
