"""Measure the time offset between the LiDAR and the event camera and IMU.

Three checks on one sequence:

1. Gyro lag. The ground-truth poses carry the LiDAR timestamps. The turning rate
   between consecutive poses is compared with the IMU gyro averaged over the same
   100 ms, shifted by a range of lags. This uses neither the extrinsic nor events.
2. Edge score against offset. LiDAR point times are shifted by -100 to +250 ms
   before they are matched with events. Event maps are rebuilt for every offset
   and thrown away, so memory stays small.
3. Calibration at 0 ms and at OFFSET_MS, on all sweeps and on the even and odd
   halves, with the translation held at FARO.
"""

import os
from pathlib import Path

import numpy as np

from lidar_event_extrinsics.calibration import calibrate, prepare_sweeps
from lidar_event_extrinsics.edges import TimedEventMaps, depth_discontinuity_weights
from lidar_event_extrinsics.geometry import apply, invert, rotation_vector_from_matrix
from lidar_event_extrinsics.vector import (
    load_event_camera,
    load_events,
    load_extrinsics,
    load_ground_truth,
    load_imu,
    load_lidar_sweep,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
SEQUENCE = os.environ.get("VECTOR_SEQUENCE", "school_scooter1")
SCAN_SWEEP_STEP = 20
CALIBRATION_SWEEP_STEP = int(os.environ.get("SWEEP_STEP", "10"))
LAGS_MS = np.arange(-300, 301, 5)
OFFSETS_MS = np.arange(-100, 251, 10)
OFFSET_MS = 90
MIN_EDGE_POINTS = 50

files = sequence_files(DATA_DIR, SEQUENCE)
camera = load_event_camera(files["calib"])
extrinsics = load_extrinsics(files["calib"])
T_faro = extrinsics["T_event_from_lidar_faro"]
ground_truth = load_ground_truth(files["ground_truth"])
imu = load_imu(files["imu"])

print(f"{SEQUENCE}")

# 1. Gyro lag

# Angular velocity between consecutive poses, written in the LiDAR frame.
rotations = ground_truth.T_world_from_lidar[:, :3, :3]
pose_rates = []
for index in range(len(ground_truth) - 1):
    dt_s = (ground_truth.t_us[index + 1] - ground_truth.t_us[index]) * 1e-6
    step = rotations[index].T @ rotations[index + 1]
    pose_rates.append(rotation_vector_from_matrix(step) / dt_s)
pose_rates = np.array(pose_rates)

gyro_in_lidar = imu.gyro @ extrinsics["T_lidar_from_imu"][:3, :3].T

lag_errors = []
for lag_ms in LAGS_MS:
    # Average the gyro over each pose interval, shifted by the lag, so both signals
    # are smoothed the same way.
    gyro_averages = np.full(pose_rates.shape, np.nan)
    for index in range(len(pose_rates)):
        start_us = ground_truth.t_us[index] + 1000 * lag_ms
        end_us = ground_truth.t_us[index + 1] + 1000 * lag_ms
        in_interval = (imu.t_us >= start_us) & (imu.t_us < end_us)
        if np.any(in_interval):
            gyro_averages[index] = gyro_in_lidar[in_interval].mean(axis=0)

    usable = np.all(np.isfinite(gyro_averages), axis=1)
    difference = gyro_averages[usable] - pose_rates[usable]
    lag_errors.append(np.degrees(np.sqrt(np.mean(difference**2))))
lag_errors = np.array(lag_errors)

best_lag = LAGS_MS[np.argmin(lag_errors)]
error_at_zero = lag_errors[LAGS_MS == 0][0]
print(
    f"gyro lag: best {best_lag:+d} ms ({lag_errors.min():.2f} deg/s rms), "
    f"0 ms gives {error_at_zero:.2f} deg/s"
)

# 2. Edge score against offset

scan_sweeps = []
for stamp_us in ground_truth.t_us[::SCAN_SWEEP_STEP]:
    sweep = load_lidar_sweep(files["lidar"], stamp_us)
    weights = depth_discontinuity_weights(sweep)
    is_edge = weights > 0
    scan_sweeps.append((sweep.xyz[is_edge], sweep.point_time_us[is_edge], weights[is_edge]))

offset_scores = []
for offset_ms in OFFSETS_MS:
    sweep_scores = []
    for points, times_us, weights in scan_sweeps:
        uv, _, visible = camera.project(apply(T_faro, points))
        if np.count_nonzero(visible) < MIN_EDGE_POINTS:
            continue
        shifted_times_us = times_us[visible] + 1000 * offset_ms
        events = load_events(
            files["events"], shifted_times_us.min() - 10_000, shifted_times_us.max() + 10_000
        )
        maps = TimedEventMaps(events, camera.width, camera.height, kind="idt")
        sweep_scores.append(maps.score(uv[visible], shifted_times_us, weights[visible]))
    offset_scores.append(np.mean(sweep_scores))

print(f"edge score at FARO, every {SCAN_SWEEP_STEP}th sweep:")
print(
    "  "
    + " ".join(
        f"{offset:+d}:{score:.3f}" for offset, score in zip(OFFSETS_MS, offset_scores, strict=True)
    )
)
print(f"  best offset {OFFSETS_MS[int(np.argmax(offset_scores))]:+d} ms")

# 3. Calibration at 0 ms and at OFFSET_MS

sweeps = prepare_sweeps(
    files,
    ground_truth.t_us[::CALIBRATION_SWEEP_STEP],
    camera,
    event_margin_us=1000 * OFFSET_MS + 20_000,
)
print(f"calibration, every {CALIBRATION_SWEEP_STEP}th sweep, rotation from FARO (deg):")
for offset_ms in (0, OFFSET_MS):
    for name, subset in (("all", sweeps), ("even", sweeps[0::2]), ("odd", sweeps[1::2])):
        T_found, history = calibrate(
            subset, camera, T_faro, rotation_only=True, time_offset_us=1000 * offset_ms
        )
        delta = T_found @ invert(T_faro)
        rotation_deg = np.degrees(rotation_vector_from_matrix(delta[:3, :3]))
        print(
            f"  offset {offset_ms:+d} ms, {name:<4}: "
            f"[{rotation_deg[0]:+.2f} {rotation_deg[1]:+.2f} {rotation_deg[2]:+.2f}], "
            f"score {history[-1][1]:.3f}"
        )
