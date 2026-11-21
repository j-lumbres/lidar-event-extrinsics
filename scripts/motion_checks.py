"""Check whether the calibration drifts with speed or turning rate.

The sweeps are split into thirds by linear speed, and again by turning rate, both
from the ground truth. Each third is calibrated on its own with the translation held
at FARO. A rotation that drifts with speed points to motion rather than the rig.
The time offset between the LiDAR and the events is checked in time_offset.py.
"""

import os
from pathlib import Path

import numpy as np

from lidar_event_extrinsics.calibration import calibrate, prepare_sweeps
from lidar_event_extrinsics.geometry import (
    invert,
    rotation_angle_deg,
    rotation_vector_from_matrix,
)
from lidar_event_extrinsics.vector import (
    load_event_camera,
    load_extrinsics,
    load_ground_truth,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
SEQUENCE = os.environ.get("VECTOR_SEQUENCE", "school_scooter1")
SWEEP_STEP = int(os.environ.get("SWEEP_STEP", "10"))
EVENT_MARGIN_US = 10_000

files = sequence_files(DATA_DIR, SEQUENCE)
camera = load_event_camera(files["calib"])
T_faro = load_extrinsics(files["calib"])["T_event_from_lidar_faro"]
ground_truth = load_ground_truth(files["ground_truth"])

sweep_indices = np.arange(0, len(ground_truth) - 1, SWEEP_STEP)
sweeps = prepare_sweeps(
    files, ground_truth.t_us[sweep_indices], camera, event_margin_us=EVENT_MARGIN_US
)
print(f"{SEQUENCE}: {len(sweeps)} sweeps, every {SWEEP_STEP}th")

# Speed and turning rate between each sweep's pose and the next one.
speeds = []
turn_rates = []
for index in sweep_indices:
    dt_s = (ground_truth.t_us[index + 1] - ground_truth.t_us[index]) * 1e-6
    T_now = ground_truth.T_world_from_lidar[index]
    T_next = ground_truth.T_world_from_lidar[index + 1]
    speeds.append(np.linalg.norm(T_next[:3, 3] - T_now[:3, 3]) / dt_s)
    turn_rates.append(rotation_angle_deg(T_now[:3, :3].T @ T_next[:3, :3]) / dt_s)
speeds = np.array(speeds)
turn_rates = np.array(turn_rates)


def rotation_from_faro(T_event_from_lidar):
    delta = T_event_from_lidar @ invert(T_faro)
    return np.degrees(rotation_vector_from_matrix(delta[:3, :3]))


def format_rotation(rotation_deg):
    return f"[{rotation_deg[0]:+.2f} {rotation_deg[1]:+.2f} {rotation_deg[2]:+.2f}]"


# Calibrated rotation on all sweeps, translation held at FARO.
T_peak, history = calibrate(sweeps, camera, T_faro, rotation_only=True)
print(f"rotation from FARO, all sweeps: {format_rotation(rotation_from_faro(T_peak))} deg")

for label, values, unit in (("speed", speeds, "m/s"), ("turn rate", turn_rates, "deg/s")):
    print(f"\ncalibration by {label} third")
    edges = np.percentile(values, [0, 100 / 3, 200 / 3, 100])
    for third in range(3):
        in_third = (values >= edges[third]) & (values <= edges[third + 1])
        subset = [sweep for sweep, keep in zip(sweeps, in_third, strict=True) if keep]
        T_found, _ = calibrate(subset, camera, T_faro, rotation_only=True)
        print(
            f"  {label} {edges[third]:.2f} to {edges[third + 1]:.2f} {unit} "
            f"({len(subset)} sweeps, median {np.median(values[in_third]):.2f}): "
            f"{format_rotation(rotation_from_faro(T_found))} deg"
        )
