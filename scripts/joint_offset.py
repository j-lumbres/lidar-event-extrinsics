"""Estimate the LiDAR time offset and the rotation together.

For every candidate offset the rotation is calibrated from scratch, with the
translation held at FARO, and the offset with the highest score wins. A coarse pass
over -20 to +200 ms in 20 ms steps finds the neighborhood, and a fine pass in 4 ms
steps refines it. The fine pass is repeated on the even and odd halves of the sweeps
to show how much the answer moves with the data.
"""

import os
from pathlib import Path

import numpy as np

from lidar_event_extrinsics.calibration import calibrate_with_time_offset, prepare_sweeps
from lidar_event_extrinsics.geometry import invert, rotation_vector_from_matrix
from lidar_event_extrinsics.vector import (
    load_event_camera,
    load_extrinsics,
    load_ground_truth,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
SEQUENCE = os.environ.get("VECTOR_SEQUENCE", "school_scooter1")
SWEEP_STEP = int(os.environ.get("SWEEP_STEP", "10"))
COARSE_OFFSETS_MS = np.arange(-20, 201, 20)
FINE_HALF_RANGE_MS = 16
FINE_STEP_MS = 4

files = sequence_files(DATA_DIR, SEQUENCE)
camera = load_event_camera(files["calib"])
T_faro = load_extrinsics(files["calib"])["T_event_from_lidar_faro"]
ground_truth = load_ground_truth(files["ground_truth"])

# The event margin must reach the largest offset plus the event window.
largest_offset_ms = COARSE_OFFSETS_MS.max() + FINE_HALF_RANGE_MS
sweeps = prepare_sweeps(
    files,
    ground_truth.t_us[::SWEEP_STEP],
    camera,
    event_margin_us=1000 * largest_offset_ms + 20_000,
)
print(f"{SEQUENCE}: {len(sweeps)} sweeps, every {SWEEP_STEP}th")


def rotation_from_faro(T_event_from_lidar):
    delta = T_event_from_lidar @ invert(T_faro)
    return np.degrees(rotation_vector_from_matrix(delta[:3, :3]))


def print_results(title, results):
    print(title)
    for offset_us, T_found, score in results:
        rotation_deg = rotation_from_faro(T_found)
        print(
            f"  {offset_us / 1000:+6.0f} ms: "
            f"[{rotation_deg[0]:+.2f} {rotation_deg[1]:+.2f} {rotation_deg[2]:+.2f}] deg, "
            f"score {score:.3f}"
        )
    best_offset_us, best_T, best_score = max(results, key=lambda result: result[2])
    rotation_deg = rotation_from_faro(best_T)
    print(
        f"  best: {best_offset_us / 1000:+.0f} ms, "
        f"[{rotation_deg[0]:+.2f} {rotation_deg[1]:+.2f} {rotation_deg[2]:+.2f}] deg, "
        f"score {best_score:.3f}"
    )
    return best_offset_us


coarse_results = calibrate_with_time_offset(sweeps, camera, T_faro, 1000 * COARSE_OFFSETS_MS)
coarse_best_us = print_results("coarse, all sweeps", coarse_results)

fine_offsets_us = coarse_best_us + 1000 * np.arange(
    -FINE_HALF_RANGE_MS, FINE_HALF_RANGE_MS + 1, FINE_STEP_MS
)
for name, subset in (
    ("all sweeps", sweeps),
    ("even half", sweeps[0::2]),
    ("odd half", sweeps[1::2]),
):
    fine_results = calibrate_with_time_offset(subset, camera, T_faro, fine_offsets_us)
    print_results(f"fine, {name}", fine_results)
