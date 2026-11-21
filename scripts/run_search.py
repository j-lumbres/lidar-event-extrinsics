"""Run the calibration from the FARO and CAD extrinsics, and on two halves of the data.

The calibration is a coarse rotation grid followed by a local search.

Every result is reported as a correction to the FARO extrinsic, in event camera
axes: a rotation vector in degrees and a translation in centimeters. The two halves
use alternating sweeps, so the gap between their results shows how much the answer
moves with the data alone.
"""

import os
import time
from pathlib import Path

import numpy as np

from lidar_event_extrinsics.calibration import calibrate, prepare_sweeps, total_score
from lidar_event_extrinsics.geometry import invert, rotation_vector_from_matrix
from lidar_event_extrinsics.vector import (
    load_event_camera,
    load_extrinsics,
    load_ground_truth,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
SWEEP_STEP = 10

SEQUENCE = os.environ.get("VECTOR_SEQUENCE", "school_scooter1")

files = sequence_files(DATA_DIR, SEQUENCE)
camera = load_event_camera(files["calib"])
extrinsics = load_extrinsics(files["calib"])
ground_truth = load_ground_truth(files["ground_truth"])
T_faro = extrinsics["T_event_from_lidar_faro"]
T_cad = extrinsics["T_event_from_lidar_cad"]


def describe(T_event_from_lidar):
    """Correction from FARO to T, as a rotation vector in degrees and a translation in cm."""
    delta = T_event_from_lidar @ invert(T_faro)
    rotation_deg = np.degrees(rotation_vector_from_matrix(delta[:3, :3]))
    translation_cm = 100 * delta[:3, 3]
    return (
        f"rot [{rotation_deg[0]:+.2f} {rotation_deg[1]:+.2f} {rotation_deg[2]:+.2f}] deg "
        f"({np.linalg.norm(rotation_deg):.2f}), "
        f"shift [{translation_cm[0]:+.1f} {translation_cm[1]:+.1f} {translation_cm[2]:+.1f}] cm "
        f"({np.linalg.norm(translation_cm):.1f})"
    )


start_time = time.time()
stamps_us = ground_truth.t_us[::SWEEP_STEP]
sweeps = prepare_sweeps(files, stamps_us, camera)
print(f"prepared {len(sweeps)} sweeps in {time.time() - start_time:.0f} s")

runs = {
    "all sweeps, from FARO": (sweeps, T_faro),
    "all sweeps, from CAD": (sweeps, T_cad),
    "even half, from FARO": (sweeps[0::2], T_faro),
    "odd half, from FARO": (sweeps[1::2], T_faro),
}

print(f"CAD relative to FARO: {describe(T_cad)}")
for name, (run_sweeps, T_start) in runs.items():
    start_time = time.time()
    T_result, history = calibrate(run_sweeps, camera, T_start)
    start_score = total_score(run_sweeps, camera, T_start)
    end_score = history[-1][1]
    print(
        f"\n{name}: score {start_score:.3f} -> {end_score:.3f}, "
        f"{len(history) - 1} moves, {time.time() - start_time:.0f} s"
    )
    print(f"  result relative to FARO: {describe(T_result)}")
    print(f"  FARO score on these sweeps: {total_score(run_sweeps, camera, T_faro):.3f}")
