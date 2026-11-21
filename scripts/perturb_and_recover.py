"""Perturb the FARO extrinsic by known amounts and see where the calibration lands.

The reference is the calibration started from FARO itself. Each trial starts from
FARO moved by a known rotation, translation, or both, about a random axis or along a
random direction, and runs the same calibration. A method that recovers the
extrinsic lands on the reference no matter where it starts. The distance from the
reference measures how repeatable the method is and how far off a start it can
handle. The distance from FARO mixes that with any disagreement between the score
and FARO, which this test cannot split.
"""

import os
import time
from pathlib import Path

import numpy as np

from lidar_event_extrinsics.calibration import calibrate, prepare_sweeps
from lidar_event_extrinsics.geometry import (
    invert,
    make_transform,
    rotation_from_axis_angle,
    rotation_vector_from_matrix,
)
from lidar_event_extrinsics.vector import (
    load_event_camera,
    load_extrinsics,
    load_ground_truth,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
SWEEP_STEP = 10
TRIALS_PER_SIZE = 5
ROTATION_SIZES_DEG = [0.5, 1.0, 2.0, 3.0]
TRANSLATION_SIZES_M = [0.02, 0.05, 0.10]
COMBINED_SIZES = [(2.0, 0.05)]  # (degrees, meters)
SEED = 0
# Set ROTATION_ONLY=1 to hold the translation at its start and run only rotation trials.
ROTATION_ONLY = os.environ.get("ROTATION_ONLY", "0") == "1"
if ROTATION_ONLY:
    TRANSLATION_SIZES_M = []
    COMBINED_SIZES = []

files = sequence_files(DATA_DIR)
camera = load_event_camera(files["calib"])
T_faro = load_extrinsics(files["calib"])["T_event_from_lidar_faro"]
ground_truth = load_ground_truth(files["ground_truth"])
sweeps = prepare_sweeps(files, ground_truth.t_us[::SWEEP_STEP], camera)


def correction_between(T_from, T_to):
    """Rotation vector (deg) and translation (cm) of the correction taking T_from to T_to."""
    delta = T_to @ invert(T_from)
    rotation_deg = np.degrees(rotation_vector_from_matrix(delta[:3, :3]))
    translation_cm = 100 * delta[:3, 3]
    return rotation_deg, translation_cm


def random_unit_vector(generator):
    vector = generator.normal(size=3)
    return vector / np.linalg.norm(vector)


start_time = time.time()
T_reference, reference_history = calibrate(sweeps, camera, T_faro, ROTATION_ONLY)
reference_rotation_deg, reference_translation_cm = correction_between(T_faro, T_reference)
print(f"reference: score {reference_history[-1][1]:.3f}, {time.time() - start_time:.0f} s")
print(
    f"  relative to FARO: rot {np.round(reference_rotation_deg, 2)} deg, "
    f"shift {np.round(reference_translation_cm, 1)} cm"
)

# Every trial is a (group, rotation in degrees, translation in meters) start offset.
generator = np.random.default_rng(SEED)
trials = []
for size_deg in ROTATION_SIZES_DEG:
    for _ in range(TRIALS_PER_SIZE):
        trials.append((f"rot {size_deg}", size_deg * random_unit_vector(generator), np.zeros(3)))
for size_m in TRANSLATION_SIZES_M:
    for _ in range(TRIALS_PER_SIZE):
        trials.append(
            (f"shift {size_m * 100:.0f} cm", np.zeros(3), size_m * random_unit_vector(generator))
        )
for size_deg, size_m in COMBINED_SIZES:
    for _ in range(TRIALS_PER_SIZE):
        rotation = size_deg * random_unit_vector(generator)
        translation = size_m * random_unit_vector(generator)
        trials.append((f"rot {size_deg} + shift {size_m * 100:.0f} cm", rotation, translation))

print(
    f"\n{'group':<22}{'start rot':>10}{'start cm':>9}"
    f"{'ref err deg':>13}{'ref err cm':>11}   remaining rotation vs reference (deg)"
)
results = {}
final_scores = []
for group, rotation_deg, translation_m in trials:
    perturbation = make_transform(rotation_from_axis_angle(np.radians(rotation_deg)), translation_m)
    T_start = perturbation @ T_faro

    T_found, found_history = calibrate(sweeps, camera, T_start, ROTATION_ONLY)

    remaining_rotation_deg, remaining_translation_cm = correction_between(T_reference, T_found)
    rotation_error_deg = np.linalg.norm(remaining_rotation_deg)
    translation_error_cm = np.linalg.norm(remaining_translation_cm)
    rotation_from_faro_deg, _ = correction_between(T_faro, T_found)
    results.setdefault(group, []).append(
        (rotation_error_deg, translation_error_cm, remaining_rotation_deg, rotation_from_faro_deg)
    )
    final_scores.append(found_history[-1][1])
    print(
        f"{group:<22}{np.linalg.norm(rotation_deg):10.2f}{100 * np.linalg.norm(translation_m):9.1f}"
        f"{rotation_error_deg:13.2f}{translation_error_cm:11.1f}   "
        f"[{remaining_rotation_deg[0]:+.2f} {remaining_rotation_deg[1]:+.2f} "
        f"{remaining_rotation_deg[2]:+.2f}]"
    )

print(
    f"\n{'group':<22}{'median deg':>11}{'within 0.5':>11}{'within 1':>9}{'median cm':>10}"
    f"   median |error| per axis x, y, z (deg)"
)
for group, rows in results.items():
    rotation_errors = np.array([row[0] for row in rows])
    translation_errors = np.array([row[1] for row in rows])
    per_axis = np.median(np.abs(np.array([row[2] for row in rows])), axis=0)
    print(
        f"{group:<22}{np.median(rotation_errors):11.2f}{np.mean(rotation_errors <= 0.5):11.2f}"
        f"{np.mean(rotation_errors <= 1.0):9.2f}{np.median(translation_errors):10.1f}"
        f"   {per_axis[0]:.2f}, {per_axis[1]:.2f}, {per_axis[2]:.2f}"
    )
# Where the trials land, relative to FARO, over all groups together.
all_rotations_from_faro = np.array([row[3] for rows in results.values() for row in rows])
print("\nfinal rotation relative to FARO over all trials (deg)")
for axis_index, axis_name in enumerate("xyz"):
    values = all_rotations_from_faro[:, axis_index]
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    print(
        f"  {axis_name}: median {q50:+.2f}, middle half {q25:+.2f} to {q75:+.2f}, "
        f"range {values.min():+.2f} to {values.max():+.2f}"
    )
print(
    f"final scores: median {np.median(final_scores):.3f}, "
    f"range {np.min(final_scores):.3f} to {np.max(final_scores):.3f}, "
    f"reference {reference_history[-1][1]:.3f}"
)
print(f"\ntotal time {time.time() - start_time:.0f} s")
