"""Compare edge maps by how the alignment score changes around the FARO extrinsic.

Each setting is scored at the FARO extrinsic rotated about, or shifted along, each
event camera axis. The scores are averaged over every 10th sweep. A useful setting
peaks near zero and falls off on both sides, which gives a search a slope to follow.

Settings:
    count       blurred event count against depth edges above 0.5 m, all edges equal
    surface     time surface against the same edges
    inverse     inverse distance map of the events against edges weighted by the
                size of the depth jump (Levinson and Thrun, RSS 2013)
"""

import os
from pathlib import Path

import numpy as np

from lidar_event_extrinsics.edges import (
    TimedEventMaps,
    depth_discontinuity_weights,
    depth_edge_mask,
    project_edges,
)
from lidar_event_extrinsics.geometry import make_transform, rotation_from_axis_angle
from lidar_event_extrinsics.vector import (
    load_event_camera,
    load_events,
    load_extrinsics,
    load_ground_truth,
    load_lidar_sweep,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
SWEEP_STEP = 10
MIN_EDGE_POINTS = 50
ANGLES_DEG = np.arange(-8, 8.01, 1.0)
SHIFTS_M = np.arange(-0.30, 0.301, 0.05)

SETTINGS = {
    "count": {"kind": "count", "weighted": False},
    "surface": {"kind": "surface", "weighted": False},
    "inverse, gamma 0.9": {"kind": "idt", "weighted": True, "gamma": 0.9},
    "inverse, gamma 0.98": {"kind": "idt", "weighted": True, "gamma": 0.98},
}

SEQUENCE = os.environ.get("VECTOR_SEQUENCE", "school_scooter1")

files = sequence_files(DATA_DIR, SEQUENCE)
camera = load_event_camera(files["calib"])
extrinsics = load_extrinsics(files["calib"])
ground_truth = load_ground_truth(files["ground_truth"])
T_faro = extrinsics["T_event_from_lidar_faro"]

# Every candidate is a small change applied in the event camera frame: delta @ T_faro.
candidates = []
for axis_index in range(3):
    for angle_deg in ANGLES_DEG:
        rotation_vector = np.zeros(3)
        rotation_vector[axis_index] = np.radians(angle_deg)
        delta = make_transform(rotation_from_axis_angle(rotation_vector), np.zeros(3))
        candidates.append(("rot", axis_index, angle_deg, delta @ T_faro))
    for shift_m in SHIFTS_M:
        translation = np.zeros(3)
        translation[axis_index] = shift_m
        delta = make_transform(np.eye(3), translation)
        candidates.append(("shift", axis_index, shift_m, delta @ T_faro))

scores = {name: [] for name in SETTINGS}

for sweep_index in range(0, len(ground_truth), SWEEP_STEP):
    sweep = load_lidar_sweep(files["lidar"], ground_truth.t_us[sweep_index])
    events = load_events(files["events"], sweep.stamp_us - 10_000, sweep.stamp_us + 110_000)

    for name, setting in SETTINGS.items():
        if setting["weighted"]:
            weights = depth_discontinuity_weights(sweep)
            mask = weights > 0
        else:
            mask = depth_edge_mask(sweep, min_jump_m=0.5)
            weights = mask.astype(np.float64)

        maps = TimedEventMaps(
            events,
            camera.width,
            camera.height,
            kind=setting["kind"],
            gamma=setting.get("gamma", 0.9),
        )

        uv_faro, _ = project_edges(sweep, mask, camera, T_faro)
        if len(uv_faro) < MIN_EDGE_POINTS:
            continue

        sweep_scores = []
        for _, _, _, T_candidate in candidates:
            uv, times, point_weights = project_edges(sweep, mask, camera, T_candidate, weights)
            sweep_scores.append(maps.score(uv, times, point_weights))
        scores[name].append(sweep_scores)

for name in SETTINGS:
    mean_scores = np.nanmean(np.array(scores[name]), axis=0)
    print(f"\n{name}  ({len(scores[name])} sweeps)")
    print(f"{'deg':>10} " + " ".join(f"{angle:6.0f}" for angle in ANGLES_DEG))
    for axis_index, axis_name in enumerate("xyz"):
        row = [
            s
            for (kind, axis, _, _), s in zip(candidates, mean_scores, strict=True)
            if kind == "rot" and axis == axis_index
        ]
        print(f"{'rot ' + axis_name:>10} " + " ".join(f"{value:6.2f}" for value in row))
    print(f"{'cm':>10} " + " ".join(f"{100 * shift:6.0f}" for shift in SHIFTS_M))
    for axis_index, axis_name in enumerate("xyz"):
        row = [
            s
            for (kind, axis, _, _), s in zip(candidates, mean_scores, strict=True)
            if kind == "shift" and axis == axis_index
        ]
        print(f"{'shift ' + axis_name:>10} " + " ".join(f"{value:6.2f}" for value in row))
