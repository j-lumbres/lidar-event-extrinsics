"""Check which LiDAR frame the VECtor extrinsics expect.

The bag stores points in the Ouster os_sensor frame. The other Ouster frame,
os_lidar, is turned 180 degrees about z. If the extrinsics were meant for os_lidar,
the published points would need that extra turn before projection. Each sweep is
scored both ways, with both extrinsics.
"""

import os
from pathlib import Path

import numpy as np

from lidar_event_extrinsics.edges import (
    alignment_score,
    depth_edge_mask,
    event_count_image,
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
SWEEP_STEP = 10  # score every 10th sweep, about one per second
EVENT_MARGIN_US = 5_000
MIN_EDGE_POINTS = 50

SEQUENCE = os.environ.get("VECTOR_SEQUENCE", "school_scooter1")

files = sequence_files(DATA_DIR, SEQUENCE)
camera = load_event_camera(files["calib"])
extrinsics = load_extrinsics(files["calib"])
ground_truth = load_ground_truth(files["ground_truth"])

T_sensor_from_lidar_frame = make_transform(rotation_from_axis_angle([0.0, 0.0, np.pi]), np.zeros(3))
candidates = {
    "faro": extrinsics["T_event_from_lidar_faro"],
    "faro_turned": extrinsics["T_event_from_lidar_faro"] @ T_sensor_from_lidar_frame,
    "cad": extrinsics["T_event_from_lidar_cad"],
    "cad_turned": extrinsics["T_event_from_lidar_cad"] @ T_sensor_from_lidar_frame,
}

scores = {name: [] for name in candidates}
sweep_indices = range(0, len(ground_truth), SWEEP_STEP)

for sweep_index in sweep_indices:
    sweep = load_lidar_sweep(files["lidar"], ground_truth.t_us[sweep_index])
    edges = depth_edge_mask(sweep)

    for name, T_event_from_lidar in candidates.items():
        uv, edge_times_us = project_edges(sweep, edges, camera, T_event_from_lidar)
        if len(uv) < MIN_EDGE_POINTS:
            scores[name].append(np.nan)
            continue

        # Use the events from the moments the visible edge points were captured.
        events = load_events(
            files["events"],
            edge_times_us.min() - EVENT_MARGIN_US,
            edge_times_us.max() + EVENT_MARGIN_US,
        )
        event_image = event_count_image(events, camera.width, camera.height)
        scores[name].append(alignment_score(event_image, uv))

print(f"{'sweep':>6}" + "".join(f"{name:>13}" for name in candidates))
for row, sweep_index in enumerate(sweep_indices):
    values = "".join(f"{scores[name][row]:13.2f}" for name in candidates)
    print(f"{sweep_index:>6}{values}")

print()
for name in candidates:
    print(f"{name:>12}: median {np.nanmedian(scores[name]):.2f}")
for extrinsic in ("faro", "cad"):
    published = np.array(scores[extrinsic])
    turned = np.array(scores[f"{extrinsic}_turned"])
    both = np.isfinite(published) & np.isfinite(turned)
    wins = np.count_nonzero(published[both] > turned[both])
    print(
        f"{extrinsic}: published frame scores higher on {wins} of {np.count_nonzero(both)} sweeps"
    )
