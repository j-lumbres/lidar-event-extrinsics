"""Overlay LiDAR depth edges on events, before and after the clock and rotation fix.

The Ouster sweeps across the image column by column, so each image column is seen
by the LiDAR at its own moment. The event image behind the edges is built the same
way the score reads it: an event is kept only if it falls within 3 ms of the time
the LiDAR scanned that column, shifted by the time offset.

Left: FARO extrinsic, LiDAR stamps as published. Right: +92 ms and the rotation from
the joint estimate. The sweep shown is the one, of every 10th, whose score improves
most between the two.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lidar_event_extrinsics.calibration import corrected_extrinsic
from lidar_event_extrinsics.edges import depth_discontinuity_weights, inverse_distance_map
from lidar_event_extrinsics.geometry import apply
from lidar_event_extrinsics.vector import (
    load_event_camera,
    load_events,
    load_extrinsics,
    load_ground_truth,
    load_lidar_sweep,
    sequence_files,
)

DATA_DIR = Path(os.environ.get("VECTOR_DIR", Path(__file__).parents[1] / "data" / "raw" / "vector"))
SEQUENCE = os.environ.get("VECTOR_SEQUENCE", "school_scooter1")
SWEEP_STEP = 10
HALF_WINDOW_US = 3000
OFFSET_US = 92_000
ROTATION_FROM_FARO_DEG = np.array([-0.50, -1.13, -0.38])
OUTPUT = Path(__file__).parents[1] / "figures" / "overlay.png"

files = sequence_files(DATA_DIR, SEQUENCE)
camera = load_event_camera(files["calib"])
T_faro = load_extrinsics(files["calib"])["T_event_from_lidar_faro"]
T_fixed = corrected_extrinsic(T_faro, np.r_[np.radians(ROTATION_FROM_FARO_DEG), 0.0, 0.0, 0.0])
ground_truth = load_ground_truth(files["ground_truth"])


def column_times(sweep, T_event_from_lidar):
    """Time the LiDAR scanned each image column, from all of its visible returns."""
    points_event = apply(T_event_from_lidar, sweep.xyz[sweep.valid])
    uv, _, visible = camera.project(points_event)
    columns = np.round(uv[visible, 0]).astype(int)
    times_us = sweep.point_time_us[sweep.valid][visible].astype(np.float64)

    total = np.bincount(columns, weights=times_us, minlength=camera.width)
    count = np.bincount(columns, minlength=camera.width)
    has_points = count > 0
    every_column = np.arange(camera.width)
    return np.interp(every_column, every_column[has_points], total[has_points] / count[has_points])


def scanned_event_image(sweep, events, T_event_from_lidar, offset_us):
    """Event counts, keeping each event only near the time its column was scanned."""
    scan_time_us = column_times(sweep, T_event_from_lidar) + offset_us
    near_scan = np.abs(events.t_us - scan_time_us[events.x]) <= HALF_WINDOW_US
    image = np.zeros((camera.height, camera.width))
    np.add.at(image, (events.y[near_scan], events.x[near_scan]), 1.0)
    return image


def edge_pixels(sweep, weights, T_event_from_lidar):
    is_edge = weights > 0
    uv, _, visible = camera.project(apply(T_event_from_lidar, sweep.xyz[is_edge]))
    return uv[visible], weights[is_edge][visible]


def edge_score(image, uv, weights):
    """Weighted mean of the inverse distance map at the edges, as in the calibration."""
    capped = np.minimum(image, 3.0) / 3.0
    distance_map = inverse_distance_map(capped)
    distance_map = distance_map / distance_map.mean()
    row = np.round(uv[:, 1]).astype(int)
    column = np.round(uv[:, 0]).astype(int)
    return np.sum(weights * distance_map[row, column]) / np.sum(weights)


def render(stamp_us):
    sweep = load_lidar_sweep(files["lidar"], stamp_us)
    weights = depth_discontinuity_weights(sweep)
    events = load_events(
        files["events"], sweep.stamp_us - 10_000, sweep.stamp_us + 100_000 + OFFSET_US + 10_000
    )
    panels = []
    for T_event_from_lidar, offset_us in ((T_faro, 0), (T_fixed, OFFSET_US)):
        image = scanned_event_image(sweep, events, T_event_from_lidar, offset_us)
        uv, edge_weights = edge_pixels(sweep, weights, T_event_from_lidar)
        panels.append((image, uv, edge_weights, edge_score(image, uv, edge_weights)))
    return panels


# Pick the sweep where the fix helps most, among sweeps with enough edges.
best_stamp_us, best_gain = None, -np.inf
for stamp_us in ground_truth.t_us[::SWEEP_STEP]:
    before, after = render(stamp_us)
    if len(before[1]) < 300:
        continue
    gain = after[3] / before[3]
    print(f"{stamp_us}: {before[3]:.3f} -> {after[3]:.3f}, {len(before[1])} edge points")
    if gain > best_gain:
        best_stamp_us, best_gain = stamp_us, gain

before, after = render(best_stamp_us)
seconds_in = (best_stamp_us - ground_truth.t_us[0]) / 1e6
print(f"shown: sweep at {seconds_in:.1f} s, score {before[3]:.3f} -> {after[3]:.3f}")

ZOOM = (200, 430, 135, 315)  # columns and rows of the close-up

figure, axes = plt.subplots(2, 2, figsize=(13, 11), gridspec_kw={"height_ratios": [1, 0.8]})
titles = (
    "FARO extrinsic, LiDAR stamps as published",
    f"+{OFFSET_US // 1000} ms and corrected rotation",
)
for column, ((image, uv, edge_weights, score), title) in enumerate(
    zip((before, after), titles, strict=True)
):
    for row in range(2):
        axis = axes[row, column]
        axis.imshow(np.minimum(image, 2), cmap="gray_r", vmin=0, vmax=2)
        sizes = (3 + 8 * edge_weights / edge_weights.max()) * (2.5 if row == 1 else 1)
        axis.scatter(uv[:, 0], uv[:, 1], s=sizes, c="tab:red", linewidths=0)
        axis.set_xticks([])
        axis.set_yticks([])
        if row == 0:
            axis.set_xlim(0, camera.width - 1)
            axis.set_ylim(camera.height - 1, 0)
            axis.set_title(f"{title}\nedge score {score:.2f}")
            left, right, top, bottom = ZOOM
            axis.add_patch(
                plt.Rectangle((left, top), right - left, bottom - top, fill=False, ec="tab:blue")
            )
        else:
            axis.set_xlim(ZOOM[0], ZOOM[1])
            axis.set_ylim(ZOOM[3], ZOOM[2])
            axis.set_title("close-up (blue box)")
figure.suptitle(
    f"VECtor {SEQUENCE}, sweep at {seconds_in:.1f} s. LiDAR depth edges (red) on events "
    "from the moment each column was scanned (gray)"
)
figure.tight_layout()
OUTPUT.parent.mkdir(exist_ok=True)
figure.savefig(OUTPUT, dpi=150)
print(f"saved {OUTPUT}")
