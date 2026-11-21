"""Edges in LiDAR sweeps and event streams, and a score for how well they line up.

An event camera fires where brightness changes, which on a moving sensor is mostly
along object boundaries. A LiDAR sees the same boundaries as jumps in range between
neighboring beams. With a correct extrinsic, projected LiDAR depth edges should fall
on pixels with many events.
"""

import numpy as np

from lidar_event_extrinsics.geometry import apply


def depth_edge_mask(sweep, min_jump_m=0.5):
    """Mark the near side of every horizontal range jump larger than min_jump_m.

    Only the near side is kept, because the far side is the background that the
    foreground object partly hides, and the event camera sees the object outline.
    """
    distance = np.linalg.norm(sweep.xyz, axis=-1)
    step = np.diff(distance, axis=1)  # distance[:, j + 1] - distance[:, j]

    both_valid = sweep.valid[:, :-1] & sweep.valid[:, 1:]
    left_is_near = both_valid & (step > min_jump_m)
    right_is_near = both_valid & (step < -min_jump_m)

    edges = np.zeros_like(sweep.valid)
    edges[:, :-1] |= left_is_near
    edges[:, 1:] |= right_is_near
    return edges


def gaussian_blur(image, sigma_px):
    """Blur with a separable Gaussian kernel, rows first and then columns."""
    radius = int(np.ceil(3 * sigma_px))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma_px**2))
    kernel = kernel / kernel.sum()

    blurred = np.apply_along_axis(np.convolve, 1, image, kernel, mode="same")
    blurred = np.apply_along_axis(np.convolve, 0, blurred, kernel, mode="same")
    return blurred


def event_count_image(events, width, height, sigma_px=1.5):
    """Count events per pixel, then blur so a near miss still scores."""
    image = np.zeros((height, width))
    np.add.at(image, (events.y, events.x), 1.0)
    return gaussian_blur(image, sigma_px)


def project_edges(sweep, edge_mask, camera, T_event_from_lidar):
    """Project the edge points of a sweep into the event camera.

    Returns pixel coordinates (M, 2) and capture times (M,) of the visible edge points.
    """
    edge_points_lidar = sweep.xyz[edge_mask]
    edge_times_us = sweep.point_time_us[edge_mask]

    edge_points_event = apply(T_event_from_lidar, edge_points_lidar)
    uv, _, visible = camera.project(edge_points_event)
    return uv[visible], edge_times_us[visible]


def alignment_score(event_image, uv):
    """Mean event density at the projected edge pixels, relative to the whole image.

    A score of 1 means the edges land on average pixels. Higher is better aligned.
    """
    column = np.round(uv[:, 0]).astype(int)
    row = np.round(uv[:, 1]).astype(int)
    density_at_edges = event_image[row, column].mean()
    return density_at_edges / event_image.mean()
