"""Edges in LiDAR sweeps and event streams, and a score for how well they line up.

An event camera fires where brightness changes, which on a moving sensor is mostly
along object boundaries. A LiDAR sees the same boundaries as jumps in range between
neighboring beams. With a correct extrinsic, projected LiDAR depth edges should fall
on pixels with many events.

The part of a sweep that the camera sees is captured over about 25 ms, and the
sensor moves during that time. Each edge point is therefore compared with events
from around its own capture time, not with one image for the whole sweep.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

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


def depth_discontinuity_weights(sweep, min_jump_m=0.2, exponent=0.5):
    """Weight each point by how much nearer it is than its horizontal neighbors.

    Following Levinson and Thrun (RSS 2013), the weight is the larger of the two
    jumps to the left and right neighbor, raised to a power below 1 so a few huge
    jumps do not dominate. Only points nearer than a neighbor get weight, which keeps
    the outline of the foreground object. Jumps under min_jump_m are treated as noise.
    """
    distance = np.linalg.norm(sweep.xyz, axis=-1)
    distance = np.where(sweep.valid, distance, np.nan)

    # A positive gap means the neighbor is farther away than the point itself.
    gap_to_left = np.full(distance.shape, np.nan)
    gap_to_right = np.full(distance.shape, np.nan)
    gap_to_left[:, 1:] = distance[:, :-1] - distance[:, 1:]
    gap_to_right[:, :-1] = distance[:, 1:] - distance[:, :-1]

    largest_gap = np.fmax(gap_to_left, gap_to_right)
    largest_gap = np.nan_to_num(largest_gap, nan=0.0)
    largest_gap[largest_gap < min_jump_m] = 0.0
    return largest_gap**exponent


def inverse_distance_map(edge_image, gamma=0.9, alpha=1.0 / 3.0):
    """Spread an edge image so pixels near an edge get partial credit.

    Levinson and Thrun define D = alpha * E + (1 - alpha) * max over pixels of
    E * gamma^distance. A projected point a few pixels off an edge still scores,
    which gives a search a slope to follow instead of a flat field. The max is
    found with forward and backward passes along rows and then columns, so the
    distance here is the L1 (city block) distance.
    """
    spread = edge_image.astype(np.float64).copy()

    for column in range(1, spread.shape[1]):
        spread[:, column] = np.maximum(spread[:, column], gamma * spread[:, column - 1])
    for column in range(spread.shape[1] - 2, -1, -1):
        spread[:, column] = np.maximum(spread[:, column], gamma * spread[:, column + 1])
    for row in range(1, spread.shape[0]):
        spread[row] = np.maximum(spread[row], gamma * spread[row - 1])
    for row in range(spread.shape[0] - 2, -1, -1):
        spread[row] = np.maximum(spread[row], gamma * spread[row + 1])

    return alpha * edge_image + (1.0 - alpha) * spread


def event_count_image(events, width, height, sigma_px=1.5):
    """Count events per pixel, then blur so a near miss still scores."""
    image = np.zeros((height, width))
    np.add.at(image, (events.y, events.x), 1.0)
    return gaussian_filter(image, sigma_px)


def time_surface(events, width, height, t_ref_us, tau_us, sigma_px=1.5):
    """exp(-(t_ref - t_last) / tau), where t_last is the latest event at each pixel.

    Pixels that fired just before t_ref are close to 1, and pixels that have been
    quiet for several tau fade to 0. Events after t_ref are ignored.
    """
    before = events.t_us <= t_ref_us
    last_event_us = np.full((height, width), np.iinfo(np.int64).min)
    np.maximum.at(last_event_us, (events.y[before], events.x[before]), events.t_us[before])

    has_event = last_event_us > np.iinfo(np.int64).min
    surface = np.zeros((height, width))
    age_us = t_ref_us - last_event_us[has_event]
    surface[has_event] = np.exp(-age_us / tau_us)
    return gaussian_filter(surface, sigma_px)


def project_edges(sweep, edge_mask, camera, T_event_from_lidar, weights=None):
    """Project the edge points of a sweep into the event camera.

    Returns pixel coordinates (M, 2) and capture times (M,) of the visible edge
    points, and their weights (M,) too when weights (128, 2048) are given.
    """
    edge_points_lidar = sweep.xyz[edge_mask]
    edge_times_us = sweep.point_time_us[edge_mask]

    edge_points_event = apply(T_event_from_lidar, edge_points_lidar)
    uv, _, visible = camera.project(edge_points_event)
    if weights is None:
        return uv[visible], edge_times_us[visible]
    return uv[visible], edge_times_us[visible], weights[edge_mask][visible]


def alignment_score(event_image, uv):
    """Mean event density at the projected edge pixels, relative to the whole image.

    A score of 1 means the edges land on average pixels. Higher is better aligned.
    """
    column = np.round(uv[:, 0]).astype(int)
    row = np.round(uv[:, 1]).astype(int)
    density_at_edges = event_image[row, column].mean()
    return density_at_edges / event_image.mean()


class TimedEventMaps:
    """Event maps for short time bins, built on first use and then reused.

    Edge points are grouped into bins of bin_us by capture time. The map for a bin
    uses the events within half_window_us of the bin center. A search tries many
    extrinsics on the same sweep, and the capture times do not depend on the
    extrinsic, so each bin's map is built only once.

    kind is "count" for a blurred event-count image or "surface" for a time surface
    taken at the end of the window.
    """

    def __init__(
        self,
        events,
        width,
        height,
        bin_us=2000,
        half_window_us=3000,
        sigma_px=1.5,
        kind="count",
        tau_us=3000,
        gamma=0.9,
        count_cap=3.0,
    ):
        self.events = events
        self.width = width
        self.height = height
        self.bin_us = bin_us
        self.half_window_us = half_window_us
        self.sigma_px = sigma_px
        self.kind = kind
        self.tau_us = tau_us
        self.gamma = gamma
        self.count_cap = count_cap
        self.maps = {}

    def map_for_bin(self, bin_index):
        if bin_index in self.maps:
            return self.maps[bin_index]

        center_us = bin_index * self.bin_us + self.bin_us // 2
        start_us = center_us - self.half_window_us
        end_us = center_us + self.half_window_us

        # Events are sorted by time, so the window is one slice.
        first = np.searchsorted(self.events.t_us, start_us, side="left")
        last = np.searchsorted(self.events.t_us, end_us, side="left")
        window = self.events.slice(first, last)

        if self.kind == "count":
            event_map = event_count_image(window, self.width, self.height, self.sigma_px)
        elif self.kind == "surface":
            event_map = time_surface(
                window, self.width, self.height, end_us, self.tau_us, self.sigma_px
            )
        else:
            # Cap the counts, so a flickering light does not outweigh a real edge.
            counts = event_count_image(window, self.width, self.height, sigma_px=0)
            edge_image = np.minimum(counts, self.count_cap) / self.count_cap
            event_map = inverse_distance_map(edge_image, self.gamma)

        # Normalize by the map mean, so busy and quiet moments count equally.
        mean_value = event_map.mean()
        if mean_value > 0:
            event_map = event_map / mean_value

        self.maps[bin_index] = event_map
        return event_map

    def score(self, uv, edge_times_us, weights=None):
        """Weighted mean of the event map at each edge point, read from its own time bin."""
        if weights is None:
            weights = np.ones(len(uv))
        column = np.round(uv[:, 0]).astype(int)
        row = np.round(uv[:, 1]).astype(int)
        bin_of_point = edge_times_us // self.bin_us

        total = 0.0
        for bin_index in np.unique(bin_of_point):
            in_bin = bin_of_point == bin_index
            event_map = self.map_for_bin(int(bin_index))
            values = event_map[row[in_bin], column[in_bin]]
            total += np.sum(weights[in_bin] * values)
        return total / np.sum(weights)
