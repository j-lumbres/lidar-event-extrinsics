"""Search for the extrinsic that best lines up LiDAR depth edges with events.

The score of one extrinsic is the Levinson-Thrun edge score averaged over many
sweeps. A single sweep is too noisy to calibrate from, so the search always uses
the whole set.

A candidate extrinsic is a small correction applied in the event camera frame,

    T_event_from_lidar = make_transform(R(rotation_vector), translation) @ T_start,

so the six numbers being searched are a rotation vector (radians) and a
translation (meters), both written in event camera axes.
"""

from dataclasses import dataclass

import numpy as np

from lidar_event_extrinsics.edges import TimedEventMaps, depth_discontinuity_weights
from lidar_event_extrinsics.geometry import (
    apply,
    make_transform,
    rotation_from_axis_angle,
)
from lidar_event_extrinsics.vector import load_events, load_lidar_sweep

MIN_EDGE_POINTS = 50


@dataclass
class SweepEdges:
    """The edge points of one sweep and the event maps to score them against."""

    stamp_us: int
    points: np.ndarray  # (M, 3) lidar frame
    times_us: np.ndarray  # (M,)
    weights: np.ndarray  # (M,)
    maps: TimedEventMaps


def prepare_sweeps(files, stamps_us, camera, gamma=0.9, event_margin_us=10_000):
    """Load each sweep, keep only its weighted edge points, and attach its events.

    event_margin_us is how far before and after the revolution events are kept. It
    has to cover the event window and any time offset that will be tried.
    """
    sweeps = []
    for stamp_us in stamps_us:
        sweep = load_lidar_sweep(files["lidar"], stamp_us)
        weights = depth_discontinuity_weights(sweep)
        is_edge = weights > 0

        # Events for the whole revolution plus a margin for the event windows.
        events = load_events(
            files["events"],
            sweep.stamp_us - event_margin_us,
            sweep.stamp_us + 100_000 + event_margin_us,
        )
        maps = TimedEventMaps(events, camera.width, camera.height, kind="idt", gamma=gamma)

        sweeps.append(
            SweepEdges(
                stamp_us=sweep.stamp_us,
                points=sweep.xyz[is_edge],
                times_us=sweep.point_time_us[is_edge],
                weights=weights[is_edge],
                maps=maps,
            )
        )
    return sweeps


def sweep_score(sweep, camera, T_event_from_lidar, time_offset_us=0):
    """Edge score of one sweep, or NaN if too few edge points land in the image.

    time_offset_us is added to the LiDAR point times before they are matched with
    events. A positive offset compares each point with later events.
    """
    points_event = apply(T_event_from_lidar, sweep.points)
    uv, _, visible = camera.project(points_event)
    if np.count_nonzero(visible) < MIN_EDGE_POINTS:
        return np.nan
    shifted_times_us = sweep.times_us[visible] + time_offset_us
    return sweep.maps.score(uv[visible], shifted_times_us, sweep.weights[visible])


def total_score(sweeps, camera, T_event_from_lidar, time_offset_us=0):
    """Mean edge score over all sweeps, so each sweep counts equally."""
    scores = [sweep_score(sweep, camera, T_event_from_lidar, time_offset_us) for sweep in sweeps]
    return np.nanmean(scores)


def corrected_extrinsic(T_start, correction):
    """Apply a correction [rx, ry, rz, tx, ty, tz] in the event camera frame."""
    rotation = rotation_from_axis_angle(correction[:3])
    delta = make_transform(rotation, correction[3:])
    return delta @ T_start


def grid_search_rotation(
    sweeps, camera, T_start, half_range_deg=4.0, step_deg=1.0, time_offset_us=0
):
    """Try every rotation correction on a coarse grid and return the best one.

    The score has small bumps, and a local search started from a bad guess stops on
    the first bump it meets. A grid of 9 x 9 x 9 rotations (with the default range
    and step) finds the right neighborhood before the local search refines it.

    Returns the best correction [rx, ry, rz, 0, 0, 0] and its score.
    """
    angles = np.radians(np.arange(-half_range_deg, half_range_deg + 1e-9, step_deg))

    best_correction = np.zeros(6)
    best_score = -np.inf
    for angle_x in angles:
        for angle_y in angles:
            for angle_z in angles:
                correction = np.array([angle_x, angle_y, angle_z, 0.0, 0.0, 0.0])
                T_candidate = corrected_extrinsic(T_start, correction)
                score = total_score(sweeps, camera, T_candidate, time_offset_us)
                if score > best_score:
                    best_score = score
                    best_correction = correction
    return best_correction, best_score


def search_extrinsic(
    sweeps,
    camera,
    T_start,
    start_correction=None,
    rotation_step_deg=0.5,
    translation_step_m=0.05,
    min_rotation_step_deg=0.1,
    min_translation_step_m=0.005,
    max_evaluations=2000,
    rotation_only=False,
    time_offset_us=0,
):
    """Coordinate search, rotation first, then translation, then all six together.

    At each step every active parameter is nudged up and down by its step size, and
    any nudge that raises the score is kept. When no nudge helps, the steps are
    halved. A stage ends when its steps fall below the minimum. Rotation goes first
    because the score depends far more on rotation than on translation, and a
    translation search from a badly rotated start would chase the wrong edges.

    start_correction is where the search begins, for example the best point of
    grid_search_rotation. Returns the corrected extrinsic and a list of
    (correction, score) after each accepted move.
    """
    if start_correction is None:
        correction = np.zeros(6)
    else:
        correction = np.array(start_correction, dtype=np.float64)
    best_score = total_score(
        sweeps, camera, corrected_extrinsic(T_start, correction), time_offset_us
    )
    history = [(correction.copy(), best_score)]
    evaluations = 1

    start_steps = np.array([np.radians(rotation_step_deg)] * 3 + [translation_step_m] * 3)
    min_steps = np.array([np.radians(min_rotation_step_deg)] * 3 + [min_translation_step_m] * 3)

    stages = [
        ("rotation", [0, 1, 2], 1.0),
        ("translation", [3, 4, 5], 1.0),
        # The joint stage starts small, since both parts are already close.
        ("all", [0, 1, 2, 3, 4, 5], 0.25),
    ]
    if rotation_only:
        # Edges alone barely constrain translation, and a free translation drags
        # the rotation along with it, so the translation can be held at its start.
        stages = [("rotation", [0, 1, 2], 1.0)]

    for _, active, start_fraction in stages:
        steps = start_steps * start_fraction

        while np.all(steps[active] >= min_steps[active]) and evaluations < max_evaluations:
            improved = False
            for parameter in active:
                for sign in (1.0, -1.0):
                    candidate = correction.copy()
                    candidate[parameter] += sign * steps[parameter]
                    T_candidate = corrected_extrinsic(T_start, candidate)
                    score = total_score(sweeps, camera, T_candidate, time_offset_us)
                    evaluations += 1

                    if score > best_score:
                        correction = candidate
                        best_score = score
                        history.append((correction.copy(), best_score))
                        improved = True
                        break

            if not improved:
                steps = steps / 2.0

    return corrected_extrinsic(T_start, correction), history


def calibrate(sweeps, camera, T_start, rotation_only=False, time_offset_us=0):
    """Grid search over rotation, then the local search from the best grid point.

    time_offset_us is added to the LiDAR point times before they are matched with
    events, for data whose LiDAR clock runs behind the event camera.
    """
    grid_correction, _ = grid_search_rotation(
        sweeps, camera, T_start, time_offset_us=time_offset_us
    )
    return search_extrinsic(
        sweeps,
        camera,
        T_start,
        start_correction=grid_correction,
        rotation_only=rotation_only,
        time_offset_us=time_offset_us,
    )
