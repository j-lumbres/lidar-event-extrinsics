import numpy as np
import pytest

from lidar_event_extrinsics.edges import (
    TimedEventMaps,
    alignment_score,
    depth_discontinuity_weights,
    depth_edge_mask,
    inverse_distance_map,
)
from lidar_event_extrinsics.vector import Events, LidarSweep


def make_sweep(distance_per_column):
    # One beam looking along +x at every column, with the given distance.
    distance = np.array([distance_per_column], dtype=np.float64)
    xyz = np.zeros((1, distance.shape[1], 3))
    xyz[..., 0] = distance
    return LidarSweep(
        stamp_us=0,
        xyz=xyz,
        point_time_us=np.zeros(distance.shape, dtype=np.int64),
        valid=distance > 0,
    )


def test_depth_edges_mark_the_near_side():
    # A pole at 3 m in front of a wall at 10 m.
    sweep = make_sweep([10.0, 10.0, 3.0, 3.0, 10.0, 10.0])
    edges = depth_edge_mask(sweep, min_jump_m=0.5)
    assert edges[0].tolist() == [False, False, True, True, False, False]


def test_no_edge_next_to_an_empty_return():
    sweep = make_sweep([10.0, 0.0, 3.0])
    edges = depth_edge_mask(sweep, min_jump_m=0.5)
    assert not edges.any()


def test_alignment_score_rewards_edges_on_events():
    event_image = np.ones((10, 10))
    event_image[5, :] = 11.0
    on_events = np.array([[2.0, 5.0], [7.0, 5.0]])
    off_events = np.array([[2.0, 1.0], [7.0, 8.0]])
    assert alignment_score(event_image, on_events) > 5.0
    assert alignment_score(event_image, off_events) < 1.0


def test_discontinuity_weights_grow_with_the_jump():
    sweep = make_sweep([10.0, 10.0, 3.0, 3.0, 9.0, 9.0])
    weights = depth_discontinuity_weights(sweep, min_jump_m=0.2, exponent=0.5)
    # The pole points are 7 m and 6 m nearer than the wall next to them.
    np.testing.assert_allclose(weights[0], [0.0, 0.0, np.sqrt(7.0), np.sqrt(6.0), 0.0, 0.0])


def test_inverse_distance_map_decays_with_city_block_distance():
    edge_image = np.zeros((9, 9))
    edge_image[4, 4] = 1.0
    gamma = 0.9
    alpha = 1.0 / 3.0
    spread = inverse_distance_map(edge_image, gamma=gamma, alpha=alpha)

    assert spread[4, 4] == pytest.approx(1.0)
    # Two pixels right and one down is 3 steps away.
    assert spread[5, 6] == pytest.approx((1 - alpha) * gamma**3)


def test_timed_maps_score_points_on_an_event_line_higher():
    # A vertical line of events at column 20, firing throughout 0 to 10 ms.
    rows = np.tile(np.arange(5, 35), 20)
    times = np.repeat(np.arange(0, 10_000, 500), 30)
    events = Events(
        x=np.full(len(rows), 20),
        y=rows,
        t_us=times.astype(np.int64),
        polarity=np.ones(len(rows), dtype=np.int8),
    )
    maps = TimedEventMaps(events, width=40, height=40, kind="idt", gamma=0.9)

    point_times = np.array([4000, 5000, 6000], dtype=np.int64)
    on_line = np.array([[20.0, 10.0], [20.0, 20.0], [20.0, 30.0]])
    near_line = on_line + [3.0, 0.0]
    far_from_line = on_line + [15.0, 0.0]

    score_on = maps.score(on_line, point_times)
    score_near = maps.score(near_line, point_times)
    score_far = maps.score(far_from_line, point_times)
    assert score_on > score_near > score_far
