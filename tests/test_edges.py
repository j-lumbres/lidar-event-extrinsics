import numpy as np

from lidar_event_extrinsics.edges import alignment_score, depth_edge_mask
from lidar_event_extrinsics.vector import LidarSweep


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
